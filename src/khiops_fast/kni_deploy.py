######################################################################################
# Copyright (c) 2023-2025 Orange. All rights reserved.                               #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""In-memory deployment of Khiops models with khiops-kni (Khiops Native Interface)

KNI is not reentrant: all the deployments of a process are serialized by a global lock.
Separate processes each load their own copy of the library and do not block each other:
``deploy_model_kni_parallel`` splits the main table in chunks recoded by a pool of processes.
"""
import atexit
import copy
import io
import multiprocessing
import os
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import numpy as np

from khiops_fast.dataset import SparseTable, read_internal_data_table, write_internal_data_table

DEPLOYMENT_MODES = ("khiops", "kni", "kni_parallel")
# Below this number of rows per process the pool overhead exceeds the gain
MIN_ROWS_PER_CHUNK = 5000

_kni = None
_kni_lock = threading.Lock()
_INITIAL_OUTPUT_LENGTH = 64 * 1024

_pool = None
_pool_size = 0
_pool_lock = threading.Lock()


def _get_kni():
    """Returns the process-wide KNI instance (loaded on first use, under the lock)"""
    global _kni  # pylint: disable=global-statement
    if _kni is None:
        try:
            from kni import KNI  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise ImportError(
                "deployment='kni' requires the khiops-kni package: pip install khiops-kni"
            ) from exc
        _kni = KNI()
    return _kni


def check_deployment(deployment):
    if deployment not in DEPLOYMENT_MODES:
        raise ValueError(f"'deployment' must be one of {DEPLOYMENT_MODES}, got {deployment!r}")


def check_n_jobs(n_jobs):
    if n_jobs is not None and (isinstance(n_jobs, bool) or not isinstance(n_jobs, (int, np.integer)) or n_jobs == 0):
        raise ValueError(f"'n_jobs' must be None or a non-zero integer, got {n_jobs!r}")


def resolve_n_jobs(n_jobs):
    """Number of processes for n_jobs (scikit-learn convention), bounded by the usable cores"""
    check_n_jobs(n_jobs)
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    if n_jobs is None:
        return 1
    if n_jobs < 0:
        return max(1, available + 1 + n_jobs)
    return min(n_jobs, available)


def _table_text(table, sort):
    """Header and records of a table serialized exactly as in the Khiops table files"""
    if isinstance(table, SparseTable):
        text, frame = table.create_khiops_text(), None
    else:
        frame = table.create_khiops_dataframe(sort=sort)
        with io.StringIO() as stream:
            write_internal_data_table(frame, stream)
            text = stream.getvalue()
    lines = [line.rstrip("\r") for line in text.split("\n")]
    if lines and lines[-1] == "":
        lines.pop()
    return lines[0], lines[1:], frame


def _key_tuples(frame, columns):
    return list(zip(*(frame[column].astype("str") for column in columns)))


def _output_columns(dictionary):
    columns = []
    for variable in dictionary.variables:
        if not variable.used or variable.type in ("Table", "Entity"):
            continue
        if variable.variable_block is not None:
            raise NotImplementedError(
                "deployment='kni' does not support variable blocks in the output; "
                "use deployment='khiops'"
            )
        columns.append(variable.name)
    return columns


def deploy_model_kni(
    deployment_ds,
    model_dictionary_domain,
    model_dictionary_name,
    computation_dir,
    column_dtypes,
    sort=True,
):
    """Deploys a model on a dataset in memory with KNI and returns the output dataframe

    The rows of the output are in the order of the main table of ``deployment_ds``.
    """
    dictionary_file_path = os.path.join(computation_dir, "kni_deployment.kdic")
    model_dictionary_domain.export_khiops_dictionary_file(dictionary_file_path)
    output_records = _recode_records(
        deployment_ds, model_dictionary_domain, model_dictionary_name, dictionary_file_path,
        os.path.join(computation_dir, "khiops_kni.log"), sort
    )
    return _read_output(model_dictionary_domain, model_dictionary_name, output_records, column_dtypes)


def _read_output(model_dictionary_domain, model_dictionary_name, output_records, column_dtypes):
    output_columns = _output_columns(model_dictionary_domain.get_dictionary(model_dictionary_name))
    text = "\t".join(output_columns) + "\n" + "".join(record + "\n" for record in output_records)
    with io.StringIO(text) as stream:
        return read_internal_data_table(stream, column_dtypes=column_dtypes)


def _recode_records(
    deployment_ds, model_dictionary_domain, model_dictionary_name, dictionary_file_path, log_file_path, sort
):
    """Recodes the rows of the main table with KNI and returns the output records"""
    # Fails early on unsupported outputs, before recoding
    _output_columns(model_dictionary_domain.get_dictionary(model_dictionary_name))

    # Serialize the tables; secondary records are indexed by the key of the main table
    main_header, main_records, main_frame = _table_text(deployment_ds.main_table, sort=False)
    secondary_tables = []
    if deployment_ds.is_multitable:
        main_key = deployment_ds.main_table.key
        main_keys = _key_tuples(main_frame, main_key)
        tables = {path: table for path, table, _ in deployment_ds.additional_data_tables}
        data_paths = sorted(model_dictionary_domain.extract_data_paths(model_dictionary_name),
                            key=lambda path: path.count("/"))
        for data_path in data_paths:
            if data_path not in tables:
                raise ValueError(f"Missing secondary table '{data_path}' in the dataset")
            header, records, frame = _table_text(tables[data_path], sort=sort)
            index = {}
            for row, key in enumerate(_key_tuples(frame, tables[data_path].key[: len(main_key)])):
                index.setdefault(key, []).append(row)
            secondary_tables.append((data_path, header, records, index))

    output_records = []
    step = "open_stream"
    with _kni_lock:
        kni = _get_kni()
        from kni import KNIError  # pylint: disable=import-outside-toplevel

        kni.set_log_file_name(log_file_path)
        stream = None
        try:
            stream = kni.open_stream(dictionary_file_path, model_dictionary_name, main_header, "\t")
            if secondary_tables:
                step = "set_secondary_header_line"
                for data_path, header, _, _ in secondary_tables:
                    kni.set_secondary_header_line(stream, data_path, header)
                step = "finish_opening_stream"
                kni.finish_opening_stream(stream)

            step = "recode_stream_record"
            output_length = _INITIAL_OUTPUT_LENGTH
            for row, record in enumerate(main_records):
                while True:
                    for data_path, _, records, index in secondary_tables:
                        for secondary_row in index.get(main_keys[row], ()):
                            kni.set_secondary_input_record(stream, data_path, records[secondary_row])
                    try:
                        output_records.append(kni.recode_stream_record(stream, record, output_length))
                        break
                    except KNIError as error:
                        # The secondary records are consumed: they are sent again with a larger buffer
                        if error.error_code != kni.KNI_ErrorStreamOutputRecord or output_length >= kni.KNI_MaxRecordLength:
                            raise
                        output_length = min(2 * output_length, kni.KNI_MaxRecordLength)
        except KNIError as error:
            log = ""
            if os.path.exists(log_file_path):
                with open(log_file_path, encoding="utf-8", errors="replace") as log_file:
                    log = log_file.read()[-2000:]
            raise RuntimeError(f"KNI deployment failed at {step}: {error}" + (f"\nKNI log:\n{log}" if log else "")) from error
        finally:
            if stream is not None:
                kni.close_stream(stream)
            kni.set_log_file_name("")
    return output_records


# --- parallel deployment -----------------------------------------------------------------------


def _recode_chunk(chunk_ds, model_dictionary_domain, model_dictionary_name, dictionary_file_path, log_file_path, sort):
    """Worker entry point: recodes a chunk with the KNI library of the worker process"""
    return _recode_records(
        chunk_ds, model_dictionary_domain, model_dictionary_name, dictionary_file_path, log_file_path, sort
    )


def _get_pool(n_processes):
    """Returns the persistent process pool, (re)created when its size changes (under _pool_lock)"""
    global _pool, _pool_size  # pylint: disable=global-statement
    if _pool is None or _pool_size != n_processes:
        shutdown_kni_pool()
        # forkserver rather than fork: the parent may hold a loaded KNI and a taken lock
        context = multiprocessing.get_context("forkserver")
        _pool = ProcessPoolExecutor(max_workers=n_processes, mp_context=context)
        _pool_size = n_processes
    return _pool


def shutdown_kni_pool():
    """Stops the processes used by deployment='kni_parallel' (they are restarted on demand)"""
    global _pool, _pool_size  # pylint: disable=global-statement
    if _pool is not None:
        _pool.shutdown(wait=True, cancel_futures=True)
    _pool, _pool_size = None, 0


atexit.register(shutdown_kni_pool)


def _string_keys(table, columns):
    keys = table.data_source[columns[0]].astype("str")
    for column in columns[1:]:
        keys = keys + "\x1f" + table.data_source[column].astype("str")
    return keys


def _chunk_table(table, data_source):
    chunk = copy.copy(table)
    chunk.data_source = data_source
    chunk.n_samples = data_source.shape[0]
    return chunk


def _split_dataset(deployment_ds, n_chunks):
    """Splits the dataset in chunks of consecutive main rows with their secondary rows"""
    bounds = np.linspace(0, deployment_ds.main_table.n_samples, n_chunks + 1).astype(int)
    main_source = deployment_ds.main_table.data_source
    if deployment_ds.is_multitable:
        main_key = deployment_ds.main_table.key
        main_keys = _string_keys(deployment_ds.main_table, main_key)
        secondary_keys = [
            _string_keys(table, table.key[: len(main_key)])
            for _, table, _ in deployment_ds.additional_data_tables
        ]
    chunks = []
    for start, stop in zip(bounds[:-1], bounds[1:]):
        rows = main_source.iloc[start:stop] if hasattr(main_source, "iloc") else main_source[start:stop]
        chunk = copy.copy(deployment_ds)
        chunk.main_table = _chunk_table(deployment_ds.main_table, rows)
        if deployment_ds.is_multitable:
            keys = set(main_keys.iloc[start:stop])
            chunk.additional_data_tables = [
                (path, _chunk_table(table, table.data_source[table_keys.isin(keys).to_numpy()]), is_one_to_one)
                for (path, table, is_one_to_one), table_keys in zip(deployment_ds.additional_data_tables, secondary_keys)
            ]
        chunks.append(chunk)
    return chunks


def deploy_model_kni_parallel(
    deployment_ds,
    model_dictionary_domain,
    model_dictionary_name,
    computation_dir,
    column_dtypes,
    sort=True,
    n_jobs=None,
):
    """Same as `deploy_model_kni` with the main table split among ``n_jobs`` processes

    Small datasets (less than ``MIN_ROWS_PER_CHUNK`` rows per process) are deployed in the
    current process.
    """
    n_processes = resolve_n_jobs(n_jobs)
    n_chunks = min(n_processes, deployment_ds.main_table.n_samples // MIN_ROWS_PER_CHUNK)
    if n_chunks <= 1:
        return deploy_model_kni(
            deployment_ds, model_dictionary_domain, model_dictionary_name, computation_dir, column_dtypes, sort
        )

    dictionary_file_path = os.path.join(computation_dir, "kni_deployment.kdic")
    model_dictionary_domain.export_khiops_dictionary_file(dictionary_file_path)
    chunks = _split_dataset(deployment_ds, n_chunks)
    output_records = []
    with _pool_lock:
        pool = _get_pool(n_processes)
        futures = [
            pool.submit(_recode_chunk, chunk, model_dictionary_domain, model_dictionary_name,
                        dictionary_file_path, os.path.join(computation_dir, f"khiops_kni_chunk_{index}.log"), sort)
            for index, chunk in enumerate(chunks)
        ]
        for index, future in enumerate(futures):
            try:
                output_records.extend(future.result())
            except BrokenProcessPool as error:
                shutdown_kni_pool()
                raise RuntimeError(f"KNI parallel deployment: a worker process died (chunk {index})") from error
            except Exception as error:
                for other in futures:
                    other.cancel()
                raise RuntimeError(f"KNI parallel deployment failed on chunk {index}: {error}") from error
    return _read_output(model_dictionary_domain, model_dictionary_name, output_records, column_dtypes)
