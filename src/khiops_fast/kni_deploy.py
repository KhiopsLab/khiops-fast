######################################################################################
# Copyright (c) 2023-2025 Orange. All rights reserved.                               #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""In-memory deployment of Khiops models with khiops-kni (Khiops Native Interface)

KNI is not reentrant: all the deployments of a process are serialized by a global lock.
Separate processes each load their own copy of the library and do not block each other.
"""
import io
import os
import threading

from khiops_fast.dataset import SparseTable, read_internal_data_table, write_internal_data_table

DEPLOYMENT_MODES = ("khiops", "kni")

_kni = None
_kni_lock = threading.Lock()
_INITIAL_OUTPUT_LENGTH = 64 * 1024


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
    # KNI reads the dictionary from a file
    dictionary_file_path = os.path.join(computation_dir, "kni_deployment.kdic")
    model_dictionary_domain.export_khiops_dictionary_file(dictionary_file_path)
    log_file_path = os.path.join(computation_dir, "khiops_kni.log")
    output_columns = _output_columns(model_dictionary_domain.get_dictionary(model_dictionary_name))

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

    text = "\t".join(output_columns) + "\n" + "".join(record + "\n" for record in output_records)
    with io.StringIO(text) as stream:
        return read_internal_data_table(stream, column_dtypes=column_dtypes)
