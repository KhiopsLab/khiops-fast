"""Temps de predict_proba selon le nombre de lignes : runner Khiops, khiops-kni et khiops-kni en parallèle.

    python benchmarks/bench_deploy.py [--sizes 1 100 10000 100000 1000000] [--n-jobs 2 4 8 16] [--repeats 3]
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from khiops import core as kh

from khiops_fast import KhiopsFastClassifier


def measure(clf, X_test, repeats):
    clf.predict_proba(X_test.iloc[: min(len(X_test), 20000)])  # warm-up (starts the process pool)
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        clf.predict_proba(X_test)
        durations.append(time.perf_counter() - start)
    return float(np.median(durations))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 100, 10000, 100000, 1000000])
    parser.add_argument("--n-jobs", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    df = pd.read_csv(os.path.join(kh.get_samples_dir(), "Adult", "Adult.txt"), sep="\t")
    X, y = df.drop(columns=["class", "Label"], errors="ignore"), df["class"]
    clf = KhiopsFastClassifier().fit(X, y)

    modes = [("khiops", None), ("kni", None)] + [("kni_parallel", n) for n in args.n_jobs]
    rows = []
    for size in args.sizes:
        X_test = X.sample(size, replace=size > len(X), random_state=0)
        row = {"n_rows": size}
        for deployment, n_jobs in modes:
            name = deployment if n_jobs is None else f"{deployment}_{n_jobs}"
            row[name] = measure(clf.set_params(deployment=deployment, n_jobs=n_jobs), X_test, args.repeats)
        rows.append(row)
        print(row, flush=True)
    print("\nTemps médian de predict_proba (s) sur Adult :")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
