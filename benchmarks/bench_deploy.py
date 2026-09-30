"""Temps de predict_proba selon le nombre de lignes : déploiement par le runner Khiops vs khiops-kni.

    python benchmarks/bench_deploy.py [--sizes 1 100 10000 100000] [--repeats 3]
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from khiops import core as kh

from khiops_fast import KhiopsFastClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 100, 10000, 100000])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    df = pd.read_csv(os.path.join(kh.get_samples_dir(), "Adult", "Adult.txt"), sep="\t")
    X, y = df.drop(columns=["class", "Label"], errors="ignore"), df["class"]
    clf = KhiopsFastClassifier().fit(X, y)

    rows = []
    for size in args.sizes:
        X_test = X.sample(size, replace=size > len(X), random_state=0)
        times = {}
        for deployment in ("khiops", "kni"):
            clf.set_params(deployment=deployment)
            clf.predict_proba(X_test.iloc[:1])
            durations = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                clf.predict_proba(X_test)
                durations.append(time.perf_counter() - start)
            times[deployment] = float(np.median(durations))
        rows.append({"n_rows": size, "khiops_s": times["khiops"], "kni_s": times["kni"],
                     "speedup": times["khiops"] / times["kni"]})
        print(rows[-1], flush=True)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
