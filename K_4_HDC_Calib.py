"""
FORS-EMG — HDC with per-user calibration (post-training adaptation), LOSO
=========================================================================
Same data, encodings and training as K_4_HDC_Way1_vs_Way2.py (imported from it).
For each held-out subject:
    calibration  trial CALIB_TRIAL of that subject (every gesture once, all orientations)
    test         that subject's other trials (identical for every row below)
Rows:
    LDA                 trained on the training subjects only
    LDA + calib         refitted offline on training subjects + the calibration trial
                        (the chip cannot do this: it needs the covariance re-inverted)
    HDC Way 1 / Way 2   trained on the training subjects only
    ... + calib         the trained accumulators updated with the calibration windows only,
                        CALIB_PASSES perceptron passes (additions only: possible on the chip),
                        each calibration window counted CALIB_WEIGHT times
Prints results only; writes no files.

Run:  python K_4_HDC_Calib.py
      (K_4_HDC_Way1_vs_Way2.py, K_4_LDA_Analysis.py and lda_golden_model.py in the same folder)
"""

import time
import numpy as np
from tabulate import tabulate

import K_4_HDC_Way1_vs_Way2 as H
import lda_golden_model as G

CALIB_TRIAL  = 1
CALIB_PASSES = 10
CALIB_WEIGHT = 8


def calibrate(A, Hc, yc):
    """Perceptron passes over the calibration windows, starting from the trained accumulators."""
    A = A.copy()
    X = H.pm(Hc) * CALIB_WEIGHT
    for _ in range(CALIB_PASSES):
        p = H.predict(A, X)
        wrong = p != yc
        if not wrong.any():
            break
        for k in range(H.K):
            A[k] += X[wrong & (yc == k)].sum(axis=0)
            A[k] -= X[wrong & (p == k)].sum(axis=0)
    return A


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"  FORS-EMG — HDC with per-user calibration (trial {CALIB_TRIAL}), LOSO   d = {H.DIM}")
    print("=" * 78)
    recs, ids = H.load()
    raw, H2, y, sid, trial = H.build(recs)
    del recs

    keys = ["lda", "lda_c", "w1", "w1_c", "w2", "w2_c"]
    res = {k: [] for k in keys}
    for i, s in enumerate(ids):
        tr = (sid != s) & np.isin(trial, H.TRAIN_TRIALS)
        va = (sid != s) & (trial == H.VAL_TRIAL)
        ca = (sid == s) & (trial == CALIB_TRIAL)
        te = (sid == s) & (trial != CALIB_TRIAL)

        mu, k = G.calibrate_norm(raw[tr | va])
        xq = G.normalize(raw, mu, k)

        def lda_acc(mask):
            lda, W, b = G.fit_int8_lda(xq[mask], y[mask])
            return np.mean(lda.classes_[np.argmax(xq[te] @ W.T + b, axis=1)] == y[te])

        res["lda"].append(lda_acc(tr | va))
        res["lda_c"].append(lda_acc(tr | va | ca))

        for key, Hall in (("w1", H.way1_encode(xq)), ("w2", H2)):
            A, _ = H.train(Hall[tr], y[tr], Hall[va], y[va])
            res[key].append(np.mean(H.predict(A, H.pm(Hall[te])) == y[te]))
            Ac = calibrate(A, Hall[ca], y[ca])
            res[key + "_c"].append(np.mean(H.predict(Ac, H.pm(Hall[te])) == y[te]))

        print(f"  [{i + 1:2d}/{len(ids)}] S{s:02d}   LDA {res['lda'][-1]:.1%} -> {res['lda_c'][-1]:.1%}   "
              f"Way 1 {res['w1'][-1]:.1%} -> {res['w1_c'][-1]:.1%}   "
              f"Way 2 {res['w2'][-1]:.1%} -> {res['w2_c'][-1]:.1%}", flush=True)

    print(f"\n{'─' * 78}\n  Test accuracy on trials != {CALIB_TRIAL} (mean over {len(ids)} subjects; ± = 95% CI)\n{'─' * 78}")
    rows = []
    for name, a, b in (("INT8 LDA", "lda", "lda_c"), ("HDC Way 1 (features)", "w1", "w1_c"),
                       ("HDC Way 2 (samples)", "w2", "w2_c")):
        ma, ha = H.mean_ci(res[a]); mb, hb = H.mean_ci(res[b])
        rows.append([name, f"{ma * 100:.2f} ± {ha * 100:.2f}", f"{mb * 100:.2f} ± {hb * 100:.2f}",
                     f"{(mb - ma) * 100:+.2f}"])
    print(tabulate(rows, headers=["Model", "No calibration %", "With calibration %", "Gain pp"],
                   tablefmt="simple"))
    print()
    print(tabulate([["Way 1 calib vs Way 1"] + H.paired(res["w1_c"], res["w1"]),
                    ["Way 2 calib vs Way 2"] + H.paired(res["w2_c"], res["w2"]),
                    ["LDA calib vs LDA"] + H.paired(res["lda_c"], res["lda"]),
                    ["Way 2 calib vs LDA (no calib)"] + H.paired(res["w2_c"], res["lda"]),
                    ["Way 2 calib vs LDA calib"] + H.paired(res["w2_c"], res["lda_c"])],
                   headers=["Comparison", "Δ pp", "95% CI", "Wilcoxon p", "Up/Down", "Verdict"],
                   tablefmt="simple"))
    print(f"\n  Total runtime : {(time.time() - t0) / 60:.1f} min")
    print("=" * 78)


if __name__ == "__main__":
    main()
