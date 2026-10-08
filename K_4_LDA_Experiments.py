"""
FORS-EMG — paired LOSO experiments for the INT8 LDA
===================================================
Runs several pipeline configurations on the SAME 19 leave-one-subject-out folds
and compares them per subject (paired), so that subject difficulty, which is
common to all configurations, cancels out of every comparison.

Requires K_4_LDA_Analysis.py (revision 2) in the same folder; all pipeline
pieces (loader, filters, windowing, features, LDA, INT8 PTQ) come from it.

Configurations (fixed in advance; plain LOSO is unbiased for A-C):
  A  zero-phase filter, all 8 types   original pipeline (sanity check: must
                                      reproduce the earlier 78.17 %)
  B  causal filter,     all 8 types   deployable filtering
  C  causal filter,     6 types       drop IEMG (≡ MAV) and WAMP (saturated)
  D  causal filter,     5 types       C without RMS → no square root anywhere
                                      (hardware-motivated; this candidate was
                                      also the Stage-4 pick of the selection
                                      run on these subjects, so its estimate
                                      is mildly optimistic)

Pre-specified paired comparisons:
  B vs A  two-sided        what causal filtering costs (a correction, not a choice)
  C vs B  non-inferiority  is the 6-type set as good as the full set?
  D vs C  non-inferiority  can RMS (the only sqrt) be dropped?
Split protocol is identical to the benchmark: fit on trials 1-4 of the 18
training subjects, test on all trials of the held-out subject.
"""

import time, csv
import numpy as np
from scipy.stats import t as student_t, wilcoxon
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from tabulate import tabulate

import K_4_LDA_Analysis as B

ALL8 = B.ALL_FEATURE_NAMES
SIX  = ["MAV", "RMS", "WL", "ZC", "SSC", "VAR"]
FIVE = ["MAV", "WL", "ZC", "SSC", "VAR"]          # no RMS → no square root

CONFIGS = [
    dict(key="A", name="Zero-phase, 8 types",          filt="zerophase", types=ALL8, log=False),
    dict(key="B", name="Causal, 8 types",              filt="causal",    types=ALL8, log=False),
    dict(key="C", name="Causal, 6 types",              filt="causal",    types=SIX,  log=False),
    dict(key="D", name="Causal, 5 types (−RMS, no sqrt)", filt="causal", types=FIVE, log=False),
]
COMPARISONS = [
    ("B", "A", "two-sided",      "Cost of causal (deployable) filtering"),
    ("C", "B", "non-inferiority", "6 types vs all 8"),
    ("D", "C", "non-inferiority", "Drop RMS (no square root)"),
]
TRAIN_TRIALS = (1, 2, 3, 4)
NI_MARGIN_PP = 1.0


# ─── Feature tables (one per filter mode, computed once) ──────────────────────

def build_feature_tables(subjects, modes):
    """
    Filters each recording in every requested mode and extracts the full 64-D
    feature matrix, without keeping filtered copies of the raw signals.
    Equivalent to per-fold extraction: windows never cross a recording and
    features are per-window; only scaler/model/PTQ depend on the split.
    """
    filt = B._design_filters()
    X = {m: [] for m in modes}
    y, sid, trial = [], [], []
    for s in tqdm(subjects, desc="Filtering + features"):
        for rec, lbl, meta in zip(s["recordings"], s["labels"], s["meta"]):
            for m in modes:
                sig = B.apply_preprocessing_filter(rec.T, filt, m).T
                w, yy = B.windows_from_recordings([sig], [lbl])
                X[m].append(B.extract_features_from_windows(w))
            y.append(yy)
            sid.append(np.full(len(yy), s["subject_id"]))
            trial.append(np.full(len(yy), meta["trial"]))
    return ({m: np.vstack(v) for m, v in X.items()},
            np.concatenate(y), np.concatenate(sid), np.concatenate(trial))


def fit_eval(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    Xtr_n, Xte_n = sc.transform(Xtr).astype(np.float64), sc.transform(Xte).astype(np.float64)
    lda = B.build_lda().fit(Xtr_n, ytr)
    W, b = B.lda_as_linear_layer(lda)
    q = B.calibrate_ptq([(W, b)], Xtr_n)
    return dict(
        train_int8=accuracy_score(ytr, lda.classes_[B.int8_infer(q, Xtr_n)]),
        test_fp32=accuracy_score(yte, lda.predict(Xte_n)),
        test_int8=accuracy_score(yte, lda.classes_[B.int8_infer(q, Xte_n)]),
    )


def mean_ci(v):
    v = np.asarray(v, dtype=float); n = len(v)
    return v.mean(), student_t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n)


def main():
    t0 = time.time()
    print("=" * 72)
    print("  FORS-EMG — paired LOSO experiments (LDA, INT8)")
    print("=" * 72)
    subjects = B.load_all_subjects(B.BASE_DIR)
    modes = sorted({c["filt"] for c in CONFIGS})
    Xm, y, sid, trial = build_feature_tables(subjects, modes)
    ids = [s["subject_id"] for s in subjects]
    del subjects
    print(f"\n  Windows: {len(y):,}   Subjects: {len(ids)}   Filter modes: {modes}")

    print(f"\n  Configurations:")
    print(tabulate([[c["key"], c["name"], c["filt"], len(c["types"]),
                     B.N_CHANNELS * len(c["types"]),
                     B.N_CHANNELS * len(c["types"]) * B.N_CLASSES + B.N_CLASSES * 4,
                     B.N_CHANNELS * len(c["types"]) * B.N_CLASSES]
                    for c in CONFIGS],
                   headers=["", "Configuration", "Filter", "Types", "Inputs", "Bytes", "MACs"],
                   tablefmt="simple"))

    # Transformed feature matrices, one per configuration
    Xc = {c["key"]: B.transform_features(Xm[c["filt"]], c["types"], c["log"]) for c in CONFIGS}

    # ── LOSO ─────────────────────────────────────────────────────────────────
    res = {c["key"]: [] for c in CONFIGS}
    in_train = np.isin(trial, TRAIN_TRIALS)
    for t in tqdm(ids, desc="LOSO folds"):
        pool, test = (sid != t) & in_train, (sid == t)
        for c in CONFIGS:
            X = Xc[c["key"]]
            res[c["key"]].append(fit_eval(X[pool], y[pool], X[test], y[test]))

    def col(k, m):
        return np.array([r[m] for r in res[k]])

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*72}\n  Accuracy (mean over {len(ids)} held-out subjects; ± = 95% CI)\n{'─'*72}")
    rows = []
    for c in CONFIGS:
        k = c["key"]
        m8, h8 = mean_ci(col(k, "test_int8"))
        rows.append([k, c["name"], f"{col(k,'train_int8').mean()*100:.2f}",
                     f"{col(k,'test_fp32').mean()*100:.2f}",
                     f"{m8*100:.2f} ± {h8*100:.2f}",
                     f"{(col(k,'train_int8').mean()-m8)*100:.2f}"])
    print(tabulate(rows, headers=["", "Configuration", "Train INT8 %", "Test FP32 %",
                                  "Test INT8 %", "Train−test pp"], tablefmt="simple"))

    # ── Paired comparisons ───────────────────────────────────────────────────
    print(f"\n{'─'*72}\n  Pre-specified paired comparisons (test INT8, per subject)\n{'─'*72}")
    crow = []
    for a, b, kind, desc in COMPARISONS:
        d = (col(a, "test_int8") - col(b, "test_int8")) * 100
        m, h = mean_ci(d)
        try:
            p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        except ValueError:
            p = 1.0
        lo, hi = m - h, m + h
        if kind == "non-inferiority":
            verdict = f"{'PASS' if lo > -NI_MARGIN_PP else 'FAIL'} (margin −{NI_MARGIN_PP:g} pp)"
        else:
            verdict = ("better" if lo > 0 else "worse" if hi < 0 else "no detectable difference")
        crow.append([f"{a} vs {b}", desc, f"{m:+.2f}", f"[{lo:+.2f}, {hi:+.2f}]",
                     f"{p:.3f}", f"{(d > 0).sum()}/{(d < 0).sum()}", verdict])
    print(tabulate(crow, headers=["", "Question", "Δ pp", "95% CI", "Wilcoxon p",
                                  "Up/Down", "Verdict"], tablefmt="simple"))

    # ── Per-subject ──────────────────────────────────────────────────────────
    print(f"\n  Per-subject test INT8 accuracy (%):")
    keys = [c["key"] for c in CONFIGS]
    prow = [[f"S{t:02d}"] + [f"{res[k][i]['test_int8']*100:.2f}" for k in keys]
            + [f"{(res['C'][i]['test_int8']-res['B'][i]['test_int8'])*100:+.2f}"]
            for i, t in enumerate(ids)]
    print(tabulate(prow, headers=["Subject"] + keys + ["C−B"], tablefmt="simple"))

    # ── CSV ──────────────────────────────────────────────────────────────────
    with open("fors_emg_lda_experiments.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Subject"] + [f"{k}_{m}" for k in keys
                                  for m in ("train_int8", "test_fp32", "test_int8")])
        for i, t in enumerate(ids):
            w.writerow([t] + [round(res[k][i][m], 6) for k in keys
                              for m in ("train_int8", "test_fp32", "test_int8")])
    print(f"\n  Saved → fors_emg_lda_experiments.csv")
    print(f"  Total runtime : {(time.time()-t0)/60:.1f} min")
    print("=" * 72)


if __name__ == "__main__":
    main()
