"""
FORS-EMG — ASIC-friendly feature experiments (paired LOSO, INT8 LDA)
====================================================================
Tests integer, multiplier-free versions of the validated float feature set
{MAV, WL, ZC, SSC, VAR} on the SAME 19 leave-one-subject-out folds, so every
comparison is paired per subject.

Requires K_4_LDA_Analysis.py (revision 3) in the same folder.

Reference
  R   float {MAV, WL, ZC(|x| ≥ 1% RMS), SSC, VAR}          current validated set

Integer feature sets (filtered signal quantised to SIGNAL_BITS; all use
SABS = Σ|x| for MAV, WL = Σ|Δx|, SSC by sign-bit compare):
  H1  ZC_MAV14 + POW      exact Σx² (serial shift-add squarer)
  H2  ZC_MAV14 + POW_M2   2-level Mitchell squares   (max rel. error < 6.25 %)
  H3  ZC_MAV14 + POW_M1   1-level Mitchell squares   (max rel. error < 25 %)
  H4  ZC_MAV14 + SABS2    (Σ|x|)², one square per channel per WINDOW
  H5  ZC_MAV14 only       no power term at all
  H6  ZC_NODB  + POW_M1   cheapest: no ZC deadband, 1-level Mitchell

Each H config is evaluated twice:
  standard : float z-score + float INT8 input scale (isolates the feature change)
  shift    : integer (x − μ) >> k input stage (fully integer front end)

Pre-specified comparisons (non-inferiority, margin NI_MARGIN_PP):
  Hk(standard) vs R        does the integer feature set keep the accuracy?
  Hk(shift) vs Hk(standard) does the integer input stage keep the accuracy?
"""

import time, csv
import numpy as np
from scipy.stats import t as student_t, wilcoxon
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from tabulate import tabulate

import K_4_LDA_Analysis as B

REF_TYPES = ["MAV", "WL", "ZC", "SSC", "VAR"]
CONFIGS = [
    dict(key="H1", name="Exact Σx² (serial shift-add)",   types=["SABS", "WL", "ZC_MAV14", "SSC", "POW"]),
    dict(key="H2", name="Mitchell-2 Σx²",                 types=["SABS", "WL", "ZC_MAV14", "SSC", "POW_M2"]),
    dict(key="H3", name="Mitchell-1 Σx²",                 types=["SABS", "WL", "ZC_MAV14", "SSC", "POW_M1"]),
    dict(key="H4", name="(Σ|x|)² per window",             types=["SABS", "SABS2", "WL", "ZC_MAV14", "SSC"]),
    dict(key="H5", name="No power term",                  types=["SABS", "WL", "ZC_MAV14", "SSC"]),
    dict(key="H6", name="ZC no deadband + Mitchell-1",    types=["SABS", "WL", "ZC_NODB", "SSC", "POW_M1"]),
]
PRIMITIVES   = sorted({f for c in CONFIGS for f in c["types"]})
TRAIN_TRIALS = (1, 2, 3, 4)
NI_MARGIN_PP = 1.0
FILTER_MODE  = "causal"


# ─── Feature tables ───────────────────────────────────────────────────────────

def build_tables(subjects):
    """
    Pass 1: causal filter → float reference features, and the dataset full
            scale (max |x|) that fixes the integer grid.
    Pass 2: causal filter → quantise → integer hardware primitives.
    Filtering twice avoids holding ~0.6 GB of filtered signals in memory.
    """
    filt = B._design_filters()
    Xref, y, sid, trial = [], [], [], []
    full_scale = 0.0
    for s in tqdm(subjects, desc="Pass 1: float features"):
        for rec, lbl, m in zip(s["recordings"], s["labels"], s["meta"]):
            sig = B.apply_preprocessing_filter(rec.T, filt, FILTER_MODE).T
            full_scale = max(full_scale, float(np.abs(sig).max()))
            w, yy = B.windows_from_recordings([sig], [lbl])
            Xref.append(B.transform_features(B.extract_features_from_windows(w), REF_TYPES, False))
            y.append(yy)
            sid.append(np.full(len(yy), s["subject_id"]))
            trial.append(np.full(len(yy), m["trial"]))

    prim = {p: [] for p in PRIMITIVES}
    for s in tqdm(subjects, desc="Pass 2: integer features"):
        for rec, lbl in zip(s["recordings"], s["labels"]):
            sig = B.apply_preprocessing_filter(rec.T, filt, FILTER_MODE).T
            w, _ = B.windows_from_recordings([sig], [lbl])
            wi = B.quantize_signal(w, full_scale)
            d = B.hw_feature_dict(wi, PRIMITIVES)
            for p in PRIMITIVES:
                prim[p].append(d[p])
    return (np.vstack(Xref).astype(np.float64),
            {p: np.vstack(v) for p, v in prim.items()},
            np.concatenate(y), np.concatenate(sid), np.concatenate(trial), full_scale)


def assemble(prim, types):
    """Stack primitives channel-major → (N, C·T)."""
    N, C = prim[types[0]].shape
    return np.stack([prim[t] for t in types], axis=2).reshape(N, C * len(types)).astype(np.float64)


# ─── Fit / evaluate ───────────────────────────────────────────────────────────

def fit_eval(Xtr, ytr, Xte, yte, norm):
    sc = B.make_normalizer(norm)
    Xtr_n = np.asarray(sc.fit_transform(Xtr), dtype=np.float64)
    Xte_n = np.asarray(sc.transform(Xte), dtype=np.float64)
    lda = B.build_lda().fit(Xtr_n, ytr)
    W, b = B.lda_as_linear_layer(lda)
    q = B.calibrate_ptq([(W, b)], Xtr_n, input_scale=1.0 if norm == "shift" else None)
    return dict(
        train_int8=accuracy_score(ytr, lda.classes_[B.int8_infer(q, Xtr_n)]),
        test_fp32=accuracy_score(yte, lda.predict(Xte_n)),
        test_int8=accuracy_score(yte, lda.classes_[B.int8_infer(q, Xte_n)]),
    )


def mean_ci(v):
    v = np.asarray(v, dtype=float); n = len(v)
    return v.mean(), student_t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n)


def paired(a, b):
    d = (np.asarray(a) - np.asarray(b)) * 100
    m, h = mean_ci(d)
    try:
        p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    except ValueError:
        p = 1.0
    lo, hi = m - h, m + h
    verdict = f"{'PASS' if lo > -NI_MARGIN_PP else 'FAIL'}"
    return [f"{m:+.2f}", f"[{lo:+.2f}, {hi:+.2f}]", f"{p:.3f}",
            f"{(d > 0).sum()}/{(d < 0).sum()}", verdict]


def channel_corr(a, b):
    """Mean over channels of the Pearson correlation between two (N, C) arrays."""
    return np.mean([np.corrcoef(a[:, c], b[:, c])[0, 1] for c in range(a.shape[1])])


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 76)
    print("  FORS-EMG — ASIC-friendly integer features (paired LOSO, INT8 LDA)")
    print("=" * 76)
    subjects = B.load_all_subjects(B.BASE_DIR)
    Xref, prim, y, sid, trial, fs = build_tables(subjects)
    ids = [s["subject_id"] for s in subjects]
    del subjects
    lsb = fs / (2 ** (B.SIGNAL_BITS - 1) - 1)
    print(f"\n  Windows: {len(y):,}   Subjects: {len(ids)}   Filter: {FILTER_MODE}")
    print(f"  Signal: {B.SIGNAL_BITS}-bit signed, full scale {fs:.6g} (LSB {lsb:.4g})")

    # ── Configuration table ──────────────────────────────────────────────────
    rows = [["R", "Float reference", ", ".join(REF_TYPES), 40, 176]]
    for c in CONFIGS:
        n_in = B.N_CHANNELS * len(c["types"])
        rows.append([c["key"], c["name"], ", ".join(c["types"]), n_in, n_in * B.N_CLASSES + B.N_CLASSES * 4])
    print(tabulate(rows, headers=["", "Configuration", "Features", "Inputs", "LDA bytes"],
                   tablefmt="simple"))

    # ── Fidelity of the approximations (descriptive, no labels) ──────────────
    C = B.N_CHANNELS
    ref_cols = lambda f: Xref[:, [ch * len(REF_TYPES) + REF_TYPES.index(f) for ch in range(C)]]
    fid = []
    if "POW" in prim:
        for p in [q for q in PRIMITIVES if q.startswith("POW_M")] + (["SABS2"] if "SABS2" in prim else []):
            ratio = np.median(prim[p] / np.maximum(prim["POW"], 1))
            fid.append([f"{p} vs POW (exact Σx²)", f"{channel_corr(prim[p], prim['POW']):.4f}",
                        f"{ratio:.4f}"])
        fid.append(["POW vs float VAR", f"{channel_corr(prim['POW'], ref_cols('VAR')):.4f}", ""])
    for p in [q for q in PRIMITIVES if q.startswith("ZC_")]:
        fid.append([f"{p} vs float ZC (1% RMS)", f"{channel_corr(prim[p], ref_cols('ZC')):.4f}",
                    f"{np.median(prim[p] / np.maximum(ref_cols('ZC'), 1)):.4f}"])
    fid.append(["SABS vs float MAV", f"{channel_corr(prim['SABS'], ref_cols('MAV')):.4f}", ""])
    print("\n  Feature fidelity (all windows; Pearson r averaged over channels):")
    print(tabulate(fid, headers=["Integer feature", "r", "Median ratio"], tablefmt="simple"))

    # ── LOSO ─────────────────────────────────────────────────────────────────
    Xc = {c["key"]: assemble(prim, c["types"]) for c in CONFIGS}
    runs = [("R", "standard")] + [(c["key"], n) for c in CONFIGS for n in ("standard", "shift")]
    res = {r: [] for r in runs}
    in_train = np.isin(trial, TRAIN_TRIALS)
    for t in tqdm(ids, desc="LOSO folds"):
        pool, test = (sid != t) & in_train, (sid == t)
        for key, norm in runs:
            X = Xref if key == "R" else Xc[key]
            res[(key, norm)].append(fit_eval(X[pool], y[pool], X[test], y[test], norm))

    col = lambda r, m: np.array([f[m] for f in res[r]])

    # ── Accuracy summary ─────────────────────────────────────────────────────
    print(f"\n{'─'*76}\n  Accuracy (mean over {len(ids)} held-out subjects; ± = 95% CI)\n{'─'*76}")
    arows = []
    names = {"R": "Float reference", **{c["key"]: c["name"] for c in CONFIGS}}
    for key in ["R"] + [c["key"] for c in CONFIGS]:
        m, h = mean_ci(col((key, "standard"), "test_int8"))
        shift = ""
        if key != "R":
            ms, hs = mean_ci(col((key, "shift"), "test_int8"))
            shift = f"{ms*100:.2f} ± {hs*100:.2f}"
        arows.append([key, names[key],
                      f"{col((key,'standard'),'train_int8').mean()*100:.2f}",
                      f"{col((key,'standard'),'test_fp32').mean()*100:.2f}",
                      f"{m*100:.2f} ± {h*100:.2f}", shift])
    print(tabulate(arows, headers=["", "Configuration", "Train INT8 %", "Test FP32 %",
                                   "Test INT8 % (std norm)", "Test INT8 % (shift norm)"],
                   tablefmt="simple"))

    # ── Paired comparisons ───────────────────────────────────────────────────
    print(f"\n{'─'*76}\n  Paired comparisons, test INT8 (non-inferiority margin −{NI_MARGIN_PP:g} pp)\n{'─'*76}")
    ref = col(("R", "standard"), "test_int8")
    prow = [[f"{c['key']} vs R", "integer features (std norm)"]
            + paired(col((c["key"], "standard"), "test_int8"), ref) for c in CONFIGS]
    prow += [[f"{c['key']} shift vs std", "integer input stage"]
             + paired(col((c["key"], "shift"), "test_int8"), col((c["key"], "standard"), "test_int8"))
             for c in CONFIGS]
    prow += [[f"{c['key']} shift vs R", "fully integer front end"]
             + paired(col((c["key"], "shift"), "test_int8"), ref) for c in CONFIGS]
    print(tabulate(prow, headers=["", "Question", "Δ pp", "95% CI", "Wilcoxon p", "Up/Down", "NI"],
                   tablefmt="simple"))
    print(f"  ({len(prow)} comparisons; for family-wise α = 0.05 use p < {0.05/len(prow):.4f})")

    # ── Per-subject ──────────────────────────────────────────────────────────
    keys = ["R"] + [c["key"] for c in CONFIGS]
    print(f"\n  Per-subject test INT8 accuracy (%), standard normalisation:")
    print(tabulate([[f"S{t:02d}"] + [f"{res[(k,'standard')][i]['test_int8']*100:.2f}" for k in keys]
                    for i, t in enumerate(ids)],
                   headers=["Subject"] + keys, tablefmt="simple"))

    # ── CSV ──────────────────────────────────────────────────────────────────
    with open("fors_emg_lda_hw_experiments.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Subject"] + [f"{k}_{n}_{m}" for k, n in runs
                                  for m in ("train_int8", "test_fp32", "test_int8")])
        for i, t in enumerate(ids):
            w.writerow([t] + [round(res[r][i][m], 6) for r in runs
                              for m in ("train_int8", "test_fp32", "test_int8")])
    print(f"\n  Saved → fors_emg_lda_hw_experiments.csv")
    print(f"  Total runtime : {(time.time()-t0)/60:.1f} min")
    print("=" * 76)


if __name__ == "__main__":
    main()
