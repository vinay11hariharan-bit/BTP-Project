"""
FORS-EMG — subject-independent feature-type selection for the INT8 LDA
=====================================================================
Selects which of the 8 time-domain feature TYPES (each computed on all 8
channels) to keep, using a nested leave-one-subject-out protocol so the
reported accuracy of the reduced model is an unbiased subject-independent
estimate.

Requires K_4_LDA_Analysis.py (the LDA benchmark) in the same folder: every pipeline piece
(loader, filter, windowing, features, LDA config, INT8 PTQ) is imported from it,
so the reduced models differ from the benchmark ONLY in which columns they use.

Method
------
0. A-priori (analytic, no data): IEMG = L·MAV exactly for fixed-length windows,
   and LDA is invariant to per-feature affine rescaling, so IEMG and MAV define
   the identical classifier. IEMG is excluded from the search (MAV stands for
   both). On hardware, compute Σ|x| (no divide) — the scaler absorbs 1/L.

1. Redundancy diagnostics (unsupervised, descriptive only — NOT used for
   selection): Spearman |ρ| between feature types, hierarchical clustering on
   1−|ρ|, and log-domain PCA per channel (eigen-spectrum, participation ratio).
   Rationale: for a stationary Gaussian process every feature here is a function
   of the spectral moments m0, m2, m4 (Hjorth activity/mobility/complexity);
   in the log domain those relations are linear, so PCA exposes the true
   intrinsic dimension (~3 per channel for Gaussian data; more if real sEMG is
   non-Gaussian / non-stationary within a window).

2. Selection (supervised, subject-independent): exhaustive search over all
   2^7 − 1 = 127 subsets of the 7 candidate types. Criterion = inner LOSO
   accuracy of the LDA across the TRAINING subjects only. Among all subsets,
   the one-standard-error rule picks the SMALLEST subset whose mean inner
   deficit to the best subset is within 1 PAIRED SE (SE of the per-fold
   difference; ties → higher mean). The outer paired Δ is then checked
   against a pre-specified non-inferiority margin.

3. Nested evaluation: steps 2 runs inside each of the 19 outer LOSO folds; the
   selected subset is refit on the outer training pool and tested on the unseen
   subject, alongside the full 8-type model on the same fold (paired test).

4. Final subset: step 2 run once on all subjects. The nested estimate from
   step 3 is the accuracy to expect from THIS procedure on a new person.

Splits match the benchmark: model fitting uses trials 1-4 of the training
subjects; the outer test subject contributes all 5 trials.
"""

import time, csv, itertools, warnings
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import spearmanr, t as student_t, wilcoxon
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from tabulate import tabulate

import K_4_LDA_Analysis as B

warnings.filterwarnings("ignore")

# ─── Configuration ────────────────────────────────────────────────────────────
FEATURE_NAMES     = ["MAV", "RMS", "WL", "ZC", "SSC", "VAR", "WAMP", "IEMG"]
COUNT_FEATURES    = {"ZC", "SSC", "WAMP"}      # integer counts (log1p in PCA)
EXCLUDE_A_PRIORI  = ["IEMG"]                   # exact duplicate of MAV
TRAIN_TRIALS      = (1, 2, 3, 4)               # same as benchmark (trial 5 = val)
INNER_CV          = "loso"                     # "loso" or an int k → GroupKFold(k)
SE_MULTIPLIER     = 1.0                        # 1-SE rule
SELECTION_RULE    = "paired"                   # "paired" (SE of the difference to the
                                               # best subset) or "classic" (SE of the best)
NI_MARGIN_PP      = 1.0                        # pre-specified non-inferiority margin
REDUNDANCY_RHO    = 0.90                       # |ρ| threshold for clustering report
N_JOBS            = -1                         # parallel subset evaluation

N_TYPES     = len(FEATURE_NAMES)               # 8 (columns laid out as ch*8 + type)
CANDIDATES  = [i for i, n in enumerate(FEATURE_NAMES) if n not in EXCLUDE_A_PRIORI]
SUBSETS     = [S for k in range(1, len(CANDIDATES) + 1)
               for S in itertools.combinations(CANDIDATES, k)]
FULL_SET    = tuple(range(N_TYPES))            # the benchmark's 64-D model


def type_cols(types):
    """Column indices for a set of feature types, all channels (layout ch*8+type)."""
    return np.array(sorted(c * N_TYPES + f for c in range(B.N_CHANNELS) for f in types))

def subset_name(S):
    return "{" + ", ".join(FEATURE_NAMES[i] for i in S) + "}"


# ─── Feature table (computed once) ────────────────────────────────────────────

def build_feature_table(subjects):
    """
    One row per window, all 64 features. Identical to computing them inside each
    fold: windows never cross a recording and features are per-window, so only
    the scaler/model depend on the split — and those are fit per fold below.
    """
    X, y, sid, trial = [], [], [], []
    for s in subjects:
        for rec, lbl, m in zip(s["recordings"], s["labels"], s["meta"]):
            w, yy = B.windows_from_recordings([rec], [lbl])
            X.append(B.extract_features_from_windows(w))
            y.append(yy)
            sid.append(np.full(len(yy), s["subject_id"]))
            trial.append(np.full(len(yy), m["trial"]))
    return (np.vstack(X).astype(np.float64), np.concatenate(y),
            np.concatenate(sid), np.concatenate(trial))


# ─── Stage 1: unsupervised redundancy diagnostics ─────────────────────────────

def redundancy_diagnostics(X):
    L = B.WIN_SAMPLES
    wamp = X[:, type_cols([FEATURE_NAMES.index("WAMP")])] / (L - 1)
    print(f"  WAMP saturation: mean WAMP/(L−1) = {wamp.mean():.3f} "
          f"(1.0 = every sample difference exceeds the threshold)")

    # Spearman |ρ| between feature types, per channel, averaged over channels
    rho = np.zeros((N_TYPES, N_TYPES))
    for c in range(B.N_CHANNELS):
        r = spearmanr(X[:, c * N_TYPES:(c + 1) * N_TYPES]).correlation
        rho += np.abs(np.nan_to_num(r))
    rho /= B.N_CHANNELS
    print("\n  Spearman |ρ| between feature types (mean over 8 channels):")
    print(tabulate([[FEATURE_NAMES[i]] + [f"{rho[i, j]:.3f}" for j in range(N_TYPES)]
                    for i in range(N_TYPES)],
                   headers=[""] + FEATURE_NAMES, tablefmt="simple"))

    D = 1.0 - rho
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(np.clip(D, 0, None), checks=False), method="average")
    lab = fcluster(Z, t=1.0 - REDUNDANCY_RHO, criterion="distance")
    print(f"\n  Redundancy clusters (average linkage, |ρ| ≥ {REDUNDANCY_RHO}):")
    for k in np.unique(lab):
        print("    " + ", ".join(FEATURE_NAMES[i] for i in np.where(lab == k)[0]))

    # Log-domain PCA per channel (linearises m0/m2/m4 power laws)
    evs = []
    for c in range(B.N_CHANNELS):
        F = X[:, c * N_TYPES:(c + 1) * N_TYPES]
        Zl = np.empty_like(F)
        for f, name in enumerate(FEATURE_NAMES):
            if name in COUNT_FEATURES:
                Zl[:, f] = np.log1p(F[:, f])
            else:
                Zl[:, f] = np.log(np.maximum(F[:, f], 1e-6 * np.median(F[:, f]) + 1e-30))
        sd = Zl.std(0)
        keep = sd > 0
        C = np.corrcoef(Zl[:, keep].T)
        ev = np.sort(np.linalg.eigvalsh(C))[::-1]
        evs.append(np.pad(ev, (0, N_TYPES - len(ev))))
    ev = np.mean(evs, axis=0)
    cum = np.cumsum(ev) / ev.sum()
    pr = ev.sum() ** 2 / (ev ** 2).sum()
    print("\n  Log-domain PCA of the 8 feature types (per channel, averaged):")
    print(tabulate([[f"PC{i+1}", f"{ev[i]:.3f}", f"{cum[i]*100:.2f}%"] for i in range(N_TYPES)],
                   headers=["", "Eigenvalue", "Cumulative var"], tablefmt="simple"))
    print(f"  Participation ratio (Σλ)²/Σλ² = {pr:.2f}   "
          f"PCs for 95% / 99%: {np.searchsorted(cum, 0.95)+1} / {np.searchsorted(cum, 0.99)+1}")
    print("  (Descriptive only. Variance ≠ class information: between-subject gain")
    print("   differences inflate amplitude variance without helping classification.)")
    return rho, ev


# ─── Stage 2: supervised subset search ────────────────────────────────────────

def _fit_eval(Xtr, ytr, Xte, yte, int8=False):
    sc  = StandardScaler().fit(Xtr)
    Xtr_n, Xte_n = sc.transform(Xtr), sc.transform(Xte)
    lda = B.build_lda().fit(Xtr_n, ytr)
    acc_fp = accuracy_score(yte, lda.predict(Xte_n))
    if not int8:
        return acc_fp
    W, b = B.lda_as_linear_layer(lda)
    q = B.calibrate_ptq([(W, b)], Xtr_n)
    acc_q8 = accuracy_score(yte, lda.classes_[B.int8_infer(q, Xte_n)])
    return acc_fp, acc_q8

def _make_splits(groups):
    if INNER_CV == "loso":
        cv = LeaveOneGroupOut()
    else:
        cv = GroupKFold(n_splits=int(INNER_CV))
    return list(cv.split(np.zeros(len(groups)), groups=groups))

def _score_subset(S, X, y, splits):
    Xs = X[:, type_cols(S)]
    return np.array([_fit_eval(Xs[tr], y[tr], Xs[va], y[va]) for tr, va in splits])

def inner_search(X, y, groups):
    """Returns (n_subsets, n_inner_folds) matrix of inner-CV accuracies."""
    splits = _make_splits(groups)
    rows = Parallel(n_jobs=N_JOBS)(delayed(_score_subset)(S, X, y, splits) for S in SUBSETS)
    return np.vstack(rows)

def one_se_select(scores):
    """
    One-standard-error rule. All subsets are scored on the SAME inner folds, so
    the "paired" rule uses, for each subset S, the per-fold deficit
    d_S = acc_best − acc_S and its SE, sd(d_S)/√n. Subject difficulty is common
    to every subset and cancels in d_S; the "classic" rule's SE of the best
    subset's accuracy is dominated by it and is far too permissive here.
    Returns (pick, best, mean, se) where se is the SE the rule used.
    """
    n = scores.shape[1]
    mean = scores.mean(1)
    best = int(np.argmax(mean))
    if SELECTION_RULE == "paired":
        d  = scores[best][None, :] - scores
        se = d.std(1, ddof=1) / np.sqrt(n)
        ok = np.where(d.mean(1) <= SE_MULTIPLIER * se)[0]
    else:
        se = scores.std(1, ddof=1) / np.sqrt(n)
        ok = np.where(mean >= mean[best] - SE_MULTIPLIER * se[best])[0]
    kmin = min(len(SUBSETS[i]) for i in ok)
    pick = max((i for i in ok if len(SUBSETS[i]) == kmin), key=lambda i: mean[i])
    return pick, best, mean, se

def best_by_size(mean):
    sizes = sorted({len(S) for S in SUBSETS})
    return {k: max(mean[i] for i, S in enumerate(SUBSETS) if len(S) == k) for k in sizes}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 72)
    print("  FORS-EMG — nested-LOSO feature-type selection (LDA)")
    print("=" * 72)

    subjects = B.load_all_subjects(B.BASE_DIR)
    subjects = B.filter_all_subjects(subjects)
    X, y, sid, trial = build_feature_table(subjects)
    ids = [s["subject_id"] for s in subjects]
    print(f"\n  Windows: {len(y):,}   Subjects: {len(ids)}   "
          f"Candidate types: {[FEATURE_NAMES[i] for i in CANDIDATES]}   "
          f"Subsets searched: {len(SUBSETS)}")
    print(f"  Excluded a priori: {EXCLUDE_A_PRIORI} (IEMG = L·MAV ⇒ identical LDA)")
    print(f"  Inner CV: {INNER_CV}   Selection: {SE_MULTIPLIER:g}-SE rule ({SELECTION_RULE})   "
          f"Non-inferiority margin: {NI_MARGIN_PP:g} pp")

    print(f"\n{'─'*72}\n  Stage 1 — redundancy diagnostics (all windows, no labels)\n{'─'*72}")
    redundancy_diagnostics(X)

    print(f"\n{'─'*72}\n  Stage 2/3 — nested LOSO ({len(ids)} outer folds)\n{'─'*72}")
    in_train_trials = np.isin(trial, TRAIN_TRIALS)
    folds = []
    for o, t in enumerate(ids):
        tf = time.time()
        pool = (sid != t) & in_train_trials
        test = (sid == t)
        scores = inner_search(X[pool], y[pool], sid[pool])
        pick, best, mean, se = one_se_select(scores)
        S = SUBSETS[pick]

        sel_fp, sel_q8 = _fit_eval(X[pool][:, type_cols(S)], y[pool],
                                   X[test][:, type_cols(S)], y[test], int8=True)
        full_fp, full_q8 = _fit_eval(X[pool], y[pool], X[test], y[test], int8=True)
        folds.append(dict(subject=t, subset=S, inner_mean=mean[pick],
                          inner_best=mean[best], best_subset=SUBSETS[best],
                          sel_fp32=sel_fp, sel_int8=sel_q8,
                          full_fp32=full_fp, full_int8=full_q8,
                          curve=best_by_size(mean)))
        print(f"  [{o+1:2d}/{len(ids)}] S{t:02d}  selected {subset_name(S):32s} "
              f"test INT8 {sel_q8:.2%} (full {full_q8:.2%})  [{time.time()-tf:.0f}s]")

    # ── Per-fold table ───────────────────────────────────────────────────────
    print(f"\n{'─'*72}\n  Per-fold results (test = held-out subject, all trials)\n{'─'*72}")
    print(tabulate([[f"S{f['subject']:02d}", subset_name(f["subset"]), len(f["subset"]),
                     f"{f['inner_mean']:.2%}", f"{f['sel_fp32']:.2%}", f"{f['sel_int8']:.2%}",
                     f"{f['full_int8']:.2%}",
                     f"{(f['sel_int8']-f['full_int8'])*100:+.2f}"] for f in folds],
                   headers=["Test", "Selected types", "k", "Inner acc",
                            "Test FP32", "Test INT8", "Full INT8", "Δ pp"],
                   tablefmt="simple"))

    # ── Aggregate: reduced vs full, paired ───────────────────────────────────
    n = len(folds)
    def ci(v):
        v = np.asarray(v); return v.mean(), student_t.ppf(0.975, n-1) * v.std(ddof=1) / np.sqrt(n)
    sel = np.array([f["sel_int8"] for f in folds])
    full = np.array([f["full_int8"] for f in folds])
    sel_fp = np.array([f["sel_fp32"] for f in folds])
    full_fp = np.array([f["full_fp32"] for f in folds])
    d = sel - full
    m_sel, h_sel = ci(sel); m_full, h_full = ci(full); m_d, h_d = ci(d)
    try:
        p_w = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    except ValueError:
        p_w = 1.0
    print(f"\n  Nested test accuracy, INT8 (mean ± 95% CI over {n} subjects):")
    print(f"    Selected subset : {m_sel*100:.2f} ± {h_sel*100:.2f} %   "
          f"(FP32 {sel_fp.mean()*100:.2f} %)")
    print(f"    Full 8 types    : {m_full*100:.2f} ± {h_full*100:.2f} %   "
          f"(FP32 {full_fp.mean()*100:.2f} %)")
    print(f"    Paired Δ        : {m_d*100:+.2f} ± {h_d*100:.2f} pp   "
          f"(Wilcoxon signed-rank p = {p_w:.3f})")
    lo = (m_d - h_d) * 100
    verdict = "PASS" if lo > -NI_MARGIN_PP else "FAIL"
    print(f"    Non-inferiority : lower 95% bound of Δ = {lo:+.2f} pp vs margin "
          f"−{NI_MARGIN_PP:g} pp → {verdict}")

    # ── Stability and accuracy-vs-size curve ─────────────────────────────────
    freq = {FEATURE_NAMES[i]: sum(i in f["subset"] for f in folds) for i in CANDIDATES}
    print(f"\n  Selection frequency across {n} outer folds:")
    print(tabulate([[k, v, f"{v/n:.0%}"] for k, v in sorted(freq.items(), key=lambda kv: -kv[1])],
                   headers=["Type", "Folds", "Rate"], tablefmt="simple"))
    ks = sorted({len(S) for S in SUBSETS})
    curve = {k: np.mean([f["curve"][k] for f in folds]) for k in ks}
    print("\n  Best inner accuracy vs number of feature types (mean over outer folds):")
    print(tabulate([[k, f"{curve[k]:.2%}", f"{(curve[k]-curve[max(ks)])*100:+.2f}"] for k in ks],
                   headers=["k types", "Inner acc", f"vs k={max(ks)} (pp)"], tablefmt="simple"))

    # ── Stage 4: final subset on all subjects ────────────────────────────────
    print(f"\n{'─'*72}\n  Stage 4 — final selection on all {len(ids)} subjects\n{'─'*72}")
    pool = in_train_trials
    scores = inner_search(X[pool], y[pool], sid[pool])
    pick, best, mean, se = one_se_select(scores)
    S = SUBSETS[pick]
    order = np.argsort(-mean)[:10]
    if pick not in order:
        order = np.append(order, pick)
    print(tabulate([[subset_name(SUBSETS[i]), len(SUBSETS[i]),
                     f"{mean[i]:.2%}", f"{(mean[best]-mean[i])*100:.2f}", f"{se[i]*100:.2f}",
                     "◀ selected" if i == pick else ("best" if i == best else "")]
                    for i in order],
                   headers=["Top-10 subsets", "k", "Inner acc", "Deficit pp",
                            f"{SELECTION_RULE} SE pp", ""],
                   tablefmt="simple"))
    k = len(S)
    nb = B.N_CLASSES * B.N_CHANNELS * k + B.N_CLASSES * 4
    print(f"\n  ★ Final feature types : {subset_name(S)}  (k = {k} of 8)")
    print(f"    Input dim           : {B.N_CHANNELS * k}  (was 64)")
    full_bytes = B.N_CLASSES * B.N_CHANNELS * N_TYPES + B.N_CLASSES * 4
    full_macs  = B.N_CLASSES * B.N_CHANNELS * N_TYPES
    print(f"    LDA param bytes     : {nb} (was {full_bytes})   "
          f"MACs: {B.N_CLASSES * B.N_CHANNELS * k} (was {full_macs})")
    print(f"    Expected accuracy on a new subject (nested INT8): "
          f"{m_sel*100:.2f} ± {h_sel*100:.2f} %")
    if pick != best:
        print(f"    (Best-mean subset was {subset_name(SUBSETS[best])} at "
              f"{mean[best]:.2%}; selected is within {SE_MULTIPLIER:g} SE)")

    # ── CSV export ───────────────────────────────────────────────────────────
    with open("fors_emg_featsel_folds.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["TestSubject", "Selected", "k", "InnerAcc", "SelFP32", "SelINT8",
                    "FullFP32", "FullINT8"])
        for f in folds:
            w.writerow([f["subject"], subset_name(f["subset"]), len(f["subset"]),
                        round(f["inner_mean"], 6), round(f["sel_fp32"], 6), round(f["sel_int8"], 6),
                        round(f["full_fp32"], 6), round(f["full_int8"], 6)])
    with open("fors_emg_featsel_final_ranking.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Subset", "k", "InnerMean", "InnerSE"])
        for i in np.argsort(-mean):
            w.writerow([subset_name(SUBSETS[i]), len(SUBSETS[i]),
                        round(mean[i], 6), round(se[i], 6)])
    print("\n  Saved → fors_emg_featsel_folds.csv, fors_emg_featsel_final_ranking.csv")
    print(f"  Total runtime : {(time.time()-t0)/60:.1f} min")
    print("=" * 72)


if __name__ == "__main__":
    main()
