"""
FORS-EMG — global INT8 LDA + post-training calibration on one subject
=====================================================================
1. Global LDA: hardware features (SABS, WL, ZC_NODB, SSC, POW_M1) + shift
   normaliser, trained on every subject except TARGET (trials 1-4) — the same
   fold and features as the LDA reference in K_4_CNN_LOSO.py.
2. Calibration (adaptive LDA): class means and the shared covariance are
   blended with TARGET's own statistics from 1 trial and from 2 trials,
       mu_c  <- (1 - LAMBDA) mu_c(global) + LAMBDA mu_c(target)
       Sigma <- (1 - LAMBDA) Sigma(global) + LAMBDA Sigma(target)
   and the linear layer W, b is recomputed in closed form.
3. All three (global, 1-trial, 2-trial) are tested on the same TARGET trials,
   none of which is used for calibration; FP32 and INT8 accuracies printed.

Requires K_4_LDA_Analysis.py (revision 3). Prints results only; writes no files.

Run:  python K_4_LDA_Calib.py
"""

import copy, time
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import t as student_t
from sklearn.metrics import accuracy_score
from tabulate import tabulate

import K_4_LDA_Analysis as B

# ─── Configuration ────────────────────────────────────────────────────────────
TARGET       = 15
TRAIN_TRIALS = (1, 2, 3, 4)
CALIB_SETS   = {"1 trial": (1,), "2 trials": (1, 2)}
LAMBDA       = 0.5        # 0 = global model unchanged, 1 = target statistics only
HW_TYPES     = ["SABS", "WL", "ZC_NODB", "SSC", "POW_M1"]
CHUNK        = 4000


# ─── Data (all CPU cores) ─────────────────────────────────────────────────────

def _subject_windows(s, filt):
    W, y, tr, peak = [], [], [], 0.0
    for rec, lbl, m in zip(s["recordings"], s["labels"], s["meta"]):
        sig = B.apply_preprocessing_filter(rec.T, filt, "causal").T
        peak = max(peak, float(np.abs(sig).max()))
        w, yy = B.windows_from_recordings([sig], [lbl])
        W.append(w.astype(np.float32)); y.append(yy)
        tr.append(np.full(len(yy), m["trial"]))
    y = np.concatenate(y)
    return np.concatenate(W), y, np.full(len(y), s["subject_id"]), np.concatenate(tr), peak


def _features(Wc, full_scale):
    return B.extract_hw_features(B.quantize_signal(Wc, full_scale), HW_TYPES)


def build_features(subjects):
    filt = B._design_filters()
    out = Parallel(n_jobs=-1)(delayed(_subject_windows)(s, filt) for s in subjects)
    W, y, sid, trial = (np.concatenate(z) for z in list(zip(*out))[:4])
    full_scale = max(o[4] for o in out)
    X = np.vstack(Parallel(n_jobs=-1)(delayed(_features)(W[i:i + CHUNK], full_scale)
                                      for i in range(0, len(W), CHUNK)))
    return X, y, sid, trial


# ─── LDA calibration ──────────────────────────────────────────────────────────

def class_stats(X, y, classes):
    """Class means and pooled within-class covariance (sklearn's definition)."""
    mu = np.stack([X[y == c].mean(0) for c in classes])
    R = X - mu[np.searchsorted(classes, y)]
    return mu, R.T @ R / len(X)


def calibrate(lda, Sigma_g, Xt, yt, lam):
    mu_t, Sigma_t = class_stats(Xt, yt, lda.classes_)
    mu = (1 - lam) * lda.means_ + lam * mu_t
    Sigma = (1 - lam) * Sigma_g + lam * Sigma_t
    m = copy.deepcopy(lda)
    m.coef_ = np.linalg.solve(Sigma, mu.T).T                       # W = mu Σ⁻¹
    m.intercept_ = -0.5 * np.sum(m.coef_ * mu, 1) + np.log(lda.priors_)
    return m


def scores(lda, Xtr_n, Xte_n, yte):
    W, b = B.lda_as_linear_layer(lda)
    q = B.calibrate_ptq([(W, b)], Xtr_n, input_scale=1.0)
    return (accuracy_score(yte, lda.predict(Xte_n)),
            accuracy_score(yte, lda.classes_[B.int8_infer(q, Xte_n)]))


def loso_fold(X, y, sid, trial, t):
    """Global LDA with subject t held out; tested on all of t's trials."""
    tr = (sid != t) & np.isin(trial, TRAIN_TRIALS)
    sc = B.ShiftNormalizer()
    Xtr_n = sc.fit_transform(X[tr])
    lda = B.build_lda().fit(Xtr_n, y[tr])
    te = sid == t
    return scores(lda, Xtr_n, sc.transform(X[te]), y[te])


def mean_ci(v):
    v = np.asarray(v) * 100
    return f"{v.mean():.2f} ± {student_t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)):.2f}"


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    X, y, sid, trial = build_features(B.load_all_subjects(B.BASE_DIR))
    t_feat = time.time() - t0

    # Global LDA, before calibration: full LOSO over every subject
    ids = np.unique(sid)
    acc = np.array(Parallel(n_jobs=-1)(delayed(loso_fold)(X, y, sid, trial, t) for t in ids))
    s = int(np.where(ids == TARGET)[0][0])
    print(f"\nGlobal LDA before calibration (LOSO, each subject tested on all its trials)")
    print(tabulate([[f"Mean over {len(ids)} subjects (± 95% CI)", mean_ci(acc[:, 0]), mean_ci(acc[:, 1])],
                    [f"S{TARGET:02d}", f"{acc[s, 0]*100:.2f}", f"{acc[s, 1]*100:.2f}"]],
                   headers=["", "Test FP32 %", "Test INT8 %"], tablefmt="simple"))
    print("\n" + tabulate([[f"S{t:02d}", f"{a[1]*100:.2f}"] for t, a in zip(ids, acc)],
                          headers=["Subject", "INT8 %"], tablefmt="simple"))

    idx_tr = np.where((sid != TARGET) & np.isin(trial, TRAIN_TRIALS))[0]
    used = sorted({t for ts in CALIB_SETS.values() for t in ts})
    idx_te = np.where((sid == TARGET) & ~np.isin(trial, used))[0]

    sc = B.ShiftNormalizer()
    Xtr_n, Xn = sc.fit_transform(X[idx_tr]), sc.transform(X)
    lda = B.build_lda().fit(Xtr_n, y[idx_tr])
    Sigma_g = getattr(lda, "covariance_", None)
    if Sigma_g is None:
        Sigma_g = class_stats(Xtr_n, y[idx_tr], lda.classes_)[1]

    print(f"\nCalibration on S{TARGET:02d} | λ = {LAMBDA} | "
          f"test {len(idx_te):,} windows (trials {sorted(set(trial[idx_te]))})")

    rows = [["Global (no calibration)", "-", *(f"{a*100:.2f}" for a in
                                              scores(lda, Xtr_n, Xn[idx_te], y[idx_te]))]]
    for n, ts in CALIB_SETS.items():
        ic = np.where((sid == TARGET) & np.isin(trial, ts))[0]
        m = calibrate(lda, Sigma_g, Xn[ic], y[ic], LAMBDA)
        rows.append([f"Calibrated, {n}", f"{len(ic):,}",
                     *(f"{a*100:.2f}" for a in scores(m, Xtr_n, Xn[idx_te], y[idx_te]))])

    print("\n" + tabulate(rows, headers=["Model", "Calib windows", "Test FP32 %",
                                         "Test INT8 %"], tablefmt="simple"))
    print(f"\nFeatures {t_feat:.1f} s | total {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
