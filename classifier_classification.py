"""
FORS-EMG Classifier Comparison — Pareto Frontier for INT8 Deployment
=====================================================================
Compares linear models, tree-based models, and the reference MLP on the
selected 4-gesture set, under the identical LOSO protocol.

Why this comparison
-------------------
The stated design objective is minimising MACs under a 10 KB budget. Tree
classifiers use ZERO MACs — pure compare-and-branch — so they occupy a
categorically different point on the hardware Pareto frontier, not merely a
"simpler model". Linear models (LogReg / LinearSVC / LDA) share the 64→4
hypothesis class and cost 256 MACs.

Models evaluated
----------------
  LDA                closed-form, 256 MACs, no training loop
  Logistic Regression 256 MACs
  Linear SVM         256 MACs
  Decision Tree      0 MACs, ~depth comparisons
  Random Forest      0 MACs, n_trees x depth comparisons
  Extra Trees        0 MACs
  HistGradientBoost  0 MACs
  MLP 64->16->4      1,088 MACs  (reference — the 79.38% result)

Protocol (identical to fors_emg_benchmark.py, so numbers are comparable)
-----------------------------------------------------------------------
* Gestures: Hand Close, Hand Open, Wrist Flexion, Wrist Extension
* 20-450 Hz bandpass + 50 Hz notch, applied once per recording
* 200 ms windows / 100 ms increment, cut AFTER the LOSO split
* Trial-level validation split (trial 5 held out), StandardScaler on train only
* Symmetric per-tensor INT8 quantisation:
    - linear models + MLP: INT8 weights, INT32 bias, integer requantisation
    - trees: INT8 input features and INT8-grid thresholds

Speed
-----
* Features computed ONCE per subject and cached (the naive version recomputes
  them for 18 subjects on each of 19 folds — ~19x redundant work).
* LOSO folds run in parallel across all cores via joblib.
* Ensembles use n_jobs=1 inside a fold (outer parallelism already saturates
  the cores; nesting would oversubscribe).

Usage
-----
  pip install joblib
  python classifier_comparison.py
"""

import os, sys, time, warnings, csv, copy
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, iirnotch
from scipy import stats

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from joblib import Parallel, delayed
from tqdm import tqdm
from tabulate import tabulate

warnings.filterwarnings("ignore")
torch.manual_seed(0)
np.random.seed(0)
torch.set_num_threads(1)          # outer joblib parallelism owns the cores

# ─── Configuration ────────────────────────────────────────────────────────────
BASE_DIR = r"/Users/vinayhariharan/Projects/BTP/NN_model/archive/FORS-EMG Dataset/FORS-EMG Dataset/FORS-EMG"

CLASSES = [0, 1, 2, 3]
GESTURE_FILE_MAP = {0: "Hand_Close", 1: "Hand_Open",
                    2: "Wrist_Flexion", 3: "Wrist_Extension"}
GESTURE_NAMES = {0: "Hand Close", 1: "Hand Open",
                 2: "Wrist Flexion", 3: "Wrist Extension"}
FOREARM_ORIENTATIONS = ["Rest", "Supination", "Pronation"]
N_TRIALS, N_SUBJECTS, VAL_TRIAL = 5, 19, 5

FS          = 985
WIN_SAMPLES = int(200 * FS / 1000)
INC_SAMPLES = int(100 * FS / 1000)
N_CHANNELS, N_FEATURES = 8, 8
N_CLASSES  = len(CLASSES)
INPUT_DIM  = N_CHANNELS * N_FEATURES

APPLY_FILTER   = True
BANDPASS_HZ    = (20.0, 450.0)
MAINS_NOTCH_HZ = 50.0
NOTCH_Q        = 30.0

INT8_MAX, INT8_MIN, CALIB_PCT = 127, -128, 99.9

# MLP reference
HIDDEN = 16
LR, WEIGHT_DECAY, DROPOUT = 1e-3, 1e-4, 0.2
BATCH_SIZE, MAX_EPOCHS, PATIENCE = 128, 200, 15

# Tree hyper-parameters — depth-capped to stay inside the 10 KB budget
TREE_DEPTH      = 10
RF_N_TREES      = 25
RF_DEPTH        = 8
GBT_MAX_ITER    = 60
GBT_DEPTH       = 4

N_JOBS = int(os.environ.get("N_JOBS", -1))   # -1 = all cores


# ─── Loading ──────────────────────────────────────────────────────────────────

def load_subject(base_dir, sid):
    recs, meta = [], []
    sdir = os.path.join(base_dir, f"Subject{sid}")
    if not os.path.isdir(sdir):
        return recs, meta
    for orientation in FOREARM_ORIENTATIONS:
        odir = os.path.join(sdir, orientation)
        for cls in CLASSES:
            prefix = GESTURE_FILE_MAP[cls]
            for trial in range(1, N_TRIALS + 1):
                fp = os.path.join(odir, f"{prefix}-{trial}.mat")
                if not os.path.exists(fp):
                    continue
                mat = sio.loadmat(fp)
                for k, v in mat.items():
                    if k.startswith("_") or not isinstance(v, np.ndarray) or v.ndim != 2:
                        continue
                    if v.shape == (8000, 8):
                        data = v.astype(np.float64)
                    elif v.shape == (8, 8000):
                        data = v.T.astype(np.float64)
                    else:
                        continue
                    recs.append(data)
                    meta.append({"label": cls, "subject": sid,
                                 "orientation": orientation, "trial": trial})
                    break
    return recs, meta


def _filters():
    nyq = FS / 2.0
    b_bp, a_bp = butter(4, [BANDPASS_HZ[0]/nyq,
                            min(BANDPASS_HZ[1], nyq-1.0)/nyq], btype="bandpass")
    b_n, a_n = iirnotch(MAINS_NOTCH_HZ, NOTCH_Q, FS)
    return (b_bp, a_bp), (b_n, a_n)


def extract_features(w):
    N, C, L = w.shape
    x, dx, xabs = w, np.diff(w, axis=2), np.abs(w)
    mav  = xabs.mean(axis=2)
    rms  = np.sqrt((x**2).mean(axis=2))
    wl   = np.abs(dx).sum(axis=2)
    thr  = (rms * 0.01)[:, :, None]
    sx   = np.sign(x)
    zc   = ((sx[:, :, 1:] != sx[:, :, :-1]) &
            ((xabs[:, :, :-1] >= thr) | (xabs[:, :, 1:] >= thr))).sum(axis=2)
    ssc  = ((dx[:, :, 1:] * dx[:, :, :-1]) < 0).sum(axis=2)
    var  = x.var(axis=2)
    wamp = (np.abs(dx) > (rms*0.02)[:, :, None]).sum(axis=2)
    iemg = xabs.sum(axis=2)
    return np.stack([mav, rms, wl, zc, ssc, var, wamp, iemg],
                    axis=2).reshape(N, C*8).astype(np.float32)


def _windows(recs, labels):
    W, Y = [], []
    for rec, lb in zip(recs, labels):
        sig = rec.T
        for s in np.arange(0, sig.shape[1] - WIN_SAMPLES + 1, INC_SAMPLES):
            W.append(sig[:, s:s+WIN_SAMPLES]); Y.append(lb)
    if not W:
        return (np.empty((0, N_CHANNELS, WIN_SAMPLES), np.float32),
                np.empty((0,), np.int64))
    return np.stack(W).astype(np.float32), np.array(Y, np.int64)


def build_feature_cache(base_dir):
    """
    Loads, filters, windows and featurises every subject ONCE.
    Returns a list of dicts: {X (n,64), y (n,), trial (n,)} per subject.
    This is the main speed lever — features are reused across all 19 folds
    instead of being recomputed 18 times each.
    """
    coeffs = _filters() if APPLY_FILTER else None
    cache = []
    for sid in tqdm(range(1, N_SUBJECTS+1), desc="Load+filter+featurise"):
        recs, meta = load_subject(base_dir, sid)
        if not recs:
            continue
        if APPLY_FILTER:
            (b_bp, a_bp), (b_n, a_n) = coeffs
            recs = [filtfilt(b_n, a_n,
                             filtfilt(b_bp, a_bp, r.T, axis=-1), axis=-1).T
                    for r in recs]
        labels = [m["label"] for m in meta]
        trials = [m["trial"] for m in meta]
        W, Y = _windows(recs, labels)
        T = []
        for r, t in zip(recs, trials):
            n_w = len(np.arange(0, r.shape[0]-WIN_SAMPLES+1, INC_SAMPLES))
            T.extend([t]*n_w)
        cache.append({"subject": sid, "X": extract_features(W),
                      "y": Y, "trial": np.array(T)})
    return cache


# ─── INT8 quantisation ────────────────────────────────────────────────────────

def _scale(a):
    v = np.percentile(np.abs(np.asarray(a).ravel()), CALIB_PCT)
    return float(v)/INT8_MAX if v > 0 else 1.0


def quantise_linear_predict(W, b, X_calib, X_test):
    """
    Integer-only inference for a linear model (logits = W x + b).
    Symmetric per-tensor INT8 weights, INT32 bias in the accumulator domain.
    argmax on INT32 logits — no requantisation needed on the final layer.
    """
    sx, sw = _scale(X_calib), _scale(W)
    Wq = np.clip(np.round(W/sw), INT8_MIN, INT8_MAX).astype(np.int32)
    bq = np.round(np.asarray(b)/(sw*sx)).astype(np.int32)
    xq = np.clip(np.round(X_test/sx), INT8_MIN, INT8_MAX).astype(np.int32)
    return np.argmax(xq @ Wq.T + bq[None, :], axis=1)


def quantise_tree_predict(model, X_calib, X_test):
    """
    Simulates INT8 deployment of a tree model:
      * input features quantised to the INT8 grid
      * split thresholds snapped to the same grid
    Both are then de-quantised so sklearn's traversal reproduces exactly what
    integer compare-and-branch would do on hardware.
    """
    sx = _scale(X_calib)
    Xq = np.clip(np.round(X_test/sx), INT8_MIN, INT8_MAX) * sx

    m = copy.deepcopy(model)
    ests = []
    if hasattr(m, "estimators_"):
        for e in np.asarray(m.estimators_).ravel():
            ests.append(e)
    else:
        ests = [m]
    for e in ests:
        if hasattr(e, "tree_"):
            t = e.tree_.threshold
            valid = t != -2                      # -2 marks leaves
            t[valid] = np.clip(np.round(t[valid]/sx), INT8_MIN, INT8_MAX) * sx
    return m.predict(Xq)


def quantise_histgbt_predict(model, X_calib, X_test):
    """HistGBT bins internally; quantising the input is the honest simulation."""
    sx = _scale(X_calib)
    Xq = np.clip(np.round(X_test/sx), INT8_MIN, INT8_MAX) * sx
    return model.predict(Xq)


# ─── MLP ──────────────────────────────────────────────────────────────────────

def build_mlp(hidden, n_classes=N_CLASSES):
    if hidden is None:
        return nn.Sequential(nn.Linear(INPUT_DIM, n_classes))
    return nn.Sequential(nn.Linear(INPUT_DIM, hidden), nn.ReLU(),
                         nn.Dropout(DROPOUT), nn.Linear(hidden, n_classes))


def train_mlp(X_tr, y_tr, X_va, y_va, hidden):
    model = build_mlp(hidden)
    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, MAX_EPOCHS)
    loader = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                                      torch.tensor(y_tr, dtype=torch.long)),
                        batch_size=BATCH_SIZE, shuffle=True)
    Xv = torch.tensor(X_va, dtype=torch.float32)
    yv = torch.tensor(y_va, dtype=torch.long)
    best, state, wait = float("inf"), None, 0
    for _ in range(MAX_EPOCHS):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            v = crit(model(Xv), yv).item()
        if v < best - 1e-4:
            best, wait = v, 0
            state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(state)
    return model


REQUANT_SHIFT = 31   # fixed-point precision for the requantisation multiplier


def _requant_multiplier(M):
    """
    Decompose the requantisation scale M = sw*sx/sy into (M0, shift) so that
    y = (z * M0) >> shift  reproduces  round(M * z)  using integer ops only.

    A fixed non-negative shift is used deliberately: deriving the shift from
    ceil(-log2(M)) breaks whenever M > 1 (the shift goes negative, and a
    negative right-shift is undefined). A fixed shift is valid for any M > 0
    and is what a hardware implementation would hard-wire anyway.
    """
    if M <= 0:
        return 0, REQUANT_SHIFT
    return int(np.round(M * (1 << REQUANT_SHIFT))), REQUANT_SHIFT


def _apply_requant(z, M0, shift):
    """Round-to-nearest integer multiply-and-shift, computed in int64."""
    z64 = z.astype(np.int64) * np.int64(M0)
    return (z64 + (np.int64(1) << (shift - 1))) >> shift


def mlp_int8_predict(model, X_calib, X_test):
    model.eval()
    lins = [m for _, m in model.named_modules() if isinstance(m, nn.Linear)]
    sx, x_fp, info = _scale(X_calib), X_calib.copy(), []
    for i, lin in enumerate(lins):
        W, b = lin.weight.detach().numpy(), lin.bias.detach().numpy()
        sw = _scale(W); z = x_fp @ W.T + b
        last = (i == len(lins)-1)
        sy = _scale(z if last else np.maximum(0, z))
        if not last:
            x_fp = np.maximum(0, z)
        M0, shift = _requant_multiplier((sw*sx)/sy)
        info.append((np.clip(np.round(W/sw), INT8_MIN, INT8_MAX).astype(np.int32),
                     np.round(b/(sw*sx)).astype(np.int32), M0, shift, last))
        sx = sy
    xq = np.clip(np.round(X_test/_scale(X_calib)),
                 INT8_MIN, INT8_MAX).astype(np.int32)
    for Wq, bq, M0, shift, last in info:
        z = xq.astype(np.int64) @ Wq.T.astype(np.int64) + bq[None, :].astype(np.int64)
        if last:
            # argmax on the raw INT32 accumulator — requantisation is a
            # monotone positive scaling, so it cannot change the argmax.
            return np.argmax(z, axis=1)
        xq = np.maximum(0, np.clip(_apply_requant(z, M0, shift),
                                   INT8_MIN, INT8_MAX)).astype(np.int32)
    return np.argmax(z, axis=1)


# ─── Hardware cost accounting ─────────────────────────────────────────────────

def count_tree_nodes(model):
    if hasattr(model, "estimators_"):
        return int(sum(e.tree_.node_count
                       for e in np.asarray(model.estimators_).ravel()
                       if hasattr(e, "tree_")))
    if hasattr(model, "tree_"):
        return int(model.tree_.node_count)
    return 0


def tree_cost(model):
    """
    Bytes per node: feature index (1 B) + INT8 threshold (1 B) + child index
    (2 B) + leaf class (1 B, unioned with threshold on leaves) ~= 5 B.
    MACs = 0; the runtime cost is comparisons ~ depth per tree.
    """
    n_nodes = count_tree_nodes(model)
    depth = getattr(model, "max_depth", None) or 0
    n_trees = 1
    if hasattr(model, "estimators_"):
        n_trees = len(np.asarray(model.estimators_).ravel())
    return {"bytes": n_nodes*5, "macs": 0,
            "comparisons": int(depth*n_trees) if depth else n_nodes,
            "nodes": n_nodes}


LINEAR_COST = {"bytes": INPUT_DIM*N_CLASSES + N_CLASSES*4,
               "macs": INPUT_DIM*N_CLASSES, "comparisons": 0,
               "nodes": 0}
MLP_COST    = {"bytes": INPUT_DIM*HIDDEN + HIDDEN*4 + HIDDEN*N_CLASSES + N_CLASSES*4,
               "macs": INPUT_DIM*HIDDEN + HIDDEN*N_CLASSES, "comparisons": 0,
               "nodes": 0}


# ─── One LOSO fold — trains every model ───────────────────────────────────────

def run_fold(cache, test_idx):
    """
    Executes one LOSO fold for ALL classifiers. Returns
    {model_name: {"fp32": acc, "int8": acc, "cost": {...},
                  "y_true": arr, "y_pred": arr}}
    Designed to be called in parallel across folds.
    """
    tr_X, tr_y, va_X, va_y = [], [], [], []
    for i, c in enumerate(cache):
        if i == test_idx:
            continue
        m_va = c["trial"] == VAL_TRIAL
        tr_X.append(c["X"][~m_va]); tr_y.append(c["y"][~m_va])
        va_X.append(c["X"][m_va]);  va_y.append(c["y"][m_va])
    tr_X = np.concatenate(tr_X); tr_y = np.concatenate(tr_y)
    va_X = np.concatenate(va_X); va_y = np.concatenate(va_y)
    te_X, te_y = cache[test_idx]["X"], cache[test_idx]["y"]

    sc = StandardScaler()
    tr = sc.fit_transform(tr_X).astype(np.float32)
    va = sc.transform(va_X).astype(np.float32)
    te = sc.transform(te_X).astype(np.float32)

    out = {}

    # ── Linear models: FP32 accuracy + INT8 integer inference ────────────────
    linear_specs = [
        ("LDA",                 LinearDiscriminantAnalysis()),
        ("Logistic Regression", LogisticRegression(max_iter=1000, n_jobs=1)),
        ("Linear SVM",          LinearSVC(max_iter=3000, dual="auto")),
    ]
    for name, clf in linear_specs:
        clf.fit(tr, tr_y)
        p32 = clf.predict(te)
        W = clf.coef_
        b = clf.intercept_
        if W.shape[0] == 1:            # binary form — not expected with 4 classes
            W = np.vstack([-W, W]); b = np.array([-b[0], b[0]])
        p8 = quantise_linear_predict(W, b, tr, te)
        out[name] = {"fp32": accuracy_score(te_y, p32),
                     "int8": accuracy_score(te_y, p8),
                     "cost": LINEAR_COST, "y_true": te_y, "y_pred": p8}

    # ── Tree models: 0 MACs ──────────────────────────────────────────────────
    tree_specs = [
        ("Decision Tree", DecisionTreeClassifier(max_depth=TREE_DEPTH,
                                                 random_state=0)),
        ("Random Forest", RandomForestClassifier(n_estimators=RF_N_TREES,
                                                 max_depth=RF_DEPTH,
                                                 random_state=0, n_jobs=1)),
        ("Extra Trees",   ExtraTreesClassifier(n_estimators=RF_N_TREES,
                                               max_depth=RF_DEPTH,
                                               random_state=0, n_jobs=1)),
    ]
    for name, clf in tree_specs:
        clf.fit(tr, tr_y)
        p32 = clf.predict(te)
        p8  = quantise_tree_predict(clf, tr, te)
        out[name] = {"fp32": accuracy_score(te_y, p32),
                     "int8": accuracy_score(te_y, p8),
                     "cost": tree_cost(clf), "y_true": te_y, "y_pred": p8}

    # HistGBT (separate: different internal structure)
    gbt = HistGradientBoostingClassifier(max_iter=GBT_MAX_ITER,
                                         max_depth=GBT_DEPTH,
                                         early_stopping=False, random_state=0)
    gbt.fit(tr, tr_y)
    p32 = gbt.predict(te)
    p8  = quantise_histgbt_predict(gbt, tr, te)
    n_nodes = int(sum(p.get_max_depth() if hasattr(p, "get_max_depth") else 0
                      for pred in gbt._predictors for p in pred))
    est_nodes = GBT_MAX_ITER * N_CLASSES * (2**GBT_DEPTH)
    out["Hist Gradient Boosting"] = {
        "fp32": accuracy_score(te_y, p32), "int8": accuracy_score(te_y, p8),
        "cost": {"bytes": est_nodes*5, "macs": 0,
                 "comparisons": GBT_MAX_ITER*N_CLASSES*GBT_DEPTH,
                 "nodes": est_nodes},
        "y_true": te_y, "y_pred": p8}

    # ── MLP reference ────────────────────────────────────────────────────────
    mlp = train_mlp(tr, tr_y, va, va_y, HIDDEN)
    mlp.eval()
    with torch.no_grad():
        p32 = mlp(torch.tensor(te, dtype=torch.float32)).argmax(1).numpy()
    p8 = mlp_int8_predict(mlp, tr, te)
    out[f"MLP 64-{HIDDEN}-4"] = {"fp32": accuracy_score(te_y, p32),
                                 "int8": accuracy_score(te_y, p8),
                                 "cost": MLP_COST, "y_true": te_y, "y_pred": p8}
    return out


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    n_cores = os.cpu_count() or 1
    jobs = n_cores if N_JOBS == -1 else N_JOBS
    print("="*74)
    print("  FORS-EMG Classifier Comparison — INT8 Hardware Pareto Frontier")
    print("="*74)
    print(f"  Gestures : {', '.join(GESTURE_NAMES[c] for c in CLASSES)}")
    print(f"  Cores    : {n_cores} detected, using {jobs} parallel workers")

    cache = build_feature_cache(BASE_DIR)
    if not cache:
        print(f"\n  ERROR: no data loaded from {BASE_DIR}")
        sys.exit(1)
    n_sub = len(cache)
    total_win = sum(len(c["y"]) for c in cache)
    print(f"\n  Subjects : {n_sub}")
    print(f"  Windows  : {total_win:,}")
    print(f"  Features cached once per subject "
          f"(saves ~{n_sub-1}x redundant recomputation)")

    print(f"\n{'─'*74}")
    print(f"  Running {n_sub}-fold LOSO x 8 classifiers, parallel over folds")
    print(f"{'─'*74}")
    fold_results = Parallel(n_jobs=jobs, verbose=10)(
        delayed(run_fold)(cache, i) for i in range(n_sub))

    # ── Aggregate ────────────────────────────────────────────────────────────
    names = list(fold_results[0].keys())
    agg = {}
    for nm in names:
        fp32 = np.array([f[nm]["fp32"] for f in fold_results])
        int8 = np.array([f[nm]["int8"] for f in fold_results])
        yt = np.concatenate([f[nm]["y_true"] for f in fold_results])
        yp = np.concatenate([f[nm]["y_pred"] for f in fold_results])
        cost = fold_results[0][nm]["cost"]
        agg[nm] = {"fp32": fp32, "int8": int8, "cost": cost,
                   "y_true": yt, "y_pred": yp}

    # ── Results table ────────────────────────────────────────────────────────
    print(f"\n{'─'*74}")
    print("  RESULTS  (mean ± std over subjects)")
    print(f"{'─'*74}")
    order = sorted(names, key=lambda n: -agg[n]["int8"].mean())
    rows = []
    for nm in order:
        a = agg[nm]; c = a["cost"]
        rows.append([nm,
                     f"{a['fp32'].mean()*100:.2f}±{a['fp32'].std()*100:.2f}",
                     f"{a['int8'].mean()*100:.2f}±{a['int8'].std()*100:.2f}",
                     f"{(a['fp32'].mean()-a['int8'].mean())*100:+.2f}",
                     f"{c['macs']:,}",
                     f"{c['comparisons']:,}" if c['comparisons'] else "—",
                     f"{c['bytes']:,}",
                     "✓" if c['bytes'] <= 10_000 else "✗"])
    print(tabulate(rows, headers=["Classifier", "FP32 %", "INT8 %", "Q-drop",
                                  "MACs", "Cmps", "Bytes", "≤10KB"],
                   tablefmt="rounded_outline"))

    # ── Pareto frontier ──────────────────────────────────────────────────────
    print(f"\n{'─'*74}")
    print("  PARETO FRONTIER  (INT8 accuracy vs. MACs; ties broken by bytes)")
    print(f"{'─'*74}")
    pts = [(nm, agg[nm]["int8"].mean(), agg[nm]["cost"]["macs"],
            agg[nm]["cost"]["bytes"]) for nm in names
           if agg[nm]["cost"]["bytes"] <= 10_000]
    pareto = []
    for nm, acc, macs, byts in sorted(pts, key=lambda p: (p[2], p[3])):
        if not any(a >= acc and (m < macs or (m == macs and b < byts))
                   for _, a, m, b in pts):
            pareto.append((nm, acc, macs, byts))
    seen, frontier = set(), []
    for nm, acc, macs, byts in pareto:
        if nm not in seen:
            seen.add(nm); frontier.append([nm, f"{acc*100:.2f}", f"{macs:,}", f"{byts:,}"])
    print(tabulate(frontier, headers=["Classifier", "INT8 %", "MACs", "Bytes"],
                   tablefmt="rounded_outline"))
    print("  (A model is on the frontier if nothing beats it on accuracy at")
    print("   equal-or-lower cost. Zero-MAC entries are the multiplier-free options.)")

    # ── Paired significance vs. the MLP reference ────────────────────────────
    ref = f"MLP 64-{HIDDEN}-4"
    print(f"\n{'─'*74}")
    print(f"  PAIRED TESTS vs. {ref}  (same {n_sub} subjects)")
    print(f"{'─'*74}")
    prows = []
    for nm in order:
        if nm == ref:
            continue
        d = agg[nm]["int8"] - agg[ref]["int8"]
        t, p = stats.ttest_rel(agg[nm]["int8"], agg[ref]["int8"])
        ci = stats.t.ppf(0.975, len(d)-1)*d.std(ddof=1)/np.sqrt(len(d))
        verdict = ("worse"  if p < 0.05 and d.mean() < 0 else
                   "BETTER" if p < 0.05 and d.mean() > 0 else "tied")
        prows.append([nm, f"{d.mean()*100:+.2f}",
                      f"[{(d.mean()-ci)*100:+.2f}, {(d.mean()+ci)*100:+.2f}]",
                      f"{p:.4f}", verdict])
    print(tabulate(prows, headers=["Classifier", "Δ vs MLP (pp)", "95% CI",
                                   "p", "Verdict"],
                   tablefmt="rounded_outline"))
    print("  'tied' = no statistically detectable difference at p<0.05.")

    # ── Best model detail ────────────────────────────────────────────────────
    best = order[0]
    print(f"\n  ★  Highest INT8 accuracy: {best}  "
          f"({agg[best]['int8'].mean()*100:.2f}% ± {agg[best]['int8'].std()*100:.2f}%)")
    print(f"\n  Per-class breakdown ({best}):")
    print(classification_report(agg[best]["y_true"], agg[best]["y_pred"],
                                target_names=[GESTURE_NAMES[c] for c in CLASSES],
                                digits=3))
    cm = confusion_matrix(agg[best]["y_true"], agg[best]["y_pred"])
    print("  Confusion matrix (rows=true, cols=pred):")
    print(tabulate([[GESTURE_NAMES[c]] + list(cm[i]) for i, c in enumerate(CLASSES)],
                   headers=["True \\ Pred"] + [GESTURE_NAMES[c] for c in CLASSES],
                   tablefmt="simple"))

    # ── Per-subject table ────────────────────────────────────────────────────
    print(f"\n  Per-subject INT8 accuracy (all classifiers):")
    srows = [[f"S{cache[i]['subject']:02d}"] +
             [f"{agg[nm]['int8'][i]*100:.1f}" for nm in order]
             for i in range(n_sub)]
    srows.append(["MEAN"] + [f"{agg[nm]['int8'].mean()*100:.1f}" for nm in order])
    print(tabulate(srows, headers=["Subj"] + [n[:11] for n in order],
                   tablefmt="simple"))

    # ── Save ─────────────────────────────────────────────────────────────────
    with open("classifier_comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["classifier", "fp32_mean", "fp32_std", "int8_mean",
                    "int8_std", "qdrop_pp", "macs", "comparisons", "bytes"])
        for nm in order:
            a = agg[nm]; c = a["cost"]
            w.writerow([nm, round(a["fp32"].mean(), 6), round(a["fp32"].std(), 6),
                        round(a["int8"].mean(), 6), round(a["int8"].std(), 6),
                        round((a["fp32"].mean()-a["int8"].mean())*100, 4),
                        c["macs"], c["comparisons"], c["bytes"]])
    with open("per_subject_accuracy.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject"] + order)
        for i in range(n_sub):
            w.writerow([cache[i]["subject"]] +
                       [round(agg[nm]["int8"][i], 6) for nm in order])

    print(f"\n  Saved → classifier_comparison.csv, per_subject_accuracy.csv")
    print(f"  Total runtime : {(time.time()-t0)/60:.1f} min")
    print("="*74)


if __name__ == "__main__":
    main()
