"""
FORS-EMG — merging all 12 gestures into 4 classes vs dropping 8 of them (chip INT8 LDA, LOSO)
=============================================================================================
Two ways to get a 4-class problem out of the 12 FORS-EMG gestures:

  DROP   keep Hand Close, Hand Open, Wrist Flexion, Wrist Extension; the other 8 gestures are
         not used (the set the chip is designed for).
  MERGE  keep every gesture; group the 12 into 4 classes of 3 gestures each. The grouping is
         the one that is easiest to classify, chosen inside each fold from the training
         subjects only (see below), so the held-out subject never influences it.

Grouping, per outer fold:
  * 12-class LDA, inner 3-fold cross-validation over the training subjects -> 12 x 12
    confusion matrix (counts).
  * For each of the 12! / (3!^4 4!) = 15,400 ways to split 12 gestures into 4 groups of 3,
    the predicted 4-class accuracy is the share of windows whose prediction lands in the true
    group. The best split is used.

MERGE is evaluated two ways, both INT8 like the chip:
  merge-4   a 4-class LDA trained on the merged labels (same chip, 4 class rows).
  12->4     the 12-class LDA, its decision mapped to the group (12 class rows: 3x the MACs).
Also the earlier design's fixed 6-group map (G1 holds 6 gestures, half of all windows), as a 6-class
LDA on the merged labels and as the 12-class LDA mapped to the 6 groups.

Everything else is the chip pipeline of golden_model/K_4_LDA_Golden_LOSO.py: causal filter,
8-bit ADC (one full scale for all 12 gestures), streamed 196-sample windows with a 98-sample
stride, the 5 integer features, shift normaliser, INT8 LDA with one weight scale. Trials 1-4 of
the training subjects train; every trial of the held-out subject tests.
Prints results only; writes no files.

Run from the repository root:  python analysis/K_4_LDA_Merge_vs_Drop.py
"""

import itertools
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t, wilcoxon
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tabulate import tabulate

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "analysis"), str(_ROOT / "golden_model")]
import K_4_LDA_Analysis as B          # noqa: E402  (loader, causal filter, ADC quantiser, LDA settings)
import lda_golden_model as G          # noqa: E402  (chip features, shift normaliser, INT8 LDA)

GESTURES = ["Hand_Close", "Hand_Open", "Index", "Index_Little", "Peace", "Radial_Deviation",
            "Right_Angle", "Thumb_Little", "Thumb_UP", "Ulner_Deviation", "Wrist_Extension", "Wrist_Flexion"]
SHORT = ["HC", "HO", "Idx", "IdxL", "Peace", "RadD", "RAng", "ThL", "ThUp", "UlnD", "WE", "WF"]
DROP_SET = [0, 1, 11, 10]             # HC, HO, WF, WE -> classes 0..3
# the earlier design's fixed map: 12 gestures -> 6 groups (G1 = HC, Idx, IdxL, Peace, RAng, ThUp; G2 = HO, WE;
# G3 = RadD; G4 = ThL; G5 = UlnD; G6 = WF)
MAP6 = np.array([0, 1, 0, 0, 0, 2, 0, 3, 0, 4, 1, 5])
TRAIN_TRIALS = (1, 2, 3, 4)
C = B.N_CHANNELS
INNER_FOLDS = 3


# ---------------------------------------------------------------- data

def load_features():
    """Chip features of every full window of every recording of all 12 gestures."""
    filt = B._design_filters()
    per_subject, full_scale = [], 0.0
    for sid in range(1, B.N_SUBJECTS + 1):
        recs, meta = B._load_single_subject(B.BASE_DIR, sid, classes=list(range(12)),
                                            gesture_file_map=dict(enumerate(GESTURES)))
        if not recs:
            print(f"  Subject{sid}: no recordings, skipped")
            continue
        sigs = [B.apply_preprocessing_filter(r.T, filt, "causal").T.astype(np.float32) for r in recs]
        full_scale = max(full_scale, max(float(np.abs(s).max()) for s in sigs))
        per_subject.append((sid, sigs, meta))
        print(f"  Subject{sid}: {len(recs)} recordings", flush=True)
    raw, y, sid, trial = [], [], [], []
    for s, sigs, meta in per_subject:
        for sig, m in zip(sigs, meta):
            q8 = np.asarray(B.quantize_signal(sig.T[None], full_scale, bits=8))[0].T.astype(np.int8)
            r, full = G.window_features(q8, C)
            n = int(full.sum())
            raw.append(r[full]); y.append(np.full(n, m["label"]))
            sid.append(np.full(n, s)); trial.append(np.full(n, m["trial"]))
        sigs.clear()
    return np.vstack(raw), np.concatenate(y), np.concatenate(sid), np.concatenate(trial)


# ---------------------------------------------------------------- models

def int8_lda(q_tr, y_tr):
    """Chip LDA: float fit, linear form, INT8 weights with one scale (G.quantize_lda allows <= 6 classes)."""
    lda = LinearDiscriminantAnalysis(solver=B.LDA_SOLVER, shrinkage=B.LDA_SHRINKAGE).fit(q_tr, y_tr)
    W, b = (np.asarray(a, float) for a in B.lda_as_linear_layer(lda))
    b = b.ravel()
    W = W if W.shape[0] == len(b) else W.T
    s = np.abs(W).max() / 127
    return lda.classes_, np.round(W / s).astype(np.int64), np.round(b / s).astype(np.int64)


def predict(model, q):
    classes, W, b = model
    return classes[np.argmax(q @ W.T + b, axis=1)]   # first max = strict '>' on the chip


# ---------------------------------------------------------------- grouping search

def partitions_of_3():
    """All splits of 0..11 into 4 unordered groups of 3 (15,400)."""
    out = []
    def rec(left, groups):
        if not left:
            out.append(tuple(groups)); return
        first = left[0]
        for pair in itertools.combinations(left[1:], 2):
            g = (first,) + pair
            rec([v for v in left if v not in g], groups + [g])
    rec(list(range(12)), [])
    return out


PARTS = partitions_of_3()
TRIPLES = sorted({g for p in PARTS for g in p})
T_IDX = {g: i for i, g in enumerate(TRIPLES)}
P_IDX = np.array([[T_IDX[g] for g in p] for p in PARTS])          # 15,400 x 4


def best_grouping(cm):
    """Split maximising the share of windows predicted inside their own group."""
    within = np.array([cm[np.ix_(g, g)].sum() for g in TRIPLES])
    score = within[P_IDX].sum(axis=1) / cm.sum()
    i = int(np.argmax(score))
    return PARTS[i], float(score[i])


def inner_confusion(q, y, sid):
    """12-class LDA, cross-validated over training subjects -> confusion counts."""
    subs = np.unique(sid)
    folds = np.array_split(np.random.default_rng(0).permutation(subs), INNER_FOLDS)
    cm = np.zeros((12, 12))
    for f in folds:
        te = np.isin(sid, f)
        lda = LinearDiscriminantAnalysis(solver=B.LDA_SOLVER, shrinkage=B.LDA_SHRINKAGE).fit(q[~te], y[~te])
        np.add.at(cm, (y[te], lda.predict(q[te])), 1)
    return cm


# ---------------------------------------------------------------- statistics

def mean_ci(v):
    v = np.asarray(v, float) * 100
    return v.mean(), student_t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))


def paired(a, b):
    d = (np.asarray(a) - np.asarray(b)) * 100
    m = d.mean(); h = student_t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    verdict = "better" if m - h > 0 else "worse" if m + h < 0 else "no detectable difference"
    return [f"{m:+.2f}", f"[{m - h:+.2f}, {m + h:+.2f}]", f"{p:.3g}", f"{(d > 0).sum()}/{(d < 0).sum()}", verdict]


def name(groups):
    return "  ".join("{" + ", ".join(SHORT[i] for i in g) + "}" for g in sorted(groups))


# ---------------------------------------------------------------- main

def main():
    t0 = time.time()
    print("=" * 92)
    print("  FORS-EMG — merge all 12 gestures into 4 classes vs drop to 4 gestures (chip INT8 LDA, LOSO)")
    print("=" * 92)
    raw, y, sid, trial = load_features()
    ids = [int(s) for s in np.unique(sid)]
    print(f"\n  {len(y):,} windows, {len(ids)} subjects, 12 gestures, {raw.shape[1]} features"
          f"   ({time.time() - t0:.0f} s)\n")

    acc = {"drop": [], "merge-4": [], "12->4": [], "12-class": [], "6-group": [], "12->6": []}
    chosen = []
    for i, t in enumerate(ids):
        tr = (sid != t) & np.isin(trial, TRAIN_TRIALS)
        te = sid == t
        mu, k = G.calibrate_norm(raw[tr])                 # normaliser from all 12 gestures of the training subjects
        q = G.normalize(raw, mu, k)

        # DROP: only the 4 chosen gestures, normaliser from those gestures
        dmap = {g: c for c, g in enumerate(DROP_SET)}
        dtr, dte = tr & np.isin(y, DROP_SET), te & np.isin(y, DROP_SET)
        mu_d, k_d = G.calibrate_norm(raw[dtr])
        qd = G.normalize(raw, mu_d, k_d)
        yd = np.array([dmap.get(int(v), -1) for v in y])
        acc["drop"].append(np.mean(predict(int8_lda(qd[dtr], yd[dtr]), qd[dte]) == yd[dte]))

        # MERGE: grouping from the training subjects only
        groups, pred_score = best_grouping(inner_confusion(q[tr], y[tr], sid[tr]))
        chosen.append(groups)
        gmap = np.zeros(12, int)
        for c, g in enumerate(groups):
            gmap[list(g)] = c
        ym = gmap[y]
        acc["merge-4"].append(np.mean(predict(int8_lda(q[tr], ym[tr]), q[te]) == ym[te]))
        p12 = predict(int8_lda(q[tr], y[tr]), q[te])
        acc["12->4"].append(np.mean(gmap[p12] == ym[te]))
        acc["12-class"].append(np.mean(p12 == y[te]))
        y6 = MAP6[y]
        acc["6-group"].append(np.mean(predict(int8_lda(q[tr], y6[tr]), q[te]) == y6[te]))
        acc["12->6"].append(np.mean(MAP6[p12] == y6[te]))
        print(f"  [{i + 1:2d}/{len(ids)}] S{t:02d}  drop {acc['drop'][-1]:.2%}  merge-4 {acc['merge-4'][-1]:.2%}"
              f"  12->4 {acc['12->4'][-1]:.2%}  12-class {acc['12-class'][-1]:.2%}   "
              f"groups {name(groups)} (inner {pred_score:.1%})", flush=True)

    print(f"\n{'─' * 92}\n  Accuracy on the held-out subject (mean over {len(ids)} subjects, ± 95% CI)\n{'─' * 92}")
    rows = [["drop: HC, HO, WF, WE", "2/3 of the gestures unused", "4", "160"],
            ["merge-4: 12 gestures in 4 groups", "nothing", "4", "160"],
            ["12->4: 12-class LDA, mapped to groups", "nothing", "12", "480"],
            ["12-class LDA (reference)", "nothing", "12", "480"],
            ["6-group map: 6-class LDA (earlier design)", "nothing", "6", "240"],
            ["6-group map: 12-class LDA, mapped", "nothing", "12", "480"]]
    for r, key in zip(rows, ["drop", "merge-4", "12->4", "12-class", "6-group", "12->6"]):
        m, h = mean_ci(acc[key])
        r.insert(1, f"{m:.2f} ± {h:.2f}")
    print(tabulate(rows, headers=["Model", "Test INT8 %", "Data dropped", "Class rows", "MACs"], tablefmt="simple"))
    print()
    print(tabulate([["merge-4 vs drop"] + paired(acc["merge-4"], acc["drop"]),
                    ["12->4 vs drop"] + paired(acc["12->4"], acc["drop"]),
                    ["12->4 vs merge-4"] + paired(acc["12->4"], acc["merge-4"]),
                    ["6-group vs drop"] + paired(acc["6-group"], acc["drop"]),
                    ["12->6 vs drop"] + paired(acc["12->6"], acc["drop"])],
                   headers=["Comparison", "Δ pp", "95% CI", "Wilcoxon p", "Up/Down", "Verdict"], tablefmt="simple"))

    print(f"  (6-group map: G1 holds {np.mean(MAP6[y] == 0):.0%} of all windows, so always answering G1 scores that much)")
    print(f"\n  Groupings chosen across the {len(ids)} folds:")
    counts = {}
    for g in chosen:
        key = name(g); counts[key] = counts.get(key, 0) + 1
    for key, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"    {n:2d} x  {key}")
    print(f"\n  Total runtime : {(time.time() - t0) / 60:.1f} min")
    print("=" * 92)


if __name__ == "__main__":
    main()
