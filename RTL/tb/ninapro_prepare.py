"""Prepare a NinaPro DB2 test case for the multi-board testbench (tb_mb.v), like mb_prepare.py does for FORS-EMG.

NinaPro DB2, exercise B (files S<n>_E1_A1.mat): 12 Delsys electrodes at 2 kHz. All 12 are used by
default (columns 0-7: ring around the upper forearm, 8-9: finger flexor / extensor, 10-11: biceps /
triceps). A board takes at most 8, so 12 electrodes need 2-3 boards: run_mb.py 8,4 | 6,6 | 4,4,4.
--channels 8 keeps only the ring (fits one board).
Gestures (DB2 exercise-B labels, the same mapping as the Calibration checkpoint, Appendix A.2):
     6 fingers flexed together in fist   -> Hand Close
     8 adduction of extended fingers     -> Hand Open   (fingers together, extended)
    13 wrist flexion                     -> Wrist Flexion
    14 wrist extension                   -> Wrist Extension

  1. Resample 2000 Hz -> 985 Hz (x 197/400), the chip's sample rate, then the repo's causal
     20-450 Hz band-pass + 50 Hz notch.
  2. Cut every repetition of the 4 gestures (restimulus runs) into its own recording.
  3. 8-bit ADC with one fixed gain from the training subjects; chip features and shift
     normaliser from lda_golden_model.py; LDA from K_4_LDA_Analysis.py; multi-board
     K-1 difference form W'_c = W_c - W_0, b'_c = b_c - b_0, INT8 / INT32, as mb_prepare.py.
  3b. --calib N: method C of the Calibration checkpoint (re-centre on the user, blend the general
     per-gesture summaries with N of the user's repetitions, share chosen on a validation repetition).
  4. Leave-one-subject-out accuracy of exactly that integer decision for every subject
     (--loso), and a test stream for TEST_SUBJECT: the most active 2 s of each gesture in
     repetition TEST_REP, joined into one stream, with the golden decision of every stride.

Writes RTL/tb/mb_data/ (same files as mb_prepare.py, so tb/run_mb.py runs unchanged).
This overwrites the committed FORS-EMG case; `git checkout RTL/tb/mb_data` brings it back.

    python RTL/tb/ninapro_prepare.py --data /path/to/ninapro_db2 --test 1 [--loso]
    python RTL/tb/ninapro_prepare.py --data /path/to/ninapro_db2 --test 1 --loso --calib 1   # method C
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "golden_model"), str(ROOT / "analysis")]
import K_4_LDA_Analysis as B                              # noqa: E402
import lda_golden_model as G                              # noqa: E402

GESTURES = {6: "Hand Close", 8: "Hand Open", 13: "Wrist Flexion", 14: "Wrist Extension"}
CHANNELS = list(range(12))      # DB2 electrodes used (set by --channels)
FS_IN, UP, DOWN = 2000, 197, 400  # 2000 * 197 / 400 = 985 Hz
SEG = 20 * G.SUB_WIN            # 1960 samples (about 2 s) per gesture in the test stream
NDIFF = len(GESTURES) - 1
FS_PCT = 99.99                  # ADC full scale = this percentile of |signal| (training subjects)
OUT = Path(__file__).resolve().parent / "mb_data"


def load_subject(path, filt):
    """One S<n>_E1_A1.mat -> list of (filtered 985 Hz signal N x 8, class 0..3, repetition)."""
    m = sio.loadmat(path)
    emg = m["emg"][:, CHANNELS].astype(np.float64)
    stim = m["restimulus"].ravel().astype(int)
    rep = m["rerepetition"].ravel().astype(int)
    sig = resample_poly(emg, UP, DOWN, axis=0)
    # labels resampled by nearest sample
    idx = np.minimum(np.round(np.arange(len(sig)) * DOWN / UP).astype(int), len(stim) - 1)
    stim, rep = stim[idx], rep[idx]
    sig = B.apply_preprocessing_filter(sig.T, filt, "causal").T     # filter the whole session, as the device would
    out = []
    edges = np.flatnonzero(np.diff(np.r_[0, stim, 0]) != 0)
    for s, e in zip(edges[:-1], edges[1:]):
        g = stim[s]
        if g in GESTURES and e - s >= 3 * G.SUB_WIN:
            out.append((sig[s:e], list(GESTURES).index(g), int(np.bincount(rep[s:e]).argmax())))
    return out


def load_all(data_dir):
    filt = B._design_filters()                    # fs = 985 Hz
    subj = {}
    for p in sorted(Path(data_dir).rglob("S*_E1_A1.mat")):
        sid = int(re.match(r"S(\d+)_E1_A1", p.name).group(1))
        subj[sid] = load_subject(p, filt)
        n = np.bincount([r[1] for r in subj[sid]], minlength=len(GESTURES))
        print(f"subject {sid:2d}: {len(subj[sid])} repetitions, per gesture {n.tolist()}")
    assert subj, f"no S*_E1_A1.mat found under {data_dir}"
    return subj


def features(subj, sids, full_scale):
    R, Y = [], []
    for sid in sids:
        for sig, g, _ in subj[sid]:
            raw, full = G.window_features(G.to_adc_int8(sig, full_scale), len(CHANNELS))
            R.append(raw[full]); Y += [g] * int(full.sum())
    return np.vstack(R), np.array(Y)


def full_scale_of(subj, sids):
    """ADC gain: one fixed full scale from the training subjects (a hardware constant)."""
    return float(np.percentile(np.abs(np.vstack([r[0] for s in sids for r in subj[s]])), FS_PCT))


def pack(Wf, bf, mu, k, full_scale):
    """Float LDA (K x D, K) -> chip numbers: single-chip INT8 rows and the multi-board K-1 difference form."""
    W6, b6 = G.quantize_lda(Wf, bf)                          # single-chip INT8 model
    Wd, bd = Wf[1:] - Wf[0], bf[1:] - bf[0]                  # multi-board K-1 difference form
    s = np.abs(Wd).max() / 127
    return dict(full_scale=full_scale, mu=mu, k=k, W6=W6, b6=b6,
                Wq=np.round(Wd / s).astype(np.int64), bq=np.round(bd / s).astype(np.int64))


def train(subj, test_sid):
    """Chip model trained on every subject except test_sid (no calibration)."""
    tr = [s for s in subj if s != test_sid]
    full_scale = full_scale_of(subj, tr)
    raw, y = features(subj, tr, full_scale)
    mu, k = G.calibrate_norm(raw)
    lda = B.build_lda().fit(G.normalize(raw, mu, k), y)
    Wf, bf = (np.asarray(a, float) for a in B.lda_as_linear_layer(lda))
    assert list(lda.classes_) == list(range(len(GESTURES)))
    return pack(Wf, bf, mu, k, full_scale)


# ---------------------------------------------------------------- calibration: method C of the Calibration checkpoint
#
#   1. Re-centre: every person's features are measured from their own centre (the mean of their
#      per-gesture means), so the chip's input offset mu becomes the user's own centre mu'.
#      The shift k stays the general one, so the general and the user's numbers live in one space.
#   2. Summaries per gesture (count, sum, sum of products) of the re-centred INT8 inputs q:
#      general = the other subjects, user = the calibration repetitions.
#   3. Blend with the share s (0 = general only, 1 = user only): class means and the pooled
#      within-gesture covariance are mixed (1-s) * general + s * user.
#   4. LDA formula (equal priors, small shrinkage) -> INT8 / K-1 difference form, as without calibration.
#   The chip is unchanged: calibration only writes new mu', W', b'.

SHRINK = 0.1                    # covariance shrinkage towards its mean variance (user data is small)
SHARES = (0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0)


def person_feats(subj, sid, full_scale):
    """Full windows of one subject: raw sums, gesture, repetition."""
    R, Y, P = [], [], []
    for sig, g, rep in subj[sid]:
        raw, full = G.window_features(G.to_adc_int8(sig, full_scale), len(CHANNELS))
        n = int(full.sum())
        R.append(raw[full]); Y += [g] * n; P += [rep] * n
    return np.vstack(R), np.array(Y), np.array(P)


def centre(raw, y):
    """Re-centring offset: mean of the per-gesture means (every gesture counts equally)."""
    return np.round(np.mean([raw[y == c].mean(0) for c in np.unique(y)], axis=0)).astype(np.int64)


def summary(q, y):
    K = len(GESTURES)
    q = q.astype(np.float64)
    return dict(n=np.array([(y == c).sum() for c in range(K)], float),
                S=np.array([q[y == c].sum(0) for c in range(K)]),
                P=np.array([q[y == c].T @ q[y == c] for c in range(K)]))


def add(a, b):
    return {key: a[key] + b[key] for key in a}


def moments(sm):
    m = sm["S"] / sm["n"][:, None]
    scatter = sum(sm["P"][c] - sm["n"][c] * np.outer(m[c], m[c]) for c in range(len(m)))
    return m, scatter / sm["n"].sum()


def lda_blend(gen, user, share):
    """Method C blend of two summaries -> float LDA (W, b), equal priors."""
    mg, Cg = moments(gen)
    if user is None or share == 0:
        m, C = mg, Cg
    else:
        mu_, Cu = moments(user)
        m, C = (1 - share) * mg + share * mu_, (1 - share) * Cg + share * Cu
    C = (1 - SHRINK) * C + SHRINK * np.trace(C) / len(C) * np.eye(len(C))
    W = np.linalg.solve(C, m.T).T
    b = -0.5 * np.einsum("kd,kd->k", W, m)
    return W, b


class Fold:
    """Everything for one held-out user that does not depend on the calibration repetitions."""

    def __init__(self, subj, user):
        self.tr = [s for s in subj if s != user]
        self.full_scale = full_scale_of(subj, self.tr)
        F = {s: person_feats(subj, s, self.full_scale) for s in subj}
        cen = {s: centre(F[s][0], F[s][1]) for s in self.tr}
        _, self.k = G.calibrate_norm(np.vstack([F[s][0] - cen[s] for s in self.tr]))
        self.gen = None
        for s in self.tr:
            sm = summary(G.normalize(F[s][0], cen[s], self.k), F[s][1])
            self.gen = sm if self.gen is None else add(self.gen, sm)
        # the uncalibrated model of train(), on the same features
        raw = np.vstack([F[s][0] for s in self.tr]); y = np.concatenate([F[s][1] for s in self.tr])
        mu0, k0 = G.calibrate_norm(raw)
        lda = B.build_lda().fit(G.normalize(raw, mu0, k0), y)
        self.base = pack(*(np.asarray(a, float) for a in B.lda_as_linear_layer(lda)), mu0, k0, self.full_scale)
        self.raw, self.y, self.rep = F[user]

    def calibrated(self, calib_reps, share):
        sel = np.isin(self.rep, calib_reps)
        mu_u = centre(self.raw[sel], self.y[sel])
        user = summary(G.normalize(self.raw[sel], mu_u, self.k), self.y[sel])
        return pack(*lda_blend(self.gen, user, share), mu_u, self.k, self.full_scale)

    def acc(self, m, reps):
        sel = np.isin(self.rep, reps)
        return np.mean(decide(G.normalize(self.raw[sel], m["mu"], m["k"]), m["Wq"], m["bq"]) == self.y[sel]) * 100


def calib_eval(subj, n_cal):
    """Rotating repetitions per user (as the checkpoint's DB2 Part 2): 1 test, 1 validation, n_cal calibration.
    One share for everyone, chosen on the validation repetitions; reported on the test repetitions."""
    val = {s: [] for s in SHARES}; test = {s: [] for s in SHARES}; base = []
    users = sorted(subj)
    for u in users:
        f = Fold(subj, u)
        reps = sorted(set(f.rep.tolist()))
        assert len(reps) >= n_cal + 2, f"subject {u}: {len(reps)} repetitions, need {n_cal + 2}"
        v_u = {s: [] for s in SHARES}; t_u = {s: [] for s in SHARES}; b_u = []
        for r in range(len(reps)):
            t, v = reps[r], reps[(r + 1) % len(reps)]
            cal = [reps[(r + 2 + i) % len(reps)] for i in range(n_cal)]
            b_u.append(f.acc(f.base, [t]))
            for s in SHARES:
                m = f.calibrated(cal, s)
                v_u[s].append(f.acc(m, [v])); t_u[s].append(f.acc(m, [t]))
        base.append(np.mean(b_u))
        for s in SHARES:
            val[s].append(np.mean(v_u[s])); test[s].append(np.mean(t_u[s]))
        print(f"user {u:2d}: no calibration {base[-1]:5.1f} %   C, {n_cal} rep: "
              + "  ".join(f"s={s:.1f} {test[s][-1]:5.1f}" for s in SHARES))
    best = max(SHARES, key=lambda s: np.mean(val[s]))
    a, b = np.array(test[best]), np.array(base)
    ci = lambda x: 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
    print(f"\nshare chosen on the validation repetitions: {best}  "
          f"(validation mean per share: " + ", ".join(f"{s}: {np.mean(val[s]):.1f}" for s in SHARES) + ")")
    print(f"no calibration (general model, same test repetitions): {b.mean():.2f} % +/- {ci(b):.2f}")
    print(f"method C, {n_cal} calibration repetition(s), share {best}:  {a.mean():.2f} % +/- {ci(a):.2f}")
    print(f"gain: {(a - b).mean():+.2f} points +/- {ci(a - b):.2f} (95% CI), "
          f"{int((a > b).sum())} of {len(a)} users improve, worst change {(a - b).min():+.1f}")
    return best


def decide(q, Wq, bq):
    """Head-board decision: argmax{0, D_1..D_K-1}, strict '>' (first maximum wins)."""
    d = q @ Wq.T + bq
    assert np.abs(d).max() <= G.INT32_MAX
    return np.where(d.max(1) > 0, d.argmax(1) + 1, 0)


def loso(subj):
    accs = []
    for t in sorted(subj):
        m = train(subj, t)
        raw, y = features(subj, [t], m["full_scale"])
        acc = np.mean(decide(G.normalize(raw, m["mu"], m["k"]), m["Wq"], m["bq"]) == y) * 100
        accs.append(acc)
        print(f"LOSO subject {t:2d}: {acc:5.1f} %")
    a = np.array(accs)
    print(f"LOSO mean {a.mean():.2f} % +/- {1.96 * a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0:.2f} "
          f"(95% CI, {len(a)} subjects), chance {100 / len(GESTURES):.0f} %")


def write_case(subj, test_sid, test_rep, n_cal=0, share=0.8):
    if n_cal:
        f = Fold(subj, test_sid)
        reps = sorted(set(f.rep.tolist()))
        assert test_rep in reps, f"subject {test_sid} has no repetition {test_rep}"
        others = [r for r in reps if r != test_rep]
        cal = others[:n_cal]
        m = f.calibrated(cal, share)
        print(f"calibrated (method C) on repetitions {cal} of subject {test_sid}, share {share}; "
              f"accuracy on repetition {test_rep}: {f.acc(m, [test_rep]):.1f} % "
              f"(no calibration {f.acc(f.base, [test_rep]):.1f} %)")
        note = f"calibrated (method C, reps {cal}, share {share})"
    else:
        m = train(subj, test_sid)
        note = "no calibration"
    names = list(GESTURES.values())
    seg, lab = [], []
    for g in range(len(GESTURES)):
        recs = [r for r in subj[test_sid] if r[1] == g and r[2] == test_rep]
        assert recs, f"subject {test_sid} has no repetition of {names[g]}"
        x = G.to_adc_int8(recs[0][0], m["full_scale"]).astype(np.int64)
        n = min(SEG, len(x))
        st = int(np.argmax(np.convolve(np.abs(x).sum(1), np.ones(n), "valid")))
        seg.append(x[st:st + n]); lab += [g] * n
        print(f"{names[g]:16s} samples {st}..{st + n - 1} of {len(x)} (repetition {recs[0][2]})")
    stream, lab = np.vstack(seg), np.array(lab)

    ref = G.LdaCoreRef(len(CHANNELS), m["mu"], m["k"], m["W6"], m["b6"])
    rows = []
    for n, smp in enumerate(stream):
        r = ref.step(smp)
        if r is None:
            continue
        cls = int(decide(r["q"][None], m["Wq"], m["bq"])[0])
        win = lab[max(0, n - 195):n + 1]
        rows.append((len(rows), n, int(np.bincount(win).argmax()), cls, r["cls"], int(len(set(win)) > 1)))

    D = 5 * len(CHANNELS)
    OUT.mkdir(exist_ok=True)
    with open(OUT / "samples.txt", "w") as f:
        for smp in stream:
            f.write("".join(f"{int(v) & 0xFF:02x}" for v in smp) + "\n")
    with open(OUT / "samples.csv", "w") as f:
        f.write("n,label," + ",".join(f"ch{c}" for c in range(len(CHANNELS))) + "\n")
        for n, (smp, l) in enumerate(zip(stream, lab)):
            f.write(f"{n},{l}," + ",".join(str(int(v)) for v in smp) + "\n")
    with open(OUT / "model.txt", "w") as f:
        f.write(f"# NinaPro DB2: subject {test_sid} held out; trained on the other subjects; {note}\n")
        f.write(f"# per feature f = 5*ch + t: mu k ; then W' rows (c = 1..{NDIFF}, {D} values) ; then b'\n")
        for i in range(D):
            f.write(f"mu {i} {m['mu'][i]} {m['k'][i]}\n")
        for c in range(NDIFF):
            f.write(f"w {c} " + " ".join(str(v) for v in m["Wq"][c]) + "\n")
        f.write("b " + " ".join(str(v) for v in m["bq"]) + "\n")
    with open(OUT / "expected.txt", "w") as f:
        f.write("# stride last_sample true_class expected_multi_board single_chip mixed_window\n")
        for r in rows:
            f.write(" ".join(str(v) for v in r) + "\n")

    full = [r for r in rows[1:] if not r[5]]
    print(f"{len(stream)} samples, {len(rows)} strides ({len(rows) - 1} decisions after the masked first)")
    print(f"accuracy on full single-gesture windows: multi-board {np.mean([r[3] == r[2] for r in full]) * 100:.1f}%, "
          f"single chip {np.mean([r[4] == r[2] for r in full]) * 100:.1f}%, "
          f"same decision {np.mean([r[3] == r[4] for r in rows[1:]]) * 100:.1f}%")
    split = "8" if len(CHANNELS) <= 8 else "8,4" if len(CHANNELS) == 12 else f"8,{len(CHANNELS) - 8}"
    print(f"wrote {OUT}/  ->  now: cd RTL && python3 tb/run_mb.py {split}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="folder containing the S<n>_E1_A1.mat files (searched recursively)")
    ap.add_argument("--test", type=int, default=1, help="held-out subject for the RTL test stream")
    ap.add_argument("--rep", type=int, default=1, help="repetition of the held-out subject used in the stream")
    ap.add_argument("--loso", action="store_true", help="also print leave-one-subject-out accuracy for every subject")
    ap.add_argument("--calib", type=int, default=0, metavar="N",
                    help="method C calibration with N repetitions of the user (1-3); 0 = none")
    ap.add_argument("--share", type=float, default=None,
                    help="blend share for the test stream (default: the one --loso chose, else 0.8)")
    ap.add_argument("--channels", type=int, default=12, choices=range(1, 25), metavar="N",
                    help="use DB2 electrodes 0..N-1 (default 12 = all; 8 = the ring only)")
    a = ap.parse_args()
    CHANNELS[:] = list(range(a.channels))
    subj = load_all(a.data)
    assert a.test in subj, f"subject {a.test} not found"
    share = a.share
    if a.loso and a.calib:
        best = calib_eval(subj, a.calib)
        share = best if share is None else share
    elif a.loso:
        loso(subj)
    write_case(subj, a.test, a.rep, a.calib, 0.8 if share is None else share)


if __name__ == "__main__":
    main()
