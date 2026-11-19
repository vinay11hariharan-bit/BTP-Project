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
  4. Leave-one-subject-out accuracy of exactly that integer decision for every subject
     (--loso), and a test stream for TEST_SUBJECT: the most active 2 s of each gesture in
     repetition TEST_REP, joined into one stream, with the golden decision of every stride.

Writes RTL/tb/mb_data/ (same files as mb_prepare.py, so tb/run_mb.py runs unchanged).
This overwrites the committed FORS-EMG case; `git checkout RTL/tb/mb_data` brings it back.

    python RTL/tb/ninapro_prepare.py --data /path/to/ninapro_db2 --test 1 [--loso]
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


def train(subj, test_sid):
    """Chip model trained on every subject except test_sid."""
    tr = [s for s in subj if s != test_sid]
    full_scale = float(np.percentile(np.abs(np.vstack([r[0] for s in tr for r in subj[s]])), FS_PCT))
    raw, y = features(subj, tr, full_scale)
    mu, k = G.calibrate_norm(raw)
    lda = B.build_lda().fit(G.normalize(raw, mu, k), y)
    Wf, bf = (np.asarray(a, float) for a in B.lda_as_linear_layer(lda))
    assert list(lda.classes_) == list(range(len(GESTURES)))
    W6, b6 = G.quantize_lda(Wf, bf)                          # single-chip INT8 model
    Wd, bd = Wf[1:] - Wf[0], bf[1:] - bf[0]                  # multi-board K-1 difference form
    s = np.abs(Wd).max() / 127
    return dict(full_scale=full_scale, mu=mu, k=k, W6=W6, b6=b6,
                Wq=np.round(Wd / s).astype(np.int64), bq=np.round(bd / s).astype(np.int64))


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


def write_case(subj, test_sid, test_rep):
    m = train(subj, test_sid)
    names = list(GESTURES.values())
    seg, lab = [], []
    for g in range(len(GESTURES)):
        recs = [r for r in subj[test_sid] if r[1] == g and r[2] == test_rep] or \
               [r for r in subj[test_sid] if r[1] == g]
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
        f.write(f"# NinaPro DB2: subject {test_sid} held out; trained on the other subjects\n")
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
    ap.add_argument("--channels", type=int, default=12, choices=range(1, 25), metavar="N",
                    help="use DB2 electrodes 0..N-1 (default 12 = all; 8 = the ring only)")
    a = ap.parse_args()
    CHANNELS[:] = list(range(a.channels))
    subj = load_all(a.data)
    assert a.test in subj, f"subject {a.test} not found"
    if a.loso:
        loso(subj)
    write_case(subj, a.test, a.rep)


if __name__ == "__main__":
    main()
