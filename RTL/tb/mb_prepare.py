"""Prepare a real FORS-EMG test case for the multi-board testbench (tb_mb.v).

  1. Trains the chip LDA (bit-exact features, shift normaliser) on every subject except
     TEST_SUBJECT, trials 1-4, exactly as golden_model/K_4_LDA_Golden_LOSO.py does.
  2. Turns it into the multi-board form: K-1 difference rows W'_c = W_c - W_0 and
     b'_c = b_c - b_0 (c = 1..3), quantised to INT8 / INT32 with one scale.
  3. Takes trial TEST_TRIAL of TEST_SUBJECT for the 4 gestures (one recording each), cuts
     the most active SEG samples of each and joins them into one 8-channel ADC stream.
  4. Runs the golden model on that stream and writes the expected decision of every stride.

Writes RTL/tb/mb_data/: samples.txt ($readmemh, one 64-bit word per sample, channel 0 in
the top byte), samples.csv (the same samples, readable), model.txt, expected.txt.
Run once from the repository root (takes a few minutes):  python RTL/tb/mb_prepare.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "golden_model"), str(ROOT / "analysis")]
import K_4_LDA_Golden_LOSO as GL                          # noqa: E402
from K_4_LDA_Golden_LOSO import B, G                      # noqa: E402

TEST_SUBJECT = 6
TEST_TRIAL = 1
SEG = 20 * G.SUB_WIN            # 1960 samples (about 2 s) per gesture
NDIFF = 3
OUT = Path(__file__).resolve().parent / "mb_data"
NAMES = ["Hand Close", "Hand Open", "Wrist Flexion", "Wrist Extension"]


def main():
    recs, _, full_scale = GL.load_recordings()
    chip, _, _, _ = GL.build_features(recs, full_scale)

    # ---- train on the other subjects (same as chip_fold)
    tr = (chip["sid"] != TEST_SUBJECT) & np.isin(chip["trial"], GL.TRAIN_TRIALS)
    mu, k = G.calibrate_norm(chip["raw"][tr])
    qtr = G.normalize(chip["raw"][tr], mu, k)
    lda = B.build_lda().fit(qtr, chip["y"][tr])
    Wf, bf = GL.linear_form(lda)
    assert list(lda.classes_) == [0, 1, 2, 3]

    # single-chip INT8 model (for comparison) and the K-1 difference form
    W6, b6 = G.quantize_lda(Wf, bf)
    Wd, bd = Wf[1:] - Wf[0], bf[1:] - bf[0]
    s = np.abs(Wd).max() / 127
    Wq = np.round(Wd / s).astype(np.int64)
    bq = np.round(bd / s).astype(np.int64)

    # ---- test stream: one recording of each gesture, trial TEST_TRIAL
    seg, lab = [], []
    for g in range(4):
        for sig, lbl, sid, trial in recs:
            if sid != TEST_SUBJECT or trial != TEST_TRIAL:
                continue
            _, yy = B.windows_from_recordings([sig], [lbl])
            if len(yy) and yy[0] == g:
                break
        x = GL.adc(sig, full_scale).astype(np.int64)
        act = np.convolve(np.abs(x).sum(1), np.ones(SEG), "valid")
        st = int(np.argmax(act))
        seg.append(x[st:st + SEG]); lab += [g] * SEG
        print(f"{NAMES[g]:16s} samples {st}..{st + SEG - 1} of {len(x)}")
    stream, lab = np.vstack(seg), np.array(lab)

    # ---- golden model on the stream
    ref = G.LdaCoreRef(8, mu, k, W6, b6)
    rows = []
    for n, smp in enumerate(stream):
        r = ref.step(smp)
        if r is None:
            continue
        q = r["q"]
        d = Wq @ q + bq
        cls, best = 0, 0
        for c in range(NDIFF):
            if d[c] > best:
                best, cls = d[c], c + 1
        win = lab[max(0, n - 195):n + 1]
        true = int(np.bincount(win).argmax())
        rows.append((len(rows), n, true, cls, r["cls"], int(len(set(win)) > 1)))

    OUT.mkdir(exist_ok=True)
    with open(OUT / "samples.txt", "w") as f:
        for smp in stream:
            f.write("".join(f"{int(v) & 0xFF:02x}" for v in smp) + "\n")
    with open(OUT / "samples.csv", "w") as f:
        f.write("n,label," + ",".join(f"ch{c}" for c in range(8)) + "\n")
        for n, (smp, l) in enumerate(zip(stream, lab)):
            f.write(f"{n},{l}," + ",".join(str(int(v)) for v in smp) + "\n")
    with open(OUT / "model.txt", "w") as f:
        f.write(f"# subject {TEST_SUBJECT} held out; trained on the other subjects, trials 1-4\n")
        f.write("# per feature f = 5*ch + t: mu k ; then W' rows (c = 1..3, 40 values) ; then b'\n")
        for f_ in range(40):
            f.write(f"mu {f_} {mu[f_]} {k[f_]}\n")
        for c in range(NDIFF):
            f.write(f"w {c} " + " ".join(str(v) for v in Wq[c]) + "\n")
        f.write("b " + " ".join(str(v) for v in bq) + "\n")
    with open(OUT / "expected.txt", "w") as f:
        f.write("# stride last_sample true_class expected_multi_board single_chip mixed_window\n")
        for r in rows:
            f.write(" ".join(str(v) for v in r) + "\n")

    full = [r for r in rows[1:] if not r[5]]
    print(f"{len(stream)} samples, {len(rows)} strides ({len(rows) - 1} decisions after the masked first)")
    print(f"accuracy on full single-gesture windows: multi-board {np.mean([r[3] == r[2] for r in full]) * 100:.1f}%, "
          f"single chip {np.mean([r[4] == r[2] for r in full]) * 100:.1f}%, "
          f"same decision {np.mean([r[3] == r[4] for r in rows[1:]]) * 100:.1f}%")


if __name__ == "__main__":
    main()
