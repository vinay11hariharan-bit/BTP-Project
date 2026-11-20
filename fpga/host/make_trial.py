"""Make one FORS-EMG trial for the FPGA (runs on the laptop, needs the dataset).

    python fpga/host/make_trial.py 6 1          # subject 6, trial 1

Same steps as RTL/tb/mb_prepare.py, but with the whole recordings:
  1. trains the chip LDA on every other subject (trials 1-4), K-1 difference form, INT8
  2. the test stream is trial T of subject S: one recording of each of the 4 gestures,
     8000 samples each, joined into one 8-channel 8-bit ADC stream (32000 samples)
  3. the golden model gives the expected decision of every stride
  4. the config writes for the 4 + 4 split (as RTL/tb/run_mb.py makes them)
Saves fpga/host/trials/trial_s<S>_t<T>.npz; copy it to the PYNQ next to run_trial.py.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "golden_model"), str(ROOT / "analysis")]
import K_4_LDA_Golden_LOSO as GL                          # noqa: E402
from K_4_LDA_Golden_LOSO import B, G                      # noqa: E402

NAMES = ["Hand Close", "Hand Open", "Wrist Flexion", "Wrist Extension"]
SPLIT = (4, 4)
STRIDE = 98
MAX_SAMPLES = 32768                                       # sample memory of the FPGA


def config_writes(mu, k, Wq, bq):
    """(board, addr, data) for every write; bias on the head only, head core_en last."""
    nd, w, ch0 = len(Wq), [], 0
    for brd, c in enumerate(SPLIT):
        d = 5 * c
        w += [(brd, 0x7F0, c), (brd, 0x7F2, nd)]
        w += [(brd, f, (int(mu[5 * ch0 + f]) << 5) | int(k[5 * ch0 + f])) for f in range(d)]
        for r in range(nd):
            w += [(brd, 0x100 + r * d + f, int(Wq[r][5 * ch0 + f]) & 0xFF) for f in range(d)]
            w.append((brd, 0x400 + r, (int(bq[r]) if brd == 0 else 0) & 0xFFFFFFFF))
        ch0 += c
    w.append((0, 0x7F1, 1))
    return np.array(w, dtype=np.uint32)


def main():
    subject, trial = int(sys.argv[1]), int(sys.argv[2])
    recs, _, full_scale = GL.load_recordings()
    chip, _, _, _ = GL.build_features(recs, full_scale)

    tr = (chip["sid"] != subject) & np.isin(chip["trial"], GL.TRAIN_TRIALS)
    mu, k = G.calibrate_norm(chip["raw"][tr])
    lda = B.build_lda().fit(G.normalize(chip["raw"][tr], mu, k), chip["y"][tr])
    Wf, bf = GL.linear_form(lda)
    W6, b6 = G.quantize_lda(Wf, bf)
    Wd, bd = Wf[1:] - Wf[0], bf[1:] - bf[0]
    s = np.abs(Wd).max() / 127
    Wq, bq = np.round(Wd / s).astype(np.int64), np.round(bd / s).astype(np.int64)

    seg, lab = [], []
    for g in range(4):
        for sig, lbl, sid, t in recs:
            if sid == subject and t == trial:
                _, yy = B.windows_from_recordings([sig], [lbl])
                if len(yy) and yy[0] == g:
                    break
        else:
            sys.exit(f"subject {subject} trial {trial}: no recording of {NAMES[g]}")
        x = GL.adc(sig, full_scale).astype(np.int64)
        seg.append(x); lab += [g] * len(x)
    stream, lab = np.vstack(seg)[:MAX_SAMPLES], np.array(lab)[:MAX_SAMPLES]

    ref = G.LdaCoreRef(8, mu, k, W6, b6)
    golden, true, mixed, at = [], [], [], []
    for n, smp in enumerate(stream):
        r = ref.step(smp)
        if r is None:
            continue
        d = Wq @ r["q"] + bq
        cls, best = 0, 0
        for c in range(len(d)):
            if d[c] > best:
                best, cls = d[c], c + 1
        win = lab[max(0, n - 195):n + 1]
        golden.append(cls); at.append(n)
        true.append(int(np.bincount(win).argmax())); mixed.append(len(set(win)) > 1)

    # the chip masks the first stride, so its decision i is stride i + 1
    out = Path(__file__).resolve().parent / "trials" / f"trial_s{subject}_t{trial}.npz"
    out.parent.mkdir(exist_ok=True)
    np.savez_compressed(out, samples=stream.astype(np.int8), cfg=config_writes(mu, k, Wq, bq),
             golden=np.array(golden[1:]), true=np.array(true[1:]), mixed=np.array(mixed[1:]),
             last_sample=np.array(at[1:]), subject=subject, trial=trial)
    full = ~np.array(mixed[1:])
    acc = np.mean(np.array(golden[1:])[full] == np.array(true[1:])[full]) * 100
    print(f"subject {subject} trial {trial}: {len(stream)} samples, {len(golden) - 1} decisions, "
          f"golden accuracy {acc:.1f}% on single-gesture windows -> {out}")


if __name__ == "__main__":
    main()
