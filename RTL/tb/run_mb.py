"""Run the multi-board testbench on the real FORS-EMG stream and check every decision.

    python tb/run_mb.py              # from RTL/, default split 4,4 (two boards)
    python tb/run_mb.py 3,3,2        # three boards
    python tb/run_mb.py 8            # one board (head = tail)

Needs mb_data/ from mb_prepare.py. Writes the board configuration to sim_build_mb/,
the waveform to waveform/mb_<split>.vcd, and prints the comparison with the golden model.
"""
import re
import subprocess
import sys
from pathlib import Path

TB = Path(__file__).resolve().parent
RTL = TB.parent
DATA = TB / "mb_data"
NAMES = ["Hand Close", "Hand Open", "Wrist Flex", "Wrist Ext"]


def load_model():
    mu, w, b = {}, [], None
    for line in open(DATA / "model.txt"):
        p = line.split()
        if not p or p[0] == "#":
            continue
        if p[0] == "mu":
            mu[int(p[1])] = (int(p[2]), int(p[3]))
        elif p[0] == "w":
            w.append([int(v) for v in p[2:]])
        elif p[0] == "b":
            b = [int(v) for v in p[1:]]
    return mu, w, b


def write_cfg(split, path):
    """Slice the model by board: each board gets mu/k and W' columns of its own channels."""
    mu, w, b = load_model()
    nd = len(w)
    lines, ch0 = [], 0
    for brd, c in enumerate(split):
        d = 5 * c
        lines.append((brd, 0x7F0, c))
        lines.append((brd, 0x7F2, nd))
        for f in range(d):
            m, k = mu[5 * ch0 + f]
            lines.append((brd, f, (m << 5) | k))
        for r in range(nd):
            for f in range(d):
                lines.append((brd, 0x100 + r * d + f, w[r][5 * ch0 + f] & 0xFF))
            lines.append((brd, 0x400 + r, (b[r] if brd == 0 else 0) & 0xFFFFFFFF))   # bias on the head only
        ch0 += c
    lines.append((0, 0x7F1, 1))          # head core_en last: starts every board together
    with open(path, "w") as f:
        for brd, a, v in lines:
            f.write(f"{brd:x} {a:03x} {v:08x}\n")
    return len(lines)


def main():
    split = [int(v) for v in (sys.argv[1] if len(sys.argv) > 1 else "4,4").split(",")]
    assert sum(split) == 8 and 1 <= len(split) <= 3 and max(split) <= 8, "8 channels over 1-3 boards"
    tag = "_".join(map(str, split))
    build = TB / "sim_build_mb"
    build.mkdir(exist_ok=True)
    (RTL / "waveform").mkdir(exist_ok=True)
    cfg = build / f"cfg_{tag}.txt"
    n = write_cfg(split, cfg)
    exp = [list(map(int, l.split())) for l in open(DATA / "expected.txt") if not l.startswith("#")]
    ns = sum(1 for _ in open(DATA / "samples.txt"))

    c = split + [0] * (3 - len(split))
    vvp = build / f"tb_{tag}.vvp"
    src = sorted(str(p) for p in RTL.glob("*.v")) + [str(TB / "adc_pod_model.v"), str(TB / "tb_mb.v")]
    subprocess.run(["iverilog", "-g2005", "-o", str(vvp), "-s", "tb",
                    f"-Ptb.NB={len(split)}", f"-Ptb.C0={c[0]}", f"-Ptb.C1={c[1]}", f"-Ptb.C2={c[2]}",
                    f"-Ptb.NS={ns}", f'-Ptb.CFG="{cfg}"', f'-Ptb.VCD="{RTL / "waveform" / f"mb_{tag}.vcd"}"'] + src,
                   check=True)
    print(f"split {split}: {n} config writes, {ns} samples, running ...")
    out = subprocess.run(["vvp", "-n", str(vvp)], cwd=TB, capture_output=True, text=True).stdout
    if "ERROR" in out:
        print(out); sys.exit(1)
    dec = [(int(a), int(b), int(t)) for a, b, t in re.findall(r"DEC (\d+) (\d+) (\d+)", out)]

    # decision k at the pins = stride k+1 (stride 0 is masked)
    ok, rows = 0, []
    for k, cls, t in dec:
        e = exp[k + 1]
        ok += cls == e[3]
        rows.append((k + 1, t, cls, e[3], e[2], e[5]))
    same = sum(e[3] == e[4] for e in exp[1:])
    full = [r for r in rows if not r[5]]
    print(f"{len(dec)} decisions, {ok} equal to the golden model"
          f"{'  -> BIT-EXACT' if ok == len(dec) == len(exp) - 1 else '  -> MISMATCH'}")
    print(f"golden model = single-chip LDA: {same}/{len(exp) - 1}")
    print(f"accuracy on single-gesture windows: {sum(r[2] == r[4] for r in full)}/{len(full)}")
    print("stride  time_ms  chip  golden  true")
    for r in rows:
        mark = "" if r[2] == r[3] else "   <-- differs from golden"
        print(f"{r[0]:6d} {r[1] / 1000:8.1f}  {NAMES[r[2]]:11s} {NAMES[r[3]]:11s} {NAMES[r[4]]:11s}"
              f"{' (mixed)' if r[5] else ''}{mark}")


if __name__ == "__main__":
    main()
