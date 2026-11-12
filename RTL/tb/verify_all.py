"""Full check of the multi-board design. From RTL/:   python3 tb/verify_all.py

1. Unit test of emg_chain_link (tb_chain_unit.v), 1-4 boards, 3000 random strides each,
   checked against sums and argmax computed inside the testbench.
2. Mutation test: 8 deliberate bugs are put into a copy of emg_chain_link.v one at a
   time. The unit test must FAIL on every one, otherwise it is not a trustworthy test.
3. Whole chip on real FORS-EMG data (run_mb.py) for six board splits: every decision
   must equal the golden model, and the golden model's multi-board decision must equal
   the original single-chip LDA.

Needs iverilog/vvp on PATH (and python3 with numpy only for mb_prepare.py, not here).
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

TB = Path(__file__).resolve().parent
RTL = TB.parent
LINK = RTL / "emg_chain_link.v"

MUTANTS = [
    ("carry not cleared between words", "carry <= last_bit ? 1'b0 : carry_nxt;", "carry <= carry_nxt;"),
    ("adder ignores the carry", "wire sum       = in_bit ^ my_bit ^ carry;", "wire sum       = in_bit ^ my_bit;"),
    ("tie goes to the higher class", "if (total > best) begin", "if (total >= best) begin"),
    ("last word dropped", "wire last_word = (word == nwords - 1'b1);", "wire last_word = (word == nwords - 2'd2);"),
    ("decision uses stale best_cls", "dec_cls   <= (total > best) ? word + 1'b1 : best_cls;", "dec_cls   <= best_cls;"),
    ("class 0 baseline not reset", "                best     <= 0;\n", ""),
    ("rdy never cleared", "else if (go)       rdy_o <= 1'b0;", "else if (1'b0)     rdy_o <= 1'b0;"),
    ("ids not incremented", "assign id_o    = my_id + 2'd1;", "assign id_o    = my_id;"),
]


def unit(link_file, nb, strides=3000, seed=1):
    with tempfile.TemporaryDirectory() as d:
        vvp = Path(d) / "u.vvp"
        subprocess.run(["iverilog", "-g2005", "-s", "tb_chain_unit", f"-Ptb_chain_unit.NB={nb}",
                        f"-Ptb_chain_unit.STRIDES={strides}", f"-Ptb_chain_unit.SEED={seed}",
                        "-o", str(vvp), str(link_file), str(TB / "tb_chain_unit.v")],
                       check=True, capture_output=True)
        out = subprocess.run(["vvp", "-n", str(vvp)], capture_output=True, text=True).stdout
    line = next((l for l in out.splitlines() if l.startswith("UNIT")), "UNIT FAIL (no result)")
    return line.startswith("UNIT PASS"), line


def main():
    results = []

    print("== 1. emg_chain_link unit test")
    for nb in (1, 2, 3, 4):
        ok, line = unit(LINK, nb)
        print("  ", line)
        results.append((f"unit, {nb} board(s)", ok))

    print("\n== 2. mutation test (each bug must be caught)")
    src = LINK.read_text()
    with tempfile.TemporaryDirectory() as d:
        for name, old, new in MUTANTS:
            assert src.count(old) == 1, f"mutation text not found: {name}"
            bad = Path(d) / "emg_chain_link.v"
            bad.write_text(src.replace(old, new))
            caught = not unit(bad, 3, strides=600)[0] or not unit(bad, 1, strides=600)[0]
            print(f"   {'caught ' if caught else 'MISSED '} {name}")
            results.append((f"mutant: {name}", caught))

    print("\n== 3. whole chip, real FORS-EMG (subject 6, trial 1, 79 decisions)")
    for split in ("8", "4,4", "2,6", "6,2", "5,3", "3,3,2"):
        out = subprocess.run([sys.executable, str(TB / "run_mb.py"), split], cwd=RTL,
                             capture_output=True, text=True).stdout
        m = re.search(r"(\d+) decisions, (\d+) equal to the golden model.*-> (\S+)", out)
        s = re.search(r"golden model = single-chip LDA: (\d+)/(\d+)", out)
        ok = bool(m) and m.group(3) == "BIT-EXACT" and bool(s) and s.group(1) == s.group(2)
        print(f"   split {split:6s} {m.group(2) if m else '?'}/{m.group(1) if m else '?'} bit-exact"
              f"   (golden = single chip on {s.group(1) if s else '?'}/{s.group(2) if s else '?'})")
        results.append((f"chip, split {split}", ok))

    bad = [n for n, ok in results if not ok]
    print("\n" + ("ALL PASS" if not bad else "FAILED: " + "; ".join(bad))
          + f"   ({len(results) - len(bad)}/{len(results)} checks)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
