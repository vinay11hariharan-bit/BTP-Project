# FPGA test: two-board LDA classifier on a PYNQ-Z1 / Z2, driven from Python

Two copies of the chip (`RTL/emg_top.v`) run on the FPGA as a head + tail chain (4 + 4 channels).
The Zynq's ARM core (PYNQ, Python) loads a trial into the FPGA, starts it and reads back every
decision, so the board does not have to be seen and no LEDs are needed.

```
Python on the PYNQ ──AXI──> emg_axi ─┬─ config memory  (171 writes: LDA weights, bias, settings)
                                     ├─ sample memory  (up to 32768 samples x 8 channels, 8-bit)
                                     ├─ emg_bench: config loader, 2 x fake ADC (SPI), head + tail chip
                                     └─ decision memory (class + sample index of every decision)
```

The chips run on a 10 MHz clock from the Zynq, gated down to their real 256.4 kHz
(or left at 10 MHz in fast mode; the chip only sees clock edges, so the results are the same).

## Files

| Path | What it is |
|---|---|
| `rtl/emg_axi.v` | AXI4-Lite registers and the three memories (top of the PL design) |
| `rtl/emg_bench.v` | config loader, the two chips, the sample stream, decision capture |
| `rtl/emg_clk_gate.v` | 10 MHz → chip clock (BUFGCE) |
| `rtl/fpga_pod.v` | fake 8-channel ADC with the same SPI timing as the testbench model |
| `constraints/pynq.xdc` | LED pins (optional, the design works without looking at them) |
| `build.tcl` | makes the Vivado project, the block design (Zynq + emg_axi) and the bitstream |
| `host/run_trial.py` | runs on the PYNQ: loads a trial, runs it, compares with the golden model |
| `host/make_trial.py` | runs on the laptop (needs the dataset): makes a trial file |
| `host/trials/*.npz` | ready trial files |
| `sim/tb_fpga.v`, `mem/` | simulation of the whole thing through the AXI port |

The chip RTL is read from `../RTL`, so keep the whole repo, not just this folder.

## 1. Build (Vivado GUI, 2020.x or newer)

1. Open Vivado. **Tools → Run Tcl Script…** → `fpga/build.tcl`.
2. Wait. It creates the project in `fpga/vivado_proj/`, draws the block design, and runs
   synthesis, implementation and bitstream (the GUI is busy until it finishes, ~10 min).
3. The outputs are in `fpga/out/`:
   - `emg_fpga.bit` and `emg_fpga.hwh` — **both** go to the board, with the same name
   - `timing.txt` (WNS / WHS must be ≥ 0) and `utilization.txt` (about 2k LUTs, 65 BRAM36)

Command line instead: `cd fpga && vivado -mode batch -source build.tcl`.

## 2. Make a trial (laptop)

```
python fpga/host/make_trial.py 6 1        # subject 6, trial 1 -> fpga/host/trials/trial_s6_t1.npz
```

It trains the chip LDA on the other 18 subjects (as in the paper), takes trial 1 of subject 6
(one 8000-sample recording of each gesture, 32000 samples, ~32.5 s of EMG), and stores the samples, the
config writes, the golden model's decisions and the true labels. Ready files:

- `trial_s6_t1.npz` — the full trial, 325 decisions
- `selftest_s6_t1_cut.npz` — the 7840-sample cut used in the RTL simulation, 79 decisions

## 3. Run on the PYNQ

Copy `emg_fpga.bit`, `emg_fpga.hwh`, `host/run_trial.py` and a trial `.npz` into one folder on the
board (e.g. `/home/xilinx/emg/` through Jupyter's upload button, or `scp`). Then in a Jupyter
notebook in that folder:

```
%run run_trial.py trial_s6_t1.npz            # fast: 10 MHz, under a second
%run run_trial.py trial_s6_t1.npz --real     # real time: 256 kHz, ~33 s
```

or from a terminal: `sudo python3 run_trial.py trial_s6_t1.npz`.

It prints every decision (FPGA, golden model, true gesture) and the summary:

```
325/325 decisions read, 325 equal to the golden model  -> BIT-EXACT
accuracy on single-gesture windows: FPGA xx.x%, golden xx.x%
```

`BIT-EXACT` means the hardware gave exactly the golden model's answers.

## Registers (for writing your own Python)

| Address | Name | |
|---|---|---|
| 0x00000 | CTRL | [0] run (0 = reset, 1 = start), [1] fast |
| 0x00004 | STATUS | [0] config done, [1] done, [2] overrun, [25:16] decisions so far |
| 0x00008 | NSAMP | samples to play (≤ 32768) |
| 0x0000C | NCFG | config writes (≤ 256) |
| 0x00010 | NWANT | decisions to record before done (≤ 1023) |
| 0x0001C | ID | reads 0x454D4701 |
| 0x01000 + 8i | CFG | write i: data, then +4: board << 12 \| addr |
| 0x02000 + 4i | DEC | decision i: sample count << 16 \| class |
| 0x40000 + 8i | SAMPLES | sample i: ch0..ch3, then +4: ch4..ch7 (ch0 in the top byte) |

The block is at 0x40000000 (512 KB).

## Simulate (optional, Icarus Verilog)

```
cd fpga/mem
iverilog -g2005 -DSIM -s tb_fpga -o tb.vvp ../../RTL/*.v ../rtl/*.v ../sim/tb_fpga.v
vvp -n tb.vvp
```

It loads the self-test case over AXI exactly as `run_trial.py` does and ends with
`FPGA_TOP PASS: 79 decisions equal to the golden model`.
