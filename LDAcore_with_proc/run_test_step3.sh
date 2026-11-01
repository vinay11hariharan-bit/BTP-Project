#!/bin/bash
# Regression for RTL steps 1-3 (1 kHz sampling; bank size, bank count and warm-up as settings;
# 24-bit mu' and signed k; no on-chip LDA, microcontroller link over SPI).
# Run from the RTL folder:   bash run_test_step3.sh
# Each test prints "running..." first and its result when it ends (1 to 6 minutes each).
# Logs go to logs_step3/. One test is EXPECTED to fail (it breaks an ADS1299 timing rule on
# purpose): c13_td130_expectFAIL. Each main-testbench run (t1-t6) runs 3 tests: chip,
# settings and link (skip rule, mixed read).
cd "$(dirname "$0")"
L=logs_step3
mkdir -p $L
rm -f $L/SUMMARY
run() { # name, environment settings, command...
  local name=$1; shift; local envs=$1; shift
  printf "%-24s running...  " "$name"
  env $envs "$@" > $L/$name.log 2>&1
  local r=$(grep -o "TESTS=[0-9]* PASS=[0-9]* FAIL=[0-9]*" $L/$name.log | tail -1)
  if [ -z "$r" ]; then r="ERROR, see $L/$name.log"; fi
  echo "$r"
  echo "$name | $envs | $r" >> $L/SUMMARY
}
# Stage 2 alone: the 8-bit scaling against the golden model (every k, edge cases)
run u1_shift_norm         "X=1" python tb/test_shift_norm.py
# main testbench (Python ADS1299 model): default B=50 N=3 W=2, the old 98x2, odd and extreme settings
run t1_default            "X=1" python tb/test_emg_top.py
run t2_old_98x2_W1        "EMG_BANK_LEN=98 EMG_N_BANKS=2 EMG_WARMUP=1" python tb/test_emg_top.py
run t3_7x4_W3             "EMG_BANK_LEN=7 EMG_N_BANKS=4 EMG_WARMUP=3" python tb/test_emg_top.py
run t4_C24_100x4_W3       "EMG_CHANNELS=24 EMG_BANK_LEN=100 EMG_N_BANKS=4 EMG_WARMUP=3" python tb/test_emg_top.py
run t5_5x1_W0             "EMG_BANK_LEN=5 EMG_N_BANKS=1 EMG_WARMUP=0" python tb/test_emg_top.py
run t6_old_requant        "EMG_REQUANT=old" python tb/test_emg_top.py
# chain testbench (Verilog ADS1299 models)
run c01_1x8               "X=1" python tb/test_emg_chain.py
run c01b_1x8_old          "EMG_BANK_LEN=98 EMG_N_BANKS=2 EMG_WARMUP=1" python tb/test_emg_chain.py
run c02_3x8               "X=1" python tb/test_emg_chain.py --pods 3
run c03_4x6               "X=1" python tb/test_emg_chain.py --pods 4 --pod-ch 6
run c04_4x4_C14           "X=1" python tb/test_emg_chain.py --pods 4 --pod-ch 4 --channels 14
run c05_cfg8_3x4          "X=1" python tb/test_emg_chain.py --pods 3 --pod-ch 4 --cfg-dev-ch 8
run c06_faults            "X=1" python tb/test_emg_chain.py --pods 3 --stuck 2 --quiet 5 --leadoff 7
run c07_C16_missing       "X=1" python tb/test_emg_chain.py --channels 16
run c08_drop              "X=1" python tb/test_emg_chain.py --drop-frame 130
run c09_nodrdy            "X=1" python tb/test_emg_chain.py --no-drdy
run c10_late              "X=1" python tb/test_emg_chain.py --late-drdy
run c11_badc              "X=1" python tb/test_emg_chain.py --bad-c
run c12_td32              "X=1" python tb/test_emg_chain.py --t-d 32
run c13_td130_expectFAIL  "X=1" python tb/test_emg_chain.py --t-d 130
run c15_3x8_C20           "X=1" python tb/test_emg_chain.py --pods 3 --channels 20
run c16_cfg4_2x8          "X=1" python tb/test_emg_chain.py --pods 2 --cfg-dev-ch 4
run c17_cfg0              "X=1" python tb/test_emg_chain.py --cfg-dev-ch 0
run c18_shift20           "X=1" python tb/test_emg_chain.py --shift 20
run c19_C1                "X=1" python tb/test_emg_chain.py --channels 1
run c20_6x4               "X=1" python tb/test_emg_chain.py --pods 6 --pod-ch 4
run c21_lo_unused         "X=1" python tb/test_emg_chain.py --pods 2 --channels 8 --leadoff 12
run c22_lo_frames         "X=1" python tb/test_emg_chain.py --leadoff 3 --leadoff-frames 120-140
run c23_C24_100x4_W3      "EMG_BANK_LEN=100 EMG_N_BANKS=4 EMG_WARMUP=3" python tb/test_emg_chain.py --pods 3
echo "done: see $L/SUMMARY"
