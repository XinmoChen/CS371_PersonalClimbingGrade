#!/bin/bash
# Rebuilds every result in results/, every table in tables/ and every figure in figures/ from the released data,
# in the order used for the paper. About three to four hours on two CPU cores. Logs go to logs/.
# The data must be cloned next to this folder (see README) or CLIMB_DATA must point to its data/ folder.
set -e
cd "$(dirname "$0")"
mkdir -p results logs
step=0
run() { step=$((step + 1)); echo "[$step] $*"; "$@" > "logs/step_$(printf '%02d' $step).log" 2>&1; }

run python3 run_data_stats.py                      # data and split statistics, leakage check of the published split

# 1. PACE penalties, chosen on development rows only (Section 4.5)
run python3 run_tune_pace.py indoor base
run python3 run_tune_pace.py indoor refine '[{"climber":40,"route":6,"env":30},{"climber":20,"route":6,"env":30},{"climber":40,"route":8,"env":30},{"climber":20,"route":8,"env":30}]'
run python3 run_tune_pace.py indoor refine '[{"climber":20,"route":3,"env":30},{"climber":40,"route":3,"env":30},{"climber":20,"route":5,"env":30},{"climber":40,"route":5,"env":30},{"climber":20,"route":4,"env":10},{"climber":40,"route":4,"env":10},{"climber":60,"route":4,"env":30},{"climber":30,"route":4,"env":30}]'
run env PACE_PERIOD_DAYS=7 python3 run_tune_pace.py outdoor base
run env PACE_PERIOD_DAYS=7 python3 run_tune_pace.py outdoor refine '[{"climber":20,"route":1,"env":300},{"climber":20,"route":2,"env":300},{"climber":20,"route":1,"env":30},{"climber":20,"route":2,"env":30},{"climber":10,"route":3,"env":300},{"climber":10,"route":2,"env":300}]'
run python3 run_tune_pace.py indoor inter                              # PACE-X grid around each protocol's penalties
run python3 run_tune_pace.py indoor inter '[10, 30, 100]' '[3, 10, 30]'  # extension below the first grid's lower edge

# 2. Models
run python3 run_pace.py indoor A
run python3 run_pace.py indoor B
run python3 run_pace.py outdoor A 7                 # outdoor PACE refits weekly
run python3 run_pace.py outdoor B 7
run python3 run_baselines.py indoor
run python3 run_baselines.py outdoor
run python3 run_ablation.py indoor
run python3 run_gbm_variants.py indoor
run python3 run_gbm_variants.py outdoor

# 3. Development analyses (Section 4.6)
run python3 run_consensus_shape.py
run python3 run_ash_trial.py route

# 4. Evaluation and analyses (Sections 4.7 and 5)
run python3 run_evaluate.py indoor
run python3 run_evaluate.py outdoor
run python3 run_budget.py indoor
run python3 run_errors.py
run python3 check_manual_labels.py                  # compares the error rules with results/errors_manual_labels.csv
run python3 run_gain_grid.py
run python3 run_dateonly.py
run python3 run_paired.py indoor outdoor
run python3 run_regrade.py indoor outdoor
run python3 run_transfer.py
run python3 run_window_check.py
run python3 run_extra.py
run python3 run_mechanisms.py
run python3 run_data_audit.py

# 5. Checks, environment, runtime, tables and figures
run python3 test_pace_exactness.py
run python3 verify_all.py
run python3 record_env.py
run python3 run_timing.py                           # meaningful only on an otherwise idle machine
run python3 make_tables.py indoor outdoor
run python3 make_figures.py
echo "done: see results/, tables/ and figures/"
