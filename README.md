# Is Your Grade Personal?

Code and results for the paper **"Is Your Grade Personal? Separating Personal, Community, and Contextual Effects in Perceived Climbing Difficulty"**, by Peter Chen, Roger Jin and Hector Liao (CS 371W Research Practicum in AI, Emory University, Fall 2026).

The paper asks when a climber's own history improves the prediction of how hard a route will feel. It introduces PACE (Personalization with Aggregate and Contextual Evidence), a random effects model that estimates a personal effect of the climber, a community effect of the route and a contextual effect of the setter jointly, shrinks each by its evidence and refits as logs arrive. PACE-G applies the personal effect only when its posterior interval excludes zero, and PACE-GBM adds a boosted correction. Every number, table and figure in the paper is produced by the scripts here. Nothing is typed by hand.

## What is in this repository

| Path | Contents |
|---|---|
| `*.py` | All code: data loading, features, models, evaluation, analyses, checks, tables and figures |
| `run_all.sh` | Rebuilds every result from the released data in the order used for the paper |
| `results/` | The result files behind the paper (CSV and JSON). Large prediction arrays (`.npz`) are not included and are rebuilt by `run_all.sh` |
| `requirements.txt` | Package versions used for the results |

## Data

The experiments use the Vertical-Life climbing logs released by Andrić, Ivanova and Ricci (2021) at
<https://github.com/Marina-Andric/climbing_grade_prediction> under the CC BY-SA 4.0 license. The data is not copied here. Clone it next to this repository:

```bash
git clone https://github.com/Marina-Andric/climbing_grade_prediction.git
git clone https://github.com/XinmoChen/CS371_PersonalClimbingGrade.git
```

The code looks for the data in `../climbing_grade_prediction/data` by default. Set `CLIMB_DATA` to use another location and `PACE_OUT` to write results elsewhere than `results/`.

## Reproduce

```bash
cd CS371_PersonalClimbingGrade
pip install -r requirements.txt
bash run_all.sh
```

`run_all.sh` takes about three to four hours on two CPU cores. It runs, in order: data statistics, penalty tuning on development rows, PACE and every baseline under both protocols, ablations, boosted models, evaluation, every analysis of Section 5, the correctness checks, and finally `make_tables.py` and `make_figures.py`, which write `tables/` and `figures/`. Seeded models use seeds 0 to 4 and PACE is deterministic, so a rerun on the same package versions reproduces the files in `results/`. The runtime in `results/timing_indoor.json` is meaningful only on an otherwise idle machine.

## Two evaluation protocols

- **Protocol A (published split).** The first 80% of each climber's logs in time order are training logs and the last 20% are test logs, as in Andrić et al. (2021). The last 10% of each climber's training logs, rounded up, form the development set. Because the split is per climber, 59.2% of indoor test logs have a training log on the same route dated later, so a model fit once on the training rows can learn from the future of a route.
- **Protocol B (forward split).** One time cut separates training, development and test periods (indoor test September 2019 to January 2020, outdoor test 2019 and 2020). Supervised models are refit at the start of every test month.

## Rules the code follows

- A prediction for a log uses only logs with a strictly earlier timestamp, as in the released SQL (`date < t`). PACE refits on earlier logs for every prediction.
- Only an allowlist of columns is loaded (`common.py`). The proposed grade flag, rating, recommendation, tries, repeats, ascent style and style tags are never read by any model or feature, because most of them are written together with the perceived grade. `run_data_audit.py` reads `grade_proposal` once, only to report how often it marks a disagreement (97.6% of flagged logs).
- Every tuned setting is chosen on development rows. Test rows are scored once. Each protocol tunes on its own development rows, because the Protocol A development rows run into the Protocol B test period.
- Intervals come from a climber cluster bootstrap with 1,000 resamples shared by every model and seed. Results of seeded models are computed per seed and averaged.

## Code map

| Paper section | What | File |
|---|---|---|
| 3.1 Task | loading, allowed columns, strictly earlier prefix sums, splits, history strata | `common.py` |
| 3.3 Baseline | separate shrinkage estimator (SSE) features of Andrić et al., rebuilt from the released SQL | `andric.py` |
| 3.4 PACE | Gaussian crossed random effects fit by backfitting, one penalty per block | `pace_model.py` |
| 3.5 Prequential fitting | daily warm started refit and a one step conditional update within the day | `pace_model.py` (`prequential`) |
| 3.6 Evidence gate | keep the personal effect only if its 95% interval excludes zero | `pace_model.py` (`gated`) |
| 3.7 PACE-GBM | LightGBM started from the PACE prediction | `run_gbm_variants.py` |
| 3.8 Attribution, ordinal head | Shapley attribution over personal, community and context, and a cumulative logit head | `run_evaluate.py` |
| 4.2 Splits | Protocol A and Protocol B | `common.py`, `run_data_stats.py` |
| 4.3 Baselines | official grade, SSE with linear regression and random forest, SVD, LightGBM | `run_baselines.py`, `features.py` |
| 4.4 Metrics | RMSE, displayed RMSE, per climber RMSE, disagreement RMSE, macro F1, average precision, helped and hurt shares, cluster bootstrap | `metrics.py` |
| 4.5 Settings | penalty tuning on development rows, runtime | `run_tune_pace.py`, `run_timing.py`, `record_env.py` |
| 4.6 Development | consensus shape, adaptive shrinkage trial (dropped) | `run_consensus_shape.py`, `run_ash_trial.py`, `ash.py`, `eb_ash.py` |
| 4.7 Results | main tables, paired bootstrap for every comparison in the text | `run_evaluate.py`, `run_paired.py` |
| 5.1 Attribution | Shapley values, ablations, why the ablations behave as they do | `run_evaluate.py`, `run_ablation.py`, `run_mechanisms.py` |
| 5.2 Sparse history | results by route and climber history, climber history budget, gain grid | `run_evaluate.py`, `run_budget.py`, `run_gain_grid.py` |
| 5.3 Selectivity | climbers helped and hurt, gate sweep, who is hurt, gate retention | `run_evaluate.py`, `run_extra.py` |
| 5.4 Transfer | personal effects swapped between gym and crag | `run_transfer.py` |
| 5.5 Grade relation | routes whose deviation shifts by a grade or more, disagreement by year, month and group, recent route averages | `run_regrade.py`, `run_window_check.py` |
| 5.6 By label | precision and recall per label, calibration | `run_evaluate.py` |
| 5.7 Errors | rule based causes, 100 sampled errors and their second reading | `run_errors.py`, `check_manual_labels.py` |
| Appendix A | logs without a time of day, data facts | `run_dateonly.py`, `run_data_audit.py` |
| Checks | prequential exactness, determinism, reproduction, leakage, Shapley sums, bootstrap, gate endpoints | `test_pace_exactness.py`, `verify_all.py` |
| Output | LaTeX tables and PDF figures | `make_tables.py`, `make_figures.py` |

## Notes

- The PACE-X interaction grid searched for the paper also included a first grid at larger penalties under Protocol A (λ_v of 3000 and 30000, λ_w of 1000) whose optimum sat at its lower edge. `run_all.sh` runs only the grids whose rows the final selection reads, and it selects the same penalties (λ_v = 30, λ_w = 30 under both protocols).
- `results/errors_manual_labels.csv` is an input, not an output. It holds the second reading of the 100 sampled errors, which was made by an AI assistant (Claude) from each error's route and climber history, as the paper discloses. `check_manual_labels.py` compares it with the rule based causes.
- Files in `results/` that contain rows derived from the Vertical-Life release (`errors_all.csv`, `errors_sample.csv`, `errors_manual_labels.csv`) are shared under CC BY-SA 4.0 with attribution to Andrić, Ivanova and Ricci (2021).
- Claude (Anthropic) helped write and test this code. The authors reviewed it and take responsibility for it.

## Reference

Marina Andrić, Iustina Ivanova and Francesco Ricci. 2021. Climbing route difficulty grade prediction and explanation. In *Proceedings of the IEEE/WIC/ACM International Conference on Web Intelligence and Intelligent Agent Technology (WI-IAT '21)*, pages 285 to 292. ACM. <https://doi.org/10.1145/3486622.3493932>
