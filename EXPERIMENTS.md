# Experiments

Ein Lauf pro Zeile. Details (Config, Kennzahlen, Trajektorien) unter `experiments/<run_id>/`, Format siehe `experiments/README.md`.

| Datum | Run-ID | Tag | Agent | Episoden | Mittelwert | Std | Min | Max | Commit |
|---|---|---|---|---|---|---|---|---|---|
| 2026-08-07 | `2026-08-07_2151_phase3_baselines` | phase3_baselines | random | 1000 | 10.86 | 15.30 | 0 | 82 | e89f794 |
| 2026-08-07 | `2026-08-07_2151_phase3_baselines` | phase3_baselines | greedy | 1000 | 27.72 | 22.74 | 0 | 121 | e89f794 |
| 2026-08-14 | `2026-08-14_1623_phase4_qlearning_mini` | phase4_qlearning_mini | random_mini | 1000 | 6.56 | 8.94 | 0.0 | 36.0 | 86b02ea |
| 2026-08-14 | `2026-08-14_1623_phase4_qlearning_mini` | phase4_qlearning_mini | qlearning_mini | 1000 | 8.81 | 8.95 | 0.0 | 36.0 | 86b02ea |
| 2026-08-14 | `2026-08-14_1623_phase4_qlearning_mini` | phase4_qlearning_mini | optimal_mini | - | 8.89 | 0.00 | - | - | 86b02ea |
| 2026-08-15 | `2026-08-15_0453_phase4_qlearning_mini` | phase4_qlearning_mini | random_mini | 1000 | 6.56 | 8.94 | 0.0 | 36.0 | b27139b |
| 2026-08-15 | `2026-08-15_0453_phase4_qlearning_mini` | phase4_qlearning_mini | qlearning_mini | 1000 | 8.81 | 8.95 | 0.0 | 36.0 | b27139b |
| 2026-08-15 | `2026-08-15_0453_phase4_qlearning_mini` | phase4_qlearning_mini | optimal_mini | - | 8.89 | 0.00 | - | - | b27139b |
| 2026-08-16 | `2026-08-16_0738_phase5_dqn` | phase5_dqn | dqn_full_board | 1000 | 0.00 | 0.00 | 0 | 0 | 1b40674 |
| 2026-08-16 | `2026-08-16_1227_phase5_dqn_masked` | phase5_dqn_masked | dqn_full_board_masked | 1000 | 47.94 | 19.29 | 0 | 138 | 81aa006 |
| 2026-08-16 | `2026-08-16_2030_phase5_dqn_masked` | phase5_dqn_masked | dqn_full_board_masked | 1000 | 9.89 | 14.64 | 0 | 95 | 49af315 |
| 2026-08-16 | `2026-08-16_2101_phase6_maskable_ppo` | phase6_maskable_ppo | ppo_full_board_masked | 1000 | 52.09 | 25.09 | 0 | 150 | aa354ee |
| 2026-08-16 | `2026-08-16_2101_phase6_maskable_ppo` | phase6_maskable_ppo | ppo_full_board_masked_best_checkpoint | 1000 | 53.08 | 24.10 | 0 | 125 | aa354ee |
| 2026-08-16 | `2026-08-16_2108_phase5_dqn_masked` | phase5_dqn_masked | dqn_full_board_masked | 1000 | 31.89 | 16.17 | 0 | 118 | aa354ee |
| 2026-08-16 | `2026-08-16_2108_phase5_dqn_masked` | phase5_dqn_masked | dqn_full_board_masked_best_checkpoint | 1000 | 36.21 | 14.42 | 0 | 112 | aa354ee |
