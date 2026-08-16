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
