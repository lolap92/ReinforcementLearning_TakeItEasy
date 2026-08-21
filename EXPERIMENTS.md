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
| 2026-08-16 | `2026-08-16_0738_dqn_1m_unmasked_diverged` | dqn_1m_unmasked_diverged | dqn_full_board | 1000 | 0.00 | 0.00 | 0 | 0 | 1b40674 |
| 2026-08-16 | `2026-08-16_1227_dqn_300k_masked_beforefix` | dqn_300k_masked_beforefix | dqn_full_board_masked | 1000 | 47.94 | 19.29 | 0 | 138 | 81aa006 |
| 2026-08-16 | `2026-08-16_2030_dqn_1m_masked_beforefix_collapsed` | dqn_1m_masked_beforefix_collapsed | dqn_full_board_masked | 1000 | 9.89 | 14.64 | 0 | 95 | 49af315 |
| 2026-08-16 | `2026-08-16_2101_ppo_300k_singleenv` | ppo_300k_singleenv | ppo_full_board_masked | 1000 | 52.09 | 25.09 | 0 | 150 | aa354ee |
| 2026-08-16 | `2026-08-16_2101_ppo_300k_singleenv` | ppo_300k_singleenv | ppo_full_board_masked_best_checkpoint | 1000 | 53.08 | 24.10 | 0 | 125 | aa354ee |
| 2026-08-16 | `2026-08-16_2108_dqn_300k_masked_afterfix` | dqn_300k_masked_afterfix | dqn_full_board_masked | 1000 | 31.89 | 16.17 | 0 | 118 | aa354ee |
| 2026-08-16 | `2026-08-16_2108_dqn_300k_masked_afterfix` | dqn_300k_masked_afterfix | dqn_full_board_masked_best_checkpoint | 1000 | 36.21 | 14.42 | 0 | 112 | aa354ee |
| 2026-08-16 | `2026-08-16_2112_ppo_1m_singleenv` | ppo_1m_singleenv | ppo_full_board_masked | 1000 | 92.81 | 26.41 | 5 | 169 | 20039a1 |
| 2026-08-16 | `2026-08-16_2112_ppo_1m_singleenv` | ppo_1m_singleenv | ppo_full_board_masked_best_checkpoint | 1000 | 90.39 | 25.72 | 5 | 153 | 20039a1 |
| 2026-08-16 | `2026-08-16_2133_ppo_25m_singleenv` | ppo_25m_singleenv | ppo_full_board_masked | 1000 | 108.86 | 25.19 | 20 | 190 | 6e167f0 |
| 2026-08-16 | `2026-08-16_2133_ppo_25m_singleenv` | ppo_25m_singleenv | ppo_full_board_masked_best_checkpoint | 1000 | 108.27 | 26.37 | 24 | 188 | 6e167f0 |
| 2026-08-19 | `2026-08-19_0658_ppo_1m_8envs_nstepsbug` | ppo_1m_8envs_nstepsbug | ppo_full_board_masked | 1000 | 83.35 | 22.39 | 0 | 163 | f0fcffc |
| 2026-08-19 | `2026-08-19_0658_ppo_1m_8envs_nstepsbug` | ppo_1m_8envs_nstepsbug | ppo_full_board_masked_best_checkpoint | 1000 | 82.87 | 21.64 | 0 | 168 | f0fcffc |
| 2026-08-21 | `2026-08-21_0641_sweep_baseline_200k_seed0` | sweep_baseline_200k_seed0 | ppo_full_board_masked | 300 | 51.75 | 19.97 | 0 | 99 | e768fd1 |
| 2026-08-21 | `2026-08-21_0641_sweep_baseline_200k_seed0` | sweep_baseline_200k_seed0 | ppo_full_board_masked_best_checkpoint | 300 | 51.72 | 19.96 | 0 | 99 | e768fd1 |
| 2026-08-21 | `2026-08-21_0646_sweep_baseline_200k_seed1` | sweep_baseline_200k_seed1 | ppo_full_board_masked | 300 | 36.79 | 17.70 | 0 | 111 | e768fd1 |
| 2026-08-21 | `2026-08-21_0646_sweep_baseline_200k_seed1` | sweep_baseline_200k_seed1 | ppo_full_board_masked_best_checkpoint | 300 | 36.12 | 17.02 | 0 | 107 | e768fd1 |
| 2026-08-21 | `2026-08-21_0650_sweep_baseline_200k_seed2` | sweep_baseline_200k_seed2 | ppo_full_board_masked | 300 | 56.63 | 24.67 | 0 | 140 | e768fd1 |
| 2026-08-21 | `2026-08-21_0650_sweep_baseline_200k_seed2` | sweep_baseline_200k_seed2 | ppo_full_board_masked_best_checkpoint | 300 | 56.90 | 24.73 | 0 | 140 | e768fd1 |
| 2026-08-21 | `2026-08-21_0654_sweep_reward_shaping_200k_seed0` | sweep_reward_shaping_200k_seed0 | ppo_full_board_masked | 300 | 54.93 | 21.12 | 0 | 127 | e768fd1 |
| 2026-08-21 | `2026-08-21_0654_sweep_reward_shaping_200k_seed0` | sweep_reward_shaping_200k_seed0 | ppo_full_board_masked_best_checkpoint | 300 | 54.93 | 21.12 | 0 | 127 | e768fd1 |
| 2026-08-21 | `2026-08-21_0659_sweep_reward_shaping_200k_seed1` | sweep_reward_shaping_200k_seed1 | ppo_full_board_masked | 300 | 49.88 | 21.84 | 0 | 129 | e768fd1 |
| 2026-08-21 | `2026-08-21_0659_sweep_reward_shaping_200k_seed1` | sweep_reward_shaping_200k_seed1 | ppo_full_board_masked_best_checkpoint | 300 | 45.32 | 21.40 | 0 | 118 | e768fd1 |
| 2026-08-21 | `2026-08-21_0702_sweep_reward_shaping_200k_seed2` | sweep_reward_shaping_200k_seed2 | ppo_full_board_masked | 300 | 35.70 | 21.24 | 0 | 114 | e768fd1 |
| 2026-08-21 | `2026-08-21_0702_sweep_reward_shaping_200k_seed2` | sweep_reward_shaping_200k_seed2 | ppo_full_board_masked_best_checkpoint | 300 | 34.64 | 21.45 | 0 | 114 | e768fd1 |
| 2026-08-21 | `2026-08-21_0706_sweep_constant_lr_200k_seed0` | sweep_constant_lr_200k_seed0 | ppo_full_board_masked | 300 | 36.23 | 19.11 | 0 | 102 | e768fd1 |
| 2026-08-21 | `2026-08-21_0706_sweep_constant_lr_200k_seed0` | sweep_constant_lr_200k_seed0 | ppo_full_board_masked_best_checkpoint | 300 | 34.79 | 17.82 | 0 | 105 | e768fd1 |
| 2026-08-21 | `2026-08-21_0710_sweep_constant_lr_200k_seed1` | sweep_constant_lr_200k_seed1 | ppo_full_board_masked | 300 | 47.01 | 25.31 | 0 | 137 | e768fd1 |
| 2026-08-21 | `2026-08-21_0710_sweep_constant_lr_200k_seed1` | sweep_constant_lr_200k_seed1 | ppo_full_board_masked_best_checkpoint | 300 | 45.17 | 23.36 | 0 | 116 | e768fd1 |
| 2026-08-21 | `2026-08-21_0714_sweep_constant_lr_200k_seed2` | sweep_constant_lr_200k_seed2 | ppo_full_board_masked | 300 | 40.16 | 22.63 | 0 | 108 | e768fd1 |
| 2026-08-21 | `2026-08-21_0714_sweep_constant_lr_200k_seed2` | sweep_constant_lr_200k_seed2 | ppo_full_board_masked_best_checkpoint | 300 | 40.40 | 20.71 | 0 | 120 | e768fd1 |
| 2026-08-21 | `2026-08-21_0718_sweep_no_normalize_200k_seed0` | sweep_no_normalize_200k_seed0 | ppo_full_board_masked | 300 | 46.24 | 22.97 | 0 | 119 | e768fd1 |
| 2026-08-21 | `2026-08-21_0718_sweep_no_normalize_200k_seed0` | sweep_no_normalize_200k_seed0 | ppo_full_board_masked_best_checkpoint | 300 | 46.16 | 23.05 | 0 | 119 | e768fd1 |
| 2026-08-21 | `2026-08-21_0722_sweep_no_normalize_200k_seed1` | sweep_no_normalize_200k_seed1 | ppo_full_board_masked | 300 | 53.55 | 22.18 | 0 | 116 | e768fd1 |
| 2026-08-21 | `2026-08-21_0722_sweep_no_normalize_200k_seed1` | sweep_no_normalize_200k_seed1 | ppo_full_board_masked_best_checkpoint | 300 | 51.60 | 21.12 | 0 | 116 | e768fd1 |
| 2026-08-21 | `2026-08-21_0725_sweep_no_normalize_200k_seed2` | sweep_no_normalize_200k_seed2 | ppo_full_board_masked | 300 | 48.26 | 21.17 | 0 | 109 | e768fd1 |
| 2026-08-21 | `2026-08-21_0725_sweep_no_normalize_200k_seed2` | sweep_no_normalize_200k_seed2 | ppo_full_board_masked_best_checkpoint | 300 | 48.93 | 22.65 | 0 | 109 | e768fd1 |
| 2026-08-21 | `2026-08-21_0729_sweep_n_envs4_200k_seed0` | sweep_n_envs4_200k_seed0 | ppo_full_board_masked | 300 | 51.93 | 18.79 | 0 | 102 | e768fd1 |
| 2026-08-21 | `2026-08-21_0729_sweep_n_envs4_200k_seed0` | sweep_n_envs4_200k_seed0 | ppo_full_board_masked_best_checkpoint | 300 | 51.93 | 18.79 | 0 | 102 | e768fd1 |
| 2026-08-21 | `2026-08-21_0732_sweep_n_envs4_200k_seed1` | sweep_n_envs4_200k_seed1 | ppo_full_board_masked | 300 | 44.43 | 23.55 | 0 | 108 | e768fd1 |
| 2026-08-21 | `2026-08-21_0732_sweep_n_envs4_200k_seed1` | sweep_n_envs4_200k_seed1 | ppo_full_board_masked_best_checkpoint | 300 | 40.77 | 23.39 | 0 | 130 | e768fd1 |
| 2026-08-21 | `2026-08-21_0735_sweep_n_envs4_200k_seed2` | sweep_n_envs4_200k_seed2 | ppo_full_board_masked | 300 | 45.21 | 22.63 | 0 | 107 | e768fd1 |
| 2026-08-21 | `2026-08-21_0735_sweep_n_envs4_200k_seed2` | sweep_n_envs4_200k_seed2 | ppo_full_board_masked_best_checkpoint | 300 | 45.32 | 22.62 | 0 | 107 | e768fd1 |
