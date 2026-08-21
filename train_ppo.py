"""
MaskablePPO via SB3-Contrib auf dem vollen Take-It-Easy-Board (Phase 6).

Direkter Vergleich zu train_dqn.py (Phase 5): gleiches Environment, gleiche
Netzarchitektur (128, 128), gleiches gamma=1.0 (keine Abzinsung - der Score
kommt erst im letzten Zug, alle 19 Züge zählen gleich viel), gleiche
Auswertungsmethodik. Unterschied ist bewusst nur Value-based (DQN, lernt
Q(s,a)) vs. Policy-based (PPO, lernt direkt eine Policy π(a|s)).

Warum MaskablePPO statt normalem PPO: wie bei DQN (siehe train_dqn.py) sind
19 von 20 Feldern im letzten Zug ungültig (belegt) - ohne Masking würde die
Policy ständig ungültige Züge probieren/lernen müssen. Anders als bei DQN
brauchen wir hier aber keinen eigenen Masked*-Policy-Hack: sb3-contrib bringt
Action Masking für PPO nativ mit (ActionMasker-Wrapper + MaskablePPO), das
war der eigentliche Grund, für Phase 6 auf sb3-contrib zu wechseln.

Nutzung (lokal, in deiner IDE):
    pip install -r requirements.txt
    python train_ppo.py --timesteps 300000
    python train_ppo.py --timesteps 1000000 --device cuda   # GPU erzwingen
    python train_ppo.py --timesteps 1000000 --n-envs 8      # 8 Envs parallel

--n-envs (Optimierung, siehe Chat): PPO sammelt Rollouts normalerweise über
mehrere Environments gleichzeitig statt nur einer - das gibt pro Policy-
Update vielfältigere, weniger korrelierte Trajektorien (stabilere Advantage-
Schätzung) und nutzt mehrere CPU-Kerne parallel aus (relevant, weil hier laut
--device-Hinweis unten ohnehin meist auf CPU trainiert wird). Ab n-envs > 1
werden die Environments über separate Prozesse (SubprocVecEnv) parallelisiert.

Wichtig, sonst Falle: Gesamt-Batchgröße pro Update ist n_steps * n_envs. Ein
1-Mio.-Lauf mit n_envs=8 und unverändertem n_steps=512 hat deshalb nur noch
1/8 so viele Policy-Updates wie derselbe Lauf mit n_envs=1 - das hat in der
Praxis das Ergebnis verschlechtert (92,81 -> 83,35 Mean-Score, siehe Chat-
Analyse), obwohl "mehr Environments" nach einer reinen Verbesserung klingt.
--n-steps wird deshalb standardmäßig automatisch mit n_envs runterskaliert
(siehe --n-steps unten), um die Update-Anzahl ungefähr zu erhalten.

Weitere "Wave 1"-Optimierungen (siehe Chat) gegenüber der ursprünglichen
Phase-6-Version:
  - Reward-Normalisierung (VecNormalize, nur Reward, nicht Observation) -
    der Score bis 307 bei gamma=1.0 ist ein großskaliges, hochvariantes
    Zielsignal für die Value-Function; norm_obs bleibt bewusst aus, damit es
    keine Diskrepanz zwischen Trainings- und Eval-/Replay-Normalisierung
    geben kann (replay.py/evaluate() nutzen immer die rohe Environment).
  - Linear abfallende statt konstanter Lernrate (--constant-lr zum Abschalten).
  - Größere Value-Function (vf=[256,256] statt [128,128]) - die Value-
    Function hat mit dem Sparse-Reward die schwerere Aufgabe als die Policy.

"Wave 2": --reward-shaping (Default aus). Potential-based Reward Shaping
(siehe env.py) gibt bei jedem Zug statt nur im letzten ein Lernsignal, bleibt
aber laut Ng et al. (1999) policy-invariant. Wirkt NUR auf train_env -
eval_env (für eval/mean_reward) und evaluate() (finale Auswertung/
summary.json) nutzen immer den echten, ungeshapten Score, damit Läufe mit/
ohne Shaping vergleichbar bleiben. ACHTUNG beim TensorBoard-Ablesen:
rollout/score_mean/min/max kommt aus dem (ggf. geshapten) Trainings-Reward -
mit --reward-shaping ist das rechnerisch ~2x der echten Score-Größenordnung
(siehe env.py-Docstring), kein Bug. eval/mean_reward bleibt davon unberührt
und ist immer der echte Score.

    # In einem zweiten Terminal, um live mitzuverfolgen:
    tensorboard --logdir experiments
    # dann im Browser: http://localhost:6006
    # rollout/score_mean/min/max = Spiel-Score (nicht nur Reward) über die
    # letzten 100 Trainings-Episoden, eval/mean_reward = Score des
    # Eval-Checkpoints alle 10k Steps

Nach dem Training landet alles unter experiments/<run_id>/:
  - config.json, summary.json          -> ins Git-Repo committen
  - models/, tensorboard/              -> bewusst NICHT im Repo (.gitignore),
                                           lokal reproduzierbar aus config.json
                                           (gleicher Seed + gleicher Git-Commit)
"""

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from gymnasium.wrappers import TimeLimit
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from env import TakeItEasyEnv, NUM_CELLS

REPO_ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
EXPERIMENTS_LOG = REPO_ROOT / "EXPERIMENTS.md"

# Sicherheitsnetz (defense-in-depth): mit korrektem Masking sollte eine
# Episode nie länger als 19 Züge dauern. Das Limit fängt trotzdem jeden
# unvorhergesehenen Fall ab, statt dass ein Lauf endlos hängen bleibt.
MAX_EPISODE_STEPS = 200


def mask_fn(env) -> np.ndarray:
    """Für ActionMasker: liest die Maske direkt aus der TakeItEasyEnv-Instanz,
    unabhängig davon, wie viele Wrapper (TimeLimit etc.) darüberliegen."""
    return env.unwrapped._get_action_mask()


class ScoreLoggingCallback(BaseCallback):
    """Loggt den Spiel-Score separat unter eigenen TensorBoard-Tags
    (`rollout/score_mean`, `rollout/score_min`, `rollout/score_max`), statt
    ihn nur implizit unter dem generischen SB3-Tag `rollout/ep_rew_mean`
    mitlaufen zu lassen. Rechnerisch ist Mean identisch zu ep_rew_mean: der
    Reward ist 0 in jedem Zug außer dem letzten, wo er genau dem Spiel-Score
    entspricht (siehe env.py), die Episodensumme des Rewards *ist* also der
    Score. Min/Max zeigen zusätzlich die beste/schlechteste Runde innerhalb
    des rollierenden Fensters (die letzten 100 Episoden, SB3-Default) - das
    eigene Tag macht das im TensorBoard-Dashboard sofort ohne Nachdenken
    ablesbar."""

    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0:
            scores = [ep_info["r"] for ep_info in self.model.ep_info_buffer]
            self.logger.record("rollout/score_mean", float(np.mean(scores)))
            self.logger.record("rollout/score_min", float(np.min(scores)))
            self.logger.record("rollout/score_max", float(np.max(scores)))
        return True


def git_commit_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return None


def make_env(reward_shaping=False):
    return ActionMasker(
        TimeLimit(TakeItEasyEnv(reward_shaping=reward_shaping), max_episode_steps=MAX_EPISODE_STEPS),
        mask_fn,
    )


def linear_schedule(initial_value):
    """SB3-Lernraten-Schedule: linear von initial_value auf 0 über den
    gesamten Trainingslauf (progress_remaining läuft von 1.0 auf 0.0)."""
    def schedule(progress_remaining):
        return progress_remaining * initial_value
    return schedule


def evaluate(model, n_episodes, seed):
    env = TakeItEasyEnv()
    scores = np.empty(n_episodes)
    invalid_counts = np.empty(n_episodes)
    for i in range(n_episodes):
        obs, info = env.reset(seed=seed + i)
        terminated = False
        total_reward = 0.0
        invalid = 0
        steps = 0
        while not terminated and steps < MAX_EPISODE_STEPS:
            action, _ = model.predict(obs, action_masks=info["action_mask"], deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            if info.get("invalid_action"):
                invalid += 1
            elif terminated:
                total_reward = reward
            steps += 1
        scores[i] = total_reward
        invalid_counts[i] = invalid
    return scores, invalid_counts


def steps_label(n):
    """Kurzform für Timesteps in Tags/Run-IDs, z.B. 300_000 -> '300k', 1_000_000 -> '1m'."""
    if n % 1_000_000 == 0:
        return f"{n // 1_000_000}m"
    if n % 1_000 == 0:
        return f"{n // 1000}k"
    return str(n)


def build_default_tag(args, n_steps):
    """Baut aus den tatsächlichen Settings einen selbsterklärenden Tag, z.B.
    'ppo_1m_8envs_rewardshaping' - Steps, n_envs und alle vom Default
    abweichenden Wave-1/2-Flags gehen direkt aus dem Namen hervor."""
    parts = [f"ppo_{steps_label(args.timesteps)}_{args.n_envs}envs"]
    if args.n_steps is not None:
        parts.append(f"nsteps{n_steps}")
    if args.reward_shaping:
        parts.append("rewardshaping")
    if args.constant_lr:
        parts.append("constlr")
    if args.no_normalize:
        parts.append("nonorm")
    return "_".join(parts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MaskablePPO-Training für Take It Easy (Phase 6, lokal ausführen).")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--eval-episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--tag", type=str, default=None,
        help="Freitext für den Run-Ordner-Namen. Default: automatisch aus den "
             "Settings gebaut (z.B. 'ppo_1m_8envs_rewardshaping'), damit Steps/"
             "Modus/Besonderheiten aus dem Namen hervorgehen, ohne dass man "
             "selbst daran denken muss.",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="'cpu' (Default), 'cuda' (GPU erzwingen) oder 'auto' (SB3 wählt). "
             "Hinweis: bei einem so kleinen Netz (128,128) ist der "
             "Flaschenhals meist der CPU-seitige Env-Step, nicht das Netz "
             "selbst - GPU bringt hier oft wenig bis nichts, schadet aber "
             "auch nicht.",
    )
    parser.add_argument(
        "--n-envs", type=int, default=1,
        help="Anzahl paralleler Trainings-Environments (siehe Docstring oben). "
             "1 = wie bisher (ein einzelner Prozess), >1 = separate Prozesse "
             "(SubprocVecEnv) - mehr Kerne genutzt, aber auch mehr RAM/Overhead.",
    )
    parser.add_argument(
        "--n-steps", type=int, default=None,
        help="Rollout-Länge pro Environment vor jedem Policy-Update. Default: "
             "automatisch max(32, 512 // n_envs), damit die Anzahl Updates bei "
             "mehr n_envs nicht einbricht (siehe Docstring/Chat-Analyse) - "
             "explizit setzen, um das zu überschreiben.",
    )
    parser.add_argument(
        "--no-normalize", action="store_true",
        help="Reward-Normalisierung (VecNormalize) abschalten (Default: an).",
    )
    parser.add_argument(
        "--constant-lr", action="store_true",
        help="Konstante Lernrate (3e-4) statt des linear abfallenden Default-Schedules.",
    )
    parser.add_argument(
        "--reward-shaping", action="store_true",
        help="Potential-based Reward Shaping an (siehe env.py, Docstring dort). "
             "Default aus - Trainings-Reward bleibt dann wie bisher sparse "
             "(nur im letzten Zug != 0). Wirkt nur auf das Training; "
             "eval/mean_reward und die finale Auswertung (evaluate()) nutzen "
             "immer den echten, ungeshapten Spiel-Score, damit Läufe mit und "
             "ohne Shaping vergleichbar bleiben.",
    )
    args = parser.parse_args()
    n_steps = args.n_steps if args.n_steps is not None else max(32, 512 // args.n_envs)
    if args.tag is None:
        args.tag = build_default_tag(args, n_steps)

    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    run_id = f"{timestamp[:16].replace(':', '').replace('T', '_')}_{args.tag}"
    run_dir = EXPERIMENTS_DIR / run_id
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    (run_dir / "tensorboard").mkdir(exist_ok=True)

    train_env = make_vec_env(
        make_env,
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv if args.n_envs > 1 else DummyVecEnv,
        env_kwargs=dict(reward_shaping=args.reward_shaping),
    )
    if not args.no_normalize:
        # Nur Reward normalisieren, nicht die Observation (siehe Docstring) -
        # so kann evaluate()/replay.py weiter unverändert die rohe
        # Environment nutzen, ohne gespeicherte Normalisierungs-Statistiken
        # laden zu müssen.
        train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, gamma=1.0)
    eval_env = Monitor(make_env())

    learning_rate = 3e-4 if args.constant_lr else linear_schedule(3e-4)
    net_arch = dict(pi=[128, 128], vf=[256, 256])

    model = MaskablePPO(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=64,
        n_epochs=10,
        gamma=1.0,  # keine Abzinsung - die Zielgröße ist der Score am Episodenende, alle 19 Züge zählen gleich (wie bei DQN, siehe train_dqn.py)
        gae_lambda=0.95,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=net_arch),
        tensorboard_log=str(run_dir / "tensorboard"),
        seed=args.seed,
        device=args.device,
        verbose=1,
    )
    print(f"Gewähltes Device: {model.device}")
    print(f"n_steps={n_steps} (n_envs={args.n_envs}, Batch/Update={n_steps * args.n_envs})")

    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "models"),
        log_path=str(run_dir / "tensorboard"),
        eval_freq=10_000,
        n_eval_episodes=50,
        deterministic=True,
    )
    score_callback = ScoreLoggingCallback()

    print(f"Trainiere MaskablePPO für {args.timesteps} Timesteps ...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=CallbackList([score_callback, eval_callback]),
        tb_log_name="ppo",
    )
    model.save(str(run_dir / "models" / "final_model"))

    print("\nAuswertung über echte Take-It-Easy-Episoden (19 Züge, volles Board) ...")
    scores, invalid_counts = evaluate(model, args.eval_episodes, seed=args.seed + 1)
    print(f"PPO (Endmodell nach {args.timesteps} Steps):  mean={scores.mean():.2f}  "
          f"std={scores.std():.2f}  min={scores.min():.0f}  max={scores.max():.0f}")
    print(f"Ungültige Züge je Episode (Ø, sollte 0 sein): {invalid_counts.mean():.2f}")

    # Wie bei DQN (siehe train_dqn.py): das Endmodell muss nicht das beste
    # sein, das während des Trainings gesehen wurde. MaskableEvalCallback
    # speichert deshalb laufend das beste Zwischen-Checkpoint (best_model.zip)
    # nach eval/mean_reward - das werten wir zusätzlich aus, um sowas
    # sichtbar zu machen statt es im Endmodell-Ergebnis zu verstecken.
    best_model_path = run_dir / "models" / "best_model.zip"
    best_scores = None
    if best_model_path.exists():
        best_model = MaskablePPO.load(str(best_model_path), env=train_env, device=args.device)
        best_scores, best_invalid_counts = evaluate(best_model, args.eval_episodes, seed=args.seed + 1)
        print(f"PPO (bestes Zwischen-Checkpoint):           mean={best_scores.mean():.2f}  "
              f"std={best_scores.std():.2f}  min={best_scores.min():.0f}  max={best_scores.max():.0f}")
        print(f"Ungültige Züge je Episode (Ø, sollte 0 sein): {best_invalid_counts.mean():.2f}")
        if best_scores.mean() > scores.mean():
            print(
                "-> Bestes Checkpoint schlägt Endmodell deutlich: Hinweis auf "
                "Instabilität spät im Training."
            )

    config = {
        "run_id": run_id,
        "tag": args.tag,
        "timestamp": timestamp,
        "git_commit": git_commit_hash(),
        "env": "TakeItEasyEnv",
        "algorithm": "MaskablePPO (sb3-contrib)",
        "hyperparameters": {
            "timesteps": args.timesteps,
            "learning_rate": "constant(3e-4)" if args.constant_lr else "linear(3e-4 -> 0)",
            "n_steps": n_steps,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 1.0,
            "gae_lambda": 0.95,
            "ent_coef": 0.01,
            "net_arch": net_arch,
            "reward_normalize": not args.no_normalize,
            "reward_shaping": args.reward_shaping,
        },
        "eval_episodes": args.eval_episodes,
        "master_seed": args.seed,
        "device": str(model.device),
        "n_envs": args.n_envs,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    summary = [{
        "name": "ppo_full_board_masked",
        "n_episodes": args.eval_episodes,
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "min": float(scores.min()),
        "max": float(scores.max()),
        "avg_invalid_actions_per_episode": float(invalid_counts.mean()),
    }]
    if best_scores is not None:
        summary.append({
            "name": "ppo_full_board_masked_best_checkpoint",
            "n_episodes": args.eval_episodes,
            "mean": float(best_scores.mean()),
            "std": float(best_scores.std()),
            "min": float(best_scores.min()),
            "max": float(best_scores.max()),
            "avg_invalid_actions_per_episode": float(best_invalid_counts.mean()),
        })
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(run_dir / "episodes.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["agent", "episode_index", "seed", "score"])
        for i, score in enumerate(scores):
            writer.writerow(["ppo_full_board_masked", i, args.seed + 1 + i, score])
        if best_scores is not None:
            for i, score in enumerate(best_scores):
                writer.writerow(["ppo_full_board_masked_best_checkpoint", i, args.seed + 1 + i, score])

    is_new = not EXPERIMENTS_LOG.exists()
    with open(EXPERIMENTS_LOG, "a") as f:
        if is_new:
            f.write("# Experiments\n\n")
            f.write(
                "Ein Lauf pro Zeile. Details (Config, Kennzahlen, Trajektorien) "
                "unter `experiments/<run_id>/`, Format siehe `experiments/README.md`.\n\n"
            )
            f.write("| Datum | Run-ID | Tag | Agent | Episoden | Mittelwert | Std | Min | Max | Commit |\n")
            f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for s in summary:
            f.write(
                f"| {timestamp[:10]} | `{run_id}` | {args.tag} | {s['name']} | "
                f"{s['n_episodes']} | {s['mean']:.2f} | {s['std']:.2f} | "
                f"{s['min']:.0f} | {s['max']:.0f} | {config['git_commit'] or '-'} |\n"
            )

    print(f"\nRun gespeichert unter: {run_dir.relative_to(REPO_ROOT)}")
    print("models/ und tensorboard/ sind gitignored (groß/binär) - config.json und")
    print("summary.json reichen, um den Lauf ins Repo zu committen.")
