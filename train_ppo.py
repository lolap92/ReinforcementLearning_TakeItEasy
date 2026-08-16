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

    # In einem zweiten Terminal, um live mitzuverfolgen:
    tensorboard --logdir experiments
    # dann im Browser: http://localhost:6006
    # rollout/score_mean = Spiel-Score (nicht nur Reward), eval/mean_reward
    # = Score des Eval-Checkpoints alle 10k Steps

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
from stable_baselines3.common.monitor import Monitor

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
    """Loggt den Spiel-Score separat unter einem eigenen TensorBoard-Tag
    (`rollout/score_mean`), statt ihn nur implizit unter dem generischen
    SB3-Tag `rollout/ep_rew_mean` mitlaufen zu lassen. Rechnerisch ist beides
    identisch: der Reward ist 0 in jedem Zug außer dem letzten, wo er genau
    dem Spiel-Score entspricht (siehe env.py), die Episodensumme des Rewards
    *ist* also der Score. Das eigene Tag macht das im TensorBoard-Dashboard
    aber sofort ohne Nachdenken ablesbar."""

    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0:
            scores = [ep_info["r"] for ep_info in self.model.ep_info_buffer]
            self.logger.record("rollout/score_mean", float(np.mean(scores)))
        return True


def git_commit_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return None


def make_env():
    return ActionMasker(TimeLimit(TakeItEasyEnv(), max_episode_steps=MAX_EPISODE_STEPS), mask_fn)


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MaskablePPO-Training für Take It Easy (Phase 6, lokal ausführen).")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--eval-episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", type=str, default="phase6_maskable_ppo")
    parser.add_argument(
        "--device", type=str, default="auto",
        help="'auto' (SB3 wählt), 'cuda' (GPU erzwingen) oder 'cpu'. "
             "Hinweis: bei einem so kleinen Netz (128,128) ist der "
             "Flaschenhals meist der CPU-seitige Env-Step, nicht das Netz "
             "selbst - GPU bringt hier oft wenig bis nichts, schadet aber "
             "auch nicht.",
    )
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    run_id = f"{timestamp[:16].replace(':', '').replace('T', '_')}_{args.tag}"
    run_dir = EXPERIMENTS_DIR / run_id
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    (run_dir / "tensorboard").mkdir(exist_ok=True)

    train_env = Monitor(make_env())
    eval_env = Monitor(make_env())

    model = MaskablePPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=1.0,  # keine Abzinsung - die Zielgröße ist der Score am Episodenende, alle 19 Züge zählen gleich (wie bei DQN, siehe train_dqn.py)
        gae_lambda=0.95,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=[128, 128]),
        tensorboard_log=str(run_dir / "tensorboard"),
        seed=args.seed,
        device=args.device,
        verbose=1,
    )
    print(f"Gewähltes Device: {model.device}")

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
            "learning_rate": 3e-4,
            "n_steps": 512,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 1.0,
            "gae_lambda": 0.95,
            "ent_coef": 0.01,
            "net_arch": [128, 128],
        },
        "eval_episodes": args.eval_episodes,
        "master_seed": args.seed,
        "device": str(model.device),
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
