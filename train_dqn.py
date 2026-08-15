"""
DQN via Stable-Baselines3 auf dem vollen Take-It-Easy-Board (Phase 5).

Für lokale Ausführung gedacht: das Training braucht deutlich mehr Rechenzeit
als die vorigen Phasen (zehntausende bis Millionen Environment-Schritte),
und TensorBoard live mitzuverfolgen funktioniert nur mit einem lokal
laufenden Prozess. Siehe Setup-Hinweise ganz unten in dieser Datei.

Design-Entscheidung: KEIN Action Masking in dieser Phase. Das ist bewusst
so: env.py sieht Masking (`info["action_mask"]`) ausdrücklich erst für
MaskablePPO in Phase 6 vor. Hier verlässt sich der Agent stattdessen auf
die in env.py eingebaute Bestrafung (-10 Reward, Episode läuft mit
derselben Kachel weiter) für ungültige Züge auf ein bereits belegtes Feld.
Das kostet am Anfang Trainingsbudget, weil der Agent das Vermeiden
belegter Felder erst lernen muss statt es strukturell garantiert zu
bekommen - genau der Unterschied, den Phase 6 (Value-based ohne Masking
vs. Policy-based mit Masking) sichtbar machen soll.

Nutzung (lokal, in deiner IDE):
    pip install -r requirements.txt
    python train_dqn.py --timesteps 300000

    # In einem zweiten Terminal, um live mitzuverfolgen:
    tensorboard --logdir experiments
    # dann im Browser: http://localhost:6006

Nach dem Training landet alles unter experiments/<run_id>/:
  - config.json, summary.json          -> ins Git-Repo committen
  - models/, tensorboard/              -> bewusst NICHT im Repo (.gitignore),
                                           lokal reproduzierbar aus config.json
                                           (gleicher Seed + gleicher Git-Commit)
"""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from env import TakeItEasyEnv

REPO_ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
EXPERIMENTS_LOG = REPO_ROOT / "EXPERIMENTS.md"

# Sicherheitsnetz: ohne Action Masking könnte ein schlecht trainierter Agent
# theoretisch endlos dieselbe ungültige Aktion wiederholen (Episode läuft ja
# mit derselben Kachel weiter). Bricht die Auswertungs-Episode stattdessen
# mit Score 0 ab.
MAX_STEPS_PER_EVAL_EPISODE = 200


def git_commit_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return None


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
        while not terminated and steps < MAX_STEPS_PER_EVAL_EPISODE:
            action, _ = model.predict(obs, deterministic=True)
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
    parser = argparse.ArgumentParser(description="DQN-Training für Take It Easy (Phase 5, lokal ausführen).")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--eval-episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", type=str, default="phase5_dqn")
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    run_id = f"{timestamp[:16].replace(':', '').replace('T', '_')}_{args.tag}"
    run_dir = EXPERIMENTS_DIR / run_id
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    (run_dir / "tensorboard").mkdir(exist_ok=True)

    train_env = Monitor(TakeItEasyEnv())
    eval_env = Monitor(TakeItEasyEnv())

    model = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=5e-4,
        buffer_size=100_000,
        learning_starts=1_000,
        batch_size=64,
        gamma=1.0,  # keine Abzinsung - die Zielgröße ist der Score am Episodenende, alle 19 Züge zählen gleich
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1_000,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        policy_kwargs=dict(net_arch=[128, 128]),
        tensorboard_log=str(run_dir / "tensorboard"),
        seed=args.seed,
        verbose=1,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "models"),
        log_path=str(run_dir / "tensorboard"),
        eval_freq=10_000,
        n_eval_episodes=50,
        deterministic=True,
    )

    print(f"Trainiere DQN für {args.timesteps} Timesteps ...")
    model.learn(total_timesteps=args.timesteps, callback=eval_callback, tb_log_name="dqn")
    model.save(str(run_dir / "models" / "final_model"))

    print("\nAuswertung über echte Take-It-Easy-Episoden (19 Züge, volles Board) ...")
    scores, invalid_counts = evaluate(model, args.eval_episodes, seed=args.seed + 1)
    print(f"DQN (gelernt):  mean={scores.mean():.2f}  std={scores.std():.2f}  "
          f"min={scores.min():.0f}  max={scores.max():.0f}")
    print(f"Ungültige Züge je Episode (Ø, sollte gegen 0 gehen): {invalid_counts.mean():.2f}")

    config = {
        "run_id": run_id,
        "tag": args.tag,
        "timestamp": timestamp,
        "git_commit": git_commit_hash(),
        "env": "TakeItEasyEnv",
        "algorithm": "DQN (stable-baselines3, ohne Action Masking)",
        "hyperparameters": {
            "timesteps": args.timesteps,
            "learning_rate": 5e-4,
            "buffer_size": 100_000,
            "learning_starts": 1_000,
            "batch_size": 64,
            "gamma": 1.0,
            "train_freq": 4,
            "target_update_interval": 1_000,
            "exploration_fraction": 0.3,
            "exploration_final_eps": 0.05,
            "net_arch": [128, 128],
        },
        "eval_episodes": args.eval_episodes,
        "master_seed": args.seed,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    summary = [{
        "name": "dqn_full_board",
        "n_episodes": args.eval_episodes,
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "min": float(scores.min()),
        "max": float(scores.max()),
        "avg_invalid_actions_per_episode": float(invalid_counts.mean()),
    }]
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

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
