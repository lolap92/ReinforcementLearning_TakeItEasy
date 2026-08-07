"""
Baseline-Agenten für Take It Easy: Random Agent und Greedy-Heuristik.

Liefert Referenz-Score-Verteilungen (Mittelwert, Std, Min/Max) über viele
Episoden, um spätere RL-Agenten (DQN, PPO) einordnen zu können.

Jeder Lauf wird vollständig unter experiments/<run_id>/ abgelegt (Config,
Kennzahlen, Score je Episode, Zug-für-Zug-Trajektorien einer Stichprobe)
und als Zeile in EXPERIMENTS.md protokolliert. Details zum Format:
siehe experiments/README.md.

Nutzung:
    python baselines.py --episodes 1000
"""

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from env import TakeItEasyEnv
from game import score_board

REPO_ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
EXPERIMENTS_LOG = REPO_ROOT / "EXPERIMENTS.md"


def random_agent(env, info, rng):
    valid_actions = np.where(info["action_mask"])[0]
    return rng.choice(valid_actions)


def greedy_agent(env, info, rng):
    """Wählt das Feld, das bei sofortiger Platzierung den größten
    Score-Zuwachs erzeugt (score_board() auf dem resultierenden Board)."""
    valid_actions = np.where(info["action_mask"])[0]

    best_score = -1
    best_actions = []
    for action in valid_actions:
        candidate_board = list(env.board)
        candidate_board[action] = env.current_tile
        score, _ = score_board(candidate_board)
        if score > best_score:
            best_score = score
            best_actions = [action]
        elif score == best_score:
            best_actions.append(action)

    return rng.choice(best_actions)


AGENTS = {
    "random": random_agent,
    "greedy": greedy_agent,
}


def run_episode(env, agent_fn, seed, record_trajectory=False):
    """Spielt eine Episode. `seed` bestimmt sowohl die Kachelreihenfolge
    (env.reset) als auch alle Zufallsentscheidungen des Agenten
    (eigener RNG) -> die Episode ist allein durch `seed` reproduzierbar."""
    obs, info = env.reset(seed=seed)
    agent_rng = np.random.default_rng(seed)

    terminated = False
    total_reward = 0.0
    trajectory = [] if record_trajectory else None
    step = 0

    while not terminated:
        action = agent_fn(env, info, agent_rng)
        tile = env.current_tile
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1
        if record_trajectory:
            trajectory.append({
                "step": step,
                "tile": list(tile),
                "action": int(action),
                "board_after": [list(t) if t is not None else None for t in env.board],
            })
        total_reward += reward

    return total_reward, trajectory


def evaluate(agent_fn, n_episodes, seed, n_trajectories=0):
    """Spielt n_episodes Episoden. Für die ersten n_trajectories wird die
    volle Zug-für-Zug-Trajektorie mitgeschnitten."""
    env = TakeItEasyEnv()
    master_rng = np.random.default_rng(seed)
    episode_seeds = master_rng.integers(0, 2**31 - 1, size=n_episodes)

    scores = np.empty(n_episodes)
    trajectories = []
    for i, ep_seed in enumerate(episode_seeds):
        record = i < n_trajectories
        score, trajectory = run_episode(env, agent_fn, seed=int(ep_seed), record_trajectory=record)
        scores[i] = score
        if record:
            trajectories.append({
                "episode_index": i,
                "seed": int(ep_seed),
                "final_score": float(score),
                "steps": trajectory,
            })

    return scores, episode_seeds, trajectories


def summarize(name, scores):
    print(
        f"{name:20s}  n={len(scores):5d}  "
        f"mean={scores.mean():7.2f}  std={scores.std():6.2f}  "
        f"min={scores.min():5.0f}  max={scores.max():5.0f}"
    )
    return {
        "name": name,
        "n_episodes": int(len(scores)),
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "min": float(scores.min()),
        "max": float(scores.max()),
    }


def git_commit_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return None


def write_trajectories_json(path, trajectories):
    """Eine Episode pro Zeile, kompakt (kein json.dump(indent=2)) - das würde
    für ein 19-Felder-Board jede einzelne Zahl auf eine eigene Zeile setzen
    und die Datei um ein Vielfaches aufblähen."""
    lines = [json.dumps(ep, separators=(",", ":")) for ep in trajectories]
    with open(path, "w") as f:
        f.write("[\n")
        f.write(",\n".join(lines))
        f.write("\n]\n")


def write_episodes_csv(path, agent_name, scores, episode_seeds):
    is_new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["agent", "episode_index", "seed", "score"])
        for i, (score, seed) in enumerate(zip(scores, episode_seeds)):
            writer.writerow([agent_name, i, int(seed), score])


def append_experiments_log(run_id, tag, config, summaries):
    is_new = not EXPERIMENTS_LOG.exists()
    with open(EXPERIMENTS_LOG, "a") as f:
        if is_new:
            f.write("# Experiments\n\n")
            f.write(
                "Ein Lauf pro Zeile. Details (Config, Kennzahlen, Trajektorien) "
                f"unter `experiments/<run_id>/`, Format siehe `experiments/README.md`.\n\n"
            )
            f.write("| Datum | Run-ID | Tag | Agent | Episoden | Mittelwert | Std | Min | Max | Commit |\n")
            f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for s in summaries:
            f.write(
                f"| {config['timestamp'][:10]} | `{run_id}` | {tag} | {s['name']} | "
                f"{s['n_episodes']} | {s['mean']:.2f} | {s['std']:.2f} | "
                f"{s['min']:.0f} | {s['max']:.0f} | {config['git_commit'] or '-'} |\n"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline-Agenten für Take It Easy evaluieren.")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-trajectories", type=int, default=20,
                         help="Anzahl Episoden pro Agent, für die die volle Zug-für-Zug-Trajektorie gespeichert wird.")
    parser.add_argument("--tag", type=str, default="phase3_baselines",
                         help="Kurzbezeichnung des Laufs, taucht im Run-Ordner-Namen und in EXPERIMENTS.md auf.")
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    run_id = f"{timestamp[:16].replace(':', '').replace('T', '_')}_{args.tag}"
    run_dir = EXPERIMENTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "run_id": run_id,
        "tag": args.tag,
        "timestamp": timestamp,
        "git_commit": git_commit_hash(),
        "env": "TakeItEasyEnv",
        "agents": list(AGENTS.keys()),
        "episodes_per_agent": args.episodes,
        "master_seed": args.seed,
        "n_trajectories_recorded": args.n_trajectories,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    episodes_csv_path = run_dir / "episodes.csv"
    trajectories_dir = run_dir / "sample_trajectories"

    summaries = []
    for name, agent_fn in AGENTS.items():
        scores, episode_seeds, trajectories = evaluate(
            agent_fn, args.episodes, seed=args.seed, n_trajectories=args.n_trajectories
        )
        summaries.append(summarize(name, scores))
        write_episodes_csv(episodes_csv_path, name, scores, episode_seeds)

        if trajectories:
            trajectories_dir.mkdir(exist_ok=True)
            write_trajectories_json(trajectories_dir / f"{name}.json", trajectories)

    with open(run_dir / "summary.json", "w") as f:
        json.dump(summaries, f, indent=2)

    append_experiments_log(run_id, args.tag, config, summaries)

    print(f"\nRun gespeichert unter: {run_dir.relative_to(REPO_ROOT)}")
    print(f"Eintrag ergänzt in: {EXPERIMENTS_LOG.relative_to(REPO_ROOT)}")
