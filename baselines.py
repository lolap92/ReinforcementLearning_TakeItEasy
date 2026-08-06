"""
Baseline-Agenten für Take It Easy: Random Agent und Greedy-Heuristik.

Liefert Referenz-Score-Verteilungen (Mittelwert, Std, Min/Max) über viele
Episoden, um spätere RL-Agenten (DQN, PPO) einordnen zu können.

Nutzung:
    python baselines.py --episodes 1000
"""

import argparse
import json

import numpy as np

from env import TakeItEasyEnv
from game import score_board


def random_agent(env, info, rng):
    valid_actions = np.where(info["action_mask"])[0]
    return rng.choice(valid_actions)


def greedy_agent(env, info, rng):
    """Wählt das Feld, das bei sofortiger Platzierung den größten
    Score-Zuwachs erzeugt (score_board auf dem resultierenden Board)."""
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


def run_episode(env, agent_fn, rng, seed):
    obs, info = env.reset(seed=seed)
    terminated = False
    total_reward = 0.0
    while not terminated:
        action = agent_fn(env, info, rng)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
    return total_reward


def evaluate(agent_fn, n_episodes, seed):
    env = TakeItEasyEnv()
    rng = np.random.default_rng(seed)
    episode_seeds = rng.integers(0, 2**31 - 1, size=n_episodes)
    scores = np.array([
        run_episode(env, agent_fn, rng, seed=int(s)) for s in episode_seeds
    ])
    return scores


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline-Agenten für Take It Easy evaluieren.")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="baseline_results.json")
    args = parser.parse_args()

    results = []
    random_scores = evaluate(random_agent, args.episodes, seed=args.seed)
    results.append(summarize("Random Agent", random_scores))

    greedy_scores = evaluate(greedy_agent, args.episodes, seed=args.seed)
    results.append(summarize("Greedy Heuristik", greedy_scores))

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nErgebnisse gespeichert in {args.output}")
