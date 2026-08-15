"""
Tabular Q-Learning auf dem Mini-Board (Phase 4).

Kernidee: eine Tabelle Q[state][action] wird direkt mit der Bellman-Update-
Regel aktualisiert - kein neuronales Netz, keine Funktionsapproximation.
Das macht die Lernregel Schritt für Schritt nachvollziehbar (siehe die
Ausgabe von print_manual_trace() weiter unten für eine komplette
Beispielrechnung).

Update-Regel (Q-Learning, off-policy TD(0)):

    Q(s,a) <- Q(s,a) + alpha * ( r + gamma * max_a' Q(s',a') - Q(s,a) )

    alpha            Lernrate: wie stark ein einzelnes Update die Tabelle
                     verändert (0 = nichts wird gelernt, 1 = alter Wert wird
                     komplett verworfen)
    gamma            Discount-Faktor: wie stark zukünftige Rewards zählen
                     (hier 1.0 - die Episode hat nur 3 kurze Züge, keine
                     Abzinsung nötig)
    max_a' Q(s',a')  bester geschätzter Wert des Folgezustands s'
                     (0, falls s' terminal ist - danach gibt es keine
                     weiteren Züge/Rewards mehr)
    r + gamma*max_a'Q(s',a') - Q(s,a)   der "TD-Error": Differenz zwischen
                     der neuen Schätzung (basierend auf dem tatsächlich
                     beobachteten r und dem aktuellen Wissen über s') und
                     der alten Schätzung Q(s,a)

Exploration/Exploitation: epsilon-greedy - mit Wahrscheinlichkeit epsilon
eine zufällige gültige Aktion, sonst die mit dem höchsten Q-Wert. epsilon
sinkt linear von eps_start auf eps_end über das Training (anfangs viel
Zufall zum Erkunden, am Ende fast nur noch die gelernte beste Aktion).
"""

import argparse
import csv
import json
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from mini_env import MiniTakeItEasyEnv, CELLS, VALUES
from mini_optimal import expected_optimal_score, optimal_action_values

REPO_ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
EXPERIMENTS_LOG = REPO_ROOT / "EXPERIMENTS.md"


def valid_actions_from_state(state):
    a, b, c, _tile = state
    return [i for i, v in enumerate((a, b, c)) if v == 0]


def argmax_random_tiebreak(valid, q_values, rng):
    """max(valid, key=...) würde bei Gleichstand immer die erste Option
    (kleinster Index, also A vor B vor C) bevorzugen. Gerade am Anfang des
    Trainings, wenn fast alle Q-Werte noch 0 sind, entsteht daraus ein
    systematischer Vorteil für frühe Aktionen, der sich selbst verstärkt
    (siehe Report) - obwohl B und C strukturell gleichwertig sind. Hier wird
    bei Gleichstand stattdessen zufällig unter den bestplatzierten Aktionen
    gewählt."""
    best_value = max(q_values[a] for a in valid)
    best_actions = [a for a in valid if q_values[a] == best_value]
    return int(rng.choice(best_actions))


def epsilon_greedy(q_table, state, epsilon, rng):
    valid = valid_actions_from_state(state)
    if rng.random() < epsilon:
        return int(rng.choice(valid))
    return argmax_random_tiebreak(valid, q_table[state], rng)


def train(episodes, alpha, gamma, eps_start, eps_end, seed):
    env = MiniTakeItEasyEnv()
    rng = np.random.default_rng(seed)
    q_table = defaultdict(lambda: np.zeros(3))
    visit_counts = defaultdict(int)  # wie oft musste in diesem Zustand entschieden werden

    episode_rewards = np.empty(episodes)
    for ep in range(episodes):
        progress = ep / max(1, episodes - 1)
        epsilon = eps_start + (eps_end - eps_start) * progress

        state = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        terminated = False
        total_reward = 0.0

        while not terminated:
            visit_counts[state] += 1
            action = epsilon_greedy(q_table, state, epsilon, rng)
            next_state, reward, terminated = env.step(action)

            best_next = 0.0 if terminated else max(
                q_table[next_state][a] for a in valid_actions_from_state(next_state)
            )
            td_target = reward + gamma * best_next
            td_error = td_target - q_table[state][action]
            q_table[state][action] += alpha * td_error

            state = next_state
            total_reward += reward

        episode_rewards[ep] = total_reward

    return q_table, episode_rewards, visit_counts


def evaluate_policy(q_table, n_episodes, seed):
    """Greedy (epsilon=0) Auswertung der gelernten Politik."""
    env = MiniTakeItEasyEnv()
    rng = np.random.default_rng(seed)
    scores = np.empty(n_episodes)
    for i in range(n_episodes):
        state = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        terminated = False
        total = 0.0
        while not terminated:
            valid = valid_actions_from_state(state)
            action = argmax_random_tiebreak(valid, q_table[state], rng)
            state, reward, terminated = env.step(action)
            total += reward
        scores[i] = total
    return scores


def evaluate_random(n_episodes, seed):
    env = MiniTakeItEasyEnv()
    rng = np.random.default_rng(seed)
    scores = np.empty(n_episodes)
    for i in range(n_episodes):
        state = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        terminated = False
        total = 0.0
        while not terminated:
            action = int(rng.choice(valid_actions_from_state(state)))
            state, reward, terminated = env.step(action)
            total += reward
        scores[i] = total
    return scores


def print_manual_trace(q_table, alpha, gamma, seed):
    """Eine Beispiel-Episode Schritt für Schritt mit der ausgerechneten
    Update-Formel - macht sichtbar, was 'lernen' bei Q-Learning konkret
    bedeutet: eine Zahl in einer Tabelle wird ein kleines Stück verschoben."""
    print("\n" + "=" * 78)
    print("Manuelle Nachvollziehbarkeit: eine Beispiel-Episode, Schritt für Schritt")
    print("=" * 78)
    env = MiniTakeItEasyEnv()
    tie_rng = np.random.default_rng(seed)
    state = env.reset(seed=seed)
    terminated = False
    step_num = 0

    while not terminated:
        step_num += 1
        valid = valid_actions_from_state(state)
        q_values = q_table[state].copy()
        action = argmax_random_tiebreak(valid, q_values, tie_rng)
        cell = CELLS[action]

        print(f"\n--- Zug {step_num} ---")
        print(f"Zustand s = {state}   (Feld A, Feld B, Feld C, aktuelle Kachel)")
        print(f"Q(s,.) = A:{q_values[0]:.3f}  B:{q_values[1]:.3f}  C:{q_values[2]:.3f}"
              f"   (gültige Felder: {[CELLS[a] for a in valid]})")
        print(f"-> gewählte Aktion (greedy = höchster Q-Wert unter den gültigen): Feld {cell}")

        old_q = q_table[state][action]
        next_state, reward, terminated = env.step(action)

        best_next = 0.0 if terminated else max(
            q_table[next_state][a] for a in valid_actions_from_state(next_state)
        )
        td_target = reward + gamma * best_next
        td_error = td_target - old_q
        new_q = old_q + alpha * td_error

        print(f"Beobachteter Reward r = {reward}"
              + ("  (Episode endet hier -> kein Folgezustand, max_a'Q(s',a')=0)" if terminated else ""))
        if not terminated:
            print(f"Folgezustand s' = {next_state},  max_a' Q(s',a') = {best_next:.3f}")
        print(
            f"Update:  Q(s,a) <- {old_q:.3f} + {alpha} * "
            f"({reward:g} + {gamma}*{best_next:.3f} - {old_q:.3f})\n"
            f"                <- {old_q:.3f} + {alpha} * {td_error:.3f}\n"
            f"                =  {new_q:.3f}"
        )
        q_table[state][action] = new_q
        state = next_state

    print("=" * 78)


def git_commit_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return None


def q_table_to_json(q_table):
    return {
        f"{a},{b},{c},{t}": q_table[(a, b, c, t)].round(4).tolist()
        for (a, b, c, t) in sorted(q_table.keys())
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tabular Q-Learning auf dem Mini-Board (Phase 4).")
    parser.add_argument("--episodes", type=int, default=100000)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eval-episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", type=str, default="phase4_qlearning_mini")
    args = parser.parse_args()

    print(f"Trainiere Q-Learning-Agent auf dem Mini-Board ({args.episodes} Episoden) ...")
    q_table, training_rewards, visit_counts = train(
        args.episodes, args.alpha, args.gamma, args.eps_start, args.eps_end, args.seed
    )

    eval_scores = evaluate_policy(q_table, args.eval_episodes, seed=args.seed + 1)
    random_scores = evaluate_random(args.eval_episodes, seed=args.seed + 2)
    optimal_mean = expected_optimal_score()

    print(f"\nRandom (Mini-Board):    mean={random_scores.mean():.3f}  std={random_scores.std():.3f}")
    print(f"Q-Learning (gelernt):   mean={eval_scores.mean():.3f}  std={eval_scores.std():.3f}")
    print(f"Optimal (exakt, DP):    mean={optimal_mean:.4f}")

    match_rng = np.random.default_rng(args.seed + 12345)
    matches, total = 0, 0
    weighted_matches, weighted_total = 0, 0
    for state, q_values in q_table.items():
        a, b, c, tile = state
        valid = valid_actions_from_state(state)
        learned_action = argmax_random_tiebreak(valid, q_values, match_rng)
        # Gegen ALLE optimalen Aktionen prüfen, nicht nur eine einzelne
        # Referenzaktion - bei B/C ist Gleichstand die exakt richtige
        # Antwort, kein Fehler, den man mit einem willkürlichen Tiebreak
        # der Referenz bestrafen sollte.
        opt_values = optimal_action_values((a, b, c), tile)
        best_opt_value = max(opt_values.values())
        is_match = int(abs(opt_values[learned_action] - best_opt_value) < 1e-9)
        total += 1
        matches += is_match
        visits = visit_counts.get(state, 0)
        weighted_total += visits
        weighted_matches += visits * is_match
    policy_match_rate = matches / total if total else 0.0
    weighted_match_rate = weighted_matches / weighted_total if weighted_total else 0.0
    print(f"Gelernte Politik stimmt in {matches}/{total} besuchten Zuständen mit dem Optimum überein "
          f"({policy_match_rate:.1%})")
    print(f"Besuchsgewichtet (wie oft die richtige Entscheidung in der Praxis wirklich getroffen wird): "
          f"{weighted_match_rate:.1%}")

    print_manual_trace(q_table, args.alpha, args.gamma, seed=args.seed + 999)

    # --- experiments/-Schema (siehe experiments/README.md) ---
    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    run_id = f"{timestamp[:16].replace(':', '').replace('T', '_')}_{args.tag}"
    run_dir = EXPERIMENTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "run_id": run_id,
        "tag": args.tag,
        "timestamp": timestamp,
        "git_commit": git_commit_hash(),
        "env": "MiniTakeItEasyEnv",
        "algorithm": "tabular_q_learning",
        "hyperparameters": {
            "episodes": args.episodes,
            "alpha": args.alpha,
            "gamma": args.gamma,
            "eps_start": args.eps_start,
            "eps_end": args.eps_end,
        },
        "eval_episodes": args.eval_episodes,
        "master_seed": args.seed,
        "policy_match_rate_vs_optimal": policy_match_rate,
        "policy_match_rate_vs_optimal_visit_weighted": weighted_match_rate,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    summary = [
        {"name": "random_mini", "n_episodes": args.eval_episodes,
         "mean": float(random_scores.mean()), "std": float(random_scores.std()),
         "min": float(random_scores.min()), "max": float(random_scores.max())},
        {"name": "qlearning_mini", "n_episodes": args.eval_episodes,
         "mean": float(eval_scores.mean()), "std": float(eval_scores.std()),
         "min": float(eval_scores.min()), "max": float(eval_scores.max())},
        {"name": "optimal_mini", "n_episodes": None,
         "mean": float(optimal_mean), "std": 0.0, "min": None, "max": None},
    ]
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(run_dir / "qtable.json", "w") as f:
        json.dump(q_table_to_json(q_table), f, indent=2)

    # Gleitender Durchschnitt fürs Lernkurven-Plotten - nur ca. 500 Stützpunkte
    # speichern statt jede einzelne Trainings-Episode (bei 100k+ Episoden wäre
    # die rohe Zeile-pro-Episode-Version mehrere MB groß, ohne dass eine
    # geglättete Kurve davon optisch profitiert; siehe experiments/README.md).
    window = 200
    curve_path = run_dir / "training_curve.csv"
    cumsum = np.cumsum(np.insert(training_rewards, 0, 0))
    moving_avg = (cumsum[window:] - cumsum[:-window]) / window
    stride = max(1, len(moving_avg) // 500)
    with open(curve_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "moving_avg_200"])
        for i in range(0, len(moving_avg), stride):
            ep = i + window - 1
            writer.writerow([ep, round(float(moving_avg[i]), 4)])
        last = len(moving_avg) - 1
        if last % stride != 0:
            writer.writerow([last + window - 1, round(float(moving_avg[last]), 4)])

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
                f"{s['n_episodes'] if s['n_episodes'] else '-'} | {s['mean']:.2f} | "
                f"{s['std']:.2f} | {s['min'] if s['min'] is not None else '-'} | "
                f"{s['max'] if s['max'] is not None else '-'} | {config['git_commit'] or '-'} |\n"
            )

    print(f"\nRun gespeichert unter: {run_dir.relative_to(REPO_ROOT)}")
    print(f"Eintrag ergänzt in: {EXPERIMENTS_LOG.relative_to(REPO_ROOT)}")
