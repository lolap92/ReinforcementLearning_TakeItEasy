"""
Hindsight-Orakel: der beste Score, der in einer konkreten Partie möglich gewesen wäre.

Wozu
----
Der Afterstate-Agent aus Phase 8 kommt auf Ø 160,4. Die offene Frage ist, wie
viel davon noch Spielfehler sind und wie viel schlicht Kachelpech. Das
theoretische Maximum von 307 hilft dabei nicht: es gilt nur, wenn man sich
alle 19 Kacheln frei aus dem 27er-Deck aussuchen darf (dieser Modus ist
exakt gelöst - 307, erreicht von genau 16 Boards, 8 davon verschieden bis auf
die 180°-Rotation).

Das Orakel beantwortet stattdessen die episodenweise Frage:

    Gegeben *genau die 19 Kacheln*, die in dieser Partie gezogen wurden -
    was wäre der bestmögliche Score bei freier Platzierung gewesen?

Warum die Ziehreihenfolge dabei egal ist
----------------------------------------
Das Orakel bekommt nur die *Menge* der 19 gezogenen Kacheln, nicht ihre
Reihenfolge - und das ist kein Schlampigkeitsfehler, sondern exakt richtig.
In Take It Easy darf jede Kachel auf jedes freie Feld. Nimm ein beliebiges
Zielboard B (eine Bijektion Kacheln -> Felder) und eine beliebige
Ziehreihenfolge t1..t19: lege t_i einfach auf B(t_i). Das ist immer legal,
weil B injektiv ist und B(t_i) damit nicht unter den schon belegten Feldern
B(t1)..B(t_{i-1}) sein kann. Jede Bijektion ist also in jeder Reihenfolge
realisierbar - man müsste nur die Zukunft kennen. Das Orakel ist damit exakt
der Wert des hellsehenden Spielers.

Was der Wert aussagt - und was nicht
------------------------------------
Der Abstand zum Maximum zerfällt in DREI Teile, nicht zwei:

    307    − Orakel     diese Ziehung gibt nicht mehr her        -> nie holbar
    Orakel − V*         Preis des Online-Spielens: man muss legen,
                        bevor man die nächsten Kacheln kennt     -> von KEINER
                                                                    Online-Policy
                                                                    holbar
    V*     − Agent      echte Spielfehler                        -> holbar

V* ist die optimale Online-Policy (optimales Expectimax über die echte
Deck-Verteilung). Messbar sind hier nur `307 − Orakel` und die *Summe* der
beiden unteren Zeilen - ihre Aufteilung nicht, weil V* unbekannt ist (der
Zustandsraum ist zu groß, um es auszurechnen).

Praktisch heißt das: `Orakel − Agent` ist eine **obere Schranke dafür, wieviel
Suche und weiteres Training überhaupt noch bringen können**. Ist die Lücke
klein, lohnt sich Phase 9 nicht mehr. Ist sie groß, ist noch offen, wieviel
davon holbar ist - aber es ist zumindest möglich. Das Orakel selbst ist
ausdrücklich kein erreichbares Ziel; kein Agent kann es im Mittel einholen.

Engere Schranke wäre möglich, indem man dasselbe ILP mit nur k sichtbaren
Kacheln rechnet (Rest per Erwartungswert): je kleiner k, desto näher an V*.
Deutlich mehr Aufwand, deshalb hier nicht umgesetzt.

Wie
---
Exakt per ganzzahligem Programm (ILP, CBC über `pulp`), nicht per Heuristik:

  Variablen  x[c,t] = 1, falls Feld c die Kachel t bekommt (Permutation:
                     19 Felder, 19 Kacheln, jede genau einmal)
             y[l,v] = 1, falls Linie l komplett den Wert v trägt
  Ziel       maximiere  sum over l,v  y[l,v] * v * Länge(l)
  Kopplung   y[l,v] <= sum_{t mit Wert v}  x[c,t]     für jedes Feld c in l
             (disaggregiert, nicht als eine Summe - das ist die straffe Form
             und macht die LP-Relaxierung deutlich schärfer)

Zwei Verschärfungen, ohne die CBC 4- bis 5-mal länger braucht:

  1. Vorfilter: y[l,v] existiert nur, wenn überhaupt genug passende Kacheln
     gezogen wurden (eine Linie der Länge L braucht mindestens L Stück).
  2. Kardinalitätsschnitt je (Richtung, Wert): die 5 Linien einer Richtung
     partitionieren alle 19 Felder, also kann nicht mehr Fläche auf den Wert v
     gelegt werden, als es Kacheln mit diesem Wert gibt:
         sum_{l in Richtung d}  y[l,v] * Länge(l)  <=  Anzahl(d, v)

Zusätzlich bekommt CBC eine Startlösung aus Hill-Climbing (paarweise Tausche
mit Neustarts). Die ist meist 40-60 Punkte zu schlecht, schneidet aber sofort
einen großen Teil des Suchbaums weg.

Jede Lösung wird gegengeprüft: das zurückgegebene Board muss exakt aus den
gezogenen Kacheln bestehen, und `game.score_board()` muss unabhängig denselben
Wert liefern wie die ILP-Zielfunktion. Schlägt das fehl, bricht der Lauf ab -
eine falsche obere Schranke wäre schlimmer als gar keine.

Laufzeit: 5-15 Sekunden je Episode auf einem Kern, deshalb `--jobs` (Default:
alle Kerne) und eine Stichprobe statt aller 2000 Eval-Episoden.

Nutzung
-------
    pip install -r requirements.txt
    python oracle.py --episodes 200
    python oracle.py --episodes 200 --compare experiments/<run_id>/episodes.csv

`--compare` nimmt die `episodes.csv` eines Trainingslaufs und rechnet die
Zerlegung gepaart über dieselben Seeds aus - dafür müssen die Seeds
überlappen (alle Skripte nutzen `--seed + 1 + i`, also passt Default zu Default).

Ergebnisse landen wie überall unter `experiments/<run_id>/` plus eine Zeile in
`EXPERIMENTS.md`.
"""

import argparse
import csv
import json
import os
import random
import subprocess
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pulp

from env import TakeItEasyEnv
from game import ALL_LINE_GROUPS, score_board, NUM_CELLS

REPO_ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
EXPERIMENTS_LOG = REPO_ROOT / "EXPERIMENTS.md"

# Alle 15 Linien flach als (Wertindex in der Kachel, Feld-Indizes)
FLAT_LINES = [
    (value_pos, line)
    for _direction, (lines, value_pos) in ALL_LINE_GROUPS.items()
    for line in lines
]
DIRECTIONS = sorted({vp for vp, _ in FLAT_LINES})
MAX_SCORE = 307  # Optimum bei freier Kachelwahl, siehe Docstring


def drawn_tiles(seed):
    """Die 19 Kacheln, die `TakeItEasyEnv` bei diesem Seed zieht.

    env.reset() mischt das 27er-Deck und zieht per pop() vom Ende: die erste
    Kachel liegt schon in `current_tile`, die übrigen 18 sind die letzten 18
    Einträge des Reststapels. Die Reihenfolge ist für das Orakel irrelevant -
    es platziert ohnehin frei -, deshalb reicht die Menge.
    """
    env = TakeItEasyEnv()
    env.reset(seed=seed)
    tiles = [env.current_tile] + env.deck[-(NUM_CELLS - 1):]
    assert len(tiles) == NUM_CELLS and len(set(tiles)) == NUM_CELLS
    return tiles


def hill_climb(tiles, rng, restarts=6):
    """Startlösung für CBC: zufällige Belegung, dann steilster Abstieg über
    paarweise Feldtausche, mehrfach neu gestartet. Nicht besonders gut, aber
    in Millisekunden da und als untere Schranke schon viel wert."""
    best_score, best_board = -1, None
    for _ in range(restarts):
        board = list(tiles)
        rng.shuffle(board)
        score = score_board(board)[0]
        improved = True
        while improved:
            improved = False
            for i in range(NUM_CELLS):
                for j in range(i + 1, NUM_CELLS):
                    board[i], board[j] = board[j], board[i]
                    candidate = score_board(board)[0]
                    if candidate > score:
                        score, improved = candidate, True
                    else:
                        board[i], board[j] = board[j], board[i]
        if score > best_score:
            best_score, best_board = score, list(board)
    return best_score, best_board


def solve_oracle(tiles, warm_board=None, time_limit=600):
    """Exaktes Optimum über alle Platzierungen dieser 19 Kacheln.

    Rückgabe: (score, board, proven_optimal). `proven_optimal` ist False,
    wenn CBC ins Zeitlimit gelaufen ist - dann ist `score` nur die beste
    gefundene Lösung, also eine *untere* Schranke für die obere Schranke.
    """
    counts = {
        (d, v): sum(1 for t in tiles if t[d] == v)
        for d in DIRECTIONS
        for v in {t[d] for t in tiles}
    }

    problem = pulp.LpProblem("takeiteasy_hindsight_oracle", pulp.LpMaximize)
    x = {
        (c, t): pulp.LpVariable(f"x_{c}_{t}", cat="Binary")
        for c in range(NUM_CELLS)
        for t in range(NUM_CELLS)
    }
    y = {}
    for li, (value_pos, line) in enumerate(FLAT_LINES):
        for v in {t[value_pos] for t in tiles}:
            if counts[(value_pos, v)] >= len(line):
                y[(li, v)] = pulp.LpVariable(f"y_{li}_{v}", cat="Binary")

    problem += pulp.lpSum(y[(li, v)] * v * len(FLAT_LINES[li][1]) for (li, v) in y)

    for c in range(NUM_CELLS):
        problem += pulp.lpSum(x[(c, t)] for t in range(NUM_CELLS)) == 1
    for t in range(NUM_CELLS):
        problem += pulp.lpSum(x[(c, t)] for c in range(NUM_CELLS)) == 1

    for li, (value_pos, line) in enumerate(FLAT_LINES):
        values = [v for (l, v) in y if l == li]
        if values:
            problem += pulp.lpSum(y[(li, v)] for v in values) <= 1
        for v in values:
            matching = [t for t in range(NUM_CELLS) if tiles[t][value_pos] == v]
            for c in line:
                problem += y[(li, v)] <= pulp.lpSum(x[(c, t)] for t in matching)

    for d in DIRECTIONS:
        for v in {t[d] for t in tiles}:
            terms = [
                y[(li, v)] * len(FLAT_LINES[li][1])
                for (li, vv) in y
                if vv == v and FLAT_LINES[li][0] == d
            ]
            if terms:
                problem += pulp.lpSum(terms) <= counts[(d, v)]

    if warm_board is not None:
        position = {tile: i for i, tile in enumerate(tiles)}
        for c in range(NUM_CELLS):
            for t in range(NUM_CELLS):
                x[(c, t)].setInitialValue(1 if position[warm_board[c]] == t else 0)

    status = problem.solve(pulp.PULP_CBC_CMD(
        msg=False, timeLimit=time_limit, threads=1, warmStart=warm_board is not None,
    ))

    board = [None] * NUM_CELLS
    for (c, t), var in x.items():
        value = var.value()
        if value is not None and value > 0.5:
            board[c] = tiles[t]

    score = int(round(pulp.value(problem.objective)))
    # Gegenprobe: eine falsche obere Schranke waere schlimmer als gar keine.
    if sorted(board) != sorted(tiles):
        raise RuntimeError("Orakel-Board nutzt nicht exakt die gezogenen Kacheln")
    recomputed = score_board(board)[0]
    if recomputed != score:
        raise RuntimeError(f"ILP-Zielwert {score} != score_board() {recomputed}")

    return score, board, pulp.LpStatus[status] == "Optimal"


def solve_episode(args):
    """Worker für den Prozesspool (muss auf Modulebene liegen, damit er
    picklebar ist)."""
    seed, time_limit = args
    tiles = drawn_tiles(seed)
    warm_score, warm_board = hill_climb(tiles, random.Random(seed))
    score, board, proven = solve_oracle(tiles, warm_board, time_limit)
    return {
        "seed": seed,
        "oracle": score,
        "heuristic": warm_score,
        "proven_optimal": proven,
        "board": [list(t) for t in board],
    }


def git_commit_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return None


def load_comparison(path, seeds):
    """Liest eine episodes.csv und gibt {seed: score} je Agent zurück -
    aber nur für Agenten, die alle gefragten Seeds abdecken."""
    per_agent = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            per_agent.setdefault(row["agent"], {})[int(row["seed"])] = float(row["score"])
    wanted = set(seeds)
    return {a: s for a, s in per_agent.items() if wanted <= set(s)}


def main():
    parser = argparse.ArgumentParser(
        description="Exaktes Hindsight-Orakel: bester möglicher Score je Episode."
    )
    parser.add_argument("--episodes", type=int, default=200,
                        help="Anzahl Episoden (Stichprobe, jede kostet 5-15 s CPU).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Master-Seed. Ausgewertet werden die Episoden-Seeds "
                             "seed+1 .. seed+episodes - dieselbe Konvention wie in "
                             "train_ppo.py und train_afterstate.py.")
    parser.add_argument("--jobs", type=int, default=os.cpu_count(),
                        help="Parallele Prozesse (Default: alle Kerne).")
    parser.add_argument("--time-limit", type=float, default=600,
                        help="Sekunden je Episode für CBC. Wird das erreicht, gilt "
                             "die Episode als nicht bewiesen optimal und taucht in "
                             "der Zusammenfassung gesondert auf.")
    parser.add_argument("--compare", type=str, default=None,
                        help="Pfad zu einer episodes.csv (z.B. eines "
                             "train_afterstate.py-Laufs). Rechnet die Zerlegung "
                             "gepaart über die gemeinsamen Seeds aus.")
    parser.add_argument("--tag", type=str, default="hindsight_oracle")
    args = parser.parse_args()

    seeds = [args.seed + 1 + i for i in range(args.episodes)]
    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    run_id = f"{timestamp[:16].replace(':', '').replace('T', '_')}_{args.tag}"
    run_dir = EXPERIMENTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exaktes Orakel für {args.episodes} Episoden (Seeds {seeds[0]}..{seeds[-1]}), "
          f"{args.jobs} Prozesse ...")
    results = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for i, result in enumerate(pool.map(solve_episode, [(s, args.time_limit) for s in seeds]), 1):
            results.append(result)
            if i % 25 == 0 or i == len(seeds):
                done = np.array([r["oracle"] for r in results])
                print(f"  {i:4d}/{len(seeds)}  Orakel Ø {done.mean():6.2f}")

    results.sort(key=lambda r: r["seed"])
    oracle_scores = np.array([r["oracle"] for r in results], dtype=float)
    heuristic_scores = np.array([r["heuristic"] for r in results], dtype=float)
    unproven = [r["seed"] for r in results if not r["proven_optimal"]]

    print(f"\nOrakel:     Ø {oracle_scores.mean():.2f}  std {oracle_scores.std():.2f}  "
          f"min {oracle_scores.min():.0f}  max {oracle_scores.max():.0f}")
    print(f"Hill-Climb: Ø {heuristic_scores.mean():.2f} (nur die Startlösung für CBC)")
    if unproven:
        print(f"ACHTUNG: {len(unproven)} Episoden im Zeitlimit abgebrochen, "
              f"nicht bewiesen optimal: {unproven[:10]}")
    else:
        print(f"Alle {len(results)} Episoden bewiesen optimal.")

    comparisons = {}
    if args.compare:
        by_seed = {r["seed"]: r["oracle"] for r in results}
        print(f"\nZerlegung gegen {args.compare}:")
        print(f"{'Agent':38s} {'Ø Agent':>9s} {'Ø Orakel':>9s} {'Lücke':>8s} "
              f"{'% vom Orakel':>13s} {'Ziehung':>8s}")
        for agent, scores in load_comparison(args.compare, seeds).items():
            agent_scores = np.array([scores[s] for s in seeds])
            gap = oracle_scores - agent_scores
            comparisons[agent] = {
                "agent_mean": float(agent_scores.mean()),
                "oracle_mean": float(oracle_scores.mean()),
                "gap_to_oracle": float(gap.mean()),
                "pct_of_oracle": float(agent_scores.mean() / oracle_scores.mean() * 100),
                "gap_oracle_to_max": float(MAX_SCORE - oracle_scores.mean()),
            }
            print(f"{agent:38s} {agent_scores.mean():9.2f} {oracle_scores.mean():9.2f} "
                  f"{gap.mean():8.2f} {comparisons[agent]['pct_of_oracle']:12.1f}% "
                  f"{MAX_SCORE - oracle_scores.mean():8.2f}")
        print(f"\n'Ziehung' = {MAX_SCORE} minus Orakel: durch die Kachelziehung selbst "
              f"verloren, von keiner Policy holbar.")
        print("'Luecke'  = Orakel minus Agent. ACHTUNG: das ist eine OBERE SCHRANKE fuer das,")
        print("            was Suche/Training noch holen koennen - sie enthaelt auch den Preis")
        print("            des Online-Spielens, den keine Policy vermeiden kann (siehe Docstring).")

    config = {
        "run_id": run_id,
        "tag": args.tag,
        "timestamp": timestamp,
        "git_commit": git_commit_hash(),
        "env": "TakeItEasyEnv",
        "algorithm": "Hindsight-Orakel (exaktes ILP, CBC via pulp)",
        "episodes": args.episodes,
        "master_seed": args.seed,
        "episode_seeds": [seeds[0], seeds[-1]],
        "time_limit_per_episode": args.time_limit,
        "jobs": args.jobs,
        "all_proven_optimal": not unproven,
        "unproven_seeds": unproven,
        "compare": args.compare,
        "comparisons": comparisons,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    summary = [{
        "name": "hindsight_oracle",
        "n_episodes": args.episodes,
        "mean": float(oracle_scores.mean()),
        "std": float(oracle_scores.std()),
        "min": float(oracle_scores.min()),
        "max": float(oracle_scores.max()),
    }]
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(run_dir / "episodes.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["agent", "episode_index", "seed", "score"])
        for i, r in enumerate(results):
            writer.writerow(["hindsight_oracle", i, r["seed"], float(r["oracle"])])

    with open(run_dir / "oracle_boards.json", "w") as f:
        f.write("[\n")
        f.write(",\n".join(
            json.dumps({"seed": r["seed"], "score": r["oracle"], "board": r["board"]},
                       separators=(",", ":"))
            for r in results
        ))
        f.write("\n]\n")

    is_new = not EXPERIMENTS_LOG.exists()
    with open(EXPERIMENTS_LOG, "a") as f:
        if is_new:
            f.write("# Experiments\n\n")
            f.write("| Datum | Run-ID | Tag | Agent | Episoden | Mittelwert | Std | Min | Max | Commit |\n")
            f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for s in summary:
            f.write(
                f"| {timestamp[:10]} | `{run_id}` | {args.tag} | {s['name']} | "
                f"{s['n_episodes']} | {s['mean']:.2f} | {s['std']:.2f} | "
                f"{s['min']:.0f} | {s['max']:.0f} | {config['git_commit'] or '-'} |\n"
            )

    print(f"\nRun gespeichert unter: {run_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
