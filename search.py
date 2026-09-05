"""
Phase 9: Expectimax-Suche zur Spielzeit statt reinem 1-Ply-argmax(V).

Wozu
----
Phase 8 (`train_afterstate.py`) lernt `V(Afterstate)` und spielt `argmax`
über die ≤19 möglichen Folgezustände - ein Forward-Pass pro Kandidat, dann
fertig. Der Phase-7-Report (Hebel 3) und `oracle.py`s Auswertung zeigen,
warum das nicht die letzte Ausbaustufe ist: auf denselben 200 Seeds wie der
Hindsight-Orakel-Lauf holt der 300k-Agent nur 63,6 % des episodenspezifischen
Optimums (Ø 158,2 von Ø 248,75) - eine Lücke von über 90 Punkten, von der ein
Teil (siehe unten) durch bessere Entscheidungen zur Spielzeit holbar sein
sollte, nicht durch mehr Training (siehe `reports/phase8_scaling_report.html`
- 10x mehr Selbstspiel brachte nur +2,8 Punkte).

Weil die Dynamik des Spiels vollständig bekannt ist (Afterstate ist
deterministisch, die nächste Kachel ist exakt gleichverteilt über das
Restdeck - keine Duplikate in `build_deck()`), lässt sich der Erwartungswert
über mehrere Züge hinweg direkt ausrechnen statt ihn nur zu approximieren.
Das ist in `train_afterstate.py` implementiert als `AfterstateAgent(depth=...,
endgame_exact=...)`:

  --depth N          N zusätzliche (Zufall-, Maximum-)Schichten mit dem
                      gelernten Netz als Blattbewertung an der Sohle
                      ("2-Ply" bei N=1: aktueller Zug + eine Kachel/Zug-
                      Schicht vorausgerechnet statt geraten).
  --endgame-exact K  sobald nach dem Zug ≤K freie Felder übrig blieben,
                      wird EXAKT bis zum Spielende gesucht (score_board()
                      statt Netz an den Blättern) - unabhängig von --depth.

Kostenmodell (korrigiert eine zu optimistische Aussage im Phase-7-Report)
--------------------------------------------------------------------------
Das Deck hat 27 Kacheln, das Board nur 19 Felder - es werden nie mehr als 19
gezogen, das Restdeck hat also einen Boden von 27-19=8 Kacheln, der bis zum
letzten Zug nie unterschritten wird. Die Zufallsverzweigung wird deshalb NIE
klein, anders als bei einem Kartenspiel, das sein Deck leerspielt. Exakte
Suche bis zum Ende ist deshalb nur für die letzten ~4 Züge praktikabel
(≤4 freie Felder: ~24.000 Bewertungen, <1s), nicht für ~6 wie ursprünglich
vermutet (bei 5 freien Feldern bereits ~1,4 Mio., ~45s; bei 6 ~111 Mio.,
~1 Stunde - gemessen auf dieser Maschine, ~32.000 Zeilen/s). `--depth 2`
kostet allein für den ersten Zug einer Partie schon >2 Minuten (gemessen) -
auf CPU nicht praktikabel, deshalb Default `--depth 1`.

Nutzung
-------
    python search.py --model experiments/<run_id>/models/final_model.pt
    python search.py --model ... --depth 1 --endgame-exact 4 --episodes 200
    python search.py --model ... --depth 0 --endgame-exact 0   # = Phase 8, zur Kontrolle

Wertet wie `oracle.py` standardmäßig die ersten 200 Eval-Seeds aus
(`--seed 0` -> Seeds 1..200), weil das exakt die Seeds sind, für die
`oracle.py` schon das Hindsight-Optimum berechnet hat - die Ergebnisse sind
also direkt gepaart mit `experiments/*_phase9_hindsight_oracle/` und mit
jedem anderen Lauf vergleichbar, der dieselbe seed+1+i-Konvention nutzt
(train_afterstate.py, train_ppo.py, baselines.py laufen mit dieser
Konvention... `baselines.py` bildet eine Ausnahme, siehe dessen eigene
Zufalls-Seeds).

Ergebnisse landen wie überall unter `experiments/<run_id>/` (config.json,
summary.json, episodes.csv) plus eine Zeile in `EXPERIMENTS.md`.
"""

import argparse
import csv
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from train_afterstate import AfterstateAgent, ValueNet, evaluate

REPO_ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
EXPERIMENTS_LOG = REPO_ROOT / "EXPERIMENTS.md"


def load_net(model_path):
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    net = ValueNet(
        ckpt["in_dim"], hidden=tuple(ckpt["hidden"]), head=ckpt["head"], atoms=ckpt["atoms"],
    )
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, ckpt["line_features"]


def git_commit_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Phase 9: Expectimax-Suche zur Spielzeit über ein trainiertes Afterstate-Netz."
    )
    parser.add_argument("--model", required=True, help="Pfad zu einem .pt-Modell aus train_afterstate.py.")
    parser.add_argument("--depth", type=int, default=1,
                        help="Zusätzliche (Zufall-,Maximum-)Schichten mit dem Netz als Blattbewertung "
                             "(Default 1 = '2-Ply'). >=2 ist auf CPU nicht praktikabel, siehe Docstring.")
    parser.add_argument("--endgame-exact", type=int, default=4,
                        help="Ab wieviel freien Feldern nach dem Zug exakt bis zum Ende gesucht wird "
                             "(Default 4, siehe Kostenmodell im Docstring). 0 = aus.")
    parser.add_argument("--episodes", type=int, default=200,
                        help="Default 200 = dieselben Seeds wie der Hindsight-Orakel-Lauf (direkt paarbar).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", type=str, default=None)
    args = parser.parse_args()

    if args.tag is None:
        parts = [f"search_d{args.depth}_eg{args.endgame_exact}_{Path(args.model).parent.parent.name}"]
        args.tag = "_".join(parts)

    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    run_id = f"{timestamp[:16].replace(':', '').replace('T', '_')}_{args.tag}"
    run_dir = EXPERIMENTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Lade {args.model} ...")
    net, line_features = load_net(args.model)
    agent = AfterstateAgent(net, torch.device("cpu"), line_features,
                             depth=args.depth, endgame_exact=args.endgame_exact)

    print(f"Werte {args.episodes} Episoden aus (depth={args.depth}, endgame_exact={args.endgame_exact}, "
          f"Seeds {args.seed + 1}..{args.seed + args.episodes}) ...")
    started = time.time()
    scores = evaluate(agent, args.episodes, seed=args.seed)
    elapsed = time.time() - started

    print(f"\nØ {scores.mean():.2f}  std {scores.std():.2f}  min {scores.min():.0f}  max {scores.max():.0f}")
    print(f"Laufzeit: {elapsed:.1f}s gesamt, {elapsed/args.episodes:.2f}s/Episode")

    config = {
        "run_id": run_id,
        "tag": args.tag,
        "timestamp": timestamp,
        "git_commit": git_commit_hash(),
        "env": "TakeItEasyEnv",
        "algorithm": "Afterstate-Wertfunktion + Expectimax-Suche (Phase 9)",
        "model_path": str(args.model),
        "depth": args.depth,
        "endgame_exact": args.endgame_exact,
        "episodes": args.episodes,
        "master_seed": args.seed,
        "seconds_total": elapsed,
        "seconds_per_episode": elapsed / args.episodes,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    agent_name = f"search_d{args.depth}_eg{args.endgame_exact}"
    summary = [{
        "name": agent_name,
        "n_episodes": args.episodes,
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "min": float(scores.min()),
        "max": float(scores.max()),
    }]
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(run_dir / "episodes.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["agent", "episode_index", "seed", "score"])
        for i, score in enumerate(scores):
            writer.writerow([agent_name, i, args.seed + 1 + i, float(score)])

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
