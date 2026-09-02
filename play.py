"""
Gegen ein selbst trainiertes Netz spielen.

Format wie beim echten Mehrspieler-Take-It-Easy: beide Spieler bekommen
*dieselbe* Kachelfolge (gleicher Seed), jeder legt auf sein eigenes Brett, am
Ende werden die Scores verglichen. Das ist der fairste mögliche Vergleich -
Kachelglück fällt komplett raus, es zählt nur die Platzierung.

Praktischer Nebeneffekt: weil das Netz auf seinem eigenen Brett spielt, hängt
sein Spiel überhaupt nicht von deinem ab. Seine Partie wird deshalb komplett
vorab berechnet, bevor du deinen ersten Zug machst.

Nutzung
-------
    python play.py --model experiments/<run_id>/models/final_model.pt
    python play.py --model experiments/<run_id>/models/final_model.zip --algo ppo
    python play.py --model ... --seed 42          # feste Partie, wiederholbar
    python play.py --model ... --no-live          # ohne Browser-Ansicht
    python play.py --model ... --html             # zusätzlich Einzelbretter wie replay.py

Live-Ansicht
------------
Standardmäßig wird nach jedem Zug `replay/play_<seed>.html` neu geschrieben
und beim ersten Zug im Browser geöffnet: das Brett als echte Kachelgrafik
(board_render.py), mit Feldnummern in den freien Feldern, dem zuletzt
gelegten Feld umrandet und der aktuellen Kachel daneben. Die Seite lädt sich
per meta-refresh jede Sekunde selbst neu - kein Server nötig, das Ganze läuft
über file://. Am Ende wird die Refresh-Zeile weggelassen und stattdessen dein
Brett neben dem des Netzes gezeigt. `--no-live` schaltet das ab.

Gespielt wird trotzdem im Terminal - die Textausgabe bleibt der Ort, an dem
man Feldnummern eintippt, die Grafik ist zum Draufschauen.

Unterstützt beide Modellarten aus diesem Repo:
  *.pt   -> Afterstate-Wertfunktion (train_afterstate.py), --algo wird nicht gebraucht
  *.zip  -> MaskablePPO oder DQN (train_ppo.py / train_dqn.py), --algo nötig

Eingaben während des Spiels
--------------------------
    0-18   Feldnummer, auf die die aktuelle Kachel gelegt wird
    h      Hinweis: welches Feld würde das Netz jetzt nehmen?
    d      Restliche Kacheln im Stapel anzeigen
    q      Aufgeben und beenden

`models/` ist gitignored, das Modell muss also lokal aus dem jeweiligen
Trainingslauf noch vorhanden sein.
"""

import argparse
import os
import random
import shutil
import subprocess
import webbrowser
from pathlib import Path

import numpy as np

from env import TakeItEasyEnv
from game import ROWS, NUM_CELLS, score_board

REPO_ROOT = Path(__file__).resolve().parent
REPLAY_DIR = REPO_ROOT / "replay"
CELL_WIDTH = 15


# ---------------------------------------------------------------------------
# Modelle
# ---------------------------------------------------------------------------

def load_agent(model_path, algo=None):
    """Gibt eine Funktion act(env, obs, info) -> Feldindex zurück.

    Die beiden Modellarten brauchen unterschiedliche Eingaben (das
    Afterstate-Netz bewertet Folgezustände und liest dafür direkt Board,
    Restdeck und aktuelle Kachel aus der Env; SB3 arbeitet auf der
    Observation), deshalb hier eine gemeinsame Hülle.
    Die schweren Importe passieren absichtlich erst hier drin - sonst
    bräuchte ein PPO-Replay torch-für-Afterstate und umgekehrt.
    """
    path = Path(model_path)
    if not path.exists():
        raise SystemExit(
            f"Modell nicht gefunden: {path}\n"
            "models/ ist gitignored - das Modell muss lokal aus dem "
            "Trainingslauf noch vorhanden sein."
        )

    if path.suffix == ".pt":
        import torch
        from train_afterstate import ValueNet, AfterstateAgent

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        net = ValueNet(
            checkpoint["in_dim"],
            hidden=tuple(checkpoint["hidden"]),
            head=checkpoint["head"],
            atoms=checkpoint["atoms"],
        )
        net.load_state_dict(checkpoint["state_dict"])
        agent = AfterstateAgent(net, torch.device("cpu"), checkpoint["line_features"])
        return lambda env, obs, info: agent.act(env), "Afterstate-Wertfunktion"

    if algo is None:
        raise SystemExit("Für .zip-Modelle wird --algo dqn|ppo gebraucht.")
    if algo == "dqn":
        from train_dqn import MaskedDQN
        model = MaskedDQN.load(str(path))
        return (lambda env, obs, info: int(model.predict(obs, deterministic=True)[0])), "DQN"
    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(str(path))
    return (
        lambda env, obs, info: int(
            model.predict(obs, action_masks=info["action_mask"], deterministic=True)[0]
        ),
        "MaskablePPO",
    )


def play_agent_episode(act, seed):
    """Spielt die Partie des Netzes komplett vorab durch.

    Rückgabe: (board, score, moves) mit moves = [(Kachel, Feld), ...] in
    Zugreihenfolge - daraus lässt sich später der Hinweis-Befehl bedienen und
    das Endbrett anzeigen.
    """
    env = TakeItEasyEnv()
    obs, info = env.reset(seed=seed)
    moves = []
    terminated, score = False, 0.0
    while not terminated:
        tile = env.current_tile
        action = act(env, obs, info)
        moves.append((tile, action))
        obs, reward, terminated, _truncated, info = env.step(action)
        if terminated:
            score = reward
    return list(env.board), score, moves


# ---------------------------------------------------------------------------
# Textausgabe
# ---------------------------------------------------------------------------

def render_board(board, show_indices=True, highlight=None):
    """Brett als Text, in der Sechseck-Form des echten Spiels.

    Freie Felder zeigen ihre Feldnummer - ohne die könnte man nicht sagen,
    wohin man legen will. Geometrie identisch zu env.render(): 5 Spalten mit
    3-4-5-4-3 Feldern, versetzt gestapelt.
    """
    columns = ROWS
    max_height = max(len(col) for col in columns)
    grid = [[" " * CELL_WIDTH for _ in columns] for _ in range(2 * max_height - 1)]

    for col_idx, col_indices in enumerate(columns):
        offset = max_height - len(col_indices)
        for row_in_col, cell in enumerate(col_indices):
            tile = board[cell]
            if tile is not None:
                text = f"[{tile[0]},{tile[1]},{tile[2]}]"
                if cell == highlight:
                    text = f"*{text}*"
            elif show_indices:
                text = f"( {cell:2d} )"
            else:
                text = "(    )"
            grid[offset + 2 * row_in_col][col_idx] = text.center(CELL_WIDTH)

    return "\n".join("".join(line).rstrip() for line in grid)


def score_breakdown(board):
    total, details = score_board(board)
    complete = [d for d in details if d[3] > 0]
    return total, complete


# ---------------------------------------------------------------------------
# Live-Ansicht im Browser
# ---------------------------------------------------------------------------

def open_in_browser(path):
    """Chrome bevorzugt (wie in den Trainingsskripten), sonst Standardbrowser.
    Bewusst hier lokal statt aus train_ppo/train_afterstate importiert - die
    ziehen sb3 bzw. torch mit, was ein reines Spiel nicht braucht."""
    url = path.resolve().as_uri()
    candidates = [
        shutil.which("chrome"),
        shutil.which("google-chrome"),
        os.path.join(os.environ.get("PROGRAMFILES", r"C:\Program Files"),
                     r"Google\Chrome\Application\chrome.exe"),
        os.path.join(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"),
                     r"Google\Chrome\Application\chrome.exe"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""),
                     r"Google\Chrome\Application\chrome.exe"),
    ]
    chrome = next((c for c in candidates if c and Path(c).exists()), None)
    if chrome:
        try:
            subprocess.Popen([chrome, url])
            return
        except Exception:
            pass
    try:
        webbrowser.open(url)
    except Exception:
        pass


PAGE_STYLE = """
  * { box-sizing: border-box; }
  body { margin:0; background:#0e1f19; color:#e9efec;
         font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;
         padding:28px 20px 40px; }
  .wrap { max-width:1180px; margin:0 auto; display:flex; flex-direction:column; gap:22px; }
  header { display:flex; align-items:baseline; gap:16px; flex-wrap:wrap; }
  h1 { font-size:22px; font-weight:650; margin:0; letter-spacing:-0.01em; }
  .meta { font-size:13px; color:#8fa79c; font-variant-numeric:tabular-nums; }
  .cols { display:flex; gap:28px; align-items:flex-start; flex-wrap:wrap; justify-content:center; }
  .panel { display:flex; flex-direction:column; gap:10px; align-items:center; }
  .panel h2 { font-size:13px; font-weight:650; letter-spacing:0.06em; text-transform:uppercase;
              color:#8fa79c; margin:0; }
  .score { font-size:30px; font-weight:700; font-variant-numeric:tabular-nums; margin:0; }
  .tile-now { display:flex; align-items:center; gap:14px; background:#142b23;
              border:1px solid #23453a; border-radius:12px; padding:12px 18px; }
  .tile-now .label { font-size:13px; color:#8fa79c; }
  svg { max-width:100%; height:auto; display:block; }
  .verdict { font-size:19px; font-weight:650; text-align:center; padding:14px;
             background:#142b23; border:1px solid #23453a; border-radius:12px; }
  .hint { font-size:13px; color:#8fa79c; text-align:center; }
"""


def play_page(board, seed, step, tile, score, last_cell,
              agent_board=None, agent_score=None, agent_name="Netz"):
    """Die Live-Seite. Solange `agent_board` fehlt, läuft die Partie noch:
    dann lädt sich die Seite selbst nach und das Brett des Netzes bleibt
    verdeckt (sonst könnte man abschreiben)."""
    from board_render import board_to_svg, tile_svg

    running = agent_board is None
    refresh = '<meta http-equiv="refresh" content="1">' if running else ""

    if running:
        head = (
            f'<header><h1>Zug {step + 1} von 19</h1>'
            f'<span class="meta">Seed {seed} &middot; Gegner: {agent_name}</span></header>'
            f'<div class="tile-now">'
            f'<span class="label">Diese Kachel legen:</span>{tile_svg(*tile)}'
            f'<span class="label">Feldnummer im Terminal eingeben</span></div>'
        )
        panels = (
            f'<div class="panel"><h2>Dein Brett</h2>'
            f'<p class="score">{score:.0f}</p>'
            f'{board_to_svg(board, labels=True, highlight=last_cell)}</div>'
        )
        footer = ('<p class="hint">Freie Felder zeigen ihre Nummer. '
                  'Das gelb umrandete Feld war dein letzter Zug. '
                  'Diese Seite aktualisiert sich von selbst.</p>')
    else:
        diff = score - agent_score
        if diff > 0:
            verdict = f"Du gewinnst mit {diff:.0f} Punkten Vorsprung."
        elif diff < 0:
            verdict = f"{agent_name} gewinnt mit {-diff:.0f} Punkten Vorsprung."
        else:
            verdict = "Unentschieden."
        head = (f'<header><h1>Endstand</h1>'
                f'<span class="meta">Seed {seed} &middot; dieselben 19 Kacheln für beide</span></header>')
        panels = (
            f'<div class="panel"><h2>Dein Brett</h2><p class="score">{score:.0f}</p>'
            f'{board_to_svg(board, highlight=last_cell)}</div>'
            f'<div class="panel"><h2>{agent_name}</h2><p class="score">{agent_score:.0f}</p>'
            f'{board_to_svg(agent_board)}</div>'
        )
        footer = (f'<div class="verdict">{verdict}</div>'
                  f'<p class="hint">Dieselbe Partie nochmal: --seed {seed}</p>')

    return (f'<!doctype html>\n<html lang="de"><head><meta charset="utf-8">{refresh}'
            f'<title>Take It Easy - Seed {seed}</title><style>{PAGE_STYLE}</style></head>'
            f'<body><div class="wrap">{head}<div class="cols">{panels}</div>{footer}</div>'
            f'</body></html>')


# ---------------------------------------------------------------------------
# Spielschleife
# ---------------------------------------------------------------------------

def human_turn(env, info, act, agent_moves, step):
    """Liest einen gültigen Zug ein. Rückgabe: Feldindex, oder None bei 'q'."""
    valid = set(np.flatnonzero(info["action_mask"]).tolist())
    while True:
        try:
            raw = input(f"Feld für {env.current_tile}? (0-18, h=Hinweis, d=Stapel, q=Ende) > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return None

        if raw in ("q", "quit", "exit"):
            return None
        if raw == "d":
            remaining = sorted(env.deck)
            print(f"  Noch im Stapel ({len(remaining)}): "
                  + ", ".join(f"({v},{l},{r})" for v, l, r in remaining))
            continue
        if raw == "h":
            suggestion = act(env, env._get_obs(), info)
            same = " (dasselbe wie in seiner eigenen Partie)" if agent_moves[step][1] == suggestion else ""
            print(f"  Das Netz würde auf deinem Brett Feld {suggestion} nehmen{same}.")
            continue
        if not raw.isdigit():
            print("  Bitte eine Feldnummer 0-18 eingeben (oder h / d / q).")
            continue

        cell = int(raw)
        if cell not in valid:
            if 0 <= cell < NUM_CELLS:
                print(f"  Feld {cell} ist schon belegt. Frei sind: {sorted(valid)}")
            else:
                print("  Feldnummer muss zwischen 0 und 18 liegen.")
            continue
        return cell


def main():
    parser = argparse.ArgumentParser(
        description="Gegen ein trainiertes Netz spielen - gleiche Kachelfolge, eigenes Brett."
    )
    parser.add_argument("--model", required=True,
                        help="Pfad zum Modell: .pt (Afterstate) oder .zip (PPO/DQN).")
    parser.add_argument("--algo", choices=["dqn", "ppo"], default=None,
                        help="Nur für .zip-Modelle nötig.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Feste Kachelfolge (wiederholbar). Ohne Angabe zufällig.")
    parser.add_argument("--no-live", action="store_true",
                        help="Die Live-Ansicht im Browser abschalten (Default: an, "
                             "schreibt nach jedem Zug replay/play_<seed>.html).")
    parser.add_argument("--html", action="store_true",
                        help="Am Ende zusätzlich beide Bretter als Einzelseiten in "
                             "replay/ schreiben, im Format von replay.py.")
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else random.randrange(2**31 - 1)
    act, model_name = load_agent(args.model, args.algo)

    print(f"\nGegner: {model_name}  ({Path(args.model).name})")
    print(f"Seed:   {seed}   (mit --seed {seed} exakt diese Partie nochmal spielen)")
    print("\nBeide bekommen dieselben 19 Kacheln in derselben Reihenfolge,")
    print("jeder legt auf sein eigenes Brett. Das Netz spielt vor ...", end=" ", flush=True)
    agent_board, agent_score, agent_moves = play_agent_episode(act, seed)
    print("fertig.\n")

    live_path = None
    if not args.no_live:
        REPLAY_DIR.mkdir(exist_ok=True)
        live_path = REPLAY_DIR / f"play_{seed}.html"

    env = TakeItEasyEnv()
    obs, info = env.reset(seed=seed)
    terminated, human_score = False, 0.0
    step, last_cell = 0, None

    while not terminated:
        running, _complete = score_breakdown(env.board)
        if live_path is not None:
            live_path.write_text(play_page(
                list(env.board), seed, step, env.current_tile, running, last_cell,
                agent_name=model_name,
            ))
            if step == 0:
                print(f"Live-Ansicht: {live_path.relative_to(REPO_ROOT)}")
                open_in_browser(live_path)
        print("=" * 75)
        print(f"Zug {step + 1}/19        fertige Linien bisher: {running} Punkte")
        print(render_board(env.board))
        print()
        cell = human_turn(env, info, act, agent_moves, step)
        if cell is None:
            print("\nAbgebrochen. Das Netz hatte in dieser Partie "
                  f"{agent_score:.0f} Punkte gemacht.")
            return
        obs, reward, terminated, _truncated, info = env.step(cell)
        step, last_cell = step + 1, cell
        if terminated:
            human_score = reward

    human_board = list(env.board)
    print("=" * 75)
    print(f"\nDEIN BRETT  ({human_score:.0f} Punkte)")
    print(render_board(human_board, show_indices=False))
    print(f"\nDAS NETZ    ({agent_score:.0f} Punkte)")
    print(render_board(agent_board, show_indices=False))

    diff = human_score - agent_score
    print("\n" + "=" * 75)
    if diff > 0:
        print(f"Du gewinnst mit {diff:.0f} Punkten Vorsprung.")
    elif diff < 0:
        print(f"Das Netz gewinnt mit {-diff:.0f} Punkten Vorsprung.")
    else:
        print("Unentschieden.")
    print(f"Diese Partie nochmal (auch gegen ein anderes Modell): --seed {seed}")

    if live_path is not None:
        # Ohne meta-refresh, dafür jetzt mit dem Brett des Netzes daneben.
        live_path.write_text(play_page(
            human_board, seed, step, None, human_score, last_cell,
            agent_board=agent_board, agent_score=agent_score, agent_name=model_name,
        ))
        print(f"Endstand als Bild: {live_path.relative_to(REPO_ROOT)}")

    if args.html:
        from board_render import board_to_html
        REPLAY_DIR.mkdir(exist_ok=True)
        for name, board, score in (
            ("du", human_board, human_score),
            ("netz", agent_board, agent_score),
        ):
            path = REPLAY_DIR / f"play_{seed}_{name}.html"
            path.write_text(board_to_html(
                board, score=score,
                title=f"Take It Easy - {'Du' if name == 'du' else model_name}, "
                      f"Seed {seed}, Score {score:.0f}",
            ))
            print(f"HTML: {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
