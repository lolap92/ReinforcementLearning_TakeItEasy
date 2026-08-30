"""
Afterstate-Wertfunktion für Take It Easy (Phase 8).

Warum ein neues Skript und nicht noch ein PPO-Sweep
---------------------------------------------------
Der Phase-7-Report (`reports/phase7_analysis_report.html`) hat zwei Dinge
gezeigt: (1) der beste MaskablePPO-Lauf (Ø 108,9 nach 25 Mio. Steps) liegt
unter zwei *untrainierten* Heuristiken aus `baselines.py`
(`greedy_potential` Ø 120,9, `expected_value` Ø 128,9), und (2) mehr Steps
sind nicht der Hebel - über den letzten gemessenen Abschnitt bringt das
Training nur noch ~11,5 Punkte pro Verzehnfachung der Steps.

Dieses Skript setzt stattdessen an der Struktur des Problems an. Take It Easy
ist ein Einpersonen-MDP mit **vollständig bekannter Dynamik**:

  - Nach dem Legen ist der Folgezustand deterministisch (kein Gegner, kein
    Rauschen) - dieser Zustand heißt "Afterstate".
  - Der einzige Zufall ist, welche Kachel als nächstes gezogen wird, und das
    ist eine exakt bekannte Gleichverteilung über das Restdeck.

Man muss also gar keine Policy lernen. Man lernt nur

    V(Afterstate) = erwarteter Endscore, wenn ab hier weitergespielt wird

und wählt zur Spielzeit schlicht `argmax` über die höchstens 19 möglichen
Afterstates. Das ist dieselbe Konstruktion, die TD-Gammon (Backgammon) und
die starken 2048-Agenten benutzen. Drei konkrete Vorteile gegenüber PPO auf
diesem Environment:

  1. **Ein Lernsignal je Zugmöglichkeit statt je gewählter Aktion.** PPO muss
     aus einem verrauschten Policy-Gradienten herausdestillieren, welche von
     19 Aktionen gut war. Hier ist es eine skalare Regression, und jeder der
     19 Afterstates ist ein gültiger Trainingspunkt.
  2. **Der `argmax` ist bereits eine 1-Ply-Suche.** Fehler der Wertfunktion
     werden dadurch teilweise wegkorrigiert; eine Policy hat diese Korrektur
     nicht.
  3. **Kein Reward Shaping, kein ent_coef, keine Advantage-Normalisierung.**
     Die komplette Wave-1/2-Maschinerie aus `train_ppo.py` entfällt.

Und der Unterschied zu DQN (Phase 5, kollabierte mehrfach): DQN bootstrapped
über ein `max` von *gelernten* Q-Werten, was sich selbst aufschaukeln kann.
Hier läuft das `max` über *bekannte* Folgezustände - der gefährliche Teil der
Q-Learning-Rückkopplung fehlt.

Was das Netz sieht (Hebel 2 aus dem Report)
-------------------------------------------
`env.py` liefert 60 rohe Zahlen und **nicht das Restdeck**. Der Agent kann
damit gar nicht einschätzen, ob eine angefangene Linie überhaupt noch
komplettierbar ist. Dieses Skript kodiert den Afterstate deshalb selbst
(siehe `encode_afterstates`):

  - 19 Felder x 10 One-Hot-Dimensionen (je 3 pro Richtung + "leer") = 190.
    Kachelwerte sind kategorial, nicht ordinal - die 9 bei "senkrecht" ist
    nicht "größer als" die 5, sondern eine andere Sorte.
  - 27 Dimensionen Multi-Hot: welche Kacheln liegen noch im Stapel.
  - 9 Dimensionen: wie viele Kacheln je (Richtung, Wert) noch im Stapel sind.
  - 15 Linien x 8 Features (optional, `--no-line-features` schaltet sie ab):
    Füllstand, ob die Linie noch lebt, ihr aktueller Wert, ihr
    potential_score()-Beitrag, wie viele passende Kacheln noch im Deck sind,
    und die Wahrscheinlichkeit, dass sie noch komplett wird. Das sind genau
    die Größen, aus denen `expected_value` in `baselines.py` seine 128,9
    zieht - dem Netz als Eingabe zu geben ist billiger, als sie aus rohen
    Zahlen rekonstruieren zu lassen.
  - 2 Dimensionen: Fortschritt (gelegte Kacheln, Restdeckgröße).

Wie trainiert wird
------------------
Selbstspiel mit epsilon-greedy über die aktuelle Wertfunktion, dann
**lambda-Returns** als Regressionsziel. Weil der Reward nur am Episodenende
kommt (alle Zwischenrewards sind 0) und gamma=1 ist, wird der lambda-Return
zu einer sehr kurzen Rekursion:

    target[18] = tatsächlicher Endscore          (letzter Afterstate = fertig)
    target[t]  = (1-lambda) * max_a V(s'[t+1])  +  lambda * target[t+1]

`--lambda 1.0` ist damit reines Monte Carlo (unverzerrt, hohe Varianz),
`--lambda 0.0` ist TD(0) (verzerrt solange V schlecht ist, dafür viel
ruhiger). Default 0.7. Das `max_a` wird immer über *alle* Zugmöglichkeiten
gebildet, auch wenn epsilon-greedy einen anderen Zug gespielt hat - die
Zielwerte sind dadurch off-policy und werden von der Exploration nicht
verzerrt.

Die Ziele werden beim Sammeln berechnet und danach nur auf **frischen** Daten
trainiert (mehrere Epochen, kein Replay-Buffer). Ein Replay-Buffer wäre
dateneffizienter, aber die gespeicherten lambda-Returns würden veralten,
sobald sich V weiterbewegt - für ein Lernprojekt ist der einfache, ehrliche
Weg hier mehr wert als die letzten Prozente Effizienz.

Wertkopf: statt eines skalaren Ausgangs mit MSE ist der Default ein
**Two-Hot-Klassifikationskopf** (`--value-head twohot`): der Endscore-Bereich
0..310 wird in `--atoms` Stützstellen zerlegt, das Ziel auf die zwei
benachbarten Stützstellen verteilt und mit Cross-Entropy trainiert; als Wert
dient der Erwartungswert der Verteilung. Der Endscore ist eine Summe weniger,
großer Sprünge (eine Linie bringt 0 oder z.B. 45 Punkte) - bei so einer
"lumpigen" Zielgröße lernt ein Klassifikationskopf verlässlicher als eine
MSE-Regression, die auf den Mittelwert glättet. `--value-head scalar` stellt
auf Huber-Regression um.

Nutzung
-------
    pip install -r requirements.txt
    python train_afterstate.py                        # Defaults, ~300k Episoden
    python train_afterstate.py --episodes 1000000 --device cuda
    python train_afterstate.py --lambda 1.0 --value-head scalar   # Ablation

Erstes Ziel ist nicht der Weltrekord, sondern **sauber über 128,9** zu kommen
- also die untrainierte `expected_value`-Heuristik zu schlagen. Alles darunter
heißt: das Netz hat nicht mehr gelernt, als schon in `baselines.py` steht.

Ergebnisse landen wie bei den anderen Skripten unter `experiments/<run_id>/`
(`config.json`/`summary.json`/`episodes.csv` ins Repo, `models/` und
`tensorboard/` gitignored) plus eine Zeile in `EXPERIMENTS.md`.
"""

import argparse
import csv
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from env import TakeItEasyEnv
from game import build_deck, ALL_LINE_GROUPS, NUM_CELLS

REPO_ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
EXPERIMENTS_LOG = REPO_ROOT / "EXPERIMENTS.md"

# Referenzwerte aus experiments/2026-08-30_0705_phase7_heuristic_baselines
# (2000 Episoden je Agent) - werden am Ende zum Vergleich mit ausgegeben,
# damit ein Lauf sofort einzuordnen ist statt nur eine nackte Zahl zu liefern.
REFERENCE_SCORES = {
    "random": 10.79,
    "greedy (score_board)": 27.60,
    "greedy_potential": 120.88,
    "expected_value": 128.93,
    "ppo_25m_singleenv": 108.86,
}

# ---------------------------------------------------------------------------
# Statische Spieltabellen (einmal aufgebaut, danach nur noch Lookup)
# ---------------------------------------------------------------------------

DECK = build_deck()                                  # 27 Kacheln, kanonische Reihenfolge
TILE_INDEX = {tile: i for i, tile in enumerate(DECK)}
NUM_TILES = len(DECK)                                # 27
EMPTY = NUM_TILES                                    # Sentinel-Index für "Feld leer"

# Mögliche Werte je Richtung (0 = senkrecht, 1 = "/", 2 = "\")
VALUE_SETS = np.array([[1, 5, 9], [3, 4, 8], [2, 6, 7]], dtype=np.int64)

# VIDX_BY_DIR[d, tile] = Index 0..2 des Werts, den `tile` in Richtung d hat.
# Spalte EMPTY ist ein Dummy (0), wird nie ohne Maske verwendet.
VIDX_BY_DIR = np.zeros((3, NUM_TILES + 1), dtype=np.int64)
for _t, _tile in enumerate(DECK):
    for _d in range(3):
        VIDX_BY_DIR[_d, _t] = int(np.where(VALUE_SETS[_d] == _tile[_d])[0][0])

# TILE_ONEHOT[tile] = 10 Dimensionen: 3 senkrecht, 3 "/", 3 "\", 1 "leer".
# Ein einziger Gather statt drei Scatter-Operationen beim Kodieren.
TILE_ONEHOT = np.zeros((NUM_TILES + 1, 10), dtype=np.float32)
for _t in range(NUM_TILES):
    TILE_ONEHOT[_t, VIDX_BY_DIR[0, _t]] = 1.0
    TILE_ONEHOT[_t, 3 + VIDX_BY_DIR[1, _t]] = 1.0
    TILE_ONEHOT[_t, 6 + VIDX_BY_DIR[2, _t]] = 1.0
TILE_ONEHOT[EMPTY, 9] = 1.0

# DIR_VAL_MASK[tile, 3*d + k] = 1, falls `tile` in Richtung d den Wert mit
# Index k trägt. Ein Matmul (N,27) @ (27,9) liefert damit für jeden Zustand,
# wie viele passende Kacheln je (Richtung, Wert) noch im Stapel liegen.
DIR_VAL_MASK = np.zeros((NUM_TILES, 9), dtype=np.float32)
for _t in range(NUM_TILES):
    for _d in range(3):
        DIR_VAL_MASK[_t, 3 * _d + VIDX_BY_DIR[_d, _t]] = 1.0

# Alle 15 Linien flach: (Richtungsindex, Feld-Indizes)
FLAT_LINES = [
    (value_pos, np.array(line, dtype=np.int64))
    for _direction, (lines, value_pos) in ALL_LINE_GROUPS.items()
    for line in lines
]
NUM_LINES = len(FLAT_LINES)          # 15
LINE_FEATURES = 8                    # Features je Linie, siehe encode_afterstates
MAX_LINE_SCORE = 45.0                # längste Linie (5) x höchster Wert (9)
MAX_SCORE = 307.0                    # theoretisches Maximum, siehe README

BOARD_DIMS = NUM_CELLS * 10          # 190
DECK_DIMS = NUM_TILES + 9            # 27 Multi-Hot + 9 Zähler je (Richtung, Wert)
GLOBAL_DIMS = 2                      # Fortschritt


def feature_dim(line_features=True):
    dims = BOARD_DIMS + DECK_DIMS + GLOBAL_DIMS
    if line_features:
        dims += NUM_LINES * LINE_FEATURES
    return dims


# ---------------------------------------------------------------------------
# Kodierung und Scoring - beides vektorisiert über N Boards gleichzeitig
# ---------------------------------------------------------------------------

def encode_afterstates(boards, remaining, line_features=True):
    """
    boards:    (N, 19) int, Kachelindex 0..26 oder -1 für ein leeres Feld
    remaining: (N, 27) bool, True = Kachel liegt noch im Ziehstapel

    Rückgabe: (N, feature_dim()) float32.

    Beides beschreibt einen *Afterstate*: die aktuelle Kachel wurde bereits
    gelegt (sie steht im Board) und ist bereits aus `remaining` entfernt.
    """
    n = boards.shape[0]
    filled = boards >= 0
    safe = np.where(filled, boards, EMPTY)

    parts = [TILE_ONEHOT[safe].reshape(n, BOARD_DIMS)]

    remaining_f = remaining.astype(np.float32)
    counts = remaining_f @ DIR_VAL_MASK                     # (N, 9)
    n_remaining = remaining_f.sum(axis=1)                   # (N,)
    parts.append(remaining_f)
    parts.append(counts / 9.0)

    n_placed = filled.sum(axis=1).astype(np.float32)
    parts.append(np.stack([n_placed / NUM_CELLS, n_remaining / NUM_TILES], axis=1))

    if line_features:
        feats = np.empty((n, NUM_LINES, LINE_FEATURES), dtype=np.float32)
        for li, (d, cells) in enumerate(FLAT_LINES):
            length = len(cells)
            sub_filled = filled[:, cells]                            # (N, L)
            vidx = VIDX_BY_DIR[d][safe[:, cells]]                    # (N, L)

            n_filled = sub_filled.sum(axis=1)
            has_tile = n_filled > 0
            # Eine Linie lebt, solange alle bereits gelegten Kacheln denselben
            # Wert in dieser Richtung tragen - min == max über die belegten
            # Felder (leere Felder werden auf neutrale Sentinels gesetzt).
            v_min = np.where(sub_filled, vidx, 3).min(axis=1)
            v_max = np.where(sub_filled, vidx, -1).max(axis=1)
            alive = ~has_tile | (v_min == v_max)

            line_vidx = np.where(has_tile, v_min, 0)
            value = VALUE_SETS[d][line_vidx].astype(np.float32)
            value = np.where(has_tile & alive, value, 0.0)

            n_missing = length - n_filled
            matching = counts[np.arange(n), 3 * d + line_vidx]

            # P(alle fehlenden Felder bekommen genau diesen Wert), Ziehen ohne
            # Zurücklegen. Vereinfachung: die Konkurrenz anderer Linien um
            # dieselben Kacheln wird ignoriert - dieselbe Näherung wie in
            # baselines.expected_value.
            prob = np.ones(n, dtype=np.float32)
            for j in range(length):
                active = n_missing > j
                num = np.maximum(matching - j, 0.0)
                den = np.maximum(n_remaining - j, 1.0)
                prob = np.where(active, prob * (num / den), prob)
            prob = np.where(has_tile & alive, prob, 0.0)

            feats[:, li, 0] = n_filled / length
            feats[:, li, 1] = n_missing / length
            feats[:, li, 2] = alive
            feats[:, li, 3] = value / 9.0
            feats[:, li, 4] = value * n_filled / MAX_LINE_SCORE
            feats[:, li, 5] = matching / 9.0
            feats[:, li, 6] = prob
            feats[:, li, 7] = prob * value * length / MAX_LINE_SCORE

        parts.append(feats.reshape(n, NUM_LINES * LINE_FEATURES))

    return np.concatenate(parts, axis=1, dtype=np.float32)


def score_full_boards(boards):
    """Endscore für (N, 19)-Boards, in denen alle Felder belegt sind.
    Vektorisierte Variante von game.score_board() - identisches Ergebnis,
    aber ohne Python-Schleife über die Episoden."""
    total = np.zeros(boards.shape[0], dtype=np.float32)
    for d, cells in FLAT_LINES:
        vidx = VIDX_BY_DIR[d][boards[:, cells]]
        same = (vidx == vidx[:, :1]).all(axis=1)
        total += same * VALUE_SETS[d][vidx[:, 0]] * len(cells)
    return total


# ---------------------------------------------------------------------------
# Wertfunktion
# ---------------------------------------------------------------------------

class ValueNet(nn.Module):
    """MLP über den kodierten Afterstate.

    head="twohot": Ausgabe sind Logits über `atoms` Stützstellen zwischen 0
    und `v_max`; der Wert ist der Erwartungswert der Softmax-Verteilung, das
    Training läuft als Cross-Entropy gegen die Two-Hot-Kodierung des Ziels.
    head="scalar": ein einzelner Ausgang, Huber-Loss.
    """

    def __init__(self, in_dim, hidden=(512, 512, 512), head="twohot", atoms=51, v_max=MAX_SCORE):
        super().__init__()
        self.head = head
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU()]
            prev = h
        self.body = nn.Sequential(*layers)
        out_dim = atoms if head == "twohot" else 1
        self.out = nn.Linear(prev, out_dim)
        nn.init.zeros_(self.out.bias)
        nn.init.uniform_(self.out.weight, -1e-3, 1e-3)
        self.register_buffer("support", torch.linspace(0.0, v_max, atoms))

    def forward(self, x):
        return self.out(self.body(x))

    def value(self, x):
        """Erwarteter Endscore als (N,)-Tensor."""
        out = self(x)
        if self.head == "twohot":
            return (F.softmax(out, dim=-1) * self.support).sum(dim=-1)
        return out.squeeze(-1)

    def loss(self, x, target):
        out = self(x)
        if self.head == "twohot":
            return F.cross_entropy(out, two_hot(target, self.support))
        return F.smooth_l1_loss(out.squeeze(-1), target, beta=5.0)


def two_hot(target, support):
    """Verteilt einen skalaren Zielwert auf die zwei benachbarten
    Stützstellen (linear gewichtet) - so bleibt der Erwartungswert der
    Zielverteilung exakt der Zielwert."""
    atoms = support.numel()
    delta = (support[-1] - support[0]) / (atoms - 1)
    clamped = target.clamp(support[0], support[-1])
    pos = (clamped - support[0]) / delta
    lower = pos.floor().clamp(0, atoms - 1).long()
    upper = (lower + 1).clamp(max=atoms - 1)
    w_upper = (pos - lower.float()).clamp(0.0, 1.0)

    dist = torch.zeros(target.shape[0], atoms, device=target.device, dtype=torch.float32)
    dist.scatter_add_(1, lower.unsqueeze(1), (1.0 - w_upper).unsqueeze(1))
    dist.scatter_add_(1, upper.unsqueeze(1), w_upper.unsqueeze(1))
    return dist


@torch.no_grad()
def batched_values(net, feats, device, chunk=32768):
    """Wertet viele Afterstates am Stück aus (Selbstspiel-Heißpfad)."""
    net.eval()
    outputs = []
    for start in range(0, feats.shape[0], chunk):
        batch = torch.from_numpy(feats[start:start + chunk]).to(device)
        outputs.append(net.value(batch).float().cpu().numpy())
    return np.concatenate(outputs) if len(outputs) > 1 else outputs[0]


# ---------------------------------------------------------------------------
# Selbstspiel (alle Partien einer Iteration laufen im Gleichschritt)
# ---------------------------------------------------------------------------

def play_batch(net, n_games, rng, device, epsilon, lam, line_features=True, collect=True):
    """Spielt `n_games` vollständige Partien parallel.

    Alle Partien sind gleich lang (genau 19 Züge) und liegen im selben Zug -
    dadurch lassen sich in jedem Schritt sämtliche Zugmöglichkeiten aller
    Partien in *einem* Forward-Pass bewerten (bis zu n_games x 19 Zeilen)
    statt einzeln.

    Rückgabe: (feats, targets, scores).
    feats/targets sind None, wenn collect=False (reine Auswertung).
    """
    b = n_games
    # Eine zufällige Permutation des Decks je Partie: Kachel p wird im Zug p
    # gezogen. Entspricht dem shuffle+pop in env.py.
    perms = np.argsort(rng.random((b, NUM_TILES)), axis=1)
    boards = np.full((b, NUM_CELLS), -1, dtype=np.int64)
    remaining = np.ones((b, NUM_TILES), dtype=bool)
    rows = np.arange(b)

    chosen_feats = [] if collect else None
    best_values = np.zeros((NUM_CELLS, b), dtype=np.float32)

    for step in range(NUM_CELLS):
        tile = perms[:, step]
        remaining[rows, tile] = False
        n_free = NUM_CELLS - step

        free = np.argwhere(boards < 0)              # zeilenweise sortiert
        game_ids, cells = free[:, 0], free[:, 1]
        candidates = boards[game_ids]               # fancy indexing kopiert
        candidates[np.arange(candidates.shape[0]), cells] = tile[game_ids]

        feats = encode_afterstates(candidates, remaining[game_ids], line_features)
        values = batched_values(net, feats, device).reshape(b, n_free)

        best_values[step] = values.max(axis=1)
        choice = values.argmax(axis=1)
        if epsilon > 0.0:
            explore = rng.random(b) < epsilon
            choice = np.where(explore, rng.integers(0, n_free, size=b), choice)

        if collect:
            chosen_feats.append(feats[rows * n_free + choice])

        boards[rows, cells.reshape(b, n_free)[rows, choice]] = tile

    scores = score_full_boards(boards)
    if not collect:
        return None, None, scores

    # lambda-Return, rückwärts. Alle Zwischenrewards sind 0 und gamma=1,
    # deshalb reduziert sich der Forward-View auf diese eine Zeile.
    targets = np.empty((NUM_CELLS, b), dtype=np.float32)
    targets[NUM_CELLS - 1] = scores
    for step in range(NUM_CELLS - 2, -1, -1):
        targets[step] = (1.0 - lam) * best_values[step + 1] + lam * targets[step + 1]

    feats = np.concatenate(chosen_feats, axis=0)
    return feats, targets.reshape(-1), scores


# ---------------------------------------------------------------------------
# Auswertung auf der echten TakeItEasyEnv (vergleichbar mit train_ppo.py)
# ---------------------------------------------------------------------------

class AfterstateAgent:
    """Spielt greedy: bewertet alle freien Felder und nimmt das beste.

    Läuft bewusst über die echte `TakeItEasyEnv` mit denselben Seeds wie
    `train_ppo.evaluate()` und `baselines.py`, damit die Zahlen direkt
    vergleichbar sind - und nicht über die schnelle vektorisierte Simulation
    oben, die zwar dasselbe Spiel implementiert, aber ein zweiter Codepfad
    mit eigenem Fehlerpotenzial wäre.
    """

    def __init__(self, net, device, line_features=True):
        self.net = net
        self.device = device
        self.line_features = line_features

    def act(self, env):
        board = np.array(
            [TILE_INDEX[t] if t is not None else -1 for t in env.board], dtype=np.int64
        )
        remaining = np.zeros(NUM_TILES, dtype=bool)
        for tile in env.deck:
            remaining[TILE_INDEX[tile]] = True

        free = np.flatnonzero(board < 0)
        candidates = np.repeat(board[None, :], len(free), axis=0)
        candidates[np.arange(len(free)), free] = TILE_INDEX[env.current_tile]

        feats = encode_afterstates(
            candidates, np.repeat(remaining[None, :], len(free), axis=0), self.line_features
        )
        values = batched_values(self.net, feats, self.device)
        return int(free[int(values.argmax())])


def evaluate(agent, n_episodes, seed):
    env = TakeItEasyEnv()
    scores = np.empty(n_episodes, dtype=np.float32)
    for i in range(n_episodes):
        _obs, _info = env.reset(seed=seed + i)
        terminated = False
        total = 0.0
        while not terminated:
            _obs, reward, terminated, _truncated, _info = env.step(agent.act(env))
            if terminated:
                total = reward
        scores[i] = total
    return scores


# ---------------------------------------------------------------------------
# Infrastruktur (bewusst gleich wie in train_ppo.py / train_dqn.py)
# ---------------------------------------------------------------------------

def git_commit_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return None


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def open_in_chrome(url):
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
    chrome_exe = next((p for p in candidates if p and Path(p).exists()), None)
    if chrome_exe:
        try:
            subprocess.Popen([chrome_exe, url])
            return
        except Exception:
            pass
    try:
        webbrowser.open(url)
    except Exception:
        pass


def start_tensorboard(logdir, port=6006):
    url = f"http://localhost:{port}"
    if is_port_in_use(port):
        print(f"TensorBoard läuft bereits auf {url} - starte kein zweites, öffne Chrome trotzdem.")
        open_in_chrome(url)
        return None
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "tensorboard.main", "--logdir", str(logdir), "--port", str(port)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        print(f"TensorBoard konnte nicht gestartet werden ({exc}) - Training läuft trotzdem weiter.")
        return None
    time.sleep(3)
    print(f"TensorBoard läuft auf {url}")
    open_in_chrome(url)
    return proc


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_default_tag(args):
    parts = [f"afterstate_{episodes_label(args.episodes)}"]
    if args.value_head != "twohot":
        parts.append(args.value_head)
    if args.lam != 0.7:
        parts.append(f"lam{args.lam:g}".replace(".", ""))
    if args.no_line_features:
        parts.append("nolinefeat")
    return "_".join(parts)


def episodes_label(n):
    if n % 1_000_000 == 0:
        return f"{n // 1_000_000}m"
    if n % 1_000 == 0:
        return f"{n // 1000}k"
    return str(n)


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Afterstate-Wertfunktion für Take It Easy (Phase 8, lokal ausführen)."
    )
    parser.add_argument("--episodes", type=int, default=300_000,
                        help="Anzahl Selbstspiel-Episoden insgesamt (Default: 300000).")
    parser.add_argument("--games-per-iter", type=int, default=256,
                        help="Partien pro Iteration, alle parallel im Gleichschritt "
                             "(Default: 256). Größer = bessere Auslastung, aber "
                             "weniger Gradientenschritte bei gleichem Episodenbudget.")
    parser.add_argument("--epochs", type=int, default=4,
                        help="Durchläufe über die frisch gesammelten Daten je Iteration.")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Startlernrate, fällt cosinusförmig auf 0 (wie der lineare "
                             "Abfall bei PPO, der sich in Phase 6 klar bewährt hat).")
    parser.add_argument("--lam", "--lambda", dest="lam", type=float, default=0.7,
                        help="lambda für den lambda-Return: 1.0 = Monte Carlo, "
                             "0.0 = TD(0). Default 0.7.")
    parser.add_argument("--epsilon-start", type=float, default=0.20)
    parser.add_argument("--epsilon-end", type=float, default=0.01)
    parser.add_argument("--hidden", type=str, default="512,512,512",
                        help="Schichtgrößen des MLP, kommagetrennt.")
    parser.add_argument("--value-head", choices=["twohot", "scalar"], default="twohot",
                        help="twohot = Klassifikation über Score-Stützstellen (Default), "
                             "scalar = klassische Huber-Regression.")
    parser.add_argument("--atoms", type=int, default=51,
                        help="Stützstellen des Two-Hot-Kopfs zwischen 0 und 307.")
    parser.add_argument("--no-line-features", action="store_true",
                        help="Die 15x8 Linien-Features weglassen (Ablation: wie viel "
                             "trägt die Feature-Konstruktion, wie viel das Netz bei?).")
    parser.add_argument("--eval-episodes", type=int, default=2000,
                        help="Episoden für die finale Auswertung auf der echten Env.")
    parser.add_argument("--eval-every", type=int, default=25,
                        help="Alle N Iterationen eine schnelle Zwischenauswertung "
                             "(greedy, vektorisiert) für TensorBoard und die "
                             "Best-Checkpoint-Auswahl.")
    parser.add_argument("--eval-games", type=int, default=1024,
                        help="Partien je Zwischenauswertung.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu",
                        help="cpu (Default), cuda oder auto. Anders als bei PPO lohnt "
                             "sich hier eine GPU eher: pro Zug werden bis zu "
                             "games-per-iter x 19 Afterstates in einem Batch bewertet.")
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--tensorboard-port", type=int, default=6006)
    args = parser.parse_args()

    if args.tag is None:
        args.tag = build_default_tag(args)

    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    run_id = f"{timestamp[:16].replace(':', '').replace('T', '_')}_{args.tag}"
    run_dir = EXPERIMENTS_DIR / run_id
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    (run_dir / "tensorboard").mkdir(exist_ok=True)

    if not args.no_tensorboard:
        start_tensorboard(EXPERIMENTS_DIR, port=args.tensorboard_port)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = resolve_device(args.device)
    line_features = not args.no_line_features

    hidden = tuple(int(h) for h in args.hidden.split(",") if h.strip())
    in_dim = feature_dim(line_features)
    net = ValueNet(in_dim, hidden=hidden, head=args.value_head, atoms=args.atoms).to(device)
    n_params = sum(p.numel() for p in net.parameters())

    iterations = max(1, args.episodes // args.games_per_iter)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(run_dir / "tensorboard" / "afterstate"))

    print(f"Device: {device} | Feature-Dimension: {in_dim} | Parameter: {n_params:,}")
    print(f"{iterations} Iterationen x {args.games_per_iter} Partien "
          f"= {iterations * args.games_per_iter:,} Episoden")
    print(f"Wertkopf: {args.value_head} | lambda: {args.lam} | "
          f"Linien-Features: {'an' if line_features else 'aus'}\n")

    best_eval = -np.inf
    started = time.time()

    for iteration in range(iterations):
        progress = iteration / max(1, iterations - 1)
        epsilon = args.epsilon_start + (args.epsilon_end - args.epsilon_start) * progress

        feats, targets, scores = play_batch(
            net, args.games_per_iter, rng, device, epsilon, args.lam, line_features
        )

        net.train()
        feats_t = torch.from_numpy(feats).to(device)
        targets_t = torch.from_numpy(targets).to(device)
        n_samples = feats_t.shape[0]
        losses = []
        for _epoch in range(args.epochs):
            order = torch.randperm(n_samples, device=device)
            for start in range(0, n_samples, args.batch_size):
                idx = order[start:start + args.batch_size]
                loss = net.loss(feats_t[idx], targets_t[idx])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.detach()))
        scheduler.step()

        step = (iteration + 1) * args.games_per_iter
        writer.add_scalar("rollout/score_mean", float(scores.mean()), step)
        writer.add_scalar("rollout/score_max", float(scores.max()), step)
        writer.add_scalar("train/loss", float(np.mean(losses)), step)
        writer.add_scalar("train/epsilon", epsilon, step)
        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)

        if (iteration + 1) % args.eval_every == 0 or iteration == iterations - 1:
            _f, _t, eval_scores = play_batch(
                net, args.eval_games, np.random.default_rng(12345), device,
                epsilon=0.0, lam=args.lam, line_features=line_features, collect=False,
            )
            mean = float(eval_scores.mean())
            writer.add_scalar("eval/score_mean", mean, step)
            if mean > best_eval:
                best_eval = mean
                torch.save(
                    {"state_dict": net.state_dict(), "in_dim": in_dim, "hidden": hidden,
                     "head": args.value_head, "atoms": args.atoms,
                     "line_features": line_features},
                    run_dir / "models" / "best_model.pt",
                )
            elapsed = time.time() - started
            print(f"Iter {iteration + 1:5d}/{iterations}  Episoden {step:>8,}  "
                  f"Selbstspiel Ø {scores.mean():6.2f}  greedy Ø {mean:6.2f}  "
                  f"(bestes {best_eval:6.2f})  eps {epsilon:.3f}  "
                  f"loss {np.mean(losses):.4f}  {elapsed / 60:.1f} min")

    torch.save(
        {"state_dict": net.state_dict(), "in_dim": in_dim, "hidden": hidden,
         "head": args.value_head, "atoms": args.atoms, "line_features": line_features},
        run_dir / "models" / "final_model.pt",
    )
    writer.close()

    print(f"\nAuswertung über {args.eval_episodes} echte Take-It-Easy-Episoden "
          f"(TakeItEasyEnv, Seeds {args.seed + 1}..{args.seed + args.eval_episodes}) ...")
    agent = AfterstateAgent(net, device, line_features)
    scores = evaluate(agent, args.eval_episodes, seed=args.seed + 1)
    print(f"Afterstate (Endmodell):  mean={scores.mean():.2f}  std={scores.std():.2f}  "
          f"min={scores.min():.0f}  max={scores.max():.0f}")

    best_scores = None
    best_path = run_dir / "models" / "best_model.pt"
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device, weights_only=True)
        best_net = ValueNet(in_dim, hidden=hidden, head=args.value_head, atoms=args.atoms).to(device)
        best_net.load_state_dict(checkpoint["state_dict"])
        best_scores = evaluate(AfterstateAgent(best_net, device, line_features),
                               args.eval_episodes, seed=args.seed + 1)
        print(f"Afterstate (bestes Checkpoint):  mean={best_scores.mean():.2f}  "
              f"std={best_scores.std():.2f}  min={best_scores.min():.0f}  "
              f"max={best_scores.max():.0f}")

    reached = max(scores.mean(), best_scores.mean() if best_scores is not None else -np.inf)
    print("\nEinordnung (Referenzwerte aus experiments/*_phase7_heuristic_baselines):")
    for name, value in sorted(REFERENCE_SCORES.items(), key=lambda kv: -kv[1]):
        marker = "<-- hier" if value < reached else ""
        print(f"  {name:24s} {value:7.2f}  {marker}")
    if reached > REFERENCE_SCORES["expected_value"]:
        print(f"\n-> Ziel erreicht: {reached:.2f} liegt über der besten untrainierten "
              f"Heuristik ({REFERENCE_SCORES['expected_value']:.2f}).")
    else:
        print(f"\n-> Ziel noch nicht erreicht: {reached:.2f} liegt unter "
              f"expected_value ({REFERENCE_SCORES['expected_value']:.2f}). Mehr Episoden, "
              f"größeres Netz oder anderes lambda probieren.")

    config = {
        "run_id": run_id,
        "tag": args.tag,
        "timestamp": timestamp,
        "git_commit": git_commit_hash(),
        "env": "TakeItEasyEnv",
        "algorithm": "Afterstate-Wertfunktion (lambda-Return, Selbstspiel)",
        "hyperparameters": {
            "episodes": iterations * args.games_per_iter,
            "iterations": iterations,
            "games_per_iter": args.games_per_iter,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": f"cosine({args.lr} -> 0)",
            "lambda": args.lam,
            "gamma": 1.0,
            "epsilon": f"linear({args.epsilon_start} -> {args.epsilon_end})",
            "hidden": list(hidden),
            "value_head": args.value_head,
            "atoms": args.atoms,
            "line_features": line_features,
            "feature_dim": in_dim,
            "parameters": n_params,
        },
        "eval_episodes": args.eval_episodes,
        "master_seed": args.seed,
        "device": str(device),
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    summary = [{
        "name": "afterstate_full_board",
        "n_episodes": args.eval_episodes,
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "min": float(scores.min()),
        "max": float(scores.max()),
    }]
    if best_scores is not None:
        summary.append({
            "name": "afterstate_full_board_best_checkpoint",
            "n_episodes": args.eval_episodes,
            "mean": float(best_scores.mean()),
            "std": float(best_scores.std()),
            "min": float(best_scores.min()),
            "max": float(best_scores.max()),
        })
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(run_dir / "episodes.csv", "w", newline="") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["agent", "episode_index", "seed", "score"])
        for i, score in enumerate(scores):
            writer_csv.writerow(["afterstate_full_board", i, args.seed + 1 + i, score])
        if best_scores is not None:
            for i, score in enumerate(best_scores):
                writer_csv.writerow(
                    ["afterstate_full_board_best_checkpoint", i, args.seed + 1 + i, score]
                )

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
