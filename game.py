"""
Take It Easy - Kernspiellogik (unabhängig von RL/Gym).

Board-Layout (19 Felder, Index 0-18), Reihen von oben nach unten:
        0   1   2
      3   4   5   6
    7   8   9  10  11
     12  13  14  15
       16  17  18

Jede Kachel hat 3 Werte für 3 Richtungen:
  - "senkrecht" (|)   -> Werte aus {1, 5, 9}
  - "diagonal /"      -> Werte aus {3, 4, 8}
  - "diagonal \\"     -> Werte aus {2, 6, 7}

Es gibt alle 27 Kombinationen (3 x 3 x 3), jede genau einmal im Deck.
"""

import random
from itertools import product

# ---------------------------------------------------------------------------
# Kacheldeck
# ---------------------------------------------------------------------------

VERTICAL_VALUES = (1, 5, 9)
LEFT_DIAG_VALUES = (3, 4, 8)   # "/"
RIGHT_DIAG_VALUES = (2, 6, 7)  # "\"


def build_deck():
    """Erzeugt alle 27 Kacheln als Tupel (vertikal, links-diagonal, rechts-diagonal)."""
    return [
        (v, l, r)
        for v, l, r in product(VERTICAL_VALUES, LEFT_DIAG_VALUES, RIGHT_DIAG_VALUES)
    ]


# ---------------------------------------------------------------------------
# Board-Geometrie
# ---------------------------------------------------------------------------
# 19 Felder in 5 Reihen (3,4,5,4,3). Wir definieren für jede der 3 Richtungen
# die Liste der Linien, wobei jede Linie eine Liste von Feld-Indizes ist
# (in der Reihenfolge, wie sie entlang der Linie liegen - Reihenfolge ist
# fürs Scoring irrelevant, nur Gruppierung zählt).

# Reihen (für Übersicht / spätere Nutzung z.B. bei Rendering)
ROWS = [
    [0, 1, 2],
    [3, 4, 5, 6],
    [7, 8, 9, 10, 11],
    [12, 13, 14, 15],
    [16, 17, 18],
]

# Richtung 1: "senkrechte" Linien = genau die 5 Reihen oben (3,4,5,4,3 Felder)
LINES_VERTICAL = ROWS

# Richtung 2: Linien "/" (von unten-links nach oben-rechts)
LINES_LEFT_DIAG = [
    [7, 12, 16],
    [3, 8, 13, 17],
    [0, 4, 9, 14, 18],
    [1, 5, 10, 15],
    [2, 6, 11],
]

# Richtung 3: Linien "\" (von oben-links nach unten-rechts)
LINES_RIGHT_DIAG = [
    [0, 3, 7],
    [1, 4, 8, 12],
    [2, 5, 9, 13, 16],
    [6, 10, 14, 17],
    [11, 15, 18],
]

ALL_LINE_GROUPS = {
    "vertical": (LINES_VERTICAL, 0),     # 0 = Index im Kachel-Tupel für diese Richtung
    "left_diag": (LINES_LEFT_DIAG, 1),
    "right_diag": (LINES_RIGHT_DIAG, 2),
}

NUM_CELLS = 19


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_board(board):
    """
    Berechnet den Gesamtscore eines fertigen (oder teilweise gefüllten) Boards.

    board: Liste/Array der Länge 19, board[i] = Kachel-Tupel (v,l,r) oder None
           falls Feld leer.

    Rückgabe: (total_score, details) wobei details eine Liste von
              (richtung, linie_indices, wert_oder_None, teilscore) ist -
              nützlich zum Debuggen/Visualisieren.
    """
    total = 0
    details = []

    for direction, (lines, value_pos) in ALL_LINE_GROUPS.items():
        for line in lines:
            tiles = [board[i] for i in line]
            if any(t is None for t in tiles):
                # Linie noch nicht vollständig belegt -> zählt nicht
                details.append((direction, line, None, 0))
                continue
            values = [t[value_pos] for t in tiles]
            if len(set(values)) == 1:
                line_score = values[0] * len(line)
                total += line_score
                details.append((direction, line, values[0], line_score))
            else:
                details.append((direction, line, None, 0))

    return total, details


def potential_score(board):
    """
    Wie score_board(), aber für (auch teilweise gefüllte) Linien: statt nur
    komplette, konsistente Linien zu zählen, wird jede noch "lebendige" Linie
    (alle bisher gelegten Kacheln darin haben denselben Wert in dieser
    Richtung) mit Wert * Anzahl bereits gelegter Kacheln bewertet. Sobald eine
    Linie durch zwei unterschiedliche Werte "tot" ist, trägt sie 0 bei -
    genau wie bei score_board(). Für ein vollständiges Board ist das Ergebnis
    identisch zu score_board() (jede Linie ist dann entweder komplett und
    konsistent, oder komplett und inkonsistent -> 0 in beiden Funktionen).

    Grundlage für Potential-based Reward Shaping (siehe env.py,
    reward_shaping=True): Φ(board) = potential_score(board),
    F(s,a,s') = Φ(s') - Φ(s). Policy-invariant nach Ng et al. (1999), da rein
    additiv zum ursprünglichen Reward - ändert die optimale Policy nicht,
    macht aber jeden Zug statt nur den letzten informativ.
    """
    total = 0
    for _direction, (lines, value_pos) in ALL_LINE_GROUPS.items():
        for line in lines:
            tiles = [board[i] for i in line if board[i] is not None]
            if not tiles:
                continue
            values = [t[value_pos] for t in tiles]
            if len(set(values)) == 1:
                total += values[0] * len(tiles)
    return total


# ---------------------------------------------------------------------------
# Kleine Selbsttests (führe `python game.py` aus, um zu prüfen)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    deck = build_deck()
    assert len(deck) == 27, f"Deck sollte 27 Kacheln haben, hat {len(deck)}"
    assert len(set(deck)) == 27, "Alle Kacheln sollten einzigartig sein"

    # Geometrie-Check: jede Richtung muss alle 19 Felder genau einmal abdecken
    for direction, (lines, _) in ALL_LINE_GROUPS.items():
        covered = sorted(i for line in lines for i in line)
        assert covered == list(range(19)), (
            f"Richtung {direction} deckt nicht exakt alle 19 Felder ab: {covered}"
        )
        assert sum(len(l) for l in lines) == 19

    # Score-Test: leeres Board -> Score 0
    empty_board = [None] * NUM_CELLS
    total, _ = score_board(empty_board)
    assert total == 0

    # Score-Test: konstruiertes Board, bei dem wir den Score von Hand nachrechnen
    random.seed(42)
    deck_shuffled = deck[:]
    random.shuffle(deck_shuffled)
    test_board = deck_shuffled[:19]
    total, details = score_board(test_board)
    print(f"Zufalls-Board Score: {total}")
    manual_total = sum(d[3] for d in details)
    assert manual_total == total

    # Bekanntes Beispiel: alle Felder mit gleicher Kachel (nicht regelkonform,
    # aber gut um die Scoring-Formel zu prüfen) -> maximaler Score pro Richtung
    same_tile_board = [(9, 8, 7)] * NUM_CELLS
    total, details = score_board(same_tile_board)
    # Jede Richtung hat 5 Linien mit Längen 3,4,5,4,3 = 19 Felder gesamt
    # Score pro Richtung = 19 * Wert (jede Kachel zählt einmal je Richtung)
    expected = 19 * 9 + 19 * 8 + 19 * 7
    assert total == expected, f"Erwartet {expected}, bekommen {total}"

    # potential_score()-Tests: leeres Board -> 0, volles Board -> identisch zu
    # score_board() (egal ob Zufalls-Board oder alle-gleich-Board)
    assert potential_score(empty_board) == 0
    assert potential_score(test_board) == score_board(test_board)[0]
    assert potential_score(same_tile_board) == score_board(same_tile_board)[0]
    # Teilweise gefülltes Board (Felder 0+1 belegt, Rest leer): jedes Feld
    # trägt zu 3 Linien bei (eine je Richtung), auch wenn diese noch nicht
    # komplett sind - senkrecht (Linie [0,1,2], beide Felder Wert 5) trägt
    # 2*5, die restlichen 4 Linien haben je nur 1 Feld gelegt und tragen
    # Wert*1 bei: links-diag von Feld 0 (3*1) und Feld 1 (4*1), rechts-diag
    # von Feld 0 (2*1) und Feld 1 (6*1). Summe von Hand nachgerechnet: 25.
    partial_board = [None] * NUM_CELLS
    partial_board[0] = (5, 3, 2)
    partial_board[1] = (5, 4, 6)
    partial_total = potential_score(partial_board)
    expected_partial = 2 * 5 + 3 * 1 + 4 * 1 + 2 * 1 + 6 * 1
    assert partial_total == expected_partial, f"Erwartet {expected_partial}, bekommen {partial_total}"

    print("Alle Selbsttests erfolgreich!")
    print(f"Beispiel-Deck (erste 3 Kacheln): {deck[:3]}")
