"""
Mini-Version von Take It Easy für Phase 4 (Tabular Q-Learning).

Warum eine Mini-Version? Das echte Board (19 Felder, 27 Kacheln mit je 3
Werten) hat einen viel zu großen Zustandsraum für eine Q-Tabelle - jeder
Zustand ("Board-Belegung x aktuelle Kachel") bräuchte eine eigene Zeile,
das sind astronomisch viele Kombinationen. Tabular Q-Learning (eine simple
Tabelle statt eines neuronalen Netzes) braucht einen Zustandsraum, den man
komplett auflisten kann.

Diese Mini-Version behält die Kernidee des Originals - ein Feld kann Teil
mehrerer Linien sein, die Platzierung ist also eine echte Entscheidung unter
Unsicherheit -, reduziert aber radikal:

  - 3 Felder statt 19, in Y-Form: ein Zentrum A und zwei Arme B, C

          A
         / \
        B   C

  - Kacheln haben nur 1 Wert statt 3 (aus {1, 5, 9}, den "senkrechten"
    Werten des Originals), gleichverteilt gezogen - mit Zurücklegen, da es
    hier kein 27-Kacheln-Deck gibt, sondern nur die 3 Werte
  - 2 Linien statt 3 Richtungen: A-B und A-C, je Länge 2

Scoring (gleiches Prinzip wie im Original: Linienlänge x Wert, wenn alle
Felder der Linie gleich sind):
    score = 2*A, falls A == B
          + 2*A, falls A == C

Feld A gehört zu beiden Linien - genau hier liegt die Entscheidung: eine
frühe Kachel auf A zu legen hält beide Linien offen, ist aber riskanter
(beide Nachbarn müssten später zum selben Wert passen). Auf einen Arm (B
oder C) zu legen sichert höchstens eine Linie, dafür mit nur einer
Bedingung (nur A muss noch passen).
"""

import numpy as np

VALUES = (1, 5, 9)
CELLS = ("A", "B", "C")
LINES = (("A", "B"), ("A", "C"))


def score(board):
    """board: dict cell -> Wert (0 = leer). Summiert 2*Wert je vollständiger,
    einheitlicher Linie (Feld A zählt für bis zu zwei Linien)."""
    total = 0
    for line in LINES:
        values = [board[c] for c in line]
        if 0 not in values and len(set(values)) == 1:
            total += len(line) * values[0]
    return total


class MiniTakeItEasyEnv:
    """Bewusst kein gymnasium.Env: in Phase 4 geht es um die nackte
    Q-Learning-Update-Regel, nicht um die Gym-API (die kam schon in Phase 2).
    reset()/step() sind trotzdem bewusst analog zur echten Env benannt."""

    def __init__(self):
        self.board = None
        self.current_tile = None
        self.rng = np.random.default_rng()

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.board = {c: 0 for c in CELLS}
        self.current_tile = int(self.rng.choice(VALUES))
        return self._state()

    def _state(self):
        return (self.board["A"], self.board["B"], self.board["C"], self.current_tile)

    def valid_actions(self):
        return [i for i, c in enumerate(CELLS) if self.board[c] == 0]

    def step(self, action):
        cell = CELLS[action]
        assert self.board[cell] == 0, f"Feld {cell} ist bereits belegt"
        self.board[cell] = self.current_tile

        terminated = all(v != 0 for v in self.board.values())
        if terminated:
            reward = float(score(self.board))
            self.current_tile = None
        else:
            reward = 0.0
            self.current_tile = int(self.rng.choice(VALUES))

        return self._state(), reward, terminated


if __name__ == "__main__":
    env = MiniTakeItEasyEnv()
    state = env.reset(seed=0)
    print(f"Start-Zustand (A,B,C,Kachel): {state}")
    terminated = False
    while not terminated:
        valid = env.valid_actions()
        action = valid[0]  # einfachster Test: immer erstes freies Feld
        state, reward, terminated = env.step(action)
        print(f"Feld {CELLS[action]} belegt -> Zustand {state}, Reward {reward}")
