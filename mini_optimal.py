"""
Exakte optimale Lösung des Mini-Boards per Backward Induction (Dynamic
Programming) - dient als Referenzwert ("Ground Truth"), gegen den sich die
gelernte Q-Tabelle aus qlearning.py prüfen lässt.

Bei nur 3 Feldern und 3 möglichen Kachelwerten lässt sich der exakt optimale
erwartete Score direkt ausrechnen, ganz ohne Simulation oder Lernen: für
jeden möglichen Zustand wird rekursiv der beste erreichbare Erwartungswert
berechnet (Bellman-Optimalitätsgleichung, exakt statt geschätzt). Q-Learning
muss diesen Wert im Grenzfall (genug Episoden) treffen - das macht die
Mini-Version zu einem guten Test, ob die Lernregel tatsächlich funktioniert.
"""

from functools import lru_cache

from mini_env import CELLS, VALUES, score


@lru_cache(maxsize=None)
def optimal_value(board, tile):
    """board: Tupel (a, b, c), 0 = leer. Erwarteter Score bei optimalem
    Spiel ab diesem Zustand (vor der Entscheidung, wohin `tile` kommt)."""
    board_dict = dict(zip(CELLS, board))
    valid = [i for i, c in enumerate(CELLS) if board_dict[c] == 0]

    best = 0.0
    for action in valid:
        best = max(best, _action_value(board_dict, action, tile))
    return best


def _action_value(board_dict, action, tile):
    cell = CELLS[action]
    new_board = dict(board_dict)
    new_board[cell] = tile

    if all(v != 0 for v in new_board.values()):
        return float(score(new_board))

    new_board_tuple = tuple(new_board[c] for c in CELLS)
    # Erwartungswert über die noch unbekannte nächste Kachel (gleichverteilt)
    return sum(optimal_value(new_board_tuple, t) for t in VALUES) / len(VALUES)


def optimal_action_values(board, tile):
    """Q*-Wert für JEDE gültige Aktion (nicht nur die beste) - damit lässt
    sich prüfen, ob eine gelernte Aktion optimal ist, auch wenn mehrere
    Aktionen exakt gleich gut sind (z.B. die beiden symmetrischen Arme B/C:
    ein einzelner "bester" Referenzwert würde eine der beiden willkürlich
    bevorzugen und einen fair auswürfelnden Agenten fälschlich bestrafen)."""
    board_dict = dict(zip(CELLS, board))
    valid = [i for i, c in enumerate(CELLS) if board_dict[c] == 0]
    return {action: _action_value(board_dict, action, tile) for action in valid}


def optimal_action(board, tile):
    """Bestes Feld (Action-Index) für `tile` im Zustand `board`, plus dessen Q-Wert."""
    board_dict = dict(zip(CELLS, board))
    valid = [i for i, c in enumerate(CELLS) if board_dict[c] == 0]

    best_action, best_q = valid[0], -1.0
    for action in valid:
        q = _action_value(board_dict, action, tile)
        if q > best_q:
            best_action, best_q = action, q
    return best_action, best_q


def expected_optimal_score():
    empty = (0, 0, 0)
    return sum(optimal_value(empty, t) for t in VALUES) / len(VALUES)


if __name__ == "__main__":
    print(f"Exakt optimaler erwarteter Score (Backward Induction): {expected_optimal_score():.4f}")
    print(f"Zustände im Cache: {optimal_value.cache_info().currsize}")

    print("\nOptimale Erstzug-Entscheidung je gezogener Kachel (leeres Board):")
    for t in VALUES:
        action, q = optimal_action((0, 0, 0), t)
        print(f"  Kachel={t}  ->  Feld {CELLS[action]}  (erwarteter Score ab hier: {q:.3f})")
