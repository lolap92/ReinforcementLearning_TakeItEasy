"""
Gymnasium-Environment für Take It Easy.

Installiere vor der Nutzung:
    pip install gymnasium stable-baselines3 sb3-contrib

Design-Entscheidungen (bewusst einfach gehalten für den Lernstart):
  - Observation: Board (19 Felder x 3 Werte, 0 falls leer) + aktuelle Kachel
    (3 Werte) -> alles flach als ein Vektor. Das ist die einfachste sinnvolle
    Repräsentation; Verbesserungen (z.B. One-Hot, Embeddings) sind ein
    späterer Optimierungsschritt.
  - Action: Discrete(19) - Index des Feldes, auf das die aktuelle Kachel
    gelegt wird. Bereits belegte Felder sind ungültig -> Action Masking
    (siehe get_action_mask), relevant für MaskablePPO in Phase 6.
  - Reward: standardmäßig 0 nach jedem Zug, kompletter Score erst im letzten
    Schritt (Sparse Reward - genau das macht Credit Assignment interessant).
    Opt-in Alternative: reward_shaping=True (siehe __init__) gibt zusätzlich
    bei jedem Zug ein Potential-based-Shaping-Signal (Ng et al. 1999) über
    game.py's potential_score() aus - policy-invariant, ändert die optimale
    Policy nicht, macht aber jeden Zug statt nur den letzten informativ.
    Default bleibt False, damit sich am bisherigen Verhalten nichts ändert.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from game import build_deck, score_board, potential_score, NUM_CELLS, ROWS


class TakeItEasyEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, reward_shaping=False):
        super().__init__()
        self.render_mode = render_mode
        self.reward_shaping = reward_shaping
        self._potential = 0.0

        # Action: welches der 19 Felder wird mit der aktuellen Kachel belegt
        self.action_space = spaces.Discrete(NUM_CELLS)

        # Observation: 19 Felder x 3 Werte (0 = leer) + aktuelle Kachel (3 Werte)
        # Werte liegen zwischen 0 und 9 -> Box reicht, MultiDiscrete wäre auch
        # denkbar, aber Box ist die gängigste Wahl für SB3-MLP-Policies.
        obs_len = NUM_CELLS * 3 + 3
        self.observation_space = spaces.Box(
            low=0, high=9, shape=(obs_len,), dtype=np.float32
        )

        self.board = None          # Liste[19] von Tupel(v,l,r) oder None
        self.deck = None           # verbleibende Kacheln (Ziehstapel)
        self.current_tile = None   # aktuell zu platzierende Kachel
        self.step_count = 0

    # -----------------------------------------------------------------
    # Kern-API (Pflichtmethoden von gymnasium.Env)
    # -----------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.board = [None] * NUM_CELLS
        self.deck = build_deck()
        self.np_random.shuffle(self.deck)  # nutzt den geseedeten RNG von gym.Env
        self.step_count = 0
        self.current_tile = self.deck.pop()
        self._potential = 0.0  # potential_score() eines leeren Boards ist immer 0

        observation = self._get_obs()
        info = {"action_mask": self._get_action_mask()}
        return observation, info

    def step(self, action):
        if self.board[action] is not None:
            # Ungültiger Zug (belegtes Feld). Mit Action Masking (Phase 6)
            # sollte das nie passieren - hier trotzdem sauber abgefangen,
            # damit die Env auch ohne Masking (z.B. simples DQN) nutzbar ist:
            # harte Bestrafung + Episode läuft weiter mit gleicher Kachel.
            # Board ändert sich nicht -> Potential unverändert, kein Shaping nötig.
            return self._get_obs(), -10.0, False, False, {
                "action_mask": self._get_action_mask(),
                "invalid_action": True,
            }

        # Kachel platzieren
        self.board[action] = self.current_tile
        self.step_count += 1

        terminated = self.step_count == NUM_CELLS
        reward = 0.0

        if terminated:
            reward, _ = score_board(self.board)
            self.current_tile = None
        else:
            self.current_tile = self.deck.pop()

        if self.reward_shaping:
            new_potential = potential_score(self.board)
            reward += new_potential - self._potential
            self._potential = new_potential

        observation = self._get_obs()
        info = {"action_mask": self._get_action_mask()}
        return observation, float(reward), terminated, False, info

    def render(self):
        if self.render_mode != "human":
            return
        print(f"\n--- Zug {self.step_count}/19 ---")

        # Board wie im Original: 5 Spalten (3,4,5,4,3 Felder) nebeneinander,
        # jede Spalte untereinander gestapelt und passend versetzt, sodass
        # die Sechseck-Form des physischen Bretts entsteht.
        columns = ROWS  # [[0,1,2],[3,4,5,6],[7,8,9,10,11],[12,13,14,15],[16,17,18]]
        max_height = max(len(col) for col in columns)  # 5
        total_lines = 2 * max_height - 1  # 9
        tile_width = 14

        grid = [[" " * tile_width for _ in columns] for _ in range(total_lines)]
        for col_idx, col_indices in enumerate(columns):
            offset = max_height - len(col_indices)
            for row_in_col, cell_idx in enumerate(col_indices):
                line = offset + 2 * row_in_col
                grid[line][col_idx] = self._tile_str(self.board[cell_idx]).center(tile_width)

        for line in grid:
            print("".join(line).rstrip())
        if self.current_tile is not None:
            print(f"Aktuelle Kachel: {self.current_tile}")

    @staticmethod
    def _tile_str(tile):
        if tile is None:
            return "[ . , . , . ]"
        return f"[{tile[0]},{tile[1]},{tile[2]}]"

    # -----------------------------------------------------------------
    # Hilfsfunktionen
    # -----------------------------------------------------------------

    def _get_obs(self):
        board_flat = np.zeros(NUM_CELLS * 3, dtype=np.float32)
        for i, tile in enumerate(self.board):
            if tile is not None:
                board_flat[i * 3: i * 3 + 3] = tile

        tile_part = np.zeros(3, dtype=np.float32)
        if self.current_tile is not None:
            tile_part[:] = self.current_tile

        return np.concatenate([board_flat, tile_part])

    def _get_action_mask(self):
        """Bool-Array (19,): True = Feld ist frei = gültige Aktion.
        Wird von MaskablePPO (sb3-contrib) in Phase 6 direkt genutzt."""
        return np.array([cell is None for cell in self.board], dtype=bool)


# ---------------------------------------------------------------------------
# Schneller manueller Test (führe `python env.py` aus)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    env = TakeItEasyEnv(render_mode="human")
    obs, info = env.reset(seed=0)
    print("Observation shape:", obs.shape)
    print("Erste Action-Mask (sollte alles True sein):", info["action_mask"])

    terminated = False
    total_reward = 0.0
    rng = np.random.default_rng(0)

    while not terminated:
        mask = info["action_mask"]
        valid_actions = np.where(mask)[0]
        action = rng.choice(valid_actions)  # zufällige gültige Aktion
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    env.render()
    print(f"\nFinaler Score (Random Agent, 1 Episode): {total_reward}")
