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
  - Reward: 0 nach jedem Zug, kompletter Score erst im letzten Schritt
    (Sparse Reward - genau das macht Credit Assignment interessant).
    Alternative (Reward Shaping) besprechen wir separat in Phase 5/6.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from game import build_deck, score_board, NUM_CELLS


class TakeItEasyEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

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

        observation = self._get_obs()
        info = {"action_mask": self._get_action_mask()}
        return observation, info

    def step(self, action):
        if self.board[action] is not None:
            # Ungültiger Zug (belegtes Feld). Mit Action Masking (Phase 6)
            # sollte das nie passieren - hier trotzdem sauber abgefangen,
            # damit die Env auch ohne Masking (z.B. simples DQN) nutzbar ist:
            # harte Bestrafung + Episode läuft weiter mit gleicher Kachel.
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

        observation = self._get_obs()
        info = {"action_mask": self._get_action_mask()}
        return observation, float(reward), terminated, False, info

    def render(self):
        if self.render_mode != "human":
            return
        print(f"\n--- Zug {self.step_count}/19 ---")
        rows = [[0,1,2],[3,4,5,6],[7,8,9,10,11],[12,13,14,15],[16,17,18]]
        middle_row = [7, 8, 9, 10, 11]
        for row_indices in rows:
            if row_indices == middle_row:
                for i in row_indices:
                    print(self._tile_str(self.board[i]).center(40))
            else:
                row_str = "  ".join(
                    self._tile_str(self.board[i]) for i in row_indices
                )
                print(row_str.center(40))
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
