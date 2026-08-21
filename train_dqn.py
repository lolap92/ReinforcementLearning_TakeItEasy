"""
DQN via Stable-Baselines3 auf dem vollen Take-It-Easy-Board (Phase 5).

Für lokale Ausführung gedacht: das Training braucht deutlich mehr Rechenzeit
als die vorigen Phasen (zehntausende bis Millionen Environment-Schritte),
und TensorBoard live mitzuverfolgen funktioniert nur mit einem lokal
laufenden Prozess. Siehe Setup-Hinweise ganz unten in dieser Datei.

Design-Entscheidung (revidiert): MIT Action Masking, nicht ohne.
Die erste Version dieses Skripts hatte bewusst auf Masking verzichtet
(nur die eingebaute -10-Bestrafung aus env.py für ungültige Züge) - das
war als "Value-based ohne Masking vs. Policy-based mit Masking"-Vergleich
für Phase 5 vs. 6 gedacht. Ein echter 1-Mio-Timesteps-Lauf hat aber
gezeigt: das funktioniert nicht bloß langsamer, es divergiert. Episodenlänge
und negativer Reward liefen im Training immer weiter auseinander (siehe
experiments/2026-08-16_0738_phase5_dqn/ - Score blieb bei 0,0 über alle
1000 Auswertungs-Episoden). Grund: eine ungültige Aktion führt zu genau
demselben Folgezustand (Board ändert sich nicht) - der Bellman-Update für
diesen Fall ist dadurch selbstreferentiell (Q(s,a) bootstrapt von Q(s,*)
desselben Zustands), was sich mit wachsender Episodenlänge aufschaukelt,
zusätzlich verschärft durch gamma=1.0 (keine Abzinsung, die das begrenzen
könnte) und einen mit ungültigen Übergängen zugemüllten Replay-Buffer.

Ein Vergleichslauf mit Masking (gleiches Budget, 60k Schritte testweise)
zeigte stattdessen sofortige, stabile Konvergenz auf ~19 Schritte/Episode
und positiven Score. Masking ist hier also keine Optimierung, sondern
Voraussetzung dafür, dass DQN auf diesem Environment überhaupt lernt.
Der Vergleich Value-based vs. Policy-based in Phase 6 findet deshalb jetzt
bei gleicher Ausgangslage statt: beide Algorithmen mit Masking.

Stable-Baselines3 hat für DQN kein eingebautes Masking (anders als
sb3-contribs MaskablePPO für Phase 6). Umgesetzt über die offiziellen
Erweiterungspunkte von SB3: eine QNetwork-Unterklasse maskiert die
Q-Werte vor dem argmax, eine DQNPolicy-Unterklasse verwendet dieses
Netz, und DQN.predict() wird überschrieben, damit auch die zufällige
Exploration (epsilon-greedy) nur unter gültigen Feldern auswählt.
Zusätzlich überschreibt MaskedDQN auch _sample_action(): SB3s Original
zieht dort vor learning_starts komplett ungemaskt über
self.action_space.sample() (reiner Warmup vor jeglichem Lernen), was
sonst learning_starts Steps lang den Replay-Buffer mit größtenteils
ungültigen Zügen füllt. Die Maske wird direkt aus der Beobachtung
abgeleitet (ein Feld ist leer, wenn alle 3 zugehörigen Werte 0 sind,
siehe env.py _get_obs) - der Observation Space musste dafür nicht
verändert werden.

Nutzung (lokal, in deiner IDE):
    pip install -r requirements.txt
    python train_dqn.py --timesteps 300000
    python train_dqn.py --timesteps 1000000 --device cuda   # GPU erzwingen

    # In einem zweiten Terminal, um live mitzuverfolgen:
    tensorboard --logdir experiments
    # dann im Browser: http://localhost:6006
    # rollout/score_mean/min/max = Spiel-Score (nicht nur Reward) über die
    # letzten 100 Trainings-Episoden, eval/mean_reward = Score des
    # Eval-Checkpoints alle 10k Steps

Nach dem Training landet alles unter experiments/<run_id>/:
  - config.json, summary.json          -> ins Git-Repo committen
  - models/, tensorboard/              -> bewusst NICHT im Repo (.gitignore),
                                           lokal reproduzierbar aus config.json
                                           (gleicher Seed + gleicher Git-Commit)
"""

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch as th
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor

from env import TakeItEasyEnv, NUM_CELLS

REPO_ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
EXPERIMENTS_LOG = REPO_ROOT / "EXPERIMENTS.md"

# Sicherheitsnetz (defense-in-depth): mit korrektem Masking sollte eine
# Episode nie länger als 19 Züge dauern. Das Limit fängt trotzdem jeden
# unvorhergesehenen Fall ab (z.B. während der Entwicklung), statt dass ein
# Lauf endlos hängen bleibt.
MAX_EPISODE_STEPS = 200


def obs_to_mask(obs):
    """Leitet die Action-Mask direkt aus der Beobachtung ab, ohne den
    Observation Space zu verändern: die ersten NUM_CELLS*3 Werte sind die
    Board-Belegung (3 Werte je Feld), ein Feld ist leer <=> alle 3 Werte
    sind 0 (siehe env.py _get_obs). Funktioniert für einzelne (60,) und
    gebatchte (n, 60) Beobachtungen gleichermaßen."""
    board_part = obs[..., :NUM_CELLS * 3].reshape(*obs.shape[:-1], NUM_CELLS, 3)
    return (board_part == 0).all(axis=-1)


class MaskedQNetwork(QNetwork):
    """Setzt die Q-Werte ungültiger Felder vor dem argmax auf -inf, statt
    sie wie Standard-DQN einfach mitzubewerten."""

    def _predict(self, observation, deterministic=True):
        q_values = self(observation)
        mask = obs_to_mask(observation.detach().cpu().numpy())
        mask_t = th.as_tensor(mask, device=q_values.device)
        q_values = q_values.masked_fill(~mask_t, -1e8)
        return q_values.argmax(dim=1).reshape(-1)


class MaskedDQNPolicy(DQNPolicy):
    def make_q_net(self):
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return MaskedQNetwork(**net_args).to(self.device)


def masked_random_action(mask):
    """Zufällige Aktion nur unter gültigen (unmaskierten) Feldern - gemeinsam
    genutzt für epsilon-greedy-Exploration (predict) und die Zufalls-Warmup-
    Phase vor learning_starts (_sample_action)."""
    if mask.ndim == 1:
        return np.array(np.random.choice(np.flatnonzero(mask)))
    return np.array([np.random.choice(np.flatnonzero(m)) for m in mask])


class MaskedDQN(DQN):
    """Überschreibt predict() UND _sample_action(), damit wirklich jede
    Zufallsaktion nur unter gültigen Feldern gewählt wird - nicht nur die
    epsilon-greedy-Exploration nach dem Warmup (predict), sondern auch die
    initiale Zufalls-Phase vor learning_starts. SB3s Basisimplementierung von
    _sample_action zieht dort über self.action_space.sample() komplett ohne
    Masking, was learning_starts Steps lang den Replay-Buffer mit größtenteils
    ungültigen Zügen füllt (der Greedy-Pfad ist schon über
    MaskedQNetwork/MaskedDQNPolicy abgedeckt)."""

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        if not deterministic and np.random.rand() < self.exploration_rate:
            mask = obs_to_mask(observation)
            return masked_random_action(mask), state
        return super().predict(observation, state, episode_start, deterministic)

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            assert self._last_obs is not None, "self._last_obs was not set"
            mask = obs_to_mask(self._last_obs)
            unscaled_action = masked_random_action(mask)
        else:
            assert self._last_obs is not None, "self._last_obs was not set"
            unscaled_action, _ = self.predict(self._last_obs, deterministic=False)
        # Discrete-Action-Space (wie hier): keine Rescale/Noise-Logik nötig,
        # die SB3s Original nur für Box-Spaces braucht - buffer_action == action.
        return unscaled_action, unscaled_action


class ScoreLoggingCallback(BaseCallback):
    """Loggt den Spiel-Score separat unter eigenen TensorBoard-Tags
    (`rollout/score_mean`, `rollout/score_min`, `rollout/score_max`), statt
    ihn nur implizit unter dem generischen SB3-Tag `rollout/ep_rew_mean`
    mitlaufen zu lassen. Rechnerisch ist Mean identisch zu ep_rew_mean: der
    Reward ist 0 in jedem Zug außer dem letzten, wo er genau dem Spiel-Score
    entspricht (siehe env.py), die Episodensumme des Rewards *ist* also der
    Score. Min/Max zeigen zusätzlich die beste/schlechteste Runde innerhalb
    des rollierenden Fensters (die letzten 100 Episoden, SB3-Default) - das
    eigene Tag macht das im TensorBoard-Dashboard sofort ohne Nachdenken
    ablesbar."""

    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0:
            scores = [ep_info["r"] for ep_info in self.model.ep_info_buffer]
            self.logger.record("rollout/score_mean", float(np.mean(scores)))
            self.logger.record("rollout/score_min", float(np.min(scores)))
            self.logger.record("rollout/score_max", float(np.max(scores)))
        return True


def git_commit_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return None


def evaluate(model, n_episodes, seed):
    env = TakeItEasyEnv()
    scores = np.empty(n_episodes)
    invalid_counts = np.empty(n_episodes)
    for i in range(n_episodes):
        obs, info = env.reset(seed=seed + i)
        terminated = False
        total_reward = 0.0
        invalid = 0
        steps = 0
        while not terminated and steps < MAX_EPISODE_STEPS:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            if info.get("invalid_action"):
                invalid += 1
            elif terminated:
                total_reward = reward
            steps += 1
        scores[i] = total_reward
        invalid_counts[i] = invalid
    return scores, invalid_counts


def steps_label(n):
    """Kurzform für Timesteps in Tags/Run-IDs, z.B. 300_000 -> '300k', 1_000_000 -> '1m'."""
    if n % 1_000_000 == 0:
        return f"{n // 1_000_000}m"
    if n % 1_000 == 0:
        return f"{n // 1000}k"
    return str(n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN-Training für Take It Easy (Phase 5, lokal ausführen).")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--eval-episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--tag", type=str, default=None,
        help="Freitext für den Run-Ordner-Namen. Default: automatisch aus den "
             "Settings gebaut (z.B. 'dqn_300k_masked'), damit Steps/Modus/"
             "Besonderheiten aus dem Namen hervorgehen, ohne dass man selbst "
             "daran denken muss.",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="'cpu' (Default), 'cuda' (GPU erzwingen) oder 'auto' (SB3 wählt). "
             "Hinweis: bei einem so kleinen Netz (128,128) und Batch-Size 64 "
             "ist der Flaschenhals meist der CPU-seitige Env-Step, nicht das "
             "Netz selbst - GPU bringt hier oft wenig bis nichts, schadet "
             "aber auch nicht.",
    )
    args = parser.parse_args()
    if args.tag is None:
        args.tag = f"dqn_{steps_label(args.timesteps)}_masked"

    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    run_id = f"{timestamp[:16].replace(':', '').replace('T', '_')}_{args.tag}"
    run_dir = EXPERIMENTS_DIR / run_id
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    (run_dir / "tensorboard").mkdir(exist_ok=True)

    # TimeLimit auf beiden Envs als zusätzliches Sicherheitsnetz (siehe oben) -
    # mit funktionierendem Masking sollte es nie greifen.
    train_env = Monitor(TimeLimit(TakeItEasyEnv(), max_episode_steps=MAX_EPISODE_STEPS))
    eval_env = Monitor(TimeLimit(TakeItEasyEnv(), max_episode_steps=MAX_EPISODE_STEPS))

    model = MaskedDQN(
        MaskedDQNPolicy,
        train_env,
        learning_rate=5e-4,
        buffer_size=100_000,
        learning_starts=1_000,
        batch_size=64,
        gamma=1.0,  # keine Abzinsung - die Zielgröße ist der Score am Episodenende, alle 19 Züge zählen gleich
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1_000,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        policy_kwargs=dict(net_arch=[128, 128]),
        tensorboard_log=str(run_dir / "tensorboard"),
        seed=args.seed,
        device=args.device,
        verbose=1,
    )
    print(f"Gewähltes Device: {model.device}")

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "models"),
        log_path=str(run_dir / "tensorboard"),
        eval_freq=10_000,
        n_eval_episodes=50,
        deterministic=True,
    )
    score_callback = ScoreLoggingCallback()

    print(f"Trainiere DQN (mit Action Masking) für {args.timesteps} Timesteps ...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=CallbackList([score_callback, eval_callback]),
        tb_log_name="dqn",
    )
    model.save(str(run_dir / "models" / "final_model"))

    print("\nAuswertung über echte Take-It-Easy-Episoden (19 Züge, volles Board) ...")
    scores, invalid_counts = evaluate(model, args.eval_episodes, seed=args.seed + 1)
    print(f"DQN (Endmodell nach {args.timesteps} Steps):  mean={scores.mean():.2f}  "
          f"std={scores.std():.2f}  min={scores.min():.0f}  max={scores.max():.0f}")
    print(f"Ungültige Züge je Episode (Ø, sollte 0 sein): {invalid_counts.mean():.2f}")

    # DQN ist bekanntermaßen instabil über lange Trainingsläufe (Q-Value-
    # Überschätzung, siehe Docstring oben) - das Endmodell kann schlechter
    # sein als ein Zwischenstand. EvalCallback speichert deshalb laufend das
    # beste Zwischen-Checkpoint (best_model.zip) nach eval/mean_reward -
    # das werten wir hier zusätzlich aus, um genau diesen Fall sichtbar zu
    # machen, statt ihn stillschweigend im Endmodell-Ergebnis zu verstecken.
    best_model_path = run_dir / "models" / "best_model.zip"
    best_scores = None
    if best_model_path.exists():
        best_model = MaskedDQN.load(str(best_model_path), env=train_env, device=args.device)
        best_scores, best_invalid_counts = evaluate(best_model, args.eval_episodes, seed=args.seed + 1)
        print(f"DQN (bestes Zwischen-Checkpoint):           mean={best_scores.mean():.2f}  "
              f"std={best_scores.std():.2f}  min={best_scores.min():.0f}  max={best_scores.max():.0f}")
        print(f"Ungültige Züge je Episode (Ø, sollte 0 sein): {best_invalid_counts.mean():.2f}")
        if best_scores.mean() > scores.mean():
            print(
                "-> Bestes Checkpoint schlägt Endmodell deutlich: Hinweis auf "
                "Instabilität/Überschätzung spät im Training (siehe Docstring)."
            )

    config = {
        "run_id": run_id,
        "tag": args.tag,
        "timestamp": timestamp,
        "git_commit": git_commit_hash(),
        "env": "TakeItEasyEnv",
        "algorithm": "DQN (stable-baselines3, mit Action Masking)",
        "hyperparameters": {
            "timesteps": args.timesteps,
            "learning_rate": 5e-4,
            "buffer_size": 100_000,
            "learning_starts": 1_000,
            "batch_size": 64,
            "gamma": 1.0,
            "train_freq": 4,
            "target_update_interval": 1_000,
            "exploration_fraction": 0.3,
            "exploration_final_eps": 0.05,
            "net_arch": [128, 128],
        },
        "eval_episodes": args.eval_episodes,
        "master_seed": args.seed,
        "device": str(model.device),
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    summary = [{
        "name": "dqn_full_board_masked",
        "n_episodes": args.eval_episodes,
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "min": float(scores.min()),
        "max": float(scores.max()),
        "avg_invalid_actions_per_episode": float(invalid_counts.mean()),
    }]
    if best_scores is not None:
        summary.append({
            "name": "dqn_full_board_masked_best_checkpoint",
            "n_episodes": args.eval_episodes,
            "mean": float(best_scores.mean()),
            "std": float(best_scores.std()),
            "min": float(best_scores.min()),
            "max": float(best_scores.max()),
            "avg_invalid_actions_per_episode": float(best_invalid_counts.mean()),
        })
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(run_dir / "episodes.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["agent", "episode_index", "seed", "score"])
        for i, score in enumerate(scores):
            writer.writerow(["dqn_full_board_masked", i, args.seed + 1 + i, score])
        if best_scores is not None:
            for i, score in enumerate(best_scores):
                writer.writerow(["dqn_full_board_masked_best_checkpoint", i, args.seed + 1 + i, score])

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
                f"{s['n_episodes']} | {s['mean']:.2f} | {s['std']:.2f} | "
                f"{s['min']:.0f} | {s['max']:.0f} | {config['git_commit'] or '-'} |\n"
            )

    print(f"\nRun gespeichert unter: {run_dir.relative_to(REPO_ROOT)}")
    print("models/ und tensorboard/ sind gitignored (groß/binär) - config.json und")
    print("summary.json reichen, um den Lauf ins Repo zu committen.")
