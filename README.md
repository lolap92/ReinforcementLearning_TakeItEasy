# Take It Easy – RL Lernprojekt

Reinforcement Learning anhand des Brettspiels **Take It Easy**. Ein Agent
lernt, 19 zufällig gezogene Kacheln nacheinander auf ein 19-Felder-Hexboard
zu legen, um am Ende möglichst viele vollständige Linien zu bilden. Fokus
des Projekts liegt auf dem **Verständnis der RL-Grundlagen**, nicht primär
auf State-of-the-Art-Performance – Details zum Lernpfad und den bisherigen
Ergebnissen stehen in [`PROJECT_HANDOFF.md`](PROJECT_HANDOFF.md), eine
Zeile pro Trainingslauf in [`EXPERIMENTS.md`](EXPERIMENTS.md).

Warum Take It Easy als RL-Testfeld: kleiner Zustandsraum (19 Felder), fixe
Episodenlänge (genau 19 Züge), stochastische Kachelzüge, **Sparse Reward**
(der Score kommt erst am Ende der Episode – gutes Beispiel für das
Credit-Assignment-Problem), kein Gegner.

**Stand nach Phase 7:** der beste trainierte Agent (MaskablePPO, 25 Mio.
Steps) kommt auf Ø 108,9 Punkte – und liegt damit *unter* zwei untrainierten
Heuristiken in `baselines.py` (`greedy_potential` Ø 120,9, `expected_value`
Ø 128,9). Die ursprüngliche `greedy`-Baseline (Ø 27,6) war als Messlatte
unbrauchbar, weil sie mit `score_board()` bewertet und dadurch 15 Züge lang
faktisch zufällig legt. Analyse und Empfehlungen:
[`reports/phase7_analysis_report.html`](reports/phase7_analysis_report.html).

Zwei RL-Algorithmen werden trainiert und verglichen:

- **DQN** ([`train_dqn.py`](train_dqn.py)) – value-based, lernt Q(s,a), mit
  eigenem Action-Masking-Hack (`MaskedDQN`/`MaskedQNetwork`), da
  Stable-Baselines3 kein natives Masking für DQN mitbringt
- **MaskablePPO** ([`train_ppo.py`](train_ppo.py)) – policy-based, lernt
  direkt π(a|s), mit nativem Action Masking über `sb3-contrib`

Action Masking ist in beiden Fällen notwendig, nicht optional: im letzten
Zug sind 18 von 19 Feldern belegt, ohne Masking probiert/lernt die Policy
ständig ungültige Züge (siehe Docstring in `train_dqn.py` für einen
konkreten Divergenz-Befund ohne Masking).

## Setup

```bash
pip install -r requirements.txt
```

Enthält `gymnasium`, `stable-baselines3`, `sb3-contrib`, `numpy`,
`tensorboard`. Für GPU-Training zusätzlich PyTorch mit CUDA-Support
installieren (Standard-`pip install torch` liefert oft nur die CPU-Variante):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

## Training

### DQN

```bash
python train_dqn.py --timesteps 300000
python train_dqn.py --timesteps 1000000 --device cuda --tag mein_lauf
```

| Parameter | Typ | Default | Bedeutung |
|---|---|---|---|
| `--timesteps` | int | `300000` | Anzahl Environment-Steps zum Trainieren |
| `--eval-episodes` | int | `1000` | Anzahl Episoden für die finale Auswertung nach dem Training |
| `--seed` | int | `0` | Master-Seed (Trainingslauf; Eval-Episoden nutzen `seed + 1 + i`) |
| `--tag` | str | automatisch (z.B. `dqn_300k_masked`) | Freitext, wird Teil des Run-Ordner-Namens unter `experiments/`. Ohne eigenen Wert wird er aus den Settings gebaut, damit Steps/Modus direkt aus dem Namen hervorgehen |
| `--device` | str | `cpu` | `cpu` (Default), `cuda` (GPU erzwingen) oder `auto` (SB3 wählt). Bei diesem kleinen Netz (128,128) ist der Flaschenhals meist der CPU-seitige Env-Step, GPU bringt oft wenig |

### MaskablePPO

```bash
python train_ppo.py --timesteps 300000
python train_ppo.py --timesteps 1000000 --device cuda --tag mein_lauf
python train_ppo.py --timesteps 1000000 --n-envs 8   # 8 Environments parallel
```

| Parameter | Typ | Default | Bedeutung |
|---|---|---|---|
| `--timesteps` | int | `300000` | Anzahl Environment-Steps zum Trainieren |
| `--eval-episodes` | int | `1000` | Anzahl Episoden für die finale Auswertung nach dem Training |
| `--seed` | int | `0` | Master-Seed (Trainingslauf; Eval-Episoden nutzen `seed + 1 + i`) |
| `--tag` | str | automatisch (z.B. `ppo_1m_8envs_rewardshaping`) | Freitext, wird Teil des Run-Ordner-Namens unter `experiments/`. Ohne eigenen Wert wird er aus Steps, `n_envs` und allen vom Default abweichenden Wave-1/2-Flags gebaut |
| `--device` | str | `cpu` | `cpu` (Default), `cuda` (GPU erzwingen) oder `auto` (SB3 wählt) – gleicher Hinweis wie bei DQN |
| `--n-envs` | int | `1` | Anzahl paralleler Trainings-Environments. `1` = wie bisher ein einzelner Prozess; `>1` läuft über separate Prozesse (`SubprocVecEnv`) – vielfältigere, weniger korrelierte Trajektorien pro Policy-Update und bessere CPU-Auslastung. Achtung: Gesamt-Batchgröße pro Update ist `n_steps * n_envs` |
| `--n-steps` | int | `max(32, 512 // n_envs)` | Rollout-Länge pro Environment vor jedem Policy-Update. Wird automatisch mit `n_envs` runterskaliert, damit die Anzahl Policy-Updates bei mehr parallelen Envs nicht einbricht – ein 1-Mio.-Lauf mit `n_envs=8` und unverändertem `n_steps=512` hatte nur noch 1/8 so viele Updates und schnitt dadurch schlechter ab (92,8 → 83,4 Mean-Score) |
| `--no-normalize` | flag | aus | Schaltet die Reward-Normalisierung (`VecNormalize`, nur Reward, nicht Observation) ab, die standardmäßig an ist |
| `--constant-lr` | flag | aus | Konstante Lernrate (3e-4) statt des linear auf 0 abfallenden Default-Schedules |
| `--reward-shaping` | flag | aus | Potential-based Reward Shaping (siehe `env.py`) – gibt bei jedem Zug statt nur im letzten ein Lernsignal, bleibt aber policy-invariant (Ng et al. 1999). Wirkt nur auf das Training; `eval/mean_reward` und die finale Auswertung nutzen immer den echten, ungeshapten Score. Mit Shaping ist `rollout/score_mean` in TensorBoard rechnerisch ~2× der echten Score-Größenordnung – kein Bug |

Weitere Default-Änderungen ggü. dem ursprünglichen Phase-6-Stand: Value-Function jetzt größer als die Policy (`net_arch={"pi": [128,128], "vf": [256,256]}`), da sie mit dem Sparse-Reward die schwerere Lernaufgabe hat.

### Hyperparameter-Sweep

Statt eine Konfiguration zu erraten und dafür gleich Millionen Steps zu
investieren, testet `sweep_ppo.py` mehrere Konfigurationen bei kleinem,
gemeinsamem Budget gegeneinander (mehrere Seeds je Konfiguration, um
Lauf-zu-Lauf-Streuung nicht mit echten Unterschieden zu verwechseln):

```bash
python sweep_ppo.py
python sweep_ppo.py --timesteps 200000 --seeds 0 1 2
python sweep_ppo.py --configs baseline reward_shaping   # nur diese testen
```

Ruft `train_ppo.py` je Kombination aus Konfiguration × Seed per Subprocess
auf, jeder Einzellauf landet normal unter `experiments/<run_id>/` und in
`EXPERIMENTS.md`; am Ende druckt das Skript eine nach Mittelwert sortierte
Rangliste. Die Gewinner-Konfiguration danach mit deutlich mehr `--timesteps`
final trainieren.

### Baselines

```bash
python baselines.py --episodes 2000
```

Vier Referenz-Agenten ohne jedes Training, als Messlatte für die trainierten
Modelle:

| Agent | Ø | Bewertungsfunktion |
|---|---|---|
| `random` | 10,8 | – |
| `greedy` | 27,6 | `score_board()` – zählt nur *fertige* Linien, deshalb während der ersten ~15 Züge praktisch blind. Nur noch aus historischen Gründen dabei |
| `greedy_potential` | 120,9 | `potential_score()` – bewertet auch angefangene, noch lebendige Linien |
| `expected_value` | 128,9 | Erwarteter Endscore je Linie unter Berücksichtigung des **Restdecks** |

`greedy_potential` ist die Baseline, an der sich ein trainierter Agent messen
lassen muss – nicht `greedy`.

Beide Trainingsskripte legen ihre Ergebnisse unter `experiments/<run_id>/` ab
(`run_id` = Zeitstempel + `--tag`): `config.json`/`summary.json` (git-tracked,
klein), `models/`/`tensorboard/` (gitignored, groß/binär, lokal aus
`config.json` reproduzierbar – siehe `experiments/README.md`). Zusätzlich
wird automatisch eine Zeile in `EXPERIMENTS.md` angehängt.

## TensorBoard

Während oder nach einem Trainingslauf, in einem zweiten Terminal:

```bash
python -m tensorboard.main --logdir experiments
```

Danach im Browser öffnen: **[http://localhost:6006](http://localhost:6006)**

Relevante Tags im Dashboard:

- `rollout/score_mean`, `rollout/score_min`, `rollout/score_max` – Spiel-Score
  (nicht nur Reward) über die letzten 100 Trainings-Episoden
- `eval/mean_reward` – Score des Eval-Checkpoints, alle 10.000 Steps
  automatisch mitgeloggt

`--logdir experiments` bündelt alle Runs aller Algorithmen gemeinsam im
selben Dashboard – einzelne Läufe lassen sich über die Run-Auswahl links
in TensorBoard vergleichen.

## Replay

Eine einzelne Episode mit einem gespeicherten Modell nachspielen und das
fertige Board anzeigen (Seed aus der `episodes.csv`-Zeile des jeweiligen
Runs, z. B. um die beste oder schlechteste Runde eines Laufs anzusehen):

```bash
python replay.py --model experiments/<run_id>/models/final_model.zip --algo ppo --seed 195
python replay.py --model experiments/<run_id>/models/best_model.zip --algo dqn --seed 42 --html
```

| Parameter | Typ | Pflicht | Bedeutung |
|---|---|---|---|
| `--model` | str | ja | Pfad zu einer `.zip`-Modelldatei (`final_model.zip` oder `best_model.zip`); `models/` ist gitignored, muss also lokal aus dem jeweiligen Trainingslauf noch vorhanden sein |
| `--algo` | `dqn` \| `ppo` | ja | Welcher Algorithmus/welche Modellklasse geladen wird |
| `--seed` | int | ja | Seed aus der `episodes.csv`-Zeile der gewünschten Runde (nicht `episode_index` – nur der Seed reproduziert exakt die gleiche Kachelreihenfolge) |
| `--html` | str, optional | nein | Erzeugt zusätzlich eine visuelle HTML-Ansicht des fertigen Boards, landet immer im `replay/`-Ordner (gitignored). Mit Dateinamen: `--html board.html` → `replay/board.html`. Ohne Wert: `--html` → automatisch `replay/<algo>_<seed>.html` |

Ohne `--html` gibt `replay.py` das Board nur als Textausgabe im Terminal aus.
