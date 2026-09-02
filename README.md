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

Drei Ansätze werden trainiert und verglichen:

- **DQN** ([`train_dqn.py`](train_dqn.py)) – value-based, lernt Q(s,a), mit
  eigenem Action-Masking-Hack (`MaskedDQN`/`MaskedQNetwork`), da
  Stable-Baselines3 kein natives Masking für DQN mitbringt
- **MaskablePPO** ([`train_ppo.py`](train_ppo.py)) – policy-based, lernt
  direkt π(a|s), mit nativem Action Masking über `sb3-contrib`
- **Afterstate-Wertfunktion** ([`train_afterstate.py`](train_afterstate.py)) –
  lernt `V(Zustand nach dem Legen)` statt einer Policy und wählt zur Spielzeit
  `argmax` über die ≤19 möglichen Folgezustände. Kein SB3, sondern direkt
  PyTorch – die Konstruktion passt nicht in die SB3-API, weil sie die bekannte
  Dynamik des Spiels ausnutzt statt sie zu lernen

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

### Afterstate-Wertfunktion (Phase 8)

```bash
python train_afterstate.py
python train_afterstate.py --episodes 1000000 --device cuda
python train_afterstate.py --lam 1.0 --value-head scalar     # Ablation
```

Statt einer Policy wird `V(Afterstate)` gelernt – der erwartete Endscore des
Boards *nachdem* die Kachel gelegt wurde. Zur Spielzeit wird einfach `argmax`
über alle freien Felder gebildet; das ist bereits eine 1-Ply-Suche und nutzt
aus, dass die Dynamik des Spiels vollständig bekannt ist (nach dem Legen ist
der Folgezustand deterministisch, der einzige Zufall ist eine bekannte
Gleichverteilung über das Restdeck). Gleiche Konstruktion wie bei TD-Gammon
und den starken 2048-Agenten. Ausführliche Begründung im Docstring des
Skripts und in [`reports/phase7_analysis_report.html`](reports/phase7_analysis_report.html).

| Parameter | Typ | Default | Bedeutung |
|---|---|---|---|
| `--episodes` | int | `300000` | Selbstspiel-Episoden insgesamt (eine Episode = 19 Züge) |
| `--games-per-iter` | int | `256` | Partien pro Iteration, alle parallel im Gleichschritt. Dadurch lassen sich pro Zug sämtliche Zugmöglichkeiten aller Partien in *einem* Forward-Pass bewerten |
| `--epochs` | int | `4` | Durchläufe über die frisch gesammelten Daten je Iteration |
| `--batch-size` | int | `512` | Minibatch-Größe der Gradientenschritte |
| `--lr` | float | `1e-3` | Startlernrate, fällt cosinusförmig auf 0 |
| `--lam` / `--lambda` | float | `0.7` | λ-Return: `1.0` = Monte Carlo (unverzerrt, hohe Varianz), `0.0` = TD(0) |
| `--epsilon-start` / `--epsilon-end` | float | `0.20` / `0.01` | Exploration, linear fallend. Die Zielwerte nutzen immer `max_a V`, sind also off-policy und werden von der Exploration nicht verzerrt |
| `--hidden` | str | `512,512,512` | Schichtgrößen des MLP |
| `--value-head` | `twohot` \| `scalar` | `twohot` | `twohot`: Score-Bereich 0–307 in `--atoms` Stützstellen, Cross-Entropy gegen die Two-Hot-Kodierung des Ziels, Wert = Erwartungswert der Verteilung. Bei einer Zielgröße, die aus wenigen großen Sprüngen besteht, lernt das verlässlicher als MSE. `scalar`: klassische Huber-Regression |
| `--atoms` | int | `51` | Stützstellen des Two-Hot-Kopfs |
| `--no-line-features` | flag | aus | Die 15 × 8 Linien-Features weglassen (Ablation: wieviel trägt die Feature-Konstruktion, wieviel das Netz bei?) |
| `--eval-every` / `--eval-games` | int | `25` / `1024` | Zwischenauswertung (greedy, vektorisiert) für TensorBoard und die Best-Checkpoint-Auswahl |
| `--eval-episodes` | int | `2000` | Finale Auswertung auf der echten `TakeItEasyEnv`, gleiche Seeds wie `train_ppo.py` und `baselines.py` |
| `--device` | str | `cpu` | Anders als bei PPO lohnt sich hier eine GPU eher: pro Zug werden bis zu `games-per-iter × 19` Afterstates in einem Batch bewertet |

Was das Netz sieht (und `env.py` nicht liefert): 19 × 10 One-Hot fürs Board,
**27 Dimensionen Restdeck-Multi-Hot**, 9 Zähler je (Richtung, Wert), 15 × 8
Linien-Features (Füllstand, lebendig, aktueller Wert, `potential_score()`-
Beitrag, passende Kacheln im Deck, Komplettierungswahrscheinlichkeit) und 2
Fortschrittswerte – zusammen 348 Dimensionen statt der 60 rohen Zahlen aus
`env.py`.

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

Alle Trainingsskripte legen ihre Ergebnisse unter `experiments/<run_id>/` ab
(`run_id` = Zeitstempel + `--tag`): `config.json`/`summary.json` (git-tracked,
klein), `models/`/`tensorboard/` (gitignored, groß/binär, lokal aus
`config.json` reproduzierbar – siehe `experiments/README.md`). Zusätzlich
wird automatisch eine Zeile in `EXPERIMENTS.md` angehängt.

## Hindsight-Orakel

```bash
python oracle.py --episodes 200
python oracle.py --episodes 200 --compare experiments/<run_id>/episodes.csv
```

Beantwortet je Episode: *gegeben genau die 19 Kacheln, die in dieser Partie
gezogen wurden – was wäre der bestmögliche Score bei freier Platzierung
gewesen?*

Die Ziehreihenfolge wird dabei bewusst ignoriert, und das ist exakt richtig:
weil jede Kachel auf jedes freie Feld darf, ist *jede* Bijektion
Kacheln → Felder in *jeder* Reihenfolge realisierbar (lege `t_i` auf sein
Zielfeld – das ist nie belegt, weil die Zuordnung injektiv ist). Das Orakel
ist damit exakt der Wert des hellsehenden Spielers.

Der Abstand zum Maximum zerfällt entsprechend in drei Teile:

| Anteil | Bedeutung | holbar? |
|---|---|---|
| 307 − Orakel | diese Ziehung gibt nicht mehr her | nie |
| Orakel − V\* | Preis des Online-Spielens: legen, bevor man die nächsten Kacheln kennt | von keiner Online-Policy |
| V\* − Agent | echte Spielfehler | ja |

`V*` ist die optimale Online-Policy und nicht ausrechenbar, deshalb sind nur
`307 − Orakel` und die Summe der beiden unteren Zeilen messbar.
**`Orakel − Agent` ist damit eine obere Schranke dafür, wieviel Suche und
weiteres Training überhaupt noch bringen können** – nicht der tatsächlich
holbare Betrag.

Gelöst wird exakt per ganzzahligem Programm (CBC über `pulp`), nicht per
Heuristik: Zuordnungsvariablen Feld × Kachel plus eine Binärvariable je
(Linie, Wert), zwei Verschärfungen (Vorfilter auf Linien, für die überhaupt
genug passende Kacheln gezogen wurden, und ein Kardinalitätsschnitt je
Richtung und Wert) und eine Startlösung aus Hill-Climbing. Jede Lösung wird
gegengeprüft – das Board muss exakt aus den gezogenen Kacheln bestehen und
`game.score_board()` muss unabhängig denselben Wert liefern wie die
ILP-Zielfunktion.

**Das Orakel ist eine obere Schranke, kein erreichbares Ziel.** Es kennt alle
19 Kacheln von Anfang an, jede Online-Policy sieht immer nur die aktuelle.

| Parameter | Typ | Default | Bedeutung |
|---|---|---|---|
| `--episodes` | int | `200` | Anzahl Episoden. Jede kostet 5–15 s CPU, deshalb Stichprobe statt aller 2000 Eval-Episoden |
| `--seed` | int | `0` | Master-Seed; ausgewertet werden `seed+1 .. seed+episodes` – gleiche Konvention wie in den Trainingsskripten, damit die Seeds überlappen |
| `--jobs` | int | alle Kerne | Parallele Prozesse |
| `--time-limit` | float | `600` | Sekunden je Episode für CBC. Nicht bewiesen optimale Episoden werden gesondert ausgewiesen |
| `--compare` | str | – | Pfad zu einer `episodes.csv`; rechnet die Zerlegung gepaart über die gemeinsamen Seeds |

Zur Einordnung: der verwandte Modus „alle 27 Kacheln offen, freie Wahl" ist
kein Lernproblem, sondern exakt gelöst – Optimum **307**, erreicht von genau
16 Boards (8 davon verschieden bis auf die 180°-Rotation), per Brute Force
über die 3^15 Linien-Wertzuweisungen in wenigen Sekunden findbar.

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

## Gegen das Netz spielen

```bash
python play.py --model experiments/<run_id>/models/final_model.pt
python play.py --model experiments/<run_id>/models/final_model.zip --algo ppo
python play.py --model ... --seed 42 --html
```

Format wie beim echten Mehrspieler-Take-It-Easy: beide Spieler bekommen
**dieselbe Kachelfolge** (gleicher Seed), jeder legt auf sein eigenes Brett,
am Ende werden die Scores verglichen. Kachelglück fällt damit komplett raus –
verglichen wird nur die Platzierung. Weil das Netz auf seinem eigenen Brett
spielt und sein Zug nicht von deinem abhängt, wird seine Partie vorab
berechnet.

| Parameter | Typ | Default | Bedeutung |
|---|---|---|---|
| `--model` | str | – | `.pt` (Afterstate) oder `.zip` (PPO/DQN). `models/` ist gitignored, muss also lokal noch vorhanden sein |
| `--algo` | `dqn` \| `ppo` | – | Nur für `.zip`-Modelle nötig |
| `--seed` | int | zufällig | Feste, wiederholbare Kachelfolge – praktisch, um dieselbe Partie gegen verschiedene Modelle zu spielen |
| `--html` | flag | aus | Beide Endbretter zusätzlich als HTML in `replay/` |

Eingaben während des Spiels: Feldnummer `0`–`18`, `h` für einen Hinweis
(welches Feld würde das Netz auf *deinem* Brett nehmen?), `d` für die
restlichen Kacheln im Stapel, `q` zum Beenden. Freie Felder zeigen im
Textbrett ihre Nummer an.

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
