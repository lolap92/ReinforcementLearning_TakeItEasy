# experiments/ – Format aller Läufe

Jeder Skript-Lauf (Baselines, später Q-Learning/DQN/PPO-Training) legt einen
eigenen Ordner an: `experiments/<run_id>/`, z. B.
`experiments/2026-08-07_1430_phase3_baselines/`.

Der `run_id` setzt sich zusammen aus Zeitstempel + `--tag` (siehe
`baselines.py --help`).

## Warum dieses Format

Git ist gut für Text/kleine Dateien mit sinnvoller History, schlecht für
große binäre oder extrem hochfrequente Daten. Deshalb: **Reproduzierbarkeit
statt Rohdaten-Sammelwut.** Jede Episode ist allein über ihren `seed`
deterministisch reproduzierbar (Kachelreihenfolge *und* alle
Zufallsentscheidungen des Agenten hängen nur an diesem einen Seed) – wir
müssen also nicht jeden Zug jeder Episode aufheben, um sie später exakt
nachvollziehen zu können. Volles Logging lohnt sich nur für eine kleine
Stichprobe zur direkten Inspektion.

## Inhalt eines Run-Ordners

| Datei/Ordner | Inhalt |
|---|---|
| `config.json` | Agenten, Anzahl Episoden, Master-Seed, Git-Commit-Hash, Zeitstempel – alles, was zur Reproduktion des Laufs nötig ist |
| `summary.json` | Aggregierte Kennzahlen je Agent: `mean`, `std`, `min`, `max`, `n_episodes` |
| `episodes.csv` | Eine Zeile je gespielter Episode: `agent, episode_index, seed, score` – Rohdaten für eigene Auswertungen/Plots |
| `sample_trajectories/<agent>.json` | Volle Zug-für-Zug-Trajektorie der ersten `--n-trajectories` Episoden dieses Agenten: pro Zug die gezogene Kachel, die gewählte Aktion (Feld-Index) und der Board-Zustand danach |

## Nicht im Repo (siehe `.gitignore`)

Modell-Checkpoints (`experiments/*/models/`) und TensorBoard-Eventlogs
(`experiments/*/tensorboard/`) – zu groß/binär, lokal reproduzierbar aus
`config.json` (gleicher Seed + gleicher Git-Commit).

## Gesamtüberblick

`../EXPERIMENTS.md` listet eine Zeile pro Run (Datum, Run-ID, Agent,
Kennzahlen) als schnellen Vergleich über alle Phasen hinweg.
