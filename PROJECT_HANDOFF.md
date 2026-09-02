# Take It Easy – RL Lernprojekt

## Ziel des Projekts

Reinforcement Learning anhand des Brettspiels **Take It Easy** lernen und ausprobieren.
Fokus liegt auf dem **Verständnis der RL-Grundlagen** (nicht primär auf State-of-the-Art-Performance).

- **Erfahrungsstand des Nutzers:** RL-Anfänger
- **Hauptziel:** RL-Grundlagen anhand eines konkreten, überschaubaren Beispiels lernen
- **Tech-Stack:** Python, `gymnasium` (Environment-API), `stable-baselines3` + `sb3-contrib` (Algorithmen, insb. `MaskablePPO`)
- **Vorgehen:** Schrittweiser Lernpfad in 7 Phasen (siehe unten), Konzepte werden vor der Umsetzung erklärt

## Warum Take It Easy als RL-Testfeld?

- Kleiner, klar begrenzter Zustandsraum (19 Felder)
- Kurze, fixe Episodenlänge (genau 19 Züge)
- Stochastische Kachelzüge (Zufallselement wie bei vielen realen Problemen)
- **Sparse Reward**: Score kommt erst am Ende der Episode → gutes Beispiel für das Credit-Assignment-Problem
- Kein Gegner → Fokus liegt rein auf der RL-Mechanik, nicht auf Multi-Agent-Komplexität

## Spielregeln (Kurzfassung)

- Board: 19 Felder, hexagonal in 5 Reihen (3-4-5-4-3) angeordnet
- Deck: 27 Kacheln, jede mit 3 Werten für 3 Richtungen (senkrecht: {1,5,9}, diagonal links "/": {3,4,8}, diagonal rechts "\\": {2,6,7}) – alle 27 Kombinationen kommen genau einmal vor
- Ablauf: Kachel wird zufällig gezogen, muss sofort auf ein freies Feld gelegt werden (19 Züge insgesamt)
- Scoring: Am Ende zählt jede vollständige Linie (alle Kacheln darin haben gleichen Wert in dieser Richtung) mit Score = Linienlänge × Wert. Unvollständige/inkonsistente Linien = 0 Punkte
- Theoretisches Maximum: 307 Punkte (praktisch unerreichbar); gute menschliche Spieler: ca. 150-170 Punkte

## Lernpfad – Phasenübersicht

1. **✅ Theorie-Grundlagen** (erledigt)
   Besprochen: Agent/Environment, State, Action, Reward, Policy, Episode, Value Function/Q-Function, Exploration vs. Exploitation, Value-based vs. Policy-based RL (DQN vs. PPO als spätere Vertreter)

2. **✅ Environment bauen** (erledigt)
   Siehe Abschnitt "Bisheriger Stand" unten für Details

3. **✅ Baselines** (erledigt)
   Random Agent (Ø 10.86) und Greedy-Heuristik (Ø 27.72), je 1000 Episoden, siehe `experiments/2026-08-07_2151_phase3_baselines/`

4. **✅ Tabular Q-Learning auf Mini-Board** (erledigt)
   Y-förmiges Mini-Board, Q-Learning (Ø 8.81) knapp über Random (Ø 6.56), nahe am errechneten Optimum (8.89) – siehe `experiments/2026-08-1[45]_*_phase4_qlearning_mini/`

5. **✅ DQN via Stable-Baselines3** (erledigt, mit wichtigem Instabilitäts-Befund)
   - Erster Lauf ohne Action Masking divergierte vollständig (Score 0 über 1000 Episoden trotz 1 Mio. Steps) – Ursache: ungültige Züge führen zu einem selbstreferentiellen Bellman-Update, verschärft durch `gamma=1.0`. Siehe `experiments/2026-08-16_0738_dqn_1m_unmasked_diverged/NOTES.md`.
   - Mit Action Masking (eigene `MaskedDQN`/`MaskedQNetwork`, da SB3-DQN kein natives Masking hat) lief ein 300k-Steps-Lauf stabil und gut: Ø 47.94, deutlich über Greedy (`experiments/2026-08-16_1227_dqn_300k_masked_beforefix/`).
   - Ein 1-Mio-Steps-Lauf mit identischer Config fiel danach überraschend auf Ø 9.89 zurück (`experiments/2026-08-16_2030_dqn_1m_masked_beforefix_collapsed/`) – **kein Masking-Problem** (0 ungültige Züge), sondern vermutlich klassische DQN-Instabilität durch späte Q-Value-Überschätzung (kein Double-DQN, `gamma=1.0`, `buffer_size=100k` verdrängt bei 1M Steps alte gute Erfahrung).
   - Zusätzlicher gefundener Bug: `_sample_action()` (SB3-intern) samplete vor `learning_starts` komplett ungemaskt – gefixt in `train_dqn.py` (siehe Commit `c758a6a`).
   - **Wiederholter 300k-Lauf nach dem Fix: Ø 31.89 (Endmodell) / 36.21 (bestes Checkpoint)** (`experiments/2026-08-16_2108_dqn_300k_masked_afterfix/`) – schlechter als der ursprüngliche 300k-Lauf (47.94)! Der Fix behebt zwar den konkreten Bug, aber die Kernaussage bleibt: **DQN-Ergebnisse auf diesem Environment streuen extrem von Lauf zu Lauf** (9.89 / 31.89 / 47.94 bei ansonsten identischen Hyperparametern) – ein einzelner Lauf ist hier kein verlässlicher Leistungsindikator, egal ob Endmodell oder Best-Checkpoint.
   - `train_dqn.py` wertet seitdem zusätzlich automatisch das von `EvalCallback` gespeicherte beste Zwischen-Checkpoint aus (`*_best_checkpoint` in `summary.json`/`EXPERIMENTS.md`), loggt den Spiel-Score explizit als eigenes TensorBoard-Tag (`rollout/score_mean`) und unterstützt `--device cuda|cpu|auto` zur expliziten GPU-Wahl (bisher liefen aber alle Läufe laut `config.json` auf `cpu`).

6. **✅ PPO / MaskablePPO via SB3** (erledigt – klarer Gewinner ggü. DQN)
   `train_ppo.py` (analog zu `train_dqn.py`, aber natives Masking über `sb3-contrib`s `ActionMasker`/`MaskablePPO`, kein eigener Policy-Hack nötig) wurde mit 300k, 1 Mio. und 25 Mio. Steps trainiert:

   | Steps | PPO Ø (Endmodell) | PPO Ø (Best-Checkpoint) | zum Vergleich DQN Ø |
   |---|---|---|---|
   | 300k | 52.09 | 53.08 | 47.94 / 31.89 (zwei Läufe, siehe oben) |
   | 1 Mio. | 92.81 | 90.39 | 9.89 (kollabiert) |
   | 25 Mio. | 108.86 | 108.27 | – |

   Zwei zentrale Beobachtungen: (1) PPO verbessert sich **monoton** mit mehr Trainingszeit statt zu kollabieren wie DQN – Endmodell und Best-Checkpoint liegen bei PPO in allen drei Läufen praktisch gleichauf, es gibt keinen "Rettungsanker" durch Best-Checkpoint-Auswertung nötig, weil nichts einbricht. (2) Deutlich abnehmender Grenznutzen: 300k→1M (+3.3× Steps) bringt +40 Punkte, 1M→25M (+25× Steps) nur noch +16 Punkte – GPU/mehr Compute ist hier vermutlich nicht der wirksamste Hebel mehr, siehe Nächster Schritt.

7. **✅ Evaluation / Standortbestimmung** (erledigt – mit unbequemem Befund)
   Report: `reports/phase7_analysis_report.html`. Kernbefund: **die Greedy-Baseline aus Phase 3 war kaputt** und hat alle bisherigen Vergleiche verzerrt. `greedy_agent` bewertet Züge mit `score_board()`, das nur *fertige* Linien zählt – während der ersten ~15 Züge ist keine Linie fertig, alle Felder sehen gleich gut aus, es wird faktisch zufällig gelegt (daher Ø 27,72).

   Dieselbe Heuristik mit `potential_score()` (existiert seit Wave 2 im Repo, wurde nur fürs Reward Shaping genutzt) kommt **ohne jedes Training auf Ø 120,88**. Eine 1-Ply-Erwartungswert-Heuristik, die zusätzlich das Restdeck auswertet, kommt auf **Ø 128,93**. Beide sind jetzt als Agenten in `baselines.py` (`greedy_potential`, `expected_value`), Lauf: `experiments/2026-08-30_0705_phase7_heuristic_baselines/` (2000 Episoden je Agent).

   | Agent | Ø | Std | ≥150 Punkte |
   |---|---|---|---|
   | expected_value (Heuristik, kein Training) | **128,93** | 23,40 | 19,9 % |
   | greedy_potential (Heuristik, kein Training) | **120,88** | 35,76 | 22,1 % |
   | ppo_25m_singleenv (25 Mio. Steps) | 108,86 | 25,19 | 3,2 % |
   | greedy (score_board, alte Baseline) | 27,60 | 22,72 | 0 % |

   Damit ist die Antwort auf „ist der Agent schlechter als ein Mensch?" eindeutig **ja** – er liegt sogar unter zwei untrainierten Heuristiken. Zur Einordnung: die Spielanleitung nennt 150 als „gutes", 200 als „sehr gutes" Ergebnis; der bekannteste öffentliche Take-It-Easy-Agent (NN + MCTS) liegt bei Ø ≈ 167, was dort als praktische Obergrenze bei zufälligem Kachelzug argumentiert wird.

   Zwei weitere Befunde: (1) **Compute ist nicht der Hebel.** Über den letzten Abschnitt (1 Mio. → 25 Mio.) bringt das Training nur noch ≈11,5 Punkte pro Verzehnfachung der Steps – für 150 Punkte wären ≈10^11 Steps nötig. (2) **Hyperparameter sind es auch nicht.** Im 1-Mio.-Sweep lagen vier von fünf Konfigurationen zwischen Ø 87,7 und 92,6, bei einer Seed-zu-Seed-Streuung von ±4 bis ±9 – kein signifikanter Unterschied (nur `constant_lr` mit Ø 67,5 ist klar schlechter).

8. **✅ Afterstate-Wertfunktion** (erledigt – der Durchbruch)
   `train_afterstate.py` (PyTorch direkt, kein SB3 – die Konstruktion passt nicht in die SB3-API, weil sie die bekannte Dynamik des Spiels ausnutzt statt sie zu lernen). Statt einer Policy wird `V(Afterstate)` gelernt, also der erwartete Endscore des Boards *nachdem* die Kachel gelegt wurde; zur Spielzeit wird `argmax` über die ≤19 möglichen Folgezustände gebildet – das ist bereits eine 1-Ply-Suche. Gleiche Konstruktion wie TD-Gammon und die starken 2048-Agenten.

   **Ergebnis nach 300.000 Selbstspiel-Episoden (≈5,7 Mio. Environment-Steps, 20 Minuten auf 4 CPU-Kernen): Ø 160,38** über 2000 Episoden auf der echten `TakeItEasyEnv` mit denselben Seeds wie alle anderen Läufe (`experiments/2026-08-30_0726_afterstate_300k/`).

   | Agent | Ø | Std | Median | ≥150 | ≥200 | Steps |
   |---|---|---|---|---|---|---|
   | **afterstate_300k** | **160,38** | 27,28 | 161 | **67,4 %** | 7,8 % | 5,7 Mio. |
   | expected_value (Heuristik) | 128,93 | 23,40 | 128 | 20,0 % | 0,3 % | – |
   | greedy_potential (Heuristik) | 120,88 | 35,76 | 121 | 22,1 % | 1,0 % | – |
   | ppo_25m_singleenv | 108,86 | 25,19 | 113 | 3,2 % | 0,0 % | 25 Mio. |

   **+51,5 Punkte gegenüber PPO bei rund einem Viertausendstel der Environment-Steps.** Der Agent liegt damit erstmals über der 150er-Schwelle, die die Spielanleitung „gutes Ergebnis" nennt, und erreicht sie in 67 % statt in 3 % der Partien – also nicht mehr schlechter als ein durchschnittlicher Mensch, sondern besser. Damit ist auch die Phase-7-These bestätigt: die 25 Mio. PPO-Steps waren nie ein Compute-Problem.

   Aufschlussreich ist der Verlauf: schon nach **6.400 Episoden** (2 % des Budgets, 30 Sekunden) lag die Wertfunktion bei Ø 128,7 – auf dem Niveau der besten untrainierten Heuristik. Nach 25.600 Episoden waren es 151,7. Die Lernkurve stieg bis zum letzten Update (Ø 159,98 in der Schlussiteration), das Episodenbudget war also nicht der begrenzende Faktor.

   Was das Netz sieht (348 statt 60 Dimensionen): 19 × 10 One-Hot fürs Board, **27 Dimensionen Restdeck-Multi-Hot** (fehlt in `env.py` komplett), 9 Zähler je (Richtung, Wert), 15 × 8 Linien-Features inkl. Komplettierungswahrscheinlichkeit, 2 Fortschrittswerte.

   Trainingskonstruktion: λ-Return als Regressionsziel – weil alle Zwischenrewards 0 sind und γ=1 gilt, reduziert sich der Forward-View auf `target[t] = (1-λ)·max_a V(s'[t+1]) + λ·target[t+1]` mit `target[18] = Endscore`. Die Zielwerte nutzen immer `max_a V` und sind damit off-policy, werden also von der ε-greedy-Exploration nicht verzerrt. Wertkopf ist per Default ein Two-Hot-Klassifikationskopf über den Score-Bereich 0–307 statt einer MSE-Regression (der Endscore ist eine Summe weniger großer Sprünge – da lernt Klassifikation verlässlicher).

   Verifiziert gegen `game.py`: die vektorisierte Scoring-Funktion stimmt exakt mit `score_board()` überein, das Linien-Feature „potential" summiert sich exakt zu `potential_score()`, und ein „Netz", das nur das vorgerechnete Erwartungswert-Feature aufsummiert, reproduziert die `expected_value`-Heuristik (Ø 127,1) – in der schnellen vektorisierten Simulation wie über die echte Env.

9. **✅ Hindsight-Orakel** (erledigt – Standortbestimmung für Phase 9)
   `oracle.py` rechnet je Episode exakt aus, was mit **genau den 19 gezogenen Kacheln** bei freier Platzierung maximal möglich gewesen wäre – als ganzzahliges Programm (CBC über `pulp`), nicht per Heuristik. Lauf über 200 Episoden (`experiments/2026-09-02_1225_phase9_hindsight_oracle/`), **alle 200 bewiesen optimal**:

   | Größe | Ø | Std | p10 | Median | p90 |
   |---|---|---|---|---|---|
   | Maximum bei freier Kachelwahl | 307 | – | – | – | – |
   | **Hindsight-Orakel** | **248,75** | 14,63 | 230 | 248 | 267 |
   | afterstate_300k (gleiche Seeds) | 158,22 | 28,49 | – | – | – |

   58,3 Punkte gehen allein durch die Ziehung verloren (307 − 248,75), 90,5 liegen zwischen Agent und Orakel – der Agent erreicht **63,6 %** des in seinen eigenen Partien Möglichen.

   **Lesart, wichtig:** die 90,5 sind eine *obere Schranke*, nicht der holbare Betrag. Der Abstand zum Maximum zerfällt in drei Teile: `307 − Orakel` (nie holbar), `Orakel − V*` (Preis des Online-Spielens, von keiner Policy holbar) und `V* − Agent` (echte Spielfehler). `V*` ist nicht ausrechenbar, also sind nur der erste Teil und die Summe der beiden anderen messbar.

   Die Ziehreihenfolge ignoriert das Orakel bewusst, und das ist exakt richtig: weil jede Kachel auf jedes freie Feld darf, ist jede Bijektion Kacheln → Felder in jeder Reihenfolge realisierbar (lege `t_i` auf sein Zielfeld – nie belegt, weil die Zuordnung injektiv ist). Das Orakel ist damit exakt der Wert des hellsehenden Spielers.

   **Der eigentlich interessante Befund:** das Orakel streut mit Std 14,6 auffallend wenig (p10 230, p90 267) – praktisch jede Ziehung ist ähnlich gut. Die Korrelation zwischen Orakel und Agentenscore je Episode liegt bei nur **0,38**. Dass eine Partie gut oder schlecht ausgeht, liegt also kaum am Kachelglück; die Streuung des Agenten (28,5) ist fast doppelt so groß wie die des Orakels und kommt überwiegend aus dem Spiel selbst. Nur in 8 % der Partien holt er ≥80 % des Möglichen. Das spricht für Suche zur Spielzeit statt für mehr Trainingsbudget.

   Zur Einordnung noch: das Hill-Climbing, das dem Solver nur als Startlösung dient, kommt mit *voller* Kachelkenntnis auch nur auf Ø 183,1 – gute Platzierung ist selbst mit Vollinformation schwer.

   Nebenbefund: der verwandte Modus „alle 27 Kacheln offen, freie Wahl" ist kein Lernproblem, sondern exakt gelöst – Optimum **307**, erreicht von genau 16 Boards (8 davon verschieden bis auf die 180°-Rotation), per Brute Force über die 3^15 Linien-Wertzuweisungen in Sekunden findbar. Ein Netz darauf zu trainieren hieße, eines von 16 auswendig zu lernen.

   Das Orakel ist **nicht Teil des Trainings** und soll es nicht werden: 5–15 s ILP je Episode gegen ~500 Selbstspiel-Episoden/s sind vier bis fünf Größenordnungen Unterschied, und inhaltlich wäre es falsch – `V(Afterstate)` muss der Erwartungswert über künftige Ziehungen sein, das Orakel ist ein Maximum mit Hindsight. Darauf zu trainieren hieße systematischer Hindsight-Bias.

## Bisheriger Stand (Details zu Phase 2)

### Dateien

```
take_it_easy/
├── game.py            # Kernspiellogik (Board-Geometrie, Deck, Scoring) – unabhängig von RL
├── env.py             # Gymnasium-Environment (TakeItEasyEnv), nutzt game.py
└── requirements.txt   # gymnasium, stable-baselines3, sb3-contrib, numpy, tensorboard
```

### `game.py`

- `build_deck()` – erzeugt alle 27 Kacheln als Tupel `(vertikal, links_diag, rechts_diag)`
- `ROWS`, `LINES_VERTICAL`, `LINES_LEFT_DIAG`, `LINES_RIGHT_DIAG` – Board-Geometrie als Listen von Feld-Indizes je Richtung
- `score_board(board)` – berechnet Gesamtscore + Detail-Breakdown pro Linie
- Enthält Selbsttests (`if __name__ == "__main__"`), **alle bestanden**:
  - Deck hat 27 einzigartige Kacheln
  - Jede Richtung deckt exakt alle 19 Felder ab (Geometrie korrekt)
  - Scoring stimmt mit manueller Nachrechnung überein
  - Extremfall (alle Felder gleiche Kachel) ergibt erwarteten Maximalscore je Richtung

### `env.py` – `TakeItEasyEnv(gym.Env)`

Design-Entscheidungen:

- **Observation Space:** `Box(low=0, high=9, shape=(60,), dtype=float32)` – flacher Vektor: 19 Felder × 3 Werte (0 = leer) + 3 Werte der aktuellen Kachel
  - *Bekannte Design-Schwäche, bewusst in Kauf genommen für den Start:* `Box` suggeriert eine Ordnung zwischen Kachelwerten, die eigentlich nur kategorial sind. Mögliche spätere Verbesserung: `MultiDiscrete` oder `Dict`-Space mit One-Hot-Encoding – aktuell zurückgestellt, um zuerst ein funktionierendes Baseline-Modell zu haben
- **Action Space:** `Discrete(19)` – Index des Feldes für die aktuelle Kachel
- **Reward:** 0 nach jedem Zug, kompletter Score erst im letzten Schritt (Sparse Reward, bewusst so gelassen für Credit-Assignment-Lerneffekt; Reward Shaping ist eine spätere Option, aber noch nicht entschieden)
- **Action Masking:** vorbereitet über `info["action_mask"]` (Bool-Array, `True` = Feld frei) – wird ab Phase 6 von `MaskablePPO` (`sb3-contrib`) genutzt. Ohne Masking wird ein ungültiger Zug (belegtes Feld) aktuell mit Reward `-10` bestraft, Episode läuft mit gleicher Kachel weiter
- Enthält `render()` (Textausgabe des Boards) und einen manuellen Testlauf mit Random Agent

### Testing-Hinweis

Die Sandbox, in der dieses Projekt begonnen wurde, hatte **keinen Internetzugriff**, daher konnte `gymnasium` dort nicht installiert werden. Die Kernlogik von `env.py` (reset/step-Ablauf, Action Masking, Terminierung nach 19 Zügen) wurde stattdessen über eine `numpy`-only Simulation validiert – alle Tests bestanden (u.a. Random-Agent-Score über eine Beispiel-Episode: 12 Punkte, korrekte Erkennung ungültiger Züge). **`env.py` selbst wurde noch nicht mit echtem `gymnasium` ausgeführt** – das sollte der erste Schritt in der neuen Umgebung sein:

```bash
pip install -r requirements.txt
python game.py   # sollte "Alle Selbsttests erfolgreich!" ausgeben
python env.py    # sollte Board-Render + Random-Agent-Score ausgeben
```

## Besprochene Konzepte (für Kontext, falls relevant)

- Gymnasium-API: `reset()` → `(observation, info)`, `step(action)` → `(observation, reward, terminated, truncated, info)`
- `observation_space` / `action_space` als formale Deklaration für SB3, um automatisch passende Policy-Netzwerke zu bauen
- Value-based (DQN: lernt Q(S,A), wählt Maximum) vs. Policy-based (PPO: lernt direkt π(A|S) als Verteilung)

## Nächster Schritt

**Phase 9: Expectimax zur Spielzeit.** Das Orakel (Phase 9-Vorarbeit, oben)
hat die Frage beantwortet, ob sich Suche noch lohnt: der Agent holt 63,6 % des
in seinen Partien Möglichen, und seine Streuung kommt überwiegend aus dem
Spiel selbst, nicht aus dem Kachelglück. Konkret: weil das Restdeck bekannt
ist, lässt sich über die nächste Kachel exakt erwartungswerten statt zu
sampeln. Bei ≤19 Zügen × ≤27 Kacheln ist 2-Ply-Expectimax problemlos
rechenbar, in den letzten ~6 Zügen sogar vollständige Suche bis zum Ende.
Braucht keine Änderung an der Wertfunktion – nur eine andere
Kandidatenauswertung zur Spielzeit.

Daneben, in absteigender Priorität:

- **Längerer Lauf.** Die Lernkurve stieg bis zum letzten Update.
  `--episodes 3000000` wäre etwa 3,5 Stunden CPU.
- **Ablationen.** `--no-line-features`, `--value-head scalar` und `--lam 1.0`
  sind vorbereitet und würden zeigen, wieviel von den 160,4 auf die
  Feature-Konstruktion und wieviel auf das eigentliche Lernen entfällt. Für
  ein Lernprojekt die interessanteste offene Frage.
- **Mehrere Seeds.** Bisher genau ein Afterstate-Lauf.
- **k-Lookahead als Curriculum.** „Wähle eine der nächsten k aufgedeckten
  Kacheln": k=1 ist das reguläre Spiel, k=27 der vollständig offene Modus.
  Braucht nur eine andere Kandidatenerzeugung in `play_batch` plus `k/27` als
  Eingabefeature (sonst widersprechen sich die Zielwerte: bei k=1 ist V ein
  Erwartungswert, bei k=27 ein Maximum) und einen zweiten 27-dim Block
  „aufgedeckt" im Encoder.
- **`replay.py` erweitern.** Lädt aktuell nur SB3-Modelle (`.zip`); das
  Afterstate-Modell liegt als `.pt` vor. Die Orakel-Boards
  (`oracle_boards.json`) wären daneben eine gute Vergleichsansicht: dasselbe
  Kachelset, einmal wie der Agent es gelegt hat und einmal optimal.
