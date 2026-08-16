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

3. **⏳ Baselines** (nächster Schritt – noch nicht begonnen)
   Geplant: Random Agent + einfache Greedy-Heuristik als Referenzwerte, über viele Episoden gemittelt, um später RL-Ergebnisse einordnen zu können

4. **Tabular Q-Learning auf Mini-Board** (offen)
   Kleines, künstlich verkleinertes Board, um die Q-Learning-Update-Regel von Hand nachzuvollziehen, bevor Deep RL eingesetzt wird

5. **DQN via Stable-Baselines3** (offen)
   Erstes "echtes" Deep-RL-Training auf der vollen Environment, inkl. Action Masking

6. **PPO / MaskablePPO via SB3** (offen)
   Vergleich Value-based (DQN) vs. Policy-based (PPO), Lernkurven-Vergleich via TensorBoard

7. **Evaluation** (offen)
   Viele Testepisoden je Agent, Score-Verteilungen vs. Baselines aus Phase 3 vergleichen

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

**Phase 3: Baselines implementieren**
- Random Agent (zufällige gültige Aktion) – Kernlogik existiert bereits als Testcode in `env.py`, sollte aber als eigenständiger, wiederverwendbarer Agent ausgelagert werden
- Einfache Greedy-Heuristik (z.B. Kachel auf das Feld legen, das den größten sofortigen Score-Zuwachs oder die beste Line-Fortsetzung verspricht)
- Beide Agenten über viele Episoden (z.B. 1000+) laufen lassen, Score-Verteilung (Mittelwert, Std, Min/Max) als Referenz für spätere RL-Agenten festhalten
