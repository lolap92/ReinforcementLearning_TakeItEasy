# Diagnose: dieser Lauf ist divergiert

`summary.json` zeigt Score 0,0 (std 0,0!) über alle 1000 Auswertungs-
Episoden, Ø 199 von 200 möglichen Zügen ungültig - trotz 1.000.000
Trainings-Schritten. Kein Zufall: Diagnoseläufe (gleicher Code, kleineres
Budget) zeigten, dass die Episodenlänge im Training zunächst sinkt, dann
aber kontinuierlich wieder steigt (von ~36 auf über 240 Schritte im
Schnitt, Spitzen bis 848) - klassische Divergenz, kein einfaches
"noch nicht auskonvergiert".

**Ursache:** Dieser Lauf nutzte noch kein Action Masking (siehe
`config.json`: `"algorithm": "DQN (stable-baselines3, ohne Action
Masking)"`). Eine ungültige Aktion in `env.py` führt zu genau demselben
Folgezustand (Board ändert sich nicht) - der Bellman-Update dafür ist
dadurch selbstreferentiell (`Q(s,a)` bootstrapt von `Q(s,*)` desselben
Zustands `s`), was sich mit wachsender Episodenlänge aufschaukelt,
verschärft durch `gamma=1.0` (keine Abzinsung) und einen zunehmend mit
ungültigen Übergängen zugemüllten Replay-Buffer.

Ein Vergleichslauf mit Action Masking (gleiches Budget, nur 60k statt
1M Schritte) zeigte stattdessen sofortige, stabile Konvergenz auf
~19 Schritte/Episode und positiven Score.

**Fix:** `train_dqn.py` maskiert jetzt Q-Werte ungültiger Felder vor dem
argmax (siehe `MaskedDQN`/`MaskedDQNPolicy`/`MaskedQNetwork` im Skript).
Dieser Lauf bleibt als Beleg für den Unterschied in der Historie stehen,
statt gelöscht zu werden - siehe `experiments/README.md`.
