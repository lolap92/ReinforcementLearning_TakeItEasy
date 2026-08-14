# reports/ – veröffentlichte HTML-Grafiken/Diagramme

Handgebaute HTML-Reports und Diagramme (Charts, Board-Skizzen), die im Laufe
des Projekts als Artifact gezeigt wurden - im Unterschied zu den
automatisch generierten Rohdaten unter `experiments/<run_id>/`.

Namenskonvention: `<phase>_<kurzbeschreibung>.html`, z. B.
`phase3_baselines_report.html`, `phase4_qlearning_report.html`.

Wichtig: jede Datei braucht `<meta charset="utf-8">` als erste Zeile im
`<head>` (bzw. ganz am Anfang der Datei, siehe bestehende Reports) - sonst
werden Umlaute falsch dargestellt, wenn die Datei direkt (nicht über die
Artifact-Vorschau) im Browser geöffnet wird.
