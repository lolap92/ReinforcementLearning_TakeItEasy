"""
Visuelle SVG-Darstellung von Take-It-Easy-Kacheln und -Brettern.

Jede Kachel ist ein Hexagon mit 3 farbigen Balken (einer je Richtung:
senkrecht, "/", "\\"), an deren Enden der jeweilige Wert steht - angelehnt
an die Optik des echten Spiels, aber eigenständig gestaltet (kein Nachbau
des Original-Artworks/Copyrights, nur des Spielmechanik-Layouts).

Nicht offensichtlich: welcher Werte-Index (vertical/left_diag/right_diag aus
game.py) auf welche visuelle Diagonale ("/" oder "\\") gehört, ist keine freie
Wahl - es muss zur Geometrie von env.py's render() und zu game.py's
LINES_LEFT_DIAG/LINES_RIGHT_DIAG passen, sonst verlaufen zusammengehörige
Linien im Bild nicht gerade. Nachgerechnet anhand der Zellkoordinaten, die
env.py's render() erzeugt (Spalte = col_idx, Zeile = offset + 2*row_in_col):
LINES_LEFT_DIAG-Gruppen liegen visuell auf einer "\\"-Diagonale (oben-links
nach unten-rechts), LINES_RIGHT_DIAG-Gruppen auf einer "/"-Diagonale
(unten-links nach oben-rechts) - umgekehrt zu den Kommentaren/Symbolen in
game.py (die vermutlich nur die Blickrichtung vertauscht benennen). Für die
Bild-Geometrie zählt hier nur, was tatsächlich zusammenhängt.
"""

from pathlib import Path

from game import ROWS, VERTICAL_VALUES, LEFT_DIAG_VALUES, RIGHT_DIAG_VALUES, build_deck

REPO_ROOT = Path(__file__).resolve().parent
TILE_ASSETS_DIR = REPO_ROOT / "assets" / "tiles"

# Eine feste, gut unterscheidbare Farbe je Wert 1-9 (Werte sind pro Richtung
# disjunkt, ein Wert taucht also nie in zwei Richtungen auf - eine Palette
# über alle 9 Werte reicht).
VALUE_COLORS = {
    1: "#4C72B0",
    2: "#DD8452",
    3: "#55A868",
    4: "#C44E52",
    5: "#8172B2",
    6: "#937860",
    7: "#DA8BC3",
    8: "#8C8C8C",
    9: "#CCB974",
}

# Hexagon-Geometrie (flat-top, wie schon in env.py's Text-Render): R = Mitte
# bis Ecke. TILE_W/TILE_H = Bounding-Box einer Kachel, COL_STEP/LINE_STEP =
# Versatz zwischen benachbarten Spalten/Halb-Zeilen beim Zusammensetzen eines
# ganzen Bretts (siehe board_to_svg).
R = 50
TILE_W = 2 * R
TILE_H = 1.7320508 * R  # sqrt(3) * R
COL_STEP = 0.75 * TILE_W
LINE_STEP = TILE_H / 2


def _hex_points(cx, cy, r=R):
    return [
        (cx + 0.5 * r, cy - 0.8660254 * r),
        (cx - 0.5 * r, cy - 0.8660254 * r),
        (cx - r, cy),
        (cx - 0.5 * r, cy + 0.8660254 * r),
        (cx + 0.5 * r, cy + 0.8660254 * r),
        (cx + r, cy),
    ]


def _badge(x, y, value):
    color = VALUE_COLORS[value]
    return (
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="11" fill="{color}" stroke="white" stroke-width="1.5"/>'
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="13" font-weight="700" font-family="Arial, sans-serif" '
        f'fill="white" text-anchor="middle" dominant-baseline="central">{value}</text>'
    )


def tile_group(v, l, r_val, cx, cy, empty=False):
    """SVG-<g> für eine Kachel (oder ein leeres Hexagon-Umriss, falls empty),
    zentriert auf (cx, cy)."""
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in _hex_points(cx, cy))
    if empty:
        return (
            f'<g><polygon points="{pts}" fill="#f4f4f4" stroke="#c9c9c9" '
            f'stroke-width="1.5" stroke-dasharray="4,3"/></g>'
        )

    top = (cx, cy - 0.8660254 * R)
    bottom = (cx, cy + 0.8660254 * R)
    upper_left = (cx - 0.75 * R, cy - 0.433 * R)
    lower_right = (cx + 0.75 * R, cy + 0.433 * R)
    lower_left = (cx - 0.75 * R, cy + 0.433 * R)
    upper_right = (cx + 0.75 * R, cy - 0.433 * R)

    parts = [f'<polygon points="{pts}" fill="white" stroke="#333" stroke-width="2"/>']
    # Reihenfolge der Balken: senkrecht zuletzt gezeichnet (liegt oben), damit
    # sich die drei Linien in der Mitte sauber kreuzen statt sich gegenseitig
    # zu verdecken.
    for (x1, y1), (x2, y2), value in (
        (upper_left, lower_right, l),
        (lower_left, upper_right, r_val),
        (top, bottom, v),
    ):
        color = VALUE_COLORS[value]
        parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="7" stroke-linecap="round"/>'
        )
    for (x1, y1), (x2, y2), value in (
        (upper_left, lower_right, l),
        (lower_left, upper_right, r_val),
        (top, bottom, v),
    ):
        parts.append(_badge(x1, y1, value))
        parts.append(_badge(x2, y2, value))
    return f"<g>{''.join(parts)}</g>"


def tile_svg(v, l, r_val):
    """Eigenständige SVG-Datei (ein Tile, zentriert im eigenen viewBox)."""
    margin = 6
    w, h = TILE_W + 2 * margin, TILE_H + 2 * margin
    cx, cy = w / 2, h / 2
    group = tile_group(v, l, r_val, cx, cy)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.1f} {h:.1f}" '
        f'width="{w:.0f}" height="{h:.0f}">{group}</svg>'
    )


def save_all_tile_svgs(out_dir=TILE_ASSETS_DIR):
    """Erzeugt alle 27 Kachel-SVGs (aus game.py build_deck()) unter out_dir/,
    Dateiname {vertikal}_{links}_{rechts}.svg."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for v, l, r_val in build_deck():
        path = out_dir / f"{v}_{l}_{r_val}.svg"
        path.write_text(tile_svg(v, l, r_val))
        paths.append(path)
    return paths


def _cell_position(cell_idx):
    """col_idx/line-Position einer Feld-Nummer (0-18), analog zu env.py's
    render(): ROWS sind hier die 5 Spalten des Bretts."""
    max_height = max(len(col) for col in ROWS)
    for col_idx, col_indices in enumerate(ROWS):
        if cell_idx in col_indices:
            row_in_col = col_indices.index(cell_idx)
            offset = max_height - len(col_indices)
            return col_idx, offset + 2 * row_in_col
    raise ValueError(f"Unbekannter Feld-Index: {cell_idx}")


def board_to_svg(board):
    """Ganzes Brett (Liste[19] von (v,l,r)-Tupeln oder None) als eine SVG-
    Grafik, im gleichen Spalten-Layout wie env.py's Text-Render."""
    margin = 30
    max_line = 2 * (max(len(col) for col in ROWS) - 1)
    width = margin * 2 + (len(ROWS) - 1) * COL_STEP + TILE_W
    height = margin * 2 + max_line * LINE_STEP + TILE_H

    groups = []
    for cell_idx in range(19):
        col_idx, line = _cell_position(cell_idx)
        cx = margin + TILE_W / 2 + col_idx * COL_STEP
        cy = margin + TILE_H / 2 + line * LINE_STEP
        tile = board[cell_idx]
        if tile is None:
            groups.append(tile_group(None, None, None, cx, cy, empty=True))
        else:
            v, l, r_val = tile
            groups.append(tile_group(v, l, r_val, cx, cy))

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.1f} {height:.1f}" '
        f'width="{width:.0f}" height="{height:.0f}">'
        f'<rect x="0" y="0" width="{width:.1f}" height="{height:.1f}" fill="#1b3a2f"/>'
        f"{''.join(groups)}</svg>"
    )


def board_to_html(board, score=None, title="Take It Easy - Board"):
    svg = board_to_svg(board)
    score_line = f"<p>Score: <strong>{score:.0f}</strong></p>" if score is not None else ""
    return f"""<!doctype html>
<html lang="de">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ background:#0e1f19; color:#eee; font-family:Arial,sans-serif; text-align:center; padding:20px; }}
  svg {{ max-width:100%; height:auto; }}
</style>
</head>
<body>
<h1>{title}</h1>
{score_line}
{svg}
</body>
</html>"""


if __name__ == "__main__":
    paths = save_all_tile_svgs()
    print(f"{len(paths)} Kachel-SVGs geschrieben nach {TILE_ASSETS_DIR.relative_to(REPO_ROOT)}/")
