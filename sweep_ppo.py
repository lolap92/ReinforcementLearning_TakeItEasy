"""
Kleines Hyperparameter-Sweep-Skript für MaskablePPO ("Wave 2" der Score-
Optimierung, siehe Chat).

Warum: statt eine einzelne teure Konfiguration zu erraten und dafür gleich
Millionen Steps zu investieren, testet dieses Skript mehrere Konfigurationen
bei kleinem, gemeinsamem Budget gegeneinander - genau das hätte den
n_envs=8-Fehlschlag (siehe EXPERIMENTS.md, 2026-08-19) vorab sichtbar
gemacht, bevor 1 Mio. Steps hineinliefen. Jede Konfiguration läuft über
mehrere Seeds, um Lauf-zu-Lauf-Streuung nicht mit echten Konfigurations-
unterschieden zu verwechseln (siehe DQN: 9,89 / 31,89 / 47,94 Mean-Score bei
identischen Hyperparametern, nur unterschiedlicher Zufall).

Nutzung:
    python sweep_ppo.py
    python sweep_ppo.py --timesteps 200000 --seeds 0 1 2
    python sweep_ppo.py --n-envs 4 --device cuda
    python sweep_ppo.py --configs baseline reward_shaping   # nur diese testen

Ruft train_ppo.py für jede Kombination aus Konfiguration x Seed per
Subprocess auf (kein Code-Duplikat), sammelt danach die summary.json-
Ergebnisse ein und druckt eine Rangliste. Jeder Einzellauf landet ganz normal
unter experiments/<run_id>/ und wird wie gewohnt in EXPERIMENTS.md
protokolliert - dieses Skript fügt nur eine Zusammenfassung obendrauf.

Wichtig: das Budget hier ist bewusst klein (Default 150k Steps) - es geht um
ein *relatives* Ranking der Konfigurationen, nicht um das finale Modell. Die
Gewinner-Konfiguration danach mit deutlich mehr --timesteps final trainieren.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"

# Konfigurationsname -> zusätzliche CLI-Flags für train_ppo.py (über die
# "Wave 1"-Defaults hinaus, die für alle Konfigurationen gleich bleiben).
CONFIGS = {
    "baseline": [],
    "reward_shaping": ["--reward-shaping"],
    "constant_lr": ["--constant-lr"],
    "no_normalize": ["--no-normalize"],
    "n_envs4": ["--n-envs", "4"],  # überschreibt das globale --n-envs unten (argparse: letzter Wert gewinnt)
}


def steps_label(n):
    """Kurzform für Timesteps in Tags/Run-IDs, z.B. 300_000 -> '300k', 1_000_000 -> '1m'."""
    if n % 1_000_000 == 0:
        return f"{n // 1_000_000}m"
    if n % 1_000 == 0:
        return f"{n // 1000}k"
    return str(n)


def run_one(config_name, extra_args, seed, timesteps, eval_episodes, device, n_envs):
    # Tag enthält Steps + Seed, damit jeder Sweep-Einzellauf im Ordnernamen
    # eindeutig und selbsterklärend ist (z.B. sweep_reward_shaping_200k_seed1).
    tag = f"sweep_{config_name}_{steps_label(timesteps)}_seed{seed}"
    before = {p.name for p in EXPERIMENTS_DIR.glob(f"*_{tag}")} if EXPERIMENTS_DIR.exists() else set()

    cmd = [
        sys.executable, "train_ppo.py",
        "--timesteps", str(timesteps),
        "--eval-episodes", str(eval_episodes),
        "--seed", str(seed),
        "--tag", tag,
        "--device", device,
        "--n-envs", str(n_envs),
    ] + extra_args
    print(f"\n=== {config_name} (seed={seed}) ===\n{' '.join(cmd)}\n")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    after = {p.name for p in EXPERIMENTS_DIR.glob(f"*_{tag}")}
    new_dirs = after - before
    if len(new_dirs) != 1:
        raise RuntimeError(
            f"Konnte den neuen Run-Ordner für {config_name}/seed={seed} nicht "
            f"eindeutig bestimmen: {new_dirs}"
        )
    run_dir = EXPERIMENTS_DIR / next(iter(new_dirs))
    summary = json.loads((run_dir / "summary.json").read_text())
    final = next(s for s in summary if s["name"] == "ppo_full_board_masked")
    return run_dir.name, final["mean"], final["std"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter-Sweep für train_ppo.py bei kleinem, gemeinsamem Budget.")
    parser.add_argument("--timesteps", type=int, default=150_000, help="Budget je Einzellauf (klein halten - relatives Ranking, nicht finales Modell)")
    parser.add_argument("--eval-episodes", type=int, default=300)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument(
        "--configs", type=str, nargs="+", default=list(CONFIGS), choices=list(CONFIGS),
        help="Nur diese Konfigurationen testen (Default: alle)",
    )
    args = parser.parse_args()

    results = []
    for config_name in args.configs:
        for seed in args.seeds:
            run_id, mean, std = run_one(
                config_name, CONFIGS[config_name], seed,
                args.timesteps, args.eval_episodes, args.device, args.n_envs,
            )
            results.append({"config": config_name, "seed": seed, "run_id": run_id, "mean": mean, "std": std})

    print("\n" + "=" * 70)
    print(f"Sweep-Ergebnisse ({args.timesteps} Steps je Lauf, Seeds {args.seeds})")
    print("=" * 70)

    by_config = {}
    for r in results:
        by_config.setdefault(r["config"], []).append(r["mean"])

    ranked = sorted(by_config.items(), key=lambda kv: sum(kv[1]) / len(kv[1]), reverse=True)
    for config_name, means in ranked:
        avg = sum(means) / len(means)
        detail = ", ".join(f"{m:.1f}" for m in means)
        print(f"{config_name:16s}  Ø über Seeds: {avg:7.2f}   (Einzelwerte: {detail})")

    print("\nEinzelläufe stehen unter experiments/<run_id>/ und in EXPERIMENTS.md.")
    print("Gewinner-Konfiguration danach mit deutlich mehr --timesteps final trainieren.")
