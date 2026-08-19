"""
Spielt eine einzelne Take-It-Easy-Episode mit einem gespeicherten Modell
(DQN oder MaskablePPO) nach und zeigt das fertige Brett.

Nützlich, um eine konkrete Runde aus einer episodes.csv (Spalten: agent,
episode_index, seed, score) visuell nachzuvollziehen - z.B. die beste Runde
eines Laufs. Die Kachelreihenfolge ist über den Seed deterministisch, die
Zug-Entscheidungen kommen aus dem geladenen Modell - für's exakte Nachspielen
also unbedingt den Seed aus der episodes.csv-Zeile verwenden, nicht den
episode_index.

Nutzung (lokal, wo das models/-Verzeichnis des jeweiligen Laufs noch
existiert - models/ ist gitignored, siehe .gitignore):
    python replay.py --model experiments/<run_id>/models/final_model.zip --algo ppo --seed 195
    python replay.py --model experiments/<run_id>/models/best_model.zip --algo dqn --seed 42

Mit --html zusätzlich eine visuelle HTML-Ansicht des fertigen Bretts
schreiben (echte Kachel-Grafiken statt Text, siehe board_render.py):
    python replay.py --model ... --algo ppo --seed 195 --html board.html
"""

import argparse

from env import TakeItEasyEnv


def load_model(path, algo):
    if algo == "dqn":
        from train_dqn import MaskedDQN
        return MaskedDQN.load(path)
    from sb3_contrib import MaskablePPO
    return MaskablePPO.load(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eine Take-It-Easy-Episode mit einem gespeicherten Modell nachspielen und das Brett anzeigen.")
    parser.add_argument("--model", required=True, help="Pfad zu einer .zip-Modelldatei (final_model.zip oder best_model.zip)")
    parser.add_argument("--algo", choices=["dqn", "ppo"], required=True)
    parser.add_argument("--seed", type=int, required=True, help="Seed aus der episodes.csv-Zeile der gewünschten Runde")
    parser.add_argument("--html", type=str, default=None, help="Pfad, unter dem eine visuelle HTML-Ansicht des Bretts gespeichert wird")
    args = parser.parse_args()

    model = load_model(args.model, args.algo)
    env = TakeItEasyEnv(render_mode="human")
    obs, info = env.reset(seed=args.seed)

    terminated = False
    reward = 0.0
    while not terminated:
        if args.algo == "ppo":
            action, _ = model.predict(obs, action_masks=info["action_mask"], deterministic=True)
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))

    env.render()
    print(f"\nFinaler Score: {reward:.0f}")

    if args.html:
        from board_render import board_to_html
        html = board_to_html(env.board, score=reward, title=f"Take It Easy - {args.algo.upper()}, Seed {args.seed}, Score {reward:.0f}")
        with open(args.html, "w") as f:
            f.write(html)
        print(f"Visuelles Brett gespeichert unter: {args.html}")
