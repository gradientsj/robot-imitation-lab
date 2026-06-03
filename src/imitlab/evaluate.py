"""Roll out a policy in the PushT environment and report success statistics.

Works with either a local checkpoint directory (from imitlab-train) or a
pretrained checkpoint on the Hugging Face Hub (e.g. lerobot/diffusion_pusht),
so our from-scratch run can be compared against the reference checkpoint
under identical evaluation conditions: same seeds, same episode budget, same
step limit.

In gym-pusht, an episode terminates with success when the T block covers at
least 95% of the goal region; reward is shaped coverage in [0, 1]. We report
success rate with a Wilson 95% interval (n is small) plus mean max coverage,
which gives partial credit to near-misses that a binary success hides.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .metrics import summarize_episodes


@dataclass
class EvalConfig:
    checkpoint: str = "lerobot/diffusion_pusht"
    name: str = "pretrained"
    n_episodes: int = 50
    seed: int = 100_000  # episode i is seeded with seed + i
    max_steps: int = 300
    gif_episodes: int = 2
    out_dir: str = "results"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = EvalConfig()
    parser.add_argument("--checkpoint", default=defaults.checkpoint,
                        help="local checkpoint dir or HF Hub repo id")
    parser.add_argument("--name", default=defaults.name,
                        help="label for this run in reports and output paths")
    parser.add_argument("--n-episodes", type=int, default=defaults.n_episodes)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--max-steps", type=int, default=defaults.max_steps)
    parser.add_argument("--gif-episodes", type=int, default=defaults.gif_episodes)
    parser.add_argument("--out-dir", default=defaults.out_dir)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = EvalConfig(
        checkpoint=args.checkpoint,
        name=args.name,
        n_episodes=args.n_episodes,
        seed=args.seed,
        max_steps=args.max_steps,
        gif_episodes=args.gif_episodes,
        out_dir=args.out_dir,
    )

    import gym_pusht  # noqa: F401  (registers gym_pusht/PushT-v0)
    import gymnasium as gym
    import imageio
    import numpy as np
    import torch
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = DiffusionPolicy.from_pretrained(config.checkpoint)
    policy.eval().to(device)
    print(f"loaded {config.checkpoint} on {device}")

    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        max_episode_steps=config.max_steps,
    )

    run_dir = Path(config.out_dir) / config.name
    run_dir.mkdir(parents=True, exist_ok=True)

    episodes: list[dict] = []
    for ep in range(config.n_episodes):
        obs, _ = env.reset(seed=config.seed + ep)
        policy.reset()
        frames = [env.render()] if ep < config.gif_episodes else None
        success = False
        max_reward = 0.0
        sum_reward = 0.0
        steps = 0
        while True:
            state = torch.from_numpy(obs["agent_pos"].astype(np.float32))
            image = torch.from_numpy(obs["pixels"].astype(np.float32) / 255.0)
            image = image.permute(2, 0, 1)  # HWC -> CHW
            batch = {
                "observation.state": state.unsqueeze(0).to(device),
                "observation.image": image.unsqueeze(0).to(device),
            }
            with torch.inference_mode():
                action = policy.select_action(batch)
            obs, reward, terminated, truncated, _ = env.step(
                action.squeeze(0).cpu().numpy()
            )
            steps += 1
            max_reward = max(max_reward, float(reward))
            sum_reward += float(reward)
            if frames is not None:
                frames.append(env.render())
            if terminated:  # gym-pusht terminates only on success (95% coverage)
                success = True
            if terminated or truncated:
                break
        episodes.append(
            {
                "episode": ep,
                "seed": config.seed + ep,
                "success": success,
                "steps": steps,
                "max_reward": round(max_reward, 4),
                "sum_reward": round(sum_reward, 2),
            }
        )
        print(
            f"episode {ep + 1:>3}/{config.n_episodes}  "
            f"{'SUCCESS' if success else 'fail   '}  max_reward {max_reward:.3f}"
        )
        if frames is not None:
            gif_path = run_dir / f"rollout_ep{ep}.gif"
            imageio.mimsave(gif_path, frames, fps=20)
            print(f"  saved {gif_path}")

    env.close()

    summary = summarize_episodes(
        successes=[e["success"] for e in episodes],
        max_rewards=[e["max_reward"] for e in episodes],
        sum_rewards=[e["sum_reward"] for e in episodes],
    )
    payload = {"name": config.name, "config": asdict(config), "summary": summary,
               "episodes": episodes}
    out_path = run_dir / "eval.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    low, high = summary["success_ci95"]
    print(
        f"\n{config.name}: success {summary['n_success']}/{summary['n_episodes']} "
        f"= {summary['success_rate']:.0%} (95% CI {low:.0%}-{high:.0%}), "
        f"mean max reward {summary['mean_max_reward']:.3f}"
    )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
