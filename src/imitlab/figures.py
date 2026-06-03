"""Publication-style figures from eval JSONs and training logs.

Produces three artifacts under --out-dir:
  loss_curve.png          log-scale training loss (if --train-log given)
  success_comparison.png  success rate per policy with Wilson 95% CI bars
  reward_distribution.png per-episode max coverage, one column per policy,
                          with the 95% success threshold marked

The distribution plot exists because a binary success rate hides failure
modes: a policy that always reaches 90% coverage and one that alternates
between 100% and 20% can have the same success rate and very different
behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# gym-pusht reward is goal coverage normalized by the 95% success threshold
# and clipped to [0, 1], so an episode succeeds iff its max reward reaches 1.0.
SUCCESS_REWARD = 1.0


def _load_eval(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def plot_loss(csv_path: Path, out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps, losses = [], []
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            losses.append(float(row["loss"]))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, losses, linewidth=1.2)
    ax.set_xlabel("training step")
    ax.set_ylabel("diffusion loss (100-step mean)")
    ax.set_yscale("log")
    ax.set_title("Diffusion Policy training on PushT")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def plot_success_comparison(evals: list[dict], out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [e["name"] for e in evals]
    rates = [e["summary"]["success_rate"] for e in evals]
    los = [r - e["summary"]["success_ci95"][0] for r, e in zip(rates, evals, strict=True)]
    his = [e["summary"]["success_ci95"][1] - r for r, e in zip(rates, evals, strict=True)]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = range(len(names))
    ax.bar(x, rates, width=0.55, color="#4878cf", alpha=0.9)
    ax.errorbar(x, rates, yerr=[los, his], fmt="none", ecolor="#1d1d1f", capsize=6)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylabel("success rate")
    ax.set_ylim(0, 1)
    n = evals[0]["summary"]["n_episodes"]
    ax.set_title(f"PushT success over {n} matched-seed episodes (Wilson 95% CI)")
    for xi, (rate, e) in enumerate(zip(rates, evals, strict=True)):
        s = e["summary"]
        ax.annotate(
            f"{s['n_success']}/{s['n_episodes']}",
            (xi, rate),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=10,
        )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def plot_reward_distribution(evals: list[dict], out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, e in enumerate(evals):
        rewards = [ep["max_reward"] for ep in e["episodes"]]
        # Deterministic horizontal jitter so the figure is reproducible.
        xs = [i + ((j * 37) % 21 - 10) / 60 for j in range(len(rewards))]
        colors = ["#2a9d4e" if ep["success"] else "#c44e52" for ep in e["episodes"]]
        ax.scatter(xs, rewards, s=22, c=colors, alpha=0.75, edgecolors="none")
    ax.axhline(SUCCESS_REWARD, linestyle="--", color="#1d1d1f", linewidth=1,
               label="success = reward 1.0 (95% goal coverage)")
    ax.set_xticks(range(len(evals)))
    ax.set_xticklabels([e["name"] for e in evals])
    ax.set_ylabel("episode max reward (normalized goal coverage)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-episode outcomes (green = success)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_jsons", nargs="+", help="eval.json paths, plotted in order")
    parser.add_argument("--train-log", default=None, help="training_log.csv for the loss curve")
    parser.add_argument("--out-dir", default="results/figures")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    evals = [_load_eval(Path(p)) for p in args.eval_jsons]

    plot_success_comparison(evals, out_dir / "success_comparison.png")
    plot_reward_distribution(evals, out_dir / "reward_distribution.png")
    print(f"wrote {out_dir / 'success_comparison.png'}")
    print(f"wrote {out_dir / 'reward_distribution.png'}")
    if args.train_log:
        plot_loss(Path(args.train_log), out_dir / "loss_curve.png")
        print(f"wrote {out_dir / 'loss_curve.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
