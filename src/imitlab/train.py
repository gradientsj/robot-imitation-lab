"""Train a Diffusion Policy on a LeRobot dataset (default: PushT).

A deliberately compact, readable training loop built directly on the LeRobot
0.4.x API (dataset + policy classes) rather than the framework's trainer CLI,
so every step is visible: feature wiring, delta-timestamp construction,
normalization stats, optimization, and checkpointing.

Heavy imports live inside main() so `import imitlab` stays light enough for
CPU-only CI.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    repo_id: str = "lerobot/pusht"
    steps: int = 25_000
    batch_size: int = 64
    grad_clip_norm: float = 10.0
    num_workers: int = 4
    seed: int = 42
    log_every: int = 100
    out_dir: str = "outputs/train/diffusion_pusht"
    video_backend: str = "pyav"  # torchcodec is flaky on Windows; pyav is portable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = TrainConfig()
    parser.add_argument("--repo-id", default=defaults.repo_id)
    parser.add_argument("--steps", type=int, default=defaults.steps)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--log-every", type=int, default=defaults.log_every)
    parser.add_argument("--out-dir", default=defaults.out_dir)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = TrainConfig(
        repo_id=args.repo_id,
        steps=args.steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        log_every=args.log_every,
        out_dir=args.out_dir,
    )

    import numpy as np
    import torch
    from lerobot.configs.types import FeatureType
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.utils import dataset_to_policy_features
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.policies.factory import make_pre_post_processors

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Wire the policy's input/output features from the dataset schema, then
    # derive the temporal windows (observation history + action horizon) the
    # policy wants, expressed as delta timestamps at the dataset's fps.
    metadata = LeRobotDatasetMetadata(config.repo_id)
    features = dataset_to_policy_features(metadata.features)
    output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {k: ft for k, ft in features.items() if k not in output_features}
    policy_config = DiffusionConfig(
        input_features=input_features, output_features=output_features
    )
    delta_timestamps = {
        key: [i / metadata.fps for i in policy_config.observation_delta_indices]
        for key in input_features
    }
    delta_timestamps["action"] = [
        i / metadata.fps for i in policy_config.action_delta_indices
    ]

    dataset = LeRobotDataset(
        config.repo_id,
        delta_timestamps=delta_timestamps,
        video_backend=config.video_backend,
    )
    print(
        f"dataset {config.repo_id}: {dataset.num_episodes} episodes, "
        f"{dataset.num_frames} frames, fps {metadata.fps}"
    )

    policy = DiffusionPolicy(policy_config)
    policy.train().to(device)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"policy: DiffusionPolicy, {n_params / 1e6:.1f}M parameters, device {device}")

    # In lerobot 0.4.x the policy itself does NOT normalize; normalization
    # lives in processor pipelines built from the dataset statistics. Skipping
    # this step silently trains on raw pixel-coordinate states/actions and
    # produces a useless policy (the loss still goes down, which is the trap).
    preprocessor, postprocessor = make_pre_post_processors(
        policy_config, dataset_stats=dataset.meta.stats
    )

    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=policy_config.optimizer_lr,
        betas=policy_config.optimizer_betas,
        eps=policy_config.optimizer_eps,
        weight_decay=policy_config.optimizer_weight_decay,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=config.num_workers > 0,
    )

    log_path = out_dir / "training_log.csv"
    log_file = log_path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(log_file)
    writer.writerow(["step", "loss", "steps_per_s"])

    step = 0
    running_loss = 0.0
    window_start = time.perf_counter()
    train_start = time.perf_counter()
    done = False
    while not done:
        for batch in loader:
            # Normalizes features and moves tensors to the policy device.
            batch = preprocessor(batch)
            loss, _ = policy.forward(batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), config.grad_clip_norm)
            optimizer.step()

            step += 1
            running_loss += loss.item()
            if step % config.log_every == 0:
                elapsed = time.perf_counter() - window_start
                avg_loss = running_loss / config.log_every
                sps = config.log_every / elapsed
                writer.writerow([step, f"{avg_loss:.5f}", f"{sps:.2f}"])
                log_file.flush()
                print(f"step {step:>6}/{config.steps}  loss {avg_loss:.4f}  {sps:.1f} steps/s")
                running_loss = 0.0
                window_start = time.perf_counter()
            if step >= config.steps:
                done = True
                break

    log_file.close()
    total_min = (time.perf_counter() - train_start) / 60
    checkpoint_dir = out_dir / "checkpoint"
    policy.save_pretrained(checkpoint_dir)
    # Save the processor pipelines next to the weights so the checkpoint is
    # complete: evaluation can reload normalization without the dataset.
    preprocessor.save_pretrained(checkpoint_dir)
    postprocessor.save_pretrained(checkpoint_dir)
    (out_dir / "train_config.json").write_text(
        json.dumps(asdict(config) | {"wall_minutes": round(total_min, 1)}, indent=2) + "\n",
        encoding="utf-8",
    )
    _plot_loss_curve(log_path, out_dir / "loss_curve.png")
    print(f"done in {total_min:.1f} min; checkpoint at {checkpoint_dir}")
    return 0


def _plot_loss_curve(log_path: Path, png_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps, losses = [], []
    with log_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            losses.append(float(row["loss"]))
    if not steps:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, losses, linewidth=1.2)
    ax.set_xlabel("training step")
    ax.set_ylabel("diffusion loss (100-step mean)")
    ax.set_yscale("log")
    ax.set_title("Diffusion Policy on PushT")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
