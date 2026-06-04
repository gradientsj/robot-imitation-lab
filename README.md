# robot-imitation-lab

[![ci](https://github.com/gradientsj/robot-imitation-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/gradientsj/robot-imitation-lab/actions/workflows/ci.yml)

**Imitation-learning robot policies on open data: train a Diffusion Policy
from human demonstrations on PushT, evaluate it with confidence intervals,
and compare against the pretrained reference checkpoint under a matched
protocol.** Plus a link-verified survey of the open VLA model landscape
([docs/vla-landscape.md](docs/vla-landscape.md)).

Everything runs on open Hugging Face assets: the
[`lerobot/pusht`](https://huggingface.co/datasets/lerobot/pusht) dataset
(206 human demonstrations, 25,650 frames), the
[`lerobot/diffusion_pusht`](https://huggingface.co/lerobot/diffusion_pusht)
reference checkpoint, and the [LeRobot](https://github.com/huggingface/lerobot)
library (0.4.x), trained and evaluated on a single RTX 4090.

## Results

Both policies, same architecture (262M-parameter Diffusion Policy), same
data, same evaluation: 50 rollouts in gym-pusht with identical seeds, a
300-step limit, and success defined by the environment (95% goal coverage).

| Policy | Training | Success rate | Wilson 95% CI | Mean max reward |
|---|---|---|---|---|
| `ours_diffusion_25k` | 25k steps, constant lr (~2 h) | 10% (5/50) | 4% - 21% | 0.445 |
| `ours_diffusion_50k` | 50k milestone of the 200k run | 6% (3/50) | 2% - 16% | 0.569 |
| `ours_diffusion_100k` | 100k milestone | 24% (12/50) | 14% - 37% | 0.572 |
| `ours_diffusion_150k` | 150k milestone | **28%** (14/50) | 17% - 42% | 0.669 |
| `ours_diffusion_200k` | 200k steps, cosine + warmup (~14 h) | 14% (7/50) | 7% - 26% | 0.595 |
| `lerobot/diffusion_pusht` | 200k steps, reference recipe | **68%** (34/50) | 54% - 79% | 0.957 |

(Milestones are snapshots of one 200k cosine schedule, so their learning
rate had not finished decaying; they trace the trajectory rather than
standing in for independent runs at those budgets.)

![scaling curve](results/figures/scaling_curve.png)

Three readings of this table, and the third is the interesting one:

1. **The protocol is validated.** The reference checkpoint reports 65.4%
   success / 0.955 avg max reward on its model card (500 episodes). Our
   independent 50-episode protocol lands at 68% / 0.957, inside the CI of
   the published number. An evaluation harness that cannot reproduce a known
   result cannot be trusted to compare anything.
2. **Compute helps, noisily, for a while.** Success climbs from 10% toward
   28% as training proceeds, with overlapping CIs that 50 episodes cannot
   fully separate.
3. **Steps alone did not close the gap.** At the full 200k budget with the
   reference's batch size, learning rate, and warmup schedule, our run
   reached 14% against the reference's 68%, and the final checkpoint scored
   *below* the 150k milestone while training loss fell monotonically to
   0.0020. Diffing the two checkpoints' configs pinpoints the difference:
   the reference trained with random 84x84 crop augmentation
   (`crop_shape: [84, 84]`), while LeRobot 0.4.x defaults `crop_shape` to
   `None`, so our run had no image augmentation at all. On 206
   demonstrations, that is a recipe for the vision encoder to memorize the
   training frames, and a success curve that peaks mid-run and regresses
   while the loss keeps improving is what that looks like from the outside.
   The corrected run (same budget, crop augmentation restored) is queued as
   next step #1.

![success comparison](results/figures/success_comparison.png)

The per-episode picture says more than the rates. Both of our checkpoints
show the full failure spectrum (clean successes, near-misses, partial
pushes, whiffs), while the reference is essentially bimodal at the top:

![per-episode outcomes](results/figures/reward_distribution.png)

Training loss for the 200k run, for the record; note that nothing in this
curve hints that task success peaked five hours earlier (log scale,
100-step means):

![loss curve](results/figures/loss_curve.png)

Rollouts on the same seed (left to right: ours at 25k, ours at 200k, the
reference):

| ours 25k | ours 200k | pretrained reference |
|---|---|---|
| ![ours 25k rollout](results/ours_diffusion_25k/rollout_ep0.gif) | ![ours 200k rollout](results/ours_diffusion_200k/rollout_ep0.gif) | ![reference rollout](results/pretrained_diffusion_pusht/rollout_ep0.gif) |

Raw per-episode records live in `results/<run>/eval.json` for all six
evaluations. Training logs and config snapshots:
[`results/training_log.csv`](results/training_log.csv) /
[`results/train_config.json`](results/train_config.json) (25k run),
[`results/training_log_200k.csv`](results/training_log_200k.csv) /
[`results/train_config_200k.json`](results/train_config_200k.json) (200k run).

## Three failure modes this project had to catch

All three failed silently, and all three are recorded in the commit history
as they happened:

1. **The loss can lie.** LeRobot 0.4.x moved normalization out of the policy
   into processor pipelines; the older `dataset_stats` constructor kwarg is
   silently swallowed. A loop wired the old way trains on raw pixel
   coordinates, and the diffusion loss still goes down. Only rollouts expose
   it. Consequence in the design: every batch goes through the preprocessor
   explicitly, and the processor pipelines are saved next to the weights so
   a checkpoint is a complete, self-describing artifact.
2. **Normalization stats are part of a checkpoint's contract.** The
   reference checkpoint normalizes images with ImageNet statistics, not
   PushT dataset statistics (mean ~0.97: the board is mostly white).
   Rebuilding normalization from dataset stats degraded it from 65% to 4%
   success while it still pushed the block to 0.66 mean coverage, which is
   exactly what makes the failure dangerous: it reads as "mediocre policy,"
   not "broken pipeline." The evaluator therefore resolves normalization in
   priority order: processors saved with the checkpoint, then stats embedded
   in old-format state dicts (extracted, unit tested), then dataset stats
   with a warning.
3. **Defaults drift across library versions.** The reference recipe trained
   with random 84x84 crop augmentation; LeRobot 0.4.x changed the
   `crop_shape` default to `None`, so a training run written against current
   defaults silently loses the augmentation the published results depend on.
   Nothing fails, the loss improves, and task success quietly stalls then
   regresses (28% at 150k steps, 14% at 200k). The config diff between
   checkpoints is what surfaced it, which is an argument for always
   committing config snapshots next to results.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and ideally an NVIDIA GPU (CUDA
12.8 wheels are pinned for Windows; evaluation alone also works, slowly, on
CPU).

```bash
uv sync --extra dev
uv run pytest                  # 9 tests, CPU-only (metrics, parsing, reports)

# train from scratch (~2 h on an RTX 4090)
uv run imitlab-train --steps 25000 --num-workers 8 --out-dir outputs/train/diffusion_pusht_25k

# evaluate both policies on identical seeds
uv run imitlab-eval --checkpoint outputs/train/diffusion_pusht_25k/checkpoint --name ours_diffusion_25k
uv run imitlab-eval --checkpoint lerobot/diffusion_pusht --name pretrained_diffusion_pusht

# figures
uv run imitlab-figures results/ours_diffusion_25k/eval.json results/pretrained_diffusion_pusht/eval.json --train-log outputs/train/diffusion_pusht_25k/training_log.csv
```

CI runs lint and the unit tests CPU-only; training and rollouts happen on
real hardware and their outputs are committed under `results/`.

## The VLA survey

[docs/vla-landscape.md](docs/vla-landscape.md) maps the open robot
foundation model landscape as of June 2026: GR00T N1 through N1.7, pi0 /
pi0-FAST / pi0.5, RDT-1B and RDT2, OpenVLA and OpenVLA-OFT, Octo, SmolVLA,
WALL-OSS, CogACT, SpatialVLA, and MolmoAct, organized by action-head
taxonomy (discrete tokens, FAST tokens, diffusion, flow matching), with
licensing notes and a "what fine-tunes on a 24 GB consumer GPU" table. Every
HF repo id was verified to resolve; unverifiable claims are flagged.

The connection to this repo: diffusion/flow-matching action heads are what
those 3B-parameter models use to produce motor commands. Training one from
scratch at PushT scale is the cheapest way to build real intuition for
their design choices (observation history, action horizon, chunk-prefix
execution) and for the evaluation variance that makes success-rate
comparisons without confidence intervals meaningless.

## Repository layout

```
src/imitlab/
  train.py      # compact Diffusion Policy training loop (lerobot 0.4.x API)
  evaluate.py   # matched-seed rollout eval; normalization provenance chain
  figures.py    # scaling curve, success comparison, distributions, loss
  metrics.py    # Wilson intervals + episode summaries (pure Python, tested)
  report.py     # markdown comparison table
docs/
  vla-landscape.md            # the open VLA model survey
results/
  ours_diffusion_25k/         # 25k run: eval.json + rollout GIFs
  ours_diffusion_{50,100,150}k/  # 200k-run milestones: eval.json each
  ours_diffusion_200k/        # final 200k checkpoint: eval.json + GIFs
  pretrained_diffusion_pusht/ # reference: eval.json + rollout GIFs
  figures/                    # committed PNGs used above
  training_log{,_200k}.csv    # loss/throughput logs for both runs
  train_config{,_200k}.json   # exact run configurations
tests/                        # 9 CPU-only tests
```

## Limitations and next steps

A single task and a single embodiment in simulation, evaluated over 50
episodes: the CIs in the table are the true width of what this measures.
Next steps in rough priority order:

1. **Recipe-matched run**: the compute-matched run is done and answered its
   question in the negative; the follow-up restores the reference's random
   84x84 crop augmentation at the same 200k budget and tests whether recipe
   parity closes the remaining 14% vs 68% gap.
2. **Tighter evaluation**: 200-500 episodes to shrink the CIs before making
   any finer-grained claims.
3. **A VLA fine-tune**: SmolVLA (0.5B, designed for consumer GPUs) on a
   LeRobot community dataset, connecting the survey to practice.
4. **A second task/embodiment** (ALOHA sim transfer-cube) to test that the
   harness generalizes beyond PushT.

## License

MIT
