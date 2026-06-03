# The open-source robot foundation model landscape (June 2026)

A working survey of the major open vision-language-action (VLA) models and
robot foundation models, written as a companion to this repo. The policy
trained here (a Diffusion Policy on PushT) is a from-scratch, small-scale
instance of the same component these billion-parameter models use as their
action head, so the survey doubles as context for why that architecture
choice matters.

Facts below were link-verified in June 2026 against model cards, GitHub
READMEs, and papers; every HuggingFace repo id listed resolves. Claims that
could not be confirmed against a primary source are flagged [UNVERIFIED].
This field moves fast; treat dates as the ground truth and expect the
frontier to have moved by the time you read this.

## The one-table version

| Model | Org | Released | Params | VLM backbone | Action head | License | HF repo |
|---|---|---|---|---|---|---|---|
| GR00T N1 | NVIDIA | Mar 2025 | 2B | Eagle-2 (Qwen-based) | flow matching (DiT) | research-only | `nvidia/GR00T-N1-2B` |
| GR00T N1.5 | NVIDIA | Jun 2025 | 3B | Eagle 2.5 (frozen) | flow matching + FLARE | research-only | `nvidia/GR00T-N1.5-3B` |
| GR00T N1.7 | NVIDIA | Apr 2026 | 3B | Cosmos-Reason2-2B (Qwen3-VL) | flow matching, relative EE actions | Apache code / NVIDIA Open Model weights | `nvidia/GR00T-N1.7-3B` |
| pi0 | Physical Intelligence | Oct 2024 (open Feb 2025) | 3.3B | PaliGemma 3B | flow matching, 50-step chunks | Apache code / Gemma weights | via `openpi`, `lerobot/pi0*` |
| pi0-FAST | Physical Intelligence | Jan 2025 | 3B | PaliGemma 3B | autoregressive FAST tokens | Apache code / Gemma weights | via `openpi` |
| pi0.5 | Physical Intelligence | Apr 2025 | ~3.3B | PaliGemma 3B + web co-training | hybrid AR tokens + flow matching | Apache code / Gemma weights | `lerobot/pi05_base` |
| RDT-1B | Tsinghua TSAIL | Oct 2024 | 1.2B | SigLIP + T5-XXL encoders | diffusion transformer, 64-step chunks | MIT | `robotics-diffusion-transformer/rdt-1b` |
| RDT2 | Tsinghua TSAIL | Sep 2025 | [UNVERIFIED] | Qwen2.5-VL-7B (RDT2-VQ) | flow matching, 24-step chunks | Apache-2.0 | `robotics-diffusion-transformer/RDT2-FM` |
| OpenVLA | Stanford/Berkeley/TRI | Jun 2024 | 7B | Prismatic (DINOv2+SigLIP, Llama-2) | autoregressive discrete tokens | MIT code / Llama weights | `openvla/openvla-7b` |
| OpenVLA-OFT | Stanford | Feb 2025 | 7B | same as OpenVLA | parallel decoding, continuous L1 head | MIT | `moojink/openvla-7b-oft-*` |
| Octo | UC Berkeley RAIL | 2024 (RSS) | 27M / 93M | transformer (non-VLM) | diffusion head | MIT | `rail-berkeley/octo-base-1.5` |
| SmolVLA | Hugging Face | Jun 2025 | ~0.5B | SmolVLM-2 | flow matching expert (~100M) | Apache-2.0 (paper; card field empty) | `lerobot/smolvla_base` |
| WALL-OSS | X Square Robot | Sep 2025 / May 2026 (0.5: 4B) | 4B | Qwen2.5-VL MoE | dual flow-matching + FAST branches | [UNVERIFIED on card] | `x-square-robot/wall-oss-flow` |
| CogACT | Microsoft Research | Nov 2024 | 7B + 300M DiT | Prismatic | diffusion transformer module | MIT | `CogACT/CogACT-Base` |
| SpatialVLA | Shanghai AI Lab | Jan 2025 | 4B | PaliGemma2-3B + 3D encoding | adaptive action grids (discrete) | MIT | `IPEC-COMMUNITY/spatialvla-4b-224-pt` |
| MolmoAct | Allen Institute | Aug 2025 | 7B | Qwen2.5-7B + SigLip2 | reasoning traces + 7-D actions | Apache-2.0, open data | `allenai/MolmoAct-7B-D-0812` |

## The organizing idea: action heads

The cleanest way to read this field is by how each model turns a
vision-language representation into motor commands. Four generations of
answer, all still in active use:

1. **Discrete autoregressive tokens** (OpenVLA, Magma): bin each action
   dimension into 256 buckets and emit them like text tokens. Simple,
   reuses the whole LLM toolchain, but slow at inference (one token at a
   time) and lossy at fine resolution.
2. **FAST tokens** (pi0-FAST, WALL-OSS fast branch): compress action
   sequences with a DCT + BPE tokenizer before autoregressive decoding,
   about 5x faster to train than diffusion on the same data.
3. **Diffusion heads** (Octo, RDT-1B, CogACT, and the policy trained in
   this repo): denoise a chunk of future actions conditioned on the
   backbone's features. Handles multimodal action distributions well, which
   is exactly what imitation data has.
4. **Flow matching** (pi0/pi0.5, GR00T N1.x, SmolVLA, RDT2, WALL-OSS flow
   branch): same continuous-chunk idea as diffusion but trained with a
   flow-matching objective and far fewer integration steps at inference.
   This is the clear 2025-2026 default for new models.

Two other convergences worth noticing: action *chunking* is universal
(predict 16 to 64 future actions, execute a prefix, replan), and the VLM
backbones have collapsed onto a few families: PaliGemma (pi0, SpatialVLA),
Qwen-VL variants (GR00T's Eagle/Cosmos, RDT2, WALL-OSS), and
DINOv2+SigLIP/Prismatic (OpenVLA, CogACT).

## Model family notes

### NVIDIA Isaac GR00T (N1 -> N1.5 -> N1.7; N2 previewed)

NVIDIA's humanoid-focused line pairs an Eagle VLM with a flow-matching
diffusion-transformer action head. N1.5 (June 2025) froze the VLM during
training and added the FLARE latent-alignment loss; its headline result was
language-following on the real GR-1 humanoid jumping from 46.6% to 93.3%.
N1.7 (April 2026, Early Access) swapped the backbone to Cosmos-Reason2-2B,
added a relative end-effector action space, and trained on ~21k hours of
egocentric human video ("EgoScale"). N2 was previewed at GTC March 2026 with
world-action-model claims, but there are no weights yet; "by end of year."

The licensing arc matters for anyone building on these: N1 and N1.5 are
research-only (NVIDIA OneWay Non-Commercial), and only N1.7 moved to a
commercially usable license (Apache-2.0 code, NVIDIA Open Model weights).
Fine-tuning is documented at 40GB+ VRAM; inference fits in 16GB.

Sources: research.nvidia.com/labs/gear/gr00t-n1_5, github.com/NVIDIA/Isaac-GR00T,
huggingface.co/blog/nvidia/gr00t-n1-7

### Physical Intelligence pi0 / pi0-FAST / pi0.5 (openpi)

pi0 set the template most of the field now follows: a 3B PaliGemma VLM plus
a 300M action expert producing 50-step action chunks at 50Hz via flow
matching, pretrained on 10k+ hours of teleop data across 7 platforms. The
FAST variant replaced the flow head with compressed discrete tokens for ~5x
cheaper training. pi0.5 (April 2025) is the generalization push: co-trained
on web data with "knowledge insulation," demonstrated tidying homes it had
never seen. The openpi repo documents LoRA fine-tuning at >22.5GB VRAM,
which is precisely the "fits on one RTX 4090" line.

Sources: github.com/Physical-Intelligence/openpi, pi.website/blog/pi05,
huggingface.co/blog/pi0

### RDT-1B and RDT2 (Tsinghua TSAIL)

RDT-1B (October 2024) was the largest pure diffusion-transformer policy of
its generation: 1.2B parameters, SigLIP vision plus T5-XXL text encoders,
64-action chunks, pretrained on 1M+ episodes across 46 datasets and
fine-tuned on 6k+ bimanual ALOHA episodes, all MIT-licensed. RDT2
(September 2025) is a substantial redesign rather than a scale-up: it moves
to a Qwen2.5-VL-7B backbone (RDT2-VQ), switches from iterative denoising to
flow matching with 24-step chunks of 20-D bimanual actions, trains on 10k+
hours of UMI manipulation data, and targets zero-shot deployment on unseen
embodiments. Apache-2.0, with documented inference around 16GB VRAM. The
RDT2 paper reference circulating online looks garbled, so the arXiv id is
[UNVERIFIED]; the HF release and date are confirmed.

Sources: huggingface.co/robotics-diffusion-transformer/rdt-1b and /RDT2-FM,
rdt-robotics.github.io

### OpenVLA and OpenVLA-OFT

OpenVLA (June 2024) is the reference open autoregressive VLA: 7B Prismatic
backbone (fused DINOv2+SigLIP vision into Llama-2), 256-bin discrete action
tokens, pretrained on ~970K Open X-Embodiment trajectories. It beat the 55B
RT-2-X with 7x fewer parameters and became the standard fine-tuning
baseline. OpenVLA-OFT (February 2025) is the lesson the whole field
absorbed: swapping autoregressive decoding for parallel decoding with
chunked continuous actions (a plain L1 regression head) plus proprioception
produced large gains in both speed and LIBERO success. Documented minimum
for fine-tuning is ~27GB, slightly over a 24GB consumer card.

Sources: github.com/openvla/openvla, openvla-oft.github.io

### Octo (Berkeley RAIL)

The small-model counterpoint: 27M and 93M parameter transformers with a
diffusion action head, goal-image and language conditioning, trained on
800K OXE trajectories. Pre-VLM by design and tiny by today's standards, but
it normalized two ideas the big models kept: diffusion heads and
multi-embodiment pretraining. MIT licensed.

Sources: octo-models.github.io, arxiv.org/abs/2405.12213

### SmolVLA (Hugging Face)

The accessibility play: ~0.5B total (SmolVLM-2 backbone, ~100M flow-matching
action expert), trained exclusively on community-contributed LeRobot
datasets (under 30K episodes), explicitly designed to fine-tune on a single
consumer GPU and run asynchronously for ~2x control throughput. The paper
and blog say Apache-2.0; the model card's license field is empty, so verify
before commercial use.

Sources: huggingface.co/blog/smolvla, huggingface.co/lerobot/smolvla_base

### The 2025-2026 wave in brief

- **WALL-OSS** (X Square Robot): Qwen2.5-VL MoE backbone with parallel
  flow-matching and FAST action branches; integrated into LeRobot. License
  not declared on the model cards.
- **CogACT** (Microsoft Research): OpenVLA-style backbone but with a
  componentized 300M DiT action module; claims >55% real-robot improvement
  over similarly sized OpenVLA. MIT.
- **SpatialVLA** (Shanghai AI Lab): PaliGemma2 plus explicit 3D spatial
  encodings and adaptive action grids; topped LIBERO averages at release.
  MIT.
- **MolmoAct** (AI2): an "action reasoning model" that emits depth tokens
  and visual reasoning traces before actions; fully open including data.
  Apache-2.0.
- **Magma** (Microsoft): one 8B agent across UI navigation and manipulation
  rather than a dedicated policy; interesting as a generalist data point.
  MIT.

## What actually fine-tunes on a 24GB consumer GPU

The practical question for anyone with a single RTX 4090:

| Feasibility | Models |
|---|---|
| Comfortable | SmolVLA (designed for it), Octo, anything LeRobot-native at PushT/ALOHA scale, including this repo's Diffusion Policy |
| Fits with LoRA | pi0 (>22.5GB documented) |
| Just over the line | OpenVLA (~27GB minimum documented) |
| Needs datacenter hardware | GR00T full fine-tune (40GB+), pi0 full fine-tune (70GB+), RDT/SpatialVLA/CogACT pretraining |

## License gotchas worth knowing

"Open" spans four different things here: truly open (MIT/Apache code and
weights: RDT-1B, Octo, CogACT, SpatialVLA, MolmoAct, OpenVLA-OFT),
weights-encumbered (pi0 family inherits the Gemma license; OpenVLA weights
inherit Llama-2's), research-only (GR00T N1/N1.5), and undeclared (SmolVLA
and WALL-OSS model cards omit the license field even where papers claim
Apache). If a use case is commercial, read the actual card, not the blog
post.

## How this repo relates

Every flow-matching or diffusion VLA above is, at its action head, doing
what this repo does in miniature: learning to denoise a chunk of future
actions conditioned on visual features, trained by imitation. Training that
component from scratch at PushT scale (262M parameters, one consumer GPU,
one afternoon) is the cheapest honest way to build intuition for the design
choices that show up at 3B scale: observation history length, action
horizon, chunk-prefix execution, and the evaluation variance that makes
success-rate comparisons without confidence intervals meaningless.
