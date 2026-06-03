"""Episode-level evaluation statistics. Pure Python, no heavy deps.

Success rate over N rollouts is a binomial estimate, and with N around 50 the
uncertainty is large; reporting a bare percentage invites over-reading a few
points of difference between policies. We report Wilson score intervals,
which behave sensibly at small N and at success rates near 0 or 1 (a normal
approximation does not).
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (default 95%)."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError("successes must be in [0, n]")
    p_hat = successes / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    half = (z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def summarize_episodes(
    successes: Sequence[bool], max_rewards: Sequence[float], sum_rewards: Sequence[float]
) -> dict:
    """Aggregate per-episode outcomes into the summary the report renders."""
    if not (len(successes) == len(max_rewards) == len(sum_rewards)):
        raise ValueError("per-episode sequences must have equal length")
    if not successes:
        raise ValueError("no episodes to summarize")
    n = len(successes)
    k = sum(bool(s) for s in successes)
    low, high = wilson_interval(k, n)
    return {
        "n_episodes": n,
        "n_success": k,
        "success_rate": k / n,
        "success_ci95": [low, high],
        "mean_max_reward": sum(max_rewards) / n,
        "mean_sum_reward": sum(sum_rewards) / n,
    }
