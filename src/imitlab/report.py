"""Render evaluation summaries into the markdown comparison table.

Deterministic output (no timestamps): identical inputs produce identical
files, so diffs show real changes only.
"""

from __future__ import annotations


def render_comparison(rows: list[dict]) -> str:
    """rows: [{"name": ..., "summary": <summarize_episodes output>}, ...]"""
    lines = [
        "| Policy | Success rate | 95% CI | Mean max reward | Episodes |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        s = row["summary"]
        low, high = s["success_ci95"]
        lines.append(
            f"| {row['name']} | {s['success_rate']:.0%} ({s['n_success']}/{s['n_episodes']}) "
            f"| {low:.0%} - {high:.0%} | {s['mean_max_reward']:.3f} | {s['n_episodes']} |"
        )
    return "\n".join(lines)
