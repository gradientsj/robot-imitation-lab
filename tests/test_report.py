from imitlab.metrics import summarize_episodes
from imitlab.report import render_comparison


def test_render_comparison_table():
    summary = summarize_episodes([True, True, False, True], [1.0, 0.9, 0.2, 0.8], [9, 8, 2, 7])
    table = render_comparison(
        [
            {"name": "ours (25k steps)", "summary": summary},
            {"name": "lerobot/diffusion_pusht", "summary": summary},
        ]
    )
    assert table.count("\n") == 3  # header + separator + 2 rows
    assert "ours (25k steps)" in table
    assert "75% (3/4)" in table
    assert "0.725" in table
