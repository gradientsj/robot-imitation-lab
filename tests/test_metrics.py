import math

import pytest

from imitlab.metrics import summarize_episodes, wilson_interval


def test_wilson_known_value():
    # 25/50 with z=1.96: standard worked example.
    low, high = wilson_interval(25, 50)
    assert low == pytest.approx(0.3673, abs=1e-3)
    assert high == pytest.approx(0.6327, abs=1e-3)


def test_wilson_extremes_stay_in_bounds():
    low, high = wilson_interval(0, 20)
    assert low == 0.0
    assert 0 < high < 0.25
    low, high = wilson_interval(20, 20)
    assert 0.75 < low < 1.0
    assert high == 1.0


def test_wilson_tightens_with_n():
    narrow = wilson_interval(80, 100)
    wide = wilson_interval(8, 10)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_wilson_validates_inputs():
    with pytest.raises(ValueError):
        wilson_interval(1, 0)
    with pytest.raises(ValueError):
        wilson_interval(5, 4)


def test_summarize_episodes():
    summary = summarize_episodes(
        successes=[True, False, True, True],
        max_rewards=[0.9, 0.4, 1.0, 0.8],
        sum_rewards=[10.0, 5.0, 12.0, 9.0],
    )
    assert summary["n_episodes"] == 4
    assert summary["n_success"] == 3
    assert summary["success_rate"] == pytest.approx(0.75)
    assert summary["mean_max_reward"] == pytest.approx(0.775)
    low, high = summary["success_ci95"]
    assert 0 <= low < 0.75 < high <= 1.0
    assert not math.isnan(low)


def test_summarize_validates():
    with pytest.raises(ValueError):
        summarize_episodes([], [], [])
    with pytest.raises(ValueError):
        summarize_episodes([True], [1.0, 2.0], [1.0])
