from imitlab.evaluate import extract_stats_from_state_dict

FEATURES = ["observation.image", "observation.state", "action"]


def test_extracts_old_format_buffers():
    state_dict = {
        "normalize_inputs.buffer_observation_image.mean": [0.485, 0.456, 0.406],
        "normalize_inputs.buffer_observation_image.std": [0.229, 0.224, 0.225],
        "normalize_inputs.buffer_observation_state.min": [13.5, 32.9],
        "normalize_inputs.buffer_observation_state.max": [496.1, 510.9],
        "normalize_targets.buffer_action.min": [12.0, 25.0],
        "normalize_targets.buffer_action.max": [511.0, 511.0],
        # Duplicate stats under another module must not break extraction.
        "unnormalize_outputs.buffer_action.min": [12.0, 25.0],
        "unnormalize_outputs.buffer_action.max": [511.0, 511.0],
        # Unrelated weights are ignored.
        "diffusion.unet.conv.weight": [0.0],
    }
    stats = extract_stats_from_state_dict(state_dict, FEATURES)
    assert stats is not None
    assert stats["observation.image"]["mean"] == [0.485, 0.456, 0.406]
    assert stats["observation.state"]["max"] == [496.1, 510.9]
    assert stats["action"]["min"] == [12.0, 25.0]
    assert set(stats) == {"observation.image", "observation.state", "action"}


def test_returns_none_for_new_format_state_dict():
    state_dict = {"diffusion.unet.conv.weight": [0.0]}
    assert extract_stats_from_state_dict(state_dict, FEATURES) is None
