import torch

from kemudi.engine import Axis
from kemudi.extract import candidate_layers, choose_layer
from kemudi.measure import random_direction


def test_candidate_blocks_cover_the_middle_band():
    assert candidate_layers(24) == list(range(4, 15))
    assert candidate_layers(12) == [2, 3, 4, 5, 6, 7]


def test_block_choice_prefers_separation_among_equally_good_blocks():
    dev_auc = {4: 0.80, 5: 1.00, 6: 0.99, 7: 0.97}
    separation = {4: 0.90, 5: 0.02, 6: 0.05, 7: 0.20}
    assert choose_layer(dev_auc, separation) == 6


def test_block_choice_never_trades_held_out_separation_for_size():
    assert choose_layer({4: 0.70, 5: 0.95}, {4: 0.50, 5: 0.01}) == 5
    assert choose_layer({4: 1.0, 5: 1.0}, {4: 0.03, 5: 0.03}) == 4


def test_random_direction_is_a_unit_vector_at_right_angles_and_repeatable():
    unit = torch.zeros(32)
    unit[3] = 1.0
    axis = Axis("mood", "bleak", "hopeful", layer=7, unit=unit, scale=9.0)
    placebo = random_direction(axis, seed=5)
    assert abs(float(placebo.unit.norm()) - 1.0) < 1e-5
    assert abs(float(placebo.unit @ unit)) < 1e-6
    assert (placebo.layer, placebo.scale, placebo.name) == (7, 9.0, "mood")
    assert torch.equal(placebo.unit, random_direction(axis, seed=5).unit)
    assert not torch.equal(placebo.unit, random_direction(axis, seed=6).unit)


class PoleTokenizer:
    def apply_chat_template(self, messages, **_ignored):
        return messages[0]["content"] + " ."

    def __call__(self, text, **_ignored):
        return type("Encoded", (), {"input_ids": text.split()})()


class PoleEngine:
    """Ten blocks of width 6. Text marked HIGH is shifted along one direction, more so in later blocks."""

    layers = [None] * 10
    width = 6
    tokenizer = PoleTokenizer()
    direction = torch.tensor([0.0, 0.6, 0.0, 0.8, 0.0, 0.0])

    def hidden_states(self, text):
        words = text.split()
        generator = torch.Generator().manual_seed(len(text))
        states = 3.0 + 0.01 * torch.randn(10, len(words), 6, generator=generator)
        sign = 1.0 if "HIGH" in text else -1.0
        strength = torch.arange(10).float().view(10, 1, 1) * 0.1
        return states + sign * strength * self.direction


def test_extract_recovers_the_direction_and_reports_held_out_separation():
    from kemudi.extract import extract

    data = {
        "extraction_openers": ["the boat left", "a door opened slowly", "rain fell"],
        "axes": {"mood": {
            "low": "bleak", "high": "hopeful", "low_styles": ["grim"], "high_styles": ["HIGH"],
            "low_examples": ["all was lost here", "nobody came back", "the well ran dry", "it ended badly"],
            "high_examples": ["HIGH we won it", "HIGH they came home", "HIGH the rain came", "HIGH it ended well"],
        }},
    }
    summary, units = extract(PoleEngine(), data, "toy", log=lambda _line: None)
    entry = summary["axes"]["mood"]
    assert entry["layer"] == 6, "block 6 is the deepest block in the 20-60% band, where the poles are furthest apart"
    assert float(units["mood"] @ PoleEngine.direction) > 0.99
    assert entry["dev_auc"] == 1.0 and entry["test_auc"] == 1.0
    assert entry["gauge_half"] > 0 and entry["scale"] > 0 and entry["pairs"] == 3
    assert summary["blocks"] == 10 and summary["width"] == 6
