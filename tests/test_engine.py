import threading

import pytest
import torch

from kemudi.engine import Axis, sample_token

IDS = torch.tensor([[5, 9, 12, 7, 30]])


def states(engine):
    """Residual after every block, read from the model itself: [blocks, tokens, width]."""
    with torch.inference_mode():
        return torch.stack(engine.model(input_ids=IDS, output_hidden_states=True).hidden_states[1:])[:, 0]


def logits(engine):
    with torch.inference_mode():
        return engine.model(input_ids=IDS).logits


def test_zero_knobs_change_nothing(engine, axes):
    before = logits(engine)
    engine.set_knobs(axes, {"a": 0.0, "c": 0})
    assert torch.equal(logits(engine), before)


def test_knob_adds_exactly_dose_times_scale_times_unit_at_its_block(engine, axes):
    plain = states(engine)
    engine.set_knobs(axes, {"a": 0.25})
    steered = states(engine)
    assert torch.equal(steered[0], plain[0])
    expected = 0.25 * axes["a"].scale * axes["a"].unit
    assert torch.allclose(steered[1] - plain[1], expected.expand(5, 16), atol=1e-5)
    assert not torch.allclose(steered[3], plain[3], atol=1e-3)


def test_two_knobs_on_one_block_add_up_and_blocks_stay_separate(engine, axes):
    plain = states(engine)
    engine.set_knobs(axes, {"a": 0.2, "b": -0.4, "c": 0.1})
    steered = states(engine)
    expected = 0.2 * 5.0 * axes["a"].unit - 0.4 * 3.0 * axes["b"].unit
    assert torch.allclose(steered[1] - plain[1], expected.expand(5, 16), atol=1e-5)
    assert set(engine._delta) == {1, 2}


def test_opposite_doses_move_the_output_in_opposite_directions(engine, axes):
    plain = states(engine)[1]
    engine.set_knobs(axes, {"a": 0.3})
    up = states(engine)[1]
    engine.set_knobs(axes, {"a": -0.3})
    down = states(engine)[1]
    assert torch.allclose(up - plain, plain - down, atol=1e-5)
    assert ((up - plain) @ axes["a"].unit > 0).all()


def test_unsteered_runs_clean_and_puts_the_knobs_back(engine, axes):
    before = logits(engine)
    engine.set_knobs(axes, {"a": 0.5})
    steered = logits(engine)
    with engine.unsteered():
        assert torch.equal(logits(engine), before)
    assert torch.equal(logits(engine), steered)


def test_gauge_reads_the_state_before_the_vector_is_added(engine, axes):
    engine.set_gauges(axes)
    plain = states(engine)
    expected = (float(plain[1, -1] @ axes["a"].unit) - 0.5) / 2.0
    assert engine.readings["a"] == pytest.approx(expected, abs=1e-5)
    engine.set_knobs(axes, {"a": 0.9})
    states(engine)
    assert engine.readings["a"] == pytest.approx(expected, abs=1e-5)
    assert set(engine.readings) == {"a", "b", "c"}


def test_closed_engine_no_longer_steers(engine, axes):
    before = logits(engine)
    engine.set_knobs(axes, {"a": 0.5})
    engine.close()
    assert torch.equal(logits(engine), before)


def test_stream_is_repeatable_and_a_knob_turned_mid_stream_changes_only_later_tokens(engine):
    """Steer the last block toward token 33's own embedding: from the next token on, 33 must win."""
    toward_33 = engine.model.transformer.wte.weight[33].detach()
    axes = {"to33": Axis("to33", "low", "high", layer=3, unit=toward_33 / toward_33.norm(), scale=1.0)}

    def run(turn_at=None):
        engine.set_knobs(axes, {})
        pieces = []
        for index, (piece, _readings) in enumerate(engine.stream("5 9 12", max_new_tokens=10, temperature=0.0)):
            pieces.append(piece.strip())
            if index + 1 == turn_at:
                engine.set_knobs(axes, {"to33": 200.0})
        return pieces

    plain = run()
    assert plain == run()
    assert "33" not in plain
    turned = run(turn_at=4)
    assert turned[:4] == plain[:4]
    assert turned[4:] == ["33"] * 6


def test_stream_stops_when_asked(engine):
    stop = threading.Event()
    pieces = []
    for piece, _readings in engine.stream("5 9 12", max_new_tokens=20, temperature=0.0, stop=stop):
        pieces.append(piece)
        if len(pieces) == 3:
            stop.set()
    assert len(pieces) == 3


def test_sample_token_greedy_nucleus_and_seed():
    scores = torch.tensor([0.1, 3.0, 2.9, -1.0])
    generator = torch.Generator().manual_seed(0)
    assert sample_token(scores, 0.0, 0.95, generator) == 1
    assert {sample_token(scores, 1.0, 0.01, generator) for _ in range(20)} == {1}
    draws = {sample_token(scores, 1.0, 0.9, generator) for _ in range(200)}
    assert draws == {1, 2}
    first = [sample_token(scores, 1.0, 1.0, torch.Generator().manual_seed(3)) for _ in range(5)]
    assert first == [sample_token(scores, 1.0, 1.0, torch.Generator().manual_seed(3)) for _ in range(5)]
