import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from kemudi.engine import Axis, Engine


class WordTokenizer:
    """Stand-in tokenizer: every word is the number it spells."""

    eos_token_id = 0
    pad_token_id = 0

    def __call__(self, text, return_tensors=None, **_ignored):
        ids = [int(word) for word in text.split()]
        return type("Encoded", (), {"input_ids": torch.tensor([ids])})()

    def decode(self, ids, **_ignored):
        return " ".join(str(token) for token in ids)


@pytest.fixture()
def engine():
    torch.manual_seed(0)
    config = GPT2Config(vocab_size=60, n_positions=64, n_embd=16, n_layer=4, n_head=2, eos_token_id=0, bos_token_id=0)
    model = GPT2LMHeadModel(config)
    return Engine(model, WordTokenizer())


@pytest.fixture()
def axes():
    generator = torch.Generator().manual_seed(1)

    def unit():
        vector = torch.randn(16, generator=generator)
        return vector / vector.norm()

    return {
        "a": Axis("a", "low", "high", layer=1, unit=unit(), scale=5.0, gauge_mid=0.5, gauge_half=2.0),
        "b": Axis("b", "low", "high", layer=1, unit=unit(), scale=3.0),
        "c": Axis("c", "low", "high", layer=2, unit=unit(), scale=4.0),
    }
