"""Run a causal language model with steering vectors added to its residual stream.

A steering vector is a direction in the model's hidden state. Adding it to the output of one
transformer block shifts everything the later blocks compute, so the text changes without any
change to the prompt. The engine keeps one forward hook per block; a hook does nothing until a
knob or a gauge is set for its block.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch

STORY_SYSTEM = "You are a novelist. Continue the story from exactly where it stops. Write only the continuation."


@dataclass
class Axis:
    """One steering direction with what is needed to apply and read it."""

    name: str
    low: str
    high: str
    layer: int
    unit: torch.Tensor  # unit vector pointing from the low pole to the high pole
    scale: float  # mean residual norm at `layer`; a knob value is a fraction of it
    gauge_mid: float = 0.0  # projection halfway between the two poles
    gauge_half: float = 1.0  # half the distance between the poles' mean projections


class Engine:
    def __init__(self, model, tokenizer):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.layers = _find_layers(model)
        self.width = model.config.hidden_size
        self.readings: dict[str, float] = {}
        self._delta: dict[int, torch.Tensor] = {}
        self._gauges: dict[int, list[Axis]] = {}
        self._hooks = [layer.register_forward_hook(self._hook_for(index)) for index, layer in enumerate(self.layers)]
        self.lock = threading.Lock()

    @classmethod
    def load(cls, folder: str | Path, threads: int | None = None) -> "Engine":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if threads:
            torch.set_num_threads(threads)
        tokenizer = AutoTokenizer.from_pretrained(str(folder))
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(str(folder)).float()
        return cls(model, tokenizer)

    def close(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def set_knobs(self, axes: dict[str, Axis], values: dict[str, float]) -> None:
        """Set the vector each block adds: the sum over axes of value * scale * unit.

        The dictionary is replaced in one assignment, so a generation running in another thread
        sees either the old knobs or the new ones, from its next token on.
        """
        delta: dict[int, torch.Tensor] = {}
        for name, value in values.items():
            if not value:
                continue
            axis = axes[name]
            step = float(value) * axis.scale * axis.unit
            delta[axis.layer] = delta[axis.layer] + step if axis.layer in delta else step
        self._delta = delta

    def set_gauges(self, axes: dict[str, Axis]) -> None:
        """Read each axis at its block on every forward pass, before anything is added there."""
        gauges: dict[int, list[Axis]] = {}
        for axis in axes.values():
            gauges.setdefault(axis.layer, []).append(axis)
        self._gauges = gauges
        self.readings = {}

    def _hook_for(self, index: int):
        def hook(_module, _inputs, output):
            delta = self._delta.get(index)
            gauges = self._gauges.get(index)
            if delta is None and not gauges:
                return None
            hidden = output[0] if isinstance(output, tuple) else output
            if gauges:
                last = hidden[:, -1, :]
                for axis in gauges:
                    projection = (last @ axis.unit).mean().item()
                    self.readings[axis.name] = (projection - axis.gauge_mid) / axis.gauge_half
            if delta is None:
                return None
            steered = hidden + delta
            return (steered, *output[1:]) if isinstance(output, tuple) else steered

        return hook

    def story_prompt(self, story: str, controls: str = "") -> str:
        """Chat prompt that asks for a continuation. `controls` is extra system text (prompt control)."""
        system = STORY_SYSTEM + (f"\n{controls}" if controls else "")
        messages = [{"role": "system", "content": system}, {"role": "user", "content": story.strip()}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _stop_ids(self) -> set[int]:
        ids = {self.tokenizer.eos_token_id}
        extra = self.model.generation_config.eos_token_id
        ids.update(extra if isinstance(extra, list) else [extra])
        return {token for token in ids if token is not None}

    @torch.inference_mode()
    def stream(
        self,
        prompt: str,
        max_new_tokens: int = 120,
        temperature: float = 0.8,
        top_p: float = 0.95,
        seed: int = 0,
        stop: threading.Event | None = None,
    ) -> Iterator[tuple[str, dict[str, float]]]:
        """Yield (text piece, gauge readings) one token at a time. Knobs may change between tokens."""
        generator = torch.Generator().manual_seed(seed)
        stop_ids = self._stop_ids()
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        output = self.model(input_ids=input_ids, use_cache=True)
        token_ids: list[int] = []
        emitted = ""
        for _ in range(max_new_tokens):
            if stop is not None and stop.is_set():
                return
            token = sample_token(output.logits[0, -1], temperature, top_p, generator)
            if token in stop_ids:
                return
            token_ids.append(token)
            text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            if not text.endswith("\ufffd"):
                yield text[len(emitted) :], dict(self.readings)
                emitted = text
            output = self.model(
                input_ids=torch.tensor([[token]]), past_key_values=output.past_key_values, use_cache=True
            )

    @torch.inference_mode()
    def generate_batch(
        self, prompts: list[str], max_new_tokens: int, temperature: float, top_p: float, seed: int
    ) -> list[str]:
        """Sample one continuation per prompt in a single batch, with the current knobs."""
        torch.manual_seed(seed)
        batch = self.tokenizer(prompts, return_tensors="pt", padding=True)
        output = self.model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=0,
            repetition_penalty=1.0,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        new_tokens = output[:, batch.input_ids.shape[1] :]
        return self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    @torch.inference_mode()
    def hidden_states(self, text: str) -> torch.Tensor:
        """Residual stream after every block for `text`: [blocks, tokens, width]."""
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids
        output = self.model(input_ids=input_ids, output_hidden_states=True)
        return torch.stack(output.hidden_states[1:])[:, 0]

    @torch.inference_mode()
    def continuation_nll(self, prompt: str, continuation: str) -> float:
        """Mean negative log-likelihood (nats per token) of `continuation` after `prompt`."""
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        continuation_ids = self.tokenizer(continuation, return_tensors="pt", add_special_tokens=False).input_ids
        if continuation_ids.shape[1] == 0:
            return float("nan")
        input_ids = torch.cat([prompt_ids, continuation_ids], dim=1)
        logits = self.model(input_ids=input_ids).logits[0, prompt_ids.shape[1] - 1 : -1]
        return torch.nn.functional.cross_entropy(logits, continuation_ids[0]).item()

    def unsteered(self) -> "_Unsteered":
        """Context manager: run with all knobs at zero, then put the knobs back."""
        return _Unsteered(self)


class _Unsteered:
    def __init__(self, engine: Engine):
        self.engine = engine

    def __enter__(self):
        self.saved = self.engine._delta
        self.engine._delta = {}

    def __exit__(self, *_error):
        self.engine._delta = self.saved


def sample_token(logits: torch.Tensor, temperature: float, top_p: float, generator: torch.Generator) -> int:
    """Nucleus sampling. Temperature 0 picks the most likely token."""
    if temperature <= 0:
        return int(logits.argmax())
    probabilities = torch.softmax(logits / temperature, dim=-1)
    ordered, order = probabilities.sort(descending=True)
    mass_before = ordered.cumsum(0) - ordered
    ordered = torch.where(mass_before < top_p, ordered, torch.zeros_like(ordered))
    choice = torch.multinomial(ordered / ordered.sum(), 1, generator=generator)
    return int(order[choice])


def _find_layers(model):
    for path in ("model.layers", "transformer.h", "gpt_neox.layers"):
        target = model
        try:
            for part in path.split("."):
                target = getattr(target, part)
        except AttributeError:
            continue
        return list(target)
    raise ValueError("this model's transformer blocks were not found (tried model.layers, transformer.h)")
