"""Find one steering vector per axis from contrasting prompts, and check it on text it never saw.

Method (difference of means): the same story opening is written under two opposite style
instructions. The vector is the mean difference between the two residual streams, taken at one
block. The block is chosen on half of a set of hand-written labelled sentences and the reported
separation comes from the other half. Many blocks separate the sentences equally well, so among
those the block is taken where the two poles lie furthest apart relative to the size of the
hidden state: there a given dose is the largest real change of style.
"""

from __future__ import annotations

import torch

from .engine import Engine
from .stats import auc

BAND = (0.2, 0.6)  # candidate blocks, as a fraction of the model's depth
AUC_TIE = 0.02  # blocks this close to the best dev AUC count as equally good


def pole_prompt(engine: Engine, style: str, opener: str) -> tuple[str, int]:
    """Prompt whose reply already starts with `opener`. Also returns how many tokens precede the opener."""
    messages = [
        {"role": "system", "content": f"You are a novelist. Write the story so that it is {style}."},
        {"role": "user", "content": "Continue the story."},
    ]
    head = engine.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return head + opener, len(engine.tokenizer(head).input_ids)


def opener_state(engine: Engine, style: str, opener: str) -> torch.Tensor:
    """Mean residual over the opener's tokens, after every block: [blocks, width]."""
    prompt, head_tokens = pole_prompt(engine, style, opener)
    return engine.hidden_states(prompt)[:, head_tokens:].mean(dim=1)


def sentence_state(engine: Engine, sentence: str) -> torch.Tensor:
    """Mean residual over a plain sentence, skipping its first token: [blocks, width]."""
    return engine.hidden_states(sentence)[:, 1:].mean(dim=1)


def candidate_layers(blocks: int) -> list[int]:
    return list(range(int(BAND[0] * blocks), int(BAND[1] * blocks) + 1))


def choose_layer(dev_auc: dict[int, float], separation: dict[int, float]) -> int:
    """Among the blocks within AUC_TIE of the best dev AUC, the one with the largest pole separation."""
    best = max(dev_auc.values())
    close = [layer for layer, value in dev_auc.items() if value >= best - AUC_TIE]
    return max(close, key=lambda layer: (separation[layer], -layer))


def extract(engine: Engine, data: dict, model_name: str, log=print) -> tuple[dict, dict[str, torch.Tensor]]:
    openers = data["extraction_openers"]
    layers = candidate_layers(len(engine.layers))
    summary = {"model": model_name, "blocks": len(engine.layers), "width": engine.width, "axes": {}}
    units: dict[str, torch.Tensor] = {}

    for name, axis in data["axes"].items():
        low = torch.stack([opener_state(engine, style, opener) for style in axis["low_styles"] for opener in openers])
        high = torch.stack([opener_state(engine, style, opener) for style in axis["high_styles"] for opener in openers])
        difference = (high - low).mean(dim=0)
        directions = difference / difference.norm(dim=-1, keepdim=True)
        norms = torch.cat([low, high]).norm(dim=-1).mean(dim=0)
        pole_distance = difference.norm(dim=-1) / norms  # as a fraction of the hidden-state size

        low_examples = torch.stack([sentence_state(engine, text) for text in axis["low_examples"]])
        high_examples = torch.stack([sentence_state(engine, text) for text in axis["high_examples"]])

        def separation(layer: int, split: slice) -> float:
            unit = directions[layer]
            return auc((low_examples[split, layer] @ unit).tolist(), (high_examples[split, layer] @ unit).tolist())

        dev, test = slice(0, None, 2), slice(1, None, 2)
        dev_auc = {layer: separation(layer, dev) for layer in layers}
        layer = choose_layer(dev_auc, {candidate: float(pole_distance[candidate]) for candidate in layers})
        unit = directions[layer]
        low_mean = float((low[:, layer] @ unit).mean())
        high_mean = float((high[:, layer] @ unit).mean())

        units[name] = unit
        summary["axes"][name] = {
            "low": axis["low"],
            "high": axis["high"],
            "layer": layer,
            "scale": float(norms[layer]),
            "pole_separation": round(float(pole_distance[layer]), 4),
            "gauge_mid": (low_mean + high_mean) / 2,
            "gauge_half": (high_mean - low_mean) / 2,
            "pairs": len(low),
            "dev_auc_by_layer": {str(candidate): round(value, 3) for candidate, value in dev_auc.items()},
            "dev_auc": round(dev_auc[layer], 3),
            "test_auc": round(separation(layer, test), 3),
        }
        log(f"{name}: block {layer}, dev AUC {dev_auc[layer]:.2f}, test AUC {summary['axes'][name]['test_auc']:.2f}")
    return summary, units
