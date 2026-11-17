"""A stand-in for GLM-5.3 on one card: its registered config cut in layers, routed experts and attention heads, served
from vLLM's random weights (``load_format="dummy"``), with everything else of the config kept: the hidden size, the
vocabulary, the indexer's keys and heads, the cache layout, the MTP layer, the FP8 blocks."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

__all__ = ["COPIED", "EXPERTS", "GLM53", "LAYERS", "cut", "stand_in"]

GLM53 = Path(__file__).resolve().parent / "hf_configs" / "GLM53.config.json"
#: decoder layers (GLM-5.3's three dense layers, then MoE layers with a shared and an own index) and routed experts
LAYERS, EXPERTS = 7, 16
#: a checkpoint's files the stand-in's directory copies: its tokenizer, chat template and generation config, never its
#: weights or the index of them
COPIED = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json")


def cut(cfg: dict[str, Any], layers: int = LAYERS, experts: int = EXPERTS, heads: int | None = None) -> dict[str, Any]:
    """``cfg`` cut to its first ``layers`` decoder layers, ``experts`` routed experts and, where given, ``heads``
    attention heads. Every list of one entry a layer is cut with them, and the MTP layer, which vLLM names after the
    last decoder layer, keeps the real one's modules left out of FP8."""
    n = cfg["num_hidden_layers"]
    if not 0 < layers <= n or experts < cfg["num_experts_per_tok"]:
        raise ValueError(f"stand-in: {layers} of {n} layers and {experts} experts, at least {cfg['num_experts_per_tok']} "
                         "(the experts a token routes to)")
    out = {**cfg, "num_hidden_layers": layers, "n_routed_experts": experts,
           **({"num_attention_heads": heads, "num_key_value_heads": heads} if heads else {})}
    for k, v in cfg.items():
        if isinstance(v, list) and len(v) == n:
            out[k] = v[:layers]
    q = cfg.get("quantization_config")
    if q and "modules_to_not_convert" in q:
        kept = []
        for m in q["modules_to_not_convert"]:
            i = re.match(r"model\.layers\.(\d+)\.", m)
            if i is None or int(i[1]) < layers:
                kept.append(m)
            elif int(i[1]) >= n:
                kept.append(m.replace(f"model.layers.{i[1]}.", f"model.layers.{int(i[1]) - n + layers}.", 1))
        out["quantization_config"] = {**q, "modules_to_not_convert": kept}
    return out


def stand_in(where: Path, layers: int = LAYERS, experts: int = EXPERTS, heads: int | None = None,
             tokenizer_from: Path | None = None) -> dict[str, Any]:
    """Writes the stand-in's ``config.json`` into ``where`` and, from the checkpoint ``tokenizer_from``, its `COPIED`
    files (refused where it has no tokenizer); returns the config."""
    cfg = cut(json.loads(GLM53.read_text()), layers, experts, heads)
    where.mkdir(parents=True, exist_ok=True)
    (where / "config.json").write_text(json.dumps(cfg, indent=1) + "\n")
    if tokenizer_from is not None:
        if not (tokenizer_from / "tokenizer.json").is_file():
            raise FileNotFoundError(f"stand-in: {tokenizer_from} holds no tokenizer.json")
        for name in COPIED:
            if (tokenizer_from / name).is_file():
                shutil.copyfile(tokenizer_from / name, where / name)
    return cfg
