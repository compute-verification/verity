"""The observer-side boundary values of the families the Llama-shaped rules of `noninterference.observer_boundary_hashes` leave
empty: unfused norms (Gemma-2) and the parallel residual (GPT-NeoX)."""
from __future__ import annotations

import re

#: the final norm under its family's vLLM name: Llama-style `model.norm`, GPT-NeoX `gpt_neox.final_layer_norm` (Pythia)
FINAL_NORM_MODULES = ("model.norm", "gpt_neox.final_layer_norm")
_PRE_FF_RE = re.compile(r"^model\.layers\.(\d+)\.pre_feedforward_layernorm$")
_POST_FF_RE = re.compile(r"^model\.layers\.(\d+)\.post_feedforward_layernorm$")


def unfused_norm_boundaries(se, by_key: dict, s: int, rec: dict, problems: list[str]) -> None:
    """[PROPOSED GEN dense2] the (a)-side boundaries of a family whose norms run unfused (Gemma-2: `GemmaRMSNorm.forward_native`,
    no `fused_add_rms_norm` op, so the Llama-shaped rules above leave `residual` / `final` empty and every comparison fails by
    construction).  What the (b) hooks hash: `model.layers.L` returns `(post_feedforward_layernorm(x), residual)` where
    `residual` is the bf16 copy of the fp32 sum formed inside `pre_feedforward_layernorm`; `model.norm` returns its normalised
    output; `logits_processor` returns the softcapped logits.  So
        hidden[L]   := outs of the LAST dispatch in module model.layers.L.post_feedforward_layernorm   (the bf16 cast)
        residual[L] := outs of the aten.to.dtype whose input is the two-tensor aten.add.Tensor output in ...pre_feedforward_layernorm
        final       := outs of the last dispatch in module model.norm
        logits      := outs of the last dispatch in module logits_processor (replaces the raw lm_head GEMM output above)."""
    from verity_vllm.observe.events import DispatchEnter

    last: dict[str, object] = {}
    adds: dict[int, int] = {}                 # layer -> alloc of the fp32 residual sum
    for node in se.all_nodes():
        ev = node.enter
        if not isinstance(ev, DispatchEnter):
            continue
        mod = ev.module or ""
        if _POST_FF_RE.match(mod) or mod in ("model.norm", "logits_processor"):
            last[mod] = node
        m = _PRE_FF_RE.match(mod)
        if m:
            L = int(m.group(1))
            outs = node.exit.outs if node.exit is not None else None
            if ev.op == "aten.add.Tensor" and sum(hasattr(a, "alloc") for a in ev.args) == 2 and hasattr(outs, "alloc"):
                adds[L] = outs.alloc
            elif ev.op == "aten.to.dtype" and ev.args and getattr(ev.args[0], "alloc", None) == adds.get(L) and L not in rec["residual"]:
                sha = by_key.get((s, ev.id, "outs"))
                if sha is None:
                    problems.append(f"step {s}: no snapshot for the residual cast id {ev.id} outs (layer {L})")
                else:
                    rec["residual"][L] = sha
    if not last:
        return
    for mod, node in last.items():
        sha = by_key.get((s, node.id, "outs"))
        if sha is None:
            problems.append(f"step {s}: no snapshot for {mod} id {node.id} outs")
            continue
        m = _POST_FF_RE.match(mod)
        if m:
            rec["hidden"][int(m.group(1))] = sha       # replaces the down_proj GEMM output recorded above
        elif mod == "model.norm":
            rec["final"] = sha
        else:
            rec["logits"] = sha


_NEOX_LAYER_RE = re.compile(r"^gpt_neox\.layers\.(\d+)$")


def parallel_residual_boundaries(se, by_key: dict, s: int, rec: dict, problems: list[str]) -> None:
    """The (a)-side boundaries of vLLM's GPT-NeoX family (Pythia): `nn.LayerNorm` runs as
    `aten.layer_norm` (no fused residual op), the MLP's down projection is `mlp.dense_4h_to_h` (no `down_proj` launch) and
    `GPTNeoXLayer.forward` returns ONE tensor, the parallel residual `mlp(ln2(x)) + attn(ln1(x)) + x` formed by two
    `aten.add.Tensor` calls in the layer module itself -- so the Llama and Gemma-2 rules above leave every boundary empty.
    What the (b) hooks hash: `gpt_neox.layers.L` output[0] (that sum), `gpt_neox.final_layer_norm` output[0], the logits.  So
        hidden[L]   := outs of the LAST aten.add.Tensor dispatched in module gpt_neox.layers.L itself (the second residual add)
        residual[L] := (none: the layer returns no separate residual; both sides leave it empty)
        final       := outs of the last dispatch in module gpt_neox.final_layer_norm (the aten.layer_norm output)
        logits      := the lm_head GEMM output recorded above (Pythia's `logits_processor` applies no softcap)."""
    from verity_vllm.observe.events import DispatchEnter

    last_add: dict[int, object] = {}
    final_node = None
    for node in se.all_nodes():
        ev = node.enter
        if not isinstance(ev, DispatchEnter):
            continue
        mod = ev.module or ""
        m = _NEOX_LAYER_RE.match(mod)
        if m and ev.op.startswith("aten.add.Tensor"):
            last_add[int(m.group(1))] = node
        elif mod == "gpt_neox.final_layer_norm":
            final_node = node
    if not last_add and final_node is None:
        return
    for L, node in sorted(last_add.items()):
        sha = by_key.get((s, node.id, "outs")) or by_key.get((s, node.id, "outs[0]"))
        if sha is None:
            problems.append(f"step {s}: no snapshot for the layer-{L} residual add id {node.id} outs")
        else:
            rec["hidden"][L] = sha
    if final_node is not None:
        sha = by_key.get((s, final_node.id, "outs")) or by_key.get((s, final_node.id, "outs[0]"))
        if sha is None:
            problems.append(f"step {s}: no snapshot for gpt_neox.final_layer_norm id {final_node.id} outs")
        rec["final"] = sha
