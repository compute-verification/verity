"""verity_vllm/check/root_policy.py — PROTECTED (gate G3, root closure).  Frozen root classes, each with an independent validator.

    uv run verity-vllm root-policy --log LOG --snapshots DIR --profile NAME [--safetensors GLOB] [--cos-sin NPY] --out root_policy.json

A Program operand is either produced by another Program instance or is a *root*.  The fold already refuses reads with no
producer (`Unresolved`); this module decides whether the roots the fold *did* accept are legitimate, using a source of
authority that is independent of the run for each class:

| class            | roots                                                      | validator                                              |
|------------------|------------------------------------------------------------|--------------------------------------------------------|
| weight           | `named_parameters()` mapped to a `weights` Record field    | checkpoint bytes: the snapshot of the root's storage == the checkpoint's words (`WeightSource`); roots the observer did not snapshot (over `snapshot-max-bytes`) fall back to *checkpoint identity*: the header's frozen `weights_sha256` verified at engine build (`--verify-weights`) |
| analytic-table   | the rotary `cos_sin_cache` buffer; Gemma-2's `model.normalizer` (0-d bf16 `sqrt(H)` buffer) | recomputed from `C` (`analytic.check_cos_sin`; `analytic.check_normalizer`, exact) |
| alias            | another name for a root's storage (layer-k `cos_sin_cache`, tied `lm_head.weight`, MLA's absorbed `W_UK_T` / `W_UV`) | identical allocation + view in the observation log (storage identity proven by the runtime, never assumed from the family); for `W_UK_T` / `W_UV` the view of a validated `kv_b_proj.weight` root that C's MLA dimensions give |
| request-input    | prompt token rows                                          | the request record (`RunHeader.workload`)              |
| config           | the static configuration `C` behind every specialization   | the checkpoint's `config.json` subset in the header (`analytic.config_of`), field by field; an MLA configuration's fields (`MLA_CONFIG_KEYS`) each need their key recorded |
| ignored          | placeholders never read by a modeled kernel (`*_scale` with `kv_cache_dtype=auto`) | every read of their storage happens in a frozen set of non-semantic argument slots (`IGNORED_ROOT_CONSUMERS`, or `CONDITIONAL_IGNORED_SLOTS` when the same event's arguments select the branch that never reads the slot) |

Anything else — an unknown root name, a `Root.role` outside the table, an `Opaque`/prescribed fallback — is a **new root
class** and fails.  A lane cannot make replay pass by declaring an inconvenient tensor an input: that would show up here as
`new_root_classes` or as a weight root whose bytes match no checkpoint field.

This module and the tables in it change only through an approved decision in `out/capture/decisions.md` (see
`out/gen/RULES.md`, protected paths).  Torch-free.
"""
from __future__ import annotations

import glob
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from dataclasses import dataclass

from verity_vllm.config import option
from verity_vllm.program.frontend.rules.family import _field_intervals

from verity_vllm.program import model as analytic
from verity_vllm.observe.events import DispatchEnter, Root, RunHeader, TritonLaunch, ViewDesc
from verity_vllm.observe.log import read_log

#: Frozen.  Root roles the observer may emit and the class each one belongs to (before name-based refinement).
FROZEN_ROLE_CLASSES = {"weight": "weight", "buffer": "buffer", "prompt_token": "request-input", "position": "request-input"}
#: Frozen.  Dispatcher ops that may receive an `ignored` root as an argument (non-semantic slots under κ: the bf16 KV cache
#: path ignores the k/v scale placeholders; `aten.expand` only builds the view that is passed to it).
IGNORED_ROOT_CONSUMERS = frozenset({"_C_cache_ops.reshape_and_cache_flash.default", "aten.expand.default",
                                    "vllm.unified_attention_with_output.default"})
#: Frozen.  Analytic-table roots and the checker for each.  [ADAPTER lane cov-dense 2026-09-17] `normalizer`: Gemma-2's
#: `model.normalizer` buffer (`torch.tensor(hidden_size ** 0.5)` at model dtype, no checkpoint tensor) is one bf16 word that is
#: a function of `C["H"]` alone — the same class as the rotary table, validated exactly against the captured snapshot word.
ANALYTIC_TABLES = {"cos_sin": "analytic.check_cos_sin", "normalizer": "analytic.check_normalizer"}


def analytic_table_of_buffer(name: str) -> str | None:
    """The ANALYTIC_TABLES field a runtime BUFFER (`named_buffers()` path) is the observation of, or None: the one place the two derived
    tables are recognised by name (`m1_capture.dump_cos_sin` / `dump_normalizer` dump them; `verity_vllm.acquire.plan` names them as a
    residual).  vLLM naming facts: the rotary table lives on every layer's `rotary_emb.cos_sin_cache` (one shared table), Gemma-2's
    normalizer on `model.normalizer`."""
    if name.endswith("rotary_emb.cos_sin_cache"):
        return "cos_sin"
    if name == "model.normalizer" or name.endswith(".normalizer"):
        return "normalizer"
    return None
#: Frozen.  Name patterns of the two non-weight root classes (the policy classifies; a profile's own `root_class` is advisory).
#: [SHARED-CODE lane cov-ln] the module prefix (`model` for Llama/Qwen/Gemma/Phi, `gpt_neox` for GPT-NeoX/Pythia) and the attention
#: submodule name (`self_attn` / `attention`) are vLLM naming facts of the family, not of the root class: the two classes below are
#: "another layer's name for the shared rotary table" and "the attention layer's KV/Q scale placeholders", whatever the family
#: calls its modules.  Before this generalization the patterns were Llama-only (`model.layers.N.self_attn.…`).
_MODULE_PREFIX = r"(?:model|gpt_neox)"
_ATTN_MODULE = r"(?:self_attn|attention)"
ALIAS_OF_COS_SIN = re.compile(rf"^{_MODULE_PREFIX}\.layers\.[1-9]\d*\.{_ATTN_MODULE}\.rotary_emb\.cos_sin_cache$")
#: [ADAPTER lane glm47-flash] multi-head latent attention keeps its kv/q/prob scale placeholders on `MLAAttention`
#: (`self_attn.mla_attn.mla_attn`, vLLM d9105ea80 `set_default_quant_scales`), the same class as the attention layer's.
IGNORED_ROOT_NAMES = (re.compile(rf"^{_MODULE_PREFIX}\.layers\.\d+\.{_ATTN_MODULE}\.attn\.(k_scale|v_scale|q_scale|prob_scale|_k_scale|_v_scale|_q_scale|_prob_scale)$"),
                      re.compile(rf"^{_MODULE_PREFIX}\.layers\.\d+\.{_ATTN_MODULE}\.mla_attn\.mla_attn\.(_k_scale|_v_scale|_q_scale|_prob_scale)$"))
_LAYER0_COS_SIN = re.compile(rf"^{_MODULE_PREFIX}\.layers\.0\.{_ATTN_MODULE}\.rotary_emb\.cos_sin_cache$")
#: Frozen.  [ADAPTER lane glm47-flash] MLA's absorbed up-projections.  vLLM d9105ea80 `MLAAttention.process_weights_after_loading`
#: (model_executor/layers/attention/mla_attention.py:1154-1191, 1248-1252) splits `kv_b_proj.weight.T.view(KV_LORA, heads, QK_NOPE +
#: V_HEAD)` into `[QK_NOPE, V_HEAD]` and registers `W_UK_T = W_UK.permute(1, 2, 0)`, `W_UV = W_UV.transpose(0, 1)`: views of the layer's
#: weight of record, so their words are its words.  `replace_parameter(..., prefer_copy=True)` may copy instead, so the class is
#: proven per root from the log (same allocation), never assumed.
MLA_KV_B_VIEW = re.compile(rf"^(?P<layer>{_MODULE_PREFIX}\.layers\.\d+\.{_ATTN_MODULE})\.mla_attn\.mla_attn\.(?P<view>W_UK_T|W_UV)$")


@dataclass(frozen=True)
class IgnoredSlots:
    """The argument slots of one consumer that may hold an `ignored` root, when the event is the code `pin` names (a dispatcher op's
    schema, a Triton kernel's source sha256) and every `selectors` argument of the same event takes a value in `admitted` (a tensor
    argument stands for its dtype)."""
    slots: frozenset
    selectors: tuple
    admitted: frozenset
    pin: str

    def admits(self, e: DispatchEnter | TritonLaunch, reads: list[tuple]) -> bool:
        pin = e.schema if isinstance(e, DispatchEnter) else e.src_sha
        return pin == self.pin and set(reads) <= self.slots and all(_arg_value(e, p) in self.admitted for p in self.selectors)


#: Frozen.  [ADAPTER lane glm47-flash] MLA's bf16 KV path (vLLM d9105ea80): consumers that receive the `_k_scale` placeholder in a
#: slot their code reads only for an fp8 KV cache.
#:  * `concat_and_cache_mla`: `kv_cache_dtype` "auto" dispatches `kAuto` (csrc/libtorch_stable/cache_kernels.cu:971), whose copy is
#:    `dst = src`; `*scale` is read only in the fp8 branch (:443-448).
#:  * `unified_mla_kv_cache_update` only forwards `k_scale` to that cache update (model_executor/layers/attention/mla_attention.py:1335-1367).
#:  * `_fwd_grouped_kernel_stage1` loads `k_scale` / `v_scale` and uses them only under `k.dtype.is_fp8()`, `kpe.dtype.is_fp8()`,
#:    `v.dtype.is_fp8()` of values loaded from `K_Buffer` / `V_Buffer` (v1/attention/ops/triton_decode_attention.py:367-427): dead
#:    for bf16 / f16 buffers.
CONDITIONAL_IGNORED_SLOTS = {
    "_C_cache_ops.concat_and_cache_mla.default": IgnoredSlots(
        slots=frozenset({("args", 5)}), selectors=(("args", 4),), admitted=frozenset({"auto"}),
        pin="_C_cache_ops::concat_and_cache_mla(Tensor kv_c, Tensor k_pe, Tensor($0! -> ) kv_cache, Tensor slot_mapping, str kv_cache_dtype, Tensor scale) -> ()"),
    "vllm.unified_mla_kv_cache_update.default": IgnoredSlots(
        slots=frozenset({("args", 4)}), selectors=(("args", 3),), admitted=frozenset({"auto"}),
        pin="vllm::unified_mla_kv_cache_update(Tensor kv_c_normed, Tensor k_pe, PyObject layer_name, str kv_cache_dtype, Tensor k_scale) -> Tensor"),
    "triton:vllm.v1.attention.ops.triton_decode_attention._fwd_grouped_kernel_stage1": IgnoredSlots(
        slots=frozenset({("args", "k_scale"), ("args", "v_scale")}), selectors=(("args", "K_Buffer"), ("args", "V_Buffer")),
        admitted=frozenset({"bfloat16", "float16"}), pin="fdd0368d448e87105168c26088c95ad46b4c08c45149db273a9451111936f9b4"),
}
#: Frozen.  [ADAPTER lane glm47-flash] The config.json keys of an MLA configuration's (C carries `KV_LORA`) attention and MoE fields,
#: the first key recorded being compared.  Every field C carries needs its key recorded in the header: a field no recorded key states
#: is not validated, so it fails.
MLA_CONFIG_KEYS = {
    "Q_LORA": ("q_lora_rank",), "KV_LORA": ("kv_lora_rank",), "QK_NOPE": ("qk_nope_head_dim",), "QK_ROPE": ("qk_rope_head_dim",),
    "V_HEAD": ("v_head_dim",), "FIRST_DENSE": ("first_k_dense_replace",), "I_DENSE": ("intermediate_size",),
    "MOE_I_FULL": ("moe_intermediate_size",), "NUM_EXPERTS": ("num_experts", "num_local_experts", "n_routed_experts"),
    "TOPK": ("num_experts_per_tok",), "NORM_TOPK_PROB": ("norm_topk_prob",), "RENORMALIZE": ("norm_topk_prob",), "N_GROUP": ("n_group",),
    "TOPK_GROUP": ("topk_group",), "ROUTED_SCALING": ("routed_scaling_factor",), "SHARED_EXPERTS": ("n_shared_experts",),
}


def frozen_root_class(name: str, weight_field) -> str:
    if weight_field(name) is not None:
        return "weight"
    if ALIAS_OF_COS_SIN.match(name):
        return "alias-of-cos_sin"
    if MLA_KV_B_VIEW.match(name):
        return "alias-of-kv_b"
    if any(r.match(name) for r in IGNORED_ROOT_NAMES):
        return "ignored"
    return "unknown"


def _slotted(x: Any, path: tuple):
    """(argument path, view) for every tensor view in a recorded argument value."""
    if isinstance(x, ViewDesc):
        yield path, x
    elif isinstance(x, (list, tuple)):
        for i, y in enumerate(x):
            yield from _slotted(y, path + (i,))
    elif isinstance(x, dict):
        for k, y in x.items():
            yield from _slotted(y, path + (k,))


def _arg_value(e: DispatchEnter | TritonLaunch, path: tuple) -> Any:
    """The recorded scalar at an argument path of an event, a tensor's dtype, or None."""
    x: Any = {"args": e.args, "kwargs": getattr(e, "kwargs", None) or {}}
    for p in path:
        try:
            x = x[p]
        except (KeyError, IndexError, TypeError):
            return None
    return x.dtype if isinstance(x, ViewDesc) else (x if isinstance(x, (str, int, float, bool)) else None)


def _field_name(path: tuple) -> str:
    """("layers", 3, "qkv_w") -> "layers[3].qkv_w"; ("embed",) -> "embed" (the `_field_intervals` naming)."""
    out = ""
    for p in path:
        out = f"{out}[{p}]" if isinstance(p, int) else (f"{out}.{p}" if out else p)
    return out


def evaluate(log_path: str | Path, snapshots: str | Path | None, profile, *, safetensors: list[Path] | None = None,
             cos_sin: Path | None = None, weight_bytes: bool = True) -> dict[str, Any]:
    from verity_vllm.check.fold_compare import SnapshotStore, WeightSource, _extent
    from verity_vllm.observe.fold.fold import config_for_weights

    t0 = time.time()
    events = list(log_path) if isinstance(log_path, list) else list(read_log(log_path))
    header = next((e for e in events if isinstance(e, RunHeader)), None)
    roots = [e for e in events if isinstance(e, Root)]
    C = config_for_weights(dict(profile.config or {}))
    weights_type = analytic.weights_type(C)
    intervals = {name: (lo, hi) for lo, hi, name in _field_intervals(weights_type, "")}
    weight_field = getattr(profile, "weight_field", None)
    if weight_field is None:
        raise SystemExit("profile has no weight_field (root policy needs the name -> Record field map)")
    snaps_dir = Path(snapshots) if snapshots else None
    store = SnapshotStore(snaps_dir) if snaps_dir and (snaps_dir / "snapshots.json").exists() else None
    repo = (profile.config or {}).get("model")
    cos_sin_path = cos_sin or (snaps_dir / "cos_sin_cache.npy" if snaps_dir and (snaps_dir / "cos_sin_cache.npy").exists() else None)
    ws = WeightSource(weights_type, safetensors, cos_sin_path, repo=repo, config=C) if weight_bytes else None   # [SHARED-CODE lane cov-ln]
    have_ckpt = bool(ws and ws.tensors)

    out: dict[str, Any] = {"schema": "verity-gen/root-policy/v1", "status": "pass", "log": str(log_path if not isinstance(log_path, list) else "<events>"), "profile": profile.id,
                           "classes": {}, "roots": [], "new_root_classes": [], "alias_claims": [], "problems": []}
    classes: dict[str, Counter] = defaultdict(Counter)
    by_name: dict[str, Root] = {}
    for r in roots:
        by_name.setdefault(r.name, r)

    # --- weight identity (header manifest sha, verified at build?) ------------------------------------------------------
    ident = {"weights_sha256": None, "verified_at_build": None, "repo": None, "revision": None}
    if header is not None:
        ident["weights_sha256"] = ((header.profile or {}).get("model") or {}).get("weights_sha256")
        ident["verified_at_build"] = (header.notes or {}).get("weights_verified")
        ident["repo"], ident["revision"] = (header.model or {}).get("repo"), (header.model or {}).get("revision")
        if ident["repo"] and repo and ident["repo"] != repo:
            out["problems"].append(f"header repo {ident['repo']} != profile model {repo}")
    out["weight_identity"] = ident

    # --- classify and validate every root --------------------------------------------------------------------------------
    seen_names: set[str] = set()
    # [SHARED-CODE lane cov-ln] the embedding / LM-head parameters under their family's vLLM names (Llama-style `model.embed_tokens` /
    # `lm_head`; GPT-NeoX `gpt_neox.embed_in` / `embed_out`): the tie check below reads the runtime's statement either way
    embed_root = by_name.get("model.embed_tokens.weight") or by_name.get("gpt_neox.embed_in.weight")
    for r in roots:
        if r.name in seen_names and r.role in ("weight", "buffer"):
            continue                                        # the observer may re-emit parameters per step
        seen_names.add(r.name)
        entry: dict[str, Any] = {"name": r.name, "role": r.role, "alloc": r.view.alloc, "shape": list(r.view.shape), "dtype": r.view.dtype}
        base_class = FROZEN_ROLE_CLASSES.get(r.role)
        if base_class is None:
            entry.update(**{"class": f"role:{r.role}", "validator": None, "status": "fail", "reason": "Root.role outside the frozen table"})
            out["new_root_classes"].append(f"role:{r.role}")
        elif r.role in ("weight", "buffer"):
            path = weight_field(r.name)
            pc = frozen_root_class(r.name, weight_field)
            if path is not None and path[0] == "cos_sin":
                entry["class"], entry["validator"] = "analytic-table", "analytic.check_cos_sin"
                if cos_sin_path is None:
                    entry.update(status="fail", reason="no captured cos_sin_cache.npy to check against the analytic table")
                else:
                    chk = analytic.check_cos_sin(C, cos_sin_path)
                    tab = np.load(cos_sin_path, mmap_mode="r")
                    shape_ok = tuple(r.view.shape) == tuple(tab.shape)
                    entry.update(status="pass" if chk["status"] == "pass" and shape_ok else "fail",
                                 detail={k: chk.get(k) for k in ("max_ulp", "n_diff", "n_outside_bound", "n_elements", "shape_ok")})
                    if not shape_ok:
                        entry["reason"] = f"root view {r.view.shape} vs captured table {tab.shape}"
                    elif chk["status"] != "pass":
                        entry["reason"] = f"captured table outside the analytic bound at {chk.get('n_outside_bound')} elements"
            elif path is not None and path[0] == "normalizer":
                # [ADAPTER lane cov-dense] the one-word analytic table, observed the way `cos_sin_cache.npy` is: the capture dumps
                # the runtime buffer's word to <snapshots>/normalizer.npy (`m1_capture.dump_normalizer`; buffers are not among
                # the activation snapshots), and it must equal bf16(sqrt(H)).  No observation is a fail, not a fallback (there
                # is no checkpoint identity to fall back to).
                entry["class"], entry["validator"] = "analytic-table", ANALYTIC_TABLES["normalizer"]
                npy = (snaps_dir / "normalizer.npy") if snaps_dir is not None else None
                s = store.find(r.view) if store is not None else None
                if npy is not None and npy.exists():
                    captured, source = np.load(npy), "capture/normalizer.npy"
                elif s is not None:
                    captured, source = store.extract(s, r.view), f"snapshot step {s.step}"
                else:
                    captured = source = None
                if captured is None:
                    entry.update(status="fail", reason="no observation of the normalizer buffer (capture/normalizer.npy or a snapshot) to check against the analytic value")
                else:
                    try:
                        chk = analytic.check_normalizer(C, captured)
                        entry.update(status=chk["status"], detail={**{k: chk.get(k) for k in ("H", "expected_word", "expected_value", "captured_words", "n_elements")},
                                                                  "source": source})
                        if chk["status"] != "pass":
                            entry["reason"] = chk.get("reason")
                    except (LookupError, ValueError, KeyError) as e:
                        entry.update(status="fail", reason=f"{type(e).__name__}: {e}")
            elif path is not None:
                entry["class"] = "weight"
                fname = _field_name(path)
                lo_hi = intervals.get(fname)
                if lo_hi is None:
                    entry.update(validator=None, status="fail", reason=f"field {fname} not in the weights Record")
                else:
                    lo, hi = lo_hi
                    if hi - lo != r.view.numel:
                        entry.update(validator=None, status="fail", reason=f"field {fname} has {hi - lo} leaves, root view {r.view.numel} elements")
                    else:
                        s = store.find(r.view) if store is not None else None
                        if s is not None and have_ckpt:
                            try:
                                got = store.extract(s, r.view)
                                exp = ws.words(lo, hi)
                                n_diff = int((got.astype(np.uint32) != exp.astype(np.uint32)).sum())
                                entry.update(validator="checkpoint-bytes", status="pass" if n_diff == 0 else "fail",
                                             detail={"n_elements": int(exp.size), "n_diff": n_diff, "snapshot_step": s.step})
                                if n_diff:
                                    entry["reason"] = f"{n_diff} words differ from checkpoint field {fname}"
                            except (LookupError, ValueError, KeyError) as e:
                                entry.update(validator="checkpoint-bytes", status="fail", reason=f"{type(e).__name__}: {e}")
                        else:
                            ok = bool(ident["weights_sha256"]) and bool(ident["verified_at_build"])
                            entry.update(validator="checkpoint-identity", status="pass" if ok else "fail",
                                         detail={"snapshot": s is not None, "checkpoint_loaded": have_ckpt},
                                         reason=None if ok else "no snapshot of this root and the checkpoint sha was not verified at engine build (--verify-weights)")
            elif pc == "alias-of-cos_sin":
                entry["class"], entry["validator"] = "alias", "storage-identity"
                tgt = next((x for x in roots if _LAYER0_COS_SIN.match(x.name)), None)   # [SHARED-CODE lane cov-ln] family-neutral
                same = tgt is not None and (tgt.view.alloc, tgt.view.offset, tuple(tgt.view.shape)) == (r.view.alloc, r.view.offset, tuple(r.view.shape))
                entry.update(status="pass" if same else "fail", reason=None if same else "claimed alias has different storage than layer-0 cos_sin")
                out["alias_claims"].append({"name": r.name, "alias_of": tgt.name if tgt else None, "proven_by": "alloc+offset+shape", "status": entry["status"]})
            elif pc == "alias-of-kv_b":
                entry.update(**{"class": "alias", "validator": "storage-identity+mla-view", "status": "pending"})   # `_resolve_kv_b_views`
            elif pc == "ignored":
                entry["class"], entry["validator"] = "ignored", "never-read-semantically"
                entry["status"] = "pending"                 # filled by the read scan below
            else:
                entry.update(**{"class": "unknown", "validator": None, "status": "fail", "reason": "no Record field, not an alias, not ignored"})
                out["new_root_classes"].append(f"unknown:{r.name}")
        elif r.role == "prompt_token":
            entry["class"], entry["validator"] = "request-input", "request-record"
            ok, why = _check_prompt_root(r, header)
            entry.update(status="pass" if ok else "fail", reason=None if ok else why)
        elif r.role == "position":
            entry["class"], entry["validator"] = "request-input", "step-context"
            entry["status"] = "pass" if r.values is not None else "fail"
        out["roots"].append(entry)
        classes[entry["class"]][entry["status"]] += 1

    _resolve_kv_b_views(out, by_name, C, classes)
    _scan_ignored(events, out, classes)

    # --- tied / untied lm_head: proven by the runtime, checked against C -----------------------------------------------------
    lm = by_name.get("lm_head.weight") or by_name.get("embed_out.weight")
    tied_claim = bool(C.get("tie_word_embeddings", True))
    if lm is not None and embed_root is not None:
        same = (lm.view.alloc, lm.view.offset, tuple(lm.view.shape)) == (embed_root.view.alloc, embed_root.view.offset, tuple(embed_root.view.shape))
        ok = same == tied_claim
        out["alias_claims"].append({"name": "lm_head.weight", "alias_of": "model.embed_tokens.weight", "proven_by": "alloc+offset+shape",
                                    "runtime_tied": same, "config_tied": tied_claim, "status": "pass" if ok else "fail"})
        if not ok:
            out["problems"].append(f"lm_head tie: runtime says {'tied' if same else 'untied'}, C[tie_word_embeddings]={tied_claim}")
    elif lm is None and embed_root is not None:
        # vLLM dedups a tied lm_head out of named_parameters(): absence of the name is the runtime's statement that it is tied
        ok = tied_claim
        out["alias_claims"].append({"name": "lm_head.weight", "alias_of": "model.embed_tokens.weight", "proven_by": "absent from named_parameters (tie_weights)",
                                    "runtime_tied": True, "config_tied": tied_claim, "status": "pass" if ok else "fail"})
        if not ok:
            out["problems"].append("lm_head tie: runtime has no separate lm_head.weight parameter but C says untied")

    cfg_entry = _config_root(header, C, repo, out)
    out["config_root"] = cfg_entry
    classes["config"][cfg_entry["status"]] += 1

    out["classes"] = {k: {"count": sum(v.values()), "validated": v.get("pass", 0), "failed": v.get("fail", 0),
                          "other": {s: n for s, n in v.items() if s not in ("pass", "fail")}} for k, v in classes.items()}
    failed = [e for e in out["roots"] if e["status"] == "fail"] + ([cfg_entry] if cfg_entry["status"] == "fail" else [])
    out["n_roots"] = len(out["roots"])
    out["n_failed"] = len(failed)
    out["first_failures"] = [{k: e.get(k) for k in ("name", "class", "validator", "reason")} for e in failed[:12]]
    alias_fail = any(a["status"] == "fail" for a in out["alias_claims"])
    out["status"] = "pass" if not failed and not out["new_root_classes"] and not alias_fail and not out["problems"] and cfg_entry["status"] == "pass" else "fail"
    out["seconds"] = round(time.time() - t0, 2)
    return out


def _kv_b_view_mismatch(view: ViewDesc, which: str, weight: ViewDesc, C: dict[str, Any]) -> str | None:
    """None when `view` is MLA's `which` (`W_UK_T` | `W_UV`) of `weight`, the layer's row-major `kv_b_proj.weight`
    [heads·(QK_NOPE + V_HEAD)][KV_LORA], as `MLA_KV_B_VIEW` states vLLM takes it; else why not.  `heads` is the weight's own (a rank's
    under tensor parallelism)."""
    DN, DV, L = (int(C[k]) for k in ("QK_NOPE", "V_HEAD", "KV_LORA"))
    if (view.alloc, view.dtype) != (weight.alloc, weight.dtype):
        return f"allocation/dtype {(view.alloc, view.dtype)} is not kv_b_proj.weight's {(weight.alloc, weight.dtype)}"
    if len(weight.shape) != 2 or weight.shape[1] != L or tuple(weight.strides) != (L, 1) or weight.shape[0] % (DN + DV):
        return f"kv_b_proj.weight {tuple(weight.shape)} strides {tuple(weight.strides)} is not row-major [heads*{DN + DV}][{L}]"
    nh, hs = weight.shape[0] // (DN + DV), (DN + DV) * L
    want = {"W_UK_T": (weight.offset, (nh, DN, L), (hs, L, 1)), "W_UV": (weight.offset + DN * L, (nh, L, DV), (hs, 1, L))}[which]
    got = (view.offset, tuple(view.shape), tuple(view.strides))
    return None if got == want else f"(offset, shape, strides) {got} != {want}, the {which} view of QK_NOPE={DN} V_HEAD={DV} KV_LORA={L}"


def _resolve_kv_b_views(out: dict[str, Any], by_name: dict[str, Root], C: dict[str, Any], classes: dict[str, Counter]) -> None:
    """A pending `alias-of-kv_b` root passes iff its layer's `kv_b_proj.weight` is a weight root that passed its own validator and the
    root is the view of it that `_kv_b_view_mismatch` derives from C."""
    entries = {e["name"]: e for e in out["roots"]}
    pending = [e for e in out["roots"] if e["class"] == "alias" and e["status"] == "pending"]
    for ent in pending:
        m = MLA_KV_B_VIEW.match(ent["name"])
        wname = f"{m['layer']}.kv_b_proj.weight"
        w = entries.get(wname)
        if w is None or w["class"] != "weight" or w["status"] != "pass":
            why = f"{wname} is not a validated weight root"
        else:
            try:
                why = _kv_b_view_mismatch(by_name[ent["name"]].view, m["view"], by_name[wname].view, C)
            except (KeyError, ValueError, TypeError) as e:
                why = f"{type(e).__name__}: {e}"
        ent["status"] = "pass" if why is None else "fail"
        if why:
            ent["reason"] = why
        out["alias_claims"].append({"name": ent["name"], "alias_of": wname, "proven_by": "alloc+dtype+view from C", "status": ent["status"]})
    if pending:
        classes["alias"] = Counter(e["status"] for e in out["roots"] if e["class"] == "alias")


def _scan_ignored(events: list, out: dict[str, Any], classes: dict[str, Counter]) -> None:
    """Every read of an `ignored` root's storage must sit in a frozen non-semantic slot: any slot of an `IGNORED_ROOT_CONSUMERS` op, or a
    `CONDITIONAL_IGNORED_SLOTS` slot of an event that admits it."""
    ignored_allocs = {e["alloc"]: e for e in out["roots"] if e["class"] == "ignored"}
    consumers: Counter = Counter()
    bad_consumers: Counter = Counter()
    if ignored_allocs:
        for e in events:
            if isinstance(e, DispatchEnter):
                name, reads = e.op, [p for p, v in (*_slotted(e.args, ("args",)), *_slotted(e.kwargs, ("kwargs",))) if v.alloc in ignored_allocs]
            elif isinstance(e, TritonLaunch):
                name, reads = f"triton:{e.fn}", [p for p, v in _slotted(e.args, ("args",)) if v.alloc in ignored_allocs]
            else:
                continue
            if reads:
                consumers[name] += len(reads)
                cond = CONDITIONAL_IGNORED_SLOTS.get(name)
                if name not in IGNORED_ROOT_CONSUMERS and not (cond is not None and cond.admits(e, reads)):
                    bad_consumers[name] += len(reads)
        for ent in ignored_allocs.values():
            ent["status"] = "pass" if not bad_consumers else "fail"
            if bad_consumers:
                ent["reason"] = f"read by non-frozen consumers {dict(bad_consumers)}"
        classes["ignored"] = Counter(ent["status"] for ent in ignored_allocs.values())
    out["ignored_root_consumers"] = dict(consumers)


def _config_root(header: RunHeader | None, C: dict[str, Any], repo: str | None, out: dict[str, Any]) -> dict[str, Any]:
    """C's hf-config fields against the checkpoint's config.json subset in the header, field by field (`analytic.config_of`, no family
    rules).  An MLA configuration (C carries `KV_LORA`) reads its head width `D` as `qk_nope_head_dim + qk_rope_head_dim` (vLLM's
    `qk_head_dim`; transformers' MLA configs set `head_dim = qk_rope_head_dim`), its `I` as the routed experts' `moe_intermediate_size`
    when C states that width (`MOE_I_FULL`; `I_DENSE` is the dense layers' `intermediate_size`), and compares every `MLA_CONFIG_KEYS`
    field C carries; any of these whose keys the header does not record fails."""
    cfg_entry: dict[str, Any] = {"class": "config", "validator": "hf-config (header.model.config)", "status": "skipped", "fields": {}}
    hf = (header.model or {}).get("config") if header is not None else None
    if not hf:
        cfg_entry["reason"] = "header carries no HF config subset"
        return cfg_entry
    try:
        derived, prov = analytic.config_of(hf, model=repo)
        fields, unrecorded = ["H", "NL", "NH", "KVH", "D", "I", "V", "EPS", "ROPE_THETA", "MAX_POS", "tie_word_embeddings"], []
        if C.get("KV_LORA"):
            def recorded(*keys: str) -> Any:
                return next((hf[k] for k in keys if hf.get(k) is not None), None)
            nope, rope = recorded("qk_nope_head_dim"), recorded("qk_rope_head_dim")
            mla = {"D": (None if nope is None or rope is None else int(nope) + int(rope), "qk_nope_head_dim + qk_rope_head_dim")}
            if "MOE_I_FULL" in C:
                mla["I"] = (recorded("moe_intermediate_size"), "moe_intermediate_size")
            mla.update({k: (recorded(*keys), " | ".join(keys)) for k, keys in MLA_CONFIG_KEYS.items() if k in C})
            for k, (v, src) in mla.items():
                if v is None:
                    unrecorded.append(f"{k} ({src})")
                    derived.pop(k, None)
                    prov[k] = "unrecorded"
                else:
                    derived[k], prov[k] = v, "hf-config"
            fields += [k for k in MLA_CONFIG_KEYS if k in C]
        mism = {}
        for k in fields:
            if k in derived and k in C and prov.get(k) == "hf-config" and derived[k] != C[k]:
                mism[k] = {"C": C[k], "hf": derived[k]}
            cfg_entry["fields"][k] = prov.get(k, "n/a")
        cfg_entry["status"] = "pass" if not mism and not unrecorded else "fail"
        cfg_entry["mismatches"] = mism
        if mism:
            out["problems"].append(f"C differs from the checkpoint config: {mism}")
        if unrecorded:
            cfg_entry["unrecorded"] = unrecorded
            out["problems"].append(f"the header's config records no key of C's {unrecorded}")
    except Exception as e:  # noqa: BLE001
        cfg_entry.update(status="fail", reason=f"{type(e).__name__}: {e}")
    return cfg_entry


def _check_prompt_root(r: Root, header: RunHeader | None) -> tuple[bool, str]:
    """`prompt[<request>][<pos>]` carries the request record's token at that position."""
    if header is None:
        return False, "no RunHeader"
    if not r.name.startswith("prompt[") or r.values is None or len(r.values) != 1:
        return False, f"unexpected prompt root {r.name!r} / values {r.values!r}"
    try:
        rid = r.name[len("prompt["):r.name.index("]")]
        pos = int(r.name[r.name.index("][") + 2:-1])
    except Exception:  # noqa: BLE001
        return False, f"unparseable prompt root name {r.name!r}"
    reqs = {q["request_id"]: q for q in (header.workload or {}).get("requests", [])}
    q = reqs.get(rid)
    if q is None:
        return False, f"request {rid} not in the workload record"
    toks = q.get("prompt_token_ids") or []
    if pos >= len(toks):
        return False, f"position {pos} beyond the prompt ({len(toks)} tokens)"
    if int(toks[pos]) != int(r.values[0]):
        return False, f"token {r.values[0]} != request record {toks[pos]} at position {pos}"
    return True, ""


@dataclass(frozen=True, kw_only=True)
class Options:
    log: str = option("--log")
    snapshots: str | None = option("--snapshots", default=None)
    profile: str = option("--profile")
    safetensors: str | None = option("--safetensors", default=None,
                                     help="glob of the checkpoint's model*.safetensors (default: HF cache of the profile's model)")
    cos_sin: str | None = option("--cos-sin", default=None)
    no_weight_bytes: bool = option("--no-weight-bytes", default=False, action="store_true",
                                   help="skip the snapshot-vs-checkpoint byte check (identity only)")
    out: str = option("--out")


def main(a: Options) -> int:
    from verity_vllm.check.fold_compare import load_profile

    profile = load_profile(a.profile)
    st = [Path(p) for p in sorted(glob.glob(a.safetensors))] if a.safetensors else None
    rep = evaluate(a.log, a.snapshots, profile, safetensors=st, cos_sin=Path(a.cos_sin) if a.cos_sin else None, weight_bytes=not a.no_weight_bytes)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1, default=str) + "\n")
    print(f"[root_policy] {rep['status']}: {rep['n_roots']} roots, {rep['n_failed']} failed, classes "
          + ", ".join(f"{k}={v['validated']}/{v['count']}" for k, v in rep["classes"].items())
          + (f"; new classes {rep['new_root_classes']}" if rep["new_root_classes"] else "")
          + (f"; problems {rep['problems'][:3]}" if rep["problems"] else ""), file=sys.stderr)
    return 0 if rep["status"] == "pass" else 1
