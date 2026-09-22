"""verity_capture/root_policy.py — PROTECTED (gate G3, root closure).  Frozen root classes, each with an independent validator.

    uv run python -m verity_capture.root_policy --log LOG --snapshots DIR --profile NAME [--safetensors GLOB] [--cos-sin NPY] --out root_policy.json

A Program operand is either produced by another Program instance or is a *root*.  The fold already refuses reads with no
producer (`Unresolved`); this module decides whether the roots the fold *did* accept are legitimate, using a source of
authority that is independent of the run for each class:

| class            | roots                                                      | validator                                              |
|------------------|------------------------------------------------------------|--------------------------------------------------------|
| weight           | `named_parameters()` mapped to a `weights` Record field    | checkpoint bytes: the snapshot of the root's storage == the checkpoint's words (`WeightSource`); roots the observer did not snapshot (over `snapshot-max-bytes`) fall back to *checkpoint identity*: the header's frozen `weights_sha256` verified at engine build (`--verify-weights`) |
| analytic-table   | the rotary `cos_sin_cache` buffer; Gemma-2's `model.normalizer` (0-d bf16 `sqrt(H)` buffer) | recomputed from `C` (`analytic.check_cos_sin`; `analytic.check_normalizer`, exact) |
| alias            | another name for a root's storage (layer-k `cos_sin_cache`, tied `lm_head.weight`) | identical allocation + view in the observation log (storage identity proven by the runtime, never assumed from the family) |
| request-input    | prompt token rows                                          | the request record (`RunHeader.workload`)              |
| config           | the static configuration `C` behind every specialization   | the checkpoint's `config.json` subset in the header (`analytic.config_of`), field by field |
| ignored          | placeholders never read by a modeled kernel (`*_scale` with `kv_cache_dtype=auto`) | every read of their storage happens in a frozen set of non-semantic argument slots |

Anything else — an unknown root name, a `Root.role` outside the table, an `Opaque`/prescribed fallback — is a **new root
class** and fails.  A lane cannot make replay pass by declaring an inconvenient tensor an input: that would show up here as
`new_root_classes` or as a weight root whose bytes match no checkpoint field.

This module and the tables in it change only through an approved decision in `out/capture/decisions.md` (see
`out/gen/RULES.md`, protected paths).  Torch-free.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from verity_vllm.frontend.rules.family import _field_intervals

from . import analytic
from .events import DispatchEnter, Root, RunHeader, TritonLaunch, ViewDesc
from .log import read_log

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
#: Frozen.  Name patterns of the two non-weight root classes (the policy classifies; a profile's own `root_class` is advisory).
#: [SHARED-CODE lane cov-ln] the module prefix (`model` for Llama/Qwen/Gemma/Phi, `gpt_neox` for GPT-NeoX/Pythia) and the attention
#: submodule name (`self_attn` / `attention`) are vLLM naming facts of the family, not of the root class: the two classes below are
#: "another layer's name for the shared rotary table" and "the attention layer's KV/Q scale placeholders", whatever the family
#: calls its modules.  Before this generalization the patterns were Llama-only (`model.layers.N.self_attn.…`).
_MODULE_PREFIX = r"(?:model|gpt_neox)"
_ATTN_MODULE = r"(?:self_attn|attention)"
ALIAS_OF_COS_SIN = re.compile(rf"^{_MODULE_PREFIX}\.layers\.[1-9]\d*\.{_ATTN_MODULE}\.rotary_emb\.cos_sin_cache$")
IGNORED_ROOT_NAMES = (re.compile(rf"^{_MODULE_PREFIX}\.layers\.\d+\.{_ATTN_MODULE}\.attn\.(k_scale|v_scale|q_scale|prob_scale|_k_scale|_v_scale|_q_scale|_prob_scale)$"),)
_LAYER0_COS_SIN = re.compile(rf"^{_MODULE_PREFIX}\.layers\.0\.{_ATTN_MODULE}\.rotary_emb\.cos_sin_cache$")


def frozen_root_class(name: str, weight_field) -> str:
    if weight_field(name) is not None:
        return "weight"
    if ALIAS_OF_COS_SIN.match(name):
        return "alias-of-cos_sin"
    if any(r.match(name) for r in IGNORED_ROOT_NAMES):
        return "ignored"
    return "unknown"


def _views(x: Any):
    if isinstance(x, ViewDesc):
        yield x
    elif isinstance(x, (list, tuple)):
        for y in x:
            yield from _views(y)
    elif isinstance(x, dict):
        for y in x.values():
            yield from _views(y)


def _field_name(path: tuple) -> str:
    """("layers", 3, "qkv_w") -> "layers[3].qkv_w"; ("embed",) -> "embed" (the `_field_intervals` naming)."""
    out = ""
    for p in path:
        out = f"{out}[{p}]" if isinstance(p, int) else (f"{out}.{p}" if out else p)
    return out


def evaluate(log_path: str | Path, snapshots: str | Path | None, profile, *, safetensors: list[Path] | None = None,
             cos_sin: Path | None = None, weight_bytes: bool = True) -> dict[str, Any]:
    from .compare import SnapshotStore, WeightSource, _extent
    from .fold import config_for_weights

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

    # --- ignored roots: every read must sit in a frozen non-semantic slot ---------------------------------------------------
    ignored_allocs = {e["alloc"]: e for e in out["roots"] if e["class"] == "ignored"}
    consumers: Counter = Counter()
    bad_consumers: Counter = Counter()
    if ignored_allocs:
        for e in events:
            if isinstance(e, DispatchEnter):
                vs = [v for v in list(_views(e.args)) + list(_views(e.kwargs)) if v.alloc in ignored_allocs]
                if vs:
                    consumers[e.op] += len(vs)
                    if e.op not in IGNORED_ROOT_CONSUMERS:
                        bad_consumers[e.op] += len(vs)
            elif isinstance(e, TritonLaunch):
                vs = [v for v in _views(e.args) if v.alloc in ignored_allocs]
                if vs:
                    consumers[f"triton:{e.fn}"] += len(vs)
                    bad_consumers[f"triton:{e.fn}"] += len(vs)
        for ent in ignored_allocs.values():
            ent["status"] = "pass" if not bad_consumers else "fail"
            if bad_consumers:
                ent["reason"] = f"read by non-frozen consumers {dict(bad_consumers)}"
        classes["ignored"] = Counter(ent["status"] for ent in ignored_allocs.values())
    out["ignored_root_consumers"] = dict(consumers)

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

    # --- config root: C's hf-config fields against the checkpoint's config.json subset in the header ------------------------
    cfg_entry = {"class": "config", "validator": "hf-config (header.model.config)", "status": "skipped", "fields": {}}
    hf = (header.model or {}).get("config") if header is not None else None
    if hf:
        try:
            derived, prov = analytic.config_of(hf, model=repo)
            mism = {}
            for k in ("H", "NL", "NH", "KVH", "D", "I", "V", "EPS", "ROPE_THETA", "MAX_POS", "tie_word_embeddings"):
                if k in derived and k in C and prov.get(k) == "hf-config" and derived[k] != C[k]:
                    mism[k] = {"C": C[k], "hf": derived[k]}
                cfg_entry["fields"][k] = prov.get(k, "n/a")
            cfg_entry["status"] = "pass" if not mism else "fail"
            cfg_entry["mismatches"] = mism
            if mism:
                out["problems"].append(f"C differs from the checkpoint config: {mism}")
        except Exception as e:  # noqa: BLE001
            cfg_entry.update(status="fail", reason=f"{type(e).__name__}: {e}")
    else:
        cfg_entry["reason"] = "header carries no HF config subset"
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


def main(argv: list[str] | None = None) -> int:
    from .compare import load_profile

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--log", required=True)
    ap.add_argument("--snapshots", default=None)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--safetensors", default=None, help="glob of the checkpoint's model*.safetensors (default: HF cache of the profile's model)")
    ap.add_argument("--cos-sin", default=None)
    ap.add_argument("--no-weight-bytes", action="store_true", help="skip the snapshot-vs-checkpoint byte check (identity only)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
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


if __name__ == "__main__":
    sys.exit(main())
