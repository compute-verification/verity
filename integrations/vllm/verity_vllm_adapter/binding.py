"""Statement binding (GUIDANCE C3): the verifier fixes *what* is being verified from references it
holds itself, before any challenge randomness exists.

Three bindings, each a pure function of the bundle and of documents the verifier owns:

C3a **model** -- :func:`model_binding`: every leaf of the committed ``weights`` tree is recomputed
    from the pinned checkpoint (``manifests/checkpoints.json``: repository, revision, sha256 of
    every file; the local snapshot's files are re-hashed first) through the semantic profile's
    weight map (vLLM fuses ``q/k/v`` and ``gate/up``, pads the vocabulary, derives the rotary
    ``cos_sin_cache`` from ``config.json``), the weights root is rebuilt from the *recomputed*
    leaves and must equal the committed root.  The prover's ``weights/index.json`` is used only
    to learn which tensors it claims; nothing in it is trusted.

C3b **architecture** -- :func:`architecture_binding`: ``description.json`` and ``required.json``
    are regenerated from the pinned ``config.json`` plus the semantic profile
    (``manifests/semantic-profiles/*.json``: node templates, wiring, kernels, relations, shapes as
    expressions of the config) and must equal the prover's documents on every binding field;
    the only fields not compared are observational (``instances_observed``, ``arg_violations`` --
    the latter must be empty).

C3c **output** -- :func:`output_binding`: every boundary root in ``commitments.json`` is
    recomputed from the production record (``production.jsonl``: prompt ids, output ids,
    sampling params, seed, finish reason) plus the committed step records, and
    ``sha256(production.jsonl)`` enters the statement.  This is the check the fresh verifier used
    not to perform (it never read the production record; ``rowcheck`` did, after the fact).

:func:`bind` runs all three, returns the statement fragment the challenge transcript's ``T_0``
covers, and the list of problems.  Any problem is a structural failure of the row; the
challenge is still drawn (so the negatives can exercise the rejection path), the verdict is FAIL.

Negatives (:func:`binding_negatives`): N15 a committed weight differs from the checkpoint; N16 the
description deviates from the architecture (dropped node / changed shape / changed relation);
N17 the production record differs from the committed boundary.  Each mutation must be rejected
by the corresponding binding.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verity_vllm import hashing, identity
from verity_vllm import required as vrequired
from verity_vllm.capture import LOGPROB_DTYPE, WEIGHTS_TREE_ID, CaptureError, CaptureReader, boundary_tree_id, row_block_leaves
from verity_vllm.commit import MerkleTree
from verity_vllm.description import Description
from verity_vllm.rowcheck import logprob_summary, production_boundaries_with_status

REPO_ROOT = Path(__file__).resolve().parents[1]
SEMANTIC_PROFILES = REPO_ROOT / "manifests" / "semantic-profiles"
# The frozen v1 profile (GUIDANCE §24): the contract of the three-patch recorder that captured the committed B0 CPU bundles
# (bundles/B0-*-2026-09-07*), which verify_cpu.py re-verifies by default. verify.py defaults to `auto`, which resolves to the
# current-recorder v2 twins / the Hopper profile by architecture + manifest requirements and never to a frozen profile.
DEFAULT_SEMANTIC_PROFILE = SEMANTIC_PROFILES / "llama-eager-batch-invariant-v1.json"


class BindingError(ValueError):
    pass


@dataclass
class BindingResult:
    name: str
    ok: bool
    detail: str
    data: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail, "seconds": round(self.seconds, 3), **self.data}


# --- expression templates ------------------------------------------------------------------------


def _symbols(profile: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    env: dict[str, Any] = {k: v for k, v in config.items() if isinstance(v, int | float) and not isinstance(v, bool)}
    out: dict[str, Any] = {}
    for name, expr in profile["symbols"].items():
        out[name] = eval(expr, {"__builtins__": {}}, {**env, **out})  # noqa: S307 - profile is the verifier's own file
    return out


def _num(value: Any) -> Any:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _ev(value: Any, env: Mapping[str, Any]) -> Any:
    """Evaluate ``"=expr"`` strings, substitute ``{expr}`` inside strings, recurse into containers."""
    if isinstance(value, str):
        if value.startswith("="):
            return _num(eval(value[1:], {"__builtins__": {}}, dict(env)))  # noqa: S307
        return re.sub(r"\{([^{}]+)\}", lambda m: str(_num(eval(m.group(1), {"__builtins__": {}}, dict(env)))), value)  # noqa: S307
    if isinstance(value, list):
        return [_ev(v, env) for v in value]
    if isinstance(value, dict):
        return {k: _ev(v, env) for k, v in value.items()}
    return value


# --- C3b: architecture --------------------------------------------------------------------------


def load_semantic_profile(path: str | Path = DEFAULT_SEMANTIC_PROFILE) -> dict[str, Any]:
    doc = json.loads(Path(path).read_text())
    if doc.get("schema") != "verity-vllm/semantic-profile/v1":
        raise BindingError(f"{path}: not a semantic profile")
    return doc


def select_semantic_profile(config: Mapping[str, Any], manifest: Mapping[str, Any] | None = None, *,
                            profiles_dir: Path = SEMANTIC_PROFILES) -> Path:
    """The one shipped semantic profile whose ``architectures`` covers the checkpoint's architecture and whose
    ``requires`` the bundle's manifest satisfies (compute capability, eager/batch-invariant flags, dtype, TP, quant).

    ``verify.py --semantic-profile auto`` (the default since the HP merge, 2026-09-08) resolves through here, so a B1
    (``Qwen2ForCausalLM``) bundle binds through the Qwen2 profile of its hardware class without the operator naming it --
    HP's Level-3 finding was the Llama default refusing Qwen2, and the Ampere/Ada profile (FlashAttention-2) then refusing
    the H100's ``fa_version`` 3. Zero or several matches are a ``BindingError`` (the verifier does not guess between
    profiles; the message names each candidate's unmet requirement); an explicit path bypasses the selection.
    Profiles marked ``"status": "frozen"`` (historical recorder regimes, GUIDANCE §24) are never auto-selected."""
    arch = (config.get("architectures") or [None])[0]
    hits, rejected = [], []
    for p in sorted(profiles_dir.glob("*.json")):
        doc = json.loads(p.read_text())
        if doc.get("schema") != "verity-vllm/semantic-profile/v1" or arch not in doc.get("architectures", []):
            continue
        if doc.get("status") == "frozen":  # historical profile (GUIDANCE §24): re-verification of old archives by explicit path only
            continue
        unmet = _profile_requirements(doc, manifest) if manifest is not None else []
        (hits if not unmet else rejected).append((p, unmet))
    if len(hits) != 1:
        raise BindingError(f"config architecture {config.get('architectures')} is covered by {len(hits)} shipped semantic profiles "
                           f"whose requirements this bundle meets ({[h.name for h, _ in hits]}); "
                           f"rejected: {[(h.name, u) for h, u in rejected]}; pass --semantic-profile explicitly")
    return hits[0][0]


def _profile_requirements(profile: Mapping[str, Any], manifest: Mapping[str, Any]) -> list[str]:
    problems = []
    for dotted, want in profile.get("requires", {}).items():
        cur: Any = manifest
        for part in dotted.split("."):
            cur = cur.get(part) if isinstance(cur, Mapping) else None
        ok = (cur in want) if isinstance(want, list) else (cur == want)  # a list is "one of" (e.g. the measured compute capabilities)
        if not ok:
            problems.append(f"manifest.{dotted} = {cur!r}, semantic profile requires {want!r}")
    return problems


def generate_description(profile: Mapping[str, Any], config: Mapping[str, Any], *, profile_id: str, model_repo: str) -> dict[str, Any]:
    """The verifier's expected ``description.json`` for this profile and checkpoint config."""
    if config.get("architectures", [None])[0] not in profile["architectures"]:
        raise BindingError(f"config architecture {config.get('architectures')} not covered by semantic profile {profile['id']}")
    sym = _symbols(profile, config)
    impls = profile["runtime_impl"]
    tied = bool(config.get("tie_word_embeddings", False))
    lm_head_param = profile["weights"]["lm_head_param"]["tied" if tied else "untied"]
    phases = list(profile["phases"])

    def render(tpl: Mapping[str, Any], env: Mapping[str, Any]) -> dict[str, Any]:
        t = _ev({k: v for k, v in tpl.items() if k not in ("when", "interface")}, env)
        iface = _ev(tpl["interface"], env)
        impl = t["runtime_impl"]
        if isinstance(impl, str) and impl.startswith("@"):
            impl = impls[impl[1:]]
        args = []
        for a in t["args"]:
            a = dict(a)
            if a.get("param") == "@lm_head_param":
                a["param"] = lm_head_param
            args.append(a)
        node = {
            "op": t["op"], "layer": str(t["layer"]), "op_type": t["op_type"], "dtype": t["dtype"], "shape": t["shape"],
            "ports": t["ports"], "outputs": t["outputs"], "phases": phases, "positions": t["positions"], "args": args,
            "relation": t["relation"], "origin": "captured",
            "decompositions": [{"id": "row", "interface": {"input_bytes": iface["input_bytes"], "output_bits": iface["output_bits"], "range": "row"}}],
            "runtime_impl": impl, "family": t["family"], "runtime_op": t["runtime_op"], "consts": t["consts"],
            "arg_cols": t["arg_cols"], "unbound_args": t["unbound_args"], "module_path": t["module_path"],
        }
        # declared output domain (GUIDANCE §31.2/§32.5, lane OUT16): additive pass-through —
        # emitted only when the profile's node template declares it, so descriptions regenerated
        # from the existing profiles are byte-identical; a profile revision that pins the 16-bit
        # token domain adds `"declared_domain": 16` to its sampler template and the description
        # (digest-bound at T_0 via bindings.architecture) then carries the declaration the
        # verifier's unconditional output-domain check reads.
        if "declared_domain" in t:
            node["declared_domain"] = t["declared_domain"]
        return node

    nodes: list[dict[str, Any]] = []
    for i in range(int(sym["L"])):
        env = {**sym, "i": i}
        for tpl in profile["layer_nodes"]:
            cond = tpl.get("when")
            if cond is not None and not eval(cond, {"__builtins__": {}}, dict(env)):  # noqa: S307
                continue
            nodes.append(render(tpl, env))
    for tpl in profile["nodes"]:
        nodes.append(render(tpl, sym))
    nodes.sort(key=lambda n: node_sort_key(n["op"]))
    return {
        "schema": "verity-vllm/description/v1",
        "profile_id": profile_id,
        "model": model_repo,
        "symbols": {k: int(sym[k]) for k in profile["description_symbols"]},
        "interception": profile["interception"],
        "patches": list(profile["patches"]),
        "nodes": nodes,
    }


def node_sort_key(op: str) -> tuple:
    """The adapter's node order (``verity_vllm_adapter.capture.node_sort_key``, copied so this module
    imports without torch/vLLM)."""
    parts = op.split(".")
    if parts[0] == "layers" and parts[1].isdigit():
        return (1, int(parts[1]), op)
    if parts[0] == "embed_tokens":
        return (0, 0, op)
    if parts[0] == "norm":
        return (2, 0, op)
    if parts[0] == "lm_head":
        return (3, 0, op)
    return (4, 0, op)


def _strip(node: Mapping[str, Any], ignored: set[str]) -> dict[str, Any]:
    return {k: v for k, v in node.items() if k not in ignored}


def architecture_binding(
    prover_description: Mapping[str, Any], prover_required: Mapping[str, Any] | None, *,
    profile: Mapping[str, Any], config: Mapping[str, Any], manifest: Mapping[str, Any], profile_id: str, model_repo: str,
) -> BindingResult:
    t0 = time.perf_counter()
    problems: list[str] = _profile_requirements(profile, manifest)
    ignored = set(profile.get("ignored_description_fields", []))
    expected = generate_description(profile, config, profile_id=profile_id, model_repo=model_repo)
    got_nodes = {n["op"]: n for n in prover_description.get("nodes", [])}
    exp_nodes = {n["op"]: n for n in expected["nodes"]}
    missing = sorted(set(exp_nodes) - set(got_nodes))
    unexpected = sorted(set(got_nodes) - set(exp_nodes))
    if missing:
        problems.append(f"{len(missing)} architecture node(s) absent from description.json, e.g. {missing[0]}")
    if unexpected:
        problems.append(f"{len(unexpected)} node(s) in description.json the architecture does not have, e.g. {unexpected[0]}")
    differing: list[dict[str, Any]] = []
    for op in sorted(set(exp_nodes) & set(got_nodes)):
        g, e = _strip(got_nodes[op], ignored), _strip(exp_nodes[op], ignored)
        if g != e:
            fields = sorted(k for k in set(g) | set(e) if g.get(k) != e.get(k))
            differing.append({"op": op, "fields": fields, "expected": {k: e.get(k) for k in fields}, "got": {k: g.get(k) for k in fields}})
        if got_nodes[op].get("arg_violations"):
            problems.append(f"{op}: arg_violations recorded ({len(got_nodes[op]['arg_violations'])}); the observed args violate the identity rule")
    if differing:
        d = differing[0]
        problems.append(f"{len(differing)} node(s) differ from the architecture, e.g. {d['op']} in {d['fields']}: expected {d['expected']} got {d['got']}")
    for key in ("schema", "model", "symbols", "interception", "patches"):
        if prover_description.get(key) != expected[key]:
            problems.append(f"description.{key}: {prover_description.get(key)!r} != expected {expected[key]!r}")
    if prover_description.get("profile_id") != profile_id:
        problems.append(f"description.profile_id {prover_description.get('profile_id')} != manifest profile {profile_id}")
    # required.json regenerated from the *expected* description under the profile's partition
    exp_required = None
    try:
        exp_desc = Description(expected)
        exp_required = vrequired.derive_required(exp_desc, profile["partition"])
    except Exception as e:  # noqa: BLE001
        problems.append(f"expected description does not parse: {type(e).__name__}: {e}")
    if prover_required is None:
        problems.append("no required.json")
    elif exp_required is not None:
        if prover_required.get("interfaces") != exp_required["interfaces"]:
            a = {json.dumps(i, sort_keys=True) for i in prover_required.get("interfaces", [])}
            b = {json.dumps(i, sort_keys=True) for i in exp_required["interfaces"]}
            problems.append(f"required.json interfaces differ from the regenerated set: {len(a - b)} extra, {len(b - a)} missing")
        if prover_required.get("partition_id") != exp_required["partition_id"]:
            problems.append(f"required.partition_id {prover_required.get('partition_id')} != {exp_required['partition_id']}")
        try:
            prover_digest = Description(prover_description).digest().hex()
        except Exception:  # noqa: BLE001
            prover_digest = None
        if prover_required.get("description_digest") != prover_digest:
            problems.append("required.description_digest does not bind the prover's description.json")
    exp_digest = hashing.json_digest(expected).hex()
    return BindingResult(
        "architecture", not problems,
        (f"description.json equals the regeneration from config.json + semantic profile {profile['id']} on every binding field "
         f"({len(exp_nodes)} nodes; ignored observational fields {sorted(ignored)}); required.json equals the regenerated set "
         f"({len(exp_required['interfaces']) if exp_required else '?'} interfaces)") if not problems else "; ".join(problems[:4]),
        {"semantic_profile": profile["id"], "semantic_profile_sha256": None, "config_sha256": hashlib.sha256(hashing.canonical_json_bytes(config)).hexdigest(),
         "expected_nodes": len(exp_nodes), "expected_description_digest": exp_digest,
         "expected_required_interfaces": len(exp_required["interfaces"]) if exp_required else None,
         "missing_nodes": missing[:5], "unexpected_nodes": unexpected[:5], "differing_nodes": differing[:5], "problems": problems},
        time.perf_counter() - t0,
    )


# --- C3a: model ----------------------------------------------------------------------------------


def rope_cos_sin_cache(rotary_dim: int, base: float, max_position: int, dtype: str, device: str = "cpu") -> bytes:
    """vLLM ``RotaryEmbedding._compute_cos_sin_cache``: fp32 ``cat(cos, sin)`` of ``t x inv_freq``, cast to the model dtype."""
    import torch

    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float, device=device) / rotary_dim))
    t = torch.arange(max_position, dtype=torch.float, device=device)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(getattr(torch, dtype))
    return cache.contiguous().cpu().view(torch.uint8).numpy().tobytes()


class _Checkpoint:
    """Tensors of the pinned safetensors files, read once each (the model is small; a large model would
    memory-map per tensor)."""

    def __init__(self, local: Path, files: list[str]) -> None:
        from safetensors import safe_open

        self._handles = [safe_open(str(local / f), "pt") for f in files]
        self.keys: set[str] = set()
        for h in self._handles:
            self.keys |= set(h.keys())

    def tensor(self, name: str) -> Any:
        for h in self._handles:
            if name in h.keys():
                return h.get_tensor(name)
        raise KeyError(name)


def _rule_for(param: str, rules: list[Mapping[str, Any]], sym: Mapping[str, Any]) -> tuple[Mapping[str, Any], dict[str, Any]] | None:
    for rule in rules:
        pat = rule["param"]
        if "{i}" in pat:
            m = re.fullmatch(re.escape(pat).replace(r"\{i\}", r"(\d+)"), param)
            if m and int(m.group(1)) < int(sym["L"]):
                return rule, {**sym, "i": int(m.group(1))}
        elif pat == param:
            return rule, dict(sym)
    return None


def _recompute_entry(entry: Any, rule: Mapping[str, Any], env: Mapping[str, Any], ckpt: "_Checkpoint", wdtype: str,
                     derived: dict[str, str]) -> bytes:
    """The bytes of one committed weight as the registered reference defines them: a checkpoint tensor (or a
    fusion / padding of several) in the profile's dtype, or a derived buffer the profile pins (RoPE cache from
    ``config.json``; runtime constants). ``derived`` collects a note per derived buffer ("NOT" marks a mismatch)."""
    import torch

    if rule.get("derived") == "rope_cos_sin_cache":
        raw = None
        for device in ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",):
            cand = rope_cos_sin_cache(int(_ev(rule["rotary_dim"], env)), float(_ev(rule["base"], env)), int(_ev(rule["max_position"], env)), entry.dtype, device)
            if hashing.sha256_hex(cand) == entry.sha256:
                raw, derived[entry.param] = cand, f"recomputed from config.json (rope_theta, head_dim, max_position_embeddings) on {device}"
                break
            raw = raw or cand
        if entry.param not in derived:
            derived[entry.param] = "recomputed from config.json: does NOT match the committed bytes"
        assert raw is not None
        return raw
    if rule.get("derived") == "constant":
        # a buffer the runtime fixes independent of the checkpoint (e.g. the attention quantisation
        # scales of an unquantised model): the profile pins its value, dtype and shape
        t = torch.full(tuple(int(x) for x in rule.get("shape", [])), float(rule["value"]), dtype=getattr(torch, rule["dtype"]))
        raw = t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        if entry.dtype != rule["dtype"] or tuple(entry.shape) != tuple(int(x) for x in rule.get("shape", [])):
            derived[entry.param] = f"constant {rule['value']} {rule['dtype']}{list(rule.get('shape', []))}: committed dtype/shape does NOT match"
        elif hashing.sha256_hex(raw) == entry.sha256:
            derived[entry.param] = f"constant {rule['value']} ({rule['dtype']}, shape {list(rule.get('shape', []))}) fixed by the profile"
        else:
            derived[entry.param] = f"constant {rule['value']} fixed by the profile: committed bytes do NOT match"
        return raw
    parts = [ckpt.tensor(_ev(name, env)) for name in rule["from"]]
    t = torch.cat(parts, dim=int(rule.get("cat_dim", 0))) if len(parts) > 1 else parts[0]
    if str(t.dtype).replace("torch.", "") != wdtype:
        t = t.to(getattr(torch, wdtype))
    mult = rule.get("pad_rows_to_multiple")
    if mult and t.shape[0] % int(mult):
        pad = int(mult) - t.shape[0] % int(mult)
        t = torch.cat([t, torch.zeros((pad, *t.shape[1:]), dtype=t.dtype)], dim=0)
    return t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def checkpoint_resolver(*, cp: Mapping[str, Any], config: Mapping[str, Any], profile: Mapping[str, Any]) -> Callable[[Any], bytes]:
    """A ``WeightsReader.resolver``: weight values from the registered reference instead of the bundle.

    A self-contained bundle need not carry the checkpoint's bytes (270 MB here, 30 GB for a 7B model): the
    weights index records each committed weight's leaf and sha256, and this resolver rebuilds the bytes from
    the pinned checkpoint through the same profile weight map that model binding (C3a) uses. The reader
    checks the resolved bytes' sha256 against the index before handing them out, so a wrong checkpoint or a
    wrong map is a ``CaptureError``, never a silently different value. Unknown parameters raise ``KeyError``."""
    local = Path(cp["local_path"])
    sym = _symbols(profile, config)
    rules = profile["weights"]["rules"]
    wdtype = profile["weights"]["dtype"]
    ckpt = _Checkpoint(local, sorted(cp["safetensors_sha256"]))
    cache: dict[str, bytes] = {}

    def resolve(entry: Any) -> bytes:
        if entry.canonical_id in cache:
            return cache[entry.canonical_id]
        hit = _rule_for(entry.param, rules, sym)
        if hit is None:
            raise KeyError(f"{entry.param}: no weight-map rule in the semantic profile")
        rule, env = hit
        raw = _recompute_entry(entry, rule, env, ckpt, wdtype, {})
        cache[entry.canonical_id] = raw
        return raw

    return resolve


def model_binding(
    reader: CaptureReader, commitments: Mapping[str, Any], *, cp: Mapping[str, Any], config: Mapping[str, Any],
    profile: Mapping[str, Any], verify_files: bool = True,
) -> BindingResult:
    """Recompute every committed weight leaf from the pinned checkpoint and rebuild the weights root."""
    t0 = time.perf_counter()
    problems: list[str] = []
    local = Path(cp["local_path"])
    # 1. the reference itself: the frozen shas of the pinned revision
    file_shas: dict[str, str] = {}
    if verify_files:
        for fname, sha in cp["safetensors_sha256"].items():
            h = hashlib.sha256()
            with (local / fname).open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 24), b""):
                    h.update(chunk)
            file_shas[fname] = h.hexdigest()
            if h.hexdigest() != sha:
                problems.append(f"{fname}: local sha256 {h.hexdigest()[:16]} != frozen {sha[:16]}")
    config_sha = hashlib.sha256((local / "config.json").read_bytes()).hexdigest()
    if cp.get("config_sha256") and config_sha != cp["config_sha256"]:
        problems.append(f"config.json sha256 {config_sha[:16]} != frozen {cp['config_sha256'][:16]}")
    manifest_shas = dict((reader.manifest.get("model") or {}).get("weights_sha256") or {})
    if manifest_shas != dict(cp["safetensors_sha256"]):
        problems.append("manifest.model.weights_sha256 is not the pinned checkpoint's")
    if (reader.manifest.get("model") or {}).get("revision") != cp.get("revision"):
        problems.append(f"manifest.model.revision {(reader.manifest.get('model') or {}).get('revision')} != pinned {cp.get('revision')}")
    if problems:
        return BindingResult("model", False, "; ".join(problems), {"problems": problems}, time.perf_counter() - t0)

    # 2. recompute every leaf the prover committed, from the checkpoint through the weight map
    sym = _symbols(profile, config)
    rules = profile["weights"]["rules"]
    wdtype = profile["weights"]["dtype"]
    ckpt = _Checkpoint(local, sorted(cp["safetensors_sha256"]))
    index = reader.weights.index
    recomputed: dict[str, bytes] = {}
    matched = 0
    mismatched: list[str] = []
    unexpected: list[str] = []
    derived: dict[str, str] = {}
    seen_rules: set[str] = set()
    bytes_recomputed = 0
    rows_layout_entries = 0
    for entry in index.entries:
        hit = _rule_for(entry.param, rules, sym)
        if hit is None:
            unexpected.append(entry.param)
            continue
        rule, env = hit
        seen_rules.add(_ev(rule["param"], env))
        raw = _recompute_entry(entry, rule, env, ckpt, wdtype, derived)
        bytes_recomputed += len(raw)
        if tuple(entry.shape) != tuple(_expected_shape(raw, entry)):
            pass  # shape is part of the leaf hash; a wrong shape shows up as a leaf mismatch below
        leaf = hashing.leaf_hash(entry.canonical_id, entry.dtype, entry.shape, raw)
        if entry.layout.kind == "rows":
            # capture format v2 (W3c): the tree leaves are the row blocks, recomputed from the same bytes;
            # the whole-tensor leaf stays the per-parameter match criterion below (informational in the tree)
            for start, block_leaf in zip(entry.block_starts(), row_block_leaves(entry, raw), strict=True):
                recomputed[entry.block_id(start)] = block_leaf
            rows_layout_entries += 1
        else:
            recomputed[entry.canonical_id] = leaf
        if leaf.hex() == entry.leaf and hashing.sha256_hex(raw) == entry.sha256:
            matched += 1
        else:
            mismatched.append(entry.param)
    # rules the checkpoint has that the prover did not commit (a missing tensor is a different model)
    expected_params: set[str] = set()
    for rule in rules:
        if rule.get("optional_when") and config.get(rule["optional_when"]):
            continue
        if "{i}" in rule["param"]:
            expected_params |= {_ev(rule["param"], {**sym, "i": i}) for i in range(int(sym["L"]))}
        else:
            expected_params.add(rule["param"])
    committed_params = {e.param for e in index.entries}
    missing = sorted(expected_params - committed_params)
    # 3. the root from the recomputed leaves (the prover's leaf hashes are not used)
    root_ok = None
    rec = (commitments.get("trees") or {}).get(WEIGHTS_TREE_ID)
    if rec is None:
        problems.append("no weights tree committed")
    elif not unexpected and not missing:
        tree = MerkleTree(recomputed, WEIGHTS_TREE_ID)
        root_ok = tree.root.hex() == rec["root"] and tree.leaf_count == int(rec["leaf_count"]) and tree.ids_digest.hex() == rec.get("ids_digest")
        if not root_ok:
            problems.append(f"weights root rebuilt from the checkpoint {tree.root.hex()[:16]} != committed {rec['root'][:16]}")
    if mismatched:
        problems.append(f"{len(mismatched)} committed weight(s) differ from the pinned checkpoint, e.g. {mismatched[0]}")
    if unexpected:
        problems.append(f"{len(unexpected)} committed weight(s) the architecture does not have, e.g. {unexpected[0]}")
    if missing:
        problems.append(f"{len(missing)} checkpoint tensor(s) not committed, e.g. {missing[0]}")
    bad_derived = [p for p, d in derived.items() if "NOT" in d]
    if bad_derived:
        problems.append(f"{len(bad_derived)} derived buffer(s) do not match their recomputation from config.json, e.g. {bad_derived[0]}")
    return BindingResult(
        "model", not problems,
        (f"{matched}/{len(index.entries)} weight leaves recomputed from the pinned checkpoint {cp['repo']}@{cp['revision'][:12]} "
         f"(files re-hashed against manifests/checkpoints.json); weights root rebuilt from the recomputed leaves == committed root; "
         f"{len(derived)} derived buffers (rotary cos_sin_cache recomputed from config.json; constant attention scales fixed by the profile)") if not problems else "; ".join(problems[:4]),
        {"repo": cp["repo"], "revision": cp["revision"], "file_sha256": file_shas or dict(cp["safetensors_sha256"]), "config_sha256": config_sha,
         "leaves_total": len(index.entries), "leaves_matched": matched, "leaves_mismatched": mismatched[:5], "unexpected_params": unexpected[:5],
         "tree_leaves_recomputed": len(recomputed), "rows_layout_entries": rows_layout_entries, "weights_format_version": index.format_version,
         "missing_params": missing[:5], "root_matches": root_ok, "bytes_recomputed": bytes_recomputed,
         "derived_buffers": {k: v for k, v in list(derived.items())[:3]}, "derived_buffers_total": len(derived), "problems": problems},
        time.perf_counter() - t0,
    )


def _expected_shape(raw: bytes, entry: Any) -> tuple[int, ...]:
    return tuple(entry.shape)


# --- C3c: output ---------------------------------------------------------------------------------


LOGPROB_LABEL = {
    "bound": "bound (one float32 output_logprob leaf per produced token)",
    "unbound": "unbound (boundary trees predate the output_logprob leaves; tokens, presence, params, seed, finish reason only)",
}


def output_binding(reader: CaptureReader, commitments: Mapping[str, Any], production_path: Path) -> BindingResult:
    """Every committed boundary root is the production record's (plus the committed step records).

    The record's protected outputs are the prompt and output tokens, presence, sampling parameters,
    seed, finish reason and -- for a bundle whose boundary trees carry ``output_logprob`` leaves --
    the chosen logprobs, bit-exact as binary32 words (``capture.LOGPROB_DTYPE``).  Which leaf set a
    tree carries is read off its committed ``ids_digest`` (``rowcheck.logprob_binding_status``);
    ``data["logprobs"]`` is ``bound`` / ``unbound`` (or the mixture) and the row carries it.  A bundle
    from before the field verifies as before and is labelled ``unbound``; it is never upgraded."""
    t0 = time.perf_counter()
    rows = [json.loads(l) for l in production_path.read_text().splitlines() if l.strip()]
    production = {str(r["request_id"]): r for r in rows}
    sha = hashlib.sha256(production_path.read_bytes()).hexdigest()
    boundaries, lp_status = production_boundaries_with_status(production, commitments)
    btrees = {rec["request_id"]: rec for rec in commitments["trees"].values() if rec.get("kind") == "boundary"}
    matched, problems, logprob_leaves = 0, [], 0
    for rid, b in boundaries.items():
        rec = btrees.get(rid)
        if rec is None:
            problems.append(f"{rid}: in the production record, no boundary tree committed")
            continue
        if lp_status[rid] == "bound" and not isinstance(production[rid].get("chosen_logprobs"), list):
            problems.append(f"{rid}: the committed boundary binds the chosen logprobs, the production record carries none")
            continue
        probs = b.validate()
        if probs:
            problems.append(f"{rid}: {probs[0]}")
            continue
        tree = MerkleTree.from_values(b.leaves(reader.profile_id), boundary_tree_id(rid))
        if tree.root.hex() != rec["root"] or tree.leaf_count != int(rec["leaf_count"]) or tree.ids_digest.hex() != rec.get("ids_digest"):
            what = "chosen logprob" if lp_status[rid] == "bound" and tree.ids_digest.hex() == rec.get("ids_digest") else "boundary"
            problems.append(f"{rid}: {what} root from the production record {tree.root.hex()[:16]} != committed {rec['root'][:16]}"
                            + (" (leaf set differs)" if tree.ids_digest.hex() != rec.get("ids_digest") else ""))
        else:
            matched += 1
            if b.binds_logprobs:
                logprob_leaves += len(b.output_token_ids)
    extra = sorted(set(btrees) - set(boundaries))
    if extra:
        problems.append(f"{len(extra)} committed boundary(ies) not in the production record, e.g. {extra[0]}")
    # the boundaries the checker will read (bundle ru/<rid>/boundary.json) must be these very leaves
    for rid in reader.request_ids():
        if rid in boundaries and lp_status[rid] in ("bound", "unbound") and not boundaries[rid].validate():
            try:
                same = reader.boundary(rid).leaves(reader.profile_id) == boundaries[rid].leaves(reader.profile_id)
            except (CaptureError, KeyError) as e:
                same, problems = False, problems + [f"{rid}: bundle boundary.json: {e}"]
            if not same:
                problems.append(f"{rid}: bundle boundary.json differs from the production record")
    logprobs = logprob_summary(lp_status)
    label = LOGPROB_LABEL.get(logprobs, logprobs)
    return BindingResult(
        "output", not problems,
        (f"{matched}/{len(btrees)} boundary roots recomputed from production.jsonl (sha256 {sha[:16]}...) + committed step records; "
         f"chosen logprobs {label}") if not problems else "; ".join(problems[:4]) + f"; chosen logprobs {logprobs}",
        {"production_record_sha256": sha, "requests": len(production), "boundary_roots_matched": matched, "boundary_roots_committed": len(btrees),
         "logprobs": logprobs, "logprobs_per_request": lp_status, "logprob_leaves": logprob_leaves, "logprob_leaf_dtype": LOGPROB_DTYPE,
         "logprob_leaf_encoding": "IEEE-754 binary32 word of the reported logprob, little-endian, shape []", "problems": problems},
        time.perf_counter() - t0,
    )


# --- all three -----------------------------------------------------------------------------------


@dataclass
class Bindings:
    model: BindingResult
    architecture: BindingResult
    output: BindingResult
    semantic_profile_sha256: str

    @property
    def ok(self) -> bool:
        return self.model.ok and self.architecture.ok and self.output.ok

    @property
    def problems(self) -> list[str]:
        return [f"{r.name} binding: {r.detail}" for r in (self.model, self.architecture, self.output) if not r.ok]

    def to_json(self) -> dict[str, Any]:
        return {"ok": self.ok, "semantic_profile_sha256": self.semantic_profile_sha256,
                "model": self.model.to_json(), "architecture": self.architecture.to_json(), "output": self.output.to_json()}

    def statement(self) -> dict[str, Any]:
        """What enters ``T_0``: the references and the binding outcomes (not the timings)."""
        return {
            "model": {"repo": self.model.data.get("repo"), "revision": self.model.data.get("revision"), "file_sha256": self.model.data.get("file_sha256"),
                      "config_sha256": self.model.data.get("config_sha256"), "ok": self.model.ok},
            "architecture": {"semantic_profile": self.architecture.data.get("semantic_profile"), "semantic_profile_sha256": self.semantic_profile_sha256,
                             "expected_description_digest": self.architecture.data.get("expected_description_digest"), "ok": self.architecture.ok},
            "output": {"production_record_sha256": self.output.data.get("production_record_sha256"), "ok": self.output.ok},
        }


def bind(
    reader: CaptureReader, commitments: Mapping[str, Any], *, cp: Mapping[str, Any], production_path: Path,
    semantic_profile_path: str | Path = DEFAULT_SEMANTIC_PROFILE, verify_files: bool = True,
) -> Bindings:
    profile = load_semantic_profile(semantic_profile_path)
    sp_sha = hashlib.sha256(Path(semantic_profile_path).read_bytes()).hexdigest()
    config = json.loads((Path(cp["local_path"]) / "config.json").read_text())
    required = reader.required if reader.has("required.json") else None
    arch = architecture_binding(reader.description, required, profile=profile, config=config, manifest=reader.manifest,
                                profile_id=reader.profile_id, model_repo=cp["repo"])
    arch.data["semantic_profile_sha256"] = sp_sha
    model = model_binding(reader, commitments, cp=cp, config=config, profile=profile, verify_files=verify_files)
    out = output_binding(reader, commitments, production_path)
    return Bindings(model, arch, out, sp_sha)


# --- negatives -----------------------------------------------------------------------------------


BINDING_NEGATIVE_IDS = ("N15", "N16", "N17", "N19")


def binding_negatives(
    reader: CaptureReader, commitments: Mapping[str, Any], *, cp: Mapping[str, Any] | None, production_path: Path, out_dir: Path,
    semantic_profile_path: str | Path = DEFAULT_SEMANTIC_PROFILE, ids: Iterable[str] = BINDING_NEGATIVE_IDS,
) -> list[dict[str, Any]]:
    """N15 / N16 / N17 / N19: each mutation of the statement must be rejected by its binding.  Returns
    Table 3 style dicts (``negative_id, attempted, rejected, unexpectedly_accepted, reproducer``) and
    writes a reproducer per negative.  ``cp`` (the pinned checkpoint) is needed by N15/N16 only; N17/N19
    are CPU-only and run without it.  A negative whose target this bundle does not have (N19 on a bundle
    whose boundaries do not bind logprobs) reports ``attempted: 0`` with ``inapplicable`` set -- it is
    neither evidence nor NOT_RUN."""
    from verity_vllm.negatives import PatchedReader, flip_bit

    wanted = [x for x in ids]
    profile = load_semantic_profile(semantic_profile_path)
    config = json.loads((Path(cp["local_path"]) / "config.json").read_text()) if cp is not None else None
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    def finish(nid: str, variants: list[dict[str, Any]], inapplicable: str | None = None) -> None:
        rejected = sum(1 for v in variants if v["outcome"] == "rejected")
        path = out_dir / f"{nid}.json"
        doc = {"negative_id": nid, "run_dir": str(reader.run_dir), "variants": variants, "attempted": len(variants),
               "rejected": rejected, "unexpectedly_accepted": len(variants) - rejected}
        if inapplicable:
            doc["inapplicable"] = inapplicable
        path.write_text(json.dumps(doc, indent=1) + "\n")
        results.append({"negative_id": nid, "attempted": len(variants), "rejected": rejected, "unexpectedly_accepted": len(variants) - rejected,
                        "reproducer": str(path), "variants": variants, "inconclusive": 0, **({"inapplicable": inapplicable} if inapplicable else {})})

    class Inapplicable(Exception):
        """The bundle has no target for this negative (recorded, not counted)."""

    def attempt(nid: str, fn) -> None:
        if nid not in wanted:
            return
        # one negative's failure to *run* (e.g. a bundle without weight values) must not hide the others
        try:
            finish(nid, fn())
        except Inapplicable as e:
            finish(nid, [], inapplicable=str(e))
        except Exception as e:  # noqa: BLE001
            results.append({"negative_id": nid, "attempted": 0, "rejected": 0, "unexpectedly_accepted": 0, "reproducer": None,
                            "variants": [], "inconclusive": 0, "error": f"{type(e).__name__}: {e}"})

    # N15: a committed weight differs from the checkpoint (one bit of one tensor, the tree rebuilt over it)
    def n15() -> list[dict[str, Any]]:
        variants = []
        entries = list(reader.weights.index.entries)
        for entry in [entries[0], entries[len(entries) // 2], entries[-1]][: min(3, len(entries))]:
            raw = flip_bit(reader.weights.read(entry.canonical_id), 7, 0x01)
            patched = PatchedReader(reader, weights={entry.canonical_id: raw})
            # the prover recommits honestly over its mutated weights: rebuild the weights record
            from verity_vllm.commit import TreeCache
            tree = TreeCache(patched, recompute_leaves=True).tree(WEIGHTS_TREE_ID)
            doc = json.loads(json.dumps(commitments))
            old = doc["trees"][WEIGHTS_TREE_ID]
            meta = {k: v for k, v in old.items() if k not in ("kind", "root", "leaf_count", "ids_digest")}
            doc["trees"][WEIGHTS_TREE_ID] = tree.record("weights", **meta)
            r = model_binding(patched, doc, cp=cp, config=config, profile=profile, verify_files=False)
            variants.append({"variant": f"flip one bit of {entry.param}, weights tree rebuilt", "outcome": "rejected" if not r.ok else "accepted", "detail": r.detail[:300]})
        return variants

    # N16: the description deviates from the architecture
    def n16() -> list[dict[str, Any]]:
        variants = []
        desc = json.loads(json.dumps(reader.description))
        required = reader.required if reader.has("required.json") else None
        mutations = [
            ("drop one node (a layer's down_proj)", lambda d: d["nodes"].pop(next(i for i, n in enumerate(d["nodes"]) if n["op"].endswith("mlp.down_proj.gemm")))),
            ("change one gemm's row shape", lambda d: next(n for n in d["nodes"] if n["op"].endswith("qkv_proj.gemm")).__setitem__("shape", [1])),
            ("swap one node's relation to an independent one", lambda d: next(n for n in d["nodes"] if n["op"].endswith("input_layernorm.rmsnorm")).__setitem__("relation", "rmsnorm.independent.v1")),
            ("rewire a residual to the wrong layer", lambda d: next(a for n in d["nodes"] if n["op"] == "layers.2.post_attention_layernorm.rmsnorm" for a in n["args"] if a["name"] == "residual").__setitem__("op", "layers.1.input_layernorm.rmsnorm")),
            ("point a weight arg at another tensor", lambda d: next(a for n in d["nodes"] if n["op"] == "layers.3.self_attn.o_proj.gemm" for a in n["args"] if a["name"] == "w").__setitem__("param", "layers.4.self_attn.o_proj.weight")),
        ]
        for label, mutate in mutations:
            d = json.loads(json.dumps(desc))
            try:
                mutate(d)
            except StopIteration:
                continue  # mutation target absent: inconclusive, not counted
            r = architecture_binding(d, required, profile=profile, config=config, manifest=reader.manifest, profile_id=reader.profile_id, model_repo=cp["repo"])
            variants.append({"variant": label, "outcome": "rejected" if not r.ok else "accepted", "detail": r.detail[:300]})
        return variants

    # N17: the production record differs from the committed boundary
    def n17() -> list[dict[str, Any]]:
        variants = []
        rows = [json.loads(l) for l in production_path.read_text().splitlines() if l.strip()]
        tmp = out_dir / "N17.production.jsonl"
        n = len(rows)  # targets spread over the requests; a small record (2 requests) wraps around
        for label, mutate in (
            ("change one output token of one request", lambda rs: rs[0]["output_token_ids"].__setitem__(len(rs[0]["output_token_ids"]) // 2, (rs[0]["output_token_ids"][len(rs[0]["output_token_ids"]) // 2] + 1) % 1000)),
            ("change one prompt token of one request", lambda rs: rs[-1]["prompt_token_ids"].__setitem__(3, (rs[-1]["prompt_token_ids"][3] + 1) % 1000)),
            ("drop the last output token of one request", lambda rs: rs[1 % n]["output_token_ids"].pop()),
            ("change the sampling temperature of one request", lambda rs: rs[2 % n].setdefault("sampling_params", {}).__setitem__("temperature", 0.7)),
        ):
            rs = json.loads(json.dumps(rows))
            mutate(rs)
            tmp.write_text("".join(json.dumps(r) + "\n" for r in rs))
            r = output_binding(reader, commitments, tmp)
            variants.append({"variant": label, "outcome": "rejected" if not r.ok else "accepted", "detail": r.detail[:300]})
        tmp.unlink(missing_ok=True)
        return variants

    # N19 (WB2, audit finding 1 / WA's M2): a falsified chosen logprob in the production record.  Applicable only
    # when the committed boundary trees carry output_logprob leaves; the mutations are bit-level (one adjacent
    # binary32, WA's absolute -0.0001), a decimal rounding (not a binary32 value), a null, and the field dropped.
    def n19() -> list[dict[str, Any]]:
        import struct

        rows = [json.loads(l) for l in production_path.read_text().splitlines() if l.strip()]
        production = {str(r["request_id"]): r for r in rows}
        _, lp_status = production_boundaries_with_status(production, commitments)
        bound = [rid for rid, st in lp_status.items() if st == "bound"]
        if not bound:
            raise Inapplicable(f"chosen logprobs are not bound by this bundle's boundary trees (logprobs: {logprob_summary(lp_status)}); "
                               "N19 needs a bundle whose boundaries carry output_logprob leaves")
        targets = [r for r in rows if r["request_id"] in bound and isinstance(r.get("chosen_logprobs"), list) and r["chosen_logprobs"]]
        if not targets:
            raise Inapplicable("no request with recorded chosen logprobs")
        tmp = out_dir / "N19.production.jsonl"
        n = len(targets)

        def one_ulp(x: float) -> float:  # the adjacent binary32 towards zero: exactly representable, one bit away
            bits = struct.unpack("<I", struct.pack("<f", x))[0]
            return struct.unpack("<f", struct.pack("<I", bits - 1 if bits & 0x7FFFFFFF else bits + 1))[0]

        def between(x: float) -> float:  # the midpoint of two adjacent binary32 values: a double, never a binary32
            return (x + one_ulp(x)) / 2

        variants = []
        for label, mutate in (
            ("set one chosen logprob of one request to -0.0001 (WA's M2)", lambda rs: rs[0]["chosen_logprobs"].__setitem__(min(5, len(rs[0]["chosen_logprobs"]) - 1), -0.0001)),
            ("move one chosen logprob by one binary32 ulp", lambda rs: rs[1 % n]["chosen_logprobs"].__setitem__(0, one_ulp(rs[1 % n]["chosen_logprobs"][0]))),
            ("replace one chosen logprob by a double between two adjacent binary32 values (a re-typed / rounded value)",
             lambda rs: rs[2 % n]["chosen_logprobs"].__setitem__(-1, between(rs[2 % n]["chosen_logprobs"][-1]))),
            ("set one chosen logprob to null", lambda rs: rs[3 % n]["chosen_logprobs"].__setitem__(0, None)),
            ("drop chosen_logprobs from one request", lambda rs: rs[0].pop("chosen_logprobs")),
        ):
            rs = json.loads(json.dumps(targets))
            mutate(rs)
            others = [r for r in rows if r["request_id"] not in {t["request_id"] for t in targets}]
            tmp.write_text("".join(json.dumps(r) + "\n" for r in others + rs))
            r = output_binding(reader, commitments, tmp)
            variants.append({"variant": label, "outcome": "rejected" if not r.ok else "accepted", "detail": r.detail[:300]})
        tmp.unlink(missing_ok=True)
        return variants

    def needs_checkpoint(nid: str, fn):
        def run():
            if cp is None:
                raise RuntimeError(f"{nid} needs the pinned checkpoint (cp); none given")
            return fn()
        return run

    attempt("N15", needs_checkpoint("N15", n15))
    attempt("N16", needs_checkpoint("N16", n16))
    attempt("N17", n17)
    attempt("N19", n19)
    return results
