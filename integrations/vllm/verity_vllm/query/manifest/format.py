"""The required-value manifest FORMAT: the vocabulary every manifest producer and consumer shares, whichever engine built the manifest.

A manifest (`verity-sweep/required-manifest/v1`) is one identity row per required Value keyed by `IDENTITY_KEY`, digested by
`manifest_digest`; this module holds that encoding and the readers built on it, and nothing that DECIDES which Values are required:

* identity encoding  -- `SCHEMA`, `IDENTITY_KEY`, `FAMILIES`, the TP member spellings, `row_key` / `canonical_identity` / `manifest_digest`,
                        `components_fold_digest` (a B>=2 manifest's program digest before GP-01's), `request_label` (`r<k>`)
* Program names      -- `split_name` / `spec_family` / `spec_statics` over an `instances.json.gz` row, `collector_path` / `wrapper_prefix_of`
                        (Program module path -> the implementation's `named_modules()` path), `segment_steps` at the sampling events
* stream geometry    -- the FA2 / FA3 M1 launch geometry and word count, the fused-MoE block planes (`moe_geometry` / `moe_identities`)
* manifest rows      -- the lifted dtypes of the padded construction, the promoted-row observation rule (`promoted_observation`: a pre-flip
                        v1 manifest's `promoted: true` rows), the committed-name reader and the layout-level `coverage_check`
* Build inputs       -- `load_program`, `program_dirs_for` / `programs_for` (one request Program per declared (prompt_len, cap-1) shape),
                        `RegistryUnavailable` (the environment fault a manifest build refuses under)

Which Values are required is the query's: `verity_vllm.query` (`required.QUERY_OF_RECORD`, Q_word v1 since the re-baseline; TP rank Programs
too, with its TP-04 policy).  Dependency-free (json/gzip/hashlib).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
from typing import Any, Callable, Iterable

SCHEMA = "verity-sweep/required-manifest/v1"

IDENTITY_KEY = ("program_digest", "request_id", "step", "invocation", "op_path", "output_member", "element_range", "rank")

#: [R15 tp TP-04] `tp_rank_partials`: the rank-LOCAL input of every tensor-parallel collective of a rank Program (`derive_step --tp W --rank r`):
#: the row-parallel Gemm partial before its all-reduce (`part_rank<r>`), the masked embedding shard (`part_rank<r>`), the lm_head logits
#: shard before its all-gather (`shard_rank<r>`).  Required IN ADDITION to the collective's output (the module's bound member): the output
#: is `AllReduce2/AllGather2(part_rank0, part_rank1)`, a value both ranks receive, while the partial is the rank's own computation --
#: conservative reading (both required) until a DECISION says the collective output may be a validated derived value.
FAMILIES = ("input_token_ids", "instance_outputs", "sampled_token_ids", "weights", "fa2_hidden_m1_stream", "fa3_hidden_m1_stream", "moe_block_stream",
            "tp_rank_partials",                 # [R15 tp TP-04]
            "sampler_seed",                     # [R15 stoch] additive: only Programs with a stochastic selector emit it  [rel int-3: union, manbind M-0247 T3]
            "sampler_splits",                   # [R16 reqval REQ-03, stoch M-0254 derived S] additive: only a Program whose `splits` Input a stochastic selector reads
            "norm_scales",                      # opt-in (query.norm_scales): one f32 per norm Call per row, `<norm module>/scale`; never the record
            "router_softmax",                   # opt-in (query.router_softmax): the ordered router's softmax per token row, `moe/L<k>/router_softmax`
            "vocab_range",                      # opt-in (query.vocab_range): a TP rank's range values per token row, `<embedding module>/vocab_range`
            "scale_products",                   # additive (query.scale_products): a shared-scale FP8 GEMM's products per row, `<linear module>/scale_products`
            "call_boundaries")                  # Q_word: a Value two Calls of one module body exchange, `<module>/<fx node>[/<port>]`

TP_PARTIAL_MEMBER = "part_rank{rank}"

TP_SHARD_MEMBER = "shard_rank{rank}"

_ROW_RE = re.compile(r"\[(\d+)\]$")


def spec_family(spec: str) -> str:
    return spec.split("{", 1)[0]


def spec_statics(spec: str) -> dict[str, Any]:
    """`Gemm_v1{K=576,N=960}` -> {"K": 576, "N": 960} (integers; other values kept as strings)."""
    if "{" not in spec:
        return {}
    body = spec[spec.index("{") + 1: spec.rindex("}")]
    out: dict[str, Any] = {}
    depth, key, buf = 0, None, ""
    for ch in body + ",":
        if ch in "{[(":
            depth += 1
        elif ch in "}])":
            depth -= 1
        if ch == "," and depth == 0:
            if key is not None:
                out[key] = int(buf) if buf.lstrip("-").isdigit() else buf
            key, buf = None, ""
            continue
        if ch == "=" and depth == 0 and key is None:
            key, buf = buf, ""
            continue
        buf += ch
    return out


def split_name(name: str) -> tuple[str, str, str, int | None]:
    """Program instance name -> (module path, fx node, member hint, row).  `model.model.layers.0.self_attn.rotary_emb/rotary_embedding_3/key[7]`
    -> ("model.model.layers.0.self_attn.rotary_emb", "rotary_embedding_3", "key", 7); `argmax_2[0]` -> ("", "argmax_2", "", 0)."""
    row = None
    m = _ROW_RE.search(name)
    if m:
        row = int(m.group(1))
        name = name[: m.start()]
    parts = name.split("/")
    if len(parts) == 1:
        return "", parts[0], "", row
    mod, node = parts[0], parts[1]
    hint = parts[2] if len(parts) > 2 else ""
    return mod, node, hint, row


def collector_path(program_module: str, wrapper_prefix: str) -> str:
    """Program module path (under the export wrapper) -> the implementation's `named_modules()` path (the collector's op_path)."""
    if wrapper_prefix and program_module.startswith(wrapper_prefix):
        return program_module[len(wrapper_prefix):]
    return program_module


def wrapper_prefix_of(rows: Iterable[dict], attention_paths: Iterable[str]) -> str:
    """The export wrapper's attribute prefix: the Program names `model.model.layers.0.self_attn.attn`, the implementation
    `model.layers.0.self_attn.attn` (result.json `attention_impls_observed`).  Determined, not assumed."""
    attn = set(attention_paths or ())
    for r in rows:
        mod, _n, _h, _row = split_name(r.get("name") or "")
        if mod and spec_family(r["spec"]).startswith("Attention"):
            for a in attn:
                if mod.endswith(a) and len(mod) > len(a):
                    return mod[: len(mod) - len(a)]
            return "model." if mod.startswith("model.model.") else ""
    return "model."

#: [revb F-r17b-52 (iii)] the padded slot units are LIFTED words of the construction (MOE-PAD-SPEC; moe's materialisation: "one machine word per
#: lifted word: Value<17> -> u32, Value<33> -> u64; tag not split from payload"; pad's enc_bot(w) = 1 << w, Active(x) = x): the manifest names
#: the lifted dtype so the store is read at the plane's word width and the compare knows the tag.  `l17` = Value<17> (bf16 payload, u32
#: word, bot 0x1_0000); `l33` = Value<33> (f32 / i32 payload, u64 word, bot 0x1_0000_0000).
LIFTED_BF16 = "l17"

LIFTED_F32 = "l33"

LIFTED_DTYPES = {LIFTED_BF16: (16, 4), LIFTED_F32: (32, 8)}        # dtype -> (payload bits w, machine word bytes); bot = 1 << w


def lifted_bot(dtype: str) -> int:
    w, _b = LIFTED_DTYPES[dtype]
    return 1 << w

#: the ONE member a promoted row's output is named as -- a fixed convention: the row's Definition returns one array, the module vocabulary's
#: word for a single returned tensor is `out` (Gemm under a LogitsProcessor, RMSNormTriton, SiluMul, Attention, Embedding); a promoted member
#: collides with a vocabulary member of the same module only by name, and the collision is refused (fail closed), never merged
PROMOTED_MEMBER = "out"

#: the boundary class of a promoted row: the manifest row's `boundary_class`
INTERIOR_CROSSING = "interior_crossing"


def promoted_annotations(d: dict[str, Any]) -> dict[str, Any]:
    """The additive manifest-row fields of a promoted identity (a pre-flip v1 manifest's `promoted: true` row)."""
    return {"rows": d["rows"], "width": d["width"], "dtype": d["dtype"], "spec": d["spec"], "promoted": True, "boundary_class": INTERIOR_CROSSING,
            "consumers": list(d["consumers"])}

#: the fail class `row_pod.sh` / the harness classify a RegistryUnavailable under (precheck vocabulary: an environment fault, not a Commit verdict)
REGISTRY_UNAVAILABLE_FAIL_CLASS = "PRECHECK_FAIL_REGISTRY_UNAVAILABLE"

REGISTRY_UNAVAILABLE_EXIT = 3                                                    # row_pod.sh's PRECHECK_FAIL_* exit status


class RegistryUnavailable(ImportError):
    """The core `verity` package (verity.ml.circuits.defs REGISTRY) is not importable where the manifest is being built.  An ENVIRONMENT fault
    (infra / precondition: PYTHONPATH, the shipped tree's root, an incomplete venv), never a semantic result: without the
    registry no Definition's signature can be derived, so a manifest built here would misreport every promoted family as `unmodelled`
    (complete: false) -- the wire lane saw exactly that on #57 (163052 rows complete:false vs 163484 complete:true).  Raised instead of
    building; the CLI exits REGISTRY_UNAVAILABLE_EXIT naming REGISTRY_UNAVAILABLE_FAIL_CLASS."""

    fail_class = REGISTRY_UNAVAILABLE_FAIL_CLASS
    kind = "infra/precondition"

    def __init__(self, cause: BaseException):
        super().__init__(f"{REGISTRY_UNAVAILABLE_FAIL_CLASS}: the core verity package is not importable here (verity.ml.circuits.defs: "
                         f"{type(cause).__name__}: {cause}); the required-value manifest cannot be built without the registry "
                         f"(sys.path lacks the tree's root? an incomplete venv?) -- an environment fault, not a manifest result")
        self.cause = cause

#: [R19 int-20, F-r19-int-20] the bytes per element of a promoted member's manifest dtype (PROMOTED_DTYPES)
PROMOTED_DTYPE_BYTES = {"bf16": 2, "f32": 4}

PROMOTED_OBSERVATION_RULE = ("a PROMOTED manifest row (`promoted: true`: an interior Program row whose output another module reads; identity (producer "
                             "module, `out`)) is observed at the INPUT of its first consumer in Program order (`consumers[0]`): the collector acquires "
                             "that module's first positional tensor argument as `<consumer>/in` (the existing input-acquisition mechanism, no longer "
                             "restricted to a module class) and that committed tensor binds the producer's identity -- the producer module's own "
                             "forward output never does (the promoted value is by definition not a module output).  One observation point per "
                             "promoted member; the table is refused by name when two producers share a first consumer, when a consumer's `/in` is "
                             "itself a manifest identity, or when a producer's first consumer differs between its rows.  A consumer that is not a "
                             "module of the built model, or whose first argument is not the value, leaves the identity unbound: coverage fails by "
                             "name, nothing is guessed.")


def promoted_observation(manifest: dict[str, Any]) -> dict[str, Any]:
    """[R19 int-20] From the required manifest's promoted rows: the committed names the collector must acquire and the identities they bind
    (PROMOTED_OBSERVATION_RULE).  Returns {"names": {"<consumer>/in": [producer op_path, member]}, "consumers": {consumer op_path: bytes per
    token of the acquired input}, "producers": {producer: {"consumer", "width", "dtype", "spec", "rows_n", "member"}}, "problems": [text]} --
    `problems` non-empty = the table is not usable (the caller refuses by name).  The one definition the collector's arming
    (bench.commit_delta via binding.promoted_input_acquisition), the binding map and the layout-level `coverage_check` share."""
    names: dict[str, list[str]] = {}
    consumers: dict[str, int] = {}
    producers: dict[str, dict[str, Any]] = {}
    problems: list[str] = []
    rows = manifest.get("identities") or manifest.get("entries") or manifest.get("rows") or []
    own_inputs = {str(r.get("op_path")) for r in rows if str(r.get("output_member")) == "in" and not r.get("promoted")}
    for r in rows:
        if not r.get("promoted"):
            continue
        op = str(r.get("op_path"))
        member = str(r.get("output_member", PROMOTED_MEMBER))
        cons = [str(c) for c in (r.get("consumers") or [])]
        if not cons:
            problems.append(f"promoted row {op}/{member} (spec {r.get('spec')}) names no consumer: no observation point")
            continue
        first = cons[0]
        w, dt = r.get("width"), str(r.get("dtype"))
        bpe = PROMOTED_DTYPE_BYTES.get(dt)
        if not isinstance(w, int) or isinstance(w, bool) or w <= 0 or bpe is None:
            problems.append(f"promoted row {op}/{member}: width {w!r} / dtype {dt!r} not usable for the arena bound (PROMOTED_DTYPE_BYTES {sorted(PROMOTED_DTYPE_BYTES)})")
            continue
        p = producers.get(op)
        if p is None:
            producers[op] = {"consumer": first, "width": int(w), "dtype": dt, "spec": r.get("spec"), "rows_n": 1, "member": member}
        else:
            p["rows_n"] += 1
            if p["consumer"] != first:
                problems.append(f"promoted rows of {op}/{member} name different first consumers ({p['consumer']!r} vs {first!r}): one observation point per member")
            if p["member"] != member:
                problems.append(f"promoted rows of {op} name different members ({p['member']!r} vs {member!r})")
            if (p["width"], p["dtype"]) != (int(w), dt):
                problems.append(f"promoted rows of {op}/{member} disagree on the signature ({p['width']} {p['dtype']} vs {w} {dt})")
    for op, p in sorted(producers.items()):
        name = f"{p['consumer']}/in"
        if p["consumer"] in own_inputs:
            problems.append(f"promoted {op}/{p['member']}: its first consumer's input {name} is itself a manifest identity ({p['consumer']}, in)")
            continue
        other = names.get(name)
        if other is not None:
            problems.append(f"promoted {op}/{p['member']} and {other[0]}/{other[1]} share the first consumer {p['consumer']}: one committed {name} cannot bind two identities")
            continue
        names[name] = [op, p["member"]]
        consumers[p["consumer"]] = int(p["width"]) * PROMOTED_DTYPE_BYTES[p["dtype"]]
    return {"names": names, "consumers": consumers, "producers": producers, "problems": problems, "rule": PROMOTED_OBSERVATION_RULE}

MOE_LEAF = "experts"                    # vLLM FusedMoE module leaf (`model.layers.k.mlp.experts`); its interior is the MoE block stream

MOE_PLANES = ("router_w", "router_ids", "up", "act", "down", "sum")
#: block-FP8 experts (GLM-5.3's `TritonExperts` with DeepGEMM's ue8m0 quant): the w13 launch reads the input's e4m3 group quant
#: (`in_q`, `in_scale`), the w2 launch the activation's (`act_q`, `act_scale`), each a `_C.per_token_group_fp8_quant` under the runner
MOE_FP8_PLANES = ("router_w", "router_ids", "in_q", "in_scale", "up", "act", "act_q", "act_scale", "down", "sum")
#: Definition family prefix -> the MoE plane kind of its rows, first match wins (a `W` launch before its prefix, the fused SiLU quant
#: before SiluMul).  The kinds in `_RUNNER_KINDS` are plane rows only under the MoE runner module (`MOE_LEAF`).
MOE_PLANE_FAMILIES = (("MoeRouterTopK", "router"), ("MoeRouterSigmoidBias", "router"), ("MoeExpertGemmW", "down"),
                      ("MoeExpertFp8GemmW", "down"), ("MoeExpertGemm", "up"), ("MoeExpertFp8Gemm", "up"), ("MoeSum", "sum"),
                      ("SiluMulFp8BlockQuant", "act_quant"), ("SiluMul", "act"), ("Fp8GroupQuant", "quant"))
_RUNNER_KINDS = frozenset({"act_quant", "act", "quant"})


def _on_runner(module: str) -> bool:
    return module.rsplit(".", 1)[-1] == MOE_LEAF


def _group_static(g: dict[str, Any], st: dict[str, Any]) -> None:
    """An expert GEMM with a group `G` static reads block-FP8 operands (xq, sx)."""
    if "G" in st:
        g["fp8"], g["G"] = True, int(st["G"])


def moe_geometry(seg_rows: Iterable[dict]) -> dict[str, dict[str, Any]]:
    """[R14 moe] per MoE module of one step: M (tokens routed = router instances), TOPK, H, I, N_up from the Program's MoE instances.
    The block's values are produced INSIDE one module forward (`MoERunner`), so they are not module outputs: the query opens on them through
    the `moe_block_stream` family (MOE-03 source, names `moe/L<k>/<plane>`), one set of planes per MoE layer per step whatever the routing
    chose (pair-major rows p = t*TOPK + k -- the sorted block schedule never enters the values).  Expert GEMMs with a group `G` static
    (block FP8) set `fp8` and `G`; the runner's group-quant rows are counted by their row width: K = H the input's (`in_quant_rows`), else
    the activation's (`act_quant_rows`, with the fused SiLU quant's), and its SiluMul rows as `act_rows`."""
    geo: dict[str, dict[str, Any]] = {}
    quant_k: dict[str, list[int]] = {}
    for r in seg_rows:
        mod = split_name(r.get("name") or "")[0]
        kind = moe_plane_kind(spec_family(r["spec"]), mod)
        if kind is None:
            continue
        st = spec_statics(r["spec"])
        g = geo.setdefault(mod, {"M": 0, "TOPK": 0, "E": 0, "H": 0, "I": 0, "N_up": 0, "expert_gemm_rows": 0, "act_rows": 0,
                                 "fused_act_quant_rows": 0})
        if kind == "router":
            g["M"] += 1
            g["TOPK"] = int(st.get("TOPK", g["TOPK"]))
            g["E"] = int(st.get("E", g["E"]))
        elif kind == "down":
            g["H"] = int(st.get("N", g["H"]))
            g["I"] = int(st.get("K", g["I"]))
            _group_static(g, st)
        elif kind == "up":
            g["N_up"] = int(st.get("N", g["N_up"]))
            g["expert_gemm_rows"] += 1
            _group_static(g, st)
        elif kind == "sum":
            g["H"] = int(st.get("H", g["H"]))
            g["TOPK"] = int(st.get("TOPK", g["TOPK"]))
        elif kind == "act_quant":
            g["fused_act_quant_rows"] += 1
        elif kind == "act":
            g["act_rows"] += 1
        else:
            quant_k.setdefault(mod, []).append(int(st.get("K", 0)))
    for mod, g in geo.items():
        ks = quant_k.get(mod, [])
        n_in = sum(1 for k in ks if k == g["H"]) if g["H"] != g["I"] else min(len(ks), g["M"])
        g["in_quant_rows"], g["act_quant_rows"] = n_in, len(ks) - n_in + g.pop("fused_act_quant_rows")
    return geo


def moe_plane_kind(family: str, module: str) -> str | None:
    """[MAN-06f v3] the fused-MoE block row kinds whose values the collector commits as `moe_block_stream` PLANES (MOE-03: pair-major
    planes per MoE layer per step) rather than as module outputs (`MOE_PLANE_FAMILIES`): `router` (router_w + router_ids per token),
    `up` (the w13 launch per pair), `down` (the w2 launch per pair), `sum` (per token) and, under the MoE runner module `...experts`
    only, `act` (the SiluMul per pair), `quant` (a group quant: the input's per token or the activation's per pair -- `ProgramIndex`
    tells them apart by what they read) and `act_quant` (the activation's quant fused with the SiLU).  None for every other row.  From
    the Program's spec family + module name alone."""
    for prefix, kind in MOE_PLANE_FAMILIES:
        if family.startswith(prefix):
            return kind if kind not in _RUNNER_KINDS or _on_runner(module) else None
    return None


def moe_identities(g: dict[str, Any]) -> list[tuple[str, int, str]]:
    """(plane, elements, dtype) of the block-stream tensors for one MoE layer in one step: `MOE_PLANES`, or under block-FP8 experts the
    `MOE_FP8_PLANES` the module's rows produce (a quant or SiluMul the Program has no rows for is no plane)."""
    M, K, H, I, N = g["M"], g["TOPK"], g["H"], g["I"], g["N_up"]
    route = [("router_w", M * K, "f32"), ("router_ids", M * K, "i32")]
    tail = [("down", M * K * H, "bf16"), ("sum", M * H, "bf16")]
    if not g.get("fp8"):
        return route + [("up", M * K * N, "bf16"), ("act", M * K * I, "bf16")] + tail
    G = int(g.get("G") or 1)
    return (route + ([("in_q", M * H, "w8"), ("in_scale", M * H // G, "f32")] if g.get("in_quant_rows") else [])
            + [("up", M * K * N, "bf16")] + ([("act", M * K * I, "bf16")] if g.get("act_rows") else [])
            + ([("act_q", M * K * I, "w8"), ("act_scale", M * K * I // G, "f32")] if g.get("act_quant_rows") else []) + tail)


def fa2_geometry(T_max: int, rows: int, NH: int, KVH: int, D: int, n_seq: int, windowed: bool = False) -> tuple[int, int, int, int, str]:
    """(HB, M, NB, BN, name) of the FA2 launch the step's attention rows imply (mirror of `native_host.launch_words`).
    [R15 capacity/dense] `windowed`: the layer's attention has a sliding window (vLLM passes `window_size=(W-1, 0)` to FA2 whatever the
    request length) -> the pinned FA2 does NOT apply the GQA decode `seqlenq_ngroups_swapped` fold (flash_api.cpp:646; mirror of
    `hidden_source.fa2_swapped`): the decode launch is b*H slabs of one row (`g8x1x..` on Gemma-2's even layers), not `g4x2x..`."""
    from_words = None  # noqa: F841 (documentation: identical rule to hidden_source.stream_name / native_host.launch_words)
    BN = 128 if D <= 64 else 64                                                  # fa2_kblock_n(D, num_splits=1)
    seqlen_q = rows // max(1, n_seq)
    NB = -(-T_max // BN)
    if seqlen_q == 1 and NH > KVH and not windowed:                             # decode "swapped" kernel: rows = query heads of a KV group
        HB, M = n_seq * KVH, NH // KVH
    else:
        HB, M = n_seq * NH, seqlen_q
    base = f"g{HB}x{M}x{NB}"
    return HB, M, NB, BN, (base if (BN, D) == (128, 64) else f"{base}x{BN}x{D}")


def fa3_geometry(T_max: int, rows: int, NH: int, KVH: int, D: int, n_seq: int) -> tuple[int, int, int, int, str]:
    g = NH // max(1, KVH)
    seqlen_q = rows // max(1, n_seq)
    one_wg = seqlen_q * g <= 64
    BN = (192 if one_wg else 128) if D == 64 else 128
    NB = -(-T_max // BN)
    HB, M = n_seq * KVH, seqlen_q * g
    return HB, M, NB, BN, f"g{HB}x{M}x{NB}x{BN}x{D}"


def mat_total_words(HB: int, M: int, NB: int, BN: int, D: int, ms: bool = False) -> int:
    N = NB * BN
    return HB * M * N * 2 + HB * M * (N // 2) + HB * M * NB * 8 + HB * M * NB * D + HB * M * 4 + (HB * M * NB if ms else 0)


def load_program(program_dir: str) -> dict[str, Any]:
    from verity_vllm.program.compact import expanded_file_args
    with gzip.open(os.path.join(program_dir, "instances.json.gz"), "rt") as f:
        inst = json.load(f)
    for r in inst.get("rows") or ():   # a v2 file's argument progressions -> the v1 per-leaf run lists every consumer reads
        a = r.get("args")
        if a:
            r["args"] = expanded_file_args(a)
    res = json.load(open(os.path.join(program_dir, "result.json")))
    art = None
    p = os.path.join(program_dir, "artifact.json")
    if os.path.exists(p):
        art = json.load(open(p))
    return {"instances": inst, "result": res, "artifact": art, "dir": program_dir}

#: the sampling event that ends an engine step: greedy `TokenSelect_v1`, and the stochastic / explicit-argmax selectors of `registry.sampling`
#: (`GumbelTokenSelect_v1{V}` = temperature + top-p + Gumbel argmax with the seed as a registered runtime input; `ArgmaxTokenSelect_v1`) --
#: every one of them samples ONE id per sequence per step (`sampled_token_ids`); the chain's interior values are the sampler's, not module outputs.
TOKEN_SELECT_FAMILIES = ("TokenSelect", "GumbelTokenSelect", "ArgmaxTokenSelect", "GumbelTopPTokenSelect")

#: [R15 stoch M-0107] the stochastic selectors (`Gumbel*TokenSelect_v1`: temperature -> top-p mask -> Gumbel noise from Philox(seed, pos) ->
#: argmax) consume ONE registered runtime input the greedy chain has not: the per-request SEED.  It is a required value of the sampling
#: event (identity `runner.sampler/seed`, one i64 per sequence per step, family `sampler_seed`): without it the committed sampled id is not
#: reproducible from the committed logits, so the population REQUIRES it and a collector that does not bind it fails coverage (named).
STOCHASTIC_SELECT_PREFIXES = ("Gumbel",)

#: [R16 reqval REQ-03, stoch M-0254 derived S] the derived-S Program has a second request-owned runtime input beside the seed: `splits`
#: (`Array<T+1,Value<32>>`, one i32 per sampling event = the split count S the top-p mask kernel launched with, `SplitsFor_v1(n_live, num_SMs)`),
#: read by every `GumbelTopPTokenSelect_v1{V}` row as its 6th operand.  It is a required value of the sampling event exactly as the seed is
#: (identity `runner.sampler/splits`, one i32 per sequence per step, family `sampler_splits`): the committed sampled id is reproducible from
#: the committed logits only with the seed AND the S the mask used (F-stoch-01 is a wrong S).  Emitted iff the Program HAS the `splits`
#: parameter (R15 static-form `{V,S}` Programs and greedy Programs have none: byte-for-byte unchanged).
SPLITS_PARAM = "splits"


def has_splits_param(params) -> bool:
    return any(str(n) == SPLITS_PARAM for n, _t in (params or []))


def is_token_select(family: str) -> bool:
    """A sampling event = the spec family's base name (before `_v<n>`) ends with `TokenSelect` (TokenSelect_v1, GumbelTokenSelect_v1,
    GumbelTopPTokenSelect_v1, ArgmaxTokenSelect_v1, ...): every one samples one id per sequence and closes the engine step."""
    base = family.split("_v", 1)[0]
    return base.endswith("TokenSelect") or family.startswith(TOKEN_SELECT_FAMILIES)


def is_stochastic_select(family: str) -> bool:
    return is_token_select(family) and family.startswith(STOCHASTIC_SELECT_PREFIXES)


def segment_steps(rows: list[dict]) -> list[tuple[int, int]]:
    """Engine steps as [lo, hi] inclusive index ranges at `TokenSelect` boundaries (one sampling event per step; `cb_a.compare._segment`)."""
    segs, lo = [], 0
    for r in rows:
        if is_token_select(spec_family(r["spec"])):
            segs.append((lo, r["i"]))
            lo = r["i"] + 1
    if lo < len(rows):
        segs.append((lo, len(rows) - 1))                                          # trailing instances without a sampling event (padded wrapper)
    return segs


def layer_of(module: str) -> int:
    return int(module.split(".layers.")[1].split(".")[0]) if ".layers." in module else -1


def alias_family(member: str) -> str:
    """The manifest family of a member emitted through an alias: a rank-local partial member is `tp_rank_partials`, anything else an
    instance output (TP-04; the shared-RoPE members are `0` / `1`)."""
    return "tp_rank_partials" if member.startswith(("part_rank", "shard_rank")) else "instance_outputs"

REQ_RE = re.compile(r"^r(\d+)$")


def request_label(rid: Any, workload: dict[str, Any] | None = None, *, position: int | None = None) -> str:
    """[R15 COORD ruling 2] the identity key's `request_id` is `r<k>`, k = the request's position (`index`) in the workload manifest --
    the convention BIND-01's map and the run header use.  A UUID is a WORKLOAD id (`requests[i].request_id`), never a request id:
    MAN-01 c41cae46a emitting it broke the rc2 canary (F-rel-01: coverage 303/12943, only the request-less weights matched).
    Accepts `r<k>`, an int / digit string k, or a workload id resolvable through `workload["requests"]`; anything else is an error
    (fail closed -- a manifest keyed by an id no map can carry is a manifest that covers nothing)."""
    reqs = (workload or {}).get("requests") or []
    if rid is None:
        if position is not None:
            return f"r{int(position)}"
        if reqs and isinstance(reqs[0], dict):
            return f"r{int(reqs[0].get('index', 0))}"
        return "r0"
    if isinstance(rid, bool):
        raise ValueError(f"request id {rid!r}: not a request index")
    if isinstance(rid, int):
        return f"r{rid}"
    s = str(rid)
    if REQ_RE.match(s):
        return s
    if s.isdigit():
        return f"r{int(s)}"
    for i, r in enumerate(reqs):
        if isinstance(r, dict) and str(r.get("request_id")) == s:
            return f"r{int(r.get('index', i))}"
    raise ValueError(f"request id {s!r} is not `r<k>` and is not a request of the workload manifest (a workload UUID is not a request id: COORD R15 ruling 2)")


def row_key(r: dict[str, Any]):
    s = r["step"]
    e = r.get("engine_step", s)
    return (r["rank"], (1, 0) if e == "*" else (0, int(e)), str(r["request_id"]), (1, 0) if s == "*" else (0, int(s)), r["op_path"], r["output_member"], r["invocation"])


def canonical_identity(r: dict[str, Any]) -> list:
    """The identity key tuple (JSON list) of a manifest row -- what bind's binding map is keyed by."""
    return [r["program_digest"], r["request_id"], r["step"], r["invocation"], r["op_path"], r["output_member"], list(r["element_range"]), r["rank"]]


def components_fold_digest(component_digests: Iterable[str]) -> str:
    """The B>1 manifest's `program_digest` when the Build has not (yet) produced gprog's Workload_v1 digest: a sha256 fold of the sorted
    request-Program digests -- one string (the identity key needs a string), labelled `program_digest_kind: components-fold` beside it,
    replaced by the workload digest (`--workload-digest`, kind `workload`) once the Build carries it."""
    return hashlib.sha256(("workload-components/" + ",".join(sorted(str(d) for d in component_digests))).encode()).hexdigest()


def tp_ranks_fold_digest(rank_digests: Iterable[str]) -> str:
    """[R15 tp TP-04, rev-cb M-0108] A merged TP manifest's header `program_digest`: the labelled fold of the per-rank Program digests
    (`program_digest_kind: tp-ranks-fold`) -- one string for manbind's loader; every identity row keeps its own rank's digest."""
    return hashlib.sha256(("tp-ranks/" + ",".join(sorted(str(d) for d in rank_digests))).encode()).hexdigest()


TP_PEER_BINDING_RULE = ("a rank Program's collective row (AllReduce2 / AllGather2) consumes the rank's OWN partial and the RECEIVED peer partial "
                        "(its `peers_<k>` input); the received value IS the peer rank's own partial identity (`tp_rank_partials`: part_rank<peer> / "
                        "shard_rank<peer>) at the same (request, step, invocation, module) -- so the merged population binds them identity for "
                        "identity in both directions: a collective output without the peer's partial, or a partial without the peer's collective "
                        "output, is named and the merge is not complete; the element RANGE binds too (an all_reduce partial is the output's full "
                        "width, an all_gather shard is 1/world of it; `width_checked` counts the pairs both sides sized) -- the identity level; "
                        "the committed BYTES are bound at Commit by D90 rank_match.peer_binding / TP-12, not here.  [revb F-r17b-17] `invocation` "
                        "is 0 for every (module, member, step) on both engines (the shared-RoPE alias is the one exception: layer order): a module "
                        "path issued TWICE in one step is ONE identity whose element_range is both calls' rows, not two identities -- the binding "
                        "here sees one pair per (step, module) and says bound; the second issue fails one stage later at `coverage_check`, whose "
                        "k-th committed occurrence binds to invocation k with the FIRST call's numel < the identity's hi -> MISSING by name")


def tp_peer_binding(parts: list[dict[str, Any]], collective_kind: Callable[[str], str | None]) -> dict[str, Any]:
    """[R17 reqval finding 7] The TP merge's identity-level cross-rank check (see `TP_PEER_BINDING_RULE`).  `parts` are the rank manifests
    (one per rank); `collective_kind` classifies a row's `spec` ("all_reduce" / "all_gather" / None: the registry's, passed in so this module
    stays dependency-free).  Returns {world, collective_outputs, peer_partials, bound, n_unbound, unbound: [first 16 by name], rule}."""
    world = max(int(p.get("ranks") or 1) for p in parts)

    def _width(r: dict[str, Any]) -> int | None:
        er = r.get("element_range")
        return int(er[1]) - int(er[0]) if isinstance(er, (list, tuple)) and len(er) == 2 else None
    partials: dict[tuple, int | None] = {}                        # (request, step, invocation, op_path, member, rank) -> element width
    outputs: list[tuple] = []                                     # (request, step, invocation, op_path, rank, kind, element width)
    for p in parts:
        for r in p["identities"]:
            if r["family"] == "tp_rank_partials":
                partials[(str(r["request_id"]), int(r["step"]), int(r.get("invocation", 0)), r["op_path"], r["output_member"], int(r["rank"]))] = _width(r)
            elif r["family"] == "instance_outputs" and collective_kind(str(r.get("spec") or "")) is not None:
                outputs.append((str(r["request_id"]), int(r["step"]), int(r.get("invocation", 0)), r["op_path"], int(r["rank"]),
                                collective_kind(str(r["spec"])), _width(r)))
    unbound: list[str] = []
    bound = 0
    width_checked = 0
    out_keys: set[tuple] = set()
    for rq, s, inv, op, rk, kind, w_out in outputs:
        out_keys.add((rq, s, inv, op, rk))
        member = (TP_SHARD_MEMBER if kind == "all_gather" else TP_PARTIAL_MEMBER)
        for peer in range(world):
            if peer == rk:
                continue
            key = (rq, s, inv, op, member.format(rank=peer), peer)
            if key not in partials:
                unbound.append(f"rank {rk} {op} step {s} {kind} output: no {member.format(rank=peer)} identity on rank {peer} (request {rq})")
                continue
            # [revb M-0143 (a)] the RANGE binds too: an all_reduce partial is the output's full width; an all_gather shard is 1/world of it
            w_part = partials[key]
            if w_out is not None and w_part is not None:
                width_checked += 1
                want = w_out if kind == "all_reduce" else (w_out // world if w_out % world == 0 else None)
                if want is None or w_part != want:
                    unbound.append(f"rank {rk} {op} step {s} {kind} output width {w_out}: {member.format(rank=peer)} on rank {peer} is {w_part} elements, "
                                   f"not {want if want is not None else f'{w_out}/{world}'} (request {rq})")
                    continue
            bound += 1
    # the reverse direction: a partial whose collective output the peer does not carry (a peer part without the collective row)
    for rq, s, inv, op, member, rk in sorted(partials):
        for peer in range(world):
            if peer != rk and (rq, s, inv, op, peer) not in out_keys:
                unbound.append(f"rank {rk} {op} step {s} partial {member}: no collective output identity on rank {peer} (request {rq})")
    return {"world": world, "collective_outputs": len(outputs), "peer_partials": len(partials), "bound": bound, "width_checked": width_checked,
            "n_unbound": len(unbound), "unbound": sorted(unbound)[:16], "rule": TP_PEER_BINDING_RULE}


def manifest_digest(identities: list[dict[str, Any]]) -> str:
    """sha256 of the canonical JSON of the sorted identity tuples (+ family), independent of the annotations."""
    h = hashlib.sha256()
    for r in sorted(identities, key=row_key):
        h.update(json.dumps([r["family"]] + canonical_identity(r), separators=(",", ":"), sort_keys=True).encode())
        h.update(b"\n")
    return h.hexdigest()


def committed_identity(name: str) -> tuple[str, str] | None:
    """Committed tensor name -> (op_path, output_member) in the manifest vocabulary; None for extra committed data that binds no required
    identity (`/row4_f32` regenerate baselines)."""
    if name.startswith(("fa2.m1/", "fa3.m1/")):
        tap, rest = name.split("/", 1)
        layer, geo = rest.split("/", 1)
        return f"{tap}/{layer}", geo
    if name.startswith("moe/"):
        _m, layer, plane = name.split("/", 2)
        return f"moe/{layer}", plane
    if name.endswith("/row4_f32"):
        return None
    if "/" not in name:
        return name, "out"
    base, member = name.rsplit("/", 1)
    return base, member


def coverage_check(manifest: dict[str, Any], layouts: dict[int | str, list[dict[str, Any]]], *, request_id: str | int | None = None,
                   rank: int = 0, max_report: int = 12) -> dict[str, Any]:
    """Every manifest identity of (`request_id`, `rank`) must have a committed tensor of the same (step, op_path, output_member,
    invocation) whose element range CONTAINS the manifest's; `layouts` = {step: [{"name", "numel"|"shape", ...} in layout order]}.
    Reports missing identities (by family / step), and the committed extras (allowed).  Identity coverage only -- value correspondence
    is bind's separate check."""
    from collections import Counter

    if manifest.get("complete") is False:                                        # [MAN-02] an incomplete manifest never yields coverage
        um = manifest.get("unmodelled") or {}
        n = len(manifest.get("identities") or [])
        return {"ok": False, "error": "manifest incomplete: unmodelled Program families " + "; ".join(f"{k} x{v}" for k, v in sorted(um.items())),
                "manifest_digest": manifest.get("manifest_digest"), "checked": n, "missing_n": n, "missing_by_family": {"(unmodelled)": n},
                "checked_by_family": {}, "families_missing_entirely": sorted(um), "steps_with_missing": [], "missing": [],
                "committed_unmatched_n": 0, "committed_unmatched": [], "committed_extras_n": 0, "duplicate_names_n": 0, "steps_committed": [], "unmodelled": um}
    aliases = manifest.get("op_path_aliases") or {}
    # [R19 int-20] a PROMOTED identity (producer module, `out`) is committed under its first consumer's input name (`<consumer>/in`,
    # PROMOTED_OBSERVATION_RULE): that name binds the producer identity here; a committed tensor named for the producer module itself (its
    # forward output -- by definition NOT the promoted value) binds nothing and is counted under `promoted_misbound`
    promoted_keys = {(str(r["op_path"]), str(r["output_member"])) for r in manifest.get("identities") or [] if r.get("promoted")}
    rebind: dict[str, tuple[str, str]] = {}
    if promoted_keys:
        rebind = {k: (v[0], v[1]) for k, v in promoted_observation(manifest)["names"].items()}
    misbound: list[str] = []
    have: dict[tuple, list[int]] = {}
    extras = 0
    for step, metas in layouts.items():
        seen: Counter = Counter()
        for m in metas:
            ident = rebind.get(m["name"]) or committed_identity(m["name"])
            if ident is None:
                extras += 1
                continue
            if ident in promoted_keys and m["name"] not in rebind:
                misbound.append(m["name"])
                extras += 1
                continue
            op, member = ident
            if op in aliases:                                  # a collector that names the shared module per layer -> canonical path + invocation
                op, inv = aliases[op][0], int(aliases[op][1])
            else:
                inv = seen[(op, member)]
                seen[(op, member)] += 1
            numel = m.get("numel")
            if numel is None:
                numel = 1
                for d in m.get("shape") or []:
                    numel *= int(d)
            have.setdefault((int(step), op, member, inv), []).append(int(numel))
    missing: list[dict[str, Any]] = []
    by_family_missing: Counter = Counter()
    by_family_checked: Counter = Counter()
    matched_keys: set[tuple] = set()
    for r in manifest["identities"]:
        if request_id is not None and str(r["request_id"]) not in (str(request_id), "*"):
            continue
        if int(r["rank"]) != int(rank):
            continue
        if r["family"] == "weights":
            continue                                       # registered inputs: checked against the weights root registration, not the step layouts
        by_family_checked[r["family"]] += 1
        k = (int(r.get("engine_step", r["step"])), r["op_path"], r["output_member"], int(r["invocation"]))
        lo, hi = r["element_range"]
        got = have.get(k)
        seq_short = None
        if got is None and r.get("request_id") == "*" and isinstance(r.get("requests"), list) and r["requests"]:
            # [R15 B8] a batch-level hidden row (B>1: one attention launch per layer per engine step over the co-scheduled sequences) names
            # the padded whole-step geometry; the tap commits per-SEQUENCE planes (`s<i>/g<HB>x<M>x<NB>/<plane>`) at a segmented prefill, or
            # one whole-launch plane set at decode.  The requirement is op-level per engine step: every co-scheduled sequence has its planes
            # (>= n_seq distinct `s<i>` prefixes), or the launch is whole (unsegmented members present).  Same rule as binding.coverage_check.
            seqs, whole, numels = set(), False, []
            for (st, op, mem, inv), v in have.items():
                if st == k[0] and op == k[1] and inv == k[3]:
                    numels += v
                    if mem.startswith("s") and "/" in mem and mem[1:mem.index("/")].isdigit():
                        seqs.add(mem[:mem.index("/")])
                    else:
                        whole = True
            if numels and (whole or len(seqs) >= len(r["requests"])):
                got, hi = numels, None                                       # op-level: covered; matched below
                for kk in list(have):
                    if kk[0] == k[0] and kk[1] == k[1] and kk[3] == k[3]:
                        matched_keys.add(kk)
            elif numels:
                got, seq_short = None, f"{len(seqs)} sequence plane sets committed, {len(r['requests'])} co-scheduled sequences declared ({sorted(seqs)})"
        if not got or (hi is not None and max(got) < int(hi)):
            missing.append({"family": r["family"], "step": r["step"], "engine_step": k[0], "request_id": r["request_id"], "op_path": r["op_path"],
                            "output_member": r["output_member"], "invocation": r["invocation"], "want_range": [lo, hi], "committed_numel": got}
                           | ({"why": seq_short} if seq_short else {}))
            by_family_missing[r["family"]] += 1
        else:
            matched_keys.add(k)
    unexpected = [k for k in have if k not in matched_keys]
    dup = {k: len(v) for k, v in have.items() if len(v) > 1}
    families_missing_whole = [f for f, n in by_family_checked.items() if by_family_missing.get(f, 0) == n]
    steps_missing_whole = sorted({int(r["engine_step"]) for r in missing}) if missing else []
    requests_missing = sorted({str(r["request_id"]) for r in missing}) if missing else []
    # [R15 B8] global manifests address request-locally, so a layouts-only check cannot see WHICH request a block belongs to; it can and
    # must see the declared composition's size per engine step: the packed ids member has exactly the declared rows and the sampled-ids
    # member exactly one slot per co-scheduled sequence -- a request the server never ran shortens both (named per engine step).  The
    # per-request partition itself is the binding map's `composition_check`.
    composition: list[dict[str, Any]] = []
    if manifest.get("global") and manifest.get("engine_steps") and request_id is None:
        ref = manifest.get("composition_reference") or {}
        for es in manifest["engine_steps"]:
            e = int(es["engine_step"])
            want_rows, want_seq = int(es.get("rows", sum(int(m["rows"]) for m in es["members"]))), int(es.get("n_seq", len(es["members"])))
            for fam_name, want in (("input_token_ids", want_rows), ("sampled_token_ids", want_seq)):
                rf = ref.get(fam_name)
                if not rf:
                    continue
                got = have.get((e, rf["op_path"], rf["output_member"], 0))
                if got is None:
                    continue                                # the identities loop already reported the member as missing
                if max(got) != want * int(rf.get("per_row", 1)):
                    composition.append({"engine_step": e, "member": f"{rf['op_path']}/{rf['output_member']}", "declared": want, "committed": max(got),
                                        "requests": [m["request_id"] for m in es["members"]]})
    return {"ok": not missing and not composition, "manifest_digest": manifest["manifest_digest"], "checked": sum(by_family_checked.values()),
            "missing_n": len(missing), "missing_by_family": dict(by_family_missing), "checked_by_family": dict(by_family_checked),
            "families_missing_entirely": families_missing_whole, "steps_with_missing": sorted(set(steps_missing_whole) | {c["engine_step"] for c in composition})[:64],
            "requests_with_missing": requests_missing[:64],
            "missing": missing[:max_report], "committed_unmatched_n": len(unexpected), "committed_unmatched": [list(k) for k in unexpected[:max_report]],
            "committed_extras_n": extras, "duplicate_names_n": len(dup), "steps_committed": sorted(int(s) for s in layouts),
            "composition_mismatch_n": len(composition), "composition_mismatch": composition[:max_report],
            **({"promoted_observed_as": dict(sorted((k, list(v)) for k, v in rebind.items())), "promoted_misbound_n": len(misbound),
                "promoted_misbound": sorted(set(misbound))[:max_report]} if promoted_keys else {})}


def programs_for(fx: dict[str, Any], programs_root: str) -> dict[str, Any]:
    """The request Program per declared request (build-global / verify), loaded: see `program_dirs_for` for the lookup."""
    return {rid: load_program(d) for rid, d in program_dirs_for(fx, programs_root).items()}


def program_dirs_for(fx: dict[str, Any], programs_root: str) -> dict[str, str]:
    """The request Program DIR per declared request: `*LP<prompt_len>_T<max_tokens-1>` under programs_root, else any
    build_request*/result.json declaring exactly (LP, T) (the row's default request Program); never a near shape.  [R16 reqval] the
    traversal engine streams these (rows never all in memory: 57 MB vs 1 GB peak on the Llama B8 fold, same digest)."""
    import glob
    progs: dict[str, str] = {}
    for r in fx["requests"]:
        lp, t = int(r["prompt_len"]), int(r["max_tokens"]) - 1
        pat = os.path.join(programs_root, f"*LP{lp}_T{t}")
        hits = sorted(glob.glob(pat))
        if not hits:
            for cand in sorted(glob.glob(os.path.join(programs_root, "build_request*"))):
                rj = os.path.join(cand, "result.json")
                if os.path.isdir(cand) and os.path.exists(rj):
                    try:
                        res = json.load(open(rj))
                    except Exception:  # noqa: BLE001
                        continue
                    if res.get("ok") and int(res.get("LP", -1)) == lp and int(res.get("T", -1)) == t:
                        hits = [cand]
                        break
        if not hits:
            raise SystemExit(f"no request Program for {r['request_id']} at {pat} (nor a build_request*/result.json declaring LP={lp} T={t})")
        progs[r["request_id"]] = hits[0]
    return progs
