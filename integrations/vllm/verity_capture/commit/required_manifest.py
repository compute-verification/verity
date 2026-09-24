"""R14 lane manifest (MAN-01): the REQUIRED-VALUE MANIFEST -- the expected required-value population, derived from the Program and the
fixed canonical VU query --
never from the collector's attempted acquisitions and never from the first observed step (reviewer R-14: a producer that never ran
contributed zero to both "expected" and "committed"; F-rev-01: dropping output Value `/1` and binding `/0` twice keeps counts + classes).

One identity row per required Value (glossary: required values = boundary values ∪ prescribed inputs ∪ protocol-required values),
in the key agreed with lane bind (`out/gen/r14/TABLE-RULES.md`):

    (program_digest, request_id, step, invocation, op_path, output_member, element_range, rank)

`op_path` is the producing module's path in the implementation's vocabulary (`model.layers.3.self_attn.qkv_proj`; the collector's
`named_modules()` names), hidden families use the tap's path (`fa2.m1/L3`, `fa3.m1/L3`), the sampler `runner.sampler`.  `output_member`
is `k` for tuple slot k, `out` for a single tensor, `in` for the acquired embedding ids, `sampled_token_ids`, or the hidden stream's
geometry name (`g<HB>x<M>x<NB>[x<BN>x<D>]`).  `element_range` is `[lo, hi)` in row-major elements of that Value -- compact: one identity row
covers a whole Value (rows x width), never a gate.  Every identity row carries a `family` label so "omit a whole producer family" and "omit one
request / step / rank" are coverage failures by construction (the required-value coverage check walks the manifest).

Derivation (single-request Program, `derive_step --wrapper request`): the flat Call sequence (`instances.json.gz`, the comparator's
unit) is segmented into engine steps at `TokenSelect` boundaries (exactly as `cb_a.compare`), every Call name carries the module path
(`nn_module_stack`) + fx node + row index (the runtime correspondence, recorded by hand here), and the registry spec statics give the
Value widths (Gemm N, RMSNorm N, RoPE NHEADS*D, SiluMul I, Attention NH*D, Embedding H, TokenSelect V).  The attention hidden
family's geometry per (layer, step) is the FA2/FA3 launch the
Program's `Attention_v3{T,NH,KVH,D,BN}` rows imply (`hidden_source.fa2_kblock_n` / `fa3_tile`, `mat_total_words`), so a tap that never
fires is a MISSING identity, not a zero.  The canonical VU query (`verity_capture.vu_canonical`; the verification query) decides which families are required:
every family it names or defaults is required in full; the attention override (`AttnBlock` interiors: s[c] dots, PV chains, max chain)
is what requires the hidden stream.

    PYTHONPATH=. python -m verity_capture.commit.required_manifest build --program <derive dir> --workload <wl.json> --out manifest.json
    PYTHONPATH=. python -m verity_capture.commit.required_manifest check --manifest manifest.json --layouts layouts.json

Dependency-free (json/gzip/hashlib) so bind's checker, the finalizer and tests can import it without torch.

Vocabulary (Draft 3 §12, frozen): `Q_module_v1` is called **Q_module_body_v1** in prose -- the verification query whose units are
the module BODIES of the served model (one verification unit per module call; the submodule calls inside a body are excluded and
are units of their own), so the boundary values are the union of every module body's inputs and outputs and the required values
are those plus the prescribed inputs (weights, prompt ids) and the protocol-required values (sampled ids).  The serialized id in
the manifest header (`query`) and in every manifest_digest stays `Q_module_v1` until a labelled decision renames it -- changing
the string changes the digest of every manifest of record.  The query IMPLEMENTATION is `verity_vllm.query` (`query.cli build` builds
the manifest; `sampled_replay.ProgramIndex` addresses a v2-query manifest from `v1_bridge.population`).  `members_for` and the by-name
tables below are its hand-maintained predecessor, still needed for (a) TP rank Programs -- the v2 query names no rank-local partials and
`compose_global` has no rank merge (TP-04) -- and (b) re-deriving v1 records of record (the harness's v1 engine, `replay_partition`'s
recompute of a record whose `addresses.rule` is v1).  They retire when both are gone; `tests/test_no_by_name_rules.py` pins them until then.
The words in this module: an 'identity row' is one required Value's manifest entry; `op_path` + `output_member` + `invocation`
are the occurrence key (the serialization of a runtime occurrence); 'the manifest' always means the required-value manifest.

PROTOCOL SUMMARY (Commit stage: the required-value manifest of record = the cached RequiredValues(P, Q) population)
  Purpose: name, BEFORE the Commit runs, every required Value the collector must bind -- derived from the Build's Program
    (instances.json.gz Calls) under the verification query Q_module_body_v1 (serialized `Q_module_v1`), never from what value
    acquisition happened to observe.
  Inputs: `build --program <derive dir> --workload wl.json` (B=1) | `build-global --fixture wl.json --programs-root <row dir>
    [--workload-digest]` (B>=2: one component Program per declared (LP, T)); `verify --manifest` rebuilds and compares after the Commit.
  Outputs: manifest.json (SCHEMA: header program_digest / manifest_digest / query / complete / unmodelled / cross_check / guard,
    populations{identities, by_family}, one `identities` row per required identity), manifest_verify.json;
    `coverage_check` = the required-value coverage check (`coverage`).
  Invariants: identity key = IDENTITY_KEY (program_digest, request_id, step, invocation, op_path, output_member, element_range, rank);
    DEFAULT_ENGINE traversal, the inventory engine kept as the one-release cross-check (both are built and must agree row for row,
    else the build fails by name); a family with no naming rule -> UnmodelledFamily -> `complete: False` -> coverage refused
    (never a narrowed PASS); a zero-width Value is never an identity row (MAN-02); the manifest's program_digest must equal the Build's
    (binding.require_build_program_digest, before any engine).
  Protocol spec: Notion -> "vLLM integration — protocol spec (Build · Match · Commit)".
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
from typing import Any, Iterable

from .manifest_format import (alias_family, canonical_identity, collector_path, components_fold_digest, coverage_check, fa2_geometry as _fa2_geometry,
                              fa3_geometry as _fa3_geometry, FAMILIES, has_splits_param, IDENTITY_KEY, INTERIOR_CROSSING, is_stochastic_select,
                              is_token_select, layer_of as _layer_of, LIFTED_BF16, LIFTED_F32, load_program, manifest_digest, mat_total_words,
                              moe_geometry, moe_identities, MOE_LEAF, program_dirs_for as _program_dirs_for, programs_for as _programs_for,
                              PROMOTED_MEMBER, REGISTRY_UNAVAILABLE_EXIT, RegistryUnavailable, REQ_RE as _REQ_RE, request_label,
                              row_key as _row_key, SCHEMA, segment_steps, spec_family, spec_statics, split_name, SPLITS_PARAM, TP_PARTIAL_MEMBER,
                              TP_SHARD_MEMBER, wrapper_prefix_of)

#: [R19 moe, row #70 / F-r19-moe-1] a MID-module collective SITE: the frontend names a collective whose rank-local operand is produced by
#: ANOTHER module (OLMoE `OlmoeAttention._apply_qk_norm`: q and k all-gathered inside `self_attn`, each a slice of `qkv_proj`'s Gemm) under
#: `<module>.<fx node>` (`model.layers.3.self_attn.all_gather2_7`), not under the bare module.  The leaf is the collective's fx node name.
_COLLECTIVE_SITE_RE = re.compile(r"^(all_reduce2|all_gather2)(_\d+)?$")
#: vLLM `LinearBase.forward` returns `(output, output_bias)` (return_bias=True default) -> the Gemm output is tuple slot 0
_LINEAR_LEAVES = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj", "gate_proj", "up_proj", "out_proj", "fc1", "fc2", "dense",
                  "gate")            # [R14 moe] the MoE router logits GEMM `mlp.gate` is a ReplicatedLinear -> member "0", not a logits `out`
#: [R15 dense] interior families of the unfused norm chain / soft caps / Gemma2Model's embed scale (never a module output; `members_for` -> [])
_DENSE_INTERIOR = ("SquareBf16", "SquareF32", "MeanTriton", "AddScalarF32", "AddScalarBf16", "AddWidenedBf16", "RsqrtF32", "ScaleRowBf16",
                   "ScaleRowF32", "MulVecF32", "Bf16DivScalar", "Bf16Tanh", "Bf16MulScalarTensor")
#: [R15 dense, manbind M-0323] families whose `members_for` rows carry `producer_ordinal` (the k-th op of that family under the module, per
#: row, in Program order) AND whose ordinal `build_manifest` confirms: it counts these rows per (module, row) and refuses a declared ordinal
#: the Program's own row order contradicts (fail closed: the C2 oracle would read the wrong fold instance).  A family emitting ordinals
#: belongs here; a family absent from it is not confirmed (its `producer_ordinal` lands on the row as declared).
_ORDINAL_FAMILIES = ("NarrowF32ToBf16",)


# ---------------------------------------------------------------------------------------------------------------------------------
# spec / name parsing
# ---------------------------------------------------------------------------------------------------------------------------------


def _operand_family(row: dict, rows: list[dict]) -> tuple[str | None, str]:
    """[rev-pm M-0402] the family (spec base) of the Program instance whose output `row`'s FIRST operand reads -- the row's `args` are
    per-operand lists of refs `["n", <instance index>, lo, hi]` (an instance output) | `["p", ...]` (a Program parameter).  Returns
    (family, "") or (None, why): the first operand is a parameter, the row carries no args (a synthetic Program), or the ref is malformed.
    [rev-pm M-0468] the caller must treat (None, why) as UNCONFIRMABLE, not as confirmed: a declared `producer_operand` on such a row is
    refused by name, never emitted on the strength of its presence."""
    args = row.get("args") if isinstance(row, dict) else None
    if not args:
        return None, "the Program row carries no args"
    try:
        ref = args[0][0]
    except (IndexError, TypeError):
        return None, "the Program row has no first operand ref"
    try:
        if ref[0] == "p":
            return None, "the Program feeds this op from a parameter, not an instance output"
        if ref[0] != "n":
            return None, f"the Program row's first operand ref kind is {ref[0]!r}, not an instance output"
        return spec_family(rows[int(ref[1])]["spec"]), ""
    except (KeyError, IndexError, TypeError, ValueError):
        return None, "the Program row's first operand ref does not resolve to an instance"


def collective_site(module: str) -> tuple[str, str] | None:
    """[R19 moe F-r19-moe-1] `<module>.<collective fx node>` -> (module, node) for a MID-module collective site (see `_COLLECTIVE_SITE_RE`);
    None for every other module path.  The collector commits the site's output and rank-local shard under the MODULE (its hook is on the
    module; the k-th collective of the module in the step is occurrence k), so the site's identities live at (module, invocation k) and
    the site path is an `op_path_aliases` entry (`collective_site_aliases`) -- exactly the shared-RoPE alias mechanism."""
    if "." not in module:
        return None
    parent, leaf = module.rsplit(".", 1)
    return (parent, leaf) if parent and _COLLECTIVE_SITE_RE.match(leaf) else None


# ---------------------------------------------------------------------------------------------------------------------------------
# the family table: spec family -> (module kind, required output members, width of each member in elements per row)
# ---------------------------------------------------------------------------------------------------------------------------------

#: [MAN-07] epilogue families: a row of one of these under a module makes that module's Gemm rows interior (the epilogue returns the tensor)
EPILOGUE_FAMILIES = ("BiasAdd",)


def epilogue_modules(seg: list[dict[str, Any]]) -> dict[str, int]:
    """module -> epilogue width N for the modules of one step segment that carry an epilogue row (Gemm -> BiasAdd under one biased
    linear, Qwen2.5 qkv_proj).  Membership is the interior test; the width is rev-cb D2's agreement check against the Gemm's N."""
    out: dict[str, int] = {}
    for r in seg:
        if spec_family(r["spec"]).startswith(EPILOGUE_FAMILIES):
            out[split_name(r.get("name") or "")[0]] = int(spec_statics(r["spec"]).get("N", 0))
    return out


def is_epilogue_interior(family: str, module: str, epilogue_mods: "set[str] | dict[str, int]") -> bool:
    """True for the Gemm row of a module whose step carries an epilogue row: its output is the epilogue's operand, not the module's."""
    return family.startswith("Gemm") and module in epilogue_mods


def padded_slot_members(statics: dict[str, Any]) -> list[tuple]:
    """[Q_module_pad_v1 = the (b) query: a DECLARED padded-boundary query] the slot UNITS of one padded fixed-slot MoE block row, named after
    the block Definition's nodes (verity_vllm/registry/moe_pad.py `_fixed_core` / `MoeExpertBody`): per slot e in 0..E-1 `slot<e>/up` (the w13
    Gemm, 2I bf16), `slot<e>/act` (SiluMul, I bf16), `slot<e>/down` (MoeSlotDown, H bf16), `slot<e>/g` (the routed weight produced inside the
    slot, 1 f32; 0.0f when the router did not pick e) -- moe's 5,121 words/slot basis (M-1119 (b): a UNION of distinct boundary identities,
    up-out = act-in once, act-out = down-in once, W_e prescribed and excluded) -- plus the router's `router_ids` (TOPK i32) and `router_w`
    (TOPK f32): the ids fix the ORDER of the f32 sum (`MoeSum_v1` adds the selected slots in id order; f32 addition is not associative), so
    they are required for the block output to be a function of committed words alone.  The block output `out` is the row's module member."""
    E, TOPK, H, I = int(statics["E"]), int(statics["TOPK"]), int(statics["H"]), int(statics["I"])
    out: list[tuple] = [("router_ids", TOPK, LIFTED_F32), ("router_w", TOPK, LIFTED_F32)]
    for e in range(E):
        out += [(f"slot{e}/up", 2 * I, LIFTED_BF16), (f"slot{e}/act", I, LIFTED_BF16), (f"slot{e}/down", H, LIFTED_BF16), (f"slot{e}/g", 1, LIFTED_F32)]
    return out


def members_for(family: str, statics: dict[str, Any], module: str, hint: str, tp: dict[str, Any] | None = None, *, padded_slots: bool = False) -> list[tuple]:
    """(output_member, width per row, dtype[, extras]) required for one Program row of `family` under `module`.

    `padded_slots` (Q_module_pad_v1 only): a padded fixed-slot MoE block row additionally names its slot units (`padded_slot_members`).

    `tp` (rank Program only): {"rank": r, "world": W, "collective": None | "all_reduce" | "all_gather"} -- the collective this module
    issues in this step (from the AllReduce2/AllGather2 rows under the same module).  [R15 tp TP-04] a Gemm/EmbeddingShard feeding a
    collective is the rank-local PARTIAL (`part_rank<r>` / `shard_rank<r>`, family `tp_rank_partials`); the collective row itself is the
    module's bound output member (`0` for a linear, `out` otherwise) of the all-reduced / all-gathered width.

    The optional 4th element is a dict of ADDITIVE manifest-row fields.  `producer_ordinal` (int): when a module's forward is exploded into
    several Program ops of the SAME family per row and two of them produce two members (dense's Gemma fused norm: NarrowF32ToBf16 #1 =
    the new residual = member "1", NarrowF32ToBf16 #2 = the normed row = member "0"), the ordinal (index of the producing op among the
    module's same-family ops in Program order, per row) tells the C2 oracle which fold instance holds the member -- without it the oracle
    reports the member 'ambiguous producer' (partial, never guessed; oracle_compare.member_bytes).  [rel int-4: manbind 331c2fccc + tp TP-04]"""
    leaf = module.rsplit(".", 1)[-1]
    coll = (tp or {}).get("collective")
    rank = int((tp or {}).get("rank", 0))
    world = int((tp or {}).get("world", 1))
    if family.startswith("MoeSum") and coll == "all_reduce":
        # [R15 tp TP-06, moe M-0249] tensor-parallel fused MoE: the block's summed expert output on THIS rank's expert shards is the input of
        # the module's all-reduce (MoERunner -> tensor_model_parallel_all_reduce) -> the rank-local partial, width H per routed token row,
        # in addition to the block stream planes (the rank's own interior) and the received all-reduced output (the AllReduce2 row below)
        return [(TP_PARTIAL_MEMBER.format(rank=rank), int(statics.get("H", 0)), "bf16")]
    if family.startswith(("MoeExpertGemm", "MoeRouterTopK", "MoeSum")) or (family.startswith("SiluMul") and leaf == MOE_LEAF):
        return []                                                                # covered by the MoE block stream family (`moe_identities`)
    if family.startswith(PADDED_MOE_FAMILIES):
        # [moe MOE-PAD-SPEC route A (3), reqval RequiredValues authority 03:40Z] the PADDED construction emits the whole fixed-slot MoE block
        # as ONE Definition call per token row (`MoeFfnFixedL_v1{E,TOPK,VPT,H,I}(x, logits, w13, w2) -> y`; its hand lift `LMoeFfnL_v1`).
        # Under Q_module_v1 that row IS the module-boundary VU of `mlp.experts`: its required output is the block output y (H words, bf16);
        # the E slot bodies (up 2I / act I / down H / g per slot) are INTERIOR to the row -- on this Program they are values of no
        # Q_module_v1 identity (MOE-01's pair rows, which ARE rows there, do not exist here: R17-45 (1), two representations of one
        # workload).  The slot-body planes are R(P_pad, Q) only under a query whose text partitions the block into slot units
        # (Lift(Q_fine_v1), or a declared padded-boundary Q with its own digest) -- the query decides the boundaries, never the collector.
        out = [("out", int(statics.get("H", 0)), "bf16")]
        return out + (padded_slot_members(statics) if padded_slots else [])
    if family.startswith(("AllReduce2", "AllGather2")):
        n = int(statics.get("N", 0))
        gather = family.startswith("AllGather2")
        # the all-reduced module output (received by every rank) / the gathered row (rank shards side by side)
        out = [("out", n * world, "bf16")] if gather else [("0" if (leaf in _LINEAR_LEAVES or "proj" in leaf) else "out", n, "bf16")]
        if tp is not None and collective_site(module) is not None:
            # [R19 moe F-r19-moe-1, row #70] a MID-module collective site (OLMoE: q / k all-gathered inside `self_attn`, each a slice of the
            # qkv Gemm's output): no Gemm / EmbeddingShard row under this module produces the rank-local operand, so the collective row
            # ITSELF carries the partial member (`shard_rank<r>` for a gather, `part_rank<r>` for a reduce; width N = the rank's operand).
            # Without it the gathered output had no peer partial to bind to (tp_peer_binding: 6,944 unbound per rank on row #70).
            out.append(((TP_SHARD_MEMBER if gather else TP_PARTIAL_MEMBER).format(rank=rank), n, "bf16"))
        return out
    if family.startswith("EmbeddingShard"):
        return [(TP_PARTIAL_MEMBER.format(rank=rank), int(statics.get("H", 0)), "bf16")]   # this rank's masked shard lookup (before its all-reduce)
    if family.startswith("Gemm"):
        n = int(statics.get("N", 0))
        if coll == "all_reduce":
            return [(TP_PARTIAL_MEMBER.format(rank=rank), n, "bf16")]           # row-parallel partial (Gemm on the rank's K/W shard)
        if coll == "all_gather":
            return [(TP_SHARD_MEMBER.format(rank=rank), n, "bf16")]             # lm_head shard (N = V/W columns of the logits)
        if leaf in _LINEAR_LEAVES or "proj" in leaf:
            return [("0", n, "bf16")]                  # LinearBase -> (output, bias)
        return [("out", n, "bf16")]                   # LogitsProcessor -> logits tensor
    if family.startswith("RMSNormTriton"):
        return [("out", int(statics.get("N", 0)), "bf16")]                      # norm without residual -> one tensor
    if family.startswith("RMSNormFusedCuda"):
        n = int(statics.get("N", 0))
        return [("0", n, "bf16"), ("1", n, "bf16")]                             # fused add + norm -> (normed, residual)
    if family.startswith("RoPE"):
        w = int(statics.get("NHEADS", 0)) * int(statics.get("D", 0))
        return [("0" if hint == "query" else "1", w, "bf16")]
    if family.startswith(("SiluMul", "GeluTanhMul", "GeluMul")):
        return [("out", int(statics.get("I", 0)), "bf16")]                      # vLLM SiluAndMul / GeluAndMul(approximate=tanh): one [rows, I] tensor
    if family.startswith(("Attention", "AttentionHead")):
        return [("out", int(statics.get("NH", 0)) * int(statics.get("D", 0)), "bf16")]
    if family.startswith("Embedding"):
        return [("out", int(statics.get("H", 0)), "bf16")]
    if family.startswith("BiasAdd"):
        # [MAN-02 / MAN-07, exec M-0528] a biased linear (Qwen2 qkv_proj): the Program has Gemm_v1 -> BiasAdd_v1 under ONE module whose
        # single returned tensor (POST-bias) is what the collector binds as member "0".  The BiasAdd row IS that tensor's producer (its
        # spec names the fold instance the oracle reads, and the relation the replay evaluates); the module's Gemm row is INTERIOR
        # (`epilogue_modules` / `is_epilogue_interior`: build_manifest and sampled_replay.ProgramIndex drop its default claim).  Reading the
        # member at the Gemm instance (pre-bias) was a false C2 mismatch on every Qwen2.5 layer x step (M-0528).  Never zero-width.
        return [("0", int(statics.get("N", 0)), "bf16")]
    # [R15 dense, F-dense-05] the unfused norm chain (vLLM GemmaRMSNorm.forward_native as ATen on an f32 copy), GeGLU and the soft caps
    # (verity_vllm.registry.dense; Build rules `frontend/rules/vllm_bindings.py` Widen.../Narrow.../Square/MeanDimBI/Rsqrt/AddScalar/
    # AddWidened/MulF32/GeluTanhAndMul/Bf16ScalarChain).  The Build names the rows that ARE the module's returned tensors with an explicit
    # member hint (`<module>/<node>/<member>[row]`): `NarrowF32ToBf16` = the norm's `.to(bf16)` (`out` alone, `0` normed + `1` residual
    # in the fused form), `Bf16MulScalar` with `out` = the last op of a logits soft cap `C * tanh(x / C)` (LogitsProcessor.soft_cap).
    # Everything else in these chains is interior (f32 intermediates, the (1 + w) vector, the div / tanh of a soft cap, Gemma2Model's
    # `embed * normalizer`) -- never an identity, never zero-width.  A hinted-out row makes the module's UN-hinted default claims interior
    # (see `build_manifest`: a soft-capped LogitsProcessor's Gemm output is not what the module returns).
    if family.startswith("NarrowF32ToBf16"):
        n = int(statics.get("N", 0))
        if hint == "out":
            return [("out", n, "bf16")]                                          # the module's single narrowing: no producer to tell apart
        if hint in ("0", "1"):
            # [manbind M-0323] the fused form has TWO NarrowF32ToBf16 ops per row under one module; `producer_ordinal` = the producing op's
            # index among them in Program order, from the same dataflow fact the Build named the member by (NarrowF32ToBf16Rule._member):
            # Narrow #0 narrows the AddWidened sum (`residual = s32.to(bf16)`, member "1") -- vLLM GemmaRMSNorm.forward_native stores the
            # residual BEFORE the variance chain; Narrow #1 narrows the MulVec row (`x.to(orig_dtype)`, member "0") at the return.
            # Confirmed on the Gemma-2-2B B1 256/15 request Program (52 fused norms x 15 steps, 14040 rows per member: every "1" row
            # consumes AddWidenedBf16_v1 and precedes its "0" row, which consumes MulVecF32_v1; 0 violations).  `build_manifest` re-checks
            # the ordinal against every Program's own row order and fails closed on a disagreement.
            # [rev-pm M-0402] `producer_operand` = the family the producing narrowing consumes -- an order-free selector for the oracle (the
            # ordinal is defined in Program order but a `dag`-criterion fold may emit the two independent narrowings permuted); confirmed
            # by `build_manifest` from the Program's own args.
            if hint == "1":
                return [("1", n, "bf16", {"producer_ordinal": 0, "producer_operand": "AddWidenedBf16_v1"})]
            return [("0", n, "bf16", {"producer_ordinal": 1, "producer_operand": "MulVecF32_v1"})]
        raise UnmodelledFamily(f"{family} (no member hint: a narrowing that is not a declared module output)", module)
    if family.startswith("Bf16MulScalar_"):                                     # not Bf16MulScalarTensor (interior, below)
        return [("out", int(statics.get("N", 0)), "bf16")] if hint == "out" else []
    if family.startswith("GeluTanhMul"):
        return [("out", int(statics.get("I", 0)), "bf16")]                       # GeluAndMul (GeGLU) -> one tensor, as SiluMul
    if family.startswith(_DENSE_INTERIOR):
        return []
    if family.startswith("Const"):
        # [R15 stoch M-0379 / manbind ACK M-0395] `Const32[top_p]_v1` / `Const64[pos]_v1` / `Const32[temperature]_v1`: the stochastic Program's
        # constants written at Build from the DECLARED sampling policy -- structure (the value is inside the spec string, hence inside the
        # Program digest), not a produced value: no identity, never a zero-width one.
        return []
    if family.startswith("ScaledMmFp8Block"):
        # [R15 fp8] a blockwise-FP8 linear (Fp8LinearMethod, block_quant, CutlassFp8BlockScaledMMKernel): the Program has
        # Fp8GroupQuant_v1 -> ScaledMmFp8Block_v1 under ONE LinearBase module; the module's output tensor is the scaled GEMM's bf16 output
        # (`_C.cutlass_scaled_mm(out_dtype=bf16)`), bound as member "0" exactly like a bf16 Gemm.  The e4m3 words and the f32 group scales
        # are interior to the module forward (never a module output) -- they are Match's, re-derived from the bf16 input by the Program.
        return [("0", int(statics.get("N", 0)), "bf16")]
    if family.startswith(("Fp8GroupQuant", "Fp8GroupScale")):
        return []                                        # [R15 fp8] the activation quant: interior of the FP8 linear; required via ScaledMmFp8Block
    # [MAN-02] an unmodelled family is NOT a value with an empty range: the manifest is incomplete and must fail closed (see build_manifest).
    raise UnmodelledFamily(family, module)


class UnmodelledFamily(Exception):
    """A Program row family `members_for` has no rule for.  build_manifest records it and marks the manifest incomplete (fail closed)."""

    def __init__(self, family: str, module: str):
        super().__init__(f"{family} under {module}")
        self.family, self.module = family, module


# ---------------------------------------------------------------------------------------------------------------------------------
# boundary promotion [R19 int-20, F-r19-int-20]: an interior row whose output crosses a module boundary is a required member
# ---------------------------------------------------------------------------------------------------------------------------------

#: families whose values are bound through a DECLARED aggregate family rather than as their own module output: the fused-MoE block
#: stream planes (`moe/L<k>/<plane>`, MOE-03) cover the router / expert GEMM / act / sum rows of `mlp.experts` (the SiluMul under the
#: MoE leaf too).  Never promoted: the crossing is the aggregate's.
_AGGREGATE_PREFIXES = ("MoeExpertGemm", "MoeRouterTopK", "MoeSum")
#: structure rows (the value is inside the spec string, hence inside the Program digest): no identity, never promoted
_STRUCTURE_PREFIXES = ("Const",)
#: the manifest dtype of a promoted member from its Definition's output word width -- the two widths this registry's composites return as
#: their bf16 / f32 arrays (`_weights()` labels the weight fields by the same map).  Anything else (a lifted word `Value<17>` / `Value<33>`,
#: an i64 `Value<64>`, a bit) is NOT derivable: the manifest never guesses a dtype (fail closed by name, `UNDERIVABLE`).
PROMOTED_DTYPES = {16: "bf16", 32: "f32"}
#: the named fault of a promoted family whose member the registry cannot size / type; the row stays an unnamed crossing (incomplete)
UNDERIVABLE = "unnamed crossing: promoted member has no derivable signature"
PROMOTION_RULE = ("an INTERIOR-classified row (the vocabulary `members_for` names no member for its family; not an aggregate, not structure) whose "
                  "output the Program's own dataflow (operand runs `args`) reads from ANOTHER module is PROMOTED to a required member of its module: "
                  f"(module, `{PROMOTED_MEMBER}`) per step, family instance_outputs, `rows` = the crossing rows, width / dtype from the registry "
                  "Definition's output signature (the same registry the Program was built from), `consumers` = the reading modules in Program order "
                  f"(the collector observes the value at a consumer's input: `<consumer>/in`), boundary class `{INTERIOR_CROSSING}`.  ADDITIVE: a row the "
                  "vocabulary names stays named whether or not it crosses (TP-04 partials, MoE aggregates, the logits Gemm, a non-crossing narrowing); "
                  "a Program without operand runs promotes nothing (the audit is not armed).  Fail closed by name: a member the registry cannot size "
                  f"({UNDERIVABLE}), two promoted families on one member, a promoted member colliding with a vocabulary member, a partially crossing "
                  "group, an unrooted producer path whose consumers are not under it (MAN-08)")
#: the integration's registry modules whose composites a Program row can name (imported once, on the first promotion); an import error is
#: recorded and REPORTED with the fault, never swallowed as 'not registered'
_REGISTRY_MODULES = ("b1", "dense", "fp8", "moe", "sampling", "b1_tp2", "moe_pad")
_registry_cache: dict[str, Any] = {}


def is_aggregate_family(family: str, module: str) -> bool:
    return family.startswith(_AGGREGATE_PREFIXES) or (family.startswith("SiluMul") and module.rsplit(".", 1)[-1] == MOE_LEAF)


def is_structure_family(family: str) -> bool:
    return family.startswith(_STRUCTURE_PREFIXES)


def _registry() -> tuple[Any, list[str]]:
    """(verity.ir REGISTRY with the integration's composite modules imported, [import errors of the integration's OWN registry modules])
    -- once per process.  The core package not importable is RegistryUnavailable (raised, not cached as None: an environment fault
    must stop the build, never land in the manifest as an `unmodelled` row)."""
    if "registry" not in _registry_cache:
        try:
            from verity.ir.defs import REGISTRY
        except Exception as e:  # noqa: BLE001
            raise RegistryUnavailable(e) from e
        import importlib

        errors: list[str] = []
        for m in _REGISTRY_MODULES:
            try:
                importlib.import_module(f"verity_vllm.registry.{m}")
            except Exception as e:  # noqa: BLE001  (named beside the fault; a missing optional module is not a guess about the family)
                errors.append(f"verity_vllm.registry.{m}: {type(e).__name__}: {e}")
        _registry_cache["registry"] = REGISTRY
        _registry_cache["errors"] = errors
    return _registry_cache["registry"], _registry_cache["errors"]


def promoted_signature(spec: str) -> tuple[int | None, str | None, str | None]:
    """(width per row, dtype, None) of the single output member a Program row of `spec` produces, from the registry Definition's output
    signature bound to the row's statics -- or (None, None, why) when it is not derivable: the family is not a registered Definition, the
    statics do not bind, the output is a tuple / record (no single `out`), or its word width has no manifest dtype (`PROMOTED_DTYPES`).
    Never a guess: the caller fails closed on `why` (`UNDERIVABLE`)."""
    base = spec_family(spec)
    statics = spec_statics(spec)
    reg, errors = _registry()                                                    # RegistryUnavailable propagates: not a `why`, a fault
    d = reg.defs.get(base)
    if d is None:
        try:
            from verity.ir.codec import lookup_primitive
            d = lookup_primitive(reg, base)
        except Exception:  # noqa: BLE001
            d = None
    if d is None:
        return None, None, f"{base} is not a registered Definition" + (f" (registry import errors: {'; '.join(errors)})" if errors else "")
    try:
        if hasattr(d, "bind"):
            sp = d.bind(**statics)
            if not sp.complete:
                missing = [k for k in d.statics if k not in statics]
                return None, None, f"{base}: statics {missing} are not in the row's spec {spec!r}"
            _params, ret = sp.signature()
        else:
            ret = d.ret
    except Exception as e:  # noqa: BLE001
        return None, None, f"{base}: output signature not evaluable on statics {statics}: {type(e).__name__}: {e}"
    from verity.ir.types import Array, Value
    n, t = 1, ret
    while isinstance(t, Array):
        n *= int(t.n)
        t = t.elem
    if not isinstance(t, Value):
        return None, None, f"{base}: output type {ret!r} is not one array of one word width (a tuple / record output has no single `{PROMOTED_MEMBER}` member)"
    dtype = PROMOTED_DTYPES.get(int(t.w))
    if dtype is None:
        return None, None, f"{base}: output word width {int(t.w)} has no manifest dtype ({PROMOTED_DTYPES}; lifted / i64 / bit words are not derivable)"
    if n <= 0:
        return None, None, f"{base}: output type {ret!r} has no elements (a zero-width member is never an identity, MAN-02)"
    return n, dtype, None


def promote_interior_crossings(interior: list[tuple], crosses: Any, consumers: Any, *, named: Any, aliased: Any, prefix: str) -> dict[str, Any]:
    """The promotion pass, shared by the inventory, the traversal and the reference engine (`PROMOTION_RULE`).

    `interior`: [(step, module, family, spec, statics, row indices)] -- the claims the vocabulary called interior (`members_for` -> [] and
    neither an aggregate nor structure), one per (step, module, family, hint, spec) claim group, rows in Program order.
    `crosses(i)`: truthy iff row i's output is read by a row of ANOTHER module.  `consumers(i)`: the Program module paths of those readers,
    Program order.  `named(step)`: the set of (module, member) the vocabulary claimed in that step.  `aliased(step)`: the modules the step's
    alias table maps (a shared module: refused, never silently re-addressed).  `prefix`: the export wrapper prefix (`collector_path`).

    Returns {"identities": [row dicts: step, module, op_path, member, rows, width, dtype, spec, consumers, program_consumers],
             "faults": {text: rows}, "promoted_rows": [i], "named_families": {family}, "families": {family: summary}}."""
    by_group: dict[tuple[int, str], list[tuple]] = {}
    for step, mod, fam, spec, statics, idx in interior:
        by_group.setdefault((int(step), mod), []).append((fam, spec, statics, idx))
    identities: list[dict[str, Any]] = []
    faults: dict[str, int] = {}
    promoted_rows: list[int] = []
    named_families: set[str] = set()
    families: dict[str, dict[str, Any]] = {}

    def fault(text: str, n: int, fams: Iterable[str]) -> None:
        faults[text] = faults.get(text, 0) + n
        named_families.update(fams)

    for (step, mod), claims in sorted(by_group.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        op = collector_path(mod, prefix)
        xfam: dict[str, dict[str, Any]] = {}                                     # family -> {rows, crossing, specs, idx}
        for fam, spec, statics, idx in claims:
            xs = [int(i) for i in idx if crosses(int(i))]
            f = xfam.setdefault(fam, {"rows": 0, "crossing": 0, "specs": {}, "idx": []})
            f["rows"] += len(idx)
            f["crossing"] += len(xs)
            if xs:
                f["specs"][spec] = f["specs"].get(spec, 0) + len(xs)
                f["idx"].extend(xs)
        crossing = {fam: f for fam, f in xfam.items() if f["crossing"]}
        if not crossing:
            continue                                                             # every interior row of this module stays inside it
        n_cross = sum(f["crossing"] for f in crossing.values())
        head = f"{'/'.join(sorted(crossing))} under {op}: {n_cross} row(s) whose output the Program reads across a module boundary while the vocabulary names no member for it"
        if len(crossing) > 1:
            fault(f"{head} (unnamed crossing: {len(crossing)} promoted families {sorted(crossing)} claim one member `{PROMOTED_MEMBER}`)", n_cross, crossing)
            continue
        fam, f = next(iter(crossing.items()))
        if f["crossing"] != f["rows"]:
            fault(f"{head} (unnamed crossing: {f['crossing']} of {f['rows']} rows of the group cross -- a partially crossing group is not one member)", f["crossing"], [fam])
            continue
        sig: set[tuple[int, str]] = set()
        why = None
        for spec in sorted(f["specs"]):
            w, dt, why = promoted_signature(spec)
            if why:
                break
            sig.add((int(w), str(dt)))
        if why:
            fault(f"{head} ({UNDERIVABLE}: {why})", f["crossing"], [fam])
            continue
        if len(sig) != 1:
            fault(f"{head} ({UNDERIVABLE}: the group's specs {sorted(f['specs'])} have unequal signatures {sorted(sig)})", f["crossing"], [fam])
            continue
        (width, dtype), = sig
        rows_x = sorted(f["idx"])
        program_consumers: list[str] = []
        for i in rows_x:
            for c in consumers(i):
                if c not in program_consumers:
                    program_consumers.append(c)
        if "." not in mod:
            # [MAN-08, promoted form] an UNROOTED producer path is a bare leaf that lost its parent (`gate`) -- unless the dataflow shows it is
            # the ANCESTOR of every module reading it (a root-level module whose forward body produced the value): the consumers' own paths
            # decide, never the path's name
            if not program_consumers or not all(c.startswith(mod + ".") for c in program_consumers):
                fault(f"{head} (unnamed crossing: unrooted module path {mod!r} is not the parent of its consumers {program_consumers[:3]}; MAN-08)", f["crossing"], [fam])
                continue
        if (mod, PROMOTED_MEMBER) in (named(step) or ()):
            fault(f"{head} (unnamed crossing: the promoted member `{PROMOTED_MEMBER}` collides with the module's vocabulary member `{PROMOTED_MEMBER}` at step {step})", f["crossing"], [fam])
            continue
        if mod in (aliased(step) or ()):
            fault(f"{head} (unnamed crossing: the module is an alias site at step {step}; a promoted member of a shared module is not addressed)", f["crossing"], [fam])
            continue
        identities.append({"step": step, "module": mod, "op_path": op, "member": PROMOTED_MEMBER, "rows": len(rows_x), "width": width, "dtype": dtype,
                           "spec": fam, "consumers": [collector_path(c, prefix) for c in program_consumers], "program_consumers": program_consumers})
        promoted_rows.extend(rows_x)
        s = families.setdefault(fam, {"rows": 0, "identities": 0, "steps": 0, "width": width, "dtype": dtype, "consumers": []})
        s["rows"] += len(rows_x)
        s["identities"] += 1
        s["steps"] += 1
        for c in program_consumers:
            cc = collector_path(c, prefix)
            if cc not in s["consumers"]:
                s["consumers"].append(cc)
    return {"identities": identities, "faults": faults, "promoted_rows": promoted_rows, "named_families": named_families, "families": families}


def program_crossings(rows: list[dict]) -> tuple[bytearray, dict[int, list[str]]]:
    """The in-memory dataflow pass of the inventory / reference engines (the traversal's `_audit_args`, streamed): per row i, `crosses[i]`
    = its output is read by a row of another module; `consumers[i]` = those readers' module paths in Program order.  Operand runs are
    `args = [[["n", producer, lo, hi] | ["p", param, lo, hi], ..], ..]` (one list per operand)."""
    n = len(rows)
    mod_of = [split_name(r.get("name") or "")[0] for r in rows]
    crosses = bytearray(n)
    consumers: dict[int, list[str]] = {}
    for i, r in enumerate(rows):
        args = r.get("args")
        if not args:
            continue
        me = mod_of[i]
        for operand in args:
            for ref in operand:
                try:
                    if ref[0] != "n":
                        continue
                    j = int(ref[1])
                except (IndexError, TypeError, ValueError):
                    continue
                if 0 <= j < i and mod_of[j] != me:
                    crosses[j] = 1
                    c = consumers.setdefault(j, [])
                    if me not in c:
                        c.append(me)
    return crosses, consumers


def promoted_annotations(d: dict[str, Any]) -> dict[str, Any]:
    """The ADDITIVE manifest-row fields of a promoted identity (both engines emit exactly these)."""
    return {"rows": d["rows"], "width": d["width"], "dtype": d["dtype"], "spec": d["spec"], "promoted": True, "boundary_class": INTERIOR_CROSSING,
            "consumers": list(d["consumers"])}


def promotion_header(result: dict[str, Any] | None) -> dict[str, Any]:
    """The query header's `promotion` block (identical on every engine): the rule, the member, what was promoted."""
    fams = (result or {}).get("families") or {}
    faults = (result or {}).get("faults") or {}
    return {"rule": PROMOTION_RULE, "member": PROMOTED_MEMBER, "boundary_class": INTERIOR_CROSSING,
            "families": {k: {kk: vv for kk, vv in v.items()} for k, v in sorted(fams.items())},
            "identities": sum(int(v["identities"]) for v in fams.values()), "rows": sum(int(v["rows"]) for v in fams.values()),
            "faults": {k: int(v) for k, v in sorted(faults.items())},
            "audit": (f"a promoted row is class `{INTERIOR_CROSSING}` in boundary_audit.families.<family> and counted under boundary_audit.promoted "
                      "(traversal engine); a faulted one stays `interior` with its crossings and is named under `unmodelled` (complete: false)")}


#: the padded fixed-slot MoE block Definitions (registry/moe_pad.py: the unlifted twin, what the frontend emits, and their hand lifts)
PADDED_MOE_FAMILIES = ("MoeFfnFixedL", "MoeFfnFixed", "LMoeFfnL", "LMoeFfn")


# ---------------------------------------------------------------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------------------------------------------------------------


#: [COORD R17-10 = Daniel 17:41Z] the sweep's query OF RECORD, named with a stable id: one unit per Program row / module boundary.  It is
#: what every Commit record's population IS (revb S-r17b-04: every accepted population was this, consistently) -- the id and its digest go
#: on the manifest header, the replay's `population.query_total`, and the verdict's scope line.  `RequiredValues(P, Q)` stays query-
#: parametric: `Q_fine_v1` (verity_capture.vu_canonical, the fine relation-level partition) selects the required FAMILIES here and is
#: measured beside it as the research item (out/gen/r17/Q-EXPANSION.md); a later protocol may designate another Q, at which point Commit
#: and overhead are revalidated for that Q.  Nothing is "canonical".
Q_MODULE_ID = "Q_module_v1"


def q_module_text(profile_key: str, hidden_family: str, family_selector_digest: str | None) -> str:
    """The definition of Q_module_v1 as text (its digest is `query_id_digest` on the header): the UNIT, the boundary, the required
    families and the family selector it consumes.  Change the text -> the id must change (v2); the digest makes a silent edit visible."""
    return (f"{Q_MODULE_ID}: unit = Program row = module-boundary VU (one identity per module OUTPUT member per (request, step, invocation, "
            f"module), plus input_token_ids, sampled_token_ids, weights, {hidden_family}); boundary = module; profile = {profile_key}; "
            f"family selector = Q_fine_v1 (verity_capture.vu_canonical) digest {family_selector_digest}; identity key = {list(IDENTITY_KEY)}")


Q_MODULE_PAD_ID = "Q_module_pad_v1"


def q_module_pad_text(profile_key: str, hidden_family: str, family_selector_digest: str | None) -> str:
    """Q_module_pad_v1 = Q_module_v1's text + the padded-block clause: on a Program built under the PADDED fixed-slot MoE construction
    (rows MoeFfnFixedL_v1 / MoeFfnFixed_v1 and their hand lifts), every slot unit of the block is a unit of the query (`padded_slot_members`).
    A DIFFERENT query with its own id and digest -- never a change to Q_module_v1's population (revb S-r17b-51: the two are rows side by side
    in the padded table, the query named per cell)."""
    return (q_module_text(profile_key, hidden_family, family_selector_digest).replace(f"{Q_MODULE_ID}:", f"{Q_MODULE_PAD_ID}:", 1)
            + "; padded clause = on MoeFfnFixedL_v1 / MoeFfnFixed_v1 rows the slot units are units: per slot e slot<e>/up (2I), slot<e>/act (I), "
              "slot<e>/down (H), slot<e>/g (1 f32) + router_ids / router_w (TOPK) + out (H); boundary = module-pad"
              "; lifted-word clause (F-r17b-52) = the slot units and the router members are LIFTED words of the construction (Value<17> -> u32 "
              "word, bf16 payload; Value<33> -> u64 word, f32/i32 payload; Active(x) = x, bot = 1 << w): the slot enable a_e = [e in router_ids "
              "of the row]; the required value of a slot unit with a_e = 0 is enc(bot) = 1 << w in EVERY word, exactly; with a_e = 1 it is the "
              "Active word of the unlifted twin's value; a value word under a_e = 0 fails, a bot word under a_e = 1 fails (pad R17-42 s5); "
              "router_ids / router_w are Active in every executed step (the block's enable is the step's run bit); out (H) stays the module's "
              "bf16 output word")


def query_requirements_pad(qi: dict[str, Any]) -> dict[str, Any]:
    """Rewrite a Q_module_v1 header (`query_requirements`) into Q_module_pad_v1's, in place: id, digest, boundary, unit."""
    key, hidden, qdig = qi["profile"], qi["attention_hidden_family"], qi["query_digest"]
    qi.update({"query_id": Q_MODULE_PAD_ID, "query_id_digest": hashlib.sha256(q_module_pad_text(key, hidden, qdig).encode()).hexdigest(),
               "implemented_query": Q_MODULE_PAD_ID, "boundary": "module-pad",
               "unit": (f"Program row = module-boundary VU under {Q_MODULE_ID}, PLUS on padded fixed-slot MoE rows every slot unit "
                        f"(padded_slot_members) -- {Q_MODULE_PAD_ID}: the declared padded-boundary query (revb S-r17b-51 (b)); the planes moe's "
                        "construction materialises ARE identities here, by the query's text, not by the collector"),
               "based_on": {"query_id": Q_MODULE_ID, "query_id_digest": qi["query_id_digest"]}})
    qi["why"]["padded_slots"] = ("Q_module_pad_v1's padded clause: the E slot bodies of every MoeFfnFixedL_v1 row (up / act / down / g per slot) and "
                                 "the router's ids / weights are required per (token row, MoE layer) as LIFTED words (l17 = Value<17> u32 / l33 = "
                                 "Value<33> u64; Active(x) = x, bot = 1 << w). The slot enable a_e = [e in the row's committed router_ids]: a_e = 0 "
                                 "-> every word of slot<e>/up|act|down|g is REQUIRED to be enc(bot) exactly (the construction materialises them; "
                                 "moe Stage 3b); a_e = 1 -> the Active word of the unlifted twin's value. A value word under a_e = 0 and a bot word "
                                 "under a_e = 1 both fail by name (pad R17-42 s5; revb F-r17b-52 (i)-(iii))")
    return qi


def query_requirements(profile: str) -> dict[str, Any]:
    """What the query of record (Q_module_v1) requires of the commitment, by family -- the families selected by Q_fine_v1's text (the
    selector's digest is `query_digest`), the unit and id of the implemented query (`query_id`, `query_id_digest`)."""
    try:
        from verity_capture.vu_canonical import QUERY_ID as QFINE, VERSION as QV, canonical_query_text, resolve_profile
        key = resolve_profile(profile)
        text = canonical_query_text(profile)
        qdig = hashlib.sha256(text.encode()).hexdigest()
    except Exception:  # pragma: no cover - the query module is in the same tree; keep the manifest buildable in a bare checkout
        key, QV, qdig, QFINE = ("hopper" if "90" in str(profile) or "fa3" in str(profile).lower() else "ampere"), "unknown", None, "Q_fine_v1"
    hidden = "fa3_hidden_m1_stream" if key == "hopper" else "fa2_hidden_m1_stream"
    return {"query": "verity_capture.vu_canonical", "query_version": QV, "profile": key, "query_digest": qdig,
            # [R17-10] the query OF RECORD, by id: Q_module_v1 (the unit the identities below are), digest of its definition text
            "query_id": Q_MODULE_ID, "query_id_digest": hashlib.sha256(q_module_text(key, hidden, qdig).encode()).hexdigest(),
            "family_selector": {"query_id": QFINE, "module": "verity_capture.vu_canonical", "version": QV, "digest": qdig,
                                "role": "selects WHICH families are required; its unit (the fine partition) is NOT the population unit of record"},
            # [R17-9 (ii) / revb F-r17b-05 / reqval] the record STATES ITS UNIT and names the IMPLEMENTED Q: the population's identities are
            # the module-boundary values (every module OUTPUT member per request x step, plus the families below) -- Q_module_v1, the
            # boundary `required_values.Query.boundary == "module"` and this inventory's rule alike.  Q_fine_v1 (vu_canonical, the
            # query_digest above) decides WHICH families are required and is the relation-level partition (Gemm coordinates, AttnBlock
            # chains, norm interiors: 1.4-3.6 G units on the Llama records) -- it is NOT the implemented population unit.  A Program row
            # is one module-boundary VU under Q_module_v1 (Attention_v3 / Gemm_v1 rows are units here, parents under the fine partition).
            "implemented_query": Q_MODULE_ID, "boundary": "module",
            "unit": (f"Program row = module-boundary VU under {Q_MODULE_ID}: identities are the module output members per (request, step, module) "
                     "+ ids, sampled ids, weights, the attention M1 stream; Q_fine_v1 (vu_canonical) selects the required "
                     "families (query_digest) and is not the implemented population unit (R17-9 (ii), F-r17b-05, R17-10)"),
            "required_families": ["input_token_ids", "instance_outputs", "sampled_token_ids", "weights", hidden],
            "attention_hidden_family": hidden,
            "why": {"instance_outputs": "every VU family of the query (Gemm y[n], norm normalize path, SiluMul, RoPE, Embedding) opens on the "
                                        "producing instance's OUTPUT words -> every Program instance output member per step is required",
                    hidden: "the attention override refines AttentionHead into AttnBlock interiors (s[c] dots, PV k-step chains, max chain, "
                            "Fa2InvSum) -> the M1 export stream of every attention launch (layer x step) is required",
                    "input_token_ids": "Embedding_v1 opens on its ids input (Input32)", "sampled_token_ids": "TokenSelect_v1 output per step",
                    "weights": "Input16 registered inputs: every weight field the Program's `weights` parameter names"}}


def module_signature(groups: dict[tuple, dict], mod: str) -> tuple:
    """The logical producer a module's rows describe: every member's (member, family, width, dtype, extras), sorted.  Two modules with
    different signatures are two logical producers whatever their paths look like."""
    return tuple(sorted((str(member), str(g["family"]), int(g["width"]), str(g["dtype"]), json.dumps(g.get("extras") or {}, sort_keys=True))
                        for (m, member), g in groups.items() if m == mod))


def shared_module_aliases(groups: dict[tuple, dict], *, shared: bool, family_prefix: str = "RoPE") -> dict[str, tuple[str, int]]:
    """[COORD R17-39 (c) / A2 aliasing] THE ONE rule every builder (inventory, Traversal, the streaming rows) applies for a shared module:
    module -> (canonical module, invocation).  vLLM's `get_rope` returns ONE RotaryEmbedding instance for IDENTICAL parameters, registered
    under the first layer's path and called once per layer -> the per-layer Program paths of that instance alias to the first layer's,
    invocation = layer order.  The alias is a DECLARATION (`shared`) over the family -- never inferred from coinciding bytes or offsets --
    and it is FAIL-CLOSED on the signature: modules of the family are grouped by `module_signature` (member set x family x width x dtype x
    extras) and only modules with the SAME signature are one instance; a module whose rows describe a different logical producer (another
    version, another width, another member set) is its own group with its own canonical path, never merged into a neighbour because
    the family name coincides.  Before this rule every `RoPE*` module of the Program was folded into one instance by the family name alone."""
    mods = sorted({mod for (mod, _m), g in groups.items() if str(g["family"]).startswith(family_prefix)}, key=_layer_of)
    if not shared or not mods:
        return {}
    by_sig: dict[tuple, list[str]] = {}
    for m in mods:
        by_sig.setdefault(module_signature(groups, m), []).append(m)
    out: dict[str, tuple[str, int]] = {}
    for sig, members in by_sig.items():
        members = sorted(members, key=_layer_of)
        for k, m in enumerate(members):
            out[m] = (members[0], k)
    return out


def collective_site_aliases(groups: dict[tuple, dict]) -> dict[str, tuple[str, int]]:
    """[R19 moe F-r19-moe-1, row #70] THE ONE rule every builder applies for a MID-module collective site (`collective_site`): site path ->
    (module, invocation).  The collector hooks the MODULE and commits the module's k-th collective of the step as occurrence k of
    (module, member) (binding.build_binding_map's counter), so the site's identities (`out` and `shard_rank<r>` / `part_rank<r>`) are
    required at (module, k) and the site path is recorded in `op_path_aliases` -- the table sampled_replay resolves every Program row
    through (`identities_unreachable_by_alias`: an invocation >= 1 identity is addressed ONLY through it).  k counts the module's sites in
    PROGRAM order (`groups` insertion order = the step's row order; both members of one site share its k); when the module also carries
    an ordinary (non-site) member, its own identity keeps invocation 0 and the sites start at 1 -- never a collision by (op_path, member,
    invocation).  OLMoE `self_attn`: the q gather is invocation 0, the k gather invocation 1, on both ranks (same fx node order)."""
    out: dict[str, tuple[str, int]] = {}
    next_k: dict[str, int] = {}
    bare = {mod for (mod, _m) in groups if collective_site(mod) is None}
    for (mod, _member) in groups:                                  # insertion order: the step's Program order
        if mod in out:
            continue
        site = collective_site(mod)
        if site is None:
            continue
        parent = site[0]
        k = next_k.get(parent, 1 if parent in bare else 0)
        out[mod] = (parent, k)
        next_k[parent] = k + 1
    return out


#: [R16 reqval] which implementation derives the population: `inventory` = this module's R14/R15 per-family walk (the code of record until
#: r17-int-2, the cross-check for one release after), `traversal` = `required_values.Traversal` (RequiredValues(P,Q): lazy, per step,
#: dataflow-audited; the manifest is its cache).  Selected per call (`engine=`) or by `VERITY_REQUIRED_VALUES_ENGINE`; anything else is
#: refused by name; the default is below (R17 task 3: promoted row by row through the ladder, BRIEF §7A).
ENGINES = ("inventory", "traversal")
ENGINE_ENV = "VERITY_REQUIRED_VALUES_ENGINE"
#: [R17 reqval task 3 -- THE FLIP, r17-int-2] the traversal engine is the DEFAULT for a TP1 Program: promotion ladder (1a) dense B1, (1a')
#: flag-ON live row, (1b) global B8, (1c) OLMoE B1/B2 (pre-LAG and LAG-declared), (1d) TP2 B1 -- record == inventory == traversal row for
#: row, digest for digest, `complete` included (out/gen/r17/REQUIRED-VALUES-PROMOTION.md; revb S-r17b-02 + M-0099).  For ONE release the
#: manifest path stays the CROSS-CHECK of record: the row's manifest stage computes BOTH and they must agree row for row (`cross_check` on
#: the header) or the row fails by name (`_main` build / build-global, `verify`).  Explicit `--engine` / `VERITY_REQUIRED_VALUES_ENGINE`
#: always wins (the ladder tool, the differential tests, a reader who wants one side).
#: The default covers a RANK Program (ranks > 1, derive_step --tp) too: M-0099 held the TP engine explicit until rung (1d) was posted and
#: revb-checked; (1d) posted M-0124 (finding 7), revb reproduced it and extended S-r17b-02 to the TP paths (M-0143: "the flip may include the
#: TP paths -- an unnamed value at B8 TP reads complete False by name, never a narrowed PASS").  The cross-check runs on the merged TP
#: manifest like any other.
DEFAULT_ENGINE = "traversal"
#: "0" disables the one-release cross-check (never on a row of record: row_pod.sh / tp2 set nothing); anything else = on
CROSS_CHECK_ENV = "VERITY_REQUIRED_VALUES_CROSS_CHECK"
#: [revb S-r17b-51 (b), int-8 form] the query BOUNDARY the manifest is built under: module (Q_module_v1, the default and the rows of record) |
#: module-pad (Q_module_pad_v1: Q_module_v1 + the slot units of padded fixed-slot MoE rows as DECLARED requirements).  module-pad is the
#: TRAVERSAL engine's (the inventory engine has no padded-slot rule): the one-release cross-check does not apply to it and the header says so
#: (`cross_check.skipped`); unset = module, nothing of record changes.  (The lane's fine boundaries of R17-43 s8 are not in this cut.)
BOUNDARY_ENV = "VERITY_REQUIRED_BOUNDARY"
BOUNDARIES = ("module", "module-pad")


def manifest_boundary(boundary: str | None = None) -> str:
    b = boundary or os.environ.get(BOUNDARY_ENV) or "module"
    if b not in BOUNDARIES:
        raise ValueError(f"required-manifest boundary {b!r}: not one of {BOUNDARIES} ({BOUNDARY_ENV})")
    return b


def manifest_engine(engine: str | None = None) -> str:
    e = engine or os.environ.get(ENGINE_ENV) or DEFAULT_ENGINE
    if e not in ENGINES:
        raise ValueError(f"required-manifest engine {e!r}: not one of {ENGINES} ({ENGINE_ENV} / --engine)")
    return e


def engine_explicit(engine: str | None = None) -> bool:
    """True when the caller (arg) or the environment chose the engine; False = the default applied (the one-release cross-check runs)."""
    return bool(engine or os.environ.get(ENGINE_ENV))


def cross_check_enabled() -> bool:
    return os.environ.get(CROSS_CHECK_ENV, "1") != "0"


CROSS_CHECK_RULE = ("one release (from r17-int-2): the manifest of record is the traversal engine's; the inventory (the manifest path of record "
                    "until then) is computed beside it and the two must agree row for row -- identities, annotations, digest, `complete`, the "
                    "unmodelled families -- or the row fails by name; explicit --engine / VERITY_REQUIRED_VALUES_ENGINE skips the check (one side asked for)")


def cross_check_manifests(primary: dict[str, Any], other: dict[str, Any], *, primary_engine: str, other_engine: str) -> dict[str, Any]:
    """[R17 task 3] the two engines' manifests of the SAME Program x workload, compared row for row (both are sorted by `_row_key`).
    Returns {agree, primary/other engine + digest + identities, complete, unmodelled_families, first_difference | None, rule}."""
    pa, pb = primary["identities"], other["identities"]
    first = None
    if pa != pb:
        n = min(len(pa), len(pb))
        i = next((k for k in range(n) if pa[k] != pb[k]), n)
        first = {"index": i, primary_engine: pa[i] if i < len(pa) else None, other_engine: pb[i] if i < len(pb) else None}
    fam_p, fam_o = sorted({k.split(" (")[0] for k in (primary.get("unmodelled") or {})}), sorted({k.split(" (")[0] for k in (other.get("unmodelled") or {})})
    rec = {"primary_engine": primary_engine, "other_engine": other_engine,
           "digests": {primary_engine: primary["manifest_digest"], other_engine: other["manifest_digest"]},
           "identities": {primary_engine: len(pa), other_engine: len(pb)},
           "complete": {primary_engine: primary.get("complete"), other_engine: other.get("complete")},
           "unmodelled_families": {primary_engine: fam_p, other_engine: fam_o},
           "rows_equal": pa == pb, "digest_equal": primary["manifest_digest"] == other["manifest_digest"],
           "program_digest_equal": primary.get("program_digest") == other.get("program_digest"),
           "complete_equal": bool(primary.get("complete")) == bool(other.get("complete")), "unmodelled_equal": fam_p == fam_o,
           "first_difference": first, "rule": CROSS_CHECK_RULE}
    rec["agree"] = bool(rec["rows_equal"] and rec["digest_equal"] and rec["program_digest_equal"] and rec["complete_equal"] and rec["unmodelled_equal"])
    if not rec["agree"]:
        why = []
        if not rec["rows_equal"]:
            why.append(f"rows differ at index {first['index']} of {len(pa)} / {len(pb)}: {primary_engine} {json.dumps(first[primary_engine])[:300]} vs "
                       f"{other_engine} {json.dumps(first[other_engine])[:300]}")
        if not rec["digest_equal"]:
            why.append(f"manifest_digest {primary['manifest_digest'][:16]} ({primary_engine}) != {other['manifest_digest'][:16]} ({other_engine})")
        if not rec["complete_equal"]:
            why.append(f"complete {primary.get('complete')} ({primary_engine}) != {other.get('complete')} ({other_engine})")
        if not rec["unmodelled_equal"]:
            why.append(f"unmodelled families {fam_p} ({primary_engine}) != {fam_o} ({other_engine})")
        rec["why"] = "ENGINES DISAGREE (the row fails by name; `required_values diff` names every identity): " + "; ".join(why)
    return rec


def _inventory_guard() -> dict[str, Any]:
    """[R17 revb F-r17b-09] the manifest header's `guard`: the inventory has no dataflow audit -- said by name, never read as clean."""
    from .required_values import guard_summary
    return guard_summary(None, engine="inventory")


# stays for TP rank manifests (row_pod_tp2.sh, until TP=N) and the harness's BASELINE_ENGINE=v1 oracle
def build_manifest(program: dict[str, Any], workload: dict[str, Any] | None = None, *, request_id: str | int | None = None,
                   rank: int = 0, ranks: int = 1, profile: str | None = None, shared_rope: bool = True, engine: str | None = None) -> dict[str, Any]:
    """The manifest of one request Program (B=1, `--wrapper request`).  `workload` (sweep manifest) supplies request id / arrival step."""
    boundary = manifest_boundary()
    if manifest_engine(engine) == "traversal" or boundary != "module":
        from .required_values import manifest_from_traversal
        man = manifest_from_traversal(program, workload, request_id=request_id, rank=rank, ranks=ranks, profile=profile, shared_rope=shared_rope, boundary=boundary)
        if boundary != "module" and manifest_engine(engine) != "traversal":
            man["cross_check"] = {"skipped": f"boundary {boundary}: the traversal engine's (the inventory engine has no padded-slot rule); engine {manifest_engine(engine)} not consulted"}
        return man
    inst, res = program["instances"], program["result"]
    rows: list[dict] = inst["rows"]
    pdig = (program.get("artifact") or {}).get("program_digest") or res["program"]["digest"]
    attn_obs = res.get("attention_impls_observed") or {}
    prefix = wrapper_prefix_of(rows, attn_obs.keys())
    tgt = (res.get("target") or {}).get("declared") or {}
    fa = None
    for v in attn_obs.values():
        fa = v.get("vllm_flash_attn_version")
        break
    if profile is None:
        cc = tgt.get("compute_capability") or [8, 9]
        profile = "hopper" if (fa == 3 or (fa is None and int(cc[0]) >= 9)) else "ampere"
    q = query_requirements(profile)
    hidden_family = q["attention_hidden_family"]
    # [R15 tp TP-04] a rank Program (derive_step --tp W --rank r) describes ONE rank: the manifest's (rank, ranks) must be that rank's --
    # never a rank-0 manifest for a rank-1 Program, never ranks folded together (TABLE-RULES: `rank` is a key component)
    tpinfo = res.get("tp") if isinstance(res.get("tp"), dict) else {}
    prog_world = int(tpinfo.get("world") or 1)
    prog_rank = int(tpinfo.get("rank") or 0)
    if prog_world > 1 or ranks > 1:
        if int(ranks) != prog_world or int(rank) != prog_rank:
            raise ValueError(f"rank Program mismatch: Program is rank {prog_rank} of world {prog_world}, manifest requested rank {rank} of {ranks}")
        q["required_families"].append("tp_rank_partials")
        q["why"]["tp_rank_partials"] = (f"TP world {prog_world}: every collective of rank {prog_rank} consumes this rank's OWN partial (row-parallel Gemm on the "
                                        "K/W weight shard, masked embedding shard, lm_head logits shard) -> the partial is required per rank IN ADDITION to the "
                                        "collective output the module binds (AllReduce2/AllGather2 of both ranks' partials; TP-04 conservative rule)")
        q["tp"] = {"world": prog_world, "rank": prog_rank, "rule": "TP-04: partials AND collective outputs both required, per rank"}
    tp_ctx_base = {"rank": prog_rank, "world": prog_world}
    sweep = (workload or {}).get("sweep") or {}
    wr = (workload or {}).get("requests") or []
    workload_request_id = wr[0].get("request_id") if wr and isinstance(wr[0], dict) else None   # the workload's own id, recorded beside
    request_id = request_label(request_id, workload)               # `r<k>` (COORD R15 ruling 2); a bare Program is r0
    k_req = int(_REQ_RE.match(request_id).group(1))
    if wr and k_req < len(wr) and isinstance(wr[k_req], dict) and str(wr[k_req].get("request_id")) != str(workload_request_id):
        workload_request_id = wr[k_req].get("request_id")
    arr = sweep.get("arrive_steps") or []
    arrive = int(arr[k_req] if k_req < len(arr) else (arr[0] if arr else 0)) if sweep else 0
    n_seq = 1

    segs = segment_steps(rows)
    identities: list[dict[str, Any]] = []
    fam_counts: dict[str, int] = {f: 0 for f in FAMILIES}
    aliases: dict[str, list] = {}
    unmodelled: dict[str, int] = {}                                              # [MAN-02] families without a rule -> manifest incomplete
    moe_layers_seen: set[int] = set()
    stochastic_selectors: set[str] = set()
    splits_param = has_splits_param(inst.get("params"))                          # [R16 REQ-03] the derived-S request-owned `splits` Input
    # [R19 int-20 promotion] the claims the vocabulary calls interior, per step, and the vocabulary's own claims per step -- the promotion
    # pass after the walk (`promote_interior_crossings`) reads the Program's dataflow for the ones whose output leaves their module
    interior_claims: list[tuple] = []
    named_by_step: dict[int, set[tuple[str, str]]] = {}
    aliased_by_step: dict[int, set[str]] = {}

    def add(family: str, step: int | str, invocation: int, op_path: str, member: str, lo: int, hi: int | None, **extra):
        row = {"family": family, "program_digest": pdig, "request_id": request_id, "step": step, "invocation": invocation,
               "op_path": op_path, "output_member": member, "element_range": [lo, hi], "rank": rank}
        if step != "*":
            row["engine_step"] = arrive + int(step)          # programmed engine step (declared arrival + request-relative step); the key's `step` is request-relative
        row.update(extra)
        identities.append(row)
        fam_counts[family] = fam_counts.get(family, 0) + 1

    # registered inputs (weights): every field the Program's `weights` parameter names, once per rank (step / request = "*")
    wparam = next((t for n, t in inst.get("params", []) if n == "weights"), None)
    wnames = []
    wwidth: dict[str, int] = {}
    if wparam and wparam.startswith("{"):
        wnames = re.findall(r"([A-Za-z0-9_.]+):\s*(?:Array|Value)", wparam)
        # [R15 fp8] the leaf width the Program declares for each field (`Value<8>` e4m3 words, `Value<32>` f32 block scales, `Value<16>`
        # bf16): the identity's dtype label follows the Program, never a literal -- the FP8 checkpoint's weights are not bf16
        for m in re.finditer(r"([A-Za-z0-9_.]+):\s*((?:Array<\d+,)*Value<(\d+)>)", wparam):
            wwidth[m.group(1)] = int(m.group(3))
    if not wnames:
        wnames = list(((res.get("weights") or {}).get("roles") or {}).keys())
    for w in sorted(wnames):
        add("weights", "*", 0, collector_path(w, prefix), "weight", 0, None, dtype={8: "w8", 32: "f32", 16: "bf16"}.get(wwidth.get(w, 16), f"w{wwidth.get(w)}"))

    for s, (lo_i, hi_i) in enumerate(segs):
        step = s
        seg = rows[lo_i: hi_i + 1]
        # group: (module, member) -> {fx nodes in order, row count, width, dtype, family}; attention rows -> hidden geometry
        groups: dict[tuple[str, str], dict[str, Any]] = {}
        attn: dict[str, dict[str, Any]] = {}
        softcap_mods: dict[str, str] = {}                                        # [rev-cb D1] module -> the soft-cap family seen un-hinted
        fam_pos: dict[tuple[str, str, Any], int] = {}                            # [M-0323] (module, family, row) -> same-family ops seen so far
        n_tokens = 0
        # [R15 tp TP-04] which modules issue a collective in this step (the AllReduce2 / AllGather2 rows under them): the Gemm /
        # EmbeddingShard rows of those modules are the rank-local partials, not the module's output
        coll_of: dict[str, str] = {}
        if prog_world > 1:
            for r in seg:
                f0 = spec_family(r["spec"])
                if f0.startswith("AllReduce2"):
                    coll_of[split_name(r.get("name") or "")[0]] = "all_reduce"
                elif f0.startswith("AllGather2"):
                    coll_of[split_name(r.get("name") or "")[0]] = "all_gather"
        epi_mods = epilogue_modules(seg)                                          # [MAN-07] biased linears: BiasAdd produces member 0
        for k_row, r in enumerate(seg):
            i_row = lo_i + k_row                                                 # the row's Program index (the operand runs' `["n", i, ..]`)
            fam = spec_family(r["spec"])
            mod, node, hint, _row = split_name(r.get("name") or "")
            st = spec_statics(r["spec"])
            if is_epilogue_interior(fam, mod, epi_mods) and not coll_of.get(mod):
                if int(st.get("N", 0)) != epi_mods[mod]:                         # rev-cb D2: the epilogue must return the Gemm's width
                    k = f"{fam} under {collector_path(mod, prefix)}: Gemm width {int(st.get('N', 0))} != its BiasAdd epilogue width {epi_mods[mod]} (D2)"
                    unmodelled[k] = unmodelled.get(k, 0) + 1
                else:
                    interior_claims.append((step, mod, fam, r["spec"], st, [i_row]))   # [int-20] interior by the vocabulary: promotable if it crosses
                continue                                                         # the Gemm's pre-bias output: interior, never an identity
            if is_token_select(fam):
                add("sampled_token_ids", step, 0, "runner.sampler", "sampled_token_ids", 0, n_seq, dtype="i64", spec=fam)
                if is_stochastic_select(fam):                                     # [R15 stoch] the per-sequence seed the selector consumed
                    add("sampler_seed", step, 0, "runner.sampler", "seed", 0, n_seq, dtype="i64", spec=fam)
                    stochastic_selectors.add(fam)
                    if splits_param:                                              # [R16 REQ-03] the split count S the top-p mask used at this event
                        add("sampler_splits", step, 0, "runner.sampler", "splits", 0, n_seq, dtype="i32", spec=fam)
                continue
            if fam.startswith("Embedding"):
                n_tokens += 1
            if fam.startswith(("Bf16DivScalar", "Bf16Tanh", "Bf16MulScalar_")) and not hint:
                softcap_mods.setdefault(mod, fam)                                # a soft-cap chain's interior / an UN-hinted last mul
            program_ordinal = program_operand = None
            operand_why = ""
            if fam.startswith(_ORDINAL_FAMILIES):                                # this row's index among the module's same-family rows (per row)
                program_ordinal = fam_pos.get((mod, fam, _row), 0)
                fam_pos[(mod, fam, _row)] = program_ordinal + 1
                program_operand, operand_why = _operand_family(r, rows)          # [rev-pm M-0402] the family whose output this op consumes
            if fam.startswith(("Attention", "AttentionHead")):
                a = attn.setdefault(mod, {"rows": 0, "T_max": 0, "NH": int(st.get("NH", 0)), "KVH": int(st.get("KVH", st.get("NH", 0))), "D": int(st.get("D", 0))})
                a["rows"] += 1
                a["T_max"] = max(a["T_max"], int(st.get("T", 0)))
            try:
                members = members_for(fam, st, mod, hint, tp=(dict(tp_ctx_base, collective=coll_of.get(mod)) if prog_world > 1 else None))
            except UnmodelledFamily as e:
                unmodelled[f"{e.family} under {collector_path(mod, prefix)}"] = unmodelled.get(f"{e.family} under {collector_path(mod, prefix)}", 0) + 1
                continue
            if not members:
                # [R19 int-20] the vocabulary names no member: interior by rule -- unless the Program's dataflow reads the row's output from
                # another module (the promotion pass after the walk); aggregates and structure never promote (`PROMOTION_RULE`)
                if not is_aggregate_family(fam, mod) and not is_structure_family(fam):
                    interior_claims.append((step, mod, fam, r["spec"], st, [i_row]))
                continue
            if members and "." not in mod:
                # [MAN-08, moe M-0583 GATE-PATH] an identity-bearing row whose module path is UNROOTED (a bare leaf such as `gate`: the
                # exporter's nn_module_stack lost the parent) would fold every same-named module of the model into ONE op_path with a
                # concatenated element range (48 routers -> `gate/0` [0, 48*V*T)) that no collector binds -- a Program naming defect,
                # incomplete + named here, not 130 missing / 6240 extra at Commit coverage.  Sampler rows (`argmax_k`) and Const rows
                # legitimately have no module: they never reach this branch (handled above / members == []).
                key = f"{fam} under unrooted module path {mod!r} (Program names {len([q for q in seg if split_name(q.get('name') or '')[0] == mod])} rows of this step under a bare leaf)"
                unmodelled[key] = unmodelled.get(key, 0) + 1
                continue
            for mem in members:
                member, width, dtype = mem[0], mem[1], mem[2]
                extras = dict(mem[3]) if len(mem) > 3 and mem[3] else {}
                if width <= 0:                                                   # [MAN-02] never a zero-width identity
                    unmodelled[f"{fam} under {collector_path(mod, prefix)} ({member}: width 0)"] = unmodelled.get(f"{fam} under {collector_path(mod, prefix)} ({member}: width 0)", 0) + 1
                    continue
                # [R15 dense, manbind M-0323] a declared `producer_ordinal` must be the Program's own: the C2 oracle reads the member at the
                # k-th same-family fold instance of the row, so a rule whose k the Program's row order contradicts would read another op's
                # bytes (the residual for the normed row).  Fail closed, named, rather than emit a wrong ordinal.
                if program_ordinal is not None and "producer_ordinal" in extras and extras["producer_ordinal"] != program_ordinal:
                    key = (f"{fam} under {collector_path(mod, prefix)} ({member}: declared producer_ordinal {extras['producer_ordinal']} but the "
                           f"Program's row order puts this op at {program_ordinal})")
                    unmodelled[key] = unmodelled.get(key, 0) + 1
                    continue
                # [rev-pm M-0402] `producer_operand` = the family whose output the producing op consumes (member "1" <- AddWidenedBf16, the
                # residual sum; member "0" <- MulVecF32, the (1+w)-scaled row): an ORDER-FREE selector for the oracle (a fold whose events are
                # a dependency-respecting permutation of the Program is a Match PASS under `dag`, and the two narrowings are independent, so
                # the event-order ordinal alone could hand a member the other narrowing's bytes).  Confirmed from the Program's own args.
                if program_ordinal is not None and "producer_operand" in extras and extras["producer_operand"] != program_operand:
                    # [rev-pm M-0468] an UNRESOLVABLE operand (parameter ref, args absent) is refused too: a selector the Program cannot
                    # confirm would otherwise be emitted on the strength of its presence, exactly on the Programs where it cannot be checked
                    why = (f"the Program's dataflow feeds this op from {program_operand}" if program_operand is not None
                           else f"{operand_why}: the declared operand cannot be confirmed")
                    key = f"{fam} under {collector_path(mod, prefix)} ({member}: declared producer_operand {extras['producer_operand']} but {why})"
                    unmodelled[key] = unmodelled.get(key, 0) + 1
                    continue
                # [R15 dense] an EXPLICIT member hint on a row (`.../<member>[row]`, the Build's naming of the tensor a module RETURNS)
                # beats a default (un-hinted) claim on the same (module, member): a soft-capped LogitsProcessor's Gemm rows are interior
                # and its `Bf16MulScalar .../out` rows are the logits.  Two hinted families on one member is a Program defect -> unmodelled.
                # (No pre-R15 family names rows `hint == member`: RoPE's hints are `query` / `key` -> members `0` / `1`; unchanged.)
                explicit = hint == member
                g = groups.get((mod, member))
                if g is None:
                    g = groups[(mod, member)] = {"nodes": [], "rows": 0, "width": width, "dtype": dtype, "family": fam, "hinted": explicit, "extras": extras}
                elif g["hinted"] and not explicit:
                    if (g["width"], g["dtype"]) != (width, dtype):              # [rev-cb D2] order-independent: same check as the replace path
                        key = (f"{g['family']} under {collector_path(mod, prefix)} ({member}: hinted producer {g['width']}x{g['dtype']} vs a default "
                               f"claim {fam} {width}x{dtype})")
                        unmodelled[key] = unmodelled.get(key, 0) + 1
                    continue                                                     # default claim under a module with a hinted producer
                elif g["hinted"] and explicit and g["family"] != fam:
                    key = f"{fam} under {collector_path(mod, prefix)} ({member}: two hinted producers {g['family']} / {fam})"
                    unmodelled[key] = unmodelled.get(key, 0) + 1
                    continue
                elif not g["hinted"] and explicit:                              # default rows seen first, now the hinted producer: replace
                    # [rev-cb D2, M-0210] the hinted producer must return the SAME tensor shape the default claim described: a width / dtype
                    # change on replace is a Program defect (a cap that is not of the Gemm's logits) -> unmodelled, not a silent narrowing
                    if (g["width"], g["dtype"]) != (width, dtype):
                        key = (f"{fam} under {collector_path(mod, prefix)} ({member}: hinted producer {width}x{dtype} replaces a default claim "
                               f"{g['family']} {g['width']}x{g['dtype']})")
                        unmodelled[key] = unmodelled.get(key, 0) + 1
                        continue
                    g.update(nodes=[], rows=0, width=width, dtype=dtype, family=fam, hinted=True, extras=extras)
                elif g["hinted"] and explicit and (g["width"], g["dtype"]) != (width, dtype):
                    key = f"{fam} under {collector_path(mod, prefix)} ({member}: hinted rows of unequal width {g['width']} / {width})"
                    unmodelled[key] = unmodelled.get(key, 0) + 1
                    continue
                # [R15 manbind 331c2fccc] the rows of one member carry one set of extras (producer_ordinal): a disagreement is a Program defect
                if extras and g.get("extras", {}) != extras:
                    raise ValueError(f"{mod}/{member} step {step}: the rows of one member disagree on their extras ({g.get('extras')} vs {extras})")
                if node not in g["nodes"]:
                    g["nodes"].append(node)
                g["rows"] += 1
        # [rev-cb D1, M-0210] a module that carries a soft-cap chain (Bf16DivScalar / Bf16Tanh / an UN-hinted Bf16MulScalar) and whose
        # required member is still a DEFAULT claim (no hinted producer) would be required at the Gemm's PRE-cap tensor while the collector
        # binds the module's returned (post-cap) tensor: the identity key carries no family, so nothing would name the miss.  Fail closed.
        for mod, fam in softcap_mods.items():
            for (m, member), g in groups.items():
                if m == mod and not g["hinted"]:
                    key = f"{fam} under {collector_path(mod, prefix)} ({member}: soft-cap chain without a hinted producer; default claim {g['family']})"
                    unmodelled[key] = unmodelled.get(key, 0) + 1
        # shared modules: vLLM's `get_rope` returns ONE RotaryEmbedding instance for identical parameters, registered under the first layer's
        # path and called once per layer -> the Program's per-layer paths alias (first path, invocation = layer order); recorded in `op_path_aliases`
        rope_alias = shared_module_aliases(groups, shared=shared_rope)          # [R17-39 (c)] one rule, signature-grouped, fail-closed
        rope_alias.update(collective_site_aliases(groups))                       # [R19 moe F-r19-moe-1] mid-module collective sites -> (module, k)
        for m, (canon, k) in rope_alias.items():
            if m != canon:
                aliases[collector_path(m, prefix)] = [collector_path(canon, prefix), k]
        named_by_step[step] = set(groups)                                        # [int-20] what the vocabulary claims here (a promoted member never collides silently)
        aliased_by_step[step] = set(rope_alias)
        for (mod, member), g in sorted(groups.items(), key=lambda kv: (_layer_of(kv[0][0]), kv[0])):
            op = collector_path(mod, prefix)
            if mod in rope_alias:
                canon, k = rope_alias[mod]
                add(alias_family(member), step, k, collector_path(canon, prefix), member, 0, g["rows"] * g["width"], rows=g["rows"], width=g["width"],
                    dtype=g["dtype"], spec=g["family"], program_op_path=op, **g.get("extras", {}))
                continue
            # every module of the request wrapper is called ONCE per engine step (prefill: one call over all prompt rows -- the Program's
            # per-row instances / per-row fx nodes are that one call's rows); a module shared across call sites is an alias (above)
            # [TP-04] a rank-local collective input is its own family (omit every partial of a rank = a named family failure)
            fam_label = "tp_rank_partials" if member.startswith(("part_rank", "shard_rank")) else "instance_outputs"
            add(fam_label, step, 0, op, member, 0, g["rows"] * g["width"], rows=g["rows"], width=g["width"], dtype=g["dtype"],
                spec=g["family"], fx_nodes=len(g["nodes"]), **g.get("extras", {}))
            if g["family"].startswith("Embedding"):                              # Embedding_v1 (TP1) or EmbeddingShard_v1 (TP rank): the ids input
                add("input_token_ids", step, 0, op, "in", 0, g["rows"], rows=g["rows"], width=1, dtype="i32")
        # [R14 moe] the fused-MoE block: six planes per MoE layer per step from the Program's MoE instances (M routed tokens x TOPK pairs);
        # a manifest without them would let a collector whose MoE producer never fired pass (the R-14 zero/zero shape)
        for mod, g in sorted(moe_geometry(seg).items(), key=lambda kv: _layer_of(kv[0])):
            layer = _layer_of(mod)
            if g["M"] <= 0 or g["TOPK"] <= 0:
                continue
            if g["expert_gemm_rows"] != g["M"] * g["TOPK"]:
                raise ValueError(f"MoE layer {layer} step {step}: {g['expert_gemm_rows']} expert-GEMM instances != M*TOPK = {g['M']}*{g['TOPK']}")
            moe_layers_seen.add(layer)
            for plane, n, dtype in moe_identities(g):
                add("moe_block_stream", step, 0, f"moe/L{layer}", plane, 0, n, dtype=dtype, rows=g["M"], topk=g["TOPK"], moe_module=collector_path(mod, prefix))
        for mod, a in sorted(attn.items()):
            if hidden_family.startswith("fa3"):
                geo = _fa3_geometry(a["T_max"], a["rows"], a["NH"], a["KVH"], a["D"], n_seq)
            else:
                # [R15] the layer's OBSERVED sliding window (result.json attention_impls_observed, the fact vLLM hands FA2): [W-1, 0] -> windowed
                win = (attn_obs.get(collector_path(mod, prefix)) or attn_obs.get(mod) or {}).get("sliding_window")
                w0 = win[0] if isinstance(win, (list, tuple)) and win else (win if isinstance(win, int) else -1)
                geo = _fa2_geometry(a["T_max"], a["rows"], a["NH"], a["KVH"], a["D"], n_seq, windowed=(w0 is not None and int(w0) >= 0))
            HB, M, NB, BN, gname = geo
            layer = int(mod.split(".layers.")[1].split(".")[0]) if ".layers." in mod else -1
            tap = "fa3.m1" if hidden_family.startswith("fa3") else "fa2.m1"
            add(hidden_family, step, 0, f"{tap}/L{layer}", gname, 0, mat_total_words(HB, M, NB, BN, a["D"]), dtype="u32",
                geometry={"HB": HB, "M": M, "NB": NB, "BN": BN, "D": a["D"], "seqlen_k": a["T_max"], "rows": a["rows"]}, attention_module=collector_path(mod, prefix))

    # [R19 int-20] the promotion pass: the Program's dataflow over ALL rows (a consumer may read an earlier step's producer), then every
    # interior claim whose output another module reads becomes a required member -- the same pass the traversal runs (cross-check: row for row)
    crosses, consumers = program_crossings(rows)
    promo = promote_interior_crossings(interior_claims, crosses.__getitem__, lambda i: consumers.get(i, ()), named=named_by_step.get, aliased=aliased_by_step.get, prefix=prefix)
    for text, n in promo["faults"].items():
        unmodelled[text] = unmodelled.get(text, 0) + n
    for d in promo["identities"]:
        add("instance_outputs", d["step"], 0, d["op_path"], d["member"], 0, d["rows"] * d["width"], **promoted_annotations(d))
    if promo["identities"] or promo["faults"]:
        q["promotion"] = promotion_header(promo)

    if stochastic_selectors:
        q["required_families"].append("sampler_seed")
        q["why"]["sampler_seed"] = (f"the stochastic selector(s) {sorted(stochastic_selectors)} consume the per-request seed as a registered runtime input "
                                    "(Philox(seed, pos) Gumbel noise): the committed sampled id is reproducible from the committed logits only with it")
        q["stochastic_selectors"] = sorted(stochastic_selectors)
        if splits_param:
            q["required_families"].append("sampler_splits")
            q["why"]["sampler_splits"] = ("the Program's request-owned `splits` input (one i32 per sampling event: the split count S the top-p mask kernel "
                                          "launched with, SplitsFor_v1(n_live, num_SMs)) is read by the stochastic selector as its 6th operand: the committed "
                                          "sampled id is reproducible from the committed logits only with the seed AND that S (REQ-03; F-stoch-01 is a wrong S)")
    if moe_layers_seen:
        q["required_families"].append("moe_block_stream")
        q["why"]["moe_block_stream"] = ("the query's MoE families (MoeRouterTopK weights/ids, MoeExpertGemm up, SiluMul act, MoeExpertGemmW down, "
                                        "MoeSum) are produced inside one module forward -> the six-plane block stream of every MoE layer x step is required")
        q["moe_layers"] = sorted(moe_layers_seen)
    identities.sort(key=_row_key)
    digest = manifest_digest(identities)
    return {"schema": SCHEMA, "identity_key": list(IDENTITY_KEY), "program_digest": pdig,
            "program": {"dir": program.get("dir"), "function": res["program"].get("function"), "wrapper": res.get("wrapper"), "LP": res.get("LP"), "T": res.get("T"),
                        "model_pin": res.get("model_pin"), "artifact_identity": (program.get("artifact") or {}).get("identity"),
                        "instances": len(rows), "instances_digest": inst.get("digest")},
            "workload": {"row_id": sweep.get("row_id"), "batch": sweep.get("batch"), "request_id": request_id, "workload_request_id": workload_request_id,
                         "arrive_step": arrive} if workload else None,
            "query": q, "wrapper_prefix": prefix, "op_path_aliases": aliases, "target": tgt, "flash_attn_version": fa, "ranks": ranks, "rank": rank,
            "steps": list(range(len(segs))), "engine_steps": [arrive + s for s in range(len(segs))], "requests": [request_id],
            "populations": {"identities": len(identities), "by_family": {k: v for k, v in fam_counts.items() if v},
                            "steps": len(segs), "attention_layers": len({r["op_path"] for r in identities if r["family"] == hidden_family})},
            "complete": not unmodelled, "unmodelled": unmodelled,                # [MAN-02] fail closed: coverage_check refuses an incomplete manifest
            "manifest_digest": digest, "identities": identities, "guard": _inventory_guard()}


def build_global_manifest(fixture: dict[str, Any], programs: dict[str, dict[str, Any]], *, workload_digest: str | None = None,
                          rank: int = 0, ranks: int = 1, profile: str | None = None, engine: str | None = None) -> dict[str, Any]:
    """The manifest of gprog's GLOBAL Program (GP-01 `Workload_v1{requests, shared}`): `fixture` = the workload declaration
    (`requests`: request_id, index, prompt_len, max_tokens, arrive_step), `programs` = {request_id: loaded request Program (LP = prompt_len,
    T = max_tokens - 1)}.  Identity reading = TABLE-RULES' global reading: `program_digest` = the workload Program digest (component request
    digests recorded beside), `step` = request-relative (0 = that request's prefill), programmed engine step = arrive_step + step
    (`chronology.json` units).  Per engine step the co-scheduled requests share ONE packed member per module; request r's identity is its
    own row block, addressed REQUEST-LOCALLY (`element_range` = [0, rows x width) of r's block, `rows` = r's tokens in the step): the block's
    position inside the packed member is the engine's row order -- a runtime fact the binding map records per entry (`element_offset`),
    never a declaration (vLLM's InputBatch condenses finished slots, so declaration order is NOT the row order at B>1).  The declared
    composition of every engine step -- which requests, how many rows each -- is `engine_steps[e].members`; `binding.composition_check`
    holds the map to exactly that set (an omitted / relabelled / duplicated request fails named).  The sampler ids member has one element
    per sequence sampled ([0, 1) of r's slot); the attention hidden stream is ONE launch per layer per engine step over all co-scheduled
    sequences (request_id `*`, members listed).  `program_digest` = `workload_digest` (gprog's Workload_v1 Program, kind `workload`) or,
    until the Build carries it, a sha256 fold of the sorted component digests (kind `components-fold`, recorded)."""
    if manifest_boundary() != "module":                                            # module-pad is B=1 (`build`) in this cut: never silently ignored
        raise ValueError(f"required-manifest boundary {manifest_boundary()!r} ({BOUNDARY_ENV}) is not implemented for the global (B>=2) manifest in this cut; "
                         "unset it or build the padded row at B=1")
    if manifest_engine(engine) == "traversal":
        from .required_values import global_manifest_from_traversal
        return global_manifest_from_traversal(fixture, programs, workload_digest=workload_digest, rank=rank, ranks=ranks, profile=profile)
    # [R15 COORD 2] request ids are `r<index>` (declaration position); the fixture's own `request_id` (a workload UUID) is recorded beside
    declared_in = [dict(r, index=int(r.get("index", i))) for i, r in enumerate(fixture["requests"])]
    if len({r["index"] for r in declared_in}) != len(declared_in):
        raise ValueError("workload declaration: duplicate request index")
    reqs = []
    for r in sorted(declared_in, key=lambda r: (int(r.get("arrive_step", 0)), r["index"])):
        rid = f"r{r['index']}"
        reqs.append(dict(r, workload_request_id=r.get("request_id"), request_id=rid))
    parts: dict[str, dict[str, Any]] = {}
    for r in reqs:
        rid = r["request_id"]
        prog = programs.get(rid) if rid in programs else programs.get(str(r.get("workload_request_id")))
        if prog is None:
            raise KeyError(f"no request Program for {rid} (workload id {r.get('workload_request_id')})")
        parts[rid] = build_manifest(prog, {"sweep": {"arrive_steps": [int(r.get("arrive_step", 0))], "row_id": fixture.get("fixture"), "batch": len(reqs)}},
                                    request_id=rid, rank=rank, ranks=ranks, profile=profile)
        got_steps = len(parts[rid]["steps"])
        if got_steps != int(r["max_tokens"]):
            raise ValueError(f"request {rid}: Program has {got_steps} steps, the declaration caps {r['max_tokens']} tokens (LP={r['prompt_len']}): wrong component")
    component_digests = sorted({p["program_digest"] for p in parts.values()})
    pdig = str(workload_digest) if workload_digest else components_fold_digest(component_digests)
    pdig_kind = "workload" if workload_digest else "components-fold"
    # engine step -> members (request, request-relative step), from the declaration (arrive_step + s), never from an observation;
    # the list order is (arrive_step, index) for READING only -- the engine's row order is bound per entry, not declared here
    by_engine: dict[int, list[tuple[str, int]]] = {}
    for r in reqs:
        rid = r["request_id"]
        for s in range(int(r["max_tokens"])):
            by_engine.setdefault(int(r.get("arrive_step", 0)) + s, []).append((rid, s))
    hidden_family = parts[reqs[0]["request_id"]]["query"]["attention_hidden_family"]
    ids: list[dict[str, Any]] = []
    seen_w: set = set()
    for rid, p in parts.items():
        for row in p["identities"]:
            if row["family"] == "weights":
                if (row["rank"], row["op_path"]) in seen_w:
                    continue
                seen_w.add((row["rank"], row["op_path"]))
                ids.append(dict(row, program_digest=pdig, component_digest=row["program_digest"]))
    engine_steps_out: list[dict[str, Any]] = []
    for e in sorted(by_engine):
        members = by_engine[e]
        # per-request rows of this engine step (the request's tokens in the step: prefill LP or decode 1), from the component manifests
        rows_of: dict[str, int] = {}
        for rid, s in members:
            emb = next(x for x in parts[rid]["identities"] if x["family"] == "input_token_ids" and x["step"] == s)
            rows_of[rid] = int(emb["rows"])
        attn_union: dict[str, dict[str, Any]] = {}
        for rid, s in members:
            n_rows = rows_of[rid]
            for row in parts[rid]["identities"]:
                if row["step"] != s or row["family"] == "weights":
                    continue
                # request-local addressing: the component row's own [0, rows x width) IS the identity's range inside r's block of the
                # packed member (the sampler ids row is [0, 1) of r's slot); `rows` names the block height the composition check holds
                g = dict(row, program_digest=pdig, component_digest=row["program_digest"], engine_step=e)
                # [R15 tp TP-04] a rank-local partial (part_rank<r>: (tokens, N) before its all-reduce; shard_rank<r>: (n_seq, V/W) before its
                # all-gather) is packed over the co-scheduled requests exactly like the module output it precedes -> same request-local block
                if row["family"] in ("instance_outputs", "input_token_ids", "tp_rank_partials"):
                    # [R17 reqval C2 audit finding 1] the identity's `rows` is the COMPONENT's own block height ((hi - lo) / width): per-token
                    # members carry the request's tokens in the step (== n_rows), per-SEQUENCE members (logits_processor/out, an lm_head
                    # shard_rank<r>) carry 1.  Overriding every identity with n_rows annotated the prefill logits row with LP (245 on the
                    # B8 record of record; digest-neutral, no consumer read it) -- a false statement on the manifest.  The per-token
                    # reference member the composition check reads (input_token_ids) is asserted to agree with the engine step's count.
                    lo, hi = row["element_range"]
                    if hi is not None and hi <= lo:                                   # [MAN-02] never a zero-width identity
                        raise ValueError(f"request {rid} step {s} {row['op_path']}/{row['output_member']}: zero-width identity {row['element_range']}")
                    if row["family"] == "input_token_ids" and int(row.get("rows") or 0) != n_rows:
                        raise ValueError(f"request {rid} step {s} {row['op_path']}/in: ids rows {row.get('rows')} != the engine step's {n_rows}")
                    ids.append(g)
                elif row["family"] == hidden_family:
                    a = attn_union.setdefault(row["op_path"], {"requests": [], "n_seq": 0, "max_rows": 0, "T_max": 0, "geo": row["geometry"], "module": row.get("attention_module")})
                    a["requests"].append(rid); a["n_seq"] += 1
                    a["max_rows"] = max(a["max_rows"], int(row["geometry"]["rows"])); a["T_max"] = max(a["T_max"], int(row["geometry"]["seqlen_k"]))
                else:
                    ids.append(g)
        for op, a in sorted(attn_union.items()):
            geo0 = a["geo"]
            NH = geo0["HB"] // 1 if geo0["M"] != 1 and geo0["rows"] > 1 else None      # recover NH/KVH/D from the component geometry + module
            comp = parts[a["requests"][0]]
            attn_row = next(x for x in comp["identities"] if x["family"] == hidden_family and x["op_path"] == op)
            D = int(attn_row["geometry"]["D"])
            # NH / KVH from the component's prefill geometry (HB = NH x 1 seq, decode HB = KVH, M = NH/KVH) -- both are in the component rows
            pre = next(x for x in comp["identities"] if x["family"] == hidden_family and x["op_path"] == op and x["step"] == 0)
            dec = next((x for x in comp["identities"] if x["family"] == hidden_family and x["op_path"] == op and x["step"] == 1), None)
            NH = int(pre["geometry"]["HB"])
            KVH = int(dec["geometry"]["HB"]) if dec is not None and int(dec["geometry"]["M"]) > 1 else NH
            fn = _fa3_geometry if hidden_family.startswith("fa3") else _fa2_geometry
            HB, M, NB, BN, gname = fn(a["T_max"], a["max_rows"] * a["n_seq"], NH, KVH, D, a["n_seq"])
            ids.append({"family": hidden_family, "program_digest": pdig, "request_id": "*", "requests": a["requests"], "step": "*", "engine_step": e, "invocation": 0,
                        "op_path": op, "output_member": gname, "element_range": [0, mat_total_words(HB, M, NB, BN, D)], "rank": rank, "dtype": "u32",
                        "geometry": {"HB": HB, "M": M, "NB": NB, "BN": BN, "D": D, "seqlen_k": a["T_max"], "rows": a["max_rows"] * a["n_seq"], "n_seq": a["n_seq"]},
                        "attention_module": a["module"]})
        engine_steps_out.append({"engine_step": e, "members": [{"request_id": rid, "step": s, "rows": rows_of[rid]} for rid, s in members],
                                 "rows": sum(rows_of.values()), "n_seq": len(members)})
    ids.sort(key=_row_key)
    fam: dict[str, int] = {}
    for r in ids:
        fam[r["family"]] = fam.get(r["family"], 0) + 1
    # the reference members the composition check reads the map's per-step request->rows partition from (model-specific op paths)
    ref = {}
    for r in ids:
        if r["family"] in ("input_token_ids", "sampled_token_ids") and r["family"] not in ref:
            ref[r["family"]] = {"op_path": r["op_path"], "output_member": r["output_member"], "per_row": 1}
    return {"schema": SCHEMA, "identity_key": list(IDENTITY_KEY), "program_digest": pdig, "program_digest_kind": pdig_kind, "component_digests": component_digests,
            "global": True, "addressing": "request-local", "composition_reference": ref,
            "fixture": {k: v for k, v in fixture.items() if k in ("schema", "fixture", "model", "revision", "sampling", "expected_engine_steps", "case")}
                       | ({"row_id": fixture["sweep"].get("row_id"), "seed": fixture["sweep"].get("seed")} if isinstance(fixture.get("sweep"), dict) else {}),
            "requests_declared": [{k: r.get(k) for k in ("request_id", "workload_request_id", "index", "prompt_len", "max_tokens", "arrive_step")} for r in reqs],
            "components": {rid: {"program_digest": p["program_digest"], "steps": len(p["steps"]), "LP": p["program"]["LP"], "T": p["program"]["T"],
                                 "workload_request_id": next(r.get("workload_request_id") for r in reqs if r["request_id"] == rid)} for rid, p in parts.items()},
            "complete": all(p.get("complete", True) for p in parts.values()),          # [MAN-02] an unmodelled family in any component -> incomplete
            "unmodelled": {k: v for p in parts.values() for k, v in (p.get("unmodelled") or {}).items()},
            "query": parts[reqs[0]["request_id"]]["query"], "op_path_aliases": parts[reqs[0]["request_id"]]["op_path_aliases"], "ranks": ranks, "rank": rank,
            "engine_steps": engine_steps_out, "steps": sorted(by_engine), "requests": [r["request_id"] for r in reqs],
            "populations": {"identities": len(ids), "by_family": fam, "engine_steps": len(by_engine), "requests": len(reqs)},
            "manifest_digest": manifest_digest(ids), "identities": ids, "guard": _inventory_guard()}


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


def tp_peer_binding(parts: list[dict[str, Any]]) -> dict[str, Any]:
    """[R17 reqval finding 7] The TP merge's identity-level cross-rank check (see `TP_PEER_BINDING_RULE`).  `parts` are the rank manifests
    (one per rank).  Returns {world, collective_outputs, peer_partials, bound, n_unbound, unbound: [first 16 by name], rule}."""
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
            elif r["family"] == "instance_outputs" and str(r.get("spec") or "").startswith(("AllReduce2", "AllGather2")):
                outputs.append((str(r["request_id"]), int(r["step"]), int(r.get("invocation", 0)), r["op_path"], int(r["rank"]),
                                "all_gather" if str(r["spec"]).startswith("AllGather2") else "all_reduce", _width(r)))
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


def merge_manifests(parts: list[dict[str, Any]]) -> dict[str, Any]:
    """Global manifest of several parts.  Two cases, decided by what the parts ARE (never by an option):
    * request components of ONE Program view (gprog's B>1 fixture until the global Program carries the request dimension; all parts the
      same rank): the union of the per-request identity sets under ONE key digest (`components-fold`, the component digests beside);
      the weights are shared and appear once per rank.
    * [R15 tp TP-04] the RANK Programs of a TP row (parts carry `ranks` W > 1 and distinct `rank`): one part per rank, ranks 0..W-1 each
      exactly once (a missing or duplicated rank is refused, never folded); every identity row KEEPS its rank's Program digest (a
      consumer keys rank r's rows by `program_digest_per_rank[r]`), the header `program_digest` is one string -- the labelled fold of the
      per-rank Program digests (`tp-ranks-fold`) -- because manbind's loader takes one string (rev-cb M-0108)."""
    comp = sorted({p["program_digest"] for p in parts})
    worlds = sorted({int(p.get("ranks") or 1) for p in parts})
    world = max(worlds)
    ranks_seen = [int(p.get("rank", 0)) for p in parts]
    tp_parts = world > 1
    # [R15 rev-pm M-0339 T1] every part must declare the SAME world: a part with ranks=1 (a TP1 manifest hand-labelled `rank 1`) among
    # ranks=2 parts was merged as "world = max" before -- refused here, named
    if tp_parts and len(worlds) != 1:
        raise ValueError(f"TP manifest: the parts declare different worlds {worlds} (every rank part must carry ranks == {world})")
    if tp_parts and sorted(set(ranks_seen)) != list(range(world)):
        raise ValueError(f"TP manifest: parts cover ranks {sorted(set(ranks_seen))}, world {world} needs every rank 0..{world - 1}")
    if tp_parts and len(ranks_seen) != len(set(ranks_seen)):
        raise ValueError(f"TP manifest: a rank appears twice in the parts ({ranks_seen}); merge each rank's request components first")
    if tp_parts:
        # [R15 rev-pm M-0339 T2] the rank is a structural input of a rank Program (shard bounds, peer signature): two ranks with ONE
        # Program digest are one rank Program labelled twice -- refused, never folded to that single digest
        per_rank_dig = {int(p.get("rank", 0)): p["program_digest"] for p in parts}
        if len(set(per_rank_dig.values())) != len(per_rank_dig):
            dup = sorted(r for r, d in per_rank_dig.items() if list(per_rank_dig.values()).count(d) > 1)
            raise ValueError(f"TP manifest: ranks {dup} carry the same Program digest {per_rank_dig[dup[0]][:16]} (one rank Program labelled twice; "
                             f"a rank Program is specific to its rank)")
        pdig = hashlib.sha256(("tp-ranks/" + ",".join(comp)).encode()).hexdigest()
        # a B>=2 TP row: every rank's part is the global manifest of the SAME declaration -> identical engine-step composition on every rank
        es = [json.dumps(p.get("engine_steps") if p.get("global") else None, sort_keys=True) for p in parts]
        if len(set(es)) != 1:
            raise ValueError("TP manifest: the rank parts declare different engine-step compositions (not the same workload on every rank)")
    else:
        pdig = components_fold_digest(comp)
    ids: list[dict[str, Any]] = []
    seen_w: set[tuple] = set()
    for p in parts:
        for r in p["identities"]:
            if r["family"] == "weights":
                k = (r["rank"], r["op_path"])
                if k in seen_w:
                    continue
                seen_w.add(k)
            ids.append(r if tp_parts else dict(r, program_digest=pdig, component_digest=r["program_digest"]))      # one key digest for a components merge
    ids.sort(key=_row_key)
    fam: dict[str, int] = {}
    for r in ids:
        fam[r["family"]] = fam.get(r["family"], 0) + 1
    unmodelled = {k: v for p in parts for k, v in (p.get("unmodelled") or {}).items()}
    # [R17 revb F-r17b-38] the parts' `op_path_aliases` (shared modules: per-layer Program path -> [canonical path, invocation]) TRAVEL with the
    # merge: every rank part / request component carries the table its identities are addressed through, and a merged manifest whose
    # identities sit at invocation >= 1 while the table is gone is unreachable from any Program row (the TP2 B1 row of record on r17-int-2
    # read vus_outside_query 960 + identities_without_rows 960 on every rank for exactly this).  Union; parts disagreeing on one alias are
    # not one Program's parts -- refused by name, never folded
    aliases: dict[str, list] = {}
    for p in parts:
        for k, v in (p.get("op_path_aliases") or {}).items():
            want = [str(v[0]), int(v[1])]
            if k in aliases and aliases[k] != want:
                raise ValueError(f"merge: the parts disagree on alias {k!r} ({aliases[k]} vs {want}); one Program's parts share one alias table")
            aliases[k] = want
    binding = tp_peer_binding(parts) if tp_parts else None
    if binding and binding["unbound"]:
        unmodelled[f"tp peer binding: {binding['n_unbound']} collective output(s) / partial(s) without the peer identity (first: {binding['unbound'][0]})"] = binding["n_unbound"]
    return {"schema": SCHEMA, "identity_key": list(IDENTITY_KEY), "program_digest": pdig,
            "program_digest_kind": ("tp-ranks-fold" if tp_parts else "components-fold"), "component_digests": comp,
            "parts": [{"program_digest": p["program_digest"], "request_id": p["requests"], "steps": p["steps"], "rank": int(p.get("rank", 0))} for p in parts],
            "program_digest_per_rank": ({str(int(p.get("rank", 0))): p["program_digest"] for p in parts} if tp_parts else None),
            "query": parts[0]["query"], "op_path_aliases": aliases, "ranks": world, "steps": sorted({s for p in parts for s in p["steps"]}),
            # TP: the same requests on every rank (order-preserving union); components: the concatenation of the parts' requests
            "requests": (list(dict.fromkeys(rq for p in parts for rq in p["requests"])) if tp_parts else [rq for p in parts for rq in p["requests"]]),
            # a TP merge of GLOBAL (B>=2) rank manifests keeps the declaration the loader's composition check reads (identical on every rank)
            **({"global": True, "engine_steps": parts[0].get("engine_steps"), "components": parts[0].get("components"), "addressing": parts[0].get("addressing"),
                "requests_declared": parts[0].get("requests_declared"), "composition_reference": parts[0].get("composition_reference")}
               if tp_parts and parts[0].get("global") else {}),
            "complete": all(p.get("complete", True) for p in parts) and not (binding and binding["unbound"]), "unmodelled": unmodelled,
            "populations": {"identities": len(ids), "by_family": fam}, "manifest_digest": manifest_digest(ids), "identities": ids,
            # [R17 reqval finding 7] the collectives, identity for identity: every AllReduce2/AllGather2 output on rank r has the peer's own
            # partial on every other rank at the same (request, step, invocation, module), and every partial has its collective output there
            "tp_peer_binding": binding,
            # [R17 F-r17b-09] the guard of the merge: armed only when every part's was; the parts' states kept beside
            "guard": {"engine": (lambda es: es[0] if len(es) == 1 else es)(sorted({str((p.get("guard") or {}).get("engine", "inventory")) for p in parts})),
                      "armed": all(bool((p.get("guard") or {}).get("armed")) for p in parts),
                      "available": all(bool((p.get("guard") or {}).get("available")) for p in parts),
                      "fully_wired": all(bool((p.get("guard") or {}).get("fully_wired")) for p in parts),
                      "flags": ({k: sum(int(((p.get("guard") or {}).get("flags") or {}).get(k) or 0) for p in parts)
                                 for k in ("unnamed_crossing", "unnamed_input", "hidden_bn_disagreement")}
                                if all((p.get("guard") or {}).get("flags") is not None for p in parts) else None),
                      "named_input": sorted({s for p in parts for s in ((p.get("guard") or {}).get("named_input") or [])}),
                      "parts": [p.get("guard") for p in parts], "rule": (parts[0].get("guard") or {}).get("rule")}}


# ---------------------------------------------------------------------------------------------------------------------------------
# check: committed identities (the collector's per-step layouts) vs the manifest
# ---------------------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------------------------------


def _main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--engine", choices=ENGINES, default=None,
                    help=f"[R16 reqval] population engine: inventory (default; the R14/R15 per-family walk) | traversal (required_values.Traversal); env {ENGINE_ENV}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="Program artifact dir(s) -> manifest.json")
    b.add_argument("--program", action="append", required=True, help="derive_step output dir (repeat for a B>1 fixture: one per request Program)")
    b.add_argument("--workload", default=None, help="sweep workload manifest (request ids / arrival steps)")
    b.add_argument("--request-id", action="append", default=None)
    b.add_argument("--profile", default=None, help="ampere | hopper (default: from the Program's target / FA version)")
    b.add_argument("--rank", type=int, action="append", default=None, help="TP rank of the Program (repeat per --program for a TP row: --program r0dir --rank 0 --program r1dir --rank 1)")
    b.add_argument("--ranks", type=int, default=1, help="TP world size (1 = TP1)")
    b.add_argument("--out", required=True)
    b.add_argument("--compact", action="store_true", help="omit the identities list from stdout summary (file always complete)")
    g = sub.add_parser("build-global", help="gprog fixture OR sweep workload manifest (requests: request_id, prompt_len, max_tokens, arrive_step) + one request Program per shape -> global manifest")
    g.add_argument("--fixture", required=True, help="fixture json or workloads/<row>.json (both carry `requests`)")
    g.add_argument("--programs-root", action="append", required=True, help="dir holding one derive_step output per request shape, named *LP<prompt_len>_T<max_tokens-1> "
                                                                             "(TP row: repeat per rank with --rank r -- one root per rank Program set, merged with the rank rule)")
    g.add_argument("--workload-digest", action="append", default=None, help="gprog's Workload_v1 Program digest (the key's program_digest); default: sorted component digests (TP: repeat per rank)")
    g.add_argument("--profile", default=None)
    g.add_argument("--rank", type=int, action="append", default=None, help="TP rank of the --programs-root at the same position (default 0)")
    g.add_argument("--ranks", type=int, default=1)
    g.add_argument("--out", required=True)
    v = sub.add_parser("verify", help="[rev-cb F-cb-11] is <manifest> THE manifest of record of this Program x workload?  Rebuilds it from the Build "
                                      "(B=1: --program; B>=2: --fixture + --programs-root, as build/build-global) and compares digests: exit 0 iff equal")
    v.add_argument("--manifest", required=True, help="the manifest the Commit used (evidence dir manifest.json / population.manifest path)")
    v.add_argument("--program", action="append", default=None, help="B=1: the Build's derive_step dir (build_request); a TP row repeats it per rank with --rank, as build")
    v.add_argument("--workload", default=None, help="B=1: the sweep workload manifest (request ids / arrivals) the build used")
    v.add_argument("--fixture", default=None, help="B>=2: the workload manifest / fixture (as build-global)")
    v.add_argument("--programs-root", action="append", default=None, help="B>=2: the Build dir holding the request Programs; a TP row repeats it per rank with --rank, as build-global")
    v.add_argument("--workload-digest", action="append", default=None, help="B>=2: GP-01's workload digest (one per --programs-root, as build-global)")
    v.add_argument("--profile", default=None)
    v.add_argument("--rank", type=int, action="append", default=None, help="TP rank (one per --program / --programs-root; default 0)")
    v.add_argument("--ranks", type=int, default=1)
    v.add_argument("--out", default=None, help="write the verification record (json)")
    c = sub.add_parser("check", help="manifest.json + layouts.json ({step: [{name, numel|shape}]}) -> coverage result")
    c.add_argument("--manifest", required=True)
    c.add_argument("--layouts", required=True)
    c.add_argument("--request-id", default=None)
    c.add_argument("--rank", type=int, default=0)
    c.add_argument("--out", default=None)
    a = ap.parse_args()
    engine = manifest_engine(a.engine)
    # [R17 task 3] the one-release cross-check: the default engine applied (nothing explicit) -> the other engine is computed beside it
    do_cross = engine == "traversal" and not engine_explicit(a.engine) and cross_check_enabled() and manifest_boundary() == "module"

    def _cross_or_fail(man: dict[str, Any], rebuild_with: Any, out_path: str) -> int | None:
        """Attach `cross_check` to `man`, or write the disagreement beside `out_path` and return 1 (the row fails by name)."""
        if not do_cross:
            if manifest_boundary() != "module":                                   # a non-module boundary is the traversal engine's only: SKIPPED by name
                man["cross_check"] = {"skipped": f"boundary {manifest_boundary()} ({BOUNDARY_ENV}): the traversal engine's; the inventory engine has no padded-slot rule"}
            return None
        cc = cross_check_manifests(man, rebuild_with("inventory"), primary_engine="traversal", other_engine="inventory")
        man["cross_check"] = cc
        if cc["agree"]:
            return None
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        json.dump({k: v for k, v in man.items() if k != "identities"}, open(out_path + ".cross_check_fail.json", "w"), indent=1, default=str)
        print(json.dumps(cc, indent=1, default=str)[:6000])
        print(f"[manifest] FAIL: {cc['why']} -> no manifest written ({out_path}.cross_check_fail.json)")
        return 1

    if a.cmd == "build":
        wl = json.load(open(a.workload)) if a.workload else None
        parts = []
        rks = a.rank or [0]
        if len(rks) not in (1, len(a.program)):
            raise SystemExit(f"--rank given {len(rks)} times for {len(a.program)} programs")
        for i, d in enumerate(a.program):
            rk = rks[i] if len(rks) > 1 else rks[0]
            # a TP row: one Program per rank (same request) -> request id from --request-id or the workload, not the program index
            per_rank_parts = a.ranks > 1 and len(a.program) == a.ranks and len(rks) == len(a.program)
            rid = (a.request_id[i] if a.request_id and i < len(a.request_id) else (None if per_rank_parts else (i if len(a.program) > 1 else None)))
            parts.append(build_manifest(load_program(d), wl, request_id=rid, rank=rk, ranks=a.ranks, profile=a.profile, engine=engine))
        man = parts[0] if len(parts) == 1 else merge_manifests(parts)

        def _rebuild_b1(eng: str) -> dict[str, Any]:
            ps = []
            for i, d in enumerate(a.program):
                rk = rks[i] if len(rks) > 1 else rks[0]
                per_rank_parts = a.ranks > 1 and len(a.program) == a.ranks and len(rks) == len(a.program)
                rid = (a.request_id[i] if a.request_id and i < len(a.request_id) else (None if per_rank_parts else (i if len(a.program) > 1 else None)))
                ps.append(build_manifest(load_program(d), wl, request_id=rid, rank=rk, ranks=a.ranks, profile=a.profile, engine=eng))
            return ps[0] if len(ps) == 1 else merge_manifests(ps)

        rc = _cross_or_fail(man, _rebuild_b1, a.out)
        if rc:
            return rc
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(man, open(a.out, "w"), indent=None, separators=(",", ":"), sort_keys=False)
        summ = {k: v for k, v in man.items() if k != "identities"}
        print(json.dumps(summ, indent=1, default=str))
        print(f"[manifest] {man['populations']['identities']} identities, digest {man['manifest_digest']} -> {a.out}")
        return 0
    if a.cmd == "verify":
        # the manifest of record is a FUNCTION of the Program x declared workload (deterministic, digested over its rows): a weakened manifest
        # present in the evidence dir has the wrong digest here even when its own manifest_digest is honest (rev-cb M-0334 F-cb-11 residual)
        used = json.load(open(a.manifest))

        def _rebuild(eng: str) -> dict[str, Any]:
            # [R16 reqval] the same shapes as `build` / `build-global`: one Program (B=1), one root (B>=2), or one per rank (a TP row: the
            # per-rank manifests merged with the rank rule, exactly as the manifest of record was built)
            if a.program:
                wl = json.load(open(a.workload)) if a.workload else None
                rks = list(a.rank) if a.rank else [0] * len(a.program)
                if len(rks) != len(a.program):
                    raise SystemExit(f"verify: {len(a.program)} --program, {len(rks)} --rank: give one per rank")
                if a.ranks > 1 and len(a.program) > 1 and sorted(rks) != list(range(a.ranks)):
                    raise SystemExit(f"verify: TP world {a.ranks} needs one --program per rank 0..{a.ranks - 1} (got ranks {rks})")
                parts = [build_manifest(load_program(d), wl, rank=rk, ranks=a.ranks, profile=a.profile, engine=eng) for d, rk in zip(a.program, rks)]
                return parts[0] if len(parts) == 1 else merge_manifests(parts)
            if a.fixture and a.programs_root:
                fx = json.load(open(a.fixture))
                roots = list(a.programs_root)
                rks = list(a.rank) if a.rank else [0] * len(roots)
                wds = list(a.workload_digest) if a.workload_digest else [None] * len(roots)
                if len(rks) != len(roots) or len(wds) != len(roots):
                    raise SystemExit(f"verify: {len(roots)} --programs-root, {len(rks)} --rank, {len(wds)} --workload-digest: give one per rank")
                if a.ranks > 1 and len(roots) > 1 and sorted(rks) != list(range(a.ranks)):
                    raise SystemExit(f"verify: TP world {a.ranks} needs one --programs-root per rank 0..{a.ranks - 1} (got ranks {rks})")
                parts = []
                for root, rk, wd in zip(roots, rks, wds):
                    if eng == "traversal":
                        from .required_values import ProgramView
                        progs = {rid: ProgramView.streaming(d) for rid, d in _program_dirs_for(fx, root).items()}
                    else:
                        progs = _programs_for(fx, root)
                    parts.append(build_global_manifest(fx, progs, workload_digest=wd, rank=rk, ranks=a.ranks, profile=a.profile, engine=eng))
                return parts[0] if len(parts) == 1 else merge_manifests(parts)
            raise SystemExit("verify: give --program [--workload] (B=1) or --fixture --programs-root [--workload-digest] (B>=2); repeat --program / --programs-root with --rank for a TP row")
        rebuilt = _rebuild(engine)
        rows_digest = manifest_digest(used.get("identities") or [])
        rec = {"schema": "verity-sweep/manifest-verify/v1", "manifest": a.manifest, "used_digest": used.get("manifest_digest"), "used_rows_digest": rows_digest,
               "rebuilt_digest": rebuilt["manifest_digest"], "used_program_digest": used.get("program_digest"), "rebuilt_program_digest": rebuilt["program_digest"],
               "used_identities": len(used.get("identities") or []), "rebuilt_identities": rebuilt["populations"]["identities"], "rebuilt_complete": rebuilt.get("complete"),
               "rebuilt_engine": engine}
        rec["ok"] = bool(rec["used_digest"] == rows_digest == rec["rebuilt_digest"] and rec["used_program_digest"] == rec["rebuilt_program_digest"] and rebuilt.get("complete") is True)
        if engine == "traversal":
            # [R16 reqval] the traversal engine verifies against BOTH implementations: the manifest of record must be the inventory's rebuild
            # too (the code of record until the traversal is integrated) -- a disagreement between the engines is named, never averaged
            inv = _rebuild("inventory")
            rec["cross_check"] = {"inventory_digest": inv["manifest_digest"], "traversal_digest": rebuilt["manifest_digest"], "agree": inv["manifest_digest"] == rebuilt["manifest_digest"],
                                  "inventory_identities": inv["populations"]["identities"], "inventory_complete": inv.get("complete"),
                                  "boundary_audit": {k: v for k, v in (rebuilt.get("boundary_audit") or {}).items() if k != "families"} if not rebuilt.get("global") else
                                                    {rid: {k: v for k, v in ba.items() if k != "families"} for rid, ba in (rebuilt.get("boundary_audit") or {}).items()}}
            rec["ok"] = rec["ok"] and rec["cross_check"]["agree"]
        if not rec["ok"]:
            why = []
            if engine == "traversal" and not rec["cross_check"]["agree"]:
                why.append(f"ENGINES DISAGREE: inventory {rec['cross_check']['inventory_digest'][:16]} ({rec['cross_check']['inventory_identities']} identities) != "
                           f"traversal {rec['rebuilt_digest'][:16]} ({rec['rebuilt_identities']}): a traversal bug or an inventory defect -- `required_values diff` names the identities")
            if rec["used_digest"] != rows_digest:
                why.append(f"manifest_digest {str(rec['used_digest'])[:16]} != digest of its rows {rows_digest[:16]} (edited after digesting)")
            if rows_digest != rec["rebuilt_digest"]:
                why.append(f"rows digest {rows_digest[:16]} != the Program x workload's manifest {rec['rebuilt_digest'][:16]} ({rec['used_identities']} vs {rec['rebuilt_identities']} identities): not the manifest of record")
            if rec["used_program_digest"] != rec["rebuilt_program_digest"]:
                u, b = str(rec["used_program_digest"]), str(rec["rebuilt_program_digest"])
                if u[:16] == b[:16] or len(u) != len(b):                          # [exec M-0545] a prefix passed as --workload-digest: say so, full width
                    why.append(f"program_digest {u} ({len(u)} hex) != the rebuilt {b} ({len(b)} hex): pass the FULL digest (workload_program.json), not a prefix")
                else:
                    why.append(f"program_digest {u[:16]} != the Build's {b[:16]}")
            if rebuilt.get("complete") is not True:
                why.append(f"the Program x workload manifest is incomplete: {rebuilt.get('unmodelled')}")
            rec["why"] = "; ".join(why)
        if a.out:
            json.dump(rec, open(a.out, "w"), indent=1)
        print(json.dumps(rec, indent=1))
        print(f"[manifest] VERIFY {'OK: the manifest of record' if rec['ok'] else 'FAIL: ' + rec['why']}")
        return 0 if rec["ok"] else 1
    if a.cmd == "build-global":
        fx = json.load(open(a.fixture))
        roots = list(a.programs_root)
        rks = list(a.rank) if a.rank else [0] * len(roots)
        wds = list(a.workload_digest) if a.workload_digest else [None] * len(roots)
        if len(rks) != len(roots) or len(wds) not in (0, len(roots)):
            raise SystemExit(f"build-global: {len(roots)} --programs-root, {len(rks)} --rank, {len(wds)} --workload-digest: give one per rank")
        if a.ranks > 1 and sorted(rks) != list(range(a.ranks)):
            raise SystemExit(f"build-global: TP world {a.ranks} needs one --programs-root per rank 0..{a.ranks - 1} (got ranks {rks})")
        def _rebuild_global(eng: str) -> dict[str, Any]:
            rank_parts = []
            for programs_root, rank_i, wd_i in zip(roots, rks, wds):
                if eng == "traversal":                                           # [R16 reqval] streamed: the same lookup, rows never all in memory
                    from .required_values import ProgramView
                    progs = {rid: ProgramView.streaming(d) for rid, d in _program_dirs_for(fx, programs_root).items()}
                else:
                    progs = _programs_for(fx, programs_root)                     # [MAN-05] shared with `verify` (same lookup, same fallback)
                rank_parts.append(build_global_manifest(fx, progs, workload_digest=wd_i, rank=rank_i, ranks=a.ranks, profile=a.profile, engine=eng))
            # [R15 tp TP-04] a TP row: the per-rank GLOBAL manifests merged with the rank rule (rows keep their rank's digest, ranks never folded)
            return rank_parts[0] if len(rank_parts) == 1 else merge_manifests(rank_parts)

        man = _rebuild_global(engine)
        rc = _cross_or_fail(man, _rebuild_global, a.out)
        if rc:
            return rc
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(man, open(a.out, "w"), separators=(",", ":"))
        print(json.dumps({k: v for k, v in man.items() if k != "identities"}, indent=1, default=str))
        print(f"[manifest] GLOBAL {man['populations']['identities']} identities, digest {man['manifest_digest']} -> {a.out}")
        return 0
    man = json.load(open(a.manifest))
    lay = json.load(gzip.open(a.layouts, "rt") if a.layouts.endswith(".gz") else open(a.layouts))
    res = coverage_check(man, lay, request_id=a.request_id, rank=a.rank)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1)[:4000])
    return 0 if res["ok"] else 1


def main() -> int:
    """`_main` with the one environment fault named: RegistryUnavailable -> its message on stderr, exit REGISTRY_UNAVAILABLE_EXIT (3),
    the PRECHECK_FAIL_* vocabulary row_pod.sh classifies (no manifest.json is written)."""
    try:
        return _main()
    except RegistryUnavailable as e:
        import sys
        print(str(e), file=sys.stderr, flush=True)
        return REGISTRY_UNAVAILABLE_EXIT


if __name__ == "__main__":
    raise SystemExit(main())
