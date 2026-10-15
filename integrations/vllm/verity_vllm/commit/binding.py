"""R14 BIND-01 — the collector binding map and the required-value coverage check (`coverage`; brief §4, TABLE-RULES r14-v1 Commit column).

Glossary (Draft 3 §12): `op_path` / `output_member` / `invocation` together are the occurrence key -- the serialization of a runtime
occurrence; the `member` field names a Value of the Call at that occurrence (the field name stays); the binding map is the record of
value acquisition (which committed bytes each required Value was taken from) and 'the manifest' is the required-value manifest.

Identity key (agreed with lane manifest, TABLE-RULES v1):
    (program_digest, request_id, step, invocation, op_path, output_member, element_range, rank)

* `op_path`        producing module path in the implementation (`model.layers.3.self_attn.qkv_proj`), the sampler `runner.sampler`, a
                   hidden family `fa2.m1/L3` / `fa3.m1/L3` / `moe/L3`.
* `output_member`  `"k"` for tuple slot `/k`, `"out"` for a single tensor output, `"in"` for an acquired input, `"sampled_token_ids"`,
                   the hidden/MoE plane name (`g4x64x128/...`, `router_w`, ...).
* `invocation`     0-based call index of `op_path` within the step (a shared RotaryEmbedding is invocation k in layer k).
* `element_range`  `[lo, hi)` in row-major elements of the Value; a map entry may cover the whole Value and satisfy any contained range.

The BINDING MAP is built from the committer's published per-step layouts (`_layouts`: name, dtype, shape, byte offset, leaf range) + the
run's step roots at `finalize()` -- i.e. it is fixed BEFORE the challenge exists (the challenge is derived from the run root AND the map
digest).  The EXPECTED POPULATION is independent of the collector: model structure (module set + a fixed per-class Value rule -- config,
not observation) x the workload (requests, engine steps, TP ranks).  Lane manifest's Program/query-derived required-value manifest has the
same key and replaces `expected_population` when delivered (`coverage_check(pop, bmap)` takes any iterable of identities).

Two separate verdicts (brief §4): `coverage_check` = required-value coverage (every required Value has >= 1 binding; duplicates/aliases/extra
bytes cannot hide a missing one) and `value_correspondence` (the opened bytes under an identity are the producer's bytes -- the late-read / relabel negatives;
openings that verify against the root do NOT imply correspondence).
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Iterable, Iterator

from verity_vllm.commit import challenge as CH
from verity_vllm.query.manifest.format import MOE_PLANES
from verity_vllm.query.manifest.format import PROMOTED_DTYPE_BYTES  # noqa: F401  -- re-exported (R19 int-20)
from verity_vllm.query.manifest.format import PROMOTED_OBSERVATION_RULE as PROMOTED_INPUT_RULE  # noqa: F401  -- re-exported
from verity_vllm.query.manifest.format import promoted_observation as promoted_input_acquisition  # noqa: F401  -- re-exported

IDENTITY_KEY = ("program_digest", "request_id", "step", "invocation", "op_path", "output_member", "element_range", "rank")
RULES_VERSION = "r14-v1"
MAP_SCHEMA = "binding-map/1"
HIDDEN_FAMILIES = ("fa2.m1/", "fa3.m1/", "moe/")
MOE_FAMILY = "moe/"                 # [R15 moe] block-stream planes: per-token (pair-major) members, split by request rows at B>1 (not per-launch)

# Fixed per-module-class output-member rule (pre-execution structural knowledge of the vLLM Llama-family forward, NOT observation).
# A module class -> the members its forward materialises.  RMSNorm: vLLM's `RMSNorm.forward(x, residual)` returns a (out, residual) tuple
# whenever a residual is passed -- every norm of a Llama-family model except the very first `layers.0.input_layernorm` (residual None).
MEMBER_RULES: dict[str, tuple[str, ...]] = {
    "Linear": ("0",), "RMSNorm": ("0", "1"), "GemmaRMSNorm": ("0", "1"), "LayerNorm": ("out",),
    "SiluAndMul": ("out",), "GeluAndMul": ("out",), "GeluAndMulSparse": ("out",), "MulAndSilu": ("out",),
    "RotaryEmbedding": ("0", "1"), "Attention": ("out",), "VocabParallelEmbedding": ("in", "out"), "LogitsProcessor": ("out",),
    "Sampler": ("sampled_token_ids",),
}
FIRST_NORM_SUFFIX = ".layers.0.input_layernorm"

#: [R19 int-20, F-r19-int-20] a promoted member's observation point (`PROMOTED_INPUT_RULE`, `promoted_input_acquisition`): the manifest
#: module owns the rule (manifest_format.PROMOTED_OBSERVATION_RULE / promoted_observation, shared by the collector's arming, this map and
#: the layout-level manifest_format.coverage_check); imported above under the binding's names for its callers (bench.commit_delta, tests).


# ---------------------------------------------------------------------- names -> identity parts

def split_name(name: str) -> tuple[str, str]:
    """Committed tensor name -> (op_path, output_member)."""
    for fam in HIDDEN_FAMILIES:
        if name.startswith(fam):
            parts = name.split("/")
            return "/".join(parts[:2]), "/".join(parts[2:]) or "out"       # fa2.m1/L3/g4x64x128/... -> ("fa2.m1/L3", "g4x64x128/...")
    if "/" in name:
        base, member = name.rsplit("/", 1)                                 # `<module>/<fx node>/<member>`: a call boundary's op_path has a `/`
        return base, member
    return name, "out"


def logical_request(rid: str, order: dict[str, str]) -> str:
    """Engine row id -> workload `r<i>` (`order`: workload request id -> `r<i>`).  vLLM v1 suffixes the external request id with
    `-<8 hex>` internally; match exact, then with the suffix stripped, then by prefix.  Unmapped ids stay raw (-> coverage FAILS for that
    request, fail-closed)."""
    rid = str(rid)
    if rid in order:
        return order[rid]
    base = rid.rsplit("-", 1)[0]
    if base in order:
        return order[base]
    hits = [v for k, v in order.items() if rid.startswith(k)]
    return hits[0] if len(hits) == 1 else rid


def identity(program_digest: str, request_id: str, step: int, invocation: int, op_path: str, member: str, lo: int, hi: int, rank: int = 0) -> tuple:
    return (program_digest, request_id, int(step), int(invocation), op_path, member, (int(lo), int(hi)), int(rank))


def identity_dict(ident: tuple) -> dict[str, Any]:
    d = dict(zip(IDENTITY_KEY, ident, strict=True))
    d["element_range"] = list(d["element_range"])
    return d


def _numel(shape: Iterable[int]) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n


# ---------------------------------------------------------------------- the binding map

def build_binding_map(layouts: dict[int, list], steps: list, run_root: bytes, program_digest: str, *, request_rows: dict[int, list[tuple[str, int, int]]] | None = None,
                      request_id: str = "r0", rank: int = 0, layout_version: str = "chunk-leaf-v1", chunk: int = 256,
                      weight_registration: dict | None = None, collector: dict | None = None,
                      name_bindings: dict[str, list[str] | tuple[str, str]] | None = None) -> dict[str, Any]:
    """identity -> commitment location, for every committed tensor of every step, from the published layouts + the step roots.

    `request_rows[step]` = [(request_id, row_lo, row_hi)] splits the leading (token/row) dimension of a batched member between the
    requests of that step (the scheduler's row order, recorded by the engine wrapper); without it the whole member binds `request_id`.
    Entries are compact (one per tensor x request); `digest` is over the canonical JSON of the entries -- recorded in the Commit record and
    mixed into the challenge, so the map cannot be chosen after the openings.
    `name_bindings` [R19 int-20]: committed name -> (op_path, member) of the identity it binds when that is not the name's own (a promoted
    member observed at its consumer's input: `<consumer>/in` binds (producer, `out`); `promoted_input_acquisition`).  The entry keeps the
    committed name and says so (`observation`)."""
    entries: list[dict[str, Any]] = []
    rebind = {str(k): (str(v[0]), str(v[1])) for k, v in (name_bindings or {}).items()}
    for step in sorted(layouts):
        metas = layouts[step]
        sc = steps[step] if step < len(steps) else None
        step_root = sc.root.hex() if sc is not None else None
        inv: Counter = Counter()
        for m in metas:
            op, member = split_name(m.name)
            bound_via = rebind.get(m.name)
            if bound_via is not None:
                op, member = bound_via
            k = inv[m.name]
            inv[m.name] += 1
            numel = _numel(m.shape)
            off = m.stream_off if m.stream_off is not None and m.stream_off >= 0 else m.first_leaf * m.chunk
            rr = (request_rows or {}).get(step)
            splits: list[tuple[str, int, int, int]] = []            # (request, request-local lo, request-local hi, global element offset)
            if rr and m.shape and numel > 0 and not op.startswith(HIDDEN_FAMILIES):
                rows = int(m.shape[0])
                per_row = numel // max(rows, 1)
                tokens = max(hi for _r, _lo, hi in rr)
                if rows == tokens:                                                    # per-token member: the request's token rows
                    splits = [(rid, 0, (hi - lo) * per_row, lo * per_row) for rid, lo, hi in rr]
                elif rows == len(rr):                                                 # per-request member (logits, sampled ids): the request's row
                    splits = [(rid, 0, per_row, i * per_row) for i, (rid, _lo, _hi) in enumerate(rr)]
                else:                                                                 # shared by every request of the step (whole member each)
                    splits = [(rid, 0, numel, 0) for rid, _lo, _hi in rr]
            elif rr and m.shape and numel > 0 and op.startswith(MOE_FAMILY):
                # [R15 moe / MOE-B2] the fused-MoE block-stream planes are PER-TOKEN members in pair-major order (MOE-03 source; the Program
                # addresses pair p = t*TOPK + k): router_w/router_ids [M, TOPK], up/down [M, TOPK, N], sum [M, H] have rows == tokens, act
                # [M*TOPK, I] has rows == tokens*TOPK.  Request r's rows are ITS token rows (x TOPK for act) -- the required set is
                # routing-independent (M_r*TOPK pairs whatever the experts chose) and its position comes from the step's request->row map,
                # never from the routing.  Binding the whole batched plane to every request (the hidden-launch rule below) pointed r1's
                # request-local range at r0's rows and let a range beyond r1's own rows pass by containment.
                rows = int(m.shape[0])
                tokens = max(hi for _r, _lo, hi in rr)
                if tokens > 0 and rows >= tokens and rows % tokens == 0:
                    f = rows // tokens                                                # 1 for token planes, TOPK for the act plane
                    per_row = numel // rows
                    splits = [(rid, 0, (hi - lo) * f * per_row, lo * f * per_row) for rid, lo, hi in rr]
                else:                                                                 # not a token multiple: bind whole, reported as an extra shape
                    splits = [(rid, 0, numel, 0) for rid, _lo, _hi in rr]
            elif rr and op.startswith(HIDDEN_FAMILIES):
                splits = [(rid, 0, numel, 0) for rid, _lo, _hi in rr]
            if not splits:
                splits = [(request_id, 0, numel, 0)]
            for rid, lo, hi, goff in splits:
                entries.append({
                    "identity": identity_dict(identity(program_digest, rid, step, k, op, member, lo, hi, rank)),
                    "kind": "leaf", "step_root": step_root, "layout": layout_version, "chunk": int(m.chunk),
                    "leaf_lo": int(off // chunk), "leaf_hi": int((off + max(m.nbytes, 1) - 1) // chunk + 1),
                    "stream_offset": int(off), "width_bytes": int(m.nbytes), "encoding": {"dtype": m.dtype, "shape": list(m.shape)}, "element_offset": int(goff),
                    "tensor_ordinal": int(m.ordinal), "name": m.name,
                    **({"observation": "consumer-input"} if bound_via is not None else {}),
                })
    if weight_registration:
        entries.append({"identity": identity_dict(identity(program_digest, "*", -1, 0, "weights", "registered", 0, 0, rank)), "kind": "registered-input",
                        "digest": weight_registration.get("root") or weight_registration.get("digest") or json.dumps(weight_registration, sort_keys=True, default=str)[:128]})
    canon = json.dumps(entries, sort_keys=True, separators=(",", ":"), default=str).encode()
    return {"schema": MAP_SCHEMA, "rules_version": RULES_VERSION, "program_digest": program_digest, "run_root": run_root.hex(), "rank": rank,
            "layout": layout_version, "collector": collector or {}, "entries_n": len(entries), "digest": hashlib.sha256(canon).hexdigest(), "entries": entries}


def member_matches(required: str, got: str) -> bool:
    """`*` matches any member; otherwise equal, or one is the other's prefix at an `x` / `/` boundary (hidden geometry names)."""
    if required == "*" or required == got:
        return True
    a, b = (required, got) if len(required) < len(got) else (got, required)
    return b.startswith(a) and b[len(a)] in "x/"


def map_index(bmap: dict[str, Any]) -> dict[tuple, list[dict[str, Any]]]:
    """(request, step, invocation, op_path, member, rank) -> entries (several = several physical copies / ranges)."""
    idx: dict[tuple, list[dict[str, Any]]] = {}
    for e in bmap["entries"]:
        i = e["identity"]
        idx.setdefault((i["request_id"], i["step"], i["invocation"], i["op_path"], i["output_member"], i["rank"]), []).append(e)
    return idx


def _op_key_index(idx: dict[tuple, Any]) -> dict[tuple, list[tuple]]:
    """(request, step, invocation, op_path, rank) -> the map keys of every member of that op, in map order.  [R18 tap, F-r18-tap-2] the
    member-`*` identities (one per request x step x attention layer for the hidden family) used to scan the WHOLE index each: O(|*| x |map|)
    = 132,096 x ~890k at B64 llama32-1b 1024/128 -- the serving thread sat hours at 100% CPU in coverage_check AFTER the run (attempt 2/3 of
    the #23 device diagnostic, cp-r18-l40u 22:2xZ; B8: 16,512 x 111k, a few minutes, unnoticed).  Same keys, same order, built once."""
    out: dict[tuple, list[tuple]] = {}
    for k in idx:
        out.setdefault((k[0], k[1], k[2], k[3], k[5]), []).append(k)
    return out


def map_context_check(bmap: dict[str, Any], run) -> tuple[bool, str]:
    """The map's roots/program must be the run's (wrong root / layout -> context failure)."""
    if bmap["run_root"] != run.run_root.hex():
        return False, f"binding map run_root {bmap['run_root'][:16]} != run {run.run_root.hex()[:16]}"
    for e in bmap["entries"]:
        st = e["identity"]["step"]
        if e["kind"] != "leaf":
            continue
        if st >= len(run.steps) or e["step_root"] != run.steps[st].root.hex():
            return False, f"entry {e['name']} step {st}: step_root does not match the run's step root"
        if e["layout"] != bmap["layout"]:
            return False, f"entry {e['name']} step {st}: layout {e['layout']} != map layout {bmap['layout']}"
    return True, "PASS"


# ---------------------------------------------------------------------- the expected population (collector-independent)

def structure_of(committer, *, stochastic: bool = False, derived_splits: bool = False) -> dict[str, Any]:
    """Model structure from the INSTALLED module set (the model's config/architecture; the committer's hooks are on exactly these modules
    but the member rule comes from MEMBER_RULES, not from what was emitted).  Serialisable, so a fixture can be stored and diffed.
    `stochastic` (the workload's DECLARED sampling policy, temperature > 0 with per-request seeds): the sampler additionally materialises
    the consumed per-sequence seed (`runner.sampler/seed`, the stochastic Program's registered runtime input; manifest family `sampler_seed`).
    `derived_splits` [R16 stoch REQ-03] (the workload DECLARES `sampling.splits == "derived"`): the sampler also materialises the top-p
    split count its kernels ran with (`runner.sampler/splits`, the derived-S Program's `splits` input; manifest family `sampler_splits`)."""
    mods = []
    rope_calls = 0
    attn_layers: set[int] = set()
    for name, m in committer._modules:
        cls = committer._module_class(name)
        if cls == "RotaryEmbedding":
            rope_calls += 1                               # one shared module, called once per attention layer
        if cls == "Attention" and ".layers." in name:
            attn_layers.add(int(name.split(".layers.")[1].split(".")[0]))
        mods.append({"op_path": name, "class": cls})
    # the shared RotaryEmbedding appears ONCE in named_modules but is invoked per layer: invocations = attention layers
    for d in mods:
        if d["class"] == "RotaryEmbedding":
            d["invocations"] = max(len(attn_layers), 1)
        else:
            d["invocations"] = 1
        members = list(MEMBER_RULES.get(d["class"], ("out",)))
        if d["class"] == "Linear":
            mod = dict(committer._modules).get(d["op_path"])
            if mod is not None and getattr(mod, "bias", None) is not None and getattr(mod, "skip_bias_add", False):
                members.append("1")
        if d["class"] in ("RMSNorm", "GemmaRMSNorm") and d["op_path"].endswith(FIRST_NORM_SUFFIX):
            members = ["out"]
        # [R14 moe] a norm INSIDE the attention module (OLMoE / Qwen3 `self_attn.q_norm` / `k_norm`) is called without a residual and
        # returns ONE tensor: expecting ("0", "1") over-required 2048 identities on OLMoE b1 256/32 (16 layers x 2 norms x 2 members x 32
        # steps) while the collector bound `out` (coverage FAIL, first `model.layers.0.self_attn.q_norm/0`).  Residual-stream norms
        # (input_layernorm after layer 0, post_attention_layernorm, model.norm) keep the fused (hidden, residual) pair.  The member
        # rule is a stand-in for the Program's output arity (lane manifest's Program-derived manifest replaces this table).
        if d["class"] in ("RMSNorm", "GemmaRMSNorm") and ".self_attn." in d["op_path"]:
            members = ["out"]
        d["members"] = members
    # the sampled ids are acquired from the wrapped `runner.sample`, not from a module hook: required once per step whenever the runner
    # samples (omitting the sampler must be a coverage failure, not an 'extra')
    runner = getattr(committer, "runner", None)
    if runner is not None and hasattr(runner, "sample") and not any(d["op_path"] == "runner.sampler" for d in mods):
        mods.append({"op_path": "runner.sampler", "class": "Sampler", "invocations": 1,
                     "members": ["sampled_token_ids"] + (["seed"] if stochastic else [])       # [R15 stoch] the consumed seed, stochastic rows only
                     + (["splits"] if (stochastic and derived_splits) else [])})               # [R16 stoch REQ-03] the kernels' split count, derived-S rows
    hidden = None
    for s in committer.sources:
        sn = getattr(s, "short_name", "") or type(s).__name__
        if "FA3" in type(s).__name__ or sn == "hidden_m1_fa3":
            hidden = "fa3.m1"
        elif "HiddenM1" in type(s).__name__ or sn == "hidden_m1":
            hidden = "fa2.m1"
    moe_src = next((s for s in committer.sources if getattr(s, "short_name", "") == "moe"), None)
    moe_layers = sorted(set(moe_src.layers.values())) if moe_src is not None else []
    return {"modules": mods, "attention_layers": sorted(attn_layers), "hidden": hidden, "moe_layers": moe_layers,
            "moe_planes": list(getattr(moe_src, "planes", MOE_PLANES)) if moe_layers else [],
            "weights_registered": bool(getattr(committer, "weights", None))}


def expected_population(structure: dict[str, Any], workload: dict[str, Any]) -> Iterator[tuple]:
    """Every required identity: structure x requests x steps x ranks.  `workload` = {program_digest, requests: [ids], steps: n (engine
    steps, prefill = 0), ranks: [..], request_steps: {rid: [steps]} (optional: a request present only in some steps)}.
    element_range is the whole member (`(0, -1)` = 'whole'): the population is compact; the checker matches by containment."""
    pd = workload["program_digest"]
    ranks = list(workload.get("ranks") or [0])
    steps = list(range(int(workload["steps"])))
    if structure.get("weights_registered"):
        for rank in ranks:
            yield identity(pd, "*", -1, 0, "weights", "registered", 0, 0, rank)      # the registered-input binding of the weights
    for rid in workload["requests"]:
        rsteps = (workload.get("request_steps") or {}).get(rid, steps)
        for step in rsteps:
            for rank in ranks:
                for d in structure["modules"]:
                    for inv in range(int(d.get("invocations", 1))):
                        for mem in d["members"]:
                            yield identity(pd, rid, step, inv, d["op_path"], mem, 0, -1, rank)
                if structure.get("hidden"):
                    for L in structure["attention_layers"]:
                        yield identity(pd, rid, step, 0, f"{structure['hidden']}/L{L}", "*", 0, -1, rank)
                for L in structure.get("moe_layers") or []:
                    for plane in structure.get("moe_planes") or []:
                        yield identity(pd, rid, step, 0, f"moe/L{L}", plane, 0, -1, rank)


def load_required_manifest(path: str, program_digest: str, *, requests: list[str] | None = None, steps: int | None = None, ranks: list[int] | None = None,
                           arrivals: dict[str, int] | None = None, request_steps: dict[str, Any] | None = None) -> tuple[list[tuple], dict[str, Any]]:
    """Lane manifest's `manifest.json` (Program x canonical VU query; TABLE-RULES key + family label per row) -> identity tuples.
    Row fields: request_id / step / invocation / rank may be an int|str, `"*"` (every one of the workload's), or a `[lo, hi]` range;
    `element_range` `[lo, hi)` (absent = whole member).  Registered weights rows (`op_path` `weights`, step `*`) map onto the single
    registered-input identity.  Returns (identities, header) where header carries the manifest's own digest for the record.
    [R17-1 / F-moe-r17-03] `request_steps` = {r<i>: engine steps the request EXECUTED (its executed prefix of record: arrival .. arrival +
    executed_prefix_len)}: an identity of a listed request at an engine step outside its prefix is programmed-but-unexecuted -- NOT required
    (counted in header `programmed_not_executed_n`).  The manifest itself stays the Program's (cap shape); only the requirement is placed."""
    doc = json.loads(open(path).read())
    rows = doc.get("entries") or doc.get("rows") or doc.get("identities") or []
    hdr = {"path": path, "manifest_digest": doc.get("manifest_digest") or doc.get("digest"), "program_digest": doc.get("program_digest"),
           "complete": doc.get("complete", True), "unmodelled": doc.get("unmodelled") or {},   # [MAN-02] incomplete manifest -> fail closed (commit_delta)
           "rows_n": len(rows), "version": doc.get("version") or doc.get("schema"), "query": doc.get("query") or doc.get("vu_query")}
    # [R15 manbind, rev-cb M-0108] the manifest is an INPUT the consumer must not trust blindly: (F-cb-07) its `manifest_digest` is recomputed
    # over the rows it carries (a manifest edited after digesting is refused); (F-cb-10/09) a zero-width identity is refused, never
    # "vacuously covered"; (F-cb-13) a request id that is not `r<k>` / `*` (nor resolvable through requests_declared) is refused.
    from verity_vllm.query.manifest import format as MF

    if doc.get("schema") != MF.SCHEMA:
        raise ValueError(f"required manifest {path}: schema {doc.get('schema')!r} is not {MF.SCHEMA!r}: refused")
    if not isinstance(doc.get("program_digest"), str) or not doc.get("program_digest"):
        raise ValueError(f"required manifest {path}: program_digest must be one string (got {type(doc.get('program_digest')).__name__}): refused")
    recomputed = MF.manifest_digest(rows)
    if recomputed != doc.get("manifest_digest"):
        raise ValueError(f"required manifest {path}: manifest_digest {str(doc.get('manifest_digest'))[:16]} != digest of its rows {recomputed[:16]}: edited after digesting, refused")
    hdr["manifest_digest_verified"] = True
    for r in rows:
        ident = r.get("identity") or r
        if r.get("family") == "weights" or ident.get("op_path") == "weights":
            continue
        er = ident.get("element_range")
        if er is None or len(er) != 2 or er[0] is None or (er[1] is not None and er[1] != -1 and int(er[1]) <= int(er[0])):
            raise ValueError(f"required manifest {path}: zero-width / malformed identity element_range {er} at step {ident.get('step')} "
                             f"{ident.get('op_path')}/{ident.get('output_member')} req {ident.get('request_id')}: refused (MAN-02)")
    df = ((doc.get("program") or {}).get("dataflow") or {}) if isinstance(doc.get("program"), dict) else {}
    if df:                                                                    # [R15 compiled, rev-pm M-0589] P_compiled: the dataflow is in the digest
        hdr["dataflow_in_digest"] = True
        hdr["dataflow"] = {k: df.get(k) for k in ("n_edges", "n_launch_edges", "n_undefined", "wiring_declared", "split_graph")}
    all_req = list(requests or [])
    all_steps = list(range(int(steps))) if steps is not None else []
    all_ranks = list(ranks or [0])
    # global (B>1) manifests name requests by the workload's own ids; `requests_declared[i].index` (else declaration order) -> r<i>
    declared = {str(d.get("request_id")): f"r{int(d.get('index', i))}" for i, d in enumerate(doc.get("requests_declared") or []) if d.get("request_id") is not None}
    hdr["requests_declared_n"] = len(declared)
    hdr["global"] = bool(doc.get("global", False))
    # [R15 manbind B8] the declared per-engine-step composition (which requests, how many rows each) and the members the map's partition is
    # read from -- `composition_check` holds the map to it; request ids normalised to r<i>
    # A B=1 `build` manifest writes engine_steps as the request's engine-step NUMBERS (ints: arrival + s, rc3 schema); only `build-global`
    # declares the dict composition.  [rel M-0245 / r15-int-2 canary #2] the numbers are kept as `engine_step_numbers`, the composition is
    # declared only from dict entries (composition_check then skips, named) -- never subscript an int.
    raw_es = list(doc.get("engine_steps") or [])
    if raw_es and all(isinstance(es, dict) for es in raw_es):
        hdr["engine_steps"] = [{"engine_step": int(es["engine_step"]), "rows": es.get("rows"), "n_seq": es.get("n_seq"),
                                "members": [dict(m, request_id=declared.get(str(m["request_id"]), str(m["request_id"]))) for m in es["members"]]} for es in raw_es]
        hdr["engine_step_numbers"] = [int(es["engine_step"]) for es in raw_es]
    elif raw_es and all(isinstance(es, (int, float)) and not isinstance(es, bool) for es in raw_es):
        hdr["engine_steps"] = []
        hdr["engine_step_numbers"] = [int(es) for es in raw_es]
    elif raw_es:
        raise ValueError(f"required manifest {path}: engine_steps mixes composition dicts and step numbers ({sorted({type(es).__name__ for es in raw_es})}): refused")
    else:
        hdr["engine_steps"] = []
        hdr["engine_step_numbers"] = []
    hdr["composition_reference"] = doc.get("composition_reference") or {}
    hdr["addressing"] = doc.get("addressing", "global" if hdr["global"] else "request-local")
    hdr["program_digest_kind"] = doc.get("program_digest_kind")

    def expand(v, universe, default):
        if v is None:
            return [default]
        if v == "*":
            return universe or [default]
        if isinstance(v, list) and len(v) == 2 and all(isinstance(x, int) for x in v):
            return list(range(int(v[0]), int(v[1]) + 1)) if v[0] <= v[1] else []
        return [v]

    out: list[tuple] = []
    executed = None if request_steps is None else {str(k): {int(s) for s in v} for k, v in request_steps.items()}
    hdr["request_steps_of_record"] = None if executed is None else {k: [min(v), max(v)] if v else [] for k, v in sorted(executed.items())}
    hdr["programmed_not_executed_n"] = 0
    # [COORD R17-26/27; rel M-0410 (a)] the classification is ON THE RECORD by family / request / kind so the population's third bucket
    # (sampled_replay.population.not_executed_by_rule) reconciles against it identity-count by identity-count (executed_prefix.reconcile_not_executed);
    # the manifest of record is not edited: rows_n / manifest_digest above are the file's
    from verity_vllm.check import executed_prefix as _EPX

    hdr["not_executed_by_rule"] = None if executed is None else _EPX.new_classification()
    for r in rows:
        ident = r.get("identity") or r
        op = ident.get("op_path")
        if op == "weights" or (r.get("family") == "weights"):
            for rk in expand(ident.get("rank"), all_ranks, 0):
                out.append(identity(program_digest, "*", -1, 0, "weights", "registered", 0, 0, int(rk)))
            continue
        er = list(ident.get("element_range") or [0, -1])
        if er[1] is None:
            er[1] = -1
        req_field = ident.get("request_id")
        member = str(ident.get("output_member", "out"))
        if req_field == "*" and isinstance(r.get("requests"), list) and r["requests"]:
            req_field = list(r["requests"])                                # batch-level member: the requests it is bound to, explicitly
            if op.startswith(HIDDEN_FAMILIES):
                # a batch-level hidden row names the padded whole-step launch geometry; the tap commits per-sequence planes
                # (`s<i>/g<HB>x<M>x<NB>/<plane>`), so the requirement is op-level per request: >= 1 plane of that op/step bound to it
                member = "*"
                hdr["hidden_batch_rows_op_level"] = hdr.get("hidden_batch_rows_op_level", 0) + 1
        engine_step = ident.get("engine_step", r.get("engine_step"))       # absolute engine step (batch-level rows) beats request-relative `step` ...
        # [R16 compiled, H100 B8] ... unless the caller passes the request schedule of record (`arrivals`) and the row carries a request-
        # relative `step`: then engine step = step + arrival (the compiled manifest's declared step_basis; its `engine_step` annotation is the
        # step the row was DERIVED for under the declared arrive_steps, which drift from the committed-step axis on a workload with gaps)
        if arrivals is not None and ident.get("step") is not None and ident.get("request_id") not in (None, "*"):
            if engine_step is not None:
                hdr["engine_step_rebased_rows"] = hdr.get("engine_step_rebased_rows", 0) + 1
            engine_step = None
        for rid in (req_field if isinstance(req_field, list) and not (len(req_field) == 2 and all(isinstance(x, int) for x in req_field)) else expand(req_field, all_req, "r0")):
            rid = declared.get(str(rid), f"r{rid}" if str(rid).isdigit() else str(rid))   # workload id | index -> r<i>
            if not MF.REQ_RE.match(rid):                                   # (F-cb-13) a request the workload does not declare (e.g. a bare UUID)
                raise ValueError(f"required manifest {path}: request id {rid!r} is not r<k> and not declared in requests_declared: refused (COORD R15 ruling 2)")
            arrive = int((arrivals or {}).get(rid, 0))                     # gprog reading: manifest step is request-relative; map step = engine step
            for st in ([int(engine_step)] if engine_step is not None else expand(ident.get("step"), all_steps, 0)):
                st = int(st) if engine_step is not None else int(st) + arrive
                if executed is not None and rid in executed and st not in executed[rid]:
                    hdr["programmed_not_executed_n"] += 1                    # [R17-1] past the request's executed prefix of record: the named class
                    _EPX._classify(hdr["not_executed_by_rule"], dict(ident, family=r.get("family"), requests=r.get("requests"),
                                                                        request_id=ident.get("request_id")), rid)
                    continue
                for inv in expand(ident.get("invocation"), [0], 0):
                    for rk in expand(ident.get("rank"), all_ranks, 0):
                        out.append(identity(program_digest, str(rid), int(st), int(inv), op, member, int(er[0]), int(er[1]), int(rk)))
    return out, hdr


def population_digest(pop: Iterable[tuple]) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    for ident in pop:
        h.update(json.dumps(identity_dict(ident), sort_keys=True, separators=(",", ":")).encode())
        h.update(b"\n")
        n += 1
    return h.hexdigest(), n


# ---------------------------------------------------------------------- coverage

def _entry_where(e: dict[str, Any]) -> str:
    """Where a leaf entry's bytes live, for the ambiguity text: name @ stream_offset (leaf range)."""
    return f"{e.get('name')}@{e.get('stream_offset')}[{e.get('leaf_lo')},{e.get('leaf_hi')})"


def coverage_check(population: Iterable[tuple], bmap: dict[str, Any], *, max_report: int = 12) -> dict[str, Any]:
    """Every required identity has EXACTLY ONE map entry with the same (request, step, invocation, op_path, member, rank) whose element_range
    contains the required one (member `*` = any member of that op_path/invocation).  Duplicated copies of ANOTHER identity do not count;
    the map's request/step/rank must match exactly, so a missing request, step or TP rank is a missing identity.  Extra entries are
    reported, allowed.  [R16 rev F-r16-08 / M-0077; BRIEF §3 C1 'one UNAMBIGUOUS binding'] a required identity bound by MORE THAN ONE leaf
    entry (two names / two stream ranges under one identity key, both covering the required range) is AMBIGUOUS: the replay would read
    one copy and the other would carry an unchecked value under a required identity -- `ok` requires `ambiguous_n == 0`; the first
    `max_report` are named with where each copy lives.  `physical_copies_n` (every duplicate in the map, required or not) stays reported."""
    idx = map_index(bmap)
    by_op: dict[tuple, int] = Counter()
    for (rid, st, inv, op, mem, rk), es in idx.items():
        by_op[(rid, st, inv, op, rk)] += len(es)
    op_keys = _op_key_index(idx)
    missing: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    expected_n = bound_n = ambiguous_n = 0
    address_mismatch = 0                 # the (request, step, op, member) IS bound, but not at the required element_range: an address disagreement
    used: set[tuple] = set()
    for ident in population:
        expected_n += 1
        pd, rid, st, inv, op, mem, (lo, hi), rk = ident
        es = []
        covering: list[dict[str, Any]] = []
        if mem == "*":
            ok = by_op.get((rid, st, inv, op, rk), 0) > 0
            if ok:
                keys = op_keys.get((rid, st, inv, op, rk), [])
                used.update(keys)
                # any member of that op bound twice = two bindings of one value: ambiguous
                covering = next((idx[k] for k in keys if len(idx[k]) > 1), [])
        else:
            es = idx.get((rid, st, inv, op, mem, rk), [])
            key_used = (rid, st, inv, op, mem, rk)
            if not es and op.startswith(HIDDEN_FAMILIES):
                # hidden planes: the manifest names the launch geometry `g<HB>x<M>x<NB>[x<BN>x<D>]`, the tap row carries its prefix
                for kk, ee in idx.items():
                    if kk[0] == rid and kk[1] == st and kk[2] == inv and kk[3] == op and kk[5] == rk and member_matches(mem, kk[4]):
                        es, key_used = ee, kk
                        break
            ok = False
            for e in es:
                elo, ehi = e["identity"]["element_range"]
                off = int(e.get("element_offset", 0) or 0)                 # B>1: the entry's request-local range sits at `element_offset`
                if hi == -1 or (elo <= lo and ehi >= hi) or (elo + off <= lo and ehi + off >= hi):   # of the batched member (manifest: global)
                    ok = True
                    covering.append(e)                                      # keep counting: >1 covering entry = ambiguous (F-r16-08)
            if ok:
                used.add(key_used)
        if ok:
            bound_n += 1
            if len(covering) > 1:
                ambiguous_n += 1
                if len(ambiguous) < max_report:
                    d = identity_dict(ident)
                    d["bindings_n"] = len(covering)
                    d["bound_at"] = [_entry_where(e) for e in covering[:4]]
                    ambiguous.append(d)
        else:
            if es:
                address_mismatch += 1
            if len(missing) < max_report:
                d = identity_dict(ident)
                if es:
                    d["bound_at"] = [[e["identity"]["element_range"], int(e.get("element_offset", 0) or 0)] for e in es[:2]]
                missing.append(d)
    extra = [k for k in idx if k not in used]
    copies = sum(len(es) - 1 for es in idx.values() if len(es) > 1)
    return {"ok": expected_n > 0 and bound_n == expected_n and ambiguous_n == 0, "rules_version": RULES_VERSION, "expected_n": expected_n, "bound_n": bound_n,
            "missing_n": expected_n - bound_n, "address_mismatch_n": address_mismatch, "missing": missing, "extra_n": len(extra),
            "extra": [list(k) for k in extra[:max_report]], "physical_copies_n": copies,
            "ambiguous_n": ambiguous_n, "unambiguous": ambiguous_n == 0, "ambiguous": ambiguous,       # [F-r16-08] C1 'one unambiguous binding'
            "map_digest": bmap["digest"], "map_entries_n": bmap["entries_n"]}


def require_build_program_digest(required_manifest: str | None, program_digest: str | None) -> None:
    """[COORD M-0205 (C1) / rev-cb F-cb-06] a manifest accepted on its own program_digest is a wrong-Program hole: the digest of record comes
    from the BUILD (row_pod reads build_summary.json / workload_program.json), never from the manifest.  `--required-manifest` without
    `--program-digest` is refused by name before any engine is built (SystemExit -> the row's Commit is FAIL rc!=0, the reason in the log)."""
    if required_manifest and not (isinstance(program_digest, str) and program_digest.strip()):
        raise SystemExit("[delta] REFUSED: --required-manifest requires --program-digest <the Build's Program digest of record> "
                         "(B=1: build_request program_digest; B>=2: the workload digest) -- a manifest cannot vouch for its own Program (COORD M-0205 C1)")


def composition_check(man_hdr: dict[str, Any], bmap: dict[str, Any], *, max_report: int = 8, request_steps: dict[str, Any] | None = None) -> dict[str, Any]:
    """[R15 manbind B8] B>1: the map's per-engine-step request->rows PARTITION of the packed reference members must be exactly the manifest's
    declared composition (`engine_steps[e].members`: request id x rows, from Program x workload -- never from the collector).  Read from the
    per-token ids member (`composition_reference.input_token_ids`) and the per-sequence sampled-ids member: (1) the multiset {(request, rows)}
    equals the declared one -- an omitted request is `missing`, a relabelled one `missing` + `unexpected`/duplicated; (2) the blocks
    (`element_offset`, width) are disjoint and tile the member -- a copy of another request's block relabelled into an absent request's
    slot `overlaps`; (3) every sampled-ids slot is bound once per declared request.  Request-local identity ranges are otherwise blind to
    WHICH block a request was bound to, so this check is what makes relabel / duplicate-into-slot fail named.  B=1 manifests declare no
    composition: skipped, recorded.
    [R17-1 / F-moe-r17-03] `request_steps` = {r<i>: executed engine steps}: the declared composition (the Program's cap shape) is read
    over each request's executed prefix of record -- a request that stopped (EOS + LAG rows) is dropped from the later steps' expectation,
    so a row of it committed there is `unexpected` (execution past the stop) and its absence is not `missing`; a step no request executes
    is skipped (recorded as `steps_not_executed`).
    [B9.1 / revb F-r17b-66] a PADDED map (`activity: bot-by-enablement` entries, `padded-binding-map/1`): the declared composition is the
    Program's CAP shape and is read UNRESTRICTED; the EXECUTED packed member is tiled by the ACTIVE members only, in the record's row
    order, cursor == its live rows (the record composition unchanged); a bot-by-enablement entry is its OWN row space -- a tensor of its
    own, exactly one per (absent request, declared rows) per step, `element_offset` never read into the tiling -- and the UNION multiset
    {(request, rows)} over executed + padding must equal the declared cap composition; a request with BOTH an executed block and a
    padding entry at one step FAILS (double binding); a padding entry for a request the executed prefix of record places AT that step
    FAILS (padding inside the prefix, moe 7eff2a558); sampled-ids slots are checked over the executed member, the padding counted beside."""
    steps = man_hdr.get("engine_steps") or []
    if not steps or not isinstance(steps[0], dict):
        # a B=1 (`build`) manifest's engine_steps are the request's engine-step NUMBERS (rc3 schema, ints); only `build-global` declares
        # the dict composition {engine_step, members, rows, n_seq}.  [rel M-0213 (1): the int list must skip, never es["engine_step"]]
        return {"ok": True, "checked_steps": 0, "skipped": "manifest declares no engine-step composition (B=1 / non-global manifest)"}
    padded = any(e.get("activity") == "bot-by-enablement" for e in bmap.get("entries", []))
    executed_of = None if request_steps is None else {str(k): {int(x) for x in v} for k, v in request_steps.items()}
    restricted = None
    if request_steps is not None and not padded:
        from verity_vllm.check.executed_prefix import restrict_engine_steps
        steps, restricted = restrict_engine_steps(steps, request_steps)
    ref = man_hdr.get("composition_reference") or {}
    tok, samp = ref.get("input_token_ids"), ref.get("sampled_token_ids")
    if not tok:
        return {"ok": False, "checked_steps": 0, "error": "global manifest names no input_token_ids reference member: composition unverifiable"}
    by_key: dict[tuple, list[dict[str, Any]]] = {}
    for e in bmap["entries"]:
        if e.get("kind") != "leaf":
            continue
        i = e["identity"]
        by_key.setdefault((int(i["step"]), i["op_path"], i["output_member"], int(i["invocation"])), []).append(e)
    problems: list[dict[str, Any]] = []
    checked = 0

    def _blocks(ents, per_row):
        out = []
        for e in ents:
            lo, hi = e["identity"]["element_range"]
            w = (int(hi) - int(lo)) // max(int(per_row), 1)
            off = int(e.get("element_offset", 0) or 0) // max(int(per_row), 1)
            out.append((off, off + w, e["identity"]["request_id"], w))
        return sorted(out)

    padding_steps_n = padding_blocks_n = 0
    for es in steps:
        e = int(es["engine_step"])
        expected = Counter((str(m["request_id"]), int(m["rows"])) for m in es["members"])
        ents_all = by_key.get((e, tok["op_path"], tok["output_member"], 0), [])
        ents = [x for x in ents_all if x.get("activity") != "bot-by-enablement"]
        pads = [x for x in ents_all if x.get("activity") == "bot-by-enablement"] if padded else []
        checked += 1
        per_row = max(int(tok.get("per_row", 1) or 1), 1)
        pad_got = Counter((str(x["identity"]["request_id"]), (int(x["identity"]["element_range"][1]) - int(x["identity"]["element_range"][0])) // per_row) for x in pads)
        pad_reqs = Counter(str(x["identity"]["request_id"]) for x in pads)
        if pads:
            padding_steps_n += 1
            padding_blocks_n += len(pads)
        if not ents and not pads:
            problems.append({"engine_step": e, "kind": "step-unbound", "member": f"{tok['op_path']}/{tok['output_member']}",
                             "missing": sorted(f"{r} x{n}" for r, n in expected)})
            continue
        blocks = _blocks(ents, per_row)
        got = Counter((rid, w) for _lo, _hi, rid, w in blocks)
        # [padded] the executed member's expectation = the declared members minus the padded ones; the UNION must be the declaration
        expected_exec = (expected - pad_got) if padded else expected
        missing = sorted(f"{r} x{n}" for (r, n), c in (expected_exec - got).items() for _ in range(c))
        unexpected = sorted(f"{r} x{n}" for (r, n), c in (got - expected_exec).items() for _ in range(c))
        dup = sorted(r for r, c in Counter(rid for _lo, _hi, rid, _w in blocks).items() if c > 1)
        pad_problems: list[str] = []
        if padded:
            extra_pad = sorted(f"{r} x{n}" for (r, n), c in (pad_got - expected).items() for _ in range(c))
            if extra_pad:
                pad_problems.append(f"padding for no declared member: {extra_pad[:3]}")
            dup_pad = sorted(r for r, c in pad_reqs.items() if c > 1)
            if dup_pad:
                pad_problems.append(f"padding twice for {dup_pad[:3]}")
            both = sorted(set(pad_reqs) & {rid for _lo, _hi, rid, _w in blocks})
            if both:
                pad_problems.append(f"double binding: executed block AND padding entry for {both[:3]}")
            if executed_of is not None:
                inside = sorted(r for r in pad_reqs if r in executed_of and e in executed_of[r])
                if inside:
                    pad_problems.append(f"padding inside the executed prefix (the inactive fallback, R17-42) for {inside[:3]}")
                past = sorted({rid for _lo, _hi, rid, _w in blocks if rid in executed_of and e not in executed_of[rid]})
                if past:
                    pad_problems.append(f"executed block past the stop for {past[:3]}")
        overlaps, cursor, gaps = [], 0, []
        total = int((ents[0].get("encoding") or {}).get("shape", [0])[0] or 0) if ents else 0
        for lo, hi, rid, _w in blocks:
            if lo < cursor:
                overlaps.append(f"{rid} rows [{lo},{hi}) overlaps the previous block ending at {cursor}")
            elif lo > cursor:
                gaps.append(f"rows [{cursor},{lo}) bound to no request")
            cursor = max(cursor, hi)
        if total and cursor != total:
            gaps.append(f"rows [{cursor},{total}) bound to no request")
        if missing or unexpected or dup or overlaps or gaps or pad_problems:
            problems.append({"engine_step": e, "kind": "composition", "member": f"{tok['op_path']}/{tok['output_member']}", "missing": missing,
                             "unexpected": unexpected, "duplicated": dup, "overlaps": overlaps[:4], "gaps": gaps[:4],
                             **({"padding": pad_problems[:4], "padded_members": sorted(f"{r} x{n}" for r, n in pad_got.elements())} if padded else {}),
                             "declared": sorted(f"{r} x{n}" for r, n in expected.elements()), "bound": [f"{rid} rows [{lo},{hi})" for lo, hi, rid, _w in blocks]})
            continue
        if samp:
            sents_all = by_key.get((e, samp["op_path"], samp["output_member"], 0), [])
            sents = [x for x in sents_all if x.get("activity") != "bot-by-enablement"]
            spads = sorted(str(x["identity"]["request_id"]) for x in sents_all if x.get("activity") == "bot-by-enablement") if padded else []
            want = sorted(str(m["request_id"]) for m in es["members"] if not padded or str(m["request_id"]) not in pad_reqs)
            have = sorted(x["identity"]["request_id"] for x in sents)
            offs = [int(x.get("element_offset", 0) or 0) for x in sents]
            want_pad = sorted(pad_reqs.elements()) if padded else []
            if have != want or len(set(offs)) != len(offs) or (padded and spads != want_pad):
                problems.append({"engine_step": e, "kind": "sampled-slots", "member": f"{samp['op_path']}/{samp['output_member']}",
                                 "missing": sorted(set(want) - set(have)), "unexpected": sorted(set(have) - set(want)),
                                 "duplicated": sorted(r for r, c in Counter(have).items() if c > 1), "shared_slots": len(offs) - len(set(offs)),
                                 **({"padding_slots": spads, "padding_slots_wanted": want_pad} if padded else {})})
    ok = not problems
    first = problems[0] if problems else None
    err = None
    if first is not None:
        parts = [f"engine step {first['engine_step']} {first['kind']} on {first['member']}"]
        for k in ("missing", "unexpected", "duplicated", "overlaps", "gaps", "padding"):
            if first.get(k):
                parts.append(f"{k} {first[k][:3]}")
        err = "composition: " + "; ".join(parts) + (f" (+{len(problems) - 1} more steps)" if len(problems) > 1 else "")
    return {"ok": ok, "checked_steps": checked, "problem_steps_n": len(problems), "problems": problems[:max_report], "error": err,
            "reference": {"input_token_ids": tok, "sampled_token_ids": samp},
            **({"executed_prefix_restriction": restricted, "steps_not_executed": restricted["steps_dropped"]} if restricted is not None else {}),
            **({"padded": {"rule": "B9.1 / F-r17b-66: executed member tiled by the active members (cursor == live rows); a bot-by-enablement entry "
                                   "is its own row space, one per (absent request, declared rows); executed + padding == the declared cap composition",
                           "steps_with_padding": padding_steps_n, "padding_blocks": padding_blocks_n, "declared_composition_unrestricted": True}} if padded else {})}


# ---------------------------------------------------------------------- sampling from the expected population

def challenge_identities(population: Iterable[tuple], bmap: dict[str, Any], run_root: bytes, k: int, *, salt: bytes = b"",
                         forced_late_step: bool = True, legacy: bool = CH.LEGACY,
                         source: bytes | None = None) -> list[tuple[tuple, dict[str, Any], int]]:
    """Draw `k` identities from the EXPECTED population (not from the leaves that exist) with `challenge.identity_picker` over
    (run_root, map digest, salt); map each to one committed leaf of its binding (a leaf chosen by the same picker inside the entry's
    leaf range).  Forced picks: the
    last step and one identity of every op class family present.  Returns [(identity, entry, leaf_index)]; an identity WITHOUT a
    binding is returned with entry None (the opening then fails -- coverage and openings are both fail-closed)."""
    pop = sorted(p for p in set(population) if p[2] >= 0)      # leaf-bound identities only (registered inputs open by digest, not by leaf)
    if not pop or k <= 0:
        return []
    idx = {kk: [e for e in es if e.get("kind") == "leaf"] for kk, es in map_index(bmap).items()}
    idx = {kk: es for kk, es in idx.items() if es}
    pick = CH.identity_picker(run_root, bmap["digest"], salt, legacy=legacy, source=source)

    chosen: list[tuple] = [pop[pick(i, len(pop))] for i in range(k)]
    if forced_late_step:
        last = max(p[2] for p in pop)
        late = [p for p in pop if p[2] == last]
        chosen.append(late[pick(k + 1, len(late))])
    fams: dict[str, tuple] = {}
    for p in pop:
        fam = p[4].split(".")[-1].split("/")[0] if not p[4].startswith(HIDDEN_FAMILIES) else p[4].split("/")[0]
        fams.setdefault(fam, p)
    chosen.extend(fams.values())
    out = []
    op_keys = _op_key_index(idx)
    for j, ident in enumerate(chosen):
        pd, rid, st, inv, op, mem, (lo, hi), rk = ident
        es = idx.get((rid, st, inv, op, mem, rk), [])
        if not es and mem == "*":
            es = [e for kk in op_keys.get((rid, st, inv, op, rk), []) for e in idx[kk]]
        if not es:
            out.append((ident, None, -1))
            continue
        e = es[pick(1000 + j, len(es))]
        leaf = e["leaf_lo"] + pick(2000 + j, max(e["leaf_hi"] - e["leaf_lo"], 1))
        out.append((ident, e, leaf))
    return out


# ---------------------------------------------------------------------- value correspondence (separate from coverage)

def value_correspondence(openings: list, entries: list[dict[str, Any] | None], reference, *, chunk: int) -> dict[str, Any]:
    """For each (opening, map entry): the opened chunk bytes must equal the PRODUCER's bytes for that identity.  `reference(entry)` returns
    the bytes the producer wrote for that (step, name, invocation) -- the test's producer, or a replay -- or None (not checkable).
    Openings can verify against the root and still fail here (late read after overwrite; relabelled occurrence)."""
    bad = []
    checked = 0
    for o, e in zip(openings, entries, strict=True):
        if e is None:
            bad.append({"step": o.step, "index": o.index, "why": "no binding"})
            continue
        ref = reference(e)
        if ref is None:
            continue
        checked += 1
        start = o.index * chunk - e["stream_offset"]
        want = ref[max(start, 0):max(start, 0) + len(o.value)] if start >= 0 else ref[:len(o.value) + start]
        got = o.value[:len(want)] if start >= 0 else o.value[-start:]
        if got[:len(want)] != want:
            bad.append({"step": o.step, "index": o.index, "name": e["name"], "why": "opened bytes != producer bytes"})
    return {"ok": not bad, "checked": checked, "bad_n": len(bad), "bad": bad[:8]}
