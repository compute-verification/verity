"""The collector binding map and the required-value coverage check (`coverage`; rules `RULES_VERSION`).

`op_path` / `output_member` / `invocation` together are the occurrence key -- the serialization of a runtime
occurrence; the `member` field names a Value of the Call at that occurrence (the field name stays); the binding map is the record of
value acquisition (which committed bytes each required Value was taken from) and 'the manifest' is the required-value manifest.

Identity key (the required-value manifest's):
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
not observation) x the workload (requests, engine steps, TP ranks).  The Program/query-derived required-value manifest has the
same key and replaces `expected_population` when delivered (`coverage_check(pop, bmap)` takes any iterable of identities).

Two separate verdicts: `coverage_check` = required-value coverage (every required Value has >= 1 binding; duplicates/aliases/extra
bytes cannot hide a missing one), and value correspondence (the opened bytes under an identity are the producer's bytes), which is not
graded here: a late read or a relabel passes coverage, and openings that verify against the root do NOT imply correspondence.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Collection, Iterable
from typing import Any

from verity_vllm.commit import executed_prefix as _EPX
from verity_vllm.commit import json_chunks as JC
from verity_vllm.query.manifest import format as MF
from verity_vllm.query.manifest.bind import HIDDEN_FAMILIES, split_name
from verity_vllm.query.manifest.format import (
    PROMOTED_OBSERVATION_RULE as PROMOTED_INPUT_RULE,  # noqa: F401  -- re-exported
)
from verity_vllm.query.manifest.format import (
    promoted_observation as promoted_input_acquisition,
)

__all__ = ["HIDDEN_FAMILIES", "RULES_VERSION", "build_binding_map", "identity_dict", "load_required_manifest", "logical_request", "promoted_input_acquisition"]

IDENTITY_KEY = ("program_digest", "request_id", "step", "invocation", "op_path", "output_member", "element_range", "rank")
RULES_VERSION = "r14-v1"
MAP_SCHEMA = "binding-map/1"
MOE_FAMILY = "moe/"                 # block-stream planes: per-token (pair-major) members, split by request rows at B>1 (not per-launch)

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

#: a promoted member's observation point: the manifest module owns the rule (format.PROMOTED_OBSERVATION_RULE), re-exported above.


# ---------------------------------------------------------------------- names -> identity parts

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


class LaunchIdentity(tuple):
    """A batch-level hidden identity (member `*`: one per request a B>1 launch row lists, at its engine step) and the launch the op's
    committed stream must carry there: `launch` = {"geometry": the row's launch geometry, "rows": {r<i>: token rows} of the step's declared
    composition} (`format.batch_launch_shortfall`).  It equals its identity tuple: the population's digest and record read the identity only."""
    launch: dict[str, Any]

    def __new__(cls, ident: tuple, launch: dict[str, Any] | None = None) -> LaunchIdentity:
        self = super().__new__(cls, ident)
        self.launch = launch or {}
        return self


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


def runtime_place(binder: Any, step: int, key: tuple[str, str, int], request: str, lo: int, hi: int, goff: int,
                  meta: Any) -> tuple[tuple[str, str, int], int, int, str | None]:
    """A request's binding-map entry of `key` at `step`: its share [`lo`, `hi`) of `meta`'s runtime tensor at element `goff`, started at
    row r0 of the share when the request's Values start there (`runtime_rows`).  -> (key, hi, goff, None), the range ending at the
    share's end; unmoved, with why, when the share is not whole rows of the tensor or has no row r0; as it is without a binder."""
    r0 = binder.offsets.get(int(step), {}).get(key, {}).get(str(request), 0) if binder is not None else 0
    if not r0:
        return key, hi, goff, None
    rows = int(meta.shape[0]) if meta.shape else 0
    per_row = math.prod(int(s) for s in meta.shape) // rows if rows else 0
    n = (hi - lo) // per_row if per_row and (hi - lo) % per_row == 0 else None
    if n is not None and r0 < n:
        return key, hi - r0 * per_row, goff + r0 * per_row, None
    return key, hi, goff, (f"{meta.name}: {request}'s values start at its row {r0}; its share is "
                           + (f"rows [0, {n})" if n is not None else "not whole rows") + ": bound to no row")

def build_binding_map(layouts: dict[int, list], steps: list, run_root: bytes, program_digest: str, *, request_rows: dict[int, list[tuple[str, int, int]]] | None = None,
                      request_id: str = "r0", rank: int = 0, layout_version: str = "chunk-leaf-v1", chunk: int | None = None,
                      weight_registration: dict | None = None, collector: dict | None = None,
                      binder: Any = None, padded: Collection[str] = ()) -> dict[str, Any]:
    """identity -> commitment location, for every committed tensor of every step, from the published layouts + the step roots.

    `request_rows[step]` = [(request_id, row_lo, row_hi)] splits the leading (token/row) dimension of a batched member between the requests of that step
    (the scheduler's row order, recorded by the engine wrapper; a `padded` member, the step-input port's, has rows past the step's tokens); else the whole member binds `request_id`.
    A MoE block plane whose rows are not a token multiple binds no request: its entry is request `*` with `refused`, the reason; a request-owned value (`Binder.owners`) is its request's whole.
    Entries are compact (one per tensor x request); `digest` is over the canonical JSON of the entries -- recorded in the Commit record and
    mixed into the challenge, so the map cannot be chosen after the openings.
    `binder` (`query.manifest.bind.Binder`, from the required manifest): each tensor's identity, a call boundary's under each request's
    own spelling (`Binder.spellings`), how if not by label or slot (`observation`), its first row (`runtime_place`); else its name (`split_name`).
    A leaf index counts each meta's own leaves, its `chunk` (256 B under vllm-v1, rows-sha512's 4 KiB rows); a differing `chunk` is refused."""
    entries: list[dict[str, Any]] = []
    for step in sorted(layouts):
        metas = layouts[step]
        sc = steps[step] if step < len(steps) else None
        step_root = sc.root.hex() if sc is not None else None
        if binder is not None:
            bound = binder.bind(step, [m.name for m in metas])
        else:
            seen: Counter = Counter()
            bound = []
            for m in metas:
                bound.append((*split_name(m.name), seen[m.name], "label"))
                seen[m.name] += 1
        owners = binder.owners(step, bound, (request_rows or {}).get(step)) if binder is not None else [None] * len(bound)
        for m, (op, member, k, how), owner in zip(metas, bound, owners):
            numel = _numel(m.shape)
            if chunk is not None and int(chunk) != int(m.chunk):
                raise ValueError(f"build_binding_map: chunk={chunk} but {m.name} (step {step}) has {m.chunk}-byte leaves; pass the committer's or none")
            off = m.stream_off if m.stream_off is not None and m.stream_off >= 0 else m.first_leaf * m.chunk
            rr = None if owner else (request_rows or {}).get(step)
            splits: list[tuple[str, int, int, int]] = []            # (request, request-local lo, request-local hi, global element offset)
            refused = None
            if rr and m.shape and numel > 0 and not op.startswith(HIDDEN_FAMILIES):
                rows = int(m.shape[0])
                per_row = numel // max(rows, 1)
                tokens = max(hi for _r, _lo, hi in rr)
                if rows == tokens or (m.name in padded and rows >= tokens):           # per-token member: the request's token rows
                    splits = [(rid, 0, (hi - lo) * per_row, lo * per_row) for rid, lo, hi in rr]
                elif rows == len(rr):                                                 # per-request member (logits, sampled ids): the request's row
                    splits = [(rid, 0, per_row, i * per_row) for i, (rid, _lo, _hi) in enumerate(rr)]
                else:                                                                 # shared by every request of the step (whole member each)
                    splits = [(rid, 0, numel, 0) for rid, _lo, _hi in rr]
            elif rr and m.shape and numel > 0 and op.startswith(MOE_FAMILY):
                # the fused-MoE block-stream planes are PER-TOKEN members in pair-major order (the MoE block source; the Program
                # addresses pair p = t*TOPK + k): router_w/router_ids [M, TOPK], up/down [M, TOPK, N], sum [M, H] have rows == tokens, act
                # [M*TOPK, I] has rows == tokens*TOPK.  Request r's rows are ITS token rows (x TOPK for act) -- the required set is
                # routing-independent (M_r*TOPK pairs whatever the experts chose) and its position comes from the step's request->row map,
                # never from the routing: the hidden-launch rule below would point r1's request-local range at r0's rows, passed by containment.
                rows = int(m.shape[0])
                tokens = max(hi for _r, _lo, hi in rr)
                if tokens > 0 and rows >= tokens and rows % tokens == 0:
                    f = rows // tokens                                                # 1 for token planes, TOPK for the act plane
                    per_row = numel // rows
                    splits = [(rid, 0, (hi - lo) * f * per_row, lo * f * per_row) for rid, lo, hi in rr]
                else:                                                                 # not a token multiple: no request's rows can be read off it
                    refused = f"{m.name}: {rows} rows, not a multiple of the step's {tokens} token rows: bound to no request"
                    splits = [("*", 0, numel, 0)]
            elif rr and op.startswith(HIDDEN_FAMILIES):
                splits = [(rid, 0, numel, 0) for rid, _lo, _hi in rr]
            if not splits:
                splits = [(owner or request_id, 0, numel, 0)]
            spelt = binder.spellings(step, (op, member, k)) if binder is not None and how == "port" else {}
            for rid, lo, hi, goff in splits:
                (op_r, member_r, k_r), hi, goff, why = runtime_place(binder, step, spelt.get(rid, (op, member, k)), rid, lo, hi, goff, m)
                entries.append({
                    "identity": identity_dict(identity(program_digest, rid, step, k_r, op_r, member_r, lo, hi, rank)),
                    "kind": "leaf", "step_root": step_root, "layout": layout_version, "chunk": int(m.chunk),
                    "leaf_lo": int(off // m.chunk), "leaf_hi": int((off + max(m.nbytes, 1) - 1) // m.chunk + 1),
                    "stream_offset": int(off), "width_bytes": int(m.nbytes), "encoding": {"dtype": m.dtype, "shape": list(m.shape)}, "element_offset": int(goff),
                    "tensor_ordinal": int(m.ordinal), "name": m.name,
                    **({"observation": how} if how not in ("label", "slot") else {}),
                    **({"refused": refused or why} if refused or why else {}),
                })
    if weight_registration:
        entries.append({"identity": identity_dict(identity(program_digest, "*", -1, 0, "weights", "registered", 0, 0, rank)), "kind": "registered-input",
                        "digest": weight_registration.get("root") or weight_registration.get("digest") or json.dumps(weight_registration, sort_keys=True, default=str)[:128]})
    digest = JC.sha256_hex(JC.list_pieces(entries, sort_keys=True, separators=(",", ":"), default=str))   # sha256 of json.dumps(entries, <same>)
    return {"schema": MAP_SCHEMA, "rules_version": RULES_VERSION, "program_digest": program_digest, "run_root": run_root.hex(), "rank": rank,
            "layout": layout_version, "collector": collector or {}, "entries_n": len(entries), "digest": digest, "entries": entries}


# ---------------------------------------------------------------------- the expected population (collector-independent)


def load_required_manifest(path: str, program_digest: str, *, requests: list[str] | None = None, steps: int | None = None, ranks: list[int] | None = None,
                           arrivals: dict[str, int] | None = None, request_steps: dict[str, Any] | None = None) -> tuple[list[tuple], dict[str, Any]]:
    """The required-value manifest's `manifest.json` (Program x canonical VU query; identity key + family label per row) -> identity tuples.
    Row fields: request_id / step / invocation / rank may be an int|str, `"*"` (every one of the workload's), or a `[lo, hi]` range;
    `element_range` `[lo, hi)` (absent = whole member).  Registered weights rows (`op_path` `weights`, step `*`) map onto the single
    registered-input identity.  Returns (identities, header) where header carries the manifest's own digest for the record.
    `request_steps` = {r<i>: engine steps the request EXECUTED (its executed prefix of record: arrival .. arrival +
    executed_prefix_len)}: an identity of a listed request at an engine step outside its prefix is programmed-but-unexecuted -- NOT required
    (counted in header `programmed_not_executed_n`).  The manifest itself stays the Program's (cap shape); only the requirement is placed."""
    with open(path) as f:
        doc = json.loads(f.read())
    rows = doc.get("entries") or doc.get("rows") or doc.get("identities")   # the rows, under the first of their spellings that holds any
    if not rows:
        rows = []                                                           # a manifest that carries no row is digested and loaded as empty
    unmodelled = doc.get("unmodelled")
    hdr = {"path": path, "manifest_digest": doc.get("manifest_digest") or doc.get("digest"), "program_digest": doc.get("program_digest"),
           "complete": doc.get("complete", True), "unmodelled": unmodelled if unmodelled else {},   # incomplete manifest -> fail closed (commit_delta)
           "rows_n": len(rows), "version": doc.get("version") or doc.get("schema"), "query": doc.get("query") or doc.get("vu_query")}
    # the manifest is an INPUT the consumer must not trust blindly: its `manifest_digest` is recomputed
    # over the rows it carries (a manifest edited after digesting is refused); a zero-width identity is refused, never
    # "vacuously covered"; a request id that is not `r<k>` / `*` (nor resolvable through requests_declared) is refused.
    if doc.get("schema") != MF.SCHEMA:
        raise ValueError(f"required manifest {path}: schema {doc.get('schema')!r} is not {MF.SCHEMA!r}: refused")
    if not isinstance(doc.get("program_digest"), str) or not doc.get("program_digest"):
        raise ValueError(f"required manifest {path}: program_digest must be one string (got {type(doc.get('program_digest')).__name__}): refused")
    recomputed = MF.manifest_digest(rows)
    if recomputed != doc.get("manifest_digest"):
        raise ValueError(f"required manifest {path}: manifest_digest {str(doc.get('manifest_digest'))[:16]} != digest of its rows {recomputed[:16]}: " + ("a 64-hex SHA-256 digest: a manifest from before manifest digests became SHA-512, so rebuild it (`verity-vllm manifest build`), refused" if len(str(doc.get("manifest_digest"))) == 64 else "edited after digesting, refused"))
    hdr["manifest_digest_verified"] = True
    for r in rows:
        ident = r.get("identity") or r
        if r.get("family") == "weights" or ident.get("op_path") == "weights":
            continue
        er = ident.get("element_range")
        if er is None or len(er) != 2 or er[0] is None or (er[1] is not None and er[1] != -1 and int(er[1]) <= int(er[0])):
            raise ValueError(f"required manifest {path}: zero-width / malformed identity element_range {er} at step {ident.get('step')} "
                             f"{ident.get('op_path')}/{ident.get('output_member')} req {ident.get('request_id')}: refused")
    program = doc.get("program")
    df = program.get("dataflow") if isinstance(program, dict) else None   # only a manifest that carries its Program's block can state one
    if df:                                                                    # P_compiled: the dataflow is in the digest
        hdr["dataflow_in_digest"] = True
        hdr["dataflow"] = {k: df.get(k) for k in ("n_edges", "n_launch_edges", "n_undefined", "wiring_declared", "split_graph")}
    all_req = list(requests or [])
    all_steps = list(range(int(steps))) if steps is not None else []
    all_ranks = list(ranks or [0])
    # global (B>1) manifests name requests by the workload's own ids; `requests_declared[i].index` (else declaration order) -> r<i>
    requests_declared = doc.get("requests_declared")
    declared = {str(d.get("request_id")): f"r{int(d.get('index', i))}" for i, d in enumerate(requests_declared if requests_declared else ()) if d.get("request_id") is not None}
    hdr["requests_declared_n"] = len(declared)
    hdr["global"] = bool(doc.get("global", False))
    # the declared per-engine-step composition (which requests, how many rows each) and the members the map's partition is
    # read from -- `composition_check` holds the map to it; request ids normalised to r<i>
    # A B=1 `build` manifest writes engine_steps as the request's engine-step NUMBERS (ints: arrival + s, rc3 schema); only `build-global`
    # declares the dict composition.  The numbers are kept as `engine_step_numbers`, the composition is
    # declared only from dict entries (composition_check then skips, named) -- never subscript an int.
    engine_steps = doc.get("engine_steps")
    raw_es = list(engine_steps) if engine_steps else []
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
    rows_at = {es["engine_step"]: {str(m["request_id"]): int(m["rows"]) for m in es["members"]} for es in hdr["engine_steps"]}
    reference = doc.get("composition_reference")   # the members the map's partition is read from, which a global manifest names
    hdr["composition_reference"] = reference if reference else {}
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
    # the classification is ON THE RECORD by family / request / kind so the population's third bucket
    # (sampled_replay.population.not_executed_by_rule) reconciles against it identity-count by identity-count (executed_prefix.reconcile_not_executed);
    # the manifest of record is not edited: rows_n / manifest_digest above are the file's
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
        launch_row, geometry = False, None
        if req_field == "*" and isinstance(r.get("requests"), list) and r["requests"]:
            req_field = list(r["requests"])                                # batch-level member: the requests it is bound to, explicitly
            if op.startswith(HIDDEN_FAMILIES):
                # the tap stages the row's launch whole or per sequence (`s<i>/g<h>x<rows_i>x<NB>/<class>`) under other spellings: the member
                # is `*`, and the requirement the op's whole stream at the step, by the row's geometry and the step's rows (`LaunchIdentity`)
                member = "*"
                launch_row, geometry = True, ident.get("geometry", r.get("geometry"))
                hdr["hidden_batch_rows_op_level"] = hdr.get("hidden_batch_rows_op_level", 0) + 1
        engine_step = ident.get("engine_step", r.get("engine_step"))       # absolute engine step (batch-level rows) beats request-relative `step` ...
        # ... unless the caller passes the request schedule of record (`arrivals`) and the row carries a request-
        # relative `step`: then engine step = step + arrival (the compiled manifest's declared step_basis; its `engine_step` annotation is the
        # step the row was DERIVED for under the declared arrive_steps, which drift from the committed-step axis on a workload with gaps)
        if arrivals is not None and ident.get("step") is not None and ident.get("request_id") not in (None, "*"):
            if engine_step is not None:
                hdr["engine_step_rebased_rows"] = hdr.get("engine_step_rebased_rows", 0) + 1
            engine_step = None
        for rid in (req_field if isinstance(req_field, list) and not (len(req_field) == 2 and all(isinstance(x, int) for x in req_field)) else expand(req_field, all_req, "r0")):
            rid = declared.get(str(rid), f"r{rid}" if str(rid).isdecimal() else str(rid))   # workload id | index -> r<i>
            if not MF.REQ_RE.match(rid):                                   # a request the workload does not declare (e.g. a bare UUID)
                raise ValueError(f"required manifest {path}: request id {rid!r} is not r<k> and not declared in requests_declared: refused")
            arrive = int((arrivals or {}).get(rid, 0))                     # gprog reading: manifest step is request-relative; map step = engine step
            for st in ([int(engine_step)] if engine_step is not None else expand(ident.get("step"), all_steps, 0)):
                st = int(st) if engine_step is not None else int(st) + arrive
                if executed is not None and rid in executed and st not in executed[rid]:
                    hdr["programmed_not_executed_n"] += 1                    # past the request's executed prefix of record: the named class
                    _EPX._classify(hdr["not_executed_by_rule"], dict(ident, family=r.get("family"), requests=r.get("requests"),
                                                                        request_id=ident.get("request_id")), rid)
                    continue
                for inv in expand(ident.get("invocation"), [0], 0):
                    for rk in expand(ident.get("rank"), all_ranks, 0):
                        i = identity(program_digest, str(rid), int(st), int(inv), op, member, int(er[0]), int(er[1]), int(rk))
                        out.append(LaunchIdentity(i, {"geometry": geometry, "rows": rows_at.get(int(st), {})}) if launch_row else i)
    return out, hdr


# ---------------------------------------------------------------------- coverage


# ---------------------------------------------------------------------- sampling from the expected population

