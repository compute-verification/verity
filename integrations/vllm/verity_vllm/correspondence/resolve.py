"""verity_vllm.correspondence.resolve -- occurrence resolution: a runtime occurrence -> the Program value it is an observation of (v2 Phase 4b).

    resolve(program, correspondence, occurrence, live, observation=None) -> ValueRef

is a PURE function of its arguments: the Program (`Program`: the request component's calls per request-relative step), the runtime
correspondence (`Correspondence`: `owning_module(call)`, `return_slot(call, port)`, `arg_slot(call, i)` -- today an adapter over the
descriptor annotations of `instances.json.gz`, `DescriptorCorrespondence`; the correspondence lane's namespace exposes the same three
functions), the occurrence, and the LIVE correspondence state (`LiveState`: per request its prefill steps / sampling events seen / the
ordinal cursor of the current request-relative step; one owner per already-resolved value id; the engine steps' row tables).  It writes
nothing; `LiveState.advance(observation, ref)` is the one mutation and it keeps only what a later resolution reads -- never the expanded
list of observations.

The REQUEST channel wraps the Match stage's attribution of record -- `batch_decomp` (X-09) / `global_match` (G2) -- rather than
restating it: channel order meta -> dataflow -> weight-only -> rows, the row-count rule IS `BD.attribute`, the weight-only rule IS the
predicate `BD.is_weight_only` tests (every node operand a shared weight or an earlier weight-only value), the shared owner IS `BD.SHARED`.
An occurrence no channel attributes is resolved to no request with the refusal NAMED (`ValueRef.problems`) -- never dropped or guessed.

The request-relative STEP is X-09's segmentation (a request's sampling events; a chunked prefill's engine steps are all step 0 and each
chunk's discarded sample is not a step), the ORDINAL is the value's index among the request's values at (step, site, slot) in emission
order, and the CALL is the Program call the correspondence puts at that (step, site, slot, ordinal) -- `<consumer>/in` (a promoted interior
value observed at its consumer's input, F-r19-int-20) resolves through `arg_slot`.

`validate(observation, resolved, ...)` compares the resolution with the observation's HINT and the collector's own statements and yields
named faults: `unattributable`, `ambiguous-request`, `reads-several-requests`, `request-mismatch` (meta vs rows channel), `layer-mismatch`,
`position-mismatch`, `site-unknown`, `ordinal-out-of-range`, `slot-mismatch`, `view-mutated`, `hint-mismatch`, `rank-mismatch`.

`attribute_record(record_dir)` runs the stream (`observe.iter_observations`) through resolution and returns the per-request value lists
in node order exactly in the shape `BD.decompose` / `global_match._check` build (`per_req`, `shared_idx`, `unattributed`), so the two
paths can be compared identity set by identity set.
"""
from __future__ import annotations

import collections
import gzip
import json
import os
import re
from array import array
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Protocol

from verity_vllm.check.match import batch_decomp as BD

from verity_vllm.observe import contract as OB

SHARED = BD.SHARED                     # "*weight-only*": the owner of a value carrying no request's data (F-r19-int-15)
_T_RE = re.compile(r"(?:^|/)t=(\d+)(?:/|$)")
_LAYERS_RE = re.compile(r"(?:^|/)layers\.(\d+)(?:/|$)")


# ------------------------------------------------------------------------------------------------------------------ the contract


@dataclass(frozen=True)
class ValueRef:
    """The Program value an occurrence resolved to.  `request` is the record's request id (the caller maps it to `r<k>`), or SHARED.
    `channel` names HOW the request was attributed (meta | dataflow | weight-only | rows:<how>).  `call` is the Program call at
    (step, site, slot, ordinal) when the correspondence knows one, else None -- and `problems` names why not."""

    request: str | None
    step: int | None                    # request-relative step
    site: str
    slot: str
    ordinal: int | None
    rank: int
    call: Any = None
    channel: str = ""
    problems: tuple[str, ...] = ()

    def identity(self) -> tuple:
        """The identity key without the call: (request, step, site, slot, ordinal, rank)."""
        return (self.request, self.step, self.site, self.slot, self.ordinal, self.rank)


@dataclass(frozen=True)
class Fault:
    kind: str
    detail: str
    hint: Any = None

    def __str__(self) -> str:
        return f"{self.kind}: {self.detail}" + (f" (hint {self.hint!r})" if self.hint is not None else "")


class Correspondence(Protocol):
    """The three functions the correspondence lane owns (`DescriptorCorrespondence`: the runtime-correspondence record first, the descriptor
    annotations as the recorded fallback)."""

    def owning_module(self, call: Any) -> str | None: ...
    def return_slot(self, call: Any, port: int | str = 0) -> str | None: ...
    def arg_slot(self, call: Any, i: int) -> str | None: ...


class Program(Protocol):
    """The Program side of resolution: its calls per request-relative step, in Program order."""

    def calls_at(self, step: int) -> list[Any]: ...
    @property
    def n_steps(self) -> int: ...


# ------------------------------------------------------------------------------------------------------------------ correspondence today


def _norm_module(mod: str, prefix: str) -> str:
    """Program module path -> the implementation's path: strip the export wrapper's DETERMINED prefix (`collector_path`).  The prefix
    is `wrapper_prefix_of(rows, attention_paths)` -- the attention module paths the record names (result.json `attention_impls_observed`)
    matched against the Program's attention rows -- so a TP stub's `inner.model.` falls out of the same match; no wrapper spelling is
    assumed here (by-name lint, Draft 3.1 §12.4)."""
    from verity_vllm.query.manifest.format import collector_path
    return collector_path(mod, prefix)


class DescriptorCorrespondence:
    """`owning_module` / `return_slot` / `arg_slot` of a derived component Program.  Calls are the rows' indices `i` of `instances.json.gz`.

    Two readers, the canonical one first (the same shim shape as `verity_vllm.correspondence.reader_for_acquire` / `verity_vllm.correspondence.reader_for_query`):

      correspondence    `source = "correspondence"`: the Build's `runtime-correspondence/v1` record (`runtime_correspondence.read_program_dir`
                        in `ComponentProgram.load`, or the record on the descriptor / instances doc's `annotations["correspondence"]`, or a
                        `table=` handed in).  `owning_module(call)` is the record's `owning_module(call_id)` mapped to the implementation's
                        path by `query.correspondence.implementation_prefix` over the record's module set x `attention_paths` ("" when
                        nothing is observed: the record's own path stands).
      v1_annotations    `source = "v1_annotations"`: no record -> the rows' names (`<module path>/<fx node>/<member hint>[row]`,
                        `manifest_format.split_name`) under the DETERMINED wrapper prefix (`wrapper_prefix_of(rows, attention_paths)`,
                        `_norm_module`).  Taken only when the record is absent (or does not describe this Program: `problem`), and recorded.

    `provenance()` states which reader served (`source`, `digest`, `problem`).  `view_of(call)` is the row's declared output view when the
    row carries one (`out` type string / `view`), for `validate`."""

    def __init__(self, inst: dict[str, Any], *, prefix: str | None = None, attention_paths: Iterable[str] = (), table: Any = None,
                 digest: str | None = None, problem: str | None = None):
        from verity_vllm.query.manifest import format as MF
        from verity_vllm.correspondence import runtime as RC
        self.rows: list[dict] = list(inst.get("rows") or [])
        self.by_i: dict[int, dict] = {int(r["i"]): r for r in self.rows}
        self._split = MF.split_name
        self.segs = MF.segment_steps(self.rows)
        self._attention_paths = tuple(attention_paths)
        if table is None and RC._document_of(inst) is not None:                    # the record travels on the doc itself
            table = RC.load_correspondence(inst)
        if table is not None and len(table) != len(self.rows):
            problem = (problem or "") + (f"; " if problem else "") + f"correspondence record has {len(table)} Calls, the Program {len(self.rows)}: the record is of another Program"
            table = None
        self.table = table
        self.problem = problem
        if table is not None:
            from verity_vllm.correspondence.reader_for_query import implementation_prefix
            self.source = "correspondence"
            self.digest: str | None = digest or RC.correspondence_digest(table.doc)
            # the implementation (collector) path prefix, determined over the RECORD's module paths x the observed attention paths
            self.prefix = implementation_prefix({rec.module for rec in table}, self._attention_paths) if prefix is None else prefix
        else:
            self.source = "v1_annotations"
            self.digest = None
            self.prefix = MF.wrapper_prefix_of(self.rows, list(self._attention_paths)) if prefix is None else prefix

    def provenance(self) -> dict[str, Any]:
        """What served `owning_module`: `source` (correspondence | v1_annotations), the record's `digest`, the determined `prefix`, any `problem`."""
        from verity_vllm.correspondence import runtime as RC
        out: dict[str, Any] = {"source": self.source, "schema": RC.SCHEMA if self.table is not None else None, "digest": self.digest, "prefix": self.prefix}
        if self.table is None:
            out["fallback"] = "instances.json.gz row names parsed as <module>/<fx>[/<member>][row] under wrapper_prefix_of(rows, attention_paths)"
        if self.problem:
            out["problem"] = self.problem
        return out

    def owning_module(self, call: Any) -> str | None:
        r = self.by_i.get(int(call))
        if r is None:
            return None
        if self.table is not None:                                                  # canonical: the record's owning module
            mod = self.table.owning_module(int(call))
            return _norm_module(mod, self.prefix) if mod else None
        mod, _node, _hint, _row = self._split(r.get("name") or "")
        return _norm_module(mod, self.prefix) if mod else None

    def return_slot(self, call: Any, port: int | str = 0) -> str | None:
        r = self.by_i.get(int(call))
        if r is None:
            return None
        _mod, _node, hint, _row = self._split(r.get("name") or "")
        if hint in ("0", "1", "in", "sampled_token_ids"):
            return hint
        return str(port) if port not in (0, "0", "out") else "out"

    def arg_slot(self, call: Any, i: int) -> str | None:
        """The slot at which the call's i-th argument is observed: the consumer-side acquisition `<consumer>/in` (a promoted interior
        value, F-r19-int-20: the collector observes it at the consumer's input)."""
        mod = self.owning_module(call)
        return f"{mod}/in" if mod else None

    def view_of(self, call: Any) -> tuple | None:
        r = self.by_i.get(int(call))
        if r is None:
            return None
        v = r.get("view") or r.get("output_view")
        return OB._view(v) if v is not None else None

    def producers(self, call: Any) -> list[int]:
        r = self.by_i.get(int(call))
        out: list[int] = []
        for a in (r or {}).get("args") or []:
            for run in a:
                if run and run[0] == "n":
                    out.append(int(run[1]))
        return out


class ComponentProgram:
    """A request component Program (derived `instances.json.gz` doc) as `Program`: calls per request-relative step in Program order.
    `at(step, site, slot)` indexes the calls of one (step, site, slot) once, through the correspondence -- the lookup `resolve` makes."""

    def __init__(self, inst: dict[str, Any], corr: DescriptorCorrespondence | None = None):
        self.corr = corr if corr is not None else DescriptorCorrespondence(inst)
        self.segs = self.corr.segs
        self.rows = self.corr.rows
        self._index: dict[int, dict[tuple[str, str], list[int]]] = {}

    @property
    def n_steps(self) -> int:
        return len(self.segs)

    def calls_at(self, step: int) -> list[Any]:
        if step < 0 or step >= len(self.segs):
            return []
        lo, hi = self.segs[step]
        return [int(r["i"]) for r in self.rows[lo: hi + 1]]

    def at(self, step: int, site: str, slot: str) -> list[int]:
        idx = self._index.get(step)
        if idx is None:
            idx = self._index[step] = {}
            for c in self.calls_at(step):
                mod = self.corr.owning_module(c)
                if mod is None:
                    continue
                idx.setdefault((mod, self.corr.return_slot(c) or "out"), []).append(c)
                # the consumer-side acquisition of the call's inputs
                idx.setdefault((mod, "in"), []).append(c)
        return idx.get((site, slot), [])

    @classmethod
    def load(cls, derived_dir: str, *, attention_paths: Iterable[str] = ()) -> "ComponentProgram":
        """The component at `derived_dir`: its `instances.json.gz`, with the Build's correspondence record when the dir carries one
        (`runtime_correspondence.read_program_dir`: result.json `correspondence_digest` + descriptor annotation), else the v1 reading."""
        from verity_vllm.correspondence.runtime import read_program_dir
        from verity_vllm.program.compact import expanded_file_args
        with gzip.open(os.path.join(derived_dir, "instances.json.gz"), "rt") as f:
            inst = json.load(f)
        for r in inst.get("rows") or ():   # a v2 file's argument progressions -> the v1 per-leaf run lists (`producers` reads "n" runs)
            a = r.get("args")
            if a:
                r["args"] = expanded_file_args(a)
        table, digest, problem = read_program_dir(derived_dir)
        return cls(inst, DescriptorCorrespondence(inst, attention_paths=attention_paths, table=table, digest=digest, problem=problem))


# ------------------------------------------------------------------------------------------------------------------ live state


@dataclass
class _RequestCursor:
    prompt_len: int | None = None
    prefill_steps: set[int] = field(default_factory=set)      # engine steps in which the request had a prompt row (position < LP)
    ts_seen: int = 0                                          # sampling events of the request seen so far
    cur_step: int | None = None                               # request-relative step of the cursor
    ordinals: dict[tuple[str, str], int] = field(default_factory=dict)
    first_engine_step: int | None = None

    def local_step(self, engine_step: int, position: int | None) -> int:
        """The request-relative step of an occurrence at `engine_step` (pure): 0 while the request is in its prefill (a prompt-position
        row, an engine step already known as a prefill chunk, or the request's first appearance), else X-09's count -- the sampling
        events seen so far minus the chunk samples the engine discarded (one per prefill chunk), plus one."""
        if position is not None and self.prompt_len is not None and position < self.prompt_len:
            return 0
        if engine_step in self.prefill_steps or self.first_engine_step is None:
            return 0
        chunks = max(1, len(self.prefill_steps))
        return max(1, self.ts_seen - chunks + 1)


@dataclass
class LiveState:
    """What a later resolution reads -- nothing else.  `owner[value id] = request index` (int32 per value; -1 unknown, -2 SHARED) is the
    dataflow channel's table; `steps` the engine steps' row tables (the rows channel's input); `cursors` the per-request step/ordinal
    cursors; `views` the first view LAYOUT (shape, strides, dtype) seen per (site, slot, spec) (`validate`'s mutated-view reference)."""

    n_input_nodes: int
    owned_params: dict[int, str]                              # request-owned root parameter index -> request id (prompt, seed, peers)
    weights_params: frozenset[int]
    steps: dict[int, dict] | None = None
    requests: dict[str, _RequestCursor] = field(default_factory=dict)
    owner: array = field(default_factory=lambda: array("i"))
    req_index: dict[str, int] = field(default_factory=dict)
    req_names: list[str] = field(default_factory=list)
    views: dict[tuple[str, str, str | None], tuple] = field(default_factory=dict)

    @classmethod
    def of_header(cls, hdr: OB.RecordHeader, steps: dict[int, dict] | None, run_header: dict | None = None,
                  owned_extra: dict[int, str] | None = None) -> "LiveState":
        """The layout the fold contract fixes: prompt parameters (one per header request, in order), then seeds / peers (`owned_extra`),
        then the weights Record (its Input nodes = every input node no request owns)."""
        n_in = int(hdr.n_input_nodes or 0)
        owned: dict[int, str] = {}
        cursors: dict[str, _RequestCursor] = {}
        reqs = hdr.requests
        if run_header is not None:
            hreqs, param_of, _parent = BD.request_params(run_header)
            for rid, pidx in param_of.items():
                owned[pidx] = rid if pidx not in owned else owned[pidx]
            for r in hreqs:
                cursors[r["request_id"]] = _RequestCursor(prompt_len=len(r.get("prompt_token_ids") or ()))
            for c in (run_header.get("workload") or {}).get("children") or []:
                cursors[c["request_id"]] = _RequestCursor(prompt_len=cursors.get(c["parent"], _RequestCursor()).prompt_len)
        else:
            for r in reqs:
                owned[int(r["index"])] = str(r["request_id"])
                cursors[str(r["request_id"])] = _RequestCursor(prompt_len=r.get("prompt_len"))
        # the fold contract's parameter names (program.json `params`: prompt[_r], seed[_i], peers[_i], weights): a seed / peers parameter
        # is request-owned like the prompt (BD.decompose's seed_node_of / peer_node_of: the i-th is the i-th header request's)
        order = [str(r["request_id"]) for r in (run_header.get("workload") or {}).get("requests") or []] if run_header is not None else \
                [str(r["request_id"]) for r in reqs]
        for k, name in enumerate(hdr.params):
            head, _, tail = name.partition("_")
            if head in ("seed", "peers") and k not in owned:
                i = int(tail) if tail.isdigit() else 0
                if i < len(order):
                    owned[k] = order[i]
        for k, rid in (owned_extra or {}).items():
            owned[int(k)] = rid
        weights = frozenset(k for k in range(n_in) if k not in owned) if n_in else frozenset()
        st = cls(n_input_nodes=n_in, owned_params=owned, weights_params=weights, steps=steps, requests=cursors)
        for rid in cursors:
            st._req(rid)
        return st

    def _req(self, rid: str) -> int:
        i = self.req_index.get(rid)
        if i is None:
            i = self.req_index[rid] = len(self.req_names)
            self.req_names.append(rid)
        return i

    def owner_of(self, node: int) -> str | None:
        """The request that owns value `node` (SHARED for a weight-only value; None when not yet resolved / unknown)."""
        if node < len(self.owner):
            v = self.owner[node]
            if v == -2:
                return SHARED
            if v >= 0:
                return self.req_names[v]
        return None

    def _set_owner(self, node: int, owner: str | None) -> None:
        if node >= len(self.owner):
            self.owner.extend([-1] * (node + 1 - len(self.owner)))
        self.owner[node] = -2 if owner == SHARED else (-1 if owner is None else self._req(owner))

    def cursor(self, rid: str) -> _RequestCursor:
        c = self.requests.get(rid)
        if c is None:
            c = self.requests[rid] = _RequestCursor()
            self._req(rid)
        return c

    def advance(self, obs: OB.ValueObservation, ref: ValueRef) -> None:
        """Record what a later resolution reads: the value's owner, the request's sampling-event count, the ordinal taken."""
        node = obs.hint_value_id
        if isinstance(node, int):
            self._set_owner(self.n_input_nodes + node, ref.request)
        if ref.request is None or ref.request == SHARED:
            return
        c = self.cursor(ref.request)
        if c.first_engine_step is None:
            c.first_engine_step = obs.occurrence.step
        if ref.step == 0:
            c.prefill_steps.add(obs.occurrence.step)
        if is_sampling_event(obs.spec):
            c.ts_seen += 1
        key = (ref.site, ref.slot)
        c.ordinals[key] = c.ordinals.get(key, -1) + 1
        vk = (ref.site, ref.slot, obs.spec)
        if obs.occurrence.view is not None and vk not in self.views:
            self.views[vk] = OB.view_layout(obs.occurrence.view)


def is_sampling_event(spec: str | None) -> bool:
    from verity_vllm.program.sampling_event import is_sampling_event as _ise
    return bool(spec) and _ise(spec)


# ------------------------------------------------------------------------------------------------------------------ the request channel


def attribute_request(obs: OB.ValueObservation, live: LiveState) -> tuple[str | None, str]:
    """(request | SHARED | None, channel) for one observation -- the Match stage's channel order (BD.decompose / global_match._check):
    meta (the runtime stamped the row), dataflow (every request-owned operand names ONE request), weight-only (BD.is_weight_only's
    predicate: every value read is a shared weight or an earlier weight-only value, and there is one), rows (BD.attribute over the
    engine step's StepStart rows -- the fold's own row-count rule).  The channel text is the record's (`how` counts join)."""
    occ = obs.occurrence
    if occ.request is not None:
        return occ.request, "meta"
    deps: set = set()
    unknown: list[str] = []
    reads_weight = False
    for p in obs.param_reads:
        if p in live.owned_params:
            deps.add(live.owned_params[p])
        elif p in live.weights_params or (live.n_input_nodes and p >= live.n_input_nodes - 1):
            reads_weight = True
        else:
            unknown.append(f"param {p}")
    for idx in obs.reads:
        if idx < live.n_input_nodes:
            if idx in live.owned_params:
                deps.add(live.owned_params[idx])
            elif idx in live.weights_params:
                reads_weight = True
            else:
                unknown.append(f"input {idx}")
        else:
            own = live.owner_of(idx)
            if own == SHARED:
                reads_weight = True
            elif own is None:
                unknown.append(f"node {idx}")
            else:
                deps.add(own)
    if unknown:
        return None, "reads unattributed " + ", ".join(unknown[:6])
    if len(deps) == 1:
        return next(iter(deps)), "dataflow"
    if len(deps) > 1:
        return None, "reads several requests: " + ", ".join(sorted(deps))[:200]
    if reads_weight:
        return SHARED, "weight-only"
    st = (live.steps or {}).get(occ.step)
    if st is None:
        return None, f"no StepStart for step {occ.step}"
    rid, how = BD.attribute({"row": occ.row, "rows": occ.rows}, {"rows": st["rows"], "logits_rows": st["logits_rows"]})
    return rid, "rows:" + how


# ------------------------------------------------------------------------------------------------------------------ resolve


def resolve(program: Program | None, correspondence: Correspondence | None, occurrence: OB.RuntimeOccurrence, live: LiveState,
            observation: OB.ValueObservation | None = None) -> ValueRef:
    """The Program value `occurrence` observes.  Pure: reads `program`, `correspondence`, `live`; mutates none of them (the caller
    advances `live` with the result).  With no `program` / `correspondence` the request-relative identity is still resolved (request,
    step, site, slot, ordinal, rank) and `call` stays None with the reason named."""
    obs = observation if observation is not None else OB.ValueObservation(occurrence=occurrence)
    rid, how = attribute_request(obs, live)
    problems: list[str] = []
    if rid is None:
        kind = ("ambiguous-request" if how.startswith("rows:ambiguous") else
                "reads-several-requests" if how.startswith("reads several") else "unattributable")
        return ValueRef(None, None, occurrence.site, occurrence.slot, None, occurrence.rank, None, how, (f"{kind}: {how}",))
    if rid == SHARED:
        return ValueRef(SHARED, None, occurrence.site, occurrence.slot, None, occurrence.rank, None, how, ())
    c = live.cursor(rid)
    step = c.local_step(occurrence.step, occurrence.position)
    if step != c.cur_step:
        # the cursor's ordinals belong to ONE request-relative step; a new step starts the count over (read-only here: `advance` resets)
        ordinal = 0
    else:
        ordinal = c.ordinals.get((occurrence.site, occurrence.slot), -1) + 1
    call = None
    if program is not None and correspondence is not None:
        calls = _calls_at(program, correspondence, step, occurrence.site, occurrence.slot)
        if not calls:
            problems.append(f"site-unknown: no call of {occurrence.site!r}/{occurrence.slot!r} at request step {step} of the Program")
        elif ordinal >= len(calls):
            problems.append(f"ordinal-out-of-range: {ordinal} >= {len(calls)} calls of {occurrence.site!r}/{occurrence.slot!r} at step {step}")
        else:
            call = calls[ordinal]
    return ValueRef(rid, step, occurrence.site, occurrence.slot, ordinal, occurrence.rank, call, how, tuple(problems))


def _calls_at(program: Program, correspondence: Correspondence, step: int, site: str, slot: str) -> list[Any]:
    at = getattr(program, "at", None)
    if at is not None and getattr(program, "corr", None) is correspondence:
        return at(step, site, slot)
    out = []
    for call in program.calls_at(step):
        mod = correspondence.owning_module(call)
        if mod != site:
            continue
        if slot == "in" or (correspondence.return_slot(call) or "out") == slot:
            out.append(call)
    return out


def advance(live: LiveState, obs: OB.ValueObservation, ref: ValueRef) -> None:
    """Mutate the live state after `resolve`: reset the request's ordinal cursor on a new request-relative step, then record."""
    if ref.request not in (None, SHARED):
        c = live.cursor(ref.request)
        if ref.step != c.cur_step:
            c.cur_step = ref.step
            c.ordinals = {}
    live.advance(obs, ref)


# ------------------------------------------------------------------------------------------------------------------ validate


def validate(observation: OB.ValueObservation, resolved: ValueRef, *, correspondence: Correspondence | None = None, live: LiveState | None = None,
             expected_call_of_hint: Callable[[Any], Any] | None = None, expected_request_of_hint: Callable[[Any], str | None] | None = None) -> list[Fault]:
    """Named faults of one resolution against the observation's HINT and the collector's own statements.  Never raises."""
    faults: list[Fault] = []
    occ = observation.occurrence
    for p in resolved.problems:
        kind, _, detail = p.partition(": ")
        faults.append(Fault(kind, detail, observation.hint_value_id))
    if resolved.request in (None, SHARED):
        return faults
    # the meta channel vs the rows channel (v1's `row_rule_disagreements`): the runtime's stamp must agree with the row-count rule
    if occ.request is not None and live is not None and live.steps and occ.step in live.steps and occ.row is not None and occ.rows is not None:
        rr, hh = BD.attribute({"row": occ.row, "rows": occ.rows}, live.steps[occ.step])
        if rr is not None and rr != resolved.request and hh in ("token", "logits"):
            faults.append(Fault("request-mismatch", f"stamped {resolved.request!r}, the row-count rule ({hh}) names {rr!r}", observation.hint_value_id))
    # the collector's own name: `t=<position>` and `layers.<k>`
    if observation.name:
        m = _T_RE.search(observation.name)
        if m and occ.position is not None and int(m.group(1)) != occ.position:
            faults.append(Fault("position-mismatch", f"name says t={m.group(1)}, occurrence position {occ.position}", observation.hint_value_id))
        m = _LAYERS_RE.search(observation.name)
        lay = occ.layer()
        if m and lay is not None and int(m.group(1)) != lay:
            faults.append(Fault("layer-mismatch", f"name says layers.{m.group(1)}, site {occ.site!r} is layer {lay}", observation.hint_value_id))
    if resolved.call is not None and correspondence is not None:
        mod = correspondence.owning_module(resolved.call)
        if mod is not None and mod != resolved.site:
            faults.append(Fault("site-mismatch", f"call {resolved.call!r} is owned by {mod!r}, observed at {resolved.site!r}", observation.hint_value_id))
        if resolved.slot != "in":
            rs = correspondence.return_slot(resolved.call)
            if rs is not None and rs != resolved.slot:
                faults.append(Fault("slot-mismatch", f"call {resolved.call!r} returns at slot {rs!r}, observed at {resolved.slot!r}", observation.hint_value_id))
        vf = getattr(correspondence, "view_of", None)
        if vf is not None and occ.view is not None:
            want = vf(resolved.call)
            if want is not None and OB.view_layout(tuple(want)) != OB.view_layout(occ.view):
                faults.append(Fault("view-mutated", f"declared view {want!r}, observed {occ.view!r}", observation.hint_value_id))
        if expected_call_of_hint is not None and observation.hint_value_id is not None:
            want_call = expected_call_of_hint(observation.hint_value_id)
            if want_call is not None and want_call != resolved.call:
                faults.append(Fault("hint-mismatch", f"hint names call {want_call!r}, resolution gives {resolved.call!r}", observation.hint_value_id))
    if expected_request_of_hint is not None and observation.hint_value_id is not None:
        want_r = expected_request_of_hint(observation.hint_value_id)
        if want_r is not None and want_r != resolved.request:
            faults.append(Fault("hint-mismatch", f"hint names request {want_r!r}, resolution gives {resolved.request!r}", observation.hint_value_id))
    if live is not None and occ.view is not None:
        # the layout (shape, strides, dtype) of the first occurrence at (site, slot, spec) is the reference; the storage offset is the
        # row's address inside the launch's buffer and moves with every row (row k at k*H): not a mutation
        seen = live.views.get((resolved.site, resolved.slot, observation.spec))
        if seen is not None and seen != OB.view_layout(occ.view) and not any(f.kind == "view-mutated" for f in faults):
            faults.append(Fault("view-mutated", f"first layout at {resolved.site!r}/{resolved.slot!r} was {seen!r}, now {OB.view_layout(occ.view)!r}", observation.hint_value_id))
    return faults


# ------------------------------------------------------------------------------------------------------------------ the record, end to end


@dataclass
class Attribution:
    """The per-request value lists of a record in NODE order -- the shape BD.decompose / global_match._check build."""

    per_req: "collections.OrderedDict[str, list[int]]"        # request id -> node ids (root node index = n_input_nodes + instance index)
    shared_nodes: list[tuple[int, int]]                       # (node id, engine step) of every weight-only value
    unattributed: list[dict]
    how: collections.Counter
    faults: collections.Counter
    fault_examples: list[str]
    n_input_nodes: int
    disagreements: int = 0
    present_by_step: dict[int, set[str]] = field(default_factory=dict)    # engine step -> requests a resolved value of that step belongs to
    live: LiveState | None = None                                          # the final live state (per-request cursors: prefill steps, events)

    def legs(self, steps: dict[int, dict] | None) -> dict[str, list[int]]:
        """Each request's node list with the weight-only values of the engine steps it has rows in merged in (BD.shared_legs).  `steps`
        (the capture log's StepStart tables) says which requests had rows in a step; without the log the stream's own statement is used
        (`present_by_step`: the requests whose values were resolved in the step -- the same set when every row is stamped)."""
        if not self.shared_nodes:
            return dict(self.per_req)
        st = steps if steps else {s: {"rows": [(r, 0) for r in sorted(rs)], "logits_rows": []} for s, rs in self.present_by_step.items()}
        return BD.shared_legs(self.per_req, self.shared_nodes, st)

    def prefill_steps(self, rid: str) -> list[int]:
        """The engine steps in which the request had a prompt row (its prefill chunks), from the live cursors."""
        c = self.live.requests.get(rid) if self.live is not None else None
        return sorted(c.prefill_steps) if c is not None else []

    def instance_indices(self, legs: dict[str, list[int]] | None = None) -> dict[str, list[int]]:
        """Node ids -> instance indices (`node - n_input_nodes`, the index into `compare.Prog.inst` when no root node was dropped)."""
        src = legs if legs is not None else self.per_req
        return {rid: [k - self.n_input_nodes for k in ks] for rid, ks in src.items()}


def attribute_record(record_dir: str, *, programs: dict[str, ComponentProgram] | None = None, validate_each: bool = True,
                     max_fault_examples: int = 24, observations: Iterable[OB.ValueObservation] | None = None,
                     header: OB.RecordHeader | None = None, run_header: dict | None = None, steps: dict[int, dict] | None = None) -> Attribution:
    """Stream the record through resolve / validate.  `programs`: request id -> its component Program (optional: with none the identity
    is resolved without the call).  Returns the attribution in `BD.decompose`'s shape plus the named faults' counts."""
    hdr = header if header is not None else OB.read_header(record_dir)
    if run_header is None or steps is None:
        try:
            run_header, steps = OB.read_steps(record_dir)
        except FileNotFoundError:
            run_header, steps = None, None
    live = LiveState.of_header(hdr, steps, run_header)
    per_req: "collections.OrderedDict[str, list[int]]" = collections.OrderedDict()
    shared: list[tuple[int, int]] = []
    unattributed: list[dict] = []
    how: collections.Counter = collections.Counter()
    faults: collections.Counter = collections.Counter()
    examples: list[str] = []
    disagree = 0
    present: dict[int, set[str]] = {}
    stream = observations if observations is not None else OB.iter_observations(record_dir, header=hdr)
    for obs in stream:
        occ = obs.occurrence
        prog = programs.get(occ.request) if programs and occ.request is not None else None
        corr = prog.corr if prog is not None else None
        ref = resolve(prog, corr, occ, live, obs)
        node = int(obs.hint_value_id) + live.n_input_nodes if isinstance(obs.hint_value_id, int) else None
        if programs and prog is None and ref.request not in (None, SHARED):
            p2 = programs.get(ref.request)
            if p2 is not None:
                ref = resolve(p2, p2.corr, occ, live, obs)
        h = ref.channel
        how[h.split(":")[0] if h.startswith("rows") else h.split(" ")[0]] += 1
        if validate_each:
            for f in validate(obs, ref, correspondence=corr, live=live):
                faults[f.kind] += 1
                if f.kind == "request-mismatch":
                    disagree += 1
                if len(examples) < max_fault_examples:
                    examples.append(str(f))
        if ref.request is None:
            unattributed.append({"node": node, "spec": obs.spec, "step": occ.step, "row": occ.row, "rows": occ.rows, "why": h})
        elif ref.request == SHARED:
            shared.append((node, occ.step))
        else:
            per_req.setdefault(ref.request, []).append(node)
            present.setdefault(occ.step, set()).add(ref.request)
        advance(live, obs, ref)
    return Attribution(per_req=per_req, shared_nodes=shared, unattributed=unattributed, how=how, faults=faults, fault_examples=examples,
                       n_input_nodes=live.n_input_nodes, disagreements=disagree, present_by_step=present, live=live)


def compare_attributions(v1: dict[str, list[int]], v2: dict[str, list[int]]) -> dict[str, Any]:
    """Identity-SET equality per request of two attributions (request -> value ids); differences NAMED (never a count alone)."""
    out: dict[str, Any] = {"equal": True, "requests": {}, "only_v1": sorted(set(v1) - set(v2)), "only_v2": sorted(set(v2) - set(v1))}
    if out["only_v1"] or out["only_v2"]:
        out["equal"] = False
    for rid in sorted(set(v1) & set(v2)):
        a, b = set(v1[rid]), set(v2[rid])
        ent = {"n_v1": len(a), "n_v2": len(b), "equal": a == b}
        if a != b:
            out["equal"] = False
            ent["only_v1"] = sorted(a - b)[:12]
            ent["only_v2"] = sorted(b - a)[:12]
            ent["n_only_v1"], ent["n_only_v2"] = len(a - b), len(b - a)
        out["requests"][rid] = ent
    return out
