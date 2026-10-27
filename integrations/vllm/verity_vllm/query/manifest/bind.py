"""One naming scheme: every committed value's identity is a row of the required manifest, and the runtime events that carry the
value are bound to that row by structure.  Nothing reconciles a tap's spelling with the manifest's.

IDENTITY.  The manifest of record (`verity_vllm.query.required`) spells each required Value as (op_path, output_member, invocation) at
an engine step on a rank, from the Program and its correspondence alone: op_path is the producing Call's owning module (a collective
site's enclosing module at invocation k, `<module>/<fx node>` for a Value two Calls of one module body exchange, and for each fx node's
Value when two fx nodes of the module produce a member of one name at the step); output_member is the Definition's port name when it
returns several (`<hint>.<port>` when the Build recorded a hint, which names the Call), else the recorded hint, else `out`.  The
protocol's encodings are identity format, each one constant the manifest and the committing tap share:
`part_rank<r>` / `shard_rank<r>` (`format.TP_PARTIAL_MEMBER`), a narrowed linear's rows `<member>@<width>` beside the whole rows'
`<member>` (`format.NARROWED_MEMBER`), a norm site's `all_reduce` (`collective_names.NORM_SITE_MEMBER`), the
token ids `<consumer>/in`, `runner.sampler/sampled_token_ids`, the kernel streams `fa2.m1/L<k>/<geometry>` and `moe/L<k>/<plane>`.  A
row also carries its Call's `spec`, `port` and `position` (Program order), outside the manifest digest.

EVENT.  What a tap commits during an engine step: a tensor name in commit order, `<module>[/<label>]`.  A label is either a tuple
slot of the module's return (`<module>` the whole single-tensor return, `<module>/<k>` slot k of a module path) -- vLLM's Python
return convention, which the Program does not record -- or the label a source commits a value it took elsewhere under (a call
boundary's `<module>/<fx node>/<member>` among them, whatever its member).

BINDING.  `Binder(rows, ports=..., rebind=...)` is built from the rank's manifest rows and what the committing sources declare, and binds
each step's events in four passes; the k-th event of a name in the step is occurrence k:

1. `rebind`: a name the collector armed for another identity binds it (a promoted Value observed at its consumer's input,
   `format.promoted_observation`); a promoted identity binds through nothing else, since its producer module's own output is by
   definition not the promoted Value;
2. a label a source declares (`ports`: label -> `Port(Definitions, port, call_boundary)`) binds the row of its module (for a call
   boundary, of `<module>/<fx node>` or of the module itself) that port `port` of one of those Definitions produces: DeepGEMM's quant
   branch `x_q` / `x_s` is its ports 0 / 1, the attention taps the DSA indexer's ports and the sparse MLA and W_UK_T absorb boundaries.
   The occurrences of every label declaring one Port at one module bind its candidates in (invocation, Program) order when there are as
   many of each, else none: the mHC taps' attention and FFN post_pre are one Definition's port twice per layer, in commit order.  A
   call boundary's fx node is numbered over its request's own steps, so a step that batches requests at different steps of their own
   spells one Call once per request (mixed arrivals): then occurrence k binds every request's k-th candidate, when each request has as
   many as there are occurrences, under the first request's spelling, and `spellings` names each request's own (the binding map's
   per-request entries);
3. any other label binds the member of its module that it equals, at invocation k (the call-boundary source, the partial, norm,
   sampler, token-id and plane taps, a capture map's taps whose label is the Call's hint);
4. the tuple slots of occurrence k of a module bind the members of that module at invocation k that 1-3 left, in Program order: slots
   0..n-1 for n members, slot i binds the i-th; one member left, slot 0 binds it; otherwise none binds (a hook that returned y while
   the quant branch went unacquired never binds y to port 0).  A narrowed linear's rows (a row's `narrowed_of`: some columns of the
   rows of a tensor the module returns) are no tuple slot: only a label binds them.

A row whose Values are rows r0.. of the runtime tensor (`runtime_rows`) binds like any other; the binding map starts each request's
entry at row r0 of its share (`runtime_place`).

An event no pass binds is an extra: a source label keeps itself as its identity (no manifest row has it), a tuple slot binds member
`""`, which no row has.  The coverage check (`binding.coverage_check`) then holds the map to the manifest's own spelling and names
every missing or doubly bound identity.  What stays by name, each with its reason: a source's declared label (the tap knows which
Call's port it reads) and the encodings above (identity format).
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from verity_vllm.query.manifest.format import batch_launch_shortfall, promoted_observation

__all__ = ["HIDDEN_FAMILIES", "Port", "Binder", "coverage_check", "runtime_place", "source_ports", "split_name"]

HIDDEN_FAMILIES = ("fa2.m1/", "fa3.m1/", "moe/")


@dataclass(frozen=True)
class Port:
    """What a source's label carries: port `port` of a Call of one of the Definitions `families`; `call_boundary`: the Call is one of
    several in its module's body, so its Value is spelt `<module>/<fx node>` when two Calls exchange it or two fx nodes produce its member
    at the step, else `<module>`."""

    families: frozenset[str]
    port: str
    call_boundary: bool = False


def source_ports(sources: Iterable[Any]) -> dict[str, Port]:
    """The committing sources' declared labels (`ports()`), merged; a label two sources declare differently is refused."""
    out: dict[str, Port] = {}
    for s in sources:
        f = getattr(s, "ports", None)
        for label, p in (f() if callable(f) else {}).items():
            if out.setdefault(label, p) != p:
                raise ValueError(f"label {label!r}: declared as {out[label]} and as {p} ({type(s).__name__})")
    return out


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


def _family(spec: Any) -> str:
    return str(spec or "").split("{", 1)[0]


def _engine_steps(r: Mapping[str, Any], arrivals: Mapping[str, int] | None, declared: Mapping[str, str]) -> list[int]:
    """The engine steps of a row, placed as `binding.load_required_manifest` places them (less its executed-prefix cut, which is the
    population's): its absolute `engine_step`, unless `arrivals` is given and the row has a request-relative `step` (`[lo, hi]` = each
    step of it), which lands at step + the request's arrival."""
    st, rid = r.get("step"), r.get("request_id")
    if r.get("engine_step") is not None and not (arrivals is not None and st is not None and rid not in (None, "*")):
        return [int(r["engine_step"])]
    if st in (None, "*"):
        return []
    lo, hi = st if isinstance(st, list) else (st, st)
    return [int(s) + int((arrivals or {}).get(_request(r, declared), 0)) for s in range(int(lo), int(hi) + 1)]


def _request(r: Mapping[str, Any], declared: Mapping[str, str]) -> str:
    """The row's request as the engine names it (`r<i>`): a declared request id through `declared`, a bare index prefixed."""
    rid = r.get("request_id")
    return declared.get(str(rid), f"r{rid}" if str(rid).isdigit() else str(rid))


def _on_rank(r: Mapping[str, Any], rank: int | None) -> bool:
    rk = r.get("rank", 0)
    if rank is None or rk in (None, "*"):
        return True
    return int(rk[0]) <= int(rank) <= int(rk[1]) if isinstance(rk, list) else int(rk) == int(rank)


class Binder:
    """The rank's manifest identities and the declarations that bind runtime events to them (module docstring)."""

    def __init__(self, rows: Iterable[Mapping[str, Any]], *, rank: int | None = None, ports: Mapping[str, Port] | None = None,
                 rebind: Mapping[str, Sequence[str]] | None = None, aliases: Mapping[str, Sequence[Any]] | None = None,
                 arrivals: Mapping[str, int] | None = None, declared: Mapping[str, str] | None = None) -> None:
        self.ports = dict(ports or {})
        self.rebind = {k: (str(v[0]), str(v[1])) for k, v in (rebind or {}).items()}
        self.aliases = {k: (str(v[0]), int(v[1])) for k, v in (aliases or {}).items()}
        #: engine step -> (op_path, member, invocation) -> (family, port, position)
        self.rows: dict[int, dict[tuple[str, str, int], tuple[str, str, tuple]]] = defaultdict(dict)
        #: engine step -> (op_path, member, invocation) -> {request whose rows spell it: the row's position in that request's Program}
        self.requests: dict[int, dict[tuple[str, str, int], dict[str, tuple]]] = defaultdict(dict)
        self.promoted: dict[int, set[tuple[str, str, int]]] = defaultdict(set)
        self.narrowed: dict[int, set[tuple[str, str, int]]] = defaultdict(set)
        #: engine step -> (op_path, member, invocation) -> {request: the first row of the runtime tensor its Values are (`runtime_rows`)}
        self.offsets: dict[int, dict[tuple[str, str, int], dict[str, int]]] = defaultdict(dict)
        for row in rows:
            r = {**row, **(row.get("identity") or {})}
            if r.get("family") == "weights" or r.get("op_path") == "weights" or not _on_rank(r, rank):
                continue
            key = (str(r["op_path"]), str(r.get("output_member", "out")), int(r.get("invocation", 0)))
            pos = tuple(r.get("position") or ())
            for st in _engine_steps(r, arrivals, declared or {}):
                self.rows[st].setdefault(key, (_family(r.get("spec")), str(r.get("port", key[1])), pos))
                self.requests[st].setdefault(key, {}).setdefault(_request(r, declared or {}), pos)
                if r.get("promoted"):
                    self.promoted[st].add(key)
                if r.get("narrowed_of"):
                    self.narrowed[st].add(key)
                if r.get("runtime_rows"):
                    self.offsets[st].setdefault(key, {})[_request(r, declared or {})] = int(r["runtime_rows"][0])
        self._by_op: dict[int, dict[str, list[tuple[str, str, int]]]] = {}
        self._spelt: dict[tuple[int, tuple[str, str, int]], dict[str, tuple[str, str, int]]] = {}

    @classmethod
    def of_manifest(cls, manifest: Mapping[str, Any], *, rank: int | None = None, ports: Mapping[str, Port] | None = None,
                    arrivals: Mapping[str, int] | None = None) -> "Binder":
        """The binder of a manifest document: its rows (of `rank`), its shared-instance aliases, its promoted observations and its
        declared requests."""
        rows = manifest.get("identities") or manifest.get("entries") or manifest.get("rows") or []
        rebind = promoted_observation(dict(manifest))["names"] if any(r.get("promoted") for r in rows) else None
        declared = {str(d["request_id"]): f"r{int(d.get('index', i))}" for i, d in enumerate(manifest.get("requests_declared") or [])
                    if d.get("request_id") is not None}
        return cls(rows, rank=rank, ports=ports, rebind=rebind, aliases=manifest.get("op_path_aliases"), arrivals=arrivals, declared=declared)

    def _ops(self, step: int) -> dict[str, list[tuple[str, str, int]]]:
        """op_path -> its row keys at `step`, in (invocation, Program) order."""
        if step not in self._by_op:
            ops: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
            for key, (_f, _p, pos) in sorted(self.rows.get(step, {}).items(), key=lambda kv: (kv[0][2], kv[1][2], kv[0][1])):
                ops[key[0]].append(key)
            self._by_op[step] = ops
        return self._by_op[step]

    def _candidates(self, step: int, op: str, port: Port) -> list[tuple[str, str, int]]:
        rows, ops = self.rows.get(step, {}), self._ops(step)
        keys = [k for o, ks in ops.items() if o == op or "/" in o and o.rsplit("/", 1)[0] == op for k in ks] if port.call_boundary else ops.get(op, [])
        return [k for k in keys if rows[k][0] in port.families and rows[k][1] == port.port]

    def _per_request(self, step: int, cands: list[tuple[str, str, int]], n: int | None = None) -> dict[str, list[tuple[str, str, int]]]:
        """A call-boundary port's `cands` at `step` as each request spells them, in (invocation, Program) order of the request's own
        Program, when every request has as many (`n` if given); else {}.  A key two requests at different steps of their own both spell
        (the module's own output) sits at a different place in each one's Program, and `rows` keeps only the first request's position."""
        reqs, each = self.requests.get(step, {}), defaultdict(list)
        for c in cands:
            for q in sorted(reqs.get(c, ())):
                each[q].append(c)
        for q, ks in each.items():
            ks.sort(key=lambda c: (c[2], reqs[c][q], c[1]))
        sizes = {len(v) for v in each.values()}
        return dict(sorted(each.items())) if "*" not in each and len(sizes) == 1 and n in (None, *sizes) else {}

    def spellings(self, step: int, key: tuple[str, str, int]) -> dict[str, tuple[str, str, int]]:
        """Each request's own spelling of the call boundary `key` at `step` (module docstring, pass 2): {request: its key at the same place
        in its Program order}, {} for a key no call-boundary port declares."""
        step = int(step)
        if (step, key) not in self._spelt:
            fam, prt, _pos = self.rows.get(step, {}).get(key, ("", "", ()))
            port = next((p for p in self.ports.values() if p.call_boundary and fam in p.families and p.port == prt), None)
            each = self._per_request(step, self._candidates(step, key[0].rsplit("/", 1)[0], port)) if port and "/" in key[0] else {}
            for at in zip(*each.values()):
                spelt = dict(zip(each, at))
                self._spelt.update({(step, c): spelt for c in at})
            self._spelt.setdefault((step, key), {})
        return self._spelt[(step, key)]

    def bind(self, step: int, names: Sequence[str]) -> list[tuple[str, str, int, str]]:
        """The identity each event of `step` binds, as (op_path, member, invocation, how); how = `consumer-input` | `port` | `label` |
        `slot` | `unbound`."""
        step = int(step)
        rows = self.rows.get(step, {})
        out: list[tuple[str, str, int, str] | None] = [None] * len(names)
        taken = set(self.promoted.get(step, ()))
        seen: Counter = Counter()
        declared: dict[tuple[str, Port], list[tuple[int, int, str]]] = defaultdict(list)
        slots: dict[tuple[str, int], dict[int, int]] = defaultdict(dict)
        for i, name in enumerate(names):
            k = seen[name]
            seen[name] += 1
            if name in self.rebind:
                op, member = self.rebind[name]
                out[i] = (op, member, k, "consumer-input")
                taken.add((op, member, k))
                continue
            op, label = split_name(name)
            if op in self.aliases:
                op, k = self.aliases[op]
            if "/" not in name or (label.isdigit() and "/" not in op):
                slot = int(label) if "/" in name else 0
                if slot in slots[(op, k)]:
                    out[i] = (op, "", k, "unbound")
                else:
                    slots[(op, k)][slot] = i
            elif label in self.ports:
                declared[(op, self.ports[label])].append((i, k, label))
            elif (op, label, k) in rows and (op, label, k) not in taken:
                out[i] = (op, label, k, "label")
                taken.add((op, label, k))
            else:
                out[i] = (op, label, k, "unbound")
        for (op, port), events in declared.items():
            cands = [c for c in self._candidates(step, op, port) if c not in taken]
            each = self._per_request(step, cands, len(events)) if port.call_boundary and len(cands) != len(events) else {}
            firsts = next(iter(each.values())) if each else cands
            ok = len(firsts) == len(events)
            for j, ((i, k, label), c) in enumerate(zip(events, firsts if ok else [None] * len(events))):
                out[i] = (*c, "port") if c else (op, label, k, "unbound")
                if c:
                    taken.update([v[j] for v in each.values()] if each else [c])
        ops, narrowed = self._ops(step), self.narrowed.get(step, ())
        for (op, k), sl in slots.items():
            left = [c for c in ops.get(op, []) if c[2] == k and c not in taken and c not in narrowed]
            order = sorted(sl)
            pairs = list(zip(order, left)) if order == list(range(len(left))) else [(0, left[0])] if len(left) == 1 and 0 in sl else []
            for slot, c in pairs:
                out[sl[slot]] = (*c, "slot")
                taken.add(c)
            for slot in order:
                if out[sl[slot]] is None:
                    out[sl[slot]] = (op, "", k, "unbound")
        return out  # type: ignore[return-value]


def runtime_place(binder: Binder | None, step: int, key: tuple[str, str, int], request: str, lo: int, hi: int, goff: int,
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


def _request_share_shortfall(numel: int, rows: int | None, tokens: int, rows_r: int | None, hi: int) -> str | None:
    """Whether a request-local range [0, hi) of one batched per-token plane (a MoE block plane over the step's requests) fits inside the
    request's own `rows_r` token rows of it: the plane's rows are a multiple f of the step's `tokens` (f = TOPK for `act`), the request's
    share is rows_r x f of them.  Without a shape the plane's words must split into `tokens` equal rows.  None, or the shortfall named."""
    if rows_r is None:
        return "the request has no declared rows at the step"
    if tokens <= 0:
        return "the step declares no token rows"
    if rows is None:
        if numel % tokens:
            return f"{numel} words do not split into the step's {tokens} token rows"
        share = numel // tokens * rows_r
    else:
        if rows < tokens or rows % tokens:
            return f"{rows} rows, not a multiple of the step's {tokens} token rows: the plane covers no request"
        share = rows_r * (rows // tokens) * (numel // rows)
    return None if hi <= share else f"the request-local range ends at {hi}; the request's {rows_r} token rows of the plane hold {share} words"


def coverage_check(manifest: dict[str, Any], layouts: dict[int | str, list[dict[str, Any]]], *, request_id: str | int | None = None,
                   rank: int = 0, max_report: int = 12, ports: Any = None) -> dict[str, Any]:
    """Every manifest identity of (`request_id`, `rank`) must have a committed tensor bound to the same (step, op_path, output_member,
    invocation) whose element range CONTAINS the manifest's, past the rows before a row's `runtime_rows`; `layouts` = {step: [{"name",
    "numel"|"shape", ...} in layout order]}, each name bound by the manifest's binder (`Binder`, with the committing sources' declared
    `ports`).  Reports missing
    identities (by family / step), and the committed extras (allowed).  Identity coverage only -- value correspondence is bind's separate
    check.  A kernel-stream tensor (`HIDDEN_FAMILIES`) the binder leaves unbound is still committed under its own name, as the binding
    map keeps it: a B>1 batch-level hidden row (request `*`) names the padded whole-step geometry, and is covered when the tensors of its
    op at the step carry the whole stream of the sequences its `requests` list (`format.batch_launch_shortfall`, their rows from the
    declared composition), whatever the spelling.  In a global manifest a MoE block plane is one batched tensor per step: a request's
    request-local range must fit inside its own token rows of it, and a plane whose rows are not a multiple of the step's tokens covers
    no request (named)."""
    if manifest.get("complete") is False:                                        # an incomplete manifest never yields coverage
        um = manifest.get("unmodelled") or {}
        n = len(manifest.get("identities") or [])
        return {"ok": False, "error": "manifest incomplete: unmodelled Program families " + "; ".join(f"{k} x{v}" for k, v in sorted(um.items())),
                "manifest_digest": manifest.get("manifest_digest"), "checked": n, "missing_n": n, "missing_by_family": {"(unmodelled)": n},
                "checked_by_family": {}, "families_missing_entirely": sorted(um), "steps_with_missing": [], "missing": [],
                "committed_unmatched_n": 0, "committed_unmatched": [], "committed_extras_n": 0, "duplicate_names_n": 0, "steps_committed": [], "unmodelled": um}
    binder = Binder.of_manifest(manifest, rank=rank, ports=ports)
    have: dict[tuple, list[int]] = {}
    rows_of: dict[tuple, list[int | None]] = {}
    by_op: dict[tuple, list[tuple[str, int]]] = defaultdict(list)
    extras = 0
    for step, metas in layouts.items():
        for m, (op, member, inv, how) in zip(metas, binder.bind(int(step), [m["name"] for m in metas])):
            if how == "unbound" and not op.startswith(HIDDEN_FAMILIES):
                extras += 1
                continue
            numel = m.get("numel")
            if numel is None:
                numel = 1
                for d in m.get("shape") or []:
                    numel *= int(d)
            rows = int(m["shape"][0]) if m.get("shape") else None
            for key in (set(binder.spellings(int(step), (op, member, inv)).values()) if how == "port" else ()) or [(op, member, inv)]:
                have.setdefault((int(step), *key), []).append(int(numel))
                rows_of.setdefault((int(step), *key), []).append(rows)
                by_op[(int(step), key[0], key[2])].append((key[1], int(numel)))
    # the declared composition: engine step -> {request: token rows}, and the step's token rows
    comp: dict[int, dict[str, int]] = {}
    tokens: dict[int, int] = {}
    for es in manifest.get("engine_steps") or []:
        if isinstance(es, dict):
            comp[int(es["engine_step"])] = {str(m["request_id"]): int(m["rows"]) for m in es.get("members") or []}
            tokens[int(es["engine_step"])] = int(es.get("rows", sum(comp[int(es["engine_step"])].values())))
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
        why = None
        if r.get("request_id") == "*" and isinstance(r.get("requests"), list) and r["requests"]:
            # a batch-level hidden row (B>1: one attention launch per layer per engine step over the co-scheduled
            # sequences) names the padded whole-step geometry; the tap stages the launch whole or as per-SEQUENCE planes
            # (`s<i>/g<h>x<rows_i>x<NB>/<class>`).  Covered when the op's tensors at the step carry the whole stream of the sequences the row
            # lists, sized by their names' geometry; the same rule as binding.coverage_check.
            members = by_op.get((k[0], k[1], k[3]), [])
            undeclared = [q for q in r["requests"] if str(q) not in comp.get(k[0], {})]
            why = (f"requests {undeclared[:3]} have no declared rows at engine step {k[0]}" if undeclared else
                   batch_launch_shortfall(r.get("geometry"), [comp[k[0]][str(q)] for q in r["requests"]], members))
            got, hi = ([w for _m, w in members] or None), None
            if why is None:
                matched_keys.update((k[0], k[1], mem, k[3]) for mem, _w in members)
        elif got and manifest.get("global") and r["family"] == "moe_block_stream" and k[0] in tokens:
            why = _request_share_shortfall(max(got), rows_of[k][got.index(max(got))], tokens[k[0]], comp[k[0]].get(str(r["request_id"])), int(hi))
        skip = int((r.get("runtime_rows") or [0])[0]) * int(r.get("width") or 0)          # the rows before them the Program does not read
        if not got or why or (hi is not None and max(got) < int(hi) + skip):
            missing.append({"family": r["family"], "step": r["step"], "engine_step": k[0], "request_id": r["request_id"], "op_path": r["op_path"],
                            "output_member": r["output_member"], "invocation": r["invocation"], "want_range": [lo, hi],
                            "committed_numel": got if got is None or len(got) <= 8 else got[:8] + ["..."]} | ({"why": why} if why else {}))
            by_family_missing[r["family"]] += 1
        elif r.get("request_id") != "*":
            matched_keys.add(k)
    unexpected = [k for k in have if k not in matched_keys]
    dup = {k: len(v) for k, v in have.items() if len(v) > 1}
    families_missing_whole = [f for f, n in by_family_checked.items() if by_family_missing.get(f, 0) == n]
    steps_missing_whole = sorted({int(r["engine_step"]) for r in missing}) if missing else []
    requests_missing = sorted({str(r["request_id"]) for r in missing}) if missing else []
    # global manifests address request-locally, so a layouts-only check cannot see WHICH request a block belongs to; it can and
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
            **({"promoted_observed_as": dict(sorted((k, list(v)) for k, v in binder.rebind.items()))} if binder.rebind else {})}
