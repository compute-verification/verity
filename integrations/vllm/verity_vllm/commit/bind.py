"""One naming scheme: every committed value's identity is a row of the required manifest, and the runtime events that carry the
value are bound to that row by structure.  Nothing reconciles a tap's spelling with the manifest's.

IDENTITY.  The manifest of record (`verity_vllm.query.required`) spells each required Value as (op_path, output_member, invocation) at
an engine step on a rank, from the Program and its correspondence alone: op_path is the producing Call's owning module (a collective
site's enclosing module at invocation k, `<module>/<fx node>` for a Value two Calls of one module body exchange); output_member is the
Definition's port name when it returns several (`<hint>.<port>` when the Build recorded a hint, which names the Call), else the recorded
hint, else `out`.  The protocol's encodings are identity format, each one constant the manifest and the committing tap share:
`part_rank<r>` / `shard_rank<r>` (`format.TP_PARTIAL_MEMBER`), a norm site's `all_reduce` (`collective_names.NORM_SITE_MEMBER`), the
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
   boundary, of `<module>/<fx node>`) that port `port` of one of those Definitions produces: DeepGEMM's quant branch `x_q` / `x_s` is its
   ports 0 / 1, the attention taps the DSA indexer's ports and the sparse MLA and W_UK_T absorb boundaries.  The occurrences of the
   label bind the candidates in (invocation, Program) order when there are as many of each, else none;
3. any other label binds the member of its module that it equals, at invocation k (the call-boundary source, the partial, norm,
   sampler, token-id and plane taps, a capture map's taps whose label is the Call's hint);
4. the tuple slots of occurrence k of a module bind the members of that module at invocation k that 1-3 left, in Program order: slots
   0..n-1 for n members, slot i binds the i-th; one member left, slot 0 binds it; otherwise none binds (a hook that returned y while
   the quant branch went unacquired never binds y to port 0).

An event no pass binds is an extra: a source label keeps itself as its identity (no manifest row has it), a tuple slot binds member
`""`, which no row has.  The coverage check (`binding.coverage_check`) then holds the map to the manifest's own spelling and names
every missing or doubly bound identity.  What stays by name, each with its reason: a source's declared label (the tap knows which
Call's port it reads) and the encodings above (identity format).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from verity_vllm.commit.binding import split_name

__all__ = ["Port", "Binder", "source_ports"]


@dataclass(frozen=True)
class Port:
    """What a source's label carries: port `port` of a Call of one of the Definitions `families`; `call_boundary`: the Value is one two
    Calls of the module body exchange, spelt `<module>/<fx node>`."""

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
    rid = declared.get(str(rid), f"r{rid}" if str(rid).isdigit() else str(rid))
    lo, hi = st if isinstance(st, list) else (st, st)
    return [int(s) + int((arrivals or {}).get(rid, 0)) for s in range(int(lo), int(hi) + 1)]


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
        self.promoted: dict[int, set[tuple[str, str, int]]] = defaultdict(set)
        for row in rows:
            r = {**row, **(row.get("identity") or {})}
            if r.get("family") == "weights" or r.get("op_path") == "weights" or not _on_rank(r, rank):
                continue
            key = (str(r["op_path"]), str(r.get("output_member", "out")), int(r.get("invocation", 0)))
            for st in _engine_steps(r, arrivals, declared or {}):
                self.rows[st].setdefault(key, (_family(r.get("spec")), str(r.get("port", key[1])), tuple(r.get("position") or ())))
                if r.get("promoted"):
                    self.promoted[st].add(key)
        self._by_op: dict[int, dict[str, list[tuple[str, str, int]]]] = {}

    @classmethod
    def of_manifest(cls, manifest: Mapping[str, Any], *, rank: int | None = None, ports: Mapping[str, Port] | None = None,
                    arrivals: Mapping[str, int] | None = None) -> "Binder":
        """The binder of a manifest document: its rows (of `rank`), its shared-instance aliases, its promoted observations and its
        declared requests."""
        from verity_vllm.query.manifest.format import promoted_observation

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
        keys = [k for o, ks in ops.items() if o.rsplit("/", 1)[0] == op and "/" in o for k in ks] if port.call_boundary else ops.get(op, [])
        return [k for k in keys if rows[k][0] in port.families and rows[k][1] == port.port]

    def bind(self, step: int, names: Sequence[str]) -> list[tuple[str, str, int, str]]:
        """The identity each event of `step` binds, as (op_path, member, invocation, how); how = `consumer-input` | `port` | `label` |
        `slot` | `unbound`."""
        step = int(step)
        rows = self.rows.get(step, {})
        out: list[tuple[str, str, int, str] | None] = [None] * len(names)
        taken = set(self.promoted.get(step, ()))
        seen: Counter = Counter()
        declared: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
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
                declared[(op, label)].append((i, k))
            elif (op, label, k) in rows and (op, label, k) not in taken:
                out[i] = (op, label, k, "label")
                taken.add((op, label, k))
            else:
                out[i] = (op, label, k, "unbound")
        for (op, label), events in declared.items():
            cands = [c for c in self._candidates(step, op, self.ports[label]) if c not in taken]
            ok = len(cands) == len(events)
            for (i, k), c in zip(events, cands if ok else [None] * len(events)):
                out[i] = (*c, "port") if c else (op, label, k, "unbound")
                if c:
                    taken.add(c)
        ops = self._ops(step)
        for (op, k), sl in slots.items():
            left = [c for c in ops.get(op, []) if c[2] == k and c not in taken]
            order = sorted(sl)
            pairs = list(zip(order, left)) if order == list(range(len(left))) else [(0, left[0])] if len(left) == 1 and 0 in sl else []
            for slot, c in pairs:
                out[sl[slot]] = (*c, "slot")
                taken.add(c)
            for slot in order:
                if out[sl[slot]] is None:
                    out[sl[slot]] = (op, "", k, "unbound")
        return out  # type: ignore[return-value]
