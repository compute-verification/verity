"""One naming scheme: every committed value's identity is a row of the required manifest, and the runtime events that carry the
value are bound to that row by structure.  Nothing reconciles a tap's spelling with the manifest's.

IDENTITY.  The manifest of record (`verity_vllm.query.required`) spells each required Value as (op_path, output_member, invocation) at
an engine step on a rank, from the Program and its correspondence alone: op_path is the producing Call's owning module (a collective
site's enclosing module at invocation k, `<module>/<fx node>` for a Value two Calls of one module body exchange, and for each fx node's
Value when two fx nodes of the module produce a member of one name at the step); output_member is the Definition's port name when it
returns several (`<hint>.<port>` when the Build recorded a hint, which names the Call), else the recorded hint, else `out`.  The
protocol's encodings are identity format, each one constant the manifest and the committing tap share:
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
   boundary, of `<module>/<fx node>` or of the module itself) that port `port` of one of those Definitions produces: DeepGEMM's quant
   branch `x_q` / `x_s` is its ports 0 / 1, the attention taps the DSA indexer's ports and the sparse MLA and W_UK_T absorb boundaries.
   The occurrences of every label declaring one Port at one module bind its candidates in (invocation, Program) order when there are as
   many of each, else none: the mHC taps' attention and FFN post_pre are one Definition's port twice per layer, in commit order;
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

from verity_vllm.query.manifest.format import promoted_observation

__all__ = ["HIDDEN_FAMILIES", "Port", "Binder", "coverage_check", "source_ports", "split_name"]

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
            ok = len(cands) == len(events)
            for (i, k, label), c in zip(events, cands if ok else [None] * len(events)):
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


def coverage_check(manifest: dict[str, Any], layouts: dict[int | str, list[dict[str, Any]]], *, request_id: str | int | None = None,
                   rank: int = 0, max_report: int = 12, ports: Any = None) -> dict[str, Any]:
    """Every manifest identity of (`request_id`, `rank`) must have a committed tensor bound to the same (step, op_path, output_member,
    invocation) whose element range CONTAINS the manifest's; `layouts` = {step: [{"name", "numel"|"shape", ...} in layout order]}, each
    name bound by the manifest's binder (`Binder`, with the committing sources' declared `ports`).  Reports missing
    identities (by family / step), and the committed extras (allowed).  Identity coverage only -- value correspondence is bind's separate
    check.  A kernel-stream tensor (`HIDDEN_FAMILIES`) the binder leaves unbound is still committed under its own name, as the binding
    map keeps it: a B>1 batch-level hidden row (request `*`) names the padded whole-step geometry, and is covered op-level by the
    per-sequence planes (`s<i>/...`) or the whole launch the tap committed under another spelling."""
    if manifest.get("complete") is False:                                        # [MAN-02] an incomplete manifest never yields coverage
        um = manifest.get("unmodelled") or {}
        n = len(manifest.get("identities") or [])
        return {"ok": False, "error": "manifest incomplete: unmodelled Program families " + "; ".join(f"{k} x{v}" for k, v in sorted(um.items())),
                "manifest_digest": manifest.get("manifest_digest"), "checked": n, "missing_n": n, "missing_by_family": {"(unmodelled)": n},
                "checked_by_family": {}, "families_missing_entirely": sorted(um), "steps_with_missing": [], "missing": [],
                "committed_unmatched_n": 0, "committed_unmatched": [], "committed_extras_n": 0, "duplicate_names_n": 0, "steps_committed": [], "unmodelled": um}
    binder = Binder.of_manifest(manifest, rank=rank, ports=ports)
    have: dict[tuple, list[int]] = {}
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
            **({"promoted_observed_as": dict(sorted((k, list(v)) for k, v in binder.rebind.items()))} if binder.rebind else {})}
