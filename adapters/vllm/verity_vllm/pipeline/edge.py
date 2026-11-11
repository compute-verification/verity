"""The edge between two Commit windows, on the advice's side: which requests window w leaves for window w + 1 to read, and
where each carried request's history is among them.

A request window w leaves with outputs still to keep (`window_advice.close`'s ``must`` entries) runs on in w + 1 from the state
w left it at, which `window_advice.follows` holds w + 1's advice to. Its history, the rows w + 1 reads and doesn't compute, is
window w's: w returns the ports of `VALUES` after its requests' outputs, and w + 1's Program takes them as registered
parameters (the one-stage registration's ``outputs``, R7), so every read of it opens under w's root, domain and hiding leaves.
The Program's side, each field's type and the reads of it, is `program.workload.Edge`'s, built by `predict.segment.edge_of`.

- **Which requests.** A window's edge members are its `close` entries that must carry and have computed a row, in close order
  (`members`): its carried requests in the order w − 1 left them, then the ones it admitted, by first row. A request that only
  may carry (its last kept output is in w's last step) has no Program in w + 1, and one that never ran has no history.
- **Which fields.** Each port is a Tuple with one field per member holding it, in member order (`fields`). `edge.kv` holds
  every member's latent rows [0, computed) of every layer (a request Program's `prefix`). `edge.ik` and `edge.iks` hold the
  indexer's keys and scales (`prefix_ik`, `prefix_iks`) of each member whose record's last row reads past the model's
  INDEX_TOPK keys (`prompt_len + output_len - 1` > INDEX_TOPK, `predict.segment.edge_groups`), the same in every window; a
  model without an indexer has neither port. `edge.last` holds the last kept output (`last_token`) of each member past its
  prompt.
- **The verifier's side.** `carried(last, advice)` refuses ``advice`` unless it follows ``last``, the window before it
  (`window_advice.NotFollowing`, by name), and gives each carried request that reads history its field in each of ``last``'s
  ports (`Slice`).
- **The registration.** Each port's value is rows of its `row_words` leaves: one position of one layer of its history
  group, and one token of `edge.last`. Window w commits each out-port it returns as a registered value under its own
  program and open root (`commit`, core's ``registered.Registered``) and names it among its roots (`roots`), with the
  domain its verifier derives from the Program alone (`out_ports`). Window w + 1 registers each in-port under that root and
  domain (`entries`), and its verifier's parameters give the value as a history port, ``{"port": name}`` (`history`): port
  P of the log's last record, the one R6 holds ``previous`` to, so w + 1's history is held to w's root, domain and leaf
  count (``R7-output``, ``R7-held``, ``R7-domain``) and to no record a prover names.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

from verity.protocols.spot.verifier.one_stage import registered as REG
from verity.circuits.codec import program_digest
from verity.circuits.program import Program
from verity.circuits.types import Array, RecordType, Tuple as TupleT, Type, field_types

from . import window_advice as WA

__all__ = ["VALUES", "Slice", "Value", "carried", "commit", "entries", "fields", "history", "members", "out_ports", "roots",
           "row_words"]


class Value(NamedTuple):
    """One edge port: its name (a port of window w's record and a registered value of window w + 1's), the request Program
    Input its fields fill, whether a member holds a field only once it is past its prompt, and whether only if its rows read
    past INDEX_TOPK keys."""

    name: str
    history: str
    past_prompt: bool
    past_topk: bool = False


VALUES = (Value("edge.kv", "prefix", False), Value("edge.ik", "prefix_ik", False, True), Value("edge.iks", "prefix_iks", False, True),
          Value("edge.last", "last_token", True))


class Slice(NamedTuple):
    """A carried request's history in the last window's edge: its index in the advice that carries it and its id, its
    history's positions (its carried ``computed``, its Program's C), and its field in each port that holds one."""

    request: int
    id: str
    computed: int
    fields: Mapping[str, int]


def members(advice: WA.Advice) -> tuple[WA.Close, ...]:
    """The requests a window's edge holds, in its order: its `close`'s entries that must carry and have computed a row."""
    return tuple(c for c in WA.close(advice) if c.must and c.state.computed)


def fields(advice: WA.Advice, topk: int | None = None) -> dict[str, tuple[int, ...]]:
    """Each port's members holding a field, as indices in ``advice``, in field order. ``topk``: the model's INDEX_TOPK, None for a
    model without an indexer (no `edge.ik` or `edge.iks`)."""
    held = members(advice)

    def holds(v: Value, c: WA.Close) -> bool:
        q = advice.requests[c.request]
        return (not v.past_prompt or c.state.computed >= q.prompt_len) and (not v.past_topk or q.prompt_len + q.output_len - 1 > topk)
    return {v.name: tuple(c.request for c in held if holds(v, c)) for v in VALUES if topk is not None or not v.past_topk}


def carried(last: WA.Advice, advice: WA.Advice, topk: int | None = None) -> tuple[Slice, ...]:
    """The verifier's: ``advice`` held to ``last``, the window before it (`window_advice.follows`, refusing by name), and each
    carried request's fields in ``last``'s edge (`fields` at ``topk``), in ``advice``'s order. A carried request that isn't one
    of ``last``'s members reads no history and has no slice."""
    pairs = WA.follows(last, advice)
    held = fields(last, topk)
    at = set(held[VALUES[0].name])
    return tuple(Slice(k, advice.requests[k].id, advice.carried[k].computed,
                       {p: f.index(i) for p, f in held.items() if i in f}) for k, i in pairs if i in at)


def row_words(net: Any) -> dict[str, int]:
    """Each port's row width in leaves, from the decoder ``net`` (`predict.build.decoder`): one position of one layer of its
    history group as ``net.prefix_params`` lays it out (the dims past the position axis), and one token of `edge.last`."""
    one, two = dict(net.prefix_params(1)), dict(net.prefix_params(2))
    out = {}
    for v in VALUES:
        if v.history == "last_token":
            out[v.name] = 1
        elif v.history in one:
            a, b = _dims(one[v.history]), _dims(two[v.history])
            k = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
            out[v.name] = math.prod(a[k + 1:])
    return out


def _dims(t: Type) -> tuple[int, ...]:
    shape = []
    while isinstance(t, Array):
        shape.append(int(t.n))
        t = t.elem
    return tuple(shape)


def _shape(name: str, t: Type, words: int) -> tuple[int, int]:
    """Port ``name`` of type ``t`` as rows of ``words`` leaves: (rows, word_bits); ValueError where it isn't."""
    w = t.homogeneous_width()
    if w is None or not 0 < w <= max(REG.WIDTHS) or type(words) is not int or words < 1 or t.leaves % words:
        raise ValueError(f"{name}: {t.leaves} leaves of widths {sorted(t.width_set())} are no rows of {words!r} leaves of one width")
    return t.leaves // words, 16 * -(-w // 16)


def _returned(program: Program, ports: Sequence[str]) -> dict[str, Type]:
    """The out-ports' types: the last fields of the Program's returned Tuple, one per port in `VALUES` order."""
    if list(ports) != [v.name for v in VALUES if v.name in set(ports)]:
        raise ValueError(f"edge ports {list(ports)} aren't some of {[v.name for v in VALUES]}, in order")
    ret = program.fn.signature()[1]
    fs = field_types(ret) if isinstance(ret, (TupleT, RecordType)) else ()
    if len(fs) < len(ports):
        raise ValueError(f"the Program returns {len(fs)} fields, fewer than its {len(ports)} edge ports")
    return dict(zip(ports, fs[len(fs) - len(ports):]))


def out_ports(program: Program, ports: Sequence[str], words: Mapping[str, int], window: str) -> dict[str, dict[str, Any]]:
    """Window w's verifier's ports for the edge ``ports`` its Program returns (`fields`' ports some member holds), as R3 holds
    its record's roots to them: each one's rows of `row_words` leaves and the registered domain of its value under the
    Program and the window's open root ``window`` (core's ``registered.domain``), which window w + 1's history takes."""
    prog = program_digest(program)
    out = {}
    for p, t in _returned(program, ports).items():
        n, bits = _shape(p, t, words[p])
        out[p] = {"leaves": n, "domain": REG.domain(prog, p, words[p], n, bits, window).domain_id.hex()}
    return out


def commit(program: Program, values: Mapping[str, Sequence[int]], words: Mapping[str, int], window: str,
           salts: Mapping[str, Sequence[bytes]] | None = None) -> dict[str, REG.Registered]:
    """Window w's prover's commitment of each edge out-port it returns: ``values`` each port's leaves (its returned field's,
    in order), in rows of its ``words``, committed as a registered value under the Program and the open root ``window``
    (fresh salts unless ``salts`` gives each port's), so its root and domain are the ones `out_ports` derives."""
    prog, types = program_digest(program), _returned(program, list(values))
    out = {}
    for p, t in types.items():
        n, bits = _shape(p, t, words[p])
        leaves = [int(x) for x in values[p]]
        if len(leaves) != t.leaves:
            raise ValueError(f"{p}: {len(leaves)} leaves, not its type's {t.leaves}")
        rows = [leaves[i * words[p]:(i + 1) * words[p]] for i in range(n)]
        out[p] = REG.Registered.commit(prog, p, rows, None if salts is None else salts[p], word_bits=bits, window=window)
    return out


def roots(committed: Mapping[str, REG.Registered]) -> list[dict[str, Any]]:
    """Window w's record's roots for its edge out-ports (`commit`): each port's root, leaf count and domain."""
    return [{"port": p, "root": r.root.hex(), "leaves": r.leaves, "domain": r.domain.domain_id.hex()} for p, r in committed.items()]


def entries(committed: Mapping[str, REG.Registered]) -> list[dict[str, Any]]:
    """Window w + 1's registered entries for the history it reads: each of window w's edge out-ports (`commit`) under its
    own name, root and domain, as R7 holds a history port to the previous record's port."""
    return [r.entry() for r in committed.values()]


def history(ports: Sequence[str], words: Mapping[str, int]) -> dict[str, Any]:
    """Window w + 1's verifier's parameters for the edge ``ports`` its Program takes as registered parameters
    (`program.workload.WorkloadProgram.ports`): each value's row width (``registered``) and its history port,
    ``{"port": name}``, port ``name`` of the log's last record (``outputs``); merged into the one-stage parameters."""
    return {"registered": {p: int(words[p]) for p in ports}, "outputs": {p: {"port": p} for p in ports}}
