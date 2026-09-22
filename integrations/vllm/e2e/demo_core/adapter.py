"""[INTEGRATOR D1, E3's contract] The position adapter of the registered CORE program for the SHIPPED verifier.

``protocol_instance.json#program.builder`` names ``e2e.demo_core.circuit.build`` (E6's pin); E3's ``locator.position_adapter`` then calls
``position_adapter(program, locator, C)`` from the builder module and expects a dict with

* ``row_gates(loc) -> (canonical id -> (call lo, call hi, gates in row order) | None)``: every step-1 (boundary) leaf this prover commits,
  mapped to the producing call and the gates its words hold -- RU r's output row ``h_r`` (the stage call's return gates) and the three
  typed-input rows ``a, b, c`` (the Input region ``[0, input_gates)``).
* ``weight_word(g) -> GatePosition | None``: the committed position of an Input gate (this circuit has no weights; its prescribed inputs are
  the three Input collections, committed in ``boundary/<request 0>`` with origin ``input``).
* ``prefix(request_id) -> str``: the leaf-id prefix of the interior tree ``interior/<req>/<step>`` under E2's rule
  (``<prefix>call=<c>/chunk=<k>``).

``regions`` and ``calls`` are E3's generic defaults (one RU per root call, step = the call's ordinal): exactly this circuit's partition.
Everything here is a pure function of ``(N, R)`` -- the leaf ids are the record's deterministic ids (:mod:`e2e.demo_core.record`), so the
verifier derives every position from the pinned program alone and never reads a prover file for positions.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from . import circuit as C
from . import record as RC

__all__ = ["GatePosition", "interior_prefix", "params_of", "position_adapter", "row_table"]


@dataclass(frozen=True)
class _GatePosition:
    """Duck-typed stand-in for E3's ``locator.GatePosition`` (same fields, same order) when E3's module is not on the branch yet."""

    gate: int
    origin: str
    tree: str
    leaf: str
    shape: tuple[int, ...]
    flat: int
    width: int


def _gate_position_cls():
    try:
        from e2e.verifier_v3.locator import GatePosition  # E3 (D1)
    except ImportError:
        return _GatePosition
    return GatePosition


GatePosition = _gate_position_cls()


def params_of(C_: Mapping[str, Any] | None) -> tuple[int, int]:
    """``(N, R)`` from the instance's ``program`` section as E3 hands it over (``builder.params`` per E6, ``builder.kwargs`` per E3, or the identity)."""
    C_ = dict(C_ or {})
    b = C_.get("builder") if isinstance(C_.get("builder"), Mapping) else {}
    for src in (b.get("params"), b.get("kwargs"), C_.get("params"), C_):
        if isinstance(src, Mapping) and "N" in src and "R" in src:
            return int(src["N"]), int(src["R"])
    raise ValueError("program.builder carries neither params nor kwargs with N and R")


def interior_prefix(pid: str, rid: str) -> str:
    """The ``<prefix>`` of E2's interior leaf id for request ``rid``: ``p=<profile>/r=<request>/``."""
    return f"p={pid}/r={rid}/"


def row_table(part: C.Partition) -> dict[str, tuple[int, int, tuple[int, ...]]]:
    """Every step-1 leaf id -> (producing call lo, hi, gates in row order)."""
    pid = RC.profile_of(part)
    rid0 = RC.request_id(0)
    table: dict[str, tuple[int, int, tuple[int, ...]]] = {}
    for name in ("a", "b", "c"):
        table[RC._vid(pid, rid0, 0, f"input.{name}")] = (0, part.input_gates, tuple(part.inputs[name]))
    for ru in part.rus:
        table[RC._vid(pid, RC.request_id(ru.ru_index), ru.ru_index, "stage.add")] = (ru.lo, ru.hi, tuple(ru.returns))
    return table


def position_adapter(program, locator, C_: Mapping[str, Any] | None) -> dict[str, Any]:
    """E3's ``fn(program, locator, C)``: the adapter dict of ``DemoCore_v1{N,R}``."""
    N, R = params_of(C_)
    digest = None
    if isinstance(C_, Mapping):
        digest = (C_.get("builder") or {}).get("program_digest") if isinstance(C_.get("builder"), Mapping) else None
    part = C.partition(N, R, program, digest) if digest else C.partition(N, R)
    if part.gates != int(program.gates):
        raise ValueError(f"program has {program.gates} gates, DemoCore_v1{{N={N},R={R}}} has {part.gates}")
    rids = (C_ or {}).get("request_ids") if isinstance(C_, Mapping) else None
    if isinstance(rids, list) and len(rids) == 1 and RC.request_mode() == "per-ru":  # the epoch's one request names the RU ids: <request>-ru<r> (E9's ru_ids_of)
        RC.set_ru_base(str(rids[0]))
    pid = RC.profile_of(part)
    # step-1 rows keyed by the id's (l=<ru>, op=) segments: the ``r=`` segment (the request / RU id) is spelled by the record (``<base>-ru<r>`` or
    # the one request id) and the verifier's C does not carry it -- the row a leaf id names is the same under every spelling
    rows: dict[tuple[int, str], tuple[int, int, tuple[int, ...]]] = {}
    for name in ("a", "b", "c"):
        rows[(0, f"input.{name}")] = (0, part.input_gates, tuple(part.inputs[name]))
    for ru in part.rus:
        rows[(ru.ru_index, "stage.add")] = (ru.lo, ru.hi, tuple(ru.returns))
    inputs = {}
    for name in ("a", "b", "c"):
        gates = part.inputs[name]
        for i, g in enumerate(gates):
            inputs[int(g)] = (name, len(gates), i)

    def _learn_base(request_id: str) -> None:
        # per-ru: the request ids the oracle hands us ARE the RU ids (E9's ru_ids_of: ``<base>-ru<r>``); read the base off, never guess it
        if RC.request_mode() == "per-ru":
            head, sep, tail = str(request_id).rpartition("-ru")
            if sep and tail.isdigit() and head and head != RC.ru_base():
                RC.set_ru_base(head)

    def _row(cid: str):
        f = dict(seg.split("=", 1) for seg in str(cid).split("/") if "=" in seg)
        if f.get("p") != pid or "l" not in f or "op" not in f:
            return None
        try:
            return rows.get((int(f["l"]), f["op"]))
        except ValueError:
            return None

    def row_gates(_loc) -> Callable[[str], tuple[int, int, tuple[int, ...]] | None]:
        return _row

    def weight_word(g: int):
        hit = inputs.get(int(g))
        if hit is None:
            return None
        name, n, flat = hit
        rid0 = RC.request_id(0)
        return GatePosition(int(g), "input", f"boundary/{rid0}", RC._vid(pid, rid0, 0, f"input.{name}"), (n,), flat, 16)

    def prefix(rid: str) -> str:
        _learn_base(rid)
        return interior_prefix(pid, str(rid))

    def calls(loc):
        # the call id of the interior leaf id (E2 ``CallSpan.call``) is THIS program's spelling: the stage call's node NAME (``stage<r>``), the same
        # string ``record._commit_interior_e2`` commits -- given here so record and verifier agree by construction whatever E3's default (node
        # index today, node name earlier) is; E3: "a registered program that wants another spelling gives ``calls`` in its adapter"
        from e2e.verifier_v3.locator import calls_from_intervals

        return calls_from_intervals([(str(ru.name), int(ru.lo), int(ru.hi)) for ru in part.rus])

    out: dict[str, Any] = {"row_gates": row_gates, "weight_word": weight_word, "prefix": prefix, "calls": calls,
                           "request_ids": list(dict.fromkeys(RC.request_id(r) for r in range(R))), "request_mode": RC.request_mode(),
                           "step_of": RC.step_of, "profile_id": pid, "partition_digest": part.digest}
    if RC.request_mode() == "per-ru":
        # E3's D1 default (``locator.generic_regions``) spells ONE request whose steps are the root calls (``interior/<req>/<step>``). The per-RU
        # spelling (E9's ``ru_ids_of``: request ``<base>-ru<r>`` = RU r, one boundary tree ``boundary/<req>`` and one interior tree
        # ``interior/<req>/000000`` each) is "a program whose RUs are not its root calls" in E3's words: this adapter names the ONE region of
        # request ``<base>-ru<r>`` -- stage call r at step 0.  Nothing is guessed: an unknown request id has no region.
        def regions(loc, layout, request_id: str, *, committed_trees: Mapping[str, Any] | None = None):
            from e2e.verifier_v3.locator import Region, Regions

            _learn_base(request_id)
            ru = next((ru for ru in part.rus if RC.request_id(ru.ru_index) == str(request_id)), None)
            if ru is None:
                return Regions([])
            return Regions([Region(layout.tree_id(str(request_id), RC.step_of(ru.ru_index)), int(ru.lo), int(ru.hi), str(request_id), RC.step_of(ru.ru_index))])

        out["regions"] = regions
    return out


# --- v_in(C) = x for E9's step2_check_registered(input_check=...) ---------------------------------------------------------------------------


def input_check(record_dir: Any, params: Mapping[str, Any] | None = None, **kw: Any) -> Callable[[str, Mapping[str, Any], Any], list[str]]:
    """[INTEGRATOR D20] the builder module's ``input_check`` for E9's ``driver.py`` / ``reverify.py`` registered-program path: a FACTORY
    ``input_check(record_dir, params) -> check(rid, {ru_id: c0 boundary tree record}, declared) -> problems[]`` (``params`` = the instance's
    ``program.identity.params`` = ``{"N", "R"}``; ``N=``/``R=`` keywords also accepted).  ``check`` is ``make_input_check``'s (below)."""
    p = dict(params or {})
    p.update({k: v for k, v in kw.items() if k in ("N", "R")})
    return make_input_check(record_dir, int(p["N"]), int(p["R"]))


def make_input_check(record_dir: Any, N: int, R: int) -> Callable[[str, Mapping[str, Any], Any], list[str]]:
    """[E9, D1] ``input_check(rid, {ru_id: c0 boundary tree record}, declared) -> problems[]`` for ``steps.step2_check_registered``: the typed input
    x = (a, b, c) of DemoCore_v1 is committed as the three ``input``-origin leaves of RU 0's boundary tree (``boundary/<request>-ru0``).  The check
    reads the record's leaf files (``record_dir/adapter/selective/boundary/<ru_id>/{index.json,leaf-<i>.bin}``), recomputes every leaf hash and the
    tree root with the engine's ``MerkleTree`` and requires it to equal the committed record (nothing is trusted from the prover's index), then
    digests the 3N input words in Input-gate order with the shipped ``protocol_instance.declared_input_digest`` and compares with
    ``declared.prompt_sha256`` / ``declared.prompt_len`` (the auditor's x by digest, pinned before the first commitment).  Fail-closed: any missing
    file, root mismatch or count mismatch is a problem string; an empty list means v_in(C) = x."""
    import json
    from pathlib import Path

    import numpy as np

    from e2e import protocol_instance as pi
    from verity_vllm import hashing
    from verity_vllm.commit import MerkleTree

    sel = Path(record_dir) / "adapter" / "selective"
    prog, d = C.build(int(N), int(R))
    part = C.partition(int(N), int(R), prog, d)
    order = [int(g) for name in ("a", "b", "c") for g in part.inputs[name]]

    def check(rid: str, boundary_records: Mapping[str, Any], declared: Any) -> list[str]:
        problems: list[str] = []
        ru0 = next((u for u in boundary_records if u.endswith("-ru0")), None) or next(iter(boundary_records), None)
        rec = boundary_records.get(ru0) if ru0 else None
        if not rec:
            return [f"no boundary tree record for RU 0 of request {rid} (the input leaves live there)"]
        d0 = sel / "boundary" / str(ru0)
        try:
            idx = json.loads((d0 / "index.json").read_text())
        except OSError as e:
            return [f"{d0}/index.json unreadable: {e}"]
        leaves: dict[str, bytes] = {}
        words: dict[int, int] = {}
        for e in idx.get("entries") or []:
            data = (d0 / f"leaf-{e['index']}.bin").read_bytes()
            leaves[e["canonical_id"]] = hashing.leaf_hash(e["canonical_id"], e["dtype"], tuple(e["shape"]), data)
            if e.get("origin") == "input":
                for g, w in zip(range(e["gates"][0], e["gates"][1]), np.frombuffer(data, dtype="<u2")):
                    words[int(g)] = int(w)
        t = MerkleTree(leaves, f"boundary/{ru0}")
        if t.root.hex() != rec.get("root") or t.leaf_count != int(rec.get("leaf_count", -1)) or t.ids_digest.hex() != rec.get("ids_digest"):
            problems.append(f"boundary/{ru0}: recomputed root/leaf_count/ids_digest != the committed record (the input leaves are not authenticated)")
            return problems
        x = [words.get(g) for g in order]
        if any(w is None for w in x):
            problems.append(f"boundary/{ru0}: {sum(w is None for w in x)} of the {len(order)} Input gates have no committed input leaf word")
            return problems
        digest = pi.declared_input_digest([int(w) for w in x])
        want = getattr(declared, "prompt_sha256", None) if not isinstance(declared, Mapping) else declared.get("prompt_sha256")
        want_len = getattr(declared, "prompt_len", None) if not isinstance(declared, Mapping) else declared.get("prompt_len")
        if digest != want:
            problems.append(f"v_in(C) != declared x: sha256 of the {len(order)} committed input words {digest[:16]} != epoch.inputs.prompt_sha256 {str(want)[:16]}")
        if want_len is not None and int(want_len) != len(order):
            problems.append(f"declared prompt_len {want_len} != {len(order)} Input gates of DemoCore_v1{{N={N},R={R}}}")
        return problems

    return check
