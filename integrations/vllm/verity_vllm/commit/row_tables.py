"""The row tables of a serving-rows window (`commit.serving_rows.RowTable`), as serving commits them step by step and PoUW binds
``pouw.roots`` to them.

One row table per int8 quantize output: the token rows of one linear at one engine step, one ``hm96-sha512/row-seg/v1`` leaf each
(``verity.ml.commitments.row_table``).  At the serving loop's step boundary, once every quantize output of the step is written,
``step_roots`` gives the step's root: the tree over its tables' roots, 64 bytes, which the loop records with the challenger before
that step's coin (``LiveRecord.record``) and which that step's PoUW noise reads (``window.round_key``'s ``root``).  The window's
``row-table`` port is the tree over the step roots, and the window's commit holds its step roots to the recorded ones.

``find`` gives a (step, layer, linear)'s table: its step's root (what the verifier fills ``pouw.roots`` with), its index in the
step, the domain id and leaf count an opening verifies under, and each token row's leaf index (``Table.rank``).  The verifier's
tables come from its own window (``pipeline.serving_rows.population`` over the Build's Programs and the arrivals, the partition
file's ``row_tables: true``, the live coin commitment's root as its context), the step roots serving recorded, and the
registration ``registration.check`` accepted, whose ``row-table`` root must be the tree over them (``of_registration``;
``check_row_tables`` ties the linked chunks' public ``x`` strings to the same roots); the
prover's from the window directory serving wrote (``of_dir``), with ``opening`` (the table's root opened in its step's tree, and a
set of ranks' strings with their multiproof against it: a ``row_table.StepOpening``, from ``window.json`` and ``t<k>-public.bin``)
and ``witness`` (their rows and salts, from ``t<k>-private.bin``: what a tile's proof takes privately).  ``verify`` checks an
opening against its table's step root, with the rows and salts in the clear when given.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from verity.ml.commitments import row_table as RT
from verity.ml.commitments.merkle import CommitmentDomain

from . import serving_rows as CS

__all__ = ["StepRoots", "Table", "check_row_tables", "find", "of_dir", "of_registration", "opening", "step_roots", "verify", "witness"]


@dataclass(frozen=True)
class StepRoots:
    """One engine step's row tables as serving committed them: ``root``, what the loop records before the step's coin, and each
    table's ``(op_path, root)`` in the step's order."""
    engine_step: int
    root: bytes
    tables: tuple[tuple[str, bytes], ...]


def step_roots(context: bytes, engine_step: int, outputs: Mapping[str, tuple[Sequence[tuple[str, int]], Any, Any]], key: bytes, *,
               workers: int = 1, device: bool = False) -> Any:
    """Engine step ``engine_step``'s root, from its int8 quantize outputs once all are written: ``outputs[op_path] = (rows, codes,
    scales)``, ``rows[b]`` batch row ``b``'s ``(request, token row)``, ``codes`` the (B, K) s8 codes and ``scales`` the (B,) f32
    scales (host copies).  Its tables go in ``serving_rows.table_key`` order, each table's rows by (request, token row), under
    ``context`` (64 bytes: the live coin commitment's root) and salted by ``serving_rows.step_salts`` under ``key``, the run's row
    key (32 bytes), which the window's commit takes again as ``row_salt_key``.  With ``device``, ``codes`` and ``scales`` are CUDA
    tensors and the step hashes on the GPU behind the current stream (``row_tables_device.launch``): this returns its
    ``Pending``, whose ``result()`` is this ``StepRoots``."""
    if device:
        from . import row_tables_device
        return row_tables_device.launch(context, engine_step, outputs, key)
    order = sorted(outputs, key=CS.table_key)
    rows, ys, domains = {}, {}, {}
    ss = CS.step_salts(key, engine_step, sum(len(outputs[op][0]) for op in order))
    at = 0
    for i, op in enumerate(order):
        ids, codes, scales = outputs[op]
        x = np.ascontiguousarray(np.asarray(codes, dtype=np.int8))
        s = np.ascontiguousarray(np.asarray(scales, dtype="<f4").reshape(-1))
        if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] % 2 or s.shape != (x.shape[0],) or len(ids) != x.shape[0]:
            raise ValueError(f"{op}: codes {x.shape}, scales {s.shape} and {len(ids)} row ids are not B >= 1 rows of an even K")
        keys = [(str(r), int(t)) for r, t in ids]
        perm = sorted(range(len(keys)), key=keys.__getitem__)
        if len(set(keys)) != len(keys):
            raise ValueError(f"{op}: a (request, token row) twice in one step")
        words = np.concatenate([x.view(np.uint8), s.view("<u4").view(np.uint8).reshape(-1, 4)], axis=1)[perm]
        rows[op] = np.ascontiguousarray(words).view("<u2")
        ys[op], at = ss[at:at + len(keys) * CS.SALT_BYTES], at + len(keys) * CS.SALT_BYTES
        domains[op] = RT.table_domain(context, engine_step, i, len(keys))
    roots, _bc = CS.commit_ports(domains, rows, {}, ys, workers=workers, schemas=dict.fromkeys(order, CS.TABLE_SCHEMA))
    tables = tuple((op, roots[op]) for op in order)
    return StepRoots(engine_step, RT.step_tree(context, engine_step, [r for _, r in tables]).root, tables)


@dataclass(frozen=True)
class Table:
    """A row table in its step: ``spec`` (its engine step, op_path, words and each rank's token row), its serve-time ``domain``,
    ``k`` its index among the window's row tables (its files ``t<k>-*``), ``index`` its place among its step's ``tables`` tables,
    the window's ``context`` and its step's root ``step_root``."""
    spec: CS.RowTable
    domain: CommitmentDomain
    k: int
    index: int
    tables: int
    context: bytes
    step_root: bytes

    @property
    def port(self) -> str:
        return self.spec.port

    @property
    def leaves(self) -> int:
        return self.spec.n

    @property
    def domain_id(self) -> bytes:
        return self.domain.domain_id

    @property
    def engine_step(self) -> int:
        return self.spec.engine_step

    @property
    def layer(self) -> int | None:
        return self.spec.layer

    @property
    def op_path(self) -> str:
        return self.spec.op_path

    def rank(self, request: str, token_row: int) -> int:
        """The leaf index of ``request``'s token row ``token_row``."""
        return self.spec.rank(request, token_row)


def of_registration(window: CS.Window, rec: Mapping[str, Any], step_roots: Mapping[int, bytes]) -> list[Table]:
    """The verifier's tables: each row table of its own ``window`` under its step's root as serving recorded it before the step's
    coin (``step_roots``, by engine step); refused unless the registration ``rec``'s ``row-table`` port holds the window's step
    count and domain (``registration.check``'s R3) and its root is the tree over those step roots."""
    r = {x["port"]: x for x in rec["roots"]}.get(CS.TABLE_PORT)
    dom = window.tables_domain
    if r is None or r["leaves"] != len(window.steps) or r["domain"] != dom.domain_id.hex():
        raise ValueError(f"the registration does not hold the window's {CS.TABLE_PORT} port ({len(window.steps)} steps, its served domain)")
    recorded = [step_roots.get(s) for s, _ in window.steps]
    if any(x is None for x in recorded) or RT.RootTree(dom, recorded).root.hex() != r["root"]:
        raise ValueError(f"the registration's {CS.TABLE_PORT} root is not the tree over the step roots serving recorded")
    ctx, count = bytes.fromhex(window.context), {s: len(ks) for s, ks in window.steps}
    return [Table(t, window.table_domain(k), k, window.places[k], count[t.engine_step], ctx, step_roots[t.engine_step])
            for k, t in enumerate(window.row_tables)]


def check_row_tables(window: CS.Window, publics: Mapping[str, bytes], roots: Mapping[str, str | bytes],
                     step_roots: Mapping[int, bytes] | None = None) -> dict[str, str]:
    """The verifier's tie between the row tables, their linked members and serving's step roots: each step's root recomputed
    from the linked members' public ``x`` strings (``publics[member]``, their public bodies, which the leaf layer binds), two
    members' strings for one row the same, and the ``row-table`` port's registered root (``roots``) the tree over them; with
    ``step_roots`` (what serving recorded before each step's coin, by engine step), each step's recomputed root is the recorded
    one and the registered root the tree over the recorded ones.  ``{key: why}`` keyed by a member, a table's port, a step
    (``row-table/<engine step>``) or the port; empty when everything holds."""
    strings: list[list[bytes | None]] = [[None] * t.n for t in window.row_tables]
    bad: dict[str, str] = {}
    for m in window.members:
        if not m.links:
            continue
        body = publics.get(m.key, b"")
        if len(body) < CS.BC_BYTES * len(m.links):
            bad[m.key] = f"its public body is {len(body)} bytes, short of its {len(m.links)} {CS.LINKED} strings"
            continue
        for j, (t, r) in enumerate(m.links):
            s = body[j * CS.BC_BYTES:(j + 1) * CS.BC_BYTES]
            if strings[t][r] is not None and strings[t][r] != s:
                bad.setdefault(window.row_tables[t].port, f"row {r}: two members' strings differ ({m.key})")
            strings[t][r] = s
    table_roots: dict[int, bytes] = {}
    for k, (t, ss) in enumerate(zip(window.row_tables, strings, strict=True)):
        if t.port in bad:
            continue
        if any(s is None for s in ss):
            bad[t.port] = "a row no member's"
        else:
            table_roots[k] = RT.RowTable(window.table_domain(k), ss).root
    if not window.row_tables:
        return bad
    want = roots.get(CS.TABLE_PORT)
    want = bytes.fromhex(want) if isinstance(want, str) else want
    steps = {s: window.step_root(s, [table_roots[k] for k in ks]) for s, ks in window.steps if all(k in table_roots for k in ks)}
    if step_roots is not None:
        for s, root in steps.items():
            if root != step_roots.get(s):
                bad[f"{CS.TABLE_PORT}/{s}"] = "the members' strings are not the step root serving recorded"
        recorded = [step_roots.get(s) for s, _ in window.steps]
        if any(r is None for r in recorded) or RT.RootTree(window.tables_domain, recorded).root != want:
            bad[CS.TABLE_PORT] = "its root is not the tree over the step roots serving recorded"
    elif len(steps) < len(window.steps):
        bad[CS.TABLE_PORT] = "a step whose tables do not hold"
    elif RT.RootTree(window.tables_domain, list(steps.values())).root != want:
        bad[CS.TABLE_PORT] = "the members' strings are not its root's rows"
    return bad


def _window(window_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(window_dir) / "window.json").read_text())


def of_dir(window_dir: str | Path) -> list[Table]:
    """The prover's tables, from the window serving wrote (``window.json``'s ``row_tables`` and ``row_table_steps``)."""
    win, out = _window(window_dir), []
    if not win.get("row_tables"):
        return out
    st = win["row_table_steps"]
    ctx, steps = bytes.fromhex(st["context"]), {int(e["engine_step"]): e for e in st["steps"]}
    for e in win["row_tables"]:
        spec = CS.RowTable(int(e["engine_step"]), e["op_path"], e["template"], int(e["words"]), tuple((str(a), int(b), int(c)) for a, b, c in e["rows"]))
        s, i = steps[spec.engine_step], int(e["step_index"])
        dom = RT.table_domain(ctx, spec.engine_step, i, spec.n)
        if dom.domain_id.hex() != e["domain"] or spec.port != e["port"] or s["tables"][i] != int(e["k"]):
            raise ValueError(f"row table {e['k']}: not table {i} of step {spec.engine_step} under its domain {e['domain'][:16]}")
        out.append(Table(spec, dom, int(e["k"]), i, len(s["tables"]), ctx, bytes.fromhex(s["root"])))
    return out


def find(tables: Sequence[Table], step: int, layer: int, linear: str) -> Table:
    """The table of engine step ``step``'s ``linear`` in layer ``layer``: ``linear`` its op_path, or the op_path's last dotted parts
    (``qkv_proj``, ``self_attn.qkv_proj``)."""
    hits = [t for t in tables if t.engine_step == int(step) and t.layer == int(layer)
            and (t.op_path == linear or t.op_path.endswith("." + linear))]
    if len(hits) != 1:
        raise LookupError(f"{len(hits)} row tables of step {step}, layer {layer}, linear {linear!r}"
                          + (f": {[t.op_path for t in hits]}" if hits else ""))
    return hits[0]


def _strings(window_dir: str | Path, t: Table) -> list[bytes]:
    pub = (Path(window_dir) / f"t{t.k}-public.bin").read_bytes()
    if len(pub) != CS.BC_BYTES * t.leaves:
        raise ValueError(f"t{t.k}-public.bin is {len(pub)} bytes, not {t.leaves} strings")
    return [pub[i * CS.BC_BYTES:(i + 1) * CS.BC_BYTES] for i in range(t.leaves)]


def opening(window_dir: str | Path, t: Table, ranks: Iterable[int]) -> RT.StepOpening:
    """The prover's opening of ``t``'s rows at ``ranks``: ``t``'s root opened in its step's tree, and the rows' strings with their
    multiproof against it; refused when the window's strings are not ``t``'s rows or its step's tables are not its step root."""
    tab = RT.RowTable(t.domain, _strings(window_dir, t))
    roots = [bytes.fromhex(e["root"]) for e in sorted(_window(window_dir)["row_tables"], key=lambda e: int(e["k"]))
             if int(e["engine_step"]) == t.engine_step]
    if len(roots) != t.tables or roots[t.index] != tab.root:
        raise ValueError(f"t{t.k}-public.bin is not {t.port}'s rows")
    tree = RT.step_tree(t.context, t.engine_step, roots)
    if tree.root != t.step_root:
        raise ValueError(f"step {t.engine_step}'s tables are not its root")
    return RT.StepOpening(tree.open([t.index]), tab.open(ranks))


def witness(window_dir: str | Path, t: Table, ranks: Iterable[int]) -> dict[int, tuple[bytes, int, bytes]]:
    """The prover's rows and salts of ``t`` at ``ranks``: ``{rank: (row, n_bits, salt)}``, ``row`` the s8 codes then the f32 scale
    (``row_table.s8_row``)."""
    priv = (Path(window_dir) / f"t{t.k}-private.bin").read_bytes()
    rb = 2 * t.spec.words
    if len(priv) != t.leaves * (rb + CS.SALT_BYTES):
        raise ValueError(f"t{t.k}-private.bin is {len(priv)} bytes, not {t.leaves} rows of {rb} bytes and their salts")
    base = t.leaves * rb
    return {r: (priv[r * rb:(r + 1) * rb], 8 * rb, priv[base + r * CS.SALT_BYTES:base + (r + 1) * CS.SALT_BYTES]) for r in sorted(set(ranks))}


def verify(t: Table, o: RT.StepOpening, rows: Mapping[int, tuple[bytes, int, bytes]] | None = None) -> bool:
    """Whether ``o`` opens ``t``'s rows at its ranks through ``t``'s step root (``row_table.verify_step``), and, with ``rows``, they
    are its rows and salts; ``o.root`` is then ``t``'s root."""
    return o.index == t.index and RT.verify_step(t.step_root, t.context, t.engine_step, t.tables, t.leaves, o, rows)
