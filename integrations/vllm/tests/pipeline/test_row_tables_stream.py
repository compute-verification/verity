"""A row-table window (`test_row_tables`' window: ScaledMmS8Coordinate chunks whose `x` ports link shared token rows of
one table per engine step and linear) end to end through the proof system: its Spec declares every port's Value, the
window's `row-table` port a tree over the step roots (`service.ROOTS`) over its tables' hiding leaves
(`over=row_table.SCHEMA`), and one `Stream` over the window's template registers the window's record at the
challenger's counts, draws from the auditor's challenger's coins, and runs each outcome by the Lean verifier of record,
while the window's tables hold against the members' strings and the step roots serving recorded
(`tables.check_row_tables`)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from verity.ml.circuits.program import Program
from verity.ml.commitments import row_table as RT
from verity.core.protocols.sampled_proofs import service as S
from verity.core.protocols.sampled_proofs.one_stage import partition as P, registration as R
from verity.ml.windows import rows as CS, tables as TB

from .test_row_tables import DEFN, UNITS, committed

_SERVICE_TESTS = Path(__file__).resolve().parents[4] / "verity/core/protocols/sampled_proofs/tests"
if str(_SERVICE_TESTS) not in sys.path:
    sys.path.insert(0, str(_SERVICE_TESTS))

from test_challenger_coins import assert_only_honest_coins_accepted, coin_outcomes  # noqa: E402
from test_remote import signer  # noqa: E402,F401  (the challenger key's signer fixture)
from test_service import DELTA, Clock, _refused  # noqa: E402

TEMPLATE = P.template_id(DEFN)


def _spec(mine: CS.Binding, table: bool = True) -> S.Spec:
    ports = CS.expected_ports(mine)
    schemas = {m.port(p): m.schema(p) for m in mine.members for p in CS.port_leaves(m)}
    values = tuple(S.Value(p, S.ROOTS, "serve", over=RT.SCHEMA) if p == CS.TABLE_PORT else S.Value(p, schemas[p], "serve")
                   for p in sorted(ports) if table or p != CS.TABLE_PORT)
    public = (S.Public("population", "structure", "N, the draw's range"),
              S.Public(f"population:{TEMPLATE}", "structure", "the template's stratum"),
              S.Public("window", "structure", "provenance: served, on test prompts (step 0)"),
              S.Public("leaf_layer", "structure", "the members' public bodies' digest"),
              *(S.Public(f"{k}:{p}", "structure", f"{p}'s layout (step 0)") for p in sorted(ports) for k in ("leaves", "domain")))
    check = S.Check((P.program_digest(Program(DEFN)),), mine.query, proof_deadline=60.0)
    return S.Spec(values, S.Law("stratified:2"), check, public, b"row-table-stream-test")


@pytest.mark.slow
def test_a_row_table_window_runs_through_the_stream_on_the_challengers_coins(tmp_path, signer, monkeypatch):
    """The window's record, at row 101 + i for case i, registers on a stream whose Spec declares the `row-table` root
    as a tree over roots of row-seg leaves; drawn from the served challenger's coins after its digest, the honest draw's
    outcome is accepted by the Lean run and each of `coin_outcomes`' negatives is refused at its select (`coins`). Every
    case's tables hold against the members' strings and the live step roots. A Spec without the `row-table` Value
    refuses the record (`roots`): the tree over the step roots is one of the window's salted roots. About 70 s, the coins'
    outcomes each run by the Lean verifier: the only test that takes a row-table window's record through a Stream, so a
    `row-table` root the service can't register or the verifier refuses shows here and not on a served window."""
    c, _s, _got = committed(monkeypatch)
    mine = c.window
    publics = {m.key: c.public_body(k) for k, m in enumerate(mine.members)}
    ports, layer, live = CS.expected_ports(mine), CS.leaf_layer(publics), dict(c.segment_roots)
    assert {m.port(CS.LINKED) for m in mine.members if m.links} and ports[CS.TABLE_PORT]["leaves"] == len(mine.segments) == 2
    spec = _spec(mine)
    held = []

    def stream(spec: S.Spec, coins) -> S.Stream:
        return S.Stream(spec, [DEFN], budget=1, delta=DELTA, challenger="challenger-test", coins=coins, clock=Clock())

    def register(coins, i, window_root):
        rec = CS.registration(c, {"row": 101 + i}, registrar=R)
        roots = {r["port"]: r["root"] for r in rec["roots"]}
        held.append(TB.check_row_tables(mine, publics, roots, live))
        s = stream(spec, coins)
        return s, s.register(rec, ports, leaf_layer=layer, window_root=window_root, counts={TEMPLATE: UNITS})

    outcomes, _ = coin_outcomes(tmp_path, signer, monkeypatch, register, positions=64)
    assert held == [{}] * len(outcomes)
    assert_only_honest_coins_accepted(outcomes)
    honest, out, audit = outcomes["honest"]
    units = audit["draw"]["units"]
    assert out.accepted and honest._entries[0].read and (audit["draw"]["population"], len(units)) == (UNITS, 2)
    assert all(m.links for m in mine.members for u in units if m.base <= u < m.base + m.n)
    bare = stream(_spec(mine, table=False), lambda n: b"\x07" * n)
    rec = CS.registration(c, {"row": 100}, registrar=R)
    assert "roots" in _refused(lambda: bare.register(rec, ports, leaf_layer=layer, counts={TEMPLATE: UNITS}))
    with pytest.raises(ValueError, match="over"):
        S.Value(CS.TABLE_PORT, S.ROOTS, "serve")
