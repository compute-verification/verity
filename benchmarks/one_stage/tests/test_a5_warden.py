"""a5_warden's window on a warden run driven tick by tick (`test_warden_serve._run`), before anything is lowered or proved
(`a5_warden.main` does, through `a5.audit`, on a node): each transfer-gate chunk's registered port's rows, the warden's
bucket rows committed under its salts, open the warden's signed root at the rows the verifier's own reads name, each
unit's read (q, k) the Program's units' (`warden_rows.Units`, from public structure alone) placed by the wiring the warden
signed on connection q's own cell, and the header names the units' value and the same rows; the registration passes the
verifier's check with every transfer-gate unit proved (work 0, floor all of them); the window's provenance is the
verifier's read of the warden's registrations; the prover's own root under the warden's certificate, provenance naming
another root, a foreign, a duplicate, a padding-slot and a past-the-cells read, and a statement made for another units'
wiring (a pad in a data cell's place, its rows under the Program's value, two units swapped) are refused by name; and the
altered window's unit 0 doesn't open."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from verity.core.protocols.certifier import warden_rows as WR
from verity.core.protocols.certifier.active import HEADER
from verity.core.protocols.sampled_proofs.one_stage import draw as D, verifier as V
from verity.ml.windows import rows as SR

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "network_traces" / "tests"))
import a5_warden as AW  # noqa: E402
import test_warden_serve as tws  # noqa: E402

N, REST = 8, 4


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("a5-warden")
    out, key = tmp / "run", tmp / "key"
    out.mkdir()
    requests = [tws.ws.http_request({"prompt": "x" * n}, "h:1") for n in (300, 10)]
    tws._run(out, key, requests, [tws._response([" word"] * 40), tws._response([" a"])])
    return tmp, AW.links(out, N)


def committed(tmp: Path, at: dict, name: str, altered: bool = False, made: str | None = None):
    mine = AW.window(at, N, REST)
    c = AW.commit(tmp / name, mine, at, altered, made)
    return mine, c, json.loads((tmp / name / "registration.json").read_text()), json.loads((tmp / name / "window.json").read_text())


def bcs(c, m, port: str) -> list[bytes]:
    b = c.bc[m.port(port)]
    return [b[i * AW.CS.BC_BYTES:(i + 1) * AW.CS.BC_BYTES] for i in range(m.n)]


def test_each_chunk_opens_one_connections_cells_under_the_wardens_root(run):
    tmp, at = run
    assert {x.head.value for x in at.values()} == {"warden/ingress/0", "warden/egress/0"}
    assert len({x.connection for x in at.values()}) == 1 and all(len(x.reads) == N for x in at.values())
    mine, c, _rec, win = committed(tmp, at, "honest")
    reads = AW.reads(mine, at)
    for name, m in AW.chunks(mine).items():
        x, port = at[name], AW.LINKS[name][1]
        B, S = x.params.cell_bytes, x.params.lanes * x.params.slots
        assert dict(m.rows)["row"] == S * B // 2 and x.registration["words"] == S * B // 2
        first = {AW.ws.INGRESS: 0, AW.ws.EGRESS: x.params.delay * x.params.slots}[name]
        assert x.units == WR.Units.span(x.connection, first, N) and x.reads == list(x.units.cells) and x.connection >= 1
        file = reads[m.key]["ports"][port]
        assert file == WR.port(x.head, x.opening.wiring, x.params, x.units, m.base) and file.get("reads_from", 0) == m.base
        assert [(t, j) for t, j in zip(x.positions, x.slots)] == x.opening.wiring.place(x.params, x.reads)
        assert len(set(x.positions)) < N
        for (q, _k), t, j in zip(x.reads, x.positions, x.slots):
            assert q == x.connection and HEADER.unpack_from(x.opening.rows[t][j * B:(j + 1) * B])[1] == q
        header = SR.registered_header(win, next(e["k"] for e in win["members"] if e["name"] == m.key))[port]
        assert header["value"] == file["value"] == x.units.value(x.head) != x.head.value
        assert header["positions"] == file["reads"] == x.positions
        assert WR.refusal(file, x.dom, header, bcs(c, m, port)) is None
        assert [WR.opens(x.dom, x.head.root, p, bc, bytes.fromhex(path))
                for p, bc, path in zip(x.positions, bcs(c, m, port), header["paths"])] == [True] * N
        assert bcs(c, m, port) == [WR.bc(x.opening.rows[p], x.opening.salts[p]) for p in x.positions]
    rest = mine.members[-1]
    assert "registered" not in win["members"][-1] and not set(c.salts[rest.port("row")]) & {s for x in at.values()
                                                                                             for s in x.opening.salts}


def test_the_registration_proves_every_transfer_gate_unit_beside_the_drawn_rest(run):
    tmp, at = run
    mine, c, rec, _win = committed(tmp, at, "law")
    tg, rid = mine.members[0].template, mine.members[-1].template
    assert tg != rid and mine.work_table == {tg: {"work": 0, "floor": 2 * N}, rid: {"work": 1, "floor": 1}}
    publics = {m.key: c.public_body(k) for k, m in enumerate(mine.members)}
    v = V.check(rec, mine.program, mine.query, mine.law, AW.CS.SCHEME, AW.CS.expected_ports(mine),
                leaf_layer=AW.CS.leaf_layer(publics), work_table=mine.work_table)
    assert v.ok, v.codes
    strata = {s["template"]: s for s in D.derive(mine.program, mine.query, mine.law, mine.work_table)["strata"]}
    assert strata[tg]["units"] == [[0, 2 * N]] and strata[tg]["k"] == 2 * N


def test_the_provenance_is_the_verifiers_read_and_the_negatives_are_refused_by_name(run):
    tmp, at = run
    mine, _c, rec, _win = committed(tmp, at, "source")
    source = AW.source_of(mine, at)
    assert source(rec["window"])["ok"]
    assert source({**rec["window"], "run": "00" * 64})["codes"] == ["run"]
    neg = AW.negatives(at, source, rec)
    assert all(x["refused"] for x in neg.values()), neg
    assert "doesn't sign its head" in neg["prover-root-under-the-wardens-certificate (check_registration)"]["why"]
    x = at[AW.ws.INGRESS]
    foreign = neg["foreign-read-under-a-crossed-wiring (port)"]
    assert foreign["cell_of"] == foreign["opens"][0] != foreign["read"][0]
    assert "isn't the one the warden signed" in foreign["why"]
    assert "is read twice" in neg["duplicate-read (port)"]["why"]
    assert "holds no slot" in neg["padding-slot-read-of-a-connection-holding-no-slot (port)"]["why"]
    padded = neg["padding-slot-read-under-a-padded-wiring (port)"]
    assert padded["empty"] and "isn't the one the warden signed" in padded["why"]
    assert f"connection {x.connection} has" in neg["read-past-the-connections-cells (port)"]["why"]
    e, pads = at[AW.ws.EGRESS], at[AW.ws.EGRESS].params.delay * at[AW.ws.EGRESS].params.slots
    pad = neg["statement-made-for-a-pad-in-a-data-cells-place (registered reads)"]
    assert pad["reads"][0] == [e.connection, 0] and pad["rows"][0] != pad["program_rows"][0]
    assert f"{e.units.value(e.head)}, not another value" in pad["why"]
    named = neg["pad-rows-under-the-programs-units-value (registered reads)"]
    assert f"unit 0 reads position {named['rows'][0]}, and the verifier's program reads {e.positions[0]}" in named["why"]
    swapped = neg["statement-made-for-two-units-swapped (registered reads)"]
    assert swapped["reads"] == [[e.connection, pads + 1], [e.connection, pads]] and swapped["rows"] == swapped["program_rows"]
    assert "not another value" in swapped["why"] and swapped["units"] != e.units.digest().hex()
    assert len(neg) == 10
    gates = AW.gate_rows(at[AW.ws.EGRESS])
    lay, B = at[AW.ws.EGRESS].registration["layout"], at[AW.ws.EGRESS].params.cell_bytes
    assert [g["bucket"] - lay["first_bucket"] for g in gates] == at[AW.ws.EGRESS].positions
    assert [g["words"] for g in gates] == [[j * B // 2, (j + 1) * B // 2] for j in at[AW.ws.EGRESS].slots]


def test_the_crossed_and_padded_wirings_would_open_a_foreign_cell_and_an_empty_slot(run):
    """What the two forged wirings would have the verifier read, had `port` placed them: connection q2's read on q1's
    cell, and a connection the warden never served on B zero bytes."""
    _tmp, at = run
    x = at[AW.ws.INGRESS]
    B, w = x.params.cell_bytes, x.opening.wiring
    forged, (q2, k2), (q1, k1), (t, j) = AW.crossed(x)
    assert forged.digest() != x.head.wiring and q1 != q2
    assert forged.place(x.params, [(q2, k2)]) == w.place(x.params, [(q1, k1)]) == [(t, j)]
    assert HEADER.unpack_from(x.opening.rows[t][j * B:(j + 1) * B])[1] == q1
    pad, (g, k), (tp, jp) = AW.padded(x)
    assert pad.digest() != x.head.wiring and g not in {c for c, _, _ in w.connections}
    assert pad.place(x.params, [(g, k)]) == [(tp, jp)] and x.opening.rows[tp][jp * B:(jp + 1) * B] == bytes(B)


def test_the_altered_cell_is_committed_under_the_wardens_salt_and_doesnt_open(run):
    tmp, at = run
    mine, c, _rec, win = committed(tmp, at, "altered", altered=True)
    m, x = AW.chunks(mine)[AW.ws.INGRESS], at[AW.ws.INGRESS]
    paths = SR.registered_header(win, next(e["k"] for e in win["members"] if e["name"] == m.key))["row"]["paths"]
    assert [WR.opens(x.dom, x.head.root, p, bc, bytes.fromhex(path))
            for p, bc, path in zip(x.positions, bcs(c, m, "row"), paths)] == [False] + [True] * (N - 1)
    assert c.salts[m.port("row")] == b"".join(x.opening.salts[p] for p in x.positions)
    assert (c.rows[m.port("o0")] == c.rows[m.port("row")]).all()
    B, j = x.params.cell_bytes, x.slots[0]
    got, want = bytes(c.rows[m.port("row")][0].astype("<u2").tobytes()), x.opening.rows[x.positions[0]]
    assert [i for i in range(len(want)) if got[i] != want[i]] == [(j + 1) * B - 1]


@pytest.mark.parametrize("made", sorted(AW.MADE_FOR))
def test_a_statement_made_for_another_units_wiring_is_refused_by_the_registered_reads(run, made):
    """``--made-for``'s window: c1's header made for another units' wiring, its rows the warden's under the warden's salts
    (they open the signed root at the positions the header names), its provenance the Program's, so the registration's
    checks pass and only the verifier's port of the Program's units refuses it, by `MADE_FOR`'s reason; c0 stays honest."""
    tmp, at = run
    mine, c, rec, win = committed(tmp, at, f"made-{made}", made=made)
    assert AW.source_of(mine, at)(rec["window"])["ok"]
    files = AW.reads(mine, at)
    for name, m in AW.chunks(mine).items():
        x, port = at[name], AW.LINKS[name][1]
        header = SR.registered_header(win, next(e["k"] for e in win["members"] if e["name"] == m.key))[port]
        assert all(WR.opens(x.dom, x.head.root, p, bc, bytes.fromhex(path))
                   for p, bc, path in zip(header["positions"], bcs(c, m, port), header["paths"]))
        why = WR.refusal(files[m.key]["ports"][port], x.dom, header, bcs(c, m, port))
        if name == AW.ws.INGRESS:
            assert why is None
        else:
            assert why.startswith("registered reads:") and AW.MADE_FOR[made] in why, why
            assert (header["value"] == x.units.value(x.head)) == (made == "pad-named")
            assert (header["positions"] == x.positions) == (made == "swapped")


def test_salts_of_names_one_salt_for_each_row_of_a_port_of_the_window(run):
    _tmp, at = run
    mine = AW.window(at, N, REST)
    data = [{"rows": {"row": np.zeros((k.n, w), np.uint16)}, "outs": {"o0": np.zeros((k.n, w), np.uint16)},
             **({"refs": np.arange(k.n)[:, None]} if k.tables else {})} for k in mine.members for w in [dict(k.rows)["row"]]]
    with pytest.raises(ValueError, match="one 192-byte salt for each row"):
        AW.CS.commit(mine, data, salts_of={mine.members[0].port("row"): bytes(AW.CS.SALT_BYTES)})
    with pytest.raises(ValueError, match="one 192-byte salt for each row"):
        AW.CS.commit(mine, data, salts_of={"no-such/row": bytes(AW.CS.SALT_BYTES * N)})
