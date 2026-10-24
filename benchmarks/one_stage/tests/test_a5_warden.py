"""a5_warden's window on a warden run driven tick by tick (`test_warden_serve._run`), before anything is lowered or proved
(`a5_warden.main` does, through `a5.audit`, on a node): each transfer-gate chunk's registered port's rows, committed under
the warden's salts, open the warden's signed root at the rows the verifier's own reads name, and the header names the same
rows; the registration passes the verifier's check with every transfer-gate unit proved (work 0, floor all of them); the
window's provenance is the verifier's read of the warden's registrations; the prover's own root under the warden's
certificate and provenance naming another root are refused; and the altered window's unit 0 doesn't open."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from verity.core.protocols.certifier import warden_rows as WR
from verity.core.protocols.sampled_proofs.one_stage import draw as D, verifier as V

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


def committed(tmp: Path, at: dict, name: str, altered: bool = False):
    mine = AW.window(at, N, REST)
    c = AW.commit(tmp / name, mine, at, altered)
    return mine, c, json.loads((tmp / name / "registration.json").read_text()), json.loads((tmp / name / "window.json").read_text())


def bcs(c, m, port: str) -> list[bytes]:
    b = c.bc[m.port(port)]
    return [b[i * AW.CS.BC_BYTES:(i + 1) * AW.CS.BC_BYTES] for i in range(m.n)]


def test_each_chunk_opens_one_connections_cells_under_the_wardens_root(run):
    tmp, at = run
    assert {x.head.value for x in at.values()} == {"warden/ingress/0", "warden/egress/0"}
    assert len({x.connection for x in at.values()}) == 1 and all(len(x.positions) == N for x in at.values())
    mine, c, _rec, win = committed(tmp, at, "honest")
    reads = AW.reads(mine, at)
    for name, m in AW.chunks(mine).items():
        x, port = at[name], AW.LINKS[name][1]
        file = reads[m.key]["ports"][port]
        assert file == WR.port(x.head, x.positions, m.base) and file.get("reads_from", 0) == m.base
        header = next(e for e in win["members"] if e["name"] == m.key)["registered"][port]
        assert header["value"] == file["value"] == x.head.value and header["positions"] == file["reads"]
        assert [WR.opens(x.dom, x.head.root, p, bc, bytes.fromhex(path))
                for p, bc, path in zip(x.positions, bcs(c, m, port), header["paths"])] == [True] * N
        assert bcs(c, m, port) == [WR.bc(x.opening.rows[p], x.opening.salts[p]) for p in x.positions]
    rest = mine.members[-1]
    assert "registered" not in win["members"][-1] and not set(c.salts[rest.port("cell")]) & {s for x in at.values()
                                                                                              for s in x.opening.salts}


def test_the_registration_proves_every_transfer_gate_unit_beside_the_drawn_rest(run):
    tmp, at = run
    mine, c, rec, _win = committed(tmp, at, "law")
    tg, rid = mine.members[0].template, mine.members[-1].template
    assert mine.work_table == {tg: {"work": 0, "floor": 2 * N}, rid: {"work": 1, "floor": 1}}
    publics = {m.key: c.public_body(k) for k, m in enumerate(mine.members)}
    v = V.check(rec, mine.program, mine.query, mine.law, AW.CS.SCHEME, AW.CS.expected_ports(mine),
                leaf_layer=AW.CS.leaf_layer(publics), work_table=mine.work_table)
    assert v.ok, v.codes
    strata = {s["template"]: s for s in D.derive(mine.program, mine.query, mine.law, mine.work_table)["strata"]}
    assert strata[tg]["units"] == [[0, 2 * N]] and strata[tg]["k"] == 2 * N


def test_the_provenance_is_the_verifiers_read_and_the_negatives_are_refused(run):
    tmp, at = run
    mine, _c, rec, _win = committed(tmp, at, "source")
    source = AW.source_of(mine, at)
    assert source(rec["window"])["ok"]
    assert source({**rec["window"], "run": "00" * 64})["codes"] == ["run"]
    neg = AW.negatives(at, source, rec)
    assert all(x["refused"] for x in neg.values()), neg
    assert "doesn't sign its head" in neg["prover-root-under-the-wardens-certificate (check_registration)"]["why"]
    gates = AW.gate_rows(at[AW.ws.EGRESS])
    lay = at[AW.ws.EGRESS].registration["layout"]
    assert [(g["bucket"] - lay["first_bucket"]) * lay["slots"] + g["slot"] for g in gates] == at[AW.ws.EGRESS].positions


def test_the_altered_cell_is_committed_under_the_wardens_salt_and_doesnt_open(run):
    tmp, at = run
    mine, c, _rec, win = committed(tmp, at, "altered", altered=True)
    m, x = AW.chunks(mine)[AW.ws.INGRESS], at[AW.ws.INGRESS]
    paths = next(e for e in win["members"] if e["name"] == m.key)["registered"]["cell"]["paths"]
    assert [WR.opens(x.dom, x.head.root, p, bc, bytes.fromhex(path))
            for p, bc, path in zip(x.positions, bcs(c, m, "cell"), paths)] == [False] + [True] * (N - 1)
    assert c.salts[m.port("cell")] == b"".join(x.opening.salts[p] for p in x.positions)
    assert (c.rows[m.port("o0")] == c.rows[m.port("cell")]).all()


def test_salts_of_names_one_salt_for_each_row_of_a_port_of_the_window(run):
    _tmp, at = run
    mine = AW.window(at, N, REST)
    data = [{"rows": {"cell": np.zeros((k.n, w), np.uint16)}, "outs": {"o0": np.zeros((k.n, w), np.uint16)},
             **({"refs": np.arange(k.n)[:, None]} if k.tables else {})} for k in mine.members for w in [dict(k.rows)["cell"]]]
    with pytest.raises(ValueError, match="one 192-byte salt for each row"):
        AW.CS.commit(mine, data, salts_of={mine.members[0].port("cell"): bytes(AW.CS.SALT_BYTES)})
    with pytest.raises(ValueError, match="one 192-byte salt for each row"):
        AW.CS.commit(mine, data, salts_of={"no-such/cell": bytes(AW.CS.SALT_BYTES * N)})
