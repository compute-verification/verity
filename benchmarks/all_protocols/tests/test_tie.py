"""The runner's output tie (`verity_all_protocols.tie`):

* the out rows are the responses' u16 words in the plan's order, row ``(j·steps + s)·h + n`` response j's step s word n;
* the network certifier's record pins its root under the domain the verifier derives for the reading program and the
  loop's shape, and a record of another window, another row count or another key is refused by name
  (`out_commit.check_out`); the loop's opening commits the same root again, and the pin reads with the weights' in
  records_audit's ``--registered`` list;
* a tied loop period (the runner's ``loop_period`` on a relay the stub's outputs fill): ``period.json`` names the tie
  and the period's pins, every output unit's ``o0`` row opens the pinned root, and on a relay with one output byte
  changed exactly the row it changed doesn't (slow).
"""
from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from verity.core.protocols.certifier import out_commit as OC
from verity.core.protocols.sampled_proofs.one_stage import registered as REG
from verity.ml.windows.rows import BC_BYTES
from verity_all_protocols import runner as RN, table as T, tie as TIE

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks" / "network_traces"))
sys.path.insert(0, str(ROOT / "benchmarks" / "prove_path"))
import active_replay as ar  # noqa: E402
import records_audit as RA  # noqa: E402
import torch_loop_stub as S  # noqa: E402

H, STEPS = 128, 1
PROGRAM = "ab" * 64


def responses(tokens: list[int]) -> list[bytes]:
    model = S.L.Model(h=H)
    return [b"".join(S.step_outputs(model, t, STEPS)) for t in tokens]


def test_the_out_rows_are_each_responses_words_in_order():
    rs = [bytes(range(2 * H)) * 2, bytes(range(255, 255 - 2 * H, -1)) * 2]
    rows = TIE.out_rows(rs, H, 2)
    assert len(rows) == 2 * 2 * H and all(len(r) == 1 for r in rows)
    j, s, n = 1, 1, 5
    assert rows[(j * 2 + s) * H + n] == [int.from_bytes(rs[j][s * 2 * H + 2 * n:s * 2 * H + 2 * n + 2], "little")]
    with pytest.raises(ValueError, match="steps of"):
        TIE.out_rows([rs[0][:-2]], H, 2)


def test_the_record_pins_its_root_and_another_window_count_or_key_is_refused(tmp_path):
    c = TIE.certifier(tmp_path / "c")
    rows = TIE.out_rows(responses([3]), H, STEPS)
    got = OC.commit_out(rows, program=PROGRAM, words=1, key=c.issuer, session=c.session, window=2)
    pin = TIE.pin(got.public, c, program=PROGRAM, rows=len(rows), window=2)
    entry = pin["values"]["out"]
    assert entry["root"] == got.root and entry["domain"] == got.domain and entry["leaves"] == H
    assert RA.pinned([{"program": "torch-loop/w", "values": {}}, pin]) == {"out": entry}
    with pytest.raises(ValueError, match="pinned twice"):
        RA.pinned([pin, pin])
    for kw, name in (({"window": 3}, "out-window"), ({"rows": len(rows) + 1}, "out-domain")):
        with pytest.raises(OC.OutRefused) as e:
            TIE.pin(got.public, c, **({"program": PROGRAM, "rows": len(rows), "window": 2} | kw))
        assert e.value.name == name
    other = TIE.certifier(tmp_path / "other")
    with pytest.raises(OC.OutRefused) as e:
        TIE.pin(got.public, other, program=PROGRAM, rows=len(rows), window=2)
    assert e.value.name == "out-key"
    L = TIE._loop()
    assert L.opened_out(L.out_opening(PROGRAM, rows, list(got.salts))).root.hex() == got.root


def _unopened(d: Path, period: dict) -> list[int]:
    """The positions of ``out`` whose output unit's ``o0`` row doesn't open the root the period pinned."""
    entry = RA.pinned(json.loads(Path(period["registered"]).read_text()))["out"]
    program = json.loads(Path(period["tie"]["record"]).read_text())["program"]
    dom = REG.domain(program, "out", 1, entry["leaves"], REG.WORD_BITS)
    bad = []
    for m in json.loads((d / "window.json").read_text())["members"]:
        if "o0" not in (m.get("registered") or {}):
            continue
        sizes, names = [int(m["shared_rows"][p]) for p, _ in m["ports"]], [p for p, _ in m["ports"]]
        pub, off = (d / f"m{m['k']}-public.bin").read_bytes(), BC_BYTES * sum(sizes[:names.index("o0")])
        e = m["registered"]["o0"]
        for r, (pos, path) in enumerate(zip(e["positions"], e["paths"])):
            bc = pub[off + r * BC_BYTES:off + (r + 1) * BC_BYTES]
            if not REG.verify_read(dom, bytes.fromhex(entry["root"]), pos, bc, bytes.fromhex(path), REG.SCHEMA_V2):
                bad.append(pos)
    return bad


@pytest.mark.slow
def test_a_tied_period_opens_every_output_row_and_a_changed_byte_does_not(tmp_path, monkeypatch):
    honest = responses([3])
    byte = [ar.other_hex(honest[0].hex().encode()).decode()]
    relays = {"honest": honest, "byte": [bytes.fromhex(byte[0])]}
    monkeypatch.setattr(TIE, "load", lambda name: Namespace(relayed=lambda rec, strategy, i: (i, relays[strategy])))
    a = Namespace(law="work:4", cap=8, requests="3", steps=STEPS, tamper=set(), node="test-node", tools={}, periods=2,
                  weights=None, tie_tamper={1}, relay={"h": H, "steps": STEPS}, certifier=TIE.certifier(tmp_path / "c"))
    for i, strategy, bad in ((0, "honest", []), (1, "byte", [0])):
        d, _ = RN.loop_period(i, "test", a, tmp_path / f"s{i}", tmp_path / f"p{i}")
        p = T.period_of(d)
        assert {k: p["tie"][k] for k in ("value", "port", "stratum", "strategy", "window")} == {
            "value": "out", "port": "o0", "stratum": "output class", "strategy": strategy, "window": i}
        assert p["registered"] == str(tmp_path / f"p{i}" / "pin.json") and p["prover_env"] == {}
        assert RA.output_class(json.loads((d / "circuit.json").read_text())) == [[H, 2 * H]]
        assert _unopened(d, p) == bad
