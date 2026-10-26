"""The runner's output tie (`verity_all_protocols.tie`):

* the out rows are the responses' u16 words in the plan's order, row ``(j·steps + s)·h + n`` response j's step s word n;
* the network certifier's record pins its root under the domain the verifier derives for the reading program and the
  loop's shape, and a record of another window, another row count, another key or another session is refused by name
  (`out_commit.check_out`); the loop's opening commits the same root again, and the pin reads with the weights' in
  records_audit's ``--registered`` list;
* the runner reads a relay run's out record and witness as the network certifier wrote them (`certify_out`) and its
  key and session off its deployment, and refuses a link-window the certifier signed no out record for;
* a tied loop period (the runner's ``loop_period`` on relay runs whose out records the certifier's own `certify_out`
  wrote from the stub's outputs): ``period.json`` names the tie and the period's pins, every output unit's ``o0`` row
  opens the pinned root, and on a relay with one output byte changed exactly the row it changed doesn't (slow).
"""
from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity.core.protocols.certifier import active, out_commit as OC
from verity.core.protocols.certifier.certificate import Issuer
from verity.core.protocols.sampled_proofs.one_stage import registered as REG
from verity.ml.windows.rows import BC_BYTES
from verity_all_protocols import runner as RN, table as T, tie as TIE

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks" / "network_traces"))
sys.path.insert(0, str(ROOT / "benchmarks" / "prove_path"))
import active_replay as ar  # noqa: E402
import certifier_serve as cs  # noqa: E402
import records_audit as RA  # noqa: E402
import torch_loop_stub as S  # noqa: E402

H, STEPS = 128, 1
PROGRAM = "ab" * 64
SESSION = bytes(range(32))


def responses(tokens: list[int]) -> list[bytes]:
    model = S.L.Model(h=H)
    return [b"".join(S.step_outputs(model, t, STEPS)) for t in tokens]


def egress(window: int, rs: list[bytes]):
    """The egress record of a link-window whose responses are ``rs``, each as its hex text, 64 bytes an event."""
    text = b"".join(r.hex().encode() for r in rs)
    frame = lambda body: active.HEADER.pack(active.DATA, active.CANONICAL_STREAM, len(body)) + body  # noqa: E731
    return SimpleNamespace(window=window, grid=[[frame(active.event(text[i:i + active.TOKEN_BYTES]))
                                                 for i in range(0, len(text), active.TOKEN_BYTES)]])


def relay_run(run: Path, issuer: Issuer, program: str, windows: dict[int, list[bytes]]) -> None:
    """A relay run's files as the network certifier writes them: its deployment's key and session, and each
    link-window's out record and witness (`certify_out`)."""
    run.mkdir(parents=True, exist_ok=True)
    (run / "deployment.json").write_text(json.dumps({"certifier_key": cs.public_json(issuer.signer.public_key()),
                                                     "session": issuer.session.hex()}))
    for w, rs in windows.items():
        assert cs.certify_out(run, egress(w, rs), issuer, program) is not None


def test_the_out_rows_are_each_responses_words_in_order():
    rs = [bytes(range(2 * H)) * 2, bytes(range(255, 255 - 2 * H, -1)) * 2]
    rows = TIE.out_rows(rs, H, 2)
    assert len(rows) == 2 * 2 * H and all(len(r) == 1 for r in rows)
    j, s, n = 1, 1, 5
    assert rows[(j * 2 + s) * H + n] == [int.from_bytes(rs[j][s * 2 * H + 2 * n:s * 2 * H + 2 * n + 2], "little")]
    assert cs.out_rows(egress(0, rs)) == rows
    with pytest.raises(ValueError, match="steps of"):
        TIE.out_rows([rs[0][:-2]], H, 2)


def test_the_record_pins_its_root_and_another_window_count_key_or_session_is_refused(tmp_path):
    signer, other = cs.signer(tmp_path / "c", 4), cs.signer(tmp_path / "other", 4)
    try:
        rows = TIE.out_rows(responses([3]), H, STEPS)
        got = OC.commit_out(rows, program=PROGRAM, words=1, key=Issuer(signer, SESSION), session=SESSION, window=2)
        key = signer.public_key()
        pin = TIE.pin(got.public, key, SESSION, program=PROGRAM, rows=len(rows), window=2)
        entry = pin["values"]["out"]
        assert entry["root"] == got.root and entry["domain"] == got.domain and entry["leaves"] == H
        assert RA.pinned([{"program": "torch-loop/w", "values": {}}, pin]) == {"out": entry}
        with pytest.raises(ValueError, match="pinned twice"):
            RA.pinned([pin, pin])
        for kw, name in (({"window": 3}, "out-window"), ({"rows": len(rows) + 1}, "out-domain"),
                         ({"key": other.public_key()}, "out-key"), ({"session": bytes(32)}, "out-session")):
            args = {"key": key, "session": SESSION, "program": PROGRAM, "rows": len(rows), "window": 2} | kw
            with pytest.raises(OC.OutRefused) as e:
                TIE.pin(got.public, args.pop("key"), args.pop("session"), **args)
            assert e.value.name == name
    finally:
        signer.close()
        other.close()
    L = TIE._loop()
    assert L.opened_out(L.out_opening(PROGRAM, rows, list(got.salts))).root.hex() == got.root


def test_the_runner_reads_the_certifiers_out_record_witness_and_device_off_the_relay_run(tmp_path):
    signer = cs.signer(tmp_path / "key", 4)
    try:
        issuer = Issuer(signer, SESSION)
        relay_run(tmp_path / "run", issuer, PROGRAM, {0: responses([3]), 1: responses([5])})
        refused = Issuer(signer, bytes(32))
        assert cs.certify_out(tmp_path / "run", egress(2, responses([3])), refused, PROGRAM) is None
        key = signer.public_key()
    finally:
        signer.close()
    record, witness = TIE.certified_out(tmp_path / "run", 1)
    assert TIE.device(tmp_path / "run") == (key, SESSION)
    assert witness["rows"] == TIE.out_rows(responses([5]), H, STEPS) and len(witness["salts"]) == H
    assert TIE.pin(record, key, SESSION, program=PROGRAM, rows=H, window=1)["values"]["out"]["root"] == record["root"]
    with pytest.raises(RuntimeError, match="refused link-window 2"):
        TIE.certified_out(tmp_path / "run", 2)
    with pytest.raises(RuntimeError, match="0 out records and 0 witnesses of link-window 3"):
        TIE.certified_out(tmp_path / "run", 3)


def test_the_tie_draws_under_a_work_law_and_refuses_any_other(tmp_path, monkeypatch, capsys):
    """a5 proves the output class only beside a streamed work draw, so a ``--tie`` run under ``subset:2`` would reject
    every period: the runner defaults ``--tie`` to ``work:64`` and refuses any other law before it relays."""
    base = ["--out", str(tmp_path / "out"), "--tools", str(tmp_path / "missing.json")]
    for law in ("subset:2", "stratified:4", "bernoulli:1/2"):
        with pytest.raises(SystemExit) as x:
            RN.main([*base, "--tie", "--law", law])
        assert x.value.code == 2
        assert f"give --law work:K (the default with --tie is work:64), not {law}" in capsys.readouterr().err
    (tmp_path / "tools.json").write_text("{}")
    base[3] = str(tmp_path / "tools.json")
    laws = []

    def serve(argv):
        laws.append(argv[argv.index("--law") + 1])
        raise SystemExit(0)

    monkeypatch.setattr(RN, "serve", serve)
    for argv, law in ((["--tie"], "work:64"), (["--tie", "--law", "work:4"], "work:4"), ([], "subset:2")):
        with pytest.raises(SystemExit):
            RN.main([*base, *argv, "--periods", "1"])
        assert laws.pop() == law


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
    relays = {"honest": honest, "byte": [bytes.fromhex(ar.other_hex(honest[0].hex().encode()).decode())]}
    monkeypatch.setattr(TIE, "load", lambda name: Namespace(relayed=lambda rec, strategy, i: (i, relays[strategy])))
    a = Namespace(law="work:4", cap=8, requests="3", steps=STEPS, tamper=set(), node="test-node", tools={}, periods=2,
                  weights=None, tie=True, tie_tamper={1}, relay={"h": H, "steps": STEPS, "dir": str(tmp_path / "s")})
    program = RN.serve([*RN.shape_of(0, "test", a), "--circuit-only"])["program"]
    signer = cs.signer(tmp_path / "key", 4)
    try:
        issuer = Issuer(signer, SESSION)
        for strategy in relays:
            relay_run(tmp_path / "s" / "rehearsal" / strategy, issuer, program, {0: relays[strategy],
                                                                                 1: relays[strategy]})
    finally:
        signer.close()
    for i, strategy, bad in ((0, "honest", []), (1, "byte", [0])):
        d, _ = RN.loop_period(i, "test", a, tmp_path / f"s{i}", tmp_path / f"p{i}")
        p = T.period_of(d)
        assert {k: p["tie"][k] for k in ("value", "port", "stratum", "strategy", "window")} == {
            "value": "out", "port": "o0", "stratum": "output class", "strategy": strategy, "window": i}
        assert p["registered"] == str(tmp_path / f"p{i}" / "pin.json") and p["prover_env"] == {}
        assert RA.output_class(json.loads((d / "circuit.json").read_text())) == [[H, 2 * H]]
        assert _unopened(d, p) == bad
