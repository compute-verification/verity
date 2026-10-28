"""n2_zk's link-windows on certifier-serve's runs on an injected clock (`n2_zk.virtual`, a stub endpoint): every honest
link-window's output word is verdict 1 and deadline 1; window 0's egress head one count off, committed by the prover,
fails on its counts and its signed digest; the extra run's egress heads fail on their counts alone, the late run's on
their fields and counts. The verifier's own ``pub`` row, from public fields and the certificates alone, is the prover's,
in whole u16 words. The scan of what the verifier received leaves out exactly the witness pieces the public circuit holds.
``--jobs`` is 1 unless given and 2 is refused, naming the rule (`n2_zk.ONE_SESSION`). Nothing here lowers or proves
(`n2_zk.prove` does, on a node)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parents[1]


def _load():
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("network_traces_n2_zk", HERE / "n2_zk.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


n2 = _load()


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    out = tmp_path_factory.mktemp("n2zk")
    plan = n2.virtual(out, requests=8, tokens=4)
    return {"plan": plan, "honest": n2.link_windows(out / "honest"),
            "tampered": n2.link_windows(out / "honest", edit=n2.one_count),
            "extra": n2.link_windows(out / "extra"), "late": n2.link_windows(out / "late")}


def test_each_round_refuses_what_its_references_say(runs):
    honest = runs["honest"]
    assert len(honest.windows) == runs["plan"]["windows"] >= 2 and not honest.issuer_refused
    assert honest.lw.head_key(0)[2] == 1 and honest.lw.head_key(1)[2] == 0
    assert all(x.word == 0b11 and x.expect == 1 for x in honest.windows)
    for name in n2.ROUNDS:
        assert n2.expected(name, runs[name]), name
        assert (name == "honest") == all(x.expect for x in runs[name].windows), name
    e0, t0 = honest.windows[0].opening.records[0].head, runs["tampered"].windows[0].opening.records[0].head
    assert sum(a != b for a, b in zip(e0.counts, t0.counts)) == 1 and sum(t0.counts) == sum(e0.counts) + 1
    assert runs["tampered"].windows[0].certificates == honest.windows[0].certificates


def test_the_verifier_derives_the_provers_public_row(runs):
    for name in n2.ROUNDS:
        rd = runs[name]
        lw = rd.lw
        for x in rd.windows:
            mine = lw.public_arguments(x.window, x.root, x.certificates, rd.syncs)
            assert lw.row(mine, "pub") == lw.row(x.args, "pub"), (name, x.window)
            assert all(len(lw.row(x.args, r)) * 8 == lw.bits(r) and lw.bits(r) % 16 == 0 for r in n2.SVC.ROWS)


def _needles(blobs: dict[str, bytes]) -> dict[bytes, str]:
    out: dict[bytes, str] = {}
    for name, b in blobs.items():
        for o in range(0, len(b) - 15, 16):
            x = b[o:o + 16]
            for nd in (x, x.hex().encode(), x.hex().upper().encode()):
                out.setdefault(nd, name)
    return out


def test_the_scan_leaves_out_only_the_pieces_the_public_circuit_holds(runs):
    scan = SimpleNamespace(PIECE=16, needles=_needles, find=lambda data, t: {n: 1 for nd, n in t.items() if nd in data})
    blobs, held = n2.blobs_of(runs["honest"])
    assert {k.rsplit(" ", 1)[0] for k in blobs} >= {"head egress", "head salt ingress", "certifier salt egress",
                                                     "record salt ingress", "sibling 1 egress"}
    assert any(len(b) % 16 for b in blobs.values())  # a head's tail (403 bytes), which only the last pieces cover
    every = {b[o:o + 16].ljust(16, b"\0") for b in blobs.values()
             for o in [*range(0, len(b) - 15, 16)] + [len(b) - 16, len(b) - len(b) % 16] * (len(b) % 16 > 0)}
    head = blobs["head egress 0"][16:32]
    kept, held_kept, out = n2.public_pieces(scan, blobs, held, b"g 1 2\n" + bytes(16).hex().encode() + b"\n" + head)
    assert out == sorted(x.hex() for x in every & {bytes(16), head})
    assert set(kept.values()) == every - {bytes(16), head} and len(kept) == len(set(kept.values()))
    tails = {f"{k} @{len(blobs[k]) - len(blobs[k]) % 16}" for k in held if len(blobs[k]) % 16}
    assert tails and tails <= set(kept)  # a zero-padded tail is scanned for, but no instance file holds it as it is
    assert held_kept == {k for k in kept if k.rsplit(" @", 1)[0] in held} - tails and held_kept


def _zk():
    """Sampled proofs' mirror of `OneStageZk`'s decoders (`one_stage_zk_case`), loaded by path."""
    path = HERE.parents[1] / "verity/core/service/sampled_proofs/tests/one_stage_zk_case.py"
    if "one_stage_zk_case" not in sys.modules:
        sys.path.insert(0, str(path.parent))
        spec = importlib.util.spec_from_file_location("one_stage_zk_case", path)
        sys.modules["one_stage_zk_case"] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules["one_stage_zk_case"])
    return sys.modules["one_stage_zk_case"]


def test_one_stage_v2_reads_each_link_window_as_its_stream_registers_it(runs, tmp_path):
    """`one_stage_params` is one-stage v2's parameters (`OneStageZk.paramsOfJson`'s keys, without ports), and with each
    link-window's port as certifier/v0's period check derives it (its served domain, from the partition object and
    N2's run, `LinkWindow.domainId`) and the stream's earlier registrations as its log, each honest link-window's
    registration passes one-stage's registration check (R1-R7); `one_stage_evidence` of an outcome record is v2's
    evidence, its proof's files by SHA-512."""
    from verity.core.protocols.sampled_proofs.one_stage import partition as P, proofs as PF, registration as R
    from verity.ml.commitments.window import window_domain

    ZK, rd = _zk(), runs["honest"]
    lw = rd.lw
    template = n2.one_stage_params(lw, "ab" * 64, n2.cs.DELTA)
    assert template["delta"] == "1/1099511627776" and template["log"] == [] and "ports" not in template
    assert {*template, "ports"} == ZK.ONE_STAGE.PARAM_KEYS | {"proofs"} and sorted(template["proofs"]) == [
        "circuit", "registered"]
    assert template["program"] == json.loads(lw.program) and P.digest(template["partition"]) == P.digest(lw.partition)
    log = []
    for x in rd.windows:
        domain = window_domain(lw.partition["program"], P.digest(lw.partition), n2.SVC.VALUE, n2.SVC.SCHEMA,
                               n2.SVC.LEAVES, f"{lw.run}/{lw.link}/{x.window}")
        ports = {n2.SVC.VALUE: {"domain": domain, "leaves": n2.SVC.LEAVES}}
        assert ports == lw.ports(x.window)
        rec = lw.registration(x.opening, R.digest(log[-1]) if log else None)
        v = R.check(rec, lw._program, lw.query, template["law"], template["scheme"], ports, log=log,
                    partition=lw.partition)
        assert v.ok, (x.window, v.codes)
        log.append(rec)
    files = {n: tmp_path / n for n in ("circuit.txt", "pub-1.bin", "s/session.json", "s/0.bin")}
    for n, f in files.items():
        f.parent.mkdir(exist_ok=True)
        f.write_bytes(n.encode())
    proof = PF.Proof(files["circuit.txt"], files["pub-1.bin"], (tmp_path / "s",))
    sessions = [PF.file_sha512(files["s/session.json"])]
    audit = {"draw": {"k": 1, "law": "subset", "population": 1, "units": [0]}, "proved_units": [0],
             "receipt": R.receipt(log[0], "v", received="2026-10-26T00:00:00Z"), "statement": {"x": 1}}
    e = n2.one_stage_evidence(PF, {"audit": audit, "sessions": sessions}, proof, "cd" * 64)
    ZK.evidence(json.dumps(e, sort_keys=True, separators=(",", ":")).encode())
    assert e["session_roots"] == {n2.SVC.VALUE: "cd" * 64} and e["instances"] == 1
    assert e["proofs"] == {"public": PF.file_sha512(files["pub-1.bin"]), "sessions": [
        {"name": sessions[0], "record": sessions[0], "reps": {"0.bin": PF.file_sha512(files["s/0.bin"])}}]}


def test_jobs_is_one_firewalled_session_at_a_time(monkeypatch):
    monkeypatch.setattr(n2, "prove", lambda a: a.jobs)
    argv = ["prove", "--run", "r", "--out", "o", "--dry"]
    assert n2.main(argv) == 1 and n2.main([*argv, "--jobs", "1"]) == 1
    with pytest.raises(SystemExit, match="outer proofs go through the firewall one at a time"):
        n2.main([*argv, "--jobs", "2"])
