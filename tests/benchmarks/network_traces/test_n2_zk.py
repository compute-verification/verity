"""n2_zk's execution phases on an active network attester's synthetic run (the chain tests' `gate_run`): every
honest execution phase's output word is verdict 1 and deadline 1, and window 0's egress head one count off, committed
by the prover, fails on its signed digest alone. The verifier's own ``pub`` row, from public fields and the
certificates alone, is the prover's, in whole u16 words. The scan of what the verifier received leaves out exactly the
witness pieces the public circuit holds. ``--jobs`` is 1 unless given and 2 is refused, naming the rule
(`n2_zk.ONE_SESSION`). The printed log is public: a child's or the driver's failure text and every witness line go to
the witness side's ``n2.log``, and without ``--witness`` nowhere. Nothing here lowers or proves (`n2_zk.prove` does, on
a node)."""

from __future__ import annotations

import json
import secrets
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity.protocols.resource_accounting.network.prover import n2_zk as n2
from tests.benchmarks.network_traces.test_network_attester_chain import gate_run
from benchmarks.network_traces import zk_drivers

zk_drivers.install()

RELEASED_KEYS = Path(n2.__file__).resolve().parents[0] / "released_keys.py"


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    run = gate_run(tmp_path_factory.mktemp("n2zk"))
    return {"honest": n2.phases(run), "tampered": n2.phases(run, edit=n2.one_count)}


def test_each_round_refuses_what_its_references_say(runs):
    honest = runs["honest"]
    assert len(honest.windows) == honest.plan["windows"] >= 2 and not honest.issuer_refused
    assert all(x.word == 0b11 and x.expect == 1 for x in honest.windows)
    for name in n2.ROUNDS:
        assert n2.expected(name, runs[name]), name
        xs = runs[name].windows
        assert (name == "honest") == (bool(xs) and all(x.expect for x in xs)), name
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
    assert {k.rsplit(" ", 1)[0] for k in blobs} >= {"head egress", "head salt ingress", "attester salt egress",
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


def _summary(run: Path, at: Path) -> tuple[dict, dict]:
    assert n2.main(["prove", "--warden-run", str(run), "--out", str(at / "out"), "--dry", "--witness",
                    str(at / "witness" / "n2" / "summary.json")]) == 0
    return json.loads((at / "out" / "summary.json").read_text()), json.loads((at / "witness" / "n2" / "summary.json")
                                                                            .read_text())


def _released(doc: dict, at: Path) -> tuple[int, str]:
    """released_keys.py's exit and what it said, on a record whose ``n2/summary.json`` is `doc`."""
    (at / "n2").mkdir(parents=True)
    (at / "n2" / "summary.json").write_text(json.dumps(doc))
    got = subprocess.run([sys.executable, str(RELEASED_KEYS), str(at)], capture_output=True, text=True)
    return got.returncode, got.stdout + got.stderr


def _public_lines(out: str) -> list[str]:
    return [line.split(" ", 2)[2] for line in out.splitlines() if line.startswith("N2ZK ")]


def test_the_summary_holds_its_released_keys_and_the_rest_goes_to_the_witness(tmp_path, capfd):
    """The public summary holds only keys its list of released_keys.json declares; the run's path, the device, the
    timings, the reference's failed parts and which execution phases the signer refused are written to ``--witness``
    alone, which the list refuses, and without it nowhere. The printed log is the summary's split too: the failed
    parts go to ``n2.log`` beside the witness."""
    run = gate_run(tmp_path / "g")
    capfd.readouterr()
    doc, witness = _summary(run, tmp_path)
    said = "\n".join(_public_lines(capfd.readouterr().out))
    assert "execution phases, words" in said and "failed parts" not in said
    assert "failed parts" in (tmp_path / "witness" / "n2" / "n2.log").read_text()
    assert _released(doc, tmp_path / "public") == (0, "released keys: 1 files checked by key, 0 with a key not declared "
                                                      "public; 0 not yet checked; 0 with no keys\n")
    assert {m["attester_refused"] for m in doc["members"].values()} == {False} and "issuer_refused" not in str(doc)
    assert {"run", "gpu", "instances_s"} <= set(witness) and not {"run", "gpu", "instances_s"} & set(doc)
    assert {k: v["issuer_refused"] for k, v in witness["members"].items()} == {k: [] for k in doc["members"]}
    assert all(set(r) == {"failed"} for m in witness["members"].values() for r in m["reference"].values())
    rc, said = _released(witness, tmp_path / "private")
    assert rc == 3 and all(f"key {k} is not declared public" in said
                           for k in ("run", "gpu", "instances_s", "members/*/issuer_refused"))
    assert n2.main(["prove", "--warden-run", str(run), "--out", str(tmp_path / "again"), "--dry"]) == 0
    assert "issuer_refused" not in (tmp_path / "again" / "summary.json").read_text()


def test_a_refused_execution_phase_leaves_the_public_summary_the_same_wherever_it_fell(tmp_path, monkeypatch, capfd):
    """When the signer refused an execution phase, the public summary is each member's flag alone, with no window
    count, so nothing public says where the refusal fell; the witness holds the members whole, and the log's
    per-phase lines go to the witness's ``n2.log``."""
    run, real = gate_run(tmp_path / "g"), n2.phases

    def refused_at(w: int):
        def phases(r, **kw):
            rd = real(r, **kw)
            rd.issuer_refused = [] if kw else [w]
            return rd
        return phases

    seen, printed = [], []
    for w in (0, 1):
        monkeypatch.setattr(n2, "phases", refused_at(w))
        capfd.readouterr()
        doc, witness = _summary(run, tmp_path / str(w))
        printed.append(_public_lines(capfd.readouterr().out))
        assert "execution phases" in (tmp_path / str(w) / "witness" / "n2" / "n2.log").read_text()
        assert doc["members"] == {k: {"attester_refused": True} for k in doc["members"]} and "windows" not in doc["link"]
        assert witness["members"]["honest"]["issuer_refused"] == [w] and witness["link"]["windows"]
        assert _released(doc, tmp_path / str(w) / "record")[0] == 0
        seen.append(doc)
    assert seen[0] == seen[1] and printed[0] == printed[1] and not any("phases" in x for x in printed[0])


def test_jobs_is_one_zk_server_session_at_a_time(monkeypatch):
    monkeypatch.setattr(n2, "prove", lambda a: a.jobs)
    argv = ["prove", "--warden-run", "r", "--out", "o", "--dry"]
    assert n2.main(argv) == 1 and n2.main([*argv, "--jobs", "1"]) == 1
    with pytest.raises(SystemExit, match="outer proofs go through the ZK server one at a time"):
        n2.main([*argv, "--jobs", "2"])


#: built at run time: a literal of this shape reads as a credential (`test_no_secret_like_literals`)
MARKER = "n2-marker-" + secrets.token_hex(8)


def _failing_child(r, **kw):
    subprocess.run([sys.executable, "-c", f"import sys; print('{MARKER} out'); print('{MARKER} err', file=sys.stderr)"])
    print(f"{MARKER} stray", file=sys.stderr)
    raise SystemExit(f"{MARKER}: the run is refused")


@pytest.mark.parametrize("witness", [True, False])
def test_a_failure_and_its_childrens_output_reach_only_the_witness_side(tmp_path, monkeypatch, capfd, witness):
    """A child that prints a secret on stdout and stderr, a stray print and a failure whose text holds it: the public
    stream says the failure's name alone, and the text is in the witness's ``n2.log``, or nowhere without one."""
    monkeypatch.setattr(n2, "phases", _failing_child)
    w = tmp_path / "witness" / "n2" / "summary.json"
    capfd.readouterr()
    rc = n2.main(["prove", "--warden-run", "r", "--out", str(tmp_path / "out"), "--dry",
                  *(["--witness", str(w)] if witness else [])])
    said = capfd.readouterr()
    assert rc == 1 and MARKER not in said.out + said.err
    assert _public_lines(said.out)[-1] == "failed: SystemExit" and not said.err
    if witness:
        held = w.with_name("n2.log").read_text()
        assert all(f"{MARKER} {k}" in held for k in ("out", "err", "stray")) and f"{MARKER}: the run is refused" in held
    else:
        assert not (tmp_path / "witness").exists()
        assert _public_lines(said.out)[0].startswith("no --witness asked for")


def test_a_run_without_witness_prints_no_witness_line(tmp_path, capfd):
    run = gate_run(tmp_path / "g")
    _summary(run, tmp_path / "with")
    lines = {x.split(" ", 2)[2] for x in (tmp_path / "with" / "witness" / "n2" / "n2.log").read_text().splitlines()}
    assert lines
    capfd.readouterr()
    assert n2.main(["prove", "--warden-run", str(run), "--out", str(tmp_path / "out"), "--dry"]) == 0
    said = capfd.readouterr()
    assert not lines & set(_public_lines(said.out)) and "failed parts" not in said.out + said.err


def test_debug_clear_writes_only_to_the_witness_side(tmp_path):
    argv = ["prove", "--warden-run", "r", "--out", str(tmp_path / "out"), "--dry", "--debug-clear"]
    with pytest.raises(SystemExit, match="it needs --witness"):
        n2.main(argv)
    with pytest.raises(SystemExit, match="--witness is under --out"):
        n2.main([*argv, "--witness", str(tmp_path / "out" / "w.json")])
    verdict = {"accepted": True, "clear": {"accepted": True}}
    body = {"audit": {"verdicts": {n2.SVC.VERIFIER: verdict}}}
    assert n2._without_clear(body) == {"audit": {"verdicts": {n2.SVC.VERIFIER: {"accepted": True}}}}
    assert "clear" in verdict and n2._without_clear({"audit": {}}) == {"audit": {}}
