"""`flock-audit` (`FlockAudit`): the one-stage audit's auditor as a Lean program, its workers answering requests. The program
registers, draws the unit draw itself (its one coin request, `auditor_draws_once`), sets up the drawn statement at its own
draw, asks the serving worker for that session, refuses a record whose draw is not its own (U1) and runs `Zk.verify`. The
worker (`audit_workers.py`) replays the fixture's drawn session (a subset draw of 16 of 32) into the auditor's content
store, and `flock-audit-tape` reads the coins from a tape that draws that session's units, so the run reproduces a recorded
session without a prover; `--tamper` makes one of the worker's answers dishonest."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
LEAN = TESTS.parent / "verifier" / "lean"
VERIFIER = LEAN.parent
BIN = LEAN / ".lake" / "build" / "bin"
#: RoPE d64 staged by verity_flock.circuit with hidden outputs, with the live build's --zk sessions (test_lean_zk.py):
#: sessions/drawn is a subset draw of 16 of 32 at pub-32.bin, accepted by its server
FIXTURE = "art:5542740a"
STATEMENT = "verity/flock-circuit+seed-injection"

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")


@pytest.fixture(scope="module")
def built():
    subprocess.run(["lake", "build"], cwd=LEAN, check=True, capture_output=True)


@pytest.fixture(scope="module")
def zk(built, tmp_path_factory):
    sys.path.insert(0, str(VERIFIER))
    import ci

    try:
        return ci.fetch(FIXTURE, tmp_path_factory.mktemp("store"))
    except (subprocess.CalledProcessError, OSError) as e:
        pytest.skip(f"the evidence store is not reachable: {e}")


def _drawn(root: Path) -> Path:
    return next(p for p in (root / "sessions" / "drawn").iterdir() if p.is_dir())


def _tape(draw: dict) -> bytes:
    """Coins on which `Draw.subset` draws `draw`'s units: its partial Fisher–Yates inverted, each step's `uniform(n - i)` a
    `width(n)`-byte big-endian word, padded to the one request of `2·k·width(n) + 64` bytes the sampler makes."""
    n, k = draw["population"], draw["k"]
    width = (n.bit_length() + 64 + 7) // 8
    sw, out = list(range(n)), b""
    for i, u in enumerate(draw["units"]):
        j = sw.index(u)
        out += (j - i).to_bytes(width, "big")
        sw[i], sw[j] = sw[j], sw[i]
    return out + bytes(2 * k * width + 64 - len(out))


def _own(pin: str) -> list[str]:
    return ["--zk", "--statement", STATEMENT, "--circuit", pin, "--k", "16"]


def _run(root: Path, d: Path, tamper: str | None = None) -> tuple[dict, subprocess.CompletedProcess]:
    """The auditor's run over the worker in `d`, with the fixture's draw on its coin tape: its verdict and the process."""
    circuit = (root / "stage" / "circuit.txt").read_bytes()
    pin = hashlib.sha512(circuit).hexdigest()
    (d / "archive" / "sha512").mkdir(parents=True)
    (d / "archive" / "sha512" / pin).write_bytes(circuit)
    draw = json.loads((_drawn(root) / "session.json").read_text())["unit_draw"]
    (d / "tape.bin").write_bytes(_tape(draw))
    argv = [sys.executable, str(TESTS / "audit_workers.py"), "--archive", str(d / "archive"), "--session", str(_drawn(root)),
            "--public", str(root / "stage" / "pub-32.bin"), "--log", str(d / "log.jsonl"), *(["--tamper", tamper] if tamper else [])]
    (d / "workers.json").write_text(json.dumps({"workers": [{"name": "py", "channels": ["developer", "serving", "audit"],
                                                             "argv": argv}]}))
    p = subprocess.run([str(BIN / "flock-audit-tape"), "--coin-tape", str(d / "tape.bin"), "run", *_own(pin),
                        "--archive", str(d / "archive"), "--workers", str(d / "workers.json"), "--transcript", str(d / "t.bin")],
                       capture_output=True, text=True, check=False)
    return _verdict(p), p


def _verdict(p: subprocess.CompletedProcess) -> dict:
    [v] = [json.loads(line.split(" ", 1)[1]) for line in p.stdout.splitlines() if line.startswith("VERDICT ")]
    assert p.returncode == (0 if v["accepted"] else 1), (p.returncode, v)
    return v


def _replay(root: Path, transcript: Path) -> dict:
    pin = hashlib.sha512((root / "stage" / "circuit.txt").read_bytes()).hexdigest()
    return _verdict(subprocess.run([str(BIN / "flock-audit"), "replay", *_own(pin), "--transcript", str(transcript)],
                                   capture_output=True, text=True, check=False))


def _frame(b: bytes, off: int) -> tuple[bytes, int]:
    n = int.from_bytes(b[off:off + 8], "big")
    return b[off + 8:off + 8 + n], off + 8 + n


def _events(b: bytes) -> list[tuple]:
    """A transcript's events (`FlockAudit.Transcript`): `("asked", chan, body, reply)` and `("drew", bytes)`."""
    out, off = [], 0
    while off < len(b):
        tag, off = b[off], off + 1
        if tag == 1:
            chan, off = _frame(b, off)
            body, off = _frame(b, off)
            reply, off = _frame(b, off)
            out.append(("asked", chan.decode(), body, reply))
        else:
            assert tag == 2, tag
            drew, off = _frame(b, off)
            out.append(("drew", drew))
    return out


def _fr(b: bytes) -> bytes:
    return len(b).to_bytes(8, "big") + b


def _encode(es: list[tuple]) -> bytes:
    return b"".join(bytes([1]) + _fr(e[1].encode()) + _fr(e[2]) + _fr(e[3]) if e[0] == "asked" else bytes([2]) + _fr(e[1])
                    for e in es)


@pytest.mark.slow
def test_the_honest_run_accepts_and_its_transcript_replays(zk, tmp_path):
    """The program draws the recorded session's units and accepts it, the serving side was asked for that draw, and the run's
    transcript replays to the same verdict (`replay_simulate`); a transcript whose last proof is not the object its key
    names is refused on replay as in a run. It is the one test of an accepting run through the workers, the frames and the
    content store (three Lean runs of the session's setup, two of them verifying)."""
    v, p = _run(zk, tmp_path)
    recorded = json.loads((_drawn(zk) / "session.json").read_text())["unit_draw"]
    assert v["accepted"], v["why"]
    assert v["draw"] == recorded and v["report"] == {"accepted": True, "report": "received"}
    asks = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert [a["ask"] for a in asks] == ["register", "session", "audit"]
    assert asks[1]["draw"] == recorded and asks[2]["verdict"]["accepted"]
    es = _events((tmp_path / "t.bin").read_bytes())
    assert [len(e[1]) for e in es if e[0] == "drew"] == [352]
    assert "1 coin requests" in p.stderr, p.stderr
    assert _replay(zk, tmp_path / "t.bin") == v
    i = max(i for i, e in enumerate(es) if e[0] == "asked" and e[1] == "archive")
    reply = es[i][3]
    es[i] = es[i][:3] + (reply[:-1] + bytes([reply[-1] ^ 1]),)
    (tmp_path / "flipped.bin").write_bytes(_encode(es))
    w = _replay(zk, tmp_path / "flipped.bin")
    assert not w["accepted"] and w["why"].startswith("archive: the object held under ") and w["why"].endswith("does not hash to it")


@pytest.mark.slow
def test_a_second_draw_cannot_be_asked_for(zk, tmp_path):
    """A serving worker that answers the auditor's draw with a request to draw again is refused: the program has no coin
    node after its draw (`auditor_draws_once`), so its run makes one coin request, and a transcript with a second draw does
    not replay."""
    v, p = _run(zk, tmp_path, "redraw")
    assert not v["accepted"] and v["why"] == "serving: record missing", v["why"]
    assert v["draw"] is not None
    es = _events((tmp_path / "t.bin").read_bytes())
    drew = [i for i, e in enumerate(es) if e[0] == "drew"]
    assert len(drew) == 1 and "1 coin requests" in p.stderr, p.stderr
    assert _replay(zk, tmp_path / "t.bin") == v
    es.insert(drew[0] + 1, ("drew", bytes(352)))
    (tmp_path / "twice.bin").write_bytes(_encode(es))
    w = _replay(zk, tmp_path / "twice.bin")
    assert not w["accepted"] and w["why"] == "replay: the program asks where the transcript does not", w["why"]


#: each dishonest answer and the reason the auditor refuses it
TAMPERED = {
    "draw": "U1: the record's unit draw is not the auditor's own",
    "hash": "archive: the object held under ",
    "proof": "S17: circuit/rep0: the proof is not the one the server received",
    "canon": "developer: the answer is not canonical JSON",
    "long": "run: frame: 1099511627776 bytes, over the request's bound of 1048576",
    "exit": "run: frame: the stream ended after 0 of 8 bytes",
}


@pytest.mark.parametrize("tamper", [pytest.param(t, marks=[] if t == "canon" else [pytest.mark.slow]) for t in TAMPERED])
def test_a_tampered_answer_is_refused(zk, tmp_path, tamper):
    """A record at another draw, an object that is not the one its key names, a flipped proof, a non-canonical answer, a
    frame over its request's bound (refused before its body is read) and a worker that exits: each refused by name."""
    v, _ = _run(zk, tmp_path, tamper)
    assert not v["accepted"] and v["why"].startswith(TAMPERED[tamper]), v["why"]
    assert (v["draw"] is None) == (tamper in ("canon", "long", "exit")), v


def test_the_archive_channel_is_the_auditors_own(built, tmp_path):
    """A worker that claims the content store's channel is refused before any process starts."""
    (tmp_path / "w.json").write_text(json.dumps({"workers": [{"name": "x", "channels": ["archive"], "argv": ["true"]}]}))
    p = subprocess.run([str(BIN / "flock-audit"), "run", *_own("0" * 128), "--archive", str(tmp_path),
                        "--workers", str(tmp_path / "w.json")], capture_output=True, text=True, check=False)
    assert p.returncode == 2 and "VERDICT" not in p.stdout, p.stdout
    assert p.stderr.startswith("--workers: worker x: the archive channel is the auditor's own"), p.stderr
