"""The coin tree's opening checks in Lean (`Flock/Firewall/Coins.lean`, `flock-firewall coins`), the verdict a firewalled
session loop takes for every coin it hands on, on `live/src/coin_open_vectors.json`, which coin_tree.rs's test holds
Rust's `Checked` (an unfirewalled `--zk` prover's) to: docs/coin-tree-v2.md §9's tree, its
honest openings and altered ones (a bit of a coin, of `T_n`, of the salt or of a path node; a word short or long; another
round's, stream's or count's opening; past the block or the schedule; another nonce, key or root; a malformed answer or
`Hello`). Lean hands on the coins Rust does and refuses with Rust's reason, word for word. The session stream's count is
`Hello`'s link points, whatever the request says, and nothing after a refusal is checked: the process ends there."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

LEAN = Path(__file__).resolve().parents[3] / "core" / "proof_system" / "flock"
BIN = LEAN / ".lake" / "build" / "bin" / "flock-firewall"
VECTORS = Path(__file__).resolve().parents[1] / "live" / "src" / "coin_open_vectors.json"
CASES = {c["case"]: c["steps"] for c in json.loads(VECTORS.read_text())["cases"]}


@pytest.fixture(scope="module", autouse=True)
def built():
    if shutil.which("lake") is None:
        pytest.skip("no Lean toolchain")
    subprocess.run(["lake", "build", "flock-firewall"], cwd=LEAN, check=True, capture_output=True)


def _request(step: dict) -> bytes:
    if "hello" in step:
        b = step["hello"].encode()
        return f"hello {len(b)}\n".encode() + b
    if "answer" in step:
        b = bytes.fromhex(step["answer"])
        return f"answer {len(b)}\n".encode() + b
    b = bytes.fromhex(step["frame"])
    return f"open {step['open']} {step['n']} {len(b)}\n".encode() + b


def _run(requests: list[bytes]) -> tuple[list[dict], int]:
    """Every request at once; Lean's answers, each `{}` (`ok`), `{"coins": HEX}` (`ok N` and the coins), `{"stop": WHY}`
    or `{"err": WHY}` (`err WHY`), and its exit code."""
    p = subprocess.run([str(BIN), "coins"], input=b"".join(requests), capture_output=True)
    out, at, answers = p.stdout, 0, []
    while at < len(out):
        end = out.index(b"\n", at)
        line, at = out[at:end].decode(), end + 1
        if line == "ok":
            answers.append({})
        elif line.startswith("ok "):
            n = 16 * int(line[3:])
            answers.append({"coins": out[at:at + n].hex()})
            at += n
        elif line.startswith("stop "):
            answers.append({"stop": line[5:]})
        else:
            answers.append({"err": line.removeprefix("err ")})
    return answers, p.returncode


def _rust(steps: list[dict]) -> list[dict]:
    """What Rust's `Checked` answered each step."""
    return [{k: s[k] for k in ("coins", "stop") if k in s} for s in steps]


@pytest.mark.parametrize("case", list(CASES))
def test_lean_answers_each_step_as_rust_does(case):
    steps = CASES[case]
    assert _run([_request(s) for s in steps]) == (_rust(steps), 0), case


def test_the_vectors_cover_honest_and_refused_openings():
    stops = [s[-1]["stop"] for s in CASES.values() if "stop" in s[-1]]
    assert len(stops) == len(CASES) - 2 and all(w.startswith("PROVER-STOPPED: ") for w in stops)
    assert sum("do not open its step-0 commitment" in w for w in stops) >= 10
    assert sum(1 for s in CASES["honest"] if "coins" in s) == 7


@pytest.mark.parametrize("case", [c for c, s in CASES.items() if "stop" in s[-1]])
def test_nothing_after_a_refusal_is_checked(case):
    """A refused case followed by the honest openings: Lean answers nothing after its refusal and ends the session."""
    steps = CASES[case]
    more = [_request(s) for s in CASES["honest"] if "open" in s]
    assert _run([_request(s) for s in steps] + more) == (_rust(steps), 0), case


@pytest.mark.parametrize("n", [0, 1, 7])
def test_the_session_streams_count_is_hellos(n):
    """The session stream's opening is of `Hello`'s link points (2 here), whatever count the request names."""
    steps = [dict(s, n=n) if s.get("open") == "session" else s for s in CASES["honest"]]
    assert _run([_request(s) for s in steps]) == (_rust(CASES["honest"]), 0)


def test_a_frame_rust_never_sends_is_refused():
    """The opening is read as the wire frames it (`Resp::Coins`: tag 2, the word count, the words): any other frame of the
    same bytes is refused."""
    hello, answer, first = CASES["honest"][:3]
    frame = bytes.fromhex(first["frame"])
    for bad in (b"\x03" + frame[1:], frame[:1] + bytes([frame[1] + 1]) + frame[2:], frame + b"\x00"):
        got, rc = _run([_request(hello), _request(answer), _request(dict(first, frame=bad.hex()))])
        assert rc == 0 and got[:2] == [{}, {}] and got[2]["stop"].startswith("PROVER-STOPPED: coin opening of session round 0:")


@pytest.mark.parametrize("requests,why", [
    (lambda h, a: [_request(a)], "Hello's answer before the Hello"),
    (lambda h, a: [_request(h), _request(h)], "a second Hello"),
    (lambda h, a: [_request(h), _request(a), _request(a)], "a second answer to Hello"),
    (lambda h, a: [b"open session\n"], "a request `open session`"),
])
def test_a_request_out_of_order_is_an_error(requests, why):
    hello, answer = CASES["honest"][:2]
    got, rc = _run(requests(hello, answer))
    assert rc == 1 and got[-1] == {"err": why}
