"""The bandwidth audit's piece (`served_bandwidth.sh`, BANDWIDTH_CMD) as `served_verdicts.py pieces` runs it, on the toy's
P2 store on the CPU (P2's reference decode) against `p2_party --listen` on loopback in the auditor gate's party shape:
the honest session's answer is a core accept that `piece_row` reads, with its credited reads, eps, δ and γ, and the
scope beside the served window; a prover that serves later passes from its cache of the first's, reading nothing of
the store, is rejected (`receipt`); and a session that left no record, asked about a window that wasn't served, is a
reject that says both. ``collect``, the composed run's piece, reads the answer the PoUS piece's session wrote, which
`pieces` has run first; it names the audit missing where there is none, and refuses another window's or protocol's. It
credits an accept only from the auditor's host, through its gate (`benchmarks/live_verifier/host.py`'s own, over a
directory standing for the host's root): the bandwidth party's own answer of its first session under the window's
binding must be an accept saying what the public answer does, and the session's record must be it. A session run under
window 1's audit, collected at window 2, is undecided and credits nothing, as is a forged record (another window's
audit, an accept the host never gave), a retry, a host with no answer, and a gate that refuses."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import sys
from fractions import Fraction
from pathlib import Path

import pytest

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks" / "one_stage"))
sys.path.insert(0, str(ROOT / "benchmarks" / "pous"))
sys.path.insert(0, str(ROOT / "examples" / "h200" / "interfaces"))
import served_bandwidth as SB  # noqa: E402
import served_verdicts as S  # noqa: E402
import served_window as SW  # noqa: E402
from p2_v1 import bandwidth as B  # noqa: E402
from verity_e2e_demo import block_record as BR  # noqa: E402

from verity.core.protocols.remote_bandwidth import p2, p2_party  # noqa: E402

CMD = 'bash benchmarks/one_stage/served_bandwidth.sh "$1" "$2" "$3"'


def _serve_dir(path: Path) -> Path:
    """A serve dir whose window 0 is sealed, its Commit passed over 3 steps, with one checked request."""
    path.mkdir()
    (path / "seals.jsonl").write_text(json.dumps({"window": 0, "requests": ["u0"]}) + "\n")
    (path / "windows.jsonl").write_text(json.dumps({"window": 0, "passed": True, "commit": {"steps": 3}}) + "\n")
    (path / "requests.jsonl").write_text(json.dumps({"user": "u0", "windows": [0], "check": {"passed": True}}) + "\n")
    return path


def _env(tmp: Path, cheat: str = "") -> dict[str, str]:
    toy = tmp / "toy"
    party = (f"cd {ROOT} && exec {sys.executable} -S -E -m verity.core.protocols.remote_bandwidth.p2_party "
             f"--device {tmp / 'party'} --listen 127.0.0.1:0 --registration {toy / 'registration.json'} "
             "--windows 1 --passes 2 --samples 64 --window-ns 1000000 --deadline-ns 1000000")
    return dict(os.environ) | {"BANDWIDTH_CMD": CMD, "PY": sys.executable, "BW_STORE": str(toy / "store"),
            "BW_LAYOUT": str(toy / "layout.json"), "BW_REGISTRATION": str(toy / "registration.json"),
            "BW_NODE": "p2-toy-node", "BW_DEVICE": "cpu", "BW_REF_DECODE": "1", "BW_LEASE_MIN": "0", "BW_WORKERS": "1",
            "BW_WINDOWS": "1", "BW_PASSES": "2", "BW_SAMPLES": "64", "BW_WINDOW_S": "0.001", "BW_DEADLINE_S": "0.001",
            "BW_PARTY": party, "BW_PARTY_HOST": "127.0.0.1", "BW_R_BYTES_PER_S": "1000000000", "BW_CHEAT": cheat}


def test_the_honest_session_is_a_core_accept_and_the_cache_cheat_a_reject(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(ROOT)
    reg = B.toy(tmp_path / "toy")
    serve = _serve_dir(tmp_path / "serve")

    got = S.pieces(serve, 0, tmp_path / "honest", env=_env(tmp_path))
    assert got["bandwidth"]["verdict"] == "accept", (tmp_path / "honest" / "bandwidth" / "piece.log").read_text()
    row = S.piece_row("bandwidth", tmp_path / "honest")
    a = row["answer"]
    assert row["verdict"] == "accept" and a["commitment"] == reg.digest.hex() and a["reason"] is None
    q = row["quantity"]
    assert q["q_bytes"] == (a["profile"]["receipts"] - a["profile"]["wrong"]) * p2.UNIT_BYTES > 0
    assert a["profile"]["receipts"] == 2 * reg.blocks and row["delta"] == str(Fraction(1, 2**40))
    assert Fraction(q["eps"]) == Fraction(1, 2**40) + Fraction(sum(a["profile"]["draws"]), 2**64)
    assert 0 < Fraction(q["gamma"]) == 1 - Fraction(q["q_bytes"] * 10**9, 10**9 * q["span_ns"]) < 1
    assert a["profile"]["scope"].startswith("beside the served window") and a["profile"]["gamma_terms"]
    assert BR._quantity("bandwidth", row)[0] == {"symbol": "Q_cert", "value": q["q_bytes"], "unit": "bytes"}
    served = json.loads((tmp_path / "honest" / "bandwidth" / "served.json").read_text())
    assert served["served"] == {"window": 0, "serve_dir": str(serve), "steps": [0, 3], "requests": 1}
    kept = [json.loads(x) for x in (tmp_path / "party" / "answers.jsonl").read_text().splitlines()]
    assert p2_party.core_answer(kept[-1], 10**9)["public"] == {k: q[k] for k in ("eps", "gamma", "q_bytes",
                                                                                  "r_bytes_per_s", "span_ns")}
    assert not (tmp_path / "honest" / "bandwidth" / "scratch").exists()

    got = S.pieces(serve, 0, tmp_path / "cheat", env=_env(tmp_path, "cache"))
    row = S.piece_row("bandwidth", tmp_path / "cheat")
    assert got["bandwidth"]["verdict"] == row["verdict"] == "reject"
    assert "receipt at window 0, draw [1, " in row["reason"] and "beside the served window" in row["reason"]
    assert row["quantity"] is None and row["answer"]["public"] is None


COLLECT = 'python benchmarks/one_stage/served_bandwidth.py collect --serve-dir "$1" --window "$2" --out "$3"'
HELD = {"commitment": "ab" * 64, "protocol": p2_party.PROTOCOL, "reason": None, "verdict": "accept",
        "public": {"eps": "1/1099511627776", "gamma": "1/2", "q_bytes": 8224, "r_bytes_per_s": 10**9, "span_ns": 16448},
        "profile": {"delta": "1/1099511627776", "draws": [0], "receipts": 1, "wrong": 0, "units": 1,
                    "scope": "beside the served window"}}
GATE = ROOT / "benchmarks" / "live_verifier"


def _own(audit: bytes | None, verdict: str = "accept", **kw) -> dict:
    """The bandwidth party's own answer (`p2_party.answer`, `session`'s ``host`` and ``audit``) of the session `HELD`
    is the public answer of: one unit read once, no draw rejected, over 16,448 ns."""
    own = {"protocol": p2_party.PROTOCOL, "registration": "cd" * 64, "verdict": verdict, "host": "auditor"}
    own |= {} if audit is None else {"audit": audit.hex()}
    if verdict == "accept":
        own |= {"receipts": 1, "wrong_at_most": 0, "delta": "1/1099511627776", "bound": {}, "draws": [0],
                "q_bytes": 8224, "span_ns": 16448, "windows": [{}], "units": 1, "plan_units": 1, "unit_bytes": 8224}
    return own | kw


def _host(tmp: Path, *answers: dict) -> str:
    """The auditor's host with `answers` in its bandwidth party's ``answers.jsonl`` (`p2_party --device`), and a case
    beside them: the command that reaches its gate, `host.gate` over that root under a read-only key."""
    root = tmp / "auditor"
    (root / "parties" / "bw" / "private").mkdir(parents=True, exist_ok=True)
    (root / "parties" / "bw" / "answers.jsonl").write_text("".join(json.dumps(a, sort_keys=True) + "\n" for a in answers))
    (root / "parties" / "bw" / "private" / "cases.jsonl").write_text(json.dumps({"commitment": "c4" * 32}) + "\n")
    gate = tmp / "gate.py"
    gate.write_text(f"import sys\nfrom pathlib import Path\nsys.path.insert(0, {str(GATE)!r})\nimport host\n"
                    f"sys.exit(host.gate(['--ssh', sys.argv[1], '--key', '-'], host.Host(root=Path({str(root)!r}))))\n")
    return shlex.join([sys.executable, str(gate)])


def _held(tmp: Path, window: int, answer: dict = HELD) -> Path:
    """What the PoUS piece's bandwidth session leaves in the bandwidth piece's directory (`held_bandwidth.session`)."""
    d = tmp / "held"
    d.mkdir(exist_ok=True)
    (d / "answer.json").write_text(json.dumps(answer))
    (d / "bandwidth.json").write_text(json.dumps({"window": window, "verdict": answer["verdict"]}))
    return d


def _served(serve: Path) -> Path:
    """`serve` with the serve.json the window's binding is of (`served_window.of_serve_dir`)."""
    serve.mkdir(parents=True, exist_ok=True)
    (serve / "serve.json").write_text(json.dumps({"model": "meta-llama/Llama-3.2-1B", "revision": "4e20de36" * 5}))
    return serve


def _record(private: Path, own: dict | None) -> Path:
    """The session's record the PoUS piece's bandwidth session keeps (``session.json``): the bandwidth party's own answer
    `own` as the prover relayed it."""
    private.mkdir(parents=True, exist_ok=True)
    (private / "session.json").write_text(json.dumps({"answer": own or {"verdict": "accept"}, "verifier_record": own,
                                                      "verifier_rc": 0}))
    return private


def _composed(tmp: Path, serve: Path, window: int, audit_window: int) -> tuple[dict, dict]:
    """`pieces` at `window`, its PoUS piece leaving an accepted bandwidth answer of `window` whose session ran under
    window `audit_window`'s audit, as the auditor's host kept it: what `pieces` printed of the bandwidth piece, and its
    row."""
    own = _own(SW.of_serve_dir(serve, audit_window))
    held, kept, gate = _held(tmp, window), _record(tmp / "kept", own), _host(tmp, own)
    pous = (f'mkdir -p "$3/../bandwidth" "$1/private/bandwidth" && cp {held}/*.json "$3/../bandwidth/" && '
            f'cp {kept}/session.json "$1/private/bandwidth/"')
    collect = f"{COLLECT.replace('python', sys.executable, 1)} --gate {shlex.quote(gate)}"
    env = dict(os.environ) | {"POUS_CMD": pous, "BANDWIDTH_CMD": collect}
    out = tmp / f"pieces-{window}-{audit_window}"
    return S.pieces(serve, window, out, env=env)["bandwidth"], S.piece_row("bandwidth", out)


def test_the_pous_piece_runs_first_and_the_bandwidth_piece_collects_what_its_session_wrote(tmp_path, monkeypatch):
    """`pieces` starts PoUS's piece before the bandwidth audit's, so the answer PoUS's piece wrote to the bandwidth
    piece's directory, and the session's record in ``SERVE_DIR/private/bandwidth``, are there when ``collect`` reads
    them; the auditor's host kept the party's own accept of the window's session, which the public answer agrees with and
    the record is, so the record reads the public answer, as it was written, as the bandwidth audit's."""
    monkeypatch.chdir(ROOT)
    serve = _served(_serve_dir(tmp_path / "serve"))
    got, row = _composed(tmp_path, serve, 0, 0)
    assert got == {"verdict": "accept", "rc": 0}
    log = (tmp_path / "pieces-0-0" / "bandwidth" / "piece.log").read_text()
    assert log.startswith("BANDWIDTH ") and json.loads(log.split(" ", 1)[1])["public"] == HELD["public"]
    assert json.loads((tmp_path / "pieces-0-0" / "bandwidth" / "answer.json").read_text()) == HELD
    assert row["verdict"] == "accept" and row["quantity"]["q_bytes"] == 8224
    assert BR._quantity("bandwidth", row)[0] == {"symbol": "Q_cert", "value": 8224, "unit": "bytes"}


def test_a_session_under_an_earlier_windows_audit_is_undecided_and_credits_nothing(tmp_path, monkeypatch):
    """The prover names window 1's audit at window 2, so the bandwidth party held window 1's PoUS store and accepted:
    the answer is window 2's, but the auditor's host kept no answer of window 2's PoUS session, so the bandwidth piece
    is undecided and credits nothing, naming why and neither binding. The same session collected at window 1, whose
    audit it ran under, is credited."""
    monkeypatch.chdir(ROOT)
    serve = _served(_serve_dir(tmp_path / "serve"))
    got, row = _composed(tmp_path, serve, 2, 1)
    assert got["verdict"] == row["verdict"] == "undecided" and got["rc"] == 1
    assert row["quantity"] is None and row["answer"]["public"] is None and row["answer"]["commitment"] is None
    assert "is not of window 2's PoUS session: the auditor's host kept no answer of a bandwidth session under window " \
           "2's binding" in row["reason"]
    text = (tmp_path / "pieces-2-1" / "bandwidth" / "answer.json").read_text()
    assert all(SW.of_serve_dir(serve, k).hex() not in text for k in (1, 2))
    assert _composed(tmp_path, serve, 1, 1)[0] == {"verdict": "accept", "rc": 0}


NOT_KEPT = "the auditor's host kept no answer of a bandwidth session under window 3's binding"
ANOTHER = "the auditor's host kept another answer than this accept for window 3's session"
NOT_HOSTS = "the session's record of the bandwidth party's answer is not the one the auditor's host kept"


@pytest.mark.parametrize("case, why", [
    ("a record of another window's audit", NOT_HOSTS),
    ("an accept the host never gave", ANOTHER),
    ("more reads than the host credited", ANOTHER),
    ("a retry after a reject", ANOTHER),
    ("no answer on the host", NOT_KEPT),
    ("only another window's on the host", NOT_KEPT),
    ("only a fixture session's on the host", NOT_KEPT),
    ("no record", "there is no record of the session in "),
    ("a record of no answer", NOT_HOSTS),
    ("a gate that refuses", "the auditor's host gave no answers of its bandwidth party"),
    ("no serve.json", "window 3's binding can't be derived: "),
])
def test_an_accept_the_auditors_host_did_not_keep_for_the_window_is_undecided_naming_no_value(tmp_path, case, why):
    """A public accept and the prover's record of it (``session.json``), forged or of another session, against what the
    auditor's host kept: only the host's first answer under the window's binding credits, and only when the public answer
    says what it does and the record is it. The undecided answer names neither a binding nor what the host kept."""
    serve, out = tmp_path / "serve", _held(tmp_path, 3)
    if case != "no serve.json":
        _served(serve)
    bind = SW.of_serve_dir(serve, 3) if case != "no serve.json" else hashlib.sha512(b"some window's binding").digest()
    other = SW.of_serve_dir(serve, 2) if case != "no serve.json" else hashlib.sha512(b"another").digest()
    reject = _own(bind, "reject", failure="receipt", window=0, at=[1, 2])
    honest = _own(bind)
    kept, record = {
        "a record of another window's audit": ([honest], _own(other)),
        "an accept the host never gave": ([reject], honest),
        "more reads than the host credited": ([_own(bind, q_bytes=4112)], honest),
        "a retry after a reject": ([reject, honest], honest),
        "no answer on the host": ([], honest),
        "only another window's on the host": ([_own(other)], _own(other)),
        "only a fixture session's on the host": ([_own(None)], _own(None)),
        "a record of no answer": ([honest], None),
    }.get(case, ([honest], honest))
    gate = "false" if case == "a gate that refuses" else _host(tmp_path, *kept)
    if case != "no record":
        _record(serve / "private" / "bandwidth", record)
    assert SB.main(["collect", "--serve-dir", str(serve), "--window", "3", "--out", str(out), "--gate", gate]) == 1
    text = (out / "answer.json").read_text()
    a = json.loads(text)
    assert a["verdict"] == "undecided" and S.check_answer(a) and a["public"] is None and a["commitment"] is None
    assert a["reason"].startswith(f"the bandwidth answer in {out} is not of window 3's PoUS session: {why}")
    assert all(v not in text for v in (bind.hex(), other.hex(), "cd" * 64, "c4" * 32, "4112"))


def test_collect_reads_the_record_where_the_pous_piece_kept_it_and_passes_any_other_verdict(tmp_path):
    """The record is read from ``--private`` when the PoUS piece kept it elsewhere (its ``--bw-private``); an answer
    that credits nothing keeps its own reason, and asks the auditor's host nothing."""
    serve = _served(tmp_path / "serve")
    own = _own(SW.of_serve_dir(serve, 3))
    elsewhere, gate = _record(tmp_path / "elsewhere", own), _host(tmp_path, own)
    out = _held(tmp_path, 3)
    assert SB.main(["collect", "--serve-dir", str(serve), "--window", "3", "--out", str(out),
                    "--private", str(elsewhere), "--gate", gate]) == 0
    assert json.loads((out / "answer.json").read_text()) == HELD
    rejected = {"commitment": None, "profile": None, "protocol": p2_party.PROTOCOL, "public": None,
                "reason": "the session was rejected: receipt at window 0, draw [1, 2]", "verdict": "reject"}
    out = _held(tmp_path, 3, rejected)
    assert SB.main(["collect", "--serve-dir", str(serve), "--window", "3", "--out", str(out), "--gate", "false"]) == 1
    assert json.loads((out / "answer.json").read_text()) == rejected


def test_collect_names_the_audit_missing_where_the_pous_piece_wrote_nothing(tmp_path, capsys):
    out = tmp_path / "pieces" / "bandwidth"
    assert SB.main(["collect", "--serve-dir", str(tmp_path), "--window", "3", "--out", str(out)]) == 2
    assert capsys.readouterr().out.startswith("MISSING the bandwidth audit: the PoUS piece wrote no bandwidth answer")
    assert not (out / "answer.json").exists()
    row = S.piece_row("bandwidth", tmp_path / "pieces")
    assert row["verdict"] == "missing" and row["owner"] == S.PIECES["bandwidth"][2]


def test_collect_replaces_another_windows_answer_and_errs_on_another_protocols(tmp_path, capsys):
    out = _held(tmp_path, 2)
    assert SB.main(["collect", "--serve-dir", str(tmp_path), "--window", "3", "--out", str(out)]) == 1
    a = json.loads((out / "answer.json").read_text())
    assert a["verdict"] == "undecided" and "is window 2's, not window 3's" in a["reason"] and S.check_answer(a)
    out = _held(tmp_path, 3, HELD | {"protocol": "verity/pous/p2/v4/run"})
    assert SB.main(["collect", "--serve-dir", str(tmp_path), "--window", "3", "--out", str(out)]) == 1
    assert "not the remote bandwidth audit's" in capsys.readouterr().out.splitlines()[-1]


def test_a_session_with_no_record_is_a_reject_that_says_its_window_was_not_served(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(ROOT)
    out = tmp_path / "pieces" / "bandwidth"
    assert SB.main(["answer", "--session", str(out / "session" / "session.json"), "--serve-dir", str(tmp_path / "none"),
                    "--window", "0", "--out", str(out)]) == 1
    row = S.piece_row("bandwidth", tmp_path / "pieces")
    assert row["verdict"] == "reject" and row["quantity"] is None and "no session record" in row["reason"]
    assert "with no served window: window 0 of " in row["reason"] and "beside the served window" not in row["reason"]
    assert json.loads((out / "served.json").read_text())["scope"].startswith("with no served window")


def test_the_served_pieces_request_is_the_auditors_session() -> None:
    """`served_bandwidth.sh` requests, by default, the session the auditor's party holds every session to
    (`p2_party.SESSION`, the held session's in the PoUS piece), as `bandwidth.py session` turns its flags into the
    request: one off it is refused before any coin."""
    text = (ROOT / "benchmarks" / "one_stage" / "served_bandwidth.sh").read_text()
    got = dict(re.findall(r"\$\{BW_(WINDOWS|PASSES|SAMPLES|WINDOW_S|DEADLINE_S):-([0-9.]+)\}", text))
    assert {"windows": int(got["WINDOWS"]), "passes": int(got["PASSES"]), "samples": int(got["SAMPLES"]),
            "window_ns": int(float(got["WINDOW_S"]) * 1e9),
            "deadline_ns": int(float(got["DEADLINE_S"]) * 1e9)} == p2_party.SESSION
