"""The Lean verifier's reading of the inner session's transcript (`RecVStar.innerCommits`, `verify --rec-inner-transcript`,
`flock-verify rec-transcript`) on candidate 3's whole inner session: the fetched fixture `rec-c3-inner` is the coin server's
record (`public.json`) and the prover's record (`prover/session.json`, the one check 5 reads) of `r20261012-081440-2d61`
(`pod/88-rec-c3-inner.sh` at Shape(25,22,8), N=16, SPARSE=18,13,6,0 STACKED=1, with the relay keeping the prover's
Hello, R's top registered at staging, R's shape in META and each sparse phase's claim its last round): the circuit table's two reps (95 rounds each in the transcript, 94 in the prover's
record, which keeps z_partial's round 40 whole), both reps' sparse phases (34 each) and the stacked table R (36), whose root
is no Hello table's but the top its stream's binding round commits. The reps' coins are checked against the prover's record,
the sparse streams and R's are the coin server's alone; `session/session.json` is the relay's record, which is refused. The
verifier holds every stream's `c_k` and tops, and refuses the session with a stream missing, a top another stream
committed, or its table named as a sparse phase's or R's stream (`sparse`, `stacked`). R has no root at `Commit`, so
its salted top is registered at staging (#1798: the entry `stacked` of the
statement's `verity/registration/v2` circuit registration, `circuit-registration.json`), whose digest the inner circuit's
META binds (`circuit-meta.txt`, the staged `circuit.txt`'s header): the verifier takes both (`--rec-inner-circuit`,
`--rec-inner-registration`) and refuses, each with its `FIREWALL-REFUSED` reason, a record META does not bind, a session
with R's stream and no registered top, and R's binding round committing a top other than the registered one; a
registration naming R needs R's stream. R's `m` and claims are META's `stacked`, the shape staging committed R at (the
registered cap fixes neither: m 22 to 24 share it), and R's stream is held to that shape's rounds: a META naming no shape or
another, and R's stream opened at another `m` over the registered cap, are refused by name. The plain session (`rec-coins`' inner session) is read as before, its two reps'
`c_k` and nothing else."""
from __future__ import annotations

import copy
import hashlib
import json
import re
import subprocess
from pathlib import Path

import pytest
from test_rec_outer import BIN

HERE = Path(__file__).resolve().parent
C3 = HERE / "fixtures" / "rec-c3-inner"
PLAIN = HERE / "fixtures" / "rec-coins" / "16-16-8-n64" / "inner-honest"

pytestmark = pytest.mark.skipif(not BIN.exists(), reason="the Lean verifier is not built")

#: what the verifier holds of candidate 3's session: each stream's message rounds and its tops' node counts by level (level 0
#: its table's root_B, R's the top its binding round commits; the sparse streams belong to no table)
C3_STREAMS = [{"commits": 77, "stream": "circuit/rep0", "tops": [128, 64, 64, 32]},
              {"commits": 77, "stream": "circuit/rep1", "tops": [128, 64, 64, 32]},
              {"commits": 33, "stream": "sparse/rep0", "tops": []},
              {"commits": 33, "stream": "sparse/rep1", "tops": []},
              {"commits": 25, "stream": "stacked/rep0", "tops": [128, 64, 64]}]


def _fixture(d: Path) -> Path:
    assert (d / "public.json").is_file(), (
        f"{d} is not fetched: `uv run research data fetch-fixtures {d.relative_to(HERE.parents[3])}`")
    return d


def _inner(circuit: Path | None = None, registration: Path | None = None) -> list[str]:
    """The inner circuit's header and its circuit registration, 2d61's when not given."""
    return ["--rec-inner-circuit", str(circuit or _fixture(C3) / "circuit-meta.txt"),
            "--rec-inner-registration", str(registration or C3 / "circuit-registration.json")]


def _meta_registration() -> str:
    return json.loads((_fixture(C3) / "circuit-meta.txt").read_text().splitlines()[1][5:])["registration"]


def _registered(tmp_path: Path, change, bind: bool = True) -> list[str]:
    """2d61's circuit registration changed by ``change``, and its circuit header with META's `registration` the changed
    record's digest when ``bind`` (as a statement staged with that record would be), 2d61's own otherwise."""
    from verity.core.protocols.sampled_proofs.one_stage.registration import digest

    rec = json.loads((_fixture(C3) / "circuit-registration.json").read_text())
    change(rec)
    (tmp_path / "circuit-registration.json").write_text(json.dumps(rec))
    head, meta = (C3 / "circuit-meta.txt").read_text().splitlines()
    m = json.loads(meta[5:])
    if bind:
        m["registration"] = digest(rec)
    (tmp_path / "circuit-meta.txt").write_text(f"{head}\nMETA {json.dumps(m)}\n")
    return _inner(tmp_path / "circuit-meta.txt", tmp_path / "circuit-registration.json")


def _write(path: Path, j: dict) -> Path:
    path.write_text(json.dumps(j))
    return path


HONEST = object()


def _read(d: Path, transcript: Path | None = None, record: Path | None = None,
          inner: list[str] | object = HONEST) -> tuple[int, dict | None, str]:
    record = record or (d / "prover" if (d / "prover").is_dir() else d / "session")
    if inner is HONEST:
        inner = _inner() if d == C3 else []
    p = subprocess.run([str(BIN), "rec-transcript", "--rec-inner-transcript", str(transcript or d / "public.json"),
                        "--rec-inner-session", str(record), *inner], capture_output=True, text=True)
    line = [json.loads(ln[14:]) for ln in p.stdout.splitlines() if ln.startswith("RECTRANSCRIPT ")]
    return p.returncode, (line[0] if line else None), p.stderr.strip()


def _changed(tmp_path: Path, change) -> Path:
    t = json.loads((_fixture(C3) / "public.json").read_text())
    change(t)
    f = tmp_path / "transcript.json"
    f.write_text(json.dumps(t))
    return f


def _round(t: dict, stream: str, tops: bool = False, k: int | None = None) -> dict:
    return next(r for r in sorted(t["rounds"], key=lambda r: r["k"])
                if r["stream"] == stream and (r["tops"] or not tops) and k in (None, r["k"]))


def test_the_verifier_holds_every_stream_of_candidate_3s_session():
    rc, line, err = _read(_fixture(C3))
    assert (rc, line) == (0, {"reps": [77, 77], "streams": C3_STREAMS}), err


def test_verify_reads_candidate_3s_transcript(tmp_path):
    """`verify --rec-inner-transcript` reads the transcript among its inputs, before it sets anything up: on candidate 3's
    session it goes on to the statement (here none, so the session is refused for its files), and on the session with a
    stream missing it stops there (exit 2). The inner statement is its circuit's header (whose META binds R's registration)
    beside stand-ins for its public input and partition, which nothing reads before the session."""
    (tmp_path / "partition.json").write_text("{}")
    (tmp_path / "empty").write_text("")

    def verify(transcript: Path) -> subprocess.CompletedProcess:
        return subprocess.run([str(BIN), "verify", "--zk", "--circuit", str(tmp_path / "empty"), "--public", str(tmp_path / "empty"),
                               "--rec-inner-public", str(tmp_path / "empty"), "--rec-inner-partition",
                               str(tmp_path / "partition.json"), "--rec-inner-session", str(C3 / "prover"),
                               "--rec-inner-transcript", str(transcript), *_inner(),
                               "--session", str(tmp_path / "none")],
                              capture_output=True, text=True)

    p = verify(_fixture(C3) / "public.json")
    assert p.returncode == 1 and "rec inner transcript" not in p.stderr, p.stderr
    assert any(ln.startswith("VERDICT ") and "session files" in ln for ln in p.stdout.splitlines()), p.stdout
    bad = _changed(tmp_path, lambda t: t.__setitem__("rounds", [r for r in t["rounds"] if r["stream"] != "sparse/rep1"]))
    p = verify(bad)
    assert p.returncode == 2 and p.stderr.startswith("rec inner transcript: its message rounds' streams"), p.stderr


STREAMS = "[circuit/rep0, circuit/rep1, sparse/rep0, sparse/rep1, stacked/rep0]"
FOUR = "[circuit/rep0, circuit/rep1, sparse/rep0, sparse/rep1]"


def _drop_stacked(t: dict) -> None:
    t["rounds"] = [r for r in t["rounds"] if r["stream"] != "stacked/rep0"]


def _drop_stacked_everywhere(t: dict) -> None:
    _drop_stacked(t)
    del t["streams"]["stacked/rep0"]


def _drop_sparse_round(t: dict) -> None:
    t["rounds"].remove(_round(t, "sparse/rep0"))


@pytest.mark.parametrize("change,why", [
    (_drop_stacked, f"it declares the streams {STREAMS}, not {FOUR}"),
    (lambda t: t.__setitem__("rounds", [r for r in t["rounds"] if r["stream"] != "sparse/rep1"]),
     "its message rounds' streams [circuit/rep0, circuit/rep1, sparse/rep0, stacked/rep0] are not one table's rep0 and rep1, "
     f"both reps' sparse phases and at most R's stream ({STREAMS})"),
    (_drop_sparse_round, "sparse/rep0 declares 34 rounds and records 33"),
], ids=["stacked-rounds", "sparse-rep1", "sparse-round"])
def test_a_stream_missing_is_refused(tmp_path, change, why):
    rc, line, err = _read(C3, _changed(tmp_path, change))
    assert (rc, line, err) == (1, None, f"rec inner transcript: {why}")


def test_the_sparse_streams_and_rs_are_the_coin_servers_alone(tmp_path):
    """The prover's record holds the reps only, so whether the session committed R is the coin server's record's to say,
    held to the registration: a transcript with R's stream gone from its rounds and its declarations is read as a session
    without R when no registration names R, and refused as a downgrade when the one META binds does."""
    bad = _changed(tmp_path, _drop_stacked_everywhere)
    rc, line, err = _read(C3, bad, inner=[])
    assert (rc, line) == (0, {"reps": [77, 77], "streams": C3_STREAMS[:4]}), err
    rc, line, err = _read(C3, bad)
    assert (rc, line, err) == (1, None, "rec inner transcript: the registration names R's top (stacked), and the session "
                                        "has no stacked/rep0: R's stream is missing (a downgrade)")


def _stacked(reg: dict) -> dict:
    return next(e for e in reg["registered"] if e["value"] == "stacked")


def _flip(x: str) -> str:
    c = bytearray.fromhex(x)
    c[-1] ^= 1
    return c.hex()


CAP = "FIREWALL-REFUSED: registration: stacked/rep0's binding round's top is not stacked's registered top"
NO_TOP = "FIREWALL-REFUSED: registration: stacked has no root at Commit and no registered top"


def _top(change):
    def transcript(t: dict) -> None:
        change(_round(t, "stacked/rep0", k=0)["tops"][0])
    return transcript


@pytest.mark.parametrize("change", [
    _top(lambda top: top.__setitem__(0, _flip(top[0]))),
    _top(lambda top: top.__setitem__(5, _flip(top[5]))),
    _top(lambda top: top.__setitem__(127, _flip(top[127]))),
    _top(lambda top: top.reverse()),
], ids=["node-0", "node-5", "node-127", "reversed"])
def test_a_cap_other_than_the_registered_top_is_refused(tmp_path, change):
    """The registration fixed at staging, R's binding round commits another top: one bit off in its first, a middle or its
    last node, or its nodes in another order."""
    assert _read(C3, _changed(tmp_path, change)) == (1, None, CAP)


@pytest.mark.parametrize("change", [
    lambda r: _stacked(r).__setitem__("root", _flip(_stacked(r)["root"])),
    lambda r: _stacked(r).__setitem__("domain", _flip(_stacked(r)["domain"])),
    lambda r: _stacked(r).__setitem__("leaves", 64),
    lambda r: _stacked(r).__setitem__("words", 32),
    lambda r: _stacked(r).__setitem__("word_bits", 32),
    lambda r: _stacked(r).__delitem__("root"),
    lambda r: _stacked(r).__delitem__("leaves"),
    lambda r: _stacked(r).__delitem__("domain"),
    lambda r: r.__setitem__("program", "cd" * 64),
], ids=["root", "domain", "leaves", "words", "word-bits", "no-root", "no-leaves", "no-domain", "program"])
def test_a_registered_entry_the_top_does_not_make_is_refused(tmp_path, change):
    """A record META binds whose entry `stacked` is not the one R's binding round's top makes: its root, domain, leaf count
    or row width another or absent, or its program another (the domain is the program's, derived, then checked)."""
    assert _read(C3, inner=_registered(tmp_path, change)) == (1, None, CAP)


def test_the_registered_entry_is_the_one_python_rebuilds(tmp_path):
    """Lean's entry over R's top (`RecVStar.topEntry`) is `rec_live.top_entry`'s under any program: the record under another
    program, its entry rebuilt for that program (so another domain), and META binding it, is accepted."""
    from verity_flock.rec_live import top_entry

    top = [bytes.fromhex(b) for b in _round(json.loads((C3 / "public.json").read_text()), "stacked/rep0", k=0)["tops"][0]]
    honest = json.loads((C3 / "circuit-registration.json").read_text())
    assert top_entry(honest["program"], "stacked", top) == _stacked(honest)

    def change(r):
        r["program"] = "cd" * 64
        r["registered"] = [top_entry(r["program"], "stacked", top)]

    rc, line, err = _read(C3, inner=_registered(tmp_path, change))
    assert (rc, line) == (0, {"reps": [77, 77], "streams": C3_STREAMS}), err


@pytest.mark.parametrize("change", [
    lambda r: r.__setitem__("registered", []),
    lambda r: r.__setitem__("registered", [{**_stacked(r), "value": "weights"}]),
    lambda r: r["registered"].append(dict(_stacked(r))),
    lambda r: r.__setitem__("registered", {"stacked": _stacked(r)}),
    lambda r: r.__delitem__("registered"),
    lambda r: r.__delitem__("program"),
    lambda r: r.__setitem__("format", "verity/registration/v1"),
], ids=["none", "another-value", "twice", "not-a-list", "no-registered", "no-program", "v1"])
def test_an_r_stream_with_no_stacked_entry_is_refused(tmp_path, change):
    """A record META binds that registers no top of R (no entry `stacked`, or none it can be read from): R's stream binds R
    to nothing before the coins."""
    assert _read(C3, inner=_registered(tmp_path, change)) == (1, None, NO_TOP)


def test_an_r_stream_with_no_registration_is_refused():
    """With neither the inner circuit nor its registration, nothing binds R before the coins."""
    assert _read(C3, inner=[]) == (1, None, "FIREWALL-REFUSED: registration: stacked/rep0 binds R, and the inner circuit's "
                                            "META binds no circuit registration (--rec-inner-circuit, --rec-inner-registration)")


@pytest.mark.parametrize("change", [
    lambda r: r["window"].__setitem__("kind", "another"),
    lambda r: r.__setitem__("population", 17),
    lambda r: _stacked(r).__setitem__("root", _flip(_stacked(r)["root"])),
    lambda r: r.__setitem__("registered", []),
], ids=["window", "population", "root", "no-stacked"])
def test_a_record_whose_digest_is_not_metas_is_refused(tmp_path, change):
    """The record changed and META left as staged: whatever the change, the record is not the one META binds."""
    why = f"FIREWALL-REFUSED: registration: the record is not the circuit registration META binds ({_meta_registration()})"
    assert _read(C3, inner=_registered(tmp_path, change, bind=False)) == (1, None, why)


def test_meta_and_the_record_go_together(tmp_path):
    """A record with a META binding none is refused, and so is a META binding a record the verifier does not hold."""
    head, meta = (_fixture(C3) / "circuit-meta.txt").read_text().splitlines()
    m = json.loads(meta[5:])
    del m["registration"]
    (tmp_path / "circuit-meta.txt").write_text(f"{head}\nMETA {json.dumps(m)}\n")
    assert _read(C3, inner=_inner(tmp_path / "circuit-meta.txt")) == (
        1, None, "FIREWALL-REFUSED: registration: the record is not the circuit registration META binds (META binds none)")
    assert _read(C3, inner=["--rec-inner-circuit", str(C3 / "circuit-meta.txt")]) == (
        1, None, f"FIREWALL-REFUSED: registration: META binds the circuit registration {_meta_registration()}, and the "
                 "verifier holds none (--rec-inner-registration)")


#: R's shape as staging committed R (META's `stacked`, `rec_session.stage_stacked`): `rec_sparse.table_m` at the bucket
#: 18,13,6,0 and the two reps' claims
R_SHAPE = {"m": 23, "claims": 2}
SHAPE = ("FIREWALL-REFUSED: registration: stacked/rep0 is not R's stream at the shape staging committed R at (m {m}, claims "
         "{claims}, META's stacked): ")


def _meta(tmp_path: Path, change) -> list[str]:
    """The inner circuit's header with META changed by ``change`` (its `registration` left as staged), and the record."""
    head, meta = (_fixture(C3) / "circuit-meta.txt").read_text().splitlines()
    m = json.loads(meta[5:])
    change(m)
    (tmp_path / "circuit-meta.txt").write_text(f"{head}\nMETA {json.dumps(m)}\n")
    return _inner(tmp_path / "circuit-meta.txt")


def test_rs_shape_is_metas():
    """META carries R's shape, and R's stream is R's at it round for round (`RecVStar.stackedRounds`, Python's
    `rec_live.stacked_rounds`)."""
    from verity_flock.rec_live import stacked_rounds

    m = json.loads((_fixture(C3) / "circuit-meta.txt").read_text().splitlines()[1][5:])
    assert m["stacked"] == R_SHAPE
    rs = sorted((r for r in json.loads((C3 / "public.json").read_text())["rounds"] if r["stream"] == "stacked/rep0"),
                key=lambda r: r["k"])
    assert [(len(r["coins"]), [len(t) for t in r["tops"]], r["level0"], bool(r["commits"])) for r in rs] == [
        (n, tops, level0, bool(absorbs)) for n, tops, level0, absorbs in stacked_rounds(R_SHAPE["m"])]


@pytest.mark.parametrize("change,why", [
    (lambda m: m.__delitem__("stacked"), "FIREWALL-REFUSED: registration: stacked/rep0 binds R, and the inner circuit's META "
     "names no shape of R (stacked): R's m and claims are bound nowhere"),
    (lambda m: m.__setitem__("stacked", {"m": 22, "claims": 2}), SHAPE.format(m=22, claims=2) +
     "round 3 is (coins 16, tops [], level0 1, commits true), not (coins 15, tops [], level0 1, commits true)"),
    (lambda m: m.__setitem__("stacked", {"m": 24, "claims": 2}), SHAPE.format(m=24, claims=2) +
     "round 3 is (coins 16, tops [], level0 1, commits true), not (coins 17, tops [], level0 1, commits true)"),
    (lambda m: m.__setitem__("stacked", {"m": 23, "claims": 3}), SHAPE.format(m=23, claims=3) +
     "R is opened at the two reps' sparse claims, not 3"),
    (lambda m: m.__setitem__("stacked", {"m": 40, "claims": 2}), SHAPE.format(m=40, claims=2) +
     "no fast100 schedule at m 40"),
    (lambda m: m.__setitem__("stacked", {"m": "23", "claims": 2}), "rec inner circuit: META's stacked is not R's shape (m, claims)"),
], ids=["none", "m22", "m24", "claims3", "m40", "not-a-shape"])
def test_a_meta_shape_other_than_rs_stream_is_refused(tmp_path, change, why):
    """R's m and claims are META's alone: a META naming no shape, or one other than the shape R's stream was opened at, is
    refused."""
    assert _read(C3, inner=_meta(tmp_path, change)) == (1, None, why)


def _fresh(*tag) -> str:
    return (hashlib.sha512(repr(tag).encode()).digest() * 2).hex()


def _opened_at(m: int):
    """stacked/rep0 rebuilt as R's stream at ``m`` (`rec_live.stacked_rounds(m)`) over the registered cap: its binding round as
    committed (R's registered top, the first claim's ring switch), every later round's coins, tops and `c_k` fresh."""
    from verity_flock.rec_live import stacked_rounds

    def change(t: dict) -> None:
        s = "stacked/rep0"
        out = [_round(t, s, k=0)]
        for k, (n, tops, level0, absorbs) in list(enumerate(stacked_rounds(m)))[1:]:
            out.append({"stream": s, "k": k, "n": n, "coins": [_fresh(m, k, "coin", i)[:32] for i in range(n)],
                        "tops": [[_fresh(m, k, j, i) for i in range(w)] for j, w in enumerate(tops)], "level0": level0,
                        "commits": [_fresh(m, k, "c")] if absorbs else []})
        t["rounds"] = [r for r in t["rounds"] if r["stream"] != s] + out
        t["streams"][s]["rounds"] = len(out)
    return change


@pytest.mark.parametrize("m,round3", [(22, 15), (24, 17)])
def test_r_opened_at_another_m_over_the_registered_cap_is_refused(tmp_path, m, round3):
    """The fourth control: R's binding round commits the registered cap, which three m share (m 22 to 24, cap depth 7, 36
    rounds), and R's stream is opened at another m than META's from there on. The cap check passes and the shape check
    refuses it."""
    assert _read(C3, _changed(tmp_path, _opened_at(m))) == (1, None, SHAPE.format(**R_SHAPE) + (
        f"round 3 is (coins {round3}, tops [], level0 1, commits true), not (coins 16, tops [], level0 1, commits true)"))


@pytest.mark.parametrize("args,why", [
    (["--rec-inner-session", "P", "--rec-inner-circuit", "M", "--rec-inner-registration", "R"],
     "--rec-inner-registration goes with --rec-inner-transcript"),
    (["--rec-inner-transcript", "T", "--rec-inner-session", "P", "--rec-inner-registration", "R"],
     "--rec-inner-registration goes with --rec-inner-circuit, whose META binds it"),
], ids=["no-transcript", "no-circuit"])
def test_the_registration_goes_with_the_transcript_and_the_circuit(args, why):
    paths = {"P": C3 / "prover", "M": C3 / "circuit-meta.txt", "R": C3 / "circuit-registration.json",
             "T": C3 / "public.json"}
    p = subprocess.run([str(BIN), "rec-transcript", *[str(paths.get(a, a)) for a in args]], capture_output=True, text=True)
    assert (p.returncode, p.stderr.strip()) == (1, why), p.stderr


REPS = "[circuit/rep0, circuit/rep1]"


def _record(tmp_path: Path, change) -> Path:
    r = json.loads((_fixture(C3) / "session" / "session.json").read_text())
    change(r)
    (tmp_path / "rec").mkdir()
    (tmp_path / "rec" / "session.json").write_text(json.dumps(r))
    return tmp_path / "rec"


def _coins(stream: str, k: int, coins: list[str] | None = None):
    def change(t):
        r = _round(t, stream, k=k)
        r["coins"] = coins if coins is not None else ["00" * 16] * len(r["coins"])
    return change


@pytest.mark.parametrize("change,why", [
    (_coins("circuit/rep0", 2), "circuit/rep0 round 2's coins are not the inner session record's"),
    (_coins("circuit/rep1", 41), "circuit/rep1 round 41's coins are not the inner session record's"),
    (_coins("circuit/rep0", 42), "circuit/rep0 round 42's coins are not the inner session record's"),
    (_coins("circuit/rep0", 40, ["00" * 16]), "circuit/rep0 records 95 rounds and the inner session's record 94"),
    (_coins("circuit/rep0", 41, []), "circuit/rep0 has rounds of no coins at #[40, 41], and only z_partial's round is split"),
], ids=["before-split", "h-round", "after-split", "no-split", "two-splits"])
def test_the_reps_coins_are_the_prover_records_across_z_partials_split(tmp_path, change, why):
    """The transcript splits z_partial's round 40 (no coins, then h's round 41, which takes the prover's round 40's coins), so
    the prover's round `k` is the transcript's `k` below 40 and `k + 1` from 40 on; a rep's coins are checked at every
    round."""
    rc, line, err = _read(C3, _changed(tmp_path, change))
    assert (rc, line, err) == (1, None, f"rec inner transcript: {why}")


@pytest.mark.parametrize("record,why", [
    (lambda tmp: _fixture(C3) / "session", f"the inner session's record has the streams {STREAMS}, not {REPS} (the prover's "
     "record, which check 5 reads)"),
    (lambda tmp: _record(tmp, lambda r: r.__setitem__("streams", r["streams"][:2])),
     "circuit/rep0 splits z_partial's round 40 in two and records 95 rounds; the inner session's record has 95, not 94 (it is "
     "the prover's record, check 5's)"),
], ids=["relay-record", "relay-reps"])
def test_the_relays_record_is_not_the_inner_sessions(tmp_path, record, why):
    """The relay's record holds the transcript's rounds as the firewall split them, and the sparse streams and R's: it is not
    the record check 5 reads, and `innerCommits` reads that one."""
    rc, line, err = _read(C3, record=record(tmp_path))
    assert (rc, line, err) == (1, None, f"rec inner transcript: {why}")


def test_a_top_another_stream_committed_is_refused(tmp_path):
    """R's level-1 top swapped for the circuit table's rep 0 level-1 top (the same width): each top is committed under a fresh
    salt, so the session holds that `b ‖ c` twice."""
    def swap(t):
        top = _round(t, "circuit/rep0", tops=True)["tops"]
        _round(t, "stacked/rep0", k=11)["tops"] = copy.deepcopy(top)
        swap.first = top[0][0]

    rc, line, err = _read(C3, _changed(tmp_path, swap))
    assert (rc, line) == (1, None)
    assert err == (f"rec inner transcript: the b ‖ c {swap.first[:16]}… is committed twice; the firewall commits each under a "
                   "fresh salt"), err


@pytest.mark.parametrize("change,why", [
    (lambda t: _round(t, "sparse/rep1").__setitem__("tops", [["00" * 128]]),
     "sparse/rep1 round 0 commits a top, and a sparse stream has no table"),
    (lambda t: _round(t, "stacked/rep0", k=11)["tops"][0].__setitem__(0, "00" * 64),
     "stacked/rep0 round 11's top 0: a b ‖ c of 64 bytes, not 128"),
    (lambda t: t["hello"].__setitem__("tables", ["circuit", "stacked"]),
     "Hello's tables #[circuit, stacked], not #[circuit]"),
    (lambda t: t["link"]["roots"].__setitem__("stacked", t["link"]["roots"]["circuit"]),
     "the committed roots are [circuit, stacked]'s, not #[circuit]'s"),
    (lambda t: _round(t, "stacked/rep0", k=0).__setitem__("tops", []),
     "stacked/rep0's binding round commits no top: R's root is committed nowhere"),
    (lambda t: t["streams"]["sparse/rep1"].__setitem__("parent", "circuit/rep0"),
     "sparse/rep1 is not declared at circuit/rep1's round 95"),
], ids=["sparse-top", "short-node", "tables", "roots", "r-unbound", "parent"])
def test_candidate_3s_session_is_held_to_its_layout(tmp_path, change, why):
    """Where `rec_live.schedule_for` puts each stream, and what its rounds hold: tops are `b ‖ c` lists in a table's streams
    only, and R's binding round commits R's root."""
    rc, line, err = _read(C3, _changed(tmp_path, change))
    assert (rc, line, err) == (1, None, f"rec inner transcript: {why}")


def _table(tmp_path: Path, table: str) -> tuple[Path, Path]:
    """2d61's session with its circuit table named ``table``: the reps' streams, Hello's table and the committed root,
    in the transcript and the prover's record alike."""
    t = json.loads((_fixture(C3) / "public.json").read_text().replace('"circuit/rep', f'"{table}/rep'))
    t["hello"]["tables"], t["link"]["roots"] = [table], {table: t["link"]["roots"]["circuit"]}
    (tmp_path / "rec").mkdir(parents=True)
    r = (C3 / "prover" / "session.json").read_text().replace('"circuit/rep', f'"{table}/rep')
    (tmp_path / "rec" / "session.json").write_text(r)
    return _write(tmp_path / "transcript.json", t), tmp_path / "rec"


@pytest.mark.parametrize("table,streams", [
    ("sparse", "[sparse/rep0, sparse/rep1, stacked/rep0]"),
    ("stacked", "[stacked/rep0, stacked/rep1, sparse/rep0, sparse/rep1]"),
])
def test_a_table_named_as_candidate_3s_own_streams_is_refused(tmp_path, table, streams):
    """A table whose reps' streams `<table>/rep<x>` are named as a sparse phase's (`sparse/rep<x>`) or R's
    (`stacked/rep0`), where `holoMap`'s lookup by name finds the reps first: `innerCommits` takes the table's prefix
    from a stream named neither way, so it finds no table's reps. The session with its table named otherwise is read,
    its reps renamed."""
    rc, line, err = _read(C3, *_table(tmp_path / "other", "other"))
    assert rc == 0, err
    assert [s["stream"] for s in line["streams"]][:2] == ["other/rep0", "other/rep1"]
    rc, line, err = _read(C3, *_table(tmp_path / table, table))
    why = f"rec inner transcript: its message rounds' streams {streams} hold no table's reps"
    assert (rc, line, err) == (1, None, why)


def test_the_plain_session_is_read_as_before(tmp_path):
    """`rec-coins`' inner session, one table's two reps: the verifier holds their `c_k` and no stream besides, and the
    session without rep 1's rounds is refused with the message it always had."""
    rc, line, err = _read(_fixture(PLAIN))
    assert (rc, line) == (0, {"reps": [78, 78], "streams": []}), err
    t = json.loads((PLAIN / "public.json").read_text())
    t["rounds"] = [r for r in t["rounds"] if r["stream"] != "circuit/rep1"]
    f = tmp_path / "plain.json"
    f.write_text(json.dumps(t))
    rc, line, err = _read(PLAIN, f)
    assert (rc, line, err) == (1, None, "rec inner transcript: its message rounds' streams [circuit/rep0] are not one "
                                        "table's rep0 and rep1")


def test_a_plain_session_under_rs_registration_is_a_downgrade(tmp_path):
    """A registration of R's top holds the session to R's stream even when it has no candidate 3 stream at all, and even
    when it declares `stacked/rep0` with no rounds."""
    why = ("rec inner transcript: the registration names R's top (stacked), and the session has no stacked/rep0: R's stream "
           "is missing (a downgrade)")
    assert _read(_fixture(PLAIN), inner=_inner()) == (1, None, why)
    t = json.loads((PLAIN / "public.json").read_text())
    t["streams"]["stacked/rep0"] = json.loads((C3 / "public.json").read_text())["streams"]["stacked/rep0"]
    assert _read(PLAIN, _write(tmp_path / "plain.json", t), inner=_inner()) == (1, None, why)


#: the bucket's RecSparse shape 2d61 ran its sparse phase at, and META's range of its inner statement's hidden circuit
#: (`pod/rec_inner.py stage --n 16 --plain-leaves`, circuit_sha512 676d7254…, which `rec_vstage stage --circuit` reads)
SPARSE = (18, 13, 6, 0)
RANGE = {"count": 64, "net": "unit", "pos0": 256, "slot_log": 13, "type": "unit"}


@pytest.fixture(scope="module")
def sparse(tmp_path_factory) -> Path:
    """2d61's sparse phase staged as its RecSparse statement over both reps, as `rec_vstage stage --sparse` stages it: its
    message ports read as the firewall committed them (`rec_vstage.sparse_messages`), h as the algebra's h round."""
    from verity_flock import rec_sparse as SP
    from verity_flock import rec_vstage as VS

    d = _fixture(C3)
    public_b = (d / "public.json").read_bytes()
    public, private = json.loads(public_b), json.loads((d / "private.json").read_text())
    vf = VS.verifier(public, json.loads((d / "fold.json").read_text()), json.loads((d / "points.json").read_text()))
    program = VS.O.session_program(public_b)
    regs, own = VS.messages(program, VS.message_rounds(public, private, vf.sh))
    out = tmp_path_factory.mktemp("rec-sparse")
    VS.stage_sparse(public, private, vf, program, regs, own, SP.Shape(*SPARSE), RANGE, out, evaluate=False)
    return out


def _statement(d: Path, transcript: Path | None = None) -> subprocess.CompletedProcess:
    rec = [] if transcript is None else ["--rec-inner-transcript", str(transcript), "--rec-inner-session", str(C3 / "prover"),
                                         *_inner()]
    return subprocess.run([str(BIN), "statement", "--statement", "verity/flock-circuit", "--circuit", str(d / "circuit.txt"),
                           "--public", str(d / "pub.bin"), "--partition", str(d / "partition.json"), "--program",
                           str(d / "program.json"), "--registered", str(d / "registered.json"), *rec],
                          capture_output=True, text=True, check=False)


@pytest.mark.slow
def test_a_rec_sparse_statements_message_rows_are_the_sparse_streams_c_k(sparse, tmp_path):
    """`RecVStar.checkPubRows` on RecSparse (`fixedRow`'s `rec-sparse-<p>`): 2d61's statement, about 300 MB at
    n18-e13-s6-t0, is the only RecSparse statement whose message rows a coin server recorded, so nothing smaller shows
    the verifier reads `sparse/rep<x>`'s rounds at the ports' order. `statement` with the inner transcript sets it up
    (its `rec-sparse-v` compared with no round), and refuses it at a transcript whose `sparse/rep1`
    `c_k` of `u` (the round before the claim's) is one bit off; without the transcript it sets up as before. Its reads file
    holds the statement's message ports as `RecVStar.checkOwn` asks (`rec-statements --registered`, at 2d61's private
    inner circuits, the only ones a sparse phase runs over: `ownPorts` and RecSparse's other messages, `sparseMsgs`)."""
    unit = f"rec-sparse/n{SPARSE[0]}-e{SPARSE[1]}-s{SPARSE[2]}-t{SPARSE[3]}"
    ports = {k: x["value"] for k, x in json.loads((sparse / "registered.json").read_text())["ports"].items()}
    assert {"rec-sparse-u", "rec-sparse-v"} <= set(ports.values()), ports
    p = subprocess.run([str(BIN), "rec-statements", *_shape(tmp_path), "--unit", unit, "--instances", "2",
                        "--registered", str(sparse / "registered.json")], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    (port,) = [k for k, v in ports.items() if v == "rec-sparse-u"]
    for t in (None, _fixture(C3) / "public.json"):
        p = _statement(sparse, t)
        assert p.returncode == 0, p.stdout + p.stderr

    def flip(t):
        r = [r for r in sorted(t["rounds"], key=lambda r: r["k"]) if r["stream"] == "sparse/rep1" and r["commits"]][-2]
        c = bytearray.fromhex(r["commits"][0])
        c[0] ^= 1
        r["commits"] = [c.hex()]
        flip.k = r["k"]

    p = _statement(sparse, _changed(tmp_path, flip))
    assert p.returncode == 1, p.stdout + p.stderr
    assert re.fullmatch(rf"setup: rec-pub: {re.escape(unit)}: port {port}'s table row \d+ is not sparse/rep1's c_31 \(u, inner "
                        rf"round {flip.k}\) of the inner transcript", p.stdout.strip()), p.stdout


def _shape(tmp_path: Path, sparse=SPARSE, stacked=R_SHAPE, holo=True) -> list[str]:
    """`rec-inner`'s line at 2d61's table shape (m 25, k_log 22, claims 8 over 3 regions), its circuits private (`holo`,
    `rec-inner --hide`: its reps split z_partial's round), with the bucket's RecSparse shape and R's when given (`rec-inner
    --sparse`, META's `stacked`)."""
    j = {"format": "verity/rec-inner/v1", "m": 25, "k_log": 22, "claims": 8, "regions": 3, **({"holo": True} if holo else {}),
         **({"sparse": list(sparse)} if sparse else {}), **({"stacked": stacked} if stacked else {})}
    return ["--rec-inner", str(_write(tmp_path / "rec-inner.json", j))]


def _own(tmp_path: Path, d: Path) -> Path:
    """`d`'s coin server's record in its own format (`rec_live.record_bytes`), which the verifier reads at its inner
    shape (`RecSetup.checkReads`): the fixtures' records predate it."""
    from verity_flock import rec_live as RL

    f = tmp_path / f"own-{d.name}.json"
    f.write_text(RL.record_bytes(json.loads((d / "public.json").read_text())))
    return f


def _covers(tmp_path: Path, shape: list[str], transcript: Path | None = None) -> tuple[int, str]:
    d = _fixture(C3)
    p = subprocess.run([str(BIN), "rec-transcript", "--rec-inner-transcript", str(transcript or _own(tmp_path, d)),
                        "--rec-inner-session", str(d / "prover"), *_inner(), *shape], capture_output=True, text=True)
    return p.returncode, p.stderr.strip()


NO_SPARSE = ("rec-inner: the inner transcript holds candidate 3's sparse phases, and the verifier's inner shape names no "
             "RecSparse shape (rec-inner --sparse): its V* list would have no RecSparse statement")


@pytest.mark.parametrize("sparse,stacked,why", [
    (SPARSE, R_SHAPE, ""),
    (None, R_SHAPE, NO_SPARSE),
    (None, None, NO_SPARSE),
    (SPARSE, None, "rec-inner: the inner transcript holds R's stream (stacked/rep0), and the verifier's inner shape names no "
     "shape of R (stacked): its V* list would have no level of R's or R's opening"),
    (SPARSE, {"m": 24, "claims": 2}, "rec-inner: the verifier's inner shape names R's shape (m 24, claims 2), and the inner "
     "circuit's META (m 23, claims 2) (stacked)"),
], ids=["honest", "no-sparse", "plain", "no-r", "r-m24"])
def test_the_verifiers_inner_shape_covers_candidate_3s_session(tmp_path, sparse, stacked, why):
    """`RecVStar.checkCovers`, as `verify` and `statement` run it beside the V* checks: 2d61's session has both sparse phases
    and R's stream, so the verifier's inner shape names the bucket's RecSparse shape and META's R, else its V* list would
    leave RecSparse, R's levels or R's opening out; each shape short of that is refused by name."""
    assert _covers(tmp_path, _shape(tmp_path, sparse, stacked)) == (0 if not why else 1, why)


def _moved(stream: str, j: int, k: int):
    """``stream``'s ``j``-th message round's `c_k` moved to its round ``k``, which commits none."""
    def change(t: dict) -> None:
        rs = sorted((r for r in t["rounds"] if r["stream"] == stream), key=lambda r: r["k"])
        src, dst = [r for r in rs if r["commits"]][j], rs[k]
        assert not dst["commits"]
        dst["commits"], src["commits"] = src["commits"], []
    return change


def test_each_reps_stream_holds_its_schedules_rounds(tmp_path):
    """`RecVStar.checkCommitRounds` first holds every stream but the sparse phases' and R's to the rep's schedule at the
    verifier's inner shape, round for round (`RepsDrawn`'s second half): 2d61's reps split z_partial's round, as private
    circuits' do (`holo_rounds`, 95 rounds), so under a shape whose circuits are public (`rep_rounds`, 94) they are
    refused for their rounds."""
    assert _covers(tmp_path, _shape(tmp_path, holo=False)) == (
        1, "rec inner transcript: circuit/rep0 holds 95 rounds, and its schedule (m 25, k_log 22, claims 8) 94")


@pytest.mark.parametrize("change,why", [
    (_moved("circuit/rep1", 20, 23), "rep 1's message round 20 is at round 23, and its schedule's at round 22 (77 message "
     "rounds, the schedule's 77)"),
    (_moved("sparse/rep0", 0, 0), "sparse/rep0's message round 0 is at round 0, and its schedule's at round 1 (33 message "
     "rounds, the schedule's 33)"),
], ids=["rep-round", "sparse-round"])
def test_each_streams_c_k_sit_at_its_schedules_message_rounds(tmp_path, change, why):
    """`RecVStar.checkCommitRounds`, the last of `checkCovers`: the verifier holds each rep's `c_k` at the message rounds of
    `rec_live.rep_rounds` at its own inner shape (`holo_rounds` when its circuits are private, as 2d61's are), each
    sparse stream's at `sparse_rounds` of its RecSparse shape, and R's at `stacked_rounds`. 2d61's transcript with one
    rep's or one sparse stream's `c_k` moved to a round that commits none (which the reps' coins and the streams' counts
    do not see) is refused by the round."""
    assert _covers(tmp_path, _shape(tmp_path), _changed(tmp_path, change)) == (1, f"rec inner transcript: {why}")


@pytest.mark.parametrize("change,why", [
    (lambda t: t["link"]["roots"]["circuit"].pop(), "circuit's root_B has 127 nodes, and its schedule's level 0 top 128"),
    (lambda t: _round(t, "circuit/rep1", k=70)["tops"][0].pop(),
     "rep 1's tops (round, nodes) are [(59, 64), (70, 63), (81, 32)], and its schedule's [(59, 64), (70, 64), (81, 32)]"),
    (lambda t: _round(t, "circuit/rep0", k=81).__setitem__("tops", []),
     "rep 0's tops (round, nodes) are [(59, 64), (70, 64)], and its schedule's [(59, 64), (70, 64), (81, 32)]"),
], ids=["root", "top-short", "top-missing"])
def test_the_circuit_tables_tops_sit_at_its_schedule(tmp_path, change, why):
    """`RecVStar.checkLinkRounds`: the circuit table's `root_B` has its schedule's level-0 `2^cap` nodes, and each rep's later
    tops (`rec-top-L<l>` past level 0, which each level's openings climb to) sit at its schedule's rounds with its node
    counts, here `holo_rounds`' at 2d61's shape. Neither `candidate3` nor `checkCovers` counts the circuit table's tops, so
    a `root_B` short of a node, or a rep's top short of one or missing, is refused here."""
    assert _covers(tmp_path, _shape(tmp_path), _changed(tmp_path, change)) == (1, f"rec inner transcript: {why}")


POINTS_WHY = ("rec inner transcript: the inner session's record's link points are not the ones the coin server drew "
              "(link.points): check 5 would take the extra claims at points the prover chose")


def _bit_off(h: str) -> str:
    return f"{int(h, 16) ^ 1:032x}"


@pytest.mark.parametrize("d", [C3, PLAIN], ids=["c3", "plain"])
def test_the_records_link_points_are_the_coin_servers(tmp_path, d):
    """`RecVStar.innerLink`: check 5 takes the extra claims at the inner session record's link points (the prover's record),
    so they must be the ones the coin server drew after `root_B` (the transcript's `link.points`, `points × m` coins in
    point order). 2d61's and rec-coins' sessions pass; a transcript whose first link coin is one bit off is refused, and so
    is a record whose last coin is, or whose two points are swapped."""
    assert _read(_fixture(d))[0] == 0
    t = json.loads((d / "public.json").read_text())
    t["link"]["points"][0] = _bit_off(t["link"]["points"][0])
    assert _read(d, _write(tmp_path / "transcript.json", t)) == (1, None, POINTS_WHY)
    held = json.loads(((d / "prover" if (d / "prover").is_dir() else d / "session") / "session.json").read_text())
    for name, change in [("last", lambda p: p[-1].__setitem__(-1, _bit_off(p[-1][-1]))), ("swap", lambda p: p.reverse())]:
        s = json.loads(json.dumps(held))
        change(s["link"]["points"])
        (tmp_path / name).mkdir()
        _write(tmp_path / name / "session.json", s)
        assert _read(d, record=tmp_path / name) == (1, None, POINTS_WHY), name


def test_rs_c_k_off_its_schedule_is_refused(tmp_path):
    """R's stream with its second `c_k` moved to round 2, which commits none, is refused under the honest inner shape:
    `checkStackedShape` holds R's transcript rounds to `stackedRounds` of META's shape before `checkCommitRounds`
    holds the `c_k` and tops the verifier keeps of them to the same rounds (its refusals, unreachable past the first,
    are the Lean guards in `Test/InnerStreams.lean`)."""
    assert _covers(tmp_path, _shape(tmp_path), _changed(tmp_path, _moved("stacked/rep0", 1, 2))) == (
        1, SHAPE.format(**R_SHAPE) + "round 1 is (coins 7, tops [], level0 0, commits false), not (coins 7, tops [], "
        "level0 0, commits true)")


def test_a_third_sparse_stream_is_refused(tmp_path):
    """A `sparse/rep2` beside the two reps' sparse phases (sparse/rep1's rounds, each `c_k` fresh) is refused under the
    honest inner shape: `candidate3` holds the session's streams to the reps', their sparse phases and R's before
    `checkCommitRounds`, which reads `sparse/rep0` and `sparse/rep1` only, refuses any other held `sparse/…` stream (its
    refusal is the Lean guards' in `Test/InnerStreams.lean`)."""
    def change(t: dict) -> None:
        t["streams"]["sparse/rep2"] = dict(t["streams"]["sparse/rep1"])
        t["rounds"] += [{**r, "stream": "sparse/rep2", "commits": [_fresh("rep2", r["k"])] if r["commits"] else []}
                        for r in t["rounds"] if r["stream"] == "sparse/rep1"]
    assert _covers(tmp_path, _shape(tmp_path), _changed(tmp_path, change)) == (
        1, "rec inner transcript: its message rounds' streams [circuit/rep0, circuit/rep1, sparse/rep0, sparse/rep1, "
        "stacked/rep0, sparse/rep2] are not one table's rep0 and rep1, both reps' sparse phases and at most R's stream "
        "([circuit/rep0, circuit/rep1, sparse/rep0, sparse/rep1, stacked/rep0])")


def test_an_inner_shape_naming_candidate_3_over_a_plain_session_is_refused(tmp_path):
    """The plain session (`rec-coins`' inner session, two reps and no other stream) passes under its own inner shape, and is
    refused under one naming the bucket's RecSparse shape: the shape names a phase the session does not have."""
    d = _fixture(PLAIN)
    run = lambda shape: subprocess.run([str(BIN), "rec-transcript", "--rec-inner-transcript", str(_own(tmp_path, d)),
                                        "--rec-inner-session", str(d / "session"), *shape], capture_output=True, text=True)
    p = run(["--rec-inner", str(_write(tmp_path / "plain.json", {"format": "verity/rec-inner/v1", "m": 27, "k_log": 22,
                                                                 "claims": 8, "regions": 3}))])
    assert p.returncode == 0 and p.stdout.startswith('RECTRANSCRIPT {"reps":[78,78],"streams":[]}'), p
    p = run(_shape(tmp_path, stacked=None))
    assert p.returncode == 1 and p.stderr.strip() == (
        "rec-inner: the verifier's inner shape names a RecSparse shape (--sparse), and the inner transcript holds no sparse "
        "phase"), p


#: the verifier's inner shapes of the plain session (`rec-coins`' inner statement, public circuits) and of 2d61's
PLAIN_SHAPE = {"format": "verity/rec-inner/v1", "m": 27, "k_log": 22, "claims": 8, "regions": 3}
C3_SHAPE = {**PLAIN_SHAPE, "m": 25, "holo": True, "sparse": list(SPARSE), "stacked": R_SHAPE}
OFF = ("rec inner transcript: the coin server's record is off the schedule of the verifier's inner shape (m {m}, k_log "
       "{k_log}, claims {claims}): ")


def _python_off(t: dict, shape: dict) -> bool:
    """Whether `rec_live.on_schedule`, `RecVStar.checkSchedule`'s mirror, refuses the record ``t`` at the verifier's
    inner shape ``shape`` (its schedule `rec_live.schedule_for`'s, over the record's Hello)."""
    from verity_flock import rec_live as RL
    from verity_flock import rec_sparse as SP

    st = {"dense_m": shape["m"], "k_log": shape["k_log"], "nbl": t["hello"]["link"]["m"] - RL.PT_LOCAL,
          "regions": shape["regions"]}
    sched = RL.schedule_for(st, shape.get("holo", False), (shape.get("stacked") or {}).get("m"),
                            SP.Shape(*shape["sparse"]) if shape.get("sparse") else None)
    try:
        RL.on_schedule(t, sched)
    except ValueError:
        return True
    return False


def _bypassed(tmp_path: Path, change, declare: bool = True) -> tuple[Path, Path]:
    """The plain session as one that bypassed the proxy leaves it: ``change`` on circuit/rep0's rounds in the coin
    server's record (which still ends ``finish``, and declares the stream's new count when ``declare``), and the
    prover's record holding the same coins at every round, so the reps' coins are the record's (`innerCommits`)."""
    t = json.loads((_fixture(PLAIN) / "public.json").read_text())
    r = json.loads((PLAIN / "session" / "session.json").read_text())
    rs = [x for x in t["rounds"] if x["stream"] == "circuit/rep0"]
    change(rs)
    t["rounds"] = rs + [x for x in t["rounds"] if x["stream"] != "circuit/rep0"]
    if declare:
        t["streams"]["circuit/rep0"]["rounds"] = len(rs)
    (s,) = [x for x in r["streams"] if x["stream"] == "circuit/rep0"]
    s["rounds"] = [{**s["rounds"][min(k, len(s["rounds"]) - 1)], "k": k, "n": len(x["coins"]), "coins": x["coins"]}
                   for k, x in enumerate(rs)]
    (tmp_path / "rec").mkdir()
    _write(tmp_path / "rec" / "session.json", r)
    return _write(tmp_path / "transcript.json", t), tmp_path / "rec"


def _transcript_of(transcript: Path, record: Path, shape: dict | None, tmp_path: Path) -> tuple[int, str]:
    inner = [] if shape is None else ["--rec-inner", str(_write(tmp_path / "rec-inner.json", shape))]
    p = subprocess.run([str(BIN), "rec-transcript", "--rec-inner-transcript", str(transcript), "--rec-inner-session",
                        str(record), *inner], capture_output=True, text=True)
    return p.returncode, p.stderr.strip()


def test_the_honest_records_are_on_the_verifiers_schedule(tmp_path):
    """`RecVStar.checkSchedule` and its mirror `rec_live.on_schedule` pass the plain session's record and 2d61's at
    their inner shapes, round for round, and both read in the coin server's own format as their JSON readers read
    them (`RecSetup.checkReads`)."""
    t = json.loads((_fixture(PLAIN) / "public.json").read_text())
    assert not _python_off(t, PLAIN_SHAPE)
    assert not _python_off(json.loads((_fixture(C3) / "public.json").read_text()), C3_SHAPE)
    assert _transcript_of(_own(tmp_path, _fixture(PLAIN)), PLAIN / "session", PLAIN_SHAPE, tmp_path) == (0, "")
    assert _covers(tmp_path, _shape(tmp_path)) == (0, "")


NOT_OWN_FORMAT = ("rec inner transcript: the coin server's record does not read in its own format (RecRecord.readFull: "
                  "rec_live.record_bytes, its rounds first, then its kind, tops, Hello in its keys' order and the "
                  "link's roots and points, at json.dumps' indent=1)")


@pytest.mark.parametrize("write", [
    json.dumps,
    lambda t: json.dumps({"rounds": t["rounds"], **t}, indent=2),
    lambda t: json.dumps({"rounds": t["rounds"], **t, "hello": dict(reversed(t["hello"].items()))}, indent=1),
], ids=["compact", "indent-2", "hello-reordered"])
def test_a_record_not_in_the_coin_servers_format_is_refused(tmp_path, write):
    """`RecSetup.checkReads`: the recursion's transcript model reads the coin server's record in its own format
    (`RecRecord.readFull`), so the verifier reads it so too. The plain session's record with its values unchanged,
    written compactly, at another indent or with Hello's keys in another order, passes every check of the transcript
    read without the verifier's inner shape, and is refused at that shape."""
    t = json.loads((_fixture(PLAIN) / "public.json").read_text())
    f = tmp_path / "transcript.json"
    f.write_text(write(t))
    assert _transcript_of(f, PLAIN / "session", None, tmp_path)[0] == 0
    assert _transcript_of(f, PLAIN / "session", PLAIN_SHAPE, tmp_path) == (1, NOT_OWN_FORMAT)


def _extra(rs: list) -> None:
    rs.append({**rs[-1], "k": len(rs)})


def _coin_count(rs: list) -> None:
    rs[3]["coins"], rs[3]["n"] = rs[3]["coins"] * 2, 2 * rs[3]["n"]


@pytest.mark.parametrize("change,declare,why", [
    (_extra, True, "circuit/rep0 is not declared at no parent with 96 rounds"),
    (lambda rs: rs.pop(), False, "191 rounds, and the schedule's 192"),
    (_coin_count, True, "round 3 is circuit/rep0 round 3 (n 2) (coins 2, tops [], level0 0, commits true), and the "
     "schedule's circuit/rep0 round 3 (n 1) (coins 1, tops [], level0 0, commits true)"),
], ids=["extra-round", "missing-round", "coin-count"])
def test_a_finished_record_off_the_verifiers_schedule_is_refused(tmp_path, change, declare, why):
    """`RecVStar.checkSchedule`, held to its mirror `rec_live.on_schedule`: the proxy that holds a session to its
    schedule runs on the prover's side, so a session that bypassed it has a coin server's record that ends ``finish``
    with any rounds, and the prover's record to match. The plain session with an extra round, a missing round or a
    round of another coin count, in both records, passes every other check of the transcript (read without the
    verifier's inner shape) and is refused at the verifier's inner shape, by both."""
    transcript, record = _bypassed(tmp_path, change, declare)
    t = json.loads(transcript.read_text())
    assert t["end"] == "finish" and _python_off(t, PLAIN_SHAPE)
    assert _transcript_of(transcript, record, None, tmp_path)[0] == 0
    assert _transcript_of(transcript, record, PLAIN_SHAPE, tmp_path) == (1, OFF.format(**PLAIN_SHAPE) + why)


def test_a_reps_round_of_another_level0_is_refused(tmp_path):
    """2d61's rep 0 binding round with no copy of level 0's top: no other check of candidate 3's session reads a rep's
    `level0`, and the verifier's schedule and its mirror refuse it."""
    transcript = _changed(tmp_path, lambda t: _round(t, "circuit/rep0", k=0).__setitem__("level0", 0))
    assert _python_off(json.loads(transcript.read_text()), C3_SHAPE)
    assert _covers(tmp_path, _shape(tmp_path), transcript) == (1, OFF.format(**C3_SHAPE) + (
        "round 0 is circuit/rep0 round 0 (n 6) (coins 6, tops [], level0 0, commits false), and the schedule's "
        "circuit/rep0 round 0 (n 6) (coins 6, tops [], level0 1, commits false)"))


#: `rec-coins`' inner statement, a whole inner circuit (plain, with no registration and no R in its META)
STAGE = PLAIN.parent / "u" / "stage"


#: candidate 3's inner shape at `STAGE` (m 27, k_log 22, claims 8 over 3 regions), with a RecSparse shape whose sparse
#: range check 5 passes on it (`RecCoins.checkAtC3`), and R's
AT_STAGE = {"format": "verity/rec-inner/v1", "m": 27, "k_log": 22, "claims": 8, "regions": 3, "sparse": [1, 12, 1, 0],
            "stacked": R_SHAPE}


NO_TRANSCRIPT = ("rec inner transcript: candidate 3's session needs the inner transcript (--rec-inner-transcript, "
                 "--rec-inner-session), which its V* list and rows are checked against")
NO_REGISTRATION = ("rec inner registration: candidate 3's session needs R's circuit registration "
                   "(--rec-inner-registration), which binds R before the coins")


@pytest.mark.parametrize("cmd,transcript,why", [
    pytest.param("statement", False, (1, f"setup: {NO_TRANSCRIPT}"), marks=pytest.mark.slow),
    ("statement", True, (2, NO_REGISTRATION)),
    ("verify", True, (2, NO_REGISTRATION)),
], ids=["no-transcript", "no-registration", "verify-no-registration"])
def test_candidate_3s_session_needs_the_inner_transcript_and_rs_registration(tmp_path, cmd, transcript, why):
    """What the bridge to `AcceptsC3` reads beside the inner statement's files. `statement` on `rec-coins`' inner
    statement at candidate 3's inner shape there, with that statement's files, which pass check 5's sparse range
    (`RecCoins.checkAtC3`) and whose circuit's META is its header's (`RecSetup.checkMeta`), refuses it at setup without
    the inner transcript (`RecSetup.checks`), and refuses its inputs (exit 2) with the transcript and no circuit
    registration (`RecSetup.commits`). `verify` reads its inputs as `statement` does (shown on an empty outer circuit,
    which it sets up no further than the V* checks)."""
    _fixture(PLAIN)
    (tmp_path / "s").mkdir()
    (tmp_path / "s" / "session.json").write_text("{}")
    (tmp_path / "empty").write_bytes(b"")
    draw = _write(tmp_path / "draw.json", {"law": "subset", "population": 64, "k": 64, "units": list(range(64))})
    files = ["--rec-inner-circuit", str(STAGE / "circuit.txt"), "--rec-inner-public", str(STAGE / "pub-64.bin"),
             "--rec-inner-partition", str(STAGE / "partition.json"), "--rec-inner-statement",
             "verity/flock-circuit+plain-leaves", "--rec-inner-statement-coins", "os", "--rec-inner-draw", str(draw),
             "--rec-inner-session", str(PLAIN / "session"),
             *(["--rec-inner-transcript", str(PLAIN / "public.json")] if transcript else [])]
    inner = ["--rec-inner", str(_write(tmp_path / "rec-inner.json", AT_STAGE)), *files]
    if cmd == "statement":
        args = ["statement", "--statement", "verity/flock-circuit+plain-leaves", "--circuit",
                str(STAGE / "circuit.txt"), "--public", str(STAGE / "pub-64.bin"), "--partition",
                str(STAGE / "partition.json"), "--coins", "os"]
    else:
        args = ["verify", "--circuit", str(tmp_path / "empty"), "--public", str(tmp_path / "empty"), "--session",
                str(tmp_path / "s")]
    p = subprocess.run([str(BIN), *args, *inner], capture_output=True, text=True)
    out = p.stdout.strip() if why[0] == 1 else p.stderr.strip()
    assert (p.returncode, out) == why, p.stdout + p.stderr


def test_rec_inner_refuses_r_without_the_sparse_shape():
    """`rec-inner` reads R's shape from the inner circuit's META (2d61's header) and refuses it without the bucket's RecSparse
    shape (`--sparse`), before it reads the circuit's tables: R opens the sparse phase's claims. A `--sparse` that is not
    four numbers is refused as the flag reads."""
    d = _fixture(C3)
    base = [str(BIN), "rec-inner", "--circuit", str(d / "circuit-meta.txt"), "--public", str(d / "public.json")]
    p = subprocess.run(base, capture_output=True, text=True)
    assert p.returncode == 1 and p.stdout.strip() + p.stderr.strip() == (
        "rec-inner: the inner circuit's META names R's shape (m 23, claims 2), and the verifier names no RecSparse shape "
        "(--sparse NU,ELL,S_LOG,T_LOG): R opens the sparse phase's claims, so V*'s list would miss RecSparse"), p
    p = subprocess.run([*base, "--sparse", "18,13,6"], capture_output=True, text=True)
    assert p.returncode == 2 and p.stderr.strip() == "--sparse 18,13,6: NU,ELL,S_LOG,T_LOG, four natural numbers", p


STOPPED = "rec inner transcript: the inner session ended stop, not finish: a stopped session is no inner proof"
NO_END = "rec inner transcript: the coin server's record names no end; only a finished session is read"


@pytest.mark.parametrize("d", [C3, PLAIN], ids=["c3", "plain"])
def test_a_stopped_sessions_record_is_refused(tmp_path, d):
    """`RecVStar.checkFinished`: the coin server's record of a session that ran every round and then stopped (a refusal, or
    a connection that ended without `Finish`) is the finished one's but for `end`, and is refused as no inner proof; so is a
    record that names no end."""
    t = json.loads((_fixture(d) / "public.json").read_text())
    assert t["end"] == "finish"
    stopped, endless = copy.deepcopy(t), copy.deepcopy(t)
    stopped["end"] = "stop"
    del endless["end"]
    assert _read(d, _write(tmp_path / "stopped.json", stopped)) == (1, None, STOPPED)
    assert _read(d, _write(tmp_path / "endless.json", endless)) == (1, None, NO_END)


def test_the_record_rec_live_writes_for_a_stopped_session_is_refused(tmp_path):
    """A session `rec_live`'s firewall stops (here at a `Link` whose `y` is not the schedule's length) still gets the coin
    server's record, ending `stop` (`test_rec_live_firewall`); presented as the inner transcript, it is refused as no inner
    proof before anything else is read. The same session run to `Finish` ends `finish` and is refused, if at all, for
    something else."""
    import test_rec_live_firewall as F

    stopped = F._proxy(F._schedule())
    assert F._refusal(F._run(stopped, y=bytes(31))).startswith("rec proxy: FIREWALL-REFUSED: Link's y: ")
    assert stopped.coins.record()["end"] == "stop"
    assert _read(PLAIN, _write(tmp_path / "stopped.json", stopped.coins.record())) == (1, None, STOPPED)
    finished = F._proxy(F._schedule())
    assert F._held(F._run(finished)) and finished.coins.record()["end"] == "finish"
    _, _, err = _read(PLAIN, _write(tmp_path / "finished.json", finished.coins.record()))
    assert err not in (STOPPED, NO_END), err
