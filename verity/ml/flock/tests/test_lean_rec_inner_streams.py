"""The Lean verifier's reading of the inner session's transcript (`RecVStar.innerCommits`, `verify --rec-inner-transcript`,
`flock-verify rec-transcript`) on candidate 3's whole inner session: the fetched fixture `rec-c3-inner` is the coin server's
record (`public.json`) and the prover's record (`prover/session.json`, the one check 5 reads) of `r20261011-181721-c16e`
(#1759's `pod/88-rec-c3-inner.sh` at Shape(25,22,8), N=16, SPARSE=18,13,6,0 STACKED=1, with the relay keeping the prover's
Hello): the circuit table's two reps (95 rounds each in the transcript, 94 in the prover's record, which keeps z_partial's
round 40 whole), both reps' sparse phases (33 each) and the stacked table R (36), whose root is no Hello table's but the top
its stream's binding round commits. The reps' coins are checked against the prover's record, the sparse streams and R's are
the coin server's alone; `session/session.json` is the relay's record, which is refused. The verifier holds every stream's
`c_k` and tops, and refuses the session with a stream missing or a top another stream committed. R has no root at `Commit`,
so its salted top is registered before the session (#1798's `rec_live.registration_record`, `registration.json`), and the
verifier takes it (`--rec-inner-registration`): R's binding round must commit exactly that top, a session with R's stream
needs it, and a registration of R's top needs R's stream. c16e predates the registration, so the tests register the top its
binding round committed (`_registration`). The plain session (`rec-coins`' inner session) is read as before, its two reps'
`c_k` and nothing else."""
from __future__ import annotations

import copy
import functools
import json
import re
import subprocess
import tempfile
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
              {"commits": 32, "stream": "sparse/rep0", "tops": []},
              {"commits": 32, "stream": "sparse/rep1", "tops": []},
              {"commits": 25, "stream": "stacked/rep0", "tops": [128, 64, 64]}]


def _fixture(d: Path) -> Path:
    assert (d / "public.json").is_file(), (
        f"{d} is not fetched: `uv run research data fetch-fixtures {d.relative_to(HERE.parents[3])}`")
    return d


#: `rec_live.registration_record`'s format (#1798), which `RecVStar.registration` reads
REGISTRATION = {"kind": "rec/inner-registration/v0", "tops": "hm96-sha512/v1/node-row"}


def _registration_of(t: dict) -> dict:
    """The registration of the top R's binding round commits: `{"stacked": {"top": [b ‖ c, …]}}`, as `registration.json`."""
    return {**REGISTRATION, "registered": {"stacked": {"top": list(_round(t, "stacked/rep0", k=0)["tops"][0])}}}


def _write(path: Path, j: dict) -> Path:
    path.write_text(json.dumps(j))
    return path


@functools.cache
def _registration() -> Path:
    """c16e's R registered: the top its binding round committed."""
    t = json.loads((_fixture(C3) / "public.json").read_text())
    return _write(Path(tempfile.mkdtemp(prefix="rec-c3-registration-")) / "registration.json", _registration_of(t))


def _reg_args(registration: Path | None) -> list[str]:
    return [] if registration is None else ["--rec-inner-registration", str(registration)]


HONEST = object()


def _read(d: Path, transcript: Path | None = None, record: Path | None = None,
          registration: Path | None | object = HONEST) -> tuple[int, dict | None, str]:
    record = record or (d / "prover" if (d / "prover").is_dir() else d / "session")
    if registration is HONEST:
        registration = _registration() if d == C3 else None
    p = subprocess.run([str(BIN), "rec-transcript", "--rec-inner-transcript", str(transcript or d / "public.json"),
                        "--rec-inner-session", str(record), *_reg_args(registration)], capture_output=True, text=True)
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
    stream missing it stops there (exit 2)."""
    (tmp_path / "coins.json").write_text("{}")
    (tmp_path / "empty").write_text("")

    def verify(transcript: Path) -> subprocess.CompletedProcess:
        return subprocess.run([str(BIN), "verify", "--zk", "--circuit", str(tmp_path / "empty"), "--public", str(tmp_path / "empty"),
                               "--rec-inner-coins", str(tmp_path / "coins.json"), "--rec-inner-session", str(C3 / "prover"),
                               "--rec-inner-transcript", str(transcript), *_reg_args(_registration()),
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
    (_drop_sparse_round, "sparse/rep0 declares 33 rounds and records 32"),
], ids=["stacked-rounds", "sparse-rep1", "sparse-round"])
def test_a_stream_missing_is_refused(tmp_path, change, why):
    rc, line, err = _read(C3, _changed(tmp_path, change))
    assert (rc, line, err) == (1, None, f"rec inner transcript: {why}")


def test_the_sparse_streams_and_rs_are_the_coin_servers_alone(tmp_path):
    """The prover's record holds the reps only, so whether the session committed R is the coin server's record's to say,
    held to the registration: a transcript with R's stream gone from its rounds and its declarations is read as a session
    without R when nothing registered R's top, and refused as a downgrade when R's top was registered."""
    bad = _changed(tmp_path, _drop_stacked_everywhere)
    rc, line, err = _read(C3, bad, registration=None)
    assert (rc, line) == (0, {"reps": [77, 77], "streams": C3_STREAMS[:4]}), err
    rc, line, err = _read(C3, bad)
    assert (rc, line, err) == (1, None, "rec inner transcript: the registration has R's top, and the session has no "
                                        "stacked/rep0: R's stream is missing (a downgrade)")


def _flip_node(i: int):
    def change(reg: dict) -> None:
        top = reg["registered"]["stacked"]["top"]
        c = bytearray.fromhex(top[i])
        c[-1] ^= 1
        top[i] = c.hex()
    return change


@pytest.mark.parametrize("change", [
    _flip_node(0), _flip_node(127),
    lambda reg: reg["registered"]["stacked"].__setitem__("top", reg["registered"]["stacked"]["top"][:64]),
    lambda reg: reg["registered"]["stacked"]["top"].reverse(),
], ids=["node-0", "node-127", "half", "reversed"])
def test_an_r_whose_top_is_not_the_registered_one_is_refused(tmp_path, change):
    """R's binding round must commit exactly the registered `b ‖ c`, node for node: a registration one bit off in its first
    or last node, of half the nodes, or in another order is refused (as is R's top changed after registration)."""
    reg = _registration_of(json.loads((_fixture(C3) / "public.json").read_text()))
    change(reg)
    rc, line, err = _read(C3, registration=_write(tmp_path / "registration.json", reg))
    assert (rc, line, err) == (1, None, "rec inner transcript: stacked/rep0's binding round commits a top other than R's "
                                        "registered top (--rec-inner-registration)")


def test_a_session_with_r_needs_rs_registered_top():
    """R has no root at `Commit`, so nothing but the registration binds it before the coins: a session with R's stream and
    no registered top is refused."""
    rc, line, err = _read(C3, registration=None)
    assert (rc, line, err) == (1, None, "rec inner transcript: stacked/rep0 binds R, and the verifier holds no registered "
                                        "top of R (--rec-inner-registration)")


@pytest.mark.parametrize("change,why", [
    (lambda r: r.__setitem__("kind", "verity/registration/v2"), "not a rec/inner-registration/v0 record"),
    (lambda r: r.__setitem__("tops", "hm96-sha512/v1/row"), "its tops are not salted as hm96-sha512/v1/node-row"),
    (lambda r: r["registered"].__setitem__("circuit", r["registered"]["stacked"]),
     "it registers the tops of #[circuit, stacked], not R's (stacked) alone"),
    (lambda r: r.__setitem__("registered", {}), "it registers the tops of #[], not R's (stacked) alone"),
    (lambda r: r["registered"]["stacked"].__setitem__("top", []), "R's top has no nodes"),
    (lambda r: r["registered"]["stacked"].__setitem__("top", "00"), "R's top is not a list of b ‖ c"),
    (lambda r: r["registered"]["stacked"]["top"].__setitem__(3, "00" * 64),
     "R's registered top: a b ‖ c of 64 bytes, not 128"),
], ids=["kind", "tops", "two-tables", "no-table", "no-nodes", "not-a-list", "short-node"])
def test_the_registration_is_held_to_its_format(tmp_path, change, why):
    """`RecVStar.registration` reads `rec_live.registration_record`'s format and nothing else (`REGISTRATION`)."""
    reg = _registration_of(json.loads((_fixture(C3) / "public.json").read_text()))
    change(reg)
    rc, line, err = _read(C3, registration=_write(tmp_path / "registration.json", reg))
    assert (rc, line, err) == (1, None, f"rec inner registration: {why}")


def test_the_registration_goes_with_the_transcript():
    p = subprocess.run([str(BIN), "rec-transcript", "--rec-inner-session", str(_fixture(C3) / "prover"),
                        "--rec-inner-registration", str(_registration())], capture_output=True, text=True)
    assert (p.returncode, p.stderr.strip()) == (1, "--rec-inner-registration goes with --rec-inner-transcript"), p.stderr


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
    why = ("rec inner transcript: the registration has R's top, and the session has no stacked/rep0: R's stream is missing "
           "(a downgrade)")
    assert _read(_fixture(PLAIN), registration=_registration()) == (1, None, why)
    t = json.loads((PLAIN / "public.json").read_text())
    t["streams"]["stacked/rep0"] = json.loads((C3 / "public.json").read_text())["streams"]["stacked/rep0"]
    assert _read(PLAIN, _write(tmp_path / "plain.json", t), registration=_registration()) == (1, None, why)


#: the bucket's RecSparse shape b907 ran its sparse phase at, and META's range of its inner statement's hidden circuit
#: (`pod/rec_inner.py stage --n 16 --plain-leaves`, circuit_sha512 e1148161…, which `rec_vstage stage --circuit` reads)
SPARSE = (18, 13, 6, 0)
RANGE = {"count": 64, "net": "unit", "pos0": 256, "slot_log": 13, "type": "unit"}


@pytest.fixture(scope="module")
def sparse(tmp_path_factory) -> Path:
    """b907's sparse phase staged as its RecSparse statement over both reps, as `rec_vstage stage --sparse` stages it: its
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
                                         *_reg_args(_registration())]
    return subprocess.run([str(BIN), "statement", "--statement", "verity/flock-circuit", "--circuit", str(d / "circuit.txt"),
                           "--public", str(d / "pub.bin"), "--partition", str(d / "partition.json"), "--program",
                           str(d / "program.json"), "--registered", str(d / "registered.json"), *rec],
                          capture_output=True, text=True, check=False)


@pytest.mark.slow
def test_a_rec_sparse_statements_message_rows_are_the_sparse_streams_c_k(sparse, tmp_path):
    """`RecVStar.checkPubRows` on RecSparse (`fixedRow`'s `rec-sparse-<p>`): b907's statement, about 300 MB at
    n18-e13-s6-t0, is the only RecSparse statement whose message rows a coin server recorded, so nothing smaller shows
    the verifier reads `sparse/rep<x>`'s rounds at the ports' order. `statement` with the inner transcript sets it up
    (its `rec-sparse-v` compared with no round), and refuses it at a transcript whose `sparse/rep1`
    `c_k` of `u` is one bit off; without the transcript it sets up as before."""
    unit = f"rec-sparse/n{SPARSE[0]}-e{SPARSE[1]}-s{SPARSE[2]}-t{SPARSE[3]}"
    ports = {k: x["value"] for k, x in json.loads((sparse / "registered.json").read_text())["ports"].items()}
    assert {"rec-sparse-u", "rec-sparse-v"} <= set(ports.values()), ports
    (port,) = [k for k, v in ports.items() if v == "rec-sparse-u"]
    for t in (None, _fixture(C3) / "public.json"):
        p = _statement(sparse, t)
        assert p.returncode == 0, p.stdout + p.stderr

    def flip(t):
        r = [r for r in sorted(t["rounds"], key=lambda r: r["k"]) if r["stream"] == "sparse/rep1" and r["commits"]][-1]
        c = bytearray.fromhex(r["commits"][0])
        c[0] ^= 1
        r["commits"] = [c.hex()]
        flip.k = r["k"]

    p = _statement(sparse, _changed(tmp_path, flip))
    assert p.returncode == 1, p.stdout + p.stderr
    assert re.fullmatch(rf"setup: rec-pub: {re.escape(unit)}: port {port}'s table row \d+ is not sparse/rep1's c_31 \(u, inner "
                        rf"round {flip.k}\) of the inner transcript", p.stdout.strip()), p.stdout
