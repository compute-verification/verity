"""`flock-circuit prove --zk --stream -` behind the ZK server (`ZK_SERVER`, live PROTOCOL.md §10.4), on real builds.
Three members, sessions of `plain_hidden.py`'s statement at its two committed input sets, each on a sample of its own,
are proved through SPOT's proof server (`verity_flock.proof_server`: one kept `prove --stream` process, its verifier on a
challenger), all three submitted before the first is awaited; and each member once more as the process pair a session
behind the ZK server otherwise is, one `serve` and one `prove` process on a challenger of their own.

* **A member through the stream is the process pair's member, less the seconds** (`differ`). A session's coins,
  commitments and proofs are its own, so no two sessions are equal byte for byte. What is held of the stream's record
  (`session.json`), its `LIVE` line and Lean's verdict on its record (`VERIFY`'s `verify --zk`) against the pair's:
  every key, every list's length and every value is the pair's, but a clock's reading (a float) and a field word or digest
  the session's coins decide (lowercase hex of 32 digits or more, compared by its length). Lean's verdict names the
  record it read, each side's own directory. A `LIVE` line's fields about its process and not its session are held on
  their own (`OWN`, `TEXT`), and the digests its statement decides are equal (`DECIDED`). The Lean it ran is the pair's
  as well: one tap for the process, then for each member one session loop, started with the pair's arguments, so on
  that member's own circuit and tables.
* **Member i + 1's first coin follows member i's closing round in the challenger's release log** (`order_reason`). A
  member's closing round is the last round the challenger opened of its coin tree. The verifier has that round's coins
  only from the challenger's answer, which the challenger writes the round's line before; the prover's proofs follow those
  coins and the verdict follows the proofs, so a member's verdict follows its closing round's receipt. The stream starts
  member i + 1's loop only once member i's verdict is emitted, so no line of member i + 1's tree, the one that samples
  it at its `Hello` included, is before member i's closing round.
* **Both paths' lines name their seconds** (`seconds_reason`). A proved session's `LIVE` line holds ``steps_s``: the
  prover's seconds from the instant before `Register` to the start of its repetitions, a lap of one clock a step
  (`STEPS`) and their ``total``. Its server's `SESSION` line holds where the session's thread spent its seconds
  (`BUSY`). Each is a float on the stream's lines and on the pair's, whatever the session ran: a step a session did not
  take reads 0.0, never null and never an integer, so `differ` and every reader take it as a clock's reading.
* **It refuses to verify ahead**: with ``FC_VERIFY_AHEAD`` above 0 a session would start before the verdict of the one
  before it.

These skip unless ``FLOCK_CIRCUIT`` names a proving build of `flock-circuit`
(`verity/protocols/flock/pod/94-stream-zk-server.sh` builds one and runs them); they build the Lean verifier's two
executables (`VERIFY`, `LOOP`) and the challenger's Lean. No stand-in shows what they hold, which is what the real
prover, the real Lean loop and the real challenger do in six sessions. The two rules themselves (`differ`,
`order_reason`) are tested on values written here, with no build."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest
from verity.protocols.spot.device import challenger as C, remote as R

import settings
from verity_flock import challengers as CH
from verity_flock.proof_server import ProofServer, Statement

LEAN = Path(__file__).resolve().parents[2] / "verifier"
#: the Lean verifier, and the ZK server's Lean: its tap and its session loop
VERIFY = LEAN / ".lake" / "build" / "bin" / "flock-verify"
LOOP = LEAN / ".lake" / "build" / "bin" / "flock-firewall"
BIN = settings.flock_circuit()
#: what puts a `--zk` prover behind the ZK server
ZK_SERVER = {"FC_FIREWALL": "1"}
real = pytest.mark.skipif(BIN is None or shutil.which("lake") is None,
                          reason="FLOCK_CIRCUIT names no proving build of flock-circuit, or there is no Lean toolchain")

#: `plain_hidden.py`'s committed input sets
SEEDS = (20261025, 20261020)
#: each member: its input set, and the units of its 32 that its session samples
MEMBERS = ((0, list(range(32))), (1, list(range(0, 32, 2))), (0, list(range(16, 32))))
#: what a `LIVE` line says of the process that proved its session: the session's place among the process's, the stream's
#: name for it, and the process's statement cache so far
OWN = ("run", "label", "stmt_cache")
#: the bytes the prover received, the verdict's seconds as text among them: a count on both sides
TEXT = "wire_recv"
#: the `LIVE` line's digests that the statement and its public file decide
DECIDED = ("circuit_sha512", "statement_digest", "public_sha512", "public_file_sha512")
#: a field word, or a digest, root, key or message of them, in a record's hex
WORDS = re.compile(r"[0-9a-f]{32,}")
#: the steps of a `LIVE` line's ``steps_s``, in the prover's order: the `Register` round trip, the drawn statement's
#: tables, connecting the ZK server's Lean finish, the session's configuration and what it starts beside the witness, the
#: witness (``witness_s`` is inside it), the waits for level 0's salts and for rep 0's pads, the next session's level 0
#: started, and the wait for the previous session's repetitions
STEPS = ("register", "tables", "lean_finish", "config", "witness", "level0_wait", "zk_pads_wait", "level0_next", "device_wait")
#: a `SESSION` line's seconds of its session's thread: building its server, writing its record and proofs, waiting for
#: the prover's frames, and all of it but that wait
BUSY = ("build_s", "record_s", "prover_wait_s", "busy_s")
#: what the clock reads between two laps may add up to, in seconds
SLACK = 1e-3


def differ(a, b, at: tuple = ()) -> list[str]:
    """Where `b` is not `a` less the seconds and what a session's coins decide, each place by its keys and indices; empty
    when it is. Two floats are two readings of a clock; two strings of one length in lowercase hex, 32 digits or more,
    are two sessions' field words or digests; two strings that open with a brace each hold a JSON object (a record's
    `Hello`) and are that object, and one that is no JSON raises. Everything else is equal: keys, lengths, counts, names,
    verdicts."""
    here = "/".join(map(str, at)) or "(the whole)"
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return [f"{here}: {sorted(a.keys() ^ b.keys())} in one alone"]
        return [w for k in a for w in differ(a[k], b[k], (*at, k))]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{here}: {len(a)} against {len(b)}"]
        return [w for i, (x, y) in enumerate(zip(a, b)) for w in differ(x, y, (*at, i))]
    if isinstance(a, float) and isinstance(b, float):
        return []
    if isinstance(a, str) and isinstance(b, str):
        if len(a) == len(b) and WORDS.fullmatch(a) and WORDS.fullmatch(b):
            return []
        if a.startswith("{") and b.startswith("{"):
            return differ(json.loads(a), json.loads(b), at)
    return [] if type(a) is type(b) and a == b else [f"{here}: {a!r} against {b!r}"]


def order_reason(log: Sequence[str], roots: Sequence[str]) -> str | None:
    """Why the challenger's release log `log` doesn't show the sessions on the coin trees under `roots`, in that order,
    one at a time; None when it does. A tree's lines are the line that samples it (`C.TREE_EVENT`, written at the
    session's `Hello`, before any coin of it) and a line for each round opened of it (`C.OPEN_EVENT`), the last its
    closing round. One at a time: every line of a tree follows the closing round of the tree before it."""
    docs = [d if isinstance(d, dict) else {} for d in map(json.loads, log)]
    lines = {r: [i for i, d in enumerate(docs)
                 if (d.get("event"), d.get("root")) == (C.TREE_EVENT, r) or (d.get("event"), d.get("tree")) == (C.OPEN_EVENT, r)]
             for r in roots}
    for k, r in enumerate(roots):
        if len(lines[r]) < 2 or docs[lines[r][0]]["event"] != C.TREE_EVENT:
            return f"member {k}: the log shows no coin tree under its root sampled and then opened"
    for k in range(1, len(roots)):
        closing, first = lines[roots[k - 1]][-1], lines[roots[k]][0]
        if first < closing:
            return f"member {k}'s first line is line {first}, before member {k - 1}'s closing round, line {closing}"
    return None


def seconds_reason(live: dict, served: dict) -> str | None:
    """Why a proved session's `LIVE` line `live` and its server's `SESSION` line `served` don't name their seconds as
    every session's do; None when they do. ``steps_s`` holds each step of `STEPS` and ``total`` and nothing else, and
    the `SESSION` line each field of `BUSY`: every one a float, none below 0. The steps are laps of one clock, so they
    add up to the total, which is inside ``e2e_s`` and holds what ``e2e_s`` has before ``tp_s`` starts; the witness's
    lap holds ``witness_s``; and the busy seconds hold the build's, the record's, ``handle_s`` and ``verify_s``."""
    steps = live.get("steps_s")
    if not isinstance(steps, dict) or set(steps) != {*STEPS, "total"}:
        return f"the LIVE line's steps_s is {sorted(steps) if isinstance(steps, dict) else steps!r}, not {[*STEPS, 'total']}"
    seconds = {**{f"steps_s/{k}": v for k, v in steps.items()}, **{k: served.get(k) for k in BUSY}}
    for name, v in seconds.items():
        if type(v) is not float or v < 0:
            return f"{name} is {v!r}, not a float of 0 or more"
    laps = sum(steps[k] for k in STEPS)
    if abs(laps - steps["total"]) > SLACK:
        return f"steps_s's steps add up to {laps}, not its total {steps['total']}"
    if not live["e2e_s"] - live["tp_s"] - SLACK <= steps["total"] <= live["e2e_s"]:
        return f"steps_s/total {steps['total']} is not between e2e_s - tp_s = {live['e2e_s'] - live['tp_s']} and e2e_s = {live['e2e_s']}"
    if live["witness_s"] > steps["witness"] + SLACK:
        return f"witness_s {live['witness_s']} is not inside steps_s/witness {steps['witness']}"
    parts = sum(served[k] for k in ("build_s", "record_s", "handle_s", "verify_s"))
    if parts > served["busy_s"] + SLACK:
        return f"the SESSION line's build_s, record_s, handle_s and verify_s add up to {parts}, above its busy_s {served['busy_s']}"
    return None


def test_seconds_reason_names_a_field_that_is_missing_or_no_float_or_a_sum_that_is_off():
    steps = {**dict.fromkeys(STEPS, 0.0), "register": 0.5, "lean_finish": 20.0, "witness": 0.75, "total": 21.25}
    live = {"e2e_s": 60.0, "tp_s": 39.0, "witness_s": 0.5, "steps_s": steps}
    served = {"handle_s": 0.125, "verify_s": 0.25, "build_s": 0.5, "record_s": 0.125, "prover_wait_s": 59.0, "busy_s": 1.0}
    assert seconds_reason(live, served) is None
    names = [*STEPS, "total"]
    assert seconds_reason({**live, "steps_s": None}, served) == f"the LIVE line's steps_s is None, not {names}"
    less = {k: v for k, v in steps.items() if k != "level0_next"}
    assert seconds_reason({**live, "steps_s": less}, served) == f"the LIVE line's steps_s is {sorted(less)}, not {names}"
    more = {**steps, "hello": 0.0}
    assert seconds_reason({**live, "steps_s": more}, served) == f"the LIVE line's steps_s is {sorted(more)}, not {names}"
    # a step the session did not take is 0.0: never null, and never an integer
    for v in (None, 0, -0.5, "0.0"):
        assert seconds_reason({**live, "steps_s": {**steps, "device_wait": v}}, served) \
            == f"steps_s/device_wait is {v!r}, not a float of 0 or more"
        assert seconds_reason(live, {**served, "record_s": v}) == f"record_s is {v!r}, not a float of 0 or more"
    assert seconds_reason(live, {k: v for k, v in served.items() if k != "busy_s"}) == "busy_s is None, not a float of 0 or more"
    assert seconds_reason({**live, "steps_s": {**steps, "config": 0.25}}, served) \
        == "steps_s's steps add up to 21.5, not its total 21.25"
    assert seconds_reason({**live, "tp_s": 38.5}, served) \
        == "steps_s/total 21.25 is not between e2e_s - tp_s = 21.5 and e2e_s = 60.0"
    assert seconds_reason({**live, "witness_s": 1.0}, served) == "witness_s 1.0 is not inside steps_s/witness 0.75"
    assert seconds_reason(live, {**served, "busy_s": 0.5}) \
        == "the SESSION line's build_s, record_s, handle_s and verify_s add up to 1.0, above its busy_s 0.5"


def test_differ_takes_a_sessions_own_coins_and_seconds_and_nothing_else():
    root = "ab" * 64
    pair = {"accepted": True, "root": root, "s": 1.5, "n": 6, "rounds": [{"coins": ["0" * 32, "1" * 32]}], "why": "",
            "hello": json.dumps({"nonce": "2" * 64, "m": 26})}
    assert differ(pair, {**pair, "root": "cd" * 64, "s": 0.25, "rounds": [{"coins": ["3" * 32, "4" * 32]}],
                         "hello": json.dumps({"nonce": "5" * 64, "m": 26})}) == []
    assert differ(pair, {**pair, "accepted": False}) == ["accepted: True against False"]
    assert differ(pair, {**pair, "n": 7}) == ["n: 6 against 7"]
    assert differ(pair, {**pair, "why": "S2"}) == ["why: '' against 'S2'"]
    assert differ(pair, {**pair, "s": None}) == ["s: 1.5 against None"]
    assert differ(pair, {**pair, "root": root[:-2]}) == [f"root: {root!r} against {root[:-2]!r}"]
    assert differ(pair, {**pair, "rounds": [{"coins": ["0" * 32]}]}) == ["rounds/0/coins: 2 against 1"]
    assert differ(pair, {**pair, "hello": json.dumps({"nonce": "2" * 64, "m": 25})}) == ["hello/m: 26 against 25"]
    assert differ(pair, {**pair, "label": "m0"}) == ["(the whole): ['label'] in one alone"]
    # a short name in hex digits is a name, not a word
    assert differ({"unit": "cafe"}, {"unit": "face"}) == ["unit: 'cafe' against 'face'"]
    # a string that opens with a brace is read as the JSON it holds, and is not taken as text where it holds none
    with pytest.raises(ValueError):
        differ({"hello": "{no JSON"}, {"hello": "{no JSON"})


def test_order_reason_names_a_member_whose_first_line_is_before_the_closing_round_of_the_one_before():
    def tree(r):
        return json.dumps({"event": C.TREE_EVENT, "root": r})

    def opened(r, k):
        return json.dumps({"event": C.OPEN_EVENT, "tree": r, "round": k})

    head = json.dumps({"event": "commitment", "root": "a"})
    log = [head, tree("a"), opened("a", 0), opened("a", 1), tree("b"), opened("b", 0), opened("b", 1)]
    assert order_reason(log, ["a", "b"]) is None
    assert order_reason(log, ["b", "a"]) == "member 1's first line is line 1, before member 0's closing round, line 6"
    early = [head, tree("a"), opened("a", 0), tree("b"), opened("a", 1), opened("b", 0), opened("b", 1)]
    assert order_reason(early, ["a", "b"]) == "member 1's first line is line 3, before member 0's closing round, line 4"
    coin = [head, tree("a"), tree("b"), opened("a", 0), opened("b", 0), opened("a", 1), opened("b", 1)]
    assert order_reason(coin, ["a", "b"]) == "member 1's first line is line 2, before member 0's closing round, line 5"
    assert order_reason(log[:5], ["a", "b"]) == "member 1: the log shows no coin tree under its root sampled and then opened"


def _stamping(d: Path) -> Path:
    """`LOOP` behind a script that first writes its arguments as a line of ``d/loops.txt``: which Lean processes a prover
    started, in order, and on what."""
    d.mkdir(parents=True, exist_ok=True)
    script = d / "loop.sh"
    script.write_text(f"#!/bin/sh\nprintf '%s\\n' \"$*\" >> {shlex.quote(str(d / 'loops.txt'))}\nexec {shlex.quote(str(LOOP))} \"$@\"\n")
    script.chmod(0o755)
    return script


def _loops(d: Path) -> list[list[str]]:
    """The Lean processes `_stamping` under `d` started, each by its arguments less the two that name the prover's
    process and not the member: the verifier's address, and the public file, which the prover writes for the loop from
    the member's instances."""
    out = []
    for line in (d / "loops.txt").read_text().splitlines():
        args = line.split()
        for flag in ("--verifier", "--public"):
            if flag in args:
                args[args.index(flag) + 1] = "-"
        out.append(args)
    return out


def _verdict(st: Statement, stage: Path, session: Path) -> dict:
    """Lean's verdict on a session's record."""
    p = subprocess.run([str(VERIFY), "verify", "--zk", "--session-tables", "1", "--statement", "verity/flock-circuit",
                        "--circuit", str(st.circuit), "--public", str(st.public), "--partition", str(stage / "partition.json"),
                        "--program", str(stage / "program.json"), "--session", str(session)], capture_output=True, text=True, check=False)
    lines = [x.split(" ", 1)[1] for x in p.stdout.splitlines() if x.startswith("VERDICT ")]
    assert len(lines) == 1, (p.returncode, p.stdout[-600:], p.stderr[-600:])
    return json.loads(lines[0])


def _pair(env: dict, st: Statement, sample: dict, d: Path) -> dict:
    """One member as a process pair: `serve` for one session at `sample` on a challenger of its own, and `prove`. Its
    `LIVE` line, its server's `SESSION` line, its record's directory and its challenger's audit."""
    d.mkdir(parents=True, exist_ok=True)
    (d / "sample.json").write_text(json.dumps(sample))
    ch, flags = CH.start(d / "challenger")
    serve = None
    try:
        with open(d / "serve.err", "w") as err:
            serve = subprocess.Popen([str(BIN), "serve", "--zk", "--listen", "127.0.0.1:0", "--out", str(d / "sessions"),
                                      "--instances", str(st.public), "--circuit", str(st.circuit), "--tables", str(st.tables),
                                      "--pin", st.pin, "--operator", "proof-server", "--draw-file", str(d / "sample.json"),
                                      "--sessions", "1", *flags], stdout=subprocess.PIPE, stderr=err, text=True, env=env)
        addr = next((x.split()[1] for x in serve.stdout if x.startswith("SERVING ")), None)
        assert addr, (d / "serve.err").read_text()[-600:]
        p = subprocess.run([str(BIN), "prove", "--zk", "--verifier", addr, "--instances", str(st.instances),
                            "--circuit", str(st.circuit), "--tables", str(st.tables), "--runs", "1", "--warm", "0"],
                           capture_output=True, text=True, env=env, check=False)
        (d / "prove.out").write_text(p.stdout)
        (d / "prove.err").write_text(p.stderr)
        if p.returncode != 0:
            serve.kill()
        (d / "serve.out").write_text(serve.stdout.read())
        serve.wait()
    finally:
        if serve is not None and serve.poll() is None:
            serve.kill()
            serve.wait()
        records = CH.records(d / "sessions")
        audit = CH.end(d / "challenger", ch, records)
    lives = [json.loads(x.split("\t", 1)[1]) for x in p.stdout.splitlines() if x.startswith("LIVE\t")]
    served = [json.loads(x.split("\t", 1)[1]) for x in (d / "serve.out").read_text().splitlines() if x.startswith("SESSION\t")]
    assert len(lives) == 1 and len(served) == 1 and len(records) == 1, (p.returncode, p.stdout[-600:], p.stderr[-600:])
    return {"live": lives[0], "served": served[0], "dir": Path(records[0]), "challenger": audit, "loops": _loops(d)}


@pytest.fixture(scope="module")
def window(tmp_path_factory):
    """The three members through the stream, and each as a process pair."""
    import plain_hidden as PH

    subprocess.run(["lake", "build", VERIFY.name, LOOP.name], cwd=LEAN, check=True, capture_output=True)
    d = tmp_path_factory.mktemp("stream-zk-server")
    # the Lean the challengers and their audits run, built as a pod builds it
    built = subprocess.run(["bash", "-c", '. "$0" && challenger_build && challenger_built', str(CH.SH)], capture_output=True,
                           text=True, check=False, env=CH._env(d, d / "challenger-lean"))
    assert built.returncode == 0, built.stderr[-600:]
    stages = [d / f"stage-{seed}" for seed in SEEDS]
    for stage, seed in zip(stages, SEEDS):
        PH.stage(stage, seed)
    statements = [Statement(hashlib.sha512((s / "circuit.txt").read_bytes()).hexdigest(), s / "circuit.txt", s / "tables",
                            s / f"inst-{PH.N}.bin", s / f"pub-{PH.N}.bin") for s in stages]
    # one circuit, so one verifier and one challenger's log for the stream; and one program and partition, so one process's
    # `Hello` flags serve every member
    assert len({st.pin for st in statements}) == 1
    assert all((s / f).read_bytes() == (stages[0] / f).read_bytes() for s in stages for f in ("partition.json", "program.json"))
    assert statements[0].instances.read_bytes() != statements[1].instances.read_bytes()
    members = [(statements[i], {"k": len(units), "law": "subset", "population": PH.N, "units": units}) for i, units in MEMBERS]

    def env(at: Path) -> dict:
        return {**os.environ, **ZK_SERVER, "FC_FIREWALL_LEAN": str(_stamping(at)),
                "FC_FIREWALL_HELLO_ARGS": f"--partition {stages[0] / 'partition.json'} --program {stages[0] / 'program.json'}"}

    out = d / "stream"
    with ProofServer(BIN, out / "server", env=env(out), challenger=True, verify_ahead=0) as server:
        sessions = [server.submit(st, sample, f"m{k}", out=out / f"m{k}") for k, (st, sample) in enumerate(members)]
        for s in sessions:
            s.result()
    (name,) = server.challengers
    stream = [{"live": s.live, "served": s.served, "dir": s.dir, "challenger": s.challenger} for s in sessions]
    pairs = [_pair(env(d / f"pair-m{k}"), st, sample, d / f"pair-m{k}") for k, (st, sample) in enumerate(members)]
    for (st, _), *both in zip(members, stream, pairs):
        for one in both:
            one["record"] = json.loads((one["dir"] / "session.json").read_text())
            one["verdict"] = _verdict(st, stages[0], one["dir"])
    return {"members": members, "stream": stream, "pairs": pairs, "exits": server.exits, "loops": _loops(out),
            "log": R.log_lines(out / "server" / f"{name}-challenger" / R.RELEASES)}


@real
def test_a_member_through_the_stream_is_the_process_pairs_member_less_the_seconds(window):
    assert set(window["exits"].values()) == {0}, window["exits"]
    # the Lean the stream started: its tap, then a session loop a member, each the one its process pair started
    assert window["loops"] == [["tap"], *(pair["loops"][1] for pair in window["pairs"])], window["loops"]
    for k, ((st, sample), got, pair) in enumerate(zip(window["members"], window["stream"], window["pairs"])):
        tap, loop = pair["loops"]
        assert tap == ["tap"] and loop[0] == "session", pair["loops"]
        assert (loop[loop.index("--circuit") + 1], loop[loop.index("--tables") + 1]) == (str(st.circuit), str(st.tables))
        for one in (got, pair):
            assert one["live"]["accepted"] is True and one["verdict"]["accepted"] is True and one["challenger"]["ok"] is True, (k, one["live"].get("why"), one["verdict"].get("why"), one["challenger"])
            assert one["record"]["unit_draw"] == sample, k
            assert seconds_reason(one["live"], one["served"]) is None, k
        assert differ(pair["record"], got["record"]) == [], k
        # Lean's verdict names the record it read, each side's own
        assert [one["verdict"]["session"] for one in (pair, got)] == [str(one["dir"]) for one in (pair, got)], k
        assert differ(*({f: v for f, v in one["verdict"].items() if f != "session"} for one in (pair, got))) == [], k
        session = [{f: v for f, v in one["live"].items() if f not in (*OWN, TEXT)} for one in (pair, got)]
        assert differ(*session) == [], k
        assert [got["live"][f] for f in DECIDED] == [pair["live"][f] for f in DECIDED], k
        assert type(got["live"][TEXT]) is type(pair["live"][TEXT]) is int, k
        assert (pair["live"]["run"], pair["live"].get("label")) == (0, None) and (got["live"]["run"], got["live"]["label"]) == (k, f"m{k}")
        assert got["live"]["stmt_cache"].keys() == pair["live"]["stmt_cache"].keys(), k


@real
def test_member_i_plus_1s_first_coin_follows_member_is_closing_round_in_the_challengers_log(window):
    records = [got["record"] for got in window["stream"]]
    roots = [rec["coin_tree"]["root"] for rec in records]
    assert order_reason(window["log"], roots) is None
    # each closing round is the last round of its stream in its member's record, which holds that round's coins beside the
    # verdict on them; the challenger's audit holds every round of the record to the log's
    opened = [json.loads(line) for line in window["log"] if C.open_line(line)]
    for k, (rec, got) in enumerate(zip(records, window["stream"])):
        closing = [o for o in opened if o["tree"] == roots[k]][-1]
        rounds = next(s["rounds"] for s in rec["streams"] if s["stream"] == closing["stream"])
        assert (closing["round"], closing["after"]) == (len(rounds) - 1, rounds[-1]["msg_sha512"]), k
        assert closing["coins"] == C.coins_digest([int(c, 16) for c in rounds[-1]["coins"]]), k
        assert rec["verdict"]["accepted"] is True and got["challenger"]["ok"] is True, k


@real
def test_the_stream_behind_the_zk_server_refuses_to_verify_ahead():
    p = subprocess.run([str(BIN), "prove", "--zk", "--stream", "-"], input="", capture_output=True, text=True, check=False,
                       env={**os.environ, **ZK_SERVER, "FC_VERIFY_AHEAD": "2"})
    assert p.returncode == 2 and p.stdout.startswith("REFUSED\t") and "FC_VERIFY_AHEAD=0" in p.stdout, (p.stdout, p.stderr[-300:])
