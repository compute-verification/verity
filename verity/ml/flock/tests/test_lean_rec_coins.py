"""V*'s verifier-registered values against the verifier's own coins (`Flock/RecCoins.lean`, `verify --rec-inner-*`): the
smallest level of a recorded recursive session (class (16,16,8) at 64 instances: level 3,
`rec-open/l16-h4`, 106 openings), restaged by `rec_vstage.stage` from the fetched fixture `rec-coins/16-16-8-n64`: the inner
statement, the inner session's public record and the prover's, V*'s fold, points and running-sum salts, the level's
recorded `pub.bin` and `registered.json`, and its recorded outer session. The level and its outer session were proved
again at the current leaf scheme (the inner statement's leaves are unsalted, so the rest of the
session is the first record's), before the OS masks, as `verity/flock-circuit@cec3e347`. `verify` sets V*'s statements up only
as `verity/flock-circuit`, so the recorded outer session is refused at the record's first check (S2/R7) once every V*
check has passed at setup. The honest level passes setup at the honest inner session and the coin server's transcript of
it (`--rec-inner-transcript`), and is refused without either, at a transcript whose level-3 tops or link points are not
the ones it climbs to and the record's (`RecVStar.checkTopRows`, `RecVStar.innerLink`) or not in the coin server's
format (`RecSetup.checkReads`), and where V* registered its
`rec-coef` one bit off, at that opening, though its registration is consistent. Set up as another statement than V*'s `verity/flock-circuit`, or with a lookup table held,
the honest level is refused. Every verify names the session's statement (`--rec-part J`, its index in `rec-statements`'
list) and passes the verifier's own bindings (`--rec-bindings F`, `rec_bindings` of its partition object and program).
Staging the level takes about a minute and each verify sets up its 158 MB circuit, so the
module is slow; it catches what no test before it can: the Lean verifier's coefficients, positions and salts at real
coins, against the ones V*'s Python registers. The algebra's parts are staged too, for their verifier-registered `v` and
zero `out`, their message rows and their row ports, and every session's `rec-acc` reads, against the verifier's record of V*'s shared
rows (`--rec-rows`), and their rows of the chain. `--rec-inner-coins`, a record of what check 5 reads in place of the
inner statement, is refused (`RecCoins.refuseCoins`).
A level staged with bit rows, which hidden outputs admit and V*'s statement does not, is refused under the inner shape.
RecCap, staged over the honest inner session, passes setup at the shared rows registering level 0's cap and is refused
without them, at another registration, at a transcript whose `root_B` differs at a node, and without its reads.
At candidate 3's inner shape, each session's setup first holds the inner statement to check 5's sparse refusals
(`RecCoins.checkAtC3`), and refuses a session without the inner statement's files."""
from __future__ import annotations

import concurrent.futures as cf
import functools
import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
from pathlib import Path

import keyed
import pytest
from verity_flock import rec_bindings as RB
from verity_flock import rec_live as RL
from verity_flock import rec_outer as O
from verity_flock import rec_vstage as S

HERE = Path(__file__).resolve().parent
LEAN = HERE.parents[2] / "core" / "service" / "flock"
BIN = LEAN / ".lake" / "build" / "bin" / "flock-verify"
SET = HERE / "fixtures" / "rec-coins" / "16-16-8-n64"
LEVEL = 3
UNIT = "rec-open/l16-h4"
#: the population programs' version the recorded levels were staged at (`TemplateInstances_v0`, before v1), which every
#: staging here restages them at, so that they read the recorded `pub.bin` and `registered.json`
VERSION = 0
#: the recorded level's circuit (`v/L3/circuit.txt`), which its outer session proves
CIRCUIT_SHA512 = ("885c04a867b23838a460033b3cfcbd8cad28bc5f994dd10e907724d293479cf7a525e331ed7152ab5e451bb27f39905f695af6295408db"
                  "459a42e2dda3e07463")
#: the recorded outer session's refusal as `verity/flock-circuit`, after setup
S2_R7 = "S2/R7: the session parameters are not the verifier's"
#: a staging's key (`once`): the inner session and V*'s fold, points and salts it reads, and the import closure of this
#: module, which holds the code each staging runs
_KEY = functools.partial(keyed.key, SET / "inner-honest", SET / "fold-honest.json", SET / "points-honest.json",
                         SET / "acc-salts.bin", code=[__name__])

pytestmark = [pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain"), pytest.mark.slow]


@pytest.fixture(scope="module")
def built() -> None:
    assert (SET / "u" / "stage" / "circuit.txt").is_file(), (
        f"{SET} is not fetched: `uv run research data fetch-fixtures verity/ml/flock/tests/fixtures/rec-coins`")
    subprocess.run(["lake", "build", "flock-verify"], cwd=LEAN, check=True, capture_output=True)


@pytest.fixture(scope="module")
def forged(built, once) -> Path:
    """Level 3 staged with rep 0's opening 0 `coef` one bit off (bit 0 of its first word) where V* computes it, so its
    registration, the running sums and the prover's row all agree with it. The circuit, program and partition do not read
    the coefficients: they are the recorded level's."""
    honest = S.coefficients

    def coefficients(vf, qs):
        cs = honest(vf, qs)
        c = cs[(0, LEVEL, 0)]
        cs[(0, LEVEL, 0)] = [c[0] ^ 1, *c[1:]]
        return cs

    def make(d):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(S, "coefficients", coefficients)
            res = S.stage(SET / "inner-honest", json.loads((SET / "fold-honest.json").read_text()), d, levels=[LEVEL],
                          algebra=False, pts=json.loads((SET / "points-honest.json").read_text()), salts=SET / "acc-salts.bin",
                          version=VERSION)
        assert res["levels"][f"L{LEVEL}"]["opened"] == res["levels"][f"L{LEVEL}"]["instances"] == 106

    out = once("rec-coins-forged", make, _KEY) / f"L{LEVEL}"
    assert hashlib.sha512((out / "circuit.txt").read_bytes()).hexdigest() == CIRCUIT_SHA512
    return out


@pytest.fixture(scope="module")
def shape(built, tmp_path_factory) -> Path:
    """The verifier's own inner shape (`rec-inner`) from the inner statement, which `verify` needs for any V* statement,
    and beside it the coin server's record of the honest inner session in its own format (``transcript.json``,
    `rec_live.record_bytes`: the fixture's record predates it, and the verifier reads that format at the inner shape,
    `RecSetup.checkReads`)."""
    u = SET / "u" / "stage"
    p = keyed.run([str(BIN), "rec-inner", "--circuit", str(u / "circuit.txt"), "--public", str(u / "pub-64.bin"),
                   "--partition", str(u / "partition.json")], check=True)
    f = tmp_path_factory.mktemp("rec-inner") / "inner.json"
    f.write_text(p.stdout)
    _form(f.parent)
    honest = json.loads((SET / "inner-honest" / "public.json").read_text())
    (f.parent / "transcript.json").write_text(RL.record_bytes(honest))
    return f


#: the draw of every one of the inner statement's 64 units, in order, as `--rec-inner-draw` takes it
FULL_DRAW = {"law": "subset", "population": 64, "k": 64, "units": list(range(64))}


def _form(d: Path, draw: dict | None = None) -> list[str]:
    """How Flock's plain verifier sets the inner statement up (`RecSetup.checkForm`, beside its files): a
    `verity/flock-circuit+plain-leaves` statement on the coin server's coins at `draw` (every unit unless given), written
    to `d`."""
    f = d / "draw.json"
    if draw is not None or not f.exists():
        f.write_text(json.dumps(FULL_DRAW if draw is None else draw))
    return ["--rec-inner-statement", "verity/flock-circuit+plain-leaves", "--rec-inner-statement-coins", "os",
            "--rec-inner-draw", str(f)]


@functools.cache
def _statements(shape: str) -> list[str]:
    p = keyed.run([str(BIN), "rec-statements", "--rec-inner", shape], check=True)
    (line,) = [json.loads(ln[9:]) for ln in p.stdout.splitlines() if ln.startswith("RECVSTAR ")]
    return [s[0] for s in line["statements"]]


def _own(level: Path, shape: Path) -> list[str]:
    """`--rec-part J` and `--rec-bindings F` for the statement staged in `level`: `J` its unit's index in the verifier's list
    of V*'s statements at the inner shape (`rec-statements`), `F` the verifier's own bindings (`rec_bindings`) of its
    partition object and program."""
    names = _statements(str(shape))
    with (level / "circuit.txt").open() as f:
        unit = f.readline().split()[1]
    f = level / "rec-bindings.json"
    if not f.exists():  # written whole, by rename: another worker's verify may read it at once
        tmp = level / f"rec-bindings.{os.getpid()}.{threading.get_ident()}"
        tmp.write_text(json.dumps(RB.bindings(json.loads((level / "partition.json").read_text()),
                                              json.loads((level / "program.json").read_text()))))
        tmp.rename(f)
    return ["--rec-part", str(names.index(unit)), "--rec-bindings", str(f)]


def _files(level: Path, public: Path, registered: Path, shape: Path, rows: Path | None = None,
           statement: str = "verity/flock-circuit") -> list[str]:
    return ["--statement", statement, "--circuit", str(level / "circuit.txt"), "--public", str(public), "--partition",
            str(level / "partition.json"), "--program", str(level / "program.json"), "--registered", str(registered),
            "--rec-inner", str(shape), *_own(level, shape), *([] if rows is None else ["--rec-rows", str(rows)])]


def _run(level: Path, public: Path, registered: Path, shape: Path, inner: str | Path | None, *extra: str,
         session: Path = SET / "outer-L3" / "session", rows: Path | None = None,
         statement: str = "verity/flock-circuit", transcript: bool = True) -> subprocess.CompletedProcess:
    """`verify` of a staged statement; with an inner session, the coin server's record of the honest one
    (`--rec-inner-transcript`, whose link points and tops every V* statement is held to) unless `extra` names another or
    `transcript` is false."""
    u = SET / "u" / "stage"
    files = ["--rec-inner-circuit", str(u / "circuit.txt"), "--rec-inner-public", str(u / "pub-64.bin"),
             "--rec-inner-partition", str(u / "partition.json"), *_form(shape.parent)]
    held = ["--rec-inner-transcript", str(shape.parent / "transcript.json")]
    rec = [] if inner is None else [*files, "--rec-inner-session", str(SET / inner / "session"),
                                    *(held if transcript and "--rec-inner-transcript" not in extra else [])]
    return keyed.run([str(BIN), "verify", "--zk", *_files(level, public, registered, shape, rows, statement), *rec, *extra,
                      "--session", str(session)])


def _verify(level: Path, public: Path, registered: Path, shape: Path, inner: str | Path | None, *extra: str,
            session: Path = SET / "outer-L3" / "session", rows: Path | None = None,
            statement: str = "verity/flock-circuit", transcript: bool = True) -> dict:
    p = _run(level, public, registered, shape, inner, *extra, session=session, rows=rows, statement=statement,
             transcript=transcript)
    (v,) = [json.loads(line[8:]) for line in p.stdout.splitlines() if line.startswith("VERDICT ")]
    return v


def _acc_rows(dst: Path, level: Path) -> Path:
    """Shared rows (`--rec-rows`) whose `acc0` is staged level `level`'s `rec-acc` registration (its `acc` port), with its
    `out` port's registration of the table's tops `rec-top-L<l>`, which `checkShared` requires registered."""
    ports = json.loads((level / "registered.json").read_text())["ports"]
    shared = {"acc0": ports["acc"], **({ports["out"]["value"]: ports["out"]} if "out" in ports else {})}
    dst.write_text(json.dumps({"format": "verity/registered-reads/v0", "ports": shared}))
    return dst


def test_the_honest_level_passes_setup_at_the_honest_inner_session(forged, shape, tmp_path):
    """Its VERDICT names its part (`rec_part`), the SHA-512 of the shared rows file it read (`rec_rows_sha512`), of
    the verifier's inner shape file (`rec_inner_sha512`), and of the inner session's record and the coin server's
    transcript of it (`rec_inner_session_sha512`, `rec_inner_transcript_sha512`), which `rec-compose` holds every part
    to."""
    rows = _acc_rows(tmp_path / "rows.json", SET / "L3")
    v = _verify(forged, SET / "L3" / "pub.bin", SET / "L3" / "registered.json", shape, "inner-honest", rows=rows)
    assert (v["accepted"], v["why"]) == (False, S2_R7), v
    sha = {k: hashlib.sha512(f.read_bytes()).hexdigest() for k, f in (
        ("rec_rows_sha512", rows), ("rec_inner_sha512", shape),
        ("rec_inner_session_sha512", SET / "inner-honest" / "session" / "session.json"),
        ("rec_inner_transcript_sha512", shape.parent / "transcript.json"))}
    assert {k: v.get(k) for k in sha} == sha, v
    assert str(v["rec_part"]) == _own(forged, shape)[1], v


def test_rec_compose_refuses_parts_at_another_inner_shape_or_session(shape, tmp_path):
    """`flock-verify rec-compose` at the verifier's inner shape: one accepted log per V* statement, each naming the one
    rows file, the inner shape file and one inner session and transcript, composes, and names them; one part whose
    VERDICT names another inner shape file's SHA-512, and two parts from different inner sessions, are refused, as
    `Flock/Test/RecCompose.lean` and `test_pod_lean_verdicts.py`'s stand-in refuse them."""
    rows = _acc_rows(tmp_path / "rows.json", SET / "L3")
    hr, hi = (hashlib.sha512(f.read_bytes()).hexdigest() for f in (rows, shape))
    hs, ht = "ee" * 64, "11" * 64
    n = len(_statements(str(shape)))
    assert n > 1, n

    def compose(last: dict[str, str]) -> dict:
        logs = []
        for j in range(n):
            v = {"session": f"s{j}", "accepted": True, "why": "", "rec_part": j, "rec_rows_sha512": hr,
                 "rec_inner_sha512": hi, "rec_inner_session_sha512": hs, "rec_inner_transcript_sha512": ht,
                 **(last if j == n - 1 else {})}
            (log := tmp_path / f"part-{j}.log").write_text(f"VERDICT {json.dumps(v)}\n")
            logs.append(str(log))
        p = subprocess.run([str(BIN), "rec-compose", "--rec-inner", str(shape), "--rec-rows", str(rows), *logs],
                           capture_output=True, text=True)
        (line,) = [json.loads(ln[12:]) for ln in p.stdout.splitlines() if ln.startswith("REC-COMPOSE ")]
        assert p.returncode == (0 if line["accepted"] else 1), p
        return line

    assert compose({}) == {"accepted": True, "why": "", "statements": n, "parts": n, "rec_rows_sha512": hr,
                           "rec_inner_sha512": hi, "rec_inner_session_sha512": hs, "rec_inner_transcript_sha512": ht}
    log, other = tmp_path / f"part-{n - 1}.log", "cd" * 64
    for key, what, whose, h in (("rec_inner_sha512", "inner shape file", "the composition's", hi),
                                ("rec_inner_session_sha512", "inner session record", f"{tmp_path / 'part-0.log'}'s", hs),
                                ("rec_inner_transcript_sha512", "inner transcript", f"{tmp_path / 'part-0.log'}'s", ht)):
        line = compose({key: other})
        assert (line["accepted"], line["why"]) == (False, f"{log}: read the {what} {other}, not {whose} {what} {h}"), line
        assert (line["rec_rows_sha512"], line["rec_inner_sha512"], "rec_inner_session_sha512" in line) == (hr, hi, False)


def test_the_honest_level_is_refused_without_the_inner_session(forged, shape, tmp_path):
    v = _verify(forged, SET / "L3" / "pub.bin", SET / "L3" / "registered.json", shape, None,
                rows=_acc_rows(tmp_path / "rows.json", SET / "L3"))
    assert v["why"].startswith(f"setup: rec-coins: {UNIT}: the verifier holds no inner session"), v["why"]


def test_the_honest_level_is_refused_without_the_inner_transcript(forged, shape, tmp_path):
    """`RecVStar.checkTopRows`: level 3's `out` reads `rec-top-L3`, the tops its openings climb to, which only the coin
    server's record of the inner session fixes, so without it the honest level is refused at setup."""
    v = _verify(forged, SET / "L3" / "pub.bin", SET / "L3" / "registered.json", shape, "inner-honest",
                rows=_acc_rows(tmp_path / "rows.json", SET / "L3"), transcript=False)
    assert v["why"] == (f"setup: rec-tops: {UNIT}: the verifier holds no inner transcript (--rec-inner-transcript with "
                        f"--rec-inner-session), so no committed tops to check port out's rows of rec-top-L{LEVEL} against"), v["why"]


@pytest.mark.parametrize("rep", [0, 1])
def test_a_level_climbing_to_tops_the_coin_server_did_not_commit_is_refused(forged, shape, tmp_path, rep):
    """`RecVStar.checkTopRows` at a real level: the honest level 3 against an inner transcript whose rep `rep` level-3 top
    (its last salted top, 32 nodes; `rec-top-L3` is rep 0's, then rep 1's) has node 0's `b ‖ c` one bit off is refused at
    the table row that reads the node: the prover's registered top is not the one the coin server committed before the
    level's query coins. `root_B`, which only level 0 reads, is tested in `test_lean_registered_reads` and
    `Flock.Test.InnerLink`."""
    t = json.loads((SET / "inner-honest" / "public.json").read_text())
    r = max((r for r in t["rounds"] if r["stream"] == f"circuit/rep{rep}" and r["tops"]), key=lambda r: r["k"])
    c = bytearray.fromhex(r["tops"][0][0])
    c[0] ^= 1
    r["tops"][0][0] = c.hex()
    (tmp_path / "transcript.json").write_text(RL.record_bytes(t))
    v = _verify(forged, SET / "L3" / "pub.bin", SET / "L3" / "registered.json", shape, "inner-honest",
                "--rec-inner-transcript", str(tmp_path / "transcript.json"), rows=_acc_rows(tmp_path / "rows.json", SET / "L3"))
    assert re.fullmatch(rf"setup: rec-tops: {re.escape(UNIT)}: port out's table row \d+ is not node 0 of rep {rep}'s level "
                        rf"{LEVEL} top, which the coin server committed", v["why"]), v["why"]


def test_a_level_at_link_points_the_coin_server_did_not_draw_is_refused(forged, shape, tmp_path):
    """`RecVStar.innerLink`: check 5 takes the extra claims at the inner session record's link points, so the honest level
    against an inner transcript whose first link coin is one bit off from the record's is refused before setup."""
    t = json.loads((SET / "inner-honest" / "public.json").read_text())
    t["link"]["points"][0] = f"{int(t['link']['points'][0], 16) ^ 1:032x}"
    (tmp_path / "transcript.json").write_text(RL.record_bytes(t))
    p = _run(forged, SET / "L3" / "pub.bin", SET / "L3" / "registered.json", shape, "inner-honest",
             "--rec-inner-transcript", str(tmp_path / "transcript.json"), rows=_acc_rows(tmp_path / "rows.json", SET / "L3"))
    assert (p.returncode, p.stderr.strip()) == (2, "rec inner transcript: the inner session's record's link points are not "
                                                   "the ones the coin server drew (link.points): check 5 would take the extra "
                                                   "claims at points the prover chose")


def test_a_level_over_a_transcript_not_in_the_coin_servers_format_is_refused(forged, shape, tmp_path):
    """`RecSetup.checkReads` among `verify`'s inputs (`RecSetup.commits`): the honest level against the honest inner
    transcript's values written as compact JSON is refused before setup, since the recursion's transcript model reads
    the record in the coin server's own format."""
    t = json.loads((SET / "inner-honest" / "public.json").read_text())
    f = tmp_path / "transcript.json"
    f.write_text(json.dumps(t))
    p = _run(forged, SET / "L3" / "pub.bin", SET / "L3" / "registered.json", shape, "inner-honest",
             "--rec-inner-transcript", str(f), rows=_acc_rows(tmp_path / "rows.json", SET / "L3"))
    assert (p.returncode, p.stderr.strip()) == (2, "rec inner transcript: the coin server's record does not read in "
                                                   "its own format (RecRecord.readFull: rec_live.record_bytes, its "
                                                   "rounds first, then its kind, tops, Hello in its keys' order and "
                                                   "the link's roots and points, at json.dumps' indent=1)")


def _inner_drawn(tmp_path: Path, draw: dict) -> Path:
    """The honest inner session, its record given `draw` as its `unit_draw`."""
    inner = shutil.copytree(SET / "inner-honest", tmp_path / "inner-drawn")
    s = json.loads((inner / "session" / "session.json").read_text())
    s["unit_draw"] = draw
    (inner / "session" / "session.json").write_text(json.dumps(s))
    return inner


def test_an_inner_record_whose_draw_leaves_a_unit_out_is_refused(forged, shape, tmp_path):
    """`RecCoins.drawnSpec`: V* recurses on an inner session that proves every unit, so a record whose draw is a well
    formed sample of the 64 units is refused before its record is read."""
    inner = _inner_drawn(tmp_path, {"law": "subset", "population": 64, "k": 1, "units": [0]})
    v = _verify(forged, SET / "L3" / "pub.bin", SET / "L3" / "registered.json", shape, inner,
                rows=_acc_rows(tmp_path / "rows.json", SET / "L3"))
    assert v["why"] == (f"setup: rec-coins: {UNIT}: the inner session's draw: the session's draw proves 1 units, not "
                        "every one of the statement's 64 in order"), v["why"]


def test_an_inner_record_given_the_full_draw_is_read_at_the_drawn_statement(forged, shape, tmp_path):
    """`RecCoins.drawnSpec`: a record whose draw is every unit is decoded against the drawn statement's spec, whose
    header carries the draw. The honest session ran without one, so its record is refused there, at its parameters."""
    inner = _inner_drawn(tmp_path, {"law": "subset", "population": 64, "k": 64, "units": list(range(64))})
    v = _verify(forged, SET / "L3" / "pub.bin", SET / "L3" / "registered.json", shape, inner,
                rows=_acc_rows(tmp_path / "rows.json", SET / "L3"))
    assert v["why"] == f"setup: rec-coins: {UNIT}: the inner session's record: {S2_R7}", v["why"]


def test_a_coef_registered_one_bit_off_is_refused_at_its_opening(forged, shape, tmp_path):
    v = _verify(forged, forged / "pub.bin", forged / "registered.json", shape, "inner-honest",
                rows=_acc_rows(tmp_path / "rows.json", forged))
    assert v["why"] == (f"setup: rec-coins: {UNIT}: level {LEVEL} rep 0 opening 0: the statement's coef row is not the "
                        "verifier's rec-coef row 0")


def _stage(d: Path, **kw) -> dict:
    return S.stage(SET / "inner-honest", json.loads((SET / "fold-honest.json").read_text()), d,
                   pts=json.loads((SET / "points-honest.json").read_text()), salts=SET / "acc-salts.bin", version=VERSION, **kw)


@pytest.fixture(scope="module")
def dirs_forged(built, once) -> Path:
    """Level 3 staged with rep 0's opening 0 `dirs` registered one bit off (bit 0 of its first word), the opening's climb
    aside."""
    honest = S.R.dirs_words

    def make(d):
        first = []

        def dirs_words(pos, H):
            w = honest(pos, H)
            if not first:
                first.append(pos)
                w = [w[0] ^ 1, *w[1:]]
            return w

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(S.R, "dirs_words", dirs_words)
            _stage(d, levels=[LEVEL], algebra=False)

    return once("rec-coins-dirs", make, _KEY) / f"L{LEVEL}"


def test_a_dirs_registered_one_bit_off_is_refused_at_its_opening(dirs_forged, shape, tmp_path):
    """`HDirCols`' check (`RecCoins.checkOpen`'s `dirs`): level 3 staged with rep 0's opening 0 `dirs` registered one bit
    off, the opening's climb aside, is refused at that opening by the position the verifier's coins give."""
    out = dirs_forged
    v = _verify(out, out / "pub.bin", out / "registered.json", shape, "inner-honest", rows=_acc_rows(tmp_path / "rows.json", out))
    assert v["why"] == (f"setup: rec-coins: {UNIT}: level {LEVEL} rep 0 opening 0: the statement's dirs row is not the "
                        "verifier's rec-dirs row 0"), v["why"]


@pytest.fixture(scope="module")
def undrawn(tmp_path_factory) -> Path:
    """Level 3's outer session with its unit draw removed: `verify` sets a part's statement up over its whole population, so
    the V* checks run on it, and the session, not the part's, is refused after setup."""
    d = tmp_path_factory.mktemp("undrawn") / "session"
    shutil.copytree(SET / "outer-L3" / "session", d)
    s = json.loads((d / "session.json").read_text())
    s["unit_draw"] = None
    (d / "session.json").write_text(json.dumps(s))
    return d


@pytest.fixture(scope="module")
def algebra(built, once) -> Path:
    """V*'s algebra parts (`p<j>`) staged over the honest inner session."""
    return once("rec-coins-algebra", lambda d: _stage(d, levels=[], algebra=True, evaluate=False), _KEY) / "alg"


@pytest.fixture(scope="module")
def comb(built, once) -> Path:
    """The algebra staged with rep 0's first comb element of v one bit off (`rec_vstage --forge comb`)."""
    return once("rec-coins-comb", lambda d: _stage(d, levels=[], algebra=True, evaluate=False, forge="comb"), _KEY) / (
        "forged-comb/alg")


@pytest.fixture(scope="module")
def salted(built, once) -> Path:
    """Level 3 and the algebra staged under another salt of the running sums' row 0 (`_salted`)."""
    def make(d):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(S, "acc_salts", _salted)
            _stage(d, levels=[LEVEL], algebra=True, evaluate=False)

    return once("rec-coins-salted", make, _KEY)


@pytest.fixture(scope="module")
def level_major(built, once) -> Path:
    """Level 3 staged on running sums whose openings run level-major (`rec_outer.chain_order` by level, then rep and
    query): its `rec-acc` registration, reads file and `pub.bin`'s paths agree with that chain, whose level 3 starts at
    row `2 · 395`."""
    def order(queries):
        return sorted(((q.rep, q.level, q.j) for q in queries), key=lambda k: (k[1], k[0], k[2]))

    def make(d):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(O, "chain_order", order)
            _stage(d, levels=[LEVEL], algebra=False, evaluate=False)

    return once("rec-coins-level-major", make, _KEY) / f"L{LEVEL}"


def _rec_rows(dst: Path, parts: list[Path]) -> Path:
    """The verifier's record of V*'s shared rows (`--rec-rows`) from staged parts (`rec_vstage.shared_rows`), with the
    recorded level 3's registration of its tops `rec-top-L3`."""
    dst.write_text(json.dumps(S.shared_rows(parts, (), [SET / "L3"])))
    return dst


@pytest.fixture(scope="module")
def rows(algebra, tmp_path_factory) -> Path:
    """The verifier's shared rows of the honest algebra: its message rounds `m<k>` and the running sums' `acc0`, `acc1`, with
    level 3's tops `rec-top-L3`."""
    return _rec_rows(tmp_path_factory.mktemp("rec-rows") / "rows.json", sorted(algebra.iterdir()))


def test_a_v_registered_one_element_off_is_refused_at_its_part(comb, algebra, shape, undrawn, rows):
    """`HVCoins`' check (`RecCoins.checkResiduals`): the algebra staged with rep 0's first comb element of v one bit off
    (`rec_vstage --forge comb`) changes one part's registration, and that part is refused by the v the verifier's coins and
    fold give. Its one verify sets the part up over its whole population, minutes at one core: at this shape V*'s algebra
    is two parts of about 400 MB of circuit each, and no level reads v."""
    parts = sorted(comb.iterdir())
    changed = [p for p in parts if (p / "registered.json").read_bytes() != (algebra / p.name / "registered.json").read_bytes()]
    assert len(changed) == 1 and len(parts) > 1, changed
    (p,) = changed
    t = shape.parent / "transcript.json"
    why = _verify(p, p / "pub.bin", p / "registered.json", shape, "inner-honest", "--rec-inner-transcript", str(t),
                  session=undrawn, rows=rows)["why"]
    assert re.fullmatch(rf"setup: rec-coins: rec-residuals/m\d+-k\d+-c\d+-{p.name}: rep 0: part {p.name[1:]}'s \d+ elements of v: "
                        "the statement's v row is not the verifier's rec-v row 0", why), why


def test_a_parts_out_registered_under_another_zero_salt_is_refused_at_its_zero_rows(algebra, shape, undrawn, rows, once):
    """`ZeroReads`' check (`RecCoins.checkZero` at `rec-residuals-zero`): the algebra staged with the zero outputs' salt 0
    one bit off (`rec_residuals.zero_salts`) registers the same zero rows as another `rec-residuals-zero`, consistently,
    and its first part is refused at rep 0's `out` row by the salts the verifier computes. Its one verify sets the part up
    over its whole population, as the comb test's does; only a staged part reaches check 5's `out` rows."""
    honest = S.RR.zero_salts

    def zero_salts(value, n):
        ss = honest(value, n)
        return [bytes([ss[0][0] ^ 1]) + ss[0][1:], *ss[1:]]

    def make(d):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(S.RR, "zero_salts", zero_salts)
            _stage(d, levels=[], algebra=True, evaluate=False)

    p = sorted((once("rec-coins-zero", make, _KEY) / "alg").iterdir())[0]
    out = lambda d: json.loads((d / "registered.json").read_text())["ports"]["out"]["root"]
    assert out(p) != out(algebra / p.name)
    why = _verify(p, p / "pub.bin", p / "registered.json", shape, "inner-honest", "--rec-inner-transcript",
                  str(shape.parent / "transcript.json"), session=undrawn, rows=rows)["why"]
    assert why == (f"setup: rec-coins: {_unit(p)}: rep 0's zero outputs: the statement's out row is not the verifier's "
                   "rec-residuals-zero row 0"), why


#: RecCap at the fixture's inner shape (m 27, cap₀ 7): V*'s last statement, over the 128 nodes of level 0's cap
CAP = "rec-cap/n128"


@pytest.fixture(scope="module")
def cap(built, once) -> Path:
    """RecCap staged over the honest inner session (`rec_vstage.stage_cap`): instance `i` reads node `i` of `rec-top-L0`."""
    def make(d):
        res = S.stage_cap(SET / "inner-honest", d / "cap", levels=[0])["L0"]
        assert res["instances"] == 128 and res["unit_agrees"] == [True] * 128, res

    return once("rec-coins-cap", make, _KEY) / "cap"


def test_rec_cap_holds_every_node_of_level_0s_cap_at_the_level_0_registration(cap, shape, undrawn, tmp_path):
    """`RecCoins.checkCap`, `RecVStar.checkTopRows` and `checkShared` on RecCap: at the shared rows that register `rec-top-L0`
    as RecCap reads it (`rec_vstage.shared_rows`' tops, as level 0's `out` and RecCap's `top` register it) the honest
    statement passes setup (`undrawn`'s session is refused after it). Without that entry it is refused, since then nothing
    ties its rows to the ones level 0's openings read; at an entry of another root (level 0's queries reading another
    registration of the cap than RecCap holds) it is refused by the root; and at an inner transcript whose `root_B` differs
    at node 5 it is refused at the row reading that node, so every node RecCap holds is the coin server's. Each verify sets
    RecCap up over its 128 instances; they run at once."""
    rows = tmp_path / "rows.json"
    rows.write_text(json.dumps(S.shared_rows([], (), [cap])))
    top = json.loads((cap / "registered.json").read_text())["ports"]["top"]
    assert (top["value"], top["leaves"], top["reads"]) == ("rec-top-L0", 128, list(range(128)))
    flipped = top["root"][:-1] + ("1" if top["root"][-1] == "0" else "0")
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"format": "verity/registered-reads/v0", "ports": {"rec-top-L0": {**top, "root": flipped}}}))
    t = json.loads((SET / "inner-honest" / "public.json").read_text())
    root = t["link"]["roots"]["circuit"]
    assert len(root) == 128
    root[5] = f"{int(root[5][:2], 16) ^ 1:02x}" + root[5][2:]
    (tmp_path / "transcript.json").write_text(RL.record_bytes(t))
    run = lambda shared, *x: _verify(cap, cap / "pub.bin", cap / "registered.json", shape, "inner-honest", *x,  # noqa: E731
                                     session=undrawn, rows=shared)["why"]
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        honest, bare, off, forged = [ex.submit(run, *a) for a in (
            (rows,), (None,), (other,), (rows, "--rec-inner-transcript", str(tmp_path / "transcript.json")))]
    assert honest.result() == S2_R7, honest.result()
    assert bare.result() == (f"setup: {CAP}: its port top reads the table's tops rec-top-L0, which its level's RecCap holds; "
                             "pass the verifier's shared rows (--rec-rows F), which register it"), bare.result()
    assert off.result() == (f"setup: {CAP}: its port top reads rec-top-L0, which the verifier's shared rows (--rec-rows) "
                            f"register as rec-top-L0: root {top['root']}, against {flipped}"), off.result()
    assert re.fullmatch(rf"setup: rec-tops: {re.escape(CAP)}: port top's table row \d+ is not node 5 of circuit's root_B, "
                        "which the coin server committed", forged.result()), forged.result()


def test_rec_cap_is_refused_without_its_top_or_out_in_the_reads_file(cap, shape, undrawn, tmp_path):
    """`RecCoins.checkCap`'s reads: RecCap with `top` committed afresh (gone from its reads file and `pub.bin`'s registered
    rows) is refused, since its rows would then be the prover's alone; with its zero outputs `out` gone, likewise.
    `Flock.Test.RecCapSmall` guards the rest of `checkCap`'s refusals (a top at another value, size or reads)."""
    rows = tmp_path / "rows.json"
    rows.write_text(json.dumps(S.shared_rows([], (), [cap])))
    why = {}
    for port in ("top", "out"):
        d = _unregistered(cap, tmp_path / f"no-{port}", port)
        why[port] = _verify(d, d / "pub.bin", d / "registered.json", shape, "inner-honest", session=undrawn, rows=rows)["why"]
    assert why["top"] == (f"setup: rec-coins: {CAP}: the verifier's registered reads hold no port top: the cap's rows would be "
                          "the prover's"), why["top"]
    assert why["out"] == (f"setup: rec-coins: {CAP}: the verifier's registered reads hold no port out: its rec-cap-zero would be "
                          "the prover's"), why["out"]


#: `Test/RecCoins.lean`'s root of R's opening part's one zero row at program "p" (`RecCoins.checkOpening`'s `checkZero`)
OPENING_ZERO_ROOT = ("af8c0fa801627686aad3af3cc4c63840162705a903222f507a84c01e9f1414742bb6985e4eee425f0dd1539277a0b71fcb6d895abd7"
                     "bad14ce4f03dc487265b9")


def test_r_openings_zero_row_is_the_one_check_5_computes_and_another_salt_is_not():
    """`RecCoins.checkOpening`'s `checkZero` at `rec-residuals-zero`, one row: the root `rec_residuals.stage_opening`
    registers for R's opening part's `out` is the one the Lean verifier computes (its `#guard`), and the same zero row under
    salt 0 one bit off has another root, which check 5's root comparison refuses. No staged session reaches check 5 at R's
    opening here: its honest sessions are replayed on the pod."""
    from verity.core.protocols.sampled_proofs.one_stage import registered as RG

    ss = S.RR.zero_salts(S.RR.ZERO_VALUE, 1)
    commit = lambda salts: RG.Registered.commit("p", S.RR.ZERO_VALUE, [[0] * S.RR.OUT_WORDS], salts=salts).root.hex()
    assert commit(ss) == OPENING_ZERO_ROOT
    assert commit([bytes([ss[0][0] ^ 1]) + ss[0][1:]]) != OPENING_ZERO_ROOT


def _transcript(dst: Path, change) -> Path:
    t = json.loads((SET / "inner-honest" / "public.json").read_text())
    change(next(r for r in sorted(t["rounds"], key=lambda r: r["k"]) if r["stream"].endswith("/rep0") and r.get("commits")))
    dst.write_text(RL.record_bytes(t))
    return dst


def test_a_parts_message_rows_are_the_c_k_of_the_inner_transcript(algebra, shape, undrawn, rows, tmp_path):
    """`HPubRows`' check (`RecVStar.checkPubRows`): the `pub.bin` rows of the first algebra part that reads `rec-c0` are,
    at its `rec-c<k>` reads, the `c_k` the coin server recorded. The honest part passes setup with the inner transcript
    (`undrawn`'s session is refused after it), is refused without it, and is refused at a transcript whose rep 0 `c_0` is
    one bit off, or whose coins there are not the inner session's."""
    reads = lambda p: {k: x["value"] for k, x in json.loads((p / "registered.json").read_text())["ports"].items()}
    a = next(p for p in sorted(algebra.iterdir()) if "rec-c0" in reads(p).values())
    (port,) = [k for k, x in reads(a).items() if x == "rec-c0"]
    with (a / "circuit.txt").open() as f:
        unit = f.readline().split()[1]
    t = shape.parent / "transcript.json"

    def flip(r):
        c = bytearray.fromhex(r["commits"][0])
        c[0] ^= 1
        r["commits"] = [c.hex()]

    bad = _transcript(tmp_path / "c0.json", flip)
    with cf.ThreadPoolExecutor(max_workers=3) as ex:  # each verify sets the part up for minutes; none reuses another's
        honest, bare, c0 = [ex.submit(_verify, a, a / "pub.bin", a / "registered.json", shape, "inner-honest", *x,
                                      session=undrawn, rows=rows, transcript=bool(x))
                            for x in (["--rec-inner-transcript", str(t)], [], ["--rec-inner-transcript", str(bad)])]
    assert honest.result()["why"] == S2_R7, honest.result()["why"]
    v = bare.result()
    assert v["why"].startswith(f"setup: rec-pub: {unit}: the verifier holds no inner transcript"), v["why"]
    v = c0.result()
    assert re.fullmatch(rf"setup: rec-pub: {re.escape(unit)}: port {port}'s table row \d+ is not rep 0's c_0 \(inner round \d+\) "
                        "of the inner transcript", v["why"]), v["why"]
    bad = _transcript(tmp_path / "coins.json", lambda r: r.__setitem__("coins", ["00" * 16] * len(r["coins"])))
    p = _run(a, a / "pub.bin", a / "registered.json", shape, "inner-honest", "--rec-inner-transcript", str(bad), session=undrawn, rows=rows)
    assert p.returncode == 2 and "round" in p.stderr and "coins are not the inner session record's" in p.stderr, p.stderr


def _unit(d: Path) -> str:
    with (d / "circuit.txt").open() as f:
        return f.readline().split()[1]


def _salted(path, n):
    """The running sums' salts with salt 0 one bit off: another registration of the same rows (`rec_vstage.acc_salts`)."""
    ss = _honest_salts(path, n)
    return [bytes([ss[0][0] ^ 1]) + ss[0][1:], *ss[1:]]


_honest_salts = S.acc_salts


def _unregistered(src: Path, dst: Path, port: str) -> Path:
    """Part `src` with `port` committed afresh: gone from its reads file and from `pub.bin`'s registered rows."""
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("rec-bindings*"))  # `_own` writes them into `src` as it's copied
    r = json.loads((dst / "registered.json").read_text())
    del r["ports"][port]
    (dst / "registered.json").write_text(json.dumps(r))
    data = (dst / "pub.bin").read_bytes()
    nl = data.index(b"\n")
    head = json.loads(data[:nl])
    del head["registered"][port]
    (dst / "pub.bin").write_bytes(json.dumps(head).encode() + data[nl:])
    return dst


def _part(ex, d: Path, shape: Path, undrawn: Path, shared: Path | None = None) -> cf.Future:
    """`ex` running a verify of algebra part `d` at the inner transcript, under `undrawn`'s session, at shared rows `shared`."""
    return ex.submit(_verify, d, d / "pub.bin", d / "registered.json", shape, "inner-honest", "--rec-inner-transcript",
                     str(shape.parent / "transcript.json"), session=undrawn, rows=shared)


def test_a_parts_row_ports_are_the_verifiers_shared_rows(algebra, shape, undrawn, rows, tmp_path):
    """`OwnNamesRows` and `PartPortsOne`'s check (`RecVStar.checkRows`): the honest part that reads `acc0` passes setup at the
    verifier's shared rows (`--rec-rows`, the honest algebra's) and is refused without them. The honest part with `acc0`
    committed afresh (gone from its reads file and `pub.bin`'s registered rows) is refused by the missing row port.

    Each verify sets the part's statement up over its whole population before any V* check (about 9 minutes at one core and
    4 GB), and none can reuse another's: each runs the verifier on other files. So they run at once, the next test's beside
    these (with the salted staging, the five took 49 minutes one after the other)."""
    a = next(p for p in sorted(algebra.iterdir()) if "acc0" in json.loads((p / "registered.json").read_text())["ports"])
    unit = _unit(a)
    with cf.ThreadPoolExecutor(max_workers=3) as ex:
        honest, bare = _part(ex, a, shape, undrawn, rows), _part(ex, a, shape, undrawn)
        fresh = _part(ex, _unregistered(a, tmp_path / "fresh", "acc0"), shape, undrawn, rows)
    assert honest.result()["why"] == S2_R7, honest.result()["why"]
    assert bare.result()["why"] == (f"setup: {unit}: one of V*'s parts; pass the verifier's record of V*'s shared rows "
                                    "(--rec-rows F), which each of its row ports must be"), bare.result()["why"]
    assert re.fullmatch(rf"setup: {re.escape(unit)}: its row port acc0 is not a port of the registered reads \(\[.*\]\)",
                        fresh.result()["why"]), fresh.result()["why"]


def test_a_part_staged_under_another_salt_is_refused_at_the_verifiers_shared_rows(salted, algebra, shape, undrawn, rows,
                                                                                tmp_path):
    """`checkRows` at another registration: the algebra staged under another salt of the running sums' row 0 registers the
    same rows as another `rec-acc`, consistently: its part that reads `acc0` passes setup at its own staging's shared rows
    and is refused at the verifier's, by `acc0`'s root. Its two verifies run at once, as the previous test's do."""
    a = next(p for p in sorted(algebra.iterdir()) if "acc0" in json.loads((p / "registered.json").read_text())["ports"])
    unit, b = _unit(a), salted / "alg" / a.name
    own = _rec_rows(tmp_path / "own.json", sorted((salted / "alg").iterdir()))
    with cf.ThreadPoolExecutor(max_workers=2) as ex:
        salted_own, other = _part(ex, b, shape, undrawn, own), _part(ex, b, shape, undrawn, rows)
    assert salted_own.result()["why"] == S2_R7, salted_own.result()["why"]
    root = lambda f: json.loads(f.read_text())["ports"]["acc0"]["root"]
    assert root(b / "registered.json") != root(rows)
    assert other.result()["why"] == (f"setup: {unit}: its row port acc0 is not the verifier's shared row acc0 (--rec-rows): "
                                     f"root {root(b / 'registered.json')}, against {root(rows)}"), other.result()["why"]


def test_the_registered_reads_hold_no_message_round_port_the_statement_lacks(algebra, shape, undrawn, rows, tmp_path):
    """`RecVStar.checkOwn`: a part's reads file with one null read (a unit outside the population, which the registered
    reads' own check accepts) appended to one message-round port is refused by that port against the statement's, with the
    verifier's shared rows unedited; `checkRows` would refuse the same file by the shared row, so the exact refusal pins
    `checkOwn`. One verify sets the part's statement up over its whole population, about 9 minutes."""
    ports = lambda p: json.loads((p / "registered.json").read_text())["ports"]
    a = next(p for p in sorted(algebra.iterdir()) if any(re.fullmatch(r"m\d+", k) for k in ports(p)))
    d = shutil.copytree(a, tmp_path / "extra")
    r = json.loads((d / "registered.json").read_text())
    port = next(k for k in r["ports"] if re.fullmatch(r"m\d+", k))
    value, honest = r["ports"][port]["value"], r["ports"][port]["reads"]
    r["ports"][port]["reads"] = [*honest, None]
    (d / "registered.json").write_text(json.dumps(r))
    show = lambda rd: "[" + ", ".join("null" if x is None else str(x) for x in rd) + "]"
    v = _verify(d, d / "pub.bin", d / "registered.json", shape, "inner-honest", "--rec-inner-transcript",
                str(shape.parent / "transcript.json"), session=undrawn, rows=rows)
    assert v["why"] == (f"setup: {_unit(d)}: the registered reads' port {port} under {value} reading {show([*honest, None])} "
                        f"is not the statement's: the statement's is {port} under {value} reading {show(honest)}"), v["why"]


def test_a_sessions_rec_acc_reads_are_the_verifiers_one_registration(forged, shape, rows, tmp_path):
    """`EndKeysS` and `OpenKeysS`'s one registration (`RecVStar.checkAcc`): the recorded level 3 reads `rec-acc` as `acc` and
    `acc_out`, at the registration of the honest algebra's `acc0` and `acc1`, and passes setup at those shared rows
    (`--rec-rows`; S2/R7 refuses its outer session after) and is refused without them."""
    reads = lambda d: json.loads((d / "registered.json").read_text())["ports"]
    site = lambda e: {k: x for k, x in e.items() if k != "reads"}
    shared = json.loads(rows.read_text())["ports"]
    assert site(reads(SET / "L3")["acc"]) == site(reads(SET / "L3")["acc_out"]) == site(shared["acc0"]) == site(shared["acc1"])
    v = _verify(forged, SET / "L3" / "pub.bin", SET / "L3" / "registered.json", shape, "inner-honest", rows=rows)
    assert (v["accepted"], v["why"]) == (False, S2_R7), v
    v = _verify(forged, SET / "L3" / "pub.bin", SET / "L3" / "registered.json", shape, "inner-honest")
    assert v["why"] == (f"setup: {UNIT}: it reads rec-acc; pass the verifier's record of V*'s shared rows (--rec-rows F), whose "
                        "rec-acc registration it must be"), v["why"]


def test_a_value_the_shared_rows_register_is_read_at_that_registration(forged, shape, tmp_path):
    """`SharedOwnRoot`'s check (`RecVStar.checkShared`): a session's port reading a value the verifier's shared rows
    (`--rec-rows`) register must be that registration, but in reads. Shared rows that register level 3's `rec-coef` under
    another root refuse the recorded level by its `coef`, which passes setup without that entry
    (`test_the_honest_level_passes_setup_at_the_honest_inner_session`). R's tops, which need an entry, are
    `Flock.Test.SharedRows`'."""
    ports = json.loads((SET / "L3" / "registered.json").read_text())["ports"]
    coef = ports["coef"]
    other = {**coef, "root": coef["root"][:-1] + ("1" if coef["root"][-1] == "0" else "0")}
    rows = tmp_path / "rows.json"
    rows.write_text(json.dumps({"format": "verity/registered-reads/v0", "ports": {
        "acc0": ports["acc"], ports["out"]["value"]: ports["out"], "x-coef": other}}))
    v = _verify(forged, SET / "L3" / "pub.bin", SET / "L3" / "registered.json", shape, "inner-honest", rows=rows)
    assert v["why"] == (f"setup: {UNIT}: its port coef reads {coef['value']}, which the verifier's shared rows (--rec-rows) "
                        f"register as x-coef: root {coef['root']}, against {other['root']}"), v["why"]


def test_a_session_staged_under_another_salt_reads_rec_acc_as_another_registration(salted, forged, shape, rows, tmp_path):
    """`checkAcc` at another registration: level 3 staged under another salt of the running sums' row 0 reads the same rows
    as another `rec-acc`, consistently: it passes setup at its own registration and is refused at the verifier's by `acc`'s
    root. The recorded level with its `acc_out` read at that other registration (its reads file's entry and `pub.bin`'s
    paths; the rows it reads are the same, row 0 being no opening's `acc_out`) and `acc` at the honest one is refused as two
    registrations."""
    reads = lambda d: json.loads((d / "registered.json").read_text())["ports"]
    b = salted / f"L{LEVEL}"
    other, honest = reads(b)["acc"]["root"], json.loads(rows.read_text())["ports"]["acc0"]["root"]
    assert other != honest
    v = _verify(forged, b / "pub.bin", b / "registered.json", shape, "inner-honest", rows=_acc_rows(tmp_path / "own.json", b))
    assert v["why"] == S2_R7, v["why"]
    v = _verify(forged, b / "pub.bin", b / "registered.json", shape, "inner-honest", rows=rows)
    assert v["why"] == (f"setup: {UNIT}: its rec-acc port acc is not the verifier's rec-acc registration (--rec-rows acc0): "
                        f"root {other}, against {honest}"), v["why"]
    c = tmp_path / "mixed"
    c.mkdir()
    r = json.loads((SET / "L3" / "registered.json").read_text())
    r["ports"]["acc_out"] = reads(b)["acc_out"]
    (c / "registered.json").write_text(json.dumps(r))
    data, salted = (SET / "L3" / "pub.bin").read_bytes(), (b / "pub.bin").read_bytes()
    nl, snl = data.index(b"\n"), salted.index(b"\n")
    head = json.loads(data[:nl])
    head["registered"]["acc_out"] = json.loads(salted[:snl])["registered"]["acc_out"]
    (c / "pub.bin").write_bytes(json.dumps(head).encode() + data[nl:])
    v = _verify(forged, c / "pub.bin", c / "registered.json", shape, "inner-honest", rows=rows)
    assert v["why"] == (f"setup: {UNIT}: its rec-acc ports acc and acc_out are two registrations: root {honest}, against "
                        f"{other}"), v["why"]


def test_a_level_reading_rec_acc_off_its_openings_rows_of_the_chain_is_refused(level_major, forged, shape, tmp_path):
    """`OpenKeysS`' positions (`RecVStar.checkAccReads`): the recorded level 3's `acc` reads rows 395 to 447 for rep 0 and
    843 to 895 for rep 1 (the chain rep-major, then by level and query, at 448 openings a rep) and `acc_out` the next. On a
    level-major chain, consistently registered, `acc` reads row `790 + u` at unit `u`, which passes `checkAcc` at its own
    registration and is refused at unit 0."""
    reads = lambda d: json.loads((d / "registered.json").read_text())["ports"]
    honest = [395 + j for j in range(53)] + [843 + j for j in range(53)]
    assert reads(SET / "L3")["acc"]["reads"] == honest
    assert reads(SET / "L3")["acc_out"]["reads"] == [p + 1 for p in honest]
    assert reads(level_major)["acc"]["reads"] == [790 + u for u in range(106)]
    v = _verify(forged, level_major / "pub.bin", level_major / "registered.json", shape, "inner-honest",
                rows=_acc_rows(tmp_path / "own.json", level_major))
    assert v["why"] == (f"setup: {UNIT}: its port acc reads rec-acc at 790 in unit 0, not at 395 (the chain runs rep-major, "
                        "then by level and query)"), v["why"]


@pytest.mark.parametrize("other", ["verity/flock-circuit+seed-injection", "verity/flock-circuit@cec3e347"])
def test_the_honest_level_is_refused_as_another_statement_than_v_stars(forged, shape, tmp_path, other):
    """V*'s statements are `verity/flock-circuit`'s: the honest level set up as `other`, which parses the same file, is
    refused by that check alone, since `statement` derives it with every other flag under V*'s own; `verify` refuses it at
    setup. `verity/flock-circuit@cec3e347` is the statement the recorded outer session was proved as."""
    why = (f"setup: {UNIT}: one of V*'s statements, which are verity/flock-circuit statements (--statement "
           f"verity/flock-circuit), not {other}")
    rows = _acc_rows(tmp_path / "rows.json", SET / "L3")
    pub, reg = SET / "L3" / "pub.bin", SET / "L3" / "registered.json"
    p = keyed.run([str(BIN), "statement", *_files(forged, pub, reg, shape, rows)])
    assert p.returncode == 0 and json.loads(p.stdout)["statement"] == "verity/flock-circuit", (p.stdout, p.stderr)
    p = keyed.run([str(BIN), "statement", *_files(forged, pub, reg, shape, rows, other)])
    assert p.returncode == 1 and p.stdout.strip() == why, p.stdout
    v = _verify(forged, pub, reg, shape, "inner-honest", rows=rows, statement=other)
    assert v["why"] == why, v["why"]


def test_the_honest_level_is_refused_with_a_lookup_table_held(forged, shape, tmp_path):
    """V*'s statements read no lookup table, so under `--rec-inner` the verifier holds none: the honest level with a table
    under `--tables` (any bytes; the level reads no table, so nothing else reads it) is refused by that check alone, which
    `statement` shows, and `verify` refuses it at setup."""
    (tmp_path / "tables").mkdir()
    (tmp_path / "tables" / "rcp.u32").write_bytes(bytes(64))
    why = f"setup: {UNIT}: one of V*'s statements, which read no lookup table; the verifier holds [rcp] (--tables, --archive)"
    rows = _acc_rows(tmp_path / "rows.json", SET / "L3")
    pub, reg = SET / "L3" / "pub.bin", SET / "L3" / "registered.json"
    p = keyed.run([str(BIN), "statement", *_files(forged, pub, reg, shape, rows), "--tables", str(tmp_path / "tables")])
    assert p.returncode == 1 and p.stdout.strip() == why, p.stdout
    v = _verify(forged, pub, reg, shape, "inner-honest", "--tables", str(tmp_path / "tables"), rows=rows)
    assert v["why"] == why, v["why"]


@pytest.fixture(scope="module")
def bit_rows(built, once) -> Path:
    """Level 3 staged with every row a `sha512/row/v2` bit row of its words' 16 bits (`compose`'s `port_bits`, META
    `row_schemas`), which `ProvedScope.scopeOkH` admits under hidden outputs (`bitRows`) and V*'s `scopeOk` does not. Its
    rows read no registered value: V*'s registered rows are `sha512/row/v1` rows (`RecCoins.checkRows`), so the verifier
    holds none (no `--registered`), and `scopeOk` alone stands between it and V*."""
    from verity_flock import circuit as FC

    compose, write = FC.compose, FC.write

    def make(d):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(FC, "compose", lambda low, sub, **kw: compose(
                low, sub, port_bits=[16 * w for w, _ in low.leaves["in_ports"]], **kw))
            mp.setattr(FC, "write", lambda *a, registered=None, **kw: write(*a, **kw))
            S.stage(SET / "inner-honest", json.loads((SET / "fold-honest.json").read_text()), d, levels=[LEVEL],
                    algebra=False, pts=json.loads((SET / "points-honest.json").read_text()), salts=SET / "acc-salts.bin",
                    version=VERSION)

    out = once("rec-coins-bit-rows", make, _KEY) / f"L{LEVEL}"
    assert '"row_schemas":[["hm96-sha512/row/v2",' in (out / "circuit.txt").open().readlines()[1]
    return out


def test_a_v_star_file_of_bit_rows_is_refused_under_the_inner_shape(bit_rows, shape):
    """`VStmt.hscope` is `scopeOk` (every row `sha512/row/v1`), so under `--rec-inner` a V* statement whose rows are bit
    rows, which `ProvedScope.check` admits with hidden outputs (`scopeOkH`'s `bitRows`), is refused by `checkStmt` by name
    before its file is compared with V*'s: by `statement`, and by `verify` at setup. The honest level passes both
    (`test_the_honest_level_passes_setup_at_the_honest_inner_session`)."""
    why = (f"{UNIT}: one of V*'s statements, whose circuits are in ProvedScope.scopeOk (every row sha512/row/v1, VStmt.hscope), "
           "not scopeOkH's bit rows: a row that is not a sha512/row/v1 row of 16 · words bits in at least one chunk")
    files = _files(bit_rows, bit_rows / "pub.bin", bit_rows / "registered.json", shape)
    i = files.index("--registered")
    files = files[:i] + files[i + 2:]
    p = subprocess.run([str(BIN), "statement", *files], capture_output=True, text=True)
    assert p.returncode == 1 and p.stdout.strip() == f"setup: {why}", (p.stdout, p.stderr)
    u = SET / "u" / "stage"
    p = subprocess.run([str(BIN), "verify", "--zk", *files, "--rec-inner-circuit", str(u / "circuit.txt"), "--rec-inner-public",
                        str(u / "pub-64.bin"), "--rec-inner-partition", str(u / "partition.json"), *_form(shape.parent),
                        "--rec-inner-session", str(SET / "inner-honest" / "session"), "--session",
                        str(SET / "outer-L3" / "session")],
                       capture_output=True, text=True)
    (v,) = [json.loads(line[8:]) for line in p.stdout.splitlines() if line.startswith("VERDICT ")]
    assert (v["accepted"], v["why"]) == (False, f"setup: {why}"), v


def test_rec_inner_coins_is_refused(built, tmp_path):
    """`--rec-inner-coins` is refused before any file is read (`RecCoins.refuseCoins`), by `statement` and by `verify`, in
    place of the inner statement's files or beside them: check 5 reads the inner statement's own files, and a record of
    what it reads binds only the inner session's record, not the coins that session drew."""
    u = SET / "u" / "stage"
    (tmp_path / "empty").write_bytes(b"")
    (tmp_path / "s").mkdir()
    (tmp_path / "s" / "session.json").write_text("{}")
    session = ["--rec-inner-session", str(tmp_path / "s")]
    files = ["--rec-inner-circuit", str(u / "circuit.txt"), "--rec-inner-public", str(u / "pub-64.bin"),
             "--rec-inner-partition", str(u / "partition.json")]
    for cmd in ("statement", "verify"):
        for rec in (["--rec-inner-coins", str(tmp_path / "absent.json")], [*files, "--rec-inner-coins"]):
            p = subprocess.run([str(BIN), cmd, "--circuit", str(tmp_path / "empty"), "--public", str(tmp_path / "empty"),
                                *rec, *session, *(["--session", str(tmp_path / "s")] if cmd == "verify" else [])],
                               capture_output=True, text=True)
            assert p.returncode == 2 and p.stderr.startswith("--rec-inner-coins is refused: "), (cmd, rec, p.stderr)


#: the inner statement's unit range (META's `ranges`, the unit net's: 2 slots of 2^12 rows at 516) at k_log 22, so 16 inner
#: point coordinates and a block of 2^(16 − (12 − 6)) = 1024 slots: the one RecSparse shape here at ELL 12 and S_LOG 1
C3_SPARSE = (1, 12, 1, 0)


def _c3(tmp_path: Path, sparse: tuple | None, m: int = 27, circuit: Path | None = None, files: bool = True,
        verify: bool = False, claims: int = 8, form: list[str] | None = None) -> str:
    """`statement` (or `verify` on a session) at the inner statement's inner shape with the RecSparse shape ``sparse``, on an
    empty outer circuit, which candidate 3's inner checks precede: its refusal at setup. ``form`` stands in for the inner
    statement's setup flags (`_form`)."""
    u = SET / "u" / "stage"
    (tmp_path / "empty").write_bytes(b"")
    inner = tmp_path / "inner.json"
    inner.write_text(json.dumps({"format": "verity/rec-inner/v1", "m": m, "k_log": 22, "claims": claims, "regions": 3,
                                 **({"sparse": list(sparse)} if sparse else {})}))
    session = ["--rec-inner-session", str(SET / "inner-honest" / "session")]
    rec = ["--rec-inner-circuit", str(circuit or u / "circuit.txt"), "--rec-inner-public", str(u / "pub-64.bin"),
           "--rec-inner-partition", str(u / "partition.json"), *(_form(tmp_path) if form is None else form),
           *session] if files else []
    base = [str(BIN), "--circuit", str(tmp_path / "empty"), "--public", str(tmp_path / "empty"), "--rec-inner", str(inner),
            *rec]
    if verify:
        (tmp_path / "s").mkdir(exist_ok=True)
        (tmp_path / "s" / "session.json").write_text("{}")
        base.insert(1, "verify")
        p = subprocess.run([*base, "--session", str(tmp_path / "s")], capture_output=True, text=True)
        (v,) = [json.loads(line[8:]) for line in p.stdout.splitlines() if line.startswith("VERDICT ")]
        return v["why"]
    base.insert(1, "statement")
    p = subprocess.run(base, capture_output=True, text=True)
    assert p.returncode == 1, (p.stdout, p.stderr)
    return p.stdout.strip()


NOT_A_CIRCUIT = "setup: circuit: not a flock-circuit file"
UNIT_RANGE = "rec-inner: the inner circuit's unit range (slot_log 12, pos0 516, count 2) is not at most {} slots of 2^{} rows inside the block"


@pytest.mark.parametrize("m,sparse,why", [
    (27, C3_SPARSE, None),
    (26, C3_SPARSE, "rec-inner: the verifier's inner shape (m 26, k_log 22, regions 3) is not the inner statement's (m 27, "
     "k_log 22, regions 3)"),
    (27, (0, 12, 1, 0), "rec-inner: rec-sparse/n0-e12-s1-t0: rec-sparse's NU is at least 1, ELL at least 6 and T_LOG at most 16"),
    (27, (1, 23, 1, 0), "rec-inner: the inner points are short of rec-sparse/n1-e23-s1-t0's"),
    (27, (1, 12, 0, 0), UNIT_RANGE.format(1, 12)),
], ids=["honest", "shape", "invalid", "short", "slots"])
def test_candidate_3s_session_checks_the_sparse_range_on_the_inner_statement(built, tmp_path, m, sparse, why):
    """`RecCoins.checkAtC3`, first in every candidate-3 session's setup: the inner statement set up from its files is the
    inner shape's (`checkInnerStmt`) and passes check 5's sparse refusals at zero points (`checkInnerRange`). The honest
    shape is refused only at its (empty) outer circuit; each other is refused by name."""
    assert _c3(tmp_path, sparse, m) == (NOT_A_CIRCUIT if why is None else f"setup: {why}")


def test_candidate_3s_session_needs_the_inner_statements_files(built, tmp_path):
    """Candidate 3's session without the inner statement's files is refused at setup, by `statement` and by `verify`; an inner shape naming no RecSparse shape needs none. The unit range's
    own refusal (`RecCoins.unitRangeOf`) sits behind the inner statement's setup, which refuses a META without one unit
    range first; its Lean guards are `Test/InnerStreams.lean`'s."""
    need = ("setup: rec-inner: candidate 3's session needs the inner statement (--rec-inner-circuit, --rec-inner-public, "
            "--rec-inner-partition, --rec-inner-session): its sparse range is checked on it")
    assert _c3(tmp_path, C3_SPARSE, files=False) == need
    assert _c3(tmp_path, C3_SPARSE, files=False, verify=True) == need
    assert _c3(tmp_path, (0, 12, 1, 0), verify=True) == (
        "setup: rec-inner: rec-sparse/n0-e12-s1-t0: rec-sparse's NU is at least 1, ELL at least 6 and T_LOG at most 16")
    assert _c3(tmp_path, None, files=False) == NOT_A_CIRCUIT
    head, meta, rest = (SET / "u" / "stage" / "circuit.txt").read_bytes().split(b"\n", 2)
    m = json.loads(meta[5:])
    m["ranges"] = [r for r in m["ranges"] if r["net"] != "unit"]
    (tmp_path / "no-unit.txt").write_bytes(head + b"\nMETA " + json.dumps(m).encode() + b"\n" + rest)
    assert _c3(tmp_path, C3_SPARSE, circuit=tmp_path / "no-unit.txt") == (
        "setup: rec-coins: the inner session: the inner statement: not one unit range of vus_per_block x units_per_vu (a "
        "power of two) units")


def _off_files(m: int, claims: int) -> str:
    return (f"setup: rec-inner: the verifier's inner shape (m {m}, k_log 22, regions 3, claims {claims}) is not the inner "
            "statement's (m 27, k_log 22, regions 3, claims 8), which its files (--rec-inner-circuit, --rec-inner-public, "
            "--rec-inner-partition) set up")


def test_the_verifiers_inner_shape_is_the_inner_statements_where_it_holds_its_files(built, tmp_path):
    """`RecSetup.checkShape`, at every recursive session's setup that holds the inner statement's files, dense or not: the
    inner shape's `m`, `k_log`, regions and claims are the ones the files set the inner statement up at (two link points,
    so claims 8 over 3 regions), the shape link `SchedMatches` at the dense form reads. A dense shape off the files' `m`
    is refused, as is one whose claims are off, which candidate 3's own check on the statement (`checkInnerStmt`, the
    `m`, `k_log` and regions) passes; a dense shape without the files has nothing to compare."""
    assert _c3(tmp_path, None) == NOT_A_CIRCUIT
    assert _c3(tmp_path, None, m=26) == _off_files(26, 8)
    assert _c3(tmp_path, None, m=26, verify=True) == _off_files(26, 8)
    assert _c3(tmp_path, C3_SPARSE, claims=10) == _off_files(27, 10)
    assert _c3(tmp_path, None, m=26, files=False) == NOT_A_CIRCUIT


FORM = "verity/flock-circuit+plain-leaves"


def test_the_inner_statement_is_set_up_as_its_files_set_it_up(built, tmp_path):
    """`RecSetup.checkForm`, last in `RecSetup.checkShape`: how Flock's plain verifier is told to set the inner statement up
    is how its files set it up (`RecCoins.Files.stmt`), so its `m` and regions are the files' (`innerSetup_files`): a
    `verity/flock-circuit+plain-leaves` statement on the coin server's coins with no lookup table, at a draw of every one of
    the files' units in order. The honest form is refused only at the (empty) outer circuit; each other by name."""
    (tmp_path / "tables").mkdir()
    (tmp_path / "tables" / "rsq.u32").write_bytes(b"")
    (tmp_path / "not-json").write_text("{")
    one = tmp_path / "one"
    one.mkdir()
    honest = _form(tmp_path)
    draw = honest[-1]

    def why(form: list[str]) -> str:
        return _c3(tmp_path, None, form=form)

    assert why(honest) == NOT_A_CIRCUIT
    assert why([]) == ("setup: rec-inner: the verifier holds no setup of the inner statement (--rec-inner-statement, "
                       "--rec-inner-statement-coins, --rec-inner-draw) to hold to its files' setup")
    assert why(["--rec-inner-statement", "verity/flock-circuit", *honest[2:]]) == (
        f"setup: rec-inner: the inner statement is a verity/flock-circuit statement, not {FORM}, as its files set it up")
    assert why(["--rec-inner-statement", FORM, "--rec-inner-draw", draw]) == (
        "setup: rec-inner: the inner statement's coins are not the coin server's (--rec-inner-statement-coins os), as its "
        "files set it up")
    assert why([*honest, "--rec-inner-tables", str(tmp_path / "tables")]) == (
        "setup: rec-inner: the inner statement names lookup tables (--rec-inner-tables), and its files set it up with none")
    assert why(honest[:4]) == "setup: rec-inner: the inner statement has no draw (--rec-inner-draw)"
    assert why([*honest[:4], "--rec-inner-draw", str(tmp_path / "not-json")]).startswith(
        "setup: rec-inner: the inner statement's draw: not JSON: ")
    assert why(_form(one, {"law": "subset", "population": 64, "k": 1, "units": [0]})) == (
        "setup: rec-inner: the inner statement's draw proves 1 units, not every one of its files' 64 in order")


def test_the_inner_statements_setup_flags_go_with_its_statement(built, tmp_path):
    """`RecCoins.formInput`: the inner statement's coins, tables and draw go with `--rec-inner-statement`, and its coins
    are `os` or `seed`."""
    u = SET / "u" / "stage"
    (tmp_path / "empty").write_bytes(b"")
    base = [str(BIN), "statement", "--circuit", str(tmp_path / "empty"), "--public", str(tmp_path / "empty"),
            "--rec-inner-circuit", str(u / "circuit.txt"), "--rec-inner-public", str(u / "pub-64.bin"),
            "--rec-inner-partition", str(u / "partition.json"), "--rec-inner-session", str(SET / "inner-honest" / "session")]
    p = subprocess.run([*base, "--rec-inner-draw", str(tmp_path / "empty")], capture_output=True, text=True)
    assert "--rec-inner-statement-coins, --rec-inner-tables and --rec-inner-draw go with --rec-inner-statement" in (
        p.stdout + p.stderr), (p.returncode, p.stdout, p.stderr)
    p = subprocess.run([*base, "--rec-inner-statement", FORM, "--rec-inner-statement-coins", "keyed"], capture_output=True,
                       text=True)
    assert "--rec-inner-statement-coins keyed: not os or seed" in p.stdout + p.stderr, (p.returncode, p.stdout, p.stderr)
