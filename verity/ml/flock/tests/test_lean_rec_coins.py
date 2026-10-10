"""V*'s verifier-registered values against the verifier's own coins (`Flock/RecCoins.lean`, `verify --rec-inner-*`): the
smallest level of a recorded recursive session (`r20261009-111638-6720`, class (16,16,8) at 64 instances: level 3,
`rec-open/l16-h4`, 106 openings), restaged by `rec_vstage.stage` from the fetched fixture `rec-coins/16-16-8-n64`: the inner
statement, the inner session's public record and the prover's, V*'s fold, points and running-sum salts, the level's
recorded `pub.bin` and `registered.json`, and its recorded outer session. The level and its outer session were proved
again at the current leaf scheme (`r20261010-040650-84a9`: the inner statement's leaves are unsalted, so the rest of the
session is 6720's). The honest level is accepted at the honest inner
session and refused without it, and a level whose `rec-coef` V* registered one bit off is refused at that opening, though
its registration is consistent. Staging the level takes about a minute and each verify sets up its 158 MB circuit, so the
module is slow; it catches what no test before it can: the Lean verifier's coefficients, positions and salts at real
coins, against the ones V*'s Python registers."""
from __future__ import annotations

import functools
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from verity_flock import rec_bindings as RB
from verity_flock import rec_vstage as S

HERE = Path(__file__).resolve().parent
LEAN = HERE.parents[2] / "core" / "service" / "flock"
BIN = LEAN / ".lake" / "build" / "bin" / "flock-verify"
SET = HERE / "fixtures" / "rec-coins" / "16-16-8-n64"
LEVEL = 3
UNIT = "rec-open/l16-h4"
STATEMENT = "verity/flock-circuit"
#: the recorded level's circuit (`v/L3/circuit.txt`), which its outer session proves
CIRCUIT_SHA512 = ("885c04a867b23838a460033b3cfcbd8cad28bc5f994dd10e907724d293479cf7a525e331ed7152ab5e451bb27f39905f695af6295408db"
                  "459a42e2dda3e07463")

pytestmark = [pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain"), pytest.mark.slow]


@pytest.fixture(scope="module")
def forged(tmp_path_factory) -> Path:
    """Level 3 staged with rep 0's opening 0 `coef` one bit off (bit 0 of its first word) where V* computes it, so its
    registration, the running sums and the prover's row all agree with it. The circuit, program and partition do not read
    the coefficients: they are the recorded level's."""
    assert (SET / "u" / "stage" / "circuit.txt").is_file(), (
        f"{SET} is not fetched: `uv run research data fetch-fixtures verity/ml/flock/tests/fixtures/rec-coins`")
    subprocess.run(["lake", "build", "flock-verify"], cwd=LEAN, check=True, capture_output=True)
    honest = S.coefficients

    def coefficients(vf, qs):
        cs = honest(vf, qs)
        c = cs[(0, LEVEL, 0)]
        cs[(0, LEVEL, 0)] = [c[0] ^ 1, *c[1:]]
        return cs

    d = tmp_path_factory.mktemp("rec-coins")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(S, "coefficients", coefficients)
        res = S.stage(SET / "inner-honest", json.loads((SET / "fold-honest.json").read_text()), d, levels=[LEVEL],
                      algebra=False, pts=json.loads((SET / "points-honest.json").read_text()), salts=SET / "acc-salts.bin")
    out = d / f"L{LEVEL}"
    assert res["levels"][f"L{LEVEL}"]["opened"] == res["levels"][f"L{LEVEL}"]["instances"] == 106
    assert hashlib.sha512((out / "circuit.txt").read_bytes()).hexdigest() == CIRCUIT_SHA512
    return out


@pytest.fixture(scope="module")
def shape(forged, tmp_path_factory) -> Path:
    """The verifier's own inner shape (`rec-inner`) from the inner statement, which `verify` needs for any V* statement."""
    u = SET / "u" / "stage"
    p = subprocess.run([str(BIN), "rec-inner", "--circuit", str(u / "circuit.txt"), "--public", str(u / "pub-64.bin"),
                        "--partition", str(u / "partition.json")], capture_output=True, text=True, check=True)
    f = tmp_path_factory.mktemp("rec-inner") / "inner.json"
    f.write_text(p.stdout)
    return f


@functools.cache
def _statements(shape: Path) -> tuple[str, ...]:
    """V*'s statements' unit names at the inner shape, in `rec-statements`' order, the order `--rec-part` indexes."""
    p = subprocess.run([str(BIN), "rec-statements", "--rec-inner", str(shape)], capture_output=True, text=True, check=True)
    [line] = [json.loads(ln[9:]) for ln in p.stdout.splitlines() if ln.startswith("RECVSTAR ")]
    return tuple(s[0] for s in line["statements"])


def _own(level: Path, shape: Path) -> list[str]:
    """The verifier's own `--rec-part` and `--rec-bindings` for the statement staged in `level`: its unit's index among V*'s
    statements, and `rec_bindings` of its partition object and program, written beside it."""
    with (level / "circuit.txt").open() as f:
        unit = f.readline().split()[1]
    b = level / "rec-bindings.json"
    b.write_text(json.dumps(RB.bindings(json.loads((level / "partition.json").read_text()),
                                        json.loads((level / "program.json").read_text()))))
    return ["--rec-part", str(_statements(shape).index(unit)), "--rec-bindings", str(b)]


def _run(level: Path, public: Path, registered: Path, shape: Path, inner: str | None, *extra: str,
         session: Path = SET / "outer-L3" / "session") -> subprocess.CompletedProcess:
    u = SET / "u" / "stage"
    rec = [] if inner is None else ["--rec-inner-circuit", str(u / "circuit.txt"), "--rec-inner-public", str(u / "pub-64.bin"),
                                    "--rec-inner-partition", str(u / "partition.json"), "--rec-inner-session",
                                    str(SET / inner / "session")]
    return subprocess.run([str(BIN), "verify", "--zk", "--statement", STATEMENT, "--circuit",
                           str(level / "circuit.txt"), "--public", str(public), "--partition", str(level / "partition.json"),
                           "--program", str(level / "program.json"), "--registered", str(registered), "--rec-inner", str(shape),
                           *_own(level, shape), *rec, *extra, "--session", str(session)],
                          capture_output=True, text=True, check=False)


def _verify(level: Path, public: Path, registered: Path, shape: Path, inner: str | None, *extra: str,
            session: Path = SET / "outer-L3" / "session") -> dict:
    p = _run(level, public, registered, shape, inner, *extra, session=session)
    (v,) = [json.loads(line[8:]) for line in p.stdout.splitlines() if line.startswith("VERDICT ")]
    return v


def test_the_honest_level_is_accepted_at_the_honest_inner_session(forged, shape):
    v = _verify(forged, SET / "L3" / "pub.bin", SET / "L3" / "registered.json", shape, "inner-honest")
    assert v["accepted"], v.get("why")


def test_the_honest_level_is_refused_without_the_inner_session(forged, shape):
    v = _verify(forged, SET / "L3" / "pub.bin", SET / "L3" / "registered.json", shape, None)
    assert v["why"].startswith(f"setup: rec-coins: {UNIT}: the verifier holds no inner session"), v["why"]


def test_a_coef_registered_one_bit_off_is_refused_at_its_opening(forged, shape):
    v = _verify(forged, forged / "pub.bin", forged / "registered.json", shape, "inner-honest")
    assert v["why"] == (f"setup: rec-coins: {UNIT}: level {LEVEL} rep 0 opening 0: the statement's coef row is not the "
                        "verifier's rec-coef row 0")


def _stage(d: Path, **kw) -> dict:
    return S.stage(SET / "inner-honest", json.loads((SET / "fold-honest.json").read_text()), d,
                   pts=json.loads((SET / "points-honest.json").read_text()), salts=SET / "acc-salts.bin", **kw)


def test_a_dirs_registered_one_bit_off_is_refused_at_its_opening(forged, shape, tmp_path):
    """`HDirCols`' check (`RecCoins.checkOpen`'s `dirs`): level 3 staged with rep 0's opening 0 `dirs` registered one bit
    off, the opening's climb aside, is refused at that opening by the position the verifier's coins give."""
    honest = S.R.dirs_words
    first = []

    def dirs_words(pos, H):
        w = honest(pos, H)
        if not first:
            first.append(pos)
            w = [w[0] ^ 1, *w[1:]]
        return w

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(S.R, "dirs_words", dirs_words)
        _stage(tmp_path, levels=[LEVEL], algebra=False)
    out = tmp_path / f"L{LEVEL}"
    v = _verify(out, out / "pub.bin", out / "registered.json", shape, "inner-honest")
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
def algebra(forged, tmp_path_factory) -> Path:
    """V*'s algebra parts (`p<j>`) staged over the honest inner session."""
    d = tmp_path_factory.mktemp("rec-alg")
    _stage(d, levels=[], algebra=True, evaluate=False)
    return d / "alg"


def test_a_v_registered_one_element_off_is_refused_at_its_part(algebra, shape, undrawn, tmp_path):
    """`HVCoins`' check (`RecCoins.checkResiduals`): the algebra staged with rep 0's first comb element of v one bit off
    (`rec_vstage --forge comb`) changes one part's registration, and that part is refused by the v the verifier's coins and
    fold give."""
    _stage(tmp_path, levels=[], algebra=True, evaluate=False, forge="comb")
    parts = sorted((tmp_path / "forged-comb" / "alg").iterdir())
    changed = [p for p in parts if (p / "registered.json").read_bytes() != (algebra / p.name / "registered.json").read_bytes()]
    assert len(changed) == 1 and len(parts) > 1, changed
    (p,) = changed
    t = SET / "inner-honest" / "public.json"
    why = _verify(p, p / "pub.bin", p / "registered.json", shape, "inner-honest", "--rec-inner-transcript", str(t),
                  session=undrawn)["why"]
    assert re.fullmatch(rf"setup: rec-coins: rec-residuals/m\d+-k\d+-c\d+-{p.name}: rep 0: part {p.name[1:]}'s \d+ elements of v: "
                        "the statement's v row is not the verifier's rec-v row 0", why), why


def _transcript(dst: Path, change) -> Path:
    t = json.loads((SET / "inner-honest" / "public.json").read_text())
    change(next(r for r in sorted(t["rounds"], key=lambda r: r["k"]) if r["stream"].endswith("/rep0") and r.get("commits")))
    dst.write_text(json.dumps(t))
    return dst


def test_a_parts_message_rows_are_the_c_k_of_the_inner_transcript(algebra, shape, undrawn, tmp_path):
    """`HPubRows`' check (`RecVStar.checkPubRows`): the `pub.bin` rows of the first algebra part that reads `rec-c0` are,
    at its `rec-c<k>` reads, the `c_k` the coin server recorded. The honest part passes setup with the inner transcript
    (`undrawn`'s session is refused after it), is refused without it, and is refused at a transcript whose rep 0 `c_0` is
    one bit off, or whose coins there are not the inner session's."""
    reads = lambda p: {k: x["value"] for k, x in json.loads((p / "registered.json").read_text())["ports"].items()}
    a = next(p for p in sorted(algebra.iterdir()) if "rec-c0" in reads(p).values())
    (port,) = [k for k, x in reads(a).items() if x == "rec-c0"]
    with (a / "circuit.txt").open() as f:
        unit = f.readline().split()[1]
    t = SET / "inner-honest" / "public.json"
    v = _verify(a, a / "pub.bin", a / "registered.json", shape, "inner-honest", "--rec-inner-transcript", str(t), session=undrawn)
    assert not v["why"].startswith("setup:"), v["why"]
    v = _verify(a, a / "pub.bin", a / "registered.json", shape, "inner-honest", session=undrawn)
    assert v["why"].startswith(f"setup: rec-pub: {unit}: the verifier holds no inner transcript"), v["why"]

    def flip(r):
        c = bytearray.fromhex(r["commits"][0])
        c[0] ^= 1
        r["commits"] = [c.hex()]

    bad = _transcript(tmp_path / "c0.json", flip)
    v = _verify(a, a / "pub.bin", a / "registered.json", shape, "inner-honest", "--rec-inner-transcript", str(bad), session=undrawn)
    assert re.fullmatch(rf"setup: rec-pub: {re.escape(unit)}: port {port}'s table row \d+ is not rep 0's c_0 \(inner round \d+\) "
                        "of the inner transcript", v["why"]), v["why"]
    bad = _transcript(tmp_path / "coins.json", lambda r: r.__setitem__("coins", ["00" * 16] * len(r["coins"])))
    p = _run(a, a / "pub.bin", a / "registered.json", shape, "inner-honest", "--rec-inner-transcript", str(bad), session=undrawn)
    assert p.returncode == 2 and "round" in p.stderr and "coins are not the inner session record's" in p.stderr, p.stderr
