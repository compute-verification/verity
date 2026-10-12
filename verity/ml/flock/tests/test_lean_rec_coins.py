"""V*'s verifier-registered values against the verifier's own coins (`Flock/RecCoins.lean`, `verify --rec-inner-*`): the
smallest level of a recorded recursive session (`r20261009-111638-6720`, class (16,16,8) at 64 instances: level 3,
`rec-open/l16-h4`, 106 openings), restaged by `rec_vstage.stage` from the fetched fixture `rec-coins/16-16-8-n64`: the inner
statement, the inner session's public record and the prover's, V*'s fold, points and running-sum salts, the level's
recorded `pub.bin` and `registered.json`, and its recorded outer session. The level and its outer session were proved
again at the current leaf scheme (`r20261010-040650-84a9`: the inner statement's leaves are unsalted, so the rest of the
session is 6720's), before #1677's OS masks, as `verity/flock-circuit@cec3e347`. `verify` sets V*'s statements up only
as `verity/flock-circuit`, so the recorded outer session is refused at the record's first check (S2/R7) once every V*
check has passed at setup. The honest level passes setup at the honest inner session and is refused without it, and a
level whose `rec-coef` V* registered one bit off is refused at that opening, though
its registration is consistent. Set up as another statement than V*'s `verity/flock-circuit`, or with a lookup table held,
the honest level is refused. Every verify names the session's statement (`--rec-part J`, its index in `rec-statements`'
list) and passes the verifier's own bindings (`--rec-bindings F`, `rec_bindings` of its partition object and program).
Staging the level takes about a minute and each verify sets up its 158 MB circuit, so the
module is slow; it catches what no test before it can: the Lean verifier's coefficients, positions and salts at real
coins, against the ones V*'s Python registers. The algebra's parts are staged too, for their verifier-registered `v`,
their message rows and their row ports, and every session's `rec-acc` reads, against the verifier's record of V*'s shared
rows (`--rec-rows`); and check 5 on `rec-coins`' record of the inner session in place of its statement (`--rec-inner-coins`)."""
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

import pytest
from verity_flock import rec_bindings as RB
from verity_flock import rec_vstage as S

HERE = Path(__file__).resolve().parent
LEAN = HERE.parents[2] / "core" / "service" / "flock"
BIN = LEAN / ".lake" / "build" / "bin" / "flock-verify"
SET = HERE / "fixtures" / "rec-coins" / "16-16-8-n64"
LEVEL = 3
UNIT = "rec-open/l16-h4"
#: the recorded level's circuit (`v/L3/circuit.txt`), which its outer session proves
CIRCUIT_SHA512 = ("885c04a867b23838a460033b3cfcbd8cad28bc5f994dd10e907724d293479cf7a525e331ed7152ab5e451bb27f39905f695af6295408db"
                  "459a42e2dda3e07463")
#: the recorded outer session's refusal as `verity/flock-circuit`, after setup
S2_R7 = "S2/R7: the session parameters are not the verifier's"

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
                          algebra=False, pts=json.loads((SET / "points-honest.json").read_text()), salts=SET / "acc-salts.bin")
        assert res["levels"][f"L{LEVEL}"]["opened"] == res["levels"][f"L{LEVEL}"]["instances"] == 106

    out = once("rec-coins-forged", make) / f"L{LEVEL}"
    assert hashlib.sha512((out / "circuit.txt").read_bytes()).hexdigest() == CIRCUIT_SHA512
    return out


@pytest.fixture(scope="module")
def shape(built, tmp_path_factory) -> Path:
    """The verifier's own inner shape (`rec-inner`) from the inner statement, which `verify` needs for any V* statement."""
    u = SET / "u" / "stage"
    p = subprocess.run([str(BIN), "rec-inner", "--circuit", str(u / "circuit.txt"), "--public", str(u / "pub-64.bin"),
                        "--partition", str(u / "partition.json")], capture_output=True, text=True, check=True)
    f = tmp_path_factory.mktemp("rec-inner") / "inner.json"
    f.write_text(p.stdout)
    return f


@functools.cache
def _statements(shape: str) -> list[str]:
    p = subprocess.run([str(BIN), "rec-statements", "--rec-inner", shape], capture_output=True, text=True, check=True)
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
         statement: str = "verity/flock-circuit", coins: Path | None = None) -> subprocess.CompletedProcess:
    u = SET / "u" / "stage"
    files = ["--rec-inner-circuit", str(u / "circuit.txt"), "--rec-inner-public", str(u / "pub-64.bin"),
             "--rec-inner-partition", str(u / "partition.json")]
    rec = [] if inner is None else [*(files if coins is None else ["--rec-inner-coins", str(coins)]),
                                    "--rec-inner-session", str(SET / inner / "session")]
    p = subprocess.run([str(BIN), "verify", "--zk", *_files(level, public, registered, shape, rows, statement), *rec, *extra,
                        "--session", str(session)], capture_output=True, text=True, check=False)
    return p


@pytest.fixture(scope="module")
def coins(built, once) -> Path:
    """`rec-coins`' record of what check 5 reads of the honest inner session, set up once from the inner statement."""
    u = SET / "u" / "stage"

    def make(d):
        p = subprocess.run([str(BIN), "rec-coins", "--rec-inner-circuit", str(u / "circuit.txt"), "--rec-inner-public",
                            str(u / "pub-64.bin"), "--rec-inner-partition", str(u / "partition.json"), "--rec-inner-session",
                            str(SET / "inner-honest" / "session")], capture_output=True, text=True, check=True)
        (d / "coins.json").write_text(p.stdout)

    return once("rec-coins-coins", make) / "coins.json"


def _verify(level: Path, public: Path, registered: Path, shape: Path, inner: str | Path | None, *extra: str,
            session: Path = SET / "outer-L3" / "session", rows: Path | None = None,
            statement: str = "verity/flock-circuit", coins: Path | None = None) -> dict:
    p = _run(level, public, registered, shape, inner, *extra, session=session, rows=rows, statement=statement, coins=coins)
    (v,) = [json.loads(line[8:]) for line in p.stdout.splitlines() if line.startswith("VERDICT ")]
    return v


def _acc_rows(dst: Path, level: Path) -> Path:
    """Shared rows (`--rec-rows`) whose `acc0` is staged level `level`'s `rec-acc` registration (its `acc` port)."""
    acc = json.loads((level / "registered.json").read_text())["ports"]["acc"]
    dst.write_text(json.dumps({"format": "verity/registered-reads/v0", "ports": {"acc0": acc}}))
    return dst


def test_the_honest_level_passes_setup_at_the_honest_inner_session(forged, shape, tmp_path):
    v = _verify(forged, SET / "L3" / "pub.bin", SET / "L3" / "registered.json", shape, "inner-honest",
                rows=_acc_rows(tmp_path / "rows.json", SET / "L3"))
    assert (v["accepted"], v["why"]) == (False, S2_R7), v


def test_the_honest_level_is_refused_without_the_inner_session(forged, shape, tmp_path):
    v = _verify(forged, SET / "L3" / "pub.bin", SET / "L3" / "registered.json", shape, None,
                rows=_acc_rows(tmp_path / "rows.json", SET / "L3"))
    assert v["why"].startswith(f"setup: rec-coins: {UNIT}: the verifier holds no inner session"), v["why"]


def test_an_inner_record_that_keeps_a_unit_draw_is_refused(forged, shape, tmp_path):
    """`Record.decode`'s U1 (`Draw.check`): the inner statement never carries a draw (`Inner.load` refuses one in its
    header), so an honest inner record given a `unit_draw` is refused by U1 alone; nothing else reads the record's draw."""
    inner = shutil.copytree(SET / "inner-honest", tmp_path / "inner-drawn")
    s = json.loads((inner / "session" / "session.json").read_text())
    s["unit_draw"] = {"law": "subset", "units": [0]}
    (inner / "session" / "session.json").write_text(json.dumps(s))
    v = _verify(forged, SET / "L3" / "pub.bin", SET / "L3" / "registered.json", shape, inner,
                rows=_acc_rows(tmp_path / "rows.json", SET / "L3"))
    assert v["why"] == (f"setup: rec-coins: {UNIT}: the inner session's record: U1: the record keeps a unit draw the "
                        "statement does not carry"), v["why"]


def test_a_coef_registered_one_bit_off_is_refused_at_its_opening(forged, shape, tmp_path):
    v = _verify(forged, forged / "pub.bin", forged / "registered.json", shape, "inner-honest",
                rows=_acc_rows(tmp_path / "rows.json", forged))
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
    return once("rec-coins-algebra", lambda d: _stage(d, levels=[], algebra=True, evaluate=False)) / "alg"


@pytest.fixture(scope="module")
def comb(built, once) -> Path:
    """The algebra staged with rep 0's first comb element of v one bit off (`rec_vstage --forge comb`)."""
    return once("rec-coins-comb", lambda d: _stage(d, levels=[], algebra=True, evaluate=False, forge="comb")) / (
        "forged-comb/alg")


@pytest.fixture(scope="module")
def salted(built, once) -> Path:
    """Level 3 and the algebra staged under another salt of the running sums' row 0 (`_salted`)."""
    def make(d):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(S, "acc_salts", _salted)
            _stage(d, levels=[LEVEL], algebra=True, evaluate=False)

    return once("rec-coins-salted", make)


def _rec_rows(dst: Path, parts: list[Path]) -> Path:
    """The verifier's record of V*'s shared rows (`--rec-rows`) from staged parts: each row port's entry (every port but `v`
    and `out`), which every part that has the port must agree on."""
    ports: dict = {}
    for d in parts:
        for nm, e in json.loads((d / "registered.json").read_text())["ports"].items():
            if nm not in ("v", "out"):
                assert ports.setdefault(nm, e) == e, (d, nm)
    dst.write_text(json.dumps({"format": "verity/registered-reads/v0", "ports": ports}))
    return dst


@pytest.fixture(scope="module")
def rows(algebra, tmp_path_factory) -> Path:
    """The verifier's shared rows of the honest algebra: its message rounds `m<k>` and the running sums' `acc0`, `acc1`."""
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
    t = SET / "inner-honest" / "public.json"
    why = _verify(p, p / "pub.bin", p / "registered.json", shape, "inner-honest", "--rec-inner-transcript", str(t),
                  session=undrawn, rows=rows)["why"]
    assert re.fullmatch(rf"setup: rec-coins: rec-residuals/m\d+-k\d+-c\d+-{p.name}: rep 0: part {p.name[1:]}'s \d+ elements of v: "
                        "the statement's v row is not the verifier's rec-v row 0", why), why


def _transcript(dst: Path, change) -> Path:
    t = json.loads((SET / "inner-honest" / "public.json").read_text())
    change(next(r for r in sorted(t["rounds"], key=lambda r: r["k"]) if r["stream"].endswith("/rep0") and r.get("commits")))
    dst.write_text(json.dumps(t))
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
    t = SET / "inner-honest" / "public.json"

    def flip(r):
        c = bytearray.fromhex(r["commits"][0])
        c[0] ^= 1
        r["commits"] = [c.hex()]

    bad = _transcript(tmp_path / "c0.json", flip)
    with cf.ThreadPoolExecutor(max_workers=3) as ex:  # each verify sets the part up for minutes; none reuses another's
        honest, bare, c0 = [ex.submit(_verify, a, a / "pub.bin", a / "registered.json", shape, "inner-honest", *x,
                                      session=undrawn, rows=rows)
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


def _part(ex, d: Path, shape: Path, undrawn: Path, shared: Path | None = None, coins: Path | None = None) -> cf.Future:
    """`ex` running a verify of algebra part `d` at the inner transcript, under `undrawn`'s session, at shared rows `shared`."""
    return ex.submit(_verify, d, d / "pub.bin", d / "registered.json", shape, "inner-honest", "--rec-inner-transcript",
                     str(SET / "inner-honest" / "public.json"), session=undrawn, rows=shared, coins=coins)


def test_a_parts_row_ports_are_the_verifiers_shared_rows(algebra, shape, undrawn, rows, tmp_path):
    """`OwnNamesRows` and `PartPortsOne`'s check (`RecVStar.checkRows`): the honest part that reads `acc0` passes setup at the
    verifier's shared rows (`--rec-rows`, the honest algebra's) and is refused without them. The honest part with `acc0`
    committed afresh (gone from its reads file and `pub.bin`'s registered rows) is refused by the missing row port.

    Each verify sets the part's statement up over its whole population before any V* check (about 9 minutes at one core and
    4 GB), and none can reuse another's: each runs the verifier on other files. So they run at once, the next test's beside
    these (with the salted staging, the five took 49 minutes one after the other, `r20261010-121137-2b46`)."""
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


@pytest.mark.parametrize("other", ["verity/flock-circuit+seed-injection", "verity/flock-circuit@cec3e347"])
def test_the_honest_level_is_refused_as_another_statement_than_v_stars(forged, shape, tmp_path, other):
    """V*'s statements are `verity/flock-circuit`'s: the honest level set up as `other`, which parses the same file, is
    refused by that check alone, since `statement` derives it with every other flag under V*'s own; `verify` refuses it at
    setup. `verity/flock-circuit@cec3e347` is the statement the recorded outer session was proved as."""
    why = (f"setup: {UNIT}: one of V*'s statements, which are verity/flock-circuit statements (--statement "
           f"verity/flock-circuit), not {other}")
    rows = _acc_rows(tmp_path / "rows.json", SET / "L3")
    pub, reg = SET / "L3" / "pub.bin", SET / "L3" / "registered.json"
    p = subprocess.run([str(BIN), "statement", *_files(forged, pub, reg, shape, rows)], capture_output=True, text=True)
    assert p.returncode == 0 and json.loads(p.stdout)["statement"] == "verity/flock-circuit", (p.stdout, p.stderr)
    p = subprocess.run([str(BIN), "statement", *_files(forged, pub, reg, shape, rows, other)], capture_output=True, text=True)
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
    p = subprocess.run([str(BIN), "statement", *_files(forged, pub, reg, shape, rows), "--tables", str(tmp_path / "tables")],
                       capture_output=True, text=True)
    assert p.returncode == 1 and p.stdout.strip() == why, p.stdout
    v = _verify(forged, pub, reg, shape, "inner-honest", "--tables", str(tmp_path / "tables"), rows=rows)
    assert v["why"] == why, v["why"]


def test_check_5_gives_the_part_holding_vs_comb_one_verdict_with_the_record_or_without(algebra, shape, undrawn, rows, coins):
    """Check 5's setup once per inner session at a part: the part holding v's comb gets the same verdict on `rec-coins`' record
    as on the inner statement's files. Each verify sets the part up over its whole population (minutes), so the two run at
    once; no level reads v."""
    p = sorted(algebra.iterdir())[-1]
    with cf.ThreadPoolExecutor(max_workers=2) as ex:
        both = [f.result() for f in [_part(ex, p, shape, undrawn, rows, c) for c in (None, coins)]]
    assert both[0] == both[1] and both[0]["why"] == S2_R7, both


def test_check_5_reads_the_verifiers_coins_record_as_the_inner_statement(forged, shape, coins, tmp_path):
    """Check 5's setup once per inner session (`rec-coins`, `--rec-inner-coins`): the honest level passes setup on the record,
    and the coef registered one bit off is refused with the message the inner statement's files give. The record is refused
    against another inner session's record, and in place of nothing: beside the inner statement's files."""
    acc = _acc_rows(tmp_path / "rows.json", SET / "L3")
    v = _verify(forged, SET / "L3" / "pub.bin", SET / "L3" / "registered.json", shape, "inner-honest", rows=acc, coins=coins)
    assert (v["accepted"], v["why"]) == (False, S2_R7), v
    v = _verify(forged, forged / "pub.bin", forged / "registered.json", shape, "inner-honest",
                rows=_acc_rows(tmp_path / "forged-rows.json", forged), coins=coins)
    assert v["why"] == (f"setup: rec-coins: {UNIT}: level {LEVEL} rep 0 opening 0: the statement's coef row is not the "
                        "verifier's rec-coef row 0"), v["why"]
    other = tmp_path / "other"
    shutil.copytree(SET / "inner-honest" / "session", other / "session")
    (other / "session" / "session.json").write_text((other / "session" / "session.json").read_text() + "\n")
    v = _verify(forged, SET / "L3" / "pub.bin", SET / "L3" / "registered.json", shape, other, rows=acc, coins=coins)
    assert v["why"] == "setup: rec inner coins: made from another inner session record than --rec-inner-session's", v["why"]
    u = SET / "u" / "stage"
    p = _run(forged, SET / "L3" / "pub.bin", SET / "L3" / "registered.json", shape, "inner-honest", "--rec-inner-circuit",
             str(u / "circuit.txt"), rows=acc, coins=coins)
    assert p.returncode == 2 and "--rec-inner-coins stands in for" in p.stderr, p.stderr
