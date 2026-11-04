"""Pearl+'s run verdict in Lean (`pouw-pearlplus`, `PouwCheck.PearlPlus.check`) is the Python's (`verify_run.verify_passes`).

Each case proves passes on the CPU as verify_run's tests do (`test_pearl_c_vllm.proved_pass`: the H100 record's -h3 scheme
at k = 128 in 16 x 16 tiles, a call with filler rows and an excluded row, y committed), changes them as those tests do, and
holds the Lean's verdict to the Python's on the same files and coin: the verdict, the reason, the refusal, the run's
commitment and source, every opened tile's verdict in order, and an accepted run's work (its credit at Pearl's assumption,
`credit_at_assumption`: the credited MACs, tiles and MACs, ρ and ε), and the window record W it held the passes to
(`verify_run.tie`, ``--tie W``). Passes salted by a live session's coins load under that session's commitment, and a coin
record's challenger draw (the v2 record's ``draw``) holds only as coin ``pouw/draw`` 0 of a draw session opened after the
run's commitment, its coin the record's source; ``--of-record`` takes no other draw. `test_a_recorded_h200_run` does the same on an H200 run's
retained passes (r20261103-140949-ca34's, `art:4e1431256cceafc7…`) against the verdicts r20261103-160350-f592 recorded,
where `POUW_PEARLPLUS_RECORDED` names a directory holding them (`passes/` and `verdicts/{run,control}.json`).

Where the Python raises past its own verdict the Lean refuses instead, and no case here reaches it: a debit replay or a
checked U that rejects (the Lean names it), and an opening whose leaf doesn't parse (an opening that fails). The Lean also
holds a y opening to the tile's real rows' positions, which the Python's own opener supplies.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

import verity_difftest as D  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
PACKAGE = ROOT / "verity" / "core" / "protocols" / "pouw" / "lean" / "pearlplus"
SOURCE = b"verifier" * 4
H100 = "pearl-c-h100-v1-h3"
RECORDED = os.environ.get("POUW_PEARLPLUS_RECORDED")


def _module(name: str, path: Path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules[name])
    return sys.modules[name]


T = _module("test_pearl_c_vllm", ROOT / "benchmarks" / "pouw" / "tests" / "test_pearl_c_vllm.py")
VR = T.VR
Wd = T.Wd
Ch = T.Ch
VP = T.load("pearl_c_vllm_volunteer_pass", T.KERNEL.parent / "pearl_c_vllm" / "volunteer_pass.py")
pytestmark = [pytest.mark.slow, pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")]


@pytest.fixture(scope="module")
def exe() -> Path:
    return D.build(PACKAGE, "pouw-pearlplus")


def lean(exe: Path, paths: list[Path], draws: int, source: bytes | None = SOURCE, record: Path | None = None,
         coins: Path | None = None, tie: Path | None = None, of_record: bool = False) -> dict:
    argv = [str(exe), "--coins", str(coins or paths[0].parent / "coins.json"), "--draws", str(draws)]
    argv += [] if source is None else ["--source", source.hex()]
    argv += [] if record is None else ["--record", str(record)]
    argv += [] if tie is None else ["--tie", str(tie)]
    argv += ["--of-record"] if of_record else []
    p = subprocess.run([*argv, *map(str, paths)], capture_output=True, text=True)
    assert p.returncode in (0, 1) and not p.stderr, p.stderr
    got = json.loads(p.stdout)
    assert p.returncode == (got["verdict"] != "accept")
    assert got.pop("drawn") in (None, draws)
    return got


def answer(out: dict) -> dict:
    """`verify_passes`' (or a served verify document's) outcome in the Lean's terms."""
    return {"verdict": out["answer"]["verdict"], "reason": out["answer"]["reason"], "refused": out.get("refused"),
            "commitment": out.get("commitment"), "source": out.get("source"), "verdicts": out.get("verdicts", []),
            "work": (out["answer"]["public"] or {}).get("work"), "tie": out.get("tie")}


def same(exe: Path, paths: list[Path], draws: int = 3, source: bytes | None = SOURCE, record: Path | None = None,
         tie: Path | None = None, rounds: list[int] | None = None, dropped: tuple[int, ...] = (), draw=None,
         of_record: bool = False) -> dict:
    """The Python's verdict on `paths` (under their coins file, held to the window record at `tie`, the passes rounds
    `rounds` and `dropped`, drawn by `draw`), asserted to be the Lean's; the Lean reads the record the Python wrote, if
    any, and its rounds."""
    window = None if tie is None else json.loads(tie.read_text())
    want = answer(VR.verify_passes(paths, draws, 1, "test", VR.epoch_coins(paths[0].parent / "coins.json"), source,
                                   window=window, rounds=rounds, dropped=dropped, record=record,
                                   **({} if draw is None else {"draw": draw})))
    got = lean(exe, paths, draws, source, record, tie=tie, of_record=of_record)
    assert got == want
    return got


def challenger_draw(entropy: int = 7, after: bytes | None = None):
    """`bench_arm.LiveDraw`'s draw on the run's commitment C from a seeded draw session (`window.draw_challenger`): its
    coin ``pouw/draw`` 0, opened after C (or `after`), and the record's ``draw`` naming it."""
    def draw(root: bytes, asking) -> tuple[bytes, dict]:
        ch = Wd.draw_challenger(0, random.Random(entropy).randbytes)
        session = {"session": "s20261104T120000Z-dr01", "seq": 3, "peer": "vy-dc-h200", "by": "vllm", "period": None}
        asking({"reserved": session})
        o = ch.open(Wd.DRAW, 0, root if after is None else after)
        return o.coin, {"from": VR.DRAW_CHALLENGER, "label": Wd.DRAW, "index": 0, "reserved": session,
                        "session": {"session": session["session"], "root": root.hex()},
                        "commitment": ch.commitment.to_json(), "opening": o.to_json()}
    return draw


def edit(path: Path, f) -> None:
    man = json.loads(path.read_text())
    f(man)
    path.write_text(json.dumps(man))


def test_an_honest_pass(exe, tmp_path):
    path = T.proved_pass(tmp_path, H100)[0]
    for draws in (3, 16):
        got = same(exe, [path], draws)
        assert got["verdict"] == "accept" and len(got["verdicts"]) == draws + 1
        assert 0 < got["work"]["credited"] < got["work"]["macs"]


def test_a_row_over_its_cap_and_its_give_up(exe, tmp_path):
    """Under the zero key a credited row is over its row cap; given up (V-EX), every tile passes; more given-up rows than
    V-EX allows, or records unlike the published root, are refused before the coin."""
    path = T.proved_pass(tmp_path, H100, key=bytes(32))[0]
    over = same(exe, [path], 16)
    assert over["verdict"] == "reject" and "the row cap" in over["reason"]
    assert VP.volunteer(path, T.published(path), 1)["ok"]
    assert same(exe, [path], 16)["verdict"] == "accept"
    man = json.loads(path.read_text())
    u = next(u for u in man["matmuls"] if u.get("voluntary_a"))
    many = [i for i in range(u["real_rows"]) if i not in u["excluded_a"]][:T.CW.voluntary_cap(u["m"]) + 1]
    for given, why in ((many, "more voluntary rows than V-EX allows"), ([], "not the ones whose root the pass published")):
        path.write_text(json.dumps({**man, "matmuls": [{**x, "voluntary_a": given} if x is u else x for x in man["matmuls"]]}))
        got = same(exe, [path])
        assert got["verdict"] == "reject" and why in got["reason"] and got["source"] is None


def test_openings_that_fail(exe, tmp_path):
    """A weight row, an activation row or a transcript leaf that isn't what the pass committed; a file whose size or
    SHA-256 isn't its manifest entry's; and records that leave out the excluded row."""
    path = T.proved_pass(tmp_path, H100)[0]
    man = json.loads(path.read_text())
    leaves = man["leaves"]
    for spec, why in ((man["weights"]["0"]["rows"], "weight opening"), (man["matmuls"][0]["rows"], "activation opening"),
                      (leaves, "leaves: its SHA-256 is not the manifest's")):
        f = path.parent / spec["path"]
        good = f.read_bytes()
        bad = bytearray(good)
        for j in range(0, len(bad), 256):
            bad[j] ^= 0x01
        f.write_bytes(bytes(bad))
        got = same(exe, [path], 8)
        assert got["verdict"] == "reject" and why in got["reason"], got["reason"]
        f.write_bytes(good)
    assert "sha256" in leaves
    f = path.parent / leaves["path"]
    good = f.read_bytes()
    bad = bytes(b ^ 0x01 if j % 256 == 0 else b for j, b in enumerate(good))
    f.write_bytes(bad)
    edit(path, lambda m: m["leaves"].update(sha256=hashlib.sha256(bad).hexdigest()))
    assert "transcript opening" in same(exe, [path], 8)["reason"]
    f.write_bytes(good + b"\0")
    edit(path, lambda m: m["leaves"].update(sha256=leaves["sha256"]))
    assert f"leaves: {len(good) + 1} bytes; the manifest says {len(good)}" in same(exe, [path], 8)["reason"]
    f.write_bytes(good)
    edit(path, lambda m: m["matmuls"][0].update(excluded_a=[]))
    assert "records are not the ones" in same(exe, [path])["reason"]


def test_a_served_y_that_is_not_the_unscale_of_its_u(exe, tmp_path):
    path = T.proved_pass(tmp_path, H100)[0]
    man = json.loads(path.read_text())
    old = [u["y"]["root"] for u in man["matmuls"]]
    for u in man["matmuls"]:
        y = bytearray((path.parent / u["y"]["rows"]["path"]).read_bytes())
        y[0::2] = bytes(b ^ 0x01 for b in y[0::2])
        u["y"] = T.SY.retain_y(path.parent, u["index"], bytes(y), u["n"], bytes.fromhex(man["salt"]),
                               bytes.fromhex(man["hm96_salt_key"]))
    path.write_text(json.dumps(man))
    assert "is not the unscale of the tile's U" in same(exe, [path])["reason"]
    for u, root in zip(man["matmuls"], old):
        u["y"]["root"] = root
    path.write_text(json.dumps(man))
    assert "y opening" in same(exe, [path])["reason"]
    for u in man["matmuls"]:
        f = path.parent / u["y"]["rows"]["path"]
        f.write_bytes(f.read_bytes() + b"\0")
    assert "the prover's files don't open its y: ValueError: " in same(exe, [path])["reason"]


def test_calls_refused_before_anything_is_opened(exe, tmp_path):
    path = T.proved_pass(tmp_path, H100)[0]
    for index, code in ((T.C.CALL_RANGE, "call-range"), (0, "repeated-call")):
        edit(path, lambda m: m["matmuls"][1].update(index=index))
        got = same(exe, [path])
        assert got["refused"][0] == code and got["verdicts"] == []


def test_an_epochs_passes(exe, tmp_path):
    """Two passes of one epoch sharing a call are refused; numbered on, both are drawn over and each opens its excluded
    row's tile."""
    a, b = (T.proved_pass(tmp_path / n, H100, ix)[0] for n, ix in (("a", (0, 1)), ("b", (1, 2))))
    assert same(exe, [a, b])["refused"][0] == "repeated-call"
    c = T.proved_pass(tmp_path / "c", H100, (2, 3))[0]
    got = same(exe, [a, c], 8)
    assert got["verdict"] == "accept" and {v["pass"] for v in got["verdicts"]} == {0, 1} and len(got["verdicts"]) == 10


def test_a_give_up_after_the_coin(exe, tmp_path):
    """The coin's record pins the run's commitment and source: the Lean writes the record the Python writes, reads the
    Python's, and refuses (`after-coin`) passes whose records changed after it."""
    path = T.proved_pass(tmp_path, H100, key=bytes(32))[0]
    py, lr = tmp_path / "coin.json", tmp_path / "lean-coin.json"
    first = same(exe, [path], record=py)
    assert lean(exe, [path], 3, SOURCE, lr) == first
    held = lambda r: {k: json.loads(r.read_text())[k] for k in ("commitment", "source")}  # noqa: E731
    assert held(lr) == held(py) == {"commitment": first["commitment"], "source": SOURCE.hex()}
    assert same(exe, [path], source=None, record=py) == first
    assert VP.volunteer(path, T.published(path), 1)["ok"]
    late = same(exe, [path], source=None, record=py)
    assert late["refused"][0] == "after-coin" and late["source"] is None
    assert lean(exe, [path], 3, None, lr) == late


def test_a_live_epoch_and_a_challenger_draw(exe, tmp_path):
    """Passes salted by a live session (``coins.json`` of `window.LIVE_SCHEME`, each salt coin opened after its salt
    epoch's root) and a coin record whose draw is a draw session's coin on the run's commitment: the Lean's verdict is the
    Python's, of record or not, and reads a v2 record the Lean wrote itself on the node's draw."""
    a, b = (T.proved_pass(tmp_path / n, H100, ix, live=0)[0] for n, ix in (("a", (0, 1)), ("b", (2, 3))))
    (b.parent / "coins.json").write_text((a.parent / "coins.json").read_text())
    rec = tmp_path / "coin.json"
    got = same(exe, [a, b], 8, None, rec, draw=challenger_draw())
    assert got["verdict"] == "accept" and len(got["source"]) == 128
    held = json.loads(rec.read_text())
    assert held["schema"] == VR.COIN_SCHEMA and held["draw"]["from"] == "challenger"
    assert lean(exe, [a, b], 8, None, rec, of_record=True) == got
    lr = tmp_path / "lean-coin.json"
    mine = lean(exe, [a, b], 8, SOURCE, lr)
    assert mine["verdict"] == "accept" and json.loads(lr.read_text())["draw"] == {"from": "node", "claim": VR.NODE_CLAIM}
    assert lean(exe, [a, b], 8, None, lr) == mine
    edit(b, lambda m: m.update(pass_index=1))
    late = same(exe, [b], 3, SOURCE)
    assert late["verdict"] == "reject" and "the salt's coin was not opened after what the verifier recorded" in late["reason"]


def test_a_verdict_of_record_takes_only_a_challenger_draw_on_live_coins(exe, tmp_path):
    """``--of-record`` refuses (`of-record`) the node's draw, a v1 coin record (which names no draw), no record, a source
    given on the command line and an in-process challenger's epoch coins; without it each is the Python's verdict."""
    live = T.proved_pass(tmp_path / "live", H100, live=0)[0]
    plain = T.proved_pass(tmp_path / "plain", H100)[0]
    node = tmp_path / "node.json"
    assert same(exe, [live], record=node)["verdict"] == "accept"
    v1 = tmp_path / "v1.json"
    v1.write_text(json.dumps({**{k: v for k, v in json.loads(node.read_text()).items() if k != "draw"},
                              "schema": "verity-pouw/pearl-plus-coin/v1"}))
    drawn = tmp_path / "drawn.json"
    assert same(exe, [plain], 3, None, drawn, draw=challenger_draw())["verdict"] == "accept"
    for paths, source, record, why in (([live], None, node, "drawn on the serving node"),
                                       ([live], None, v1, "not verity-pouw/pearl-plus-coin/v2"),
                                       ([live], SOURCE, None, "the source is given on the command line"),
                                       ([live], SOURCE, node, "the source is given on the command line"),
                                       ([plain], None, drawn, "an in-process challenger's")):
        got = lean(exe, paths, 3, source, record, of_record=True)
        assert (got["verdict"], got["refused"][0], got["verdicts"]) == ("reject", "of-record", []), why
        assert why in got["reason"] and got["reason"] == f"of-record: {got['refused'][1]}"
        assert lean(exe, paths, 3, source, record)["verdict"] == "accept"


def test_a_challenger_draw_that_is_not_the_runs_coin(exe, tmp_path):
    """A record whose challenger draw was opened after another commitment, whose source isn't its coin, whose commitment
    isn't a draw session's or whose opening doesn't parse: the Lean refuses it (`draw`) with the compose's words
    (`served_verdicts.pearlplus_live_draw`), of record or not, where the Python, which leaves the draw to the compose,
    accepts."""
    path = T.proved_pass(tmp_path, H100, live=0)[0]
    rec = tmp_path / "coin.json"
    other = VR.verify_passes([path], 3, 1, "test", VR.epoch_coins(path.parent / "coins.json"), record=rec,
                             draw=challenger_draw(after=b"\x0c" * 64))
    assert other["answer"]["verdict"] == "accept"
    good = tmp_path / "good.json"
    same(exe, [path], 3, None, good, draw=challenger_draw())
    held = json.loads(good.read_text())
    salt_only = Ch.LiveChallenger(Wd.LIVE_SCHEME, 0, Wd.SALT_FAMILIES, random.Random(7).randbytes).commitment.to_json()
    cases = [(json.loads(rec.read_text()), "the record's draw coin: coin was not opened after what the verifier recorded"),
             ({**held, "source": "ab" * 64}, "the record's source is not its draw session's coin"),
             ({**held, "draw": {**held["draw"], "commitment": salt_only}},
              "the record's draw coin: the coin commitment's coins are not a live window's draw session's"),
             ({**held, "draw": {**held["draw"], "opening": {"coin": "00"}}}, "the record's draw is malformed: ")]
    for doc, why in cases:
        (bad := tmp_path / "bad.json").write_text(json.dumps(doc))
        for of_record in (False, True):
            got = lean(exe, [path], 3, None, bad, of_record=of_record)
            assert (got["verdict"], got["refused"][0], got["verdicts"]) == ("reject", "draw", []), why
            assert got["refused"][1].startswith(why) and got["reason"] == f"draw: {got['refused'][1]}"


def test_a_run_tied_to_its_window_record(exe, tmp_path):
    """A window record W registering each call's A rows and y: the Lean holds the passes to W as the Python does and
    prints the same tie, W's digest and its table count, also when W lists a round dropped before the coin (the coin
    record's rounds and dropped, which the Python wrote)."""
    warm, a, b = (T.proved_pass(tmp_path / n, H100, ix)[0] for n, ix in (("w", (0, 1)), ("a", (2, 3)), ("b", (4, 5))))
    (w := tmp_path / "w.json").write_text(json.dumps(T.window_record(a, b)))
    got = same(exe, [a, b], 8, tie=w, record=tmp_path / "coin.json")
    assert got["verdict"] == "accept" and got["tie"] == {"record": T.REG.digest(json.loads(w.read_text())), "tables": 8}
    (w3 := tmp_path / "w3.json").write_text(json.dumps(T.window_record(warm, a, b)))
    got = same(exe, [a, b], 8, tie=w3, rounds=[1, 2], dropped=(0,), record=tmp_path / "coin3.json")
    assert got["verdict"] == "accept" and got["tie"]["tables"] == 8
    rec = json.loads((tmp_path / "coin3.json").read_text())
    assert (rec["rounds"], rec["dropped"]) == ([1, 2], [0])


def test_a_pass_that_is_not_ws(exe, tmp_path):
    """W whose entry for a call has another root, or lacks one, or registers a call the run doesn't have, or one twice,
    or isn't a v2 record: refused before the coin (Python's `ValueError`, in its words), as the Python does."""
    a, b = (T.proved_pass(tmp_path / n, H100, ix)[0] for n, ix in (("a", (0, 1)), ("b", (2, 3))))
    good = T.window_record(a, b)
    reg = good["registered"]
    cases = [({**good, "registered": [{**e, "root": "00" * 64} if e["value"] == "pearl-a/0/0" else e for e in reg]},
              "ValueError: W's pearl-a/0/0: its root not the call's"),
             ({**good, "registered": [{**e, "word_bits": 16, "leaves": 1} if e["value"] == "pearl-y/1/1" else e
                                      for e in reg]}, "ValueError: W's pearl-y/1/1: its leaves, word_bits not the call's"),
             ({**good, "registered": [e for e in reg if e["value"] != "pearl-y/1/0"]}, "ValueError: W registers no pearl-y/1/0"),
             ({**good, "registered": [*reg, {**reg[0], "value": "pearl-a/2/0"}]},
              "ValueError: W registers pearl-a/2/0, no call of the run's"),
             ({**good, "registered": [*reg, reg[1]]}, "ValueError: W registers pearl-y/0/0 twice"),
             ({**good, "format": "verity/registration/v1"},
              "ValueError: W is no verity/registration/v2 record with registered values")]
    for doc, why in cases:
        (w := tmp_path / "w.json").write_text(json.dumps(doc))
        got = same(exe, [a, b], tie=w)
        assert (got["verdict"], got["reason"], got["tie"], got["commitment"]) == ("reject", why, None, None), got
    edit(b, lambda m: m["matmuls"][1].update(root_a=m["matmuls"][1]["root_a"].upper()))
    (w := tmp_path / "w.json").write_text(json.dumps(good))
    got = same(exe, [a, b], tie=w)
    assert got["reason"] == "ValueError: W's pearl-a/1/1: its root not the call's"


@pytest.mark.skipif(not RECORDED, reason="POUW_PEARLPLUS_RECORDED names no recorded run")
@pytest.mark.parametrize("name, passes", [("run", ("prefill", "decode")), ("control", ("prefill", "decode", "control"))])
def test_a_recorded_h200_run(exe, name, passes):
    """ca34's retained passes, the served run's (accepted on 459 drawn tiles) and with its control pass, whose activations
    aren't the ones it committed (rejected on 17): the Lean's verdict is the one f592 recorded at 39025eba."""
    d = Path(RECORDED)
    want = json.loads((d / "verdicts" / f"{name}.json").read_text())
    paths = [d / "passes" / p / "manifest.json" for p in passes]
    got = lean(exe, paths, want["draws"], bytes.fromhex(want["source"]), coins=d / "passes" / "coins.json")
    assert got == answer(want)
