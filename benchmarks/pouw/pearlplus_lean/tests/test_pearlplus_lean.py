"""Pearl+'s run verdict in Lean (`pouw-pearlplus`, `PouwCheck.PearlPlus.check`) is the Python's (`verify_run.verify_passes`).

Each case proves passes on the CPU as verify_run's tests do (`test_pearl_c_vllm.proved_pass`: the H100 record's -h3 scheme
at k = 128 in 16 x 16 tiles, a call with filler rows and an excluded row, y committed), changes them as those tests do, and
holds the Lean's verdict to the Python's on the same files and coin: the verdict, the reason, the refusal, the run's
commitment and source, every opened tile's verdict in order, and an accepted run's work (its credit at Pearl's assumption,
`credit_at_assumption`: the credited MACs, tiles and MACs, ρ, ε and the draws over the inert-work bound) under each
`over_cap` rule, `uncredited` (the default: a tile over its bound is credited nothing and the run goes on) and `reject`,
and the window record W it held the passes to (`verify_run.tie`, ``--tie W``: each call's A rows and y, and each
weight's B rows at the root PoUW's weights tree commits them under).
The passes are a test instance's, read as one (``test_instance``, ``--test-instance``), so the work credits nothing;
read as a served run's they are refused (their tile is not 64 x 64), as passes of two salts are. A layer past `K_MAX` is
refused at load on both sides. Passes salted by a live session's coins load under that session's commitment, and a coin
record's challenger draw (the v2 record's ``draw``) holds only as coin ``pouw/draw`` 0 of a draw session opened after the
run's commitment, its coin the record's source; ``--of-record`` takes no other draw. `test_a_recorded_h200_run` does the
same on an H200 run's retained passes at the draw count and source of the verdicts the Python recorded on them, where
`POUW_PEARLPLUS_RECORDED` names a directory holding them (`passes/` and `verdicts/{run,control}.json`).

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
from fractions import Fraction
from pathlib import Path

import pytest
import settings

pytest.importorskip("torch")

import verity_difftest as D  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
PACKAGE = ROOT / "verity" / "protocols" / "pouw" / "verifier" / "lean" / "pearlplus"
SOURCE = b"verifier" * 4
H100 = "pearl-c-h100-v1-h3"


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
         coins: Path | None = None, over_cap: str | None = None, test_instance: bool = False, tie: Path | None = None,
         of_record: bool = False) -> dict:
    argv = [str(exe), "--coins", str(coins or paths[0].parent / "coins.json"), "--draws", str(draws)]
    argv += [] if source is None else ["--source", source.hex()]
    argv += [] if record is None else ["--record", str(record)]
    argv += [] if over_cap is None else ["--over-cap", over_cap]
    argv += ["--test-instance"] if test_instance else []
    argv += [] if tie is None else ["--tie", str(tie)]
    argv += ["--of-record"] if of_record else []
    p = subprocess.run([*argv, *map(str, paths)], capture_output=True, text=True)
    assert p.returncode in (0, 1) and not p.stderr, p.stderr
    got = json.loads(p.stdout)
    assert p.returncode == (got["verdict"] != "accept")
    assert got.pop("drawn") in (None, draws)
    return got


def answer(out: dict) -> dict:
    """`verify_passes`' (or a served verify document's) outcome in the Lean's terms.  A document written before
    ``over_cap`` ran under ``reject``, where an accepted run has no draw over the inert-work bound."""
    work = (out["answer"]["public"] or {}).get("work")
    return {"verdict": out["answer"]["verdict"], "reason": out["answer"]["reason"], "refused": out.get("refused"),
            "commitment": out.get("commitment"), "source": out.get("source"),
            "over_cap": out.get("over_cap", VR.REJECT), "verdicts": out.get("verdicts", []),
            "work": None if work is None else {"over_cap_draws": 0, **work}, "tie": out.get("tie")}


def same(exe: Path, paths: list[Path], draws: int = 3, source: bytes | None = SOURCE, record: Path | None = None,
         over_cap: str | None = None, test_instance: bool = True, tie: Path | None = None, rounds: list[int] | None = None,
         dropped: tuple[int, ...] = (), draw=None, of_record: bool = False) -> dict:
    """The Python's verdict on `paths` (under their coins file, held to the window record at `tie`, the passes rounds
    `rounds` and `dropped`, drawn by `draw`), asserted to be the Lean's; the Lean reads the record the Python wrote, if
    any, and its rounds. ``over_cap`` None is each side's default; the passes are a test instance's unless
    ``test_instance`` is false."""
    rule = {} if over_cap is None else {"over_cap": over_cap}
    window = None if tie is None else json.loads(tie.read_text())
    want = answer(VR.verify_passes(paths, draws, 1, "test", VR.epoch_coins(paths[0].parent / "coins.json"), source,
                                   window=window, rounds=rounds, dropped=dropped, record=record,
                                   test_instance=test_instance, **rule, **({} if draw is None else {"draw": draw})))
    got = lean(exe, paths, draws, source, record, over_cap=over_cap, test_instance=test_instance, tie=tie,
               of_record=of_record)
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
        works = []
        for rule in (None, *VR.OVER_CAP):
            got = same(exe, [path], draws, over_cap=rule)
            assert got["verdict"] == "accept" and len(got["verdicts"]) == draws + 1
            assert got["over_cap"] == (rule or VR.UNCREDITED) and got["work"]["over_cap_draws"] == 0
            assert got["work"]["credited"] == 0 < got["work"]["macs"]
            works.append(got["work"])
        assert works[0] == works[1] == works[2]


def test_an_excluded_weight_row_zeroes_its_block(exe, tmp_path):
    """The band rule on both sides: weight 1's row 20 fails liveness, so it is excluded and takes its block of 64 columns,
    all 32 of the second call's.  That call has no ticket and no credited cell, its exclusion tile is opened and passes,
    and the run's credited cells' MACs (``macs``; a test instance's credit is 0) are the first call's alone, its 19
    credited rows x 16 columns at |K| = 128."""
    path = T.proved_pass(tmp_path, H100, bad_b=20)[0]
    for rule in (None, *VR.OVER_CAP):
        got = same(exe, [path], 16, over_cap=rule)
        assert got["verdict"] == "accept" and got["work"]["over_cap_draws"] == 0
        assert (got["work"]["macs"], got["work"]["credited_tiles"]) == (19 * 16 * 128, 2)


def test_a_row_over_its_cap_and_its_give_up(exe, tmp_path):
    """Under the zero key a credited row is over its row cap: under ``reject`` the run is rejected; under ``uncredited``
    (the default) it is accepted, each tile over its inert-work bound ok with its bound's reason and credited nothing,
    the credit taking X_O, the draws on such tiles, at a larger ε.  Given up (V-EX), every tile passes; more given-up rows
    than V-EX allows, or records unlike the published root, are refused before the coin."""
    path = T.proved_pass(tmp_path, H100, key=bytes(32))[0]
    draws = 16
    over = same(exe, [path], draws, over_cap=VR.REJECT)
    assert over["verdict"] == "reject" and "the row cap" in over["reason"] and over["work"] is None
    for rule in (None, VR.UNCREDITED):
        took = same(exe, [path], draws, over_cap=rule)
        assert took["verdict"] == "accept" and took["over_cap"] == VR.UNCREDITED and took["reason"] is None
        assert [v["reason"] for v in took["verdicts"]] == [v["reason"] for v in over["verdicts"]]
        assert all(v["ok"] for v in took["verdicts"])
        drawn_over = sum(1 for v in took["verdicts"][:draws] if v["reason"])
        assert drawn_over > 0 and took["work"]["over_cap_draws"] == drawn_over
        assert all("the row cap" in v["reason"] for v in took["verdicts"] if v["reason"])
    p = VR.load(path, T.published(path))
    honest = VR.credit_at_assumption([p], draws, VR.DELTA)
    assert took["work"] == {**VR.credit_at_assumption([p], draws, VR.DELTA, drawn_over), "credited": 0}
    assert VR.credit_at_assumption([p], draws, VR.DELTA, drawn_over)["credited"] < honest["credited"]
    assert Fraction(took["work"]["epsilon"]) > Fraction(honest["epsilon"])
    assert VP.volunteer(path, T.published(path), 1)["ok"]
    for rule in VR.OVER_CAP:
        given = same(exe, [path], draws, over_cap=rule)
        assert given["verdict"] == "accept" and given["work"]["over_cap_draws"] == 0
    man = json.loads(path.read_text())
    u = next(u for u in man["matmuls"] if u.get("voluntary_a"))
    many = [i for i in range(u["real_rows"]) if i not in u["excluded_a"]][:T.CW.voluntary_cap(u["m"]) + 1]
    for given, why in ((many, "more voluntary rows than V-EX allows"), ([], "not the ones whose root the pass published")):
        path.write_text(json.dumps({**man, "matmuls": [{**x, "voluntary_a": given} if x is u else x for x in man["matmuls"]]}))
        got = same(exe, [path])
        assert got["verdict"] == "reject" and why in got["reason"] and got["source"] is None


def test_a_salt_dead_element_over_the_row_cap(exe, tmp_path, monkeypatch):
    """P3's forming term: credited row 9 holds one entry 300 times its RMS, which keeps the per-row rules and is salt-dead
    (its forming's two ends round to one code), inert work of fs·|cols|/n = 64 that puts the row over its inert-work
    bound, 41 at k = 128, where its words flag no atom. Under ``reject`` both verdicts reject the run on that tile; under
    ``uncredited`` both accept it with that tile, and only it, over its bound; without the term the Python's finds no
    tile over."""
    path = T.proved_pass(tmp_path, H100, spike=300.0)[0]
    why = "A row 9's replayed debit 64 exceeds its cap 41 (the row cap"
    over = same(exe, [path], 16, over_cap=VR.REJECT)
    assert over["verdict"] == "reject" and why in over["reason"]
    took = same(exe, [path], 16)
    reasons = [v["reason"] for v in took["verdicts"] if v["reason"]]
    assert took["verdict"] == "accept" and reasons and all(why in r for r in reasons)
    monkeypatch.setattr(T.CW.D, "forming_elements", lambda flags, dead, device=None: 0)
    without = answer(VR.verify_passes([path], 16, 1, "test", T.published(path), SOURCE, test_instance=True))
    assert without["verdict"] == "accept" and not any(v["reason"] for v in without["verdicts"])


def test_the_forming_term_reads_a_block_every_credited_word_flags_and_not_one_some_flag(exe, tmp_path, monkeypatch):
    """E's ∀ over the credited columns both ways (`forming_elements`, `Rule.rowElements`): under the zero key, credited row
    11 of the first call has atom-words flagging its blocks 0 … 3, all in one column.  On 16 x 16 tiles some of the row's
    16 credited words flag each block and the rest don't, so no block is inert and its debit is the chain's, 160, over
    its row cap, with or without the term.  On 16 x 1 tiles that column's tile has one credited word, which flags all
    four blocks, so every element of the row is inert and the tile's debit is 672, 160 without the term.  The Lean's
    verdicts are the Python's on both, under each ``over_cap`` rule."""
    cases = [((16, 16), 16, "A row 11's replayed debit 160 exceeds", "A row 11's replayed debit 160 exceeds"),
             ((16, 1), 64, "the replayed debit 672 exceeds", "the replayed debit 160 exceeds")]
    paths = []
    for tile, draws, why, _ in cases:
        path = T.proved_pass(tmp_path / f"{tile[0]}x{tile[1]}", H100, key=bytes(32), tile=tile)[0]
        paths.append(path)
        for rule in (None, *VR.OVER_CAP):
            got = same(exe, [path], draws, over_cap=rule)
            assert any(why in (v["reason"] or "") for v in got["verdicts"]), (tile, rule, got["verdicts"])
    monkeypatch.setattr(T.CW.D, "forming_elements", lambda flags, dead, device=None: 0)
    for path, (tile, draws, _, without) in zip(paths, cases):
        out = VR.verify_passes([path], draws, 1, "test", T.published(path), SOURCE, test_instance=True)
        assert any(without in (v["reason"] or "") for v in out["verdicts"]), tile


def test_the_forming_term_reads_a_row_some_of_whose_blocks_are_inert(exe, tmp_path, monkeypatch):
    """E's per-block read on one row (`forming_elements`, `Rule.rowElements`) where neither its all-inert shortcut nor a
    salt-dead element decides: under key 4, on 16 x 1 tiles, tile 14's one flagged row has its one credited word flag one
    of its four atom blocks and not the other three, and no element of the tile is salt-dead.  So 32 of the row's 128
    elements are inert and the tile's debit is 168 (40 for the flagged atom-word, 4 per inert element), 40 without the
    term; reading the three unflagged blocks as the inert ones would make it 424.  The Lean's verdicts are the Python's
    under each ``over_cap`` rule."""
    path = T.proved_pass(tmp_path, H100, key=bytes([4]) * 32, tile=(16, 1))[0]
    for rule in (None, *VR.OVER_CAP):
        got = same(exe, [path], 96, over_cap=rule)
        assert any(v["tile"] == 14 and "the replayed debit 168 exceeds the cap 38" in (v["reason"] or "")
                   for v in got["verdicts"]), (rule, got["verdicts"])
    monkeypatch.setattr(T.CW.D, "forming_elements", lambda flags, dead, device=None: 0)
    out = VR.verify_passes([path], 96, 1, "test", T.published(path), SOURCE, test_instance=True)
    assert any(v["tile"] == 14 and "the replayed debit 40 exceeds the cap 38" in (v["reason"] or "") for v in out["verdicts"])


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


def test_a_served_y_row_that_ends_before_its_tiles_columns(exe, tmp_path):
    """A prover's own y whose last real row is short (`short_y`: the rows file cut, the tree and root over the row as it
    is, so it opens), on a row whose unscale is +0 at every column (`zero_a`: its A row is zero, so it is excluded and its
    tile is opened whatever the draw), where the word a read past the Lean row's end gives, 0, is the word the check
    wants.  Both sides open the row the file holds and refuse it by its length, and half a word is no row on either."""
    path = T.proved_pass(tmp_path, H100, zero_a=19)[0]
    assert same(exe, [path])["verdict"] == "accept"
    T.short_y(path, 0, 8)
    assert same(exe, [path])["reason"].endswith("y[19]: the opened row holds 12 words, the tile's columns end at 16")
    T.short_y(path, 0, 1)
    assert same(exe, [path])["reason"].endswith(
        "the prover's files don't open its y: ValueError: buffer size must be a multiple of element size")


def test_a_layer_past_k_max(exe, tmp_path):
    """A weight of K_MAX + 1 = 2^16 + 1 channels, its call at the split's k (whole promotion groups) and |K|, with the
    records root the pass publishes over them: the pass load refuses it (`DomainError`, k ≤ 2^16 among the domain's
    bounds) before the coin, in both; `checkTile` refuses the same past any load."""
    path = T.proved_pass(tmp_path, H100)[0]
    man = json.loads(path.read_text())
    w, k_max = man["matmuls"][0]["weight"], 2 ** 16
    man["weights"][str(w)]["k"] = k_max + 1
    for u in man["matmuls"]:
        if u["weight"] == w:
            u["k"], u["k_kept"] = T.PW.Split(k_max + 1).k_unit, k_max + 1
    epoch, _ = T.PPD.manifest_epoch(man, T.published(path), False)
    calls = tuple(T.Matmul(u["index"], T.Shape(u["m"], u["k"], u["n"]), u["weight"]) for u in man["matmuls"])
    splits = {x: T.PW.Split(man["weights"][str(x)]["k"]) for x in epoch.workload.weights()}
    man["records_root"] = T.PW.records_root(epoch.salt, epoch.weights_root, splits, VR.records_of(man, calls)).hex()
    path.write_text(json.dumps(man))
    got = same(exe, [path])
    assert got["verdict"] == "reject" and got["source"] is None and got["verdicts"] == []
    assert got["reason"].startswith("DomainError: ") and f"k={T.PW.Split(k_max + 1).k_unit}," in got["reason"]
    assert f"<= k <= {k_max}" in got["reason"]


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


def test_a_served_runs_tile_and_salt(exe, tmp_path):
    """Read as a served run's, a test instance's pass is refused before the coin: its instance is k_min 32 at 16 x 16 or
    64 x 64 tiles, not the served k_min 1,024 at 64 x 64. Passes of two salts (other weights under the one coin), each
    holding calls 0 and 1, are refused, test instance or not."""
    a = T.proved_pass(tmp_path / "a", H100)[0]
    wide = T.proved_pass(tmp_path / "wide", H100, tile=(64, 64))[0]
    for path, tile in ((a, 16), (wide, 64)):
        served = same(exe, [path], test_instance=False)
        assert served["verdict"] == "reject" and served["source"] is None and served["verdicts"] == []
        assert served["reason"] == (f"ValueError: pass 0's instance is k_min 32 at {tile} x {tile} tiles, not the served "
                                    "1024 at 64 x 64: a test instance is read only as one, crediting nothing")
    assert same(exe, [wide])["verdict"] == "accept"
    b = T.proved_pass(tmp_path / "b", H100, seed=12)[0]
    for test_instance in (False, True):
        two = same(exe, [a, b], test_instance=test_instance)
        assert two["reason"] == "ValueError: the passes are not all of one salt" and two["refused"] is None


def test_a_give_up_after_the_coin(exe, tmp_path):
    """The coin's record pins the run's commitment and source: the Lean writes the record the Python writes, reads the
    Python's, and refuses (`after-coin`) passes whose records changed after it."""
    path = T.proved_pass(tmp_path, H100, key=bytes(32))[0]
    py, lr = tmp_path / "coin.json", tmp_path / "lean-coin.json"
    first = same(exe, [path], record=py)
    assert lean(exe, [path], 3, SOURCE, lr, test_instance=True) == first
    held = lambda r: {k: json.loads(r.read_text())[k] for k in ("commitment", "source")}  # noqa: E731
    assert held(lr) == held(py) == {"commitment": first["commitment"], "source": SOURCE.hex()}
    assert same(exe, [path], source=None, record=py) == first
    assert VP.volunteer(path, T.published(path), 1)["ok"]
    late = same(exe, [path], source=None, record=py)
    assert late["refused"][0] == "after-coin" and late["source"] is None
    assert lean(exe, [path], 3, None, lr, test_instance=True) == late


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
    assert lean(exe, [a, b], 8, None, rec, test_instance=True, of_record=True) == got
    lr = tmp_path / "lean-coin.json"
    mine = lean(exe, [a, b], 8, SOURCE, lr, test_instance=True)
    assert mine["verdict"] == "accept" and json.loads(lr.read_text())["draw"] == {"from": "node", "claim": VR.NODE_CLAIM}
    assert lean(exe, [a, b], 8, None, lr, test_instance=True) == mine
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
        got = lean(exe, paths, 3, source, record, test_instance=True, of_record=True)
        assert (got["verdict"], got["refused"][0], got["verdicts"]) == ("reject", "of-record", []), why
        assert why in got["reason"] and got["reason"] == f"of-record: {got['refused'][1]}"
        assert lean(exe, paths, 3, source, record, test_instance=True)["verdict"] == "accept"


def test_a_challenger_draw_that_is_not_the_runs_coin(exe, tmp_path):
    """A record whose challenger draw was opened after another commitment, whose source isn't its coin, whose commitment
    isn't a draw session's or whose opening doesn't parse: the Lean refuses it (`draw`) with the compose's words
    (`served_verdicts.pearlplus_live_draw`), of record or not, where the Python, which leaves the draw to the compose,
    accepts."""
    path = T.proved_pass(tmp_path, H100, live=0)[0]
    rec = tmp_path / "coin.json"
    other = VR.verify_passes([path], 3, 1, "test", VR.epoch_coins(path.parent / "coins.json"), record=rec,
                             draw=challenger_draw(after=b"\x0c" * 64), test_instance=True)
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
            got = lean(exe, [path], 3, None, bad, test_instance=True, of_record=of_record)
            assert (got["verdict"], got["refused"][0], got["verdicts"]) == ("reject", "draw", []), why
            assert got["refused"][1].startswith(why) and got["reason"] == f"draw: {got['refused'][1]}"


def test_a_run_tied_to_its_window_record(exe, tmp_path):
    """A window record W registering each call's A rows and y and each weight's B rows once: the Lean holds the passes to W as the Python does and
    prints the same tie, W's digest and its table count, also when W lists a round dropped before the coin (the coin
    record's rounds and dropped, which the Python wrote). A W that states no version (version 1, the run of record's,
    registering the calls alone) accepts in both on the calls' tables, the tie naming its weights untied."""
    warm, a, b = (T.proved_pass(tmp_path / n, H100, ix)[0] for n, ix in (("w", (0, 1)), ("a", (2, 3)), ("b", (4, 5))))
    (w := tmp_path / "w.json").write_text(json.dumps(T.window_record(a, b)))
    got = same(exe, [a, b], 8, tie=w, record=tmp_path / "coin.json")
    assert got["verdict"] == "accept" and got["tie"] == {"record": T.REG.digest(json.loads(w.read_text())), "tables": 10}
    (w3 := tmp_path / "w3.json").write_text(json.dumps(T.window_record(warm, a, b)))
    got = same(exe, [a, b], 8, tie=w3, rounds=[1, 2], dropped=(0,), record=tmp_path / "coin3.json")
    assert got["verdict"] == "accept" and got["tie"]["tables"] == 10
    rec = json.loads((tmp_path / "coin3.json").read_text())
    assert (rec["rounds"], rec["dropped"]) == ([1, 2], [0])
    (w1 := tmp_path / "w1.json").write_text(json.dumps(T.window_record(a, b, version=1)))
    got = same(exe, [a, b], 8, tie=w1, record=tmp_path / "coin1.json")
    assert got["verdict"] == "accept" and got["tie"] == {"record": T.REG.digest(json.loads(w1.read_text())), "tables": 8,
                                                         "weights": VR.WEIGHTS_UNTIED}


def test_a_pass_that_is_not_ws(exe, tmp_path):
    """W whose entry for a call or a weight has another root (B another weight's: a foreign rootB) or domain, or lacks
    one, or registers a call or weight the run doesn't have, or one twice, or registers no weight at all at version 2
    (every call's entry right: untied, not a tie), or states no version (version 1) and registers a weight, or states a
    version other than 2, or isn't a v2 record: refused before the coin (Python's `ValueError`, in its words), as the
    Python does."""
    a, b = (T.proved_pass(tmp_path / n, H100, ix)[0] for n, ix in (("a", (0, 1)), ("b", (2, 3))))
    good = T.window_record(a, b)
    reg = good["registered"]
    b0, b1 = (next(e for e in reg if e["value"] == f"pearl-b/{x}") for x in (0, 1))

    def at(v: str, **kw) -> dict:
        return {**good, "registered": [{**e, **kw} if e["value"] == v else e for e in reg]}
    cases = [({**good, "registered": [{**e, "root": "00" * 64} if e["value"] == "pearl-a/0/0" else e for e in reg]},
              "ValueError: W's pearl-a/0/0: its root not the call's"),
             ({**good, "registered": [{**e, "word_bits": 16, "leaves": 1} if e["value"] == "pearl-y/1/1" else e
                                      for e in reg]}, "ValueError: W's pearl-y/1/1: its leaves, word_bits not the call's"),
             ({**good, "registered": [e for e in reg if e["value"] != "pearl-y/1/0"]}, "ValueError: W registers no pearl-y/1/0"),
             ({**good, "registered": [*reg, {**reg[0], "value": "pearl-a/2/0"}]},
              "ValueError: W registers pearl-a/2/0, no call of the run's"),
             ({**good, "registered": [*reg, reg[1]]}, "ValueError: W registers pearl-y/0/0 twice"),
             (at("pearl-b/0", root=b1["root"]), "ValueError: W's pearl-b/0: its root not the weight's"),
             (at("pearl-b/1", domain=b0["domain"]), "ValueError: W's pearl-b/1: its domain not the weight's"),
             ({**good, "registered": [e for e in reg if e["value"] != "pearl-b/1"]}, "ValueError: W registers no pearl-b/1"),
             ({**good, "registered": [*reg, b0]}, "ValueError: W registers pearl-b/0 twice"),
             ({**good, "registered": [*reg, {**b0, "value": "pearl-b/2"}]},
              "ValueError: W registers pearl-b/2, no weight of the run's"),
             ({**good, "registered": [e for e in reg if not e["value"].startswith("pearl-b/")]},
              f"ValueError: {VR.UNTIED_WEIGHTS}, so the run's rootB is no W's (pearl-b/0 first)"),
             ({**(old := T.window_record(a, b, version=1)), "registered": [*old["registered"], b1]},
              f"ValueError: {VR.V1_WEIGHTS} pearl-b/1: a W that registers the weights states version 2"),
             ({**good, "window": {"kind": "served", "version": 1}}, f"ValueError: {T.PWR.UNKNOWN_VERSION}"),
             ({**good, "window": {"kind": "served", "version": "2"}}, f"ValueError: {T.PWR.UNKNOWN_VERSION}"),
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


@pytest.mark.skipif(not settings.RECORDED, reason="POUW_PEARLPLUS_RECORDED names no recorded run")
@pytest.mark.parametrize("name, passes", [("run", ("prefill", "decode")), ("control", ("prefill", "decode", "control"))])
def test_a_recorded_h200_run(exe, name, passes):
    """An H200 run's retained passes, the served run's (accepted) and with its control pass, whose activations aren't the
    ones it committed (rejected): under ``over_cap`` ``uncredited``, the run of record's, the Lean's verdict is the
    Python's on them at the draw count and source the Python recorded, and accepts or rejects as the recorded verdict did
    (written before the forming term and the band rule, so its credit isn't today's).  The Python replays as ``served_passes.py
    verify`` does, on the batch forms with every 256th tile on the scalar reference too (the scalar replay alone takes
    hours here).  It is the one case on a served instance's 64 x 64 tiles at k_min 1,024; each Lean replay takes about 45
    minutes on a CPU node, so ``reject`` is left to the CPU cases."""
    d = Path(settings.RECORDED)
    rec = json.loads((d / "verdicts" / f"{name}.json").read_text())
    paths = [d / "passes" / p / "manifest.json" for p in passes]
    coins, source = d / "passes" / "coins.json", bytes.fromhex(rec["source"])
    want = answer(VR.verify_passes(paths, rec["draws"], len(os.sched_getaffinity(0)), "test", VR.epoch_coins(coins), source,
                                   fast=True, cross_check=256, over_cap=VR.UNCREDITED))
    assert want["verdict"] == rec["answer"]["verdict"], want["reason"]
    assert lean(exe, paths, rec["draws"], source, coins=coins, over_cap=VR.UNCREDITED) == want
