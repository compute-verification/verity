"""Pearl+'s run verdict in Lean (`pouw-pearlplus`, `PouwVerifier.PearlPlus.check`) is the Python's (`verify_run.verify_passes`).

Each case proves passes on the CPU as verify_run's tests do (`test_pearl_c_vllm.proved_pass`: the H100 record's -h3 scheme
at k = 128 in 16 x 16 tiles, a call with filler rows and an excluded row, y committed), changes them as those tests do, and
holds the Lean's verdict to the Python's on the same files and coin: the verdict, the reason, the refusal, the run's
commitment and source, every opened tile's verdict in order, and an accepted run's work (its credit at its scheme's named assumption,
`credit_at_assumption`: the credited MACs, tiles and MACs, ρ, ε and the samples over the inert-work bound) under each
`over_cap` rule, `uncredited` (the default: a tile over its bound is credited nothing and the run goes on) and `reject`,
and the window record W it held the passes to (`verify_run.tie`, ``--tie W``: each call's A rows and y, and each
fixed operand's B rows at the root PoUW's fixed operands' tree commits them under), with the class whose fixed load it held them at
(`verify_run.held_to_load`, ``--class``). Held to a class, the work is in the grid form on both sides (`verify_run.grid_work`,
`PouwPearlPlus/Grid.lean`: the structural credit, MACs and tiles, and the give-up padded up to H1's window grid), with the
exact credit it replaces beside it (``exact_work``).
The passes are a test instance's, read as one (``test_instance``, ``--test-instance``), so the work credits nothing;
read as a served run's they are refused (their tile is not 64 x 64), as passes of two salts are. A layer past `K_MAX` is
refused at load on both sides. Passes salted by a live session's coins load under that session's commitment, and a coin
record's challenger sample (the v2 record's ``draw``) holds only as coin ``pouw/draw`` 0 of a sample session opened after the
run's commitment, its coin the record's source; ``--of-record`` takes no other sample. `test_a_recorded_h200_run` does the
same on an H200 run's retained passes at the sample count and source of the verdicts the Python recorded on them, where
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
from unittest.mock import patch

import pytest
import settings

pytest.importorskip("torch")

import verity_difftest as D  # noqa: E402
from verity.commitments import docket_digest as DD  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
PACKAGE = ROOT / "verity" / "protocols" / "accounting" / "compute" / "pouw" / "verifier" / "lean" / "pearlplus"
SOURCE = b"verifier" * 4
H100 = "pearl-c-h100-v1-h3"


def _module(name: str, path: Path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules[name])
    return sys.modules[name]


CASES = ROOT / "tests" / "verity" / "protocols" / "accounting" / "compute" / "pouw" / "verifier" / "lean" / "check"
RC = _module("pouw_rounds_cases", CASES / "pouw_rounds_cases.py")
PC = _module("pouw_pass_cases", CASES / "pouw_pass_cases.py")
T = _module("test_pearl_c_vllm", ROOT / "tests" / "benchmarks" / "pouw" / "test_pearl_c_vllm.py")
VR = T.VR
Wd = T.Wd
Ch = T.Ch
VP = T.load("pearl_c_vllm_volunteer_pass", T.POUW / "pearl_c_vllm" / "volunteer_pass.py")
pytestmark = [pytest.mark.slow, pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")]


@pytest.fixture(scope="module")
def exe() -> Path:
    return D.build(PACKAGE, "pouw-pearlplus")


def lean(exe: Path, paths: list[Path], draws: int, source: bytes | None = SOURCE, record: Path | None = None,
         coins: Path | None = None, over_cap: str | None = None, test_instance: bool = False, tie: Path | None = None,
         of_record: bool = False, load: tuple[int, int, int] | None = None, draw_live: Path | None = None,
         salt_live: Path | None = None) -> dict:
    argv = [str(exe), "--coins", str(coins or paths[0].parent / "coins.json"), "--draws", str(draws)]
    argv += [] if draw_live is None else ["--draw-live", str(draw_live)]
    argv += [] if salt_live is None else ["--salt-live", str(salt_live)]
    argv += [] if source is None else ["--source", source.hex()]
    argv += [] if record is None else ["--record", str(record)]
    argv += [] if over_cap is None else ["--over-cap", over_cap]
    argv += ["--test-instance"] if test_instance else []
    argv += [] if tie is None else ["--tie", str(tie)]
    argv += [] if load is None else ["--class", ",".join(map(str, load))]
    argv += ["--of-record"] if of_record else []
    p = subprocess.run([*argv, *map(str, paths)], capture_output=True, text=True)
    assert p.returncode in (0, 1) and not p.stderr, p.stderr
    got = json.loads(p.stdout)
    assert p.returncode == (got["verdict"] != "accept")
    assert got.pop("drawn") in (None, draws)
    return got


def answer(out: dict) -> dict:
    """`verify_passes`' (or a served verify document's) outcome in the Lean's terms.  A document written before
    ``over_cap`` ran under ``reject``, where an accepted run has no sample over the inert-work bound."""
    work = (out["answer"]["public"] or {}).get("work")
    return {"verdict": out["answer"]["verdict"], "reason": out["answer"]["reason"], "refused": out.get("refused"),
            "commitment": out.get("commitment"), "source": out.get("source"),
            "over_cap": out.get("over_cap", VR.REJECT), "verdicts": out.get("verdicts", []),
            "work": None if work is None else {"over_cap_draws": 0, **work},
            "exact_work": (out["answer"]["profile"] or {}).get("exact_work"), "tie": out.get("tie")}


def same(exe: Path, paths: list[Path], draws: int = 3, source: bytes | None = SOURCE, record: Path | None = None,
         over_cap: str | None = None, test_instance: bool = True, tie: Path | None = None, rounds: list[int] | None = None,
         dropped: tuple[int, ...] = (), draw=None, of_record: bool = False,
         load: tuple[int, int, int] | None = None) -> dict:
    """The Python's verdict on `paths` (under their coins file, held to the window record at `tie` and to the class
    `load`'s fixed load, the passes rounds `rounds` and `dropped`, sampled by `draw`), asserted to be the Lean's; the Lean
    reads the record the Python wrote, if any, and its rounds. ``over_cap`` None is each side's default; the passes are a test instance's unless
    ``test_instance`` is false."""
    rule = {} if over_cap is None else {"over_cap": over_cap}
    window = None if tie is None else json.loads(tie.read_text())
    want = answer(VR.verify_passes(paths, draws, 1, "test", VR.epoch_coins(paths[0].parent / "coins.json"), source,
                                   window=window, rounds=rounds, dropped=dropped, record=record,
                                   test_instance=test_instance, window_class=load, **rule,
                                   **({} if draw is None else {"draw": draw})))
    got = lean(exe, paths, draws, source, record, over_cap=over_cap, test_instance=test_instance, tie=tie,
               of_record=of_record, load=load)
    assert got == want
    return got


def challenger_sample(entropy: int = 7, after: bytes | None = None):
    """`bench_arm.LiveDraw`'s sample on the run's commitment C from a seeded sample session (`window.draw_challenger`): its
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


def pinned_y(monkeypatch, seed: int = 0) -> None:
    """`proved_pass` commits each call's y under fresh salts (`served_y.retain_y`), and y's roots are in the run's commitment,
    which keys the sample: seeded salts (`seeded_salts`) fix the sample, so a case that names a tile its draws reach reaches it
    on every run."""
    monkeypatch.setattr(T.SY, "fresh_salts", T.seeded_salts(seed))


def edit(path: Path, f) -> None:
    man = json.loads(path.read_text())
    f(man)
    path.write_text(json.dumps(man))


def test_an_honest_pass(exe, tmp_path):
    path = T.proved_pass(tmp_path, H100)[0]
    for draws in (3, 16):
        works = []
        for rule in (None, *VR.OVER_INERT_BOUND):
            got = same(exe, [path], draws, over_cap=rule)
            assert got["verdict"] == "accept" and len(got["verdicts"]) == draws + 1
            assert got["over_cap"] == (rule or VR.UNCREDITED) and got["work"]["over_cap_draws"] == 0
            assert got["work"]["credited"] == 0 < got["work"]["checked_macs"]
            works.append(got["work"])
        assert works[0] == works[1] == works[2]


def test_an_excluded_weight_row_zeroes_its_block(exe, tmp_path):
    """The band rule on both sides: fixed operand 1's row 20 fails liveness, so it is excluded and takes its block of 64 columns,
    all 32 of the second call's.  That call has no ticket and no credited cell, its exclusion tile is opened and passes,
    and the run's credited cells' MACs (``macs``; a test instance's credit is 0) are the first call's alone, its 19
    credited rows x 16 columns at |K| = 128."""
    path = T.proved_pass(tmp_path, H100, bad_b=20)[0]
    for rule in (None, *VR.OVER_INERT_BOUND):
        got = same(exe, [path], 16, over_cap=rule)
        assert got["verdict"] == "accept" and got["work"]["over_cap_draws"] == 0
        assert (got["work"]["checked_macs"], got["work"]["credited_tiles"]) == (19 * 16 * 128, 2)


def test_a_row_over_its_inert_bound_and_its_give_up(exe, tmp_path):
    """Under the zero key a credited row is over its row cap: under ``reject`` the run is rejected; under ``uncredited``
    (the default) it is accepted, each tile over its inert-work bound ok with its bound's reason and credited nothing,
    the credit taking X_O, the samples on such tiles, at a larger ε.  Given up (V-EX), every tile passes; more given-up rows
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
    for rule in VR.OVER_INERT_BOUND:
        given = same(exe, [path], draws, over_cap=rule)
        assert given["verdict"] == "accept" and given["work"]["over_cap_draws"] == 0
    man = json.loads(path.read_text())
    u = next(u for u in man["matmuls"] if u.get("voluntary_a"))
    many = [i for i in range(u["real_rows"]) if i not in u["excluded_a"]][:T.CW.voluntary_cap(u["m"]) + 1]
    for given, why in ((many, "more voluntary rows than V-EX allows"), ([], "not the ones whose root the pass published")):
        path.write_text(json.dumps({**man, "matmuls": [{**x, "voluntary_a": given} if x is u else x for x in man["matmuls"]]}))
        got = same(exe, [path])
        assert got["verdict"] == "reject" and why in got["reason"] and got["source"] is None


def test_a_salt_dead_element_over_the_row_inert_bound(exe, tmp_path, monkeypatch):
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
    monkeypatch.setattr(T.CW.D, "inert_elements", lambda flags, dead, device=None: 0)
    without = answer(VR.verify_passes([path], 16, 1, "test", T.published(path), SOURCE, test_instance=True))
    assert without["verdict"] == "accept" and not any(v["reason"] for v in without["verdicts"])


def test_the_forming_term_reads_a_block_every_credited_word_flags_and_not_one_some_flag(exe, tmp_path, monkeypatch):
    """E's ∀ over the credited columns both ways (`inert_elements`, `Rule.rowElements`): under the zero key, credited row
    11 of the first call has atom-words flagging its blocks 0 … 3, all in one column.  On 16 x 16 tiles some of the row's
    16 credited words flag each block and the rest don't, so no block is inert and its debit is the chain's, 160, over
    its row cap, with or without the term.  On 16 x 1 tiles that column's tile has one credited word, which flags all
    four blocks, so every element of the row is inert and the tile's debit is 672, 160 without the term.  The Lean's
    verdicts are the Python's on both, under each ``over_cap`` rule; y's salts are pinned, so the draws reach that tile."""
    cases = [((16, 16), 16, "A row 11's replayed debit 160 exceeds", "A row 11's replayed debit 160 exceeds"),
             ((16, 1), 64, "the replayed debit 672 exceeds", "the replayed debit 160 exceeds")]
    pinned_y(monkeypatch)
    paths = []
    for tile, draws, why, _ in cases:
        path = T.proved_pass(tmp_path / f"{tile[0]}x{tile[1]}", H100, key=bytes(32), tile=tile)[0]
        paths.append(path)
        for rule in (None, *VR.OVER_INERT_BOUND):
            got = same(exe, [path], draws, over_cap=rule)
            assert any(why in (v["reason"] or "") for v in got["verdicts"]), (tile, rule, got["verdicts"])
    monkeypatch.setattr(T.CW.D, "inert_elements", lambda flags, dead, device=None: 0)
    for path, (tile, draws, _, without) in zip(paths, cases):
        out = VR.verify_passes([path], draws, 1, "test", T.published(path), SOURCE, test_instance=True)
        assert any(without in (v["reason"] or "") for v in out["verdicts"]), tile


def test_the_forming_term_reads_a_row_some_of_whose_blocks_are_inert(exe, tmp_path, monkeypatch):
    """E's per-block read on one row (`inert_elements`, `Rule.rowElements`) where neither its all-inert shortcut nor a
    salt-dead element decides: under key 4, on 16 x 1 tiles, tile 14's one flagged row has its one credited word flag one
    of its four atom blocks and not the other three, and no element of the tile is salt-dead.  So 32 of the row's 128
    elements are inert and the tile's debit is 168 (40 for the flagged atom-word, 4 per inert element), 40 without the
    term; reading the three unflagged blocks as the inert ones would make it 424.  The Lean's verdicts are the Python's
    under each ``over_cap`` rule; y's salts are pinned, so the 96 draws reach tile 14."""
    pinned_y(monkeypatch)
    path = T.proved_pass(tmp_path, H100, key=bytes([4]) * 32, tile=(16, 1))[0]
    for rule in (None, *VR.OVER_INERT_BOUND):
        got = same(exe, [path], 96, over_cap=rule)
        assert any(v["tile"] == 14 and "the replayed debit 168 exceeds the cap 38" in (v["reason"] or "")
                   for v in got["verdicts"]), (rule, got["verdicts"])
    monkeypatch.setattr(T.CW.D, "inert_elements", lambda flags, dead, device=None: 0)
    out = VR.verify_passes([path], 96, 1, "test", T.published(path), SOURCE, test_instance=True)
    assert any(v["tile"] == 14 and "the replayed debit 40 exceeds the cap 38" in (v["reason"] or "") for v in out["verdicts"])


def test_openings_that_fail(exe, tmp_path):
    """A fixed-operand row, an activation row or a transcript leaf that isn't what the pass committed; a file whose size or
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
        u["y"] = T.SY.retain_y(path.parent, u["index"], bytes(y), u["n"], bytes.fromhex(man["salt"]))
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


def test_a_credited_call_that_commits_no_y_root(exe, tmp_path):
    """A call with a credited cell whose manifest names no y is refused at load on both sides, before the coin, since the
    tile check ties U to y only where y is committed. One with no credited cell (weight 1's excluded row 20 takes all its
    columns) commits none and is read: its exclusion tile is opened and passes, with no y to hold it to."""
    path = T.proved_pass(tmp_path / "credited", H100)[0]
    index = json.loads(path.read_text())["matmuls"][1]["index"]
    edit(path, lambda m: m["matmuls"][1].pop("y"))
    got = same(exe, [path])
    assert got["verdict"] == "reject" and got["source"] is None and got["verdicts"] == []
    assert got["reason"] == f"ValueError: matmul {index}: a credited call commits no y root"
    path = T.proved_pass(tmp_path / "uncredited", H100, bad_b=20)[0]
    edit(path, lambda m: m["matmuls"][1].pop("y"))
    assert same(exe, [path], 16)["verdict"] == "accept"


def test_a_served_y_row_that_ends_before_its_tiles_columns(exe, tmp_path):
    """A prover's own y whose last real row is short (`short_y`: the rows file cut, the tree and root over the row as it
    is, so it opens), on a row whose unscale is +0 at every column (`zero_a`: its A row is zero, so it is excluded and its
    tile is opened whatever the sample), where the word a read past the Lean row's end gives, 0, is the word the check
    wants.  Both sides open the row the file holds and refuse it by its length, and half a word is no row on either."""
    path = T.proved_pass(tmp_path, H100, zero_a=19)[0]
    assert same(exe, [path])["verdict"] == "accept"
    T.short_y(path, 0, 8)
    assert same(exe, [path])["reason"].endswith("y[19]: the opened row holds 12 words, the tile's columns end at 16")
    T.short_y(path, 0, 1)
    assert same(exe, [path])["reason"].endswith(
        "the prover's files don't open its y: ValueError: buffer size must be a multiple of element size")


def test_a_layer_past_k_max(exe, tmp_path):
    """A fixed operand of K_MAX + 1 = 2^16 + 1 channels, its call at the split's k (whole promotion groups) and |K|, with the
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
    """A call's index is its place, recomputed on both sides (`verify_run.passes_placed`, `Load.passesPlaced`, over
    `protocol.call_place` and `Load.callPlace`): pass i, round ``rounds[i]`` (the coin record's, default i), holds its call c
    at round * 2^16 + c.  The run is refused by name before anything is opened where a call holds another index
    (``call-place``: past 2^32, an earlier position's, another round's, not an integer), where the rounds are not one a
    pass (``call-place``), where two passes are one round (``repeated-call``), and where a round reaches 2^16
    (``call-range``)."""
    path = T.proved_pass(tmp_path, H100)[0]
    for index in (T.C.CALL_RANGE, 0, T.call_place(1, 1), "1"):
        edit(path, lambda m: m["matmuls"][1].update(index=index))
        got = same(exe, [path])
        assert got["refused"][0] == VR.CALL_PLACE and got["verdicts"] == [] and f"call 1 holds index {index!r}," in got["refused"][1]
    edit(path, lambda m: m["matmuls"][1].update(index=1))
    record = tmp_path / "coin.json"
    for passes, rounds, code in ((1, [1 << 16], "call-range"), (2, [0, 0], "repeated-call"), (1, [0, 1], VR.CALL_PLACE)):
        record.write_text(json.dumps({"schema": VR.COIN_SCHEMA, "commitment": "00" * 64, "source": SOURCE.hex(),
                                      "rounds": rounds, "dropped": []}))
        got = same(exe, [path] * passes, record=record, rounds=rounds)
        assert got["refused"][0] == code and got["verdicts"] == []


def test_an_epochs_passes(exe, tmp_path):
    """Two passes of one epoch at one place, each proved as round 0, are refused: the second is round 1, whose calls' places
    are 2^16 and 2^16 + 1.  At their places, rounds 0 and 1, both are sampled over and each opens its excluded row's
    tile."""
    a, b = (T.proved_pass(tmp_path / n, H100)[0] for n in ("a", "b"))
    assert same(exe, [a, b])["refused"][0] == VR.CALL_PLACE
    c = T.proved_pass(tmp_path / "c", H100, 1)[0]
    got = same(exe, [a, c], 8)
    assert got["verdict"] == "accept" and {v["pass"] for v in got["verdicts"]} == {0, 1} and len(got["verdicts"]) == 10


def test_the_tiles_the_lean_draws_are_pearl_plus_credited_blocks_drawn(exe, tmp_path):
    """Over two passes of one epoch, the tiles the Lean opens for its sample, (pass, tile id) in its order, are the
    credited blocks `verify_run.sampled_blocks` takes on the run's key, each pass's before its exclusion tiles, and each a block
    of `verify_run.credited_blocks` with a ticket."""
    a, c = (T.proved_pass(tmp_path / n, H100, s)[0] for n, s in (("a", 0), ("c", 1)))
    draws = 16
    got = same(exe, [a, c], draws)
    passes = [VR.load(x, T.published(x)) for x in (a, c)]
    records = [p.records for p in passes]
    key = VR.derive(SOURCE, VR.RUN_DRAW_DOMAIN, {"run": VR.run_commitment(passes)})
    drawn = VR.sampled_blocks(passes[0].scheme, records, key, draws)
    opened = [(v["pass"], v["tile"]) for v in got["verdicts"]]
    for i in (0, 1):
        mine = [b.id for b in drawn if b.id[0] == i]
        assert [x for x in opened if x[0] == i][:len(mine)] == mine
    assert len(opened) == draws + sum(len(p.tile_ids(T.CW.exclusion_tiles(p.scheme, p.records))) for p in passes)
    blocks = VR.credited_blocks(passes[0].scheme, records)
    assert all(b in blocks and b.ticket > 0 for b in drawn)
    assert [(b.id, b.rows, b.cols) for b in blocks] == [((i, r.id), r.tile.rows, r.tile.cols) for i, p in enumerate(passes)
                                                        for r in p.verifier.refs]


def test_a_served_runs_tile_and_salt(exe, tmp_path):
    """Read as a served run's, a test instance's pass is refused before the coin: its instance is k_min 32 at 16 x 16 or
    64 x 64 tiles, not the served k_min 1,024 at 64 x 64. Passes of two salts (other fixed operands under the one coin), each
    at its place, are refused, test instance or not."""
    a = T.proved_pass(tmp_path / "a", H100)[0]
    wide = T.proved_pass(tmp_path / "wide", H100, tile=(64, 64))[0]
    for path, tile in ((a, 16), (wide, 64)):
        served = same(exe, [path], test_instance=False)
        assert served["verdict"] == "reject" and served["source"] is None and served["verdicts"] == []
        assert served["reason"] == (f"ValueError: pass 0's instance is k_min 32 at {tile} x {tile} tiles, not the served "
                                    "1024 at 64 x 64: a test instance is read only as one, crediting nothing")
    assert same(exe, [wide])["verdict"] == "accept"
    b = T.proved_pass(tmp_path / "b", H100, 1, seed=12)[0]
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
    epoch's root) and a coin record whose sample is a sample session's coin on the run's commitment: the Lean's verdict is the
    Python's, of record or not, and reads a v2 record the Lean wrote itself on the node's sample."""
    a, b = (T.proved_pass(tmp_path / n, H100, ix, live=0)[0] for n, ix in (("a", 0), ("b", 1)))
    (b.parent / "coins.json").write_text((a.parent / "coins.json").read_text())
    rec = tmp_path / "coin.json"
    got = same(exe, [a, b], 8, None, rec, draw=challenger_sample())
    assert got["verdict"] == "accept" and len(got["source"]) == 128
    held = json.loads(rec.read_text())
    assert held["schema"] == VR.COIN_SCHEMA and held["draw"]["from"] == "challenger"
    undrawn = lean(exe, [a, b], 8, None, rec, test_instance=True, of_record=True)
    assert (undrawn["verdict"], undrawn["refused"][0]) == ("reject", "draw-session") and "--draw-live" in undrawn["reason"]
    lr = tmp_path / "lean-coin.json"
    mine = lean(exe, [a, b], 8, SOURCE, lr, test_instance=True)
    assert mine["verdict"] == "accept" and json.loads(lr.read_text())["draw"] == {"from": "node", "claim": VR.NODE_CLAIM}
    assert lean(exe, [a, b], 8, None, lr, test_instance=True) == mine
    edit(a, lambda m: m.update(pass_index=1))
    late = same(exe, [a], 3, SOURCE)
    assert late["verdict"] == "reject" and "the salt's coin was not opened after what the verifier recorded" in late["reason"]


def test_a_verdict_of_record_takes_only_a_challenger_draw_on_live_coins(exe, tmp_path):
    """``--of-record`` refuses (`of-record`) the node's sample, a v1 coin record (which names no sample), no record, a source
    given on the command line and an in-process challenger's epoch coins; without it each is the Python's verdict."""
    live = T.proved_pass(tmp_path / "live", H100, live=0)[0]
    plain = T.proved_pass(tmp_path / "plain", H100)[0]
    node = tmp_path / "node.json"
    assert same(exe, [live], record=node)["verdict"] == "accept"
    v1 = tmp_path / "v1.json"
    v1.write_text(json.dumps({**{k: v for k, v in json.loads(node.read_text()).items() if k != "draw"},
                              "schema": "verity-pouw/pearl-plus-coin/v1"}))
    drawn = tmp_path / "drawn.json"
    assert same(exe, [plain], 3, None, drawn, draw=challenger_sample())["verdict"] == "accept"
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
    """A record whose challenger sample was opened after another commitment, whose source isn't its coin, whose commitment
    isn't a sample session's or whose opening doesn't parse: the Lean refuses it (`draw`) with the compose's words
    (`served_verdicts.pearlplus_live_sample`), of record or not, where the Python, which leaves the sample to the compose,
    accepts."""
    path = T.proved_pass(tmp_path, H100, live=0)[0]
    rec = tmp_path / "coin.json"
    other = VR.verify_passes([path], 3, 1, "test", VR.epoch_coins(path.parent / "coins.json"), record=rec,
                             draw=challenger_sample(after=b"\x0c" * 64), test_instance=True)
    assert other["answer"]["verdict"] == "accept"
    good = tmp_path / "good.json"
    same(exe, [path], 3, None, good, draw=challenger_sample())
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


def test_a_challenger_draw_its_draw_session_does_not_hold_is_refused_by_the_lean(exe, tmp_path):
    """``--draw-live`` holds a challenger sample to its draw session (`Coins.liveDrawReason`): a challenger key that doesn't
    parse, or a ledger that opens no session on the run's commitment, rejects (`draw-session`) the run the Lean accepts
    without it, of record or not; a node's sample is no draw session's, and the verdict on it is the same with or without."""
    path = T.proved_pass(tmp_path, H100, live=0)[0]
    good, node = tmp_path / "good.json", tmp_path / "node.json"
    assert same(exe, [path], 3, None, good, draw=challenger_sample())["verdict"] == "accept"
    live = {"key": None, "ledger": [], "pins": [], "main": [], "session": {}}
    key = {"H": 5, "W": 4, "public": "ab" * 80}
    pin = hashlib.sha512(bytes.fromhex(key["public"])).hexdigest()
    for doc, why in ((live, "the draw session's challenger key is malformed: "),
                     ({**live, "key": key}, "the sample session's challenger key is not the one the verifier pins"),
                     ({**live, "key": key, "key_sha512": pin}, "the draw session is not trusted: ")):
        (f := tmp_path / "live.json").write_text(json.dumps(doc))
        for of_record in (False, True):
            got = lean(exe, [path], 3, None, good, test_instance=True, of_record=of_record, draw_live=f)
            assert (got["verdict"], got["refused"][0], got["work"]) == ("reject", "draw-session", None), why
            assert got["refused"][1].startswith(why) and got["reason"] == f"draw-session: {got['refused'][1]}"
    plain = same(exe, [path], record=node)
    assert lean(exe, [path], 3, record=node, test_instance=True, draw_live=f) == plain


def salt_session_pass(tmp_path: Path, k: int = 0) -> tuple[Path, dict]:
    """A test instance's pass salted by a signed salt session (`pouw_pass_cases.honest_live`'s): the session S of the
    auditor's challenger, its salt coin opened after salt epoch `k`'s root R over the pass's fixed operands' root W, the
    pass proved under that coin, the auditor's ledger reserving S, opening it on R and ending it complete, and the
    manifest carrying the gate's answer; with ``{man, r, s, w, live}``, ``live`` what ``--salt-live`` reads."""
    rng, held, real = random.Random(k + 5), {}, T.weights_root

    def weights_root(roots):
        held["w"] = real(roots)
        return held["w"]

    def coin(after: bytes):
        held["s"], held["r"] = PC.salt_session(rng, after, 0, RC.session_id(rng)), after
        return held["s"].ch.commitment, held["s"].salt[0]
    with patch.object(T, "weights_root", weights_root), patch.object(T, "live_coin", coin):
        path = T.proved_pass(tmp_path, H100, live=k)[0]
    s = held["s"]
    h = {"man": json.loads(path.read_text()), "r": held["r"], "s": s, "w": held["w"],
         "live": {"key": s.key(), "key_sha512": hashlib.sha512(bytes.fromhex(s.key()["public"])).hexdigest(),
                  "ledger": None, "pins": list(RC.PINS), "main": list(RC.MAIN), "session": s.files()}}
    PC.live_ledger(rng, h)
    path.write_text(json.dumps(h["man"]))
    return path, h


def test_a_live_pass_is_held_to_its_signed_salt_session(exe, tmp_path):
    """``--salt-live`` holds each live pass's salt to its salt session (`Coins.liveSaltReason`, `core_check.live_salt`'s
    session half): a pass salted by a signed session's coin, opened after its salt epoch's root R over the fixed operands'
    root W, gets the verdict it gets without it, and is refused (`salt-session`) when the auditor's ledger is signed under
    another key than the pinned one, when the session's salt coin followed another root than R (so was released before
    W was logged), when the ledger opened the session on another salt epoch's root over W, when the challenger key
    doesn't parse or isn't the one ``key_sha512`` pins, and when the manifest carries another session than the gate's
    answer."""
    path, h = salt_session_pass(tmp_path)
    plain = lean(exe, [path], 3, test_instance=True)
    assert plain["verdict"] == "accept"

    def held(live: dict) -> dict:
        (f := tmp_path / "salt-live.json").write_text(json.dumps(live))
        return lean(exe, [path], 3, test_instance=True, salt_live=f)

    def ledger(s=None, **kw) -> dict:
        g = {**h, "s": s or h["s"], "live": {**h["live"]}, "man": {**h["man"]}}
        if s is not None:
            g["live"]["session"] = s.files()
        PC.live_ledger(random.Random(9), g, **kw)
        return g["live"]

    assert held(h["live"]) == plain
    rng, sid = random.Random(11), h["s"].sid
    other = T.PS.epoch_open_root(h["w"], 1)
    cases = [(ledger(k=1), "the salt session is not trusted: "),
             (ledger(PC.salt_session(rng, rng.randbytes(64), 0, sid)),
              "the challenger's log does not show the epoch's root as the one root its salt coin followed"),
             (ledger(PC.salt_session(rng, other, 0, sid), root=other), "the salt session is not trusted: "),
             ({**h["live"], "key": None}, "the salt session's challenger key is malformed: "),
             ({**h["live"], "key_sha512": "ab" * 64}, "the salt session's challenger key is not the one the verifier pins")]
    for live, why in cases:
        got = held(live)
        assert (got["verdict"], got["refused"][0], got["work"]) == ("reject", "salt-session", None), why
        assert got["refused"][1].startswith(f"pass 0: {why}") and got["reason"] == f"salt-session: {got['refused'][1]}"
    edit(path, lambda m: m["session"].update(secret="00" * 32))
    got = held(h["live"])
    assert got["refused"] == ["salt-session", "pass 0: the pass's session is not the gate's answer, without its secret, for "
                              "the session the auditor's ledger opened on its epoch's root: its session, seq, root and "
                              "commitment"]


def signed_sample(held: dict):
    """`challenger_sample` from a signed sample session: a `pouw_rounds_cases.Session` of a sample session's layout, its
    record and close signed under the test key, its coin 0 opened after the run's commitment C; ``held`` keeps the
    session and C, and a second ask on the same C answers the same session."""
    def sample(root: bytes, asking) -> tuple[bytes, dict]:
        if held.get("c") != root:
            rng = random.Random(13)
            held.update(c=root, ds=RC.Session(rng, Wd.draw_challenger(0, rng.randbytes), [], salt=None,
                                              sid=RC.session_id(rng), leaf=RC.DRAW_LEAF, draw=lambda _: root))
        ds = held["ds"]
        session = {"session": ds.sid, "seq": 3, "peer": "vy-dc-h200", "by": "vllm", "period": None}
        asking({"reserved": session})
        o = ds.draw[0]
        return o.coin, {"from": VR.DRAW_CHALLENGER, "label": Wd.DRAW, "index": 0, "reserved": session,
                        "session": {"session": ds.sid, "root": root.hex()},
                        "commitment": ds.ch.commitment.to_json(), "opening": o.to_json()}
    return sample


def both_sessions(h: dict, held: dict) -> list[str]:
    """The auditor's ledger, signed under the test key, reserving and opening `h`'s salt session on its epoch's root and
    ending it as the gate does, then reserving the sample session `held` keeps and opening it withheld on C; and the
    manifest's ``session`` the gate's answer to that salt open."""
    rng, s, ds = random.Random(17), h["s"], held["ds"]

    def asked(c: Ch.LiveCommitment) -> dict:
        return {"scheme": c.scheme, "epoch": c.epoch, "families": [list(f) for f in c.families]}
    events = RC.ledger_events(rng, {"root": h["r"].hex()}, s, reserved=True, opened=asked(s.ch.commitment),
                              **PC.gate_end(s))
    o = {**RC._open(rng, {"root": held["c"].hex()}, ds.pub), "withheld": True, **asked(ds.ch.commitment)}
    events = RC._bind(events + RC._reserved(rng, ds.sid, o) + [("end", ds.sid, RC._end(ds))])
    ledger = RC.ledger_lines(events, RC._key(0), RC.leaves_of(events, s.sid))
    h["man"]["session"] = PC.carried(next(e for e in map(json.loads, ledger)
                                          if e["kind"] == "open" and e["session"] == s.sid), s.ch.commitment)
    return ledger


def test_a_live_run_of_record_is_held_to_both_signed_sessions(exe, tmp_path):
    """A live run of record end to end: its salt from a signed salt session and its sample from a signed sample session,
    both opened in one auditor's ledger under the pinned key. Given both sessions (``--draw-live``, ``--salt-live``) the
    verdict of record accepts with the work the verdict gets without ``--of-record``. Given the sample session alone it is
    refused (`salt-session`) for want of the salt session, and given the salt session alone it is refused for want of
    the sample session. It is the one test where a verdict of record accepts on signed sessions, and the one that reaches
    the missing-salt-session refusal, which the missing-sample-session refusal otherwise shadows; that needs a proved
    pass, which no cheaper test can stand in for."""
    path, h = salt_session_pass(tmp_path)
    held, rec = {}, tmp_path / "coin.json"
    VR.verify_passes([path], 3, 1, "test", VR.epoch_coins(path.parent / "coins.json"), None,
                     record=tmp_path / "probe.json", draw=signed_sample(held), test_instance=True)
    ledger = both_sessions(h, held)
    path.write_text(json.dumps(h["man"]))
    plain = same(exe, [path], 3, None, rec, draw=signed_sample(held))
    assert plain["verdict"] == "accept"
    files = {name: tmp_path / f"{name}-live.json" for name in ("sample", "salt")}
    for name, s in (("sample", held["ds"]), ("salt", h["s"])):
        files[name].write_text(json.dumps({**h["live"], "ledger": ledger, "session": s.files()}))

    def of_record(**live: Path) -> dict:
        return lean(exe, [path], 3, None, rec, test_instance=True, of_record=True, **live)
    both = of_record(draw_live=files["sample"], salt_live=files["salt"])
    assert (both["verdict"], both["refused"], both["work"]) == ("accept", None, plain["work"]), both
    sample_only = of_record(draw_live=files["sample"])
    assert sample_only["refused"] == ["salt-session", "no salt session was given (--salt-live), so nothing checked the "
                                                    "challenger's signatures on it"], sample_only
    assert (sample_only["verdict"], sample_only["work"]) == ("reject", None)
    salt_only = of_record(salt_live=files["salt"])
    assert (salt_only["verdict"], salt_only["refused"][0]) == ("reject", "draw-session") and \
        "--draw-live" in salt_only["reason"], salt_only


def test_a_run_tied_to_its_window_record(exe, tmp_path):
    """A window record W registering each call's A rows and y and each fixed operand's B rows once: the Lean holds the passes to W as the Python does and
    prints the same tie, W's digest and its table count, also when W lists a round dropped before the coin (the coin
    record's rounds and dropped, which the Python wrote). A W that states no version (version 1, the run of record's,
    registering the calls alone) accepts in both on the calls' tables, the tie naming its fixed operands untied."""
    warm, a, b = (T.proved_pass(tmp_path / n, H100, s)[0] for n, s in (("w", 0), ("a", 1), ("b", 2)))
    (w := tmp_path / "w.json").write_text(json.dumps(T.window_record(warm, a)))
    got = same(exe, [warm, a], 8, tie=w, record=tmp_path / "coin.json")
    assert got["verdict"] == "accept" and got["tie"] == {"record": DD.digest(json.loads(w.read_text())), "tables": 10}
    (w3 := tmp_path / "w3.json").write_text(json.dumps(T.window_record(warm, a, b)))
    got = same(exe, [a, b], 8, tie=w3, rounds=[1, 2], dropped=(0,), record=tmp_path / "coin3.json")
    assert got["verdict"] == "accept" and got["tie"]["tables"] == 10
    rec = json.loads((tmp_path / "coin3.json").read_text())
    assert (rec["rounds"], rec["dropped"]) == ([1, 2], [0])
    (w1 := tmp_path / "w1.json").write_text(json.dumps(T.window_record(warm, a, version=1)))
    got = same(exe, [warm, a], 8, tie=w1, record=tmp_path / "coin1.json")
    assert got["verdict"] == "accept" and got["tie"] == {"record": DD.digest(json.loads(w1.read_text())), "tables": 8,
                                                         "weights": VR.WEIGHTS_UNTIED}


def test_a_tied_run_held_to_its_classs_fixed_load(exe, tmp_path):
    """With the window's class (``--class``, `verify_passes`' ``window_class``) the Lean holds the tied passes to the class's
    fixed load as the Python does. Passes that are the class's (2 slots of 8 prompt ids and 2 outputs: round 0's calls of
    16 rows, round 1's of 2 in 16) accept in both, the tie naming the class, also with a round dropped before the coin
    (the coin record's rounds, which the Python wrote), and their work is in the grid form in both: the passes' whole
    records' MACs and tiles, the give-up the excluded rows' credit padded up to the grid, and the exact credit beside it
    (``exact_work``); without the class the work is the exact credit and there is no ``exact_work``. Passes of another load (a round short or over, the rounds in
    another order, a call of other rows or of other committed rows) are refused before the coin in both, in the same
    words, the class's alone, also over a retention whose rounds a coin record names. The Lean takes ``--class`` only
    beside ``--tie``, and only as three positive integers."""
    cls = (2, 8, 2)
    a = T.proved_pass(tmp_path / "a", H100, 0, load=(16, 16))[0]
    b = T.proved_pass(tmp_path / "b", H100, 1, load=(2, 16))[0]
    wide = T.proved_pass(tmp_path / "wide", H100, 1, load=(2, 32))[0]
    b0, a1 = (T.proved_pass(tmp_path / n, H100, s, load=rows)[0] for n, s, rows in (("b0", 0, (2, 16)), ("a1", 1, (16, 16))))

    def w(*paths: Path) -> Path:
        (f := tmp_path / f"w{len(list(tmp_path.glob('w*.json')))}.json").write_text(json.dumps(T.window_record(*paths)))
        return f
    wab = w(a, b)
    got = same(exe, [a, b], 8, tie=wab, load=cls, record=tmp_path / "coin.json")
    assert got["verdict"] == "accept" and got["tie"] == {"record": DD.digest(json.loads(wab.read_text())), "tables": 10,
                                                         "load": [2, 8, 2]}
    passes = [VR.load(x, VR.epoch_coins(x.parent / "coins.json")) for x in (a, b)]
    work, kept = got["work"], got["exact_work"]
    exact = VR.credit_at_assumption(passes, 8, VR.DELTA, work["over_cap_draws"])
    whole = VR.structural_credit(passes, 8, VR.DELTA, work["over_cap_draws"])
    assert kept == {"credited": 0, "credited_tiles": exact["credited_tiles"], "checked_macs": exact["checked_macs"]}
    assert work == {**whole, "credited": 0, "give_up": Wd.pad(whole["credited"] - exact["credited"])}
    assert work["checked_macs"] > kept["checked_macs"] and work["credited_tiles"] >= kept["credited_tiles"] and work["give_up"] > 0
    got = same(exe, [a, b], 8, tie=wab, record=tmp_path / "coin.json")
    assert got["verdict"] == "accept" and "load" not in got["tie"]
    assert got["exact_work"] is None and got["work"] == {**exact, "credited": 0}
    got = same(exe, [b], 8, tie=wab, load=cls, rounds=[1], dropped=(0,), record=tmp_path / "coin1.json")
    assert got["verdict"] == "accept" and got["tie"]["load"] == [2, 8, 2]
    assert got["work"]["checked_macs"] == VR.pass_macs(VR.whole_pass(passes[1])) > got["exact_work"]["checked_macs"] == VR.pass_macs(passes[1])
    cases = [([a, b], wab, (2, 8, 3), {}), ([a, b], wab, (2, 8, 1), {}), ([a, b], wab, (4, 4, 2), {}),
             ([a, b], wab, (2, 4, 2), {}), ([a, b], wab, (1, 16, 2), {}), ([b0, a1], w(b0, a1), cls, {}),
             ([a, wide], w(a, wide), cls, {}),
             ([b], wab, (2, 8, 3), {"rounds": [1], "dropped": (0,), "record": tmp_path / "coin1.json"})]
    for paths, tie, c, kw in cases:
        got = same(exe, paths, tie=tie, load=c, **kw)
        why = f"ValueError: {VR.off_load(c)}"
        assert (got["verdict"], got["reason"], got["refused"], got["tie"], got["commitment"]) == (
            "reject", why, None, None, None), got
    coins = ["--coins", str(a.parent / "coins.json"), "--draws", "3", "--source", SOURCE.hex(), "--test-instance"]
    for argv in (["--class", "2,8,2"], ["--tie", str(wab), "--class", "2,8"], ["--tie", str(wab), "--class", "2,0,2"],
                 ["--tie", str(wab), "--class", "2,8,x"]):
        p = subprocess.run([str(exe), *coins, *argv, str(a), str(b)], capture_output=True, text=True)
        assert p.returncode != 0 and not p.stdout.strip() and "--class" in p.stderr, (argv, p.stderr)


def test_a_pass_that_is_not_ws(exe, tmp_path):
    """W whose entry for a call or a fixed operand has another root (B another fixed operand's: a foreign rootB) or domain, or lacks
    one, or registers a call or fixed operand the run doesn't have, or one twice, or registers no fixed operand at all at version 2
    (every call's entry right: untied, not a tie), or states no version (version 1) and registers a fixed operand, or states a
    version other than 2, or isn't a v2 record: refused before the coin (Python's `ValueError`, in its words), as the
    Python does."""
    a, b = (T.proved_pass(tmp_path / n, H100, s)[0] for n, s in (("a", 0), ("b", 1)))
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


def test_the_version_2_tie_at_the_run_of_records_size(exe, tmp_path, capsys):
    """At the run of record's size (`test_pearl_c_vllm.record_size_passes`: class (4, 32, 4), 4 passes of
    Llama-3.2-1B's 64 cut linears, 256 calls, a served instance's 64 x 64 tiles, each root an hm96 root over fake row
    digests), the Lean's verdict with no sample, held to the producer's version-2 W (`pearl_c.window.record`), is the
    Python's: it accepts tied to W on its 576 tables, 64 of them the weights' B rows. It refuses, in the Python's words
    and before the coin, a W whose ``pearl-b/37`` is at another root than the passes' rootB, one that registers no
    ``pearl-b/37`` and one that registers no weight at all (`UNTIED_WEIGHTS`). With no sample and no excluded row no tile
    is opened, so the passes' files, which aren't written, are never read. Prints the Lean's seconds on the accepted
    run (its load of the four manifests, the tie and its credit), and asserts none."""
    paths = T.record_size_passes(tmp_path)
    good = T.PWR.record([json.loads(p.read_text()) for p in paths], version=2)
    reg = good["registered"]
    assert len(reg) == 576 and [e["value"] for e in reg[512:]] == [f"pearl-b/{x}" for x in range(64)]
    (w := tmp_path / "w.json").write_text(json.dumps(good, indent=1) + "\n")
    want = answer(VR.verify_passes(paths, 0, 1, "test", VR.epoch_coins(paths[0].parent / "coins.json"), SOURCE, window=good))
    got, seconds = T.timed(lambda: lean(exe, paths, 0, tie=w))
    assert got == want
    assert got["verdict"] == "accept" and got["tie"] == {"record": DD.digest(good), "tables": 576} and not got["verdicts"]
    other = next(e["root"] for e in reg if e["value"] == "pearl-b/38")
    cases = [({**good, "registered": [{**e, "root": other} if e["value"] == "pearl-b/37" else e for e in reg]},
              "ValueError: W's pearl-b/37: its root not the weight's"),
             ({**good, "registered": [e for e in reg if e["value"] != "pearl-b/37"]}, "ValueError: W registers no pearl-b/37"),
             ({**good, "registered": reg[:512]},
              f"ValueError: {VR.UNTIED_WEIGHTS}, so the run's rootB is no W's (pearl-b/0 first)")]
    for doc, why in cases:
        w.write_text(json.dumps(doc))
        got = same(exe, paths, 0, tie=w, test_instance=False)
        assert (got["verdict"], got["reason"], got["tie"], got["commitment"]) == ("reject", why, None, None), got
    with capsys.disabled():
        print(f"\nLean pouw-pearlplus at the run of record's size, no draw, tied to W v2 (576 values): {seconds:.2f} s")


@pytest.mark.skipif(not settings.RECORDED, reason="POUW_PEARLPLUS_RECORDED names no recorded run")
def test_a_recorded_h200_run(exe):
    """An H200 run's retained passes, the served run's (accepted): under ``over_cap`` ``uncredited``, the run of record's,
    the Lean's verdict is the Python's on them at the sample count and source the Python recorded, and accepts as the
    recorded verdict did (written before the forming term and the band rule, so its credit isn't today's).  The Python
    replays as ``served_passes.py verify`` does, on the batch forms with every 256th tile on the scalar reference too (the
    scalar replay alone takes hours here).  It is the one case on a served instance's 64 x 64 tiles at k_min 1,024; each
    Lean replay takes about 45 minutes on a CPU node, so ``reject`` is left to the CPU cases."""
    d = Path(settings.RECORDED)
    rec = json.loads((d / "verdicts" / "run.json").read_text())
    paths = [d / "passes" / p / "manifest.json" for p in ("prefill", "decode")]
    coins, source = d / "passes" / "coins.json", bytes.fromhex(rec["source"])
    want = answer(VR.verify_passes(paths, rec["draws"], len(os.sched_getaffinity(0)), "test", VR.epoch_coins(coins), source,
                                   fast=True, cross_check=256, over_cap=VR.UNCREDITED))
    assert want["verdict"] == rec["answer"]["verdict"], want["reason"]
    assert lean(exe, paths, rec["draws"], source, coins=coins, over_cap=VR.UNCREDITED) == want
