"""Pearl+'s run verdict in Lean (`pouw-pearlplus`, `PouwCheck.PearlPlus.check`) is the Python's (`verify_run.verify_passes`).

Each case proves passes on the CPU as verify_run's tests do (`test_pearl_c_vllm.proved_pass`: the H100 record's -h3 scheme
at k = 128 in 16 x 16 tiles, a call with filler rows and an excluded row, y committed), changes them as those tests do, and
holds the Lean's verdict to the Python's on the same files and coin: the verdict, the reason, the refusal, the run's
commitment and source, every opened tile's verdict in order, and an accepted run's work (its credit at Pearl's assumption,
`credit_at_assumption`: the credited MACs, tiles and MACs, ρ, ε and the draws over the inert-work bound) under each
`over_cap` rule, `uncredited` (the default: a tile over its bound is credited nothing and the run goes on) and `reject`.
The passes are a test instance's, read as one (``test_instance``, ``--test-instance``), so the work credits nothing;
read as a served run's they are refused (their tile is not 64 x 64), as passes of two salts are. A layer past `K_MAX` is
refused at load on both sides. `test_a_recorded_h200_run` does the same on an H200 run's retained
passes against the verdicts the Python recorded on them, where `POUW_PEARLPLUS_RECORDED` names a directory holding them
(`passes/` and `verdicts/{run,control}.json`).

Where the Python raises past its own verdict the Lean refuses instead, and no case here reaches it: a debit replay or a
checked U that rejects (the Lean names it), and an opening whose leaf doesn't parse (an opening that fails). The Lean also
holds a y opening to the tile's real rows' positions, which the Python's own opener supplies.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from fractions import Fraction
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
VP = T.load("pearl_c_vllm_volunteer_pass", T.KERNEL.parent / "pearl_c_vllm" / "volunteer_pass.py")
pytestmark = [pytest.mark.slow, pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")]


@pytest.fixture(scope="module")
def exe() -> Path:
    return D.build(PACKAGE, "pouw-pearlplus")


def lean(exe: Path, paths: list[Path], draws: int, source: bytes | None = SOURCE, record: Path | None = None,
         coins: Path | None = None, over_cap: str | None = None, test_instance: bool = False) -> dict:
    argv = [str(exe), "--coins", str(coins or paths[0].parent / "coins.json"), "--draws", str(draws)]
    argv += [] if source is None else ["--source", source.hex()]
    argv += [] if record is None else ["--record", str(record)]
    argv += [] if over_cap is None else ["--over-cap", over_cap]
    argv += ["--test-instance"] if test_instance else []
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
            "work": None if work is None else {"over_cap_draws": 0, **work}}


def same(exe: Path, paths: list[Path], draws: int = 3, source: bytes | None = SOURCE, record: Path | None = None,
         over_cap: str | None = None, test_instance: bool = True) -> dict:
    """The Python's verdict on `paths`, asserted to be the Lean's; the Lean reads the record the Python wrote, if any.
    ``over_cap`` None is each side's default; the passes are a test instance's unless ``test_instance`` is false."""
    rule = {} if over_cap is None else {"over_cap": over_cap}
    want = answer(VR.verify_passes(paths, draws, 1, "test", T.published(paths[0]), source, record=record,
                                   test_instance=test_instance, **rule))
    got = lean(exe, paths, draws, source, record, over_cap=over_cap, test_instance=test_instance)
    assert got == want
    return got


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
    without = answer(VR.verify_passes([path], 16, 1, "test", T.published(path), SOURCE))
    assert without["verdict"] == "accept" and not any(v["reason"] for v in without["verdicts"])


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
    """Read as a served run's, a test instance's pass is refused before the coin: its tile is 16 x 16, not 64 x 64. Passes
    of two salts (other weights under the one coin), each holding calls 0 and 1, are refused, test instance or not."""
    a = T.proved_pass(tmp_path / "a", H100)[0]
    served = same(exe, [a], test_instance=False)
    assert served["verdict"] == "reject" and served["source"] is None and served["verdicts"] == []
    assert served["reason"] == ("ValueError: pass 0's tile is 16 x 16, not 64 x 64: a test instance is read only as one, "
                                "crediting nothing")
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


@pytest.mark.skipif(not RECORDED, reason="POUW_PEARLPLUS_RECORDED names no recorded run")
@pytest.mark.parametrize("name, passes", [("run", ("prefill", "decode")), ("control", ("prefill", "decode", "control"))])
def test_a_recorded_h200_run(exe, name, passes):
    """An H200 run's retained passes, the served run's (accepted) and with its control pass, whose activations aren't the
    ones it committed (rejected): the Lean's verdict is the one the Python recorded, under the rule it recorded."""
    d = Path(RECORDED)
    want = json.loads((d / "verdicts" / f"{name}.json").read_text())
    paths = [d / "passes" / p / "manifest.json" for p in passes]
    got = lean(exe, paths, want["draws"], bytes.fromhex(want["source"]), coins=d / "passes" / "coins.json",
               over_cap=want.get("over_cap", VR.REJECT))
    assert got == answer(want)
