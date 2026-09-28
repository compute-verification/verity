"""The Lean verifier's trust surface: a pinned toolchain, no dependencies, and no escape hatch that would let compiled code
differ from the definitions a proof is about (`native_decide`, `implemented_by`, `extern`, `unsafe`)."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

LEAN = Path(__file__).resolve().parents[1] / "verifier" / "lean"
TOOLCHAIN = "leanprover/lean4:v4.34.0"  # the soundness package's Mathlib and ArkLib pin v4.34.0
FORBIDDEN = re.compile(r"\b(native_decide|implemented_by|extern|unsafe)\b")


def sources():
    """The executable's sources (the proofs' package, `soundness/`, has its own dependencies and checks)."""
    return sorted(p for p in LEAN.rglob("*.lean") if ".lake" not in p.parts and "soundness" not in p.relative_to(LEAN).parts)


def test_toolchain_is_pinned():
    assert (LEAN / "lean-toolchain").read_text().strip() == TOOLCHAIN


def test_no_dependencies():
    assert "require" not in (LEAN / "lakefile.toml").read_text()
    assert json.loads((LEAN / "lake-manifest.json").read_text())["packages"] == []


def test_no_escape_hatches():
    assert sources()
    hits = [f"{p.relative_to(LEAN)}:{i}: {line.strip()}" for p in sources()
            for i, line in enumerate(p.read_text().splitlines(), 1) if FORBIDDEN.search(line)]
    assert not hits, hits


AUDIT = LEAN / "soundness" / "FlockSoundness" / "Audit"
AUDIT_IMPORTS = re.compile(r"import (Mathlib\.|FlockSoundness\.Game\.|FlockSoundness\.Audit\.)")


def test_audit_layer_is_abstract():
    """The audit theorems import only the interaction model, Mathlib and each other, so any backend can instantiate them
    (`soundness/DESIGN.md` §12). The Flock instantiations are the only files that import Flock's theorems:
    `Flock.lean` (oracle layer), `FlockCompiled.lean` (one table), `FlockBatched.lean` (several tables),
    `FlockBatchedAcc.lean` (the expected-time extractor), `FlockLink.lean` (the link theorem), `FlockLinked.lean` (the
    two composed) and `FlockWork.lean` (the executable's work rule). No proof is admitted."""
    files = sorted(AUDIT.glob("*.lean"))
    flock = {"Flock.lean", "FlockCompiled.lean", "FlockBatched.lean", "FlockBatchedAcc.lean", "FlockLink.lean",
             "FlockLinked.lean", "FlockWork.lean"}
    assert files and flock <= {p.name for p in files}
    bad = [f"{p.name}: {line}" for p in files if p.name not in flock for line in p.read_text().splitlines()
           if line.startswith("import ") and not AUDIT_IMPORTS.match(line)]
    assert not bad, bad
    admitted = [p.name for p in files if re.search(r"\b(sorry|admit)\b|^axiom ", p.read_text(), re.M)]
    assert not admitted, admitted


def _has_unit_cut():
    try:
        from verity.ir.partition import validate_unit_cut  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(shutil.which("lake") is None or not _has_unit_cut(), reason="no Lean toolchain or no validate_unit_cut")
def test_unit_cut_agrees_with_the_reference(tmp_path):
    subprocess.run(["lake", "build"], cwd=LEAN, check=True, capture_output=True)
    p = subprocess.run(["python3", str(LEAN.parent / "unit_cut_agree.py"), "--lean", str(LEAN / ".lake/build/bin/flock-verify"),
                        "--out", str(tmp_path), "--n", "300"], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    assert json.loads(p.stdout.strip().splitlines()[-1])["v2"]["refused_by_v1_alone"] > 0       # Q_word v2's rule is exercised


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_circuit_type_agrees_with_the_reference(tmp_path):
    """`flock-verify circuit-type` gives `verity_flock.circuit_types`' digest and verdict on every type (random types, call
    trees and one mutation per rule; `circuit_type_agree.py`)."""
    subprocess.run(["lake", "build"], cwd=LEAN, check=True, capture_output=True)
    p = subprocess.run(["python3", str(LEAN.parent / "circuit_type_agree.py"), "--lean", str(LEAN / ".lake/build/bin/flock-verify"),
                        "--out", str(tmp_path), "--n", "200"], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_layout_agrees_with_the_reference(tmp_path):
    """`flock-verify layout` gives `verity_flock.layouts`' verdict, layout digests and own-region sizes on every archive
    (random types and call trees, placed and inline calls and reads, nested units, one mutation per rule;
    `layout_agree.py`)."""
    subprocess.run(["lake", "build"], cwd=LEAN, check=True, capture_output=True)
    p = subprocess.run(["python3", str(LEAN.parent / "layout_agree.py"), "--lean", str(LEAN / ".lake/build/bin/flock-verify"),
                        "--out", str(tmp_path), "--n", "150"], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr


def _stream_uniform(n, stream, pos):
    """`verity.randomness`'s exact reduction on a raw stream: (value, next position), or None when it runs out."""
    from verity.randomness import uniform_from_words
    w = (n.bit_length() + 64 + 7) // 8
    at = [pos]

    def words():
        while at[0] + w <= len(stream):
            x = int.from_bytes(stream[at[0]:at[0] + w], "big")
            at[0] += w
            yield x
    try:
        return uniform_from_words(n, words(), 8 * w), at[0]
    except AssertionError:
        return None


def _ref_subset(k, n, stream):
    swapped, pos = {}, 0
    for i in range(k):
        r = _stream_uniform(n - i, stream, pos)
        if r is None:
            return None
        j, pos = i + r[0], r[1]
        swapped[i], swapped[j] = swapped.get(j, j), swapped.get(i, i)
    return sorted(swapped.get(i, i) for i in range(k))


def _ref_bernoulli(a, b, n, stream):
    if a == 0:
        return []
    if a == b:
        return list(range(n))
    out, pos = [], 0
    for u in range(n):
        r = _stream_uniform(b, stream, pos)
        if r is None:
            return None
        if r[0] < a:
            out.append(u)
        pos = r[1]
    return out


def _lean_draw_obj(args):
    p = subprocess.run([str(LEAN / ".lake/build/bin/flock-verify"), "draw", *args], capture_output=True, text=True)
    return json.loads(p.stdout[5:]) if p.stdout.startswith("DRAW ") else None


def _lean_draw(args):
    p = subprocess.run([str(LEAN / ".lake/build/bin/flock-verify"), "draw", *args], capture_output=True, text=True)
    return json.loads(p.stdout[5:])["units"] if p.stdout.startswith("DRAW ") else None


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_unit_draw_samplers_are_the_exact_reduction():
    import random
    subprocess.run(["lake", "build"], cwd=LEAN, check=True, capture_output=True)
    rng = random.Random(20260927)
    for case in range(40):
        n = rng.choice([1, 2, 3, 7, 100, 1000, 2 ** 20 + 3, 2 ** 40 - 1])
        k = rng.randint(0, min(n, 12))
        stream = bytearray(rng.randbytes(rng.choice([8, 64, 400])))
        if case % 4 == 0 and len(stream) >= 16:
            stream[:16] = b"\xff" * 16  # a first word above the largest multiple of n: rejected
        want = _ref_subset(k, n, bytes(stream))
        assert _lean_draw(["--population", str(n), "--k", str(k), "--stream", stream.hex()]) == want, (n, k, case)
        a, b = sorted(rng.sample(range(0, 9), 2)) if case % 5 else (0, 3)
        b = max(b, 1)
        want = _ref_bernoulli(a, b, n % 50 + 1, bytes(stream))
        assert _lean_draw(["--population", str(n % 50 + 1), "--p", f"{a}/{b}", "--stream", stream.hex()]) == want, (a, b, case)
    units = _lean_draw(["--population", "1000", "--k", "10"])
    assert len(set(units)) == 10 and units == sorted(units) and all(0 <= u < 1000 for u in units)


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_unit_draw_check(tmp_path):
    subprocess.run(["lake", "build"], cwd=LEAN, check=True, capture_output=True)
    d = {"law": "subset", "population": 100, "k": 3, "units": [4, 17, 90]}
    bern = {"law": "bernoulli", "population": 10, "p": "1/2", "units": [0, 3, 9]}
    cases = [
        ({"record": None, "statement": None, "instances": 5}, ""),
        ({"record": d, "statement": d, "instances": 3}, ""),
        ({"record": bern, "statement": bern, "instances": 3}, ""),
        ({"record": d, "statement": None, "instances": 3}, "U1: the record keeps"),
        ({"record": None, "statement": d, "instances": 3}, "U1: the statement carries"),
        ({"record": d, "statement": {**d, "units": [4, 17, 91]}, "instances": 3}, "U1: the record's unit draw is not"),
        ({"record": d, "statement": d, "instances": 4}, "U3:"),
        ({"record": {**d, "units": [17, 4, 90]}, "statement": {**d, "units": [17, 4, 90]}, "instances": 3}, "U2: the drawn units"),
        ({"record": {**d, "units": [4, 17, 100]}, "statement": {**d, "units": [4, 17, 100]}, "instances": 3}, "U2: the drawn units"),
        ({"record": {**d, "k": 4}, "statement": {**d, "k": 4}, "instances": 3}, "U2: a subset draw of the wrong size"),
        ({"record": {**d, "seed": "x"}, "statement": {**d, "seed": "x"}, "instances": 3}, "U2: unit_draw's keys"),
        ({"record": {**bern, "p": "3/2"}, "statement": {**bern, "p": "3/2"}, "instances": 3}, "U2: unit_draw.p is not a probability"),
        ({"record": {**bern, "p": "0/2"}, "statement": {**bern, "p": "0/2"}, "instances": 3}, "U2: a p = 0 draw"),
        ({"record": {**d, "law": "beacon"}, "statement": {**d, "law": "beacon"}, "instances": 3}, "U2: unit_draw.law"),
    ]
    f = tmp_path / "cases.json"
    f.write_text(json.dumps([c for c, _ in cases]))
    p = subprocess.run([str(LEAN / ".lake/build/bin/flock-verify"), "draw-test", str(f)], capture_output=True, text=True, check=True)
    got = [json.loads(line[10:]) for line in p.stdout.splitlines() if line.startswith("DRAWCHECK ")]
    assert len(got) == len(cases)
    for (case, want), g in zip(cases, got):
        assert g["ok"] == (want == "") and g["why"].startswith(want), (case, g)


def _template_vectors():
    try:
        import verity
    except ImportError:
        return None
    v = Path(verity.__file__).resolve().parents[2] / "tests" / "ir" / "template_instance_vectors.json"
    return v if v.exists() else None



def _stratified_agree():
    import importlib.util
    spec = importlib.util.spec_from_file_location("stratified_agree", LEAN.parent / "stratified_agree.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_stratified_draw_from_the_verifiers_own_program(tmp_path):
    """`draw --program --partition --stratified K`: the strata derived from the program (one per template, ranges merged,
    in order of first unit), `k_s = max(1, round(K n_s / N))`, each stratum's `subset` on its own segment of the stream,
    and U2 on stratified draws, held to the verifier's own law."""
    import hashlib
    subprocess.run(["lake", "build"], cwd=LEAN, check=True, capture_output=True)
    ref = _stratified_agree()
    desc = {"schema": "verity-ir/descriptor/v1", "callees": ["Input8_v1", "T0_v0", "T1_v0"], "root": "Root_v0",
            "definitions": {"Root_v0": {"form": "composite", "body": {"nodes": [
                {"form": "primitive", "fn": 0}, {"form": "batch", "fn": 1, "n": 2}, {"form": "batch", "fn": 2, "n": 3},
                {"form": "batch", "fn": 1, "n": 1}]}},
                "Input8_v1": {"form": "primitive"}, "T0_v0": {"form": "primitive"}, "T1_v0": {"form": "primitive"}}}
    text = json.dumps(desc, sort_keys=True, separators=(",", ":"))
    (tmp_path / "p.desc").write_text(text)
    obj = {"format": "verity/partition/v1", "program": hashlib.sha512(text.encode()).hexdigest(),
           "query": {"name": "Q_template_instances", "version": 0, "params": {"templates": ["T1_v0", "T0_v0"]}}}
    (tmp_path / "q.json").write_text(json.dumps(obj))
    flags = ["--program", str(tmp_path / "p.desc"), "--partition", str(tmp_path / "q.json")]
    law = {"law": "stratified", "k": 4, "strata": [{"template": "T0_v0", "units": [[0, 2], [5, 6]], "k": 2},
                                                   {"template": "T1_v0", "units": [[2, 5]], "k": 2}]}
    import random
    rng = random.Random(20260927)
    for _ in range(20):
        src = rng.randbytes(2 * 2 * 9 + 64 + 2 * 2 * 9 + 64)
        (tmp_path / "s.bin").write_bytes(src)
        d = _lean_draw_obj([*flags, "--stratified", "4", "--stream-file", str(tmp_path / "s.bin")])
        assert {k: d[k] for k in ("law", "k", "strata")} == law and d["population"] == 6
        assert d["units"] == ref.reference_draw(law, src)
    d = _lean_draw_obj([*flags, "--stratified", "4"])
    assert len(d["units"]) == 4 and sum(u in (0, 1, 5) for u in d["units"]) == 2
    assert [s["k"] for s in _lean_draw_obj([*flags, "--stratified", "1"])["strata"]] == [1, 1]
    bad = [{**d, "k": 5}, {**d, "strata": d["strata"][::-1]}, {**d, "units": d["units"][:-1]},
           {**d, "strata": [{**d["strata"][0], "units": [[0, 2], [2, 2], [5, 6]]}, d["strata"][1]]},
           {**d, "strata": [{**d["strata"][0], "units": [[0, 1], [1, 2], [5, 6]]}, d["strata"][1]]},
           {**d, "strata": [{**d["strata"][0], "k": True}, d["strata"][1]]}]
    (tmp_path / "cases.json").write_text(json.dumps([{"record": x, "statement": x, "instances": len(x["units"])} for x in [d, *bad]]))
    p = subprocess.run([str(LEAN / ".lake/build/bin/flock-verify"), "draw-test", str(tmp_path / "cases.json"), *flags,
                        "--stratified", "4"], capture_output=True, text=True)
    got = [json.loads(line[10:]) for line in p.stdout.splitlines()]
    assert [g["ok"] for g in got] == [True] + [False] * len(bad), got
    for k in ("0", "7"):
        assert _lean_draw_obj([*flags, "--stratified", k]) is None
    (tmp_path / "w.json").write_text(json.dumps({**obj, "query": {"name": "Q_word", "version": 1, "params": {"X": 16, "W": 32}}}))
    assert _lean_draw_obj(["--program", str(tmp_path / "p.desc"), "--partition", str(tmp_path / "w.json"), "--stratified", "4"]) is None


def _work_program(tmp_path):
    """Three templates' units under `Q_template_instances` (T0 at [0, 2) and [5, 6), T1 at [2, 5), T2 at [6, 10)), and a
    writer of work tables that returns the verifier's flags for it."""
    import hashlib
    desc = {"schema": "verity-ir/descriptor/v1", "callees": ["Input8_v1", "T0_v0", "T1_v0", "T2_v0"], "root": "Root_v0",
            "definitions": {"Root_v0": {"form": "composite", "body": {"nodes": [
                {"form": "primitive", "fn": 0}, {"form": "batch", "fn": 1, "n": 2}, {"form": "batch", "fn": 2, "n": 3},
                {"form": "batch", "fn": 1, "n": 1}, {"form": "batch", "fn": 3, "n": 4}]}},
                "Input8_v1": {"form": "primitive"}, "T0_v0": {"form": "primitive"}, "T1_v0": {"form": "primitive"},
                "T2_v0": {"form": "primitive"}}}
    text = json.dumps(desc, sort_keys=True, separators=(",", ":"))
    (tmp_path / "p.desc").write_text(text)
    obj = {"format": "verity/partition/v1", "program": hashlib.sha512(text.encode()).hexdigest(),
           "query": {"name": "Q_template_instances", "version": 0, "params": {"templates": ["T1_v0", "T0_v0", "T2_v0"]}}}
    (tmp_path / "q.json").write_text(json.dumps(obj))

    def table(name, work):
        (tmp_path / name).write_text(json.dumps(work))
        return ["--program", str(tmp_path / "p.desc"), "--partition", str(tmp_path / "q.json"), "--work-table", str(tmp_path / name)]

    return table


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_work_draw_from_the_verifiers_own_program(tmp_path):
    """`draw --program --partition --work K --work-table F`: the stratified law's strata, `k_s = min(n_s, max(f_s,
    ⌈K w_s n_s / W⌉))` with each stratum's floor `f_s` from the table (one by default, so a stratum that does no work draws
    one), each stratum's `subset` on its own segment of the stream, and U2 on work draws, held to the verifier's own law."""
    import random
    subprocess.run(["lake", "build"], cwd=LEAN, check=True, capture_output=True)
    ref = _stratified_agree()
    table = _work_program(tmp_path)
    flags = table("w.json", {"T0_v0": 3, "T1_v0": 1, "T2_v0": 0})
    law = {"law": "work", "k": 2, "strata": [{"template": "T0_v0", "units": [[0, 2], [5, 6]], "work": 3, "floor": 1, "k": 2},
                                             {"template": "T1_v0", "units": [[2, 5]], "work": 1, "floor": 1, "k": 1},
                                             {"template": "T2_v0", "units": [[6, 10]], "work": 0, "floor": 1, "k": 1}]}
    need = sum(2 * s["k"] * ref.width(sum(hi - lo for lo, hi in s["units"])) + 64 for s in law["strata"])
    rng = random.Random(20260929)
    for _ in range(20):
        src = rng.randbytes(need)
        (tmp_path / "s.bin").write_bytes(src)
        d = _lean_draw_obj([*flags, "--work", "2", "--stream-file", str(tmp_path / "s.bin")])
        assert {k: d[k] for k in ("law", "k", "strata")} == law and d["population"] == 10
        assert d["units"] == ref.reference_draw(law, src)
    # W = 3·3 + 1·3 = 12: sized by work, floored at one, capped at n_s
    for k, want in {"1": [1, 1, 1], "4": [3, 1, 1], "10": [3, 3, 1]}.items():
        assert [s["k"] for s in _lean_draw_obj([*flags, "--work", k])["strata"]] == want, k
    d = _lean_draw_obj([*flags, "--work", "2"])
    assert len(d["units"]) == 4 and sum(u in (0, 1, 5) for u in d["units"]) == 2 and sum(u >= 6 for u in d["units"]) == 1
    s0, s1, s2 = d["strata"]
    bad = [{**d, "k": 3}, {**d, "law": "stratified"}, {**d, "units": d["units"][:-1]},
           {**d, "strata": [s0, s1, {**s2, "k": 0}]}, {**d, "strata": [s0, s1, {k: v for k, v in s2.items() if k != "work"}]},
           {**d, "strata": [{**s0, "work": "3"}, s1, s2]}, {**d, "strata": [s0, s1, {**s2, "k": True}]},
           {**d, "strata": [s0, s1, {**s2, "work": 0, "n": 4}]}, {**d, "strata": [s0, s1]},
           # U2 alone accepts a work of 4 (the same k_s); the verifier's own table refuses it
           {**d, "strata": [{**s0, "work": 4}, s1, s2]}]

    def check(cases, *extra):
        (tmp_path / "cases.json").write_text(json.dumps([{"record": x, "statement": x, "instances": len(x["units"])} for x in cases]))
        p = subprocess.run([str(LEAN / ".lake/build/bin/flock-verify"), "draw-test", str(tmp_path / "cases.json"), *extra],
                           capture_output=True, text=True)
        return [json.loads(line[10:]) for line in p.stdout.splitlines()]

    got = check([d, *bad], *flags, "--work", "2")
    assert [g["ok"] for g in got] == [True] + [False] * len(bad), got
    assert [g["ok"] for g in check([d, bad[-1]])] == [True, True]
    # another table's law, or the stratified law, is not this verifier's
    other = _lean_draw_obj([*table("w2.json", {"T0_v0": 1, "T1_v0": 1, "T2_v0": 0}), "--work", "2"])
    strat = _lean_draw_obj([*flags[:4], "--stratified", "2"])
    assert [g["ok"] for g in check([other, strat])] == [True, True]
    assert [g["ok"] for g in check([other, strat], *flags, "--work", "2")] == [False, False]
    # floors per stratum from the table: T2 (no work, 4 of 10 units) at floor 2 draws what stratified:4 draws from it,
    # at floor 9 all four; a floor on a working stratum lifts it too
    cases = [({"T0_v0": 3, "T1_v0": 1, "T2_v0": {"work": 0, "floor": 2}}, [1, 1, 2], [2, 1, 2]),
             ({"T0_v0": 3, "T1_v0": 1, "T2_v0": {"work": 0, "floor": 9}}, [1, 1, 9], [2, 1, 4]),
             ({"T0_v0": 3, "T1_v0": {"work": 1, "floor": 3}, "T2_v0": 0}, [1, 3, 1], [2, 3, 1])]
    for i, (work, floors, want) in enumerate(cases):
        df = _lean_draw_obj([*table(f"wf{i}.json", work), "--work", "2"])
        assert [s["floor"] for s in df["strata"]] == floors and [s["k"] for s in df["strata"]] == want, work
    assert _lean_draw_obj([*flags[:4], "--stratified", "4"])["strata"][2]["k"] == 2
    ff = table("wf.json", cases[0][0])
    df = _lean_draw_obj([*ff, "--work", "2"])
    f0, f1, f2 = df["strata"]
    # U2 refuses a floor of 0, a missing floor and a count key; floor 1 with k 1 is the rule's, and only the verifier's own
    # table refuses it
    fbad = [{**df, "strata": [f0, f1, {**f2, "floor": 0}]}, {**df, "strata": [f0, f1, {k: v for k, v in f2.items() if k != "floor"}]},
            {**df, "count": 4}, {**df, "strata": [f0, f1, {**f2, "floor": 1, "k": 1}], "units": [u for u in df["units"] if u < 6] + [6]}]
    assert [g["ok"] for g in check([df, *fbad], *ff, "--work", "2")] == [True] + [False] * 4
    assert [g["ok"] for g in check([df, fbad[3]])] == [True, True]
    assert [g["ok"] for g in check([df, d], *flags, "--work", "2")] == [False, True]
    assert _lean_draw_obj([*table("wf0z.json", {"T0_v0": 3, "T1_v0": 1, "T2_v0": {"work": 0, "floor": 0}}), "--work", "2"]) is None
    assert _lean_draw_obj([*table("wfx.json", {"T0_v0": {"work": 3, "floor": 1, "x": 1}, "T1_v0": 1, "T2_v0": 0}), "--work", "2"]) is None
    for k in ("0", "11"):
        assert _lean_draw_obj([*flags, "--work", k]) is None
    assert _lean_draw_obj([*table("w3.json", {"T0_v0": 3, "T1_v0": 1}), "--work", "2"]) is None
    assert _lean_draw_obj([*table("w4.json", {"T0_v0": 0, "T1_v0": 0, "T2_v0": 0}), "--work", "2"]) is None
    assert _lean_draw_obj([*flags, "--work", "2", "--stratified", "2"]) is None
    assert _lean_draw_obj([*flags[:4], "--work", "2"]) is None


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_verify_takes_a_work_draw_only_from_its_own_table(tmp_path):
    """`verify` refuses a session whose record carries a work draw unless it holds its own K, work table, program and
    partition, since the draw's K, works and floors set every k_s; with all four, the law must be its own derivation, so a
    draw made at another K (k = 1, say, which U2 alone accepts) or under other floors is refused."""
    subprocess.run(["lake", "build"], cwd=LEAN, check=True, capture_output=True)
    table = _work_program(tmp_path)
    flags = table("w.json", {"T0_v0": 3, "T1_v0": 1, "T2_v0": 0})
    other = table("w2.json", {"T0_v0": 1, "T1_v0": 1, "T2_v0": 0})
    (tmp_path / "circuit.txt").write_text("not a circuit\n")
    (tmp_path / "pub.bin").write_bytes(b"{}\n")

    def session(name, draw):
        s = tmp_path / name
        s.mkdir()
        (s / "session.json").write_text(json.dumps({"kind": "flock-live-session/v1", "unit_draw": draw}))
        for r in (0, 1):
            (s / f"circuit.rep{r}.bin").write_bytes(b"\0" * 16)
        return s

    at2 = session("k2", _lean_draw_obj([*flags, "--work", "2"]))
    at1 = session("k1", _lean_draw_obj([*flags, "--work", "1"]))

    def why(s, *extra):
        p = subprocess.run([str(LEAN / ".lake/build/bin/flock-verify"), "verify", "--circuit", str(tmp_path / "circuit.txt"),
                            "--public", str(tmp_path / "pub.bin"), *extra, "--session", str(s)],
                           capture_output=True, text=True)
        (v,) = [json.loads(line[8:]) for line in p.stdout.splitlines() if line.startswith("VERDICT ")]
        assert not v["accepted"] and p.returncode == 1
        return v["why"]

    need = "setup: U2: a work draw needs the verifier's own K, work table, program and partition"
    assert why(at2).startswith(need)
    assert why(at2, *flags[:4], "--work", "2").startswith(need)
    assert why(at2, *flags[4:], "--work", "2").startswith(need)
    assert why(at2, *flags).startswith(need)
    assert why(at2, *other, "--work", "2").startswith("setup: U2: the draw's work law is not the verifier's own")
    assert why(at2, *flags, "--work", "3").startswith("setup: U2: the draw's K is 2, not the verifier's 3")
    assert why(at1, *flags, "--work", "2").startswith("setup: U2: the draw's K is 1, not the verifier's 2")
    # at its own K and table the law is its own derivation, and setup goes on to the circuit, which this file is not
    assert not why(at2, *flags, "--work", "2").startswith("setup: U2:")
    assert not why(at1, *flags, "--work", "1").startswith("setup: U2:")
    # the floors are the verifier's too: a draw made under a table with T2's floor at 3 is refused under the verifier's
    # table (floor 1), and one made under floor 1 is refused under a table with floor 3
    floored = table("wf.json", {"T0_v0": 3, "T1_v0": 1, "T2_v0": {"work": 0, "floor": 3}})
    at2f = session("k2f", _lean_draw_obj([*floored, "--work", "2"]))
    assert why(at2f, *flags, "--work", "2").startswith("setup: U2: the draw's work law is not the verifier's own")
    assert why(at2, *floored, "--work", "2").startswith("setup: U2: the draw's work law is not the verifier's own")
    assert not why(at2f, *floored, "--work", "2").startswith("setup: U2:")


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_verify_takes_a_stratified_draw_only_at_its_own_k(tmp_path):
    """`verify` refuses a session whose record carries a stratified draw unless it holds its own K (`--stratified K`),
    program and partition, since the draw's K and strata set every k_s (U2 alone accepts a draw at k = 1, and strata it
    cannot check); with all three, the law must be its own derivation, so a draw at another K or over other strata is
    refused."""
    subprocess.run(["lake", "build"], cwd=LEAN, check=True, capture_output=True)
    flags = _work_program(tmp_path)("w.json", {})[:4]
    (tmp_path / "circuit.txt").write_text("not a circuit\n")
    (tmp_path / "pub.bin").write_bytes(b"{}\n")

    def session(name, draw):
        s = tmp_path / name
        s.mkdir()
        (s / "session.json").write_text(json.dumps({"kind": "flock-live-session/v1", "unit_draw": draw}))
        for r in (0, 1):
            (s / f"circuit.rep{r}.bin").write_bytes(b"\0" * 16)
        return s

    d4 = _lean_draw_obj([*flags, "--stratified", "4"])
    at4 = session("k4", d4)
    at1 = session("k1", _lean_draw_obj([*flags, "--stratified", "1"]))
    # T0's and T1's names swapped (both 3 units at k 1): well formed, and not the verifier's strata
    s0, s1, s2 = d4["strata"]
    swapped = session("swapped", {**d4, "strata": [{**s0, "template": s1["template"]}, {**s1, "template": s0["template"]}, s2]})

    def why(s, *extra):
        p = subprocess.run([str(LEAN / ".lake/build/bin/flock-verify"), "verify", "--circuit", str(tmp_path / "circuit.txt"),
                            "--public", str(tmp_path / "pub.bin"), *extra, "--session", str(s)],
                           capture_output=True, text=True)
        (v,) = [json.loads(line[8:]) for line in p.stdout.splitlines() if line.startswith("VERDICT ")]
        assert not v["accepted"] and p.returncode == 1
        return v["why"]

    need = "setup: U2: a stratified draw needs the verifier's own K, program and partition"
    assert why(at4).startswith(need)
    assert why(at4, *flags).startswith(need)
    assert why(at4, "--stratified", "4").startswith(need)
    assert why(at4, *flags, "--stratified", "1").startswith("setup: U2: the draw's K is 4, not the verifier's 1")
    assert why(at1, *flags, "--stratified", "4").startswith("setup: U2: the draw's K is 1, not the verifier's 4")
    assert why(swapped, *flags, "--stratified", "4").startswith("setup: U2: the draw's strata are not the verifier's own")
    # at its own K, program and partition, setup goes on to the circuit, which this file is not
    assert not why(at4, *flags, "--stratified", "4").startswith("setup: U2:")
    assert not why(at1, *flags, "--stratified", "1").startswith("setup: U2:")


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_closure_draw_proves_the_verifiers_closure(tmp_path):
    """`draw --work K --work-table F --closure C`: the work draw, unchanged, widened by the verifier's closure map (each
    drawn unit's closure, less the drawn units). U2 on the closure, U3 counting the drawn units and their closure, and
    `verify` holding the closure to the verifier's own map."""
    import random
    subprocess.run(["lake", "build"], cwd=LEAN, check=True, capture_output=True)
    ref = _stratified_agree()
    table = _work_program(tmp_path)
    flags = table("w.json", {"T0_v0": 3, "T1_v0": 1, "T2_v0": 0})
    cmap = {0: [6], 1: [6, 7], 5: [8], 2: [9], 3: [9], 4: [7]}
    (tmp_path / "cl.json").write_text(json.dumps({str(u): cs for u, cs in cmap.items()}))
    (tmp_path / "cl2.json").write_text(json.dumps({}))
    cl = ["--closure", str(tmp_path / "cl.json")]

    def closure_of(units):
        return sorted({c for u in units for c in cmap.get(u, [])} - set(units))

    rng = random.Random(20260929)
    for _ in range(10):
        plain = _lean_draw_obj([*flags, "--work", "2"])
        law = {k: plain[k] for k in ("law", "k", "strata")}
        need = sum(2 * s["k"] * ref.width(sum(hi - lo for lo, hi in s["units"])) + 64 for s in law["strata"])
        src = rng.randbytes(need)
        (tmp_path / "s.bin").write_bytes(src)
        d = _lean_draw_obj([*flags, "--work", "2", *cl, "--stream-file", str(tmp_path / "s.bin")])
        assert d["units"] == ref.reference_draw(law, src)
        assert d.get("closure", []) == closure_of(d["units"])
    d = _lean_draw_obj([*flags, "--work", "2", *cl])
    while not d.get("closure"):
        d = _lean_draw_obj([*flags, "--work", "2", *cl])
    n = len(d["units"]) + len(d["closure"])

    def check(cases, *extra):
        (tmp_path / "cases.json").write_text(json.dumps([{"record": x, "statement": x, "instances": i} for x, i in cases]))
        p = subprocess.run([str(LEAN / ".lake/build/bin/flock-verify"), "draw-test", str(tmp_path / "cases.json"), *extra],
                           capture_output=True, text=True)
        return [json.loads(line[10:]) for line in p.stdout.splitlines()]

    # U3 counts the drawn units and their closure
    got = check([(d, n), (d, len(d["units"]))])
    assert [g["ok"] for g in got] == [True, False] and got[1]["why"].startswith("U3:"), got
    c = d["closure"]
    bad = [{**d, "closure": []}, {**d, "closure": c[::-1] if len(c) > 1 else c + [c[0]]},
           {**d, "closure": sorted(c + [d["units"][0]])}, {**d, "closure": c + [10]}, {**d, "closure": ["7"]}]
    assert [g["ok"] for g in check([(x, len(x["units"]) + len(x["closure"])) for x in bad])] == [False] * len(bad)
    strat = _lean_draw_obj([*flags[:4], "--stratified", "2"])
    assert not check([({**strat, "closure": [9]}, len(strat["units"]) + 1)])[0]["ok"]
    # the verifier's own map: its closure is accepted, a smaller one refused, and a plain draw lacks its closure
    smaller = {**d, "closure": c[:-1]} if len(c) > 1 else {k: v for k, v in d.items() if k != "closure"}
    assert [g["ok"] for g in check([(d, n), (smaller, n - 1)], *flags, "--work", "2", *cl)] == [True, False]
    (tmp_path / "circuit.txt").write_text("not a circuit\n")
    (tmp_path / "pub.bin").write_bytes(b"{}\n")

    def session(name, draw):
        s = tmp_path / name
        s.mkdir()
        (s / "session.json").write_text(json.dumps({"kind": "flock-live-session/v1", "unit_draw": draw}))
        for r in (0, 1):
            (s / f"circuit.rep{r}.bin").write_bytes(b"\0" * 16)
        return s

    def why(s, *extra):
        p = subprocess.run([str(LEAN / ".lake/build/bin/flock-verify"), "verify", "--circuit", str(tmp_path / "circuit.txt"),
                            "--public", str(tmp_path / "pub.bin"), *flags, "--work", "2", *extra, "--session", str(s)],
                           capture_output=True, text=True)
        (v,) = [json.loads(line[8:]) for line in p.stdout.splitlines() if line.startswith("VERDICT ")]
        assert not v["accepted"] and p.returncode == 1
        return v["why"]

    widened = session("widened", d)
    assert why(widened).startswith("setup: U3: a closure draw needs the verifier's own closure map (--closure)")
    assert why(widened, "--closure", str(tmp_path / "cl2.json")).startswith("setup: U3: the draw's closure is not the verifier's own")
    assert not why(widened, *cl).startswith(("setup: U2:", "setup: U3:"))
    plain = session("plain", {k: v for k, v in d.items() if k != "closure"})
    assert why(plain, *cl).startswith("setup: U3: the draw's closure is not the verifier's own")
    assert not why(plain).startswith(("setup: U2:", "setup: U3:"))


def _one_stage_draw():
    try:
        from verity_sampled_proofs.one_stage import draw
    except ImportError:
        return None
    return draw if hasattr(draw, "derive") else None


@pytest.mark.skipif(shutil.which("lake") is None or _one_stage_draw() is None,
                    reason="no Lean toolchain or no stratified law in core (PRs #161, #164)")
def test_stratified_draw_agrees_with_core(tmp_path):
    subprocess.run(["lake", "build"], cwd=LEAN, check=True, capture_output=True)
    p = subprocess.run(["python3", str(LEAN.parent / "stratified_agree.py"), "--lean", str(LEAN / ".lake/build/bin/flock-verify"),
                        "--out", str(tmp_path), "--random", "20", "--os-draws", "100"], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr

@pytest.mark.skipif(shutil.which("lake") is None or _template_vectors() is None, reason="no Lean toolchain or no template vectors (PR #131)")
def test_template_queries_agree_with_the_vectors(tmp_path):
    subprocess.run(["lake", "build"], cwd=LEAN, check=True, capture_output=True)
    p = subprocess.run(["python3", str(LEAN.parent / "template_query_agree.py"), "--lean", str(LEAN / ".lake/build/bin/flock-verify"),
                        "--vectors", str(_template_vectors()), "--out", str(tmp_path)], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr


def _has_check_cut():
    try:
        from verity.ir.cut import check_cut  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(shutil.which("lake") is None or not _has_check_cut(), reason="no Lean toolchain or no verity.ir.cut (PR #111)")
def test_check_cut_agrees_with_the_reference(tmp_path):
    subprocess.run(["lake", "build"], cwd=LEAN, check=True, capture_output=True)
    p = subprocess.run(["python3", str(LEAN.parent / "cut_check_agree.py"), "--lean", str(LEAN / ".lake/build/bin/flock-verify"),
                        "--out", str(tmp_path), "--n", "300"], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    assert json.loads(p.stdout.strip().splitlines()[-1])["v2"]["refused_by_v1_alone"] > 0       # Q_word v2's rule is exercised


def _qword_vector_file():
    try:
        import verity
        from verity.ir.codec import decode_program  # noqa: F401
    except ImportError:
        return None
    v = Path(verity.__file__).resolve().parents[2] / "tests" / "ir" / "qword_vectors.json"
    return v if v.exists() else None


@pytest.mark.skipif(shutil.which("lake") is None or _qword_vector_file() is None,
                    reason="no Lean toolchain or no qword_vectors.json in core (PR #111)")
def test_qword_cuts_agree_with_core_verify(tmp_path):
    """Every Call's cut on its whole graph (`qword-program --cuts`: the partition invariant and the width rule), against core's
    `partition_object.verify`: its pinned cut refusals, and each of its programs at several (X, W), under `Q_word` v1 and v2
    (v2's evaluations, refusals, the program it accepts that v1 refuses, and the recompute it reports)."""
    subprocess.run(["lake", "build"], cwd=LEAN, check=True, capture_output=True)
    p = subprocess.run(["python3", str(LEAN.parent / "qword_program_agree.py"), "--lean", str(LEAN / ".lake/build/bin/flock-verify"),
                        "--out", str(tmp_path), "--qword-vectors", str(_qword_vector_file())], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    r = json.loads(p.stdout.strip().splitlines()[-1])["qword_vectors"]
    assert r["cut_refusals"] == 3 and r["cuts"] == 100, r
    assert r["v2"] == {"evaluations": 9, "cut": 2, "accepted": 1, "inapplicable": 2}, r


def _qword_vectors():
    try:
        import verity.ir.partition_object  # noqa: F401
        from verity.ir.cut import cut_word  # noqa: F401
    except ImportError:
        return None
    import verity
    v = Path(verity.__file__).resolve().parents[2] / "tests" / "ir" / "partition_vectors.json"
    return v if v.exists() else None


@pytest.mark.skipif(shutil.which("lake") is None or _qword_vectors() is None, reason="no Lean toolchain or no Q_word evaluator (PR #111)")
def test_qword_evaluator_agrees_with_the_reference(tmp_path):
    subprocess.run(["lake", "build"], cwd=LEAN, check=True, capture_output=True)
    p = subprocess.run(["python3", str(LEAN.parent / "qword_agree.py"), "--lean", str(LEAN / ".lake/build/bin/flock-verify"),
                        "--out", str(tmp_path), "--n", "300", "--vectors", str(_qword_vectors())], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr


def _format_vectors():
    try:
        import verity
    except ImportError:
        return None
    ir = Path(verity.__file__).resolve().parents[2] / "tests" / "ir"
    need = [ir / "format_vectors.json", ir / "partition_vectors.json", ir / "test_format_spec.py"]
    return ir if all(p.exists() for p in need) else None


@pytest.mark.skipif(shutil.which("lake") is None or _format_vectors() is None,
                    reason="no Lean toolchain or no program format vectors (PR #120)")
def test_qword_over_programs_agrees_with_the_reference(tmp_path):
    ir = _format_vectors()
    subprocess.run(["lake", "build"], cwd=LEAN, check=True, capture_output=True)
    p = subprocess.run(["python3", str(LEAN.parent / "qword_program_agree.py"), "--lean", str(LEAN / ".lake/build/bin/flock-verify"),
                        "--out", str(tmp_path), "--vectors", str(ir / "partition_vectors.json"),
                        "--format", str(ir / "format_vectors.json")]
                       + (["--qword-vectors", str(ir / "qword_vectors.json")] if (ir / "qword_vectors.json").exists() else []),
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr


@pytest.mark.skipif(shutil.which("lake") is None or _format_vectors() is None
                    or not (_format_vectors() / "qcall_vectors.json").exists(), reason="no Lean toolchain or no Q_call vectors")
def test_qcall_and_input_bindings_agree_with_the_reference(tmp_path):
    ir = _format_vectors()
    subprocess.run(["lake", "build"], cwd=LEAN, check=True, capture_output=True)
    p = subprocess.run(["python3", str(LEAN.parent / "qcall_agree.py"), "--lean", str(LEAN / ".lake/build/bin/flock-verify"),
                        "--out", str(tmp_path), "--qcall", str(ir / "qcall_vectors.json"), "--format", str(ir / "format_vectors.json")],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_merkle_collision_is_not_met_by_a_split_string(tmp_path):
    """`MerkleScheme.Collision` is a collision of the hash, not of how its input is split: one byte string split two ways
    (`[0] ‖ []` against `[] ‖ [0]`) gives one node, and must not count, since the children are not of the digest's width
    (PR #146's red-team review). The same witness for `Sha256.Collision` needs two different inputs, which it does not have."""
    subprocess.run(["lake", "build", "FlockProofs"], cwd=LEAN, check=True, capture_output=True)
    for goal, witness in [
        ('(MerkleScheme.sha256 "flock-leaf/sha256-unsalted").Collision',
         "⟨.node (ByteArray.mk #[0]) ByteArray.empty ByteArray.empty (ByteArray.mk #[0]), by simp only [MerklePair.Collides]; "
         "exact ⟨by decide, by decide, by decide, by decide, by decide, rfl⟩⟩"),
        ("MerkleScheme.hm96Sha512.Collision",
         "⟨.node (ByteArray.mk #[0]) ByteArray.empty ByteArray.empty (ByteArray.mk #[0]), by simp only [MerklePair.Collides]; "
         "exact ⟨by decide, by decide, by decide, by decide, by decide, rfl⟩⟩"),
    ]:
        f = tmp_path / "Split.lean"
        f.write_text(f"import FlockProofs\nopen Flock\nexample : {goal} :=\n  {witness}\n")
        p = subprocess.run(["lake", "env", "lean", str(f)], cwd=LEAN, capture_output=True, text=True)
        assert p.returncode != 0 and "decide" in p.stdout + p.stderr, (goal, p.stdout[-800:])


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_region_refuses_a_repeated_free_bit(tmp_path):
    """`mkRegion` refuses a region whose free bits repeat, since the pinned table-soundness theorems assume them distinct
    (`Statement.LinkLayout`); a repeat would pass the other checks and be covered by none (PR #278's red-team review).
    Two ranges of free bits that overlap in one bit, as `bits 0 w ++ bits slotLog c` do when `slotLog < w`, are refused."""
    subprocess.run(["lake", "build", "Flock"], cwd=LEAN, check=True, capture_output=True)
    f = tmp_path / "Region.lean"
    f.write_text("import Flock\nopen Flock\n"
                 "def verdict (free : Array Nat) : String :=\n"
                 "  match mkRegion 20 \"r\" free 0 fun _ => .empty with | .ok r => s!\"ok {r.free}\" | .error e => e\n"
                 "#eval verdict #[3, 9, 3]\n#eval verdict (#[0, 1, 2] ++ #[2, 3])\n#eval verdict #[3, 9, 4]\n")
    p = subprocess.run(["lake", "env", "lean", str(f)], cwd=LEAN, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    repeat = '"r: a free bit repeats; a region\'s free bits are distinct"'
    assert p.stdout.split("\n")[:3] == [repeat, repeat, '"ok #[3, 9, 4]"'], p.stdout


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_pin_is_a_column_of_the_block(tmp_path):
    """Both statements' pin is a column of the block, whatever the layout checks before it established (R9c's refinement
    uses it as a local fact): a first range of one slot starting at `2^k_log` gives a pin outside the block, refused."""
    subprocess.run(["lake", "build", "Flock"], cwd=LEAN, check=True, capture_output=True)
    f = tmp_path / "Pin.lean"
    f.write_text("import Flock\nopen Flock\n"
                 "def at_ (pos0 : Nat) : Circuit :=\n  { (default : Circuit) with\n    kLog := 10, nets := #[(\"u\", default)],\n"
                 "    ranges := #[{ kind := .net 0, slotLog := 7, pos0, count := 1, role := .none, packed := true }] }\n"
                 "def show_ : V Nat → String | .ok p => s!\"ok {p}\" | .error e => e\n"
                 "#eval (show_ (HmRow.pin (at_ 8)), show_ ((at_ 8).pin default))\n"
                 "#eval (show_ (HmRow.pin (at_ 7)), show_ ((at_ 7).pin default))\n")
    p = subprocess.run(["lake", "env", "lean", str(f)], cwd=LEAN, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    out = '"the pinned constant column is outside the block"'
    assert p.stdout.split("\n")[:2] == [f"({out}, {out})", '("ok 896", "ok 896")'], p.stdout


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_hm96_range_slots_are_words_inside_the_block(tmp_path):
    """`HmRow.check` refuses a range whose slots are smaller than a 128-bit word (below 64 bits the fold is no
    `αA₀ᵀe + B₀ᵀe`) or larger than the block (a packed range of no slots would escape the `k_log` bound), as the BLAKE3
    path's layout check does (R9c's refinement). A composed RoPE circuit is accepted; either change to it is refused."""
    pytest.importorskip("verity_numerical")
    from verity_flock import circuit as FC
    from verity_numerical.bench import templates as TM
    from verity_numerical.bench.generate import generate
    from verity_numerical.bench.input_sets import InputSet

    subprocess.run(["lake", "build"], cwd=LEAN, check=True, capture_output=True)
    s = InputSet.open(generate(TM.parse_subcircuit_id("rope-head/d64/neox-bf16"), 4, 20260926, tmp_path)["path"])
    low = FC.lowering_for_set(s)
    text = FC.compose(low, s.subcircuit)
    FC.stage(low, s, 0, 4, tmp_path / "st")
    head, meta_line, rest = text.split("\n", 2)
    meta = json.loads(meta_line[5:])

    def statement(circuit):
        f = tmp_path / "c.circuit"
        f.write_text(circuit)
        return subprocess.run([str(LEAN / ".lake/build/bin/flock-verify"), "statement", "--statement", FC.STATEMENT,
                               "--circuit", str(f), "--public", str(tmp_path / "st" / "pub-4.bin")], capture_output=True, text=True)

    p = statement(text)
    assert p.returncode == 0, p.stderr
    mask = next(r for r in meta["ranges"] if r["net"] == "mask")
    packed = next(r for r in meta["ranges"] if r.get("packed"))
    for r, change in [(mask, {"slot_log": 6}), (packed, {"count": 0, "slot_log": meta["k_log"] + 1})]:
        saved = dict(r)
        r.update(change)
        p = statement(f"{head}\nMETA {json.dumps(meta)}\n{rest}")
        r.clear()
        r.update(saved)
        assert p.returncode == 1 and "a range's slots are smaller than a word or larger than the block" in p.stdout, (change, p.stdout)


TOY_EXTRACTOR = r"""import FlockProofs
open Flock

/-- A toy scheme with 1-byte digests: a node is the sum of its children's bytes, a leaf the row encoding's first byte. -/
def toy : MerkleScheme :=
  { id := "toy", digestLen := 1, pcsIndex := 0, node := fun l r => ⟨#[l.get! 0 + r.get! 0]⟩,
    leaf := fun row _ => ⟨#[(MerkleScheme.rowBytes row).get! 0]⟩ }

def b1 (x : UInt8) : ByteArray := ⟨#[x]⟩
def cap : Array ByteArray := #[b1 6]
def row1 : Array F128 := #[⟨1, 0⟩]
def row2 : Array F128 := #[⟨2, 0⟩]
def row3 : Array F128 := #[⟨257, 0⟩]

def accepts (rows : Array (Array F128)) (sibs : Array ByteArray) : Bool :=
  match merkleCheck toy cap 0 1 #[0] rows sibs #[] 1 with | .ok () => true | .error _ => false

#eval (accepts #[row1] #[b1 5], accepts #[row2] #[b1 4], accepts #[row3] #[b1 5])

/-- `MerklePair.Collides toy`, decided. -/
def collides : MerklePair → Bool
  | .node a b a' b' => a.size == 1 && b.size == 1 && a'.size == 1 && b'.size == 1 && (a != a' || b != b') &&
      toy.node a b == toy.node a' b'
  | .leaf r r' y y' => r != r' && toy.leaf r y == toy.leaf r' y'

#eval (merklePair toy 0 1 #[0] #[row1] #[row2] #[b1 5] #[b1 4] #[] #[] 0).render
#eval (merklePair toy 0 1 #[0] #[row1] #[row3] #[b1 5] #[b1 5] #[] #[] 0).render
#eval (collides (merklePair toy 0 1 #[0] #[row1] #[row2] #[b1 5] #[b1 4] #[] #[] 0),
  collides (merklePair toy 0 1 #[0] #[row1] #[row3] #[b1 5] #[b1 5] #[] #[] 0))
"""


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_merkle_extractor_computes_a_collision(tmp_path):
    """`merkle_binding` is stated over `merklePair`, a computable extractor, since a collision of a fixed hash exists by
    counting (PR #146's red-team delta check). On a toy scheme with 1-byte digests, two accepted checks with different rows
    give its pair: two children pairs under one node, or two rows with one leaf, and each collides."""
    subprocess.run(["lake", "build", "FlockProofs"], cwd=LEAN, check=True, capture_output=True)
    f = tmp_path / "Toy.lean"
    f.write_text(TOY_EXTRACTOR)
    p = subprocess.run(["lake", "env", "lean", str(f)], cwd=LEAN, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    assert p.stdout.split("\n")[:4] == ["(true, true, true)", '"node 01 05 | 02 04"',
        '"leaf 01000000000000000000000000000000  | 01010000000000000000000000000000 "', "(true, true)"], p.stdout


def _net(rows, ins="1 0 1 1 2", outs="1 1 1 1 1"):
    """A `flock-ir-unit/v2` expanded circuit: one input group of 2 bits (rows 2..127 input padding), one output group."""
    return "\n".join([f"flock-ir-unit/v2 tiny {len(rows)} {len(rows) - 1} {ins} {outs}"] + rows) + "\n"


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_net_parser_refuses_rows_out_of_order(tmp_path):
    """`Net.parse` holds every computed row to the order the soundness package's `Rows.topo` names (#144's
    `test_pinned_rows_read_earlier_columns`): it reads only earlier columns that are not input padding, or the constant. Port
    groups must fit their words, and input groups the input rows."""
    base = ["1 0 1 0", "1 1 1 1"] + ["0 0"] * 126 + ["1 0 1 1", "2 128 1 1 130", "1 130 1 130"]

    def edit(r, row):
        return base[:r] + [row] + base[r + 1:]

    cases = {
        "ok": (_net(base), None),
        "later": (_net(edit(128, "1 129 1 1")), "row 128 reads column 129, not an earlier one"),
        "itself": (_net(edit(128, "1 128 1 1")), "row 128 reads column 128, not an earlier one"),
        "padding": (_net(edit(128, "1 5 1 1")), "row 128 reads column 5, an input padding column"),
        "padding-b": (_net(edit(129, "2 128 1 1 7")), "row 129 reads column 7, an input padding column"),
        "overrun": (_net(base, ins="1 0 1 1 200"), "a port group's ports overrun its words"),
        "reach": (_net(base, ins="1 0 2 1 2"), "the input rows reach the constant"),
    }
    for name, (text, _) in cases.items():
        (tmp_path / f"{name}.net").write_text(text)
    subprocess.run(["lake", "build"], cwd=LEAN, check=True, capture_output=True)
    out = subprocess.run([str(LEAN / ".lake/build/bin/flock-verify"), "net-check", *(str(tmp_path / f"{n}.net") for n in cases)],
                         capture_output=True, text=True).stdout
    got = {Path(j["file"]).stem: j for j in (json.loads(l.split(" ", 1)[1]) for l in out.splitlines() if l.startswith("NETCHECK "))}
    for name, (_, why) in cases.items():
        assert got[name]["ok"] == (why is None) and (why is None or why in got[name]["error"]), (name, got[name])
