"""The Lean verifier's trust surface: a pinned toolchain, no dependencies, and no escape hatch that would let compiled code
differ from the definitions a proof is about (`native_decide`, `implemented_by`, `extern`, `unsafe`)."""
from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

LEAN = Path(__file__).resolve().parents[1] / "verifier" / "lean"
PROOFS = Path(__file__).resolve().parents[3] / "verity" / "Security" / "Proofs"
TOOLCHAIN = "leanprover/lean4:v4.34.0"  # the security packages' Mathlib and ArkLib pin v4.34.0
FORBIDDEN = re.compile(r"\b(native_decide|implemented_by|extern|unsafe)\b")


def build_proofs_verifier() -> None:
    """`lake build Proofs.Flock.Verifier` in `security_proofs`, one process at a time: on a fresh tree Lake first clones that
    package's dependencies, and two Lake processes cloning them at once (two xdist workers) both fail."""
    key = hashlib.sha256(str(PROOFS).encode()).hexdigest()[:16]
    with open(Path(tempfile.gettempdir()) / f"verity-security-proofs-{key}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        subprocess.run(["lake", "build", "Proofs.Flock.Verifier"], cwd=PROOFS, check=True, capture_output=True)


def sources():
    """The executable's sources (its proofs are under `verity/Security/Proofs/Flock/`, with their own dependencies and checks)."""
    return sorted(p for p in LEAN.rglob("*.lean") if ".lake" not in p.parts)


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


AUDIT = PROOFS / "Flock" / "Soundness" / "Audit"
AUDIT_IMPORTS = re.compile(r"import (Mathlib\.|Proofs\.Flock\.Soundness\.Game\.|Proofs\.Flock\.Soundness\.Audit\.)")


def test_audit_layer_is_abstract():
    """The audit theorems import only the interaction model, Mathlib and each other, so any backend can instantiate them
    (`verity/Security/Proofs/Flock/Soundness/DESIGN.md` §12). The Flock instantiations are the only files that import Flock's theorems:
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
        from verity.primitives.circuits.partition import validate_unit_cut  # noqa: F401
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
    """`verity.primitives.randomness`'s exact reduction on a raw stream: (value, next position), or None when it runs out."""
    from verity.primitives.randomness import uniform_from_words
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


def _ir_tests() -> Path:
    """The circuits' `tests`, inside the `verity.primitives.circuits` the import system finds (whichever member ships it): the vectors the
    Lean verifier agrees with.  Located, never checked for here: one that is missing fails `test_the_reference_vectors_exist`, not a skip."""
    return Path(importlib.util.find_spec("verity.primitives.circuits").origin).resolve().parent / "tests"


def _template_vectors():
    try:
        import verity  # noqa: F401
    except ImportError:
        return None
    return _ir_tests() / "template_instance_vectors.json"



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
        from verity.protocols.verification.sampled_proofs.one_stage import draw
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
        from verity.primitives.circuits.cut import check_cut  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(shutil.which("lake") is None or not _has_check_cut(), reason="no Lean toolchain or no verity.primitives.circuits.cut (PR #111)")
def test_check_cut_agrees_with_the_reference(tmp_path):
    subprocess.run(["lake", "build"], cwd=LEAN, check=True, capture_output=True)
    p = subprocess.run(["python3", str(LEAN.parent / "cut_check_agree.py"), "--lean", str(LEAN / ".lake/build/bin/flock-verify"),
                        "--out", str(tmp_path), "--n", "300"], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    assert json.loads(p.stdout.strip().splitlines()[-1])["v2"]["refused_by_v1_alone"] > 0       # Q_word v2's rule is exercised


def _qword_vector_file():
    try:
        from verity.primitives.circuits.codec import decode_program  # noqa: F401
    except ImportError:
        return None
    return _ir_tests() / "qword_vectors.json"


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
        import verity.primitives.circuits.partition_object  # noqa: F401
        from verity.primitives.circuits.cut import cut_word  # noqa: F401
    except ImportError:
        return None
    return _ir_tests() / "partition_vectors.json"


@pytest.mark.skipif(shutil.which("lake") is None or _qword_vectors() is None, reason="no Lean toolchain or no Q_word evaluator (PR #111)")
def test_qword_evaluator_agrees_with_the_reference(tmp_path):
    subprocess.run(["lake", "build"], cwd=LEAN, check=True, capture_output=True)
    p = subprocess.run(["python3", str(LEAN.parent / "qword_agree.py"), "--lean", str(LEAN / ".lake/build/bin/flock-verify"),
                        "--out", str(tmp_path), "--n", "300", "--vectors", str(_qword_vectors())], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr


def _format_vectors():
    try:
        import verity  # noqa: F401
    except ImportError:
        return None
    return _ir_tests()


def test_the_reference_vectors_exist():
    """The agreement tests below skip without Lean or without core, never for a vectors file that is not where they look."""
    ir = _format_vectors()
    files = [_template_vectors(), _qword_vector_file(), _qword_vectors(),
             *(ir / n for n in ("format_vectors.json", "partition_vectors.json", "test_format_spec.py", "qcall_vectors.json"))]
    missing = [str(f) for f in files if not f.exists()]
    assert not missing, missing


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


@pytest.mark.skipif(shutil.which("lake") is None or _format_vectors() is None, reason="no Lean toolchain or no Q_call vectors")
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
    build_proofs_verifier()
    for goal, witness in [
        ('(MerkleScheme.sha256 "flock-leaf/sha256-unsalted").Collision',
         "⟨.node (ByteArray.mk #[0]) ByteArray.empty ByteArray.empty (ByteArray.mk #[0]), by simp only [MerklePair.Collides]; "
         "exact ⟨by decide, by decide, by decide, by decide, by decide, rfl⟩⟩"),
        ("MerkleScheme.hm96Sha512.Collision",
         "⟨.node (ByteArray.mk #[0]) ByteArray.empty ByteArray.empty (ByteArray.mk #[0]), by simp only [MerklePair.Collides]; "
         "exact ⟨by decide, by decide, by decide, by decide, by decide, rfl⟩⟩"),
    ]:
        f = tmp_path / "Split.lean"
        f.write_text(f"import Proofs.Flock.Verifier\nopen Flock\nexample : {goal} :=\n  {witness}\n")
        p = subprocess.run(["lake", "env", "lean", str(f)], cwd=PROOFS, capture_output=True, text=True)
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


def first_and_row(lines: list[str], name: str) -> int:
    """The line of section `name`'s first AND row: its first row that is neither an input's self row nor empty."""
    at = next(k for k, line in enumerate(lines) if line.startswith(f"CIRCUIT {name} ")) + 2
    return next(at + q for q, line in enumerate(lines[at:]) if line not in (f"1 {q} 1 {q}", "0 0"))


ROW_NETS = [("sha512x3", "three SHA-512 compressions"), ("hm96", "hm96-sha512/row/v1 commit")]


#: Bit-row lengths at every residue `sha512/row/v2`'s constants turn on: `l` mod 8 of 0, 1 and 7, and `ceil(l / 8)` mod 128
#: of 0, 1, 111, 112 and 127 (the last block all padding, or shared with 1, 111, 112 or 127 row bytes).
BIT_ROW_LENGTHS = [0, 8, 1024, 2048] + [8 * b - r for b in (1, 111, 112, 127, 128, 129, 239, 240, 255, 256) for r in (7, 1)]

BIT_ROW_BC = r"""import Flock
open Flock HmRow

/-- A bit row's `b ‖ c` from the verifier's constants: block `k`'s column `t` (`blockBit`: byte `o`, bit `b`) is `constBit`'s
constant, else the row's bit `8 (128 k + o) + b`; the blocks compressed from `midstateOf`, the digest under hm96's default
key with `salt`. -/
def bcOf (l : Nat) (row salt : String) : String := Id.run do
  let row := match unhex row with | .ok b => b | .error _ => ByteArray.empty
  let salt := match unhex salt with | .ok b => b | .error _ => ByteArray.empty
  let p : Port := { name := "p", words := (l + 15) / 16, chunks := 0, nb := 0, bits := l, v2 := true }
  let tl := tailOf p
  let mut cv := midstateOf p
  for k in [0:(p.bytes + 17 + 127) / 128] do
    let mut blk := ByteArray.mk (Array.replicate 128 0)
    for t in [0:1024] do
      let (o, b) := blockBit t
      let v := match constBit p tl k t with
        | some c => c
        | none => ((row.get! (128 * k + o)).toNat >>> b) % 2 == 1
      if v then blk := blk.set! o (blk.get! o ||| ((1 : UInt8) <<< b.toUInt8))
    cv := Sha512.compress cv blk 0
  let x : ByteArray := ⟨Array.ofFn (n := 64) fun i => (cv[i.val / 8]! >>> (8 * (7 - i.val % 8)).toUInt64).toUInt8⟩
  return hexOf (Hm96.Default512.commitString x salt)
"""


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_row_nets_are_the_verifiers(tmp_path):
    """The `sha512x3` and `hm96` sections come from the prover, so setup refuses either unless it is the net the verifier
    generates (`Flock.HmNets`, PROTOCOL.md §16.10). A composed RoPE circuit is accepted; one AND row changed in either
    section is refused, naming the row."""
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

    def statement(circuit):
        f = tmp_path / "c.circuit"
        f.write_text(circuit)
        return subprocess.run([str(LEAN / ".lake/build/bin/flock-verify"), "statement", "--statement", FC.STATEMENT,
                               "--circuit", str(f), "--public", str(tmp_path / "st" / "pub-4.bin")], capture_output=True, text=True)

    p = statement(text)
    assert p.returncode == 0, p.stderr
    for name, what in ROW_NETS:
        lines = text.split("\n")
        at = first_and_row(lines, name)
        lines[at] = "1 0 1 1"
        p = statement("\n".join(lines))
        row = at - next(k for k, line in enumerate(lines) if line.startswith(f"CIRCUIT {name} ")) - 2
        assert p.returncode == 1 and f"circuit {name}: not the verifier's {what} (row {row} differs)" in p.stdout, p.stdout


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_bit_row_constants_complete_core_s_v2_commit_strings(tmp_path):
    """The message a bit row's Δ constants leave (`HmRow.constBit` past the row's bits: its last byte's fill, then SHA-512's
    padding `tailOf`, placed by `blockBit`), compressed from `midstateOf`, gives core's ``b || c`` over
    ``sha512_row_v2_digest`` at every residue of the length."""
    import random

    from verity.primitives.commitments import hm96
    from verity.primitives.commitments.rowleaf import sha512_row_v2_digest

    rng, cases = random.Random(20261001), []
    for n_bits in BIT_ROW_LENGTHS:
        row = bytearray(rng.getrandbits(8) for _ in range(-(-n_bits // 8)))
        if n_bits % 8:
            row[-1] &= (1 << n_bits % 8) - 1
        cases.append((n_bits, bytes(row), rng.randbytes(192)))
    subprocess.run(["lake", "build", "Flock"], cwd=LEAN, check=True, capture_output=True)
    f = tmp_path / "BitRowBc.lean"
    f.write_text(BIT_ROW_BC + "".join(f'#eval bcOf {n} "{r.hex()}" "{s.hex()}"\n' for n, r, s in cases))
    p = subprocess.run(["lake", "env", "lean", str(f)], cwd=LEAN, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    inst = hm96.SHA512
    want = [f'"{inst.commit_string(inst.default_key, sha512_row_v2_digest(r, n), s).hex()}"' for n, r, s in cases]
    assert p.stdout.splitlines() == want


def _bit_row_toy(bits: list[int], d: Path, edit=None, n: int = 4, seg: frozenset = frozenset(), rows: dict | None = None,
                 salts: dict | None = None) -> str:
    """A statement of bit-row ports of lengths ``bits``, staged in ``d`` (``circuit.txt``, ``pub.bin``, the prover's
    ``inst.bin``): one unit reads every leaf, its output the XOR of all leaves with AND(bit 0, bit 1) of the first added at
    bit 0; the rows random below their lengths (seed 0; ``rows``: these), the salts under key ``k`` × 32 (``salts``: these,
    each row port's or none). The ports named in ``seg`` are `hm96-sha512/row-seg/v1` rows. ``edit`` changes META before the
    public file is staged for it."""
    import numpy as np

    from verity_flock import circuit as FC
    from verity_flock import class_statement as CS
    from verity_flock import gf2
    from verity_flock import ir_lower as IL

    words = [-(-b // 16) for b in bits]
    C = gf2.Circuit()
    ins = [C.input(f"in{j}", 16) for j in range(sum(words))]
    out = [gf2.ZERO] * 16
    for x in ins:
        out = [a ^ b for a, b in zip(out, x)]
    out[0] = out[0] ^ C.AND(ins[0][0], ins[0][1])
    C.output("out0", out)
    units = IL.Units(target=None, in_widths=[16] * len(ins), out_widths=[16], in_src=[[("leaf", p) for p in range(len(ins))]],
                     out_src=[[("ret", 0)]], signature=("toy",), gates=[], arg_gates=[], ret_gates=[])
    low = IL.Lowering("unit-class/toy", CS.ClassDefinition("unit-class/toy", tuple(words), 1), units, C,
                      IL.layout(C, [f"in{j}" for j in range(len(ins))], ["out0"]))
    CS.cache_slot_circuits()
    sub, iset = CS._shims("toy", json.dumps(bits).encode(), len(bits))
    text = FC.compose(low, sub, port_bits=bits, seg_ports=seg)
    if edit is not None:
        head, meta_line, rest = text.split("\n", 2)
        meta = json.loads(meta_line[5:])
        edit(meta)
        text = f"{head}\nMETA {json.dumps(meta)}\n{rest}"
    if rows is None:
        rng, rows = np.random.default_rng(0), {}
        for i, (b, w) in enumerate(zip(bits, words)):
            r = rng.integers(0, 1 << 16, (n, w), dtype=np.uint64).astype(np.uint16)
            if w and b % 16:
                r[:, -1] &= np.uint16((1 << (b % 16)) - 1)
            rows[f"p{i}"] = r
    got, _cut, ok = low.evaluate(np.concatenate([rows[f"p{i}"] for i in range(len(bits))], axis=1).astype(np.uint64))
    assert int(ok) == (1 << n) - 1
    d.mkdir(parents=True, exist_ok=True)
    (d / "circuit.txt").write_text(text)
    if salts is not None:
        meta = json.loads(text.split("\n", 2)[1][5:])
        salts = {nm: salts.get(nm) or FC.stage_salts(iset, 0, n, [p], b"k" * 32)[nm] for nm, p in
                 ((p.name, p) for p in FC._meta_ports(meta))}
    FC.write(text, iset, 0, n, d / "inst.bin", d / "pub.bin", b"k" * 32, rows=rows, outs={"out": got.astype(np.uint16)}, salts=salts)
    return text


def _seg_writer_toy(out_words: list[int], d: Path, n: int = 4, seg: bool = True, in_bits: int = 1024,
                    share_rows: bool = False) -> tuple[str, dict, dict]:
    """A statement whose one unit writes output ports ``a0, a1, …`` of ``out_words`` words, committed as one
    `hm96-sha512/row-seg/v1` row ``a0`` of exactly their words (META ``out_row_ports``; ``seg`` false: one exact-length
    `hm96-sha512/row/v2` row, META ``out_row_ports`` only for a run of several), staged in ``d`` like `_bit_row_toy`:
    one input row ``p0`` of 64 words and ``in_bits`` bits, 1009 to 1024 (seed 1; the bits past ``in_bits`` zero), output
    word ``o`` the XOR of input word ``o mod 64`` and input word ``o + 1 mod 64`` rotated by ``o mod 16``, with AND(bit 0,
    bit 1) of the first added at word 0's bit 0; a lone 1-word output is that AND alone, zero above, so an admission value
    like PoUW's ok word. The salts are SHAKE-256 of ``row-seg-pair`` and the row. ``share_rows``: the files hold the input
    row as a shared-row table (`FC.write`'s). Returns the circuit text, the row's words per instance and its salts."""
    import hashlib
    from types import SimpleNamespace

    import numpy as np

    from verity.primitives.circuits.types import Array, Tuple, Value
    from verity_flock import circuit as FC
    from verity_flock import class_statement as CS
    from verity_flock import gf2
    from verity_flock import ir_lower as IL

    C = gf2.Circuit()
    ins = [C.input(f"in{j}", 16) for j in range(64)]
    W = sum(out_words)
    for o in range(W):
        a, b, r = ins[o % 64], ins[(o + 1) % 64], o % 16
        out = [a[t] ^ b[(t + r) % 16] for t in range(16)]
        if out_words == [1]:
            out = [C.AND(ins[0][0], ins[0][1])] + [gf2.ZERO] * 15
        elif o == 0:
            out[0] = out[0] ^ C.AND(ins[0][0], ins[0][1])
        C.output(f"out{o}", out)
    units = IL.Units(target=None, in_widths=[16] * 64, out_widths=[16] * W, in_src=[[("leaf", p) for p in range(64)]],
                     out_src=[[("ret", o) for o in range(W)]], signature=("toy",), gates=[], arg_gates=[], ret_gates=[])
    sig = ((("p0", Array(64, Value(16))),), Tuple(tuple(Array(w, Value(16)) for w in out_words)))
    definition = SimpleNamespace(id="unit-class/seg-writer", signature=lambda: sig)
    low = IL.Lowering("unit-class/seg-writer", definition, units, C, IL.layout(C, [f"in{j}" for j in range(64)], [f"out{o}" for o in range(W)]))
    CS.cache_slot_circuits()
    port = lambda nm: SimpleNamespace(name=nm, dtype="u16")  # noqa: E731
    content = json.dumps(out_words).encode()
    sub = SimpleNamespace(id="unit-class/seg-writer", template=SimpleNamespace(id="unit-class"), inputs=[port("p0")],
                          outputs=[port(f"a{i}") for i in range(len(out_words))],
                          program_digests=lambda statics: {"unit-class/seg-writer": hashlib.sha256(content).hexdigest()})
    iset = SimpleNamespace(subcircuit=sub, name="unit-class-seg-writer", content_digest=hashlib.sha256(content).hexdigest())
    names = tuple(f"a{i}" for i in range(len(out_words)))
    assert 1009 <= in_bits <= 1024
    text = FC.compose(low, sub, port_bits=[in_bits], **({"seg_out_rows": (names,)} if seg else {"v2_out_rows": (names,)}))
    rows = {"p0": np.random.default_rng(1).integers(0, 1 << 16, (n, 64), dtype=np.uint64).astype(np.uint16)}
    if in_bits % 16:
        rows["p0"][:, -1] &= np.uint16((1 << (in_bits % 16)) - 1)
    got, _cut, ok = low.evaluate(rows["p0"].astype(np.uint64))
    assert int(ok) == (1 << n) - 1
    got = got.astype(np.uint16)
    at = np.cumsum([0] + out_words)
    outs = {nm: got[:, at[i]:at[i + 1]] for i, nm in enumerate(names)}
    salts = {"a0": [hashlib.shake_256(b"row-seg-pair\0" + got[i].tobytes()).digest(192) for i in range(n)]}
    meta = json.loads(text.split("\n", 2)[1][5:])
    salts["p0"] = FC.stage_salts(iset, 0, n, FC._meta_ports(meta)[:1], b"k" * 32)["p0"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "circuit.txt").write_text(text)
    FC.write(text, iset, 0, n, d / "inst.bin", d / "pub.bin", b"k" * 32, rows=rows, outs=outs, salts=salts, share_rows=share_rows)
    return text, got, salts["a0"]


def _public_bcs(pub: Path, rows: int) -> list[list[bytes]]:
    """A non-shared public file's ``b || c``, per instance per row."""
    head, body = pub.read_bytes().split(b"\n", 1)
    n = json.loads(head)["instances"]
    return [[body[128 * (i * rows + p):128 * (i * rows + p + 1)] for p in range(rows)] for i in range(n)]


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_bit_row_statements_are_the_rust_verifier_s_and_a_meta_that_differs_is_refused(tmp_path):
    """Two statements whose bit-row ports have every residue between them (length 0; length mod 8 of 0, 1 and 7; bytes mod
    128 of 0, 1, 111, 112 and 127) have the statement digests `flock-circuit statement` gives them under its build's identity
    `verity/flock-circuit` (its `row_leaf_in_circuit` naming `hm96-sha512/row/v2`; the output row a bit row of all its words'
    bits, §16.13), seeded and under `FC_COINS=os` (`--coins os`: the live-OS identity, `Tags.withCoins`, composed with the
    bit-row and hidden-output text), so the two verifiers' Δ constants, forced-zero leaf bits, row wires and identities
    agree. A META that differs from what the verifier derives is refused, with the Rust verifier's message: a wrong length
    (its prefix, or its words and compressions), a length of 2^64 or one past `MAX_COMPS`, a v1 prefix and midstate under v2,
    a changed padding, both row-schema keys, a v1 schema in `row_schemas`, a schema missing, no row with a compression, and a
    pinned dummy ``b || c`` that is not the padding instance's row of the port's bits (here, a v1 row's of its words)."""
    from verity.primitives.commitments import hm96
    from verity.primitives.commitments.rowleaf import ROLE_X, sha512_row_digest
    from verity_flock import circuit as FC

    subprocess.run(["lake", "build"], cwd=LEAN, check=True, capture_output=True)

    def statement(d, text, *extra):
        (d / "c.txt").write_text(text)
        return subprocess.run([str(LEAN / ".lake/build/bin/flock-verify"), "statement", "--statement", "verity/flock-circuit",
                               "--circuit", str(d / "c.txt"), "--public", str(d / "pub.bin"), *extra], capture_output=True, text=True)

    digests = {}
    for i, bits in enumerate([[0, 1, 881, 887, 889, 895], [1009, 1015, 1024, 1031, 2041, 1905]]):
        d = tmp_path / f"s{i}"
        text = _bit_row_toy(bits, d)
        for coins in ("seed", "os"):
            p = statement(d, text, "--coins", coins)
            assert p.returncode == 0, p.stdout + p.stderr
            digests[i, coins] = json.loads(p.stdout.strip().splitlines()[-1])["statement_digest"]
    assert digests == {
        (0, "seed"): "4e13f289098265f4b9266a73f42bc319bbb3bb6c389465fb459e0c0c33bc828b1124a0354b66873acfb70024d7ef7a770eb7b922ade9da2ee2f0063ce8558c5f",
        (0, "os"): "848227d1e54b802ba9c32f0da36a538d2a6c29be7802695c2ed379b08f36088edac6923c1fb7555f72c8ecb234ff823bc70a237f9d00c4f32e4b6a02d97031c9",
        (1, "seed"): "d7f0a1ca2271a68379f5c44d76beaf97769b32b9b064f7c0ae6ba9d2fc406c6241e9fb6aff61e21a79dcd0fd1ba740e323d46015015f61b45b8196ccbb7a29ec",
        (1, "os"): "6ac4156572a422da1be1be1c91b66ecdf66aa949c5cf1ac36b759fcdbec355de60550d16706517fe7fc0c781ca984933866c216178c948eff1894f0f9b1410ca"}

    d = tmp_path / "s0"
    head, meta_line, rest = (d / "c.txt").read_text().split("\n", 2)
    inst = hm96.SHA512
    v1 = FC._row_consts(FC.Port("p2", 56))

    def at(path, value):
        def f(m):
            *keys, last = path
            for k in keys:
                m = m[k]
            m[last] = value(m[last]) if callable(value) else value
        return f

    def drop(m):
        for k in ("ports", "row_schemas", "rows"):
            m[k] = m[k][:4]

    cases = {
        "a wrong length, its prefix": (at(["row_schemas", 2, 1], 882), "META: a row's constants are not sha512/row/v2 or sha512/row-seg/v1's"),
        "a wrong length, its words": (at(["row_schemas", 2, 1], 897),
                                      "port p2: a row of 897 bits is 57 u16 leaves and 1 compressions, not META's 56 and 0"),
        "2^64": (at(["row_schemas", 2, 1], 1 << 64), "META: a bit row's length is not a u64"),
        "past MAX_COMPS": (at(["row_schemas", 2, 1], (1 << 64) - 1), "port p2: a row of 18446744073709551615 bits is"),
        "v1 prefix": (lambda m: m["rows"][2].update(prefix=v1["prefix"], midstate=v1["midstate"]),
                      "META: a row's constants are not sha512/row/v2 or sha512/row-seg/v1's"),
        "padding": (at(["rows", 2, "pad"], lambda s: s[:-1] + format(int(s[-1], 16) ^ 1, "x")),
                    "META: a row's constants are not sha512/row/v2 or sha512/row-seg/v1's"),
        "both keys": (at(["row_schema"], "hm96-sha512/row/v1"), "META: rows are not frame-v3-sha512/hm96-sha512"),
        "v1 entry": (at(["row_schemas", 2, 0], "hm96-sha512/row/v1"), "META: row_schemas: each port's is"),
        "one short": (at(["row_schemas"], lambda s: s[:-1]), "META: row_schemas is not one [schema, bits] per port"),
        "no compression": (drop, "META: no row has a compression before its last block"),
    }
    for name, (change, why) in cases.items():
        meta = json.loads(meta_line[5:])
        change(meta)
        p = statement(d, f"{head}\nMETA {json.dumps(meta)}\n{rest}")
        assert p.returncode != 0 and why in p.stdout + p.stderr, (name, p.stdout + p.stderr)
    # the dummy is the circuit's, so the public file is staged for the changed circuit
    v1_zero = inst.commit_string(inst.default_key, sha512_row_digest([0] * 56, 16, ROLE_X), bytes(192)).hex()
    d = tmp_path / "dummy"
    p = statement(d, _bit_row_toy([0, 1, 881, 887, 889, 895], d, edit=at(["dummy", "bc", 2], v1_zero)))
    assert p.returncode != 0 and "the pinned dummy b || c of row p2 is not the padding instance's row" in p.stdout + p.stderr, p.stdout + p.stderr


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_segmented_row_statements_are_the_rust_verifier_s_and_a_meta_that_differs_is_refused(tmp_path):
    """Statements of `hm96-sha512/row-seg/v1` ports beside v2 ones (lengths 8193 and 65,536 with 881; and 0, 1, 8192 and 24,581
    all segmented) have the statement digests a proving build of `flock-circuit statement` gives them, seeded and under
    `FC_COINS=os`, so the two verifiers' segment chains, outer messages, row wires and the per-port `row_leaf_in_circuit`
    identity agree. A META that differs is refused with the Rust verifier's message: a segmented row's length (its prefix, or
    its words), a seg port read as v2 or a v2 port as seg (their compressions), its outer padding or midstate."""
    subprocess.run(["lake", "build"], cwd=LEAN, check=True, capture_output=True)

    def statement(d, text, *extra):
        (d / "c.txt").write_text(text)
        return subprocess.run([str(LEAN / ".lake/build/bin/flock-verify"), "statement", "--statement", "verity/flock-circuit",
                               "--circuit", str(d / "c.txt"), "--public", str(d / "pub.bin"), *extra], capture_output=True, text=True)

    digests = {}
    for i, (bits, seg) in enumerate([([8193, 881, 65536], {"p0", "p2"}), ([0, 1, 8192, 24581], {"p0", "p1", "p2", "p3"})]):
        d = tmp_path / f"s{i}"
        text = _bit_row_toy(bits, d, seg=frozenset(seg))
        for coins in ("seed", "os"):
            p = statement(d, text, "--coins", coins)
            assert p.returncode == 0, p.stdout + p.stderr
            digests[i, coins] = json.loads(p.stdout.strip().splitlines()[-1])["statement_digest"]
    assert digests == {
        (0, "seed"): "64611b46eae0b2d107a009e6eab3036caeb99937ffd6b83ca028ba39b2d6d795a92d7f287e1cacbefaf4b4957387c162e90788f993af83b2498a2f8c5a2b705e",
        (0, "os"): "1bede19b999228b0515cd7def8b7e62a97e003af303cec10ed1415496970b170d1a3479333ea9dc1ad238ac653707b6b91947b6984fa7b8798c322b77bcb8f3b",
        (1, "seed"): "a8fb31e615a07bf6ebf24259d6340dbc0c97ccd325950456ccfb16eb415a070db97600985dbb314b080c9cdff6fa9f6cb788946c01b189784a5bf365947d798e",
        (1, "os"): "92071601dd7ca72a7e31105e5df1290b99a86cea829843b4a900b08424b531ecd8484e146e757c756eb4439999895026fad85dfebd7c529e739ad1fc7380f028"}

    d = tmp_path / "s0"
    head, meta_line, rest = (d / "c.txt").read_text().split("\n", 2)
    bad = "META: a row's constants are not sha512/row/v2 or sha512/row-seg/v1's"
    cases = {
        "a wrong length, its prefix": (lambda m: m["row_schemas"][0].__setitem__(1, 8194), bad),
        "a wrong length, its words": (lambda m: m["row_schemas"][0].__setitem__(1, 8209),
                                      "port p0: a row of 8209 bits is 514 u16 leaves and 11 compressions, not META's 513 and 11"),
        "seg read as v2": (lambda m: m["row_schemas"][0].__setitem__(0, "hm96-sha512/row/v2"),
                           "port p0: a row of 8193 bits is 513 u16 leaves and 8 compressions, not META's 513 and 11"),
        "v2 read as seg": (lambda m: m["row_schemas"][1].__setitem__(0, "hm96-sha512/row-seg/v1"),
                           "port p1: a row of 881 bits is 56 u16 leaves and 1 compressions, not META's 56 and 0"),
        "outer padding": (lambda m: m["rows"][0].__setitem__("pad", m["rows"][0]["pad"][:-1] + "1"), bad),
        "outer midstate": (lambda m: m["rows"][0]["midstate"].__setitem__(0, "0" * 16), bad),
    }
    for name, (change, why) in cases.items():
        meta = json.loads(meta_line[5:])
        change(meta)
        p = statement(d, f"{head}\nMETA {json.dumps(meta)}\n{rest}")
        assert p.returncode != 0 and why in p.stdout + p.stderr, (name, p.stdout + p.stderr)


#: the row-seg writer/reader pairs: each case's output ports' words (PoUW's A′ ‖ P_A: nvfp4-p at k = 4096, its codes, scales
#: and P; fp8-p at k = 1024, its codes and P), instance 0's row digest and hm96 tree leaf, and the writer's and reader's
#: statement digests (`flock-circuit statement`'s, seeded and under `FC_COINS=os`)
ROW_SEG_PAIRS = {
    "row-seg-pair/nvfp4-p-k4096": ([1024, 128, 64], {
        "x": "16d8057b621b6658f26fecb25b40198d94d23c8a4fd0bf76b134400bde35c7cf8eb4ab3741a6c99bcb84489ed4c7d0fb4cd14c9732bcf6e3da448740beaf1c41",
        "tree_leaf": "002d453763e96ffadb7264dddbffcb9f21acdb15d9f77ba7d9af2c3132993143c2d454622247f2a032f4ece37c89f8428f9801397bdadb6a2528545e852575bc",
        ("writer", "seed"): "d19bf5a1599da4afe3b4c6d7a0f633ece5884676ba38f2d889dd48a930a7d7a5960bdc5ed14ac4e4f7dfbd4e648f88fbbef2a45f5eb20112d853f9d872ab1b40",
        ("writer", "os"): "c0705aab599d65b26d47505deec97940ead97733371a56a180951aede0f48f6425c6ad3509ee9d9be567c5cf4673143fd6934bfd9e4c131f464310a3a10de3f0",
        ("reader", "seed"): "af75f370c3d32bcf7bc1b6b442f781b7620e6e38fe87b7aadb0fab3455627d5716fc5dc998e5b9791e0d0bc805bfab1d9611875b49adbdd037359de4c943e9ce",
        ("reader", "os"): "7c6de494a0e39341ebbb80d5058753c2a3dc8e086475ab4f0c0e8b291351631e9bf97f1fdad53bdec6fbbed33472171b442b2b12ea55758c20cd10095281ebc4"}),
    "row-seg-pair/fp8-p-k1024": ([512, 64], {
        "x": "85b8da5a6bc9542ab5eb4c07f54226866b64d8616fa957a2ba0bcb5f8d3f16170749a88095e67c7ec02ad6086fef1f356c15b77bd3f20118477cf3146952c534",
        "tree_leaf": "16ca66a6e6c34877c7fc2b82d1a2fb57b7d90120244012c969b5f80221fd52e402a41c98cb8d1c8192d3abab8ff87dc8715fbf3693861fa27e2b08a48e1e7848",
        ("writer", "seed"): "29f7afaae44933923b2a297b28c860e914f5ce820317a5173fcaf4ca5e040296f2ec00f51a549dfec1109c838e366b4e617dfb616005b65a364c90c14c910242",
        ("writer", "os"): "16b5a0b69375602be8668506c396cf384c046ad7de613a26430b69c24a1af6be4b3d7160d9abe816175cabd211be183eb77405ee7d96397e4a2ce55c24ecfde3",
        ("reader", "seed"): "5f1284d76f44fd9620631a22ce04e61ad509b6124fbd200312cb450766f12cab86ba2ea68427ef7baee4aa0c41d3f3c59d0cd8a8028b6faf103687b8d01d2b65",
        ("reader", "os"): "49fc573f216a6661850183f6b5b1a843c2402c0a7be532729dffcce9a6e7c0205932a738d7576be4fedb834f784dfa81e168346c504bcd3fff64093af196acfa"}),
}


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_a_segmented_row_written_as_several_output_ports_is_the_row_its_reader_commits(tmp_path):
    """One row written by one unit as several output ports (META `out_row_ports`: one `hm96-sha512/row-seg/v1` row of exactly
    their words) and read by another as one input port has, on both sides, the same row digest (core's
    `sha512_row_seg_word_digest`), commitment ``b || c`` and hm96 tree leaf, so their frame leaves agree under one domain; each
    side's statement digest is the one `flock-circuit statement` gives it, seeded and under `FC_COINS=os`. A writer's META
    that the statement can't link is refused with the Rust verifier's message: runs that don't cover the output ports once or
    don't match the row's length, no runs (one row per port), a row named after another port, a later port of its run named
    after an input row, and its length or schema changed. The composer refuses a run that isn't consecutive output ports."""
    from verity.primitives.commitments import hm96
    from verity.primitives.commitments.rowleaf import sha512_row_seg_word_digest
    from verity_flock import circuit as FC

    subprocess.run(["lake", "build"], cwd=LEAN, check=True, capture_output=True)
    inst = hm96.SHA512

    def statement(d, text, *extra):
        (d / "c.txt").write_text(text)
        return subprocess.run([str(LEAN / ".lake/build/bin/flock-verify"), "statement", "--statement", "verity/flock-circuit",
                               "--circuit", str(d / "c.txt"), "--public", str(d / "pub.bin"), *extra], capture_output=True, text=True)

    for vid, (out_words, want) in ROW_SEG_PAIRS.items():
        dw, dr = tmp_path / vid / "writer", tmp_path / vid / "reader"
        text, got, salts = _seg_writer_toy(out_words, dw)
        rtext = _bit_row_toy([16 * sum(out_words)], dr, seg=frozenset({"p0"}), rows={"p0": got}, salts={"p0": salts})
        bw, br = _public_bcs(dw / "pub.bin", 2), _public_bcs(dr / "pub.bin", 2)
        for i in range(len(got)):
            x = sha512_row_seg_word_digest([int(v) for v in got[i]], 16)
            assert bw[i][1] == br[i][0] == inst.commit_string(inst.default_key, x, salts[i]), (vid, i)
        assert sha512_row_seg_word_digest([int(v) for v in got[0]], 16).hex() == want["x"], vid
        assert inst.tree_leaf(inst.default_key, bw[0][1]).hex() == want["tree_leaf"] == inst.tree_leaf(inst.default_key, br[0][0]).hex()
        for side, d, t in (("writer", dw, text), ("reader", dr, rtext)):
            for coins in ("seed", "os"):
                p = statement(d, t, "--coins", coins)
                assert p.returncode == 0, p.stdout + p.stderr
                assert json.loads(p.stdout.strip().splitlines()[-1])["statement_digest"] == want[side, coins], (vid, side, coins)

    d = tmp_path / "row-seg-pair/nvfp4-p-k4096" / "writer"
    head, meta_line, rest = (d / "circuit.txt").read_text().split("\n", 2)
    rows_bad = "the rows are not the input rows, then one output row per output port"
    runs_bad = "META: out_row_ports is not each output row's run of output ports, in order, covering every output port once"
    cases = {
        "runs of another length": (lambda m: m.__setitem__("out_row_ports", [2, 1]), rows_bad),
        "runs past the ports": (lambda m: m.__setitem__("out_row_ports", [4]), runs_bad),
        "an empty run": (lambda m: m.__setitem__("out_row_ports", [0, 3]), runs_bad),
        "no runs": (lambda m: m.pop("out_row_ports"), rows_bad),
        "named after another port": (lambda m: m["ports"][1].__setitem__(0, "a1"), rows_bad),
        "a later port of the run named after an input row": (lambda m: m["out_ports"][1].__setitem__(0, "p0"), rows_bad),
        "its length": (lambda m: m["row_schemas"][1].__setitem__(1, 19440),
                       "port a0: a row of 19440 bits is 1215 u16 leaves and 23 compressions, not META's 1216 and 23"),
        "read as v2": (lambda m: m["row_schemas"][1].__setitem__(0, "hm96-sha512/row/v2"),
                       "port a0: a row of 19456 bits is 1216 u16 leaves and 19 compressions, not META's 1216 and 23"),
    }
    for name, (change, why) in cases.items():
        meta = json.loads(meta_line[5:])
        change(meta)
        p = statement(d, f"{head}\nMETA {json.dumps(meta)}\n{rest}")
        assert p.returncode != 0 and why in p.stdout + p.stderr, (name, p.stdout + p.stderr)
    with pytest.raises(ValueError, match="a run of consecutive output ports"):
        FC._out_ports([["a0", 4], ["a1", 4], ["a2", 4]], True, (("a0", "a2"),))
    with pytest.raises(ValueError, match="a run of consecutive output ports"):
        FC._out_ports([["a0", 4], ["a1", 4], ["a2", 4]], True, (("a0", "a1"), ("a1", "a2")))


ROW_V2_PAIRS = {
    "row-v2-pair/ok": ([1], {
        "x": "5bba43fb72473ee2391e7fc71d9f3ec753c6397c80fde57eae6a81c30cd9d4a3522fda587d392d6ddef14ed97b95d76a48cb6748575f089ae802b25e3564f411",
        "tree_leaf": "d96220619679f36cc1864d8a4563b8e626ca22706fbe4a44136ac3299b40dc573a3626c14b4f309581481315a59eeb3ec4d98a6c7464f1a93d4f98b5fec92f0c",
        ("writer", "seed"): "50ce4ab9fa2439e96ae27fb90b50d64f464a21fc5048e88c441776e556bf51a220854c52e93d8d0d0286ae9494754e911f234682cdb47adae9e889472dfa4a12",
        ("writer", "os"): "2dba6f2e5c2e2c89c218b20db02e83844fb536a076df41049cc65b6f031fc6b278c2e8705874578ae06e57a149b978048f4668546550b795d60b801d6d603f24",
        ("reader", "seed"): "aa6c9ce63bf54284788781782ddc7787bb76b0a5790258987163a7c88daed0c9ed1327d1e6bd25d82601396f2d56eb7f8dc23dce98a0a2e1f4349afa89705884",
        ("reader", "os"): "5ed878518c8fd815a12bd600f543af50041d0968864b4775144de24c0121303f8ce593adb1cce14006139212cb18fd56fcebdb85b0737c23d8ffcc49da0b23bf"}),
    "row-v2-pair/tile": ([16], {
        "x": "dcf6bf88ec1c4b03b2dccbfc2aa35c24197f25dc9f30918ad3f48c63acce67eb3dc1eef2da7b82f8ab25a50d7ef4e8364b488e6d6967c2e49fa5dc78615934ad",
        "tree_leaf": "5bb689f861dd1a92700580a52755d7f7e3743604dbf7e05b9e497e15547c60552cbeec381b4106062661ebc032afd854e1c2f2b8b3c605209d7b7f618d2b9ecd",
        ("writer", "seed"): "7c508853bf2c73d45e42e93a5f0634fab17506738767b9f471cc94c2807ab66da182846685f4a3116caa6ad1019c9dae2119f0ecb7af53aa66e3ca8fc7fcc5f1",
        ("writer", "os"): "9a016e162e4b597d33be389716f07268a2e2b33c4072214fe1b3d6ceef9fa515baeafd9ad4f78a669fdc446960941ef046b13ff308017106f1fae703ec91aade",
        ("reader", "seed"): "c9831b806b7fc442abdecd783e4c7cb1751a58b1ed4521fdbe47e027ca3aaf8ea84569eda2019485a9cb10d52d504e04c700c4be601ef8f1c0cc6e20fef8cb41",
        ("reader", "os"): "0518a93ebfc6902eb2cf4ee34ff6bb339418f1ad9269de3e8b3bad6bc287b37746bfdc1e299e99ca2872621f51c37af024107cfb56d7c5292d66b77a774e2fc5"}),
    "row-v2-pair/nvfp4-p-w-k1024": ([256, 32, 64], {
        "x": "a7c153b39c1a75d0d084a36f2bfd904c426e5e9d5cd10aecef9bff874db9161388fce6403418f852609acdd0025017f9ee8080d03b9b754c9488eef5201d3230",
        "tree_leaf": "e428c85590300f43a826da068c2d855bd61ae044de61a0871a7a944ba55b2ba27fb30ba143369554326e3e328522c02acb6945febc615177f1f24718bd3ba61e",
        ("writer", "seed"): "017f3b3be47a2b7c6db6a3b0a76a2b32b0be8f0e9ff7a24292560c6950d79cf9096614ded05245f8d073f8b03c09f6dc7d24592b4d653bf3d92a4e52a141c5e5",
        ("writer", "os"): "f9e3c7db2b0dc9426491903f8bd1ed67caf93618bb33f8da5c4847a5335d9a8b38dd166ced2bb9ccf32b98ac03ae521110516b57045409722f84c95c6eb15388",
        ("reader", "seed"): "8bddd38ba29090611353001dbe8dd7d935c8b700a151a6a1b52723e628a75d50a920e3f05a86909c1849944c54db3324080e403a457103310cf21692ce1f9c54",
        ("reader", "os"): "0c52a99db90a9383c404a73f506dfb41f3185e6ad5dc352d557ed715db75ced7c9f95b037d88006f36409a9eaf976f924491d009ed906496de03e79604a52ac4"}),
    "row-v2-pair/fp8-p-w-k1024": ([512, 64], {
        "x": "bc97c0abcf9b6bc376b822f075765c31f605a85b219d95a486b60f7d605168eac1df645ac8c4d5d534007611b2b8d6ff50281d6c6e3cbc8fc227e1f9cfe6891f",
        "tree_leaf": "9cce701f152b279895a52bc92133245075ca902d0b65707540641e026d2bc9b07abf74b53e4436fe1bc247c393f03ad7055e4138c5eb30991119e409ebe9baf8",
        ("writer", "seed"): "1152379000c052fd0a63f3d85e314a310883f9a6c98f1d6a5a3431caf0c7f21c4811f0d675797a8aed3d7209dfaa97f177228b98705ccfbcbe5561873fa9138a",
        ("writer", "os"): "b5af933e90eb7123941c1db13d30aa99a29a93eae06a013691d0ec63b0ff42b8306c62ef5d45c61cdfc46b1ba68d6412260a84790faa4a276cb288ac12af8b6b",
        ("reader", "seed"): "10f709750e425d97b57f60e3a58fe8c8d5c5849fb8fbad400e97ca52dc0d4cd4781fcb690518af5149dfb873e3a9de701dd9520677e6c2f200b85ebeb2bd4396",
        ("reader", "os"): "8ceab99220805cd73fce947d36c76d20adebc6d483e25a387bf055878a2c37200684884e773546cfebdd74517e2530edfcd3bdde8d183feeab783fdc89ff1c1b"}),
}


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_an_exact_v2_output_row_of_one_or_several_ports_is_the_row_its_reader_commits(tmp_path):
    """An output row of exactly its words (`hm96-sha512/row/v2`, not padded to whole blocks), written as one output port
    (PoUW's 1-word ok, its 16-word tile digest) or as a run of them (META `out_row_ports`: the nvfp4-p and fp8-p W rows), and
    read by another unit as one input port has, on both sides, the same row digest (core's `sha512_row_v2_digest`),
    commitment ``b || c`` and hm96 tree leaf; each side's statement digest is the one `flock-circuit statement` gives it,
    seeded and under `FC_COINS=os`. A run written as a v1 row, or at a length other than its words, is refused."""
    import numpy as np
    from verity.primitives.commitments import hm96
    from verity.primitives.commitments.rowleaf import sha512_row_v2_digest

    subprocess.run(["lake", "build"], cwd=LEAN, check=True, capture_output=True)
    inst = hm96.SHA512

    def statement(d, text, *extra):
        (d / "c.txt").write_text(text)
        return subprocess.run([str(LEAN / ".lake/build/bin/flock-verify"), "statement", "--statement", "verity/flock-circuit",
                               "--circuit", str(d / "c.txt"), "--public", str(d / "pub.bin"), *extra], capture_output=True, text=True)

    def digest(row, words):
        return sha512_row_v2_digest(np.ascontiguousarray(row, dtype="<u2").tobytes(), 16 * words)

    for vid, (out_words, want) in ROW_V2_PAIRS.items():
        dw, dr = tmp_path / vid / "writer", tmp_path / vid / "reader"
        text, got, salts = _seg_writer_toy(out_words, dw, seg=False)
        rtext = _bit_row_toy([16 * sum(out_words)], dr, rows={"p0": got}, salts={"p0": salts})
        bw, br = _public_bcs(dw / "pub.bin", 2), _public_bcs(dr / "pub.bin", 2)
        for i in range(len(got)):
            assert bw[i][1] == br[i][0] == inst.commit_string(inst.default_key, digest(got[i], sum(out_words)), salts[i]), (vid, i)
        assert digest(got[0], sum(out_words)).hex() == want["x"], vid
        assert inst.tree_leaf(inst.default_key, bw[0][1]).hex() == want["tree_leaf"] == inst.tree_leaf(inst.default_key, br[0][0]).hex()
        for side, d, t in (("writer", dw, text), ("reader", dr, rtext)):
            for coins in ("seed", "os"):
                p = statement(d, t, "--coins", coins)
                assert p.returncode == 0, p.stdout + p.stderr
                assert json.loads(p.stdout.strip().splitlines()[-1])["statement_digest"] == want[side, coins], (vid, side, coins)

    d = tmp_path / "row-v2-pair/nvfp4-p-w-k1024" / "writer"
    head, meta_line, rest = (d / "circuit.txt").read_text().split("\n", 2)
    cases = {
        "a run as a v1 row": lambda m: m["row_schemas"][1].__setitem__(0, "hm96-sha512/row/v1"),
        "a run padded to whole blocks": lambda m: m["row_schemas"][1].__setitem__(1, 6144),
        "a run one word short": lambda m: m["row_schemas"][1].__setitem__(1, 5616),
    }
    for name, change in cases.items():
        meta = json.loads(meta_line[5:])
        change(meta)
        p = statement(d, f"{head}\nMETA {json.dumps(meta)}\n{rest}")
        assert p.returncode != 0, name


#: frame_v3 PROTOCOL §6's rule for META ``out_row_ports`` (present iff an output row is row-seg or a run covers more than one
#: port), per writer: its output words, whether its row is row-seg, the ``out_row_ports`` META names, and the circuit's SHA-512
OUT_ROW_PORTS_RULE = {
    "v2-lone-1": ([1], False, None,
                  "47aa7213fd618ec4889ebc494a856009061ce448b52fcc3206275ef34cac782bc130cc3ab7b58bae18815ef69f72f1929c7a41e733af21b7750e8a12b80ddc33"),
    "v2-lone-16": ([16], False, None,
                   "fc0533fdb73983738defcc8e009c9a4748b4d9342f1f91cbb88b1b615dda9eaa293a20b341f93f2df46e3dd890975967de0283e18f33137005ad09f30a5b7a84"),
    "v2-run-3": ([256, 32, 64], False, [3],
                 "f05cfe6e0b54dc144b762d023592439afc2796c318bddaafaafcad60f5d6879f27fee2c24be5aaae16d263240a2ab0c176455da7e6aaf9648e4c14f8d39fbef4"),
    "seg-lone-1": ([64], True, [1],
                   "8dfc5343c90d516e681dec89c9833cecd6ab006b70dc25d301d7fe24c07969bf2a93fc5d15a09d4535dfe3daed548f779d7b9a80f5c233023f655f589a8b3400"),
    "seg-run-2": ([512, 64], True, [2],
                  "d3a3df53b670bc95dc0a28e2d2e2f8712e4bdc89868fa370dd09415e5262e054e1e5b390214424ea50038357d1defb0f91f12426fbacaf4b6dcb99a11bbaf0fd"),
}


@pytest.mark.parametrize("vid", list(OUT_ROW_PORTS_RULE))
def test_out_row_ports_is_in_meta_only_for_a_row_seg_output_row_or_a_run_of_several_ports(vid, tmp_path):
    """`circuit.compose` names META ``out_row_ports`` exactly when an output row is row-seg or a run covers more than one port
    (frame_v3 PROTOCOL §6), and the writer's circuit text, META included, is the pinned one: the rule decides the statement's
    identity, so moving it moves these digests."""
    from verity_flock import circuit as FC

    out_words, seg, runs, sha = OUT_ROW_PORTS_RULE[vid]
    text, _got, _salts = _seg_writer_toy(out_words, tmp_path, seg=seg)
    meta = json.loads(text.split("\n", 2)[1][5:])
    assert meta.get("out_row_ports") == runs == ([len(out_words)] if seg or len(out_words) > 1 else None)
    assert FC.digest(text) == sha


TOY_EXTRACTOR = r"""import Proofs.Flock.Verifier
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
    build_proofs_verifier()
    f = tmp_path / "Toy.lean"
    f.write_text(TOY_EXTRACTOR)
    p = subprocess.run(["lake", "env", "lean", str(f)], cwd=PROOFS, capture_output=True, text=True)
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
