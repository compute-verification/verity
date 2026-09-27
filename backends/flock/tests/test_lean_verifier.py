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


STANDARD_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}


def check_axioms(root, target):
    subprocess.run(["lake", "build", target], cwd=root, check=True, capture_output=True)
    out = subprocess.run(["lake", "env", "lean", "CheckAxioms.lean"], cwd=root, check=True, capture_output=True, text=True).stdout
    lines = [l for l in out.splitlines() if "depends on axioms" in l or "does not depend on any axioms" in l]
    assert lines, out
    for l in lines:
        used = set(re.findall(r"[\w.]+", l.split(":", 1)[1])) if "depends on axioms" in l else set()
        assert used <= STANDARD_AXIOMS, l


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_proofs_use_only_standard_axioms():
    check_axioms(LEAN, "FlockProofs")


LEVEL3 = LEAN / "level3"


@pytest.mark.skipif(shutil.which("lake") is None or not (LEVEL3 / ".lake" / "packages" / "mathlib" / ".lake" / "build").is_dir(),
                    reason="level3 needs Mathlib's build cache (`lake exe cache get` in lean/level3)")
def test_field_proofs_use_only_standard_axioms():
    assert (LEVEL3 / "lean-toolchain").read_text().strip() == TOOLCHAIN
    check_axioms(LEVEL3, "FlockLevel3")


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
