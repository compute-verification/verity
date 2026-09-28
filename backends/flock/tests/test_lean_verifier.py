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
    `FlockBatchedAcc.lean` (the expected-time extractor), `FlockLink.lean` (the link theorem) and `FlockLinked.lean` (the
    two composed). No proof is admitted."""
    files = sorted(AUDIT.glob("*.lean"))
    flock = {"Flock.lean", "FlockCompiled.lean", "FlockBatched.lean", "FlockBatchedAcc.lean", "FlockLink.lean",
             "FlockLinked.lean"}
    assert files and flock <= {p.name for p in files}
    bad = [f"{p.name}: {line}" for p in files if p.name not in flock for line in p.read_text().splitlines()
           if line.startswith("import ") and not AUDIT_IMPORTS.match(line)]
    assert not bad, bad
    admitted = [p.name for p in files if re.search(r"\b(sorry|admit)\b|^axiom ", p.read_text(), re.M)]
    assert not admitted, admitted


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


def _one_stage_draw():
    try:
        from verity_one_stage import draw
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
