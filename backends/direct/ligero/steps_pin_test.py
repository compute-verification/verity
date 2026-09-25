"""steps-pin (red-team-leaf-3 H1 / H2): the Python verifier binds a chain statement's VU shape to the relation.  steps
(columns per VU) must be the relation's canonical value (``Relation.steps``, fp4 ``STEPS``, vu.py ``STEPS``) and never
exceed the leaf's structural bound (Ajtai: the ring degree n); a v5 / v6 header's K must be K_VU = 1536 (what every hashed writer emits).  The twin
of ligero-verify's ``check_vu_shape`` (tests/relations.rs ``steps_pin_*`` / ``a_statement_whose_vu_shape_*``).

Must-reject: the red-team collide pairs (n64 steps 96, n128 steps 192) and n64 steps 32; real steps = 32 proofs of the
bare and the +hash fp8-ada relation (``redteam/steps_pin_forge.py``); header rewrites of every family's steps and a v5
K; a real steps = 32 +shared (v6) pair and v6 header rewrites, built on the CPU (through ``$LIGERO_VERIFY`` too when it
is set).  Honest: every committed fixture still verifies."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("blake3", exc_type=ImportError)

from .chain import Layout  # noqa: E402
from .protocol import BF16_HOOKS, RelationHooks, layout_error  # noqa: E402
from .relations import RELATIONS, relation  # noqa: E402
from .serialize import verify_files  # noqa: E402

FX = Path(__file__).resolve().parents[2] / "ligero-verify" / "fixtures"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
STEPS_AT, K_AT = 50, 54          # the statement header's u32 steps / K (v2, v4 and v5 alike)


def _files(d: str) -> tuple[Path, Path]:
    if d == "b1":
        return FX / "b1.stmt", FX / "b1.proof"
    return FX / d / "sub_00.stmt", FX / d / "sub_00.proof"


def _shape(steps: int, canonical: int) -> str:
    return f"statement: steps = {steps} columns per VU, the relation's VU is {canonical} columns"


def _bound(steps: int, n: int) -> str:
    return f"statement: steps = {steps} columns per VU exceeds the leaf's bound {n}"


def test_layout_error_is_the_rust_check() -> None:
    rel = RelationHooks(None, steps=48, max_steps=64)
    assert layout_error(Layout(1024, 2, 48), rel) is None
    assert layout_error(Layout(1024, 2, 32), rel).startswith(_shape(32, 48))
    assert layout_error(Layout(1024, 2, 96), rel).startswith(_bound(96, 64))            # the leaf bound is checked first
    assert layout_error(Layout(1024, 2, 96), RelationHooks(None, max_steps=64)).startswith(_bound(96, 64))
    assert layout_error(Layout(1024, 2, 32), RelationHooks(None, max_steps=64)) is None
    assert layout_error(Layout(1024, 2, 7), BF16_HOOKS) is None                         # unit-mode / test systems


def test_every_verifier_runner_carries_its_relations_vu_shape() -> None:
    from .fp4.chain import FP4_HOOKS
    from .fp4.relation import STEPS as FP4_STEPS
    from .leaf import registry as leaf_registry
    from .vu import STEPS as BF16_STEPS
    from .vu import ChainRunner
    for name, rel in RELATIONS.items():
        assert rel.hooks.steps == rel.steps == 1536 // rel.k, name
        assert rel.hooks.max_steps is None, name
    assert FP4_HOOKS.steps == FP4_STEPS == 24
    assert ChainRunner("cpu", -128.0).hooks.steps == BF16_STEPS == 96
    assert leaf_registry.get("ajtai-n64").max_steps == 64 and leaf_registry.get("ajtai-n128").max_steps == 128
    assert getattr(leaf_registry.get("poseidon2"), "max_steps", None) is None


RED_TEAM = [
    ("steps-pin/fp8-ada-ajtai-n64-collide-P", _bound(96, 64)),
    ("steps-pin/fp8-ada-ajtai-n64-collide-Q", _bound(96, 64)),
    ("steps-pin/bf16-hopper-ajtai-n128-collide-P", _bound(192, 128)),
    ("steps-pin/bf16-hopper-ajtai-n128-collide-Q", _bound(192, 128)),
    ("steps-pin/fp8-ada-ajtai-n64-steps32", _shape(32, 48)),
    ("steps-pin/fp8-ada-bare-steps32", _shape(32, 48)),
    ("steps-pin/fp8-ada-poseidon2-steps32", _shape(32, 48)),
]


@pytest.mark.parametrize("d,why", RED_TEAM, ids=[d.split("/")[1] for d, _ in RED_TEAM])
def test_the_red_team_and_forged_steps_proofs_are_refused(d: str, why: str) -> None:
    sp, pp = _files(d)
    if not sp.exists():
        pytest.skip(f"{d} not in this tree")
    ok, got, _ = verify_files(sp, pp, DEV)
    assert not ok and got.startswith(why), got


def test_the_collide_pairs_are_two_statements_over_one_pair_of_roots() -> None:
    for pair in ("fp8-ada-ajtai-n64-collide", "bf16-hopper-ajtai-n128-collide"):
        side_p, side_q = json.loads((FX / "steps-pin" / f"{pair}-P" / "collide.json").read_text())
        assert side_p["py"] and side_q["py"] and side_p["pinned"] and side_q["pinned"]      # both accepted before the fix
        p, q = (_files(f"steps-pin/{pair}-{s}")[0].read_bytes() for s in "PQ")
        assert p != q and len(p) == len(q)
        assert side_p["roots"]["y"] != side_q["roots"]["y"]
        for tree in ("a", "b"):
            root = bytes.fromhex(side_p["roots"][tree])                                       # the recorded 16-byte prefix
            assert side_q["roots"][tree] == side_p["roots"][tree] and root in p and root in q


def _rewrite(tmp: Path, d: str, edits: list[tuple[int, int]]) -> tuple[Path, Path]:
    sp, pp = _files(d)
    raw = bytearray(sp.read_bytes())
    for at, v in edits:
        raw[at:at + 4] = int(v).to_bytes(4, "little")
    out = tmp / (d.replace("/", "_") + ".stmt")
    out.write_bytes(bytes(raw))
    return out, pp


HEADER = [
    ("fp8-ada", [(STEPS_AT, 32)], _shape(32, 48)),
    ("b1", [(STEPS_AT, 32)], _shape(32, 96)),
    ("fp4-nvf4", [(STEPS_AT, 12)], _shape(12, 24)),
    ("fp8-ada-hash", [(STEPS_AT, 32), (K_AT, 32 * 32)], _shape(32, 48)),
    ("bf16-ampere-hash", [(STEPS_AT, 32), (K_AT, 32 * 16)], _shape(32, 96)),
    ("fp8-ada-hash", [(STEPS_AT, 32)], _shape(32, 48)),
    ("fp8-ada-hash", [(K_AT, 48 * 16)], "statement: K = 768 words per row, a hashed statement's row is K = 1536"),
    ("fp4-nvf4-hash", [(STEPS_AT, 12), (K_AT, 12 * 64)], _shape(12, 24)),
    ("fp4-nvf4-hash", [(K_AT, 24 * 68)], "statement: K = 1632 words per row, a hashed statement's row is K = 1536"),
    ("fp8-ada-ajtai-n64", [(STEPS_AT, 96), (K_AT, 96 * 32)], _bound(96, 64)),
]


@pytest.mark.parametrize("d,edits,why", HEADER, ids=[f"{d}-{'-'.join(str(v) for _, v in e)}" for d, e, _ in HEADER])
def test_a_header_with_another_vu_shape_is_refused_by_the_shape_check(tmp_path, d, edits, why) -> None:
    sp, pp = _rewrite(tmp_path, d, edits)
    ok, got, _ = verify_files(sp, pp, DEV)
    assert not ok and got.startswith(why), got


HONEST = ["b1", "fp8-ada", "fp4-nvf4", "fp4-nvf4-hash", "fp8-ada-hash", "bf16-ampere-hash", "fp8-ada-ajtai-n64", "bf16-hopper-ajtai-n128"]


@pytest.mark.parametrize("d", HONEST)
def test_every_honest_fixture_still_verifies(d: str) -> None:
    sp, pp = _files(d)
    ok, why, _ = verify_files(sp, pp, DEV)
    assert ok, why


def test_relation_registry_steps_are_the_rust_pins() -> None:
    # relation.rs's Relation.steps for the bare Table 2 relations (vu_shapes_are_the_python_registry_and_match_the_decodes)
    want = {"bf16-hopper": 96, "fp8-hopper": 48, "fp8-ada": 48, "bf16-ampere": 96}
    for name, steps in want.items():
        if name in RELATIONS:
            assert relation(name).steps == steps, name


# ---- +shared (v6): both sides of the pair carry the relation's VU shape (SharedHashedRunner.hooks / .hooks_h; the Rust
# pair verifier runs check_vu_shape on G and H).  fp8-ada, 2x2 tile, 4 VUs, Fiat-Shamir + ZK, on the CPU as
# redteam/leaf2_share_pair.py builds them; the steps = 32 pair is the honest prover with K_VU bumped (steps_pin_forge.py).

def _shared_pair(root: Path, name: str, steps: int | None = None) -> tuple[Path, object]:
    from . import relations as RM
    from . import relchain as RC
    from .relchain import SharedHashedRunner, tile_instances
    from .serialize import _statement_of, proof_bytes, statement_bytes, system_bytes
    rel = relation("fp8-ada")
    k_vu = RM.K_VU
    try:
        if steps is not None:
            RM.K_VU = RC.K_VU = steps * rel.k
        R = SharedHashedRunner(rel, "cpu", -128.0, zk=True, mode="fiat-shamir")
        data, x_rows, w_cols = tile_instances(rel, 2, 2)
        R.commit_vus(data, tile=(2, 2), x_rows=x_rows, w_cols=w_cols)
        pf, pubs, lay = R.prove_vus(data[:4], check=False, vu_ids=range(4))
        st = _statement_of(R, pubs, lay, R.cfg(lay.l), 1)
        d = root / name
        d.mkdir()
        (d / "sub_00.stmt").write_bytes(statement_bytes(st))
        (d / "sub_00.proof").write_bytes(proof_bytes(pf.g))
        (d / "sub_00.hproof").write_bytes(proof_bytes(pf.h))
        if steps is None:
            (root / "system.bin").write_bytes(system_bytes(R.sys))
            (root / "system_h.bin").write_bytes(system_bytes(R.sys_h))
    finally:
        RM.K_VU = RC.K_VU = k_vu
    return d, R


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    root = tmp_path_factory.mktemp("shared")
    honest, R = _shared_pair(root, "honest")
    forged, _ = _shared_pair(root, "steps32", steps=32)
    return root, honest, forged, R


def _rust_pair(root: Path, stmt: Path, d: Path) -> dict | None:
    """``ligero-verify verify --system-h`` on the pinned (canonical) G / H systems; None without ``$LIGERO_VERIFY``."""
    exe = os.environ.get("LIGERO_VERIFY")
    if not exe or not Path(exe).is_file():
        return None
    r = subprocess.run([exe, "verify", "--system", str(root / "system.bin"), "--system-h", str(root / "system_h.bin"),
                        "--statement", str(stmt), "--proof", str(d / "sub_00.proof"), "--proof-h", str(d / "sub_00.hproof"),
                        "--soundness-bits", "100"], capture_output=True, text=True)
    return json.loads(r.stdout.strip().splitlines()[-1])


def _as_pair(root: Path, d: Path, edits: list[tuple[int, int]]) -> Path:
    """The pair ``d`` with header words rewritten, as ``root/<name>/sub_00.{stmt,proof,hproof}`` (the hash-side proof is
    found beside the statement)."""
    out = root / f"{d.name}-{'-'.join(f'{a}_{v}' for a, v in edits)}"
    out.mkdir(exist_ok=True)
    raw = bytearray((d / "sub_00.stmt").read_bytes())
    for at, v in edits:
        raw[at:at + 4] = int(v).to_bytes(4, "little")
    (out / "sub_00.stmt").write_bytes(bytes(raw))
    for f in ("sub_00.proof", "sub_00.hproof"):
        (out / f).write_bytes((d / f).read_bytes())
    return out


def test_the_shared_runner_carries_the_relations_vu_shape(shared) -> None:
    *_, R = shared
    assert R.hooks.steps == R.hooks_h.steps == relation("fp8-ada").steps == 48


def test_an_honest_shared_pair_verifies(shared) -> None:
    root, honest, _, _ = shared
    ok, why, _ = verify_files(honest / "sub_00.stmt", honest / "sub_00.proof", DEV)
    assert ok, why
    js = _rust_pair(root, honest / "sub_00.stmt", honest)
    if js is not None:
        assert js["accepted"] and js["system_pinned"], js


def test_reverify_fails_a_shared_pair_closed(shared) -> None:
    from .reverify import commitment_problems
    root, honest, _, _ = shared
    man = {"set": {"total_vus": 4, "instances": "0" * 64}}
    assert commitment_problems(root, man, "fp8-ada+hash", [honest.name]) == (
        True, ["commitment: 1 row-sharing (v6) statement(s): shared-row layouts are not recomputed"])


SHARED_NEG = [
    ("steps32", [], _shape(32, 48), "G: statement: steps = 32 columns per VU, the fp8-ada relation's VU is 48 columns"),
    ("honest", [(STEPS_AT, 32)], _shape(32, 48), "G: statement: steps = 32 columns per VU, the fp8-ada relation's VU is 48 columns"),
    ("honest", [(K_AT, 48 * 16)], "statement: K = 768 words per row, a hashed statement's row is K = 1536",
     "G: statement: K = 768 words per row, a hashed statement's row is K = 1536"),
]


@pytest.mark.parametrize("which,edits,why,why_rust", SHARED_NEG, ids=["forged-steps32", "header-steps32", "header-K768"])
def test_a_shared_pair_with_another_vu_shape_is_refused(shared, which, edits, why, why_rust) -> None:
    root, honest, forged, _ = shared
    d = _as_pair(root, forged if which == "steps32" else honest, edits)
    ok, got, _ = verify_files(d / "sub_00.stmt", d / "sub_00.proof", DEV)
    assert not ok and got.startswith(why), got
    js = _rust_pair(root, d / "sub_00.stmt", d)
    if js is not None:
        assert not js["accepted"] and js["reason"].startswith(why_rust), js
