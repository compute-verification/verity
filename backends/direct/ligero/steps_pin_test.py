"""steps-pin (red-team-leaf-3 H1 / H2): the Python verifier binds a chain statement's VU shape to the relation.  steps
(columns per VU) must be the relation's canonical value (``Relation.steps``, fp4 ``STEPS``, vu.py ``STEPS``) and never
exceed the leaf's structural bound (Ajtai: the ring degree n); a v5 header's K must be K_VU = 1536 (what every hashed writer emits).  The twin
of ligero-verify's ``check_vu_shape`` (tests/relations.rs ``steps_pin_*`` / ``a_statement_whose_vu_shape_*``).

Must-reject: the red-team collide pairs (n64 steps 96, n128 steps 192) and n64 steps 32; real steps = 32 proofs of the
bare and the +hash fp8-ada relation (``redteam/steps_pin_forge.py``); header rewrites of every family's steps and a v5
K.  Honest: every committed fixture still verifies."""
from __future__ import annotations

import json
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
    ("fp8-ada-ajtai-n64", [(STEPS_AT, 96), (K_AT, 96 * 32)], _bound(96, 64)),
]


@pytest.mark.parametrize("d,edits,why", HEADER, ids=[f"{d}-{'-'.join(str(v) for _, v in e)}" for d, e, _ in HEADER])
def test_a_header_with_another_vu_shape_is_refused_by_the_shape_check(tmp_path, d, edits, why) -> None:
    sp, pp = _rewrite(tmp_path, d, edits)
    ok, got, _ = verify_files(sp, pp, DEV)
    assert not ok and got.startswith(why), got


HONEST = ["b1", "fp8-ada", "fp4-nvf4", "fp8-ada-hash", "bf16-ampere-hash", "fp8-ada-ajtai-n64", "bf16-hopper-ajtai-n128"]


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
