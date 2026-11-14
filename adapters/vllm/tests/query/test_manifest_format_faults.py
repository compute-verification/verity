"""What the manifest format names as a fault, each beside the input that is whole: a batch-level launch's committed
stream (`batch_launch_shortfall`), the table of promoted rows (`promoted_observation`) and a rank merge's peer binding
(`tp_peer_binding`); and the FA2 launch geometry a step's attention rows imply (`fa2_geometry`), whose name
`hidden_stream_geometry` reads back.
"""
from __future__ import annotations

from typing import Any

import pytest

from verity_vllm.query.manifest import format as MF
from verity_vllm.query.required import collective_kind

GEOMETRY = {"HB": 8, "M": 3, "NB": 2, "BN": 128, "D": 64, "n_seq": 2}
WHOLE = MF.mat_total_words(8, 3, 2, 128, 64)


def _planes(rows: list[int], *, h: int = 4, nb: int = 2) -> list[tuple[str, int]]:
    """The planes of one sequence per entry of `rows`, `h` slabs each, every class at the words its geometry declares."""
    return [(f"s{i}/g{h}x{sl}x{nb}/{c}", h * sl * w) for i, sl in enumerate(rows) for c, w in MF.hidden_stream_widths(nb, 128, 64).items()]


def _renamed(planes: list[tuple[str, int]], plane: str, name: str) -> list[tuple[str, int]]:
    return [(name if member == plane else member, words) for member, words in planes]


@pytest.mark.parametrize(("geometry", "active", "members", "why"), [
    (GEOMETRY, [3, 1], _planes([3, 1]), None),
    (GEOMETRY, [3, 1], [("g8x3x2", WHOLE)], None),
    (None, [3, 1], _planes([3, 1]), "the batch-level row carries no launch geometry"),
    ([8, 3, 2, 128, 64, 2], [3, 1], _planes([3, 1]), "the batch-level row carries no launch geometry"),
    ({k: v for k, v in GEOMETRY.items() if k != "n_seq"}, [3, 1], _planes([3, 1]), "the batch-level row carries no launch geometry"),
    ({**GEOMETRY, "n_seq": 3}, [3, 1], _planes([3, 1]), "launch HB 8 is not 3 sequences of slabs"),
    ({**GEOMETRY, "n_seq": 0}, [3, 1], _planes([3, 1]), "launch HB 8 is not 0 sequences of slabs"),
    (GEOMETRY, [], _planes([3, 1]), "0 sequences executing, the launch is declared over 2"),
    (GEOMETRY, [3, 1, 1], _planes([3, 1, 1]), "3 sequences executing, the launch is declared over 2"),
    (GEOMETRY, [3, 1], [("g8x3x2", WHOLE), ("g8x4x2", MF.mat_total_words(8, 4, 2, 128, 64))],
     "2 whole launches committed: ['g8x3x2', 'g8x4x2']"),
    (GEOMETRY, [3, 1], [], "no M1 stream tensor committed"),
    (GEOMETRY, [3, 1], [("s0/g4x3x2/bot", 1), ("s1/g4x1x2/bot", 1)], "no M1 stream tensor committed"),
    (GEOMETRY, [3, 1], [("8x3x2", WHOLE), ("s0/4x3x2/S", 4 * 3 * 256), ("s0/g4x3x2/T", 1)], "no M1 stream tensor committed"),
    (GEOMETRY, [3, 1], _renamed(_planes([3, 1]), "s0/g4x3x2/P", "s0/g4x4x2/P"), "s0: its planes name 2 geometries"),
    (GEOMETRY, [3, 1], [(m, w - (m == "s0/g4x3x2/ROW")) for m, w in _planes([3, 1])], "s0/ROW: 191 words committed, g4x3x2 declares 192"),
    (GEOMETRY, [3, 1], _planes([3]) + [(m.replace("s0/", "s1/"), w) for m, w in _planes([1], h=2)],
     "the sequences of one launch name 2 different (slabs, NB, BN, D)"),
    (GEOMETRY, [3, 1], _planes([3, 2]), "sequence rows [2, 3] committed, [1, 3] scheduled"),
    (GEOMETRY, [1, 3], _planes([3, 1]), None),
])
def test_what_a_batch_launchs_committed_stream_lacks_is_named(geometry, active, members, why):
    assert MF.batch_launch_shortfall(geometry, active, members) == why


@pytest.mark.parametrize(("args", "windowed", "launch"), [
    ((64, 128, 4, 4, 64, 2), False, (8, 64, 1, 128, "g8x64x1")),             # a prefill of two sequences: every head a slab of its rows
    ((64, 6, 8, 4, 64, 2), False, (16, 3, 1, 128, "g16x3x1")),               # ... of grouped heads too: the fold is a decode's
    ((129, 1, 8, 4, 64, 1), False, (4, 2, 2, 128, "g4x2x2")),                # a GQA decode: a slab per KV head, its query heads the rows
    ((129, 1, 8, 4, 64, 1), True, (8, 1, 2, 128, "g8x1x2")),                 # ... never under a sliding window
    ((129, 2, 8, 4, 64, 2), False, (8, 2, 2, 128, "g8x2x2")),                # ... and per sequence
    ((129, 1, 4, 4, 64, 1), False, (4, 1, 2, 128, "g4x1x2")),                # as many KV heads as query heads: no group to fold
    ((65, 1, 8, 4, 128, 1), False, (4, 2, 2, 64, "g4x2x2x64x128")),          # a head wider than 64: key blocks of 64, named with the tile
    ((128, 3, 2, 2, 32, 1), False, (2, 3, 1, 128, "g2x3x1x128x32")),
    ((7, 5, 2, 2, 64, 0), False, (0, 5, 1, 128, "g0x5x1")),
])
def test_the_fa2_launch_a_steps_attention_rows_imply_and_its_name_read_back(args, windowed, launch):
    assert MF.fa2_geometry(*args, windowed=windowed) == launch
    HB, M, NB, BN, name = launch
    assert MF.hidden_stream_geometry(name) == (HB, M, NB, BN, args[4])
    for no_name in (name[1:], name + "x", name + "x64", f"s0/{name}", name.upper(), ""):
        assert MF.hidden_stream_geometry(no_name) is None, no_name


ROW: dict[str, Any] = {"family": "instance_outputs", "op_path": "m.a", "output_member": "out", "promoted": True, "consumers": ["m.b", "m.c"],
                       "width": 8, "dtype": "bf16", "spec": "Add_v1"}
PRODUCER = {"consumer": "m.b", "width": 8, "dtype": "bf16", "spec": "Add_v1", "rows_n": 1, "member": "out"}


def _observed(*rows: dict[str, Any]) -> tuple[dict, dict, dict, list]:
    got = MF.promoted_observation({"identities": list(rows)})
    assert got["rule"] == MF.PROMOTED_OBSERVATION_RULE and set(got) == {"names", "consumers", "producers", "problems", "rule"}
    return got["names"], got["consumers"], got["producers"], got["problems"]


def test_a_promoted_row_is_observed_at_its_first_consumers_input():
    plain = {"family": "instance_outputs", "op_path": "m.c", "output_member": "out", "consumers": ["m.b"]}
    assert _observed(plain) == ({}, {}, {}, [])
    assert _observed(ROW, plain) == ({"m.b/in": ["m.a", "out"]}, {"m.b": 16}, {"m.a": PRODUCER}, [])
    assert _observed(ROW, ROW, {**ROW, "consumers": ["m.b"]}) == ({"m.b/in": ["m.a", "out"]}, {"m.b": 16}, {"m.a": {**PRODUCER, "rows_n": 3}}, [])
    wide = {**ROW, "op_path": "m.w", "consumers": ["m.x"], "width": 3, "dtype": "f32"}
    unnamed = {k: v for k, v in wide.items() if k != "output_member"}
    for row in (wide, unnamed):
        assert _observed(ROW, row) == ({"m.b/in": ["m.a", "out"], "m.x/in": ["m.w", "out"]}, {"m.b": 16, "m.x": 12},
                                       {"m.a": PRODUCER, "m.w": {**PRODUCER, "consumer": "m.x", "width": 3, "dtype": "f32"}}, [])


@pytest.mark.parametrize(("rows", "problems", "bound"), [
    ([{**ROW, "consumers": []}], ["promoted row m.a/out (spec Add_v1) names no consumer: no observation point"], False),
    ([{**ROW, "consumers": None}], ["promoted row m.a/out (spec Add_v1) names no consumer: no observation point"], False),
    ([{k: v for k, v in ROW.items() if k != "consumers"}], ["promoted row m.a/out (spec Add_v1) names no consumer: no observation point"], False),
    ([{**ROW, "width": 0}], ["promoted row m.a/out: width 0 / dtype 'bf16' not usable for the arena bound (PROMOTED_DTYPE_BYTES ['bf16', 'f32'])"], False),
    ([{**ROW, "width": True}], ["promoted row m.a/out: width True / dtype 'bf16' not usable for the arena bound (PROMOTED_DTYPE_BYTES ['bf16', 'f32'])"], False),
    ([{**ROW, "width": "8"}], ["promoted row m.a/out: width '8' / dtype 'bf16' not usable for the arena bound (PROMOTED_DTYPE_BYTES ['bf16', 'f32'])"], False),
    ([{k: v for k, v in ROW.items() if k != "width"}],
     ["promoted row m.a/out: width None / dtype 'bf16' not usable for the arena bound (PROMOTED_DTYPE_BYTES ['bf16', 'f32'])"], False),
    ([{**ROW, "dtype": "w8"}], ["promoted row m.a/out: width 8 / dtype 'w8' not usable for the arena bound (PROMOTED_DTYPE_BYTES ['bf16', 'f32'])"], False),
    ([ROW, {**ROW, "consumers": ["m.c", "m.b"]}],
     ["promoted rows of m.a/out name different first consumers ('m.b' vs 'm.c'): one observation point per member"], True),
    ([ROW, {**ROW, "output_member": "0"}], ["promoted rows of m.a name different members ('out' vs '0')"], True),
    ([ROW, {**ROW, "width": 4}], ["promoted rows of m.a/out disagree on the signature (8 bf16 vs 4 bf16)"], True),
    ([ROW, {**ROW, "dtype": "f32"}], ["promoted rows of m.a/out disagree on the signature (8 bf16 vs 8 f32)"], True),
    ([ROW, {"family": "input_token_ids", "op_path": "m.b", "output_member": "in"}],
     ["promoted m.a/out: its first consumer's input m.b/in is itself a manifest identity (m.b, in)"], False),
    ([{**ROW, "op_path": "m.z"}, ROW], ["promoted m.z/out and m.a/out share the first consumer m.b: one committed m.b/in cannot bind two identities"], True),
])
def test_a_table_of_promoted_rows_that_gives_no_one_observation_point_is_named(rows, problems, bound):
    names, consumers, _producers, got = _observed(*rows)
    assert got == problems
    assert (names, consumers) == (({"m.b/in": ["m.a", "out"]}, {"m.b": 16}) if bound else ({}, {}))


def _rank(rank: int, *, spec: str = "AllReduce2_v1", out: int | None = 8, part: int | None = 8, member: str = MF.TP_PARTIAL_MEMBER,
          output: bool = True) -> dict[str, Any]:
    """Rank `rank`'s manifest of two: one collective's output of `out` elements and its rank-local partial of `part`."""
    def row(family: str, output_member: str, width: int | None, **more: Any) -> dict[str, Any]:
        return {"family": family, "request_id": "r0", "step": 0, "invocation": 0, "op_path": "m.x", "output_member": output_member, "rank": rank,
                **({} if width is None else {"element_range": [0, width]}), **more}

    return {"ranks": 2, "rank": rank, "identities": [
        row("weights", "weight", 8), *([row("instance_outputs", "out", out, spec=spec)] if output else []),
        row("instance_outputs", "plain", 8, spec="Gemm_v1"), row("tp_rank_partials", member.format(rank=rank), part)]}


def _binding(*parts: dict[str, Any]) -> tuple[int, int, int, list[str]]:
    got = MF.tp_peer_binding(list(parts), collective_kind)
    assert (got["world"], got["peer_partials"], got["rule"], got["n_unbound"]) == (2, 2, MF.TP_PEER_BINDING_RULE, len(got["unbound"]))
    return got["collective_outputs"], got["bound"], got["width_checked"], got["unbound"]


def test_a_collective_output_is_bound_to_each_peers_partial_of_the_width_its_kind_gives():
    assert _binding(_rank(0), _rank(1)) == (2, 2, 2, [])
    assert _binding(_rank(0, out=None), _rank(1, part=None)) == (2, 2, 1, []), "a pair either side of which names no range is bound unsized"
    assert _binding(_rank(0, out=None, part=4), _rank(1, out=None, part=2)) == (2, 2, 0, [])
    assert _binding(_rank(0), _rank(1, part=4)) == (
        2, 1, 2, ["rank 0 m.x step 0 all_reduce output width 8: part_rank1 on rank 1 is 4 elements, not 8 (request r0)"])
    gather = {"spec": "AllGather2_v1", "member": MF.TP_SHARD_MEMBER}
    assert _binding(_rank(0, part=4, **gather), _rank(1, part=4, **gather)) == (2, 2, 2, [])
    assert _binding(_rank(0, part=4, **gather), _rank(1, **gather)) == (
        2, 1, 2, ["rank 0 m.x step 0 all_gather output width 8: shard_rank1 on rank 1 is 8 elements, not 4 (request r0)"])
    assert _binding(_rank(0, out=7, part=4, **gather), _rank(1, out=7, part=3, **gather)) == (
        2, 0, 2, ["rank 0 m.x step 0 all_gather output width 7: shard_rank1 on rank 1 is 3 elements, not 7/2 (request r0)",
                  "rank 1 m.x step 0 all_gather output width 7: shard_rank0 on rank 0 is 4 elements, not 7/2 (request r0)"])


def test_a_partial_whose_peer_carries_no_collective_output_is_named():
    assert _binding(_rank(0), _rank(1, output=False)) == (
        1, 1, 1, ["rank 0 m.x step 0 partial part_rank0: no collective output identity on rank 1 (request r0)"])
    assert _binding(_rank(0, output=False), _rank(1, output=False)) == (
        0, 0, 0, ["rank 0 m.x step 0 partial part_rank0: no collective output identity on rank 1 (request r0)",
                  "rank 1 m.x step 0 partial part_rank1: no collective output identity on rank 0 (request r0)"])
