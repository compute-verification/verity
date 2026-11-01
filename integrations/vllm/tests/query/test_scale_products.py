"""The scale-products policy (`verity_vllm.query.scale_products`): `ScaledMmFp8BlockSharedScale_v1` commits exactly its cdiv(N, G) * K/G
F32Mul products, a ragged N's partial block in `fp8_twins.scale_products`' order; on a small FP8 Program, one `scale_products` identity per (step, linear module) on the op path of the module's output; a
Program of `ScaledMmFp8Block_v1` (`fp8_scale_construction = per-coordinate`) gets none and a byte-identical manifest; the plan, coverage,
form (B), the Build and the word check know the family.  The source: tests/acquire/test_scale_products_source.py."""
from __future__ import annotations

import json
import types

import pytest

from tests.check.test_sampled_replay_fp8 import G, H, I, _program_fp8
from tests.check.test_sampled_replay_v2_addresses import _v2_manifest
from verity.ml.definitions.registry import fp8
from verity.ml.definitions.numerics import fp8_twins
from verity_vllm.query import scale_products as SP
from verity_vllm.query.manifest import format as fmt

GU, DN = "model.layers.0.mlp.gate_up_proj", "model.layers.0.mlp.down_proj"


def fp8_program(shared: bool) -> dict:
    """`_program_fp8`'s instance sequence, its GEMMs `ScaledMmFp8BlockSharedScale_v1` when `shared`."""
    inst, _woff, _wbits = _program_fp8()
    if shared:
        for r in inst["rows"]:
            r["spec"] = r["spec"].replace(f"{fp8.ScaledMmFp8Block.id}{{", f"{fp8.ScaledMmFp8BlockSharedScale.id}{{")
    return inst


@pytest.mark.parametrize("K,N", [(256, 256), (384, 128), (128, 512), (256, 192), (128, 64), (256, 320)])
def test_the_shared_tile_commits_exactly_its_products(K, N):
    assert len(SP.slots(K, N, 128)) == SP.words({"K": K, "N": N, "G": 128}) == -(-N // 128) * (K // 128)


def test_a_ragged_n_commits_its_partial_blocks_products_in_the_sources_order():
    """N = 320 = 2 * 128 + 64: the partial last weight block's K/G products are committed after the whole blocks', nb major, kb minor --
    word for word `fp8_twins.scale_products(x_s, w_s)` over the [cdiv(N, G), K/G] `weight_scale_inv`, the words the source acquires.
    GLM-5.3's `fused_qkv_a_proj` (K = 6144, N = 2624) is 21 x 48 of them per row."""
    import random

    import numpy as np

    from verity.ml.circuits.evaluation.reference import standalone_call
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluation import evaluate_call

    K, N, G = 256, 320, 128
    nb, kb = SP.blocks({"K": K, "N": N, "G": G})
    assert (nb, kb) == (3, 2) and SP.blocks({"K": 6144, "N": 2624, "G": G}) == (21, 48)
    rng = random.Random(0x2624)
    edge = [0x7FC00001, 0xFFC00000, 0x7F800000, 0x00000001, 0x80000000, 0x5F800000]
    xs = [rng.choice(edge) if rng.random() < 0.3 else rng.randrange(100, 127) << 23 | rng.getrandbits(23) for _ in range(kb)]
    ws = [[rng.choice(edge) if rng.random() < 0.3 else rng.randrange(100, 127) << 23 | rng.getrandbits(23) for _ in range(kb)]
          for _ in range(nb)]
    prog, call = standalone_call(bind(fp8.ScaledMmFp8BlockSharedScale, K=K, N=N, G=G))
    t: dict[int, int] = {}
    evaluate_call(prog.circuit, call, [0] * K + xs + [0] * (N * K) + [w for row in ws for w in row], trace=t)
    got = [t[call.lo + j] for j in SP.slots(K, N, G)]
    assert got == fp8_twins.scale_products(np.asarray(xs, dtype=np.uint32), np.asarray(ws, dtype=np.uint32)).reshape(-1).tolist()


def test_a_ragged_n_site_is_named_with_its_partial_block():
    from verity.ml.definitions import word as W_

    g = {"family": fp8.ScaledMmFp8BlockSharedScale.id, "statics": {"K": 256, "N": 320, "G": 128},
         "definition": f"{fp8.ScaledMmFp8BlockSharedScale.id}{{K=256,N=320,G=128}}", "calls": 3, "where": DN}
    assert W_.check_calls([g], acquired=SP.TAP_WORDS)["acquired_interior_words"] == 3 * 3 * 2
    rows = []
    pop = types.SimpleNamespace(P=types.SimpleNamespace(calls=[types.SimpleNamespace(id=i, family=g["family"], statics=g["statics"],
                                                                                     definition=g["definition"]) for i in range(3)]),
                                step_of=lambda i: 0, program_path=lambda i: DN)
    stats = SP.kernel_rows(pop, lambda *a, **kw: rows.append((a, kw)), lambda p: p, {})
    assert stats == {"words": 3 * 3 * 2, "rows_by_site": {DN: 3}}
    (a, kw), = rows
    assert a == (SP.FAMILY, 0, 0, DN, SP.MEMBER, 0, 18) and (kw["n_blocks"], kw["k_blocks"], kw["width"]) == (3, 2, 6)


def test_the_policy_names_the_products_of_every_linear_module_and_step():
    inst = fp8_program(True)
    off, on = _v2_manifest(inst, taps=()), _v2_manifest(inst, taps=(SP.POLICY,))
    rows = [x for x in on["identities"] if x["family"] == SP.FAMILY]
    dn, gu = (H // G) * (I // G), (2 * I // G) * (H // G)
    assert [(x["step"], x["op_path"], x["rows"], x["width"]) for x in rows] == [
        (0, DN, 2, dn), (0, GU, 2, gu), (1, DN, 1, dn), (1, GU, 1, gu), (2, DN, 1, dn), (2, GU, 1, gu)]
    assert all(x["output_member"] == SP.MEMBER and x["dtype"] == "f32" and x["element_range"] == [0, x["rows"] * x["width"]]
               and x["spec"] == fp8.ScaledMmFp8BlockSharedScale.id for x in rows)
    outs = {(x["op_path"], x["output_member"]) for x in on["identities"] if x["family"] == "instance_outputs" and x["op_path"] in (GU, DN)}
    assert outs == {(GU, "out"), (DN, "out")}, "the products sit on the op path of the linear's output"
    strip = lambda m: sorted(json.dumps(x, sort_keys=True) for x in m["identities"] if x["family"] != SP.FAMILY)  # noqa: E731
    assert strip(on) == strip(off) and on["complete"]
    assert SP.policy_of(on) and not SP.policy_of(off)
    assert on["query"]["scale_products"]["words"] == 4 * (dn + gu) and on["query"]["required_families"][-1] == SP.FAMILY
    assert SP.manifest_sites(on) == {GU: {"n_blocks": 2 * I // G, "k_blocks": H // G}, DN: {"n_blocks": H // G, "k_blocks": I // G}}


def test_a_program_without_the_shared_tile_is_unchanged():
    inst = fp8_program(False)
    on = _v2_manifest(inst, taps=(SP.POLICY,))
    assert json.dumps(on, sort_keys=True) == json.dumps(_v2_manifest(inst, taps=()), sort_keys=True)
    assert not SP.policy_of(on) and SP.manifest_sites(on) == {}


def test_manifest_sites_refuses_two_geometries_at_one_site():
    man = _v2_manifest(fp8_program(True), taps=(SP.POLICY,))
    man["identities"].append(dict(next(x for x in man["identities"] if x["family"] == SP.FAMILY), k_blocks=3, step=7))
    with pytest.raises(ValueError, match="two block geometries"):
        SP.manifest_sites(man)


def test_the_family_is_in_every_vocabulary():
    from verity_vllm.acquire import plan
    from verity_vllm.check import oracle_compare as OC
    from verity_vllm.check.replay import coverage
    from verity_vllm.pipeline import manifest as M
    from verity.ml.definitions import word as W_

    assert SP.FAMILY in fmt.FAMILIES and plan.ROUTE_OF_FAMILY[SP.FAMILY] == "scale_tap" and SP.FAMILY in coverage.QUERY_FAMILY_MECHANISM
    assert M.DEFAULT_TAPS == (SP.POLICY,)
    entry = {"identity": {"op_path": DN, "output_member": SP.MEMBER}, "name": f"{DN}/{SP.MEMBER}"}
    assert OC._family_of(entry) == "hidden_stream"
    g = {"family": fp8.ScaledMmFp8BlockSharedScale.id, "statics": {"K": 256, "N": 384, "G": 128},
         "definition": f"{fp8.ScaledMmFp8BlockSharedScale.id}{{K=256,N=384,G=128}}", "calls": 3, "where": DN}
    assert W_.check_calls([g], acquired=SP.TAP_WORDS)["acquired_interior_words"] == 3 * 3 * 2
