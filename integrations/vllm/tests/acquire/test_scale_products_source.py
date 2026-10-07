"""The scale-products source (`acquire.sources.scale_products_source`) on CPU tensors with a recording committer: its words are the IR's
(`ScaledMmFp8BlockSharedScale_v1` evaluated as a standalone Call, its committed interior gates `query.scale_products.slots` read in
order, NaN payloads, infinities and subnormals included); it commits `<linear module>/scale_products` only inside the manifest site during
a committed step and refuses a launch of another block geometry; `attach` refuses a manifest without the family, and `taps.attach`
attaches the source exactly when the manifest names it."""
from __future__ import annotations

import random
import types

import numpy as np
import pytest

from tests.acquire.test_vocab_range_source import Recorder
from tests.check.test_sampled_replay_v2_addresses import _v2_manifest
from tests.query.test_scale_products import fp8_program
from verity_vllm.acquire.sources import call_boundary_plan as CBP
from verity_vllm.program.registry import fp8
from verity_vllm.query import scale_products as SP

torch = pytest.importorskip("torch")

from verity_vllm.acquire.sources import scale_products_source as S  # noqa: E402
from verity_vllm.acquire.sources import taps  # noqa: E402

K, N, G = 256, 384, 128
KB, NB = K // G, N // G
#: x_s rows and the [N/G, K/G] w_s: NaNs of different payloads and signs, +-inf against zero, subnormal and overflowing products
XS = [[0x7FC00001, 0x00000001], [0xFFC00000, 0x7F800000], [0x3C000000, 0x3B124925]]
WS = [[0x7FA00000, 0x00400000], [0x80000000, 0x5F800000], [0x3B124925, 0xFFFFFFFF]]


def _f32(words) -> "torch.Tensor":
    return torch.from_numpy(np.asarray(words, dtype=np.uint32).view(np.int32)).view(torch.float32)


def _ir(xs: list[int], ws: list[list[int]]) -> list[int]:
    from verity.ml.circuits.evaluation.reference import standalone_call
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluate import evaluate_call

    prog, call = standalone_call(bind(fp8.ScaledMmFp8BlockSharedScale, K=K, N=N, G=G))
    t: dict[int, int] = {}
    evaluate_call(prog.circuit, call, [0] * K + xs + [0] * (N * K) + [w for row in ws for w in row], transcript=t)
    return [t[call.lo + j] for j in SP.slots(K, N, G)]


def test_the_products_are_the_irs_committed_words():
    got = S.products(torch, _f32(XS), _f32(WS).t())
    assert got.dtype == torch.float32 and tuple(got.shape) == (len(XS), NB * KB)
    words = got.view(torch.int32).numpy().view(np.uint32)
    for m, xs in enumerate(XS):
        assert words[m].tolist() == _ir(xs, WS), m


def _model():
    m = torch.nn.Module()
    m.down_proj = torch.nn.Identity()
    return m


def test_the_source_commits_only_inside_the_site_during_a_step():
    com = Recorder()
    src = S.ScaleProductsSource(com, _model(), {"down_proj": {"n_blocks": NB, "k_blocks": KB}})
    launches = []
    fn = src._wrap(lambda *a: launches.append(a) or None)
    rng = random.Random(0x5CA1E)
    xs = _f32([[rng.randrange(100, 127) << 23 | rng.getrandbits(23) for _ in range(KB)] for _ in range(2)])
    ws = _f32([[rng.randrange(100, 127) << 23 | rng.getrandbits(23) for _ in range(KB)] for _ in range(NB)]).t()
    args = (None, None, None, xs, ws)
    fn(*args)                                                   # outside a step
    enter, leave = src._enter("down_proj"), src._leave("down_proj")
    enter(None, ())
    fn(*args)                                                   # inside the site, no step
    com._cur = types.SimpleNamespace(step=0)
    fn(*args)
    leave(None, (), None)
    fn(*args)                                                   # a step, outside the site
    assert len(launches) == 4 and len(com.acquired) == 1
    name, t = com.acquired[0]
    assert name == f"down_proj/{SP.MEMBER}" and torch.equal(t.view(torch.int32), S.products(torch, xs, ws).view(torch.int32))
    enter(None, ())
    fn(None, None, None, xs[:, :1], ws[:1])                     # not the manifest's block geometry
    assert len(com.acquired) == 1 and len(com.errors) == 1 and len(launches) == 5
    assert src.extra_step_bytes({}, None)["per_token_bytes"] == 4 * NB * KB


def test_attach_refuses_a_manifest_without_the_family_and_taps_attach_it_when_named():
    com = Recorder()
    assert S.attach(com, False, {}, ("a",)) == (("a",), None)
    with pytest.raises(SystemExit):
        S.attach(com, True, {"identities": []}, ("a",))
    plain, shared = (_v2_manifest(fp8_program(s), taps=(SP.POLICY,)) for s in (False, True))
    # the quant's x_q/x_s are Q_word call boundaries, the call-boundary source's (tests/acquire/test_call_boundary_source.py)
    plain, shared = ({**m, "identities": [r for r in m["identities"] if r["family"] != CBP.FAMILY]} for m in (plain, shared))
    assert taps.attach(com, plain, ("a",), taps.TapSources()) == ("a",) and com.sources == []
    assert taps.attach(com, shared, ("a",), taps.TapSources()) == ("a", SP.FAMILY)
    assert [s.sites for s in com.sources] == [SP.manifest_sites(shared)]
    assert taps.attach(com, shared, ("a",), taps.TapSources(), explicit_classes=True) == ("a",)
