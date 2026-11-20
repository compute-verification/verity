"""The scale-products source (`acquire.sources.scale_products_source`) on CPU tensors with a recording committer: its words are the circuit's
(`ScaledMmFp8BlockSharedScale_v1` evaluated as a standalone Call, its committed interior gates `query.scale_products.slots` read in
order, NaN payloads, infinities and subnormals included); it commits `<linear module>/scale_products` only inside the plan's site during
a committed step and refuses a launch of another block geometry, the same product included; `attach` refuses a capture plan without the class or its rows, and
`taps.attach` attaches the source exactly when the plan names it."""
from __future__ import annotations

import random
import types
from typing import TYPE_CHECKING

import numpy as np
import pytest

from tests.acquire.recorder import Recorder
from verity_vllm.definitions.registry import fp8
from verity_vllm.query import scale_products as SP

if TYPE_CHECKING:
    import torch
else:
    torch = pytest.importorskip("torch")

from verity_vllm.acquire.sources import scale_products_source as S
from verity_vllm.acquire.sources import taps

K, N, G = 256, 384, 128
SHARED = "ScaledMmFp8BlockSharedScale_v1"
KB, NB = K // G, N // G
#: x_s rows and the [N/G, K/G] w_s: NaNs of different payloads and signs, +-inf against zero, subnormal and overflowing products
XS = [[0x7FC00001, 0x00000001], [0xFFC00000, 0x7F800000], [0x3C000000, 0x3B124925]]
WS = [[0x7FA00000, 0x00400000], [0x80000000, 0x5F800000], [0x3B124925, 0xFFFFFFFF]]


def _f32(words) -> torch.Tensor:
    return torch.from_numpy(np.asarray(words, dtype=np.uint32).view(np.int32)).view(torch.float32)


def _ir(xs: list[int], ws: list[list[int]]) -> list[int]:
    from verity.circuits.defs import bind
    from verity.circuits.evaluation import evaluate_call
    from verity.circuits.evaluation.reference import standalone_call

    prog, call = standalone_call(bind(fp8.ScaledMmFp8BlockSharedScale, K=K, N=N, G=G))
    t: dict[int, int] = {}
    evaluate_call(prog.circuit, call, [0] * K + xs + [0] * (N * K) + [w for row in ws for w in row], trace=t)
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
    src = S.ScaleProductsSource(com, _model(), {"down_proj": (NB, KB)})
    launches = []

    def launch(*a):
        launches.append(a)

    fn = src._wrap(launch)
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
    fn(None, None, None, xs[:, :1], ws[:1])                     # not the plan's block geometry
    fn(None, None, None, xs.repeat(1, 2)[:, :NB], ws.t())      # other blocks at the same product: [M, 3] x [3, 2]
    assert len(com.acquired) == 1 and len(com.errors) == 2 and len(launches) == 6
    assert src.extra_step_bytes({}, None)["per_token_bytes"] == 4 * NB * KB


def _plan(width=NB * KB, member=SP.MEMBER):
    """The plan classes (`capture_plan.of_build`) of a row with one block-scaled FP8 linear."""
    return [{"op_path": "down_proj", "member": "0", "capture": "module_output", "width": N, "dtype": "bf16", "definitions": [SHARED]},
            {"op_path": "down_proj", "member": member, "capture": "scale_products", "width": width, "dtype": "f32", "definitions": [SHARED]}]


def _rows(k=K, n=N, g=G):
    """The plan rows (`capture_plan.rows_of_build`) of that linear."""
    return [{"op_path": "down_proj", "statics": {"K": k, "N": n, "G": g}, "classes": {"0": "module_output", SP.MEMBER: "scale_products"}}]


def test_the_sites_are_the_plans_scale_products_classes_at_their_rows_blocks():
    assert S.sites(_plan(), _rows()) == {"down_proj": (NB, KB)} and S.sites(_plan()[:1], _rows()) == {}
    with pytest.raises(ValueError, match="down_proj/products is a scale_products class and not the member scale_products"):
        S.sites(_plan(member="products"), _rows())
    with pytest.raises(ValueError, match="rows give down_proj 0 block geometries"):
        S.sites(_plan(), [])
    with pytest.raises(ValueError, match="rows give down_proj 2 block geometries"):
        S.sites(_plan(), _rows() + _rows(k=N, n=K))
    with pytest.raises(ValueError, match="down_proj is 4 wide, and its rows' blocks are 3 x 2"):
        S.sites(_plan(width=4), _rows())


def test_attach_refuses_a_plan_without_the_class_or_its_rows_and_taps_attach_it_when_named():
    com = Recorder()
    com.model = _model()
    assert S.attach(com, False, [], ("a",)) == (("a",), None)
    for plan, rows in (([], _rows()), (_plan()[:1], _rows()), (_plan(member="products"), _rows()), (_plan(), None)):
        with pytest.raises(SystemExit) as e:
            S.attach(com, True, plan, ("a",), rows=rows)
        assert e.value.code == 3 and com.sources == []
    with pytest.raises(SystemExit):
        taps.attach(com, _plan(), ("a",))                       # the classes alone hold the product, not the blocks
    assert taps.attached(_plan()[:1]) == [] and taps.attach(com, _plan()[:1], ("a",)) == ("a",) and com.sources == []
    assert taps.attached(_plan()) == [S.ScaleProductsSource.short_name]
    assert taps.attach(com, _plan(), ("a",), rows=_rows()) == ("a", "scale_products")
    assert [s.sites for s in com.sources] == [{"down_proj": (NB, KB)}]
    assert taps.attach(com, _plan(), ("a",), rows=_rows(), explicit_classes=True) == ("a",)
