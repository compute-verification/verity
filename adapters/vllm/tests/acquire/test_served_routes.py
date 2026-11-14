"""`verity_vllm.acquire.served.probe_routes` on CPU, against a stand-in for the GPU that computes one cuBLAS route's words.

The stand-in takes the place of what the probe asks of the device: `torch.mm(x, w.t(), out=y)` at a shape (M, K, N) is the twin
(`bf16_gemm_check.cublas_route_rows`) of the route the test gives that shape, and what the probe makes on the device or moves to
it stays on the host.  So the device computes exactly one route at each shape, and the test holds the probe: the products it asks
for, the words it reads of them and the route it names.  That a twin computes a GPU's words is not held here; a card's capture
shows it.

It catches a probe that names another route than the device's, at each kind of route a split count leaves; a product asked at
another shape than the one probed, where cuBLAS may take another route, or of an operand left on the host; and a route answered
rather than refused, with the words each route missed, for a device that no route models (a pin's aligned grid, a split count
past the candidates), for one that leaves its route in one row of one product, and where two routes compute every word; and a
shape the builder has no split count rows for (`bf16_gemm_check.splitk_count_probe`) that ends the served Match's command
(`pipeline.served_match`) with a traceback and no record, where it is a Build the command refuses."""
from __future__ import annotations

import ast
import json
from typing import Any

import numpy as np
import pytest
import torch
from verity.definitions.registry import bf16_gemm_check as BC

from tests.pipeline import test_served_match_main as MM
from verity_vllm.acquire import served as SV
from verity_vllm.acquire.inductor_templates import Unmodelled
from verity_vllm.commit import challenge as CH
from verity_vllm.pipeline import served_match as sm

Shape = tuple[int, int, int]
Route = dict[str, Any]

#: the device's route at each shape (M, K, N): one of each kind `cublas_routes_at` leaves
ROUTES: dict[Shape, Route] = {
    (4, 1024, 64): {"kernel": "splitk", "SPLIT_K": 3},
    (3, 512, 300): {"kernel": "splitk", "SPLIT_K": 2, "partial": "bf16"},
    (2, 2048, 128): {"kernel": "serialk", "SPLIT_K": 5},
    (5, 512, 64): {"kernel": "nosplit"},
    (1, 2048, 128): {"kernel": "gemv", "LANES": 16},
    (1, 1024, 64): {"kernel": "gemv", "LANES": 32},
}

#: routes whose twin `cublas_route_rows` has and `cublas_routes_at` never lists
UNNAMED: list[Route] = [
    {"kernel": "splitk", "SPLIT_K": 3, "partial": "bf16", "align": 8, "tile_k": 16},
    {"kernel": "serialk", "SPLIT_K": 3, "align": 8, "tile_k": 16},
    {"kernel": "splitk", "SPLIT_K": BC.SPLITK_CANDIDATES[-1] + 1},
]

#: shapes the builder has no split count rows for, each with the builder's reason: two counts that split K at the same
#: columns, two its rows do not tell apart, a count with no boundary a row can test, and more rows than W has
UNPROBED: dict[Shape, str] = {
    (4, 128, 64): "split count probe cannot separate [(7, 8)] at K 128",
    (1, 384, 64): "split count probe does not tell 5 from 16 apart at K 384 (f32)",
    (4, 32, 64): "split count probe has no usable boundary for 2 splits at K 32",
    (4, 4096, 8): "split count probe needs 10 rows at K 4096, only 8 available",
}

#: the shape the refusals are read at, and its route
SHAPE: Shape = (4, 1024, 64)
ROUTE = ROUTES[SHAPE]


def _words(t: torch.Tensor) -> np.ndarray:
    return t.contiguous().view(torch.int16).numpy().view(np.uint16)


class Gpu:
    """What the test gives the stand-in: a shape's route (`routes`) and where the device leaves it (`off`: the route of one row of
    a shape's n-th product, by (n, row)).  What the stand-in was asked: the shape of each product (`products`) and the words of
    its operands (`operands`: x, and W whole)."""

    def __init__(self) -> None:
        self.routes: dict[Shape, Route] = {}
        self.off: dict[tuple[int, int], Route] = {}
        self.products: list[Shape] = []
        self.operands: list[tuple[np.ndarray, np.ndarray]] = []
        self._storages: set[int] = set()

    def holds(self, t: torch.Tensor) -> torch.Tensor:
        self._storages.add(t.untyped_storage().data_ptr())
        return t

    def mm(self, x: torch.Tensor, wt: torch.Tensor, *, out: torch.Tensor) -> None:
        shape = (x.shape[0], x.shape[1], wt.shape[1])
        assert shape in self.routes, f"the probe asked the device for a product at {shape}, a shape it was not given to probe"
        assert tuple(out.shape) == (shape[0], shape[2])
        assert all(t.untyped_storage().data_ptr() in self._storages for t in (x, wt, out)), "an operand of the product is on the host"
        nth = self.products.count(shape)
        self.products.append(shape)
        xs, ws = _words(x), _words(wt.t())
        self.operands.append((xs.copy(), ws.copy()))
        y = np.ascontiguousarray(BC.cublas_route_rows(self.routes[shape], xs, ws))
        for (n, row), route in self.off.items():
            if n == nth:
                y[row] = BC.cublas_route_rows(route, xs[row:row + 1], ws)[0]
        out.copy_(torch.from_numpy(y.view(np.int16)).view(torch.bfloat16))


@pytest.fixture
def gpu(monkeypatch: pytest.MonkeyPatch) -> Gpu:
    """The stand-in in the GPU's place: its `mm`, and on the host what `zeros`, `empty` and `.cuda()` put on the device."""
    gpu = Gpu()

    def cuda(t: torch.Tensor) -> torch.Tensor:
        return gpu.holds(t)

    def on_host(make: Any) -> Any:
        def made(*a: Any, device: Any = None, **kw: Any) -> torch.Tensor:
            t = make(*a, **kw)
            return gpu.holds(t) if str(device).startswith("cuda") else t
        return made

    monkeypatch.setattr(torch.Tensor, "cuda", cuda)
    monkeypatch.setattr(torch, "zeros", on_host(torch.zeros))
    monkeypatch.setattr(torch, "empty", on_host(torch.empty))
    monkeypatch.setattr(torch, "mm", gpu.mm)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "a stand-in")
    return gpu


def _refused() -> dict[str, int]:
    """The probe of `SHAPE` refuses; what its refusal says each route missed, by the route's JSON."""
    with pytest.raises(Unmodelled) as why:
        SV.probe_routes([list(SHAPE)])
    said, head, tail = str(why.value), "cuBLAS at M=4 K=1024 N=64: words missed by each route ", ": no one route models it"
    assert said.startswith(head) and said.endswith(tail), said
    missed: dict[str, int] = ast.literal_eval(said[len(head):-len(tail)])
    return missed


def test_the_probe_names_the_devices_route_at_each_shape(gpu: Gpu) -> None:
    """Each kind of route is read back from the device's words alone, under its shape's key, beside what the device reports of
    itself; the probe asks three products a shape, each at the shape whole and of operands on the device: the builder's split
    count rows against rows of ones, then the spread and the cancelling rows the generator of the shape's seed gives, each
    product's rows at the top of a W that is zero below them."""
    gpu.routes.update(ROUTES)
    got = SV.probe_routes([list(shape) for shape in ROUTES])
    assert got == {"device": "a stand-in", "versions": {"torch": torch.__version__, "cuda": torch.version.cuda},
                   "routes": {",".join(map(str, shape)): route for shape, route in ROUTES.items()}}
    assert gpu.products == [shape for shape in ROUTES for _ in range(3)]
    asked = [pair for M, K, N in ROUTES for pair in [(np.full((M, K), 0x3F80, np.uint16), BC.splitk_count_probe(K, N)),
                                                     *BC.route_probe_rows(M, K, min(N, 256), CH.generator([M, K, N]))]]
    for (x, rows), (xs, ws) in zip(asked, gpu.operands, strict=True):
        assert np.array_equal(xs, x) and np.array_equal(ws[:len(rows)], rows) and not ws[len(rows):].any()


@pytest.mark.parametrize("route", UNNAMED, ids=json.dumps)
def test_a_device_on_a_route_the_probe_has_no_name_for_is_refused(gpu: Gpu, route: Route) -> None:
    """No route is answered for a device that none computes: the refusal names the shape and, for each route the split counts
    left, how many of the words read it missed, none of them zero."""
    gpu.routes[SHAPE] = route
    missed = _refused()
    assert [json.loads(named) for named in missed][:3] == BC.cublas_routes_at([])
    assert all(n > 0 for n in missed.values()), missed


@pytest.mark.parametrize("row", [0, SHAPE[0] // 2, SHAPE[0] - 1], ids=["first", "middle", "last"])
@pytest.mark.parametrize("product", [1, 2], ids=["spread", "cancelling"])
def test_a_device_off_its_route_in_one_row_of_one_product_is_refused(gpu: Gpu, product: int, row: int) -> None:
    """The route named computes every word read: both of a shape's products that tell the routes apart, at its first, middle
    and last rows.  A device that computes its route everywhere else is given no route."""
    gpu.routes[SHAPE] = ROUTE
    gpu.off[product, row] = {"kernel": "serialk", "SPLIT_K": ROUTE["SPLIT_K"]}
    missed = _refused()
    assert missed[json.dumps(ROUTE)] > 0 and all(n > 0 for n in missed.values()), missed


def test_two_routes_that_compute_every_word_are_refused(gpu: Gpu, monkeypatch: pytest.MonkeyPatch) -> None:
    """A split-K route is one function with its ``partial`` left to the default or named "f32".  Were both spellings left as
    candidates, each computing every word, the probe names neither."""
    spelled = {**ROUTE, "partial": "f32"}
    routes_at = BC.cublas_routes_at
    monkeypatch.setattr(BC, "cublas_routes_at", lambda fits: [*routes_at(fits), spelled])
    gpu.routes[SHAPE] = ROUTE
    assert [json.loads(named) for named, n in _refused().items() if n == 0] == [ROUTE, spelled]


@pytest.mark.parametrize(("shape", "reason"), UNPROBED.items(), ids=[str(shape[1]) for shape in UNPROBED])
def test_a_shape_the_builder_has_no_split_count_rows_for_is_refused_before_any_product(gpu: Gpu, shape: Shape, reason: str) -> None:
    """The refusal names the shape and gives the builder's reason, and is the builder's error raised on: the device is asked
    for nothing at that shape, after the three products of a shape before it that the probe could read."""
    gpu.routes.update({SHAPE: ROUTE, shape: {"kernel": "nosplit"}})
    with pytest.raises(Unmodelled) as why:
        SV.probe_routes([list(SHAPE), list(shape)])
    M, K, N = shape
    assert str(why.value) == f"cuBLAS at M={M} K={K} N={N}: {reason}"
    assert type(why.value.__cause__) is ValueError and str(why.value.__cause__) == reason
    assert gpu.products == [SHAPE] * 3


def test_a_match_whose_build_misses_such_a_shape_ends_as_a_build_refused_with_its_record(
        gpu: Gpu, monkeypatch: pytest.MonkeyPatch, tmp_path: Any, capsys: pytest.CaptureFixture[str]) -> None:
    """The served Match's command among its stand-ins (`test_served_match_main._Run`), with its own `build_here` and this
    probe: a Build that misses a route at such a shape is one the command refuses.  It prints the refusal, writes it into
    match.json with the run's facts, exits 3, and writes no routes and no Build."""
    shape, reason = next(iter(UNPROBED.items()))
    build_here = sm.build_here
    graph = tmp_path / "cache" / "backbone" / "computation_graph.py"
    graph.parent.mkdir(parents=True)
    graph.write_text("# the split graph\n")
    held = {"cache_dir": str(tmp_path / "cache"), "cudagraph_capture_sizes": [1], "max_num_seqs": 8}
    run = MM._Run(monkeypatch, tmp_path, facts={**MM.FACTS, **held})
    monkeypatch.setattr(run.source, "launch_configs", dict, raising=False)
    monkeypatch.setattr(sm, "build_here", build_here)
    monkeypatch.setattr(SV, "build", lambda inductor_dir, **kw: {"routes_missing": [list(shape)]})
    assert run.main() == 3
    refused = f"cuBLAS at M=4 K=128 N=64: {reason}"
    assert run.read("match.json") == {"ok": False, "build_refused": refused, "facts": {**run.facts(served=True), **held}}
    assert run.files() == ["launch_configs.json", "match.json"]
    assert capsys.readouterr().out == MM._lines(MM.ENGINE_LINE, MM.SERVED_LINE, f"Build refused: {refused}")
    assert gpu.products == []
