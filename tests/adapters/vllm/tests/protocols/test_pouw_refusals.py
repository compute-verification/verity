"""The pouw option's refusals that no other CPU test of the adapter reaches, each held to its own words: `Settings.of`'s of
a transcript directory that is no string; `refusals`' of a tensor-parallel world, of a forward anchor under an executor
that counts its own, and of a schedule Pearl-C's path can't run; `_layers`' of a model with no linear layer and of two
layers that share a quant method; and H1's install's of a layout without its rows or its steps.  Pearl-C's of a scheme
its path doesn't run or that a test instance names otherwise, of a kernel directory without one cubin, of a pass whose coins don't parse or are no object, of a live salt over other weights than its workload's, of a group of
experts that repeat, are another group's, differ in shape or whose bytes are not the kernel's size for them, of a
grouped call under a schedule with other side steps, with counts that aren't its rows, past its pass's capacity or
under a CUDA graph capture, and of B̃'s prefetch and the
shared-memory split of pd_commit and pd_tree on a kernel without them, and the in-process certifier's of fewer coin
bytes than it asks for."""
from __future__ import annotations

import random
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity import hooks as H
from verity.protocols.resource_accounting.compute.pouw.prover.schemes.pearl_c_device import DEVICES
from verity.protocols.resource_accounting.compute.pouw.prover.serving import PEARL_C_SCHEMES, pearl_c_scheme
from verity.protocols.resource_accounting.compute.pouw.prover.serving import pearl_c as PPD
from verity.protocols.resource_accounting.compute.pouw.prover.serving.pearl_c import kernel as PK
from verity.protocols.resource_accounting.compute.pouw.prover.serving.pearl_c.lanes import (
    schedule_of,
    whole_options,
)
from verity.protocols.resource_accounting.compute.pouw.verifier import core_check as CC
from verity.protocols.resource_accounting.compute.pouw.verifier import salt as PS
from verity.protocols.resource_accounting.compute.pouw.verifier import window as Wd
from verity.protocols.resource_accounting.compute.pouw.verifier.audit import (
    Workload,
    salt_of,
    weights_root,
)
from verity.protocols.resource_accounting.compute.pouw.verifier.coins import Challenger
from verity.protocols.resource_accounting.compute.pouw.verifier.protocol import DomainError, Matmul
from verity.protocols.resource_accounting.compute.pouw.verifier.protocol import Shape as MatmulShape
from verity.protocols.resource_accounting.compute.pouw.verifier.serving import (
    SCHEDULES,
    Schedule,
    SchemeId,
)
from verity.protocols.proofs.spot.verifier.challenger import challenger as LC
from verity_torch.pouw.pearl_c import grouped as PG

from verity_vllm.protocols import pouw as P
from verity_vllm.protocols.interface import Context

H1 = P.H1_SCHEME


def _serve(scheme: str, world: int = 1, **settings) -> Context:
    return Context(stage="serve", enabled=(f"{P.NAME}:{scheme}",), world=world,
                   settings={P.NAME: {"coins": Challenger(0, random.Random(0).randbytes), **settings}})


@pytest.mark.parametrize("given", [3, Path("/t"), b"/t", ("/t",)])
def test_settings_refuse_a_transcript_directory_that_is_no_string(given):
    said = "pouw setting 'transcript_dir' is a path string"
    with pytest.raises(ValueError, match=f"^{re.escape(said)}$"):
        P.Settings.of({"transcript_dir": given})
    assert P.refusals(H1, _serve(H1, transcript_dir=given)) == (said,)


def test_settings_take_a_path_string_and_none_for_no_transcript_directory():
    assert P.Settings.of({"transcript_dir": "/t"}).transcript_dir == "/t"
    assert P.Settings.of({"transcript_dir": None}).transcript_dir is None
    assert P.Settings.of({}).transcript_dir is None
    assert P.refusals(H1, _serve(H1, transcript_dir="/t")) == P.refusals(H1, _serve(H1))


def test_a_tensor_parallel_world_is_refused():
    scheme = H1
    tp = _serve(scheme, world=2)
    assert P.refusals(scheme, _serve(scheme)) == P.refusals(scheme, tp)
    assert [r.why for _, r in H.unmet([(scheme, r) for r in P.requests(scheme, tp)], tp.properties())] == [
        "PoUW's units and transcript are per rank, which is not wired"]


def test_a_schedule_pearl_cs_path_cannot_run_is_refused_under_the_schemes_name(monkeypatch):
    """Every registered schedule runs under every Pearl-C scheme, so `schedule_of` refuses only once a schedule is
    registered that the path can't run.  One that takes the GEMM to a side lane stands in for it."""
    for scheme in PEARL_C_SCHEMES:
        for name, schedule in SCHEDULES.items():
            assert schedule_of(name, pearl_c_scheme(scheme)) is schedule
    monkeypatch.setitem(SCHEDULES, "side-gemm", Schedule("side-gemm", frozenset({"gemm"}), 2, "greatest", "least"))
    for scheme in PEARL_C_SCHEMES:
        said = (f"{scheme}: side-gemm: gemm can't leave the main lane under h3: a later step of the call reads what it "
                "writes; side-gemm: this path runs only a_tree, tile_hash on a side lane, not gemm")
        with pytest.raises(ValueError, match=f"^{re.escape(said)}$"):
            schedule_of("side-gemm", pearl_c_scheme(scheme))


def test_a_pearl_c_scheme_is_built_only_for_a_name_this_path_runs_and_only_under_that_name():
    """A test instance takes `PearlC`'s keywords; one of another device's prices names itself with them, and is refused."""
    with pytest.raises(KeyError) as e:
        PPD.refusals("pearl-c-nvfp4-v0", {})
    assert e.value.args == (f"'pearl-c-nvfp4-v0': this device path runs {', '.join(PEARL_C_SCHEMES)}",)
    for name in PEARL_C_SCHEMES:
        own = DEVICES[SchemeId.parse(name).device].prices
        assert pearl_c_scheme(name).name == name
        for device in DEVICES.values():
            if device.prices == own:
                assert pearl_c_scheme(name, prices=device.prices).name == name
                continue
            built = f"{name}@{device.prices.device}"
            said = f"the scheme built for {name!r} names itself {built!r}"
            with pytest.raises(ValueError, match=f"^{re.escape(said)}$"):
                pearl_c_scheme(name, prices=device.prices)


@pytest.mark.parametrize(("rows", "steps"), [(None, 2), (4, None), (None, None)])
def test_h1s_install_refuses_a_layout_without_its_rows_or_its_steps(monkeypatch, rows, steps):
    """The refusal comes after the committed weights are read and before any other setting is: these settings hold no
    other."""
    monkeypatch.setattr(P.PHK, "layers", lambda model, types, ks: [])
    monkeypatch.setattr(P, "linear_types", lambda: ())
    target = SimpleNamespace(model=None, weight_commitments={"entry": {"weight_roots": {}, "shapes": {}}})
    with pytest.raises(ValueError, match=r"^entry: the static layout needs the pouw settings 'rows' \(T\) and 'steps' \(S\)$"):
        P._install_h1("entry", SimpleNamespace(ks=None, rows=rows, steps=steps), target)


def test_a_kernel_directory_without_one_cubin_is_refused_and_its_one_cubin_is_taken(tmp_path):
    with pytest.raises(ValueError, match=f"^{re.escape(str(tmp_path))}: 0 cubins, not one: name the cubin$"):
        PK._the_cubin(tmp_path)
    (tmp_path / "b.cubin").write_bytes(b"")
    assert PK._the_cubin(tmp_path) == tmp_path / "b.cubin"
    (tmp_path / "a.cubin").write_bytes(b"")
    with pytest.raises(ValueError, match=f"^{re.escape(str(tmp_path))}: 2 cubins, not one: name the cubin$"):
        PK._the_cubin(tmp_path)


def _coin_pass() -> tuple[Challenger, dict]:
    """A challenger and a pass of no matmul and no weight whose coins are its commitment and its salt coin, opened after
    zeros."""
    ch = Challenger(0, random.Random(0).randbytes)
    return ch, {"schema": CC.MANIFEST_SCHEMA, "matmuls": [], "weights": {}, "coins": ch.commitment.to_json(),
                "salt_opening": ch.open("salt", bytes(64)).to_json()}


def test_a_pass_whose_coins_do_not_parse_is_refused_in_those_words():
    """A coin commitment or a salt coin that its parser refuses is refused as malformed, with the parser's words; the
    same pass with both parsed goes on to the coins' own checks, here that its salt coin was opened after zeros."""
    ch, man = _coin_pass()
    malformed = "the pass's coins are malformed: "
    cases: tuple[tuple[str, str], ...] = (("coins", f"{malformed}a coin commitment's scheme is 'verity/pouw/epoch-coins/v1'"),
                                          ("salt_opening", f"{malformed}'epoch'"))
    for key, said in cases:
        with pytest.raises(DomainError, match=f"^{re.escape(said)}$"):
            PPD.manifest_epoch({**man, key: {}}, ch.commitment)
    with pytest.raises(DomainError, match="^the salt's coin was not opened after what the verifier recorded$"):
        PPD.manifest_epoch(man, ch.commitment)


def test_a_pass_whose_coins_are_no_object_is_refused_as_malformed_before_a_parser_reads_them():
    """A coin commitment or a salt coin that is no JSON object is refused as malformed by its name, and never escapes as
    the parser's own error: a pass is the prover's bytes."""
    ch, man = _coin_pass()
    values: tuple[object, ...] = ("", "00", [], [{}], 0, True, None)
    for key in ("coins", "salt_opening"):
        said = f"the pass's coins are malformed: {key} is not an object"
        for value in values:
            with pytest.raises(DomainError, match=f"^{re.escape(said)}$"):
                PPD.manifest_epoch({**man, key: value}, ch.commitment)
    with pytest.raises(DomainError, match="^the pass's coins are malformed: 'coins'$"):
        PPD.manifest_epoch({k: v for k, v in man.items() if k != "coins"}, ch.commitment)


def test_a_live_salt_is_refused_unless_every_weight_of_the_workload_and_only_those_is_committed():
    """A live session's salt coin re-starts an execution phase only over its workload's own weights: roots that miss one,
    or name a weight the workload doesn't have, are refused before the session's coin is checked; its weights' roots
    alone give the session's salt."""
    roots = {0: b"\x05" * 64}
    workload = Workload((Matmul(0, MatmulShape(16, 1024, 128), 0),))
    live = LC.LiveChallenger(Wd.LIVE_SCHEME, 0, Wd.SALT_FAMILIES, random.Random(2).randbytes)
    opening = LC.LiveRecord(live, Wd.SALT).record(PS.epoch_open_root(weights_root(roots), 3))
    said = "every weight of the workload is committed, and only those"
    others: tuple[dict[int, bytes], ...] = ({}, {**roots, 1: b"\x06" * 64}, {1: roots[0]})
    for other in others:
        with pytest.raises(DomainError, match=f"^{re.escape(said)}$"):
            PPD.epoch_of(workload, other, live.commitment, opening, 3)
    assert PPD.epoch_of(workload, roots, live.commitment, opening, 3).salt == salt_of(live.commitment, opening.coin,
                                                                                     weights_root(roots))


def test_a_group_of_experts_is_refused_unless_distinct_unheld_and_of_one_shape(monkeypatch):
    """`group` refuses, before it stacks anything, experts that repeat or are none, weights another group holds, and
    experts of more than one (n, k); a group it holds is given back."""
    monkeypatch.setattr(PG, "module", lambda *_: pytest.fail("a group was stacked before the refusal"))
    shapes = {0: (64, 32), 1: (64, 32), 2: (64, 48), 3: (64, 32), 4: (128, 32)}
    run = SimpleNamespace(entry="e", weights={w: SimpleNamespace(n=n, k=k) for w, (n, k) in shapes.items()}, pipes={})
    cases: tuple[tuple[tuple[int, ...], str], ...] = (((0, 0), "e: a group's experts are distinct weights, not (0, 0)"),
                                                     ((), "e: a group's experts are distinct weights, not ()"),
                                                     ((0, 2), "e: a group's experts are of one (n, k)"),
                                                     ((0, 4), "e: a group's experts are of one (n, k)"))
    for ws, said in cases:
        with pytest.raises(ValueError, match=f"^{re.escape(said)}$"):
            PG.group(run, ws)
    held = object()
    run._groups = {(0, 1): held}
    assert PG.group(run, (0, 1)) is held
    for ws, taken in (((1, 3), [1]), ((3, 0, 1), [0, 1])):
        said = f"e: weights {taken} are already another group's"
        with pytest.raises(ValueError, match=f"^{re.escape(said)}$"):
            PG.group(run, ws)
    assert run._groups == {(0, 1): held}


def test_a_group_is_stacked_only_from_experts_whose_bytes_are_the_kernels_size_for_them():
    """`_Group` stacks a stacked operand only when every expert's is the size the kernel's pipeline gives for its
    (n, k): a byte short or over, on one expert or on all, is refused and stacks nothing; stacked, each expert reads its
    own rows of the stack."""
    import torch
    R = SimpleNamespace(PD_MAX_ROWS=64, BM=16, Pipeline=SimpleNamespace(sizes=lambda bm, n, k, bn: {"b": 8 * n + k}))
    G = SimpleNamespace(STACKED=("b",))

    def run_of(lengths: tuple[int, ...]) -> SimpleNamespace:
        weights = {w: SimpleNamespace(n=4, k=8, b={"b": torch.full((m,), w, dtype=torch.uint8)}) for w, m in enumerate(lengths)}
        return SimpleNamespace(entry="e", weights=weights, kernel=SimpleNamespace(run=R), device=SimpleNamespace(type="cpu"),
                               driver=None)

    for lengths in ((40, 39), (40, 41), (39, 39)):
        run = run_of(lengths)
        with pytest.raises(ValueError, match="^e: an expert's b isn't its 40 bytes, so the stack's stride isn't$"):
            PG._Group(run, G, (0, 1))
        assert [len(run.weights[w].b["b"]) for w in (0, 1)] == list(lengths)
    run = run_of((40, 40))
    g = PG._Group(run, G, (0, 1))
    assert g.stacked["b"].tolist() == [0] * 40 + [1] * 40
    assert [run.weights[w].b["b"].data_ptr() - g.stacked["b"].data_ptr() for w in (0, 1)] == [0, 40]


def test_a_grouped_call_is_refused_under_a_schedule_with_other_side_steps_before_its_group(monkeypatch):
    """A grouped call runs on a side lane only `SIDE_STEPS`' steps: a schedule with any other there is refused, naming the
    schedule and its side steps sorted, before the call's group is read; a schedule with only those goes on to it."""
    reached: list[tuple[int, ...]] = []

    class Reached(Exception):
        pass

    def group(run, ws):
        reached.append(ws)
        raise Reached

    monkeypatch.setattr(PG, "group", group)
    refused: tuple[tuple[str, ...], ...] = (("gemm",), ("tile_hash", "screen", "gemm"))
    taken: tuple[tuple[str, ...], ...] = ((), ("tile_hash",), ("a_tree", "tile_hash"))
    for side in refused:
        run = SimpleNamespace(entry="e", schedule=SimpleNamespace(name="s", side=side))
        said = f"e: a grouped call runs only a_tree, tile_hash on a side lane, not s's {', '.join(sorted(side))}"
        with pytest.raises(ValueError, match=f"^{re.escape(said)}$"):
            PG.gemm_grouped(run, [0], None, [0])
    assert reached == []
    for side in taken:
        with pytest.raises(Reached):
            PG.gemm_grouped(SimpleNamespace(entry="e", schedule=SimpleNamespace(name="s", side=side)), [0, 1], None, [0, 0])
    assert reached == [(0, 1)] * 3


def test_a_grouped_call_is_refused_unless_its_counts_are_its_experts_rows_of_x_and_with_no_unit_returns_no_rows(monkeypatch):
    """Past its group, a grouped call takes one count an expert, none negative, summing to x's rows, each of the group's
    k; else it is refused before its units are read.  Counts that make no unit return no rows of the group's n, with
    the bias where one is given, before the run's pass or kernel is read."""
    import torch
    asked: list[tuple[int, ...]] = []

    class Reached(Exception):
        pass

    def units(counts: tuple[int, ...]) -> list:
        asked.append(counts)
        if any(counts):
            raise Reached
        return []

    monkeypatch.setattr(PG, "group", lambda run, ws: SimpleNamespace(G=None, n=8, k=4, units=units))
    run = SimpleNamespace(entry="e", schedule=SimpleNamespace(name="s", side=()), cur=None, kernel=SimpleNamespace(run=None))
    cases: tuple[tuple[tuple[int, ...], tuple[int, int], tuple[int, ...]], ...] = (
        ((0, 1), (3, 4), (3,)), ((0, 1), (3, 4), (1, 1)), ((0, 1), (3, 4), (4, -1)), ((0,), (0, 4), (0, 0)),
        ((0, 1), (3, 5), (1, 2)))
    for ws, shape, counts in cases:
        said = f"e: x {shape} isn't Σ counts = {sum(counts)} rows of k = 4 over {len(ws)} experts"
        with pytest.raises(ValueError, match=f"^{re.escape(said)}$"):
            PG.gemm_grouped(run, ws, torch.zeros(shape), counts)
    assert asked == []
    with pytest.raises(Reached):
        PG.gemm_grouped(run, (0, 1), torch.zeros(3, 4), (1, 2))
    for bias, dtype in ((None, torch.float32), (torch.ones(8, dtype=torch.float64), torch.float64)):
        out = PG.gemm_grouped(run, (0, 1), torch.zeros(0, 4), (0, 0), bias)
        assert (out.shape, out.dtype) == ((0, 8), dtype)
    assert asked == [(1, 2), (0, 0), (0, 0)]


def test_a_grouped_call_is_refused_past_its_passs_capacity_and_under_a_cuda_graph_capture(monkeypatch):
    """A hashed grouped call fits its pass's tiles, calls and screened rows, each to the last; past any one it is refused,
    and an unhashed call is not held to them.  A call on a CUDA device is refused while its stream is being captured, as
    its counts are the host's.  Each refusal comes before the call's side lanes are read."""
    import torch

    class Reached(Exception):
        pass

    def side(run, G):
        raise Reached

    unit = SimpleNamespace(seg=16, tiles=2)
    monkeypatch.setattr(PG, "group", lambda run, ws: SimpleNamespace(G=None, n=8, k=4, units=lambda counts: [unit] * 2))
    monkeypatch.setattr(PG, "_side", side)

    def call(tiles: int, calls: int, flag_rows: int, hashing: bool = True, device: str = "cpu") -> None:
        cur = SimpleNamespace(tiles=tiles, matmuls=[None] * calls, flag_rows=flag_rows, base=0)
        run = SimpleNamespace(entry="e", schedule=SimpleNamespace(name="s", side=()), cur=cur, _used=0,
                              kernel=SimpleNamespace(run=None), hashing=hashing, capacity=10, flag_buf=[0] * 8,
                              device=SimpleNamespace(type=device))
        PG.gemm_grouped(run, (0, 1), torch.zeros(3, 4), (1, 2))

    said = (f"e: the pass is over its capacity (10 tiles, {PPD.CALL_CAPACITY} calls, 8 screened rows); end it and begin "
            "another")
    last = PPD.CALL_CAPACITY - 2
    overs: tuple[tuple[int, int, int], ...] = ((7, 0, 0), (0, last + 1, 0), (0, 0, 6))
    for tiles, calls, rows in overs:
        with pytest.raises(RuntimeError, match=f"^{re.escape(said)}$"):
            call(tiles, calls, rows)
        with pytest.raises(Reached):
            call(tiles, calls, rows, hashing=False)
    with pytest.raises(Reached):
        call(6, last, 5)
    capturing = [True]
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing[0])
    said = "e: a grouped call's counts are the host's, so a CUDA graph capture can't take it"
    with pytest.raises(RuntimeError, match=f"^{re.escape(said)}$"):
        call(0, 0, 0, device="cuda")
    with pytest.raises(Reached):
        call(0, 0, 0)
    capturing[0] = False
    with pytest.raises(Reached):
        call(0, 0, 0, device="cuda")


def test_a_prefetch_of_b_is_refused_on_a_kernel_whose_pipeline_lacks_it():
    said = "e: whole_prefetch_b runs the kernel's B̃ prefetch (`Pipeline.prefetch_b`), which it lacks"
    lacks = SimpleNamespace(run=SimpleNamespace(Pipeline=type("Pipeline", (), {})))
    with pytest.raises(ValueError, match=f"^{re.escape(said)}$"):
        whole_options("e", lacks, True, False, False, False, whole_prefetch_b=True)
    has = SimpleNamespace(run=SimpleNamespace(Pipeline=type("Pipeline", (), {"prefetch_b": None})))
    assert whole_options("e", has, True, False, False, False, whole_prefetch_b=True) is None


def test_a_shared_memory_split_is_refused_on_a_kernel_whose_pipeline_lacks_it_with_or_without_a_deferred_capture():
    said = ("e: whole_pd_shared launches pd_commit and pd_tree in half an SM's shared memory (`Pipeline.pd_shared`), "
            "which the kernel's pipeline lacks")
    lacks = SimpleNamespace(run=SimpleNamespace(Pipeline=type("Pipeline", (), {"prefetch_b": None})))
    has = SimpleNamespace(run=SimpleNamespace(Pipeline=type("Pipeline", (), {"pd_shared": None})))
    for defer in (True, False):
        with pytest.raises(ValueError, match=f"^{re.escape(said)}$"):
            whole_options("e", lacks, defer, False, False, False, whole_pd_shared=True)
        assert whole_options("e", has, defer, False, False, False, whole_pd_shared=True) is None
        assert whole_options("e", lacks, defer, False, False, False) is None
