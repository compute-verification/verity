"""The pouw option's refusals that no other CPU test of the adapter reaches, each held to its own words: `Settings.of`'s of
a transcript directory that is no string; `refusals`' of a tensor-parallel world, of a forward anchor under an executor
that counts its own, and of a schedule Pearl-C's path can't run; `_layers`' of a model with no linear layer and of two
layers that share a quant method; and H1's install's of a layout without its rows or its steps.  Under the option,
`circuit`'s of a scheme without a circuit and of an input group of another length than its Definition's, the
quantizer's of a value that is not finite, and the native prover's of a unit outside a call's population.  Pearl-C's of
a scheme its path doesn't run or that a test instance names otherwise, of a kernel directory without one cubin, of a
pass whose coins don't parse or are no object, of a live salt over other weights than its workload's, of a group of
experts that repeat, are another group's, differ in shape or whose bytes are not the kernel's size for them, of a
grouped call under a schedule with other side steps or with counts that aren't its rows, and of B̃'s prefetch and the
shared-memory split of pd_commit and pd_tree on a kernel without them, and the in-process certifier's of fewer coin
bytes than it asks for."""
from __future__ import annotations

import random
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from verity.protocols.accounting.pouw.prover.circuit import SCHEMES
from verity.protocols.accounting.pouw.prover.circuit.ncp2 import Shape
from verity.protocols.accounting.pouw.prover.schemes.pearl_c_device import DEVICES
from verity.protocols.accounting.pouw.verifier import core_check as CC
from verity.protocols.accounting.pouw.verifier import salt as PS
from verity.protocols.accounting.pouw.verifier import window as Wd
from verity.protocols.accounting.pouw.verifier.audit import (
    Workload,
    salt_of,
    weights_root,
)
from verity.protocols.accounting.pouw.verifier.coins import Challenger
from verity.protocols.accounting.pouw.verifier.protocol import DomainError, Matmul
from verity.protocols.accounting.pouw.verifier.protocol import Shape as MatmulShape
from verity.protocols.accounting.pouw.verifier.serving import (
    SCHEDULES,
    Schedule,
    SchemeId,
)
from verity.protocols.proofs.spot.device import challenger as LC

from verity_vllm.protocols import pouw as P
from verity_vllm.protocols.interface import Context
from verity_vllm.protocols.pouw import circuit as PC
from verity_vllm.protocols.pouw import native as PN
from verity_vllm.protocols.pouw import pearl_c as PPD
from verity_vllm.protocols.pouw.lead import feed as PF
from verity_vllm.protocols.pouw.operands import quantize
from verity_vllm.protocols.pouw.pearl_c import grouped as PG
from verity_vllm.protocols.pouw.pearl_c import kernel as PK
from verity_vllm.protocols.pouw.pearl_c.lanes import schedule_of, whole_options

HOST = ("ncp-v1", "ncp-v1-shift24", "ncp-v2")
#: the least (n, k) of a linear the host schemes' domain takes
SHAPE = (1600, 1088)


def _serve(scheme: str, world: int = 1, **settings) -> Context:
    return Context(stage="serve", enabled=(f"{P.NAME}:{scheme}",), world=world,
                   settings={P.NAME: {"coins": Challenger(0, random.Random(0).randbytes), **settings}})


@pytest.mark.parametrize("given", [3, Path("/t"), b"/t", ("/t",)])
def test_settings_refuse_a_transcript_directory_that_is_no_string(given):
    said = "pouw setting 'transcript_dir' is a path string"
    with pytest.raises(ValueError, match=f"^{re.escape(said)}$"):
        P.Settings.of({"transcript_dir": given})
    assert P.refusals(HOST[0], _serve(HOST[0], transcript_dir=given)) == (said,)


def test_settings_take_a_path_string_and_none_for_no_transcript_directory():
    assert P.Settings.of({"transcript_dir": "/t"}).transcript_dir == "/t"
    assert P.Settings.of({"transcript_dir": None}).transcript_dir is None
    assert P.Settings.of({}).transcript_dir is None
    assert P.refusals(HOST[0], _serve(HOST[0], transcript_dir="/t")) == ()


@pytest.mark.parametrize("scheme", HOST)
def test_a_tensor_parallel_world_is_refused(scheme):
    assert P.refusals(scheme, _serve(scheme)) == ()
    assert P.refusals(scheme, _serve(scheme, world=2)) == (
        "TP 2: PoUW's units and transcript are per rank, which is not wired",)


def test_a_forward_anchor_is_refused_under_an_executor_that_counts_its_own_forwards():
    """``start_forward`` places the circuit-backed executor's first pass.  The host's other executors count their forwards
    from 0, so an anchor past 0 is refused there, and an anchor of 0 is none."""
    circuits = ", ".join(P.pouw_circuit.SCHEMES)
    for scheme in ("ncp-v1", "ncp-v1-shift24"):
        assert P.refusals(scheme, _serve(scheme, start_forward=0)) == ()
        said = (f"{scheme}: the pouw setting 'start_forward' anchors the circuit-backed executor's forward indices "
                f"({circuits}); this executor counts its own from 0")
        assert P.refusals(scheme, _serve(scheme, start_forward=3)) == (said,)
    assert "ncp-v2" in P.pouw_circuit.SCHEMES
    assert P.refusals("ncp-v2", _serve("ncp-v2", start_forward=3)) == ()


def test_a_schedule_pearl_cs_path_cannot_run_is_refused_under_the_schemes_name(monkeypatch):
    """Every registered schedule runs under every Pearl-C scheme, so `refusals` has `schedule_of`'s refusal to pass on only
    once a schedule is registered that the path can't run.  One that takes the GEMM to a side lane stands in for it."""
    for scheme in PPD.SCHEMES:
        for name, schedule in SCHEDULES.items():
            assert schedule_of(name, PPD.scheme_of(scheme)) is schedule
    monkeypatch.setitem(SCHEDULES, "side-gemm", Schedule("side-gemm", frozenset({"gemm"}), 2, "greatest", "least"))
    for scheme in PPD.SCHEMES:
        assert P.refusals(scheme, _serve(scheme, device="cuda", kernel="k")) == ()
        said = (f"{scheme}: side-gemm: gemm can't leave the main lane under h3: a later step of the call reads what it "
                "writes; side-gemm: this path runs only a_tree, tile_hash on a side lane, not gemm")
        assert P.refusals(scheme, _serve(scheme, device="cuda", kernel="k", schedule="side-gemm")) == (said,)
    assert P.refusals("ncp-v1", _serve("ncp-v1", schedule="side-gemm")) == (
        "ncp-v1: the pouw setting 'schedule' is Pearl-C's; this executor runs serially",)


def test_a_pearl_c_scheme_is_built_only_for_a_name_this_path_runs_and_only_under_that_name():
    """A test instance takes `PearlC`'s keywords; one of another device's prices names itself with them, and is refused."""
    with pytest.raises(KeyError) as e:
        PPD.scheme_of("pearl-c-sm120-v1")
    assert e.value.args == (f"'pearl-c-sm120-v1': this device path runs {', '.join(PPD.SCHEMES)}",)
    for name in PPD.SCHEMES:
        own = DEVICES[SchemeId.parse(name).device].prices
        assert PPD.scheme_of(name).name == name
        for device in DEVICES.values():
            if device.prices == own:
                assert PPD.scheme_of(name, prices=device.prices).name == name
                continue
            built = f"{name}@{device.prices.device}"
            said = f"the scheme built for {name!r} names itself {built!r}"
            with pytest.raises(ValueError, match=f"^{re.escape(said)}$"):
                PPD.scheme_of(name, prices=device.prices)


class Linear:
    """A linear as far as `_layers` reads one: its weight's dtype and (n, k), and its quant method."""

    def __init__(self, method: object):
        self.weight = SimpleNamespace(dtype="float32", shape=SHAPE)
        self.quant_method = method


def _model(*layers: object) -> SimpleNamespace:
    return SimpleNamespace(model=SimpleNamespace(modules=lambda: iter(layers)))


@pytest.mark.parametrize("scheme", HOST)
def test_a_model_with_no_linear_layer_and_layers_that_share_a_quant_method_are_refused(monkeypatch, scheme):
    """An executor replaces each layer's own ``quant_method.apply``.  Two layers on one method would be one site, so every
    layer after the first on a method is refused by its ordinal; layers each on their own are taken with their (n, k)."""
    monkeypatch.setattr(P, "linear_types", lambda: (Linear,))
    with pytest.raises(ValueError, match=f"^pouw:{scheme}: the model has no linear layer to execute$"):
        P._layers(scheme, _model())
    with pytest.raises(ValueError, match=f"^pouw:{scheme}: the model has no linear layer to execute$"):
        P._layers(scheme, _model(SimpleNamespace(weight=None, quant_method=None)))
    one, other = object(), object()
    shared = "its quant method is shared with another layer"
    with pytest.raises(ValueError, match=f"^pouw:{scheme} refused at install: linear 1: {shared}; linear 3: {shared}$"):
        P._layers(scheme, _model(Linear(one), Linear(one), Linear(other), Linear(one)))
    layers = (Linear(one), Linear(other))
    _scheme, took, shapes = P._layers(scheme, _model(*layers))
    assert took == list(layers) and shapes == {0: SHAPE, 1: SHAPE}


@pytest.mark.parametrize(("rows", "steps"), [(None, 2), (4, None), (None, None)])
def test_h1s_install_refuses_a_layout_without_its_rows_or_its_steps(monkeypatch, rows, steps):
    """The refusal comes after the committed weights are read and before any other setting is: these settings hold no
    other."""
    monkeypatch.setattr(P.PHK, "layers", lambda model, types, ks: [])
    monkeypatch.setattr(P, "linear_types", lambda: ())
    target = SimpleNamespace(model=None, weight_commitments={"entry": {"weight_roots": {}, "shapes": {}}})
    with pytest.raises(ValueError, match=r"^entry: the static layout needs the pouw settings 'rows' \(T\) and 'steps' \(S\)$"):
        P._install_h1("entry", SimpleNamespace(ks=None, rows=rows, steps=steps), target)


@pytest.mark.parametrize("scheme", ["ncp-v1", "ncp-v1-shift24", "ncp-v3", ""])
def test_a_scheme_without_a_circuit_has_no_permutation_and_no_program(scheme):
    assert scheme not in SCHEMES and "ncp-v2" in SCHEMES
    said = f"{scheme}: no circuit (verity.protocols.accounting.pouw.prover.circuit.SCHEMES = {', '.join(SCHEMES)})"
    with pytest.raises(ValueError, match=f"^{re.escape(said)}$"):
        PC.kind_of(scheme)
    with pytest.raises(ValueError, match=f"^{re.escape(said)}$"):
        PC.program(scheme, 16, 64, 16)


def test_a_scheme_with_a_circuit_has_its_permutation_and_its_two_programs():
    for scheme, kind in SCHEMES.items():
        assert PC.kind_of(scheme) == kind
        c = PC.program(scheme, 16, 64, 16)
        assert (c.traced.scheme, c.traced.kind, c.shape) == (scheme, kind, Shape(16, 64, 16))
        assert (len(_groups(c.program)), len(_groups(c.weight_program))) == (11, 4)


def _groups(prog) -> list[int]:
    """How many values each of a Program's input groups takes, in the Definition's order."""
    return [len(group.refs) for group in prog.root.input_colls]


def test_a_program_is_not_evaluated_on_an_input_group_of_another_length(monkeypatch):
    """Each group of a call's Program and of its weight side's, one value short and one over, with every other group
    at its length: the refusal names the group's two lengths and comes before any gate is evaluated."""
    c = PC.program("ncp-v2", 16, 64, 16)
    monkeypatch.setattr(PC, "evaluate_region", lambda *_: pytest.fail("a gate was evaluated before the refusal"))
    for prog in (c.program, c.weight_program):
        sizes = _groups(prog)
        for at, size in enumerate(sizes):
            for given in (size - 1, size + 1):
                args = [[0] * (given if g == at else s) for g, s in enumerate(sizes)]
                with pytest.raises(ValueError, match=f"^an input group has {given} values, the Definition takes {size}$"):
                    PC._transcript(prog, args)


def test_a_program_is_evaluated_whole_on_input_groups_of_its_lengths():
    c = PC.program("ncp-v2", 16, 64, 16)
    for prog in (c.program, c.weight_program):
        trace = PC._transcript(prog, [[0] * size for size in _groups(prog)])
        assert len(trace) == prog.circuit.size
        assert all(trace[gate] == 0 for group in prog.root.input_colls for (_space, _zero, gate) in group.refs)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_the_quantizer_refuses_rows_with_a_value_that_is_not_finite_and_takes_them_without_it(bad):
    rows = np.array([[63.0, -31.0], [0.0, 0.0], [0.5, bad]], dtype=np.float32)
    with pytest.raises(ValueError, match="^pouw: the quantizer refuses NaN and infinity$"):
        quantize(rows)
    rows[2, 1] = -0.125
    q, scale = quantize(rows)
    assert (q.dtype, scale.dtype) == (np.int8, np.float32)
    assert q.tolist() == [[63, -31], [0, 0], [63, -16]]
    assert scale.tolist() == [1.0, 1.0, float(np.float32(0.5) * (np.float32(1.0) / np.float32(63.0)))]


class Rows:
    """A weight's int7 rows and their FP32 scale encodings, as `circuit.Rows` hands them on."""

    def __init__(self, b: np.ndarray, t: np.ndarray):
        self.b, self.t, (self.n, self.k) = b, t, b.shape

    def rows(self, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray]:
        return self.b[lo:hi], self.t[lo:hi]


def _native(m: int, k: int, n: int) -> PN.Native:
    """The native prover of a call of zeros at m × k → n, every scale FP32's 1 (its encoding ``0x3F800000``)."""
    one = 0x3F800000
    weight = Rows(np.zeros((n, k), np.int8), np.full(n, one, np.uint32))
    call = PC.Call(np.zeros((m, k), np.int8), np.full(m, one, np.uint32), weight, np.zeros(n, np.uint16), bytes(32),
                   forward=0, layer=0, site=0, weight=0, rowids=np.arange(m, dtype=np.uint32),
                   colids=np.arange(n, dtype=np.uint32))
    return PN.Native(call, "ncp-v2")


def test_a_unit_outside_a_calls_population_or_its_weight_sides_is_refused():
    """A call of 32 × 64 → 48 has 1,548 units, and its weight side has 3, one a 16-column strip.  The unit before the
    first and the unit after the last have no place, no template and no outputs; the first and the last have theirs."""
    nat = _native(32, 64, 48)
    assert (nat.population, nat.side.population) == (1548, 3)
    for u in (-1, 1548):
        for read in (nat.locate, nat.template, nat.outputs):
            with pytest.raises(IndexError, match=f"^unit {u} outside 0..1547$"):
                read(u)
    assert (nat.locate(0), nat.locate(1547)) == (("digest", (0,)), ("dequant", (1, 2, 255)))
    for u in (-1, 3):
        with pytest.raises(IndexError, match=f"^Y unit {u} outside 0..2$"):
            nat.side.outputs(u)
    assert [nat.side.outputs(u).shape for u in (0, 2)] == [(3088,), (3088,)]


def test_a_kernel_directory_without_one_cubin_is_refused_and_its_one_cubin_is_taken(tmp_path):
    with pytest.raises(ValueError, match=f"^{re.escape(str(tmp_path))}: 0 cubins, not one: name the cubin$"):
        PK._the_cubin(tmp_path)
    (tmp_path / "b.cubin").write_bytes(b"")
    assert PK._the_cubin(tmp_path) == tmp_path / "b.cubin"
    (tmp_path / "a.cubin").write_bytes(b"")
    with pytest.raises(ValueError, match=f"^{re.escape(str(tmp_path))}: 2 cubins, not one: name the cubin$"):
        PK._the_cubin(tmp_path)


def test_a_certifier_given_fewer_coin_bytes_than_it_asks_for_opens_no_window():
    """The refusal is held from its counts on: the words before them are a retired term, which this file doesn't add."""
    with pytest.raises(PF.FeedError, match=f" gave {PF.ROOT_BYTES - 1} bytes for {PF.ROOT_BYTES}$"):
        PF.InProcessCertifier(lambda n: bytes(n - 1)).open(0, bytes(PF.ROOT_BYTES), bytes(PF.ROOT_BYTES), 1)
    cert = PF.InProcessCertifier(bytes)
    assert cert.open(0, bytes(PF.ROOT_BYTES), bytes(PF.ROOT_BYTES), 1) == bytes(PF.ROOT_BYTES)
    assert list(cert.windows) == [0]


def _coin_pass() -> tuple[Challenger, dict]:
    """A challenger and a pass of no matmul and no weight whose coins are its commitment and its salt coin, opened after
    zeros."""
    ch = Challenger(0, random.Random(0).randbytes)
    return ch, {"schema": CC.MANIFEST_SCHEMA, "matmuls": [], "weights": {}, "coins": ch.commitment.to_json(),
                "salt_opening": ch.open("salt", bytes(PF.ROOT_BYTES)).to_json()}


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
