"""Values the pouw protocols' own lines never hand these functions, and what each function does with one: it checks nothing
of its own, so the value goes on as it is and the line that reads it decides.  A pass before its first call has no base, a
run without graphs has no driver, a kernel file is a Python file, a manifest's schema is a string, a whole step's sets are
keyed by a pass kind's key and a committed linear's weight has two dimensions; a call with side steps took a side lane, and
one whose A tree is aside a tree lane; a retained call's pass has a directory, and so has a retained pass's end, whose run
has a salt key; a retained capture has its stages, a capture holds its calls and its side steps not yet joined, a
replayed forward has the slots its replay uploaded, and a Pearl-C scheme's settings name its kernel; each test hands the
other value."""
from __future__ import annotations

import contextlib
import random
import re
import sys
from types import SimpleNamespace

import pytest
from verity.protocols.accounting.pouw.verifier import core_check as CC
from verity.protocols.accounting.pouw.verifier.coins import Challenger
from verity.protocols.accounting.pouw.verifier.serving import SCHEDULES

from verity_vllm import protocols as PO
from verity_vllm.protocols import pouw as P
from verity_vllm.protocols.pouw import pearl_c as PPD
from verity_vllm.protocols.pouw.pearl_c import graphs as PG
from verity_vllm.protocols.pouw.pearl_c import kernel as PK
from verity_vllm.protocols.pouw.pearl_c import lanes as L
from verity_vllm.protocols.pouw.pearl_c import manifest as PM
from verity_vllm.protocols.pouw.pearl_c.host import joined
from verity_vllm.protocols.pouw.pearl_c.whole import WholeStep

NO_BASE = r"unsupported operand type\(s\) for -: 'int' and 'NoneType'"


def test_a_pass_without_a_base_fails_where_a_calls_position_is_computed():
    """A call's position in its pass is its index less the pass's base.  Before the pass's first call the base is None,
    and the subtraction refuses it: in a graph's offsets, and at the copy of A's root after a call's steps."""
    cur = PPD._Pass(None)
    with pytest.raises(TypeError, match=NO_BASE):
        PG.GraphCalls._offsets(SimpleNamespace(nb=64, cur=cur), 3, 0, 0)
    copies: list = []
    run = SimpleNamespace(hashing=True, nb=64, roots_buf=SimpleNamespace(data_ptr=lambda: 0), copy=copies.append)
    with pytest.raises(TypeError, match=NO_BASE):
        PPD.PearlCDeviceRun._after(run, cur, 0, SimpleNamespace(n=8), None, SimpleNamespace(size={"tree_a": 128}), 3, 16, 32,
                                   16, 1, None, (), None)
    assert copies == [] and [m.index for m in cur.matmuls] == [3]


def test_a_pass_with_a_base_places_a_call_past_it():
    cur = PPD._Pass(None, base=2)
    assert PG.GraphCalls._offsets(SimpleNamespace(nb=64, cur=cur), 5, 4, 6) == (256, 12, 192)


def test_a_kernel_file_that_is_no_python_file_fails_as_no_module_and_leaves_none(tmp_path):
    """`_load` reads a kernel's Python by its path.  A path with another suffix has no module spec, and making the module
    from none fails before anything is kept under the name."""
    path = tmp_path / "kernel.txt"
    path.write_text("X = 1\n")
    before = set(sys.modules)
    with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'loader'"):
        PK._load("no_module", path)
    assert set(sys.modules) == before
    source = tmp_path / "kernel.py"
    source.write_text("X = 1\n")
    module, digest = PK._load("a_module", source)
    try:
        assert module.X == 1 and module.__name__ == f"a_module_{digest[:16]}"
    finally:
        del sys.modules[module.__name__]


class SchemaLike:
    """No string, and equal to the manifest's schema id."""

    def __eq__(self, other: object) -> bool:
        return other == CC.MANIFEST_SCHEMA

    def __hash__(self) -> int:
        return hash(CC.MANIFEST_SCHEMA)


def test_a_manifests_schema_is_compared_with_the_ids_and_returned_as_it_is():
    """`manifest_schema` compares the manifest's schema with the two ids and returns the manifest's own value: one that
    equals an id is returned whatever its type, and one that equals neither is refused by name."""
    like = SchemaLike()
    assert PM.manifest_schema({"schema": like}) is like
    assert PM.manifest_schema({"schema": CC.MANIFEST_SCHEMA}) == CC.MANIFEST_SCHEMA
    for other in (None, 7, ["x"]):
        with pytest.raises(PM.DomainError, match=re.escape(f"schema {other!r}, not {CC.MANIFEST_SCHEMA}")):
            PM.manifest_schema({"schema": other})


def test_a_dispatch_without_a_key_asks_the_sets_for_none():
    """A forward with no key (outside a pass) is dispatched from the set under None.  The whole step keeps none there, so
    such a forward runs eagerly; a set put there is the one it replays from, and no eager forward is counted."""
    desc, eager = "a graph's descriptor", "an eager forward"
    mgr = SimpleNamespace(graphs={"vllm's": 0}, _graphs_captured=True)
    whole = object.__new__(WholeStep)
    whole.mgr, whole.sets, whole.preparing = mgr, {}, False
    whole.counts = {"replayed": 0, "staged": 0, "eager": 0, "captured": 0}
    whole.stack = SimpleNamespace(executing=True, open_pass=None)
    assert whole.key() is None
    dispatch = whole._dispatch(lambda: desc if mgr._graphs_captured else eager)
    assert dispatch() == eager and whole.counts["eager"] == 1 and mgr.graphs == {"vllm's": 0}
    held = {desc: "a graph"}
    whole.sets[None] = held
    assert dispatch() == desc and whole.counts["eager"] == 1 and mgr.graphs is held


class Uploads:
    def __init__(self):
        self.put: list = []

    def put_raw(self, ptr, data, lane, driver):
        self.put.append((ptr, data, lane, driver))


def test_lane_graphs_without_a_driver_upload_the_slot_and_fail_at_the_first_launch():
    """`_replay_lanes` hands the run's driver on unchecked.  With none, the slot's upload is asked of none, and the main
    graph's launch is where it fails."""
    uploads = Uploads()
    lane = SimpleNamespace(raw=9, event=object(), tree_event=object())
    graph = SimpleNamespace(slot_ptr=7, execs=("main", None, "side"), ev_tree=object())
    with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'launch'"):
        L.LaneCalls._replay_lanes(SimpleNamespace(driver=None, uploads=uploads), graph, SimpleNamespace(done=[None]), 0, lane,
                                  lane, 5, None, b"slot")
    assert uploads.put == [(7, b"slot", 5, None)]


def test_forked_steps_without_a_driver_are_built_and_fail_when_the_first_runs(monkeypatch):
    """`_forked` builds the lane's steps around the run's driver unchecked.  With none it still returns them, the caller's
    own in their order; the first to run is where it fails."""
    monkeypatch.setattr(PG, "_event", object)
    ran: list = []
    steps = [lambda: ran.append("a"), lambda: ran.append("b")]
    run = SimpleNamespace(fork=SimpleNamespace(raw=3, stream=None), driver=None, device="cpu",
                          kernel=SimpleNamespace(dev=SimpleNamespace(stream=5)))
    out = PG.GraphCalls._forked(run, steps, [lambda: ran.append("beside")], 1)
    assert len(out) == 5 and out[1] is steps[0] and out[3] is steps[1]
    with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'record'"):
        out[0]()
    assert ran == [] and run.kernel.dev.stream == 5


def test_a_calls_side_steps_run_on_the_side_lane_it_took_and_fail_at_its_handle_where_the_run_holds_none():
    """`_lanes` hands the side lane it took for the call on unchecked.  Off the GPU the call's own steps run on the
    caller's handle and its side steps on the lane's.  With None in the run's ``sides`` the own steps still run, no side
    step does, and the lane's handle is where it fails; the device's stream and the launching lane are unset after both."""
    pytest.importorskip("torch")
    ran: list = []
    buf = SimpleNamespace(data_ptr=lambda: 64)
    rotated = dict.fromkeys(("c", "segdig_t", "u"), buf)
    kernel = SimpleNamespace(dev=SimpleNamespace(stream=5))
    steps, side = [lambda: ran.append(("own", kernel.dev.stream))], [lambda: ran.append(("side", kernel.dev.stream))]

    def launched(lane):
        sh = SimpleNamespace(turn=0, sets=[rotated], m16=16, done=[None], buf=lambda name: buf)
        run = SimpleNamespace(device=SimpleNamespace(type="cpu"), schedule=SCHEDULES["deferred-caller"], main=None,
                              sides=[lane], _side_turn=0, hashing=False, nb=64, kernel=kernel, _launching="a lane")
        pipe = SimpleNamespace(b={}, keys={})
        try:
            return L.LaneCalls._lanes(run, 7, sh, pipe, steps, side, 3, "the caller's stream", 0, False)
        finally:
            assert (sh.turn, run._side_turn, sh.done) == (1, 1, [None]) and pipe.keys == {"dom_a": 64, "unit": 128}
            assert pipe.b == dict.fromkeys(rotated, 64) and kernel.dev.stream is None and run._launching is None

    assert launched(SimpleNamespace(raw=11, stream=None, event=None, tree_event=None)) == (None, False)
    assert ran == [("own", 3), ("side", 11)]
    ran.clear()
    with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'raw'"):
        launched(None)
    assert ran == [("own", 3)]


def test_a_call_whose_a_tree_is_aside_fails_at_the_event_of_a_lane_it_did_not_take(monkeypatch):
    """On the card (a stand-in device) `_lanes` hands on unchecked the tree lane and, for a tree that ran apart, the
    call's side lane.  A call whose A tree is aside has side steps, the tree among them, so it took a tree lane: with no
    side step its own steps run up to the cut, and the tree lane's event is where it fails.  From `TREE_APART_ROWS` rows
    the tree runs on the side lane after the call's: with None for the call's in the run's ``sides`` the tree runs, A's
    root is copied to the pass, and the call's lane's tree event is where it fails, before the rest of the side steps."""
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: contextlib.nullcontext())
    ran: list = []
    tree, roots = torch.arange(128, dtype=torch.uint8), torch.zeros(64, dtype=torch.uint8)
    kernel = SimpleNamespace(dev=SimpleNamespace(stream=5))
    after = SimpleNamespace(raw=13, stream=SimpleNamespace(wait_event=lambda event: ran.append("the tree lane waits")),
                            event=SimpleNamespace(record=lambda stream: ran.append(f"the cut on {stream}")), tree_event=None)
    steps, side = ([lambda x=x: ran.append((x, kernel.dev.stream)) for x in names]
                   for names in (("keying", "rest"), ("tree", "hashing")))

    def launched(side_steps):
        sh = SimpleNamespace(turn=0, sets=[dict.fromkeys(("c", "segdig_t", "u"), tree)], m16=L.TREE_APART_ROWS,
                             done=[None], buf=lambda name: tree)
        run = SimpleNamespace(device=SimpleNamespace(type="cuda"), schedule=SCHEDULES["deferred-caller"], main=None,
                              sides=[None, after], _side_turn=0, hashing=False, nb=64, kernel=kernel, _launching="a lane",
                              roots_buf=roots, cur=SimpleNamespace(first=0))
        pipe = SimpleNamespace(b={}, keys={}, lane_cuts=(1, 1), size={"tree_a": 128})
        try:
            return L.LaneCalls._lanes(run, 7, sh, pipe, steps, side_steps, 3, "the caller's stream", 0, True)
        finally:
            assert kernel.dev.stream is None and run._launching is None

    with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'event'"):
        launched([])
    assert ran == [("keying", 3)] and not roots.any()
    ran.clear()
    with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'tree_event'"):
        launched(side)
    assert ran == [("keying", 3), "the cut on the caller's stream", ("rest", 3), "the tree lane waits", ("tree", 13)]
    assert roots.tolist() == tree[64:].tolist()


NO_BASE_YET = r"unsupported operand type\(s\) for \+: 'NoneType' and 'int'"
NO_STAGES = "'NoneType' object is not subscriptable"
CALL = PG.WholeCall(7, 32, 4096, 4096, 32, 64)


def test_a_retained_call_of_a_pass_without_a_directory_fails_at_its_first_path():
    """`_retained` makes the call's paths under the pass's directory unchecked.  A pass that retains nothing has none, and
    the first path made of it is where it fails, before a directory, a record or a file is kept."""
    pytest.importorskip("torch")
    cur = PPD._Pass(None)
    with pytest.raises(TypeError, match=r"unsupported operand type\(s\) for /: 'NoneType' and 'str'"):
        PPD.PearlCDeviceRun._retained(SimpleNamespace(), cur, None, 1, None, None, None)
    assert cur.dirs == set() and cur.records == [] and cur.files == []


def test_a_retained_passes_end_without_a_directory_fails_at_its_first_path():
    """`_write_retained` makes the pass's last paths under its directory unchecked: `end` calls it for a pass that has one.
    A pass that retains nothing has none, and the weights' directory is where it fails, before a file is written."""
    pytest.importorskip("torch")
    cur = PPD._Pass(None)
    with pytest.raises(TypeError, match=r"unsupported operand type\(s\) for /: 'NoneType' and 'str'"):
        PPD.PearlCDeviceRun._write_retained(SimpleNamespace(nb=64), cur, None, b"", b"", None, 0, {})
    assert cur.writes == []


def test_a_retained_passes_end_of_a_run_without_a_salt_key_fails_at_the_manifests_head(tmp_path):
    """`_write_retained` puts the run's hm96 salt key in the manifest's head unchecked: a run holds one from its start,
    which refuses a run without (`host._constants`).  Handed a run with none, it fails where the head reads the key: the
    pass's leaves and its tile tree are written behind by then, and no manifest is."""
    torch = pytest.importorskip("torch")
    d = tmp_path / "pass"
    cur, tree = PPD._Pass(d, run="a run"), torch.arange(64, dtype=torch.uint8)
    run = SimpleNamespace(nb=64, leaves=torch.ones(128, dtype=torch.uint8), weights={}, _weight_files={}, entry="an entry",
                          scheme=SimpleNamespace(name=PPD.SCHEMES[0]), coins=SimpleNamespace(to_json=dict),
                          opening=SimpleNamespace(to_json=dict), salt=bytes(32), commits={}, details={"schedule": {}},
                          kernel=SimpleNamespace(cubin_sha256="", source_sha256="", files={}), salt_key=None)
    with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'hex'"):
        PPD.PearlCDeviceRun._write_retained(run, cur, None, b"", bytes(64), tree, 1, {})
    joined(cur.writes)
    assert (d / "leaves").read_bytes() == b"\1" * 64 and (d / "tree_t").read_bytes() == bytes(range(64))
    assert sorted(p.name for p in d.iterdir()) == ["leaves", "tree_t", "weights"] and not list((d / "weights").iterdir())


def test_whole_calls_kept_without_stages_fail_at_the_first_calls_stage():
    """`_keep_whole` reads each call's stage of the run's stages unchecked.  A run that staged nothing has none."""
    cur = PPD._Pass(None, matmuls=["the call's matmul"])
    with pytest.raises(TypeError, match=NO_STAGES):
        PG.GraphCalls._keep_whole(SimpleNamespace(cur=cur, _stages=None), (CALL,), 0)


def test_a_forwards_slots_and_its_calls_start_at_the_passes_base_unchecked():
    """`_slots` and `whole_replayed` add to the pass's base as it is.  Before the pass's first call it is None, and the
    sum refuses it, before `whole_replayed` appends a call."""
    cur = PPD._Pass(None)
    with pytest.raises(TypeError, match=NO_BASE_YET):
        PG.GraphCalls._slots(SimpleNamespace(cur=cur), (), (0, 0, 0))
    with pytest.raises(TypeError, match=NO_BASE_YET):
        PG.GraphCalls.whole_replayed(SimpleNamespace(cur=cur), (CALL,))
    assert cur.matmuls == []


def test_a_replayed_forward_with_no_uploaded_slots_appends_its_calls_and_fails_at_the_slots():
    """`whole_replayed` takes the slots the replay uploaded unchecked.  With none (no `whole_replay` before it), the
    forward's calls are appended to the pass first, and reading the slots is where it fails."""
    cur = PPD._Pass(None, base=2)
    run = SimpleNamespace(cur=cur, hashing=True, _uploaded=None)
    with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'predicted'"):
        PG.GraphCalls.whole_replayed(run, (CALL,))
    assert [(mm.index, mm.weight) for mm in cur.matmuls] == [(2, 7)] and run._uploaded is None
    assert (cur.tiles, cur.flag_rows) == (0, 0)


def test_a_graphed_call_of_a_retained_capture_without_stages_fails_where_it_reads_them(monkeypatch):
    """`_graph_call` reads the run's stages unchecked, in a retained capture and at the warm-up before it.  With none, the
    capture's call is recorded and launched before its stage is read, and the warm-up fails at the lookup, before any
    stage is made."""
    sh = SimpleNamespace(k=4096, n=4096, m16=32)
    launched: list = []

    def into(*a):
        launched.append(a)
        return "y", True
    run = SimpleNamespace(_whole=[], _whole_retained=True, _stages=None, _tree_joined=lambda: launched.append("joined"),
                          _into=into)
    monkeypatch.setattr(PG, "_capturing", lambda: True)
    with pytest.raises(TypeError, match=NO_STAGES):
        PG.GraphCalls._graph_call(run, 7, sh, None, 5, 64, 32, 3, None)
    assert run._whole == [CALL] and launched == [(7, sh, None, 5, 64, 32), "joined"]
    monkeypatch.setattr(PG, "_capturing", lambda: False)
    with pytest.raises(TypeError, match="argument of type 'NoneType'"):
        PG.GraphCalls._graph_call(run, 7, sh, None, 5, 64, 32, 3, None)
    assert run._whole == [CALL] and run._stages is None and len(launched) == 2


def test_a_deferred_call_launched_while_the_capture_holds_no_calls_fails_after_its_steps_at_their_count():
    """`_into` counts the capture's calls unchecked, after a deferred call's steps, to tell the forward's last call.  With
    no list of them, the steps run and are forked first, and the count is where it fails; the kernel's stream is put
    back."""
    ran: list = []
    buf = SimpleNamespace(data_ptr=lambda: 0)
    graph = SimpleNamespace(graph="a graph", y=buf, leaves=buf, cuts=(1, 2), stats_at=None, tree_at=None,
                            steps=[lambda: ran.append("head"), lambda: ran.append("gemm"), lambda: ran.append("tail")])
    run = SimpleNamespace(_graph_key=lambda w, sh, m: ("key", True), _graphs={"key": graph}, _widening=None, entry="entry",
                          _forks_stats=lambda: False, _pd_shared=lambda: False, _defers=lambda: True,
                          _defers_a=lambda: False, _prefetches=lambda: False, whole_shape_joins=True,
                          _fork=lambda sh, steps, raw: [f() for f in steps], _whole=None, order={7: 0},
                          kernel=SimpleNamespace(dev=SimpleNamespace(stream=None)))
    pipe = SimpleNamespace(b={})
    with pytest.raises(TypeError, match=r"object of type 'NoneType' has no len\(\)"):
        PG.GraphCalls._into(run, 7, SimpleNamespace(m16=32), pipe, 5, 64, 32)
    assert ran == ["head", "gemm", "tail"] and run.kernel.dev.stream is None and pipe.tiles_group is False


def test_side_steps_forked_with_no_list_of_those_not_joined_run_and_fail_where_they_are_noted(monkeypatch):
    """`_fork` notes the forked steps among those no call has joined, unchecked.  With no list of them (outside a
    capture), the steps run on the side stream first, the kernel's stream is put back, and the note is where it fails."""
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: "the capturing stream")
    monkeypatch.setattr(torch.cuda, "stream", contextlib.nullcontext)
    waits: list = []
    ran: list = []
    side = SimpleNamespace(cuda_stream=11, wait_stream=waits.append)
    run = SimpleNamespace(_side_stream=side, device="cuda", hashing=False, _tree_joined=waits.append, _pending=None,
                          whole_shape_joins=False, kernel=SimpleNamespace(dev=SimpleNamespace(stream=None)))
    with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'append'"):
        PG.GraphCalls._fork(run, "a shape", [lambda: ran.append(run.kernel.dev.stream)], 5)
    assert waits == ["the capturing stream", side] and ran == [11] and run.kernel.dev.stream == 5


class Refused(Exception):
    pass


class Served:
    """A period's served record, as far as `_install_h1` reaches before the weights' root: it keeps the shapes it is
    handed and refuses."""

    def __init__(self):
        self.shapes: list = []

    def weights_root(self, shapes, *rest):
        self.shapes.append(shapes)
        raise Refused


def test_h1_hands_each_layers_weight_shape_on_whole(monkeypatch):
    """`_install_h1` takes each layer's weight's shape as it is, every dimension an int, and hands the list to the period's
    served record: a weight of three dimensions reaches it as three."""
    layers = [SimpleNamespace(weight=SimpleNamespace(shape=shape)) for shape in ((8, 16), (2, 3, 4))]
    monkeypatch.setattr(P.PHK, "layers", lambda model, types, ks: layers)
    monkeypatch.setattr(P, "linear_types", lambda: ())
    monkeypatch.setattr(P, "_named", lambda model, layers: {})
    served = Served()
    settings = SimpleNamespace(ks=None, rows=4, steps=2, rounds=1, served=served)
    target = SimpleNamespace(model=None, weight_commitments={"entry": {"weight_roots": {}, "shapes": {}}},
                             context=SimpleNamespace(stage="serve"))
    with pytest.raises(Refused):
        P._install_h1("entry", settings, target)
    assert served.shapes == [[(8, 16), (2, 3, 4)]]


def test_a_pearl_c_scheme_whose_settings_name_no_kernel_hands_the_kernels_load_none(monkeypatch):
    """`refusals` refuses a Pearl-C scheme whose settings name no ``kernel``.  `commit_weights` and `install` check nothing
    of it: each hands the kernel's load the setting as it is, None, beside the setting ``cubin``, and the load fails at
    its directory's path before it reads a file."""
    scheme, seen = "pearl-c-sm120-v1-h3", []
    entry, kernels = f"{P.NAME}:{scheme}", dict(PK._KERNELS)
    ctx = PO.Context(stage="serve", enabled=(entry,),
                     settings={P.NAME: {"coins": Challenger(0, random.Random(0).randbytes), "device": "cuda", "cubin": "c"}})
    assert [r for r in P.refusals(scheme, ctx) if "'kernel'" in r] == [
        f"{scheme}: runs on the device only, so it needs the pouw settings 'device': 'cuda' and 'kernel'"]
    with pytest.raises(TypeError, match="NoneType"):
        PPD.Kernel.load(None)
    assert kernels == PK._KERNELS

    def load(*a):
        seen.append(a)
        raise Refused

    monkeypatch.setattr(PPD.Kernel, "load", staticmethod(load))
    monkeypatch.setattr(P, "_SALT_KEYS", {})
    monkeypatch.setattr(P, "_layers",
                        lambda scheme, target: (PPD.scheme_of(scheme), [SimpleNamespace(weight=None)], {0: (128, 1024)}))
    committed = {entry: {"weight_roots": {"0": "05" * 64}, "shapes": {"0": [128, 1024]}}}
    for call, target in ((P.commit_weights, SimpleNamespace(model=None, context=ctx)),
                         (P.install, SimpleNamespace(model=None, context=ctx, weight_commitments=committed))):
        with pytest.raises(Refused):
            call(scheme, target)
    assert seen == [(None, "c"), (None, "c")]
