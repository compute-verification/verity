"""Values the pouw protocols' own lines never hand these functions, and what each function does with one: it checks nothing
of its own, so the value goes on as it is and the line that reads it decides.  A pass before its first call has no base, a
run without graphs has no driver, a kernel file is a Python file, a manifest's schema is a string, a whole step's sets are
keyed by a pass kind's key and a committed linear's weight has two dimensions; each test hands the other value."""
from __future__ import annotations

import re
import sys
from types import SimpleNamespace

import pytest
from verity.protocols.pouw.verifier import core_check as CC

from verity_vllm.protocols import pouw as P
from verity_vllm.protocols.pouw import pearl_c as PPD
from verity_vllm.protocols.pouw.pearl_c import graphs as PG
from verity_vllm.protocols.pouw.pearl_c import kernel as PK
from verity_vllm.protocols.pouw.pearl_c import lanes as L
from verity_vllm.protocols.pouw.pearl_c import manifest as PM
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
