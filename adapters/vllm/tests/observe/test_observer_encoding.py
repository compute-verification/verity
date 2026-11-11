"""Torch-dependent tests of the observer's encoding half on CPU tensors (they run on the pod; skipped on a laptop
without torch): encode_arg, StorageAdapter.view_of / generations / pending drain, the Observer mode itself."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from verity_vllm.observe.events import DispatchEnter, DispatchExit, Note, StorageAlloc, StorageFree, ViewDesc  # noqa: E402
from verity_vllm.observe.observer import EmptyMode, Observer, Recorder, encode_arg, op_name  # noqa: E402
from verity_vllm.observe.storage import StorageAdapter  # noqa: E402


def _rec():
    events: list = []
    st = StorageAdapter()
    rec = Recorder(st, events.append)
    rec.active = True
    rec.step = 3
    rec.phase = "forward"
    return rec, events


def test_encode_scalars_and_containers():
    st = StorageAdapter()
    assert encode_arg(None, st) is None
    assert encode_arg(True, st) is True and encode_arg(7, st) == 7 and encode_arg(2.5, st) == 2.5 and encode_arg("x", st) == "x"
    assert encode_arg((1, [2, None]), st) == [1, [2, None]]
    assert encode_arg(torch.bfloat16, st) == "torch.bfloat16"
    assert encode_arg(torch.device("cpu"), st) == "cpu"
    assert encode_arg(torch.contiguous_format, st) == "torch.contiguous_format"
    assert encode_arg(torch.strided, st) == "torch.strided"
    other = encode_arg(object(), st)
    assert isinstance(other, dict) and set(other) == {"$repr"} and len(other["$repr"]) <= 200


def test_view_of_fields_and_shared_storage():
    st = StorageAdapter()
    a = torch.zeros(4, 6, dtype=torch.bfloat16)
    v = st.view_of(a)
    assert isinstance(v, ViewDesc)
    assert v.offset == 0 and v.shape == (4, 6) and v.strides == (6, 1) and v.dtype == "bfloat16" and v.device == "cpu"
    w = st.view_of(a[1:3, ::2])
    assert w.alloc == v.alloc and w.offset == 6 and w.shape == (2, 3) and w.strides == (6, 2)
    t = st.view_of(a.t())
    assert t.alloc == v.alloc and t.shape == (6, 4) and t.strides == (1, 6)
    assert encode_arg(a, st) == v
    assert encode_arg([a, None], st) == [v, None]


def test_events_pending_precede_first_reference():
    st = StorageAdapter()
    a = torch.ones(8)
    v = st.view_of(a)
    ev = st.events_pending()
    assert len(ev) == 1 and isinstance(ev[0], StorageAlloc)
    assert ev[0].alloc == v.alloc and ev[0].nbytes == 32 and ev[0].device == "cpu" and ev[0].generation == 0
    st.view_of(a)                       # a second observation of the same storage queues nothing
    assert st.events_pending() == []
    assert st.observe(a) == v.alloc


def test_empty_tensor_gets_the_null_allocation():
    st = StorageAdapter()
    e = torch.empty(0)
    v = st.view_of(e)
    v2 = st.view_of(torch.empty(0, 3))
    assert v.alloc == v2.alloc and v.shape == (0,) and v2.shape == (0, 3)
    assert [type(x).__name__ for x in st.events_pending()] == ["StorageAlloc"]


def test_generation_bump_on_pointer_reuse_raw():
    st = StorageAdapter(use_weakrefs=True)
    a1 = st.observe_raw(0x1000, 64, "cpu", key=1)
    assert st.events_pending()[0].generation == 0
    a2 = st.observe_raw(0x1000, 64, "cpu", key=2)          # same range, a NEW storage object -> implied free + bump
    ev = st.events_pending()
    assert a2 != a1
    assert [type(x).__name__ for x in ev] == ["StorageFree", "StorageAlloc"]
    assert ev[0] == StorageFree(a1, True)
    assert ev[1].alloc == a2 and ev[1].generation == 1 and ev[1].ptr == 0x1000
    assert st.stats["generation_bumps"] == 1 and st.stats["frees_implied"] == 1
    # a partially overlapping new allocation frees the old one too
    a3 = st.observe_raw(0x1020, 64, "cpu", key=3)
    ev = st.events_pending()
    assert ev[0] == StorageFree(a2, True) and ev[1].alloc == a3 and ev[1].generation == 0
    # same storage object observed at a new pointer (resize_ / set_) -> implied free of its old entry
    a4 = st.observe_raw(0x9000, 16, "cpu", key=3)
    ev = st.events_pending()
    assert ev[0] == StorageFree(a3, True) and ev[1].alloc == a4
    assert st.live_count() == 1


def test_weakref_death_is_a_direct_free_and_reuse_bumps_generation():
    st = StorageAdapter()
    if not st.use_weakrefs:
        pytest.skip("weakrefs to UntypedStorage unsupported in this torch")
    n = 1024                              # 4 KB: below glibc's mmap threshold, so the freed chunk is handed back
    a = torch.ones(n)
    va = st.view_of(a)
    ptr = a.untyped_storage().data_ptr()
    st.events_pending()
    del a
    ev = st.events_pending()
    assert ev == [StorageFree(va.alloc, False)]
    # allocate the same size until the allocator hands the pointer back (usually immediately)
    for _ in range(50):
        b = torch.ones(n)
        if b.untyped_storage().data_ptr() == ptr:
            break
        del b
        st.events_pending()
    else:
        pytest.skip("allocator did not reuse the pointer")
    vb = st.view_of(b)
    ev = st.events_pending()
    assert vb.alloc != va.alloc
    assert ev[-1].alloc == vb.alloc and ev[-1].ptr == ptr and ev[-1].nbytes == 4 * n and ev[-1].generation >= 1
    assert ev[-1].origin == "op" and ev[-1].device == "cpu"


def test_observer_records_a_cpu_op():
    rec, events = _rec()
    a = torch.randn(3, 4)
    b = torch.randn(4, 5)
    with Observer(rec):
        c = torch.mm(a, b)
    kinds = [type(e).__name__ for e in events]
    assert kinds == ["StorageAlloc", "StorageAlloc", "DispatchEnter", "StorageAlloc", "DispatchExit"]
    enter, exit_ = events[2], events[4]
    assert isinstance(enter, DispatchEnter) and enter.op == "aten.mm.default" and enter.parent is None
    assert enter.step == 3 and enter.phase == "forward" and enter.schema.startswith("aten::mm(")
    assert enter.args == [rec.storage.view_of(a), rec.storage.view_of(b)] and enter.kwargs == {}
    assert isinstance(exit_, DispatchExit) and exit_.id == enter.id and exit_.outs == rec.storage.view_of(c)
    assert exit_.outs.alloc == events[3].alloc
    assert rec.open == [] and rec.counts["dispatch"] == 1
    assert op_name(torch.ops.aten.mm.default) == "aten.mm.default"


def test_nested_mode_instances_share_the_recorder_and_link_parents():
    # the re-entry mechanism: a fresh inner mode instance on top of the outer one; the inner records first and
    # its func(...) call reaches the outer mode, whose record has the inner call as parent
    rec, events = _rec()
    a = torch.randn(2, 2)
    with Observer(rec):
        with Observer(rec):
            torch.mm(a, a)
    enters = [e for e in events if isinstance(e, DispatchEnter)]
    assert len(enters) == 2 and enters[0].parent is None and enters[1].parent == enters[0].id
    exits = [e.id for e in events if isinstance(e, DispatchExit)]
    assert exits == [enters[1].id, enters[0].id]


def test_exception_survives_with_a_note():
    rec, events = _rec()
    with Observer(rec):
        with pytest.raises(RuntimeError):
            torch.mm(torch.zeros(2, 3), torch.zeros(2, 3))
    notes = [e for e in events if isinstance(e, Note)]
    assert notes and notes[0].text == "exception in dispatcher call" and notes[0].data["op"] == "aten.mm.default"
    assert [e for e in events if isinstance(e, DispatchExit)][-1].outs is None
    assert rec.open == []


def test_missing_reentry_impl_keeps_why_each_path_failed(monkeypatch):
    from verity_vllm.observe import observer as O
    monkeypatch.setattr(O, "_REENTER_IMPL", {})
    monkeypatch.setattr(O, "_REENTER_MISSING", {})
    monkeypatch.setitem(O.REENTER_IMPL_PATHS, "verity.no_such_op",
                        [("verity_vllm.no_such_module", "impl"), ("verity_vllm.observe.observer", "no_such_impl")])
    assert O.reenter_impl("verity.no_such_op") is None
    why = O._REENTER_MISSING["verity.no_such_op"]
    assert [w.split(":")[0] for w in why] == ["verity_vllm.no_such_module.impl", "verity_vllm.observe.observer.no_such_impl"]
    assert "ModuleNotFoundError" in why[0] and "AttributeError" in why[1]


def test_inactive_recorder_and_empty_mode_pass_through():
    rec, events = _rec()
    rec.active = False
    with Observer(rec):
        torch.mm(torch.zeros(2, 2), torch.zeros(2, 2))
    assert events == []
    with EmptyMode():
        assert torch.equal(torch.ones(2) + 1, torch.full((2,), 2.0))
