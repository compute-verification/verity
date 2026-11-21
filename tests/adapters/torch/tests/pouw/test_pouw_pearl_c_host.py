"""Pearl-C's pinned host arena (`pearl_c.host._Arena`), CPU only, over a stand-in for pinned memory: a take within half a
ring lands in the first ring with room, and its give-back frees that ring's region; a take past half a ring gets memory of
its own and makes no ring; a take no ring has room for waits for a give, and every give wakes the takes that wait; bytes
taken inside inference mode take copies outside it.  The copies that land in the arena on a card are the e2e's.  And a
pass's hm96 salts (`fresh_into`, `PassSalts`): the next leaves of its own keyed stream, its key drawn at its first fill and
dropped at its end, a fill after it or in a forked process refused by name, one card fill on the caller's stream with no
host bytes, and every fill in `verity_torch` naming its run's current pass; the card's bytes are `salt_stream`'s card
tests'.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
from verity.protocols.accounting.compute.pouw.verifier.audit import HM96_SALT

from verity_torch.pouw.pearl_c import host as PH
from verity_torch.pouw.pearl_c.host import _Arena


@pytest.fixture
def unpinned(monkeypatch):
    """`torch.empty` blind to `pin_memory`: a host with no accelerator has no pinned memory to give."""
    torch = pytest.importorskip("torch")
    empty = torch.empty
    monkeypatch.setattr(torch, "empty", lambda *a, pin_memory=False, **kw: empty(*a, **kw))


class Waits:
    """An arena's condition where no other thread runs: a take that waits gets the next give-back on hand, as a thread that
    gave meanwhile would have left it."""

    def __init__(self, gives):
        self.gives, self.waited, self.notified = list(gives), 0, 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def wait(self):
        self.waited += 1
        self.gives.pop(0)()

    def notify_all(self):
        self.notified += 1


def _at(ring, host):
    """Where in the ring's bytes the host tensor lies: (its first byte, the byte past its last)."""
    first = host.data_ptr() - ring.buf.data_ptr()
    return [first, first + host.numel()]


def test_a_take_lands_in_the_first_ring_with_room_and_its_give_back_frees_that_rings_region(unpinned):
    """`_Arena.take`: sizes are rounded up to 512 bytes; the rings are made at the first take; a take goes to the first
    ring that has room.  Each give-back frees its own region in its own ring, whatever was taken after it: a region
    given back before an earlier one of its ring stays live until that one is given back too."""
    arena = _Arena(4096, 2)
    assert arena.rings == []
    (give_a, a), (give_b, b), (_, c), (give_d, d) = (arena.take(n) for n in (2000, 1500, 100, 2048))
    first, second = arena.rings
    assert [_at(first, t) for t in (a, b, c)] == [[0, 2048], [2048, 3584], [3584, 4096]] and _at(second, d) == [0, 2048]
    give_d()
    assert not second.live and [r[:2] for r in first.live] == [[0, 2048], [2048, 3584], [3584, 4096]]
    give_b()
    assert [r[:2] for r in first.live] == [[0, 2048], [2048, 3584], [3584, 4096]]         # b is free only once a is
    give_a()
    assert [r[:2] for r in first.live] == [[3584, 4096]] and not second.live


def test_a_take_past_half_a_ring_gets_memory_of_its_own_and_makes_no_ring(unpinned):
    """`_Arena.take`: a copy past half a ring is no ring's: its bytes are as many as asked, and its give-back gives
    nothing back."""
    arena = _Arena(4096, 2)
    give, host = arena.take(2049)
    assert host.numel() == 2049 and give() is None and arena.rings == []


def test_a_take_no_ring_has_room_for_waits_for_a_give_and_every_give_wakes_the_takes_that_wait(unpinned, monkeypatch):
    """`_Arena.take` and `_give`: with its one ring full a take waits on the arena's condition, and lands once a give has
    made room (here the region past the ring's end is skipped, so it lands at the ring's start).  Each give notifies."""
    arena = _Arena(4096, 1)
    (give_a, _), (give_b, _) = arena.take(2048), arena.take(2048)
    waits = Waits([give_a])
    monkeypatch.setattr(arena, "cv", waits)
    give_c, c = arena.take(1024)
    (ring,) = arena.rings
    assert (waits.waited, waits.notified, waits.gives) == (1, 1, []) and _at(ring, c) == [0, 1024]
    give_b()
    assert [r[:2] for r in ring.live] == [[0, 1024]]
    give_c()
    assert not ring.live and waits.notified == 3


def test_bytes_taken_inside_inference_mode_take_copies_outside_it(unpinned):
    """`_Arena.take` inside inference mode (the first copies of a retained pass, made in `PearlCDeviceRun.gemm`) makes the
    rings and a take of memory of its own as normal tensors, so a pass's end, which copies into them outside inference mode,
    is not refused as an in-place update to an inference tensor."""
    import torch
    arena = _Arena(4096, 1)
    with torch.inference_mode():
        (_, ringed), (_, own) = arena.take(1024), arena.take(4096)
    (ring,) = arena.rings
    assert not ring.buf.is_inference() and not own.is_inference()
    _, later = arena.take(512)
    src = torch.arange(512, dtype=torch.int32).to(torch.uint8)
    later.copy_(src)
    own[:512].copy_(src)
    assert torch.equal(later, src) and torch.equal(own[:512], src)


@pytest.fixture
def keys(monkeypatch):
    """Every key `derive` makes, recorded with its arguments."""
    made, derive = [], PH.derive
    monkeypatch.setattr(PH, "derive", lambda *a: made.append((a, derive(*a))) or made[-1][1])
    return made


def _leaves(key, first: int, n: int) -> bytes:
    return b"".join(key.block(PH.SALT_INDEX, c) for c in range(3 * first, 3 * (first + n)))


def test_off_the_gpu_each_fill_is_the_next_leaves_of_the_passs_keyed_stream(keys):
    """`fresh_into` on host bytes (the recording device's) writes `Key.block`'s bytes, each fill the leaves after the last's,
    under a key drawn at the pass's first fill from 32 fresh bytes in the salts' domain; bytes past the fill stay."""
    import torch
    salts = PH.PassSalts()
    assert salts.key is None and not keys
    a, b = torch.full((HM96_SALT * 5,), 0xA5, dtype=torch.uint8), torch.zeros(HM96_SALT * 2, dtype=torch.uint8)
    PH.fresh_into(a, 3, salts=salts)
    PH.fresh_into(b, 2, salts=salts)
    ((source, domain, context), key), = keys
    assert len(source) == 32 and domain == PH.SALT_STREAM and context == {} and salts.key is key
    assert a[:HM96_SALT * 3].numpy().tobytes() == _leaves(key, 0, 3) and (a[HM96_SALT * 3:] == 0xA5).all()
    assert b.numpy().tobytes() == _leaves(key, 3, 2) and salts.next == 5


def test_a_dropped_pass_holds_no_key_and_refuses_a_fill_by_name(keys):
    """After `drop` (a pass's end) the key attribute is cleared and a fill is refused by name before it writes; so is a
    fill that names no pass.  The key's bytes are an immutable `bytes`, which Python can't zero: nothing here holds it."""
    import torch
    salts, t = PH.PassSalts(), torch.zeros(HM96_SALT, dtype=torch.uint8)
    PH.fresh_into(t, 1, salts=salts)
    salts.drop()
    assert salts.key is None and not salts.open
    t[:] = 0xA5
    for named in (salts, None):
        with pytest.raises(PH.SaltsRefused, match="NO_OPEN_PASS") as refused:
            PH.fresh_into(t, 1, salts=named)
        assert refused.value.name == "NO_OPEN_PASS"
    assert (t == 0xA5).all() and len(keys) == 1


def test_each_pass_draws_its_own_key(keys):
    import torch
    t = torch.zeros(HM96_SALT, dtype=torch.uint8)
    first, second = PH.PassSalts(), PH.PassSalts()
    PH.fresh_into(t, 1, salts=first)
    PH.fresh_into(t, 1, salts=second)
    assert first.key.value != second.key.value and t.numpy().tobytes() == _leaves(second.key, 0, 1)


def test_a_forked_process_fills_none_of_its_parents_pass(keys, monkeypatch):
    import torch
    salts, t = PH.PassSalts(), torch.zeros(HM96_SALT, dtype=torch.uint8)
    monkeypatch.setattr(PH.os, "getpid", lambda: -1)
    with pytest.raises(PH.SaltsRefused, match="OTHER_PROCESS"):
        PH.fresh_into(t, 1, salts=salts)


def test_a_pass_past_its_keys_stream_is_refused(keys):
    salts = PH.PassSalts()
    salts.next = PH.SX.LEAVES_MAX - 1
    assert salts.take(1)[1] == PH.SX.LEAVES_MAX - 1
    with pytest.raises(RuntimeError, match="past its key's stream"):
        salts.take(1)


class _CardBytes:
    """What `fresh_into` reads of device bytes on a card: its device, a slice and a view, recorded."""

    is_cuda = True

    def __init__(self, device, log):
        self.device, self.log = device, log

    def __getitem__(self, s):
        self.log.append(("slice", s.stop))
        return self

    def view(self, *shape):
        self.log.append(("view", shape))
        return self


def test_on_a_card_one_fill_launch_on_the_given_stream_else_the_current_one(keys, monkeypatch):
    """No host bytes: one `salt_stream` fill of the pass's next leaves, on raw stream `raw` where given, else the
    device's current stream (`_stream`), each block's message held to its one length, 87 bytes."""
    import torch
    dev, log, fills, salts = torch.device("cuda", 0), [], [], PH.PassSalts()
    monkeypatch.setattr(PH, "fill_lib", lambda d: ("lib", d))
    monkeypatch.setattr(PH, "_stream", lambda d: 777)
    monkeypatch.setattr(PH.SX, "fill", lambda lib, out, key, index, first, **kw: fills.append((lib, index, first, kw)))
    monkeypatch.setattr(PH.os, "urandom", lambda n: pytest.fail("a card's salts drew host bytes") if n != 32 else bytes(n))
    PH.fresh_into(_CardBytes(dev, log), 32, 123, salts=salts)
    PH.fresh_into(_CardBytes(dev, log), 7, salts=salts)
    assert [f[:3] for f in fills] == [(("lib", dev), PH.SALT_INDEX, 0), (("lib", dev), PH.SALT_INDEX, 32)]
    assert [f[3] for f in fills] == [{"stream": 123, "length": 87}, {"stream": 777, "length": 87}]
    assert log == [("slice", HM96_SALT * 32), ("view", (32, HM96_SALT)), ("slice", HM96_SALT * 7), ("view", (7, HM96_SALT))]


#: every salts fill in `verity_torch`, by module of `pearl_c` and enclosing function
FILLS = {("ends.py", "_close"), ("graphs.py", "_graph_call"), ("grouped.py", "_gemm_grouped"), ("lanes_device.py", "_lanes"),
         ("lanes_device.py", "_replay_lanes"), ("whole_calls.py", "whole_replay")}


def _binds_current_pass(f: ast.AST) -> bool:
    """Whether function `f` binds ``cur`` to its run's current pass (``self.cur`` or ``run.cur``) and to nothing else."""
    got = set()
    for a in ast.walk(f):
        if isinstance(a, ast.Assign) and len(a.targets) == 1:
            names, values = a.targets[0], a.value
            pairs = zip(names.elts, values.elts, strict=True) if isinstance(names, ast.Tuple) and isinstance(values, ast.Tuple) \
                else [(names, values)]
            got |= {ast.unparse(v) for n, v in pairs if isinstance(n, ast.Name) and n.id == "cur"}
    return bool(got) and got <= {"self.cur", "run.cur"}


def test_every_salts_fill_names_its_runs_current_pass():
    """Each `fresh_into` call in `verity_torch` is one of `FILLS` and passes ``salts=`` its run's current pass's: a fill
    before any begin is the run's first pass's, dropped at its end or the next begin, and one after an end is refused by
    name (`NO_OPEN_PASS`).  A new fill is added to `FILLS` with how it reaches its pass."""
    root = Path(PH.__file__).parents[2]
    found = set()
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(), str(path))
        for f in ast.walk(tree):
            if not isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for call in ast.walk(f):
                if not (isinstance(call, ast.Call) and getattr(call.func, "id", getattr(call.func, "attr", None)) == "fresh_into"):
                    continue
                at = (path.name if path.parent.name == "pearl_c" else str(path.relative_to(root)), f.name)
                salts = {k.arg: ast.unparse(k.value) for k in call.keywords}.get("salts")
                assert salts == "self.cur.salts" or (salts == "cur.salts" and _binds_current_pass(f)), \
                    f"{at} line {call.lineno}: a fill names its run's current pass (salts=self.cur.salts), not {salts}"
                found.add(at)
    assert found == FILLS, f"the salts fills are {sorted(found)}: list a new one in FILLS with how it reaches its pass"
