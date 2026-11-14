"""`verity_vllm.observe.triton_adapter`'s wrapper of a Triton launch, with the engine's patch of `JITFunction.run`, Triton's
driver and a JIT function stood in (no CPU machine has Triton): `install` patches once and binds the recorder, a second
`install` and `set_recorder` only rebind it, `uninstall` takes the patch off once, and the wrapper hands a launch straight to
the run it replaced while no recorder is bound or the bound one is not recording. While one records, a launch is an event
of every parameter by name, its grid as ints and its warps and stages; a warm-up is none; a launch whose binder fails is
noted and recorded by position, and one that cannot be recorded is noted and still returns its kernel."""
from __future__ import annotations

import hashlib
import sys
import weakref
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("torch")

import torch

from verity_vllm.observe import triton_adapter as T
from verity_vllm.observe.events import Note, TritonLaunch
from verity_vllm.observe.observer import Recorder
from verity_vllm.observe.storage import StorageAdapter

PARAMS = ("x", "n", "BLOCK")
DEVICE = 7
SRC = "def kernel(x, n, BLOCK):\n    pass  # at most one block a program: n \u2264 BLOCK\n"
#: the SHA-256 of `SRC` as UTF-8 and of the empty text: the digest the fold's patterns pin a kernel's source by
SRC_DIGEST = "2f59b1342e04a6d00d53a937c0a46f02cccff66c73981d74888ac00c16740d9a"
EMPTY_DIGEST = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


class _Patch:
    """What `hooks.triton_launch` returns, as far as the adapter reads it."""

    def __init__(self) -> None:
        self.launches: list[tuple] = []
        self.uninstalled = 0
        self.kernel: Any = "the kernel"

    def original(self, jit_fn, *args, **kwargs):
        self.launches.append((jit_fn, args, kwargs))
        return self.kernel

    def uninstall(self) -> None:
        self.uninstalled += 1


@pytest.fixture
def patched(monkeypatch) -> tuple[_Patch, list]:
    """The adapter with nothing installed and the engine's patch stood in: the patch, and what was installed with it."""
    for key in ("orig_run", "rec", "patch"):
        monkeypatch.setitem(T._STATE, key, None)
    patch, made = _Patch(), []

    def triton_launch(make):
        made.append(make("the run Triton has"))
        return patch

    monkeypatch.setattr(T.hooks, "triton_launch", triton_launch)
    return patch, made


def test_install_patches_the_launch_once_and_uninstall_takes_it_off_once(patched) -> None:
    patch, made = patched
    first: Any = SimpleNamespace(active=False)
    second: Any = SimpleNamespace(active=False)
    assert not T.is_installed()
    T.install(first)
    assert T.is_installed() and made == [T._run]
    assert dict(T._STATE) == {"orig_run": patch.original, "rec": first, "patch": patch}
    T.install(second)
    assert made == [T._run] and T._STATE["rec"] is second and T._STATE["patch"] is patch
    T.set_recorder(None)
    assert T._STATE["rec"] is None and T.is_installed()
    T.set_recorder(first)
    T.uninstall()
    assert patch.uninstalled == 1 and not T.is_installed() and T._STATE["rec"] is None
    T.uninstall()
    assert patch.uninstalled == 1


@pytest.mark.parametrize("recorder", [None, SimpleNamespace(active=False)], ids=["no recorder", "a recorder that is not recording"])
def test_the_wrapper_hands_a_launch_to_the_run_it_replaced_when_nothing_records(patched, recorder) -> None:
    patch, _ = patched
    T.install(recorder)
    assert T._run("the jit function", 1, 2, grid=(4,), warmup=False, num_warps=8) == "the kernel"
    assert patch.launches == [("the jit function", (1, 2), {"grid": (4,), "warmup": False, "num_warps": 8})]


class _Kernels:
    """Holds the Python function a stand-in JIT function wraps, so that its qualified name is not its name."""

    @staticmethod
    def kernel(x, n, BLOCK) -> None:
        return None


NAME = f"{__name__}._Kernels.kernel"


def _binder(*args, **kwargs):
    """Triton's binder for the kernel, as far as the wrapper reads it: every parameter by name, a specialization, and of
    what the caller passed by name the options that are no parameter. The arguments are not its own dict."""
    bound = dict(zip(PARAMS, args))
    bound.update((k, v) for k, v in kwargs.items() if k in PARAMS)
    return MappingProxyType(bound), ("the specialization",), {k: v for k, v in kwargs.items() if k not in PARAMS}


class _Jit(SimpleNamespace):
    """A namespace that takes a weak reference, which a bare `SimpleNamespace` does not."""


def _jit(binder: Any = _binder, **fields: Any) -> Any:
    """A `JITFunction`, as far as the wrapper reads one: its function, source and parameters, and for the driver's device the
    binder, last of that device's caches."""
    params = [SimpleNamespace(name=name, is_constexpr=name == "BLOCK") for name in PARAMS]
    caches = {DEVICE: ("kernels", "keys", "target", "backend", binder)}
    return _Jit(**{"fn": _Kernels.kernel, "src": SRC, "params": params, "device_caches": caches, **fields})


@pytest.fixture
def recording(patched, monkeypatch) -> tuple[Recorder, list, _Patch]:
    """A recorder that records, bound to the wrapper, with Triton's driver stood in on device 7: the recorder, what it has
    emitted and the engine's patch. No source digest or constexpr names are kept from another test's function, and none of
    its functions is held: the adapter keeps both by a function's `id`, each with the function."""
    patch, _ = patched
    monkeypatch.setattr(T, "_SRC_SHA", {})
    monkeypatch.setattr(T, "_CONSTEXPR_NAMES", {})
    driver = SimpleNamespace(active=SimpleNamespace(get_current_device=lambda: DEVICE))
    monkeypatch.setitem(sys.modules, "triton.runtime.driver", SimpleNamespace(driver=driver))
    events: list = []
    rec = Recorder(StorageAdapter(), events.append)
    rec.active = True
    T.install(rec)
    return rec, events, patch


def _launches(events: list) -> list[TritonLaunch]:
    return [e for e in events if isinstance(e, TritonLaunch)]


def _notes(events: list) -> list[Note]:
    return [e for e in events if isinstance(e, Note)]


def test_a_launch_is_an_event_of_every_parameter_by_name_under_the_innermost_open_call(recording, monkeypatch) -> None:
    rec, events, patch = recording
    jit, x, told = _jit(), torch.zeros(2, 3), []
    rec.step, rec.phase, rec.open, rec.module_stack, rec.next_id = 4, "forward", [9, 11], ["layers.0", "layers.0.attn"], 12
    monkeypatch.setattr(rec, "stream", lambda: 81)      # no device here has a stream: the recorder's reading of one

    def after_launch(eid: int, fn: str, bound: dict) -> None:
        told.append((eid, fn, bound))

    def grid(bound: dict) -> tuple:
        return (bound["n"], 2)

    rec.after_launch = after_launch
    assert T._run(jit, x, 6, BLOCK=16, grid=grid, warmup=False, num_warps=8) == "the kernel"
    assert patch.launches == [(jit, (x, 6), {"BLOCK": 16, "grid": grid, "warmup": False, "num_warps": 8})]
    args = {"x": rec.storage.view_of(x), "n": 6, "BLOCK": 16}
    assert args["x"].shape == (2, 3) and list(_launches(events)[0].args) == list(PARAMS)
    assert _launches(events) == [TritonLaunch(12, 11, 4, "forward", NAME, SRC_DIGEST, args, {"BLOCK": 16}, [6, 2], 8, None, 81, "layers.0.attn")]
    assert _notes(events) == [] and rec.counts["triton"] == 1 and rec.next_id == 13
    [(eid, fn, bound)] = told
    assert (eid, fn) == (12, NAME) and type(bound) is dict and list(bound) == list(PARAMS) and bound["x"] is x


def test_a_launch_outside_any_call_and_any_module_has_no_parent_and_no_module(recording) -> None:
    rec, events, _ = recording
    T._run(_jit(), torch.zeros(2), 6, 16, grid=(1,), warmup=False)
    [launch] = _launches(events)
    assert (launch.id, launch.parent, launch.step, launch.phase, launch.module) == (1, None, -1, "other", None)
    assert launch.constexprs == {"BLOCK": 16} and rec.after_launch is None


@pytest.mark.parametrize(("grid", "recorded"), [
    ((4, 2), [4, 2]), ([3], [3]), (5, [5]), (None, []), (lambda bound: bound["n"], [6]), (lambda bound: [bound["BLOCK"], 1], [16, 1]),
    (lambda bound: None, []), ((torch.tensor(4), 2.0), [4, 2]), (3.0, [3]),
], ids=["a tuple", "a list", "one number", "none", "a callable that gives a number", "a callable that gives a list",
        "a callable that gives none", "a tensor and a float", "one float"])
def test_a_launchs_grid_is_a_list_of_ints_and_a_callable_grid_what_it_gives_for_the_bound_arguments(recording, grid, recorded) -> None:
    _, events, _ = recording
    T._run(_jit(), torch.zeros(2), 6, 16, grid=grid, warmup=False)
    [launch] = _launches(events)
    assert launch.grid == recorded and all(type(v) is int for v in launch.grid)


@pytest.mark.parametrize(("options", "metadata", "recorded"), [
    ({"num_warps": 8, "num_stages": 2}, {"num_warps": 4, "num_stages": 3}, (8, 2)),
    ({"num_warps": 8}, {"num_warps": 4, "num_stages": 3}, (8, 3)),
    ({"num_stages": 2}, {"num_warps": 4, "num_stages": 3}, (4, 2)),
    ({}, {"num_warps": 4.0, "num_stages": "3"}, (4, 3)),
    ({}, {"num_stages": 3}, (None, 3)),
    ({}, None, (None, None)),
], ids=["both passed", "the warps passed", "the stages passed", "neither passed", "a kernel that says no warps", "a kernel with no metadata"])
def test_a_launchs_warps_and_stages_are_the_callers_options_and_the_kernels_own_where_none_is_passed(recording, options, metadata,
                                                                                                    recorded) -> None:
    _, events, patch = recording
    if metadata is not None:
        patch.kernel = SimpleNamespace(metadata=SimpleNamespace(**metadata))
    T._run(_jit(), torch.zeros(2), 6, 16, grid=(1,), warmup=False, **options)
    [launch] = _launches(events)
    assert (launch.num_warps, launch.num_stages) == recorded
    assert all(v is None or type(v) is int for v in (launch.num_warps, launch.num_stages))


def test_options_that_are_no_dict_are_read_as_none_passed(recording) -> None:
    _, events, patch = recording
    patch.kernel = SimpleNamespace(metadata=SimpleNamespace(num_warps=4, num_stages=3))

    def binder(*args, **kwargs):
        return dict(zip(PARAMS, args)), (), [("num_warps", 8)]

    T._run(_jit(binder), torch.zeros(2), 6, 16, grid=(1,), warmup=False, num_warps=8)
    [launch] = _launches(events)
    assert (launch.num_warps, launch.num_stages) == (4, 3) and list(launch.args) == list(PARAMS)


def test_a_warm_up_compilation_runs_and_is_no_launch(recording) -> None:
    rec, events, patch = recording
    jit, x, told = _jit(), torch.zeros(2), []
    rec.after_launch = lambda *launch: told.append(launch)
    assert T._run(jit, x, 6, 16, grid=(1,), warmup=True) == "the kernel"
    assert patch.launches == [(jit, (x, 6, 16), {"grid": (1,), "warmup": True})]
    assert events == [] and rec.counts == {"dispatch": 0, "triton": 0, "notes": 0, "reentered": 0} and rec.next_id == 1
    assert told == []


def _raises(*args, **kwargs):
    raise TypeError("an unexpected keyword argument: " + "BLOCKS" * 60)


@pytest.mark.parametrize("broken", ["no driver", "no binder for the device", "a binder that raises"])
@pytest.mark.parametrize(("grid", "recorded"), [((4, 2), [4, 2]), (lambda bound: pytest.fail("called with no bound arguments"), [])],
                         ids=["a tuple", "a callable"])
def test_a_launch_whose_binder_fails_is_noted_and_recorded_by_position(recording, monkeypatch, broken, grid, recorded) -> None:
    rec, events, patch = recording
    jit, x, told = _jit(_raises if broken == "a binder that raises" else _binder), torch.zeros(2, 3), []
    if broken == "no driver":
        monkeypatch.setitem(sys.modules, "triton.runtime.driver", None)
    if broken == "no binder for the device":
        jit.device_caches = {DEVICE + 1: jit.device_caches[DEVICE]}
    patch.kernel = SimpleNamespace(metadata=SimpleNamespace(num_warps=4, num_stages=3))

    def after_launch(eid: int, fn: str, bound: dict) -> None:
        told.append((eid, fn, bound))

    rec.after_launch = after_launch
    assert T._run(jit, x, 6, BLOCK=16, grid=grid, warmup=False, num_warps=8) is patch.kernel
    assert patch.launches == [(jit, (x, 6), {"BLOCK": 16, "grid": grid, "warmup": False, "num_warps": 8})]
    [note] = _notes(events)
    assert note.text == "triton binder failed; recording positional arguments" and list(note.data) == ["fn", "error"]
    assert note.data["fn"] == NAME and rec.counts["notes"] == 1
    error = {"no driver": "ModuleNotFoundError(", "no binder for the device": "KeyError(7)",
             "a binder that raises": repr(TypeError("an unexpected keyword argument: " + "BLOCKS" * 60))[:300]}[broken]
    assert note.data["error"].startswith(error) and len(note.data["error"]) <= 300
    if broken != "no driver":
        assert note.data["error"] == error
    args = {"arg0": rec.storage.view_of(x), "arg1": 6}
    assert _launches(events) == [TritonLaunch(1, None, -1, "other", NAME, SRC_DIGEST, args, {}, recorded, 4, 3, rec.stream(), None)]
    assert events.index(note) < events.index(_launches(events)[0]) and rec.counts["triton"] == 1
    [(eid, fn, bound)] = told
    assert (eid, fn) == (1, NAME) and list(bound) == ["arg0", "arg1"] and bound["arg0"] is x and bound["arg1"] == 6


def test_a_launch_that_cannot_be_recorded_is_noted_and_still_returns_its_kernel(recording) -> None:
    rec, events, patch = recording

    def grid(bound: dict) -> tuple:
        raise ValueError("no grid for " + "a block " * 60)

    told: list = []
    rec.after_launch = lambda *launch: told.append(launch)
    assert T._run(_jit(), torch.zeros(2), 6, 16, grid=grid, warmup=False) == "the kernel"
    assert len(patch.launches) == 1 and _launches(events) == [] and rec.counts["triton"] == 0 and told == []
    [note] = _notes(events)
    error = repr(ValueError("no grid for " + "a block " * 60))
    assert len(error) > 300 and note == Note("triton launch record failed", {"fn": NAME, "error": error[:300]})


def test_a_functions_source_digest_and_constexpr_names_are_read_once(recording) -> None:
    jit = _jit()
    assert T.src_sha(jit) == SRC_DIGEST and T.constexpr_names(jit) == frozenset({"BLOCK"})
    jit.src, jit.params = "another source", []
    assert T.src_sha(jit) == SRC_DIGEST and T.constexpr_names(jit) == frozenset({"BLOCK"})
    other = _jit(src="another source", params=[SimpleNamespace(name=name, is_constexpr=name != "BLOCK") for name in PARAMS])
    assert T.src_sha(other) not in (SRC_DIGEST, EMPTY_DIGEST) and len(T.src_sha(other)) == 64
    assert T.constexpr_names(other) == frozenset({"x", "n"})


def test_a_function_made_once_another_is_dropped_has_its_own_source_digest_and_constexpr_names(recording) -> None:
    """What the adapter read of a function it keeps by the function's `id`, with the function: one the caller drops stays
    alive, so no later function is made at its address and answered for by it."""
    params = [SimpleNamespace(name=name, is_constexpr=name != "BLOCK") for name in PARAMS]
    for k in range(64):
        first = _jit()
        assert T.src_sha(first) == SRC_DIGEST and T.constexpr_names(first) == frozenset({"BLOCK"})
        alive = weakref.ref(first)
        del first
        assert alive() is not None
        second = _jit(src=f"the source of kernel {k}", params=params)
        assert T.src_sha(second) == hashlib.sha256(f"the source of kernel {k}".encode()).hexdigest()
        assert T.constexpr_names(second) == frozenset({"x", "n"})
    assert len(T._SRC_SHA) == len(T._CONSTEXPR_NAMES) == 128


def test_a_function_with_no_source_has_the_empty_texts_digest_and_one_with_no_parameters_to_read_no_constexpr(recording) -> None:
    _, events, _ = recording
    bare, none, unmarked = _jit(), _jit(params=None), _jit(params=[SimpleNamespace(name="BLOCK")])
    del bare.src, bare.params
    assert T.src_sha(bare) == EMPTY_DIGEST and T.fn_name(bare) == NAME
    assert T.constexpr_names(bare) == T.constexpr_names(none) == T.constexpr_names(unmarked) == frozenset()
    T._run(bare, torch.zeros(2), 6, 16, grid=(1,), warmup=False)
    [launch] = _launches(events)
    assert (launch.src_sha, launch.constexprs, list(launch.args)) == (EMPTY_DIGEST, {}, list(PARAMS)) and _notes(events) == []
