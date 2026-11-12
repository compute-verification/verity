"""`engine.env`: the vLLM pins are defaults an operator's export overrides, but for the fixed ones, which refuse another inherited
value; a target profile's values replace the process's, and every caller reads the one table."""
from __future__ import annotations

import os
import sys

import pytest

from verity_vllm.engine import env

PINNED = sorted({*env.ENGINE, *env.EXPORT, "HF_HOME", "HF_HUB_OFFLINE"})
RBLOCK = "TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK"
#: every stage that builds `vllm.LLM` itself, at any world size
TP_ENGINES = [("verity_vllm.pipeline.tp.capture", "main"), ("verity_vllm.pipeline.tp.commit", "main"),
              ("verity_vllm.pipeline.tp.match", "_engine_main")]


@pytest.fixture
def clean(monkeypatch):
    for k in PINNED:
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_pin_fills_what_is_unset_and_keeps_what_the_operator_exported(clean) -> None:
    clean.setenv("VLLM_LOGGING_LEVEL", "DEBUG")
    got = env.pin(env.EXPORT)
    assert got == {**env.EXPORT, "VLLM_LOGGING_LEVEL": "DEBUG"}
    assert {k: os.environ[k] for k in env.EXPORT} == got


def test_apply_engine_pins_the_profile_and_the_checkpoint_store(clean, tmp_path) -> None:
    got = env.apply_engine(str(tmp_path))
    assert got == env.ENGINE
    assert os.environ["HF_HOME"] == str(tmp_path) and os.environ["HF_HUB_OFFLINE"] == "1"


def test_apply_engine_without_a_store_leaves_hf_home_alone(clean) -> None:
    env.apply_engine(None)
    assert "HF_HOME" not in os.environ and os.environ["HF_HUB_OFFLINE"] == "1"


def test_a_target_profile_replaces_the_pin(clean) -> None:
    clean.setenv("VLLM_USE_FLASHINFER_SAMPLER", "1")
    clean.setenv("VLLM_USE_DEEP_GEMM", "1")
    assert env.apply_target({"VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_USE_DEEP_GEMM": "0"}) == {
        "VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_USE_DEEP_GEMM": "0"}
    assert env.apply_engine(None)["VLLM_USE_FLASHINFER_SAMPLER"] == "0"


def test_a_step_gets_the_stage_rpc_bound_unless_the_operator_set_one(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", raising=False)
    assert env.step_timeout(1800.0) == {"VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "1800"}
    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "60")
    assert env.step_timeout(1800.0) == {"VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "60"}


@pytest.mark.parametrize("module, fn", TP_ENGINES)
def test_every_tp_engine_bounds_a_step_by_its_rpc_timeout_before_it_is_built(module, fn) -> None:
    """vLLM's own 300 s stopped a GLM-5.3 B8 capture in its first observed step, a 256-token prefill; the greedy workload's first one
    took 147 s on an idle host."""
    import importlib
    import inspect

    src = inspect.getsource(getattr(importlib.import_module(module), fn))
    assert src.index("engine_env.step_timeout(args.rpc_timeout)") < src.index("LLM(**kw)")


def test_hidden_devices_override_what_the_process_inherited(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    env.hide_devices()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""


def test_a_capture_at_served_defaults_never_pins_a_switch_back() -> None:
    """A capture that clears `SERVED_DEFAULTS` imports no module that pins the engine's switches (`vllm_meta` pins `EXPORT` at
    import, batch invariance among it; `engine_profile` pins `ENGINE`) and calls no pinning function, at any level of its code."""
    import ast
    from pathlib import Path

    pinning = {"verity_vllm.program.frontend.vllm_meta", "verity_vllm.program.frontend.export_compat",
               "verity_vllm.engine.engine_profile", "pin", "apply_engine", "apply_target", "apply_env"}
    captures = [p for p in sorted((Path(__file__).parents[1] / "properties").glob("*_gpu.py")) if "served_defaults" in p.read_text()]
    assert len(captures) >= 7
    for p in captures:
        used = set()
        for n in ast.walk(ast.parse(p.read_text())):
            if isinstance(n, ast.Import):
                used |= {a.name for a in n.names}
            elif isinstance(n, ast.ImportFrom) and n.module:
                used |= {n.module, *(f"{n.module}.{a.name}" for a in n.names)}
            elif isinstance(n, ast.Call):
                called = getattr(n.func, "attr", getattr(n.func, "id", None))
                if called is not None:
                    used.add(called)
        assert not used & pinning, (p.name, sorted(used & pinning))


def test_the_engine_profile_reads_this_table() -> None:
    from verity_vllm.engine import engine_profile as prof

    assert prof.DECLARED_ENV is env.ENGINE
    assert env.ENGINE == {"VLLM_USE_V2_MODEL_RUNNER": "1", "VLLM_ENABLE_V1_MULTIPROCESSING": "0", "VLLM_USE_FLASHINFER_SAMPLER": "0",
                          "VLLM_DISABLE_SHARED_EXPERTS_STREAM": "1", RBLOCK: "0"}
    assert set(env.REASONS) == set(env.ENGINE) and all(env.REASONS.values())
    assert set(env.FIXED) <= set(env.ENGINE)
    assert env.EXPORT == {"VLLM_USE_LAYERNAME": "0", "VLLM_DISABLE_SHARED_EXPERTS_STREAM": "1",
                          "HF_HUB_OFFLINE": "1", "VLLM_LOGGING_LEVEL": "WARNING", "TOKENIZERS_PARALLELISM": "false"}


def test_a_checkpoint_not_on_this_host_is_refused_by_name(tmp_path) -> None:
    import json

    import pytest

    from verity_vllm.engine import engine_profile as prof

    entry = {"repo": "org/model", "role": "ROLE_X", "downloaded": True, "local_path": str(tmp_path / "gone")}
    (m := tmp_path / "checkpoints.json").write_text(json.dumps({"checkpoints": [entry]}))
    with pytest.raises(SystemExit, match=f"org/model: local snapshot {tmp_path / 'gone'} missing"):
        prof.checkpoint_for("ROLE_X", m)
    m.write_text(json.dumps({"checkpoints": [dict(entry, downloaded=False)]}))
    with pytest.raises(SystemExit, match="org/model is not downloaded on this pod"):
        prof.checkpoint_for("ROLE_X", m)
    made = dict(entry, downloaded=False, recipe={"id": "verity-int8-channel127/v1"})
    m.write_text(json.dumps({"checkpoints": [made]}))
    with pytest.raises(SystemExit, match=f"org/model: local snapshot {tmp_path / 'gone'} missing"):
        prof.checkpoint_for("ROLE_X", m)
    (tmp_path / "gone").mkdir()
    m.write_text(json.dumps({"checkpoints": [entry]}))
    assert prof.checkpoint_for("ROLE_X", m) == entry
    m.write_text(json.dumps({"checkpoints": [made]}))
    assert prof.checkpoint_for("ROLE_X", m) == made


def test_a_role_two_pins_carry_is_refused_with_both_revisions(tmp_path) -> None:
    """A role names one checkpoint.  Two pins under one role are refused, each named by its revision's first twelve characters,
    where the first pin's snapshot would have been served; under a role of its own each resolves to its own snapshot."""
    import json

    from verity_vllm.engine import engine_profile as prof

    for name in ("first", "second"):
        (tmp_path / name).mkdir()
    first = {"repo": "org/model", "role": "ROLE_X", "revision": "9535bd9b1d1d" + "0" * 28, "downloaded": True,
             "local_path": str(tmp_path / "first")}
    second = dict(first, revision="4e20de362430" + "1" * 28, local_path=str(tmp_path / "second"))
    (m := tmp_path / "checkpoints.json").write_text(json.dumps({"checkpoints": [first, second]}))
    with pytest.raises(SystemExit) as refused:
        prof.checkpoint_for("ROLE_X", m)
    assert str(refused.value) == ("case 'ROLE_X': 2 pins carry role 'ROLE_X' (9535bd9b1d1d, 4e20de362430); a role names one "
                                  "checkpoint, so another revision of a model is pinned under a role of its own")
    m.write_text(json.dumps({"checkpoints": [first, dict(second, role="ROLE_Y")]}))
    assert prof.checkpoint_for("ROLE_X", m) == first
    assert prof.checkpoint_for("ROLE_Y", m) == dict(second, role="ROLE_Y")


def test_serving_and_the_build_put_a_shared_expert_on_one_stream() -> None:
    """A Build orders a shared expert on the main stream, so a served run must run it there too: on two streams, its log's order
    is not the execution order the fold takes it to be."""
    assert env.ENGINE["VLLM_DISABLE_SHARED_EXPERTS_STREAM"] == env.EXPORT["VLLM_DISABLE_SHARED_EXPERTS_STREAM"] == "1"


def test_every_engine_runs_a_reduction_kernel_at_its_static_tiling(clean) -> None:
    assert env.FIXED[RBLOCK] == ("dynamic_scale_rblock", False)
    assert env.apply_engine(None)[RBLOCK] == os.environ[RBLOCK] == "0"
    clean.setenv(RBLOCK, "0")
    assert env.apply_engine(None)[RBLOCK] == "0"


@pytest.mark.parametrize("inherited", ["1", "", "true"])
def test_an_inherited_dynamic_tiling_is_refused_by_name_before_anything_is_pinned(clean, inherited) -> None:
    clean.setenv(RBLOCK, inherited)
    with pytest.raises(RuntimeError, match=f"fail-closed: this process inherited {RBLOCK}={inherited!r}.*unset it"):
        env.apply_engine(None)
    with pytest.raises(RuntimeError, match=f"inherited {RBLOCK}"):
        env.pin({**env.ENGINE, "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION": "1"})
    assert not (set(env.ENGINE) - set(env.FIXED)) & set(os.environ)
    assert "HF_HUB_OFFLINE" not in os.environ


def test_an_engine_refuses_an_inductor_that_read_the_tiling_otherwise(monkeypatch) -> None:
    from types import SimpleNamespace

    monkeypatch.delitem(sys.modules, "torch._inductor.config", raising=False)
    env.require_inductor({})
    monkeypatch.setitem(sys.modules, "torch._inductor.config", SimpleNamespace(dynamic_scale_rblock=False))
    env.require_inductor({"compilation_config": {"mode": 3}})
    env.require_inductor({"compilation_config": {"inductor_compile_config": {"dynamic_scale_rblock": False}}})
    env.require_inductor({"compilation_config": {"inductor_compile_config": None}})
    for over in ({"inductor_compile_config": {"dynamic_scale_rblock": True}},
                 SimpleNamespace(inductor_compile_config={"dynamic_scale_rblock": True})):
        with pytest.raises(RuntimeError, match="inductor_compile_config sets dynamic_scale_rblock=True"):
            env.require_inductor({"compilation_config": over})
    monkeypatch.setitem(sys.modules, "torch._inductor.config", SimpleNamespace(dynamic_scale_rblock=True))
    with pytest.raises(RuntimeError, match=f"torch._inductor.config was imported before {RBLOCK}=0 was pinned"):
        env.require_inductor({})


def test_every_process_of_the_suite_loaded_inductor_under_the_pin() -> None:
    """The suite's `conftest.py` imports ``torch._inductor.config`` itself, under the pin, wherever torch is installed, so no test is
    the first to import it with the pin cleared from its environment, and an engine built in any test is not refused for the tests
    that ran before it."""
    import importlib.util

    loaded, installed = "torch._inductor.config" in sys.modules, importlib.util.find_spec("torch") is not None
    assert loaded == installed, "conftest.py imports torch._inductor.config right after it pins the fixed switches, where torch is installed"
    env.require_inductor({})


@pytest.mark.parametrize("plugin", [False, True], ids=["plain", "under-the-suite-guard"])
def test_the_suites_guard_fails_the_test_that_leaves_a_switch_changed_and_no_test_whose_monkeypatch_undoes_it(tmp_path, plugin) -> None:
    """`conftest.py`'s Inductor guard on `inductor_guard_cases.py`, in a pytest of their own: a test that sets the switch, or puts a
    stand-in module in, with `monkeypatch` passes; the test that leaves the switch changed is an error at its teardown, by name,
    with the guard's message; and the test after it reads the pin again.  The same under the plugin `suites.py` loads into every
    suite, loaded by name as it loads it and from where it loaded it into this run (skipped in a run it didn't start, which has
    none): its autouse fixture sets `monkeypatch` up before any fixture of this suite, so `monkeypatch` undoes a test's change
    after every other fixture is torn down."""
    import subprocess
    from pathlib import Path

    pytest.importorskip("torch")
    cases = "tests/engine/inductor_guard_cases.py"
    # under `suites.py` the child's plugin must not act for this run: no report written over this run's, no test left out as
    # another part's or as a reused pass; the child still traces what it reads, as every Python a test starts does
    runner = ("VERITY_SUITE_REPORT", "VERITY_SUITE_PART", "VERITY_SUITE_REUSE", "VERITY_SUITE_HEAVY")
    environ = {k: v for k, v in os.environ.items() if k not in runner}
    if plugin:
        loaded = sys.modules.get("verity_suite_guard")
        if loaded is None or loaded.__file__ is None:
            pytest.skip("`suites.py` loads its plugin into the suites it runs, and it didn't start this run")
        else:
            guard = Path(loaded.__file__).parent
            environ["PYTHONPATH"] = os.pathsep.join([*filter(None, environ.get("PYTHONPATH", "").split(os.pathsep)), str(guard)])
    r = subprocess.run([sys.executable, "-m", "pytest", cases, "-rA", "-p", "no:cacheprovider", f"--basetemp={tmp_path / 'b'}",
                        *(["-p", "verity_suite_guard"] if plugin else [])], cwd=Path(__file__).resolve().parents[2], env=environ,
                       capture_output=True, text=True)
    lines = r.stdout.splitlines()
    left = f"{cases}::test_a_switch_left_changed_is_failed_by_name"
    assert sorted(line for line in lines if line.startswith(("PASSED ", "FAILED ", "ERROR "))) == sorted([
        f"PASSED {cases}::test_a_switch_set_with_monkeypatch_is_not_left",
        f"PASSED {cases}::test_a_stand_in_module_set_with_monkeypatch_is_not_left",
        f"PASSED {left}", f"ERROR {left}",
        f"PASSED {cases}::test_the_test_after_it_reads_the_pin"]), r.stdout + r.stderr
    assert r.returncode == 1 and any("ERROR at teardown of test_a_switch_left_changed_is_failed_by_name" in line for line in lines)
    assert "test left torch._inductor.config reading {'dynamic_scale_rblock': True}" in r.stdout


@pytest.mark.parametrize("module, fn", TP_ENGINES)
def test_every_tp_engine_checks_its_inductor_before_it_is_built(module, fn) -> None:
    import importlib
    import inspect

    src = inspect.getsource(getattr(importlib.import_module(module), fn))
    assert src.index("engine_env.require_inductor(kw)") < src.index("LLM(**kw)")


def test_every_engine_records_the_tiling_it_runs(clean, tmp_path) -> None:
    import json

    from verity_vllm.engine import build

    clean.setenv("VLLM_BATCH_INVARIANT", "0")
    clean.setenv("HF_HOME", str(tmp_path))
    (m := tmp_path / "checkpoints.json").write_text(json.dumps({"checkpoints": [
        {"repo": "r/m", "role": "M", "downloaded": True, "local_path": str(tmp_path)}]}))
    for execution in ("bi-eager", "bi-compiled"):
        _, env_flags, _ = build.engine_kwargs_for(m, case="M", execution=execution)
        assert env_flags[RBLOCK] == "0"
    with pytest.raises(RuntimeError, match="inductor_compile_config sets dynamic_scale_rblock=True"):
        build.engine_kwargs_for(m, case="M", execution="bi-compiled",
                                engine_args={"compilation_config": {"inductor_compile_config": {"dynamic_scale_rblock": True}}})


_RANKS = """
import multiprocessing as mp
import sys

from verity_vllm.engine import env


def rank(q):
    import torch._inductor.config as c
    q.put(c.dynamic_scale_rblock)


if __name__ == "__main__":
    if sys.argv[1] == "late":
        import torch._dynamo  # noqa: F401
    env.apply_engine(None)
    import torch._inductor.config as c
    try:
        env.require_inductor({})
    except RuntimeError as e:
        sys.exit(f"refused: {e}")
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=rank, args=(q,))
    p.start()
    print(c.dynamic_scale_rblock, q.get())
    p.join()
"""


@pytest.mark.slow
def test_torch_reads_the_pin_in_the_engine_process_and_a_rank_it_spawns_or_the_engine_is_refused(tmp_path) -> None:
    """Runs torch's own reading of the variable, which a fake config can't: pinned before torch._inductor.config is imported, Inductor
    reads the static tiling in the engine's process and in a rank it spawns; imported first, the engine is refused."""
    import subprocess

    (script := tmp_path / "ranks.py").write_text(_RANKS)
    inherited = {k: v for k, v in os.environ.items() if k != RBLOCK}
    pinned = subprocess.run([sys.executable, str(script), "pinned"], env=inherited, cwd=tmp_path, capture_output=True, text=True)
    assert (pinned.returncode, pinned.stdout.split()) == (0, ["False", "False"]), pinned.stderr
    late = subprocess.run([sys.executable, str(script), "late"], env=inherited, cwd=tmp_path, capture_output=True, text=True)
    assert late.returncode == 1 and "torch._inductor.config was imported before" in late.stderr, late.stderr
