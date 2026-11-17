"""The stand-in for vLLM (`tests.fake_vllm`): it holds vLLM's surface and nothing else, and leaves no trace."""
from __future__ import annotations

import importlib
import sys
import types

import pytest

from tests import fake_vllm
from tests.lint.vllm_surface import SURFACE


def _vllm_modules() -> dict[str, types.ModuleType]:
    return {n: m for n, m in sys.modules.items() if n.split(".")[0] == "vllm"}


def _import(key: str) -> object:
    """What the import statement entry ``key`` stands for binds: the module, or its name, which may be a submodule."""
    module, name = fake_vllm.split(key)
    imported = importlib.import_module(module)
    if not name:
        return imported
    try:
        return getattr(imported, name)
    except AttributeError:
        return importlib.import_module(f"{module}.{name}")


def test_every_import_of_the_surface_works_under_the_stand_in(monkeypatch):
    fake = fake_vllm.install(monkeypatch)
    assert len(SURFACE) > 50
    for key in SURFACE:
        module, name = fake_vllm.split(key)
        bound = _import(key)
        if not name:
            assert bound is fake[module] is sys.modules[module], key
        elif f"{module}.{name}" in fake.modules:
            assert bound is fake[f"{module}.{name}"], key
        else:
            assert isinstance(bound, fake_vllm.Unset) and repr(bound) == f"<{key}, not set>", key


def test_a_module_is_one_object_however_it_is_reached(monkeypatch: pytest.MonkeyPatch) -> None:
    """Its first assignment is also what the mypy lint reads: a test sets a module's attribute, which mypy refuses on a plain
    `types.ModuleType` (it comes before the asserts, whose ``is`` would narrow the module to vLLM's untyped one)."""
    fake = fake_vllm.install(monkeypatch)
    fake["vllm.envs"].VLLM_BATCH_INVARIANT = True
    import vllm
    import vllm.distributed.parallel_state as by_path
    from vllm import envs
    from vllm.distributed import parallel_state

    assert envs is vllm.envs is fake["vllm.envs"]
    assert parallel_state is by_path is vllm.distributed.parallel_state is fake["vllm.distributed.parallel_state"]
    assert vllm.envs.VLLM_BATCH_INVARIANT is True


def test_vllm_is_as_it_was_once_the_test_is_over():
    before = _vllm_modules()
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, "vllm.left_by_another_stub", types.ModuleType("vllm.left_by_another_stub"))
        fake = fake_vllm.install(mp)
        assert _vllm_modules() == fake.modules
    assert _vllm_modules() == before


def test_an_install_inside_a_context_starts_over_and_gives_way_to_the_outer_one(monkeypatch):
    outer = fake_vllm.install(monkeypatch)
    outer["vllm.envs"].VLLM_BATCH_INVARIANT = False
    with monkeypatch.context() as mp:
        inner = fake_vllm.install(mp)
        assert _vllm_modules() == inner.modules and not hasattr(inner["vllm.envs"], "VLLM_BATCH_INVARIANT")
    from vllm import envs

    assert _vllm_modules() == outer.modules and envs is outer["vllm.envs"] and envs.VLLM_BATCH_INVARIANT is False


def test_an_import_the_surface_lacks_fails_as_where_vllm_is_absent(monkeypatch):
    fake_vllm.install(monkeypatch)
    with pytest.raises(ModuleNotFoundError, match="vllm.no_such_module"):
        _import("vllm.no_such_module")
    with pytest.raises(ModuleNotFoundError, match="vllm.envs.no_such_module"):
        _import("vllm.envs.no_such_module")

    def from_statement() -> object:
        from vllm import NoSuchName

        return NoSuchName

    with pytest.raises(ImportError, match="^cannot import name 'NoSuchName' from 'vllm'"):
        from_statement()


def test_a_name_the_test_did_not_set_refuses_every_use_and_names_its_entry(monkeypatch):
    fake_vllm.install(monkeypatch)
    from vllm import LLM

    def subclass() -> None:
        class Mine(LLM):
            pass

    for use in (lambda: LLM(model="m"), lambda: LLM.generate, subclass, lambda: isinstance(1, LLM),
                lambda: issubclass(int, LLM)):
        with pytest.raises(AssertionError, match=r"vllm:LLM was .*fake\.set\('vllm:LLM', value\)"):
            use()
    assert not hasattr(LLM, "__wrapped__")


def test_reading_off_a_module_what_no_test_set_is_an_attribute_error_that_says_what_to_do(monkeypatch):
    """`tests.fake_vllm.Module`: the error is the one any module gives, so ``hasattr``, a ``getattr`` with a default and an
    import of a name the surface lacks answer as before; its message names the assignment that sets the attribute and the
    other way out, for a library module that read it as it was itself imported."""
    fake = fake_vllm.install(monkeypatch)
    envs = fake["vllm.envs"]

    def read(name: str) -> object:
        return getattr(envs, name)

    def from_statement() -> object:
        from vllm.envs import NOT_SET

        return NOT_SET

    with pytest.raises(AttributeError) as refused:
        read("NOT_SET")
    said = str(refused.value)
    assert said.startswith("vllm.envs.NOT_SET was read, and this test did not set it: fake['vllm.envs'].NOT_SET = value. ")
    assert "import it before fake_vllm.install" in said
    assert not hasattr(envs, "NOT_SET") and getattr(envs, "NOT_SET", 7) == 7
    with pytest.raises(AttributeError, match="^module 'vllm.envs' has no attribute '__wrapped__'$"):
        read("__wrapped__")
    with pytest.raises(ImportError, match="^cannot import name 'NOT_SET' from 'vllm.envs'"):
        from_statement()
    envs.NOT_SET = 1
    assert envs.NOT_SET == 1 and from_statement() == 1


def test_a_module_that_acts_on_an_imported_vllm_module_is_imported_before_the_install(monkeypatch):
    """`tests.fake_vllm.install`: every module of the surface is imported from then on.  A switch that patches its site now
    when the site is imported, and waits for the site's import otherwise, patches nothing when it is armed before the
    install, as under the hand-written stub of ``vllm`` alone; armed after it, it reads a module on which nothing is set."""
    site, patched = "vllm.v1.sample.ops.topk_topp_triton", []

    def arm() -> None:
        if site in sys.modules:
            patched.append(sys.modules[site].kernel)

    for name in _vllm_modules():
        monkeypatch.delitem(sys.modules, name)
    arm()
    fake_vllm.install(monkeypatch)
    assert patched == [] and site in sys.modules
    with pytest.raises(AttributeError, match="import it before fake_vllm.install"):
        arm()


def test_a_test_sets_only_names_and_reads_only_modules_of_the_surface(monkeypatch):
    fake = fake_vllm.install(monkeypatch)

    class LLM:
        pass

    fake.set("vllm:LLM", LLM)
    fake.set("vllm.distributed:get_tensor_model_parallel_rank", lambda: 3)
    from vllm import LLM as bound
    from vllm.distributed import get_tensor_model_parallel_rank

    assert bound is LLM and get_tensor_model_parallel_rank() == 3
    for key in ("vllm:NoSuchName", "vllm.no_such_module:name", "vllm.envs"):
        with pytest.raises(AssertionError, match="is no name of vLLM's surface"):
            fake.set(key, 1)
    with pytest.raises(AssertionError, match="vllm.no_such_module is no module of vLLM's surface"):
        fake["vllm.no_such_module"]


def test_a_script_outside_the_library_names_its_own_imports(monkeypatch):
    fake = fake_vllm.install(monkeypatch, extra=["vllm.scalar_type:scalar_types", "vllm.models.common.ops"])
    fake.set("vllm.scalar_type:scalar_types", types.SimpleNamespace(float4_e2m1f="e2m1"))
    from vllm.scalar_type import scalar_types

    assert scalar_types.float4_e2m1f == "e2m1" and _import("vllm.models.common.ops") is fake["vllm.models.common.ops"]
    with pytest.raises(AssertionError, match="not vLLM's"):
        fake_vllm.install(monkeypatch, extra=["torch.nn:Module"])
