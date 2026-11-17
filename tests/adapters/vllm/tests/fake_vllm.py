"""One stand-in for vLLM, built from the closed list of what the library imports from it (`tests.lint.vllm_surface`).

    fake = install(monkeypatch)                                     # vLLM is importable for this test, and gone after it
    fake["vllm.envs"].VLLM_BATCH_INVARIANT = True                   # a module of the surface: set what the test reads
    fake.set("vllm.distributed:get_tensor_model_parallel_rank", lambda: 0)

`install` puts a module in ``sys.modules`` for every module `SURFACE` names and each package above one, each a child of its
parent, and takes out any other ``vllm`` module, so an import the surface doesn't list fails as it does where vLLM is absent.
A name the library imports from a module (``"vllm.mod:name"``) is the module of that name where the surface has one, and
otherwise `Unset`, which the import binds and any use refuses, naming the entry: a test sets what it reads and nothing else.

A stand-in can't drift from the surface: it holds no module the surface lacks, and `FakeVllm.set` and
``FakeVllm.__getitem__`` refuse a name or a module it doesn't list.  A test of a script outside the library (a GPU probe,
whose imports the surface doesn't hold) names them itself, in the surface's form, as ``extra``.

Every module of the surface is in ``sys.modules`` from `install` on, so library code that asks whether a vLLM module is
imported is told that it is.  A library module that acts on that as it is itself imported (a serving switch that arms
itself: it patches its site now when the site is imported, and hooks the site's import otherwise) then reads a module on
which nothing is set, and its import fails with what `Module` says.  A test that may be the first in its process to import
such a module imports it before `install`, where the hand-written stub it replaces never held the site.
"""
from __future__ import annotations

import sys
import types
from collections.abc import Iterable
from importlib.machinery import ModuleSpec
from typing import Any, NoReturn

import pytest

from tests.lint.vllm_surface import SURFACE

__all__ = ["FakeVllm", "Module", "Unset", "install"]

LIST = "tests/lint/vllm_surface.py"


class Unset:
    """A name of the surface that the test did not set: importing it works, using it fails."""

    def __init__(self, key: str) -> None:
        self._key = key

    def _refuse(self, use: str) -> NoReturn:
        raise AssertionError(f"{self._key} was {use}, and this test did not set it: fake.set({self._key!r}, value)")

    def __repr__(self) -> str:
        return f"<{self._key}, not set>"

    def __call__(self, *args: Any, **kwargs: Any) -> NoReturn:
        self._refuse("called")

    def __getattr__(self, name: str) -> NoReturn:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        self._refuse(f"read for .{name}")

    def __mro_entries__(self, bases: tuple) -> NoReturn:
        self._refuse("subclassed")

    def __instancecheck__(self, instance: object) -> NoReturn:
        self._refuse("the type of an isinstance")

    def __subclasscheck__(self, subclass: type) -> NoReturn:
        self._refuse("the type of an issubclass")


class Module(types.ModuleType):
    """A module of the stand-in.  A test sets on it what the library reads from the module; declaring ``__setattr__`` is what
    lets a typed test write ``fake["vllm.envs"].NAME = value``, which mypy refuses on a plain `types.ModuleType`.  Reading
    what no test set is the `AttributeError` any module gives, so ``hasattr`` and a ``getattr`` with a default answer as
    they do on vLLM's own; its message says what to do."""

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)

    def __getattr__(self, name: str) -> Any:
        module = self.__name__
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(f"module {module!r} has no attribute {name!r}")
        raise AttributeError(f"{module}.{name} was read, and this test did not set it: fake[{module!r}].{name} = value. "
                             "Read by a library module as that module was itself imported? Then it acts on a vLLM module "
                             "that is already imported: import it before fake_vllm.install (tests/fake_vllm.py's docstring).")


def split(key: str) -> tuple[str, str]:
    """An entry's module and name (``""`` for a module imported whole)."""
    module, _, name = key.partition(":")
    return module, name


def packages(module: str) -> list[str]:
    """``module`` and each package above it, outermost first."""
    parts = module.split(".")
    return [".".join(parts[:i]) for i in range(1, len(parts) + 1)]


class FakeVllm:
    """The installed stand-in: its modules by name, and the entries it was built from."""

    def __init__(self, keys: Iterable[str]) -> None:
        self.keys = frozenset(keys)
        outside = sorted(k for k in self.keys if split(k)[0].split(".")[0] != "vllm")
        assert not outside, f"not vLLM's: {outside}"
        names = sorted({p for k in self.keys for p in packages(split(k)[0])})
        self.modules = {n: Module(n) for n in names}
        for n, m in self.modules.items():
            is_package = any(other.startswith(n + ".") for other in names)
            m.__spec__ = ModuleSpec(n, None, is_package=is_package)
            if is_package:
                m.__path__ = []
            parent, _, leaf = n.rpartition(".")
            if parent:
                setattr(self.modules[parent], leaf, m)
        for k in sorted(self.keys):
            module, name = split(k)
            if name and f"{module}.{name}" not in self.modules:
                setattr(self.modules[module], name, Unset(k))

    def __getitem__(self, module: str) -> Module:
        assert module in self.modules, f"{module} is no module of vLLM's surface ({LIST}): the library does not import it"
        return self.modules[module]

    def set(self, key: str, value: Any) -> None:
        """Give the name of entry ``key`` (``"vllm.mod:name"``) its value for this test."""
        module, name = split(key)
        assert name and key in self.keys, f"{key} is no name of vLLM's surface ({LIST}): the library does not import it"
        setattr(self.modules[module], name, value)


def install(monkeypatch: pytest.MonkeyPatch, *, extra: Iterable[str] = ()) -> FakeVllm:
    """Make a new stand-in the only vLLM in ``sys.modules`` until ``monkeypatch`` undoes it.  Each call starts over, an
    earlier stand-in's modules among what it takes out: a helper takes the stand-in its test installed, and an install under
    ``monkeypatch.context()`` gives way to the outer one at the context's end."""
    fake = FakeVllm([*SURFACE, *extra])
    for name in [n for n in sys.modules if n.split(".")[0] == "vllm"]:
        monkeypatch.delitem(sys.modules, name)
    for name, module in fake.modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return fake
