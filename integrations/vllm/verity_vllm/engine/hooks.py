"""The one owner of runtime patches: every attribute the integration replaces on a vLLM, torch or Triton object (a model
runner's methods, a layer's ``forward``, a kernel entry point, a collective, an op namespace entry, an export shim) is
installed through this module, and every one can be put back.

`patch(owner, name, value)` records what ``owner`` itself held under ``name`` (its own ``__dict__`` entry, or that it
had none because the attribute was inherited, computed or missing) and returns a `Patch`.  `Patch.uninstall` restores
exactly that: afterwards ``vars(owner).get(name)`` is the object it was, and an inherited method is inherited again.
`Hooks` is a set of patches one component installs together and uninstalls newest first.  `live()` lists the patches
installed in this process and not yet uninstalled.  A `Service` is an in-process service a component starts beside its
patches (a responder thread); it is an entry like a patch, and uninstalling it stops it.  Entries made through a
``Hooks(tag=...)`` carry its tag, so `live(tag)` finds what one component (a protocol option) installed.

The patch sites keep their wrappers (they close over the component's state); this module holds the mechanism and the
targets more than one site patches.  It imports nothing from ``verity_vllm`` and nothing from vLLM, torch or Triton at
import time, so every layer may use it.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, MutableSequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any


class _Absent:
    __slots__ = ()

    def __repr__(self) -> str:
        return "ABSENT"


#: ``own(owner, name)`` when ``owner`` holds nothing under ``name`` itself.
ABSENT: Any = _Absent()


def own(owner: Any, name: str) -> Any:
    """What ``owner`` itself holds under ``name`` (not inherited, not produced by ``__getattr__``), or `ABSENT`.  An object
    without a ``__dict__`` (slots, extension types) answers with ``getattr``."""
    try:
        d = vars(owner)
    except TypeError:
        return getattr(owner, name, ABSENT)
    return d.get(name, ABSENT)


_LIVE: list[Any] = []


@dataclass(eq=False)
class Patch:
    """``owner.name`` replaced by ``value``.  ``prior`` is what ``owner`` itself held before (`ABSENT` if nothing);
    ``original`` is what ``getattr(owner, name)`` returned before, the callable a wrapper forwards to.  ``by_value`` is
    set when the owner does not keep the attribute in its ``__dict__`` (a config proxy, a property): uninstall then sets
    ``original`` back through the owner's own ``__setattr__``."""

    owner: Any
    name: str
    value: Any
    prior: Any
    original: Any
    by_value: bool = False
    installed: bool = True
    tag: str | None = None

    def uninstall(self) -> None:
        """Put back what the owner held: set ``prior`` again, or remove the own attribute when there was none.
        Idempotent."""
        if not self.installed:
            return
        self.installed = False
        _forget(self)
        if self.by_value:
            if self.original is ABSENT:
                delattr(self.owner, self.name)
            else:
                setattr(self.owner, self.name, self.original)
        elif self.prior is ABSENT:
            if own(self.owner, self.name) is not ABSENT:
                delattr(self.owner, self.name)
        else:
            setattr(self.owner, self.name, self.prior)


@dataclass(eq=False)
class Appended:
    """``item`` appended to a hook list a library reads (for example Inductor's ``INTERMEDIATE_HOOKS``)."""

    seq: MutableSequence[Any]
    item: Any
    installed: bool = True
    tag: str | None = None

    def uninstall(self) -> None:
        """Remove ``item`` (this object, not an equal one) if it is still in the list.  Idempotent."""
        if not self.installed:
            return
        self.installed = False
        _forget(self)
        for i, x in enumerate(self.seq):
            if x is self.item:
                del self.seq[i]
                break


@dataclass(eq=False)
class Service:
    """An in-process service started beside a component's patches: ``handle`` is what ``start()`` returned, and
    `uninstall` calls ``stop(handle)`` once, keeping its return value (the service's counters) in ``result``."""

    name: str
    handle: Any
    stop: Callable[[Any], Any]
    installed: bool = True
    tag: str | None = None
    result: Any = None

    def uninstall(self) -> None:
        """Stop the service.  Idempotent: a stop that raised is not retried."""
        if not self.installed:
            return
        self.installed = False
        _forget(self)
        self.result = self.stop(self.handle)


def _forget(entry: Any) -> None:
    for i, x in enumerate(_LIVE):
        if x is entry:
            del _LIVE[i]
            return


def patch(owner: Any, name: str, value: Any) -> Patch:
    """Set ``owner.name = value`` and return the `Patch` that undoes it."""
    prior, original = own(owner, name), getattr(owner, name, ABSENT)
    setattr(owner, name, value)
    p = Patch(owner, name, value, prior, original, by_value=own(owner, name) is not value)
    _LIVE.append(p)
    return p


def wrap(owner: Any, name: str, make: Callable[[Any], Any]) -> Patch:
    """Replace ``owner.name`` with ``make(original)``, where ``original`` is ``getattr(owner, name)`` now."""
    return patch(owner, name, make(getattr(owner, name)))


def append(seq: MutableSequence[Any], item: Any) -> Appended:
    """Append ``item`` to a library's hook list and return the entry that removes it."""
    seq.append(item)
    a = Appended(seq, item)
    _LIVE.append(a)
    return a


@contextmanager
def patched(owner: Any, name: str, value: Any) -> Iterator[Patch]:
    """``owner.name = value`` for the duration of the block."""
    p = patch(owner, name, value)
    try:
        yield p
    finally:
        p.uninstall()


def service(name: str, start: Callable[[], Any], stop: Callable[[Any], Any]) -> Service:
    """Start a service now (``handle = start()``) and return the `Service` whose uninstall calls ``stop(handle)``."""
    s = Service(name, start(), stop)
    _LIVE.append(s)
    return s


def live(tag: str | None = None) -> tuple[Patch | Appended | Service, ...]:
    """The entries installed in this process and not yet uninstalled, oldest first; with ``tag``, only those carrying it."""
    return tuple(e for e in _LIVE if tag is None or e.tag == tag)


class Hooks:
    """Patches (and services) one component installs together; `uninstall` removes them newest first, so stacked patches
    of one attribute unwind to the original and a service stops before the patches it serves come off.  Every entry
    carries ``tag``.  Usable as a context manager."""

    def __init__(self, tag: str | None = None) -> None:
        self.tag = tag
        self._entries: list[Patch | Appended | Service] = []

    def _add(self, e: Any) -> Any:
        e.tag = self.tag
        self._entries.append(e)
        return e

    def patch(self, owner: Any, name: str, value: Any) -> Patch:
        return self._add(patch(owner, name, value))

    def wrap(self, owner: Any, name: str, make: Callable[[Any], Any]) -> Patch:
        return self._add(wrap(owner, name, make))

    def append(self, seq: MutableSequence[Any], item: Any) -> Appended:
        return self._add(append(seq, item))

    def service(self, name: str, start: Callable[[], Any], stop: Callable[[Any], Any]) -> Service:
        return self._add(service(name, start, stop))

    @property
    def entries(self) -> tuple[Patch | Appended | Service, ...]:
        return tuple(e for e in self._entries if e.installed)

    def uninstall(self) -> None:
        """Uninstall every entry, newest first.  An entry that raises (a service's stop) does not leave the older ones
        installed: the first error is raised once all of them are off."""
        err: BaseException | None = None
        while self._entries:
            try:
                self._entries.pop().uninstall()
            except BaseException as e:  # noqa: BLE001
                err = err or e
        if err is not None:
            raise err

    def __enter__(self) -> Hooks:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.uninstall()


# --- targets more than one site patches --------------------------------------------------------------------------------

def triton_launch(make: Callable[[Any], Callable[..., Any]]) -> Patch:
    """Replace ``triton.runtime.jit.JITFunction.run``, the entry point of every direct Triton launch, with
    ``make(current run)``."""
    from triton.runtime.jit import JITFunction
    return wrap(JITFunction, "run", make)
