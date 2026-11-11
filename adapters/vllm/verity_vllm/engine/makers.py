"""The parts plugins add to one seam (`Makers`): functions added by name, each making its plugin's part for one vLLM
object (a scheduler, a model runner, a communicator), made once per object in the order the seam declares.

A seam declares its parts' names in the order they run, and refuses a name it doesn't declare, a name added twice, and a
part added once some object has made its parts, which would leave that object without it: plugins add theirs from their
``register``, before vLLM builds the objects. An object is made its parts at its first use, and a maker that returns None
gives that object no part.
"""
from __future__ import annotations

import weakref
from collections.abc import Callable
from typing import Any, Generic, TypeVar

__all__ = ["Makers"]

T = TypeVar("T")


class Makers(Generic[T]):
    """The makers added to one seam: ``names`` declares every part's name in the order the parts run, and ``where`` says
    where in a refusal (``"engine.step_end's STEPS"``)."""

    def __init__(self, names: tuple[str, ...], where: str) -> None:
        self.names, self.where = names, where
        self._makers: dict[str, Callable[[Any], T | None]] = {}
        self._made: weakref.WeakKeyDictionary[Any, tuple[T, ...]] = weakref.WeakKeyDictionary()

    def add(self, name: str, make: Callable[[Any], T | None]) -> None:
        """Add the part ``name``: ``make(obj)`` makes it for one object, or returns None where that object has none."""
        if name not in self.names:
            raise ValueError(f"{name!r} is not one of {self.names}: give it its place in {self.where}")
        if name in self._makers:
            raise ValueError(f"{name!r} is already added to {self.where}")
        if len(self._made):
            raise RuntimeError(f"{name!r} was added to {self.where} after an object in this process made its parts; add "
                               "each from a plugin's register, before vLLM builds the engine")
        self._makers[name] = make

    @property
    def added(self) -> tuple[str, ...]:
        """The names added so far, in `names` order."""
        return tuple(n for n in self.names if n in self._makers)

    def made(self, obj: Any) -> tuple[T, ...]:
        """``obj``'s parts in `names` order, made at the first call for it and the same tuple after."""
        parts = self._made.get(obj)
        if parts is None:
            made = [self._makers[n](obj) for n in self.added]
            parts = tuple(p for p in made if p is not None)
            self._made[obj] = parts
        return parts

    def clear(self) -> None:
        """Forget every maker and every object's parts."""
        self._makers.clear()
        self._made.clear()
