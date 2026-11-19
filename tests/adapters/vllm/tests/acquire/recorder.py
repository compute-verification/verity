"""A committer that records what a source acquires and attaches, and whether a source's declared evidence is a module of the tree."""
from __future__ import annotations

import importlib.util
import types
from typing import Any


class Recorder:
    def __init__(self) -> None:
        self._cur: types.SimpleNamespace | None = None
        self.errors: list[str] = []
        self.acquired: list[tuple[str, Any]] = []
        self.sources: list = []
        self.name = "rec"
        self.model = None

    def _acquire(self, name, t, job):
        self.acquired.append((name, t.clone()))

    def attach_source(self, src):
        self.sources.append(src)


def exactness_in_tree(source: Any) -> bool:
    """Whether each class a source declares ends its evidence on a module that the tree has: the one that holds the exactness check of
    what the source commits.  A name that is no module sends the declaration's reader nowhere."""
    return all(importlib.util.find_spec(d["evidence"].rsplit(" ", 1)[-1]) is not None for d in source.declaration_classes())
