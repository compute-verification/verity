"""The derive's storage aliases (`Ctx.storage`): node -> the storage base node it views, with a reverse index, so an in-place write
finds the other aliases of its base without scanning every node the export defined (quadratic in the graph)."""
from __future__ import annotations

from typing import Any


class AliasMap(dict):
    """A `dict` node -> base whose `aliases(base, node)` is `[n for n, b in self.items() if b is base and n is not node]`, in that order,
    from an index kept on every write (`[k] = v`, `setdefault`, `pop`, `del`)."""

    def __init__(self) -> None:
        super().__init__()
        self._pos: dict[Any, int] = {}              # a key's position in the dict's order (set on insertion, kept on update)
        self._by_base: dict[int, dict[Any, None]] = {}
        self._n = 0

    def __setitem__(self, node: Any, base: Any) -> None:
        if node in self:
            self._by_base[id(dict.__getitem__(self, node))].pop(node, None)
        else:
            self._pos[node] = self._n
            self._n += 1
        dict.__setitem__(self, node, base)
        self._by_base.setdefault(id(base), {})[node] = None

    def setdefault(self, node: Any, base: Any = None) -> Any:
        if node not in self:
            self[node] = base
        return dict.__getitem__(self, node)

    def pop(self, node: Any, *default: Any) -> Any:
        if node in self:
            self._by_base[id(dict.__getitem__(self, node))].pop(node, None)
            del self._pos[node]
        return dict.pop(self, node, *default)

    def __delitem__(self, node: Any) -> None:
        self.pop(node)

    def update(self, *a: Any, **kw: Any) -> None:
        for k, v in dict(*a, **kw).items():
            self[k] = v

    def aliases(self, base: Any, node: Any) -> list[Any]:
        """The nodes whose base is `base` (by identity), `node` excluded, in the dict's order."""
        return sorted((n for n in self._by_base.get(id(base), ()) if n is not node and dict.__getitem__(self, n) is base), key=self._pos.__getitem__)
