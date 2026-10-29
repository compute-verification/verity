"""Every number the model reads, with where it comes from: one registry, addressed by dotted keys.

The registry is the TOML files under `data/`, one per layer. A leaf is either a bare value (a choice, such as a kind or a
name) or a parameter: an inline table with `value` and optionally `unit`, `low` and `high` (the plausible range that
sweeps and tornados span), `source` (a datasheet, a measurement, a Lean name, a ruling) and `status`
(`datasheet`, `measured`, `declared`, `assumed`, `ruled`, `conjecture`)::

    [location.h200-hgx]
    phi = { value = 1.583e16, unit = "FLOP/s", source = "census h100-sxm5-80gb/e4m3 x 8", status = "datasheet" }

A table becomes an object through its layer's `build`, which takes its fields by name (`Registry.fields`). Every
parameter of every layer is then one key, so a sweep is `registry.override({key: value})` and a rebuild, whatever the
layer.
"""

from __future__ import annotations

import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

DATA = Path(__file__).resolve().parent.parent / "data"

STATUSES = frozenset({"datasheet", "measured", "declared", "assumed", "ruled", "conjecture", "derived"})


@dataclass(frozen=True)
class Param:
    key: str
    value: Any
    unit: str = ""
    low: float | None = None
    high: float | None = None
    source: str = ""
    status: str = ""

    @property
    def ranged(self) -> bool:
        return self.low is not None and self.high is not None


class Registry:
    def __init__(self, params: Mapping[str, Param], seen: set[str] | None = None,
                 index: Mapping[str, tuple[str, ...]] | None = None):
        self._params = dict(params)
        #: the keys read so far, when tracking (`tracked`)
        self.seen = seen
        #: each table's leaf keys, by the table's prefix; shared by every registry with the same keys
        self._index = index

    def _tables(self) -> Mapping[str, tuple[str, ...]]:
        if self._index is None:
            index: dict[str, list[str]] = {}
            for k in self._params:
                index.setdefault(k.rpartition(".")[0], []).append(k)
            self._index = {t: tuple(ks) for t, ks in index.items()}
        return self._index

    @classmethod
    def load(cls, paths: Iterable[Path] | None = None) -> Registry:
        params: dict[str, Param] = {}
        for path in sorted(paths if paths is not None else DATA.glob("*.toml")):
            with open(path, "rb") as f:
                for key, p in _flatten(tomllib.load(f)):
                    if key in params:
                        raise ValueError(f"{key} is set twice (again in {path.name})")
                    params[key] = p
        return cls(params)

    def __contains__(self, key: str) -> bool:
        return key in self._params

    def __getitem__(self, key: str) -> Any:
        return self.param(key).value

    def get(self, key: str, default: Any = None) -> Any:
        return self.param(key).value if key in self._params else default

    def param(self, key: str) -> Param:
        try:
            p = self._params[key]
        except KeyError:
            raise KeyError(f"no parameter {key!r} in the registry (data/*.toml)") from None
        if self.seen is not None:
            self.seen.add(key)
        return p

    def tracked(self) -> Registry:
        """This registry, recording in `seen` every key read through it."""
        return Registry(self._params, set(), self._tables())

    def params(self) -> list[Param]:
        return list(self._params.values())

    def names(self, prefix: str) -> list[str]:
        """The tables directly under `prefix` (`names("fleet")` lists the fleets)."""
        head = prefix + "."
        return sorted({t[len(head):].split(".")[0] for t in self._tables() if t.startswith(head)})

    def table(self, prefix: str) -> dict[str, Any]:
        """The leaves directly in table `prefix`, by their last name."""
        keys = self._tables().get(prefix)
        if not keys:
            raise KeyError(f"no table {prefix!r} in the registry (data/*.toml)")
        return {k[len(prefix) + 1:]: self.param(k).value for k in keys}

    def fields(self, cls: type, prefix: str, skip: Iterable[str] = (), **given: Any) -> Any:
        """An instance of dataclass `cls` from table `prefix`: each field from the table (or `given`), and no other keys
        in the table than fields, `kind`, `name` and `skip` (the keys a caller reads and converts itself)."""
        table = {k: v for k, v in self.table(prefix).items() if k not in set(skip)}
        names = {f.name for f in fields(cls)}
        stray = set(table) - names - {"kind", "name"}
        if stray:
            raise ValueError(f"{prefix}: {sorted(stray)} are not fields of {cls.__name__}")
        values = {k: v for k, v in table.items() if k in names} | given
        if "name" in names and "name" not in values:
            values["name"] = prefix.split(".")[-1]
        return cls(**values)

    def override(self, values: Mapping[str, Any]) -> Registry:
        """A registry with these values in place of the old (each key must exist: a typo is an error, not a no-op). A
        `Param` replaces the whole parameter: its range, source and status with its value."""
        params = dict(self._params)
        for key, value in values.items():
            old = self.param(key)
            params[key] = replace(value, key=key) if isinstance(value, Param) else replace(old, value=value)
        return Registry(params, index=self._tables())


def _flatten(table: Mapping[str, Any], prefix: str = "") -> Iterable[tuple[str, Param]]:
    for name, v in table.items():
        key = f"{prefix}{name}"
        if isinstance(v, dict) and "value" in v:
            stray = set(v) - {"value", "unit", "low", "high", "source", "status", "note"}
            if stray:
                raise ValueError(f"{key}: unknown parameter fields {sorted(stray)}")
            status = v.get("status", "")
            if status and status not in STATUSES:
                raise ValueError(f"{key}: status {status!r} is not one of {sorted(STATUSES)}")
            yield key, Param(key, v["value"], v.get("unit", ""), v.get("low"), v.get("high"), v.get("source", ""),
                             status)
        elif isinstance(v, dict):
            yield from _flatten(v, key + ".")
        else:
            yield key, Param(key, v)
