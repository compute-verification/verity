"""Every number the model reads, with where it comes from: one registry, addressed by dotted keys.

The registry is the TOML files under `data/`, one per layer. A leaf is either a bare value (a choice, such as a kind or a
name) or a parameter: an inline table with `value` and optionally `unit`, `low` and `high` (the plausible range that
sweeps and tornados span), `source` (a datasheet, a measurement, a Lean name, a ruling) and `status`
(`datasheet`, `measured`, `declared`, `assumed`, `ruled`, `conjecture`)::

    [location.h200-hgx]
    phi = { value = 1.583e16, unit = "FLOP/s", source = "census h100-sxm5-80gb/e4m3 x 8", status = "datasheet" }

What another file states is read from it, not restated in a layer (`sourced`; a file that sets one of these keys is
refused). From `verity`: the monitoring periods are core's (`PERIODS`), each the registry's `horizon.<name>.seconds`; the
bits an incorrect proof block injects are the widest public size class of a block's outputs (`BLOCK_OUTPUT_BITS`); the
per-window δ is core's (`profile.WINDOW_DELTA`); and the incorrect-outputs row reports at core's reporting δs. From the
inference-only policy's calibrations (`benchmarks/nci/calibrations.json`): the calibrated policy's `K`, its range, and
`Γ`. The session policy's `K` is a ruling its layer states, and only its range is the calibrations' (`ranges`; a file
that gives such a key a range is refused).

The census's numbers are the exception, by the rule that `verity` takes the census's numbers as parameters and never
reads it: nothing in this package reads the census. The layers state them, each naming the census entry it is (the
reference datacenter's machines, `DATACENTER_FLEETS`; the world's active parameters; the session policy's `Γ` at the
world's model), and the tests hold each equal to the census.

A table becomes an object through its layer's `build`, which takes its fields by name (`Registry.fields`). Every
parameter of every layer is then one key, so a sweep is `registry.override({key: value})` and a rebuild, whatever the
layer.
"""

from __future__ import annotations

import json
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

from verity.core.protocols import profile, reporting
from verity.ml.circuits import partition_v2

DATA = Path(__file__).resolve().parent.parent / "data"
#: the inference-only policy's calibrations, in the directory beside this package's
CALIBRATIONS = Path(__file__).resolve().parents[2] / "nci" / "calibrations.json"

STATUSES = frozenset({"datasheet", "measured", "declared", "assumed", "ruled", "conjecture", "derived"})

#: core's monitoring periods (`reporting`), seconds by name
PERIODS = dict(reporting.PERIODS)
#: the keys that hold the output bits an accepted incorrect proof block injects: a block's outputs fit its Definition's
#: public size class, so the widest class (`partition_v2.UNIT_CLASS_MAX`), over the classes down to the smallest
BLOCK_OUTPUT_BITS = ("regime.remote-sampled.step_bits", "incorrect.step_bits")
#: the census entries whose numbers the layers state, for the tests that hold them equal: the reference datacenter and
#: the fleets that are it, a machine a location; and the adversary, the world's inference, with the registry's model it
#: serves
DATACENTER = "dc-12500-machines-600s"
DATACENTER_FLEETS = ("reference-h200", "reference-h100", "reference-b200")
WORLD = "world-inference-2027"
WORLD_MODEL = "deepseek-v4-pro"
#: the calibration the policy of its name takes `K` and `Γ` from; the session policy, the reference datacenter's at the
#: world's model, whose `K` is ruled at the same size; and the calibrations at the ends of `K`'s range for both (the
#: coded prompts, the write-up's `K`)
CALIBRATION = "kimi-k2-1m-raw"
SESSION = "v4-pro-session"
K_RANGE = ("kimi-k2-1m-coded", "kimi-k2-11mb-raw")


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
        params, spans = {p.key: p for p in sourced()}, ranges()
        for path in sorted(paths if paths is not None else DATA.glob("*.toml")):
            with open(path, "rb") as f:
                for key, p in _flatten(tomllib.load(f)):
                    if key in params:
                        raise ValueError(f"{key} is set twice (again in {path.name})")
                    if key in spans:
                        if p.low is not None or p.high is not None:
                            raise ValueError(f"{key}'s range is read from its source, not set (in {path.name})")
                        p = replace(p, low=spans[key][0], high=spans[key][1])
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


def entry(path: Path, name: str) -> dict:
    """The entry `name` of a JSON list of entries by `id` (the calibrations)."""
    return next(e for e in json.loads(path.read_text()) if e["id"] == name)


def sourced() -> list[Param]:
    """The parameters another file states, read from it: no file under `data/` restates one (`Registry.load` refuses a
    key set twice). The census's numbers are not among them (module docstring)."""
    periods = [Param(f"horizon.{name}.seconds", s, "s", source="verity.core.protocols.reporting PERIODS",
                     status="declared") for name, s in PERIODS.items()]
    widest, smallest = partition_v2.UNIT_CLASS_MAX, partition_v2.UNIT_CLASSES[0]
    bits = [Param(key, widest, "bit", low=smallest, high=widest,
                  source="verity.ml.circuits.partition_v2 UNIT_CLASS_MAX: a proof block's outputs fit its Definition's "
                         "public size class, and an incorrect block injects the widest; low: UNIT_CLASSES' smallest",
                  status="derived") for key in BLOCK_OUTPUT_BITS]
    return periods + bits + _deltas() + _policies()


def _deltas() -> list[Param]:
    window, report = float(profile.WINDOW_DELTA), [float(d) for d in reporting.DELTAS]
    core = "verity.core.protocols.profile WINDOW_DELTA, the per-window delta"
    credit = "CircuitCredit's per-window failure p = delta_s + delta_bind + delta_ks, three of it (eps_agg's source)"
    return [
        Param("regime.remote.window_delta_coin", window, status="derived",
              source=f"{core}: {credit}: delta_s, the coins' sample"),
        Param("regime.remote.window_delta_crypto", 2 * window, status="derived",
              source=f"{core}: {credit}: delta_bind + delta_ks, commitment binding (SHA-512) and the proof system's "
                     "knowledge soundness"),
        Param("regime.remote-sampled.p_evade", window, low=1e-15, high=2.0 ** -20, status="declared",
              source=f"{core}, the planner's (benchmarks/nci/tiers.py DEFAULTS p_evade); high: 2^-20"),
        Param("incorrect.delta", min(report), low=min(report), high=max(report), status="ruled",
              source="verity.core.protocols.reporting DELTAS, the deltas a period's bounds are stated at: the smaller; "
                     "high: the larger (ruled: the reference datacenter, wherever a probabilistic term enters)"),
    ]


def ranges() -> dict[str, tuple[float, float]]:
    """The ranges another file states for a parameter a layer sets: the layer states the value and no range
    (`Registry.load` refuses one). The session policy's `K` is its layer's, a ruling; its range is the calibrations'."""
    return {f"policy.{SESSION}.k": _k_range()}


def _k_range() -> tuple[float, float]:
    low, high = (entry(CALIBRATIONS, name)["K"] for name in K_RANGE)
    return low, high


def _policies() -> list[Param]:
    cal = entry(CALIBRATIONS, CALIBRATION)
    low, high = _k_range()
    where = f"benchmarks/nci/calibrations.json {CALIBRATION}"
    return [
        Param(f"policy.{CALIBRATION}.gamma", cal["Gamma"], "FLOP", status="declared",
              source=f"{where} Gamma: 1M tokens x 2 x 32B active"),
        Param(f"policy.{CALIBRATION}.k", cal["K"], "bit", low, high,
              f"{where} K (2.5 MB); low: {K_RANGE[0]}; high: {K_RANGE[1]}", "declared"),
    ]


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
