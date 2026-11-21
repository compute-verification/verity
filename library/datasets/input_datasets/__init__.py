"""Input datasets: what a workload reads its circuits' inputs from, each a definition under `definitions/`, read in order."""
from __future__ import annotations

import hashlib
import json
import random
import tomllib
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

from input_datasets.generators import GENERATORS

DEFINITIONS = Path(__file__).resolve().parents[1] / "definitions"
#: where a recorded dataset's bytes are read from: the checkout, where `research data fetch-fixtures` puts each entry of
#: `artifacts.json` at its path
CACHE = Path(__file__).resolve().parents[3]
DIGESTS = ("sha256", "sha512")


class Refused(Exception):
    """A read refused by ``name`` (``unknown-dataset``, ``unknown-split``, ``digest``, ``missing``, ``timed-twice``)."""

    def __init__(self, name: str, text: str) -> None:
        super().__init__(text)
        self.name = name


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class Dataset:
    """Records in order, each a plain JSON value nothing here interprets; read afresh on each iteration."""

    def __init__(self, kind: str, identity: str, read: Callable[[], list[Any]],
                 splits: Mapping[str, tuple[int, int]] | None = None) -> None:
        self.kind, self.identity, self._read, self._ranges = kind, identity, read, dict(splits or {})
        self._timed = False

    def __iter__(self) -> Iterator[Any]:
        return iter(self._read())

    def __len__(self) -> int:
        return len(self._read())

    def _derived(self, step: str, pick: Callable[[list[Any]], list[Any]]) -> Dataset:
        ds = Dataset(self.kind, f"{self.identity}|{step}", lambda: pick(self._read()))
        ds._timed = self._timed
        return ds

    @property
    def splits(self) -> dict[str, Dataset]:
        return {name: self._derived(f"split:{name}={a}:{b}", lambda r, a=a, b=b: r[a:b]) for name, (a, b) in self._ranges.items()}

    def sample(self, n: int, seed: int) -> Dataset:
        """``n`` records drawn without replacement under ``seed``, in their order here."""
        picked = sorted(random.Random(seed).sample(range(len(self)), n))
        return self._derived(f"sample:n={n},seed={_canonical(seed)}", lambda r: [r[i] for i in picked])

    def shuffled(self, seed: int) -> Dataset:
        def pick(r: list[Any]) -> list[Any]:
            random.Random(seed).shuffle(r)
            return r
        return self._derived(f"shuffle:seed={_canonical(seed)}", pick)

    def timed(self, rule: str, *, seed: int = 0, **parameters: Any) -> Dataset:
        """Each record as ``[seconds from the start, record]``, in order, times never decreasing; the kind stays the
        parent's, since the pairing is this view's. ``at-once``: every time 0.0 (no seed). ``rate``: exponential gaps at
        ``rate`` a second, drawn from ``seed``. Refused (``timed-twice``) on a dataset already timed."""
        if self._timed:
            raise Refused("timed-twice", f"{self.identity} is timed already: time the dataset it was timed from")
        if rule == "at-once" and not parameters:
            step, times = "timed:at-once", lambda n: [0.0] * n
        elif rule == "rate" and set(parameters) == {"rate"}:
            rate = float(parameters["rate"])

            def times(n: int) -> list[float]:
                rng, at, out = random.Random(seed), 0.0, []
                for _ in range(n):
                    at += rng.expovariate(rate)
                    out.append(at)
                return out
            step = f"timed:rate,rate={_canonical(rate)},seed={_canonical(seed)}"
        else:
            raise ValueError(f"timed takes at-once, or rate with rate=R, not {rule!r} with {sorted(parameters)}")
        ds = self._derived(step, lambda r: [[t, x] for t, x in zip(times(len(r)), r)])
        ds._timed = True
        return ds


def _names() -> list[str]:
    return sorted(p.stem for p in DEFINITIONS.glob("*.toml"))


def _recorded(name: str, r: Mapping[str, Any]) -> tuple[str, Callable[[], list[Any]]]:
    if not str(r["source"]).startswith("art:"):
        raise ValueError(f"{name}: source {r['source']!r} is not an `art:` id, the one source read so far")
    algorithm = next(a for a in DIGESTS if a in r)
    path, want, fetch = CACHE / r["path"], r[algorithm], f"research data fetch-fixtures {r['path']}"
    if r["format"] not in ("jsonl", "json"):
        raise ValueError(f"{name}: format {r['format']!r} is not jsonl or json")

    def read() -> list[Any]:
        if not path.is_file():
            raise Refused("missing", f"input dataset {name!r}: its bytes are not at {r['path']}; fetch them with `{fetch}`")
        data = path.read_bytes()
        if (got := hashlib.new(algorithm, data).hexdigest()) != want:
            raise Refused("digest", f"input dataset {name!r}: {r['path']} has {algorithm} {got}, not its definition's "
                                    f"{want}; fetch it again with `{fetch}`")
        text = data.decode()
        if r["format"] == "jsonl":
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        return json.loads(text)[r["records"]]

    shape = r["format"] + (f":{r['records']}" if r["format"] == "json" else "")
    return f"recorded:{algorithm}={want}:{shape}", read


def _generated(name: str, g: Mapping[str, Any]) -> tuple[str, Callable[[], list[Any]]]:
    fn = GENERATORS.get((g["generator"], g["version"]))
    if fn is None:
        raise ValueError(f"{name}: no generator {g['generator']!r} at version {g['version']!r}")
    seed, params = g["seed"], g.get("parameters", {})
    digest = hashlib.sha256(_canonical(params).encode()).hexdigest()
    return (f"generated:{g['generator']}/v{g['version']}:seed={_canonical(seed)}:parameters-sha256={digest}",
            lambda: fn(seed, params))


def open(name: str, split: str | None = None) -> Dataset:
    """The input dataset ``name`` (a file ``definitions/NAME.toml``), or its split ``split``."""
    if name not in _names():
        raise Refused("unknown-dataset", f"no input dataset {name!r}: each is a definition NAME.toml under {DEFINITIONS}")
    d = tomllib.loads((DEFINITIONS / f"{name}.toml").read_text())
    identity, read = _recorded(name, d["recorded"]) if "recorded" in d else _generated(name, d["generated"])
    ranges = {k: (int(a), int(b)) for k, (a, b) in d.get("splits", {}).items()}
    ds = Dataset(d["kind"], identity, read, ranges)
    if split is None:
        return ds
    if split not in ranges:
        raise Refused("unknown-split", f"input dataset {name!r} has no split {split!r} (its splits: {', '.join(ranges) or 'none'})")
    return ds.splits[split]
