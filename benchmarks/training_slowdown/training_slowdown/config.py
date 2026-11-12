"""An attack config: one TOML file naming a start, a route and its dials, the training budget, the seeds, which of the
start's data the job is served, and what the grade reads (the target gain, the policy, the honest config). `load(path)`
reads one and refuses, by name, a key the schema lacks, a start the registry lacks, a route `routes/` lacks, a dial the
route lacks, of another type or of a value its plan refuses, two dials written of which the route reads one, a `windows`
outside `WINDOWS`, a policy neither the calculator's registry nor `policy.py` names, and an honest config at another
start or gain.

    start = "s-t20"                       # a start in starts.toml
    route = "subset"                      # routes/subset.py
    target = { gain = 1.1 }               # effective compute over the start's own, for the grade
    budget = { steps = 1400, batch = 64 } # training steps of `batch` windows; `minutes` stops a run past it
    seeds = { search = [0], final = [0, 101, 102] }
    policy = "console"                    # policy.CONSOLE's rule, or a [policy.*] table of verity/protocols/inference_only/rho/data
    honest = "s-t20-1.1x-reference"       # another config beside this one, at the same start and gain
    windows = "fresh"                     # only the windows the start's own training did not read

    [dials]                               # the route's DIALS; any other key is refused
    trainable = "blocks.5.*,norm"

Which windows a job is served is the harness's, and a config's to ask for, never a route's. Without `windows` a job is
served every window of the start's data, in its seed's order. With `windows = "fresh"` it is served that order without
the windows of any part the start's own training read (`training_slowdown.data.Stream`), and its record lists the
windows it was served; the job refuses the config on a start whose installed `start.json` doesn't name the parts it
read (`training_slowdown.starts.parts_read`).

The digest (`Config.digest`) is the sha256 of the canonical JSON of what decides a job's run (`Config.run`): the start
and its θ's file, its data, the route and its VERSION, every dial (the route's defaults filled in), the budget and,
only where the config asks, `windows`, less the dials of the route's `LATER` that the config leaves at their default:
such a dial's default is the step without the dial, so a config digests as it did before its route had it. The seeds
and the split name a job within it (`Job.label`); the target, the policy and the honest config are only the grade's,
and a new `training_slowdown` VERSION leaves it alone (the record says which VERSION wrote it).
"""
from __future__ import annotations

import hashlib
import importlib
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from . import calculator, starts
from .data import SPLITS
from .screen import screen

ROUTES = Path(__file__).with_name("routes")
#: the policies stated relative to the start, in `training_slowdown.policy` (`CONSOLE`)
RULES = ("console",)
LABEL_KEY = "training-slowdown"

SCHEMA = {"start": str, "route": str, "target": dict, "budget": dict, "seeds": dict, "policy": str, "honest": str,
          "windows": str, "dials": dict}
OPTIONAL = ("honest", "windows", "dials")
#: what a config's `windows` may ask for; a config without the key is served every window
WINDOWS = ("fresh",)
TABLES = {"target": {"gain": (int, float)}, "budget": {"steps": int, "batch": int, "minutes": (int, float)},
          "seeds": {s: list for s in SPLITS}}
REQUIRED_IN = {"target": ("gain",), "budget": ("steps", "batch"), "seeds": ()}


def route(name: str) -> ModuleType:
    """The route module `name`, its source screened before it is imported (`screen.screen`)."""
    path = ROUTES / f"{name}.py"
    if not path.is_file() or name.startswith("_"):
        have = sorted(p.stem for p in ROUTES.glob("*.py") if not p.stem.startswith("_"))
        raise SystemExit(f"config: no route {name!r} in training_slowdown/routes (have {', '.join(have)})")
    if bad := screen(path.read_text()):
        raise SystemExit(f"{path}: refused, it names {bad} (training_slowdown/routes/__init__.py, the route contract)")
    return importlib.import_module(f"training_slowdown.routes.{name}")


def _is(value, kind) -> bool:
    kinds = kind if isinstance(kind, tuple) else (kind,)
    return isinstance(value, kinds) and not (isinstance(value, bool) and bool not in kinds)


def dials(mod: ModuleType, given: dict, where: str) -> dict:
    """The route's DIALS with `given` over them, each of its default's type (an int where a float is due is one), and
    at most one written of each group of the route's `EITHER`."""
    out = dict(mod.DIALS)
    for k, v in given.items():
        if k not in out:
            raise SystemExit(f"{where}: route {mod.__name__.rsplit('.', 1)[-1]} has no dial {k!r} "
                             f"(have {', '.join(out)})")
        want = (int, float) if isinstance(out[k], float) else type(out[k])
        if not _is(v, want):
            raise SystemExit(f"{where}: dial {k!r} is {type(v).__name__} {v!r}, the route's is "
                             f"{type(out[k]).__name__} ({out[k]!r})")
        out[k] = float(v) if isinstance(out[k], float) else v
    for group in getattr(mod, "EITHER", ()):
        if len(both := [k for k in group if k in given]) > 1:
            raise SystemExit(f"{where}: dials {' and '.join(both)} say one thing in two ways and route "
                             f"{mod.__name__.rsplit('.', 1)[-1]} reads one of them (its EITHER): write one")
    return out


@dataclass(frozen=True)
class Job:
    config: Config
    seed: int
    split: str

    @property
    def label(self) -> str:
        return f"{self.config.digest}/{self.seed}/{self.split}"


@dataclass(frozen=True)
class Config:
    path: Path
    text: str
    start: starts.Start
    route: str
    route_version: int
    dials: dict
    #: the route's `LATER` dials this config leaves at their default, which its digest leaves out
    undigested: tuple
    budget: dict
    seeds: dict
    target: dict
    policy: str
    honest: str
    windows: str = ""

    @property
    def name(self) -> str:
        return self.path.stem

    def run(self) -> dict:
        """What decides a job's run. `windows` is in it only where the config asks, so a config that doesn't keeps
        the digest it had before a config could."""
        return {"start": self.start.name, "theta": self.start.theta_file_sha256, "data": self.start.data,
                "route": self.route, "route_version": self.route_version,
                "dials": {k: v for k, v in self.dials.items() if k not in self.undigested}, "budget": self.budget,
                **({"windows": self.windows} if self.windows else {})}

    @property
    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.run(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def jobs(self) -> list[Job]:
        return [Job(self, seed, split) for split in SPLITS for seed in self.seeds.get(split, [])]

    def record(self) -> dict:
        """The config as a job's record states it."""
        return {"name": self.name, "digest": self.digest, "run": self.run(), "text": self.text}


def load(path: Path, reg: dict | None = None) -> Config:
    path = Path(path)
    where = f"config {path}"
    if not path.is_file():
        raise SystemExit(f"{where}: no such file")
    text = path.read_text()
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise SystemExit(f"{where}: not TOML ({e})") from None
    for k, v in raw.items():
        if k not in SCHEMA:
            raise SystemExit(f"{where}: no key {k!r} in the schema (have {', '.join(SCHEMA)})")
        if not _is(v, SCHEMA[k]):
            raise SystemExit(f"{where}: {k} is {type(v).__name__}, not {SCHEMA[k].__name__}")
    if missing := [k for k in SCHEMA if k not in raw and k not in OPTIONAL]:
        raise SystemExit(f"{where}: missing {', '.join(missing)}")
    for t, fields in TABLES.items():
        for k, v in raw[t].items():
            if k not in fields:
                raise SystemExit(f"{where}: no key {k!r} in {t} (have {', '.join(fields)})")
            if not _is(v, fields[k]):
                raise SystemExit(f"{where}: {t}.{k} is {type(v).__name__} {v!r}")
        if missing := [k for k in REQUIRED_IN[t] if k not in raw[t]]:
            raise SystemExit(f"{where}: {t} has no {', '.join(missing)}")
    if raw["target"]["gain"] <= 1:
        raise SystemExit(f"{where}: target.gain is {raw['target']['gain']}, a gain over the start's own compute (> 1)")
    if raw["budget"]["steps"] <= 0 or raw["budget"]["batch"] <= 0 or raw["budget"].get("minutes", 1) <= 0:
        raise SystemExit(f"{where}: budget {raw['budget']} has a step count, batch or minutes that isn't positive")
    seeds = raw["seeds"]
    for split, ss in seeds.items():
        if not all(_is(s, int) and s >= 0 for s in ss) or len(set(ss)) != len(ss):
            raise SystemExit(f"{where}: seeds.{split} is {ss}, not distinct non-negative integers")
    if not any(seeds.values()):
        raise SystemExit(f"{where}: no seed")
    if raw.get("windows", WINDOWS[0]) not in WINDOWS:
        raise SystemExit(f"{where}: windows is {raw['windows']!r}, not {' or '.join(map(repr, WINDOWS))} (the windows "
                         f"the start's own training did not read); drop the key for every window of the start's data")
    if raw["policy"] not in (names := [*RULES, *calculator.policies()]):
        raise SystemExit(f"{where}: no policy {raw['policy']!r} (have {', '.join(names)}: policy.py's rule, or a "
                         f"[policy.*] table of verity/protocols/inference_only/rho/data)")
    start = starts.start(raw["start"], reg)
    mod = route(raw["route"])
    given = dials(mod, raw.get("dials", {}), where)
    try:
        mod.plan(given)
    except ValueError as e:
        raise SystemExit(f"{where}: {e}") from None
    return Config(path=path, text=text, start=start, route=raw["route"], route_version=int(mod.VERSION),
                  dials=given, undigested=tuple(k for k in getattr(mod, "LATER", ()) if given[k] == mod.DIALS[k]),
                  budget=dict(raw["budget"]), seeds=dict(seeds),
                  target=dict(raw["target"]), policy=raw["policy"], honest=raw.get("honest", ""),
                  windows=raw.get("windows", ""))


def start_of(path: Path, reg: dict | None = None) -> starts.Start:
    """The start a config names, read without its route: all a process that runs no route's code may read of it."""
    try:
        return starts.start(tomllib.loads(Path(path).read_text())["start"], reg)
    except (OSError, tomllib.TOMLDecodeError, KeyError) as e:
        raise SystemExit(f"config {path}: no start to read ({type(e).__name__}: {e})") from None


def honest(cfg: Config, reg: dict | None = None) -> Config | None:
    """The config's honest config, `<honest>.toml` beside it, refused at another start or gain."""
    if not cfg.honest:
        return None
    h = load(cfg.path.with_name(f"{cfg.honest}.toml"), reg)
    if h.path.resolve() == cfg.path.resolve():
        raise SystemExit(f"config {cfg.path}: honest names the config itself")
    if (h.start.name, h.target) != (cfg.start.name, cfg.target):
        raise SystemExit(f"config {cfg.path}: honest config {h.name} is at start {h.start.name}, target {h.target}, "
                         f"not {cfg.start.name}, {cfg.target}")
    return h
