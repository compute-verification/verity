"""The serving profiles of `benchmarks/production.toml`, read strictly into frozen objects (`load`), with each profile's
digest (`digest`) and its dataset, cut and load (`dataset`, `cut`, `load_point`), and a sessions profile's ladder,
the users `--find-load` serves stock at (`rungs`). It builds no command line: a
profile's server and client are serve_tax's own (`benchmarks/serve_tax`, `--profile`). A model's launch is the same on each of its `cards` (the card files'
``short``, `verity_bench.kit.card_short`), and a profile's load is the model's on one card."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from types import MappingProxyType

from verity_config import ConfigError, load as load_toml, sha256

PATH = Path(__file__).resolve().parents[2] / "production.toml"
SCHEMA = "verity-bench/production/v1"
NOT_FOUND = "not found yet"
PROFILES = ("coding-agent", "chat", "fixed-size")
GOODPUT = {"first_token": ("ttft", "first_token_s"), "token": ("tpot", "token_s")}


@dataclass(frozen=True)
class Dataset:
    repository: str
    revision: str
    file: str
    bytes: int
    sha512: str
    license: str


@dataclass(frozen=True)
class Cut:
    dataset: str
    turns: int
    cut: str


@dataclass(frozen=True)
class Ladder:
    """Users ``start``, then each rung ``factor`` times the last, up to ``most``; the load is the fewest users past
    which stock's output tokens a second, at the next rung, rise by less than ``gain`` (a fraction)."""
    start: int
    factor: int
    most: int
    gain: float


@dataclass(frozen=True)
class Profile:
    version: int
    what: str
    dataset_name: str
    within: tuple[str, ...]
    prompts: int | None = None
    input: int | None = None
    output: int | None = None
    block: int | None = None
    cuts: Mapping[str, Cut] = field(default_factory=dict)
    datasets: Mapping[str, Dataset] = field(default_factory=dict)
    dataset: Dataset | None = None
    users: Mapping[str, Mapping[str, int | str]] = field(default_factory=dict)
    in_flight: Mapping[str, Mapping[str, int | str]] = field(default_factory=dict)
    ladder: Ladder | None = None


@dataclass(frozen=True)
class Model:
    repository: str
    revision: str
    served: str
    cards: tuple[str, ...]
    gpus: int
    context: int
    quick_check: bool
    cap_minutes: int
    serve: tuple[str, ...]
    max_model_len: int | None = None


@dataclass(frozen=True)
class Client:
    command: tuple[str, ...]
    backend: str
    endpoint: str
    stream: bool


@dataclass(frozen=True)
class Server:
    command: tuple[str, ...]
    fixed: tuple[str, ...]
    prefix_caching: bool


@dataclass(frozen=True)
class Sampling:
    defaults: str
    stop: str


@dataclass(frozen=True)
class Bounds:
    first_token_s: float
    token_s: float


@dataclass(frozen=True)
class Difference:
    command: str
    profile: str
    why: str
    models: tuple[str, ...] = ()
    profiles: tuple[str, ...] = ()


@dataclass(frozen=True)
class Production:
    schema: str
    vllm: str
    client: Client
    server: Server
    sampling: Sampling
    bounds: Bounds
    models: Mapping[str, Model]
    profiles: Mapping[str, Profile]
    differences: tuple[Difference, ...]


SHAPES = {
    "timed_trace": {"block", "cuts", "datasets", "users", "ladder"},
    "sharegpt": {"prompts", "dataset", "in_flight"},
    "random": {"prompts", "input", "output", "in_flight"},
}
OPTIONAL = {"prompts", "input", "output", "block", "cuts", "datasets", "dataset", "users", "in_flight", "ladder"}


def _check(doc: Production, path: Path) -> Production:
    def refuse(message: str) -> ConfigError:
        return ConfigError(f"{path}: {message}")

    if doc.schema != SCHEMA:
        raise refuse(f"schema is {doc.schema!r}; this reader reads {SCHEMA!r}")
    if tuple(doc.profiles) != PROFILES:
        raise refuse(f"profiles are {list(doc.profiles)}; there are three, in order: {list(PROFILES)}")
    if (doc.client.backend, doc.client.endpoint, doc.client.stream) != ("openai", "/v1/completions", True):
        raise refuse("the client is the openai backend on /v1/completions, streamed: the only client vllm bench serve "
                     "streams token ids through")
    if not doc.server.prefix_caching:
        raise refuse("prefix caching is vLLM's default and the profiles keep it on; there is no flag to write for off")
    for name, model in doc.models.items():
        if model.max_model_len not in (None, model.context):
            raise refuse(f"models.{name}: max_model_len {model.max_model_len} is not its context {model.context}")
        if "--max-model-len" in model.serve or "--seed" in model.serve:
            raise refuse(f"models.{name}: serve repeats a flag the file sets elsewhere (max_model_len, server.fixed)")
    for name, profile in doc.profiles.items():
        shape = SHAPES.get(profile.dataset_name)
        if shape is None:
            raise refuse(f"profiles.{name}: dataset_name {profile.dataset_name!r} is none of {sorted(SHAPES)}")
        given = {key for key in OPTIONAL if getattr(profile, key) not in (None, {})}
        if given != shape:
            raise refuse(f"profiles.{name}: a {profile.dataset_name} profile has exactly {sorted(shape)}; "
                         f"it has {sorted(given)}")
        if unknown := sorted(set(profile.within) - set(GOODPUT)):
            raise refuse(f"profiles.{name}: within names {unknown}; the bounds are {sorted(GOODPUT)}")
        loads = profile.users or profile.in_flight
        if set(loads) != set(doc.models):
            raise refuse(f"profiles.{name}: its load names models {sorted(loads)}, not {sorted(doc.models)}")
        for model, by_card in loads.items():
            if set(by_card) != set(doc.models[model].cards):
                raise refuse(f"profiles.{name}: its load for {model} names cards {sorted(by_card)}, not its cards "
                             f"{sorted(doc.models[model].cards)}")
            for card, count in by_card.items():
                if count != NOT_FOUND and not (isinstance(count, int) and count > 0):
                    raise refuse(f"profiles.{name}: load for {model}.{card} is {count!r}: a positive count, or "
                                 f"{NOT_FOUND!r}")
        if (ladder := profile.ladder) and not (1 <= ladder.start <= ladder.most and ladder.factor >= 2
                                               and 0 < ladder.gain < 1):
            raise refuse(f"profiles.{name}.ladder: start of at least 1 and at most most, factor of at least 2, and "
                         f"gain a fraction between 0 and 1; it is {ladder}")
        for context, cut in profile.cuts.items():
            if not context.isdigit() or cut.dataset not in profile.datasets:
                raise refuse(f"profiles.{name}.cuts.{context}: a context in tokens, on one of {sorted(profile.datasets)}")
        for model in doc.models.values():
            if profile.cuts and str(model.context) not in profile.cuts:
                raise refuse(f"profiles.{name}: no cut at {model.served}'s context {model.context}")
    for row in doc.differences:
        if unknown := sorted(set(row.models) - set(doc.models)) + sorted(set(row.profiles) - set(doc.profiles)):
            raise refuse(f"differences: {row.command!r} names {unknown}, which the file does not hold")
    return doc


def _freeze(obj):
    if is_dataclass(obj):
        return replace(obj, **{f.name: _freeze(getattr(obj, f.name)) for f in fields(obj)})
    if isinstance(obj, Mapping):
        return MappingProxyType({key: _freeze(value) for key, value in obj.items()})
    return obj


def load(path: str | Path = PATH) -> Production:
    """The file, every key declared and every profile of its shape; anything else is refused."""
    path = Path(path)
    return _check(_freeze(load_toml(path, Production)), path)


def digest(doc: Production, profile: str) -> str:
    """What a run under ``profile`` depends on, its version left out: the profile and every shared table."""
    body = {key: getattr(doc, key) for key in ("schema", "vllm", "client", "server", "sampling", "bounds", "models")}
    # a profile with no ladder digests as it did before profiles had one, so its pin holds
    own = {key: value for key, value in vars(doc.profiles[profile]).items()
           if key != "version" and (key != "ladder" or value is not None)}
    rows = [row for row in doc.differences if not row.profiles or profile in row.profiles]
    return sha256({**body, "profile": own, "differences": rows})


def cut(doc: Production, profile: str, model: str) -> Cut:
    """The coding-agent cut at ``model``'s context."""
    return doc.profiles[profile].cuts[str(doc.models[model].context)]


def dataset(doc: Production, profile: str, model: str) -> Dataset:
    prof = doc.profiles[profile]
    return prof.datasets[cut(doc, profile, model).dataset] if prof.cuts else prof.dataset


def load_point(doc: Production, profile: str, model: str, card: str) -> int:
    """Users of a session profile, requests in flight of the others, for ``model`` on ``card``; refused on a card the
    model has no load for, and while not found yet."""
    prof = doc.profiles[profile]
    by_card = (prof.users or prof.in_flight)[model]
    if card not in by_card:
        raise ConfigError(f"{profile} on {model}: no load on card {card}; its cards are {', '.join(by_card)}")
    count = by_card[card]
    if count == NOT_FOUND:
        raise ConfigError(f"{profile} on {model}.{card}: the load is {NOT_FOUND}")
    return count


def rungs(doc: Production, profile: str) -> tuple[int, ...]:
    """The users a sessions profile's `--find-load` serves stock at, its ladder's rungs in order."""
    ladder = doc.profiles[profile].ladder
    out = [ladder.start]
    while out[-1] * ladder.factor <= ladder.most:
        out.append(out[-1] * ladder.factor)
    return tuple(out)
