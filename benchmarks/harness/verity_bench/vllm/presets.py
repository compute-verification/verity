"""Strict benchmark specs and deterministic request sweeps for vLLM."""

from __future__ import annotations

import re
import tomllib
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path, PurePosixPath
from typing import Any
from collections.abc import Mapping

from verity_config import Admission, ConfigError, gpus, load, sha256

from verity_bench import profiles

_PRODUCTION = profiles.load()
#: the vLLM commit every spec runs: the serving profiles' (`benchmarks/production.toml`'s ``vllm``)
VLLM_PIN = _PRODUCTION.vllm
#: each model the serving profiles pin, its repository to its revision (`production.toml`'s ``models``): a spec naming
#: one of them and no revision runs that revision
REVISIONS = {m.repository: m.revision for m in _PRODUCTION.models.values()}
SPEC_ROOT = Path(__file__).resolve().parents[2] / "specs"
SOURCE_ROOT = Path(__file__).resolve().parents[4]
#: A spec's path from the source tree's root while the specs had a directory of their own, and that directory. A run
#: recorded then names its spec by that path, and so does a command written then: each is the spec of that name in
#: `SPEC_ROOT` (`former_name`).
FORMER_SPEC = "benchmarks/specs/{name}.toml"
FORMER_SPECS = PurePosixPath(FORMER_SPEC).parent.as_posix()
GPUS = gpus()
#: Each spec id a run was recorded under before its spec named no card, as the spec and card it is now: read for a stored
#: run's record alone (`former_spec`); no launch takes one
FORMER_IDS = Path(__file__).resolve().parents[1] / "data" / "former_spec_ids.toml"
ENV_NAME = re.compile(r"[A-Z_][A-Z0-9_]*")
# What the dispatcher, the queue or the launcher sets: a spec naming one would be overridden or would override them.
RESERVED_ENV = re.compile(r"VERITY_BENCH_\w*|RESEARCH_\w*|GPU_LEASE_\w*|CUDA_VISIBLE_DEVICES|CUDA_DEVICE_ORDER|PYTHONPATH|PATH")
#: Every point's sampling, as vLLM `SamplingParams` fields.
GREEDY: Mapping[str, float] = {"temperature": 0.0}


@dataclass(frozen=True)
class SpecPreset:
    """One reproducible model, precision and engine configuration, as a spec's file writes it: no card."""

    id: str
    model: str
    precision: str
    gpu_count: int
    revision: str | None = None
    case: str | None = None
    engine_settings: Mapping[str, Any] = field(default_factory=dict)
    vllm_pin: str = VLLM_PIN


@dataclass(frozen=True)
class Preset(SpecPreset):
    """A spec's preset on the card a run is on (`gpu_id`, the card's `short`, `resolve_spec`)."""

    gpu_id: str = field(kw_only=True)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


#: Engine precisions an arm may name (`engine.engine_args`)
PRECISIONS = ("bfloat16", "fp8")
#: The keys a preset's `engine_settings` may set: those `engine.engine_args` reads, and the two `served` sets from the
#: spec's `eager` (``cuda_graphs``, ``enforce_eager``), which a spec's own value doesn't change
ENGINE_SETTINGS = (
    "max_model_len", "max_num_seqs", "max_num_batched_tokens", "resident_gpu_memory_utilization",
    "block_gpu_memory_utilization", "resident_kv_cache_memory_bytes", "block_kv_cache_memory_bytes", "capture_sizes",
    "serving_defaults", "cuda_graphs", "enforce_eager", "batch_invariant",
)
#: A protocol arm's `control` that gives it none: its tokens are checked by no other arm's, and the table names it
NO_CONTROL = "none"
#: The executions a served variant of a built-in arm (`BUILTIN@LABEL`, `arms.served_variant`) builds its engine in: vLLM's
#: serving defaults, eager or compiled (`verity_vllm.engine.served.EXECUTIONS`)
SERVED_EXECUTIONS = ("eager", "compiled")


@dataclass(frozen=True)
class ArmSettings:
    """A protocol arm's own engine (`[arm_settings."pouw:SCHEME"]`): the precision its engine loads, its control (an arm
    the spec names, or `NO_CONTROL`; default stock FP8), and the protocols config handed to `verity_vllm.LLM`
    (``{protocol: {...}}``), where a string value ``"$NAME"`` is the point's environment variable NAME, which the spec's
    setup exports. A served variant (`[arm_settings."stock-fp8@compiled"]`) takes only its control (default its
    built-in). All of it is in the point's identity."""

    precision: str | None = None
    control: str | None = None
    protocols_config: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class CardSettings:
    """What a spec's run takes on one of its cards in place of the spec's own (`[card_settings.SHORT]`): only where the
    card needs it."""

    env: Mapping[str, str] = field(default_factory=dict)
    setup: str | None = None
    admission: Admission | None = None


@dataclass(frozen=True)
class BenchmarkSpec:
    """One runnable benchmark configuration before its named sweep and its card are resolved: the cards it runs on
    (``cards``, by their `short`; none where any card will do), what differs on one (``card_settings``), and why it
    refuses a card it doesn't list (``refused``, each said in that card's refusal)."""

    schema: str
    backend: str
    id: str
    arms: tuple[str, ...]
    sweep: str
    reps: int
    eager: bool
    preset: SpecPreset
    cards: tuple[str, ...] = ()
    card_settings: Mapping[str, CardSettings] = field(default_factory=dict)
    refused: Mapping[str, str] = field(default_factory=dict)
    arm_modules: tuple[str, ...] = ()
    arm_settings: Mapping[str, ArmSettings] = field(default_factory=dict)
    env: Mapping[str, str] = field(default_factory=dict)
    setup: str | None = None
    admission: Admission = Admission()
    warmups: int = 1
    max_warmups: int | None = None


@dataclass(frozen=True)
class Sweep:
    schema: str
    id: str
    batches: tuple[int, ...]
    prompt_tokens: tuple[int, ...]
    traffic: tuple[str, ...]
    output_tokens: Mapping[str, int]
    # greedy only; a resolved spec's identity hashes it
    sampling: tuple[str, ...] = ("greedy",)


@dataclass(frozen=True)
class ResolvedSpec:
    schema: str
    backend: str
    id: str
    arms: tuple[str, ...]
    sweep: Sweep
    reps: int
    eager: bool
    preset: Preset
    arm_modules: tuple[str, ...] = ()
    arm_settings: Mapping[str, ArmSettings] = field(default_factory=dict)
    env: Mapping[str, str] = field(default_factory=dict)
    setup: str | None = None
    admission: Admission = Admission()
    warmups: int = 1
    max_warmups: int | None = None

    @property
    def config_sha256(self) -> str:
        return sha256(self, identity=True)


def former_name(spec: str | Path) -> str | None:
    """The name of the spec that a path under the specs' former directory (`FORMER_SPECS`) names; None for a name or any
    other path."""

    path = PurePosixPath(Path(spec).as_posix())
    under = path.parent.as_posix()
    return path.name.removesuffix(".toml") if under == FORMER_SPECS or under.endswith("/" + FORMER_SPECS) else None


def split_card(spec: str | Path) -> tuple[str, str | None]:
    """A spec argument as (the spec, the card it names): ``SPEC@CARD`` names the card by its `short`, and so does a path
    recorded for one (``.../SPEC@CARD.toml``, `jobs._relative_spec`'s); else no card."""

    text = Path(spec).as_posix() if isinstance(spec, Path) else spec
    head, _, name = text.rpartition("/")
    stem, toml, _ = name.partition(".toml")
    if "@" not in stem:
        return text, None
    stem, card = stem.split("@", 1)
    return (f"{head}/" if head else "") + stem + toml, card


def former_spec(spec_id: str) -> tuple[str, str] | None:
    """The spec and card a former spec id is now (`FORMER_IDS`), for reading a run recorded under it; None for any
    other id."""

    return former_specs().get(spec_id)


def former_specs() -> dict[str, tuple[str, str]]:
    """Every former spec id (`FORMER_IDS`) as the spec and card it is now."""

    return {old: (entry["spec"], entry["card"]) for old, entry in tomllib.loads(FORMER_IDS.read_text())["former"].items()}


def declared_schemes() -> tuple[str, ...]:
    """The declared schemes' names (`verity_vllm.messages.declarations`), each an arm id that attaches it by name."""
    from verity_vllm import messages

    return tuple(messages.declarations.load())


def former_arm(arm_id: str) -> str | None:
    """The arm id a former arm id is now (`FORMER_IDS`' ``[former_arms]``), for reading a run recorded under it; None
    for any other id. No launch takes a former arm id."""

    return tomllib.loads(FORMER_IDS.read_text()).get("former_arms", {}).get(arm_id)


def retired_spec(spec_id: str) -> str | None:
    """Why a spec deleted with no successor refuses a launch (`FORMER_IDS`' ``[retired]``); None for any other id. Its
    stored runs read under the id itself."""

    return tomllib.loads(FORMER_IDS.read_text()).get("retired", {}).get(spec_id)


def current_spec(spec_id: str) -> str:
    """The name a stored run's spec id has now: for a former id (`FORMER_IDS`), its spec, ``@CARD`` where that spec runs on
    more than its card; else the id itself."""

    now = former_spec(spec_id)
    if now is None:
        return spec_id
    doc = tomllib.loads(_spec_path(now[0]).read_text())
    return now[0] if doc.get("cards", [doc.get("card")]) == [now[1]] else f"{now[0]}@{now[1]}"


def spec_file(spec: str | Path) -> Path | None:
    """The spec file ``spec`` names, by path or id (one under `FORMER_SPECS` by its name); None where none is there: a
    former, retired or unknown id."""

    path = Path(spec)
    if path.is_file():
        return path
    named = SPEC_ROOT / f"{former_name(spec) or spec}.toml"
    return named if named.is_file() else None


def _spec_path(spec: str | Path) -> Path:
    if (path := spec_file(spec)) is not None:
        return path
    name = former_name(spec) or str(spec)
    if now := former_spec(name):
        raise ConfigError(f"spec {name!r} is {now[0]!r} on the {now[1]}, `{now[0]}@{now[1]}`: its former id names a "
                          "stored run's spec, and no run is launched under it")
    if why := retired_spec(name):
        raise ConfigError(f"spec {name!r} is refused: {why}")
    raise ConfigError(f"no benchmark spec {spec!r}")


def _card_shorts() -> set[str]:
    from verity_bench import kit

    return {c["short"] for c in kit._gpu_cards(kit.cards_directory()) if c.get("short")}


def load_spec(spec: str | Path) -> BenchmarkSpec:
    """The spec file ``spec`` names, by id or path (a card it names set aside, `split_card`), as written: on no card."""

    path = _spec_path(split_card(spec)[0])
    doc = load(path, BenchmarkSpec)
    if doc.schema != "verity-bench/vllm-spec/v0" or doc.backend != "vllm":
        raise ConfigError(f"{path}: expected a verity-bench/vllm-spec/v0 document")
    if doc.id != doc.preset.id:
        raise ConfigError(f"{path}: spec id {doc.id!r} differs from preset id {doc.preset.id!r}")
    if unknown := sorted(set(doc.cards) - (set(GPUS) & _card_shorts())):
        raise ConfigError(f"{path}: cards {unknown} are no card's `short` (calibration/hardware/cards) with a GPU id")
    for card in doc.cards:
        if doc.preset.gpu_count < 1 or doc.preset.gpu_count > GPUS[card].count_per_node:
            raise ConfigError(f"{path}: gpu_count is outside {card!r}'s per-node capacity")
    if stray := sorted(set(doc.card_settings) - set(doc.cards)):
        raise ConfigError(f"{path}: card_settings {stray} name no card of the spec's cards {list(doc.cards)}")
    if stray := sorted(set(doc.refused) & set(doc.cards) | set(doc.refused) - _card_shorts()):
        raise ConfigError(f"{path}: refused {stray}: each names a card's `short` the spec doesn't list in cards")
    unknown = sorted(set(doc.preset.engine_settings) - set(ENGINE_SETTINGS))
    if unknown:
        raise ConfigError(f"{path}: unknown engine_settings {unknown}; known: {list(ENGINE_SETTINGS)}")
    if doc.reps < 1 or doc.warmups < 1 or not doc.arms:
        raise ConfigError(f"{path}: arms must be nonempty and reps and warmups must be positive")
    if doc.max_warmups is not None and doc.max_warmups < doc.warmups:
        raise ConfigError(f"{path}: max_warmups must be at least warmups")
    envs = [*doc.env, *(name for on in doc.card_settings.values() for name in on.env)]
    bad = sorted(name for name in envs if not ENV_NAME.fullmatch(name) or RESERVED_ENV.fullmatch(name))
    if bad:
        raise ConfigError(f"{path}: env names must be upper-case and not the dispatcher's, queue's or launcher's: {bad}")
    for given in (doc.setup, *(on.setup for on in doc.card_settings.values())):
        if given is None:
            continue
        setup = PurePosixPath(given)
        if setup.is_absolute() or ".." in setup.parts or not (SOURCE_ROOT / setup).is_file():
            raise ConfigError(f"{path}: setup {given!r} must name a script in the source tree, relative to its root")
    for name, arm in doc.arm_settings.items():
        variant = name.partition("@")[2] in SERVED_EXECUTIONS
        if name not in doc.arms or not (":" in name or variant or name in declared_schemes()):
            raise ConfigError(f"{path}: arm_settings {name!r} must name a protocol arm (protocol:scheme or a declared "
                              "scheme's name) or a served "
                              f"variant (BUILTIN@{'|'.join(SERVED_EXECUTIONS)}) in arms")
        if variant and (arm.precision is not None or arm.protocols_config):
            raise ConfigError(f"{path}: arm_settings {name!r}: a served variant takes only its control")
        if arm.precision is not None and arm.precision not in PRECISIONS:
            raise ConfigError(f"{path}: arm_settings {name!r}: precision is one of {', '.join(PRECISIONS)}")
        if arm.control is not None and arm.control != NO_CONTROL and (arm.control not in doc.arms or arm.control == name):
            raise ConfigError(f"{path}: arm_settings {name!r}: control must be another arm in arms, or {NO_CONTROL!r}")
    if doc.preset.revision is None and doc.preset.model in REVISIONS:
        doc = replace(doc, preset=replace(doc.preset, revision=REVISIONS[doc.preset.model]))
    return doc


def load_sweep(name: str) -> Sweep:
    path = SPEC_ROOT / f"{name}.toml"
    doc = load(path, Sweep)
    if doc.schema != "verity-bench/vllm-sweep/v0" or doc.id != name:
        raise ConfigError(f"{path}: expected sweep {name!r}")
    if not doc.batches or not doc.prompt_tokens or not doc.traffic:
        raise ConfigError(f"{path}: sweep axes must be nonempty")
    missing = sorted(set(doc.traffic) - set(doc.output_tokens))
    if missing:
        raise ConfigError(f"{path}: output_tokens has no value for {missing}")
    if tuple(doc.sampling) != ("greedy",):
        raise ConfigError(f"{path}: sampling {list(doc.sampling)}: every point samples greedily; drop the key")
    return doc


def run_card(doc: BenchmarkSpec, card: str | None) -> str:
    """The card a run of ``doc`` is on: the one named, else the spec's one card, else the card the driver reports here
    (`verity_bench.kit.card_short`); refused, naming the spec's cards, where it is none of them."""

    if card is None and len(doc.cards) == 1:
        card = doc.cards[0]
    if card is None:
        from verity_bench import kit

        found = kit.host_stamp()["card"]
        if found == "none":
            raise ConfigError(f"spec {doc.id!r} runs on {' or '.join(doc.cards) or 'any card'}, and no card is here: name "
                              f"one, `{doc.id}@CARD`")
        card = kit.card_short(found)
    if card in doc.refused:
        raise ConfigError(f"spec {doc.id!r} refuses the {card}: {doc.refused[card]}")
    if doc.cards and card not in doc.cards:
        raise ConfigError(f"spec {doc.id!r} runs on {', '.join(doc.cards)}, not the {card}")
    if card not in GPUS:
        raise ConfigError(f"spec {doc.id!r}: the {card} is no GPU of verity_config's registry ({', '.join(GPUS)})")
    if doc.preset.gpu_count > GPUS[card].count_per_node:
        raise ConfigError(f"spec {doc.id!r}: gpu_count is outside {card!r}'s per-node capacity")
    return card


def resolve_spec(spec: str | Path) -> ResolvedSpec:
    """The spec ``spec`` names (an id or a path, ``@CARD`` naming its card, `split_card`) on its run's card (`run_card`),
    with what the spec sets for that card (``card_settings``) in place of its own."""

    doc = load_spec(spec)
    card = run_card(doc, split_card(spec)[1])
    on = doc.card_settings.get(card, CardSettings())
    preset = Preset(**{f.name: getattr(doc.preset, f.name) for f in fields(SpecPreset)}, gpu_id=card)
    return ResolvedSpec(
        schema=doc.schema,
        backend=doc.backend,
        id=doc.id,
        arms=doc.arms,
        sweep=load_sweep(doc.sweep),
        reps=doc.reps,
        eager=doc.eager,
        preset=preset,
        arm_modules=doc.arm_modules,
        arm_settings=doc.arm_settings,
        env={**doc.env, **on.env},
        setup=on.setup or doc.setup,
        admission=on.admission or doc.admission,
        warmups=doc.warmups,
        max_warmups=doc.max_warmups,
    )


def served(preset: Preset, eager: bool) -> tuple[Preset, bool]:
    """The preset as its engines are built, and whether they run eagerly: when asked, and with CUDA graphs exactly when
    not."""

    eager = bool(eager)
    settings = {**preset.engine_settings, "cuda_graphs": not eager, "enforce_eager": eager}
    return replace(preset, engine_settings=settings), eager


@dataclass(frozen=True)
class RequestShape:
    prompt_tokens: int
    output_tokens: int
    arrive_step: int

    def to_json(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class SweepPoint:
    """One batch, prompt envelope and traffic mix, sampled greedily."""

    batch: int
    prompt_tokens: int
    traffic: str
    output_tokens: int

    @property
    def id(self) -> str:
        return f"b{self.batch}-p{self.prompt_tokens}-{self.traffic}"

    @property
    def headline(self) -> bool:
        return self.batch == 32 and self.traffic == "mixed"

    def requests(self) -> tuple[RequestShape, ...]:
        if self.traffic == "decode-heavy":
            return tuple(RequestShape(self.prompt_tokens, self.output_tokens, 0) for _ in range(self.batch))
        if self.traffic == "prefill-heavy":
            return tuple(RequestShape(self.prompt_tokens, self.output_tokens, 0) for _ in range(self.batch))
        if self.traffic != "mixed":
            raise ValueError(f"unknown traffic shape {self.traffic!r}")
        wave = max(1, self.batch // 4)
        return tuple(
            RequestShape(
                self.prompt_tokens if i % 2 == 0 else max(16, self.prompt_tokens // 4),
                self.output_tokens if i % 3 else max(8, self.output_tokens // 2),
                i // wave,
            )
            for i in range(self.batch)
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "batch": self.batch,
            "prompt_tokens": self.prompt_tokens,
            "traffic": self.traffic,
            "output_tokens": self.output_tokens,
            "sampling": "greedy",  # a point's identity hashes this JSON
            "staggered_arrivals": any(r.arrive_step for r in self.requests()),
        }


def expand_sweep(name: str) -> tuple[SweepPoint, ...]:
    """Expand a strict sweep document into its fixed Cartesian product."""

    sweep = load_sweep(name)
    return tuple(
        SweepPoint(batch, prompt, traffic, sweep.output_tokens[traffic])
        for batch in sweep.batches
        for prompt in sweep.prompt_tokens
        for traffic in sweep.traffic
    )
