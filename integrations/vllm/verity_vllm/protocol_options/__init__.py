"""The protocol options of a vLLM run: PoUW, POUS and sampled proofs, each enabled on its own, any of them in one run.

A run enables a set of ``name:scheme`` entries (`config.parse_protocols`), off by default.  `verity_vllm.LLM` takes it as
``protocols=``; a row takes it from ``PROTOCOLS`` (``verity-vllm row --protocols``) or its workload target, and every stage
reads it as the target's ``protocols`` (`TargetProfile.protocols`), so the Build, the Commit and the Check read one copy.
A row refuses a set it cannot run before any stage starts (`at_row`); the Commit checks it again before its engine exists
(`commit_guard`).  Each protocol has one adapter module here (`interface`), imported only when it is enabled.  The
protocol logic is its package's (`verity.core.protocols.pouw`, `verity.core.protocols.pous`, `verity.core.protocols.sampled_proofs`), never this package's.

**Order.**  The nesting at each site is stated in `SITES`.  Adapters install in `config.PROTOCOLS` order, innermost
first, and uninstall in reverse:

1. ``pouw`` executes at each linear layer's ``quant_method.apply``: it replaces the GEMM, reading the weight from the
   layer it is called with;
2. ``pous`` wraps the same ``apply`` around it: it decodes the weight from its encoded store and calls the inner
   ``apply`` with a layer holding the decoded weight, so the GEMMs read what POUS stores; its responder starts last;
3. ``sampled-proofs`` is outermost: the Commit's committer and taps commit the values the model produced.

**Phases.**  First every enabled option commits to the plaintext weights of record (``commit_weights``), with the model
untouched and no beacon read; then each installs.  A Commit installs its composition after its committers registered the
weights (`at_commit`), and after its warm-ups, so every ``apply`` an executor sees is serving.

**Refused** (`refusals`): an unknown protocol; a scheme its package does not register, or an adapter not on this tree;
a protocols config naming a protocol the run does not enable; a set at TP > 1 (each rank would install its own, which is
not wired); two executors at one site; beside sampled proofs, an executor whose protocol package has no circuit for its
scheme (`CIRCUITS`, `traced_as`: every PoUW scheme but ``ncp-v2``, and Pearl's noise-dependent output for good);
with ``pous`` on, an option that keeps a weight-derived copy resident across forwards (``keeps_weight_copy``); and
whatever an adapter refuses for its context (sampled proofs outside the Commit stage, for one).

**Weights of record.**  An option that releases weights from the model (POUS) returns their plaintext from
``Installed.weights()``; the Commit's replay reads and re-registers the live weights through `weights_view` /
`register_weights`, which is the model itself when no option released any.

**The record.**  `Composition.record` is ``verity-vllm/protocol-composition/v0``: the stage, the profile digest and, per
protocol, its entry and what it commits to, with the SHA-256 of its canonical JSON.  It is a placeholder: each protocol's
verifier checks its own claims, and nothing checks the links between them yet.  A Commit's verdict carries the set and the
record's digest (`Composition.verdict`); it is of record like any other, because every option a Commit admits is one the
Program describes: POUS's decode is bit-identical and leaves the circuit untouched, and an executor is refused until its
protocol package has a circuit for its scheme (``traced_as``).

**Circuits.**  What sampled proofs traces an executor's computation as is its protocol package's (`CIRCUITS`: for PoUW,
`verity.core.protocols.pouw.circuit`, whose ``SCHEMES`` have a circuit and whose ``traced_as(scheme, m, k, n)`` is the circuit of one
executed call).  The selector admits an executor beside sampled proofs only for a scheme with a circuit, and `traced_as`
gives the Commit the circuit of each executed call.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

from verity_vllm import config
from verity_vllm.engine import hooks
from verity_vllm.protocol_options.interface import Context, Installed, Option, Target

__all__ = ["ADAPTERS", "RECORD_SCHEMA", "SITES", "Composition", "Context", "Installed", "Option", "Refused", "Target", "adapter",
           "WeightsView", "at_capture", "at_commit", "at_row", "capture_context", "commit_context", "commit_guard", "install", "into_verdict", "load_settings",
           "parse", "refusals", "register_weights", "undeclared", "weights_view", "CIRCUITS", "circuit_registry", "traced_as"]

#: protocol -> its adapter module in this package
ADAPTERS = {"pouw": "pouw", "pous": "pous", "sampled-proofs": "sampled_proofs"}
#: Each hook site and who is at it, outermost first; the last may execute there (``EXECUTES``), the others wrap.  At a
#: linear layer's ``apply`` POUS hands the decoded weight inward and PoUW replaces the GEMM.  Sampled proofs commits at the
#: module boundary, outside every site, and its replay evaluates Definitions, never an executor.  `config.PROTOCOLS`, the
#: install order, is innermost first at every site.
SITES = {"LinearBase.quant_method.apply": ("pous", "pouw")}
#: protocol -> the module of its protocol package that says what sampled proofs traces its executed computation as:
#: ``SCHEMES`` (the schemes with a circuit) and ``traced_as(scheme, m, k, n)`` (one executed call's circuit, None without one)
CIRCUITS = {"pouw": "verity.core.protocols.pouw.circuit"}
RECORD_SCHEMA = "verity-vllm/protocol-composition/v0"

parse = config.parse_protocols


class Refused(ValueError):
    """The run cannot enable this set; ``reasons`` lists every refusal."""

    def __init__(self, reasons: tuple[str, ...] | list[str]) -> None:
        self.reasons = tuple(reasons)
        super().__init__("protocol options refused: " + "; ".join(self.reasons))


def adapter(name: str) -> ModuleType:
    """The adapter module of protocol ``name``; `Refused` when the protocol is unknown or its adapter cannot be imported
    on this tree (not landed yet, or its protocol package missing)."""
    if name not in ADAPTERS:
        raise Refused([f"unknown protocol {name!r} (known: {', '.join(ADAPTERS)})"])
    module = f"{__name__}.{ADAPTERS[name]}"
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as e:
        why = "is not on this tree" if e.name == module else f"needs {e.name}, which is not installed"
        raise Refused([f"{name}: its adapter {module} {why}"]) from e


def load_settings(value: str | Path | Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """The protocols config, ``{name: {...}}``: a mapping as given, or read from a JSON file; empty for None."""
    if value is None:
        return {}
    doc = dict(value) if isinstance(value, Mapping) else json.loads(Path(value).read_text())
    bad = sorted(k for k, v in doc.items() if not isinstance(v, Mapping))
    if bad:
        raise ValueError(f"protocols config: the entries of {bad} are not objects")
    return {str(k): dict(v) for k, v in doc.items()}


def circuit_registry(name: str) -> ModuleType | None:
    """Protocol ``name``'s circuit registry (`CIRCUITS`), None when it declares none; `Refused` when it is not installed."""
    module = CIRCUITS.get(name)
    if module is None:
        return None
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as e:
        raise Refused([f"{name}: its circuit registry {module} is not installed ({e.name})"]) from e


def traced_as(entry: str, m: int, k: int, n: int) -> Any:
    """The circuit sampled proofs traces for one executed call of ``entry`` (``pouw:ncp-v2``) of shape m × k → n: its
    Definition, partition query, served outputs and draw layout.  None when the scheme has no circuit; a shape outside the
    circuit raises the protocol package's ``DomainError``."""
    name, _, scheme = entry.partition(":")
    reg = circuit_registry(name)
    return None if reg is None else reg.traced_as(scheme, m, k, n)


def _has_circuit(name: str, scheme: str, out: list[str]) -> bool:
    try:
        reg = circuit_registry(name)
    except Refused as r:
        out.extend(r.reasons)
        return False
    return reg is not None and scheme in reg.SCHEMES


def refusals(ctx: Context) -> tuple[str, ...]:
    """Every reason the run described by ``ctx`` cannot enable ``ctx.enabled`` (empty: it can)."""
    out: list[str] = []
    names = [e.partition(":")[0] for e in ctx.enabled]
    if ctx.enabled and ctx.world > 1:
        out.append(f"protocol options at TP {ctx.world}: each rank would install its own composition, which is not wired yet")
    stray = sorted(set(ctx.settings) - set(names))
    if stray:
        out.append(f"the protocols config names {stray}, which this run does not enable")
    executors: dict[str, str] = {}
    for entry, name in zip(ctx.enabled, names):
        scheme = entry.partition(":")[2]
        try:
            mod = adapter(name)
        except Refused as r:
            out.extend(r.reasons)
            continue
        known = tuple(mod.schemes())
        if scheme not in known:
            out.append(f"{entry}: {name} registers no scheme {scheme!r} (registered: {', '.join(known) or 'none'})")
            continue
        out.extend(f"{entry}: {r}" for r in mod.refusals(scheme, ctx))
        sites = tuple(getattr(mod, "EXECUTES", ()))
        for site in sites:
            if site in executors:
                out.append(f"{entry}: {site} already has an executor, {executors[site]}; a site has one")
            executors.setdefault(site, entry)
        if sites and ctx.scheme("sampled-proofs") is not None and not _has_circuit(name, scheme, out):
            out.append(f"{entry}: executes {', '.join(sites)}, and the Program has no Definition for what it computes there, so "
                       f"sampled proofs would trace it as the stock computation (refused until the option names one: traced_as)")
        if ctx.scheme("pous") is not None and name != "pous" and getattr(mod, "keeps_weight_copy", lambda s, c: False)(scheme, ctx):
            out.append(f"{entry}: keeps a weight-derived copy resident across forwards, so with pous on its GEMMs would not read "
                       f"what POUS stores (prepare its weight operands per forward from the weight it is handed)")
    return tuple(out)


class Composition:
    """The options one run installed, in install order.  `close` uninstalls them newest first; `record` and `verdict`
    say what they committed to and whether the run's verdict can be of record."""

    def __init__(self, ctx: Context) -> None:
        self.context = ctx
        self.installed: list[Installed] = []
        self.weight_commitments: dict[str, Any] = {}
        self.closed = False

    @property
    def entries(self) -> tuple[str, ...]:
        return tuple(i.name for i in self.installed)

    def get(self, name: str) -> Installed | None:
        """The installed option of protocol ``name``, None when it is not enabled."""
        return next((i for i in self.installed if i.name.partition(":")[0] == name), None)

    def bind(self, name: str, **commitment: Any) -> None:
        """Add to what protocol ``name`` commits to, once its commitment is known (the Commit's roots, for sampled proofs)."""
        opt = self.get(name)
        if not isinstance(opt, Option):
            raise ValueError(f"no bindable option {name!r} in {list(self.entries)}")
        opt.commits.update(commitment)

    def close(self) -> None:
        """Close every option, newest first.  Idempotent; an option whose close raises does not keep the older ones
        installed (the first error is raised once all are closed)."""
        if self.closed:
            return
        self.closed = True
        err: BaseException | None = None
        for opt in reversed(self.installed):
            try:
                opt.close()
            except BaseException as e:  # noqa: BLE001
                err = err or e
        if err is not None:
            raise err

    def weights(self) -> dict[str, Any]:
        """The plaintext of every tensor an installed option released, by qualname (`Installed.weights`)."""
        out: dict[str, Any] = {}
        for i in self.installed:
            got = getattr(i, "weights", None)
            got = got() if callable(got) else None
            for k, v in (got or {}).items():
                if k in out:
                    raise ValueError(f"{k!r} is released by two options ({i.name} and an earlier one)")
                out[k] = v
        return out

    def record(self, profile_digest: str | None = None) -> dict[str, Any]:
        body = {"schema": RECORD_SCHEMA, "stage": self.context.stage, "profile_digest": profile_digest,
                "protocols": [{"entry": i.name, "commitment": i.commitment(),
                               **({"weights": dict(self.weight_commitments[i.name])} if i.name in self.weight_commitments else {})}
                              for i in self.installed]}
        canon = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        return {**body, "digest": hashlib.sha256(canon).hexdigest()}

    def verdict(self, profile_digest: str | None = None) -> dict[str, Any]:
        """The block a Commit's verdict carries: the set and the record's digest."""
        return {"protocols": list(self.entries), "record": self.record(profile_digest)["digest"]}

    def info(self) -> dict[str, Any]:
        return {i.name: i.info() for i in self.installed}

    def __enter__(self) -> Composition:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def install(model: Any, ctx: Context, *, rank: int = 0) -> Composition:
    """Install every option of ``ctx.enabled`` on ``model``, in order; `Refused` (before any patch) names every refusal.
    An install that raises closes the options already installed, and uninstalls every hook the failing one made, before
    the error propagates."""
    bad = refusals(ctx)
    if bad:
        raise Refused(bad)
    stray = undeclared(ctx.enabled)
    if stray:
        raise Refused([f"options installed outside this composition are live: {list(stray)}"])
    comp = Composition(ctx)
    target = Target(model, ctx, rank)
    comp.weight_commitments = target.weight_commitments
    for entry in ctx.enabled:                          # phase 1: every commitment to the plaintext weights, model untouched
        name, _, scheme = entry.partition(":")
        commit = getattr(adapter(name), "commit_weights", None)
        if commit is not None:
            target.weight_commitments[entry] = dict(commit(scheme, target))
    try:
        for entry in ctx.enabled:                      # phase 2: install, innermost first
            name, _, scheme = entry.partition(":")
            opt = adapter(name).install(scheme, target)
            comp.installed.append(opt)
            if opt.name != entry:
                raise ValueError(f"the {name} adapter installed {opt.name!r} for {entry!r}")
    except BaseException:
        try:
            for h in reversed(target.made):            # newest first: the failing option's hooks before the ones it wraps
                h.uninstall()
        finally:
            comp.close()
        raise
    return comp


class WeightsView:
    """A model's parameters and buffers as a reader of live weights sees them (`named_parameters`, `named_buffers`), every
    tensor a protocol option released replaced by its plaintext.  The Commit's replay reads the weights of record through
    it (`check.replay.opening.model_weights_provider`) and re-registers them through it (`register_weights`)."""

    def __init__(self, model: Any, plaintext: Mapping[str, Any]) -> None:
        names = {n for n, _ in model.named_parameters()} | {n for n, _ in model.named_buffers()}
        stray = sorted(set(plaintext) - names)
        if stray:
            raise ValueError(f"released weights name no parameter or buffer of the model: {stray[:5]}")
        self.model, self.plaintext = model, dict(plaintext)

    def named_parameters(self, *a: Any, **kw: Any) -> Any:
        return ((n, self.plaintext.get(n, p)) for n, p in self.model.named_parameters(*a, **kw))

    def named_buffers(self, *a: Any, **kw: Any) -> Any:
        return ((n, self.plaintext.get(n, b)) for n, b in self.model.named_buffers(*a, **kw))


def weights_view(comp: Composition | None, model: Any) -> Any:
    """``model`` itself when no installed option released a weight, else its `WeightsView`."""
    plain = comp.weights() if comp is not None else {}
    return WeightsView(model, plain) if plain else model


def register_weights(comp: Composition | None, com: Any) -> Any:
    """``com.register_weights()`` over the weights of record: through `weights_view` while an option holds them released."""
    view = weights_view(comp, com.model)
    if view is com.model:
        return com.register_weights()
    with hooks.patched(com, "model", view):
        return com.register_weights()


def undeclared(enabled: tuple[str, ...] = ()) -> tuple[str, ...]:
    """The tags of live hook entries that name a protocol option outside ``enabled``: an option some other path installed."""
    return tuple(sorted({e.tag for e in hooks.live() if e.tag and e.tag.partition(":")[0] in ADAPTERS and e.tag not in enabled}))


def commit_context(profile: Any, *, world: int = 1, settings: Any = None) -> Context | None:
    """The Commit's composition context from its declared target.  None when the target declares no protocols (sampled
    proofs alone): the Commit runs as it always has, and no adapter is imported.  `Refused` names every refusal."""
    if profile is None or getattr(profile, "protocols", None) is None:
        return None
    ctx = Context(stage="commit", enabled=tuple(profile.protocols), world=int(world),
                  enforce_eager=getattr(profile, "compilation", None) is None, settings=load_settings(settings))
    bad = refusals(ctx)
    if bad:
        raise Refused(bad)
    return ctx


def at_row(knob: Any, workload: Mapping[str, Any], *, world: int = 1) -> tuple[str, ...] | None:
    """A row's protocol set, checked before any stage runs (the fail-closed point of single-GPU and TP rows alike): the
    ``PROTOCOLS`` knob, else the workload target's ``protocols``; None for sampled proofs alone, the row as it has always
    run.  Either is the whole set, sampled proofs included.  `Refused` names every refusal: a set without sampled proofs,
    a knob that contradicts the target, and every refusal of the set at the Commit (a TP row, an adapter not on this tree)."""
    enabled, declared = parse(knob), parse((workload.get("target") or {}).get("protocols"))
    if enabled and declared and enabled != declared:
        raise Refused([f"PROTOCOLS {list(enabled)} contradicts the workload target's protocols {list(declared)}"])
    enabled = enabled or declared
    if not enabled or enabled == config.DEFAULT_ROW_PROTOCOLS:
        return None
    if not any(e.startswith("sampled-proofs:") for e in enabled):
        raise Refused([f"{list(enabled)}: a row is a sampled-proofs run, so its set names sampled-proofs:<scheme> "
                       "(serve without it through verity_vllm.LLM)"])
    bad = refusals(Context(stage="commit", enabled=enabled, world=int(world), settings=load_settings(workload.get("protocols_config"))))
    if bad:
        raise Refused(bad)
    return enabled


def own_auditor(config: str | Path | Mapping[str, Any] | None, profile: Any) -> dict[str, dict[str, Any]]:
    """The protocols config (`load_settings`) of a pipeline that is its run's own auditor (the Commit, a Match capture):
    PoUW's ``coins`` a challenger of the epoch held in this process when ``profile`` declares PoUW (`pouw.challenged`)."""
    from verity_vllm.protocol_options import pouw
    return pouw.challenged(load_settings(config), getattr(profile, "protocols", None))


def commit_guard(profile: Any, *, world: int = 1, hot: bool = False, installs: bool = True, settings: Any = None) -> Any:
    """The Commit's check before its engine is built: ``profile`` back unchanged when the Commit can run its set, else
    exit 3 (fail-closed) with every reason printed.  ``hot`` is a hot Commit worker and ``installs`` False a Commit that
    installs no options (the TP Commit): both refuse any declared set."""
    try:
        ctx = commit_context(profile, world=world, settings=settings)
        if ctx is not None and (hot or not installs):
            raise Refused([f"this Commit ({'the hot worker' if hot else 'TP'}) installs no protocol options yet"])
    except Refused as e:
        print(f"[delta] REFUSE (fail-closed): {e}", flush=True)
        raise SystemExit(3) from None
    return profile


def capture_context(profile: Any, *, settings: Any = None) -> Context | None:
    """A Match capture's composition context: the declared target's executors and wrappers without sampled proofs, which
    commits only in the Commit, so the captured run executes what the Commit serves. None when the target declares no
    other protocol. `Refused` names every refusal."""
    enabled = tuple(e for e in (getattr(profile, "protocols", None) or ()) if e.partition(":")[0] != "sampled-proofs")
    if not enabled:
        return None
    ctx = Context(stage="capture", enabled=enabled, enforce_eager=getattr(profile, "compilation", None) is None,
                  settings={k: v for k, v in load_settings(settings).items() if k != "sampled-proofs"})
    bad = refusals(ctx)
    if bad:
        raise Refused(bad)
    return ctx


def at_capture(profile: Any, model: Any, settings: Any = None) -> Composition | None:
    """Install a Match capture's composition (`capture_context`) on its engine's model: PoUW's executor there serves each call
    and records it as one op (`pouw.ncp_linear_op`), committing nothing. None when the target declares no such protocol."""
    ctx = capture_context(profile, settings=settings)
    if ctx is None:
        return None
    comp = install(model, ctx)
    print(f"[m1] protocol options installed for the capture: {list(comp.entries)}", flush=True)
    return comp


def at_commit(profile: Any, model: Any, versions: dict[str, Any], settings: Any = None) -> Composition | None:
    """Install the Commit's composition on its model once every committer registered the weights of record, and record
    its entries in ``versions``.  None when the target declares no protocols."""
    ctx = commit_context(profile, settings=settings)
    if ctx is None:
        return None
    comp = install(model, ctx)
    versions["protocols"] = {"entries": list(comp.entries), "info": comp.info()}
    print(f"[delta] protocol options installed beside the committers: {list(comp.entries)}", flush=True)
    return comp


def into_verdict(comp: Composition | None, verdict: dict[str, Any], out: str | Path, profile: Any, *,
                 program_digest: str | None, manifest: Mapping[str, Any] | None) -> None:
    """At the Commit's end: close the composition, bind sampled proofs' commitment (the run roots, the Program and
    manifest digests), add the ``protocols`` block to ``verdict`` and write ``protocols.json``.  Nothing without one."""
    if comp is None:
        return
    comp.close()
    comp.bind("sampled-proofs", run_roots=verdict.get("run_roots"), program_digest=program_digest,
              manifest_digest=(manifest or {}).get("manifest_digest"))
    verdict["protocols"] = comp.verdict(profile.digest())
    Path(out, "protocols.json").write_text(json.dumps(comp.record(profile.digest()), indent=1, default=str) + "\n")
