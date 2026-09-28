"""The protocol options of a vLLM run: PoUW, POUS and sampled proofs, each enabled on its own, any of them in one run.

A run enables a set of ``name:scheme`` entries (`config.parse_protocols`), off by default.  `verity_vllm.LLM` takes it as
``protocols=``; a row declares it as its target's ``protocols`` (`TargetProfile.protocols`), so the Build, the Commit and
the Check read one copy.  Each protocol has one adapter module here (`interface`), imported only when it is enabled.  The
protocol logic is its package's (`verity_pouw`, `verity_pous`, `verity_sampled_proofs`), never this package's.

**Order.**  Adapters install in `config.PROTOCOLS` order, innermost first, and uninstall in reverse:

1. ``pouw`` wraps each linear layer's ``quant_method.apply`` and replaces its GEMM, reading the weight from the layer it
   is called with;
2. ``pous`` wraps the same ``apply`` around it: it decodes the weight from its encoded store and calls the inner
   ``apply`` with a layer holding the decoded weight, so the GEMMs read what POUS stores; its responder starts last;
3. ``sampled-proofs`` is outermost: the Commit's committer and taps commit the values the model produced.

Nothing changes the model before every enabled protocol has committed to the plaintext weights of record, so a Commit
installs its composition after its committers registered them (`at_commit`).

**Refused** (`refusals`): an unknown protocol; a scheme its package does not register, or an adapter not on this tree;
a protocols config naming a protocol the run does not enable; a set at TP > 1 (each rank would install its own, which is
not wired); and whatever an adapter refuses for its context (sampled proofs outside the Commit stage, for one).

**The record.**  `Composition.record` is ``verity-vllm/protocol-composition/v0``: the stage, the profile digest and, per
protocol, its entry and what it commits to, with the SHA-256 of its canonical JSON.  It is a placeholder: each protocol's
verifier checks its own claims, and nothing checks the links between them yet.  An option the Program does not describe
(``Installed.outside_program``) makes the Commit's verdict not of record (`Composition.verdict`).
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

__all__ = ["ADAPTERS", "RECORD_SCHEMA", "Composition", "Context", "Installed", "Option", "Refused", "Target", "adapter",
           "at_commit", "commit_context", "commit_guard", "install", "into_verdict", "load_settings", "parse", "refusals",
           "undeclared"]

#: protocol -> its adapter module in this package
ADAPTERS = {"pouw": "pouw", "pous": "pous", "sampled-proofs": "sampled_proofs"}
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


def refusals(ctx: Context) -> tuple[str, ...]:
    """Every reason the run described by ``ctx`` cannot enable ``ctx.enabled`` (empty: it can)."""
    out: list[str] = []
    names = [e.partition(":")[0] for e in ctx.enabled]
    if ctx.enabled and ctx.world > 1:
        out.append(f"protocol options at TP {ctx.world}: each rank would install its own composition, which is not wired yet")
    stray = sorted(set(ctx.settings) - set(names))
    if stray:
        out.append(f"the protocols config names {stray}, which this run does not enable")
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
    return tuple(out)


class Composition:
    """The options one run installed, in install order.  `close` uninstalls them newest first; `record` and `verdict`
    say what they committed to and whether the run's verdict can be of record."""

    def __init__(self, ctx: Context) -> None:
        self.context = ctx
        self.installed: list[Installed] = []
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

    def outside_program(self) -> dict[str, str]:
        """Each option the Program of record does not describe, with the reason."""
        return {i.name: i.outside_program for i in self.installed if i.outside_program}

    def record(self, profile_digest: str | None = None) -> dict[str, Any]:
        body = {"schema": RECORD_SCHEMA, "stage": self.context.stage, "profile_digest": profile_digest,
                "protocols": [{"entry": i.name, "commitment": i.commitment()} for i in self.installed]}
        canon = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        return {**body, "digest": hashlib.sha256(canon).hexdigest()}

    def verdict(self, profile_digest: str | None = None) -> dict[str, Any]:
        """The block a Commit's verdict carries: the set, whether a verdict under it can be of record, and the record's digest."""
        out = self.outside_program()
        return {"protocols": list(self.entries), "of_record": not out, "outside_program": out,
                "record": self.record(profile_digest)["digest"]}

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
    try:
        for entry in ctx.enabled:
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
