"""The messages between the adapter and the prover, in both directions.

In one process, through one `Exchange` a build (`build`; `current` is the build in scope): the adapter's hooks emit the events of `verity.hooks` to the
subscribers of their hooks, and a scheme, attached by its name before the engine is built, files its requests, each
answered yes or refused by the scheme's name before serving starts.  A subscriber only reads, so it may attach at any
time; a request changes what is served: one on what the workload declares (a property, a process beside it) is filed
before the build's answer, and any other before serving begins, at the first step.  The exchange names no scheme; the adapter's hooks translate vLLM's engine,
requests, tokens and KV blocks into the circuit's words.

On disk, by the types the proof layer and each declared scheme declare (`types`, `Scheme.messages`): a body is checked against its declared
type as it is written and as it is read, and its bytes are left as they are. `tests/workload_messages.toml` lists each
message's writers and readers.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, NamedTuple

from verity import hooks as H
from verity.channel import Message, MessageTypes, Refused
from verity.protocols.proofs.spot.prover import workload as spot
from verity.protocols.compose import schemes as declarations
from verity.protocols.resource_accounting import scheme as schemes
from verity_vllm import settings

__all__ = ["Exchange", "append", "attach_listed", "body", "build", "close", "current", "declaring", "line", "lines", "listed_refusals",
           "named", "option", "providing", "read", "write"]

type Subscriber = Callable[[Any], None]


def _pins(entries: Any) -> tuple[Mapping[str, str], ...]:
    """The application's ``[[pin]]`` entries, read-only (`Exchange.attach`'s ``ctx["pins"]``)."""
    return tuple(MappingProxyType(dict(p)) for p in entries)

_KEYS = itertools.count()


class Exchange:
    """One process's exchange: subscribers by hook, requests by scheme (filed until `answer`, fixed after it), and the
    workload's step, as the last `StepBegan` emitted it."""

    def __init__(self) -> None:
        self.subscribers: dict[str, list[tuple[str, Subscriber]]] = {h: [] for h in H.HOOKS}
        self.requests: list[tuple[str, H.Request]] = []
        self.declared: H.Declared | None = None
        self.fixed = False
        self._step: int | None = None
        #: the workload's readers a launcher offers a process beside it, by what they do; none: no launcher
        self.readers: Mapping[str, Callable[..., Any]] = {}
        #: the adapter's answer to a refused input (`RefuseInput`), called as it is filed with the input and the reason
        self.refuse: Callable[[str, str], None] | None = None
        #: the schemes `attach` attached, by name, each with its settings: `Declared.schemes` and `Declared.settings`
        self.attached: dict[str, Mapping[str, Any]] = {}
        #: the schemes `attach` attached, by name: what `weight_copy_refusals` reads of the next one's
        self.schemes: dict[str, schemes.Scheme] = {}
        #: the pairs `weight_copy` allowed among the attached schemes, each with what it does not show: serve.json's
        #: ``weight_copy_allowed``
        self.allowed: list[WeightCopyNote] = []
        #: what each attached scheme's ``attach`` returned, by its name, read by `scheme_plugin.write_reports`
        self.reports: dict[str, Any] = {}
        #: this build's key, unique in the process: a scheme's ``ctx["build"]``
        self.key = f"build-{next(_KEYS)}"
        #: the model's `H.RegisteredValues` has been emitted: its weights are loaded, and `attach` refuses
        self.registered = False

    def attach(self, scheme: schemes.Scheme, given: Mapping[str, Any] | None = None, *,
               gate: Callable[..., Mapping[str, Any]] | None = None, private: str | Path | None = None,
               **launcher: Any) -> Any:
        """Attach ``scheme`` with ``ctx`` (`verity.protocols.resource_accounting.scheme`) ``{"settings": its settings,
        ``given`` over its defaults; "build": this build's key; "gate": ``gate``; "private"; **launcher}``: its
        ``attach(self, ctx)``, then each of its ``requests(ctx)`` its attach did not file; returns what attach returns.
        ``gate`` is the auditor device's challenger gate in its own words, ``gate(*words) -> answer``
        (`commit_coins.ChallengerGate.call`), or None where the launcher holds none.  ``ctx["private"]`` is the scheme's
        own directory under the launcher's ``private`` (``private/<scheme's name>``, made here), outside every run
        output, where it keeps what never leaves the node across servers; None where the launcher holds none.
        ``ctx["pins"]`` is the application's ``[[pin]]`` entries, read-only, as the launcher's ``pins`` gives them or
        else as it was given them (``VERITY_PINS``), ``()`` for an application that pins nothing and None where neither
        is given; ``ctx["configuration"]`` the committed configuration's hash likewise (``configuration``,
        ``VERITY_CONFIGURATION``); a keyword and a setting that differ are refused (`ValueError`).
        A scheme that `weight_copy` refuses beside those already attached, in either order, is refused by name
        (`Refused`) before its attach runs, and the exchange is left as it was; the pairs it allows are `allowed`.
        Attached before the build is answered, it is declared (`H.Declared`); after, its requests are refused as
        `request` refuses them.

        In an engine's process (``ctx["process"]`` `hooks.ENGINE`) a launcher attaches every scheme before the
        workload's model begins to load, so an attach that returns has run with none of the workload's bytes on the
        device; the exchange refuses an engine's attach once the model's `hooks.RegisteredValues` has been emitted on
        it (`RuntimeError`). That catches a late attach, not one during the load: a scheme that needs an empty device
        checks the device itself (the wipe's idle-card check). An attach that names no engine process may come
        after the load (`serve_openai.network_admission` attaches admission to the loaded engine's exchange)."""
        if self.registered and launcher.get("process") == H.ENGINE:
            raise RuntimeError(f"{scheme.name} attached after the model's weights were registered: an attach in the "
                               "engine runs before the load")
        bad, allowed = weight_copy([*self.schemes.values(), scheme])
        if bad:
            raise Refused(scheme.name, "; ".join(bad))
        for key, name, read in (("pins", "VERITY_PINS", _pins), ("configuration", "VERITY_CONFIGURATION", str)):
            handed, held = launcher.get(key), settings.get(name)
            if handed is not None and held is not None and read(handed) != read(held):
                raise ValueError(f"{scheme.name}: the launcher's {key} differ from {name}, the {key} it was given: "
                                 "hand one, or the same")
            launcher[key] = None if (got := held if handed is None else handed) is None else read(got)
        own = None
        if private is not None:
            own = Path(private) / scheme.name
            own.mkdir(parents=True, exist_ok=True)
        ctx = {"settings": schemes.settings(scheme, given or {}), "build": self.key, "gate": gate, "private": own,
               **launcher}
        got = scheme.attach(self, ctx)
        for r in scheme.requests(ctx):
            if (scheme.name, r) not in self.requests:
                self.request(scheme.name, r)
        self.attached[scheme.name] = ctx["settings"]
        self.schemes[scheme.name] = scheme
        self.allowed = allowed
        if got is not None:
            self.reports[scheme.name] = got
        return got

    @property
    def step(self) -> int:
        """The workload's step: 0 before serving begins."""
        return 0 if self._step is None else self._step

    @property
    def next_step(self) -> int:
        """The step the workload schedules next, which a step being scheduled is until its `StepBegan`."""
        return 0 if self._step is None else self._step + 1

    @step.setter
    def step(self, value: int) -> None:
        self._step = value

    def subscribe(self, name: str, hook: str, fn: Subscriber) -> Callable[[], None]:
        """``name`` receives every event of ``hook`` from now on, in the order the adapter emits them; returns its
        removal."""
        if hook not in H.HOOKS:
            raise Refused(name, f"{hook!r} is not a hook of a served run {H.HOOKS}")
        entry = (name, fn)
        self.subscribers[hook].append(entry)
        return lambda: self.subscribers[hook].remove(entry) if entry in self.subscribers[hook] else None

    def request(self, name: str, request: H.Request) -> None:
        """File scheme ``name``'s request; kernels and gates filed earlier run outside later ones."""
        kind = type(request).__name__
        if isinstance(request, H.RefuseInput):
            self.requests.append((name, request))
            if self.refuse is not None:
                self.refuse(request.input, request.why)
            return
        if isinstance(request, H.ServeRoute) and (other := self.route(request.path)) is not None:
            raise Refused(name, f"serves route {request.path}, which {other[0]} serves")
        if self.declared is not None and isinstance(request, (H.RequireProperty, H.StartBeside)):
            raise Refused(name, f"files {kind} after the build was answered; file it before the engine is built")
        if self._step is not None:
            raise Refused(name, f"files {kind} after serving began (step {self._step}); file it before the first step")
        self.requests.append((name, request))

    def _filed[R](self, kind: type[R]) -> list[tuple[str, R]]:
        return [(n, r) for n, r in self.requests if isinstance(r, kind)]

    def answer(self, declared: H.Declared) -> bool:
        """Answer every request before an engine is built: `Refused` under the first refused scheme's name, with all
        its reasons; else True when a scheme's required property fixes the build.  Emits ``declared``."""
        why: dict[str, list[str]] = {}
        for name, r in H.unmet(self.requests, declared.properties):
            why.setdefault(name, []).append(f"requires {r.property} = {r.value!r}: {r.why}")
        for name, r in self.requests:
            if isinstance(r, H.StartBeside) and (r.process not in schemes.PROCESSES or set(r.readers) - set(self.readers)):
                missing = sorted(set(r.readers) - set(self.readers))
                why.setdefault(name, []).append(
                    f"starts {r.process} beside the workload, which no scheme registered" if r.process not in schemes.PROCESSES
                    else f"starts {r.process} beside the workload with readers {missing} no launcher here offers")
        if why:
            name, reasons = next(iter(why.items()))
            raise Refused(name, "; ".join(reasons))
        self.declared = declared
        self.fixed = any(r.fixes for _, r in self._filed(H.RequireProperty))
        self.emit(declared)
        return self.fixed

    def wants(self, hook: str) -> bool:
        """Whether any scheme reads ``hook``'s events: an emitter builds no event nobody reads."""
        return bool(self.subscribers[hook])

    def emit(self, event: Any) -> None:
        if isinstance(event, H.StepBegan):
            self.step = event.step
        elif isinstance(event, H.RegisteredValues):
            self.registered = True
        for _, fn in tuple(self.subscribers[event.hook]):
            fn(event)

    def compute(self, due: H.BlockDue, original: Callable[[Mapping[str, H.Handle] | None], H.Handle]) -> H.Handle:
        """A due block's output: emitted to ``blocks``' subscribers, then computed by the filed kernels, the workload's
        ``original`` innermost; inputs a kernel hands inward replace the block's for every kernel inside it."""
        self.emit(due)
        kernels = [r.kernel for _, r in self._filed(H.ComputeBlockWith)][::-1]

        def level(i: int, inputs: Mapping[str, H.Handle] | None) -> H.Handle:
            if i < 0:
                return original(inputs)
            seen = due if inputs is None else replace(due, inputs={**due.inputs, **inputs})
            return kernels[i](seen, lambda given: level(i - 1, inputs if given is None else {**(inputs or {}), **given}))

        return level(len(kernels) - 1, None)

    def computes(self) -> bool:
        """Whether a scheme computes due blocks: the adapter wraps its blocks only then."""
        return bool(self._filed(H.ComputeBlockWith))

    def start(self, input: str, start: Callable[[], None]) -> None:
        """Start ``input`` once every gate filed by `StartOnBoundary` starts it; with none, now."""
        for _, r in reversed(self._filed(H.StartOnBoundary)):
            start = (lambda g, s: lambda: g.add(input, s))(r.gate, start)
        start()

    def pending(self) -> bool:
        """Whether a gate holds an input not yet started."""
        return any(r.gate.pending() for _, r in self._filed(H.StartOnBoundary))

    def ready(self, due: H.StepDue) -> bool:
        """Whether step ``due.step`` may run: every gate is asked, and may start inputs."""
        return all([r.gate.ready(due) for _, r in self._filed(H.StartOnBoundary)])

    def open(self, why: str) -> None:
        """Every gate starts what it holds, and every later input at once."""
        for _, r in self._filed(H.StartOnBoundary):
            r.gate.open(why)

    def route(self, path: str) -> tuple[str, H.Route] | None:
        """The scheme serving ``path`` (`ServeRoute`) and its answer, None when none does."""
        return next(((n, r.serve) for n, r in self._filed(H.ServeRoute) if r.path == path), None)

    def refused(self, input: str) -> list[tuple[str, str]]:
        """Each scheme that refused input ``input`` (`RefuseInput`), with its reason."""
        return [(n, r.why) for n, r in self._filed(H.RefuseInput) if r.input == input]

    def oldest_freed_first(self) -> bool:
        """Whether a freed range is handed out again only after every range freed before it (`ReuseOldestFreed`)."""
        return bool(self._filed(H.ReuseOldestFreed))

    def beside(self) -> list[tuple[str, Callable[[], Any]]]:
        """The processes schemes asked to start beside the workload, each by its name and bound to the readers it
        named, for the launcher to call before serving."""
        out = []
        for _, r in self._filed(H.StartBeside):
            start, readers = schemes.PROCESSES[r.process], [self.readers[k] for k in r.readers]
            out.append((r.process, (lambda f, a: lambda: f(*a))(start, readers)))
        return out


def providing(role: str) -> schemes.Scheme:
    """The one declared scheme that provides ``role`` (`Scheme.provides`), the only one imported; `LookupError` naming
    those that do otherwise (`compose.schemes.named`)."""
    return declarations.get(declarations.named("provides", role))


def option(entry: str | None) -> schemes.Scheme | None:
    """The declared scheme whose option is ``entry`` (``name:scheme``, `Scheme.option`), which every launcher refuses,
    naming the scheme's name to list instead, the only one imported; None when none is."""
    try:
        return declarations.get(declarations.named("option", entry)) if entry else None
    except LookupError:
        return None


def declaring(port: str) -> schemes.Scheme:
    """The one declared scheme with a port named ``port`` (`Scheme.ports`), the only one imported; `LookupError` naming
    those that have otherwise (`compose.schemes.named`)."""
    return declarations.get(declarations.named("ports", port))


def named(entry: str) -> schemes.Scheme | None:
    """The declared scheme named ``entry``, the only one imported; None when none is (a declared scheme's option names it
    only to refuse it, `option`)."""
    return declarations.get(entry) if entry in declarations.DECLARATIONS else None


#: what a storage scheme declares: its resource (`Scheme.resource`) and the workload's store among its needs (`Scheme.needs`)
STORAGE = ("memory", "store")


def storage(s: Any) -> bool:
    needs = getattr(s, "needs", schemes.UNDECLARED)
    return getattr(s, "resource", None) == STORAGE[0] and not isinstance(needs, schemes.Lacks) and STORAGE[1] in needs


class WeightCopyNote(NamedTuple):
    """Storage ``storage`` attached beside ``keeps``, which declares it keeps a weight-derived copy and the ruling of
    that copy beside storage, with what the ruling does not show."""

    storage: str
    keeps: str
    ruling: str
    not_shown: tuple[str, ...]

    def record(self) -> dict[str, Any]:
        """Both declared facts, by the fields that declare them: serve.json's ``weight_copy_allowed`` entry."""
        return {"storage": self.storage, "keeps_weight_copy": self.keeps,
                schemes.BESIDE_STORAGE: {"ruling": self.ruling, "not_shown": list(self.not_shown)}}


def weight_copy(listed: Sequence[schemes.Scheme]) -> tuple[list[str], list[WeightCopyNote]]:
    """Storage (`storage`) beside a scheme that keeps a weight-derived copy (``keeps_weight_copy``) is refused by name
    unless that scheme's declaration rules its copy beside storage (`schemes.BESIDE_STORAGE`, a non-empty sentence),
    and then the pair is allowed and noted with the ruling and what a ruling does not show
    (`schemes.BESIDE_STORAGE_NOT_SHOWN`); beside one that has not declared whether it keeps one, refused.
    In any order, sorted by name, so the order listed or attached changes neither. A scheme keeps a weight-derived
    copy when, across forwards, it holds or keeps alive resident bytes from which the weights can be had without
    decoding storage's store; whether a product reads them does not enter."""
    refused, noted = [], []
    for st in sorted({s.name: s for s in listed if storage(s)}.values(), key=lambda s: s.name):
        for o in sorted({s.name: s for s in listed if s.name != st.name}.values(), key=lambda s: s.name):
            keeps = getattr(o, "keeps_weight_copy", schemes.UNDECLARED)
            ruled = getattr(o, schemes.BESIDE_STORAGE, schemes.UNDECLARED)
            if keeps is True and isinstance(ruled, str) and ruled.strip():
                noted.append(WeightCopyNote(st.name, o.name, ruled, schemes.BESIDE_STORAGE_NOT_SHOWN))
            elif keeps is True:
                refused.append(f"{st.name} is storage (it needs the workload's store) and {o.name} keeps a weight-derived copy "
                               f"(keeps_weight_copy), and {o.name}'s declaration does not say that its copy beside storage has "
                               f"been ruled ({schemes.BESIDE_STORAGE} is undeclared or empty), so the two are not attached in one process")
            elif isinstance(keeps, schemes.Lacks):
                refused.append(f"{st.name} is storage (it needs the workload's store) and {o.name}'s declaration does not say whether it keeps a "
                               f"weight-derived copy (keeps_weight_copy is undeclared), so storage is not attached beside it "
                               "until it declares that, and a scheme that keeps one beside storage is refused until its copy beside storage is ruled")
    return refused, noted


def weight_copy_refusals(listed: Sequence[schemes.Scheme]) -> list[str]:
    return weight_copy(listed)[0]


def listed_refusals(listed: Sequence[schemes.Scheme]) -> list[str]:
    """Every reason ``listed``, a configuration's schemes in its order, can't be attached together in one process,
    checked before any of them attaches; empty: it can."""
    names = [s.name for s in listed]
    twice = [f"{n} is listed {names.count(n)} times" for n in dict.fromkeys(names) if names.count(n) > 1]
    return twice + weight_copy_refusals(listed) + schemes.order_refusals(listed)


def attach_listed(exchange: Exchange, listed: Sequence[tuple[schemes.Scheme, Mapping[str, Any]]], **ctx: Any) -> list[Any]:
    """Each scheme of ``listed`` (a configuration's list, each with its settings) attached on ``exchange`` with ``ctx``
    (`Exchange.attach`) in list order, once `listed_refusals` refuses none: the order `schemes.order_refusals` holds,
    in which the scheme listed earlier is the outer (`Exchange.request`); what each attach returned."""
    if bad := listed_refusals([s for s, _ in listed]):
        raise Refused("listed schemes", "; ".join(bad))
    return [exchange.attach(s, given, **ctx) for s, given in listed]


_CURRENT: Exchange | None = None


def current() -> Exchange:
    """The exchange of the build in scope: the last build's (`build`), or the one `using` names; opened on first use."""
    global _CURRENT
    if _CURRENT is None:
        _CURRENT = Exchange()
    return _CURRENT


def build() -> Exchange:
    """The exchange the build about to start files on and answers: the current one until a build answered it, then a
    fresh one, made current. An answered exchange stays with its engine, which puts it back in scope with `using`."""
    global _CURRENT
    if _CURRENT is None or _CURRENT.declared is not None:
        _CURRENT = Exchange()
    return _CURRENT


@contextmanager
def using(exchange: Exchange) -> Iterator[Exchange]:
    """``exchange`` is the build in scope (`current`) inside the block, the one before it after."""
    global _CURRENT
    before, _CURRENT = _CURRENT, exchange
    try:
        yield exchange
    finally:
        _CURRENT = before


def close() -> None:
    """End the exchange in scope; the next `current` opens a fresh one."""
    global _CURRENT
    _CURRENT = None


_TYPES: tuple[int, MessageTypes] | None = None


def types() -> MessageTypes:
    """The message types on disk: the proof layer's, useful work's workload list (H1's, which no scheme declares), and
    those of every scheme registered so far (`Scheme.messages`); `body` loads every declaration only for a type none
    of them declares (`compose.schemes.workload_messages`)."""
    global _TYPES
    registry = schemes.SCHEMES
    if _TYPES is None or _TYPES[0] != len(registry):
        declared = {id(m): m for s in registry.values() for m in s.messages}
        _TYPES = (len(registry), MessageTypes((*spot.MESSAGES, *declarations.workload_messages(), *declared.values())))
    return _TYPES[1]


def body(type_: str, data: bytes) -> bytes:
    """``data`` as the body of a ``type_`` message, unchanged; `verity.channel.Refused` when its declaration refuses
    it."""
    msg = Message(type_, body=data)
    try:
        schema = types().schema(msg.type, msg.version)
    except Refused as e:
        if e.name != "unknown-type":
            raise
        declarations.load()
        schema = types().schema(msg.type, msg.version)
    if (why := schema.why(msg.head(), msg.body)) is not None:
        raise Refused(*why)
    return data


def write(path: Path, type_: str, data: bytes) -> None:
    """Write ``data`` to ``path`` as a ``type_`` message's body."""
    path.write_bytes(body(type_, data))


def read(path: Path, type_: str) -> bytes:
    """``path``'s bytes, a ``type_`` message's body."""
    return body(type_, path.read_bytes())


def line(type_: str, data: bytes) -> bytes:
    """``data`` as one line of a JSON-lines file, a ``type_`` message's body."""
    return body(type_, data) + b"\n"


def append(path: Path, type_: str, data: bytes) -> None:
    """Append ``data`` to ``path`` as one line, a ``type_`` message's body."""
    with open(path, "ab") as fh:
        fh.write(line(type_, data))


def lines(path: Path, type_: str) -> list[bytes]:
    """``path``'s lines but blank ones, each a ``type_`` message's body."""
    return [body(type_, line) for line in path.read_bytes().splitlines() if line.strip()]
