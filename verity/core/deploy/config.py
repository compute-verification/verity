"""The deployment configuration: the public file a run starts from (`verity run CONFIG`), which
`verity config new | check | hash` writes, checks and hashes.

    format = "verity/config/v0"
    protocols = ["sampled-proofs", "certifier"]      the components, by the six subprotocols' names
    [workload]  name, params                         the computation audited, and its public shape
    [service]   law                                  the proof system's selection law (no ε, δ or threshold)
    [program]   commit                               the commit of the auditor-trusted code the parties run
    [params."<id>"]                                  a listed component's public structural parameters
    [[party]]   name, machine, principal             machine: `local`, or a name `verity run --on` gives
    [[channel]] from, to, messages, after            after: the messages sent before this channel's

A file states only its choices: a component's parameters and the workload's default to what the component or the
workload declares (``default_params``), and `check` returns the configuration with them filled, which is what the hash
covers, so stating a default changes no hash.

What a protocol declares (`Component`: its id, the workload hooks it needs, the components it requires, its parties and
channels as its manifest's `Role`s and `Channel`s, the parties that must share a machine, its parameters' schema and the
service laws it runs under) and what a workload declares (`Workload`: the hooks it provides, its parties as `Role`s, its
parameters' schema) are handed in by the caller: this module imports no protocol and no workload. `check` refuses by
name (`verity.core.wire.Refused`) an unknown key or version, a component whose requirements aren't listed or whose needs
the workload doesn't provide, a party or channel other than the declared ones, and a placement the declarations forbid.

The canonical form is canonical JSON of the checked configuration (sorted keys, no whitespace, strings ASCII without
DEL, no float), and its hash is SHA-512 of that form without the transport-only fields (`TRANSPORT`: a party's address
and listen port, and which side of a channel dials), so moving a listener changes no hash.
"""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from verity.core.deploy.manifest import Channel, Role
from verity.core.wire import Refused

FORMAT = "verity/config/v0"
#: the transport-only fields, by table: hashed out, so that they can change without changing the configuration
TRANSPORT = {"party": ("address", "listen"), "channel": ("dial",)}
TOP = ("channel", "format", "params", "party", "program", "protocols", "service", "workload")
#: the top-level keys a file must state: the rest default (``params``: each component's own)
REQUIRED = ("channel", "format", "party", "program", "protocols", "service", "workload")
PARTY = ("machine", "name", "principal", *TRANSPORT["party"])
CHANNEL = ("after", "from", "messages", "to", *TRANSPORT["channel"])
LOCAL = "local"
_COMMIT = re.compile(r"[0-9a-f]{40}")
_NAME = re.compile(r"[a-z][a-z0-9-]*")


def _no_params(params: Mapping[str, Any]) -> dict[str, Any]:
    if params:
        raise Refused("unknown-key", f"this component takes no parameters, not {sorted(params)}")
    return {}


@dataclass(frozen=True)
class Component:
    """What a protocol declares for a configuration. ``params`` checks its parameters (the file's, over
    ``default_params``) and returns them normalized; ``law`` refuses a service law the component can't run under
    (raising `Refused`), given its checked parameters; ``product``: whether `Core.Composition.product` takes it."""

    name: str
    id: str
    parties: tuple[Role, ...]
    channels: tuple[Channel, ...] = ()
    needs: frozenset[str] = frozenset()
    requires: tuple[str, ...] = ()
    colocated: tuple[tuple[str, str], ...] = ()
    product: bool = True
    params: Callable[[Mapping[str, Any]], dict[str, Any]] = _no_params
    default_params: Mapping[str, Any] = field(default_factory=dict)
    law: Callable[[str, Mapping[str, Any]], None] | None = None
    default_law: str | None = None


@dataclass(frozen=True)
class Workload:
    """What a workload declares: the hooks it provides protocols (``windows``, ``ingress-relay``, ...), its parties, and
    its parameters' schema (``params`` over ``default_params``, as for `Component`)."""

    name: str
    provides: frozenset[str]
    parties: tuple[Role, ...]
    params: Callable[[Mapping[str, Any]], dict[str, Any]]
    default_params: Mapping[str, Any] = field(default_factory=dict)


def parse(text: str) -> dict[str, Any]:
    """The TOML ``text`` as a dict, its syntax refused by name."""
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise Refused("toml", str(e)) from None


def _keys(where: str, got: Mapping[str, Any], allowed: Iterable[str], required: Iterable[str] = ()) -> None:
    if not isinstance(got, Mapping):
        raise Refused("type", f"{where} is a table, not {type(got).__name__}")
    if extra := sorted(set(got) - set(allowed)):
        raise Refused("unknown-key", f"{where}: {', '.join(extra)} (known: {', '.join(sorted(allowed))})")
    if missing := [k for k in required if k not in got]:
        raise Refused("missing-key", f"{where}: {', '.join(missing)}")


def _str(where: str, v: Any) -> str:
    if not isinstance(v, str) or not v:
        raise Refused("type", f"{where} is a non-empty string, not {v!r}")
    return v


def _names(where: str, v: Any) -> tuple[str, ...]:
    if isinstance(v, str):
        v = [v]
    if not isinstance(v, list) or not all(isinstance(x, str) and x for x in v):
        raise Refused("type", f"{where} is a list of names, not {v!r}")
    if len(set(v)) != len(v):
        raise Refused("repeated", f"{where} names one twice: {v}")
    return tuple(v)


def _ascii(where: str, s: str) -> None:
    if not s.isascii() or "\x7f" in s:
        raise Refused("ascii", f"{where}: string {s[:40]!r} is not ASCII without DEL, as canonical JSON's strings are")


def _plain(where: str, v: Any) -> None:
    """A value is a string (ASCII without DEL), an integer, a boolean, a list or a table of them: never a float or a
    date."""
    if isinstance(v, str):
        _ascii(where, v)
        return
    if isinstance(v, bool | int):
        return
    if isinstance(v, float):
        raise Refused("float", f"{where} = {v!r}: a configuration holds integers and exact ratios as strings, no float")
    if isinstance(v, list):
        for i, x in enumerate(v):
            _plain(f"{where}[{i}]", x)
    elif isinstance(v, Mapping):
        for k, x in v.items():
            if isinstance(k, str):
                _ascii(f"{where} key", k)
            _plain(f"{where}.{k}", x)
    else:
        raise Refused("type", f"{where} = {v!r}: not a string, integer, boolean, list or table")


def declared(components: Sequence[Component],
             workload: Workload) -> tuple[dict[str, Role], dict[tuple[str, str], Channel]]:
    """The parties (name -> `Role`) and channels ((sender, receiver) -> `Channel`) that ``components`` and ``workload``
    declare between them. Two declarations of one party must agree on its principal and cardinality, and it needs what
    either needs; two of one channel are its union, under the tighter latency bound."""
    parties: dict[str, Role] = {}
    channels: dict[tuple[str, str], Channel] = {}
    for who, roles in ((workload.name, workload.parties), *((c.name, c.parties) for c in components)):
        for r in roles:
            if (old := parties.get(r.name)) is not None:
                if (old.principal, old.per) != (r.principal, r.per):
                    raise ValueError(f"{who} declares party {r.name} as {r.principal} per {r.per}, another declaration "
                                     f"as {old.principal} per {old.per}")
                r = replace(old, needs=tuple(dict.fromkeys(old.needs + r.needs)))
            parties[r.name] = r
    for c in components:
        for ch in c.channels:
            if (old := channels.get((ch.sender, ch.receiver))) is not None:
                bounds = [b for b in (old.latency_ms, ch.latency_ms) if b is not None]
                ch = replace(old, messages=tuple(dict.fromkeys(old.messages + ch.messages)),
                             after=tuple(dict.fromkeys(old.after + ch.after)), latency_ms=min(bounds, default=None),
                             carries_coins=old.carries_coins or ch.carries_coins)
            channels[ch.sender, ch.receiver] = ch
    carried = {m for ch in channels.values() for m in ch.messages}
    for ch in channels.values():
        if unknown := [m for m in ch.after if m not in carried]:
            raise ValueError(f"channel {ch.sender} -> {ch.receiver} is after {unknown}, which no channel carries")
        if unknown := [p for p in (ch.sender, ch.receiver) if p not in parties]:
            raise ValueError(f"channel {ch.sender} -> {ch.receiver} names undeclared parties {unknown}")
    return parties, channels


def check(cfg: Mapping[str, Any], components: Mapping[str, Component],
          workloads: Mapping[str, Workload]) -> dict[str, Any]:
    """``cfg`` checked against the declarations (by subprotocol name and by workload name), or `Refused` by name; the
    checked configuration, each parameter table as its schema normalizes it over its defaults."""
    _keys("the configuration", cfg, TOP, REQUIRED)
    _plain("the configuration", cfg)
    fmt = _str("format", cfg["format"])
    if fmt != FORMAT:
        if fmt.startswith(FORMAT.rsplit("/", 1)[0] + "/"):
            raise Refused("unknown-version", f"format {fmt!r}: this verity reads {FORMAT}")
        raise Refused("unknown-format", f"format {fmt!r}, not {FORMAT}")
    names = _names("protocols", cfg["protocols"])
    if not names:
        raise Refused("no-protocol", "protocols lists no component")
    if unknown := [n for n in names if n not in components]:
        raise Refused("unknown-protocol", f"{', '.join(unknown)} (known: {', '.join(components)})")
    listed = [components[n] for n in names]
    for c in listed:
        if missing := [r for r in c.requires if r not in names]:
            raise Refused("requires", f"{c.name} requires {', '.join(missing)}, which protocols doesn't list")
    _keys("[workload]", cfg["workload"], ("name", "params"), ("name",))
    wname = _str("workload.name", cfg["workload"]["name"])
    if wname not in workloads:
        raise Refused("unknown-workload", f"{wname} (known: {', '.join(workloads)})")
    workload = workloads[wname]
    for c in listed:
        if lacks := sorted(c.needs - workload.provides):
            raise Refused("needs", f"{c.name} needs {', '.join(lacks)} from the workload, which {wname} doesn't provide "
                                   f"(it provides {', '.join(sorted(workload.provides)) or 'nothing'})")
    wparams = workload.params({**workload.default_params, **_table("[workload.params]",
                                                                   cfg["workload"].get("params", {}))})
    given = _table("[params]", cfg.get("params", {}))
    _keys("[params]", given, [c.id for c in components.values()])
    if extra := sorted(set(given) - {c.id for c in listed}):
        raise Refused("unknown-params", f"[params] for {', '.join(extra)}, which protocols doesn't list")
    params = {c.id: c.params({**c.default_params, **_table(f"[params.\"{c.id}\"]", given.get(c.id, {}))})
              for c in listed}
    _keys("[service]", cfg["service"], ("law",), ("law",))
    law = _str("service.law", cfg["service"]["law"])
    for c in listed:
        if c.law is not None:
            c.law(law, params[c.id])
    _keys("[program]", cfg["program"], ("commit",), ("commit",))
    if not _COMMIT.fullmatch(str(cfg["program"]["commit"])):
        raise Refused("commit", f"[program] commit is a 40-hex git commit, not {cfg['program']['commit']!r}")
    try:
        parties, channels = declared(listed, workload)
    except ValueError as e:
        raise Refused("declaration", str(e)) from None
    got = _parties(cfg["party"], parties)
    links = _channels(cfg["channel"], channels)
    for c in listed:
        for a, b in c.colocated:
            if got[a]["machine"] != got[b]["machine"]:
                raise Refused("placement", f"{c.name} runs {a} and {b} on one machine, not on {got[a]['machine']} and "
                                           f"{got[b]['machine']}")
    return {**dict(cfg), "protocols": list(names), "params": params, "workload": {"name": wname, "params": wparams},
            "party": list(got.values()), "channel": links}


def _table(where: str, v: Any) -> dict[str, Any]:
    if not isinstance(v, Mapping):
        raise Refused("type", f"{where} is a table, not {type(v).__name__}")
    return dict(v)


def _parties(entries: Any, declared_parties: Mapping[str, Role]) -> dict[str, dict]:
    if not isinstance(entries, list):
        raise Refused("type", "[[party]] is an array of tables")
    got: dict[str, dict] = {}
    for i, p in enumerate(entries):
        _keys(f"[[party]] {i}", p, PARTY, ("machine", "name", "principal"))
        name = _str(f"[[party]] {i} name", p["name"])
        if name in got:
            raise Refused("repeated-party", name)
        if name not in declared_parties:
            raise Refused("unknown-party", f"{name}: no listed component or the workload declares it (declared: "
                                           f"{', '.join(sorted(declared_parties))})")
        if p["principal"] != (principal := declared_parties[name].principal):
            raise Refused("principal", f"{name} is the {principal}'s, not the {p['principal']}'s")
        if not _NAME.fullmatch(_str(f"party {name} machine", p["machine"])):
            raise Refused("machine", f"party {name}: machine {p['machine']!r} is {LOCAL!r} or a lowercase name")
        if "listen" in p and (type(p["listen"]) is not int or not 0 < p["listen"] < 65536):
            raise Refused("listen", f"party {name}: listen is a port, not {p['listen']!r}")
        if "address" in p:
            _str(f"party {name} address", p["address"])
        got[name] = dict(p)
    if missing := sorted(set(declared_parties) - set(got)):
        raise Refused("missing-party", ", ".join(missing))
    return got


def _channels(entries: Any, declared_channels: Mapping[tuple[str, str], Channel]) -> list[dict]:
    """The channels checked against the declared ones, each with its ``after`` as a list (empty when it names none)."""
    if not isinstance(entries, list):
        raise Refused("type", "[[channel]] is an array of tables")
    seen, out = set(), []
    for i, c in enumerate(entries):
        _keys(f"[[channel]] {i}", c, CHANNEL, ("from", "messages", "to"))
        key = (_str(f"[[channel]] {i} from", c["from"]), _str(f"[[channel]] {i} to", c["to"]))
        if key in seen:
            raise Refused("repeated-channel", f"{key[0]} -> {key[1]}")
        seen.add(key)
        want = declared_channels.get(key)
        if want is None:
            raise Refused("undeclared-channel", f"{key[0]} -> {key[1]}: no listed component declares it")
        messages, after = _names(f"channel {key[0]} -> {key[1]} messages", c["messages"]), _names(
            f"channel {key[0]} -> {key[1]} after", c.get("after", []))
        if set(messages) != set(want.messages) or set(after) != set(want.after):
            raise Refused("channel", f"{key[0]} -> {key[1]} carries {list(want.messages)} after {list(want.after)}, "
                                     f"not {list(messages)} after {list(after)}")
        if "dial" in c and c["dial"] not in ("from", "to"):
            raise Refused("dial", f"{key[0]} -> {key[1]}: dial is 'from' or 'to', not {c['dial']!r}")
        out.append({**dict(c), "messages": list(messages), "after": list(after)})
    if missing := sorted(set(declared_channels) - seen):
        raise Refused("missing-channel", ", ".join(f"{a} -> {b}" for a, b in missing))
    return out


def canonical(value: Any) -> bytes:
    """Canonical JSON: `verity.ml.circuits.codec.canonical_json`'s bytes, which this unit may not import; a float or a
    string past ASCII without DEL is refused (`_plain`)."""
    _plain("the value", value)
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def hashed(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """``cfg`` without its transport-only fields (`TRANSPORT`): what the hash covers."""
    out = dict(cfg)
    for table, fields in TRANSPORT.items():
        out[table] = [{k: v for k, v in e.items() if k not in fields} for e in cfg.get(table, [])]
    return out


def digest(cfg: Mapping[str, Any]) -> str:
    """The configuration's hash: SHA-512 of the canonical JSON of `hashed` of the checked configuration, in hex."""
    return hashlib.sha512(canonical(hashed(cfg))).hexdigest()


def new(workload: str, protocols: Sequence[str], components: Mapping[str, Component],
        workloads: Mapping[str, Workload], commit: str, machine: str = LOCAL) -> dict[str, Any]:
    """A configuration of ``protocols`` on ``workload`` that states only its choices: every declared party on
    ``machine``, the declared channels, and the first default law of a listed component that every listed component
    runs under; parameters are left to their defaults. Refused by name like `check`, which it passes."""
    if unknown := [n for n in protocols if n not in components]:
        raise Refused("unknown-protocol", f"{', '.join(unknown)} (known: {', '.join(components)})")
    if workload not in workloads:
        raise Refused("unknown-workload", f"{workload} (known: {', '.join(workloads)})")
    listed = [components[n] for n in protocols]
    w = workloads[workload]
    try:
        parties, channels = declared(listed, w)
    except ValueError as e:
        raise Refused("declaration", str(e)) from None
    law = next((d for d in (c.default_law for c in listed) if d and _runs_under(d, listed)), "")
    cfg = {"format": FORMAT, "protocols": list(protocols), "workload": {"name": workload},
           "service": {"law": law}, "program": {"commit": commit},
           "party": [{"name": p, "machine": machine, "principal": r.principal} for p, r in parties.items()],
           "channel": [{"from": c.sender, "to": c.receiver, "messages": list(c.messages),
                        **({"after": list(c.after)} if c.after else {})} for c in channels.values()]}
    check(cfg, components, workloads)
    return cfg


def _runs_under(law: str, listed: Sequence[Component]) -> bool:
    try:
        for c in listed:
            if c.law is not None:
                c.law(law, c.params(dict(c.default_params)))
    except Refused:
        return False
    return True


def _bare(key: str) -> str:
    return key if re.fullmatch(r"[A-Za-z0-9_-]+", key) else json.dumps(key)


def _value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, list):
        return "[" + ", ".join(_value(x) for x in v) + "]"
    raise TypeError(f"no TOML rendering of {v!r}")


def _render_table(lines: list[str], name: str, table: Mapping[str, Any]) -> None:
    lines += ["", f"[{name}]"]
    lines += [f"{_bare(k)} = {_value(v)}" for k, v in table.items() if not isinstance(v, Mapping)]
    for k, v in table.items():
        if isinstance(v, Mapping):
            _render_table(lines, f"{name}.{_bare(k)}", v)


def render(cfg: Mapping[str, Any], header: Sequence[str] = ()) -> str:
    """``cfg`` as TOML text that `parse` reads back to ``cfg``, after ``header``'s comment lines."""
    lines = [f"# {h}" if h else "#" for h in header]
    lines += [f"{k} = {_value(cfg[k])}" for k in ("format", "protocols")]
    for k in ("workload", "service", "program"):
        _render_table(lines, k, cfg[k])
    for pid, p in cfg.get("params", {}).items():
        _render_table(lines, f"params.{_bare(pid)}", p)
    for k in ("party", "channel"):
        for e in cfg[k]:
            lines += ["", f"[[{k}]]", *(f"{_bare(f)} = {_value(v)}" for f, v in e.items())]
    return "\n".join(lines) + "\n"


__all__ = ["CHANNEL", "FORMAT", "LOCAL", "PARTY", "REQUIRED", "TOP", "TRANSPORT", "Component", "Workload", "canonical",
           "check", "declared", "digest", "hashed", "new", "parse", "render"]
