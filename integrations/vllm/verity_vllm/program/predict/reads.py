"""What the compiler reads (`steps.compile_steps`, with `batch` and `segment`): every field of the advice and of a certified body,
each with how the output accounting takes it (`READS`), and the gate the compiler reads the advice through (`gate`), which refuses
a field `READS` doesn't declare by name (`UndeclaredRead`). The table is the code path: `advice_of` returns the gated advice, and
`policy_of` and `request_of` take a body's parameters by their roles here.

The output accounting bounds what the compiled circuit can carry of the served output y, given the certified inputs x and the
registration, by the advice's length and the wrong blocks' bits. That holds where each field the compiler reads is one of
`ACCOUNTS`:
- ``output``: a checked circuit output, one a request Program returns and the audit holds to y;
- ``certified``: a field of x, a request's certified body, or a field of the advice held to one, or the window's seed salt,
  public and nobody serving's choice;
- ``advice``: a field of the committed advice, charged in its length, whether or not y fixes it.

**A stochastic request's seed** is certified: its body's ``seed``, else derived from its certified id and the window's seed salt
(``window.seed_salt``, `engine.served_seed.derived`), never one the server draws or fills in. The seed salt is the challenger's
coin released at the window's open and held in the window record as certified input, 128 lowercase hex characters, which the
compile takes as their bytes (``bytes.fromhex(window["seed_salt"])``). `seed_of` refuses by name any other: a request with no
seed in its body where the compile has no seed salt, and one the window carries in, whose seed is its admitting window's. The
seed its sampler reads is a witness its Program holds to that one: a stochastic request's Program returns its seed rule
(``program.seed_rule``, `frontend.seed_rule`'s `SeedCheck_v2` over its certified ``seeded``, ``body_seed`` and ``user`` and the
window's ``seed_salt``), which any other seed makes 0 and the replay refuses by name (`check.replay.checks`), so the seed is
charged nothing. Without that output a seed would be 64 bits a request of the prover's choosing: `refusals` refuses a seed
read where the table doesn't declare it, and the gate refuses a seed read from the advice by name.

**A prefix-cache hit** (a row's ``cache_hit`` and ``sources``, and a carried request's ``C``, the hit it was admitted on) is
advice charged at its width, its words in the committed advice (I1's ``cache_hit`` and ``sources``), where the registration
states the prefix cache on or leaves it unstated (vLLM's default), and absent where it states it off (``switch``, `off`):
`steps.plan_steps` refuses a hit by name there. No hit comes from anywhere else: never derived by verifier-side Python (a
replay of the engine's block pool) nor read from the served record (`refusals`).

A field is read ``ahead`` where it shapes a request's unit (`steps.RequestUnit`, the fields named) before the step that outputs
the part of y it fixes. Such a field is advice, its bits charged whatever the circuit does with it. Read from y instead, as a
checked output, it would need prefix stability: changing y at or after a step changes nothing the circuit computes before it
(the plan's rows and launches up to there, and its Programs' calls). `test_reads` holds the plan prefix stable at every step of
its windows, and the Programs too but in one read: a batch-invariant Program's length sizes only its later decodes, but in
class G a request's Program computes its indexer's keys and scales for the rows it runs before the window's close only where
the close returns them, where its last row, after the close, reads past INDEX_TOPK keys. y's parts:
- its token ids: each kept output is a token select's sampled id in its request's Program (`steps.EGRESS`), an output the
  compiler never reads;
- which response a token belongs to: the request's id, certified (the body it names is the request's, by its `user`);
- each response's length, where its terminator goes: ``output_len``, advice, read ahead three times: on a batch-invariant
  target a request's Program is its whole request's (T = ``output_len`` − 1) in every window; in class G a window's close
  returns a request it leaves with outputs to keep with its indexer's keys and scales on its edge where its last row reads past
  INDEX_TOPK keys (`segment.edge_groups`); and it sets the request's length rule (`steps.stop_of`, `RequestUnit.stop`);
- each response's finish reason: ``finish``, advice, read ahead to set the request's length rule (`RequestUnit.stop`), and to
  refuse a record its body contradicts (`ingress_policy.request_of`);
- the outputs a request kept before the window opened: its carried ``kept``, advice, held to the last window's close
  (`window_advice.follows`);
- when each output was released: ``release``, never read; the gate refuses it.

**y as the request Programs return it.** A request Program returns its kept outputs' token ids first, ``Array(n, I32)`` in
output order, n the outputs its rows keep in the window (``output_len`` for a request whole in one), and on a batch-invariant
target nothing else; in class G then, for its rows past INDEX_TOPK keys, the bits that their top-k orders are the canonical
ones, and for a request the window leaves with outputs to keep, its edge (`build.build_program`'s ``edge``). It returns no
terminator and no finish reason: its ids end where its shape does, at ``output_len``, which its length rule
(`RequestUnit.stop`: `StopCheck_v1` over `StopRule_v1`) checks against the ids (no stop token before the last output, which
is a stop token or the cap's); a finish reason is the advice's, eos or length held by the same rule to the ids and the cap,
and abort the client's. The egress record (the network certifier's, per request) is n ``DATA`` cells, the token ids in
order, then one ``END`` at cell n, and ``PAD`` elsewhere; it sends no finish reason. Its ``DATA`` cells are the Program's
ids, and its ``END`` sits at ``output_len``.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from typing import Any, NamedTuple

from verity_vllm.engine import served_seed
from verity_vllm.pipeline import window_advice as WA
from verity_vllm.program.predict.common import Unexpressible

ACCOUNTS = ("output", "certified", "advice")
#: the advice's records by name, as `window_advice.decode` builds them
RECORDS = {"advice": WA.Advice, "request": WA.RequestAdvice, "step": WA.StepAdvice, "row": WA.Row, "source": WA.Source,
           "carried": WA.Carried}
#: what the compiler reads of the window beside its advice and bodies: its seed salt (`steps.compile_steps`'s ``salt``)
WINDOW = ("seed_salt",)
#: the fields a stochastic request's seed is read from, the only two (`seed_of`)
SEED = (("body", "seed"), ("window", "seed_salt"))
#: the request Program's output that holds its sampler to that seed (`frontend.seed_rule.NAME`, its Call)
SEED_RULE = ("program", "seed_rule")
_SEED_WHY = ("a stochastic request's seed is certified, its body's seed or derived from its id and the window's seed salt, and "
             "its Program's seed rule holds the seed its sampler reads to it: never a value the server draws or fills in")
#: the registration's switch the prefix-cache hit's fields are under
PREFIX_CACHE = "enable_prefix_caching"
#: a certified body's field roles: the prompt (read apart, `ingress_policy.request_of`); how the response travels, which computes
#: nothing; what only finishes a request (its T, which the advice's rows carry; `max_tokens` and `ignore_eos` hold its `output_len`
#: and `finish`); what the sampling policy reads (`ingress_policy.policy_of`); and a parameter refused off its inert values
ROLES = ("prompt", "transport", "finish", "policy", "inert")


class Field(NamedTuple):
    """A field the compiler reads: ``record`` one of `RECORDS`, ``body`` (a certified body's), ``window`` (`WINDOW`) or
    ``program`` (a request Program's outputs), its ``account`` (`ACCOUNTS`), the part of y it fixes (``y``, empty for none), the
    request unit's fields it shapes ahead of the step that outputs that part (``ahead``), a body field's ``role`` (`ROLES`) and,
    for an inert one, the values that leave a Program unchanged (``inert``), and the registered engine switch off which an
    advice field is absent (``switch``, `off`)."""

    record: str
    name: str
    account: str
    y: str = ""
    ahead: tuple[str, ...] = ()
    role: str = ""
    inert: tuple = ()
    switch: str = ""


READS: tuple[Field, ...] = (
    *(Field("advice", n, "advice") for n in ("requests", "steps", "carried")),
    Field("request", "id", "certified", y="the response a token belongs to"),
    Field("request", "prompt_len", "certified"),
    Field("request", "output_len", "advice", y="a response's length", ahead=("T", "edge", "stop")),
    Field("request", "finish", "advice", y="a response's finish reason", ahead=("stop",)),
    Field("step", "rows", "advice"),
    *(Field("row", n, "advice", switch=PREFIX_CACHE if n in ("cache_hit", "sources") else "")
      for n in ("request", "kind", "start", "n", "cache_hit", "preempted", "sources")),
    *(Field("source", n, "advice") for n in ("request", "n")),
    *(Field("carried", n, "advice", y="the outputs kept before the window" if n == "kept" else "",
            switch=PREFIX_CACHE if n == "C" else "") for n in ("C", "computed", "kept", "made")),
    Field("body", "prompt", "certified", role="prompt"),
    *(Field("body", n, "certified", role="transport") for n in ("model", "user", "stream", "stream_options", "return_token_ids")),
    *(Field("body", n, "certified", role="finish") for n in ("max_tokens", "stop", "stop_token_ids", "ignore_eos")),
    *(Field("body", n, "certified", role="policy") for n in ("temperature", "top_p", "seed")),
    *(Field("body", n, "certified", role="inert", inert=v) for n, v in (
        ("min_p", (None, 0, 0.0)), ("presence_penalty", (None, 0, 0.0)), ("frequency_penalty", (None, 0, 0.0)),
        ("repetition_penalty", (None, 1, 1.0)), ("min_tokens", (None, 0)), ("logit_bias", (None, {})), ("n", (None, 1)),
        ("top_k", (None, 0, -1)))),
    Field("window", "seed_salt", "certified"),
    Field(*SEED_RULE, "output"),
)


def body(*roles: str) -> tuple[Field, ...]:
    """The body fields of ``roles``, in `READS` order."""
    return tuple(f for f in READS if f.record == "body" and f.role in roles)


def off(engine: Mapping[str, Any]) -> frozenset[tuple[str, str]]:
    """The advice fields a registration with engine args ``engine`` switches off, each absent from its advice: those whose
    ``switch`` it states False (a switch it doesn't state is vLLM's default, on)."""
    return frozenset((f.record, f.name) for f in READS if f.switch and engine.get(f.switch) is False)


def seed_of(i: int, request_id: str, seed: int | None, seed_salt: bytes | None, carried: bool = False) -> int:
    """Request ``i``'s seed, a stochastic request's, read from `SEED`: its certified body's ``seed``, else `served_seed.derived`
    from its certified ``request_id`` and the window's ``seed_salt``. Refused by name where it would be neither: with no seed
    salt, and for a request the window carries in (``carried``), whose seed is derived from its admitting window's seed salt."""
    if seed is not None:
        return int(seed)
    if seed_salt is None or carried or type(seed_salt) is not bytes or not seed_salt:
        why = ("the window carries it in, so its seed derives from its admitting window's seed salt, which this compile doesn't "
               "read" if carried else "the compile has no window seed salt to derive one from" if seed_salt is None else
               f"the window's seed salt isn't non-empty bytes ({type(seed_salt).__name__} {seed_salt!r})")
        raise Unexpressible(f"request {i} ({request_id!r}): stochastic with no seed in its certified body, and {why}: a seed serving "
                            f"fills in (`served_seed.DEFAULT_SEED`) or draws is neither its body's nor derived from its id and the "
                            f"window's seed salt (`served_seed.derived`)")
    return served_seed.derived(seed_salt, request_id)


def hits_off(advice: Any, engine: Mapping[str, Any]) -> list[tuple[int, str]]:
    """Each prefix-cache hit ``advice`` carries, a row's or a carried request's, where the registration's ``engine`` states the
    prefix cache off (`off`), as (request, why): its advice has none there."""
    gone, out = off(engine), []
    for s, step in enumerate(advice.steps):
        out += [(r.request, f"request {r.request}: a prefix-cache hit of {r.cache_hit} tokens, and the registration states the "
                            f"prefix cache off ({PREFIX_CACHE} False), at step {s}: a hit is advice charged at its width only with "
                            f"the prefix cache on") for r in step.rows if r.cache_hit and ("row", "cache_hit") in gone]
    return out + [(i, f"request {i}: carried in on a prefix-cache hit of {c.C} tokens, and the registration states the prefix "
                      f"cache off ({PREFIX_CACHE} False)") for i, c in enumerate(advice.carried) if c.C and ("carried", "C") in gone]


def refusals(table: tuple[Field, ...] = READS) -> list[str]:
    """Each field of ``table`` the output accounting can't take, by name and why: declared twice; of an account other than
    `ACCOUNTS`; of no record the compiler receives; an advice record's field its encoding doesn't carry (neither committed nor
    held to x); an advice or body field declared a circuit output; a body field off a role, or with inert values off the inert
    role; a field read ahead that isn't advice; a seed read from anything but `SEED`, certified, or where the request Program's
    seed rule (`SEED_RULE`) isn't an output; or a prefix-cache hit's field other than advice under `PREFIX_CACHE`."""
    out, seen = [], set()
    if any((f.record, f.name) in SEED for f in table) and SEED_RULE not in {(f.record, f.name) for f in table if f.account == "output"}:
        out.append(f"{'.'.join(SEED_RULE)}: not a declared output, so a seed read from {', '.join('.'.join(s) for s in SEED)} holds "
                   f"nothing the sampler reads, which is 64 bits a request of the prover's choosing: {_SEED_WHY}")
    for f in table:
        at = f"{f.record}.{f.name}"
        if (f.record, f.name) in seen:
            out.append(f"{at}: declared twice")
        seen.add((f.record, f.name))
        if f.account not in ACCOUNTS:
            out.append(f"{at}: account {f.account!r}, none of {', '.join(ACCOUNTS)}: a read the output accounting can't charge")
        if ((f.record, f.name) in SEED or f.name == "seed") and ((f.record, f.name) not in SEED or f.account != "certified"):
            out.append(f"{at}: {_SEED_WHY}")
        if f.name in ("cache_hit", "sources") and (f.record, f.account, f.switch) != ("row", "advice", PREFIX_CACHE):
            out.append(f"{at}: a prefix-cache hit is a row's advice, charged at its width, under {PREFIX_CACHE}: never derived by "
                       f"verifier-side Python (a replay of the engine's block pool) nor read from the served record")
        elif f.switch and (f.record not in RECORDS or f.account != "advice"):
            out.append(f"{at}: switched by {f.switch}, so it is advice the registration can switch off, not {f.account!r}")
        if f.record in RECORDS:
            if f.name not in {x.name for x in fields(RECORDS[f.record])}:
                out.append(f"{at}: not a field of the advice's {f.record} record, so neither committed nor held to x")
            elif f.account == "output":
                out.append(f"{at}: a field of the advice, not a circuit output")
        elif f.record == "body":
            if f.account != "certified":
                out.append(f"{at}: a certified body's field, account {f.account!r}")
            if f.role not in ROLES or bool(f.inert) != (f.role == "inert"):
                out.append(f"{at}: role {f.role!r} with inert values {f.inert!r}, not one of {', '.join(ROLES)} (inert values only "
                           f"for an inert one)")
        elif f.record == "window":
            if f.name not in WINDOW or f.account != "certified":
                out.append(f"{at}: of the window the compiler reads only its seed salt, certified ({', '.join(WINDOW)})")
        elif f.record != "program" or f.account != "output":
            out.append(f"{at}: of no record the compiler receives (the advice's {', '.join(RECORDS)}, a certified body, the window's "
                       f"seed salt, or a request Program's outputs, account output)")
        if f.ahead and f.account != "advice":
            out.append(f"{at}: read ahead of the step that outputs it (it shapes {', '.join(f.ahead)}), so it must be committed "
                       f"advice: the circuit before that step may depend on it (class G's edge does)")
    return out


class UndeclaredRead(AttributeError):
    """A read of an advice field `READS` doesn't declare."""


_KIND = {cls: name for name, cls in RECORDS.items()}
_DECLARED = frozenset((f.record, f.name) for f in READS if f.record in RECORDS)


def _wrap(x: Any) -> Any:
    if type(x) in _KIND:
        return _View(x)
    return tuple(map(_wrap, x)) if type(x) is tuple else x


class _View:
    """An advice record as the compiler reads it: its declared fields, each record in them a view again; any other field an
    `UndeclaredRead`."""

    __slots__ = ("_record", "_read")

    def __init__(self, record: Any) -> None:
        object.__setattr__(self, "_record", record)
        object.__setattr__(self, "_read", {})

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        record, read = self._record, self._read
        if name not in read:
            kind = _KIND[type(record)]
            if (kind, name) not in _DECLARED and "seed" in name:
                raise UndeclaredRead(f"the compiler reads the advice's {kind}.{name}: {_SEED_WHY}")
            if (kind, name) not in _DECLARED:
                raise UndeclaredRead(f"the compiler reads {kind}.{name}, which `predict.reads.READS` doesn't declare: declare it with "
                                     f"its account (output, certified or advice) or don't read it")
            read[name] = _wrap(getattr(record, name))
        return read[name]

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"the gated advice is read-only ({name})")

    def __repr__(self) -> str:
        return f"gate({self._record!r})"


def gate(advice: WA.Advice) -> Any:
    """``advice`` as the compiler reads it: each field `READS` declares, and any other refused by name (`UndeclaredRead`)."""
    if type(advice) is not WA.Advice:
        raise TypeError(f"the gate takes a window's Advice, not a {type(advice).__name__}")
    return _View(advice)
