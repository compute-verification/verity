"""What the compiler reads (`steps.compile_steps`, with `batch` and `segment`): every field of the advice and of a certified body,
each with how the output accounting takes it (`READS`), and the gate the compiler reads the advice through (`gate`), which refuses
a field `READS` doesn't declare by name (`UndeclaredRead`). The table is the code path: `advice_of` returns the gated advice, and
`policy_of` and `request_of` take a body's parameters by their roles here.

The output accounting bounds what the compiled circuit can carry of the served output y, given the certified inputs x and the
registration, by the advice's length and the wrong blocks' bits. That holds where each field the compiler reads is one of
`ACCOUNTS`:
- ``output``: a checked circuit output, one a request Program returns and the audit holds to y;
- ``certified``: a field of x, a request's certified body, or a field of the advice held to one;
- ``advice``: a field of the committed advice, charged in its length, whether or not y fixes it.

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
- each response's length, where its terminator goes: ``output_len``, advice, read ahead twice: on a batch-invariant target a
  request's Program is its whole request's (T = ``output_len`` − 1) in every window; and in class G a window's close returns a
  request it leaves with outputs to keep with its indexer's keys and scales on its edge where its last row reads past
  INDEX_TOPK keys (`segment.edge_groups`);
- each response's finish reason: ``finish``, advice, read only to refuse a record its body contradicts (`steps.request_of`);
- the outputs a request kept before the window opened: its carried ``kept``, advice, held to the last window's close
  (`window_advice.follows`);
- when each output was released: ``release``, never read; the gate refuses it.

**y as the request Programs return it.** A request Program returns its kept outputs' token ids first, ``Array(n, I32)`` in
output order, n the outputs its rows keep in the window (``output_len`` for a request whole in one), and on a batch-invariant
target nothing else; in class G then, for its rows past INDEX_TOPK keys, the bits that their top-k orders are the canonical
ones, and for a request the window leaves with outputs to keep, its edge (`build.build_program`'s ``edge``). It returns no
terminator and no finish reason: its ids end where its shape does, at ``output_len``, advice until a request Program checks
the stop rule (no stop token before the last output, which is a stop token or the cap's); a finish reason is the advice's
(eos or length follows from the ids and the cap once the stop rule is checked, and abort is the client's). A record of y that
holds each response's ids and then a terminator is those ids with the terminator after the n-th.
"""
from __future__ import annotations

from dataclasses import fields
from typing import Any, NamedTuple

from verity_vllm.pipeline import window_advice as WA

ACCOUNTS = ("output", "certified", "advice")
#: the advice's records by name, as `window_advice.decode` builds them
RECORDS = {"advice": WA.Advice, "request": WA.RequestAdvice, "step": WA.StepAdvice, "row": WA.Row, "source": WA.Source,
           "carried": WA.Carried}
#: a certified body's field roles: the prompt (read apart, `steps.request_of`); how the response travels, which computes nothing;
#: what only finishes a request (its T, which the advice's rows carry; `max_tokens` and `ignore_eos` hold its `output_len` and
#: `finish`); what the sampling policy reads (`steps.policy_of`); and a parameter refused off its inert values
ROLES = ("prompt", "transport", "finish", "policy", "inert")


class Field(NamedTuple):
    """A field the compiler reads: ``record`` one of `RECORDS`, ``body`` (a certified body's) or ``program`` (a request Program's
    outputs), its ``account`` (`ACCOUNTS`), the part of y it fixes (``y``, empty for none), the request unit's fields it shapes
    ahead of the step that outputs that part (``ahead``), and a body field's ``role`` (`ROLES`) and, for an inert one, the values
    that leave a Program unchanged (``inert``)."""

    record: str
    name: str
    account: str
    y: str = ""
    ahead: tuple[str, ...] = ()
    role: str = ""
    inert: tuple = ()


READS: tuple[Field, ...] = (
    *(Field("advice", n, "advice") for n in ("requests", "steps", "carried")),
    Field("request", "id", "certified", y="the response a token belongs to"),
    Field("request", "prompt_len", "certified"),
    Field("request", "output_len", "advice", y="a response's length", ahead=("T", "edge")),
    Field("request", "finish", "advice", y="a response's finish reason"),
    Field("step", "rows", "advice"),
    *(Field("row", n, "advice") for n in ("request", "kind", "start", "n", "cache_hit", "preempted", "sources")),
    *(Field("source", n, "advice") for n in ("request", "n")),
    *(Field("carried", n, "advice", y="the outputs kept before the window" if n == "kept" else "")
      for n in ("C", "computed", "kept", "made")),
    Field("body", "prompt", "certified", role="prompt"),
    *(Field("body", n, "certified", role="transport") for n in ("model", "user", "stream", "stream_options", "return_token_ids")),
    *(Field("body", n, "certified", role="finish") for n in ("max_tokens", "stop", "stop_token_ids", "ignore_eos")),
    *(Field("body", n, "certified", role="policy") for n in ("temperature", "top_p", "seed")),
    *(Field("body", n, "certified", role="inert", inert=v) for n, v in (
        ("min_p", (None, 0, 0.0)), ("presence_penalty", (None, 0, 0.0)), ("frequency_penalty", (None, 0, 0.0)),
        ("repetition_penalty", (None, 1, 1.0)), ("min_tokens", (None, 0)), ("logit_bias", (None, {})), ("n", (None, 1)),
        ("top_k", (None, 0, -1)))),
)


def body(*roles: str) -> tuple[Field, ...]:
    """The body fields of ``roles``, in `READS` order."""
    return tuple(f for f in READS if f.record == "body" and f.role in roles)


def refusals(table: tuple[Field, ...] = READS) -> list[str]:
    """Each field of ``table`` the output accounting can't take, by name and why: declared twice; of an account other than
    `ACCOUNTS`; of no record the compiler receives; an advice record's field its encoding doesn't carry (neither committed nor
    held to x); an advice or body field declared a circuit output; a body field off a role, or with inert values off the inert
    role; or a field read ahead that isn't advice."""
    out, seen = [], set()
    for f in table:
        at = f"{f.record}.{f.name}"
        if (f.record, f.name) in seen:
            out.append(f"{at}: declared twice")
        seen.add((f.record, f.name))
        if f.account not in ACCOUNTS:
            out.append(f"{at}: account {f.account!r}, none of {', '.join(ACCOUNTS)}: a read the output accounting can't charge")
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
        elif f.record != "program" or f.account != "output":
            out.append(f"{at}: of no record the compiler receives (the advice's {', '.join(RECORDS)}, a certified body, or a request "
                       f"Program's outputs, account output)")
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
