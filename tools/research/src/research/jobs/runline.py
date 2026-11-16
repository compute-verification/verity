"""A `research run` line's options, each checked against an allowlist: the control pod's `metrics-rerun` loop
(infra/control/metrics_rerun.py) and the dispatcher's `research-run` jobs (`research.jobs.dispatch`) launch only what theirs
holds.

An allowlist maps each option to the form of its value: None for a flag, a frozenset of the values it may take, a compiled
pattern its value must match, or a function that returns why a value is refused (None when it is not). An option outside it
is refused, and so is a value that starts with `-`, which `research run` would read as an option."""
from __future__ import annotations

import re
from typing import Callable

Form = None | frozenset | re.Pattern | Callable[[str], str | None]


class Refused(ValueError):
    """An option line refused, with why: a sentence that follows "it ", as "gives --pool twice"."""


def _why(form: Form, value: str) -> str | None:
    if value.startswith("-"):
        return "which starts with '-', so `research run` would read it as an option"
    if isinstance(form, frozenset):
        return None if value in form else "which isn't " + " or ".join(sorted(form))
    if isinstance(form, re.Pattern):
        return None if form.match(value) else f"which isn't of the form {form.pattern}"
    return form(value)


def options(tokens: list[str], allowed: dict[str, Form], *, unknown: Callable[[str], str],
            repeatable: frozenset[str] = frozenset()) -> list[tuple[str, str | None]]:
    """``tokens``, the options between `research run` and `--`, as (option, value) pairs, a flag's value None. Raises Refused
    for an option outside ``allowed`` (``unknown(option)`` says why), one given twice that isn't ``repeatable``, a flag given a
    value, an option given none, or a value not of its option's form."""
    raw, out, seen = list(tokens), [], set()
    while raw:
        name, eq, value = raw.pop(0).partition("=")
        if name not in allowed:
            raise Refused(unknown(name))
        if name in seen and name not in repeatable:
            raise Refused(f"gives {name} twice")
        seen.add(name)
        form = allowed[name]
        if form is None:
            if eq:
                raise Refused(f"gives the flag {name} a value")
            out.append((name, None))
            continue
        if not eq:
            if not raw:
                raise Refused(f"gives {name} no value")
            value = raw.pop(0)
        if why := _why(form, value):
            raise Refused(f"gives {name} {value!r}, {why}")
        out.append((name, value))
    return out
