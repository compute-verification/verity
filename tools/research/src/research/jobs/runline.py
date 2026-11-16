"""A `research run` line's options, each checked against an allowlist."""
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
    """(option, value) pairs, a flag's value None; raises Refused for an option outside ``allowed``."""
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
