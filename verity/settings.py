"""Every environment variable ``verity`` reads, declared once: the distribution's one settings registry, in the shape of the vLLM
integration's (``integrations/vllm/verity_vllm/settings.py``) and vLLM's ``vllm/envs.py``.

A `Setting` has a name, a type (the parser of the variable's string), a default, a role and a doc.  The role says what the
setting can change:

* ``value`` (`VALUE`): what is computed, committed or decided.  `record` carries every value setting in force, with a digest,
  so a run's record says what it ran with.  A credential (declared ``secret``, or a name holding one of `SECRET_WORDS`) is
  recorded by its name alone, `REDACTED` in place of its value.
* ``where`` (`WHERE`): only where and how long things run (paths, devices, which tests run).  Telemetry: its doc says why it
  changes nothing a run computes.
* ``debug`` (`DEBUG`): diagnostics (dumps, traces, profiles), refused on a run of record: `validate_environ(of_record=True)`.

Read a setting as an attribute, ``settings.VERITY_ML_BOOLEAN_EXHAUSTIVE`` (or `get` by name): its value parsed from the
environment at that read, never cached, or its default when the variable is unset or empty.  An undeclared name is an
AttributeError, and a string that doesn't parse is a ValueError naming the variable.

The prefixes in `PREFIXES` are this registry's alone, so an undeclared variable under one is a misspelt or retired setting.
`validate_environ` refuses it by name (`UnknownSetting`), and so does every read and `record`, since a library has no startup
to check at: a typo fails the run that carries it instead of being ignored.  An area adds its prefix once every variable
under it is declared here.

An area reads its variables here, never ``os.environ`` itself: ``tests/lint/test_repo_env_reads.py`` holds the direct reads not
yet moved here, by file, to a list that only shrinks.  Stdlib only and nothing from ``verity``, so every unit may import it,
those on the soundness path included (``tests/test_boundaries.py``).
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

VALUE, WHERE, DEBUG = "value", "where", "debug"
ROLES = (VALUE, WHERE, DEBUG)
RECORD_SCHEMA = "verity/settings/v1"

_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")


def flag(raw: str) -> bool:
    v = raw.strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    raise ValueError(f"not a flag: one of {', '.join(_TRUE)} or {', '.join(_FALSE)}")


@dataclass(frozen=True)
class Setting:
    name: str
    type: Callable[[str], Any]
    default: Any
    role: str
    doc: str
    #: a credential, whatever its name says: `record` names it and never holds its value
    secret: bool = False


def _declare(*settings: Setting) -> dict[str, Setting]:
    out: dict[str, Setting] = {}
    for s in settings:
        if s.name in out:
            raise ValueError(f"{s.name} is declared twice")
        if s.role not in ROLES:
            raise ValueError(f"{s.name}: role {s.role!r} is not one of {ROLES}")
        out[s.name] = s
    return out


S = Setting
SETTINGS: dict[str, Setting] = _declare(
    # ---- verity.ml.circuits, its suites included
    S("VERITY_ML_BOOLEAN_EXHAUSTIVE", flag, False, WHERE, "the Boolean lowerings' exhaustive tests run, on every f32 word and every "
      "pair of bf16 words (2^32 instances each, minutes): which tests run, never what a circuit computes"),
)
del S

#: The prefixes only this registry declares under: an undeclared variable under one is refused.
PREFIXES: tuple[str, ...] = ("VERITY_ML_",)


class UnknownSetting(ValueError):
    """A variable under one of `PREFIXES` that `SETTINGS` doesn't declare."""


class DebugOnRecord(ValueError):
    """A ``debug`` setting in force on a run of record."""


def setting(name: str) -> Setting:
    s = SETTINGS.get(name)
    if s is None:
        raise AttributeError(f"{name} is not a declared setting (verity/settings.py)")
    return s


def unknown(environ: Iterable[str] | None = None) -> list[str]:
    return sorted(k for k in (os.environ if environ is None else environ) if k.startswith(PREFIXES) and k not in SETTINGS)


def validate_environ(environ: Mapping[str, str] | None = None, *, of_record: bool = False) -> None:
    """Refuse each variable under `PREFIXES` that nothing declares and, on a run of record, each ``debug`` setting set to
    anything but the empty string, naming them."""
    env = os.environ if environ is None else environ
    if bad := unknown(env):
        them = "them" if len(bad) > 1 else "it"
        raise UnknownSetting(f"unknown setting{'s' * (len(bad) > 1)} {', '.join(bad)}: verity/settings.py declares no such "
                             f"variable under {', '.join(PREFIXES)}; unset {them}, fix the name, or declare {them} there")
    if of_record and (on := sorted(k for k, v in env.items() if v != "" and k in SETTINGS and SETTINGS[k].role == DEBUG)):
        raise DebugOnRecord(f"debug setting{'s' * (len(on) > 1)} {', '.join(on)} on a run of record: a diagnostic setting may "
                            f"print or keep values, so a run of record runs without it; unset {'them' if len(on) > 1 else 'it'}")


def get(name: str, environ: Mapping[str, str] | None = None) -> Any:
    """A declared setting's value in ``environ`` (default the process environment)."""
    s = setting(name)
    env = os.environ if environ is None else environ
    validate_environ(env)
    v = env.get(name)
    if v is None or v == "":
        return s.default
    try:
        return s.type(v)
    except ValueError as e:
        raise ValueError(f"{name}={v!r}: {e}") from None


def __getattr__(name: str) -> Any:
    if name in SETTINGS:
        return get(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*globals(), *SETTINGS])


#: A variable whose name holds one of these anywhere is a credential, declared ``secret`` or not.
SECRET_WORDS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
#: What `record` holds in place of a credential's value.
REDACTED = "<redacted>"


def is_secret(name: str) -> bool:
    s = SETTINGS.get(name)
    return bool(s and s.secret) or any(w in name.upper() for w in SECRET_WORDS)


def record(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """The value settings in force: every declared ``value`` setting that is set, as the environment holds it (a credential by
    name, `REDACTED` in place of its value), with the digest, SHA-512 of the canonical JSON of ``values``."""
    env = os.environ if environ is None else environ
    validate_environ(env)
    values = {k: REDACTED if is_secret(k) else env[k] for k in sorted(env) if k in SETTINGS and SETTINGS[k].role == VALUE}
    blob = json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return {"schema": RECORD_SCHEMA, "values": values, "digest": hashlib.sha512(blob).hexdigest()}
