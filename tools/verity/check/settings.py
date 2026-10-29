#!/usr/bin/env python3
"""Every environment variable `check` reads, declared once: the template is ``integrations/vllm/verity_vllm/settings.py``.

A `Setting` has a name, a type (the parser of the variable's string), a default, a role, a doc and, for a ``where`` setting, the
reason it can't change a verdict.  The role says what the setting can change:

* ``value`` (`VALUE`): what a step computes or how a verdict is reached.  `record` hashes every value setting in force, and the
  value families (`FAMILIES`: the ``VERITY_*`` and ``FLOCK_*`` switches tests read, which `check.py` passes its steps and
  `suites.py` keys a pass on), into the run's ``result.json``.
* ``where`` (`WHERE`): only where and when things run: paths, caches, slots, memory budgets, run ids.  `record` carries them as
  telemetry, outside the digest, and `suites.py` keys no pass on one.  Each says why (`Setting.reason`).
* ``debug`` (`DEBUG`): a switch for a hand run that a run of record never takes: `refusals` refuses one set inside a recorded run
  (`RESEARCH_RUN_ID`), and `train.sh` refuses one in the lander's shell (``settings.py --of-record``).

Read a setting as an attribute, ``S.CHECK_CACHE`` (or `get` by name): its value parsed from the process environment at that read,
never cached, or its default when the variable is unset or empty.  An undeclared name is an AttributeError.  `refusals` names
each variable of a prefix this registry owns (`OWNED`: the ``CHECK_*`` knobs and the Lean slots' ``LEAN_SLOT*``) that it doesn't
declare and nothing else in the tree reads (`ELSEWHERE`): `check.py` and `suites.py` refuse to start with one, so a misspelt or
retired knob fails the run instead of being ignored.

The check package reads ``os.environ`` here.  The reads left outside are ``tools/verity/check/tests/test_settings.py``'s
`DIRECT`, which only shrinks: `slot.py` and `lean_slot.py` run alone, their source sent to a machine as ``python3 -``
(``check.py --slots``, ``research.auto``, ``research.remote``), `common.py` is in every Lean package's audit key
(`lean_audit.py`), and the read guard (``guard/``) loads in every test process before anything else; each variable they read
is declared here all the same.  Stdlib only, and nothing from beside it: a machine's ``python3`` can run it, and every module
of the package loads it.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Callable, Iterable, Mapping
from typing import Any, NamedTuple

VALUE, WHERE, DEBUG = "value", "where", "debug"
ROLES = (VALUE, WHERE, DEBUG)
RECORD_SCHEMA = "verity-check/settings/v1"


class Setting(NamedTuple):
    name: str
    type: Callable[[str], Any]
    default: Any
    role: str
    doc: str
    #: a ``where`` setting's reason it changes no verdict
    reason: str = ""
    #: a credential, whatever its name says: `record` names it and never holds its value
    secret: bool = False


def _declare(*settings: Setting) -> dict[str, Setting]:
    out: dict[str, Setting] = {}
    for s in settings:
        if s.name in out:
            raise ValueError(f"{s.name} is declared twice")
        if s.role not in ROLES:
            raise ValueError(f"{s.name}: role {s.role!r} is not one of {ROLES}")
        if (s.role == WHERE) != bool(s.reason):
            raise ValueError(f"{s.name}: a {WHERE} setting, and only one, says why it changes no verdict (reason=)")
        out[s.name] = s
    return out


S = Setting
SETTINGS: dict[str, Setting] = _declare(
    # ---- check's own knobs
    S("CHECK_CACHE", str, None, WHERE, "check's caches: circuit-check's reports, the Lean audits, the agreement's passes; unset, "
      "~/.cache/verity-check (`common.cache_dir`)", "a kept verdict is reused only under its inputs' hash"),
    S("CHECK_LEAN_DEPS_URLS", str, None, WHERE, "a file of the Lean dependency bundles' presigned URLs, for a run sent none "
      "(`--lean-deps-files`)", "each bundle is pinned by its hash (`lean-deps.json`)"),
    S("CHECK_MIN_MEM_GB", str, "", WHERE, "slot.py takes only a slot whose mem= is at least this many GB (`check.py --record "
      "--priority` sets it)", "placement: which slot a check waits for"),
    S("CHECK_PEERS", str, None, WHERE, "the ssh names of the machines a twin claims on; unset, /etc/vy/peers (slot.py, "
      "pod_setup.sh)", "placement: where a twin's claim is made"),
    S("CHECK_RUN_MEM_GB", str, None, WHERE, "the memory a `research run --mem-gb` run asks of a slot (research's cli sets it): "
      "slot.py raises each slot's mem= to it", "placement: which slot a run waits for"),
    S("CHECK_SCOPE_UNIT", str, None, WHERE, "the check's own systemd scope, which slot.py sets on a scope=1 slot: the spans read "
      "its memory (`spans.check_memory`)", "telemetry: the memory a check took"),
    S("CHECK_SLOT", str, None, WHERE, "the check slot this process holds (slot.py sets it for CMD)", "placement: the slot held"),
    S("CHECK_SLOTS", str, None, WHERE, "the machine's slots file; unset, /workspace/research/locks/slots (slot.py, "
      "pod_setup.sh)", "placement: which slots a check may take"),
    S("CHECK_SLOT_MEM_GB", str, "", WHERE, "the GB the check slot gives the run it holds (slot.py sets it): circuit-check's "
      "budget and the suites' jobs", "it sizes how much runs at once, never what a step computes"),
    S("CHECK_SLOT_RANK", str, "", WHERE, "drive's train position, which `slot.py --rank` orders priority waiters by",
      "placement: the order trains take slots in"),
    S("CHECK_STORE_CUSTODY", str, None, WHERE, "the recording run's custody dir, which check gives its steps in place of the "
      "RESEARCH_* that named the store (`store_io`)", "where the store's key is, not what a step computes"),
    # ---- the Lean slots (lean_slot.py)
    S("LEAN_SLOT", str, None, WHERE, "the Lean audit slot this process holds, POOL/I (lean_slot.py sets it)",
      "placement: the slot held"),
    S("LEAN_SLOTS", str, None, WHERE, "the machine's Lean slots file; unset, /workspace/research/locks/lean-slots",
      "placement: which slots an audit may take"),
    S("LEAN_SLOT_POOL", str, None, WHERE, "the Lean slot pool a wait takes; check sets `check` for its steps",
      "placement: which pool an audit waits in"),
    # ---- what check hands its steps (`check.env`), and the suites' runner (suites.py)
    S("VERITY_STORE_REQUIRED", str, None, VALUE, "1: a store-backed test fails where the store is unreachable instead of "
      "skipping; check sets it in a recorded run"),
    S("VERITY_REGRESSION", str, None, VALUE, "1: integrations/vllm's reference regression runs; check sets it in a recorded run"),
    S("VERITY_SKIP_STORE", str, None, DEBUG, "1: the store-backed tests skip, for a hand run without the store"),
    S("VERITY_TEST_CACHE", str, None, WHERE, "the suites' kept passes; unset, $XDG_CACHE_HOME/verity/tests",
      "a kept pass is reused only under its inputs' hash"),
    S("VERITY_TEST_CACHE_MIN_S", str, None, WHERE, "keep per-test passes only of tests whose call took at least this many "
      "seconds (`suites.py --per-test-min-s`)", "which passes are kept, not what they say"),
    S("FLOCK_AGREEMENT_INPUTS", str, None, WHERE, "a directory check also looks in for the agreement's pinned upstream build "
      "and inputs (a local recorded run's launcher sets it)", "each is found by its sha256 in the pin"),
    S("PYTHONPATH", str, "", VALUE, "where Python imports from: check hands its steps the entries inside this tree"),
    S("PATH", str, "", WHERE, "where programs are", "which tools a suite finds is keyed (`suites.TOOLS`), their versions pinned"),
    S("UV", str, None, WHERE, "the uv that `uv run` runs under, which the preflight's `uv lock --check` uses",
      "pod_setup.sh pins its version, which the preflight checks"),
    S("VIRTUAL_ENV", str, None, WHERE, "the virtualenv `uv run` put first on the PATH, which a planning process drops",
      "`suites.py` refuses another checkout's environment"),
    S("XDG_CACHE_HOME", str, None, WHERE, "the cache root under which VERITY_TEST_CACHE defaults", "as VERITY_TEST_CACHE"),
    S("TMPDIR", str, None, WHERE, "scratch: train.sh's worktrees and logs; unset, /tmp", "scratch space"),
    # ---- the run this one is part of (`research run` sets them)
    S("RESEARCH_RUN_ID", str, None, WHERE, "the research run this process is part of: a check inside one is recorded, a run of "
      "record", "it names the run"),
    S("RESEARCH_RUN_DIR", str, None, WHERE, "that run's directory", "where its records go"),
    S("RESEARCH_RUN_MEM_GB", str, None, WHERE, "the memory `research run --mem-gb` reserved for the run, which sizes the suites' "
      "jobs", "it sizes how much runs at once"),
    S("RESEARCH_LANE", str, None, WHERE, "the lane that asked for the run, which slot waiters and plans name", "telemetry"),
    S("RESEARCH_AGENT_URL", str, None, WHERE, "the agent that asked for the run, when no lane is named", "telemetry"),
    S("RESEARCH_SOURCE_SHA", str, None, WHERE, "the commit `research run` shipped, which the preflight holds the tree to",
      "telemetry: the tree itself is what is checked"),
    S("RESEARCH_RAMLOCK_DIR", str, None, WHERE, "research.mem's reservations; unset, /workspace/ramlock (slot.py)",
      "placement: what memory is reserved"),
    S("RESEARCH_PREFLIGHT_INPUTS", str, None, WHERE, "the files sent with the run, which the preflight reads (`research run "
      "--preflight`)", "the preflight decides whether a machine is prepared, never a verdict"),
    S("RESEARCH_PREFLIGHT_CUSTODY", str, None, WHERE, "the run's custody dir, through which the preflight reads the store",
      "as RESEARCH_PREFLIGHT_INPUTS"),
    # ---- train.sh (the lander's launch) and pod_setup.sh
    S("TRAIN_AS", str, None, WHERE, "the handle train.sh posts a failed launch as; unset, ci", "telemetry"),
    S("MERGE_THREAD", str, None, WHERE, "the Slack thread train.sh posts a failed launch to", "telemetry"),
    S("AWS_ACCESS_KEY_ID", str, None, WHERE, "the parent store key, which train.sh requires to mint the run's custody key",
      "a credential, never what a step computes", secret=True),
    S("AWS_SESSION_TOKEN", str, None, WHERE, "a temporary credential's token, which train.sh refuses: it cannot mint the "
      "custody key", "a credential", secret=True),
    S("R2_ENDPOINT", str, None, WHERE, "the store's endpoint, which train.sh requires beside AWS_ACCESS_KEY_ID",
      "where the store is"),
    S("UV_VERSION", str, None, WHERE, "the uv pod_setup.sh installs; unset, its pin (`tests/test_uv_pin.py`)",
      "the preflight checks the pinned version"),
    S("POD_SETUP_MARKER", str, None, WHERE, "the file pod_setup.sh writes when a pod is prepared; unset, "
      "~/.research/pod-setup.json", "where the marker is"),
    S("LC_ALL", str, None, WHERE, "the locale pod_setup.sh generates when the image names it without it",
      "the suites' outputs are compared under one locale"),
)
del S

#: Namespaces of variables read or passed as a whole: the role of each variable of the family `SETTINGS` doesn't declare.  The
#: longest matching prefix wins.  ``VERITY_SUITE_*`` is the runner's own (`VERITY_SUITE_ROOT`, `..._MODULE`, `..._READS`, ...),
#: where it tells the read guard (``guard/``) what the suite may read: never what a test computes.
FAMILIES: dict[str, str] = {"VERITY_SUITE_": WHERE, "VERITY_": VALUE, "FLOCK_": VALUE}
#: The prefixes this registry owns: a variable of one that `SETTINGS` doesn't declare and `ELSEWHERE` doesn't name is refused.
OWNED = ("CHECK_", "LEAN_SLOT")
#: Names of an owned prefix that the rest of the tree reads, which a check may inherit: benchmarks' shell scripts.
ELSEWHERE = frozenset({"CHECK_ARGS", "CHECK_ONLY"})

#: A variable whose name holds one of these anywhere is a credential, declared or not.
SECRET_WORDS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
#: What `record` holds in place of a credential's value.
REDACTED = "<redacted>"


def setting(name: str) -> Setting:
    s = SETTINGS.get(name)
    if s is None:
        raise AttributeError(f"{name} is not a declared setting (tools/verity/check/settings.py)")
    return s


def _env(environ: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if environ is None else environ


def raw(name: str, environ: Mapping[str, str] | None = None) -> str | None:
    """A declared setting's string as the environment holds it, None when unset."""
    setting(name)
    return _env(environ).get(name)


def is_set(name: str, environ: Mapping[str, str] | None = None) -> bool:
    """The variable is set (to anything, the empty string included)."""
    return raw(name, environ) is not None


def get(name: str, environ: Mapping[str, str] | None = None) -> Any:
    """A declared setting's value in ``environ`` (default the process environment): its default when unset or empty."""
    s = setting(name)
    v = _env(environ).get(name)
    if not v:
        return s.default
    try:
        return s.type(v)
    except ValueError as e:
        raise ValueError(f"{name}={v!r}: {e}") from None


def require(name: str, environ: Mapping[str, str] | None = None) -> Any:
    """`get`, for a variable the caller can't do without: a KeyError naming it when it is unset or empty."""
    if not raw(name, environ):
        raise KeyError(name)
    return get(name, environ)


def environ() -> dict[str, str]:
    """A copy of the whole process environment, to pass on or to key on.  Read one variable through its declaration instead."""
    return dict(os.environ)


def __getattr__(name: str) -> Any:
    if name in SETTINGS:
        return get(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*globals(), *SETTINGS])


# ---------------------------------------------------------------------------------------------------------------- roles

def role_of(name: str) -> str | None:
    """A variable's role: its declaration's, else its family's (the longest prefix); None for one the registry doesn't know."""
    if name in SETTINGS:
        return SETTINGS[name].role
    return next((FAMILIES[p] for p in sorted(FAMILIES, key=len, reverse=True) if name.startswith(p)), None)


def is_secret(name: str) -> bool:
    """A credential: declared ``secret``, or its name holds one of `SECRET_WORDS`."""
    s = SETTINGS.get(name)
    return bool(s and s.secret) or any(w in name.upper() for w in SECRET_WORDS)


def recorded(environ: Mapping[str, str] | None = None) -> bool:
    """Whether this is a run of record: a `research run` (`RESEARCH_RUN_ID`), as a recorded check is."""
    return bool(get("RESEARCH_RUN_ID", environ))


def record(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """The settings in force: ``values``, every ``value`` variable set (a credential by name, `REDACTED` for its value), and
    ``digest``, the SHA-512 of their canonical JSON; then ``where``, every declared ``where`` variable set, as telemetry the
    digest leaves out."""
    env = _env(environ)

    def held(k: str) -> str:
        return REDACTED if is_secret(k) else env[k]

    values = {k: held(k) for k in sorted(env) if role_of(k) == VALUE}
    blob = json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return {"schema": RECORD_SCHEMA, "values": values, "digest": hashlib.sha512(blob).hexdigest(),
            "where": {k: held(k) for k in sorted(env) if k in SETTINGS and SETTINGS[k].role == WHERE}}


# ---------------------------------------------------------------------------------------------------------------- refusals

def unknown(environ: Iterable[str] | None = None) -> list[str]:
    """The variables of an `OWNED` prefix that nothing in the tree reads."""
    return sorted(k for k in _env(environ) if k.startswith(OWNED) and k not in SETTINGS and k not in ELSEWHERE)


def refusals(environ: Mapping[str, str] | None = None, of_record: bool | None = None) -> list[str]:
    """Why this environment may not run: each `unknown` variable, and each ``debug`` setting set in a run of record (``of_record``,
    by default `recorded`), named."""
    env = _env(environ)
    out = []
    if bad := unknown(env):
        them = "them" if len(bad) > 1 else "it"
        out.append(f"unknown setting{'s' * (len(bad) > 1)} {', '.join(bad)}: nothing in the tree reads {them}; unset {them}, fix "
                   f"the name, or declare {them} in tools/verity/check/settings.py")
    if recorded(env) if of_record is None else of_record:
        if debug := sorted(k for k in env if env[k] and role_of(k) == DEBUG):
            out.append(f"{', '.join(f'{k}={env[k]}' for k in debug)} in a run of record: a recorded check takes no debug setting "
                       f"(tools/verity/check/settings.py); unset {'them' if len(debug) > 1 else 'it'}")
    return out


if __name__ == "__main__":
    if sys.argv[1:] not in ([], ["--of-record"]):
        sys.exit("usage: settings.py [--of-record]: refuse this environment's unknown check settings (and, --of-record, its "
                 "debug ones), naming each")
    why = refusals(of_record=sys.argv[1:] == ["--of-record"])
    for line in why:
        print(f"settings: {line}", file=sys.stderr)
    sys.exit(2 if why else 0)
