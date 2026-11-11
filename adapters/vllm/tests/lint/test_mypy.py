"""mypy's errors on the integration, held to ``allowlists/mypy.json``: mypy under vLLM's settings (``[tool.mypy]`` in
``adapters/vllm/pyproject.toml``), run from ``adapters/vllm`` with the mypy the root ``mypy`` dependency group pins,
in two groups as vLLM's ``tools/pre_commit/mypy.py`` runs them: the library at its configured ``follow_imports = "silent"``,
and ``tests/`` on its own at ``skip``.

An error is keyed ``(file, kind, symbol, "")``: the file relative to ``adapters/vllm``, mypy's error code and the
enclosing ``Class.function`` (``<module>`` at top level), with a count, so lines that move and types that change elsewhere
don't churn the list; a failure prints mypy's messages.  ``CLEAN`` names the directories with no error, held to none: fixing
a directory adds it there.

mypy's findings depend on what it can import: CPU torch with vLLM absent makes every vLLM type ``Any``, and another torch or
mypy finds other things.  So the list records the environment it was taken in (``environment``), and the lint runs only in
that one, the suite's own, refusing any other rather than miscounting.  ``python -m tests.lint.test_mypy --record`` (from
``adapters/vllm``) writes the list from what this environment finds, as after a pin moves; the ratchet gate holds it to
no growth.  The gate counts the list but can't rerun this lint, whose findings need the suite's environment, so it has no
``NAME``: the suite runs it, both groups at once, which no cheaper test can stand in for.

In the suite, an error the list doesn't hold and an entry that matches nothing warn (``Bookkeeping``) and a later PR fixes
them: a type error makes no claim false, and failing the suite would fail, and revert, each in-flight PR that adds one.  An
error in a ``CLEAN`` directory, and a list taken in another environment, fail.
"""
from __future__ import annotations

import json
import subprocess
import sys
import warnings
from collections import Counter, defaultdict
from functools import cache
from importlib import metadata

import pytest

from . import _ratchet as R
from . import merge_allowlist as MA
from . import test_ruff

LIST = "mypy"
RULE = "mypy with vLLM's settings (adapters/vllm/pyproject.toml's [tool.mypy])"
#: (the paths of a group, from adapters/vllm; its ``follow_imports``, None: the configured one)
GROUPS = ((("verity_vllm",), None), (("tests",), "skip"))
CLEAN = (
    "tests/ops",
    "tests/protocols/linear",
    "tests/records",
    "verity_vllm/acquire/norm_tap_src",
    "verity_vllm/acquire/router_tap_src",
    "verity_vllm/program/quarantine",
    "verity_vllm/properties/difftests",
)
#: the distributions whose versions change what mypy finds
PACKAGES = ("mypy", "pydantic", "torch", "vllm")


class Bookkeeping(UserWarning):
    """What the list doesn't match: a later PR fixes it."""


def environment() -> dict[str, str | None]:
    def version(name: str) -> str | None:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None
    return {"python": f"{sys.version_info.major}.{sys.version_info.minor}", **{p: version(p) for p in PACKAGES}}


def recorded() -> dict:
    return json.loads((R.ALLOWLISTS / f"{LIST}.json").read_text(encoding="utf-8"))


def assert_recorded_environment() -> None:
    was, now = recorded().get("environment"), environment()
    assert now["mypy"], "no mypy here: `uv sync --all-packages --extra torch-cpu` installs the one the root `mypy` group pins"
    assert was == now, (f"allowlists/{LIST}.json was recorded under {was}, and this is {now}: run the suite in its own "
                        "environment (`uv sync --all-packages --extra torch-cpu`), or, where a pin moved, re-record the list "
                        "with `python -m tests.lint.test_mypy --record` from adapters/vllm")


def errors() -> list[dict]:
    """mypy's errors in both groups, run at once, each a record of mypy's JSON output."""
    procs = []
    for paths, follow in GROUPS:
        cmd = [sys.executable, "-m", "mypy", "--config-file", "pyproject.toml", "--cache-dir", "/dev/null", "-O", "json",
               *(["--follow-imports", follow] if follow else []), *paths]
        procs.append((cmd, subprocess.Popen(cmd, cwd=R.ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)))
    out = []
    for cmd, p in procs:
        stdout, stderr = p.communicate()
        assert p.returncode in (0, 1), f"{' '.join(cmd[1:])} exited {p.returncode}:\n{(stderr or stdout)[-2000:]}"
        out += [r for r in map(json.loads, stdout.splitlines()) if r["severity"] == "error"]
    return out


def violation(r: dict) -> R.Violation:
    return R.Violation(r["file"], r["code"] or "misc", test_ruff.enclosing(R.parse(R.ROOT / r["file"]), r["line"]), "",
                       r["line"])


@cache
def scan() -> tuple[tuple[R.Violation, str], ...]:
    """Each error as a violation, with mypy's message."""
    assert_recorded_environment()
    return tuple((violation(r), r["message"]) for r in errors())


def violations() -> list[R.Violation]:
    return [v for v, _ in scan()]


def test_the_list_was_recorded_in_this_environment() -> None:
    assert_recorded_environment()


def no_new_errors() -> None:
    allowed, found = R.load_allowlist(LIST), Counter(v.key for v in violations())
    by_key = defaultdict(list)
    for v, message in scan():
        by_key[v.key].append(f"      {v.file}:{v.line}: {message}")
    new = [f"  {R.entry_line(k, n)}  ({allowed.get(k, 0)} allowed)\n" + "\n".join(sorted(by_key[k]))
           for k, n in sorted(found.items()) if n > allowed.get(k, 0)]
    assert not new, (f"{RULE}: {len(new)} key(s) found more often than allowlists/{LIST}.json allows. Fix the code (if an "
                     "error only moved, move its entry in the same change).\n" + "\n".join(new))


def no_stale_entries() -> None:
    R.assert_no_stale(LIST, violations(), RULE)


@pytest.mark.slow
@pytest.mark.parametrize("check", [no_new_errors, no_stale_entries], ids=["new", "stale"])
def test_the_list_matches_or_warns(check) -> None:
    try:
        check()
    except AssertionError as e:
        warnings.warn(Bookkeeping(str(e)), stacklevel=1)


@pytest.mark.slow
def test_clean_directories_have_no_error() -> None:
    missing = [d for d in CLEAN if not (R.ROOT / d).is_dir()]
    assert not missing, f"CLEAN names directories that aren't there: {missing}"
    dirty = sorted(f"{v.file}:{v.line}: {m}" for v, m in scan() if v.file.startswith(tuple(f"{d}/" for d in CLEAN)))
    assert not dirty, "mypy errors in a directory CLEAN holds to none:\n" + "\n".join(dirty)


def test_the_groups_cover_the_package_and_its_tests() -> None:
    assert sorted(p for paths, _ in GROUPS for p in paths) == ["tests", "verity_vllm"]
    assert all(d.startswith(("tests/", "verity_vllm/")) for d in CLEAN) and list(CLEAN) == sorted(set(CLEAN))


def main(argv: list[str]) -> int:
    if argv != ["--record"]:
        print(f"usage: python -m tests.lint.test_mypy --record   (from adapters/vllm)\n\n{__doc__}", file=sys.stderr)
        return 2
    found = Counter(violation(r).key for r in errors())
    doc = {"rule": RULE, "environment": environment(),
           "entries": [json.loads(R.entry_line(k, n)) for k, n in sorted(found.items())]}
    (R.ALLOWLISTS / f"{LIST}.json").write_text(MA.dump(doc), encoding="utf-8")
    print(f"allowlists/{LIST}.json: {len(doc['entries'])} entries, {sum(found.values())} errors")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
