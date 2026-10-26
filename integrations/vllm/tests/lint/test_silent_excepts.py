"""Recorded outcomes: in ``verity_vllm/check/`` and ``verity_vllm/commit/`` a caught exception is recorded, never dropped.

These packages decide verdicts and write the record, and a guard never fails open (AGENTS.md).  A handler that skips a file it
can't read, or puts a default in place of a value it couldn't get, shrinks the evidence a verdict is decided on and leaves no
trace of it.  So a handler there either raises, or records what it caught: an ``error`` result or a reason the verdict reads
(``check/gates.py``'s ``_gate``: "a gate never raises; a bug is an `error` result"), or an entry in the record such as
``unreadable: <path>: <error>`` that the comparison sees.

Checked against ``allowlists/silent_excepts.json``, keyed ``(file, kind, symbol, detail)``:

* ``silent-except``: an ``except`` handler whose every statement is ``pass``, ``continue``, ``break``, a bare constant, or a
  ``return`` or assignment of ``None``, a non-string constant, a name or an empty container.  Such a handler neither raises,
  nor writes a reason, nor calls anything.  ``return False`` is a predicate's or verifier's answer, failing closed, and
  counts as recorded; ``return True`` doesn't.  The detail is the handler, as ``except OSError: continue``.

Most entries are the handlers that are fine as they are: a probe of a Record's shape whose ``None`` the caller handles, a
parse helper's default, a resource or cleanup call.  P7's ``broad-except`` (``test_p07_declared_inputs.py``) checks a
broad handler everywhere; this one checks every handler in these two packages, narrow ones included.
"""
from __future__ import annotations

import ast
from functools import lru_cache

from . import _ratchet as R

NAME = "silent_excepts"
RULE = "Recorded outcomes (in check/ and commit/, a caught exception is recorded, never dropped)"
PACKAGES = ("verity_vllm/check/", "verity_vllm/commit/")
_EMPTY_CALLS = frozenset({"dict", "list", "set", "tuple", "frozenset", "Counter"})


def _trivial(v: ast.expr | None) -> bool:
    """``None``, a non-string constant, a name or an empty container: a value that says nothing about what was caught."""
    if v is None or isinstance(v, ast.Name):
        return True
    if isinstance(v, ast.Constant):
        return not isinstance(v.value, str)
    if isinstance(v, (ast.List, ast.Tuple, ast.Set)):
        return not v.elts
    if isinstance(v, ast.Dict):
        return not v.keys
    return isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id in _EMPTY_CALLS and not v.args and not v.keywords


def _silent_statement(s: ast.stmt) -> bool:
    if isinstance(s, (ast.Pass, ast.Continue, ast.Break)):
        return True
    if isinstance(s, ast.Expr):
        return isinstance(s.value, ast.Constant)
    if isinstance(s, ast.Return):
        return not (isinstance(s.value, ast.Constant) and s.value.value is False) and _trivial(s.value)
    if isinstance(s, (ast.Assign, ast.AnnAssign)):
        return _trivial(s.value)
    return False


def silent(h: ast.ExceptHandler) -> bool:
    return all(_silent_statement(s) for s in h.body)


def _detail(h: ast.ExceptHandler) -> str:
    caught = ast.unparse(h.type) if h.type is not None else ""
    return (f"except {caught}".rstrip() + ": " + "; ".join(ast.unparse(s) for s in h.body))[:120]


def file_violations(src: str, rel: str) -> list[R.Violation]:
    return [R.Violation(rel, "silent-except", symbol, _detail(node), node.lineno)
            for node, symbol in R.walk(R.parse_src(src)) if isinstance(node, ast.ExceptHandler) and silent(node)]


@lru_cache(maxsize=None)
def scan() -> tuple[R.Violation, ...]:
    out: list[R.Violation] = []
    for p in R.library_files():
        if R.rel(p).startswith(PACKAGES):
            out += file_violations(R.source(p), R.rel(p))
    return tuple(out)


def test_no_new_violations() -> None:
    R.assert_no_new(NAME, scan(), RULE,
                    "Record what the handler caught: an `error` result or a reason the verdict reads, or an entry in the "
                    "record (`unreadable: <path>: <error>`) that the comparison sees, so a verdict can't pass on less evidence "
                    "than it names; or catch only the exception you expect and let the rest raise.")


def test_allowlist_has_no_stale_entries() -> None:
    R.assert_no_stale(NAME, scan(), RULE)


def test_detectors_on_examples() -> None:
    bad = (
        "import json\n"
        "def scan(paths):\n"
        "    out, cfg = [], None\n"
        "    for p in paths:\n"
        "        try:\n"
        "            out.append(open(p).read())\n"
        "        except OSError:\n"
        "            continue\n"
        "    try:\n"
        "        cfg = json.loads(out[0])\n"
        "    except Exception:\n"
        "        cfg = {}\n"
        "    try:\n"
        "        return cfg['n']\n"
        "    except KeyError:\n"
        "        return True\n"
        "class C:\n"
        "    def f(self):\n"
        "        try:\n"
        "            pass\n"
        "        except:\n"
        "            ...\n"
    )
    got = [(v.symbol, v.detail) for v in file_violations(bad, "verity_vllm/check/x.py")]
    assert got == [("scan", "except OSError: continue"), ("scan", "except Exception: cfg = {}"),
                   ("scan", "except KeyError: return True"), ("C.f", "except: ...")], got
    good = (
        "def scan(paths, unreadable):\n"
        "    for p in paths:\n"
        "        try:\n"
        "            open(p).read()\n"
        "        except OSError as e:\n"
        "            unreadable.append({'path': p, 'error': f'{type(e).__name__}: {e}'})\n"
        "            continue\n"
        "    try:\n"
        "        return verify(paths)\n"
        "    except ValueError:\n"
        "        return False\n"
        "    except KeyError:\n"
        "        return 'unparseable'\n"
        "    except TypeError:\n"
        "        raise\n"
        "    except LookupError:\n"
        "        skipped += 1\n"
    )
    assert file_violations(good, "verity_vllm/check/x.py") == []


def test_both_packages_are_scanned() -> None:
    files = {R.rel(p) for p in R.library_files()}
    missing = [pkg for pkg in PACKAGES if not any(f.startswith(pkg) for f in files)]
    assert not missing, f"{missing} hold no library file: point PACKAGES at where they moved, or the lint checks nothing"
