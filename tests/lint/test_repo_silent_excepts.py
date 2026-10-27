"""Except handlers that swallow what they catch, in every tracked Python file outside ``integrations/vllm/`` and ``archive/``,
held to ``allowlists/repo_silent_excepts.json``: S4 of the quality rollout, recorded outcomes (the vLLM coordinator, 27 Oct
02:05Z), so no area adds one while its owner turns the ones it has into raises and records.

A handler swallows when every statement in it is ``pass``, ``continue``, ``break``, a bare constant (``...``), a call to a
function that does nothing, or a ``return`` or an assignment of a default: nothing, ``None``, a constant other than a string or
``False``, a name other than the caught exception's, or an empty container.  A function does nothing when it is the file's one
binding of its name, an undecorated module-level ``def`` whose body only passes, states or returns a constant, or calls
another such function, or a ``lambda`` of a constant; a call to it does nothing when no call is in its arguments.  Such a
handler neither raises, nor logs, nor records what it caught, nor calls anything that does, so the code goes on with less
than it had and leaves no trace of why; a guard never fails open (AGENTS.md).  ``contextlib.suppress(E, ...)``, however the
file imports it, is ``except E: pass`` written as a context manager, so each call counts one too.  Broad and typed handlers
count alike, and a file that doesn't parse counts one, since its handlers can't be seen.  A string says why (``return
"unparseable"``), ``return False`` is a predicate's or a verifier's answer, failing closed, and a handler that names what it
caught (``return e``) hands it on, so none of those swallows.  This is the integration's rule
(``integrations/vllm/tests/lint/test_silent_excepts.py``, whose README section "Recorded outcomes" says why), which holds two
of its packages per handler; here every file is held, per file.  When the rule widens (``suppress`` and calls that do nothing:
27 Oct, VLLM-134), the list takes what it newly finds in files the change leaves as they were, and only shrinks after.

Handlers are counted per file, the key the ratchet gate counts by too: an entry ``{"file", "count"}`` allows ``count``
swallowing handlers in its file.  A file with more than its entry allows, or with one and no entry, fails.  An entry that
allows more than is found doesn't: the list can run ahead of a tree, holding the handlers of PRs queued below it on ``next``,
and ``--prune`` deletes and lowers what this tree no longer finds.  A file that moves takes its entry with it, in the same
change (``--move``): this lint reads no history, so a renamed file's handlers count under its new path.  The ratchet gate
(``integrations/vllm/tools/ratchet_gate.py``) refuses a change whose Python gains a swallowing handler, or that grows this list
or leaves a renamed file's entry behind, and judges a change to this module by running it as the change's base has it on the
change's tree.  So it imports only the standard library, ``test_no_new_silent_excepts`` takes no fixture, and the gate reads
``LEFT_OUT``, ``FIX``, ``handlers`` and ``parse`` from the base's copy of this module.

    python3 tests/lint/test_repo_silent_excepts.py --move OLD NEW   # a file (or directory) moved: its entry follows it
    python3 tests/lint/test_repo_silent_excepts.py --prune          # delete and lower what this tree no longer finds
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

NAME = "repo_silent_excepts"
ROOT = Path(__file__).resolve().parents[2]
LIST = Path(__file__).resolve().parent / "allowlists" / f"{NAME}.json"
#: the integration's own lint holds it, and the archive isn't developed
LEFT_OUT = ("integrations/vllm/", "archive/")
WHY = "integrations/vllm/tests/lint/README.md (Recorded outcomes)"
FIX = ("Re-raise; or record the outcome in your area's typed record or log, saying what was caught and what the code did "
       "instead; or narrow it away: test for the case you expect (`path.exists()`, `d.get(k)`) and let the rest raise, since "
       "a narrower handler that still swallows still counts, and so does `contextlib.suppress(E)`, which is `except E: pass` "
       f"(why: {WHY}). If the file moved, move its entry in the same "
       f"change: `python3 tests/lint/test_{NAME}.py --move OLD NEW`.")
EMPTY_CALLS = frozenset({"dict", "list", "set", "tuple", "frozenset", "Counter"})
BODIES = ("body", "orelse", "finalbody", "handlers", "cases")


def idle_call(v: ast.expr, noops: frozenset[str]) -> bool:
    """Whether ``v`` calls one of ``noops`` with no call in its arguments, so that nothing happens."""
    return (isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id in noops
            and not any(isinstance(n, ast.Call) for a in [*v.args, *v.keywords] for n in ast.walk(a)))


def idle(s: ast.stmt, noops: frozenset[str]) -> bool:
    """Whether ``s`` does nothing: ``pass``, a constant (a docstring), a ``return`` of nothing or of a constant, or an idle
    call to one of ``noops``."""
    if isinstance(s, ast.Expr):
        return isinstance(s.value, ast.Constant) or idle_call(s.value, noops)
    return isinstance(s, ast.Pass) or isinstance(s, ast.Return) and (s.value is None or isinstance(s.value, ast.Constant))


def noops(tree: ast.Module) -> frozenset[str]:
    """The module's functions that do nothing: each undecorated module-level ``def`` whose every statement is idle, calls to
    another such function included, and each module-level ``NAME = lambda ...: CONSTANT``.  A name the file binds more than
    once is none, since which binding a call reaches isn't plain."""
    defs = {s.name: s.body for s in tree.body if isinstance(s, ast.FunctionDef) and not s.decorator_list}
    defs |= {s.targets[0].id: [ast.Return(s.value.body)] for s in tree.body
             if isinstance(s, ast.Assign) and isinstance(s.value, ast.Lambda) and [type(t) for t in s.targets] == [ast.Name]}
    if not (defs := {f: body for f, body in defs.items() if all(idle(s, frozenset(defs)) for s in body)}):
        return frozenset()
    bound = Counter(n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store))
    bound.update(n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))
    bound.update((a.asname or a.name).split(".")[0] for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
                 for a in n.names)
    out: frozenset[str] = frozenset()
    while more := {f for f, body in defs.items() if f not in out and bound[f] == 1 and all(idle(s, out) for s in body)}:
        out |= more
    return out


def suppressions(tree: ast.Module) -> list[ast.Call]:
    """Each ``contextlib.suppress(E, ...)`` call in ``tree``, however the file imports it: an ``except E: pass`` that reads as
    a context manager."""
    names, modules = set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module == "contextlib":
            names |= {a.asname or a.name for a in n.names if a.name == "suppress"}
        elif isinstance(n, ast.Import):
            modules |= {a.asname or a.name for a in n.names if a.name == "contextlib"}
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call) and n.args and (
        isinstance(n.func, ast.Name) and n.func.id in names
        or isinstance(n.func, ast.Attribute) and n.func.attr == "suppress" and isinstance(n.func.value, ast.Name)
        and n.func.value.id in modules)] if names or modules else []


def default(v: ast.expr | None, caught: str | None) -> bool:
    """Whether ``v`` is a default that says nothing about what was caught: nothing, ``None``, a non-string constant, a name
    other than the caught exception's, or an empty container."""
    if v is None:
        return True
    if isinstance(v, ast.Name):
        return v.id != caught
    if isinstance(v, ast.UnaryOp) and isinstance(v.operand, ast.Constant):
        v = v.operand
    if isinstance(v, ast.Constant):
        return not isinstance(v.value, str)
    if isinstance(v, (ast.List, ast.Tuple, ast.Set)):
        return not v.elts
    if isinstance(v, ast.Dict):
        return not v.keys
    return isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id in EMPTY_CALLS and not v.args and not v.keywords


def swallows(h: ast.ExceptHandler, noops: frozenset[str] = frozenset()) -> bool:
    def quiet(s: ast.stmt) -> bool:
        if isinstance(s, (ast.Pass, ast.Continue, ast.Break)):
            return True
        if isinstance(s, ast.Expr):
            return isinstance(s.value, ast.Constant) or idle_call(s.value, noops)
        if isinstance(s, ast.Return):
            return not (isinstance(s.value, ast.Constant) and s.value.value is False) and default(s.value, h.name)
        if isinstance(s, (ast.Assign, ast.AnnAssign)):
            return default(s.value, h.name)
        return False

    return all(quiet(s) for s in h.body)


def handlers(source: str) -> list[tuple[int, str]]:
    """(line, what) of each handler in ``source`` that swallows what it catches: ``what`` is the handler, as ``except OSError:
    continue``, or the call, as ``contextlib.suppress(OSError)``.  A file that doesn't parse counts one, at its error, since
    its handlers can't be seen."""
    if "except" not in source and "suppress" not in source:
        return []
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError) as e:
        return [(getattr(e, "lineno", None) or 0, f"unparseable: {getattr(e, 'msg', e)}"[:120])]
    out = [(c.lineno, ast.unparse(c)[:120]) for c in suppressions(tree)] if "suppress" in source else []
    todo, idle = list(tree.body), noops(tree)
    while todo:
        s = todo.pop()
        if isinstance(s, ast.ExceptHandler) and swallows(s, idle):
            caught = f"except {ast.unparse(s.type)}" if s.type is not None else "except"
            out.append((s.lineno, f"{caught}: {'; '.join(ast.unparse(b) for b in s.body)}"[:120]))
        for f in BODIES:
            todo += getattr(s, f, None) or []
    return sorted(out)


def parse(doc: object) -> Counter:
    """Each file's allowed swallowing handlers, from a list's document."""
    assert isinstance(doc, dict) and isinstance(doc.get("entries"), list), f"{LIST.name}: {{'entries': [...]}} expected"
    counts: Counter = Counter()
    for e in doc["entries"]:
        assert isinstance(e, dict) and set(e) == {"file", "count"} and isinstance(e["file"], str), \
            f"{LIST.name}: an entry is {{file, count}}: {e}"
        assert type(e["count"]) is int and e["count"] >= 1, f"{LIST.name}: count must be a positive int: {e}"
        assert e["file"] not in counts, f"{LIST.name}: duplicate entry {e}"
        counts[e["file"]] = e["count"]
    return counts


def files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z", "--", "*.py"], cwd=ROOT, capture_output=True, check=True).stdout
    return sorted(p for p in out.decode().split("\0") if p and not p.startswith(LEFT_OUT))


@lru_cache(maxsize=None)
def scan() -> tuple[tuple[str, int, str], ...]:
    """Each swallowing handler's file, line and text."""
    return tuple((f, line, what) for f in files() if (ROOT / f).is_file()
                 for line, what in handlers((ROOT / f).read_text(encoding="utf-8", errors="replace")))


def allowed() -> Counter:
    return parse(json.loads(LIST.read_text(encoding="utf-8")))


def found() -> Counter:
    return Counter(f for f, *_ in scan())


def test_no_new_silent_excepts() -> None:
    have = allowed()
    sites: dict[str, list[tuple[int, str]]] = {}
    for f, line, what in scan():
        sites.setdefault(f, []).append((line, what))
    new = [f"  {f}: {n} found, {have[f]} allowed\n" + "\n".join(f"    {f}:{line} {what}" for line, what in sites[f])
           for f, n in sorted(found().items()) if n > have[f]]
    assert not new, (f"silent except handlers: {len(new)} file(s) swallow more exceptions than tests/lint/allowlists/{NAME}.json "
                     f"allows. {FIX}\n" + "\n".join(new))


def test_what_swallows_and_what_does_not() -> None:
    src = (
        "def f(paths, log, out):\n"
        "    for p in paths:\n"
        "        try:\n"
        "            out.append(open(p).read())\n"
        "        except OSError:\n"
        "            continue\n"
        "        except (KeyError, ValueError) as e:\n"
        "            cfg = {}\n"
        "        except Exception:\n"
        "            pass\n"
        "    try:\n"
        "        return out[0]\n"
        "    except IndexError:\n"
        "        return -1\n"
        "    except LookupError:\n"
        "        return True\n"
        "    except TypeError:\n"
        "        '''why'''\n"
        "    except:\n"
        "        x: int = None\n"
        "class C:\n"
        "    def g(self):\n"
        "        with x:\n"
        "            try:\n"
        "                pass\n"
        "            except* ValueError:\n"
        "                ...\n"
        "            else:\n"
        "                try:\n"
        "                    pass\n"
        "                except AttributeError:\n"
        "                    return\n"
        "def h(log, unreadable, p):\n"
        "    try:\n"
        "        return verify(p)\n"
        "    except ValueError:\n"
        "        return False\n"
        "    except KeyError:\n"
        "        return 'unparseable'\n"
        "    except TypeError:\n"
        "        raise\n"
        "    except OSError as e:\n"
        "        return e\n"
        "    except LookupError as e:\n"
        "        unreadable.append(f'{p}: {e}')\n"
        "    except RuntimeError:\n"
        "        log.warning('no %s', p)\n"
        "    except ArithmeticError:\n"
        "        skipped += 1\n"
        "    except Exception as e:\n"
        "        err = e\n"
        "    except BaseException:\n"
        "        pass\n"
        "        log.exception('x')\n"
    )
    assert handlers(src) == [
        (5, "except OSError: continue"), (7, "except (KeyError, ValueError): cfg = {}"), (9, "except Exception: pass"),
        (13, "except IndexError: return -1"), (15, "except LookupError: return True"), (17, "except TypeError: 'why'"),
        (19, "except: x: int = None"), (26, "except ValueError: ..."), (31, "except AttributeError: return")]
    assert handlers("try:\n    pass\nexcept (\n") == [(3, "unparseable: '(' was never closed")]
    assert handlers("X = (\n") == [] and handlers("X = 1\n") == []


def test_a_suppress_call_and_a_handler_that_only_calls_a_function_that_does_nothing_swallow() -> None:
    src = (
        "import contextlib\n"
        "import contextlib as cl\n"
        "from contextlib import suppress, suppress as quietly, nullcontext\n"
        "def nothing(*a):\n"
        "    '''Does nothing.'''\n"
        "def also(e):\n"
        "    nothing(e)\n"
        "    return None\n"
        "ignore = lambda *a: None\n"
        "@register\n"
        "def kept(e):\n"
        "    pass\n"
        "def twice(e):\n"
        "    pass\n"
        "twice = print\n"
        "def f(self, p, log):\n"
        "    with contextlib.suppress(OSError), cl.suppress(KeyError):\n"
        "        p.unlink()\n"
        "    with suppress(ValueError), quietly(TypeError), nullcontext(), contextlib.suppress():\n"
        "        p.chmod(0)\n"
        "    try:\n"
        "        p.touch()\n"
        "    except OSError as e:\n"
        "        nothing(e)\n"
        "    except ValueError as e:\n"
        "        also(e)\n"
        "    except KeyError:\n"
        "        ignore()\n"
        "    except TypeError as e:\n"
        "        kept(e)\n"
        "    except LookupError as e:\n"
        "        twice(e)\n"
        "    except RuntimeError as e:\n"
        "        nothing(log.warning('%s', e))\n"
        "    except Exception as e:\n"
        "        self.nothing(e)\n"
    )
    assert handlers(src) == [
        (17, "cl.suppress(KeyError)"), (17, "contextlib.suppress(OSError)"), (19, "quietly(TypeError)"),
        (19, "suppress(ValueError)"), (23, "except OSError: nothing(e)"), (25, "except ValueError: also(e)"),
        (27, "except KeyError: ignore()")], handlers(src)
    assert handlers("from mylib import suppress\nwith suppress(OSError):\n    pass\n") == []


def test_the_list_is_well_formed_and_an_entry_moves_with_its_file() -> None:
    assert parse({"entries": [{"file": "a.py", "count": 2}]}) == Counter({"a.py": 2})
    for bad in ({"file": "a.py", "count": 0}, {"file": "a.py", "count": True}, {"file": "a.py", "registry": True},
                {"file": "a.py", "count": 1, "rule": "x"}):
        try:
            parse({"entries": [bad]})
        except AssertionError as e:
            assert str(e).startswith(f"{LIST.name}: "), e
        else:
            raise AssertionError(f"{bad} parsed")
    assert moved(Counter({"a/x.py": 2, "a/y.py": 1, "b.py": 1}), "a", "c") == Counter({"c/x.py": 2, "c/y.py": 1, "b.py": 1})
    assert moved(Counter({"a.py": 2, "b.py": 1}), "a.py", "b.py") == Counter({"b.py": 3})


def moved(counts: Counter, old: str, new: str) -> Counter:
    """``counts`` with the file ``old``'s, or the directory ``old``'s files', under ``new``: a count already there adds."""
    old, new = old.rstrip("/"), new.rstrip("/")
    out: Counter = Counter()
    for f, n in counts.items():
        out[new + f[len(old):] if f == old or f.startswith(f"{old}/") else f] += n
    return out


def dump(counts: Counter) -> None:
    lines = [f"    {json.dumps({'file': f, 'count': n}, ensure_ascii=False)}" for f, n in sorted(counts.items()) if n > 0]
    LIST.write_text(f'{{\n  "rule": "silent except handlers (tests/lint/test_{NAME}.py)",\n  "entries": [\n'
                    + ",\n".join(lines) + "\n  ]\n}\n", encoding="utf-8")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--move"] and len(sys.argv) == 4:
        have = allowed()
        after = moved(have, *sys.argv[2:])
        dump(after)
        print(f"{sum(1 for f in have if f not in after)} entries moved from {sys.argv[2]} to {sys.argv[3]}")
    elif sys.argv[1:] == ["--prune"]:
        dump(allowed() & found())
    else:
        sys.exit(__doc__.rsplit("\n\n", 1)[-1])
