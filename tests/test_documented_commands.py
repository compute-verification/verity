"""Every command line the docs and messages tell an agent to run parses against the CLI it names, so a renamed flag or a
retired subcommand that misses a skill, a README or a refusal fails here instead of in the next agent's shell.

A command line is a `research ...`, `verity ...`, `uv run [python] tools/....py ...` or `python3 tools/....py ...` in a
tracked AGENTS.md, skill or README (a line of a code block, or a backticked span), or in a backticked span of a Python
string (a docstring or a message) outside the tests' own directories, whose strings hold refused commands on purpose. A
code block's line may hold several, each with its description, apart by two or more spaces; a `#` starts a comment. It
is split as a shell would split it (`&&`, `;` and pipes; the command inside `bash -c` and after `research run ... --`),
with what a template leaves open filled in: `$(...)` and `<...>` become a word, a `[...]` group is dropped, `A|B` is A
(a table's `\\|` is a `|`), and a code block's own `NAME="..."` assignments are substituted.

It is parsed, not run: one child process calls the CLI's entry point (`research.cli.main`, `verity_cli.main`, a script's
`__main__`) under an audit hook that refuses every subprocess, exec, socket, and write outside a scratch directory, and
stops once an outermost argparse parse has read every word; an argparse error anywhere fails it. A `python SCRIPT ARGS` or
`python -m MODULE ARGS` the CLI hands off to is followed. A command whose CLI acts before it parses (spawns anything else,
writes, connects), or refuses by its own check after a parse that left words for later, isn't judged.

A mention is not a command line, so these pass: `research run` without `--`; a chain of subcommand words, alone or
before one placeholder (`research tele SUBCOMMAND`), unless the CLI has no such subcommand; outside a code block, a span
that leaves out a required argument or a flag's value (`research status --also`); and an upper-case placeholder (`RUN`,
`NAME`) the CLI's refusal names back. KNOWN lists the command lines that didn't parse when this test landed, each with its owner (the handle whose
`owns` covers the file) and why; it only shrinks, and an entry this tree no longer fails on fails too.
"""
from __future__ import annotations

import ast
import contextlib
import functools
import io
import itertools
import json
import os
import re
import runpy
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: (file, the command as parsed) -> "owner: why", each a documented command line that didn't parse when this test landed
KNOWN = {
    ("benchmarks/commitments/README.md", "verity bench commitments"): "proofs: says it doesn't apply (no run.py or run.sh)",
    ("examples/README.md", "verity demo dither_full_util"): "top: says it runs nothing yet (no runner on main)",
    ("examples/dither_full_util/README.md", "verity demo dither_full_util"): "compute-accounting: says it runs nothing yet",
    ("tools/research/README.md", "research keeper set NAME --on NODE"):
        "infra: names set, stop and status at once, and `set` takes its command after `--`",
    ("tools/research/src/research/slack.py", "research msg router"): "comms: a subcommand to come, named as one",
    ("verity/core/protocols/remote/README.md", "verity verify proof"): "soundness: a command to come, named as one",
    ("verity/core/protocols/remote/README.md", "verity verify remote"): "soundness: a command to come, named as one",
}

SPAN = re.compile(r"`([^`\n]+)`")
FENCE = re.compile(r"^\s*(```|~~~)")
HEAD = re.compile(r"(?:^|[\s\"'(;&|])((?:research|verity) [a-z][\w-]*|uv run |python3? tools/)")
ASSIGN = re.compile(r"^([A-Z][A-Z0-9_]*)=(\"[^\"]*\"|'[^']*'|\S+)\s*$")
PLACEHOLDER = re.compile(r"[A-Z][A-Z0-9_]*")
WORD = re.compile(r"[a-z][a-z0-9-]*")
PYTHON = re.compile(r"python3?(\.\d+)?")
UV_VALUED = {"--extra", "--package", "-p", "--python", "--with", "--group", "--project", "--directory"}


def documents() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z", "AGENTS.md", ".agents/skills/*/SKILL.md", "*README.md", "*.py"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout.split("\0")
    return [f for f in out if f and not re.search(r"(^|/)tests?/", f)]


def _markdown(text: str):
    """(line, text, in a code block) for each code block line and each backticked span outside one."""
    fenced, env, held = False, {}, ""
    for n, line in enumerate(text.splitlines(), 1):
        if FENCE.match(line):
            fenced, env, held = not fenced, {}, ""
        elif fenced:
            line = held + line.strip()
            if line.endswith("\\"):
                held = line[:-1].rstrip() + " "
                continue
            held = ""
            if m := ASSIGN.match(line):
                env[m[1]] = m[2].strip("\"'")
                continue
            for k, v in env.items():
                line = line.replace("${" + k + "}", v).replace("$" + k, v)
            for part in re.split(r"\s{2,}", re.split(r"(?:^|\s)#", line)[0]):
                if HEAD.search(part):
                    yield n, part, True
        else:
            for m in SPAN.finditer(line):
                yield n, m[1].replace("\\|", "|"), False


def _python(text: str):
    """(line, span, False) for each backticked span of a string literal; an f-string's fields read as `X`."""
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            s = node.value
        elif isinstance(node, ast.JoinedStr):
            s = "".join(v.value if isinstance(v, ast.Constant) else "X" for v in node.values)
        else:
            continue
        for m in SPAN.finditer(s):
            yield node.lineno, m[1], False


def _fill(s: str) -> str:
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"\$\([^()]*\)", "X", s)
        s = re.sub(r"\[[^\[\]]*\]", "", s)
    s = re.sub(r"<[^<>]*>", "X", s)
    s = re.sub(r"(?<=[^\s|])\|[^\s|]+", "", s)
    return s.replace("…", " ").replace("...", " ")


def _simple(s: str):
    lex = shlex.shlex(_fill(s), posix=True, punctuation_chars=";&|()")
    lex.whitespace_split, lex.commenters = True, ""
    try:
        words = list(lex)
    except ValueError:
        return "a quote left open: no command line"
    cmd: list[str] = []
    for w in [*words, ";"]:
        if w and set(w) <= set(";&|()"):
            yield from _one(cmd)
            cmd = []
        else:
            cmd.append(w)


def _one(argv: list[str]):
    while argv and (re.fullmatch(r"\w+=.*", argv[0]) or argv[0] in ("env", "exec", "time", "nohup")):
        argv = argv[1:]
    if argv[:1] == ["timeout"]:
        argv = argv[2:]
    if not argv:
        return
    if Path(argv[0]).name in ("bash", "sh") and len(argv) > 2 and argv[1] in ("-c", "-lc", "-ec"):
        yield from _simple(argv[2])
        return
    if argv[:2] == ["uv", "run"]:
        rest, i = argv[2:], 0
        while i < len(rest) and rest[i].startswith("-"):
            i += 2 if rest[i] in UV_VALUED else 1
        argv = rest[i:]
    if argv and PYTHON.fullmatch(argv[0]):
        argv = argv[1:]
    if not argv:
        return
    if argv[0] in ("research", "verity") and len(argv) > 1:
        yield argv
        if argv[1] == "run" and "--" in argv:
            yield from _one(argv[argv.index("--") + 1:])
    elif argv[0].startswith("tools/") and argv[0].endswith(".py") and (ROOT / argv[0]).is_file():
        yield argv


def command_lines() -> list[dict]:
    seen, out = set(), []
    for f in documents():
        text = (ROOT / f).read_text(errors="replace")
        for line, s, block in (_python(text) if f.endswith(".py") else _markdown(text)):
            if HEAD.search(s):
                for argv in _simple(s.strip()):
                    if (key := (f, tuple(argv), block)) not in seen:
                        seen.add(key)
                        out.append({"file": f, "line": line, "argv": argv, "block": block})
    return out


# -- the child: parse each command line on stdin without running it


class _Parsed(BaseException):
    """The outermost parse succeeded; a BaseException, so the CLI's own `except Exception` lets it through, as below."""


class _Acts(BaseException):
    pass


class _HandOff(BaseException):
    pass


class _ArgError(BaseException):
    pass


def _child() -> None:
    import argparse

    sys.dont_write_bytecode = True
    scratch = Path(tempfile.mkdtemp(prefix="documented-commands-")).resolve()
    tempfile.tempdir = str(scratch)
    # A verdict reads the tree alone, the same on a laptop, an agent VM and in a job: none of the operator's variables
    # (store credentials, job variables, secret names), a HOME with no machines file, and an empty notes root, which
    # `research notes` would otherwise clone first.  The suite's read tracer keeps its own.
    keep = ("PATH", "PYTHONPATH", "VIRTUAL_ENV", "LANG", "LC_ALL", "PYTEST_CURRENT_TEST")
    for k in [k for k in os.environ if k not in keep and not k.startswith("VERITY_SUITE_")]:
        del os.environ[k]
    os.environ.update(HOME=str(scratch), TMPDIR=str(scratch))
    (scratch / ".research" / "notes").mkdir(parents=True)
    known = argparse.ArgumentParser.parse_known_args
    state = {"depth": 0, "armed": False, "parsed": False, "read": None, "script": None}

    def outer(self, args, namespace, strict):
        """The outermost parse ends the command line's parse once it has read every word. An outermost parse_known_args
        that leaves some returns, for the CLI may parse them next; what the CLI refuses after that (a machine with no
        project, a deadline past) is no longer the parse, and isn't judged."""
        if state["depth"]:
            return known(self, args, namespace)
        state["depth"] += 1
        try:
            ns, rest = known(self, args, namespace)
        finally:
            state["depth"] -= 1
        if strict and rest:
            self.error("unrecognized arguments: " + " ".join(rest))
        if strict or not rest:
            raise _Parsed
        state["parsed"] = True
        return ns, rest

    def error(self, message):
        raise _ArgError(f"{self.prog}: error: {message}")

    argparse.ArgumentParser.parse_known_args = lambda self, args=None, namespace=None: outer(self, args, namespace, False)
    argparse.ArgumentParser.parse_args = lambda self, args=None, namespace=None: outer(self, args, namespace, True)
    argparse.ArgumentParser.error = error

    def inside(path) -> bool:
        try:
            return Path(os.fsdecode(path)).resolve().is_relative_to(scratch)
        except (TypeError, ValueError):
            return False

    # What reads the same on every machine: the tree, the scratch directory and the Python the child runs on. Anything else
    # (a node's slots file, /sys/fs/cgroup) is the machine's state: a CLI that read some and then refuses without argparse
    # may refuse for the machine's sake, so that refusal is acting, not a failed parse.
    shared = tuple({os.path.join(f(p), "") for p in (str(ROOT), str(scratch), sys.prefix, sys.base_prefix, sys.exec_prefix,
                                                      *filter(None, sys.path)) for f in (os.path.abspath, os.path.realpath)})

    def machine_state(path) -> bool:
        if not isinstance(path, (str, bytes, os.PathLike)):
            return False
        p = os.fsdecode(path)
        return os.path.realpath(p) != state["script"] and not any(
            os.path.join(f(p), "").startswith(shared) for f in (os.path.abspath, os.path.realpath))

    def handoff(argv) -> None:
        argv = [os.fsdecode(a) for a in argv]
        if argv and (PYTHON.fullmatch(Path(argv[0]).name) or argv[0] == sys.executable):
            rest = argv[1:]
            while rest and rest[0] in ("-P", "-u", "-B", "-E", "-I", "-s", "-S", "-O"):
                rest = rest[1:]
            if rest[:1] == ["-m"] and len(rest) > 1 or rest and rest[0].endswith(".py"):
                raise _HandOff(rest)
        raise _Acts(f"starts {' '.join(argv)[:120]}")

    def tracing() -> bool:
        """Whether this event is the suite's read tracer's own (it logs from its audit hook), not the CLI's."""
        f = sys._getframe(1)
        while f is not None:
            if f.f_code.co_filename.endswith("verity_suite_reads.py"):
                return True
            f = f.f_back
        return False

    def hook(event: str, args: tuple) -> None:
        if not state["armed"] or event in ("sys._getframe", "object.__getattr__") or tracing():
            return
        if event == "subprocess.Popen":
            handoff(args[1] if isinstance(args[1], (list, tuple)) else [args[1]])
        elif event in ("os.exec", "os.posix_spawn"):
            handoff(args[1])
        elif event in ("os.system", "os.spawn", "os.fork", "os.forkpty", "socket.connect", "socket.bind", "socket.getaddrinfo",
                       "os.kill", "os.killpg", "os.truncate", "os.symlink", "os.link", "urllib.Request"):
            raise _Acts(event)
        elif event in ("os.mkdir", "os.remove", "os.rename", "os.rmdir", "shutil.rmtree", "os.chmod") and not inside(args[0]):
            raise _Acts(f"{event} {args[0]}")
        elif event == "open" and not inside(args[0]):
            mode, flags = args[1], args[2]
            if (isinstance(mode, str) and set(mode) & set("wax+")) or (isinstance(flags, int) and flags & (
                    os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)):
                raise _Acts(f"writes {args[0]}")
            if machine_state(args[0]) and not state["read"]:
                state["read"] = os.fsdecode(args[0])
        elif event in ("os.listdir", "os.scandir") and args and machine_state(args[0]) and not state["read"]:
            state["read"] = os.fsdecode(args[0])

    sys.addaudithook(hook)
    import research.cli
    import research.remote
    import research.store.local
    import research.store.remote
    import verity_cli

    def acting(owner, name: str, why: str) -> None:
        real = getattr(owner, name)

        def wrapped(*a, **k):
            if state["armed"]:
                raise _Acts(why)
            return real(*a, **k)

        setattr(owner, name, wrapped)

    # Resolving a machine, opening the evidence store or its remote, or finding a program on PATH reads the operator's state,
    # past the parse: `check.py --record --on auto` resolves a machine before it parses, `check.py --agreement-files` signs
    # URLs with the store's remote, and `check.py --preflight` runs uv when there is one.
    acting(research.remote, "load_machines", "reads the machine registry")
    acting(research.store.local.LocalStore, "__init__", "opens the evidence store")
    acting(research.store.remote, "load_remote", "reads the store's remote config")
    acting(shutil, "which", "looks for a program to run")

    def enter(argv: list[str]):
        if argv[0] == "research":
            return research.cli.main(argv[1:])
        if argv[0] == "verity":
            return verity_cli.main(argv[1:])
        if argv[0] == "-m":
            sys.argv = [argv[1], *argv[2:]]
            return runpy.run_module(argv[1], run_name="__main__", alter_sys=True)
        script = str(ROOT / argv[0]) if not os.path.isabs(argv[0]) else argv[0]
        sys.argv, sys.path[0:0], state["script"] = [script, *argv[1:]], [str(Path(script).parent)], os.path.realpath(script)
        return runpy.run_path(script, run_name="__main__")

    def refused(why: str) -> tuple[str, str]:
        """A refusal argparse didn't make: past the parse if one read every word it was given, or the machine's state was read."""
        if state["parsed"]:
            return "acts", why[:400]
        if state["read"]:
            return "acts", f"refuses after reading {state['read']}: {why}"[:400]
        return "fails", why[:400]

    def parse(argv: list[str]) -> tuple[str, str]:
        err, cwd, env, path = io.StringIO(), os.getcwd(), dict(os.environ), list(sys.path)
        state["parsed"], state["read"], state["script"] = False, None, None
        try:
            for _ in range(4):
                try:
                    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                        state["armed"] = True
                        try:
                            rc = enter(argv)
                        finally:
                            state["armed"] = False
                    break
                except _HandOff as h:
                    argv = list(h.args[0])
                    state["parsed"] = False
            else:
                return "acts", "hands off more than 4 times"
        except _Parsed:
            return "ok", ""
        except _ArgError as e:
            return "fails", str(e)[:400]
        except _Acts as e:
            return "acts", str(e)
        except SystemExit as e:
            if e.code in (0, None):
                return "ok", ""
            said = err.getvalue().strip().splitlines()
            return refused(str(e.code) if isinstance(e.code, str) else said[-1] if said else f"exit {e.code}")
        except Exception as e:  # noqa: BLE001  a crash is this command line's verdict, not the child's
            return refused(f"{type(e).__name__}: {e}")
        finally:
            os.chdir(cwd)
            os.environ.clear()
            os.environ.update(env)
            sys.path[:] = path
        if isinstance(rc, int) and rc:
            said = err.getvalue().strip().splitlines()
            return refused(f"exit {rc}: {said[-1] if said else ''}")
        return "ok", ""

    os.chdir(ROOT)
    try:
        for line in sys.stdin:
            item = json.loads(line)
            status, why = parse(item["argv"])
            print(json.dumps({**item, "status": status, "why": why}), flush=True)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


# -- judging


def judge(r: dict) -> str | None:
    """Why a documented command line fails to parse, or None: it parsed, it is a mention, or its CLI acts before parsing."""
    a, why = r["argv"], r["why"]
    if r["status"] != "fails" or a[:2] == ["research", "run"] and "--" not in a:
        return None
    if any(re.search(rf"(?<![\w-]){re.escape(t)}(?![\w-])", why) for t in a[1:] if PLACEHOLDER.fullmatch(t)):
        return None
    rest = list(itertools.dropwhile(WORD.fullmatch, a[1:]))
    if (not rest or rest == [rest[0]] and PLACEHOLDER.fullmatch(rest[0])) and not re.search(
            r"invalid choice|unknown command|no command|no such|has no run", why):
        return None
    if not r["block"] and re.search(r"arguments are required|expected one argument", why):
        return None
    return why


@functools.cache
def results() -> tuple[dict, ...]:
    lines = command_lines()
    p = subprocess.run([sys.executable, __file__, "--child"], cwd=ROOT, input="".join(json.dumps(c) + "\n" for c in lines),
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-2000:]
    out = tuple(json.loads(x) for x in p.stdout.splitlines())
    assert len(out) == len(lines), p.stderr[-2000:]
    return out


def failures() -> dict[tuple[str, str], str]:
    return {(r["file"], " ".join(r["argv"])): why for r in results() if (why := judge(r)) is not None}


@pytest.mark.slow
def test_every_documented_command_line_parses():
    judged = sum(r["status"] != "acts" for r in results())
    assert judged > len(results()) / 2, (f"the child judged only {judged} of {len(results())} command lines: something acts before "
                                         f"every CLI parses (an audit hook's own writes?): {[r['why'] for r in results()][:3]}")
    new = {k: v for k, v in failures().items() if k not in KNOWN}
    assert not new, "documented command lines that don't parse against their CLI (fix the line, or the CLI):\n" + "\n".join(
        f"  {f}: `{cmd}`: {why}" for (f, cmd), why in sorted(new.items()))


@pytest.mark.slow
def test_known_lists_only_lines_that_still_fail():
    stale = sorted(set(KNOWN) - set(failures()))
    assert not stale, "KNOWN entries that parse now or are gone (delete them):\n" + "\n".join(f"  {f}: `{c}`" for f, c in stale)


def test_the_child_parses_without_running():
    """The scan's own check: a good line parses, a bad flag fails, and a CLI that would act is stopped, not run."""
    probe = [{"file": "-", "line": 0, "argv": a, "block": True} for a in (
        ["research", "status", "RUN", "--json"], ["research", "status", "--no-such-flag"], ["research", "doctor", "--json"])]
    p = subprocess.run([sys.executable, __file__, "--child"], cwd=ROOT, input="".join(json.dumps(c) + "\n" for c in probe),
                       capture_output=True, text=True)
    got = [json.loads(x) for x in p.stdout.splitlines()]
    assert [g["status"] for g in got][:2] == ["ok", "fails"] and got[2]["status"] in ("ok", "acts"), (
        [(g["status"], g["why"]) for g in got], p.stderr[-2000:])


def test_a_verdict_reads_the_tree_alone(tmp_path):
    """Lines whose CLI reads the operator's state before or instead of parsing (a machine, the store, a program on PATH, the
    notes clone) get one verdict on every machine: here, and with none of this process's variables and a HOME whose
    machines file lists no machine, as on a node. A script that refuses after reading a file outside the tree (a node's
    slots file) acts; one that refuses having read none fails."""
    (tmp_path / ".research" / "notes" / "machines.d").mkdir(parents=True)
    (tmp_path / "slots").write_text("0 0-15\n")
    (tmp_path / "refuses.py").write_text("import sys\nopen(sys.argv[1]).read()\nsys.exit('none of its slots is within these cores')\n")
    (tmp_path / "bad.py").write_text("import sys\nsys.exit('bad: no such option')\n")
    probe = "".join(json.dumps({"file": "-", "line": 0, "argv": a, "block": True}) + "\n" for a in (
        ["tools/verity/check/check.py", "--record", "--on", "auto", "--pool", "research"],
        ["tools/verity/check/check.py", "--agreement-files"], ["tools/verity/check/check.py", "--preflight"],
        ["research", "notes", "sync", "--path", "kb"], [str(tmp_path / "refuses.py"), str(tmp_path / "slots")],
        [str(tmp_path / "bad.py")]))
    bare = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path),
            **{k: v for k, v in os.environ.items() if k == "PYTHONPATH" or k.startswith("VERITY_SUITE_")}}
    got = [[(r["status"], r["why"]) for r in map(json.loads, subprocess.run(
        [sys.executable, __file__, "--child"], cwd=ROOT, input=probe, capture_output=True, text=True, env=env).stdout.splitlines())]
        for env in (None, bare)]
    statuses = [s for s, _ in got[0]]
    assert statuses == [s for s, _ in got[1]] and "fails" not in statuses[:4] and statuses[4:] == ["acts", "fails"], got


if __name__ == "__main__" and sys.argv[1:] == ["--child"]:
    _child()
