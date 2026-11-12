"""The suite's set-up: where a test gets a benchmark's script from, and the fixtures more than one test file uses.

A directory under ``benchmarks/pouw/`` is scripts, not a package, and a test imports a script as a module by its file's
name. This file puts every script directory on ``sys.path``, once, so a test neither edits ``sys.path`` nor loads a file
itself.

A name one directory holds is imported by that name. A name two directories hold is imported by name from one of them at
most, the one whose own scripts import it so: `AHEAD` lists those directories and puts them first, and `FirstHolder`, a
finder ahead of ``sys.path``, keeps the name theirs whoever imports it, whatever a script has put first on the path since
(a launcher starts a script with its own directory first, and many scripts put it there themselves). Another holder's
module comes from `load`, under a name of its own. ``test_script_imports.py`` holds the tree to this.
"""
from __future__ import annotations

import asyncio
import importlib.abc
import importlib.util
import sys
import threading
from collections import defaultdict
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType

import pytest
from verity.protocols.certifier.verifier import active

POUW = Path(__file__).resolve().parents[1]
#: what under ``benchmarks/pouw/`` this suite takes no script from: the tests, the package the environment installs,
#: and ``edited_gemm``, a suite of its own whose files this suite's inputs do not hold
NOT_SCRIPTS = frozenset({"tests", "exact_kit", "edited_gemm"})
#: the directories whose own scripts import by name a module whose name another directory holds too (``harness``:
#: derive and report; ``h1_lanes``: run)
AHEAD = ("harness", "h1_lanes")


def scripts() -> list[Path]:
    """Every script under ``benchmarks/pouw/``."""
    return sorted(p for p in POUW.rglob("*.py") if not NOT_SCRIPTS.intersection(p.relative_to(POUW).parts[:-1]))


def script_dirs() -> list[Path]:
    """The directories that hold a script, in their order on ``sys.path``: `AHEAD`'s, then the rest by name."""
    ahead = [POUW / d for d in AHEAD]
    return [*ahead, *sorted({p.parent for p in scripts()}.difference(ahead))]


def holders() -> dict[str, list[Path]]:
    """Each script's module name with the directories that hold it, in the path's order."""
    held, order = defaultdict(set), script_dirs()
    for p in scripts():
        held[p.stem].add(p.parent)
    return {name: [d for d in order if d in dirs] for name, dirs in held.items()}


class FirstHolder(importlib.abc.MetaPathFinder):
    """``import NAME``, for a name two directories hold: its first holder's file, found before ``sys.path`` is read."""

    def __init__(self) -> None:
        self.files = {name: dirs[0] / f"{name}.py" for name, dirs in holders().items() if len(dirs) > 1}

    def find_spec(self, name: str, path: object = None, target: object = None) -> ModuleSpec | None:
        if path is None and name in self.files:
            return importlib.util.spec_from_file_location(name, self.files[name])
        return None


def load_name(file: str | Path) -> str:
    """The module name `load` gives ``benchmarks/pouw/FILE``: its directory and its file (``gpu_kit/sass_gate.py`` is
    ``gpu_kit_sass_gate``), and ``pouw_NAME`` for a file directly under ``benchmarks/pouw/``."""
    parts = Path(file).with_suffix("").parts
    return "_".join(parts if len(parts) > 1 else ("pouw", *parts))


def load(file: str) -> ModuleType:
    """``benchmarks/pouw/FILE`` as a module named `load_name` of it, run once however many tests ask. For a module whose
    name another directory holds and `AHEAD` gives to that one, and for one its own directory's scripts load from its
    file under this name."""
    name = load_name(file)
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, POUW / file)
        sys.modules[name] = module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[name]
            raise
    return sys.modules[name]


sys.path[:0] = map(str, script_dirs())
sys.meta_path.insert(0, FirstHolder())


@pytest.fixture(autouse=True)
def _outside_any_run(monkeypatch):
    """A benchmark's `run_result.write` writes into ``$RESEARCH_RUN_DIR``: under a `research run` of the suites that is the
    run's own result.json, which a test's verdict must never replace."""
    monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)


@pytest.fixture
def reveals():
    """The network certifier's reveal endpoint for the test's proxies: `reveals(proxy)` serves it (`serve_reveals`) on
    localhost, on a loop of its own thread, and returns its (host, port), so a synchronous challenger asks it through a
    `CertifierClient`. Every server closes, and the loop stops, when the test ends."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    servers: list[asyncio.Server] = []

    def serve(proxy: active.Proxy) -> tuple[str, int]:
        servers.append(asyncio.run_coroutine_threadsafe(active.serve_reveals(proxy, "127.0.0.1", 0), loop).result())
        return servers[-1].sockets[0].getsockname()[:2]

    async def close() -> None:
        for s in servers:
            s.close()
            await s.wait_closed()

    try:
        yield serve
    finally:
        asyncio.run_coroutine_threadsafe(close(), loop).result()
        loop.call_soon_threadsafe(loop.stop)
        thread.join()
        loop.close()
