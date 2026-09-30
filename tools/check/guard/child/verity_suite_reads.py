"""What a suite's tests read outside the test process itself, for `suites.py`'s per-test verdicts. Standard library only: the guard
loads it for the children it forks, and `sitecustomize` beside it for every Python a test starts.

Each open of one of the suite's test modules (a `test_*.py` under `VERITY_SUITE_DIR`), or of a repository file outside the suite's
inputs (`VERITY_SUITE_ALLOW`, relative to `VERITY_SUITE_ROOT`), is appended at once to `VERITY_SUITE_READS` as a line
`R|O<tab>module<tab>path`. The module is the test module `VERITY_SUITE_MODULE` names: the guard sets it while one is collected or
its tests run, and a child inherits it. One write per open, so a forked child that ends through `os._exit` loses nothing.
"""
from __future__ import annotations

import json
import os
import sys


class Reads:
    def __init__(self, env=os.environ):
        self.root = os.path.realpath(env.get("VERITY_SUITE_ROOT", ""))
        self.suite = os.path.join(os.path.realpath(env.get("VERITY_SUITE_DIR") or os.getcwd()), "")
        paths = [os.path.normpath(os.path.join(self.root, p)) for p in json.loads(env.get("VERITY_SUITE_ALLOW", "[]"))]
        self.allowed = tuple(os.path.join(p, "") for p in paths if os.path.isdir(p))
        self.allowed += tuple({os.path.join(os.path.realpath(sys.prefix), ""), os.path.join(os.path.abspath(sys.prefix), ""),
                               os.path.join(os.path.dirname(os.path.realpath(__file__)), "")})       # the tracer itself
        self.allowed_files = frozenset(p for p in paths if not os.path.isdir(p))
        self.inside = os.path.join(self.root, "")
        self.log = env.get("VERITY_SUITE_READS")
        self.busy = False

    def classify(self, path: str) -> tuple[str, str] | None:
        """("R", path) for one of the suite's test modules, ("O", path) for a repository file outside the inputs, else None."""
        path = os.path.normpath(path if os.path.isabs(path) else os.path.join(os.getcwd(), path))
        name = os.path.basename(path)
        if path.startswith(self.suite) and name.startswith("test_") and name.endswith(".py"):
            return "R", os.path.relpath(path, self.root)
        if path.startswith(self.inside) and not path.startswith(self.allowed) and path not in self.allowed_files:
            return "O", os.path.relpath(path, self.root)
        return None

    def write(self, kind: str, module: str, rel: str) -> None:
        if not self.log:
            return
        self.busy = True
        try:
            fd = os.open(self.log, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            try:
                os.write(fd, f"{kind}\t{module}\t{rel}\n".encode())
            finally:
                os.close(fd)
        finally:
            self.busy = False

    def seen(self, event: str, args: tuple, module: str) -> None:
        if event != "open" or not args or isinstance(args[0], int) or self.busy:
            return
        try:
            path = os.fsdecode(args[0])
        except TypeError:
            return
        if (hit := self.classify(path)) is not None:
            self.write(hit[0], module, hit[1])

    def install(self) -> None:
        """Trace this process (a child a test started): every open, attributed to the module it inherited."""
        module = os.environ.get("VERITY_SUITE_MODULE", "*")
        sys.addaudithook(lambda event, args: self.seen(event, args, module))
