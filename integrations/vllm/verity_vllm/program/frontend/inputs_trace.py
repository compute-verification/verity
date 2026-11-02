"""Record-freeness evidence: an audit hook that logs every file `open` (and `os.listdir`/`glob`) the construction performs,
so a reviewer can verify no execution record / accepted Program / hand-written emitter was read.

The predictor (`verity_vllm.program.predict`) is a forbidden input of the trace: a switched row's Build is the AOT Build
(`predict.aot`), and the trace is what checks it (its comparison, `pipeline.neutrality`'s heavy tier), so a trace that read the
predictor would check it against itself."""

from __future__ import annotations

import json
import os
import sys
import threading
from importlib.util import find_spec

#: the predictor, found by its module name without running it: a layout move rewrites the name, and a name that no longer
#: resolves refuses the import of this module rather than leaving a marker that matches nothing
_PREDICTOR = find_spec("verity_vllm.program.predict")
if _PREDICTOR is None:
    raise ImportError("inputs_trace: verity_vllm.program.predict does not resolve; name the predictor's module here")
_FORBIDDEN_MARKERS = ("program.json", "instances.jsonl", "experimental/aot/build.py", "experimental/aot/",
                      _PREDICTOR.name.replace(".", "/") + "/", "/vol/corpus", "census/B0-", "log.jsonl")


class InputsTrace:
    def __init__(self, path: str) -> None:
        self.path = path
        self.opened: list[dict] = []
        self.forbidden: list[dict] = []
        self._lock = threading.Lock()
        self._enabled = False

    def _hook(self, event: str, args) -> None:
        if not self._enabled or event not in ("open", "os.listdir", "os.scandir", "glob.glob"):
            return
        target = args[0] if args else None
        if not isinstance(target, (str, bytes, os.PathLike)):
            return
        p = os.fspath(target)
        if isinstance(p, bytes):
            p = p.decode(errors="replace")
        mode = args[1] if event == "open" and len(args) > 1 else None
        rec = {"event": event, "path": p, "mode": mode}
        with self._lock:
            self.opened.append(rec)
            if any(m in p for m in _FORBIDDEN_MARKERS):
                self.forbidden.append(rec)

    def __enter__(self):
        sys.addaudithook(self._hook)
        self._enabled = True
        return self

    def __exit__(self, *exc):
        self._enabled = False
        self.flush()
        return False

    def summary(self) -> dict:
        paths = sorted({r["path"] for r in self.opened})
        # collapse the interpreter's own noise: site-packages .py/.pyc imports are reported as counts per top dir
        site = [p for p in paths if "site-packages/" in p]
        other = [p for p in paths if "site-packages/" not in p]
        by_pkg: dict[str, int] = {}
        for p in site:
            tail = p.split("site-packages/", 1)[1] or "."
            by_pkg[tail.split("/", 1)[0]] = by_pkg.get(tail.split("/", 1)[0], 0) + 1
        return {"n_open_events": len(self.opened), "distinct_paths": len(paths), "site_packages_by_package": by_pkg,
                "non_site_packages_paths": other, "forbidden_hits": self.forbidden}

    def flush(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.summary(), f, indent=1, sort_keys=True)


__all__ = ["InputsTrace"]
