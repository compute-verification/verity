"""libvc's conformance (their conformance.py and vclib.py, unedited, imported from DIR) with one launch sabotaged at a time:
each sabotage must turn at least one of the cases it reaches false, and the clean pass before and after must be all true.

    python libvc_sabotage.py DIR LIB > sabotage.jsonl
"""
from __future__ import annotations

import json
import random
import sys

DIR, LIB = sys.argv[1], sys.argv[2]
sys.path.insert(0, DIR)
import conformance as C  # noqa: E402


class Sabotaged:
    """Forwards to libvc, but `name`'s launch goes through `fn(f, *args)`."""

    def __init__(self, lib, name: str, fn) -> None:
        self._lib, self._name, self._fn, self.calls = lib, name, fn, 0

    def __getattr__(self, name: str):
        f = getattr(self._lib, name)
        if name != self._name:
            return f

        def call(*args):
            self.calls += 1
            return self._fn(f, self.calls, *args)
        return call


def short_n(i_n: int):
    def fn(f, k, *a):
        a = list(a)
        if a[i_n] > 1:
            a[i_n] -= 1
        return f(*a)
    return fn


def skip(f, k, *a):
    return 0


def every_second(f, k, *a):
    return 0 if k % 2 == 0 else f(*a)


# launch, argument index of its count, the cases it reaches
SABOTAGES = [
    ("rows short a row", "rows_launch", short_n(4), ["case_rows", "case_pearl"]),
    ("commit short a row", "commit_rows_launch", short_n(5), ["case_hm96_rows", "case_graph", "case_fold_wide"]),
    ("commit's second call skipped", "commit_rows_launch", every_second, ["case_hm96_rows", "case_fold_wide"]),
    ("tree skipped", "tree_launch", skip, ["case_frame", "case_pearl"]),
    ("leaves_x short a leaf", "leaves_x_launch", short_n(3), ["case_hm96_x", "case_pearl"]),
    ("open short a record", "open_launch", short_n(8), ["case_open"]),
    ("tiles skipped", "tiles_launch", skip, ["case_pearl"]),
    ("row seeds short a seed", "row_seeds_launch", short_n(2), ["case_pearl"]),
    ("salts skipped", "salts_launch", skip, ["case_hm96_x", "case_pearl"]),
]
CASES = {c.__name__: c for c in C.CASES}


def run(names: list[str], lib_name: str | None = None, fn=None) -> dict:
    C.RESULTS.clear()
    d = C.Dev(LIB)
    wrap = None
    if lib_name:
        wrap = d.lib = Sabotaged(d.lib, lib_name, fn)
    for name in names:
        try:
            CASES[name](d, random.Random(name), True)
        except Exception as e:  # noqa: BLE001 - a raise is a rejection too
            C.report(name, False, error=f"{type(e).__name__}: {e}"[:200])
        d.release()
    bad = sorted({r["case"] for r in C.RESULTS if not r["ok"]})
    return {"cases": len(C.RESULTS), "failed": sum(not r["ok"] for r in C.RESULTS), "failed_kinds": bad,
            "calls": wrap.calls if wrap else None}


def main() -> int:
    every = sorted({c for *_x, cs in SABOTAGES for c in cs})
    out, ok = [], True
    clean = run(every)
    out.append({"sabotage": None, **clean, "ok": clean["failed"] == 0})
    for label, name, fn, cases in SABOTAGES:
        r = run(cases, name, fn)
        rejected = r["failed"] > 0 and bool(r["calls"])
        out.append({"sabotage": label, "launch": name, **r, "ok": rejected})
    again = run(every)
    out.append({"sabotage": None, "after": True, **again, "ok": again["failed"] == 0})
    for rec in out:
        ok &= rec["ok"]
        print(json.dumps({"record": True, **rec}), flush=True)
    print(json.dumps({"summary": True, "ok": ok, "sabotages": len(SABOTAGES),
                      "rejected": sum(r["ok"] for r in out if r["sabotage"])}), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
