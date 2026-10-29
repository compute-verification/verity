"""Replay recorded served_zk windows through the window's Lean port (`pouw-window-difftest`) and its Python mirror
(`verity.core.protocols.pouw.window`), each against what the run recorded.

  python replay_window.py OUT [--runs DIR ...] [--add DIR ...]    OUT/replay-window.json and one line on stdout

Without ``--runs`` it fetches every served_zk `job.sh` attempt's ``out/`` from the evidence store (`replay_core.store_runs`;
``--add`` names more).  Each run's ``request`` and ``other`` sides with an ``advice.json`` are a window.  For a window whose
entries are the 336-byte entry rows:

* **the statement**: `window_problem` on the advice's rows at the registered work, the check's bound and the pass's -h3
  root, with the calls' records and the side indices their keeps give, holds (the committer held it before committing),
  and so does each entry's `layout_problem` and each call's `record_problem`;
* **the root**: each entry's leaf (`entry_bc` of its row under its recorded salt, `entry_leaf` in the window's served
  domain, `core_check.window_domain`) hashes up its recorded path to the registered root (`core_check.verify_path`), and
  the served domain is the registered one;
* **the draw**: `tile_at` of each drawn unit and of the filler unit is the recorded opening's entry and side rows;
* **the registered items**: `work_at` of the registered level gives, through `work_to_json` and `sampling`, the
  registration's ``work`` and ``sampling``, and `side_domain` of its window domain is defined for both sides;
* **negative controls**: the window broken in each way `window_problem` names (`BREAKS`), each refused for that reason;
  the root with an entry's salt, a sibling or the index tampered, each failing its path; and a unit past the padded count,
  in no entry.

A window of an older entry layout gets the registered items only.  Every input goes to both sides, which
must give the same outcome, reason included; a check holds when the Python's outcome is the recorded value.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

import verity_difftest as D

from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw import window as Wd
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain

REPO = Path(__file__).resolve().parents[3]
LEAN = REPO / "verity/core/protocols/pouw/lean/window"
EXE = "pouw-window-difftest"
SIDES = ("request", "other")
NULL = ("output", "null")


def load(p: Path) -> Any:
    return json.loads(p.read_text())


def _work(v: dict) -> Wd.Work:
    return Wd.Work(v["level"], v["credited"], v["n"])


def _entry(v: dict) -> Wd.Entry:
    return Wd.Entry.from_row(bytes.fromhex(v["row"]))


def _dom(v: dict) -> CommitmentDomain:
    return CommitmentDomain(bytes.fromhex(v["binding"]), Wd.OWNER, RangeIndexedDomain(0, v["count"]), hash="sha512")


#: each function the replay calls, as the window difftest's ports call it (`lean/window/tests/test_difftest.py`)
REFERENCES: dict[str, Callable[[dict], Any]] = {
    "work_at": lambda v: asdict(Wd.Work.at(v["ell"], v["bits"])),
    "work_to_json": lambda v: _work(v["work"]).to_json(v["tile_work"]),
    "sampling": lambda v: Wd.sampling(Fraction(*v["eps"]), Fraction(*v["delta"]), _work(v["work"]), v["tile_work"],
                                      v["bits"]),
    "side_domain": lambda v: Wd.side_domain(bytes.fromhex(v["window"]), v["side"]).domain_id.hex(),
    "entry_row": lambda v: Wd.Entry.from_json(v["entry"]).row().hex(),
    "layout_problem": lambda v: Wd.layout_problem(_entry(v)),
    "record_problem": lambda v: Wd.record_problem(_entry(v), v["record"]),
    "window_problem": lambda v: Wd.window_problem([bytes.fromhex(r) for r in v["rows"]], _work(v["work"]), v["bound"],
                                                  bytes.fromhex(v["root_h3"]), v["bits"], v["records"], v["indices"]),
    "tile_at": lambda v: Wd.tile_at([Wd.Entry.from_row(bytes.fromhex(r)) for r in v["rows"]], v["t"]),
    "entry_bc": lambda v: Wd.entry_bc(_entry(v), bytes.fromhex(v["salt"])).hex(),
    "entry_leaf": lambda v: Wd.entry_leaf(_dom(v["domain"]), v["e"], bytes.fromhex(v["bc"])).hex(),
}

#: `window_problem`'s reasons, by a fragment each holds (as the window difftest's `WINDOW` names them)
REASONS = {"bound": "entries, not the bound", "row": "not an entry row", "layout": "not the layout",
           "kinds": "the calls, then", "gap": "does not start where", "size": "tiles, not its",
           "h3": "-h3 root is not the pass's", "overlap": "before the last call's end", "fillerbig": "the filler's range is",
           "end": "the ranges end at", "level": "real tiles are level", "records": "records, not one per call",
           "indices": "rows' indices are not", "record": "record"}


def reason_of(why: str) -> str:
    return next((k for k, frag in REASONS.items() if frag in why), f"other: {why}")


class Sides:
    """Each input through Python (`REFERENCES`) and, with `lean`, through `pouw-window-difftest`, counting agreement."""

    def __init__(self, lean: bool = True) -> None:
        self.ports = {fn: D.Port(EXE, fn, reference=ref, inputs=None, package=LEAN, rejects=(ValueError,))
                      for fn, ref in REFERENCES.items()}
        self.server = D.Server(self.ports["work_at"].argv()) if lean else None
        self.compared, self.same, self.apart = 0, 0, []

    def __call__(self, fn: str, x: dict) -> D.Outcome:
        """Python's outcome of `fn` on `x`; Lean's is compared with it."""
        port = self.ports[fn]
        py = port.python(x)
        if self.server is None:
            return py
        try:
            lean = port.lean(self.server, x)
        except (D.ProtocolError, TypeError) as e:
            lean = ("error", str(e))
        self.compared += 1
        if "error" not in (py[0], lean[0]) and port.key(py) == port.key(lean):
            self.same += 1
        else:
            self.apart.append({"fn": fn, "python": py, "lean": lean, "input": D.canonical(x)[:2000]})
        return py

    def close(self) -> None:
        if self.server is not None:
            self.server.close()


def out_of(v: Any) -> D.Outcome:
    return "output", D.canonical(v)


def value(o: D.Outcome) -> Any:
    if o[0] != "output":
        raise ValueError(f"not an output: {o}")
    return json.loads(o[1])


def row_layout(advice: dict) -> bool:
    """Whether the advice's entries are the 336-byte entry rows (`Entry.to_json`'s keys)."""
    keys = set(Wd.Entry("none").to_json())
    return all(isinstance(x, dict) and set(x) == keys for x in advice.get("entries") or [None])


class Window:
    """One recorded window's checks (each the Python's outcome against the recorded value) and controls (each refused)."""

    def __init__(self, sides: Sides, out: Path, side: str) -> None:
        self.sides, self.out, self.side = sides, out, side
        self.advice = load(out / side / "advice.json")
        self.reg = load(out / side / "registration.json")
        self.params = load(out / side / "params.json")
        self.pub = self.reg["window"]["public"]
        self.bits = self.params["grid_bits"]
        self.checks: list[dict] = []
        self.controls: list[dict] = []

    def check(self, name: str, fn: str, x: dict, want: D.Outcome) -> D.Outcome:
        got = self.sides(fn, x)
        self.checks.append({"check": name, "ok": got == want, **({} if got == want else {"got": got, "want": want})})
        return got

    def holds(self, name: str, ok: bool, detail: Any = None) -> None:
        """A check of the Python's own (a path to the root, a domain), on values both sides agreed on."""
        self.checks.append({"check": name, "ok": ok, **({} if ok or detail is None else {"detail": detail})})

    def control(self, name: str, refused: bool, detail: Any = None) -> None:
        self.controls.append({"control": name, "refused": refused, **({} if refused or detail is None else {"detail": detail})})

    def registered(self) -> None:
        w, tw = self.pub["work"], self.pub["check"]["tile_work"]
        at = self.sides("work_at", {"ell": w["level"], "bits": self.bits})
        self.holds("work_at", at[0] == "output", at)
        if at[0] != "output":
            return
        work = value(at)
        self.check("work", "work_to_json", {"work": work, "tile_work": tw}, out_of(w))
        q = {k: [Fraction(self.params[k]).numerator, Fraction(self.params[k]).denominator] for k in ("eps", "delta")}
        self.check("sampling", "sampling", {**q, "work": work, "tile_work": tw, "bits": self.bits},
                   out_of(self.pub["sampling"]))
        dom = {r["port"]: r for r in self.reg["roots"]}.get(Wd.PORT, {}).get("domain")
        for s in ("a", "b") if dom else ():
            o = self.sides("side_domain", {"window": dom, "side": s})
            self.holds(f"side-domain-{s}", o[0] == "output", o)

    def statement(self) -> dict:
        """The window's base `window_problem` input, after its rows and records pass their own checks."""
        rows = []
        for e, x in enumerate(self.advice["entries"]):
            o = self.sides("entry_row", {"entry": x})
            self.holds(f"entry-row-{e}", o[0] == "output", o)
            rows.append(value(o) if o[0] == "output" else Wd.Entry("none").row().hex())
        records = self.advice["records"]
        w = self.pub["work"]
        base = {"rows": rows, "work": {"level": w["level"], "credited": w["credited_tiles"], "n": w["n"]},
                "bound": self.params["entries"], "root_h3": self.advice["root_h3"], "bits": self.bits,
                "records": records, "indices": None}
        self.check("window", "window_problem", base, NULL)
        entries = [Wd.Entry.from_row(bytes.fromhex(r)) for r in rows]
        indices = [{s: list(r["keep"][s]) for s in "ab"} if r is not None else
                   {"a": [0] * x.n_a, "b": [0] * x.n_b} if x.kind == "filler" else None for x, r in zip(entries, records)]
        self.check("window-indices", "window_problem", {**base, "indices": indices}, NULL)
        for e, r in enumerate(rows):
            self.check(f"layout-{e}", "layout_problem", {"row": r}, NULL)
            if records[e] is not None:
                self.check(f"record-{e}", "record_problem", {"row": r, "record": records[e]}, NULL)
        return {**base, "_entries": entries, "_indices": indices}

    def breaks(self, base: dict) -> None:
        entries: list[Wd.Entry] = base.pop("_entries")
        indices = base.pop("_indices")
        calls = [x.kind for x in entries].index("filler")
        c0, filler = entries[0], entries[calls]

        def put(rows: list[str], e: int, **kw: Any) -> list[str]:
            rows = list(rows)
            rows[e] = Wd.Entry(**(entries[e].__dict__ | kw)).row().hex()
            return rows

        tag = hashlib.sha512(b"replay_window control").digest()
        size = filler.hi - filler.lo
        w = base["work"]
        up = Wd.Work.at(w["level"] + 1, self.bits)
        cases: dict[str, dict] = {
            "bound": {"bound": base["bound"] + 1},
            "row": {"rows": [base["rows"][0][:-2], *base["rows"][1:]]},
            "layout": {"rows": put(base["rows"], 0, m=c0.m + 1)},
            "kinds": {"rows": [base["rows"][calls], *base["rows"][1:calls], base["rows"][0], *base["rows"][calls + 1:]]},
            "gap": {"rows": put(base["rows"], 0, lo=c0.lo + 1, hi=c0.hi + 1)},
            "size": {"rows": put(base["rows"], 0, **({"n_a": c0.n_a + 1} if c0.n_a < c0.m else {"n_b": c0.n_b - 1}))},
            "h3": {"rows": put(base["rows"], 0, root_h3=tag)},
            "end": {"work": {**w, "n": w["n"] + 1}},
            "level": {"work": {"level": up.level, "credited": up.credited, "n": w["n"]}},
            "records": {"records": base["records"][:-1]},
            "record": {"records": [{**base["records"][0], "replay_window": 1}, *base["records"][1:]]},
            "indices": {"indices": [{**indices[0], "a": [*indices[0]["a"][:-1], indices[0]["a"][-1] + 1]}, *indices[1:]]},
        }
        if calls > 1:
            prev = entries[calls - 2]
            cases["overlap"] = {"rows": put(base["rows"], calls - 1, tile0=prev.tile0 + Wd.pass_tiles(prev) - 1)}
        if size > filler.n_a:
            cases["fillerbig"] = {"rows": put(base["rows"], calls, n_b=(size - 1) // filler.n_a)}
        for why, edit in cases.items():
            o = self.sides("window_problem", {**base, **edit})
            got = reason_of(value(o)) if o[0] == "output" and value(o) is not None else o[0] if o[0] != "output" else None
            self.control(f"window-{why}", got == why, o)

    def root(self, rows: list[str]) -> None:
        roots = {r["port"]: r for r in self.reg["roots"]}
        dom = CC.window_domain({"check": {"entries": self.params["entries"]}, "context": self.params["context"]}, self.reg)
        self.holds("served-domain", dom.domain_id.hex() == roots[Wd.PORT]["domain"])
        root = bytes.fromhex(roots[Wd.PORT]["root"])
        d = {"binding": dom.binding.hex(), "count": dom.count}
        salts, paths = self.advice["entry_salts"], self.advice["window_path"]

        def leaf(e: int, row: str, salt: str, at: int | None = None) -> bytes | None:
            bc = self.sides("entry_bc", {"row": row, "salt": salt})
            if bc[0] != "output":
                return None
            lf = self.sides("entry_leaf", {"domain": d, "e": e if at is None else at, "bc": value(bc)})
            return bytes.fromhex(value(lf)) if lf[0] == "output" else None

        def up(e: int, lf: bytes | None, sibs: list[str]) -> bool:
            return lf is not None and CC.verify_path(dom, root, e, lf, [bytes.fromhex(s) for s in sibs])

        for e, row in enumerate(rows):
            self.holds(f"root-{e}", up(e, leaf(e, row, salts[e]), paths[e]))
        flipped = f"{int(salts[0][:2], 16) ^ 1:02x}" + salts[0][2:]
        self.control("root-salt", not up(0, leaf(0, rows[0], flipped), paths[0]))
        sib = f"{int(paths[0][0][:2], 16) ^ 1:02x}" + paths[0][0][2:]
        self.control("root-sibling", not up(0, leaf(0, rows[0], salts[0]), [sib, *paths[0][1:]]))
        self.control("root-index", not up(1, leaf(0, rows[0], salts[0], at=1), paths[1]))

    def draw(self, rows: list[str]) -> None:
        if self.side != "request":
            return
        summary = load(self.out / "summary.json") if (self.out / "summary.json").is_file() else {}
        units = load(self.out / "draw.json")["units"] if (self.out / "draw.json").is_file() else []
        opened = [(t, self.out / f"u{t}" / "opening.json") for t in units]
        if summary.get("filler_unit") is not None:
            opened.append((summary["filler_unit"], self.out / "filler" / "opening.json"))
        for t, p in opened:
            if not p.is_file():
                continue
            op = load(p)
            self.check(f"tile-at-{t}", "tile_at", {"rows": rows, "t": t},
                       out_of([op["entry"], op["a"]["index"], op["b"]["index"]]))
        o = self.sides("tile_at", {"rows": rows, "t": self.pub["work"]["n"]})
        self.control("tile-at-past", o[0] == "reject", o)

    def run(self) -> dict:
        self.registered()
        current = row_layout(self.advice) and isinstance(self.advice.get("records"), list)
        if current:
            base = self.statement()
            rows = list(base["rows"])
            self.breaks(base)
            self.root(rows)
            self.draw(rows)
        return {"layout": "entry rows" if current else "older", "checks": self.checks, "controls": self.controls}


def replay(sides: Sides, runs: list[Path]) -> dict:
    report: dict[str, Any] = {"runs": {}}
    for r in runs:
        out = r / "out" if (r / "out").is_dir() else r
        got = {}
        for side in SIDES:
            if all((out / side / f).is_file() for f in ("advice.json", "registration.json", "params.json")):
                try:
                    got[side] = Window(sides, out, side).run()
                except (KeyError, TypeError, ValueError, IndexError) as e:
                    got[side] = {"unmapped": f"{type(e).__name__}: {e}"}
        if got:
            report["runs"][r.name] = got
    windows = [w for v in report["runs"].values() for w in v.values()]
    checks = [c for w in windows for c in w.get("checks", [])]
    controls = [c for w in windows for c in w.get("controls", [])]
    t = {"runs": len(report["runs"]), "windows": len(windows),
         "entry_rows": sum(w.get("layout") == "entry rows" for w in windows),
         "older": sum(w.get("layout") == "older" for w in windows),
         "unmapped": sum("unmapped" in w for w in windows),
         "checks": len(checks), "checks_ok": sum(c["ok"] for c in checks),
         "controls": len(controls), "controls_refused": sum(c["refused"] for c in controls),
         "draws": sum(c["check"].startswith("tile-at-") for c in checks),
         "compared": sides.compared, "lean_same": sides.same}
    t["line"] = (f"{t['windows']} windows of {t['runs']} runs ({t['entry_rows']} of entry rows, {t['older']} older: "
                 f"registered items only; {t['unmapped']} unmapped): {t['checks_ok']}/{t['checks']} checks the recorded "
                 f"value ({t['draws']} draws against their openings), {t['controls_refused']}/{t['controls']} negative "
                 f"controls refused; Lean = Python on {t['lean_same']}/{t['compared']} inputs")
    report["totals"] = t
    report["apart"] = sides.apart
    return report


def passed(report: dict) -> bool:
    t = report["totals"]
    return (t["checks_ok"] == t["checks"] and t["controls_refused"] == t["controls"] and t["lean_same"] == t["compared"]
            and t["unmapped"] == 0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--runs", type=Path, nargs="*")
    ap.add_argument("--add", type=Path, nargs="*", default=[], metavar="RUN")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        if a.runs is None:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from replay_core import store_runs
            runs = store_runs(Path(tmp) / "runs")
        else:
            runs = list(a.runs)
        runs = [*runs, *(r for r in a.add if r.name not in {x.name for x in runs})]
        sides = Sides()
        try:
            report = replay(sides, sorted(runs, key=lambda p: p.name))
        finally:
            sides.close()
    (a.out / "replay-window.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")
    print(report["totals"]["line"])
    return 0 if passed(report) else 1


if __name__ == "__main__":
    sys.exit(main())
