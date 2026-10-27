"""Replay recorded served_zk windows through PoUW's check: the Python reference (`core_check`) and its Lean port
(`pouw-check serve`, and `pouw-check-difftest`'s `opening`), each case against the verdict the run recorded.

  python replay_core.py OUT [--runs DIR ...] [--add DIR ...] [--v2 DIR|TAR]    OUT/replay.json and one line on stdout

Without ``--runs`` it selects every `job.sh` attempt in the evidence store and fetches its run record's ``out/``
(``--add`` names more, such as runs the store has not indexed yet); without
``--v2`` (A4's tree, or a tarball of it) it fetches A4's v2 run (`V2_ART`).  For each run with a summary:

* **core cases** are the checks `core_check.case` decides, mapped to (params, registration, the service's answer):
  ``receive-honest`` and the registration controls ``receive-foreign``, ``receive-extra``, ``receive-raised`` and
  ``register-extra`` (the request's registration with the calls' shapes added, as `job.sh` asks the committer), each paired
  with the honest outcome's answer (in a window stopped before its outcome, the controls with `NO_OUTCOME` and
  ``receive-honest`` not at all), and every ``outcome-*`` the run wrote (its ``outcome.json`` and ``audit.json``).  The
  params are the request's (`params.json`), the check item and the query the request registration's: ``receive-honest``
  REGISTERED says they were the verifier's own.  A REGISTERED check must pass every registration rule (the honest
  outcome it is paired with may still reject); an ACCEPT check must get accept; a REFUSED registration a reject by a
  registration rule (before ``outcome:``); a REJECT outcome a reject.  An accept's public outputs must be the recorded
  ``outcome.json``'s.  A window recorded before the fixed-length rule (no ``length`` item) is
  replayed under the verifier's length of its request (`LENGTH`, where its params name none), and a REGISTERED or ACCEPT
  of it must now get a reject as not of record (``window-form: no-fixed-length``).
* **openings** (``open-u<t>``, ``open-filler``, ``open-foreign-*``), as `job.sh` maps them, go to `core_check.opening`
  and Lean's.  In a window of the 336-byte entry rows (9 Oct on) an ACCEPT opening must open and a REJECT one must not;
  the windows before them hold entries of an earlier layout, so theirs compare Python with Lean only.
* **coin cases** go to `core_check.pass_epoch` and `core_check.draw` and Lean's (`of_record`): ``select`` (the recorded
  draw from the challenger's coin commitment, its draw coin's opening, the stream's digest, the receipt and draw.json),
  with generated controls (a unit changed, the coin tampered, opened after another registration, another stream or
  receipt, one unit fewer under the registration's K), each refused; and where ``--pass RUN=MANIFEST`` gives the run's pass (its ``served_run``),
  ``receive-honest``'s pass and the coin controls ``receive-coin-<case>`` and ``receive-unrecorded``, regenerated as
  `job.sh` made them (a v2 pass's ``receive-unrecorded`` relabelled v1 under a fixed beacon, where `job.sh` drew one).
  A DRAWN or REGISTERED check must get no refusal; a REFUSED one or a control a refusal.  Each is a replay of a recorded run, so the rules read the recorded coin versions too (`core_check.COINS_RECORDED`,
  `COIN_EARLY`).
* the other checks (coins, the stream's state, the firewall, Lean's tile verifier, replays, shapes) are not core_check's.

A run of an older record version (no registration, or another scheme) and the v2 registrations run through both sides
for agreement only.  Python and Lean must give the same outcome on every input, reason included.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any

import verity_difftest as D

from verity.core.protocols.pouw import coins as Co
from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw import verifier as PV
from verity.core.protocols.pouw import window as Wd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import of_record as OR  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
LEAN = REPO / "verity/core/protocols/pouw/lean/check"
V2_ART = "art:08afd6b23db47295569411eb024868fb39e6dbae6a11a3835657be263321621d"
V2_TILE = "Pc8TileSide_v1{K=4096,TM=1,TN=1,DEV='sm120'}"
REGISTRATION_RULES = ("window", "window-form", "public-items", "level", "public-value", "population", "law", "query",
                      "scheme")
#: the verifier's length of every recorded window: e2e.py's served requests, one slot each at SERVED_TOKENS under ignore_eos
LENGTH = {"max_tokens": [4], "stop": "fixed"}
#: what PoUW's rule says of a window registered before the fixed-length rule: not of record
NO_FIXED_LENGTH = "window-form: no-fixed-length"
V1_PASS = "verity-vllm/pouw-pearl-c-run/v1"
#: The evidence a registration control is paired with in a window stopped before its outcome: a registration rule
#: refuses before PoUW's rule reads it, and a control no rule refused is rejected at ``outcome:``, not by a rule.
NO_OUTCOME = {"protocol": CC.ONE_STAGE, "verdict": "reject", "reason": "no outcome", "commitment": None, "public": None,
              "profile": None}
CONTROLS = {"receive-foreign": "other/registration.json", "receive-extra": "registration-extra.json",
            "receive-raised": "registration-raised.json", "register-extra": None}


def research(*args: str) -> str:
    r = subprocess.run(["research", *args], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"research {' '.join(args)}: exit {r.returncode}\n{r.stderr[-2000:]}")
    return r.stdout


def store_runs(into: Path) -> list[Path]:
    """Every served_zk `job.sh` attempt's run record ``out/``, fetched under `into` (empty for a record without one, a job
    that failed before writing it)."""
    rows = research("data", "sql", "select a.id, o.artifact from attempts a join attempt_outputs o on o.attempt = a.id "
                    "where a.manifest_json like '%served_zk/job.sh%' and o.name = 'run_record' order by a.start_utc")
    runs = []
    for line in rows.splitlines()[1:]:
        run, art = line.split("\t")
        try:
            research("data", "fetch", art, "--to", str(into / run), "--path", "out/*")
        except SystemExit as e:
            if "no file matches" not in str(e):
                raise
            (into / run).mkdir(parents=True, exist_ok=True)
        runs.append(into / run)
    return runs


def load(p: Path) -> Any:
    return json.loads(p.read_text())


def hexed(j: Any) -> str:
    return D.canonical(j).encode().hex()


def ratio_of(x: Any, want: str) -> str:
    """A recorded profile's δ as `Core.Integrity`'s ratio string: the window's own when the float is it."""
    if isinstance(x, str):
        return x
    return want if float(Fraction(want)) == x else str(Fraction(x).limit_denominator(10**9))


def answer_of(outcome_dir: Path, delta: str) -> dict:
    """The service's answer (`Core.Answer`) one ``outcome-*`` directory records."""
    o, a = load(outcome_dir / "outcome.json"), load(outcome_dir / "audit.json")
    return OR.service_answer(o["accepted"], (o.get("verdict") or {}).get("codes") or [], o["profile"], a["registration"],
                             (a.get("audit") or {}).get("draw"), delta)


def params_of(out: Path, reg: dict) -> dict:
    p = load(out / "request/params.json")
    return {"check": reg["window"]["public"]["check"], "context": p["context"], "delta": p["delta"], "eps": p["eps"],
            "grid_bits": p["grid_bits"], "length": p.get("length", LENGTH), "query": reg["query"], "range": p["range"]}


def legacy(case: dict) -> bool:
    """Whether the case's registration is a window registered before the fixed-length rule: no length item."""
    try:
        items = json.loads(bytes.fromhex(case["commitment"]))["window"]["public"]
    except (KeyError, TypeError, ValueError):
        return False
    return isinstance(items, dict) and sorted(items) == sorted(Wd.LEGACY_REGISTERED)


def current(reg: Any) -> bool:
    return isinstance(reg, dict) and reg.get("scheme") == CC.SCHEME and any(
        r.get("port") == "entries" for r in reg.get("roots", []))


def core_cases(out: Path, checks: dict) -> list[tuple[str, dict, str, Any]]:
    """(check, case, recorded verdict, the recorded public outputs on accept) of every check core_check decides."""
    reg = load(out / "request/registration.json")
    params = params_of(out, reg)
    honest = answer_of(out / "outcome-honest", params["delta"]) if (out / "outcome-honest").is_dir() else None
    cases = []
    for name, rec in sorted(checks.items()):
        if name == "receive-honest" or name in CONTROLS:
            if honest is None and name == "receive-honest":
                continue
            path = CONTROLS.get(name)
            r = load(out / path) if path else copy.deepcopy(reg)
            if name == "register-extra":
                r["window"]["public"]["shapes"] = [[1, 4096, 4096]]
            pub = load(out / "outcome-honest/outcome.json")["public"] if name == "receive-honest" else None
            cases.append((name, {"params": params, "commitment": hexed(r), "evidence": hexed(honest or NO_OUTCOME)},
                          rec["verdict"], pub))
        elif name.startswith("outcome-") and (out / name / "outcome.json").is_file():
            ans = answer_of(out / name, params["delta"])
            pub = load(out / name / "outcome.json")["public"] if ans["verdict"] == "accept" else None
            cases.append((name, {"params": params, "commitment": hexed(reg), "evidence": hexed(ans)}, rec["verdict"], pub))
    return cases


def opening_inputs(out: Path, checks: dict, summary: dict) -> list[tuple[str, dict]]:
    """The openings `job.sh`'s ``opencheck`` judged, as `pouw-check-difftest`'s ``opening`` inputs."""
    regs = {s: load(out / s / "registration.json") for s in ("request", "other")}
    drawn = summary.get("units") or []
    u0 = drawn[0] if drawn else None
    ou = u0 if u0 is not None and u0 < regs["other"]["population"] else 0
    plan = {f"open-u{t}": ("request", t, f"u{t}", f"u{t}", True) for t in drawn}
    plan["open-filler"] = ("request", summary.get("filler_unit"), "filler", "filler", False)
    plan["open-foreign-proof"] = ("request", u0, "other-tile", f"u{u0}", True)
    plan["open-foreign-commit"] = ("other", ou, f"u{u0}", "other-tile", False)
    got = []
    for name, (side, unit, pub, op, is_drawn) in plan.items():
        if name not in checks or unit is None or not (out / pub / "pub.bin").is_file():
            continue
        p = load(out / side / "params.json")
        reg = regs[side]
        got.append((name, {"params": {"check": {**reg["window"]["public"]["check"], "entries": p["entries"]},
                                      "context": p["context"], "delta": p["delta"], "eps": p["eps"],
                                      "grid_bits": p["grid_bits"], "length": p.get("length", LENGTH), "query": reg["query"],
                                      "range": p["range"]},
                           "commitment": hexed(reg), "units": drawn if is_drawn else None, "unit": unit,
                           "pub": (out / pub / "pub.bin").read_bytes().hex(), "opening": load(out / op / "opening.json")}))
    return got


COIN_CONTROLS = ("foreign", "early", "mislabelled", "tampered", "resalted")


def _variant(man: dict, coins: dict, opening: dict, case: str) -> dict:
    """`served_commit.coin_variant`: the pass carrying a control's salt coin made of the genuine one."""
    o = dict(opening)
    if case == "foreign":
        o = Co.Challenger(coins["epoch"], PV.coins).open("salt", bytes.fromhex(o["after"])).to_json()
    elif case == "early":
        o["after"] = "00" * (len(o["after"]) // 2)
    elif case == "mislabelled":
        o["label"] = "draw"
    elif case == "tampered":
        o["coin"] = f"{int(o['coin'][:2], 16) ^ 1:02x}" + o["coin"][2:]
    else:
        man = {**man, "salt": "00" * 32}
    return {**{k: v for k, v in man.items() if k != "beacon"}, "schema": CC.MANIFEST_SCHEMA, "coins": coins, "salt_opening": o}


def selected_stream(ch: Path) -> str | None:
    """The stream digest the honest select read, from its DRAWN reply (a window stopped before its outcome has no
    audit.json); None for a run whose replies don't carry it."""
    for line in (ch / "requests.jsonl").read_text().splitlines():
        r, q = json.loads(line), json.loads(json.loads(line)["request"])
        if (q.get("op"), q.get("name")) == ("select", "honest") and r["reply"].startswith("DRAWN "):
            return json.loads(r["reply"].split(" ", 1)[1]).get("stream")
    return None


def coin_inputs(out: Path, checks: dict, summary: dict, passes: dict[str, Path]) -> list[tuple[str, str, dict, str]]:
    """(check, ``pass`` or ``draw``, its input, the recorded verdict or ``CONTROL``) of the run's coin checks."""
    ch = out / "challenger"
    if not (ch / "coins.json").is_file() or not (out / "draw.json").is_file():
        return []
    coins = load(ch / "coins.json")
    recorded = True
    reg, receipt, d = load(out / "request/registration.json"), load(out / "receipt.json"), load(out / "draw.json")
    openings = [json.loads(x) for x in (ch / "openings.jsonl").read_text().splitlines() if x]
    drawn = next((o for o in openings if o["label"] == "draw"), None)
    audit = out / "outcome-honest/audit.json"
    stream = load(audit)["stream"] if audit.is_file() else selected_stream(ch)
    got: list[tuple[str, str, dict, str]] = []
    if drawn is not None and stream is not None and "select" in checks:
        x = {"coins": coins, "recorded": recorded, "opening": drawn, "stream": stream,
             "registration": receipt["registration"], "receipt": receipt, "k": reg["law"]["k"],
             "population": reg["population"], "units": d["units"]}
        got.append(("select", "draw", x, checks["select"]["verdict"]))
        other = load(out / "other/registration.json") if (out / "other/registration.json").is_file() else {"x": 1}
        for name, edit in (("draw-unit", {"units": sorted({*x["units"][1:], next(u for u in range(x["population"])
                                                                                    if u not in x["units"])})}),
                           ("draw-coin", {"opening": {**drawn, "coin": f"{int(drawn['coin'][:2], 16) ^ 1:02x}" + drawn["coin"][2:]}}),
                           ("draw-after", {"registration": hashlib.sha512(D.canonical(other).encode()).hexdigest()}),
                           ("draw-stream", {"stream": "00" * 64}),
                           ("draw-receipt", {"receipt": {**receipt, "received": "1"}}),
                           ("draw-fewer", {"units": x["units"][:-1]})):
            got.append((name, "draw", {**x, **edit}, "CONTROL"))
    pass_path = passes.get(summary.get("served_run") or "")
    salt = out / "salt-opening.json"
    if pass_path is not None and salt.is_file():
        raw = pass_path.read_bytes()
        man, context = json.loads(raw), load(out / "request/params.json")["context"]
        x = {"coins": coins, "raw": raw.hex(), "context": context, "recorded": recorded}
        if "receive-honest" in checks:
            got.append(("receive-honest", "pass", x, checks["receive-honest"]["verdict"]))
        for case in COIN_CONTROLS:
            if f"receive-coin-{case}" in checks:
                v = json.dumps(_variant(man, coins, load(salt), case), sort_keys=True).encode()
                got.append((f"receive-coin-{case}", "pass", {**x, "raw": v.hex()}, checks[f"receive-coin-{case}"]["verdict"]))
        if "receive-unrecorded" in checks:
            un = {k: w for k, w in man.items() if k not in ("coins", "salt_opening")}
            if man.get("schema") != V1_PASS:
                un.update(schema=V1_PASS, beacon=hashlib.sha256(b"replay_core unrecorded").hexdigest())
            v = json.dumps(un).encode()
            got.append(("receive-unrecorded", "pass", {**x, "raw": v.hex()}, checks["receive-unrecorded"]["verdict"]))
    return got


def coin_judged(recorded: str, kind: str, out: Any) -> bool:
    """Whether the coin check's outcome `out` is the recorded verdict (a control's: refused)."""
    refused = out is not None and "refused" in out
    if recorded in ("DRAWN", "REGISTERED"):
        return not refused and (kind == "draw" or out.get("how") is not None)
    return refused


def opening_reference(x: dict) -> Any:
    p = CC.params(CC.canonical(D.canonical(x["params"]).encode()))
    c = CC.commitment(bytes.fromhex(x["commitment"]))
    return CC.opening(p, c, x["units"], x["unit"], bytes.fromhex(x["pub"]), x["opening"])


class Sides:
    """Python and Lean on one input each: `core` (a core_check case) and `opening`."""

    def __init__(self) -> None:
        self.core = D.Server(D.ProtocolPort("pouw-check", CC.case, inputs=None, package=LEAN).argv())
        self.opening = D.Server(D.Port("pouw-check-difftest", "opening", opening_reference, inputs=None,
                                       package=LEAN).argv())
        self.record = OR.Record()

    def coin(self, kind: str, x: dict) -> tuple[Any, bool]:
        """Lean's outcome of a pass or draw check, and whether Python's is the same (`of_record`)."""
        try:
            return (self.record.pass_epoch if kind == "pass" else self.record.draw)(x), True
        except OR.Disagree as e:
            return e.lean, False

    def close(self) -> None:
        self.core.close()
        self.opening.close()
        self.record.close()

    def case(self, case: dict) -> tuple[dict, bool]:
        """Python's outcome, and whether Lean's is the same, reason included."""
        py = CC.case(copy.deepcopy(case))
        return py, D.canonical(self.core.exchange(case)) == D.canonical(py)

    def open(self, x: dict) -> tuple[Any, bool]:
        try:
            py: tuple[str, Any] = ("output", opening_reference(copy.deepcopy(x)))
        except CC.Refused as e:
            py = ("reject", str(e))
        lean = self.opening.call("opening", x)
        same = lean == ("output", D.canonical(py[1])) if py[0] == "output" else lean[0] == "reject"
        return py, same


def registration_stage(reason: str | None) -> bool:
    return reason is not None and reason.split(":")[0] in REGISTRATION_RULES


def judged(recorded: str, a: dict, public: Any, old: bool = False) -> bool:
    """Whether core_check's outcome `a` is the recorded verdict: REGISTERED is the registration's, so an outcome past
    the registration rules holds it; a window registered before the fixed-length rule (`old`) that the run registered or
    accepted must now be refused as not of record."""
    if recorded in ("REGISTERED", "ACCEPT") and old:
        return a["verdict"] == "reject" and (a["reason"] or "").startswith(NO_FIXED_LENGTH)
    if a["verdict"] == "accept" and public is not None and D.canonical(a["public"]) != D.canonical(public):
        return False
    if recorded == "REGISTERED":
        return a["verdict"] == "accept" or not registration_stage(a["reason"])
    if recorded == "ACCEPT":
        return a["verdict"] == "accept"
    if recorded == "REFUSED":
        return a["verdict"] == "reject" and registration_stage(a["reason"])
    return recorded == "REJECT" and a["verdict"] == "reject"


def entry_rows(out: Path) -> bool:
    """Whether the run's window holds the 336-byte entry rows (its advice's entries are `Entry.to_json`'s)."""
    p = out / "request/advice.json"
    keys = set(Wd.Entry("none").to_json())
    return p.is_file() and all(isinstance(x, dict) and set(x) == keys for x in load(p).get("entries") or [None])


def opening_judged(recorded: str, py: tuple[str, Any]) -> bool:
    """Whether core_check.opening's outcome is the recorded verdict: an ACCEPT opening opens, a REJECT one does not."""
    opens = py == ("output", None)
    return opens if recorded == "ACCEPT" else recorded == "REJECT" and not opens


def replay_run(sides: Sides, out: Path, passes: dict[str, Path]) -> dict:
    summary = load(out / "summary.json")
    checks = summary.get("checks") or {}
    reg_p = out / "request/registration.json"
    reg = load(reg_p) if reg_p.is_file() else None
    row: dict[str, Any] = {"checks": len(checks), "cases": [], "openings": [], "coins": [], "apart": []}
    if not current(reg):
        row["version"] = "no registration" if reg is None else (reg.get("scheme") or "no scheme")
        try:
            older = core_cases(out, checks) if reg is not None else []
        except (KeyError, TypeError, OSError) as e:
            row["unmapped"] = f"{type(e).__name__}: {e}"
            older = []
        for name, case, recorded, _ in older:
            a, same = sides.case(case)
            row["cases"].append({"check": name, "recorded": recorded, "verdict": a["verdict"], "reason": a["reason"],
                                 "lean_same": same})
        return row
    row["version"] = CC.SCHEME
    mapped = set()
    for name, case, recorded, public in core_cases(out, checks):
        a, same = sides.case(case)
        mapped.add(name)
        row["cases"].append({"check": name, "recorded": recorded, "verdict": a["verdict"], "reason": a["reason"],
                             "lean_same": same, "same_verdict": judged(recorded, a, public, legacy(case))})
    rows = entry_rows(out)
    for name, x in opening_inputs(out, checks, summary):
        py, same = sides.open(x)
        mapped.add(name)
        o = {"check": name, "recorded": checks[name]["verdict"], "python": py, "lean_same": same}
        row["openings"].append({**o, "same_verdict": opening_judged(o["recorded"], py)} if rows else o)
    for name, kind, x, recorded in coin_inputs(out, checks, summary, passes):
        lean, same = sides.coin(kind, x)
        mapped.add(name)
        row["coins"].append({"check": name, "kind": kind, "recorded": recorded, "lean": lean, "lean_same": same,
                             "same_verdict": coin_judged(recorded, kind, lean)})
    row["apart"] = sorted(set(checks) - mapped)
    return row


def replay_v2(sides: Sides, root: Path) -> list[dict]:
    """A4's v2 registrations, each through both sides under the served path's pinned check item (`V2_TILE`), so each
    reaches the registration rules: none is a PoUW window (no served window), and its `registered` values wait on R7."""
    tile = CC.TILES[V2_TILE]
    rows = []
    for p in sorted(root.rglob("registration.json")):
        reg = load(p)
        case = {"params": {"check": {"entries": 2, "scheme": CC.SCHEME, "template": V2_TILE, "tile_work": tile.tile_work},
                           "context": "v2", "delta": "1/10", "eps": "1/2", "grid_bits": 3, "length": LENGTH,
                           "query": tile.query, "range": [1, 1 << 40]},
                "commitment": hexed(reg), "evidence": hexed({"protocol": CC.ONE_STAGE, "verdict": "reject",
                                                            "reason": "v2", "commitment": None, "public": None,
                                                            "profile": None})}
        a, same = sides.case(case)
        rows.append({"registration": str(p.relative_to(root)), "format": reg.get("format"), "verdict": a["verdict"],
                     "reason": a["reason"], "lean_same": same})
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--runs", type=Path, nargs="*")
    ap.add_argument("--add", type=Path, nargs="*", default=[], metavar="RUN",
                    help="run records beside the store's (or --runs'), such as runs the store has not indexed yet")
    ap.add_argument("--v2", type=Path)
    ap.add_argument("--pass", dest="passes", action="append", default=[], metavar="RUN=MANIFEST",
                    help="a served run's retained pass manifest, for its windows' pass checks")
    a = ap.parse_args(argv)
    passes = {k: Path(v) for k, _, v in (x.partition("=") for x in a.passes)}
    a.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        runs = a.runs if a.runs is not None else store_runs(Path(tmp) / "runs")
        runs = [*runs, *(r for r in a.add if r.name not in {x.name for x in runs})]
        v2 = a.v2
        if v2 is None:
            research("data", "fetch", V2_ART, "--to", str(Path(tmp) / "v2"))
            v2 = Path(tmp) / "v2"
        elif v2.is_file():
            with tarfile.open(v2) as tf:
                tf.extractall(Path(tmp) / "v2", filter="data")
            v2 = Path(tmp) / "v2"
        sides = Sides()
        report: dict[str, Any] = {"runs": {}, "v2": {"art": V2_ART, "registrations": replay_v2(sides, v2)}}
        for r in runs:
            out = r / "out" if (r / "out").is_dir() else r
            if not (out / "summary.json").is_file():
                report["runs"][r.name] = {"version": "no summary (the job failed before its checks)"}
                continue
            report["runs"][r.name] = replay_run(sides, out, passes)
        sides.close()
    cur = {k: v for k, v in report["runs"].items() if v.get("version") == CC.SCHEME}
    cases = [c for v in cur.values() for c in v["cases"]]
    coins = [c for v in cur.values() for c in v["coins"]]
    opens = [o for v in cur.values() for o in v["openings"]]
    older = [c for v in report["runs"].values() if v.get("version") != CC.SCHEME for c in v.get("cases", [])]
    every = cases + coins + opens + older + report["v2"]["registrations"]
    opened = [o for o in opens if "same_verdict" in o]
    judged_ = cases + coins + opened
    negative = ("REFUSED", "REJECT", "CONTROL")
    t = {"runs": len(report["runs"]), "current": len(cur), "cases": len(judged_),
         "same_verdict": sum(c["same_verdict"] for c in judged_),
         "controls": sum(c["recorded"] in negative for c in judged_),
         "controls_same": sum(c["same_verdict"] for c in judged_ if c["recorded"] in negative),
         "coin_cases": len(coins), "generated_controls": sum(c["recorded"] == "CONTROL" for c in coins),
         "controls_by_rule": sum(c["recorded"] in ("REFUSED", "REJECT") and registration_stage(c["reason"])
                                 for c in cases),
         "lean_same": sum(c["lean_same"] for c in every), "compared": len(every),
         "openings": len(opens) - len(opened), "opened": len(opened),
         "opening_controls": sum(o["recorded"] == "REJECT" for o in opened),
         "older_runs": sorted(k for k in report["runs"] if k not in cur), "older_cases": len(older),
         "v2": len(report["v2"]["registrations"]),
         "apart": dict(sorted(Counter(n.split("-")[0] for v in cur.values() for n in v["apart"]).items()))}
    report["totals"] = t
    t["line"] = (f"{t['current']} runs, {t['same_verdict']}/{t['cases']} same verdict, {t['controls']} negative controls "
                 f"({t['controls_same']} same; {t['controls_by_rule']} refused by a registration rule, "
                 f"{t['controls'] - t['controls_by_rule'] - t['generated_controls'] - t['opening_controls']} by the "
                 f"service or the coin rule, {t['opening_controls']} foreign openings, {t['generated_controls']} "
                 f"generated draw controls; {t['coin_cases']} coin cases, {t['opened']} openings); Lean = Python on "
                 f"{t['lean_same']}/{t['compared']} inputs "
                 f"({t['openings']} pre-layout openings, {t['older_cases']} cases of {len(t['older_runs'])} older runs, "
                 f"{t['v2']} v2 registrations apart)")
    (a.out / "replay.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")
    print(t["line"])
    return 0 if t["same_verdict"] == t["cases"] and t["lean_same"] == t["compared"] else 1


if __name__ == "__main__":
    sys.exit(main())
