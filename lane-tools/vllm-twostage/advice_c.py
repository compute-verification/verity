"""vllm-twostage (VLLM-250): each boundary window's advice (`verity-vllm advice`) from a boundary-rows serve's private records.

    python advice_c.py ARM PRIV_ADVICE OUT [--controls] [--T 600] [--tau 0.1]

ARM is the serve's boundary-rows arm (twostage_serve.sh's PRIV/out/boundary-rows): its step trace
(server/step_trace.*.jsonl[.gz]), each class's index.json (server/boundary/w<k>/p<P>-o<O>/) and the client records
(*/client.jsonl, anywhere under ARM).

A boundary window's requests are its classes' index.json spans. The cut (`cut`): a window's steps are the range its requests'
rows fill, sliced by `advice.windows_of`; it lines up when every request the trace saw is in exactly one window and each
window's slice holds exactly its own requests. Otherwise it refuses (`CutError`): a request counted in the wrong window (a
span carried across a close puts its rows in another window's range) or in none.

Per window, under PRIV_ADVICE/w<k>/: trace.jsonl (the slice), requests.jsonl (the Commit record: the engine id, the client's
id, which the engine id carries from the client's X-Request-Id, the prompt, the ids served and the finish) and the advice
(`advice.write_one`: registration.json and private/). OUT/advice/w<k>/registration.json holds only the registration's
``binding``, the part `two_stage.py first --advice` reads: its ``bits_by_window`` counts per egress window stay private.
--controls: the cut must refuse a request moved to another window and a request dropped from every window.

Prints no count, id, length or timing: only window labels, verdicts and the binding's root prefix.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path

from verity_vllm.engine import step_trace as ST
from verity_vllm.pipeline import advice as ADV
from verity_vllm.pipeline import boundary_rows as BR


class CutError(ValueError):
    pass


def records(path: Path) -> list[dict]:
    with (gzip.open if path.name.endswith(".gz") else open)(path, "rt") as f:
        recs = [json.loads(x) for x in f if x.strip()]
    if not recs or recs[0].get("kind") != "header" or recs[0].get("schema") != ST.SCHEMA:
        raise SystemExit(f"advice-c: {path.name} is not a {ST.SCHEMA} step trace")
    return recs


def boundary(arm: Path) -> dict[int, dict[str, BR.Span]]:
    """Each boundary window's spans by engine id, from its classes' index.json."""
    out: dict[int, dict[str, BR.Span]] = {}
    for idx in sorted((arm / "server" / "boundary").glob(f"w*/p*-o*/{BR.INDEX}")):
        w = int(idx.parent.parent.name[1:])
        for s in json.loads(idx.read_text())["spans"]:
            span = BR.Span.from_json(s)
            if any(span.engine_id in m for m in out.values()) or span.engine_id in out.get(w, {}):
                raise CutError("a request is a span of two boundary classes or windows")
            out.setdefault(w, {})[span.engine_id] = span
    if not out:
        raise SystemExit("advice-c: no boundary window under ARM/server/boundary")
    return out


def cut(recs: list[dict], windows: dict[int, set[str]]) -> dict[int, list[dict]]:
    """Each window's slice of the trace, refused (CutError) unless it lines up with the boundary windows."""
    if any(r["kind"] == "window" for r in recs):
        raise CutError("the trace already holds Commit windows")
    seen = [r["request"] for r in recs if r["kind"] == "arrive"]
    if len(set(seen)) != len(seen):
        raise CutError("a request arrives twice in the trace")
    home: dict[str, int] = {}
    for w, ids in windows.items():
        for rid in ids:
            home[rid] = w
    if set(seen) - set(home):
        raise CutError("a request the trace saw is in no boundary window")
    if set(home) - set(seen):
        raise CutError("a boundary span never arrives in the trace")
    steps = [r for r in recs if r["kind"] == "step"]
    marks: list[dict] = []
    last = -1
    for w in sorted(windows):
        mine = [st["step"] for st in steps if any(row[0] in windows[w] for row in st["rows"])]
        if not mine:
            raise CutError(f"window w{w}'s requests have no rows in the trace")
        lo, hi = min(mine), max(mine) + 1
        if lo <= last:
            raise CutError(f"window w{w}'s steps overlap the window before it")
        last = hi - 1
        marks += [{"kind": "window", "window": w, "event": "open", "before_step": lo},
                  {"kind": "window", "window": w, "event": "close", "before_step": hi}]
    sliced, still_open = ADV.windows_of(recs + marks)
    if still_open or set(sliced) != set(windows):
        raise CutError("a boundary window has no slice")
    for w, part in sliced.items():
        ids = {row[0] for r in part if r["kind"] == "step" for row in r["rows"]}
        if ids != windows[w]:
            raise CutError(f"window w{w}'s steps hold a request of another window (a request counted in the wrong window)")
    return sliced


def users(arm: Path, engine_ids: set[str]) -> dict[str, str]:
    """Each engine id's client id: the one client record id it carries (vLLM's request id is made from X-Request-Id)."""
    ids = {json.loads(x)["id"] for f in arm.rglob("client.jsonl") for x in f.read_text().splitlines() if x.strip()}
    out = {}
    for e in engine_ids:
        hit = [c for c in ids if re.search(rf"(?<![0-9A-Za-z]){re.escape(c)}(?![0-9A-Za-z])", e)]
        if len(hit) != 1:
            raise CutError(f"an engine id carries {'no' if not hit else 'more than one'} client id")
        out[e] = hit[0]
    return out


def controls(recs: list[dict], windows: dict[int, set[str]]) -> list[str]:
    """The cut's refusals the test needs: a request moved to another window, and one dropped from every window."""
    failed = []
    ws = sorted(windows)
    trials = []
    if len(ws) > 1:
        moved = {w: set(v) for w, v in windows.items()}
        rid = sorted(moved[ws[-1]])[0]
        moved[ws[-1]].discard(rid)
        moved[ws[0]].add(rid)
        trials.append(("a request in the wrong window", {w: v for w, v in moved.items() if v}))
    dropped = {w: set(v) for w, v in windows.items()}
    dropped[ws[-1]].discard(sorted(dropped[ws[-1]])[0])
    trials.append(("a request in no window", {w: v for w, v in dropped.items() if v}))
    for name, trial in trials:
        try:
            cut(recs, trial)
        except CutError:
            print(f"advice-c: control, {name}: refused", flush=True)
            continue
        print(f"advice-c: control, {name}: NOT REFUSED", flush=True)
        failed.append(name)
    return failed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", type=Path)
    ap.add_argument("priv", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--controls", action="store_true")
    ap.add_argument("--T", type=int, default=600)
    ap.add_argument("--tau", type=float, default=0.1)
    a = ap.parse_args()
    traces = sorted((a.arm / "server").glob("step_trace.*.jsonl*"))
    if len(traces) != 1:
        raise SystemExit("advice-c: not one step trace under ARM/server (one engine core)")
    recs = records(traces[0])
    spans = boundary(a.arm)
    windows = {w: set(m) for w, m in spans.items()}
    try:
        sliced = cut(recs, windows)
        who = users(a.arm, set().union(*windows.values()))
    except CutError as e:
        print(f"advice-c: the cut does not line up with the boundary windows: {e}", flush=True)
        return 1
    print(f"advice-c: the cut lines up with the boundary windows {', '.join(f'w{w}' for w in sorted(sliced))}", flush=True)
    failed = controls(recs, windows) if a.controls else []
    compiler = ADV.registered_compiler()
    for w in sorted(sliced):
        d = a.priv / f"w{w}"
        d.mkdir(parents=True, exist_ok=True)
        tr, rq = d / "trace.jsonl", d / "requests.jsonl"
        tr.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in sliced[w]))
        rq.write_text("".join(json.dumps({"request": e, "user": who[e], "prompt_ids": list(s.prompt),
                                          "output_ids": list(s.served), "finish": s.finish}, separators=(",", ":")) + "\n"
                              for e, s in spans[w].items()))
        rep = ADV.write_one(a.arm, d, sliced[w], tr, rq, ADV.io_record(rq), T=a.T, tau=a.tau, compiler=compiler)
        if not rep["roundtrip"]:
            print(f"advice-c: w{w}'s advice does not round-trip", flush=True)
            return 1
        binding = json.loads((d / "registration.json").read_text())["binding"]
        o = a.out / "advice" / f"w{w}"
        o.mkdir(parents=True, exist_ok=True)
        (o / "registration.json").write_text(json.dumps({"binding": binding}, indent=1) + "\n")
        print(f"advice-c: w{w}: advice written and round-trips, binding root {binding['advice']['root'][:16]}…", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
