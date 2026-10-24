"""The ingress-ids tie (`io_ids.IoTieRequestIds`) on a sealed served window's real rows and step-input ports.

    python benchmarks/prove_path/ids_tie_window.py --serve-run DIR --window K --out DIR [--circuit]

* **The Commit side** is `served_tie_window`'s: window K's dumped store, each step's port chunks opened with their salts
  and folded to the step's root and the run root, and the prover's served rows registered to the window's record
  (`served_tie_window.bindings`). Each opened step's first KI chunks are the tie's ``ids_<s>_<j>`` rows.
* **The ingress side**: the run has no ingress capture, so the warden's cells are each request's canonical id request
  (`active.id_request`) carrying its served prompt ids, two requests a link-window (T = 2 buckets of R cells of
  4,096 bytes, the own frame first). The windows under the tie are those whose requests the served window prefills.
* **The map** is `io_ids.window_map` from the served rows (each slot's request, start and n), the requests' prompt
  lengths and no prefix-cache source (the run serves with the prefix cache off).
* **The cases:**
  - the honest rows and prompts;
  - ``relabel``: two prefill slots of one step swap request ids, start and n kept, registered consistently. The served
    rows' tie (`served_tie`) binds start and n, so it accepts; the ids tie refuses it in both windows, by the chain
    where the relabelled slot covers its new request's prompt (that request's ids aren't its user's) and by the map
    where it doesn't (a prompt position no step computes);
  - ``wrong_id``: the user sent another id at one prompt position than the one the model computed on (REDTEAM phase 9
    #1), refused by the chain in the window holding that field;
  - ``decode_id``: `served_tie_window`'s ``id`` case, a decode slot's request id changed: the ids tie reads it only
    where its position is a prompt position of its new request; a decode's input is the egress tie's.

``--circuit`` also evaluates the Boolean unit on every window with a map (one batch), against `io_ids.reference`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import served_tie_window as STW  # noqa: E402

from verity.core.protocols.certifier import io_ids as I, io_request as Q, served_rows as SR, served_tie as ST  # noqa: E402
from verity.core.protocols.certifier.active import id_request, id_request_frames  # noqa: E402

B = 4096
PER_WINDOW = 2


def link(prompts: list[list[int]], per_window: int = PER_WINDOW) -> tuple[Q.Shape, list[list[bytes]], tuple[int, ...]]:
    """The ingress link of the requests (arrival order): `per_window` requests a window, each window's frames in its
    two buckets (the own frame first), R the fewest cells that hold every window's."""
    frames = [id_request_frames(len(p), B) for p in prompts]
    per = [sum(frames[k:k + per_window]) for k in range(0, len(prompts), per_window)]
    R = max(2, max(-(-(f + 1) // 2) for f in per))
    counts = [c for f in per for c in (min(R - 1, f), f - min(R - 1, f))]
    shape = Q.Shape(B, 2, R, 1)
    reqs = [id_request(model="served", user=f"u{q}", prompt=p, max_tokens=16, seed=q, frame_bytes=B)
            for q, p in enumerate(prompts)]
    return shape, Q.windows_of(shape, reqs, counts), tuple(per)


def served_steps(window: list) -> list[list[I.Row]]:
    return [[I.Row(req, start, n) for _slot, req, _kind, start, n in rows or ()] for rows in window]


def targets(window: list, per_window: int = PER_WINDOW) -> list[int]:
    """The link-windows of the requests the served window prefills."""
    return sorted({req // per_window for rows in window for _s, req, kind, _st, _n in rows or () if kind == 0})


def cases(window: list, prompts: list[list[int]], per_window: int = PER_WINDOW) -> tuple[dict, dict]:
    """Each case's (served rows, prompts as the users sent them), and what each changed."""
    pre = [(u, k, r) for u, rows in enumerate(window) for k, r in enumerate(rows or ()) if r[2] == 0]
    (u, a, ra), (_, b, rb) = next((x, y) for x in pre for y in pre
                                  if x[0] == y[0] and x[1] < y[1] and x[2][1] // per_window != y[2][1] // per_window)
    relabel = STW._edit(STW._edit(window, u, a, request=rb[1]), u, b, request=ra[1])
    q = min(r[1] for _u, _k, r in pre)
    pos = min(5, len(prompts[q]) - 1)
    sent = [list(p) for p in prompts]
    sent[q][pos] ^= 1
    _table, served = STW.cases(window)
    dec, k, row = served["decode"]
    decode = STW._edit(window, dec, k, request=window[dec][1][1])
    return ({"honest": (window, prompts), "relabel": (relabel, prompts), "wrong_id": (window, sent),
             "decode_id": (decode, prompts)},
            {"relabel": {"step": u, "slots": [a, b], "rows": [list(ra), list(rb)]}, "wrong_id": {"request": q, "position": pos},
             "decode_id": {"step": dec, "slot": k, "row": list(row), "request": window[dec][1][1]}})


def run_cases(S, St, window: list, steps: list[SimpleNamespace], prompts: list[list[int]], circuit: bool = False
              ) -> SimpleNamespace:
    """Each case's verdicts: the served rows' tie (`served_tie.check` on the window's unit) and, in each target
    link-window, the ids tie's map (or its refusal) and `io_ids.reference` (and the circuit's, with `circuit`). The
    unit opens S steps, the window's prefill steps and one more (a step a relabel makes a prompt's)."""
    PL = [len(p) for p in prompts]
    table, edits = cases(window, prompts)
    wins = targets(window)
    S_open = 1 + len({u for u, rows in enumerate(window) for r in rows or () if r[2] == 0})
    out, lanes = {}, []
    for name, (win, sent) in table.items():
        rq, cells, received = link(sent)
        sh = I.Shape(rq.B, rq.T, rq.R, rq.OWN, St.T, S_open, St.IB)
        tie = ST.check(St, STW.unit_rows(St, SR.window_bits(S, win), steps))
        res = {"served_tie": "accepted" if all(tie.values()) else f"refused ({', '.join(k for k, v in tie.items() if not v)})",
               "windows": {}}
        for w in wins:
            view = Q.request_view(PL, received, w, B)
            try:
                mp = I.window_map(sh, PL, view, served_steps(win), {})
            except ValueError as e:
                res["windows"][w] = {"map": f"refused: {e}"}
                continue
            inst = I.Instance(Q.instance(rq, cells[w], view, w), mp, tuple(tuple(steps[s].chunks[:sh.KI]) for s in mp.steps))
            res["windows"][w] = {"map": "ok", "steps": list(mp.steps), "reference": I.reference(sh, inst)}
            lanes.append((name, w, sh, inst))
        out[name] = res
    ands = None
    if circuit and lanes:
        ands = _circuit(out, lanes)
    for res in out.values():
        res["ids_tie"] = ("accepted" if all(v.get("reference") == 1 and v.get("circuit", 1) == 1 for v in res["windows"].values())
                          else "refused in window " + ", ".join(str(w) for w, v in res["windows"].items()
                                                                 if v.get("reference") != 1 or v.get("circuit", 1) != 1))
    expect = {"honest": "accepted", "relabel": "refused", "wrong_id": "refused"}
    passed = (out["honest"]["served_tie"] == "accepted" and out["relabel"]["served_tie"] == "accepted"
              and all(out[n]["ids_tie"].startswith(v) for n, v in expect.items())
              and all(v.get("circuit", v.get("reference")) == v.get("reference") for r in out.values()
                      for v in r["windows"].values()))
    return SimpleNamespace(cases=out, edits=edits, windows=wins, passed=passed, ands=ands)


def _circuit(out: dict, lanes: list) -> int:
    import numpy as np
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluation.bits import evaluate_bits
    from verity.core.protocols.certifier import head_check as HC

    sh = lanes[0][2]
    fn = bind(I.IoTieRequestIds, B=sh.B, T=sh.T, R=sh.R, OWN=sh.OWN, P=sh.P, S=sh.S, IB=sh.IB)
    args = [I.arguments(s, i, Q.switches(s.request, i.request), I.route_switches(s, i), I.sort_switches(s, i))
            for _n, _w, s, i in lanes]
    got = np.asarray(evaluate_bits(fn, *[np.asarray([a[k] for a in args], dtype=np.uint8) for k in range(len(args[0]))]))
    for (name, w, _s, _i), v in zip(lanes, got.reshape(-1).tolist()):
        out[name]["windows"][w]["circuit"] = int(v)
    return HC.ands(fn)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--serve-run", type=Path, required=True)
    ap.add_argument("--window", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--circuit", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.monotonic()
    x = STW.load(a.serve_run, a.window)
    S = SimpleNamespace(**x.priv["statics"])
    port = x.ctx["served_window"]["step_input_port"]
    St = SimpleNamespace(**ST.statics(S, IB=port["IB"], PB=port["PB"]))
    ok, steps, native = STW.bindings(x, S, St)
    (window,) = STW._session(x.priv)
    window = list(window) + [None] * (len(steps) - len(window))
    reqs = sorted((r for r in x.reqs if r.get("arrival") is not None), key=lambda r: int(r["arrival"]))
    if [int(r["arrival"]) for r in reqs] != list(range(len(reqs))):
        raise SystemExit("the run's requests aren't arrivals 0, 1, …")
    prompts = [list(map(int, r["prompt_ids"])) for r in reqs]
    t1 = time.monotonic()
    run = run_cases(S, St, window, steps, prompts, circuit=a.circuit)
    t2 = time.monotonic()
    passed = all(ok.values()) and run.passed
    result = {"window": a.window, "run_root": x.run.run_root.hex(), "served_rows_root": x.rec["served_rows"]["root"],
              "port": port, "bindings": ok, "native": native, "link_windows": run.windows, "edits": run.edits,
              "cases": run.cases, "passed": passed, "ands": run.ands,
              "prefix_sources": sorted({json.dumps(r["check"].get("prefix_source")) for r in reqs}),
              "seconds": {"open": round(t1 - t0, 1), "cases": round(t2 - t1, 1)}}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("bindings", "native", "link_windows", "edits", "cases", "ands", "seconds",
                                             "passed")}, indent=1))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
