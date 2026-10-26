"""The served rows' tie (`served_tie.ServedRowsTie_v2`) on a sealed served window's real rows.

    python benchmarks/prove_path/served_tie_window.py --serve-run DIR --window K --out DIR

* **The window** is serve-openai's sealed window K of the run DIR: its dumped store
  (``windows/wK/commit/replay_bundle_p0/store``), its served-rows record (``windows.jsonl``), the prover's served rows and
  their salts (``served_rows/wK.json``), and the run's requests (``requests.jsonl``).
* **The bindings**, native, as the verifier holds them:
  - the context's step-input port is the tie's: ``step.input/ids``, then ``step.input/positions``, each T i64s and each the
    first leaves of every step's stream;
  - each step's port chunks, opened with their salts, fold to the step's root and the run root
    (``DumpedStore.verify_range``);
  - the served-rows record is the window's under that run root, and the prover's rows and salts register to it
    (`served_rows.register`).

  A step's N is its layout's length of the embedding's input (the record's ``ids_name``).
* **The tie.** `served_tie.check` runs on the window's whole unit (G = P steps, the idle ones with N = 0). The circuit runs
  on each executed step as its own block (G = 1, block j the step, the switches the prover's), and those verdicts AND to
  the whole unit's, because the unit is P independent steps. The whole unit is P steps' ANDs and their AND (``size``).
* **The cases:**
  - the honest rows;
  - three with one field altered, each registered consistently: a prefill's start one on, a prefill one token short, and
    a decode row's request id changed to another arrival's;
  - one with a row of another request in place of one.

  ``ids`` is IoTieRequest's statement in integers: each token's committed input id is its request's prompt-and-output id
  at its position, read through `served_tie.ids`. The tie binds start and n, so the id is ``ids``' to refuse.

The port is a stream tensor of its own. The embedding reads ``model.embed_tokens/in``, a second copy, until vllm aliases the
port's ranges as the embedding's and the rotary's inputs. So the tie binds the rows to the stream's port chunks, not yet
to what the embedding read. ``embed_equal`` reports, natively, whether the two copies agree on this run.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from verity.core.protocols.certifier import served_rows as SR
from verity.ml.protocols.certifier import served_statics as SS, served_tie as ST

PORT_NAMES = ("step.input/ids", "step.input/positions")
RECORD_KEYS = ("schema", "binding", "root", "count", "n_bits")


def load(run_dir: Path, w: int) -> SimpleNamespace:
    from verity_vllm.commit import store_dump

    bundle = run_dir / "windows" / f"w{w}" / "commit" / "replay_bundle_p0"
    rec = next(r for r in map(json.loads, (run_dir / "windows.jsonl").read_text().splitlines()) if r["window"] == w)
    store, run = store_dump.load(bundle / "store")
    return SimpleNamespace(ctx=json.loads((bundle / "context.json").read_text()), rec=rec, store=store, run=run,
                           priv=json.loads((run_dir / "served_rows" / f"w{w}.json").read_text()),
                           reqs=[json.loads(l) for l in (run_dir / "requests.jsonl").read_text().splitlines() if l.strip()])


def statics_of(x: SimpleNamespace) -> SimpleNamespace:
    """The run's compile-slice statics. A run sealed before the cache's statics served without prefix caching, so its CN,
    CTW and CU are 0; one sealed before the replay's async statics committed its rows without the async group or walk
    entries, whatever its engine's async_scheduling, so its AS and EW are 0 (`bindings` recomputes the run's root)."""
    return SimpleNamespace(**{"CN": 0, "CTW": 0, "CU": 0, "AS": 0, "EW": 0, **x.priv["statics"]})


def _session(priv: dict) -> list[list[tuple | None]]:
    return [[None if s is None else tuple(map(tuple, s)) for s in win] for win in priv["session"]]


def _ints(chunks: list[bytes], read, St) -> list[int]:
    return [ST._int(v) for v in read(St, [ST._bits(c) for c in chunks])]


def bindings(x: SimpleNamespace, S, St) -> tuple[dict, list[SimpleNamespace], dict]:
    """The verifier's native checks, and each executed step's N, port chunks and embedding input."""
    import numpy as np

    C, G = ST.CHUNK, ST.geometry(St)
    T, IB, PB = St.T, St.IB, St.PB
    sw, record, run_root = x.ctx["served_window"], x.rec["served_rows"], x.run.run_root
    ok = {"port": sw["step_input_port"] == {"names": list(PORT_NAMES), "T": T, "IB": IB, "PB": PB, "ids_bytes": [0, T * IB],
                                             "positions_bytes": [T * IB, T * (IB + PB)]},
          "run_root": run_root.hex() == x.rec["run_root"] == x.ctx["run_root"],
          "step_roots": [s.root.hex() for s in x.run.steps] == x.rec["step_roots"],
          "record": all(record[k] == sw["served_rows"][k] for k in RECORD_KEYS),
          "binding": record["binding"] == SR.binding(run_root).hex()}
    served = SR.register(S, _session(x.priv), run_root, salts=[bytes.fromhex(s) for s in x.priv["salts"]])
    ok["registered"] = all(served.record()[k] == record[k] for k in RECORD_KEYS)
    layout = opened = embed_equal = padded = True
    roots: dict[int, bytes] = {}
    steps = []
    for u in range(len(x.run.steps)):
        metas = x.store._layouts[u]
        layout &= [(m.name, m.stream_off, m.first_leaf, m.n_leaves) for m in metas[:2]] == [
            (PORT_NAMES[0], 0, 0, T * IB // C), (PORT_NAMES[1], T * IB, T * IB // C, T * PB // C)]
        ro = x.store.open_range(u, 0, G.hi)
        opened &= ro is not None and x.store.verify_range(x.run, ro, roots)
        chunks = [ro.value[k * C:(k + 1) * C] for k in range(G.hi)]
        emb = next(m for m in metas if m.name == x.rec["ids_name"])
        er = x.store.open_range(u, emb.first_leaf, emb.first_leaf + emb.n_leaves)
        opened &= er is not None and x.store.verify_range(x.run, er, roots)
        N = int(emb.shape[0])
        embed = np.frombuffer(er.value[:emb.nbytes], dtype=np.dtype(str(emb.dtype).rsplit(".", 1)[-1])).astype(np.int64).tolist()
        ids, pos = _ints(chunks[:-(-T * IB // C)], ST.ids, St), _ints(chunks[G.lo:G.hi], ST.positions, St)
        embed_equal &= ids[:N] == embed
        padded &= not any(ids[N:]) and not any(pos[N:])
        steps.append(SimpleNamespace(N=N, chunks=chunks, ids=ids, positions=pos))
    ok.update(layout=layout, opened=opened)
    return ok, steps, {"embed_equal": embed_equal, "zero_past_N": padded}


def statics_held(window: list, steps: list[SimpleNamespace]) -> bool:
    """Whether the Programs' statics the window's rows describe (`served_statics.program_of`: each step's rows in batch
    order as (start, n)) are the positions its step's port committed: its N tokens at start … start + n − 1, in turn."""
    for prog, s in zip(SS.program_of(window), steps):
        if [start + t for start, n in prog or () for t in range(n)] != s.positions[:s.N]:
            return False
    return True


def ids_ok(St, window: list, steps: list[SimpleNamespace], seqs: dict[int, list[int]]) -> bool:
    """IoTieRequest's statement in integers: slot k's tokens [lo, lo + n) carry its request's ids at [start, start + n)."""
    for u, rows in enumerate(window):
        lo = 0
        for _slot, req, _kind, start, n in rows or ():
            seq = seqs.get(req, [])
            if any(start + i >= len(seq) or steps[u].ids[lo + i] != seq[start + i] for i in range(n)):
                return False
            lo += n
    return True


def _edit(window: list, u: int, k: int, **field) -> list:
    at = {"request": 1, "start": 3, "n": 4}
    rows = [list(r) for r in window[u]]
    for name, v in field.items():
        rows[k][at[name]] = v
    return [tuple(map(tuple, rows)) if i == u else s for i, s in enumerate(window)]


def cases(window: list) -> tuple[dict[str, tuple[list, str]], dict]:
    """The honest window and four rows changed, each with what refuses it, and the rows changed. The decode step is the
    first of two or more decodes: its slot 0 takes slot 1's request id (``id``), or slot 1's request's row of a later
    step (``other``)."""
    pre = next(u for u, s in enumerate(window) if s and any(r[2] == 0 for r in s))
    k = next(i for i, r in enumerate(window[pre]) if r[2] == 0)
    dec = next(u for u, s in enumerate(window) if s and len(s) > 1 and all(r[2] == 1 for r in s))
    b = window[dec][1]
    later = next(r for s in window[dec + 1:] if s for r in s if r[1] == b[1])
    table = {"honest": (window, "accepted"),
             "start": (_edit(window, pre, k, start=window[pre][k][3] + 1), "tie"),
             "short": (_edit(window, pre, k, n=window[pre][k][4] - 1), "tie"),
             "id": (_edit(window, dec, 0, request=b[1]), "ids"),
             "other": (_edit(window, dec, 0, request=later[1], start=later[3], n=later[4]), "tie")}
    return table, {"prefill": [pre, k, list(window[pre][k])], "decode": [dec, 0, list(window[dec][0])], "other_row": list(later)}


def run_cases(S, st: dict, window: list, steps: list[SimpleNamespace], seqs: dict[int, list[int]]) -> SimpleNamespace:
    """Each case's verdicts: `served_tie.check` on the window's whole unit, the circuit on every executed step as its own
    block, a G = 1 unit (ANDed, and each step's against its own `check`), and ``ids``."""
    St, st1 = SimpleNamespace(**{**st, "G": st["P"]}), {**st, "G": 1}
    S1 = SimpleNamespace(**st1)
    port = [(s.N, s.chunks) for s in steps]
    table, edits = cases(window)
    out, lanes, keys = {}, [], []
    for name, (win, _by) in table.items():
        bits = SR.window_bits(S, win)
        ref = ST.check(St, ST.unit_rows(St, bits, port))
        for u in range(len(steps)):
            r1 = ST.unit_rows(S1, bits, port, u)
            f1 = ST.check(S1, r1)
            lanes.append(ST.Case(r1, f1, ST.verdict(f1)))
            keys.append(name)
        out[name] = {"reference": ST.verdict(ref), "flags": ref, "ids": ids_ok(St, win, steps, seqs)}
    got = ST.evaluate(st1, lanes)
    for name in out:
        per = [v for n, v in zip(keys, got) if n == name]
        out[name].update(circuit=_and(per), circuit_matches=per == [c.verdict for n, c in zip(keys, lanes) if n == name])
    expect = {name: by for name, (_w, by) in table.items()}
    verdicts = {name: _said(r["circuit"] == ST.ACCEPT, r["ids"]) for name, r in out.items()}
    passed = (all(r["circuit_matches"] and r["circuit"] == r["reference"] for r in out.values())
              and all(verdicts[n] == "accepted" if by == "accepted" else by in verdicts[n] for n, by in expect.items()))
    return SimpleNamespace(cases=out, verdicts=verdicts, expected=expect, edits=edits, passed=passed)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--serve-run", type=Path, required=True)
    ap.add_argument("--window", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    t0 = time.monotonic()
    x = load(a.serve_run, a.window)
    S = statics_of(x)
    st = ST.statics(S, IB=x.ctx["served_window"]["step_input_port"]["IB"], PB=x.ctx["served_window"]["step_input_port"]["PB"])
    St = SimpleNamespace(**st)
    ok, steps, native = bindings(x, S, St)
    (window,) = _session(x.priv)
    window = list(window) + [None] * (len(steps) - len(window))
    seqs = {int(r["arrival"]): list(r["prompt_ids"]) + list(r["streamed_ids"]) for r in x.reqs if r.get("arrival") is not None}
    t1 = time.monotonic()
    run = run_cases(S, st, window, steps, seqs)
    t2 = time.monotonic()
    passed = all(ok.values()) and run.passed
    result = {"window": a.window, "run_root": x.run.run_root.hex(), "served_rows_root": x.rec["served_rows"]["root"],
              "statics": st, "steps": len(steps), "N": [s.N for s in steps], "bindings": ok, "native": native,
              "edits": run.edits, "cases": run.cases, "verdicts": run.verdicts, "expected": run.expected, "passed": passed,
              "ands": {"step": ST.step_size(**st)["ands"], "window": ST.size(**st)["ands"]},
              "seconds": {"open": round(t1 - t0, 1), "cases": round(t2 - t1, 1)}}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("bindings", "native", "verdicts", "ands", "seconds", "passed")}, indent=1))
    return 0 if passed else 1


def _and(vs: list[int]) -> int:
    out = ST.ACCEPT
    for v in vs:
        out &= v
    return out


def _said(tie: bool, ids: bool) -> str:
    if tie and ids:
        return "accepted"
    return "refused by " + " and ".join(w for w, held in (("tie", tie), ("ids", ids)) if not held)


if __name__ == "__main__":
    sys.exit(main())
