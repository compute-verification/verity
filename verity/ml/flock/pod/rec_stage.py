"""Staging `UniversalUnit_v1{G, N_IN, N_OUT}` as the recursion's inner statement at the size of a real private circuit
(``88-rec-stage.sh``).

    python rec_stage.py measure --cls G,N_IN,N_OUT --n N --out DIR [--seed S] [--profile FILE]
        today's path (`rec_inner.universal`'s stage half: bind and trace, the partition's units and class, the lanes,
        `class_statement.stage`), one JSON line per phase: its seconds and the process's peak RSS within it
    python rec_stage.py fast --cls G,N_IN,N_OUT --n N --out DIR [--seed S] [--netlist FILE] [--profile FILE]
        `rec_inner.universal --fast` (`verity_flock.universal_stage`): DIR/stage and DIR/other as 87-rec-private.sh stages
        them, under a fresh salt key, phase by phase as ``measure``; RESULT: universal's record, the phases, and the staged
        size beside the class's closed forms
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rec_inner as RI  # noqa: E402

SALT_KEY = bytes(range(32))


def _rss_reset() -> None:
    try:
        Path("/proc/self/clear_refs").write_text("5")
    except OSError:
        pass


def _hwm_kib() -> int:
    for ln in Path("/proc/self/status").read_text().splitlines():
        if ln.startswith("VmHWM:"):
            return int(ln.split()[1])
    return -1


class Phases:
    """Each phase's seconds and the peak RSS within it (VmHWM reset through ``clear_refs`` at its start); a phase nested in
    another is counted in both, its own line naming its parent."""

    def __init__(self, out):
        self.out, self.stack, self.rows = out, [], []

    @contextmanager
    def __call__(self, name: str):
        _rss_reset()
        t0 = time.monotonic()
        self.stack.append(name)
        try:
            yield
        finally:
            self.stack.pop()
            row = {"phase": name, "parent": self.stack[-1] if self.stack else None, "s": round(time.monotonic() - t0, 3),
                   "peak_rss_gib": round(_hwm_kib() / 2**20, 3)}
            self.rows.append(row)
            print(json.dumps(row), file=self.out, flush=True)

    def wrap(self, owner, attr: str, name: str):
        f = getattr(owner, attr)

        def g(*a, **k):
            with self(name):
                return f(*a, **k)

        setattr(owner, attr, g)


def measure(cls: str, n: int, out: Path, seed: int = 0, log=sys.stdout) -> dict:
    """`rec_inner.universal`'s staging of the registered netlist (not the other), phase by phase, the salt key fixed."""
    B = RI._benchmark()
    from verity_flock import circuit as FC
    from verity_flock import class_statement as CS
    from verity_flock import ir_lower as IL
    from verity_flock import partition_units as PU

    from verity.ml.circuits import partition_object as PO
    from verity.ml.circuits.defs import bind
    from verity.ml.private_circuits import universal as U

    ph = Phases(log)
    G, N_IN, N_OUT = (int(v) for v in cls.split(","))
    rng = random.Random(f"rec-private/{seed}/{G}/{N_IN}/{N_OUT}")
    net = U.pad(B._netlist(rng, G, N_IN, N_OUT), G, N_IN)
    xs = [[rng.randrange(2) for _ in range(N_IN)] for _ in range(n)]
    for owner, attr, name in ((CS, "class_lowering", "stage/lowering"), (CS, "pack", "stage/pack"),
                              (IL.Lowering, "evaluate", "stage/evaluate"), (FC, "compose", "stage/compose"),
                              (FC, "write", "stage/write"), (CS, "partition_binding", "stage/partition_binding")):
        ph.wrap(owner, attr, name)
    with ph("total"):
        with ph("trace"):
            fn = bind(U.UniversalUnit, G=G, N_IN=N_IN, N_OUT=N_OUT)
            tr = U.traced(G, N_IN, N_OUT)
        with ph("units"):
            program = PU.one_call(fn)
            query = PO.template_instance_query(fn)
            units = PU.PartitionUnits(program, query=query)
            [(shape, cl)] = units.classes().items()
        with ph("lanes"):
            ins, outs, idents = B.lanes(units, cl, net, xs)
        with ph("stage"):
            st = CS.stage(units, B.SPEC, shape, cl, (ins, outs, idents), out, SALT_KEY, packed=True)
        with ph("partition"):
            (out / "partition.json").write_text(json.dumps(PO.build(program, query), sort_keys=True))
    want = [U.interpret(net, x) for x in xs]
    return {"class": [G, N_IN, N_OUT], "n": n, "ands": U.and_count(G, N_IN, N_OUT), "traced_ands": getattr(tr, "ands", None),
            "outputs_agree": [[outs[j][t] for j in range(N_OUT)] for t in range(n)] == want, "phases": ph.rows} | \
        {k: st[k] for k in ("k_log", "vus_per_block", "unit_rows", "unit_nnz", "circuit_bytes", "ands", "circuit_sha512")}


def fast(cls: str, n: int, out: Path, seed: int = 0, netlist: str | None = None, log=sys.stdout) -> dict:
    """`rec_inner.universal --fast`'s two statements, phase by phase."""
    RI._benchmark()
    from verity_flock import circuit as FC
    from verity_flock import universal_stage as US

    from verity.ml.private_circuits import universal as U

    ph = Phases(log)
    for owner, attr, name in ((US, "trace", "trace"), (US, "program", "program"), (US, "unit", "unit"), (US, "rows_of", "stage/rows"),
                              (US, "lower", "stage/lower"), (FC, "compose", "stage/compose"), (FC, "write", "stage/write")):
        ph.wrap(owner, attr, name)
    with ph("total"):
        res = RI.universal(cls, n, out, seed, Path(netlist) if netlist else None, fast=True)
    G, N_IN, N_OUT = (int(v) for v in cls.split(","))
    K = U.index_bits(G, N_IN)
    st = res["stage"]
    size = {"ands": st["ands"], "ands_closed_form": U.and_count(G, N_IN, N_OUT), "unit_rows": st["unit_rows"],
            "unit_nnz": st["unit_nnz"], "program_bits": 2 * G + 2 * G * K + N_OUT * K,
            "statement_bits": st["n"] * (1 << st["k_log"]) // st["vus_per_block"], "circuit_bytes": st["circuit_bytes"]}
    return {"universal": {k: v for k, v in res.items() if k not in ("stage", "other")} |
            {nm: {k: v for k, v in res[nm].items() if k != "outputs"} for nm in ("stage", "other")},
            "outputs_agree_interpret": True, "size": size, "phases": ph.rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("measure", "fast"):
        m = sub.add_parser(cmd)
        m.add_argument("--cls", required=True)
        m.add_argument("--n", type=int, required=True)
        m.add_argument("--out", required=True)
        m.add_argument("--seed", type=int, default=0)
        m.add_argument("--profile")
        if cmd == "fast":
            m.add_argument("--netlist")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    f, args = (measure, (a.cls, a.n, out, a.seed)) if a.cmd == "measure" else (fast, (a.cls, a.n, out, a.seed, a.netlist))
    if a.profile:
        import cProfile

        pr = cProfile.Profile()
        res = pr.runcall(f, *args)
        pr.dump_stats(a.profile)
    else:
        res = f(*args)
    print("RESULT " + json.dumps(res), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
