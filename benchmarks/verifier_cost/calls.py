"""`calls.sh`'s Python half: one verifier call's cost on a large program, step by step. It builds `population.py`'s
population of N units (`--pad K` sizes its descriptor), a valid v1 registration of it under `subset:3` (the difftest's
ports, roots and window), and times the Python reference (`registration.check`), the program's digest
(`codec.program_digest`) and encoding (`codec.encode_program`), the Lean's parameters (`verifier._params`), the request
line carrying the descriptor (`verifier._dumps`), the Lean's answer through `sampled-proofs-verifier` (`verifier._server`)
`--calls` times in one process (the first loads the program, the rest name it by digest), the same request carrying the
whole descriptor once, and `verifier.check` whole, `--calls` times. It writes the request line to DIR/registration.line
for `calls/Main.lean`, and one JSON row per step on stdout (seconds, the Python's peak RSS, the Lean server's peak and
current RSS).

  python calls.py DIR N [--pad K] [--calls C] [--roots R] [--templates T] [--levels L] [--width W]
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "verity/core/service/sampled_proofs/tests"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import population as POP  # noqa: E402
import test_difftest as T  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import partition as P, registration as R  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import verifier as V  # noqa: E402
from verity.ml.circuits import codec, partition_object as PO  # noqa: E402


def _status_mb(pid: int, field: str) -> int:
    status = Path(f"/proc/{pid}/status")
    for line in (status.read_text().splitlines() if status.exists() else ()):
        if line.startswith(field):
            return int(line.split()[1]) // 1024
    return 0


def row(step: str, s: float, **extra) -> None:
    proc = V._server._proc
    alive = proc is not None and proc.poll() is None
    print(json.dumps({"step": step, "s": round(s, 3),
                      "python_hwm_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024,
                      "lean_hwm_mb": _status_mb(proc.pid, "VmHWM:") if alive else 0,
                      "lean_rss_mb": _status_mb(proc.pid, "VmRSS:") if alive else 0, **extra}), flush=True)


def timed(f):
    t = time.perf_counter()
    out = f()
    return out, time.perf_counter() - t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("n", type=int)
    ap.add_argument("--pad", type=int, default=0)
    ap.add_argument("--calls", type=int, default=2)
    ap.add_argument("--roots", type=int, default=234)
    ap.add_argument("--templates", type=int, default=234)
    ap.add_argument("--levels", type=int, default=6)
    ap.add_argument("--width", type=int, default=1800)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    (program, query), s = timed(lambda: POP.population(a.n, a.roots, a.templates, a.levels, a.width, a.pad))
    row("build", s, units=a.n)
    sha, s = timed(lambda: PO.program_sha512(program))
    row("program_sha512", s)
    part, s = timed(lambda: P.partition(program, query))
    population = P.units(program, query)
    row("partition", s, population=population)
    ports = {p: {"domain": R.window_domain(sha, P.digest(part), p, schema, leaves, "r-test"), "leaves": leaves}
             for p, schema, leaves in (("out", "u16", 2 * population), ("x", "hm96-sha512/row/v1", population))}
    law = {"k": 3, "law": "subset"}
    rec = R.record(program, query, law, T.SCHEME, [{"port": p, "root": T.ROOTS[p], **ports[p]} for p in ports],
                   T.WINDOW, leaf_layer=T.LEAF)
    py, s = timed(lambda: R.check(rec, program, query, law, T.SCHEME, ports, T.LEAF))
    row("python_check", s, ok=py.ok, codes=list(py.codes))
    digest, s = timed(lambda: codec.program_digest(program))
    row("program_digest", s, same=digest == sha)
    desc, s = timed(lambda: codec.encode_program(program))
    row("encode_program", s, definitions=len(desc["definitions"]))
    params, s = timed(lambda: V._params(program, query, law, T.SCHEME, ports, T.LEAF, (), [], None, None, None))
    row("params", s)
    value = {"params": params, "record": dict(rec)}
    line, s = timed(lambda: V._dumps({"fn": "registration", "input": value}))
    (a.out / "registration.line").write_text(line + "\n")
    row("dumps", s, bytes=len(line))
    for k in range(a.calls):
        got, s = timed(lambda: V._server.call("registration", value))
        row("lean_call", s, call=k, answer=got, loaded=len(V._server._loaded))
    whole = {"params": {**params, "program": desc}, "record": dict(rec)}
    got, s = timed(lambda: V._server.call("registration", whole))
    row("lean_call_whole", s, answer=got)
    for k in range(a.calls):
        v, s = timed(lambda: V.check(rec, program, query, law, T.SCHEME, ports, T.LEAF))
        row("check", s, call=k, ok=v.ok, codes=list(v.codes), verifiers=v.verifiers)


if __name__ == "__main__":
    main()
