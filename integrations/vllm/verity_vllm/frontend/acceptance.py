"""Write the acceptance record of the torch frontend prototype (`docs/data/frontend/`).

    python -m verity_vllm.frontend.acceptance docs/data/frontend
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import torch

from verity.ir import Program, bind, encode_program, program_digest, descriptor_size, query
from tests.ir.test_running_example import Add16, Id16, Mul16
from verity_vllm.frontend import FrontendError, export
from verity_vllm.frontend import examples as ex


def main(out_dir: str) -> int:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rec: dict = {"schema": "verity-ir/frontend-acceptance/v0", "torch": torch.__version__, "python": platform.python_version(),
                 "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                 "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "digest_equality": [], "identity": {}, "negatives": []}
    ok = True
    for label, front, ta1 in ex.ta1_pairs():
        t0 = time.time()
        P1, P2 = Program(front.specialize()), Program(ta1)
        d1, d2 = program_digest(P1), program_digest(P2)
        row = {"definition": label, "frontend_digest": d1, "ta1_digest": d2, "equal": d1 == d2, "gates": P1.gates,
               "input_gates": P1.input_gates, "descriptor_bytes": descriptor_size(P1),
               "definitions": sorted(encode_program(P1)["definitions"]), "seconds": round(time.time() - t0, 3)}
        ok &= row["equal"]
        rec["digest_equality"].append(row)
        print(f"{'OK ' if row['equal'] else 'NEQ'} {label:36s} gates={P1.gates:>8} digest={d1[:16]} ta1={d2[:16]} ({row['seconds']}s)")

    Top, TopIR = ex.identity_pair(Add16, Mul16, Id16)
    P = export(Top, (torch.empty((), dtype=torch.bfloat16, device="meta"),))
    P_ir = Program(bind(TopIR))
    f = query(P.circuit, "//F")[0]
    rec["identity"] = {"program": "Top(v) = Id16(F(v, v)[0]); F(a, b) = (Add16(a, a), a, Add16(a, a), Mul16(a, b))",
                       "frontend_digest": program_digest(P), "hand_built_digest": program_digest(P_ir),
                       "equal": program_digest(P) == program_digest(P_ir),
                       "F.args": list(f.args), "F.returns": list(f.returns), "F.in": list(f.in_),
                       "F.distinct_returns (declared adapter)": list(f.distinct_returns), "F.out (graph boundary)": list(f.out),
                       "gates": [repr(g) for g in P.circuit[0:P.gates]]}
    ok &= (rec["identity"]["equal"] and f.args == (0, 0) and f.returns == (1, 0, 1, 2) and f.in_ == (0,)
           and f.distinct_returns == (1, 2) and f.out == (1,))
    print("identity:", "OK" if rec["identity"]["equal"] else "NEQ", f.args, f.returns, f.in_, f.distinct_returns, f.out)

    for label, fn, code in ex.NEGATIVES:
        try:
            export(fn, ex.NEG_INPUTS)
            row = {"case": label, "expected": code, "got": None, "message": "ACCEPTED (bug)"}
            ok = False
        except FrontendError as e:
            row = {"case": label, "expected": code, "got": e.code, "message": e.message, "where": e.where}
            ok &= e.code == code
        rec["negatives"].append(row)
        print(f"{'OK ' if row['got'] == code else 'BAD'} {code:24s} {label}: {row['message'][:110]}")

    rec["all_passed"] = ok
    (out / "torch-frontend-acceptance.json").write_text(json.dumps(rec, indent=1))
    print("all_passed" if ok else "FAILED", "->", out / "torch-frontend-acceptance.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "docs/data/frontend"))
