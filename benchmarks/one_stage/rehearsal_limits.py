"""The inference-only policy's Lean check on a window class's own Program, before any serve: the limits a serve of that
class is run at (`rehearsal.sh check`).

    rehearsal_limits.py SERVE_PHI CLASS SCHEME POLICY OUT [--rejects]

Builds the class's circuit from phi (as `prove_path.py class-circuit` and `inference_only.py policy` do), takes the
verifier's population Program over its templates and registered values (`Circuit.program`), makes the inference-only
case from it as `served_cases.inference_only_case` does from a window's audit, and asks the Lean check (the policy's,
built here, in a scope of its own as `served_cases.lean_answer` runs every check):

- at no budget (`served_cases.UNBOUNDED`): its profile, each block's gates and input bits in the check's own count;
  gamma is the largest block's gates and alpha the most input bits a gate of any block (`served_cases.division_policy`);
- at those numbers: accept expected. They are the class's own two numbers: a window of the class is inside them by
  construction, so the slowdown they prove is 1, that is none;
- where POLICY is given, the record form's pair beside them (``held``), from the class's numbers alone: a block a slot, so
  gamma is the profile's over the class's slots, refused unless the slots divide it, and alpha is the policy's k fresh
  input bits a block over that, slots × k over the profile's gamma. No Lean call makes it and no check here reads it;
- with ``--rejects``, at POLICY's alpha = k/c and gamma = c (the count rule's numbers, in Boolean gates) and at alpha = k
  over the profile's gamma: the verdict and its reason at each.

The case's audit record is a stand-in naming that Program, the one the suite's class test makes (`test_served_cases.py`'s
``v1_case``): the check reads the Program and its division, and nothing of a served window. OUT gets one JSON object,
written after the Program's digest and again after the profile, so a later failure keeps both. Nothing here reads a run.
"""
import importlib.util
import json
import resource
import sys
import tempfile
import time
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "adapters/vllm"), str(HERE), str(ROOT / "benchmarks/prove_path")]


def main(argv: list[str]) -> int:
    phi, cls_arg, scheme, policy, out = argv[:5]
    rejects = "--rejects" in argv[5:]

    import prove_path as PP
    from verity.circuits import codec
    from verity.circuits.types import type_from_json
    from verity_vllm.program.predict.window_circuit import class_circuit
    from verity_vllm.program.predict.window_class import PearlTables

    spec = importlib.util.spec_from_file_location("served_cases_tests", HERE / "tests" / "test_served_cases.py")
    T = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = T
    spec.loader.exec_module(T)
    SC = T.SC

    doc: dict = {"class": cls_arg, "scheme": scheme or None, "seconds": {}}

    def timed(name, f):
        t = time.time()
        got = f()
        doc["seconds"][name] = round(time.time() - t, 1)
        return got

    def write() -> None:
        Path(out).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")

    registered = PP.registered_of_serve(json.loads(Path(phi).read_text()), None, None, PP.CAP_BITS)
    circuit = timed("circuit", lambda: class_circuit(registered, PP.window_class_of(cls_arg),
                                                     tables=PearlTables(scheme) if scheme else None))
    program, query = timed("program", circuit.program)
    v1 = timed("v1", lambda: T.v1_case(program, query))
    # a window's case carries its program function's parameters where values are registered (replay.py)
    v1["params"]["parameters"] = [[n, t.to_json()] for n, t in program.fn.signature()[0]]
    desc = v1["params"]["program"]
    doc |= {"templates": len(circuit.templates), "program": codec.program_digest(desc),
            "program_bytes": len(codec.canonical_json(desc))}
    write()
    try:
        case = timed("case", lambda: SC.inference_only_case(v1, *SC.UNBOUNDED))
        bits = {"templates": 0, "values": 0}
        for name, t in case["params"]["parameters"]:  # a template's own parameter (``<j>.<name>``) or a registered value
            head, _, rest = name.partition(".")
            ty = type_from_json(t)
            ws = sorted(ty.width_set())
            if len(ws) == 1:
                bits["templates" if head.isdecimal() and rest else "values"] += ty.leaves * ws[0]
            else:
                bits["mixed_width_parameters"] = bits.get("mixed_width_parameters", 0) + 1
        doc["parameter_bits"] = bits
        pol = json.loads(Path(policy).read_text()) if policy and Path(policy).is_file() else None
        if pol is not None:
            doc["count_rule"] = {k: pol[k] for k in ("c", "k", "n")}
        exe = timed("build", lambda: SC.build(*SC.INFERENCE_ONLY_LEAN))
        with tempfile.TemporaryDirectory() as s:
            scratch = Path(s)
            alpha, gamma = timed("profile", lambda: SC.division_policy(v1, (), exe, scratch))
            doc["profile"] = {"gamma": gamma, "alpha": [alpha.numerator, alpha.denominator],
                              "input_bits": str(alpha * gamma)}
            write()
            if pol is not None:
                slots = PP.window_class_of(cls_arg).slots
                if gamma % slots:
                    raise SC.Refused(f"the class's {slots} slots don't divide the profile's gamma {gamma}, so no block "
                                     "a slot has gamma over the slots as its gates: no held pair")
                held = Fraction(slots * pol["k"], gamma)
                doc["held"] = {"alpha": [held.numerator, held.denominator], "gamma": gamma // slots}
                write()

            def ask(a, g):
                got, _ = SC.lean_answer(exe, SC.inference_only_case(v1, a, g), scratch)
                return {"verdict": got.get("verdict"), "reason": (got.get("reason") or None) and str(got.get("reason"))[:300]}

            doc["at_profile"] = timed("at_profile", lambda: ask(alpha, gamma))
            if rejects and pol is not None:
                doc["at_count_rule_numbers"] = timed("at_count_rule", lambda: ask(Fraction(pol["k"], pol["c"]), pol["c"]))
                doc["at_k_over_gamma"] = timed("at_k_over_gamma", lambda: ask(Fraction(pol["k"], gamma), gamma))
            # the largest child is the Lean check (or its build): what a node needs to ask it of this class
            doc["children_peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**20, 2)
            if doc["at_profile"]["verdict"] != "accept":
                raise SC.Refused(f"the check refuses the class at its own profile: {doc['at_profile']['reason']}")
    except (SC.Refused, SC.PF.LeanError) as e:
        doc["refused"] = f"{type(e).__name__}: {e}"[:600]
    write()
    print(json.dumps(doc, sort_keys=True))
    return 1 if "refused" in doc else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
