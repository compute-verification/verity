"""What the door admits: `one_circuit.template_form`, the clauses of `Spot.OneStageZk.admitted` a block's template must
meet, on every template a circuit's root calls, at in(S)'s cuts under a template query listing them all; no prover runs.
One ``FORM`` line a circuit (its description's sha256 and SHA-512, the templates admitted and refused by clause) and one
``TEMPLATE`` line a template; then one ``BLOCKS`` line a partition of it: `one_circuit.admitted_block`, the door's
second predicate, on every proof block of its last level (the blocks refused by clause, and the first refusal's text).

  python door_form.py product                 # product(k=576, s=8, units=288, block=2, registered=False): Friday's benchmark
  python door_form.py class --tree TREE       # tests/benchmarks/prove_path/test_class_rows._circuit, TINY's class circuit
  python door_form.py chain --tree TREE       # the one-circuit chain (test_one_circuit_chain.registered)
  python door_form.py one_linear --tree TREE  # benchmarks/one_stage/one_linear.two_levels
  python door_form.py file C.json             # a description, as `served_door.py build` writes it
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

from verity.circuits import edges as EG
from verity.circuits import partition_object as PO
from verity.circuits import restrict as R
from verity.circuits.codec import descriptor_chunks, descriptor_id
from verity.circuits.program import input_id
from verity.protocols.proof.spot.verifier.one_stage import one_circuit as OC

CLAUSE = re.compile(re.escape(OC.NOT_ADMITTED) + r", ([a-z-]+): (.*)", re.S)


def census(name: str, c: bytes) -> dict:
    D = R.program(c)
    templates = sorted({descriptor_id(nd.fn) for nd in D.root.body.nodes if input_id(nd.fn.id) is None})
    try:
        cuts = EG.io_cuts(D.desc, PO.TemplatePartition(D, PO.template_instances_query(templates)).ranges)
    except ValueError as e:
        cuts, cut_error = {}, str(e)
    else:
        cut_error = None
    rows = []
    for t in templates:
        try:
            ins, fields = OC.template_form(D, t, cuts.get(t, {}))
            rows.append({"template": t, "admitted": True, "ins": ins, "fields": fields})
        except OC.Refused as e:
            m = CLAUSE.search(str(e))
            rows.append({"template": t, "admitted": False, "clause": m.group(1) if m else "?",
                         "why": m.group(2) if m else str(e)})
    out = {"circuit": name, "sha256": hashlib.sha256(c).hexdigest(), "sha512": hashlib.sha512(c).hexdigest(),
           "bytes": len(c), "templates": len(rows), "admitted": sum(r["admitted"] for r in rows),
           "refused": dict(Counter(r["clause"] for r in rows if not r["admitted"])), "cuts_refused": cut_error}
    print("FORM " + json.dumps(out), flush=True)
    for r in rows:
        print("TEMPLATE " + json.dumps({"sha256": out["sha256"], **r}), flush=True)
    return out


def blocks(name: str, c: bytes, part: dict) -> dict:
    D = R.program(c)
    tp = PO.TemplatePartition(D, part["query"])
    level = part["blocks"][-1]
    refused, first = Counter(), None
    for lo, hi in level:
        k, _m = tp.locate(lo)
        try:
            OC.admitted_block(D, part, [(lo, hi)], descriptor_id(D.root.body.nodes[k].fn))
        except OC.Refused as e:
            m = CLAUSE.search(str(e))
            refused[m.group(1) if m else "?"] += 1
            first = first or {"block": [lo, hi], "why": str(e)}
    out = {"circuit": name, "blocks": len(level), "refused": dict(refused), "first": first}
    print("BLOCKS " + json.dumps(out), flush=True)
    return out


def _windows(name: str, circuit, tree: Path) -> None:
    """`blocks` on the circuit's window partitions (`interface.Window`) at two caps: 2 units a block, and its population,
    each member then cut only at the order's ranges and every wire's and registered read's ends."""
    from verity.commitments.periods import interface as I
    from verity.circuits.codec import descriptor_chunks

    TI = _load(tree / "tests/verity/commitments/periods/test_interface.py", "door_form_periods")
    program, _query = circuit.program()
    c = b"".join(descriptor_chunks(program))
    n = sum(k for _, k in circuit.templates)
    for cap in (2, n):
        blocks(f"{name} cap={cap}", c, I.Window(circuit, TI.RUN, TI.SAMPLING_DESIGN, cap).window.partition)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["product", "class", "chain", "one_linear", "served", "file"])
    ap.add_argument("path", nargs="?")
    ap.add_argument("--tree", default=str(Path(__file__).resolve().parents[2]))
    a = ap.parse_args()
    tree = Path(a.tree)
    if a.what == "product":
        from verity.circuits.ops.product import product
        # k_log and g only bound the block against the prover's floor; the description is the same at any
        p = product(k=576, s=8, units=288, block=2, registered=False, k_log=24, g=2)
        census("product(k=576,s=8,units=288,block=2,registered=False)", p.description)
        blocks("product(k=576,s=8,units=288,block=2,registered=False)", p.description, p.partition)
    elif a.what == "class":
        sys.path.insert(0, str(tree / "tests/benchmarks/prove_path"))
        mod = _load(tree / "tests/benchmarks/prove_path/test_class_rows.py", "door_form_class_rows")
        program, _query = mod._circuit().program()
        census("test_class_rows._circuit", b"".join(descriptor_chunks(program)))
        _windows("test_class_rows._circuit", mod._circuit(), tree)
    elif a.what == "chain":
        mod = _load(tree / "tests/verity/protocols/proof/spot/verifier/test_one_circuit_chain.py", "door_form_chain")
        census("test_one_circuit_chain.registered", mod.registered()["C"])
        _windows("test_one_circuit_chain.circuit", mod.circuit(), tree)
    elif a.what == "one_linear":
        sys.path.insert(0, str(tree / "benchmarks/one_stage"))
        OL = _load(tree / "benchmarks/one_stage/one_linear.py", "door_form_one_linear")
        c, part = OL.two_levels()
        census("one_linear", c)
        blocks("one_linear", c, part)
    elif a.what == "served":
        sys.path.insert(0, str(tree / "tests/adapters/vllm"))
        from verity_vllm.program.predict import window_circuit as WCirc
        from verity_vllm.program.predict import window_class as WC
        from tests.program.predict import test_steps as S
        cfg = json.loads((tree / "tests/adapters/vllm/tests/program/predict/fixtures/llama32-1b.config.json").read_text())
        reg = S.Registered.of(S.Phi(model={"repo": "org/m", "revision": "r", "hf_config": cfg},
                                    target=dict(S.RTXPRO6000)), default_seed=7)
        circuit = WCirc.lower(reg, WC.WindowClass(4, 27, 2), None,
                              requests=[WC.Request(0, j, 27, 2) for j in range(4)]).circuit
        _windows("served llama32-1b class 4,27,2", circuit, tree)
    else:
        census(a.path, Path(a.path).read_bytes())
    return 0


if __name__ == "__main__":
    sys.exit(main())
