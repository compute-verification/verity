"""`verity_vllm.query.units` held to the canonical partition, `verity.ml.circuits.partition_object.CallPartition` (`Q_call` v1).

On a request Program read as the audit reads its Build (`write_build`, `from_instances`, the correspondence record), the Values a Call
produces that `units.required_values` requires under the query of record (each Call its own proof block) and serving's policy
(`required.population`) are exactly the Call outputs CallPartition commits: a port is committed when its Call's derived committed set
holds its returned computing gates, all of them or none. Prescribed inputs (Values of negative id) are the input unit's, in no proof
block, and aren't compared. The bare boundary (`Policy.none()`) is that set less the Call outputs no Call reads (a fused norm's
residual nobody consumes, the last sampled token): CallPartition commits every returned computing gate, and serving's policy requires
them as returned tensors and sampling-event outputs.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from verity.ml.circuits import partition_object as PO
from verity.ml.circuits.cut import call_scope
from verity.ml.circuits.layout import resolve
from verity_vllm.correspondence.reader_for_query import Correspondence
from verity_vllm.program.model import INT8_EXACT_QUANT
from verity_vllm.program.predict.naming import named_program, write_build
from verity_vllm.program.predict.phi import Phi
from verity_vllm.query.program_view import ProgramView, from_instances
from verity_vllm.query.required import population
from verity_vllm.query.units import Policy, ValueRef, partition_by, required_values

from tests.program.predict.test_naming import TRACES, phi_of
from tests.program.predict.test_steps import TINY

SERVED = next(c for c in TRACES["cases"] if c["trace"] == "int8-LP3_T2_greedy")


def committed_outputs(program, P: ProgramView) -> frozenset[ValueRef]:
    """The Call outputs `CallPartition` commits, as Values of `P`: its Calls are the root's non-Input nodes, `P`'s in order, and
    activations of one Definition commit alike, so each Definition's gates are laid out once."""
    part = PO.evaluate(program, PO.call_query())
    assert len(part.calls) == len(P.calls)
    masks: dict[str, list[bool]] = {}
    out = set()
    for p, (_k, cut) in enumerate(part.calls):
        fn = cut.fn
        if fn.id not in masks:
            G, committed = part.graph(p), frozenset(part.committed(p))
            _c, call = call_scope(fn)
            rets = [0] if not hasattr(fn, "body") else [resolve(call, r) - call.offset for r in fn.body.ret]
            masks[fn.id] = [0 <= g < G.n and G.idx[g] in committed for g in rets]
        leaf, c = masks[fn.id], P.calls[p]
        assert c.port_table[-1].hi == len(leaf), (p, c.definition, fn.id)
        for port in c.port_table:
            got = leaf[port.lo:port.hi]
            assert all(got) or not any(got), f"Call {p} ({c.definition}) commits part of port {port.name!r}"
            if any(got):
                out.add(ValueRef(p, port.name))
    return frozenset(out)


def produced(values) -> frozenset[ValueRef]:
    return frozenset(v for v in values if v.call >= 0)


def difference(P: ProgramView, ours: frozenset[ValueRef], theirs: frozenset[ValueRef]) -> str:
    def side(vs: frozenset[ValueRef]) -> str:
        ex = [f"Call {v.call} {P.calls[v.call].definition} {P.calls[v.call].name!r} port {v.port!r}" for v in sorted(vs)[:3]]
        return f"{len(vs)}: {'; '.join(ex)}"
    return f"units.py only {side(ours - theirs)}; CallPartition only {side(theirs - ours)}"


def check(phi: Phi, LP: int, T: int, C: int, out: Path) -> None:
    b = named_program(phi, LP, T, C)
    write_build(str(out), b.program, b.named)
    P = from_instances(str(out / "instances.json.gz"))
    res = json.loads((out / "result.json").read_text())
    corr = Correspondence.of(P, str(out), implementation_paths=Correspondence.implementation_paths_of(res))
    pop = population(P, corr)
    Q = partition_by(lambda c: c.id)
    required = required_values(P, Q, pop.policy)
    assert required.values == pop.required.values
    committed = committed_outputs(b.program, P)
    assert produced(required.values) == committed, difference(P, produced(required.values), committed)
    bare = produced(required_values(P, Q, Policy.none()).values)
    read = {r for c in P.calls for r in c.reads}
    assert bare <= committed and not (committed - bare) & read, difference(P, bare, committed)


def test_tiny_int8_program(tmp_path):
    """TINY's int8 request Program on the served target, at the served shape."""
    phi = Phi(model={"repo": "org/m", "revision": "r", "hf_config": {**TINY, "quantization_config": dict(INT8_EXACT_QUANT)}, "dtype": "int8"},
              target=SERVED["target"], sampling=SERVED["sampling"])
    check(phi, SERVED["LP"], SERVED["T"], SERVED["C"], tmp_path)


@pytest.mark.slow
def test_served_h200_int8_program(tmp_path):
    """The served H200 int8 request Program (Llama-3.2-1B INT8, the pinned vLLM's trace of it in `naming_traces.json`). Beyond
    TINY's it reaches the served Definitions, the 128256-wide LM head and token select and the int8 linears at K = 2048 and
    8192, and the served names serving's policy reads; CallPartition lays out their 19.7M gates."""
    check(phi_of(SERVED), SERVED["LP"], SERVED["T"], SERVED["C"], tmp_path)
