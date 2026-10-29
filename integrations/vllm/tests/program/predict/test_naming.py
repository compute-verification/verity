"""verity_vllm.program.predict.naming: a request Program named as the pinned vLLM's trace of it names it, from the registered
commit alone.

* Against the pinned vLLM's traces (`fixtures/naming_traces.json`: Llama-3.2-1B's request Programs on the H200 target, int8 and
  bf16, greedy, cached, stochastic and the served window's shape, each as digests): the predicted Program is the trace's, and so
  is what its Build says beside it: its rows' names, its parameters' names and types (the weights under the wrapper's
  ``model.``), its correspondence record as `recorded_names` reads it (each Call's name, the tensorless Calls, each placed Call's
  row and columns) with the record's inputs, weights, rank and body, and its cached-prefix layout.
* `compile_steps(named=True)` names every Program it compiles, the same on every compile, with no vLLM importable, and
  `write_build` writes the Build the audit reads: `ProgramIndex` takes its record under its digest, and reads each row as the
  rows' own names say.
* What the naming doesn't reach is refused by name.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import sys
from dataclasses import astuple, fields
from pathlib import Path
from typing import Any

import pytest
from verity.ml.circuits.types import Record
from verity_vllm.check.replay import index as IX
from verity_vllm.correspondence.reader_for_query import recorded_names
from verity_vllm.correspondence.runtime import validate_document
from verity_vllm.program.frontend.target_profile import TargetProfile
from verity_vllm.program.instances_file import read_instances
from verity_vllm.program.model import INT8_EXACT_QUANT
from verity_vllm.program.predict.build import Unexpressible
from verity_vllm.program.predict.naming import WRAPPER, named_program, write_build
from verity_vllm.program.predict.phi import Phi
from verity_vllm.program.predict.steps import compile_steps

from tests.program.predict.test_steps import TINY, _hit, _registered

FIX = Path(__file__).parent / "fixtures"
TRACES = json.loads((FIX / "naming_traces.json").read_text())


def _h(x: Any) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def pins(names: dict[int, str], params: list, doc: dict, prefix: dict | None) -> dict:
    """What a request Build says beside its Program (its rows' names, its parameters, its correspondence record as the audit
    reads it, its cached-prefix layout), each as a digest; the layout as it is."""
    called, tensorless, rows, columns = recorded_names(doc)
    return {"names": _h(sorted((int(k), v) for k, v in names.items())), "params": _h(params),
            "recorded": _h([[[k, *astuple(v)] for k, v in sorted(called.items())], sorted(tensorless), sorted(rows.items()),
                            sorted([k, *v] for k, v in columns.items())]),
            "record": _h({k: doc.get(k) for k in ("schema", "body", "inputs", "weights", "rank")}), "prefix": prefix}


def case_of(d: Path) -> dict:
    """`naming_traces.json`'s case of a Build dir the pinned vLLM traced (`steps.trace_commands`): its φ and shape from its
    ``result.json``, three of its rows' names and `pins` of what it says beside its Program; ``python -m
    tests.program.predict.test_naming DIR...`` (from ``integrations/vllm``) prints each."""
    r = json.loads((d / "result.json").read_text())
    inst = read_instances(str(d / "instances.json.gz"), expand=False)
    names = {int(x["i"]): x["name"] for x in inst["rows"]}
    with gzip.open(d / "descriptor.json.gz", "rt") as f:
        ann = json.load(f)["annotations"]
    assert {int(k): v for k, v in ann["node_hints"].items()} == names, f"{d}: node_hints aren't the instances' names"
    tf = {f.name for f in fields(TargetProfile)}
    target = {k: v for k, v in r["target"]["declared"].items() if k in tf and v is not None}
    s = r["sampling"]
    pc = d / "prefix_cache.json"
    attn = next(k for k, v in sorted(names.items()) if "[t=" in v)
    return {"trace": f"{'int8' if target.get('int8_exact') else 'bf16'}-{d.name.removeprefix('build_request_')}",
            "model": r["model_pin"]["model"], "revision": r["model_pin"]["revision"], "target": target,
            "sampling": {"kind": "greedy"} if s["kind"] == "greedy" else {k: s[k] for k in ("kind", "temperature", "top_p", "top_k", "seeded", "splits")},
            "LP": r["LP"], "T": r["T"], "C": r.get("C") or 0, "digest": r["program"]["digest"],
            "names": {str(k): names[k] for k in (0, attn, max(names))},
            "pins": pins(names, inst["params"], ann["correspondence"], json.loads(pc.read_text()) if pc.exists() else None),
            "slow": s["kind"] != "greedy"}


def phi_of(case: dict) -> Phi:
    """φ of a traced Build (`naming_traces.json`'s case): the fixture's config, int8's quantization where the target runs
    int8-exact linears."""
    cfg = json.loads((FIX / TRACES["config"]).read_text())
    int8 = case["target"].get("int8_exact") == "linears"
    model = {"repo": case["model"], "revision": case["revision"],
             "hf_config": {**cfg, **({"quantization_config": dict(INT8_EXACT_QUANT)} if int8 else {})}, **({"dtype": "int8"} if int8 else {})}
    return Phi(model=model, target=case["target"], sampling=case["sampling"])


def _case(case: dict) -> Any:
    return pytest.param(case, id=case["trace"], marks=[pytest.mark.slow] if case["slow"] else [])


@pytest.mark.parametrize("case", [_case(c) for c in TRACES["cases"]])
def test_a_request_program_is_named_as_its_trace_names_it(case):
    """The stochastic case (slow: about a minute and 4 GB, the select over Llama's 128k vocabulary) is the one that holds the
    select's names and its constants' Calls to a trace; TINY's stochastic Programs have no trace."""
    b = named_program(phi_of(case), case["LP"], case["T"], case["C"])
    n = b.named
    assert b.digest == case["digest"]
    assert {k: n.names[int(k)] for k in case["names"]} == case["names"]
    assert pins(n.names, n.params, n.correspondence, n.prefix) == case["pins"]
    assert validate_document(n.correspondence) == []


def test_compile_steps_names_each_program_the_same_on_every_compile_with_no_vllm(tmp_path, monkeypatch):
    """A session with a cached hit: each Program it compiles is named, the same on a second compile, with its cached prefix's
    layout, and its Build (`write_build`) is read by the audit's `ProgramIndex` through its own record, which agrees row for row
    with the rows' names; no vLLM is importable throughout. (A stochastic Program's select is pinned to its trace above: TINY's
    takes seconds to define.)"""
    monkeypatch.setitem(sys.modules, "vllm", None)
    reg, (advice, ingress) = _registered(), _hit(16)
    c = compile_steps(reg, advice, ingress, named=True)
    assert set(c.named) == set(c.programs) and c.named == compile_steps(reg, advice, ingress, named=True).named
    for dg, n in c.named.items():
        u = next(u for u in c.requests if u.digest == dg)
        body = c.programs[dg].fn.body
        assert sorted(n.names) == list(range(len(body.nodes))) and n.params == [
            [p, str(Record(*((WRAPPER + f, x) for f, x in t.fields)) if isinstance(t, Record) else t)] for p, t in body.params]
        assert (n.shape["LP"], n.shape["T"], n.shape["C"], n.shape["sampling"]) == (u.LP, u.T, u.C, u.policy)
        d = tmp_path / dg[:16]
        assert write_build(str(d), c.programs[dg], n) == dg
        inst = read_instances(str(d / "instances.json.gz"), expand=False)
        assert {int(r["i"]): r["name"] for r in inst["rows"]} == n.names and inst["params"] == n.params
        mine, by_names = IX.ProgramIndex(inst, program_dir=str(d)), IX.ProgramIndex(inst)
        assert mine.correspondence["source"] == "correspondence" and "problem" not in mine.correspondence
        assert mine.correspondence["digest"] == json.loads((d / "result.json").read_text())["program"]["correspondence_digest"]
        assert [mine.describe(r) for r in mine.rows] == [by_names.describe(r) for r in by_names.rows]
        assert mine.prefix == "model." and mine.kv_prefix == n.prefix
        with gzip.open(d / "descriptor.json.gz", "rt") as f:
            hints = json.load(f)["annotations"]["node_hints"]
        assert {int(k): v for k, v in hints.items()} == n.names
    assert [c.named[u.digest].prefix for u in c.requests] == [None, {
        "input": 1, "C": 16, "row_leaves": TINY["num_key_value_heads"] * TINY["hidden_size"] // TINY["num_attention_heads"],
        "layers": [f"model.layers.{i}.self_attn.attn" for i in range(TINY["num_hidden_layers"])]}]


@pytest.mark.parametrize("config,fact", [("qwen25-05b", "a q/k/v bias"), ("qwen3-4b", "q/k norms")])
def test_what_the_naming_does_not_reach_is_refused_by_name(config, fact):
    phi = Phi.from_json({**_registered(json.loads((FIX / f"{config}.config.json").read_text())).phi.to_json(), "sampling": {"kind": "greedy"}})
    with pytest.raises(Unexpressible, match=f"^naming: {fact} "):
        named_program(phi, 4, 1)
    tiny = Phi.from_json({**_registered().phi.to_json(), "sampling": {"kind": "greedy"}, "tp": {"world": 2, "rank": 0}})
    with pytest.raises(Unexpressible, match="^naming: the .* decoder at TP2 "):
        named_program(tiny, 4, 1)


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print(json.dumps(case_of(Path(arg))))
