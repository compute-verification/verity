"""The Build's path for a single-request stochastic workload, on CPU.

The request Program is derived (the stand-in `StochasticServe` through the importer and the vLLM sampler rule, S the constant a
single request launches: `GumbelTopPTokenSelect_v2{V,S}`, the keep word as word gates) and written as derive_step writes it. The
workload Program is composed with GP-01's command in a fresh interpreter, since the Build runs it as its own process whose registry
holds only what GP-01 imports. The composed Program is then decoded in another fresh interpreter, through the Match's registry.
Three defects once failed only here: the descriptor writer aliased a node argument to its own node (`F32Eq(x, x)`), GP-01's
registry lacked the keep word's word gates (`NvLogf_v1`), and the manifest's program view lacked `GumbelTopPTokenSelect_v2`.

Deriving workload 101's own request Program needs vLLM (a GPU pod); the Programs its Build wrote, through every later step the CPU
can run, are `test_single_request_build_path_of_record`.
"""
from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from verity.ml.circuits.codec import canonical_json, encode_program, program_digest
from verity_vllm.program.frontend import derive_examples as ex
from verity_vllm.program.frontend.derive import derive_report
from verity_vllm.program.frontend.rules import DEFAULT_RULES
from verity_vllm.program.frontend.rules.base import RuleSet
from verity_vllm.program.frontend.rules.vllm_sampling import VLLM_SAMPLING_RULES
from verity_vllm.records import workload as WL
from verity.ml.definitions.registry import b1

VLLM_DIR = Path(__file__).resolve().parents[2]


def _derive(V: int, S: int, LP: int, STEPS: int):
    C = dict(b1.B0_CONFIG, NL=1, V=V, MAX_POS=64)
    rules = RuleSet("default+stoch", list(DEFAULT_RULES.rules) + list(VLLM_SAMPLING_RULES), list(DEFAULT_RULES.families))
    with torch.device("meta"):
        m = ex.StochasticServe(C, STEPS, constant_S=S, temperature=0.8, top_p=0.95)
    inputs = (torch.empty(LP, dtype=torch.int32, device="meta"), torch.empty(1, dtype=torch.int64, device="meta"))
    rep = derive_report(m, inputs, mode="profile", profile="b1-eager-v3", config=C, rules=rules, name="StochServe")
    assert rep.ok, rep.unsupported
    return rep


def _write_request(rep, d: Path, LP: int, STEPS: int, S: int) -> None:
    """The files of derive_step's request dir that GP-01 reads."""
    d.mkdir(parents=True)
    desc = encode_program(rep.program)
    with gzip.open(d / "descriptor.json.gz", "wb") as f:
        f.write(canonical_json(desc))
    with gzip.open(d / "instances.json.gz", "wt") as f:
        json.dump({"params": [[n, repr(t)] for n, t in rep.flat_definition.signature()[0]], "rows": []}, f)
    sampling = {"kind": "stochastic", "temperature": 0.8, "top_p": 0.95, "top_k": 0, "seeded": True, "splits": S}
    json.dump({"program": {"digest": program_digest(desc)}, "sampling": sampling,
               "applicability": {"execution": "bi-eager", "numerics_profile": "stand-in", "sampler": "stochastic",
                                 "engine": {"enforce_eager": True, "enable_prefix_caching": False}}}, open(d / "result.json", "w"))
    json.dump({"structural_inputs": {"LP": LP, "T": STEPS, "sampling": sampling}}, open(d / "construction_manifest.json", "w"))
    json.dump({"provenance": {"source": {"example_inputs": [["torch.int32", [LP]], ["torch.int64", [1]]]},
                              "weights": rep.provenance.get("weights", [])}}, open(d / "derive-report.json", "w"))


def _compose(workload: Path, derived: Path, out: Path, eos: int) -> dict:
    """GP-01 as the Build runs it: `verity-vllm global-program` in its own interpreter."""
    p = subprocess.run([sys.executable, "-m", "verity_vllm.pipeline.cli", "global-program", "--workload", str(workload),
                        "--derived", str(derived), "--out", str(out), "--eos-token-ids", str(eos)],
                       cwd=VLLM_DIR, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-3000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


def _decode_cold(descriptor: Path) -> int:
    """The gates of `descriptor` decoded in a fresh interpreter through the registry the Match reads (`catalog.load`), as the Build's
    next phase reads the workload Program."""
    code = ("import gzip, json, sys\n"
            "from verity.ml.circuits.codec import decode_program\n"
            "from verity.ml.definitions.registry.catalog import load\n"
            "print(decode_program(json.load(gzip.open(sys.argv[1])), load()[0]).gates)\n")
    p = subprocess.run([sys.executable, "-c", code, str(descriptor)], cwd=VLLM_DIR, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-3000:]
    return int(p.stdout.strip().splitlines()[-1])


@pytest.mark.slow
def test_a_single_request_stochastic_workload_composes_as_the_build_does(tmp_path):
    """At V = 1000 and S = 32 the keep word's NaN tests read multi-part sequences (the writer's alias case)."""
    V, S, LP, STEPS = 1000, 32, 3, 1
    rep = _derive(V, S, LP, STEPS)
    assert any(n.fn.id == f"GumbelTopPTokenSelect_v2{{V={V},S={S}}}" for n in rep.flat_definition.body.nodes)
    _write_request(rep, tmp_path / "build_request", LP, STEPS, S)
    wl = tmp_path / "workload.json"
    json.dump({"schema": WL.SCHEMA, "model": "toy", "engine": {}, "variants": {"stoch-t0.8-p0.95": {"temperature": 0.8, "top_p": 0.95, "top_k": 0}},
               "requests": [{"request_id": "r-0", "prompt_token_ids": list(range(1, LP + 1)), "max_tokens": STEPS + 1, "arrive_step": 0}],
               "sampling": {"variant": "stoch-t0.8-p0.95", "seed": 7}, "sweep": {"execution": "bi-eager"}}, open(wl, "w"))
    r = _compose(wl, tmp_path / "build_request", tmp_path / "build_workload", eos=0)
    assert r["requests"] == 1
    assert _decode_cold(tmp_path / "build_workload" / "descriptor.json.gz") == r["gates"]


def _run(code: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code, *args], cwd=VLLM_DIR, capture_output=True, text=True)


def test_every_consumers_registry_holds_every_registered_definition():
    """The global Program (`pipeline.global_program`) and the word Definitions look Definitions up through entry points of their own,
    beside the `catalog.load` the Match, the query views and the descriptor equivalence read: in a fresh interpreter each one's
    registry holds every Definition and primitive the registry package registers."""
    want = _run("from verity.ml.definitions.registry.catalog import load\nprint(len(load()[0].defs))")
    assert want.returncode == 0, want.stderr[-2000:]
    n = int(want.stdout.split()[-1])
    for setup in ("import verity_vllm.pipeline.global_program as m; from verity.ml.circuits.defs import REGISTRY as R",
                  "from verity.ml.definitions import word as W; W.specialization('GumbelTopPTokenSelect_v2', {'V': 16, 'S': 4}); "
                  "from verity.ml.circuits.defs import REGISTRY as R"):
        p = _run(setup + "\nfor k in ('GumbelTopPTokenSelect_v2', 'TopPMask_v2', 'TopPKeepWord_v1', 'NvLogf_v1', 'TopPNumkeep_v1'):\n"
                 "    assert k in R.defs, k\nprint(len(R.defs))")
        assert p.returncode == 0, (setup, p.stderr[-2000:])
        assert int(p.stdout.split()[-1]) >= n, setup
