"""The Build's path for a single-request stochastic workload, on CPU.

The request Program is derived (the stand-in `StochasticServe` through the importer and the vLLM sampler rule, S the constant a
single request launches: `GumbelTopPTokenSelect_v2{V,S}`, the keep word as word gates) and written as derive_step writes it. The
workload Program is composed with GP-01's command in a fresh interpreter, since the Build runs it as its own process whose registry
holds only what GP-01 imports. The composed Program is then decoded in another fresh interpreter, through the Match's registry.
Two defects once failed only here:
the descriptor writer aliased a node argument to its own node (`F32Eq(x, x)`), and GP-01's registry lacked the keep word's word
gates (`NvLogf_v1`).

Deriving #101's own request Program needs vLLM (a GPU pod), so `test_101s_request_program_composes_as_its_build_does` takes the one
#101's Build derived (preserved in the evidence store) through every later step, and skips where the store is unreachable.
"""
from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from verity.ir.codec import canonical_json, encode_program, program_digest
from verity_vllm.program.frontend import derive_examples as ex
from verity_vllm.program.frontend.derive import derive_report
from verity_vllm.program.frontend.rules import DEFAULT_RULES
from verity_vllm.program.frontend.rules.base import RuleSet
from verity_vllm.program.frontend.rules.vllm_sampling import VLLM_SAMPLING_RULES
from verity_vllm.program.registry import b1

VLLM_DIR = Path(__file__).resolve().parents[2]
#: #101's Build on train H `be354ab0` (run r20260928-145638-f5d8): its `build_request/` is the derived request Program
BUILD_101 = "art:adf90df48cf23dc7e75ff876c07102eb888134e4b88e9165c3243576651ec130"
ROW_101 = "llama32-1b__bf16__l40s__tp1__b1__i256__o32__mixed__stoch-t0.8-p0.95__bi-eager"


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
        json.dump({"params": [[n, repr(t)] for n, t in rep.flat_definition.signature()[0]]}, f)
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
                       cwd=VLLM_DIR, capture_output=True, text=True, timeout=1800)
    assert p.returncode == 0, p.stderr[-3000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


def _decode_cold(descriptor: Path) -> int:
    """The gates of `descriptor` decoded in a fresh interpreter through the Match's registry (`program_compare._registry`, its own
    module list), as the Build's next stage reads the workload Program."""
    code = ("import gzip, json, sys\n"
            "from verity.ir.codec import decode_program\n"
            "from verity_vllm.check.match.program_compare import _registry\n"
            "print(decode_program(json.load(gzip.open(sys.argv[1])), _registry()).gates)\n")
    p = subprocess.run([sys.executable, "-c", code, str(descriptor)], cwd=VLLM_DIR, capture_output=True, text=True, timeout=1800)
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
    json.dump({"requests": [{"request_id": "r-0", "prompt_token_ids": list(range(1, LP + 1)), "max_tokens": STEPS + 1, "arrive_step": 0}],
               "sampling": {"temperature": 0.8, "top_p": 0.95, "top_k": 0}, "seeds": [7], "execution": "bi-eager"}, open(wl, "w"))
    r = _compose(wl, tmp_path / "build_request", tmp_path / "build_workload", eos=0)
    assert r["requests"] == 1
    assert _decode_cold(tmp_path / "build_workload" / "descriptor.json.gz") == r["gates"]


@pytest.mark.skipif(os.environ.get("VERITY_SKIP_STORE") == "1", reason="VERITY_SKIP_STORE=1")
@pytest.mark.slow
def test_101s_request_program_composes_as_its_build_does(tmp_path):
    """#101's request Program (V = 128256, S = 32, 23.26 x 10^9 word gates once composed) through GP-01 and a cold decode."""
    from tests.regression import store_io
    if not store_io.reachable():
        pytest.skip("the evidence store is not reachable here")
    build = store_io.fetch(BUILD_101, to=tmp_path / "build")
    r = _compose(VLLM_DIR / "workloads" / f"{ROW_101}.json", build / "build_request", tmp_path / "build_workload", eos=128001)
    assert r["requests"] == 1 and r["gates"] == 23_256_312_239
    assert _decode_cold(tmp_path / "build_workload" / "descriptor.json.gz") == r["gates"]
