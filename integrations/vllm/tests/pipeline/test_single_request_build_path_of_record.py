"""Workload 101's Build (`test_single_request_build_path`), fetched from the evidence store: deriving its own request Program needs
vLLM (a GPU pod), so these take the Programs its Build wrote through every later step the CPU can run: GP-01, the manifest step with
its word check, the query's program view and the Match's and the Commit's lookups, each in a fresh interpreter.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.pipeline.test_single_request_build_path import VLLM_DIR, _compose, _decode_cold, _run
from verity.ml.circuits.codec import canonical_json, program_digest

#: workload 101's Build: `build_request/` (the derived request Program) and
#: `build_workload/` (the workload Program GP-01 composed)
BUILD_101 = "art:2d65d5d7bdc21af3df01c041272e5a72b40a86bc74cc09075c7e726cc01f1fd0"
#: the workload's manifest command (`row_stages.required_manifest`): every tap policy on, the word check at its default 16/32
MANIFEST_POLICY = ("--norm-scales", "--router-softmax", "--vocab-range")
ROW_101 = "llama32-1b__bf16__l40s__tp1__b1__i256__o32__mixed__stoch-t0.8-p0.95__bi-eager"


def _build_101(tmp_path) -> Path:
    from tests.regression import store_io
    if os.environ.get("VERITY_SKIP_STORE") == "1" or not store_io.reachable():
        pytest.skip("the evidence store is not reachable here")
    build = store_io.fetch(BUILD_101, to=tmp_path / "build")
    _rekey_sha512(build / "build_request")
    return build


def _rekey_sha512(d: Path) -> None:
    """Workload 101's Build recorded its Program digest under SHA-256, before the identities moved to SHA-512. The record must be the SHA-256
    digest of the descriptor the Build wrote; each file of the request dir then carries that same descriptor's SHA-512 digest in its
    place, so nothing but the digest changes."""
    desc = json.load(gzip.open(d / "descriptor.json.gz"))
    old = hashlib.sha256(canonical_json({k: v for k, v in desc.items() if k != "annotations"})).hexdigest()
    assert json.load(open(d / "result.json"))["program"]["digest"] == old
    new = program_digest(desc)
    for p in d.glob("*.json"):
        p.write_text(p.read_text().replace(old, new))


@pytest.mark.slow
def test_101s_request_program_composes_as_its_build_does(tmp_path):
    """Workload 101's request Program (V = 128256, S = 32, 23.26 x 10^9 word gates once composed) through GP-01 and a cold decode."""
    build = _build_101(tmp_path)
    r = _compose(VLLM_DIR / "workloads" / f"{ROW_101}.json", build / "build_request", tmp_path / "build_workload", eos=128001)
    assert r["requests"] == 1 and r["gates"] == 23_256_312_239
    assert _decode_cold(tmp_path / "build_workload" / "descriptor.json.gz") == r["gates"]


@pytest.mark.slow
def test_101s_manifest_step_resolves_every_call_as_its_build_does(tmp_path):
    """`manifest build` as the workload runs it, in its own interpreter: the program view finds every Call's Definition and the word check
    runs over them all.  At the default gate limit its one violation is the sampler Call's size (the workload raises the limit for it on a
    big-memory pod, `VERITY_QWORD_MAX_GATES`), never a Definition the registry lacks."""
    build = _build_101(tmp_path)
    env = {k: v for k, v in os.environ.items() if k not in ("VERITY_QWORD_MAX_GATES", "VERITY_WORD_CHECK")}
    p = subprocess.run([sys.executable, "-m", "verity_vllm.pipeline.cli", "manifest", "build", "--program", str(build / "build_request"),
                        "--workload", f"workloads/{ROW_101}.json", *MANIFEST_POLICY, "--out", str(tmp_path / "manifest.json")],
                       cwd=VLLM_DIR, capture_output=True, text=True, env=env)
    assert "not a registered Definition" not in p.stderr and "not in the registry" not in p.stderr, p.stderr[-3000:]
    assert p.returncode != 0 and "1 violation(s): GumbelTopPTokenSelect_v2{V=128256,S=32}: too-large, 32 Call(s)" in p.stderr, p.stderr[-3000:]


@pytest.mark.slow
def test_101s_program_view_and_word_lookups_resolve_every_call(tmp_path):
    """In a fresh interpreter: the query's program view of workload 101's request Program (as the manifest reads it), every Call's ports and its
    Definition as the word check specialises it, the sampler's unit rule, and the Commit's replay evaluator for every family."""
    build = _build_101(tmp_path)
    code = ("import json, sys\n"
            "from verity_vllm.query import program_view as PV\n"
            "from verity.ml.definitions import word as WQ\n"
            "from verity.ml.definitions.numerics.rows import EVALUATORS\n"
            "P = PV.from_instances(sys.argv[1] + '/build_request/instances.json.gz')\n"
            "specs = {(c.family, json.dumps(c.statics, sort_keys=True, default=str)): c for c in P.calls}\n"
            "for (fam, st), c in specs.items():\n"
            "    PV.definition_ports(c.definition)\n"
            "    WQ.specialization(fam, json.loads(st))\n"
            "sel = next(c for c in P.calls if c.family == 'GumbelTopPTokenSelect_v2')\n"
            "rule = WQ.unit_rule(WQ.specialization(sel.family, sel.statics))\n"

            "fams = sorted({c.family for c in P.calls if not c.family.startswith(('Const', 'Input'))})\n"
            "print(json.dumps({'calls': len(P.calls), 'specs': len(specs), 'sampler': [v['class'] for v in rule['violations']],\n"
            "                  'no_evaluator': [f for f in fams if f not in EVALUATORS]}))\n")
    p = _run(code, str(build))
    assert p.returncode == 0, p.stderr[-3000:]
    r = json.loads(p.stdout.strip().splitlines()[-1])
    assert r["calls"] > 0 and r["sampler"] == ["too-large"] and r["no_evaluator"] == [], r
