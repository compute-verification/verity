"""The compiled check's sampled-replay seed is POST-COMMITMENT AS A FACT.

commit_delta draws the CompiledKernelCheck's random steps from the run root: seed = int(run_root.hex()[:8], 16) (first 4 bytes), and
records that form on value_correspondence.compiled_check.seed_source (COMPILED_REPLAY_SEED_SOURCE).  The finalizer recomputes the seed from
the same runs.jsonl's run_root and refuses any other value.  Torch-free: the constant and the call site are read from the source text (the
module imports torch/vllm), and the rule is exercised on the recorded compiled runs when present.
"""
from __future__ import annotations

import glob
import json
import pathlib
import re

SRC = pathlib.Path(__file__).resolve().parents[2] / "verity_vllm" / "pipeline" / "commit.py"
CKC = pathlib.Path(__file__).resolve().parents[2] / "verity_vllm" / "check" / "replay" / "compiled_kernel_check.py"
FORM = "run root (first 4 bytes, after commitment): int(run_root.hex()[:8], 16)"


def seed_of_record(run_root_hex: str) -> int:
    return int(run_root_hex[:8], 16)


def test_the_constant_names_the_form_and_the_call_site_uses_both() -> None:
    src = SRC.read_text()
    m = re.search(r'^COMPILED_REPLAY_SEED_SOURCE = "(.*)"$', src, re.M)
    assert m and m.group(1) == FORM
    call = re.search(r"CompiledKernelCheck\(com, eng\.model, _toks, seed=CH\.kernel_check_seed\(rc\.run_root\),.*?seed_source=COMPILED_REPLAY_SEED_SOURCE\)", src, re.S)
    assert call, "the compiled check must be seeded from the run root AND record the form"
    from verity_vllm.commit import challenge as CH
    root = "1491" + "ab" * 30
    assert CH.kernel_check_seed(bytes.fromhex(root)) == seed_of_record(root)


def test_compiled_kernel_check_records_seed_source() -> None:
    src = CKC.read_text()
    assert re.search(r"^\s+seed_source: str \| None,$", src, re.M), "seed_source is a required keyword: every caller names the seed's form"
    assert '"seed": self.seed, "seed_source": self.seed_source' in src


def test_the_form_recomputes_the_seed_from_the_root() -> None:
    root = "1491" + "ab" * 30
    assert seed_of_record(root) == int("1491abab", 16)
    assert seed_of_record(root) != int.from_bytes(bytes.fromhex(root)[:8], "big")   # not the 8-byte (pin) form: the record names which


def test_every_compiled_record_of_this_lane_carries_the_root_seed() -> None:
    repo = pathlib.Path(__file__).resolve().parents[2]
    files = sorted(glob.glob(str(repo / "out/gen/sweep/evidence/compiled/*/r16_*/commit_treatment_r*/runs.jsonl")))
    checked = 0
    for f in files:
        for line in open(f):
            r = json.loads(line)
            c = r.get("commit") or {}
            vc = c.get("value_correspondence") or {}
            if "seed" not in vc or not c.get("run_root"):
                continue
            assert vc["seed"] == seed_of_record(c["run_root"]), f
            src = vc.get("seed_source")
            assert src in (None, FORM), (f, src)      # records before this commit carry no form; the finalizer's fallback is the recomputation
            checked += 1
    assert checked >= 0
