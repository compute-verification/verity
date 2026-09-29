"""Repository invariants (AGENTS.md "Where writing goes"): the tree holds executable code, tests, small fixtures and a
few maintained current-truth documents.  Lane reports, handoffs, findings and drafts live in the notes (``~/.research/notes``),
never in Git.  Generated output (rendered tables, reports, plots, evidence copies, run results) is never tracked: it is
regenerated from code plus the evidence store, or fetched from the store by ``art:`` id.  Model weights live in artifact
storage too: every tracked blob stays under ``BLOB_LIMIT`` bytes.

Enforced here rather than remembered.  ``.gitignore`` keeps the report-genre paths out of ``git add``; this test is the
backstop for anything that slips past it.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Basenames a tracked markdown file may have.  Everything else is a report, a handoff or a draft and belongs in the
#: notes store (cite it from here by ``note:<id>``).
ALLOWED_MD_BASENAMES = frozenset({"README.md", "AGENTS.md", "PROTOCOL.md", "DESIGN.md", "SKILL.md", "DISCREPANCIES.md",
                                  "CHANGELOG.md"})
#: Layout/spec documents left in place by the 2026-09-22 migration; a human folds them into PROTOCOL.md / README.md.
ALLOWED_EXTRA = (
    "backends/direct/V2_AIR.md",
    "backends/direct/encode/V2_SHAPE.md",
    "backends/vole/ASSUMPTIONS.md",
    "backends/flock/verifier/lean/soundness/ASSUMPTIONS.md",
    "backends/numerical/python/verity_numerical/redteam/TRANSCRIPT_CHECKLIST.md",
)
#: One file per entry behind an index document, so that no one document outgrows its cap: the Flock soundness package's
#: detail behind its named assumptions (its ``ASSUMPTIONS.md`` states them in full) and ligero-verify's discrepancies
#: (indexed by its ``DISCREPANCIES.md``).
SPLIT_DOCUMENT_DIRS = ("backends/flock/verifier/lean/soundness/assumptions/", "backends/ligero-verify/discrepancies/")
#: The one evidence file left in the tree: not read by code, kept with the rest of fixtures/redteam until the red team rules
#: on it.  The frozen campaign evidence that code reads (the overhead ledger, the run results the frozen backends' report
#: scripts compare against) is in the evidence store (``verity_numerical.bench.frozen``).
PINNED_EVIDENCE = ("fixtures/redteam/vu-battery/v1/a/results_r20260922-070024-e7d1.jsonl",)
#: Generated output, by where it sits or what it is: no tracked path has one of these directories, suffixes or stamps.
GENERATED_DIRS = frozenset({"reports", "results", "renders", "previews", "plots", "evidence", "assets", "outputs"})
GENERATED_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".pdf", ".html", ".htm"})
#: ``<YYYYMMDDTHHMMZ>-...``: a stamped render (the steward's ``<stamp>-tables.json``) or a lane note.
STAMPED = re.compile(r"^\d{8}T\d{4}Z[-_.]")

REPORT_GENRE = re.compile(r"handoff|report|urgent|draft|triage|status|^results?_", re.IGNORECASE)
#: Source files are exempt from the basename rule (``report.py`` is a script, not a report).
SOURCE_SUFFIXES = frozenset({".py", ".rs", ".toml", ".sh", ".cu", ".c", ".h", ".lock"})

MD_SIZE_CAP = 48 * 1024
PROTOCOL_SIZE_CAP = 128 * 1024
BLOB_LIMIT = 256 * 1024  # a tracked file holds at most this, except source code (CODE_SUFFIXES) and ALLOWLIST
#: Suffixes of source, exempt from BLOB_LIMIT (a generated module or a pinned Lean file can be larger); markdown has its own caps.
CODE_SUFFIXES = SOURCE_SUFFIXES | {".lean", ".cuh", ".cpp", ".md"}
#: Data over BLOB_LIMIT that is not a test fixture: runtime tables a module loads, the generated census, three vLLM workload specs
#: and two vLLM logs. Each is capped at its size rounded up to 64 KiB, so none can grow past that and nothing joins them
#: silently. A test fixture over BLOB_LIMIT is registered in fixtures/artifacts.json instead and fetched
#: (`research data fetch-fixtures`).
ALLOWLIST = {
    "census/workloads.json": 384 * 1024,
    "integrations/vllm/data/logs/m1.jsonl.gz": 576 * 1024,
    "integrations/vllm/data/logs/m6/log.jsonl.gz": 576 * 1024,
    "integrations/vllm/verity_vllm/check/tables/B0-divergence-20260907T1604Z/cos_sin_cache.npy": 1088 * 1024,
    "integrations/vllm/verity_vllm/program/registry/quarantine/dense/tables/mufu_tanh_sm89.xzblocks": 448 * 1024,
    "integrations/vllm/workloads/llama32-1b__bf16__l40s__tp1__b128__i1024__o128__mixed__greedy__bi-eager.json": 512 * 1024,
    "integrations/vllm/workloads/olmoe-1b-7b__bf16__l40s__tp1__b128__i1024__o128__mixed__greedy__bi-eager.json": 512 * 1024,
    "integrations/vllm/workloads/qwen25-15b__bf16__l40s__tp1__b128__i1024__o128__mixed__greedy__bi-eager.json": 512 * 1024,
}
#: Fixtures over BLOB_LIMIT that the fixture migration registers and untracks; each is capped at its own size until then. The
#: migration empties this, and nothing may be added to it.
PENDING_REGISTRATION = {
    "backends/direct/fixtures/crosscheck_units.json": 307575,
    "backends/gkr/fixtures/babybear-transcripts/transcript_vu4.json": 2382717,
    "backends/gkr/fixtures/babybear-transcripts/transcript_vu64.json": 3685380,
    "backends/gkr/fixtures/v2_synth/circuit.txt": 2808149,
    "backends/ligerito-verify/fixtures/forged/noncanonical_v2.proof.neg": 440652,
    "backends/ligerito-verify/fixtures/forged/virtrow_y16_lgsc0003.json": 806210,
    "backends/ligerito-verify/fixtures/lgto/fp8ada_l256_legacy/honest_fiat-shamir.lgto": 323797,
    "backends/ligerito-verify/fixtures/lgto/fp8ada_l256_v1/canon_y_offend.lgto.neg": 328409,
    "backends/ligerito-verify/fixtures/lgto/fp8ada_l256_v1/honest.lgto": 328441,
    "backends/ligerito-verify/fixtures/lgto/fp8ada_l256_v1/v1_06.lgto.neg": 328761,
    "backends/ligerito-verify/fixtures/lgto/fp8ada_l256_v1/v1_07.lgto.neg": 328985,
    "backends/ligerito-verify/fixtures/lgto/fp8ada_l256_v1/v1_08.lgto.neg": 327929,
    "backends/ligerito-verify/fixtures/lgto/fp8ada_l256_v1/v1_09.lgto.neg": 327993,
    "backends/ligerito-verify/fixtures/lgto/fp8ada_l256_v1/v1_10.lgto.neg": 327001,
    "backends/ligerito-verify/fixtures/lgto/fp8ada_l256_v1/zk_honest.lgto": 353781,
    "backends/ligerito-verify/fixtures/lgto/fp8ada_l256_zk_lgsc4/cheat_all.lgto.neg": 432940,
    "backends/ligerito-verify/fixtures/lgto/fp8ada_l256_zk_lgsc4/claim_value.lgto.neg": 434028,
    "backends/ligerito-verify/fixtures/lgto/fp8ada_l256_zk_lgsc4/honest.lgto": 434028,
    "backends/ligerito-verify/fixtures/lgto/fp8ada_l256_zk_lgsc4/zk_mnext.lgto.neg": 433804,
    "backends/ligerito-verify/fixtures/lgto/fp8ada_l256_zk_lgsc4/zk_next0.lgto.neg": 432492,
    "backends/ligerito-verify/fixtures/session/fp8ada_l256_legacy/batch_live_0.lgto": 324654,
    "backends/ligerito-verify/fixtures/session/fp8ada_l256_zk_live/context_grinding.lgto": 429829,
    "backends/ligerito-verify/fixtures/session/fp8ada_l256_zk_live/honest.lgto": 429925,
    "backends/ligerito-verify/fixtures/session/fp8ada_l256_zk_live/label_grinding.lgto": 429893,
    "backends/ligerito-verify/fixtures/sumcheck/fp8ada_l64_S2.json": 537529,
    "backends/ligerito-verify/fixtures/sumcheck/lgsc0003_fp8ada_l64_S2.json": 805689,
    "backends/ligerito-verify/fixtures/sumcheck/lgsc0004_fp8ada_l64_S2.json": 4146969,
    "backends/ligerito-verify/fixtures/sumcheck/lgsc0004_fp8ada_l64_S2_zksmall.json": 1198788,
    "backends/ligero-verify/fixtures/b1.proof": 2814256,
    "backends/ligero-verify/fixtures/bf16-ampere-hash/sub_00.proof": 4602772,
    "backends/ligero-verify/fixtures/bf16-ampere-hash/system.bin": 903707,
    "backends/ligero-verify/fixtures/bf16-hopper-ajtai-n128/sub_00.proof": 3701620,
    "backends/ligero-verify/fixtures/bf16-hopper-ajtai-n128/system.bin": 811783,
    "backends/ligero-verify/fixtures/fp4-nvf4-hash/sub_00.proof": 3530008,
    "backends/ligero-verify/fixtures/fp4-nvf4-hash/system.bin": 832135,
    "backends/ligero-verify/fixtures/fp4-nvf4/sub_00.proof": 1723100,
    "backends/ligero-verify/fixtures/fp4-nvf4/sub_00.stmt": 573513,
    "backends/ligero-verify/fixtures/fp8-ada-ajtai-n64-g1-decoy/sub_00.proof": 5911564,
    "backends/ligero-verify/fixtures/fp8-ada-ajtai-n64-g1-decoy/system.bin": 573831,
    "backends/ligero-verify/fixtures/fp8-ada-ajtai-n64/sub_00.proof": 3844504,
    "backends/ligero-verify/fixtures/fp8-ada-ajtai-n64/system.bin": 570247,
    "backends/ligero-verify/fixtures/fp8-ada-hash/sub_00.proof": 4866616,
    "backends/ligero-verify/fixtures/fp8-ada-hash/system.bin": 926203,
    "backends/ligero-verify/fixtures/fp8-ada/sub_00.proof": 3467980,
    "backends/ligero-verify/fixtures/fp8-ada/sub_00.stmt": 278600,
    "backends/ligero-verify/fixtures/steps-pin/bf16-hopper-ajtai-n128-collide-P/sub_00.proof": 5414500,
    "backends/ligero-verify/fixtures/steps-pin/bf16-hopper-ajtai-n128-collide-Q/sub_00.proof": 5414500,
    "backends/ligero-verify/fixtures/steps-pin/fp8-ada-ajtai-n64-collide-P/sub_00.proof": 5624668,
    "backends/ligero-verify/fixtures/steps-pin/fp8-ada-ajtai-n64-collide-Q/sub_00.proof": 5624668,
    "backends/ligero-verify/fixtures/steps-pin/fp8-ada-ajtai-n64-steps32/sub_00.proof": 5624668,
    "backends/ligero-verify/fixtures/steps-pin/fp8-ada-bare-steps32/sub_00.proof": 4413700,
    "backends/ligero-verify/fixtures/steps-pin/fp8-ada-poseidon2-steps32/sub_00.proof": 7128092,
    "backends/ligero-verify/fixtures/systems/bf16-ampere-v3x4.system.bin": 695017,
    "backends/ligero-verify/fixtures/systems/bf16-ampere-x4.system.bin": 950485,
    "backends/ligero-verify/fixtures/systems/bf16-hopper-v3x4.system.bin": 630353,
    "backends/ligero-verify/fixtures/systems/bf16-hopper-x4.system.bin": 835397,
    "backends/ligero-verify/fixtures/systems/fp8-ada-v3x4.system.bin": 662597,
    "backends/ligero-verify/fixtures/systems/fp8-ada-x4.system.bin": 975077,
    "backends/ligero-verify/fixtures/systems/fp8-hopper-v3x4.system.bin": 599029,
    "backends/ligero-verify/fixtures/systems/fp8-hopper-x4.system.bin": 880197,
    "backends/sp1/common/tests/embed_vectors.json": 777940,
    "backends/sp1/common/tests/reduce_vectors.json": 4479185,
    "backends/sp1/common/tests/vectors.json": 278473,
    "fixtures/lean-vectors/plan.json": 314762,
    "fixtures/lean-vectors/q7.json": 1583778,
    "fixtures/lean-vectors/tc-ampere-bf16-total-hw.json": 2623281,
    "fixtures/lean-vectors/tc-ampere-bf16-total.json": 409812,
    "fixtures/lean-vectors/tc-ampere-bf16.json": 409776,
    "fixtures/redteam-v2/gkr/circuit.txt": 2808149,
    "fixtures/redteam-v2/gkr/neg/b1_honest/circuit.txt": 2808149,
    "fixtures/redteam-v2/gkr/neg/b1_wrong_word/circuit.txt": 2808149,
    "fixtures/redteam-v2/gkr/neg/chain_spec_stripped/circuit.txt": 2808149,
    "fixtures/redteam-v2/gkr/neg/epilogue_one_zero/circuit.txt": 2808149,
    "fixtures/redteam-v2/gkr/neg/epilogue_ovf_flip/circuit.txt": 2808149,
    "fixtures/redteam-v2/gkr/neg/epilogue_wrong_unit/circuit.txt": 2808149,
    "fixtures/redteam-v2/gkr/neg/honest/circuit.txt": 2808149,
    "fixtures/redteam-v2/gkr/neg/swap_vu_epilogues/circuit.txt": 2808149,
    "fixtures/redteam-v2/gkr/neg/unit0_noninitial_state/circuit.txt": 2808149,
    "fixtures/tc/autoproof-probe-vectors.json": 660839,
    "fixtures/typed-obligation-v0/stage2--qkv-proj-decode-0-k1536.json": 4464579,
    "integrations/vllm/fixtures/B0-divergence-20260907T1604Z/layers.0.self_attn.rotary_emb/compiled/in_qkv.npy": 983168,
    "integrations/vllm/fixtures/B0-divergence-20260907T1604Z/layers.0.self_attn.rotary_emb/compiled/out_q.npy": 589952,
    "integrations/vllm/fixtures/B0-divergence-20260907T1604Z/layers.0.self_attn.rotary_emb/eager/in_q.npy": 589952,
    "integrations/vllm/fixtures/B0-divergence-20260907T1604Z/layers.0.self_attn.rotary_emb/eager/out_q.npy": 589952,
    "integrations/vllm/tests/commit/fa2_prototype/fixtures/p0/p0_40x200_h1kv1.npz": 373471,
    "integrations/vllm/tests/commit/fa2_prototype/fixtures/p0/p0_64x128_h1kv1.npz": 346859,
    "integrations/vllm/tests/commit/fa2_prototype/fixtures/p0/p0_64x300_h1kv1.npz": 649495,
    "integrations/vllm/tests/program/data/topp_split_fixture/fixture.npz": 294033,
    "integrations/vllm/tests/program/data/topp_split_fixture/fstoch01_rows.npz": 618071,
    "integrations/vllm/tests/regression/expected/gemma2-2b__bf16__l40s__tp1__b8__i1024__o128__mixed__greedy__bi-eager.json": 738923,
    "integrations/vllm/tests/regression/expected/llama32-1b__bf16__l40s__tp1__b1__i4096__o512__mixed__greedy__bi-eager.json": 291254,
    "integrations/vllm/tests/regression/expected/mistral-7b__bf16__l40s__tp1__b8__i1024__o128__mixed__greedy__bi-eager.json": 2066198,
    "integrations/vllm/tests/regression/expected/olmoe-1b-7b__bf16__l40s__tp1__b32__i1024__o128__mixed-arrivals__greedy__bi-eager.json": 4476805,
    "integrations/vllm/tests/regression/expected/olmoe-1b-7b__bf16__l40s__tp1__b32__i1024__o128__mixed__greedy__bi-eager.json": 5102854,
    "integrations/vllm/tests/regression/expected/qwen25-15b__bf16__l40s__tp1__b1__i4096__o512__mixed__greedy__bi-eager.json": 789638,
    "integrations/vllm/tests/regression/expected/qwen3-4b-fp8__fp8__h100__tp1__b8__i1024__o128__mixed__greedy__bi-eager.json": 2616291,
    "integrations/vllm/tests/regression/expected/qwen3-4b__bf16__h100__tp1__b8__i1024__o128__mixed__greedy__bi-eager.json": 2928985,
    "packages/verity/tests/ml/fixtures/gemm-B1-b3eba618b1b5-20260907T1731Z/lm_head_decode_0/out_row.u16": 303872,
    "packages/verity/tests/ml/fixtures/gemm-B1-b3eba618b1b5-20260907T1731Z/lm_head_decode_1/out_row.u16": 303872,
    "packages/verity/tests/ml/fixtures/gemm-B1-b3eba618b1b5-20260907T1731Z/lm_head_decode_2/out_row.u16": 303872,
    "packages/verity/tests/ml/fixtures/gemm-B1-b3eba618b1b5-20260907T1731Z/lm_head_prefill_0/out_row.u16": 303872,
    "packages/verity/tests/ml/fixtures/gemm-B1-b3eba618b1b5-20260907T1731Z/lm_head_prefill_1/out_row.u16": 303872,
    "packages/verity/tests/ml/fixtures/gemm-B1-b3eba618b1b5-20260907T1731Z/lm_head_prefill_2/out_row.u16": 303872,
    "packages/verity/tests/ml/fixtures/tc-hopper-2026-09-07/special.reduced.json.gz": 2696343,
    "packages/verity/tests/ml/fixtures/tc-hopper-wgmma-bf16-2026-09-26/records.json.gz": 345362,
    "packages/verity/tests/ml/fixtures/tc-total-2026-09-07/4090/cvt_special.json": 336730,
    "packages/verity/tests/ml/fixtures/tc-total-2026-09-07/4090/mma_special.reduced.json.gz": 2041088,
    "packages/verity/tests/ml/fixtures/tc-total-2026-09-07/a100/cvt_special.json": 336728,
    "packages/verity/tests/ml/fixtures/tc-total-2026-09-07/a100/mma_special.reduced.json.gz": 2041041,
    "packages/verity/tests/ml/fixtures/vectors/tc-ampere-bf16-total-hw.json": 2623281,
    "packages/verity/tests/ml/fixtures/vectors/tc-ampere-bf16-total.json": 409812,
}
#: Tracked data (everything but CODE_SUFFIXES and PENDING_REGISTRATION) in the whole tree, so that many small files cannot
#: add up to what one large one may not be.
DATA_BUDGET = 40 * 1024 * 1024
REGISTRY = "fixtures/artifacts.json"


def _tracked() -> list[str]:
    if not (ROOT / ".git").exists():                # a shipped `git archive` (research run --source): the commit's tree as the
        ready = ROOT / "READY.json"                 # machine's bare repository holds it, never what steps have built in it since
        bare = ROOT.parent.parent / "git" / "verity.git"
        if ready.exists() and bare.exists():
            tree = json.loads(ready.read_text())["tree"]
            out = subprocess.run(["git", "--git-dir", str(bare), "ls-tree", "-r", "-z", "--name-only", tree], check=True,
                                 capture_output=True).stdout
            return sorted(p.decode() for p in out.split(b"\0") if p)
        out = []
        for d, dirs, files in os.walk(ROOT):
            dirs[:] = [x for x in dirs if x not in (".venv", "__pycache__", ".lake")]
            out += [os.path.relpath(os.path.join(d, f), ROOT) for f in files]
        registered = [v.rstrip("/") for k, v in json.loads((ROOT / "fixtures" / "artifacts.json").read_text()).items()
                      if k.startswith("art:") and isinstance(v, str)] if (ROOT / "fixtures" / "artifacts.json").is_file() else []
        return sorted(p for p in out if p != "READY.json" and not any(p == r or p.startswith(r + "/") for r in registered))
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True).stdout
    return [p.decode() for p in out.split(b"\0") if p]


def test_tracked_markdown_is_maintained_documentation() -> None:
    offenders = [p for p in _tracked() if p.endswith(".md") and os.path.basename(p) not in ALLOWED_MD_BASENAMES
                 and not p.endswith(".template.md") and p not in ALLOWED_EXTRA
                 and not p.startswith(SPLIT_DOCUMENT_DIRS)]
    assert not offenders, (
        "tracked markdown that is not a maintained document (reports, handoffs and drafts belong in ~/.research/notes):\n"
        + "\n".join(offenders)
    )


def test_no_report_genre_basenames() -> None:
    offenders = [p for p in _tracked() if REPORT_GENRE.search(os.path.basename(p))
                 and os.path.splitext(p)[1] not in SOURCE_SUFFIXES and p not in PINNED_EVIDENCE]
    assert not offenders, "report-genre files belong in ~/.research/notes/lanes/<lane>/:\n" + "\n".join(offenders)


def generated_output(path: str) -> str | None:
    """Why ``path`` looks like generated output, or None."""
    parts = path.split("/")
    if hit := GENERATED_DIRS.intersection(parts[:-1]):
        return f"under {sorted(hit)[0]}/"
    if os.path.splitext(path)[1].lower() in GENERATED_SUFFIXES:
        return f"a {os.path.splitext(path)[1].lower()} render"
    if STAMPED.match(parts[-1]):
        return "a stamped render or note"
    return None


def test_generated_output_patterns() -> None:
    assert generated_output("backends/numerical/reports/ledger/b-zk.jsonl") == "under reports/"
    assert generated_output("renders/daily/20260927T0857Z-tables.json") == "under renders/"
    assert generated_output("lanes/x/evidence/run.log") == "under evidence/"
    assert generated_output("backends/numerical/plot_overhead.png") == "a .png render"
    assert generated_output("integrations/vllm/20260929T0300Z-tables.json") == "a stamped render or note"
    assert generated_output("backends/numerical/python/verity_numerical/bench/tables.py") is None
    assert generated_output("tools/research/tests/fixtures/run_v0.2/out/result.json") is None


def test_no_generated_output() -> None:
    """Rendered tables, reports, plots, evidence copies and run results are regenerated from code plus the evidence store, or
    fetched from it by ``art:`` id (``research data put --preserve``); none is tracked."""
    offenders = [f"{p} ({why})" for p in _tracked() if p not in PINNED_EVIDENCE and (why := generated_output(p))]
    assert not offenders, (
        "generated output belongs in the evidence store, not Git (AGENTS.md \"Where writing goes\"):\n" + "\n".join(offenders)
    )


def test_pinned_evidence_still_exists() -> None:
    """Drop an entry from PINNED_EVIDENCE when its file leaves the tree."""
    tracked = set(_tracked())
    missing = [p for p in PINNED_EVIDENCE if p not in tracked]
    assert not missing, "PINNED_EVIDENCE lists files that are no longer tracked; remove them:\n" + "\n".join(missing)


def test_markdown_size_caps() -> None:
    offenders = []
    for p in _tracked():
        if not p.endswith(".md"):
            continue
        cap = PROTOCOL_SIZE_CAP if os.path.basename(p) == "PROTOCOL.md" else MD_SIZE_CAP
        try:
            size = os.path.getsize(ROOT / p)
        except FileNotFoundError:  # tracked but deleted in the working tree
            continue
        if size > cap:
            offenders.append(f"{p}: {size / 1024:.0f} KB > {cap // 1024} KB")
    assert not offenders, "maintained documents are small; move detail to the notes store:\n" + "\n".join(offenders)


#: Frozen backends whose other test files need torch, a GPU or cargo, and were never part of `check` (backends/README.md).
UNSUITED_TEST_DIRS = ("backends/direct/", "backends/gkr/", "backends/shared/")


def _testpaths(tracked: list[str]) -> list[str]:
    """Every suite's testpaths, relative to the root: a suite is a directory whose pyproject.toml configures pytest."""
    testpaths = []
    for p in tracked:
        if os.path.basename(p) == "pyproject.toml":
            ini = tomllib.loads((ROOT / p).read_text()).get("tool", {}).get("pytest", {}).get("ini_options")
            if ini is not None:
                testpaths += [os.path.normpath(os.path.join(os.path.dirname(p), t)) for t in ini.get("testpaths", ["."])]
    return testpaths


def test_every_test_file_is_in_a_suite() -> None:
    """`tools/check/suites.py` runs suites, so a test file outside every suite's testpaths would never run. A suite is a
    directory whose pyproject.toml configures pytest; give a new package one, or put the test in the package it tests."""
    tracked = _tracked()
    testpaths = _testpaths(tracked)
    tests = [p for p in tracked if p.endswith(".py") and (os.path.basename(p).startswith("test_") or p.endswith("_test.py"))]
    offenders = [p for p in tests if not any(t == "." or p.startswith(t + "/") for t in testpaths) and not p.startswith(UNSUITED_TEST_DIRS)]
    assert not offenders, "test files that no suite runs:\n" + "\n".join(offenders)


#: Credential formats with a fixed shape, as secret scanners (GitGuardian's PR check among them) recognize them.
CREDENTIAL_FORMATS = {
    "an AWS access key id": r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b",
    "a private key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "a GitHub token": r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,})",
    "a Slack token": r"\bxox[abposr]-[A-Za-z0-9-]{10,}",
    "a JSON web token": r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.",
    "a RunPod API key": r"\brpa_[A-Za-z0-9]{20,}",
}
_FORMATS = re.compile("|".join(f"(?P<f{i}>{p})" for i, p in enumerate(CREDENTIAL_FORMATS.values())))
#: A file can match a format only if it holds one of these.
_FORMAT_HINTS = ("AKIA", "ASIA", "PRIVATE KEY", "ghp_", "gho_", "ghu_", "ghs_", "ghr_", "github_pat_", "xox", "eyJ", "rpa_")
#: A value assigned to a credential name: `name = "value"`, `"name": "value"`, or `NAME=value` inside a string.
CREDENTIAL_ASSIGNMENTS = (re.compile(r"(?<![\w-])(?P<name>[A-Za-z_][\w-]*)[\"']?\s*[:=]\s*[bru]{0,2}[\"'](?P<value>[^\"'\s]{12,})[\"']"),
                          re.compile(r"\b(?P<name>[A-Z][A-Z0-9_]*)=(?P<value>[^\"'\s]{12,})"))
CREDENTIAL_WORDS = {"secret", "token", "password", "passwd", "credential", "credentials"}
CREDENTIAL_PAIRS = {("api", "key"), ("access", "key"), ("secret", "key"), ("private", "key"), ("auth", "token")}
#: A file holds a credential name only if its lowercased text holds one of these.
_NAME_HINTS = ("secret", "token", "passw", "credential", "apikey", "api_key", "api-key", "accesskey", "access_key", "access-key",
               "privatekey", "private_key", "private-key")
#: Binary data, never read; and data files (vectors, fixtures), read only for the fixed credential formats.
_BINARY = frozenset({".bin", ".npy", ".npz", ".lgto", ".neg", ".proof", ".gz", ".png", ".pdf", ".olean", ".ilean", ".tar", ".zip"})
_DATA = frozenset({".json", ".jsonl", ".csv"})
#: Values that say they are not real: documented examples (AWS's SigV4 test vector keys contain EXAMPLE) and placeholders.
PLACEHOLDER = re.compile(r"EXAMPLE|example|dummy|DUMMY|fake|FAKE|placeholder|redacted|xxxx|XXXX|\*\*\*\*|[<{$]")


def _credential_name(name: str) -> bool:
    parts = [p.lower() for p in re.split(r"[_\-.]|(?<=[a-z0-9])(?=[A-Z])", name) if p]
    return bool(parts) and (parts[-1] in CREDENTIAL_WORDS or tuple(parts[-2:]) in CREDENTIAL_PAIRS)


def _looks_random(value: str) -> bool:
    counts = [value.count(c) for c in set(value)]
    entropy = -sum(n / len(value) * math.log2(n / len(value)) for n in counts)
    classes = sum(bool(re.search(p, value)) for p in (r"[a-z]", r"[A-Z]", r"[0-9]", r"[^A-Za-z0-9]"))
    return entropy >= 3.5 and classes >= 3


def test_no_secret_like_literals() -> None:
    """No tracked file holds something a secret scanner reads as a credential. A dummy credential is short and says what it is
    (`"dummy"`, `"not-to-be-copied"`); a test that needs a realistic one, to test redaction or scanning, splits it into adjacent
    string literals, which Python joins and no scanner reads as one."""
    offenders, formats = [], list(CREDENTIAL_FORMATS)
    for p in _tracked():
        path, suffix = ROOT / p, os.path.splitext(p)[1]
        try:
            if suffix in _BINARY or path.stat().st_size > 1024 * 1024:
                continue
            text = path.read_text()
        except (FileNotFoundError, IsADirectoryError, UnicodeDecodeError):
            continue
        found = [(m, formats[int(m.lastgroup[1:])]) for m in _FORMATS.finditer(text) if not PLACEHOLDER.search(m.group(0))] \
            if any(h in text for h in _FORMAT_HINTS) else []
        low = text.lower()
        if suffix not in _DATA and any(w in low for w in _NAME_HINTS):
            found += [(m, f"a random-looking {m['name']}") for pattern in CREDENTIAL_ASSIGNMENTS for m in pattern.finditer(text)
                      if _credential_name(m["name"]) and _looks_random(m["value"]) and not PLACEHOLDER.search(m["value"])]
        offenders += [f"{p}:{text.count(chr(10), 0, m.start()) + 1}: {what}" for m, what in found]
    assert not offenders, "credential-like literals (see this test's docstring for what to write instead):\n" + "\n".join(offenders)


def test_no_tracked_blob_exceeds_limit() -> None:
    offenders = []
    for p in _tracked():
        if os.path.splitext(p)[1] in CODE_SUFFIXES:
            continue
        try:
            size = os.path.getsize(ROOT / p)
        except FileNotFoundError:  # tracked but deleted in the working tree
            continue
        if size > ALLOWLIST.get(p, PENDING_REGISTRATION.get(p, BLOB_LIMIT)):
            offenders.append(f"{p}: {size / 1024:.0f} KiB")
    assert not offenders, (
        f"tracked files over {BLOB_LIMIT // 1024} KiB: register a fixture in {REGISTRY} and gitignore its path "
        "(`research data put --kind fixture/v1 --preserve`, then `research data export`); evidence and run outputs belong in "
        "artifact storage, not Git:\n" + "\n".join(offenders)
    )


def test_allowlists_name_tracked_files() -> None:
    tracked = set(_tracked())
    stale = sorted(p for p in (*ALLOWLIST, *PENDING_REGISTRATION) if p not in tracked)
    assert not stale, "ALLOWLIST / PENDING_REGISTRATION name files that are no longer tracked; remove them:\n" + "\n".join(stale)


def test_tracked_data_within_budget() -> None:
    total = 0
    for p in _tracked():
        if os.path.splitext(p)[1] in CODE_SUFFIXES or p in PENDING_REGISTRATION:
            continue
        try:
            total += os.path.getsize(ROOT / p)
        except FileNotFoundError:
            continue
    assert total <= DATA_BUDGET, (f"tracked data is {total / 2**20:.1f} MiB, over the {DATA_BUDGET / 2**20:.0f} MiB budget: register large "
                                  f"fixtures in {REGISTRY} rather than tracking them")


def test_fetched_fixtures_are_gitignored() -> None:
    """An entry of the fixture registry that git does not track is fetched (`research data fetch-fixtures`), so its path must be
    ignored: a fetched copy must never be committed back."""
    if not (ROOT / ".git").exists():
        return
    doc = json.loads((ROOT / REGISTRY).read_text(encoding="utf-8"))
    tracked = set(_tracked())
    fetched = sorted(v for k, v in doc.items() if k.startswith("art:") and isinstance(v, str)
                     and v not in tracked and not any(t.startswith(v.rstrip("/") + "/") for t in tracked))
    if not fetched:
        return
    out = subprocess.run(["git", "check-ignore", "--no-index", "--stdin"], cwd=ROOT, input="\n".join(fetched) + "\n",
                         capture_output=True, text=True).stdout.split("\n")
    unignored = sorted(set(fetched) - {x for x in out if x})
    assert not unignored, "registry entries git does not track must be gitignored (add them to .gitignore):\n" + "\n".join(unignored)
