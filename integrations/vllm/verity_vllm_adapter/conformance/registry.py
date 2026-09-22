"""Committed artefacts per port for `python -m verity_vllm_adapter.conformance reproduce` (no GPU).

Paths are repository-relative. Every entry is an *existing* artefact produced by the existing modules; the
harness only reads them. A port absent here reproduces nothing (`reproduce` refuses it), which is the state
of the HP/TP ports until their pods deliver results.

Keys per port: `results` (rows / bundles / commit stores), `correspondence` (TA1 files or directories),
`obligations` + `proof_records` (C4), `int_records` (C5: INT `run_composed` output directories — none committed
yet; `ladder.py` documents the contract).
"""

from __future__ import annotations

REPRODUCE: dict[str, dict[str, list[str]]] = {
    # RTX 4090 (Ada), eager-production profile. 40f0…: the REPORT §1 row (W1 verifier, diagnostic challenge).
    # b3eb…: W3c's row whose capture is the source of PW's proved GEMM coordinate (Level 4). c37c…: the first
    # bundle re-verified under the fixed verifier (REPORT §1). Typed correspondence for this static
    # configuration exists only on the A100 capture (port b1-a100-eager): Level 2 is not attempted here.
    # INT-B1-20260908T0941Z: lane INT3's B1 record of the composed command (docs/e2e-composed.md §12) —
    # RC selective preparation (k=2/12), PASS, re-verified by the pod's second process (reverify/reverify.json).
    "b1-eager": {
        "results": ["fixtures/results/B1-40f0cebeb670-20260907T1242Z", "fixtures/results/B1-b3eba618b1b5-20260907T1731Z",
                    "fixtures/results/B1-c37c3f0b8363-20260907T1328Z"],
        "obligations": ["fixtures/typed-obligation-v0/stage2--qkv-proj-decode-0-k1536.json"],
        "proof_records": ["docs/data/verified-backend/packed-word-proof"],
        "int_records": ["fixtures/e2e-composed/INT-B1-20260908T0941Z"],
    },
    # A100 (Ampere), same static configuration: the REPORT §1 A100 row and TA1's commit store + correspondence.
    "b1-a100-eager": {
        "results": ["fixtures/results/B1-3c19aef8539d-20260907T1317Z", "fixtures/results/TA1-typed-b1-20260907/B1"],
        "correspondence": ["fixtures/results/TA1-typed-b1-20260907/B1/correspondence"],
    },
    "b0-eager": {
        "results": ["fixtures/results/B0-9f5fa192a813-20260907T1225Z", "fixtures/results/B0-1d8aa45d3f0e-20260907T0849Z"],
    },
    # TA1's B0 commit store (A100): correspondence only — no production/compare/verify artefacts are committed
    # for that run, so Level 1 and Level 3 are not shown for this port.
    "b0-a100-eager": {
        "results": ["fixtures/results/TA1-typed-b1-20260907/B0"],
        "correspondence": ["fixtures/results/TA1-typed-b1-20260907/B0/correspondence"],
    },
    # W1c/W1cF: stock compiled production on the 4090, launch-level capture in a compiled-nograph replay.
    "b0-compiled": {"results": ["fixtures/results/B0-ab014e164e43-20260907T2006Z"]},
    # W1d: stock compiled production on the A100, compiled-preserving capture from inside the Inductor graph.
    "b0-compiled-fx": {"results": ["fixtures/results/B0-d48d41099714-20260907T1826Z", "fixtures/results/B0-d48d41099714-20260907T1857Z",
                                   "fixtures/results/B0-d48d41099714-20260907T1935Z"]},
    "b1-compiled": {"results": ["fixtures/results/B1-bdd4a899b1dd-20260907T2138Z", "fixtures/results/B1-bdd4a899b1dd-20260907T2232Z"]},
    # W9: Qwen1.5-MoE-A2.7B-Chat on the H100, eager-production profile.
    "m1-eager": {"results": ["fixtures/results/M1-0c9af072e5a9-20260907T2128Z"]},
    # TP: B1 with tensor_parallel_size=2 on 2x A100 (NVLink). One row: two productions (production/,
    # production-r2/), the joined per-rank commit store's metadata (the 84 GB value stores stay on the pod),
    # two references, compare.json, the verify row (FAIL: the verifier's semantic profile is Llama-only),
    # and the TP=2 correspondence over requests 917b9faa and 147eed15 (`docs/vllm-poc/tp2-profile.md`).
    # `correspondence/` is the lead's re-run from the merged tree under `RMSNormFusedCuda@2` (05:32Z: 1,314 leaves exact,
    # 336 rank-1 copies equal, 0 mismatch); TP's own `@1` run (3 tie-rounding mismatches) is kept as `correspondence-rmsnormv1/`.
    "b1-tp2": {
        "results": ["fixtures/results/B1-7f26cdcedcd0-20260908T0229Z"],
        "correspondence": ["fixtures/results/B1-7f26cdcedcd0-20260908T0229Z/correspondence"],
    },
    # QZ (`docs/vllm-poc/fp8.md`): B1 under `--quantization fp8` on an H100, eager, batch-invariant. Committed: both
    # production passes (production/, production-r2/: same profile id, 16/16 identical), the commit store's metadata
    # (`commit/ru` 32 GB and `weights/values.bin` stay on the pod / are reconstructible from the pinned build), two
    # references, compare.json, the verify-security row (FAIL: no shipped semantic profile admits `quantization='fp8'`,
    # C3 `REPRESENTATION_MISSING`), and the correspondence corr-A + corr-B2 (614 exact, 0 mismatch); corr-B's
    # missing-operand run is kept as `correspondence-diagnostic/`.
    "b1-fp8": {
        "results": ["fixtures/results/B1-0ca394ec17b9-20260908T0601Z"],
        "correspondence": ["fixtures/results/B1-0ca394ec17b9-20260908T0601Z/correspondence"],
    },
    # CF2 (`docs/vllm-poc/held-out-cell.md`): the held-out cell, B1 + Hopper + compiled (Inductor, W1d compiled-fx capture)
    # on an H100, scored against CF1's written prediction. Committed: both productions, the commit store's metadata (the
    # 27 GB value store + 3 GB weights stayed on the pod), two references, compare.json, four verify records (refused /
    # fixed / forced-profile / -noseed), three Hopper correspondence runs (836 records) + the Ampere compiled control
    # (348), scorecards and the arch-independence word table under analysis/.
    "b1-hopper-compiled": {
        "results": ["fixtures/results/B1-0fc5b2ce24f2-20260908T0642Z"],
        "correspondence": ["fixtures/results/B1-0fc5b2ce24f2-20260908T0642Z/correspondence"],
    },
    # SD (`docs/vllm-poc/speculative.md`): two-model speculative decoding (B1 target + Qwen2.5-0.5B draft, k = 3, greedy) on
    # an A100. Committed: both productions (prod-1/, prod-2/), the commit store's metadata, compare.json, the 82 per-step
    # branch correspondence files (corr/: target VerifyBranch + draft ServeDraft per step) and the batch-invariance
    # records (bi/); `harness/` is the lead's symlink view in the harness layout (production, production-r2, commit,
    # correspondence, compare.json). The 1.2 GB value store and the two references stay in the archive.
    "b1-spec": {
        "results": ["fixtures/results/SD-b1-spec-7d678ac1ddee-20260908T0745Z/harness"],
        "correspondence": ["fixtures/results/SD-b1-spec-7d678ac1ddee-20260908T0745Z/harness/correspondence"],
    },
}
