# A GPU prover: the packed/graphed kernel path fails on the A100 where the torch path passes (coordinator, 2026-09-22 10:53Z)

Post-merge smoke of `main` (451b6e8: a-gpu2 + a-gpu-v2 + a-verifier fixes) on `vy-a100b` (A100 80GB), instance sets
exported on that pod by a-gpu-v2 (`/workspace/bb/pos4096` v1, `/workspace/v2l/pos4096` v2 limb-form; the a-gpu-v2
lane's own 7.93 s v1 control and 5.73 s v2 rows ran on exactly these sets from its branch head bb7a5c0).

| run id | source | path | result |
|---|---|---|---|
| r20260922-103426-30a5 | main 451b6e8 | v2 B=4096, default (packed + CUDA graphs) | prover raises `LogUp: fractional sum is not zero (a query is not in its table)` in `logup_packed.prove_ext_table_graphed` |
| r20260922-104144-d89f | main 451b6e8 | v2 B=64 `VERITY_GPU_NO_GRAPHS=1` (packed, eager) | same raise |
| r20260922-104144-d89f | main 451b6e8 | v2 B=64 `VERITY_GPU_NO_PACKED=1` (torch LogUp, other kernels packed) | proof produced, own Python verifier REJECTS: `LogUp level 0: final check` |
| r20260922-104144-d89f | main 451b6e8 | **v1** B=64 default | same raise (`fractional sum is not zero`) |
| r20260922-104918-6221 | **lane/a-gpu2 head 2e1ef3f (pre-merge)** | v1 B=64 default | same raise -- so the a-gpu-v2 merge is NOT the cause |
| r20260922-104923-78ef | main 451b6e8 | v1 B=64 `VERITY_GPU_TORCH_ONLY=1` | **accepted** (`verified: true`, prover 16.6 s, verifier 10.4 s) -- the instance set is fine |

Reading: a-gpu2's Triton/packed LogUp path (and, per the NO_PACKED row, at least one other packed kernel that
feeds the LogUp leaves or transcript) produces wrong values on the A100 for these instances, while the torch
reference path over the same instances verifies. This is the same failure class gate4-h100 saw for the two a-gpu2
heads on `vy-g4` (H100: "fractional sum is not zero" at B=4096) and a-verifier saw for the committed a-gpu kernel path
on `vy-g5` (rejected at LogUp POW level 0), and it is consistent with a-protocol-diff's open item (a-gpu2's proofs
on `vy-g5` pass the candidate's own Python verifier but the independent Rust verifier rejects at LogUp POW level 2).

Consequences for the report: a-gpu2's 1.79-1.93 s/4096 rows (H100, `vy-g5`) are reproducible **only on the pod they
were recorded on**; from the same commit on the A100 the prover does not produce a proof at all. Until the kernel bug
is localised, the A GPU numbers with a verifier-accepted proof on more than one machine are the torch-path ones
(25.5 s a-verifier r20260922-091240-4bf9; 16.6 s / 64 VUs here) and a-gpu-v2's own v2 rows (Python-verified,
A100 only).

Handoff: `vy-a100b` is kept alive as the reproduction machine (venv `/workspace/venv312/bin/python`, sets above).
Owners: a-verifier-2 (F1 triage) / a-protocol-diff. First things to check: Triton kernels compiled for sm_80 vs sm_90
(a-packed2/a-fusion/a-gpu2 developed on 4090/H100 only), int32/Montgomery overflow on values present in these
instances, `graph_ext` region alignment fallback, and whether `VERITY_GPU_NO_PACKED` really disables every kernel that
touches the LogUp leaves.
