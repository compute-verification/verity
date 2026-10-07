# SP1 backend: where the approved guest and the IR disagree

Statuses follow `verity/ml/circuits/silicon/discrepancy_log.json` (`open` / `resolved` / `excluded`). That log is scoped to tensor-core
targets and instructions, so entries about the guest's other gate families live here until the log takes non-tensor-core
entries.

## S1. Two NaN operands: the guest's `f32_add` / `f32_mul` return the first, the IR primitives the second

- **Guest:** `veritor_zk_common::reduce::{f32_add, f32_mul}` (`reduce.rs`), mirrored by `verity_sp1.proofs.reduce_mufu` and
  inherited by `archive/sp1/common/src/ftz.rs`. It pins the x86-SSE rule: a NaN operand is returned quieted, the
  **first** operand first. An invalid operation (`inf - inf`, `0 * inf`) returns `0xFFC0_0000`.
- **IR:** the registered `F32Add` / `F32Mul` evaluators call numpy. On the x86 hosts where the registry's evidence was
  measured, two NaN operands return the **second**, quieted.
- **Observable?** Not by any current template. In RMSNorm (fused and Triton), RoPE and GemmCoordinate, every NaN path ends
  in `rsqrt_approx`, `f2fp_bf16` or bf16 canonicalization, which erase the payload. Found by lane `sp1-evaluator`
  (handoff `20260926T0250Z`, runs r20260926-020127-276d … r20260926-024138-fa0c, PR #52).
- **Status:** open. Owner: the coordinator.
- **What resolves it:** before any template whose output can carry an f32 NaN payload from a two-NaN add or mul is proved
  through this path, pin one rule on both sides and add a vector with two distinct NaN payloads. Either the IR evaluator
  returns the first operand (numpy with an explicit select), or a new guest family returns the second (new ELF and vk,
  approval).
