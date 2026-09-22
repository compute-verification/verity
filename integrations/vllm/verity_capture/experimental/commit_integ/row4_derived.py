"""Row 4 (VALUE-FIRST §3): the fp32 conversion of the bf16 residual sum inside vLLM's fused add-RMSNorm.

Correspondence [source-inspected; exact model measured 2,340/2,340 rows]: vLLM `d9105ea80` `csrc/layernorm_kernels.cu`, generic
`fused_add_rms_norm_kernel<c10::BFloat16, 0, true>` (the kernel launched under `VLLM_BATCH_INVARIANT=1`; `verity_vllm/registry/b1.py`
`RMSNormFusedCuda` v2): per element j, `z_j = bf16(x_j + res_j)` is written back IN PLACE into `residual` (the naturally
materialised residual output word, manifest class `rmsnormfusedcuda2/out/bf16addf2fp_v1/bf16addf2fp`), then
  occurrence 1 (`bf16tof32_fo1`): `variance += (float)z_j * (float)z_j`   -- thread-private register, lifetime one FMA
  occurrence 2 (`bf16tof32_fo2`): `out_j = bf16(((float)residual_j * s) * w_j)` -- re-read of the stored word, converted again
Both occurrences are `Bf16ToF32(z_j)` = `u16 << 16` (c10 BFloat16 -> float): ONE logical value per element (5,840,640 per B1 C256
request; 97,344 more for the final `rmsnormtriton1` norm whose input is the last layer's residual word), TWO physical occurrences,
neither ever stored to memory.  The fp32 word is never materialised: a hook cannot read it; only regeneration or derivation can bind it.

Two candidates for the same requirement:
  baseline   REGENERATE + COMMIT: in the RMSNorm hook compute `residual.float()` (the identical conversion, on the GPU) and bind it as an
             additional tensor of the step stream (`NativeHostCommitter(regen_row4=True)` -> path `native_host+row4regen`): +4 B/element
             committed (23.4 MB / request for the one logical value; 46.8 MB if both occurrences were bound separately -- they are not:
             one logical identity, one leaf position).
  derived    DERIVED-VALUE OPENING (`out/gen/r8/fa2-commit/derived_value_rule.md` §1): the fp32 word has NO leaf; an opening of it is
             (identity of the derived gate, claimed word w, phi-name, opening of the SOURCE bf16 word under the ORIGINAL root) and the
             verifier accepts iff the source authenticates at the position the verifier derives, phi is the declared `Bf16ToF32`, and
             w == phi(u).  Every consumer (variance FMA chain, normalise path) is lowered to read the source and apply phi, so two
             consumers cannot be handed two different fp32 words: there is no second place a value could come from.

This module implements the derived opening over the `native_host` commitment (packed step stream, 64 KB leaves) and its verifier, plus
the accounting used by `delta_table.py`.  Tests: `tests/test_row4_derived.py` (binding: forged word, wrong source position, other source
class, tampered path, inconsistent uses; value: phi exhaustive over 2^16 on CPU, and on the GPU against torch's `.float()` in
`value_check.py`).
"""
from __future__ import annotations

from dataclasses import dataclass

from verity_capture.experimental.commit_integ.api import Opening, RunCommitment

PHI_NAME = "Bf16ToF32_v1"
SOURCE_CLASS = "rmsnormfusedcuda2/out/bf16addf2fp_v1/bf16addf2fp"
DERIVED_CLASSES = ("rmsnormfusedcuda2/int/bf16tof32_v1/bf16tof32_fo1", "rmsnormfusedcuda2/int/bf16tof32_v1/bf16tof32_fo2",
                   "rmsnormtriton1/int/bf16tof32_v1/bf16tof32_fo1", "rmsnormtriton1/int/bf16tof32_v1/bf16tof32_fo2")


def phi_bf16_to_f32(u16: bytes) -> bytes:
    """c10 `BFloat16 -> float`: the 16-bit word becomes the high half of the fp32 word (little-endian bytes)."""
    if len(u16) != 2:
        raise ValueError("bf16 word")
    return b"\0\0" + bytes(u16)


@dataclass(frozen=True)
class DerivedOpening:
    derived_class: str          # one of DERIVED_CLASSES (the consumer occurrence being opened)
    step: int
    tensor_name: str            # the source tensor (RMSNorm residual output) in the committer's layout
    element: int                # element index in that tensor
    word: bytes                 # claimed fp32 word (4 B)
    phi: str
    source: Opening             # the committer's opening of the leaf holding the source bf16 word
    source_offset: int          # byte offset of the bf16 word inside `source.value`


def source_position(committer, step: int, tensor_name: str, element: int) -> tuple[int, int]:
    """(leaf index, offset inside the leaf) of bf16 element `element` of `tensor_name` in step `step` (packed layout)."""
    for m in committer._layouts[step]:
        if m.name == tensor_name:
            if m.dtype not in ("bfloat16", "torch.bfloat16", "bf16"):
                raise ValueError(f"source is not bf16: {m.dtype}")
            if not 0 <= element < m.nbytes // 2:
                raise IndexError(element)
            b = m.stream_off + 2 * element
            return b // committer.chunk, b % committer.chunk
    raise KeyError(tensor_name)


def open_derived(committer, derived_class: str, step: int, tensor_name: str, element: int) -> DerivedOpening:
    if derived_class not in DERIVED_CLASSES:
        raise ValueError(derived_class)
    leaf, off = source_position(committer, step, tensor_name, element)
    (src,) = committer.open([(step, leaf)])
    u = src.value[off:off + 2]
    return DerivedOpening(derived_class, step, tensor_name, element, phi_bf16_to_f32(u), PHI_NAME, src, off)


def verify_derived(committer, run: RunCommitment, d: DerivedOpening) -> bool:
    """Verifier side.  The position of the SOURCE is derived from the public layout (never taken from the prover); the source
    opening must authenticate under the ORIGINAL run root; phi must be the declared conversion; w == phi(u); width 4."""
    try:
        if d.derived_class not in DERIVED_CLASSES or d.phi != PHI_NAME or len(d.word) != 4:
            return False
        leaf, off = source_position(committer, d.step, d.tensor_name, d.element)
        if d.source.step != d.step or d.source.index != leaf or d.source_offset != off:
            return False
        # the source tensor must be the declared source class: an RMSNorm residual output (tuple index 1) of the pinned model
        metas = {m.name: m for m in committer._layouts[d.step]}
        m = metas[d.tensor_name]
        if committer._module_class(d.tensor_name) not in ("RMSNorm", "GemmaRMSNorm") or not d.tensor_name.endswith("/1"):
            return False
        if m.dtype not in ("bfloat16", "torch.bfloat16", "bf16"):
            return False
        if not committer.verify(run, d.source):
            return False
        return phi_bf16_to_f32(d.source.value[off:off + 2]) == d.word
    except Exception:  # noqa: BLE001 -- a verifier never raises on a malformed opening
        return False


def accounting(run: RunCommitment, committer) -> dict:
    """Bytes of the row-4 requirement covered by the derived rule for this run (from the committed residual words)."""
    n_words = 0
    for step, metas in committer._layouts.items():
        for m in metas:
            if m.name.endswith("/1") and committer._module_class(m.name) in ("RMSNorm", "GemmaRMSNorm") and m.dtype in ("bfloat16", "torch.bfloat16", "bf16"):
                n_words += m.nbytes // 2
    return {"logical_values": n_words, "physical_occurrences": 2 * n_words, "derived_bytes_not_committed": 4 * n_words,
            "source_bytes_committed": 2 * n_words, "opening_bytes": committer.chunk + 32 * 20, "phi": PHI_NAME,
            "note": "one logical fp32 value per residual element; two in-kernel occurrences; no leaf; opening = source leaf + path + phi (a shift)"}
