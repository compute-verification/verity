"""One record for every property, and the citation a run carries.

A property harness writes one document under its own `schema`: the evidence, `ok` (the property held over that evidence)
and `digest`, the sha256 of the canonical JSON of every other field.  `seal` adds the last two.  `REGISTRY` names every
property by the schema of its document; a run cites a record by `(name, digest)`, and a citation is trusted only after
`PropertyRecord.problems` recomputes the digest.  Records a run cites live in `<run>/properties/*.json`.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from verity_vllm.correspondence.runtime import canonical_json

DIR = "properties"


@dataclass(frozen=True)
class Property:
    name: str
    schema: str
    claim: str


REGISTRY: dict[str, Property] = {p.name: p for p in (
    Property("fa_tap_exactness", "verity-vllm/fa-tap-exactness/v1",
             "The FA tap build computes exactly the installed FA kernel's output, writes exactly the declared stream words, "
             "and skips a launch whose declared tile is not the kernel's."),
    Property("fa2_target_capture", "verity-vllm/fa2-target-capture/v1",
             "On the capture's GPU, vLLM selects FA2, the installed kernel launches kBlockM 64 and kBlockN fa2_kblock_n(D), MUFU ex2 / rcp "
             "are core's tables on every input, and every checked output row equals the Hopper-step Attention_v2 wherever its operands "
             "are finite and FA2's own Check_inf placement, the registered Attention_v5, on every row, which the target binds."),
    Property("fa2_softcap_capture", "verity-vllm/fa2-softcap-capture/v1",
             "On the capture's GPU, the installed FA2 softcap kernel launches kBlockM 64 and kBlockN fa2_kblock_n(D), MUFU ex2 / rcp are "
             "core's tables and MUFU.TANH is MufuTanh_v1 on every input, and every checked output row equals the registered "
             "AttentionSoftcap_v2, which the target binds, on every head."),
    Property("glm53_block_fp8_capture", "verity-vllm/glm53-block-fp8-capture/v2",
             "On the capture's Hopper GPU, vLLM selects CutlassFp8BlockScaledMMKernel for GLM-5.3's FP8 linears and Triton's "
             "fused_moe_kernel (TritonExperts) for its experts, every compiled expert kernel runs its dot as the step fp8_moe gives its "
             "tile (k32 e4m3 wgmma, or f16 mma.sync m16n8k16 on widened e4m3), and at GLM-5.3's shapes every output word of every launch "
             "equals ScaledMmFp8Block_v1 or MoeExpertFp8BlockGemm(W)_v3 at the launch's statics with that step as DOT, and every "
             "activation quant equals Fp8GroupQuant_v1, bit for bit."),
    Property("glm53_dsa_capture", "verity-vllm/glm53-dsa-capture/v2",
             "On the capture's Hopper GPU, vLLM selects FLASH_ATTN_MLA_SPARSE and the bf16 _fused_q_kernel for GLM-5.3's sparse MLA, "
             "and at GLM-5.3's TP8 statics every indexer-cache slot equals DsaIndexerKRow_v1, every index query and weight "
             "DsaIndexerQ_v1, every compared prefill and paged logit DsaMqaLogit_v3, every top-k row of at most 2048 keys "
             "DsaTopKShortRow_v1, and every compared FA3 sparse output SparseMlaHead_v1 at its row's valid count, bit for bit; "
             "the k_pe and q_pe RoPE words are each one named contraction."),
    Property("blackwell_fp8_capture", "verity-vllm/blackwell-fp8-capture/v1",
             "On the capture's cc 12.0 GPU, vLLM selects CutlassFp8BlockScaledMMKernel (and TritonFp8BlockScaledMMKernel under "
             "--linear-backend triton) for the 235B FP8 TP8 row's linears and TritonExperts at block [64, 64] for its experts, the "
             "three M classes run three CUTLASS kernels, and at those shapes every output word equals the k32 BlackwellE4m3QmmaDot32 "
             "block model of its kernel (CUTLASS's FFMA(d, FMUL(sx, sw), acc), Triton's FFMA(FMUL(d, sx), sw, acc)) and every "
             "activation quant Fp8GroupQuant_v1 at G 64 or 128, bit for bit, while each competing model is refuted."),
    Property("norm_tap_exactness", "verity-vllm/norm-tap-exactness/v1",
             "The tapped norm kernels (fused CUDA op, Triton copy) compute exactly the installed kernels' outputs, write every row's "
             "scale, and each scale is the IR model's scale bit for bit; the Triton copy's compiled arithmetic is the installed kernel's."),
    Property("router_tap_exactness", "verity-vllm/router-tap-exactness/v1",
             "The tapped topk_softmax computes exactly the installed op's outputs, writes every word of every row's softmax, and each word "
             "is the ordered router's committed value in the IR bit for bit (ties, NaN, +-inf and +-0 included), in vLLM's own MoE layers too."),
    Property("sigmoid_router_capture", "verity-vllm/sigmoid-router-capture/v1",
             "On the capture's GPU, MUFU ex2 / rcp are core's tables on every input, and vLLM's fused sigmoid noaux_tc router "
             "(_moe_C.grouped_topk, 256 experts, top-8, renormalized, x2.5) returns through its block and its warp kernel exactly the "
             "ids and weights of MoeRouterSigmoidBias_v1 for that kernel on every captured row and every logit whose sigmoid reads MUFU."),
    Property("dsv4_moe_capture", "verity-vllm/dsv4-moe-capture/v1",
             "On the capture's GPU, MUFU lg2 is MufuLg2Mantissa_v1 on every 1.m, MUFU ex2 / rcp / sqrt are core's tables, and "
             "the Triton router dsv4_topk (384 experts, top-6, renormalized, x2.5), the hash router "
             "dsv4HashTopkSoftplusSqrt over tid2eid and the clamped SwiGLU (silu_and_mul_with_clamp, limit 10, vectorised at "
             "d = 384 and scalar at d = 9) return exactly MoeRouterSqrtSoftplus_v1, MoeRouterHash_v1 and SiluMulClamp_v1 on "
             "every captured row."),
    Property("vocab_range_exactness", "verity-vllm/vocab-range-exactness/v1",
             "The vocabulary-range values a TP rank commits equal the IR's EmbeddingShard_v1 values on every token, and the fused "
             "embedding kernel's output agrees with them (zero rows exactly where the rank does not own the token, else the shard row)."),
    Property("noninterference", "verity-vllm/noninterference/v2",
             "At the record's tensor-parallel world, the observed engine generated the bare engine's tokens for every request (and, "
             "where boundary hashes were taken, every compared boundary value is bit-identical)."),
    Property("census", "verity-capture/census-reconciliation/v1",
             "Every device kernel the bare engine launched is claimed by a kernel the Match resolution declares or by the frozen "
             "kernel allowlist, at every step."),
    Property("golden", "verity-gen/golden-check/v1",
             "Every golden log still resolves to its recorded attribution under the current patterns."),
    Property("holdout", "verity-gen/holdouts/v1",
             "Every holdout run passed each of its invariants."),
    Property("difftest", "verity-gen/difftest/v1",
             "The quarantined Definition evaluates every produced case to the adapter's bytes."),
    Property("quarantine_lint", "verity-gen/quarantine-lint/v1",
             "Every quarantined Definition module keeps to the quarantine rules, with no duplicate names."),
    Property("protected", "verity-gen/provenance/v1",
             "The candidate diff touches no protected path."),
)}
BY_SCHEMA = {p.schema: p for p in REGISTRY.values()}


def digest_of(doc: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json({k: v for k, v in doc.items() if k != "digest"})).hexdigest()


def seal(doc: Mapping[str, Any], ok: bool) -> dict[str, Any]:
    """`doc` with `ok` and its `digest`; `doc["schema"]` must be a registered property's."""
    if doc.get("schema") not in BY_SCHEMA:
        raise ValueError(f"schema {doc.get('schema')!r} is not a registered property's")
    body = {k: v for k, v in doc.items() if k != "digest"}
    body["ok"] = bool(ok)
    return {**body, "digest": digest_of(body)}


@dataclass(frozen=True)
class PropertyRecord:
    doc: Mapping[str, Any]
    source: str | None = None

    @property
    def name(self) -> str | None:
        p = BY_SCHEMA.get(self.doc.get("schema"))
        return p.name if p else None

    @property
    def ok(self) -> bool:
        return self.doc.get("ok") is True

    @property
    def digest(self) -> str | None:
        return self.doc.get("digest")

    def problems(self) -> list[str]:
        out = [] if self.name else [f"schema {self.doc.get('schema')!r} is not a registered property's"]
        if not isinstance(self.doc.get("ok"), bool):
            out.append("no boolean ok")
        if self.digest != digest_of(self.doc):
            out.append("digest does not match the record")
        return out

    def citation(self) -> dict[str, Any]:
        return {"name": self.name, "schema": self.doc.get("schema"), "digest": self.digest, "ok": self.ok, "source": self.source,
                "problems": self.problems()}

    def write(self, run_dir: str | Path) -> Path:
        p = Path(run_dir) / DIR / f"{self.name}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.doc, indent=1) + "\n")
        return p

    @classmethod
    def load(cls, path: str | Path, root: str | Path | None = None) -> "PropertyRecord":
        """An unreadable file loads as an empty document, so its citation names the problems instead of dropping the file."""
        p = Path(path)
        try:
            doc = json.loads(p.read_text())
        except (OSError, ValueError):
            doc = {}
        return cls(doc if isinstance(doc, dict) else {}, str(p.relative_to(root)) if root else str(p))


def records_of(run_dir: str | Path) -> list[PropertyRecord]:
    """The property records a run cites: every `<run_dir>/properties/*.json`, by file name."""
    d = Path(run_dir)
    return [PropertyRecord.load(p, d) for p in sorted((d / DIR).glob("*.json"))]


def citations(run_dir: str | Path) -> tuple[list[dict[str, Any]], list[str]]:
    """The citation of every record of the run, and a note for each record that does not hold or whose digest does not recompute."""
    cites = [r.citation() for r in records_of(run_dir)]
    notes = [f"property {c['name'] or c['schema']} ({c['source']}): {'; '.join(c['problems']) or 'does not hold'}"
             for c in cites if c["problems"] or not c["ok"]]
    return cites, notes
