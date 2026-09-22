"""Workload volume calculator for the alternatives note (deliverable 2).  Pure arithmetic, no measurement.

Every number is derived from the stated workload shape and the encoding of `encoding.py` (|row| = 2T + (D+6)·nb(T) fp32
words; P16 derived) — see `alternatives.md` for the sources of the shape constants.

    PYTHONPATH=. python -m verity_capture.experimental.fa2_commit.workload --out out/gen/r8/fa2-commit
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .encoding import Geometry, RowLayout

# ---- constants with their sources (all [S] unless marked) -----------------------------------------------------------
LAYERS, HEADS, KV_HEADS, D_HEAD, HIDDEN = 30, 9, 3, 64, 576            # SmolLM2-135M (census §1, BRIEF pins)
BN, BM = 128, 64                                                     # FA2 kBlockN / kBlockM at D=64 (b1.py:364-370, B-sem STATUS)
NORMS_PER_LAYER = 2                                                  # two fused_add_rms_norm per decoder layer
WORD = 4                                                             # every committed stage is fp32 in this encoding
KV_BYTES_PER_TOKEN = LAYERS * KV_HEADS * D_HEAD * 2 * 2              # k and v, bf16 -> 23,040 B
# hash-cost coefficients
SHA_BLOCK = 64
LEAF_FRAME = 26                                                      # pos_leaf framing: tag(1)+pos(8)+len(2)+...; measured in costs.py as 26+4w -> blocks
NODE_BLOCKS = 2                                                      # node = H(NODE_TAG || l || r): 65 B -> 2 blocks
GPU_SHA_PER_S_A100 = 9.56e9                                          # cost-model.md:73 (hashcat, A100-SXM4)
GPU_SHA_PER_S_H100 = 1.62e10
FLOPEQ_PER_SHA_A100 = 3.3e4                                          # cost-model.md:73
CPU_SHA_GBPS = 1.57                                                  # commitment-cost-study §3 (streaming, SHA-NI)
CPU_TREE_GBPS = 0.62                                                 # same study, packed tree effective
PCIE_D2H_GBPS = 20.0                                                 # [X] assumed PCIe-4 x16 sustained
MODEL_PARAMS = 135e6                                                 # SmolLM2-135M non-embedding+embedding, order of magnitude
DECODE_STEP_S = 0.011                                                # commitment-cost-study §8 forward_s per step-RU (H2) [M there]
REQUEST_S = (0.18, 0.87)                                             # same study: prefill + <=15 decode steps


def blocks_for(nbytes: int) -> int:
    """SHA-256 compression count for a message of nbytes (Merkle–Damgård padding)."""
    return (nbytes + 9 + SHA_BLOCK - 1) // SHA_BLOCK


@dataclass
class Shape:
    name: str
    prompt: int
    generated: int

    @property
    def rows(self) -> list[int]:
        """Visible-key count T of every (row) = prefill rows T=1..P then decode rows T=P+1..P+G."""
        return list(range(1, self.prompt + 1)) + list(range(self.prompt + 1, self.prompt + self.generated + 1))

    @property
    def tokens(self) -> int:
        return self.prompt + self.generated


def per_layer_head(sh: Shape) -> dict:
    geo = Geometry(D=D_HEAD, BN=BN, QB=BM)
    pre = list(range(1, sh.prompt + 1))
    dec = list(range(sh.prompt + 1, sh.prompt + sh.generated + 1))
    pairs_pre, pairs_dec = sum(pre), sum(dec)
    nb_pre = sum(math.ceil(T / BN) for T in pre)
    nb_dec = sum(math.ceil(T / BN) for T in dec)
    words_pre = sum(RowLayout(geo, 0, 0, T - 1).count for T in pre)
    words_dec = sum(RowLayout(geo, 0, 0, T - 1).count for T in dec)
    # physical (masked-inclusive) pairs of REAL rows: the kernel computes S for every column of every visited block.
    # prefill: rows of q-block m visit n_block_max(m) = min(ceil(Lk/BN), ceil((m+1)*BM/BN)) blocks (bottom-right causal, Lq == Lk)
    masked_pre = 0
    for T in pre:
        m = (T - 1) // BM
        nbm = min(math.ceil(sh.prompt / BN), math.ceil((m + 1) * BM / BN))
        masked_pre += nbm * BN - T
    masked_dec = sum(math.ceil(T / BN) * BN - T for T in dec)
    # padded ROWS (decode: seqlenq_ngroups_swapped -> 3 real rows in a 64-row m-block per kv head; prefill: last q-block)
    padded_rows_pre = (math.ceil(sh.prompt / BM) * BM - sh.prompt)
    padded_rows_dec = sh.generated * (BM - HEADS // KV_HEADS) / (HEADS // KV_HEADS)  # per head-equivalent
    return {
        "pairs_prefill": pairs_pre, "pairs_decode": pairs_dec, "pairs": pairs_pre + pairs_dec,
        "blocks_prefill": nb_pre, "blocks_decode": nb_dec,
        "words_prefill": words_pre, "words_decode": words_dec, "words": words_pre + words_dec,
        "pair_words": 2 * (pairs_pre + pairs_dec),                       # S + P32
        "block_state_words": (D_HEAD + 6) * (nb_pre + nb_dec),           # M, ALPHA, L x4, O x D (+INV - first ALPHA = 0)
        "p16_bytes_derived": 2 * (pairs_pre + pairs_dec),
        "masked_pairs_prefill": masked_pre, "masked_pairs_decode": masked_dec,
        "padded_rows_prefill": padded_rows_pre, "padded_rows_decode_head_equiv": padded_rows_dec,
        "attention_flops": 4 * D_HEAD * (pairs_pre + pairs_dec),           # QK^T and PV, 2 flops per MAC each
    }


def hash_blocks(words: int, leaf_words: int, packed: bool) -> dict:
    leaves = math.ceil(words / leaf_words)
    if packed:
        # chunk leaf = H(CHUNK_TAG || pos || bytes): one long message, then a small tree over chunk digests
        leaf_blocks = leaves * blocks_for(LEAF_FRAME + WORD * leaf_words)
    else:
        leaf_blocks = leaves * blocks_for(LEAF_FRAME + WORD * leaf_words)
    node_blocks = max(leaves - 1, 0) * NODE_BLOCKS
    return {"leaves": leaves, "sha_blocks": leaf_blocks + node_blocks}


def request_volumes(sh: Shape) -> dict:
    plh = per_layer_head(sh)
    lh = LAYERS * HEADS
    words = plh["words"] * lh
    out = {
        "shape": asdict(sh), "per_layer_head": plh,
        "committed_words": words, "committed_bytes": words * WORD,
        "pair_bytes": plh["pair_words"] * lh * WORD,
        "block_state_bytes": plh["block_state_words"] * lh * WORD,
        "p16_derived_bytes": plh["p16_bytes_derived"] * lh,
        "explicit_materialization_bytes": words * WORD + plh["p16_bytes_derived"] * lh,
        "census_style_intrinsic_bytes": round(plh["pairs"] * lh * 10.1),
        "masked_pairs": (plh["masked_pairs_prefill"] + plh["masked_pairs_decode"]) * lh,
        "attention_flops": plh["attention_flops"] * lh,
        "forward_flops_est": 2 * MODEL_PARAMS * sh.tokens,
        "kv_boundary_bytes": KV_BYTES_PER_TOKEN * sh.tokens,
        "norm_conversion_bytes": LAYERS * NORMS_PER_LAYER * sh.tokens * HIDDEN * WORD,
        "norm_conversion_values": LAYERS * NORMS_PER_LAYER * sh.tokens * HIDDEN,
        "hash": {},
    }
    for lw in (1, 4, 16, 64):
        hb = hash_blocks(words, lw, packed=False)
        out["hash"][f"hier_leaf{lw}"] = {
            **hb,
            "gpu_s_A100": hb["sha_blocks"] / GPU_SHA_PER_S_A100,
            "gpu_s_H100": hb["sha_blocks"] / GPU_SHA_PER_S_H100,
            "flopeq_A100": hb["sha_blocks"] * FLOPEQ_PER_SHA_A100,
            "ratio_to_attention_flops": hb["sha_blocks"] * FLOPEQ_PER_SHA_A100 / out["attention_flops"],
            "ratio_to_forward_flops": hb["sha_blocks"] * FLOPEQ_PER_SHA_A100 / out["forward_flops_est"],
        }
    for cw in (64, 256, 1024, 4096):
        hb = hash_blocks(words, cw, packed=True)
        out["hash"][f"packed_chunk{cw}"] = {
            **hb,
            "gpu_s_A100": hb["sha_blocks"] / GPU_SHA_PER_S_A100,
            "flopeq_A100": hb["sha_blocks"] * FLOPEQ_PER_SHA_A100,
            "ratio_to_attention_flops": hb["sha_blocks"] * FLOPEQ_PER_SHA_A100 / out["attention_flops"],
            "ratio_to_forward_flops": hb["sha_blocks"] * FLOPEQ_PER_SHA_A100 / out["forward_flops_est"],
            "retained_bytes_chunk_digests": hb["leaves"] * 32,
        }
    out["explicit"] = {
        "d2h_s": out["explicit_materialization_bytes"] / (PCIE_D2H_GBPS * 1e9),
        "cpu_sha_stream_s": out["explicit_materialization_bytes"] / (CPU_SHA_GBPS * 1e9),
        "cpu_tree_s": out["explicit_materialization_bytes"] / (CPU_TREE_GBPS * 1e9),
    }
    return out


def fmt_b(n: float) -> str:
    for unit, div in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n >= div:
            return f"{n / div:.3g} {unit}"
    return f"{n:.0f} B"


def render_md(all_: dict) -> str:
    L = [f"# fa2-commit workload volumes — {all_['utc']} (calculated; `workload.py`)", ""]
    a = all_["accepted"]
    p = a["per_layer_head"]
    L += ["## Accepted run (155 + 14, 15 steps, 30 layers × 9 heads), per (layer, head)", "",
          "| term | prefill | decode | total |", "|---|---:|---:|---:|",
          f"| causal (q,k) pairs | {p['pairs_prefill']:,} | {p['pairs_decode']:,} | {p['pairs']:,} |",
          f"| visited 128-key blocks | {p['blocks_prefill']:,} | {p['blocks_decode']:,} | {p['blocks_prefill'] + p['blocks_decode']:,} |",
          f"| committed fp32 words (S,P32 + M,ALPHA,L,O,INV) | {p['words_prefill']:,} | {p['words_decode']:,} | {p['words']:,} |",
          f"| masked (computed, not committed) pairs in real rows | {p['masked_pairs_prefill']:,} | {p['masked_pairs_decode']:,} | {p['masked_pairs_prefill'] + p['masked_pairs_decode']:,} |",
          f"| padded rows (kernel rows with no Program value) | {p['padded_rows_prefill']} | {p['padded_rows_decode_head_equiv']:.0f} (head-equiv.) | |",
          ""]
    L += ["## Per request, by shape (B1; B8 = ×8 identical independent requests)", "",
          "| shape | assumed (P, G) | pairs / (layer,head) | in-kernel committed (fp32 stream) | pair terms (S,P32) | block state | P16 derived (not committed) | explicit materialization (stream + P16) | census-style 10.1 B/pair | KV step boundary | norm Bf16ToF32 values (2/layer) |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, r in all_["shapes"].items():
        s = r["shape"]
        L.append(f"| {key} | ({s['prompt']}, {s['generated']}) | {r['per_layer_head']['pairs']:,} | {fmt_b(r['committed_bytes'])} | {fmt_b(r['pair_bytes'])} | {fmt_b(r['block_state_bytes'])} | {fmt_b(r['p16_derived_bytes'])} | {fmt_b(r['explicit_materialization_bytes'])} | {fmt_b(r['census_style_intrinsic_bytes'])} | {fmt_b(r['kv_boundary_bytes'])} | {fmt_b(r['norm_conversion_bytes'])} |")
    L += ["", "## In-kernel hash work per request (accepted run), by layout", "",
          "| layout | leaf / chunk | leaves | SHA-256 blocks | A100 hash time at 9.56e9/s | flop-eq (3.3e4/compression) ÷ attention flops | ÷ forward flops (2·135M·169) |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for k, h in a["hash"].items():
        L.append(f"| {k.split('_')[0]} | {k.split('_')[1]} | {h['leaves']:,} | {h['sha_blocks']:,} | {h['gpu_s_A100'] * 1e3:.2f} ms | {h['ratio_to_attention_flops']:.1f}× | {h['ratio_to_forward_flops']:.2f}× |")
    L += ["", f"Attention flops per request {a['attention_flops']:.3g}; forward estimate {a['forward_flops_est']:.3g}; masked pairs {a['masked_pairs']:,} "
          f"(= {a['masked_pairs'] / (a['per_layer_head']['pairs'] * LAYERS * HEADS):.0%} of the logical pairs, computed by the kernel and NOT committed).",
          f"Explicit materialization of {fmt_b(a['explicit_materialization_bytes'])}: D2H {a['explicit']['d2h_s'] * 1e3:.1f} ms at {PCIE_D2H_GBPS:g} GB/s [X]; "
          f"CPU SHA-256 stream {a['explicit']['cpu_sha_stream_s'] * 1e3:.0f} ms at {CPU_SHA_GBPS} GB/s; C1 packed tree {a['explicit']['cpu_tree_s'] * 1e3:.0f} ms at {CPU_TREE_GBPS} GB/s.", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    shapes = {
        "C256 (accepted)": Shape("C256-accepted", 155, 14),
        "C1024 A (P=C-14)": Shape("C1024-A", 1010, 14),
        "C1024 B (P=G=C/2)": Shape("C1024-B", 512, 512),
        "C4096 A (P=C-14)": Shape("C4096-A", 4082, 14),
        "C4096 B (P=G=C/2)": Shape("C4096-B", 2048, 2048),
    }
    all_ = {"schema": "fa2-commit/workload/v1", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "host": platform.platform(), "constants": {
                "LAYERS": LAYERS, "HEADS": HEADS, "KV_HEADS": KV_HEADS, "D": D_HEAD, "BN": BN, "BM": BM, "HIDDEN": HIDDEN,
                "GPU_SHA_PER_S_A100": GPU_SHA_PER_S_A100, "FLOPEQ_PER_SHA_A100": FLOPEQ_PER_SHA_A100,
                "CPU_SHA_GBPS": CPU_SHA_GBPS, "CPU_TREE_GBPS": CPU_TREE_GBPS, "PCIE_D2H_GBPS": PCIE_D2H_GBPS},
            "accepted": request_volumes(shapes["C256 (accepted)"]),
            "shapes": {k: request_volumes(v) for k, v in shapes.items()}}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "workload.json").write_text(json.dumps(all_, indent=1))
    (args.out / "workload.md").write_text(render_md(all_))
    print(render_md(all_))


if __name__ == "__main__":
    main()
