"""GLM-5.3's `workload` card (`cards/workload.glm-5.3.json`) from the census of its request Program
(`benchmarks/one_stage/a11_kinds.py`'s `kinds.json`, `a10_inputs.json`'s `kinds`): the root Calls of one served request
on rank 0 of the TP8 B1 row, by Definition kind.

Each kind becomes one stratum of proof blocks whose outputs are at most 16 bits where the circuit allows it, per
forward-pass token over the 8 ranks (the ranks run the same Program on their shards):

* a matmul kind is its output coordinates (the chain's coordinate Definition, `K` MACs each, its output words wide); a
  kind with several specs scales the probed spec's coordinates by the kind's gates over that spec's (the census records
  only the probed spec's chain);
* every other kind is its output words, one proof block each, its MACs not counted.

The forward-pass tokens are the embedding's Calls.  A model of the partition, not the partition: attention's growth with
context past the row's is not in it.

    research data fetch art:<census> --to DIR
    uv run python benchmarks/spot_figures/glm53_partition.py DIR/kinds.json --census art:<census>
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from cards import SCHEMA, write

RANKS = 8
DEVICE = "H200"
#: the census row's request: 256 prompt tokens, 32 generated
REQUEST = {"prefill": 256, "decode": 32}
#: GLM-5.3's block-FP8 checkpoint at its pinned revision, as `adapters/vllm/manifests/checkpoints.json` (role
#: GLM53) records its size
WEIGHTS_BYTES = 7.557e11
#: its parameters at that revision, the hub's safetensors metadata (its F8_E4M3, BF16 and F32 tensors' elements)
PARAMS = 753_329_940_480
WORD_BITS = 16
#: the chain's matmul coordinate Definitions, whose `K` is a proof block's MACs
COORDINATES = ("MoeExpertFp8Coordinate_v1", "MoeExpertFp8CoordinateW_v1", "ScaledMmFp8BlockCoordinateGivenScale_v1",
               "LlBf16DotprodDot_v1", "GemmCoordinate_v2", "DotBf16_v2")
#: a coordinate's output words where it isn't one bf16 word: split-K's f32 partial sums
COORDINATE_WORDS = {"DotBf16_v2": 2}
#: every other kind's output words a Call, from its spec's parameters (the census's `out_words` where it records them)
WORDS = {
    "AllReduceMultimem_v1": 1572864,
    "MoeSum_v1": 6144,
    "RMSNormFusedCudaVector8_v1": 6144,
    "Fp8GroupQuantUe8m0_v1": 128 + 4,
    "AllReduceRmsNormMnnvl_v1": 6144,
    "MoeRouterSigmoidBias_v1": 2 * 8,
    "DsaRMSNorm_v1": 2048,
    "DsaMqaLogits_v3": 2,
    "DsaIndexerQ_v1": 32 * (64 + 2),
    "ResidualAdd_v1": 6144,
    "SiluMul_v2": 256,
    "DsaTopKShortRow_v1": 4096,
    "TokenSelect_v1": 2,
    "RoPEHeadGptJ_v2": 64,
    "RMSNormCudaVector8_v1": 6144,
    "DsaIndexerKRow_v1": 64 + 2,
    "AllReduceMnnvl_v1": 6144,
    "EmbeddingShard_v1": 6144,
    "SparseMlaSplit_v1": 8 * 512,
}
TOKENS_KIND = "EmbeddingShard_v1"


def _k(definition: str) -> int:
    m = re.search(r"[{,]K=(\d+)", definition)
    if m is None:
        raise ValueError(f"no K in {definition!r}")
    return int(m.group(1))


def _coordinate(entry: dict) -> tuple[str, int] | None:
    """The kind's matmul coordinate in its chain and how many a Call holds, or None for a kind with none."""
    for link in entry.get("path") or entry.get("fold_chain") or []:
        if link["id"].split("{")[0] in COORDINATES:
            return link["id"], int(link["per_call"])
    return None


def _scale(entry: dict) -> float:
    """The kind's gates over its probed spec's, its Calls all taken at that spec's gates."""
    chain = entry.get("path") or entry.get("fold_chain") or []
    call = entry.get("call_gates") or (chain[0]["gates"] if chain else 0)
    return entry["gates"] / (entry["calls"] * call) if call else 1.0


def strata(census: dict) -> tuple[int, list[dict]]:
    """The forward-pass tokens and every kind that does work as a stratum of proof blocks per token over the ranks."""
    kinds = {name: e for name, e in census["kinds"].items() if e.get("gates", 0) > 0 and e.get("calls", 0) > 0}
    tokens = kinds[TOKENS_KIND]["calls"]
    out = []
    for name, e in sorted(kinds.items(), key=lambda kv: (-kv[1]["gates"], kv[0])):
        coord = _coordinate(e)
        if coord is not None:
            unit, per_call = coord
            scale = _scale(e)
            words = COORDINATE_WORDS.get(unit.split("{")[0], 1)
            macs = _k(unit)
        elif name in WORDS:
            unit, per_call, scale, words, macs = name, WORDS[name], 1.0, 1, 0
        elif name.startswith("Const32["):
            continue
        else:
            raise KeyError(f"census kind {name!r} is neither a matmul with a coordinate nor in WORDS: "
                           "add its output words")
        out.append({"block": unit, "kind": name, "per_token": RANKS * e["calls"] * per_call * scale / tokens,
                    "macs": macs, "out_bits": WORD_BITS * words})
    return tokens, out


def card(census: dict, source: str) -> dict:
    tokens, rows = strata(census)
    return {"schema": SCHEMA, "slot": "workload", "model": "GLM-5.3", "status": "modeled", "provisional": True,
            "source": f"{source}: benchmarks/one_stage/a11_kinds.py's census of GLM-5.3's request Program, rank 0 "
                      f"of the TP8 B1 row ({census.get('descriptor', '')}), by `glm53_partition.py`; the weights' "
                      "bytes from adapters/vllm/manifests/checkpoints.json, role GLM53, and its parameters from "
                      "the hub's safetensors metadata at that revision",
            "runs": [], "owner": "@proofs",
            "value": {"ranks": RANKS, "device": DEVICE, "request": REQUEST, "tokens": tokens,
                      "weights_bytes": WEIGHTS_BYTES, "params": PARAMS, "strata": rows,
                      "boundaries": {"every_block": sum(r["per_token"] * r["out_bits"] for r in rows) / 8}}}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("kinds", type=Path, help="the census's kinds.json")
    ap.add_argument("--census", required=True, help="the census's art: id, the card's source")
    args = ap.parse_args(argv)
    print(write(card(json.loads(args.kinds.read_text()), args.census), force=True))


if __name__ == "__main__":
    main()
