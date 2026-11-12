"""A model's `spot-card/v1` `workload` card from `benchmarks/one_stage/block_census.py`'s census of its served circuit.

One stratum per proof-block Definition the census cuts, named by the circuit's canonical specialization id: its
blocks per forward-pass token over the ranks, averaged over the census's request (prefill and every decode step), the
serving row families it serves (`kind`), its own output bits, and its MACs, input bits, estimated ANDs (the
estimator's upper bound) and k_log. Rows no cut fits (the census's `gaps`) are in no stratum:
`served_macs_per_token` is the whole circuit's MACs a token, so the figures show them as uncovered.
`boundaries.every_block` is one token's proof-block output bytes. The card is the census and nothing else: no
stratum is edited by hand after it.

    research data fetch art:<census> --to DIR
    python benchmarks/spot_figures/workload_card.py DIR/census.json --census art:<census> --run r<census run> \
        --model GLM-5.3 --params 753329940480 --weights-bytes 7.557e11 --out workload.glm-5.3.json
    python benchmarks/spot_figures/cards.py check workload.glm-5.3.json
    research data put --kind spot-card/v1 --file workload.glm-5.3.json --preserve --ref census=art:<census> --json
    python benchmarks/spot_figures/cards.py fetch art:<card>              # keeps it in cards/ with its art id

`--params` and `--weights-bytes` are the model's total parameters and its served checkpoint's bytes, as the
checkpoint's safetensors metadata and manifest give them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def card(census: dict, *, model: str, source: str, runs: list[str], params: float | None, weights_bytes: float,
         device: str = "H200", owner: str = "@proofs", provisional: bool = True) -> dict:
    tokens = census["tokens"]["prefill"] + census["tokens"]["decode"]
    strata = []
    for block, d in census["definitions"].items():
        n = d["prefill"] + d["decode"]
        if n == 0:
            continue
        strata.append({"block": block, "kind": ", ".join(sorted(d["of"])), "per_token": n / tokens,
                       "macs": d["macs"], "out_bits": d["out_bits"], "in_bits": d["in_bits"], "ands": d["ands"],
                       "k_log": d["k_log"]})
    strata.sort(key=lambda s: (-s["per_token"] * s["macs"], -s["per_token"], s["block"]))
    req = census["per_request"]
    gaps = sorted({g["family"] for g in census["gaps"].values()})
    return {
        "schema": "spot-card/v1", "slot": "workload", "model": model, "status": "measured", "provisional": provisional,
        "source": f"{source}: benchmarks/one_stage/block_census.py's census of {model}'s served circuit, every Program "
                  f"row of its Commit's Build on all {len(census['ranks'])} ranks and every step of one request "
                  f"({census['prompt_len']} prompt tokens, {census['steps']} generated), cut into proof blocks; "
                  "blocks, output bits and MACs counted from the circuit, ANDs the estimator's upper bound; rows no "
                  f"cut fits ({', '.join(gaps)}) are in no stratum and in served_macs_per_token; the weights' bytes "
                  "and parameters as the checkpoint's manifest and safetensors metadata give them",
        "runs": runs, "owner": owner,
        "value": {"ranks": len(census["ranks"]), "device": device,
                  "request": {"prefill": census["prompt_len"], "decode": census["steps"]}, "tokens": tokens,
                  "weights_bytes": weights_bytes, "params": params, "strata": strata,
                  "served_macs_per_token": req["circuit_macs"] / tokens,
                  "boundaries": {"every_block": sum(s["per_token"] * s["out_bits"] for s in strata) / 8}},
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("census", type=Path, help="the census's census.json")
    ap.add_argument("--census", dest="art", required=True, help="the census's art: id, the card's source")
    ap.add_argument("--run", action="append", default=[], help="the census's run id (repeatable)")
    ap.add_argument("--model", required=True, choices=("GLM-5.3", "DeepSeek-V4-Pro"))
    ap.add_argument("--params", type=float, help="the model's total parameters")
    ap.add_argument("--weights-bytes", type=float, required=True, help="the served checkpoint's bytes")
    ap.add_argument("--device", default="H200")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    c = card(json.loads(a.census.read_text()), model=a.model, source=a.art, runs=a.run,
             params=int(a.params) if a.params is not None else None, weights_bytes=a.weights_bytes, device=a.device)
    a.out.write_text(json.dumps(c, indent=1) + "\n")
    print(a.out)


if __name__ == "__main__":
    main()
