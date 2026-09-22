#!/usr/bin/env python3
"""[REPLAY, additive] Reconstruct TA1's B0 workload.json (GUIDANCE §32.9A).

TA1's original ``/workspace/ta1-store/B0/workload.json`` lives on PB's pod (see
``docs/vllm-poc/pb3-brief.md``), not in the repo. The workload is fully determined
by the committed record: the prompt token ids and sampling parameters are in
``fixtures/results/TA1-typed-b1-20260907/B0/ru/dc159809-.../boundary.json`` and the
request id is ``uuid5(NAMESPACE, "<model>|<prompt ids csv>|<max_tokens>")``
(``verity_vllm_sampler/workload.py``). This script re-emits the workload file and
asserts the derived request id equals the recorded one, so a production rerun binds
to the same request identity.

``prompt_text`` is NOT reconstructed (execution and ids use token ids only; the
text field is cosmetic). ``text_roundtrip`` is set false accordingly.

Usage:  python replay_b0_workload.py --out workload.json
"""

from __future__ import annotations

import argparse
import json
import uuid

SCHEMA = "verity-vllm/workload/v1"
NAMESPACE = uuid.UUID("6f1e2c7a-1d3b-4f7e-9c3a-5b2d8e4a1c9f")
MODEL = "HuggingFaceTB/SmolLM2-135M"
TOKENIZER_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"

# From results/TA1-typed-b1-20260907/B0/ru/dc159809-.../boundary.json (committed record).
PROMPT_TOKEN_IDS = [
    70, 462, 3088, 39567, 260, 8758, 3964, 338, 501, 7459, 735, 2072, 281, 1759,
    29, 3510, 8820, 30, 330, 11087, 3179, 32386, 260, 6219, 11805, 335, 897,
    7107, 30, 20401, 5053, 38208,
]
MAX_TOKENS = 8
RECORDED_REQUEST_ID = "dc159809-b295-51b0-9148-a405bc703cc1"

# The recorded outputs of the original A100 pass (consistency-check targets, same record).
RECORDED_OUTPUT_TOKEN_IDS = [359, 253, 1123, 1183, 282, 253, 1772, 338]
RECORDED_CHOSEN_LOGPROBS = [
    -2.3287830352783203, -3.135054349899292, -3.046064853668213,
    -1.4307031631469727, -0.34137848019599915, -1.4645650386810303,
    -3.579011917114258, -0.6411636471748352,
]


def request_id() -> str:
    body = ",".join(str(t) for t in PROMPT_TOKEN_IDS)
    return str(uuid.uuid5(NAMESPACE, f"{MODEL}|{body}|{MAX_TOKENS}"))


def workload_doc() -> dict:
    rid = request_id()
    assert rid == RECORDED_REQUEST_ID, f"derived {rid} != recorded {RECORDED_REQUEST_ID}"
    return {
        "schema": SCHEMA,
        "model": MODEL,
        "tokenizer_revision": TOKENIZER_REVISION,
        "seed": 0,
        "prompt_lengths": [32, 128, 512, 2048],
        "max_tokens": [16, 64],
        "variants": {
            "greedy": {"temperature": 0.0, "top_p": 1.0, "top_k": 0},
            "temp": {"temperature": 1.0, "top_p": 1.0, "top_k": 0},
            "topk_topp": {"temperature": 1.0, "top_p": 0.9, "top_k": 50},
        },
        "seeds": [1, 2, 3],
        "requests": [
            {
                "request_id": rid,
                "index": 0,
                "prompt_len": len(PROMPT_TOKEN_IDS),
                "max_tokens": MAX_TOKENS,
                "prompt_token_ids": list(PROMPT_TOKEN_IDS),
                "prompt_text": "",  # not reconstructed; ids are authoritative
                "text_roundtrip": False,
                "corpus_seed": 0,
            }
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    doc = workload_doc()
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=1, sort_keys=False)
        f.write("\n")
    print(f"wrote {args.out} (request {doc['requests'][0]['request_id']})")


if __name__ == "__main__":
    main()
