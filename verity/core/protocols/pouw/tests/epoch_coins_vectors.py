"""Pinned vectors for PoUW's epoch coins (`verity.core.protocols.pouw.coins`) and the salt and draw keys derived
from them: ``uv run python verity/core/protocols/pouw/tests/epoch_coins_vectors.py`` rewrites
``tests/vectors/epoch_coins_v1.json``.  A change to that file is a change to the coin commitment or a derivation, and needs a
reason in the PR.

A case's challenger draws from `epoch_coins.entropy(seed)`; the weights' root and the transcript's root are
SHA-256("verity/pouw/epoch-coins-vectors/v1", case, "weights" | "transcript").  A case pins every coin, hm96 salt,
committed value and path, the root, the epoch's salt (`audit.salt_of`), the generic draw's key (`audit.AUDIT_DOMAIN`) and
Pearl-C's (`pearl_c_work.draw_key`) under a records root SHA-256(…, case, "records")."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import epoch_coins as E
from verity.ml.randomness import derive
from verity.core.protocols.pouw import audit as au
from verity.core.protocols.pouw import coins as EC
from verity.core.protocols.pouw.schemes import pearl_c_work as W

OUT = Path(__file__).with_name("vectors") / "epoch_coins_v1.json"
#: (case, the challenger's seed, epoch)
CASES = (("epoch-7", E.SEED, 7), ("epoch-0", b"another challenger", 0), ("epoch-2^40", E.SEED, 1 << 40))


def label(case: str, part: str) -> bytes:
    return hashlib.sha256("\0".join(("verity/pouw/epoch-coins-vectors/v1", case, part)).encode()).digest()


def case_vector(case: str, seed: bytes, epoch: int) -> dict:
    ch = E.challenger(seed, epoch)
    weights, transcript, records = label(case, "weights"), label(case, "transcript"), label(case, "records")
    salt_coin, draw_coin = ch.open("salt", weights), ch.open("draw", transcript)
    salt = au.salt_of(ch.commitment, salt_coin.coin, weights)
    return {
        "case": case, "seed": seed.hex(), "epoch": epoch, "commitment": ch.commitment.to_json(),
        "values": {o.label: EC.leaf_value(o.coin, o.salt).hex() for o in (salt_coin, draw_coin)},
        "openings": [salt_coin.to_json(), draw_coin.to_json()],
        "weights_root": weights.hex(), "salt": salt.hex(),
        "transcript_root": transcript.hex(),
        "audit_key": derive(draw_coin.coin, au.AUDIT_DOMAIN, {"transcript": transcript, "salt": salt}).value.hex(),
        "records_root": records.hex(),
        "pearl_c_draw_key": W.draw_key(draw_coin.coin, transcript, records, salt).value.hex(),
    }


def vectors() -> dict:
    return {"schema": "verity-pouw/epoch-coins-vectors/v1", "scheme": EC.SCHEME, "salt_domain": au.EPOCH_SALT_DOMAIN,
            "cases": [case_vector(*c) for c in CASES]}


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(vectors(), indent=1) + "\n")
    print(f"wrote {OUT}")
