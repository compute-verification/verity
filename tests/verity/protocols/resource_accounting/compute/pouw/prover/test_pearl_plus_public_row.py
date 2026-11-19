"""The vectors of the noise a credited unit derives inside from its public row (`PearlPlusTileNoise_v2`, the eb, fa, fb and
g of `pearl_plus_credited.PearlPlusTileCoreWords_v3`): ``fixtures/pearl_plus_public_row.json``, regenerated here from the
reference functions (`pearl_plus_credited.row_eb` …) and compared byte for byte, and the Definition, on its words and on
bits, held to them.

Each case gives its inputs (salt, root_B, k, col, tn, dev), seed_B, and each value's SHA-512 digest; at k ≤ 64 it also gives
the value's full bytes. Its `--write` rewrites the file; `REWRITE` names the command that runs it."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from verity.circuits.defs import bind
from verity.circuits.evaluation import evaluate
from verity.circuits.evaluation.bits import evaluate_words
from verity.protocols.resource_accounting.compute.pouw.prover.circuit import pearl_plus_credited as PC
from verity.protocols.resource_accounting.compute.pouw.prover.circuit import pearl_plus_tile as PT
from verity.protocols.resource_accounting.compute.pouw.prover.schemes import pearl_c as C

PATH = Path(__file__).parent / "fixtures" / "pearl_plus_public_row.json"
REWRITE = ("the vectors differ from the reference's; a change meant to move them rewrites them with "
           "`uv run python tools/check/pins.py --write tests/verity/protocols/resource_accounting/compute/pouw/prover/fixtures/pearl_plus_public_row.json`")
#: the largest k whose fields the file holds in full; above it, their digests only
FULL_K = 64
#: (k, col, tn, dev): two k in full and k = 256, every device, four column tiles and three tile widths
CASES = (
    (32, 0, 1, "h100"),
    (32, 1, 2, "sm120"),
    (64, 2, 2, "sm120-unpromoted"),
    (64, 5, 8, "h100"),
    (256, 0, 4, "h100"),
    (256, 3, 1, "sm120"),
    (256, 1, 2, "sm120-unpromoted"),
)


def _inputs(i: int) -> tuple[bytes, bytes]:
    if i == 0:
        return bytes(32), b"\xff" * 64
    h = hashlib.sha512(b"pearl-plus/public-row/vector" + bytes([i])).digest()
    return h[:32], hashlib.sha512(h).digest()


def build() -> dict:
    cases = []
    for i, (k, col, tn, dev) in enumerate(CASES):
        salt, root = _inputs(i)
        fields = {"eb": PC.row_eb(salt, root, col, tn), "fa": PC.row_fa(salt, root, k), "fb": PC.row_fb(salt, root, k),
                  "gram": PC.row_gram(salt, root, k, dev)}
        case = {"salt": salt.hex(), "root_b": root.hex(), "k": k, "col": col, "tn": tn, "dev": dev,
                "seed_b": C.seed_b(salt, root, "h3").hex()}
        for name, b in fields.items():
            case[name] = {"nbytes": len(b), "sha512": hashlib.sha512(b).hexdigest(),
                          **({"bytes": b.hex()} if k <= FULL_K or name == "eb" else {})}
        cases.append(case)
    return {"about": ("the noise a credited unit derives inside from its public row: each value's bytes as its core port "
                      "reads them, from `pearl_plus_credited`'s reference functions (the derivation is that module's "
                      "docstring); written by "
                      "tests/verity/protocols/resource_accounting/compute/pouw/prover/test_pearl_plus_public_row.py --write"),
            "cases": cases}


def test_the_vectors_are_the_references_bytes():
    assert json.loads(PATH.read_text()) == build(), REWRITE


def test_the_vectors_cover_two_devices_two_columns_and_full_bytes():
    cases = json.loads(PATH.read_text())["cases"]
    assert {c["dev"] for c in cases} >= {"h100", "sm120"} and len({c["col"] for c in cases}) >= 2
    full = [c for c in cases if "bytes" in c["gram"]]
    assert {c["dev"] for c in full} >= {"h100", "sm120"}
    for c in full:
        for name in ("eb", "fa", "fb", "gram"):
            b = bytes.fromhex(c[name]["bytes"])
            assert len(b) == c[name]["nbytes"] and hashlib.sha512(b).hexdigest() == c[name]["sha512"]
        assert c["fa"]["nbytes"] == 32 * c["k"] and c["eb"]["nbytes"] == 32 * c["tn"] and c["gram"]["nbytes"] == 4096


def _noise(out: list[int], k: int, tn: int) -> dict[str, bytes]:
    """The Definition's outputs as the reference's bytes: eb's, fa's and fb's codes, then g's FP32 words little-endian."""
    eb, fa, fb = (bytes(out[lo:hi]) for lo, hi in ((0, 32 * tn), (32 * tn, 32 * (tn + k)), (32 * (tn + k), 32 * (tn + 2 * k))))
    return {"eb": eb, "fa": fa, "fb": fb, "gram": b"".join(w.to_bytes(4, "little") for w in out[32 * (tn + 2 * k):])}


def _agrees(c: dict, got: dict[str, bytes]) -> bool:
    return all(hashlib.sha512(got[n]).hexdigest() == c[n]["sha512"] for n in ("eb", "fa", "fb", "gram"))


def _args(c: dict) -> list[list[int]]:
    return [list(bytes.fromhex(c["salt"])), list(bytes.fromhex(c["root_b"])), [c["col"]]]


def test_the_units_noise_on_words_is_the_vectors():
    """`PearlPlusTileNoise_v2`'s word body at every case: col · TN + j from the column word, both chains and every device."""
    for c in json.loads(PATH.read_text())["cases"]:
        word = bind(PT.PearlPlusTileNoiseV2.sigma_of, K=c["k"], TN=c["tn"], DEV=c["dev"])
        assert _agrees(c, _noise(list(evaluate(word, *_args(c))), c["k"], c["tn"])), c


@pytest.mark.slow
def test_the_units_noise_on_bits_is_the_vectors():
    """Slow: `PearlPlusTileNoise_v2` on bits (2K + TN noise lines and the Gram, lowered) at every case, by its word view:
    the only check that the σ of the noise, the Definition the unit Calls, derives the vectors' bytes bit for bit."""
    for c in json.loads(PATH.read_text())["cases"]:
        fn = bind(PT.PearlPlusTileNoiseV2, K=c["k"], TN=c["tn"], DEV=c["dev"])
        out = evaluate_words(fn, *(np.asarray([x], dtype=np.uint64) for x in _args(c))).reshape(-1).tolist()
        assert _agrees(c, _noise(out, c["k"], c["tn"])), c


if __name__ == "__main__" and "--write" in sys.argv:
    PATH.write_text(json.dumps(build(), indent=1) + "\n")
    print(f"wrote {PATH}")
