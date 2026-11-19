"""The vectors of the noise a credited unit derives inside from its public row (the eb, fa, fb and g of
`pearl_plus_credited.PearlPlusTileCoreWords_v3`, its units `PearlPlusTileSeed_v1`, `PearlPlusTileLine_v2` and
`PearlPlusGramRow_v2`): ``fixtures/pearl_plus_public_row.json``, regenerated here from the reference functions
(`pearl_plus_credited.row_eb` …) and compared byte for byte, and the units, on their words and on bits, held to them.

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
from verity.circuits.evaluation import evaluate_batch
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


def _words(bits: list[int], w: int) -> list[int]:
    return [sum(bits[w * i + j] << j for j in range(w)) for i in range(len(bits) // w)]


def noise_units(c: dict, *, bits: bool) -> dict[str, bytes]:
    """The case's noise as the node's units derive it, wired as the node wires them: the seed (key and line indices),
    one line per index, one Gram row per F_B column; on their word bodies, or on bits (the seed's words in and bits out,
    the lines' and rows' σ by word view).  eb's, fa's and fb's codes, then g's FP32 words little-endian, as the
    reference's bytes."""
    k, tn, dev = c["k"], c["tn"], c["dev"]
    if bits:
        run, seed, line, row = evaluate_words, PT.seed(k, tn), PT.line, PT.gram_row(k, dev)
    else:
        run, seed, row = evaluate_batch, bind(PT._SEED, K=k, TN=tn), bind(PT._GRAM_ROW, K=k, DEV=dev)
        line = lambda kind: bind(PT._LINE, **dict(zip(("SIDE", "FACTOR"), PT.LINES[kind])))  # noqa: E731

    def go(fn, *xs):
        return run(fn, *(np.asarray(x, dtype=np.uint64 if bits else np.int64) for x in xs)).reshape(len(xs[0]), -1)

    s = evaluate_batch(seed, *(np.asarray(x, dtype=np.int64) for x in (
        [list(bytes.fromhex(c["salt"]))], [list(bytes.fromhex(c["root_b"]))], [[c["col"]]]))).reshape(-1).tolist()
    if bits:
        s = _words(s[:256], 8) + _words(s[256:], 32)
    key, ebi, fi = s[:32], s[32:32 + tn], s[32 + tn:]
    eb = go(line("eb"), [key] * tn, [[i] for i in ebi])
    fa, fb = (go(line(kind), [key] * k, [[i] for i in fi]) for kind in ("fa", "fb"))
    g = go(row, [fb[:, p] for p in range(32)], [fa.reshape(-1)] * 32)
    return {"eb": bytes(eb.reshape(-1).tolist()), "fa": bytes(fa.reshape(-1).tolist()), "fb": bytes(fb.reshape(-1).tolist()),
            "gram": b"".join(int(w).to_bytes(4, "little") for w in g.reshape(-1))}


def _agrees(c: dict, got: dict[str, bytes]) -> bool:
    return all(hashlib.sha512(got[n]).hexdigest() == c[n]["sha512"] for n in ("eb", "fa", "fb", "gram"))


def test_the_units_noise_on_words_is_the_vectors():
    """The noise units' word bodies at every case: col · TN + j from the column word, both chains and every device."""
    for c in json.loads(PATH.read_text())["cases"]:
        assert _agrees(c, noise_units(c, bits=False)), c


@pytest.mark.slow
def test_the_units_noise_on_bits_is_the_vectors():
    """Slow: the noise units on bits (the seed, 2K + TN lines and 32 Gram rows, lowered) at every case, by their word view:
    the only check that the σ the node Calls and maps derive the vectors' bytes bit for bit."""
    for c in json.loads(PATH.read_text())["cases"]:
        assert _agrees(c, noise_units(c, bits=True)), c


if __name__ == "__main__" and "--write" in sys.argv:
    PATH.write_text(json.dumps(build(), indent=1) + "\n")
    print(f"wrote {PATH}")
