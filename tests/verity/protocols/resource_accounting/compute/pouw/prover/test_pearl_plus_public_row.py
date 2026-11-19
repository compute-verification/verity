"""The vectors of the public row's supplied noise (`pearl_plus_credited.SUPPLIED`), the bytes a supplied function's Lean body
is held to: ``fixtures/pearl_plus_public_row.json``, regenerated here from the reference and compared byte for byte.

Each case gives its inputs (salt, root_B, k, col, tn, dev), seed_B, and each field's SHA-512 digest; at k ≤ 64 it also gives
the field's full bytes. Its `--write` rewrites the file; `REWRITE` names the command that runs it."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from verity.protocols.resource_accounting.compute.pouw.prover.circuit import pearl_plus_credited as PC
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
            case[name] = {"supplied": f"pouw.pearlplus.{name}/v1", "nbytes": len(b), "sha512": hashlib.sha512(b).hexdigest(),
                          **({"bytes": b.hex()} if k <= FULL_K or name == "eb" else {})}
        cases.append(case)
    return {"about": ("the public row's supplied noise: each field's bytes as its port reads them, from "
                      "`pearl_plus_credited.SUPPLIED` (the derivation is that module's docstring); written by "
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


def test_public_reads_its_noise_back_from_the_supplied_bytes():
    salt, root = _inputs(1)
    t = PC.Tile(None, (0, 1), (2, 3), 1)
    row = PC.public("pearl-c-h100-v1-h3", 32, t, 7, salt, root)
    assert bytes(sum(row["eb"], [])) == PC.row_eb(salt, root, 1, 2)
    assert bytes(sum(row["fa"], [])) == PC.row_fa(salt, root, 32) and bytes(sum(row["fb"], [])) == PC.row_fb(salt, root, 32)
    assert b"".join(w.to_bytes(4, "little") for r in row["g"] for w in r) == PC.row_gram(salt, root, 32, "h100")


if __name__ == "__main__" and "--write" in sys.argv:
    PATH.write_text(json.dumps(build(), indent=1) + "\n")
    print(f"wrote {PATH}")
