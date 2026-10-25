"""`p2_v1/store.py` on the CPU, at a 128-bit width with 4-block segments and the scheme's own codec in the GPU's place: the
store it writes is the registered scheme's encoding of the stream (the public-encoder check accepts it), its root is
`setup.server_root`'s tree over every leaf, W's root is the same tree over W's payloads, `verify` catches a changed byte,
and a rerun resumes to the same store."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))

from p2_v1 import store as ST  # noqa: E402
from verity.core.protocols.pous import setup, verifier  # noqa: E402
from verity.core.protocols.pous.schemes import p2 as S  # noqa: E402

POINT = "128,173,4"
SOURCE = "11" * 32
GENERATOR = '''
import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Shape:
    name: str
    size: int


SHAPES = {"t": Shape("t", 1000)}


def stream_bytes(shape):
    return shape.size


def stream(shape, seed, lo, hi):
    out = b"".join(hashlib.sha512(seed + i.to_bytes(8, "little")).digest() for i in range(lo // 64, -(-hi // 64)))
    return out[lo - (lo // 64) * 64:hi - (lo // 64) * 64]


def program(shape, seed):
    return "test/" + shape.name + "/" + hashlib.sha512(seed).hexdigest()[:8]
'''


@pytest.fixture()
def gen(tmp_path: Path) -> str:
    p = tmp_path / "gen.py"
    p.write_text(GENERATOR)
    return str(p)


def _encode(gen: str, store: Path, out: Path) -> int:
    return ST.main(["encode", "--store", str(store), "--out", str(out), "--generator", gen, "--seed", "text:seed",
                    "--shape", "t", "--point", POINT, "--source", f"hex:{SOURCE}", "--workers", "1", "--cpu-encoders", "2"])


def _segments(store: Path, q: S.P2Params) -> list[list[bytes]]:
    nb = q.block_bytes
    out = []
    for s in range(q.segments):
        c = (store / "seg" / f"{s:05d}.c").read_bytes()
        out.append([c[i * nb:(i + 1) * nb] for i in range(q.blocks_per_segment)])
    return out


def test_store_is_the_registered_encoding(gen: str, tmp_path: Path) -> None:
    store, out = tmp_path / "store", tmp_path / "out"
    assert _encode(gen, store, out) == 0
    doc = json.loads((out / "store.json").read_text())
    q = ST.params_of(doc["manifest"])
    assert q.segments == 18 and q.w_bytes >= 1000 > q.w_bytes - q.segment_bytes
    w = ST.Weights(gen, "t", b"seed").read(0, q.w_bytes)
    assert w[1000:] == bytes(q.w_bytes - 1000)
    commitment = setup.commit(q, w)
    salt = setup.draw_salt(bytes.fromhex(SOURCE), commitment)
    assert (doc["commitment"], doc["salt"]) == (commitment.hex(), salt.hex())
    vk = setup.verifier_key_from_encoding(q, None, commitment, salt, _segments(store, q))
    assert doc["root"] == vk.root.hex()
    pb = q.payload_bytes
    w_leaves = [verifier.row_leaf(q, salt, i, verifier.row_digest(pb, w[i * pb:(i + 1) * pb])) for i in range(q.blocks)]
    assert doc["w_root"] == verifier.MerkleTree(w_leaves).root.hex()
    rows = b"".join((store / "rows" / f"{s:05d}.rows").read_bytes() for s in range(q.segments))
    assert verifier.VerifierKey.from_row_digests(q, salt, [rows[64 * i:64 * i + 64] for i in range(q.blocks)]).root == vk.root
    assert doc["tie"] == ST.tie(doc) and doc["registration"] is None
    assert all(m["reference_spots"] for m in json.loads((out / "segments.json").read_text()))
    assert ST.main(["verify", "--store", str(store), "--out", str(out), "--decode", "--workers", "2"]) == 0
    assert ST.main(["spot", "--store", str(store), "--out", str(out), "--generator", gen, "--seed", "text:seed",
                    "--count", "16"]) == 0
    assert json.loads((out / "spot.json").read_text())["passed"]
    assert ST.main(["tie", "--store", str(store), "--out", str(out), "--registration", "ab" * 64]) == 0
    tied = json.loads((store / "store.json").read_text())
    assert tied["tie"] == ST.tie(tied) != doc["tie"]


def test_verify_catches_a_changed_byte(gen: str, tmp_path: Path) -> None:
    store, out = tmp_path / "store", tmp_path / "out"
    assert _encode(gen, store, out) == 0
    seg = store / "seg" / "00007.c"
    c = bytearray(seg.read_bytes())
    c[20] ^= 1
    seg.write_bytes(bytes(c))
    assert ST.main(["verify", "--store", str(store), "--out", str(out), "--workers", "1"]) == 1
    report = json.loads((out / "verify.json").read_text())
    assert [b["segment"] for b in report["segments"]["bad"]] == [7] and not report["root_ok"]


def test_rerun_resumes_to_the_same_store(gen: str, tmp_path: Path) -> None:
    store, out = tmp_path / "store", tmp_path / "out"
    assert _encode(gen, store, out) == 0
    first = json.loads((out / "store.json").read_text())
    (store / "seg" / "00003.json").unlink()
    assert _encode(gen, store, tmp_path / "out2") == 0
    again = json.loads((tmp_path / "out2" / "store.json").read_text())
    assert again["encode"]["resumed"] == first["manifest"]["segments"] - 1
    assert (again["root"], again["w_root"], again["tie"]) == (first["root"], first["w_root"], first["tie"])
    with pytest.raises(SystemExit):
        ST.main(["encode", "--store", str(store), "--out", str(out), "--generator", gen, "--seed", "text:seed",
                 "--shape", "t", "--point", POINT, "--source", f"hex:{'22' * 32}", "--workers", "1"])


def test_parts_join_to_the_whole_store(gen: str, tmp_path: Path) -> None:
    assert _encode(gen, tmp_path / "whole", tmp_path / "out") == 0
    whole = json.loads((tmp_path / "out" / "store.json").read_text())
    for name, part, order in (("a", "0:7", "up"), ("b", "7:18", "down")):
        assert ST.main(["encode", "--store", str(tmp_path / name), "--out", str(tmp_path / f"out-{name}"), "--generator", gen,
                        "--seed", "text:seed", "--shape", "t", "--point", POINT, "--source", f"hex:{SOURCE}",
                        "--workers", "1", "--segments", part, "--order", order]) == 0
    assert ST.main(["finish", "--store", str(tmp_path / "a"), "--out", str(tmp_path / "out-a")]) == 1
    for d in ("seg", "rows"):
        for f in (tmp_path / "b" / d).iterdir():
            (tmp_path / "a" / d / f.name).write_bytes(f.read_bytes())
    assert ST.main(["finish", "--store", str(tmp_path / "a"), "--out", str(tmp_path / "out-a")]) == 0
    joined = json.loads((tmp_path / "out-a" / "store.json").read_text())
    assert (joined["root"], joined["w_root"], joined["tie"]) == (whole["root"], whole["w_root"], whole["tie"])


def test_rows_below_p() -> None:
    torch = pytest.importorskip("torch")
    p = (1 << 128) - 173
    vals = [0, 1, p - 1, p, p + 1, (1 << 128) - 1, p - (1 << 64)]
    rows = torch.tensor([list(v.to_bytes(16, "big")) for v in vals], dtype=torch.uint8)
    bound = torch.tensor(list(p.to_bytes(16, "big")), dtype=torch.uint8)
    assert ST._rows_below(torch, rows, bound).tolist() == [v < p for v in vals]
