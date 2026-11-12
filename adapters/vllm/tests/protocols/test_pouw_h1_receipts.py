"""serve-openai's receipt producer (`pouw_h1_receipts.Receipts.serving`, ``--h1-receipts``), and a served window's
receipts as a lanes-form K's combine statement reads them (`Window`, which `test_serve_h1` holds a served window to).

`Receipts.serving` computes on the device the served weights are on, each call's B its linear's own weight tensor (no
second copy), with noise.cu's noise on a CUDA device and hashlib's elsewhere. Its receipt of a proof block is v1's
(`h1tile.reference`), which `h1zy.reference` returns without J10-R, and the 32 lane partials of the proof block
(`h1zy.lane_partial`, every admit bit 1) XOR to it (`h1zy.combine_reference`). The combine's reader rebuilds a call's
``receipt@<c>`` from the window's record, the prover's kept rows (`pouw_h1_prover`) and the served weights, as the word
reference computes it: proof block t at position u0 + t (u0 the call's first proof block, `credit.call_work`), seeded from
the window's Program salt and the registered roots of ``a/<K>@<s>`` and ``b/<K>``, each row its receipt's u16 words, the
table committed at ``RECEIPT_NONCE`` + c under the prover's key and bound to the window's root. It refuses the table
unless the record registers it once at that root and domain."""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from verity.protocols.pouw.prover.circuit import h1zy as Z
from verity.protocols.pouw.verifier import credit as CR
from verity.protocols.pouw.verifier import leaf_lanes as LL

from verity_vllm.protocols.pouw.h1 import prover as PV
from verity_vllm.protocols.pouw.h1 import receipts as RC
from verity_vllm.protocols.pouw.h1 import registered as NR
from verity_vllm.protocols.pouw.h1 import served as SV

M32 = (1 << 32) - 1
R, C = TILE = (32, 32)
SALT, RA, RB = (hashlib.sha512(x).digest() for x in (b"salt", b"A", b"B"))


def receipt_words(rc: Any) -> np.ndarray:
    """A receipt's u32 words as its table row's u16 words, low half first."""
    return np.array([int(v) & M32 for v in rc], dtype="<u4").view("<u2").astype(np.uint16)


def u32_words(words: Any) -> list[int]:
    return [int(v) for v in np.ascontiguousarray(np.asarray(words, np.uint16).astype("<u2")).view("<u4")]


def tile_rows(a: np.ndarray, b: np.ndarray, t: int) -> tuple[list[list[int]], list[list[int]]]:
    """Proof block t of A [M, K] and B [N, K] (s8): its 32 A rows and 32 B rows as u8 encodings, in tile order."""
    i, j = divmod(t, b.shape[0] // C)
    return a[R * i:R * i + R].view(np.uint8).tolist(), b[C * j:C * j + C].view(np.uint8).tolist()


def lane_partials(a: list, b: list, salt: bytes, position: int, roots: tuple[bytes, bytes]) -> list[list[int]]:
    """A proof block's 32 lane partials, the rows its lanes commit: `h1zy.lane_partial` of each lane's 4 A rows and 8 B
    rows (`leaf_lanes.lane_rows`, `lane_cols`), every admit bit 1."""
    return [Z.lane_partial([a[i] for i in LL.lane_rows(R, C, ln)], [b[j] for j in LL.lane_cols(R, C, ln)], [1] * R,
                           salt, position, roots, ln, TILE) for ln in range(LL.lanes(R, C))]


def combine_ok(partials: list[list[int]], receipt: Any) -> int:
    """The combine's ok: `h1zy.combine_reference` of a proof block's lane partials and its receipt row's u16 words."""
    return Z.combine_reference(partials, u32_words(receipt))


class Window:
    """A served window as the combine's reader takes it: its record ``record`` (`pouw_h1.CALLS_FILE`'s), the period
    Program's descriptor bytes ``raw``, the window's root ``root``, its Program salt ``salt``, the prover's directory
    ``prover`` (its key and kept rows) and ``weights``, each linear's served int8 codes (n, K)."""

    def __init__(self, record: dict, raw: bytes, root: str, salt: bytes, prover: Path, weights: list[np.ndarray]):
        self.layout, why = CR.layout_of(raw)
        assert why is None, why
        self.program, self.prefix = hashlib.sha512(raw).hexdigest(), CR.shape_of(raw)["prefix"]
        self.calls = {int(row["index"]): row for row in record["calls"]}
        self.entries: dict[str, list[dict]] = {}
        for e in record["commitment"]["served"]["entries"]:
            self.entries.setdefault(e["value"], []).append(e)
        self.root, self.salt, self.prover, self.weights = root, salt, Path(prover), weights
        self.sizes = [size for size, _work in CR.call_work(self.layout)]
        self.u0 = [sum(self.sizes[:c]) for c in range(len(self.sizes))]

    def receipt(self, c: int) -> str:
        return f"{self.prefix}{LL.RECEIPT}@{c}"

    def entry(self, value: str) -> dict:
        """``value``'s entry in the record; AssertionError, the reader's refusal, unless it is registered exactly once."""
        got = self.entries.get(value, [])
        assert len(got) == 1, f"the window registers {value} {len(got)} times, not once"
        return got[0]

    def roots(self, c: int) -> tuple[bytes, bytes]:
        """Call c's (root(a/<K>@<s>), root(b/<K>)) as the record registers them."""
        k, s = int(self.calls[c]["k"]), int(self.calls[c]["step"])
        return tuple(bytes.fromhex(self.entry(f"{self.prefix}{v}")["root"]) for v in (f"a/{k}@{s}", f"b/{k}"))

    def operands(self, c: int) -> tuple[np.ndarray, np.ndarray]:
        """Call c's A over the Program's M slots (its served codes as the prover kept them, then zero rows) and its B."""
        row = self.calls[c]
        k, m = int(row["k"]), int(row["served"])
        a: np.ndarray = np.zeros((int(self.layout["m"]), k), np.int8)
        if m:
            with np.load(self.prover / PV.CALLS / f"{c}.npz") as kept:
                a[:m] = kept["a"].reshape(m, k)
        return a, self.weights[int(row["layer"])]

    def commit(self, c: int, words: np.ndarray) -> Any:
        """Call c's receipt table of ``words`` as `Served.commit` registers it, its rows kept."""
        return NR.commit(self.program, self.receipt(c), words, PV.key_of(self.prover), SV.RECEIPT_NONCE + c,
                         keep_rows=True, window=self.root)

    def receipt_value(self, c: int) -> Any:
        """Call c's ``receipt@<c>`` rebuilt as the word reference computes it (`h1zy.reference`, the module docstring)."""
        a, b = self.operands(c)
        rows = [receipt_words(Z.reference(*tile_rows(a, b, t), [0] * R, [0] * C, ([], []), self.salt, self.u0[c] + t,
                                          self.roots(c))[0]) for t in range(self.sizes[c])]
        return self.commit(c, np.array(rows, np.uint16))

    def held(self, tree: Any) -> bool:
        """Whether a rebuilt table is the tree the record registers under its name: its root and domain."""
        e = self.entry(tree.value)
        return (tree.root.hex(), tree.domain.domain_id.hex()) == (e["root"], e["domain"])

    def partials(self, c: int, t: int) -> list[list[int]]:
        """`lane_partials` of call c's proof block t."""
        a, b = self.operands(c)
        return lane_partials(*tile_rows(a, b, t), self.salt, self.u0[c] + t, self.roots(c))


class _Linear:
    def __init__(self, n: int, k: int, seed: int):
        self.weight = torch.from_numpy(np.random.default_rng(seed).integers(-127, 128, (n, k), dtype=np.int8))


def test_serving_computes_on_the_served_weights_in_place_and_binds_the_run_once_it_exists():
    """On the device the served weights are on (the CPU here: hashlib's noise, no cubin), each call's B the linear's own
    weight tensor, never a copy, and the run bound after the producer is made (`H1Windows.open` makes it before the
    install that makes the run); with a layer holding no weight, B is the codes `Served` hands over, copied once."""
    layers, served, run = [_Linear(64, 64, 1), _Linear(32, 64, 2)], SimpleNamespace(m=32), SimpleNamespace(salt=SALT)
    made = RC.Receipts.serving(served, None, layers)
    assert (made.device, made.cubin, made.run, made.served) == (torch.device("cpu"), None, None, served)
    call = SimpleNamespace(index=1, layer=1, n=32, k=64)
    assert made.weight_of(call, None) is layers[1].weight
    assert made.bind(run).run is run
    bare = RC.Receipts.serving(served, run, [layers[0], SimpleNamespace()])
    handed: np.ndarray = np.ones((32, 64), np.int8)
    got = bare.weight_of(call, handed)
    assert got is not layers[1].weight and np.array_equal(got.numpy(), handed) and bare.weight_of(call, handed) is got


def test_the_producers_receipt_is_the_word_references_and_its_lane_partials_xor_to_it():
    """A call's receipts from the torch chain (v1's proof block, no J10-R) are `h1zy.reference`'s at each proof block's
    position, a live and a dead row block alike, and the combine's ok on the live proof block's lane partials and its
    receipt is 1; one flipped receipt bit and it is 0."""
    rng = np.random.default_rng(20261101)
    a: np.ndarray = np.zeros((64, 64), np.int8)
    b, u0 = rng.integers(-127, 128, (32, 64), dtype=np.int8), 10
    a[:20] = rng.integers(-127, 128, (20, 64), dtype=np.int8)
    got = RC.call_receipts(RC.Call(torch.from_numpy(a), torch.from_numpy(b)), SALT, u0, RA, RB)
    assert got.shape == (2, 2 * (64 // 32 + 2))
    for t in range(2):
        rc, _ok = Z.reference(*tile_rows(a, b, t), [0] * R, [0] * C, ([], []), SALT, u0 + t, (RA, RB))
        assert got[t].tolist() == rc, f"proof block {t}"
    partials, words = lane_partials(*tile_rows(a, b, 0), SALT, u0, (RA, RB)), receipt_words(got[0])
    assert combine_ok(partials, words) == 1
    words[0] ^= 1
    assert combine_ok(partials, words) == 0
