"""The toy model served from its P2-encoded weights store under PoUB: the CPU reference of the
serving loop, and the fixture a decode kernel's sketch is held to.

    serve.py [--steps 12] [--per-window 3] [--prompt 1]   # serve the toy, print its tokens and each window's verdict
    serve.py fixture [--out receipts.json]                 # write the kernel fixture (stdout by default)

**The step.** Each engine step is one pass of the audit (`remote_bandwidth.p2`). It decodes the store layer by layer
(`store.decode_block`: block `i`'s 2,056 bytes of `C*` to its 2,054 of `W`), runs the layer's matmuls on the staging
buffer, and, as it decodes block `i`, sketches the same 2,056 bytes under the pass's seed (`p2.Sketcher.block`, π).
At the step's end it sums each unit's four partials (σ), hashes them to the receipts (y), commits them in the pass's
tree, and sends the root. The next step's seed is `receipts.seed(σ_t, i + 1, root)`: one root a step, chained. The
plan is every unit, since a step reads every weight; the store's tail past its last whole unit (the toy's block 32,
which holds no tensor) is in no unit.

**The windows.** On the verifier's clock (one injected clock here) a window's boundary leaves once its stated window
has passed, the next step's root goes to the next window, and that root makes the window before final: the verifier
draws its openings, and the prover answers them before its next step, while serving goes on. The session's last
window closes at its deadline.

**The fixture** (`fixture`): two chained passes over the toy's `C*` at 16 sign rows under a fixed challenge and fixed
leaf salts, each with its seed, the 16 sign rows the kernel reads, every block's partial π, every unit's σ and y, and
the pass's root. A decode kernel's sketch matches it block for block (`PROTOCOL.md` §11 of
`verity/core/protocols/remote_bandwidth` is the interface). It is regenerated, not tracked: the repository's data
budget has no room for it, and `FIXTURE_SHA512` pins its bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from p2_v1 import store as S
else:
    from . import store as S

from verity.core.protocols.pous import setup
from verity.core.protocols.pous.schemes.p2 import expand_prefix
from verity.core.protocols.remote_bandwidth import audit, coins, p2, receipts

FIXTURE_SCHEMA = "verity-remote-bandwidth/p2-sketch-fixture/v0"
FIXTURE_SHA512 = ("658f6d40a5ecdcb252ca2d4b844f41c6b2af0550f291a4d2e33f975d3158fac2"
                  "b3da2a65666d146f6aabac37167b4d21aaf4d1ef5c02127ac1ad2f8380f88b9a")
NODE = b"p2-toy-node"
EMBED = b"verity-pous/p2-serve/toy/embedding/v0\0"
FIXTURE_CHALLENGE = hashlib.sha512(b"verity-pous/p2-serve/fixture/challenge/v0").digest()[:receipts.CHALLENGE_BYTES]
FIXTURE_SALTS = b"verity-pous/p2-serve/fixture/leaf-salts/v0\0"


class Entropy:
    """A deterministic byte source: SHA-512 of `label` and a counter."""

    def __init__(self, label: bytes) -> None:
        self.label, self.count = label, 0

    def __call__(self, n: int) -> bytes:
        out = b""
        while len(out) < n:
            out += hashlib.sha512(self.label + self.count.to_bytes(8, "big")).digest()
            self.count += 1
        return out[:n]


class Clock:
    """The session's clock, advanced by the serving loop: `step_ns` an engine step."""

    def __init__(self) -> None:
        self.now = 0

    def __call__(self) -> int:
        return self.now


def toy() -> tuple[S.Store, dict, bytes]:
    """The tracked toy: its store, `toy.json` and `C*`."""
    doc = json.loads((S.TOY_DIR / "toy.json").read_text())
    c_star = (S.TOY_DIR / "c-star.bin").read_bytes()
    if hashlib.sha512(c_star).hexdigest() != doc["c_star_sha512"]:
        raise ValueError("c-star.bin is not toy.json's C*")
    return S.toy_store(doc), doc, c_star


def blocks_of(store: S.Store, c_star: bytes) -> list[bytes]:
    nb = store.params.block_bytes
    return [c_star[nb * j:nb * (j + 1)] for j in range(store.params.blocks)]


def registration(store: S.Store, doc: dict, c_star: bytes, node: bytes = NODE) -> p2.Registration:
    """The verifier's registration of the toy's store: PoUS's setup decoding `C*` to the committed `W` under the salt
    it drew from its source after the commitment, then `p2.register`."""
    params, n = store.params, store.params.blocks_per_segment
    commitment, salt, source = (bytes.fromhex(doc[k]) for k in ("commitment", "salt", "source"))
    blocks = blocks_of(store, c_star)
    vk = setup.verifier_key_from_encoding(params, None, commitment, salt,
                                          [blocks[n * s:n * (s + 1)] for s in range(params.segments)])
    return p2.register(source, node, commitment, vk)


def _requant(v: np.ndarray, k: int) -> np.ndarray:
    """int32 accumulators of a `k`-deep int8 dot product back to int8: an arithmetic shift, saturated."""
    return np.clip(v >> ((k.bit_length() + 13) // 2), -128, 127).astype(np.int8)


def _add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.clip(x.astype(np.int16) + y, -128, 127).astype(np.int8)


class Engine:
    """The toy model (`store.toy_shape`'s MQA transformer, all int8) served from `C*`: `step(token, sketcher)` decodes
    every layer's blocks, sketching each block's bytes as it decodes them, runs the layer on them, and returns the next
    token (greedy) with every audited block's partial π. Attention is hard (each query head takes the value of its
    highest-scoring cached key, the latest on a tie), a KV cache a layer. With `w` given, it reads that plaintext `W`
    instead of decoding, and sketches nothing: the model the decode must agree with."""

    def __init__(self, store: S.Store, c_star: bytes, salt: bytes, w: bytes | None = None) -> None:
        self.store, self.c_star, self.w = store, c_star, w
        self.prefix = expand_prefix(store.params, salt)
        self.audited = p2.UNIT_BLOCKS * (store.params.blocks // p2.UNIT_BLOCKS)
        self.cache: list[list[tuple[np.ndarray, np.ndarray]]] = [[] for _ in range(store.shape.layers)]
        self.decoded = 0

    def _layer(self, layer: int, sketcher: p2.Sketcher | None, partials: dict[int, bytes]) -> dict[str, np.ndarray]:
        params, nb = self.store.params, self.store.params.block_bytes
        out = {}
        for t in self.store.layer(layer):
            if self.w is not None:
                data = self.w[t.offset:t.offset + t.size]
            else:
                buf = bytearray()
                for i in range(t.first_block, t.first_block + t.blocks):
                    block = self.c_star[nb * i:nb * (i + 1)]
                    if sketcher is not None and i < self.audited:
                        partials[i] = sketcher.block(i, block)
                    buf += S.decode_block(params, self.prefix, i, block)
                    self.decoded += 1
                if any(buf[t.size:]):
                    raise ValueError(f"{t.name}'s last block is nonzero past its end (flag bit 3)")
                data = bytes(buf[:t.size])
            out[t.name.rsplit(".", 2)[-2] if t.name != "lm_head.weight" else "head"] = (
                np.frombuffer(data, dtype=np.int8).reshape(t.n, t.k).astype(np.int32))
        return out

    def embed(self, token: int) -> np.ndarray:
        d = self.store.shape.d
        return np.frombuffer(hashlib.shake_256(EMBED + token.to_bytes(4, "big")).digest(d), dtype=np.int8).copy()

    def step(self, token: int, sketcher: p2.Sketcher | None = None) -> tuple[int, dict[int, bytes]]:
        shape, partials = self.store.shape, {}
        x = self.embed(token)
        hd, kv = shape.head_dim, shape.kv_heads * shape.head_dim
        for layer in range(shape.layers):
            w = self._layer(layer, sketcher, partials)
            qkv = _requant(w["qkv_proj"] @ x.astype(np.int32), shape.d)
            q, k, v = qkv[:shape.heads * hd], qkv[shape.heads * hd:shape.heads * hd + kv], qkv[shape.heads * hd + kv:]
            self.cache[layer].append((k.astype(np.int32), v))
            heads = []
            for h in range(shape.heads):
                qh = q[hd * h:hd * (h + 1)].astype(np.int32)
                scores = [int(qh @ kc[:hd]) for kc, _ in self.cache[layer]]
                best = max(range(len(scores)), key=lambda j: (scores[j], j))
                heads.append(self.cache[layer][best][1][:hd])
            a = np.concatenate(heads).astype(np.int32)
            x = _add(x, _requant(w["o_proj"] @ a, shape.heads * hd))
            u = np.maximum(_requant(w["up_proj"] @ x.astype(np.int32), shape.d), 0).astype(np.int32)
            x = _add(x, _requant(w["down_proj"] @ u, shape.ff))
        head = self._layer(shape.layers, sketcher, partials)["head"]
        logits = head @ x.astype(np.int32)
        return int(np.flatnonzero(logits == logits.max())[-1]), partials


def unit_receipts(sketcher: p2.Sketcher, units: int, partials: dict[int, bytes]) -> tuple[list[bytes], list[bytes]]:
    """Each unit's σ (its four partials summed) and y, in unit order."""
    sigmas = [sketcher.total([partials[p2.UNIT_BLOCKS * g + b] for b in range(p2.UNIT_BLOCKS)]) for g in range(units)]
    return sigmas, [sketcher.receipt(g, s) for g, s in enumerate(sigmas)]


@dataclass
class Served:
    """A served session: the tokens, which pass covered each step (window, pass), each window's verdict with the step
    before which the prover answered its openings, and the session's outcome."""

    tokens: list[int] = field(default_factory=list)
    passes: list[tuple[int, int]] = field(default_factory=list)
    verdicts: list[tuple[audit.Verdict, int]] = field(default_factory=list)
    outcome: str | None = None


Open = Callable[[audit.Pass, int], p2.Opening]
Step = Callable[[Engine, int, p2.Sketcher], tuple[int, dict[int, bytes]]]


def serve(engine: Engine, store: p2.Store, *, prompt: int, windows: int, per_window: int, step_ns: int = 1000,
          samples: int | None = None, entropy: Callable[[int], bytes] | None = None, step: Step | None = None,
          open_: Callable[[p2.Prover], Open] | None = None) -> Served:
    """Serve `windows · per_window` steps from `prompt` under one audit session over `store` (the prover's `C*`, the
    engine's), one pass a step, each window `per_window` steps on the session's clock, `samples` openings a window
    (every receipt by default). `step` and `open_` replace the honest step and opening, to fault them."""
    reg = store.registration
    clock, units, plan = Clock(), reg.blocks, receipts.Plan.whole(reg.blocks)
    params = audit.Params(blocks=units, deadline_ns=step_ns, windows=windows, samples=samples or per_window * units,
                          delta=1e-6, window_ns=per_window * step_ns, codec=reg.codec, receipt=reg.receipt,
                          layout=reg.layout, rows=reg.rows)
    entropy = entropy or Entropy(b"verity-pous/p2-serve/entropy/v0")
    session = p2.Session(params, reg, coins.local(reg.digest, 0, windows, entropy), not_before_ns=0, clock=clock)
    prover = p2.Prover(store, session.commitment, entropy=entropy)
    opener = (open_ or (lambda pr: pr.open))(prover)
    out, token, previous = Served(), prompt, reg.digest
    challenge, t, i, closed, passes = session.start(), 0, 0, 0, {}

    def answer(u: int, before: int) -> bool:
        draws = session.close(u)
        if isinstance(draws, audit.Verdict):
            out.verdicts.append((draws, before))
            return False
        v = session.verify(u, [opener(passes[u, a], q) for a, q in draws])
        out.verdicts.append((v, before))
        return v.accepted

    for n in range(windows * per_window):
        due = session.next_boundary_ns()
        if t + 1 < windows and due is not None and clock.now >= due:
            challenge, t, i = session.boundary(), t + 1, 0
        seed = receipts.seed(coins.coin(challenge), i, previous)
        sketcher = p2.Sketcher(seed, reg.rows)
        token, partials = (step or (lambda e, x, sk: e.step(x, sk)))(engine, token, sketcher)
        clock.now += step_ns
        ps = prover.commit(t, i, seed, plan, unit_receipts(sketcher, units, partials)[1])
        if (v := session.receive(t, i, ps.root, step=n)) is not None:
            out.verdicts.append((v, n))
            out.outcome = session.outcome
            return out
        out.tokens.append(token)
        out.passes.append((t, i))
        passes[t, i], previous, i = ps, ps.root, i + 1
        while closed < t:
            if not answer(closed, n + 1):
                out.outcome = session.outcome
                return out
            closed += 1
    if session.boundary() is not None:
        raise RuntimeError("the session's last window is its own")
    clock.now = session.deadline(t) + 1
    answer(t, windows * per_window)
    out.outcome = session.outcome
    return out


def fixture(passes: int = 2) -> dict:
    """The kernel fixture: `passes` chained passes over the toy's `C*` under `FIXTURE_CHALLENGE`, at 16 rows."""
    store, doc, c_star = toy()
    reg = registration(store, doc, c_star)
    blocks, units = blocks_of(store, c_star), reg.blocks
    salts, plan, previous, out = Entropy(FIXTURE_SALTS), receipts.Plan.whole(units), reg.digest, []
    for i in range(passes):
        seed = receipts.seed(FIXTURE_CHALLENGE, i, previous)
        sk = p2.Sketcher(seed, reg.rows)
        partials = {j: sk.block(j, blocks[j]) for j in range(p2.UNIT_BLOCKS * units)}
        sigmas, ys = unit_receipts(sk, units, partials)
        leaf_salts = [salts(receipts.SALT_BYTES) for _ in range(receipts.leaves(plan.length))]
        root = receipts.tree(seed, plan, ys, leaf_salts).commitment.root
        out.append({"pass": i, "previous": previous.hex(), "seed": seed.hex(), "signs": [r.hex() for r in sk.signs()],
                    "partials": [partials[j].hex() for j in sorted(partials)], "sigmas": [s.hex() for s in sigmas],
                    "receipts": [y.hex() for y in ys], "leaf_salts": [s.hex() for s in leaf_salts], "root": root.hex()})
        previous = root
    return {
        "schema": FIXTURE_SCHEMA,
        "interface": "verity/core/protocols/remote_bandwidth/PROTOCOL.md §11",
        "store": {"toy": "toy.json", "c_star_sha512": doc["c_star_sha512"], "blocks": store.params.blocks,
                  "units": units, "unaudited": list(range(p2.UNIT_BLOCKS * units, store.params.blocks))},
        "registration": {"node": NODE.hex(), "rows": reg.rows, "digest": reg.digest.hex()},
        "challenge": FIXTURE_CHALLENGE.hex(),
        "framing": {
            "seed": "pass i's seed = receipts.seed(challenge, i, previous): previous is the registration digest at "
                    "i = 0 and pass i - 1's root after",
            "signs": "16 rows of 1,028 bytes; bit x of a row (byte x // 8, LSB first) 1 means -1, 0 means +1; block j "
                     "reads bytes 257 (j mod 4) to 257 (j mod 4) + 256 of each row",
            "partial": "block j's: row c's sum over x < 2056 of sign(row c, 2056 (j mod 4) + x) * int8(C*[2056 j + x]), "
                       "16 int32 little-endian, one entry a block in block order",
            "sigma": "unit g's: its blocks 4g .. 4g + 3's partials summed row by row, 16 int32 little-endian",
            "receipt": "unit g's: SHA512(receipts.sketch_prefix(seed) || u64be(g) || sigma)",
            "root": "receipts.tree(seed, Plan.whole(units), receipts, leaf_salts).commitment.root, sixteen receipts a leaf",
        },
        "passes": out,
    }


def fixture_bytes() -> bytes:
    return (json.dumps(fixture(), indent=1) + "\n").encode()


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", nargs="?", choices=("serve", "fixture"), default="serve")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--per-window", type=int, default=3)
    ap.add_argument("--prompt", type=int, default=1)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    if a.command == "fixture":
        data = fixture_bytes()
        if hashlib.sha512(data).hexdigest() != FIXTURE_SHA512:
            print("the fixture moved: update FIXTURE_SHA512 and republish it", file=sys.stderr)
            return 1
        if a.out is None:
            sys.stdout.buffer.write(data)
        else:
            a.out.write_bytes(data)
            print(f"wrote {a.out}", file=sys.stderr)
        return 0
    if a.steps % a.per_window:
        ap.error("--steps is a whole number of windows of --per-window steps")
    store, doc, c_star = toy()
    reg = registration(store, doc, c_star)
    served = serve(Engine(store, c_star, bytes.fromhex(doc["salt"])), p2.Store(reg, blocks_of(store, c_star)),
                   prompt=a.prompt, windows=a.steps // a.per_window, per_window=a.per_window)
    print(json.dumps({"tokens": served.tokens, "passes": served.passes, "outcome": served.outcome,
                      "verdicts": [{"window": v.window, "accepted": v.accepted, "failure": v.failure,
                                    "answered_before_step": n} for v, n in served.verdicts]}))
    return 0 if served.outcome == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
