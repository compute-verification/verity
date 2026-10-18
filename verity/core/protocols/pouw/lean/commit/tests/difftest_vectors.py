"""The difftest's data: generated cases and the Python's outcome on each, for `PouwCommit/Difftest.lean`.

    python tests/difftest_vectors.py <verity checkout> --out DIR

Writes `DIR/difftest/cases.jsonl`, one case a line, `{"kind", "in", "python"}`. The outcome is the checkout's
`verity.core.protocols.pouw`: `{"ok": v}`, `{"reason": text}` for a coin `coin_reason` refuses, `{"error": type}` when it
raises. The Lean is PoUW's verifier of record (`pouw.verifier`): the Lean audit runs `PouwCommit/DifftestMain.lean` on the
file (`runs` in `lean-audit.json`, which pins its SHA-256), which fails on any case whose Python outcome is not the
Lean's, and `tests/test_pouw_commit_lean_difftest.py` checks that this script still writes it, that its cases reach every
refusal, and runs the Lean on it.

The cases: `tagged` over bytes, strings (non-ASCII among them) and integers past 64 bits; the bindings and `weights_root`
over random weight-root maps; domain ids under both frame hashes with bindings, owners and counts out of range; the roots of
random row and tile-digest trees under `sha256`, `sha512` and `hm96-sha512` (rows of several segments, values of the wrong
size, missing or short salt keys); openings of those trees, honest and tampered (value, path, position, root, count, schema,
path and value lengths); epoch coin openings from a `Challenger`, honest and tampered to hit each of `coin_reason`'s
refusals and each refusal of the JSON forms; `salt_of` and `Epoch.unverified`'s salt over random coins, seeds (empty among
them), epochs and weights' roots; `Epoch.start`'s salt and `draw_coin`'s coin from honest and tampered openings, and a draw
with no coin commitment; the audit, draw and replay keys and `draw_after`; and `Sampled.select`'s and `pearl_c_work.draw`'s
work-weighted draws, each run as written with only its tickets given (`tickets` patched), over ticket lists with zeros,
totals past 64 bits, none, or all zero. Then the live coins: `layout_problem` and `live_domain` over layouts good and bad;
live coin openings from a `LiveChallenger`, honest and tampered to hit each of `live_reason`'s refusals and each refusal of
the JSON forms; a family's openings from a `LiveRecord`, and a live epoch's rounds, honest and tampered (a coin opened twice,
one opened before its root was recorded, one never opened, swapped, after the wrong root, from a foreign epoch or
commitment, more roots than coins, a commitment that is not a live epoch's); round keys; and the coin modes of schemes.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

COUNTS = {"tagged": 1000, "bindings": 500, "domain": 1000, "commit_rows": 1500, "transcript": 500, "verify": 4000, "coins": 100,
          "salt_of": 400, "unverified_salt": 200, "epoch_salt": 60, "key": 600, "sampled": 300, "pearl_draw": 300,
          "live_layout": 500, "live_coins": 40, "live_order": 60, "rounds": 30, "round_key": 300}


def _outcome(f) -> dict:
    try:
        return {"ok": f()}
    except Exception as e:  # noqa: BLE001 - every exception is an outcome, its kind compared
        return {"error": type(e).__name__}


def generate(seed: int = 0) -> list[dict]:
    from verity.core.protocols.pouw import audit, coins
    from verity.core.protocols.pouw.protocol import DomainError, Matmul, Shape, tagged
    from verity.core.protocols.pouw.schemes import pearl_c_work as pcw
    from verity.ml.randomness import derive
    from verity.ml.commitments import (Commitment, CommitmentDomain, MerkleTree, Opening, RangeIndexedDomain,
                                       VerificationLimits, verify_opening)

    rng = random.Random(seed)
    cases: list[dict] = []

    def add(kind: str, inp: dict, python: dict) -> None:
        cases.append({"kind": kind, "in": inp, "python": python})

    texts = ["", "a", "verity/pouw/x/v1", "é", "中文", "🙂 tag", "line\nbreak", "q'uote\"", "\x01ctl"]

    def big() -> int:
        return rng.choice([0, 1, 255, 256, 2**32, 2**63, 2**64 - 1, 2**64, 2**200 + 7, rng.randrange(2**80)])

    for _ in range(COUNTS["tagged"]):
        parts = []
        for _ in range(rng.randrange(6)):
            k = rng.randrange(3)
            parts.append(rng.randbytes(rng.choice([0, 1, 32, 64, rng.randrange(200)])) if k == 0
                         else rng.choice(texts) if k == 1 else big())
        tag = rng.choice(texts[1:] + ["verity/pouw/weights/v1"])
        add("tagged", {"tag": tag, "parts": [{"b": p.hex()} if isinstance(p, bytes) else {"s": p} if isinstance(p, str)
                                              else {"i": p} for p in parts]},
            _outcome(lambda: tagged(tag, *parts).hex()))

    for _ in range(COUNTS["bindings"]):
        roots = {w: rng.randbytes(rng.choice([32, 64])) for w in rng.sample(range(10**6), rng.randrange(7))}
        if rng.random() < 0.2:
            roots[big()] = rng.randbytes(64)
        salt, w, index = rng.randbytes(rng.choice([32, 32, 64, 0])), big(), rng.choice([0, 1, rng.randrange(10**6), big()])
        epoch = SimpleNamespace(salt=salt, weights_root=audit.weights_root(roots))
        add("bindings", {"weight": w, "index": index, "salt": salt.hex(),
                         "weight_roots": [[k, v.hex()] for k, v in rng.sample(list(roots.items()), len(roots))]},
            _outcome(lambda: {"weight": audit.weight_binding(w).hex(), "activation": audit.activation_binding(salt, index).hex(),
                              "weights_root": epoch.weights_root.hex(), "transcript": audit.transcript_binding(epoch).hex()}))

    def binding() -> bytes:
        return rng.randbytes(rng.choice([64] * 6 + [32] * 2 + [0, 31, 33, 63, 65, 128]))

    def owner() -> int:
        return rng.choice([-1] * 4 + [-2] * 2 + [0, 1, 7, 2**40, -3])

    for _ in range(COUNTS["domain"]):
        b, o, n, h = binding(), owner(), rng.choice([0, 1, 2, 3, rng.randrange(300), 2**33 + 1]), rng.choice(["sha256", "sha512"])
        add("domain", {"binding": b.hex(), "owner": o, "count": n, "hash": h},
            _outcome(lambda: CommitmentDomain(b, o, RangeIndexedDomain(n), hash=h).domain_id.hex()))

    def rows_for(hash: str, digests: bool) -> list[bytes]:
        n = rng.choice([0, 1, 2, 3, 4, 5, rng.randrange(10)])
        if digests:
            return [rng.randbytes(64 if rng.random() < 0.97 else rng.choice([0, 63, 65])) for _ in range(n)]
        return [rng.randbytes(rng.choice([0, 1, 7, 8, 64, 100, 1023, 1024, 1025, 2100] if rng.random() < 0.3 else [rng.randrange(80)]))
                for _ in range(n)]

    def salt_key() -> bytes | None:
        return rng.choice([rng.randbytes(32)] * 8 + [None, rng.randbytes(31)])

    def nonce() -> int:
        return rng.choice([0, 1, 5, 1 << 62, 1 << 63, (1 << 64) - 1, rng.randrange(1 << 64)])

    trees = []
    for _ in range(COUNTS["commit_rows"]):
        h = rng.choice(["sha256", "sha512", "hm96-sha512", "hm96-sha512"])
        digests = rng.random() < 0.4
        b, o, rows, key, nn = binding(), owner(), rows_for(h, digests), salt_key(), nonce()
        res = _outcome(lambda: audit.commit_rows(b, o, rows, h, key, nn, digests).commitment.root.hex())
        add("commit_rows", {"binding": b.hex(), "owner": o, "rows": [r.hex() for r in rows], "hash": h,
                            "salt_key": None if key is None else key.hex(), "nonce": nn, "digests": digests}, res)
        if "ok" in res and rows:
            trees.append((b, o, h, digests, audit.commit_rows(b, o, rows, h, key, nn, digests), audit.ROW_SCHEMA))

    for _ in range(COUNTS["transcript"]):
        h = rng.choice(["sha256", "sha512", "hm96-sha512", "hm96-sha512"])
        salt, wr, key = rng.randbytes(32), rng.randbytes(64), rng.randbytes(rng.choice([32] * 9 + [31]))
        digs = rows_for(h, True)

        def root() -> str:
            values = [d + audit.hm96_salt(key, audit.HM96_TRANSCRIPT, i) for i, d in enumerate(digs)] if h == audit.HM96 else digs
            epoch = SimpleNamespace(salt=salt, weights_root=wr)
            tree = MerkleTree(audit.transcript_domain(epoch, len(values), h), dict(enumerate(values)), lambda _: audit.TILE_SCHEMA)
            return tree.commitment.root.hex()
        add("transcript", {"salt": salt.hex(), "weights_root": wr.hex(), "digests": [d.hex() for d in digs], "hash": h,
                           "salt_key": key.hex()}, _outcome(root))

    def flip(b: bytes) -> bytes:
        if not b:
            return b"\x00"
        i = rng.randrange(len(b))
        return b[:i] + bytes([b[i] ^ (1 << rng.randrange(8))]) + b[i + 1:]

    tampers = ["honest"] * 4 + ["value", "path", "position", "past", "root", "count", "schema", "drop", "extra", "half",
                                "odd", "short", "long", "rootlen", "rootodd"]
    for _ in range(COUNTS["verify"]):
        b, o, h, digests, tree, schema = rng.choice(trees)
        n = tree.domain.count
        pos = rng.randrange(n)
        op = tree.open(pos)
        root, count, value, path = tree.commitment.root, n, op.value, list(op.path)
        t = rng.choice(tampers)
        if t == "value":
            value = flip(value)
        elif t == "path" and path:
            k = rng.randrange(len(path))
            path[k] = flip(path[k])
        elif t == "position":
            pos = rng.randrange(n)
        elif t == "past":
            pos = n + rng.randrange(3)
        elif t == "root":
            root = flip(root)
        elif t == "count":
            count = rng.choice([n + 1, max(0, n - 1), 2 * n, 0])
        elif t == "schema":
            schema = audit.TILE_SCHEMA if schema == audit.ROW_SCHEMA else audit.ROW_SCHEMA
        elif t == "drop" and path:
            path.pop(rng.randrange(len(path)))
        elif t == "extra":
            path.append(rng.randbytes(len(root)))
        elif t == "half" and path:
            path[0] = path[0][:32]
        elif t == "odd" and path:
            path[0] = path[0][:33]
        elif t == "short":
            value = value[:rng.choice([0, 100, 191, max(0, len(value) - 1)])]
        elif t == "long":
            value = value + b"\x00"
        elif t == "rootlen":
            root = root[:32] if len(root) == 64 else root + root
        elif t == "rootodd":
            root = root[:33]

        def check() -> bool:
            dom = audit.rows_domain(b, o, count, h, digests)
            return verify_opening(dom, Commitment(root, count), Opening(pos, value, tuple(path)), schema, VerificationLimits())
        add("verify", {"binding": b.hex(), "owner": o, "count": count, "hash": h, "digests": digests, "schema": schema,
                       "root": root.hex(), "position": pos, "value": value.hex(), "path": [p.hex() for p in path]},
            _outcome(check))

    def coin_case(cj: dict, oj: dict, label: str, after: str) -> None:
        def reason():
            r = coins.coin_reason(coins.CoinCommitment.from_json(cj), coins.CoinOpening.from_json(oj), label, bytes.fromhex(after))
            return r
        res = _outcome(reason)
        if "ok" in res and res["ok"] is not None:
            res = {"reason": res["ok"]}
        add("coins", {"commitment": cj, "opening": oj, "label": label, "after": after}, res)

    for _ in range(COUNTS["coins"]):
        epoch = rng.choice([0, 1, 5, rng.randrange(2**40), 2**70])
        ch = coins.Challenger(epoch, rng.randbytes)
        other = coins.Challenger(epoch, rng.randbytes)
        label = rng.choice(coins.LABELS)
        after = rng.randbytes(64)
        op = ch.open(label, after)
        cj, oj, ah = ch.commitment.to_json(), op.to_json(), after.hex()
        coin_case(cj, oj, label, ah)
        coin_case(cj, oj | {"epoch": epoch + 1}, label, ah)
        coin_case(cj | {"epoch": epoch + 2}, oj, label, ah)
        coin_case(cj, oj | {"epoch": -3}, label, ah)
        coin_case(cj, oj, [lb for lb in coins.LABELS if lb != label][0], ah)
        coin_case(cj, oj | {"label": rng.choice(["Salt", "q'uote", "dq\"", "b'o\"th", "back\\slash", "tab\tnl\n", "\x00\x7f",
                                                  "é", "中", "\x85", "\xa0", "\xad", "🙂", ""])}, label, ah)
        coin_case(cj, oj, "other", ah)
        coin_case(cj, oj | {"coin": rng.randbytes(rng.choice([31, 33, 0])).hex()}, label, ah)
        coin_case(cj, oj | {"salt": rng.randbytes(rng.choice([191, 193])).hex()}, label, ah)
        coin_case(cj, oj | {"coin": flip(op.coin).hex()}, label, ah)
        coin_case(cj, oj | {"salt": flip(op.salt).hex()}, label, ah)
        coin_case(cj, oj | {"path": [flip(op.path[0]).hex()]}, label, ah)
        coin_case(cj, oj | {"path": []}, label, ah)
        coin_case(cj, oj | {"path": [op.path[0].hex(), op.path[0].hex()]}, label, ah)
        coin_case(cj, oj | {"path": [op.path[0][:32].hex()]}, label, ah)
        coin_case(cj, oj | {"path": [op.path[0][:33].hex()]}, label, ah)
        coin_case(cj, oj | {"path": 7}, label, ah)
        coin_case(cj, oj, label, flip(after).hex())
        coin_case(cj | {"root": flip(ch.commitment.root).hex()}, oj, label, ah)
        coin_case(other.commitment.to_json(), oj, label, ah)
        coin_case(cj | {"root": ch.commitment.root[:63].hex()}, oj, label, ah)
        coin_case(cj | {"scheme": "verity/pouw/epoch-coins/v0"}, oj, label, ah)
        coin_case({k: v for k, v in cj.items() if k != "scheme"}, oj, label, ah)
        coin_case(cj | {"epoch": rng.choice([-1, True, 1.5, "1", None])}, oj, label, ah)
        coin_case({k: v for k, v in cj.items() if k != "root"}, oj, label, ah)
        coin_case(cj, {k: v for k, v in oj.items() if k != rng.choice(list(oj))}, label, ah)
        coin_case(cj, oj | {"coin": op.coin.hex().upper()}, label, ah)
        coin_case(cj, oj | {"salt": " ".join(op.salt.hex()[i:i + 2] for i in range(0, 384, 2)) + "\n"}, label, ah)
        coin_case(cj, oj | {"after": ah + "0"}, label, ah)
        coin_case(cj, oj | {"coin": "zz" + op.coin.hex()[2:]}, label, ah)
        coin_case(cj | {"root": 5}, oj, label, ah)
        coin_case(cj, oj | {"label": 5}, label, ah)
        coin_case(cj, oj | {"epoch": str(epoch)}, label, ah)
        if epoch in (0, 1):
            coin_case(cj, oj | {"epoch": bool(epoch)}, label, ah)
            coin_case(cj, oj | {"epoch": float(epoch)}, label, ah)

    def roots_map() -> dict[int, bytes]:
        return {w: rng.randbytes(rng.choice([32, 64])) for w in rng.sample(range(10**6), rng.randrange(5))}

    def size() -> int:
        return rng.choice([32] * 6 + [64] * 3 + [0, 1, 31, 33, 200])

    for _ in range(COUNTS["salt_of"]):
        cj = {"scheme": coins.SCHEME, "epoch": rng.choice([0, 1, 5, rng.randrange(2**40), 2**70]), "root": rng.randbytes(64).hex()}
        coin, wr = rng.randbytes(size()), rng.randbytes(size())
        add("salt_of", {"commitment": cj, "coin": coin.hex(), "weights_root": wr.hex()},
            _outcome(lambda: audit.salt_of(coins.CoinCommitment.from_json(cj), coin, wr).hex()))

    for _ in range(COUNTS["unverified_salt"]):
        roots, seed = roots_map(), rng.randbytes(size())

        def unverified() -> str:
            wl = audit.Workload(tuple(Matmul(i, Shape(1, 1, 1), w) for i, w in enumerate(roots)))
            return audit.Epoch.unverified(wl, roots, seed).salt.hex()
        add("unverified_salt", {"seed": seed.hex(), "weight_roots": [[k, v.hex()] for k, v in roots.items()]}, _outcome(unverified))

    def refusable(f) -> dict:
        res = _outcome(f)
        return res["ok"] if "ok" in res else res

    def epoch_salt_case(cj: dict, oj: dict, roots: dict[int, bytes]) -> None:
        def start() -> dict:
            c, o = coins.CoinCommitment.from_json(cj), coins.CoinOpening.from_json(oj)
            wl = audit.Workload(tuple(Matmul(i, Shape(1, 1, 1), w) for i, w in enumerate(roots)))
            try:
                return {"ok": audit.Epoch.start(wl, roots, c, o).salt.hex()}
            except DomainError as e:
                if str(e).startswith("the salt's "):
                    return {"reason": str(e)}
                raise
        add("epoch_salt", {"commitment": cj, "opening": oj, "weight_roots": [[k, v.hex()] for k, v in roots.items()]},
            refusable(start))

    def draw_coin_case(cj: dict | None, oj: dict, after: bytes) -> None:
        def draw() -> dict:
            c = None if cj is None else coins.CoinCommitment.from_json(cj)
            o = coins.CoinOpening.from_json(oj)
            coin, why = audit.draw_coin(SimpleNamespace(coins=c), SimpleNamespace(open=lambda _label, _after: o), after)
            return {"ok": coin.hex()} if why is None else {"reason": why}
        add("draw_coin", {"commitment": cj, "opening": oj, "after": after.hex()}, refusable(draw))

    for _ in range(COUNTS["epoch_salt"]):
        epoch = rng.choice([0, 1, 5, rng.randrange(2**40), 2**70])
        ch, other = coins.Challenger(epoch, rng.randbytes), coins.Challenger(epoch, rng.randbytes)
        roots = roots_map()
        wr = audit.weights_root(roots)
        cj, salt_op, draw_op = ch.commitment.to_json(), ch.open("salt", wr).to_json(), ch.open("draw", wr).to_json()
        other_salt, other_draw = other.open("salt", wr).to_json(), other.open("draw", wr).to_json()
        epoch_salt_case(cj, salt_op, roots)
        epoch_salt_case(cj, draw_op, roots)
        epoch_salt_case(cj, salt_op | {"after": rng.randbytes(64).hex()}, roots)
        epoch_salt_case(cj, other_salt, roots)
        epoch_salt_case(cj, salt_op | {"epoch": epoch + 1}, roots)
        epoch_salt_case(cj, salt_op | {"coin": flip(bytes.fromhex(salt_op["coin"])).hex()}, roots)
        epoch_salt_case(cj, salt_op | {"path": [rng.randbytes(33).hex()]}, roots)
        epoch_salt_case(cj, salt_op, roots | {10**6 + 1: rng.randbytes(64)})
        draw_coin_case(cj, draw_op, wr)
        draw_coin_case(None, draw_op, wr)
        draw_coin_case(cj, draw_op, flip(wr))
        draw_coin_case(cj, salt_op, wr)
        draw_coin_case(cj, other_draw, wr)
        draw_coin_case(other.commitment.to_json(), other_draw, wr)
        draw_coin_case(cj, draw_op | {"salt": flip(bytes.fromhex(draw_op["salt"])).hex()}, wr)
        draw_coin_case(cj, draw_op | {"path": [rng.randbytes(65).hex()]}, wr)

    for _ in range(COUNTS["key"]):
        which = rng.choice(["audit", "draw", "replay", "after"])
        source, tr, rd, salt = rng.randbytes(size()), rng.randbytes(64), rng.randbytes(rng.choice([64, 32, 0])), rng.randbytes(size())
        rnd = rng.choice([0, 1, rng.randrange(2**32), 2**64 + 3])

        def key() -> str:
            if which == "audit":
                return derive(source, audit.AUDIT_DOMAIN, {"transcript": tr, "salt": salt}).value.hex()
            if which == "draw":
                return pcw.draw_key(source, tr, rd, salt).value.hex()
            if which == "replay":
                return pcw.replay_key(pcw.RecordedDraw(rnd, source), tr, rd, salt).value.hex()
            return pcw.draw_after(tr, rd).hex()
        add("key", {"which": which, "source": source.hex(), "transcript": tr.hex(), "records": rd.hex(), "salt": salt.hex(),
                    "round": rnd}, _outcome(key))

    def tickets() -> list[int]:
        n = rng.choice([0, 1, 2, 3, 7, 20, rng.randrange(60)])
        shape = rng.choice(["small", "zeros", "big", "allzero"])
        if shape == "allzero":
            return [0] * n
        if shape == "big":
            return [rng.choice([0, rng.randrange(1, 2**70)]) for _ in range(n)]
        return [rng.choice([0, 1, 2, 3, 1000] if shape == "zeros" else [1, 2, 7, rng.randrange(1, 10**6)]) for _ in range(n)]

    def t_draws() -> int:
        return rng.choice([0, 1, 2, 5, 32, rng.randrange(100)])

    for _ in range(COUNTS["sampled"]):
        source, tr, salt, tk, t = rng.randbytes(rng.choice([32] * 12 + [0])), rng.randbytes(64), rng.randbytes(32), tickets(), t_draws()

        def sampled() -> list[int]:
            with mock.patch.object(audit, "tickets", lambda _scheme, _workload, _refs: list(tk)):
                epoch = SimpleNamespace(salt=salt, workload=None)
                return list(audit.Sampled(t).select(None, epoch, SimpleNamespace(commitment=SimpleNamespace(root=tr)), source, [], None))
        add("sampled", {"source": source.hex(), "transcript": tr.hex(), "salt": salt.hex(), "tickets": tk, "t": t}, _outcome(sampled))

    for _ in range(COUNTS["pearl_draw"]):
        source, tr, rd, salt = rng.randbytes(rng.choice([32] * 12 + [0])), rng.randbytes(64), rng.randbytes(64), rng.randbytes(32)
        tk, t = tickets(), t_draws()

        def pearl() -> list[int]:
            with mock.patch.object(pcw, "tickets", lambda _scheme, _records: [(i, None, w) for i, w in enumerate(tk)]):
                return [i for i, _ in pcw.draw(None, [], pcw.draw_key(source, tr, rd, salt), t)]
        add("pearl_draw", {"source": source.hex(), "transcript": tr.hex(), "records": rd.hex(), "salt": salt.hex(), "tickets": tk,
                           "t": t}, _outcome(pearl))

    _live_cases(rng, add, flip)
    return cases


def _live_cases(rng, add, flip) -> None:
    """The live coins' cases: layouts, single openings, a family's openings in order, a live epoch's rounds, round keys and
    coin modes."""
    from verity.core.protocols.pouw import coins

    def reasoned(f) -> dict:
        res = _outcome(f)
        return {"reason": res["ok"]} if res.get("ok") is not None else res

    def refusable(f) -> dict:
        res = _outcome(f)
        return res["ok"] if "ok" in res else res

    names = ["salt", "draw", "round", "commit", "a", "x-1", "r2", "0", "-"]
    schemes = [coins.LIVE_SCHEME, coins.SCHEME, "verity/vllm/commit-coins/v1", "a", "/", "a//b"]

    def epoch() -> int:
        return rng.choice([0, 1, 5, rng.randrange(2**40), 2**70])

    def families() -> list[list]:
        return [[lb, rng.choice([1, 1, 2, 3, 5, rng.randrange(1, 12)])] for lb in rng.sample(names, rng.choice([1, 1, 2, 3, 4]))]

    def layout_case(scheme, ep, fams) -> None:
        def run():
            fs = tuple(tuple(f) for f in fams) if type(fams) is list and all(type(f) is list for f in fams) else fams
            why = coins.layout_problem(scheme, ep, fs)
            return {"reason": why} if why is not None else {"ok": coins.live_domain(scheme, ep, fs).domain_id.hex()}
        add("live_layout", {"scheme": scheme, "epoch": ep, "families": fams}, refusable(run))

    bad_families = [[], "x", 5, None, {"a": 1}, [["a"]], [["a", 1, 2]], [["a", 0]], [["a", -1]], [["a", True]], [["a", 1.5]],
                    [["a", "1"]], [[5, 1]], [["A", 1]], [["a/b", 1]], [["", 1]], [["a b", 1]], [["é", 1]], [["a", 1], "b"],
                    [["a", 1], ["a", 2]], [["a", coins.MAX_COINS], ["b", 1]], [["a", coins.MAX_COINS]], [["a", 2**70]],
                    [["a", coins.MAX_COINS - 1], ["b", 1]]]
    for _ in range(COUNTS["live_layout"]):
        scheme = rng.choice(schemes) if rng.random() < 0.85 else rng.choice(["", "Verity/x", "a b", "é", "a_b", 5, None])
        ep = epoch() if rng.random() < 0.85 else rng.choice([-1, True, 1.5, "1", None])
        fams = families() if rng.random() < 0.7 else rng.choice(bad_families)
        layout_case(scheme, ep, fams)

    def live_case(cj: dict, oj: dict, label: str, index: int, after: str) -> None:
        def reason():
            return coins.live_reason(coins.LiveCommitment.from_json(cj), coins.LiveOpening.from_json(oj), label, index,
                                     bytes.fromhex(after))
        add("live_coins", {"commitment": cj, "opening": oj, "label": label, "index": index, "after": after}, reasoned(reason))

    for _ in range(COUNTS["live_coins"]):
        scheme, ep, fams = rng.choice(schemes[:4]), epoch(), families()
        ch, other = (coins.LiveChallenger(scheme, ep, [tuple(f) for f in fams], rng.randbytes) for _ in range(2))
        label, n = rng.choice(fams)
        index, after = rng.randrange(n), rng.randbytes(64)
        op = ch.open(label, index, after)
        cj, oj, ah = ch.commitment.to_json(), op.to_json(), after.hex()

        def case(c: dict = cj, o: dict = oj, lb: str = label, i: int = index, a: str = ah) -> None:
            live_case(c, o, lb, i, a)
        case()
        case(o=oj | {"epoch": ep + 1})
        case(c=cj | {"epoch": ep + 2})
        case(o=oj | {"epoch": -3})
        for lb, k in fams:
            if lb != label:
                case(lb=lb, i=rng.randrange(k))
        if n > 1:
            case(i=(index + 1) % n)
        case(i=n)
        case(i=-1)
        case(lb="nope")
        case(o=oj | {"index": rng.choice([index + 1, -1, 2**70])})
        case(o=oj | {"label": rng.choice(["Round", "q'uote", "dq\"", "back\\slash", "tab\tnl\n", "\x00\x7f", "é", "中", "\x85",
                                          "\xa0", "🙂", ""])})
        case(o=oj | {"coin": rng.randbytes(rng.choice([31, 33, 0])).hex()})
        case(o=oj | {"salt": rng.randbytes(rng.choice([191, 193])).hex()})
        case(o=oj | {"coin": flip(op.coin).hex()})
        case(o=oj | {"salt": flip(op.salt).hex()})
        if op.path:
            case(o=oj | {"path": [flip(op.path[0]).hex()] + oj["path"][1:]})
            case(o=oj | {"path": oj["path"][1:]})
            case(o=oj | {"path": [op.path[0][:32].hex()] + oj["path"][1:]})
            case(o=oj | {"path": [op.path[0][:33].hex()] + oj["path"][1:]})
        case(o=oj | {"path": oj["path"] + [rng.randbytes(64).hex()]})
        case(o=oj | {"path": 7})
        case(a=flip(after).hex())
        case(c=cj | {"root": flip(ch.commitment.root).hex()})
        case(c=other.commitment.to_json())
        case(o=other.open(label, index, after).to_json())
        case(c=cj | {"root": ch.commitment.root[:63].hex()})
        case(c=cj | {"scheme": rng.choice([s for s in schemes if s != scheme])})
        case(c=cj | {"scheme": rng.choice(["Verity/x", "", 5])})
        case(c=cj | {"families": [[lb, k + 1] if lb == label else [lb, k] for lb, k in fams]})
        if len(fams) > 1:
            case(c=cj | {"families": fams[::-1]})
        case(c=cj | {"families": "x"})
        case(c=cj | {"families": fams + ["b"]})
        case(c=cj | {"families": [[label, n], [label, n]]})
        case(c=cj | {"epoch": rng.choice([-1, True, 1.5, "1", None])})
        case(c={k: v for k, v in cj.items() if k != rng.choice(list(cj))})
        case(o={k: v for k, v in oj.items() if k != rng.choice(list(oj))})
        case(o=oj | {"coin": op.coin.hex().upper()})
        case(o=oj | {"salt": " ".join(op.salt.hex()[i:i + 2] for i in range(0, 384, 2)) + "\n"})
        case(o=oj | {"after": ah + "0"})
        case(o=oj | {"coin": "zz" + op.coin.hex()[2:]})
        case(c=cj | {"root": 5})
        case(o=oj | {"label": 5})
        case(o=oj | {"index": rng.choice([str(index), 1.5, None])})
        case(o=oj | {"epoch": str(ep)})
        if ep in (0, 1):
            case(o=oj | {"epoch": bool(ep)})
            case(o=oj | {"epoch": float(ep)})
        if index in (0, 1):
            case(o=oj | {"index": bool(index)})

    def order_case(kind: str, cj: dict, label: str, roots: list[str], ops: list[dict]) -> None:
        def reason():
            c = coins.LiveCommitment.from_json(cj)
            rs, os_ = [bytes.fromhex(r) for r in roots], [coins.LiveOpening.from_json(o) for o in ops]
            return coins.order_reason(c, label, rs, os_) if kind == "live_order" else coins.rounds_reason(c, rs, os_)
        inp = {"commitment": cj, "roots": roots, "openings": ops} | ({"label": label} if kind == "live_order" else {})
        add(kind, inp, reasoned(reason))

    def record(ch, label: str, k: int) -> tuple[list[str], list[dict]]:
        rec = coins.LiveRecord(ch, label)
        for _ in range(k):
            rec.record(rng.randbytes(rng.choice([64] * 5 + [32])))
        return [r.hex() for r in rec.roots], [o.to_json() for o in rec.openings]

    def tampered(kind: str, ch, label: str, n: int, roots: list[str], ops: list[dict], foreign) -> None:
        cj = ch.commitment.to_json()

        def case(rs: list[str] = roots, os_: list[dict] = ops, c: dict = cj, lb: str = label) -> None:
            order_case(kind, c, lb, rs, os_)
        k = len(roots)
        case()
        if ops:
            j = rng.randrange(k)
            case(os_=ops[:j + 1] + [ops[j]] + ops[j + 1:])
            case(os_=ops[:-1])
            case(os_=ops[:j] + [ops[j] | {"after": flip(bytes.fromhex(ops[j]["after"])).hex()}] + ops[j + 1:])
            case(rs=roots[:j] + [flip(bytes.fromhex(roots[j])).hex()] + roots[j + 1:])
            case(os_=ops[:j] + [ops[j] | {"coin": ops[j]["coin"][:-2]}] + ops[j + 1:])
            case(os_=ops[:j] + [ops[j] | {"index": "0"}] + ops[j + 1:])
            case(rs=roots[:-1])
            case(os_=ops + [ops[0]])
        if k > 1:
            case(os_=[ops[1], ops[0]] + ops[2:])
        if k < n:
            early = ch.open(label, k, bytes.fromhex(roots[-1]) if roots else rng.randbytes(64))
            case(os_=ops + [early.to_json()])
        case(rs=roots + [rng.randbytes(64).hex() for _ in range(n + 1 - k)])
        frs, fops = foreign
        case(rs=frs, os_=fops)
        case(rs=roots[:1], os_=[])
        case(rs=[], os_=[])
        case(rs=roots + ["zz"])
        if kind == "live_order":
            case(lb="nope")
            for lb, _ in ch.commitment.families:
                if lb != label:
                    case(lb=lb)

    for _ in range(COUNTS["live_order"]):
        scheme, ep, fams = rng.choice(schemes[:4]), epoch(), families()
        label, n = rng.choice(fams)
        layout = [tuple(f) for f in fams]
        ch = coins.LiveChallenger(scheme, ep, layout, rng.randbytes)
        k = rng.randrange(n + 1)
        roots, ops = record(ch, label, k)
        kk = max(1, k)
        foreign = rng.choice([coins.LiveChallenger(scheme, ep + 1, layout, rng.randbytes),
                              coins.LiveChallenger(scheme, ep, layout, rng.randbytes)])
        tampered("live_order", ch, label, n, roots, ops, record(foreign, label, kk))

    for _ in range(COUNTS["rounds"]):
        ep, rounds = epoch(), rng.choice([1, 2, 3, 4, 6])
        ch = coins.live_challenger(ep, rounds, rng.randbytes)
        k = rng.randrange(rounds + 1)
        roots, ops = record(ch, coins.ROUND, k)
        ch.open("salt", 0, rng.randbytes(64))
        foreign = coins.live_challenger(rng.choice([ep, ep + 1]), rounds, rng.randbytes)
        tampered("rounds", ch, coins.ROUND, rounds, roots, ops, record(foreign, coins.ROUND, max(1, k)))
        for scheme, layout in ((coins.LIVE_SCHEME, (("draw", 1), ("salt", 1), (coins.ROUND, rounds))),
                               (coins.LIVE_SCHEME, (("salt", 1), ("draw", 1), ("rounds", rounds))),
                               (coins.LIVE_SCHEME, (("salt", 1), ("draw", 2), (coins.ROUND, rounds))),
                               (coins.LIVE_SCHEME, (("salt", 1), (coins.ROUND, rounds))),
                               ("verity/vllm/commit-coins/v1", coins.live_families(rounds))):
            odd = coins.LiveChallenger(scheme, ep, layout, rng.randbytes)
            if odd.commitment.family(coins.ROUND) is not None:
                rs, os_ = record(odd, coins.ROUND, 1)
            else:
                rs, os_ = [], []
            order_case("rounds", odd.commitment.to_json(), coins.ROUND, rs, os_)
        salt_ch = coins.live_challenger(ep, rounds, rng.randbytes)
        root = rng.randbytes(64)
        order_case("rounds", salt_ch.commitment.to_json(), coins.ROUND, [root.hex()],
                   [salt_ch.open("salt", 0, root).to_json()])

    for _ in range(COUNTS["round_key"]):
        cj = {"epoch": epoch(), "families": families(), "root": rng.randbytes(64).hex(),
              "scheme": rng.choice([coins.LIVE_SCHEME] * 3 + schemes)}
        coin, root = rng.randbytes(rng.choice([32] * 8 + [0, 1, 64])), rng.randbytes(rng.choice([64] * 6 + [32, 0]))
        r = rng.choice([0, 1, 2, rng.randrange(2**32), 2**64 + 3])
        add("round_key", {"commitment": cj, "coin": coin.hex(), "round": r, "root": root.hex()},
            _outcome(lambda: coins.round_key(coins.LiveCommitment.from_json(cj), coin, r, root).value.hex()))

    for scheme in [coins.SCHEME, coins.LIVE_SCHEME, "verity/pouw/epoch-coins/v2", "verity/pouw/epoch-coins/live/v2", "",
                   "verity/vllm/commit-coins/v1", coins.SCHEME + " ", coins.LIVE_SCHEME.upper()]:
        add("mode", {"scheme": scheme}, _outcome(lambda: coins.mode(scheme)))


def write(out: Path, seed: int = 0) -> Path:
    path = out / "difftest" / "cases.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(c, ensure_ascii=False, separators=(",", ":")) + "\n" for c in generate(seed)))
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("repo", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(args.repo))
    path = write(args.out)
    print(f"{sum(1 for _ in path.open())} cases -> {path}")


if __name__ == "__main__":
    main()
