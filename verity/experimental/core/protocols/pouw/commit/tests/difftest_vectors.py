"""The difftest's data: generated cases and the Python's outcome on each, for `PouwCommit/Difftest.lean`.

    python tests/difftest_vectors.py <verity checkout> --out DIR

Writes `DIR/difftest/cases.jsonl`, one case a line, `{"kind", "in", "python"}` and, where Lean refuses by name what Python
refuses another way or accepts, `"lean"` and `"why"`. The outcome is the checkout's `verity.experimental.protocols.pouw`:
`{"ok": v}`, `{"reason": text}` for a coin `coin_reason` refuses, `{"error": type}` when it raises. The Lean audit runs
`PouwCommit/DifftestMain.lean` on the file (`runs` in `lean-audit.json`, which pins its SHA-256), and
`tests/test_pouw_commit_lean_difftest.py` checks that this script still writes it and that its cases reach every refusal.

The cases: `tagged` over bytes, strings (non-ASCII among them) and integers past 64 bits; the bindings and `weights_root`
over random weight-root maps; domain ids under both frame hashes with bindings, owners and counts out of range; the roots of
random row and tile-digest trees under `sha256`, `sha512` and `hm96-sha512` (rows of several segments, values of the wrong
size, missing or short salt keys); openings of those trees, honest and tampered (value, path, position, root, count, schema,
path and value lengths); and epoch coin openings from a `Challenger`, honest and tampered to hit each of `coin_reason`'s
refusals and each refusal of the JSON forms.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

COUNTS = {"tagged": 1000, "bindings": 500, "domain": 1000, "commit_rows": 1500, "transcript": 500, "verify": 4000, "coins": 100}


def _outcome(f) -> dict:
    try:
        return {"ok": f()}
    except Exception as e:  # noqa: BLE001 - every exception is an outcome, its kind compared
        return {"error": type(e).__name__}


def generate(seed: int = 0) -> list[dict]:
    from verity.experimental.protocols.pouw import audit, coins
    from verity.experimental.protocols.pouw.protocol import tagged
    from verity.ml.commitments import (Commitment, CommitmentDomain, MerkleTree, Opening, RangeIndexedDomain,
                                       VerificationLimits, verify_opening)

    rng = random.Random(seed)
    cases: list[dict] = []

    def add(kind: str, inp: dict, python: dict, lean: dict | None = None, why: str | None = None) -> None:
        case = {"kind": kind, "in": inp, "python": python}
        if lean is not None:
            case |= {"lean": lean, "why": why}
        cases.append(case)

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

    def coin_case(cj: dict, oj: dict, label: str, after: str, lean: dict | None = None, why: str | None = None) -> None:
        def reason():
            r = coins.coin_reason(coins.CoinCommitment.from_json(cj), coins.CoinOpening.from_json(oj), label, bytes.fromhex(after))
            return r
        res = _outcome(reason)
        if "ok" in res and res["ok"] is not None:
            res = {"reason": res["ok"]}
        add("coins", {"commitment": cj, "opening": oj, "label": label, "after": after}, res, lean, why)

    byname = {"error": "refused by name"}
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
        coin_case(cj, oj | {"label": 5}, label, ah, byname, "Lean refuses an opening whose label is not a string; Python compares it")
        coin_case(cj, oj | {"epoch": str(epoch)}, label, ah, byname,
                  "Lean refuses an opening whose epoch is not an integer; Python compares it")
        if epoch in (0, 1):
            coin_case(cj, oj | {"epoch": bool(epoch)}, label, ah, byname,
                      "Lean refuses a boolean epoch, which Python's == takes for the integer")
            coin_case(cj, oj | {"epoch": float(epoch)}, label, ah, byname,
                      "Lean refuses an epoch written as a float, which Python's == takes for the integer")
    return cases


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
