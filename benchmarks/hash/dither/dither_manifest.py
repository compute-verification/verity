"""A dither-a-so -we window's manifest (`dither_so.SoWindow.end`): what an ended window binds, the manifest.json its end
writes, and the verifier's check of it against the challenger's own run (`coins.check`).

An ended window binds, in order: the epoch round (round 0, the tree over the B tables' roots), each step s's round (its
root the tree over the round's leaves, each leaf the tree over that kind's per-call roots: a-step, admit-step, z-step,
y-step and receipt-step, the last holding step s - D + 1's receipt-table roots), the final round (the tree over the
tail's receipt-tail roots, each the tree over one tail step's receipt-table roots, steps S - D + 1 to S - 1), and the
manifest root, the tree over every round root of the window in order, which the draw coin follows.  manifest.json holds
the window, every round's root, each step's round leaves and unit salt, the B roots, the final round's leaves, the
manifest root and the coins in `coins.check`'s form; each step's per-call roots go to a file beside it (`calls`, by
sha512).  `check` recomputes every round root from the manifest's leaves (the epoch round from the B roots, the final
round from the tail's leaves), the manifest root from those, and holds the challenger's run to them: the salt coin after
the window's root, each round's coin after its root, the draw coin after the manifest root, the session run from a
commit the host's pins log (``--pins``) records on the verifier's own first-parent history of main (``--main``).

    python dither_manifest.py check --manifest OUT/manifest-SESSION.json --challenger-out SESSION_DIR --ledger HEAD [...]
        --pins PINS --main FIRST_PARENT [--key-record KEY] [--period P] [--root R]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import dither_one as DO
from dither_fused import CM, CO

SCHEMA = "verity/dither-so/window-manifest/v1"
TAIL_KIND, FINAL_KIND, LATE_KIND, MANIFEST_KIND = "receipt-tail", "round-final", "round-late", "window-manifest"
CALLS_LAYOUT = ("per step, int64 words (h1c.words, each 64-byte root eight big-endian u64): each round kind's per-call "
                "roots in linear order, kinds in the round's order, then the round's leaves")


def final_fields(window: str, tail: list[int]) -> dict:
    return {"window": window, "first": tail[0], "steps": len(tail), "leaves": f"{TAIL_KIND} roots"}


def manifest_fields(window: str, k: int, root: bytes, s0: int, s1: int) -> dict:
    return {"window": window, "index": k, "root": root.hex(), "first": s0, "steps": s1 - s0,
            "rounds": "epoch round, each step's round, the final round"}


def epoch_root(window: str, b_roots: list[bytes]) -> bytes:
    return CM.step_root(CM.domain({"window": window, "linears": len(b_roots)}, len(b_roots), "b-step"), b_roots)


def step_leaf(window: str, s: int, kind: str, roots: list[bytes]) -> bytes:
    return CM.step_root(CM.domain({"window": window, "step": s, "linears": len(roots)}, len(roots), kind), roots)


def round_root(window: str, s: int, kinds: tuple | list, leaves: list[bytes]) -> bytes:
    return CM.step_root(CM.domain({"window": window, "step": s, "leaves": DO.round_leaves(kinds)}, len(kinds), "round"),
                        leaves)


def tail_leaf(window: str, j: int, roots: list[bytes]) -> bytes:
    return step_leaf(window, j, TAIL_KIND, roots)


def final_root(window: str, tail: list[int], leaves: list[bytes]) -> bytes:
    return CM.step_root(CM.domain(final_fields(window, tail), len(tail), FINAL_KIND), leaves)


def late_root(window: str, j: int, leaf: bytes) -> bytes:
    return CM.step_root(CM.domain(final_fields(window, [j]), 1, LATE_KIND), [leaf])


def manifest_root(window: str, k: int, root: bytes, s0: int, s1: int, rounds: list[bytes]) -> bytes:
    return CM.step_root(CM.domain(manifest_fields(window, k, root, s0, s1), len(rounds), MANIFEST_KIND), rounds)


def write(out_dir: str | Path, name: str, doc: dict, calls: bytes, calls_dir: str | Path) -> dict:
    """`doc` as `out_dir`/manifest-`name`.json, its per-call roots `calls` in `calls_dir` named by their sha512
    (`doc["calls"]` their file, size and digest); where both went."""
    digest = hashlib.sha512(calls).hexdigest()
    cpath = Path(calls_dir) / f"dither-so-calls-{digest[:32]}.bin"
    cpath.write_bytes(calls)
    doc["calls"] = {"file": str(cpath), "bytes": len(calls), "sha512": digest, "layout": CALLS_LAYOUT}
    out = Path(out_dir) / f"manifest-{name}.json"
    out.write_text(json.dumps(doc, indent=1) + "\n")
    return {"manifest": str(out), "calls": str(cpath), "calls_sha512": digest[:32]}


def expect(m: dict) -> dict:
    """What the verifier recomputes from manifest `m` alone: every round root (the epoch round from the B roots, each
    step's from its leaves, the final from the tail's leaves), the manifest root over them (the draw coin's root), the
    unit salts as the manifest lists them, and where the recomputed roots differ from the manifest's own."""
    W, kinds, s0 = m["window"], m["round_kinds"], m["first"]
    rounds = [epoch_root(W, [bytes.fromhex(x) for x in m["b_roots"]])]
    rounds += [round_root(W, s0 + r, kinds, [bytes.fromhex(x) for x in lv]) for r, lv in enumerate(m["round_leaves"])]
    f = m["final"]
    rounds.append(final_root(W, f["tail"], [bytes.fromhex(x) for x in f["leaves"]]))
    root = bytes.fromhex(m["open_root"])
    mroot = manifest_root(W, m["index"], root, s0, s0 + m["steps"], rounds)
    differ = [i for i, (a, b) in enumerate(zip(rounds, m["rounds"])) if a.hex() != b]
    return {"rounds": rounds, "salts": {int(k): v for k, v in m["salts"].items()}, "draw_after": mroot,
            "differ": differ + ([] if len(rounds) == len(m["rounds"]) else ["count"]),
            "manifest_root_listed": mroot.hex() == m["manifest_root"]}


def check(manifest: str | Path, challenger_out: str | Path, heads: list[str], pins: list[str], main: list[str], key=None,
          period: str | None = None, root: bytes | None = None) -> dict:
    """`coins.check` of the window's coins against the challenger's run, over the round roots and manifest root `expect`
    recomputes, with the host's pins log `pins` (its lines) and the verifier's first-parent history of main `main`;
    ``ok`` only when it passes and the recomputed roots are the manifest's."""
    m = json.loads(Path(manifest).read_text())
    e = expect(m)
    got = CO.check(challenger_out, m["coins"], heads, pins, main, e["rounds"], e["salts"], e["draw_after"], key, period,
                   root)
    got["recomputed"] = {"rounds": len(e["rounds"]), "differ_from_listed": e["differ"][:8],
                         "manifest_root_listed": e["manifest_root_listed"], "steps": m["steps"], "salts": len(e["salts"])}
    got["ok"] = bool(got["ok"] and not e["differ"] and e["manifest_root_listed"])
    return got


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="an ended window's manifest against the challenger's own run")
    c.add_argument("--manifest", required=True)
    c.add_argument("--challenger-out", required=True, help="the session as the challenger's gate publishes it, or its out/")
    c.add_argument("--ledger", action="append", required=True, help="a head of the auditor's ledger, once for each")
    c.add_argument("--pins", type=Path, required=True, help="the challenger host's pins log as infra published it, a "
                   "line a pin (its `challenger-pins head` label)")
    c.add_argument("--main", type=Path, required=True, help="the verifier's own `git rev-list --first-parent "
                   "origin/main`, a commit a line")
    c.add_argument("--key-record", help="the challenger's key record from its deployment (challenger-key/v1)")
    c.add_argument("--period", help="the window's period, as the verifier knows it")
    c.add_argument("--root", help="the window's root as the verifier recomputes it (128 hex digits)")
    a = ap.parse_args(argv)
    key = CO.key_record(a.key_record) if a.key_record else None
    pins = [x for x in a.pins.read_text().split("\n") if x]
    got = check(a.manifest, a.challenger_out, a.ledger, pins, CO.R.main_lines(a.main), key, a.period,
                bytes.fromhex(a.root) if a.root else None)
    print(json.dumps(got, indent=1))
    return 0 if got["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
