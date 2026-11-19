"""Whether Pearl+'s device trees are the credited block's port trees, from retained passes on a processor, both ways.

  port_trees.py PASS_DIR --out FILE [--call C] [--round S] [--program P] [--window R]
  port_trees.py --reuse W PASS_DIR... --out FILE

The first rebuilds one call's A and y trees and one weight's B tree from the pass's rows and salts as the device builds
them (`audit.Hm96Domain`'s leaves, `merkle.tree_levels`): under Pearl+'s own domain each must give the manifest's root,
and under the registered value's own (`registered_value.domain`: A's rows as `LinearARow_v3`'s output
``pearl-a/<s>/<c>``, y as ``pearl-y/<s>/<c>``, B as ``pearl-b/<w>``, 32-bit words, ``row-seg`` rows) the same leaves must
give `registered_value.Registered.commit`'s root, the device's leaf and tree being the port's but for the 64-byte domain
id, which the device takes as data. ``--program`` and ``--window`` (the Program's digest and the window's open root)
enter only that id; without them it takes stand-ins and says so. ``--call`` defaults to the call of the fewest
multiply-adds, B being its weight. The second runs `pearl_c.window.reuse` over the passes of the window record W: every
call's A rows and y as a registered value under the call's own domain with the pass's kept salts, which must be W's,
so the device's roots, unchanged.

Writes roots, booleans and `reuse`'s refusal only (a root is a commitment), and exits non-zero unless all hold. The
check exists only while the device's trees take Pearl+'s own domains: the change that settles the domain deletes it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from verity.commitments import registered_value as RV
from verity.commitments.merkle import tree_levels
from verity.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_SEG as SEG
from verity.protocols.resource_accounting.compute.pouw.prover.serving.pearl_c import served_y as SY
from verity.protocols.resource_accounting.compute.pouw.prover.serving.pearl_c import window as W
from verity.protocols.resource_accounting.compute.pouw.prover.serving.pearl_c.window import a_value, b_value, y_value
from verity.protocols.resource_accounting.compute.pouw.verifier.audit import (HM96, HM96_SALT, Hm96Domain,
                                                                             activation_binding, rows_domain, weight_binding)

STAND_IN_PROGRAM = hashlib.sha512(b"port-trees/stand-in-program").hexdigest()
STAND_IN_WINDOW = hashlib.sha512(b"port-trees/stand-in-window").hexdigest()


def _file(base: Path, spec: dict) -> bytes:
    data = (base / spec["path"]).read_bytes()
    if len(data) != spec["bytes"]:
        raise ValueError(f"{spec['path']}: {len(data)} bytes, the manifest says {spec['bytes']}")
    return data


def _f32_rows(bf16: bytes, words: int, count: int) -> list[bytes]:
    rows = SY.f32_rows(bf16, words)
    if len(rows) != count:
        raise ValueError(f"{len(rows)} rows of {words} words where the manifest commits {count}")
    return rows


def _salts(data: bytes, count: int) -> list[bytes]:
    if len(data) != HM96_SALT * count:
        raise ValueError(f"{len(data)} bytes of salts for {count} rows of {HM96_SALT}")
    return [data[HM96_SALT * i:HM96_SALT * (i + 1)] for i in range(count)]


def device_root(dom, rows: list[bytes], salts: list[bytes]) -> bytes:
    """The root of `rows` under `salts` as the device's tree launches make it, in `dom`'s domain id."""
    d = Hm96Domain(dom.binding, dom.owner, dom.positions, hash="sha512")
    return tree_levels(d, [d.leaf(i, i, SEG, r + s) for i, (r, s) in enumerate(zip(rows, salts))])[-1][0]


def port_root(program: str, value: str, rows: list[bytes], salts: list[bytes], window: str | None) -> tuple[bytes, object]:
    words = [tuple(np.frombuffer(r, "<u4").tolist()) for r in rows]
    reg = RV.Registered.commit(program, value, words, salts, word_bits=32, schema=SEG, window=window)
    return reg.root, reg.domain


def _tree(pearl_dom, want: str, program: str, value: str, rows, salts, window) -> dict:
    pearl = device_root(pearl_dom, rows, salts)
    registered, dom = port_root(program, value, rows, salts, window)
    under_port = device_root(dom, rows, salts)
    return {"value": value, "leaves": len(rows), "pearl_root": pearl.hex(), "manifest_root": want,
            "device_under_port_root": under_port.hex(), "registered_root": registered.hex(),
            "pearl_is_manifest": pearl.hex() == want, "device_is_port": under_port == registered}


def inputs(base: Path, call: int | None = None, round_: int = 0, program: str | None = None,
           window: str | None = None) -> tuple[dict, dict]:
    """The call's facts and, for A, y and B, `_tree`'s arguments."""
    man = json.loads((base / "manifest.json").read_text())
    mms = man["matmuls"]
    c = min(range(len(mms)), key=lambda i: (mms[i]["n"] * mms[i]["k"], i)) if call is None else call
    u, salt = mms[c], bytes.fromhex(man["salt"])
    index, k, n, m, real, w = u["index"], u["k"], u["n"], u["m"], u["real_rows"], u["weight"]
    wt = man["weights"][str(w)]
    prog = program or STAND_IN_PROGRAM
    win = window or STAND_IN_WINDOW
    args = {"a": (rows_domain(activation_binding(salt, index), index, m, HM96), u["root_a"], prog, a_value(round_, c),
                  _f32_rows(_file(base, u["rows"]), k, m), _salts(_file(base, u["salts"]), m), win),
            "y": (SY.y_domain(salt, index, real), u["y"]["root"], prog, y_value(round_, c),
                  _f32_rows(_file(base, u["y"]["rows"]), n, real), _salts(_file(base, u["y"]["salts"]), real), win),
            "b": (rows_domain(weight_binding(w), -2, n, HM96), bytes.fromhex(wt["root"]).hex(), prog, b_value(w),
                  _f32_rows(_file(base, wt["rows"]), k, n), _salts(_file(base, wt["salts"]), n), None)}
    return ({"call": c, "index": index, "k": k, "n": n, "m16": m, "real_rows": real, "weight": w, "round": round_,
             "program": prog, "window": win, "stand_in": {"program": program is None, "window": window is None}}, args)


def check(base: Path, call: int | None = None, round_: int = 0, program: str | None = None,
          window: str | None = None) -> dict:
    facts, args = inputs(base, call, round_, program, window)
    trees = {t: _tree(*a) for t, a in args.items()}
    return {**facts, "trees": trees, "ok": all(t["pearl_is_manifest"] and t["device_is_port"] for t in trees.values())}


def reuse(w_path: Path, dirs: list[Path]) -> dict:
    """`pearl_c.window.reuse` over W's passes, ``dirs`` in round order."""
    w = json.loads(w_path.read_text())
    mans = [json.loads((d / "manifest.json").read_text()) for d in dirs]
    head = a_value(0, 0).rsplit("/", 2)[0]
    rounds = sorted({int(e["value"].split("/")[1]) for e in w["registered"] if e["value"].startswith(head + "/")})
    at = dict(zip(rounds, zip(dirs, mans)))

    def rows_of(name: str, n: int, _empty: np.ndarray) -> np.ndarray:
        s, c = map(int, name.rsplit("/", 2)[1:])
        d, man = at[s]
        u = man["matmuls"][c]
        spec, k = (u["rows"], u["k"]) if name == a_value(s, c) else (u["y"]["rows"], u["n"])
        return (np.frombuffer(_file(d, spec), "<u2").reshape(n, k).astype("<u4") << 16).view("<u2")

    try:
        return {"rounds": rounds, "values": len(W.reuse(w, mans, rows_of, rounds, dirs)), "ok": True}
    except ValueError as e:
        return {"rounds": rounds, "ok": False, "refused": str(e)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("pass_dirs", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reuse", type=Path)
    ap.add_argument("--call", type=int)
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--program")
    ap.add_argument("--window")
    a = ap.parse_args(argv)
    if a.reuse:
        got = reuse(a.reuse, a.pass_dirs)
        print(json.dumps({k: v for k, v in got.items() if k != "refused"}), "ok" if got["ok"] else "FAILED")
    else:
        got = check(a.pass_dirs[0], a.call, a.round, a.program, a.window)
        print(json.dumps({x: {k: v for k, v in t.items() if isinstance(v, bool)} for x, t in got["trees"].items()}),
              "ok" if got["ok"] else "FAILED")
    a.out.write_text(json.dumps(got, indent=1) + "\n")
    return 0 if got["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
