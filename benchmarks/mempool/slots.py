"""CacheStep_v3's operation slots as Boolean circuits (a prototype), evaluated against `pool_ram`'s logged operations.

A slot runs one operation of one kind (`pool_ram.SHAPE`). Its inputs:
- s ≥ 1, its place in the window's stream of slots (a constant where the stream has a fixed shape): position j accesses
  at time 8s + j;
- the operation's descriptor (`DESC`);
- for each position j, the value it reads and that value's time (witness). A read the operation requires to be 0 (the
  request's block slot it fills, the key of the block it caches) has no value input;
- the coins α and γ.
Its outputs, per position: whether it is used, its address and the value it writes; and the products of α + e over its
read records and over its write records (`pool_ram.element`), an unused position's factor 1. Its assertions: the
operation's preconditions, every used read's time below its access's, and every unused position's inputs zero, so that
the valid witness is unique.

    python3 slots.py [--seeds N] [--vectors N] [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pool_ram as PR

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from verity.circuits.boolean import forms as F
from verity.circuits.boolean.forms import NOT, ONE, ZERO, const_bits, mux, or_reduce, sub, add
from verity.circuits.boolean.gf128 import mul_forms

DESC = {"lookup": (("key", "HW"),), "take": (("rid", "RW"), ("k", "KW"), ("b", "NW")),
        "alloc": (("rid", "RW"), ("k", "KW")), "cache": (("rid", "RW"), ("k", "KW"), ("key", "HW")),
        "free": (("rid", "RW"), ("k", "KW"), ("cur", "NW"))}
MEMS = {"lookup": ("M", "B"), "take": ("B", "B", "B", "RB"), "alloc": ("B", "B", "B", "K", "M", "B", "B", "RB"),
        "cache": ("RB", "K", "M", "B", "B"), "free": ("RB", "B", "K", "B", "B")}
#: reads the operation requires to be 0
ZERO_READ = {("take", 3), ("alloc", 7), ("cache", 1)}
BF = ("prev", "next", "cprev", "cnext", "ref", "owner")


def vwidth(mem: str, S: PR.Widths) -> int:
    return {"B": 4 * S.NW + S.REFW + S.RW, "K": S.HW, "M": 2 * S.NW, "RB": S.NW}[mem]


def awidth(mem: str, S: PR.Widths) -> int:
    return {"B": S.NW, "K": S.NW, "M": S.HW, "RB": S.RW + S.KW}[mem]


def _bw(S: PR.Widths) -> dict[str, int]:
    return {"prev": S.NW, "next": S.NW, "cprev": S.NW, "cnext": S.NW, "ref": S.REFW, "owner": S.RW}


def bsplit(v: list, S: PR.Widths) -> dict[str, list]:
    out, i = {}, 0
    for f, w in _bw(S).items():
        out[f] = v[i:i + w]
        i += w
    return out


def bjoin(d: dict[str, list]) -> list:
    return [x for f in BF for x in d[f]]


def nz(C, x: list):
    return or_reduce(C, x)


def eqz(C, x: list):
    return NOT(or_reduce(C, x))


def eq(C, x: list, y: list):
    return eqz(C, [a ^ b for a, b in zip(x, y)])


def zmux(C, s, x: list) -> list:
    """s ? 0 : x."""
    return [C.AND(NOT(s), b) for b in x]


class Slot:
    def __init__(self, kind: str, S: PR.Widths, caching: bool = True) -> None:
        C = self.C = F.Circuit(cse=True)
        self.kind, self.S = kind, S
        self.s = C.input("s", S.TW - 3)
        self.alpha = C.input("alpha", 128)
        self.gamma = C.input("gamma", 128)
        self.d = {n: C.input(n, getattr(S, w)) for n, w in DESC[kind]}
        self.v: dict[int, list] = {}
        self.t: dict[int, list] = {}
        for j, mem in enumerate(MEMS[kind]):
            zero = (kind, j) in ZERO_READ
            self.v[j] = const_bits(0, vwidth(mem, S)) if zero else C.input(f"v{j}", vwidth(mem, S))
            self.t[j] = C.input(f"t{j}", S.TW)
        self.acc: dict[int, tuple] = {}
        getattr(self, "_" + kind)(caching)
        self._finish()

    def put(self, j: int, en, addr: list, w: list) -> None:
        mem = MEMS[self.kind][j]
        assert len(addr) == awidth(mem, self.S) and len(w) == vwidth(mem, self.S), (self.kind, j)
        self.acc[j] = (en, addr, w)

    def now(self, j: int) -> list:
        return const_bits(j, 3) + self.s

    def factor(self, en, mem: str, addr: list, val: list, t: list) -> list:
        C = self.C
        bits = const_bits(PR.TAGS[mem], 2) + addr + val + t
        bits += [ZERO] * (-len(bits) % 128)
        chunks = [bits[i:i + 128] for i in range(0, len(bits), 128)]
        e, g = chunks[0], self.gamma
        for i, c in enumerate(chunks[1:]):
            e = [a ^ b for a, b in zip(e, mul_forms(C, g, c))]
            if i + 2 < len(chunks):
                g = mul_forms(C, g, self.gamma)
        return mux(C, en, [a ^ b for a, b in zip(self.alpha, e)], const_bits(1, 128))

    def product(self, fs: list[list]) -> list:
        p = fs[0]
        for f in fs[1:]:
            p = mul_forms(self.C, p, f)
        return p

    def _finish(self) -> None:
        C, S = self.C, self.S
        fr, fw = [], []
        for j, mem in enumerate(MEMS[self.kind]):
            en, addr, w = self.acc[j]
            v, t, now = self.v[j], self.t[j], self.now(j)
            wit = ([] if (self.kind, j) in ZERO_READ else v) + t
            for x in wit:
                C.assert_zero(f"unused {j}", C.AND(NOT(en), x))
            _, below = sub(C, t, now, S.TW)
            C.assert_zero(f"time {j}", C.AND(en, NOT(below)))
            fr.append(self.factor(en, mem, addr, v, t))
            fw.append(self.factor(en, mem, addr, w, now))
            C.output(f"en{j}", [en])
            C.output(f"a{j}", addr)
            C.output(f"w{j}", w)
        C.output("R", self.product(fr))
        C.output("W", self.product(fw))

    # --- the operations (`pool_ram.RamPool`'s, position for position) ---

    def _lookup(self, caching: bool) -> None:
        C, S, v = self.C, self.S, self.v
        first = v[0][:S.NW]
        self.put(0, ONE, self.d["key"], v[0])
        hit = nz(C, first)
        self.put(1, hit, first, v[1])
        C.output("hit", [hit])
        C.output("evictable", [C.AND(hit, eqz(C, bsplit(v[1], S)["ref"]))])

    def _take(self, caching: bool) -> None:
        C, S, v, d = self.C, self.S, self.v, self.d
        b = d["b"]
        C.assert_zero("b is a block", eqz(C, b))
        r0 = bsplit(v[0], S)
        free = eqz(C, r0["ref"])
        C.assert_zero("holders fit", F.and_reduce(C, r0["ref"]))
        w0 = dict(r0, ref=add(C, r0["ref"], const_bits(1, S.REFW), width=S.REFW),
                  prev=zmux(C, free, r0["prev"]), next=zmux(C, free, r0["next"]))
        self.put(0, ONE, b, bjoin(w0))
        p, n = r0["prev"], r0["next"]
        self.put(1, free, p, bjoin(dict(bsplit(v[1], S), next=n)))
        self.put(2, free, n, bjoin(dict(bsplit(v[2], S), prev=p)))
        self.put(3, ONE, d["rid"] + d["k"], b)

    def _alloc(self, caching: bool) -> None:
        C, S, v, d = self.C, self.S, self.v, self.d
        zn = const_bits(0, S.NW)
        r0, r1, r2 = bsplit(v[0], S), bsplit(v[1], S), bsplit(v[2], S)
        b, n = r0["next"], r1["next"]
        C.assert_zero("the free queue is not empty", eqz(C, b))
        C.assert_zero("the head is free", nz(C, r1["ref"]))
        self.put(0, ONE, zn, bjoin(dict(r0, next=n)))
        owner = add(C, d["rid"], const_bits(1, S.RW), width=S.RW)
        self.put(1, ONE, b, bjoin({"prev": zn, "next": zn, "cprev": zn, "cnext": zn,
                                   "ref": const_bits(1, S.REFW), "owner": owner}))
        self.put(2, ONE, n, bjoin(dict(r2, prev=zn)))
        if caching:
            key = v[3]
            ev = nz(C, key)
            self.put(3, ONE, b, const_bits(0, S.HW))
            cp, cn = r1["cprev"], r1["cnext"]
            first, last = v[4][:S.NW], v[4][S.NW:]
            self.put(4, ev, key, mux(C, eq(C, first, b), cn, first) + mux(C, eq(C, last, b), cp, last))
            self.put(5, C.AND(ev, nz(C, cp)), cp, bjoin(dict(bsplit(v[5], S), cnext=cn)))
            self.put(6, C.AND(ev, nz(C, cn)), cn, bjoin(dict(bsplit(v[6], S), cprev=cp)))
        else:
            for j in (3, 4, 5, 6):
                mem = MEMS["alloc"][j]
                self.put(j, ZERO, const_bits(0, awidth(mem, S)), const_bits(0, vwidth(mem, S)))
        self.put(7, ONE, d["rid"] + d["k"], b)

    def _cache(self, caching: bool) -> None:
        C, S, v, d = self.C, self.S, self.v, self.d
        b = v[0]
        C.assert_zero("the slot holds a block", eqz(C, b))
        self.put(0, ONE, d["rid"] + d["k"], b)
        self.put(1, ONE, b, d["key"])
        f, l = v[2][:S.NW], v[2][S.NW:]
        self.put(2, ONE, d["key"], mux(C, nz(C, f), f, b) + b)
        self.put(3, nz(C, l), l, bjoin(dict(bsplit(v[3], S), cnext=b)))
        self.put(4, ONE, b, bjoin(dict(bsplit(v[4], S), cprev=l, cnext=const_bits(0, S.NW))))

    def _free(self, caching: bool) -> None:
        C, S, v, d = self.C, self.S, self.v, self.d
        zn = const_bits(0, S.NW)
        b, cur = v[0], d["cur"]
        C.assert_zero("the slot holds a block", eqz(C, b))
        self.put(0, ONE, d["rid"] + d["k"], zn)
        r1, r3, r4 = bsplit(v[1], S), bsplit(v[3], S), bsplit(v[4], S)
        C.assert_zero("the block is held", eqz(C, r1["ref"]))
        ret = eq(C, r1["ref"], const_bits(1, S.REFW))
        if caching:
            self.put(2, ret, b, v[2])
            hashless = eqz(C, v[2])
        else:
            self.put(2, ZERO, zn, const_bits(0, S.HW))
            hashless = ONE
        front = C.AND(ret, hashless)
        x = mux(C, front, cur, r4["prev"])
        y = mux(C, front, r3["next"], zn)
        w1 = dict(r1, ref=sub(C, r1["ref"], const_bits(1, S.REFW), S.REFW)[0],
                  prev=mux(C, ret, x, r1["prev"]), next=mux(C, ret, y, r1["next"]))
        self.put(1, ONE, b, bjoin(w1))
        self.put(3, ret, x, bjoin(dict(r3, next=b)))
        self.put(4, ret, y, bjoin(dict(r4, prev=b)))
        C.output("ret", [ret])
        C.output("cur", mux(C, front, b, cur))


# --- evaluation against the reference ------------------------------------------------------------------------------------


def venc(mem: str, val: Any, S: PR.Widths) -> int:
    if mem == "B":
        x, s = 0, 0
        for f, w in _bw(S).items():
            x |= getattr(val, f) << s
            s += w
        return x
    if mem == "M":
        return val[0] | val[1] << S.NW
    return val


def aenc(a: tuple, S: PR.Widths) -> int:
    return a[1] | a[2] << S.RW if a[0] == "RB" else a[1]


def _slice(vals: list[int], width: int) -> list[int]:
    out = [0] * width
    for vi, x in enumerate(vals):
        i = 0
        while x:
            if x & 1:
                out[i] |= 1 << vi
            x >>= 1
            i += 1
    return out


def _unslice(bits: list[int], T: int) -> list[int]:
    out = [0] * T
    for i, x in enumerate(bits):
        vi = 0
        while x:
            if x & 1:
                out[vi] |= 1 << i
            x >>= 1
            vi += 1
    return out


def evaluate(slot: Slot, ops: list[PR.PoolOp], log: list, alpha: int, gamma: int) -> dict[str, int]:
    """Runs the slot on each operation (one bit-sliced vector each) and holds its outputs to the log's; returns counts of
    vectors that agree and of assertion failures."""
    C, S, kind = slot.C, slot.S, slot.kind
    T = len(ops)
    inputs = {n: _slice(v, len(C.inputs[n])) for n, v in _inputs(slot, ops, log, alpha, gamma).items()}
    outs, fail = C.evaluate(inputs, T)
    got = {n: _unslice(bits, T) for n, bits in outs.items()}
    bad = Counter()
    for vi, o in enumerate(ops):
        R = W = 1
        for j, mem in enumerate(MEMS[kind]):
            used = j in o.at
            if got[f"en{j}"][vi] != used:
                bad[f"en{j}"] += 1
                continue
            if not used:
                continue
            a, v, t, w, now = log[o.at[j]]
            if got[f"a{j}"][vi] != aenc(a, S):
                bad[f"a{j}"] += 1
            if got[f"w{j}"][vi] != venc(mem, w, S):
                bad[f"w{j}"] += 1
            R = PR.gf_mul(R, alpha ^ PR.element((a, v, t), gamma, S))
            W = PR.gf_mul(W, alpha ^ PR.element((a, w, now), gamma, S))
        bad["R"] += got["R"][vi] != R
        bad["W"] += got["W"][vi] != W
    return {"vectors": T, "assertion failures": bin(fail).count("1"), **{k: v for k, v in bad.items() if v}}


def tampered(slot: Slot, ops: list[PR.PoolOp], log: list, alpha: int, gamma: int, rng: random.Random) -> int:
    """How many of the vectors, with one used read's value or time bit flipped, the slot still accepts with the honest
    read product (an accepted tamper would have to be caught by nothing; expected 0)."""
    C, kind = slot.C, slot.kind
    bad = {n: list(v) for n, v in _inputs(slot, ops, log, alpha, gamma).items()}
    T = len(ops)
    for vi, o in enumerate(ops):
        js = [j for j in o.at if (kind, j) not in ZERO_READ] or list(o.at)
        j = rng.choice(js)
        name = rng.choice([f"v{j}", f"t{j}"] if f"v{j}" in bad else [f"t{j}"])
        bad[name][vi] ^= 1 << rng.randrange(len(C.inputs[name]))
    inputs = {n: _slice(bad[n], len(bits)) for n, bits in C.inputs.items()}
    outs, fail = C.evaluate(inputs, T)
    R_t = _unslice(outs["R"], T)
    clean = {n: _slice(v, len(C.inputs[n])) for n, v in _inputs(slot, ops, log, alpha, gamma).items()}
    R_h = _unslice(C.evaluate(clean, T)[0]["R"], T)
    return sum(1 for vi in range(T) if not (fail >> vi) & 1 and R_t[vi] == R_h[vi])


def _inputs(slot: Slot, ops: list[PR.PoolOp], log: list, alpha: int, gamma: int) -> dict[str, list[int]]:
    S, kind = slot.S, slot.kind
    T = len(ops)
    vals: dict[str, list[int]] = {"s": [o.base // 8 for o in ops], "alpha": [alpha] * T, "gamma": [gamma] * T}
    for n, _ in DESC[kind]:
        vals[n] = [o.desc[n] for o in ops]
    for j, mem in enumerate(MEMS[kind]):
        vals[f"v{j}"] = [venc(mem, log[o.at[j]][1], S) if j in o.at else 0 for o in ops]
        vals[f"t{j}"] = [log[o.at[j]][2] if j in o.at else 0 for o in ops]
    return {n: vals[n] for n in slot.C.inputs}


def counts(slot: Slot) -> dict[str, int]:
    c = slot.C.counts()
    S = slot.S
    public = (S.TW - 3) + 256
    return {"ands": c["ands"], "witness_bits": c["inputs"] - public, "inputs": c["inputs"]}


def main() -> None:
    import difftest as DT

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--vectors", type=int, default=4000)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    rng = random.Random(5)
    by_kind: dict[tuple[bool, str], list[tuple[PR.PoolOp, list]]] = {}
    runs = [("async-crush.json.gz", DT.load(DT.FIXTURES / "async-crush.json.gz"), False),
            ("async-prefix-cache.jsonl", DT.load(DT.FIXTURES / "async-prefix-cache.jsonl"), False)]
    if (DT.FIXTURES / "async-kv-pressure.jsonl").exists():
        runs.append(("async-kv-pressure.jsonl", DT.load(DT.FIXTURES / "async-kv-pressure.jsonl"), False))
    for s in range(a.seeds):
        runs.append((f"synthetic {s}", DT.synthetic(s, NB=rng.choice((24, 48)), B=rng.choice((4, 16))), True))
    for name, (eng, reqs, admit), arrivals in runs:
        how = {"decide": lambda i, k, o, admit=admit: admit[i] <= k} if arrivals else {"admit": admit}
        m = PR.run(eng, reqs, every=10**9, **how)
        if m.pool is None:
            continue
        for o in m.pool.ram.ops:
            by_kind.setdefault((m.pool.caching, o.kind), []).append((o, m.pool.ram.log))
    report: dict[str, Any] = {"widths": PR.GLM._asdict(), "slots": {}}
    alpha, gamma = rng.getrandbits(128), rng.getrandbits(128)
    for (caching, kind), items in sorted(by_kind.items()):
        slot = Slot(kind, PR.GLM, caching)
        rng.shuffle(items)
        items = items[: a.vectors]
        groups: dict[int, list[PR.PoolOp]] = {}
        logs: dict[int, list] = {}
        for o, log in items:
            groups.setdefault(id(log), []).append(o)
            logs[id(log)] = log
        res: Counter = Counter()
        for g, ops in groups.items():
            res.update(evaluate(slot, ops, logs[g], alpha, gamma))
            res["tampered accepted"] += tampered(slot, ops, logs[g], alpha, gamma, rng)
        row = {**counts(slot), **dict(res)}
        key = f"{kind}{'' if caching else ' (no caching)'}"
        report["slots"][key] = row
        print(key, row)
        if not res["vectors"] or any(v for k, v in res.items() if k != "vectors"):
            report.setdefault("failures", []).append(key)
    if a.json:
        a.json.write_text(json.dumps(report, indent=1))
    if report.get("failures"):
        raise SystemExit(f"slots that leave the reference: {report['failures']}")


if __name__ == "__main__":
    main()
