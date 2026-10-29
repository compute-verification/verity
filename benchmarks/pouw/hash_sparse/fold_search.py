"""Search for programs cheaper than honest for one step of the multiply-xor fold N <- (N ^ D) k at zero data.

At zero data a k32 word is D = R + C, R the row's term and C the tile column's, so the honest step is three writes:
the word (an MMA, or an IADD3 from the prefixes), a LOP3 and an IMUL. Checks (seeded, exact integer arithmetic):

  superopt     every straight-line program of one or two instructions over N, R, C, k and free per-row, per-column
               and per-key terms (R k, C k, k^-1, negations, constants), meet in the middle: the first instruction
               enumerated (LOP3 with all 256 functions, IADD3 with signs, IMAD, IMUL.HI, LEA, shifts, rotates,
               funnel shifts, IMNMX), the last inverted and looked up (IMAD, IADD3, LOP3, LEA, shifts, rotates,
               funnel shifts, IMNMX), on 64 zero-data cases. Sanity targets with a known two-instruction program
               must be found.
  bitslice     the whole step for 32 outputs of a row, a register a bit: carry chain of R + C with R broadcast, the
               xor into the state, and the multiply by NAF shift-add; LOP3s a step, bit for bit.
  tables       the multiply by 8-bit and 16-bit lookup tables built once a step and call (free), the xor and the
               index extraction per output; instructions a step, bit for bit.

usage: fold_search.py [OUT.json]
"""
import itertools
import json
import sys

import numpy as np

U32 = (1 << 32) - 1
NC = 64


def u(x):
    return np.asarray(x, dtype=np.uint64) & np.uint64(U32)


def inv(k):
    return u([pow(int(v), -1, 1 << 32) for v in k])


def cases(rng):
    n = u(rng.integers(0, 1 << 32, NC, dtype=np.int64))
    r = u(rng.integers(2, 900, NC))
    c = u(rng.integers(-500_000, 500_000, NC))
    k = u(rng.integers(0, 1 << 31, NC, dtype=np.int64) * 2 + 1)
    return n, r, c, k


def base_regs(n, r, c, k):
    z = u(np.zeros(NC, dtype=np.int64))
    regs = {"N": n, "R": r, "C": c, "k": k, "Rk": u(r * k), "Ck": u(c * k), "kinv": inv(k), "-k": u(-k.astype(np.int64)),
            "-R": u(-r.astype(np.int64)), "-C": u(-c.astype(np.int64)), "0": z, "1": z + np.uint64(1),
            "~0": z + np.uint64(U32)}
    return list(regs), list(regs.values())


def fp(v):
    return v.tobytes()


def shl(v, s):
    return u(v << np.uint64(s))


def shr(v, s):
    return v >> np.uint64(s)


def rotl(v, s):
    return u((v << np.uint64(s)) | (v >> np.uint64(32 - s)))


def sar(v, s):
    return u(v.astype(np.uint32).view(np.int32).astype(np.int64) >> s)


def lop3_all(x, y, z):
    """All 256 LOP3 functions of (x, y, z): row f is the function with truth table f (bit 4a+2b+c)."""
    mins = []
    for m in range(8):
        a, b, cc = (m >> 2) & 1, (m >> 1) & 1, m & 1
        mins.append((x if a else ~x & np.uint64(U32)) & (y if b else ~y & np.uint64(U32)) & (z if cc else ~z & np.uint64(U32)))
    mins = np.stack(mins)
    f = np.arange(256)[:, None]
    out = np.zeros((256, NC), dtype=np.uint64)
    for m in range(8):
        out |= np.where(((f >> m) & 1).astype(bool), mins[m][None, :], np.uint64(0))
    return out


def first_ops(names, regs):
    """Every one-instruction value over the base registers: {fingerprint: description}."""
    table = {}
    vals = []

    def add(v, d):
        key = fp(v)
        if key not in table:
            table[key] = d
            vals.append(v)

    idx = range(len(regs))
    for i, j, l in itertools.combinations_with_replacement(idx, 3):
        out = lop3_all(regs[i], regs[j], regs[l])
        for f in range(256):
            add(out[f], f"LOP3.{f:#04x}({names[i]},{names[j]},{names[l]})")
    signs = list(itertools.product((1, -1), repeat=3))
    for i, j, l in itertools.combinations_with_replacement(idx, 3):
        for s in signs:
            v = u(s[0] * regs[i].astype(np.int64) + s[1] * regs[j].astype(np.int64) + s[2] * regs[l].astype(np.int64))
            add(v, f"IADD3({'+-'[s[0] < 0]}{names[i]},{'+-'[s[1] < 0]}{names[j]},{'+-'[s[2] < 0]}{names[l]})")
    for i, j in itertools.combinations_with_replacement(idx, 2):
        p = u(regs[i] * regs[j])
        add(u((regs[i] * regs[j]) >> np.uint64(32)), f"IMUL.HI({names[i]},{names[j]})")
        for l in idx:
            add(u(p + regs[l]), f"IMAD({names[i]},{names[j]},+{names[l]})")
            add(u(p - regs[l]), f"IMAD({names[i]},{names[j]},-{names[l]})")
        for mn, op in (("IMNMX.U.MIN", np.minimum), ("IMNMX.U.MAX", np.maximum)):
            add(op(regs[i], regs[j]), f"{mn}({names[i]},{names[j]})")
        si, sj = regs[i].astype(np.uint32).view(np.int32), regs[j].astype(np.uint32).view(np.int32)
        add(u(np.minimum(si, sj).astype(np.int64)), f"IMNMX.S.MIN({names[i]},{names[j]})")
        add(u(np.maximum(si, sj).astype(np.int64)), f"IMNMX.S.MAX({names[i]},{names[j]})")
    for s in range(1, 32):
        for i in idx:
            add(shl(regs[i], s), f"SHL({names[i]},{s})")
            add(shr(regs[i], s), f"SHR({names[i]},{s})")
            add(sar(regs[i], s), f"SAR({names[i]},{s})")
            add(rotl(regs[i], s), f"ROTL({names[i]},{s})")
            for j in idx:
                add(u(shl(regs[i], s) + regs[j]), f"LEA({names[i]}<<{s},+{names[j]})")
                add(u(shl(regs[i], s) - regs[j]), f"LEA({names[i]}<<{s},-{names[j]})")
                add(u(shr(regs[j], s) | shl(regs[i], 32 - s)), f"SHF.R({names[i]}:{names[j]},{s})")
    return table, np.stack(vals)


def search(target, names, regs, table, vals):
    """Two-instruction programs ending in each final form, by inverting the final instruction."""
    found = []
    t64 = target.astype(np.int64)
    if fp(target) in table:
        found.append(("one instruction", table[fp(target)]))

    def look(v, how):
        d = table.get(fp(u(v)))
        if d is not None:
            found.append((how, d))

    idx = range(len(regs))
    signs = list(itertools.product((1, -1), repeat=3))
    for i, j in itertools.combinations_with_replacement(idx, 2):
        a, b = regs[i].astype(np.int64), regs[j].astype(np.int64)
        for s in signs:
            look(s[0] * (t64 - s[1] * a - s[2] * b), f"IADD3(v1,{names[i]},{names[j]})")
        look(t64 - (regs[i] * regs[j]).astype(np.int64), f"IMAD({names[i]},{names[j]},+v1)")
    for j in idx:
        b = regs[j]
        odd = bool((b & np.uint64(1)).all())
        for l in idx:
            for sg in (1, -1):
                z = sg * regs[l].astype(np.int64)
                if odd:
                    look(u(u(t64 - z) * inv(b)), f"IMAD(v1,{names[j]},{'+-'[sg < 0]}{names[l]})")
                else:
                    prod = (u(vals * b[None, :]) + u(z)[None, :]) & np.uint64(U32)
                    hit = np.nonzero((prod == target[None, :]).all(1))[0]
                    for h in hit:
                        found.append((f"IMAD(v1,{names[j]},{'+-'[sg < 0]}{names[l]})", table[fp(vals[h])]))
    bits = np.arange(32, dtype=np.uint64)
    tb = ((target[:, None] >> bits) & np.uint64(1)).ravel()
    for i, j in itertools.combinations_with_replacement(idx, 2):
        ab = ((regs[i][:, None] >> bits) & np.uint64(1)).ravel() * 2 + ((regs[j][:, None] >> bits) & np.uint64(1)).ravel()
        cls = [ab == q for q in range(4)]
        nonempty = [q for q in range(4) if cls[q].any()]
        const = [q for q in nonempty if tb[cls[q]].min() == tb[cls[q]].max()]
        if const:
            mask_vals = []
            for choice in itertools.product((0, 1), repeat=len(nonempty)):
                want = np.zeros_like(tb)
                care = np.zeros_like(tb, dtype=bool)
                for q, ch in zip(nonempty, choice):
                    if q in const:
                        continue
                    want[cls[q]] = tb[cls[q]] ^ ch
                    care |= cls[q]
                mask_vals.append((want, care))
            for lo in range(0, len(vals), 4096):
                chunk = vals[lo:lo + 4096]
                vb = ((chunk[:, :, None] >> bits) & np.uint64(1)).reshape(len(chunk), -1)
                for want, care in mask_vals:
                    hit = np.nonzero((vb[:, care] == want[care]).all(1))[0]
                    for h in hit[:3]:
                        found.append((f"LOP3(v1,{names[i]},{names[j]}) partial", table[fp(chunk[h])]))
            continue
        for choice in itertools.product((0, 1), repeat=len(nonempty)):
            vbits = tb.copy()
            for q, ch in zip(nonempty, choice):
                vbits[cls[q]] ^= np.uint64(ch)
            v = (vbits.reshape(NC, 32) << bits).sum(1).astype(np.uint64)
            look(v, f"LOP3(v1,{names[i]},{names[j]})")
    for s in range(1, 32):
        look(rotl(target, 32 - s), f"ROTL(v1,{s})")
        for i in idx:
            look(t64 - shl(regs[i], s).astype(np.int64), f"LEA({names[i]}<<{s},+v1)")
            look(shl(regs[i], s).astype(np.int64) - t64, f"LEA({names[i]}<<{s},-v1)")
            r = u(t64 - regs[i].astype(np.int64))
            if not (r & np.uint64((1 << s) - 1)).any():
                hit = np.nonzero((shl(vals, s) == r[None, :]).all(1))[0]
                for h in hit:
                    found.append((f"LEA(v1<<{s},{names[i]})", table[fp(vals[h])]))
            if (shr(regs[i], s) == (target & np.uint64((1 << (32 - s)) - 1))).all():
                hit = np.nonzero((shr(target, 32 - s)[None, :] == (vals & np.uint64((1 << s) - 1))).all(1))[0]
                for h in hit:
                    found.append((f"SHF.R(v1:{names[i]},{s})", table[fp(vals[h])]))
            if (shl(regs[i], 32 - s) == (target & ~np.uint64((1 << (32 - s)) - 1) & np.uint64(U32))).all():
                hit = np.nonzero((shr(vals, s) == (target & np.uint64((1 << (32 - s)) - 1))[None, :]).all(1))[0]
                for h in hit:
                    found.append((f"SHF.R({names[i]}:v1,{s})", table[fp(vals[h])]))
        if not (target & np.uint64((1 << s) - 1)).any():
            hit = np.nonzero((shl(vals, s) == target[None, :]).all(1))[0]
            found += [(f"SHL(v1,{s})", table[fp(vals[h])]) for h in hit]
        if not (target >> np.uint64(32 - s)).any():
            hit = np.nonzero((shr(vals, s) == target[None, :]).all(1))[0]
            found += [(f"SHR(v1,{s})", table[fp(vals[h])]) for h in hit]
    for i in idx:
        eq = target == regs[i]
        if eq.any() and not eq.all():
            hit = np.nonzero((vals[:, ~eq] == target[None, ~eq]).all(1))[0]
            found += [(f"IMNMX(v1,{names[i]}) candidate", table[fp(vals[h])]) for h in hit[:3]]
    return found


def superopt(rng):
    n, r, c, k = cases(rng)
    names, regs = base_regs(n, r, c, k)
    d = u(r + c)
    targets = {
        "step (N ^ (R + C)) k": u(u(n ^ d) * k),
        "sanity: (N ^ R) k": u(u(n ^ r) * k),
        "sanity: N k + R": u(n * k + r),
        "sanity: (N + R) ^ C": u(n + r) ^ c,
    }
    table, vals = first_ops(names, regs)
    out = {"first_instructions_distinct": len(table), "base": names}
    for name, t in targets.items():
        hits = search(t, names, regs, table, vals)
        verified = []
        for how, d1 in hits:
            verified.append(f"{how} <- {d1}")
        out[name] = {"programs_found": len(verified), "examples": verified[:4]}
    return out


def naf(k):
    out, i = [], 0
    while k:
        if k & 1:
            d = 2 - (k & 3)
            out.append((i, d))
            k -= d
        k >>= 1
        i += 1
    return out


def bitslice(rng, s_steps=16):
    """The whole step for 32 outputs of one row, a register a bit, counted and checked against the word fold."""
    lanes = 1 << np.arange(32, dtype=np.int64)
    rr = rng.integers(2, 900, s_steps)
    cc = rng.integers(-500_000, 500_000, (s_steps, 32)) & U32
    ks = rng.integers(0, 1 << 31, s_steps) * 2 + 1
    n0 = rng.integers(0, 1 << 32, 32, dtype=np.int64)
    want = n0.copy()
    for t in range(s_steps):
        want = ((want ^ ((int(rr[t]) + cc[t]) & U32)) * int(ks[t])) & U32
    st = [int(((n0 >> b) & 1) @ lanes) for b in range(32)]
    ops = {"add_xor": 0, "multiply": 0}
    for t in range(s_steps):
        cb = [int(((cc[t] >> b) & 1) @ lanes) for b in range(32)]
        ab = [U32 if (int(rr[t]) >> b) & 1 else 0 for b in range(32)]
        carry = 0
        for b in range(32):
            st[b] ^= ab[b] ^ cb[b] ^ carry
            ops["add_xor"] += 1 if b == 0 else 2
            if b < 31:
                x, y = ab[b], cb[b]
                carry = (x & y) | (x & carry) | (y & carry)
                ops["add_xor"] += 1
        k = int(ks[t])
        digits = naf(k)
        top = max(digits)
        acc = [0] * top[0] + st[:32 - top[0]]
        for s, d in digits:
            if (s, d) == top:
                continue
            y = [0] * s + st[:32 - s]
            if d < 0:
                y = [v ^ U32 if b >= s else 0 for b, v in enumerate(y)]
            carry = U32 if d < 0 else 0
            for b in range(s, 32):
                a_, y_ = acc[b], y[b]
                acc[b] = a_ ^ y_ ^ carry
                ops["multiply"] += 1
                if b < 31:
                    carry = (a_ & y_) | (a_ & carry) | (y_ & carry)
                    ops["multiply"] += 1
        st = acc
    got = np.array([sum(((st[b] >> j) & 1) << b for b in range(32)) for j in range(32)], dtype=np.int64)
    per = {key: v / (32 * s_steps) for key, v in ops.items()}
    return {"state_match": bool((got == want).all()), "lop3_a_step": per,
            "lop3_a_step_total": sum(per.values()), "honest_writes_a_step": 3}


def tables(rng, s_steps=16, m=256):
    """N k by tables T_b[x] = (x << 8b) k (8-bit: 4 lookups) or (x << 16b) k (16-bit: 2 lookups), built once a step and
    call. Per output and step: the word (IADD3), the xor (LOP3), the index of each lookup (one PRMT, BFE or LOP3,
    the shared-memory scaling folded into the load), the loads, and the adds to recombine (IADD3 takes three)."""
    rr = rng.integers(2, 900, (m, 1))
    cc = rng.integers(-500_000, 500_000, (s_steps, m)) & U32
    ks = rng.integers(0, 1 << 31, s_steps) * 2 + 1
    out = {}
    for width in (8, 16):
        parts = 32 // width
        n = rng.integers(0, 1 << 32, m, dtype=np.int64)
        want = n.copy()
        for t in range(s_steps):
            k = int(ks[t])
            tab = [np.array([((x << (width * b)) * k) & U32 for x in range(1 << width)], dtype=np.int64)
                   for b in range(parts)]
            x = n ^ ((rr[:, 0] + cc[t]) & U32)
            n = sum(tab[b][(x >> (width * b)) & ((1 << width) - 1)] for b in range(parts)) & U32
            want = ((want ^ ((rr[:, 0] + cc[t]) & U32)) * k) & U32
        adds = (parts - 1 + 1) // 2
        instr = 1 + 1 + parts + parts + adds
        out[f"{width}-bit"] = {"state_match": bool((n == want).all()), "instructions_a_step": instr,
                               "table_bytes_a_step": parts * (1 << width) * 4, "honest_writes_a_step": 3}
    return out


def main():
    rng = np.random.default_rng(20261029)
    res = {"superopt": superopt(rng), "bitslice": bitslice(rng), "tables": tables(rng)}
    s = json.dumps(res, indent=1)
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as f:
            f.write(s + "\n")
    print(s)


if __name__ == "__main__":
    main()
