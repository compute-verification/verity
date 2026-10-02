"""A read at a private index of a registered table (``verity_flock.hidden_gather``) as a ``flock-circuit`` statement for C-Flock's
CPU prover, M0 and ``--zk`` (``76-hidden-gather.sh``), priced against #726's estimates (``hidden_reads.py``).

    python hidden_gather.py parts --out FILE        each table's opening by part: the hm96 leaf, the frame leaf, every level
                                                    (swap and node), the entry pick; ANDs and rows (``hidden_reads.counts``)
    python hidden_gather.py stage --table gelu_tanh_bf16|tanh_rn --n N [N ...] --out DIR
                                                    circuit.txt, inst-N.bin, pub-N.bin (verity_flock.circuit's composite)
    python hidden_gather.py summary --runs RUN [RUN ...] --out FILE
                                                    from runs of 76-hidden-gather.sh: rows and prover seconds per read

A unit is one opening (:func:`verity_flock.hidden_gather.gather`) under the table's registered domain for the statement's
program. Its inputs are the row (port ``leaf``, 64 u16 words) and ``aux``: the salt (96 words), the path (32 words per level,
leaf level first) and the index (low word first), zero to a multiple of 64 words. Its outputs are the root the opening climbs
to (32 words) and the entry (one word, two for ``tanh_rn``), both public in this statement: the root because the statement's
padding instances are all-zero, so a unit asserting the registered root would fail them, and the entry so that the verifier
sees what was read (in a program the entry stays inside the piece, and the root is the table's public input, asserted:
``hidden_gather.read``). ``gelu_tanh_bf16``'s tree is over the whole pinned table; ``tanh_rn``'s is sparse over the rows its
instances read, each the pinned table's (``Tree.sparse``: an opening costs what it costs in the tree of all 2^22 rows).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

import hidden_reads as HR
from verity_flock import gf2
from verity_flock import hidden_gather as HG
from verity_flock import tail_pieces as TP

#: #726's estimate per read (its summary at 77c21f68e): a packed hm96 leaf, c2 nodes (SHA-512 of the two digests), the entry
#: picked; rows, and prover milliseconds at its measured SHA-512-bound rates (1.93 ns/bit M0, 3.22 ZK on 48 loaded cores)
ESTIMATE_726 = {"gelu_tanh_bf16": {"depth": 10, "ands": 1_344_563, "rows": 1_874_146, "ms_m0": 3.6, "ms_zk": 6.0},
                "tanh_rn": {"depth": 22, "ands": 2_575_608, "rows": 3_588_263, "ms_m0": 6.9, "ms_zk": 11.6}}
SEED = 20261002


def _aux_words(t: HG.Table) -> int:
    need = HG.H5.salt_bytes // 2 + 32 * t.depth + -(-t.index_bits // 16)
    return 64 * -(-need // 64)


def _program(name: str) -> str:
    return hashlib.sha512(f"hidden-gather/{name}".encode()).hexdigest()


def unit_circuit(name: str):
    """The unit (inputs ``in{j}``, one u16 word each; outputs ``out{j}``), its ports, and the table's registered domain."""
    t = HG.TABLES[name]
    dom = HG.domain(_program(name), name, t.rows)
    aux = _aux_words(t)
    C = gf2.Circuit()
    w = [C.input(f"in{j}", 16) for j in range(HG.ROW_WORDS + aux)]
    row = [b for x in w[:HG.ROW_WORDS] for b in x]
    a = [b for x in w[HG.ROW_WORDS:] for b in x]
    sb = 8 * HG.H5.salt_bytes
    path = [a[sb + 512 * k:sb + 512 * (k + 1)] for k in range(t.depth)]
    at = sb + 512 * t.depth
    value, root = HG.gather(C, t, a[at:at + t.index_bits], row, a[:sb], path, HG._const(dom.domain_id))
    vw = -(-t.value_bits // 16)
    outb = root + value + [gf2.ZERO] * (16 * vw - len(value))
    for j in range(len(outb) // 16):
        C.output(f"out{j}", outb[16 * j:16 * j + 16])
    return C, [("leaf", HG.ROW_WORDS), ("aux", aux)], [("out", 32 + vw)], dom


def _row(t: HG.Table, entries, position: int) -> tuple[int, ...]:
    """``HG.table_rows``' row at ``position``, from the table's entries."""
    e = entries[position * t.per_row:(position + 1) * t.per_row]
    return tuple((int(x) >> (16 * k)) & 0xFFFF for x in e for k in range(t.entry_words))


def instances(name: str, dom, n: int):
    """``n`` honest openings: the tree, the indices read, and the unit's input words (n, 64 + aux)."""
    t = HG.TABLES[name]
    rng = np.random.default_rng(SEED + n)
    entries = gf2.table(name)
    if name == "tanh_rn":
        lo, hi = TP.TANH_BOUNDS[name]
        idx = [int(x) & ((1 << 27) - 1) for x in rng.integers(lo, hi, n)]
    else:
        idx = [int(x) for x in rng.integers(0, 1 << t.index_bits, n)]
    srng = random.Random(SEED + n)
    if name == "gelu_tanh_bf16":
        tree = HG.Tree.commit(t, dom, [_row(t, entries, p) for p in range(t.rows)], [srng.randbytes(HG.H5.salt_bytes) for _ in range(t.rows)])
    else:
        pos = sorted({i >> t.entry_log for i in idx})
        tree = HG.Tree.sparse(t, dom, {p: _row(t, entries, p) for p in pos}, f"hidden-gather/{n}".encode(),
                              {p: srng.randbytes(HG.H5.salt_bytes) for p in pos})
    flat = np.zeros((n, HG.ROW_WORDS + _aux_words(t)), dtype=np.uint64)
    for k, i in enumerate(idx):
        p = i >> t.entry_log
        words = np.frombuffer(tree.salts[p] + b"".join(tree.path(p)), dtype="<u2")
        flat[k, :HG.ROW_WORDS] = tree.rows[p]
        flat[k, HG.ROW_WORDS:HG.ROW_WORDS + len(words)] = words
        flat[k, HG.ROW_WORDS + len(words):HG.ROW_WORDS + len(words) + 2] = (i & 0xFFFF, i >> 16)
    return tree, idx, flat


def lowering(name: str):
    from verity_flock import ir_lower as IL

    C, ins, outs, dom = unit_circuit(name)
    n_in, n_out = sum(w for _, w in ins), sum(w for _, w in outs)
    U = IL.Units(None, [16] * n_in, [16] * n_out, [[("leaf", p) for p in range(n_in)]], [[("ret", p) for p in range(n_out)]],
                 (), [], [], [], builder=lambda: C)
    lay = IL.layout(C, [f"in{j}" for j in range(n_in)], [f"out{j}" for j in range(n_out)])
    leaves = {"in": [list(range(n_in))], "out": [list(range(n_out))], "in_ports": [[w, 16] for _, w in ins],
              "out_ports": [[w, 16] for _, w in outs]}

    class Low(IL.Lowering):
        @property
        def leaves(self):
            return leaves

    return Low(f"hidden-gather-{name}", None, U, C, lay), HR._Sub(f"gather-{name}", ins, outs), ins, outs, dom


def stage(name: str, ns: list[int], out: Path) -> dict:
    """The statement and each size's prover and verifier files; every instance's outputs are the tree's root and the entry."""
    from verity_flock import circuit as FC

    low, sub, ins, outs, dom = lowering(name)
    t = HG.TABLES[name]
    FC.program_digests = lambda s, statics=None: {s.id: hashlib.sha512(s.id.encode()).hexdigest()}
    text = FC.compose(low, sub, program=_program(name))
    out.mkdir(parents=True, exist_ok=True)
    (out / "circuit.txt").write_text(text)
    meta = json.loads(text.split("\n", 2)[1][5:])
    s = HR._Set(sub)
    info = {"table": name, "depth": t.depth, "domain": dom.domain_id.hex(), "unit_counts": HR.counts(low.circuit),
            "layout": low.layout.counts, "unit_log": meta["unit_log"], "k_log": meta["k_log"], "vus_per_block": meta["vus_per_block"],
            "per_vu": meta["per_vu"], "pin": FC.digest(text), "estimate_726": ESTIMATE_726[name], "n": {}}
    for n in ns:
        tree, idx, flat = instances(name, dom, n)
        res, _, ok = low.evaluate(flat)
        assert int(ok) == (1 << n) - 1, "every instance satisfies the unit rows"
        vw = outs[0][1] - 32
        want = np.array([list(np.frombuffer(tree.root, dtype="<u2")) + [(tree.entry(i) >> (16 * k)) & 0xFFFF for k in range(vw)]
                         for i in idx], dtype=np.uint64)
        assert (res == want).all(), "every opening climbs to the tree's root and reads the table's entry"
        at, rows = 0, {}
        for nm, w in ins:
            rows[nm] = flat[:, at:at + w].astype(np.uint16)
            at += w
        FC.write(text, s, 0, n, out / f"inst-{n}.bin", out / f"pub-{n}.bin", rows=rows, outs={"out": res.astype(np.uint16)})
        blocks = -(-n // meta["vus_per_block"])
        info["n"][str(n)] = {"blocks": blocks, "m": meta["k_log"] + max(0, (blocks - 1).bit_length()), "root": tree.root.hex(),
                             "rows_read": len({i >> t.entry_log for i in idx})}
    return info


def parts() -> dict:
    """Each table's opening by part, each built alone with its inputs as wires (``hidden_reads.delta``: rows exclude inputs),
    the domain a constant."""
    out = {}
    for name, t in HG.TABLES.items():
        d = HG._const(HG.domain(_program(name), name, t.rows).domain_id)
        nb = t.index_bits - t.entry_log
        p = {"hm96_leaf": HR.delta(lambda C: HG.hm96_leaf(C, C.input("row", 1024), C.input("salt", 1536))),
             "frame_leaf": HR.delta(lambda C: HG.frame_leaf_gadget(C, d, C.input("pos", nb), C.input("v", 512)))}
        for lv in range(t.depth):
            def level(C, lv=lv):
                pos = C.input("pos", nb)
                lo, hi = HG.swap(C, pos[lv], C.input("cur", 512), C.input("sib", 512))
                return HG.node_gadget(C, d, lv, pos[lv + 1:], lo, hi)
            p[f"level{lv}"] = HR.delta(level)
        p["pick"] = HR.delta(lambda C: HG.mux_tree(C, C.input("i", t.entry_log), [C.input(f"e{k}", t.value_bits) for k in range(t.per_row)]))
        total = {k: sum(c[k] for c in p.values()) for k in ("ands", "rows")}
        levels = {k: sum(c[k] for part, c in p.items() if part.startswith("level")) for k in ("ands", "rows")}
        est = ESTIMATE_726[name]
        out[name] = {"parts": p, "levels": levels, "total": total, "estimate_726": est,
                     "over_726": {k: round(total[k] / est[k] - 1, 4) for k in ("ands", "rows")}}
    return out


def summary(runs: list[Path]) -> dict:
    """Per table and mode: the marginal prover seconds per opening between the two largest sizes (``hidden_reads.rates``:
    witness and both reps' proofs, medians), the largest size's seconds per opening, and #726's estimate."""
    outs = [r / "out" for r in runs]
    rt = HR.rates(outs)
    stages = {}
    for o in outs:
        for f in o.glob("stage-*.json"):
            st = json.loads(f.read_text())
            stages[st["table"]] = st
    table = {}
    for name, st in sorted(stages.items()):
        row = {"rows": st["unit_counts"]["rows"], "ands": st["unit_counts"]["ands"], "unit_log": st["unit_log"], "k_log": st["k_log"],
               "estimate_726": st["estimate_726"]}
        for md in ("m0", "zk"):
            r = rt.get(f"{name}/{md}")
            if r:
                last = r["points"][-1]
                row[f"ms_{md}_marginal"] = round(1e3 * r["s_per_instance"], 3)
                row[f"ms_{md}_at_{last['n']}"] = round(1e3 * last["prove_s"] / last["n"], 3)
                row[f"ns_per_bit_{md}"] = r["ns_per_bit"]
        table[name] = row
    return {"runs": [r.name for r in runs], "rates": rt, "table": table}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["parts", "stage", "summary"])
    ap.add_argument("--table", choices=sorted(HG.TABLES))
    ap.add_argument("--n", type=int, nargs="+", default=[])
    ap.add_argument("--runs", nargs="+", default=[], help="summary: run dirs whose out/ holds live.tsv and stage-*.json")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.cmd == "parts":
        r = parts()
        Path(a.out).write_text(json.dumps(r, indent=1))
        print(json.dumps({k: {"total": v["total"], "over_726": v["over_726"]} for k, v in r.items()}))
        return 0
    if a.cmd == "summary":
        r = summary([Path(x) for x in a.runs])
        Path(a.out).write_text(json.dumps(r, indent=1))
        print(json.dumps(r["table"], indent=1))
        return 0
    info = stage(a.table, a.n, Path(a.out))
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    sys.exit(main())
