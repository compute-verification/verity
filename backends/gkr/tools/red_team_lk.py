"""red-team-lk: independent checks of the A-GKR statement rewrites of lanes agkr-fp8 / agkr-nvf4 (2026-09-24).

Run with ``backends/gkr`` of the PRODUCER tree under test first on ``PYTHONPATH`` (this file only reads its modules):

  static-merge  --plain DIR --merged DIR     text-level: every merged query is its plain query with the key constant shifted by
                                             tag 2^20, a constant tag column and zero pads; LK's rows of tag i are exactly the
                                             plain table's rows (key + i 2^20, i, outs, 0..); every other line is identical
  static-nvf4   [--paired]                   gpu.nvf4.circuit constraint systems: merge (export merge=False vs True), depth-1
                                             flatten (max_depth 99 vs 1), and with --paired BOOL_QUADRATIC / PAIRED
  forge         --relation fp8-ada|fp4-nvf4 --out DIR --verifier BIN [--vus 64]
                                             honest control + targeted forgeries proved by this process's prover with the LogUp
                                             self-checks off and adversarial multiplicities (first-column match); Python and the
                                             given Rust verifier must reject every forgery

A pass of ``static-*`` means: a witness satisfies the new statement iff (after the aux columns are fixed to their definitions)
it satisfies the old one, given LogUp soundness for a row-listed table (the verifier's, unchanged).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

P = 15 * 2**27 + 1
TAG_SHIFT = 2**20
STATEMENT_FILES = ("circuit.txt", "epilogue.txt", "chain.txt", "manifest.json")


# ---------------------------------------------------------------------------------------------------------------------------------
# text level
# ---------------------------------------------------------------------------------------------------------------------------------

def _lin(l, one: int) -> tuple:
    """canonical (terms over columns, constant) with the one-column terms folded into the constant"""
    k = l.konst % P
    t: dict[int, int] = {}
    for i, c in l.terms:
        if i == one:
            k = (k + c) % P
        else:
            t[i] = (t.get(i, 0) + c) % P
    return tuple(sorted((i, c) for i, c in t.items() if c)), k


def _rw(name: str) -> int | None:
    """``R<w>`` -> w (the implicit range tables); None for any other name (``RLO``, ``PR5``, ``LK``, ...)"""
    return int(name[1:]) if name.startswith("R") and name[1:].isdigit() else None


def _head(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if ln.split() and ln.split()[0] not in ("table", "itable", "row", "query")]


def _rows(t) -> np.ndarray:
    return (t.device_rows("cpu").numpy().astype(np.int64) % P).reshape(t.nrows, t.cols)


def _sorted_rows(a: np.ndarray) -> np.ndarray:
    return a[np.lexsort(a.T[::-1])] if len(a) else a


def check_merge(plain_text: str, merged_text: str, merged_name: str = "LK") -> dict:
    from gpu.circuit import parse_circuit

    bad: list[str] = []
    if _head(plain_text) != _head(merged_text):
        bad.append("non-table lines differ")
    return _check_merge(parse_circuit(plain_text), parse_circuit(merged_text), merged_name, bad)


def _check_merge(pc, mc, merged_name: str, bad: list[str]) -> dict:
    one = pc.one_col
    if len(pc.queries) != len(mc.queries):
        bad.append(f"query count {len(pc.queries)} vs {len(mc.queries)}")
    lk = mc.table(merged_name)
    W = lk.cols
    tag: dict[str, int] = {}
    width: dict[str, int] = {}
    for k, (qp, qm) in enumerate(zip(pc.queries, mc.queries)):
        n = len(qp.cols)
        if qm.table != merged_name:
            bad.append(f"q{k}: into {qm.table}")
            continue
        if pc.table(qp.table).cols != n:
            bad.append(f"q{k}: plain width {n} vs table {qp.table} {pc.table(qp.table).cols}")
        if len(qm.cols) != W:
            bad.append(f"q{k}: merged width {len(qm.cols)} != {W}")
            continue
        tt, tk = _lin(qm.cols[1], one)
        if tt or not 1 <= tk < 2**10:
            bad.append(f"q{k}: tag column not a small constant: {tt} {tk}")
            continue
        if tag.setdefault(qp.table, tk) != tk:
            bad.append(f"q{k}: {qp.table} tag {tk} vs {tag[qp.table]}")
        width.setdefault(qp.table, n)
        pt, pk = _lin(qp.cols[0], one)
        mt, mk = _lin(qm.cols[0], one)
        if pt != mt or (pk + tk * TAG_SHIFT) % P != mk:
            bad.append(f"q{k}: key column not plain key + tag 2^20")
        for j in range(1, n):
            if _lin(qp.cols[j], one) != _lin(qm.cols[j + 1], one):
                bad.append(f"q{k}: column {j} differs")
        for j in range(n + 1, W):
            if _lin(qm.cols[j], one) != ((), 0):
                bad.append(f"q{k}: pad column {j} not zero")
    if len(set(tag.values())) != len(tag):
        bad.append(f"tags not injective: {tag}")
    rows = _rows(lk)
    seen = 0
    per_tag = {}
    for name, i in sorted(tag.items(), key=lambda kv: kv[1]):
        n = width[name]
        sel = rows[rows[:, 1] == i]
        seen += len(sel)
        if np.any(sel[:, n + 1:] != 0):
            bad.append(f"{name}: nonzero pad cells in LK")
        got = np.concatenate([((sel[:, :1] - i * TAG_SHIFT) % P), sel[:, 2:n + 1]], axis=1)
        want = _rows(pc.table(name))
        same = got.shape == want.shape and np.array_equal(_sorted_rows(got), _sorted_rows(want))
        if not same:
            bad.append(f"{name}: LK rows of tag {i} != plain rows ({got.shape} vs {want.shape})")
        per_tag[name] = {"tag": i, "rows": int(len(sel)), "width": n, "max_key": int(want[:, 0].max())}
    if seen != len(rows):
        bad.append(f"LK has {len(rows) - seen} rows with an unused tag")
    # LogUp needs char p > number of lookups (Hab22 Lemma 5); report the per-unit count for the caller's bound
    return {"ok": not bad, "problems": bad[:40], "queries_per_unit": len(mc.queries), "lk_rows": int(len(rows)),
            "lk_width": W, "tables": per_tag, "plain_tables_used": sorted(tag)}


def check_paired(plain_text: str, paired_text: str, widths=(3, 5, 6, 7)) -> dict:
    """``paired_text`` = ``plain_text`` with pairs of R<b> queries x, y as (x + 2^b y, x, y) into PR<b> (rows exactly that)."""
    from gpu.circuit import parse_circuit

    bad: list[str] = []
    if _head(plain_text) != _head(paired_text):
        bad.append("non-table lines differ")
    return _check_paired(parse_circuit(plain_text), parse_circuit(paired_text), widths, bad)


def _check_paired(pc, qc, widths, bad: list[str]) -> dict:
    one = pc.one_col
    other_p = Counter((q.table, tuple(_lin(l, one) for l in q.cols)) for q in pc.queries if _rw(q.table) not in widths)
    other_q = Counter((q.table, tuple(_lin(l, one) for l in q.cols)) for q in qc.queries
                      if not q.table.startswith("PR") and _rw(q.table) not in widths)
    if other_p != other_q:
        bad.append("unpaired queries differ")
    out = {}
    for b in widths:
        rp = Counter(_lin(q.cols[0], one) for q in pc.queries if q.table == f"R{b}")
        rq = Counter(_lin(q.cols[0], one) for q in qc.queries if q.table == f"R{b}")
        npair = 0
        for q in qc.queries:
            if q.table != f"PR{b}":
                continue
            npair += 1
            if len(q.cols) != 3:
                bad.append(f"PR{b}: width {len(q.cols)}")
                continue
            k, x, y = (_lin(l, one) for l in q.cols)
            s = dict(x[0])
            for i, c in y[0]:
                s[i] = (s.get(i, 0) + c * 2**b) % P
            if k != (tuple(sorted((i, c) for i, c in s.items() if c)), (x[1] + y[1] * 2**b) % P):
                bad.append(f"PR{b}: key column != x + 2^{b} y")
            rq[x] += 1
            rq[y] += 1
        if rp != rq:
            bad.append(f"R{b}: plain range queries != paired components + leftovers")
        if npair:
            t = _rows(qc.table(f"PR{b}"))
            want = np.array([(x + y * 2**b, x, y) for y in range(2**b) for x in range(2**b)], dtype=np.int64)
            if not np.array_equal(_sorted_rows(t), _sorted_rows(want)):
                bad.append(f"PR{b}: rows are not {{(x + 2^b y, x, y) : x, y < 2^b}}")
        out[f"R{b}"] = {"plain": sum(rp.values()), "pairs": npair, "leftover": sum(Counter(
            _lin(q.cols[0], one) for q in qc.queries if q.table == f"R{b}").values())}
    return {"ok": not bad, "problems": bad[:40], "per_width": out}


# ---------------------------------------------------------------------------------------------------------------------------------
# constraint-system level (gpu.nvf4.circuit): aux rows resolved through their definitions
# ---------------------------------------------------------------------------------------------------------------------------------

class Sys:
    def __init__(self, ctx):
        self.ctx = ctx
        self.rows = ctx.sys.rows
        self.aux_def = {op[1]: op[2] for op in ctx.sys.program if op[0] == "aux"}
        self.memo: dict[int, tuple] = {}

    def lin(self, e_lin: dict, c: int) -> tuple:
        acc: dict[tuple, int] = {}
        k = c % P
        for j, v in e_lin.items():
            r = self.rows[j]
            if r.kind == "aux":
                terms, kk = self.memo.get(j) or self.memo.setdefault(j, self.expr(self.aux_def[j]))
                k = (k + v * kk) % P
                for t, cc in terms:
                    acc[t] = (acc.get(t, 0) + v * cc) % P
            else:
                t = (r.kind, r.name)
                acc[t] = (acc.get(t, 0) + v) % P
        return tuple(sorted((t, cc) for t, cc in acc.items() if cc)), k

    def expr(self, e) -> tuple:
        return self.lin(e.lin, e.c)

    def names(self) -> Counter:
        return Counter((r.kind, r.name) for r in self.rows if r.kind != "aux")

    def aux_defs_ok(self) -> list[str]:
        """every aux row a has exactly one linear constraint a - def(a) = 0"""
        want = {}
        for j, e in self.aux_def.items():
            d = {j: 1}
            for i, v in e.lin.items():
                d[i] = (d.get(i, 0) - v) % P
            want[j] = (tuple(sorted((i, v) for i, v in d.items() if v)), (-e.c) % P)
        have = Counter()
        for l in self.ctx.sys.linear:
            key = (tuple(sorted((i, v % P) for i, v in l.lin.items() if v % P)), l.c % P)
            for j, w in want.items():
                if key == w or key == (tuple((i, (-v) % P) for i, v in w[0]), (-w[1]) % P):
                    have[j] += 1
        return [f"aux {self.rows[j].name}: {have[j]} definitions" for j in want if have[j] < 1]

    def parts(self) -> dict:
        s = self.ctx.sys
        prods = Counter()
        for r in self.rows:
            if r.kind == "prod":
                a, b = r.spec
                prods[(r.name, tuple(sorted([self.expr(a), self.expr(b)])))] += 1
        lin = Counter(x for x in (self.lin(l.lin, l.c) for l in s.linear) if x != ((), 0))
        quad = Counter((tuple(sorted([self.expr(q.a), self.expr(q.b)])), self.expr(q.c)) for q in s.quadratic)
        rng = Counter((b, self.expr(e)) for b, e in self.ctx.ranges)
        nat = Counter((n.table, self.expr(n.key), tuple(self.rows[i].name for _, i in n.cols)) for n in self.ctx.nat)
        return {"prods": prods, "linear": lin, "quadratic": quad, "ranges": rng, "nat": nat,
                "outputs": {k: self.expr(v) for k, v in s.outputs.items()}, "chain": json.dumps(s.chain, default=str)}


def _cmp(a: dict, b: dict, skip=()) -> list[str]:
    return [f"{k} differ" + (f" (+{sum((a[k] - b[k]).values())} / -{sum((b[k] - a[k]).values())})" if isinstance(a[k], Counter) else "")
            for k in a if k not in skip and a[k] != b[k]]


def static_nvf4(paired: bool) -> dict:
    from gpu.nvf4 import circuit as C

    res: dict = {}
    # merge: the same systems, exported unmerged / merged
    u, e = C.build_unit(), C.build_epilogue()
    res["merge_unit"] = check_merge(C.export_circuit(u, merge=False).text, C.export_circuit(u, merge=True).text, C.MERGED)
    ep = C.export_circuit(e, merge=False).text
    res["epilogue"] = {"queries": sum(1 for ln in ep.splitlines() if ln.startswith("query")),
                       "asserts": [ln.split()[1] for ln in ep.splitlines() if ln.startswith("assert")]}
    # flatten: max_depth 99 (no aux columns) vs the default
    u0, u1 = Sys(C.build_unit(max_depth=99)), Sys(C.build_unit())
    pr = []
    if u0.names() != u1.names():
        pr.append("non-aux rows differ")
    pr += u0.aux_defs_ok() + u1.aux_defs_ok()
    pr += _cmp(u0.parts(), u1.parts())
    ex1 = C.export_circuit(u1.ctx)
    res["flatten"] = {"ok": not pr, "problems": pr[:40], "aux_rows": [len(u0.aux_def), len(u1.aux_def)],
                      "max_depth": [C.export_circuit(u0.ctx).stats["max_depth"], ex1.stats["max_depth"]],
                      "columns": ex1.stats["columns"], "wires": ex1.stats["wires"]}
    if paired:
        # BOOL_QUADRATIC: every R1 range becomes exactly one e e = e (unflattened systems, the flatten is checked above)
        C.BOOL_QUADRATIC = False
        b0 = Sys(C.build_unit(max_depth=99))
        C.BOOL_QUADRATIC = True
        b1 = Sys(C.build_unit(max_depth=99))
        p0, p1 = b0.parts(), b1.parts()
        pr = []
        if b0.names() != b1.names():
            pr.append("non-aux rows differ")
        pr += b0.aux_defs_ok() + b1.aux_defs_ok()
        pr += _cmp(p0, p1, skip=("quadratic", "ranges"))
        gone = p0["ranges"] - p1["ranges"]
        new_q = p1["quadratic"] - p0["quadratic"]
        if p1["ranges"] - p0["ranges"]:
            pr.append("new ranges")
        if p0["quadratic"] - p1["quadratic"]:
            pr.append("quadratics dropped")
        if any(b != 1 for b, _ in gone):
            pr.append("non-R1 ranges dropped")
        as_q = Counter(((ex, ex), ex) for _, ex in gone.elements())
        if as_q != new_q:
            pr.append(f"R1 ranges ({sum(gone.values())}) != new e e = e quadratics ({sum(new_q.values())})")
        res["bool_quadratic"] = {"ok": not pr, "problems": pr[:40], "r1_removed": sum(gone.values()),
                                 "quadratics_added": sum(new_q.values())}
        # PAIRED: the same (bool-quadratic, flattened) system exported with and without pairing, unmerged
        keep = C.PAIRED
        C.PAIRED = ()
        t0 = C.export_circuit(u1.ctx, merge=False).text
        C.PAIRED = keep
        t1 = C.export_circuit(u1.ctx, merge=False).text
        res["paired"] = check_paired(t0, t1, keep)
        res["merge_paired_unit"] = check_merge(t1, C.export_circuit(u1.ctx, merge=True).text, C.MERGED)
    res["ok"] = all(v.get("ok", True) for v in res.values() if isinstance(v, dict))
    return res


def selftest(plain_text: str, merged_text: str, paired_plain: str | None = None, paired_text: str | None = None) -> dict:
    """planted defects the static checks must catch (non-vacuity): each entry must come out ``caught``"""
    import copy

    from gpu.circuit import Lin, parse_circuit

    pc0, mc0 = parse_circuit(plain_text), parse_circuit(merged_text)
    one = pc0.one_col
    out = {}

    def run(name, mutate, check):
        pc, mc = copy.deepcopy(pc0), copy.deepcopy(mc0)
        mutate(pc, mc)
        for t in mc.tables + pc.tables:
            t._dev_rows = {}
        out[name] = "caught" if not check(pc, mc)["ok"] else "MISSED"

    merged = lambda pc, mc: _check_merge(pc, mc, "LK", [])          # noqa: E731
    tags = sorted({_lin(q.cols[1], one)[1] for q in mc0.queries})

    def q_tag(pc, mc):
        q = mc.queries[len(mc.queries) // 2]
        t = _lin(q.cols[1], one)[1]
        q.cols[1] = Lin([], next(x for x in tags if x != t))

    def row_tag(pc, mc):
        lk = mc.table("LK")
        lk.rows[len(lk.rows) // 2][1] = next(x for x in tags if x != lk.rows[len(lk.rows) // 2][1])

    def row_drop(pc, mc):
        del mc.table("LK").rows[len(mc.table("LK").rows) // 3]

    def key_shift(pc, mc):
        q = mc.queries[len(mc.queries) // 3]
        q.cols[0] = Lin(list(q.cols[0].terms), (q.cols[0].konst + TAG_SHIFT) % P)

    def pad_one(pc, mc):
        q = next(q for q in mc.queries if all(_lin(l, one) == ((), 0) for l in q.cols[-1:]))
        q.cols[-1] = Lin([], 1)

    def out_col(pc, mc):
        q = next(q for q in mc.queries if _lin(q.cols[2], one)[0])
        q.cols[2] = Lin(list(q.cols[2].terms), (q.cols[2].konst + 1) % P)

    for n, f in (("query tag changed", q_tag), ("LK row tag changed", row_tag), ("LK row dropped", row_drop),
                 ("query key shifted by 2^20", key_shift), ("pad column nonzero", pad_one), ("output column +1", out_col)):
        run(n, f, merged)
    if paired_text is not None:
        pp0, qq0 = parse_circuit(paired_plain), parse_circuit(paired_text)

        def prun(name, mutate):
            pp, qq = copy.deepcopy(pp0), copy.deepcopy(qq0)
            mutate(qq)
            for t in qq.tables:
                t._dev_rows = {}
            out[name] = "caught" if not _check_paired(pp, qq, (3, 5, 6, 7), [])["ok"] else "MISSED"

        def pr_row(qq):
            t = qq.table("PR5")
            t.rows[3][1] = (t.rows[3][1] + 32) % P

        def pr_swap(qq):
            q = next(q for q in qq.queries if q.table == "PR5")
            q.cols[1], q.cols[2] = q.cols[2], q.cols[1]

        def pr_drop(qq):
            i = next(i for i, q in enumerate(qq.queries) if q.table == "PR3")
            del qq.queries[i]

        for n, f in (("PR5 row x out of range", pr_row), ("PR5 query x/y swapped", pr_swap), ("PR3 query dropped", pr_drop)):
            prun(n, f)
    return {"ok": all(v == "caught" for v in out.values()), "cases": out}


def selftest_nvf4(paired: bool) -> dict:
    """planted defects in the constraint systems (flatten; bool quadratic) the Sys comparison must catch"""
    from gpu.nvf4 import circuit as C

    out = {}
    u0 = Sys(C.build_unit(max_depth=99))
    p0 = u0.parts()

    def flat_case(name, mutate):
        u1 = Sys(C.build_unit())
        mutate(u1)
        u1.memo.clear()
        pr = u1.aux_defs_ok() + _cmp(p0, u1.parts())
        out[name] = "caught" if pr else "MISSED"

    def drop_def(u1):
        j = max(u1.aux_def)
        nm = u1.rows[j].name + ".def"
        u1.ctx.sys.linear = [l for l in u1.ctx.sys.linear if not l.name.endswith(nm)]

    def wrong_operand(u1):
        r = next(r for r in u1.rows if r.kind == "prod" and any(u1.rows[j].name.startswith("flat.") for j in r.spec[0].lin))
        a, b = r.spec
        r.spec = (b, b)

    def drop_quad(u1):
        u1.ctx.sys.quadratic.pop(len(u1.ctx.sys.quadratic) // 2)

    def drop_range(u1):
        u1.ctx.ranges.pop(len(u1.ctx.ranges) // 2)

    for n, f in (("flatten def dropped", drop_def), ("flattened product operand swapped", wrong_operand),
                 ("quadratic dropped", drop_quad), ("range dropped", drop_range)):
        flat_case(n, f)
    return {"ok": all(v == "caught" for v in out.values()), "cases": out}


# ---------------------------------------------------------------------------------------------------------------------------------
# forgeries
# ---------------------------------------------------------------------------------------------------------------------------------

def _cheating_prover():
    """the LogUp self-checks off; multiplicities by first-column match only (what a cheating prover sends for a key collision)"""
    import torch

    from gpu import logup, logup_packed

    def mults(t, trows, values):
        m = torch.zeros((trows.shape[0],), dtype=torch.int64, device=trows.device)
        keys = trows[:, 0]
        order = torch.argsort(keys)
        skeys = keys[order]
        for v in values:
            idx = order[torch.searchsorted(skeys, v[:, 0].contiguous()).clamp(max=skeys.shape[0] - 1)]
            m += torch.bincount(idx, minlength=trows.shape[0])
        return m

    logup.multiplicities = mults
    n = 0
    for mod in (logup, logup_packed):
        src = Path(mod.__file__).read_text()
        n += src.count('raise ValueError("LogUp: fractional sum is not zero')
    return n


def _write_rows(a, path: Path) -> None:
    import struct

    a = np.asarray(a, dtype=np.int64) % P
    if a.ndim == 1:
        a = a[:, None]
    with open(path, "wb") as f:
        f.write(struct.pack("<QQ", a.shape[0], a.shape[1]))
        f.write(a.astype("<u4").tobytes())


def _honest(relation: str, vus: int, stmt: Path, plain: Path):
    import torch

    if relation == "fp8-ada":
        import bench_result as br
        from gpu.v2.export import circuits
        from gpu.v2.fp8 import relation_params
        from gpu.v2.witness import Generator, Ops

        circuits(stmt, "ada_e4m3_m16n8k32", merge=True)
        circuits(plain, "ada_e4m3_m16n8k32", merge=False)
        src = Path(__file__).resolve().parents[3]
        x, w, y, rel = br.load_relation(relation, src, 0, vus, 16)
        p, _ = relation_params(rel)
        ops = Ops("cuda")
        rows = Generator(ops, p).run(ops.asarray(x), ops.asarray(w), ops.asarray(y))
        pub = np.array([[int(v)] for v in y], dtype=np.int64)
        return torch.as_tensor(rows.units).long().cuda() % P, torch.as_tensor(rows.epilogue).long().cuda() % P, pub
    from gpu.nvf4 import circuit as C
    from gpu.nvf4.witness import NVF4Generator, load_instances

    C.export(stmt)
    plain.mkdir(parents=True, exist_ok=True)
    (plain / "circuit.txt").write_text(C.export_circuit(C.build_unit(), merge=False).text)
    st = C.statement()
    gen = NVF4Generator(st, "cuda")
    A, B, Y = load_instances(0, vus, 1)
    r = gen.run(torch.as_tensor(A).to(gen.dev).long(), torch.as_tensor(B).to(gen.dev).long(), torch.as_tensor(Y).to(gen.dev))
    if bool(r.bad.any()):
        raise SystemExit("honest VUs rejected by the generator")
    return r.units.long() % P, r.epilogue.long() % P, r.public.cpu().numpy().astype(np.int64)


def _lin_val(l, row) -> int:
    return (l.konst + sum(c * int(row[i]) for i, c in l.terms)) % P


def _var(l, one: int) -> list[tuple[int, int]]:
    return [(i, c) for i, c in l.terms if i != one and c % P]


def _solve(units, U: int, l, target: int, one: int) -> int:
    """set l's first non-constant column in unit U so that l(row) = target; returns the column"""
    col, coef = _var(l, one)[0]
    row = units[U].tolist()
    delta = (target - _lin_val(l, row)) % P
    units[U, col] = (int(row[col]) + delta * pow(coef, P - 2, P)) % P
    return col


def forge_cases(relation: str, uc, pc, units, steps: int) -> dict:
    """{name: (units', what)}: every forgery targets exactly one rewrite; U = VU 3, step 5"""
    U = 3 * steps + 5
    one = uc.one_col
    tag_of = {}
    for qp, qm in zip(pc.queries, uc.queries):
        tag_of.setdefault(qp.table, _lin(qm.cols[1], one)[1])
    by_table: dict[str, list[int]] = {}
    for k, qp in enumerate(pc.queries):
        by_table.setdefault(qp.table, []).append(k)
    cases = {}
    lk = _rows(uc.table("LK"))

    def single(k):
        return len(_var(uc.queries[k].cols[0], one)) >= 1

    # (1) merged-table tag: a query of range table A given the first column of a row of range table B (key + tag_B 2^20);
    #     only the constant tag column still tells them apart
    rng = sorted((t for t in by_table if _rw(t) is not None), key=_rw)
    if len(rng) >= 2:
        a, b = rng[0], rng[-1]
        k = next(k for k in by_table[a] if single(k))
        key_b = 2 ** _rw(a) + 1                                      # a valid B key, out of A's range
        target = (key_b + tag_of[b] * TAG_SHIFT) % P
        u = units.clone()
        col = _solve(u, U, uc.queries[k].cols[0], target, one)
        assert int(lk[(lk[:, 0] == target)][0, 1]) == tag_of[b]
        cases[f"tag.{a}_as_{b}"] = (u, f"q{k} ({a}, tag {tag_of[a]}) first column = LK row ({key_b} + {tag_of[b]} 2^20, tag {tag_of[b]}) of {b}; col {col}")
    # (1b) the same for a row-listed lookup table query given another table's full row (key shifted, outputs copied)
    listed = [t for t in by_table if _rw(t) is None and not t.startswith("PR")]
    for a in listed:
        for b in listed:
            if a == b or pc.table(a).cols != pc.table(b).cols:
                continue
            k = next((k for k in by_table[a] if all(len(_var(l, one)) == 1 for l in uc.queries[k].cols if _var(l, one))), None)
            if k is None:
                continue
            q = uc.queries[k]
            cols_used = [_var(l, one)[0][0] for l in q.cols if _var(l, one)]
            if len(set(cols_used)) != len(cols_used):
                continue
            rb = lk[lk[:, 1] == tag_of[b]]
            rb = rb[len(rb) // 2]
            u = units.clone()
            for j, l in enumerate(q.cols):
                if _var(l, one):
                    _solve(u, U, l, int(rb[j]), one)
            cases[f"tag.{a}_as_{b}"] = (u, f"q{k} ({a}) = LK row {rb.tolist()[:4]}.. of {b} but for the tag column")
            break
        else:
            continue
        break
    # (2) PR<b>: x + 2^b, y - 1 keeps the key column (x + 2^b y); x leaves [0, 2^b)
    for t in sorted(t for t in by_table if t.startswith("PR")):
        b = int(t[2:])
        for k in by_table[t]:
            q = uc.queries[k]
            vx, vy = _var(q.cols[2], one), _var(q.cols[3], one)
            if len(vx) != 1 or len(vy) != 1 or vx[0][1] != 1 or vy[0][1] != 1:
                continue
            (cx, _), (cy, _) = vx[0], vy[0]
            Us = [v * steps + s for v in range(units.shape[0] // steps) for s in range(steps)]
            Uy = next((w for w in Us if int(units[w, cy]) >= 1), None)
            if Uy is None:
                continue
            u = units.clone()
            u[Uy, cx] = (int(u[Uy, cx]) + 2**b) % P
            u[Uy, cy] = (int(u[Uy, cy]) - 1) % P
            cases[f"paired.{t}.carry"] = (u, f"q{k} unit {Uy}: x {int(units[Uy, cx])} -> +2^{b}, y {int(units[Uy, cy])} -> -1 (same key)")
            u = units.clone()
            u[Uy, cx] = (int(u[Uy, cx]) + 2**b * (P - 1)) % P          # x - 2^b wrapped, y + 1: key again unchanged mod p
            u[Uy, cy] = (int(u[Uy, cy]) + 1) % P
            cases[f"paired.{t}.wrap"] = (u, f"q{k} unit {Uy}: x -> x - 2^{b} mod p, y -> y + 1 (same key)")
            break
        break
    # (3) bits as b b = b: a bool column set to 2, and to (p + 1) / 2
    wires = uc.wires
    ncols = uc.ncols
    bool_cols = []
    for name, l in uc.asserts:
        t = dict(l.terms)
        if len(t) != 2:
            continue
        w = next((i for i in t if i >= ncols and t[i] == 1), None)
        if w is None or wires[w][0] != "prod":
            continue
        _, _, a, bb = wires[w]
        c = next(i for i in t if i != w)
        if t[c] == P - 1 and a.terms == [(c, 1)] and bb.terms == [(c, 1)] and not a.konst and not bb.konst:
            bool_cols.append((name, c))
    if relation == "fp4-nvf4" and bool_cols:
        name, c = bool_cols[len(bool_cols) // 2]
        for v, lab in ((2, "two"), ((P + 1) // 2, "half")):
            u = units.clone()
            u[U, c] = v
            cases[f"bool.{lab}"] = (u, f"assert {name}: column {uc.col_names[c]} = {v}")
    # (4) flatten: an aux column (bound to its operand by `<name>.def`) + 1
    aux = [i for i, n in enumerate(uc.col_names) if n.startswith("flat.")]
    if aux:
        c = aux[len(aux) // 2]
        u = units.clone()
        u[U, c] = (int(u[U, c]) + 1) % P
        cases["flatten.aux+1"] = (u, f"column {uc.col_names[c]} + 1")
    return cases


def forge(relation: str, vus: int, out: Path, verifier: str, threads: int) -> dict:
    import subprocess

    import torch

    from gpu import prover
    from gpu.circuit import layers, load_circuit
    from gpu.prover import VerifyError
    from gpu.run import read_chain

    out.mkdir(parents=True, exist_ok=True)
    stmt, plain = out / "stmt", out / "plain"
    units, epi, pub = _honest(relation, vus, stmt, plain)
    man = json.loads((stmt / "manifest.json").read_text())
    steps = man["steps"]
    uc, ec = load_circuit(stmt / "circuit.txt"), load_circuit(stmt / "epilogue.txt")
    pc = load_circuit(plain / "circuit.txt")
    ul, el = layers(uc), layers(ec)
    chain = read_chain(stmt / "chain.txt", uc, ec, steps, [int(v) for v in pub.reshape(-1)])
    n_checks = _cheating_prover()

    def run(name: str, u, what: str) -> dict:
        inst = prover.Instance([prover.Segment("unit", uc, ul, u, uc.hash), prover.Segment("epilogue", ec, el, epi, ec.hash)], chain)
        d = out / "cases" / name
        d.mkdir(parents=True, exist_ok=True)
        for f in STATEMENT_FILES:
            shutil.copyfile(stmt / f, d / f)
        _write_rows(pub, d / "public.bin")
        _write_rows(u.cpu().numpy(), d / "units.bin")
        _write_rows(epi.cpu().numpy(), d / "epilogue.bin")
        try:
            proof, _ = prover.prove(inst, True)
        except Exception as e:  # noqa: BLE001
            return {"case": name, "what": what, "prover": f"refused: {type(e).__name__}: {str(e)[:120]}"}
        (d / "proof.bin").write_bytes(proof.to_bytes())
        try:
            prover.verify(inst, proof, True)
            py = "accept"
        except VerifyError as e:
            py = f"reject: {str(e)[:120]}"
        r = subprocess.run([verifier, "verify", "--dir", str(d), "--proof", str(d / "proof.bin"), "--vus", str(vus),
                            "--threads", str(threads), "--json", str(d / "verify.json")], capture_output=True, text=True)
        doc = json.loads((d / "verify.json").read_text()) if (d / "verify.json").is_file() else {}
        rust = "accept" if (doc.get("accepted") and r.returncode == 0) else f"reject: {str(doc.get('error') or r.stderr[-200:])[:160]}"
        return {"case": name, "what": what, "python": py, "rust": rust, "rust_rc": r.returncode, "proof_bytes": (d / "proof.bin").stat().st_size}

    res = {"relation": relation, "vus": vus, "self_checks_found": n_checks, "verifier": verifier,
           "circuit_sha256": __import__("hashlib").sha256((stmt / "circuit.txt").read_bytes()).hexdigest(), "results": []}
    res["results"].append(run("honest", units, "honest control"))
    print(json.dumps(res["results"][-1]), flush=True)
    for name, (u, what) in forge_cases(relation, uc, pc, units, steps).items():
        res["results"].append(run(name, u, what))
        print(json.dumps(res["results"][-1]), flush=True)
    forged = [r for r in res["results"] if r["case"] != "honest"]
    h = res["results"][0]
    res["ok"] = (h.get("python") == "accept" and h.get("rust") == "accept" and bool(forged)
                 and all(r.get("rust", "").startswith("reject") and not r.get("python", "").startswith("accept") for r in forged))
    (out / "forge.json").write_text(json.dumps(res, indent=1) + "\n")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="red_team_lk")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("static-merge")
    a.add_argument("--plain", type=Path, required=True)
    a.add_argument("--merged", type=Path, required=True)
    b = sub.add_parser("static-nvf4")
    b.add_argument("--paired", action="store_true")
    s = sub.add_parser("selftest")
    s.add_argument("--plain", type=Path, help="fp8: the --no-merge statement dir (else gpu.nvf4.circuit is exported)")
    s.add_argument("--merged", type=Path)
    s.add_argument("--paired", action="store_true")
    c = sub.add_parser("forge")
    c.add_argument("--relation", required=True, choices=("fp8-ada", "fp4-nvf4"))
    c.add_argument("--vus", type=int, default=64)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--verifier", required=True)
    c.add_argument("--threads", type=int, default=12)
    args = ap.parse_args(argv)
    if args.cmd == "static-merge":
        res = check_merge((args.plain / "circuit.txt").read_text(), (args.merged / "circuit.txt").read_text())
        for f in ("epilogue.txt", "chain.txt"):
            res[f"{f}_identical"] = (args.plain / f).read_bytes() == (args.merged / f).read_bytes()
        mp, mm = (json.loads((d / "manifest.json").read_text()) for d in (args.plain, args.merged))
        res["manifest_keys_differing"] = sorted(k for k in set(mp) | set(mm) if mp.get(k) != mm.get(k))
    elif args.cmd == "static-nvf4":
        res = static_nvf4(args.paired)
    elif args.cmd == "selftest":
        if args.plain is not None:
            res = {"merge": selftest((args.plain / "circuit.txt").read_text(), (args.merged / "circuit.txt").read_text())}
        else:
            from gpu.nvf4 import circuit as C

            u = C.build_unit()
            pp = pt = None
            if args.paired:
                keep, C.PAIRED = C.PAIRED, ()
                pp = C.export_circuit(u, merge=False).text
                C.PAIRED = keep
                pt = C.export_circuit(u, merge=False).text
            res = {"merge": selftest(C.export_circuit(u, merge=False).text, C.export_circuit(u, merge=True).text, pp, pt),
                   "systems": selftest_nvf4(args.paired)}
        res["ok"] = all(v["ok"] for v in res.values())
    else:
        res = forge(args.relation, args.vus, args.out, args.verifier, args.threads)
    print(json.dumps(res, indent=1, default=str))
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
