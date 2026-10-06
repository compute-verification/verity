"""V*'s statements over one inner session (hiding model architectures, lane rec-v0): each Ligerito level's `RecOpen`
statement (`verity_flock.rec_outer`) and the algebra's `InnerRepCheck` (`verity_flock.rec_residuals`), staged from the inner
session's directory (`verity_flock.rec_live.Proxy.write`: ``public.json``, ``private.json``, ``session/``). The algebra is
staged as its parts (`rec_residuals.parts`, ``alg/p<j>/``), each within --zk's coin schedule, and holds when every part does.

    python -m verity_flock.rec_vstage points INNER CIRCUIT PUBLIC > points.json
    inner-fold [--partition FILE] verity/flock-circuit CIRCUIT PUBLIC points.json os > fold.json    (tests/lean/InnerFold.lean)
    python -m verity_flock.rec_vstage stage INNER fold.json OUT [--levels all|none|0,1,..] [--no-algebra] [--forge KIND]
        [--salts FILE]

**The verifier's side** reads the inner session's public record and the inner statement, never the prover's records: the
table's shape as its setup derives it (`shape`), each rep's coins (its stream's rounds), the fold points they give
(`rec_algebra.fold_points`, ``points``), the extra claims at the link points and the comb at the fold points (``fold.json``,
`InnerFold.lean`: the Lean verifier of record's `Stmt.extraClaims` and `CircuitFold.folded`, no proof read), then each rep's
v, the final check's κ and its query rows (`rec_algebra.verifier_rows`), each opening's coefficients
(`rec_outer.coefficients`, its position checked against `rec_vstar`'s), and the entries its reads files name: per round i that
absorbs message elements, the value ``rec-c<i>`` rebuilt from that round's two ``c_k`` (`rec_residuals.commits_entry`), and
each level's committed tops, rebuilt from their recorded ``b || c`` (`rec_outer.TopValue`). **The prover's side**: the openings
from its proofs, its tops with the salts the gateway committed them under (`rec_vstar.tops`), its running sums
(`rec_outer.Chain`, under salts drawn once and kept in ``--salts FILE``, so that every statement staged with one FILE reads one
chain), and its message rows with the salts it committed them under in the inner session, registered as ``rec-c<i>`` (row r:
rep r's) under `rec_outer.session_program` of ``public.json``, which the chain shares.

**Forgeries** (``--forge``), staged under ``OUT/forged-<kind>/``, each beside the verifier's files of the honest session:

* ``message``: the prover opens rep 0's first message round to another row (one bit of its first word), under the salt and
  the ``c_k`` it committed: the verifier's ``pub.bin`` holds the honest ``c_k``, the prover's ``inst.bin`` the other row
  (the part that reads ``m0``).
* ``sum``: the prover's running sum after rep 0's last opening changed (one bit of its first word), the rows either side of
  it honest: the openings that read it (rep 0's last as ``acc_out``, rep 1's first as ``acc``: ``L<level>/``) and the
  algebra (rep 0's ``acc1``, rep 1's ``acc0``: the part that reads them).
* ``comb``: the verifier's comb for rep 0 one bit off the inner circuit's fold, as if the inner proof were of another circuit
  (the part whose v holds it).
* ``top``: a wrong top. The prover commits the last level's first opening with its first sibling one bit off (a fresh row it
  commits honestly), so the opening climbs to another node, and gives that node as its output row under the ``b || c`` the
  gateway committed for the top node its position names (``L<last>/``).
* ``top-salt``: a top that does not match its salt. Level 0's first opening, honest but for its output row's salt, one bit off
  the salt the gateway committed ``root_B``'s node under (``L0/``).

In both, the opening climbs to the output row the prover gives, and that row does not open the committed ``b || c``.

* ``coef``: level 0's first opening with the prover's ``coef`` row one bit off (bit 0 of its first word), under the salt the
  verifier registered it with: the verifier's ``pub.bin`` holds its own ``b || c`` of ``rec-coef`` (``L0/``).
* ``dirs``: the same for its ``dirs`` row, so the opening's lowest level takes the other side (``L0/``).
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

from verity_catalog.definitions.boolean import gf2k as G

from . import rec_algebra as A
from . import rec_open as R
from . import rec_outer as O
from . import rec_residuals as RR
from . import rec_vstar as V

TABLE = "circuit"
FORGERIES = ("message", "sum", "comb", "top", "top-salt", "coef", "dirs")
VERIFIER_FILES = ("circuit.txt", "pub.bin", "program.json", "partition.json", "registered.json")


def meta(circuit: Path) -> dict:
    with open(circuit) as f:
        f.readline()
        return json.loads(f.readline()[5:])


def instances(pub: Path) -> int:
    with open(pub, "rb") as f:
        return int(json.loads(f.readline())["instances"])


def shape(meta_: dict, n: int, claims: int = 2) -> A.Shape:
    """The inner table's shape as the verifier's setup derives it (`Stmt.setup`): ``m = k_log + max(3, ⌈log2 blocks⌉)`` for
    ``n`` instances in blocks of ``vus_per_block``."""
    blocks = max(1, -(-n // meta_["vus_per_block"]))
    return A.Shape(meta_["k_log"] + max(3, (blocks - 1).bit_length()), meta_["k_log"], claims)


def _stream(public: dict, rep: int) -> list[dict]:
    s = f"{TABLE}/rep{rep}"
    return sorted((r for r in public["rounds"] if r["stream"] == s), key=lambda r: r["k"])


def rep_coins(public: dict, rep: int) -> list[list[int]]:
    """A rep's coins by round, as the coin server issued them."""
    return [[G.of_hex(c) for c in r["coins"]] for r in _stream(public, rep)]


def link_points(public: dict) -> list[list[str]]:
    """The link points the coin server drew (``points × m`` coins), one list of ``m`` a point."""
    spec, flat = public["hello"]["link"], public["link"]["points"]
    if len(flat) != spec["points"] * spec["m"]:
        raise ValueError(f"{len(flat)} link coins for {spec['points']} points of {spec['m']}")
    return [flat[i * spec["m"]:(i + 1) * spec["m"]] for i in range(spec["points"])]


def points(public: dict, meta_: dict, n: int) -> dict:
    """``InnerFold.lean``'s POINTS: the link points and each rep's fold points, from the coins alone."""
    sh = shape(meta_, n)
    reps = []
    for r in range(public["hello"]["reps"]):
        f = A.fold_points(sh, rep_coins(public, r))
        reps.append({"alpha": G.hex128(f["alpha"]), "beta": G.hex128(f["beta"]),
                     **{k: [G.hex128(x) for x in f[k]] for k in ("lw", "rho_in", "ts")}})
    return {"m": sh.m, "k_log": sh.k_log, "link": link_points(public), "reps": reps}


def verifier(public: dict, fold: dict, pts: dict | None = None) -> SimpleNamespace:
    """The verifier's per-rep values: the shape, the extra claims, and each rep's coins, comb, v and `verifier_rows`' info.
    ``pts``: the POINTS ``fold`` was computed at, whose shape it must have."""
    if pts is not None and (fold["m"], fold["k_log"]) != (pts["m"], pts["k_log"]):
        raise ValueError(f"InnerFold's shape {fold['m'], fold['k_log']} is not the points' {pts['m'], pts['k_log']}")
    claim = lambda e: A.Claim(G.of_hex(e["value"]), tuple(map(G.of_hex, e["skip"])), tuple(map(G.of_hex, e["x"])))
    extras = [claim(e) for e in fold["extras"]]
    sh = A.Shape(fold["m"], fold["k_log"], 2 + len(extras))
    reps = []
    for r, d in enumerate(fold["reps"]):
        rounds = rep_coins(public, r)
        comb = [G.of_hex(x) for x in d["comb"]]
        if len(comb) != 64:
            raise ValueError(f"rep {r}: a comb of {len(comb)}")
        v, info = A.verifier_rows(sh, rounds, extras, comb)
        reps.append(SimpleNamespace(rounds=rounds, comb=comb, v=v, info=info))
    return SimpleNamespace(sh=sh, extras=extras, reps=reps)


def message_rounds(public: dict, private: dict, sh: A.Shape) -> list[dict]:
    """Per round that absorbs message elements (`rec_algebra.rounds`), both reps' ``c_k`` (the verifier's) and their rows and
    salts (the prover's), checked to be one row a round of the algebra's element counts."""
    prv = {(p["stream"], p["k"]): p for p in private["rounds"]}
    want = [len(x) for x in A.rounds(sh)]
    per = []
    for r in range(2):
        rs = [x for x in _stream(public, r) if x["commits"]]
        ps = [prv[(x["stream"], x["k"])] for x in rs]
        if [len(p["elements"]) for p in ps] != want or any(len(x["commits"]) != 1 or len(p["rows"]) != 1 for x, p in zip(rs, ps)):
            raise ValueError(f"rep {r}: its message rounds are not one row a round of {want} elements")
        per.append([(bytes.fromhex(x["commits"][0]), p["rows"][0], bytes.fromhex(p["salts"][0])) for x, p in zip(rs, ps)])
    return [{"commits": [per[r][i][0] for r in range(2)], "rows": [per[r][i][1] for r in range(2)],
             "salts": [per[r][i][2] for r in range(2)]} for i in range(len(want))]


def messages(program: str, mrs: list[dict]) -> tuple[dict, dict]:
    """The prover's registrations ``rec-c<i>`` and the verifier's own entries of them (from the ``c_k`` alone), by port."""
    from verity.protocols.verification.sampled_proofs.one_stage import registered as RG

    regs, own = {}, {}
    for i, mr in enumerate(mrs):
        value = f"rec-c{i}"
        reg = RG.Registered.commit(program, value, mr["rows"], mr["salts"])
        own[f"m{i}"] = RR.commits_entry(program, value, len(mr["rows"][0]), mr["commits"])
        if own[f"m{i}"]["root"] != reg.entry()["root"]:
            raise ValueError(f"round {i}: the prover's rows are not the rows its c_k commit")
        regs[f"m{i}"] = reg
    return regs, own


def coefficients(vf: SimpleNamespace, queries: list[V.Query]) -> dict:
    """Each opening's ``coef`` words at its rep's coins, its position the one the verifier's query coins give."""
    out = {}
    for q in queries:
        info = vf.reps[q.rep].info
        pos, ea = info["queries"][q.level]
        if pos[q.j] != q.pos:
            raise ValueError(f"rep {q.rep} level {q.level} query {q.j}: position {q.pos}, the verifier's coins give {pos[q.j]}")
        out[(q.rep, q.level, q.j)] = O.coefficients(info["kappa"], ea, q.level, q.j)
    return out


def acc_salts(path: Path, n: int) -> list[bytes]:
    """The running sums' ``n`` salts: drawn when ``path`` does not exist and kept there (the prover's), read back otherwise."""
    from verity.primitives.commitments import hm96

    k = hm96.SHA512.salt_bytes
    if not path.exists():
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(b"".join(hm96.SHA512.fresh_salt() for _ in range(n)))
        tmp.rename(path)
    data = path.read_bytes()
    if len(data) != n * k:
        raise ValueError(f"{path}: {len(data)} bytes, not {n} salts of {k}")
    return [data[i * k:(i + 1) * k] for i in range(n)]


def forge_chain(chain: O.Chain, row: int) -> O.Chain:
    """The prover's running sums with row ``row`` one bit off (its first word), every other row and every salt honest."""
    from verity.protocols.verification.sampled_proofs.one_stage import registered as RG

    rows = [list(r) for r in chain.reg.rows]
    rows[row][0] ^= 1
    return O.Chain(RG.Registered.commit(chain.reg.program, O.ACC_VALUE, rows, list(chain.reg.salts)), dict(chain.at))


def _offsets(data: bytes, circuit_text: str, port: str, row: int) -> tuple[int, int]:
    """Where table row ``row`` of ``port`` and its salt start in `circuit.write`'s prover file with shared rows: the header,
    each row port's table rows in the circuit's port order (u16 little-endian), then their salts in that order."""
    nl = data.index(b"\n") + 1
    head = json.loads(data[:nl])
    ports = json.loads(circuit_text.split("\n", 2)[1][5:])["ports"]
    if port not in [p[0] for p in ports]:
        raise ValueError(f"no row port {port}")
    rows_at, salts_at = nl, nl + sum(2 * head["shared_rows"][name] * words for name, words, *_ in ports)
    for name, words, *_ in ports:
        if name == port:
            return rows_at + 2 * row * words, salts_at + V.SALT * row
        rows_at += 2 * head["shared_rows"][name] * words
        salts_at += V.SALT * head["shared_rows"][name]
    raise AssertionError


def _patch(inst: Path, at: int, want: bytes, new: bytes, what: str) -> None:
    data = bytearray(inst.read_bytes())
    if bytes(data[at:at + len(want)]) != want or len(new) != len(want):
        raise AssertionError(f"{inst}: {what} is not where the layout puts it")
    data[at:at + len(new)] = new
    inst.write_bytes(bytes(data))


def _u16(words: list[int]) -> bytes:
    return b"".join(int(w).to_bytes(2, "little") for w in words)


def forge_row(inst: Path, circuit_text: str, port: str, row: int, word: int, want: int) -> None:
    """Flip bit 0 of word ``word`` of the prover's table row ``row`` of ``port`` in ``inst``, whose value there must be
    ``want``; the header, the salts and every ``b || c`` are left as they were."""
    at = _offsets(inst.read_bytes(), circuit_text, port, row)[0] + 2 * word
    _patch(inst, at, _u16([want]), _u16([want ^ 1]), f"{port} row {row} word {word}")


def set_row(inst: Path, circuit_text: str, port: str, row: int, want: list[int], new: list[int]) -> None:
    """Replace the prover's table row ``row`` of ``port`` (its words ``want``) with ``new``, every salt and ``b || c`` kept."""
    _patch(inst, _offsets(inst.read_bytes(), circuit_text, port, row)[0], _u16(want), _u16(new), f"{port} row {row}")


def set_salt(inst: Path, circuit_text: str, port: str, row: int, want: bytes, new: bytes) -> None:
    """Replace the prover's salt of table row ``row`` of ``port`` (``want``) with ``new``, every row and ``b || c`` kept."""
    _patch(inst, _offsets(inst.read_bytes(), circuit_text, port, row)[1], want, new, f"{port} row {row}'s salt")


def forged_query(q: V.Query) -> tuple[V.Query, bytes]:
    """``top``'s opening: ``q`` with its first sibling one bit off (bit 0 of its first byte), and the node it climbs to."""
    sibs = [bytes([q.sibs[0][0] ^ 1]) + q.sibs[0][1:], *q.sibs[1:]]
    return V.Query(**{**q.__dict__, "sibs": sibs}), V.climb(V.leaf(q.row), q.pos, sibs)


def forge_top(d: Path, level: dict, tops: O.TopValue, kind: str, i: int = 0) -> dict:
    """The prover's ``inst.bin`` of a level's statement under ``d`` with opening ``i``'s output row forged as ``kind`` names
    (the module docstring): for ``top``, the level as staged (its query ``i`` `forged_query`'s), and the output row the node
    that query climbs to; for ``top-salt``, the honest row under another salt. The verifier's files are the staging's."""
    q = level["queries"][i]
    text = (d / "circuit.txt").read_text()
    at = tops.position(q)
    rec = {"kind": kind, "level": level["level"], "instance": i, "rep": q.rep, "j": q.j, "top_row": at,
           "committed_node": q.cap_node.hex()}
    if kind == "top":
        node = V.climb(V.leaf(q.row), q.pos, q.sibs)
        if node == q.cap_node:
            raise ValueError("top: the staged opening climbs to its committed top node")
        set_row(d / "inst.bin", text, "out", i, R.expected(q.cap_node), R.expected(node))
        return {**rec, "node": node.hex()}
    salt = tops.reg.salts[at]
    set_salt(d / "inst.bin", text, "out", i, salt, bytes([salt[0] ^ 1]) + salt[1:])
    return {**rec, "salt_byte0": [salt[0], salt[0] ^ 1]}


def _copy_verifier(src: Path, dst: Path) -> None:
    for f in VERIFIER_FILES:
        shutil.copyfile(src / f, dst / f)


def stage(inner: Path, fold: dict, out: Path, levels: list[int] | None = None, algebra: bool = True, forge: str | None = None,
          evaluate: bool = True, pts: dict | None = None, salts: Path | None = None) -> dict:
    """V*'s statements over the inner session ``inner``: the levels ``levels`` (None: every level) and, with ``algebra``, the
    algebra, honest under ``out``; with ``forge``, that forgery's statements alone under ``out/forged-<forge>`` (the module
    docstring). ``salts``: the running sums' salts file (`acc_salts`; None: fresh ones). Returns what each statement costs and
    holds, and the verifier's own seconds."""
    inner, out = Path(inner), Path(out)
    t = {}
    t0 = time.perf_counter()
    public_b = (inner / "public.json").read_bytes()
    public, private, proofs = V.load(inner, TABLE)
    vf = verifier(public, fold, pts)
    t["verifier_rows_s"] = round(time.perf_counter() - t0, 2)
    sh = vf.sh
    program = O.session_program(public_b)
    t0 = time.perf_counter()
    qs = V.queries(public, private, proofs, TABLE)
    coefs = coefficients(vf, qs)
    t["coefficients_s"] = round(time.perf_counter() - t0, 2)
    ts = {rep: V.tops(public, private, f"{TABLE}/rep{rep}", TABLE) for rep in sorted(proofs)}
    chain = O.Chain.build(program, qs, coefs, None if salts is None else acc_salts(Path(salts), len(qs) + 1))
    lvls = O.levels(qs)
    res: dict = {"shape": [sh.m, sh.k_log, sh.claims], "openings": len(qs), "chain_rows": chain.reg.leaves,
                 "acc_root": chain.reg.root.hex(), "program": program, "verifier": t}
    p_last = chain.rep_ends(0)[1] - 1
    if forge is not None:
        if forge not in FORGERIES:
            raise ValueError(f"--forge {forge}: one of {FORGERIES}")
        out = out / f"forged-{forge}"
        res["forge"] = forge
        if forge == "sum":
            keys = {k for k, p in chain.at.items() if p in (p_last, p_last + 1)}
            levels = sorted({lv for _, lv, _ in keys})
            res["forged_row"] = p_last + 1
            chain = forge_chain(chain, p_last + 1)
            res["forged_acc_root"] = chain.reg.root.hex()
        elif forge in ("top", "top-salt"):
            levels, algebra = [lvls[-1 if forge == "top" else 0]["level"]], False
        elif forge in ("coef", "dirs"):
            levels, algebra = [lvls[0]["level"]], False
        else:
            levels = []
    want = None if levels is None else set(levels)
    res["levels"] = {}
    for lv in lvls:
        if want is None or lv["level"] in want:
            t0 = time.perf_counter()
            tops = O.TopValue.build(program, lv["level"], {rep: t_[lv["level"]] for rep, t_ in ts.items()})
            if forge == "top":
                lv = {**lv, "queries": [forged_query(lv["queries"][0])[0], *lv["queries"][1:]]}
            s = O.stage(lv, out / f"L{lv['level']}", chain, coefs, tops)
            if forge in ("top", "top-salt"):
                res["forged_top"] = forge_top(out / f"L{lv['level']}", lv, tops, forge)
            if forge in ("coef", "dirs"):
                d = out / f"L{lv['level']}"
                q = lv["queries"][0]
                want_ = coefs[(q.rep, q.level, q.j)][0] if forge == "coef" else R.dirs_words(q.pos, lv["H"])[0]
                forge_row(d / "inst.bin", (d / "circuit.txt").read_text(), forge, 0, 0, want_)
                res[f"forged_{forge}"] = {"level": lv["level"], "instance": 0, "rep": q.rep, "j": q.j, "word": 0, "bit": 0}
            s["stage_s"] = round(time.perf_counter() - t0, 1)
            res["levels"][f"L{lv['level']}"] = s
    if algebra:
        t0 = time.perf_counter()
        mrs = message_rounds(public, private, sh)
        regs, own = messages(program, mrs)
        t["entries_s"] = round(time.perf_counter() - t0, 2)
        a0, a1 = zip(*(chain.rep_ends(r) for r in range(2)))
        reads = {nm: (regs[nm], [0, 1]) for nm in regs}
        reads["acc0"], reads["acc1"] = (chain.reg, list(a0)), (chain.reg, list(a1))
        vs = [list(r.v) for r in vf.reps]
        if forge == "comb":
            vs[0][-64] ^= 1
        res["algebra"] = []
        for j, p in enumerate(RR.parts(RR.spec(sh))):
            st = RR.statement(p.S, f"rec-residuals/m{sh.m}-k{sh.k_log}-c{sh.claims}-p{j}", 2)
            names = [nm for nm, _ in p.S["ports"]]
            d = out / "alg" / f"p{j}"
            s = RR.write(st, {nm: reads[nm] for nm in names}, [[v[i] for i in p.v] for v in vs], d, evaluate=evaluate,
                         own={nm: own[nm] for nm in names if nm in own})
            if forge == "message" and "m0" in names:
                forge_row(d / "inst.bin", st.text, "m0", 0, 0, mrs[0]["rows"][0][0])
                res["forged_message"] = {"part": j, "port": "m0", "rep": 0, "word": 0, "bit": 0}
            res["algebra"].append({**s, "part": j, "of_residuals": p.res, "regions": RR.regions(p.S["ports"])})
    return res


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m verity_flock.rec_vstage")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("points", help="InnerFold.lean's POINTS from the inner session's public record and the inner statement")
    p.add_argument("inner")
    p.add_argument("circuit")
    p.add_argument("public")
    s = sub.add_parser("stage", help="V*'s statements over the inner session")
    s.add_argument("inner")
    s.add_argument("fold", help="InnerFold.lean's output at `points`' POINTS")
    s.add_argument("out")
    s.add_argument("--points", default=None, help="the POINTS the fold was computed at (its shape is checked)")
    s.add_argument("--levels", default="all", help="all, none, or comma-separated levels")
    s.add_argument("--no-algebra", action="store_true")
    s.add_argument("--forge", default=None, choices=FORGERIES)
    s.add_argument("--no-evaluate", action="store_true", help="skip evaluating the algebra's unit against the reference")
    s.add_argument("--salts", default=None, help="the running sums' salts, drawn and kept there if absent: every statement "
                   "staged with one FILE reads one chain")
    a = ap.parse_args(argv)
    if a.cmd == "points":
        inner = Path(a.inner)
        t0 = time.perf_counter()
        pts = points(json.loads((inner / "public.json").read_text()), meta(Path(a.circuit)), instances(Path(a.public)))
        pts["points_s"] = round(time.perf_counter() - t0, 2)
        print(json.dumps(pts))
        return 0
    levels = None if a.levels == "all" else [] if a.levels == "none" else [int(x) for x in a.levels.split(",")]
    pts = json.loads(Path(a.points).read_text()) if a.points else None
    res = stage(Path(a.inner), json.loads(Path(a.fold).read_text()), Path(a.out), levels, not a.no_algebra, a.forge,
                not a.no_evaluate, pts, a.salts)
    print(json.dumps(res, sort_keys=True), flush=True)
    alg = res.get("algebra", [])
    if any(p["unit_agrees"] not in (None, [True] * p["instances"]) for p in alg):
        return 1
    if a.forge is not None:
        return 0
    held = all(not s["not_opened"] and not s["not_summed"] for s in res["levels"].values())
    return 0 if held and all(all(p["holds"]) for p in alg) else 1


if __name__ == "__main__":
    raise SystemExit(main())
