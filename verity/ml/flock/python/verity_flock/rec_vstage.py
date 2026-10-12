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
from its proofs, its tops with the salts the firewall committed them under (`rec_vstar.tops`), its running sums
(`rec_outer.Chain`, under salts drawn once and kept in ``--salts FILE``, so that every statement staged with one FILE reads one
chain), and its message rows with the salts it committed them under in the inner session, registered as ``rec-c<i>`` (row r:
rep r's) under `rec_outer.session_program` of ``public.json``, which the chain shares.

**Hidden unit circuits** (`verity_flock.rec_session`, ``fold.json``'s ``hidden``: ``InnerFold.lean --hide``): the comb is the
public part's fold, the shape is holo (`rec_algebra.Shape.holo`), and h's round is one of the message rounds, its row
registered as the others' are; z_partial's round draws no coins, which `rec_algebra.Coins` skips.

**The sparse phase and R** (``--sparse``, with ``--circuit``): RecSparse over both reps' sparse streams (`stage_sparse`) and,
when the session committed the stacked table R, R's levels as RecOpen statements, its openings on the chain after both
reps' (rep `STACKED_REP`), and its registered opening (`stage_stacked`), which reads the ``rec-claim`` RecSparse reads and
the stream's messages as ``rec-opening-c<i>`` in the verifier's round order (`opening_messages`).

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
  firewall committed for the top node its position names (``L<last>/``).
* ``top-salt``: a top that does not match its salt. Level 0's first opening, honest but for its output row's salt, one bit off
  the salt the firewall committed ``root_B``'s node under (``L0/``).

In both, the opening climbs to the output row the prover gives, and that row does not open the committed ``b || c``.

* ``coef``: level 0's first opening with the prover's ``coef`` row one bit off (bit 0 of its first word), under the salt the
  verifier registered it with: the verifier's ``pub.bin`` holds its own ``b || c`` of ``rec-coef`` (``L0/``).
* ``dirs``: the same for its ``dirs`` row, so the opening's lowest level takes the other side (``L0/``).
* ``filler``: the inner session's own forgery, a padded inner statement's filler output row one bit off in the inner prover's
  file (`rec_shape.forge_filler`), so the session it gave is of an unsatisfied table. V*'s algebra over it is staged as it
  is, under ``forged-filler/``, and must not hold (the openings are the session's, which the forgery leaves consistent).

``--cap`` (default REC_CAP, `rec_shape.Cap.parse`): the deployment's public cap; a session whose V* is not the cap's shape is
refused before anything is staged (`rec_shape.check`, exit 2).
"""
from __future__ import annotations

import dataclasses
import json
import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

from verity.ml.circuits.ops.boolean import gf2k as G

from . import rec_algebra as A
from . import rec_live as RL
from . import rec_open as R
from . import rec_outer as O
from . import rec_residuals as RR
from . import rec_sparse as SP
from . import rec_vstar as V

TABLE = "circuit"
FORGERIES = ("message", "sum", "comb", "top", "top-salt", "coef", "dirs", "filler")
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
    """The verifier's per-rep values: the shape (holo when ``fold`` names hidden circuits), the extra claims, and each rep's
    coins, comb, v and `verifier_rows`' info. ``pts``: the POINTS ``fold`` was computed at, whose shape it must have."""
    if pts is not None and (fold["m"], fold["k_log"]) != (pts["m"], pts["k_log"]):
        raise ValueError(f"InnerFold's shape {fold['m'], fold['k_log']} is not the points' {pts['m'], pts['k_log']}")
    claim = lambda e: A.Claim(G.of_hex(e["value"]), tuple(map(G.of_hex, e["skip"])), tuple(map(G.of_hex, e["x"])))
    extras = [claim(e) for e in fold["extras"]]
    sh = A.Shape(fold["m"], fold["k_log"], 2 + len(extras), holo=bool(fold.get("hidden")))
    reps = []
    for r, d in enumerate(fold["reps"]):
        rounds = rep_coins(public, r)
        comb = [G.of_hex(x) for x in d["comb"]]
        if len(comb) != 64:
            raise ValueError(f"rep {r}: a comb of {len(comb)}")
        v, info = A.verifier_rows(sh, rounds, extras, comb)
        reps.append(SimpleNamespace(rounds=rounds, comb=comb, v=v, info=info))
    return SimpleNamespace(sh=sh, extras=extras, reps=reps)


def message_rounds(public: dict, private: dict, sh: A.Shape, streams: list[str] | None = None) -> list[dict]:
    """Per round that absorbs message elements (`rec_algebra.rounds`), both reps' ``c_k`` (the verifier's) and their rows and
    salts (the prover's), checked to be one row a round of the algebra's element counts. ``streams``: other streams than
    the reps' (row r stream r's)."""
    prv = {(p["stream"], p["k"]): p for p in private["rounds"]}
    want = [len(x) for x in A.rounds(sh)]
    per = []
    for stream in streams or [f"{TABLE}/rep{r}" for r in range(2)]:
        rs = sorted((x for x in public["rounds"] if x["stream"] == stream and x["commits"]), key=lambda x: x["k"])
        ps = [prv[(x["stream"], x["k"])] for x in rs]
        if [len(p["elements"]) for p in ps] != want or any(len(x["commits"]) != 1 or len(p["rows"]) != 1 for x, p in zip(rs, ps)):
            raise ValueError(f"{stream}: its message rounds are not one row a round of {want} elements")
        per.append([(bytes.fromhex(x["commits"][0]), p["rows"][0], bytes.fromhex(p["salts"][0])) for x, p in zip(rs, ps)])
    return [{k: [p[i][j] for p in per] for j, k in enumerate(("commits", "rows", "salts"))} for i in range(len(want))]


def messages(program: str, mrs: list[dict], ports: list[tuple[str, str]] | None = None) -> tuple[dict, dict]:
    """The prover's registrations of the message rounds and the verifier's own entries of them (from the ``c_k`` alone), by
    port: ``ports`` (port, value) a round, default ``m<i>`` as ``rec-c<i>``."""
    from verity.core.protocols.sampled_proofs.one_stage import registered as RG

    regs, own = {}, {}
    for i, ((port, value), mr) in enumerate(zip(ports or [(f"m{i}", f"rec-c{i}") for i in range(len(mrs))], mrs, strict=True)):
        reg = RG.Registered.commit(program, value, mr["rows"], mr["salts"])
        own[port] = RR.commits_entry(program, value, len(mr["rows"][0]), mr["commits"])
        if own[port]["root"] != reg.entry()["root"]:
            raise ValueError(f"round {i}: the prover's rows are not the rows its c_k commit")
        regs[port] = reg
    return regs, own


def _sparse_stream(public: dict, rep: int) -> list[dict]:
    s = f"{RL.SPARSE}/rep{rep}"
    return sorted((r for r in public["rounds"] if r["stream"] == s), key=lambda r: r["k"])


def sparse_reps(public: dict, private: dict, vf: SimpleNamespace, sp, rng: dict) -> list[tuple]:
    """Each rep's sparse phase (`rec_live.sparse_rounds` at the bucket's `rec_sparse.Shape` ``sp``, T = 1) as the session holds
    it, per rep ``(coins, messages)``: its coins from the coin server's record of its stream (v2's round, which draws none,
    skipped), the inner points from the rep's coins (`rec_sparse.inner_points`) and the slots' layout factors from META's
    range ``rng`` of the hidden circuit (`rec_sparse.layout_factors`), all public; its messages the prover's rows of its
    rounds, h its rep's h round's, and the registered claim ``Σ_p eq(ζ_p, p) u_p``."""
    if sp.stacked:
        raise ValueError("the sparse phase is staged at T = 1")
    want, hk = RL.sparse_rounds(sp), RL.zp_round(vf.sh.m, vf.sh.k_log) + 1
    prv = {(p["stream"], p["k"]): p for p in private["rounds"]}
    nsc, at = len(sp.degrees), 2 + SP.PRE_ROUNDS
    out = []
    for r, rv in enumerate(vf.reps):
        rs = _sparse_stream(public, r)
        el = [RL.row_elements(prv[(x["stream"], x["k"])]["rows"][0])[:e] if e else [] for x, (*_, e) in zip(rs, want)]
        if [(x["n"], len(x["commits"])) for x in rs] != [(n, int(e > 0)) for n, *_, e in want] or \
                any(len(prv[(x["stream"], x["k"])]["elements"]) != e for x, (*_, e) in zip(rs, want)):
            raise ValueError(f"rep {r}: its sparse stream is not the sparse phase's rounds at {sp}")
        coins = [[G.of_hex(c) for c in x["coins"]] for x in rs if x["n"]]
        pts = SP.inner_points(vf.sh, rv.rounds)
        U = SP.layout_factors(sp, pts["x_rest"], pts["y_rest"], rng)
        c = SP.Coins(pts["alpha"], pts["z"], pts["x_rest"], pts["y_rest"], tuple(coins[0]), tuple(x for [x] in coins[1:at - 1]),
                     tuple(x for [x] in coins[at - 1:at - 1 + nsc]), tuple(coins[at - 1 + nsc]), (), tuple(U))
        u = el[at + nsc]
        out.append((c, {"h": RL.row_elements(prv[(f"{TABLE}/rep{r}", hk)]["rows"][0]), "pre": [tuple(e) for e in el[1:at - 1]],
                        "v2": el[at - 1][0], "sc": el[at:at + nsc], "u": u, "claim": SP._dot(SP.eq_table(c.zeta_p), u)}))
    return out


def sparse_messages(program: str, public: dict, private: dict, sp) -> tuple[dict, dict]:
    """The sparse streams' message rounds as `messages` registers them, port ``<p>`` as ``rec-sparse-<p>``
    (`rec_sparse.registrations`' values; row r rep r's), with the verifier's own entries."""
    names = [nm for nm, _ in SP.message_ports(sp)][1:-1]
    prv = {(p["stream"], p["k"]): p for p in private["rounds"]}
    per = [[(bytes.fromhex(x["commits"][0]), prv[(x["stream"], x["k"])]["rows"][0], bytes.fromhex(prv[(x["stream"], x["k"])]["salts"][0]))
            for x in _sparse_stream(public, r) if x["commits"]] for r in range(2)]
    if any(len(p) != len(names) for p in per):
        raise ValueError(f"the sparse streams' message rounds are not {names}")
    mrs = [{k: [per[r][i][j] for r in range(2)] for j, k in enumerate(("commits", "rows", "salts"))} for i in range(len(names))]
    return messages(program, mrs, [(nm, f"rec-sparse-{nm}") for nm in names])


def stage_sparse(public: dict, private: dict, vf: SimpleNamespace, program: str, regs: dict, own: dict, sp, rng: dict, out: Path,
                 evaluate: bool = True, claim=None) -> dict:
    """The sparse phase's RecSparse statement over both reps (`rec_sparse.stage`): each rep's instance from `sparse_reps`, its
    message ports read as the firewall committed them (`sparse_messages`), ``h`` as the algebra's h round (``regs`` and
    ``own``, `messages`' of the reps' rounds), ``claim`` the prover's ``rec-claim`` registration (default fresh salts)."""
    reps_ = sparse_reps(public, private, vf, sp, rng)
    inst = SP.session_rows(sp, [m for _, m in reps_], [SP.verifier_row(sp, c) for c, _ in reps_])
    sregs, sown = sparse_messages(program, public, private, sp)
    hr = f"m{A.h_round(vf.sh)}"
    sregs["h"], sown["h"] = regs[hr], own[hr]
    if claim is not None:
        sregs["claim"] = claim
    return SP.stage(sp, inst, out, program, regs=sregs, own=sown, evaluate=evaluate)


#: R's openings follow both reps' in the chain (`rec_outer.chain_order` is rep-major)
STACKED_REP = 2


def stacked_reads(inner: Path, public: dict, private: dict, vf: SimpleNamespace, sp, rng: dict) -> SimpleNamespace:
    """The stacked table R's registered opening (``flock-circuit stacked``'s proof on ``stacked/rep0``) as the verifier reads
    it: its opening-only shape at R's m (`rec_sparse.table_m`), its points both reps' registered claims (`sparse_reps`), its v
    and query rows from the stream's coins (`rec_algebra.verifier_rows`), its openings as rep `STACKED_REP`'s, their
    coefficients and R's tops."""
    sh = A.Shape(SP.table_m(sp), 6, RL.REPS, salt_len=0, opening_only=True)
    reps_ = sparse_reps(public, private, vf, sp, rng)
    stream = f"{RL.STACKED}/rep0"
    rounds = [[G.of_hex(c) for c in r["coins"]] for r in sorted((r for r in public["rounds"] if r["stream"] == stream), key=lambda r: r["k"])]
    v, info = A.verifier_rows(sh, rounds, [SP.registered_claim(sp, c, r) for r, (c, _) in enumerate(reps_)], [])
    qs = [dataclasses.replace(q, rep=STACKED_REP) for q in V.queries(public, private, V.load(inner, RL.STACKED)[2], RL.STACKED)]
    coefs = coefficients(SimpleNamespace(reps={STACKED_REP: SimpleNamespace(info=info)}), qs)
    return SimpleNamespace(sh=sh, reps=reps_, qs=qs, coefs=coefs, tops=V.tops(public, private, stream, RL.STACKED), v=v)


def opening_messages(public: dict, private: dict, sh: A.Shape, program: str) -> tuple[dict, dict]:
    """R's registered opening's messages (`messages`), port ``m<i>`` reading ``rec-opening-c<i>``: round i of
    `rec_algebra.rounds` at the opening-only shape ``sh`` (``Flock/RecOpening.lean``'s ``rounds``, which its unit's row ports
    follow: each claim's ring switch, then Ligerito's), the i-th round of ``stacked/rep0`` that commits a message. Each must sit
    at a round `rec_live.stacked_rounds` absorbs a message in, so that the coin server recorded its ``c_k`` before it drew
    that round's coins and every later one."""
    stream = f"{RL.STACKED}/rep0"
    ks = sorted(r["k"] for r in public["rounds"] if r["stream"] == stream and r["commits"])
    want = [k for k, r in enumerate(RL.stacked_rounds(sh.m)) if r[3]]
    if ks != want:
        raise ValueError(f"{stream}: its messages are committed at rounds {ks}, not at {want}, the rounds that absorb them")
    mrs = message_rounds(public, private, sh, [stream])
    return messages(program, mrs, [(f"m{i}", f"rec-opening-c{i}") for i in range(len(mrs))])


def stage_stacked(public: dict, private: dict, program: str, sr: SimpleNamespace, chain: O.Chain, coefs: dict, claim, out: Path,
                  evaluate: bool = True) -> dict:
    """R's levels as RecOpen statements on the chain (their tops ``rec-top-stacked-L<level>``) and its registered opening
    (`rec_residuals.stage_opening`): the stacked stream's messages (`opening_messages`), ``claim`` the prover's
    ``rec-claim``, which RecSparse reads too, and the chain's rows either side of R's openings."""
    levels = {}
    for lv in O.levels(sr.qs):
        tops = O.TopValue.build(program, lv["level"], {STACKED_REP: sr.tops[lv["level"]]}, f"rec-top-stacked-L{lv['level']}")
        levels[f"L{lv['level']}"] = O.stage(lv, out / f"L{lv['level']}", chain, coefs, tops)
    regs, own = opening_messages(public, private, sr.sh, program)
    acc = (chain.reg, *chain.rep_ends(STACKED_REP))
    opening = RR.stage_opening(sr.sh, {**regs, "claim": claim}, acc, sr.v, out / "open", evaluate, own)
    return {"m": sr.sh.m, "openings": len(sr.qs), "levels": levels, "opening": opening}


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
    from verity.ml.commitments import hm96

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
    from verity.core.protocols.sampled_proofs.one_stage import registered as RG

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
          evaluate: bool = True, pts: dict | None = None, salts: Path | None = None, cap=None, sparse: tuple | None = None) -> dict:
    """V*'s statements over the inner session ``inner``: the levels ``levels`` (None: every level) and, with ``algebra``, the
    algebra, honest under ``out``; with ``forge``, that forgery's statements alone under ``out/forged-<forge>`` (the module
    docstring). ``salts``: the running sums' salts file (`acc_salts`; None: fresh ones). ``cap`` (`rec_shape.Cap`): refuse a
    session whose V* is not the cap's shape (`rec_shape.check`) before staging. ``sparse``: the bucket's `rec_sparse.Shape`
    and META's range of the hidden circuit; the sparse phase's RecSparse statement over both reps (`sparse_reps`), its
    messages read as the firewall committed them (`sparse_messages`, h as the algebra's h round), under ``out/sparse``; with
    R in the session, `stage_stacked` under ``out/stacked``.
    Returns what each statement costs and holds, and the verifier's own seconds."""
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
    lvls = O.levels(qs)
    if cap is not None:
        from . import rec_shape

        rec_shape.check(sh, lvls, cap)
    coefs = coefficients(vf, qs)
    sr = stacked_reads(inner, public, private, vf, *sparse) if sparse and RL.STACKED in public["link"]["roots"] else None
    allq = qs + (sr.qs if sr else [])
    coefs |= sr.coefs if sr else {}
    t["coefficients_s"] = round(time.perf_counter() - t0, 2)
    ts = {rep: V.tops(public, private, f"{TABLE}/rep{rep}", TABLE) for rep in sorted(proofs)}
    chain = O.Chain.build(program, allq, coefs, None if salts is None else acc_salts(Path(salts), len(allq) + 1))
    res: dict = {"shape": [sh.m, sh.k_log, sh.claims], "holo": sh.holo, "openings": len(qs), "chain_rows": chain.reg.leaves,
                 "acc_root": chain.reg.root.hex(), "program": program, "verifier": t} | ({"cap": cap.spec()} if cap else {})
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
    if algebra or sparse:
        t0 = time.perf_counter()
        mrs = message_rounds(public, private, sh)
        regs, own = messages(program, mrs)
        t["entries_s"] = round(time.perf_counter() - t0, 2)
    if sparse:
        t0 = time.perf_counter()
        claim = None
        if sr is not None:
            inst = SP.session_rows(sparse[0], [m for _, m in sr.reps], [SP.verifier_row(sparse[0], c) for c, _ in sr.reps])
            claim = SP.registrations(sparse[0], program, inst)["claim"]
        res["sparse"] = stage_sparse(public, private, vf, program, regs, own, *sparse, out / "sparse", evaluate, claim)
        res["sparse"]["stage_s"] = round(time.perf_counter() - t0, 1)
        if sr is not None:
            t0 = time.perf_counter()
            res["stacked"] = stage_stacked(public, private, program, sr, chain, coefs, claim, out / "stacked", evaluate)
            res["stacked"]["stage_s"] = round(time.perf_counter() - t0, 1)
    if algebra:
        a0, a1 = zip(*(chain.rep_ends(r) for r in range(2)))
        reads = {nm: (regs[nm], [0, 1]) for nm in regs}
        reads["acc0"], reads["acc1"] = (chain.reg, list(a0)), (chain.reg, list(a1))
        vs = [list(r.v) for r in vf.reps]
        if forge == "comb":
            vs[0][-64] ^= 1
        res["algebra"] = []
        for j, p in enumerate(RR.parts(RR.spec(sh))):
            st = RR.statement(p.S, f"rec-residuals/m{sh.m}-k{sh.k_log}-c{sh.claims}-p{j}", 2, holo=sh.holo)
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
    s.add_argument("--cap", default=os.environ.get("REC_CAP") or None, help="the deployment's cap, instances=N,claims=C,k_log=K,"
                   "vus_per_block=V (rec_shape.Cap.parse; default REC_CAP): refuse a session of another shape, exit 2")
    s.add_argument("--sparse", metavar="NU,ELL,S_LOG,T_LOG", help="stage the sparse phase at this RecSparse shape (with --circuit)")
    s.add_argument("--circuit", help="--sparse: the inner statement's circuit.txt, whose META has the hidden circuit's range")
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
    fold = json.loads(Path(a.fold).read_text())
    sparse = None
    if a.sparse:
        if not a.circuit:
            ap.error("--sparse needs --circuit")
        [rng] = [r for r in meta(Path(a.circuit))["ranges"] if r["type"] == "unit" and r["net"] in fold.get("hidden", [])]
        sparse = (SP.Shape(*(int(v) for v in a.sparse.split(","))), rng)
    from . import rec_shape

    try:
        res = stage(Path(a.inner), fold, Path(a.out), levels, not a.no_algebra, a.forge, not a.no_evaluate, pts, a.salts,
                    rec_shape.Cap.parse(a.cap) if a.cap else None, sparse)
    except rec_shape.Refused as e:
        print(json.dumps({"refused": str(e)}), flush=True)
        return 2
    print(json.dumps(res, sort_keys=True), flush=True)
    alg = res.get("algebra", [])
    if any(p["unit_agrees"] not in (None, [True] * p["instances"]) for p in alg):
        return 1
    if a.forge is not None:
        return 0
    held = all(not s["not_opened"] and not s["not_summed"] for s in res["levels"].values())
    sp, sr = res.get("sparse"), res.get("stacked")
    held = held and (sr is None or (all(not s["not_opened"] and not s["not_summed"] for s in sr["levels"].values())
                                    and all(all(p["holds"]) for p in sr["opening"])))
    return 0 if held and all(all(p["holds"]) for p in alg) and (sp is None or all(sp["holds"])) else 1


if __name__ == "__main__":
    raise SystemExit(main())
