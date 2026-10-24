"""Candidate 3's inner session (hiding model architectures with hidden unit circuits): the unmodified C-Flock prover proves the
inner statement against `rec_live.Proxy` through a relay on the prover's side, which gives the verifier ``h``, the hidden
circuits' share of the lincheck's fold, as a round of its own after z_partial (`rec_live.holo_rounds`,
`rec_algebra.Shape.holo`).

    python -m verity_flock.rec_session --listen 127.0.0.1:PORT --out DIR --statement STATEMENT --circuit CIRCUIT [--hidden unit]
    python -m verity_flock.rec_session --check-fold FOLD --inner DIR      InnerFold --hide's comb plus the session's h is its full fold

**The relay** (`Relay`) passes every request to the proxy except each rep's round that absorbs z_partial and squeezes r′. That
round reaches the proxy as two: z_partial's frames ending in a squeeze of no coins, then a round that absorbs h's 64 elements
and squeezes r′. h is computed between them, at the rep's fold points (`rec_algebra.fold_points`, from the coins the prover
has had: they are drawn before z_partial). So the coin server draws r′ after the ``c_k`` of z_partial and of h, on the rep's
one coin stream, and the prover gets r′ as its round's answer and never sees h.

**h** (`hidden_fold`) is `Stmt.partialRanges` (``Flock/Statement.lean``) on the hidden circuits' ranges alone: per range, its
circuit's ``α·A₀ᵀe_lo + B₀ᵀe_lo`` on the slot's local tensor ``e_lo = lw ⊗ eq(ρ[.. sl − 6])``, projected on ``eq(r[.. sl − 6])``
and weighted by ``Σ_q eq(ρ[sl − 6 ..], q)·eq(r[sl − 6 ..], q)`` over the range's slots, for ``r`` the lincheck's ``ts`` reversed.
The copies between slots (Δ) and the pin stay public: the verifier's comb is the fold of everything else (``InnerFold.lean
--hide``), and comb + h is the circuit's whole fold (`check_fold`).

**The sparse phase and R** (``--sparse``, ``--stacked``): R depends on the hidden circuit alone, so it is registered with it
at staging (`stage_stacked`: R's cells and root, the prover's, and the root's top salted once, the registered value
``stacked`` of the staged statement's circuit registration, `verity_flock.inner_registered.bind`, whose digest META binds).
`serve` reads that entry, its salts and R from the staged statement and refuses to start without them. At the prover's
Finish the relay runs each rep's sparse phase as a stream of its own, each its rep's child (`Relay.sparse_phase`), then
opens the registered R at both reps' registered claims with ``flock-circuit stacked`` on a connection of its own into the
same session (`Relay.stacked_phase`), whose binding round absorbs the circuit reps' statement digest, and only then passes
the prover's Finish on. The prover's Hello and Commit pass as they are, the circuit table's alone: R's root is the salted
top of its stream's binding round (`rec_live.stacked_rounds`), which the firewall holds to the registered top, so the
verifier's record has the statement's Hello, the bytes of the prover's own record.

`serve` writes the proxy's directory (`rec_live.Proxy.write`: ``public.json``, ``private.json`` and ``session/``, whose rounds
are the verifier's, z_partial's of no coins then h's) and ``prover/``: ``session.json``, the session as the prover ran it
(z_partial's round whole, with r′), which upstream's replay and the Lean verifier check against the proofs beside it.
"""
from __future__ import annotations

import contextlib
import copy
import json
import os
import socket
import struct
import subprocess
import threading
from collections.abc import Callable, Sequence
from pathlib import Path

from verity.ml.circuits.ops.boolean import gf2k as G
from verity.ml.circuits.ops.boolean.gf2k import mul

from . import rec_algebra as A
from . import rec_live as RL
from .record_text import session_text

HIDDEN = ("unit",)
UNIT_FORMAT = "flock-ir-unit/v2"
#: the sparse streams' Open label
SPARSE_LABEL = b"verity/rec-sparse/v1"


def _frame(op: int, kind: int, n: int, payload: bytes = b"") -> bytes:
    """A transcript-v2 frame (`rec_live.frames`): ``[op][kind][0; 6][len u64 LE]``, the payload zero-padded to 16 bytes."""
    return bytes([op, kind]) + bytes(6) + struct.pack("<Q", n) + payload + bytes(-len(payload) % 16)


def observe(xs: Sequence[int]) -> bytes:
    """The frame that absorbs ``xs`` as one slice of field elements."""
    return _frame(RL.OP_OBSERVE, RL.KIND_SLICE, len(xs), b"".join(RL.f128_bytes(x) for x in xs))


# --- h ---------------------------------------------------------------------------------------------------------------------------

def log2ceil(n: int) -> int:
    return (n - 1).bit_length() if n > 1 else 0


def hidden_circuits(circuit: Path, hidden: Sequence[str] = HIDDEN) -> tuple[dict, list[dict], dict]:
    """The hidden circuits as the verifier's parser reads them (`Circuit.parse`, `Net.parse`): each one's slot log
    (``max(7, ⌈log2 useful⌉)``) and its useful rows' A and B columns; META's ranges of them, in META's order; and META.
    Flat statements only: a typed template's root reads its parts' columns across slots, so its A₀ is not its rows."""
    nets: dict[str, dict] = {}
    with open(circuit) as f:
        f.readline()
        meta = json.loads(f.readline()[5:])
        if "types" in meta or "unit_type" in meta:
            raise ValueError("a typed statement: its root's rows read columns outside its slot")
        while line := f.readline():
            _, name, n = line.split()
            n = int(n)
            if name not in hidden:
                for _ in range(n):
                    f.readline()
                continue
            head = f.readline().split()
            if head[:1] != [UNIT_FORMAT]:
                raise ValueError(f"{name}: not a {UNIT_FORMAT} circuit")
            useful = int(head[2])
            a, b = [], []
            for i in range(useful):
                v = [int(t) for t in f.readline().split()]
                na = v[0] if v else 0
                nb = v[1 + na] if len(v) > 1 + na else 0
                if not v or len(v) != 2 + na + nb:
                    raise ValueError(f"{name}: row {i} is malformed")
                a.append(v[1:1 + na])
                b.append(v[2 + na:])
            for _ in range(n - 1 - useful):
                f.readline()
            ul = max(7, log2ceil(useful))
            if any(c >= 1 << ul for row in a + b for c in row):
                raise ValueError(f"{name}: a column outside its 2^{ul}-bit slot")
            nets[name] = {"unit_log": ul, "a": a, "b": b}
    if any(nm.startswith("lookup:") for nm in hidden) or set(nets) != set(hidden):
        raise ValueError(f"hidden circuits {sorted(hidden)}: the file's expanded circuits hold {sorted(nets)} of them")
    ranges = [r for r in meta["ranges"] if r["type"] == "unit" and r["net"] in nets]
    if sorted(r["net"] for r in ranges) != sorted(nets):
        raise ValueError("a hidden circuit has not exactly one range")
    for r in ranges:
        if nets[r["net"]]["unit_log"] != r["slot_log"]:
            raise ValueError(f"{r['net']}: a range of slots of 2^{r['slot_log']}, its circuit's are 2^{nets[r['net']]['unit_log']}")
    return nets, ranges, meta


def tensor(lw: Sequence[int], rho: Sequence[int]) -> list[int]:
    """``e[s + 64 u] = lw[s] · eq(ρ, u)`` (`Stmt.tensor`)."""
    e_in = A.eq_table(rho)
    return [mul(lw[k % len(lw)], e_in[k // len(lw)]) for k in range(len(lw) * len(e_in))]


def fold_t(rows: Sequence[Sequence[int]], e: Sequence[int], width: int) -> list[int]:
    """``Xᵀ e`` (`Sparse.foldT`): ``out[c] = Σ_{i : c ∈ row i} e[i]``."""
    out = [0] * width
    for i in range(min(len(rows), len(e))):
        if e[i]:
            for c in rows[i]:
                out[c] ^= e[i]
    return out


def project(base: Sequence[int], e: Sequence[int]) -> list[int]:
    """A slot type's share of the 64 entries (`Stmt.project`): ``Σ_u base[s + 64 u] · e[u]`` at ``s``."""
    out = [0] * 64
    for u, eu in enumerate(e):
        if eu:
            for s in range(64):
                out[s] ^= mul(base[s + 64 * u], eu)
    return out


def hidden_fold(nets: dict, ranges: Sequence[dict], alpha: int, lw: Sequence[int], rho: Sequence[int], ts: Sequence[int]) -> list[int]:
    """h: `Stmt.partialRanges` on ``ranges`` (the hidden circuits'), at the fold points ``alpha``, ``lw``, ``rho`` (ρ_in) and
    ``ts`` (the module docstring)."""
    if len(lw) != 64 or len(rho) != len(ts):
        raise ValueError("the fold points are not the block's")
    rr, out, bases = list(ts[::-1]), [0] * 64, {}
    for r in ranges:
        sl, net = r["slot_log"], nets[r["net"]]
        if net["unit_log"] != sl or not 6 <= sl <= len(rho) + 6:
            raise ValueError(f"{r['net']}: a slot of the wrong size")
        key = (r["net"], sl)
        if key not in bases:
            e_lo = tensor(lw, rho[:sl - 6])
            fa, fb = fold_t(net["a"], e_lo, 1 << sl), fold_t(net["b"], e_lo, 1 << sl)
            bases[key] = [mul(alpha, x) ^ y for x, y in zip(fa, fb, strict=True)]
        h = sl - 6
        e_hi, r_hi = A.eq_table(rho[h:]), A.eq_table(rr[h:])
        if r["pos0"] + r["count"] > len(e_hi):
            raise ValueError(f"{r['net']}: a range outside the block")
        wt = 0
        for q in range(r["pos0"], r["pos0"] + r["count"]):
            wt ^= mul(e_hi[q], r_hi[q])
        out = [o ^ mul(wt, x) for o, x in zip(out, project(bases[key], A.eq_table(rr[:h])), strict=True)]
    return out


def session_h(private: dict, m: int, k_log: int) -> list[list[int]]:
    """Each rep's h as the session holds it: its h round's elements (`rec_live.holo_rounds`) in ``private.json``."""
    k = RL.zp_round(m, k_log) + 1
    by = {(p["stream"], p["k"]): p for p in private["rounds"]}
    return [[G.of_hex(x) for x in by[(f"{RL.TABLE}/rep{r}", k)]["elements"]] for r in range(RL.REPS)]


def check_fold(fold: dict, private: dict) -> dict:
    """``InnerFold.lean --hide``'s output (each rep's public ``comb`` and ``full`` fold) against the session's h: per rep,
    whether comb + h is the full fold, and whether h is nonzero."""
    hs = session_h(private, fold["m"], fold["k_log"])
    ok = []
    for d, h in zip(fold["reps"], hs, strict=True):
        comb, full = ([G.of_hex(x) for x in d[k]] for k in ("comb", "full"))
        ok.append([c ^ x for c, x in zip(comb, h, strict=True)] == full)
    return {"hidden": fold.get("hidden"), "comb_plus_h_is_full": ok, "h_nonzero": [any(h) for h in hs]}


# --- R's registration ------------------------------------------------------------------------------------------------------------

#: the prover's own R and its root (``flock-circuit stacked --root-out``), beside the staged ``inst-N.bin``
STACKED_TABLE, STACKED_ROOT = "R.bin", "ROOT.json"
#: META's key for R's opening-only shape as staging commits R, ``{"m", "claims"}`` (`stacked_shape`)
STACKED_SHAPE_KEY = "stacked"


def sparse_entries(circuit: Path, sparse, hidden: Sequence[str] = HIDDEN) -> tuple[list, dict]:
    """The one hidden circuit's entries in the bucket ``sparse`` (`rec_sparse.entries`) and its range."""
    from . import rec_sparse as SP

    nets, ranges, _ = hidden_circuits(Path(circuit), hidden)
    if len(ranges) != 1:
        raise ValueError(f"the sparse phase at T = 1 runs one hidden circuit, not {len(ranges)}")
    net = nets[ranges[0]["net"]]
    return SP.entries(sparse, net["a"], net["b"]), ranges[0]


def stage_stacked(stage: Path, bin_: Path, sparse, program, partition: dict, forge: str | None = None,
                  fresh_salt: Callable[[], bytes] | None = None) -> dict:
    """The stacked table R registered with the staged statement in ``stage``, at staging: R's cells from the hidden
    circuit's entries (`rec_sparse.table` in the bucket ``sparse``; with ``forge`` ``cell``, cell 0 flipped: a prover
    registering an R that is not its circuit's) in `STACKED_TABLE`, its root by ``flock-circuit stacked --root-out``
    (`STACKED_ROOT`), and the root's top registered as the value `rec_live.STACKED` (`rec_live.register_top`, salts from
    ``fresh_salt``, default the operating system's) of the circuit registration (`inner_registered.bind`, over ``program``
    and ``partition``, the statement's): its entry in the record META binds, its nodes and salts the prover's. META's
    `STACKED_SHAPE_KEY` is R's opening-only ``Shape(m, 6, claims)``: ``m`` R's variables (`rec_sparse.table_m`), ``claims``
    the two reps' sparse claims, which the entry, a cap of 128 nodes, does not fix. Returns `rec_live.register_top`'s
    opening."""
    from . import inner_registered as IR
    from . import rec_sparse as SP

    stage = Path(stage)
    IR.bind_meta(stage, STACKED_SHAPE_KEY, {"m": SP.table_m(sparse), "claims": RL.REPS})
    raw = bytearray(SP.table(sparse, sparse_entries(stage / "circuit.txt", sparse)[0]))
    if forge == "cell":
        raw[0] ^= 1
    (stage / STACKED_TABLE).write_bytes(bytes(raw))
    subprocess.run([str(bin_), "stacked", "--table", str(stage / STACKED_TABLE), "--root-out", str(stage / STACKED_ROOT)],
                   check=True, capture_output=True)
    cap = bytes.fromhex(json.loads((stage / STACKED_ROOT).read_text())["cap"])
    reg = RL.register_top(partition["program"], RL.STACKED, cap, *([fresh_salt] if fresh_salt else []))
    IR.bind(stage, program, partition, [reg["entry"]], {RL.STACKED: {"nodes": reg["nodes"], "salts": reg["salts"]}})
    return reg


def staged_stacked(stage: Path) -> tuple[dict, str, dict]:
    """The circuit registration META binds in ``stage`` (`inner_registered.bound`), its digest, and the `rec_live.Proxy`'s
    ``registered`` opening of its `rec_live.STACKED` entry (`inner_registered.private`, its ``top`` the nodes' ``b || c``
    under their salts); refused when the record carries no ``stacked`` entry or the prover holds no opening of it."""
    from verity.core.protocols.sampled_proofs.one_stage import registration as R

    from . import inner_registered as IR

    rec = IR.bound(stage)
    if RL.registered_entry(rec, RL.STACKED) is None:
        raise ValueError(f"{stage}: the circuit registration META binds carries no {RL.STACKED} entry: R was not registered at staging")
    own = IR.private(stage, RL.STACKED)
    cap = bytes.fromhex("".join(own["nodes"]))
    top = RL.commit_top(cap, [bytes.fromhex(s) for s in own["salts"]])
    return rec, R.digest(rec), {**own, "top": [b.hex() for b in top]}


def stacked_shape(stage: Path, sparse) -> int:
    """R's ``m`` as staging committed R (META's `STACKED_SHAPE_KEY`), refused unless it is the bucket ``sparse``'s
    (`rec_sparse.table_m`) and R is opened at the two reps' claims."""
    from . import inner_registered as IR
    from . import rec_sparse as SP

    got = IR.meta(stage).get(STACKED_SHAPE_KEY)
    want = {"m": SP.table_m(sparse), "claims": RL.REPS}
    if got != want:
        raise ValueError(f"{stage}: META's {STACKED_SHAPE_KEY} is {got}, not R's shape {want} at the bucket {sparse}")
    return got["m"]


# --- the relay -------------------------------------------------------------------------------------------------------------------

class Relay:
    """The prover's side of candidate 3's inner session (the module docstring): every request to ``proxy`` but each rep's
    z_partial round, which it splits. ``h_of(points)``: h at a rep's fold points (`rec_algebra.fold_points`' ints)."""

    def __init__(self, proxy: RL.Proxy, m: int, k_log: int, h_of: Callable[[dict], list[int]], sparse: tuple | None = None,
                 workers: int = 1, stacked: dict | None = None):
        self.proxy, self.h_of = proxy, h_of
        self.sh, self.zp = A.Shape(m, k_log, 2), RL.zp_round(m, k_log)
        self.reps = {f"{RL.TABLE}/rep{r}" for r in range(RL.REPS)}
        #: each stream's coins by round, as the prover had them
        self.coins: dict[str, list[list[int]]] = {}
        #: per rep, its z_partial round as the prover sent it, h and the fold points it was computed at
        self.split: dict[str, dict] = {}
        #: the sparse phase (`sparse_phase`): the bucket's `rec_sparse.Shape`, the hidden circuit's entries and its range;
        #: ``workers`` processes share each rep's H3
        self.sparse, self.workers = sparse, workers
        #: per rep, its sparse phase's coins and messages (`rec_sparse.prove`)
        self.sparse_out: dict[int, tuple] = {}
        #: the stacked table R (`stacked_phase`): ``bin`` (flock-circuit), ``dir`` (R.bin and ROOT.json as staged, P.json and
        #: its logs), ``raw`` (R's cells), ``forge`` (None, ``cell``: R registered at staging with cell 0 flipped, ``late``:
        #: R's cell 0 flipped after its registration, or ``point``: rep 0's point off by one bit)
        self.stacked = stacked

    def _refuse(self, why: str) -> bytes:
        self.proxy.stop(f"rec session: {why}")
        return RL.resp_str(RL.RESP_REFUSED, self.proxy.stopped)

    def handle(self, req: bytes) -> bytes:
        r = None
        if req[:1] == bytes([RL.REQ_FINISH]) and self.sparse is not None and not self.sparse_out and self.proxy.stopped is None:
            self.sparse_phase()
            if self.stacked is not None and self.proxy.stopped is None:
                self.stacked_phase()
        if req[:1] == bytes([RL.REQ_ROUND]) and self.proxy.stopped is None:
            try:
                r = RL.decode_req(req)
            except ValueError:
                r = None
        if r is not None and r[1] in self.reps and len(self.coins.get(r[1], [])) == self.zp:
            resp = self._split(*r[1:])
        else:
            resp = self.proxy.handle(req)
        if r is not None and resp[:1] == bytes([RL.RESP_COINS]):
            self.coins.setdefault(r[1], []).append(RL.decode_resp(resp)[1])
        return resp

    def _split(self, stream: str, content: bytes, n: int) -> bytes:
        try:
            fr, free = RL.frames(content), RL.split_round(content, False).free
        except ValueError as e:
            return self._refuse(f"{stream} round {self.zp}: {e}")
        if n != 1 or len(free) != 64 or fr[-1][0] != RL.OP_SQUEEZE:
            return self._refuse(f"{stream} round {self.zp} is not z_partial's (64 elements, then r′)")
        squeeze = content[-16:]
        resp = self.proxy.handle(RL.encode_req(RL.REQ_ROUND, stream, content[:-16] + _frame(RL.OP_SQUEEZE, RL.KIND_SLICE, 0), 0))
        if RL.decode_resp(resp) != (RL.RESP_COINS, []):
            return resp
        try:
            pts = A.fold_points(self.sh, self.coins[stream] + [[0]])
            h = self.h_of(pts)
        except Exception as e:  # noqa: BLE001 - the session stops, never the relay
            return self._refuse(f"{stream}: h: {type(e).__name__}: {e}")
        resp = self.proxy.handle(RL.encode_req(RL.REQ_ROUND, stream, observe(h) + squeeze, n))
        if resp[:1] == bytes([RL.RESP_COINS]):
            self.split[stream] = {"content": content, "h": h, "points": pts}
        return resp

    def _round(self, stream: str, elems: list[int], n: int) -> list[int]:
        content = (observe(elems) if elems else b"") + _frame(RL.OP_SQUEEZE, RL.KIND_SLICE, n)
        resp = self.proxy.handle(RL.encode_req(RL.REQ_ROUND, stream, content, n))
        if resp[:1] != bytes([RL.RESP_COINS]):
            raise RuntimeError(f"{stream}: {RL.decode_resp(resp)[1]}")
        return RL.decode_resp(resp)[1]

    def sparse_phase(self) -> None:
        """Each rep's sparse stream (`rec_live.sparse_rounds`), opened as its rep's child at the rep's end once the prover has
        sent Finish: `rec_sparse.prove` at the rep's fold points and h, each message a round of the relay's own. A refusal,
        or a sum the phase cannot send, stops the session."""
        from . import rec_sparse as SP

        sh, ents, rng = self.sparse
        try:
            for r in range(RL.REPS):
                rep, stream = f"{RL.TABLE}/rep{r}", f"{RL.SPARSE}/rep{r}"
                at = len(self.proxy.streams[rep]["rounds"])
                resp = self.proxy.handle(RL.encode_req(RL.REQ_OPEN, stream, rep, SPARSE_LABEL, at))
                if resp[:1] != bytes([RL.RESP_OK]):
                    return
                pts = SP.inner_points(self.sh, self.coins[rep][:self.zp] + [[0]])
                U = SP.layout_factors(sh, pts["x_rest"], pts["y_rest"], rng)
                self.sparse_out[r] = SP.prove(sh, ents, pts, U, self.split[rep]["h"], lambda e, n, s=stream: self._round(s, e, n),
                                              self.workers)
        except Exception as e:  # noqa: BLE001 - the session stops, never the relay
            self.proxy.stop(f"rec session: sparse phase: {type(e).__name__}: {e}")

    def stacked_phase(self) -> None:
        """The registered R (`stage_stacked`) opened at both reps' registered claims (`rec_sparse.registered_claim`, each
        checked to be R's value there, the sparse phase's claim) by ``flock-circuit stacked --points P.json
        --statement-digest D --verifier`` on a connection of its own into the same session, whose binding round absorbs the
        circuit reps' statement digest ``D`` (the firewall's `rec_live.Proxy.statement_digest`) and commits R's root, the
        registered top; its Finish is answered here, the prover's goes on to the proxy."""
        from . import rec_sparse as SP

        sh, st = self.sparse[0], self.stacked
        d = Path(st["dir"])
        try:
            if RL.STACKED not in self.proxy.registered or self.proxy.statement_digest is None:
                raise AssertionError("R was never registered, or no circuit rep bound the statement")
            table = d / STACKED_TABLE
            if st.get("forge") == "late":
                raw = bytearray(st["raw"])
                raw[0] ^= 1
                table = d / "R-late.bin"
                table.write_bytes(bytes(raw))
            x = []
            for r in range(RL.REPS):
                c, msgs = self.sparse_out[r]
                cl = SP.registered_claim(sh, c, r)
                point = [*c.rounds[:sh.NU], *c.zeta_p, *c.rho] + [0] * cl.pad
                if st.get("forge") is None and SP.table_value(st["raw"], point) != msgs["claim"]:
                    raise AssertionError(f"rep {r}: R's value at the claim's point is not the sparse phase's claim")
                x.append(list(cl.x))
            if st.get("forge") == "point":
                # a point's first coordinate enters only the verifier's weights; the prover's ring switch reads the rest
                x[0][1] ^= 1
            (d / "P.json").write_text(json.dumps({"x": [[RL.f128_hex(v) for v in xs] for xs in x]}))
            with socket.create_server(("127.0.0.1", 0)) as srv, open(d / "open.log", "w") as log:
                host, port = srv.getsockname()[:2]
                proc = subprocess.Popen([str(st["bin"]), "stacked", "--table", str(table), "--points", str(d / "P.json"),
                                         "--statement-digest", self.proxy.statement_digest.hex(), "--verifier", f"{host}:{port}"],
                                        stdout=log, stderr=subprocess.STDOUT)

                def wake():
                    proc.wait()
                    with contextlib.suppress(OSError):
                        socket.create_connection((host, port)).close()

                threading.Thread(target=wake, daemon=True).start()
                conn, _ = srv.accept()
                with conn, conn.makefile("rb") as rf, conn.makefile("wb") as wf:
                    while (req := RL.read_frame(rf)) is not None:
                        if req[:1] == bytes([RL.REQ_FINISH]):
                            RL.write_frame(wf, RL.resp_ok())
                            break
                        RL.write_frame(wf, self.proxy.handle(req))
            if proc.wait() != 0 and self.proxy.stopped is None:
                raise RuntimeError(f"flock-circuit stacked exited {proc.returncode}: {(d / 'open.log').read_text()[-400:]}")
            bound = (self.proxy.streams.get(f"{RL.STACKED}/rep0") or {}).get("cap")
            if self.proxy.stopped is None and (bound is None or bound.hex() != json.loads((d / "ROOT.json").read_text())["cap"]):
                raise AssertionError("R's stream bound a root other than R's (ROOT.json)")
        except Exception as e:  # noqa: BLE001 - the session stops, never the relay
            self.proxy.stop(f"rec session: stacked phase: {type(e).__name__}: {e}")

    def prover_record(self) -> dict:
        """The session as the prover ran it: `rec_live.Proxy.session_record` with each rep's z_partial round whole again,
        answered with r′, and only the reps' streams and proofs (what upstream's replay and the Lean verifier check the
        proofs against)."""
        p = copy.copy(self.proxy)
        p.streams, p.proofs = {}, {k: v for k, v in self.proxy.proofs.items() if k[0] == RL.TABLE}
        for name, st in self.proxy.streams.items():
            if name not in self.reps:
                continue
            rounds = list(st["rounds"])
            if name in self.split:
                k = self.zp
                if rounds[k]["n"] or rounds[k + 1]["n"] != 1:
                    raise AssertionError(f"{name}: rounds {k} and {k + 1} are not z_partial's and h's")
                rounds[k:k + 2] = [{"content": self.split[name]["content"], "n": 1, "coins": rounds[k + 1]["coins"]}]
            p.streams[name] = {**st, "rounds": rounds}
        return p.session_record()

    def write(self, out: Path) -> Path:
        """The proxy's directory (`rec_live.Proxy.write`) and ``prover/`` (`prover_record` and the proofs)."""
        self.proxy.write(out)
        d = Path(out) / "prover"
        d.mkdir(parents=True, exist_ok=True)
        (d / "session.json").write_text(session_text(self.prover_record()))
        for (t, r), b in self.proxy.proofs.items():
            if t == RL.TABLE:
                (d / f"{t}.rep{r}.bin").write_bytes(b)
        return d


def serve(listen: str, out: Path, statement: dict, circuit: Path, hidden: Sequence[str] = HIDDEN,
          rand: Callable[[int], bytes] = os.urandom, ready: Callable[[str], None] | None = None, sparse=None,
          workers: int = 1, stacked: Path | None = None, forge: str | None = None, draw: dict | None = None) -> dict:
    """One session on ``listen``, held to the statement's holo schedule (`rec_live.schedule_for`, written to
    ``out/schedule.json`` first), through a `Relay` whose h is the ``hidden`` circuits' fold in ``circuit``; written into
    ``out`` (`Relay.write`), or, when it stops, the coin server's record alone. ``sparse``: the bucket's `rec_sparse.Shape`,
    whose sparse phase (`Relay.sparse_phase`, on ``workers`` processes) the one hidden circuit's entries run. ``stacked``
    (with ``sparse``): flock-circuit, which opens R at the sparse claims (`Relay.stacked_phase`, under ``out/stacked``;
    ``forge`` its forgery), R and its registered top as staged beside ``circuit`` (`stage_stacked`, read by
    `staged_stacked`): the session is refused before it starts unless the circuit registration META binds carries the
    entry `rec_live.STACKED` and the prover holds its opening. ``draw``: the verifier's draw, answered at ``Register`` and
    kept as both records' ``unit_draw`` (`rec_live.Proxy`). Returns its summary."""
    m, k_log = statement["dense_m"], statement["k_log"]
    nets, ranges, meta = hidden_circuits(Path(circuit), hidden)
    if meta["k_log"] != k_log:
        raise ValueError(f"the circuit's k_log {meta['k_log']} is not the statement's {k_log}")
    phase = None
    if sparse is not None:
        from . import rec_sparse as SP

        phase = (sparse, *sparse_entries(Path(circuit), sparse, hidden))
    st, registration, bound, registered = None, None, None, {}
    if stacked is not None:
        if sparse is None:
            raise ValueError("the stacked table is opened at the sparse phase's claims: stacked needs sparse")
        stage = Path(circuit).parent
        registration, bound, registered[RL.STACKED] = staged_stacked(stage)
        m_r = stacked_shape(stage, sparse)
        d = Path(out) / "stacked"
        d.mkdir(parents=True, exist_ok=True)
        raw = (stage / STACKED_TABLE).read_bytes()
        (d / STACKED_TABLE).write_bytes(raw)
        (d / STACKED_ROOT).write_bytes((stage / STACKED_ROOT).read_bytes())
        st = {"bin": Path(stacked), "dir": d, "forge": forge, "raw": raw, "m": m_r}
    schedule = RL.schedule_for(statement, holo=True, sparse=sparse, stacked=None if st is None else st["m"])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "schedule.json").write_text(json.dumps(schedule, indent=1))
    proxy = RL.Proxy(RL.CoinServer(rand), schedule=schedule, registration=registration, registered=registered, bound=bound,
                     draw=draw)
    relay = Relay(proxy, m, k_log, lambda p: hidden_fold(nets, ranges, p["alpha"], p["lw"], p["rho_in"], p["ts"]), phase, workers, st)
    host, port = listen.rsplit(":", 1)
    with socket.create_server((host, int(port))) as srv:
        h, p = srv.getsockname()[:2]
        if ready:
            ready(f"{h}:{p}")
        conn, _ = srv.accept()
        try:
            with conn, conn.makefile("rb") as rf, conn.makefile("wb") as wf:
                while (req := RL.read_frame(rf)) is not None:
                    RL.write_frame(wf, relay.handle(req))
                    if proxy.verdict is not None:
                        break
        except (OSError, ValueError):
            pass
    if proxy.verdict is None and proxy.stopped is None:
        proxy.stop("rec session: the session ended without Finish")
    if proxy.stopped is not None:
        (out / "public.json").write_text(RL.record_bytes(proxy.coins.record()))
        return {"session": None, "refused": proxy.stopped, "rounds": len(proxy.coins.rounds)}
    relay.write(out)
    rec = proxy.coins.record()
    return {"session": str(out / "session"), "prover": str(out / "prover"), "rounds": len(rec["rounds"]),
            "message_rows": rec["message_rows"], "coins": sum(r["n"] for r in rec["rounds"]), "hidden": list(hidden),
            "h_round": relay.zp + 1, "proof_bytes": {f"{t}/rep{r}": len(b) for (t, r), b in proxy.proofs.items()},
            "sparse": None if sparse is None else {"shape": [sparse.NU, sparse.ELL, sparse.S_LOG, sparse.T_LOG], "reps": len(relay.sparse_out)},
            "stacked": None if st is None else {"m": st["m"], "forge": forge, "root": json.loads((st["dir"] / "ROOT.json").read_text()),
                                                "registration": str(out / RL.REGISTRATION)}}


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m verity_flock.rec_session")
    ap.add_argument("--listen", default="127.0.0.1:0")
    ap.add_argument("--out")
    ap.add_argument("--statement", help="flock-circuit statement's output (its STATEMENT record)")
    ap.add_argument("--circuit", help="the inner statement's circuit.txt (the hidden circuits' rows)")
    ap.add_argument("--hidden", default=",".join(HIDDEN), help="the hidden circuits, comma-separated")
    ap.add_argument("--draw", help="the verifier's draw, answered at Register and recorded (rec_inner.py stage's draw.json)")
    ap.add_argument("--check-fold", metavar="FOLD", help="InnerFold.lean --hide's output: check its comb plus the session's h "
                    "is its full fold, and exit (1 if not)")
    ap.add_argument("--inner", help="--check-fold: the session directory, whose private.json holds h")
    ap.add_argument("--sparse", metavar="NU,ELL,S_LOG,T_LOG", help="each rep's sparse phase at this RecSparse shape (the bucket's)")
    ap.add_argument("--workers", type=int, help="--sparse: processes sharing each rep's H3 (default: the cores it may run on)")
    ap.add_argument("--stacked", metavar="FLOCK_CIRCUIT", help="--sparse: commit R and open it at the sparse claims with this build")
    ap.add_argument("--forge", choices=("cell", "late", "point"),
                    help="--stacked: R registered at staging with cell 0 flipped (staged so), R's cell 0 flipped after its "
                    "registration, or rep 0's point off")
    a = ap.parse_args(argv)
    if a.check_fold:
        if not a.inner:
            ap.error("--check-fold needs --inner")
        res = check_fold(json.loads(Path(a.check_fold).read_text()), json.loads((Path(a.inner) / "private.json").read_text()))
        print(json.dumps(res), flush=True)
        return 0 if all(res["comb_plus_h_is_full"]) and all(res["h_nonzero"]) else 1
    if not (a.out and a.statement and a.circuit):
        ap.error("--out, --statement and --circuit are required to serve")
    sparse = None
    if a.sparse:
        from .rec_sparse import Shape
        sparse = Shape(*(int(v) for v in a.sparse.split(",")))
    res = serve(a.listen, Path(a.out), RL.statement_of(Path(a.statement).read_text()), Path(a.circuit), a.hidden.split(","),
                ready=lambda addr: print(f"SERVING {addr}", flush=True), sparse=sparse,
                workers=len(os.sched_getaffinity(0)) if a.workers is None else a.workers, stacked=a.stacked, forge=a.forge,
                draw=json.loads(Path(a.draw).read_text()) if a.draw else None)
    print("REC_INNER\t" + json.dumps(res), flush=True)
    return 0 if res["session"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
