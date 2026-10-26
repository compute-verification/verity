"""One-stage audit, run A5, on a TP2 transfer-gate stand-in: a synthetic two-rank Llama TP2 step whose cross-rank transfers
(each row-parallel reduction's partials, which rank 1 − r receives from rank r) are registered values, and whose reductions are
transfer-gate units reading both ranks' partials through ``--registered`` ports, audited by `a5.audit` as each rank's
window on that rank's `service.Stream`, on the verifier's own coins (``--coins verifier``). No served window is TP2 with
committed advice yet (the registered compiler's TP2 rank Programs, `b1_tp2`, bind a rank's peer input to the other rank's
partial only by the host's Match, in the clear), so this step stands in for one.

* **What crosses ranks** (`COLLECTIVES`): in a Llama TP2 Program each layer's o_proj and down_proj, and the vocab-parallel
  embedding, end in `AllReduce2_v1{H}`: each rank's partial row (one a token) and the other rank's, summed element by
  element, ``bf16_rn(f32(a) + f32(b))``. The stand-in step is layer 0's: the embedding's, o_proj's and down_proj's, at
  H = ``--hidden`` (2048, LLAMA32_1B's). The logits' `AllGather2_v1` is a copy, and the sampling
  rank's units that read it are egress, a gate already (`a5_warden`).
* **Request by request.** The step serves ``--requests`` requests, each its own ``--tokens`` token rows (positions 0 to
  T − 1 of its prefill). The registered compiler's TP2 request Programs read each collective's peer row as an Input of the
  request's own (``peers_k``, k = pass·34 + site: site 0 the embedding, 1 + 2L layer L's o_proj, 2 + 2L its down_proj), and
  R1 takes request q's ``peers_k`` at rank r as request q's own on the premise that it is request q's partial from rank
  1 − r, at that site and position, never another request's. The transfer gates hold that premise: each request's partial
  is its own registered value, and request q's units read request q's.
* **The partials** (`partials`): rank r's partial of collective k for request q is the registered value `value_name`
  (r, q, k), one row a token of H bf16 words, committed under hiding salts (`registered.Registered.commit`, row schema v2),
  its domain binding the TP run root (`ROOT`) and the TP step (`PROGRAM`). Rank r's registration names each of its partials'
  entries (``of.partials``) and each peer partial's it reads (``of.peers``), by value, before any rank's coin. Synthetic:
  each rank's f32 activations, rounded to bf16.
* **The window** (`window`), one a rank under ``<ROOT>/rank<r>`` and the law ``work:2``: Q·K·T units of the transfer-gate
  template, the registered `AllReduce2_v1{H}` itself (`b1_tp2.AllReduce2`), its input ``parts`` staged as two ports,
  ``part0`` and ``part1``, its rows ``parts[0]`` and ``parts[1]``: the ranks' partials in rank order (`PORT_PARAMS`), a
  chunk ``q<q>c<k>`` a request's collective, unit i
  of it token i; and the rest, ``--rest`` units of `WardenRowIdentity{128}` over rows of their own, standing in for the
  compiled model steps. The verifier's work table proves every transfer-gate unit (selection All) beside the rest's drawn
  share. Each chunk's ``part<j>`` is a shared-row table of rank j's partial's rows under rank j's salts, so its ``b || c``
  is the registered value's, and its header names the value, the rows' positions and their paths
  (`registered.Registered.statement`); its ``o0``, the reduced row, is committed afresh. Both ranks prove the same sums:
  AllReduce2's add is commutative.
* **The verifier's reads** (`reads`): every rank's registration read before any rank's coin, and from them each chunk's own
  ``--registered`` file: ``part<j>`` at the root rank j registered for request q's partial of collective k, unit i reading
  row i, which a5 hands Lean's verify of each chunk (`a5.audit`'s ``registered``). So both ends of a transfer open the one
  value the sender registered: a received row other than the sender's opens no root ("does not open"), another token's row
  is read at a position the Program doesn't read ("reads position"), and another collective's partial, another request's,
  or the receiving rank's own read as the sender's, is another value ("not another value").
* **Provenance** (`source_of`): rank r's registration is a stand-in of its rank's run, of this step at the TP run root, its
  partials each the verifier's own shape and domain, and its peers exactly the partials rank 1 − r registered.
* **Negatives**, beside a5's (its registered reads' three, the transfer-gate negatives, R1 to R4 and the draw's): rank r's
  registration naming its peer's partial under another root, naming its own partial as its peer's, naming another
  request's partial as its peer's, naming a partial under another TP run's domain, and the other rank's registration as its
  own, each refused by `source`; rank r's ports built before the peer's registration is read, refused by `reads`; and the
  receiving port of request 0's c0 opened by the statement of each `TRANSFERS` kind, each refused by the host's read of
  Lean's registered reads (`warden_rows.refusal`) by name.
* **Refused runs**, each to pass every check before the draw on both ranks, accept rank 0, draw rank 1's q0c0 and be refused
  by Lean's registered reads by name: ``--altered``, rank 1's ``part0`` row 0 (rank 0's embedding partial of request 0 as
  rank 1 received it) one bit off, under rank 0's salt, the advice bit moved through the transfer, its sum computed from the
  altered row (the circuit holds) ("does not open"); ``--made-for swapped|other-collective|other-request|own``, that port's
  statement made for two tokens swapped, rank 0's o_proj partial, rank 0's embedding partial of request 1, or rank 1's own
  embedding partial.

The run exits 0 when it is as expected: both ranks accepted with every negative refused, or (``--altered``, ``--made-for``)
rank 0 accepted and rank 1 refused by Lean's registered reads by its reason. ``--out`` holds the windows as committed
(``window/rank<r>/``, their private files too) and a5's records (``rank<r>/``, ``summary.json``).

    PYTHONPATH=<tree>:<tree>/benchmarks/numerical/python:<tree>/integrations/vllm \\
      python benchmarks/one_stage/a5_tp_gates.py --m0-pythonpath PATH --bin flock-circuit --lean flock-verify \\
        --firewall flock-firewall --coins verifier [--altered | --made-for KIND] --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "network_traces"))

from a5_warden import composer as row_composer, free_ports, refusal  # noqa: E402
from warden_zk import WardenRowIdentity  # noqa: E402
from verity.ml.circuits import evaluation as E  # noqa: E402
from verity.ml.circuits.defs import bind  # noqa: E402
from verity.ml.circuits.ops import fp32 as W  # noqa: E402
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2  # noqa: E402
from verity.core.protocols.certifier import warden_rows as WR  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import draw as D, partition as P, produced, registration as R  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import registered as G  # noqa: E402
from verity.ml.windows import rows as CS  # noqa: E402
from verity_vllm.commit import index as IX  # noqa: E402
from verity_vllm.pipeline import serving_rows as SRW  # noqa: E402
from verity.ml.definitions.registry.b1_tp2 import AllReduce2  # noqa: E402
from verity_vllm.commit import index as IX  # noqa: E402

#: the stand-in's TP run root, which every rank's window and every partial's domain bind
ROOT = hashlib.sha512(b"verity/a5/tp2-gates-stand-in/v1").hexdigest()
#: the TP step the partials are values of, which their domains bind
PROGRAM = "a5_tp_gates/llama32-1b-tp2/step0"
RULE = "work:2"
RANKS = (0, 1)
#: the step's cross-rank reductions, each AllReduce2_v1{H} in the registered compiler's TP2 rank Programs (`b1_tp2`)
COLLECTIVES = ("model.embed_tokens", "model.layers.0.self_attn.o_proj", "model.layers.0.mlp.down_proj")
#: the rank whose received partial the refused runs move a bit through, and the sender's port it reads it on
RECEIVER = 1
#: the rest's rows: WardenRowIdentity{REST_BYTES}, another template than the gates', so another stratum; a row and its salt
#: must span more than one SHA-512 block for the composer
REST_BYTES = 128
#: the receiving port's statement made for another read than the Program's (`statement`), and what Lean's registered reads,
#: and the host's read of them (`warden_rows.refusal`), refuse it by
MADE_FOR = {"swapped": "reads position", "other-collective": "not another value", "other-request": "not another value",
            "own": "not another value"}
TRANSFERS = {"altered": "does not open", **MADE_FOR}


# --- the unit ---------------------------------------------------------------------------------------------------------

#: each port's row: rank j's partial, the registered AllReduce2_v1's input ``parts`` row j
PORT_PARAMS = {f"part{j}": f"parts[{j}]" for j in RANKS}


def sums(part0: np.ndarray, part1: np.ndarray) -> np.ndarray:
    """AllReduce2_v1's sums of two partials' bf16 words: `Bf16Add_v1`, the word view of its `Bf16Add_v2`."""
    a, b = np.asarray(part0, np.uint16), np.asarray(part1, np.uint16)
    return np.asarray(E.evaluate_batch(W.Bf16Add, a.reshape(-1), b.reshape(-1)), np.uint16).reshape(a.shape)


def bf16(x: np.ndarray) -> np.ndarray:
    """f32 values rounded to bf16 words, to nearest even (finite values only)."""
    u = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def templates(hidden: int) -> tuple:
    """The transfer-gate template, the registered AllReduce2_v1{H} the TP2 rank Programs end each reduction in, and the
    rest's."""
    return bind(AllReduce2, N=hidden), bind(WardenRowIdentity, B=REST_BYTES)


# --- the partials --------------------------------------------------------------------------------------------------------

def value_name(r: int, q: int, k: int) -> str:
    return f"tp2/request{q}/rank{r}/{COLLECTIVES[k]}"


def sites(requests: int) -> list[tuple[int, int]]:
    """The transfer-gate chunks' (request, collective), in window order."""
    return [(q, k) for q in range(requests) for k in range(len(COLLECTIVES))]


def partials(hidden: int, tokens: int, requests: int = 2, seed: int = 0) -> dict[int, list[list[G.Registered]]]:
    """The prover's registered partials, ``[r][q][k]``: rank r's of collective k for request q, the request's `tokens` rows of
    `hidden` bf16 words, under fresh salts and the domain binding the TP run root."""
    rng = np.random.default_rng(seed)
    return {r: [[G.Registered.commit(PROGRAM, value_name(r, q, k), bf16(rng.normal(0, 0.05, (tokens, hidden))), word_bits=16,
                                     schema=V2, window=ROOT) for k in range(len(COLLECTIVES))] for q in range(requests)]
            for r in RANKS}


def domain(r: int, q: int, k: int, hidden: int, tokens: int):
    """The verifier's own domain of rank r's partial of collective k for request q."""
    return G.domain(PROGRAM, value_name(r, q, k), hidden, tokens, 16, window=ROOT)


def own_entry(r: int, q: int, k: int, hidden: int, tokens: int) -> dict:
    """The verifier's own name, shape and domain of rank r's partial of collective k for request q."""
    return {"value": value_name(r, q, k), "leaves": tokens, "words": hidden,
            "domain": domain(r, q, k, hidden, tokens).domain_id.hex()}


def statement(parts: dict, r: int, j: int, q: int, k: int, kind: str | None = None) -> SimpleNamespace:
    """Rank r's port ``part<j>`` of chunk q<q>c<k> as the prover makes its statement: rank j's partial of collective k for
    request q at every token row (None), or for another read: ``swapped``, tokens 0 and 1 reading each other's rows;
    ``other-collective``, rank j's partial of the next collective; ``other-request``, rank j's partial of collective k for the
    next request; ``own``, rank r's own partial (j ≠ r); ``altered``, the Program's rows with row 0 one bit off (the lowest
    mantissa bit of its word 0) under its salt. Its rows, salts and the header's entry."""
    tree, positions = parts[j][q][k], list(range(parts[j][q][k].leaves))
    if kind == "swapped":
        if len(positions) < 2:
            raise SystemExit("swapped exchanges two tokens' rows: --tokens ≥ 2")
        positions[:2] = positions[1::-1]
    elif kind == "other-collective":
        tree = parts[j][q][(k + 1) % len(COLLECTIVES)]
    elif kind == "other-request":
        if len(parts[j]) < 2:
            raise SystemExit("other-request reads another request's partial: --requests ≥ 2")
        tree = parts[j][(q + 1) % len(parts[j])][k]
    elif kind == "own":
        if j == r:
            raise ValueError("own reads the receiving rank's own partial on its peer's port: j ≠ r")
        tree = parts[r][q][k]
    elif kind not in (None, "altered"):
        raise ValueError(f"a statement is made for the Program's reads or one of {sorted(TRANSFERS)}, not {kind!r}")
    rows, salts, entry = tree.statement(positions)
    rows = [np.asarray(x, np.uint16) for x in rows]
    if kind == "altered":
        rows[0] = rows[0].copy()
        rows[0][0] ^= 1
    return SimpleNamespace(rows=rows, salts=salts, entry=entry, positions=positions, value=tree.value)


def bc(row: np.ndarray, salt: bytes) -> bytes:
    """A partial's row's hm96 commitment string ``b || c`` under its salt, its bit row the row's u16 words little-endian."""
    return WR.bc(np.asarray(row, "<u2").tobytes(), salt)


# --- the window ------------------------------------------------------------------------------------------------------------

def window(r: int, hidden: int, tokens: int, requests: int, rest: int) -> CS.Window:
    """Rank r's window: a chunk of `tokens` transfer-gate units for each request's collective, then the rest."""
    tg, other = templates(hidden)
    units = len(sites(requests)) * tokens
    program, query = CS.population_program([(tg, units), (other, rest)])
    (tid, _, _), (rid, _, _) = P.ranges(program, query)
    law = D.parse_law(RULE)
    ports = tuple(f"part{j}" for j in RANKS)
    gates = tuple(CS.Member(tid, rows=tuple((p, hidden) for p in ports), outs=(("o0", hidden),), n=tokens, base=u * tokens,
                            tables=tuple((p, tokens) for p in ports), name=f"{tid}/q{q}c{k}",
                            schemas=tuple((p, V2) for p in (*ports, "o0"))) for u, (q, k) in enumerate(sites(requests)))
    w = REST_BYTES // 2
    others = (CS.Member(rid, rows=(("row", w),), outs=(("o0", w),), n=rest, base=units, schemas=(("row", V2), ("o0", V2))),)
    return CS.Window(program, query, law, f"{ROOT}/rank{r}", gates + others,
                     work_table=SRW.work_table_of(law, {tid: units, rid: rest}, [tid]))


def chunks(mine: CS.Window) -> list[tuple[int, int, CS.Member]]:
    """The transfer-gate chunks, q<q>c<k> request q's collective k's, with their request and collective."""
    gates = [m for m in mine.members if m.template == mine.members[0].template]
    return [(q, k, m) for (q, k), m in zip(sites(len(gates) // len(COLLECTIVES)), gates, strict=True)]


def step(r: int, hidden: int, tokens: int, requests: int) -> dict:
    return {"stand-in": "benchmarks/one_stage/a5_tp_gates.py", "tp_run_root": ROOT, "program": PROGRAM, "world": len(RANKS),
            "rank": r, "collectives": list(COLLECTIVES), "hidden": hidden, "requests": requests, "tokens": tokens}


def provenance(parts: dict, r: int) -> dict:
    """What rank r's registration names: the step, its own partials' entries and the peer partials' it reads, by value."""
    tree, requests = parts[r][0][0], len(parts[r])
    return {**step(r, tree.words, tree.leaves, requests),
            "partials": {value_name(r, q, k): parts[r][q][k].entry() for q, k in sites(requests)},
            "peers": {value_name(1 - r, q, k): parts[1 - r][q][k].entry() for q, k in sites(requests)}}


def rest_rows(m: CS.Member) -> np.ndarray:
    w = dict(m.rows)["row"]
    return np.asarray([WR.words(hashlib.shake_256(b"a5-tp-gates/rest/%d" % i).digest(2 * w)) for i in range(m.n)], np.uint16)


def commit(out: Path, mine: CS.Window, parts: dict, r: int, altered: bool = False, made: str | None = None) -> CS.Committed:
    """Rank r's window committed into ``out`` as serving would commit it: each chunk's ``part<j>`` table rank j's partial's rows
    under rank j's salts, its header naming the value, positions and paths, its ``o0`` the sums of the rows it read, and the
    rest under fresh salts. ``altered`` / ``made``: chunk q0c0's port of the sender's partial (``part<1 − r>``) made for
    that read (`statement`)."""
    data, salts_of, registered, entries = [], {}, {}, []
    gates = {m.key: (q, k) for q, k, m in chunks(mine)}
    for m in mine.members:
        if m.key not in gates:
            rows = rest_rows(m)
            data.append({"rows": {"row": rows}, "outs": {"o0": rows}})
            entries.append([[m.base, m.base + m.n, "rest", 0, 0, "rest", m.key, 0, 0]])
            continue
        (q, k), tables = gates[m.key], {}
        for j in RANKS:
            kind = ("altered" if altered else made) if ((q, k) == (0, 0) and j != r) else None
            st = statement(parts, r, j, q, k, kind)
            tables[f"part{j}"] = np.stack(st.rows)
            salts_of[m.port(f"part{j}")] = b"".join(st.salts)
            registered.setdefault(m.key, {})[f"part{j}"] = st.entry
        data.append({"rows": tables, "outs": {"o0": sums(tables["part0"], tables["part1"])},
                     "refs": np.repeat(np.arange(m.n)[:, None], len(RANKS), axis=1)})
        entries.append([[m.base + i, m.base + i + 1, f"synthetic/{q}", 0, 0, COLLECTIVES[k], m.key, i, i] for i in range(m.n)])
    c = CS.commit(mine, data, salts_of=salts_of)
    CS.write(c, out, {"of": provenance(parts, r), "committer": {"module": "a5_tp_gates.py"}},
             IX.index_doc(mine.members, entries, mine.run), kind="stand-in", registered=registered, registrar=R)
    return c


# --- the verifier ----------------------------------------------------------------------------------------------------------

def registrations(windows: Path) -> dict[int, dict]:
    """Every rank's registration, as the verifier reads it before any rank's coin."""
    return {r: json.loads((windows / f"rank{r}" / "registration.json").read_text()) for r in RANKS}


def reads(mine: CS.Window, r: int, regs: dict[int, dict]) -> dict[str, dict]:
    """The verifier's own ``--registered`` file of each of rank r's chunks: ``part<j>`` at the root rank j's registration names
    for its partial of collective k for request q, unit i (from the chunk's first) reading row i. Refused while any rank's
    registration is unread: a rank's ports read every rank's registered partials, so every registration comes before any
    rank's coin."""
    if missing := [j for j in RANKS if j not in regs]:
        raise ValueError(f"rank {r}'s ports read the registered partials of ranks {missing}, whose registrations the verifier "
                         "hasn't read: every rank's registration is read before any rank's coin")
    out = {}
    for q, k, m in chunks(mine):
        ports = {}
        for j in RANKS:
            e = own_entry(j, q, k, dict(m.rows)["part0"], m.n)
            root = regs[j]["window"]["of"]["partials"][e["value"]]["root"]
            ports[f"part{j}"] = produced.port(e["value"], e, root, list(range(m.n)), V2) | ({G.READS_FROM: m.base} if m.base else {})
        out[m.key] = WR.reads_file(ports)
    return out


def source_of(mine: CS.Window, r: int, regs: dict[int, dict]):
    """The verifier's check of rank r's registration's window: a stand-in of rank r's run, of this step at the TP run root, its
    partials each the verifier's own shape and domain (any root), and its peers exactly the partials rank 1 − r registered."""
    gates = chunks(mine)
    hidden, tokens, requests = dict(gates[0][2].rows)["part0"], gates[0][2].n, len(gates) // len(COLLECTIVES)
    peers = regs[1 - r]["window"]["of"].get("partials")
    want = {value_name(r, q, k): own_entry(r, q, k, hidden, tokens) for q, k in sites(requests)}

    def partials_ok(got) -> bool:
        if not isinstance(got, dict) or sorted(got) != sorted(want):
            return False
        for name, own in want.items():
            e = got[name]
            if not isinstance(e, dict) or sorted(e) != sorted(G.ENTRY_KEYS) or {x: e[x] for x in own} != own:
                return False
            if not (isinstance(e["root"], str) and len(e["root"]) == 128 and all(ch in "0123456789abcdef" for ch in e["root"])):
                return False
        return True

    def source(w: dict) -> dict:
        of, at = w.get("of") or {}, step(r, hidden, tokens, requests)
        ok = {"kind": w.get("kind") == "stand-in", "run": w.get("run") == mine.run,
              "step": {x: of.get(x) for x in at} == at,
              "partials": partials_ok(of.get("partials")), "peers": of.get("peers") == peers}
        return {"ok": all(ok.values()), "codes": [c for c, v in ok.items() if not v], "detail": {"kind": w.get("kind"), "run": w.get("run")}}
    return source


def transfer_refusals(mine: CS.Window, parts: dict, r: int, regs: dict[int, dict]) -> dict[str, dict]:
    """Rank r's q0c0 port of the sender's partial (``part<1 − r>``) opened by the statement of each `TRANSFERS` kind, read as
    Lean's registered reads read it (`warden_rows.refusal`) against the verifier's own port: each refused by its reason."""
    j, (q, k, m) = 1 - r, chunks(mine)[0]
    port = reads(mine, r, regs)[m.key]["ports"][f"part{j}"]
    dom = domain(j, q, k, dict(m.rows)["part0"], m.n)
    honest = statement(parts, r, j, q, k)
    if (why := WR.refusal(port, dom, honest.entry, [bc(x, s) for x, s in zip(honest.rows, honest.salts)])) is not None:
        raise SystemExit(f"rank {r}: the host's read refuses the Program's own statement of {port['value']}: {why}")
    out = {}
    for kind, reason in TRANSFERS.items():
        st = statement(parts, r, j, q, k, kind)
        why = WR.refusal(port, dom, st.entry, [bc(x, s) for x, s in zip(st.rows, st.salts)]) or ""
        out[f"received-partial-{kind} (registered reads)"] = {
            "refused": why.startswith("registered reads:") and reason in why, "why": why, "value": st.value,
            "positions": st.positions[:2]}
    return out


def negatives(mine: CS.Window, parts: dict, r: int, regs: dict[int, dict]) -> dict[str, dict]:
    """Rank r's registration naming its peer's partial under another root, its own partial as its peer's, another request's
    partial as its peer's, a partial under another TP run's domain, and the other rank's registration as its own, each
    refused by `source`; its ports built before the peer's registration is read, refused by `reads`; and the receiving port's
    statements of `TRANSFERS`."""
    source, w = source_of(mine, r, regs), regs[r]["window"]
    of, own, peer = w["of"], value_name(r, 0, 0), value_name(1 - r, 0, 0)
    other_request = value_name(1 - r, 1 % len(parts[r]), 0)
    other_root = hashlib.sha512(b"verity/a5/tp2-gates-stand-in/another-run").hexdigest()
    elsewhere = G.Registered.commit(PROGRAM, own, parts[r][0][0].rows, word_bits=16, schema=V2, window=other_root)

    def named(**change) -> dict:
        s = source({**w, "of": {**of, **change}})
        return {"refused": not s["ok"], "codes": s["codes"]}
    s = source(regs[1 - r]["window"])
    return {"registration-naming-the-peers-partial-under-another-root (source)":
            named(peers={**of["peers"], peer: {**of["peers"][peer], "root": "00" * 64}}),
            "registration-naming-its-own-partial-as-the-peers (source)": named(peers={**of["peers"], peer: of["partials"][own]}),
            "registration-naming-another-requests-partial-as-the-peers (source)":
            named(peers={**of["peers"], peer: of["peers"][other_request]}),
            "partial-under-another-tp-runs-domain (source)": named(partials={**of["partials"], own: elsewhere.entry()}),
            "registration-of-the-other-rank (source)": {"refused": not s["ok"], "codes": s["codes"]},
            "ports-before-the-peers-registration (reads)": refusal(lambda: reads(mine, r, {r: regs[r]}),
                                                                   "before any rank's coin")} | transfer_refusals(mine, parts, r, regs)


def gate_rows(mine: CS.Window, regs: dict[int, dict]) -> list[dict]:
    """Each chunk's request, collective, units and the registered partial each port opens: unit base + i reads row i of both."""
    return [{"chunk": m.key, "request": q, "collective": COLLECTIVES[k], "units": [m.base, m.base + m.n],
             "ports": {f"part{j}": {"value": value_name(j, q, k), "rank": j,
                                    "root": regs[j]["window"]["of"]["partials"][value_name(j, q, k)]["root"]} for j in RANKS}}
            for q, k, m in chunks(mine)]


def composer(defn, hidden: int, cache: Path | None = None):
    """``a5.audit``'s composer of the transfer-gate template's circuit: C-Flock's `UnitStager` over its ports ``part0`` and
    ``part1`` (16·H bits each, the rows ``parts[0]`` and ``parts[1]`` of AllReduce2_v1's input) in and ``o0`` out, bound to
    the window's partition and program. ``cache``: a directory of the
    circuits composed so far, by template, partition and program, which both ranks' windows and every run share."""
    def compose(circuit: Path, partition: str, program: str) -> dict:
        from verity_flock.class_statement import UnitStager

        key = hashlib.sha512(json.dumps([defn.id, partition, program]).encode()).hexdigest()[:32]
        if cache is not None and (cache / f"{key}.json").exists():
            shutil.copyfile(cache / f"{key}.txt", circuit)
            return json.loads((cache / f"{key}.json").read_text())
        st = UnitStager(defn, "unit/a5-tp-gates", [(p, V2, 16 * hidden, [PORT_PARAMS[p]]) for p in PORT_PARAMS], [(V2, (0,))],
                        circuit=circuit, bind={"partition": partition, "program": program})
        made = {"subcircuit": st.meta.get("subcircuit"), "pin": st.pin, "class_sha512": st.class_sha512, "k_log": st.meta["k_log"]}
        if cache is not None:
            cache.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(circuit, cache / f"{key}.txt")
            (cache / f"{key}.json").write_text(json.dumps(made))
        return made
    return compose


def expected(a: argparse.Namespace, ranks: dict[int, dict]) -> bool:
    """Honest: both ranks accepted with every negative refused. ``--altered`` (``--made-for``): rank 0 accepted, every check
    before rank 1's draw passed, and Lean's verify of rank 1's chunk q0c0 refused at its registered reads by its reason."""
    if not (a.altered or a.made_for):
        return all(x["ok"] for x in ranks.values())
    reason, s = TRANSFERS["altered" if a.altered else a.made_for], ranks[RECEIVER]
    why = ((s.get("audit") or {}).get("why") or {}).get("lean-verify/m0") or ""
    return all(x["ok"] for r, x in ranks.items() if r != RECEIVER) and all((s.get("checks_before_draw") or {}).values()) \
        and bool(s.get("checks_before_draw")) and (s.get("audit") or {}).get("accepted") is False \
        and "registered reads" in why and reason in why


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="a5_tp_gates.py", description=__doc__.splitlines()[0])
    ap.add_argument("--hidden", type=int, default=2048, help="H: each partial's bf16 words a token (LLAMA32_1B's 2048)")
    ap.add_argument("--requests", type=int, default=2, help="Q: the step's requests, each with partials of its own")
    ap.add_argument("--tokens", type=int, default=2, help="T: each request's token rows, each of its collectives' chunk's units")
    ap.add_argument("--rest", type=int, default=8, help="the rest's units, drawn by the work law")
    ap.add_argument("--seed", type=int, default=0, help="the synthetic activations' seed")
    ap.add_argument("--m0-pythonpath", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--lean", required=True)
    ap.add_argument("--firewall", required=True)
    ap.add_argument("--coins", choices=["verifier"], required=True, help="draw each rank on the verifier's own live coins")
    ap.add_argument("--altered", action="store_true",
                    help="rank 1 receives rank 0's embedding partial of request 0's row 0 one bit off")
    ap.add_argument("--made-for", choices=sorted(MADE_FOR), help="rank 1's port of rank 0's embedding partial of request 0: "
                    "its statement made for another read than the Program's")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--delta", type=float, default=0.5)
    ap.add_argument("--port", type=int, default=0, help="the sessions' first port (0: a free block)")
    ap.add_argument("--session-timeout", type=float, default=3600)
    ap.add_argument("--compose-cache", type=Path, help="the transfer-gate circuits composed so far, shared by runs on this machine")
    a = ap.parse_args(argv)
    if a.altered and a.made_for:
        ap.error("one refused statement a run: --altered or --made-for")
    if a.requests < 2:
        ap.error("--requests ≥ 2: the gates bind each request's peer reads to its own partials, and not another request's")
    import a5

    out = a.out
    parts = partials(a.hidden, a.tokens, a.requests, a.seed)
    mine = {r: window(r, a.hidden, a.tokens, a.requests, a.rest) for r in RANKS}
    committed = {r: commit(out / "window" / f"rank{r}", mine[r], parts, r, a.altered and r == RECEIVER,
                           a.made_for if r == RECEIVER else None) for r in RANKS}
    regs = registrations(out / "window")
    tg, rid = mine[0].members[0].template, mine[0].members[-1].template
    defs = dict(zip((tg, rid), templates(a.hidden)))
    sets = {tg: (composer(defs[tg], a.hidden, a.compose_cache), defs[tg]), rid: (row_composer(defs[rid], 8 * REST_BYTES), defs[rid])}
    a.rule, a.window_root, a.compose_cache, a.transfer_gate = RULE, ROOT, None, [tg]
    a.verifier_coins, a.coins = True, None
    port = a.port or free_ports(a5.RANK_PORTS * len(RANKS))
    ranks: dict[int, dict] = {}
    for r in RANKS:
        s, ok = a5.audit(a, sets, out / f"rank{r}", mine[r], out / "window" / f"rank{r}", source_of(mine[r], r, regs),
                         port + a5.RANK_PORTS * r, {"rank": r, "standin": step(r, a.hidden, a.tokens, a.requests)},
                         more_negatives=negatives(mine[r], parts, r, regs), rank=r, registered=reads(mine[r], r, regs))
        ranks[r] = {"ok": ok, **{x: s.get(x) for x in ("checks_before_draw", "refused", "audit", "draw", "draw_counts",
                                                      "negatives", "coins")}}
    o0 = {r: [committed[r].rows[m.port("o0")] for _q, _k, m in chunks(mine[r])] for r in RANKS}
    summary = {"root": ROOT, "program": PROGRAM, "coins": "verifier", "altered": a.altered, "made_for": a.made_for,
               "receiver": RECEIVER, "hidden": a.hidden, "requests": a.requests, "tokens": a.tokens,
               "collectives": list(COLLECTIVES),
               "registrations_before_coins": sorted(regs),
               "sums_agree": [bool(np.array_equal(x, y)) for x, y in zip(o0[0], o0[1])],
               "window": {"population": mine[0].population, "transfer_gates": [tg], "work_table": mine[0].work_table,
                          "members": [{"key": m.key, "template": m.template, "n": m.n, "base": m.base} for m in mine[0].members]},
               "gates": {r: gate_rows(mine[r], regs) for r in RANKS}, "ranks": ranks,
               "accepted": all(x["ok"] for x in ranks.values())}
    summary["as_expected"] = expected(a, ranks)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    print("A5-TP-GATES " + json.dumps({x: summary[x] for x in ("accepted", "as_expected", "altered", "made_for", "coins", "sums_agree")}
                                      | {"ranks": {r: {"ok": x["ok"], "draw": x.get("draw"), "why": (x.get("audit") or {}).get("why")}
                                                   for r, x in ranks.items()}}, sort_keys=True, default=str))
    return 0 if summary["as_expected"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
