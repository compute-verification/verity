"""Remote's tie end to end on a toy: one H1 window at K = 2048 and the V3c window circuit that audits it, registered as
the remote composite reads them (`Core.Remote`: each period window's Commit record is the one record one-stage audits).

**The toy** (`_toy`): a one-layer Llama of hidden 2048 at the exact-int8 checkpoint, vocabulary 512, at class (1 slot,
prompts of 32, 2 outputs). So H1's window has 2 rounds, round 0 the prompt's 32 rows and round 1 one decode row, and its
tables at K = 2048 are M = 32 rows a call and 3 calls a round (vLLM's fused qkv, o and gate/up). Its proof blocks can be
proved: C-Flock's composer needs a row past 111 bytes in every template, which takes prompts of at least 27 ids
(`TokenFeed_v1`) and a vocabulary of at least 448 (the token select). H1's window Program is a stand-in digest, since the tie
reads only its digest, its open root and its tables' names.

* **W**, H1's window registration (`pouw_h1_commit.registration`): each round's A table of K and input-ids table, made H1's
  way from the served values (`chunk_words` of each call's int8 codes and scale over M slots, call after call; `ids_words`
  of the step's input ids), committed under H1's run key at H1's nonces and domains (round 0's ids bound to the session).
* **R**, the audited window's registration, as `prove_path.py window-commit --window-registration W --h1-key` makes it:
  the window circuit with H1's tables (`window_circuit.lower` with `Tables`), its units filled from the same values
  (`window_rows.Rows`), its tables W's rebuilt from the rows its proof blocks make (`pouw_h1_commit.reuse`), recorded,
  committed and written through `interface.Window` as a v2 record chained right after W.

**Honest** (slow): every row a proof block holds to a table is W's row there, every served row of W is held by one, and
`reuse` makes each table W's entry of it under the circuit's name.
**One flipped table row** (slow): flipping a bit of round 1's o-projection row changes W's leaf there alone, so the honest
reader's row opens against no root W registers and `reuse` refuses W'; that reader is the one proof block of a template
of its own, which `stratified:1` always draws; and a prover who records the flipped row has that proof block off its
Definition, so its proof fails.

**Records** (opt-in, `VERITY_PROVE_PATH_REMOTETIE`: the window's record, commit and write take minutes): credit's ROOTS1 on R
(`credit._arow`) passes, R reuses W's tables (`pouw_h1_commit.reused`) and is the record H1 names (`named`), its digest is
credit's (`commit_of`) and one-stage's, and R1-R7 (`registration.check` with the log [W] and each table an output value of
W's) accept it. With the flipped row they accept W' and an R' written by hand after it: the records can't see a row, and
only the reader's proof refuses it.

**Lean** (opt-in as the records, on a machine with Lean's toolchain): `served_cases.py --serve-dir` on a serve dir under H1's tie (V3c's
audit's v1 case of R, H1's log of W then R, and PoUW's period case of one window naming R) writes three cases: one-stage
v1's, and those of two of remote's three subprotocols, the inference-only policy's and the period's.
`sampled-proofs-one-stage check` (whose log record reader takes W and whose R1-R7 accept R) and `inference-only check`
accept theirs, each naming R's digest, and the period's window names R (`served_cases.period_case`, the record tie remote
checks). The v1 case's verdict of record is an accepting stand-in reporting R's tables' roots as their readers opened them:
the proof blocks' proofs aren't run here, so the audit holds no case of one-stage v2, the audit on the proofs themselves.
That is the only case remote's third subprotocol, sampled proofs, takes (`VStar.Recursive.Sampled` refuses one-stage v1's,
whose verdicts are the evidence's own), so `served_cases` refuses to write it. `composite.py remote` on the two cases the
serve dir has of remote's subprotocols rejects naming sampled proofs missing: a composition answers no subprotocol's case
while one has none (`Core.Composition.run`), so remote reads neither the policy's case nor the period's, whose Lean
refuses it for want of an H1 window's proofs, and holds no tie, which it does only where no subprotocol rejects
(`Core.Composition.holdTies`). So no test here hands remote a whole composition: the tie is `served_cases`' check, not
remote's, and remote carrying a subprotocol's refusal, holding its own tie and accepting all three are a served window's.
A period naming W is refused by `served_cases` before any composite. With the flipped row, one-stage's Lean accepts R' on
the log [W'] and refuses it on [W].
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import settings
from verity_vllm.program.model import INT8_EXACT_QUANT
from verity_vllm.program.predict import scan_state as SS
from verity_vllm.program.predict import window_circuit as WC
from verity_vllm.program.predict import window_rows as WRS
from verity_vllm.program.predict import window_weights as WW
from verity_vllm.program.predict.phi import Phi
from verity_vllm.program.predict.steps import Registered
from verity_vllm.program.predict.window_class import Tables, WindowClass
from verity_vllm.pipeline import serve_tie as TIE
from verity_vllm.protocols.pouw.h1 import commit as CT
from verity_vllm.protocols.pouw.h1 import registered as NR
from verity_vllm.protocols.pouw.h1 import served as SV
from verity_vllm.protocols.pouw.h1.words import chunk_words, ids_words

from verity.protocols.pouw.verifier import credit as CR
from verity.protocols.spot.verifier.one_stage import draw as D, partition as P, registered as G, registration as REG
from verity.circuits.codec import canonical_json
from verity.circuits.evaluation import evaluate_batch
from verity.commitments.windows import interface as IF, rows as CS

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import records_audit as RA  # noqa: E402

K, M, CALLS, ROUNDS = 2048, 32, 3, 2
CONFIG = {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_size": K, "intermediate_size": 256,
          "num_attention_heads": 32, "num_key_value_heads": 8, "num_hidden_layers": 1, "vocab_size": 512,
          "rms_norm_eps": 1e-5, "tie_word_embeddings": True, "max_position_embeddings": 64, "hidden_act": "silu",
          "quantization_config": dict(INT8_EXACT_QUANT)}
TARGET = {"compute_capability": [12, 0], "num_sms": 188, "int8_exact": "linears"}
CLASS, TABLES = WindowClass(1, 32, 2), Tables(K, M, CALLS)
#: H1's linears at K and their calls, a round each linear once (`pouw_h1_served.Served`'s)
LINEARS = [("qkv", K, 3072), ("o", K, K), ("gate_up", K, 512)]
#: the committers' secret keys: V3c's, and H1's run key
KEY, H1_KEY = bytes(range(32)), bytes(range(64, 96))
H1 = hashlib.sha512(b"the H1 window Program's descriptor").hexdigest()
OPEN = hashlib.sha512(b"the H1 window's open root").hexdigest()
#: the V3c window's run root
RUN = "ab" * 64
LAW = D.parse_design("stratified:1")
PROMPT = [(7 * i + 3) % 500 for i in range(CLASS.prompt)]
#: the flipped row: round 1's o-projection's (call 1 of the round) first slot
FLIP = ("row-table/1/k2048", M * 1)
RECORDS = pytest.mark.skipif(not settings.REMOTETIE, reason="VERITY_PROVE_PATH_REMOTETIE=1 records, commits and writes the "
                             "toy's 11,252 proof blocks and builds the Lean verifiers that check them, a few minutes")
ROOT = HERE.parents[2]
ONE_STAGE = ROOT / "benchmarks" / "one_stage"
#: why `served_cases` writes no sampled proofs case from the toy's audit (`served_cases.v2_case`)
NO_V2_CASE = "no v2 case: the audit's a5 writes no case-v2.json (a5.case_v2)"


def _bf16(x) -> np.ndarray:
    return (np.asarray(x, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _weights(fn) -> dict[str, np.ndarray]:
    """Finite random weights, each field's elements in their unsigned width: s8 codes, small positive f32 scales (an
    attention's scales 1), norms near 1, the rotary cache in [-1, 1]."""
    rng, out = np.random.default_rng(3), {}
    for label, ty in dict(fn.signature()[0])[WC.WEIGHTS].fields:
        (bits,), n = ty.width_set(), ty.leaves
        if bits == 8:
            out[label] = rng.integers(-127, 128, n).astype(np.int8).view(np.uint8)
        elif bits == 32:
            out[label] = (np.ones(1) if n == 1 else rng.uniform(5e-3, 2e-2, n)).astype(np.float32).view(np.uint32)
        elif "cos_sin" in label:
            out[label] = _bf16(rng.uniform(-1, 1, n))
        elif "norm" in label:
            out[label] = _bf16(1 + 0.1 * rng.standard_normal(n))
        else:
            out[label] = _bf16(0.5 * rng.standard_normal(n))
    return out


def _h1_row(p: int) -> tuple[int, int]:
    """Position p of the one slot as H1 batches it: its prompt's tokens are round 0's rows in order, and its decode step at
    position p >= L round p - L + 1's first row."""
    return (0, p) if p < CLASS.prompt else (p - CLASS.prompt + 1, 0)


def _h1_tables(low, rows, slot) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Each round's A table of K and input-ids table as H1 commits them from the served values: each call's rows its
    quantizer's int8 codes and f32 scale at each of the round's positions (a position's 3 int8 linears at K in node
    order), `chunk_words` over M slots, call after call; and the step's input ids, `ids_words` over M slots."""
    pos, P = WRS.positions(low.fn), low.programs[0]
    quants: dict[int, list] = {}
    for j in sorted(P.packed):
        quants.setdefault(pos[j], []).append(P.packed[j])
    codes = {s: [[] for _ in range(CALLS)] for s in range(ROUNDS)}
    scales = {s: [[] for _ in range(CALLS)] for s in range(ROUNDS)}
    ids: dict[int, list[int]] = {s: [] for s in range(ROUNDS)}
    for p in range(CLASS.positions):
        s, i = _h1_row(p)
        assert len(quants[p]) == CALLS and len(ids[s]) == i
        for c, r in enumerate(quants[p]):
            codes[s][c].append(rows.outs[r.tid][0][r.unit].view(np.int8))
            scales[s][c].append(rows.outs[r.tid][1][r.unit].view("<f4")[0])
        ids[s].append(int(slot.tokens[p]))
    a = {s: np.concatenate([chunk_words(np.array(codes[s][c]), np.array(scales[s][c], "<f4"), 1, M) for c in range(CALLS)])
         for s in range(ROUNDS)}
    return a, {s: ids_words(ids[s], M) for s in range(ROUNDS)}


class _Served(SV.Served):
    """H1's served trees under the stand-in window Program: its A tables named as `h1window.table` names them."""

    def a_table(self, c: int) -> str:
        return f"{SV.PREFIX}a/{K}@{c // len(self.linears)}"


def _window(a_words: dict[int, np.ndarray], id_words: dict[int, np.ndarray]) -> SimpleNamespace:
    """H1's window with these tables committed as `pouw_h1_served.Served` commits them, and its registration W."""
    a = {s: NR.commit(H1, f"{SV.PREFIX}a/{K}@{s}", w, H1_KEY, SV.nonce(s * CALLS, "a"), keep_rows=True, window=OPEN)
         for s, w in a_words.items()}
    ids = {s: NR.commit(H1, f"{SV.PREFIX}ids@{s}", w, H1_KEY, SV.IDS_NONCE + s, keep_rows=True,
                        **({"session": SV.LOCAL_SESSION} if s == 0 else {"window": OPEN})) for s, w in id_words.items()}
    served = _Served(program=H1, program_digest=H1, root=OPEN, registered={v.value: v for v in a.values()},
                     linears=list(LINEARS), calls=[(n, b) for _ in range(ROUNDS) for b, (n, _k, _n) in enumerate(LINEARS)],
                     m=M, ids=ids)
    run = SimpleNamespace(served=served, entry="toy-window", steps=ROUNDS)
    run.W = CT.registration(run, None)
    run.tables = {**{TABLES.a(s): a[s] for s in a}, **{TABLES.ids(s): ids[s] for s in ids}}
    return run


def _names() -> dict[str, str]:
    """The circuit's name of each table, -> W's (`credit.commit_rows`, `credit.commit_ids`)."""
    return {**{TABLES.a(s): f"{SV.PREFIX}a/{K}@{s}" for s in range(ROUNDS)},
            **{TABLES.ids(s): f"{SV.PREFIX}ids@{s}" for s in range(ROUNDS)}}


def _layout() -> dict:
    """H1's window Program's layout (`pouw_h1_served.write_program`'s meta) as the tie reads it."""
    return {"calls": [[n, b] for _ in range(ROUNDS) for b, (n, _k, _n) in enumerate(LINEARS)],
            "linears": [list(x) for x in LINEARS], "m": M}


def _reuse(toy, W: dict) -> dict:
    return CT.reuse(W, _layout(), ROUNDS, H1_KEY, toy.rows.value)


def _reads(circuit) -> dict[tuple[str, str, int], tuple[str, int]]:
    """Each output row a table read holds, ``(template, port, proof block)``, as its table and row."""
    out = {}
    for r in (r for r in circuit.registered.reads if r.param in circuit.registered.values):
        words = dict(circuit.layouts[r.reader][0].outs)[r.port]
        for i, leaf in enumerate(r.source(range(r.units))):
            out[(r.reader, r.port, r.at + i)] = (r.param, int(leaf) // words)
    return out


def _flipped(run) -> SimpleNamespace:
    """``run``'s window with bit 0 of row `FLIP` of its table flipped, committed alike."""
    a = {s: np.array(run.tables[TABLES.a(s)].rows[:], np.uint16) for s in range(ROUNDS)}
    s = int(FLIP[0].split("/")[1])
    a[s][FLIP[1], 0] ^= 1
    return _window(a, {s: np.array(run.tables[TABLES.ids(s)].rows[:], np.uint16) for s in range(ROUNDS)})


@pytest.fixture(scope="module")
def toy() -> SimpleNamespace:
    """The toy's window circuit, its honest rows from one served slot (a prompt of 32, 2 outputs) and H1's window over the
    same values."""
    reg = Registered.of(Phi(model={"repo": "org/toy", "revision": "r", "hf_config": CONFIG}, target=TARGET), default_seed=7)
    low = WC.lower(reg, CLASS, None, WC.Grain(), tables=TABLES)
    fields = _weights(low.fn)
    state = WRS.standin_state(WC.NAME, CLASS, [([1, CLASS.prompt, *PROMPT], CLASS.outputs)], KEY, 0, window=RUN)
    values = {**WW.weight_values(reg, fields, KEY, keep_rows=True), SS.SCAN_STATE: state}
    slots = [WRS.Slot(low.fn, CLASS, row, lambda label, lo, hi: fields[label][lo:hi])
             for row, _ in WRS.scan_slots(low, state.rows)]
    rows = WRS.Rows(low, slots, values)
    return SimpleNamespace(low=low, values=values, slots=slots, rows=rows, h1=_window(*_h1_tables(low, rows, slots[0])))


@pytest.mark.slow
def test_each_table_row_a_proof_block_holds_is_w_s_row_there_and_each_served_row_is_held(toy):
    """The circuit reads each round's tables of W (2 rounds, an A table of 3 calls of 32 rows of K's 1,026-word leaf and an
    input-ids table of 32 rows of a u32), and every row a proof block holds to one is W's row there: the quantizers' leaf
    rows ``o2`` the A rows H1 committed from their codes and scales, the token rows the ids. W's served rows (round 0's
    32 a call, round 1's first) are each held once, and its padding rows by none. So `reuse` makes each table, from the
    rows its proof blocks make, W's entry of it under the circuit's name."""
    c = toy.low.circuit
    assert c.registered.values == ("row-table/0/k2048", "input-ids/0", "row-table/1/k2048", "input-ids/1")
    assert {p: toy.h1.tables[p].leaves for p in c.registered.values} == {
        "row-table/0/k2048": CALLS * M, "input-ids/0": M, "row-table/1/k2048": CALLS * M, "input-ids/1": M}
    reads = _reads(c)
    for (tid, port, u), (table, row) in reads.items():
        assert np.array_equal(toy.rows.outs[tid][int(port[1:])][u], toy.h1.tables[table].rows[row]), (tid, port, u, table, row)
    served = {(TABLES.a(0), M * c_ + i) for c_ in range(CALLS) for i in range(CLASS.prompt)}
    served |= {(TABLES.a(1), M * c_) for c_ in range(CALLS)} | {(TABLES.ids(0), i) for i in range(CLASS.prompt)}
    assert sorted(reads.values()) == sorted(served | {(TABLES.ids(1), 0)})
    w = {e["value"]: e for e in toy.h1.W["registered"]}
    assert {p: v.entry() for p, v in _reuse(toy, toy.h1.W).items()} == {p: {**w[_names()[p]], "value": p}
                                                                        for p in c.registered.values}


@pytest.mark.slow
def test_a_flipped_table_row_opens_against_no_reader_and_its_reader_is_always_drawn_and_off_its_definition(toy):
    """W' (round 1's o-projection row flipped in one bit) differs from W at that leaf alone, so the honest reader's row,
    committed under that leaf's salt, opens against no root W' registers, and `reuse` refuses W'. Its reader is the one
    proof block of its template (`Int8RowQuant_v5` at round 1 and the o-projection's R), a stratum of its own whose sample
    under `stratified:1` is that proof block (`draw.stratum_k`; R's own strata in the records' tier); and its Definition on
    its input gives the honest row, so a prover that records W''s row instead has a proof block whose proof fails."""
    flipped = _flipped(toy.h1)
    honest, other = toy.h1.tables[FLIP[0]], flipped.tables[FLIP[0]]
    assert honest.root != other.root
    assert [j for j, (x, y) in enumerate(zip(honest.levels[0], other.levels[0])) if x != y] == [FLIP[1]]
    with pytest.raises(ValueError, match=f"^{FLIP[0]}: the rows the audited circuit's proof blocks make of it are not"):
        _reuse(toy, flipped.W)
    (tid, port, u), = [k for k, v in _reads(toy.low.circuit).items() if v == FLIP]
    units = toy.low.circuit.units
    assert units[tid] == 1 and "ROUND=1" in tid and port == "o2" and u == 0
    assert D.stratum_k(LAW["k"], units[tid], sum(units.values())) == 1
    j = next(j for j, r in toy.low.programs[0].packed.items() if (r.tid, r.unit) == (tid, u))
    x = toy.slots[0].operand(toy.low.fn.body.nodes[j].args[0])
    got = np.asarray(evaluate_batch(toy.rows.fns[tid], x[None])).reshape(-1)[K + 1:].astype("<u4").view("<u2")
    assert np.array_equal(got, honest.rows[FLIP[1]]) and not np.array_equal(got, other.rows[FLIP[1]])


@pytest.fixture(scope="module")
def written(toy, tmp_path_factory) -> SimpleNamespace:
    """The toy's window recorded, committed and written as window-commit writes it after W (R in its
    ``registration.json``), and the verifier's own window from its records."""
    c, out = toy.low.circuit, tmp_path_factory.mktemp("window")
    win = IF.Window(c, RUN, LAW, 64, registered={**toy.values, **_reuse(toy, toy.h1.W)}, forms=WRS.FORMS)
    win.supply(G)
    toy.rows.record(win)
    committed = win.close()
    prov = {"window_circuit": {"class": [CLASS.slots, CLASS.prompt, CLASS.outputs, CLASS.lanes, CLASS.every], "window": 0},
            "activations": {"program": WC.NAME, "values": {SS.SCAN_STATE: toy.values[SS.SCAN_STATE].entry()}}}
    win.write(committed, out, prov, registrar=REG, index_salt=WRS.index_salt(KEY, 0), previous=REG.digest(toy.h1.W))
    mine, _doc = IF.of_records(out, LAW, RA.catalog_resolve)
    publics = {m.key: (out / f"m{k}-public.bin").read_bytes() for k, m in enumerate(mine.members)}
    return SimpleNamespace(dir=out, R=json.loads((out / "registration.json").read_text()), mine=mine,
                           leaf_layer=CS.leaf_layer(publics))


def _by_hand(toy, written, W: dict) -> dict:
    """The window's record R chained after ``W`` instead and registering ``W``'s tables, written by hand."""
    return {**written.R, "previous": REG.digest(W), "registered": [
        {**{e["value"]: e for e in W["registered"]}[_names()[p]], "value": p} for p in toy.low.circuit.registered.values]}


def _rounds(run) -> list[dict]:
    table = "cd" * 64
    return [{"request_table": table, "requests": CR.requests_root(bytes.fromhex(table), run.tables[TABLES.ids(0)].root).hex()},
            *({} for _ in range(ROUNDS - 1))]


def _words(c) -> dict[str, int]:
    """Each table the circuit registers, -> the words of the row its readers hold to it."""
    words = {r.param: dict(c.layouts[r.reader][0].outs)[r.port] for r in c.registered.reads if r.param in c.registered.values}
    return {p: words[p] for p in c.registered.values}


def _check(written, toy, W: dict, R: dict):
    """R1-R7 on ``R`` against the verifier's own window, with the log [W] and each table an output value reusing W's."""
    mine, dW, words = written.mine, REG.digest(W), _words(toy.low.circuit)
    return REG.check(R, mine.program, mine.query, LAW, CS.SCHEME, CS.expected_ports(mine), written.leaf_layer, (), [W],
                     registered=words, outputs={p: {"record": dW, "value": _names()[p]} for p in words}, window_root=OPEN)


@pytest.mark.slow
@RECORDS
def test_r_is_the_commit_record_credit_reads_and_the_registration_one_stage_accepts(toy, written):
    """R registers the four tables at W's entries and nothing else, chained right after W; its sampling design (R1 holds
    it to the verifier's own) draws a reader of every table, each table's readers templates of their own; ROOTS1 holds
    W's tables to it, H1 names it as its Commit record, credit's digest of it is one-stage's, and R1-R7 accept it."""
    run, R, W = toy.h1, written.R, toy.h1.W
    assert R["format"] == REG.FORMAT_V2 and R["previous"] == REG.digest(W)
    assert sorted(e["value"] for e in R["registered"]) == sorted(toy.low.circuit.registered.values)
    strata, readers = {s["template"]: s for s in R["law"]["strata"]}, {}
    for (tid, _port, _u), (table, _row) in _reads(toy.low.circuit).items():
        readers.setdefault(table, set()).add(tid)
    assert all(strata[t]["k"] >= 1 for ts in readers.values() for t in ts)
    assert len({t for ts in readers.values() for t in ts}) == sum(len(ts) for ts in readers.values())
    assert CR._arow(_rounds(run), W["registered"], R["registered"], _layout(), SV.PREFIX) is None
    assert CT.reused(run, W, R) is None
    named, details = CT.named(SimpleNamespace(**vars(run), commit_record=lambda _run: (W, R)))
    assert details == {"named": True, "registration": REG.digest(W), "record": REG.digest(R)}
    (digest, _entries), why = CR.commit_of(named)
    assert why is None and digest.hex() == REG.digest(R) == REG.digest(json.loads((written.dir / "registration.json").read_text()))
    assert list(_check(written, toy, W, R).codes) == []


@pytest.mark.slow
@RECORDS
def test_the_records_accept_a_flipped_table_row_so_only_its_readers_proof_refuses_it(toy, written):
    """W' and an R' written by hand (the same window's record, chained after W' and registering its tables, which
    window-commit's `reuse` refuses to make) pass ROOTS1, H1's reuse check and R1-R7 as W and R do: no record holds a row,
    so the flipped row is refused by its reader's proof alone. R' against W is refused by each, so the checks aren't
    vacuous."""
    flipped = _flipped(toy.h1)
    W2 = flipped.W
    R2 = _by_hand(toy, written, W2)
    assert canonical_json(R2) != canonical_json(written.R)
    assert CR._arow(_rounds(flipped), W2["registered"], R2["registered"], _layout(), SV.PREFIX) is None
    assert CT.reused(flipped, W2, R2) is None
    assert list(_check(written, toy, W2, R2).codes) == []
    assert CR._arow(_rounds(toy.h1), toy.h1.W["registered"], R2["registered"], _layout(), SV.PREFIX) is not None
    assert CT.reused(toy.h1, toy.h1.W, R2) is not None
    assert list(_check(written, toy, toy.h1.W, R2).codes) != []


def _one_stage() -> SimpleNamespace:
    """`served_cases`, `composite` and `replay` of ``benchmarks/one_stage``, which make a served window's cases and compose
    them (`replay` under the name its own tests load it by)."""
    if str(ONE_STAGE) not in sys.path:
        sys.path.insert(0, str(ONE_STAGE))
    import composite
    import served_cases
    replay = sys.modules.get("one_stage_replay")
    if replay is None:
        spec = importlib.util.spec_from_file_location("one_stage_replay", ONE_STAGE / "replay.py")
        replay = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = replay
        spec.loader.exec_module(replay)
    return SimpleNamespace(SC=served_cases, C=composite, RP=replay)


def _v1_case(one, toy, written, W: dict, R: dict) -> dict:
    """a5's v1 case of an audit of ``R`` (``window/case.json``: R, the evidence its record was handed and the verifier's
    params): `_check`'s params, the log [W] and each table an output value of W's; a stratified sample's first proof
    blocks of each stratum, the verifier's receipt of R, and a verdict of record that accepted, reporting R's tables'
    roots as the ones its readers opened against, standing in for their proofs."""
    mine, words = written.mine, _words(toy.low.circuit)
    program, query = mine.program, mine.query
    drawn = D.derive(program, query, LAW)
    units = sorted(u for s in drawn["strata"] for u in [u for lo, hi in s["units"] for u in range(lo, hi)][:s["k"]])
    draw = {**drawn, "population": P.units(program, query), "units": units}
    proved = D.proved(draw)
    opened = {e["value"]: {"root": e["root"], "value": e["value"]} for e in R["registered"]}
    evidence = {"draw": draw, "instances": len(proved), "proved_units": proved,
                "receipt": REG.receipt(R, "v", received="2000-01-01T00:00:00Z"), "served": None, "session_roots": None,
                "sessions": [], "statement": {},
                "verdicts": {"lean": {"accepted": True, "of_record": True, "registered": opened}}}
    params = one.RP.params(program, query, LAW, CS.expected_ports(mine), written.leaf_layer, "1/20", log=[W],
                           registered=words, scheme=CS.SCHEME)
    params |= {"outputs": {p: {"record": REG.digest(W), "value": _names()[p]} for p in words}, "window_root": OPEN,
               "parameters": [[n, t.to_json()] for n, t in program.fn.signature()[0]]}
    return {"commitment": canonical_json(R).hex(), "evidence": canonical_json(evidence).hex(),
            "params": json.loads(json.dumps(params))}


def _period(one, exe: Path, d: Path, commits: list[dict]) -> None:
    """PoUW's period case in ``d`` beside its Lean answer (``exe``), as `pouw_piece.py period` writes them: the auditor's
    ledger empty and a window a record of ``commits`` names as its Commit record each, carrying none of an H1 window's
    proofs."""
    evidence = {"pins": [], "windows": [{"commit": canonical_json(r).hex()} for r in commits]}
    case = {"blobs": [], "commitment": canonical_json([]).hex(), "evidence": canonical_json(evidence).hex(),
            "params": {"key": None, "main": [], "name": "toy"}}
    _answer, line = one.SC.lean_answer(exe, case, d.parent / "scratch")
    one.SC.write(d, case, line)


def _serve(one, root: Path, v1: dict, W: dict, R: dict, period_exe: Path) -> Path:
    """A serve dir under H1's tie as `served_cases.py --serve-dir` reads it: V3c's audit's v1 case, H1's log of W then R
    (`serve_tie`'s, each record in its window's file), and PoUW's period case of one window naming R."""
    audit = root / f"v3c-w0-{CLASS.slots}-{CLASS.prompt}-{CLASS.outputs}" / "audit" / "window"
    audit.mkdir(parents=True)
    (audit / "case.json").write_text(json.dumps(v1, sort_keys=True))
    (root / "h1" / "w0").mkdir(parents=True)
    lines = []
    for kind, name, r in (("registration", TIE.W_FILE, W), ("record", TIE.R_FILE, R)):
        (root / "h1" / "w0" / name).write_text(json.dumps(r, indent=1) + "\n")
        lines.append({"window": 0, "kind": kind, "file": f"w0/{name}", "digest": REG.digest(r)})
    (root / "h1" / TIE.LOG).write_text("".join(json.dumps(x) + "\n" for x in lines))
    _period(one, period_exe, root / one.SC.PERIOD_PIECE, [R])
    return root


@pytest.fixture(scope="module")
def lean(toy, written, tmp_path_factory) -> SimpleNamespace:
    """The Lean verifiers, and the toy's cases as `served_cases.py --serve-dir` writes them from its serve dir under H1's
    tie (α and Γ read off the division): its exit and its CASE and NOT_WRITTEN lines."""
    one = _one_stage()
    SC = one.SC
    exes = SimpleNamespace(v1=SC.build(*SC.V1_LEAN), policy=SC.build(*SC.INFERENCE_ONLY_LEAN),
                           period=SC.build(ROOT / "verity/protocols/pouw/verifier/lean/check", "pouw-period"),
                           remote=SC.build(ROOT / "verity/protocols/remote/verifier", "remote"))
    W, R = toy.h1.W, written.R
    serve = _serve(one, tmp_path_factory.mktemp("serve"), _v1_case(one, toy, written, W, R), W, R, exes.period)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = SC.main(["--serve-dir", str(serve), "--private", str(serve / "private"), "--from-division",
                      "--v1-exe", str(exes.v1), "--inference-only-exe", str(exes.policy)])
    lines = [(head, json.loads(body)) for head, _, body in (x.partition(" ") for x in out.getvalue().splitlines())
             if head in ("CASE", "NOT_WRITTEN")]
    return SimpleNamespace(one=one, exes=exes, private=serve / "private", rc=rc, lines=lines)


def _answer(lean, id: str) -> dict:
    return json.loads((lean.private / lean.one.SC.DIRS[id] / "answer.json").read_text())


def _remote(lean, out: Path, period: Path) -> tuple[int, dict, list[str]]:
    """`composite.py remote` on the cases the serve dir has of remote's subprotocols, the policy's and ``period`` (its
    sampled proofs case is not written): its exit, remote's Lean answer and the subprotocols the composite names missing."""
    SC = lean.one.SC
    cases = [lean.private / SC.DIRS[SC.INFERENCE_ONLY], period]
    with contextlib.redirect_stdout(io.StringIO()):
        rc = lean.one.C.main(["remote", *(x for c in cases for x in ("--cases", str(c))), "--private-out",
                              str(out / "private"), "--public-out", str(out / "public"), "--exe", str(lean.exes.remote)])
    return (rc, json.loads((out / "private" / "answer.json").read_text()),
            json.loads((out / "public" / "composite.json").read_text())["missing"])


@pytest.mark.slow
@RECORDS
def test_one_stage_and_the_policy_accept_r_on_the_log_w_and_the_period_names_r(toy, written, lean, tmp_path):
    """`served_cases` writes three cases from the serve dir and refuses a fourth, so it exits 1. One-stage's Lean accepts R
    on the log [W] (its log record reader takes W, and its R1-R7 accept R chained right after W with each table an output
    value of W's), and the policy's Lean accepts its case, whose log is H1's through W; both name R's digest. The period
    names R as its window's Commit record (`served_cases.period_case`), and its Lean refuses it for want of an H1 window's
    proofs. The fourth is remote's sampled proofs case, one-stage v2's: the toy's audit ran no proof, so it holds none.
    Remote, on the policy's case and the period's, rejects naming sampled proofs missing: it reads no subprotocol's case
    while one has none. So its answer here is that refusal and not the period's, and this test does not check that remote
    carries a subprotocol's refusal or holds its own tie."""
    SC, d = lean.one.SC, REG.digest(written.R)
    assert (lean.rc, [(head, c["protocol"], c.get("verdict")) for head, c in lean.lines]) == (1, [
        ("CASE", SC.V1, "accept"), ("NOT_WRITTEN", SC.SAMPLED, None), ("CASE", SC.INFERENCE_ONLY, "accept"),
        ("CASE", SC.PERIOD, "reject")]), lean.lines
    assert lean.lines[1][1]["reason"] == NO_V2_CASE, lean.lines[1]
    for id in (SC.V1, SC.INFERENCE_ONLY):
        got = _answer(lean, id)
        assert (got["protocol"], got["commitment"], got["verdict"]) == (id, d, "accept"), got
    policy = json.loads((lean.private / SC.DIRS[SC.INFERENCE_ONLY] / "case.json").read_text())
    assert policy["params"]["log"] == [toy.h1.W] and policy["commitment"] == canonical_json(written.R).hex()
    assert SC.period_case(lean.private / SC.DIRS[SC.PERIOD], d)[2] == "reject"
    rc, got, missing = _remote(lean, tmp_path, lean.private / SC.DIRS[SC.PERIOD])
    assert (rc, got["verdict"], got["reason"], missing) == (1, "reject", f"params: {SC.SAMPLED} is missing",
                                                            [SC.SAMPLED]), got


@pytest.mark.slow
@RECORDS
def test_a_period_naming_w_is_refused_before_any_composite(toy, written, lean, tmp_path):
    """A period whose window names H1's W as its Commit record is refused by `served_cases` (remote's record tie) before any
    composite. Handed to remote anyway, beside the policy's case on R, it is rejected as the period naming R is: remote
    names sampled proofs missing, whose case the toy's audit doesn't write, and reads no subprotocol's case while one has
    none. So its answer here is that refusal and not the period's, nothing of remote's tells the two periods apart on
    this toy, and `served_cases`' refusal is the only check here that a period naming W goes no further."""
    SC = lean.one.SC
    period = tmp_path / "pouw-period"
    _period(lean.one, lean.exes.period, period, [toy.h1.W])
    with pytest.raises(SC.Refused, match="remote's record tie rejects it"):
        SC.period_case(period, REG.digest(written.R))
    rc, got, missing = _remote(lean, tmp_path / "remote", period)
    assert (rc, got["verdict"], got["reason"], missing) == (1, "reject", f"params: {SC.SAMPLED} is missing",
                                                            [SC.SAMPLED]), got


@pytest.mark.slow
@RECORDS
def test_one_stage_s_lean_accepts_a_flipped_w_s_record_only_on_its_own_log(toy, written, lean, tmp_path):
    """The R' written by hand after W' (the flipped row's) is accepted by one-stage's Lean on the log [W'], as R1-R7 accept
    it: no record holds a row. On the log [W] its registration is refused."""
    SC, W2 = lean.one.SC, _flipped(toy.h1).W
    R2 = _by_hand(toy, written, W2)
    got, _ = SC.lean_answer(lean.exes.v1, _v1_case(lean.one, toy, written, W2, R2), tmp_path)
    assert (got["verdict"], got["commitment"]) == ("accept", REG.digest(R2)), got
    got, _ = SC.lean_answer(lean.exes.v1, _v1_case(lean.one, toy, written, toy.h1.W, R2), tmp_path)
    assert got["verdict"] == "reject" and got["reason"].startswith("registration: "), got
