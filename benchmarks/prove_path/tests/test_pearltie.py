"""Pearl+'s tie end to end on a toy: Pearl+'s window record W over a window's passes and the window circuit, cut at its
linears, that audits them, registered as the remote composite reads them (the audited window's registration R chained
right after W, each of Pearl+'s values an output value of W's).

**The toy** (`toy`): a one-layer bf16 Llama of hidden 128 (two heads of 64, one KV head) and vocabulary 512 on the H200, at
class (1 slot, prompts of 27, 2 outputs), cut at its linears (`window_class.PearlTables`, each y handed on in rows of 64
words). So the window has 2 rounds, round 0 the prompt's 27 rows and round 1 one decode row, of 4 calls each (q/k/v, o,
gate/up and down), and its 895 units of 58 templates are proof blocks C-Flock's composer takes.

* **Pearl+'s passes**, a pass a round: each call's A rows (its x widened to FP32, then zero rows up to m₁₆) and y (its value
  widened) as the circuit's proof blocks make them (`window_rows.Rows.value`), committed as Pearl+'s reference commits them
  (`audit.commit_rows`, `served_y.commit_y`) under the prover's salt key, and each layer weight's B rows (a call's
  weight, random BF16 rows of n x K: no proof block reads B) as PoUW's weights tree commits them, in manifests of the
  fields W and the verdict's W check read.
* **W**, Pearl+'s window record over them (`pearl_c.window.record`), which Pearl+'s verdict holds to the roots its passes
  credited (`verify_run.tie`).
* **R**, the audited window's registration, as window-commit makes it: the cut circuit's units filled from the same
  values, its values W's rebuilt from the rows its proof blocks make (`pearl_c.window.reuse`), recorded, committed and
  written through `interface.Window` as a v2 record chained right after W.

**Honest** (slow): the circuit reads each call's real rows once and its A rows' filler rows never, the verdict's W check
accepts W, and `reuse` makes each value W's entry.
**A root other than the credited one** (slow): W', the record of passes that committed round 1's o-projection A row with
one bit flipped, differs from W at that value's root alone. The verdict's W check refuses it on the honest passes and
`reuse` refuses it. On the flipped passes the verdict accepts W' (it is their record), and `reuse` still refuses it: the
proof blocks don't make that row. Its reader is the one proof block of a template of its own, which `stratified:1` always
draws, and its Definition on its input gives the honest row, so a prover who records W''s row instead has a proof block
whose proof fails.

**Records** (opt-in, `VERITY_PROVE_PATH_REMOTETIE`: the window's record, commit and write take a minute): R registers the
circuit's values at W's entries and nothing else, chained right after W, and its sampling design draws a reader of every
value, each value's readers templates of their own. `records_audit.tie_of` with no prefix ties the audit to W (each value
by its own name, in ``row-seg/v1`` rows of 32-bit words), and R1-R7 (`registration.check`) accept R on the log [W] and
refuse it on [W'] (R6, and R7 at the flipped value's root). An R' written by hand after W' is accepted on [W']: no record
holds a row, so the flipped row is refused by its reader's proof alone.

**Lean** (opt-in as the records, on a machine with Lean's toolchain): one-stage's Lean verifier (`served_cases.V1_LEAN`) on
a5's v1 case of R with the tie's params accepts R on [W], refuses its registration on [W'], and accepts R' on [W']. The
case's verdict of record is an accepting stand-in reporting R's values' roots as their readers opened them; the proofs of
a sample of R's proof blocks, through both one-stage verifiers, are `records_audit.py audit --tie W`'s.

Pearl+'s verdict is the prover's development verifier: it reads the prover's salt key and opens drawn rows in the clear,
values the one-stage proof hides once tied.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import settings
from verity_vllm.program.predict import scan_state as SS
from verity_vllm.program.predict import window_circuit as WC
from verity_vllm.program.predict import window_rows as WRS
from verity_vllm.program.predict import window_weights as WW
from verity_vllm.program.predict.phi import Phi
from verity_vllm.program.predict.steps import Registered
from verity_vllm.program.predict.window_class import PearlTables, WindowClass
from verity_vllm.protocols.pouw.pearl_c import served_y as SY
from verity_vllm.protocols.pouw.pearl_c import window as PWR

from verity.protocols.pouw.verifier.audit import HM96, HM96_WEIGHTS, Workload, activation_binding, commit_rows, weight_binding
from verity.protocols.pouw.verifier.protocol import Matmul, Shape
from verity.protocols.spot.verifier.one_stage import draw as D, partition as P, registered as G, registration as REG
from verity.circuits.codec import canonical_json
from verity.circuits.evaluation import evaluate_batch
from verity.commitments.windows import interface as IF, rows as CS

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import records_audit as RA  # noqa: E402

ROOT = HERE.parents[2]
ONE_STAGE = ROOT / "benchmarks" / "one_stage"
H200 = {"compute_capability": [9, 0], "num_sms": 132}
CONFIG = {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_size": 128, "intermediate_size": 256,
          "num_attention_heads": 2, "num_key_value_heads": 1, "num_hidden_layers": 1, "vocab_size": 512,
          "rms_norm_eps": 1e-5, "tie_word_embeddings": True, "max_position_embeddings": 64, "hidden_act": "silu"}
SCHEME = "pearl-c-h100-v1-h3"
CLASS, GRAIN, PEARL = WindowClass(1, 27, 2), WC.Grain(gemm_words=8192, rows=256, select=512), PearlTables(SCHEME, words=64)
#: a round's calls: the layer's q/k/v, o, gate/up and down
CALLS = 4
#: the committers' secret keys: the audited window's, and Pearl+'s salt key, which the prover's manifests retain
KEY, SALT_KEY = bytes(range(32)), bytes(range(64, 96))
#: the window's epoch salt, Pearl+'s salt coin's
SALT = bytes(range(100, 164))
#: the audited window's run root
RUN = "ab" * 64
LAW = D.parse_design("stratified:1")
PROMPT = [(7 * i + 3) % 500 for i in range(CLASS.prompt)]
#: the flipped bit: round 1's o-projection's (call 1) A row 0, its first word's high half (the BF16 of x's first element)
FLIP = (PEARL.a(1, 1), 0, 1)
RECORDS = pytest.mark.skipif(not settings.REMOTETIE, reason="VERITY_PROVE_PATH_REMOTETIE=1 records, commits and writes the "
                             "toy's 895 proof blocks and builds the Lean verifier that checks them, a few minutes")


def _bf16(x) -> np.ndarray:
    return (np.asarray(x, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _weights(fn) -> dict[str, np.ndarray]:
    """Finite random bf16 weights, each field's elements in their unsigned width: norms near 1, the rotary cache in
    [-1, 1], an attention's f32 scales 1."""
    rng, out = np.random.default_rng(3), {}
    for label, ty in dict(fn.signature()[0])[WC.WEIGHTS].fields:
        (bits,), n = ty.width_set(), ty.leaves
        if bits == 32:
            out[label] = (np.ones(1) if n == 1 else rng.uniform(5e-3, 2e-2, n)).astype(np.float32).view(np.uint32)
        elif "cos_sin" in label:
            out[label] = _bf16(rng.uniform(-1, 1, n))
        elif "norm" in label:
            out[label] = _bf16(1 + 0.1 * rng.standard_normal(n))
        else:
            out[label] = _bf16(0.5 * rng.standard_normal(n))
    return out


def _shape(c, s: int, call: int) -> tuple[int, int]:
    """Round ``s``'s call ``call``'s K and N, from the words of the values the circuit's program takes."""
    params = c.registered.params
    return (params[PEARL.a(s, call)] // (2 * PEARL.a_rows(CLASS, s)), params[PEARL.y(s, call)] // (2 * PEARL.rows(CLASS, s)))


def _b_roots(c) -> dict[int, str]:
    """Each call's weight's B root (weight id the call's place in its round), as PoUW's weights tree commits its n rows of
    K FP32 words (`audit.commit_rows` under `weight_binding`, salted at ``HM96_WEIGHTS + w``) under the prover's salt key."""
    rng, out = np.random.default_rng(5), {}
    for w in range(CALLS):
        k, n = _shape(c, 0, w)
        rows = (rng.standard_normal((n, k)).astype(np.float32).view(np.uint32) & 0xFFFF0000).astype("<u4")
        out[w] = commit_rows(weight_binding(w), -2, [r.tobytes() for r in rows], HM96, SALT_KEY,
                             HM96_WEIGHTS + w).commitment.root.hex()
    return out


def _manifests(c, pearl: dict[str, np.ndarray]) -> list[dict]:
    """Pearl+'s passes' manifests, a pass a round, of the fields W and the verdict's W check read: each call's A rows
    ``pearl[pearl-a/<s>/<c>]`` committed as `audit.commit_rows` commits them (salted at the call's index) and its y
    ``pearl[pearl-y/<s>/<c>]`` as `served_y.commit_y` does (its BF16 words, each committed row's high halves), under the
    epoch's salt and the prover's salt key, the call's index ``s · CALLS + c``; and each weight's B root (`_b_roots`)."""
    out, roots = [], _b_roots(c)
    for s in range(CLASS.outputs):
        mm = []
        for call in range(CALLS):
            i, (k, n) = s * CALLS + call, _shape(c, s, call)
            a, y = pearl[PEARL.a(s, call)], pearl[PEARL.y(s, call)]
            root_a = commit_rows(activation_binding(SALT, i), i, [r.tobytes() for r in a], HM96, SALT_KEY, i).commitment.root
            root_y = SY.commit_y(np.ascontiguousarray(y[:, 1::2]).tobytes(), n, SALT, i, SALT_KEY)[0]
            mm.append({"index": i, "m": len(a), "k": k, "n": n, "weight": call, "real_rows": len(y), "root_a": root_a.hex(),
                       "y": {"root": root_y.hex()}})
        out.append({"entry": f"pouw:{SCHEME}", "scheme": SCHEME, "salt": SALT.hex(), "hm96_salt_key": SALT_KEY.hex(),
                    "matmuls": mm, "weights": {str(u["weight"]): {"n": u["n"], "k": u["k"], "root": roots[u["weight"]]}
                                               for u in mm}})
    return out


def _window(c, pearl: dict[str, np.ndarray]) -> SimpleNamespace:
    """Pearl+'s passes committing ``pearl`` and its window record W over them."""
    manifests = _manifests(c, pearl)
    return SimpleNamespace(pearl=pearl, manifests=manifests, W=PWR.record(manifests))


def _flipped(toy) -> SimpleNamespace:
    """``toy``'s passes with bit `FLIP` of their A rows flipped, committed alike, and their W'."""
    pearl = {v: r.copy() for v, r in toy.pearl.items()}
    pearl[FLIP[0]][FLIP[1], FLIP[2]] ^= 1
    return _window(toy.low.circuit, pearl)


def _passes(manifests: list[dict]) -> list[SimpleNamespace]:
    """The passes of ``manifests`` as Pearl+'s verdict holds them, the fields its W check reads: the scheme's hashing,
    each call's matmul, the manifest and the epoch's salt, workload and weight roots (the toy has no tiles for the verdict
    to replay)."""
    out = []
    for man in manifests:
        mm = [Matmul(u["index"], Shape(u["m"], u["k"], u["n"]), u["weight"]) for u in man["matmuls"]]
        epoch = SimpleNamespace(salt=SALT, workload=Workload(mm),
                                weight_roots={int(w): bytes.fromhex(x["root"]) for w, x in man["weights"].items()})
        out.append(SimpleNamespace(scheme=SimpleNamespace(commitment_hash=HM96), man=man, epoch=epoch,
                                   records=[SimpleNamespace(matmul=u) for u in mm]))
    return out


def _calls(W: dict) -> list[dict]:
    """W's entries of the calls' values, the ones the audited circuit reads (it reads no B row)."""
    return [e for e in W["registered"] if not e["value"].startswith("pearl-b/")]


def _verify_run():
    """Pearl+'s verdict, ``benchmarks/pouw/pearl_c_vllm/verify_run.py``, under the name its own tests load it by."""
    name = "pearl_c_vllm_verify_run"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / "benchmarks" / "pouw" / "pearl_c_vllm" / "verify_run.py")
        sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules[name])
    return sys.modules[name]


@pytest.fixture(scope="module")
def toy() -> SimpleNamespace:
    """The toy's cut window circuit, its honest rows from one served slot (a prompt of 27, 2 outputs), each of Pearl+'s
    values as its proof blocks make it, and Pearl+'s passes committing those values with their W."""
    reg = Registered.of(Phi(model={"repo": "org/toy", "revision": "r", "hf_config": CONFIG}, target=H200), default_seed=7)
    low = WC.lower(reg, CLASS, None, GRAIN, tables=PEARL)
    fields = _weights(low.fn)
    state = WRS.standin_state(WC.NAME, CLASS, [([1, CLASS.prompt, *PROMPT], CLASS.outputs)], KEY, 0, window=RUN)
    values = {**WW.weight_values(reg, fields, KEY, GRAIN, keep_rows=True), SS.SCAN_STATE: state}
    slots = [WRS.Slot(low.fn, CLASS, row, lambda label, lo, hi: fields[label][lo:hi], cut=low.cut)
             for row, _ in WRS.scan_slots(low, state.rows)]
    rows, c = WRS.Rows(low, slots, values), low.circuit
    pearl = {}
    for s in range(CLASS.outputs):
        for call in range(CALLS):
            k, n = _shape(c, s, call)
            pearl[PEARL.a(s, call)] = rows.value(PEARL.a(s, call), PEARL.a_rows(CLASS, s), np.zeros(2 * k, np.uint16))
            pearl[PEARL.y(s, call)] = rows.value(PEARL.y(s, call), PEARL.rows(CLASS, s), np.zeros(2 * n, np.uint16))
    return SimpleNamespace(low=low, values=values, slots=slots, rows=rows, **vars(_window(c, pearl)))


def _reuse(toy, W: dict, manifests: list[dict] | None = None) -> dict:
    return PWR.reuse(W, toy.manifests if manifests is None else manifests, toy.rows.value)


def _reads(circuit) -> dict[tuple[str, str, int], tuple[str, int]]:
    """Each row a read of a value holds or fills, ``(template, port, proof block)``, as its value and row."""
    out = {}
    for r in (r for r in circuit.registered.reads if r.param in circuit.registered.values):
        lay = circuit.layouts[r.reader][0]
        words = dict(lay.rows + lay.outs)[r.port]
        for i, leaf in enumerate(r.source(range(r.units))):
            out[(r.reader, r.port, r.at + i)] = (r.param, int(leaf) // words)
    return out


@pytest.mark.slow
def test_the_circuit_reads_each_calls_committed_rows_and_the_verdict_and_reuse_take_w(toy):
    """The circuit takes each call's A rows and y, round by round and call by call, as 32-bit ``row-seg/v1`` values; it reads
    each real row of each once (an A row as a `LinearARow_v1`'s output, a y row as a `LinearOut_v1`'s input) and the A
    rows' filler rows never. The verdict's W check accepts W on the passes, on all 16 values and the 4 weights' B rows,
    and `reuse` makes each value, from the rows its proof blocks make, W's entry."""
    c = toy.low.circuit
    assert c.registered.values == tuple(v for s in range(CLASS.outputs) for k in range(CALLS) for v in (PEARL.a(s, k),
                                                                                                       PEARL.y(s, k)))
    assert c.registered.formats == {v: {"row_schema": IF.SEG, "word_bits": 32} for v in c.registered.values}
    assert {v: r.shape for v, r in toy.pearl.items() if v.endswith("/1")} == {
        "pearl-a/0/1": (32, 256), "pearl-y/0/1": (27, 256), "pearl-a/1/1": (16, 256), "pearl-y/1/1": (1, 256)}
    reads = _reads(c)
    assert {port for _t, port, _u in reads} == {"o0", "y"}
    real = {(v, i) for v in c.registered.values for i in range(PEARL.rows(CLASS, int(v.split("/")[1])))}
    assert sorted(reads.values()) == sorted(real)
    VR = _verify_run()
    assert VR.tie(toy.W, _passes(toy.manifests)) == {"record": REG.digest(toy.W), "tables": (2 * CLASS.outputs + 1) * CALLS}
    assert [e["value"] for e in toy.W["registered"]][-CALLS:] == [PEARL.b(w) for w in range(CALLS)]
    assert {v: x.entry() for v, x in _reuse(toy, toy.W).items()} == {e["value"]: e for e in _calls(toy.W)}


@pytest.mark.slow
def test_a_w_whose_root_isnt_the_credited_one_is_refused_and_its_row_is_off_its_readers_definition(toy):
    """W' differs from W at the flipped value's root alone. The verdict's W check refuses it on the honest passes (its root
    not the call's) and accepts it on the flipped ones, whose record it is; `reuse` refuses it on either, since the proof
    blocks make the honest row. That row's reader is the one proof block of its template (`LinearARow_v1` at round 1 and
    the o-projection's K), a stratum of its own whose sample under `stratified:1` is that proof block (`draw.stratum_k`),
    and its Definition on its input gives the honest row, not W''s."""
    VR, flipped, c = _verify_run(), _flipped(toy), toy.low.circuit
    assert [x["value"] for x, y in zip(toy.W["registered"], flipped.W["registered"], strict=True) if x != y] == [FLIP[0]]
    with pytest.raises(ValueError, match=f"W's {FLIP[0]}: its root not the call's"):
        VR.tie(flipped.W, _passes(toy.manifests))
    assert VR.tie(flipped.W, _passes(flipped.manifests))["record"] == REG.digest(flipped.W)
    with pytest.raises(ValueError, match="W is not the window record of these passes"):
        _reuse(toy, flipped.W)
    with pytest.raises(ValueError, match=f"^{FLIP[0]}: the rows the audited circuit's proof blocks make of it are not"):
        _reuse(toy, flipped.W, flipped.manifests)
    (tid, port, u), = [k for k, v in _reads(c).items() if v == FLIP[:2]]
    assert c.units[tid] == 1 and "ROUND=1" in tid and "CALL=1" in tid and port == "o0" and u == 0
    assert D.stratum_k(LAW["k"], c.units[tid], sum(c.units.values())) == 1
    j = next(j for j, r in toy.low.programs[0].arows.items() if (r.tid, r.unit) == (tid, u))
    x = toy.slots[0].operand(toy.low.fn.body.nodes[j].args[0])
    got = np.asarray(evaluate_batch(toy.rows.fns[tid], x[None])).astype("<u4").view("<u2").reshape(-1)
    assert np.array_equal(got, toy.pearl[FLIP[0]][FLIP[1]]) and not np.array_equal(got, flipped.pearl[FLIP[0]][FLIP[1]])


@pytest.fixture(scope="module")
def written(toy, tmp_path_factory) -> SimpleNamespace:
    """The toy's window recorded, committed and written as window-commit writes it after W (R in its
    ``registration.json``), and the verifier's own window and circuit record from its records."""
    c, out = toy.low.circuit, tmp_path_factory.mktemp("window")
    win = IF.Window(c, RUN, LAW, 64, registered={**toy.values, **_reuse(toy, toy.W)}, forms=WRS.FORMS)
    win.supply(G)
    toy.rows.record(win)
    committed = win.close()
    prov = {"window_circuit": {"class": [CLASS.slots, CLASS.prompt, CLASS.outputs, CLASS.lanes, CLASS.every], "window": 0},
            "activations": {"program": WC.NAME, "values": {SS.SCAN_STATE: toy.values[SS.SCAN_STATE].entry()}}}
    win.write(committed, out, prov, registrar=REG, index_salt=WRS.index_salt(KEY, 0), previous=REG.digest(toy.W))
    mine, doc = IF.of_records(out, LAW, RA.catalog_resolve)
    publics = {m.key: (out / f"m{k}-public.bin").read_bytes() for k, m in enumerate(mine.members)}
    return SimpleNamespace(dir=out, R=json.loads((out / "registration.json").read_text()), mine=mine, doc=doc,
                           leaf_layer=CS.leaf_layer(publics))


def _by_hand(written, W: dict) -> dict:
    """The window's record R chained after ``W`` instead and registering ``W``'s entries, written by hand."""
    theirs = {e["value"]: e for e in W["registered"]}
    return {**written.R, "previous": REG.digest(W), "registered": [theirs[e["value"]] for e in written.R["registered"]]}


def _check(written, W: dict, R: dict):
    """R1-R7 on ``R`` against the verifier's own window, tied to ``W`` as records_audit ties it (`tie_of`, no prefix)."""
    tied, _pins = RA.tie_of(written.doc, W, None)
    mine = written.mine
    return REG.check(R, mine.program, mine.query, LAW, CS.SCHEME, CS.expected_ports(mine), written.leaf_layer, (),
                     tied["log"], registered=tied["registered"], outputs=tied["outputs"], schemas=tied["schemas"],
                     window_root=RUN)


@pytest.mark.slow
@RECORDS
def test_r_registers_ws_values_and_r1_r7_accept_it_on_w_and_refuse_it_on_a_w_of_another_root(toy, written):
    """R registers the 16 values at W's entries and nothing else, chained right after W; its sampling design (R1 holds it to
    the verifier's own) draws a reader of every value, each value's readers templates of their own. R1-R7 tied to W accept
    it, and tied to W' refuse it: it is chained after W (``R6-previous``), and of its values only the flipped one's root is
    not the one W' registers (``R7-held``)."""
    R, W = written.R, toy.W
    assert R["format"] == REG.FORMAT_V2 and R["previous"] == REG.digest(W)
    assert sorted(R["registered"], key=lambda e: e["value"]) == sorted(_calls(W), key=lambda e: e["value"])
    strata, readers = {s["template"]: s for s in R["law"]["strata"]}, {}
    for (tid, _port, _u), (v, _row) in _reads(toy.low.circuit).items():
        readers.setdefault(v, set()).add(tid)
    assert all(strata[t]["k"] >= 1 for ts in readers.values() for t in ts)
    assert len({t for ts in readers.values() for t in ts}) == sum(len(ts) for ts in readers.values()) == len(_calls(W))
    assert list(_check(written, W, R).codes) == []
    got = _check(written, _flipped(toy).W, R)
    assert list(got.codes) == ["R6-previous", "R7-held"] and got.detail["R7-held"]["value"] == FLIP[0], got


@pytest.mark.slow
@RECORDS
def test_the_records_accept_a_flipped_row_on_its_own_w_so_only_its_readers_proof_refuses_it(toy, written):
    """An R' written by hand after W' (registering its entries, which window-commit's `reuse` refuses to make) passes R1-R7
    tied to W', and the verdict's W check accepts W' on the passes that committed the flipped row: no record holds a row,
    so the flipped row is refused by its reader's proof alone. R' tied to W is refused, so the check isn't vacuous."""
    flipped = _flipped(toy)
    R2 = _by_hand(written, flipped.W)
    assert canonical_json(R2) != canonical_json(written.R)
    assert _verify_run().tie(flipped.W, _passes(flipped.manifests))["record"] == REG.digest(flipped.W)
    assert list(_check(written, flipped.W, R2).codes) == []
    assert list(_check(written, toy.W, R2).codes) != []


def _one_stage() -> SimpleNamespace:
    """`served_cases` and `replay` of ``benchmarks/one_stage`` (`replay` under the name its own tests load it by)."""
    if str(ONE_STAGE) not in sys.path:
        sys.path.insert(0, str(ONE_STAGE))
    import served_cases
    replay = sys.modules.get("one_stage_replay")
    if replay is None:
        spec = importlib.util.spec_from_file_location("one_stage_replay", ONE_STAGE / "replay.py")
        replay = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = replay
        spec.loader.exec_module(replay)
    return SimpleNamespace(SC=served_cases, RP=replay)


def _v1_case(one, written, W: dict, R: dict) -> dict:
    """a5's v1 case of an audit of ``R`` tied to ``W`` (``window/case.json``: R, the evidence its record was handed and the
    verifier's params): `_check`'s params; a stratified sample's first proof blocks of each stratum, the verifier's
    receipt of R, and a verdict of record that accepted, reporting R's values' roots as the ones its readers opened
    against, standing in for their proofs."""
    mine, (tied, _pins) = written.mine, RA.tie_of(written.doc, W, None)
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
    params = one.RP.params(program, query, LAW, CS.expected_ports(mine), written.leaf_layer, "1/20", log=tied["log"],
                           registered=tied["registered"], scheme=CS.SCHEME, outputs=tied["outputs"], schemas=tied["schemas"])
    params |= {"window_root": RUN}
    return {"commitment": canonical_json(R).hex(), "evidence": canonical_json(evidence).hex(),
            "params": json.loads(json.dumps(params))}


@pytest.fixture(scope="module")
def lean() -> SimpleNamespace:
    """One-stage's Lean verifier, built."""
    one = _one_stage()
    return SimpleNamespace(one=one, v1=one.SC.build(*one.SC.V1_LEAN))


@pytest.mark.slow
@RECORDS
def test_one_stage_s_lean_accepts_r_on_w_and_refuses_it_on_a_w_of_another_root(toy, written, lean, tmp_path):
    """One-stage's Lean verifier accepts R tied to W (its log record reader takes Pearl+'s W, and its R1-R7 take each value
    as W's 32-bit ``row-seg/v1`` entry) and names R's digest; tied to W' it refuses R's registration; and it accepts R'
    tied to W', as R1-R7 do."""
    SC, flipped = lean.one.SC, _flipped(toy)
    got, _ = SC.lean_answer(lean.v1, _v1_case(lean.one, written, toy.W, written.R), tmp_path)
    assert (got["verdict"], got["commitment"]) == ("accept", REG.digest(written.R)), got
    got, _ = SC.lean_answer(lean.v1, _v1_case(lean.one, written, flipped.W, written.R), tmp_path)
    assert got["verdict"] == "reject" and got["reason"].startswith("registration: "), got
    R2 = _by_hand(written, flipped.W)
    got, _ = SC.lean_answer(lean.v1, _v1_case(lean.one, written, flipped.W, R2), tmp_path)
    assert (got["verdict"], got["commitment"]) == ("accept", REG.digest(R2)), got
