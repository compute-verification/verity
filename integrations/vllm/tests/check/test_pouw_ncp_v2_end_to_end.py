"""`pouw:ncp-v2` beside sampled proofs, end to end on a small model (CPU).

A two-request Program, lowered at its PoUW site by the Build's `pouw_circuit.lower`:
Embedding → RMSNorm → up_proj (the PoUW linear, 64 → 16) → logits → TokenSelect, over a 16-token prefill and one decode
step. Each step's up_proj batch is one call of the adapter's real `ncp-v2` executor, installed through the composition.
The call is bound to its `NcpLinear_v1` Program and committed. The model's module outputs are committed as the collector
packs them.

The verifier then runs the real sampled replay, with the PoUW rows answered by their calls (`pouw_circuit.linked`), and
the circuit replay of every call (`pouw_circuit.replay`: closure draws under `CountLaw`, each proved unit re-evaluated).

- The honest run is accepted.
- A wrong X strip, a wrong tile word and an A strip that is not the quantized input are each rejected, each committed by
  a prover that recomputes its root.
- Without the delegate the PoUW rows are not evaluable, so the run fails closed.
- Every other PoUW scheme stays refused beside sampled proofs.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from verity_pouw.circuit import plan as CPL

from verity_vllm import protocol_options as PO
from verity_vllm.check.replay import driver as SR
from verity_vllm.check.replay import evaluate as EV
from verity_vllm.check.replay import index as IX
from verity_vllm.check.replay import pouw_circuit as RC
from verity_vllm.protocol_options import pouw
from verity_vllm.protocol_options import pouw_circuit as PC
from verity_vllm.query.required import Q_MODULE_BODY_ID

EPS = '{"f64":"0x1.4f8b588e368f1p-17"}'
V, H, NU, LP, T = 8, 64, 16, 16, 1          # vocab, hidden (the PoUW linear's k), its n, prompt length, decode steps
SITE = "model.layers.0.mlp.up_proj"         # the collector's path of the PoUW linear
BEACON = "5a" * 32


@pytest.fixture(scope="module", autouse=True)
def _module_body_query():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(IX, "DEFAULT_QUERY", Q_MODULE_BODY_ID)
        yield


def _bf16(rng, n, scale=1.0):
    return (np.asarray(rng.standard_normal(n) * scale, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _program():
    """Stage-major rows per step (every token's norm before the step's PoUW rows), so a step's PoUW rows are one batch."""
    fields = [("model.model.embed_tokens.weight", V * H, f"Array<{V},Array<{H},Value<16>>>"),
              ("model.model.layers.0.input_layernorm.weight", H, f"Array<{H},Value<16>>"),
              ("model.model.layers.0.mlp.up_proj.weight", NU * H, f"Array<{NU},Array<{H},Value<16>>>"),
              ("model.lm_head.weight", V * NU, f"Array<{V},Array<{NU},Value<16>>>")]
    off, woff = 0, {}
    for n, c, _t in fields:
        woff[n] = (off, off + c)
        off += c
    wtype = "{" + ", ".join(f"{n}: {t}" for n, _c, t in fields) + "}"
    rows = []

    def add(spec, name, args):
        rows.append({"i": len(rows), "spec": spec, "form": "call", "name": name, "args": args})
        return len(rows) - 1

    W = lambda n: [["p", 1, woff[n][0], woff[n][1]]]  # noqa: E731
    sel = None
    for s in range(1 + T):
        toks = list(range(LP)) if s == 0 else [0]
        es = [add(f"Embedding_v1{{V={V},H={H}}}", f"model.model.embed_tokens/embedding_{s}[{k}]",
                  [[["p", 0, k, k + 1]] if s == 0 else [["n", sel, 0, 1]], W("model.model.embed_tokens.weight")]) for k in toks]
        lns = [add(f"RMSNormTriton_v1{{N={H},EPS={EPS}}}", f"model.model.layers.0.input_layernorm/triton_launch_{s}[{k}]",
                   [[["n", e, 0, H]], W("model.model.layers.0.input_layernorm.weight")]) for k, e in zip(toks, es)]
        ups = [add(f"Gemm_v1{{K={H},N={NU}}}", f"model.{SITE}/triton_launch_{s}a[{k}]",
                   [[["n", ln, 0, H]], W("model.model.layers.0.mlp.up_proj.weight")]) for k, ln in zip(toks, lns)]
        lg = add(f"Gemm_v1{{K={NU},N={V}}}", f"model.logits_processor/triton_launch_{s}c[0]", [[["n", ups[-1], 0, NU]], W("model.lm_head.weight")])
        sel = add(f"TokenSelect_v1{{V={V}}}", f"argmax_{s}[0]", [[["n", lg, 0, V]]])
    inst = {"schema": "cb-a/instance-sequence/v1", "digest": "t" * 64, "params": [["prompt", f"Array<{LP},Value<32>>"], ["weights", wtype]],
            "rows": rows}
    return inst, woff


class Apply:
    def apply(self, layer, x, bias=None):
        raise AssertionError("the PoUW executor replaces this apply")


class Linear:
    def __init__(self, weight):
        self.weight, self.quant_method = weight, Apply()


class Model:
    def __init__(self, *weights):
        self.layers = [Linear(w) for w in weights]

    def modules(self):
        return iter(self.layers)


class Run:
    """The prover: the model rows by the registry evaluators, each step's PoUW batch by the adapter's `ncp-v2` executor,
    module outputs packed per (step, member) with the requests side by side (r0 then r1)."""

    def __init__(self, inst, woff, weights, prompts, tmp_path, monkeypatch):
        self.inst, self.woff, self.weights, self.prompts = inst, woff, weights, prompts
        self.pi = IX.ProgramIndex(inst)
        self.values, self.blocks, self.dtypes = {}, {}, {}
        lo, hi = woff["model.model.layers.0.mlp.up_proj.weight"]
        w_up = (weights[lo:hi].astype(np.uint32) << 16).view(np.float32).reshape(NU, H)
        monkeypatch.setattr(pouw, "linear_types", lambda: (Linear,))
        self.model = Model(w_up)
        ctx = PO.Context(stage="serve", enabled=("pouw:ncp-v2",), settings={"pouw": {"beacon": BEACON, "transcript_dir": str(tmp_path)}})
        with PO.install(self.model, ctx) as comp:
            self._run()
            opt = comp.get("pouw")
        self.commitment = opt.commitment()
        import json
        doc = json.loads((tmp_path / PC.CALLS_FILE).read_text())
        assert doc["commitment"] == self.commitment
        self.records = [PC.record_of(c) for c in doc["calls"]]

    def _inputs(self, rid, r):
        pi, inputs, wsl = self.pi, [], []
        for group in r["args"]:
            words, w = [], None
            for kind, a, lo, hi in group:
                if kind == "p" and pi.pnames[a] == "weights":
                    w = self.weights[lo:hi]
                elif kind == "p":
                    words.append(np.asarray(self.prompts[rid][lo:hi], dtype=np.uint32))
                else:
                    words.append(self.values[(rid, a)][lo:hi])
            inputs.append(words)
            wsl.append(w)
        return inputs, wsl

    def _commit(self, rid, d, fam, flat, ids=None):
        m = d["members"][0]
        key = (d["step"], rid, d["op_path"], m[0])
        self.dtypes[key] = str(m[2])
        blk = flat.astype(np.uint64 if fam == "TokenSelect_v1" else np.uint16)
        self.blocks[key] = np.concatenate([self.blocks[key], blk]) if key in self.blocks else blk
        if ids is not None:
            k2 = (d["step"], rid, d["op_path"], "in")
            self.blocks[k2] = np.concatenate([self.blocks[k2], ids]) if k2 in self.blocks else ids

    def _run(self):
        pi, done = self.pi, set()
        for r in pi.rows:
            d = pi.describe(r)
            fam = d["family"]
            if fam == PC.ROW_FAMILY:
                if d["step"] in done:
                    continue
                done.add(d["step"])
                batch = [(rid, rr) for rid in ("r0", "r1") for rr in pi.rows
                         if pi.describe(rr)["family"] == PC.ROW_FAMILY and pi.describe(rr)["step"] == d["step"]]
                x = np.stack([(np.concatenate(self._inputs(rid, rr)[0][0]).astype(np.uint32) << 16).view(np.float32) for rid, rr in batch])
                y = self.model.layers[0].quant_method.apply(self.model.layers[0], x)
                ybits = (np.asarray(y, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)
                for (rid, rr), row in zip(batch, ybits):
                    self.values[(rid, rr["i"])] = row
                    self._commit(rid, pi.describe(rr), fam, row)
                continue
            for rid in ("r0", "r1"):
                inputs, wsl = self._inputs(rid, r)
                got = EV.evaluate(fam, d["statics"], inputs, wsl)
                flat = np.asarray(next(iter(got.values()))).reshape(-1)
                self.values[(rid, r["i"])] = flat.astype(np.uint64 if fam == "TokenSelect_v1" else np.uint16)
                self._commit(rid, d, fam, flat, ids=inputs[0][0].astype(np.uint32) if fam == "Embedding_v1" else None)

    def bmap_and_reader(self):
        packed: dict = {}
        for (step, rid, op, mem), blk in sorted(self.blocks.items(), key=lambda kv: (kv[0][0], kv[0][2], kv[0][3], ("r0", "r1").index(kv[0][1]))):
            packed.setdefault((step, op, mem), []).append((rid, blk))
        entries, store = [], {}
        for (step, op, mem), parts in packed.items():
            name, off, arrs = f"{op}/{mem}", 0, []
            for rid, blk in parts:
                dt = self.dtypes.get((step, rid, op, mem)) or ("i64" if op == IX.SAMPLER_OP else ("i32" if mem == "in" else "bf16"))
                entries.append({"identity": {"program_digest": "t" * 64, "request_id": rid, "step": step, "invocation": 0, "op_path": op,
                                             "output_member": mem, "element_range": [0, int(blk.size)], "rank": 0},
                                "name": name, "element_offset": off, "encoding": {"dtype": dt}, "family": "instance_outputs"})
                off += int(blk.size)
                arrs.append(blk)
            store[(step, name)] = np.concatenate(arrs).tobytes()

        def committed(step, name, lo, hi):
            b = store.get((int(step), name))
            return None if b is None else b[lo:hi]
        return {"schema": "test", "entries": entries}, committed

    def weights_reader(self):
        def read(field, lo, hi):
            flo, _fhi = self.woff[field]
            return self.weights[flo + lo: flo + hi]
        return read


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    rng = np.random.default_rng(5)
    inst, woff = _program()
    inst = PC.lower(inst, {f"model.{SITE}": "ncp-v2"})
    weights = _bf16(rng, max(hi for _lo, hi in woff.values()), 0.5)
    lo, hi = woff["model.model.layers.0.input_layernorm.weight"]
    weights[lo:hi] = 0x3F80
    with pytest.MonkeyPatch.context() as mp:
        return Run(inst, woff, weights, {"r0": [(i + 1) % V for i in range(LP)], "r1": [(3 * i) % V for i in range(LP)]},
                   tmp_path_factory.mktemp("pouw"), mp)


def _verdict(world, records=None, commitment=None, source=b"\x07" * 32, law=None):
    records = world.records if records is None else records
    commitment = world.commitment if commitment is None else commitment
    bmap, committed = world.bmap_and_reader()
    with RC.linked(records, {SITE: 0}) as issues:
        rep = SR.sampled_replay({rid: world.inst for rid in world.prompts}, bmap, committed, world.weights_reader(), per_stratum=2, seed=3, workers=0)
    return rep, RC.replay(records, commitment, source, law), issues


def _recommitted(world, records):
    """A prover that commits its (wrong) records honestly: the root is recomputed over them."""
    salt, wroot = bytes.fromhex(world.commitment["salt"]), bytes.fromhex(world.commitment["weights_root"])
    return dict(world.commitment, calls_root=PC.calls_root(salt, wroot, records).hex())


def test_the_build_lowers_only_the_pouw_site():
    inst, _ = _program()
    low = PC.lower(inst, {f"model.{SITE}": "ncp-v2"})
    specs = [r["spec"] for r in low["rows"]]
    assert sum(s == f"{PC.ROW_FAMILY}{{K={H},N={NU}}}" for s in specs) == LP + T
    assert all(s.startswith(PC.ROW_FAMILY) for r, s in zip(low["rows"], specs) if SITE in r["name"])
    assert not any(s.startswith(PC.ROW_FAMILY) for r, s in zip(low["rows"], specs) if SITE not in r["name"])
    assert PC.program("ncp-v2", 16, H, NU) is PC.program("ncp-v2", 16, H, NU)          # one Program per distinct shape


def test_the_rows_native_evaluation_is_the_registered_definition():
    """`NcpLinearRow_v1` (`program.registry.pouw_rows`) evaluated by the IR reference is what the replay computes natively, on rows at
    three scales, a row of subnormals and a zero row."""
    from verity.evaluation import evaluate
    from verity.ir.defs import bind
    from verity_vllm.program.registry import pouw_rows
    rng = np.random.default_rng(1)
    d = bind(pouw_rows.NcpLinearRow, K=H, N=NU)
    for scale in (1e-3, 1.0, 50.0, 1e-39, 0.0):
        x = _bf16(rng, H, scale)
        w = _bf16(rng, NU * H, 0.5)
        a, s, b, t, z = RC.int7_linear(x, w, NU)
        assert list(evaluate(d, [int(v) for v in x], [int(v) for v in w])) == RC.dequant(z, s, t, (0,) * NU), scale


def test_each_step_is_one_ncp_v2_call_bound_committed_and_served_as_y(world):
    assert [(r.call.forward, r.call.weight, r.shape.m, r.shape.k, r.shape.n) for r in world.records] == [(0, 0, 32, H, NU), (1, 0, 16, H, NU)]
    assert world.commitment["scheme"] == "ncp-v2" and world.commitment["calls"] == 2 and world.commitment["calls_root"]
    for rec in world.records:
        c = PC.program("ncp-v2", rec.shape.m, H, NU)
        assert c.traced.definition.id.startswith("NcpLinear_v1") and c.traced.served == rec.shape.m * NU
        from verity_pouw.circuit import reference as CR
        ref = CR.evaluate(rec.call)
        assert [list(r) for r in rec.y] == ref.y                                        # the Program's y is the reference's


def test_the_honest_run_is_accepted(world):
    rep, pw, issues = _verdict(world)
    assert rep["result"] is True and rep["grade"] == "complete", rep["why"]
    assert PC.ROW_FAMILY in rep["population"]["by_family"] and rep["sample"]["mismatch_n"] == 0
    assert pw["result"] is True and pw["law"] == "count" and pw["proved_units"] > 0, pw["why"]
    assert issues == []
    roles = {r["call"]: r for r in pw["calls"]}
    assert all(c["proved"] >= c["drawn"] > 0 for c in roles.values())


def test_without_the_delegate_the_pouw_rows_are_named_unreplayable_never_evaluated(world):
    bmap, committed = world.bmap_and_reader()
    rep = SR.sampled_replay({rid: world.inst for rid in world.prompts}, bmap, committed, world.weights_reader(), per_stratum=2, seed=3, workers=0)
    named = [u for u in rep["unreplayable"] if u["family"].startswith(PC.ROW_FAMILY)]
    assert named and named[0]["n"] == 2 * (LP + T) and named[0]["why"] == "no registered evaluator"
    assert not any(k.startswith(PC.ROW_FAMILY) for k in rep["strata"])                   # never evaluated, never a stratum


def _form_x_out(rec):
    c = PC.program(rec.scheme, rec.shape.m, rec.shape.k, rec.shape.n)
    lay = c.traced.layout()
    u = next(u.index for u in lay.units if u.role == "form_x")
    return c.partition.outputs(u)


def test_a_wrong_x_strip_is_rejected(world):
    bad = world.records[1]
    g = _form_x_out(bad)[5]
    rec = dataclasses.replace(bad, committed={**bad.committed, g: bad.committed[g] ^ 1})
    records = [world.records[0], rec]
    _rep, pw, _issues = _verdict(world, records, _recommitted(world, records))
    assert pw["result"] is False and any("form_x" in w for w in pw["why"]), pw["why"]


def test_a_wrong_tile_word_is_rejected(world):
    bad = world.records[0]
    c = PC.program(bad.scheme, bad.shape.m, H, NU)
    lay = c.traced.layout()
    from verity.randomness import derive
    source = b"\x07" * 32
    drawn = CPL.draw(derive(source, CPL.DRAW_DOMAIN, {"call": bad.index}), lay, CPL.CountLaw(8)).drawn
    tile = next(u for u in drawn if lay.units[u].role == "tile")
    g = c.partition.outputs(tile)[7]
    rec = dataclasses.replace(bad, committed={**bad.committed, g: bad.committed[g] ^ 1})
    records = [rec, world.records[1]]
    _rep, pw, _issues = _verdict(world, records, _recommitted(world, records), source)
    assert pw["result"] is False and any("tile" in w for w in pw["why"]), pw["why"]


@pytest.mark.slow
def test_an_a_strip_that_is_not_the_quantized_input_is_rejected(world):
    """A prover that runs the call on a different first strip of A, consistently: its proofs hold, and the model rows it
    serves are that call's y. The sampled rows at the PoUW site are not the int7 linear of their committed inputs, and the
    call's A is not their quantized input."""
    bad = world.records[0]
    a = tuple(tuple(((v + 1 + 63) % 127) - 63 for v in row) if i < 16 else row for i, row in enumerate(bad.call.a))
    rec = PC.commit_call(bad.index, bad.scheme, dataclasses.replace(bad.call, a=a))
    records = [rec, world.records[1]]
    _rep, pw, issues = _verdict(world, records, _recommitted(world, records))
    assert pw["result"] is True                                                         # the circuit alone can't see it
    assert any("A row" in i for i in issues) and any("served y row" in i for i in issues), issues


@pytest.mark.parametrize("scheme", ["ncp-v1", "ncp-v1-shift24", "pearl-fp8-v4"])
def test_every_other_scheme_stays_refused_beside_sampled_proofs(scheme):
    ctx = PO.Context(stage="commit", enabled=(f"pouw:{scheme}", "sampled-proofs:vllm-v1"), settings={"pouw": {"beacon": BEACON}})
    assert any("no Definition" in r for r in PO.refusals(ctx))
    ok = PO.Context(stage="commit", enabled=("pouw:ncp-v2", "sampled-proofs:vllm-v1"), settings={"pouw": {"beacon": BEACON}})
    assert PO.refusals(ok) == ()
