"""Sampled exact replay of the block-scaled FP8 linear (Qwen3-4B-FP8's `Fp8GroupQuant_v1 -> ScaledMmFp8Block_v1` under one
LinearBase module): the e4m3 `weight` and the f32 `weight_scale_inv` are REGISTERED inputs read in their declared widths (`Value<8>`,
`Value<32>` -- a mixed-width `weights` struct), the activation words and group scales are INTERIOR (composed from the committed bf16
row, two ports read as two arg slices), the GEMM row is the VU.  Positive on a tiny honest Program (twins = the Definitions,
tests/program/test_derived_rows_fp8.py); negatives: a flipped output word, a perturbed block scale, a 16-bit provider for an 8-bit
field (refused by name, never a silent cast)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from verity_vllm.check.replay import compare as CMP, driver as SR, evaluate as EV, index as IX, opening as OP
from verity_vllm.query.required import Q_MODULE_BODY_ID

EPS = '{"f64":"0x1.4f8b588e368f1p-17"}'
V, H, I, LP, T, G = 8, 256, 128, 2, 2, 128     # H, I multiples of the 128 block; gate_up N = 2I = 256



@pytest.fixture(scope="module", autouse=True)
def _under_module_body():
    """The FP8 quant is interior to the linear's module body here (the replay composes it into the Gemm VU): `Q_module_body_v1` only.
    Under the query of record the quant's outputs are Call boundaries, committed like any other."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(IX, "DEFAULT_QUERY", Q_MODULE_BODY_ID)
        yield

def _bf16(rng, n, scale=1.0):
    return (np.asarray(rng.standard_normal(n) * scale, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _e4m3(rng, n):
    b = rng.integers(0, 256, n, dtype=np.int64)
    b = np.where((b & 0x7F) == 0x7F, b & ~0x40, b)                 # no NaN bytes
    return b.astype(np.uint8)


def _program_fp8():
    """One request Program: per token row Embedding -> RMSNormTriton -> [Fp8GroupQuant -> ScaledMmFp8Block](gate_up) -> SiluMul ->
    [Fp8GroupQuant -> ScaledMmFp8Block](down) -> RMSNormFused(norm); per step Gemm(logits, tied embed) -> TokenSelect.  Row names / args
    in the Build's instance-sequence schema (the FP8 rows exactly as derive_step emits them: `<module>/per_token_group_fp8_quant_<s>[k]`,
    `<module>/cutlass_scaled_mm_<s>[k]`, the GEMM's args = (quant refs [0,K), quant refs [K, K+K/G), weight slice, scale slice))."""
    fields = [("model.model.embed_tokens.weight", V * H, 16, f"Array<{V},Array<{H},Value<16>>>"),
              ("model.model.layers.0.input_layernorm.weight", H, 16, f"Array<{H},Value<16>>"),
              ("model.model.layers.0.mlp.gate_up_proj.weight", 2 * I * H, 8, f"Array<{2 * I},Array<{H},Value<8>>>"),
              ("model.model.layers.0.mlp.gate_up_proj.weight_scale_inv", (2 * I // G) * (H // G), 32, f"Array<{2 * I // G},Array<{H // G},Value<32>>>"),
              ("model.model.layers.0.mlp.down_proj.weight", H * I, 8, f"Array<{H},Array<{I},Value<8>>>"),
              ("model.model.layers.0.mlp.down_proj.weight_scale_inv", (H // G) * (I // G), 32, f"Array<{H // G},Array<{I // G},Value<32>>>"),
              ("model.model.norm.weight", H, 16, f"Array<{H},Value<16>>")]
    off, woff, wbits = 0, {}, {}
    for n, c, b, _t in fields:
        woff[n] = (off, off + c)
        wbits[n] = b
        off += c
    wtype = "{" + ", ".join(f"{n}: {t}" for n, _c, _b, t in fields) + "}"
    rows, i = [], 0

    def add(spec, name, args):
        nonlocal i
        rows.append({"i": i, "spec": spec, "form": "call", "name": name, "args": args})
        i += 1
        return i - 1

    W = lambda n: [["p", 1, woff[n][0], woff[n][1]]]  # noqa: E731
    sel_prev = None
    for s in range(1 + T):
        toks = list(range(LP)) if s == 0 else [0]
        norm_rows = []
        for k in toks:
            ids = [["p", 0, k, k + 1]] if s == 0 else [["n", sel_prev, 0, 1]]
            e = add(f"Embedding_v1{{V={V},H={H}}}", f"model.model.embed_tokens/embedding_{s}[{k}]", [ids, W("model.model.embed_tokens.weight")])
            ln = add(f"RMSNormTriton_v1{{N={H},EPS={EPS}}}", f"model.model.layers.0.input_layernorm/triton_launch_{s}[{k}]",
                     [[["n", e, 0, H]], W("model.model.layers.0.input_layernorm.weight")])
            q1 = add(f"Fp8GroupQuant_v1{{K={H},G={G}}}", f"model.model.layers.0.mlp.gate_up_proj/per_token_group_fp8_quant_{s}[{k}]", [[["n", ln, 0, H]]])
            gu = add(f"ScaledMmFp8Block_v1{{K={H},N={2 * I},G={G}}}", f"model.model.layers.0.mlp.gate_up_proj/cutlass_scaled_mm_{s}[{k}]",
                     [[["n", q1, 0, H]], [["n", q1, H, H + H // G]], W("model.model.layers.0.mlp.gate_up_proj.weight"),
                      W("model.model.layers.0.mlp.gate_up_proj.weight_scale_inv")])
            act = add(f"SiluMul_v1{{I={I}}}", f"model.model.layers.0.mlp.act_fn/silu_and_mul_{s}[{k}]", [[["n", gu, 0, 2 * I]]])
            q2 = add(f"Fp8GroupQuant_v1{{K={I},G={G}}}", f"model.model.layers.0.mlp.down_proj/per_token_group_fp8_quant_{s}[{k}]", [[["n", act, 0, I]]])
            dn = add(f"ScaledMmFp8Block_v1{{K={I},N={H},G={G}}}", f"model.model.layers.0.mlp.down_proj/cutlass_scaled_mm_{s}[{k}]",
                     [[["n", q2, 0, I]], [["n", q2, I, I + I // G]], W("model.model.layers.0.mlp.down_proj.weight"),
                      W("model.model.layers.0.mlp.down_proj.weight_scale_inv")])
            nm = add(f"RMSNormFusedCuda_v2{{N={H},EPS={EPS}}}", f"model.model.norm/fused_add_rms_norm_{s}[{k}]",
                     [[["n", dn, 0, H]], [["n", e, 0, H]], W("model.model.norm.weight")])
            norm_rows.append(nm)
        lg = add(f"Gemm_v1{{K={H},N={V}}}", f"model.logits_processor/triton_launch_{s}c[0]", [[["n", norm_rows[-1], 0, H]], W("model.model.embed_tokens.weight")])
        sel_prev = add(f"TokenSelect_v1{{V={V}}}", f"argmax_{s}[0]", [[["n", lg, 0, V]]])
    inst = {"schema": "cb-a/instance-sequence/v1", "digest": "f" * 64, "params": [["prompt", f"Array<{LP},Value<32>>"], ["weights", wtype]], "rows": rows}
    return inst, woff, wbits


class HonestFp8:
    """Executes the FP8 Program per request with `EV.evaluate` (the twins) and commits the module outputs per (engine step, member):
    registered weights held PER FIELD in the declared width (u8 / u16 / u32), interior rows (the quant's two ports) flow on unwidened."""

    def __init__(self, inst, woff, wbits, fields: dict[str, np.ndarray], prompts):
        self.inst, self.woff, self.wbits, self.fields, self.prompts = inst, woff, wbits, fields, prompts
        self.pi = IX.ProgramIndex(inst)
        self.values: dict[tuple[str, int], dict[str, np.ndarray]] = {}
        self.blocks: dict[tuple[int, str, str, str], np.ndarray] = {}
        for rid, prompt in prompts.items():
            self._run(rid, prompt)

    def read(self, field, lo, hi):
        return self.fields[field][lo:hi]

    def _run(self, rid, prompt):
        pi = self.pi
        for r in pi.rows:
            d = pi.describe(r)
            fam = d["family"]
            inputs, wsl = [], []
            for group in r["args"]:
                words, w = [], None
                for kind, a, lo, hi in group:
                    if kind == "p" and pi.pnames[a] == "weights":
                        ws = pi.weight_slice(lo, hi)
                        w = self.read(ws[0], ws[1], ws[2])
                    elif kind == "p":
                        words.append(np.asarray(prompt[lo:hi], dtype=np.uint32))
                    else:
                        words.append(CMP._interior_slice(self.values[(rid, a)], lo, hi))
                inputs.append(words)
                wsl.append(w)
            got = EV.evaluate(fam, d["statics"], inputs, wsl)
            self.values[(rid, r["i"])] = {k: np.asarray(v).reshape(-1) for k, v in got.items()}
            if not d["members"]:
                continue
            for m in d["members"]:
                v = np.asarray(got[m[0]] if m[0] in got or len(got) != 1 else next(iter(got.values()))).reshape(-1)   # one output: the member
                v = v.astype(np.uint16) if fam != "TokenSelect_v1" else v.astype(np.uint64)
                key = (d["step"], rid, d["op_path"], m[0])
                self.blocks[key] = np.concatenate([self.blocks[key], v]) if key in self.blocks else v
            if fam == "Embedding_v1":
                ids = inputs[0][0].astype(np.uint32)
                key = (d["step"], rid, d["op_path"], "in")
                self.blocks[key] = np.concatenate([self.blocks[key], ids]) if key in self.blocks else ids

    def bmap_and_reader(self, order=("r0", "r1"), mutate=None):
        packed: dict[tuple[int, str, str], list] = {}
        for (step, rid, op, mem), blk in sorted(self.blocks.items(), key=lambda kv: (kv[0][0], kv[0][2], kv[0][3], order.index(kv[0][1]))):
            packed.setdefault((step, op, mem), []).append((rid, blk))
        entries, store = [], {}
        dtype_of = lambda op, mem: "i64" if op == IX.SAMPLER_OP else ("i32" if mem == "in" else "bf16")  # noqa: E731
        for (step, op, mem), parts in packed.items():
            name = f"{op}/{mem}"
            off, arrs = 0, []
            for rid, blk in parts:
                entries.append({"identity": {"program_digest": "f" * 64, "request_id": rid, "step": step, "invocation": 0, "op_path": op, "output_member": mem,
                                             "element_range": [0, int(blk.size)], "rank": 0}, "name": name, "element_offset": off,
                                "encoding": {"dtype": dtype_of(op, mem)}, "family": "instance_outputs"})
                off += int(blk.size)
                arrs.append(blk)
            store[(step, name)] = np.concatenate(arrs).tobytes()
        if mutate:
            mutate(store, entries)

        def committed(step, name, lo, hi):
            b = store.get((int(step), name))
            return None if b is None else b[lo:hi]

        return {"schema": "test", "entries": entries}, committed


@pytest.fixture(scope="module")
def fp8_world():
    rng = np.random.default_rng(19)
    inst, woff, wbits = _program_fp8()
    fields = {}
    for n, (lo, hi) in woff.items():
        c, b = hi - lo, wbits[n]
        if b == 8:
            fields[n] = _e4m3(rng, c)
        elif b == 32:
            fields[n] = np.exp2(rng.uniform(-9, -3, c)).astype(np.float32).view(np.uint32)      # block scales: positive f32 words
        else:
            fields[n] = _bf16(rng, c, 0.5)
    fields["model.model.layers.0.input_layernorm.weight"][:] = 0x3F80
    fields["model.model.norm.weight"][:] = 0x3F80
    return HonestFp8(inst, woff, wbits, fields, {"r0": [1, 2], "r1": [5, 3]})


def _programs(world):
    return {rid: world.inst for rid in world.prompts}


def test_mixed_width_struct_parses_the_declared_leaf_widths(fp8_world):
    pi = fp8_world.pi
    assert pi.weight_bits("model.model.layers.0.mlp.gate_up_proj.weight") == 8
    assert pi.weight_bits("model.model.layers.0.mlp.gate_up_proj.weight_scale_inv") == 32
    assert pi.weight_bits("model.model.norm.weight") == 16
    ws = pi.weight_slice(*fp8_world.woff["model.model.layers.0.mlp.down_proj.weight_scale_inv"])
    assert ws[0] == "model.model.layers.0.mlp.down_proj.weight_scale_inv" and ws[3] == [H // G, I // G]


def test_honest_fp8_program_replays_bit_exactly_the_gemm_rows_are_vus_the_quant_is_interior(fp8_world):
    bmap, committed = fp8_world.bmap_and_reader()
    rep = SR.sampled_replay(_programs(fp8_world), bmap, committed, fp8_world.read, per_stratum=2, seed=5, workers=0)
    assert rep["result"] is True and rep["grade"] == "complete", json.dumps(rep, indent=1, default=str)[:3000]
    assert rep["sample"]["mismatch_n"] == 0 and rep["sample"]["evaluated"] == rep["sample"]["picked"] > 0
    fams = set(rep["population"]["by_family"])
    assert "ScaledMmFp8Block_v1" in fams and "RMSNormFusedCuda_v2" in fams and "SiluMul_v1" in fams
    assert "Fp8GroupQuant_v1" not in fams                                               # interior: no VU of its own
    interior = [k for k in rep["population"]["not_evaluable_by_family"] if k.startswith("Fp8GroupQuant_v1 (interior")]
    assert interior and rep["population"]["not_evaluable_by_family"][interior[0]] == 2 * 2 * (LP + T)   # 2 quants x 2 requests x rows
    assert not any(k.startswith("ScaledMmFp8Block") for k in rep["population"]["not_evaluable_by_family"])
    gemm_strata = {k: v for k, v in rep["strata"].items() if k.startswith("ScaledMmFp8Block")}
    assert gemm_strata and all(v["evaluated"] >= 1 for v in gemm_strata.values())
    assert {k.split("|")[4].rsplit(".", 1)[-1] for k in gemm_strata} == {"gate_up_proj", "down_proj"}
    assert rep["evaluators"]["ScaledMmFp8Block_v1"].startswith("verity.ml.definitions.numerics.derived_rows.scaled_mm_fp8_block_row")
    assert rep["evaluators"]["Fp8GroupQuant_v1"].startswith("verity.ml.definitions.numerics.derived_rows.fp8_group_quant_row")
    # the twins name what pinned them: fixture sha256 (the file in the tree), kernels, device, date, tests
    import hashlib
    from pathlib import Path
    prov = rep["evaluator_provenance"]
    assert set(prov) == {"Fp8GroupQuant_v1", "ScaledMmFp8Block_v1"}
    npz = Path(__file__).resolve().parents[2] / prov["ScaledMmFp8Block_v1"]["fixture"]
    assert hashlib.sha256(npz.read_bytes()).hexdigest() == prov["ScaledMmFp8Block_v1"]["fixture_sha256"]
    assert "cp-h100" in prov["ScaledMmFp8Block_v1"]["device"] and prov["ScaledMmFp8Block_v1"]["captured_utc"] == "2026-09-19"
    assert "Embedding_v1" not in prov                                                    # registry-own evaluators carry no twin provenance
    # the record names the composition: the GEMM VU's inputs are the interior quant (from the committed norm / act row) + 8- and 32-bit weights
    ev = [e for e in rep.get("evaluated_examples", []) if e.get("family") == "ScaledMmFp8Block_v1"] if isinstance(rep.get("evaluated_examples"), list) else []
    if ev:
        kinds = [(x["kind"], x.get("family"), x.get("bits")) for x in ev[0]["inputs"]]
        assert ("interior", "Fp8GroupQuant_v1", None) in kinds and ("weights", None, 8) in kinds and ("weights", None, 32) in kinds


def test_gemm_vu_replay_record_composes_the_quant_and_reads_both_weight_widths(fp8_world):
    bmap, committed = fp8_world.bmap_and_reader()
    store = OP.CommittedStore(bmap, committed)
    pi = fp8_world.pi
    r = next(r for r in pi.rows if r["spec"].startswith("ScaledMmFp8Block") and "gate_up" in r["name"] and pi.step_of(r["i"]) == 1)
    rec = EV.replay_vu(pi, store, fp8_world.read, {"i": r["i"], "request_id": "r1", "phase": "first-decode"}, arrive=0)
    assert rec["result"] is True, rec
    kinds = [(x["kind"], x.get("family"), x.get("bits")) for x in rec["inputs"]]
    assert kinds[0] == ("interior", "Fp8GroupQuant_v1", None) and kinds[1] == ("interior", "Fp8GroupQuant_v1", None)
    assert ("weights", None, 8) in kinds and ("weights", None, 32) in kinds
    assert rec["inputs"][0]["range"] == [0, H] and rec["inputs"][1]["range"] == [H, H + H // G]          # the two ports of the quant
    assert rec["inputs"][0]["inputs"][0]["kind"] == "committed" and rec["inputs"][0]["inputs"][0]["identity"][2] == "model.layers.0.input_layernorm"
    assert rec["outputs"] == [{"member": "out", "width": 2 * I, "equal": True, "differing_elements": 0, "first_diff": None}]


def test_flipped_fp8_gemm_output_word_fails_named(fp8_world):
    def mutate(store, entries):
        k = (1, "model.layers.0.mlp.down_proj/out")
        b = bytearray(store[k])
        b[3] ^= 0x01
        store[k] = bytes(b)
    bmap, committed = fp8_world.bmap_and_reader(mutate=mutate)
    rep = SR.sampled_replay(_programs(fp8_world), bmap, committed, fp8_world.read, per_stratum=2, seed=5, workers=0)
    assert rep["result"] is False
    assert any("ScaledMmFp8Block_v1 r0 step 1 model.layers.0.mlp.down_proj/out" in str(m.get("why")) for m in rep["mismatch"])


def test_perturbed_block_scale_fails_the_gemm_vu_a_load_bearing_registered_input(fp8_world):
    """One mantissa bit of one `weight_scale_inv` block: F_V(committed inputs) moves on that block's columns -> mismatch, named."""
    bad = dict(fp8_world.fields)
    f = "model.model.layers.0.mlp.gate_up_proj.weight_scale_inv"
    bad[f] = bad[f].copy()
    bad[f][0] ^= np.uint32(0x00080000)

    def read(field, lo, hi):
        return bad[field][lo:hi]

    bmap, committed = fp8_world.bmap_and_reader()
    rep = SR.sampled_replay(_programs(fp8_world), bmap, committed, read, per_stratum=2, seed=5, workers=0)
    assert rep["result"] is False
    assert all("gate_up_proj" in str(m.get("why")) for m in rep["mismatch"]) and rep["mismatch"]


def test_weight_field_provided_in_the_wrong_width_is_refused_by_name_never_cast(fp8_world):
    """A provider handing a Value<8> field as 16-bit words (the older `np.uint16` cast) is an unresolved input, not a silent evaluation."""
    def read16(field, lo, hi):
        a = fp8_world.fields[field][lo:hi]
        return a.astype(np.uint16) if a.dtype == np.uint8 else a

    bmap, committed = fp8_world.bmap_and_reader()
    store = OP.CommittedStore(bmap, committed)
    pi = fp8_world.pi
    r = next(r for r in pi.rows if r["spec"].startswith("ScaledMmFp8Block"))
    rec = EV.replay_vu(pi, store, read16, {"i": r["i"], "request_id": "r0", "phase": "prefill"}, arrive=0)
    assert rec["result"] is None and "provided as uint16 but the Program declares Value<8>" in rec["why"]


def test_interior_slice_spans_the_quant_ports_in_return_order():
    got = {"0": np.arange(6, dtype=np.uint8), "1": np.array([7, 8], dtype=np.uint32)}
    assert CMP._interior_slice(got, 0, 6).tolist() == [0, 1, 2, 3, 4, 5]
    assert CMP._interior_slice(got, 6, 8).tolist() == [7, 8]
    assert CMP._interior_slice(got, 4, 8).tolist() == [4, 5, 7, 8]
    assert CMP._interior_slice(got, 0, 9) is None
