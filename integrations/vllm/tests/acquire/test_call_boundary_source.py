"""The call-boundary source (`acquire.sources.call_boundary_source`) on CPU tensors with a recording committer, on request Programs written
as Build dirs: a Gemma-2-style dense Program (the ATen norm chains, one input and x + residual, and the soft-capped logits) and the
block-scaled FP8 linear (`Fp8GroupQuant_v1 -> ScaledMmFp8Block_v1`).  Every `call_boundaries` identity of the manifest is committed once
per step over its element range with the words a row-by-row `verity.evaluation.evaluate` of the Program gives -- on edge words too (NaN
payloads, +-inf, +-0, subnormals, the largest finite); a manifest naming none attaches nothing and a forced attach refuses; a body the
rule cannot evaluate is refused by name; an argument of another shape is a collector error naming the site; and the consuming GEMM's
replay reads the committed words and checks the served output as before."""
from __future__ import annotations

import gzip
import json
import os
import types

import numpy as np
import pytest
from verity.evaluation import evaluate
from verity_vllm.correspondence.reader_for_query import Correspondence
from verity_vllm.query.call_scope import specialization
from verity_vllm.query.manifest.format import spec_family
from verity_vllm.query.program_view import definition_ports, from_instances, spec_statics
from verity_vllm.query.required import request_manifest

torch = pytest.importorskip("torch")

from verity_vllm.acquire.sources import call_boundary_source as CB  # noqa: E402
from verity_vllm.acquire.sources import taps  # noqa: E402

ONE, EPS, CAP = '{"f64":"0x1.0000000000000p+0"}', '{"f64":"0x1.0c6f7a0b5ed8dp-20"}', '{"f64":"0x1.e000000000000p+4"}'
V, H, LP, T = 16, 8, 7, 2
PF, IL, LG, EMB = "model.model.layers.0.post_feedforward_layernorm", "model.model.layers.1.input_layernorm", "model.logits_processor", "model.model.embed_tokens"
#: bf16 embedding rows the prompt selects: NaN payloads (quiet, negative, signalling), +-inf, +-0, subnormals, the largest finite, mixed
EDGE = [[0x7FC0, 0xFFC1, 0x7F81, 0x3F80, 0xBF80, 0x4000, 0x3F00, 0x0000],
        [0x7F80, 0xFF80, 0x3F80, 0x4040, 0xC000, 0x3E80, 0x0000, 0x8000],
        [0x0000, 0x8000] * 4,
        [0x0001, 0x807F, 0x0040, 0x8001, 0x007F, 0x0010, 0x8040, 0x0002],
        [0x7F7F, 0xFF7F] * 4,
        [0x7F7F, 0x0001, 0x8000, 0x3F80, 0xFF7F, 0x0080, 0xC2F7, 0x4B00]]
PROMPT = [0, 1, 2, 3, 4, 5, 9]
SIGNED = {1: np.uint8, 2: np.int16, 4: np.int32, 8: np.int64}
TORCH_OF = {"bf16": torch.bfloat16, "f32": torch.float32, "w8": torch.uint8}


class Recorder:
    def __init__(self, model=None) -> None:
        self._cur = None
        self.errors: list[str] = []
        self.acquired: list[tuple[int, str, object]] = []
        self.sources: list = []
        self.name = "rec"
        self.model = model

    def _acquire(self, name, t, job):
        self.acquired.append((int(job.step), name, t.clone()))

    def attach_source(self, src):
        self.sources.append(src)


class Probe(torch.nn.Module):
    def forward(self, *args, **kwargs):
        return None


def _bf16(rng, n, scale=1.0):
    return (np.asarray(rng.standard_normal(n) * scale, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _tensor(w: np.ndarray, dtype):
    return torch.from_numpy(np.ascontiguousarray(w).view(SIGNED[w.itemsize])).view(dtype)


class Rows:
    """An instance-sequence document under construction: the weights struct of `fields` [(Program name, elements, width, type)]."""

    def __init__(self, fields):
        self.rows, self.step, self.off, o = [], {}, {}, 0
        for n, c, _b, _t in fields:
            self.off[n] = (o, o + c)
            o += c
        self.wtype = "{" + ", ".join(f"{n}: {t}" for n, _c, _b, t in fields) + "}"

    def add(self, s, spec, name, *args):
        self.rows.append({"i": len(self.rows), "spec": spec, "form": "call", "name": name, "args": list(args)})
        self.step[len(self.rows) - 1] = s
        return len(self.rows) - 1

    def W(self, n):
        return [["p", 1, *self.off[n]]]

    def doc(self, lp):
        return {"schema": "cb-a/instance-sequence/v1", "digest": "e" * 64, "params": [["prompt", f"Array<{lp},Value<32>>"], ["weights", self.wtype]],
                "rows": self.rows}


DENSE = [(f"{EMB}.weight", V * H, 16, f"Array<{V},Array<{H},Value<16>>>"), (f"{PF}.weight", H, 16, f"Array<{H},Value<16>>"),
         (f"{IL}.weight", H, 16, f"Array<{H},Value<16>>")]


def dense_program(embed=EMB):
    """Per step (the prompt's LP rows, then one): embed -> a post-feedforward norm chain (one input) -> an input-layernorm chain (x +
    residual, widened) -> the soft-capped logits of the last row -> TokenSelect; node-major rows, as the Build writes a prefill.
    `embed` = the module the embedding rows are named under.  Returns (Rows, {(step, "e" | "x" | "y"): Calls})."""
    b, at = Rows(DENSE), {}
    sel = None
    for s in range(1 + T):
        ks = range(LP) if s == 0 else range(1)

        def each(spec, node, *reads):
            return [b.add(s, spec, f"{node}[{k}]", *([["n", r[0][k], 0, r[1]]] if isinstance(r, tuple) else r for r in reads)) for k in ks]

        e = [b.add(s, f"Embedding_v1{{V={V},H={H}}}", f"{embed}/embedding_{s}[{k}]", [["p", 0, k, k + 1]] if s == 0 else [["n", sel, 0, 1]],
                   b.W(f"{EMB}.weight")) for k in ks]
        a0 = each(f"AddScalarBf16_v1{{N={H},C={ONE}}}", f"{PF}/add_{s}", b.W(f"{PF}.weight"))
        sq = each(f"SquareBf16_v1{{N={H}}}", f"{PF}/pow_{s}", (e, H))
        mu = each(f"MeanTriton_v1{{N={H}}}", f"{PF}/mean_{s}", (sq, H))
        ve = each(f"AddScalarF32_v1{{N=1,C={EPS}}}", f"{PF}/add_1_{s}", (mu, 1))
        rs = each("RsqrtF32_v1{N=1}", f"{PF}/rsqrt_{s}", (ve, 1))
        sc = each(f"ScaleRowBf16_v1{{N={H}}}", f"{PF}/mul_{s}", (e, H), (rs, 1))
        mv = each(f"MulVecF32_v1{{N={H}}}", f"{PF}/mul_1_{s}", (sc, H), (a0, H))
        x = each(f"NarrowF32ToBf16_v1{{N={H}}}", f"{PF}/to_{s}/out", (mv, H))
        b0 = each(f"AddScalarBf16_v1{{N={H},C={ONE}}}", f"{IL}/add_{s}", b.W(f"{IL}.weight"))
        ad = each(f"AddWidenedBf16_v1{{N={H}}}", f"{IL}/add_1_{s}", (x, H), (e, H))
        sq = each(f"SquareF32_v1{{N={H}}}", f"{IL}/pow_{s}", (ad, H))
        mu = each(f"MeanTriton_v1{{N={H}}}", f"{IL}/mean_{s}", (sq, H))
        ve = each(f"AddScalarF32_v1{{N=1,C={EPS}}}", f"{IL}/add_2_{s}", (mu, 1))
        rs = each("RsqrtF32_v1{N=1}", f"{IL}/rsqrt_{s}", (ve, 1))
        sc = each(f"ScaleRowF32_v1{{N={H}}}", f"{IL}/mul_{s}", (ad, H), (rs, 1))
        mv = each(f"MulVecF32_v1{{N={H}}}", f"{IL}/mul_1_{s}", (sc, H), (b0, H))
        y = each(f"NarrowF32ToBf16_v1{{N={H}}}", f"{IL}/to_{s}/0", (mv, H))
        g = b.add(s, f"Gemm_v1{{K={H},N={V}}}", f"{LG}/triton_launch_{s}[0]", [["n", y[-1], 0, H]], b.W(f"{EMB}.weight"))
        dv = b.add(s, f"Bf16DivScalar_v1{{N={V},C={CAP}}}", f"{LG}/div_{s}[0]", [["n", g, 0, V]])
        th = b.add(s, f"Bf16Tanh_v1{{N={V}}}", f"{LG}/tanh_{s}[0]", [["n", dv, 0, V]])
        ml = b.add(s, f"Bf16MulScalar_v1{{N={V},C={CAP}}}", f"{LG}/mul_{s}/out[0]", [["n", th, 0, V]])
        sel = b.add(s, f"TokenSelect_v1{{V={V}}}", f"argmax_{s}[0]", [["n", ml, 0, V]])
        at.update({(s, "e"): e, (s, "x"): x, (s, "y"): y})
    return b, at


def dense_fields():
    rng = np.random.default_rng(11)
    table = _bf16(rng, V * H).reshape(V, H)
    table[: len(EDGE)] = np.asarray(EDGE, dtype=np.uint16)
    return {f"{EMB}.weight": table.reshape(-1), f"{PF}.weight": _bf16(rng, H, 0.5), f"{IL}.weight": _bf16(rng, H, 0.5)}


def reference(doc, prompt, fields, field_order):
    """Every Call's output leaves, row by row through `verity.evaluation.evaluate` (the IR's plain-integer reference)."""
    weights = [int(w) for n in field_order for w in fields[n]]
    vals: dict[int, list[int]] = {}
    for r in doc["rows"]:
        args = [[v for kind, a, lo, hi in g for v in (prompt[lo:hi] if kind == "p" and a == 0 else weights[lo:hi] if kind == "p" else vals[a][lo:hi])]
                for g in r["args"]]
        vals[r["i"]] = list(evaluate(specialization(spec_family(r["spec"]), spec_statics(r["spec"])), *args))
    return vals


def build_dir(tmp_path, doc, observed):
    """A Build dir holding `doc`, and the manifest of its one request (the implementation paths observed = `observed`)."""
    res = {"program": {"digest": "d0", "function": "f"}, "attention_impls_observed": {p: {} for p in observed},
           "target": {"declared": {"compute_capability": [8, 9]}}}
    d = tmp_path / "build_request"
    d.mkdir()
    with gzip.open(d / "instances.json.gz", "wt") as f:
        json.dump(doc, f)
    (d / "result.json").write_text(json.dumps(res))
    P = from_instances(str(d / "instances.json.gz"))
    return str(d), request_manifest(P, Correspondence(P, implementation_paths=observed), res, request_id="r0").manifest


def runtime(paths, tensors=None):
    """A module tree with a `Probe` at each implementation path and each {name: tensor} a buffer at its path."""
    root = Probe()

    def at(path):
        m = root
        for part in path.split("."):
            if part not in m._modules:
                m.add_module(part, Probe())
            m = m._modules[part]
        return m

    for p in paths:
        at(p)
    for n, t in (tensors or {}).items():
        owner, leaf = n.rsplit(".", 1)
        at(owner).register_buffer(leaf, t)
    return root


def ir_words(doc, vals, step_of, r):
    """The words of identity `r` in the reference: its Calls (the step's rows named `model.<op_path>[k]`) at the member's port."""
    op, node = r["op_path"].rsplit("/", 1)
    calls = [x for x in doc["rows"] if step_of[x["i"]] == r["step"] and x["name"].split("[")[0] == f"model.{op}/{node}"]
    port = next(q for q in definition_ports(calls[0]["spec"]) if q.name == r["output_member"])
    return np.array([vals[x["i"]][port.lo:port.hi] for x in calls], dtype=np.uint64)


@pytest.fixture(scope="module")
def dense(tmp_path_factory):
    b, at = dense_program()
    doc = b.doc(LP)
    fields = dense_fields()
    vals = reference(doc, PROMPT, fields, [n for n, *_ in DENSE])
    pdir, man = build_dir(tmp_path_factory.mktemp("dense"), doc, [PF[len("model."):]])
    return types.SimpleNamespace(b=b, at=at, doc=doc, fields=fields, vals=vals, pdir=pdir, man=man)


def _drive(world, com, src, *, steps=range(1 + T)):
    """The dense Program's module forwards, step by step, on the reference's words (the served kernels, honest)."""
    mods = dict(src.model.named_modules())
    rows = lambda calls: _tensor(np.array([world.vals[c][:H] for c in calls], dtype=np.uint16), torch.bfloat16)  # noqa: E731
    for s in steps:
        com._cur = types.SimpleNamespace(step=s)
        src.on_step_begin()
        mods[PF[len("model."):]](rows(world.at[(s, "e")]))
        mods[IL[len("model."):]](rows(world.at[(s, "x")]), residual=rows(world.at[(s, "e")]))
        mods["logits_processor"](mods["model.embed_tokens"], rows(world.at[(s, "y")][-1:]))
    com._cur = None


def _dense_source(world):
    model = runtime(["model.embed_tokens", PF[len("model."):], IL[len("model."):], "logits_processor"],
                    {n[len("model."):]: _tensor(w, torch.bfloat16) for n, w in world.fields.items()})
    com = Recorder(model)
    classes = taps.attach(com, world.man, ("instance_outputs",), programs=(None, os.path.dirname(world.pdir)))     # found under the root
    assert classes == ("instance_outputs", CB.FAMILY) and len(com.sources) == 1
    com.sources[0].install()
    return com, com.sources[0]


def test_every_call_boundary_is_committed_once_per_step_with_the_ir_words(dense):
    ids = CB.manifest_sites(dense.man)
    assert len(ids) == (1 + T) * (7 + 8 + 3)                    # per step: 7 of the one-input chain, 8 of x + residual, 3 of the logits
    com, src = _dense_source(dense)
    _drive(dense, com, src)
    assert not com.errors
    got = {(s, n): t for s, n, t in com.acquired}
    assert len(got) == len(com.acquired) == len(ids)
    assert set(got) == {(r["engine_step"], f"{r['op_path']}/{r['output_member']}") for r in ids}
    for r in ids:
        t = got[(r["engine_step"], f"{r['op_path']}/{r['output_member']}")]
        assert t.dtype == TORCH_OF[r["dtype"]] and tuple(t.shape) == (r["rows"], r["width"]) and t.numel() == r["element_range"][1] - r["element_range"][0]
        w = CB.words(torch, t).reshape(t.shape).astype(np.uint64)
        assert np.array_equal(w, ir_words(dense.doc, dense.vals, dense.b.step, r)), r["op_path"]
    body = src.values("model.layers.0.post_feedforward_layernorm", T)                       # the last step's words; earlier steps are released
    assert f"model.layers.0.post_feedforward_layernorm/rsqrt_{T}/out" in body and src.values("model.layers.0.post_feedforward_layernorm", 0) == {}
    assert src.stats["bodies"] == 3 * (1 + T) and src.stats["tensors"] == len(ids) and src.stats["errors"] == 0
    assert src.declaration_classes()[0]["class"] == CB.FAMILY and src.extra_step_bytes({}, None)["tensors_per_step"] == 18


def test_edge_words_through_the_norm_chains_are_the_ir_words(dense):
    """The prompt rows are the edge vectors: the chains' committed words are the reference's, NaN payloads, infinities and signed zeros
    included."""
    com, src = _dense_source(dense)
    _drive(dense, com, src, steps=[0])
    f32 = np.concatenate([CB.words(torch, t) for s, n, t in com.acquired if t.dtype == torch.float32])
    exp, frac = (f32 >> 23) & 0xFF, f32 & 0x7FFFFF
    nan = f32[(exp == 0xFF) & (frac != 0)]
    assert nan.size and {0x7F800000, 0x00000000, 0x80000000} <= set(f32.tolist())
    assert len(set(nan.tolist())) > 1                            # payloads kept: not one canonical NaN
    ids = [r for r in CB.manifest_sites(dense.man) if r["engine_step"] == 0]
    got = {n: t for _s, n, t in com.acquired}
    for r in ids:
        w = CB.words(torch, got[f"{r['op_path']}/{r['output_member']}"]).reshape(r["rows"], r["width"]).astype(np.uint64)
        assert np.array_equal(w, ir_words(dense.doc, dense.vals, dense.b.step, r)), r["op_path"]


def test_an_argument_the_body_does_not_read_is_a_collector_error_naming_the_site(dense):
    com, src = _dense_source(dense)
    mods = dict(src.model.named_modules())
    norm = mods[PF[len("model."):]]
    com._cur = types.SimpleNamespace(step=1)
    src.on_step_begin()
    norm(torch.zeros(2, H, dtype=torch.bfloat16))                # two rows where the Program's step 1 has one
    norm(torch.zeros(1, H, dtype=torch.bfloat16))                # the body's second forward at the step
    mods[IL[len("model."):]](torch.zeros(1, H, dtype=torch.bfloat16))   # the residual missing
    mods["logits_processor"](torch.zeros(1, H, dtype=torch.float32))    # another word width
    assert not com.acquired and len(com.errors) == 4
    assert "model.layers.0.post_feedforward_layernorm argument 0 is (2, 8)" in com.errors[0]
    assert "ran twice at engine step 1" in com.errors[1]
    assert "1 tensor arguments" in com.errors[2] and "reads 2 outside Values" in com.errors[2]
    assert "logits_processor argument 0" in com.errors[3] and "reads model.layers.1.input_layernorm/to_1/0:out as 1 rows x 8 Value<16>" in com.errors[3]
    com._cur = None
    norm(torch.zeros(1, H, dtype=torch.bfloat16))                # outside a committed step: nothing
    assert len(com.errors) == 4


def test_without_call_boundaries_nothing_attaches_and_a_forced_attach_refuses(dense, tmp_path):
    com = Recorder(Probe())
    plain = {**dense.man, "identities": [r for r in dense.man["identities"] if r["family"] != CB.FAMILY]}
    assert taps.attach(com, plain, ("instance_outputs",), programs=(dense.pdir,)) == ("instance_outputs",) and not com.sources
    assert CB.attach(com, False, dense.man, ("a",)) == (("a",), None)
    with pytest.raises(SystemExit) as e:
        CB.attach(com, True, plain, ("a",), programs=(dense.pdir,))
    assert e.value.code == 3
    with pytest.raises(SystemExit):
        CB.attach(com, True, dense.man, ("a",), programs=(str(tmp_path),))          # no Build dir of the manifest's Program
    with pytest.raises(SystemExit):
        CB.attach(com, True, {**dense.man, "requests": ["r0", "r1"]}, ("a",), programs=(dense.pdir,))
    assert not com.sources
    classes, src = CB.attach(com, True, dense.man, ("a",), explicit_classes=True, model=Probe(), programs=(dense.pdir,))
    assert classes == ("a",) and com.sources == [src] and src.plan.covered == src.plan.identities


def test_a_body_that_reads_a_program_input_or_another_step_is_refused_by_name(tmp_path):
    b, _at = dense_program(embed=PF)
    pdir, man = build_dir(tmp_path, b.doc(LP), [PF[len("model."):]])
    plan = CB.plan_of(man, pdir)
    assert plan.uncovered and plan.covered < plan.identities
    why = " ".join(plan.uncovered)
    assert "reads Program input 'prompt'" in why and "reads Call" in why and "of step 0" in why
    with pytest.raises(SystemExit):
        CB.attach(Recorder(Probe()), True, man, ("a",), programs=(pdir,))


# ------------------------------------------------------------------------------------------------------------------------------ FP8
@pytest.fixture(scope="module")
def fp8(tmp_path_factory):
    from tests.check.test_sampled_replay_fp8 import G, H as HF, I, _e4m3, _program_fp8

    inst, woff, wbits = _program_fp8()
    rng = np.random.default_rng(23)
    fields = {n: (_e4m3(rng, hi - lo) if wbits[n] == 8 else np.exp2(rng.uniform(-9, -3, hi - lo)).astype(np.float32).view(np.uint32)
                  if wbits[n] == 32 else _bf16(rng, hi - lo, 0.5)) for n, (lo, hi) in woff.items()}
    pdir, man = build_dir(tmp_path_factory.mktemp("fp8"), inst, ["model.layers.0.mlp.act_fn"])
    return types.SimpleNamespace(inst=inst, woff=woff, fields=fields, pdir=pdir, man=man, G=G, H=HF, I=I)


def _fp8_rows(rng, rows, n, edges):
    x = np.stack([_bf16(rng, n) for _ in range(rows)])
    if edges:
        x[0, :8] = [0x7FC0, 0xFFC1, 0x7F81, 0x3F80, 0x0001, 0x8000, 0x0000, 0x807F]         # group 0: NaN payloads, subnormals, +-0
        x[0, 128:132] = [0x7F80, 0xFF80, 0x3F80, 0xC000]                                   # group 1: +-inf
        x[1, :128] = np.resize(np.array([0x0000, 0x8000, 0x0001, 0x807F], np.uint16), 128)   # group 0: only +-0 and subnormals
        x[1, 128:130] = [0x7F7F, 0xFF7F]                                                   # group 1: the largest finite
    return x


def test_fp8_group_quant_on_edge_words_is_the_ir_and_the_consuming_gemm_replays(fp8):
    from verity_vllm.check.replay import evaluate as EV, index as IX, opening as OP

    rng = np.random.default_rng(5)
    model = runtime(["model.layers.0.mlp.gate_up_proj", "model.layers.0.mlp.down_proj"])
    com = Recorder(model)
    classes, src = CB.attach(com, True, fp8.man, ("instance_outputs",), programs=(fp8.pdir,))
    assert classes == ("instance_outputs", CB.FAMILY)
    src.install()
    mods = dict(model.named_modules())
    quant = specialization("Fp8GroupQuant_v1", {"K": fp8.H, "G": fp8.G})
    served: dict[tuple[int, str], np.ndarray] = {}
    for s, n in ((0, 2), (1, 1), (2, 1)):
        com._cur = types.SimpleNamespace(step=s)
        src.on_step_begin()
        served[(s, "gate_up")] = x = _fp8_rows(rng, n, fp8.H, s == 0)
        mods["model.layers.0.mlp.gate_up_proj"](_tensor(x, torch.bfloat16))
        mods["model.layers.0.mlp.down_proj"](_tensor(np.stack([_bf16(rng, fp8.I) for _ in range(n)]), torch.bfloat16))
        assert not com.errors, com.errors
        got = {name: CB.words(torch, t).reshape(t.shape) for st, name, t in com.acquired if st == s}
        op = f"model.layers.0.mlp.gate_up_proj/per_token_group_fp8_quant_{s}"
        ref = np.array([evaluate(quant, *[[int(v) for v in row]]) for row in x], dtype=np.uint64)
        assert got[f"{op}/0"].dtype == np.uint8 and np.array_equal(got[f"{op}/0"], ref[:, : fp8.H])
        assert got[f"{op}/1"].dtype == np.uint32 and np.array_equal(got[f"{op}/1"], ref[:, fp8.H:])
        assert np.array_equal(src.activation_scales("model.layers.0.mlp.gate_up_proj", s), ref[:, fp8.H:])
    assert not com.errors and len(com.acquired) == len(CB.manifest_sites(fp8.man)) == 12
    xs0 = src.activation_scales("model.layers.0.mlp.gate_up_proj", 2)
    assert xs0 is not None and src.activation_scales("model.layers.0.mlp.gate_up_proj", 0) is None     # only this step's values are held

    # the consuming GEMM (step 1): its replay reads the host words as committed inputs and checks the served output
    pi = IX.ProgramIndex(fp8.inst)
    row = next(r for r in pi.rows if r["spec"].startswith("ScaledMmFp8Block") and "gate_up" in r["name"] and pi.step_of(r["i"]) == 1)
    host = {name: CB.words(torch, t) for st, name, t in com.acquired if st == 1}
    op = "model.layers.0.mlp.gate_up_proj/per_token_group_fp8_quant_1"
    w_lo, w_hi = fp8.woff["model.model.layers.0.mlp.gate_up_proj.weight"]
    s_lo, s_hi = fp8.woff["model.model.layers.0.mlp.gate_up_proj.weight_scale_inv"]
    mm = specialization("ScaledMmFp8Block_v1", {"K": fp8.H, "N": 2 * fp8.I, "G": fp8.G})
    out = np.array(evaluate(mm, *[[int(v) for v in a] for a in (host[f"{op}/0"], host[f"{op}/1"], fp8.fields["model.model.layers.0.mlp.gate_up_proj.weight"],
                                                                 fp8.fields["model.model.layers.0.mlp.gate_up_proj.weight_scale_inv"])]), dtype=np.uint16)
    assert (w_hi - w_lo, s_hi - s_lo) == (fp8.fields["model.model.layers.0.mlp.gate_up_proj.weight"].size, 4)
    ident = {(r["op_path"], r["output_member"]): r for r in fp8.man["identities"] if r.get("engine_step") == 1}

    def replay(xq, xs, y):
        store, entries = {}, []
        for (o, m), w in (((op, "0"), xq), ((op, "1"), xs), (("model.layers.0.mlp.gate_up_proj", "0"), y)):
            r = ident[(o, m)]
            entries.append({"identity": {k: r[k] for k in ("program_digest", "request_id", "step", "invocation", "op_path", "output_member",
                                                            "element_range", "rank")}, "name": f"{o}/{m}", "element_offset": 0,
                            "encoding": {"dtype": r["dtype"]}, "family": r["family"]})
            store[(1, f"{o}/{m}")] = w.tobytes()
        cs = OP.CommittedStore({"entries": entries}, lambda st, name, lo, hi: (store.get((int(st), name)) or b"")[lo:hi] or None)
        read = lambda f, lo, hi: fp8.fields[f][lo:hi]  # noqa: E731
        return EV.replay_vu(pi, cs, read, {"i": row["i"], "request_id": "r0", "phase": "first-decode"}, arrive=0)

    rec = replay(host[f"{op}/0"], host[f"{op}/1"], out)
    assert rec["result"] is True, rec
    kinds = [(x["kind"], (x.get("identity") or [None, None, None])[2]) for x in rec["inputs"][:2]]
    assert kinds == [("committed", op), ("committed", op)]
    bad_xs = host[f"{op}/1"].copy()
    bad_xs[0] ^= 1 << 22
    assert replay(host[f"{op}/0"], bad_xs, out)["result"] is False                          # a wrong host scale: the served output disagrees
    bad_y = out.copy()
    bad_y[3] ^= 1
    assert replay(host[f"{op}/0"], host[f"{op}/1"], bad_y)["result"] is False               # a wrong served output: named by the replay


def test_identities_another_source_claims_are_left_to_it(dense, monkeypatch):
    """`CLAIMS`: a source that acquires some call boundaries itself (a tap of a pre-bias GEMM output the runtime holds) claims them; the
    host plan leaves them out, and a manifest whose every call boundary is claimed attaches nothing here."""
    ids = CB.manifest_sites(dense.man)
    logits = {f"{r['op_path']}/{r['output_member']}" for r in ids if r["op_path"].startswith("logits_processor")}
    assert logits
    monkeypatch.setattr(CB, "CLAIMS", [lambda r: "logits-tap" if r["op_path"].startswith("logits_processor") else None])
    left = CB.manifest_sites(dense.man)
    assert len(left) == len(ids) - sum(1 for r in ids if r["op_path"].startswith("logits_processor"))
    assert not {f"{r['op_path']}/{r['output_member']}" for r in left} & logits
    assert CB.plan_of(dense.man, dense.pdir).identities == len(left)
    monkeypatch.setattr(CB, "CLAIMS", [lambda r: "everything-tap"])
    com = Recorder(Probe())
    assert taps.attach(com, dense.man, ("instance_outputs",), programs=(dense.pdir,)) == ("instance_outputs",) and not com.sources
