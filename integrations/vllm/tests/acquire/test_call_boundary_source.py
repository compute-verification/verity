"""The call-boundary source (`acquire.sources.call_boundary_source`) on CPU tensors with a recording committer, on request Programs written
as Build dirs: a Gemma-2-style dense Program (the ATen norm chains, one input and x + residual, and the soft-capped logits) and the
block-scaled FP8 linear (`Fp8GroupQuant_v1 -> ScaledMmFp8Block_v1`).  Every `call_boundaries` identity of the manifest is committed once
per step over its element range with the words a row-by-row `verity.ml.circuits.evaluation.evaluate` of the Program gives -- on edge words too (NaN
payloads, +-inf, +-0, subnormals, the largest finite); a manifest naming none attaches nothing and a forced attach refuses; a body the
rule cannot evaluate is refused by name; an argument of another shape is a collector error naming the site; and the consuming GEMM's
replay reads the committed words and checks the served output as before."""
from __future__ import annotations

import json
import os
import types

import numpy as np
import pytest

from tests.acquire.call_boundary_programs import DENSE, EMB, EPS, H, IL, LP, PF, T, V, build_dir, dense_program
from verity.ml.circuits.evaluation import evaluate, evaluate_batch
from verity_vllm.commit.binding import build_binding_map, logical_request
from verity_vllm.query.compose import compose_global
from verity_vllm.query.call_scope import specialization
from verity_vllm.query.manifest.format import spec_family
from verity_vllm.query.program_view import definition_ports, spec_statics

torch = pytest.importorskip("torch")

from verity_vllm.acquire.sources import call_boundary_plan as CP  # noqa: E402
from verity_vllm.acquire.sources import call_boundary_source as CB  # noqa: E402
from verity_vllm.acquire.sources import taps  # noqa: E402

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


def dense_fields():
    rng = np.random.default_rng(11)
    table = _bf16(rng, V * H).reshape(V, H)
    table[: len(EDGE)] = np.asarray(EDGE, dtype=np.uint16)
    return {f"{EMB}.weight": table.reshape(-1), f"{PF}.weight": _bf16(rng, H, 0.5), f"{IL}.weight": _bf16(rng, H, 0.5)}


def reference(doc, prompt, fields, field_order):
    """Every Call's output leaves, row by row through `verity.ml.circuits.evaluation.evaluate` (the IR's plain-integer reference)."""
    weights = [int(w) for n in field_order for w in fields[n]]
    vals: dict[int, list[int]] = {}
    for r in doc["rows"]:
        args = [[v for kind, a, lo, hi in g for v in (prompt[lo:hi] if kind == "p" and a == 0 else weights[lo:hi] if kind == "p" else vals[a][lo:hi])]
                for g in r["args"]]
        vals[r["i"]] = list(evaluate(specialization(spec_family(r["spec"]), spec_statics(r["spec"])), *args))
    return vals


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
        src.on_step_end(com._cur)
    com._cur = None


def _dense_source(world, man=None, root=None, plan=None):
    model = runtime(["model.embed_tokens", PF[len("model."):], IL[len("model."):], "logits_processor"],
                    {n[len("model."):]: _tensor(w, torch.bfloat16) for n, w in world.fields.items()})
    com = Recorder(model)
    classes = taps.attach(com, man or world.man, ("instance_outputs",), taps.TapSources(),
                          programs=(None, root or os.path.dirname(world.pdir)), call_boundary_plan=plan)  # under the root
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
    assert all(src.targets_at(s) == {n: 1 for e, n in got if e == s} for s in range(1 + T)) and src.target_names() == {n for _e, n in got}
    for r in ids:
        t = got[(r["engine_step"], f"{r['op_path']}/{r['output_member']}")]
        assert t.dtype == TORCH_OF[r["dtype"]] and tuple(t.shape) == (r["rows"], r["width"]) and t.numel() == r["element_range"][1] - r["element_range"][0]
        w = CB.words(torch, t).reshape(t.shape).astype(np.uint64)
        assert np.array_equal(w, ir_words(dense.doc, dense.vals, dense.b.step, r)), r["op_path"]
    body = src.values("model.layers.0.post_feedforward_layernorm", T)                       # the last step's words; earlier steps are released
    assert f"model.layers.0.post_feedforward_layernorm/rsqrt_{T}/out" in body and src.values("model.layers.0.post_feedforward_layernorm", 0) == {}
    assert src.stats["bodies"] == 3 * (1 + T) and src.stats["tensors"] == len(ids) and src.stats["errors"] == 0
    assert src.declaration_classes()[0]["class"] == CB.FAMILY and src.extra_step_bytes({}, None)["tensors_per_step"] == 18


def test_a_weight_add_made_once_is_evaluated_into_every_steps_body(tmp_path):
    """Gemma-2's Build makes each norm's `weight + 1` once, at step 0: a later step's body reads it across the step and evaluates it from
    the weights, committing the reference's words."""
    b, at = dense_program(weight_add_once=True)
    doc = b.doc(LP)
    assert sum(r["spec"].startswith("AddScalarBf16_v1") for r in doc["rows"]) == 2
    fields = dense_fields()
    world = types.SimpleNamespace(b=b, at=at, doc=doc, fields=fields, vals=reference(doc, PROMPT, fields, [n for n, *_ in DENSE]))
    world.pdir, world.man = build_dir(tmp_path, doc, [PF[len("model."):]])
    ids = CB.manifest_sites(world.man)
    plan = CB.plan_of(world.man, world.pdir)
    assert not plan.uncovered and plan.covered == plan.identities == len(ids) == 18 + T * 16
    com, src = _dense_source(world)
    _drive(world, com, src)
    assert not com.errors and len(com.acquired) == len(ids)
    got = {(s, n): t for s, n, t in com.acquired}
    for r in ids:
        w = CB.words(torch, got[(r["engine_step"], f"{r['op_path']}/{r['output_member']}")]).reshape(r["rows"], r["width"]).astype(np.uint64)
        assert np.array_equal(w, ir_words(doc, world.vals, b.step, r)), (r["engine_step"], r["op_path"])


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
    assert taps.attach(com, plain, ("instance_outputs",), taps.TapSources(), programs=(dense.pdir,)) == ("instance_outputs",) and not com.sources
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
        assert src.activation_scales("model.layers.0.mlp.gate_up_proj", s) is not None      # x_s during the forward, before the step ends
        src.on_step_end(com._cur)
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
    assert taps.attach(com, dense.man, ("instance_outputs",), taps.TapSources(), programs=(dense.pdir,)) == ("instance_outputs",) and not com.sources


def test_a_definition_past_the_reference_bound_is_its_row_kernels_alone(dense, monkeypatch):
    """Past `REFERENCE_LEAVES` argument words per instance (lowered here below the logits Gemm's 8 + 128) a Definition is evaluated by its
    row kernel alone, to the reference's words; an instance the kernel declines (this K = 8 Gemm: its k16 chain needs K % 16 == 0) is a
    collector error naming it and nothing of the body is committed, never the reference; and one no row kernel claims is uncovered."""
    gemm = specialization("Gemm_v1", {"K": 32, "N": 3})
    rng = np.random.default_rng(5)
    x, w = _bf16(rng, 2 * 32).reshape(2, 32), np.broadcast_to(_bf16(rng, 3 * 32), (2, 3 * 32))
    assert np.array_equal(CP._kernel_rows(gemm, "Gemm_v1{K=32,N=3}", [x, w]), evaluate_batch(gemm, x, w, kernel="reference"))
    monkeypatch.setattr(CP, "REFERENCE_LEAVES", 100)
    com, src = _dense_source(dense)
    assert src.plan.kernel_only == {f"Gemm_v1{{K={H},N={V}}}"}
    _drive(dense, com, src)
    assert len(com.errors) == 1 + T and all(f"the row kernel of Gemm_v1{{K={H},N={V}}} declines an instance" in e for e in com.errors)
    assert com.acquired and not any(n.startswith("logits_processor") for _s, n, _t in com.acquired)
    monkeypatch.setattr(CP, "REFERENCE_LEAVES", 12)
    from verity_vllm.program.kernels import kernel_registry
    monkeypatch.delitem(kernel_registry._REGISTERED, ("Bf16DivScalar_v1", "rows"))
    why = " ".join(CB.plan_of(dense.man, dense.pdir).uncovered)
    assert "Bf16DivScalar_v1" in why and "past the reference's bound, and no rows kernel registered" in why


# ------------------------------------------------------------------------------------------------------------------------------ B = 2
EPS_R1 = '{"f64":"0x1.0c6f7a0b5ed8dp-17"}'


def _global(parts, arrive):
    reqs = [{"request_id": r, "workload_request_id": f"w-{r}", "index": i, "prompt_len": lp, "max_tokens": n, "arrive_step": arrive[r]}
            for i, (r, lp, n) in enumerate((("r0", 7, 3), ("r1", 4, 2)))]
    return compose_global({}, parts, sorted(reqs, key=lambda r: (r["arrive_step"], r["index"])), workload_digest="w" * 64)


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    """Two requests of one model under one programs root, composed by `compose_global`: r0 (LP 7, three steps) and r1 (LP 4, two steps,
    another input-layernorm epsilon, so its template of that body differs and it is evaluated per request), both arriving at step 0 --
    co-scheduled at engine steps 0 and 1, r0 alone at 2."""
    root = tmp_path_factory.mktemp("pair")
    fields = dense_fields()
    worlds, parts = {}, {}
    for rid, lp, t, eps, prompt in (("r0", LP, T, EPS, PROMPT), ("r1", 4, 1, EPS_R1, [9, 1, 3, 0])):
        b, at = dense_program(lp=lp, t=t, eps=eps)
        doc = b.doc(lp)
        _d, parts[rid] = build_dir(root, doc, [PF[len("model."):]], name=f"build_{rid}", digest=f"d-{rid}", request_id=rid)
        worlds[rid] = types.SimpleNamespace(b=b, at=at, doc=doc, vals=reference(doc, prompt, fields, [n for n, *_ in DENSE]), steps=1 + t)
    return types.SimpleNamespace(root=str(root), fields=fields, worlds=worlds, parts=parts, man=_global(parts, {"r0": 0, "r1": 0}))


def _drive_pair(pair, com, src, order, rows_of=lambda e, rows: rows):
    """Each engine step's module forwards on the rows of its requests packed in the runtime's `order`, and `step_rows` as the committer
    records it (engine ids suffixed as vLLM v1 does), through `rows_of(e, rows)`."""
    mods = dict(src.model.named_modules())
    com.step_rows = {}
    for e in range(3):
        live = [r for r in order if e < pair.worlds[r].steps]

        def pack(key, last=False):
            cs = [(r, c) for r in live for c in pair.worlds[r].at[(e, key)][-1 if last else 0:]]
            return _tensor(np.array([pair.worlds[r].vals[c][:H] for r, c in cs], dtype=np.uint16), torch.bfloat16)

        com._cur = types.SimpleNamespace(step=e)
        src.on_step_begin()
        mods[PF[len("model."):]](pack("e"))
        mods[IL[len("model."):]](pack("x"), residual=pack("e"))
        mods["logits_processor"](mods["model.embed_tokens"], pack("y", last=True))
        rows, lo = [], 0
        for r in live:
            rows.append((f"w-{r}-0badc0de", lo, lo + len(pair.worlds[r].at[(e, "e")])))
            lo = rows[-1][2]
        com.step_rows[e] = rows_of(e, rows)
        src.on_step_end(com._cur)
    com._cur = None


def test_a_global_manifest_commits_every_requests_rows_packed_as_the_binding_map_splits_them(pair):
    """The runtime runs r1's rows first: every target is one packed tensor per engine step, and the binding map over the committed
    layouts and `step_rows` binds each request's identity to its own rows (a step-shared target's one row to each) -- the IR's words of
    that request's Program."""
    ids = CB.manifest_sites(pair.man)
    com, src = _dense_source(pair, pair.man, pair.root)
    assert src.plan.covered == src.plan.identities == len(ids) == 2 * (7 + 8 + 3) + 3 * (7 + 8 + 3)
    bodies = src.plan.bodies
    assert not bodies[(0, IL[len("model."):])].packed and bodies[(0, PF[len("model."):])].packed and bodies[(1, "logits_processor")].packed
    assert [(q.request, q.rows) for q in bodies[(0, PF[len("model."):])].parts] == [("r0", LP), ("r1", 4)]
    _drive_pair(pair, com, src, ["r1", "r0"])
    assert not com.errors, com.errors
    got = {(s, n): t for s, n, t in com.acquired}
    assert len(got) == len(com.acquired) == 3 * (7 + 8 + 3)
    shared = [t.name for t in bodies[(0, PF[len("model."):])].targets if t.shared]         # `weight + 1`: one row, bound whole to each request
    assert shared == [f"{PF[len('model.'):]}/add_0/out"] and tuple(got[(0, shared[0])].shape) == (1, H)
    layouts = {e: [types.SimpleNamespace(name=n, shape=list(t.shape), stream_off=0, first_leaf=0, chunk=256, nbytes=t.numel() * t.element_size(),
                                         dtype=str(t.dtype), ordinal=i) for i, ((s, n), t) in enumerate(sorted(got.items())) if s == e] for e in range(3)}
    order = {"w-r0": "r0", "w-r1": "r1"}
    bmap = build_binding_map(layouts, [types.SimpleNamespace(root=bytes(32))] * 3, bytes(32), "w" * 64,
                             request_rows={e: [(logical_request(rid, order), lo, hi) for rid, lo, hi in rs] for e, rs in com.step_rows.items()})
    want = {(r["request_id"], r["engine_step"], r["op_path"], r["output_member"]): r for r in ids}
    seen = set()
    for x in bmap["entries"]:
        i = x["identity"]
        key = (i["request_id"], int(i["step"]), i["op_path"], i["output_member"])
        r, w = want[key], pair.worlds[key[0]]
        assert i["element_range"] == r["element_range"]
        flat = CB.words(torch, got[(key[1], x["name"])]).astype(np.uint64)
        lo, hi = (x["element_offset"] + v for v in r["element_range"])
        assert np.array_equal(flat[lo:hi].reshape(r["rows"], r["width"]), ir_words(w.doc, w.vals, w.b.step, r)), key
        seen.add(key)
    assert seen == set(want)


def test_a_packing_that_disagrees_with_the_declared_composition_is_a_collector_error_naming_the_step(pair):
    def wrong(e, rows):
        if e == 0:
            return [(rows[0][0], 0, 5), (rows[1][0], 5, 11)]       # r0 five token rows and r1 six, where they declare seven and four
        if e == 1:
            return [rows[0], (rows[0][0], 1, 2)]                  # r0's id twice, r1 absent
        return None                                               # no row order recorded

    com, src = _dense_source(pair, pair.man, pair.root)
    _drive_pair(pair, com, src, ["r0", "r1"], wrong)
    assert not com.acquired and len(com.errors) == 3, com.errors
    assert "engine step 0 ran (request, rows) [('r0', 5), ('r1', 6)]" in com.errors[0] and "declares [('r0', 7), ('r1', 4)]" in com.errors[0]
    assert "engine step 1 ran (request, rows) [('r0', 1), ('r0', 1)]" in com.errors[1]
    assert "no row order recorded at engine step 2" in com.errors[2]


def test_requests_at_other_local_steps_of_one_engine_step_are_refused_by_name(pair):
    """r1 arriving a step later: at engine steps 1 and 2 the requests' bodies are fx nodes of other local steps, which one packed tensor
    per name cannot bind; step 0 (r0 alone) is covered."""
    plan = CB.plan_of(_global(pair.parts, {"r0": 0, "r1": 1}), pair.root)
    why = " ".join(plan.uncovered)
    assert plan.uncovered and "@ engine step 0" not in why and "@ engine step 1" in why and "commit other targets" in why
    assert {e for e, _m in plan.bodies} == {0}


def test_a_warm_up_requests_rows_are_its_workload_requests_and_commit_the_same_layout(pair):
    """Gemma-2-2B B8 (m006, r20260930-190524-cd72): the bounded warm-up runs `warmup-<i>` stand-ins, which no workload id resolves, so
    every warm-up step refused ('engine step 0 ran (request, rows) [('warmup-0-aec3dda0', 11), ...]'), the learned plans held no
    call-boundary tensor, and record step 0 ran 7,392,522,240 bytes against a 4,134,862,080-byte plan.  Warm-up request i is workload
    request i's stand-in (same prompt length and schedule): it takes that request's rows and the step commits the same names and shapes."""
    idx = {"r0": 0, "r1": 1}
    com, src = _dense_source(pair, pair.man, pair.root)
    _drive_pair(pair, com, src, ["r1", "r0"])
    ref = [(s, n, tuple(t.shape), t.dtype) for s, n, t in com.acquired]
    warm, wsrc = _dense_source(pair, pair.man, pair.root)
    _drive_pair(pair, warm, wsrc, ["r1", "r0"],
                lambda e, rows: [(f"warmup-{idx[rid.split('-')[1]]}" + ("-aec3dda0" if e else ""), lo, hi) for rid, lo, hi in rows])
    assert not warm.errors, warm.errors
    assert [(s, n, tuple(t.shape), t.dtype) for s, n, t in warm.acquired] == ref
    stray, ssrc = _dense_source(pair, pair.man, pair.root)
    _drive_pair(pair, stray, ssrc, ["r1", "r0"], lambda e, rows: [("warmup-7-aec3dda0", *rows[0][1:]), *rows[1:]])
    assert not stray.acquired and "('warmup-7-aec3dda0', 4)" in stray.errors[0]


def test_the_warm_up_commits_zero_words_of_the_records_layout_and_evaluates_nothing(pair, monkeypatch):
    """The Commit's warm-up instrumented run is never a record (`taps.warmup`): every target, packed, per request or step-shared, is zero
    words of the name, shape and dtype the record commits, and no template is evaluated.  Off again, the same source commits the IR's
    words, and a bounded committer's `learn_only` follows the same switch."""
    com, src = _dense_source(pair, pair.man, pair.root)
    _drive_pair(pair, com, src, ["r1", "r0"])
    ref = [(s, n, tuple(t.shape), t.dtype, CB.words(torch, t).tobytes()) for s, n, t in com.acquired]
    calls, real = [], CP.evaluate_template
    monkeypatch.setattr(CP, "evaluate_template", lambda *a, **k: calls.append(a[2]) or real(*a, **k))
    warm, wsrc = _dense_source(pair, pair.man, pair.root)
    taps.warmup(warm, True)
    _drive_pair(pair, warm, wsrc, ["r1", "r0"], lambda e, rows: [(f"warmup-{rid[3]}", lo, hi) for rid, lo, hi in rows])
    assert not warm.errors and not calls, warm.errors
    assert [(s, n, tuple(t.shape), t.dtype) for s, n, t in warm.acquired] == [x[:4] for x in ref]
    assert not any(CB.words(torch, t).any() for _s, _n, t in warm.acquired)
    assert wsrc.stats["placeholder_bodies"] == wsrc.stats["bodies"] == len(wsrc.plan.bodies)
    taps.warmup(warm, False)
    warm.acquired.clear()
    _drive_pair(pair, warm, wsrc, ["r1", "r0"])
    assert calls and [(s, n, tuple(t.shape), t.dtype, CB.words(torch, t).tobytes()) for s, n, t in warm.acquired] == ref
    bounded = types.SimpleNamespace(bounded=True, learn_only=False, sources=[wsrc])
    taps.warmup(bounded, True)
    assert bounded.learn_only and wsrc.placeholder
    taps.warmup(bounded, False)
    assert not bounded.learn_only and not wsrc.placeholder


# ------------------------------------------------------------------------------------------------------------------------- the record
def _read_back(plan):
    return CP.of_json(json.loads(json.dumps(CP.to_json(plan))))


def _same_plan(a, b):
    """Every field of two plans equal: the templates with their operands' fx names (which template equality leaves out), the literals'
    words and each Definition with its arity."""
    assert list(a.bodies) == list(b.bodies) and a.bodies == b.bodies
    assert [[o.where for q in x.parts for o in q.template.operands] for x in a.bodies.values()] == \
           [[o.where for q in x.parts for o in q.template.operands] for x in b.bodies.values()]
    assert a.literals.keys() == b.literals.keys() and all(np.array_equal(a.literals[s], b.literals[s]) and b.literals[s].dtype == np.uint64
                                                          for s in a.literals)
    assert {s: (d, n) for s, (d, n) in a.definitions.items()} == b.definitions
    for f in ("program_dirs", "fields", "kernel_only", "composition", "order", "by_index", "identities", "uncovered", "bytes_per_token",
              "tensors_per_step"):
        assert getattr(a, f) == getattr(b, f), f


def test_a_plan_read_back_from_its_record_is_the_plan_and_commits_the_same_words(dense, monkeypatch):
    """The Commit's call-boundary plan (`pipeline.boundary_plan`) is `to_json` of the Build's: read back (`of_json`), every field is
    the derived plan's and the source attached with it commits the same words as one that derives its own; a kernel-only Definition
    stays kernel-only."""
    plan = CB.plan_of(dense.man, os.path.dirname(dense.pdir))
    back = _read_back(plan)
    _same_plan(plan, back)
    runs = []
    for given in (None, back):
        com, src = _dense_source(dense, plan=given)
        assert src.plan is given or given is None
        _drive(dense, com, src)
        assert not com.errors
        runs.append([(s, n, CB.words(torch, t).tobytes()) for s, n, t in com.acquired])
    assert runs[0] == runs[1] and len(runs[0]) == len(CB.manifest_sites(dense.man))
    monkeypatch.setattr(CP, "REFERENCE_LEAVES", 100)
    kernel = CB.plan_of(dense.man, dense.pdir)
    assert kernel.kernel_only == _read_back(kernel).kernel_only == {f"Gemm_v1{{K={H},N={V}}}"}


def test_a_global_plan_read_back_keeps_each_requests_template_and_the_composition(pair):
    """Two requests, one body evaluated per request (r1's other epsilon): the parts, their templates, the composition and the runtime's
    request names survive the record, and the binding map's split of the words committed with it is the derived plan's."""
    plan = CB.plan_of(pair.man, pair.root)
    back = _read_back(plan)
    _same_plan(plan, back)
    assert not back.bodies[(0, IL[len("model."):])].packed and back.composition == plan.composition and back.by_index == {0: "r0", 1: "r1"}
    runs = []
    for given in (None, back):
        com, src = _dense_source(pair, pair.man, pair.root, plan=given)
        _drive_pair(pair, com, src, ["r1", "r0"])
        assert not com.errors, com.errors
        runs.append([(s, n, CB.words(torch, t).tobytes()) for s, n, t in com.acquired])
    assert runs[0] == runs[1] and runs[0]

