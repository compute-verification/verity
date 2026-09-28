"""Stand-in request Programs of the call-boundary source's tests, torch-free: a Gemma-2-style dense Program (the ATen norm chains, one input
and x + residual, and the soft-capped logits) written as a Build dir, and the request manifest of its one request."""
from __future__ import annotations

import gzip
import json

from verity_vllm.correspondence.reader_for_query import Correspondence
from verity_vllm.query.program_view import from_instances
from verity_vllm.query.required import request_manifest

ONE, EPS, CAP = '{"f64":"0x1.0000000000000p+0"}', '{"f64":"0x1.0c6f7a0b5ed8dp-20"}', '{"f64":"0x1.e000000000000p+4"}'
V, H, LP, T = 16, 8, 7, 2
PF, IL, LG, EMB = "model.model.layers.0.post_feedforward_layernorm", "model.model.layers.1.input_layernorm", "model.logits_processor", "model.model.embed_tokens"


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


def dense_program(embed=EMB, lp=LP, t=T, eps=EPS, weight_add_once=False):
    """Per step (the prompt's `lp` rows, then one, `t` times): embed -> a post-feedforward norm chain (one input) -> an input-layernorm
    chain (x + residual, widened; `eps` its epsilon) -> the soft-capped logits of the last row -> TokenSelect; node-major rows, as the Build
    writes a prefill, each norm's `weight + 1` one Call of the step that every row reads (`weight_add_once`: one Call of step 0 that every later step reads
    too, as the Build writes Gemma-2's).  `embed` = the module the embedding rows are named under.  Returns (Rows, {(step, "e" | "x" | "y"): Calls})."""
    b, at = Rows(DENSE), {}
    sel, a0, b0 = None, None, None
    for s in range(1 + t):
        ks = range(lp) if s == 0 else range(1)

        def each(spec, node, *reads):
            return [b.add(s, spec, f"{node}[{k}]", *([["n", r[0][k], 0, r[1]]] if isinstance(r, tuple) else r for r in reads)) for k in ks]

        e = [b.add(s, f"Embedding_v1{{V={V},H={H}}}", f"{embed}/embedding_{s}[{k}]", [["p", 0, k, k + 1]] if s == 0 else [["n", sel, 0, 1]],
                   b.W(f"{EMB}.weight")) for k in ks]
        if not (weight_add_once and s):
            a0 = [["n", b.add(s, f"AddScalarBf16_v1{{N={H},C={ONE}}}", f"{PF}/add_{s}[0]", b.W(f"{PF}.weight")), 0, H]]
        sq = each(f"SquareBf16_v1{{N={H}}}", f"{PF}/pow_{s}", (e, H))
        mu = each(f"MeanTriton_v1{{N={H}}}", f"{PF}/mean_{s}", (sq, H))
        ve = each(f"AddScalarF32_v1{{N=1,C={EPS}}}", f"{PF}/add_1_{s}", (mu, 1))
        rs = each("RsqrtF32_v1{N=1}", f"{PF}/rsqrt_{s}", (ve, 1))
        sc = each(f"ScaleRowBf16_v1{{N={H}}}", f"{PF}/mul_{s}", (e, H), (rs, 1))
        mv = each(f"MulVecF32_v1{{N={H}}}", f"{PF}/mul_1_{s}", (sc, H), a0)
        x = each(f"NarrowF32ToBf16_v1{{N={H}}}", f"{PF}/to_{s}/out", (mv, H))
        if not (weight_add_once and s):
            b0 = [["n", b.add(s, f"AddScalarBf16_v1{{N={H},C={ONE}}}", f"{IL}/add_{s}[0]", b.W(f"{IL}.weight")), 0, H]]
        ad = each(f"AddWidenedBf16_v1{{N={H}}}", f"{IL}/add_1_{s}", (x, H), (e, H))
        sq = each(f"SquareF32_v1{{N={H}}}", f"{IL}/pow_{s}", (ad, H))
        mu = each(f"MeanTriton_v1{{N={H}}}", f"{IL}/mean_{s}", (sq, H))
        ve = each(f"AddScalarF32_v1{{N=1,C={eps}}}", f"{IL}/add_2_{s}", (mu, 1))
        rs = each("RsqrtF32_v1{N=1}", f"{IL}/rsqrt_{s}", (ve, 1))
        sc = each(f"ScaleRowF32_v1{{N={H}}}", f"{IL}/mul_{s}", (ad, H), (rs, 1))
        mv = each(f"MulVecF32_v1{{N={H}}}", f"{IL}/mul_1_{s}", (sc, H), b0)
        y = each(f"NarrowF32ToBf16_v1{{N={H}}}", f"{IL}/to_{s}/0", (mv, H))
        g = b.add(s, f"Gemm_v1{{K={H},N={V}}}", f"{LG}/triton_launch_{s}[0]", [["n", y[-1], 0, H]], b.W(f"{EMB}.weight"))
        dv = b.add(s, f"Bf16DivScalar_v1{{N={V},C={CAP}}}", f"{LG}/div_{s}[0]", [["n", g, 0, V]])
        th = b.add(s, f"Bf16Tanh_v1{{N={V}}}", f"{LG}/tanh_{s}[0]", [["n", dv, 0, V]])
        ml = b.add(s, f"Bf16MulScalar_v1{{N={V},C={CAP}}}", f"{LG}/mul_{s}/out[0]", [["n", th, 0, V]])
        sel = b.add(s, f"TokenSelect_v1{{V={V}}}", f"argmax_{s}[0]", [["n", ml, 0, V]])
        at.update({(s, "e"): e, (s, "x"): x, (s, "y"): y})
    return b, at


def build_dir(tmp_path, doc, observed, *, name="build_request", digest="d0", request_id="r0"):
    """A Build dir holding `doc`, and the manifest of its one request (the implementation paths observed = `observed`)."""
    res = {"program": {"digest": digest, "function": "f"}, "attention_impls_observed": {p: {} for p in observed},
           "target": {"declared": {"compute_capability": [8, 9]}}}
    d = tmp_path / name
    d.mkdir()
    with gzip.open(d / "instances.json.gz", "wt") as f:
        json.dump(doc, f)
    (d / "result.json").write_text(json.dumps(res))
    P = from_instances(str(d / "instances.json.gz"))
    return str(d), request_manifest(P, Correspondence(P, implementation_paths=observed), res, request_id=request_id).manifest
