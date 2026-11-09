"""Reference circuits: the declared Boolean circuits the properties are measured on, and the designation of their counted
products.

* `boolean_gemm`: `Gemm_v3{K,N,DOT=AmpereBF16TcDot16_v3}` (`verity.ml.circuits.ops.boolean.gemm`), one GEMM on bits.
* `layer`: one transformer layer of the served b1 registry (`verity.ml.definitions.registry.b1`) at T positions, on bits:
  σ(`LayerPre_v1`), the Boolean `Attention_v3` (`Attention_v6`), σ(`LayerPost_v1`), wired as `_forward_positions_v4` wires
  one layer (not layer 0: the fused add+RMSNorm with the residual). The GEMMs are σ(b1's `Gemm_v1`), `Gemm_v4`, whose dot
  steps are `AmpereBF16TcDot16_v3`: a counted product is one lane of such a step under a `Gemm` (the activation `a`, the
  weight `b`), and the outputs fan-in is read at are the GEMM coordinates (`GemmCoordinate`). Attention's dot steps are
  `AttnDotStep`s, activation by activation, and are not counted.

`configs` are the served workloads' configurations (b1's `B0_CONFIG`, `B1_CONFIG`, `SMOL360_CONFIG`) and two shapes with
no served workload, b1's Definitions bound at Qwen2.5-7B's and Llama-3.1-405B's shapes, which are benchmarks/rho's
(its `data/models.toml`, read through `rho.instances`, imported by path).
"""

from __future__ import annotations

import sys
from pathlib import Path

from verity.ml.circuits.boolean import bit_type
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import Coll, concat
from verity.ml.circuits.types import Array, Tuple

from .ops import Designation, Site

#: counted products: the lanes of each tensor-core dot step of a GEMM; fan-in at each GEMM coordinate
GEMM_PRODUCTS = Designation(sites=(Site(op=r"AmpereBF16TcDot16_v3", x="a", w="b", within=r"Gemm_v\d+"),),
                            entries=r"GemmCoordinate_v\d+")


def boolean_gemm(K: int, N: int) -> Program:
    from verity.ml.circuits.ops.boolean import gemm as BG
    return Program(bind(BG.Gemm, K=K, N=N, **BG.GEMM_AMPERE), registered=("w",), theta=("w",))


def rho_models() -> dict:
    """benchmarks/rho's models by name (`rho.instances.models`). Its directory goes last on the path: it holds a `run.py`
    and a `tests`, as this package's does, and must shadow neither."""
    rho = str(Path(__file__).resolve().parents[2] / "rho")
    if rho not in sys.path:
        sys.path.append(rho)
    from rho import instances
    return instances.models()


def shape(model) -> dict:
    """A rho model's shape in b1's configuration keys: a dense model with grouped-query attention and a SwiGLU MLP."""
    from rho import instances
    a = model.attention
    if a.get("kind") != "gqa" or set(model.matrices) != {"attention", "dense", "head"}:
        raise ValueError(f"{model.name}: b1's layer is a dense one with grouped-query attention")
    width = instances.width(model, "dense")
    return {"H": model.d_model, "NL": model.layers, "NH": a["heads"], "KVH": a["kv_heads"], "D": a["head_dim"],
            "I": width, "V": model.vocab, "QKV": (a["heads"] + 2 * a["kv_heads"]) * a["head_dim"], "GU": 2 * width,
            "tie_word_embeddings": model.tied_embeddings}


def configs() -> dict[str, tuple[dict, str]]:
    """name -> (b1 configuration, where its shape comes from)."""
    from verity.ml.definitions.registry import b1
    models = rho_models()
    qwen7 = b1.B1_CONFIG | shape(models["qwen25-7b"]) | {"model": "Qwen/Qwen2.5-7B"}
    llama = b1.B1_CONFIG | shape(models["llama3-405b"]) | {"model": "meta-llama/Llama-3.1-405B", "EPS": 1e-5,
                                                            "BIAS": False}
    return {
        "smollm2-135m": (b1.B0_CONFIG, "served: b1.B0_CONFIG"),
        "smollm2-360m": (b1.SMOL360_CONFIG, "served: b1.SMOL360_CONFIG"),
        "qwen25-1b5": (b1.B1_CONFIG, "served: b1.B1_CONFIG"),
        "qwen25-7b": (qwen7, "constructed: b1's layer Definitions at Qwen2.5-7B's shapes, rho's models.toml (no served workload)"),
        "llama3-405b": (llama, "constructed: b1's layer Definitions at Llama-3.1-405B's shapes, rho's models.toml (no served workload)"),
    }


def on_bits(word: CompositeDefinition, version: int, **bindings):
    """σ(``word``) at ``bindings`` (`boolean.lift_body`: each Call the Boolean version of its Definition), with the QKV bias
    add, `BiasAdd_v1`, which has no registered Boolean version, as σ(`BiasAdd_v1`) (its `Bf16Add`s on bits)."""
    from verity.ml.definitions import boolean as BB
    from verity.ml.definitions.registry import b1
    bias = BB.sigma_lift(b1.BiasAdd, 2, register=False)

    def override(fn):
        return bind(bias, **fn.bindings) if getattr(fn, "base_id", None) == "BiasAdd_v1" else None

    w = bind(word, **bindings)
    body = BB.lift_body(w, override)
    params, ret = w.signature()
    sig = (tuple((n, bit_type(t)) for n, t in params), bit_type(ret))
    return bind(CompositeDefinition(word.name, version, tuple(word.statics), lambda S: sig, body, register=False), **bindings)


def layer_fn(C: dict, T: int = 1):
    """One b1 layer (not layer 0) at T positions on bits, as an unregistered root Definition."""
    from verity.ml.definitions import boolean as BB
    from verity.ml.definitions.registry import b1
    BB.load()
    H, NH, KVH, D = C["H"], C["NH"], C["KVH"], C["D"]
    pre = on_bits(b1.LayerPre, 3, C=C, L0=False)
    post = on_bits(b1.LayerPost, 3, C=C)
    BN = b1.fa2_kblock_n(D)
    attn = [BB.boolean_version(bind(b1.AttentionV3, T=t + 1, NH=NH, KVH=KVH, D=D, BN=BN)) for t in range(T)]
    row = Array(T, Array(H, b1.BF16))
    params = (("h", bit_type(row)), ("res", bit_type(row)), ("w", bit_type(b1.layer_weights_type(C))),
              ("cs", bit_type(Array(T, Array(D, b1.BF16)))))
    ret = Tuple(bit_type(row), bit_type(row))
    kv = bit_type(b1.kv_type(C, T))

    def body(B, S, h, res, w, cs):
        ps = [B.call(pre, h[i], res[i], w, cs[i], name=f"t={i}/pre") for i in range(T)]
        kc = Coll(kv, concat([p[2].refs for p in ps]))
        vc = Coll(kv, concat([p[3].refs for p in ps]))
        outs = []
        for i in range(T):
            a = B.call(attn[i], ps[i][1], kc[0:i + 1], vc[0:i + 1], name=f"t={i}/attn")
            outs.append(B.call(post, a, ps[i][0], w, name=f"t={i}/post"))
        hs = Coll(bit_type(row), concat([o[0].refs for o in outs]))
        rs = Coll(bit_type(row), concat([o[1].refs for o in outs]))
        return Coll(ret, concat([hs.refs, rs.refs]))

    return bind(CompositeDefinition("CircuitPropertiesLayer", 1, ("C", "T"), lambda S: (params, ret), body, register=False),
                C=C, T=T)


def layer(C: dict, T: int = 1) -> Program:
    return Program(layer_fn(C, T), registered=("w", "cs"), theta=("w",))
