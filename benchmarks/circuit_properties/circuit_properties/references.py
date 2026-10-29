"""Reference circuits: the declared Boolean circuits the properties are measured on, and the designation of their counted
products.

* `boolean_gemm`: `Gemm_v3{K,N,DOT=AmpereBF16TcDot16_v3}` (`verity.ml.circuits.ops.boolean.gemm`), one GEMM on bits.
* `layer`: one transformer layer of the served b1 registry (`verity.ml.definitions.registry.b1`) at T positions, on bits:
  σ(`LayerPre_v1`), the Boolean `Attention_v3` (`Attention_v6`), σ(`LayerPost_v1`), wired as `_forward_positions_v4` wires
  one layer (not layer 0: the fused add+RMSNorm with the residual). The GEMMs are σ(b1's `Gemm_v1`), `Gemm_v4`, whose dot
  steps are `AmpereBF16TcDot16_v3`: a counted product is one lane of such a step under a `Gemm` (the activation `a`, the
  weight `b`), and the outputs fan-in is read at are the GEMM coordinates (`GemmCoordinate`). Attention's dot steps are
  `AttnDotStep`s, activation by activation, and are not counted.

`CONFIGS` are the served workloads' configurations (b1's `B0_CONFIG`, `B1_CONFIG`, `SMOL360_CONFIG`) and two shapes with
no served workload, b1's Definitions bound at Qwen2.5-7B's and Llama-3.1-405B's published config.json shapes.
"""

from __future__ import annotations

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


def _config(base: dict, **kw) -> dict:
    out = dict(base)
    out.update(kw)
    return out


def configs() -> dict[str, tuple[dict, str]]:
    """name -> (b1 configuration, where its shape comes from)."""
    from verity.ml.definitions.registry import b1
    qwen7 = _config(b1.B1_CONFIG, model="Qwen/Qwen2.5-7B", H=3584, NL=28, NH=28, KVH=4, D=128, I=18944, V=152064,
                    QKV=3584 + 2 * 4 * 128, GU=2 * 18944)
    llama = _config(b1.B1_CONFIG, model="meta-llama/Llama-3.1-405B", H=16384, NL=126, NH=128, KVH=8, D=128, I=53248,
                    V=128256, QKV=16384 + 2 * 8 * 128, GU=2 * 53248, EPS=1e-5, BIAS=False)
    return {
        "smollm2-135m": (b1.B0_CONFIG, "served: b1.B0_CONFIG"),
        "smollm2-360m": (b1.SMOL360_CONFIG, "served: b1.SMOL360_CONFIG"),
        "qwen25-1b5": (b1.B1_CONFIG, "served: b1.B1_CONFIG"),
        "qwen25-7b": (qwen7, "constructed: b1's layer Definitions at Qwen2.5-7B's config.json shapes (no served workload)"),
        "llama3-405b": (llama, "constructed: b1's layer Definitions at Llama-3.1-405B's config.json shapes (no served workload)"),
    }


def on_bits(word: CompositeDefinition, version: int, **bindings):
    """σ(``word``) at ``bindings`` (`sigma_lift`'s body: each Call the Boolean version of its Definition), with the QKV bias
    add, `BiasAdd_v1`, which has no registered Boolean version, as σ(`BiasAdd_v1`) (its `Bf16Add`s on bits)."""
    from verity.ml.definitions import boolean as BB
    from verity.ml.definitions.registry import b1
    w = bind(word, **bindings)
    callee, missing = BB._sigma_callees(w)
    bias = BB.sigma_lift(b1.BiasAdd, 2, register=False)
    for node in w.body.nodes:
        if getattr(node.fn, "base_id", None) == "BiasAdd_v1":
            callee[id(node.fn)] = bind(bias, **node.fn.bindings)
    missing = {k: v for k, v in missing.items() if not k.startswith("BiasAdd_v1")}
    if missing:
        raise BB.NoBooleanVersion(missing)
    params, ret = w.signature()
    sig = (tuple((n, bit_type(t)) for n, t in params), bit_type(ret))
    defn = CompositeDefinition(word.name, version, tuple(word.statics), lambda S: sig,
                               BB._bit_replayer(w.body, [t for _, t in params], callee), register=False)
    return bind(defn, **bindings)


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
