"""`verity_vllm.program.model.mla_weights_type` on a bf16 MLA-MoE configuration at toy sizes: the weights Record's
fields, their order and their types, written out by hand.  A bf16 checkpoint stores each linear as one bf16 matrix (no
e4m3 words, no block scales), its router gate in f32, and its routed experts as `w13 [E][2I][H]` and `w2 [E][H][I]`."""
from verity.circuits.types import Array, RecordType
from verity.definitions.registry.prims import BF16, F32

from verity_vllm.program import model

H, V, NL, FD, NH, QL, KV, DN, DR, DV, INTER, ID, E, S, MAX_POS = 4, 8, 3, 1, 2, 2, 2, 1, 2, 1, 2, 3, 2, 2, 4
CONFIG = {"H": H, "V": V, "NL": NL, "FIRST_DENSE": FD, "NH": NH, "Q_LORA": QL, "KV_LORA": KV, "QK_NOPE": DN, "QK_ROPE": DR,
          "V_HEAD": DV, "I": INTER, "I_DENSE": ID, "NUM_EXPERTS": E, "SHARED_EXPERTS": S, "MAX_POS": MAX_POS}


def _mat(n: int, k: int, t=BF16) -> Array:
    return Array(n, Array(k, t))


def test_a_bf16_configuration_keeps_its_checkpoints_layout():
    attn = [("ln1", Array(H, BF16)), ("qkv_a_w", _mat(QL + KV + DR, H)), ("q_a_ln", Array(QL, BF16)),
            ("q_b_w", _mat(NH * (DN + DR), QL)), ("kv_a_ln", Array(KV, BF16)), ("kv_b_w", _mat(NH * (DN + DV), KV)),
            ("o_w", _mat(H, NH * DV)), ("ln2", Array(H, BF16))]
    dense = RecordType(*attn, ("gate_up_w", _mat(2 * ID, H)), ("down_w", _mat(H, ID)))
    moe = RecordType(*attn, ("gate_w", _mat(E, H, F32)), ("gate_bias", Array(E, F32)),
                     ("sh_gate_up_w", _mat(2 * INTER * S, H)), ("sh_down_w", _mat(H, INTER * S)),
                     ("w13", Array(E, _mat(2 * INTER, H))), ("w2", Array(E, _mat(H, INTER))))
    want = RecordType(("embed", _mat(V, H)), ("dense_layers", Array(FD, dense)), ("layers", Array(NL - FD, moe)),
                      ("norm", Array(H, BF16)), ("cos_sin", _mat(MAX_POS, DR)), ("lm_head", _mat(V, H)))
    assert model.mla_layer_weights_type(CONFIG, dense=True) == dense
    assert model.mla_layer_weights_type(CONFIG, dense=False) == moe
    assert model.mla_weights_type(CONFIG) == want
    assert model.weights_type(CONFIG) == want                    # a configuration with KV_LORA takes this layout
