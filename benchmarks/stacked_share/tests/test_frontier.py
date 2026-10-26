"""`frontier.py` without the registry: the composition of Llama-3.1-8B held to the int8 census's measured units (its callees'
units an instance as the census's Program has them, art:dc25549e), each model's layer kinds and weights against its published
size, and the table, its serving arithmetic and its markdown on synthetic atoms for every model."""
from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import frontier as F  # noqa: E402

SPEC = json.loads((HERE / "frontier_models.json").read_text())
S, P = SPEC["workload"]["context"], SPEC["workload"]["prompt"]


def census_atoms() -> dict:
    """The 8B Program's callees' units an instance (units-8b.json of art:dc25549e: each callee's units over its instances), and
    the attention's fitted forms the census's coefficients give (r0 = H const, b_q = H slope of the row; w0 = G const,
    b_kv = G slope of the window)."""
    eps = 1e-05
    return {
        F.RQ(4096): {"units": {"roundings": 16385, "compare_select": 12289}},
        F.RQ(14336): {"units": {"roundings": 57345, "compare_select": 43009}},
        F.MM(4096, 6144): {"units": {"roundings": 4 * 6144}},
        F.MM(4096, 4096): {"units": {"roundings": 4 * 4096}},
        F.MM(4096, 28672): {"units": {"roundings": 4 * 28672}},
        F.MM(14336, 4096): {"units": {"roundings": 4 * 4096}},
        F.NF(4096, eps): {"units": {"roundings": 21506}},
        F.NT(4096, eps): {"units": {"roundings": 18965}},
        F.ROPE(32, 128): {"units": {"roundings": 4096}},
        F.ROPE(8, 128): {"units": {"roundings": 1024}},
        F.SILU(14336): {"units": {"roundings": 14336}},
        F.EMB(128256, 4096) + F.EMB_DYN_TABLE: {"units": {"gathers": 4113}},
        F.TS(128256): {"units": {"roundings": 128255, "compare_select": 384765}},
        F.ROW(128): {"const": {"units": {"roundings": 1290, "compare_select": 383}},
                     "slope": {"units": {"roundings": 3, "compare_select": 2, "gathers": 8}}},
        F.KVQ(128): {"const": {"units": {"roundings": 256, "compare_select": 128}},
                     "slope": {"units": {"roundings": 896, "compare_select": 640}}},
    }


def test_the_8b_composition_is_the_census_programs():
    got = F.check_anchor(SPEC["models"], census_atoms(), SPEC["anchor"], S, P)
    assert got["held"] and got["token"] == {"gathers": 4113, "roundings": 10930965, "compare_select": 2556032}
    atoms = census_atoms()
    atoms[F.NF(4096, 1e-05)] = {"units": {"roundings": 21507}}
    with pytest.raises(AssertionError, match="token row"):
        F.check_anchor(SPEC["models"], atoms, SPEC["anchor"], S, P)
    atoms = census_atoms()
    atoms[F.KVQ(128)]["slope"]["units"]["roundings"] = 895
    with pytest.raises(AssertionError, match="coefficients"):
        F.check_anchor(SPEC["models"], atoms, SPEC["anchor"], S, P)


#: published total parameters (the model cards; MTP layers left out; GLM-5.3: its block-FP8 checkpoint's bytes, about one a
#: parameter, integrations/vllm/manifests/checkpoints.json) and the layers' kinds
PUBLISHED = {"Llama-3.1-8B": 8.03e9, "Llama-3.1-70B": 70.6e9, "Qwen3-0.6B": 0.6e9, "Qwen3-8B": 8.2e9,
             "DeepSeek-V3/R1/V3.1": 671e9, "DeepSeek-V3.2-Exp": 671e9, "Kimi-K2": 1.04e12, "Qwen3-235B-A22B": 235e9,
             "Qwen3-Next-80B-A3B": 80e9, "GLM-4.5/4.6": 355e9, "GLM-5.3": 755.7e9, "MiniMax-M1": 456e9, "MiniMax-M2": 230e9}


@pytest.mark.parametrize("name", list(PUBLISHED))
def test_weights_match_the_published_size(name):
    total = F.params(SPEC["models"][name])["total"]
    assert abs(total / PUBLISHED[name] - 1) < 0.03, f"{name}: {total / 1e9:.1f}B against {PUBLISHED[name] / 1e9:.1f}B"


def test_layer_kinds():
    kinds = {n: F.layer_kinds(m) for n, m in SPEC["models"].items()}
    assert kinds["DeepSeek-V3/R1/V3.1"] == [("full", "dense", 3), ("full", "moe", 58)]
    assert kinds["Kimi-K2"] == [("full", "dense", 1), ("full", "moe", 60)]
    assert kinds["Qwen3-Next-80B-A3B"] == [("full", "moe", 12), ("linear", "moe", 36)]
    assert kinds["MiniMax-M1"] == [("full", "moe", 10), ("linear", "moe", 70)]
    assert kinds["Llama-3.1-8B"] == [("full", "dense", 32)]
    assert kinds["GLM-5.3"] == [("full", "dense", 3), ("full", "moe", 75)]


def test_sparse_attention():
    """DSA: the softmax over the top-k keys, each selected latent row a gather of its words and its index bits, the indexer in
    its layers (all of V3.2's, 21 of GLM-5.3's 78: 0-2, then every 4th from 6) scoring every key, and its key in the pool."""
    ms = SPEC["models"]
    v32, glm = ms["DeepSeek-V3.2-Exp"], ms["GLM-5.3"]
    assert [F.indexer_layers(m) for m in (v32, glm, ms["DeepSeek-V3/R1/V3.1"])] == [61, 21, 0]

    def role(c, r):
        return sum(t.n for t in c["terms"] if t.role == r)

    c = F.compose(v32, "decode", "at write", S, P)
    assert {t.s for t in c["terms"] if t.role == "attention" and t.s is not None} == {2048}
    assert role(c, "DSA selection") == 61 * 2048 * (576 + S.bit_length())
    assert sum(t.n for t in c["terms"] if t.role == "DSA indexer" and t.atom == F.CS) == 61 * 64 * S
    assert role(c, "DSA top-k") == 61 * 2 * S
    assert c["dd"]["QK^T"] == 61 * 128 * 576 * 2048 and c["dd"]["indexer q k^T"] == 61 * 64 * 128 * S
    assert F.dynamic_bytes(c["dyn"], "at write", S)["per_token"] == 61 * 576 + 61 * (128 + 4)
    g = F.compose(glm, "decode", "at write", S, P)
    assert {d["kind"]: d["layers"] for d in g["dyn"]} == {"MLA latent c_KV and the RoPE key": 78, "DSA indexer key": 21}
    assert sum(t.n for t in g["terms"] if t.role == "DSA indexer" and t.atom == F.CS) == 21 * 32 * S
    k = F.dsa_keys(glm, "prefill", S, P)
    assert k["soft"] == Fraction(2048 * 2049 // 2 + (P - 2048) * 2048, P) and k["rows"] == Fraction((P - 2048) * 2048, P)
    assert F.dsa_keys(glm, "decode", 2048, P)["rows"] == 0


def synthetic(key: str) -> dict:
    """An atom's stand-in: a rounding a static's unit and a compare-select (a fitted one: a gather a key too)."""
    _fam, st = F.parse(key)
    n = 1 + sum(v for v in st.values() if isinstance(v, int))
    u = {"roundings": n, "compare_select": 1}
    prim = {"X_v1|roundings": n, "SelectI32_v1|compare_select": 1}
    if _fam in F.FITTED:
        return {"const": {"units": u, "primitives": prim},
                "slope": {"units": {"roundings": 3, "gathers": 8}, "primitives": {"F32Rint_v1|roundings": 3, "Exp2Lut127_v1|gathers": 8}}}
    return {"units": u, "primitives": prim}


def test_the_table_on_synthetic_atoms():
    atoms = {k: synthetic(k) for k in F.needed(SPEC["models"], S, P)}
    atoms.update(census_atoms())
    for k in census_atoms():
        a = atoms[k]
        for part in ([a] if "units" in a else [a["const"], a["slope"]]):
            part["primitives"] = {f"Y_v1|{c}": v for c, v in part["units"].items()}
    t = F.table(SPEC, atoms)
    assert set(t["models"]) == set(SPEC["models"])
    for name, r in t["models"].items():
        for kv in F.KV_MODES:
            sv = r[kv]["serving"]
            db = sv["request_bytes"]
            assert sv["B"] == int(sv["M_dyn"] // db["request"]) and sv["B"] > 0, name
            write = sv["tokens_per_s_decode"] * db["per_token"] + sv["B"] * db["rewritten"] / sv["t_step_s"]
            assert sv["T_turn_s"] == pytest.approx(sv["M_dyn"] / write)
            assert sv["R_cap"] == max(sv["R_decode"], sv["R_prefill"])
            E, L = F.E_WORD["proved"], F.LIFETIMES["1 h"]
            assert sv["f"]["1 h"]["proved"] == pytest.approx((128 * sv["R_cap"] * L / E + F.R_IN * L) / sv["M_dyn"])
            fl = r[kv]["softmax_floor"]
            assert fl["R_decode"] < sv["R_decode"] and fl["R_prefill"] < sv["R_prefill"]
    m = SPEC["models"]["Llama-3.1-8B"]
    floor = F.softmax_floor(atoms)
    gap = [F.evaluate(F.compose(m, "decode", "at write", S, P)["terms"], a, primitives=False)["total"] for a in (atoms, floor)]
    assert gap[0] - gap[1] == (13 - 1) * 32 * 32 * S
    ds = t["models"]["DeepSeek-V3/R1/V3.1"]["at write"]
    assert [d["kind"] for d in ds["decode"]["dyn"]] == ["MLA latent c_KV and the RoPE key"]
    assert ds["serving"]["request_bytes"]["per_token"] == 61 * 576
    qn = t["models"]["Qwen3-Next-80B-A3B"]["at write"]["serving"]["request_bytes"]
    assert qn["state"] == 36 * (32 * 128 * 128 * 4 + 3 * 8192 * 2) and qn["per_token"] == 12 * 2 * 2 * 256
    assert t["models"]["Llama-3.1-8B"]["as built"]["serving"]["request_bytes"]["per_token"] == 32 * 2 * 8 * 128 * 2
    md = F.markdown(t)
    assert "| DeepSeek-V3/R1/V3.1 |" in md and "**KV at write.**" in md and "**Where the units go, KV at write**" in md
    json.dumps(F.jsonable(t))


def test_modeled_terms_and_kinds():
    m = SPEC["models"]["Qwen3-Next-80B-A3B"]
    c = F.compose(m, "decode", "at write", S, P)
    state = sum(t.n for t in c["terms"] if t.role == "state update")
    assert state == 36 * (4 * 32 * 128 * 128 + 2 * 32 * 128)
    assert c["dd"]["state (kS, k err^T, S^T q)"] == 36 * 3 * 32 * 128 * 128
    pre = F.compose(m, "prefill", "at write", S, P)
    chunked = sum(t.n for t in pre["terms"] if t.role == "state update")
    assert chunked == 36 * 32 * (3 * 128 * 128 // 64 + 3 * 64 + 2 * 128 + 9 * 128) and state > 25 * chunked
    m1 = SPEC["models"]["MiniMax-M1"]
    blocked = sum(t.n for t in F.compose(m1, "prefill", "at write", S, P)["terms"] if t.role == "state update")
    assert blocked == 70 * 64 * (3 * 128 * 128 // 256 + 256 + 2 * 128 + 3 * 128)
    k = F.kinds({"MoE routing|F32GtStrict_v1|compare_select": Fraction(5), "attention|SelectI32_v1|compare_select": Fraction(2),
                 "requantization|F32Fmaxf_v1|compare_select": Fraction(3), "norm|F32Div_v1|roundings": Fraction(1)})
    assert k == {"compare_select": {"top-k (MoE routing, DSA)": 5, "softmax max": 2, "max, min, clamp": 3}, "roundings": {"division": 1}}
    k = F.kinds({"attention|LutIndex_v1|roundings": Fraction(1), "attention|S32LowByte_v1|roundings": Fraction(1),
                 "requantization|F32Rint_v1|roundings": Fraction(4)})
    assert k == {"roundings": {"softmax: max subtract, LUT index, low byte": 2, "requantization: round or convert": 4}}
    sh = F.shares({"attention": Fraction(5), "linear epilogue": Fraction(2), "requantization": Fraction(1), "norm": Fraction(3),
                   "MoE dispatch": Fraction(1)})
    assert sh["softmax attention"] == 5 and sh["requantization and linear epilogues"] == 3 and sh["other"] == 3
    assert sh["MoE routing and dispatch"] == 1 and sum(sh.values()) == 12
