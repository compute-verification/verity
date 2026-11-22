"""A credited block of the served circuit at one of its linears: Pearl+'s tile (`pearl_plus_credited`) in the place of
one `Gemm_v2` unit of the served circuit of WIDE (TINY at hidden size 256) at class 1,1,1, layer 0's qkv projection,
columns 0 to 3 (unit 0 of `Gemm_v2{K=256,N=4}`). TINY's own linears (K = 64 and 128) are below the tile's least depth:
its inert-work bound needs two whole promotion windows of 128 (`inert_bound._promoted`), K >= 256.

Its B rows are the registered row that unit reads, ``weights/model.layers.0.self_attn.qkv_proj.weight@1024`` row 0 under
the committer's registration (`window_weights.weights_registration`), and its noise is seeded by that parameter's
registered root: no second commitment of the weight and no test that two commitments agree. Its x is the served
circuit's own value, the first norm's output (`RMSNormTriton_v5`) on token 7's embedding row, which the tile would read
by wire. Its y is the tile's own output, the one y the circuit would hold, read by wire by the linear's readers.

The tile is held here to its reference (`pearl_plus_tile.reference`, the scheme's semantics the circuit template's tests
hold it to). `PearlPlusTileCoreWords_v5` reads A in FP32 words, which no served writer writes; `_v6` reads the writer's
BF16 row, and its circuit template on bits is the slow test at the end.
"""
from __future__ import annotations

import random

import numpy as np
import pytest
from verity.circuits.evaluation import evaluate, evaluate_batch
from verity.commitments import salt_file as SF
from verity.protocols.accounting.compute.pouw.prover import schemes as pouw_schemes
from verity.protocols.accounting.compute.pouw.prover.circuit import inert_bound
from verity.protocols.accounting.compute.pouw.prover.circuit import pearl_plus_credited as PC
from verity.protocols.accounting.compute.pouw.prover.circuit import pearl_plus_tile as PT
from verity.protocols.accounting.compute.pouw.prover.schemes import pearl_c as C

from tests.program.predict import test_steps as S
from tests.program.predict import test_window_circuit as TW
from verity_vllm.check import weights_of_record as WR
from verity_vllm.program.predict import window_builder as WB
from verity_vllm.program.predict import window_weights as WW
from verity_vllm.program.predict.window_circuit import WindowClass, class_circuit

SCHEME = "pearl-c-h100-v1-h3"
QKV = "model.layers.0.self_attn.qkv_proj.weight"
K, N, TN = 256, 512, 4
PARAM = WB.weight_param(QKV, TN * K)
TOKEN, UNIT = 7, 0


def _checkpoint(tmp_path) -> tuple[str, str]:
    """WIDE's checkpoint, as `test_window_circuit._checkpoint` writes TINY's: random BF16 weights of its shapes in one
    safetensors shard of its HF names, and a manifest pinning it at `test_steps._registered`'s model and revision."""
    import json
    from tests.check.test_weights_of_record import _bf16, write_safetensors
    T = S.WIDE
    rng = np.random.default_rng(3)
    H, INTER, NH, KVH, V = T["hidden_size"], T["intermediate_size"], T["num_attention_heads"], T["num_key_value_heads"], T["vocab_size"]
    D = H // NH
    w = {"model.embed_tokens.weight": _bf16(rng, V, H), "model.norm.weight": _bf16(rng, H)}
    for i in range(T["num_hidden_layers"]):
        p = f"model.layers.{i}."
        w |= {p + "self_attn.q_proj.weight": _bf16(rng, NH * D, H), p + "self_attn.k_proj.weight": _bf16(rng, KVH * D, H),
              p + "self_attn.v_proj.weight": _bf16(rng, KVH * D, H), p + "self_attn.o_proj.weight": _bf16(rng, H, NH * D),
              p + "mlp.gate_proj.weight": _bf16(rng, INTER, H), p + "mlp.up_proj.weight": _bf16(rng, INTER, H),
              p + "mlp.down_proj.weight": _bf16(rng, H, INTER), p + "input_layernorm.weight": _bf16(rng, H),
              p + "post_attention_layernorm.weight": _bf16(rng, H)}
    d = tmp_path / "snap"
    d.mkdir(parents=True, exist_ok=True)
    write_safetensors(str(d / "model.safetensors"), {n: ("BF16", a) for n, a in w.items()})
    (d / "config.json").write_text(json.dumps({**T, "rope_theta": 10000.0}))
    man = tmp_path / "checkpoints.json"
    model = S._registered(T).phi.model
    man.write_text(json.dumps({"checkpoints": [{"repo": model["repo"], "revision": model["revision"], "local_path": str(d),
                                                "files": {"model.safetensors": {"sha256": WR.sha256_file(str(d / "model.safetensors"))}}}]}))
    return str(d), str(man)


def _unit(circuit, prefix: str):
    (fn,) = [fn for fn, _n in circuit.templates if fn.id.startswith(prefix)]
    return fn


@pytest.fixture(scope="module")
def linear(tmp_path_factory):
    """WIDE's class 1,1,1 circuit at the test grain, the committer's registration of WIDE's checkpoint, and the values of
    layer 0's qkv unit 0: x (the first norm on token 7's embedding row), its registered row, the served unit's y, and
    the tile's operands, certificates and public row."""
    tmp = tmp_path_factory.mktemp("served-linear")
    reg = S._registered(S.WIDE)
    circuit = class_circuit(reg, WindowClass(1, 1, 1), None, TW.GRAIN)
    d, man = _checkpoint(tmp)
    registration, rpath = TW._register(tmp / "committer", reg, d, checkpoints_manifest=man, grain=TW.GRAIN)
    pins = WW.registration_pins(registration, WW.weights_record(reg, d, man, grain=TW.GRAIN))
    values = WW.weight_values(reg, WW.record_words(reg, d, man), rpath, TW.GRAIN, keep_rows=True)
    ck = WR.CheckpointIndex(d, checkpoints_manifest=man, model="org/m", revision="r")

    emb = ck.array("model.embed_tokens.weight").view(np.uint16)[TOKEN].tolist()
    norm_w = ck.array("model.layers.0.input_layernorm.weight").view(np.uint16).tolist()
    x = [int(v) for v in evaluate(_unit(circuit, "RMSNormTriton_v5"), emb, norm_w)]
    gemm = _unit(circuit, "Gemm_v2{K=256,N=4,")
    (read,) = [r for r in circuit.registered.reads if r.param == PARAM and r.at == 0]
    row = [int(v) for v in values[PARAM].rows[UNIT]]
    w16 = [row[j * K:(j + 1) * K] for j in range(TN)]
    y_gemm = [int(v) for v in evaluate(gemm, x, row)]

    (tile,) = [t for t in PC.tiles(SCHEME, K, 1, N, tile=(1, TN)) if t.col == UNIT]
    salt, root = random.Random("served-linear").randbytes(32), values[PARAM].root
    a, b = [[v << 16 for v in x]], [[v << 16 for v in r] for r in w16]
    cert = _cert(tile, a, b, salt, root)
    v6 = PC.served_product(SCHEME, K, 1, TN, N)
    return dict(circuit=circuit, registration=registration, rpath=rpath, pins=pins, values=values, ck=ck, x=x, read=read, v6=v6,
                gemm=gemm, w16=w16, y_gemm=y_gemm, tile=tile, salt=salt, root=root, a=a, b=b, cert=cert)


def _cert(tile, a, b, salt: bytes, root: bytes) -> list:
    """The prover's inert-work certificates of the tile's one row against its TN columns under ``salt`` and ``root``."""
    s = pouw_schemes.get(SCHEME, k_min=K, tile=(1, TN))
    dev = s.device.name
    _d, ng = inert_bound._promoted(dev, K)
    ctx = C.Context(salt, C.Matmul(UNIT, C.Shape(1, K, N), 0), bytes(64), root)
    ops = s.encode(ctx, s.noise(ctx), [C.f32_row(r) for r in a], [C.f32_row(r) for r in b], C.Tile(tile.rows, tile.cols))
    assert ops is not None, "the served x and W rows don't encode under Pearl+'s scheme"
    return [[[list(inert_bound.certificates(ops.a.codes[0], ops.b.codes[j], dev))[2 * z:2 * z + 2] for z in range(ng - 1)]
             for j in range(TN)]]


def _tile(L, b=None, salt=None, cert=None) -> tuple:
    t, s = L["tile"], L["tile"].core.bindings
    return PT.reference(s["DEV"], K, N, L["a"], L["b"] if b is None else b, L["cert"] if cert is None else cert,
                        L["salt"] if salt is None else salt, L["root"], UNIT, list(t.rows), list(t.cols), s["RHO"],
                        s["RHOR"], t.real)


def test_the_tiles_b_rows_are_the_registered_row_the_served_unit_reads_and_its_root_the_registrations(linear):
    """The served `Gemm_v2{K=256,N=4}` unit 0 reads row 0 of ``qkv_proj@1024`` at ``w``, an Array of 4 rows of 256 BF16
    words, the very type of the tile's ``b`` at TN = 4; that row is q_proj's rows 0 to 3 in order (W is N × K, a row of
    the parameter TN of W's rows of K words); and the tile's ``root``, which seeds its noise, is the registration's root
    of that parameter, 64 bytes, which the verifier pins."""
    L = linear
    assert L["read"].reader.startswith("Gemm_v2{K=256,N=4,") and (L["read"].port, L["read"].base) == ("w", 0)
    assert dict(L["gemm"].signature()[0])["w"] == dict(L["tile"].core.signature()[0])["b"]
    q = L["ck"].array("model.layers.0.self_attn.q_proj.weight").view(np.uint16)
    assert L["w16"] == q[:TN].tolist()
    assert len(L["root"]) == 64 and L["root"].hex() == L["pins"][PARAM]


def test_the_tiles_a_port_takes_fp32_words_where_the_served_x_is_bf16_and_v6_takes_the_writers_row(linear):
    """The served x is the norm's 256 BF16 words (`RMSNormTriton_v5`'s output row, 16-bit leaves); `_v5`'s ``a`` is TM
    rows of K FP32 words. A wire reads a writer's row at the writer's width, so `_v5` cannot read x by wire.
    `PearlPlusTileCoreWords_v6` takes ``a`` in BF16 words, widened inside as ``b`` is (`Bf16ToF32`): its ``a`` row is
    the norm's output row and its ``b`` the served unit's ``w``; its other ports and its return are `_v5`'s."""
    L = linear
    norm = _unit(L["circuit"], "RMSNormTriton_v5")
    v5, v6 = (dict(t.signature()[0]) for t in (L["tile"].core, L["v6"]))
    assert norm.signature()[1].leaf_widths() == [16] * K and v5["a"].leaf_widths() == [32] * K
    assert v6["a"].elem == norm.signature()[1] and v6["b"] == v5["b"] == dict(L["gemm"].signature()[0])["w"]
    assert {p: t for p, t in v6.items() if p != "a"} == {p: t for p, t in v5.items() if p != "a"}
    assert L["v6"].signature()[1] == L["tile"].core.signature()[1]


def test_on_the_served_values_the_tile_accepts_and_writes_pearl_plus_y_not_the_bf16_gemms(linear):
    """On the served x and the registered W rows, with the prover's certificates under the epoch's salt and the
    registration's root, the tile's ``ok`` is 1 and it writes its 16 y words. Those are Pearl+'s served product (FP8
    noised, unscaled), not the served unit's `Gemm_v2` (BF16 on the Hopper WGMMA dot): the circuit's y at this linear
    changes, so every reader of it reads Pearl+'s y."""
    L = linear
    digest, y, ok = _tile(L)
    assert ok == 1 and digest is not None and len(y) == 1 and len(y[0]) == TN
    assert y[0] != L["y_gemm"]


def test_a_flipped_weight_word_is_refused_by_the_registered_read_and_the_tile_misses_a_low_bit(linear):
    """One weight word flipped: the row no longer opens under the registration's root (the committer's own salts give
    another root), which refuses it whatever the bit. The tile alone does not: its B rows are encoded to FP8 E4M3, so a
    flip of a word's lowest mantissa bit leaves its codes, digest, y and ok as they were; a flip of its exponent's low
    bit (bit 7) breaks the prover's certificates and the tile's ok is 0. So the block's B binding is the registered
    read, not the tile's arithmetic."""
    L = linear
    rows = np.asarray(L["values"][PARAM].rows, dtype=np.uint16).copy()
    salts = SF.read(SF.beside(L["rpath"]), PARAM)
    honest = WW.weight_value(L["registration"]["program"], PARAM, rows, salts=salts)
    rows[UNIT, 5] ^= 1
    again = WW.weight_value(L["registration"]["program"], PARAM, rows, salts=salts)
    assert honest.root.hex() == L["pins"][PARAM] != again.root.hex()
    for bit, same in ((0, True), (7, False)):
        b = [list(r) for r in L["b"]]
        b[0][5] ^= 1 << (16 + bit)
        got = _tile(L, b)
        assert (got == _tile(L)) is same and got[2] == int(same), bit


def test_a_flipped_y_word_is_refused(linear):
    """One word of the y the circuit holds flipped: the block's evaluation on its inputs writes the honest y, so the
    committed y is not the block's output."""
    L = linear
    _d, y, _ok = _tile(L)
    held = [list(r) for r in y]
    held[0][3] ^= 1
    assert _tile(L)[1] == y != held


def test_pearl_plus_y_at_the_linear_moves_with_the_epochs_salt(linear):
    """The tile's y is Pearl+'s served product, unscaled from the noised FP8 product: on the same x and W rows, another
    epoch's salt (with the prover's certificates under it) gives other y words. The served y at a credited linear is a
    function of the epoch's salt and the weight's registered root as well as of x and W."""
    L = linear
    other = bytes(31) + b"\x01"
    assert other != L["salt"]
    got = _tile(L, salt=other, cert=_cert(L["tile"], L["a"], L["b"], other, L["root"]))
    assert got[2] == 1 and got[1] != _tile(L)[1]


@pytest.mark.slow
def test_the_circuit_template_on_bits_is_the_reference_on_the_served_values(linear):
    """Slow (the tile's noise is about 370k · K ANDs, its core more): `PearlPlusTileCoreWords_v6` evaluated on bits on
    the served x as the norm wrote it, the registered row, the certificates and the public row writes the reference's
    digest, y and ok, bit for bit, and with one y word other than its own the block's output is not the y held; the only
    test that the circuit template, not just its reference, takes the served layout."""
    L = linear
    t = L["tile"]
    pub = PC.public(SCHEME, K, t, UNIT, L["salt"], L["root"])
    words = [[L["x"]], L["w16"], L["cert"], *(pub[p] for p in PC.PUBLIC)]
    g = evaluate_batch(L["v6"], *(np.asarray([_flat(x)], dtype=np.int64) for x in words)).reshape(-1).tolist()
    digest, y, ok = _tile(L)
    assert bytes(sum(g[8 * i + j] << j for j in range(8)) for i in range(64)) == digest
    assert g[512:512 + TN] == y[0] and g[512 + TN] == ok == 1


@pytest.mark.slow
def test_the_core_in_tn_plus_one_units_is_the_same_function_on_bits(linear):
    """Slow (four evaluations of the tile on bits): `PearlPlusTileCoreWords_v7`, whose core `PearlPlusTileCoreWired_v4`
    is TN + 1 units each within the composer's limit, writes `_v6`'s output bit for bit on the served values, and on them
    with a weight word's exponent bit flipped (where the tile's ok is 0); the only test that the finer laying of the core
    computes the core's function, not just its parts."""
    L = linear
    v7 = PC.bind(PC.PearlPlusTileCoreWordsV7, **L["v6"].bindings)
    assert v7.signature() == L["v6"].signature()
    pub = PC.public(SCHEME, K, L["tile"], UNIT, L["salt"], L["root"])
    flipped = [list(r) for r in L["w16"]]
    flipped[0][5] ^= 1 << 7
    for w in (L["w16"], flipped):
        words = [[L["x"]], w, L["cert"], *(pub[p] for p in PC.PUBLIC)]
        got = [evaluate_batch(t, *(np.asarray([_flat(x)], dtype=np.int64) for x in words)).reshape(-1).tolist()
               for t in (L["v6"], v7)]
        assert got[0] == got[1]
    assert got[0][512 + TN] == 0


def _flat(x) -> list[int]:
    if isinstance(x, (bytes, bytearray)):
        return list(x)
    if isinstance(x, (int, np.integer)):
        return [int(x)]
    return [w for v in x for w in _flat(v)]
