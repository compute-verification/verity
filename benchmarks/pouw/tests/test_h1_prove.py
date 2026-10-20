"""h1_prove's window and the verifier's view of it (`h1_prove/window.py`, `run.py`, `h1unit.py`), without the unit's circuit:
the window's public Program and template-instance partition derive exactly the tiles it commits and the rows each reads, a
window's drawn units open to those rows of its registered roots and to nothing else (a tampered window's, another
window's), the public row lays out the unit's salt, position and roots as its parameters read them, and the verifier's
frame-v3 roots over the opened ``b || c`` are C-Flock's own over the rows."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from verity.core.protocols.sampled_proofs.one_stage import partition as P

H1 = Path(__file__).resolve().parents[1] / "h1_prove"
sys.path.insert(0, str(H1))
import h1unit as U  # noqa: E402

spec = importlib.util.spec_from_file_location("h1_prove_run", H1 / "run.py")
RN = importlib.util.module_from_spec(spec)
spec.loader.exec_module(RN)

K = 64


def window(*args: str) -> dict:
    p = subprocess.run([sys.executable, str(H1 / "window.py"), *args], capture_output=True, text=True, check=True)
    return json.loads(p.stdout.strip().splitlines()[-1])


def layer(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    np.savez(path, tokens=np.arange(8), a=rng.integers(-126, 126, (8, K)).astype(np.int8),
             **{f"b_{c}": rng.integers(-110, 111, (16, K)).astype(np.int8) for c in ("k_proj", "q_proj")})


def committed(tmp: Path, name: str, seed: int, salt: bytes) -> tuple[dict, dict]:
    layer(tmp / f"{name}.npz", seed)
    window("commit", str(tmp / f"{name}.npz"), name, str(tmp / name), str(tmp / f"{name}-com.json"))
    served = window("serve", str(tmp / name), salt.hex(), str(tmp / f"{name}-served.json"), "--workers", "2")
    return json.loads((tmp / f"{name}-com.json").read_text()), served


def opened(tmp: Path, priv: str, salt: bytes, units: list[int], tag: str) -> list[dict]:
    window("open", str(tmp / priv), salt.hex(), ",".join(map(str, units)), str(tmp / tag))
    return json.loads((tmp / tag / "opening.json").read_text())["units"]


@pytest.fixture(scope="module")
def programs():
    """`run.window_program` once per window shape: the tile's Program traces its seed and noise circuits, about 35 s at any
    K, and its partition about 14 s more."""
    cache: dict = {}

    def get(com: dict):
        key = U.window_shape(com)
        if key not in cache:
            cache[key] = RN.window_program(com)
        return cache[key]
    return get


@pytest.mark.slow
def test_the_windows_partition_derives_exactly_the_tiles_h1_prove_proves(tmp_path, programs):
    """The window's public Program (`h1unit.Shape.window`) under its template-instance query has one unit per tile
    `window.py` commits, serves and opens, numbered as it numbers them, each an instance of the tile unit: unit u reads A
    rows R ti .. R ti + R - 1 of the slot, its call's B rows C tj .. C tj + C - 1 of ``b``, the salt, position u and the
    window's roots, and its outputs are row u of z and of the receipt. The slot pads the token rows with zero rows, and the
    public commitment holds no token count."""
    rng = np.random.default_rng(8)
    np.savez(tmp_path / "w.npz", tokens=np.arange(6), a=rng.integers(-126, 126, (6, K)).astype(np.int8),
             **{f"b_{c}": rng.integers(-110, 111, (n, K)).astype(np.int8) for c, n in (("k_proj", 16), ("q_proj", 20))})
    window("commit", str(tmp_path / "w.npz"), "w", str(tmp_path / "w"), str(tmp_path / "com.json"))
    com = json.loads((tmp_path / "com.json").read_text())
    assert (com["slot"], com["b_rows"], [(c["name"], c["n"]) for c in com["calls"]]) == (8, 32, [("k_proj", 16), ("q_proj", 16)])
    assert not {"rows", "tokens"} & set(com)
    priv = np.load(tmp_path / "w" / "rows.npz")
    assert priv["a"].shape == (8, K) and not priv["a"][6:].any()

    win = programs(com)
    T = com["tiles"]
    assert T == 2 * (8 // 4) * (16 // 8) == win.tp.population
    assert win.query["params"]["templates"] == [P.template_instance_query(U.Shape(K).definition())["params"]["template"]]
    assert [(t, first, n) for _k, t, first, n in win.tp.ranges] == [(win.query["params"]["templates"][0], u, 1) for u in range(T)]
    assert [n for n, _t in win.program.fn.signature()[0]] == [f"a/{K}", f"b/{K}", "salt", "position", "roots"]
    W = importlib.util.module_from_spec(spec_w := importlib.util.spec_from_file_location("h1_prove_window", H1 / "window.py"))
    spec_w.loader.exec_module(W)
    for u in range(T):
        c, ti, tj = W.tile_of(com, u)
        b0 = com["calls"][c]["b0"] + 8 * tj
        assert U.unit_reads(win.program, win.tp, u) == {
            **{f"a{i}": ("a", 4 * ti + i) for i in range(4)}, **{f"b{j}": ("b", b0 + j) for j in range(8)},
            "salt": ("salt", 0), "position": ("position", u), "roots": ("roots", 0), "o0": ("z", u), "o1": ("receipt", u)}

    salt = U.unit_salt(b"\x09" * 32)
    served = window("serve", str(tmp_path / "w"), salt.hex(), str(tmp_path / "served.json"), "--workers", "2")
    units = list(range(T))
    assert all(RN.check_opening(win, com, served, op, u) == [] for u, op in zip(units, opened(tmp_path, "w", salt, units, "all")))
    with pytest.raises(ValueError, match="power of two"):
        U.Shape(K).window(U.window_shape(com)[0], 6)
    assert [U.slot(n) for n in (1, 4, 5, 8, 9, 16, 17)] == [4, 4, 8, 8, 16, 16, 32]


@pytest.mark.slow
def test_drawn_units_open_to_their_window_and_no_other(tmp_path, programs):
    salt = U.unit_salt(b"\x01" * 32)
    com, served = committed(tmp_path, "w1", 1, salt)
    win = programs(com)
    assert com["tiles"] == 2 * (8 // 4) * (16 // 8)
    units = [0, 5, 7]
    for u, op in zip(units, opened(tmp_path, "w1", salt, units, "honest")):
        assert RN.check_opening(win, com, served, op, u) == []
    rows = np.load(tmp_path / "honest" / "rows.npz")
    assert rows["w_pub"].shape == (3, 100) and rows["s_pub"].shape == (3, U.SALT_BYTES)

    for flag, port in (("--z", "o0"), ("--receipt", "o1")):
        bad = window("tamper", str(tmp_path / "w1"), str(tmp_path / f"w1{flag}"), str(tmp_path / f"t{flag}.json"), flag, "5")
        ops = opened(tmp_path, f"w1{flag}", salt, units, f"open{flag}")
        assert all(RN.check_opening(win, com, bad, op, u) == [] for u, op in zip(units, ops))
        why = f"{port}: its path does not reach the registered {'z' if port == 'o0' else 'receipt'} root"
        assert [RN.check_opening(win, com, served, op, u) for u, op in zip(units, ops)] == [[why]] * len(units)

    salt2 = U.unit_salt(b"\x02" * 32)
    com2, served2 = committed(tmp_path, "w2", 2, salt2)
    op2 = opened(tmp_path, "w2", salt2, [5], "other")[0]
    assert RN.check_opening(win, com2, served2, op2, 5) == []
    assert len(RN.check_opening(win, com, served, op2, 5)) == 4 + 8 + 2


@pytest.mark.slow
def test_an_in_proof_negatives_window_opens_to_its_own_registration_with_one_value_plus_one(tmp_path, programs):
    salt = U.unit_salt(b"\x03" * 32)
    com, served = committed(tmp_path, "w1", 1, salt)
    win = programs(com)
    units = [1, 5]
    opened(tmp_path, "w1", salt, units, "honest")
    honest = np.load(tmp_path / "honest" / "rows.npz")
    for flag, port in (("--z-cell", "o0"), ("--receipt-word", "o1")):
        bad = window("tamper", str(tmp_path / "w1"), str(tmp_path / f"w1{flag}"), str(tmp_path / f"t{flag}.json"), flag, "5")
        ops = opened(tmp_path, f"w1{flag}", salt, units, f"open{flag}")
        assert all(RN.check_opening(win, com, bad, op, u) == [] for u, op in zip(units, ops))
        assert all(RN.check_opening(win, com, served, op, u) != [] for u, op in zip(units, ops))
        got = np.load(tmp_path / f"open{flag}" / "rows.npz")[f"w_{port}"]
        value = lambda w: int(w[0]) | int(w[1]) << 16  # noqa: E731
        assert value(got[1]) == value(honest[f"w_{port}"][1]) + 1 & 0xFFFFFFFF
        assert np.array_equal(got[1][2:], honest[f"w_{port}"][1][2:]) and np.array_equal(got[0], honest[f"w_{port}"][0])


def test_a_forged_statement_is_the_verifiers_file_with_the_wrong_row(tmp_path):
    from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2
    ports = U.Shape(K).ports()
    meta = {"ports": [[p.name, p.words, 0] for p in ports], "row_schemas": [[V2, p.bits] for p in ports], "vus_per_block": 1}
    circuit = tmp_path / "circuit.txt"
    circuit.write_text("FLOCK\nMETA " + json.dumps(meta) + "\n")
    head = {"meta": meta, "pin": "p" * 128, "class": "c" * 128}
    rng = np.random.default_rng(4)
    n = 2
    rows = {p.name: rng.integers(0, 1 << 16, (n, p.words)).astype(np.uint16) for p in ports}
    salts = {p.name: [bytes(rng.integers(0, 256, U.SALT_BYTES, dtype=np.uint8)) for _ in range(n)] for p in ports}
    bcs = {p.name: U.commit_strings(p, rows[p.name], salts[p.name]) for p in ports}
    pub = U.verifier_pub(head, "s", "sub", [{p.name: bcs[p.name][i] for p in ports} for i in range(n)], [3, 9])
    assert json.loads(pub[:pub.index(b"\n")])["units"]["indices"] == [3, 9]
    with pytest.raises(ValueError, match="ascending"):
        U.verifier_pub(head, "s", "sub", [{p.name: bcs[p.name][i] for p in ports} for i in range(n)], [9, 3])
    nl = pub.index(b"\n") + 1
    (tmp_path / "honest").mkdir()
    (tmp_path / "honest" / "inst.bin").write_bytes(
        json.dumps(dict(json.loads(pub[:nl]), rows=True), sort_keys=True).encode() + b"\n"
        + np.concatenate([rows[p.name] for p in ports], axis=1).astype("<u2").tobytes()
        + b"".join(salts[p.name][i] for i in range(n) for p in ports) + pub[nl:])

    wrong = rows["o1"][1].copy()
    wrong[0] ^= 4
    o1 = ports[-1]
    bad = [{p.name: (U.commit_strings(o1, wrong[None, :], [salts["o1"][1]])[0] if (i, p.name) == (1, "o1") else bcs[p.name][i])
            for p in ports} for i in range(n)]
    bad_pub = U.verifier_pub(head, "s", "sub", bad, [3, 9])
    got = U.forge_statement(tmp_path / "honest", tmp_path / "forged", bad_pub, 1, "o1", wrong, circuit)
    assert got == {"instance": 1, "port": "o1", "row_words_changed": 1}
    inst = (tmp_path / "forged" / "inst.bin").read_bytes()
    assert (tmp_path / "forged" / "pub.bin").read_bytes() == bad_pub
    inl = inst.index(b"\n") + 1
    assert json.loads(inst[:inl]) == dict(json.loads(bad_pub[:bad_pub.index(b"\n") + 1]), rows=True)
    assert inst.endswith(bad_pub[bad_pub.index(b"\n") + 1:])
    W = sum(p.words for p in ports)
    forged_rows = np.frombuffer(inst[inl:inl + 2 * n * W], "<u2").reshape(n, W)
    assert np.array_equal(forged_rows[1, -o1.words:], wrong) and np.array_equal(forged_rows[0], np.concatenate([rows[p.name][0] for p in ports]))

    with pytest.raises(ValueError, match="not these words under the prover's salt"):
        U.forge_statement(tmp_path / "honest", tmp_path / "f2", pub, 1, "o1", wrong, circuit)


def test_vllm_quant_puts_each_rows_absmax_at_127_rounding_to_nearest():
    spec_l = importlib.util.spec_from_file_location("h1_prove_layer", H1 / "layer.py")
    L = importlib.util.module_from_spec(spec_l)
    spec_l.loader.exec_module(L)
    x = np.array([[0.5, -2.0, 1.1, 0.26, 9.0], [-4.0, 1.0, 0.016, -0.015, 3.0]], np.float32)
    assert L.quant_vllm(x).tolist() == [[7, -28, 16, 4, 127], [-127, 32, 1, 0, 95]]
    assert L.quant_vllm(x, 3).tolist() == [[32, -127, 70], [-127, 32, 1]]
    rng = np.random.default_rng(5)
    q = L.quant_vllm(rng.standard_normal((64, 4096)).astype(np.float32), K)
    assert q.shape == (64, K) and q.dtype == np.int8
    assert (np.abs(q.astype(int)).max(axis=1) == 127).all() and q.min() >= -127


def test_v3_serves_a_bt_where_a_is_127_and_an_xa_flip_moves_only_z(tmp_path, monkeypatch):
    monkeypatch.setenv("H1_TILE", "v3")
    rng = np.random.default_rng(6)
    a = rng.integers(-127, 128, (8, K)).astype(np.int8)
    a[:, ::5] = 127
    np.savez(tmp_path / "w.npz", tokens=np.arange(8), a=a, **{"b_q_proj": rng.integers(-128, 128, (16, K)).astype(np.int8)})
    window("commit", str(tmp_path / "w.npz"), "w", str(tmp_path / "w"), str(tmp_path / "com.json"))
    salt = U.unit_salt(b"\x07" * 32)
    served = window("serve", str(tmp_path / "w"), salt.hex(), str(tmp_path / "served.json"), "--workers", "2")
    assert served["z_is_served_product"] == {"cells": 4 * 4 * 8, "equal": 4 * 4 * 8}
    assert served["xa"]["tile_rows_with_xa"] == served["xa"]["tile_rows"] == 16
    bad = window("tamper", str(tmp_path / "w"), str(tmp_path / "t"), str(tmp_path / "t.json"), "--xa-flip", "1")
    assert bad["receipt_root"] == served["receipt_root"] and bad["z_root"] != served["z_root"]
    flip = bad["tampered"]["xa_flip"]
    assert flip["xa_set_in_unit"] > 0 and flip["receipt_is_honest"] and flip["z_words_changed"] > 0


def test_the_public_row_is_the_parameters_bits():
    from verity.core.protocols.pouw.circuit import seeds as SD
    salt, roots = bytes(range(64)), (bytes(64 * [7]), bytes(range(64, 128)))
    w = U.pub_words(salt, 0x0123456789, roots)
    bits = np.unpackbits(w.astype("<u2").view(np.uint8), bitorder="little")
    assert list(bits) == SD.leaves(salt, True) + SD.position_leaves(0x0123456789, True) + SD.leaves(roots[0], True) + \
        SD.leaves(roots[1], True)


def test_the_verifiers_frame_is_c_flocks():
    from verity_flock import circuit as FC
    ports = U.Shape(K).ports()
    rng = np.random.default_rng(3)
    n = 3
    rows = {p.name: rng.integers(0, 1 << 16, (n, p.words)).astype(np.uint16) for p in ports}
    salts = {p.name: [bytes(rng.integers(0, 256, U.SALT_BYTES, dtype=np.uint8)) for _ in range(n)] for p in ports}
    fv, bcs = FC.statement(ports, SimpleNamespace(name="h1-test"), 0, n, rows, salts)
    assert U._frame(ports, "h1-test", n, bcs) == fv
