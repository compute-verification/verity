"""h1_prove's window and the verifier's view of it (`h1_prove/window.py`, `run.py`, `h1unit.py`), without the unit's circuit:
the window's public Program and template-instance partition derive exactly the tiles it commits and the rows each reads,
its registered values a/<K> and b/<K> are read where the window's own reads say, a window's drawn units open to those rows
of its registered roots and to nothing else (a tampered window's, another window's, a tampered A row), the public row
lays out the unit's salt, position and roots as its parameters read them, and the verifier's shared-row file over the
opened ``b || c`` and paths is C-Flock's own over the rows."""
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

from verity.core.protocols.pouw import salt as PS  # noqa: E402
from verity.core.protocols.pouw.circuit import seeds as SD  # noqa: E402

spec = importlib.util.spec_from_file_location("h1_prove_run", H1 / "run.py")
RN = importlib.util.module_from_spec(spec)
spec.loader.exec_module(RN)

K = 64
#: `layer`'s window shape: two calls of 16 B rows, a slot of 8 A rows
SHAPE = ((("k_proj", 16), ("q_proj", 16)), 8)


def window(*args: str) -> dict:
    p = subprocess.run([sys.executable, str(H1 / "window.py"), *args], capture_output=True, text=True, check=True)
    return json.loads(p.stdout.strip().splitlines()[-1])


def layer(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    np.savez(path, tokens=np.arange(8), a=rng.integers(-126, 126, (8, K)).astype(np.int8),
             **{f"b_{c}": rng.integers(-110, 111, (16, K)).astype(np.int8) for c in ("k_proj", "q_proj")})


def commit(tmp: Path, layer_path: Path, name: str, program: str) -> dict:
    window("commit", str(layer_path), name, str(tmp / name), str(tmp / f"{name}-com.json"), "--program-sha512", program)
    return json.loads((tmp / f"{name}-com.json").read_text())


def committed(tmp: Path, name: str, seed: int, salt: bytes, win) -> tuple[dict, dict]:
    layer(tmp / f"{name}.npz", seed)
    com = commit(tmp, tmp / f"{name}.npz", name, win.obj["program"])
    served = window("serve", str(tmp / name), salt.hex(), str(tmp / f"{name}-served.json"), "--workers", "2")
    return com, served


def opened(tmp: Path, priv: str, salt: bytes, units: list[int], tag: str, *flags: str) -> list[dict]:
    window("open", str(tmp / priv), salt.hex(), ",".join(map(str, units)), str(tmp / tag), *flags)
    return json.loads((tmp / tag / "opening.json").read_text())["units"]


@pytest.fixture(scope="module")
def win():
    """`run.program_of` the tests' window shape (`SHAPE`), which every commitment registers under: the tile's Program
    traces its seed and noise circuits, about 35 s at any K, and its partition about 14 s more."""
    return RN.program_of(K, *SHAPE)


@pytest.mark.slow
def test_the_windows_partition_derives_exactly_the_tiles_h1_prove_proves(tmp_path, win):
    """The window's public Program (`h1unit.Shape.window`) under its template-instance query has one unit per tile
    `window.py` commits, serves and opens, numbered as it numbers them, each an instance of the tile unit: unit u reads A
    rows R ti .. R ti + R - 1 of the slot, its call's B rows C tj .. C tj + C - 1 of ``b``, the salt, position u and the
    window's roots, and its outputs are row u of z and of the receipt. The slot pads the token rows with zero rows, and the
    public commitment holds no token count. Its A and B rows are the Program's registered values a/<K> and b/<K>, and the
    verifier's ``--registered`` file (`registered.reads_file`, from the Program alone) reads, at every unit and port, the
    row `h1window.unit_reads` does."""
    rng = np.random.default_rng(8)
    np.savez(tmp_path / "w.npz", tokens=np.arange(6), a=rng.integers(-126, 126, (6, K)).astype(np.int8),
             **{f"b_{c}": rng.integers(-110, 111, (n, K)).astype(np.int8) for c, n in (("k_proj", 16), ("q_proj", 20))})
    com = commit(tmp_path, tmp_path / "w.npz", "w", win.obj["program"])
    assert (com["slot"], com["b_rows"], [(c["name"], c["n"]) for c in com["calls"]]) == (8, 32, [("k_proj", 16), ("q_proj", 16)])
    assert U.window_shape(com) == SHAPE
    assert not {"rows", "tokens"} & set(com)
    priv = np.load(tmp_path / "w" / "rows.npz")
    assert priv["a"].shape == (8, K) and not priv["a"][6:].any()
    assert RN.check_registration(win, com) == []
    assert RN.check_registration(win, dict(com, program_sha512="0" * 128)) != []
    assert RN.check_registration(win, dict(com, root_b=com["root_a"])) != []
    port_b0 = U.Shape(K).ports()[4]
    assert [bytes(x) for x in priv["bcb"]] == U.commit_strings(port_b0, U.row_words(priv["b"]), [bytes(s) for s in priv["sb"]])

    T = com["tiles"]
    assert T == 2 * (8 // 4) * (16 // 8) == win.tp.population
    assert win.query["params"]["templates"] == [P.template_instance_query(U.Shape(K).definition())["params"]["template"]]
    assert [(t, first, n) for _k, t, first, n in win.tp.ranges] == [(win.query["params"]["templates"][0], u, 1) for u in range(T)]
    assert [n for n, _t in win.program.fn.signature()[0]] == [f"a/{K}", f"b/{K}", "salt", "position", "roots"]
    assert U.window_ports(win.program) == {**{f"a_k{K}[{i}]": f"a{i}" for i in range(4)},
                                           **{f"b_k{K}[{j}]": f"b{j}" for j in range(8)}}
    W = importlib.util.module_from_spec(spec_w := importlib.util.spec_from_file_location("h1_prove_window", H1 / "window.py"))
    spec_w.loader.exec_module(W)
    for u in range(T):
        c, ti, tj = W.tile_of(com, u)
        b0 = com["calls"][c]["b0"] + 8 * tj
        assert U.unit_reads(win.program, win.tp, u) == {
            **{f"a{i}": ("a", 4 * ti + i) for i in range(4)}, **{f"b{j}": ("b", b0 + j) for j in range(8)},
            "salt": ("salt", 0), "position": ("position", u), "roots": ("roots", 0), "o0": ("z", u), "o1": ("receipt", u)}

    reads = U.registered_reads(win.program, win.query, {f"a/{K}": com["root_a"], f"b/{K}": com["root_b"]})
    assert sorted(reads["ports"]) == sorted(U.Shape(K).in_rows)
    for port, e in reads["ports"].items():
        tree, leaves = ("a", 8) if port[0] == "a" else ("b", 32)
        assert {x: e[x] for x in e if x != "reads"} == {
            "value": f"{tree}/{K}", "root": com[f"root_{tree}"], "leaves": leaves, "words": K // 2, "word_bits": 16,
            "domain": U.registered_domain(win.obj["program"], f"{tree}/{K}", K, leaves).domain_id.hex(), "row_schema": U.V2}
        assert e["reads"] == [U.unit_reads(win.program, win.tp, u)[port][1] for u in range(T)]

    salt = U.unit_salt(U.stand_in_reveal(b"\x09" * 32), 1)
    served = window("serve", str(tmp_path / "w"), salt.hex(), str(tmp_path / "served.json"), "--workers", "2")
    units = list(range(T))
    assert all(RN.check_opening(win, com, served, op, u) == [] for u, op in zip(units, opened(tmp_path, "w", salt, units, "all")))
    with pytest.raises(ValueError, match="power of two"):
        U.Shape(K).window(U.window_shape(com)[0], 6)
    assert [U.slot(n) for n in (1, 4, 5, 8, 9, 16, 17)] == [4, 4, 8, 8, 16, 16, 32]


@pytest.mark.slow
def test_drawn_units_open_to_their_window_and_no_other(tmp_path, win):
    """A drawn unit's opening passes its own window's roots and no other's, and a prover's rows with a tampered A row
    (`window.py open --tamper-a`, at every unit that reads it) open no registered root through the path they keep."""
    salt = U.unit_salt(U.stand_in_reveal(b"\x01" * 32), 1)
    com, served = committed(tmp_path, "w1", 1, salt, win)
    assert com["tiles"] == 2 * (8 // 4) * (16 // 8)
    assert served["salt"] == PS.pouw_salt(U.stand_in_reveal(b"\x01" * 32), 1, U.STREAM, U.PROVER).hex()
    units = [0, 5, 7]
    for u, op in zip(units, opened(tmp_path, "w1", salt, units, "honest")):
        assert RN.check_opening(win, com, served, op, u) == []
    rows = np.load(tmp_path / "honest" / "rows.npz")
    assert rows["w_pub"].shape == (3, 100) and rows["s_pub"].shape == (3, U.SALT_BYTES)
    assert rows["r_a0"].tolist() == [0, 0, 4] and rows["r_b0"].tolist() == [0, 24, 24]

    a0 = U.Shape(K).ports()[0]
    ops = opened(tmp_path, "w1", salt, units, "tampered", "--tamper-a")
    assert [RN.check_opening(win, com, served, op, u) for u, op in zip(units, ops)] == [[]] * 3
    bad = np.load(tmp_path / "tampered" / "rows.npz")
    assert [int(np.count_nonzero(x != y)) for x, y in zip(bad["w_a0"], rows["w_a0"])] == [1, 1, 0]
    assert bytes(bad["s_a0"][0]) == bytes(bad["s_a0"][1]) != bytes(rows["s_a0"][0])
    assert np.array_equal(bad["w_a1"], rows["w_a1"]) and np.array_equal(bad["p_a0"], rows["p_a0"])
    assert U.statement_tables(dict(bad), [*U.Shape(K).in_rows, "pub"], ["o0", "o1"], K)["registered"]["a0"]["positions"] == [0, 4]
    for z, opens in ((rows, True), (bad, False)):
        bc = U.commit_strings(a0, z["w_a0"][:1], [bytes(z["s_a0"][0])])[0]
        assert U.verify_registered(win.obj["program"], f"a/{K}", K, 8, bytes.fromhex(com["root_a"]), 0, bc, [str(z["p_a0"][0])]) is opens

    for flag, port in (("--z", "o0"), ("--receipt", "o1")):
        bad = window("tamper", str(tmp_path / "w1"), str(tmp_path / f"w1{flag}"), str(tmp_path / f"t{flag}.json"), flag, "5")
        ops = opened(tmp_path, f"w1{flag}", salt, units, f"open{flag}")
        assert all(RN.check_opening(win, com, bad, op, u) == [] for u, op in zip(units, ops))
        why = f"{port}: its path does not reach the registered {'z' if port == 'o0' else 'receipt'} root"
        assert [RN.check_opening(win, com, served, op, u) for u, op in zip(units, ops)] == [[why]] * len(units)

    salt2 = U.unit_salt(U.stand_in_reveal(b"\x02" * 32), 2)
    com2, served2 = committed(tmp_path, "w2", 2, salt2, win)
    assert RN.check_registration(win, com2) == [] and com2["root_a"] != com["root_a"]
    op2 = opened(tmp_path, "w2", salt2, [5], "other")[0]
    assert RN.check_opening(win, com2, served2, op2, 5) == []
    assert len(RN.check_opening(win, com, served, op2, 5)) == 4 + 8 + 2


@pytest.mark.slow
def test_an_in_proof_negatives_window_opens_to_its_own_registration_with_one_value_plus_one(tmp_path, win):
    salt = U.unit_salt(U.stand_in_reveal(b"\x03" * 32), 1)
    com, served = committed(tmp_path, "w1", 1, salt, win)
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


def test_the_verifiers_shared_row_file_is_c_flocks_and_a_tampered_a_row_opens_no_root(tmp_path):
    """Three units of a window, units 0 and 2 reading the same A rows and units 0 and 1 the same B rows: the prover's
    shared-row statement (`h1unit.statement_tables` through C-Flock's `circuit.write`, as `stage.py` stages it) and the
    verifier's own file (`h1unit.verifier_pub`, from the units' ``b || c``, positions and paths alone) are one file byte
    for byte, every A and B table row opening its registered root (`run.registered_opens`, flock-verify --registered's
    check). A prover's A row the registration does not hold opens none, two rows at one position are refused, and a forged
    statement (`h1unit.forge_statement`) is the verifier's file of a window with the wrong output row."""
    from verity_flock import circuit as FC
    ports = U.Shape(K).ports()
    names = [p.name for p in ports]
    ins, outs, reg_ports = names[:-2], names[-2:], names[:-3]
    meta = {"ports": [[p.name, p.words, 0] for p in ports], "row_schemas": [[U.V2, p.bits] for p in ports],
            "input_ports": len(ins), "out_ports": [[p.name, p.words] for p in ports[-2:]], "vus_per_block": 1}
    text = "FLOCK\nMETA " + json.dumps(meta) + "\n"
    (tmp_path / "circuit.txt").write_text(text)
    head = {"meta": meta, "pin": "p" * 128, "class": "c" * 128}
    rng = np.random.default_rng(4)
    prog, n, indices = "ab" * 64, 3, [3, 9, 12]
    codes = {"a": rng.integers(-128, 128, (8, K)).astype(np.int8), "b": rng.integers(-128, 128, (16, K)).astype(np.int8)}
    reg = {t: U.register(prog, U.value(t, K), c, [rng.bytes(U.SALT_BYTES) for _ in c]) for t, c in codes.items()}
    assert list(reg["a"].bcs) == U.commit_strings(ports[0], U.row_words(codes["a"]), list(reg["a"].salts))
    com = {"unit": {"k": K}, "slot": 8, "b_rows": 16, **{f"root_{t}": r.root.hex() for t, r in reg.items()}}
    verifier = SimpleNamespace(obj={"program": prog})
    first = {"a": [0, 4, 0], "b": [0, 0, 8]}
    salts_of = lambda s: np.frombuffer(b"".join(s), np.uint8).reshape(len(s), U.SALT_BYTES)  # noqa: E731
    z = {"units": np.array(indices)}
    for p in reg_ports:
        t, pos = p[0], [f + int(p[1:]) for f in first[p[0]]]
        z.update({f"w_{p}": U.row_words(codes[t][pos]), f"s_{p}": salts_of([reg[t].salts[q] for q in pos]),
                  f"r_{p}": np.array(pos), f"p_{p}": np.array([reg[t].path(q).hex() for q in pos])})
    for p in ["pub", *outs]:
        port = ports[names.index(p)]
        z[f"w_{p}"] = rng.integers(0, 1 << 16, (n, port.words)).astype(np.uint16)
        z[f"s_{p}"] = salts_of([rng.bytes(U.SALT_BYTES) for _ in range(n)])

    def staged(z: dict, d: Path) -> bytes:
        st = U.statement_tables(z, ins, outs, K)
        d.mkdir()
        FC.write(text, SimpleNamespace(name="s", subcircuit=SimpleNamespace(id="sub")), 0, n, d / "inst.bin", d / "pub.bin",
                 None, indices, share_rows=True, outs={p: z[f"w_{p}"] for p in outs}, tables=st["tables"], refs=st["refs"],
                 salts=st["salts"], registered=st["registered"], digests=(head["pin"], head["class"]))
        return (d / "pub.bin").read_bytes()

    def bcs_reads(z: dict) -> tuple[list, list]:
        bcs = [{p: U.commit_strings(ports[names.index(p)], z[f"w_{p}"][i:i + 1], [bytes(z[f"s_{p}"][i])])[0] for p in names}
               for i in range(n)]
        return bcs, [{p: (U.value(p[0], K), int(z[f"r_{p}"][i]), str(z[f"p_{p}"][i])) for p in reg_ports} for i in range(n)]

    pub = staged(z, tmp_path / "honest")
    bcs, reads = bcs_reads(z)
    assert U.verifier_pub(head, "s", "sub", bcs, indices, reads) == pub
    h = json.loads(pub[:pub.index(b"\n")])
    assert h["shared_rows"] == {**{p: 2 for p in reg_ports}, "pub": 3, "o0": 3, "o1": 3}
    assert (h["registered"]["a0"]["positions"], h["registered"]["b3"]["positions"], h["units"]["indices"]) == ([0, 4], [3, 11], indices)
    assert all(RN.registered_opens(pub, ports, p, r, verifier, com) for p in reg_ports for r in range(2))
    with pytest.raises(ValueError, match="ascending"):
        U.verifier_pub(head, "s", "sub", bcs, [9, 3, 12], reads)

    bad, code = dict(z), codes["a"][:1].copy()
    code[0, 0] ^= 1
    bad["w_a0"], bad["s_a0"] = z["w_a0"].copy(), z["s_a0"].copy()
    bad["w_a0"][[0, 2]], bad["s_a0"][[0, 2]] = U.row_words(code)[0], np.frombuffer(rng.bytes(U.SALT_BYTES), np.uint8)
    pub_t = staged(bad, tmp_path / "tampered")
    assert U.verifier_pub(head, "s", "sub", *bcs_reads(bad)[:1], indices, bcs_reads(bad)[1]) == pub_t
    assert [RN.registered_opens(pub_t, ports, p, r, verifier, com) for p, r in (("a0", 0), ("a0", 1), ("a1", 0))] == [False, True, True]
    half = dict(bad, w_a0=bad["w_a0"].copy(), s_a0=bad["s_a0"].copy())
    half["w_a0"][2], half["s_a0"][2] = z["w_a0"][2], z["s_a0"][2]
    with pytest.raises(ValueError, match="instances 0 and 2 read position 0 as two rows"):
        U.statement_tables(half, ins, outs, K)
    with pytest.raises(ValueError, match="units 0 and 2 read position 0 as two rows"):
        U.verifier_pub(head, "s", "sub", *bcs_reads(half)[:1], indices, bcs_reads(half)[1])

    o1 = ports[-1]
    wrong = z["w_o1"][1].copy()
    wrong[0] ^= 4
    bad_bcs = [dict(b, o1=U.commit_strings(o1, wrong[None, :], [bytes(z["s_o1"][1])])[0]) if i == 1 else b
               for i, b in enumerate(bcs)]
    bad_pub = U.verifier_pub(head, "s", "sub", bad_bcs, indices, reads)
    got = U.forge_statement(tmp_path / "honest", tmp_path / "forged", bad_pub, 1, "o1", wrong, tmp_path / "circuit.txt")
    assert got == {"instance": 1, "port": "o1", "row_words_changed": 1}
    assert (tmp_path / "forged" / "pub.bin").read_bytes() == bad_pub
    inst, honest = (tmp_path / "forged" / "inst.bin").read_bytes(), (tmp_path / "honest" / "inst.bin").read_bytes()
    inl, bnl = inst.index(b"\n") + 1, bad_pub.index(b"\n") + 1
    assert json.loads(inst[:inl]) == dict(json.loads(bad_pub[:bnl]), rows=True) and inst.endswith(bad_pub[bnl:])
    at = inl + 2 * (sum(h["shared_rows"][q.name] * q.words for q in ports[:-1]) + o1.words)
    hnl = honest.index(b"\n") + 1
    assert np.array_equal(np.frombuffer(inst[at:at + 2 * o1.words], "<u2"), wrong)
    assert inst[inl:at] == honest[hnl:hnl + at - inl] and inst[at + 2 * o1.words:-len(bad_pub) + bnl] == \
        honest[hnl + at - inl + 2 * o1.words:-len(pub) + bnl]
    with pytest.raises(ValueError, match="not these words under the prover's salt"):
        U.forge_statement(tmp_path / "honest", tmp_path / "f2", pub, 1, "o1", wrong, tmp_path / "circuit.txt")
    with pytest.raises(ValueError, match="an output row of a shared-row file"):
        U.forge_statement(tmp_path / "honest", tmp_path / "f3", bad_pub, 1, "a0", z["w_a0"][1], tmp_path / "circuit.txt")


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
    window("commit", str(tmp_path / "w.npz"), "w", str(tmp_path / "w"), str(tmp_path / "com.json"), "--program-sha512", "0" * 128)
    salt = U.unit_salt(U.stand_in_reveal(b"\x07" * 32), 1)
    served = window("serve", str(tmp_path / "w"), salt.hex(), str(tmp_path / "served.json"), "--workers", "2")
    assert served["z_is_served_product"] == {"cells": 4 * 4 * 8, "equal": 4 * 4 * 8}
    assert served["xa"]["tile_rows_with_xa"] == served["xa"]["tile_rows"] == 16
    bad = window("tamper", str(tmp_path / "w"), str(tmp_path / "t"), str(tmp_path / "t.json"), "--xa-flip", "1")
    assert bad["receipt_root"] == served["receipt_root"] and bad["z_root"] != served["z_root"]
    flip = bad["tampered"]["xa_flip"]
    assert flip["xa_set_in_unit"] > 0 and flip["receipt_is_honest"] and flip["z_words_changed"] > 0


def test_the_seeds_derive_from_the_window_salt():
    """The units' salt is PoUW's window salt, SHA-512(b"verity/pouw-salt/v1" ‖ reveal ‖ u64be(p) ‖ stream ‖ prover), of the
    stand-in reveal (the salt coin's, standing in for the certifier's), and a unit's noise and receipt seeds are of that
    salt; another reveal, window, stream or prover gives another salt and other seeds."""
    import hashlib

    coin, roots = b"\x05" * 32, (bytes(64 * [7]), bytes(range(64, 128)))
    reveal = U.stand_in_reveal(coin)
    assert reveal == hashlib.sha512(b"verity/pouw/h1-prove/stand-in-reveal/v0\0" + coin).digest()
    assert U.PROVER == PS.prover_id(b"verity/pouw/h1-prove/stand-in-prover-key/v0") and len(U.STREAM) == 64
    salt = U.unit_salt(reveal, 3)
    assert salt == PS.pouw_salt(reveal, 3, U.STREAM, U.PROVER) == hashlib.sha512(
        b"verity/pouw-salt/v1" + reveal + (3).to_bytes(8, "big") + U.STREAM + U.PROVER).digest()

    def seeds(s):
        return SD.seed(s, 9, *roots), SD.receipt_seed(s, 9, *roots)

    others = [U.unit_salt(U.stand_in_reveal(b"\x06" * 32), 3), U.unit_salt(reveal, 4),
              U.unit_salt(reveal, 3, stream=bytes(64)), U.unit_salt(reveal, 3, prover=PS.prover_id(b"another key"))]
    assert len({salt, *others}) == 5
    mine = seeds(salt)
    assert all(a != b for o in others for a, b in zip(seeds(o), mine))


def test_the_public_row_is_the_parameters_bits():
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
