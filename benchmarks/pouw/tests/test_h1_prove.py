"""h1_prove's window and the verifier's view of it (`h1_prove/window.py`, `run.py`, `h1unit.py`, `stage.py`), without the
unit's circuit: the window's public Program and template-instance partition derive exactly the tiles it commits and the
rows each reads, its registered values a/<K>, b/<K> and zy (every tile's z/y record, `scaled_mm_zy`'s) are read where the
window's own reads say, a window's drawn units open to those rows of its registered roots and to nothing else (a tampered
window's, another window's, a tampered A row), each in-proof negative's window changes the one value it names and makes
the unit's word reference compute ok = 0 exactly where its record moved, a registration of z alone is refused, the public
row lays out the unit's salt, position and roots as its parameters read them, and the verifier's shared-row file over the
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
from verity.core.protocols.sampled_proofs.one_stage import registered as G

H1 = Path(__file__).resolve().parents[1] / "h1_prove"
sys.path.insert(0, str(H1))
import h1unit as U  # noqa: E402

from verity.core.protocols.pouw import salt as PS  # noqa: E402
from verity.core.protocols.pouw.circuit import h1zy as Z  # noqa: E402
from verity.core.protocols.pouw.circuit import seeds as SD  # noqa: E402


def _module(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, H1 / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


RN = _module("h1_prove_run", "run.py")
W = _module("h1_prove_window", "window.py")
ST = _module("h1_prove_stage", "stage.py")

K = 64
R, C = U.TILE
RC = R * C
#: `layer`'s window shape: two calls of 16 B rows, a slot of 8 A rows
SHAPE = ((("k_proj", 16), ("q_proj", 16)), 8)


def window(*args: str) -> dict:
    p = subprocess.run([sys.executable, str(H1 / "window.py"), *args], capture_output=True, text=True, check=True)
    return json.loads(p.stdout.strip().splitlines()[-1])


def scales(rng, n: int) -> np.ndarray:
    return (rng.uniform(0.5, 2.0, n) * 2.0 ** rng.integers(-12, -4, n)).astype(np.float32)


def layer(path: Path, seed: int, rows: int = 8, calls=(("k_proj", 16), ("q_proj", 16)), a=None) -> None:
    """A `layer.py` file: ``rows`` token rows of codes (or ``a``) with their f32 scales ``sa``, and each call's B rows with
    theirs, ``s_<call>``."""
    rng = np.random.default_rng(seed)
    a = rng.integers(-126, 126, (rows, K)).astype(np.int8) if a is None else a
    np.savez(path, tokens=np.arange(len(a)), a=a, sa=scales(rng, len(a)),
             **{f"b_{c}": rng.integers(-110, 111, (n, K)).astype(np.int8) for c, n in calls},
             **{f"s_{c}": scales(rng, n) for c, n in calls})


def commit(tmp: Path, layer_path: Path, name: str, program: str) -> dict:
    window("commit", str(layer_path), name, str(tmp / name), str(tmp / f"{name}-com.json"), "--program-sha512", program)
    return json.loads((tmp / f"{name}-com.json").read_text())


def committed(tmp: Path, name: str, seed: int, salt: bytes, win) -> tuple[dict, dict]:
    layer(tmp / f"{name}.npz", seed)
    com = commit(tmp, tmp / f"{name}.npz", name, win.obj["program"])
    served = window("serve", str(tmp / name), salt.hex(), str(tmp / f"{name}-served.json"), "--workers", "2")
    return com, served


def tampered(tmp: Path, priv: str, *flag: str) -> tuple[dict, dict]:
    """`window.py tamper` of ``priv`` under ``flag``: the tampered window's served root and its public commitment."""
    tag = "".join(flag).replace("-", "")
    served = window("tamper", str(tmp / priv), str(tmp / f"{priv}-{tag}"), str(tmp / f"{tag}-com.json"),
                    str(tmp / f"{tag}-served.json"), *flag)
    return served, json.loads((tmp / f"{tag}-com.json").read_text())


def opened(tmp: Path, priv: str, salt: bytes, units: list[int], tag: str, *flags: str) -> list[dict]:
    window("open", str(tmp / priv), salt.hex(), ",".join(map(str, units)), str(tmp / tag), *flags)
    return json.loads((tmp / tag / "opening.json").read_text())["units"]


def f32(v) -> list[int]:
    return [int(x) for x in np.asarray(v, "<f4").reshape(-1).view("<u4")]


def reference(z: dict, i: int) -> tuple[list[int], int]:
    """`h1zy.reference` on instance ``i`` of `window.py open`'s rows ``z``: its receipt and ok."""
    a = [U.split_row(z[f"w_a{x}"][i]) for x in range(R)]
    b = [U.split_row(z[f"w_b{x}"][i]) for x in range(C)]
    return Z.reference([c for c, _s in a], [c for c, _s in b], [s for _c, s in a], [s for _c, s in b],
                       U.split_record(z["w_zy"][i], RC), bytes(z["salt"]), int(z["units"][i]),
                       (bytes(z["roots"][i][0]), bytes(z["roots"][i][1])))


@pytest.fixture(scope="module")
def win():
    """`run.program_of` the tests' window shape (`SHAPE`), which every commitment registers under: the tile's Program
    traces its seed and noise circuits, about 35 s at any K, and its partition about 14 s more."""
    return RN.program_of(K, *SHAPE)


@pytest.mark.slow
def test_the_windows_partition_derives_exactly_the_tiles_h1_prove_proves(tmp_path, win):
    """The window's public Program (`h1unit.Shape.window`) under its template-instance query has one unit per tile
    `window.py` commits, serves and opens, numbered as it numbers them, each an instance of the tile unit: unit u reads A
    rows R ti .. R ti + R - 1 of the slot, its call's B rows C tj .. C tj + C - 1 of ``b``, row u of ``zy``, the salt,
    position u and the window's roots, and its outputs are row u of the receipt and of ok. The slot pads the token rows
    with zero rows of scale 0, and the public commitment holds no token count. Its rows (with their scales) and records
    are the Program's registered values a/<K>, b/<K> and zy, each record `scaled_mm_zy`'s (`h1zy.records`); a
    registration of z alone is refused; and the verifier's ``--registered`` file (`registered.reads_file`, from the
    Program alone) reads, at every unit and port, the row `h1window.unit_reads` does."""
    layer(tmp_path / "w.npz", 8, rows=6, calls=(("k_proj", 16), ("q_proj", 20)))
    com = commit(tmp_path, tmp_path / "w.npz", "w", win.obj["program"])
    assert (com["slot"], com["b_rows"], [(c["name"], c["n"]) for c in com["calls"]]) == (8, 32, [("k_proj", 16), ("q_proj", 16)])
    assert U.window_shape(com) == SHAPE
    assert not {"rows", "tokens"} & set(com)
    priv = np.load(tmp_path / "w" / "rows.npz")
    assert priv["a"].shape == (8, K) and not priv["a"][6:].any() and not priv["scale_a"][6:].any()
    assert RN.check_registration(win, com) == []
    assert RN.check_registration(win, dict(com, program_sha512="0" * 128)) != []
    assert RN.check_registration(win, dict(com, root_b=com["root_a"])) != []
    assert RN.check_registration(win, dict(com, root_zy=com["root_a"])) != []
    port = {p.name: p for p in U.Shape(K).ports()}
    assert [bytes(x) for x in priv["bcb"]] == U.commit_strings(port["b0"], U.row_words(priv["b"], priv["scale_b"]),
                                                               [bytes(s) for s in priv["sb"]])
    assert [bytes(x) for x in priv["bczy"]] == U.commit_strings(port["zy"], priv["zyw"], [bytes(s) for s in priv["szy"]])
    zonly = G.Registered.commit(com["program_sha512"], "zy", priv["zyw"][:, :2 * RC].tolist(), [bytes(s) for s in priv["szy"]],
                                G.WORD_BITS, U.V2)
    why = RN.check_registration(win, dict(com, root_zy=zonly.root.hex(),
                                          registered=[zonly.entry() if e["value"] == "zy" else e for e in com["registered"]]))
    assert len(why) == 1 and why[0].startswith("its registered entries")
    assert RN.check_registration(win, RN.z_only(com)) != []

    T = com["tiles"]
    assert T == 2 * (8 // R) * (16 // C) == win.tp.population
    assert win.query["params"]["templates"] == [P.template_instance_query(U.Shape(K).definition())["params"]["template"]]
    assert [(t, first, n) for _k, t, first, n in win.tp.ranges] == [(win.query["params"]["templates"][0], u, 1) for u in range(T)]
    assert [n for n, _t in win.program.fn.signature()[0]] == [f"a/{K}", f"b/{K}", "zy", "salt", "position", "roots"]
    assert U.window_ports(win.program) == {**{f"a_k{K}[{i}]": f"a{i}" for i in range(R)},
                                           **{f"b_k{K}[{j}]": f"b{j}" for j in range(C)}, "zy": "zy"}
    for u in range(T):
        c, ti, tj = W.tile_of(com, u)
        b0 = com["calls"][c]["b0"] + C * tj
        assert U.unit_reads(win.program, win.tp, u) == {
            **{f"a{i}": ("a", R * ti + i) for i in range(R)}, **{f"b{j}": ("b", b0 + j) for j in range(C)}, "zy": ("zy", u),
            "salt": ("salt", 0), "position": ("position", u), "roots": ("roots", 0), "o0": ("receipt", u), "o1": ("ok", u)}
        rows, cols = slice(R * ti, R * ti + R), slice(b0, b0 + C)
        a, b, sa, sw = priv["a"][rows], priv["b"][cols], priv["scale_a"][rows], priv["scale_b"][cols]
        assert np.array_equal(priv["zyw"][u], U.record_words(*U.records(a, b, sa, sw)))
        if u in (0, T - 1):
            enc = lambda m: [[int(x) & 0xFF for x in r] for r in m]  # noqa: E731
            assert U.split_record(priv["zyw"][u], RC) == Z.records(enc(a), enc(b), f32(sa), f32(sw))

    reads = U.registered_reads(win.program, win.query, {U.value(t, K): com[f"root_{t}"] for t in U.REGISTERED})
    assert sorted(reads["ports"]) == sorted(U.Shape(K).in_rows)
    leaves = {"a": 8, "b": 32, "zy": T}
    for p, e in reads["ports"].items():
        tree = U.tree_of(p)
        v = U.value(tree, K)
        assert {x: e[x] for x in e if x != "reads"} == {
            "value": v, "root": com[f"root_{tree}"], "leaves": leaves[tree], "words": U.value_words(v), "word_bits": 16,
            "domain": U.registered_domain(win.obj["program"], v, leaves[tree]).domain_id.hex(), "row_schema": U.V2}
        assert e["reads"] == [U.unit_reads(win.program, win.tp, u)[p][1] for u in range(T)]
    assert reads["ports"]["zy"]["words"] == 3 * RC and reads["ports"]["a0"]["words"] == K // 2 + 2

    salt = U.unit_salt(U.stand_in_reveal(b"\x09" * 32), 1)
    served = window("serve", str(tmp_path / "w"), salt.hex(), str(tmp_path / "served.json"), "--workers", "2")
    assert served["z_is_served_product"] == {"cells": T * RC, "equal": T * RC}
    units = list(range(T))
    assert all(RN.check_opening(win, com, served, op, u) == [] for u, op in zip(units, opened(tmp_path, "w", salt, units, "all")))
    with pytest.raises(ValueError, match="power of two"):
        U.Shape(K).window(U.window_shape(com)[0], 6)
    assert [U.slot(n) for n in (1, 4, 5, 8, 9, 16, 17)] == [4, 4, 8, 8, 16, 16, 32]


@pytest.mark.slow
def test_drawn_units_open_to_their_window_and_no_other(tmp_path, win):
    """A drawn unit's opening passes its own window's roots and no other's (a flipped record's, a receipt of another salt's,
    another window's), its ok row is the one word 1 under its public salt, and a prover's rows with a tampered A row
    (`window.py open --tamper-a`, at every unit that reads it) open no registered root through the path they keep."""
    salt = U.unit_salt(U.stand_in_reveal(b"\x01" * 32), 1)
    com, served = committed(tmp_path, "w1", 1, salt, win)
    assert com["tiles"] == 2 * (8 // R) * (16 // C)
    assert served["salt"] == PS.pouw_salt(U.stand_in_reveal(b"\x01" * 32), 1, U.STREAM, U.PROVER).hex()
    units = [0, 5, 7]
    for u, op in zip(units, opened(tmp_path, "w1", salt, units, "honest")):
        assert RN.check_opening(win, com, served, op, u) == []
    rows = np.load(tmp_path / "honest" / "rows.npz")
    assert rows["w_pub"].shape == (3, 100) and rows["s_pub"].shape == (3, U.SALT_BYTES)
    assert rows["w_a0"].shape == (3, K // 2 + 2) and rows["w_zy"].shape == (3, 3 * RC)
    assert rows["r_a0"].tolist() == [0, 0, 4] and rows["r_b0"].tolist() == [0, 24, 24] and rows["r_zy"].tolist() == units
    assert all(np.array_equal(w, U.OK) for w in rows["w_o1"])
    assert [bytes(s) for s in rows["s_o1"]] == [U.ok_salt("w1", u) for u in units]
    assert [reference(rows, i)[1] for i in range(3)] == [1, 1, 1]

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
        assert U.verify_registered(win.obj["program"], f"a/{K}", 8, bytes.fromhex(com["root_a"]), 0, bc, [str(z["p_a0"][0])]) is opens

    for flag, port, tree in (("--zy-word", "zy", "zy"), ("--receipt", "o0", "receipt")):
        served_t, com_t = tampered(tmp_path, "w1", flag, "5")
        assert RN.check_registration(win, com_t) == []
        assert served_t["tampered"]["records" if tree == "zy" else "receipts"] == [5]
        assert (com_t["root_zy"] != com["root_zy"]) is (tree == "zy")
        ops = opened(tmp_path, f"w1-{flag.replace('-', '')}5", salt, units, f"open{flag}")
        assert all(RN.check_opening(win, com_t, served_t, op, u) == [] for u, op in zip(units, ops))
        why = f"{port}: its path does not reach the registered {tree} root"
        assert [RN.check_opening(win, com, served, op, u) for u, op in zip(units, ops)] == [[why]] * len(units)

    salt2 = U.unit_salt(U.stand_in_reveal(b"\x02" * 32), 2)
    com2, served2 = committed(tmp_path, "w2", 2, salt2, win)
    assert RN.check_registration(win, com2) == [] and com2["root_a"] != com["root_a"]
    op2 = opened(tmp_path, "w2", salt2, [5], "other")[0]
    assert RN.check_opening(win, com2, served2, op2, 5) == []
    assert len(RN.check_opening(win, com, served, op2, 5)) == R + C + 1 + 1


@pytest.mark.slow
def test_each_in_proof_negatives_window_changes_its_value_and_ok_drops_where_its_record_moved(tmp_path, win):
    """Each in-proof negative's tampered window (`window.py tamper`, `run.IN_PROOF`) opens to its own registration and
    not the honest one, and changes exactly the value it names: the receipt word 0 plus one, a record's y of cell 0 plus
    one, its z of cell 0 plus one with y that z's dequant, two units' records exchanged, the record of the same tile in the
    other call, a record of z alone, a D_1 word one too large. The unit's word reference (`h1zy.reference`) on its opened
    rows computes ok = 0 exactly at the units whose record moved, and the stager's check (`stage.check_against_reference`)
    refuses the window's committed outputs there and accepts the reference's."""
    salt = U.unit_salt(U.stand_in_reveal(b"\x03" * 32), 1)
    com, served = committed(tmp_path, "w1", 1, salt, win)
    units = [1, 5]
    opened(tmp_path, "w1", salt, units, "honest")
    honest = dict(np.load(tmp_path / "honest" / "rows.npz"))
    zyw = np.load(tmp_path / "w1" / "rows.npz")["zyw"]
    for i in range(2):
        rc, ok = reference(honest, i)
        assert ok == 1 and np.array_equal(U.receipt_words(rc), honest["w_o0"][i])
    ST.check_against_reference(honest, {"o0": honest["w_o0"], "o1": honest["w_o1"]})
    value = lambda w: int(w[0]) | int(w[1]) << 16  # noqa: E731
    c, ti, tj = W.tile_of(com, 5)
    other_call = W.unit_of(com, 1 - c, ti, tj)
    cases = {("--receipt-word", "5"): [], ("--y-cell", "5"): [5], ("--z-cell", "5"): [5], ("--swap", "5", "1"): [1, 5],
             ("--call", "5"): [5], ("--z-only", "5"): [5], ("--d-word", "5"): []}
    for flag, moved in cases.items():
        served_t, com_t = tampered(tmp_path, "w1", *flag)
        assert served_t["tampered"]["records"] == moved and RN.check_registration(win, com_t) == []
        assert served_t["tampered"]["receipts"] == ([5] if flag[0] in ("--receipt-word", "--d-word") else [])
        priv = f"w1-{''.join(flag).replace('-', '')}"
        ops = opened(tmp_path, priv, salt, units, f"open{''.join(flag)}")
        assert all(RN.check_opening(win, com_t, served_t, op, u) == [] for u, op in zip(units, ops))
        assert all(RN.check_opening(win, com, served, op, u) != [] for u, op in zip(units, ops))
        got = dict(np.load(tmp_path / f"open{''.join(flag)}" / "rows.npz"))
        for p in [f"a{i}" for i in range(R)] + [f"b{j}" for j in range(C)] + ["pub", "o1"]:
            assert np.array_equal(got[f"w_{p}"], honest[f"w_{p}"]), (flag, p)
        z0, y0 = U.split_record(honest["w_zy"][1], RC)
        z1, y1 = U.split_record(got["w_zy"][1], RC)
        if flag[0] == "--receipt-word":
            assert value(got["w_o0"][1]) == value(honest["w_o0"][1]) + 1 & 0xFFFFFFFF
            assert np.array_equal(got["w_o0"][1][2:], honest["w_o0"][1][2:]) and np.array_equal(got["w_o0"][0], honest["w_o0"][0])
        elif flag[0] == "--d-word":
            assert not np.array_equal(got["w_o0"][1], honest["w_o0"][1]) and np.array_equal(got["w_o0"][0], honest["w_o0"][0])
        elif flag[0] == "--y-cell":
            assert z1 == z0 and y1[0] == y0[0] + 1 & 0xFFFF and y1[1:] == y0[1:]
        elif flag[0] == "--z-cell":
            _a, _b, sa, sw = W._rows(np.load(tmp_path / "w1" / "rows.npz"), com, 5)
            assert z1[0] == z0[0] + 1 & U.M32 and z1[1:] == z0[1:] and y1[1:] == y0[1:]
            assert y1[0] == Z.dequant(z1[0], f32(sa[0])[0], f32(sw[0])[0])
        elif flag[0] == "--swap":
            assert np.array_equal(got["w_zy"][0], honest["w_zy"][1]) and np.array_equal(got["w_zy"][1], honest["w_zy"][0])
        elif flag[0] == "--call":
            assert np.array_equal(got["w_zy"][1], zyw[other_call]) and not np.array_equal(zyw[other_call], zyw[5])
        elif flag[0] == "--z-only":
            assert z1 == z0 and y1 == [0] * RC
        ok = [reference(got, i)[1] for i in range(2)]
        assert ok == [0 if u in moved else 1 for u in units], (flag, ok)
        rc = [reference(got, i)[0] for i in range(2)]
        outs = {"o0": np.stack([U.receipt_words(r) for r in rc]), "o1": np.array([[v] for v in ok], np.uint16)}
        ST.check_against_reference(got, outs)
        with pytest.raises(AssertionError, match="not h1zy.reference's"):
            ST.check_against_reference(got, {"o0": got["w_o0"], "o1": got["w_o1"]})


def test_the_numpy_records_are_h1zys_words():
    """`h1unit.dequant` and `h1unit.records` (numpy, what `window.py` commits) are `h1zy.dequant` and `h1zy.records` (the
    unit's word reference, `PouwZyDequant_v1`) on random and edge-case z, s_a and s_w: zeros of both signs, |z| at 2^31
    and past 2^24, subnormal, huge, infinite and NaN scales (every NaN 0x7FFF), overflow to infinity and underflow to
    zero; and a row's and a record's words split back to their codes, scale, z and y."""
    rng = np.random.default_rng(11)
    zs = [0, 1, -1, 2**31 - 1, -2**31, 127 * 127 * 64, -(128 * 128 * 64), 0x00FFFFFF, 0x01000001, -0x01000003]
    ss = [0.0, -0.0, 1.0, -1.0, 1e-45, 1e-40, 2.0**-126, 1 / 127, 3.4e38, np.inf, -np.inf, np.nan]
    grid = np.array([(z, a, w) for z in zs for a in ss for w in ss], dtype=object)
    z = np.concatenate([grid[:, 0].astype(np.int64), rng.integers(-2**31, 2**31, 3000)])
    sa = np.concatenate([grid[:, 1].astype(np.float32), scales(rng, 3000) * rng.choice([-1, 1], 3000).astype(np.float32)])
    sw = np.concatenate([grid[:, 2].astype(np.float32), scales(rng, 3000)])
    got = U.dequant(z, sa, sw)
    want = [Z.dequant(int(x) & U.M32, a, w) for x, a, w in zip(z, f32(sa), f32(sw))]
    assert got.tolist() == want
    assert {int(v) for v, x in zip(got, np.isnan(sa) | np.isnan(sw)) if x} == {0x7FFF}

    a, b = rng.integers(-128, 128, (R, K)).astype(np.int8), rng.integers(-128, 128, (C, K)).astype(np.int8)
    sa, sw = scales(rng, R), scales(rng, C)
    zr, yr = U.records(a, b, sa, sw)
    enc = lambda m: [[int(x) & 0xFF for x in r] for r in m]  # noqa: E731
    assert ([int(v) & U.M32 for v in zr.reshape(-1)], yr.reshape(-1).tolist()) == Z.records(enc(a), enc(b), f32(sa), f32(sw))
    words = U.record_words(zr, yr)
    assert words.shape == (3 * RC,) and U.split_record(words, RC) == Z.records(enc(a), enc(b), f32(sa), f32(sw))
    rows = U.row_words(a, sa)
    assert rows.shape == (R, K // 2 + 2)
    assert [U.split_row(r) for r in rows] == [(c, s) for c, s in zip(enc(a), f32(sa))]


def test_the_verifiers_shared_row_file_is_c_flocks_and_a_tampered_a_row_opens_no_root(tmp_path):
    """Three units of a window, units 0 and 2 reading the same A rows and units 0 and 1 the same B rows, each its own
    record: the prover's shared-row statement (`h1unit.statement_tables` through C-Flock's `circuit.write`, as `stage.py`
    stages it) and the verifier's own file (`h1unit.verifier_pub`, from the units' ``b || c``, positions and paths alone)
    are one file byte for byte, every A, B and record table row opening its registered root (`run.registered_opens`,
    flock-verify --registered's check). A prover's A row the registration does not hold opens none, two rows at one
    position are refused, and a forged statement (`h1unit.forge_statement`) is the verifier's file of a window with the
    wrong output row."""
    from verity_flock import circuit as FC
    ports = U.Shape(K).ports()
    names = [p.name for p in ports]
    ins, outs, reg_ports = names[:-2], names[-2:], names[:-3]
    assert reg_ports[-1] == "zy" and ins[-1] == "pub"
    meta = {"ports": [[p.name, p.words, 0] for p in ports], "row_schemas": [[U.V2, p.bits] for p in ports],
            "input_ports": len(ins), "out_ports": [[p.name, p.words] for p in ports[-2:]], "vus_per_block": 1}
    text = "FLOCK\nMETA " + json.dumps(meta) + "\n"
    (tmp_path / "circuit.txt").write_text(text)
    head = {"meta": meta, "pin": "p" * 128, "class": "c" * 128}
    rng = np.random.default_rng(4)
    prog, n, indices, T = "ab" * 64, 3, [3, 9, 12], 16
    codes = {"a": rng.integers(-128, 128, (8, K)).astype(np.int8), "b": rng.integers(-128, 128, (16, K)).astype(np.int8)}
    words = {t: U.row_words(c, scales(rng, len(c))) for t, c in codes.items()}
    words["zy"] = rng.integers(0, 1 << 16, (T, 3 * RC)).astype(np.uint16)
    reg = {t: U.register(prog, U.value(t, K), w, [rng.bytes(U.SALT_BYTES) for _ in w]) for t, w in words.items()}
    assert list(reg["a"].bcs) == U.commit_strings(ports[0], words["a"], list(reg["a"].salts))
    assert list(reg["zy"].bcs) == U.commit_strings(ports[names.index("zy")], words["zy"], list(reg["zy"].salts))
    com = {"unit": {"k": K}, "slot": 8, "b_rows": 16, "tiles": T, **{f"root_{t}": r.root.hex() for t, r in reg.items()}}
    verifier = SimpleNamespace(obj={"program": prog})
    first = {"a": [0, 4, 0], "b": [0, 0, 8], "zy": indices}
    salts_of = lambda s: np.frombuffer(b"".join(s), np.uint8).reshape(len(s), U.SALT_BYTES)  # noqa: E731
    z = {"units": np.array(indices)}
    for p in reg_ports:
        t = U.tree_of(p)
        pos = first[t] if t == "zy" else [f + int(p[1:]) for f in first[t]]
        z.update({f"w_{p}": words[t][pos], f"s_{p}": salts_of([reg[t].salts[q] for q in pos]),
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
        return bcs, [{p: (U.value(U.tree_of(p), K), int(z[f"r_{p}"][i]), str(z[f"p_{p}"][i])) for p in reg_ports}
                     for i in range(n)]

    pub = staged(z, tmp_path / "honest")
    bcs, reads = bcs_reads(z)
    assert U.verifier_pub(head, "s", "sub", bcs, indices, reads) == pub
    h = json.loads(pub[:pub.index(b"\n")])
    assert h["shared_rows"] == {**{p: 2 for p in reg_ports[:-1]}, "zy": 3, "pub": 3, "o0": 3, "o1": 3}
    assert (h["registered"]["a0"]["positions"], h["registered"]["b3"]["positions"], h["registered"]["zy"]["positions"],
            h["units"]["indices"]) == ([0, 4], [3, 11], indices, indices)
    assert h["registered"]["zy"]["value"] == "zy"
    assert all(RN.registered_opens(pub, ports, p, r, verifier, com) for p in reg_ports for r in range(h["shared_rows"][p]))
    with pytest.raises(ValueError, match="ascending"):
        U.verifier_pub(head, "s", "sub", bcs, [9, 3, 12], reads)

    bad, row = dict(z), z["w_a0"][:1].copy()
    row[0, 0] ^= 1
    bad["w_a0"], bad["s_a0"] = z["w_a0"].copy(), z["s_a0"].copy()
    bad["w_a0"][[0, 2]], bad["s_a0"][[0, 2]] = row[0], np.frombuffer(rng.bytes(U.SALT_BYTES), np.uint8)
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
    assert o1.words == len(U.OK) == 1
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

    outs_staged = {"o0": z["w_o0"], "o1": z["w_o1"].copy()}
    rows_w = dict(z, w_o1=z["w_o1"].copy())
    rows_w["w_o1"][1] = wrong
    forged = RN.forge(tmp_path / "honest", tmp_path / "forged2", bad_pub, outs_staged, rows_w, tmp_path / "circuit.txt")
    assert forged == [got] and (tmp_path / "forged2" / "inst.bin").read_bytes() == inst
    with pytest.raises(SystemExit, match="no witness is wrong"):
        RN.forge(tmp_path / "honest", tmp_path / "forged3", pub, outs_staged, z, tmp_path / "circuit.txt")


def test_vllm_quant_puts_each_rows_absmax_at_127_rounding_to_nearest():
    """vLLM's quantizer (and H1's) give each row's codes and its f32 scale, the one the window commits beside them."""
    L = _module("h1_prove_layer", "layer.py")
    x = np.array([[0.5, -2.0, 1.1, 0.26, 9.0], [-4.0, 1.0, 0.016, -0.015, 3.0]], np.float32)
    q, s = L.quant_vllm(x)
    assert q.tolist() == [[7, -28, 16, 4, 127], [-127, 32, 1, 0, 95]]
    assert s.dtype == np.float32 and s.tolist() == [np.float32(9.0) / np.float32(127), np.float32(4.0) / np.float32(127)]
    q, s = L.quant_vllm(x, 3)
    assert q.tolist() == [[32, -127, 70], [-127, 32, 1]]
    assert s.tolist() == [np.float32(2.0) / np.float32(127), np.float32(4.0) / np.float32(127)]
    rng = np.random.default_rng(5)
    q, s = L.quant_vllm(rng.standard_normal((64, 4096)).astype(np.float32), K)
    assert q.shape == (64, K) and q.dtype == np.int8 and s.shape == (64,)
    assert (np.abs(q.astype(int)).max(axis=1) == 127).all() and q.min() >= -127
    qa, sa = L.quant_a(x)
    qw, sw = L.quant_w(x)
    assert sa.tolist() == [np.float32(9.0) / np.float32(126), np.float32(4.0) / np.float32(126)] and qa.max() == 125
    assert sw.tolist() == [np.float32(9.0) / np.float32(110), np.float32(4.0) / np.float32(110)] and qw.max() == 110


def test_v3_serves_a_bt_where_a_is_127_and_an_xa_flip_moves_only_the_record(tmp_path, monkeypatch):
    monkeypatch.setenv("H1_TILE", "v3")
    rng = np.random.default_rng(6)
    a = rng.integers(-127, 128, (8, K)).astype(np.int8)
    a[:, ::5] = 127
    layer(tmp_path / "w.npz", 6, calls=(("q_proj", 16),), a=a)
    window("commit", str(tmp_path / "w.npz"), "w", str(tmp_path / "w"), str(tmp_path / "com.json"), "--program-sha512", "0" * 128)
    com = json.loads((tmp_path / "com.json").read_text())
    salt = U.unit_salt(U.stand_in_reveal(b"\x07" * 32), 1)
    served = window("serve", str(tmp_path / "w"), salt.hex(), str(tmp_path / "served.json"), "--workers", "2")
    assert served["z_is_served_product"] == {"cells": 4 * RC, "equal": 4 * RC}
    assert served["xa"]["tile_rows_with_xa"] == served["xa"]["tile_rows"] == 16
    bad, com_t = tampered(tmp_path, "w", "--xa-flip", "1")
    assert bad["receipt_root"] == served["receipt_root"] and com_t["root_zy"] != com["root_zy"]
    assert bad["tampered"]["records"] == [1] and bad["tampered"]["receipts"] == []
    flip = bad["tampered"]["xa_flip"]
    assert flip["xa_set_in_unit"] > 0 and flip["receipt_is_honest"] and flip["record_words_changed"] > 0


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


def test_the_public_and_ok_rows_are_their_parameters_bits():
    """The public row is the salt's, position's and roots' leaves in order; the ok row is the one word 1, under a salt
    anyone derives from the window and the position."""
    salt, roots = bytes(range(64)), (bytes(64 * [7]), bytes(range(64, 128)))
    w = U.pub_words(salt, 0x0123456789, roots)
    bits = np.unpackbits(w.astype("<u2").view(np.uint8), bitorder="little")
    assert list(bits) == SD.leaves(salt, True) + SD.position_leaves(0x0123456789, True) + SD.leaves(roots[0], True) + \
        SD.leaves(roots[1], True)
    assert U.OK.tolist() == [1] and len(U.ok_salt("w", 3)) == U.SALT_BYTES
    assert len({U.ok_salt("w", 3), U.ok_salt("w", 4), U.ok_salt("v", 3), U.pub_salt("w", 3)}) == 4


def test_the_verifiers_frame_is_c_flocks():
    from verity_flock import circuit as FC
    ports = U.Shape(K).ports()
    rng = np.random.default_rng(3)
    n = 3
    rows = {p.name: rng.integers(0, 1 << 16, (n, p.words)).astype(np.uint16) for p in ports}
    salts = {p.name: [bytes(rng.integers(0, 256, U.SALT_BYTES, dtype=np.uint8)) for _ in range(n)] for p in ports}
    fv, bcs = FC.statement(ports, SimpleNamespace(name="h1-test"), 0, n, rows, salts)
    assert U._frame(ports, "h1-test", n, bcs) == fv
