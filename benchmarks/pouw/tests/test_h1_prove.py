"""h1_prove's window and the verifier's view of it (`h1_prove/window.py`, `run.py`, `h1unit.py`, `stage.py`), without the
unit's circuit: the window's public Program and template-instance partition derive exactly the tiles it commits and the
rows each reads, its registered values (each call's A table and z table, its tiles' z/y records, `scaled_mm_zy`'s, and
the weights' b/<K>) are read where the window's own reads say, every per-window value's domain binds the window's open
root (PRF-127) and b/<K>'s none, window credit's registration step takes the window's evidence and refuses it without
that root, one call's units a statement, a window's drawn units open
to those rows of its registered roots and to nothing else (a tampered window's, another window's, a tampered A row), each
in-proof negative's window changes the one value it names and makes
the unit's word reference compute ok = 0 exactly where its record moved, a registration of z alone is refused, the public
row lays out the unit's salt, position and roots as its parameters read them, and the verifier's shared-row file over the
opened ``b || c`` and paths is Flock's own over the rows."""
from __future__ import annotations

import hashlib
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
    `window.py` commits, serves and opens, numbered as it numbers them, each an instance of the tile unit: unit u, tile t
    of call c, reads A rows R ti .. R ti + R - 1 of call c's A table, its call's B rows C tj .. C tj + C - 1 of b/<K>, row
    t of call c's z table, the salt, position u and the window's roots, and its outputs are row u of the receipt and of
    ok. The slot pads the token rows with zero rows of scale 0, and the public commitment holds no token count. Each
    call's A table is the slot's rows (with their scales) under its own salts and its z table its tiles' records, each
    `scaled_mm_zy`'s (`h1zy.records`), the tables the verifier's Program names; a registration of z alone, or of one
    window-wide zy, is refused; the units' roots are the A tables' digest and b/<K>'s root; and the verifier's
    ``--registered`` file of a call (`h1window.table_reads_file`, from the Program alone) reads, at every unit of the call
    and port, the row `h1unit.unit_reads` does, and none at any other call's unit."""
    layer(tmp_path / "w.npz", 8, rows=6, calls=(("k_proj", 16), ("q_proj", 20)))
    com = commit(tmp_path, tmp_path / "w.npz", "w", win.obj["program"])
    assert (com["slot"], com["b_rows"], [(c["name"], c["n"]) for c in com["calls"]]) == (8, 32, [("k_proj", 16), ("q_proj", 16)])
    assert U.window_shape(com) == SHAPE
    assert not {"rows", "tokens"} & set(com)
    priv = np.load(tmp_path / "w" / "rows.npz")
    assert priv["a"].shape == (8, K) and not priv["a"][6:].any() and not priv["scale_a"][6:].any()
    T = com["tiles"]
    tables = {f"b/{K}": 32, f"a/{K}@0": 8, "zy@0": T // 2, f"a/{K}@1": 8, "zy@1": T // 2}
    assert U.tables(com) == U.program_tables(win.program, K) == tables
    assert [e["value"] for e in com["registered"]] == list(tables)
    assert RN.check_registration(win, com) == []
    assert RN.check_registration(win, dict(com, program_sha512="0" * 128)) != []
    assert RN.check_registration(win, dict(com, registered=com["registered"][:-1])) != []
    wr = U.window_open_root(com)
    assert com["window_root"] == wr == U.open_root("w", bytes.fromhex(U.registered_of(com)[f"b/{K}"][1]))
    assert [e["domain"] for e in com["registered"]] == [
        G.domain(win.obj["program"], v, U.value_words(v), n, window=None if v == f"b/{K}" else wr).domain_id.hex()
        for v, n in tables.items()]
    assert RN.check_registration(win, dict(com, window_root=U.open_root("v", bytes(64)))) != []
    why = RN.check_registration(win, RN.unrooted(com))
    assert len(why) == 1 and why[0].startswith("its registered entries")
    port = {p.name: p for p in U.Shape(K).ports()}
    assert [bytes(x) for x in priv["bcb"]] == U.commit_strings(port["b0"], U.row_words(priv["b"], priv["scale_b"]),
                                                               [bytes(s) for s in priv["sb"]])
    assert [bytes(x) for x in priv["bczy"]] == U.commit_strings(port["zy"], priv["zyw"], [bytes(s) for s in priv["szy"]])
    assert priv["sa"].shape == (2, 8, U.SALT_BYTES) and not (priv["sa"][0] == priv["sa"][1]).all(axis=1).any()
    for c in range(2):
        assert [bytes(x) for x in priv["bca"][c]] == U.commit_strings(port["a0"], U.row_words(priv["a"], priv["scale_a"]),
                                                                      [bytes(s) for s in priv["sa"][c]])
    zonly = G.Registered.commit(com["program_sha512"], "zy@0", priv["zyw"][:T // 2, :2 * RC].tolist(),
                                [bytes(s) for s in priv["szy"][:T // 2]], G.WORD_BITS, U.V2, wr)
    why = RN.check_registration(win, dict(com, registered=[zonly.entry() if e["value"] == "zy@0" else e
                                                           for e in com["registered"]]))
    assert len(why) == 1 and why[0].startswith("its registered entries")
    assert RN.check_registration(win, RN.z_only(com)) != []
    wide = G.Registered.commit(com["program_sha512"], "zy", priv["zyw"].tolist(), [bytes(s) for s in priv["szy"]],
                               G.WORD_BITS, U.V2, wr)
    assert RN.check_registration(win, dict(com, registered=[*(e for e in com["registered"] if e["value"][:3] != "zy@"),
                                                            wide.entry()])) != []
    reg = U.registered_of(com)
    a_roots = b"".join(bytes.fromhex(reg[f"a/{K}@{c}"][1]) for c in range(2))
    assert U.window_roots(com) == (hashlib.sha512(U.A_TABLES_TAG + a_roots).digest(), bytes.fromhex(reg[f"b/{K}"][1]))

    assert T == 2 * (8 // R) * (16 // C) == win.tp.population
    assert win.query["params"]["templates"] == [P.template_instance_query(U.Shape(K).definition())["params"]["template"]]
    assert [(t, first, n) for _k, t, first, n in win.tp.ranges] == [(win.query["params"]["templates"][0], c["tile0"], c["tiles"])
                                                                    for c in com["calls"]]
    assert [n for n, _t in win.program.fn.signature()[0]] == [f"a/{K}", f"b/{K}", "zy", "salt", "position", "roots"]
    assert U.window_ports(win.program) == {**{f"a_k{K}[{i}]": f"a{i}" for i in range(R)},
                                           **{f"b_k{K}[{j}]": f"b{j}" for j in range(C)}, "zy": "zy"}
    for u in range(T):
        c, ti, tj = W.tile_of(com, u)
        t0 = com["calls"][c]["tile0"]
        b0 = com["calls"][c]["b0"] + C * tj
        assert U.HW.call_of(win.program, u) == (c, u - t0)
        assert U.unit_reads(win.program, win.tp, u) == {
            **{f"a{i}": (f"a/{K}@{c}", R * ti + i) for i in range(R)}, **{f"b{j}": (f"b/{K}", b0 + j) for j in range(C)},
            "zy": (f"zy@{c}", u - t0), "salt": ("salt", 0), "position": ("position", u), "roots": ("roots", 0),
            "o0": ("receipt", u), "o1": ("ok", u)}
        rows, cols = slice(R * ti, R * ti + R), slice(b0, b0 + C)
        a, b, sa, sw = priv["a"][rows], priv["b"][cols], priv["scale_a"][rows], priv["scale_b"][cols]
        assert np.array_equal(priv["zyw"][u], U.record_words(*U.records(a, b, sa, sw)))
        if u in (0, T - 1):
            enc = lambda m: [[int(x) & 0xFF for x in r] for r in m]  # noqa: E731
            assert U.split_record(priv["zyw"][u], RC) == Z.records(enc(a), enc(b), f32(sa), f32(sw))

    okr = RN.ok_registration(win, com)
    assert okr.entry() == {"value": "ok", "root": okr.root.hex(), "leaves": T, "words": 1,
                           "domain": G.domain(win.obj["program"], "ok", 1, T, window=wr).domain_id.hex()}
    assert okr.entry()["domain"] != G.domain(win.obj["program"], "ok", 1, T).domain_id.hex()
    assert list(okr.bcs) == U.commit_strings(port["o1"], np.tile(U.OK, (T, 1)), [U.ok_salt("w", u) for u in range(T)])
    assert okr.root == U.register_ok(win.obj["program"], "w", T, wr).root != U.register_ok(win.obj["program"], "v", T, wr).root
    assert okr.root != U.register_ok(win.obj["program"], "w", T, U.open_root("v", bytes(64))).root
    roots = {v: r for v, (_n, r) in reg.items()}
    with pytest.raises(ValueError, match="holds no root of them"):
        U.registered_reads(win.program, win.tp, 0, roots, "w", wr, win.obj["program"])
    roots["ok"], leaves = okr.root.hex(), {**tables, "ok": T}
    docs = {}
    params = {port: key for key, port in U.window_ports(win.program).items() if port != key}
    for c, (first, n) in enumerate(U.HW.call_units(win.program)):
        reads = U.registered_reads(win.program, win.tp, c, roots, "w", wr, win.obj["program"])
        assert sorted(reads["ports"]) == sorted([*U.Shape(K).in_rows, "o1"])
        for p, e in reads["ports"].items():
            v = U.unit_reads(win.program, win.tp, first)[p][0]
            assert e.get(G.PARAM_KEY) == params.get(p), "a renamed A or B row names the parameter it reads"
            assert {x: e[x] for x in e if x not in ("reads", G.READS_FROM, "known", G.PARAM_KEY)} == {
                "value": v, "root": roots[v], "leaves": leaves[v], "words": U.value_words(v), "word_bits": 16,
                "domain": G.domain(win.obj["program"], v, U.value_words(v), leaves[v],
                                   window=None if v == f"b/{K}" else wr).domain_id.hex(), "row_schema": U.V2}
            assert (e[G.READS_FROM], e["reads"]) == (first, [U.unit_reads(win.program, win.tp, u)[p][1]
                                                              for u in range(first, first + n)])
        assert reads["ports"]["zy"]["words"] == 3 * RC and reads["ports"]["a0"]["words"] == K // 2 + 2
        assert reads["ports"]["o1"]["value"] == "ok" and reads["ports"]["o1"]["words"] == 1
        assert reads["format"] == G.READS_V1 and reads["ports"]["o1"]["reads"] == list(range(first, first + n))
        assert [p for p, e in reads["ports"].items() if "known" in e] == ["o1"]
        assert reads["ports"]["o1"]["known"] == U.known_ok("w", T) == {
            "rows": [[1]] * T, "salts": [U.ok_salt("w", u).hex() for u in range(T)]}
        assert U.check_known(win.obj["program"], wr, "o1", reads["ports"]["o1"]) is None
        wrote, doc = RN.registered_file(tmp_path / f"c{c}.registered.json", win, com, c, [first, first + n - 1], okr)
        assert doc == reads and json.loads((tmp_path / f"c{c}.registered.json").read_text()) == reads
        assert wrote["roots"]["ok"] == okr.root.hex() and "o1" in wrote["ports"]
        docs[c] = doc

    # flock-verify holds ok's root to its known rows (Registered.checkKnown): a root with one unit's row 0 is refused, as
    # are known rows of another window's salts, too few of them, or a wider row
    o1 = docs[0]["ports"]["o1"]
    kr = RN.ok_known_root(tmp_path / "ok-known-root.registered.json", win, com, docs[0], 3)
    assert kr["honest"] is None and kr["root"] != okr.root.hex()
    assert kr["refused"] == f"registered reads: port o1: ok's registered root is not the root of its {T} known rows"
    bad = json.loads((tmp_path / "ok-known-root.registered.json").read_text())
    assert bad["ports"]["o1"] == dict(o1, root=kr["root"]) and {p: e for p, e in bad["ports"].items() if p != "o1"} == {
        p: e for p, e in docs[0]["ports"].items() if p != "o1"}
    for known, why in ((U.known_ok("v", T), "is not the root of its"), (U.known_ok("w", T - 1), f"{T - 1} known rows of ok's {T}"),
                       (dict(o1["known"], rows=[[1, 0]] * T), "a known row is not 1 u16 words")):
        assert why in U.check_known(win.obj["program"], wr, "o1", dict(o1, known=known))
    assert "is not the root of its" in U.check_known(win.obj["program"], U.open_root("v", bytes(64)), "o1", o1)

    # window credit's registration step takes the window's evidence: one sampled proof block of each call, its finding its call's
    # --registered ports, under the descriptor whose SHA-512 is the Program's; and refuses each per-window value without
    # the window root, and another window's root
    desc, calls = RN.credit_descriptor(win), U.HW.call_units(win.program)
    assert hashlib.sha512(desc).hexdigest() == win.obj["program"]
    drawn = [calls[0][0] + 1, calls[1][0]]
    found = {u: RN.findings_of(docs[c]) for c, u in enumerate(drawn)}
    assert found[drawn[1]]["o1"] == {"value": "ok", "root": okr.root.hex()}
    entries, ok = com["registered"], {"value": "ok", "root": okr.root.hex()}
    assert RN.credit_registration(desc, drawn, entries, found, wr, ok) is None
    okd = G.domain(win.obj["program"], "ok", 1, T).domain_id.hex()
    assert RN.credit_registration(desc, drawn, [*entries, dict(okr.entry(), domain=okd)], found, wr, ok) is None, (
        "the ok rows are the verifier's own, which credit builds and `_verified` holds each session to: registration "
        "reads no entry of them")
    # ok-unrooted: the sessions opened ok at its rows registered without the window root (the domain a file states
    # without it, which flock-verify roots its known rows at), and credit's step-5 rule refuses them
    oku = RN.unrooted_ok(win, com)
    assert oku.root != okr.root and oku.domain.domain_id.hex() == okd
    assert oku.root == G.Registered.commit(win.obj["program"], "ok", [[1]] * T, [U.ok_salt("w", u) for u in range(T)],
                                           16, U.V2, None).root
    bad_found = {u: dict(f, o1=dict(f["o1"], root=oku.root.hex())) for u, f in found.items()}
    for bad, why in ((bad_found, f"call 0's session of unit {drawn[0]} opened the ok rows at another port, value or root"),
                     ({drawn[0]: found[drawn[0]], drawn[1]: {p: e for p, e in found[drawn[1]].items() if p != "o1"}},
                      f"call 1's session of unit {drawn[1]} opened the ok rows"),
                     ({u: dict(f, o2=f["o1"]) for u, f in found.items()}, "opened the ok rows at another port")):
        got = RN.credit_registration(desc, drawn, entries, bad, wr, ok)
        assert got is not None and got[0] == "registration" and why in got[1], got
    for bad_entries, root, why in (
            (RN.unrooted(com)["registered"], wr, f"registered value a/{K}@0's domain is not"),
            (entries, U.open_root("v", bytes(64)), f"registered value a/{K}@0's domain is not"),
            ([e for e in entries if e["value"] != f"b/{K}"], wr, f"against b/{K}, which the window registers not")):
        got = RN.credit_registration(desc, drawn, bad_entries, found, root, ok)
        assert got is not None and got[0] == "registration" and why in got[1], got
    assert RN.credit_registration(desc, [*drawn, 0], entries, found, wr, ok)[0] == "missing"

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
    """A sampled proof block's opening passes its own window's roots and no other's (a flipped record's, a receipt of another salt's,
    another window's), its ok row is the one word 1 under its public salt and row u of the verifier's registered ``ok``
    (a 0 row there opens no root; ``--unregistered-ok`` states no ok registration), a prover's rows with a tampered A row
    (`window.py open --tamper-a`, at every unit that reads it) open no registered root through the path they keep, and
    units of two calls, which read two calls' tables, are no one statement."""
    salt = U.unit_salt(U.stand_in_reveal(b"\x01" * 32), 1)
    com, served = committed(tmp_path, "w1", 1, salt, win)
    assert com["tiles"] == 2 * (8 // R) * (16 // C)
    assert served["salt"] == PS.pouw_salt(U.stand_in_reveal(b"\x01" * 32), 1, U.STREAM, U.PROVER).hex()
    units = [4, 5, 7]
    for u, op in zip(units, opened(tmp_path, "w1", salt, units, "honest")):
        assert RN.check_opening(win, com, served, op, u) == []
    rows = np.load(tmp_path / "honest" / "rows.npz")
    assert rows["w_pub"].shape == (3, 100) and rows["s_pub"].shape == (3, U.SALT_BYTES)
    assert rows["w_a0"].shape == (3, K // 2 + 2) and rows["w_zy"].shape == (3, 3 * RC)
    assert rows["r_a0"].tolist() == [0, 0, 4] and rows["r_b0"].tolist() == [16, 24, 24] and rows["r_zy"].tolist() == [0, 1, 3]
    assert (rows["v_a0"].tolist(), rows["v_b0"].tolist(), rows["v_zy"].tolist()) == ([f"a/{K}@1"] * 3, [f"b/{K}"] * 3, ["zy@1"] * 3)
    assert all(np.array_equal(w, U.OK) for w in rows["w_o1"])
    assert [bytes(s) for s in rows["s_o1"]] == [U.ok_salt("w1", u) for u in units]
    assert [reference(rows, i)[1] for i in range(3)] == [1, 1, 1]
    prog, T, o1, wr = win.obj["program"], com["tiles"], U.Shape(K).ports()[-1], U.window_open_root(com)
    okr = U.register_ok(prog, "w1", T, wr)
    assert (rows["v_o1"].tolist(), rows["r_o1"].tolist()) == (["ok"] * 3, units)
    for i, u in enumerate(units):
        assert str(rows["p_o1"][i]) == okr.path(u).hex()
        for w, opens in ((rows["w_o1"][i], True), (np.zeros_like(U.OK), False)):
            bc = U.commit_strings(o1, w[None, :], [bytes(rows["s_o1"][i])])[0]
            assert U.verify_registered(prog, "ok", T, okr.root, u, bc, [str(rows["p_o1"][i])], wr) is opens
        assert not U.verify_registered(prog, "ok", T, okr.root, u, U.commit_strings(o1, rows["w_o1"][i][None, :],
                                       [bytes(rows["s_o1"][i])])[0], [str(rows["p_o1"][i])], U.open_root("v", bytes(64)))
    st = U.statement_tables(dict(rows), [*U.Shape(K).in_rows, "pub"], ["o0", "o1"])
    assert st["registered"]["o1"] == {"value": "ok", "positions": units, "paths": [okr.path(u).hex() for u in units]}
    opened(tmp_path, "w1", salt, units, "ok-unregistered", "--unregistered-ok")
    loose = dict(np.load(tmp_path / "ok-unregistered" / "rows.npz"))
    assert not {"v_o1", "r_o1", "p_o1"} & set(loose) and np.array_equal(loose["w_o1"], rows["w_o1"])
    assert "o1" not in U.statement_tables(loose, [*U.Shape(K).in_rows, "pub"], ["o0", "o1"])["registered"]

    a0 = U.Shape(K).ports()[0]
    ops = opened(tmp_path, "w1", salt, units, "tampered", "--tamper-a")
    assert [RN.check_opening(win, com, served, op, u) for u, op in zip(units, ops)] == [[]] * 3
    bad = np.load(tmp_path / "tampered" / "rows.npz")
    assert [int(np.count_nonzero(x != y)) for x, y in zip(bad["w_a0"], rows["w_a0"])] == [1, 1, 0]
    assert bytes(bad["s_a0"][0]) == bytes(bad["s_a0"][1]) != bytes(rows["s_a0"][0])
    assert np.array_equal(bad["w_a1"], rows["w_a1"]) and np.array_equal(bad["p_a0"], rows["p_a0"])
    st = U.statement_tables(dict(bad), [*U.Shape(K).in_rows, "pub"], ["o0", "o1"])
    assert st["registered"]["a0"]["positions"] == [0, 4] and st["registered"]["a0"]["value"] == f"a/{K}@1"
    reg = U.registered_of(com)
    for z, opens in ((rows, True), (bad, False)):
        bc = U.commit_strings(a0, z["w_a0"][:1], [bytes(z["s_a0"][0])])[0]
        assert U.verify_registered(win.obj["program"], f"a/{K}@1", 8, bytes.fromhex(reg[f"a/{K}@1"][1]), 0, bc,
                                   [str(z["p_a0"][0])], wr) is opens
    opened(tmp_path, "w1", salt, [0, 5], "two-calls")
    with pytest.raises(ValueError, match="one call's units a statement"):
        U.statement_tables(dict(np.load(tmp_path / "two-calls" / "rows.npz")), [*U.Shape(K).in_rows, "pub"], ["o0", "o1"])

    for flag, port, value in (("--zy-word", "zy", "zy@1"), ("--receipt", "o0", "receipt")):
        served_t, com_t = tampered(tmp_path, "w1", flag, "5")
        assert RN.check_registration(win, com_t) == []
        assert served_t["tampered"]["records" if port == "zy" else "receipts"] == [5]
        assert served_t["tampered"]["tables"] == ([value] if port == "zy" else [])
        reg_t = U.registered_of(com_t)
        assert [v for v in reg if reg_t[v] != reg[v]] == ([value] if port == "zy" else [])
        ops = opened(tmp_path, f"w1-{flag.replace('-', '')}5", salt, units, f"open{flag}")
        assert all(RN.check_opening(win, com_t, served_t, op, u) == [] for u, op in zip(units, ops))
        why = f"{port}: its path does not reach the registered {value} root"
        assert [RN.check_opening(win, com, served, op, u) for u, op in zip(units, ops)] == [[why]] * len(units)

    salt2 = U.unit_salt(U.stand_in_reveal(b"\x02" * 32), 2)
    com2, served2 = committed(tmp_path, "w2", 2, salt2, win)
    assert RN.check_registration(win, com2) == [] and U.window_roots(com2)[0] != U.window_roots(com)[0]
    op2 = opened(tmp_path, "w2", salt2, [5], "other")[0]
    assert RN.check_opening(win, com2, served2, op2, 5) == []
    assert len(RN.check_opening(win, com, served, op2, 5)) == R + C + 1 + 1


@pytest.mark.slow
def test_each_in_proof_negatives_window_changes_its_value_and_ok_drops_where_its_record_moved(tmp_path, win):
    """Each in-proof negative's tampered window (`window.py tamper`, `run.IN_PROOF`) opens to its own registration and
    not the honest one, and changes exactly the value it names: the receipt word 0 plus one, a record's y of cell 0 plus
    one, its z of cell 0 plus one with y that z's dequant, two units' records exchanged, the record of the same tile in the
    other call, a record of z alone, a D_1 word one too large, a changed record re-registering its call's z table alone.
    The unit's word reference (`h1zy.reference`) on its opened
    rows computes ok = 0 exactly at the units whose record moved, and the stager's check (`stage.check_against_reference`)
    refuses the window's committed outputs there and accepts the reference's."""
    salt = U.unit_salt(U.stand_in_reveal(b"\x03" * 32), 1)
    com, served = committed(tmp_path, "w1", 1, salt, win)
    units = [4, 5]
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
    cases = {("--receipt-word", "5"): [], ("--y-cell", "5"): [5], ("--z-cell", "5"): [5], ("--swap", "5", "4"): [4, 5],
             ("--call", "5"): [5], ("--z-only", "5"): [5], ("--d-word", "5"): []}
    for flag, moved in cases.items():
        served_t, com_t = tampered(tmp_path, "w1", *flag)
        assert served_t["tampered"]["records"] == moved and RN.check_registration(win, com_t) == []
        assert served_t["tampered"]["tables"] == (["zy@1"] if moved else [])
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


def test_the_h1_statement_is_flat_unless_h1_tmpl_says_structural(monkeypatch):
    """`h1unit.layout`, the form every H1 stager takes (`h1unit.stager`): unset, empty or ``flat`` the flat statement, the
    one the Lean verifier accepts; ``structural`` the templated statement at `h1unit.LAYOUT`, another mode the templated
    one at it, and anything else refused by name."""
    monkeypatch.delenv("H1_TMPL", raising=False)
    assert U.layout() is None
    for value, want in (("", None), ("flat", None), ("structural", U.LAYOUT), ("iv/hybrid/4", "iv/hybrid/4"), ("iv", "iv")):
        monkeypatch.setenv("H1_TMPL", value)
        assert U.layout() == want
    monkeypatch.setenv("H1_TMPL", "hybrid/4")
    with pytest.raises(ValueError, match="'hybrid/4' is not a layout mode"):
        U.layout()


def test_a_unit_struct_lower_refuses_is_staged_flat(monkeypatch, capsys):
    """`h1unit.stager`: with a layout, the templated statement of the unit's own structural template; with none, the flat
    statement, which evaluates the unit alike; and where struct_lower refuses the Definition (its NotImplementedError, as
    for assertions or table reads), the flat statement, said on stderr."""
    from verity.ml.circuits.boolean import BIT
    from verity.ml.circuits.boolean.forms import add
    from verity.ml.circuits.boolean.trace import definition
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.types import Array
    from verity_flock import circuit as FC
    from verity_flock import struct_lower as SL

    d = bind(definition("H1StagerToy", 1, lambda C, a, b, w: add(C, a, b, width=20, carry_out=True) + [C.AND(w[0], w[2047]) ^ w[1000]],
                        (("a", Array(12, BIT)), ("b", Array(20, BIT)), ("w", Array(2048, BIT))), Array(22, BIT), register=False))
    rows, outs = [("ab", U.V2, 32, ["a", "b"]), ("w", U.V2, 2048, ["w"])], [(U.V2, (0,))]
    tmpl, flat = U.stager(d, "unit/toy", rows, outs, U.LAYOUT), U.stager(d, "unit/toy", rows, outs, None)
    assert hasattr(tmpl.low, "template") and tmpl.meta["statement"] == FC.TMPL_STATEMENT and tmpl.meta["layout"] == U.LAYOUT
    assert not hasattr(flat.low, "template") and flat.meta["statement"] == FC.STATEMENT
    x = np.random.default_rng(3).integers(0, 1 << 16, size=(5, 2 + 128), dtype=np.uint64)
    assert np.array_equal(tmpl.low.evaluate(x)[0], flat.low.evaluate(x)[0])

    def refuse(definition, **_):
        raise NotImplementedError(f"{definition.id}: assertions and table reads are not laid out by the structural lowering yet")

    monkeypatch.setattr(SL, "lower", refuse)
    capsys.readouterr()
    fell = U.stager(d, "unit/toy", rows, outs, U.LAYOUT)
    assert not hasattr(fell.low, "template") and fell.pin == flat.pin
    assert "the flat statement: struct_lower refuses H1StagerToy" in capsys.readouterr().err


def test_ks_statement_rows_are_the_h1_tiles_rows_under_either_ports():
    """`h1window.statement_rows`, the rows the auditor's lowering writes K's circuit file on: under h1_prove's ports
    (`h1unit.ports`) the stager's rows as `Shape` sizes them (an A or B row K / 2 + 2 words, the record 3 R C, the public
    row 100), and by default `window_ports`' A and B ports, which credit's registered file reads, with the rest unchanged."""
    from verity.core.protocols.pouw.circuit import h1window as HW
    for shape in (U.Shape(64), U.Shape(128)):
        t = shape.definition()
        bits = {p.name: p.bits for p in shape.ports()}
        reads = {port: key for key, port in U.ports(t).items()}
        want = [(p, U.V2, bits[p], [reads[p]]) for p in shape.in_rows] + [("pub", U.V2, U.PUB_BITS, list(U.PUBLIC))]
        assert HW.statement_rows(t, U.ports(t)) == want
        named = {key: port for key, port in HW.window_ports(shape.window([("q", 2 * shape.c)], 2 * shape.r)[0]).items()}
        assert HW.statement_rows(t) == [(named.get(ps[0], p), s, b, ps) for p, s, b, ps in want]
        assert [p for p, *_ in HW.statement_rows(t)][:2] == [f"a_k{shape.k}_0", f"a_k{shape.k}_1"]
    assert HW.STATEMENT_OUT_ROWS == ((U.V2, (0,)), (U.V2, (1,)))


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
    record: the prover's shared-row statement (`h1unit.statement_tables` through Flock's `circuit.write`, as `stage.py`
    stages it) and the verifier's own file (`h1unit.verifier_pub`, from the units' ``b || c``, positions and paths alone)
    are one file byte for byte, every A, B, record and ok table row opening its registered root (`run.registered_opens`,
    flock-verify --registered's check; ok's the verifier's own, `h1unit.register_ok`). An ok row of 0 opens none, ok rows
    committed afresh state no ok registration, a prover's A row the registration does not hold opens none, two rows at one
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
    values = {"a": f"a/{K}@0", "b": f"b/{K}", "zy": "zy@0"}
    wr = U.open_root("w", bytes(range(64)))
    reg = {t: U.register(prog, values[t], w, [rng.bytes(U.SALT_BYTES) for _ in w], wr) for t, w in words.items()}
    assert list(reg["a"].bcs) == U.commit_strings(ports[0], words["a"], list(reg["a"].salts))
    assert list(reg["zy"].bcs) == U.commit_strings(ports[names.index("zy")], words["zy"], list(reg["zy"].salts))
    com = {"unit": {"k": K}, "registered": [r.entry() for r in reg.values()]}
    verifier = SimpleNamespace(obj={"program": prog})
    first = {"a": [0, 4, 0], "b": [0, 0, 8], "zy": indices}
    salts_of = lambda s: np.frombuffer(b"".join(s), np.uint8).reshape(len(s), U.SALT_BYTES)  # noqa: E731
    tree_of = lambda p: "zy" if p == "zy" else p[0]  # noqa: E731
    z = {"units": np.array(indices)}
    for p in reg_ports:
        t = tree_of(p)
        pos = first[t] if t == "zy" else [f + int(p[1:]) for f in first[t]]
        z.update({f"w_{p}": words[t][pos], f"s_{p}": salts_of([reg[t].salts[q] for q in pos]), f"v_{p}": np.array([values[t]] * n),
                  f"r_{p}": np.array(pos), f"p_{p}": np.array([reg[t].path(q).hex() for q in pos])})
    for p in ["pub", "o0"]:
        port = ports[names.index(p)]
        z[f"w_{p}"] = rng.integers(0, 1 << 16, (n, port.words)).astype(np.uint16)
        z[f"s_{p}"] = salts_of([rng.bytes(U.SALT_BYTES) for _ in range(n)])
    okr = U.register_ok(prog, "w", T, wr)
    z.update({"w_o1": np.tile(U.OK, (n, 1)), "s_o1": salts_of([U.ok_salt("w", u) for u in indices]),
              "v_o1": np.array(["ok"] * n), "r_o1": np.array(indices), "p_o1": np.array([okr.path(u).hex() for u in indices])})
    stated = [*reg_ports, "o1"]
    roots = U.registered_of(com) | {"ok": (T, okr.root.hex())}

    def staged(z: dict, d: Path) -> bytes:
        st = U.statement_tables(z, ins, outs)
        d.mkdir()
        FC.write(text, SimpleNamespace(name="s", subcircuit=SimpleNamespace(id="sub")), 0, n, d / "inst.bin", d / "pub.bin",
                 None, indices, share_rows=True, outs={p: z[f"w_{p}"] for p in outs}, tables=st["tables"], refs=st["refs"],
                 salts=st["salts"], registered=st["registered"], digests=(head["pin"], head["class"]))
        return (d / "pub.bin").read_bytes()

    def bcs_reads(z: dict) -> tuple[list, list]:
        bcs = [{p: U.commit_strings(ports[names.index(p)], z[f"w_{p}"][i:i + 1], [bytes(z[f"s_{p}"][i])])[0] for p in names}
               for i in range(n)]
        return bcs, [{p: (str(z[f"v_{p}"][i]), int(z[f"r_{p}"][i]), str(z[f"p_{p}"][i])) for p in stated if f"r_{p}" in z}
                     for i in range(n)]

    pub = staged(z, tmp_path / "honest")
    bcs, reads = bcs_reads(z)
    assert U.verifier_pub(head, "s", "sub", bcs, indices, reads) == pub
    h = json.loads(pub[:pub.index(b"\n")])
    assert h["shared_rows"] == {**{p: 2 for p in reg_ports[:-1]}, "zy": 3, "pub": 3, "o0": 3, "o1": 3}
    assert (h["registered"]["a0"]["positions"], h["registered"]["b3"]["positions"], h["registered"]["zy"]["positions"],
            h["units"]["indices"]) == ([0, 4], [3, 11], indices, indices)
    assert (h["registered"]["zy"]["value"], h["registered"]["a0"]["value"]) == ("zy@0", f"a/{K}@0")
    assert h["registered"]["o1"] == {"value": "ok", "positions": indices, "paths": [okr.path(u).hex() for u in indices]}
    assert all(RN.registered_opens(pub, ports, p, r, verifier, roots, wr) for p in stated for r in range(h["shared_rows"][p]))
    elsewhere = U.open_root("v", bytes(range(64)))
    assert [p for p in stated if RN.registered_opens(pub, ports, p, 0, verifier, roots, elsewhere)] == [f"b{j}" for j in range(C)]
    zero = dict(z, w_o1=z["w_o1"].copy())
    zero["w_o1"][1] = 0
    pub_z = staged(zero, tmp_path / "ok-zero")
    assert U.verifier_pub(head, "s", "sub", *bcs_reads(zero)[:1], indices, bcs_reads(zero)[1]) == pub_z
    assert [RN.registered_opens(pub_z, ports, "o1", r, verifier, roots, wr) for r in range(n)] == [True, False, True]
    loose = {x: v for x, v in z.items() if x not in ("v_o1", "r_o1", "p_o1")}
    pub_l = staged(loose, tmp_path / "ok-unregistered")
    assert "o1" not in json.loads(pub_l[:pub_l.index(b"\n")])["registered"]
    assert U.verifier_pub(head, "s", "sub", *bcs_reads(loose)[:1], indices, bcs_reads(loose)[1]) == pub_l
    with pytest.raises(ValueError, match="ascending"):
        U.verifier_pub(head, "s", "sub", bcs, [9, 3, 12], reads)
    other = [dict(r, a0=(f"a/{K}@1", *r["a0"][1:])) if i == 2 else r for i, r in enumerate(reads)]
    with pytest.raises(ValueError, match="one call's units a statement"):
        U.verifier_pub(head, "s", "sub", bcs, indices, other)
    with pytest.raises(ValueError, match="one call's units a statement"):
        U.statement_tables(dict(z, v_a0=np.array([f"a/{K}@0", f"a/{K}@0", f"a/{K}@1"])), ins, outs)

    bad, row = dict(z), z["w_a0"][:1].copy()
    row[0, 0] ^= 1
    bad["w_a0"], bad["s_a0"] = z["w_a0"].copy(), z["s_a0"].copy()
    bad["w_a0"][[0, 2]], bad["s_a0"][[0, 2]] = row[0], np.frombuffer(rng.bytes(U.SALT_BYTES), np.uint8)
    pub_t = staged(bad, tmp_path / "tampered")
    assert U.verifier_pub(head, "s", "sub", *bcs_reads(bad)[:1], indices, bcs_reads(bad)[1]) == pub_t
    assert [RN.registered_opens(pub_t, ports, p, r, verifier, roots, wr) for p, r in (("a0", 0), ("a0", 1), ("a1", 0))] == [False, True, True]
    half = dict(bad, w_a0=bad["w_a0"].copy(), s_a0=bad["s_a0"].copy())
    half["w_a0"][2], half["s_a0"][2] = z["w_a0"][2], z["s_a0"][2]
    with pytest.raises(ValueError, match="instances 0 and 2 read position 0 as two rows"):
        U.statement_tables(half, ins, outs)
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
    """v3 serves z = A Bᵀ exactly where A rows hold 127, and `window.py tamper --xa-flip` flips one entry's X wrap bit xa
    in a unit's record: the record moves, its receipt doesn't. Which entries set xa follows the dither bits, which come
    from roots `window.py commit` blinds with os.urandom, so a tile row sets none with probability about 2^-13: the test
    asserts only that some row does, and the flipped unit's four rows all missing is about 2^-52."""
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
    assert 0 < served["xa"]["tile_rows_with_xa"] <= served["xa"]["tile_rows"] == 16
    bad, com_t = tampered(tmp_path, "w", "--xa-flip", "1")
    assert bad["receipt_root"] == served["receipt_root"] and U.registered_of(com_t)["zy@0"] != U.registered_of(com)["zy@0"]
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


def test_the_window_open_root_is_pouws_of_the_weights_and_a_stand_in_request_table():
    """The window's open root is `pouw.salt.window_open_root` of root(b/<K>), the stand-in request table's root (SHA-512
    of the window's name under its tag) and the window's name as the session; another name or weights root gives another.
    Every registered value but the weights' b/<K> binds it, ``ok`` included (`credit.is_held`)."""
    b = bytes(range(64))
    requests = hashlib.sha512(b"verity/pouw/h1-prove/stand-in-requests/v0\0h1-w1").digest()
    root = U.open_root("h1-w1", b)
    assert root == PS.window_open_root(b, requests, "h1-w1").hex() == hashlib.sha512(
        b"verity/pouw-window-open/v1" + b + requests + hashlib.sha512(b"h1-w1").digest()).hexdigest()
    assert len({root, U.open_root("h1-w2", b), U.open_root("h1-w1", bytes(64))}) == 3
    com = {"window": "h1-w1", "unit": {"k": K}, "registered": [{"value": f"b/{K}", "root": b.hex(), "leaves": 8}]}
    assert U.window_open_root(com) == root
    assert [U.bound_root(v, root) for v in (f"b/{K}", f"a/{K}@0", "zy@1", "ok")] == [None, root, root, root]


def test_the_verifiers_frame_is_c_flocks():
    from verity_flock import circuit as FC
    ports = U.Shape(K).ports()
    rng = np.random.default_rng(3)
    n = 3
    rows = {p.name: rng.integers(0, 1 << 16, (n, p.words)).astype(np.uint16) for p in ports}
    salts = {p.name: [bytes(rng.integers(0, 256, U.SALT_BYTES, dtype=np.uint8)) for _ in range(n)] for p in ports}
    fv, bcs = FC.statement(ports, SimpleNamespace(name="h1-test"), 0, n, rows, salts)
    assert U._frame(ports, "h1-test", n, bcs) == fv
