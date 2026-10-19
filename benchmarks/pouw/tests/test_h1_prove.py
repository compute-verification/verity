"""h1_prove's window and the verifier's view of it (`h1_prove/window.py`, `run.py`, `h1unit.py`), without the unit's circuit:
a window's drawn units open to its registered roots and to nothing else (a tampered window's, another window's), the public
row lays out the unit's salt, position and roots as its parameters read them, and the verifier's frame-v3 roots over the
opened ``b || c`` are C-Flock's own over the rows."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

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


def test_drawn_units_open_to_their_window_and_no_other(tmp_path):
    salt = U.unit_salt(b"\x01" * 32)
    com, served = committed(tmp_path, "w1", 1, salt)
    assert com["tiles"] == 2 * (8 // 4) * (16 // 8)
    units = [0, 5, 7]
    for u, op in zip(units, opened(tmp_path, "w1", salt, units, "honest")):
        assert RN.check_opening(com, served, op, u) == []
    rows = np.load(tmp_path / "honest" / "rows.npz")
    assert rows["w_pub"].shape == (3, 100) and rows["s_pub"].shape == (3, U.SALT_BYTES)

    for flag, port in (("--z", "o0"), ("--receipt", "o1")):
        bad = window("tamper", str(tmp_path / "w1"), str(tmp_path / f"w1{flag}"), str(tmp_path / f"t{flag}.json"), flag, "5")
        ops = opened(tmp_path, f"w1{flag}", salt, units, f"open{flag}")
        assert all(RN.check_opening(com, bad, op, u) == [] for u, op in zip(units, ops))
        why = f"{port}: its path does not reach the registered {'z' if port == 'o0' else 'receipt'} root"
        assert [RN.check_opening(com, served, op, u) for u, op in zip(units, ops)] == [[why]] * len(units)

    salt2 = U.unit_salt(b"\x02" * 32)
    com2, served2 = committed(tmp_path, "w2", 2, salt2)
    op2 = opened(tmp_path, "w2", salt2, [5], "other")[0]
    assert RN.check_opening(com2, served2, op2, 5) == []
    assert len(RN.check_opening(com, served, op2, 5)) == 4 + 8 + 2


def test_an_in_proof_negatives_window_opens_to_its_own_registration_with_one_value_plus_one(tmp_path):
    salt = U.unit_salt(b"\x03" * 32)
    com, served = committed(tmp_path, "w1", 1, salt)
    units = [1, 5]
    opened(tmp_path, "w1", salt, units, "honest")
    honest = np.load(tmp_path / "honest" / "rows.npz")
    for flag, port in (("--z-cell", "o0"), ("--receipt-word", "o1")):
        bad = window("tamper", str(tmp_path / "w1"), str(tmp_path / f"w1{flag}"), str(tmp_path / f"t{flag}.json"), flag, "5")
        ops = opened(tmp_path, f"w1{flag}", salt, units, f"open{flag}")
        assert all(RN.check_opening(com, bad, op, u) == [] for u, op in zip(units, ops))
        assert all(RN.check_opening(com, served, op, u) != [] for u, op in zip(units, ops))
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
    pub = U.verifier_pub(head, "s", "sub", [{p.name: bcs[p.name][i] for p in ports} for i in range(n)])
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
    bad_pub = U.verifier_pub(head, "s", "sub", bad)
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
