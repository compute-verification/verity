"""port_trees.py on a pass written by Pearl+'s own committers: the three trees are the manifest's under Pearl+'s domains
and the registered values' under theirs, a changed salt is caught on Pearl+'s side, and the result holds no salt."""
from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path

import numpy as np
from verity.protocols.resource_accounting.compute.pouw.prover.serving.pearl_c import served_y as SY
from verity.protocols.resource_accounting.compute.pouw.verifier.audit import (HM96, HM96_SALT, activation_binding,
                                                                             commit_rows, weight_binding)

ROOT = Path(__file__).resolve().parents[4]
_spec = importlib.util.spec_from_file_location("port_trees", ROOT / "benchmarks/kernels/port_trees/port_trees.py")
PT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PT)

K, N, M16, REAL, INDEX, WEIGHT = 64, 128, 32, 27, 7, 3


def _bf16(rng: random.Random, rows: int, words: int, zero_from: int | None = None) -> bytes:
    a = np.array([rng.randrange(0x3000, 0x4800) | rng.choice((0, 0x8000)) for _ in range(rows * words)], "<u2")
    a = a.reshape(rows, words)
    if zero_from is not None:
        a[zero_from:] = 0
    return a.tobytes()


def _write(d: Path, name: str, data: bytes) -> dict:
    (d / name).parent.mkdir(parents=True, exist_ok=True)
    (d / name).write_bytes(data)
    return {"path": name, "bytes": len(data)}


def _pass(d: Path) -> list[bytes]:
    rng = random.Random(0)
    salt = rng.randbytes(32)
    salts = {t: [rng.randbytes(HM96_SALT) for _ in range(c)] for t, c in (("a", M16), ("y", REAL), ("b", N))}
    a, y, b = _bf16(rng, M16, K, REAL), _bf16(rng, REAL, N), _bf16(rng, N, K)
    root_a = commit_rows(activation_binding(salt, INDEX), INDEX, SY.f32_rows(a, K), HM96, salts["a"]).commitment.root
    root_b = commit_rows(weight_binding(WEIGHT), -2, SY.f32_rows(b, K), HM96, salts["b"]).commitment.root
    root_y = SY.commit_y(y, N, salt, INDEX, salts["y"])[0]
    call = {"index": INDEX, "k": K, "n": N, "m": M16, "real_rows": REAL, "weight": WEIGHT, "root_a": root_a.hex(),
            "rows": _write(d, f"calls/{INDEX}.rows.bf16", a), "salts": _write(d, f"calls/{INDEX}.salts", b"".join(salts["a"])),
            "y": {"root": root_y.hex(), "rows": _write(d, f"y/{INDEX}.rows.bf16", y),
                  "salts": _write(d, f"y/{INDEX}.salts", b"".join(salts["y"]))}}
    wt = {"root": root_b.hex(), "rows": _write(d, f"weights/{WEIGHT}.rows.bf16", b),
          "salts": _write(d, f"weights/{WEIGHT}.salts", b"".join(salts["b"]))}
    (d / "manifest.json").write_text(json.dumps({"salt": salt.hex(), "matmuls": [call], "weights": {str(WEIGHT): wt}}))
    return [salt, *salts["a"], *salts["y"], *salts["b"]]


def test_device_trees_are_the_port_trees(tmp_path):
    _pass(tmp_path)
    got = PT.check(tmp_path)
    assert got["ok"] and got["stand_in"] == {"program": True, "window": True}
    for t in got["trees"].values():
        assert t["pearl_is_manifest"] and t["device_is_port"]
        assert t["pearl_root"] != t["registered_root"]
    assert [got["trees"][x]["leaves"] for x in "ayb"] == [M16, REAL, N]


def test_a_changed_salt_fails_on_pearls_side(tmp_path):
    _pass(tmp_path)
    f = tmp_path / f"y/{INDEX}.salts"
    data = bytearray(f.read_bytes())
    data[0] ^= 1
    f.write_bytes(bytes(data))
    out = tmp_path / "out.json"
    assert PT.main([str(tmp_path), "--out", str(out)]) == 1
    got = json.loads(out.read_text())
    assert not got["ok"] and not got["trees"]["y"]["pearl_is_manifest"] and got["trees"]["y"]["device_is_port"]
    assert got["trees"]["a"]["pearl_is_manifest"] and got["trees"]["b"]["pearl_is_manifest"]


def test_the_result_holds_no_salt(tmp_path):
    secrets = _pass(tmp_path)
    out = tmp_path / "out.json"
    assert PT.main([str(tmp_path), "--out", str(out), "--program", "ab" * 64, "--window", "cd" * 64]) == 0
    text = out.read_text()
    assert not any(s.hex() in text for s in secrets)
    assert json.loads(text)["stand_in"] == {"program": False, "window": False}
