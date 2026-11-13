"""benchmarks/pouw/pearl_c4/real.py, the verifier on a `pearlc_capture.py` directory, on a synthetic one-linear capture: the
stand-in split removes the massive-activation channel that fails D-NF, a chunk out of budget exits 99 with its tasks'
checkpoints, and the next run finishes the tile's replayed inert work under its bound."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import real as R
import real_table as RT

NAME, K, M, N = "model.layers.0.self_attn.q_proj", 1024, 128, 128


def bf16(a: np.ndarray) -> np.ndarray:
    """float32 values rounded to BF16 (nearest even), as BF16 codes."""
    bits = a.astype(np.float32).view(np.uint32).astype(np.uint64)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def capture(root: Path) -> Path:
    rng = np.random.default_rng(3)
    x = bf16(rng.standard_normal((M, K)))
    x[:, 5] = bf16(np.full(M, 2.0 ** 16))                    # a massive activation at an unsampled position
    captured, snap = root / "captured", root / "snap"
    captured.mkdir()
    snap.mkdir()
    np.save(captured / "L000-attn.npy", (x.astype(np.uint32) << 16).view(np.float32))
    w = bf16(rng.standard_normal((N, K)) * 0.02).tobytes()
    header = json.dumps({NAME + ".weight": {"dtype": "BF16", "shape": [N, K], "data_offsets": [0, len(w)]}}).encode()
    (snap / "model.safetensors").write_bytes(len(header).to_bytes(8, "little") + header + w)
    (snap / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {NAME + ".weight": "model.safetensors"}}))
    entry = {"file": "L000-attn.npy", "sha256": hashlib.sha256((captured / "L000-attn.npy").read_bytes()).hexdigest(),
             "shape": [M, K]}
    (captured / "manifest.json").write_text(json.dumps({"model": "synthetic", "revision": "0", "snapshot": str(snap),
                                                        "inputs": {NAME: entry}}))
    return captured


@pytest.fixture(scope="module")
def replayed(tmp_path_factory):
    """The capture directory's replay, out of budget after one task (its exit code and checkpoints) and then resumed: the
    capture, the output directory, and what the first run left there."""
    root = tmp_path_factory.mktemp("replayed")
    captured, out = capture(root), root / "out"
    args = [str(captured), str(out), "--tiles", "1", "--workers", "2"]
    rc = R.main(args + ["--budget-min", "0"])
    first = {"rc": rc, "tasks": len(list(out.glob("admit-*.json")) + list(out.glob("tile-*.json"))),
             "split": json.loads((out / "split-L000-attn.json").read_text())}
    assert R.main(args) == 0
    return captured, out, first


@pytest.mark.slow
def test_real_activation_replay(tmp_path, replayed):
    """The replay end to end on a capture: the split, a budget exit, the resumed run's inert work and the table's row,
    which the table's own test (on a summary) can't reach."""
    captured, out, first = replayed
    assert first["rc"] == 99                      # one task started, then out of budget
    assert first["tasks"] == 1
    split = first["split"]
    assert len(split["side"]) == K // 100 and 5 in split["side"]
    s = json.loads((out / "summary.json").read_text())
    x, w = s["inputs"]["L000-attn.npy"], s["weights"][NAME]
    assert x["rows"] == M and x["dnf_fail_full"] == M and x["dnf_fail_split"] == 0 and x["non_finite"] == 0
    assert w["rows"] == N and w["dnf_fail_split"] == 0
    (t,) = s["tiles"][NAME]
    assert t["rejected"] == "" and t["credited"] == [64, 64] and t["k_kept"] == K - K // 100 and t["k_unit"] == K
    assert s["tiles_debited"] == 1 and s["over_cap_tiles"] == 0 and 0 <= t["ratio"] < 1
    assert {"forming_elements", "screened_only_elements", "salt_dead_a", "unneeded_positions"} <= set(t["debit"]["counts"])
    (row,) = RT.rows([s], RT.Fraction(1, 1000))
    assert row["x_rows"] == M and row["x_dnf_fail_split"] == 0 and row["w_rows"] == N and row["debited"] == 1
    assert abs(sum(row["mean_share"].values()) - t["ratio"]) < 1e-9 and "flag" not in row
    assert row["macs"] == M * N * (K - K // 100) and RT.mac_shares([row])["synthetic"]["over_bound"] == 0
    elsewhere = s | {"capture": str(tmp_path / "not-here")}
    assert RT.rows([elsewhere], RT.Fraction(1, 1000))[0]["x_rows"] is None
    assert RT.rows([elsewhere], RT.Fraction(1, 1000), captured / "manifest.json")[0]["x_rows"] == M
    assert NAME in RT.markdown([row], RT.Fraction(1, 1000))


def test_the_table_flags_the_first_and_last_down_proj_and_counts_both_bounds():
    """A tile's bound scales with ρ: at ρ = 1/1,000 it is over when its ratio at 1/400 exceeds 0.4; a model's share over
    the bound weighs each linear's fraction of tiles over it by its MACs."""
    def tile(ratio: float) -> dict:
        items = {term: "0" for term in RT.TERMS} | {"forming": str(RT.Fraction(ratio).limit_denominator(10 ** 6)),
                                                    "counts": {}}
        return {"ratio": ratio, "cap": "1", "debit": items, "rejected": "", "m": 256, "n": 1024, "k_kept": 8111}

    names = [f"model.layers.{i}.mlp.down_proj" for i in (0, 8, 72)] + ["model.layers.0.self_attn.q_proj"]
    s = {"capture": "x.npz", "model": "m", "inputs": {}, "weights": {},
         "tiles": {n: [tile(0.3), tile(0.5 if "72" in n else 0.1)] for n in names}}
    table = RT.rows([s], RT.Fraction(1, 1000))
    rows = {r["linear"]: r for r in table}
    assert rows[names[0]]["flag"] == "first down_proj" and rows[names[2]]["flag"] == "last down_proj"
    assert "flag" not in rows[names[1]] and "flag" not in rows[names[3]]
    assert rows[names[2]]["over_bound"] == 0 and rows[names[2]]["over_bound_tight"] == 1 and rows[names[0]]["over_bound_tight"] == 0
    share = RT.mac_shares(table)["m"]
    assert share["macs"] == 4 * 256 * 1024 * 8111 and share["over_bound"] == 0 and share["over_bound_tight"] == 1 / 8
    text = RT.markdown(table, RT.Fraction(1, 1000))
    assert "12.50% at 1/1000" in text and "| Over the bound at 1/400 | Over the bound at 1/1000 |" in text


@pytest.mark.slow
def test_an_npz_capture_is_read_as_its_linears(tmp_path, replayed):
    """`fp4_v3_real.capture`'s npz (``layers.<i>.<linear>.x`` and ``.w``, float32 holding BF16): the same admission,
    split and tile as the capture directory gives (`replayed`), keyed by the capture's own sha256. Only a replay of each
    form shows that the npz reader feeds the replay the directory's rows."""
    captured, out_dir, _first = replayed
    x = np.load(captured / "L000-attn.npy")
    snap = json.loads((captured / "manifest.json").read_text())["snapshot"]
    w = (R.w_codes(Path(snap), NAME).astype(np.uint32) << 16).view(np.float32)
    npz = tmp_path / "Qwen--synthetic.npz"
    np.savez(npz, **{NAME[len("model."):] + ".x": x, NAME[len("model."):] + ".w": w})
    out_npz = tmp_path / "out-npz"
    assert R.main([str(npz), str(out_npz), "--tiles", "1", "--workers", "2"]) == 0
    a, b = (json.loads((o / "summary.json").read_text()) for o in (out_dir, out_npz))
    assert b["manifest_sha256"] == hashlib.sha256(npz.read_bytes()).hexdigest() and b["model"] == npz.name
    assert b["inputs"][NAME + ".npy"] == a["inputs"]["L000-attn.npy"] and b["weights"] == a["weights"]
    untimed = lambda s: {n: [{k: v for k, v in t.items() if k != "seconds"} for t in ts] for n, ts in s["tiles"].items()}  # noqa: E731
    assert untimed(b) == untimed(a)


@pytest.mark.slow
def test_started_by_its_command_line_the_fast_f1_is_the_file_beside_it(tmp_path):
    """`real.py CAPTURE OUT --f1 fast` by its command line: a worker's `tile` imports `f1_fast` from the script's own
    directory, which a started script has first on its path, and the tile says its check was the fast one. Only a start
    by the command line shows it: in this process the suite's path holds the directory."""
    captured, out = capture(tmp_path), tmp_path / "out"
    p = subprocess.run([sys.executable, R.__file__, str(captured), str(out), "--tiles", "1", "--workers", "2", "--f1", "fast"],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    (tile,) = json.loads((out / "summary.json").read_text())["tiles"][NAME]
    assert tile["f1"] == "fast" and tile["rejected"] == ""
