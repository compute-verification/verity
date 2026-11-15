"""The GLM-5.3 fp8_ds_mla cache-write capture without a GPU: the planted rows and weights reach every corner the verdict needs, as
the Definition computes them; a compare phase over words the Definition wrote finds no row that differs, and finds a flipped cache
byte, a flipped kv_c_out word and a flipped scale; the PTX check takes the two FP8 specialisations by their code (the GPU run is
`tests/properties/glm53_ds_mla_write_capture_gpu.py --phase gpu` then `--phase compare` on an H200)."""
from __future__ import annotations

__models__ = ("zai-org/GLM-5.3",)

import json
from pathlib import Path

import numpy as np

from tests.properties import glm53_ds_mla_write_capture_gpu as X
from verity.circuits.defs import bind
from verity.circuits.evaluation import evaluate
from verity.definitions.registry import dsa_ds_mla as M

ROWS = 24


def _fn():
    return bind(M.DsaLatentRowDsMla, KVL=X.KVL, R=X.R, TILE=X.TILE, EPS=X.EPS)


def _case(seed: int, weight: np.ndarray) -> dict[str, np.ndarray]:
    """ROWS tokens of `tile_rows` with random k_pe and rotary rows, the last one unslotted, and what the Definition writes."""
    rng = np.random.default_rng(seed)
    kv = np.concatenate([X.tile_rows(rng, ROWS), np.stack([X.FR.bf16_row(rng, X.R, 1.0) for _ in range(ROWS)])], axis=1)
    cs = np.stack([X.FR.bf16_row(rng, X.R, 0.5) for _ in range(ROWS)])
    slots = np.arange(ROWS, dtype=np.int64)
    slots[-1] = -1
    fn = _fn()
    refs = [[int(v) for v in evaluate(fn, kv[t].tolist(), weight.tolist(), cs[t].tolist())] for t in range(ROWS)]
    rows = np.stack([X.want_row(r) if slots[t] >= 0 else np.zeros(X.ROW, dtype=np.uint8) for t, r in enumerate(refs)])
    out = {"kv": kv, "w": weight, "cs": cs, "slots": slots}
    for m in X.MODES:
        out |= {f"{m}_kv_out": np.asarray([r[:X.KVL] for r in refs], dtype=np.uint16),
                f"{m}_kpe_out": np.asarray([r[X.KVL:X.KVL + X.R] for r in refs], dtype=np.uint16), f"{m}_rows": rows.copy()}
    return out


def _write(out: Path, cases: dict[str, dict[str, np.ndarray]]) -> None:
    gpu = {"gate": {"ok": True}, "errors": {}, "ptx": {"ok": True}, "cases": []}
    for case, arrays in cases.items():
        paths = X.DS._save(out / "words" / case, **arrays)
        gpu["cases"].append({"case": case, "tokens": ROWS, "slotted": ROWS - 1, "checks": {"modes_equal": True},
                             "paths": {k: str(Path(p).relative_to(out)) for k, p in paths.items()}})
    (out / X.GPU_RECORD).write_text(json.dumps(gpu))


def test_the_planted_rows_reach_every_corner_and_the_definition_writes_its_rows(tmp_path):
    served = np.full(X.KVL, 0x3F80, dtype=np.uint16)
    cases = {"served": _case(1, served), "corner": _case(2, X.corner_weight())}
    rows = np.concatenate([c["indexer_rows"][c["slots"] >= 0] for c in cases.values()])
    got = X.corners(rows)
    assert all(got[k] > 0 for k in X.CORNERS), got
    _write(tmp_path, cases)
    assert X.compare_phase(tmp_path, 1) in (0, 1)
    doc = json.loads((tmp_path / X.RECORD).read_text())
    assert not doc["errors"]
    for case in X.CASES:
        s = doc["summary"][case]
        assert s["tokens"] == ROWS and s["slotted"] == ROWS - 1 and all(v == 0 for v in s["differ"].values()), s


def test_a_flipped_byte_word_or_scale_is_a_row_that_differs(tmp_path):
    case = _case(3, X.corner_weight())
    case["shared_rows"][2, 7] ^= 1
    case["indexer_kv_out"][4, 0] ^= 1
    case["indexer_rows"][5, X.KVL + 1] ^= 0x10
    _write(tmp_path, {"served": case, "corner": _case(4, X.corner_weight())})
    assert X.compare_phase(tmp_path, 1) == 1
    s = json.loads((tmp_path / X.RECORD).read_text())["summary"]["served"]
    assert s["differ"]["shared/rows"] == 1 and s["differ"]["shared/row_bytes"] == 1 and s["differ_bytes"]["shared"] == 1
    assert s["differ"]["indexer/rows"] == 2 and s["differ"]["indexer/kv_out"] == 1 and s["differ"]["indexer/norm"] == 1
    assert s["first"]["shared/row_bytes"] == 2 and s["first"]["indexer/row_bytes"] == 5
    assert all(v == 0 for v in json.loads((tmp_path / X.RECORD).read_text())["summary"]["corner"]["differ"].values())


def test_the_ptx_check_takes_the_fp8_specialisations_by_their_code(tmp_path):
    cache = tmp_path / "cache"
    for name, body in (("A", "\tcvt.rn.satfinite.e4m3x2.f32 %rs1, %r1, %r2;\n"), ("B", "\tdiv.full.f32 %r1, %r2, %r3;\n")):
        (cache / name).mkdir(parents=True)
        (cache / name / f"{X.KERNEL}.ptx").write_text(f".reqntid 128\n{body}\t.file\t1 \"/x/kernels.py\"\n")
    rec = X.ptx(cache, tmp_path / "out")
    assert rec["ok"] is False and len(rec["specialisations"]) == 2
    assert sorted(p.name for p in (tmp_path / "out" / "ptx").iterdir()) == [f"A-{X.KERNEL}.ptx", f"B-{X.KERNEL}.ptx"]
    assert rec["specialisations"][0]["counts"] == {"cvt.rn.satfinite.e4m3x2.f32": 1}
