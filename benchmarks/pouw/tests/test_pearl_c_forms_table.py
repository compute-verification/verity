"""`forms_table.py`: which hashing forms each shape ships, from run.py's timing rows (GPU 1's forms fill)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "pearl_c"
spec = importlib.util.spec_from_file_location("forms_table", HERE / "forms_table.py")
FT = importlib.util.module_from_spec(spec)
spec.loader.exec_module(FT)


def chunk(path: Path, uuid: str, shape, variant: str, forms: dict, epi: dict | None = None, ok: bool = True, bad: int = 0) -> None:
    lines = [{"kind": "env", "uuid": uuid},
             {"kind": "check", "variants": {variant: {"pearl_c_ok": ok}}},
             {"kind": "timing", "shape": shape, "m_real": shape[0], "variant": variant, "gates": {"pearl_c": True},
              "consistency": {"x": bad}, "call_e2e_ms_by_form": forms, "call_e2e_ms_by_epilogue": epi}]
    path.write_text("".join(json.dumps(x) + "\n" for x in lines))


def test_the_fastest_pair_ships_ties_go_by_preference_and_one_die_is_provisional(tmp_path):
    forms = {"w,nodes/split": 1.0, "s,one/split": 0.803, "p,fold/fold": 0.800, "p,one/split": 0.9}
    for die, d in (("GPU-a", "a"), ("GPU-b", "b")):
        (tmp_path / d).mkdir()
        chunk(tmp_path / d / "c1.jsonl", die, [32, 8192, 8192], "sm120", forms)
        chunk(tmp_path / d / "c2.jsonl", die, [8192, 8192, 8192], "sm120-hot", {"w,nodes/split": 3.0, "p,fold/fold": 2.4,
                                                                                "s,one/split": 2.5}, {"b": 2.6, "w": 2.5})
    chunk(tmp_path / "a" / "c3.jsonl", "GPU-a", [2048, 2048, 2048], "sm120", {"w,nodes/split": 1.0, "p,one/split": 0.999},
          {"b": 1.0, "w": 0.998})
    chunk(tmp_path / "b" / "c4.jsonl", "GPU-b", [4096, 4096, 4096], "sm120", forms, bad=1)
    t = FT.table(sorted(str(p) for p in tmp_path.glob("*/*.jsonl")))
    d = t["32x8192x8192"]["sm120"]
    assert d["pick"] == "s,one/split" and d["stands"] and d["dies"] == 2 and d["vs_default"] == -0.197
    h = t["8192x8192x8192"]["sm120-hot"]
    assert h["pick"] == "p,fold/fold" and h["epilogue"] == "w" and h["stands"]
    s = t["2048x2048x2048"]["sm120"]
    assert s["pick"] == "w,nodes/split" and s["epilogue"] == "b" and not s["stands"]
    assert not t["4096x4096x4096"]["sm120"]["stands"]
