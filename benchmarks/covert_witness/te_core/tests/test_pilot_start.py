"""At a v3 pin, covert_witness's launch.py refuses a start that the pin's starts_store.json records with a tokenizer other
than data v3's (the pilot's FineWeb-Edu starts record none), with the fix in the message, and te_core.policy's v3
tokenizer is data_v3's."""
import importlib.util
import json
import re
from pathlib import Path

import pytest

from te_core import policy

CW = Path(__file__).resolve().parents[2]
FIX = "v3 branches start from the v3 ladder's starts; FineWeb-Edu starts are v2.3-pilot"


def launch():
    spec = importlib.util.spec_from_file_location("cw_launch_pilot", CW / "launch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def source(tmp_path, starts: dict) -> Path:
    cw = tmp_path / "benchmarks" / "covert_witness"
    cw.mkdir(parents=True)
    (cw / "starts_store.json").write_text(json.dumps({"starts": {n: {"shape": s} for n, s in starts.items()}}))
    (cw / "data_v3.py").write_text(f'TOKENIZER = "{policy.V3_TOKENIZER}"\n')
    return tmp_path


def test_v3_tokenizer_is_data_v3s():
    assert re.search(r'^TOKENIZER = "([^"]+)"', (CW / "data_v3.py").read_text(), re.M).group(1) == policy.V3_TOKENIZER


def test_a_v3_pin_refuses_a_pilot_start_with_the_fix(tmp_path):
    refuse = launch().refuse_pilot_start
    src = source(tmp_path, {"s50": {"vocab": 50304}, "v3-s": {"vocab": 32000, "tokenizer": policy.V3_TOKENIZER}})
    with pytest.raises(SystemExit, match=re.escape(FIX) + r".*\(v3-s\)"):
        refuse("s50", {"data": "v3"}, src)
    refuse("v3-s", {"data": "v3"}, src)
    refuse("pythia160m", {"data": "v3"}, src)
    refuse("s50", {"data": "v2.2"}, src)
    only_pilot = source(tmp_path / "pilot", {"s50": {"vocab": 50304}})
    with pytest.raises(SystemExit, match="none published yet"):
        refuse("s50", {"data": "v3"}, only_pilot)
