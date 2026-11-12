"""A Build on `ir="boolean"` records where its bits came from: `boolean.derived`'s provenance (the word root and digest it lifted,
its σ-check) reaches result.json's `program`, the instances header and `build_summary.json`; a word Build's records gain no key."""
from __future__ import annotations

import dataclasses
import gzip
import json
import os

from tests.pipeline import tp_toy as T
from tests.pipeline.test_tp_build_relabel import SAMPLING, _res
from verity_vllm.pipeline import build as B
from verity_vllm.pipeline import row_records
from verity_vllm.program.frontend.target_profile import TargetProfile

IR = {"ir": "boolean", "word_root": "w" * 64, "word_digest": "d" * 128, "sigma_check": {"nodes": 3, "definitions": 2, "ok": True}}


def _write(out: str, rep, record: list, model_root: str) -> dict:
    os.makedirs(out)
    res = _res(2, 0, record, "request")
    B._write_program(out, res, rep, target=TargetProfile.default(), applied={}, sampling=SAMPLING, launch=None, tp=2, rank=0, model_root=model_root)
    json.dump(res, open(f"{out}/result.json", "w"), default=str)
    return res


def _header(out: str) -> dict:
    with gzip.open(f"{out}/instances.json.gz", "rt") as f:
        return json.loads(f.readline())


def test_a_boolean_builds_records_carry_its_word_provenance_and_a_word_builds_dont(tmp_path):
    rep, mod, record, inner = T.derive_rank(2, 0)
    root = next(n for n, m in mod.named_modules() if m is inner)
    word = _write(f"{tmp_path}/word/build_request", rep, record, root)
    lifted = _write(f"{tmp_path}/boolean/build_request", dataclasses.replace(rep, provenance={**rep.provenance, "ir": IR}), record, root)
    assert "ir" not in word["program"] and "ir" not in _header(f"{tmp_path}/word/build_request")
    assert lifted["program"]["ir"] == IR and _header(f"{tmp_path}/boolean/build_request")["ir"] == IR
    assert {k: v for k, v in lifted["program"].items() if k != "ir"} == word["program"], "the ir record is the only difference"
    os.makedirs(f"{tmp_path}/boolean/build_request_LP4_T2")
    json.dump({"ok": True, "program": {"digest": "e" * 128, "ir": IR}}, open(f"{tmp_path}/boolean/build_request_LP4_T2/result.json", "w"))
    row_records.build_summary(f"{tmp_path}/boolean")
    row_records.build_summary(f"{tmp_path}/word")
    lifted_bs, word_bs = (json.load(open(f"{tmp_path}/{d}/build_summary.json")) for d in ("boolean", "word"))
    assert lifted_bs["request"]["ir"] == IR and lifted_bs["build_request_LP4_T2"]["ir"] == IR and "ir" not in lifted_bs["step"]
    assert "ir" not in word_bs["request"] and "ir" not in word_bs["step"]
