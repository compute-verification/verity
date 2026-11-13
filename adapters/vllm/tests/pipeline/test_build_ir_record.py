"""A Build on `ir="boolean"` records where its bits came from: `boolean.derived`'s provenance (the word root and digest it lifted,
its σ-check) reaches result.json's `program`, the instances header and `build_summary.json`; a word Build's records gain no key.
And the Program writer of `verity_vllm.pipeline.build` writes the cached prefix a derive read as `prefix_cache.json`."""
from __future__ import annotations

import dataclasses
import gzip
import json
import os
from pathlib import Path

from tests.pipeline import tp_toy as T
from tests.pipeline.test_tp_build_relabel import SAMPLING, _res
from verity_vllm.pipeline import build as B
from verity_vllm.pipeline import row_records
from verity_vllm.program.frontend.rules.vllm_bindings import kv_prefix
from verity_vllm.program.frontend.target_profile import TargetProfile

IR = {"ir": "boolean", "word_root": "w" * 64, "word_digest": "d" * 128, "sigma_check": {"nodes": 3, "definitions": 2, "ok": True}}


def _write(out: str, rep, record: list, model_root: str) -> dict:
    os.makedirs(out)
    res = _res(2, 0, record, "request")
    B._write_program(out, res, rep, target=TargetProfile.default(), applied={}, sampling=SAMPLING, launch=None, tp=2, rank=0, model_root=model_root)
    Path(f"{out}/result.json").write_text(json.dumps(res, default=str))
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
    Path(f"{tmp_path}/boolean/build_request_LP4_T2/result.json").write_text(json.dumps({"ok": True, "program": {"digest": "e" * 128, "ir": IR}}))
    row_records.build_summary(f"{tmp_path}/boolean")
    row_records.build_summary(f"{tmp_path}/word")
    lifted_bs, word_bs = (json.loads(Path(f"{tmp_path}/{d}/build_summary.json").read_text()) for d in ("boolean", "word"))
    assert lifted_bs["request"]["ir"] == IR and lifted_bs["build_request_LP4_T2"]["ir"] == IR and "ir" not in lifted_bs["step"]
    assert "ir" not in word_bs["request"] and "ir" not in word_bs["step"]


def test_a_cached_prefix_s_layout_is_written_beside_the_program_and_a_derive_without_one_writes_none(tmp_path):
    """What Match reads of a cached prefix: the layout `kv_prefix.layout_of` finds in the derive's provenance is `prefix_cache.json`
    and the result's `prefix`; a derive that read no prefix gets neither the file nor the key."""
    rep, mod, record, inner = T.derive_rank(2, 0)
    root = next(n for n, m in mod.named_modules() if m is inner)
    prefix = {"C": 4, "row_leaves": 2, "layers": ["layers.0.self_attn.attn"]}
    apps = [{**rep.provenance["applications"][0], "cached_prefix": prefix}, *rep.provenance["applications"][1:]]
    cached = dataclasses.replace(rep, provenance={**rep.provenance, "applications": apps})
    layout = kv_prefix.layout_of(cached.provenance)
    assert layout is not None and {k: layout[k] for k in prefix} == prefix
    res = _write(f"{tmp_path}/cached", cached, record, root)
    assert res["prefix"] == layout == json.loads(Path(f"{tmp_path}/cached/prefix_cache.json").read_text())
    plain = _write(f"{tmp_path}/plain", rep, record, root)
    assert "prefix" not in plain and not os.path.exists(f"{tmp_path}/plain/prefix_cache.json")
    assert {k: v for k, v in res.items() if k != "prefix"} == plain, "the prefix record is the only difference"
