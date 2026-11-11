"""A request Build's census and KV rows (`verity_vllm.pipeline.memory_map`) on CPU: a small Llama's request Program's census/v1
and KV rows are written beside its descriptor, each naming the Program by digest, and a census failure decides no Build.  A
warm request's rows are cut by its Build's ``prefix_cache.json`` and refused without it."""
from __future__ import annotations

import gzip
import json

from verity.circuits.codec import encode_program, program_digest

from tests.commit.test_memory_map import PREFIX, WIDE, warm_program
from tests.program.predict.test_predict import _phi
from verity_vllm.commit import memory_map as MM
from verity_vllm.pipeline import memory_map as PMM
from verity_vllm.program.predict.build import build_program


def _read(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def test_the_census_and_kv_rows_are_written_beside_the_descriptor(tmp_path):
    program = build_program(_phi(WIDE), "request", LP=4, T=5).program
    digest = program_digest(encode_program(program))
    got = PMM.cache(tmp_path, program, digest)
    census, kv = _read(tmp_path / PMM.CENSUS), _read(tmp_path / PMM.KV_ROWS)
    assert census == MM.census_of(program, digest)
    assert (kv["schema"], kv["program"], kv["census"]) == (PMM.KV_ROWS_SCHEMA, digest, census["digest"])
    assert MM.rows_of(kv) == MM.rows(program)
    assert got == {"census": census["digest"], "kv_rows": True}, "a Build's result.json holds what its Program determines, no timings"
    assert PMM.census_entry(tmp_path, program, digest) == got
    assert "FileNotFoundError" in PMM.census_entry(tmp_path / "gone", program, digest)["error"], "a census failure decides no Build"


def test_a_warm_builds_kv_rows_are_cut_by_its_prefix_cache(tmp_path):
    program, digest = warm_program(8, 4, 4), "1" * 128
    assert PMM.cache(tmp_path, program, digest)["kv_rows"] is False
    assert "kc reads an input of the Program" in _read(tmp_path / PMM.KV_ROWS)["refused"]
    (tmp_path / "prefix_cache.json").write_text(json.dumps(PREFIX))
    assert PMM.cache(tmp_path, program, digest)["kv_rows"] is True
    rows = MM.rows_of(_read(tmp_path / PMM.KV_ROWS))
    assert rows == MM.rows(program, prefix=PREFIX) and rows.cached == 4
