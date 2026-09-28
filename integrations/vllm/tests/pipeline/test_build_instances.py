"""The Build's instance sequence (`instances.json.gz`) is written a chunk of rows at a time with `json.dumps`, and its text is
`json.dump`'s of the whole document, whatever the chunk size."""
from __future__ import annotations

import io
import json

import pytest

from verity_vllm.pipeline.build import _write_instances

HEAD = {"schema": "cb-a/instance-sequence/v1", "digest": "d" * 64, "correspondence_digest": None, "params": [["prompt", "Array<8,Value<32>>"]]}
ROWS = [{"i": k, "spec": "Gemm_v1{K=64,N=8}", "form": "call", "name": f"model.layers.0.q_proj/mm\u00e9[{k}]",
         "args": [[["n", k, 0, 64]], [["p", 1, 64 * k, 64 * k + 64], ["n", 3, 0, 1]]]} for k in range(1000)]


@pytest.mark.parametrize("rows", [ROWS, [], ROWS[:1]])
@pytest.mark.parametrize("chunk", [1, 7, 4096])
def test_the_streamed_instance_sequence_is_json_dumps_text(rows, chunk):
    streamed, whole = io.StringIO(), io.StringIO()
    _write_instances(streamed, HEAD, iter(rows), chunk=chunk)
    json.dump({**HEAD, "rows": rows}, whole)
    assert streamed.getvalue() == whole.getvalue()
