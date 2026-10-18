"""The Commit's memory map (`verity_vllm.pipeline.memory_map`) on CPU: a small Llama's request Program of record in a Build dir
(its ``descriptor.json.gz`` and instance rows), and a committed store that serves each identity's words as a function of it.
The map written is `commit.memory_map.record`'s over the rows' committed words, and a Build dir whose descriptor is another
Program's is refused beside the map, never raised."""
from __future__ import annotations

import gzip
import json
import zlib

import numpy as np

from tests.predict.test_predict import TINY, _phi
from verity.core.protocols.remote_bandwidth.memory_map import Replay, dumps, parse
from verity.ml.circuits.codec import encode_program
from verity_vllm.check.replay.index import ProgramIndex
from verity_vllm.commit import memory_map as MM
from verity_vllm.pipeline import memory_map as PMM
from verity_vllm.pipeline.build import _instance_rows
from verity_vllm.predict.build import build_program

WIDE = TINY | {"hidden_size": 512, "num_key_value_heads": 4}


class Committed:
    """Committed words keyed by their identity, the way `CommittedStore.words` reads them."""

    def words(self, rid, step, op_path, member, elo, ehi):
        seed = zlib.crc32(repr((rid, step, op_path, member)).encode())
        return ((np.arange(elo, ehi, dtype=np.uint64) * 2654435761 + seed) & 0xFFFF).astype(np.uint16), "bf16"


def _build_dir(path, LP, T):
    r = build_program(_phi(WIDE), "request", LP=LP, T=T)
    d = path / f"build_request_LP{LP}_T{T}"
    d.mkdir()
    with gzip.open(d / "descriptor.json.gz", "wt") as f:
        json.dump(encode_program(r.program), f)
    inst = {"schema": "cb-a/instance-sequence/v1", "digest": r.digest, "params": [[n, str(t)] for n, t in r.program.fn.body.params],
            "rows": list(_instance_rows(r.program))}
    return r.program, {"dir": str(d), "instances": inst}


def test_the_commits_map_is_the_records_over_its_committed_words(tmp_path):
    program, lp = _build_dir(tmp_path, 4, 5)
    out = tmp_path / "memory_map.jsonl"
    req = {"path": str(out), "registration": "ab" * 64, "block_size": 4, "pages": 8}
    PMM.write(out, req, loaded={"r0": lp}, store=Committed(), arrivals={"r0": 2})
    pi, rows = ProgramIndex(lp["instances"]), MM.rows(program)

    def payload(rid, row):
        (_, j, lo, hi), = row.group
        pd, member, mlo, mhi, _ = pi.member_at(j, lo, hi)
        w = dict((m, n) for m, n, _ in pd["members"])[member]
        return Committed().words(rid, 2 + pd["step"], pd["op_path"], member, pd["row"] * w + mlo, pd["row"] * w + mhi)[0].tobytes()

    head, steps = MM.record({"r0": rows}, {"r0": 2}, payload, bytes.fromhex("ab" * 64), MM.Layout(block_size=4, pages=8))
    assert out.read_text() == dumps(head, steps)
    got_head, got = parse(out.read_text().splitlines())
    rp = Replay(got_head)
    for s in got:
        rp.apply(s)
    assert [s.step for s in steps] == [2, 6] and rp.extent == 2 * 8 * 2 + 8
    layout = json.loads((tmp_path / "memory_map.jsonl.json").read_text())
    assert (layout["blocks"], layout["written"], layout["freed"], layout["requests"]["r0"]["positions"]) == (32, 8, 0, 9)


def test_a_descriptor_of_another_program_is_refused_beside_the_map(tmp_path):
    _, lp = _build_dir(tmp_path, 4, 5)
    _, other = _build_dir(tmp_path, 4, 1)
    req = {"path": str(tmp_path / "memory_map.jsonl"), "registration": "00" * 64}
    (tmp_path / "req.json").write_text(json.dumps(req))
    got = PMM.after_commit(str(tmp_path / "req.json"), loaded={"r0": {**lp, "instances": other["instances"]}}, bmap={},
                           committed=None, manifest=None, arrivals={"r0": 0})
    assert "error" in got and "descriptor.json.gz is Program" in (tmp_path / "memory_map.jsonl.error.txt").read_text()
    assert not (tmp_path / "memory_map.jsonl").exists()
