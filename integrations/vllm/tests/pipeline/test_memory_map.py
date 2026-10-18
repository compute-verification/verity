"""The Commit's memory map (`verity_vllm.pipeline.memory_map`) on CPU: a small Llama's request Program of record in a Build dir
(its ``descriptor.json.gz`` and instance rows), a committed store that serves each identity's words as a function of it and the
request's slots at pages out of first-need order, and the engine's caches in vLLM 0.28's packed page layout.  The map written is
`commit.memory_map.record`'s over the rows' committed words at those pages, the engine holds its blocks, the Program's census and
KV rows are cached beside its descriptor and read back without it, and a run without committed slots, without the engine's caches
or with a descriptor of another Program is refused beside the map, never raised.  GLM-5.3 at TP2 in small: each rank's replay
writes the map of its MLA latent rows from its own Build dir, committed words and caches, at its own path and registration."""
from __future__ import annotations

import gzip
import json
import re
import types
import zlib

import numpy as np
import pytest
import torch

from tests.commit.test_memory_map import dsa_request
from tests.predict.test_predict import TINY, _phi
from tests.program import test_dsa_derive as DSA
from tests.program.test_dsa_derive import served  # noqa: F401  (the DSA stand-in's fixture)
from verity.core.protocols.remote_bandwidth.memory_map import Replay, dumps, parse
from verity.ml.circuits.codec import encode_program, program_digest
from verity_vllm.check.replay.index import ProgramIndex
from verity_vllm.commit import memory_map as MM
from verity_vllm.pipeline import memory_map as PMM
from verity_vllm.pipeline.build import _instance_rows
from verity_vllm.pipeline.tp import rank_replay as RR
from verity_vllm.predict.build import build_program

WIDE = TINY | {"hidden_size": 512, "num_key_value_heads": 4}
LAYERS = [f"model.layers.{i}.self_attn.attn" for i in range(2)]
ARRIVE, PAGES = 2, [6, 2, 4]


class Committed:
    """Committed words keyed by their identity (and `salt`: a TP rank's own), the way `CommittedStore` reads them; `slots[step]` the
    request's `<layer>/slot_mapping` at that step."""

    def __init__(self, slots: dict[int, list[int]], salt: int | None = None):
        self.slots, self.salt = slots, salt

    def entries(self, rid, step, op_path, member):
        n = len(self.slots.get(step, ()))
        return ([{"identity": {"element_range": [0, n]}}] if member == "slot_mapping" and n else []), op_path, 0, []

    def words(self, rid, step, op_path, member, elo, ehi):
        if member == "slot_mapping":
            return np.array(self.slots[step][elo:ehi], dtype=np.int64).view(np.uint64), "int64"
        seed = zlib.crc32(repr((rid, step, op_path, member, self.salt)).encode())
        return ((np.arange(elo, ehi, dtype=np.uint64) * 2654435761 + seed) & 0xFFFF).astype(np.uint16), "bf16"


def _build(d, program):
    """Build dir ``d`` of request Program ``program``: its ``descriptor.json.gz`` and instance rows, as the Commit loads them."""
    d.mkdir(parents=True)
    desc = encode_program(program)
    with gzip.open(d / "descriptor.json.gz", "wt") as f:
        json.dump(desc, f)
    inst = {"schema": "cb-a/instance-sequence/v1", "digest": program_digest(desc), "params": [[n, str(t)] for n, t in program.fn.body.params],
            "rows": list(_instance_rows(program))}
    return {"dir": str(d), "instances": inst}


def _build_dir(path, LP, T):
    r = build_program(_phi(WIDE), "request", LP=LP, T=T)
    return r.program, _build(path / f"build_request_LP{LP}_T{T}", r.program)


def _store(rows: MM.Rows, pages=PAGES, bs=4, salt=None) -> Committed:
    return Committed({ARRIVE + s: [pages[q // bs] * bs + q % bs for q, x in enumerate(rows.layers[0][0]) if x.born == s] for s in range(rows.steps)},
                     salt)


def _words(pi, store):
    """A row's committed words as the map reads them (`serving_rows._operand`): each slice's member words at its step and row."""
    def words(rid, row):
        out = []
        for _, j, lo, hi in row.group:
            pd, member, mlo, mhi, _ = pi.member_at(j, lo, hi)
            w = dict((m, n) for m, n, _ in pd["members"])[member]
            out.append(store.words(rid, ARRIVE + pd["step"], pd["op_path"], member, pd["row"] * w + mlo, pd["row"] * w + mhi)[0])
        return np.concatenate(out)
    return words


def _payload(pi, rows):
    words = _words(pi, Committed({}))
    return lambda rid, row: words(rid, row).reshape(2, 4, 128).transpose(1, 0, 2).tobytes()


def _caches(rows: MM.Rows, payload) -> dict[str, torch.Tensor]:
    """Each layer's cache `[8 pages, 4 heads, 4 slots, 2 * 128]` laid out slot-major (vLLM 0.28's FlashAttention), holding the
    request's packed rows at its pages."""
    out = {}
    for layer, name in enumerate(LAYERS):
        phys = torch.zeros((8, 4, 4, 256), dtype=torch.bfloat16)
        for q, (row,) in enumerate(zip(*rows.layers[layer])):
            phys.view(torch.int16)[PAGES[q // 4], q % 4] = torch.from_numpy(np.frombuffer(payload("r0", row), np.int16).reshape(4, 256).copy())
        out[name] = phys.permute(0, 2, 1, 3)
    return out


def test_the_commits_map_is_the_records_at_the_engines_pages(tmp_path):
    program, lp = _build_dir(tmp_path, 4, 5)
    pi, rows = ProgramIndex(lp["instances"]), MM.rows(program)
    kv, payload = MM.packed(rows), _payload(ProgramIndex(lp["instances"]), rows)
    out = tmp_path / "memory_map.jsonl"
    doc = PMM.write(out, {"path": str(out), "registration": "ab" * 64}, loaded={"r0": lp}, store=_store(rows), arrivals={"r0": ARRIVE},
                    caches=_caches(kv, payload))
    layout = MM.Layout(block_size=4, pages=8)
    head, steps = MM.record({"r0": kv}, {"r0": ARRIVE}, payload, bytes.fromhex("ab" * 64), layout, {"r0": PAGES})
    assert out.read_text() == dumps(head, steps)
    got_head, got = parse(out.read_text().splitlines())
    rp = Replay(got_head)
    for s in got:
        rp.apply(s)
    # positions 0-3 fill page 6 at step 2; 4-5 and 6-7 half page 2, reaching each half's filler at steps 3 and 5 and completing
    # it at 4 and 6; 8 reaches page 4's first half at 7 and leaves it dirty; the last step, 7, reads every written block last
    assert [(s.step, sorted(s.freed), sorted(w.slot for w in s.written)) for s in steps] == [
        (2, [], [12, 13, 28, 29]), (3, [4, 20], []), (4, [], [4, 20]), (5, [5, 21], []), (6, [], [5, 21]),
        (7, [4, 5, 8, 12, 13, 20, 21, 24, 28, 29], [])]
    assert rp.extent == 2 * 8 * 2 + 8
    assert doc["page"] == {"form": "packed", "pages": 8, "block_size": 4, "head_words": 128} and doc["requests"]["r0"]["pages"] == PAGES
    assert (doc["blocks"], doc["written"], doc["freed"], doc["engine"]) == (32, 8, 14, {"compared": 8, "equal": 8})
    assert json.loads((tmp_path / "memory_map.jsonl.json").read_text())["engine"] == doc["engine"]


def test_a_remote_sessions_map_is_written_where_its_encodes_land(tmp_path):
    program, lp = _build_dir(tmp_path, 4, 5)
    rows, out = MM.rows(program), tmp_path / "memory_map.jsonl"
    rec = {"prover": {"kv": {"lifetime": 60}},
           "memory_map": {"first_step": 100, "deferral": {"encodes": [[102, 103, 32, 4], [104, 105, 36, 2], [106, 107, 38, 2]]}}}
    (tmp_path / PMM.SESSION).write_text(json.dumps(rec))
    d = PMM.deferral(tmp_path / PMM.SESSION)
    assert d == MM.Deferral(((102, 103, 32, 4), (104, 105, 36, 2), (106, 107, 38, 2)), 100, 60)
    doc = PMM.write(out, {"path": str(out), "registration": "ab" * 64}, loaded={"r0": lp}, store=_store(rows), arrivals={"r0": ARRIVE},
                    caches=_caches(MM.packed(rows), _payload(ProgramIndex(lp["instances"]), rows)), deferral=d)
    # each append lands a step after it is made: the filler freed where a write first reaches it, page 2's second half, read
    # last at step 7 where its encode lands, written there and freed a line later
    _, steps = parse(out.read_text().splitlines())
    assert [(s.step, sorted(s.freed), sorted(w.slot for w in s.written)) for s in steps] == [
        (2, [12, 13, 28, 29], []), (3, [4, 20], [12, 13, 28, 29]), (5, [5, 21], [4, 20]),
        (7, [4, 8, 12, 13, 20, 24, 28, 29], [5, 21]), (8, [5, 21], [])]
    assert doc["deferral"] == {"first_step": 100, "appends": 3, "lifetime": 60}
    (tmp_path / PMM.SESSION).write_text(json.dumps({"prover": {"kv": {"lifetime": 60}}, "memory_map": None}))
    assert PMM.deferral(tmp_path / PMM.SESSION) is None and PMM.deferral(tmp_path / "none.json") is None


def test_no_page_is_assumed(tmp_path):
    program, lp = _build_dir(tmp_path, 4, 5)
    rows, out = MM.rows(program), tmp_path / "memory_map.jsonl"
    req, kw = {"path": str(out), "registration": "00" * 64}, {"loaded": {"r0": lp}, "arrivals": {"r0": ARRIVE}}
    with pytest.raises(ValueError, match=re.escape("no committed `model.layers.0.self_attn.attn/slot_mapping` of request r0 at step 2")):
        PMM.write(out, req, store=Committed({}), caches=_caches(MM.packed(rows), lambda r, x: bytes(2048)), **kw)
    with pytest.raises(ValueError, match="the Commit attached no KV-slots source"):
        PMM.write(out, req, store=_store(rows), caches=None, **kw)
    assert not out.exists()


def test_the_census_and_kv_rows_are_computed_once_per_program(tmp_path):
    program, lp = _build_dir(tmp_path, 4, 5)
    d, digest = tmp_path / "build_request_LP4_T5", lp["instances"]["digest"]
    out = tmp_path / "memory_map.jsonl"
    rows = MM.rows(program)
    PMM.write(out, {"path": str(out), "registration": "ab" * 64}, loaded={"r0": lp}, store=_store(rows), arrivals={"r0": ARRIVE},
              caches=_caches(MM.packed(rows), _payload(ProgramIndex(lp["instances"]), rows)))
    census = json.loads(gzip.decompress((d / PMM.CENSUS).read_bytes()))
    kv = json.loads(gzip.decompress((d / PMM.KV_ROWS).read_bytes()))
    assert census == MM.census_of(program, digest) and (kv["program"], kv["census"]) == (digest, census["digest"])
    (d / "descriptor.json.gz").unlink()
    assert PMM.of_build(d, digest) == rows
    with pytest.raises(FileNotFoundError):
        PMM.of_build(d, "f" * 128)
    got = PMM.cache(d, program, digest)
    assert got == {"census": census["digest"], "kv_rows": True}, "a Build's result.json holds what its Program determines, no timings"
    assert "FileNotFoundError" in PMM.cache_at_build(tmp_path / "gone", program, digest)["error"], "a census failure decides no Build"


def test_a_descriptor_of_another_program_is_refused_beside_the_map(tmp_path):
    _, lp = _build_dir(tmp_path, 4, 5)
    _, other = _build_dir(tmp_path, 4, 1)
    req = {"path": str(tmp_path / "memory_map.jsonl"), "registration": "00" * 64}
    (tmp_path / "req.json").write_text(json.dumps(req))
    got = PMM.after_commit(str(tmp_path / "req.json"), loaded={"r0": {**lp, "instances": other["instances"]}}, bmap={},
                           committed=None, manifest=None, arrivals={"r0": 0}, caches={LAYERS[0]: torch.zeros(1)})
    assert "error" in got and "descriptor.json.gz is Program" in (tmp_path / "memory_map.jsonl.error.txt").read_text()
    assert not (tmp_path / "memory_map.jsonl").exists()


#: GLM-5.3's MLA at TP2 in small: the DSA stand-in's KV layers, its latent rows (64 bytes) in pages of 64 slots, one block a page
DSA_LAYERS = [f"model.layers.{i}.self_attn.attn" for i in range(2)]
DSA_PAGES = [3, 5]


def _latent_caches(rows: MM.Rows, words) -> dict[str, torch.Tensor]:
    """Each layer's MLA cache `[8 pages, 64 slots, 32]` (bound squeezed of its one head) holding the request's latent rows at its pages."""
    out = {}
    for layer, name in enumerate(DSA_LAYERS):
        phys = torch.zeros((8, 64, 32), dtype=torch.bfloat16)
        for q, row in enumerate(rows.layers[layer][0]):
            phys.view(torch.int16)[DSA_PAGES[q // 64], q % 64] = torch.from_numpy(words("r0", row).view(np.int16).copy())
        out[name] = phys
    return out


def test_each_tp_rank_writes_the_map_of_its_mla_latent_rows(tmp_path, served, monkeypatch):  # noqa: F811
    # GLM-5.3 at TP2 in small: each rank's Build dir holds its request Program (the DSA stand-in: an indexer layer and an IndexShare
    # layer, a prefill of 64 rows, then two decode steps); MLA's latent cache is whole on each rank, the scheduler's pages the same
    # on both, and each rank's committed words its own.  Rank 0's Build has KV rows cached under kv-rows/v1, which refused them
    monkeypatch.setattr(DSA, "MAXPOS", 72)
    program = dsa_request(steps=((0, 64), (64, 1), (65, 1)), topk=72)
    rows = MM.rows(program)
    req = tmp_path / "req.json"
    req.write_text(json.dumps({"path": str(tmp_path / "maps" / "memory_map.jsonl"), "registration": "ab" * 64,
                               "ranks": {"1": {"registration": "cd" * 64}}}))
    for rank in (0, 1):
        lp = _build(tmp_path / f"rank{rank}" / "build_request", program)
        if rank == 0:
            with gzip.open(tmp_path / "rank0" / "build_request" / PMM.KV_ROWS, "wt") as f:
                json.dump({"schema": "verity-vllm/kv-rows/v1", "program": lp["instances"]["digest"], "refused": "no attention Call"}, f)
        store = _store(rows, DSA_PAGES, 64, salt=rank)
        words = _words(ProgramIndex(lp["instances"], tp={"rank": rank, "world": 2}, program_dir=lp["dir"]), store)
        monkeypatch.setattr(PMM.CommittedStore, "of_record", lambda *a, store=store: store)
        st = {"replay": {"memory_map": str(req)}, "bmap": {}, "manifest_doc": None,
              "com": types.SimpleNamespace(sources=[types.SimpleNamespace(short_name="kv_slots", caches=_latent_caches(rows, words))])}
        p = RR.Loaded(rank, 2, [], {"r0": lp}, {"r0": lp["instances"]}, {"r0": lp["instances"]["digest"]}, {"r0": lp["dir"]},
                      {"r0": ARRIVE}, [], None)
        got = RR.memory_map(st, p, committed=None)
        out = tmp_path / "maps" / f"memory_map.rank{rank}.jsonl"
        head, steps = MM.record({"r0": rows}, {"r0": ARRIVE}, lambda r, x, words=words: words(r, x).tobytes(), bytes.fromhex(("ab", "cd")[rank] * 64),
                                MM.Layout(block_size=64, pages=8), {"r0": DSA_PAGES})
        assert out.read_text() == dumps(head, steps)
        # the prefill fills page 3 (a block a layer), read to the last step; the decode rows reach page 5's filler and leave it dirty
        assert [(s.step, list(s.freed), [w.slot for w in s.written]) for s in steps] == [
            (ARRIVE, [], [3, 11]), (ARRIVE + 1, [5, 13], []), (ARRIVE + 2, [3, 11], [])]
        assert got == {"map": out.name, "tp": {"rank": rank, "world": 2}, "blocks": 16, "page": {"form": "latent", "pages": 8, "block_size": 64, "head_words": 32},
                       "steps": 3, "written": 2, "freed": 4, "engine": {"compared": 2, "equal": 2}, "seconds": got["seconds"]}
    assert len({w.digest for r in (0, 1) for s in parse((tmp_path / "maps" / f"memory_map.rank{r}.jsonl").read_text().splitlines())[1]
                for w in s.written}) == 4, "each rank's digests are of its own committed words"
    cached = json.loads(gzip.decompress((tmp_path / "rank0" / "build_request" / PMM.KV_ROWS).read_bytes()))
    assert cached["schema"] == PMM.KV_ROWS_SCHEMA and "refused" not in cached and MM.rows_of(cached) == rows
    assert RR.memory_map({"replay": {}}, p, None) is None
    assert "error" in RR.memory_map(st | {"replay": {"memory_map": str(tmp_path / "none.json")}}, p, None), "a map changes no verdict"
