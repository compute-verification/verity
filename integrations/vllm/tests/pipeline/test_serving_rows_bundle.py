"""The serving rows of a deferred Commit (`pipeline.deferred_rows`, ``verity-vllm serving-rows``), end to end on CPU: a Program
with RoPE_v1 rows (D=64) executed honestly, its members committed by the production CPU committer and sealed into a replay bundle with
its registered weights, then the verifier's partition (every RoPE head) read back through the bundle opened against the Commit's record.
The registration passes `registration.check` against the verifier's own copies (its population program, the run root of record, the
public bodies), every hidden row is the served value, and a partition digest or a binding map that is not the record's is refused."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from verity.ml.circuits.defs import REGISTRY, bind
from verity.core.protocols.sampled_proofs.one_stage import draw as DR, partition as P, registration as R
from verity_vllm.check.replay.index import ProgramIndex
from verity_vllm.commit import scheme
from verity_vllm.commit import serving_rows as CS
from verity_vllm.pipeline import cli
from verity_vllm.pipeline import cpu_replay as CPU
from verity_vllm.pipeline import replay_bundle as RB
from verity_vllm.pipeline import deferred_rows as DRW
from verity_vllm.program.registry import b1  # noqa: F401  (registers RoPEHead_v1)

D = 64
ROPE = f"RoPEHead_v1{{D={D}}}"
LAW = "subset:1"
CHUNK = 64
HEADS = 2 * 4 * (3 + 1)       # 2 requests x 4 tokens (2 prefill + 2 decode) x (3 query + 1 key heads)


def _commit(tmp_path: Path, monkeypatch) -> dict:
    """The deferred Commit's output dir and row dir, and the world that served it."""
    pytest.importorskip("torch")
    import tests.check.test_sampled_replay as T
    from tests.check.opened import commit_members
    from verity_vllm.commit import store_dump as SD

    monkeypatch.setattr(T, "DH", D)
    rng = np.random.default_rng(17)
    inst, woff = T._program(rope=True)
    w = T._bf16(rng, max(hi for _lo, hi in woff.values()), 0.5)
    for f in ("model.model.layers.0.input_layernorm.weight", "model.model.norm.weight"):
        w[woff[f][0]:woff[f][1]] = 0x3F80
    lo, hi = woff["model.model.layers.0.self_attn.rotary_emb.cos_sin_cache"]
    w[lo:hi] = (np.asarray(rng.uniform(-1, 1, hi - lo), dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)
    prompts = {"r0": [1, 2], "r1": [5, 3]}
    world = T.Honest(inst, woff, w, prompts)
    bmap, _ = world.bmap_and_reader()
    com, rc = commit_members(dict(sorted(world.store.items())))

    out, row_dir = tmp_path / "commit", tmp_path / "row"
    (row_dir / "build_request").mkdir(parents=True)
    with gzip.open(row_dir / "build_request" / "instances.json.gz", "wt") as f:
        json.dump(inst, f)
    (row_dir / "build_request" / "result.json").write_text(json.dumps({"program_digest": inst["digest"]}))
    p = RB.partial(out, 1)
    SD.dump(com, rc, p / "store")
    prefix = ProgramIndex(inst).prefix
    (p / "weights" / "extra").mkdir(parents=True)
    names, roots, extra = [], [], {}
    for i, (field, (flo, fhi)) in enumerate(woff.items()):
        raw = np.ascontiguousarray(w[flo:fhi]).view(np.uint8)
        name = field[len(prefix):]
        names.append([name, "bfloat16", [fhi - flo], int(raw.size)])
        roots.append(CPU.tensor_root(raw, CHUNK))
        (p / "weights" / "extra" / f"{i}.bin").write_bytes(raw.tobytes())
        extra[name] = i
    geo = bytes(range(32))
    wroot = scheme.weights_root(geo, scheme.names_digest(names), roots).hex()
    (p / "weights" / "tree.json").write_text(json.dumps({"names": names, "tensor_roots": [r.hex() for r in roots], "root": wroot,
                                                         "chunk": CHUNK, "geo_digest": geo.hex(), "extra": extra}))
    (out / "binding_map_p1.json").write_text(json.dumps(bmap))
    sr, lk = RB.deferred_records(RB.path(out, 1))
    row = {"pair": 1, "arm": "instrumented", "commit": {"run_root": rc.run_root.hex(), "commit_pass": None, "sampled_replay": sr,
                                                        "boundary_linkage": lk, "weight_registration": {"root": wroot}}}
    reqs = [{"request_id": rid, "index": k, "prompt_token_ids": pr, "max_tokens": 3, "arrive_step": 0} for k, (rid, pr) in enumerate(prompts.items())]
    ctx = {"pair": 1, "run_root": rc.run_root.hex(), "requests": reqs, "sampling": {"max_tokens": 3}, "code": "test",
           "args": {"program_dir": "/served/build_request", "programs_root": None, "program_digest": inst["digest"],
                    "required_manifest": None, "label": "rope-d64-cpu"}}
    m = RB.seal(out, 1, ctx, out / "binding_map_p1.json")
    (out / "runs.jsonl").write_text(json.dumps(row) + "\n")
    (out / "verdict.json").write_text(json.dumps({"replay_deferred": {"bundles": [{"pair": 1, "manifest_sha256": m["sha256"]}]}}))
    pf = tmp_path / "partition.json"
    pf.write_text(json.dumps({"templates": [ROPE], "law": LAW}))
    return {"out": out, "row_dir": row_dir, "partition": pf, "world": world, "run_root": rc.run_root.hex(), "inst": inst}


def test_a_deferred_commits_serving_rows_are_the_verifiers_window_over_the_served_values(tmp_path, monkeypatch):
    c = _commit(tmp_path, monkeypatch)
    wdir = tmp_path / "window"
    win = DRW.from_bundle(RB.path(c["out"], 1), str(c["row_dir"]), str(c["partition"]), wdir)
    rec = json.loads((wdir / "registration.json").read_text())
    heads = HEADS
    assert (win["population"], rec["format"], rec["window"]["kind"], rec["window"]["run"]) == (heads, R.FORMAT, "served", c["run_root"])
    assert rec["window"]["committer"]["from"] == "replay bundle" and rec["window"]["row"] == "rope-d64-cpu"

    # the verifier's own copies: its population program, the run root of record, the public body it holds
    defn = bind(REGISTRY["RoPEHead_v1"], D=D)
    program, query = P.population_program(defn, heads), P.template_instance_query(defn)
    ports = {f"{ROPE}/{p}": {"leaves": heads, "domain": R.served_domain(P.program_digest(program), P.digest(P.partition(program, query)),
                                                                          f"{ROPE}/{p}", CS.ROW_SCHEMA, heads, c["run_root"])}
             for p in ("x", "cs", "out")}
    pub = (wdir / "m0-public.bin").read_bytes()
    v = R.check(rec, program, query, DR.parse_law(LAW), CS.SCHEME, ports, leaf_layer=CS.leaf_layer({ROPE: pub}))
    assert v.ok, v.detail

    # every hidden row is what was served: x the qkv row's head slice, cs the cache row at the token's position, out the rotated head
    rows = np.frombuffer((wdir / "m0-private.bin").read_bytes()[:heads * 3 * D * 2], dtype="<u2").reshape(heads, 3, D)
    world, inst = c["world"], c["inst"]
    idx = json.loads((wdir / "index.json").read_text())
    ents = idx["members"][0]["rows"]
    assert sum(e[1] - e[0] for e in ents) == heads and [e[0] for e in ents] == sorted(e[0] for e in ents)
    for lo, hi, rid, _step, layer, _op, _mem, _tok, i in ents:
        r = inst["rows"][i]
        (_k, a, xlo, xhi), = r["args"][0]
        (_k, _p, clo, chi), = r["args"][1]
        assert layer == 0 and r["spec"].startswith("RoPE_v1")
        assert np.array_equal(rows[lo:hi, 0], world.values[(rid, a)][xlo:xhi].reshape(hi - lo, D))
        assert np.array_equal(rows[lo:hi, 1], np.broadcast_to(world.weights[clo:chi], (hi - lo, D)))
        assert np.array_equal(rows[lo:hi, 2], world.values[(rid, i)].reshape(hi - lo, D))


def test_a_partition_or_binding_map_that_is_not_the_records_is_refused(tmp_path, monkeypatch):
    c = _commit(tmp_path, monkeypatch)
    pf = json.loads(c["partition"].read_text())
    c["partition"].write_text(json.dumps(dict(pf, partition_digest="00" * 64)))
    with pytest.raises(ValueError, match="not the verifier's"):
        DRW.from_bundle(RB.path(c["out"], 1), str(c["row_dir"]), str(c["partition"]), tmp_path / "w1")
    c["partition"].write_text(json.dumps(pf))
    bm = c["out"] / "binding_map_p1.json"
    bm.write_text(bm.read_text().replace('"rank": 0', '"rank": 1', 1))
    with pytest.raises(CPU.Refused, match="not the bundle's binding map"):
        DRW.from_bundle(RB.path(c["out"], 1), str(c["row_dir"]), str(c["partition"]), tmp_path / "w2")


def test_the_command_writes_the_window(tmp_path, monkeypatch):
    c = _commit(tmp_path, monkeypatch)
    o = cli.parse("serving-rows", [str(RB.path(c["out"], 1)), "--row-dir", str(c["row_dir"]), "--partition", str(c["partition"]),
                                   "--out", str(tmp_path / "w"), "--workers", "1"])
    assert DRW.main(o) == 0
    win = json.loads((tmp_path / "w" / "window.json").read_text())
    assert win["registration"] == R.FORMAT and win["run"] == c["run_root"]
    assert json.loads((tmp_path / "w" / "partition.json").read_text())["partition_digest"] == win["partition_digest"]
