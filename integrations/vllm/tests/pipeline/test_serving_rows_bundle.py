"""The serving rows of a deferred Commit (`pipeline.deferred_rows`, ``verity-vllm serving-rows``), end to end on CPU: a Program
with RoPE_v1 rows (D=64) executed honestly, its members committed by the production CPU committer and sealed into a replay bundle with
its registered weights, then the verifier's partition (every RoPE head) read back through the bundle opened against the Commit's record.
The registration passes `registration.check` against the verifier's own copies (its population program, the run root of record, the
public bodies), every hidden row is the served value, and a partition digest, a template or a binding map that is not the record's is
refused."""
from __future__ import annotations

import json

import numpy as np
import pytest

from verity.ml.circuits.defs import REGISTRY, bind
from verity.core.protocols.sampled_proofs.one_stage import draw as DR, partition as P, registration as R
from verity_vllm.commit import serving_rows as CS
from verity_vllm.pipeline import cli
from verity_vllm.pipeline import cpu_replay as CPU
from verity_vllm.pipeline import replay_bundle as RB
from verity_vllm.pipeline import deferred_rows as DRW
from verity_vllm.program.registry import b1  # noqa: F401  (registers RoPEHead_v1)

from tests.pipeline import served_rope as SR
from tests.pipeline.served_rope import D, HEADS, LAW, ROPE


def _commit(tmp_path) -> dict:
    """The deferred Commit's output dir and row dir, and the world that served it."""
    pytest.importorskip("torch")
    return SR.commit(tmp_path)


def test_a_deferred_commits_serving_rows_are_the_verifiers_window_over_the_served_values(tmp_path):
    c = _commit(tmp_path)
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
    # the same copies from the record alone, as the audit's verifier derives them (no committed value read)
    mine = DRW.window_of_record(RB.path(c["out"], 1), str(c["row_dir"]), str(c["partition"]))
    assert (mine.partition_digest, mine.run, CS.expected_ports(mine)) == (win["partition_digest"], c["run_root"], ports)

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


def test_a_partition_or_binding_map_that_is_not_the_records_is_refused(tmp_path):
    c = _commit(tmp_path)
    pf = json.loads(c["partition"].read_text())
    c["partition"].write_text(json.dumps(dict(pf, partition_digest="00" * 64)))
    with pytest.raises(ValueError, match="not the verifier's"):
        DRW.from_bundle(RB.path(c["out"], 1), str(c["row_dir"]), str(c["partition"]), tmp_path / "w1")
    # a template is matched by its descriptor id, so a RoPE head of another D is no member of these rows
    c["partition"].write_text(json.dumps(dict(pf, templates=["RoPEHead_v1{D=128}"])))
    with pytest.raises(LookupError, match=r"no rows of RoPEHead_v1\{D=128\}"):
        DRW.from_bundle(RB.path(c["out"], 1), str(c["row_dir"]), str(c["partition"]), tmp_path / "w1")
    c["partition"].write_text(json.dumps(pf))
    bm = c["out"] / "binding_map_p1.json"
    bm.write_text(bm.read_text().replace('"rank": 0', '"rank": 1', 1))
    with pytest.raises(CPU.Refused, match="not the bundle's binding map"):
        DRW.from_bundle(RB.path(c["out"], 1), str(c["row_dir"]), str(c["partition"]), tmp_path / "w2")


def test_the_command_writes_the_window(tmp_path):
    c = _commit(tmp_path)
    o = cli.parse("serving-rows", [str(RB.path(c["out"], 1)), "--row-dir", str(c["row_dir"]), "--partition", str(c["partition"]),
                                   "--out", str(tmp_path / "w"), "--workers", "1"])
    assert DRW.main(o) == 0
    win = json.loads((tmp_path / "w" / "window.json").read_text())
    assert win["registration"] == R.FORMAT and win["run"] == c["run_root"]
    assert json.loads((tmp_path / "w" / "partition.json").read_text())["partition_digest"] == win["partition_digest"]
