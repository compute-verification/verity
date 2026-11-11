"""The per-expert fused cut over a served block-FP8 MoE layer (`tests.pipeline.served_experts`), end to end on CPU: the row
driver's in-Commit hook commits the cut's window with the per-expert tables its request names, and the deferred Commit's
sealed bundle commits the same window afterwards (`deferred_rows.from_bundle`, ``verity-vllm serving-rows``) given the tables'
registration with its key and the checkpoint of record, which holds the expert weights the bundle doesn't carry.

* The layer's rows sit at the MoE block's planes, as the replay index reads a Build's.
* The hook's window is both projections' scale-row reads and fused blocks, ``served``, its tables' roots the registered ones.
* The bundle's window is the hook's (`same_commitment`), a stand-in linked to the record; the command writes it too.
* Without the tables a read or a fused block is refused, a registration without its key before the bundle is opened, and
  without the checkpoint the expert weights by name."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from verity.commitments.windows import rows as CS
from verity_vllm.check import weights_of_record as WR
from verity_vllm.check.replay.index import ProgramIndex
from verity_vllm.commit import step_rows
from verity_vllm.pipeline import cli
from verity_vllm.pipeline import deferred_rows as DRW
from verity_vllm.pipeline import replay_bundle as RB

from tests.pipeline import served_experts as SX

SAME = ("format", "program", "query", "law", "partition", "population", "scheme")


@pytest.fixture(scope="module")
def served(tmp_path_factory) -> dict:
    """The layer's deferred Commit with the row driver's window (``hook``) over its committed step."""
    pytest.importorskip("torch")
    root = tmp_path_factory.mktemp("served_experts")
    c = SX.commit(root)
    c["hook"] = SX.hook(c, root / "hook")
    return c


def _doc(d: Path, name: str) -> dict:
    return json.loads((Path(d) / name).read_text())


def same_commitment(a: Path, b: Path) -> None:
    """Windows ``a`` and ``b`` commit one thing: one population program, partition, law and port domains (which bind the run
    root), one circuit record and index, the same registered roots, and in every member the same committed rows on the ports
    its own file holds, among them each unit's opening of its table (the table's row, its salt and its path, the committer's
    under its key).  Each window salts its own rows, so the roots of its members' trees are its own."""
    ra, rb = _doc(a, "registration.json"), _doc(b, "registration.json")
    assert {k: ra[k] for k in SAME} == {k: rb[k] for k in SAME}
    assert [(x["port"], x["domain"]) for x in ra["roots"]] == [(x["port"], x["domain"]) for x in rb["roots"]]
    assert ra["window"]["registered_roots"] == rb["window"]["registered_roots"] and ra["window"]["circuit"] == rb["window"]["circuit"]
    for name in ("partition.json", "circuit.json", "index.json"):
        assert (Path(a) / name).read_bytes() == (Path(b) / name).read_bytes(), name
    wa, wb = _doc(a, "window.json"), _doc(b, "window.json")

    def shape(w: dict) -> list:
        return [(m["name"], m["template"], m["n"], m["ports"], m.get("value_rows"), m.get("registered")) for m in w["members"]]

    assert shape(wa) == shape(wb)
    ba, bb = CS.Bodies(a, wa), CS.Bodies(b, wb)
    for k in range(len(wa["members"])):
        oa, ob = ba.own(k), bb.own(k)
        assert oa.keys() == ob.keys() and oa
        for p in oa:
            assert np.array_equal(oa[p][0], ob[p][0]), (k, p)


def test_the_layers_rows_sit_at_the_moe_blocks_planes():
    inst = SX.program()
    pi = ProgramIndex(inst)
    kinds = []
    for r in inst["rows"]:
        if not r["name"].startswith(f"{SX.X}/") or "/moe_sum[" in r["name"]:
            continue
        d = pi.describe(r)
        assert d is not None and d["op_path"] == SX.OP and d["members"] and "plane_error" not in d, (r["name"], d)
        kinds.append(d["plane_kind"])
    M, NV = SX.M, SX.NV
    assert kinds == ["in_quant"] * M + ["router"] * M + ["up"] * NV + ["act"] * NV + ["act_quant"] * NV + ["down"] * NV
    assert pi.prefix == SX.WRAP and pi.n_steps == 1


def test_the_hooks_window_is_the_fused_cut_served_under_the_registered_tables(served):
    root = served["partition"].parent
    reg, win = _doc(served["hook"], "registration.json"), _doc(served["hook"], "window.json")
    w = reg["window"]
    assert (w["kind"], w["run"], w["research_run"]) == ("served", served["run_root"], "served-experts")
    assert "of" not in w and "link" not in w
    tmpls = _doc(root, "partition.json")["templates"]
    n_of = [sum(m["n"] for m in win["members"] if m["template"] == t) for t in tmpls]
    assert n_of == [*SX.UNITS["w13"], *SX.UNITS["w2"]] and win["population"] == sum(n_of)
    tables = _doc(root, "tables-rank0.json")["values"]
    assert tables and {p: w["registered_roots"][p] for p in tables} == {p: v["root"] for p, v in tables.items()}


def test_the_deferred_commits_bundle_commits_the_hooks_window_given_the_tables_and_the_checkpoint(served, tmp_path):
    bundle = RB.path(served["out"], 1)
    ck = WR.CheckpointIndex(str(served["checkpoint"]), checkpoints_manifest=str(served["checkpoints_manifest"]))
    DRW.from_bundle(bundle, str(served["row_dir"]), str(served["partition"]), tmp_path / "w", checkpoint=ck, workers=1,
                    tables=served["tables"])
    same_commitment(served["hook"], tmp_path / "w")
    man, root = RB.open_bundle(bundle)["manifest_sha256"], served["run_root"]
    deferred, hook = (_doc(d, "registration.json")["window"] for d in (tmp_path / "w", served["hook"]))
    assert DRW.source(deferred, root, man, step_rows.NAME) == {"kind": True, "run": True, "commitment": True, "of": True, "link": True}
    assert DRW.source(hook, root, man, step_rows.NAME, served["weights_root"]) == {
        "kind": True, "run": True, "commitment": True, "sha512": True, "weights_sha512": True, "link": True}


def test_the_command_takes_the_tables_and_the_checkpoint(served, tmp_path):
    o = cli.parse("serving-rows", [str(RB.path(served["out"], 1)), "--row-dir", str(served["row_dir"]),
                                   "--partition", str(served["partition"]), "--out", str(tmp_path / "w"), "--workers", "1",
                                   "--checkpoint-dir", str(served["checkpoint"]),
                                   "--checkpoints-manifest", str(served["checkpoints_manifest"]),
                                   "--tables", served["tables"]["registration"], "--tables-key", served["tables"]["key"]])
    assert DRW.main(o) == 0
    same_commitment(served["hook"], tmp_path / "w")


def test_the_cut_is_refused_without_its_tables_their_key_or_the_checkpoint(served, tmp_path):
    bundle, at = RB.path(served["out"], 1), (str(served["row_dir"]), str(served["partition"]))
    ck = WR.CheckpointIndex(str(served["checkpoint"]), checkpoints_manifest=str(served["checkpoints_manifest"]))
    with pytest.raises(ValueError, match=r"is not among the committer's tables \(commit_tables\)"):
        DRW.from_bundle(bundle, *at, tmp_path / "w1", checkpoint=ck, workers=1)
    with pytest.raises(ValueError, match=r"read under its committer's key file \(--tables-key\)"):
        DRW.from_bundle(tmp_path / "no-bundle", *at, tmp_path / "w2", tables={"registration": served["tables"]["registration"]})
    with pytest.raises(LookupError, match=r"registered weight field \S+\.experts\.w\S+ is not available"):
        DRW.from_bundle(bundle, *at, tmp_path / "w3", workers=1, tables=served["tables"])
    assert not any((tmp_path / d / "registration.json").exists() for d in ("w1", "w2", "w3"))
