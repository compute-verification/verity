"""The per-expert fused cut over a served block-FP8 MoE layer (`tests.pipeline.served_experts`), end to end on CPU: the row
driver's in-Commit hook commits the cut's window with the per-expert tables its request names, and the deferred Commit's
sealed bundle commits the same window afterwards (``verity-vllm serving-rows``, `deferred_rows.from_bundle`) given the tables'
registration with its key and the checkpoint of record, which holds the expert weights the bundle doesn't carry.

* The layer's rows sit at the MoE block's planes, as the replay index reads a Build's.
* The hook's window is both projections' scale-row reads and fused blocks, ``served``, its tables' roots the registered ones.
* The command's window over the bundle is the hook's (`served_experts.same_commitment`), a stand-in linked to the record.
* Without the tables a read or a fused block is refused, a registration without its key before the bundle is opened, and
  without the checkpoint the expert weights by name.

Every test but the first reads a window `serving_rows.after_commit` wrote, and a window of this cut costs minutes however
small its layer: its templates' gates are built once a process and its registration walks them at every write. So the file
writes two windows, the hook's and the command's, and those tests are ``slow``."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from verity_vllm.check import weights_of_record as WR
from verity_vllm.check.replay.index import ProgramIndex
from verity_vllm.commit import step_rows
from verity_vllm.pipeline import cli
from verity_vllm.pipeline import deferred_rows as DRW
from verity_vllm.pipeline import replay_bundle as RB

from tests.pipeline import served_experts as SX


@pytest.fixture(scope="module")
def served(tmp_path_factory) -> dict:
    """The layer's deferred Commit with the row driver's window (``hook``) over its committed step."""
    pytest.importorskip("torch")
    root = tmp_path_factory.mktemp("served_experts")
    c = SX.commit(root)
    c["hook"] = SX.hook(c, root / "hook")
    return c


@pytest.fixture(scope="module")
def deferred(served, tmp_path_factory) -> Path:
    """The window ``verity-vllm serving-rows`` writes over the deferred Commit's bundle, given the tables and the checkpoint."""
    out = tmp_path_factory.mktemp("deferred") / "w"
    o = cli.parse("serving-rows", [str(RB.path(served["out"], 1)), "--row-dir", str(served["row_dir"]),
                                   "--partition", str(served["partition"]), "--out", str(out), "--workers", "1",
                                   "--checkpoint-dir", str(served["checkpoint"]),
                                   "--checkpoints-manifest", str(served["checkpoints_manifest"]),
                                   "--tables", served["tables"]["registration"], "--tables-key", served["tables"]["key"]])
    assert DRW.main(o) == 0
    return out


def _doc(d: Path, name: str) -> dict:
    return json.loads((Path(d) / name).read_text())


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


@pytest.mark.slow
def test_the_hooks_window_is_the_fused_cut_served_under_the_registered_tables(served):
    """The row driver's window as its Commit writes it (minutes, the module's note). It catches what the cut's tests on its
    readers can't: a window whose registration is not ``served``, counts other units than the layer's rows give, or binds
    other table roots than the committer registered."""
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


@pytest.mark.slow
def test_the_deferred_commits_bundle_commits_the_hooks_window_given_the_tables_and_the_checkpoint(served, deferred, tmp_path):
    """A second window, the command's over the sealed bundle (about a minute more). It catches what no test of one window
    can: a bundle's window that differs from the one the Commit itself made of the same step, in its program, its partition,
    its registered roots or any unit's committed rows, its table's opening among them. One flipped bit of one row is refused."""
    SX.same_commitment(served["hook"], deferred)
    with pytest.raises(AssertionError, match="the rows of port"):
        SX.same_commitment(served["hook"], SX.flipped(deferred, tmp_path / "flipped"))
    man, root = RB.open_bundle(RB.path(served["out"], 1))["manifest_sha256"], served["run_root"]
    mine, hook = (_doc(d, "registration.json")["window"] for d in (deferred, served["hook"]))
    assert DRW.source(mine, root, man, step_rows.NAME) == {"kind": True, "run": True, "commitment": True, "of": True, "link": True}
    assert DRW.source(hook, root, man, step_rows.NAME, served["weights_root"]) == {
        "kind": True, "run": True, "commitment": True, "sha512": True, "weights_sha512": True, "link": True}


@pytest.mark.slow
def test_the_cut_is_refused_without_its_tables_their_key_or_the_checkpoint(served, tmp_path):
    """Each refusal on the layer's own bundle (seconds once the module's templates are built, over a minute alone). The
    cut's readers refuse a template or a weight they are not given; only a Commit's bundle shows that it brings neither, so
    that the window is refused unless the tables and the checkpoint are handed to it."""
    bundle, at = RB.path(served["out"], 1), (str(served["row_dir"]), str(served["partition"]))
    ck = WR.CheckpointIndex(str(served["checkpoint"]), checkpoints_manifest=str(served["checkpoints_manifest"]))
    with pytest.raises(ValueError, match=r"is not among the committer's tables \(commit_tables\)"):
        DRW.from_bundle(bundle, *at, tmp_path / "w1", checkpoint=ck, workers=1)
    with pytest.raises(ValueError, match=r"read under its committer's key file \(--tables-key\)"):
        DRW.from_bundle(tmp_path / "no-bundle", *at, tmp_path / "w2", tables={"registration": served["tables"]["registration"]})
    with pytest.raises(LookupError, match=r"registered weight field \S+\.experts\.w\S+ is not available"):
        DRW.from_bundle(bundle, *at, tmp_path / "w3", workers=1, tables=served["tables"])
    assert not any((tmp_path / d / "registration.json").exists() for d in ("w1", "w2", "w3"))
