"""`predict node-registration`: a TP2 node Program's θ registered from a toy TP Commit of record is a log nci-check's full check accepts
(`nci_reference.check_baseline`), each rank's θ parameter the entry `registered_rows.commit` makes of that rank's committed tensor,
whether the checkpoint holds it (a shard composed at the rank) or the bundle does; and each way the Commit is not the node's is
refused by name before a row is committed."""
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from verity.ml.circuits.codec import canonical_json
from verity.ml.commitments import hm96
from verity_vllm.commit import registered_rows as RR
from verity_vllm.commit import scheme
from verity_vllm.config import UsageError
from verity_vllm.pipeline import cpu_replay as CPU
from verity_vllm.pipeline import registration_log as RL
from verity_vllm.pipeline import replay_bundle as RB
from verity_vllm.program.predict import aot as A
from verity_vllm.program.predict.common import _arr_shape
from verity_vllm.program.predict import node_registration as NR
from verity_vllm.program.predict.node import build_node_program, node_parameters
from tests.pipeline.test_cpu_replay import _shard
from tests.program.predict.test_predict import TINY, _phi

_spec = importlib.util.spec_from_file_location(
    "nci_reference", Path(__file__).resolve().parents[5] / "verity/core/protocols/network/nci/lean/scripts/nci_reference.py")
REF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(REF)

WORLD, CHUNK, PREFIX = 2, 16, "inner.model."
SHAPE = {"LP": 4, "T": 2}
PHI = _phi(TINY, tp={"world": WORLD, "rank": 0})
NODE = build_node_program(PHI, "request", **SHAPE)
PARAMS = node_parameters(PHI, "request", **SHAPE)
RNG = np.random.default_rng(5)
DOWN = "model.layers.0.mlp.down_proj.weight"
#: what the checkpoint holds: the replicated final norm, and layer 0's down projection, which each rank holds a column half of
CKPT = {"model.norm.weight": RNG.integers(0, 1 << 16, (64,), dtype=np.uint16), DOWN: RNG.integers(0, 1 << 16, (64, 128), dtype=np.uint16)}
DTYPE = {16: ("bfloat16", np.uint16), 32: ("float32", np.uint32)}


def _served(rank: int) -> dict[str, np.ndarray]:
    """Rank `rank`'s registered tensors, by name: the checkpoint's two (the down projection's columns sliced here by hand), and
    every other field the bundle carries, drawn per rank."""
    out = {}
    for name, t in PARAMS.leaves[rank]:
        w = next(iter(t.width_set()))
        out[name] = (CKPT[name] if name == "model.norm.weight" else CKPT[DOWN][:, 64 * rank:64 * (rank + 1)] if name == DOWN else
                     RNG.integers(0, 1 << w, _arr_shape(t), dtype=DTYPE[w][1]))
    return out


def _seal(bdir: Path) -> str:
    files = {p.relative_to(bdir).as_posix(): {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
             for p in sorted(bdir.rglob("*")) if p.is_file() and p.name != RB.MANIFEST}
    (bdir / RB.MANIFEST).write_text(json.dumps({"schema": RB.SCHEMA, "files": files}))
    return hashlib.sha256((bdir / RB.MANIFEST).read_bytes()).hexdigest()


@pytest.fixture
def commit(tmp_path, monkeypatch):
    """A TP2 Commit of record of TINY's request: per rank its bundle (weights tree, the tensors no checkpoint holds, context), its
    Build of record (result.json, the trace's field names) and its committed weights root; the checkpoint; the node's descriptor."""
    monkeypatch.setattr(A, "phi_at", lambda row, rank, config: PHI)
    ckd, out = tmp_path / "ck", tmp_path / "commit"
    ckd.mkdir()
    (ckd / "config.json").write_text("{}")
    _shard(ckd / "model.safetensors", CKPT)
    served, roots, bundles, cfg = {}, {}, [], {"programs": {}, "of_record": {}, "requests": [{"request_id": "r0", "prompt_len": 4, "max_tokens": 3}]}
    for r in range(WORLD):
        served[r] = _served(r)
        build = tmp_path / "build" / f"rank{r}"
        build.mkdir(parents=True)
        (build / "instances.json.gz").write_bytes(b"")
        (build / "result.json").write_text(json.dumps({"program": {"digest": f"{r}" * 128}}))
        (build / "derive-report.json").write_text(json.dumps({"provenance": {"weights": [
            {"name": PREFIX + n, "field": i} for i, (n, _t) in reversed(list(enumerate(PARAMS.leaves[r])))]}}))
        cfg["programs"][str(r)], cfg["of_record"][str(r)] = str(build), [f"{r}" * 128]
    for r in range(WORLD):
        bdir = RB.path(out, 0, rank=r)
        (bdir / "weights" / "extra").mkdir(parents=True)
        names, troots, extra = [], [], {}
        for i, (name, a) in enumerate(served[r].items() if r == 0 else reversed(served[r].items())):
            raw = np.ascontiguousarray(a).reshape(-1).view(np.uint8)
            names.append([name, DTYPE[a.dtype.itemsize * 8][0], list(a.shape), int(raw.size)])
            troots.append(CPU.tensor_root(raw, CHUNK))
            if name not in CKPT:
                extra[name] = i
                raw.tofile(bdir / "weights" / "extra" / f"{i}.bin")
        geo = bytes(32)
        roots[r] = scheme.weights_root(geo, scheme.names_digest(names), troots).hex()
        (bdir / "weights" / "tree.json").write_text(json.dumps({"names": names, "tensor_roots": [x.hex() for x in troots], "root": roots[r],
                                                                "chunk": CHUNK, "geo_digest": geo.hex(), "extra": extra}))
        (bdir / "context.json").write_text(json.dumps({"pair": 0, "tp": {"world": WORLD, "rank": r}, "replay": cfg}))
        bundles.append({"pair": 0, "rank": r, "manifest_sha256": _seal(bdir)})
    (out / "replay_deferred.json").write_text(json.dumps({"world": WORLD, "bundles": bundles}))
    rows = [{"pair": 0, "arm": "control"},
            {"pair": 0, "arm": "instrumented", "commit": {"commit_pass": True, "per_rank": {
                str(r): {"weight_registration": {"root": roots[r]}} for r in range(WORLD)}}}]
    (out / "runs.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    prog = tmp_path / "node_descriptor.json.gz"
    prog.write_bytes(gzip.compress(canonical_json(NODE.desc)))
    o = NR.Options(row="tiny", config="-", wrapper="request", **SHAPE, program=str(prog), commit_out=str(out), checkpoint_dir=str(ckd),
                   window=json.dumps({"kind": "stand-in", "deployment": "test"}), out=str(tmp_path / "log"),
                   salts=str(tmp_path / "private" / "salts"), workers=1)
    return {"o": o, "served": served, "out": out, "ck": ckd, "tmp": tmp_path}


def _instance(provenance):
    text = canonical_json(NODE.desc).decode()
    inp, _reg, spans = REF.root_nodes(text)
    computed = sum(hi - lo for k, (lo, hi) in enumerate(spans) if not inp[k])
    return {"program": text, "theta": NODE.desc["annotations"]["theta"], "owner": ["input" if i else 0 for i in inp],
            "gamma": NODE.program.gates, "alpha": [4 * 32, computed], "provenance": provenance}, computed


def _manifest(d, sha=None):
    """A checkpoints.json pinning the checkpoint's one shard at `sha` (default: its own)."""
    sha = sha or hashlib.sha256((d["ck"] / "model.safetensors").read_bytes()).hexdigest()
    p = d["tmp"] / "checkpoints.json"
    p.write_text(json.dumps({"checkpoints": [{"repo": "test/tiny", "revision": "0" * 40, "local_path": str(d["ck"]),
                                              "safetensors_sha256": {"model.safetensors": sha}}]}))
    return str(p)


@pytest.mark.slow
def test_the_nodes_theta_registers_from_each_ranks_commit_and_passes_the_full_check(commit):
    """Two spawned rank processes, on the pinned checkpoint: the log passes `check_baseline`, and each rank's θ parameter i is
    `registered_rows.commit` of the tensor that rank registered under its field i's name, under the salts written to --salts and
    nowhere under --out."""
    o = NR.Options(**{**commit["o"].__dict__, "workers": 2, "checkpoints_manifest": _manifest(commit)})
    assert NR.main(o) == 0
    log, salts = Path(o.out), Path(o.salts)
    doc, rec, trusted = (json.loads((log / f"{n}.json").read_text()) for n in ("registration", "record", "trusted"))
    assert doc["checkpoint"]["pinned_shards"] == 1 and doc["checkpoint"]["entry"]["repo"] == "test/tiny"
    provenance = json.loads((log / "provenance.json").read_text())
    inst, computed = _instance(provenance)
    assert REF.check_baseline(inst, trusted) == {"gates": NODE.program.gates, "ius": [[computed, 4 * 32]], "complies": True, "failure": None}
    assert doc["program_digest"] == NODE.digest == rec["program"] and doc["record_digest"] == REF.record_digest(rec) == trusted["run"]
    assert [e["value"] for e in rec["registered"]] == list(NODE.theta) and len(provenance) == len(NODE.theta)
    assert [rk["opened_by_source"] for rk in doc["ranks"]] == [{"checkpoint": 2, "bundle": len(PARAMS.leaves[0]) - 2}] * WORLD
    for e in rec["registered"]:
        r, name = PARAMS.fields()[e["value"]]
        t = dict(PARAMS.params)[e["value"]]
        again = RR.commit(rec["program"], e["value"], commit["served"][r][name], RL.word_bits(t), words=e["words"],
                          salts=(salts / f"{e['value']}.salts").read_bytes())
        assert again.entry == e, (e["value"], name)
    assert not list(log.rglob("*.salts")) and len(list(salts.glob("*.salts"))) == len(NODE.theta)
    assert len((salts / f"{rec['registered'][0]['value']}.salts").read_bytes()) == rec["registered"][0]["leaves"] * hm96.SHA512.salt_bytes


def _runs(d, fn):
    p = d["out"] / "runs.jsonl"
    rows = [json.loads(x) for x in p.read_text().splitlines()]
    fn(rows[1]["commit"])
    p.write_text("".join(json.dumps(x) + "\n" for x in rows))


def _reseal(d, rank):
    """Rank `rank`'s bundle sealed again and named so by replay_deferred.json: a bundle of record that holds otherwise."""
    dd = json.loads((d["out"] / "replay_deferred.json").read_text())
    dd["bundles"][rank]["manifest_sha256"] = _seal(RB.path(d["out"], 0, rank=rank))
    (d["out"] / "replay_deferred.json").write_text(json.dumps(dd))


def _ctx(d, rank, fn, decided=True):
    """Rank `rank`'s context changed by `fn` and sealed again, and named so by replay_deferred.json unless not `decided`."""
    p = RB.path(d["out"], 0, rank=rank) / "context.json"
    ctx = json.loads(p.read_text())
    fn(ctx)
    p.write_text(json.dumps(ctx))
    if decided:
        _reseal(d, rank)
    else:
        _seal(RB.path(d["out"], 0, rank=rank))


def _retree(d, rank, fn):
    """Rank `rank`'s committed tree's names changed by `fn`, refolded, sealed and of record: a Commit that registered otherwise."""
    p = RB.path(d["out"], 0, rank=rank) / "weights" / "tree.json"
    tree = json.loads(p.read_text())
    fn(tree["names"])
    tree["root"] = scheme.weights_root(bytes(32), scheme.names_digest(tree["names"]), [bytes.fromhex(x) for x in tree["tensor_roots"]]).hex()
    p.write_text(json.dumps(tree))
    _reseal(d, rank)
    _runs(d, lambda c: c["per_rank"][str(rank)]["weight_registration"].update(root=tree["root"]))


def _trace(d, rank, fn):
    p = d["tmp"] / "build" / f"rank{rank}" / "derive-report.json"
    rep = json.loads(p.read_text())
    fn({w["field"]: w for w in rep["provenance"]["weights"]})
    p.write_text(json.dumps(rep))


def _swap(ws, a, b):
    ws[a]["name"], ws[b]["name"] = ws[b]["name"], ws[a]["name"]


def _flip_checkpoint(d):
    """The checkpoint's last byte flipped: the last column of the down projection, which rank 1's half holds."""
    p = d["ck"] / "model.safetensors"
    b = bytearray(p.read_bytes())
    b[-1] ^= 1
    p.write_bytes(bytes(b))


@pytest.mark.parametrize("tamper,named", [
    (lambda d: _runs(d, lambda c: c.update(commit_pass=False, commit_fail_reason="replay")), "did not pass"),
    (lambda d: _runs(d, lambda c: c["per_rank"]["1"]["weight_registration"].update(root=c["per_rank"]["0"]["weight_registration"]["root"])),
     "rank 1: the bundle's weights root"),
    (lambda d: (RB.path(d["out"], 0, rank=1) / "weights" / "extra" / "0.bin").write_bytes(b"\0" * 2 * 8192), "not the manifest's"),
    (lambda d: (d["out"] / "replay_deferred.json").write_text(json.dumps({"bundles": []})), "names no bundle of pair 0"),
    (lambda d: _ctx(d, 1, lambda c: c.update(resealed=True), decided=False), "names no bundle of pair 0 with manifest sha256"),
    (lambda d: _ctx(d, 1, lambda c: c["tp"].update(rank=0)), "not pair 0 rank 1 of world 2"),
    (lambda d: _ctx(d, 1, lambda c: c["replay"]["of_record"].update({"1": ["e" * 128]})), "holds no Program of record for the rank"),
    (lambda d: _trace(d, 1, lambda ws: _swap(ws, 1, 2)), "rank 1: its Build's trace names"),
    (lambda d: _trace(d, 0, lambda ws: ws[5].update(name="model.inner." + ws[5]["name"][len(PREFIX):])), "rank 0: its Build's trace"),
    (_flip_checkpoint, f"rank 1: {DOWN} does not open"),
    (lambda d: _retree(d, 1, lambda ns: next(n for n in ns if n[0] == "model.embed_tokens.weight").__setitem__(1, "float32")),
     "rank 1: rank1.weights[0] (model.embed_tokens.weight) is"),
    (lambda d: _retree(d, 0, lambda ns: next(n for n in ns if n[0] == "model.norm.weight").__setitem__(0, "model.final_norm.weight")),
     "rank 0: rank0.weights[13] is 'model.norm.weight', which the rank's committed weights tree does not register"),
])
def test_a_commit_that_is_not_the_nodes_is_refused_by_name(commit, tamper, named, capsys):
    tamper(commit)
    assert NR.main(commit["o"]) == 2
    said = capsys.readouterr().out
    assert "REFUSED" in said and named in said, said
    assert not Path(commit["o"].out, "record.json").exists()


def test_a_descriptor_that_is_not_the_nodes_is_refused(commit, capsys):
    step = build_node_program(PHI, "step", M=1, pos0=8)
    Path(commit["o"].program).write_bytes(gzip.compress(canonical_json(step.desc)))
    assert NR.main(NR.Options(**{**commit["o"].__dict__, "workers": 2})) == 2 and "these are not its parameters" in capsys.readouterr().out
    desc = copy.deepcopy(NODE.desc)
    desc["annotations"]["theta"] = desc["annotations"]["theta"][1:]
    Path(commit["o"].program).write_bytes(gzip.compress(canonical_json(desc)))
    assert NR.main(commit["o"]) == 2 and "the descriptor's θ is" in capsys.readouterr().out


def test_a_checkpoint_its_manifest_does_not_pin_is_refused(commit, capsys):
    o = NR.Options(**{**commit["o"].__dict__, "checkpoints_manifest": _manifest(commit, "0" * 64)})
    assert NR.main(o) == 2 and "shards ['model.safetensors'] (1 of 1) are not the ones" in capsys.readouterr().out


def test_salts_never_go_under_the_published_out(commit):
    with pytest.raises(UsageError, match="the salts are the prover's"):
        NR.register(NR.Options(**{**commit["o"].__dict__, "salts": str(Path(commit["o"].out) / "salts")}))
