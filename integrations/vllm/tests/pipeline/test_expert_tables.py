"""`expert_tables`: a TP rank's registration of the per-expert cut's tables, on a synthetic block-FP8 MoE checkpoint (E = 8 experts,
two MoE layers, TP2).  Each registered entry is the one a window's committer commits for the same served weights
(`expert_rows.commit_tables` under `expert_rows.committer`'s key, program and nonces); a table served with another byte has another
root; the two ranks are distinct values; a registration's file is the record a window's committer reads
(`records.expert_tables`, `expert_rows.of_request`); and a key not minted for these nonces refuses (`key_of`)."""
from __future__ import annotations

import gzip
import hashlib
import json

import numpy as np
import pytest

from verity_vllm.check import weights_of_record as WR
from verity_vllm.config import UsageError
from verity_vllm.pipeline import expert_rows as ER
from verity_vllm.pipeline import expert_tables as ET
from verity_vllm.records import expert_tables as XT
from tests.check.test_weights_of_record import write_safetensors

G, I, H, E, WORLD = 128, 256, 256, 8, 2
LAYERS = (1, 2)
BLOCKS = {"w13": 1, "w2": 32}
REPO, REV = "test/fp8-moe-tables", "abc"
KEY = bytes(range(32))
WRAP = "model."


def _ty(shape, bits):
    t = f"Value<{bits}>"
    for n in reversed(shape):
        t = f"Array<{n},{t}>"
    return t


def _fields() -> list[tuple[str, list[int], int]]:
    """The rank Program's weight fields, named under the export wrapper's prefix as a Build names them."""
    I_r = I // WORLD
    out = [(WRAP + "model.embed_tokens.weight", [64, H], 16)]
    for L in LAYERS:
        c = f"{WRAP}model.layers.{L}.mlp.experts"
        out += [(f"{c}.w13_weight", [E, 2 * I_r, H], 8), (f"{c}.w13_weight_scale_inv", [E, 2 * I_r // G, H // G], 32),
                (f"{c}.w2_weight", [E, H, I_r], 8), (f"{c}.w2_weight_scale_inv", [E, H // G, I_r // G], 32)]
    return out


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    """Per-expert e4m3 weights (random bytes) and f32 block scales in the HF layout, with a rank Program dir a rank."""
    rng = np.random.default_rng(11)
    t: dict[str, tuple[str, np.ndarray]] = {}
    for L in LAYERS:
        for e in range(E):
            for part, (n, k) in (("gate_proj", (I, H)), ("up_proj", (I, H)), ("down_proj", (H, I))):
                name = f"model.layers.{L}.mlp.experts.{e}.{part}"
                t[name + ".weight"] = ("F8_E4M3", rng.integers(0, 256, size=(n, k), dtype=np.uint8))
                t[name + ".weight_scale_inv"] = ("F32", rng.random((n // G, k // G), dtype=np.float32))
    root = tmp_path_factory.mktemp("expert_tables")
    d = root / "snap"
    d.mkdir()
    write_safetensors(str(d / "model.safetensors"), t)
    cfg = {"hidden_size": H, "moe_intermediate_size": I, "num_experts": E, "tie_word_embeddings": False,
           "quantization_config": {"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic", "weight_block_size": [G, G]}}
    (d / "config.json").write_text(json.dumps(cfg))
    man = root / "checkpoints.json"
    man.write_text(json.dumps({"checkpoints": [{"repo": REPO, "revision": REV, "local_path": str(d),
                                                "files": {"model.safetensors": {"sha256": WR.sha256_file(str(d / "model.safetensors"))}}}]}))
    struct_ = "{" + ", ".join(f"{n}: {_ty(s, b)}" for n, s, b in _fields()) + "}"
    progs = []
    for rank in range(WORLD):
        p = root / f"rank{rank}"
        p.mkdir()
        with gzip.open(p / "instances.json.gz", "wt") as f:
            json.dump({"schema": "test", "digest": "e" * 128, "params": [["weights", struct_]], "rows": []}, f)
        (p / "result.json").write_text(json.dumps({"program_digest": "e" * 128, "tp": {"world": WORLD, "rank": rank}}))
        progs.append(str(p))
    return {"dir": str(d), "man": str(man), "progs": progs}


@pytest.fixture(scope="module")
def regs(ckpt):
    return [ET.register(ckpt["progs"][r], ckpt["dir"], KEY, BLOCKS, checkpoints_manifest=ckpt["man"]) for r in range(WORLD)]


def _served(ckpt, rank: int) -> dict[str, np.ndarray]:
    """The rank's expert weights as served: the checkpoint composed for the rank, by field."""
    ck = WR.CheckpointIndex(ckpt["dir"], checkpoints_manifest=ckpt["man"])
    return {f: WR.compose(f.removeprefix(WRAP), s, b, ck, world=WORLD, rank=rank)[0] for f, s, b in _fields() if ".experts." in f}


class _Plan:
    """The parts of `expert_rows.Plan` `commit_tables` reads, for one GEMM row a projection of each layer: its weight template
    (``("w", n)``) and its scale template (``("ws", G)``), each block of rows a unit, its table named as `Plan.tables` names it."""

    def __init__(self, blocks: dict[str, int], wrap: str):
        self.what, self.of, self.family, self.units, self._tables = {}, {}, {}, {}, {}
        I_r = I // WORLD
        for L in LAYERS:
            for proj, family, (N, K) in (("w13", "MoeExpertFp8Gemm_v1", (2 * I_r, H)), ("w2", "MoeExpertFp8GemmW_v1", (H, I_r))):
                c = f"{wrap}model.layers.{L}.mlp.experts.{proj}"
                w, ws = ER.ARGS[family][2], ER.ARGS[family][3]
                row = ("rid", None, {"i": 0, "args": {w: f"{c}_weight", ws: f"{c}_weight_scale_inv"}},
                       {"statics": {"E": E, "N": N, "K": K, "G": G}})
                n = blocks[proj]
                for what, label, nb, words in (("w", f"{c}_weight", N // n, n * K // 2), ("ws", f"{c}_weight_scale_inv", -(-N // G), 2 * K // G)):
                    t = f"{c}:{what}"
                    self.what[t], self.of[t], self.family[t] = (what, n if what == "w" else G), [row], family
                    self.units[t] = [(0, b) for b in range(nb)]
                    self._tables[t] = [(b, 1, b, ER.table_param(label, words, b), words * E) for b in range(nb)]

    def tables(self, t):
        return self._tables[t]


def _commit(served: dict[str, np.ndarray], doc, key=KEY, wrap=WRAP):
    """`commit_tables` over ``served``, the plan's tables named under ``wrap`` and ``served`` keyed by the same names."""
    return ER.commit_tables(_Plan(BLOCKS, wrap), None, None, None, lambda _pi, _s, _w, _rid, _a, label: [served[label]],
                            ER.committer(XT.ExpertTables.of(doc, "registration"), key))


def test_registered_entries_are_the_committers(ckpt, regs):
    """Every table the windows read (weights in blocks of `BLOCKS` rows, scales a G-block a table) is registered, at the root the
    committer commits for the rank's served weights, and nothing else is."""
    I_r = I // WORLD
    per_layer = 2 * I_r // BLOCKS["w13"] + 2 * I_r // G + H // BLOCKS["w2"] + H // G
    for rank, doc in enumerate(regs):
        assert (doc["schema"], doc["repo"], doc["revision"], doc["world"], doc["rank"]) == (XT.SCHEMA, REPO, REV, WORLD, rank)
        assert doc["program"] == ET.program(REPO, REV, WORLD, rank, doc["prefix"])
        assert XT.ExpertTables.of(doc, "registration").dump() == doc
        assert len(doc["values"]) == per_layer * len(LAYERS)
        got = _commit(_served(ckpt, rank), doc)
        assert {p: r.entry() for p, r in got.items()} == doc["values"]


def test_a_table_served_otherwise_has_another_root(ckpt, regs):
    """One byte of one expert's row changed changes exactly that row's table root."""
    served = _served(ckpt, 1)
    f = WRAP + "model.layers.2.mlp.experts.w13_weight"
    served[f] = served[f].copy()
    served[f][5, 7, 3] ^= 1
    got = {p: r.entry()["root"] for p, r in _commit(served, regs[1]).items()}
    changed = sorted(p for p, root in got.items() if root != regs[1]["values"][p]["root"])
    assert changed == [ER.table_param(f, BLOCKS["w13"] * H // 2, 7 // BLOCKS["w13"])]


def test_ranks_and_keys_are_distinct_values(ckpt, regs):
    """The two ranks' tables are registered under distinct programs at disjoint nonces, and another key draws other salts."""
    assert regs[0]["program"] != regs[1]["program"]
    n0, n1 = (set(d["nonces"].values()) for d in regs)
    assert len(n0) == len(regs[0]["values"]) and not n0 & n1 and min(n0 | n1) >= ET.TABLES_NONCE
    other = bytes(31) + b"\x01"
    got = _commit(_served(ckpt, 0), regs[0], key=other)
    assert all(r.entry()["root"] != regs[0]["values"][p]["root"] for p, r in got.items())


def test_stacks_and_refusals(ckpt, regs):
    """``stacks`` registers those stacks' tables exactly as the whole registration does, nonces included (a table's nonce is its
    own), so one key serves both; a block that does not divide N, a projection or stack no expert weight has, a short key and a
    registration whose nonces are not its tables' refuse."""
    stack = f"{WRAP}model.layers.2.mlp.experts"
    one = ET.register(ckpt["progs"][0], ckpt["dir"], KEY, BLOCKS, stacks=[stack], checkpoints_manifest=ckpt["man"])
    mine = {p for p in regs[0]["values"] if p.startswith(f"weights/{stack}.")}
    assert mine and one["values"] == {p: regs[0]["values"][p] for p in mine} and one["nonces"] == {p: regs[0]["nonces"][p] for p in mine}
    with pytest.raises(ValueError, match="power of two"):
        ET.register(ckpt["progs"][0], ckpt["dir"], KEY, {"w13": 3, "w2": 32}, checkpoints_manifest=ckpt["man"])
    with pytest.raises(ValueError, match="no expert stack has that projection"):
        ET.register(ckpt["progs"][0], ckpt["dir"], KEY, {**BLOCKS, "w4": 1}, checkpoints_manifest=ckpt["man"])
    with pytest.raises(ValueError, match="no expert stack"):
        ET.register(ckpt["progs"][0], ckpt["dir"], KEY, BLOCKS, stacks=[f"{WRAP}model.layers.7.mlp.experts"], checkpoints_manifest=ckpt["man"])
    with pytest.raises(ValueError, match="32-byte key"):
        ER.committer(XT.ExpertTables.of(regs[0], "registration"), KEY[:16])
    with pytest.raises(ValueError, match="one salt nonce for each table"):
        XT.ExpertTables.of({**regs[0], "nonces": dict(list(regs[0]["nonces"].items())[1:])}, "registration")


def test_a_shard_the_manifest_does_not_pin_refuses(ckpt, tmp_path):
    man = json.loads(open(ckpt["man"]).read())
    man["checkpoints"][0]["files"]["model.safetensors"]["sha256"] = "00" * 32
    (tmp_path / "checkpoints.json").write_text(json.dumps(man))
    with pytest.raises(ValueError, match="shards not pinned"):
        ET.register(ckpt["progs"][0], ckpt["dir"], KEY, BLOCKS, checkpoints_manifest=str(tmp_path / "checkpoints.json"))


def test_without_a_build_the_served_parameters_name_the_tables_and_pin_the_bytes(ckpt, tmp_path):
    """From a capture's ``params.json`` (vLLM's names, shapes, dtypes and SHA-512s) in place of a rank Program, the tables are
    registered under the served names at the root the committer commits for them, and a weight whose served SHA-512 is not the
    checkpoint's composition for the rank refuses."""
    served = {f.removeprefix(WRAP): a for f, a in _served(ckpt, 1).items()}
    dtype = {1: "torch.float8_e4m3fn", 4: "torch.float32"}
    params = {n: {"shape": list(a.shape), "dtype": dtype[a.dtype.itemsize], "sha512": hashlib.sha512(a.tobytes()).hexdigest()}
              for n, a in served.items()} | {"model.norm.weight": {"shape": [H], "dtype": "torch.bfloat16", "sha512": "00" * 64}}
    (tmp_path / "params.json").write_text(json.dumps(params))
    doc = ET.register(None, ckpt["dir"], KEY, BLOCKS, served=str(tmp_path / "params.json"), world=WORLD, rank=1,
                      checkpoints_manifest=ckpt["man"])
    assert (doc["prefix"], doc["rank"]) == ("", 1) and all(p.startswith("weights/model.layers.") for p in doc["values"])
    assert {p: r.entry() for p, r in _commit(served, doc, wrap="").items()} == doc["values"]
    n = "model.layers.2.mlp.experts.w2_weight_scale_inv"
    (tmp_path / "params.json").write_text(json.dumps(params | {n: {**params[n], "sha512": "ab" * 64}}))
    with pytest.raises(ValueError, match="not the served tensor"):
        ET.register(None, ckpt["dir"], KEY, BLOCKS, served=str(tmp_path / "params.json"), world=WORLD, rank=1,
                    checkpoints_manifest=ckpt["man"])
    with pytest.raises(ValueError, match="TP world and rank"):
        ET.register(None, ckpt["dir"], KEY, BLOCKS, served=str(tmp_path / "params.json"), checkpoints_manifest=ckpt["man"])


def test_a_key_not_minted_for_these_nonces_refuses_with_its_fix(tmp_path):
    """A key without its sidecar naming this nonce scheme (32 bytes written by hand, as the tables committed at the older
    layer-and-kind nonces were) refuses, saying to run with --new-key; so does one beside another key's sidecar.  A minted key
    reads back, both files mode 0600, and is minted once."""
    old = tmp_path / "old.key"
    old.write_bytes(KEY)
    with pytest.raises(UsageError, match="run with --new-key"):
        ET.key_of(str(old))
    new = tmp_path / "new.key"
    key = ET.mint_key(str(new))
    side = tmp_path / "new.key.nonces"
    assert ET.key_of(str(new)) == key and len(key) == 32 and json.loads(side.read_text())["nonces"] == ET.NONCES
    assert {p.stat().st_mode & 0o777 for p in (new, side)} == {0o600}
    with pytest.raises(UsageError, match="minted once"):
        ET.mint_key(str(new))
    (tmp_path / "old.key.nonces").write_bytes(side.read_bytes())
    with pytest.raises(UsageError, match="run with --new-key"):
        ET.key_of(str(old))


def test_the_command_registers_only_under_a_key_minted_for_its_nonces(ckpt, regs, tmp_path):
    """``tp-register-tables --key`` refuses a key not minted for its nonces before it reads the checkpoint, and writes nothing;
    ``--new-key`` mints one and registers under it, and the minted key then serves another run."""
    old, out = tmp_path / "old.key", tmp_path / "tables.json"
    old.write_bytes(KEY)

    def run(key: str, new_key: bool = False) -> int:
        return ET.main(ET.Options(program_dir=ckpt["progs"][0], checkpoint_dir=ckpt["dir"], key=key, block=["w13=1", "w2=32"],
                                  out=str(out), checkpoints_manifest=ckpt["man"], new_key=new_key))

    with pytest.raises(UsageError, match="run with --new-key"):
        run(str(old))
    assert not out.exists()
    assert run(str(tmp_path / "new.key"), new_key=True) == 0
    doc = json.loads(out.read_text())
    assert XT.ExpertTables.of(doc, "registration").program == regs[0]["program"] and set(doc["values"]) == set(regs[0]["values"])
    assert doc["values"] != regs[0]["values"]
    out.unlink()
    assert run(str(tmp_path / "new.key")) == 0 and json.loads(out.read_text()) == doc


def test_of_request_reads_the_ranks_file(tmp_path, regs):
    key = tmp_path / "key"
    key.write_bytes(KEY)
    for doc in regs:
        (tmp_path / f"tables{doc['rank']}.json").write_text(json.dumps(doc))
    spec = {"key": str(key), "registration": str(tmp_path / "tables{rank}.json")}
    assert ER.of_request(spec, rank=1) == {"key": KEY, "program": regs[1]["program"], "nonces": regs[1]["nonces"]}
    (tmp_path / "tables0.json").write_text(json.dumps(regs[1]))
    with pytest.raises(ValueError, match="of rank 1, not of rank 0"):
        ER.of_request(spec, rank=0)
