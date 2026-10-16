"""Input sets (``bench.input_sets``), the template catalogue (``bench.templates``) and the IR generator (``bench.generate``).

Captured and synthetic sets are interchangeable: the eight captured #101 manifests (``data/captured-101/manifests``, PR #42's
export ``art:b5bb0ca9``) identify their subcircuits, whose relation, set name and ports are exactly the exporter's; slices of
seven of them (``data/captured-101/slices``, ``input_sets.subset``) re-evaluate bit-exactly through the IR evaluator; a synthetic
set of each template has the captured layout and provenance fields and verifies the same way."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from verity_numerical.bench import generate as G
from verity_numerical.bench import input_sets as IS
from verity_numerical.bench import templates as TM

DATA = Path(__file__).parent / "data" / "captured-101"
CAPTURED = sorted((DATA / "manifests").glob("*.json"))
SLICES = sorted(d for d in (DATA / "slices").iterdir() if d.is_dir())
CENSUS = Path(__file__).resolve().parents[3] / "census"

#: Small bindings of every template (fast under the gate reference too).
SMALL = {"gemm-coordinate": dict(K=64), "attention-head": dict(D=16, BN=16, T="1:40"), "rope-head": dict(D=8),
         "rmsnorm-fused-cuda": dict(N=64, EPS=1e-5), "rmsnorm-triton": dict(N=64, EPS=1e-6), "silu-mul": dict(I=32),
         "silu-mul-v2": dict(I=32), "gumbel-top-p-token-select": dict(V=16), "token-select-greedy": dict(V=64),
         "moe-router-topk": dict(E=16, TOPK=4, VPT=8), "moe-router-topk-norm": dict(E=16, TOPK=4, VPT=8), "moe-sum": dict(TOPK=4, H=16), "bias-add": dict(N=16),
         "bf16-mul-scalar": dict(N=16), "embedding-row": dict(V=50, H=8), "moe-expert-coordinate": dict(E=4, K=64),
         "moe-expert-coordinate-weighted": dict(E=4, K=64), "scaled-mm-fp8-block-coordinate": dict(K=256, G=128)}


def test_catalogue_covers_the_101_templates():
    assert set(TM.TEMPLATES) == set(SMALL)
    assert len(CAPTURED) == 8 and len(SLICES) == 7


@pytest.mark.parametrize("path", CAPTURED, ids=lambda p: p.stem)
def test_captured_manifest_identifies_its_subcircuit(path):
    man = json.loads(path.read_text())
    sub = TM.identify(man["relation"])
    assert sub.set_name == man["set"] == path.stem
    assert sub.relation == man["relation"]
    assert {p.name: (p.dtype, p.words) for p in sub.ports} == {k: (v["dtype"], v["words"]) for k, v in man["ports"].items()}
    body = {k: man[k] for k in ("schema", "set", "relation", "n", "ports")}
    body["files"] = {k: v for k, v in man["files"].items() if k != "index.json"}
    assert IS.content_digest(body) == man["content_digest"]
    assert man["content_digest_rule"] == IS.CONTENT_RULE
    assert set(IS.provenance(man)) == set(IS.PROVENANCE_FIELDS) and IS.provenance(man)["source"] == "captured"
    assert TM.parse_subcircuit_id(sub.id).set_name == sub.set_name


@pytest.mark.parametrize("path", SLICES, ids=lambda p: p.name)
def test_captured_slice_reevaluates_through_the_ir(path):
    s = IS.InputSet.open(path)
    assert s.source == "captured" and s.provenance["subset_of"]["content_digest"]
    r = IS.verify(path)
    assert r["ok"] and r["checked"] == s.n, r


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    out = tmp_path_factory.mktemp("synthetic")
    return {t: G.generate(TM.subcircuit(t, **p), 6, 11, out) for t, p in SMALL.items()}


@pytest.mark.parametrize("template", sorted(SMALL))
def test_synthetic_set_has_the_captured_layout(synthetic, template):
    s = IS.InputSet.open(synthetic[template]["path"])
    assert IS.verify(s.path)["ok"]
    assert s.source == "synthetic" and s.manifest["schema"] == IS.SCHEMA
    captured = json.loads(CAPTURED[0].read_text())
    assert set(s.manifest) == set(captured)
    assert set(s.manifest["provenance"]) == set(IS.PROVENANCE_FIELDS)
    idx = json.loads((s.path / "index.json").read_text())
    assert idx["columns"] == IS.INDEX_COLUMNS == json.loads((SLICES[0] / "index.json").read_text())["columns"]
    assert [r[8]["draw"] for r in idx["rows"]] == sorted(r[8]["draw"] for r in idx["rows"])
    sub = s.subcircuit
    assert sub.id == TM.subcircuit(template, **SMALL[template]).id and s.name == sub.set_name
    digests = s.provenance["program_digests"]
    assert digests and all(len(v) == 128 for v in digests.values())
    rec = s.provenance["recipe"]
    assert rec["seed"] == 11 and rec["subcircuit"] == sub.id and rec["n"] == s.n == 6


def test_synthetic_and_captured_names_coincide_for_the_101_bindings():
    for path in CAPTURED:
        man = json.loads(path.read_text())
        sub = TM.identify(man["relation"])
        bound = TM.subcircuit(sub.template.id, **sub.params)
        assert (bound.id, bound.set_name, bound.relation) == (sub.id, man["set"], man["relation"])


def test_generation_is_deterministic_and_prefix_stable(tmp_path):
    sub = TM.subcircuit("gemm-coordinate", K=64, tile=4)
    a = G.generate(sub, 8, 3, tmp_path / "a")
    b = G.generate(sub, 16, 3, tmp_path / "b")
    c = G.generate(sub, 8, 3, tmp_path / "c")
    d = G.generate(sub, 8, 4, tmp_path / "d")
    assert a["content_digest"] == c["content_digest"] != d["content_digest"]
    sa, sb = IS.InputSet.open(a["path"]), IS.InputSet.open(b["path"])
    for port in ("x", "w", "y"):
        assert np.array_equal(sa.port(port), sb.port(port, 0, 8))
    x = sb.port("x")
    assert all(np.array_equal(x[i], x[4 * (i // 4)]) for i in range(16))    # activation rows shared per tile of 4
    assert not np.array_equal(x[0], x[4])


def test_draw_format_is_pinned():
    """Instance words are a function of (key, draw) alone: the counter stream's bytes and the word rule, fixed here."""
    sub = TM.subcircuit("gemm-coordinate", K=64)
    k = G.key_for(sub, 1)
    assert k.stream(("x", 0), 8).hex() == "cbfb52e4ff385132"
    d = sub.draw(k, 0)
    assert d.inputs["x"][:8].tolist() == [48997, 48895, 48786, 48679, 49141, 49163, 16398, 48634]
    assert d.inputs["w"][:8].tolist() == [14769, 15487, 49039, 15187, 49013, 48260, 15627, 47538]
    assert TM.bf16_words(np.array([0, 1, 0xFFFFFFFF, 0x12345678, 0x01000000], dtype=np.uint32), 127, 4).tolist() == \
        [0, 0, 49407, 16572, 15744]


def test_parallel_generation_equals_serial(tmp_path):
    sub = TM.subcircuit("rope-head", D=8)
    assert G.generate(sub, 40, 5, tmp_path / "s", chunk=8)["content_digest"] == \
        G.generate(sub, 40, 5, tmp_path / "p", procs=2, chunk=8)["content_digest"]


def test_tampering_is_caught(tmp_path, synthetic):
    d = tmp_path / "set"
    shutil.copytree(synthetic["silu-mul"]["path"], d)
    s = IS.InputSet.open(d)
    f = d / s.manifest["ports"]["out.out"]["file"]
    b = bytearray(f.read_bytes())
    b[0] ^= 1
    f.write_bytes(bytes(b))
    assert IS.verify(d)["ok"] is False and "sha256" in IS.verify(d)["why"][0]
    ports = {k: list(v) for k, v in s.instances().items()}
    IS.write(tmp_path / "re", s.name, s.relation, ports, s.index(), s.manifest["provenance"])   # a consistent manifest, a wrong output
    r = IS.verify(tmp_path / "re" / s.name)
    assert r["ok"] is False and r["bad"] == [0]


def test_evaluation_targets_equal_the_definitions():
    """Heads evaluate through a batch Definition that has a kernel; each equals the head's gate reference, and the captured
    vLLM GEMM step (core's AmpereBF16TcDot16_v2 since the re-baseline re-key; the retired _v1 evaluated the same function) equals the
    kernel."""
    from verity.ml.circuits.evaluation import evaluate_batch
    from verity.ml.circuits.defs import REGISTRY, bind
    from verity.ml.circuits.ops import gemm

    key = G.key_for(TM.subcircuit("attention-head", D=16, BN=16), 0)
    for sub, n in ((TM.subcircuit("attention-head", D=16, BN=16, T="17:40"), 4), (TM.subcircuit("rope-head", D=8), 16)):
        draws = [sub.draw(key, j) for j in range(n)]
        for d in draws:
            st = d.instance_statics
            args = [d.inputs[p.name][None, :] for p in sub.inputs]
            ref = evaluate_batch(sub.definition(**st), *args, kernel="reference").reshape(-1)
            via = np.asarray(sub.evaluate({p.name: [d.inputs[p.name]] for p in sub.inputs}, [st])[sub.outputs[0].name][0])
            assert np.array_equal(ref, via.astype(ref.dtype))
    sub = TM.subcircuit("gemm-coordinate", K=64)
    draws = [sub.draw(key, j) for j in range(64)]
    x, w = np.stack([d.inputs["x"] for d in draws]), np.stack([d.inputs["w"] for d in draws])
    from verity_vllm.program.registry import prims as VP
    assert VP.AmpereBF16TcDot16 is REGISTRY["AmpereBF16TcDot16_v2"]
    ref = evaluate_batch(bind(gemm.GemmCoordinate, K=64, DOT=VP.AmpereBF16TcDot16), x, w, kernel="reference")
    assert np.array_equal(ref, evaluate_batch(sub.target(), x, w))


def test_model_only_gemm_semantics_bind_but_do_not_generate(tmp_path):
    ids = [s["id"] for t in json.loads((CENSUS / "subcircuits.json").read_text()) if t["id"] in TM.TEMPLATES for s in t["subcircuits"]]
    for sid in ids:
        sub = TM.parse_subcircuit_id(sid)
        assert sub.id == sid
    sub = TM.parse_subcircuit_id("gemm-coordinate/k1536/sm90-mma-bf16")
    with pytest.raises(TM.NotEvaluable):
        G.generate(sub, 1, 0, tmp_path)


def test_gemm_ids_fit_the_census_schema():
    import re

    schema = json.loads((CENSUS / "schemas" / "input_sets.schema.json").read_text())
    pattern = re.compile(schema["items"]["properties"]["subcircuit"]["pattern"])
    for sem in [*TM.GEMM_SEMANTICS, *TM.GEMM_FP8]:
        assert pattern.match(TM.subcircuit("gemm-coordinate", K=2048, semantics=sem).id)


FP8 = sorted(TM.GEMM_FP8)


@pytest.mark.parametrize("sem", FP8)
def test_fp8_sets_hold_the_backends_accumulators(tmp_path, sem):
    """An FP8 set's y is the FP32 accumulator word of K/32 steps of the verity.ml.circuits.silicon model the backends chain with (ADA_E4M3_M16N8K32,
    HOPPER_E4M3_K32), from the IR reference (verity.ml.kernels.tc ADA_E4M3_DOT / HOPPER_E4M3_DOT); every operand byte is finite."""
    from verity.ml.circuits.silicon import models

    sub = TM.subcircuit("gemm-coordinate", K=64, semantics=sem, tile=4)
    s = IS.InputSet.open(G.generate(sub, 12, 5, tmp_path)["path"])
    assert IS.verify(s.path)["ok"]
    assert {k: (v["dtype"], v["words"]) for k, v in s.manifest["ports"].items()} == {"x": ("u8", 64), "w": ("u8", 64), "y": ("u32", 1)}
    assert s.relation["definition"] == TM.FP8_DEFINITION and s.subcircuit.id == sub.id == TM.parse_subcircuit_id(sub.id).id
    model = getattr(models, TM.GEMM_FP8[sem].model)
    ins = s.instances()
    for x, w, y in zip(ins["x"], ins["w"], ins["y"][:, 0]):
        acc = 0
        for c in range(0, 64, 32):
            acc = model.step(acc, x[c:c + 32].tolist(), w[c:c + 32].tolist())
        assert acc == int(y) and acc & 0x3FF == 0             # an FP8 accumulator word (B-Ligero's pack_public needs the low bits zero)
    assert not ((ins["x"] & 0x7F) == 0x7F).any() and not ((ins["w"] & 0x7F) == 0x7F).any()
    assert all(np.array_equal(ins["x"][i], ins["x"][4 * (i // 4)]) for i in range(12))
    (ref_id, digest), = s.provenance["program_digests"].items()
    assert ref_id == f'{sub.target().id}{{"K":64}}' and digest == TM.reference_digest(sub.target())


def test_nvfp4_sets_hold_the_models_accumulator(tmp_path):
    """NVFP4 (sm120): y is models.BLACKWELL_SM120_NVF4's chain over the E2M1 codes and UE4M3 block-16 scales, from the IR reference
    verity.ml.kernels.tc.SM120_NVF4_DOT; codes span 0..15, scales the valid 0x00..0x7E."""
    from verity.ml.circuits.silicon.models import BLACKWELL_SM120_NVF4 as M

    sid = "gemm-coordinate/k1536/sm120-mma-e2m1-nvf4"
    sub = TM.parse_subcircuit_id(sid)
    s = IS.InputSet.open(G.generate(sub, 8, 2, tmp_path)["path"])
    assert IS.verify(s.path)["ok"] and s.subcircuit.id == sid and s.relation["definition"] == TM.NVFP4_DEFINITION
    assert {k: (v["dtype"], v["words"]) for k, v in s.manifest["ports"].items()} == \
        {"x": ("u8", 1536), "w": ("u8", 1536), "sx": ("u8", 96), "sw": ("u8", 96), "y": ("u32", 1)}
    ins = s.instances()
    for i in range(8):
        assert M.chain(0, ins["x"][i].tolist(), ins["w"][i].tolist(), ins["sx"][i].tolist(), ins["sw"][i].tolist()) == int(ins["y"][i, 0])
    assert ins["x"].max() <= 15 and ins["sx"].max() <= 0x7E and ins["sw"].max() <= 0x7E
    assert [(p["K"], p["semantics"], n) for _t, p, n in G.SUITES["nvfp4"]] == [(1536, TM.NVFP4_SEMANTICS, 8192), (1536, TM.NVFP4_SEMANTICS, 16384)]


def test_e4m3_words_are_pinned():
    r = np.array([0, 126, 127, 253, 254, 0xFFFFFFFF], dtype=np.uint32)
    assert TM.e4m3_finite_words(r).tolist() == [0, 126, 128, 254, 0, 15]
    sub = TM.subcircuit("gemm-coordinate", K=64, semantics="sm89.mma.m16n8k32.e4m3")
    assert sub.draw(G.key_for(sub, 1), 0).inputs["x"][:8].tolist() == PINNED_ADA_X


PINNED_ADA_X = [121, 85, 110, 206, 231, 247, 42, 81]


def test_fp8_suite():
    fp8 = G.SUITES["fp8"]
    assert sorted((p["semantics"], p["K"], n) for _t, p, n in fp8) == sorted(
        (s, K, n) for s in FP8 for K, ns in ((2048, (6272, 16384)), (8192, (1920, 8192))) for n in ns)


def test_meta_and_ref(synthetic):
    s = IS.InputSet.open(synthetic["gemm-coordinate"]["path"])
    m = s.meta()
    assert (m["subcircuit"], m["template"], m["source"], m["n"], m["seed"]) == (s.subcircuit.id, "gemm-coordinate", "synthetic", 6, 11)
    ref = s.ref(0, 4, art="art:" + "0" * 64)
    from verity_numerical.bench import contract

    assert not contract._validate_instances(ref, 4, None)
    assert ref["range"] == [0, 4] and ref["content_digest"] == s.content_digest
    with pytest.raises(ValueError):
        s.ref(0, 7)


def test_suite_digests_only(tmp_path, monkeypatch):
    monkeypatch.setitem(G.SUITES, "tiny", [("rope-head", {"D": 8}, 5), ("silu-mul", {"I": 8}, 3)])
    full = G.suite("tiny", 9, tmp_path / "full")
    lean = G.suite("tiny", 9, tmp_path / "lean", digests_only=True)
    assert [s["content_digest"] for s in full] == [s["content_digest"] for s in lean]
    assert sorted(p.name for p in Path(lean[0]["path"]).iterdir()) == ["index.json", "manifest.json"]
    assert set(G.SUITES["101"][0][1]) == {"K", "semantics"} and len(G.SUITES["101"]) == 10


def test_cli(tmp_path, capsys):
    assert G.main(["--list"]) == 0
    listed = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert {x["template"] for x in listed} == set(SMALL)
    assert G.main(["--subcircuit", "silu-mul/i32/bf16", "--n", "3", "--seed", "2", "--out", str(tmp_path)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert IS.main(["verify", out["path"]]) == 0
    assert G.register_command(out["path"], "m.json")[:5] == ["research", "data", "put", "--kind", "input-set/v1"]
