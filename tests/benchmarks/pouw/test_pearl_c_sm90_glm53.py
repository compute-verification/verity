"""GLM-5.3's linears under pearl-c-h100-v1-h3 (benchmarks/pouw/pearl_c_sm90/glm53.py): one TP8 rank's shapes from the
checkpoint's config, the sm_90a kernels' domain verdict on each, the gap's formula against a Lean price twin, Llama-3.2-1B's
(llama32_1b.py) beside them, the H200 session's shapes each model's table's, and glm53_passes' full check of a retained
pass."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

HERE = Path(__file__).resolve().parents[3] / "benchmarks" / "pouw" / "pearl_c_sm90"
spec = importlib.util.spec_from_file_location("pearl_c_sm90_glm53", HERE / "glm53.py")
G = sys.modules[spec.name] = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)


def test_linears_are_one_tp8_ranks():
    """The capture map's shapes (glm-moe-dsa-eager-hopper-tp8-v0) at GLM-5.3's config: KVA = QL + KVL + DR, NHL = NH / 8,
    IKW = ID + IH, IL = I / 8, MIL = SIL = MI / 8; the per-rank K (256, 1536, 2048, 6144) are the FP8
    linears' k but kv_b_proj's 512."""
    got = {lin.name: (lin.n, lin.k) for lin in G.linears()}
    assert got == {
        "self_attn.fused_qkv_a_proj": (2624, 6144), "self_attn.q_b_proj": (2048, 2048), "self_attn.kv_b_proj": (3584, 512),
        "self_attn.o_proj": (6144, 2048), "self_attn.indexer.wq_b": (4096, 2048), "self_attn.indexer.wk_weights_proj": (160, 6144),
        "mlp.gate_up_proj": (3072, 6144), "mlp.down_proj": (6144, 1536), "mlp.shared_experts.gate_up_proj": (512, 6144),
        "mlp.shared_experts.down_proj": (6144, 256), "mlp.experts.w13": (512, 6144), "mlp.experts.w2": (6144, 256)}
    fp8 = {lin.k for lin in G.linears() if lin.stock != "cublas-bf16-gemm"}
    assert fp8 == {256, 512, 1536, 2048, 6144}
    assert G.expert_rows(64) == 2 and G.expert_rows(8192) == 256


def test_domain_refuses_the_short_k_and_narrow_n():
    """k = 512 is under the record's k ≥ 1,024 and not its census k = 256 (which takes both down projections); n = 160 is
    not a multiple of 64.  Every other linear is in, with the gap at both FADD readings, the data sheet's (cheaper add) a little
    above the record's, and the gap falling with n."""
    rows = {r["name"]: r for r in G.table()}
    refused = {name for name, r in rows.items() if r["refused"]}
    assert refused == {"self_attn.kv_b_proj", "self_attn.indexer.wk_weights_proj"}
    assert "1024 <= k" in rows["self_attn.kv_b_proj"]["refused"][0]
    assert "n of 64" in rows["self_attn.indexer.wk_weights_proj"]["refused"][0]
    for name, r in rows.items():
        if name not in refused:
            assert 0 < r["gamma"]["record"] < r["gamma"]["sheet"] < 0.011, name
    assert rows["mlp.experts.w13"]["gamma"]["record"] > rows["self_attn.o_proj"]["gamma"]["record"]


def test_expert_parallel_brings_the_experts_into_the_domain():
    """With the routed experts expert-parallel (EP = TP = 8) a rank holds 32 whole experts, w13 (4096, 6144) and w2
    (6144, 2048), both in the domain; the shared expert stays sharded, its down at the census's k = 256."""
    rows = {r["name"]: r for r in G.table(ep=True)}
    assert {x: (rows[x]["n"], rows[x]["k"]) for x in ("mlp.experts.w13", "mlp.experts.w2")} == {
        "mlp.experts.w13": (4096, 6144), "mlp.experts.w2": (6144, 2048)}
    assert {x for x, r in rows.items() if r["refused"]} == {"self_attn.kv_b_proj", "self_attn.indexer.wk_weights_proj"}
    assert (rows["mlp.shared_experts.down_proj"]["k"], rows["mlp.shared_experts.down_proj"]["refused"]) == (256, [])
    shapes = G.in_domain()
    assert (4096, 6144) in shapes and shapes[(4096, 6144)]["ep"] and not shapes[(6144, 2048)]["ep"]
    assert shapes[(6144, 2048)]["name"] == "self_attn.o_proj"


def test_gamma_is_the_lean_twins():
    """`Pouw.PearlC.pearlCGapSm120v1Cast32Rev1_8192`: v1 under rev1 (cap 1/400) at the sm_120 record, 8192³, the honest cast
    at 32 (24 over the credited 8), the gap 173561/22302500."""
    from verity.protocols.accounting.compute.pouw.prover.schemes import pearl_c as C
    from verity.protocols.accounting.compute.pouw.prover.schemes.pearl_c_device import SM120

    s = C.PearlC(forming="v1", device=replace(SM120, cap=Fraction(1, 400)), hashing="h3")
    assert G.gamma(s, 8192, 8192, Fraction(24)) == Fraction(173561, 22302500)
    h = G.scheme()
    assert h.name == G.SCHEME and h.prices.fadd == 32 and G.scheme(Fraction(1979, 67)).prices.fadd == Fraction(1979, 67)


def _llama():
    """llama32_1b.py, which imports glm53 by its own name from its directory."""
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("pearl_c_sm90_llama32_1b", HERE / "llama32_1b.py")
    mod = sys.modules[spec.name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_llama_linears_are_one_cards_and_all_in_the_domain():
    """Llama-3.2-1B at TP 1: qkv (32 + 2 · 8 heads of 64), o, the merged gate and up, down; every one in the sm_90a kernels'
    domain, with γ at both FADD readings (the data sheet's above the record's) by glm53's formula, and the decode M
    the vLLM served run serves at (8) among DECODE_M."""
    L = _llama()
    rows = {r["name"]: r for r in L.table()}
    assert {x: (r["n"], r["k"]) for x, r in rows.items()} == {
        "self_attn.qkv_proj": (3072, 2048), "self_attn.o_proj": (2048, 2048), "mlp.gate_up_proj": (16384, 2048),
        "mlp.down_proj": (2048, 8192)}
    assert not any(r["refused"] for r in rows.values())
    for name, r in rows.items():
        assert 0 < r["gamma"]["record"] < r["gamma"]["sheet"] < 0.011, name
        assert r["gamma"]["record"] == float(G.gamma(G.scheme(G.FADD_READINGS["record"]), r["n"], r["k"]))
    assert set(L.in_domain()) == {(r["n"], r["k"]) for r in rows.values()}
    assert 8 in L.DECODE_M and 2048 in L.PREFILL_M and L.SCHEME == G.SCHEME


def test_h200_session_runs_the_tables_shapes():
    """h200.sh's run.py SHAPES are every in-domain (n, k) of its MODEL (`in_domain`, each model's list), at decode's 64 and
    prefill's 8192; the capture after the check runs at MODEL's linears; and after every lease it gives up the passes' rows
    over a debit cap, then verifies them at the verdict's samples and checks their outputs in full."""
    sh = (HERE / "h200.sh").read_text()
    listed = dict(re.findall(r'^\s*(\w+)\) NKS="([\dx ]+)" ;;$', sh, re.M))
    assert set(listed) == {"glm53", "llama32_1b"} and "MODEL=${MODEL:-glm53}" in sh
    for model, mod in (("glm53", G), ("llama32_1b", _llama())):
        assert {tuple(map(int, x.split("x"))) for x in listed[model].split()} == set(mod.in_domain()), model
    assert 'for nk in $NKS; do\n  SHAPES="$SHAPES 64x$nk 8192x$nk"' in sh
    assert '"$D/glm53_passes.py" capture --model "$MODEL" --retain "$KEPT" --out "$O/passes"' in sh
    assert "KEPT=$(mktemp -d /tmp/pearlplus-glm53.XXXXXX)" in sh and """trap 'rm -rf "$S" "$KEPT"' EXIT""" in sh
    assert sh.index('"$D/glm53_passes.py" full --retain "$KEPT" --out "$O/passes" --jobs "$(nproc)"') > sh.index('echo "lease rc $?"')
    lease, volunteer = sh.index('echo "lease rc $?"'), sh.index('"$D/glm53_passes.py" volunteer --retain "$KEPT" --out "$O/passes" --jobs "$(nproc)"')
    verify = sh.index('"$D/glm53_passes.py" verify --retain "$KEPT" --out "$O/passes" --fast --draws "${DRAWS:-459}"')
    assert lease < volunteer < verify and _served_passes().DRAWS == 459


@pytest.mark.parametrize("needed, volunteer, step, rc", [
    ("yes reject", None, "1", "ship rc 2"), ("yes reject", "device", "device", "ship rc 2"),
    ("yes reject", "0", "0", "volunteer refused rc 2"), ("no uncredited", None, "0", "ship rc 2"),
    ("no uncredited", "1", "1", "ship rc 2"), ("", None, None, "volunteer-needed rc 2")])
def test_h200_session_gives_rows_up_exactly_while_the_verdict_rejects_one_over_its_cap(tmp_path, needed, volunteer, step, rc):
    """h200.sh asks ``served_passes.py volunteer-needed --verdict pass`` (a stand-in here) before the ship is unpacked:
    while the verdict glm53_passes.py verify takes rejects a pass on a sampled row over its cap, VOLUNTEER defaults to 1 and
    one that skips the give-up stops the session; once that verdict credits such a row nothing it defaults to 0, and any
    other answer stops the session too.  The stand-in ship is missing, so a session the guard lets through stops there."""
    tree, out = tmp_path / "tree", tmp_path / "out"
    (tree / "benchmarks" / "pouw" / "pearl_c_sm90").mkdir(parents=True)
    shutil.copy(HERE / "h200.sh", tree / "benchmarks" / "pouw" / "pearl_c_sm90")
    (sp := tree / "benchmarks" / "pouw" / "pearl_c_vllm" / "served_passes.py").parent.mkdir(parents=True)
    sp.write_text('import os, sys\nassert sys.argv[1:] == ["volunteer-needed", "--verdict", "pass"], sys.argv\n'
                  'print(os.environ["NEEDED"])\nsys.exit(0 if os.environ["NEEDED"] else 2)\n')
    (tree / "venv" / "bin").mkdir(parents=True)
    (tree / "venv" / "bin" / "python").symlink_to(sys.executable)
    out.mkdir()
    env = {"PATH": os.defpath, "HOME": str(tmp_path), "OUT": str(out), "RESEARCH_RUN_DIR": str(tmp_path / "run"),
           "VERITY_BENCH_VENV": str(tree / "venv"), "NEEDED": needed, **({} if volunteer is None else {"VOLUNTEER": volunteer})}
    r = subprocess.run(["bash", "benchmarks/pouw/pearl_c_sm90/h200.sh"], cwd=tree, env=env, capture_output=True, text=True,
                       check=False)
    lines = (out / "rc.txt").read_text().splitlines()
    assert r.returncode == 2 and len(lines) == 1 and lines[0].startswith(rc), r.stdout + r.stderr
    said = out / "volunteer-needed.txt"
    assert (said.read_text() if said.exists() else None) == (step and f"volunteer-needed: {needed}; VOLUNTEER={step}\n")


def _served_passes():
    spec = importlib.util.spec_from_file_location("pearl_c_vllm_served_passes", HERE.parent / "pearl_c_vllm" / "served_passes.py")
    mod = sys.modules[spec.name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _passes_module():
    spec = importlib.util.spec_from_file_location("pearl_c_sm90_glm53_passes", HERE / "glm53_passes.py")
    if spec.name not in sys.modules:
        mod = sys.modules[spec.name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return sys.modules[spec.name]


def test_the_passes_directory_is_refused_when_it_is_holds_or_sits_inside_a_runs_directory(tmp_path):
    why = _passes_module().retain_why
    run, out = tmp_path / "run", tmp_path / "run" / "out"
    for retain in (out, out / "kept", run, tmp_path):
        assert "never go under a run's directory" in why(retain, out, str(run)), retain
    assert why(out / "kept", out, None) is not None
    assert why(tmp_path / "scratch", out, str(run)) is None


@pytest.fixture(scope="module")
def retained(tmp_path_factory):
    """A retained pass in glm53_passes' layout, as `prefill` and as `decode` (the same pass): two calls of a test instance of
    pearl-c-h100-v1-h3 (k = 128, 16 x 16 tiles), one with 12 filler rows, proved by the scheme's reference prover and written
    as the device run writes them, and each call's served y from the scheme over the whole call as one tile (U and α), its
    two FP32 products numpy's and its BF16 rounding torch's, committed in its manifest as the served path does
    (`served_y.retain_y`: load refuses a credited call with no y root)."""
    import random
    import shutil

    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    GP = _passes_module()
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "verity/protocols/accounting/compute/pouw/prover/clear"))
    import clear_work as CW
    from clear_audit import Prover
    from verity.protocols.accounting.compute.pouw.prover import schemes as pouw_schemes
    from verity_torch.pouw.pearl_c import run_device as PPDR

    from verity.protocols.accounting.compute.pouw.verifier import core_check as CC
    from verity.protocols.accounting.compute.pouw.verifier.audit import (
        HM96_SALT,
        HM96_TRANSCRIPT,
        HM96_WEIGHTS,
        ROW_SCHEMA,
        TILE_SCHEMA,
        Epoch,
        Workload,
        activation_binding,
        hm96_salt,
        rows_domain,
        transcript_domain,
        weight_binding,
        weights_root,
    )
    from verity.protocols.accounting.compute.pouw.verifier.coins import Challenger
    from verity.protocols.accounting.compute.pouw.verifier.protocol import Context, Matmul, Shape, Tile
    from verity.protocols.accounting.compute.pouw.prover.schemes import pearl_c_work as PW
    from verity.protocols.accounting.compute.pouw.prover.serving.pearl_c import served_y as SY
    from verity.commitments.merkle import tree_levels

    scheme = "pearl-c-h100-v1-h3"
    s = pouw_schemes.get(scheme, k_min=32, tile=(16, 16))
    rng, k = np.random.default_rng(11), 128

    def words(r):
        return (rng.standard_normal((r, k)).astype(np.float32)).view(np.uint32) & 0xFFFF0000

    def rows(x):
        return [r.tobytes() for r in x]

    def bf16(x):
        return b"".join((np.frombuffer(r, "<u4") >> 16).astype("<u2").tobytes() for r in rows(x))

    def laid(d, leaves):
        return b"".join(b"".join(level) for level in tree_levels(d, leaves))

    def inv(alpha):
        return np.float32(1) / np.array(alpha, np.uint32).view(np.float32)

    wrows, acts = {0: words(16), 1: words(32)}, {0: words(32), 1: words(16)}
    acts[0][20:] = 0
    work = Workload((Matmul(0, Shape(32, k, 16), 0), Matmul(1, Shape(16, k, 32), 1)))
    nonces, key = iter([HM96_WEIGHTS + w for w in wrows] + [u.index for u in work.matmuls] + [HM96_TRANSCRIPT]), bytes([1]) * 32

    def pinned(n):
        nonce = next(nonces)
        return [hm96_salt(key, nonce, i) for i in range(n)]

    def kept(tree):
        return [tree.open(i).value for i in range(tree.domain.count)]

    def salts(tree):
        return b"".join(v[-HM96_SALT:] for v in kept(tree))

    with mock.patch.object(sys.modules[Prover.__module__], "fresh_salts", pinned):
        prover = Prover(s, {w: rows(x) for w, x in wrows.items()})

    ch = Challenger(0, random.Random(7).randbytes)
    coins = ch.commitment, ch.open("salt", weights_root(prover.weight_roots))
    epoch = Epoch.start(work, prover.weight_roots, *coins)
    with mock.patch.object(sys.modules[Prover.__module__], "fresh_salts", pinned):
        tr = prover.prove(epoch, {i: rows(x) for i, x in acts.items()})
    splits = {w: PW.Split(k) for w in wrows}
    records = CW.declare(s, epoch, {i: rows(x) for i, x in acts.items()}, {w: rows(x) for w, x in wrows.items()}, splits,
                         real_rows={0: 20})
    base = tmp_path_factory.mktemp("retained")
    d = base / "pass"
    for sub in ("calls", "weights", "y"):
        (d / sub).mkdir(parents=True)
    weights = {}
    for w, x in wrows.items():
        dom = rows_domain(weight_binding(w), -2, len(x), s.commitment_hash)
        weights[str(w)] = {"n": len(x), "k": k, "root": prover.weight_roots[w].hex(), "excluded_b": list(records[w].excluded_b),
                           "rows": PPDR._write(d / "weights" / f"{w}.rows.bf16", bf16(x), digest=False),
                           "tree": PPDR._write(d / "weights" / f"{w}.tree",
                                              laid(dom, [dom.leaf(j, j, ROW_SCHEMA, v) for j, v in enumerate(kept(prover.weight_trees[w]))])),
                           "salts": PPDR._write(d / "weights" / f"{w}.salts", salts(prover.weight_trees[w]))}
    matmuls, y_words = [], 0
    for u, rec, root, at in zip(work.matmuls, records, tr.activation_roots, prover._act_trees):
        x, m, n = rows(acts[u.index]), u.shape.m, u.shape.n
        dom = rows_domain(activation_binding(epoch.salt, u.index), u.index, m, s.commitment_hash)
        matmuls.append({"index": u.index, "m": m, "k": k, "n": n, "weight": u.weight, "real_rows": rec.real_rows,
                        "k_kept": rec.k_kept, "excluded_a": list(rec.excluded_a), "root_a": root.hex(),
                        "rows": PPDR._write(d / "calls" / f"{u.index}.rows.bf16", bf16(acts[u.index]), digest=False),
                        "tree": PPDR._write(d / "calls" / f"{u.index}.tree",
                                           laid(dom, [dom.leaf(i, i, ROW_SCHEMA, v) for i, v in enumerate(kept(at))])),
                        "salts": PPDR._write(d / "calls" / f"{u.index}.salts", salts(at))})
        ctx = Context(epoch.salt, u, root, epoch.weight_roots[u.weight])
        noise = s.noise(ctx)
        ops = s.encode(ctx, noise, x, rows(wrows[u.weight]), Tile(tuple(range(m)), tuple(range(n))))
        uu = np.array(s.checked(ctx, ops).u, np.uint32)[:rec.real_rows].view(np.float32)
        y = torch.from_numpy((uu * inv(ops.a.alpha)[:rec.real_rows, None]) * inv(ops.b.alpha)[None, :]).to(torch.bfloat16)
        served = y.view(torch.int16).numpy().tobytes()
        (d / "y" / f"{u.index}.bf16").write_bytes(served)
        matmuls[-1]["y"] = SY.retain_y(d, u.index, served, n, epoch.salt)
        y_words += y.numel()
    count = tr.commitment.count
    leaves = [prover.digest(t) for t in range(count)]
    dt = transcript_domain(epoch, count, s.commitment_hash)
    head = {"schema": CC.MANIFEST_SCHEMA, "run": "r20261023-120000-test", "entry": f"pouw:{scheme}", "scheme": scheme,
            "instance": {"k_min": 32, "tile": [16, 16]}, "coins": coins[0].to_json(), "salt_opening": coins[1].to_json(),
            "salt": epoch.salt.hex(), "records_root": PW.records_root(epoch.salt, epoch.weights_root, splits, records).hex(),
            "commitment": {"root": tr.commitment.root.hex(), "count": count}, "leaves": PPDR._write(d / "leaves", b"".join(leaves)),
            "tree_t": PPDR._write(d / "tree_t", laid(dt, [dt.leaf(i, i, TILE_SCHEMA, v) for i, v in enumerate(kept(prover._tree))])),
            "salts_t": PPDR._write(d / "salts_t", salts(prover._tree))}
    PPDR.write_manifest(d, head, weights, matmuls)
    (base / "coins.json").write_text(json.dumps(coins[0].to_json()) + "\n")
    for name in GP.PASSES:
        shutil.copytree(d, base / name)
    shutil.rmtree(d)
    return SimpleNamespace(base=base, count=count, y_words=y_words, GP=GP)


def _copy(retained, tmp_path) -> tuple[Path, Path]:
    """A copy of the retained passes, and a public directory apart from it holding their coins.json."""
    import shutil

    shutil.copytree(retained.base, tmp_path / "passes")
    (tmp_path / "public").mkdir()
    shutil.copy(retained.base / "coins.json", tmp_path / "public" / "coins.json")
    return tmp_path / "passes", tmp_path / "public"


def test_full_checks_every_tile_and_every_served_y_word(retained, tmp_path):
    """`full` takes every tile of prefill and decode through the verifier's check_tile (openings, then C~ and U against the
    leaf) and every real row's served y against y_unit, on the batch forms with the scalar reference on every 2nd tile too."""
    out, pub = _copy(retained, tmp_path)
    assert retained.GP.main(["full", "--retain", str(out), "--out", str(pub), "--jobs", "2", "--cross-check", "2"]) == 0
    res = json.loads((pub / "full.json").read_text())
    for name in retained.GP.PASSES:
        r = res[name]
        assert r["ok"] and r["tiles"] == retained.count and r["failed"] == 0, r["line"]
        assert r["shapes"] == {"16x128": 2, "32x128": 2} and r["words"] == 32 * 16 + 16 * 32
        assert r["y_words"] == retained.y_words == 20 * 16 + 16 * 32
        assert r["replay"] == "batch" and r["cross_checked"] == (retained.count + 1) // 2 and r["cross_check_failed"] == []
    assert res["ok"]


def test_full_fails_on_one_served_y_word_or_one_activation_row_and_names_it(retained, tmp_path):
    """One served y word off (decode's call 1, row 3, column 17) fails that tile alone, naming the word; one byte of an
    activation row off (prefill's call 0, row 4) fails that row's tiles at their openings."""
    np = pytest.importorskip("numpy")
    out, pub = _copy(retained, tmp_path)
    y = out / "decode" / "y" / "1.bf16"
    words = np.fromfile(y, "<u2").reshape(16, 32)
    words[3, 17] ^= 1
    words.tofile(y)
    a = out / "prefill" / "calls" / "0.rows.bf16"
    raw = bytearray(a.read_bytes())
    raw[2 * 128 * 4] ^= 0x40
    a.write_bytes(bytes(raw))
    assert retained.GP.main(["full", "--retain", str(out), "--out", str(pub), "--jobs", "1", "--cross-check", "0"]) == 1
    res = json.loads((pub / "full.json").read_text())
    dec, pre = res["decode"], res["prefill"]
    assert dec["failed"] == 1 and "served y differs at 1 of 256 words, first row 3 column 17" in dec["failures"][0]["reason"]
    assert pre["failed"] == 1 and pre["failures"] == [{"tile": 0, "reason": "activation opening"}]
    assert not res["ok"]


@pytest.mark.slow
def test_volunteer_leaves_each_pass_no_tile_over_an_inert_bound_and_its_transcript_as_captured(retained, tmp_path):
    """`volunteer` runs `volunteer_pass` on prefill's and decode's passes: each pass's records then leave no tile over a
    debit cap, its commitment is the capture's, its manifest's records root is the one written, and every tile passes the
    verifier's check under it."""
    from verity.protocols.accounting.compute.pouw.verifier.coins import CoinCommitment

    out, pub = _copy(retained, tmp_path)
    before = {n: json.loads((out / n / "manifest.json").read_text()) for n in retained.GP.PASSES}
    assert retained.GP.main(["volunteer", "--retain", str(out), "--out", str(pub), "--jobs", "1"]) == 0
    import verify_run as VR

    res, coins = json.loads((pub / "volunteer.json").read_text()), CoinCommitment.from_json(json.loads((pub / "coins.json").read_text()))
    for name in retained.GP.PASSES:
        r, man = res[name], json.loads((out / name / "manifest.json").read_text())
        assert r["ok"] and r["left"] == [] and r["tiles"] == retained.count, r["line"]
        assert man["commitment"] == before[name]["commitment"] and man["records_root"] == r["records_root"]["after"]
        p = VR.load(out / name / "manifest.json", coins)
        got, _ = VR.replay([p], [(0, ref.matmul, ref.id) for ref in p.verifier.refs], 1)
        assert [reason for _, _, ok, reason, *_ in got if not ok] == []
    assert res["ok"]


def test_full_requires_the_served_y_unless_told_the_pass_has_none(retained, tmp_path):
    """A pass without its served y fails `full`, which checks C~ and U alone only under --without-y."""
    import shutil

    out, pub = _copy(retained, tmp_path)
    for name in retained.GP.PASSES:
        shutil.rmtree(out / name / "y")
    assert retained.GP.main(["full", "--retain", str(out), "--out", str(pub), "--jobs", "1"]) == 1
    res = json.loads((pub / "full.json").read_text())
    assert all("call 0: no served y of 20 x 16 BF16 words" in res[x]["error"] for x in retained.GP.PASSES) and not res["ok"]
    assert retained.GP.main(["full", "--retain", str(out), "--out", str(pub), "--jobs", "1", "--without-y"]) == 0
    res = json.loads((pub / "full.json").read_text())
    assert all(res[x]["y_words"] == 0 and "served y not retained" in res[x]["line"] for x in retained.GP.PASSES)
