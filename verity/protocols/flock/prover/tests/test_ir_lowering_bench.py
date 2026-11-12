"""C-Flock's generic IR lowering on sets the numerical bench generates and in the bench's IR cell: the lowered templates
against the IR evaluator on synthetic sets, and an IR cell's plan, run result and domain."""
from __future__ import annotations

import random

import numpy as np
import pytest

P = pytest.importorskip("verity.definitions.registry.prims")
from test_ir_lowering import LOWERED, REG, SLICES, _check_set  # noqa: E402
from verity_flock import ir_lower as IL  # noqa: E402
from verity_flock import subcircuit_templates as TM  # noqa: E402
from verity_flock.input_sets import InputSet  # noqa: E402
from verity_numerical.bench.generate import generate  # noqa: E402

INGRESS = {"MODEL": "m", "LP": 3, "V": 1000}


@pytest.mark.parametrize("template", sorted(LOWERED))
def test_synthetic_sets(template, tmp_path):
    sub = TM.subcircuit(template, **LOWERED[template])
    n = 24 if template == "rope-head" else 3
    summary = generate(sub, n, 20260926, tmp_path)
    s = InputSet.open(summary["path"])
    assert s.source == "synthetic"
    assert _check_set(s) == 0


def test_the_ingress_parse_is_one_unit_equal_to_its_definition(tmp_path):
    """``ingress-prompt`` (the I/O shell's ``IngressPrompt_v1``) lowers as ONE unit (``ir_lower.whole``) whose input row is the
    body, whole SHA-512 blocks: its rows equal the IR on a synthetic set of schema bodies, and give ``io_shell.reference``'s
    (ok, ids, max_tokens) on the same bodies with one byte changed, which leaves the schema or moves a field."""
    from verity.definitions.registry import io_shell as X

    mod, sub = REG.module("ingress-prompt"), TM.subcircuit("ingress-prompt", **INGRESS)
    s = InputSet.open(generate(sub, 4, 20261024, tmp_path)["path"])
    assert _check_set(s) == 0
    low = mod.lowering(sub)
    assert low.units.n == 1 and sum(low.units.in_widths) % 1024 == 0 and low.sha256 == mod.PINS[tuple(INGRESS.values())]
    rng, LP = random.Random(3), INGRESS["LP"]
    fin = IL.flat_ports(s, sub.inputs, 0, s.n).copy()
    for row in fin:
        row[rng.randrange(len(row))] ^= 1 + rng.randrange(255)
    got, _cuts, ok = low.evaluate(fin)
    want = [X.reference(bytes(int(b) for b in row), *INGRESS.values()) for row in fin]
    assert ok == (1 << s.n) - 1 and [(int(r[0]), [int(x) for x in r[1:1 + LP]], int(r[1 + LP])) for r in got] == want
    assert any(w[0] == 0 for w in want)


def test_the_greedy_select_is_its_chunk_and_combine_units(tmp_path):
    """``token-select-chunk`` and ``token-select-combine`` (``dead_free.TokenSelect_v4``'s two calls) lower as one unit each
    (``ir_lower.whole``), the chunk's input row 64 bf16 logits, one SHA-512 block: their rows equal the IR on synthetic sets, and
    the chunks' units feeding the combine's give ``TokenSelect_v1``'s token on rows of V = 200 whose maximum is tied across a
    chunk boundary, behind a NaN at a chunk's first logit, or alone in the last, short chunk."""
    from verity.circuits.defs import bind
    from verity.circuits.evaluation import evaluate_batch
    from verity.definitions.registry import b1

    V = 200
    lows = {}
    for t, p in (("token-select-chunk", {"C": 64}), ("token-select-combine", {"V": V, "C": 64})):
        mod, sub = REG.module(t), TM.subcircuit(t, **p)
        assert _check_set(InputSet.open(generate(sub, 3, 20261024, tmp_path / t)["path"])) == 0
        lows[t] = low = mod.lowering(sub)
        assert low.units.n == 1 and low.sha256 == mod.PINS[tuple(p.values()) if len(p) > 1 else p["C"]]
    assert sum(lows["token-select-chunk"].units.in_widths) == 1024
    rng = np.random.default_rng(7)
    x = rng.integers(0, 1 << 16, (6, V)).astype(np.uint64)
    x[1] = 0xBF80
    x[1, [63, 64]] = 0x3F80
    x[2] = 0x3F80
    x[2, 128], x[2, 150] = 0x7FC0, 0x4000
    x[3] = 0xFF80
    x[3, 197] = 0xBF80
    best, idx = [], []
    for lo, hi in b1.chunks(V, 64):
        low = lows["token-select-chunk"] if hi - lo == 64 else REG.module("token-select-chunk").lowering(
            TM.subcircuit("token-select-chunk", C=hi - lo))
        got, _cuts, ok = low.evaluate(x[:, lo:hi])
        assert ok == (1 << len(x)) - 1
        best.append(got[:, 0])
        idx.append(got[:, 1])
    got, _cuts, ok = lows["token-select-combine"].evaluate(np.column_stack([x[:, 0], *best, *idx]))
    want = np.asarray(evaluate_batch(bind(b1.TokenSelect, V=V), x)).reshape(len(x))
    assert ok == (1 << len(x)) - 1 and got[:, 0].tolist() == want.tolist()
    assert want[1:4].tolist() == [63, 150, 197]


@pytest.mark.slow
def test_rmsnorm_fused_small_row_against_the_ir(tmp_path):
    """N = 64: two warps of 32 one-element threads; spine synthetic rows, outputs and warp aggregates both compared."""
    sub = TM.subcircuit("rmsnorm-fused-cuda", N=64, EPS=1e-5)
    s = InputSet.open(generate(sub, 6, 20260926, tmp_path)["path"])
    assert REG.module("rmsnorm-fused-cuda").lowering(sub).units.n == 2
    assert _check_set(s) == 0


def test_a_wrong_row_scalar_changes_the_outputs(tmp_path):
    sub = TM.subcircuit("rmsnorm-fused-cuda", N=64, EPS=1e-5)
    s = InputSet.open(generate(sub, 2, 7, tmp_path)["path"])
    low = REG.module("rmsnorm-fused-cuda").lowering(sub)
    fin = IL.flat_ports(s, sub.inputs, 0, s.n)
    cuts = IL.cut_words(low, fin)
    good, _, _ = low.evaluate(fin, cuts)
    cuts[:, -1] ^= 1 << 12
    bad, _, _ = low.evaluate(fin, cuts)
    assert (good != bad).any(axis=1).all()


@pytest.mark.slow
def test_the_netlist_accounts_for_every_cut_word(tmp_path):
    """IR1 (the Python half; the Rust verifier refuses the rest, `flock-ir-block selftest`): the netlist's CUT line lists each
    unit's cut ports and the native tail; every cut word is computed exactly once, by a unit or by the tail, from words defined
    before. The instance header only counts them."""
    import json

    sub = TM.subcircuit("rmsnorm-fused-cuda", N=64, EPS=1e-5)
    s = InputSet.open(generate(sub, 2, 3, tmp_path / "set")["path"])
    low = REG.module("rmsnorm-fused-cuda").lowering(sub)
    head = json.loads(IL.stage(low, s, 0, 2, tmp_path / "st").read_bytes().split(b"\n", 1)[0])
    cut = json.loads((tmp_path / "st" / "net.txt").read_text().splitlines()[-1].removeprefix("CUT "))
    assert cut == low.cut and "cut" not in head
    assert head["cut_words"] == cut["words"] == len(low.units.cut) == 3 and len(cut["in"]) == len(cut["out"]) == low.units.n == 2
    produced = [k for unit in cut["out"] for _, k in unit]
    tail_out = [op[0] for op in cut["tail"] if op[0] < cut["words"]]
    assert sorted(produced + tail_out) == list(range(cut["words"]))
    defined = set(produced)
    for op in cut["tail"]:
        assert set(op[2:] if op[1] != "Const" else []) <= defined
        defined.add(op[0])
    assert {op[1] for op in cut["tail"]} <= {"Const", *IL.TAIL_PRIMS}
    rs = InputSet.open(SLICES / "rope-head-d64")
    rope = REG.module("rope-head").lowering(rs.subcircuit)
    head = json.loads(IL.stage(rope, rs, 0, 1, tmp_path / "r").read_bytes().split(b"\n", 1)[0])
    assert rope.cut is None and head["cut_words"] == 0 and not rope.text.splitlines()[-1].startswith("CUT")


@pytest.mark.slow
def test_frame_staging_carries_the_cut_words(tmp_path):
    """A cut template's frame file ends with every instance's cut words (the IR's), which the verifier checks against the
    netlist's pinned tail; the empty run slot of a three-port component block is padding."""
    import json

    sub = TM.subcircuit("rmsnorm-fused-cuda", N=64, EPS=1e-5)
    s = InputSet.open(generate(sub, 3, 5, tmp_path / "set")["path"])
    low = REG.module("rmsnorm-fused-cuda").lowering(sub)
    path = REG.module("rmsnorm-fused-cuda").stage(s, 0, 3, tmp_path / "st")
    head, body = path.read_bytes().split(b"\n", 1)
    h = json.loads(head)
    assert h["cut_words"] == 3 and "cut" not in h and h["layout"]["chunks_per_block"] == 4
    assert h["blocks"][0][0] == [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], None]
    cuts = np.frombuffer(body[-3 * 3 * 4:], dtype="<u4").reshape(3, 3)
    assert (cuts == IL.cut_words(low, IL.flat_ports(s, sub.inputs, 0, 3))).all()


def test_non_gemm_results_carry_no_gemm_target():
    from verity_numerical.bench import contract

    base = dict(domain="total", B=4, proof_class="NON_ZK_PROOF", authentication="included-hash", zk_mode="none",
                security={"target": -128}, hardware={"gpu": None, "cpu": {"model": "x"}}, software={"backend": {"name": "b", "commit": None},
                "toolchain": {}}, preprocessing="none", retention="retained", instances=None)
    ok = contract.fingerprint(profile=None, K=None, subcircuit="rope-head/d64/neox-bf16", **base)
    assert contract.validate_fingerprint(ok) == []
    assert any("profile must be null" in p for p in contract.validate_fingerprint({**ok, "profile": contract.PROFILE}))
    assert any("profile must be a target" in p for p in contract.validate_fingerprint({**ok, "subcircuit": None}))
    rates = [contract.measurement(contract.OVERHEAD, 1.0)]
    assert contract._check_rates(ok, rates) and "GEMM rates" in contract._check_rates(ok, rates)[0]


def test_an_ir_cell_plans_through_the_spine():
    from verity_numerical.bench import cell as C

    two_machines = {"prover": {"machine_id": "m-p", "public_ip": "192.0.2.5"}, "verifier": {"machine_id": "m-v", "public_ip": "192.0.2.4"},
                    "link": {"peer": "192.0.2.4:7400", "peer_ip": "192.0.2.4"}}
    c = C.plan("C-interactive", "rope-head/d64/neox-bf16+frame-v3/blake3-keyed", str(SLICES / "rope-head-d64"), "vy-p", verifier="vy-v",
               verifier_addr="192.0.2.4:7400", rtt_target="192.0.2.4:22", lane="flock-ir-lowering", points="2 4", runs=5, placement=two_machines)
    assert c.relation == "irf:rope-head"
    v, p = c.jobs
    assert v["command"][:3] == ["bash", "verity/protocols/flock/pod/33-ir-cell.sh", "ROLE=verifier"] and "TEMPLATE=rope-head" in v["command"]
    assert "--send" in v["argv"] and v["argv"][v["argv"].index("--send") + 1] == p["argv"][p["argv"].index("--send") + 1]
    assert "VERIFIER=192.0.2.4:7400" in p["command"] and "LOOPBACK=1" in p["command"]


def test_attention_key_classes_and_the_class_manifest(tmp_path):
    """Key-count classes [1, 128], [129, 256], [257, 512], ...; a class manifest lists every T of its range with its v3 netlist's
    sha256 and the unit rows they share (the pin is the manifest's sha256); a class cell's sub-batches split at T boundaries."""
    import hashlib
    import json

    from verity_flock import ir_bench
    from verity_flock.templates import attention_head as A

    assert [A.key_class(t) for t in (1, 128, 129, 256, 257, 287, 512, 513)] == [(1, 128)] * 2 + [(129, 256)] * 2 + [(257, 512)] * 3 + [(513, 1024)]
    sub = TM.subcircuit("attention-head", D=64, BN=128)
    m = A.class_manifest(sub, 1, 3)
    d = json.loads(m)
    assert d["format"] == A.CLASS_FORMAT and d["T"] == [1, 3] and sorted(d["nets"]) == ["1", "2", "3"]
    assert m == (json.dumps(d, sort_keys=True, separators=(",", ":")) + "\n").encode()
    for T in (1, 2, 3):
        low = A.cached_lowering(sub, T)
        assert d["nets"][str(T)] == low.sha256 == hashlib.sha256(low.text.encode()).hexdigest()
        assert A.unit_rows_sha256(low.text) == d["unit_rows"]

    class _K(ir_bench.KeyClass):
        def __init__(self, T):
            self.T = T

    assert _K([1, 1, 1, 2, 2, 5]).subs(6, 2) == [(0, 2), (2, 3), (3, 5), (5, 6)]
    assert _K([1, 1, 1, 2, 2, 5]).subs(4, 16) == [(0, 3), (3, 4)]


def test_an_ir_run_result_is_a_point_not_a_second_cell():
    """The runner publishes a run's result.json under its own kind: ir_bench marks it a bench-point/v1 (a known kind), so the
    cell's one bench-result/v1 is bench.cell register's, and verity_flock.register drops the kind it read."""
    from pathlib import Path

    from research.store.kinds import KINDS
    from verity_flock import ir_bench, register

    assert ir_bench.POINT_KIND == "bench-point/v1" and ir_bench.POINT_KIND in KINDS
    assert 'raw.pop("kind", None)' in Path(register.__file__).read_text()


def test_a_cells_domain_is_the_one_its_statement_proves():
    """A red-team finding: ``ir_bench`` and ``bench`` wrote ``domain: finite`` for every cell.
    An IR lowering asserts nothing, so every encoding is proven to give the IR's output: ``total``.  A pure unit asserts its
    operands (and accumulator) finite: ``finite-only``, unless it is a total unit."""
    from types import SimpleNamespace

    from verity_flock import lowering as FL

    low = REG.module("silu-mul").frame_lowering(TM.subcircuit("silu-mul", I=8192))
    assert not low.circuit.asserts and low.domain == "total"
    assert IL.Lowering.domain.fget(SimpleNamespace(circuit=SimpleNamespace(asserts=[("x not finite", 0)]))) == "finite-only"
    for relation, pipe in FL.PIPES.items():
        assert pipe.domain == ("finite-only" if FL._layout(pipe)[0].asserts else "total"), relation
    assert {p.domain for p in FL.PIPES.values()} == {"finite-only"}
    assert FL.Pipe("t", None, (8, 8), 25, -132, True, "AMPERE_BF16_M16N8K16", total=True).domain == "total"


@pytest.mark.parametrize("D,BN,T", [(64, 128, 129), (128, 64, 65)])
def test_attention_v18_units_reproduce_the_ir_on_a_masked_from_head(D, BN, T, tmp_path):
    """Two key blocks at a decode launch: MASKED_FROM = 1, so the second block rescales by the running max as it is. On synthetic
    heads the laid-out units reproduce every tensor-core value of the IR."""
    sub = TM.attention_head_v18(D, BN, T=f"{T}:{T}")
    s = InputSet.open(generate(sub, 2, 20260926, tmp_path)["path"])
    mod = REG.module("attention-head-v18")
    assert mod.key_count(s) == T and sub.definition(T).statics["MASKED_FROM"] == 1
    low = mod.lowering_for_set(s)
    fin = IL.flat_ports(s, sub.inputs, 0, s.n)
    cuts = IL.cut_words(low, fin)
    _res, cut_out, ok = low.evaluate(fin, cuts)
    assert ok == (1 << (s.n * low.units.n)) - 1 and (cut_out == cuts).all()
