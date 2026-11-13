"""`inference_only.py`: a served run's inference-only verdict, its declared circuit's blocks against the (c, k, n) it
registered before it served, by `NciCheck.Count`'s rule. `BitAtx32_v1`'s price is an explicit circuit that computes the
primitive, input by input; the verdict's rule is the count rule's reference (`nci_reference.check_count_detail`) on its
rows, and the Lean instance it makes is one on which that reference counts the verdict's gates; n is the H200's sessions
at the declared utilization over the lease, rounded up to the grid, from the class's fixed load alone; the policy is its
session's gates and fresh bits with every weight held, and n, under an H200 int8 serve's φ or the one ``--serve`` states,
cut at Pearl+'s linears where it says, and a BF16 checkpoint under the int8 serve's φ is refused by name; and on toy
serve dirs over small circuit records the verdict
accepts within (c, k, n), refuses past each, and refuses as undecided what the rule can't decide (no policy, one written
after the serve began unless said retroactive, a primitive with no price, a closed window with no seal, a window the
class doesn't take), its public form naming no window and publishing no count: it judges n on the exact count, which
only the private form holds, so runs whose counts differ publish one verdict, unless n falls between them, and then
differ in their judgment alone. Of `nci-check --count` it takes the whole verdict, with ``--detail`` where the binary's
usage names it, into the private form alone, and publishes its gates a block, fresh bits and ``complies``, or its
refusal, and none of its counts, total or failure. The slow test runs `nci-check --count`."""
from __future__ import annotations

import json
import random
import stat
import sys
from fractions import Fraction
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import inference_only as IO  # noqa: E402
import served_verdicts as SV  # noqa: E402
from verity.definitions import boolean as BL  # noqa: E402
from verity.definitions.registry.catalog import load  # noqa: E402
from verity.definitions.registry.sampling import bit_at  # noqa: E402
from verity.commitments.windows import interface as IF  # noqa: E402
from verity_vllm.program.predict import block_counts as BC  # noqa: E402

GEMM = 'Gemm_v2{K=64,N=8,DOT={"fn":"HopperBF16WgmmaDot16_v1"}}'
ROPE = "RoPEHead_v1{D=16}"
FEED = "TokenFeed_v1{NP=4,FIRST=false}"
MODEL = "verity-int8/Llama-3.2-1B-INT8"
#: a checkpoint served in BF16, at its revision
BF16 = ("unsloth/Llama-3.2-1B", "9535bd9b1d1dea6acafbdc4813b728796aeb28da")
#: the toy serve's start (its serve.json's clocks' time), and a policy written before and after it
STARTED = 1_762_000_000.0
BEFORE, AFTER = "2025-11-01T12:26:39+00:00", "2025-11-01T12:26:41+00:00"


def _fn(tid: str):
    """``tid``'s Definition, from the whole registry."""
    load()
    return IF.resolve(tid)


@pytest.fixture(scope="module")
def small() -> dict:
    """`test_block_counts`' record: two slots and three steps, four `GEMM` units (a slot's two in one entry) and six
    `ROPE` units (a slot's three across the steps)."""
    entries = (IF.InferenceEntry(GEMM, 0, (2, 2), ("slot", "-"), 0), IF.InferenceEntry(ROPE, 0, (3, 2), ("step", "slot"), 0))
    circuit = IF.Circuit("test/blocks", ((_fn(GEMM), 4), (_fn(ROPE), 6)),
                         blocks=IF.Blocks(slots=2, steps=3, inference=entries))
    return IF.descriptor_record(circuit, 1)


@pytest.fixture(scope="module")
def feed() -> dict:
    """Two slots of `FEED`, which reads `BitAtx32_v1` (no Boolean version), one step."""
    entries = (IF.InferenceEntry(FEED, 0, (2,), ("slot",), 0),)
    return IF.descriptor_record(IF.Circuit("test/feed", ((_fn(FEED), 2),),
                                           blocks=IF.Blocks(slots=2, steps=1, inference=entries)), 1)


def test_bit_at_is_priced_by_a_circuit_that_computes_it():
    """`bit_at_gates(W)` is bit i of a W-bit word, 0 for an i32 outside [0, W), on every index near the range and its
    extremes and on random words and indices; at W = 32 it is 147 two-input gates. A width the circuit can't make has no
    price, and nor has any other primitive."""
    rng = random.Random(20261103)
    for width in (1, 2, 8, 32, 64):
        gates, prim = IO.bit_at_gates(width), bit_at(width)
        b = width.bit_length() - 1
        assert len(gates) == (32 - b) + (31 - b) + 3 * (width - 1) + 1
        words = [0, (1 << width) - 1, 1, 1 << (width - 1)] + [rng.getrandbits(width) for _ in range(40)]
        indices = list(range(-3, width + 3)) + [-(1 << 31), (1 << 31) - 1, 1 << 30, -width] + \
            [rng.getrandbits(32) - (1 << 31) for _ in range(50)]
        for w in words:
            for i in indices:
                assert IO.run_gates(gates, width, w, i) == prim.evaluate(w, i & 0xFFFFFFFF), (width, w, i)
    assert len(IO.bit_at_gates(32)) == 147 and IO.price("BitAtx32_v1")["gates"] == 147
    for key in ("BitAtx3_v1", "BitAtx0_v1", "BitAtx32_v2", GEMM, "TokenFeed_v1{NP=4,FIRST=false}"):
        assert IO.price(key) is None
    with pytest.raises(ValueError):
        IO.bit_at_gates(48)


def test_a_block_reading_bit_at_is_priced_and_one_with_no_price_is_undecided(feed, monkeypatch):
    """`FEED`'s block counts `BitAtx32_v1` at its price, which the table names: the gate table's own, and where the gate
    table has none for it the explicit circuit's, the same count; with no price for it the block's gates are undecided
    (`gate-table`) and not guessed."""
    row = IO.session_row(feed, None, None)
    table, unpriced = BL.gate_table([BC.session(feed)[0]])
    assert unpriced == [] and table["BitAtx32_v1"] == 147 and (row["table"], row["priced"]) == (table, {})
    assert row["gates"] == BC.gates(BC.session(feed)[0], row["table"])[0]
    bare = {key: gates for key, gates in table.items() if key != "BitAtx32_v1"}
    monkeypatch.setattr(BL, "gate_table", lambda fns: (dict(bare), ["BitAtx32_v1"]))
    own = IO.session_row(feed, None, None)
    assert own["priced"]["BitAtx32_v1"]["gates"] == 147 and (own["table"], own["gates"]) == (table, row["gates"])
    monkeypatch.setattr(IO, "price", lambda key: None)
    with pytest.raises(IO.Undecided, match="gate-table: no gates for BitAtx32_v1"):
        IO.session_row(feed, None, None)


def test_the_verdicts_rule_is_the_count_rules_on_its_rows(small):
    """On the session Program of `small`'s record (`count_instance`), the count rule's reference counts the verdict's gates
    a block and m blocks, and `judge` gives its verdict on its rows at each (c, k, n) at and just under them: complying at
    them, and failing on the gates, the fresh bits or the count."""
    ref = IO._reference()
    row = IO.session_row(small, None, None)
    big = 10 ** 15
    got = ref.check_count_detail(IO.count_instance(small, row["table"], big, big, big))
    (key, g, x, m), = got["blocks"]
    assert (key, g, m) == (row["key"], row["gates"], row["slots"]) and x > 0
    for c, k, n in ((g, x, m), (g - 1, x, m), (g, x - 1, m), (g, x, m - 1), (g - 1, x - 1, m - 1), (big, big, big)):
        want = ref.check_count_detail(IO.count_instance(small, row["table"], c, k, n))
        assert IO.judge(got["blocks"], c, k, n) == want, (c, k, n)
    assert IO.judge([["a", 5, 5, 0], ["b", 1, 1, 3]], 1, 1, 3)["complies"]
    assert IO.judge([["a", 1, 1, 2], ["b", 1, 1, 3]], 1, 1, 4)["failure"] == [None, "count"]


def test_n_is_the_cards_sessions_over_the_lease_on_the_grid():
    """A 4,32,4 session at its caps is 35 positions' linears, 630 attention pairs and four LM-head rows, 36,241,670,144
    θ-MACs with attention and the head at half rate; at u = 1/4 over a 60-minute lease the H200's tensor clock serves
    24,570,799 of them, 25,165,824 on the grid, and 62,914,560 over 150 minutes. Nothing but those inputs goes in."""
    w = IO.session_work(IO.window_class("4,32,4"), MODEL)
    assert w == {"linears": 34_057_748_480, "attention": 41_287_680, "lm_head": 1_050_673_152, "theta_macs": 36_241_670_144}
    cap = IO.capacity(w["theta_macs"], 60, Fraction(1, 4))
    assert (cap["raw"], cap["n"], cap["span_s"], cap["rate"]["macs_per_s"]) == (24_570_799, 25_165_824, 3600, 989_429_760_000_000)
    assert IO.capacity(w["theta_macs"], 150, Fraction(1, 4))["n"] == 62_914_560
    assert IO.capacity(w["theta_macs"], 60, Fraction(1, 2))["n"] > cap["n"]
    for bad in ((0, 60, Fraction(1, 4)), (w["theta_macs"], 0, Fraction(1, 4)), (w["theta_macs"], 60, Fraction(0)),
                (w["theta_macs"], 60, Fraction(5, 4))):
        with pytest.raises(ValueError):
            IO.capacity(*bad)


def test_the_policy_is_the_sessions_gates_and_fresh_bits_with_the_weights_held_and_n(small, monkeypatch):
    """The policy's c and k are its session's gates and fresh bits with every weight parameter held (and nothing else:
    a scan state's row stays fresh), its n `capacity` of the class's session work, each input to n with its source, and
    the time it was written as given."""
    assert IO.weight_pins(["weights/a@4", "weights/a@4/table", "scan_state/s@1", "weightsx/b@2"]) == \
        {"weights/a@4": "00", "weights/a@4/table": "00"}
    monkeypatch.setattr(IO, "class_doc", lambda serve, cls, scheme=None: (small, ["scan_state/s@1"]))
    pol = IO.policy("2,2,2", MODEL, "rev", 60, Fraction(1, 4), "the declared share", BEFORE, "abc")
    row = IO.session_row(small, {}, {})
    work = IO.session_work(IO.window_class("2,2,2"), MODEL)
    assert (pol["c"], pol["k"], pol["n"]) == (row["gates"], row["fresh"], IO.capacity(work["theta_macs"], 60, Fraction(1, 4))["n"])
    assert pol["written_utc"] == BEFORE and pol["session"]["definition"] == row["definition"] and pol["tree"] == "abc"
    assert pol["class"] == {"slots": 2, "prompt": 2, "outputs": 2, "positions": 3} and pol["serve"]["model"] == MODEL
    i = pol["n_inputs"]
    assert i["utilization"] == [1, 4] and i["utilization_source"] == "the declared share" and i["work"]["theta_macs"] == work["theta_macs"]
    assert i["span_source"] and i["rate"]["source"] and i["work"]["source"]


def _stated(**over) -> dict:
    """What `prove_path.py serve-phi` states of a BF16 serve of `BF16` on an H200: its target with no int8 scope."""
    from verity_vllm import TargetProfile
    from verity_vllm.engine.served_seed import DEFAULT_SEED

    target = TargetProfile.default().with_overrides(compute_capability=(9, 0), num_sms=132).to_json()
    return json.loads(json.dumps({"model": BF16[0], "revision": BF16[1], "target": target, "execution": "bi-eager",
                                  "default_seed": DEFAULT_SEED, **over}))


def test_a_policy_on_a_stated_phi_is_that_phis_circuit_cut_where_it_says(small, monkeypatch, tmp_path):
    """With φ stated (``--serve``: serve-phi's statement of the serve.json a serve will write, or a serve.json, its
    execution under its engine), the policy's session is the class's circuit under that φ, cut at a PoUW scheme's
    linears where ``--pearl-tables`` names one, and the policy records both; a statement of another model or revision
    is refused. With none it is an H200 int8 serve's φ, uncut."""
    seen = []

    def doc(serve, cls, scheme=None):
        seen.append((serve, scheme))
        return small, []

    monkeypatch.setattr(IO, "class_doc", doc)
    args = ("2,2,2", *BF16, 60, Fraction(1, 4), "the declared share", BEFORE, "abc")
    pol = IO.policy(*args, {**_stated(), "clocks": {"time": STARTED}}, "pearl-c-h100-v1-h3")
    assert seen == [(_stated(), "pearl-c-h100-v1-h3")] and (pol["serve"], pol["pearl_tables"]) == (_stated(), "pearl-c-h100-v1-h3")
    engine = {k: v for k, v in _stated().items() if k != "execution"} | {"engine": {"execution": "bi-eager"}}
    assert IO.policy(*args, engine)["serve"] == _stated() and seen[-1] == (_stated(), None)
    pol = IO.policy("2,2,2", MODEL, "rev", 60, Fraction(1, 4), "the declared share", BEFORE, "abc")
    assert seen[-1] == (IO.serve_facts(MODEL, "rev"), None) and pol["pearl_tables"] is None
    for other in ({"model": MODEL}, {"revision": "rev"}):
        with pytest.raises(ValueError, match="--serve states φ of"):
            IO.policy(*args, _stated(**other))
    (phi := tmp_path / "serve-phi.json").write_text(json.dumps(_stated()))
    assert IO.main(["policy", "--class", "2,2,2", "--model", BF16[0], "--revision", BF16[1], "--lease-min", "60",
                    "--serve", str(phi), "--pearl-tables", "pearl-c-h100-v1-h3", "--out", str(tmp_path / "p.json")]) == 0
    pol = json.loads((tmp_path / "p.json").read_text())
    assert (pol["serve"], pol["pearl_tables"], seen[-1]) == (_stated(), "pearl-c-h100-v1-h3", (_stated(), "pearl-c-h100-v1-h3"))


def test_a_bf16_checkpoint_under_the_int8_serves_phi_is_refused_by_name(tmp_path, capsys):
    """The vocabulary can't express a BF16 checkpoint on the H200 int8 serve's target, `serve_facts`' φ: the policy is
    refused saying so, and none is written. A BF16 serve's policy states its own φ (``--serve``)."""
    out = tmp_path / "policy.json"
    assert IO.main(["policy", "--class", "4,32,4", "--model", BF16[0], "--revision", BF16[1], "--lease-min", "60",
                    "--out", str(out)]) == 1
    assert capsys.readouterr().out.startswith(
        "INFERENCE-ONLY POLICY refused: Undecided: the class's circuit under the policy's φ: vocabulary cannot express: "
        "a bf16 checkpoint on a target whose int8_exact is 'linears'")
    assert not out.exists()


def _serve(d: Path, windows: dict[int, list[dict]], unsealed: tuple[int, ...] = ()) -> Path:
    """A toy serve dir: each window's requests (``prompt``, ``outputs`` lengths, ``carried``, ``failed``), sealed, and the
    windows ``unsealed`` closed with no seal."""
    d.mkdir()
    (d / "serve.json").write_text(json.dumps({"clocks": {"time": STARTED}}))
    seals, lines, reqs = [], [], []
    for k, rs in windows.items():
        ids = [f"e{k}-{j}" for j in range(len(rs))]
        seals.append({"window": k, "requests": ids})
        for j, r in enumerate(rs):
            reqs.append({"engine_id": ids[j], "user": f"u{k}-{j}", "windows": [k, k + 1] if r.get("carried") else [k],
                         "check": {"passed": not r.get("failed")}, "prompt_ids": [7] * r.get("prompt", 2),
                         "streamed_ids": [9] * r.get("outputs", 2)})
    for k in (*windows, *unsealed):
        lines.append({"window": k, "passed": True, "commit": {"steps": 3}})
    for name, rows in (("seals.jsonl", seals), ("windows.jsonl", lines), ("requests.jsonl", reqs)):
        (d / name).write_text("".join(json.dumps(x) + "\n" for x in rows))
    return d


def _policy(row: dict, **over) -> dict:
    return {"format": IO.POLICY_FORMAT, "written_utc": BEFORE, "c": row["gates"], "k": row["fresh"], "n": 6,
            "class": {"slots": 2, "prompt": 2, "outputs": 2, "positions": 3}, "session": {"definition": row["definition"]},
            **over}


FULL = {7: [{}, {}], 11: [{}, {"prompt": 1}], 13: [{"outputs": 1}]}


def test_a_run_within_its_policy_is_accepted_and_its_public_form_holds_no_count(small, tmp_path):
    """Three sealed windows of the class (one with a slot empty) are six blocks of its session: within (c, k, n) at their
    own gates and fresh bits and n = 6 the run is accepted, n judged on that exact count. The public verdict is its own
    public form: the judgment, the class's session, gates and fresh bits a block, slots and positions, saying that n was
    judged on the exact count, and no count of blocks or of windows and no window's number; the private one holds the
    exact counts."""
    row = IO.session_row(small, None, None)
    pub, full = IO.verdict(_policy(row), small, _serve(tmp_path / "s", FULL), None, None)
    assert (pub["verdict"], pub["reason"], pub["complies"], pub["failure"]) == ("accept", None, True, None)
    assert pub["classes"] == [[row["definition"], row["gates"], row["fresh"]]] and not {"blocks", "total"} & set(pub)
    assert (pub["slots"], pub["positions"], pub["c"], pub["k"], pub["n"]) == (2, row["positions"], row["gates"],
                                                                              row["fresh"], 6)
    assert (pub["format"], pub["counts"]) == ("verity-bench/inference-only-verdict/v4", IO.COUNTS)
    assert IO.COUNTS.startswith("none published") and "exact count" in IO.COUNTS
    assert pub["registered_session"] is True and pub["serve_began_utc"] == "2025-11-01T12:26:40+00:00"
    assert SV.public_json(pub) == pub and not {"sealed", "outside_class", "retroactive"} & set(pub)
    assert (full["sealed"], full["blocks"], full["total"], full["outside_class"]) == (
        3, [[row["definition"], row["gates"], row["fresh"], 6]], 6, {})
    assert IO.line(pub).startswith("accept: session ") and IO.line(pub).endswith(
        "n 6 (no count published: n is judged on the blocks' exact count, held on the node)")


@pytest.mark.parametrize(("over", "why"), [
    ({"n": 5}, "the blocks' exact count is more than n = 5"),
    ({"c": "gates-1"}, "gates, more than c ="),
    ({"k": "fresh-1"}, "fresh bits, more than k ="),
])
def test_a_run_past_its_policy_is_refused(small, tmp_path, over, why):
    """Past n, c or k the run is refused, its failure the count rule's; past n its reason names n and not the count."""
    row = IO.session_row(small, None, None)
    at = {k: (row["gates"] - 1 if v == "gates-1" else row["fresh"] - 1 if v == "fresh-1" else v) for k, v in over.items()}
    pub, _ = IO.verdict(_policy(row, **at), small, _serve(tmp_path / "s", FULL), None, None)
    assert pub["verdict"] == "refuse" and why in pub["reason"] and pub["complies"] is False
    assert pub["failure"] == ([None, "count"] if "n" in over else [0, "gates" if "c" in over else "fresh"])
    assert not {"blocks", "total"} & set(pub) and ("n" not in over or pub["reason"] == why)


def test_no_public_field_carries_a_count_and_n_is_judged_on_the_exact_one(small, tmp_path):
    """Runs of one to 17 windows, two to 34 blocks, which a 16-block floor on the grid publishes as 16 and as 36, publish
    one public verdict and one line at n = 64, and differ only in their private verdicts' exact counts, which the count
    rule judges. At n = 1 one window is refused; at n = 3 one window's two blocks are accepted and two windows' four
    refused, and the two public verdicts differ in their judgment alone."""
    row = IO.session_row(small, None, None)
    runs = {w: _serve(tmp_path / f"w{w}", {7 + j: [{}, {}] for j in range(w)}) for w in (1, 2, 3, 8, 17)}
    seen = {w: IO.verdict(_policy(row, n=64), small, d, None, None) for w, d in runs.items()}
    pubs = [pub for pub, _ in seen.values()]
    assert all(p == pubs[0] for p in pubs) and pubs[0]["verdict"] == "accept" and len({IO.line(p) for p in pubs}) == 1
    assert [full["blocks"] for _, full in seen.values()] == [[[row["definition"], row["gates"], row["fresh"], 2 * w]]
                                                             for w in runs]
    assert [full["total"] for _, full in seen.values()] == [2, 4, 6, 16, 34]
    for pub, full in seen.values():
        got = IO.judge(full["blocks"], pub["c"], pub["k"], pub["n"])
        assert (pub["complies"], pub["failure"]) == (got["complies"], got["failure"])
    one, _ = IO.verdict(_policy(row, n=1), small, runs[1], None, None)
    assert (one["verdict"], one["reason"], one["failure"]) == (
        "refuse", "the blocks' exact count is more than n = 1", [None, "count"])
    one, _ = IO.verdict(_policy(row, n=3), small, runs[1], None, None)
    two, full = IO.verdict(_policy(row, n=3), small, runs[2], None, None)
    assert (one["verdict"], two["verdict"], two["reason"], full["total"]) == (
        "accept", "refuse", "the blocks' exact count is more than n = 3", 4)
    judgment = ("verdict", "reason", "complies", "failure")
    assert {k: v for k, v in one.items() if k not in judgment} == {k: v for k, v in two.items() if k not in judgment}


@pytest.mark.parametrize(("windows", "unsealed", "why"), [
    ({7: [{}, {"carried": True}]}, (), "a window the class doesn't take"),
    ({7: [{}, {"failed": True}]}, (), "a window the class doesn't take"),
    ({7: [{"prompt": 3}]}, (), "a window the class doesn't take"),
    ({7: [{"outputs": 3}]}, (), "a window the class doesn't take"),
    ({7: [{}, {}, {}]}, (), "a window the class doesn't take"),
    ({7: [{}]}, (8,), "a window the class doesn't take"),
    ({}, (), "the serve sealed nothing"),
])
def test_what_the_count_cant_decide_is_refused(small, tmp_path, windows, unsealed, why):
    """A window carrying a request across, one whose check failed, a request past the class's prompt or outputs, more
    requests than slots, a closed window with no seal, or a serve that sealed nothing: the run's blocks are undecided, and
    it is refused, the private verdict naming each window and why and the public one none."""
    row = IO.session_row(small, None, None)
    pub, full = IO.verdict(_policy(row), small, _serve(tmp_path / "s", windows, unsealed), None, None)
    assert pub["verdict"] == "refuse" and pub["reason"].startswith("undecided: ") and why in pub["reason"]
    assert pub["complies"] is False and not {"classes", "blocks", "total"} & set(pub)
    assert not any(ch.isdigit() for ch in pub["reason"])
    assert set(full["outside_class"]) == ({"8"} if unsealed else set() if not windows else {"7"})


def test_a_policy_written_after_the_serve_began_is_read_only_when_said_retroactive(small, tmp_path):
    """A policy written after the serve began, or with no time, is undecided; read retroactively, the verdict says why.
    A serve dir that records no start is undecided even read retroactively, and no policy, another format, or no circuit
    record is undecided too."""
    row = IO.session_row(small, None, None)
    d = _serve(tmp_path / "s", FULL)
    for written in (AFTER, "2025-11-01T12:26:40+00:00", None, "yesterday", "2025-11-01T12:26:39"):
        pub, _ = IO.verdict(_policy(row, written_utc=written), small, d, None, None)
        assert pub["verdict"] == "refuse" and "not before the serve began" in pub["reason"], written
    pub, _ = IO.verdict(_policy(row, written_utc=AFTER), small, d, None, None, retroactive="served before the step existed")
    assert (pub["verdict"], pub["retroactive"]) == ("accept", "served before the step existed")
    (unstarted := _serve(tmp_path / "t", FULL)).joinpath("serve.json").unlink()
    pub, _ = IO.verdict(_policy(row), small, unstarted, None, None, retroactive="served before the step existed")
    assert pub["verdict"] == "refuse" and "records no start" in pub["reason"] and "classes" not in pub
    for pol, doc, why in ((None, small, "no policy"), ({**_policy(row), "format": "x"}, small, "no policy"),
                          (_policy(row), None, "no circuit record"),
                          (_policy(row), {k: v for k, v in small.items() if k != "blocks"}, "states no inference blocks")):
        pub, _ = IO.verdict(pol, doc, d, None, None)
        assert pub["verdict"] == "refuse" and why in pub["reason"]


def test_the_command_line_writes_the_public_verdict_and_the_private_one_0600(small, tmp_path, capsys):
    """`verdict` writes the public verdict to --out and the full one, 0600, to --private, `nci-check --count --detail`'s
    whole verdict in the private one alone, and prints the public line."""
    row = IO.session_row(small, None, None)
    d = _serve(tmp_path / "s", FULL)
    (tmp_path / "policy.json").write_text(json.dumps(_policy(row)))
    (tmp_path / "circuit.json").write_text(json.dumps(small))
    (tmp_path / "registration.json").write_text(json.dumps({}))
    (d / "private").mkdir(mode=0o700)
    lean = _nci_check(tmp_path / "lean", NEW_USAGE, {"complies": False}, _whole(row))
    assert IO.main(["verdict", "--policy", str(tmp_path / "policy.json"), "--serve-run", str(d), "--circuit",
                    str(tmp_path / "circuit.json"), "--registration", str(tmp_path / "registration.json"), "--pins",
                    str(tmp_path / "no-weights.json"), "--lean", str(lean), "--out", str(tmp_path / "out" / "inference-only.json"),
                    "--private", str(d / "private" / "inference-only.json")]) == 0
    pub = json.loads((tmp_path / "out" / "inference-only.json").read_text())
    full = json.loads((d / "private" / "inference-only.json").read_text())
    assert pub["verdict"] == "accept" and full["sealed"] == 3 and not {"sealed", "blocks", "total"} & set(pub)
    assert (full["blocks"][0][3], full["total"], pub["format"]) == (6, 6, "verity-bench/inference-only-verdict/v4")
    assert stat.S_IMODE((d / "private" / "inference-only.json").stat().st_mode) == 0o600
    assert full["lean_detail"] == {"flags": ["--count", "--detail"], "out": _whole(row)} and "lean_detail" not in pub
    said = capsys.readouterr().out
    assert said.startswith("INFERENCE-ONLY accept: session ") and "(no count published: n is judged on the blocks' exact " \
        "count, held on the node)" in said and said.rstrip().endswith(f"{row['gates']} gates a block (agrees)")
    assert str(COUNT) not in (tmp_path / "out" / "inference-only.json").read_text() and str(COUNT) not in said


def test_lean_counts_refusal_is_said_not_judged(small, tmp_path):
    """An `nci-check` that exits non-zero or prints no JSON object gives no Lean count, which the verdict says beside its
    own; what it printed stays in the private verdict."""
    row = IO.session_row(small, None, None)
    d = _serve(tmp_path / "s", FULL)
    for body, why, said in (("echo no >&2; exit 2", "exit 2", "no"), ("echo hello", "printed no JSON object", "hello"),
                            ("echo '[7, 9]'", "printed no JSON object", "[7, 9]")):
        exe = tmp_path / "nci-check"
        exe.write_text(f"#!/bin/sh\n{body}\n")
        exe.chmod(0o755)
        pub, full = IO.verdict(_policy(row), small, d, None, None, lean=exe, scratch=tmp_path)
        assert pub["verdict"] == "accept" and pub["lean"] == {"error": f"nci-check --count {why}"}
        assert full["lean_detail"] == {"flags": ["--count"], "error": f"nci-check --count {why}", "said": said}
        assert list(tmp_path.glob("tmp*.json")) == []


#: `nci-check`'s usage while ``--count`` printed its whole verdict, and once it prints that only with ``--detail``
OLD_USAGE = "usage: nci-check [--unrolled] INSTANCE TRUSTED | nci-check --policy [--unrolled] INSTANCE | nci-check --count INSTANCE"
NEW_USAGE = OLD_USAGE.replace("--count INSTANCE", "--count [--detail] INSTANCE")
#: a count of blocks no other number in the verdict holds
COUNT = 987654321


def _nci_check(where: Path, usage: str, count: dict, detail: dict | None) -> Path:
    """A stand-in `nci-check` that answers as the built one does: ``--count INSTANCE`` prints ``count``, ``--count --detail
    INSTANCE`` prints ``detail`` where it has one, and anything else gets ``usage``, exit 2. It logs each call's
    arguments to ``args.log`` beside it."""
    where.mkdir(parents=True)
    (where / "count.json").write_text(json.dumps(count))
    if detail is not None:
        (where / "detail.json").write_text(json.dumps(detail))
    exe = where / "nci-check"
    exe.write_text(f"""#!/bin/sh
echo "$*" >> '{where}/args.log'
if [ "$1" = --count ] && [ $# -eq 2 ]; then cat '{where}/count.json'; exit 0; fi
if [ "$1 $2" = "--count --detail" ] && [ $# -eq 3 ] && [ -f '{where}/detail.json' ]; then cat '{where}/detail.json'; exit 0; fi
echo '{usage}' >&2
exit 2
""")
    exe.chmod(0o755)
    return exe


def _whole(row: dict) -> dict:
    """`NciCheck.Count`'s whole verdict on the session Program with θ empty: its row's fresh bits past k, and a count."""
    return {"blocks": [[row["key"], row["gates"], row["fresh"] + 5, COUNT]], "total": COUNT, "complies": False,
            "failure": [0, "fresh"]}


@pytest.mark.parametrize(("usage", "public", "flags"), [
    (OLD_USAGE, None, ["--count"]),
    (NEW_USAGE, {"complies": False}, ["--count", "--detail"]),
], ids=["count-prints-all", "detail-prints-all"])
def test_the_record_takes_nci_checks_gates_fresh_bits_and_verdict_and_none_of_its_counts(small, tmp_path, usage, public,
                                                                                         flags):
    """An `nci-check` whose ``--count`` prints its whole verdict, and one that prints only ``{complies}`` unless given
    ``--detail``, which its usage names: each is called for its whole verdict, rows, total and failure, which the private
    verdict keeps whole, and the public verdict and its line take only its gates a block, whether they agree, and with θ
    empty its fresh bits and ``complies``, and none of its counts."""
    row = IO.session_row(small, None, None)
    exe = _nci_check(tmp_path / "lean", usage, public or _whole(row), _whole(row) if public else None)
    pub, full = IO.verdict(_policy(row), small, _serve(tmp_path / "s", FULL), None, None, lean=exe, scratch=tmp_path)
    assert pub["verdict"] == "accept" and pub["lean"] == {
        "gates": row["gates"], "agrees_on_gates": True, "with_theta_empty": {"fresh": row["fresh"] + 5, "complies": False}}
    assert str(COUNT) not in json.dumps(pub) and str(COUNT) not in IO.line(pub) and "lean_detail" not in pub
    assert IO.line(pub).endswith(f"; nci-check --count: {row['gates']} gates a block (agrees)")
    assert full["lean_detail"] == {"flags": flags, "out": _whole(row)}
    calls = (tmp_path / "lean" / "args.log").read_text().splitlines()
    assert calls[0] == "" and len(calls) == 2 and calls[1].startswith(" ".join(flags) + " " + str(tmp_path))


def test_a_binary_whose_usage_hides_detail_gives_no_rows_which_is_said(small, tmp_path):
    """An `nci-check` that prints only ``{complies}`` for ``--count`` and whose usage names no ``--detail`` gives no row
    of the block: the verdict says so beside its own, and takes nothing from it."""
    row = IO.session_row(small, None, None)
    exe = _nci_check(tmp_path / "lean", OLD_USAGE, {"complies": False}, _whole(row))
    pub, full = IO.verdict(_policy(row), small, _serve(tmp_path / "s", FULL), None, None, lean=exe, scratch=tmp_path)
    assert pub["verdict"] == "accept" and pub["lean"] == {"error": "nci-check --count printed no row of the block"}
    assert full["lean_detail"] == {"flags": ["--count"], "out": {"complies": False}}


@pytest.mark.parametrize(("usage", "flags"), [(OLD_USAGE, ["--count"]), (NEW_USAGE, ["--count", "--detail"])],
                         ids=["count-prints-all", "detail-prints-all"])
def test_an_nci_check_refusal_lands_in_lean_with_its_code_and_message_alone(small, tmp_path, usage, flags):
    """`nci-check`'s refusal, with or without ``--detail``, lands in ``lean`` as its ``error`` and ``message``, the
    verdict accepting beside it; anything else it carries, a count here, stays in the private verdict."""
    row = IO.session_row(small, None, None)
    refusal = {"error": "program", "message": "E_REINTERPRET: a reference reads a leaf at another width"}
    said = {**refusal, "blocks": [[row["key"], row["gates"], row["fresh"], COUNT]], "total": COUNT}
    exe = _nci_check(tmp_path / "lean", usage, said, said)
    pub, full = IO.verdict(_policy(row), small, _serve(tmp_path / "s", FULL), None, None, lean=exe, scratch=tmp_path)
    assert pub["verdict"] == "accept" and pub["lean"] == refusal and IO.line(pub).endswith("; nci-check --count: program")
    assert str(COUNT) not in json.dumps(pub) and str(COUNT) not in IO.line(pub)
    assert full["lean_detail"] == {"flags": flags, "out": said}


@pytest.mark.slow
def test_nci_check_counts_the_verdicts_gates_on_the_session_program(small, tmp_path):
    """The compiled `nci-check --count` on `small`'s session Program counts the verdict's gates a block: the Lean the
    verdict runs beside its own count reads the instance it is given, as only the built executable can show (the
    reference test holds the instance to the Python count, and `nci-count-difftest` the Lean to the reference)."""
    import served_cases as SC

    exe = SC.build(SC.INFERENCE_ONLY_LEAN[0], "nci-check")
    row = IO.session_row(small, None, None)
    pub, full = IO.verdict(_policy(row), small, _serve(tmp_path / "s", FULL), None, None, lean=exe, scratch=tmp_path)
    assert pub["verdict"] == "accept" and pub["lean"]["agrees_on_gates"] is True and pub["lean"]["gates"] == row["gates"]
    assert set(pub["lean"]) == {"gates", "agrees_on_gates", "with_theta_empty"}
    assert set(pub["lean"]["with_theta_empty"]) == {"fresh", "complies"}
    assert {"blocks", "total", "failure"} <= set(full["lean_detail"]["out"])
