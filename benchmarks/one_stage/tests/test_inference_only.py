"""`inference_only.py`: a served run's inference-only verdict, its declared circuit's blocks against the (c, k, n) it
registered before it served, by `NciCheck.Count`'s rule. `BitAtx32_v1`'s price is an explicit circuit that computes the
primitive, input by input; the verdict's rule is the count rule's Python reference (`nci_reference.check_count`) on its
rows, and the Lean instance it makes is one on which that reference counts the verdict's gates; n is the H200's sessions
at the declared utilization over the lease, rounded up to the grid, from the class's fixed load alone; the policy is its
session's gates and fresh bits with every weight held, and n; and on toy serve dirs over small circuit records the verdict
accepts within (c, k, n), refuses past each, and refuses as undecided what the rule can't decide (no policy, one written
after the serve began unless said retroactive, a primitive with no price, a closed window with no seal, a window the
class doesn't take), its public form naming no window and counting none. The slow test runs `nci-check --count`."""
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
from verity.ml.definitions.registry.sampling import bit_at  # noqa: E402
from verity.ml.windows import interface as IF  # noqa: E402
from verity_vllm.program.predict import block_counts as BC  # noqa: E402

GEMM = 'Gemm_v2{K=64,N=8,DOT={"fn":"HopperBF16WgmmaDot16_v1"}}'
ROPE = "RoPEHead_v1{D=16}"
FEED = "TokenFeed_v1{NP=4,FIRST=false}"
MODEL = "verity-int8/Llama-3.2-1B-INT8"
#: the toy serve's start (its serve.json's clocks' time), and a policy written before and after it
STARTED = 1_762_000_000.0
BEFORE, AFTER = "2025-11-01T12:26:39+00:00", "2025-11-01T12:26:41+00:00"


@pytest.fixture(scope="module")
def small() -> dict:
    """`test_block_counts`' record: two slots and three steps, four `GEMM` units (a slot's two in one entry) and six
    `ROPE` units (a slot's three across the steps)."""
    entries = (IF.InferenceEntry(GEMM, 0, (2, 2), ("slot", "-"), 0), IF.InferenceEntry(ROPE, 0, (3, 2), ("step", "slot"), 0))
    circuit = IF.Circuit("test/blocks", ((BC.resolve(GEMM), 4), (BC.resolve(ROPE), 6)),
                         blocks=IF.Blocks(slots=2, steps=3, inference=entries))
    return IF.descriptor_record(circuit, 1)


@pytest.fixture(scope="module")
def feed() -> dict:
    """Two slots of `FEED`, which reads `BitAtx32_v1` (no Boolean version), one step."""
    entries = (IF.InferenceEntry(FEED, 0, (2,), ("slot",), 0),)
    return IF.descriptor_record(IF.Circuit("test/feed", ((BC.resolve(FEED), 2),),
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
    """`FEED`'s block counts `BitAtx32_v1` at its price, which the table names; with no price for it the block's gates are
    undecided (`gate-table`) and not guessed."""
    row = IO.session_row(feed, None, None)
    table, unpriced = BC.gate_table(BC.primitives([BC.session(feed)[0]]))
    assert unpriced == ["BitAtx32_v1"] and row["table"] == {**table, "BitAtx32_v1": 147}
    assert row["priced"]["BitAtx32_v1"]["gates"] == 147 and row["gates"] == BC.gates(BC.session(feed)[0], row["table"])[0]
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
    got = ref.check_count(IO.count_instance(small, row["table"], big, big, big))
    (key, g, x, m), = got["blocks"]
    assert (key, g, m) == (row["key"], row["gates"], row["slots"]) and x > 0
    for c, k, n in ((g, x, m), (g - 1, x, m), (g, x - 1, m), (g, x, m - 1), (g - 1, x - 1, m - 1), (big, big, big)):
        want = ref.check_count(IO.count_instance(small, row["table"], c, k, n))
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
    monkeypatch.setattr(IO, "class_doc", lambda serve, cls: (small, ["scan_state/s@1"]))
    pol = IO.policy("2,2,2", MODEL, "rev", 60, Fraction(1, 4), "the declared share", BEFORE, "abc")
    row = IO.session_row(small, {}, {})
    work = IO.session_work(IO.window_class("2,2,2"), MODEL)
    assert (pol["c"], pol["k"], pol["n"]) == (row["gates"], row["fresh"], IO.capacity(work["theta_macs"], 60, Fraction(1, 4))["n"])
    assert pol["written_utc"] == BEFORE and pol["session"]["definition"] == row["definition"] and pol["tree"] == "abc"
    assert pol["class"] == {"slots": 2, "prompt": 2, "outputs": 2, "positions": 3} and pol["serve"]["model"] == MODEL
    i = pol["n_inputs"]
    assert i["utilization"] == [1, 4] and i["utilization_source"] == "the declared share" and i["work"]["theta_macs"] == work["theta_macs"]
    assert i["span_source"] and i["rate"]["source"] and i["work"]["source"]


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


def test_a_run_within_its_policy_is_accepted_and_its_public_form_counts_no_window(small, tmp_path):
    """Three sealed windows of the class (one with a slot empty) are six blocks of its session: within (c, k, n) at their
    own gates and fresh bits and n = 6 the run is accepted. The public verdict is its own public form, holds no window's
    number or count of them, and the private one holds the exact count."""
    row = IO.session_row(small, None, None)
    pub, full = IO.verdict(_policy(row), small, _serve(tmp_path / "s", FULL), None, None)
    assert (pub["verdict"], pub["reason"], pub["complies"], pub["failure"]) == ("accept", None, True, None)
    assert pub["blocks"] == [[row["definition"], row["gates"], row["fresh"], 6]] and pub["total"] == 6
    assert pub["registered_session"] is True and pub["serve_began_utc"] == "2025-11-01T12:26:40+00:00"
    assert SV.public_json(pub) == pub and not {"sealed", "outside_class", "retroactive"} & set(pub)
    assert (full["sealed"], full["blocks"][0][3], full["outside_class"]) == (3, 6, {})
    assert IO.line(pub).startswith("accept: 6 blocks (padded) of session")


@pytest.mark.parametrize(("over", "why"), [
    ({"n": 5}, "6 blocks (padded), more than n = 5"),
    ({"c": "gates-1"}, "gates, more than c ="),
    ({"k": "fresh-1"}, "fresh bits, more than k ="),
])
def test_a_run_past_its_policy_is_refused(small, tmp_path, over, why):
    """Past n, c or k the run is refused, its failure the count rule's."""
    row = IO.session_row(small, None, None)
    at = {k: (row["gates"] - 1 if v == "gates-1" else row["fresh"] - 1 if v == "fresh-1" else v) for k, v in over.items()}
    pub, _ = IO.verdict(_policy(row, **at), small, _serve(tmp_path / "s", FULL), None, None)
    assert pub["verdict"] == "refuse" and why in pub["reason"] and pub["complies"] is False
    assert pub["failure"] == ([None, "count"] if "n" in over else [0, "gates" if "c" in over else "fresh"])


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
    assert pub["complies"] is False and "blocks" not in pub and not any(ch.isdigit() for ch in pub["reason"])
    assert set(full["outside_class"]) == ({"8"} if unsealed else set() if not windows else {"7"})


def test_a_policy_written_after_the_serve_began_is_read_only_when_said_retroactive(small, tmp_path):
    """A policy written after the serve began, or with no time, is undecided; read retroactively, the verdict says why.
    No policy, another format, or no circuit record is undecided too."""
    row = IO.session_row(small, None, None)
    d = _serve(tmp_path / "s", FULL)
    for written in (AFTER, "2025-11-01T12:26:40+00:00", None, "yesterday", "2025-11-01T12:26:39"):
        pub, _ = IO.verdict(_policy(row, written_utc=written), small, d, None, None)
        assert pub["verdict"] == "refuse" and "not before the serve began" in pub["reason"], written
    pub, _ = IO.verdict(_policy(row, written_utc=AFTER), small, d, None, None, retroactive="served before the step existed")
    assert (pub["verdict"], pub["retroactive"]) == ("accept", "served before the step existed")
    for pol, doc, why in ((None, small, "no policy"), ({**_policy(row), "format": "x"}, small, "no policy"),
                          (_policy(row), None, "no circuit record"),
                          (_policy(row), {k: v for k, v in small.items() if k != "blocks"}, "states no inference blocks")):
        pub, _ = IO.verdict(pol, doc, d, None, None)
        assert pub["verdict"] == "refuse" and why in pub["reason"]


def test_the_command_line_writes_the_public_verdict_and_the_private_one_0600(small, tmp_path, capsys):
    """`verdict` writes the public verdict to --out and the full one, 0600, to --private, and prints the public line."""
    row = IO.session_row(small, None, None)
    d = _serve(tmp_path / "s", FULL)
    (tmp_path / "policy.json").write_text(json.dumps(_policy(row)))
    (tmp_path / "circuit.json").write_text(json.dumps(small))
    (tmp_path / "registration.json").write_text(json.dumps({}))
    (d / "private").mkdir(mode=0o700)
    assert IO.main(["verdict", "--policy", str(tmp_path / "policy.json"), "--serve-run", str(d), "--circuit",
                    str(tmp_path / "circuit.json"), "--registration", str(tmp_path / "registration.json"), "--pins",
                    str(tmp_path / "no-weights.json"), "--out", str(tmp_path / "out" / "inference-only.json"),
                    "--private", str(d / "private" / "inference-only.json")]) == 0
    pub = json.loads((tmp_path / "out" / "inference-only.json").read_text())
    full = json.loads((d / "private" / "inference-only.json").read_text())
    assert pub["verdict"] == "accept" and full["sealed"] == 3 and "sealed" not in pub
    assert stat.S_IMODE((d / "private" / "inference-only.json").stat().st_mode) == 0o600
    assert capsys.readouterr().out.startswith("INFERENCE-ONLY accept: 6 blocks (padded)")


def test_lean_counts_refusal_is_said_not_judged(small, tmp_path):
    """An `nci-check` that exits non-zero or prints no JSON gives no Lean count, which the verdict says beside its own."""
    row = IO.session_row(small, None, None)
    d = _serve(tmp_path / "s", FULL)
    for body, said in (("echo no >&2; exit 2", "exit 2: no"), ("echo hello", "printed no JSON: hello")):
        exe = tmp_path / "nci-check"
        exe.write_text(f"#!/bin/sh\n{body}\n")
        exe.chmod(0o755)
        pub, _ = IO.verdict(_policy(row), small, d, None, None, lean=exe, scratch=tmp_path)
        assert pub["verdict"] == "accept" and said in pub["lean"]["error"]
        assert list(tmp_path.glob("tmp*.json")) == []


@pytest.mark.slow
def test_nci_check_counts_the_verdicts_gates_on_the_session_program(small, tmp_path):
    """The compiled `nci-check --count` on `small`'s session Program counts the verdict's gates a block: the Lean the
    verdict runs beside its own count reads the instance it is given, as only the built executable can show (the
    reference test holds the instance to the Python count, and `nci-count-difftest` the Lean to the reference)."""
    import served_cases as SC

    exe = SC.build(SC.INFERENCE_ONLY_LEAN[0], "nci-check")
    row = IO.session_row(small, None, None)
    pub, _ = IO.verdict(_policy(row), small, _serve(tmp_path / "s", FULL), None, None, lean=exe, scratch=tmp_path)
    assert pub["verdict"] == "accept" and pub["lean"]["agrees_on_gates"] is True and pub["lean"]["gates"] == row["gates"]
    assert set(pub["lean"]["with_theta_empty"]) == {"fresh", "complies", "failure"}
