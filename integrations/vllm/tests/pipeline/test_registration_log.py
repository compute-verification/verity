"""`registration_log.write` is a log nci-check's full check accepts: on a Program whose θ is registered, its record, provenance
and trusted input pass `nci_reference.check_baseline` (`NciCheck.Baseline`'s reference), each entry is the one one-stage's
`Registered.commit` makes of the parameter's rows, and each break is refused with the code the check names."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from verity.core.protocols.sampled_proofs.one_stage import registered as G
from verity.ml.circuits import codec
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.types import Array, Record, Value
from verity.ml.commitments import hm96
from verity_vllm.pipeline import registration_log as RL

_spec = importlib.util.spec_from_file_location(
    "nci_reference", Path(__file__).resolve().parents[4] / "verity/core/protocols/network/nci/lean/scripts/nci_reference.py")
REF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(REF)

N = hm96.SHA512.salt_bytes
OP = PrimitiveDefinition("RegLogOp", 1, (("a", Value(16)), ("b", Value(8)), ("c", Value(32))), Value(16), lambda *_a: 0,
                         register=False)
PARAMS = (("prompt", Array(4, Value(16))), ("w8", Array(6, Value(8))), ("sc", Array(3, Value(32))))
FN = bind(CompositeDefinition("RegLogProgram", 1, (), lambda S: (PARAMS, Value(16)),
                              lambda B, S, prompt, w8, sc: B.call(OP, prompt[0], w8[5], sc[2]), register=False))
VALUES = {"w8": np.arange(6, dtype=np.uint8) * 41, "sc": np.array([7, 1 << 31, 0xFFFFFFFF], dtype=np.uint32)}
WINDOW = {"kind": "stand-in", "deployment": "test"}


def _case(registered=("w8", "sc"), program_digest=None):
    """The Program (θ = w8 and sc), the instance on it (the one call one IU, α = 16 bits a gate) and the log `write` makes."""
    program = Program(FN, registered=registered, theta=("w8", "sc"))
    desc = codec.encode_program(program)
    text = codec.canonical_json(desc).decode()
    theta = desc["annotations"]["theta"]
    out = RL.write(program_digest or codec.program_digest(desc), FN.signature()[0], theta, VALUES.__getitem__, WINDOW)
    inp, _reg, _spans = REF.root_nodes(text)
    inst = {"program": text, "theta": theta, "owner": ["input" if i else 0 for i in inp], "gamma": program.gates,
            "alpha": [16, 1], "provenance": out["provenance"]}
    return program, inst, out


def test_the_log_passes_the_full_check_and_theta_carries_the_weights_reads():
    program, inst, out = _case()
    assert REF.check_baseline(inst, out["trusted"]) == {"gates": program.gates, "ius": [[1, 16]], "complies": True,
                                                        "failure": None}
    # read as novel, the weights' 8 + 32 bits break α: θ is what the log makes complying
    assert REF.check_desc(inst | {"theta": [], "provenance": []})["failure"] == [0, "nci"]


def test_one_entry_per_theta_parameter_as_registered_commit_makes_it():
    program, inst, out = _case()
    rec = out["record"]
    assert rec["previous"] is None and rec["program"] == codec.program_digest(json.loads(inst["program"]))
    assert out["digest"] == REF.record_digest(rec) and out["trusted"]["log"] == [rec] and out["trusted"]["run"] == out["digest"]
    assert [p["value"] for p in out["provenance"]] == [e["value"] for e in rec["registered"]] == ["w8", "sc"]
    for e in rec["registered"]:
        name, bits, salts = e["value"], G.width(program, e["value"]), out["salts"][e["value"]]
        assert G.shape(program, name, e["words"]) == e["leaves"] and e.get("word_bits", 16) == bits
        rows = VALUES[name].reshape(-1, e["words"]).tolist()
        assert G.Registered.commit(rec["program"], name, rows, [salts[i * N:(i + 1) * N] for i in range(e["leaves"])],
                                   word_bits=bits).entry() == e
    assert _case()[2]["digest"] != out["digest"]                 # hiding: fresh salts, other roots


def test_each_break_is_refused_with_its_code():
    _p, inst, out = _case()
    bad = copy.deepcopy(inst)
    bad["provenance"][0]["root"] = "00" * 64
    assert REF.check_baseline(bad, out["trusted"]) == {"error": "theta-root"}
    _p, inst, out = _case(program_digest="cd" * 64)             # a log registered for another Program
    assert REF.check_baseline(inst, out["trusted"]) == {"error": "theta-unbound"}
    _p, inst, out = _case(registered=())                         # θ bound per run: the Program the builder made before
    assert REF.check_baseline(inst, out["trusted"]) == {"error": "theta-not-registered"}


def test_record_takes_entries_committed_elsewhere_and_refuses_any_that_is_not_its_parameters():
    _p, inst, out = _case()
    args = (out["record"]["program"], FN.signature()[0], inst["theta"])
    entries = {e["value"]: e for e in out["record"]["registered"]}
    assert RL.record(*args, entries, WINDOW) == {k: v for k, v in out.items() if k != "salts"}
    for bad, why in [({"w8": entries["w8"]}, r"θ parameters \['sc'\] have none"),
                     (entries | {"x": entries["w8"]}, r"entries for \['x'\] are no θ parameter's"),
                     (entries | {"w8": entries["w8"] | {"value": "sc"}}, "'w8': its entry registers 'sc'"),
                     (entries | {"w8": entries["w8"] | {"leaves": entries["w8"]["leaves"] + 1}}, "'w8': its entry"),
                     (entries | {"sc": entries["sc"] | {"word_bits": 16}}, "16-bit leaves, its type is 3 of 32")]:
        with pytest.raises(ValueError, match=why):
            RL.record(*args, bad, WINDOW)


def test_refusals():
    mixed = (("m", Record(("a", Value(8)), ("b", Value(32)))),)
    assert RL.lower(mixed[0][1]) == REF.lower(mixed[0][1]) == [(8, 1), (32, 1)]
    with pytest.raises(ValueError, match="more than one leaf width"):
        RL.theta_parameters(mixed, [0])
    with pytest.raises(ValueError, match="no input node"):
        RL.theta_parameters(PARAMS, [9])
    with pytest.raises(ValueError, match="5 leaves given"):
        RL.write("ab" * 64, PARAMS, [1], {"w8": np.arange(5, dtype=np.uint8)}.__getitem__, WINDOW)
