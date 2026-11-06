"""Independent tests: the v1 descriptor codec composed with the Hopper binding and the TP2 sharded program.

None of the three was validated in this combination (the codec was measured on B0/B1 `Serve`; hopper.py and b1_tp2.py
were validated by correspondence, not through the codec). Everything here is at toy scale (`TINY`: H=32, NL=2,
LP=3, STEPS=2) -- the B1-scale descriptors are never built locally.

What is checked, per program (`Serve{TINY}` as the control, `ServeHopper{TINY}`, `ServeTP2{TINY_TP2}`):

1. encode (v1) -> canonical bytes -> json -> decode -> `descriptor_equivalence.compare(original, decoded)` is
   `equivalent`, and the same through v0;
2. the v1 digest is a function of the program: re-encoding the decoded program gives the same digest
   (the codec tests' `test_v1_kv_cache_arguments_alias_...` claim, now on the ported/sharded definition sets);
3. the equivalence is discriminating across bindings: `ServeHopper{TINY}` vs `Serve{TINY}` is a `Mismatch`
   (definition sets differ), and so is `ServeTP2` vs `Serve`;
4. a *fresh* process that imports only `verity.ml.circuits` (not `verity.ml.definitions.registry.hopper`) can or cannot decode the Hopper
   descriptor -- the descriptor carries every composite body but only *names* its primitives, and the CLI
   `verity-vllm descriptor-equivalence` loads with the process's `REGISTRY` (`_load`).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from verity.ml.circuits.codec import SCHEMA_V0, canonical_json, decode_program, encode_program, program_digest
from verity.ml.circuits.defs import REGISTRY, bind
from verity_vllm.program.descriptor_equivalence import Mismatch, compare
from verity.ml.circuits.program import Program
from verity.ml.definitions.registry import b1, b1_tp2, hopper

TINY = {"model": "tiny", "revision": None, "H": 32, "NL": 2, "NH": 2, "KVH": 1, "D": 16, "I": 32, "V": 64,
        "QKV": 64, "GU": 64, "EPS": 1e-5, "ROPE_THETA": 1e4, "MAX_POS": 8, "BIAS": True, "tie_word_embeddings": True, "profile": {}}
# TP=2 needs every sharded axis even: 2 q heads, 2 kv heads, V/2, I/2 (QKV = H + 2*KVH*D = 32 + 64 = 96)
TINY_TP2 = {**TINY, "KVH": 2, "QKV": 96, "TP": 2}

PROGRAMS = {
    "Serve{TINY}": lambda: bind(b1.Serve, C=TINY, LP=3, STEPS=2),
    "ServeHopper{TINY}": lambda: bind(hopper.ServeHopper, C=TINY, LP=3, STEPS=2),
    "ServeTP2{TINY_TP2}": lambda: b1_tp2.serve_tp2(3, 2, TINY_TP2),
}


def _roundtrip(P, schema):
    desc = encode_program(P, schema=schema) if schema else encode_program(P)
    raw = canonical_json({k: v for k, v in desc.items() if k != "annotations"})
    desc2 = json.loads(raw)
    D = decode_program(desc2, REGISTRY)
    return desc, raw, D


@pytest.mark.parametrize("label", list(PROGRAMS))
@pytest.mark.parametrize("schema", [None, SCHEMA_V0], ids=["v1", "v0"])
@pytest.mark.slow
def test_codec_round_trip_is_equivalent(label, schema):
    P = Program(PROGRAMS[label]())
    desc, raw, D = _roundtrip(P, schema)
    assert D.gates == P.gates
    rep = compare(P, D, gate_samples=500, call_samples=12, eval_samples=2)
    assert rep["verdict"] == "equivalent", rep
    assert rep["definitions"] > 0 and rep["gates"] == P.gates


@pytest.mark.parametrize("label", list(PROGRAMS))
def test_v1_digest_is_a_function_of_the_program(label):
    P = Program(PROGRAMS[label]())
    desc, raw, D = _roundtrip(P, None)
    again = encode_program(D)
    assert program_digest(again) == program_digest(desc), label
    # and the v0 digest is stable across the same trip
    desc0, _, D0 = _roundtrip(P, SCHEMA_V0)
    assert program_digest(encode_program(D0, schema=SCHEMA_V0)) == program_digest(desc0), label


@pytest.mark.parametrize("a,b", [("ServeHopper{TINY}", "Serve{TINY}"), ("ServeTP2{TINY_TP2}", "Serve{TINY}")])
def test_equivalence_discriminates_bindings(a, b):
    A, B = Program(PROGRAMS[a]()), Program(PROGRAMS[b]())
    with pytest.raises(Mismatch):
        compare(A, B, gate_samples=100, call_samples=4, eval_samples=0)


def _fresh_decode(path: Path, imports: str) -> subprocess.CompletedProcess:
    code = (f"import json, sys; {imports}\n"
            "from verity.ml.circuits.codec import decode_program\nfrom verity.ml.circuits.defs import REGISTRY\n"
            f"d = decode_program(json.load(open({str(path)!r})), REGISTRY); print('gates', d.gates)")
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          cwd=str(Path(__file__).resolve().parents[2]))


def test_hopper_descriptor_decodes_in_a_fresh_process_only_if_the_binding_module_is_imported(tmp_path):
    """The descriptor names `HopperBF16WgmmaDot16` / `Fa3InvSum` as primitives; they exist in `REGISTRY` only after
    `verity.ml.definitions.registry.hopper` is imported. `verity.ml.circuits` itself imports neither `registry.b1` nor `registry.hopper`,
    so `verity-vllm descriptor-equivalence --old <hopper descriptor>` depends on what the process imported."""
    P = Program(PROGRAMS["ServeHopper{TINY}"]())
    desc = encode_program(P)
    path = tmp_path / "hopper-tiny.json"
    path.write_text(canonical_json({k: v for k, v in desc.items() if k != "annotations"}).decode())
    bare = _fresh_decode(path, "import verity.ml.circuits")
    with_b1 = _fresh_decode(path, "import verity.ml.circuits, verity.ml.definitions.registry.b1")
    with_hopper = _fresh_decode(path, "import verity.ml.circuits, verity.ml.definitions.registry.hopper")
    assert with_hopper.returncode == 0, with_hopper.stderr[-800:]
    # Record the facts for the report; the finding (if any) is the CLI's dependence on the importer.
    report = {"bare": (bare.returncode, bare.stderr.strip().splitlines()[-1] if bare.stderr.strip() else ""),
              "with_b1": (with_b1.returncode, with_b1.stderr.strip().splitlines()[-1] if with_b1.stderr.strip() else ""),
              "with_hopper": (with_hopper.returncode, with_hopper.stdout.strip())}
    print(json.dumps(report))
    if bare.returncode != 0 or with_b1.returncode != 0:
        pytest.xfail(f"a fresh process without `verity.ml.definitions.registry.hopper` cannot decode the Hopper descriptor: {report}")
