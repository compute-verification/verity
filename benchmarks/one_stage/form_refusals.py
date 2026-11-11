"""The fixture of `verify --zk`'s form refusals (`verity/protocols/spot/verifier/tests/test_form_refusals.py`): honest
M0 `--zk` sessions, each of N instances with every unit sampled and bound to its template query's partition, as
`forged_proofs.py` stages them, differing in the statement's form:

    honest     forged_proofs.py's A8 unit, flat (`Flock.ProvedScope.scopeOkH`)               verify --zk accepts
    templated  the same unit as its structural template (`verity/flock-circuit/tmpl`), its   verify --zk accepts
               rows not in the file (`ProvedScope.scopeOkT`)
    tail       `_tail`, its MUFU reads in lookup slots, its last stage returning the         refused for its form
               outputs (`ProvedScope.scopeOkI`)                                              (`Flock.KsScope`)

`verify` gives the tail's reason only once every earlier check, the proof's included, has passed. Lean's verdict on the
honest session under each of `WRONG_TAGS`, statements `verify --zk` must refuse, is recorded beside.

    PYTHONPATH=<tree>:<tree>/verity/protocols/flock/prover/python:<tree>/adapters/vllm:<tree>/tools/verity/circuit_check/src:\\
      <tree>/benchmarks/numerical/python python benchmarks/one_stage/form_refusals.py --bin flock-circuit --lean flock-verify \\
      --tmpl-unit flock-tmpl-unit --work DIR --fixture DIR

``--fixture`` gets per case ``<case>/stage/`` (circuit.txt, partition.json, program.json), ``<case>/tables/`` (the MUFU
tables its META's lookups name),
``<case>/pub-N.bin`` and ``<case>/session/``, and ``cases.json`` (each case's LIVE line, M0's verdict and Lean's at
generation, and the honest session's under each wrong statement). Never a prover's inst file; ``--work`` keeps those and
every log."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import a0  # noqa: E402
import forged_proofs as FP  # noqa: E402
from circuit_check import targets as TG  # noqa: E402
from verity.circuits.codec import canonical_json, encode_program  # noqa: E402
from verity.circuits.defs import CompositeDefinition, bind  # noqa: E402
from verity.definitions.ops.fp32 import F32AddFtz  # noqa: E402
from verity.definitions.ops.mufu import MufuEx2Ftz, MufuRcpFtz  # noqa: E402
from verity.circuits.types import Array, Value  # noqa: E402
from verity.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2  # noqa: E402
from verity_flock import circuit as FC  # noqa: E402
from verity_flock import class_statement as CS  # noqa: E402
from verity_flock import ir_lower as IL  # noqa: E402
from verity_flock import tmpl as TM  # noqa: E402
from verity.protocols.spot.verifier.one_stage import partition as P  # noqa: E402

#: statements the flat session's files don't fit, each refused for its own reason: typed, public outputs, plain leaves,
#: templated
WRONG_TAGS = ("verity/flock-circuit/types", "verity/flock-circuit@eb90718f", "verity/flock-circuit+plain-leaves",
              FC.TMPL_STATEMENT)
FORM = ": a statement form the soundness proof does not cover"
F32 = Value(32)


def _tail():
    """1 / (2^a + b) over f32 (`test_read_slots`' toy), with w, a row no node reads that spans SHA-512 blocks, as a
    composed circuit needs one."""
    def body(B, S, a, b, w):
        return B.call(MufuRcpFtz, B.call(F32AddFtz, B.call(MufuEx2Ftz, a), b))

    return bind(CompositeDefinition("FormRefusalsTail", 1, (), lambda S: ((("a", F32), ("b", F32), ("w", Array(64, F32))), F32),
                                    body, register=False))


def forms(rng) -> dict:
    """Each case's Definition, input rows, output rows, staged rows, stager options and statement."""
    x, w = FP.bf16(rng.normal(0, 1, (FP.N, FP.K))), FP.bf16(rng.normal(0, 0.02, (FP.N, FP.K)))
    dot = (TG.definition(FP.KIND), [("r0", V2, 32 * FP.K, ["x", "w"])], [(V2, (0,))],
           {"r0": np.concatenate([x, w], axis=1)})
    f32 = rng.normal(0, 2, (FP.N, 2)).astype(np.float32).view(np.uint32).astype(np.uint64)
    ab = np.stack([f32[:, 0] & 0xFFFF, f32[:, 0] >> 16, f32[:, 1] & 0xFFFF, f32[:, 1] >> 16], axis=1).astype(np.uint16)
    return {"honest": (*dot, {}, FC.STATEMENT),
            "templated": (*dot, {"layout": "iv/hybrid/4", "structural": True}, FC.TMPL_STATEMENT),
            "tail": (_tail(), [("ab", V2, 64, ["a", "b"]), ("w", V2, 2048, ["w"])], [(V2, (0,))],
                     {"ab": ab, "w": rng.integers(0, 1 << 16, (FP.N, 128), dtype=np.uint16)}, {"reads": "slots"},
                     FC.STATEMENT)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, help="flock-circuit, the proving build (sha512, glue)")
    ap.add_argument("--lean", required=True, help="flock-verify")
    ap.add_argument("--tmpl-unit", required=True, help="flock-tmpl-unit, the prover's placement of a templated unit")
    ap.add_argument("--work", required=True)
    ap.add_argument("--fixture", required=True)
    a = ap.parse_args()
    work, fx = Path(a.work), Path(a.fixture)
    TG.load_registries()
    salt_key = hashlib.sha256(b"verity/flock/form-refusals/salts").digest()
    cases: dict[str, dict] = {}
    for name, (fn, in_rows, out_rows, rows, kw, statement) in forms(np.random.default_rng(20261019)).items():
        program = P.mixed_population_program([(fn, FP.N)])
        part = P.partition(program, P.template_instances_query([fn]))
        d = work / name
        stage = d / "stage"
        stage.mkdir(parents=True, exist_ok=True)
        (stage / "partition.json").write_text(json.dumps(part, sort_keys=True))
        (stage / "program.json").write_bytes(canonical_json(encode_program(program)))
        st = CS.UnitStager(fn, f"unit/form-refusals-{name}", in_rows, out_rows, circuit=stage / "circuit.txt",
                           bind={"partition": P.digest(part), "program": part["program"]}, **kw)
        FP.stage_case(st, rows, (), d, salt_key, f"{st.iset.name}-{name}")
        (d / "tables").mkdir(exist_ok=True)
        if st.meta.get("lookups"):
            IL.write_tables(d / "tables")
            for f in (d / "tables").iterdir():
                if f.stem not in st.meta["lookups"]:
                    f.unlink()
        (d / "circuit.txt").symlink_to("stage/circuit.txt")
        sd = {"dir": d, "pin": st.pin, "partition_file": stage / "partition.json", "program_file": stage / "program.json",
              "statement": statement}
        if "layout" in kw:
            (d / "unit.tmpl").write_text(TM.write(st.low.template))
            subprocess.run([a.tmpl_unit, str(d / "unit.tmpl"), "--layout", st.meta["layout"], "--out", str(d / "unit.placed")],
                           check=True, capture_output=True)
            sd["tmpl_unit"] = d / "unit.placed"
        record, live = a0.session(a.bin, sd, FP.N, f"subset:{FP.N}", FP.free_port(), d / "run", zk=True)
        lean = a0.lean_verify(a.lean, sd, FP.N, d / "run", zk=True)
        cases[name] = {"definition": getattr(fn, "name", str(fn)), "statement": statement, "live": live, "m0": (record or {}).get("verdict"),
                       "lean": lean, "draw": (record or {}).get("unit_draw")}
        if name == "honest":
            cases[name]["wrong_tags"] = {t: a0.lean_verify(a.lean, {**sd, "statement": t}, FP.N, d / "run", zk=True,
                                                           log=f"lean-verify-{t.replace('/', '_')}") for t in WRONG_TAGS}
        print(f"form-refusals {name}: M0 {(cases[name]['m0'] or {}).get('accepted')}, Lean {lean.get('accepted')} "
              f"{lean.get('why')}", flush=True)
        recs = sorted((d / "run" / "sessions").glob("*/session.json"))
        if recs:
            shutil.copytree(recs[-1].parent, fx / name / "session")
        shutil.copytree(stage, fx / name / "stage")
        shutil.copytree(d / "tables", fx / name / "tables")
        shutil.copy(d / f"pub-{FP.N}.bin", fx / name / f"pub-{FP.N}.bin")
    (fx / "cases.json").write_text(json.dumps({"n": FP.N, "draw_law": f"subset:{FP.N}", "cases": cases}, indent=1,
                                              sort_keys=True, default=str))
    honest = cases["honest"]
    ok = (honest["lean"]["accepted"] and not any(v["accepted"] for v in honest["wrong_tags"].values())
          and cases["templated"]["lean"]["accepted"]
          and not cases["tail"]["lean"]["accepted"] and str(cases["tail"]["lean"]["why"]).endswith(FORM))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
