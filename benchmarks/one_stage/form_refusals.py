"""The fixture of `verify --zk`'s form refusals (`verity/core/protocols/sampled_proofs/tests/test_form_refusals.py`): honest
M0 `--zk` sessions of `forged_proofs.py`'s A8 unit, staged as it stages them (N instances, every unit sampled, bound to its
template query's partition), differing only in the circuit's form:

    honest      x then w in one input row: the unit's form (`Flock.ProvedScope.scopeOkH`)       verify --zk accepts
    two-groups  x and w in two input rows, so the unit reads two input groups                  refused for its form

`ProvedScope.check` admits both (`scopeOkI`); only `Flock.KsScope.filesWhy` refuses the second, and `verify` gives a form's
reason only once every other check, the proof's included, has passed. Lean's verdict on the honest session under each of
`WRONG_TAGS`, statements `verify --zk` must refuse, is recorded beside.

    PYTHONPATH=<tree>:<tree>/verity/ml/flock/python:<tree>/integrations/vllm:<tree>/tools/verity/circuit_check/src:\\
      <tree>/benchmarks/numerical/python python benchmarks/one_stage/form_refusals.py --bin flock-circuit --lean flock-verify \\
      --work DIR --fixture DIR

``--fixture`` gets per case ``<case>/stage/`` (circuit.txt, partition.json, program.json), ``<case>/pub-N.bin`` and
``<case>/session/``, and ``cases.json`` (each case's LIVE line, M0's verdict and Lean's at generation, and the honest session's
under each wrong statement). Never a prover's inst file; ``--work`` keeps those and every log."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import a0  # noqa: E402
import forged_proofs as FP  # noqa: E402
from circuit_check import targets as TG  # noqa: E402
from verity.ml.circuits.codec import canonical_json, encode_program  # noqa: E402
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2  # noqa: E402
from verity_flock import class_statement as CS  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import partition as P  # noqa: E402

#: statements outside the form, each refused for another reason: typed, public outputs, plain leaves, templated
WRONG_TAGS = ("verity/flock-circuit/types", "verity/flock-circuit@eb90718f", "verity/flock-circuit+plain-leaves",
              "verity/flock-circuit/tmpl")
SCOPE_WHY = ("refused: a circuit outside the unit's form under --zk (scopeOkH: the unit reads its one input group and returns "
             "the outputs): a statement form the soundness proof does not cover")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, help="flock-circuit, the proving build (sha512, glue)")
    ap.add_argument("--lean", required=True, help="flock-verify")
    ap.add_argument("--work", required=True)
    ap.add_argument("--fixture", required=True)
    a = ap.parse_args()
    work, fx = Path(a.work), Path(a.fixture)
    TG.load_registries()
    fn = TG.definition(FP.KIND)
    program = P.mixed_population_program([(fn, FP.N)])
    part = P.partition(program, P.template_instances_query([fn]))
    rng = np.random.default_rng(20261019)
    x, w = FP.bf16(rng.normal(0, 1, (FP.N, FP.K))), FP.bf16(rng.normal(0, 0.02, (FP.N, FP.K)))
    forms = {"honest": ([("r0", V2, 32 * FP.K, ["x", "w"])], {"r0": np.concatenate([x, w], axis=1)}),
             "two-groups": ([("r0", V2, 16 * FP.K, ["x"]), ("r1", V2, 16 * FP.K, ["w"])], {"r0": x, "r1": w})}
    salt_key = hashlib.sha256(b"verity/flock/form-refusals/salts").digest()
    cases: dict[str, dict] = {}
    for name, (in_rows, rows) in forms.items():
        d = work / name
        stage = d / "stage"
        stage.mkdir(parents=True, exist_ok=True)
        (stage / "partition.json").write_text(json.dumps(part, sort_keys=True))
        (stage / "program.json").write_bytes(canonical_json(encode_program(program)))
        st = CS.UnitStager(fn, f"unit/form-refusals-{name}", in_rows, [(V2, (0,))], circuit=stage / "circuit.txt",
                           bind={"partition": P.digest(part), "program": part["program"]})
        FP.stage_case(st, rows, (), d, salt_key, f"{st.iset.name}-{name}")
        (d / "tables").mkdir(exist_ok=True)
        (d / "circuit.txt").symlink_to("stage/circuit.txt")
        sd = {"dir": d, "pin": st.pin, "partition_file": stage / "partition.json", "program_file": stage / "program.json"}
        record, live = a0.session(a.bin, sd, FP.N, f"subset:{FP.N}", FP.free_port(), d / "run", zk=True)
        lean = a0.lean_verify(a.lean, sd, FP.N, d / "run", zk=True)
        cases[name] = {"live": live, "m0": (record or {}).get("verdict"), "lean": lean,
                       "draw": (record or {}).get("unit_draw")}
        if name == "honest":
            cases[name]["wrong_tags"] = {t: a0.lean_verify(a.lean, {**sd, "statement": t}, FP.N, d / "run", zk=True,
                                                           log=f"lean-verify-{t.replace('/', '_')}") for t in WRONG_TAGS}
        print(f"form-refusals {name}: M0 {cases[name]['m0']}, Lean {lean.get('accepted')} {lean.get('why')}", flush=True)
        recs = sorted((d / "run" / "sessions").glob("*/session.json"))
        if recs:
            shutil.copytree(recs[-1].parent, fx / name / "session")
        shutil.copytree(stage, fx / name / "stage")
        shutil.copy(d / f"pub-{FP.N}.bin", fx / name / f"pub-{FP.N}.bin")
    (fx / "cases.json").write_text(json.dumps({"kind": FP.KIND, "n": FP.N, "draw_law": f"subset:{FP.N}", "cases": cases},
                                              indent=1, sort_keys=True, default=str))
    honest, two = cases["honest"], cases["two-groups"]
    ok = (honest["lean"]["accepted"] and not any(v["accepted"] for v in honest["wrong_tags"].values())
          and not two["lean"]["accepted"] and two["lean"]["why"] == SCOPE_WHY)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
