"""The fixture of Flock's forged-proof negatives (`verity/ml/flock/tests/test_lean_forged_proofs.py`): M0 `--zk` sessions on
one small A8 unit, staged by Flock's `UnitStager` bound to a template query's partition as A8's members are, N instances
with every unit drawn (subset:N), each case's statement on the one circuit:

    honest                      the honest statement                                              M0 and Lean accept
    wrong-output-word           bit 0 of every instance's first output word flipped before        refused in the proof
                                commitment (A8's tampered member)
    wrong-output-word-one-unit  the same flip on one instance alone                               refused in the proof

A tampered statement's rows and b || c agree with the outputs it commits, so only the proof can refuse it. Its prover runs
with its self-check skipped (``FC_ZK_SELF_CHECK=skip``) and sends the proofs of the false witness; with the check, it sends
none, and Lean refuses the session for want of a proof (A8's negative).

    PYTHONPATH=<tree>:<tree>/verity/ml/flock/python:<tree>/integrations/vllm:<tree>/tools/verity/circuit_check/src:\\
      <tree>/benchmarks/numerical/python python benchmarks/one_stage/forged_proofs.py --bin flock-circuit --lean flock-verify \\
      --work DIR --fixture DIR

``--fixture`` gets the tree: ``stage/`` (circuit.txt, partition.json, program.json), per case ``<case>/pub-N.bin`` and
``<case>/session/`` (the server's session record and the proofs), and ``cases.json`` (each case's flips, the prover's LIVE line,
the M0 verifier's verdict and Lean's at generation). Never a prover's inst file; ``--work`` keeps those and every log."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import a0  # noqa: E402
from circuit_check import targets as TG  # noqa: E402
from verity.ml.circuits.codec import canonical_json, encode_program  # noqa: E402
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2  # noqa: E402
from verity_flock import circuit as FC  # noqa: E402
from verity_flock import class_statement as CS  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import partition as P  # noqa: E402

#: A8's LlBf16DotprodGemm unit (K=6144 there) at the smallest K: x then w, K bf16 leaves each, one row of both
K = 128
KIND = f"LlBf16DotprodDot_v1{{K={K},BS=128,VEC=8,FMA=1}}"
N = 4
#: each case's instances whose first output word has bit 0 flipped before commitment
CASES = {"honest": (), "wrong-output-word": tuple(range(N)), "wrong-output-word-one-unit": (1,)}


def bf16(x: np.ndarray) -> np.ndarray:
    """f32 -> bf16 bits, rounded to nearest even."""
    u = np.asarray(x, dtype=np.float32).view(np.uint32)
    return ((u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)).astype(np.uint16)


def stage_case(st, rows: dict, flip: tuple[int, ...], d: Path, salt_key: bytes, set_name: str) -> None:
    """`UnitStager.stage`, with bit 0 of the first output word of each instance in ``flip`` flipped before the set is committed
    (A8's `stage_tampered`); the prover's file is ``inst-N.bin``, the verifier's ``pub-N.bin``."""
    iset = SimpleNamespace(**{**vars(st.iset), "name": set_name})
    flat = np.concatenate([np.asarray(rows[r], dtype=np.uint16) for r in st.row_ports], axis=1)
    got, _cut, ok = st.low.evaluate(flat.astype(np.uint64))
    if int(ok) != (1 << N) - 1:
        raise AssertionError(f"{st.name}: a unit is unsatisfied on its rows")
    got = got.astype(np.uint16)
    at = np.cumsum([0] + st.out_words)
    outs = {f"o{j}": got[:, at[j]:at[j + 1]].copy() for j in range(len(st.out_words))}
    st.reference_check(rows, outs)
    for i in flip:
        outs["o0"][i, 0] ^= 1
    d.mkdir(parents=True, exist_ok=True)
    FC.write(st.text, iset, 0, N, d / f"inst-{N}.bin", d / f"pub-{N}.bin", salt_key,
             rows={r: np.asarray(rows[r], dtype=np.uint16) for r, *_ in st.in_rows}, outs=outs,
             salts=FC.stage_salts(iset, 0, N, st.ports, salt_key), public_words={}, digests=(st.pin, st.class_sha512),
             indices=list(range(N)))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, help="flock-circuit, the proving build (sha512, glue)")
    ap.add_argument("--lean", required=True, help="flock-verify")
    ap.add_argument("--work", required=True)
    ap.add_argument("--fixture", required=True)
    a = ap.parse_args()
    work, fx = Path(a.work), Path(a.fixture)
    stage = work / "stage"
    stage.mkdir(parents=True, exist_ok=True)
    TG.load_registries()
    fn = TG.definition(KIND)
    program = P.mixed_population_program([(fn, N)])
    part = P.partition(program, P.template_instances_query([fn]))
    (stage / "partition.json").write_text(json.dumps(part, sort_keys=True))
    (stage / "program.json").write_bytes(canonical_json(encode_program(program)))
    st = CS.UnitStager(fn, "unit/forged-proofs", [("r0", V2, 32 * K, ["x", "w"])], [(V2, (0,))], circuit=stage / "circuit.txt",
                       bind={"partition": P.digest(part), "program": part["program"]})
    rng = np.random.default_rng(20261019)
    rows = {"r0": np.concatenate([bf16(rng.normal(0, 1, (N, K))), bf16(rng.normal(0, 0.02, (N, K)))], axis=1)}
    salt_key = hashlib.sha256(b"verity/flock/forged-proofs/salts").digest()
    cases: dict[str, dict] = {}
    for name, flip in CASES.items():
        d = work / name
        stage_case(st, rows, flip, d, salt_key, f"{st.iset.name}-{name}")
        (d / "tables").mkdir(exist_ok=True)
        (d / "circuit.txt").symlink_to("../stage/circuit.txt")
        sd = {"dir": d, "pin": st.pin, "partition_file": stage / "partition.json", "program_file": stage / "program.json"}
        os.environ.pop("FC_ZK_SELF_CHECK", None)
        if flip:
            os.environ["FC_ZK_SELF_CHECK"] = "skip"
        try:
            record, live = a0.session(a.bin, sd, N, f"subset:{N}", free_port(), d / "run", zk=True)
        finally:
            os.environ.pop("FC_ZK_SELF_CHECK", None)
        lean = a0.lean_verify(a.lean, sd, N, d / "run", zk=True)
        cases[name] = {"flip": list(flip), "live": live, "m0": (record or {}).get("verdict"), "lean": lean,
                       "draw": (record or {}).get("unit_draw")}
        print(f"forged-proofs {name}: M0 {cases[name]['m0']}, Lean {lean.get('accepted')} {lean.get('why')}", flush=True)
        recs = sorted((d / "run" / "sessions").glob("*/session.json"))
        if recs:
            shutil.copytree(recs[-1].parent, fx / name / "session")
        (fx / name).mkdir(parents=True, exist_ok=True)
        shutil.copy(d / f"pub-{N}.bin", fx / name / f"pub-{N}.bin")
    shutil.copytree(stage, fx / "stage")
    (fx / "cases.json").write_text(json.dumps({"kind": KIND, "n": N, "pin": st.pin, "draw_law": f"subset:{N}", "cases": cases},
                                              indent=1, sort_keys=True, default=str))
    honest, forged = cases["honest"], [c for n, c in cases.items() if n != "honest"]
    ok = honest["lean"]["accepted"] and all(not c["lean"]["accepted"] and str(c["lean"]["why"]).startswith("zk inner")
                                            for c in forged)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
