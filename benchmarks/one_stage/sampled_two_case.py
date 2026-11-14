"""The two-stage form's case (`VStar.Recursive.SampledTwo`) of one honest ``two_stage.py second`` run, for the form's
accept test (``sampled_two_small.sh``).

The form's check computes the run of record's input from its parameters and evidence (`SampledTwo.inputOf`); this
inverts it. ``capture`` keeps the input the stream's two-stage outcome handed its run of record (`verifier.run`'s
``run`` call, under `service.Stream._audit_two`), and ``assemble`` lays it out as the form's parameters, commitment and
evidence beside the interior audit's per-member case (`a5.case_members`, with the second stage's coins under
``--interior-trees``): the boundary's parameters naming the challenger's key and the first stage's family; the
interior's naming the key and the second stage's family, with the stream's ``proofs`` and the members beside; the
boundary record as the commitment; and the first stage's coins, receipt place and bytes, the interior audit's
per-member evidence, the interior record and the second-stage statements as the evidence. The blobs are the per-member
case's files.

    python sampled_two_case.py RUN_INPUT.json CASE_V2_MEMBERS.json OUT_DIR
"""
from __future__ import annotations

import contextlib
import json
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from verity_difftest import canonical

FORM = "two-stage"


@contextlib.contextmanager
def capture(path: Path) -> Iterator[None]:
    """Keep, in ``path``, the input of every two-stage run of record the verifier hands its Lean (`verifier.lean`'s
    ``run`` with an ``interior``), the last one written last."""
    from verity.protocols.proofs.spot.verifier.one_stage import verifier as V

    real = V.lean

    def lean(fn: str, value: Any) -> dict[str, Any]:
        if fn == "run" and isinstance(value, Mapping) and value.get("interior") is not None:
            path.write_text(json.dumps(value, sort_keys=True))
        return real(fn, value)
    V.lean = lean
    try:
        yield
    finally:
        V.lean = real


def assemble(run: Mapping[str, Any], members: Mapping[str, Any]) -> dict[str, Any]:
    """The form's case of the run of record's input ``run`` and the interior audit's per-member case ``members``:
    {params, commitment (hex), evidence (hex), blobs (paths)}. ValueError where the run is not one the form decides:
    public outputs, no challenger key or family for a stage, or an interior partition other than the one the members'
    sessions are verified against (the interior's parameters carry that object: the members' check refuses null)."""
    it = run["interior"]
    if run.get("outputs"):
        raise ValueError(f"the check reports public outputs {run['outputs']}: the two-stage form refuses them")
    if run.get("key") is None or run.get("family") is None or it.get("family") is None:
        raise ValueError("the run of record names no challenger key or no family for a stage")
    ev = json.loads(bytes.fromhex(members["evidence"]))
    if ev.get("coins") != it.get("coins"):
        raise ValueError("the interior audit's coins are not the second stage's the run of record read")
    part = members["params"].get("partition")
    if part is None or it["params"].get("partition") not in (None, part):
        raise ValueError("the interior's partition is not the object its members' sessions are verified against")
    key = run["key"]
    params = {
        "form": FORM, "boundary": {**run["params"], "challenger": {"key": key, "label": run["family"]}},
        "challenger": run["challenger"], "delta": run["delta"], "outputs": [], "public": run["public"],
        "rule": run["rule"], "values": run["values"],
        "interior": {"params": {**it["params"], "challenger": {"key": key, "label": it["family"]}, "partition": part},
                     "members": members["params"]["proofs"]["members"], "replay": it["replay"],
                     "values": it["values"]}}
    evidence = {"form": FORM, "coins": run["coins"], "interior": ev, "received": run["received"],
                "record": it["record"], "source": run["source"], "statements": it["statements"]}
    return {"params": params, "commitment": canonical(run["record"]).encode().hex(),
            "evidence": canonical(evidence).encode().hex(), "blobs": list(members["blobs"])}


def write(case: Mapping[str, Any], out: Path) -> list[Path]:
    """The case as ``sampled-two check``'s files under ``out``: params.json, commitment.json, evidence.json, then
    the blobs' paths. -> the files in that order."""
    out.mkdir(parents=True, exist_ok=True)
    files = [out / "params.json", out / "commitment.json", out / "evidence.json"]
    files[0].write_text(canonical(case["params"]))
    files[1].write_bytes(bytes.fromhex(case["commitment"]))
    files[2].write_bytes(bytes.fromhex(case["evidence"]))
    (out / "blobs.txt").write_text("".join(f"{b}\n" for b in case["blobs"]))
    return [*files, *(Path(b) for b in case["blobs"])]


def from_run(out: Path) -> dict[str, Any]:
    """The form's case of the ``two_stage.py second --form-case`` run under ``out`` (its captured run input and its
    interior audit's per-member case), written under ``out/form``: {files, blobs} or {error}."""
    try:
        run = json.loads((out / "run-input.json").read_text())
        members = json.loads((out / "audit" / "case-v2-members.json").read_text())
        files = write(assemble(run, members), out / "form")
    except (OSError, ValueError, KeyError, TypeError) as e:
        why = (out / "audit" / "case-v2-members-error.txt")
        return {"error": f"{type(e).__name__}: {e}", **({"members": why.read_text().strip()} if why.is_file() else {})}
    return {"files": [str(f) for f in files[:3]], "blobs": len(files) - 3}


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 3:
        print(__doc__.strip().splitlines()[-1].strip(), file=sys.stderr)
        return 64
    run, members = (json.loads(Path(a).read_text()) for a in args[:2])
    try:
        files = write(assemble(run, members), Path(args[2]))
    except (ValueError, KeyError, TypeError) as e:
        print(f"CASE {json.dumps({'error': f'{type(e).__name__}: {e}'})}")
        return 1
    print("CASE " + json.dumps({"files": [str(f) for f in files[:3]], "blobs": len(files) - 3}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
