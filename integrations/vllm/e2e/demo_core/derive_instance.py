"""Derive a PREDECLARED test instance from E6's frozen CORE instance: the same program / partition / population / layout / input pins, with only the
sampling law (q, s) and the exemption budget B_ex changed (stated ONCE per epoch: ``epoch.B_ex`` == ``security.B_ex``), or -- for the V2 refusal test
-- a deliberately WRONG population pin.  The derivation is recorded in ``notes`` and the instance id names the law.  Generic protocol
demonstration (DemoCore_v1); NOT vLLM inference.

    python -m e2e.demo_core.derive_instance --from <E6 protocol_instance.json> --out <dir> --q 1/2 --s 1/2 --B-ex 0 [--mispin-population]
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
POC = HERE.parents[2]
for p in (POC, POC.parent, POC.parent / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from e2e import partition_sample as ps  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", required=True, help="E6's frozen instance dir (protocol_instance.json + population.compact.json + declared_inputs.json ...)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--q", default=None)
    ap.add_argument("--s", default=None)
    ap.add_argument("--B-ex", type=int, default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--mispin-population", action="store_true",
                    help="NEGATIVE TEST: pin a population of ANOTHER partition (partition_digest flipped in its last byte) so the verifier's V2.population_derived must refuse")
    ap.add_argument("--mispin-population-file", action="store_true",
                    help="NEGATIVE TEST: keep E6's instance, pin the population file of DemoCore_v1{N,R//2} next to it (V2.population_derived must refuse: the file is not the rule's derivation)")
    ap.add_argument("--mispin-params", action="store_true",
                    help="NEGATIVE TEST: pin the program identity's params with R := R//2 (another circuit of the same family) so the verifier's derived population disagrees with the pinned population file (V2.population_derived must refuse)")
    a = ap.parse_args(argv)
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = json.loads((src / "protocol_instance.json").read_text())
    e = copy.deepcopy(d)
    notes = list(e.get("notes") or [])
    # weights_authorized.json = the registered weight commitment (DemoCore_v1: the empty tree) -- part of the pin set the driver copies into the record and
    # the shipped reverify.py requires (V3.weights_root registered=ABSENT refused CORE-live-{sampled-B1,exempt-X1}-03a, MEASURED 01:49Z, whose derived
    # instance dirs lacked it)
    for f in ("population.compact.json", "declared_inputs.json", "pins.json", "weights_authorized.json"):
        if (src / f).exists():
            shutil.copy(src / f, out / f)
    parts = []
    if a.q is not None:
        e["sampler"]["law"]["q"] = a.q; parts.append(f"q={a.q}")
    if a.s is not None:
        e["sampler"]["law"]["s"] = a.s; parts.append(f"s={a.s}")
    if a.B_ex is not None:
        e["epoch"]["B_ex"] = int(a.B_ex); e["security"]["B_ex"] = int(a.B_ex); parts.append(f"B_ex={a.B_ex}")
        e["epoch"]["B_ex_provenance"] = e["security"]["B_ex_provenance"] = f"E8 predeclared test instance derived from {d['instance_id']}: B_ex = {a.B_ex}" + (f" ({a.label})" if a.label else "")
    if a.mispin_population:
        bad = e["partition"]["partition_digest"][:-2] + ("00" if e["partition"]["partition_digest"][-2:] != "00" else "01")
        for sect in (e["partition"], (e["partition"].get("validation") or {})):
            if "partition_digest" in sect:
                sect["partition_digest"] = bad
        parts.append("MISPINNED partition_digest (NEGATIVE TEST: the population file is of another partition than the one pinned)")
    if a.mispin_population_file:
        # the instance stays E6's (internally consistent, R=8); the DIR pins a population file of ANOTHER circuit of the family (R//2):
        # the verifier's rule-derived population must disagree with it (V2.population_derived) before any challenge
        from e2e.demo_core import circuit as C, instance as I

        prm = e["program"]["identity"]["params"]; N, R2 = int(prm["N"]), int(prm["R"]) // 2
        prog2, d2 = C.build(N, R2)
        pop2 = I.population_doc(C.partition(N, R2, prog2, d2))
        (out / "population.compact.json").write_text(json.dumps(pop2, indent=1, sort_keys=True) + "\n")
        parts.append(f"MISPINNED population file (NEGATIVE TEST: population.compact.json is DemoCore_v1{{N={N},R={R2}}}'s, the instance is {d['instance_id']})")
    if a.mispin_params:
        from e2e.protocol_instance import generic_config_sha12
        from e2e.demo_core import circuit as C

        prm = dict(e["program"]["identity"]["params"]); prm["R"] = int(prm["R"]) // 2
        prog, _d = C.build(int(prm["N"]), int(prm["R"]))
        dig = C.partition(int(prm["N"]), int(prm["R"]), prog, _d).program_digest
        idn = e["program"]["identity"]; idn["params"] = dict(prm); idn["function"] = f"DemoCore_v1{{N={prm['N']},R={prm['R']}}}"; idn["case"] = f"N{prm['N']}R{prm['R']}"
        idn["config_sha12"] = generic_config_sha12(prm)
        e["program"]["builder"]["params"] = dict(prm); e["program"]["builder"]["program_digest"] = dig; e["program"]["digest"] = dig
        e["program"]["gates"]["value"] = int(prog.gates)
        parts.append(f"MISPINNED program params R={prm['R']} (NEGATIVE TEST: the identity names another circuit than the population's)")
    e["instance_id"] = "-".join(["E8"] + (["MISPIN"] if (a.mispin_population or a.mispin_params or a.mispin_population_file) else []) + [d["instance_id"]] + [p.replace("/", "o").replace("=", "") for p in parts if not p.startswith("MISPINNED")])
    e["notes"] = notes + [f"E8 derived instance ({'; '.join(parts)}) from E6's frozen {d['instance_id']}; every other pin unchanged" + (f"; {a.label}" if a.label else "")]
    (out / "protocol_instance.json").write_text(json.dumps(e, indent=1) + "\n")
    load, _ = ps.loaders("contract")
    try:
        inst = load(out / "protocol_instance.json")
        print(f"[derive_instance] {e['instance_id']}: loads under the contract format; law q={inst.sampler.law.q} s={inst.sampler.law.s} B_ex={inst.epoch.B_ex}")
    except Exception as ex:  # noqa: BLE001
        print(f"[derive_instance] {e['instance_id']}: REFUSED by validate_instance: {type(ex).__name__}: {str(ex)[:300]}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
