"""`whole_run.sh`'s inputs: per SHAPE and N, DIR/run-SHAPE-N.json, the `run` piece's input (`Run.run`, the whole run of one
registration) for a valid v1 audit under the stratified rate of two in the population (`stratified-rate:2/P` over its P
units), on challenger coins with their evidence (`test_difftest._from_challenger`), which the run accepts. Its program is a
template population of about N units (`mixed_population_program`), with N/8 more ports (`test_difftest._many_ports`, a MoE
window's shape) and every template a stratum, so the sample is one unit of each template, and two of one that holds
more than half the units, however large N is:

  toy  the difftest's PAIR, TRIPLE and QUAD at 1:6:3
  l5   a GLM-5.3 MoE window's seven templates (a router, five EmbeddingRead reads and two expert GEMMs), at the counts a
       window of 373,792 units has, scaled to N, each at least one

  python whole_run.py DIR SHAPE N... [--check] [--desc]

--check runs the Python reference (`test_difftest.run`) on each input and refuses unless it accepts. --desc writes only
DIR/desc-SHAPE-N.json, `{"program", "query"}` (`whole-run-time desc`), without the record, whose Python partition walks
every gate of an l5 window's templates.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "verity/protocols/spot/verifier/lean/tests"))

import test_difftest as T  # noqa: E402
from verity.circuits import codec, partition_object as PO  # noqa: E402
from verity.circuits.population import mixed_population_program  # noqa: E402

L5 = [
    ('MoeRouterSigmoidBiasSlots_v2{E=256,TOPK=8,SCALE={"f64":"0x1.4000000000000p+1"},KERNEL="block"}', 32),
    ("EmbeddingRead_v1{V=256,H=3072}", 131072),
    ("EmbeddingRead_v1{V=256,H=96}", 1024),
    ('MoeExpertFp8BlockGemm_v4{K=6144,N=1,G=128,KB=128,SWAP=true,DOT={"fn":"HopperE4m3QgmmaDot32_v2"}}', 131072),
    ("EmbeddingRead_v1{V=256,H=4096}", 49152),
    ("EmbeddingRead_v1{V=256,H=4}", 12288),
    ('MoeExpertFp8BlockGemmW_v4{K=256,N=32,G=128,KB=128,SWAP=true,DOT={"fn":"HopperE4m3QgmmaDot32_v2"}}', 49152),
]


def members(shape: str, n: int) -> list:
    if shape == "toy":
        return [(t, max(1, round(n * w / 10))) for t, w in ((T.PAIR, 1), (T.TRIPLE, 6), (T.QUAD, 3))]
    if shape == "l5":
        sys.path.insert(0, str(ROOT / "benchmarks/prove_path"))
        from records_audit import catalog_resolve
        total = sum(c for _, c in L5)
        return [(catalog_resolve(t), max(1, round(n * c / total))) for t, c in L5]
    raise ValueError(f"shape {shape!r}: toy or l5")


def case(shape: str, n: int) -> dict:
    ms = members(shape, n)
    s = T.Setup(mixed_population_program(ms), PO.template_instances_query([t for t, _ in ms]))
    T.PROGRAMS[s.sha512] = s.program
    T.PROGRAMS_TEMPLATES = s.templates
    law = f"stratified-rate:2/{s.population}"
    named, table = T.service.params_law(law, s.program, s.query, None)
    drawn = T.DR.derive(s.program, s.query, named, table)
    rec = T.R.record(s.program, s.query, named, T.SCHEME, [{"port": p, "root": T.ROOTS[p], **s.ports[p]} for p in s.ports],
                     T.WINDOW, leaf_layer=T.LEAF, work_table=table)
    params = {"anchors": [], "delta": "1/10", "law": law, "leaf_layer": T.LEAF, "log": [], "partition": None,
              "ports": s.ports, "program": s.desc, "query": s.query, "registered": None, "scheme": T.SCHEME,
              "work_table": None}
    evidence = {"session_roots": dict(T.ROOTS), "statement": {"instances": sum(st["k"] for st in drawn["strata"])},
                "verdicts": {"lean": {"accepted": True, "digest": "aa" * 64, "of_record": True},
                             "upstream": {"accepted": True, "of_record": False}}}
    a = T._many_ports(T._json({"evidence": evidence, "params": params, "record": rec}), max(1, s.population // 8))
    _, c = T._from_challenger(T._run_case(a, "", 2, ["credited"]), random.Random(f"whole_run/{shape}/{n}"))
    return T._json(c)


def desc(shape: str, n: int) -> dict:
    ms = members(shape, n)
    return T._json({"program": codec.encode_program(mixed_population_program(ms)),
                    "query": PO.template_instances_query([t for t, _ in ms]), "counts": [c for _, c in ms]})


def main(out: Path, shape: str, ns: list[int], check: bool, only_desc: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for n in ns:
        t0 = time.monotonic()
        if only_desc:
            d = desc(shape, n)
            (out / f"desc-{shape}-{n}.json").write_text(json.dumps(d, sort_keys=True, separators=(",", ":")))
            print(json.dumps({"shape": shape, "n": n, "units": sum(d["counts"]), "program_bytes": len(json.dumps(d["program"])),
                              "python_s": round(time.monotonic() - t0, 1)}), flush=True)
            continue
        c = case(shape, n)
        (out / f"run-{shape}-{n}.json").write_text(json.dumps(c, sort_keys=True, separators=(",", ":")))
        row = {"shape": shape, "n": n, "units": c["record"]["population"], "ports": len(c["params"]["ports"]),
               "roots": len(c["record"]["roots"]), "program_bytes": len(json.dumps(c["params"]["program"])),
               "python_s": round(time.monotonic() - t0, 1)}
        if check:
            got = T.run(c)
            row["python"] = got
            if not (isinstance(got, dict) and got.get("symbol") == "accept"):
                raise SystemExit(f"{shape} {n}: the Python reference doesn't accept: {json.dumps(got)[:400]}")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(Path(args[0]), args[1], [int(n) for n in args[2:]], "--check" in sys.argv[1:], "--desc" in sys.argv[1:])
