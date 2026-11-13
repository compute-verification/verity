"""stage_window.py WINDOW_DIR OUT (--member K | --pick P N [--swap 0|1]): one member of a serving-rows window (a TP Commit's
``--serving-rows`` window, or ``verity-vllm tp-serving-rows``' from its sealed bundles) staged for Flock as its served statement,
for `prove_time.sh`'s ``PT_STATEMENTS``: the verifier's circuit for the member's template (`chunk_rows.statement` by
`UnitStager`, as ``benchmarks/one_stage/compose_circuit.py --template`` composes it) bound to the window's partition and program
digests, and the member's public and prover files made from the window's own bodies (`flock_inputs.flock_input_files`): its
served values under the committer's salts, nothing sampled; the public body must be the one the window's member record states
(``public_sha512``).

``--pick P N``: the first member of N units of the per-expert cut's fused block of projection P (``w13`` or ``w2``,
`expert_rows`' ``fused`` layout), on the kernel ``--swap`` names when given, or of its scale read (``s13`` or ``s2``: the read
of H = 2 K / G words a row, 96 and 4 at GLM-5.3's statics).

Writes circuit.txt, pub.bin, inst.bin (the prover's: keep OUT off custody), pin.txt and stage.json (`stage.py`'s fields, ``unit``
``window``, ``macs`` a unit's N K) under OUT."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

FUSED = {"w13": "MoeExpertFp8ReadGemm_v1", "w2": "MoeExpertFp8ReadGemmW_v1"}
SCALES = {"s13": 96, "s2": 4}
READ = "EmbeddingRead_v1"


def pick(members: list[dict], proj: str, n: int, swap: int | None) -> int:
    """The index of the first member of ``proj`` (a `FUSED` block on the kernel ``swap`` names, or a `SCALES` read) holding ``n``
    units."""
    from verity.definitions.call_scope import spec_base, spec_statics

    def kind(t: str) -> bool:
        S = spec_statics(t)
        if proj in SCALES:
            return spec_base(t) == READ and int(S.get("H", 0)) == SCALES[proj]
        return spec_base(t) == FUSED[proj] and (swap is None or str(S["SWAP"]).lower() in (("true", "1") if swap else ("false", "0")))

    if proj not in FUSED and proj not in SCALES:
        raise SystemExit(f"stage_window.py: --pick takes {sorted(FUSED) + sorted(SCALES)}, not {proj}")
    for k, m in enumerate(members):
        if int(m["n"]) == n and kind(m["template"]):
            return k
    raise SystemExit(f"stage_window.py: no {proj} member of {n} units{'' if swap is None else f' at SWAP={swap}'} "
                     f"(sizes: {sorted({int(m['n']) for m in members if kind(m['template'])})})")


def public_body(d: Path, k: int, pub: Path) -> str:
    """The SHA-512 of the staged public file ``pub``'s body (after its header line), refused unless that body is member ``k``'s
    in window ``d``: its ``m<k>-public.bin``, with the ``public_sha512`` its window.json record states."""
    body = pub.read_bytes().split(b"\n", 1)[1]
    public = hashlib.sha512(body).hexdigest()
    want = json.loads((d / "window.json").read_text())["members"][k].get("public_sha512")
    if body != (d / f"m{k}-public.bin").read_bytes() or public != want:
        raise SystemExit(f"stage_window.py: member {k}'s public body {public[:16]} is not the window's ({str(want)[:16]})")
    return public


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("window")
    ap.add_argument("out")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--member", type=int)
    g.add_argument("--pick", nargs=2, metavar=("P", "N"))
    ap.add_argument("--swap", type=int, choices=(0, 1), default=None)
    a = ap.parse_args()

    from stage import gb, nets
    from verity_flock.class_statement import UnitStager
    from verity.definitions.call_scope import spec_base, spec_statics
    from verity.commitments.windows import flock_inputs as FI
    from verity_vllm.pipeline import chunk_rows as CR
    from verity_vllm.pipeline import serving_rows as SRW

    t0 = time.time()
    d, out = Path(a.window), Path(a.out)
    part = json.loads((d / "partition.json").read_text())
    rec = json.loads((d / "registration.json").read_text())
    members = json.loads((d / "circuit.json").read_text())["members"]
    k = a.member if a.member is not None else pick(members, a.pick[0], int(a.pick[1]), a.swap)
    t, n = members[k]["template"], int(members[k]["n"])
    tp = SRW.template(t)
    stmt = CR.statement(tp)
    in_rows, out_rows = CR.unit_rows(tp)
    out.mkdir(parents=True, exist_ok=True)
    os.chmod(out, 0o700)
    st = UnitStager(tp.fn, "unit/serving-rows", in_rows, out_rows, circuit=out / "circuit.txt",
                    bind={"partition": part["partition_digest"], "program": part["partition"]["program"]}, **stmt)
    t_built = time.time() - t0
    got = FI.flock_input_files(d, k, FI.flock_circuit((out / "circuit.txt").read_bytes()), out)
    os.replace(got["pub"], out / "pub.bin")
    os.replace(got["inst"], out / "inst.bin")
    public = public_body(d, k, out / "pub.bin")
    (out / "pin.txt").write_text(got["circuit_sha512"])
    S = spec_statics(t)
    meta, layers = st.meta, nets(st.low)
    tag = f"window-{spec_base(t)}" + (f"-swap{str(S['SWAP']).lower()}" if "SWAP" in S else "")
    stage = {"unit": "window", "tag": tag, "n": n, "member": k, "template": t,
             "run": rec["window"]["run"], "partition_digest": part["partition_digest"], "public_sha512": public, "statics": {"K": int(S.get("K", 0))},
             "macs": int(S.get("N", 0)) * int(S.get("K", 0)), "definition": tp.fn.id, "statement": stmt,
             "ands": sum(x["ands"] for x in layers), "nets": layers, "public_inputs": [], "k_log": meta["k_log"],
             "unit_log": meta.get("unit_log"), "vus_per_block": meta.get("vus_per_block"), "circuit_sha512": got["circuit_sha512"],
             "class_sha512": got["class"], "circuit_bytes": (out / "circuit.txt").stat().st_size,
             "inst_bytes": (out / "inst.bin").stat().st_size, "pub_bytes": (out / "pub.bin").stat().st_size, "twin": None,
             "seconds": {"built": round(t_built, 1), "total": round(time.time() - t0, 1)}, "maxrss_gb": gb()}
    (out / "stage.json").write_text(json.dumps(stage, indent=1))
    print(json.dumps({x: stage[x] for x in ("tag", "n", "member", "ands", "k_log", "vus_per_block", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
