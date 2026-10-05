"""red-team-leaf-2: BLAKE3 role confusion against the pinned verifier.  ``--swap`` exchanges the x and W keys
(``leaf.blake3.KEYS``) before composing, so the prover's circuit, digests and trees agree with each other (Python accepts)
but hash x rows under W's key and W columns under x's; the pinned Rust verifier must refuse it (the system is not the
pinned one), ``--allow-any-system`` shows what the pin alone is guarding.  Without ``--swap``: the honest fixture, which
must be accepted pinned.

    PYTHONPATH=<tree>:... python leaf2_blake3_roles.py --out DIR --bin <ligero-verify> [--swap] [--relation fp8-ada]"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--relation", default="fp8-ada")
    ap.add_argument("--swap", action="store_true")
    ap.add_argument("--min-l", type=int, default=256)
    ap.add_argument("--soundness-bits", default="100")
    args = ap.parse_args()
    from archive.direct.ligero.leaf import blake3 as B3
    if args.swap:
        B3.KEYS = {1: B3.KEYS[2], 2: B3.KEYS[1]}
    from archive.direct.ligero.leaf.fixtures import write_fixture

    out = Path(args.out)
    man = write_fixture(out, args.relation, "blake3", n_vus=2, vus=(1,), mode="fiat-shamir", device="cpu", min_l=args.min_l)
    print(f"python verdict: {man['python_verdict']}; system {man['system']}")
    res = {}
    for flag in ([], ["--allow-any-system"]):
        r = subprocess.run([args.bin, "verify", "--system", str(out / "system.bin"), "--statement", str(out / "sub_00.stmt"),
                            "--proof", str(out / "sub_00.proof"), "--soundness-bits", args.soundness_bits, *flag],
                           capture_output=True, text=True)
        js = json.loads(r.stdout.strip().splitlines()[-1])
        res[" ".join(flag) or "pinned"] = js
        print(f"rust verify {' '.join(flag) or '(pinned)'}: accepted={js['accepted']} pinned={js.get('system_pinned')} "
              f"reason={js['reason'][:200]}")
    (out / "redteam_verdicts.json").write_text(json.dumps(res, indent=1))
    want_pinned = not args.swap
    return 0 if res["pinned"]["accepted"] == want_pinned else 1


if __name__ == "__main__":
    raise SystemExit(main())
