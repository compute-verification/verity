"""The TINY `Serve@3` pipeline target: plan -> reference-evaluator replay -> commit -> verify, CPU-side.

PAD-IMPL's landed implementation (``verity_vllm/registry/serve3.py`` + ``serve3_reference.py``,
PAD-SPEC ``docs/vllm-poc/serve3-spec.md`` v1.1) run as a pipeline stage on the spec's own fixture
(TINY-PAD, ``LP = 4``, ``D_MAX = 8``, pinned rng seeds 1001/2002 -- the same fixture as
``tests/ir/test_serve3.py``).  For each of the three EOS bindings of spec §13
(realized live counts ``l in {0, 3, 8}``):

1. **plan/program** -- ``Serve_v3{TINY_PAD(eos), LP, D_MAX}`` built from the registered
   definitions; the program digest and the measured gate budget (spec §14.1: 56,936) recorded.
2. **replay** -- the reference evaluator materializes the full transcript from the committed
   inputs (every value, live and canonical-garbage alike, by the registered gate semantics).
3. **commit** -- sha256 over the canonical transcript encoding and over the designated output
   tuple ``(out, live)``.  These are *digests, not the adapter's Merkle commitments*: the
   commit-v2/selective machinery binds a GPU capture, which TINY has no record of yet; when a
   TINY capture lands the digests here pin what its commitments must open to.
4. **verify** -- the designated outputs are cross-checked bit-for-bit against the vectorized
   twin (``pad_chain`` on ``Serve_v1``'s raw samples -- itself replayed, not pinned), the
   interpretation map must return ``(l, y_0..y_{l-1})``, and the inactive suffix must equal
   the pad constants (out = 0, live = 0: spec §6's absorbing boundary).

The record also carries the GUIDANCE §30 ``W_out`` anchors for the designated output tuple
(items 2/3/10): ``(D_MAX+1)`` token slots + ``(D_MAX+1)`` liveness bits, under both open-D1
conventions (typed vocab domain log2(V) vs wire width 32).

Usage: ``python -m e2e.tiny_serve3 --out <dir>`` from ``vllm-poc/`` (the e2e_b0.sh ``tiny``
stage does this; the repo root must be importable for ``verity.ir``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

LP, D_MAX = 4, 8
EOS_FOR = {0: 15, 3: 17, 8: 63}  # realized live decode-step count l -> EOS_ID (spec §13 fixture)
SPEC_GATES = 56936  # spec §14.1, measured


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="output directory for tiny_record.json")
    args = ap.parse_args()

    import numpy as np
    import veritor

    repo_root = Path(veritor.__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from verity.ir import Program, bind
    from verity.ir.codec import program_digest
    from verity_vllm.registry import b1, serve3
    from verity_vllm.registry import serve3_reference as ref
    from verity_vllm.registry.b1 import weights_type

    TINY = {"model": "tiny", "revision": None, "H": 32, "NL": 2, "NH": 2, "KVH": 1, "D": 16, "I": 32,
            "V": 64, "QKV": 64, "GU": 64, "EPS": 1e-5, "ROPE_THETA": 1e4, "MAX_POS": 16, "BIAS": True,
            "tie_word_embeddings": True, "profile": {}}

    # the pinned fixture inputs (identical construction to tests/ir/test_serve3.py)
    rng = np.random.default_rng(1001)
    w = ((rng.standard_normal(weights_type(TINY).leaves) * 0.05).astype(np.float32).view(np.uint32) >> 16)
    prompt = np.random.default_rng(2002).integers(0, TINY["V"], LP)
    inputs = [[int(x) for x in prompt], [int(x) for x in w]]

    # Serve_v1's raw greedy samples, replayed (not pinned): the vectorized twin's input
    v1 = Program(bind(b1.Serve, C=TINY, LP=LP, STEPS=D_MAX))
    t1 = ref.materialize(v1, inputs)
    y_raw = [t1[g] for g in v1.circuit.root_outputs()]

    rows = []
    for ell, eos in EOS_FOR.items():
        desc = serve3.serve_v3({**TINY, "EOS_ID": eos}, LP, D_MAX)
        if desc.gates != SPEC_GATES:  # the descriptor's body gates, spec §14.1's number (Program adds input gates)
            raise AssertionError(f"eos={eos}: gate budget {desc.gates} != spec §14.1 {SPEC_GATES}")
        prog = Program(desc)
        t = ref.materialize(prog, inputs)
        outs = [t[g] for g in prog.circuit.root_outputs()]
        out, live = outs[: D_MAX + 1], outs[D_MAX + 1:]

        # verify: vectorized twin, interpretation, absorbing suffix
        tw_out, tw_live = ref.pad_chain(y_raw, eos)
        if list(tw_out) != out or list(tw_live) != live:
            raise AssertionError(f"eos={eos}: gate-level (out, live) != vectorized twin")
        n, transcript = ref.interpret(out, live)
        if n != ell or list(transcript) != y_raw[:ell]:
            raise AssertionError(f"eos={eos}: interpret gave (n={n}, {transcript}), expected (l={ell}, {y_raw[:ell]})")
        if ell < D_MAX and (any(out[ell + 1:]) or any(live[ell:])):
            # slot ell is stop evidence (spec §8) and stays; everything after it must be the pad constant.
            # ell = D_MAX has no inactive suffix (EOS never sampled; all flags stay live).
            raise AssertionError(f"eos={eos}: inactive suffix is not the pad constant")

        enc = json.dumps(sorted(t.items()), separators=(",", ":")).encode()
        rows.append({
            "eos_id": eos,
            "live_steps": ell,
            "program_digest": program_digest(prog),
            "gates": desc.gates,
            "transcript_sha256": _sha(enc),
            "designated_output_sha256": _sha(json.dumps([out, live], separators=(",", ":")).encode()),
            "interpretation": {"n": n, "transcript": list(transcript)},
            "checks": ["gate budget = spec §14.1", "designated outputs = vectorized twin (bit-for-bit)",
                       "interpretation (n, y) as expected", "inactive suffix = pad constants"],
        })
        print(f"  eos={eos}: l={ell}, {desc.gates} gates, digest {rows[-1]['program_digest'][:16]}..., "
              f"transcript sha {rows[-1]['transcript_sha256'][:16]}...")

    slots = D_MAX + 1
    record = {
        "schema": "veritor/tiny-serve3-record/v1",
        "generated": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "fixture": {"config": "TINY-PAD (serve3-spec.md §14)", "LP": LP, "D_MAX": D_MAX,
                    "seeds": {"weights": 1001, "prompt": 2002}, "y_raw_replayed": y_raw},
        "bindings": rows,
        "commit_note": "sha256 digests over the reference-evaluator transcript and the designated output "
                       "tuple -- not adapter Merkle commitments (those bind a GPU capture; none exists for "
                       "TINY yet). When a TINY capture lands, its commitments must open to these values.",
        "w_out_anchors": {
            "designated_tuple": f"{slots} token slots + {slots} liveness bits (GUIDANCE §30 recommendation)",
            "declared_domain_bits": slots * 16 + slots,
            "typed_vocab_bits": round(slots * math.log2(TINY["V"]) + slots, 9),
            "wire_width_bits": slots * 32 + slots,
            "d1": "RESOLVED by OUT16 (GUIDANCE §32.5/.7, landed c4d52ad0): declare/enforce/account with "
                  "declared_domain = 16 on token slots; kind tables charge min(wire, declared), so the "
                  "operative W_out uses 16 bits per token slot (the typed-vocab and wire numbers are kept "
                  "for context). Enforcement needs the declaration in the profile "
                  "(llama-eager-batch-invariant-v3 for B0; a TINY Serve@3 capture profile must do the same).",
        },
    }
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "tiny_record.json"
    path.write_text(json.dumps(record, indent=1) + "\n")
    print(f"tiny_serve3: 3/3 EOS bindings verified (plan -> replay -> digest -> verify); "
          f"W_out {record['w_out_anchors']['declared_domain_bits']} bits (OUT16 declared) / "
          f"{record['w_out_anchors']['wire_width_bits']} wire -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
