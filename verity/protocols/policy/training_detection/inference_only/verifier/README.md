# The inference-only policy's verifier

The Lake package `inference_only_verifier` (`lean/`) and the manifest `parties.py`; the policy, its pieces and its
guarantee are in [the policy's README](../README.md).

Each executable is a `lake exe` in `lean/`:

| command | form |
| --- | --- |
| `inference-only id \| check \| serve` | `Protocol` |
| `inference-only-verifier [--unrolled] INPUT TRUSTED` | `Theta` on `Desc`, or on `Check` with `--unrolled` |
| `inference-only-verifier --policy [--unrolled] INPUT` | `Desc` alone, or `Check` |
| `inference-only-verifier --count [--detail] INPUT` | `Count` |
| `inference-only-record-check CASE` | `Record` |
| `inference-only-{,desc-,record-,theta-,count-}difftest`, `-desc-liveness`, `-desc-fuzz` | the difference tests |

Older names stay until a change of their own renames each (`tests/lint/retired_terms.md`): `benchmarks/nci`; a
verdict's rule `"nci"`, its key `ius` and the refusal codes `input-in-iu`, `gate-in-no-iu` and `empty-iu`; the toy
Definitions `NciHandBlk` and `NciDecl…`; `baseline_difftest/`, whose bytes the lock pins; "baseline", "IU" and "fresh
input bits" in the lock's reasons and difftest messages; and `novel` in the check's rate rule and its Python reference.
