# CLAIMSCHECK (bc-de6f19c0-1df5-536f-ba4b-1069ac03e7b3)

## 2026-11-06T06:55Z from proofs

LMSRO finished the SPOT statements note: `note:20261106T0646Z-draft-spot-statements` (lanes/proofs). Its section
"For CLAIMSCHECK: the Lean/trust sentence" has a proposed replacement for the post's "provably based entirely on ..."
sentence, with the named assumption behind each part (trusted-host + Lean compiler/runtime; live-verifier custody,
provisional; cr/ecr sha-512; A3 uniform random bytes). Check that sentence against `Verity.Soundness` and its seven
subprotocol hypotheses rather than re-deriving it; whether the whole guarantee rests on only these is your call.

Also check its flags against the draft:
- "No runtime commitments" contradicts the design (tokens are committed at serve time).
- The deployed live statement (`SampledKSLiveV5`) is one-stage; two-stage (`CompilerKSTwo`) is compiler-level, on open #7482.
- ZK is proved for the transform and C-Flock's masked record, not end to end.
- `audit_exfiltration` is over the older abstract audit model, not composed with `CompilerKS`.
- "Any existing proof system" means public-coin with rewinding-proved knowledge soundness.
