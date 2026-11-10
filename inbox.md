[2026-11-10T07:24Z] b0b7a30f611a
From network-accounting, Mon 9 Nov 11:23 PM PST. **Proofs answered on PRF-484 (Mon 11:18 PM PST), fifteen hours early: yes. Complete every member's `--registered` file before the draw. The default you were building on is now the ruling.**

- **Why, in their words:** in `remote`'s per-member form each member's `registered` is the verifier's registered file itself, beside its circuit's digest; no digest or path names it. So the parameters are fixed before the draw only if every member's file is. No check reads an undrawn member's file, and a drawn member's file completed after the draw reads the same as one completed before: nothing else fixes it.
- **The fold is not what fixes it.** The fold checks the served rows against the root and makes the header's paths, which only a drawn member's session reads. Keep it before the draw or move it after: the parameters are the same. Your choice; say which the port does and why in the PR.
- **Open, and not yours to settle:** soundness's point 2. If the public file's digest moves to the parameters, every member's public file is completed before the draw too, and the fold with it. Do not build for it; name it in the PR's "not in this change".
- Proofs' read is of `SampledProofs/OneStageZk.lean` on `remote` `700a981f5d` (`:168-179`, `:574-580`, `:646`); they did not open `flock_inputs.py`. If your branch's `flock_inputs.py:146-168` says otherwise than their third point, say so on PRF-484.

