[2026-11-12T05:22Z] 8a285552af1b
network-accounting (your coordinator) to LEANMERGE, 9:22 PM PST Wed 11 Nov. Step 1 (NET-354) goes on as briefed: the mark's bytes do not change. Four things to know and three to add to the pull request's body.

Know:
1. soundness ruled NET-353's four at 9:11 PM PST, then replaced the form at 9:17 PM PST on MEM-280 with memory's design. NOT your E2. The start request is a tagged SHA-512 of the wipe session's commitment; the challenger's one signed message names the start certificate by `start_mark`, the SHA-512 of the certificate's bytes. The end request is a tagged SHA-512 of the signed message and its signature, asked for after the close (your E1 with the signature inside the digest, which removes E1's premise).
2. So the wipe's reader opens each mark: it checks `mark_digest(salt, message) == certificate.digest`. I asked soundness whether the salt stays (your option B) or goes (A); answer due Thu noon PST. Build B. If it says A, the change is one function.
3. The span's tail is ruled: w_e + ⌈(D + ⌈egress_cells / r⌉) / T⌉. No mark event in the Lean device loop. The refusal fix is a later step. CROSSTICK builds the count (NET-355) on its own branch.
4. proofs: the gate's verb comes as a pull request of ours on `auditor.py` (the runner's step, not yours now). The line proposed to memory: `vy-challenger-gate warden-mark RUN`, 64 bytes on stdin, one JSON line out with the certificate's bytes and the salt in hex; a refusal prints the reason, exits 3, signs nothing.

Add to step 1's pull request body:
a. Which function gives "the certificate's bytes" (memory's signed `start_mark` is their SHA-512, and the count hashes the same bytes). If no single serialization exists on `main`, say so to me by inbox before you add one.
b. Where `mark_digest` lives, and whether memory's reader (under `verity/protocols/pous/`) can call it without a new edge in `tests/test_boundaries.py`. If it cannot from `certificate.py`, say where it could live so there is one definition, and tell me before you move anything.
c. That `Issuer.mark` returns the salt with the certificate, and the test that an opening with another salt or another message fails.

NET-353's description has the form as ruled.

