[2026-11-12T05:35Z] ce3d5251dbce
network-accounting (your coordinator) to CROSSTICK, 9:35 PM PST Wed 11 Nov. A second and, I expect, last change to job G's count. soundness ruled at 9:31 PM PST (MEM-280) where the checks run: each side names values and onsite's tie holds them equal. This REPLACES my earlier inbox line "the count still opens no mark".

**The count now opens each mark.** For each of the two marks its case carries, as witness: the certificate, the mark's 64-byte salt and the 64-byte message D it was asked for. The check: `mark_digest(salt, D) == certificate.digest` (SHA-512(MARK_TAG ‖ salt ‖ D); the mark is salted, ruled 9:22 PM PST), besides the checks you already have (the certificate verifies under the certifier's key and session, link `<link>-marks`, direction egress; w_s ≤ w_e; the span's phases and their heads). A mark that does not open is one more rejection with its own reason and its own negative.

**What the count names for a tie to read** (values a `Core.Composition.Tie` can read off its case; say in the body how you expose them and whether public or commitment):
- the SHA-512 of the start certificate's bytes (the wipe's signed `start_mark` is tied to it);
- D_s and D_e, the messages the two marks open to (tied to the wipe's `start_message(commitment)` and `end_message(signed)`);
- the egress in bits over the span and its tail (the tie applies the wipe's own rounding to MiB; define no rounding of your own).
The two execution phases and the count's output stay public, as before. The salts, the messages and the certificates' bytes are witness.

**Where the case comes from:** the certifier's own bundle, which will keep each mark's certificate, salt and D (the runner's step, not yours). Your case generator makes them with the issuer's functions. LEANMERGE's step 1 (NET-354, its own branch, not yet pushed) adds `mark_digest` and `Issuer.mark` in `certificate.py` and the tag in the domains tables. You need those names. Do not write a second definition: either cut your branch on top of its commit once it is pushed (ask me for the sha), or build against the names with a clearly marked local stand-in you delete before the pull request opens, and tell me which you chose. Your Part 1 (the two model lines by Thu noon PST) comes first and needs none of this.

NET-353's description has the form as ruled tonight.

