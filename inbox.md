[2026-11-12T06:42Z] cfedcde1653d
CROSSTICK, job G part 2, status at 10:42 PM PST Wed 11 Nov. Nothing here needs you to stop; three questions at the end.

**Where it is.** Branch `cursor/na-span-count-e7c1`, four commits of mine on the issuer's `5181e37f0973` (itself on `main` at `62bf29346f76`), head `dbb6081f58`, not pushed, no pull request yet. [ran: git log]

**Your 10:27 PM answer, applied.** [ran: local tests, 5 passed without Lean]
- Public is `egress_bits`, `phases`, `tail`, said in one function a side (`publicJson` in Lean, `span.public` in Python). The start phase and the end phase are in no output; they stay in the commitment.
- `namedOf(params, commitment)` gives a tie its values off the case: SHA-512 of the start certificate's bytes, D_s, D_e, and the egress in bits.
- The mirror holds two functions marked STAND-IN for `Certificate.encode` / `Certificate.decode`. I delete both when I recut on LEANMERGE's second commit. At 10:28 PM the branch `cursor/net354-certifier-mark-5d2a` was still at `5181e37f0973`. [ran: git fetch]

**One change to the commitment's bytes, so that `decode` can be called on exactly a certificate's bytes.** Each mark is now `u32be len(certificate) ‖ certificate ‖ salt (64) ‖ D (64)`, start then end. The certificate inside is the layout I named at 9:55 PM, unchanged. Lean reads the slot exactly: a short buffer, a length that does not fit and bytes left over in the slot are each a refusal, and the generator has a negative for each (17 malformed commitments). [ran: mirror; Lean not yet]

**Lean.** `verity lean try` on `NetworkCheck/Span.lean` with `--siblings`: lean exit 0 against its real imports, its guards included; no sibling declares a name it does; the axiom probe finds no theorem in it. Run r20261112-064046-9c04, exit 0 in 50 s. [ran]

**The merge you asked about.** `git merge-tree --write-tree --name-only origin/cursor/lean-remote-merge-crossed-fc8d HEAD` at `ae73c594d338`: it stops on 54 files and none of them is a file I add or change. The same against the closes (`a74a9caf815d`): none. Against `origin/main` (`d6075b75d289`): clean. [ran]
One thing I read there that the merge does not show: on `main` and the closes `certificateOf`, `exactly`, `hexAt` and `keysOf` are public in `NetworkCheck/Clocks.lean`; on the crossed head there is no `Clocks.lean` and the last three are `private` in `Period.lean`. My file uses only `certificateOf` of those four (public in all three), and defines its own exact-keys check, so it elaborates either way. [read; argued for the crossed head, where I ran nothing]

**Running now.** r20261112-064153-f8ab: the new test module against Lean on a project machine (a development run, not one of the three formal ones; those wait for the final head). It was told about 6 min for a slot.

**Three things to rule, none blocking.**
1. One more line outside the plan's list: a key `span_count` in `tools/verity/check/public_boundary.toml`, listing the count's two Lean files, its mirror and its two test files as private. Without it the root suite's `test_public_boundary` fails, since the files import `Period.lean`, `Identity/Domains.lean` and `period_case.py`, which are private. I put the key apart from the lines the crossed head rewrites. Say if `span.py` should be public instead.
2. A certificate's link name outside printable ASCII is refused on both sides (the parameters' link is held to the same). If LEANMERGE's `decode` accepts more, the difftest's commitment negatives will show it and I tell you.
3. The rejection reasons name execution phases by number ("execution phase 7 is missing"). A reason is not compared by the difftest and is not public output; say if you want the numbers out of them.

**Slips, for your record.** None since the 9:55 PM line. One `git stash` on my own worktree before it; the tree was checked intact after. [ran]

