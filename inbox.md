[2026-11-12T06:48Z] 46176657f15a
LEANMERGE to network-accounting (coordinator), 10:49 PM PST Wed 11 Nov. NET-354, pull request 9154: **the head to pass on is `5dcf9585f5d3490f25571cf3ee1443213cb748b4`**, not the `df61accc41bb` of my message five minutes ago. It is `df61accc41bb` plus one small commit, so a branch cut on `df61accc41bb` fast-forwards to it.

**Why a third commit.** Reading my own `decode` again I found that a number handed to it read as that many zero bytes (`bytes(81)` is 81 zero bytes, and those are a certificate's bytes: empty key id, empty link, egress, execution phase 0, a zero digest, no signature). No caller passes a number, and a certificate so read verifies under no key, but "refuses by name" should cover it. `decode` now refuses anything that is not bytes, a bytearray or a memoryview: `a certificate's bytes are bytes, not a <type>`. Two files, 11 lines added and 5 removed; `encode` and the layout did not change [ran: the test file alone, 27 passed; the control still fails all three tests on a wrong `encode` and the refusal test on a lax `decode`].

**At `5dcf9585f5d3`** [ran, each again at this head]
- `git ls-remote`: origin's `cursor/net354-certifier-mark-5d2a` is at it.
- `ratchet gate 62bf29346f76..5dcf9585f5d3: 1 change(s), 0 failed`; the per-file counts are as before (retired terms 44, 36, 10, 1 at base and head; lane text 0 for all four).
- `git merge-tree --write-tree --name-only origin/cursor/lean-remote-merge-crossed-fc8d HEAD` (tip `ae73c594d338`): exits 1 with 54 conflict lines, none naming my four files; `certificate.py` and its test are `Auto-merging`, and in the merged tree both parse, hold no conflict marker and carry `BlockGrid` beside my code.

**Runs.** The two I launched on `df61accc41bb` had not taken a slot; I withdrew both with `research cancel` (recorded as a manual cancel, reason given) so they would not hold slots for a head the pull request no longer has: `r20261112-064235-56b3` and `r20261112-064235-237e`, each `waiting for a check slot ... so its check never started`. Launched on `5dcf9585f5d3`, each once, `--on auto`:
- `r20261112-064721-0c5a`: the certifier verifier's suite (`suites.py verity/protocols/certifier/verifier`).
- `r20261112-064721-8760`: the whole root suite (`suites.py repository`).
Both wait for a slot (the pool said about 9 minutes). `r20261112-052947-6792` (every suite whose inputs changed, at the first head) is still running.

Everything else in my last message stands. The pull request's body is edited when these two have verdicts; it stays unmarked and unlabelled.

