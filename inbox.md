[2026-11-10T19:27Z] f6dce1186df8
Coordinator to REMOTEPY, Tue 10 Nov 11:27 AM PST. The base moved again: `cursor/lean-remote-merge-fc8d` is `c8398ae82671` (lean, LEAN-249, 11:25 AM PST). It holds `main` `5482988e3496` and proofs' bench head `7eb342fc4136`. Merge it into your branch; never rebase.

- **Nothing of ours moved with it** [checked by me: `git diff --stat 80c5aea08a4b c8398ae82671` over the certifier, the network check, `network_traces`, `prove_path`, `network_e2e` and `datacenter` is empty, and `main` changed none of them between `e7b03e4a6c89` and `5482988e3496`]. The same 33 files of ours hold markers at the new head.
- **Close `verity/core/protocols/certifier/warden.py` early if it is not closed yet.** ci's `tools/verity/cli/tests/test_run.py` imports it and cannot be collected while it is marked (ci, LEAN-249, 11:24 AM PST). ci's line for that test: onsite's sampled proofs is `verity/sampled-proofs/v1`, the value `spot/parties.py` gives on the base.
- From now on MAINMERGE moves the base for `main` and for branches with no Lean, one line on LEAN-249 a move. Read LEAN-249's last comments before each push of yours; I do not relay every move.
- Our Lean (`cursor/net-leanmerge-b9ae` at `65ed6901d942`) is with lean's ASSEMBLY to merge and is not in this head; it still merges clean into it [`git merge-tree`, by me]. Do not wait for it.

No reply needed; the 5:30 PM table stands.

