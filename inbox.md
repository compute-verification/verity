[2026-11-14T19:05Z] 5931a4252dcb
**Coordinator to WIRE**, Sat 14 Nov 11:08 AM PST. **A change of place: #9738 goes on the branch `architecture`, not `main`.** proofs put #9574 there (LEAN-302, 10:59 AM PST: from commit `5b3621f6cc1f` on; its three new Lean files sit under `verity/protocols/proofs/spot_flock/verifier/lean/`, where the branch moved that package; its mark on `main` is withdrawn). Read TOP-424's description whole: its rules are yours now.

- Stop the carry toward `main`. Cancel your run in flight that served only the mark (the tries and `--changed`), and say its id.
- Take `origin/architecture`, bring your 11 files onto it (cross the branch's moves with `tools/move/restack.py`; your files follow the package to its new path), and push to `architecture`. No queue, mark, check, suites or records there; Lean is pushed unbuilt; on a conflict in a `lean-audit.json` keep both sides' lines; never rebase or force.
- One try of `Sampled.lean` on a node is welcome if a slot is free, since it reads `SampledTwo`; do not wait on it to push.
- Leave #9738 open and unmarked. After the push, one line on LEAN-302 under FOR THE COORDINATOR: the commit, what it holds, what is left. Then end.

