[2026-11-10T01:10Z] 7b9f5af823ac
From network-accounting, Mon 5:10 PM PST. MEM-226 is answered: No. Do item 4's "no" branch in this pass.

- memory-accounting: PoUB's window is not an execution phase (nothing is served in it; it is 60 s, ten to a session). `RemoteAudit.Verifier.windowCheck` keeps its name.
- Scope `windowCheck` so it renames nothing under `verity/core/protocols/remote_bandwidth/`, the same way as PoUW's `windowOf`. Put its four sites back (Verifier.lean: the definition, its call, the docstring; Wire.lean: one docstring).
- The map's header: memory-accounting leaves the list of other areas too. What is left there is `windowsOn` (soundness) and `windowOf` in `FlockSoundness.KeyedDraw` (proofs).
- The same two tests cover it: fresh stage of the corrected map equals your new head, and nothing under `verity/core/protocols/remote_bandwidth/` differs from the staging's base.
- Before you push, grep the move's diff for any other file under `verity/core/protocols/remote_bandwidth/`, `verity/core/protocols/poub/` or Security's PoUB areas that our map's words touch. If there is one, list it for me and don't decide it.
- Your `@lean` comment on NET-118 names both scopings.

