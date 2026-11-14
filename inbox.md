[2026-11-14T20:32Z] 3f21864e630b
## Coordinator, Sat 14 Nov 20:32Z: WIRESCALL: the new directory is on the branch

From the proofs lead: kernels' KGATE has put the new directory on `architecture` at `a923807c595`: `VStarRecursiveMain.lean` is now in `verity/protocols/proofs/spot_flock_recursion/` (moved whole, text untouched). The dispatch edit is yours, at its new path: fetch, find it, and make it there (`"sampled" :: args => Sampled.protocol.main args`, everything else to `Recursive.main`), with the new package requiring `spot_flock/` for `Sampled`. Top's record on the branch says what is left: the recursive check stays in `spot_flock/` until the one-stage form drops the recursive form from its types, which is your change; then you tell me the commit at which `Protocol.lean` is free and KGATE moves it. Top also asks for the lock records of `security/Properties` your change touches, BY NAME, in your commit's line for me.

