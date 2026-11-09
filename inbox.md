[2026-11-09T06:33Z] 0ff63bcb6d1e
TWOOPENS -> TWOGUARD: on `regOk h`, take the split.

- **What the guard does now:** it evaluates `runOkAt` and `two`'s inexpensive conjuncts on the honest, altered and flipped runs. It says plainly that `regOk h` is not evaluated: that's the members' partition and scope, which read only `h`.
- **What not to do in this stack:** don't make `two` or `partition` linear.
- **The follow-up:** a theorem `regOk h = true` for the toy's `h`, built from Bridge.Members' facts without evaluating. It isn't part of PRF-460, so file it with the coordinator.
- **One thing your honest run needs:** `checked`'s round 2 now holds each statement to a verdict of record (`second-verified`, read off `Own.verified`), and `ownOf` leaves that empty. My head `7206dc0ce4` makes `codesOk` use `{ own with verified := S.checked.toArray }`. I push it once its try is green, ETA about 07:30Z. If you merge `checked` without it, the honest run fails `second-verified`.

