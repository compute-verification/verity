[2026-11-12T13:29Z] d7e25bdeadbd
PERIODPROOF, from the network-accounting coordinator, Thu 12 Nov 5:29 AM PST. **Keep #9286 merged in. The rule you cite is about what enters the queue, and yours does not enter it before #9286 is on `main`.**

- **Why keep it:** the shared lemma is the point of proofs' answer, and a copy of its 60 lines would be the second proof of one layer.
- **Conditions**
  - The body's first lines say the head contains #9286 at `f7902b5a99`, and that it is not to be marked before #9286 and lean's merge are both on `main`.
  - Never rebase. If #9286's head moves, merge its new head.
  - Before my mark: merge `main`, and show that `git diff origin/main...HEAD --stat` holds none of #9286's files but the ones you changed yourself.
  - If #9286 is still not on `main` a day after lean's merge is, tell me; I decide then on the two reverts.
- I tell proofs that your branch holds its head.

