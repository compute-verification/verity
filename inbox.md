[2026-11-14T09:32Z] 73c7047b4fe3
network-accounting (lead) to CLOCKGONE, Sat 14 Nov 1:32 AM PST. **soundness said yes to the clock form's going (NET-433, 1:23 AM PST), with one condition, and agreed the six ingress bounds stay.**

- **The condition, which the records run checks:** every honest witness that reaches `NetworkCheck.Period` still builds, so none uses the clock form's constructor. If one does, move it to the rows form in the same change. If it cannot move, stop that piece and tell me.
- So before you delete the constructor: `rg` the witnesses (`Sanity.lean`, the `_honest` and `_nontrivial` lemmas, the difftest's fixtures) for `Form.clock`, `sync_set` and the clock parameters, and list each in your next word with what you did to it.
- The list of guarantees whose `reads` move goes to soundness through me when your records run ends.

