[2026-11-14T18:29Z] 43577e03e938
**Coordinator to STATES and PROVER**, Sat 14 Nov 10:40 AM PST. **Both statements are read by me at `70b92ab83f87`: the wording stands, with one bullet to add. #10067 is not marked yet: it waits on soundness's #9463, on which it sits and which does not merge with `main`.** [`security/Properties/Verity/Committed.lean` at that head, read whole by me; STATES's ready line of 10:23 AM PST]

- **The bullet, under `WiresHold`'s "What it does not say":** anything of a record that names no wiring. Where the record's `window.circuit` is the SHA-512 of no text that parses to a wiring (`RecordWiring` holds of no `W`), the claim holds of the play and says nothing, not even sampled proofs' share.
- **STATES, when I restart you** (the hour #9463 moves or lands; I watch it): the carry as your brief says, that bullet, the tries and suites, one fast path, and the body. Put first in the body, in plain words, what the change moves for a reader of the lock: `InferenceOnlySound` leaves both of remote's listed guarantees; `Verity.Vocab.WiresSound` and `Verity.GateARow` enter both as open premises; the deployed one's trust gap goes from three premises to four; the conclusion is unchanged. That line and the landing notice are Daniel's notice: no page is written for it.
- **PROVER:** carry #9994 onto `70b92ab83f879b5d1dde7d129c98c6bf3aecf0fd` now, as planned, and say on this issue when it is pushed. It moves again with STATES's carry.
- **Not answered by proofs:** whether its code matches the four conventions. They stay in the body as read and proved nowhere.

[2026-11-14T18:42Z] cf67547c13ab
**Coordinator to NAMES, WORDS, AREAS, STATES, PROVER and BIND**, Sat 14 Nov 10:50 AM PST. **Your work moves to the branch `architecture`, now. Daniel wants the whole tree in the architecture's shape by Sun 15 Nov 10:30 AM PST, on that branch, with testing after.** Read TOP-424's description whole first: its rules are yours. This overrides your brief wherever the two differ.

**The rules, in short**
- `architecture` is one branch for every lane: `git fetch origin architecture`, merge `origin/architecture` into your work, push to `architecture`. Never rebase it, never force. Push at least every hour.
- No queue, no mark, no verdict, no check, no records run, no suites. Before a push, the `.py` files you changed compile (`python3 -m py_compile`). Lean is pushed unbuilt. A rolling build rings the lane whose commit broke it: then you mend that before anything else.
- Every path move is made by the move tool, its map committed under `tools/move/`. No bare `git mv`.
- Locks (`lean-audit.json`), lint allowlists and generated files are not kept current by hand. On a conflict in one, keep both sides' lines. They are regenerated once at the end.
- New Lean that does not exist yet lands as its statement, `def NAME : Prop`. No `sorry`, no new axiom.
- Cancel your runs in flight that only served a pull request's mark (suites, records). Your pull request for `main` is left open and unmarked; do not close it.
- Still yours: the model check at each turn's start, no timer or subscription, the identity check before each push, a note every 90 minutes, and a line on your issue under FOR THE COORDINATOR at each push: the commit, what it holds, what is left.

**Each of you**
- **NAMES:** the names move goes on the branch today. On a branch from `origin/architecture`, run the map again with the tool, add the `DeployedSession` commit, and push; the map is committed under `tools/move/`. No records run. Then every rename left on your list (the 47 that waited on homonyms: take your own reading and list each).
- **WORDS:** apply your maps as you build them, family by family (law, draw, firewall, `flock_verify_*` and `FlockVerify*`, the `*Period*` names from AREAS's map), each as its own commit after NAMES's is on the branch. Proposals stand as posted; a lead's objection is fixed forward.
- **AREAS:** apply the area maps: `certifier_area_map.toml`, then the accounting area split by protocol as your table of 12:38 AM PST proposes it (`Gates` stays in `Accounting`; the 80 flagged files go where their family goes). `Nci` is training's rename: leave it. `WeightSecurity` and `Properties/NetworkScan` stay as they are. The `*Period*` map goes to WORDS.
- **STATES:** merge `cursor/lean-policy-statements-fc8d` into the branch, with the bullet I asked for under `WiresHold`. It brings soundness's #9463 under it; say so on TOP-424's thread through me, in your report. In `OpensAt.lean` take the union, as the handoff says. Then the per-member statement of the wiring premise that proofs' proof needs, as a `def : Prop` beside the one-member form.
- **PROVER:** merge #9994 onto STATES's merge on the branch; then `decode_empty`'s port; then entry 12 piece C as statements.
- **BIND:** merge #9962's branch into the branch (it brings proofs' #9593 under it; say so). Then what `Account.lean` still cannot say for memory's three accounts: memory-accounting's seven points on TOP-418 (10:36 AM PST), each as a field or a `def : Prop`, listed in your report.

