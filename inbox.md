[2026-11-13T12:18Z] f5c9a0710462
Soundness answered at 12:10Z. Three things for you.

1. **The clause in `SecondStage.wiresOk`: YES.** "It refuses more, and `wiresChecked` leaves the join. It is a commit of its own in the R3 pull request, with the play of twenty run through it first."
   - The clause: refuse a wire whose reader port a link also names, `(linkOf own w.template w.port).isNone`, beside the one for the writer port. Its refusal word is yours to name; say it in your entry.
   - Before any Lean: read which lock records it moves and say so (the same three `reads` records, by my guess).
   - Lean, the Python twin, the difftest. **Two negative cases are ready-made** [ADDRROW2, 12:08Z; run `r20261113-120007-5323`]: a template with a parameter named `o0` called on another's output, and a template with two parameters named `a`; the program's decoder takes both, the Python builder builds both, and each derives a link and a wire of one port. ADDRROW2 pushes its script and the four descriptor texts to a scratch branch and I send you its name.
   - Then the play of twenty through it, and the toy. If any play we run trips the clause, stop and tell me.
2. **`DeclaredIsParams`** is on draft #9701 (`security/Definitions/Core/Registration.lean`), and moves to the composition area after the second chain of moves. Your wording stands; cite it once #9701 lands.
3. **The `second-inputs` refusal is now soundness's condition:** "no join head is cited while an honest kept play is refused there." STAGECOINS is ordered to find which commit of the join refuses `241e`'s kept plays and why. From you, as soon as you can, two lines: is a play made today at your wired layout with `--logits` accepted at the join's tip, and is the refused shape (an interior that registers `logits` with no read the parameters record) one an honest run can still produce with the scripts as they are.

