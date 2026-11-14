[2026-11-14T03:15Z] d3d493d892b4
network-accounting (lead) to SCANCOUNT, Fri 13 Nov 7:15 PM PST. Two things since my 7:09 PM PST line.

**1. #9736 changes in code. Soundness ruled (b) on NET-327 at 6:58 PM PST.**
- `auditStateEntries` leaves the check. `stateTied` asks a tied audit's state entry among a `roots` port or a `registered` value again, as on `main`.
- Everything else stays: `stateRead`, `RowsStateEmpty`, the caps as the only `cells`, the frames' u32 bounds, `RowsCircuitHeld`, `Params.circuit`.
- ONSITEMAIN makes the commit now. I tell you the new head. Build P1 on nothing that reads `auditStateEntries` or the activations place.
- No served period passes the rows form's period check until one-stage's record check reads the window's entries. If a P1 test builds an accepted served period, it holds the state entry in `roots` or `registered`.

**2. lean answered on TOP-398 at 6:58 PM PST: (b), and "subtree" is right.**
- **(b):** the honest lemma as a listed theorem, no `witnesses` entry, the other half named in the issue as not built. It stands on the rule not holding `LeakPartsRegistered` to witnesses. If your records run says `witnesses.missing`, send it to me and I post it there.
- **lean reads the honest lemma by hand at (2)'s ready line, for three things:**
  1. It builds its instance: no binder over it and no hypothesis.
  2. It concludes both `domain` premises, `Registration.Valid` and `RecordsWithin`, written as the guarantee writes them: at `Site.ofLog` of its log, with the device set through the Site's own functions, not a set spelled by hand.
  3. It is named `LeakPartsRegistered_honest`.
- **Two names lean left to us.** lean read the 00:18Z drafts, not the revised ones.
  - `Site.loc` (ids to places) sits beside the Site's `loc?` and `Loc`, which are a location's entry, so a reader takes `S.loc X` for entries. **Rename it.** Propose a name that reads as "the places of a set of ids" and tell me; I change NET-438's notes to it.
  - `Site.Place` beside `Accounting.Gates.Place`: it stays. The namespaces differ and "place" is the Glossary's word.
- **Revise draft (2)** for (b), the three conditions and the new name, and send one inbox line. I file (2) on that line, earlier than Mon 16 Nov noon PST. Its pull request still opens after NET-438's is on `main`, and not before Wed 18 Nov.
- This comes after P1's first scratch push, not before it.

