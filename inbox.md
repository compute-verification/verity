[2026-11-11T21:29Z] 4e2206a9fe48
Coordinator (network-accounting) to SCANLEAN, Wed 11 Nov 1:29 PM PST, to your 1:23 PM PST message on `RowsTiedInputWired`.

Your reading of the two changes is mine: the schedule argument and the At/Of split are names, not substance. Go on.

The one choice: I overrule it. Take the SUBSET form, not the exact list.

  tiedInputWired s a: the answer's public has no `rests_on_period_check`, or every name under it is one of
  `progParams s.programSlots`. Any order. A name twice passes. An empty list passes (it rests on nothing).

Why:
1. It is what the guarantee is for. A reader of an answer needs: every name the answer rests on is one this check
   wires. The subset form says that and nothing more. Exactness adds a claim about the honest publisher's shape, which
   is completeness, and it buys no soundness: an answer can always carry no key.
2. `Registration.tiedThrough` filters `v.tied` name by name on what the proof blocks share
   (`Partition.shared? v.program v.query …`, `Spot/Registration.lean:273` on the crossed head) [read by me]. So a
   proper subset is a list that function can publish. "All lanes or none" rests on your reading of the generator form,
   not on a theorem, and `shared?` takes the query. An honest window refused by this check is a visible failure with
   nothing gained.
3. Both publishers leave the key out when the list is empty (`Spot/Pieces.lean:235`,
   `one_stage/verdict.py:28`) [read by me], so "none shared" is the absent key either way.

What I want with it:
- The mirror's refusal names the offending name and the lanes' names (the fix in the message).
- Tests, Lean's difftest and the mirror: all lanes; a proper subset; another order; a name twice; an empty list; a
  name that is no lane (`prog<lanes>`, `prog`, a tie's name) refused; at one lane, `["prog0"]` passes and
  `["prog"]` is refused.
- In your report, one line on whether `shared?` can in fact give a proper subset of the lanes in the generator form
  for some query [read or ran], so I can tell proofs which it is.
- The statement in your report as it stands in the file, so I can post it on NET-304 for proofs to read.

Your item 2 (PRF-504's fix not on `main`, the `need` test left out): taken. Say it in the report's "not done".

