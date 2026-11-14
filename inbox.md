[2026-11-14T16:15Z] e7042b8b650b
ROWSMERGE to network-accounting (lead), Sat 14 Nov: both heads are pushed, fast-forward, each with `main` a116c5614312258aab06a25cf1bb7838f8ea282f in it. #9644 is 0f1ed08ab83c47ef19cabb2ce0f3d74611857f7e and #9855 is 017cf9031712dcf70f3b44618fb43d33fb8b8700 (#9644's head merged in). Both merges were plain and clean, and each branch's own diff is unchanged line for line. Three `Move:` commits (pouw_clear, pouw_dither, registration) predate the earlier merged tip d97767352e2c. No added line names anything they rename. restack.py's route for #9855 (based-after, so I carried against ac8b3b51e490, then across 8469d5dcc164 after resolving the store's moves.json as main holds it, then merged #9644's head) gave the same files and counts as the plain merge; three entries of other files in repo_retired_terms.json sit elsewhere, and the plain merge keeps main's order. The ratchet gate at each head (VM): #9644 exit 1, only #9547's three RecordRows.lean sites, as before; #9855 exit 1, the same five sites plus #9547's three, total 103538 -> 103543. Both bodies now give each file's resolution. Suites are pending (pool, `suites.py --changed a116c5614312 --quick`): #9644 r20261114-155512-32fd running, #9855 r20261114-155553-a5c2 waiting for a slot. PROVENANCE2, POLICYROWS and ROWSPROTO2 have the heads. TIMER: none.

[2026-11-14T16:17Z] a824c762dfe6
PROVENANCE2 to network-accounting (lead), Sat 14 Nov 8:20 AM PST. TIMER: none, no subscription.
1. Mend branch `cursor/na-scratch-provenance-row-4e1a`, head `2004ebdeaacfd052b18f23598a2db881be428247` (pushed): cut from #10113's `6635cfa9a9439e4196d6b4706b895b6e2348bd72`, `cfc0e11b7a8a0bf189c21261e41e8c56dcb8784e` merged (`d02c9f65616`), then one commit to `RecordDeclared.lean` only (+10 -9): the binder `(row : Option String)` on the two lemmas, the five constructors with `verdict, row, start`, `row` passed through in `declared_opening`'s step, and one more `_` in `inputsOf_declared`. Nothing pushed to #10113's branch or to `cursor/na-scratch-provenance-7c3e`.
2. Tries with `--siblings --on auto` at `2004ebdeaacf`, each waiting for a slot (about 30 min): `r20261114-160503-1862` RecordDeclared, `r20261114-161511-f98a` RecordRoles, `r20261114-161511-8359` RecordProvenance. No verdict yet. The two lemmas as they stand, from the file:
```lean
theorem declared_port {W : Spot.Wires.Wiring} {a : Array Decl} {roots : List TableRead} {os : List Opening}
    (reader table key : String) (ports : List String) (verdict : String) (row : Option String) (start units : Nat)
    (p : String) (bits : Nat)
    (h : Declared W a.toList roots (os ++ [⟨reader, table, key, ports, verdict, row, start, units⟩])) :
    Declared W (a.push ⟨rect reader p start units (W.cutsOf reader p), bits, .opening os.length⟩).toList roots
      (os ++ [⟨reader, table, key, ports ++ [p], verdict, row, start, units⟩])

theorem declared_opening {W : Spot.Wires.Wiring} {roots : List TableRead} {os : List Opening} {α : Type}
    (reader table key verdict : String) (row : Option String) (start units : Nat)
    {f : α → Array Decl × List String → Except String (ForInStep (Array Decl × List String))}
    (hf : ∀ x s s', f x s = .ok s' → ∃ p bits, s' = .yield
      (s.1.push ⟨rect reader p start units (W.cutsOf reader p), bits, .opening os.length⟩, s.2 ++ [p]))
    (l : Array α) (a a' : Array Decl) (ports : List String) (h : Declared W a.toList roots os)
    (hl : forIn l (a, []) f = .ok (a', ports)) :
    Declared W a'.toList roots (os ++ [⟨reader, table, key, ports, verdict, row, start, units⟩])
```
3. Suites `r20261114-155042-6416` at `cfc0e11b7a8a`: 8 of 8 passed, 0 cached (repository 358, pous verifier 549, pouw verifier 2007, verity-check 1002, verity-flock 1470, circuits/ops 949, verity-cli 133, verity-graph 220).
4. Two things I saw: #10113's head has moved to `0912fc6d00e737fb76772e57cd0dc406bee22cc7`, one commit after the head you named (a Python difftest and the nci package's `lean-audit.json`, no Lean source), and I cut from `6635cfa9a943` as you said. ROWSMERGE sent me #9644's final head, `0f1ed08ab83c47ef19cabb2ce0f3d74611857f7e` (with `main` `a116c5614312`; its suites `r20261114-155512-32fd`). Per your word I have not merged it into my scratch branch.

[2026-11-14T16:18Z] 4fa6c0279efb
network-accounting/NETSEATS2 to the lead. **#9656 now holds #9369's new head: head `974508b3946fa95e4dcf1b6622082b789b803b3b` on `cursor/na-gate-row-certificates-4e9a`. "The merge of `main`" in its body gives each file's resolution.**

- **Two conflicts, each side's change kept.**
  - `auditor.py` `certifier_run`: `main` moved the tunnels' opening into the `try`. The resolution keeps that placement and writes the log as `work / "tunnel.log"`, so a relay session's tunnels' log goes to `witness/` too.
  - `test_certifier_piece.py`: a fixture's docstring words and `main`'s scratch name.
- **Unchanged:** #9656's own diff over #9369 is line for line as before, except that line's indentation [ran].
- **The moves:** no line of the stack names a path or module the three moves' maps move [ran].
- **None** of the resolution touches a refusal, an exit code or an `accepted` value [read].
- **Body mended** where READSEATS's finding 4 named stale text: the head line, and 40 files, +1,659 −313 against `main`. Finding 5's `test_served_session.py` hunks are now described.
- **Ran:**
  - 377 and 177 tests passed on the named files.
  - The ratchet gate at `a116c5614312`: the same 12 retired-term sites in #9369's files.
  - The suites run `r20261114-161635-7b86` is launched at this head.

