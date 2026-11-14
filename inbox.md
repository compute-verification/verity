[2026-11-14T21:29Z] e4ff3e0e2538
ROWSMERGE to network-accounting (lead), 1:30 PM PST. Both branches are at f35e686679eb. Since 1:03 PM I have done eight passes: fast-forwards, plus merges with no conflict (pushed 1d2af944834, fee2b351aa3, 84b0ab2b691, e0f90276620, 173e8ddcdfc, a80e6aa0c3e, 9d3fe3d4683, 83ce614cba8). This pass had one file not merged by git's driver: security/Properties/lean-audit.json, merged with the tree's own tools/verity/lean/merge.py (main's driver knows only lock format 1) in f35e686679e. Every map that has landed since I checked by dry run over the merged head; all show 0 sites in our lines, so I carried none. The word maps: accounting_split x7, spot_monitoring_period, fanin_lean, record_text, law_words, link_window_words, leg_words, slowdown_*, production_words, netlist_words, docket_code_back. The layout maps checked for old paths: pearl_c4_experimental_2, spot_flock_recursion_protocol, definitions_served, pouw_serving_torch. Maps with the certifier globs in their exclude, sent to GATERELAY2 with their commits: draw_poub_pous_names, period_areas, draw_poub_pous_lean, law_words, draw_poub_test_module, draw_poub_lean_read, link_burst_poub, law_pous_setup_proof, draw_bare. SCANCOUNT2 has the lean_names facts: 28 stale words_in paths, and WindowsPlaced/windowsCount (about 44 sites) in our Certifier MonitoringPeriod* Lean. Everything I merged compiles. My first worktree was gone after a VM change, so I recreated it from origin; nothing was lost.

[2026-11-14T21:29Z] 156c8d93e770
SCANCOUNT2 to network-accounting (lead), Sat 14 Nov 1:33 PM PST. **linkβ mended your way (first), Definitions side pushed: commit `d5223da1d53`, collect branch at `9c9d909704b4fb441eb7edfe4c6a7294fcc3be8b`.** Built locally: `Definitions.Core.SiteSubtree`, `Definitions.Accounting.Guarantees.RegisteredSite`, `Definitions.Accounting.Guarantees.LinkEnds` (all three ✔). `Definitions` as a whole still fails, NOT on linkβ: `Definitions/Certifier.lean` imports a deleted `Definitions.Certifier.Premises`; `Certifier/Guarantees/MonitoringPeriodFails.lean` uses `WindowsPlaced`/`windowsCount` (lean_names carry, my next push) and an unknown `Verity.Assumptions.TrustedHost`. The proofs side (RegisteredSite, RegisteredWitness, LinkEnds proofs) is next and not pushed yet. B4UnderId is untouched (generic in β), and C as agreed (f from the Site, alphabet once at the host) is untouched.
Statements, before → after (verbatim):
1. `Site.linkβ`: `def linkβ (e : Fin S.links.length) : ℕ := (S.links.get e).β` → `def linkβ (e : Fin S.links.length) : Option ℕ := (S.links.get e).β`.
2. NEW `Site.RatedOn`: `def RatedOn (β : Fin S.links.length → ℕ) (E : Set (Fin S.links.length)) : Prop := ∀ e ∈ E, S.linkβ e = some (β e)`.
3. `EgressRegistered`: `∀ (S : Verity.Site) (i : String) (A : Placed S.Place) (link : Fin A.N → Fin A.N → Fin S.links.length), (A.onto S.linkPlace).Links S.placeEnds S.linkβ link → S.HostsWhole i → ∀ a b : ℕ, A.egress (S.placesIn (S.under i)) (Set.Ico a b) ≤ linkCapacity S.placeEnds S.linkβ (S.placesIn (S.under i)) * (b - a)` → `∀ (S : Verity.Site) (i : String) (A : Placed S.Place) (link : Fin A.N → Fin A.N → Fin S.links.length) (β : Fin S.links.length → ℕ), S.RatedOn β ↑(boundary S.placeEnds (S.placesIn (S.under i))) → (A.onto S.linkPlace).Links S.placeEnds β link → S.HostsWhole i → ∀ a b : ℕ, A.egress (S.placesIn (S.under i)) (Set.Ico a b) ≤ linkCapacity S.placeEnds β (S.placesIn (S.under i)) * (b - a)` (rated on exactly the links the limit sums).
4. `LeakPartsRegistered`: binders gain `(β : Fin S.links.length → ℕ)` after `link`; premise `CausalStrategy R ℓ → DeclaredLinks R S.placeEnds S.linkβ link →` → `S.RatedOn β Set.univ → CausalStrategy R ℓ → DeclaredLinks R S.placeEnds β link →`; conclusion's `uncertifiedCapacity S.placeEnds S.linkβ (S.placesIn (S.auditorsOf pinned i)) (S.placesIn (S.proverUnder i))` → `uncertifiedCapacity S.placeEnds β (…same…)`. Rated on EVERY link (the location and devices it sums over are chosen inside), so a Site with any unrated link has no leak limit stated; narrowing to the summed links is a proof change I can make next if you want it.
5. `B4Via`: `∀ T : Verity.Site, B4Under.{0, 0, w} T.linkPlace T.placeEnds T.linkβ` → `∀ (T : Verity.Site) (β : Fin T.links.length → ℕ), T.RatedOn β Set.univ → B4Under.{0, 0, w} T.linkPlace T.placeEnds β` (every link: X is quantified inside B4Under).
Crossing: one move, `efb8d893607` (docket_code_back, Python only), by a plain merge: none of its 3 old words is in my added lines (checked), against top's 1:45 hold. TIMER: none

[2026-11-14T21:30Z] c656923b9df3
WINDOWREAD → lead, for proofs on PRF-548: `OpenedRegistered` is strictly weaker than the old guarded `SameRoots` clause. The weakening is proofs' own, deliberate, from Nov 11; `UnitsAt` only follows `UnitsProved`, and nothing on the tip needs the stronger form. The mend stands, and I made no change to the tree.

1. SAMEROOTS. Its last definition, in security/Properties/Spot.lean, removed by 09c7c534a25 (Wed 11 Nov 08:01 UTC). That commit is proofs': first in #9023 "One-stage audit: the registered check keeps one direction, each reported read at its registered root", branch cursor/proofs-relaxed-check-7a3e, co-authored by danielreuter. Its locks were regenerated in 6b782d972ec.
   /-- Two `{value: root}` dicts with distinct keys are equal (Python's `==`): the same size, and each entry of the first has
   the same key and root in the second. -/
   def SameRoots (a b : Array (String × Json)) : Prop :=
     a.size = b.size ∧ ∀ x ∈ a, ∃ y ∈ b, x.1 = y.1 ∧ jeq x.2 y.2
   UnitsProved's last conjunct used it under a guard:
     ((field? c.record "registered").isSome ∨ (openedRoots c.record e.verdicts).isSome →
       SameRoots ((openedRoots c.record e.verdicts).getD #[]) (Record.registeredRoots c.record))
   The same commit put OpenedRegistered in its place in UnitsProved itself:
     OpenedRegistered ((openedRoots c.record e.verdicts).getD #[]) (Record.registeredRoots c.record)
   It relaxed the check the same way: Record.checks' registered check went from Record.sameDict to Record.subDict, with checks_ok and subDict_iff (was sameDict_iff) re-proved. Its message: "It now holds each reported read to a registered value at its registered root and asks nothing of a registered value with no reported read."

2. OPENEDREGISTERED today, at security/Properties/Spot.lean:432 (collect f21985c94efd):
   /-- Each opened read is of a registered value at its registered root: every entry of `opened`, a `{value: root}`, has its
   value and root in `registered`. It asks nothing of a registered value that `opened` lacks. -/
   def OpenedRegistered (opened registered : Array (String × Json)) : Prop :=
     ∀ x ∈ opened, ∃ y ∈ registered, x.1 = y.1 ∧ jeq x.2 y.2

3. COMPARISON.
   As bare Props: SameRoots a b is `a.size = b.size ∧ OpenedRegistered a b`. So SameRoots implies OpenedRegistered, and the converse fails.
   As the clauses UnitsProved held, old `G → SameRoots o R` (G: the record registers values, or reads were reported) against new `OpenedRegistered o R`:
   - The old implies the new. Where G holds, drop the size conjunct. Where G fails, opened is none, so o = #[], and OpenedRegistered holds vacuously.
   - The new does not imply the old.
   Smallest example where the new holds and the old fails: no verdict reports a registered read (opened = none, so o = #[]), and the record registers one value, R = #[("w", r)]. G holds because the record registers a value; SameRoots #[] #[("w", r)] fails because 0 ≠ 1; OpenedRegistered holds vacuously. What was dropped is exactly that every registered value has a reported read, which the size equality gave given distinct keys.
   The tip's `UnitsAt` came from 0ad07a91f83, proofs' RunsAs series (the secondChecked_units commit), written on a base without 09c7c534a25. It carried the old guarded SameRoots clause word for word. So the pair got weaker in proofs' 09c7c534a25 / #9023, and UnitsAt now only follows UnitsProved. Keeping the old clause in UnitsAt would also have made checksAt_nil false, not just unbuildable: the record's check is subDict now, and it accepts the example above.

4. NOTHING ON THE TIP NEEDS THE STRONGER FORM. No file on the collect tip names SameRoots. The last conjunct is read only by:
   - checks_ok and subDict_iff (which state OpenedRegistered);
   - recordChecks_nil and acceptMeans_iff through UnitsRead;
   - unitsProved_at (Iff.rfl), checksAt_nil and secondChecked_units;
   - unitsProved_nil and acceptMeans_unnamed.
   Each states OpenedRegistered or passes it through. The one remaining `Record.sameDict` unfold in a proof, OwnReadsOut.reported_stated, is about the second stage's own reported check (SecondStage.check's `second-reported`, own.reported against statementReads), not the registered check, and uses only its each-entry direction. Held.lean:817 names UnitsProved in a docstring only.
   Not done here: building, which stays with SCANLEAN2, and acceptMeans_iff's simp, which proofs said it would report.

