module

public import Proofs.Properties.Verity.SampledTwo.StmtsRound
public import Proofs.Properties.Verity.SampledTwo.AcceptsTop

/-!
# The statements' count

`StmtsCount` (`StmtsRound.lean`) says an accepted play's evidence holds one statement a template of the held copies.
Here it is a theorem where the held copies are the ones `heldOf` builds from the play's run (`HeldOfAt`) and their
parts are the interior program's strata's (`PartsHeldAt`), the two hypotheses the play's acceptance already rests on
(`playAccepts_form`).

* **The run reads the evidence's statements** (`stmts_of_accepts`): the run of record's input on an accepted case has
  an interior whose statements are the evidence's (`VStar.Recursive.SampledTwo.inputOf`).
* **The run's second-stage check is at the held copies' parts** (`SpotFlock.RunsAs.parts_of_accepted`): an accepted
  two-stage run whose held copies agree with what it reads (`RunsAs.HeldMatches`) passed round 2's statement check
  at parts that are the held copies' (`SpotFlock.partsAt`). The replay sample the parts are sized by is the first
  stage's select on the run's own bytes, so no coin is read.
* **A passing round 2 reads one statement a template** (`SpotFlock.RunsAs.count_of_parts`): `Round2.count_of_check`
  with the parts given rather than read off a round-0 coin. Each statement is for the part at its position, so
  there are at most as many as parts; every part has a statement and the templates' names are distinct, so there are
  at least as many.
* **`StmtsCount`** (`stmtsCount_of`), from the three, with the names distinct where the design evaluates
  (`twoHolds_at`).

**Not established here**: `PartsHeldAt`, a hypothesis.
-/

@[expose] public section

namespace SpotFlock.RunsAs

open Lean (Json)

/-- **A passing round 2 reads one statement a template, at the held copies' parts**: where the parts round 2 reads
the statements against are `partsAt h q` and the templates' names are distinct, the statements are as many as the
templates. It is `Round2.count_of_check` with the parts given. -/
theorem count_of_parts {h : Held} {own : Spot.SecondStage.Own} {ss : Array Json} {q : ℕ}
    (hp : own.parts = some (partsAt h q)) (hN : Round2.NamesDistinct h)
    (hc : (Spot.SecondStage.check true own (some ss)).codes = #[]) : ss.size = h.ms.length := by
  obtain ⟨-, -, ps, hp', -, -, hst, hcov, -⟩ := Spot.SecondStage.check_ok hc
  obtain rfl : ps = partsAt h q := Option.some.inj (hp'.symm.trans hp)
  apply Nat.le_antisymm
  · by_contra hlt
    obtain ⟨s, -, -, -, -, hi, -⟩ := hst h.ms.length (by omega)
    exact Nat.lt_irrefl _ (Round2.partsAt_template hi).1
  · have hsub : ((List.range h.ms.length).map fun t => h.names.getD t "") ⊆
        Spot.SecondStage.templates ss.toList := by
      intro x hx
      obtain ⟨t, htr, rfl⟩ := List.mem_map.1 hx
      have hmemp : partAt h q t ∈ (partsAt h q).toList := List.mem_map.2 ⟨t, htr, rfl⟩
      exact List.contains_iff_mem.1 (hcov _ hmemp)
    have hle := (List.Nodup.subperm hN hsub).length_le
    rw [List.length_map, List.length_range] at hle
    exact hle.trans ((List.length_filterMap_le _ _).trans (by simp))

/-- **An accepted two-stage run's second-stage check is at the held copies' parts**: where the held copies agree
with what the run reads (`HeldMatches`), round 2's statement check reported no code at an `Own` whose parts are the
held copies' at some replay sample's size. The size is the first stage's select's on the run's bytes
(`accepted_steps`, `proved_select`). -/
theorem parts_of_accepted {i : Spot.Run.Input} {it : Spot.Run.Interior} {j : Json} {h : Held}
    (ha : AcceptedTwo i it j) (hm : HeldMatches i it h) :
    ∃ own q, own.parts = some (partsAt h q) ∧ (Spot.SecondStage.check true own it.statements).codes = #[] := by
  obtain ⟨hcl, workTable, resolved, draw, rest, read, ht, hres, hsel, -, grain, replay, k, ns, ids, ks, draw₂, read₂,
    strata, hrule, -, -, -, -, -, hstrata, hcodes⟩ := accepted_steps ha
  obtain ⟨n, rule, law, workTable', resolved', -, -, -, ht', hres', hreg⟩ := hm.first
  have hwt : workTable' = workTable := Except.ok.inj (ht'.symm.trans ht)
  have hrs : resolved' = resolved := Except.ok.inj (hres'.symm.trans hres)
  subst hwt hrs
  have hsel' : Spot.Select.select h.reg.first.rule h.reg.first.program h.reg.first.query h.reg.first.table
      h.reg.first.closure i.source = .ok (draw, rest, read) := by
    rw [hreg]
    exact hsel
  have hsel0 := hsel
  rw [hcl] at hsel0
  have hproved : (Spot.Draw.proved draw).map Int.toNat = rest := proved_select (resolve_ne_all hrule hres) hsel0
  have hpa := hm.parts draw rest read strata hsel' hstrata
  rw [← hproved] at hpa
  exact ⟨runOwn i it ((Spot.Draw.proved draw).map Int.toNat) strata draw₂, _, hpa, hcodes⟩

end SpotFlock.RunsAs

namespace Verity.SampledTwo

open Claims

/-- **An accepted case's run of record reads the evidence's statements**: the input the check hands `Spot.Run.run`
has an interior, and the interior's statements are the evidence's (`stmts`), the ones the statements' message has one
entry of each. -/
theorem stmts_of_accepts {k : Case} {p : VStar.Recursive.SampledTwo.Params}
    {e : VStar.Recursive.SampledTwo.Evidence} (h : AcceptsOn k) (hf : SessionMsgs.formOf k = some (p, e)) :
    ∃ i it, inputOfCase k = some i ∧ i.interior = some it ∧
      it.statements = some (VStar.Recursive.SampledTwo.stmtsOwn p e k.blobs) := by
  obtain ⟨p', cm, e', pub, prof, hdec, hacc⟩ := h
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj (hf.symm.trans (SessionMsgs.formOf_of_decoded hdec)))
  obtain ⟨h1, h2, h3⟩ := hdec
  obtain ⟨-, c₂, ξ₂, r, i, run, hin, -⟩ := VStar.Recursive.SampledTwo.accepted_steps hacc
  obtain ⟨-, -, -, -, vs, rc, j₁, j₂, -, hieq, -⟩ := VStar.Recursive.SampledTwo.inputAt_ok hin
  have hik : inputOfCase k = some i := by
    unfold inputOfCase
    rw [h1, h2, h3]
    simp only [bind, Except.bind, hin]
    rfl
  subst hieq
  exact ⟨_, _, hik, rfl, rfl⟩

/-- **`StmtsCount`**: an accepted play's evidence holds one statement a template, where the held copies are the ones
`heldOf` builds from the play's run (`HeldOfAt`) and their parts are the interior program's strata's
(`PartsHeldAt`). The run's second-stage check read the evidence's statements (`stmts_of_accepts`) against the held
copies' parts (`RunsAs.parts_of_accepted`), one statement a part (`RunsAs.count_of_parts`). -/
theorem stmtsCount_of {A : Live₂} {C : CaseOf} (Good : Verity.Exchange → Prop) (hof : HeldOfAt A C)
    (hparts : PartsHeldAt A C) : StmtsCount A C Good := by
  intro o k p e hA _ hk hf
  have hA₀ := hA
  obtain ⟨k', hk', hacc⟩ := hA₀
  obtain rfl : k = k' := Option.some.inj (hk.symm.trans hk')
  obtain ⟨i, it, hik, hit, hss⟩ := stmts_of_accepts hacc hf
  have hi : inputOn C o = some i := by
    unfold inputOn
    rw [hk]
    exact hik
  obtain ⟨run, ha⟩ := acceptedTwo_of_formAccepts (ruleIsTwo C) hA hi hit
  obtain ⟨own, q, hp, hcodes⟩ :=
    SpotFlock.RunsAs.parts_of_accepted ha (heldAt_of_parts hof hparts o i it hA hi hit)
  rw [hss] at hcodes
  rw [← VStar.Recursive.SampledTwo.stmtsOwn_size p e k.blobs]
  exact SpotFlock.RunsAs.count_of_parts hp
    (SpotFlock.RunsAs.Round2.namesDistinct_of_two (twoHolds_at hof hA hi hit)) hcodes

end Verity.SampledTwo
