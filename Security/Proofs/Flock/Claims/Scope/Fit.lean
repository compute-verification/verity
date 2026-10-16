import Proofs.Flock.Claims.Scope.InScope
import Proofs.Flock.Claims.Scope.Tables
import Proofs.Flock.Soundness.Refine.Setup
import Proofs.Flock.Soundness.Discharge.Hidden.Facts

/-!
# Every table the verifier parses fits the first table's blocks

`tableAt`: each table of a statement `Read.tablesOf` accepts (untyped, without the verifier's own public input words)
has the circuit the scope check parses, the verifier's layout, and the well-formedness the soundness proof reads
(`StmtWF`, `RegionsWF`). `tables_fit`: so every table has the first table's circuit, `m` and link points, and fits its
blocks (`Tab.Fits`). `relS_layout`: at such a statement, `RelS` of the readable statement is every table's
`Layout.model` at `regionsOf`, the statement the soundness proof reads, satisfied by its share of the hidden values
(`2^m` bits per table).
-/

namespace Flock.Readable

open FlockSoundness FlockSoundness.Discharge Exec

theorem setup_of_tablesOf {r : Read} {sts : Array Stmt} (hs : r.tablesOf = some sts) :
    ∃ spec, r.setup = .ok (spec, sts) := by
  unfold Read.tablesOf at hs
  split at hs
  swap
  · cases hs
  rename_i spec sts' hset
  refine ⟨spec, ?_⟩
  have hsts : sts' = sts := by
    split at hs
    · split at hs
      · exact Option.some.inj hs
      · cases hs
    · exact Option.some.inj hs
  rw [← hsts]
  exact hset

/-- **Each table the verifier parses**: the scope check's circuit, the verifier's layout, `StmtWF` and `RegionsWF`. -/
theorem tableAt {r : Read} {sts : Array Stmt} (hs : r.tablesOf = some sts) (ht : r.tags.typed = false)
    (hpi : r.publicInputs = none) {j : ℕ} (hj : j < sts.size) :
    ProvedScope.circuitOf r.tags r.I.circuitFile r.I.tables = .ok sts[j].c ∧ FlockSoundness.Layout sts[j] ∧
      Refine.StmtWF sts[j] ∧ Refine.RegionsWF sts[j] := by
  obtain ⟨spec, hset⟩ := setup_of_tablesOf hs
  unfold Read.setup at hset
  rw [hpi] at hset
  obtain ⟨-, hall⟩ := HiddenExec.setupTables_setupOne hset
  obtain ⟨d, hd, -⟩ := hall j hj
  have ht' : (tagsAt r.tags r.I.count j).typed = false := (tagsAt_typed _ _ _).trans ht
  unfold HiddenExec.setupOne at hd
  rw [tagsAt_hiddenOutputs] at hd
  cases hh : r.tags.hiddenOutputs
  · rw [hh] at hd
    simp only [Bool.false_eq_true, ↓reduceIte] at hd
    obtain ⟨hp, -⟩ := setupH_scope ht' hd
    rw [Composed.parse_tagsAt] at hp
    refine ⟨?_, setupH_layout ht' hd, Refine.setupH_wf hd⟩
    simp only [ProvedScope.circuitOf, hh, Bool.false_eq_true, ↓reduceIte, hp]
  · rw [hh] at hd
    simp only [↓reduceIte] at hd
    have hp := (Hidden.setupHidden_spec' hd).parse
    rw [parseH_tagsAt ht] at hp
    refine ⟨?_, Hidden.hidden_layout' ht' hd, Hidden.setupHidden_wf hd⟩
    simp only [ProvedScope.circuitOf, hh, ↓reduceIte, hp]

/-- **Every table fits the first table's blocks**: the same circuit (so `k_log`), the same link points (so `nbl` and
`m`), `6 ≤ k_log ≤ m` and the pin in the block. -/
theorem tables_fit {r : Read} {sts : Array Stmt} (hs : r.tablesOf = some sts) (ht : r.tags.typed = false)
    (hpi : r.publicInputs = none) {j : ℕ} (hj : j < sts.size) :
    (Tab.ofStmt sts[j]).Fits (sts[0]'(by omega)).m (sts[0]'(by omega)).c.kLog := by
  have h0 : 0 < sts.size := by omega
  obtain ⟨hcj, -, wj, rj⟩ := tableAt hs ht hpi hj
  obtain ⟨hc0, -, -, r0⟩ := tableAt hs ht hpi h0
  have hc : sts[j].c = sts[0].c := Except.ok.inj (hcj.symm.trans hc0)
  obtain ⟨spec, hset⟩ := setup_of_tablesOf hs
  unfold Read.setup at hset
  rw [hpi] at hset
  have hpts := (HiddenExec.setupTables_pointsAny hset).2
  have hp : sts[j].mPts = sts[0].mPts := (hpts j hj).trans (hpts 0 h0).symm
  have hnbl : sts[j].nbl = sts[0].nbl := by have := rj.mPts; have := r0.mPts; omega
  have hk : sts[j].c.kLog = sts[0].c.kLog := by rw [hc]
  refine ⟨?_, hk, wj.hk6, wj.hk, wj.pin⟩
  show sts[j].m = sts[0].m
  rw [rj.hm, r0.hm, hk, hnbl]

/-- **`RelS` at a statement the verifier parses**: every table's model statement (`Layout.model` at `regionsOf`) is
satisfied by its share of the hidden values, `2^m` bits per table. -/
theorem relS_layout (H : Claims.Msg → Claims.Msg) {p : Parsed} (hp : Parses p) (ht : p.tags.typed = false)
    (hpi : p.publicInputs = none) (w : Claims.Flock.Wit) :
    Claims.Flock.RelS H (toReadable relation p) w ↔
      ∀ c (hc : c < p.sts.size), ∃ L : FlockSoundness.Layout p.sts[c],
        (L.model (Refine.regionsOf p.sts[c])).Satisfies (zOf (2 ^ (toReadable relation p).m) w c p.sts[c].m) := by
  have hs := tablesOf_of_parses hp
  rw [relS_relation]
  refine forall₂_congr fun c hc => ?_
  have h0 : 0 < p.sts.size := by omega
  have hm : (toReadable relation p).m = p.sts[0].m := by
    simp only [toReadable, Array.getElem?_eq_getElem h0, Option.map_some, Option.getD_some]
  have hk : (toReadable relation p).kLog = p.sts[0].c.kLog := by
    simp only [toReadable, Array.getElem?_eq_getElem h0, Option.map_some, Option.getD_some]
  obtain ⟨-, L, w0, -⟩ := tableAt hs ht hpi hc
  have hfit := tables_fit hs ht hpi hc
  have hT : 2 ^ ((toReadable relation p).m - (toReadable relation p).kLog) * 2 ^ (toReadable relation p).kLog =
      2 ^ (toReadable relation p).m := by
    rw [← Nat.pow_add, Nat.sub_add_cancel]
    rw [hm, hk, ← hfit.1, ← hfit.2.1]
    exact w0.hk
  rw [hT, getElem!_pos p.sts c hc, hm, hk]
  exact ⟨fun ⟨_, hz⟩ => ⟨L, hz⟩, fun ⟨_, hz⟩ => ⟨hfit, hz⟩⟩

end Flock.Readable
