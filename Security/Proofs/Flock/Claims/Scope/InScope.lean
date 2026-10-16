import Proofs.Flock.Claims.Scope.Check
import Proofs.Flock.Claims.Scope.Setup

/-!
# The readable scope is the verifier's scope check

For every statement the verifier parses (`Parses`) and every relation part `rel`:

* `inScope_iff`: `Claims.Flock.InScope zk (toReadable rel p)` holds exactly when `ProvedScope.check` passes and
  `22 ≤ m`. The check reads no parameter of `m` and `k_log`; the setup gives `m ≤ 35`, `6 ≤ k_log ≤ min(m, 27)`, and under
  `--zk` `25 ≤ m` (`table0`). Without `--zk`, `22 ≤ m` is `verify`'s (`fast100`'s schedule), not the setup's.
* `inScope_iff_zk`: under `--zk`, exactly when `ProvedScope.check` passes.
-/

namespace Flock.Readable

theorem tablesOf_of_parses {p : Parsed} (hp : Parses p) : p.toRead.tablesOf = some p.sts := by
  unfold Parses Read.parse at hp
  obtain ⟨sts, hs, he⟩ := Option.map_eq_some_iff.1 hp
  rw [hs, ← he]

/-- What the verifier's parser returns, it parses. -/
theorem parses_of_parse {r : Read} {p : Parsed} (h : r.parse = some p) : Parses p := by
  unfold Read.parse at h
  obtain ⟨sts, hs, rfl⟩ := Option.map_eq_some_iff.1 h
  show Read.parse r = _
  unfold Read.parse
  rw [hs]
  rfl

/-- `InScope` reads the rules of `statementRules` off the form. -/
theorem rules_of_inScope {rel : Parsed → Relation} {p : Parsed}
    (h : Claims.Flock.InScope p.zk (toReadable rel p)) :
    Rules p.tags p.zk p.I.count p.publicInputs.isSome := by
  obtain ⟨-, -, -, -, h1, h2, h3, h4, h5, h6, -⟩ := h
  simp only [toReadable, Read.form, Bool.not_eq_true] at h1 h2 h3 h4 h5 h6
  exact ⟨h1, h2, h3, h4, h5, h6⟩

theorem sharedRowsOk_iff (t : Tags) (zk reg : Bool) :
    ProvedScope.sharedRowsOk t zk reg = true ↔ zk = true ∧ t.hiddenOutputs = true ∧ reg = true := by
  simp [ProvedScope.sharedRowsOk, and_assoc]

/-- `scopeOk` (`scopeOkH` with hidden outputs) is the unit's conditions and the rows'. -/
theorem scope_iff_rows (hidden : Bool) (c : Circuit) :
    (if hidden then ProvedScope.scopeOkH c else ProvedScope.scopeOk c) = true ↔
      (c.unit.inGroups.size = 1 ∧ ¬ (c.outNet != c.unitNet) = true ∧ c.unit.unitLog ≤ 32 ∧
        ProvedScope.portReadsB c.unit = true) ∧
      (if hidden then ProvedScope.v1Ports c || ProvedScope.bitRows c else ProvedScope.v1Ports c) = true := by
  cases hidden <;>
    simp [ProvedScope.scopeOk, ProvedScope.scopeOkH, and_assoc]

/-- **The readable scope is the verifier's scope check, and `22 ≤ m`**, for every statement the verifier parses. -/
theorem inScope_iff (rel : Parsed → Relation) {p : Parsed} (hp : Parses p) :
    Claims.Flock.InScope p.zk (toReadable rel p) ↔ p.check = .ok () ∧ 22 ≤ (toReadable rel p).m := by
  rw [Read.check, check_ok_iff]
  by_cases hR : Rules p.tags p.zk p.I.count p.publicInputs.isSome
  swap
  · exact ⟨fun h => absurd (rules_of_inScope h) hR, fun h => absurd h.1.1 hR⟩
  have hR' := hR
  obtain ⟨h1, h2, h3, h4, h5, h6⟩ := hR'
  have hpi : p.publicInputs = none := Option.not_isSome_iff_eq_none.1 (by simp [h6])
  obtain ⟨st, h0, hc, ⟨pop, hpop⟩, hm, hk6, hk27, hm35, hzk⟩ := table0 (tablesOf_of_parses hp) h2 hpi
  have hcirc : p.circ = st.c := by simp only [Read.circ, hc]
  have hpub : p.pub = pop := by simp only [Read.pub, hcirc, hpop]
  have em : (toReadable rel p).m = st.m := by simp [toReadable, h0]
  have ek : (toReadable rel p).kLog = st.c.kLog := by simp [toReadable, h0]
  have ef : (toReadable rel p).form = p.form := rfl
  have eR : (∃ c pub, ProvedScope.circuitOf p.tags p.I.circuitFile p.I.tables = .ok c ∧
      ProvedScope.publicOf p.tags c p.I.publicFile = .ok pub ∧
      (pub.tables.isSome = true → ProvedScope.sharedRowsOk p.tags p.zk p.registered = true) ∧
      (if p.tags.hiddenOutputs then ProvedScope.scopeOkH c else ProvedScope.scopeOk c) = true) ↔
      (pop.tables.isSome = true → ProvedScope.sharedRowsOk p.tags p.zk p.registered = true) ∧
      (if p.tags.hiddenOutputs then ProvedScope.scopeOkH st.c else ProvedScope.scopeOk st.c) = true := by
    constructor
    · rintro ⟨c, pub, e1, e2, hX⟩
      rw [hc] at e1
      cases e1
      rw [hpop] at e2
      cases e2
      exact hX
    · intro hX
      exact ⟨st.c, pop, hc, hpop, hX⟩
  rw [eR, em]
  simp only [Claims.Flock.InScope, em, ek, ef, Read.form, hcirc, hpub]
  rw [sharedRowsOk_iff, scope_iff_rows]
  constructor
  · rintro ⟨hlo, -, -, -, -, -, -, -, -, -, hsh, hg, ht, hl, hpr, hrows⟩
    refine ⟨⟨hR, hsh, ⟨hg, ht, hl, hpr⟩, ?_⟩, ?_⟩
    · cases hh : p.tags.hiddenOutputs
      · simpa [hh] using hrows
      · simpa [hh] using hrows
    · split at hlo <;> omega
  · rintro ⟨⟨-, hsh, ⟨hg, ht, hl, hpr⟩, hrows⟩, hlo⟩
    have hlo' : (if p.zk = true then 25 else 22) ≤ st.m := by
      split
      · exact hzk (by assumption)
      · exact hlo
    refine ⟨hlo', hm35, hk6, by omega, h1, by simp [h2], h3, h4, h5, by simp [h6], hsh, hg, ht, hl, hpr, ?_⟩
    cases hh : p.tags.hiddenOutputs
    · simp only [hh, Bool.false_eq_true, ↓reduceIte] at hrows
      exact Or.inl hrows
    · simp only [hh, ↓reduceIte, Bool.or_eq_true] at hrows
      exact hrows.imp id fun h => ⟨rfl, h⟩

/-- **Under `--zk`, the readable scope is the verifier's scope check**, for every statement the verifier parses. -/
theorem inScope_iff_zk (rel : Parsed → Relation) {p : Parsed} (hp : Parses p) (hz : p.zk = true) :
    Claims.Flock.InScope p.zk (toReadable rel p) ↔ p.check = .ok () := by
  rw [inScope_iff rel hp]
  refine ⟨And.left, fun h => ⟨h, ?_⟩⟩
  have hR := ((check_ok_iff _ _ _ _ _ _ _ _).1 h).1
  have hpi : p.publicInputs = none := Option.not_isSome_iff_eq_none.1 (by simp [hR.2.2.2.2.2])
  obtain ⟨st, h0, -, -, -, -, -, -, hzk⟩ := table0 (tablesOf_of_parses hp) hR.2.1 hpi
  have em : (toReadable rel p).m = st.m := by simp [toReadable, h0]
  have := hzk hz
  omega

end Flock.Readable
