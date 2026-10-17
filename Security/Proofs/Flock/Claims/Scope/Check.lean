import Flock.ProvedScope

/-!
# `ProvedScope.check`, both ways

`check_ok_iff`: the verifier's scope check passes exactly when every rule of `statementRules` holds, the circuit parses and
the public file loads as the check reads them (`circuitOf`, `publicOf`), a shared-row public file only where
`sharedRowsOk`, the wires acyclic where they feed the unit or the outputs (`wiresFeed`, `wiresAcyclic`), and `scopeOk`
(`scopeOkI` with hidden outputs) on the circuit. `scopeOk` holds only where `wiresFeed` fails (`wiresFeed_of_scopeOk`), so
with public outputs the wire condition holds of every circuit `scopeOk` passes. `check_wires`: an accepted circuit's
wires, where they feed the unit or the outputs, are acyclic.
-/

namespace Flock.Readable

/-- The rules of `statementRules`, as propositions. -/
def Rules (t : Tags) (zk : Bool) (count : Nat) (pi : Bool) : Prop :=
  t.hm96Rows = true ∧ t.typed = false ∧ t.retainRounds = true ∧ t.merkleLeaf = "hm96-sha512/v1" ∧
    (zk = true ∨ count ≤ 1) ∧ pi = false

theorem firstFailure_none_iff (t : Tags) (zk : Bool) (count : Nat) (pi : Bool) :
    ProvedScope.firstFailure (ProvedScope.statementRules t zk count pi) = none ↔ Rules t zk count pi := by
  unfold ProvedScope.firstFailure
  rw [Option.map_eq_none_iff, List.find?_eq_none]
  simp only [ProvedScope.statementRules, List.mem_cons, List.mem_nil_iff, or_false, forall_eq_or_imp, forall_eq,
    Bool.not_eq_true', Rules]
  cases t.typed <;> cases pi <;> cases zk <;> simp

/-- `scopeOk` holds only where the wires feed neither the unit nor the outputs. -/
theorem wiresFeed_of_scopeOk {c : Circuit} (h : ProvedScope.scopeOk c = true) : ProvedScope.wiresFeed c = false := by
  simp_all [ProvedScope.scopeOk, ProvedScope.wiresFeed]

/-- **The scope check passes, both ways.** -/
theorem check_ok_iff (t : Tags) (zk : Bool) (cf pf : ByteArray) (tables : List (String × ByteArray)) (count : Nat)
    (pi reg : Bool) :
    ProvedScope.check t zk cf pf tables count pi reg = .ok () ↔
      Rules t zk count pi ∧ ∃ c pub, ProvedScope.circuitOf t cf tables = .ok c ∧
        ProvedScope.publicOf t c pf = .ok pub ∧
        (pub.tables.isSome = true → ProvedScope.sharedRowsOk t zk reg = true) ∧
        (ProvedScope.wiresFeed c = true → ProvedScope.wiresAcyclic c = true) ∧
        (if t.hiddenOutputs then ProvedScope.scopeOkI c else ProvedScope.scopeOk c) = true := by
  rw [← firstFailure_none_iff]
  unfold ProvedScope.check
  constructor
  · intro h
    split at h
    · cases h
    rename_i hf
    refine ⟨hf, ?_⟩
    split at h
    · cases h
    rename_i c hc
    split at h
    · cases h
    rename_i pub hp
    have hs : ¬ (pub.tables.isSome && !ProvedScope.sharedRowsOk t zk reg) = true := by
      intro hx
      simp only [hx, ↓reduceIte] at h
      cases h
    simp only [hs] at h
    have hw : ¬ (ProvedScope.wiresFeed c && !ProvedScope.wiresAcyclic c) = true := by
      intro hx
      simp only [hx, ↓reduceIte] at h
      cases h
    refine ⟨c, pub, hc, hp, fun ht => by simpa [ht] using hs, fun hf => by simpa [hf] using hw, ?_⟩
    simp only [hw] at h
    generalize (if t.hiddenOutputs then ProvedScope.scopeOkI c else ProvedScope.scopeOk c) = b at h
    cases b
    · simp at h
    · rfl
  · rintro ⟨hf, c, pub, hc, hp, hs, hw, hok⟩
    rw [hf]
    simp only [hc, hp]
    have hs' : ¬ (pub.tables.isSome && !ProvedScope.sharedRowsOk t zk reg) = true := by
      cases ht : pub.tables.isSome
      · simp
      · simp [hs ht]
    have hw' : ¬ (ProvedScope.wiresFeed c && !ProvedScope.wiresAcyclic c) = true := by
      cases hf' : ProvedScope.wiresFeed c
      · simp
      · simp [hw hf']
    simp only [hs', hw', hok, Bool.false_eq_true, ↓reduceIte]

/-- **The wire refusal holds of every statement the check accepts**: its circuit parses, and its wires, where they feed the
unit or the outputs (`wiresFeed`), fill every slot from the leaves (`wiresAcyclic`). -/
theorem check_wires {t : Tags} {zk : Bool} {cf pf : ByteArray} {tables : List (String × ByteArray)} {count : Nat}
    {pi reg : Bool} (h : ProvedScope.check t zk cf pf tables count pi reg = .ok ()) :
    ∃ c, ProvedScope.circuitOf t cf tables = .ok c ∧
      (ProvedScope.wiresFeed c = true → ProvedScope.wiresAcyclic c = true) := by
  obtain ⟨-, c, -, hc, -, -, hw, -⟩ := (check_ok_iff t zk cf pf tables count pi reg).1 h
  exact ⟨c, hc, hw⟩

end Flock.Readable
