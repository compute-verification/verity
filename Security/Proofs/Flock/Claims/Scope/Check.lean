import Flock.ProvedScope

/-!
# `ProvedScope.check`, both ways

`check_ok_iff`: the verifier's scope check passes exactly when every rule of `statementRules` holds, the circuit parses and
the public file loads as the check reads them (`circuitOf`, `publicOf`), a shared-row public file only where
`sharedRowsOk`, and `scopeOk` (`scopeOkH` with hidden outputs) on the circuit.
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

/-- **The scope check passes, both ways.** -/
theorem check_ok_iff (t : Tags) (zk : Bool) (cf pf : ByteArray) (tables : List (String × ByteArray)) (count : Nat)
    (pi reg : Bool) :
    ProvedScope.check t zk cf pf tables count pi reg = .ok () ↔
      Rules t zk count pi ∧ ∃ c pub, ProvedScope.circuitOf t cf tables = .ok c ∧
        ProvedScope.publicOf t c pf = .ok pub ∧
        (pub.tables.isSome = true → ProvedScope.sharedRowsOk t zk reg = true) ∧
        (if t.hiddenOutputs then ProvedScope.scopeOkH c else ProvedScope.scopeOk c) = true := by
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
    refine ⟨c, pub, hc, hp, fun ht => by simpa [ht] using hs, ?_⟩
    generalize (if t.hiddenOutputs then ProvedScope.scopeOkH c else ProvedScope.scopeOk c) = b at h
    cases b
    · simp at h
    · rfl
  · rintro ⟨hf, c, pub, hc, hp, hs, hok⟩
    rw [hf]
    simp only [hc, hp]
    have hs' : ¬ (pub.tables.isSome && !ProvedScope.sharedRowsOk t zk reg) = true := by
      cases ht : pub.tables.isSome
      · simp
      · simp [hs ht]
    simp only [hs', hok, Bool.false_eq_true, ↓reduceIte]

end Flock.Readable
