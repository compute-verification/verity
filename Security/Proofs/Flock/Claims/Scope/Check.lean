import Flock.ProvedScope

/-!
# `ProvedScope.check`, both ways

`check_ok_iff`: the verifier's scope check passes exactly when every rule of `statementRules` holds, the circuit parses and
the public file loads as the check reads them (`circuitOf`, `publicOf`), a shared-row public file only where
`sharedRowsOk`, and on a templated circuit (`Circuit.tmpl`) `scopeOkT`, on any other the wires acyclic where they feed
the unit or the outputs (`wiresFeed`, `wiresAcyclic`) and `scopeOk` (`scopeOkI` with hidden outputs). `scopeOk` holds only where `wiresFeed` fails (`wiresFeed_of_scopeOk`), so
with public outputs the wire condition holds of every circuit `scopeOk` passes. `check_wires`: an accepted flat circuit's
wires, where they feed the unit or the outputs, are acyclic.
-/

namespace Flock.Readable

/-- `leafOk`, as a proposition: `hm96-sha512/v1` leaves, or plain SHA-512 leaves with hidden outputs without `--zk`. -/
def LeafOk (t : Tags) (zk : Bool) : Prop :=
  t.merkleLeaf = "hm96-sha512/v1" ∨
    (zk = false ∧ t.hiddenOutputs = true ∧ t.merkleLeaf = "flock-leaf/sha512-unsalted")

theorem leafOk_iff (t : Tags) (zk : Bool) : ProvedScope.leafOk t zk = true ↔ LeafOk t zk := by
  unfold ProvedScope.leafOk LeafOk
  cases zk <;> cases t.hiddenOutputs <;> simp

/-- Under `--zk`, and wherever outputs are public, the leaves `leafOk` covers are `hm96-sha512/v1`. -/
theorem LeafOk.hm96 {t : Tags} {zk : Bool} (h : LeafOk t zk) (hz : zk = true ∨ t.hiddenOutputs = false) :
    t.merkleLeaf = "hm96-sha512/v1" := by
  rcases h with h | ⟨h1, h2, -⟩
  · exact h
  · rcases hz with hz | hz <;> simp_all

/-- The rules of `statementRules`, as propositions. -/
def Rules (t : Tags) (zk : Bool) (count : Nat) (pi : Bool) : Prop :=
  t.hm96Rows = true ∧ t.typed = false ∧ t.retainRounds = true ∧ LeafOk t zk ∧
    count ≤ (if zk then ProvedScope.zkTableCap else 1) ∧ pi = false

theorem firstFailure_none_iff (t : Tags) (zk : Bool) (count : Nat) (pi : Bool) :
    ProvedScope.firstFailure (ProvedScope.statementRules t zk count pi) = none ↔ Rules t zk count pi := by
  unfold ProvedScope.firstFailure
  rw [Option.map_eq_none_iff, List.find?_eq_none]
  simp only [ProvedScope.statementRules, List.mem_cons, List.mem_nil_iff, or_false, forall_eq_or_imp, forall_eq,
    Bool.not_eq_true', Rules, ← leafOk_iff]
  cases t.typed <;> cases pi <;> cases zk <;> simp

/-- `scopeOk` holds only where the wires feed neither the unit nor the outputs. -/
theorem wiresFeed_of_scopeOk {c : Circuit} (h : ProvedScope.scopeOk c = true) : ProvedScope.wiresFeed c = false := by
  simp_all [ProvedScope.scopeOk, ProvedScope.wiresFeed]

/-- What the check requires of the circuit's form: a templated circuit (`Circuit.tmpl`) in the form
`ProvedScope.scopeOkT` covers, on its description; any other with the wire condition and `scopeOk` (`scopeOkI` with
hidden outputs). -/
def FormOk (t : Tags) (zk reg : Bool) (count : Nat) (c : Circuit) : Prop :=
  match c.tmpl with
  | some tu => ProvedScope.scopeOkT t.hiddenOutputs zk reg count c tu = true
  | none => (ProvedScope.wiresFeed c = true → ProvedScope.wiresAcyclic c = true) ∧
      (if t.hiddenOutputs then ProvedScope.scopeOkI c else ProvedScope.scopeOk c) = true

/-- **The scope check passes, both ways**: a templated circuit (`Circuit.tmpl`) in the form `ProvedScope.scopeOkT` covers,
on its description; any other with the wire condition and `scopeOk` (`scopeOkI` with hidden outputs). -/
theorem check_ok_iff (t : Tags) (zk : Bool) (cf pf : ByteArray) (tables : List (String × ByteArray)) (count : Nat)
    (pi reg : Bool) :
    ProvedScope.check t zk cf pf tables count pi reg = .ok () ↔
      Rules t zk count pi ∧ ∃ c pub, ProvedScope.circuitOf t cf tables = .ok c ∧
        ProvedScope.publicOf t c pf = .ok pub ∧
        (pub.tables.isSome = true → ProvedScope.sharedRowsOk t zk reg = true) ∧ FormOk t zk reg count c := by
  rw [← firstFailure_none_iff]
  unfold ProvedScope.check ProvedScope.circuitRules
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
    cases hct : c.tmpl with
    | some tu =>
      simp only [FormOk, hct] at h ⊢
      revert h
      generalize ProvedScope.scopeOkT t.hiddenOutputs zk reg count c tu = b
      cases b <;> simp
    | none =>
      simp only [FormOk, hct] at h ⊢
      have hw : ¬ (ProvedScope.wiresFeed c && !ProvedScope.wiresAcyclic c) = true := by
        intro hx
        simp only [hx, ↓reduceIte] at h
        cases h
      refine ⟨fun hf => by simpa [hf] using hw, ?_⟩
      simp only [hw] at h
      generalize (if t.hiddenOutputs then ProvedScope.scopeOkI c else ProvedScope.scopeOk c) = b at h
      cases b
      · simp at h
      · rfl
  · rintro ⟨hf, c, pub, hc, hp, hs, hrest⟩
    rw [hf]
    simp only [hc, hp]
    have hs' : ¬ (pub.tables.isSome && !ProvedScope.sharedRowsOk t zk reg) = true := by
      cases ht : pub.tables.isSome
      · simp
      · simp [hs ht]
    simp only [hs', Bool.false_eq_true, ↓reduceIte]
    cases hct : c.tmpl with
    | some tu =>
      simp only [FormOk, hct] at hrest ⊢
      simp [hrest]
    | none =>
      simp only [FormOk, hct] at hrest ⊢
      obtain ⟨hw, hok⟩ := hrest
      have hw' : ¬ (ProvedScope.wiresFeed c && !ProvedScope.wiresAcyclic c) = true := by
        cases hf' : ProvedScope.wiresFeed c
        · simp
        · simp [hw hf']
      simp only [hw', hok, Bool.false_eq_true, ↓reduceIte]

/-- **The wire refusal holds of every flat statement the check accepts**: its circuit parses, and its wires, where they
feed the unit or the outputs (`wiresFeed`), fill every slot from the leaves (`wiresAcyclic`). -/
theorem check_wires {t : Tags} {zk : Bool} {cf pf : ByteArray} {tables : List (String × ByteArray)} {count : Nat}
    {pi reg : Bool} (h : ProvedScope.check t zk cf pf tables count pi reg = .ok ()) :
    ∃ c, ProvedScope.circuitOf t cf tables = .ok c ∧
      (c.tmpl = none → ProvedScope.wiresFeed c = true → ProvedScope.wiresAcyclic c = true) := by
  obtain ⟨-, c, -, hc, -, -, hrest⟩ := (check_ok_iff t zk cf pf tables count pi reg).1 h
  refine ⟨c, hc, fun ht => ?_⟩
  simp only [FormOk, ht] at hrest
  exact hrest.1

end Flock.Readable
