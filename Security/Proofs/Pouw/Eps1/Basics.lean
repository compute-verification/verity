import Models.Pouw.Guarantees.Eps1

/-!
# M3's basic lemmas for Dither's lower bounds (stubs)

Runs exist and are unique, a block's outside slots, the zero instance's fiber and the information count (QSTAR's
Theorem B), each a `sorry` until a prover takes it (gap list E1–E5: note:20261017T0801Z-draft-lean-stmt).
-/

namespace Pouw.SecurityProofs.Eps1

open Pouw.Eps1

/-- An argument's value reads only instructions before its own. -/
theorem argVal_congr {ι : Type} {G : ℕ} (P : M3 ι G) (x : ι → Word) (v v' : Fin G → Word) (t : Fin G)
    (h : ∀ s : Fin G, s.val < t.val → v s = v' s) (j : Fin 3) : P.argVal x v t j = P.argVal x v' t j := by
  unfold M3.argVal
  split
  · rfl
  · rename_i s _
    exact h _ s.isLt

/-- Every M3 program has a run on every input (instructions read only earlier words). -/
theorem run_exists {ι : Type} {G : ℕ} (P : M3 ι G) (x : ι → Word) : ∃ v, P.IsRun x v := by
  have key : ∀ n, ∃ v : Fin G → Word, ∀ t : Fin G, t.val < n →
      v t = P.op t (P.argVal x v t 0) (P.argVal x v t 1) (P.argVal x v t 2) := by
    intro n
    induction n with
    | zero => exact ⟨fun _ => 0, fun t h => absurd h (Nat.not_lt_zero _)⟩
    | succ n ih =>
      obtain ⟨v, hv⟩ := ih
      by_cases hn : n < G
      · refine ⟨Function.update v ⟨n, hn⟩
          (P.op ⟨n, hn⟩ (P.argVal x v ⟨n, hn⟩ 0) (P.argVal x v ⟨n, hn⟩ 1) (P.argVal x v ⟨n, hn⟩ 2)),
          fun t ht => ?_⟩
        have hle : ∀ s : Fin G, s.val < t.val → Function.update v ⟨n, hn⟩
            (P.op ⟨n, hn⟩ (P.argVal x v ⟨n, hn⟩ 0) (P.argVal x v ⟨n, hn⟩ 1) (P.argVal x v ⟨n, hn⟩ 2)) s =
              v s :=
          fun s hs => Function.update_of_ne (Fin.ne_of_val_ne (by simp only; omega)) _ _
        rw [argVal_congr P x _ v t hle 0, argVal_congr P x _ v t hle 1, argVal_congr P x _ v t hle 2]
        rcases Nat.lt_succ_iff_lt_or_eq.mp ht with h | h
        · rw [Function.update_of_ne (Fin.ne_of_val_ne (by simp only; omega)), hv t h]
        · obtain rfl : t = ⟨n, hn⟩ := Fin.ext h
          exact Function.update_self _ _ _
      · exact ⟨v, fun t ht => hv t (by omega)⟩
  obtain ⟨v, hv⟩ := key G
  exact ⟨v, fun t => hv t t.isLt⟩

/-- The run is unique. -/
theorem run_unique {ι : Type} {G : ℕ} (P : M3 ι G) (x : ι → Word) (v v' : Fin G → Word) (h : P.IsRun x v)
    (h' : P.IsRun x v') : v = v' := by
  have key : ∀ n, ∀ t : Fin G, t.val < n → v t = v' t := by
    intro n
    induction n with
    | zero => intro t ht; omega
    | succ n ih =>
      intro t ht
      have hle : ∀ s : Fin G, s.val < t.val → v s = v' s := fun s hs => ih s (by omega)
      rw [h t, h' t, argVal_congr P x v v' t hle 0, argVal_congr P x v v' t hle 1,
        argVal_congr P x v v' t hle 2]
  funext t
  exact key _ t (Nat.lt_succ_self _)

open Classical in
/-- Theorem B's counting: a block whose live part has `a` instructions, each but the writer an argument of another
live one, takes at most `2a + 1` of its input slots from outside it (at least `a − 1` are edges inside it). -/
theorem block_outside_slots {Rd Cd : Type} {m n Lseg : ℕ} {FR FU : Type} {G : ℕ}
    (P : Prog Rd Cd m n Lseg FR FU G) (blk : Fin G → Tau m n Lseg) (hW : P.WordLocal blk) (τ : Tau m n Lseg)
    (live : Finset (Fin G)) (hlive : ∀ t ∈ live, blk t = τ) (hout : P.out τ ∈ live)
    (hused : ∀ t ∈ live, t ≠ P.out τ → ∃ t' ∈ live, ∃ j s, P.prog.arg t' j = .inr s ∧ M3.earlier t' s = t) :
    ((Finset.univ : Finset (Fin 3)).product live |>.filter fun tj =>
      ∀ s, P.prog.arg tj.2 tj.1 = .inr s → M3.earlier tj.2 s ∉ live).card ≤ 2 * live.card + 1 := by
  let tgt : Fin 3 × Fin G → Fin G := fun tj =>
    match P.prog.arg tj.2 tj.1 with
    | .inl _ => tj.2
    | .inr s => M3.earlier tj.2 s
  have hsplit := Finset.card_filter_add_card_filter_not
    (s := (Finset.univ : Finset (Fin 3)).product live)
    (fun tj => ∀ s, P.prog.arg tj.2 tj.1 = .inr s → M3.earlier tj.2 s ∉ live)
  have hcard : ((Finset.univ : Finset (Fin 3)).product live).card = 3 * live.card := by
    rw [Finset.product_eq_sprod, Finset.card_product, Finset.card_univ, Fintype.card_fin]
  have hsub : live.erase (P.out τ) ⊆ (((Finset.univ : Finset (Fin 3)).product live).filter fun tj =>
      ¬ ∀ s, P.prog.arg tj.2 tj.1 = .inr s → M3.earlier tj.2 s ∉ live).image tgt := by
    intro t ht
    obtain ⟨hne, htl⟩ := Finset.mem_erase.mp ht
    obtain ⟨t', ht', j, s, harg, hearl⟩ := hused t htl hne
    refine Finset.mem_image.mpr ⟨(j, t'), ?_, ?_⟩
    · rw [Finset.mem_filter, Finset.product_eq_sprod, Finset.mem_product]
      exact ⟨⟨Finset.mem_univ _, ht'⟩, fun hall => hall s harg (hearl ▸ htl)⟩
    · simp only [tgt, harg, hearl]
  have h1 := Finset.card_erase_of_mem hout
  have h2 := Finset.card_le_card hsub
  have h3 := Finset.card_image_le (s := ((Finset.univ : Finset (Fin 3)).product live).filter fun tj =>
      ¬ ∀ s, P.prog.arg tj.2 tj.1 = .inr s → M3.earlier tj.2 s ∉ live) (f := tgt)
  have hpos : 0 < live.card := Finset.card_pos.mpr ⟨_, hout⟩
  omega

/-- The zero instance's fiber: `r ↦ r·u` determines a column `u` with entries in `±1..±64`, so the column-side
words a block reads carry at least 224 bits, at least 7 words. -/
theorem fiber_needs_seven {n : ℕ} (U U' : Cols n) (j : Fin n) (hU : ColsValid U) (hU' : ColsValid U')
    (h : ∀ r : Fin 32 → Bool, (∑ p, if r p then U j p else 0) = ∑ p, if r p then U' j p else 0) :
    ∀ p, U j p = U' j p := by
  intro p
  simpa using h fun q => decide (q = p)

/-- The information count: six words can't tell apart the `128^32 = 2^224` columns with entries in `±1..±64`; any
map from such columns to `k` words that is injective on them has `k ≥ 7` (`k ≥ 8` for the `2^256` int8 segments). -/
theorem words_needed (k : ℕ) (f : (Fin 32 → ℤ) → Fin k → Word)
    (hf : ∀ u u', (∀ p, 1 ≤ |u p| ∧ |u p| ≤ 64) → (∀ p, 1 ≤ |u' p| ∧ |u' p| ≤ 64) → f u = f u' → u = u') :
    7 ≤ k := by
  sorry

end Pouw.SecurityProofs.Eps1
