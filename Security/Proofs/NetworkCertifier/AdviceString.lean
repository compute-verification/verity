import Proofs.NetworkCertifier.Schedule
import Definitions.NetworkCertifier.Assumptions.AdviceString
import Mathlib.Data.List.Infix
import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-!
# The leak bound: the advice string's charge

`Guarantees.AdviceString`. The count is Theorem 3's for the constant-rate Program (`Schedule.lean`), with the content
advice ranging over the advice strings within the caps `b`: each is charged at most `Σ_w b w` in all, so there are at
most `2^(Σ_w b w)` of them (`BitsCount`). A precommitted, deterministic compiler gives every run the same Program, so
the composed statement is the count through the wire. The two codec lemmas show that `BitsCount` holds when the
charges sum to at least a prefix-free encoding's length, or to at least an injective encoding's length plus one.
-/

namespace NetTiming

open Finset

variable {L K S : ℕ} {Z In Act Wire : Type}

/-! ### The advice within the caps -/

theorem finite_charged {Adv : Type} {W : ℕ} (bits : Adv → Fin W → ℕ) (hbits : BitsCount bits) (c : ℕ) :
    {a | ∑ w, bits a w ≤ c}.Finite :=
  Set.finite_of_encard_le_coe (hbits c)

theorem card_charged_le {Adv : Type} {W : ℕ} (bits : Adv → Fin W → ℕ) (hbits : BitsCount bits) (c : ℕ) :
    (finite_charged bits hbits c).toFinset.card ≤ 2 ^ c := by
  have h := hbits c
  rw [Set.Finite.encard_eq_coe_toFinset_card (finite_charged bits hbits c)] at h
  exact_mod_cast h

theorem sum_le_of_withinCaps {Adv : Type} {W : ℕ} {bits : Adv → Fin W → ℕ} {b : Fin W → ℕ} {a : Adv}
    (h : WithinCaps bits b a) : ∑ w, bits a w ≤ ∑ w, b w :=
  Finset.sum_le_sum fun w _ => h w

/-! ### Two codecs that meet `BitsCount` -/

/-- The first `b` bits of a list, padded with `false`. -/
def padBits (b : ℕ) (l : List Bool) : Fin b → Bool := fun i => (l[(i : ℕ)]?).getD false

theorem prefix_of_padBits_eq {b : ℕ} {l l' : List Bool} (hl : l.length ≤ l'.length) (hb : l'.length ≤ b)
    (h : padBits b l = padBits b l') : l <+: l' := by
  rw [List.prefix_iff_eq_take]
  apply List.ext_getElem (by simp [hl])
  intro i h1 h2
  rw [List.getElem_take]
  have := congrFun h ⟨i, by omega⟩
  simpa [padBits, List.getElem?_eq_getElem h1, List.getElem?_eq_getElem (show i < l'.length by omega)] using this

/-- **A prefix-free encoding.** If no advice value's encoding is a prefix of another's and the charges sum to at least
the encoding's length, then padding to `c` bits tells apart the advice charged at most `c`, so `bits` counts. -/
theorem bitsCount_of_prefixFree {Adv : Type} {W : ℕ} (bits : Adv → Fin W → ℕ) (enc : Adv → List Bool)
    (hpf : ∀ a a', enc a <+: enc a' → a = a') (hlen : ∀ a, (enc a).length ≤ ∑ w, bits a w) : BitsCount bits := by
  intro c
  have hinj : Set.InjOn (fun a => padBits c (enc a)) {a | ∑ w, bits a w ≤ c} := by
    intro a ha a' ha' h
    have hc : ∑ w, bits a w ≤ c := ha
    have hc' : ∑ w, bits a' w ≤ c := ha'
    rcases le_total (enc a).length (enc a').length with hl | hl
    · exact hpf a a' (prefix_of_padBits_eq hl ((hlen a').trans hc') h)
    · exact (hpf a' a (prefix_of_padBits_eq hl ((hlen a).trans hc) h.symm)).symm
  calc _ ≤ ((univ : Finset (Fin c → Bool)).card : ℕ∞) :=
        encard_le_card_of_injOn _ hinj univ fun _ _ => mem_univ _
    _ = ((2 ^ c : ℕ) : ℕ∞) := by simp

theorem eq_of_padBits_append_true {c : ℕ} {l l' : List Bool} (hl' : l'.length < c) (hle : l.length ≤ l'.length)
    (h : padBits c (l ++ [true]) = padBits c (l' ++ [true])) : l = l' := by
  have hpre := prefix_of_padBits_eq (by simp; omega) (by simp; omega) h
  rcases Nat.lt_or_eq_of_le hle with hlt | heq
  · exfalso
    have := congrFun h ⟨l'.length, hl'⟩
    simp only [padBits] at this
    rw [List.getElem?_append_right (by omega), List.getElem?_append_right (by omega),
      List.getElem?_eq_none (by simp; omega)] at this
    simp at this
  · obtain ⟨t, ht⟩ := hpre
    have : t = [] := by
      have := congrArg List.length ht
      simp at this
      exact List.eq_nil_of_length_eq_zero (by omega)
    subst this
    simpa using ht

/-- **Any injective encoding, charged its length plus one.** The advice charged at most `c` bits has encodings shorter
than `c`, and there are `2^c − 1` bit strings shorter than `c`. A public cap `ℓ` on the encoding's length, charged
`ℓ + 1`, is this case. -/
theorem bitsCount_of_lengthSucc {Adv : Type} {W : ℕ} (bits : Adv → Fin W → ℕ) (enc : Adv → List Bool)
    (hinj : Function.Injective enc) (hlen : ∀ a, (enc a).length + 1 ≤ ∑ w, bits a w) : BitsCount bits := by
  intro c
  have hinj' : Set.InjOn (fun a => padBits c (enc a ++ [true])) {a | ∑ w, bits a w ≤ c} := by
    intro a ha a' ha' h
    have h1 : (enc a).length < c := by have hc : ∑ w, bits a w ≤ c := ha; have := hlen a; omega
    have h2 : (enc a').length < c := by have hc : ∑ w, bits a' w ≤ c := ha'; have := hlen a'; omega
    apply hinj
    rcases le_total (enc a).length (enc a').length with hl | hl
    · exact eq_of_padBits_append_true h2 hl h
    · exact (eq_of_padBits_append_true h1 hl h.symm).symm
  calc _ ≤ ((univ : Finset (Fin c → Bool)).card : ℕ∞) :=
        encard_le_card_of_injOn _ hinj' univ fun _ _ => mem_univ _
    _ = ((2 ^ c : ℕ) : ℕ∞) := by simp

/-! ### The count -/

/-- Each accepted sequence maps to an advice string within the caps that explains it, the syncs of the windows the
horizon reaches, and its jitter code; the map is injective (`eq_of_stepwise`). -/
theorem encard_accSeqsWithin_le_constantRate {Adv : Type} {W : ℕ} (bits : Adv → Fin W → ℕ) (hbits : BitsCount bits)
    (b : Fin W → ℕ) (T : ℕ) (ss : SyncSet) (frames : List In → Adv → Fin L → List LogicalFrame)
    (op : ℕ → List (Obs L K S) → In) (n D m : ℕ) (hm : n ≤ m * T) :
    (AccSeqsWithin bits b (constantRate (S := S) T W ss frames) op n D).encard ≤
      ((2 ^ (∑ w, b w) * ss.size ^ min W m * D ^ n : ℕ) : ℕ∞) := by
  classical
  by_cases hne : Nonempty (Adv × (Fin W → ss.Elem))
  swap
  · have : AccSeqsWithin bits b (constantRate (S := S) T W ss frames) op n D = ∅ := by
      ext o; simp only [Set.mem_empty_iff_false, iff_false]
      rintro ⟨-, -, a, x, -⟩; exact hne ⟨(a, x)⟩
    rw [this, Set.encard_empty]; exact bot_le
  let p0 : Adv × (Fin W → ss.Elem) := Classical.choice hne
  let adv : List (Obs L K S) → Adv × (Fin W → ss.Elem) := fun o =>
    if h : AcceptedWithin bits b (constantRate (S := S) T W ss frames) op n o then (h.choose, h.choose_spec.choose)
    else p0
  have adv_spec : ∀ o ∈ AccSeqsWithin bits b (constantRate (S := S) T W ss frames) op n D,
      WithinCaps bits b (adv o).1 ∧ Accepted (constantRate (S := S) T W ss frames) op n o (adv o) := by
    intro o ho; simp only [adv, ho.2.2, ↓reduceDIte]; exact ho.2.2.choose_spec.choose_spec
  let cut : (Fin W → ss.Elem) → Fin (min W m) → ss.Elem := fun σ i =>
    σ ⟨i, lt_of_lt_of_le i.2 (min_le_left _ _)⟩
  let F : Finset Adv := (finite_charged bits hbits (∑ w, b w)).toFinset
  let enc : List (Obs L K S) → (Adv × (Fin (min W m) → ss.Elem)) × (Fin n → ℕ) := fun o =>
    (((adv o).1, cut (adv o).2), jitterCode o)
  have hinj : Set.InjOn enc (AccSeqsWithin bits b (constantRate (S := S) T W ss frames) op n D) := by
    intro o ho o' ho' h
    simp only [enc, Prod.mk.injEq] at h
    obtain ⟨⟨ha, hσ⟩, hj⟩ := h
    have acc := (adv_spec o ho).2
    have acc' := (adv_spec o' ho').2
    apply eq_of_stepwise ho.1 ho'.1
    intro t ht hpre
    obtain ⟨h1, hr, hs⟩ := acc t ht
    obtain ⟨h2, hr', hs'⟩ := acc' t ht
    apply Obs.ext'
    · rw [hr, hr', ingressUpTo_congr op hpre]
      funext ℓ
      simp only [constantRate, ha]
      apply expectedRow_causal
      intro w hw
      have htm : t < m * T := lt_of_lt_of_le ht hm
      have hwm : (w : ℕ) < m :=
        lt_of_le_of_lt hw (Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm]; exact htm))
      have hσw := congrFun hσ ⟨w, lt_min w.2 hwm⟩
      simp only [cut, Fin.eta] at hσw
      simp only [hσw]
    · exact jitter_eq_of_jitterCode hj ht h1 h2
    · funext κ; rw [hs κ, hs' κ]
  calc (AccSeqsWithin bits b (constantRate (S := S) T W ss frames) op n D).encard
      ≤ (((F ×ˢ (univ : Finset (Fin (min W m) → ss.Elem))) ×ˢ Fintype.piFinset fun _ : Fin n => range D).card :
          ℕ∞) := by
        apply encard_le_card_of_injOn enc hinj
        intro o ho
        simp only [enc, mem_product, mem_univ, and_true]
        refine ⟨?_, jitterCode_mem o ho.1 ho.2.1⟩
        simp only [F, Set.Finite.mem_toFinset, Set.mem_ofPred_eq]
        exact sum_le_of_withinCaps (adv_spec o ho).1
    _ ≤ ((2 ^ (∑ w, b w) * ss.size ^ min W m * D ^ n : ℕ) : ℕ∞) := by
        rw [card_product, card_product, Fintype.card_piFinset, card_univ, Fintype.card_fun, SyncSet.card_elem,
          Fintype.card_fin]
        simp only [card_range, prod_const, card_univ, Fintype.card_fin]
        exact_mod_cast Nat.mul_le_mul_right _ (Nat.mul_le_mul_right _ (card_charged_le bits hbits _))

/-- A precommitted, deterministic compiler gives every run the same compiled Program. -/
theorem compiledProgram_eq {Adv Ω : Type} {W : ℕ} (T : ℕ) (ss : SyncSet) (compilerOf : Z → Compiler L In Adv Ω)
    (env : Z → Ω) (hpre : CompilerPrecommitted compilerOf) (hdet : ∀ z, CompilerDeterministic (compilerOf z))
    (z z0 : Z) :
    compiledProgram (S := S) (W := W) T ss (compilerOf z) (env z) =
      compiledProgram (S := S) T W ss (compilerOf z0) (env z0) := by
  unfold compiledProgram
  rw [hpre z z0]
  congr 1
  funext ins a ℓ
  rw [hdet z0 a (env z) (env z0)]

theorem encard_decodableWithin_le_compiled {Adv Ω : Type} {W : ℕ} (I : Interaction L K S Z In Act Wire) (T : ℕ)
    (ss : SyncSet) (bits : Adv → Fin W → ℕ) (hbits : BitsCount bits) (b : Fin W → ℕ)
    (compilerOf : Z → Compiler L In Adv Ω) (env : Z → Ω)
    (hpre : CompilerPrecommitted compilerOf) (hdet : ∀ z, CompilerDeterministic (compilerOf z))
    (n D m : ℕ) (hm : n ≤ m * T)
    (view : ℕ → Obs L K S → Wire) (hW2 : Regenerates I.certifier view) (hJ : JitterBounded I.certifier D)
    (Zs : Set Z)
    (hacc : ∀ z ∈ Zs,
      AcceptedWithin bits b (compiledProgram (S := S) T W ss (compilerOf z) (env z)) I.op n (I.obsRun z n))
    (dec : List Wire → Z) (hdec : ∀ z ∈ Zs, dec (I.wireRun z n) = z) :
    Zs.encard ≤ ((2 ^ (∑ w, b w) * ss.size ^ min W m * D ^ n : ℕ) : ℕ∞) := by
  rcases Zs.eq_empty_or_nonempty with rfl | ⟨z0, -⟩
  · simp
  have hinj : Set.InjOn (fun z => I.obsRun z n) Zs := by
    intro z hz z' hz' h
    simp only at h
    rw [← hdec z hz, ← hdec z' hz', I.wireRun_eq view hW2, I.wireRun_eq view hW2, h]
  rw [← hinj.encard_image]
  calc ((fun z => I.obsRun z n) '' Zs).encard
      ≤ (AccSeqsWithin bits b (compiledProgram (S := S) T W ss (compilerOf z0) (env z0)) I.op n D).encard := by
        apply Set.encard_le_encard
        rintro _ ⟨z, hz, rfl⟩
        refine ⟨I.length_obsRun z n, ?_, ?_⟩
        · intro x hx
          obtain ⟨k, xs, as, rfl⟩ := I.mem_obsRun z n x hx
          exact hJ k xs as
        · rw [← compiledProgram_eq T ss compilerOf env hpre hdet z z0]; exact hacc z hz
    _ ≤ _ := encard_accSeqsWithin_le_constantRate bits hbits b T ss _ I.op n D m hm

theorem pow_sum_mul_eq_prod {W : ℕ} (T s D : ℕ) (b : Fin W → ℕ) :
    2 ^ (∑ w, b w) * s ^ min W W * D ^ (W * T) = ∏ w : Fin W, (2 ^ b w * s * D ^ T) := by
  rw [Finset.prod_mul_distrib, Finset.prod_mul_distrib, Finset.prod_pow_eq_pow_sum, Finset.prod_const,
    Finset.prod_const, Finset.card_univ, Fintype.card_fin, min_self, ← pow_mul, Nat.mul_comm T W]

theorem encard_decodableWithin_le_compiled_perWindow {Adv Ω : Type} {W : ℕ} (I : Interaction L K S Z In Act Wire)
    (T : ℕ) (ss : SyncSet) (bits : Adv → Fin W → ℕ) (hbits : BitsCount bits) (b : Fin W → ℕ)
    (compilerOf : Z → Compiler L In Adv Ω) (env : Z → Ω)
    (hpre : CompilerPrecommitted compilerOf) (hdet : ∀ z, CompilerDeterministic (compilerOf z)) (D : ℕ)
    (view : ℕ → Obs L K S → Wire) (hW2 : Regenerates I.certifier view) (hJ : JitterBounded I.certifier D)
    (Zs : Set Z)
    (hacc : ∀ z ∈ Zs, AcceptedWithin bits b (compiledProgram (S := S) T W ss (compilerOf z) (env z)) I.op (W * T)
      (I.obsRun z (W * T)))
    (dec : List Wire → Z) (hdec : ∀ z ∈ Zs, dec (I.wireRun z (W * T)) = z) :
    Zs.encard ≤ ((∏ w : Fin W, (2 ^ b w * ss.size * D ^ T) : ℕ) : ℕ∞) := by
  have h := encard_decodableWithin_le_compiled I T ss bits hbits b compilerOf env hpre hdet (W * T) D W le_rfl
    view hW2 hJ Zs hacc dec hdec
  rwa [pow_sum_mul_eq_prod] at h

end NetTiming
