import Proofs.NetworkCertifier.Schedule
import Models.NetworkCertifier.Assumptions.AdviceString
import Mathlib.Data.List.Infix
import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-!
# The leak bound: the advice string's charge

`Guarantees.AdviceString`. The count is Theorem 3's for the constant-rate Program (`Schedule.lean`) with the content
advice replaced by its window segments: within the caps `b`, window `w`'s segments number at most `2^(b w)`
(`BitsCount`), and the segments decode (`SegmentsDecode`), so the advice strings within the caps number at most
`2^(Σ_w b w)`. A precommitted, deterministic compiler gives every run the same Program, so the composed statement is
the count through the wire. The two codec lemmas show that `BitsCount` holds for a self-delimiting code charged its
length and for any code charged its length plus one.
-/

namespace NetTiming

open Finset

variable {L K S : ℕ} {Z In Act Wire : Type}

/-! ### Window segments within a charge -/

theorem finite_segs {Adv : Type} {W : ℕ} (code : AdviceCodec Adv W) (hbits : BitsCount code) (w : Fin W) (c : ℕ) :
    {s | ∃ a, code.seg a w = s ∧ code.bits a w ≤ c}.Finite :=
  Set.finite_of_encard_le_coe (hbits w c)

theorem card_segs_le {Adv : Type} {W : ℕ} (code : AdviceCodec Adv W) (hbits : BitsCount code) (w : Fin W) (c : ℕ) :
    (finite_segs code hbits w c).toFinset.card ≤ 2 ^ c := by
  have h := hbits w c
  rw [Set.Finite.encard_eq_coe_toFinset_card (finite_segs code hbits w c)] at h
  exact_mod_cast h

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

/-- **A self-delimiting code charged at least its length meets `BitsCount`.** If no window segment is a proper
prefix of another and each is charged at least its length, then padding to `c` bits tells the segments charged at most
`c` apart, so there are at most `2^c` of them. -/
theorem bitsCount_of_prefixFree {Adv : Type} {W : ℕ} (code : AdviceCodec Adv W)
    (hpf : ∀ w a a', code.seg a w <+: code.seg a' w → code.seg a w = code.seg a' w)
    (hlen : ∀ a w, (code.seg a w).length ≤ code.bits a w) : BitsCount code := by
  intro w c
  have hinj : Set.InjOn (padBits c) {s | ∃ a, code.seg a w = s ∧ code.bits a w ≤ c} := by
    rintro _ ⟨a, rfl, ha⟩ _ ⟨a', rfl, ha'⟩ h
    rcases le_total (code.seg a w).length (code.seg a' w).length with hl | hl
    · exact hpf w a a' (prefix_of_padBits_eq hl ((hlen a' w).trans ha') h)
    · exact (hpf w a' a (prefix_of_padBits_eq hl ((hlen a w).trans ha) h.symm)).symm
  calc _ ≤ ((univ : Finset (Fin c → Bool)).card : ℕ∞) :=
        encard_le_card_of_injOn _ hinj univ fun _ _ => mem_univ _
    _ = ((2 ^ c : ℕ) : ℕ∞) := by simp

/-- **Any code charged its length plus one meets `BitsCount`**: the segments charged at most `c` bits are bit strings
shorter than `c`, and there are `2^c − 1` of those. A public cap `ℓ` on every segment, charged `ℓ + 1`, is this
case. -/
theorem bitsCount_of_lengthSucc {Adv : Type} {W : ℕ} (code : AdviceCodec Adv W)
    (hlen : ∀ a w, (code.seg a w).length + 1 ≤ code.bits a w) : BitsCount code := by
  intro w c
  let mark : List Bool → Fin c → Bool := fun l => padBits c (l ++ [true])
  have hinj : Set.InjOn mark {s | ∃ a, code.seg a w = s ∧ code.bits a w ≤ c} := by
    have key : ∀ l l' : List Bool, l.length < c → l'.length < c → l.length ≤ l'.length → mark l = mark l' →
        l = l' := by
      intro l l' hl hl' hle h
      have hpre := prefix_of_padBits_eq (by simp; omega) (by simp; omega) h
      rcases Nat.lt_or_eq_of_le hle with hlt | heq
      · exfalso
        have := congrFun h ⟨l'.length, hl'⟩
        simp only [mark, padBits] at this
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
    rintro _ ⟨a, rfl, ha⟩ _ ⟨a', rfl, ha'⟩ h
    have h1 := (hlen a w).trans ha
    have h2 := (hlen a' w).trans ha'
    rcases le_total (code.seg a w).length (code.seg a' w).length with hl | hl
    · exact key _ _ (by omega) (by omega) hl h
    · exact (key _ _ (by omega) (by omega) hl h.symm).symm
  calc _ ≤ ((univ : Finset (Fin c → Bool)).card : ℕ∞) :=
        encard_le_card_of_injOn _ hinj univ fun _ _ => mem_univ _
    _ = ((2 ^ c : ℕ) : ℕ∞) := by simp

/-! ### The count -/

/-- Each accepted sequence maps to the window segments of an advice string within the caps that explains it, the syncs
of the windows the horizon reaches, and its jitter code; the map is injective (`eq_of_stepwise`). -/
theorem encard_accSeqsWithin_le_constantRate {Adv : Type} {W : ℕ} (code : AdviceCodec Adv W)
    (hseg : SegmentsDecode code) (hbits : BitsCount code) (b : Fin W → ℕ) (T : ℕ) (ss : SyncSet)
    (frames : List In → Adv → Fin L → List LogicalFrame) (op : ℕ → List (Obs L K S) → In) (n D m : ℕ)
    (hm : n ≤ m * T) :
    (AccSeqsWithin code b (constantRate (S := S) T W ss frames) op n D).encard ≤
      ((2 ^ (∑ w, b w) * ss.size ^ min W m * D ^ n : ℕ) : ℕ∞) := by
  classical
  by_cases hne : Nonempty (Adv × (Fin W → ss.Elem))
  swap
  · have : AccSeqsWithin code b (constantRate (S := S) T W ss frames) op n D = ∅ := by
      ext o; simp only [Set.mem_empty_iff_false, iff_false]
      rintro ⟨-, -, a, x, -⟩; exact hne ⟨(a, x)⟩
    rw [this, Set.encard_empty]; exact bot_le
  let p0 : Adv × (Fin W → ss.Elem) := Classical.choice hne
  let adv : List (Obs L K S) → Adv × (Fin W → ss.Elem) := fun o =>
    if h : AcceptedWithin code b (constantRate (S := S) T W ss frames) op n o then (h.choose, h.choose_spec.choose)
    else p0
  have adv_spec : ∀ o ∈ AccSeqsWithin code b (constantRate (S := S) T W ss frames) op n D,
      code.Within b (adv o).1 ∧ Accepted (constantRate (S := S) T W ss frames) op n o (adv o) := by
    intro o ho; simp only [adv, ho.2.2, ↓reduceDIte]; exact ho.2.2.choose_spec.choose_spec
  let cut : (Fin W → ss.Elem) → Fin (min W m) → ss.Elem := fun σ i =>
    σ ⟨i, lt_of_lt_of_le i.2 (min_le_left _ _)⟩
  let F : Fin W → Finset (List Bool) := fun w => (finite_segs code hbits w (b w)).toFinset
  let enc : List (Obs L K S) → ((Fin W → List Bool) × (Fin (min W m) → ss.Elem)) × (Fin n → ℕ) := fun o =>
    ((code.seg (adv o).1, cut (adv o).2), jitterCode o)
  have hinj : Set.InjOn enc (AccSeqsWithin code b (constantRate (S := S) T W ss frames) op n D) := by
    intro o ho o' ho' h
    simp only [enc, Prod.mk.injEq] at h
    obtain ⟨⟨hc, hσ⟩, hj⟩ := h
    have ha : (adv o).1 = (adv o').1 := hseg hc
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
  have hF : ∏ w, (F w).card ≤ 2 ^ ∑ w, b w := by
    rw [← Finset.prod_pow_eq_pow_sum]
    exact Finset.prod_le_prod fun w _ => card_segs_le code hbits w (b w)
  calc (AccSeqsWithin code b (constantRate (S := S) T W ss frames) op n D).encard
      ≤ (((Fintype.piFinset F ×ˢ (univ : Finset (Fin (min W m) → ss.Elem))) ×ˢ
          Fintype.piFinset fun _ : Fin n => range D).card : ℕ∞) := by
        apply encard_le_card_of_injOn enc hinj
        intro o ho
        simp only [enc, mem_product, mem_univ, and_true]
        refine ⟨?_, jitterCode_mem o ho.1 ho.2.1⟩
        rw [Fintype.mem_piFinset]; intro w
        simp only [F, Set.Finite.mem_toFinset, Set.mem_ofPred_eq]
        exact ⟨(adv o).1, rfl, (adv_spec o ho).1 w⟩
    _ ≤ ((2 ^ (∑ w, b w) * ss.size ^ min W m * D ^ n : ℕ) : ℕ∞) := by
        rw [card_product, card_product, Fintype.card_piFinset, Fintype.card_piFinset, card_univ,
          Fintype.card_fun, SyncSet.card_elem, Fintype.card_fin]
        simp only [card_range, prod_const, card_univ, Fintype.card_fin]
        exact_mod_cast Nat.mul_le_mul_right _ (Nat.mul_le_mul_right _ hF)

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
    (ss : SyncSet) (code : AdviceCodec Adv W) (hseg : SegmentsDecode code) (hbits : BitsCount code)
    (b : Fin W → ℕ) (compilerOf : Z → Compiler L In Adv Ω) (env : Z → Ω)
    (hpre : CompilerPrecommitted compilerOf) (hdet : ∀ z, CompilerDeterministic (compilerOf z))
    (n D m : ℕ) (hm : n ≤ m * T)
    (view : ℕ → Obs L K S → Wire) (hW2 : Regenerates I.certifier view) (hJ : JitterBounded I.certifier D)
    (Zs : Set Z)
    (hacc : ∀ z ∈ Zs,
      AcceptedWithin code b (compiledProgram (S := S) T W ss (compilerOf z) (env z)) I.op n (I.obsRun z n))
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
      ≤ (AccSeqsWithin code b (compiledProgram (S := S) T W ss (compilerOf z0) (env z0)) I.op n D).encard := by
        apply Set.encard_le_encard
        rintro _ ⟨z, hz, rfl⟩
        refine ⟨I.length_obsRun z n, ?_, ?_⟩
        · intro x hx
          obtain ⟨k, xs, as, rfl⟩ := I.mem_obsRun z n x hx
          exact hJ k xs as
        · rw [← compiledProgram_eq T ss compilerOf env hpre hdet z z0]; exact hacc z hz
    _ ≤ _ := encard_accSeqsWithin_le_constantRate code hseg hbits b T ss _ I.op n D m hm

theorem pow_sum_mul_eq_prod {W : ℕ} (T s D : ℕ) (b : Fin W → ℕ) :
    2 ^ (∑ w, b w) * s ^ min W W * D ^ (W * T) = ∏ w : Fin W, (2 ^ b w * s * D ^ T) := by
  rw [Finset.prod_mul_distrib, Finset.prod_mul_distrib, Finset.prod_pow_eq_pow_sum, Finset.prod_const,
    Finset.prod_const, Finset.card_univ, Fintype.card_fin, min_self, ← pow_mul, Nat.mul_comm T W]

theorem encard_decodableWithin_le_compiled_perWindow {Adv Ω : Type} {W : ℕ} (I : Interaction L K S Z In Act Wire)
    (T : ℕ) (ss : SyncSet) (code : AdviceCodec Adv W) (hseg : SegmentsDecode code) (hbits : BitsCount code)
    (b : Fin W → ℕ) (compilerOf : Z → Compiler L In Adv Ω) (env : Z → Ω)
    (hpre : CompilerPrecommitted compilerOf) (hdet : ∀ z, CompilerDeterministic (compilerOf z)) (D : ℕ)
    (view : ℕ → Obs L K S → Wire) (hW2 : Regenerates I.certifier view) (hJ : JitterBounded I.certifier D)
    (Zs : Set Z)
    (hacc : ∀ z ∈ Zs, AcceptedWithin code b (compiledProgram (S := S) T W ss (compilerOf z) (env z)) I.op (W * T)
      (I.obsRun z (W * T)))
    (dec : List Wire → Z) (hdec : ∀ z ∈ Zs, dec (I.wireRun z (W * T)) = z) :
    Zs.encard ≤ ((∏ w : Fin W, (2 ^ b w * ss.size * D ^ T) : ℕ) : ℕ∞) := by
  have h := encard_decodableWithin_le_compiled I T ss code hseg hbits b compilerOf env hpre hdet (W * T) D W le_rfl
    view hW2 hJ Zs hacc dec hdec
  rwa [pow_sum_mul_eq_prod] at h

end NetTiming
