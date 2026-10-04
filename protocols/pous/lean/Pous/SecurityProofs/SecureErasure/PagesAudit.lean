import Mathlib.Data.Fintype.CardEmbedding
import Pous.SecurityProofs.SecureErasure.Digest
import Pous.SecurityProofs.SecureErasure.Reduction

/-!
# The page fill's audit at `D = 0`, with distinct byte reads

`Pous.Guarantees.PageFillAuditSecure`. A round bounded by `D = 0` makes no query, so its answer and next state are a
function of the round `j`, its state `σ` and its index `i` alone (`run_eq_of_bounded_zero`). Call the fill `c` **bad**
if some round `j < s` and some state `σ` predict at least `B − w` of its bytes right. Then:

- **bad fills are rare** (`pr_agree_ge_le`, `pr_page_bad_le`): a fixed prediction agrees with a uniform fill on at least
  `B − w` bytes with probability at most `C(B, B − w) · 2^(-8·(B − w))`, and there are `s · 2^S` pairs `(j, σ)`;
- **the fill is uniform** (`page_unif`): the pages are independent uniform permutations of the zero page under distinct
  tweaks, so the map from the primitive to the fill has fibres of one size (`ev_fibre_le`, by composing each page's
  permutation with a transposition);
- **on a fill that is not bad**, each round answers at most `B − w` indices right, whatever its state, so at most
  `(B − w)^s` sequences of `s` indices are all answered right (`card_accL_le`, a tree with at most `B − w` children per
  node), out of `B·(B − 1)⋯(B − s + 1)` sequences of distinct indices.
-/

namespace PousErase

open Pous PousThm1 PousDigest Pous.PageFill

attribute [local instance] instFintypePageOmega
open scoped ENNReal

variable {M : IdealModel}

/-! ## Rounds that make no query -/

theorem run_eq_of_bounded_zero {α : Type} {p : Prog M α} {Q : ℕ} (h : p.Bounded 0 Q) (f g : M.Qry → M.Ans) :
    p.run f = p.run g := by
  cases p with
  | ret a => rfl
  | round qs k => exact absurd (h (Classical.arbitrary M.Ω)).1 (by simp [Prog.depth])

theorem seqAnswers_congr_zero {ℓ S B Q : ℕ} (step : ℕ → Bits S → Fin B → Prog M (Bits ℓ × Bits S))
    (h : ∀ j σ i, (step j σ i).Bounded 0 Q) (f g : M.Qry → M.Ans) :
    ∀ (j : ℕ) (is : List (Fin B)) (σ : Bits S), seqAnswers f step j is σ = seqAnswers g step j is σ := by
  intro j is
  induction is generalizing j with
  | nil => intro _; rfl
  | cons i is ih =>
    intro σ
    simp only [seqAnswers, run_eq_of_bounded_zero (h j σ i) f g, ih]

/-- The run of pure rounds `nx` is right on every index of the list. -/
def accL {ℓ S B : ℕ} (nx : ℕ → Bits S → Fin B → Bits ℓ × Bits S) (c : Fin B → Bits ℓ) :
    ℕ → Bits S → List (Fin B) → Prop
  | _, _, [] => True
  | j, σ, i :: is => (nx j σ i).1 = c i ∧ accL nx c (j + 1) (nx j σ i).2 is

theorem seqAnswers_eq_iff {ℓ S B : ℕ} (f : M.Qry → M.Ans) (step : ℕ → Bits S → Fin B → Prog M (Bits ℓ × Bits S))
    (c : Fin B → Bits ℓ) :
    ∀ (j : ℕ) (is : List (Fin B)) (σ : Bits S),
      seqAnswers f step j is σ = is.map c ↔ accL (fun j σ i => (step j σ i).run f) c j σ is := by
  intro j is
  induction is generalizing j with
  | nil => intro _; simp [seqAnswers, accL]
  | cons i is ih =>
    intro σ
    simp only [seqAnswers, List.map_cons, List.cons.injEq, accL, ih]

/-! ## Counting the right answers -/

theorem card_filter_equiv {α β : Type} [Fintype α] [Fintype β] (e : α ≃ β) (P : β → Prop) [DecidablePred P] :
    (Finset.univ.filter fun a => P (e a)).card = (Finset.univ.filter P).card := by
  rw [← Fintype.card_subtype, ← Fintype.card_subtype]
  exact Fintype.card_congr (e.subtypeEquiv fun _ => Iff.rfl)

open Classical in
/-- **The tree count.** If every round from `j` to `j + n − 1` answers at most `g` indices right from every state, at
most `g^n` sequences of `n` indices are all answered right. -/
theorem card_accL_le {ℓ S B : ℕ} (nx : ℕ → Bits S → Fin B → Bits ℓ × Bits S) (c : Fin B → Bits ℓ) (g : ℕ) :
    ∀ (n j : ℕ) (σ : Bits S),
      (∀ j', j ≤ j' → j' < j + n → ∀ σ', (Finset.univ.filter fun i => (nx j' σ' i).1 = c i).card ≤ g) →
      (Finset.univ.filter fun f : Fin n → Fin B => accL nx c j σ (List.ofFn f)).card ≤ g ^ n := by
  classical
  intro n
  induction n with
  | zero =>
    intro j σ _
    refine (Finset.card_le_univ _).trans ?_
    simp
  | succ n ih =>
    intro j σ hg
    rw [← card_filter_equiv (Fin.consEquiv fun _ => Fin B), card_filter_prod]
    calc ∑ i, (Finset.univ.filter fun f : Fin n → Fin B =>
            accL nx c j σ (List.ofFn ((Fin.consEquiv fun _ => Fin B) (i, f)))).card
        ≤ ∑ i, if (nx j σ i).1 = c i then g ^ n else 0 := by
          refine Finset.sum_le_sum fun i _ => ?_
          have hof : ∀ f : Fin n → Fin B, List.ofFn ((Fin.consEquiv fun _ => Fin B) (i, f)) = i :: List.ofFn f := by
            intro f
            simp [Fin.consEquiv, List.ofFn_succ]
          simp only [hof, accL]
          split_ifs with hi
          · simp only [hi, true_and]
            exact ih (j + 1) _ fun j' h1 h2 σ' => hg j' (by omega) (by omega) σ'
          · simp [hi]
      _ = (Finset.univ.filter fun i => (nx j σ i).1 = c i).card * g ^ n := by
          rw [← Finset.sum_filter, Finset.sum_const, smul_eq_mul]
      _ ≤ g * g ^ n := Nat.mul_le_mul_right _ (hg j le_rfl (by omega) σ)
      _ = g ^ (n + 1) := (pow_succ' g n).symm

/-- **The agreement tail.** At most `C(B, t) · |X|^(B − t)` words agree with a fixed word `a` in at least `t` places. -/
theorem card_agree_ge_le {B : ℕ} {X : Type} [Fintype X] [DecidableEq X] (a : Fin B → X) (t : ℕ) :
    (Finset.univ.filter fun c : Fin B → X => t ≤ (Finset.univ.filter fun i => a i = c i).card).card ≤
      B.choose t * Fintype.card X ^ (B - t) := by
  classical
  have hsub : (Finset.univ.filter fun c : Fin B → X => t ≤ (Finset.univ.filter fun i => a i = c i).card) ⊆
      (Finset.univ.powersetCard t).biUnion fun T => Finset.univ.filter fun c : Fin B → X => ∀ i ∈ T, c i = a i := by
    intro c hc
    obtain ⟨T, hT, hTc⟩ := Finset.exists_subset_card_eq (Finset.mem_filter.1 hc).2
    refine Finset.mem_biUnion.2 ⟨T, Finset.mem_powersetCard.2 ⟨Finset.subset_univ _, hTc⟩, ?_⟩
    refine Finset.mem_filter.2 ⟨Finset.mem_univ _, fun i hi => ?_⟩
    exact ((Finset.mem_filter.1 (hT hi)).2).symm
  have hfix : ∀ T ∈ Finset.univ.powersetCard t,
      (Finset.univ.filter fun c : Fin B → X => ∀ i ∈ T, c i = a i).card ≤ Fintype.card X ^ (B - t) := by
    intro T hT
    have hTc : T.card = t := (Finset.mem_powersetCard.1 hT).2
    let r : (Fin B → X) → (↥(Tᶜ) → X) := fun c i => c i
    calc (Finset.univ.filter fun c : Fin B → X => ∀ i ∈ T, c i = a i).card
        ≤ (Finset.univ : Finset (↥(Tᶜ) → X)).card := by
          refine Finset.card_le_card_of_injOn r (fun _ _ => Finset.mem_coe.2 (Finset.mem_univ _)) ?_
          intro c hc c' hc' hcc
          funext i
          by_cases hi : i ∈ T
          · rw [(Finset.mem_filter.1 hc).2 i hi, (Finset.mem_filter.1 hc').2 i hi]
          · exact congrFun hcc ⟨i, Finset.mem_compl.2 hi⟩
      _ = Fintype.card X ^ (B - t) := by
          rw [Finset.card_univ, Fintype.card_fun, Fintype.card_coe, Finset.card_compl, Fintype.card_fin, hTc]
  calc _ ≤ ((Finset.univ.powersetCard t).biUnion fun T =>
          Finset.univ.filter fun c : Fin B → X => ∀ i ∈ T, c i = a i).card := Finset.card_le_card hsub
    _ ≤ ∑ T ∈ Finset.univ.powersetCard t, (Finset.univ.filter fun c : Fin B → X => ∀ i ∈ T, c i = a i).card :=
        Finset.card_biUnion_le
    _ ≤ ∑ _T ∈ Finset.univ.powersetCard t, Fintype.card X ^ (B - t) := Finset.sum_le_sum hfix
    _ = B.choose t * Fintype.card X ^ (B - t) := by
        rw [Finset.sum_const, Finset.card_powersetCard, Finset.card_univ, Fintype.card_fin, smul_eq_mul]

/-- A uniform fill of `B` blocks of `ℓ` bits agrees with a fixed one on at least `B − w` blocks with probability at most
`C(B, w) / (2^ℓ)^(B − w)`. -/
theorem pr_agree_ge_le {B ℓ : ℕ} (a : Fin B → Bits ℓ) (w : ℕ) (hw : w ≤ B) :
    pr (fun c : Fin B → Bits ℓ => B - w ≤ (Finset.univ.filter fun i => a i = c i).card) ≤
      (B.choose w : ℝ≥0∞) / (2 ^ ℓ) ^ (B - w) := by
  classical
  have h := card_agree_ge_le a (B - w)
  rw [Nat.choose_symm hw, Nat.sub_sub_self hw, card_bits] at h
  have hc := pr_le_of_card (fun c : Fin B → Bits ℓ => B - w ≤ (Finset.univ.filter fun i => a i = c i).card)
    (B.choose w) ((2 ^ ℓ) ^ (B - w)) ((2 ^ ℓ) ^ w) (by positivity)
    (by rw [Fintype.card_fun, Fintype.card_fin, card_bits, ← pow_add, Nat.sub_add_cancel hw]) h
  simpa only [Nat.cast_pow, Nat.cast_ofNat] using hc

/-! ## The fill is uniform -/

/-- `pr` sees an event of `f a` as an event of a uniform point when every fibre of `f` has the same size. -/
theorem pr_comp_of_fibres {α β : Type} [Fintype α] [Fintype β] [Nonempty α] [DecidableEq β] (f : α → β)
    (hf : ∀ b b', (Finset.univ.filter fun a => f a = b).card ≤ (Finset.univ.filter fun a => f a = b').card)
    (E : β → Prop) : pr (fun a => E (f a)) = pr E := by
  classical
  obtain ⟨a₀⟩ := ‹Nonempty α›
  set k := (Finset.univ.filter fun a => f a = f a₀).card with hk_def
  have hk : ∀ b, (Finset.univ.filter fun a => f a = b).card = k := fun b => le_antisymm (hf _ _) (hf _ _)
  have hk0 : k ≠ 0 := Finset.card_ne_zero.2 ⟨a₀, by simp⟩
  have hα : Fintype.card α = Fintype.card β * k := by
    rw [← Finset.card_univ, Finset.card_eq_sum_card_fiberwise (f := f) (t := Finset.univ)
      (fun _ _ => Finset.mem_coe.2 (Finset.mem_univ _))]
    simp only [hk, Finset.sum_const, Finset.card_univ, smul_eq_mul]
  have hE : (Finset.univ.filter fun a => E (f a)).card = (Finset.univ.filter E).card * k := by
    rw [Finset.card_eq_sum_card_fiberwise (f := f) (t := Finset.univ.filter E)
      (fun a ha => Finset.mem_coe.2 (by simpa using ha))]
    rw [← smul_eq_mul, ← Finset.sum_const]
    refine Finset.sum_congr rfl fun b hb => ?_
    rw [← hk b, Finset.filter_filter]
    congr 1
    ext a
    simp only [Finset.mem_filter, Finset.mem_univ, true_and]
    exact ⟨fun h => h.2, fun h => ⟨h ▸ (Finset.mem_filter.1 hb).2, h⟩⟩
  rw [pr_def, pr_def, hE, hα]
  push_cast
  exact ENNReal.mul_div_mul_right _ _ (by exact_mod_cast hk0) (ENNReal.natCast_ne_top k)

theorem tweak_injective (K : Bits 256) {P : ℕ} (hP : P ≤ 2 ^ 64) :
    Function.Injective fun p : Fin P => tweak K p := by
  intro p q hpq
  have hbit : ∀ i < 64, (p : ℕ).testBit i = (q : ℕ).testBit i := by
    intro i hi
    have := congrFun hpq ⟨256 + i, by omega⟩
    simpa [tweak] using this
  have hp : (p : ℕ) < 2 ^ 64 := lt_of_lt_of_le p.2 hP
  have hq : (q : ℕ) < 2 ^ 64 := lt_of_lt_of_le q.2 hP
  refine Fin.ext (Nat.eq_of_testBit_eq fun i => ?_)
  by_cases hi : i < 64
  · exact hbit i hi
  · have h64 : (2 : ℕ) ^ 64 ≤ 2 ^ i := Nat.pow_le_pow_right (by norm_num) (by omega)
    rw [Nat.testBit_lt_two_pow (lt_of_lt_of_le hp h64), Nat.testBit_lt_two_pow (lt_of_lt_of_le hq h64)]

/-- The `P` pages under key `K`. -/
def ev {pb : ℕ} (K : Bits 256) (P : ℕ) (ω : PageOmega pb) : Fin P → Page pb := fun p => page K ω p

/-- Composing each page's permutation with the transposition of `c p` and `c' p` maps the primitives whose pages are
`c` into those whose pages are `c'`, injectively. -/
theorem ev_fibre_le {pb P : ℕ} (K : Bits 256) (hP : P ≤ 2 ^ 64) (c c' : Fin P → Page pb) :
    (Finset.univ.filter fun ω : PageOmega pb => ev K P ω = c).card ≤
      (Finset.univ.filter fun ω : PageOmega pb => ev K P ω = c').card := by
  have hτ := tweak_injective K hP
  let g : Tweak → Equiv.Perm (Page pb) :=
    Function.extend (fun p : Fin P => tweak K p) (fun p => Equiv.swap (c p) (c' p)) fun _ => 1
  refine Finset.card_le_card_of_injOn (fun ω T => g T * ω T) ?_ ?_
  · intro ω hω
    rw [Finset.coe_filter] at hω ⊢
    have hω : ev K P ω = c := hω.2
    refine ⟨Finset.mem_univ _, funext fun p => ?_⟩
    have hp : ω (tweak K p) (zeroPage pb) = c p := congrFun hω p
    show (g (tweak K p) * ω (tweak K p)) (zeroPage pb) = c' p
    rw [Equiv.Perm.mul_apply, hp]
    simp only [g, hτ.extend_apply (f := fun p : Fin P => tweak K p), Equiv.swap_apply_left]
  · intro ω _ ω' _ h
    funext T
    exact mul_left_cancel (congrFun h T)

/-- The fill as `P` pages, read as `P·pb` bytes: byte `i` is byte `i mod pb` of page `⌊i / pb⌋`. -/
def split (pb P : ℕ) : (Fin P → Page pb) ≃ (Fin (P * pb) → Bits 8) where
  toFun v i := byte (v (finProdFinEquiv.symm i).1) (finProdFinEquiv.symm i).2
  invFun c p bit := c (finProdFinEquiv (p, (finProdFinEquiv.symm bit).1)) (finProdFinEquiv.symm bit).2
  left_inv v := by
    funext p bit
    simp only [byte, Equiv.symm_apply_apply, Prod.mk.eta, Equiv.apply_symm_apply]
  right_inv c := by
    funext i k
    simp only [byte, Equiv.symm_apply_apply, Prod.mk.eta, Equiv.apply_symm_apply]

/-- **The page fill is uniform**: an event of the code has the probability it has for a uniform code. -/
theorem page_unif (pb P : ℕ) [NeZero pb] [NeZero P] (hP : P ≤ 2 ^ 64) (K : Bits 256) (W : Bits 0)
    (E : (Fin (P * pb) → Bits 8) → Prop) :
    pr (fun x : PageOmega pb × Unit => E ((pageScheme pb P K).enc W x.1 x.2).2) = pr E := by
  classical
  calc pr (fun x : PageOmega pb × Unit => E ((pageScheme pb P K).enc W x.1 x.2).2)
      = pr (fun ω : PageOmega pb => E (split pb P (ev K P ω))) := pr_fst (fun ω : PageOmega pb => E (split pb P (ev K P ω)))
    _ = pr (fun v : Fin P → Page pb => E (split pb P v)) :=
        pr_comp_of_fibres (ev K P) (ev_fibre_le K hP) (fun v => E (split pb P v))
    _ = pr E := pr_equiv (split pb P) E

/-! ## The audit -/

/-- Every distinct-read sequence `I` with all answers right is a sequence of indices with all answers right. -/
theorem pr_accL_distinct_le {ℓ S B : ℕ} (nx : ℕ → Bits S → Fin B → Bits ℓ × Bits S) (c : Fin B → Bits ℓ) (s g : ℕ)
    (σ : Bits S) (hg : ∀ j < s, ∀ σ', (Finset.univ.filter fun i => (nx j σ' i).1 = c i).card ≤ g) :
    pr (fun I : Fin s ↪ Fin B => accL nx c 0 σ (List.ofFn I)) ≤ (g : ℝ≥0∞) ^ s / (B.descFactorial s : ℝ≥0∞) := by
  classical
  rw [pr_def, Fintype.card_embedding_eq, Fintype.card_fin, Fintype.card_fin]
  gcongr
  have hN : (Finset.univ.filter fun I : Fin s ↪ Fin B => accL nx c 0 σ (List.ofFn I)).card ≤ g ^ s := calc
    (Finset.univ.filter fun I : Fin s ↪ Fin B => accL nx c 0 σ (List.ofFn I)).card
      ≤ (Finset.univ.filter fun f : Fin s → Fin B => accL nx c 0 σ (List.ofFn f)).card := by
        refine Finset.card_le_card_of_injOn (fun I => (I : Fin s → Fin B)) ?_ DFunLike.coe_injective.injOn
        intro I hI
        rw [Finset.coe_filter] at hI ⊢
        exact ⟨Finset.mem_univ _, hI.2⟩
    _ ≤ g ^ s := by
        convert card_accL_le nx c g s 0 σ fun j' _ h2 σ' => hg j' (by omega) σ'
  exact_mod_cast hN

/-- **The sequential audit at `D = 0` with distinct reads, for a uniform fill.** If the public parameters are fixed and
the code is uniform, every `S`-bit prover is accepted with probability at most
`s · 2^S · C(B, w) / (2^ℓ)^(B − w) + (B − w)^s / (B·(B − 1)⋯(B − s + 1))`. -/
theorem audit_zero_distinct (sch : Scheme M) (hpp : ∀ W ω r ω' r', (sch.enc W ω r).1 = (sch.enc W ω' r').1)
    (hunif : ∀ (W : Bits sch.n) (E : sch.Code → Prop),
      pr (fun x : M.Ω × sch.Coins => E (sch.enc W x.1 x.2).2) = pr E)
    (S Q s w : ℕ) (hw : w ≤ sch.B) (hs : s ≤ sch.B) :
    AuditSecureDraw sch (distinctReads sch s) S 0 Q 0
      ((s : ℝ≥0∞) * 2 ^ S * (sch.B.choose w : ℝ≥0∞) / (2 ^ sch.ℓ) ^ (sch.B - w) +
        ((sch.B - w : ℕ) : ℝ≥0∞) ^ s / (sch.B.descFactorial s : ℝ≥0∞)) := by
  classical
  intro W A₁ A₂
  rw [zero_add]
  have : Nonempty (Fin s ↪ Fin sch.B) := Function.Embedding.nonempty_of_card_le (by simpa using hs)
  obtain ⟨ω₀⟩ := (inferInstance : Nonempty M.Ω)
  obtain ⟨r₀⟩ := (inferInstance : Nonempty sch.Coins)
  set pp₀ := (sch.enc W ω₀ r₀).1
  set nx : ℕ → Bits S → Fin sch.B → Bits sch.ℓ × Bits S := fun j σ i => (A₂.step W pp₀ j σ i).run (M.answer ω₀)
  set code : M.Ω × sch.Coins → sch.Code := fun a => (sch.enc W a.1 a.2).2
  set bad : sch.Code → Prop := fun c =>
    ∃ j ∈ Finset.range s, ∃ σ ∈ (Finset.univ : Finset (Bits S)),
      sch.B - w ≤ (Finset.univ.filter fun i => (nx j σ i).1 = c i).card
  have hacc : ∀ x : OutcomeDraw sch (Fin s ↪ Fin sch.B), seqAcceptsDraw sch (distinctReads sch s) W A₁ A₂ x →
      accL nx (code (x.1, x.2.1)) 0 (A₁ W pp₀ (code (x.1, x.2.1)) x.1) (List.ofFn x.2.2) := by
    intro x hx
    have hpp₀ : (sch.enc W x.1 x.2.1).1 = pp₀ := hpp _ _ _ _ _
    simp only [seqAcceptsDraw, distinctReads, hpp₀] at hx
    rw [seqAnswers_congr_zero _ (A₂.bounded W pp₀) _ (M.answer ω₀)] at hx
    rw [show (List.ofFn fun j => (sch.enc W x.1 x.2.1).2 (x.2.2 j)) =
      (List.ofFn x.2.2).map (sch.enc W x.1 x.2.1).2 from List.map_ofFn.symm] at hx
    exact (seqAnswers_eq_iff _ _ _ _ _ _).1 hx
  have hbad : pr bad ≤ (s : ℝ≥0∞) * 2 ^ S * ((sch.B.choose w : ℝ≥0∞) / (2 ^ sch.ℓ) ^ (sch.B - w)) := by
    refine (pr_exists_le _ _).trans ?_
    calc ∑ j ∈ Finset.range s, pr (fun c : sch.Code => ∃ σ ∈ (Finset.univ : Finset (Bits S)),
          sch.B - w ≤ (Finset.univ.filter fun i => (nx j σ i).1 = c i).card)
        ≤ ∑ _j ∈ Finset.range s, ∑ _σ ∈ (Finset.univ : Finset (Bits S)),
            (sch.B.choose w : ℝ≥0∞) / (2 ^ sch.ℓ) ^ (sch.B - w) := by
          refine Finset.sum_le_sum fun j _ => (pr_exists_le _ _).trans (Finset.sum_le_sum fun σ _ => ?_)
          exact pr_agree_ge_le (fun i => (nx j σ i).1) w hw
      _ = (s : ℝ≥0∞) * 2 ^ S * ((sch.B.choose w : ℝ≥0∞) / (2 ^ sch.ℓ) ^ (sch.B - w)) := by
          rw [Finset.sum_const, Finset.sum_const, Finset.card_range, Finset.card_univ, card_bits, nsmul_eq_mul,
            nsmul_eq_mul, mul_assoc]
          push_cast
          ring
  have hgood : ∀ a : M.Ω × sch.Coins, ¬ bad (code a) →
      pr (fun I : Fin s ↪ Fin sch.B => accL nx (code a) 0 (A₁ W pp₀ (code a) a.1) (List.ofFn I)) ≤
        ((sch.B - w : ℕ) : ℝ≥0∞) ^ s / (sch.B.descFactorial s : ℝ≥0∞) := by
    intro a ha
    refine pr_accL_distinct_le nx (code a) s (sch.B - w) _ fun j hj σ' => ?_
    by_contra h
    exact ha ⟨j, Finset.mem_range.2 hj, σ', Finset.mem_univ _, by omega⟩
  calc pr (seqAcceptsDraw sch (distinctReads sch s) W A₁ A₂)
      ≤ pr (fun x : OutcomeDraw sch (Fin s ↪ Fin sch.B) =>
          accL nx (code (x.1, x.2.1)) 0 (A₁ W pp₀ (code (x.1, x.2.1)) x.1) (List.ofFn x.2.2)) := pr_mono hacc
    _ = pr (fun y : (M.Ω × sch.Coins) × (Fin s ↪ Fin sch.B) =>
          accL nx (code y.1) 0 (A₁ W pp₀ (code y.1) y.1.1) (List.ofFn y.2)) :=
        (pr_equiv (Equiv.prodAssoc _ _ _) (fun x : OutcomeDraw sch (Fin s ↪ Fin sch.B) =>
          accL nx (code (x.1, x.2.1)) 0 (A₁ W pp₀ (code (x.1, x.2.1)) x.1) (List.ofFn x.2.2))).symm
    _ ≤ pr (fun a => bad (code a)) + ((sch.B - w : ℕ) : ℝ≥0∞) ^ s / (sch.B.descFactorial s : ℝ≥0∞) :=
        pr_prod_le (fun (a : M.Ω × sch.Coins) (I : Fin s ↪ Fin sch.B) =>
          accL nx (code a) 0 (A₁ W pp₀ (code a) a.1) (List.ofFn I)) (fun a => bad (code a)) _ hgood
    _ ≤ _ := by
        rw [hunif W bad]
        exact add_le_add_left (hbad.trans (le_of_eq (mul_div_assoc _ _ _).symm)) _

theorem page_audit_secure : Pous.Guarantees.PageFillAuditSecure := by
  intro pb P _ _ hP K S Q s w hw hs
  have h := audit_zero_distinct (pageScheme pb P K) (fun _ _ _ _ _ => rfl) (fun W E => page_unif pb P hP K W E)
    S Q s w hw hs
  have h8 : ((2 : ℝ≥0∞) ^ (pageScheme pb P K).ℓ) ^ ((pageScheme pb P K).B - w) = 2 ^ (8 * (P * pb - w)) := by
    rw [← pow_mul]
    rfl
  rwa [h8] at h

end PousErase
