import Proofs.Accounting.Memory.Counting
import Proofs.Pous.Thm1.Theorem1iiSeq
import Definitions.Accounting.Guarantees.Memory

/-!
# Memory adds: the composition with PoUS's guarantee

Fix `z`. Before challenge `j` the location's state, `N = M + H` bits with its help, lies in the fibre of `z` under
`π j`. When each of the `c` fibres has at most `2^S` states, each is coded into `S` bits (`exists_code`), and the
location's adversary, run through the codes, is an `S`-bit PoUS adversary with the same answers (`seqAnswers_code`,
along the states the audit reaches): its program maps the old one's output (`progMap`), so it makes the same queries in
the same rounds, and its bound is the old one's. PoUS's guarantee then bounds its acceptance by `δ + ε`. By the
counting core, a uniform `z` has a fibre above `2^(N − q + k) ≤ 2^S` at a given round with probability at most
`2^-k`, so at some one of the `c` rounds with probability at most `c · 2^-k` (`pr_heavy_le`, `adds_core`). The
isolation forms are the general ones at `H = 0`.
-/

namespace Accounting.MemoryProofs

open Pous Accounting.Memory
open scoped ENNReal

/-! ## A program with its output mapped -/

/-- `P` with its output mapped by `g`: the same rounds and the same queries. -/
def progMap {I : IdealModel} {α β : Type} (g : α → β) : Prog I α → Prog I β
  | .ret a => .ret (g a)
  | .round qs k => .round qs fun as => progMap g (k as)

section progMap

variable {I : IdealModel} {α β : Type} (g : α → β) (f : I.Qry → I.Ans)

theorem run_progMap (P : Prog I α) : (progMap g P).run f = g (P.run f) := by
  induction P with
  | ret a => rfl
  | round qs k ih => simp only [progMap, Prog.run]; exact ih _

theorem depth_progMap (P : Prog I α) : (progMap g P).depth f = P.depth f := by
  induction P with
  | ret a => rfl
  | round qs k ih => simp only [progMap, Prog.depth, ih]

theorem work_progMap (P : Prog I α) : (progMap g P).work f = P.work f := by
  induction P with
  | ret a => rfl
  | round qs k ih => simp only [progMap, Prog.work, ih]

theorem bounded_progMap {D Q : ℕ} {P : Prog I α} (h : P.Bounded D Q) : (progMap g P).Bounded D Q :=
  fun ω => by rw [depth_progMap, work_progMap]; exact h ω

end progMap

section ring

variable {B w p : ℕ} [NeZero p] {α β : Type} (g : α → β) (f : (ringRO B w p).Qry → (ringRO B w p).Ans)

theorem mulDepth_progMap (P : Prog (ringRO B w p) α) : (progMap g P).mulDepth f = P.mulDepth f := by
  induction P with
  | ret a => rfl
  | round qs k ih => simp only [progMap, Prog.mulDepth, ih]

theorem keyWork_progMap (P : Prog (ringRO B w p) α) : (progMap g P).keyWork f = P.keyWork f := by
  induction P with
  | ret a => rfl
  | round qs k ih => simp only [progMap, Prog.keyWork, ih]

theorem ringWork_progMap (P : Prog (ringRO B w p) α) : (progMap g P).ringWork f = P.ringWork f := by
  induction P with
  | ret a => rfl
  | round qs k ih => simp only [progMap, Prog.ringWork, ih]

theorem ringBounded_progMap {D Q L : ℕ} {P : Prog (ringRO B w p) α} (h : P.RingBounded D Q L) :
    (progMap g P).RingBounded D Q L :=
  fun ω => by rw [mulDepth_progMap, keyWork_progMap, ringWork_progMap]; exact h ω

end ring

/-! ## Coding a fibre into `S` bits -/

/-- A set of at most `|β|` elements has a code into `β` that decodes back on it. -/
theorem exists_code {α β : Type} [Nonempty α] [Fintype β] [Nonempty β] (s : Finset α) :
    ∃ (e : α → β) (d : β → α), s.card ≤ Fintype.card β → ∀ a ∈ s, d (e a) = a := by
  classical
  by_cases hs : s.card ≤ Fintype.card β
  · obtain ⟨emb⟩ := Function.Embedding.nonempty_of_card_le (α := s) (β := β) (by rwa [Fintype.card_coe])
    let e : α → β := fun a => if ha : a ∈ s then emb ⟨a, ha⟩ else Classical.arbitrary β
    have hinj : Set.InjOn e s := by
      intro a ha b hb hab
      simp only [e, Finset.mem_coe.1 ha, Finset.mem_coe.1 hb, ↓reduceDIte] at hab
      exact congrArg Subtype.val (emb.injective hab)
    exact ⟨e, Function.invFunOn e s, fun _ a ha => hinj.leftInvOn_invFunOn (Finset.mem_coe.2 ha)⟩
  · exact ⟨fun _ => Classical.arbitrary β, fun _ => Classical.arbitrary α, fun h => absurd h hs⟩

/-- **The coded responder answers as the location does.** Run through codes `e j` that `d j` inverts on the states
of `P j` before challenge `j < c`, from the code of a state whose run is in `P` before each challenge, the responder
returns the same blocks. -/
theorem seqAnswers_code {I : IdealModel} {ℓ N S B : ℕ} (f : I.Qry → I.Ans)
    (step : ℕ → Bits N → Fin B → Prog I (Bits ℓ × Bits N)) (P : ℕ → Bits N → Prop) (c : ℕ)
    (e : ℕ → Bits N → Bits S) (d : ℕ → Bits S → Bits N)
    (hed : ∀ j < c, ∀ σ, P j σ → d j (e j σ) = σ) :
    ∀ (is : List (Fin B)) (j : ℕ) (σ : Bits N), j + is.length ≤ c →
      (∀ m < is.length, P (j + m) (Sampled.seqState f step j (is.take m) σ)) →
      seqAnswers f (fun j s i => progMap (fun r => (r.1, e (j + 1) r.2)) (step j (d j s) i)) j is (e j σ) =
        seqAnswers f step j is σ
  | [], _, _, _, _ => rfl
  | i :: is, j, σ, hlen, hz => by
      have hσ : P j σ := by simpa [Sampled.seqState] using hz 0 (by simp)
      simp only [List.length_cons] at hlen
      simp only [seqAnswers, run_progMap, hed j (by omega) σ hσ]
      congr 1
      refine seqAnswers_code f step P c e d hed is (j + 1) _ (by omega) fun m hm => ?_
      have h := hz (m + 1) (by simp only [List.length_cons]; omega)
      rw [show j + (m + 1) = j + 1 + m by omega, List.take_succ_cons] at h
      exact h

/-! ## The audit's acceptance at any state width -/

/-- The sequential audit with `c` challenges accepts a preprocessing `A₁` and a step `step` with an `S`-bit state:
`seqAccepts` and `seqAcceptsOn`, each with its responder's step. -/
def accepts {I : IdealModel} (sch : Scheme I) {S : ℕ} (c : ℕ) (W : Bits sch.n) (A₁ : Preprocessor sch S)
    (step : Bits sch.n → Bits sch.t → ℕ → Bits S → Fin sch.B → Prog I (Bits sch.ℓ × Bits S))
    (x : Outcome sch c) : Prop :=
  let ω := x.1
  let pp := (sch.enc W ω x.2.1).1
  let C := (sch.enc W ω x.2.1).2
  seqAnswers (I.answer ω) (step W pp) 0 (List.ofFn x.2.2) (A₁ W pp C ω) = List.ofFn fun j => C (x.2.2 j)

/-- The location's adversary for `z`, run through the codes `e`, `d`: its preprocessing. -/
def codedPre {I : IdealModel} {sch : Scheme I} {M S : ℕ} (A₁ : Preprocessor sch M) (e : ℕ → Bits M → Bits S) :
    Preprocessor sch S :=
  fun W pp C ω => e 0 (A₁ W pp C ω)

/-- The location's adversary for `z`, run through the codes `e`, `d`: its step. -/
def codedStep {I : IdealModel} {sch : Scheme I} {M S : ℕ}
    (step : Bits sch.n → Bits sch.t → ℕ → Bits M → Fin sch.B → Prog I (Bits sch.ℓ × Bits M))
    (e : ℕ → Bits M → Bits S) (d : ℕ → Bits S → Bits M) :
    Bits sch.n → Bits sch.t → ℕ → Bits S → Fin sch.B → Prog I (Bits sch.ℓ × Bits S) :=
  fun W pp j s i => progMap (fun r => (r.1, e (j + 1) r.2)) (step W pp j (d j s) i)

/-- Where the codes invert on the sets `P j` that hold every state the audit reaches before challenge `j`, the coded
adversary is accepted exactly when the location's is. -/
theorem accepts_coded_iff {I : IdealModel} (sch : Scheme I) {N S : ℕ} (c : ℕ) (W : Bits sch.n)
    (A₁ : Preprocessor sch N)
    (step : Bits sch.n → Bits sch.t → ℕ → Bits N → Fin sch.B → Prog I (Bits sch.ℓ × Bits N))
    (P : ℕ → Bits N → Prop)
    (hreach : ∀ (x : Outcome sch c) (j : ℕ), j < c →
      P j (Sampled.seqState (I.answer x.1) (step W (sch.enc W x.1 x.2.1).1) 0 ((List.ofFn x.2.2).take j)
        (A₁ W (sch.enc W x.1 x.2.1).1 (sch.enc W x.1 x.2.1).2 x.1)))
    (e : ℕ → Bits N → Bits S) (d : ℕ → Bits S → Bits N)
    (hed : ∀ j < c, ∀ σ, P j σ → d j (e j σ) = σ) (x : Outcome sch c) :
    accepts sch c W (codedPre A₁ e) (codedStep step e d) x ↔ accepts sch c W A₁ step x := by
  have h := seqAnswers_code (I.answer x.1) (step W (sch.enc W x.1 x.2.1).1) P c e d hed (List.ofFn x.2.2) 0
    (A₁ W (sch.enc W x.1 x.2.1).1 (sch.enc W x.1 x.2.1).2 x.1) (by simp) fun m hm => by
      rw [Nat.zero_add]
      exact hreach x m (by simpa using hm)
  exact iff_of_eq (congrArg (· = List.ofFn fun j => (sch.enc W x.1 x.2.1).2 (x.2.2 j)) h)

/-! ## Markov and the union over the challenges -/

/-- A property of at most `2^(q − k)` of the `2^q` values has probability at most `2^-k`. -/
theorem pr_le_of_count {q k : ℕ} (p : Bits q → Prop) [DecidablePred p]
    (hc : 2 ^ k * (Finset.univ.filter p).card ≤ 2 ^ q) : pr p ≤ (2⁻¹ : ℝ≥0∞) ^ k := by
  rw [pr_def]
  generalize (Finset.univ.filter p).card = N at hc ⊢
  have hcard : Fintype.card (Bits q) = 2 ^ q := by simp [Bits]
  rw [hcard, Nat.cast_pow, Nat.cast_ofNat,
    ENNReal.div_le_iff (pow_ne_zero _ two_ne_zero) (ENNReal.pow_ne_top ENNReal.ofNat_ne_top)]
  have hk0 : (2 : ℝ≥0∞) ^ k ≠ 0 := pow_ne_zero _ two_ne_zero
  have hkt : (2 : ℝ≥0∞) ^ k ≠ ⊤ := ENNReal.pow_ne_top ENNReal.ofNat_ne_top
  calc (N : ℝ≥0∞) = (2⁻¹ : ℝ≥0∞) ^ k * (2 ^ k * N) := by
        rw [← mul_assoc, ← ENNReal.inv_pow, ENNReal.inv_mul_cancel hk0 hkt, one_mul]
    _ ≤ (2⁻¹ : ℝ≥0∞) ^ k * 2 ^ q := by gcongr; exact_mod_cast hc

/-- A uniform `z` leaves more than `2^(M − q + k)` states with probability at most `2^-k`. -/
theorem pr_heavy_le {M q : ℕ} (π : Bits M → Bits q) (k : ℕ) :
    pr (fun z : Bits q => 2 ^ (M + k - q) < (fibre π z).card) ≤ (2⁻¹ : ℝ≥0∞) ^ k := by
  classical
  exact pr_le_of_count _ (counting M q k π)

/-- With advice: a uniform `z` leaves more than `2^(M + a − q + k)` states with probability at most `2^-k`. -/
theorem pr_heavy_advice_le {M a q : ℕ} (π : Bits M → Bits a → Bits q) (k : ℕ) :
    pr (fun z : Bits q => 2 ^ (M + a + k - q) < (fibreAdvice π z).card) ≤ (2⁻¹ : ℝ≥0∞) ^ k := by
  classical
  exact pr_le_of_count _ (countingAdvice M a q k π)

/-- **Memory adds, over any sets of states.** Let `F j z`, for each challenge `j < c` and value `z`, hold every state
the audit reaches with `z` before challenge `j`, and let a uniform `z` give `F j z` more than `T ≤ 2^S` states with
probability at most `2^-k` at each `j`. If every coded adversary is accepted with probability at most `η`, the
location's adversary, with a uniform `z`, is accepted with probability at most `η + c · 2^-k`. -/
theorem adds_sets {I : IdealModel} (sch : Scheme I) {N q : ℕ} (S c k T : ℕ) (η : ℝ≥0∞) (W : Bits sch.n)
    (A₁ : Bits q → Preprocessor sch N)
    (step : Bits q → Bits sch.n → Bits sch.t → ℕ → Bits N → Fin sch.B → Prog I (Bits sch.ℓ × Bits N))
    (F : ℕ → Bits q → Finset (Bits N))
    (hreach : ∀ (z : Bits q) (x : Outcome sch c) (j : ℕ), j < c →
      Sampled.seqState (I.answer x.1) (step z W (sch.enc W x.1 x.2.1).1) 0 ((List.ofFn x.2.2).take j)
        (A₁ z W (sch.enc W x.1 x.2.1).1 (sch.enc W x.1 x.2.1).2 x.1) ∈ F j z)
    (hheavy : ∀ j, pr (fun z : Bits q => T < (F j z).card) ≤ (2⁻¹ : ℝ≥0∞) ^ k) (hT : T ≤ 2 ^ S)
    (hsec : ∀ (z : Bits q) (e : ℕ → Bits N → Bits S) (d : ℕ → Bits S → Bits N),
      pr (accepts sch c W (codedPre (A₁ z) e) (codedStep (step z) e d)) ≤ η) :
    pr (fun x : Bits q × Outcome sch c => accepts sch c W (A₁ x.1) (step x.1) x.2) ≤ η + c * (2⁻¹ : ℝ≥0∞) ^ k := by
  classical
  have hbad : pr (fun z : Bits q => ∃ j ∈ Finset.range c, T < (F j z).card) ≤ c * (2⁻¹ : ℝ≥0∞) ^ k := by
    refine (PousThm1.pr_exists_le _ _).trans ?_
    calc ∑ j ∈ Finset.range c, pr (fun z : Bits q => T < (F j z).card)
        ≤ ∑ _j ∈ Finset.range c, (2⁻¹ : ℝ≥0∞) ^ k := Finset.sum_le_sum fun j _ => hheavy j
      _ = c * (2⁻¹ : ℝ≥0∞) ^ k := by rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
  have hgood : ∀ z, ¬ (∃ j ∈ Finset.range c, T < (F j z).card) →
      pr (accepts sch c W (A₁ z) (step z)) ≤ η := by
    intro z hz
    have hcard : ∀ j < c, (F j z).card ≤ Fintype.card (Bits S) := by
      intro j hj
      calc (F j z).card ≤ T := not_lt.1 fun h => hz ⟨j, Finset.mem_range.2 hj, h⟩
        _ ≤ 2 ^ S := hT
        _ = Fintype.card (Bits S) := by simp [Bits]
    choose e d hed using fun j => exists_code (β := Bits S) (F j z)
    have hed' : ∀ j < c, ∀ σ, σ ∈ F j z → d j (e j σ) = σ := fun j hj σ hσ => hed j (hcard j hj) σ hσ
    have heq : accepts sch c W (A₁ z) (step z) = accepts sch c W (codedPre (A₁ z) e) (codedStep (step z) e d) :=
      funext fun x => propext
        (accepts_coded_iff sch c W (A₁ z) (step z) (fun j σ => σ ∈ F j z) (hreach z) e d hed' x).symm
    rw [heq]
    exact hsec z e d
  calc pr (fun x : Bits q × Outcome sch c => accepts sch c W (A₁ x.1) (step x.1) x.2)
      ≤ pr (fun z : Bits q => ∃ j ∈ Finset.range c, T < (F j z).card) + η :=
        PousThm1.pr_prod_le (fun z x => accepts sch c W (A₁ z) (step z) x) _ η hgood
    _ ≤ c * (2⁻¹ : ℝ≥0∞) ^ k + η := add_le_add hbad le_rfl
    _ = η + c * (2⁻¹ : ℝ≥0∞) ^ k := add_comm _ _

/-- **Memory adds, in general.** The fibres of the decoders `π j`, at `T = 2^(N − q + k)`. -/
theorem adds_core {I : IdealModel} (sch : Scheme I) {N q : ℕ} (S c k : ℕ) (η : ℝ≥0∞) (W : Bits sch.n)
    (π : ℕ → Bits N → Bits q) (A₁ : Bits q → Preprocessor sch N)
    (step : Bits q → Bits sch.n → Bits sch.t → ℕ → Bits N → Fin sch.B → Prog I (Bits sch.ℓ × Bits N))
    (hdet : Determines sch c W π A₁ step) (hNS : N + k ≤ S + q)
    (hsec : ∀ (z : Bits q) (e : ℕ → Bits N → Bits S) (d : ℕ → Bits S → Bits N),
      pr (accepts sch c W (codedPre (A₁ z) e) (codedStep (step z) e d)) ≤ η) :
    pr (fun x : Bits q × Outcome sch c => accepts sch c W (A₁ x.1) (step x.1) x.2) ≤ η + c * (2⁻¹ : ℝ≥0∞) ^ k :=
  adds_sets sch S c k (2 ^ (N + k - q)) η W A₁ step (fun j z => fibre (π j) z)
    (fun z x j hj => by simp [fibre, hdet z x j hj]) (fun j => pr_heavy_le (π j) k)
    (Nat.pow_le_pow_right (by norm_num) (by omega)) hsec

/-- **Memory adds with advice, in general.** The advice fibres of the decoders `π j`, at `T = 2^(N + a − q + k)`. -/
theorem adds_advice_core {I : IdealModel} (sch : Scheme I) {N a q : ℕ} (S c k : ℕ) (η : ℝ≥0∞) (W : Bits sch.n)
    (π : ℕ → Bits N → Bits a → Bits q) (A₁ : Bits q → Preprocessor sch N)
    (step : Bits q → Bits sch.n → Bits sch.t → ℕ → Bits N → Fin sch.B → Prog I (Bits sch.ℓ × Bits N))
    (hdet : DeterminesAdvice sch c W π A₁ step) (hNS : N + a + k ≤ S + q)
    (hsec : ∀ (z : Bits q) (e : ℕ → Bits N → Bits S) (d : ℕ → Bits S → Bits N),
      pr (accepts sch c W (codedPre (A₁ z) e) (codedStep (step z) e d)) ≤ η) :
    pr (fun x : Bits q × Outcome sch c => accepts sch c W (A₁ x.1) (step x.1) x.2) ≤ η + c * (2⁻¹ : ℝ≥0∞) ^ k :=
  adds_sets sch S c k (2 ^ (N + a + k - q)) η W A₁ step (fun j z => fibreAdvice (π j) z)
    (fun z x j hj => by simpa [fibreAdvice] using hdet z x j hj) (fun j => pr_heavy_advice_le (π j) k)
    (Nat.pow_le_pow_right (by norm_num) (by omega)) hsec

/-- The event `accepts ∧ ¬ q + S < M + k` has probability at most `η` once `accepts` does, whenever
`M + k ≤ S + q` gives that bound, since otherwise the event is empty. -/
theorem pr_and_not_le {α : Type} [Fintype α] (A : α → Prop) {q S M k : ℕ} (η : ℝ≥0∞)
    (h : M + k ≤ S + q → pr A ≤ η) : pr (fun x => A x ∧ ¬ q + S < M + k) ≤ η := by
  by_cases hlt : q + S < M + k
  · calc pr (fun x => A x ∧ ¬ q + S < M + k) ≤ pr (fun _ : α => False) := pr_mono fun _ hx => hx.2 hlt
      _ = 0 := by classical rw [pr_def]; simp
      _ ≤ η := zero_le
  · exact (pr_mono fun _ hx => hx.1).trans (h (by omega))

/-! ## The statements -/

theorem adds : Accounting.Guarantees.Memory.Adds := by
  intro I sch S D Q c δ ε hsec M H q k W π A₁ A₂ hdet
  simp only [AuditSecure] at hsec
  refine pr_and_not_le (fun x : Bits q × Outcome sch c => seqAccepts sch c W (A₁ x.1) (A₂ x.1) x.2) _ fun hNS => ?_
  exact adds_core sch S c k (δ + ε) W π A₁ (fun z => (A₂ z).step) hdet hNS fun z e d =>
    hsec W (codedPre (A₁ z) e)
      ⟨codedStep (A₂ z).step e d, fun W pp j s i => bounded_progMap _ ((A₂ z).bounded W pp j (d j s) i)⟩

theorem addsOn : Accounting.Guarantees.Memory.AddsOn := by
  intro B w p _ sch S D Q L c δ ε hsec M H q k W π A₁ A₂ hdet
  simp only [AuditSecureOn] at hsec
  refine pr_and_not_le (fun x : Bits q × Outcome sch c => seqAcceptsOn sch c W (A₁ x.1) (A₂ x.1) x.2) _
    fun hNS => ?_
  exact adds_core sch S c k (δ + ε) W π A₁ (fun z => (A₂ z).step) hdet hNS fun z e d =>
    hsec W (codedPre (A₁ z) e)
      ⟨codedStep (A₂ z).step e d, fun W pp j s i => ringBounded_progMap _ ((A₂ z).bounded W pp j (d j s) i)⟩

theorem addsP2Ring : Accounting.Guarantees.Memory.AddsP2Ring := by
  intro D Q L h M H q k W π A₁ A₂ hdet
  have := addsOn _ _ D Q L 105 _ _ h M H q k W π A₁ A₂ hdet
  rwa [Nat.cast_ofNat] at this

theorem addsIsolated : Accounting.Guarantees.Memory.AddsIsolated := by
  intro I sch S D Q c δ ε hsec M q k W π A₁ A₂ hdet
  exact adds sch S D Q c δ ε hsec M 0 q k W π A₁ A₂ hdet

theorem addsOnIsolated : Accounting.Guarantees.Memory.AddsOnIsolated := by
  intro B w p _ sch S D Q L c δ ε hsec M q k W π A₁ A₂ hdet
  exact addsOn sch S D Q L c δ ε hsec M 0 q k W π A₁ A₂ hdet

theorem addsP2RingIsolated : Accounting.Guarantees.Memory.AddsP2RingIsolated := by
  intro D Q L h M q k W π A₁ A₂ hdet
  exact addsP2Ring D Q L h M 0 q k W π A₁ A₂ hdet

theorem addsAdvice : Accounting.Guarantees.Memory.AddsAdvice := by
  intro I sch S D Q c δ ε hsec M H a q k W π A₁ A₂ hdet
  simp only [AuditSecure] at hsec
  refine pr_and_not_le (fun x : Bits q × Outcome sch c => seqAccepts sch c W (A₁ x.1) (A₂ x.1) x.2) _ fun hNS => ?_
  exact adds_advice_core sch S c k (δ + ε) W π A₁ (fun z => (A₂ z).step) hdet hNS fun z e d =>
    hsec W (codedPre (A₁ z) e)
      ⟨codedStep (A₂ z).step e d, fun W pp j s i => bounded_progMap _ ((A₂ z).bounded W pp j (d j s) i)⟩

theorem addsAdviceOn : Accounting.Guarantees.Memory.AddsAdviceOn := by
  intro B w p _ sch S D Q L c δ ε hsec M H a q k W π A₁ A₂ hdet
  simp only [AuditSecureOn] at hsec
  refine pr_and_not_le (fun x : Bits q × Outcome sch c => seqAcceptsOn sch c W (A₁ x.1) (A₂ x.1) x.2) _
    fun hNS => ?_
  exact adds_advice_core sch S c k (δ + ε) W π A₁ (fun z => (A₂ z).step) hdet hNS fun z e d =>
    hsec W (codedPre (A₁ z) e)
      ⟨codedStep (A₂ z).step e d, fun W pp j s i => ringBounded_progMap _ ((A₂ z).bounded W pp j (d j s) i)⟩

theorem addsAdviceP2Ring : Accounting.Guarantees.Memory.AddsAdviceP2Ring := by
  intro D Q L h M H a q k W π A₁ A₂ hdet
  have := addsAdviceOn _ _ D Q L 105 _ _ h M H a q k W π A₁ A₂ hdet
  rwa [Nat.cast_ofNat] at this

end Accounting.MemoryProofs
