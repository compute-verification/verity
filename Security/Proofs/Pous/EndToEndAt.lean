import Properties.Pous.EndToEndAt
import Proofs.Pous.EndToEnd
import Proofs.Pous.P2.Rounds
import Proofs.Flock.Soundness.KeyedDraw
import Proofs.Pous.Obligation

/-!
# PoUS's ring game at any block count, proved

* `drawAtMost : DrawAtMost`: the code's sampler is Flock's (`uniformOn N` is `uniformL` on the stream's bytes,
  `FlockSoundness.Audit.KeyedDraw.uniformOn_streamOf`), whose rejection loop gives each index at most `1/N`
  (`FlockSoundness.Audit.DrawExec.uniformL_le`): `card_draw_le`. Over the 105 independent streams the indices' fibre is
  a product of single-stream fibres (`card_draws_le`), and a union bound over the indices (`pr_draw_le`) moves any
  event from the streams to uniform indices.
* `endToEndRingAt : EndToEndRingAt`: `endToEndRing`'s proof with that inequality in place of `pr_passes`'s equality,
  and the bound `c` in place of `P2RingAuditSecure`.
* `endToEndRingOfRoundSecure`: `PousP2Rounds.graded` at `N`, on the server's responder read as a `VarResponderOn` with
  a constant state size (`PousP2Rounds.ofSeq`).
* `endToEndRingOfRoundSecureWide`, `endToEndRingDemo`: the same chain at one `N` (`seqAccepts_le_of_round`, which
  reads the per-round floor only at that `N`) from `P2RingRoundSecureWide`, and at Demo B's `N` with its state bound
  (`Pous.SecurityProofs.stateBound_p2ring_demo`).
-/

namespace Pous.EndToEnd.SecurityProofs

open MemoryChallenger
open scoped ENNReal

/-! ### The code's sampler at any block count -/

theorem ofFin_eq_toUInt8 (a : Fin 256) : UInt8.ofFin a = a.val.toUInt8 :=
  UInt8.toNat_inj.1 (by rw [UInt8.toNat_ofFin, Nat.toUInt8_eq, UInt8.toNat_ofNat_of_lt' a.isLt])

theorem streamOf_eq_bytesOf {n : ℕ} (x : Bytes n) : FlockSoundness.Audit.Law.streamOf x = bytesOf x := by
  simp only [FlockSoundness.Audit.Law.streamOf, bytesOf, ofFin_eq_toUInt8]

/-- **One stream**: `uniformOn N` returns `i` on at most a `1/N` share of the streams. -/
theorem card_draw_le (N : ℕ) (hN : 0 < N) (i : ℕ) :
    (Finset.univ.filter fun s : Bytes DRAW_BYTES =>
        FlockSoundness.Randomness.uniformOn N (bytesOf s) = some i).card * N ≤ Fintype.card (Bytes DRAW_BYTES) := by
  have hle := FlockSoundness.Audit.DrawExec.uniformL_le N hN (fun r _ => r = i)
    (fun r => if r = i then 1 else 0)
    (fun r _ ℓ => by
      show FlockSoundness.Game.prCoin (fun _ : Fin ℓ → Fin 256 => r = i) ≤ if r = i then 1 else 0
      split_ifs with h
      · exact FlockSoundness.Audit.DrawExec.prCoin_le_one _
      · simp only [h]
        exact FlockSoundness.Game.prCoin_false.le)
    (DRAW_BYTES + 1) DRAW_BYTES
  have hsum : (∑ r ∈ Finset.range N, if r = i then (1 : ℝ≥0∞) else 0) ≤ 1 := by
    rw [Finset.sum_ite_eq']
    split_ifs <;> simp
  classical
  have hp : FlockSoundness.Game.prCoin (fun s : Bytes DRAW_BYTES =>
      FlockSoundness.Randomness.uniformOn N (bytesOf s) = some i) ≤ (N : ℝ≥0∞)⁻¹ := by
    refine le_trans (FlockSoundness.Game.prCoin_mono fun s hs => ?_) (hle.trans ?_)
    · rw [← streamOf_eq_bytesOf, FlockSoundness.Audit.KeyedDraw.uniformOn_streamOf] at hs
      obtain ⟨⟨r, rest⟩, hu, hr⟩ := Option.map_eq_some_iff.1 hs
      exact ⟨r, rest, hu, hr⟩
    · rw [← one_div]
      exact ENNReal.div_le_div_right hsum _
  rw [FlockSoundness.Game.prCoin_eq_card] at hp
  have hC0 : (Fintype.card (Bytes DRAW_BYTES) : ℝ≥0∞) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  have hN0 : (N : ℝ≥0∞) ≠ 0 := by exact_mod_cast hN.ne'
  rw [ENNReal.div_le_iff hC0 (ENNReal.natCast_ne_top _)] at hp
  have h : (((Finset.univ.filter fun s : Bytes DRAW_BYTES =>
      FlockSoundness.Randomness.uniformOn N (bytesOf s) = some i).card * N : ℕ) : ℝ≥0∞) ≤
        Fintype.card (Bytes DRAW_BYTES) := by
    rw [Nat.cast_mul]
    calc _ ≤ (N : ℝ≥0∞)⁻¹ * Fintype.card (Bytes DRAW_BYTES) * N := by gcongr
      _ = Fintype.card (Bytes DRAW_BYTES) := by
        rw [mul_comm (N : ℝ≥0∞)⁻¹, mul_assoc, ENNReal.inv_mul_cancel hN0 (ENNReal.natCast_ne_top _), mul_one]
  exact_mod_cast h

/-- **The 105 streams**: the streams from which `uniformOn N` reads the indices `I` are a `1/N^105` share. -/
theorem card_draws_le (N : ℕ) (hN : 0 < N) (I : Fin 105 → Fin N) :
    (Finset.univ.filter fun s : Streams =>
        ∀ j, FlockSoundness.Randomness.uniformOn N (bytesOf (s j)) = some (I j).val).card *
      Fintype.card (Fin 105 → Fin N) ≤ Fintype.card Streams := by
  classical
  have hfil : (Finset.univ.filter fun s : Streams =>
        ∀ j, FlockSoundness.Randomness.uniformOn N (bytesOf (s j)) = some (I j).val) =
      Fintype.piFinset fun j => Finset.univ.filter fun t : Bytes DRAW_BYTES =>
        FlockSoundness.Randomness.uniformOn N (bytesOf t) = some (I j).val := by
    ext s
    simp [Fintype.mem_piFinset]
  have h1 : Fintype.card (Fin 105 → Fin N) = ∏ _j : Fin 105, N := by simp
  have h2 : Fintype.card Streams = ∏ _j : Fin 105, Fintype.card (Bytes DRAW_BYTES) := by simp
  rw [hfil, Fintype.card_piFinset, h1, h2, ← Finset.prod_mul_distrib]
  exact Finset.prod_le_prod₀ (fun _ _ => Nat.zero_le _) fun j _ => card_draw_le N hN (I j).val

/-- **A union bound over the draw**: if each value `i` is drawn (`d t i`) on at most a `1/|I|` share of `T`, any event
`F` of the drawn value is at most as likely as under a uniform `i`. -/
theorem pr_draw_le {A T I : Type} [Fintype A] [Fintype T] [Fintype I] [Nonempty A] [Nonempty T] [Nonempty I]
    (d : T → I → Prop) [∀ t i, Decidable (d t i)]
    (hfib : ∀ i, (Finset.univ.filter fun t => d t i).card * Fintype.card I ≤ Fintype.card T)
    (F : A → I → Prop) :
    pr (fun x : A × T => ∃ i, d x.2 i ∧ F x.1 i) ≤ pr (fun x : A × I => F x.1 x.2) := by
  classical
  have hsub : (Finset.univ.filter fun x : A × T => ∃ i, d x.2 i ∧ F x.1 i) ⊆
      (Finset.univ.filter fun x : A × I => F x.1 x.2).biUnion
        (fun y => ({y.1} : Finset A) ×ˢ Finset.univ.filter fun t => d t y.2) := by
    intro x hx
    obtain ⟨i, hd, hF⟩ := (Finset.mem_filter.1 hx).2
    exact Finset.mem_biUnion.2 ⟨(x.1, i), Finset.mem_filter.2 ⟨Finset.mem_univ _, hF⟩,
      Finset.mem_product.2 ⟨Finset.mem_singleton_self _, Finset.mem_filter.2 ⟨Finset.mem_univ _, hd⟩⟩⟩
  have hcard : (Finset.univ.filter fun x : A × T => ∃ i, d x.2 i ∧ F x.1 i).card * Fintype.card I ≤
      (Finset.univ.filter fun x : A × I => F x.1 x.2).card * Fintype.card T := by
    calc _ ≤ (∑ y ∈ Finset.univ.filter (fun x : A × I => F x.1 x.2),
          (({y.1} : Finset A) ×ˢ Finset.univ.filter fun t => d t y.2).card) * Fintype.card I :=
        Nat.mul_le_mul_right _ ((Finset.card_le_card hsub).trans Finset.card_biUnion_le)
      _ = ∑ y ∈ Finset.univ.filter (fun x : A × I => F x.1 x.2),
          (Finset.univ.filter fun t => d t y.2).card * Fintype.card I := by
        rw [Finset.sum_mul]
        simp only [Finset.card_product, Finset.card_singleton, one_mul]
      _ ≤ ∑ _y ∈ Finset.univ.filter (fun x : A × I => F x.1 x.2), Fintype.card T :=
        Finset.sum_le_sum fun y _ => hfib y.2
      _ = _ := by rw [Finset.sum_const, smul_eq_mul]
  have hcard' : ((Finset.univ.filter fun x : A × T => ∃ i, d x.2 i ∧ F x.1 i).card : ℝ≥0∞) * Fintype.card I ≤
      (Finset.univ.filter fun x : A × I => F x.1 x.2).card * Fintype.card T := by exact_mod_cast hcard
  have hI0 : (Fintype.card I : ℝ≥0∞) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  have hT0 : (Fintype.card T : ℝ≥0∞) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  rw [pr_def, pr_def, Fintype.card_prod, Fintype.card_prod, Nat.cast_mul, Nat.cast_mul]
  calc _ = ((Finset.univ.filter fun x : A × T => ∃ i, d x.2 i ∧ F x.1 i).card * (Fintype.card I : ℝ≥0∞)) /
        ((Fintype.card A : ℝ≥0∞) * Fintype.card T * Fintype.card I) :=
        (ENNReal.mul_div_mul_right _ _ hI0 (ENNReal.natCast_ne_top _)).symm
    _ ≤ ((Finset.univ.filter fun x : A × I => F x.1 x.2).card * (Fintype.card T : ℝ≥0∞)) /
        ((Fintype.card A : ℝ≥0∞) * Fintype.card T * Fintype.card I) := ENNReal.div_le_div_right hcard' _
    _ = ((Finset.univ.filter fun x : A × I => F x.1 x.2).card * (Fintype.card T : ℝ≥0∞)) /
        ((Fintype.card A : ℝ≥0∞) * Fintype.card I * Fintype.card T) := by rw [mul_right_comm]
    _ = _ := ENNReal.mul_div_mul_right _ _ hT0 (ENNReal.natCast_ne_top _)

/-- Any event of the 105 indices the code reads from the streams is at most as likely as under uniform indices. -/
theorem pr_draws_le {A : Type} [Fintype A] [Nonempty A] (N : ℕ) [NeZero N] (F : A → (Fin 105 → Fin N) → Prop) :
    pr (fun x : A × Streams => ∃ I : Fin 105 → Fin N,
        (∀ j, FlockSoundness.Randomness.uniformOn N (bytesOf (x.2 j)) = some (I j).val) ∧ F x.1 I) ≤
      pr (fun x : A × (Fin 105 → Fin N) => F x.1 x.2) :=
  pr_draw_le (fun (s : Streams) (I : Fin 105 → Fin N) =>
      ∀ j, FlockSoundness.Randomness.uniformOn N (bytesOf (s j)) = some (I j).val)
    (card_draws_le N (Nat.pos_of_ne_zero (NeZero.ne N))) F

theorem drawAtMost : DrawAtMost := by
  intro N P
  calc pr (fun s : Bytes DRAW_BYTES =>
        ∃ i : Fin (N + 1), FlockSoundness.Randomness.uniformOn (N + 1) (bytesOf s) = some i.val ∧ P i)
      = pr (fun x : Unit × Bytes DRAW_BYTES =>
          ∃ i : Fin (N + 1), FlockSoundness.Randomness.uniformOn (N + 1) (bytesOf x.2) = some i.val ∧ P i) :=
        (pr_snd _).symm
    _ ≤ pr (fun x : Unit × Fin (N + 1) => P x.2) :=
        pr_draw_le (A := Unit) (fun (s : Bytes DRAW_BYTES) (i : Fin (N + 1)) =>
            FlockSoundness.Randomness.uniformOn (N + 1) (bytesOf s) = some i.val)
          (fun i => by rw [Fintype.card_fin]; exact card_draw_le (N + 1) N.succ_pos i.val)
          (fun _ i => P i)
    _ = pr P := pr_snd P

/-! ### The ring game at any block count -/

/-- The server answers every one of the indices `I` at `N` with its block of `C`. -/
def AccAt (N : ℕ) [NeZero N] {S D Q L : ℕ} (W : Bits (schAt N).n) (A : ServerAt N S D Q L)
    (ω : (ringRO N 16448 (2 ^ 16448 - 21065)).Ω) (I : Fin 105 → Fin N) : Prop :=
  seqAnswers ((ringRO N 16448 (2 ^ 16448 - 21065)).answer ω) (A.step W ((schAt N).enc W ω ()).1) 0 (List.ofFn I)
      (A.pre W ((schAt N).enc W ω ()).1 ((schAt N).enc W ω ()).2 ω) =
    List.ofFn fun j => ((schAt N).enc W ω ()).2 (I j)

/-- The server passes at `N` on the indices `I` once the coin tree `v` is built. -/
def EvAt (N : ℕ) [NeZero N] {S D Q L : ℕ} (W : Bits (schAt N).n) (A : ServerAt N S D Q L)
    (ω : (ringRO N 16448 (2 ^ 16448 - 21065)).Ω) (v : Option (ByteArray × ByteArray)) (I : Fin 105 → Fin N) : Prop :=
  ∃ v', v = some v' ∧ AccAt N W A ω I

/-- The streams' draw is at most the uniform indices' (`ε_rs = 0` in the upper bound). -/
theorem pr_passesAt_le (N : ℕ) [NeZero N] {S D Q L : ℕ} (session : ByteArray) (W : Bits (schAt N).n)
    (A : ServerAt N S D Q L) :
    pr (PassesAt N session W A) ≤
      pr fun x : (ringRO N 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × (Fin 105 → Fin N) =>
        EvAt N W A x.1 (coinViewAt N session x.2.1 x.2.2) x.2.2 := by
  rw [← PousThm1.pr_equiv (Equiv.prodAssoc _ _ _) (PassesAt N session W A),
    ← PousThm1.pr_equiv (Equiv.prodAssoc _ _ _) (fun x : (ringRO N 16448 (2 ^ 16448 - 21065)).Ω × HmCoins ×
      (Fin 105 → Fin N) => EvAt N W A x.1 (coinViewAt N session x.2.1 x.2.2) x.2.2)]
  refine le_trans (pr_mono fun y hy => ?_)
    (pr_draws_le N fun (a : (ringRO N 16448 (2 ^ 16448 - 21065)).Ω × HmCoins) I =>
      EvAt N W A a.1 (coinViewAt N session a.2 I) I)
  obtain ⟨I, v, hu, hv, hs⟩ := hy
  exact ⟨I, hu, v, hv, hs⟩

set_option exponentiation.threshold 20000 in
/-- The server is a `SeqResponderOn` of P2's audit at `N`: it sees nothing of the coins but the indices. -/
def responderAt (N : ℕ) [NeZero N] {S D Q L : ℕ} (A : ServerAt N S D Q L) : SeqResponderOn (schAt N) S D Q L :=
  ⟨A.step, A.bounded⟩

theorem accAt_le (N : ℕ) [NeZero N] {S D Q L : ℕ} {c : ℝ≥0∞}
    (hP : ∀ (W : Bits (schAt N).n) (A₁ : Preprocessor (schAt N) S) (A₂ : SeqResponderOn (schAt N) S D Q L),
      pr (seqAcceptsOn (schAt N) 105 W A₁ A₂) ≤ c)
    (W : Bits (schAt N).n) (A : ServerAt N S D Q L) :
    pr (fun o : (ringRO N 16448 (2 ^ 16448 - 21065)).Ω × Unit × (Fin 105 → Fin N) => AccAt N W A o.1 o.2.2) ≤ c := by
  refine le_trans (pr_mono ?_) (hP W A.pre (responderAt N A))
  intro _ h
  exact h

set_option exponentiation.threshold 20000 in
theorem endToEndRingAt : EndToEndRingAt := by
  intro N _ S D Q L c hP session W A
  calc pr (PassesAt N session W A)
      ≤ pr fun x : (ringRO N 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × (Fin 105 → Fin N) =>
          EvAt N W A x.1 (coinViewAt N session x.2.1 x.2.2) x.2.2 := pr_passesAt_le N session W A
    _ ≤ pr fun x : (ringRO N 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × (Fin 105 → Fin N) =>
          AccAt N W A x.1 x.2.2 := pr_mono fun _ ⟨_, _, h⟩ => h
    _ = pr fun o : (ringRO N 16448 (2 ^ 16448 - 21065)).Ω × Unit × (Fin 105 → Fin N) =>
          AccAt N W A o.1 o.2.2 := pr_drop (AccAt N W A)
    _ ≤ c := accAt_le N hP W A

set_option exponentiation.threshold 20000 in
theorem endToEndRingOfRoundSecure : EndToEndRingOfRoundSecure := by
  intro p ε₀ D Q L hnum h N _ h1 h2 session W A
  refine (endToEndRingAt N (SAt N) D Q L
    (ENNReal.ofReal Params.δ + ((2⁻¹ : ℝ≥0∞) ^ 129 + (2⁻¹ : ℝ≥0∞) ^ 130)) (fun W' A₁ A₂ => ?_)
    session W A).trans ?_
  · have hg := PousP2Rounds.graded p ε₀ D Q L h N h1 h2 (fun _ => SAt N) 105 W' A₁ (PousP2Rounds.ofSeq A₂)
    have hl : lowBoundaries (fun _ => SAt N) ((p2ring N 16448 (2 ^ 16448 - 21065) 16432).stateBound Params.ρ) 105 =
        105 := PousP2Rounds.lowBoundaries_const _ _
    rw [hl, Nat.cast_ofNat] at hg
    calc pr (seqAcceptsOn (schAt N) 105 W' A₁ A₂)
        = pr (varPassesOn (schAt N) 105 105 W' A₁ (PousP2Rounds.ofSeq A₂)) :=
          congrArg pr (funext fun x => propext (PousP2Rounds.varPassesOn_ofSeq_iff _ W' A₁ A₂ x).symm)
      _ ≤ _ := hg
      _ ≤ _ := hnum
  · exact add_le_add le_rfl budget

set_option exponentiation.threshold 20000 in
/-- `PousP2Rounds.graded` at one block count on a constant state size: the per-round floor at `N` bounds P2's
sequential audit of a `⌊ρ·|C|⌋`-bit responder by `(1 − p)^105 + 105·ε₀`. -/
theorem seqAccepts_le_of_round {N : ℕ} [NeZero N] {p ε₀ : ℝ≥0∞} {D Q L : ℕ}
    (h : ∀ (Sz : ℕ → ℕ) (k : ℕ) (W : Bits (schAt N).n) (A₁ : Preprocessor (schAt N) (Sz 0))
      (A₂ : VarResponderOn (schAt N) Sz D Q L),
      ∃ bad : Outcome (schAt N) k → Prop, pr bad ≤ k * ε₀ ∧
        ∀ j < k, Sz j ≤ (schAt N).stateBound Params.ρ →
          pr (fun x => varPassesOn _ k (j + 1) W A₁ A₂ x ∧ ¬ bad x) ≤
            (1 - p) * pr (fun x => varPassesOn _ k j W A₁ A₂ x ∧ ¬ bad x))
    (W : Bits (schAt N).n) (A₁ : Preprocessor (schAt N) (SAt N)) (A₂ : SeqResponderOn (schAt N) (SAt N) D Q L) :
    pr (seqAcceptsOn (schAt N) 105 W A₁ A₂) ≤ (1 - p) ^ 105 + 105 * ε₀ := by
  obtain ⟨bad, hbad, hstep⟩ := h (fun _ => SAt N) 105 W A₁ (PousP2Rounds.ofSeq A₂)
  classical
  have hc := PousP2Rounds.pr_chain (fun j x => varPassesOn _ 105 j W A₁ (PousP2Rounds.ofSeq A₂) x ∧ ¬ bad x)
    (fun _ => True) (1 - p) 105
    (fun j x hx => ⟨PousP2Rounds.varPassesOn_of_succ (Sz := fun _ => SAt N) (schAt N) W A₁
      (PousP2Rounds.ofSeq A₂) x hx.1, hx.2⟩)
    (fun j hj _ => hstep j hj le_rfl) 105 le_rfl
  have hcard : ((Finset.range 105).filter fun _ : ℕ => True).card = 105 := by simp
  rw [hcard] at hc
  rw [Nat.cast_ofNat] at hbad
  calc pr (seqAcceptsOn (schAt N) 105 W A₁ A₂)
      = pr (varPassesOn (schAt N) 105 105 W A₁ (PousP2Rounds.ofSeq A₂)) :=
        congrArg pr (funext fun x => propext (PousP2Rounds.varPassesOn_ofSeq_iff _ W A₁ A₂ x).symm)
    _ ≤ pr (fun x => varPassesOn _ 105 105 W A₁ (PousP2Rounds.ofSeq A₂) x ∧ ¬ bad x) + pr bad :=
        PousP2Rounds.pr_le_and_not_add _ _
    _ ≤ (1 - p) ^ 105 + 105 * ε₀ := add_le_add hc hbad

set_option exponentiation.threshold 20000 in
theorem endToEndRingOfRoundSecureWide : EndToEndRingOfRoundSecureWide := by
  intro p ε₀ D Q L hnum h N _ h1 h2 session W A
  exact (endToEndRingAt N (SAt N) D Q L ((1 - p) ^ 105 + 105 * ε₀)
    (seqAccepts_le_of_round (h N h1 h2)) session W A).trans (hnum.trans (add_le_add le_rfl budget))

set_option exponentiation.threshold 20000 in
theorem endToEndRingDemo : EndToEndRingDemo := by
  intro p ε₀ D Q L hnum h N _ hN session W
  have hw := endToEndRingOfRoundSecureWide p ε₀ D Q L hnum h N (by rw [hN]; norm_num) (by rw [hN]; norm_num)
    session W
  have hS : SAt N = 1060008348186 := Pous.SecurityProofs.stateBound_p2ring_demo N _ _ hN
  rw [hS] at hw
  exact hw

/-! ### The honest and nontrivial lemmas at any block count

As `Pous.EndToEnd.SecurityProofs`' at `2^19`. At `N` the draw may reject a word, so the honest lemma is pointwise: the
honest full-state server passes on every `ω`, coins and streams from which the code's sampler draws its indices. -/

theorem coinViewAt_some (N : ℕ) (hN : 0 < N) (hN64 : N ≤ 2 ^ 64) (session : ByteArray) (hs : session.size = 32)
    (c : HmCoins) (I : Fin 105 → Fin N) : ∃ v, coinViewAt N session c I = some v := by
  obtain ⟨t, ht⟩ := make_ok_of N session (coinKey c.1) (Array.ofFn fun j => (I j).val)
    (Array.ofFn fun j => bytesOf (c.2 j)) hN hN64 hs (Array.size_ofFn) (by rw [Array.size_ofFn, Array.size_ofFn])
    (fun j hj => by
      rw [Array.size_ofFn] at hj
      rw [ofFn_getElem! _ j hj]; exact (I _).isLt)
    (fun j hj => by
      rw [Array.size_ofFn] at hj
      rw [ofFn_getElem! _ j hj]; exact size_bytesOf _)
    (checkKey_coinKey c.1)
  exact ⟨_, by rw [coinViewAt, ht]; rfl⟩

/-- The honest full-state server for a scheme in the ring model: it keeps all of `C` and answers each index from it,
with no query. -/
noncomputable def honestRing {B w p : ℕ} [NeZero p] (sc : Scheme (ringRO B w p)) : RingServer sc (sc.B * sc.ℓ) 0 0 0 :=
  ⟨honestPre sc, fun _ _ _ σ i => .ret ((codeEquiv sc.B sc.ℓ).symm σ i, σ),
    fun _ _ _ _ _ _ => ⟨le_rfl, le_rfl, le_rfl⟩⟩

theorem honestRing_answers {B w p : ℕ} [NeZero p] (sc : Scheme (ringRO B w p))
    (f : (ringRO B w p).Qry → (ringRO B w p).Ans) (W : Bits sc.n) (pp : Bits sc.t) (C : Fin sc.B → Bits sc.ℓ) (k : ℕ)
    (I : Fin k → Fin sc.B) :
    seqAnswers f ((honestRing sc).step W pp) 0 (List.ofFn I) (codeEquiv _ _ C) = List.ofFn fun j => C (I j) := by
  show seqAnswers f (fun _ σ i => .ret ((codeEquiv sc.B sc.ℓ).symm σ i, σ)) 0 (List.ofFn I) (codeEquiv _ _ C) = _
  rw [seqAnswers_const f (fun σ i => (codeEquiv sc.B sc.ℓ).symm σ i), List.map_ofFn]
  congr 1
  funext j
  simp

set_option exponentiation.threshold 20000 in
/-- **`endToEndRingAt` holds at the honest instance**: at every block count `N ≤ 2^64` and 32-byte session id, the
honest full-state server passes on every `ω`, coins and streams from which the code's sampler draws its indices `I`. -/
theorem endToEndRingAt_honest (N : ℕ) [NeZero N] (hN64 : N ≤ 2 ^ 64) (session : ByteArray) (hs : session.size = 32)
    (W : Bits (schAt N).n) (x : (ringRO N 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) (I : Fin 105 → Fin N)
    (hI : ∀ j, FlockSoundness.Randomness.uniformOn N (bytesOf (x.2.2 j)) = some (I j).val) :
    PassesAt N session W (honestRing (schAt N)) x := by
  obtain ⟨v, hv⟩ := coinViewAt_some N (Nat.pos_of_ne_zero (NeZero.ne N)) hN64 session hs x.2.1 I
  exact ⟨I, v, hI, hv, honestRing_answers (schAt N) _ W ((schAt N).enc W x.1 ()).1 ((schAt N).enc W x.1 ()).2 105 I⟩

/-- **`endToEndRingAt` is nontrivial**: at every block count and every bound `c < 1`, an event that holds on every
coin, a verifier accepting everything, exceeds `c`. -/
theorem endToEndRingAt_nontrivial (N : ℕ) [NeZero N] (c : ℝ≥0∞) (hc : c < 1) :
    ¬ pr (fun _ : (ringRO N 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams => True) ≤ c := by
  rw [pr_eq_one fun _ => trivial]
  exact not_le.2 hc

/-- **`endToEndRingOfRoundSecure`, `endToEndRingOfRoundSecureWide` and `endToEndRingDemo` are nontrivial**: at every
block count, a verifier accepting everything exceeds their bound `1% + 2^-128`. Their honest lemma is
`endToEndRingAt_honest`, their event being `PassesAt`. -/
theorem endToEndRingOfRoundSecure_nontrivial (N : ℕ) [NeZero N] :
    ¬ pr (fun _ : (ringRO N 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams => True) ≤
      ENNReal.ofReal Params.δ + Params.εMax :=
  endToEndRingAt_nontrivial N _ bound_lt_one

theorem endToEndRingOfRoundSecureWide_nontrivial (N : ℕ) [NeZero N] :
    ¬ pr (fun _ : (ringRO N 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams => True) ≤
      ENNReal.ofReal Params.δ + Params.εMax :=
  endToEndRingOfRoundSecure_nontrivial N

theorem endToEndRingDemo_nontrivial (N : ℕ) [NeZero N] :
    ¬ pr (fun _ : (ringRO N 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams => True) ≤
      ENNReal.ofReal Params.δ + Params.εMax :=
  endToEndRingOfRoundSecure_nontrivial N

end Pous.EndToEnd.SecurityProofs
