import Properties.Pous.Wipe
import Proofs.Pous.StreamedWipe.Family
import Proofs.Pous.Sanity
import Mathlib.Tactic.SplitIfs
import Mathlib.Tactic.DefEqTransformations

/-!
# Proof that an accepted wipe meets the survivor bound it publishes (`Properties.Pous.Wipe`)

* **The ideal game at `k` reads** (`winIdeal_le_of_reads`): `StreamedWipeMeetsReads`, proved
  (`PousStreamedWipe.streamed_wipe_meets_reads`), at a key that is the whole label table, whose labels are a bijection
  of the key, so its real game is the ideal one.
* **The distinguisher on uniform labels** (`pr_distinguisher_le`): the adversary's distinguisher is won only by a prover
  of the streamed game. The device's state when the fill ends and an egress in `A`, `chip + mem` bits and one of at most
  `2^E` strings, go one-to-one into the game's inside, the `n · 512 + R` bits of the declared memory, and its egress,
  `E' = E + chip + (mem − (n · 512 + R))`: the on-chip state and the store past the declared memory are charged as
  egress. `hold` and the strategy's fill are the prover's fill, its answers the prover's answers and `keep` its keep, so
  a run the observed event accepts with `keep` returning `z` and the egress in `A` is one the prover wins, at most
  `2^-128` at `S_φ ≥ 512·(⌈n/r⌉ − 1) + n + ⌊log₂ k⌋ + 130 + R + E'`.
* **The survivor bound** (`survivorBound_of_observed`): the game's event splits on whether the egress is in `A`. In `A`
  it is the distinguisher on the labels `prf K`: at most its value on uniform labels plus the PRF's advantage
  (`Pous.StreamedWipe.prfAdvantage`, which is that difference). Outside `A` it is `δ_net`
  (`Accounting.Wipe.Strategy.deltaNet`).
* **The ratio** (`ratioSearch_samples`, `ratio_samples`): `ratio`'s bisection keeps a lower end `lo ≥ 1` that samples,
  so at `k ≥ 1` reads `ratio k ≥ 1` and `(ratio k − 1)^k · 2^129 ≤ (ratio k)^k`, `StreamedWipeMeetsReads`' condition.
* **The decoded commitment** (`decodeCommitment_k`): `decodeCommitment` decodes with `StreamedWipe.Commitment.ofJson`
  (`exactly_decodes`), which returns only parameters `Params.check` passes (`commitment_ofJson_check`), and those have
  `1 ≤ batch ≤ k` (`params_check_k`), so at least one read.
* **What `check` publishes** (`check_accept`, `certify_certified`): an accepted `check` publishes `certify` of the
  commitment's parameters, and a certified one has `k ≤ MAX_READS`, `memory_bits` the declared tiers' bits,
  `eps = 1/2^128` and `foreign_bits = foreignBits p (ratio k) + egress_mib · 2^23`. From 1 to `MAX_READS` reads and no
  egress, `certify` certifies (`certify_zero`).
* **The honest and nontrivial lemmas** (`wipeSurvivorBoundHonest`, `wipeSurvivorBoundNontrivial`): the honest strategy
  reads back what it wrote; the cheating strategy keeps the published `foreign_bits` of foreign content in the store
  (`storeHold`, `storeKeep`) with probability 1, and with a verdict that accepts every run and every egress in `A` the
  PRF advantage and `δ_net` are 0 (`not_survivorBound_cheat`, on any device whose store holds the content). The
  segment's `ratio 2152 = 24` and its declared `2^28` bits are the kernel's to evaluate (`segment_foreign_le`).
-/

namespace MemoryChallenger.Properties.SecurityProofs

open Verity
open scoped ENNReal

/-- The value of a draw with no prover move is the fraction of the coins on which the event holds. -/
theorem value_draw {X : Type} [Fintype X] [Nonempty X] (p : X → Prop) [DecidablePred p] :
    Game.value p (Game.draw X) = ((Finset.univ.filter p).card : ℝ≥0∞) / Fintype.card X := by
  classical
  simp only [Game.draw, Game.value]
  rw [ENNReal.div_eq_inv_mul, Finset.card_filter, Nat.cast_sum, Finset.mul_sum]
  refine Finset.sum_congr rfl fun x _ => ?_
  by_cases hx : p x <;> simp [hx]

/-- The value of a draw with no prover move is the probability of its event. -/
theorem value_draw_pr {X : Type} [Fintype X] [Nonempty X] (p : X → Prop) :
    Game.value p (Game.draw X) = Pous.pr p := by
  classical
  rw [value_draw, Pous.pr_def]

/-- A union bound: an event inside `Q ∨ R` has probability at most `Q`'s plus `R`'s. -/
theorem pr_le_add {α : Type} [Fintype α] {P Q R : α → Prop} (h : ∀ x, P x → Q x ∨ R x) :
    Pous.pr P ≤ Pous.pr Q + Pous.pr R := by
  classical
  have hcard : (Finset.univ.filter P).card ≤ (Finset.univ.filter Q).card + (Finset.univ.filter R).card :=
    calc (Finset.univ.filter P).card ≤ (Finset.univ.filter Q ∪ Finset.univ.filter R).card :=
          Finset.card_le_card fun x hx => by
            simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_union] at hx ⊢
            exact h x hx
      _ ≤ _ := Finset.card_union_le _ _
  rw [Pous.pr_def, Pous.pr_def, Pous.pr_def, ← ENNReal.add_div, ← Nat.cast_add]
  gcongr

/-- Probability is invariant under a bijection of the coins. -/
theorem pr_congr_equiv {α β : Type} [Fintype α] [Fintype β] (e : α ≃ β) {P : α → Prop} {Q : β → Prop}
    (h : ∀ x, P x ↔ Q (e x)) : Pous.pr P = Pous.pr Q := by
  classical
  have hc : (Finset.univ.filter P).card = (Finset.univ.filter Q).card :=
    Finset.card_equiv e fun x => by simp [h x]
  rw [Pous.pr_def, Pous.pr_def, Fintype.card_congr e, hc]

/-- An event that never holds has probability 0. -/
theorem pr_eq_zero {α : Type} [Fintype α] {P : α → Prop} (h : ∀ x, ¬ P x) : Pous.pr P = 0 := by
  classical
  rw [Pous.pr_def, Finset.filter_false_of_mem fun x _ => h x, Finset.card_empty, Nat.cast_zero, ENNReal.zero_div]

/-- A distinguisher that always outputs true has no advantage. -/
theorem prfAdvantage_of_forall {N b κ : ℕ} {ω : Type} [Fintype ω] [Nonempty ω] [Nonempty (Fin N → Pous.Bits b)]
    (prf : Pous.Bits κ → Fin N → Pous.Bits b) {D : ω × (Fin N → Pous.Bits b) → Prop} (h : ∀ x, D x) :
    Pous.StreamedWipe.prfAdvantage prf D = 0 := by
  unfold Pous.StreamedWipe.prfAdvantage
  rw [Pous.pr_eq_one (P := fun x : ω × Pous.Bits κ => D (x.1, prf x.2)) fun _ => h _, Pous.pr_eq_one h, tsub_self]

/-- **The ideal game at `k` reads.** `StreamedWipeMeetsReads` (`PousStreamedWipe.streamed_wipe_meets_reads`) at a key
that is the whole label table: the labels are a bijection of the key, so the real game is the ideal one and the PRF's
advantage is 0. -/
theorem winIdeal_le_of_reads {n k r R q E : ℕ} (hn : 0 < n) (hr : 0 < r) (hk : (r - 1) ^ k * 2 ^ 129 ≤ r ^ k)
    (P : Pous.StreamedWipe.Prover n k 512 q (n * 512 + R) E)
    (hq : 512 * ((n + r - 1) / r - 1) + n + Nat.log2 k + 130 + R + E ≤ q) :
    Pous.StreamedWipe.winIdeal P ≤ (2⁻¹ : ℝ≥0∞) ^ 128 := by
  classical
  have hc : Fintype.card (Pous.Bits (n * 512)) = Fintype.card (Fin n → Pous.Bits 512) := by
    simp only [Fintype.card_fun, Fintype.card_bool, Fintype.card_fin, ← pow_mul, mul_comm]
  let e : Pous.Bits (n * 512) ≃ (Fin n → Pous.Bits 512) := Fintype.equivOfCardEq hc
  have hreal : Pous.StreamedWipe.winReal e P = Pous.StreamedWipe.winIdeal P := by
    unfold Pous.StreamedWipe.winReal Pous.StreamedWipe.winIdeal Pous.pr
    congr 2
    · exact Finset.card_equiv ((Equiv.refl _).prodCongr e) fun x => by simp; exact Iff.rfl
    · simp only [Fintype.card_prod, hc]
  have h := PousStreamedWipe.streamed_wipe_meets_reads r k hr hk n hn R q E (n * 512) e P hq
  rwa [Pous.StreamedWipe.prfAdvantage, ← Pous.StreamedWipe.winReal, ← Pous.StreamedWipe.winIdeal, hreal,
    tsub_self, add_zero] at h

/-- The device's state as a pair of bit strings: its chip and its store. -/
def stateEquiv {ι : Type} (d : Accounting.Device ι) :
    Verity.Device.State d ≃ (Fin d.chip → Bool) × (Fin d.mem → Bool) :=
  ⟨fun σ => (σ.chip, σ.store), fun x => ⟨x.1, x.2⟩, fun _ => rfl, fun _ => rfl⟩

noncomputable local instance instFintypeState {ι : Type} (d : Accounting.Device ι) :
    Fintype (Verity.Device.State d) :=
  Fintype.ofEquiv _ (stateEquiv d).symm

theorem card_state {ι : Type} (d : Accounting.Device ι) :
    Fintype.card (Verity.Device.State d) = 2 ^ d.chip * 2 ^ d.mem := by
  rw [Fintype.card_congr (stateEquiv d)]
  simp

/-- **The distinguisher on uniform labels.** For a strategy with `n ≥ 1` labels and `k` spot reads, a ratio `r ≥ 1` with
`(r − 1)^k · 2^129 ≤ r^k`, a verdict whose accepted runs answer every read with its label, and a set `A` of at most
`2^E` egress strings, the adversary's distinguisher outputs true on uniform labels with probability at most `2^-128`
once `S_φ` is at least `512·(⌈n/r⌉ − 1) + n + ⌊log₂ k⌋ + 130 + R + E + chip + (mem − (n · 512 + R))`. -/
theorem pr_distinguisher_le {ι : Type} {d : Accounting.Device ι} {C : Circuit} {B : ℕ} {s : Blocks C B} {b : Fin B}
    {n k r R : ℕ} (hn : 0 < n) (hr : 0 < r) (hk : (r - 1) ^ k * 2 ^ 129 ≤ r ^ k)
    (τ : Accounting.Wipe.Strategy d s b n k)
    (acc : Device.State d → Accounting.Wipe.Labels n → Accounting.Wipe.Reads n k → Prop)
    (hacc : ∀ σ L c, acc σ L c → ∀ i, τ.answer (τ.filled σ L).1 (τ.filled σ L).2
      (Pous.StreamedWipe.revealed c i) (c i) = L (c i))
    {K : ℕ} (A : Finset (Fin τ.egress → Bool)) (hA : A.card ≤ 2 ^ K) {S : ℕ}
    (hS : 512 * ((n + r - 1) / r - 1) + n + Nat.log2 k + 130 + R + (K + d.chip + (d.mem - (n * 512 + R))) ≤ S)
    (hold : Pous.Bits S → Device.State d)
    (keep : Device.State d → (Fin τ.egress → Bool) → Accounting.Wipe.Reads n k → Pous.Bits S) :
    Pous.pr (τ.distinguisher acc A hold keep) ≤ (2⁻¹ : ℝ≥0∞) ^ 128 := by
  classical
  rcases A.eq_empty_or_nonempty with rfl | ⟨a₀, ha₀⟩
  · rw [pr_eq_zero fun x hx => by simp [Accounting.Wipe.Strategy.distinguisher] at hx]
    exact zero_le
  have hcard : Fintype.card (Device.State d × A) ≤
      Fintype.card (Pous.Bits (n * 512 + R) × Pous.Bits (K + d.chip + (d.mem - (n * 512 + R)))) := by
    rw [Fintype.card_prod, Fintype.card_prod, card_state, Fintype.card_coe]
    simp only [Fintype.card_fun, Fintype.card_bool, Fintype.card_fin]
    calc 2 ^ d.chip * 2 ^ d.mem * A.card ≤ 2 ^ d.chip * 2 ^ d.mem * 2 ^ K := Nat.mul_le_mul_left _ hA
      _ = 2 ^ (d.chip + d.mem + K) := by rw [pow_add, pow_add]
      _ ≤ 2 ^ (n * 512 + R + (K + d.chip + (d.mem - (n * 512 + R)))) :=
          Nat.pow_le_pow_right (by norm_num) (by omega)
      _ = 2 ^ (n * 512 + R) * 2 ^ (K + d.chip + (d.mem - (n * 512 + R))) := pow_add _ _ _
  obtain ⟨enc⟩ := Function.Embedding.nonempty_of_card_le hcard
  have : Nonempty (Device.State d × A) := ⟨(⟨fun _ => false, fun _ => false⟩, ⟨a₀, ha₀⟩)⟩
  have hdec : ∀ x, Function.invFun enc (enc x) = x := Function.leftInverse_invFun enc.injective
  let P : Pous.StreamedWipe.Prover n k 512 S (n * 512 + R) (K + d.chip + (d.mem - (n * 512 + R))) :=
    { fill := fun z L => enc ((τ.filled (hold z) L).1,
        if h : (τ.filled (hold z) L).2 ∈ A then ⟨_, h⟩ else ⟨a₀, ha₀⟩)
      answer := fun σ ξ h i => τ.answer (Function.invFun enc (σ, ξ)).1 (Function.invFun enc (σ, ξ)).2.1 h i
      keep := fun σ ξ c => keep (Function.invFun enc (σ, ξ)).1 (Function.invFun enc (σ, ξ)).2.1 c }
  have hfill : ∀ z L (hm : (τ.filled (hold z) L).2 ∈ A),
      Function.invFun enc ((P.fill z L).1, (P.fill z L).2) = ((τ.filled (hold z) L).1, ⟨_, hm⟩) := by
    intro z L hm
    show Function.invFun enc (enc _) = _
    rw [hdec, dif_pos hm]
  calc Pous.pr (τ.distinguisher acc A hold keep) ≤ Pous.pr (Pous.StreamedWipe.wipeDistinguisher P) := by
        refine Pous.pr_mono fun x hx => ?_
        obtain ⟨hac, hkeep, hm⟩ := hx
        have hf := hfill x.1.1 x.2 hm
        refine ⟨fun j => ?_, ?_⟩
        · show τ.answer (Function.invFun enc ((P.fill x.1.1 x.2).1, (P.fill x.1.1 x.2).2)).1
              (Function.invFun enc ((P.fill x.1.1 x.2).1, (P.fill x.1.1 x.2).2)).2.1 _ _ = _
          rw [hf]
          exact hacc _ _ _ hac j
        · show keep (Function.invFun enc ((P.fill x.1.1 x.2).1, (P.fill x.1.1 x.2).2)).1
              (Function.invFun enc ((P.fill x.1.1 x.2).1, (P.fill x.1.1 x.2).2)).2.1 x.1.2 = x.1.1
          rw [hf]
          exact hkeep
    _ = Pous.StreamedWipe.winIdeal P := rfl
    _ ≤ _ := winIdeal_le_of_reads hn hr hk P hS

/-- **The streamed wipe's survivor bound on the device**, for every strategy. A strategy with `n ≥ 1` labels, `k` spot
reads and a fill of at most `n · 512 + R` bits, at a ratio `r ≥ 1` with `(r − 1)^k · 2^129 ≤ r^k`, under a verdict whose
accepted runs answer every read with its label, meets the survivor bound at
`512·(⌈n/r⌉ − 1) + n + ⌊log₂ k⌋ + 130 + R + E` bits and `ε₀ ≥ 2^-128` plus the PRF's advantage against the adversary's
distinguisher plus `δ_net`, at every set `A` of at most `2^E` egress strings. -/
theorem survivorBound_of_observed {ι : Type} {d : Accounting.Device ι} {C : Circuit} {B : ℕ} {s : Blocks C B}
    {b : Fin B} {n k r R : ℕ} [NeZero n] (hr : 0 < r) (hk : (r - 1) ^ k * 2 ^ 129 ≤ r ^ k)
    (τ : Accounting.Wipe.Strategy d s b n k) (hfill : τ.fill ≤ n * 512 + R)
    (acc : Device.State d → Accounting.Wipe.Labels n → Accounting.Wipe.Reads n k → Prop)
    (hacc : ∀ σ L c, acc σ L c → ∀ i, τ.answer (τ.filled σ L).1 (τ.filled σ L).2
      (Pous.StreamedWipe.revealed c i) (c i) = L (c i))
    {κ : ℕ} (prf : Pous.Bits κ → Accounting.Wipe.Labels n) {K : ℕ} (A : Finset (Fin τ.egress → Bool))
    (hA : A.card ≤ 2 ^ K) {foreignBits : ℕ}
    (hf : 512 * ((n + r - 1) / r - 1) + n + Nat.log2 k + 130 + R + K ≤ foreignBits) {ε₀ : ℝ≥0∞}
    (hε₀ : (2⁻¹ : ℝ≥0∞) ^ 128 ≤ ε₀) :
    (τ.wipe acc prf).SurvivorBound foreignBits fun _ hold keep =>
      ε₀ + Pous.StreamedWipe.prfAdvantage prf (τ.distinguisher acc A hold keep) + τ.deltaNet acc prf A hold := by
  classical
  intro S hS hold keep
  have hS' :
      512 * ((n + r - 1) / r - 1) + n + Nat.log2 k + 130 + R + (K + d.chip + (d.mem - (n * 512 + R))) ≤ S := by
    have : (τ.wipe acc prf).unfilled = d.chip + (d.mem - τ.fill) := rfl
    omega
  have hD := pr_distinguisher_le (Nat.pos_of_ne_zero (NeZero.ne n)) hr hk τ acc hacc A hA hS' hold keep
  let e : Pous.Bits S × Pous.Bits κ × Accounting.Wipe.Reads n k ≃
      (Pous.Bits S × Accounting.Wipe.Reads n k) × Pous.Bits κ :=
    ⟨fun x => ((x.1, x.2.2), x.2.1), fun y => (y.1.1, y.2, y.1.2), fun _ => rfl, fun _ => rfl⟩
  beta_reduce
  rw [value_draw_pr]
  unfold Accounting.Wipe.Strategy.deltaNet
  calc _ ≤ Pous.pr (fun x : Pous.Bits S × Pous.Bits κ × Accounting.Wipe.Reads n k =>
          τ.distinguisher acc A hold keep ((x.1, x.2.2), prf x.2.1)) +
        Pous.pr (fun x : Pous.Bits S × Pous.Bits κ × Accounting.Wipe.Reads n k =>
          acc (hold x.1) (prf x.2.1) x.2.2 ∧ (τ.filled (hold x.1) (prf x.2.1)).2 ∉ A) := by
        refine pr_le_add fun x hx => ?_
        by_cases hm : (τ.filled (hold x.1) (prf x.2.1)).2 ∈ A
        · exact Or.inl ⟨hx.1, hx.2, hm⟩
        · exact Or.inr ⟨hx.1, hm⟩
    _ = Pous.pr (fun y : (Pous.Bits S × Accounting.Wipe.Reads n k) × Pous.Bits κ =>
          τ.distinguisher acc A hold keep (y.1, prf y.2)) +
        Pous.pr (fun x : Pous.Bits S × Pous.Bits κ × Accounting.Wipe.Reads n k =>
          acc (hold x.1) (prf x.2.1) x.2.2 ∧ (τ.filled (hold x.1) (prf x.2.1)).2 ∉ A) := by
        congr 1
        exact pr_congr_equiv e fun _ => Iff.rfl
    _ ≤ Pous.pr (τ.distinguisher acc A hold keep) +
          Pous.StreamedWipe.prfAdvantage prf (τ.distinguisher acc A hold keep) +
        Pous.pr (fun x : Pous.Bits S × Pous.Bits κ × Accounting.Wipe.Reads n k =>
          acc (hold x.1) (prf x.2.1) x.2.2 ∧ (τ.filled (hold x.1) (prf x.2.1)).2 ∉ A) := by
        gcongr
        exact le_add_tsub
    _ ≤ _ := by
        gcongr
        exact hD.trans hε₀

/-- What a certified `certify` publishes: the session's `k` is at most `MAX_READS`, `eps` is `1/2^128`, `memory_bits`
the declared tiers' bits and `foreign_bits` `StreamedWipeMeetsReads`' threshold at `r = ratio k` plus
`egress_mib · 2^23`. -/
theorem certify_certified {p : StreamedWipe.Params} {e egressMib : Int} {eps : Core.Ratio} {fb k m r : Nat}
    (h : WipeProtocol.certify p e = ⟨true, egressMib, eps, fb, k, m, r⟩) :
    p.k ≤ WipeProtocol.MAX_READS ∧ eps = WipeProtocol.EPS ∧ m = WipeProtocol.memoryBits p ∧
      fb = WipeProtocol.foreignBits p (WipeProtocol.ratio p.k) + egressMib.toNat * WipeProtocol.UNIT_BITS := by
  unfold WipeProtocol.certify at h
  split_ifs at h with hg
  · simp only [WipeProtocol.Public.mk.injEq] at h
    obtain ⟨-, rfl, rfl, rfl, -, rfl, -⟩ := h
    simp only [WipeProtocol.gaps, List.isEmpty_iff, List.append_eq_nil_iff] at hg
    obtain ⟨hk, -⟩ := hg
    split_ifs at hk with h1
    have hk' : p.k ≤ WipeProtocol.MAX_READS := by omega
    exact ⟨hk', rfl, rfl, by simp [WipeProtocol.ratioOf, hk']⟩
  · simp at h

/-- From 1 to `MAX_READS` reads and no egress, `certify` certifies, at `eps = 1/2^128` and `foreign_bits` the threshold
at `r = ratio k`. -/
theorem certify_zero {q : StreamedWipe.Params} (hk0 : 1 ≤ q.k) (hk : q.k ≤ WipeProtocol.MAX_READS) :
    WipeProtocol.certify q 0 = ⟨true, 0, WipeProtocol.EPS, WipeProtocol.foreignBits q (WipeProtocol.ratio q.k), q.k,
      WipeProtocol.memoryBits q, WipeProtocol.ratio q.k⟩ := by
  have h0 : q.k ≠ 0 := by omega
  have h1 : ¬ WipeProtocol.MAX_READS < q.k := not_lt.mpr hk
  have hg : WipeProtocol.gaps q 0 = [] := by
    simp [WipeProtocol.gaps, h0, h1, StreamedWipe.EGRESS_MAX_UNITS]
  unfold WipeProtocol.certify
  rw [hg]
  simp [WipeProtocol.ratioOf, hk]

/-- `ratio`'s bisection keeps its lower end: from a lower end `lo ≥ 1` that samples, it returns one. -/
theorem ratioSearch_samples {k : ℕ} (fuel : ℕ) : ∀ {lo hi : ℕ}, 1 ≤ lo → WipeProtocol.samples lo k = true →
    1 ≤ WipeProtocol.ratioSearch k fuel lo hi ∧
      WipeProtocol.samples (WipeProtocol.ratioSearch k fuel lo hi) k = true := by
  induction fuel with
  | zero => intro lo hi h1 hs; exact ⟨h1, hs⟩
  | succ fuel ih =>
    intro lo hi h1 hs
    simp only [WipeProtocol.ratioSearch]
    split_ifs with hd hm
    · exact ⟨h1, hs⟩
    · exact ih (by omega) hm
    · exact ih h1 hs

/-- At `k ≥ 1` reads, `ratio k` is at least 1 and meets `StreamedWipeMeetsReads`' condition
`(r − 1)^k · 2^129 ≤ r^k`: the bisection starts from `r = 1`, where `0^k · 2^129 = 0`. -/
theorem ratio_samples {k : ℕ} (hk : 0 < k) :
    0 < WipeProtocol.ratio k ∧ (WipeProtocol.ratio k - 1) ^ k * 2 ^ 129 ≤ WipeProtocol.ratio k ^ k := by
  have h1 : WipeProtocol.samples 1 k = true := by
    simp [WipeProtocol.samples, zero_pow hk.ne']
  obtain ⟨hr, hs⟩ := ratioSearch_samples (k := k) 64 (hi := k / 89 + 2) le_rfl h1
  simp only [WipeProtocol.samples, decide_eq_true_eq] at hs
  exact ⟨hr, hs⟩

/-- An accepted `check` publishes `certify` of the commitment's parameters at some egress count. -/
theorem check_accept {schemes : Schemes} {p : WipeProtocol.Params} {c : StreamedWipe.Commitment}
    {e : WipeProtocol.Evidence} {pub : WipeProtocol.Public} {profile : StreamedWipe.Params × Int}
    (h : WipeProtocol.check schemes p c e = .accept pub profile) :
    ∃ egress : Int, WipeProtocol.certify c.params egress = pub := by
  unfold WipeProtocol.check at h
  split at h
  · cases h
  · split at h
    · cases h
    · dsimp only at h
      split at h
      · injection h with h1
        exact ⟨_, h1⟩
      · cases h

/-- What `exactly` accepts is what its decoder decoded. -/
theorem exactly_decodes {α : Type} {f : Lean.Json → Flock.V α} {enc : α → Lean.Json} {j : Lean.Json} {a : α}
    (h : MemoryChallenger.exactly f enc j = .ok a) : f j = .ok a := by
  unfold MemoryChallenger.exactly at h
  cases hf : f j with
  | error e => simp [hf, bind, Except.bind] at h
  | ok a0 =>
    cases hc1 : Flock.canon (enc a0) with
    | error e => simp [hf, hc1, bind, Except.bind] at h
    | ok s1 =>
      cases hc2 : Flock.canon j with
      | error e => simp [hf, hc1, hc2, bind, Except.bind] at h
      | ok s2 =>
        by_cases hne : s1 = s2
        · subst hne
          simp [hf, hc1, hc2, bind, Except.bind, pure, Except.pure] at h
          subst h
          rfl
        · simp [hf, hc1, hc2, hne, bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at h

/-- A commitment `StreamedWipe.Commitment.ofJson` decodes has parameters `Params.check` passes. -/
theorem commitment_ofJson_check {j : Lean.Json} {c : StreamedWipe.Commitment}
    (h : StreamedWipe.Commitment.ofJson j = .ok c) : c.params.check = .ok () := by
  unfold StreamedWipe.Commitment.ofJson at h
  simp only [bind, Except.bind] at h
  repeat (split at h; next => cases h)
  split at h
  · rename_i hp
    repeat (split at h; next => cases h)
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact hp
  · simp [throw, throwThe, MonadExceptOf.throw] at h

/-- Parameters `Params.check` passes have at least one read: `1 ≤ batch ≤ k`. -/
theorem params_check_k {q : StreamedWipe.Params} (h : q.check = .ok ()) : 0 < q.k := by
  rcases Nat.eq_zero_or_pos q.k with hk | hk
  · exfalso
    revert h
    unfold StreamedWipe.Params.check
    rcases Nat.eq_zero_or_pos q.batch with hb | hb
    · simp [hb, throw, throwThe, MonadExceptOf.throw, bind, Except.bind]
    · simp [hk, hb, throw, throwThe, MonadExceptOf.throw, bind, Except.bind]
  · exact hk

/-- **A commitment the protocol decodes has at least one read**: `decodeCommitment` decodes it with
`StreamedWipe.Commitment.ofJson`, whose `Params.check` refuses `k < batch` and `batch < 1`. -/
theorem decodeCommitment_k {bytes : ByteArray} {c : StreamedWipe.Commitment}
    (h : WipeProtocol.decodeCommitment bytes = .ok c) : 0 < c.params.k := by
  unfold WipeProtocol.decodeCommitment at h
  cases hj : Core.canonicalJson bytes with
  | error e => simp [hj, bind, Except.bind] at h
  | ok j =>
    simp only [hj, bind, Except.bind] at h
    exact params_check_k (commitment_ofJson_check (exactly_decodes h))

theorem wipeSurvivorBound : WipeSurvivorBound := by
  intro schemes p bytes c e egressMib eps foreignBits k memoryBits r profile hdec h ι d C B s b _ τ κ prf count A
    hfill hA
  have hk0 := decodeCommitment_k hdec
  obtain ⟨egress, hc⟩ := check_accept h
  obtain ⟨-, rfl, rfl, rfl⟩ := certify_certified hc
  obtain ⟨hr, hs⟩ := ratio_samples hk0
  have heps : ((1 : ℕ) : ℝ≥0∞) / ((2 ^ 128 : ℕ) : ℝ≥0∞) = (2⁻¹ : ℝ≥0∞) ^ 128 := by
    rw [Nat.cast_one, Nat.cast_pow, Nat.cast_ofNat, one_div, ENNReal.inv_pow]
  have hfill' : τ.fill ≤ c.params.labels * 512 + WipeProtocol.outsideBits c.params := by
    simp only [WipeProtocol.outsideBits, WipeProtocol.fillBits, StreamedWipe.LABEL_BYTES]
    omega
  exact survivorBound_of_observed (R := WipeProtocol.outsideBits c.params) hr hs τ hfill' _ (fun _ _ _ h => h.1) prf A
    hA (le_of_eq rfl) heps.ge

/-! ## The honest and nontrivial lemmas -/

theorem honestRead_honestFill {ι : Type} {d : Accounting.Device ι} {n : ℕ} (h : n * 512 ≤ d.mem)
    (σ : Device.State d) (L : Accounting.Wipe.Labels n) (i : Fin n) : honestRead h (honestFill σ L) i = L i := by
  funext t
  have hi := i.isLt
  have ht := t.isLt
  simp only [honestRead, honestFill]
  split_ifs with ha
  · congr 2 <;> omega
  · omega

theorem wipeSurvivorBoundHonest : WipeSurvivorBoundHonest := by
  refine ⟨fun q hk0 hk => certify_zero hk0 hk, ?_⟩
  intro ι d C B s b n k h carry
  refine ⟨rfl, ?_, fun σ L c => ⟨fun i => ?_, le_rfl⟩⟩
  · show (Finset.univ : Finset (Fin 0 → Bool)).card ≤ 2 ^ 0
    simp
  · exact honestRead_honestFill h σ L (c i)

/-- Foreign content held in the first `S` bits of the device's store. -/
def storeHold {ι : Type} (d : Accounting.Device ι) {S : ℕ} (z : Pous.Bits S) : Device.State d :=
  ⟨fun _ => false, fun a => if ha : a.val < S then z ⟨a.val, ha⟩ else false⟩

/-- The foreign content read back from the first `S` bits of the store. -/
def storeKeep {ι : Type} {d : Accounting.Device ι} {S : ℕ} (hS : S ≤ d.mem) {E : ℕ} {Ch : Type}
    (σ : Device.State d) (_ : Fin E → Bool) (_ : Ch) : Pous.Bits S :=
  fun i => σ.store ⟨i.val, lt_of_lt_of_le i.isLt hS⟩

theorem storeKeep_hold {ι : Type} {d : Accounting.Device ι} {S : ℕ} (hS : S ≤ d.mem) {E : ℕ} {Ch : Type}
    (z : Pous.Bits S) (ξ : Fin E → Bool) (c : Ch) : storeKeep hS (storeHold d z) ξ c = z := by
  funext i
  simp [storeKeep, storeHold]

/-- A strategy that keeps the whole store, under a verdict that accepts every run, fails the survivor bound at any
`ε₀ < 1` on a device whose store holds `F` bits of foreign content beside its on-chip state: the PRF's advantage and
`δ_net` are 0 there, and the foreign content is read back with probability 1. -/
theorem not_survivorBound_cheat {ι : Type} (d : Accounting.Device ι) {C : Circuit} {B : ℕ} (s : Blocks C B)
    (b : Fin B) {n k : ℕ} [NeZero n] (carry : Device.State d → s.Carried b) {κ : ℕ}
    (prf : Pous.Bits κ → Accounting.Wipe.Labels n) {F : ℕ} (hF : F + d.chip ≤ d.mem) {ε₀ : ℝ≥0∞} (hε : ε₀ < 1) :
    ¬ ((cheat d s b n k carry).wipe (fun _ _ _ => True) prf).SurvivorBound F fun _ hold keep =>
        ε₀ + Pous.StreamedWipe.prfAdvantage prf
            ((cheat d s b n k carry).distinguisher (fun _ _ _ => True) Finset.univ hold keep) +
          (cheat d s b n k carry).deltaNet (fun _ _ _ => True) prf Finset.univ hold := by
  classical
  intro hsb
  have h := hsb (F + d.chip) (by show F + (d.chip + (d.mem - d.mem)) ≤ F + d.chip; omega) (storeHold d)
    (storeKeep hF)
  have hadv : Pous.StreamedWipe.prfAdvantage prf
      ((cheat d s b n k carry).distinguisher (fun _ _ _ => True) Finset.univ (storeHold d) (storeKeep hF)) = 0 :=
    prfAdvantage_of_forall prf fun x => ⟨trivial, storeKeep_hold hF _ _ _, Finset.mem_univ _⟩
  have hdelta :
      (cheat d s b n k carry).deltaNet (fun _ _ _ => True) prf Finset.univ (storeHold d (S := F + d.chip)) = 0 :=
    pr_eq_zero fun x hx => hx.2 (Finset.mem_univ _)
  have key : ε₀ + Pous.StreamedWipe.prfAdvantage prf
        ((cheat d s b n k carry).distinguisher (fun _ _ _ => True) Finset.univ (storeHold d) (storeKeep hF)) +
      (cheat d s b n k carry).deltaNet (fun _ _ _ => True) prf Finset.univ (storeHold d (S := F + d.chip)) = ε₀ := by
    rw [hadv, hdelta, add_zero, add_zero]
  beta_reduce at h
  rw [value_draw_pr, Pous.pr_eq_one] at h
  · exact absurd (h.trans_eq key) (not_le.mpr hε)
  · exact fun x => ⟨trivial, storeKeep_hold hF _ _ _⟩

theorem segment_reads_pos : 1 ≤ segmentParams.k := by
  show 1 ≤ 2152
  norm_num

theorem segment_reads : segmentParams.k ≤ WipeProtocol.MAX_READS := by
  show 2152 ≤ 2 ^ 17
  norm_num

/-- The segment's published `foreign_bits`, 11,709,069 bits at `r = ratio 2152 = 24`, fit in its store beside its
on-chip state. -/
theorem segment_foreign_le :
    (WipeProtocol.certify segmentParams 0).foreignBits + segmentDevice.chip ≤ segmentDevice.mem := by
  have hr : WipeProtocol.ratio segmentParams.k = 24 := by decide +kernel
  have hm : WipeProtocol.memoryBits segmentParams = 2 ^ 28 := by decide +kernel
  have hl : Nat.log2 2152 < 12 := (Nat.log2_lt (by norm_num)).2 (by norm_num)
  rw [certify_zero segment_reads_pos segment_reads]
  simp only [WipeProtocol.foreignBits, WipeProtocol.outsideBits, WipeProtocol.fillBits, hr, hm]
  show 64 * 8 * ((2 ^ 19 + 24 - 1) / 24 - 1) + 2 ^ 19 + Nat.log2 2152 + 130 + (2 ^ 28 - 2 ^ 19 * 64 * 8) + 0 ≤
    2 ^ 19 * 512
  generalize Nat.log2 2152 = l at hl ⊢
  norm_num
  omega

/-- The segment's published `eps` is below 1. -/
theorem segment_eps_lt :
    ((WipeProtocol.certify segmentParams 0).eps.num : ℝ≥0∞) / (WipeProtocol.certify segmentParams 0).eps.den < 1 := by
  rw [certify_zero segment_reads_pos segment_reads]
  show ((1 : ℕ) : ℝ≥0∞) / ((2 ^ 128 : ℕ) : ℝ≥0∞) < 1
  rw [Nat.cast_one, one_div, ENNReal.inv_lt_one]
  exact_mod_cast (by norm_num : (1 : ℕ) < 2 ^ 128)

theorem wipeSurvivorBoundNontrivial : WipeSurvivorBoundNontrivial := by
  intro C B s b carry κ prf
  exact not_survivorBound_cheat segmentDevice s b carry prf segment_foreign_le segment_eps_lt

end MemoryChallenger.Properties.SecurityProofs
