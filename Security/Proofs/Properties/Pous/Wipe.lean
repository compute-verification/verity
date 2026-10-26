import Properties.Pous.Wipe
import Proofs.Pous.StreamedWipe.Family
import Mathlib.Tactic.SplitIfs

/-!
# Proof that an accepted wipe meets the survivor bound it publishes (`Properties.Pous.Wipe`)

* **The device is a prover of the streamed game** (`survivorBound_of_reads`). The device's state when the fill ends and
  the egress, `chip + mem + E` bits, go one-to-one into the game's inside, the `n · 512` bits the fill covers, and its
  egress, `E' = E + chip + (mem − n · 512)`: the on-chip state and the store the fill leaves out are charged as egress.
  `hold` and the device's fill are the prover's fill, `StreamedReads`'s `answer` its answers and `keep` its keep, so a
  run the challenger accepts with `keep` returning `z` is one the prover wins. `SurvivorBound`'s `S_φ` is at least
  `⌈n · 512/19⌉ + E + unfilled` (`unfilled = chip + (mem − fill)`), the family's `⌈M/19⌉ + E'`.
* **The ideal game from the family** (`winIdeal_le_of_family`): `StreamedWipeMeetsFamily`, proved
  (`PousStreamedWipe.streamed_wipe_meets_family`), at a key that is the whole label table, whose labels are a bijection
  of the key, so its real game is the ideal one and its PRF's advantage 0.
* **What `check` publishes** (`check_accept`, `certify_certified`): an accepted `check` publishes `certify` of the
  commitment's parameters, and a certified one has `k = 2152`, at least `2^19` labels, `memory_bits = labels · 512`,
  `eps = 1/2^128` and `foreign_bits = ⌈M/19⌉ + egress_mib · 2^23`.
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

/-- **The ideal game from the family.** `StreamedWipeMeetsFamily` (`PousStreamedWipe.streamed_wipe_meets_family`) at a
key that is the whole label table: the labels are a bijection of the key, so the real game is the ideal one and the
PRF's advantage is 0. -/
theorem winIdeal_le_of_family {n q E : ℕ} (hn : 2 ^ 19 ≤ n)
    (P : Pous.StreamedWipe.Prover n 2152 512 q (n * 512) E) (hq : (n * 512 + 18) / 19 + E ≤ q) :
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
  have h := PousStreamedWipe.streamed_wipe_meets_family n hn q E (n * 512) e P hq
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

/-- **The streamed wipe's survivor bound on the device.** A wipe of `n ≥ 2^19` labels with 2152 spot reads, a fill of
at most `n · 512` bits and its reads streamed meets the survivor bound at `⌈n · 512/19⌉ + E` bits and `2^-128`, from
the streamed wipe's family bound (`PousStreamedWipe.streamed_wipe_meets_family`). The device's state and the egress are
a prover of the streamed game whose inside is the `n · 512` bits the fill covers and whose egress is `E` plus the
on-chip state plus the store the fill leaves out. -/
theorem survivorBound_of_reads {ι : Type} {d : Accounting.Device ι} {C : Circuit} {B : ℕ} {s : Blocks C B} {b : Fin B}
    {n : ℕ}
    [NeZero n] (hn : 2 ^ 19 ≤ n)
    (w : Verity.Wipe d s b (Accounting.Wipe.Labels n) (Accounting.Wipe.Reads n 2152)) (hfill : w.fill ≤ n * 512)
    (hreads : Accounting.Assumptions.StreamedReads w) {foreignBits : ℕ}
    (hf : (n * 512 + 18) / 19 + w.egress ≤ foreignBits) :
    w.SurvivorBound foreignBits (fun _ _ _ => (2⁻¹ : ℝ≥0∞) ^ 128) := by
  classical
  intro S hS hold keep
  obtain ⟨ans, hans⟩ := hreads
  have hcard : Fintype.card (Verity.Device.State d × (Fin w.egress → Bool)) ≤
      Fintype.card (Pous.Bits (n * 512) × Pous.Bits (w.egress + d.chip + (d.mem - n * 512))) := by
    simp only [Fintype.card_prod, card_state, Fintype.card_fun, Fintype.card_bool, Fintype.card_fin, ← pow_add]
    exact Nat.pow_le_pow_right (by norm_num) (by omega)
  obtain ⟨enc⟩ := Function.Embedding.nonempty_of_card_le hcard
  have : Nonempty (Verity.Device.State d × (Fin w.egress → Bool)) :=
    ⟨(⟨fun _ => false, fun _ => false⟩, fun _ => false)⟩
  have hdec : ∀ x, Function.invFun enc (enc x) = x := Function.leftInverse_invFun enc.injective
  let P : Pous.StreamedWipe.Prover n 2152 512 S (n * 512) (w.egress + d.chip + (d.mem - n * 512)) :=
    { fill := fun z L => enc (w.filled (hold z) L)
      answer := fun σ ξ h i => ans (Function.invFun enc (σ, ξ)).1 (Function.invFun enc (σ, ξ)).2 h i
      keep := fun σ ξ c => keep (Function.invFun enc (σ, ξ)).1 (Function.invFun enc (σ, ξ)).2 c }
  have hideal := winIdeal_le_of_family hn P (by unfold Verity.Wipe.unfilled at hS; omega)
  set A : (Fin S → Bool) × Accounting.Wipe.Labels n × Accounting.Wipe.Reads n 2152 → Prop := fun x =>
    w.accepts (hold x.1) x.2.1 x.2.2 ∧
      keep (w.filled (hold x.1) x.2.1).1 (w.filled (hold x.1) x.2.1).2 x.2.2 = x.1
  have hwin : ∀ x, A x → Pous.StreamedWipe.wipeDistinguisher P ((x.1, x.2.2), x.2.1) := by
    rintro ⟨z, L, c⟩ ⟨hacc, hkeep⟩
    refine ⟨fun i => ?_, ?_⟩
    · show ans (Function.invFun enc (enc _)).1 (Function.invFun enc (enc _)).2 _ _ = _
      rw [hdec]
      exact hans _ _ _ hacc i
    · show keep (Function.invFun enc (enc _)).1 (Function.invFun enc (enc _)).2 c = z
      rw [hdec]
      exact hkeep
  have hle : (Finset.univ.filter A).card ≤ (Finset.univ.filter (Pous.StreamedWipe.wipeDistinguisher P)).card :=
    Finset.card_le_card_of_injOn (fun x => ((x.1, x.2.2), x.2.1))
      (fun x hx => by simpa using hwin x (Finset.mem_filter.1 hx).2)
      (fun x _ y _ h => by
        simp only [Prod.mk.injEq] at h
        exact Prod.ext h.1.1 (Prod.ext h.2 h.1.2))
  have hX : Fintype.card ((Fin S → Bool) × Accounting.Wipe.Labels n × Accounting.Wipe.Reads n 2152) =
      Fintype.card ((Pous.Bits S × Accounting.Wipe.Reads n 2152) × Accounting.Wipe.Labels n) := by
    simp only [Fintype.card_prod]; ring
  calc Game.value A (Game.draw _)
      = ((Finset.univ.filter A).card : ℝ≥0∞) /
          Fintype.card ((Fin S → Bool) × Accounting.Wipe.Labels n × Accounting.Wipe.Reads n 2152) := value_draw A
    _ ≤ ((Finset.univ.filter (Pous.StreamedWipe.wipeDistinguisher P)).card : ℝ≥0∞) /
          Fintype.card ((Pous.Bits S × Accounting.Wipe.Reads n 2152) × Accounting.Wipe.Labels n) := by
        rw [hX]; gcongr
    _ = Pous.StreamedWipe.winIdeal P := by unfold Pous.StreamedWipe.winIdeal Pous.pr; congr
    _ ≤ _ := hideal

/-- What a certified `certify` publishes: the session's `k` is 2152, its fill at least one segment (`2^19` labels),
`memory_bits` its `labels · 512` bits, the egress count non-negative, `eps` `1/2^128` and `foreign_bits`
`⌈M/19⌉ + egress_mib · 2^23`. -/
theorem certify_certified {p : StreamedWipe.Params} {e egressMib : Int} {eps : Core.Ratio} {foreignBits m : Nat}
    (h : WipeProtocol.certify p e = ⟨true, egressMib, eps, foreignBits, m⟩) :
    p.k = 2152 ∧ 2 ^ 19 ≤ p.labels ∧ m = p.labels * 512 ∧ 0 ≤ egressMib ∧ eps = ⟨1, 2 ^ 128⟩ ∧
      foreignBits = (p.labels * 512 + 18) / 19 + egressMib.toNat * 2 ^ 23 := by
  unfold WipeProtocol.certify at h
  split_ifs at h with hg
  · simp only [WipeProtocol.Public.mk.injEq] at h
    obtain ⟨-, rfl, rfl, rfl, rfl⟩ := h
    simp only [WipeProtocol.gaps, List.isEmpty_iff, List.append_eq_nil_iff] at hg
    obtain ⟨⟨hk, hb⟩, he⟩ := hg
    split_ifs at hk hb he with h1 h2 h3
    simp only [bne_iff_ne, ne_eq, Decidable.not_not] at h1
    simp only [Bool.or_eq_true, decide_eq_true_eq, not_or, not_lt, not_le] at h3
    refine ⟨h1, ?_, ?_, h3.1, rfl, ?_⟩
    · simp only [StreamedWipe.LABEL_BYTES, WipeProtocol.LEAST_FILL_BYTES, not_lt] at h2; omega
    · simp only [WipeProtocol.memoryBits, StreamedWipe.LABEL_BYTES]; ring
    · simp only [WipeProtocol.memoryBits, StreamedWipe.LABEL_BYTES, WipeProtocol.F, WipeProtocol.UNIT_BITS]; omega
  · simp at h

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

theorem wipeSurvivorBound : WipeSurvivorBound := by
  intro schemes p c e egressMib eps foreignBits memoryBits profile h ι d C B s b _ w hfill hegress hreads
  obtain ⟨egress, hc⟩ := check_accept h
  obtain ⟨hk, hn, hm, -, rfl, rfl⟩ := certify_certified hc
  have heps : ((1 : ℕ) : ℝ≥0∞) / ((2 ^ 128 : ℕ) : ℝ≥0∞) = (2⁻¹ : ℝ≥0∞) ^ 128 := by
    rw [Nat.cast_one, Nat.cast_pow, Nat.cast_ofNat, one_div, ENNReal.inv_pow]
  show w.SurvivorBound _ (fun _ _ _ => ((1 : ℕ) : ℝ≥0∞) / ((2 ^ 128 : ℕ) : ℝ≥0∞))
  simp only [heps]
  revert w
  generalize c.params.k = k at hk ⊢
  subst hk
  intro w hfill hegress hreads
  simp only [WipeProtocol.UNIT_BITS] at hegress
  exact survivorBound_of_reads hn w (by omega) hreads (by omega)

end MemoryChallenger.Properties.SecurityProofs
