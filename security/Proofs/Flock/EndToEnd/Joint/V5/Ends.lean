module

public import Proofs.Flock.Bridge.QsRows.FirstReads
public import Proofs.Flock.EndToEnd.Joint.V5.Member

/-!
# Some outcome ends the window's sample: a stream of zeros

`JointAuditV5.hend` says some outcome of the verifier's byte stream ends the deployed sample. Under A3 every stream of
bytes is an outcome's, and a stream of zeros ends the sample of every sampling design on strata (`Bridge.LawOn`:
`subset`, `stratified` or `work`) whose stages' first reads it holds:

* `uniformL_zeros`: the rejection sampler accepts a word of zeros, and returns `0`.
* `not_failL_zeros`, `subset_zeros`: so `Flock.Draw.subset k n` samples on `k·width n` bytes of zeros, each of its `k`
  steps taking the one word it reads.
* `firstOk_zeros`, `stage_zeros`: on a stream of zeros every stage's first read is a block of zeros (`extract_zeros`),
  on which its sampler samples.
* `Flock.Bridge.units_ok_zeros`: the deployed sample ends on a stream of zeros of at least `Bridge.firstRead law n`
  bytes (`units_ok_of_firstOk`).
* `JointAuditV5.hend_of_zeros`: `JointAuditV5.hend` under A3, where the stream holds `Bridge.firstRead law nJ` bytes.

The stream's length is needed: the deployed sample takes each stage's `need` bytes before it samples, so on a stream
shorter than `Bridge.firstRead law nJ` it runs out at every outcome.
-/

@[expose] public section

namespace FlockSoundness.Audit.DrawExec

/-! ## The sampler on a stream of zeros -/

/-- The bytes of a stream of zeros. -/
theorem bsOf_zeros (ℓ : ℕ) : bsOf (fun _ : Fin ℓ => (0 : Fin 256)) = List.replicate ℓ (0 : UInt8) := by
  refine List.ext_getElem (by rw [length_bsOf, List.length_replicate]) fun i h₁ h₂ => ?_
  simp only [bsOf, List.getElem_ofFn, List.getElem_replicate]
  rfl

/-- A word of zeros is `0`. -/
theorem beL_zeros (w : ℕ) : beL (List.replicate w (0 : UInt8)) = 0 := by
  induction w with
  | zero => rfl
  | succ w ih =>
    rw [List.replicate_succ', beL_snoc, ih]
    rfl

/-- **The rejection sampler accepts a word of zeros**: on a stream of zeros that holds one word, `uniform m` returns
`0` and the stream after that word, `0` being below every multiple of `m` it compares a word with. -/
theorem uniformL_zeros (m : ℕ) (hm : 0 < m) (fuel ℓ : ℕ) (hℓ : Flock.Draw.width m ≤ ℓ) :
    uniformL m (fuel + 1) (List.replicate ℓ (0 : UInt8)) =
      some (0, List.replicate (ℓ - Flock.Draw.width m) (0 : UInt8)) := by
  have hlen : ¬ (List.replicate ℓ (0 : UInt8)).length < Flock.Draw.width m := by
    rw [List.length_replicate]
    omega
  have htake : (List.replicate ℓ (0 : UInt8)).take (Flock.Draw.width m) =
      List.replicate (Flock.Draw.width m) (0 : UInt8) := by
    rw [List.take_replicate, Nat.min_eq_left hℓ]
  have hlt : beL (List.replicate (Flock.Draw.width m) (0 : UInt8)) <
      2 ^ (8 * Flock.Draw.width m) - 2 ^ (8 * Flock.Draw.width m) % m := by
    rw [beL_zeros]
    have h1 := reject_nat m hm
    have h2 : 2 ^ (8 * Flock.Draw.width m) % m * 2 ≤ 2 ^ (8 * Flock.Draw.width m) % m * 2 ^ 64 :=
      Nat.mul_le_mul_left _ (by decide)
    have h3 : 0 < 2 ^ (8 * Flock.Draw.width m) := Nat.two_pow_pos _
    omega
  rw [uniformL, ite_eq_right hlen, htake, ite_eq_left hlt, beL_zeros, Nat.zero_mod, List.drop_replicate]

/-- **The loop does not run out on a stream of zeros** that holds a word for each of its `c` steps from `i`: every
step's word is zeros, which the rejection sampler accepts (`uniformL_zeros`), so each step reads one word. -/
theorem not_failL_zeros (n F : ℕ) : ∀ (c i ℓ : ℕ), i + c ≤ n → c * Flock.Draw.width n ≤ ℓ →
    ¬ failL n (F + 1) (List.range' i c) (List.replicate ℓ (0 : UInt8)) := by
  intro c
  induction c with
  | zero =>
    intro i ℓ _ _ hf
    simp only [List.range'_zero, failL] at hf
  | succ c ih =>
    intro i ℓ hi hℓ hf
    have hm : 0 < n - i := by omega
    have hwn : Flock.Draw.width (n - i) ≤ Flock.Draw.width n := drawWidth_mono (Nat.sub_le n i)
    have hℓ2 : c * Flock.Draw.width n + Flock.Draw.width n ≤ ℓ :=
      Nat.le_trans (Nat.le_of_eq (Nat.add_one_mul c (Flock.Draw.width n)).symm) hℓ
    rw [List.range'_succ] at hf
    simp only [failL, uniformL_zeros (n - i) hm F ℓ (by omega)] at hf
    exact ih (i + 1) (ℓ - Flock.Draw.width (n - i)) (by omega) (by omega) hf

/-- **`Flock.Draw.subset k n` samples on a stream of zeros** of at least `k·width n` bytes. -/
theorem subset_zeros (k n N : ℕ) (hk : k ≤ n) (hN : k * Flock.Draw.width n ≤ N) :
    ∃ D, Flock.Draw.subset k n (Law.streamOf fun _ : Fin N => (0 : Fin 256)) = some D := by
  cases h : Flock.Draw.subset k n (Law.streamOf fun _ : Fin N => (0 : Fin 256)) with
  | some D => exact ⟨D, rfl⟩
  | none =>
    have hf := subset_none_failL (fun _ : Fin N => (0 : Fin 256)) k n h
    rw [bsOf_zeros] at hf
    exact absurd hf (not_failL_zeros n N k 0 N (by omega) hN)

end FlockSoundness.Audit.DrawExec

namespace Properties.Spot.SecurityProofs

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Audit.DrawExec Finset

/-! ## Every stage's first read of a stream of zeros samples -/

/-- **A block of a stream of zeros is a stream of zeros.** -/
theorem extract_zeros (L p N : ℕ) (h : p + N ≤ L) :
    (Law.streamOf fun _ : Fin L => (0 : Fin 256)).extract p (p + N) =
      Law.streamOf fun _ : Fin N => (0 : Fin 256) := by
  refine ByteArray.ext (Array.ext' ?_)
  rw [ByteArray.data_extract, Array.toList_extract, streamOf_data, streamOf_data, bsOf_zeros, bsOf_zeros,
    List.extract_eq_take_drop, List.drop_replicate, List.take_replicate, Nat.add_sub_cancel_left,
    Nat.min_eq_left (by omega)]

/-- **On a stream of zeros every stage's first read samples**, where each stage's sampler samples on its `need` bytes
of zeros and the stream holds the stages' first reads from `p`. -/
theorem firstOk_zeros (L : ℕ) : ∀ (ss : List Stage) (p : ℕ),
    (∀ s ∈ ss, ∃ D, s.run (Law.streamOf fun _ : Fin s.need => (0 : Fin 256)) = some D) →
    p + (ss.map (·.need)).sum ≤ L → FirstOk ss (Law.streamOf fun _ : Fin L => (0 : Fin 256)) p
  | [], _, _, _ => trivial
  | s :: ss, p, hrun, hL => by
    have hL' : p + (s.need + (ss.map (·.need)).sum) ≤ L := hL
    have hle : p + s.need ≤ (Law.streamOf fun _ : Fin L => (0 : Fin 256)).size := by
      rw [size_streamOf]
      omega
    obtain ⟨D, hD⟩ := hrun s List.mem_cons_self
    have hD' : s.run ((Law.streamOf fun _ : Fin L => (0 : Fin 256)).extract p (p + s.need)) = some D := by
      rw [extract_zeros L p s.need (by omega)]
      exact hD
    exact ⟨hle, ⟨D, hD'⟩,
      firstOk_zeros L ss (p + s.need) (fun s' hs' => hrun s' (List.mem_cons_of_mem _ hs')) (by omega)⟩

/-- **A stratum's stage samples on its first read of zeros**, where it samples at most its size: its first read holds
a word for each of its `k` steps. -/
theorem stage_zeros (s : Flock.Draw.Stratum) (hk : s.k ≤ s.size) (miss : Array ℕ → Prop) :
    ∃ D, (stage s miss).run (Law.streamOf fun _ : Fin (stage s miss).need => (0 : Fin 256)) = some D := by
  have hN : s.k * Flock.Draw.width s.size ≤ 2 * s.k * Flock.Draw.width s.size + 64 := by
    rw [Nat.mul_assoc]
    omega
  exact subset_zeros s.k s.size (2 * s.k * Flock.Draw.width s.size + 64) hk hN

end Properties.Spot.SecurityProofs

namespace Flock.Bridge

open FlockSoundness FlockSoundness.Game Finset

/-! ## The deployed sample ends on a stream of zeros -/

/-- **Every stratum's stage samples on its first read of zeros**, each stratum sampling at most its size. -/
theorem strata_zeros (St : Array Flock.Draw.Stratum) (hks : ∀ s ∈ St.toList, s.k ≤ s.size) :
    ∀ s ∈ (List.range' 0 St.size).map (fun i => Properties.Spot.SecurityProofs.stage St[i]! fun _ => True),
      ∃ D, s.run (Audit.Law.streamOf fun _ : Fin s.need => (0 : Fin 256)) = some D := by
  intro s hs
  obtain ⟨i, hi, rfl⟩ := List.mem_map.1 hs
  exact Properties.Spot.SecurityProofs.stage_zeros St[i]! (hks _ (stratum_mem_toList St hi)) fun _ => True

/-- **The deployed sample ends on a stream of zeros** that holds its stages' first reads, under a sampling design on
strata that samples `kd` units (`LawOn`): every stage's first read is a block of zeros, on which its sampler samples
(`firstOk_zeros`), and where every first read samples the sample ends (`units_ok_of_firstOk`). -/
theorem units_ok_zeros {law : Flock.Draw.Law} {ns : List ℕ} {kd n : ℕ} (hlaw : LawOn law ns kd)
    (hns : ns.sum = n) (L : ℕ) (hL : firstRead law n ≤ L) :
    ∃ units read, Spot.Draw.drawUnits law n (Properties.Spot.bytesOf fun _ : Fin L => (0 : Fin 256)) =
      .ok (units, read) := by
  have hL' : 0 + ((Properties.Spot.SecurityProofs.lawStages law n fun _ _ => True).map (·.need)).sum ≤ L := by
    rw [Nat.zero_add]
    exact hL
  refine Properties.Spot.SecurityProofs.units_ok_of_firstOk law n _ (fun _ _ => True)
    (Properties.Spot.SecurityProofs.firstOk_zeros L _ 0 ?_ hL')
  cases law with
  | subset k =>
    obtain ⟨hkk, hk⟩ := hlaw
    intro s hs
    obtain rfl : s = ⟨2 * k * Flock.Draw.width n + 64, Flock.Draw.subset k n, fun _ => True⟩ :=
      List.mem_singleton.1 hs
    exact Audit.DrawExec.subset_zeros k n (2 * k * Flock.Draw.width n + 64) (by omega) (by rw [Nat.mul_assoc]; omega)
  | bernoulli a b => exact hlaw.elim
  | stratified K St =>
    obtain ⟨⟨-, hsz⟩, hK, hn1, hk, -⟩ := hlaw
    have hks : ∀ s ∈ St.toList, s.k ≤ s.size := fun s hs => by
      have hm : s.size ∈ ns := hsz ▸ List.mem_map_of_mem hs
      rw [hk s hs]
      exact stratumK_le hK (hn1 _ hm) (List.le_sum_of_mem hm)
    exact strata_zeros St hks
  | work K ws =>
    obtain ⟨-, hk, -⟩ := hlaw
    have hks : ∀ s ∈ (ws.map fun w : Flock.Draw.WorkStratum => w.toStratum).toList, s.k ≤ s.size := fun s hs => by
      rw [Array.toList_map] at hs
      obtain ⟨w, hw, rfl⟩ := List.mem_map.1 hs
      show w.k ≤ w.size
      rw [hk w hw]
      exact min_le_left _ _
    exact strata_zeros (ws.map fun w : Flock.Draw.WorkStratum => w.toStratum) hks

end Flock.Bridge

namespace Flock.Guarantees.JointAuditV5

open FlockSoundness

/-- **`hend` from A3 and the stream's length**: under A3 every stream of bytes is an outcome's, and the deployed
sample ends on a stream of zeros that holds its stages' first reads (`Bridge.units_ok_zeros`), so where the verifier's
stream holds `Bridge.firstRead law nJ` bytes, some outcome's sample ends, under any sampling design on strata. No
stream is assumed. The length is needed: on a shorter stream the sample runs out at every outcome. -/
theorem hend_of_zeros {nJ : ℕ} {Ω : Type} [Fintype Ω] {Lb : Fin 1 → ℕ}
    (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256) (hA3 : Assumptions.UniformRandomBytes Lb src)
    {law : Flock.Draw.Law} {ns : List ℕ} {kd : ℕ} (hlaw : Bridge.LawOn law ns kd) (hns : ns.sum = nJ)
    (hL : Bridge.firstRead law nJ ≤ Lb 0) :
    ∃ ω : Ω, sampleEnded src law nJ ω = true := by
  obtain ⟨units, read, h⟩ := Bridge.units_ok_zeros hlaw hns (Lb 0) hL
  exact hend_of_stream src hA3 law (fun _ _ => 0) h

end Flock.Guarantees.JointAuditV5
