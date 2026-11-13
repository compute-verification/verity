module

public import Proofs.Properties.Verity.Sampled.Window
public import Proofs.Flock.Claims.Sampling.Select.Take
public import Proofs.Core.Game.Prob

/-!
# A probe: a window's link points at a sample that runs out

A scratch file, never to merge. `Window.hcls` asks each member's part of the sample to have its session's link
points at every outcome of the source. Where the source runs out the deployed sample is every unit
(`Flock.Bridge.execSel`), so there `hcls` reads the member's whole size. Under A3 every stream is an outcome, a
stream of `0xFF` bytes among them, and on it `uniform n` rejects every word when `n` does not divide the number of
words. So no window sampled by `subset` over such a population has a member whose whole size and whose session's
size have different link points (`probe_hcls_subset`, `probe_not_hcls`).
-/

@[expose] public section

namespace Verity.Sampled

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open scoped ENNReal

/-! ## At a sample that runs out -/

/-- Where the sample's source runs out, the deployed sample is every unit. -/
theorem probe_execSel_error {n : ℕ} {Ω : Type} [Fintype Ω] [Nonempty Ω] {L : Fin 1 → ℕ}
    (t : Ω → (i : Fin 1) → Fin (L i) → Fin 256) (law : Flock.Draw.Law) (pop : ℕ) (ω : Ω) {e : String}
    (h : _root_.Spot.Draw.drawUnits law pop (_root_.Properties.Spot.bytesOf (t ω 0)) = .error e) :
    (Flock.Bridge.execSel (n := n) t law pop).draw ω = Finset.univ := by
  simp only [Flock.Bridge.execSel, h]

/-- A member's part of every unit is all its proof blocks. -/
theorem probe_part_univ {nJ : ℕ} (m : Member nJ) : (m.part Finset.univ).card = m.n := by
  have hu : m.part Finset.univ = Finset.univ := Finset.filter_true_of_mem fun u _ => Finset.mem_univ _
  rw [hu, Finset.card_univ, Fintype.card_fin]

/-- **At a sample whose source runs out, each member's whole size has its session's link points.** -/
theorem probe_hcls_ranOut (W : Window) (ω : W.Ω) {e : String}
    (h : _root_.Spot.Draw.drawUnits W.law W.nJ (_root_.Properties.Spot.bytesOf (W.src ω 0)) = .error e)
    (m : Member W.nJ) (hm : m ∈ W.ms) :
    mPtsOf m.D.unroll.g m.n = mPtsOf m.D.unroll.g (m.kd / nTab m.I) := by
  have h1 := W.hcls m hm ω
  rw [probe_execSel_error (n := W.nJ) W.src W.law W.nJ ω h, probe_part_univ] at h1
  exact h1

/-! ## The link points of 8 proof blocks and of 9 -/

theorem probe_log2ceil_le {x : ℕ} (h : x ≤ 8) : Flock.log2ceil x ≤ 3 := by
  unfold Flock.log2ceil
  split
  · omega
  · have : Nat.log2 (x - 1) < 3 := (Nat.log2_lt (by omega)).2 (by omega)
    omega

theorem probe_log2ceil_ge {y : ℕ} (h : 9 ≤ y) : 4 ≤ Flock.log2ceil y := by
  unfold Flock.log2ceil
  split
  · omega
  · have : 3 ≤ Nat.log2 (y - 1) := (Nat.le_log2 (by omega)).2 (by omega)
    omega

/-- The link points of at most 8 proof blocks are not those of 9 or more. -/
theorem probe_mPts_small {g a b : ℕ} (ha : (a + g - 1) / g ≤ 8) (h : mPtsOf g b = mPtsOf g a) :
    (b + g - 1) / g ≤ 8 := by
  unfold mPtsOf at h
  apply Nat.le_of_not_lt
  intro hb
  have h1 := probe_log2ceil_le (x := max ((a + g - 1) / g) 1) (Nat.max_le.2 ⟨ha, by omega⟩)
  have h2 := probe_log2ceil_ge (y := max ((b + g - 1) / g) 1) (Nat.le_trans hb (Nat.le_max_left _ _))
  have e1 : max 3 (Flock.log2ceil (max ((a + g - 1) / g) 1)) = 3 := Nat.max_eq_left h1
  have e2 : 4 ≤ max 3 (Flock.log2ceil (max ((b + g - 1) / g) 1)) := Nat.le_trans h2 (Nat.le_max_right _ _)
  rw [e1] at h
  omega

/-! ## A stream of `0xFF` bytes runs out -/

/-- Every byte of `b` is `0xFF`. -/
def AllFF (b : ByteArray) : Prop := ∀ (i : ℕ) (h : i < b.size), b[i] = 255

theorem probe_get!_of_lt (b : ByteArray) {i : ℕ} (h : i < b.size) : b.get! i = b[i] := by
  cases b with
  | mk bs =>
    show bs[i]! = bs[i]'h
    exact getElem!_pos bs i h

theorem allFF_extract {b : ByteArray} (hb : AllFF b) (p q : ℕ) : AllFF (b.extract p q) := by
  intro i h
  rw [ByteArray.getElem_extract]
  exact hb _ _

theorem allFF_bytesOf {L : ℕ} (f : Fin L → Fin 256) (hf : ∀ i, f i = 255) :
    AllFF (_root_.Properties.Spot.bytesOf f) := by
  intro i h
  have hi : i < L := by simpa [_root_.Properties.Spot.bytesOf, ByteArray.size] using h
  show (Array.ofFn fun i => UInt8.ofFin (f i))[i]'(by simpa using hi) = 255
  first
    | (simp [hf]; done)
    | (simp [hf]; rfl)
    | (rw [Array.getElem_ofFn, hf]; rfl)

theorem probe_beNat_succ (b : ByteArray) (off len : ℕ) :
    Flock.Draw.beNat b off (len + 1) = Flock.Draw.beNat b off len * 256 + (b.get! (off + len)).toNat := by
  unfold Flock.Draw.beNat
  rw [List.range_succ, List.foldl_append]
  rfl

/-- A word of `0xFF` bytes is the largest of its width. -/
theorem beNat_allFF {b : ByteArray} (hb : AllFF b) (off : ℕ) :
    ∀ len, off + len ≤ b.size → Flock.Draw.beNat b off len + 1 = 256 ^ len
  | 0, _ => rfl
  | len + 1, h => by
    have ih := beNat_allFF hb off len (by omega)
    have hg : (b.get! (off + len)).toNat = 255 := by
      rw [probe_get!_of_lt b (by omega), hb _ _]
      rfl
    rw [probe_beNat_succ, hg]
    first
      | rw [Nat.pow_succ]
      | rw [pow_succ]
    omega

/-- `uniform` returns nothing where it rejects every word. -/
theorem uniform_reject (n : ℕ) (s : ByteArray)
    (hrej : ∀ pos, pos + Flock.Draw.width n ≤ s.size →
      ¬ Flock.Draw.beNat s pos (Flock.Draw.width n) <
        2 ^ (8 * Flock.Draw.width n) - 2 ^ (8 * Flock.Draw.width n) % n) :
    ∀ fuel pos, Flock.Draw.uniform n s fuel pos = none
  | 0, _ => rfl
  | fuel + 1, pos => by
    unfold Flock.Draw.uniform
    dsimp only
    split
    · rfl
    · rename_i h
      split
      · rename_i h2
        exact absurd h2 (hrej pos (by omega))
      · exact uniform_reject n s hrej fuel _

/-- **On bytes of `0xFF`, `uniform n` returns nothing** where `n` does not divide the number of words. -/
theorem uniform_allFF {b : ByteArray} (hb : AllFF b) (n : ℕ) (hn : 2 ^ (8 * Flock.Draw.width n) % n ≠ 0)
    (fuel pos : ℕ) : Flock.Draw.uniform n b fuel pos = none := by
  refine uniform_reject n b (fun p hp => ?_) fuel pos
  have h1 := beNat_allFF hb p (Flock.Draw.width n) hp
  have h2 : (256 : ℕ) ^ Flock.Draw.width n = 2 ^ (8 * Flock.Draw.width n) := by
    first
      | (rw [Nat.pow_mul]; done)
      | (rw [Nat.pow_mul]; norm_num)
      | (rw [pow_mul]; norm_num)
  rw [h2] at h1
  omega

/-- On bytes of `0xFF`, a subset sample of one unit or more returns nothing: its first pick does. -/
theorem subset_allFF {b : ByteArray} (hb : AllFF b) (n k : ℕ) (hk : 1 ≤ k)
    (hn : 2 ^ (8 * Flock.Draw.width n) % n ≠ 0) : Flock.Draw.subset k n b = none := by
  obtain ⟨k', rfl⟩ : ∃ k', k = k' + 1 := ⟨k - 1, by omega⟩
  rw [FlockSoundness.Audit.DrawExec.subset_eq, List.range'_succ]
  simp only [FlockSoundness.Audit.DrawExec.loopSw, Nat.sub_zero, uniform_allFF hb n hn, Option.map_none]

/-- **On bytes of `0xFF`, the deployed subset sample runs out**: every read of the stream is of `0xFF` bytes, and the
sampler returns nothing on it. -/
theorem drawUnits_allFF {b : ByteArray} (hb : AllFF b) (n k : ℕ) (hk : 1 ≤ k)
    (hn : 2 ^ (8 * Flock.Draw.width n) % n ≠ 0) :
    ∃ e, _root_.Spot.Draw.drawUnits (.subset k) n b = .error e := by
  cases h : _root_.Spot.Draw.drawUnits (.subset k) n b with
  | error e => exact ⟨e, rfl⟩
  | ok r =>
    exfalso
    obtain ⟨units, read⟩ := r
    have hseq := _root_.Properties.Spot.SecurityProofs.drawUnits_seq (.subset k) n b units read h
    obtain ⟨j, D, hst, -⟩ := hseq
    obtain ⟨-, -, hrun, -⟩ := hst
    have hc := (subset_allFF (allFF_extract hb _ _) n k hk hn).symm.trans hrun
    cases hc

/-! ## Every stream is an outcome under A3 -/

theorem probe_prCoin_ne_zero {C : Type} [Fintype C] {p : C → Prop} {c : C} (h : p c) :
    Verity.Game.prCoin p ≠ 0 := by
  classical
  intro h0
  rw [Verity.Game.prCoin_eq_card] at h0
  have hpos : 0 < (Finset.univ.filter p).card :=
    Finset.card_pos.mpr ⟨c, Finset.mem_filter.mpr ⟨Finset.mem_univ c, h⟩⟩
  have hne : ((Finset.univ.filter p).card : ℝ≥0∞) ≠ 0 := by exact_mod_cast hpos.ne'
  exact (ENNReal.div_pos_iff.2 ⟨hne, ENNReal.natCast_ne_top _⟩).ne' h0

/-- **Under A3 every stream is the source's at some outcome.** -/
theorem probe_exists_stream {Ω : Type} [Fintype Ω] {m : ℕ} {L : Fin m → ℕ}
    (t : Ω → (i : Fin m) → Fin (L i) → Fin 256) (ht : FlockSoundness.Assumptions.UniformRandomBytes L t)
    (s : (i : Fin m) → Fin (L i) → Fin 256) : ∃ ω, t ω = s := by
  by_contra hno
  have hall : ∀ ω, ¬ t ω = s := fun ω hω => hno ⟨ω, hω⟩
  have h1 : Verity.Game.prCoin (fun ω => t ω = s) =
      Verity.Game.prCoin (fun x : (i : Fin m) → Fin (L i) → Fin 256 => x = s) := ht fun x => x = s
  have hF : (fun ω => t ω = s) = fun _ : Ω => False := funext fun ω => eq_false (hall ω)
  rw [hF, Verity.Game.prCoin_false] at h1
  exact probe_prCoin_ne_zero (p := fun x : (i : Fin m) → Fin (L i) → Fin 256 => x = s) (c := s) rfl h1.symm

/-! ## No such window -/

/-- **No window sampled by `subset` over a population that does not divide the number of words has a member whose
whole size and whose session's size have different link points.** -/
theorem probe_hcls_subset (W : Window) {k : ℕ} (hlaw : W.law = .subset k) (hk : 1 ≤ k)
    (hn : 2 ^ (8 * Flock.Draw.width W.nJ) % W.nJ ≠ 0) (m : Member W.nJ) (hm : m ∈ W.ms) :
    mPtsOf m.D.unroll.g m.n = mPtsOf m.D.unroll.g (m.kd / nTab m.I) := by
  obtain ⟨ω, hω⟩ := probe_exists_stream W.src W.hA3 (fun _ _ => 255)
  have hff : AllFF (_root_.Properties.Spot.bytesOf (W.src ω 0)) :=
    allFF_bytesOf _ fun i => congrFun (congrFun hω 0) i
  obtain ⟨e, he⟩ := drawUnits_allFF hff W.nJ k hk hn
  rw [← hlaw] at he
  exact probe_hcls_ranOut W ω he m hm

/-- **`hcls` fails at one member**: no window of 9 proof blocks, sampled one at a time, has a member that holds 9
blocks or more and whose session has the link points of 8 or fewer. -/
theorem probe_not_hcls (W : Window) (hlaw : W.law = .subset 1) (hn : W.nJ = 9) (m : Member W.nJ) (hm : m ∈ W.ms)
    (hs : (m.kd / nTab m.I + m.D.unroll.g - 1) / m.D.unroll.g ≤ 8)
    (hb : 8 < (m.n + m.D.unroll.g - 1) / m.D.unroll.g) : False := by
  have h9 : 2 ^ (8 * Flock.Draw.width W.nJ) % W.nJ ≠ 0 := by
    rw [hn]
    first
      | decide
      | norm_num [Flock.Draw.width, Nat.log2]
  have h := probe_mPts_small hs (probe_hcls_subset W hlaw (Nat.le_refl 1) h9 m hm)
  exact Nat.lt_irrefl _ (Nat.lt_of_lt_of_le hb h)

end Verity.Sampled
