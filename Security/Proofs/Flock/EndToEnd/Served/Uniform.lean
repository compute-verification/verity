import Proofs.Flock.EndToEnd.Served.Tie

/-!
# The served tapes are uniform when the challenger's coins are

Stratum `i`'s tape is its segment of the coins, then spare bytes (`Served.tapes`). The segment's length is decided by
its own bytes (`used_congr`: a stratum stops where its sampler first draws), so the tapes and the bytes they leave out
(`rest`: the coins past every segment, and the spare bytes each segment displaced) determine the coins and spare bytes
(`served_inj`). The map is a bijection of finite sets of the same size, so on uniform coins and spare bytes the tapes are
uniform (`tapes_uniform`), and A3 at the challenger's coins is A3 at the tapes (`served_uniform`).
-/

namespace FlockSoundness.Audit.Served

open Classical
open FlockSoundness.Game FlockSoundness.Audit.DrawExec
open scoped ENNReal

theorem draws_congr (s : Flock.Draw.Stratum) (a b : ByteArray) (hsz : a.size = b.size) (u : ℕ)
    (hpre : a.extract 0 u = b.extract 0 u) (j : ℕ) (hj : j * s.need ≤ u) : Draws s a j ↔ Draws s b j := by
  have he : a.extract 0 (j * s.need) = b.extract 0 (j * s.need) := by
    have := congrArg (fun x : ByteArray => x.extract 0 (j * s.need)) hpre
    simp only [ByteArray.extract_extract, Nat.zero_add, Nat.min_eq_left hj] at this
    exact this
  unfold Draws
  rw [he, hsz]

/-- **A stratum's read is decided by the bytes it reads**: two streams of one length that agree on the bytes a stratum
reads of the first have it read as many of the second. -/
theorem used_congr (s : Flock.Draw.Stratum) (a b : ByteArray) (hsz : a.size = b.size)
    (hpre : a.extract 0 (used s a) = b.extract 0 (used s a)) : used s b = used s a := by
  by_cases ha : ∃ j, Draws s a j
  · have hua : used s a = Nat.find ha * s.need := by unfold used; rw [dite_eq_left ha]
    have hd := draws_congr s a b hsz _ hpre
    have hb : Draws s b (Nat.find ha) := (hd _ (by rw [hua])).1 (Nat.find_spec ha)
    have hex : ∃ j, Draws s b j := ⟨_, hb⟩
    have hfind : Nat.find hex = Nat.find ha := by
      rw [Nat.find_eq_iff]
      refine ⟨hb, fun i hi hbi => Nat.find_min ha hi ((hd i ?_).2 hbi)⟩
      rw [hua]; exact Nat.mul_le_mul_right _ hi.le
    rw [hua]
    unfold used
    rw [dite_eq_left hex, hfind]
  · have hua : used s a = a.size := by unfold used; rw [dite_eq_right ha]
    have hb : ¬ ∃ j, Draws s b j := by
      rintro ⟨j, hj⟩
      exact ha ⟨j, (draws_congr s a b hsz _ hpre j (by rw [hua, hsz]; exact hj.2.1)).2 hj⟩
    rw [hua]
    unfold used
    rw [dite_eq_right hb, hsz]

theorem starts_mono (strata : Array Flock.Draw.Stratum) (src : ByteArray) {i i' : ℕ} (h : i ≤ i') :
    starts strata src i ≤ starts strata src i' := by
  induction h with
  | refl => exact le_rfl
  | step _ ih => rw [starts_succ]; omega

/-- At most one stratum's segment holds a position. -/
theorem seg_unique (strata : Array Flock.Draw.Stratum) (src : ByteArray) {i i' j : ℕ}
    (h : starts strata src i ≤ j ∧ j < starts strata src (i + 1))
    (h' : starts strata src i' ≤ j ∧ j < starts strata src (i' + 1)) : i = i' := by
  rcases Nat.lt_trichotomy i i' with hl | he | hl
  · have := starts_mono strata src (show i + 1 ≤ i' by omega); omega
  · exact he
  · have := starts_mono strata src (show i' + 1 ≤ i by omega); omega

variable {m L : ℕ} (St : Fin m → Flock.Draw.Stratum)

/-- **The bytes the tapes leave out**: the coins past every stratum's segment, and in stratum `i`'s segment the spare
bytes of `pad i` the segment displaced. -/
noncomputable def rest (c : Fin L → Fin 256) (pad : Pads m L) : Fin L → Fin 256 := fun j =>
  if h : ∃ i : Fin m, starts (Array.ofFn St) (Law.streamOf c) i ≤ j.val ∧
      j.val < starts (Array.ofFn St) (Law.streamOf c) (i.val + 1) then
    byteAt (pad (Classical.choose h)) (j.val - starts (Array.ofFn St) (Law.streamOf c) (Classical.choose h).val)
  else c j

theorem rest_in (c : Fin L → Fin 256) (pad : Pads m L) (j : Fin L) (i : Fin m)
    (hi : starts (Array.ofFn St) (Law.streamOf c) i ≤ j.val ∧ j.val < starts (Array.ofFn St) (Law.streamOf c) (i.val + 1)) :
    rest St c pad j = byteAt (pad i) (j.val - starts (Array.ofFn St) (Law.streamOf c) i) := by
  have hex : ∃ i : Fin m, starts (Array.ofFn St) (Law.streamOf c) i ≤ j.val ∧
      j.val < starts (Array.ofFn St) (Law.streamOf c) (i.val + 1) := ⟨i, hi⟩
  have e : Classical.choose hex = i := Fin.ext (seg_unique _ _ (Classical.choose_spec hex) hi)
  unfold rest
  rw [dite_eq_left hex, e]

theorem rest_out (c : Fin L → Fin 256) (pad : Pads m L) (j : Fin L)
    (h : ¬ ∃ i : Fin m, starts (Array.ofFn St) (Law.streamOf c) i ≤ j.val ∧
      j.val < starts (Array.ofFn St) (Law.streamOf c) (i.val + 1)) :
    rest St c pad j = c j := by
  unfold rest
  rw [dite_eq_right h]

theorem byteAt_lt {L : ℕ} (c : Fin L → Fin 256) (j : ℕ) (h : j < L) : byteAt c j = c ⟨j, h⟩ := by
  unfold byteAt; rw [dite_eq_left h]

/-- **A tape gives back its stratum's read**: stratum `i` reads as many bytes of its tape, up to the coins' end past
its start, as of the coins. -/
theorem tapes_seg (c : Fin L → Fin 256) (pad : Pads m L) (i : Fin m) :
    used (Array.ofFn St)[i.val]! ((Law.streamOf (tapes St c pad i)).extract 0
      (L - starts (Array.ofFn St) (Law.streamOf c) i)) = seg (Array.ofFn St) (Law.streamOf c) i := by
  have hs0 := starts_le (Array.ofFn St) (Law.streamOf c) (i.val + 1)
  rw [starts_succ] at hs0
  have hs : starts (Array.ofFn St) (Law.streamOf c) i + seg (Array.ofFn St) (Law.streamOf c) i ≤ L := by
    have h := hs0
    rw [size_streamOf] at h
    exact h
  have hsd : seg (Array.ofFn St) (Law.streamOf c) i = used (Array.ofFn St)[i.val]!
      ((Law.streamOf c).extract (starts (Array.ofFn St) (Law.streamOf c) i) (Law.streamOf c).size) := rfl
  rw [hsd]
  refine used_congr _ _ _ ?_ ?_
  · rw [size_extract_from, ByteArray.size_extract, size_streamOf, size_streamOf]
    omega
  · rw [extract_from _ _ _ (by rw [← hsd]; exact hs0), ByteArray.extract_extract, Nat.zero_add,
      Nat.zero_add, Nat.min_eq_left (by omega), ← hsd]
    exact (tapes_extract St c pad i).symm

/-- **The tapes and the bytes they leave out determine the coins and the spare bytes.** -/
theorem served_inj (c c' : Fin L → Fin 256) (pad pad' : Pads m L) (ht : tapes St c pad = tapes St c' pad')
    (hr : rest St c pad = rest St c' pad') : c = c' ∧ pad = pad' := by
  have hS : ∀ i, i ≤ m → starts (Array.ofFn St) (Law.streamOf c) i = starts (Array.ofFn St) (Law.streamOf c') i := by
    intro i
    induction i with
    | zero => intro _; rfl
    | succ i ih =>
      intro hi
      have e := ih (by omega)
      rw [starts_succ, starts_succ, e]
      congr 1
      rw [← tapes_seg St c pad ⟨i, by omega⟩, ← tapes_seg St c' pad' ⟨i, by omega⟩, ht]
      dsimp only
      rw [e]
  have hseg : ∀ i : Fin m, seg (Array.ofFn St) (Law.streamOf c) i = seg (Array.ofFn St) (Law.streamOf c') i := by
    intro i
    have h1 := hS (i.val + 1) i.isLt
    rw [starts_succ, starts_succ, hS i.val i.isLt.le] at h1
    omega
  have hle : ∀ (c : Fin L → Fin 256) (i : Fin m),
      starts (Array.ofFn St) (Law.streamOf c) i + seg (Array.ofFn St) (Law.streamOf c) i ≤ L := by
    intro c i
    have := starts_le (Array.ofFn St) (Law.streamOf c) (i.val + 1)
    rwa [starts_succ, size_streamOf] at this
  refine ⟨funext fun j => ?_, funext fun i => funext fun x => ?_⟩
  · by_cases hj : ∃ i : Fin m, starts (Array.ofFn St) (Law.streamOf c) i ≤ j.val ∧
        j.val < starts (Array.ofFn St) (Law.streamOf c) (i.val + 1)
    · obtain ⟨i, hi1, hi2⟩ := hj
      rw [starts_succ] at hi2
      have hx : j.val - starts (Array.ofFn St) (Law.streamOf c) i < L := by omega
      have e1 := congrFun (congrFun ht i) ⟨_, hx⟩
      unfold tapes at e1
      dsimp only at e1
      rw [ite_eq_left (by omega), ite_eq_left (by rw [← hseg i]; omega)] at e1
      rw [← hS i.val i.isLt.le, Nat.add_sub_cancel' hi1, byteAt_lt c _ j.isLt, byteAt_lt c' _ j.isLt] at e1
      exact e1
    · have hj' : ¬ ∃ i : Fin m, starts (Array.ofFn St) (Law.streamOf c') i ≤ j.val ∧
          j.val < starts (Array.ofFn St) (Law.streamOf c') (i.val + 1) := by
        rintro ⟨i, hi⟩
        rw [← hS i.val i.isLt.le, ← hS (i.val + 1) i.isLt] at hi
        exact hj ⟨i, hi⟩
      rw [← rest_out St c pad j hj, ← rest_out St c' pad' j hj', hr]
  · by_cases hx : x.val < seg (Array.ofFn St) (Law.streamOf c) i
    · have hj : starts (Array.ofFn St) (Law.streamOf c) i + x.val < L := by have := hle c i; omega
      have e1 := congrFun hr ⟨_, hj⟩
      rw [rest_in St c pad _ i ⟨Nat.le_add_right _ _, by
          show starts (Array.ofFn St) (Law.streamOf c) i + x.val < starts (Array.ofFn St) (Law.streamOf c) (i.val + 1)
          rw [starts_succ]; omega⟩,
        rest_in St c' pad' _ i ⟨by rw [← hS i.val i.isLt.le]; exact Nat.le_add_right _ _, by
          show starts (Array.ofFn St) (Law.streamOf c) i + x.val < starts (Array.ofFn St) (Law.streamOf c') (i.val + 1)
          rw [← hS (i.val + 1) i.isLt, starts_succ]; omega⟩] at e1
      dsimp only at e1
      rw [← hS i.val i.isLt.le, Nat.add_sub_cancel_left, byteAt_lt _ _ x.isLt, byteAt_lt _ _ x.isLt] at e1
      exact e1
    · have e1 := congrFun (congrFun ht i) x
      unfold tapes at e1
      rw [ite_eq_right hx, ite_eq_right (by rw [← hseg i]; exact hx)] at e1
      exact e1

theorem avg_fst {C D : Type} [Fintype C] [Fintype D] [Nonempty D] (f : C → ℝ≥0∞) :
    avg (fun q : C × D => f q.1) = avg f := by
  rw [avg_prod]
  exact congrArg avg (funext fun c => avg_const (f c))

/-- **The served tapes are uniform on uniform coins and spare bytes.** -/
theorem tapes_uniform (E : ((i : Fin m) → Fin L → Fin 256) → Prop) :
    prCoin (fun p : (Fin L → Fin 256) × Pads m L => E (tapes St p.1 p.2)) = prCoin E := by
  let e : (Fin L → Fin 256) × Pads m L → Pads m L × (Fin L → Fin 256) := fun p => (tapes St p.1 p.2, rest St p.1 p.2)
  have hinj : Function.Injective e := by
    rintro ⟨c, pad⟩ ⟨c', pad'⟩ h
    obtain ⟨h1, h2⟩ := Prod.mk.inj h
    obtain ⟨rfl, rfl⟩ := served_inj St c c' pad pad' h1 h2
    rfl
  have hbij : Function.Bijective e :=
    (Fintype.bijective_iff_injective_and_card e).2 ⟨hinj, by rw [Fintype.card_prod, Fintype.card_prod, mul_comm]⟩
  unfold prCoin
  have h := avg_equiv (Equiv.ofBijective e hbij)
    (fun q : Pads m L × (Fin L → Fin 256) => if E q.1 then (1 : ℝ≥0∞) else 0)
  refine h.trans ?_
  exact avg_fst (fun q => if E q then (1 : ℝ≥0∞) else 0)

theorem avg_sum_mul {Ω C : Type} [Fintype Ω] [Fintype C] (a : C → ℝ≥0∞) (g : C → Ω → ℝ≥0∞) :
    avg (fun ω => ∑ c, a c * g c ω) = ∑ c, a c * avg (g c) := by
  unfold avg
  simp only [Finset.mul_sum]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun c _ => Finset.sum_congr rfl fun ω _ => ?_
  ring

theorem avg_fiber {Ω C : Type} [Fintype Ω] [Fintype C] (g : Ω → C) (F : C → ℝ≥0∞) :
    avg (fun ω => F (g ω)) = ∑ c, F c * prCoin (fun ω => g ω = c) := by
  unfold prCoin
  rw [← avg_sum_mul]
  congr 1
  funext ω
  rw [Finset.sum_eq_single (g ω)]
  · rw [ite_eq_left rfl, mul_one]
  · intro b _ hb; rw [ite_eq_right (fun h => hb h.symm), mul_zero]
  · intro h; exact absurd (Finset.mem_univ _) h

/-- A source uniform for every event is uniform for every average. -/
theorem avg_uniform {Ω C : Type} [Fintype Ω] [Fintype C] (g : Ω → C)
    (hg : ∀ E : C → Prop, prCoin (fun ω => E (g ω)) = prCoin E) (F : C → ℝ≥0∞) :
    avg (fun ω => F (g ω)) = avg F := by
  rw [avg_fiber g F, show avg F = avg (fun c => F (id c)) from rfl, avg_fiber id F]
  exact Finset.sum_congr rfl fun c _ => congrArg (F c * ·) (hg (fun x => x = c))

/-- One stream's A3 is a statement about its one stream. -/
theorem uniform_one {Ωc : Type} [Fintype Ωc] (coins : Ωc → Fin L → Fin 256)
    (hc : Assumptions.UniformRandomBytes (fun _ : Fin 1 => L) (fun ω _ => coins ω)) (E : (Fin L → Fin 256) → Prop) :
    prCoin (fun ω => E (coins ω)) = prCoin E := by
  refine ((hc fun f => E (f 0)) : prCoin (fun ω => E (coins ω)) = _).trans ?_
  unfold prCoin
  exact avg_equiv (Equiv.funUnique (Fin 1) (Fin L → Fin 256)) (fun c => if E c then (1 : ℝ≥0∞) else 0)

/-- **A3 at the challenger's coins is A3 at the served tapes**, with the spare bytes uniform. -/
theorem served_uniform {Ωc : Type} [Fintype Ωc] (coins : Ωc → Fin L → Fin 256)
    (hc : Assumptions.UniformRandomBytes (fun _ : Fin 1 => L) (fun ω _ => coins ω)) :
    Assumptions.UniformRandomBytes (fun _ : Fin m => L)
      (fun ω : Ωc × Pads m L => tapes St (coins ω.1) ω.2) := by
  intro E
  have ht := tapes_uniform St E
  unfold prCoin at ht ⊢
  rw [avg_prod, avg_uniform coins (uniform_one coins hc)
    (fun c => avg fun pad : Pads m L => if E (tapes St c pad) then (1 : ℝ≥0∞) else 0), ← ht, avg_prod]

end FlockSoundness.Audit.Served
