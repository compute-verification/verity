import Proofs.Flock.Soundness.TmplUnrollAt
import Proofs.Flock.Soundness.ExecRows

/-!
# A committed net in row order is a built net (PRF-133)

`Net.ofRowsAt` checks the logical rows with `Net.ofRows` and places them at `pos`. Where the placed rows pass the same
order check (`Net.checkOrder`), every other check `Net.ofRows` runs passes on them too, so the committed net is the one
`Net.ofRows` builds from the placed rows (`ofRowsAt_ofRows`): a `Built` net (`ofRowsAt_built`, `TmplUnit.built`), the
form the guarantees read (`Placed.keyProg`).
-/

open Flock

namespace FlockSoundness.TmplBuilt

/-- The checks `Net.ofRows` runs that `Net.ofRows_ok` doesn't report: its port groups are aligned powers of two, it has
an output group, its constant is its last useful column, no port group's ports overrun its words, each input group lies
inside the input rows, and the input groups pass `inputsOk`. -/
theorem ofRows_head {name : String} {useful constPos : Nat} {ig og : Array Group} {ra rb : Array (Array Nat)}
    {n : Net} (h : Net.ofRows name useful constPos ig og ra rb = .ok n) :
    (∀ g ∈ ig ++ og, (isPow2 g.words && g.col % g.words == 0) = true) ∧ og.isEmpty = false ∧
      constPos + 1 = useful ∧
      (∀ g ∈ ig, g.bits.foldl (· + ·) 0 ≤ g.words * WORD ∧
        g.col + g.words ≤ (ig.back?.map fun g => g.col + g.words).getD 0) ∧
      (∀ g ∈ og, g.bits.foldl (· + ·) 0 ≤ g.words * WORD) ∧ Net.inputsOk ig ra rb = true := by
  unfold Net.ofRows at h
  simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h
  repeat' (first | split at h | (injection h with h; subst h))
  all_goals first
    | exact absurd h (by simp)
    | (rename_i c1 c2 c3 c4 _ _ _ _ _ _ _ c6 _ _ _ _ _ _ _ _
       rw [Bool.not_eq_true, Array.any_eq_false'] at c1
       rw [Bool.not_eq_true] at c2 c3 c6
       rw [Bool.not_eq_true, Bool.or_eq_false_iff, Array.any_eq_false', Array.any_eq_false'] at c4
       refine ⟨fun g hg => ?_, c2, ?_, fun g hg => ?_, fun g hg => ?_, ?_⟩
       · have := c1 g hg
         simpa using this
       · simpa using c3
       · have := c4.1 g hg
         simp only [Bool.or_eq_true, decide_eq_true_eq, not_or] at this
         exact ⟨by omega, by omega⟩
       · have := c4.2 g hg
         simp only [decide_eq_true_eq] at this
         omega
       · simpa using c6)

/-- `ofRowsAt_ofRows` with the committed size and inverse named. -/
theorem ofRowsAt_ofRows' {name : String} {useful constPos : Nat} {ig og : Array Group} {ra rb : Array (Array Nat)}
    {pos : Array Nat} {n' : Net} {P : Nat} {inv : Array (Option Nat)} (hP : pos.getD constPos 0 = P)
    (hinv : Net.invOf pos (P + 1) = inv) (h : Net.ofRowsAt name useful constPos ig og ra rb pos = .ok n')
    (hord : Net.checkOrder "flock-ir-unit/v2" (n'.inWords * WORD) n'.constPos (Net.placed pos inv ra)
      (Net.placed pos inv rb) = .ok ()) :
    Net.ofRows name (P + 1) P ig (og.map (Net.Group.at pos)) (Net.placed pos inv ra) (Net.placed pos inv rb) =
      .ok n' := by
  obtain ⟨n, hn, huse, hcp, hlog, ha, hb, hname, hig, hiw, hlk, hog, hpos, hout, hd⟩ := Net.ofRowsAt_ok h
  obtain ⟨-, -, -, -, -, hig0, -, hreach, hca, hcb, -, hinrows, hlk0, hiw0, -⟩ := Net.ofRows_ok hn
  obtain ⟨c1, c2, c3, c4, c5, c6⟩ := ofRows_head hn
  rw [hP] at huse hcp hlog ha hb hpos
  rw [hinv] at ha hb hpos
  rw [hlog] at ha hb
  have hcl : constPos < useful := by omega
  have hinp : ∀ i < n.inWords * WORD, pos.getD i 0 = i := fun i hi => Net.posOk_inputs hpos hi
  have hPge : n.inWords * WORD ≤ P := by
    refine Nat.not_lt.mp fun hlt => ?_
    have h1 := hinp P hlt
    have h2 : constPos = P := Net.posOk_inj hpos hcl (by omega) (by rw [h1, hP])
    omega
  have hmap : ∀ i < n.inWords * WORD, ∀ r : Array Nat, ((r == #[i]) || r.isEmpty) = true →
      r.map (pos.getD · 0) = r := by
    intro i hi r hr
    simp only [Bool.or_eq_true, beq_iff_eq, Array.isEmpty_iff] at hr
    have hh := hinp i hi
    rw [Array.getD_eq_getD_getElem?] at hh
    rcases hr with rfl | rfl
    · simp [hh]
    · simp
  have hrows : ∀ i < n.inWords * WORD, (Net.placed pos inv ra).getD i #[] = ra.getD i #[] ∧
      (Net.placed pos inv rb).getD i #[] = rb.getD i #[] := by
    intro i hi
    have hb' := hinrows i hi
    have hr : ((ra.getD i #[] == #[i]) || (ra.getD i #[]).isEmpty) = true ∧
        ((rb.getD i #[] == #[i]) || (rb.getD i #[]).isEmpty) = true := by
      revert hb'
      generalize (ra.getD i #[] == #[i]) = x1
      generalize (rb.getD i #[] == #[i]) = x2
      generalize (ra.getD i #[]).isEmpty = x3
      generalize (rb.getD i #[]).isEmpty = x4
      cases x1 <;> cases x2 <;> cases x3 <;> cases x4 <;> simp
    have pa := Net.placed_at hpos ra (x := i) (by omega)
    have pb := Net.placed_at hpos rb (x := i) (by omega)
    rw [hinp i hi] at pa pb
    rw [pa, pb, hmap i hi _ hr.1, hmap i hi _ hr.2]
    exact ⟨rfl, rfl⟩
  have hconst : (Net.placed pos inv ra).getD P #[] = #[P] ∧ (Net.placed pos inv rb).getD P #[] = #[P] := by
    have pa := Net.placed_at hpos ra hcl
    have pb := Net.placed_at hpos rb hcl
    rw [hP, hca] at pa
    rw [hP, hcb] at pb
    rw [pa, pb]
    have hP' := hP
    rw [Array.getD_eq_getD_getElem?] at hP'
    simp [hP']
  have e1 : ((ig ++ og.map (Net.Group.at pos)).any fun g => !(isPow2 g.words && g.col % g.words == 0)) = false := by
    rw [Array.any_eq_false']
    intro g hg
    rcases Array.mem_append.mp hg with hg | hg
    · simp [c1 g (Array.mem_append.mpr (Or.inl hg))]
    · obtain ⟨g0, hg0, rfl⟩ := Array.mem_map.mp hg
      have h1 := c1 g0 (Array.mem_append.mpr (Or.inr hg0))
      simp only [Bool.and_eq_true, beq_iff_eq] at h1
      have hout' := hout
      unfold Net.outsOk at hout'
      have ho := (Array.all_eq_true'.mp hout') g0 hg0
      simp only [Bool.and_eq_true, beq_iff_eq] at ho
      have hdv : g0.words ∣ pos.getD (g0.col * WORD) 0 / WORD := by
        have := Nat.dvd_of_mod_eq_zero ho.1
        rw [Nat.mul_comm g0.words WORD] at this
        exact Nat.dvd_div_of_mul_dvd this
      have hm := Nat.mod_eq_zero_of_dvd hdv
      rw [Array.getD_eq_getD_getElem?] at hm
      simp [Net.Group.at, h1.1, hm]
  have e2 : (og.map (Net.Group.at pos)).isEmpty = false := by
    have := c2
    simp only [Array.isEmpty, Array.size_map] at this ⊢
    exact this
  have e4 : ((ig.any fun g => g.bits.foldl (· + ·) 0 > g.words * WORD ||
        g.col + g.words > (ig.back?.map fun g => g.col + g.words).getD 0) ||
      (og.map (Net.Group.at pos)).any fun g => g.bits.foldl (· + ·) 0 > g.words * WORD) = false := by
    rw [Bool.or_eq_false_iff, Array.any_eq_false', Array.any_eq_false']
    refine ⟨fun g hg => ?_, fun g hg => ?_⟩
    · have := c4 g hg
      simp only [Bool.or_eq_true, decide_eq_true_eq, not_or]
      exact ⟨by omega, by omega⟩
    · obtain ⟨g0, hg0, rfl⟩ := Array.mem_map.mp hg
      have := c5 g0 hg0
      simpa [Net.Group.at] using this
  have e5 : ¬((ig.back?.map fun g => g.col + g.words).getD 0 * WORD > P) := by
    rw [← hiw0]
    omega
  have e6 : (List.range ((ig.back?.map fun g => g.col + g.words).getD 0 * WORD)).find? (fun i =>
      !(((Net.placed pos inv ra).getD i #[] == #[i] && (Net.placed pos inv rb).getD i #[] == #[i]) ||
        ((Net.placed pos inv ra).getD i #[]).isEmpty && ((Net.placed pos inv rb).getD i #[]).isEmpty)) = none := by
    rw [List.find?_eq_none]
    intro i hi
    rw [List.mem_range, ← hiw0] at hi
    rw [(hrows i hi).1, (hrows i hi).2, hinrows i hi]
    decide
  have e8 : Net.checkOrder "flock-ir-unit/v2" ((ig.back?.map fun g => g.col + g.words).getD 0 * WORD) P
      (Net.placed pos inv ra) (Net.placed pos inv rb) = .ok () := by
    rw [hiw, hiw0, hcp] at hord
    exact hord
  have e9 : Net.inputsOk ig (Net.placed pos inv ra) (Net.placed pos inv rb) = true := by
    have h6 := c6
    unfold Net.inputsOk at h6 ⊢
    simp only [Bool.and_eq_true, Array.all_eq_true', List.all_eq_true, List.mem_range, beq_iff_eq] at h6 ⊢
    refine ⟨h6.1, fun g hg j hj => ?_⟩
    have hw : (g.col + g.words) * WORD ≤ n.inWords * WORD :=
      Nat.mul_le_mul_right _ (by rw [hiw0]; exact (c4 g hg).2)
    rw [Nat.add_mul] at hw
    have hlt : g.col * WORD + j < n.inWords * WORD := by have := (c4 g hg).1; omega
    rw [(hrows _ hlt).1, (hrows _ hlt).2]
    exact h6.2 g hg j hj
  have e12 : (og.map (Net.Group.at pos)).find? (fun g => decide (g.col + g.words > 2 ^ max 7 (log2ceil (P + 1)) / WORD))
      = none := by
    rw [Array.find?_eq_none]
    intro g hg
    have := hd g (by rw [hog]; exact Array.mem_toList_iff.mpr hg)
    rw [hlog] at this
    simp only [decide_eq_true_eq]
    omega
  unfold Net.ofRows
  simp only [bind, Except.bind, pure, Except.pure, e1, e2, e4, e5, e6, hconst.1, hconst.2, e8, e9, ha, hb, e12,
    bne_self_eq_false, Bool.or_self, Bool.false_eq_true, Bool.not_true, ↓reduceIte]
  congr 1
  apply TmplFold.net_ext
  · exact (hname.trans (TmplFold.ofRows_name hn)).symm
  · exact huse.symm
  · exact hcp.symm
  · exact (hig.trans hig0).symm
  · exact hog.symm
  · exact (hiw.trans hiw0).symm
  · rfl
  · rfl
  · exact hlog.symm
  · exact (hlk.trans hlk0).symm

/-- **A committed net whose placed rows are in row order is `Net.ofRows`' net on them**: its size one past the
constant's position, its output groups at their committed words. -/
theorem ofRowsAt_ofRows {name : String} {useful constPos : Nat} {ig og : Array Group} {ra rb : Array (Array Nat)}
    {pos : Array Nat} {n' : Net} (h : Net.ofRowsAt name useful constPos ig og ra rb pos = .ok n')
    (hord : Net.checkOrder "flock-ir-unit/v2" (n'.inWords * WORD) n'.constPos
      (Net.placed pos (Net.invOf pos (pos.getD constPos 0 + 1)) ra)
      (Net.placed pos (Net.invOf pos (pos.getD constPos 0 + 1)) rb) = .ok ()) :
    Net.ofRows name (pos.getD constPos 0 + 1) (pos.getD constPos 0) ig (og.map (Net.Group.at pos))
      (Net.placed pos (Net.invOf pos (pos.getD constPos 0 + 1)) ra)
      (Net.placed pos (Net.invOf pos (pos.getD constPos 0 + 1)) rb) = .ok n' :=
  ofRowsAt_ofRows' rfl rfl h hord

/-- **A committed net in row order is built.** -/
theorem ofRowsAt_built {name : String} {useful constPos : Nat} {ig og : Array Group} {ra rb : Array (Array Nat)}
    {pos : Array Nat} {n' : Net} (h : Net.ofRowsAt name useful constPos ig og ra rb pos = .ok n')
    (hord : Net.checkOrder "flock-ir-unit/v2" (n'.inWords * WORD) n'.constPos
      (Net.placed pos (Net.invOf pos (pos.getD constPos 0 + 1)) ra)
      (Net.placed pos (Net.invOf pos (pos.getD constPos 0 + 1)) rb) = .ok ()) : Built n' :=
  ⟨_, _, _, _, _, _, _, ofRowsAt_ofRows h hord⟩

end FlockSoundness.TmplBuilt

namespace Flock

/-- **A read templated unit whose committed rows are in row order is built**: the order check over its logical rows at
`Tmpl.posMap`'s positions is all that stands between the committed net and `FlockSoundness.Built`. -/
theorem TmplUnit.built (tu : TmplUnit)
    (hord : ∀ acc, tu.t.rows = .ok acc →
      Net.checkOrder "flock-ir-unit/v2" (tu.net.inWords * WORD) tu.net.constPos
        (Net.placed (Tmpl.posMap tu.t tu.lays)
          (Net.invOf (Tmpl.posMap tu.t tu.lays) ((Tmpl.posMap tu.t tu.lays).getD tu.t.constPos 0 + 1)) acc.ra)
        (Net.placed (Tmpl.posMap tu.t tu.lays)
          (Net.invOf (Tmpl.posMap tu.t tu.lays) ((Tmpl.posMap tu.t tu.lays).getD tu.t.constPos 0 + 1)) acc.rb) =
        .ok ()) :
    FlockSoundness.Built tu.net := by
  have hu := tu.unrolled
  unfold Tmpl.unrollAt at hu
  obtain ⟨acc, hacc, hat⟩ := FlockSoundness.TmplFold.ebo hu
  exact FlockSoundness.TmplBuilt.ofRowsAt_built hat (hord acc hacc)

end Flock
