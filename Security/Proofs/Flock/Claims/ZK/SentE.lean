import Proofs.Flock.Claims.ZK.Split

/-!
# The checks the deployed cut passes, from the refusals its coins show

The cut (`cutE`, `cutAt`) sends a guarded message where every check up to it in the session's order passes
(`famPasses`). At opened coins `c` and a stop `n` where no refusal is seen (`¬ Refused x c n`), every check whose coins
are answered by round `n` passes, at any coin plan reading the claim points, `β` and the pair coins at their rounds
(`PointReads`, `CoinReads`; `czExe` is one).

* `guardSlot x j r p`: the round by which table `j`'s rep `r`'s phase `p` check is seen: the rank check's (`rankSlot`),
  past both pair coins (`rhoSlot + 2`), past `β` (`betaSlot + 1`). `slots_le`: a rep's slots in order, inside the rep;
  `guardSlot_mono`: so the slots follow the session's order (`Session.pos`).
* `rankOn_of`: the rank check over a set of claims, from full rank at points carrying their weights.
* `famGuard_of`: the check at a seen slot passes (`RankBad`, `rhoIndependent`, `β.isZero` refused otherwise);
  `famPasses_of`: so does every check before it.
-/

namespace Claims.ZKMal

open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

namespace Masked

local notation "𝔽" => FlockLevel3.GF128

/-- **The round by which table `j`'s rep `r`'s phase `p` check is seen**: its rank check's round, the round after its
second pair coin's, the round after its `β`'s. -/
def guardSlot (x : Msg) (j r : ℕ) : Session.Phase → ℕ
  | .opening => rankSlot x j r
  | .restart => rhoSlot x j r + 2
  | .inner => betaSlot x j r + 1

/-- **A rep's slots in order, inside the rep**: its rank check after `14` rounds, its pair coins after the rank check's,
`β` after the pair coins', and the next rep after `β`'s. -/
theorem slots_le {x : Msg} (hx : Live x) {s spec} (hsx : FlockDeploy.setupOf true x = some (s, spec, stsOf x))
    {j r : ℕ} (hj : j < (stsOf x).size) (hr : r < 2) :
    repBase x (2 * j + r) + 14 ≤ rankSlot x j r ∧ rankSlot x j r + 3 ≤ rhoSlot x j r ∧
      rhoSlot x j r + 2 ≤ betaSlot x j r ∧ betaSlot x j r + 2 ≤ repBase x (2 * j + r + 1) := by
  obtain ⟨-, -, -, hall⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  obtain ⟨y⟩ := hx.2 j hj
  have hst : (stsOf x)[(2 * j + r) / 2]! = (stsOf x)[j] := by
    rw [show (2 * j + r) / 2 = j by omega]
    exact getElem!_pos (stsOf x) j hj
  have hz : _root_.Flock.Zk.schedule (stsOf x)[(2 * j + r) / 2]!.m = .ok (y.sch, y.tPad) := by
    rw [hst]
    exact y.hz
  have hm := (hall j hj).1
  have hk6 := y.wf.1.hk6
  have hrank : rankSlot x j r = repBase x (2 * j + r) + ((stsOf x)[j].m + (stsOf x)[j].c.kLog - 6) := by
    unfold rankSlot
    rw [getElem!_pos (stsOf x) j hj]
  have hrho : rhoSlot x j r =
      repBase x (2 * j + r) + FlockComplete.piopRoundsZk spec (stsOf x)[j] + (y.sch.k0 + 2) := by
    unfold rhoSlot
    rw [repSlot_eq hsx hz, repBase_eq hsx, hst]
  have hbeta : betaSlot x j r = repBase x (2 * j + r) + FlockComplete.piopRoundsZk spec (stsOf x)[j] +
      (y.sch.k0 + 7 + 11 * (y.sch.levels.size - 1) + 2) := by
    unfold betaSlot
    rw [repSlot_eq hsx hz, repBase_eq hsx, hst]
  have hnext : repBase x (2 * j + r + 1) =
      repBase x (2 * j + r) + FlockDeploy.repRounds true spec (stsOf x)[(2 * j + r) / 2]! := by
    rw [repBase_eq hsx, repBase_eq hsx, FlockComplete.ZkTables.repStart_succ]
  have hrr : FlockDeploy.repRounds true spec (stsOf x)[(2 * j + r) / 2]! =
      FlockComplete.piopRoundsZk spec (stsOf x)[j] + (y.sch.k0 + 7 + 11 * (y.sch.levels.size - 1)) + 4 := by
    rw [hst]
    unfold FlockDeploy.repRounds FlockComplete.piopRoundsZk
    simp only [↓reduceIte, y.hz]
    omega
  have hp : (stsOf x)[j].m + (stsOf x)[j].c.kLog - 6 + 3 ≤ FlockComplete.piopRoundsZk spec (stsOf x)[j] := by
    unfold FlockComplete.piopRoundsZk
    omega
  rw [hrank, hrho, hbeta, hnext, hrr]
  omega

theorem phase_idx_le (p : Session.Phase) : p.idx ≤ 2 := by
  cases p <;> decide

/-- **The slots follow the session's order**: a check at an earlier position (`Session.pos`) is seen by an earlier
round. -/
theorem guardSlot_mono {x : Msg} (hx : Live x) {s spec} (hsx : FlockDeploy.setupOf true x = some (s, spec, stsOf x))
    {j r j' r' : ℕ} (hj : j < (stsOf x).size) (hr : r < 2) (hj' : j' < (stsOf x).size) (hr' : r' < 2)
    {p p' : Session.Phase} (h : (2 * j' + r') * 3 + p'.idx ≤ (2 * j + r) * 3 + p.idx) :
    guardSlot x j' r' p' ≤ guardSlot x j r p := by
  obtain ⟨a0, a1, a2, a3⟩ := slots_le hx hsx hj hr
  have hpi := phase_idx_le p
  have hpi' := phase_idx_le p'
  rcases Nat.lt_or_ge (2 * j' + r') (2 * j + r) with hq | hq
  · obtain ⟨b0, b1, b2, b3⟩ := slots_le hx hsx hj' hr'
    have hmono : repBase x (2 * j' + r' + 1) ≤ repBase x (2 * j + r) := by
      rw [repBase_eq hsx, repBase_eq hsx]
      exact FlockComplete.ZkTables.repStart_mono _ _ hq
    cases p <;> cases p' <;> simp only [guardSlot] <;> omega
  · obtain rfl : j' = j := by omega
    obtain rfl : r' = r := by omega
    cases p <;> cases p' <;> simp only [guardSlot, Session.Phase.idx] at h ⊢ <;> omega

/-- **The rank check over the claims `P`** (`Table.RankOn`), from full rank (`MaskRankOk`) at points `pts` carrying
their weights, one point to each claim. -/
theorem rankOn_of {Wt Hi VL : Type} {z : ZkView.Sizes} (T : ZkView.Coined FlockLevel3.GF256 Wt Hi VL z)
    {P : Fin z.nC → Prop} {k : ℕ} (pts : Fin k → List _root_.Flock.F128) (words : Fin z.nMask → ℕ)
    (e : {i // P i} → Fin k) (he : Function.Injective e)
    (hw : ∀ (i : {i // P i}) x, T.eqw i.1 x = FlockSoundness.Refine.fOf (Composed.eqWord (pts (e i)) (words x)))
    (hM : Composed.MaskRankOk pts words) : T.tab.RankOn P := by
  classical
  intro g
  obtain ⟨u, hu⟩ := hM fun i => if h : ∃ a, e a = i then g h.choose else 0
  refine ⟨u, funext fun a => ?_⟩
  have h1 := congrFun hu (e a)
  have hex : ∃ a', e a' = e a := ⟨a, rfl⟩
  dsimp only at h1
  rw [dite_eq_left hex, he hex.choose_spec] at h1
  refine Eq.trans (Finset.sum_congr rfl fun x _ => ?_) h1
  show T.eqw a.1 x * _ = _
  rw [hw a x]

section Guards

variable {P : Composed.LigProver}
  {cz : (x : Msg) → (Fin (refK x) → FlockDeploy.Coin) → ∀ j, Composed.CoinsZK (tabC P x j).Rest}
  (hp : PointReads cz) (hc : CoinReads cz) {x : Msg} (hx : Live x) {c : Fin (refK x) → FlockDeploy.Coin} {n : ℕ}
  (hn : ¬ Refused x c n) {s spec} (hsx : FlockDeploy.setupOf true x = some (s, spec, stsOf x))

include hp hc hx hn hsx

/-- **The check at a seen slot passes**: where no refusal is seen by round `n`, table `j`'s rep `r`'s phase `p` check
passes once its slot is (`RankBad`, a dependent pair, `β = 0` would be refused). -/
theorem famGuard_of (j : Fin (jOf x)) (r : Fin 2) (p : Session.Phase) (h : guardSlot x j.val r.val p ≤ n) :
    famGuard (fun j => ((tabC P x j).T (cz x c j)).tab) (fun j => ((tabC P x j).T (cz x c j)).rep) j r p := by
  have hj := lt_of_jOf j
  obtain ⟨a0, a1, a2, a3⟩ := slots_le hx hsx hj r.isLt
  cases p with
  | opening =>
    simp only [guardSlot] at h
    have hst : (stsOf x)[j.val]! = stAt x j := getElem!_pos (stsOf x) j.val hj
    obtain ⟨-, -, -, hall⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
    have hm : 14 ≤ (stsOf x)[j.val]!.m := by
      rw [hst]
      have := (hall j.val hj).1
      exact le_trans (by norm_num) this
    have hk6 : 6 ≤ (stsOf x)[j.val]!.c.kLog := by
      rw [hst]
      exact (zkTab x j).wf.1.hk6
    have hr0 : rankSlot x j.val 0 ≤ n := by
      obtain ⟨-, b1, b2, b3⟩ := slots_le hx hsx hj (r := 0) (by norm_num)
      rcases Nat.lt_or_ge r.val 1 with hr | hr
      · rw [show r.val = 0 by omega] at h
        exact h
      · rw [show r.val = 1 by omega] at h a0
        have e0 : repBase x (2 * j.val + 0 + 1) = repBase x (2 * j.val + 1) := rfl
        omega
    have hok : FlockComplete.MaskRank.maskRankOk ((if r.val = 1 then ptsOpen c (stsOf x)[j.val]! (2 * j.val)
        else #[]) ++ ptsOpen c (stsOf x)[j.val]! (2 * j.val + r.val)) (Composed.maskWordsOf (stsOf x)[j.val]!) =
          true := by
      by_contra hb
      exact hn (.inr ⟨j.val, hj, r.val, r.isLt, .inl ⟨hm, hk6, hr0, h, Bool.eq_false_iff.2 hb⟩⟩)
    rw [hst] at hok
    have hout : ∀ r' : Fin 2, ((cz x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).rOuter r' =
        fun i => slotAt c (repBase x (2 * j.val + r'.val) + 1) i.val := by
      intro r'
      rw [(hp x c j r').2.2]
      funext i
      unfold outerAt slotAt
      rw [outerSlot_eq hsx]
    have hpts : ∀ k : Fin _root_.Flock.Zk.RANK_POINTS,
        Composed.pointsOf ((cz x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)) k =
          (ptsOpen c (stAt x j) (2 * j.val) ++ ptsOpen c (stAt x j) (2 * j.val + 1))[k.val]! :=
      fun k => pointsOf_reads _ (fun r n e => slotAt c (repBase x (2 * j.val + r) + n) e)
        (fun r => (hp x c j r).1) (fun r => (hp x c j r).2.1) hout k
    rcases Nat.lt_or_ge r.val 1 with hr | hr
    · have hr' : r = 0 := Fin.ext (by omega)
      subst hr'
      simp only [Fin.val_zero, Nat.add_zero, zero_ne_one, ↓reduceIte, Array.empty_append] at hok
      refine rankOn_of _ (k := 2) (fun i => Composed.pointsOf ((cz x c j).rest :
          Composed.RestOf (zkTab x j).table (stAt x j)) ⟨i.val, lt_of_lt_of_le i.isLt (by decide)⟩)
        (tabC P x j).maskWords (fun a => ⟨a.1.val, ?_⟩)
        (fun a b hab => Subtype.ext (Fin.ext (by have e := Fin.ext_iff.1 hab; exact e))) (fun _ _ => rfl)
        (maskRankOk_compose rfl (fun i => ?_) hok)
      · have h2 : (if a.1.val < 2 then (0 : Fin 2) else 1) ≤ 0 := a.2
        by_contra hk
        rw [ite_eq_right hk] at h2
        exact absurd h2 (by decide)
      · rw [hpts, getElem!_def, getElem!_def,
          Array.getElem?_append_left (lt_of_lt_of_eq i.isLt rfl)]
    · have hr' : r = 1 := Fin.ext (by omega)
      subst hr'
      simp only [Fin.val_one, ↓reduceIte] at hok
      exact rankOn_of _ ((tabC P x j).points (cz x c j)) (tabC P x j).maskWords (fun a => a.1)
        (fun a b hab => Subtype.ext hab) (fun _ _ => rfl) (maskRankOk_compose rfl hpts hok)
  | restart =>
    have hρ : _root_.Flock.Zk.rhoIndependent (rhoAt c j.val r.val) = true := by
      by_contra hb
      simp only [guardSlot] at h
      exact hn (.inr ⟨j.val, hj, r.val, r.isLt, .inr (.inl ⟨by omega, Bool.eq_false_iff.2 hb⟩)⟩)
    have he : ∀ a : _root_.Flock.F128, ((tabC P x j).T (cz x c j)).embed (FlockSoundness.Refine.fOf a) =
        FlockLevel3.GF256.of ⟨a, 0⟩ := fun _ => rfl
    have h0 : ((tabC P x j).T (cz x c j)).ρ r 0 = FlockLevel3.GF256.of (rhoAt c j.val r.val).1 := by
      show (if (0 : Fin 2) = 0 then FlockLevel3.GF256.of ((cz x c j).rho r).1
        else FlockLevel3.GF256.of ((cz x c j).rho r).2) = _
      rw [ite_eq_left (rfl : (0 : Fin 2) = 0), (hc x c j r).2]
    have h1 : ((tabC P x j).T (cz x c j)).ρ r 1 = FlockLevel3.GF256.of (rhoAt c j.val r.val).2 := by
      show (if (1 : Fin 2) = 0 then FlockLevel3.GF256.of ((cz x c j).rho r).1
        else FlockLevel3.GF256.of ((cz x c j).rho r).2) = _
      rw [ite_eq_right (by decide : ¬ (1 : Fin 2) = 0), (hc x c j r).2]
    exact Composed.wr_bijective ((tabC P x j).T (cz x c j)) r (rhoAt c j.val r.val) h0 h1 he hρ
  | inner =>
    have hb : (betaAt c j.val r.val).isZero = false := by
      by_contra hb
      simp only [guardSlot] at h
      exact hn (.inr ⟨j.val, hj, r.val, r.isLt, .inr (.inr ⟨by omega, Bool.eq_true_of_not_eq_false hb⟩)⟩)
    show FlockSoundness.Refine.fOf ((cz x c j).beta r) ≠ 0
    rw [(hc x c j r).1]
    exact Composed.fOf_ne_zero hb

/-- **Every check up to a seen slot passes** (`famPasses`): the earlier checks' slots are earlier (`guardSlot_mono`). -/
theorem famPasses_of (j : Fin (jOf x)) (r : Fin 2) (p : Session.Phase) (h : guardSlot x j.val r.val p ≤ n) :
    famPasses (fun j => ((tabC P x j).T (cz x c j)).tab) (fun j => ((tabC P x j).T (cz x c j)).rep) j r p :=
  fun j' r' p' hpos => famGuard_of hp hc hx hn hsx j' r' p'
    (le_trans (guardSlot_mono hx hsx (lt_of_jOf j) r.isLt (lt_of_jOf j') r'.isLt hpos) h)

end Guards

end Masked

end Claims.ZKMal
