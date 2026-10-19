import Proofs.Flock.Claims.KS.SlotDist
import Proofs.Flock.Soundness.CompiledLock

/-!
# A rep's level caps, at a distance fixed by the statement and the schedule

Rep `r`'s lockstep (`lockRep`) meets its level-`ℓ + 1` cap's table node after the same number of coins on every path that
meets it, a number read off the statement's dimensions, the number of claims and the schedule's lanes: the zerocheck's,
the lincheck's and the opening's coins, Ligerito's two first coins and its level-0 folds, then per level passed its two
OOD samples, its queries and the next level's folds. A path that rejects early meets no cap after that.

* `coinsK_zcShape`, `coinsK_lcShape`, `coinsK_opShape`, `coinsK_foldShape`, `coinsK_oodShape`, `coinsK_qbShape`: the
  phases' coins;
* `noSlot_lockFold`, `noSlot_lockTwoOod`, `noSlot_lockQuery`: the phases between caps have no table node;
* `recSlotDist`, `dist_lockRec`; `ligSlotDist`, `dist_lockLig`; `repSlotDist`, `dist_lockRep`: the distance of each cap,
  through the recursive rounds, Ligerito, and a whole rep;
* `noSlot_lockRec_ge`, `noSlot_lockLig_ge`, `noSlot_lockRep_ge`: past the last level there is no cap;
* `recSlotDist_eq`, `fast100_lanes`: with four lanes after level 0, as `fast100` has, the caps are `11` coins apart;
* `getRep_slotOf_ne`: the other rep never holds this rep's slots.
-/

namespace FlockSoundness

open Game Model Accounting Discharge.Exec

section Shapes

theorem coinsK_pairRounds (F : Type) : ∀ n, Lock.coinsK (pairRounds F n) = n
  | 0 => rfl
  | n + 1 => by simp only [pairRounds, Lock.coinsK, coinsK_pairRounds F n]

theorem coinsK_switches (F : Type) : ∀ n, Lock.coinsK (switches F n) = n
  | 0 => rfl
  | n + 1 => by simp only [switches, Lock.coinsK, coinsK_switches F n]

theorem coinsK_foldShape (F K : Type) : ∀ k, Lock.coinsK (foldShape F K k) = k
  | 0 => rfl
  | k + 1 => by simp only [foldShape, Lock.coinsK, coinsK_foldShape F K k]

theorem coinsK_zcShape (F : Type) (m : ℕ) : Lock.coinsK (zcShape F m) = m - 6 + 3 := by
  simp only [zcShape, Lock.coinsK, Lock.coinsK_append, coinsK_pairRounds]

theorem coinsK_lcShape (F : Type) (kLog : ℕ) : Lock.coinsK (lcShape F kLog) = kLog - 6 + 3 := by
  simp only [lcShape, Lock.coinsK, Lock.coinsK_append, coinsK_pairRounds]

theorem coinsK_opShape (F : Type) (n : ℕ) : Lock.coinsK (opShape F n) = n + 1 := by
  simp only [opShape, Lock.coinsK_append, coinsK_switches, Lock.coinsK]

theorem coinsK_oodShape (F K : Type) (dim : ℕ) : Lock.coinsK (oodShape F K dim) = 4 := rfl

theorem coinsK_qbShape (F K : Type) (lv : Level) (wm : Bool) : Lock.coinsK (qbShape F K lv wm) = 3 := by
  cases wm <;> rfl

end Shapes

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
variable {D : Type}

omit [Field F] [Fintype F] [DecidableEq F] in
theorem getRep_slotOf_ne {r r' : ℕ} (hr : r < 2) (hne : r' ≠ r) (i : IRep F D) (ℓ : ℕ) :
    getRep r' i (slotOf r ℓ) = none := by
  unfold getRep slotOf
  exact ite_eq_right (by omega)

omit [DecidableEq F] [Fintype K] [DecidableEq K] in
theorem noSlot_lockFold (A : Arith F K) (r slot : ℕ) (i : IRep F D) (P : Opens F D → Prop) :
    ∀ (k : ℕ) (s : K) (q : Quad K) (sF : Opens F D → K) (qF : Opens F D → Quad K)
      (h : ∀ o, P o → sF o = s ∧ qF o = q), Lock.NoSlot slot (lockFold A r i P k s q sF qF h)
  | 0, _, _, _, _, _ => .ret _
  | k + 1, s, q, sF, qF, h => by
      refine .coin _ fun w => .send _ fun msg => Lock.NoSlot.bind ?_ (noSlot_lockFold A r slot i P k _ _ _ _ _)
      intro _ x y hxy
      rcases x with ⟨rs, x₂⟩
      rcases x₂ with ⟨s'', q'⟩
      rcases y with ⟨rs', y₂⟩
      rcases y₂ with ⟨sF'', qF'⟩
      obtain ⟨_, _, _, _⟩ := hxy
      exact .ret _

omit [DecidableEq F] [Fintype K] [DecidableEq K] in
theorem noSlot_lockTwoOod (A : Arith F K) (r slot : ℕ) (i : IRep F D) (P : Opens F D → Prop)
    (dim split split' : ℕ) (hsplit : split = split') (s : K) (q : Quad K) (sF : Opens F D → K)
    (qF : Opens F D → Quad K) (h : ∀ o, P o → sF o = s ∧ qF o = q) :
    Lock.NoSlot slot (lockTwoOod A r i P dim split split' hsplit s q sF qF h) := by
  subst hsplit
  exact .coin _ fun _ => .send _ fun _ => .send _ fun _ => .coin _ fun _ =>
    .coin _ fun _ => .send _ fun _ => .send _ fun _ => .coin _ fun _ => .ret _

theorem noSlot_lockQuery (A : Arith F K) (r slot : ℕ) (C₀ : Oracle F) (i : IRep F D) (lv : Level)
    (ch : List K) (start start' : ℕ) (hstart : start = start') (st : LigState F K) (stC : LigStateC F K D)
    (h : StRel r C₀ i st stC) (wm : Bool) :
    Lock.NoSlot slot (lockQuery A r C₀ i lv ch start start' hstart st stC h wm) := by
  subst hstart
  cases wm
  · exact .coin _ fun _ => .coin _ fun _ => .coin _ fun _ => .ret _
  · exact .coin _ fun _ => .coin _ fun _ => .send _ fun _ => .coin _ fun _ => .ret _

/-- The coins from the recursion's start on level `L` to its level-`L + 1 + p` cap: per level passed, its folds, its two
OOD samples and its queries, then the folds of the cap's own level. `0` past the last cap. -/
def recSlotDist : List Level → ℕ → ℕ
  | [], _ => 0
  | [_], _ => 0
  | lv :: _ :: _, 0 => lv.lanes
  | lv :: lv' :: rest, p + 1 => lv.lanes + 7 + recSlotDist (lv' :: rest) p

/-- **The last level has no cap.** -/
theorem noSlot_lockRec_single (A : Arith F K) (r : ℕ) (hr : r < 2) (n : ℕ) (b : ℕ → F) (initial : List K)
    (ext : ℕ) (C₀ : Oracle F) (sch : Schedule) (slot : ℕ) (lv : Level) (i : IRep F D) (st : LigState F K)
    (stC : LigStateC F K D) (h : RecInv r C₀ sch [lv] i st stC) :
    Lock.NoSlot slot (lockRec A r hr n b initial ext C₀ sch [lv] i st stC h) := by
  refine Lock.NoSlot.bind ?_ (noSlot_lockFold A r _ i _ lv.lanes _ _ _ _ _)
  intro i' x y hxy
  rcases x with ⟨ch, x₂⟩
  rcases x₂ with ⟨s, q⟩
  rcases y with ⟨ch', y₂⟩
  rcases y₂ with ⟨sF, qF⟩
  obtain ⟨hi, hch, hlen, hsq⟩ := hxy
  simp only at hch hlen hsq
  subst i' hch
  refine .send _ fun yr => Lock.NoSlot.bind ?_ (noSlot_lockQuery A r _ _ _ _ _ _ _ _ _ _ _ false)
  intro i3 st3 stC3 hq
  obtain ⟨hi3, hq'⟩ := hq
  subst i3
  exact .ret _

/-- **The recursion's caps' distances.** -/
theorem dist_lockRec (A : Arith F K) (r : ℕ) (hr : r < 2) (n : ℕ) (b : ℕ → F) (initial : List K) (ext : ℕ)
    (C₀ : Oracle F) (sch : Schedule) :
    ∀ (levs : List Level) (i : IRep F D) (st : LigState F K) (stC : LigStateC F K D)
      (h : RecInv r C₀ sch levs i st stC) (p : ℕ),
      Lock.Dist (slotOf r (stC.level + 1 + p)) (recSlotDist levs p)
        (lockRec A r hr n b initial ext C₀ sch levs i st stC h)
  | [], _, _, _, _, _ => .noSlot (.ret _)
  | [lv], i, st, stC, h, _ => .noSlot (noSlot_lockRec_single A r hr n b initial ext C₀ sch _ lv i st stC h)
  | lv :: lv' :: rest, i, st, stC, h, 0 => by
      refine Lock.Dist.slot_eq (slot := slotOf r (stC.level + 1)) rfl ?_
      refine Lock.Dist.bind_eq (j := 0) ?_ (noSlot_lockFold A r _ i _ lv.lanes _ _ _ _ _)
        (foldRoundsC_exact A _ _ _) (by simp only [recSlotDist, coinsK_foldShape]; omega)
      intro i' x y hxy
      rcases x with ⟨ch, x₂⟩
      rcases x₂ with ⟨s, q⟩
      rcases y with ⟨ch', y₂⟩
      rcases y₂ with ⟨sF, qF⟩
      obtain ⟨hi, hch, hlen, hsq⟩ := hxy
      simp only at hch hlen hsq
      subst i' hch
      exact .here
  | lv :: lv' :: rest, i, st, stC, h, p + 1 => by
      refine Lock.Dist.bind_eq (j := recSlotDist (lv' :: rest) p + 3 + 4) ?_
        (noSlot_lockFold A r _ i _ lv.lanes _ _ _ _ _) (foldRoundsC_exact A _ _ _)
        (by simp only [recSlotDist, coinsK_foldShape]; omega)
      intro i' x y hxy
      rcases x with ⟨ch, x₂⟩
      rcases x₂ with ⟨s, q⟩
      rcases y with ⟨ch', y₂⟩
      rcases y₂ with ⟨sF, qF⟩
      obtain ⟨hi, hch, hlen, hsq⟩ := hxy
      simp only at hch hlen hsq
      subst i' hch
      refine .other _ (by unfold slotOf; omega) fun C' cap => ?_
      refine Lock.Dist.bind_eq (j := recSlotDist (lv' :: rest) p + 3) ?_
        (noSlot_lockTwoOod A r _ _ _ _ _ _ _ _ _ _ _ _) (twoOodC_exact A _ _ _ _)
        (by rw [coinsK_oodShape])
      intro i₂ x y hxy
      rcases x with ⟨s2, x₂⟩
      rcases x₂ with ⟨q2, oods2⟩
      rcases y with ⟨sF2, y₂⟩
      rcases y₂ with ⟨qF2, oodsC2⟩
      obtain ⟨hi2, hoods, hsq2⟩ := hxy
      simp only at hoods hsq2
      subst i₂ hoods
      refine Lock.Dist.bind_eq (j := recSlotDist (lv' :: rest) p) ?_
        (noSlot_lockQuery A r _ _ _ _ _ _ _ _ _ _ _ true) (queryAndBatchC_exact A r lv _ _ _ true)
        (by rw [coinsK_qbShape])
      intro i₃ st3 stC3 hq
      obtain ⟨hi3, hrel3, hlev, hq'⟩ := hq
      subst i₃
      have hlev' : stC3.level = stC.level := hlev
      refine Lock.Dist.slot_eq ?_ (dist_lockRec A r hr n b initial ext C₀ sch (lv' :: rest) _ _ _ _ p)
      show slotOf r (stC3.level + 1 + 1 + p) = slotOf r (stC.level + 1 + (p + 1))
      unfold slotOf
      omega

/-- The coins from the end of Ligerito's level-0 folds to its level-`ℓ + 1` cap. -/
def ligSlotDist (levs : List Level) : ℕ → ℕ
  | 0 => 0
  | p + 1 => 7 + recSlotDist levs p

/-- **Ligerito's caps' distances.** -/
theorem dist_lockLig (A : Arith F K) (r : ℕ) (hr : r < 2) (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level)
    (ext : ℕ) (hk0 : l₀.lanes = k0) (T : F) (b : ℕ → F) (C₀ : Oracle F) (ℓ : ℕ) :
    Lock.Dist (slotOf r (ℓ + 1)) (ligSlotDist (l₁ :: rest) ℓ + k0 + 1 + 1)
      (lockLig (D := D) A r hr m k0 l₀ l₁ rest ext hk0 T b C₀) := by
  refine .coin _ fun z₀ => .send _ fun y₀ => .coin _ fun β₀ => .send _ fun msg => ?_
  refine Lock.Dist.bind_eq (j := ligSlotDist (l₁ :: rest) ℓ) ?_ (noSlot_lockFold A r _ _ _ k0 _ _ _ _ _)
    (foldRoundsC_exact A _ _ _) (by rw [coinsK_foldShape])
  intro i' x y hxy
  rcases x with ⟨initial, x₂⟩
  rcases x₂ with ⟨s, q⟩
  rcases y with ⟨initial', y₂⟩
  rcases y₂ with ⟨sF, qF⟩
  obtain ⟨hi, hch, hlen, hsq⟩ := hxy
  simp only at hch hlen hsq
  subst i' hch
  cases ℓ with
  | zero => exact .here
  | succ p =>
    refine .other _ (by unfold slotOf; omega) fun C₁ cap₁ => ?_
    refine Lock.Dist.bind_eq (j := recSlotDist (l₁ :: rest) p + 3) ?_
      (noSlot_lockTwoOod A r _ _ _ _ _ _ _ _ _ _ _ _) (twoOodC_exact A _ _ _ _)
      (by simp only [coinsK_oodShape, ligSlotDist]; omega)
    intro i₂ x y hxy
    rcases x with ⟨s2, x₂⟩
    rcases x₂ with ⟨q2, oods⟩
    rcases y with ⟨sF2, y₂⟩
    rcases y₂ with ⟨qF2, oodsC⟩
    obtain ⟨hi2, hoods, hsq2⟩ := hxy
    simp only at hoods hsq2
    subst i₂ hoods
    refine Lock.Dist.bind_eq (j := recSlotDist (l₁ :: rest) p) ?_
      (noSlot_lockQuery A r _ _ _ _ _ _ _ _ _ _ _ true) (queryAndBatchC_exact A r l₀ _ _ _ true)
      (by rw [coinsK_qbShape])
    intro i₃ st3 stC3 hq
    obtain ⟨hi3, hq'⟩ := hq
    subst i₃
    refine Lock.Dist.slot_eq ?_ (dist_lockRec A r hr _ b _ ext C₀ _ (l₁ :: rest) _ _ _ _ p)
    show slotOf r (1 + 1 + p) = slotOf r (p + 1 + 1)
    unfold slotOf
    omega

/-- The coins from a rep's start to its level-`ℓ + 1` cap: the zerocheck's, the lincheck's, the opening's of its
`nExtra + 2` claims, Ligerito's two first coins and level-0 folds, then `ligSlotDist`. -/
def repSlotDist (m kLog nExtra k0 : ℕ) (levs : List Level) (ℓ : ℕ) : ℕ :=
  ligSlotDist levs ℓ + k0 + 1 + 1 + (nExtra + 1 + 1 + 1) + (kLog - 6 + 3) + (m - 6 + 3)

/-- **A rep's caps' distances**: every path of rep `r` that meets its level-`ℓ + 1` cap's table node meets it after
`repSlotDist` coins. -/
theorem dist_lockRep (A : Arith F K) (r : ℕ) (hr : r < 2) (S : Statement) (m k0 : ℕ) (l₀ l₁ : Level)
    (rest : List Level) (ext : ℕ) (hk0 : l₀.lanes = k0) (hm : 13 ≤ S.m) (C₀ : Oracle F)
    (extra : List (Claim F S.m)) (ℓ : ℕ) :
    Lock.Dist (slotOf r (ℓ + 1)) (repSlotDist S.m S.kLog extra.length k0 (l₁ :: rest) ℓ)
      (lockRep (D := D) A r hr S m k0 l₀ l₁ rest ext hk0 hm C₀ extra) := by
  unfold repSlotDist
  refine Lock.Dist.bind_eq
    (j := ligSlotDist (l₁ :: rest) ℓ + k0 + 1 + 1 + (extra.length + 1 + 1 + 1) + (S.kLog - 6 + 3)) ?_
    (Lock.noSlot_same _) (zerocheck_exact A S.m) (by rw [coinsK_zcShape])
  intro i₁ zc zc' hzc
  obtain ⟨hi, rfl⟩ := hzc
  subst i₁
  cases zc with
  | none => exact .noSlot (.ret _)
  | some zc =>
    refine Lock.Dist.bind_eq (j := ligSlotDist (l₁ :: rest) ℓ + k0 + 1 + 1 + (extra.length + 1 + 1 + 1)) ?_
      (Lock.noSlot_same _) (lincheck_exact A S zc) (by rw [coinsK_lcShape])
    intro i₂ ab ab' hab
    obtain ⟨hi, rfl⟩ := hab
    subst i₂
    cases ab with
    | none => exact .noSlot (.ret _)
    | some ab =>
      refine Lock.Dist.bind_eq (j := ligSlotDist (l₁ :: rest) ℓ + k0 + 1 + 1) ?_
        (Lock.noSlot_same _) (opening_exact A _ (ab :: cClaim A zc :: extra))
        (by rw [coinsK_opShape, List.length_cons, List.length_cons])
      intro i₃ op op' hop
      obtain ⟨hi, rfl⟩ := hop
      subst i₃
      cases op with
      | none => exact .noSlot (.ret _)
      | some op => exact dist_lockLig A r hr m k0 l₀ l₁ rest ext hk0 op.T op.b C₀ ℓ

/-! ## Past the last cap, and the deployed schedule -/

/-- **No cap past the recursion's last level.** -/
theorem noSlot_lockRec_ge (A : Arith F K) (r : ℕ) (hr : r < 2) (n : ℕ) (b : ℕ → F) (initial : List K)
    (ext : ℕ) (C₀ : Oracle F) (sch : Schedule) :
    ∀ (levs : List Level) (i : IRep F D) (st : LigState F K) (stC : LigStateC F K D)
      (h : RecInv r C₀ sch levs i st stC) (p : ℕ), levs.length ≤ p + 1 →
      Lock.NoSlot (slotOf r (stC.level + 1 + p)) (lockRec A r hr n b initial ext C₀ sch levs i st stC h)
  | [], _, _, _, _, _, _ => .ret _
  | [lv], i, st, stC, h, _, _ => noSlot_lockRec_single A r hr n b initial ext C₀ sch _ lv i st stC h
  | lv :: lv' :: rest, i, st, stC, h, 0, hp => by
      simp only [List.length_cons] at hp; omega
  | lv :: lv' :: rest, i, st, stC, h, p + 1, hp => by
      refine Lock.NoSlot.bind ?_ (noSlot_lockFold A r _ i _ lv.lanes _ _ _ _ _)
      intro i' x y hxy
      rcases x with ⟨ch, x₂⟩
      rcases x₂ with ⟨s, q⟩
      rcases y with ⟨ch', y₂⟩
      rcases y₂ with ⟨sF, qF⟩
      obtain ⟨hi, hch, hlen, hsq⟩ := hxy
      simp only at hch hlen hsq
      subst i' hch
      refine .table _ (by unfold slotOf; omega) fun C' cap => ?_
      refine Lock.NoSlot.bind ?_ (noSlot_lockTwoOod A r _ _ _ _ _ _ _ _ _ _ _ _)
      intro i₂ x y hxy
      rcases x with ⟨s2, x₂⟩
      rcases x₂ with ⟨q2, oods2⟩
      rcases y with ⟨sF2, y₂⟩
      rcases y₂ with ⟨qF2, oodsC2⟩
      obtain ⟨hi2, hoods, hsq2⟩ := hxy
      simp only at hoods hsq2
      subst i₂ hoods
      refine Lock.NoSlot.bind ?_ (noSlot_lockQuery A r _ _ _ _ _ _ _ _ _ _ _ true)
      intro i₃ st3 stC3 hq
      obtain ⟨hi3, hrel3, hlev, hq'⟩ := hq
      subst i₃
      have hlev' : stC3.level = stC.level := hlev
      refine Lock.NoSlot.slot_eq ?_ (noSlot_lockRec_ge A r hr n b initial ext C₀ sch (lv' :: rest) _ _ _ _ p ?_)
      · show slotOf r (stC3.level + 1 + 1 + p) = slotOf r (stC.level + 1 + (p + 1))
        unfold slotOf
        omega
      · simp only [List.length_cons] at hp ⊢; omega

/-- **No cap past Ligerito's last level.** -/
theorem noSlot_lockLig_ge (A : Arith F K) (r : ℕ) (hr : r < 2) (m k0 : ℕ) (l₀ l₁ : Level)
    (rest : List Level) (ext : ℕ) (hk0 : l₀.lanes = k0) (T : F) (b : ℕ → F) (C₀ : Oracle F) (ℓ : ℕ)
    (hℓ : (l₀ :: l₁ :: rest).length ≤ ℓ + 1) :
    Lock.NoSlot (slotOf r (ℓ + 1)) (lockLig (D := D) A r hr m k0 l₀ l₁ rest ext hk0 T b C₀) := by
  simp only [List.length_cons] at hℓ
  obtain ⟨p, rfl⟩ : ∃ p, ℓ = p + 1 := ⟨ℓ - 1, by omega⟩
  refine .coin _ fun z₀ => .send _ fun y₀ => .coin _ fun β₀ => .send _ fun msg => ?_
  refine Lock.NoSlot.bind ?_ (noSlot_lockFold A r _ _ _ k0 _ _ _ _ _)
  intro i' x y hxy
  rcases x with ⟨initial, x₂⟩
  rcases x₂ with ⟨s, q⟩
  rcases y with ⟨initial', y₂⟩
  rcases y₂ with ⟨sF, qF⟩
  obtain ⟨hi, hch, hlen, hsq⟩ := hxy
  simp only at hch hlen hsq
  subst i' hch
  refine .table _ (by unfold slotOf; omega) fun C₁ cap₁ => ?_
  refine Lock.NoSlot.bind ?_ (noSlot_lockTwoOod A r _ _ _ _ _ _ _ _ _ _ _ _)
  intro i₂ x y hxy
  rcases x with ⟨s2, x₂⟩
  rcases x₂ with ⟨q2, oods⟩
  rcases y with ⟨sF2, y₂⟩
  rcases y₂ with ⟨qF2, oodsC⟩
  obtain ⟨hi2, hoods, hsq2⟩ := hxy
  simp only at hoods hsq2
  subst i₂ hoods
  refine Lock.NoSlot.bind ?_ (noSlot_lockQuery A r _ _ _ _ _ _ _ _ _ _ _ true)
  intro i₃ st3 stC3 hq
  obtain ⟨hi3, hq'⟩ := hq
  subst i₃
  refine Lock.NoSlot.slot_eq ?_ (noSlot_lockRec_ge A r hr _ b _ ext C₀ _ (l₁ :: rest) _ _ _ _ p ?_)
  · show slotOf r (1 + 1 + p) = slotOf r (p + 1 + 1)
    unfold slotOf
    omega
  · simp only [List.length_cons]; omega

/-- **No cap past a rep's last level.** -/
theorem noSlot_lockRep_ge (A : Arith F K) (r : ℕ) (hr : r < 2) (S : Statement) (m k0 : ℕ) (l₀ l₁ : Level)
    (rest : List Level) (ext : ℕ) (hk0 : l₀.lanes = k0) (hm : 13 ≤ S.m) (C₀ : Oracle F)
    (extra : List (Claim F S.m)) (ℓ : ℕ) (hℓ : (l₀ :: l₁ :: rest).length ≤ ℓ + 1) :
    Lock.NoSlot (slotOf r (ℓ + 1)) (lockRep (D := D) A r hr S m k0 l₀ l₁ rest ext hk0 hm C₀ extra) := by
  refine Lock.NoSlot.bind ?_ (Lock.noSlot_same _)
  intro i₁ zc zc' hzc
  obtain ⟨hi, rfl⟩ := hzc
  subst i₁
  cases zc with
  | none => exact .ret _
  | some zc =>
    refine Lock.NoSlot.bind ?_ (Lock.noSlot_same _)
    intro i₂ ab ab' hab
    obtain ⟨hi, rfl⟩ := hab
    subst i₂
    cases ab with
    | none => exact .ret _
    | some ab =>
      refine Lock.NoSlot.bind ?_ (Lock.noSlot_same _)
      intro i₃ op op' hop
      obtain ⟨hi, rfl⟩ := hop
      subst i₃
      cases op with
      | none => exact .ret _
      | some op => exact noSlot_lockLig_ge A r hr m k0 l₀ l₁ rest ext hk0 op.T op.b C₀ ℓ hℓ

/-- With four lanes at every level, the recursion's level caps are `11` coins apart. -/
theorem recSlotDist_eq : ∀ (levs : List Level) (p : ℕ), (∀ lv ∈ levs, lv.lanes = 4) → p + 1 < levs.length →
    recSlotDist levs p = 4 + 11 * p
  | [], _, _, hp => by simp only [List.length_nil] at hp; omega
  | [_], _, _, hp => by simp only [List.length_cons, List.length_nil] at hp; omega
  | lv :: lv' :: rest, 0, hl, _ => by
      have h1 := hl lv (by simp)
      simp only [recSlotDist]; omega
  | lv :: lv' :: rest, p + 1, hl, hp => by
      have h1 := hl lv (by simp)
      have h2 := recSlotDist_eq (lv' :: rest) p (fun x hx => hl x (List.mem_cons_of_mem _ hx))
        (by simp only [List.length_cons] at hp ⊢; omega)
      simp only [recSlotDist]; omega

/-- **`fast100` folds four lanes at every level after level 0.** -/
theorem fast100_lanes {m : ℕ} {s : Schedule} (hs : fast100 m = some s) : ∀ lv ∈ s.levels.tail, lv.lanes = 4 := by
  unfold fast100 at hs
  split at hs <;> cases hs <;> decide

end FlockSoundness
