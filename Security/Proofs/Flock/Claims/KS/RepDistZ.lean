import Proofs.Flock.Claims.KS.RepDist
import Proofs.Flock.Claims.KS.SlotLead
import Proofs.Flock.Soundness.Discharge.ZkExec.Shape

/-!
# A masked rep's level caps, at a distance fixed by the statement and the schedule

A masked rep's lockstep (`lockRepZ`) meets its level-`ℓ + 1` cap's table node, `slotOf r (ℓ + 1)` moved four up, after
the same number of coins on every path that meets it: the masked zerocheck's, lincheck's and opening's coins (the
non-ZK session's shapes), level 0's two first coins, its `k0` masked lane rounds and its `e` extra draws, then, per
level passed, plain's count (`RepDist`). The pads root and `τ`'s commitment sit at slots below four, and a path whose
opening rejects goes to the inner proof and meets no cap.

* `coinsK_foldShapeZ`, `coinsK_replicate_coin`: the masked lane rounds' and the extra draws' coins;
* `noSlot_lockQueryZ`, `noSlot_lockFoldZK`, `noSlot_lockQuery0Z`, `noSlot_lockInnerZ`: the phases between caps, and
  the inner proof, have no level table node;
* `dist_lockRecZ`, `dist_lockLigTailZ`, `dist_lockLigPadZ`, `dist_lockRestZ`, `dist_lockMaskedZ`: each cap's distance
  through the recursion, the masked level 0 (`ligPadSlotDist`), the opening and the whole masked phase
  (`maskedSlotDist`);
* `noSlot_reindex`, `dist_reindex`: moving a lockstep's slots by an injective map moves its distances;
* `dist_lockRepZ`: so a whole masked rep's;
* `noSlot_lockRecZ_ge`, `noSlot_lockLigTailZ_ge`, `noSlot_lockLigPadZ_ge`, `noSlot_lockRestZ_ge`,
  `noSlot_lockMaskedZ_ge`, `noSlot_lockRepZ_ge`: past the last level there is no cap.
-/

namespace FlockSoundness.Discharge.ZkLink

open Game Model Accounting ZkSession Exec ZkExec

section Reindex

variable {α β T Dd I J : Type} {getI : I → ℕ → Option (T × Dd)} {getJ : J → ℕ → Option (T × Dd)}
  {R : I → α → β → Prop} (emb : I → J) (sl : ℕ → ℕ) (hget : ∀ i s, getJ (emb i) (sl s) = getI i s)
  (hoff : ∀ i i' s, (∀ s', sl s' ≠ s) → getJ (emb i) s = getJ (emb i') s)

/-- **Moving the slots keeps no table node of a slot**, moved. -/
theorem noSlot_reindex (hsl : Function.Injective sl) {slot : ℕ} {i : I} {g : Game α} {g' : Game β}
    {L : Lock T Dd getI R i g g'} (h : Lock.NoSlot slot L) :
    Lock.NoSlot (sl slot) (lockReindex emb sl hget hoff L) := by
  induction h with
  | ret hr => exact .ret _
  | coin C _ ih => exact .coin C ih
  | send M _ ih => exact .send M ih
  | table s hne _ ih => exact .table (sl s) (fun h => hne (hsl h)) ih

/-- **Moving the slots keeps a slot's distance**, at the moved slot. -/
theorem dist_reindex (hsl : Function.Injective sl) {slot j : ℕ} {i : I} {g : Game α} {g' : Game β}
    {L : Lock T Dd getI R i g g'} (h : Lock.Dist slot j L) :
    Lock.Dist (sl slot) j (lockReindex emb sl hget hoff L) := by
  induction h with
  | coin C _ ih => exact .coin C ih
  | send M _ ih => exact .send M ih
  | other s hne _ ih => exact .other (sl s) (fun h => hne (hsl h)) ih
  | here => exact .here
  | noSlot h0 => exact .noSlot (noSlot_reindex emb sl hget hoff hsl h0)

end Reindex

theorem coinsK_foldShapeZ (F K : Type) : ∀ k, Lock.coinsK (foldShapeZ F K k) = k
  | 0 => rfl
  | 1 => rfl
  | k + 2 => by
      have := coinsK_foldShapeZ F K (k + 1)
      simp only [foldShapeZ, Lock.coinsK]
      omega

theorem coinsK_replicate_coin (C : Type) : ∀ e, Lock.coinsK (List.replicate e (Kind.coin C)) = e
  | 0 => rfl
  | e + 1 => by rw [List.replicate_succ, Lock.coinsK, coinsK_replicate_coin C e]

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
variable {D Sl : Type} {mPts : ℕ}

/-! ## The recursive rounds -/

theorem noSlot_lockQueryZ (A : Arith F K) (r slot : ℕ) (C₀ : Oracle F) (w₀ : ℕ) (i : IRep F D) (lv : Level)
    (ch : List K) (start start' : ℕ) (hstart : start = start') (st : LigState F K) (stC : LigStateC F K D)
    (h : StRelZ r C₀ w₀ i st stC) (hlev : stC.level ≠ 0) (wm : Bool) :
    Lock.NoSlot slot (lockQueryZ A r C₀ w₀ i lv ch start start' hstart st stC h hlev wm) := by
  subst hstart
  cases wm
  · exact .coin _ fun _ => .coin _ fun _ => .coin _ fun _ => .ret _
  · exact .coin _ fun _ => .coin _ fun _ => .send _ fun _ => .coin _ fun _ => .ret _

/-- **The last level has no cap.** -/
theorem noSlot_lockRecZ_single (A : Arith F K) (r : ℕ) (hr : r < 2) (n : ℕ) (b : ℕ → F) (initial : List K)
    (ext : ℕ) (C₀ : Oracle F) (w₀ : ℕ) (sch : Schedule) (slot : ℕ) (lv : Level) (i : IRep F D) (st : LigState F K)
    (stC : LigStateC F K D) (h : RecInvZ r C₀ w₀ sch [lv] i st stC) :
    Lock.NoSlot slot (lockRecZ A r hr n b initial ext C₀ w₀ sch [lv] i st stC h) := by
  refine Lock.NoSlot.bind ?_ (noSlot_lockFold A r _ i _ lv.lanes _ _ _ _ _)
  intro i' x y hxy
  rcases x with ⟨ch, x₂⟩
  rcases x₂ with ⟨s, q⟩
  rcases y with ⟨ch', y₂⟩
  rcases y₂ with ⟨sF, qF⟩
  obtain ⟨hi, hch, hlen, hsq⟩ := hxy
  simp only at hch hlen hsq
  subst i' hch
  refine .send _ fun yr => Lock.NoSlot.bind ?_ (noSlot_lockQueryZ A r _ _ _ _ _ _ _ _ _ _ _ _ _ false)
  intro i3 st3 stC3 hq
  obtain ⟨hi3, hq'⟩ := hq
  subst i3
  exact .ret _

/-- **The masked recursion's caps' distances**, plain's (`recSlotDist`). -/
theorem dist_lockRecZ (A : Arith F K) (r : ℕ) (hr : r < 2) (n : ℕ) (b : ℕ → F) (initial : List K) (ext : ℕ)
    (C₀ : Oracle F) (w₀ : ℕ) (sch : Schedule) :
    ∀ (levs : List Level) (i : IRep F D) (st : LigState F K) (stC : LigStateC F K D)
      (h : RecInvZ r C₀ w₀ sch levs i st stC) (p : ℕ),
      Lock.Dist (slotOf r (stC.level + 1 + p)) (recSlotDist levs p)
        (lockRecZ A r hr n b initial ext C₀ w₀ sch levs i st stC h)
  | [], _, _, _, _, _ => .noSlot (.ret _)
  | [lv], i, st, stC, h, _ =>
      .noSlot (noSlot_lockRecZ_single A r hr n b initial ext C₀ w₀ sch _ lv i st stC h)
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
        (noSlot_lockQueryZ A r _ _ _ _ _ _ _ _ _ _ _ _ _ true) (queryAndBatchC_exact A r lv _ _ _ true)
        (by rw [coinsK_qbShape])
      intro i₃ st3 stC3 hq
      obtain ⟨hi3, hrel3, hlev, hq'⟩ := hq
      subst i₃
      have hlev' : stC3.level = stC.level := hlev
      refine Lock.Dist.slot_eq ?_ (dist_lockRecZ A r hr n b initial ext C₀ w₀ sch (lv' :: rest) _ _ _ _ p)
      show slotOf r (stC3.level + 1 + 1 + p) = slotOf r (stC.level + 1 + (p + 1))
      unfold slotOf
      omega

/-! ## The masked level 0 -/

omit [DecidableEq F] [Fintype K] [DecidableEq K] in
theorem noSlot_lockFoldZK (A : Arith F K) {np e : ℕ} (P : Pads0 np e) (r slot : ℕ) (i : IRep F D) :
    ∀ (k j : ℕ) (s : LinK K np) (q : QuadL K np), Lock.NoSlot slot (lockFoldZK A P r i k j s q)
  | 0, _, _, _ => .ret _
  | 1, _, _, _ => .coin _ fun _ => .ret _
  | k + 2, j, _, _ => .coin _ fun _ => .send _ fun _ =>
      Lock.NoSlot.bind (fun _ x y hxy => by
          obtain ⟨rfl, rfl, hl⟩ := hxy
          rcases x with ⟨rs, s''⟩
          exact .ret _)
        (noSlot_lockFoldZK A P r slot i (k + 1) (j + 1) _ _)

theorem noSlot_lockQuery0Z (A : Arith F K) (r slot : ℕ) (C₀ : Oracle F) (w₀ : ℕ) (i : IRep F D) (lv : Level)
    (v : ℕ → K) (col : List F → ℕ → K)
    (hcol : ∀ (row : List F) (p : ℕ), (∀ lane < w₀, row.getD lane 0 = C₀ lane p) → col row p = v p)
    (st : LigState F K) (stC : LigStateC F K D) (h : StRelZ r C₀ w₀ i st stC) (hlev : stC.level = 0) :
    Lock.NoSlot slot (lockQuery0Z A r C₀ w₀ i lv v col hcol st stC h hlev) :=
  .coin _ fun _ => .coin _ fun _ => .send _ fun _ => .coin _ fun _ => .ret _

/-- **Levels 1+ of the masked level 0: the caps' distances** from the code-switch message, plain Ligerito's after its
level-0 folds (`ligSlotDist`). -/
theorem dist_lockLigTailZ (A : Arith F K) (r : ℕ) (hr : r < 2) (sch : Schedule) (t : ℕ) (c₀ : F) (e : ℕ)
    (b : ℕ → F) (C₀ : Oracle F) (lv₀ : Level) (z₀ : Fin (sch.m - 7) → F) (β₀ : F) (initial : List K)
    (hinit : initial.length = sch.k0) (ρ : Fin e → K) (ybar : Fin t → K) (s : K) (ℓ : ℕ) :
    ∀ (rest : List Level) (hlevs : sch.levels = lv₀ :: rest),
      Lock.Dist (slotOf r (ℓ + 1)) (ligSlotDist rest ℓ)
        (lockLigTailZ (D := D) A r hr sch t c₀ e b C₀ lv₀ z₀ β₀ initial hinit ρ ybar s rest hlevs)
  | [], _ => .noSlot (.send _ fun _ => .ret _)
  | lv₁ :: rest, hlevs => by
      refine .send _ fun msg' => ?_
      cases ℓ with
      | zero => exact .here
      | succ p =>
        refine .other _ (by unfold slotOf; omega) fun C₁ cap₁ => ?_
        refine Lock.Dist.bind_eq (j := recSlotDist (lv₁ :: rest) p + 3) ?_
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
        refine Lock.Dist.bind_eq (j := recSlotDist (lv₁ :: rest) p) ?_
          (noSlot_lockQuery0Z A r _ _ _ _ _ _ _ _ _ _ _ _) (queryAndBatchPadC_exact A r lv₀ _ _ _)
          (by rw [coinsK_qbShape])
        intro i₃ st3 stC3 hq
        obtain ⟨hi3, hq'⟩ := hq
        subst i₃
        refine Lock.Dist.slot_eq ?_
          (dist_lockRecZ A r hr _ b initial sch.ext C₀ _ sch (lv₁ :: rest) _ _ _ _ p)
        show slotOf r (1 + 1 + p) = slotOf r (p + 1 + 1)
        unfold slotOf
        omega

/-- The coins from the masked level 0's start to its level-`ℓ + 1` cap: its two first coins, its `k0` lane rounds,
its `e` extra draws, then `ligSlotDist`. -/
def ligPadSlotDist (sch : Schedule) (e ℓ : ℕ) : ℕ :=
  ligSlotDist sch.levels.tail ℓ + e + sch.k0 + 1 + 1

/-- **The masked level 0's caps' distances.** -/
theorem dist_lockLigPadZ (A : Arith F K) (r : ℕ) (hr : r < 2) {np e : ℕ} (P : Pads0 np e) (t : ℕ) (c₀ : F)
    (T : F) (b : ℕ → F) (C₀ : Oracle F) (ℓ : ℕ) :
    ∀ sch : Schedule, Lock.Dist (slotOf r (ℓ + 1)) (ligPadSlotDist sch e ℓ)
      (lockLigPadZ (D := D) A r hr P t c₀ T b C₀ sch)
  | ⟨_, _, [], _⟩ => .noSlot (.ret _)
  | ⟨m, k0, lv₀ :: rest, ext⟩ => by
      show Lock.Dist _ (ligSlotDist rest ℓ + e + k0 + 1 + 1) _
      refine .coin _ fun z₀ => .send _ fun y₀ => .coin _ fun β₀ => .send _ fun msg => ?_
      refine Lock.Dist.bind_eq (j := ligSlotDist rest ℓ + e) ?_ (noSlot_lockFoldZK A P r _ _ k0 0 _ _)
        (foldRoundsZK_exact A P _ _ _ _) (by rw [coinsK_foldShapeZ])
      intro i' x y hxy
      obtain ⟨rfl, rfl, hlen⟩ := hxy
      rcases x with ⟨initial, s⟩
      refine .send _ fun ev => ?_
      refine Lock.Dist.bind_eq (j := ligSlotDist rest ℓ) ?_ (Lock.noSlot_same _) (drawKs_exact A e)
        (by rw [coinsK_replicate_coin])
      intro i'' ρ ρ' hρ
      obtain ⟨rfl, rfl⟩ := hρ
      refine .send _ fun T' => .send _ fun ybar => ?_
      refine Lock.Dist.bind_noSlot ?_ (dist_lockLigTailZ A r hr ⟨m, k0, lv₀ :: rest, ext⟩ t c₀ e b C₀ lv₀ z₀ β₀
        initial hlen _ ybar T' ℓ rest rfl)
      intro _ _ _ _
      exact .ret _

/-! ## The masked phase -/

/-- **The rest of a masked rep: the caps' distances**, after the opening's coins. -/
theorem dist_lockRestZ (A : Arith F K) (r : ℕ) (hr : r < 2) {np e : ℕ} (P₀ : Pads0 np e) (PA : PadsA np)
    (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (t : ℕ) (c₀ : F) (C₀ : Oracle F) (extra : List (Claim F S.m))
    (zc : ZcZK F np S.m) (ab : ClaimZK F np S.m) (ℓ : ℕ) :
    Lock.Dist (slotOf r (ℓ + 1)) (ligPadSlotDist sch e ℓ + (extra.length + 2 + 1))
      (lockRestZ (D := D) A r hr P₀ PA S sch hm t c₀ C₀ extra zc ab) := by
  refine Lock.Dist.bind_eq (j := ligPadSlotDist sch e ℓ) ?_ (Lock.noSlot_same _) (openingZK_exact A _ _ _ extra)
    (by rw [coinsK_opShape])
  intro i' o o' h
  obtain ⟨rfl, rfl⟩ := h
  rcases o with ⟨oo, x⟩
  cases oo with
  | none => exact .noSlot (.ret _)
  | some op =>
    refine Lock.Dist.bind_noSlot ?_ (dist_lockLigPadZ A r hr P₀ t c₀ op.T op.b C₀ ℓ sch)
    intro _ _ _ _
    exact .send _ fun _ => .ret _

/-- The coins from a masked rep's masked phase to its level-`ℓ + 1` cap: the zerocheck's, the lincheck's, the
opening's of its `nExtra + 2` claims, then `ligPadSlotDist`. -/
def maskedSlotDist (m kLog nExtra : ℕ) (sch : Schedule) (e ℓ : ℕ) : ℕ :=
  ligPadSlotDist sch e ℓ + (nExtra + 2 + 1) + (kLog - 6 + 3) + (m - 6 + 3)

/-- **The masked phase's caps' distances.** -/
theorem dist_lockMaskedZ (A : Arith F K) (r : ℕ) (hr : r < 2) {np e : ℕ} (P₀ : Pads0 np e) (PA : PadsA np)
    (tr : Triple np) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (t : ℕ) (c₀ : F) (C₀ : Oracle F)
    (extra : List (Claim F S.m)) (ℓ : ℕ) :
    Lock.Dist (slotOf r (ℓ + 1)) (maskedSlotDist S.m S.kLog extra.length sch e ℓ)
      (lockMaskedZ (D := D) A r hr P₀ PA tr S sch hm t c₀ C₀ extra) := by
  unfold maskedSlotDist
  refine Lock.Dist.bind_noSlot ?_ ?_
  · intro _ _ _ _
    exact .ret _
  refine Lock.Dist.bind_eq (j := ligPadSlotDist sch e ℓ + (extra.length + 2 + 1) + (S.kLog - 6 + 3)) ?_
    (Lock.noSlot_same _) (zerocheckZK_exact A S.m PA tr) (by rw [coinsK_zcShape])
  intro i₁ o o' h
  obtain ⟨rfl, rfl⟩ := h
  refine Lock.Dist.bind_noSlot ?_ ?_
  · intro _ _ _ _
    exact .ret _
  refine Lock.Dist.bind_eq (j := ligPadSlotDist sch e ℓ + (extra.length + 2 + 1)) ?_
    (Lock.noSlot_same _) (lincheckZK_exact A S PA o.1) (by rw [coinsK_lcShape])
  intro i₂ o₂ o₂' h₂
  obtain ⟨rfl, rfl⟩ := h₂
  refine Lock.Dist.bind_noSlot ?_ ?_
  · intro _ _ _ _
    exact .ret _
  exact dist_lockRestZ A r hr P₀ PA S sch hm t c₀ C₀ extra o.1 o₂.1 ℓ

/-! ## The inner proof, and a rep -/

theorem noSlot_lockInnerZ (Hs : Hashes F D Sl) (r : ℕ) (hr : r < 2) {N : ℕ} (dom : Fin N ↪ F) (s nP q : ℕ)
    (pos : F → Fin N) {nA : ℕ} (rows : Fin nA → Lin F s) (tr : Triple s) (lev : IRep F D) (O : Oracle F)
    (c : Cap D) (slot : ℕ) (hs : 4 ≤ slot) :
    Lock.NoSlot slot (lockInnerZ Hs r hr dom s nP q pos rows tr lev O c) :=
  .coin _ fun _ => .send _ fun _ => .coin _ fun _ => .table _ (by omega) fun _ _ =>
    .coin _ fun _ => .send _ fun _ => .coin _ fun _ => .ret _

/-- **A masked rep's caps' distances**: every path of rep `r` that meets its level-`ℓ + 1` cap's table node meets it
after `maskedSlotDist` coins; the pads root and the inner proof hold no coin before it and no level slot. -/
theorem dist_lockRepZ (A : Arith F K) (Hs : Hashes F D Sl) (T : TabZK F mPts) (r : ℕ) (hr : r < 2)
    (C₀ : Oracle F) (extra : List (Claim F T.S.m)) (ℓ : ℕ) :
    Lock.Dist (slotOf r (ℓ + 1) + 4) (maskedSlotDist T.S.m T.S.kLog extra.length T.sch T.e ℓ)
      (lockRepZ A Hs T r hr C₀ extra) := by
  refine .other r (by unfold slotOf; omega) fun O c => ?_
  refine Lock.Dist.bind_noSlot ?_ ?_
  · intro j o o' h
    rcases j with ⟨lev, pads, tau⟩
    obtain ⟨hp, ht, h⟩ := h
    simp only at hp ht h
    subst hp ht
    obtain ⟨hrows, hrec⟩ := h
    rcases o with ⟨rows, v⟩
    rcases o' with ⟨rows', p⟩
    simp only at hrows hrec
    subst hrows
    refine Lock.NoSlot.bind ?_ (noSlot_lockInnerZ Hs r hr _ _ _ _ _ _ _ _ _ _ _ (by omega))
    intro _ _ _ _
    exact .ret _
  · exact Lock.Dist.mono _ (dist_reindex _ (· + 4) _ _ (add_left_injective 4)
      (dist_lockMaskedZ A r hr T.pads T.padsA T.tr T.S T.sch T.hm T.t T.c₀ C₀ extra ℓ))

/-! ## Past the last cap -/

/-- **No cap past the masked recursion's last level.** -/
theorem noSlot_lockRecZ_ge (A : Arith F K) (r : ℕ) (hr : r < 2) (n : ℕ) (b : ℕ → F) (initial : List K) (ext : ℕ)
    (C₀ : Oracle F) (w₀ : ℕ) (sch : Schedule) :
    ∀ (levs : List Level) (i : IRep F D) (st : LigState F K) (stC : LigStateC F K D)
      (h : RecInvZ r C₀ w₀ sch levs i st stC) (p : ℕ), levs.length ≤ p + 1 →
      Lock.NoSlot (slotOf r (stC.level + 1 + p)) (lockRecZ A r hr n b initial ext C₀ w₀ sch levs i st stC h)
  | [], _, _, _, _, _, _ => .ret _
  | [lv], i, st, stC, h, _, _ => noSlot_lockRecZ_single A r hr n b initial ext C₀ w₀ sch _ lv i st stC h
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
      refine Lock.NoSlot.bind ?_ (noSlot_lockQueryZ A r _ _ _ _ _ _ _ _ _ _ _ _ _ true)
      intro i₃ st3 stC3 hq
      obtain ⟨hi3, hrel3, hlev, hq'⟩ := hq
      subst i₃
      have hlev' : stC3.level = stC.level := hlev
      refine Lock.NoSlot.slot_eq ?_
        (noSlot_lockRecZ_ge A r hr n b initial ext C₀ w₀ sch (lv' :: rest) _ _ _ _ p ?_)
      · show slotOf r (stC3.level + 1 + 1 + p) = slotOf r (stC.level + 1 + (p + 1))
        unfold slotOf
        omega
      · simp only [List.length_cons] at hp ⊢; omega

/-- **No cap past the masked level 0's last level**, from the code-switch message. -/
theorem noSlot_lockLigTailZ_ge (A : Arith F K) (r : ℕ) (hr : r < 2) (sch : Schedule) (t : ℕ) (c₀ : F) (e : ℕ)
    (b : ℕ → F) (C₀ : Oracle F) (lv₀ : Level) (z₀ : Fin (sch.m - 7) → F) (β₀ : F) (initial : List K)
    (hinit : initial.length = sch.k0) (ρ : Fin e → K) (ybar : Fin t → K) (s : K) (ℓ : ℕ) :
    ∀ (rest : List Level) (hlevs : sch.levels = lv₀ :: rest), rest.length ≤ ℓ →
      Lock.NoSlot (slotOf r (ℓ + 1))
        (lockLigTailZ (D := D) A r hr sch t c₀ e b C₀ lv₀ z₀ β₀ initial hinit ρ ybar s rest hlevs)
  | [], _, _ => .send _ fun _ => .ret _
  | lv₁ :: rest, hlevs, hℓ => by
      simp only [List.length_cons] at hℓ
      obtain ⟨p, rfl⟩ : ∃ p, ℓ = p + 1 := ⟨ℓ - 1, by omega⟩
      refine .send _ fun msg' => ?_
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
      refine Lock.NoSlot.bind ?_ (noSlot_lockQuery0Z A r _ _ _ _ _ _ _ _ _ _ _ _)
      intro i₃ st3 stC3 hq
      obtain ⟨hi3, hq'⟩ := hq
      subst i₃
      refine Lock.NoSlot.slot_eq ?_
        (noSlot_lockRecZ_ge A r hr _ b initial sch.ext C₀ _ sch (lv₁ :: rest) _ _ _ _ p ?_)
      · show slotOf r (1 + 1 + p) = slotOf r (p + 1 + 1)
        unfold slotOf
        omega
      · simp only [List.length_cons]; omega

/-- **No cap past the masked level 0's last level.** -/
theorem noSlot_lockLigPadZ_ge (A : Arith F K) (r : ℕ) (hr : r < 2) {np e : ℕ} (P : Pads0 np e) (t : ℕ) (c₀ : F)
    (T : F) (b : ℕ → F) (C₀ : Oracle F) (ℓ : ℕ) :
    ∀ sch : Schedule, sch.levels.length ≤ ℓ + 1 → Lock.NoSlot (slotOf r (ℓ + 1))
      (lockLigPadZ (D := D) A r hr P t c₀ T b C₀ sch)
  | ⟨_, _, [], _⟩, _ => .ret _
  | ⟨m, k0, lv₀ :: rest, ext⟩, hℓ => by
      simp only [List.length_cons] at hℓ
      refine .coin _ fun z₀ => .send _ fun y₀ => .coin _ fun β₀ => .send _ fun msg => ?_
      refine Lock.NoSlot.bind ?_ (noSlot_lockFoldZK A P r _ _ k0 0 _ _)
      intro i' x y hxy
      obtain ⟨rfl, rfl, hlen⟩ := hxy
      rcases x with ⟨initial, s⟩
      refine .send _ fun ev => ?_
      refine Lock.NoSlot.bind ?_ (Lock.noSlot_same _)
      intro i'' ρ ρ' hρ
      obtain ⟨rfl, rfl⟩ := hρ
      refine .send _ fun T' => .send _ fun ybar => ?_
      refine Lock.NoSlot.bind ?_ (noSlot_lockLigTailZ_ge A r hr ⟨m, k0, lv₀ :: rest, ext⟩ t c₀ e b C₀ lv₀ z₀ β₀
        initial hlen _ ybar T' ℓ rest rfl (by omega))
      intro _ _ _ _
      exact .ret _

/-- **No cap past the last level in the rest of a masked rep.** -/
theorem noSlot_lockRestZ_ge (A : Arith F K) (r : ℕ) (hr : r < 2) {np e : ℕ} (P₀ : Pads0 np e) (PA : PadsA np)
    (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (t : ℕ) (c₀ : F) (C₀ : Oracle F) (extra : List (Claim F S.m))
    (zc : ZcZK F np S.m) (ab : ClaimZK F np S.m) (ℓ : ℕ) (hℓ : sch.levels.length ≤ ℓ + 1) :
    Lock.NoSlot (slotOf r (ℓ + 1)) (lockRestZ (D := D) A r hr P₀ PA S sch hm t c₀ C₀ extra zc ab) := by
  refine Lock.NoSlot.bind ?_ (Lock.noSlot_same _)
  intro i' o o' h
  obtain ⟨rfl, rfl⟩ := h
  rcases o with ⟨oo, x⟩
  cases oo with
  | none => exact .ret _
  | some op =>
    refine Lock.NoSlot.bind ?_ (noSlot_lockLigPadZ_ge A r hr P₀ t c₀ op.T op.b C₀ ℓ sch hℓ)
    intro _ _ _ _
    exact .send _ fun _ => .ret _

/-- **No cap past the last level in the masked phase.** -/
theorem noSlot_lockMaskedZ_ge (A : Arith F K) (r : ℕ) (hr : r < 2) {np e : ℕ} (P₀ : Pads0 np e) (PA : PadsA np)
    (tr : Triple np) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (t : ℕ) (c₀ : F) (C₀ : Oracle F)
    (extra : List (Claim F S.m)) (ℓ : ℕ) (hℓ : sch.levels.length ≤ ℓ + 1) :
    Lock.NoSlot (slotOf r (ℓ + 1)) (lockMaskedZ (D := D) A r hr P₀ PA tr S sch hm t c₀ C₀ extra) := by
  refine Lock.NoSlot.bind ?_ ?_
  · intro _ _ _ _
    exact .ret _
  refine Lock.NoSlot.bind ?_ (Lock.noSlot_same _)
  intro i₁ o o' h
  obtain ⟨rfl, rfl⟩ := h
  refine Lock.NoSlot.bind ?_ ?_
  · intro _ _ _ _
    exact .ret _
  refine Lock.NoSlot.bind ?_ (Lock.noSlot_same _)
  intro i₂ o₂ o₂' h₂
  obtain ⟨rfl, rfl⟩ := h₂
  refine Lock.NoSlot.bind ?_ ?_
  · intro _ _ _ _
    exact .ret _
  exact noSlot_lockRestZ_ge A r hr P₀ PA S sch hm t c₀ C₀ extra o.1 o₂.1 ℓ hℓ

/-- **No cap past a masked rep's last level.** -/
theorem noSlot_lockRepZ_ge (A : Arith F K) (Hs : Hashes F D Sl) (T : TabZK F mPts) (r : ℕ) (hr : r < 2)
    (C₀ : Oracle F) (extra : List (Claim F T.S.m)) (ℓ : ℕ) (hℓ : T.sch.levels.length ≤ ℓ + 1) :
    Lock.NoSlot (slotOf r (ℓ + 1) + 4) (lockRepZ A Hs T r hr C₀ extra) := by
  refine .table r (by unfold slotOf; omega) fun O c => ?_
  refine Lock.NoSlot.bind ?_ ?_
  · intro j o o' h
    rcases j with ⟨lev, pads, tau⟩
    obtain ⟨hp, ht, h⟩ := h
    simp only at hp ht h
    subst hp ht
    obtain ⟨hrows, hrec⟩ := h
    rcases o with ⟨rows, v⟩
    rcases o' with ⟨rows', p⟩
    simp only at hrows hrec
    subst hrows
    refine Lock.NoSlot.bind ?_ (noSlot_lockInnerZ Hs r hr _ _ _ _ _ _ _ _ _ _ _ (by omega))
    intro _ _ _ _
    exact .ret _
  · exact Lock.NoSlot.mono _ (noSlot_reindex _ (· + 4) _ _ (add_left_injective 4)
      (noSlot_lockMaskedZ_ge A r hr T.pads T.padsA T.tr T.S T.sch T.hm T.t T.c₀ C₀ extra ℓ hℓ))

end FlockSoundness.Discharge.ZkLink
