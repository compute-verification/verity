import Proofs.Flock.Claims.Complete.Zk.Level0Model
import Proofs.Flock.Claims.Complete.Zk.Level0Premises
import Proofs.Flock.Claims.Complete.ZkTables.LaneRows
import Proofs.Flock.Claims.Complete.Replay.LigeritoHonestRun

/-!
# The honest masked Ligerito meets the masked model's final identity

`Lg.honest` for the masked prover (`ligeritoPZk`) against the masked verifier's model (`outZ`).

* **The `--zk` schedule's facts** (`zkSch_facts`): fast100's dimensions and levels, and level 0's `t_pad ≤ 2^cols`.
* **Level 0 ends in `Mid`** (`mid_lv0Z`): after the lane rounds the prover's table is the split of the blinded fold
  `y₁' = f + ρ_0 R_{2r} + ρ_1 R_{2r+1}` and the claim is `T' = ⟨W, y₁'⟩`; level 0's masked consistency claim
  (`Sync.closeZk`, the padding cancelled by `ybar`) then keeps it as plain's close does.
* **The last level** (`goal_lastZ`): plain's `goal_last` at the masked model's level loop.
-/

namespace FlockComplete.L0

open Flock Transcript FlockSoundness FlockSoundness.Refine FlockComplete Replay Replay.Lg

local notation "𝕂" => FlockLevel3.GF256

/-! ## The `--zk` schedule -/

/-- **The `--zk` schedule's facts**: fast100's dimensions, levels and caps, and level 0's `t_pad` padding coefficients
within its `2^cols` columns. -/
theorem zkSch_facts {m : ℕ} (h1 : 25 ≤ m) (h2 : m ≤ 35) {v : Schedule × ℕ} (h : Zk.schedule m = .ok v) :
    v.1.n ≠ 0 ∧ 1 ≤ v.1.k0 ∧ 2 ≤ v.1.levels.size ∧ v.1.k0 + 3 * (v.1.levels.size - 1) ≤ v.1.n ∧
      (∀ j < v.1.levels.size, (v.1.levels.getD j default).q ≠ 0) ∧
      (∀ i < v.1.levels.size, LevelSeam (v.1.levels.getD i default)) ∧
      (v.1.levels.getD 0 default).lanes = v.1.k0 ∧
      (∀ i < v.1.levels.size, 1 ≤ i → (v.1.levels.getD i default).lanes = 4) ∧
      (∀ i < v.1.levels.size, (v.1.levels.getD i default).cols = v.1.n - v.1.k0 - 3 * i) ∧
      v.2 ≤ 2 ^ (v.1.levels.getD 0 default).cols ∧
      (∀ i < v.1.levels.size, (v.1.levels.getD i default).cap ≤ (v.1.levels.getD i default).d) ∧
      v.1.ext = v.1.n - v.1.k0 - 3 * (v.1.levels.size - 1) := by
  obtain ⟨-, -, -, hk0, -, ⟨-, hL, hl0, hc0, hl4, hc, hext⟩, hseam, -, hlv⟩ :=
    FlockSoundness.Discharge.ZkExec.zk_schedule_seams m v.1 v.2 h
  have hlig := zkSchedule_lig h1 h2 h
  have hcols : ∀ i < v.1.levels.size, (v.1.levels.getD i default).cols = v.1.n - v.1.k0 - 3 * i := by
    intro i hi
    induction i with
    | zero => rw [hc0]; rfl
    | succ i ih =>
      have := hc i (by omega)
      rw [ih (by omega)] at this
      omega
  have ht : ∀ k : Fin 11, (match Zk.schedule (k.val + 25) with
      | .ok v => decide (v.2 ≤ 2 ^ (v.1.levels.getD 0 default).cols)
      | .error _ => false) = true := by
    decide +kernel
  have ht' := ht ⟨m - 25, by omega⟩
  simp only [show m - 25 + 25 = m by omega, h, decide_eq_true_eq] at ht'
  refine ⟨zkSchedule_n h, hk0, hL, by have := hlig.dims; omega, fun j hj => ?_, hseam, hl0, hl4, hcols, ht',
    hlig.cap, by have := hlig.dims; omega⟩
  exact Nat.pos_iff_ne_zero.1 (hlv _ (getD_mem _ hj)).2.1

/-! ## The masked prover's own steps -/

section Steps

variable {p : LigeritoProof} {L : ℕ} {β : Type}

/-- An F256 absorbed, read by the model as `vM`: they agree once the rest of the run fixes `v = vM`. -/
theorem HRuns.absorbKM (v vM : F256) (c : Cur) (tm : Transcript) (J : TW → Prop) :
    HRuns p L (liftT (liftTM (absorbK v))) c tm c (ap (pKv vM) tm) (fun _ => True) (fun _ => v = vM) J
      fun _ => J := by
  refine ⟨fun st t l b st' t' l' hl h => ?_, fun st t l b st' t' l' hl h hE _ hc ht hJ => ?_⟩
  · obtain ⟨hq, hg, hl', hc⟩ := PRuns.absorb (p := p) (L := L) (absorbK v) (pKv v)
      (fun t => Flock.RecAlgebra.absorbSlice_eq _ t) c tm st t l b st' t' l' hl h
    exact ⟨hq, hg, hl', fun hE hx hc' ht => by subst hx; exact hc hE trivial hc' ht⟩
  · exact (HRuns.absorb (p := p) (L := L) _ (pKv v) (fun t => Flock.RecAlgebra.absorbSlice_eq _ t) c tm J).2
      st t l b st' t' l' hl h hE trivial hc ht hJ

/-- F coordinates absorbed, read by the model as `vsM`: they agree once the rest of the run fixes `vs = vsM`. -/
theorem HRuns.absorbSliceM (vs vsM : Array F128) (c : Cur) (tm : Transcript) (J : TW → Prop) :
    HRuns p L (liftT (liftTM (absorbSlice vs))) c tm c (ap (pSlice vsM) tm) (fun _ => True) (fun _ => vs = vsM) J
      fun _ => J := by
  refine ⟨fun st t l b st' t' l' hl h => ?_, fun st t l b st' t' l' hl h hE _ hc ht hJ => ?_⟩
  · obtain ⟨hq, hg, hl', hc⟩ := PRuns.absorb (p := p) (L := L) (absorbSlice vs) (pSlice vs)
      (fun t => Flock.RecAlgebra.absorbSlice_eq _ t) c tm st t l b st' t' l' hl h
    exact ⟨hq, hg, hl', fun hE hx hc' ht => by subst hx; exact hc hE trivial hc' ht⟩
  · exact (HRuns.absorb (p := p) (L := L) _ (pSlice vs) (fun t => Flock.RecAlgebra.absorbSlice_eq _ t) c tm J).2
      st t l b st' t' l' hl h hE trivial hc ht hJ

theorem HRuns.label (s : String) (c : Cur) (tm : Transcript) (J : TW → Prop) :
    HRuns p L (liftT (liftTM (Transcript.label s))) c tm c (ap (pLabel s) tm) (fun _ => True) (fun _ => True) J
      fun _ => J :=
  HRuns.absorb _ _ (fun t => Flock.RecAlgebra.label_eq s t) c tm J

/-- A refusal followed by anything never returns. -/
theorem HRuns.throwBind {α : Type} (e : String) (f : α → LP PS β) (c c' : Cur) (tm tm' : Transcript)
    (Q Xb : β → Prop) (J : TW → Prop) (J' : β → TW → Prop) :
    HRuns p L (throw e >>= f) c tm c' tm' Q Xb J J' := by
  refine ⟨fun st t l b st' t' l' _ h => ?_, fun st t l b st' t' l' _ h => ?_⟩ <;>
  · obtain ⟨_, _, _, _, h0, _⟩ := lp_bind_ok h
    cases h0

end Steps

/-! ## Level 0 -/

section Mid

variable {P : LigeritoProof} {cf : ℕ → ℕ → F128}

theorem preZ_size (P : LigeritoProof) (cf : ℕ → ℕ → F128) (sch : Schedule) (T : F128) (cap : Array ByteArray)
    (zl : Zk.Level0) (t : Transcript) : (preZ P cf sch T cap zl t).1.1.size = sch.k0 :=
  (laneRun_fields P cf sch.k0 zl.msgs _ _ sch.k0 le_rfl).2.2.2.2.2.2.1

/-- **Level 0 ends in `Mid`** at the masked model's level-loop start: from the claim `T'` after the lane rounds on the
split of the blinded fold `y₁'`, the honest prover's commitment `cm1` of that split, two out-of-domain claims on it, its
query against the masked level-0 commitment `cm0` of `tb` and its padded consistency claim (`ybar` the padding's
combination at the lane coins and the pair's coins). -/
theorem mid_lv0Z {sch : Schedule} {T : F128} {cap : Array ByteArray} {t : Transcript} {zl : Zk.Level0}
    {cm0 cm1 : Commit} {tb b : Array F128} {extra lanePad : ℕ → ℕ → F128} (hb : b.size = 2 ^ sch.n)
    (hk : sch.k0 ≤ sch.n) (hseam : LevelSeam (sch.levels.getD 0 default))
    (hlanes : (sch.levels.getD 0 default).lanes = sch.k0)
    (hcols : (sch.levels.getD 0 default).cols = sch.n - sch.k0)
    (hcm0 : cm0.rows = encodeRowsZk (sch.levels.getD 0 default) zl.tPad tb extra lanePad) (htb : tb.size = 2 ^ sch.n)
    (hrep : zl.rep < 2) (ht : zl.tPad ≤ 2 ^ (sch.levels.getD 0 default).cols)
    {C : (Array F256 × F256 × F256) × LState × Transcript}
    (hC : C = laneRun P cf sch.k0 zl.msgs (head P cf sch.n (init T) (startZ cap T t)).1
      (head P cf sch.n (init T) (startZ cap T t)).2 sch.k0)
    (hybar : zl.msgs.ybar =
      ybarOf (eqTableK C.1.1) C.1.2 zl.rep (2 ^ (sch.levels.getD 0 default).lanes) zl.tPad lanePad)
    {tm₁ : Transcript} (htm₁ : tm₁ = ap (pBytes (capBytes P.recursiveCaps[0]!)) C.2.2)
    {o₁ o₂ : LState × Transcript} (ho₁ : o₁ = ood P cf (sch.n - sch.k0 + 1) 0 C.2.1 tm₁)
    (ho₂ : o₂ = ood P cf (sch.n - sch.k0 + 1) 0 o₁.1 o₁.2)
    {Q : (Array ℕ × Array F128 × F256) × LState × Transcript}
    (hQ : Q = query P cf (sch.levels.getD 0 default) P.initialProof.rows C.1.1 o₂.1 o₂.2)
    {Yb : Array F256} (hYb : Yb = ZkTables.Level0.blinded (foldKs C.1.1.toList (embedF tb)) C.1.2
      (ZkTables.Level0.pairLane extra (2 * zl.rep) (foldKs C.1.1.toList (embedF tb)).size)
      (ZkTables.Level0.pairLane extra (2 * zl.rep + 1) (foldKs C.1.1.toList (embedF tb)).size))
    {W : Array F256} (hW : W = addScaled (embedF b) (embedF (eqTable (vec cf (startZ cap T t) sch.n)))
      (cf ((startZ cap T t).k + 1) 0))
    (hsync₁ : Sync C.2.1.s C.2.1.q (embedF (splitK Yb)) (liftU (foldKs C.1.1.toList W)) (sch.n - sch.k0))
    (hcm : cm1.rows = encodeRows (sch.levels.getD 1 default) (splitK Yb))
    (hv₁ : P.oodValues[C.2.1.ood]! = mleF (splitK Yb) (vec cf tm₁ (sch.n - sch.k0 + 1)))
    (hm₁ : P.messages[C.2.1.msg]! =
      roundPoly (embedF (splitK Yb)) (embedF (eqTable (vec cf tm₁ (sch.n - sch.k0 + 1)))))
    (hv₂ : P.oodValues[o₁.1.ood]! = mleF (splitK Yb) (vec cf o₁.2 (sch.n - sch.k0 + 1)))
    (hm₂ : P.messages[o₁.1.msg]! =
      roundPoly (embedF (splitK Yb)) (embedF (eqTable (vec cf o₁.2 (sch.n - sch.k0 + 1)))))
    (hrp : P.initialProof.rows =
      (positions (sch.levels.getD 0 default) (queryCoins P cf (sch.levels.getD 0 default) o₂.1 o₂.2)).map
        fun p => cm0.rows[p]!)
    (hm₃ : P.messages[Q.2.1.msg]! =
      roundPoly (embedF (splitK Yb)) (liftU (embedF (consW (sch.levels.getD 0 default).cols Q.1.1 Q.1.2.1)))) :
    Mid sch.n b ((preZ P cf sch T cap zl t).1.1.toList.map gOf) (x₀Z P cf sch T cap zl t) cm1
      (sch.levels.getD 1 default)
      (embedF (splitK Yb), addScaled (addScaled (addScaled (liftU (foldKs C.1.1.toList W))
        (embedF (eqTable (vec cf tm₁ (sch.n - sch.k0 + 1)))) (cf (tm₁.k + 1) 0))
        (embedF (eqTable (vec cf o₁.2 (sch.n - sch.k0 + 1)))) (cf (o₁.2.k + 1) 0))
        (liftU (embedF (consW (sch.levels.getD 0 default).cols Q.1.1 Q.1.2.1))) (cf Q.2.2.k 0)) := by
  subst htm₁ ho₁ ho₂ hQ hW
  have hf := laneRun_fields P cf sch.k0 zl.msgs (head P cf sch.n (init T) (startZ cap T t)).1
    (head P cf sch.n (init T) (startZ cap T t)).2 sch.k0 le_rfl
  rw [← hC] at hf
  have hH := head_fields P cf sch.n (init T) (startZ cap T t)
  set lv := sch.levels.getD 0 default with hlv
  have hch : C.1.1.size = lv.lanes := by rw [hf.2.2.2.2.2.2.1, hlanes]
  have hg : tb.size = 2 ^ (lv.lanes + lv.cols) := by rw [htb, hlanes, hcols]; congr 1; omega
  obtain ⟨w1, hpost⟩ := wok_lv0 (n := sch.n) (b := b) (vec cf (startZ cap T t) sch.n)
    (cf ((startZ cap T t).k + 1) 0) C.1.1 hb (vec_size _ _ _) (by rw [hf.2.2.2.2.2.2.1]; exact hk)
    (by rw [hf.2.2.2.2.2.2.2.1, hH.2.2.2.2.2.2.2]; rfl) (by rw [hf.2.2.2.2.2.2.2.2, hH.2.2.2.2.2.2.1]; rfl)
  rw [hf.2.2.2.2.2.2.1] at w1 hpost
  have s1 := sync_ood (split := 0) hsync₁ rfl hv₁ hm₁
  have w2 := wok_ood (P := P) (cf := cf) (split := 0) (tm := ap (pBytes (capBytes P.recursiveCaps[0]!)) C.2.2) w1
    hsync₁.ws rfl rfl
  have s2 := sync_ood (split := 0) s1 rfl hv₂ hm₂
  have w3 := wok_ood (P := P) (cf := cf) (split := 0) (tm := (ood P cf (sch.n - sch.k0 + 1) 0 C.2.1
    (ap (pBytes (capBytes P.recursiveCaps[0]!)) C.2.2)).2) w2 s1.ws rfl rfl
  set o₂ := ood P cf (sch.n - sch.k0 + 1) 0 (ood P cf (sch.n - sch.k0 + 1) 0 C.2.1
    (ap (pBytes (capBytes P.recursiveCaps[0]!)) C.2.2)).1 (ood P cf (sch.n - sch.k0 + 1) 0 C.2.1
    (ap (pBytes (capBytes P.recursiveCaps[0]!)) C.2.2)).2 with ho₂
  set Q := query P cf lv P.initialProof.rows C.1.1 o₂.1 o₂.2 with hQ
  have hpa : Q.1.1.size ≤ 2 ^ Q.1.2.1.size := pos_le_alpha lv hseam _ _ (vec_size _ _ _)
  have hpos : ∀ j < Q.1.1.size, Q.1.1[j]! < 2 ^ lv.d := pos_lt hseam _
  have hfs : (foldKs C.1.1.toList (embedF tb)).size = 2 ^ lv.cols := by
    rw [foldKs_size, embedF_size, hg, Array.length_toList, hch, pow_add, Nat.mul_div_cancel_left _ (by positivity)]
  have hys : Yb.size = 2 ^ (sch.n - sch.k0) := by
    rw [hYb, ZkTables.Level0.blinded, Array.size_map, Array.size_range, hfs, hcols]
  have hy : ∀ c < 2 ^ lv.cols, gv Yb c = foldLoL (C.1.1.toList.map gOf) (gv (embedF tb)) c +
      gOf C.1.2.1 * FlockLevel3.embed (fOf (extra (2 * zl.rep) c)) +
      gOf C.1.2.2 * FlockLevel3.embed (fOf (extra (2 * zl.rep + 1) c)) := by
    intro c hc
    rw [hYb, gv_blinded _ _ _ _ (by rw [hfs]; exact hc), gv_foldKs _ _ _ (by
      rw [embedF_size, hg, Array.length_toList, hch, pow_add, Nat.mul_div_cancel_left _ (by positivity)]; exact hc)]
    simp only [ZkTables.Level0.pairLane, getElem!_range_map _ (show c < (foldKs C.1.1.toList (embedF tb)).size by
      rw [hfs]; exact hc)]
  have s3 := Sync.closeZk lv zl.tPad tb extra lanePad Q.1.1 Q.1.2.1 C.1.1 zl.rep C.1.2 Yb hch hg hpos hrep ht hcols
    hys hy s2 (cf Q.2.2.k 0)
  have he : FlockSoundness.Discharge.ZkExec.consE0 P.initialProof.rows lv.q (2 ^ lv.lanes) Q.1.2.1 C.1.1 +
      FlockSoundness.Discharge.ZkExec.extraE P.initialProof.rows Q.1.1 (2 ^ lv.lanes) zl.rep lv.cols C.1.2 Q.1.2.1
        zl.msgs.ybar =
      FlockSoundness.Discharge.ZkExec.consE0 (Q.1.1.map fun p => (encodeRowsZk lv zl.tPad tb extra lanePad)[p]!)
        Q.1.1.size (2 ^ lv.lanes) Q.1.2.1 C.1.1 +
      FlockSoundness.Discharge.ZkExec.extraE (Q.1.1.map fun p => (encodeRowsZk lv zl.tPad tb extra lanePad)[p]!)
        Q.1.1 (2 ^ lv.lanes) zl.rep lv.cols C.1.2 Q.1.2.1
        (ybarOf (eqTableK C.1.1) C.1.2 zl.rep (2 ^ lv.lanes) zl.tPad lanePad) := by
    rw [hrp, hcm0, ← hybar, show Q.1.1.size = lv.q from positions_size lv hseam _]
    rfl
  set E := FlockSoundness.Discharge.ZkExec.consE0 P.initialProof.rows lv.q (2 ^ lv.lanes) Q.1.2.1 C.1.1 +
      FlockSoundness.Discharge.ZkExec.extraE P.initialProof.rows Q.1.1 (2 ^ lv.lanes) zl.rep lv.cols C.1.2 Q.1.2.1
        zl.msgs.ybar with hE
  have w4 := wok_close (P := P) (cf := cf) (cols := sch.n - sch.k0) (start := 0) (tm := Q.2.2)
    (Q.1.1, Q.1.2.1, E) (sm := Q.2.1) w3 s2.ws rfl rfl hpa
  have s4 : Sync (close P cf (sch.n - sch.k0) 0 (Q.1.1, Q.1.2.1, E) Q.2.1 Q.2.2).1.s
      (close P cf (sch.n - sch.k0) 0 (Q.1.1, Q.1.2.1, E) Q.2.1 Q.2.2).1.q (embedF (splitK Yb))
      (addScaled (addScaled (addScaled (liftU (foldKs C.1.1.toList (addScaled (embedF b)
        (embedF (eqTable (vec cf (startZ cap T t) sch.n))) (cf ((startZ cap T t).k + 1) 0))))
        (embedF (eqTable (vec cf (ap (pBytes (capBytes P.recursiveCaps[0]!)) C.2.2) (sch.n - sch.k0 + 1))))
        (cf ((ap (pBytes (capBytes P.recursiveCaps[0]!)) C.2.2).k + 1) 0))
        (embedF (eqTable (vec cf (ood P cf (sch.n - sch.k0 + 1) 0 C.2.1
          (ap (pBytes (capBytes P.recursiveCaps[0]!)) C.2.2)).2 (sch.n - sch.k0 + 1))))
        (cf ((ood P cf (sch.n - sch.k0 + 1) 0 C.2.1 (ap (pBytes (capBytes P.recursiveCaps[0]!)) C.2.2)).2.k + 1) 0))
        (liftU (embedF (consW lv.cols Q.1.1 Q.1.2.1))) (cf Q.2.2.k 0)) (sch.n - sch.k0) := by
    show Sync (Q.2.1.s + E.smul (cf Q.2.2.k 0)) (Q.2.1.q.addScaled (Poly.ofMsg P.messages[Q.2.1.msg]! E)
      (cf Q.2.2.k 0)) _ _ _
    rw [hm₃, he]
    exact s3
  unfold Mid x₀Z preZ lv0Z
  dsimp only
  rw [← hC, show sch.n - sch.k0 + 1 - 1 = sch.n - sch.k0 by omega, ← hlv, ← ho₂, ← hQ]
  refine ⟨_, _, rfl, hcm, ?_, ?_, ?_, ?_⟩
  · exact s4
  · convert w4 using 2
    rw [hcols]
  · exact preok_next _ _ _ hpost rfl rfl (by simp) (by simp) (by simp) (by simp) (by simp)
  · refine conss_push_pos (cs := C.2.1.conss) ?_ hpa
    rw [hf.2.2.2.2.2.2.2.2, hH.2.2.2.2.2.2.1]
    simp

/-- **The last level meets the final identity**: from `Mid` before the last level (the last commitment `cm` at level
`R`), the honest prover's four challenges, whose split fold is the proof's `y_r`, and its query against `cm`, whose rows
are the proof's last opening, the masked verifier's model ends with `lhsE y_r ext res = s`. -/
theorem goal_lastZ {sch : Schedule} {T : F128} {cap : Array ByteArray} {t : Transcript} {zl : Zk.Level0}
    {b : Array F128} {basisAt : Array F256 → ℕ → Array F256} {j : ℕ} {x : Vars × LState × Transcript}
    (hx : x = loop P cf sch (x₀Z P cf sch T cap zl t) j) (hjR : j + 1 = sch.levels.size - 1)
    (hL : 2 ≤ sch.levels.size) (hk0 : 1 ≤ sch.k0)
    (hsd : sch.k0 + 3 * (sch.levels.size - 1) ≤ sch.n) {cm : Commit} {g : Array F128} {w : Array F256}
    (hrows : cm.rows = encodeRows (sch.levels.getD (sch.levels.size - 1) default) g)
    (hsync : Sync x.2.1.s x.2.1.q (embedF g) w (x.1.1 - 1))
    (hwok : WOK sch.n b ((preZ P cf sch T cap zl t).1.1.toList.map gOf) x.1.2.2.1 x.2.1 (x.1.1 - 1) w)
    (hpre : PreOK ((preZ P cf sch T cap zl t).1.1.toList.map gOf) (levelsOf x.1.2.2.1) (x.2.1.oods.toList.map oodOf)
      (x.2.1.conss.toList.map consOf) (x.1.1 - 1))
    (hcs : ∀ cs ∈ x.2.1.conss.toList, cs.pos.size ≤ 2 ^ cs.alpha.size)
    (hseam : LevelSeam (sch.levels.getD (sch.levels.size - 1) default))
    (hlanes : (sch.levels.getD (sch.levels.size - 1) default).lanes = 4)
    (hcols : (sch.levels.getD (sch.levels.size - 1) default).cols = x.1.1 - 4)
    (hbasis : ∀ q e y, q.size + e = sch.n → y < 2 ^ e →
      gOf (basisAt q e)[y]! = Model.bHat execArith sch.n (bOf b) (q.toList.map gOf ++ Model.bitsK e y))
    {C : Array F256 × LState × Transcript} (hC : C = chals P cf #[] x.2.1 x.2.2 4)
    {F : Array F256} (hF : F = foldKs C.1.toList (embedF g))
    (hsum : gOf C.2.1.s = ∑ y ∈ Finset.range (2 ^ (x.1.1 - 4)), gv F y * gv (foldKs C.1.toList w) y)
    (hyr : P.finalProof.yr = splitK F)
    (hfin : P.finalProof.rows = (positions (sch.levels.getD (sch.levels.size - 1) default)
      (queryCoins P cf (sch.levels.getD (sch.levels.size - 1) default) C.2.1
        (ap (fun P' => P.finalProof.yr.foldl (fun P'' w => pF w P'') P') C.2.2))).map fun p => cm.rows[p]!) :
    lhsE P.finalProof.yr (outZ P cf sch T cap zl t).1.2.2.2 (resE basisAt sch.n (preZ P cf sch T cap zl t).1.1
      (outZ P cf sch T cap zl t).1.2.2.1 (outZ P cf sch T cap zl t).1.2.2.2 (outZ P cf sch T cap zl t).2.1) =
      (outZ P cf sch T cap zl t).2.1.s := by
  have hD : Done P sch (t.k + 2) (outZ P cf sch T cap zl t) :=
    done_of P cf sch (inv_x₀Z P cf sch T cap zl t hL hk0) hL hsd
  have hI := inv_loop_of P cf sch (inv_x₀Z P cf sch T cap zl t hL hk0) hsd j (by omega)
  rw [← hx] at hI
  have hout : outZ P cf sch T cap zl t = stepL P cf sch x := by
    unfold outZ
    rw [← hjR, loop_stepL P cf sch _ (by omega), ← hx]
  rw [hout] at hD ⊢
  subst hC hF
  have hCf := chals_fields P cf #[] x.2.1 x.2.2 4
  have h4 : 4 ≤ x.1.1 := by rw [hI.sd]; omega
  have hN : x.1.1 - 1 = x.1.1 - 4 + 3 := by omega
  rw [hN] at hsync hwok hpre
  have hch : (chals P cf #[] x.2.1 x.2.2 4).1.size = (sch.levels.getD (sch.levels.size - 1) default).lanes := by
    rw [hCf.2.2.2.2.2.2.1, hlanes]; rfl
  have hg : g.size = 2 ^ ((sch.levels.getD (sch.levels.size - 1) default).lanes +
      (sch.levels.getD (sch.levels.size - 1) default).cols) := by
    have := hsync.ts
    rw [embedF_size] at this
    rw [this, hlanes, hcols]
    congr 1
    omega
  have hw4 : w.size = 2 ^ (x.1.1 - 4 + 4) := hsync.ws
  have w0 : WOK sch.n b ((preZ P cf sch T cap zl t).1.1.toList.map gOf) x.1.2.2.1 (chals P cf #[] x.2.1 x.2.2 4).2.1
      (x.1.1 - 4 + 3) w := by
    intro i hi
    rw [hCf.2.2.2.2.2.2.2.1, hCf.2.2.2.2.2.2.2.2]
    exact hwok i hi
  have hpre' : PreOK ((preZ P cf sch T cap zl t).1.1.toList.map gOf) (levelsOf x.1.2.2.1)
      ((chals P cf #[] x.2.1 x.2.2 4).2.1.oods.toList.map oodOf)
      ((chals P cf #[] x.2.1 x.2.2 4).2.1.conss.toList.map consOf) (x.1.1 - 4 + 3) := by
    rw [hCf.2.2.2.2.2.2.2.1, hCf.2.2.2.2.2.2.2.2]
    exact hpre
  have hfold := wok_fold w0 hpre' hw4 _ (by rw [hCf.2.2.2.2.2.2.1]; rfl)
  have h16 : ∀ a : ℕ, 2 ^ (a + 4) / 2 ^ 4 = 2 ^ a := fun a => by
    rw [pow_add, Nat.mul_div_cancel _ (by positivity)]
  have hFs : (foldKs (chals P cf #[] x.2.1 x.2.2 4).1.toList (embedF g)).size = 2 ^ (x.1.1 - 4) := by
    rw [foldKs_size, embedF_size, hg, Array.length_toList, hch, hlanes, hcols, Nat.add_comm, h16]
  have hrows' : P.finalProof.rows = (positions (sch.levels.getD (sch.levels.size - 1) default)
      (queryCoins P cf (sch.levels.getD (sch.levels.size - 1) default) (chals P cf #[] x.2.1 x.2.2 4).2.1
        (ap (fun P' => P.finalProof.yr.foldl (fun P'' w => pF w P'') P') (chals P cf #[] x.2.1 x.2.2 4).2.2))).map
      fun p => (encodeRows (sch.levels.getD (sch.levels.size - 1) default) g)[p]! := by
    rw [hfin, hrows]
  have e1 : lhsE P.finalProof.yr = lhsE (splitK (foldKs (chals P cf #[] x.2.1 x.2.2 4).1.toList (embedF g))) := by
    rw [hyr]
  have hpa := pos_le_alpha (sch.levels.getD (sch.levels.size - 1) default) hseam
    (queryCoins P cf (sch.levels.getD (sch.levels.size - 1) default) (chals P cf #[] x.2.1 x.2.2 4).2.1
      (ap (fun P' => P.finalProof.yr.foldl (fun P'' w => pF w P'') P') (chals P cf #[] x.2.1 x.2.2 4).2.2))
    (query P cf (sch.levels.getD (sch.levels.size - 1) default) P.finalProof.rows (chals P cf #[] x.2.1 x.2.2 4).1
      (chals P cf #[] x.2.1 x.2.2 4).2.1
      (ap (fun P' => P.finalProof.yr.foldl (fun P'' w => pF w P'') P') (chals P cf #[] x.2.1 x.2.2 4).2.2)).1.2.1
    (vec_size _ _ _)
  have hl : ∀ l ∈ (x.1.2.2.1.push (chals P cf #[] x.2.1 x.2.2 4).1).toList, l.size = 4 := hD.lv4
  have hlvs : (x.1.2.2.1.push (chals P cf #[] x.2.1 x.2.2 4).1).size = sch.levels.size - 1 := hD.lvs
  have hext : x.1.1 - 4 = sch.n - sch.k0 - 3 * (sch.levels.size - 1) := hD.ext
  have hk0' : (preZ P cf sch T cap zl t).1.1.size = sch.k0 := preZ_size P cf sch T cap zl t
  have horig : (origE (preZ P cf sch T cap zl t).1.1 (x.1.2.2.1.push (chals P cf #[] x.2.1 x.2.2 4).1)).size +
      (x.1.1 - 4) = sch.n := by
    rw [origE_size _ _ hl, hlvs, hext, hk0']
    omega
  unfold stepL
  dsimp only
  rw [e1]
  refine lhsE_honest _ (foldKs (chals P cf #[] x.2.1 x.2.2 4).1.toList w) (embedF (consW (x.1.1 - 4)
    (positions (sch.levels.getD (sch.levels.size - 1) default)
    (queryCoins P cf (sch.levels.getD (sch.levels.size - 1) default) (chals P cf #[] x.2.1 x.2.2 4).2.1
      (ap (fun P' => P.finalProof.yr.foldl (fun P'' w => pF w P'') P') (chals P cf #[] x.2.1 x.2.2 4).2.2)))
    (query P cf (sch.levels.getD (sch.levels.size - 1) default) P.finalProof.rows (chals P cf #[] x.2.1 x.2.2 4).1
      (chals P cf #[] x.2.1 x.2.2 4).2.1
      (ap (fun P' => P.finalProof.yr.foldl (fun P'' w => pF w P'') P') (chals P cf #[] x.2.1 x.2.2 4).2.2)).1.2.1))
    _ _ _ _ _ hFs ?_ ?_ fun y hy => ?_
  · exact hsum
  · rw [show (query P cf (sch.levels.getD (sch.levels.size - 1) default) P.finalProof.rows
        (chals P cf #[] x.2.1 x.2.2 4).1 (chals P cf #[] x.2.1 x.2.2 4).2.1
        (ap (fun P' => P.finalProof.yr.foldl (fun P'' w => pF w P'') P') (chals P cf #[] x.2.1 x.2.2 4).2.2)).1.2.2 =
        consE ((query P cf (sch.levels.getD (sch.levels.size - 1) default) P.finalProof.rows
          (chals P cf #[] x.2.1 x.2.2 4).1 (chals P cf #[] x.2.1 x.2.2 4).2.1
          (ap (fun P' => P.finalProof.yr.foldl (fun P'' w => pF w P'') P')
            (chals P cf #[] x.2.1 x.2.2 4).2.2)).1.1.map fun p =>
              (encodeRows (sch.levels.getD (sch.levels.size - 1) default) g)[p]!)
          (query P cf (sch.levels.getD (sch.levels.size - 1) default) P.finalProof.rows
            (chals P cf #[] x.2.1 x.2.2 4).1 (chals P cf #[] x.2.1 x.2.2 4).2.1
            (ap (fun P' => P.finalProof.yr.foldl (fun P'' w => pF w P'') P')
              (chals P cf #[] x.2.1 x.2.2 4).2.2)).1.1.size
          (query P cf (sch.levels.getD (sch.levels.size - 1) default) P.finalProof.rows
            (chals P cf #[] x.2.1 x.2.2 4).1 (chals P cf #[] x.2.1 x.2.2 4).2.1
            (ap (fun P' => P.finalProof.yr.foldl (fun P'' w => pF w P'') P')
              (chals P cf #[] x.2.1 x.2.2 4).2.2)).1.2.1 (chals P cf #[] x.2.1 x.2.2 4).1 by
      rw [query_pos, positions_size _ hseam, ← hrows']
      rfl]
    rw [query_pos, consE_fold _ g _ _ _ hch hg (pos_lt hseam _), hcols]
  · refine res_honest basisAt sch.n (preZ P cf sch T cap zl t).1.1 _ (x.1.1 - 4)
      (closeLast P cf (x.1.1 - 4) (sch.levels.size - 1)
        (query P cf (sch.levels.getD (sch.levels.size - 1) default) P.finalProof.rows (chals P cf #[] x.2.1 x.2.2 4).1
          (chals P cf #[] x.2.1 x.2.2 4).2.1
          (ap (fun P' => P.finalProof.yr.foldl (fun P'' w => pF w P'') P') (chals P cf #[] x.2.1 x.2.2 4).2.2)).1
        (query P cf (sch.levels.getD (sch.levels.size - 1) default) P.finalProof.rows (chals P cf #[] x.2.1 x.2.2 4).1
          (chals P cf #[] x.2.1 x.2.2 4).2.1
          (ap (fun P' => P.finalProof.yr.foldl (fun P'' w => pF w P'') P') (chals P cf #[] x.2.1 x.2.2 4).2.2)).2.1
        (query P cf (sch.levels.getD (sch.levels.size - 1) default) P.finalProof.rows (chals P cf #[] x.2.1 x.2.2 4).1
          (chals P cf #[] x.2.1 x.2.2 4).2.1
          (ap (fun P' => P.finalProof.yr.foldl (fun P'' w => pF w P'') P') (chals P cf #[] x.2.1 x.2.2 4).2.2)).2.2).1
      (bOf b) hl horig ?_ ?_ ?_ (fun y hy => hbasis _ _ y horig hy) _ _ _ _ ?_ hpa _ hfold hy
    · refine conss_push_pos ?_ hpa
      rw [(query_fields ..).2.2.2.2.2.2.1, hCf.2.2.2.2.2.2.2.2]
      exact hcs
    · intro o ho
      have := hD.oods o ho
      unfold FlockSoundness.Refine.oodFixed OodDim at *
      split at this
      · rename_i hs
        rw [hs, origE_size _ _ hl, hlvs, hext, hk0']
        omega
      · rename_i s hs
        rw [hs, oodFixed_size _ hl (by rw [hlvs]; exact this.1), hlvs, hext]
        omega
    · intro cs hc
      have := hD.conss cs hc
      unfold ConsDim at this
      rw [consFixed_size _ hl, hlvs, hext]
      omega
    · rw [hlvs]
      rfl

end Mid

end FlockComplete.L0
