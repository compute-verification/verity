import Proofs.Flock.Claims.Complete.ZkTables.Level0
import Proofs.Flock.Claims.Complete.Zk.PadAlgebra
import Proofs.Flock.Claims.Complete.Replay.LigeritoSync
import Proofs.Flock.Soundness.Discharge.ZkExec.VerdictLane

/-!
# Level 0's lane rows vanish at the honest pads

`lanes_hold`: where the masked prover's lane phase is in closed form (`Level0.LaneForm`), the two rows level 0 hands
the inner proof (`Zk.ligeritoConstraints`) vanish at its pads. Read in `K` (`ev2`), a row is the lane phase's claim
`target + β·y` through the `k_0` lane rounds, plus `ρ_0·e_0 + ρ_1·e_1 + T'`. Each masked value read at the pads is the
clear one (characteristic 2), so the lane rounds keep the sumcheck invariant `claim = Σ f_j·W_j` (`claim_inv`, from
the round lemma `step_sum`), and the closing sum `⟨W, f⟩ + ρ_0⟨W, R_0⟩ + ρ_1⟨W, R_1⟩ + ⟨W, f + ρ_0 R_0 + ρ_1 R_1⟩` is
twice a value (`lane_close`).

* `ev2_add`, `ev2_scale`, `ev2_masked`, `ev2_konst`, `ev2_lift`: `Lin2`'s operations read in `K` (`l2scale` by a `K`
  coin is multiplication by it).
* `ev2_claim0`: the first claim `T + β·y₀` is `Σ t·(b + β·eq(z))` where `T = ⟨t, b⟩`.
-/

namespace FlockComplete.Replay

open Flock FlockLevel3 FlockSoundness FlockSoundness.Refine FlockSoundness.Discharge.Composed
open ZerocheckMasked (evalL evalL_add evalL_scale evalL_masked size_masked size_add size_scale)
open Lg (gv)

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

/-! ## `Lin2` read in `K` -/

/-- A `Lin2`'s value at the pads `h`, as an element of `K`: `x₀(h) + x₁(h)·u`. -/
noncomputable def ev2 (x : Zk.Lin2) (h : Array F128) : 𝕂 :=
  embed (fOf (evalL x.1 h)) + embed (fOf (evalL x.2 h)) * u

/-- Both coordinates of a `Lin2` over `s` pads. -/
def S2 (x : Zk.Lin2) (s : ℕ) : Prop := x.1.a.size = s ∧ x.2.a.size = s

theorem evalL_konst (c : F128) (s : ℕ) (h : Array F128) : evalL (Zk.Lin.konst c s) h = c := by
  apply Zerocheck.eq_of_of
  show GF128.of (c + dot (Array.replicate s 0) h) = _
  rw [of_add, of_dot_replicate, add_zero]

theorem s2_add {x y : Zk.Lin2} {s : ℕ} (hx : S2 x s) (hy : S2 y s) : S2 (Zk.l2add x y) s :=
  ⟨by simp [Zk.l2add, hx.1, hy.1], by simp [Zk.l2add, hx.2, hy.2]⟩

theorem s2_scale {x : Zk.Lin2} {s : ℕ} (hx : S2 x s) (k : F256) : S2 (Zk.l2scale x k) s :=
  ⟨by simp [Zk.l2scale, hx.1, hx.2], by simp [Zk.l2scale, hx.1, hx.2]⟩

theorem s2_masked (v : F256) (i₀ i₁ s : ℕ) : S2 (Zk.l2masked v i₀ i₁ s) s :=
  ⟨size_masked _ _ _, size_masked _ _ _⟩

theorem s2_konst (v : F256) (s : ℕ) : S2 (Zk.l2konst v s) s :=
  ⟨by simp [Zk.l2konst, Zk.Lin.konst], by simp [Zk.l2konst, Zk.Lin.konst]⟩

theorem ev2_add {x y : Zk.Lin2} {s : ℕ} (hx : S2 x s) (hy : S2 y s) (h : Array F128) :
    ev2 (Zk.l2add x y) h = ev2 x h + ev2 y h := by
  simp only [ev2, Zk.l2add]
  rw [evalL_add _ _ _ (hx.1.trans hy.1.symm), evalL_add _ _ _ (hx.2.trans hy.2.symm), fOf_add, fOf_add, map_add,
    map_add]
  ring

theorem ev2_scale {x : Zk.Lin2} {s : ℕ} (hx : S2 x s) (k : F256) (h : Array F128) :
    ev2 (Zk.l2scale x k) h = gOf k * ev2 x h := by
  have h1 := gOf_limbs k
  have h2 := gOf_limbs (k * F256.u)
  rw [gOf_mul, gOf_u] at h2
  simp only [ev2, Zk.l2scale]
  rw [evalL_add _ _ _ (by simp [hx.1, hx.2]), evalL_add _ _ _ (by simp [hx.1, hx.2]), evalL_scale, evalL_scale,
    evalL_scale, evalL_scale, fOf_add, fOf_add, fOf_mul, fOf_mul, fOf_mul, fOf_mul]
  simp only [map_add, map_mul]
  linear_combination (-embed (fOf (evalL x.1 h))) * h1 - (embed (fOf (evalL x.2 h))) * h2

theorem ev2_masked (v : F256) {i₀ i₁ s : ℕ} (h₀ : i₀ < s) (h₁ : i₁ < s) (h : Array F128) :
    ev2 (Zk.l2masked v i₀ i₁ s) h = gOf v + gOf ⟨h[i₀]!, h[i₁]!⟩ := by
  simp only [ev2, Zk.l2masked]
  rw [evalL_masked _ _ _ h₀, evalL_masked _ _ _ h₁, gOf_limbs v, gOf_limbs ⟨h[i₀]!, h[i₁]!⟩, fOf_add, fOf_add,
    map_add, map_add]
  ring

theorem ev2_konst (v : F256) (s : ℕ) (h : Array F128) : ev2 (Zk.l2konst v s) h = gOf v := by
  simp only [ev2, Zk.l2konst]
  rw [evalL_konst, evalL_konst, gOf_limbs v]

theorem ev2_lift (x : Zk.Lin) (s : ℕ) (h : Array F128) : ev2 (x, Zk.Lin.konst 0 s) h = embed (fOf (evalL x h)) := by
  simp only [ev2]
  rw [evalL_konst, fOf_zero, map_zero, zero_mul, add_zero]

/-- **A `Lin2` that reads `0` in `K` has both coordinates `0`.** -/
theorem ev2_eq_zero {x : Zk.Lin2} {h : Array F128} (hz : ev2 x h = 0) : evalL x.1 h = 0 ∧ evalL x.2 h = 0 := by
  have e : (fun w : 𝔽 × 𝔽 => execArith.ofLimbs w.1 w.2) (fOf (evalL x.1 h), fOf (evalL x.2 h)) =
      (fun w : 𝔽 × 𝔽 => execArith.ofLimbs w.1 w.2) (fOf 0, fOf 0) := by
    show execArith.ofLimbs (fOf (evalL x.1 h)) (fOf (evalL x.2 h)) = execArith.ofLimbs (fOf 0) (fOf 0)
    rw [fOf_zero, ofLimbs_zero, FlockSoundness.Discharge.ZkExec.ofLimbs_eq]
    exact hz
  have := execArith_correct.ofLimbs_bijective.1 e
  simp only [Prod.mk.injEq] at this
  exact ⟨fOf_injective this.1, fOf_injective this.2⟩

/-- A masked value read at its own pads is the clear value. -/
theorem ev2_unmask (m : F256) {i₀ i₁ s : ℕ} (h₀ : i₀ < s) (h₁ : i₁ < s) (h : Array F128) :
    ev2 (Zk.l2masked (m + ⟨h[i₀]!, h[i₁]!⟩) i₀ i₁ s) h = gOf m := by
  rw [ev2_masked _ h₀ h₁, gOf_add]
  linear_combination (gOf ⟨h[i₀]!, h[i₁]!⟩) * (CharTwo.two_eq_zero : (2 : 𝕂) = 0)

/-! ## The executable's lane rows -/

open FlockSoundness.Discharge.ZkExec (laneStepE laneLhs)

/-- **Level 0's two lane rows**, exactly: the coordinates of `laneLhs`. -/
theorem lanes_eq {lay : Zk.PadLayout} {rec : Zk.L0Rec} {lg : LigeritoProof} {msgs : Zk.Level0Msgs}
    {v : Array (String × Zk.Lin)} (h : Zk.ligeritoConstraints lay rec lg msgs = .ok v) (h1 : lay.nOod0 = 1) :
    v = #[("ligerito lane phase (c0)", (laneLhs lay rec lg msgs).1),
      ("ligerito lane phase (c1)", (laneLhs lay rec lg msgs).2)] := by
  unfold Zk.ligeritoConstraints at h
  dsimp only at h
  split at h
  · cases h
  · rename_i hc
    simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, not_not, decide_eq_true_eq, not_lt] at hc
    obtain ⟨⟨ho, -⟩, -⟩ := hc
    rw [h1] at ho
    simp only [Std.Legacy.Range.forIn_eq_forIn_range', range_size, List.forIn_pure_yield_eq_foldl, ho] at h
    cases h
    rfl

theorem s2_laneStep (lay : Zk.PadLayout) (lg : LigeritoProof) (ch : Array F256) {x : Zk.Lin2} (hx : S2 x lay.len)
    (j : ℕ) : S2 (laneStepE lay lg ch x j) lay.len :=
  s2_add (s2_add (s2_masked _ _ _ _) (s2_scale (s2_add hx (s2_masked _ _ _ _)) _)) (s2_scale (s2_masked _ _ _ _) _)

/-- **A lane round read in `K`**: `u₀ + r·(claim + u₂) + r²·u₂` on the masked message's values. -/
theorem ev2_laneStep (lay : Zk.PadLayout) (lg : LigeritoProof) (ch : Array F256) {x : Zk.Lin2} (hx : S2 x lay.len)
    (j : ℕ) (h : Array F128) :
    ev2 (laneStepE lay lg ch x j) h =
      ev2 (Zk.l2masked lg.messages[j]!.1 (lay.lgMsg j 0) (lay.lgMsg j 1) lay.len) h +
        gOf ch[j]! * (ev2 x h + ev2 (Zk.l2masked lg.messages[j]!.2 (lay.lgMsg j 2) (lay.lgMsg j 3) lay.len) h) +
        gOf ch[j]! * gOf ch[j]! * ev2 (Zk.l2masked lg.messages[j]!.2 (lay.lgMsg j 2) (lay.lgMsg j 3) lay.len) h := by
  unfold laneStepE
  rw [ev2_add (s2_add (s2_masked _ _ _ _) (s2_scale (s2_add hx (s2_masked _ _ _ _)) _))
      (s2_scale (s2_masked _ _ _ _) _),
    ev2_add (s2_masked _ _ _ _) (s2_scale (s2_add hx (s2_masked _ _ _ _)) _), ev2_scale (s2_add hx (s2_masked _ _ _ _)),
    ev2_add hx (s2_masked _ _ _ _), ev2_scale (s2_masked _ _ _ _), gOf_mul]

/-! ## The lane rounds keep the sumcheck invariant -/

/-- **The round lemma at a `K` claim**: on the message `roundPoly t w` and the claim `c = Σ t·w`, the lane step at
`r` is the sum of the tables folded at `r`. -/
theorem step_sum (t w : Array F256) (c : 𝕂) (n : ℕ) (ht : t.size = 2 ^ (n + 1)) (hw : w.size = t.size)
    (hs : c = ∑ x ∈ Finset.range (2 ^ (n + 1)), gv t x * gv w x) (r : F256) :
    gOf (roundPoly t w).1 + gOf r * (c + gOf (roundPoly t w).2) + gOf r * gOf r * gOf (roundPoly t w).2 =
      ∑ y ∈ Finset.range (2 ^ n), gv (foldK r t) y * gv (foldK r w) y := by
  obtain ⟨h1, h2⟩ := Lg.gOf_roundPoly t w
  have hh : t.size / 2 = 2 ^ n := by rw [ht, pow_succ]; omega
  rw [hh] at h1 h2
  rw [sum_range_pairs] at hs
  have e : ∀ y ∈ Finset.range (2 ^ n), gv (foldK r t) y * gv (foldK r w) y =
      FlockSoundness.foldLo (gOf r) (gv t) y * FlockSoundness.foldLo (gOf r) (gv w) y := by
    intro y hy
    rw [Lg.gv_foldK _ _ (by rw [hh]; exact Finset.mem_range.1 hy),
      Lg.gv_foldK _ _ (by rw [hw, hh]; exact Finset.mem_range.1 hy)]
  rw [Finset.sum_congr rfl e, h1, h2, hs]
  simp only [FlockSoundness.foldLo]
  have := Lg.quad_sum (CharTwo.two_eq_zero : (2 : 𝕂) = 0) (2 ^ n) (fun k => gv t (2 * k)) (fun k => gv t (2 * k + 1))
    (fun k => gv w (2 * k)) (fun k => gv w (2 * k + 1)) (gOf r)
  linear_combination this

theorem foldKs_take_succ (ch : List F256) (a : Array F256) {j : ℕ} (hj : j < ch.length) :
    Lg.foldKs (ch.take (j + 1)) a = foldK ch[j] (Lg.foldKs (ch.take j) a) := by
  rw [List.take_add_one, List.getElem?_eq_getElem hj, Option.toList_some, Lg.foldKs, List.foldl_append]
  rfl

theorem foldKs_size_pow (ch : List F256) (a : Array F256) {n j : ℕ} (ha : a.size = 2 ^ n) (hj : j ≤ ch.length)
    (hjn : j ≤ n) : (Lg.foldKs (ch.take j) a).size = 2 ^ (n - j) := by
  rw [Lg.foldKs_size, ha, List.length_take, Nat.min_eq_left hj, Nat.pow_div hjn (by norm_num)]

theorem laneCh_length (resp : ℕ → Claims.Msg) (k0 L : ℕ) : (ZkTables.Level0.laneCh resp k0 L).length = k0 := by
  simp [ZkTables.Level0.laneCh]

theorem laneW0_size (resp : ℕ → Claims.Msg) (n L : ℕ) (b : Array F128) :
    (ZkTables.Level0.laneW0 resp n L b).size = b.size := by
  simp [ZkTables.Level0.laneW0, Lg.addScaled_size, Lg.embedF_size]

theorem toArray_get! (ch : List F256) {j : ℕ} (hj : j < ch.length) : ch.toArray[j]! = ch[j] := by
  rw [getElem!_pos ch.toArray j (by simpa using hj), List.getElem_toArray]

theorem lgMsg_lt (lay : Zk.PadLayout) {j t : ℕ} (hj : j < lay.k0) (ht : t < 4) : lay.lgMsg j t < lay.len := by
  simp only [Zk.PadLayout.lgMsg, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.hab]
  omega

theorem lgE_lt (lay : Zk.PadLayout) {i c : ℕ} (hi : i < 2) (hc : c < 2) : lay.lgE i c < lay.len := by
  simp only [Zk.PadLayout.lgE, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.hab]
  omega

theorem lgOod_lt (lay : Zk.PadLayout) (h1 : lay.nOod0 = 1) : lay.lgOod 0 < lay.len := by
  simp only [Zk.PadLayout.lgOod, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.hab]
  omega

/-! ## Level 0's first claim and its closing equation -/

/-- **The lane phase's first claim** `T + β·y₀` read at the pads: the out-of-domain value is `mle(t, z)` masked by its
pad, and `T = ⟨t, b⟩`, so the claim is `Σ t·(b + β·eq(z))`. -/
theorem ev2_claim0 {n : ℕ} {T β o : F128} {tb b z : Array F128} {lay : Zk.PadLayout} {hv : Array F128}
    (hn : n ≠ 0) (htb : tb.size = 2 ^ n) (hb : b.size = 2 ^ n) (hz : z.size = n) (hT : dot tb b = T)
    (hi : lay.lgOod 0 < lay.len) (ho : o = mleF tb z + hv[lay.lgOod 0]!) :
    ev2 ((Zk.Lin.konst T lay.len).add ((Zk.Lin.masked o (lay.lgOod 0) lay.len).scale β), Zk.Lin.konst 0 lay.len)
        hv =
      ∑ y ∈ Finset.range (2 ^ n), gv (embedF tb) y * gv (addScaled (embedF b) (embedF (eqTable z)) β) y := by
  have hn1 : n - 1 + 1 = n := by omega
  have hsum : FlockLevel3.embed (fOf T) + gOf (F256.ofF (mleF tb z)) * FlockLevel3.embed (fOf β) =
      ∑ y ∈ Finset.range (2 ^ n), gv (embedF tb) y * gv (addScaled (embedF b) (embedF (eqTable z)) β) y := by
    rw [Lg.gOf_mleF tb _ (n - 1) (by rw [htb, hn1]) (by rw [hz, hn1]), hn1, ← hT, Lg.fOf_dot_rng tb b (2 ^ n) htb,
      map_sum]
    have key : ∑ x ∈ Finset.range (2 ^ n), FlockLevel3.embed (fOf tb[x]! * fOf b[x]!) =
        ∑ x ∈ Finset.range (2 ^ n), gv (embedF tb) x * gv (embedF b) x := by
      refine Finset.sum_congr rfl fun x hx => ?_
      have hx' := Finset.mem_range.1 hx
      rw [Lg.gv_embedF _ (by omega), Lg.gv_embedF _ (by omega), map_mul]
    rw [key, Lg.sum_addScaled]
    refine Finset.sum_congr rfl fun x hx => ?_
    rw [Lg.gv_addScaled _ _ _ (by rw [Lg.embedF_size, hb]; exact Finset.mem_range.1 hx)]
  rw [ev2_lift, evalL_add _ _ _ (by simp [Zk.Lin.konst]), evalL_konst, evalL_scale, evalL_masked _ _ _ hi, ho, ← hsum,
    gOf_ofF]
  simp only [fOf_add, fOf_mul, map_add, map_mul]
  linear_combination (FlockLevel3.embed (fOf hv[lay.lgOod 0]!) * FlockLevel3.embed (fOf β)) *
    (CharTwo.two_eq_zero : (2 : 𝕂) = 0)

theorem gOf_dotK (a w : Array F256) : gOf (dotK a w) = ∑ i ∈ Finset.range a.size, gv a i * gv w i := by
  unfold dotK
  rw [gOf_foldl_add (fun i => a[i]! * w[i]!), gOf_zero, zero_add, FlockLevel3.list_sum_range]
  simp only [gOf_mul, gv]

theorem gv_blinded (f : Array F256) (rho : F256 × F256) (r0 r1 : Array F128) {c : ℕ} (hc : c < f.size) :
    gv (ZkTables.Level0.blinded f rho r0 r1) c =
      gv f c + gOf rho.1 * FlockLevel3.embed (fOf r0[c]!) + gOf rho.2 * FlockLevel3.embed (fOf r1[c]!) := by
  rw [gv, ZkTables.Level0.blinded, getElem!_range_map _ hc, gOf_add, gOf_add, gOf_smul, gOf_smul]
  rfl

theorem pairLane_size (extra : ℕ → ℕ → F128) (e s : ℕ) : (ZkTables.Level0.pairLane extra e s).size = s := by
  simp [ZkTables.Level0.pairLane]

theorem close_sum {K : Type*} [CommRing K] (h2 : (2 : K) = 0) (N : ℕ) (f W r0 r1 : ℕ → K) (a b : K) :
    ∑ y ∈ Finset.range N, f y * W y + a * ∑ y ∈ Finset.range N, W y * r0 y + b * ∑ y ∈ Finset.range N, W y * r1 y +
      ∑ y ∈ Finset.range N, W y * (f y + a * r0 y + b * r1 y) = 0 := by
  rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib, ← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
  exact Finset.sum_eq_zero fun y _ => by linear_combination (f y * W y + a * W y * r0 y + b * W y * r1 y) * h2

/-- **Level 0's closing equation**: `⟨f, W⟩ + ρ₀⟨W, R₀⟩ + ρ₁⟨W, R₁⟩ + ⟨W, f + ρ₀R₀ + ρ₁R₁⟩` is twice a value, so `0`. -/
theorem lane_close (f W : Array F256) (rho : F256 × F256) (r0 r1 : Array F128) (hW : W.size = f.size)
    (h0 : r0.size = f.size) (h1 : r1.size = f.size) :
    ∑ y ∈ Finset.range f.size, gv f y * gv W y + gOf rho.1 * gOf (dotK W (embedF r0)) +
      gOf rho.2 * gOf (dotK W (embedF r1)) + gOf (dotK W (ZkTables.Level0.blinded f rho r0 r1)) = 0 := by
  rw [gOf_dotK, gOf_dotK, gOf_dotK, hW]
  have e0 : ∀ y ∈ Finset.range f.size, gv W y * gv (embedF r0) y = gv W y * FlockLevel3.embed (fOf r0[y]!) :=
    fun y hy => by rw [Lg.gv_embedF _ (by have := Finset.mem_range.1 hy; omega)]
  have e1 : ∀ y ∈ Finset.range f.size, gv W y * gv (embedF r1) y = gv W y * FlockLevel3.embed (fOf r1[y]!) :=
    fun y hy => by rw [Lg.gv_embedF _ (by have := Finset.mem_range.1 hy; omega)]
  have eb : ∀ y ∈ Finset.range f.size, gv W y * gv (ZkTables.Level0.blinded f rho r0 r1) y =
      gv W y * (gv f y + gOf rho.1 * FlockLevel3.embed (fOf r0[y]!) + gOf rho.2 * FlockLevel3.embed (fOf r1[y]!)) :=
    fun y hy => by rw [gv_blinded _ _ _ _ (Finset.mem_range.1 hy)]
  rw [Finset.sum_congr rfl e0, Finset.sum_congr rfl e1, Finset.sum_congr rfl eb]
  exact close_sum (CharTwo.two_eq_zero : (2 : 𝕂) = 0) f.size (gv f) (gv W) (fun y => FlockLevel3.embed (fOf r0[y]!))
    (fun y => FlockLevel3.embed (fOf r1[y]!)) (gOf rho.1) (gOf rho.2)

section Lanes

open ZkTables.Level0

variable {resp : ℕ → Claims.Msg} {sch : Schedule} {T : F128} {tb b : Array F128} {lay : Zk.PadLayout}
  {hv : Array F128} {rep : ℕ} {extra : ℕ → ℕ → F128} {L : ℕ} {P : LigeritoProof} {zl : Zk.Level0Msgs}
  {rc : Zk.L0Rec}

/-- **The lane rounds keep the sumcheck invariant**: after `j` lane rounds from a claim that is `Σ t·W_0`, the claim
is `Σ f_j·W_j`, on the table and weights folded at the first `j` lane coins. -/
theorem claim_inv (hLF : LaneForm resp sch T tb b lay hv rep extra L P zl rc) (hk : lay.k0 = sch.k0)
    (htb : tb.size = 2 ^ sch.n) (hb : b.size = 2 ^ sch.n) (hkn : sch.k0 ≤ sch.n) (x₀ : Zk.Lin2)
    (hx₀ : S2 x₀ lay.len)
    (h₀ : ev2 x₀ hv = ∑ y ∈ Finset.range (2 ^ sch.n), gv (embedF tb) y * gv (laneW0 resp sch.n L b) y) :
    ∀ j ≤ sch.k0, S2 ((List.range' 0 j).foldl (laneStepE lay P rc.challenges) x₀) lay.len ∧
      ev2 ((List.range' 0 j).foldl (laneStepE lay P rc.challenges) x₀) hv =
        ∑ y ∈ Finset.range (2 ^ (sch.n - j)), gv (Lg.foldKs ((laneCh resp sch.k0 L).take j) (embedF tb)) y *
          gv (Lg.foldKs ((laneCh resp sch.k0 L).take j) (laneW0 resp sch.n L b)) y
  := by
  intro j
  induction j with
  | zero =>
    intro _
    simp only [List.range'_zero, List.foldl_nil, List.take_zero, Nat.sub_zero]
    exact ⟨hx₀, h₀⟩
  | succ j ih =>
    intro hj
    obtain ⟨hS, hE⟩ := ih (by omega)
    have hlen : (laneCh resp sch.k0 L).length = sch.k0 := laneCh_length resp sch.k0 L
    have hjl : j < (laneCh resp sch.k0 L).length := by omega
    have hrc : rc.challenges = (laneCh resp sch.k0 L).toArray := congrArg Zk.L0Rec.challenges hLF.record
    rw [List.range'_concat, List.foldl_append, List.foldl_cons, List.foldl_nil, Nat.zero_add, Nat.one_mul]
    refine ⟨s2_laneStep _ _ _ hS j, ?_⟩
    have hjk : j < lay.k0 := by omega
    rw [ev2_laneStep _ _ _ hS, hLF.msgs j (by omega)]
    simp only [maskMsg]
    rw [ev2_unmask _ (lgMsg_lt lay hjk (by norm_num)) (lgMsg_lt lay hjk (by norm_num)),
      ev2_unmask _ (lgMsg_lt lay hjk (by norm_num)) (lgMsg_lt lay hjk (by norm_num)), hrc, toArray_get! _ hjl,
      foldKs_take_succ _ _ hjl, foldKs_take_succ _ _ hjl]
    have hfs := foldKs_size_pow _ (embedF tb) (by rw [Lg.embedF_size, htb]) hjl.le (by omega)
    have hws := foldKs_size_pow _ (laneW0 resp sch.n L b) (by rw [laneW0_size, hb]) hjl.le (by omega)
    have e1 : sch.n - j = sch.n - (j + 1) + 1 := by omega
    rw [e1] at hfs hws hE
    rw [hrc] at hE
    exact step_sum _ _ _ (sch.n - (j + 1)) hfs (hws.trans hfs.symm) hE _

/-- **Level 0's two lane rows vanish at the honest pads**, where the lane phase is in closed form. -/
theorem lanes_hold (hLF : LaneForm resp sch T tb b lay hv rep extra L P zl rc) (h1 : lay.nOod0 = 1)
    (hk : lay.k0 = sch.k0) (htb : tb.size = 2 ^ sch.n) (hb : b.size = 2 ^ sch.n) (hkn : sch.k0 ≤ sch.n)
    (hn : sch.n ≠ 0) (hT : dot tb b = T) (hsz : hv.size = lay.len) {lgc : Array (String × Zk.Lin)}
    (hc : Zk.ligeritoConstraints lay rc P zl = .ok lgc) : ConsHold lgc hv := by
  have hT' : rc.target = T := congrArg Zk.L0Rec.target hLF.record
  have hβ : rc.ood[0]! = (coinOf (resp (L + 1)) 0, P.oodValues[0]!) := by rw [hLF.record]; rfl
  have hρ : rc.rho = laneRho resp sch.k0 L := congrArg Zk.L0Rec.rho hLF.record
  have hi := lgOod_lt lay h1
  set x₀ : Zk.Lin2 := ((Zk.Lin.konst rc.target lay.len).add
    ((Zk.Lin.masked P.oodValues[0]! (lay.lgOod 0) lay.len).scale rc.ood[0]!.1), Zk.Lin.konst 0 lay.len) with hx₀
  have hS0 : S2 x₀ lay.len := ⟨by simp [x₀, Zk.Lin.konst], by simp [x₀, Zk.Lin.konst]⟩
  have hE0 : ev2 x₀ hv =
      ∑ y ∈ Finset.range (2 ^ sch.n), gv (embedF tb) y * gv (laneW0 resp sch.n L b) y := by
    rw [hx₀, hT', hβ]
    exact ev2_claim0 hn htb hb (by simp [lanePoint]) hT hi hLF.ood
  obtain ⟨hSk, hEk⟩ := claim_inv hLF hk htb hb hkn x₀ hS0 hE0 sch.k0 le_rfl
  have hlen := laneCh_length resp sch.k0 L
  rw [List.take_of_length_le hlen.le] at hEk
  set f := Lg.foldKs (laneCh resp sch.k0 L) (embedF tb)
  set W := Lg.foldKs (laneCh resp sch.k0 L) (laneW0 resp sch.n L b)
  have hfs : f.size = 2 ^ (sch.n - sch.k0) := by
    have := foldKs_size_pow (laneCh resp sch.k0 L) (embedF tb) (by rw [Lg.embedF_size, htb]) hlen.ge hkn
    rwa [List.take_of_length_le hlen.le] at this
  have hWs : W.size = 2 ^ (sch.n - sch.k0) := by
    have := foldKs_size_pow (laneCh resp sch.k0 L) (laneW0 resp sch.n L b) (by rw [laneW0_size, hb]) hlen.ge hkn
    rwa [List.take_of_length_le hlen.le] at this
  have hcl : (List.range' 0 lay.k0).foldl (laneStepE lay P rc.challenges) x₀ =
      (List.range' 0 sch.k0).foldl (laneStepE lay P rc.challenges) x₀ := by rw [hk]
  have hX : laneLhs lay rc P zl = Zk.l2add (Zk.l2add ((List.range' 0 sch.k0).foldl (laneStepE lay P rc.challenges) x₀)
      (Zk.l2scale (Zk.l2masked zl.e0 (lay.lgE 0 0) (lay.lgE 0 1) lay.len) rc.rho.1))
      (Zk.l2add (Zk.l2scale (Zk.l2masked zl.e1 (lay.lgE 1 0) (lay.lgE 1 1) lay.len) rc.rho.2)
        (Zk.l2konst zl.tPrime lay.len)) := by
    rw [← hcl]; rfl
  have hSE0 := s2_scale (s2_masked zl.e0 (lay.lgE 0 0) (lay.lgE 0 1) lay.len) rc.rho.1
  have hSE1 := s2_scale (s2_masked zl.e1 (lay.lgE 1 0) (lay.lgE 1 1) lay.len) rc.rho.2
  have hSX : S2 (laneLhs lay rc P zl) lay.len := by
    rw [hX]; exact s2_add (s2_add hSk hSE0) (s2_add hSE1 (s2_konst _ _))
  have hEX : ev2 (laneLhs lay rc P zl) hv = 0 := by
    rw [hX, ev2_add (s2_add hSk hSE0) (s2_add hSE1 (s2_konst _ _)), ev2_add hSk hSE0, ev2_add hSE1 (s2_konst _ _),
      ev2_scale (s2_masked _ _ _ _), ev2_scale (s2_masked _ _ _ _), ev2_konst, hEk, hLF.e0, hLF.e1,
      ev2_unmask _ (lgE_lt lay (by norm_num) (by norm_num)) (lgE_lt lay (by norm_num) (by norm_num)),
      ev2_unmask _ (lgE_lt lay (by norm_num) (by norm_num)) (lgE_lt lay (by norm_num) (by norm_num)), hLF.tPrime, hρ,
      ← hfs]
    have := lane_close f W (laneRho resp sch.k0 L) (ZkTables.Level0.pairLane extra (2 * rep) f.size)
      (ZkTables.Level0.pairLane extra (2 * rep + 1) f.size) (hWs.trans hfs.symm) (pairLane_size _ _ _)
      (pairLane_size _ _ _)
    rw [← this]
    ring
  obtain ⟨hz0, hz1⟩ := ev2_eq_zero hEX
  rw [lanes_eq hc h1]
  intro q hq
  simp only [List.mem_toArray, List.mem_cons, List.not_mem_nil, or_false] at hq
  rcases hq with rfl | rfl
  · exact ⟨hSX.1.trans hsz.symm, hz0⟩
  · exact ⟨hSX.2.trans hsz.symm, hz1⟩

end Lanes

end FlockComplete.Replay
