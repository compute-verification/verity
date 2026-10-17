import Proofs.Flock.Claims.Complete.Replay.LigeritoBasis

/-!
# Ring switching's claim is the packed table's

`RsSum` at the deployed relation (`rsSum_relation`): at every rep, the trace's packed table against the batched basis
on the cube, `Σ_y packed(y)·basisVals(y)`, is ring switching's batched value `T = Σ_k γ_k·β_k`.

* **The packed table** (`fOf_packed`): `packed(y)` packs the trace's bits `128y + v`, so the trace's bits as a witness
  (`witOf`) unpack from it (`witOf_unpack`).
* **The identity** (`T_packed`): each `β_k = Σ_u eq(r''_k, u)·t_k[u]` recombines the claim's `ŝ_k`, the model's honest
  message on the trace's bits (`switchVec_shat`), to `Σ_y packed(y)·Φ_k(eq(z_k, y))` (`recombine_shat`), and
  `basisVals` at `y` is `Σ_k γ_k·Φ_k(eq(z_k, y))` (`Lg.fOf_basisVals`), when every claim has `m − 6` coordinates.
-/

namespace FlockComplete.Replay

open Flock FlockLevel3 FlockSoundness FlockSoundness.Refine

local notation "𝔽" => FlockLevel3.GF128

namespace Rs

/-- The trace's bits as a witness of `m` variables. -/
def witOf (tr : Trace) (m : ℕ) : Model.Witness m := fun a => if tr.z[a.val]! then 1 else 0

theorem bitF_witOf (tr : Trace) (m : ℕ) (a : Fin (2 ^ m)) : Zc.bitF tr.z a.val = ((witOf tr m a).val : 𝔽) := by
  unfold Zc.bitF witOf
  cases tr.z[a.val]! <;> simp

theorem fOf_foldl_monomial (p : ℕ → Bool) : ∀ (k : ℕ) (acc : F128),
    fOf ((List.range k).foldl (fun acc v => if p v then acc + F128.monomial v else acc) acc) =
      fOf acc + ∑ v ∈ Finset.range k, if p v then fOf (F128.monomial v) else 0 := by
  intro k
  induction k with
  | zero => intro acc; simp
  | succ k ih =>
      intro acc
      rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil, Finset.sum_range_succ]
      split_ifs with hb
      · rw [fOf_add, ih, add_assoc]
      · rw [ih, add_zero]

/-- **The packed table** (§12.1): `packed(y)` packs the trace's bits `128y + v`. -/
theorem fOf_packed (tr : Trace) (m y : ℕ) (hy : y < 2 ^ (m - 7)) :
    fOf (packed tr m)[y]! = pack fun v => if tr.z[128 * y + v.val]! then 1 else 0 := by
  unfold packed
  rw [FlockSoundness.Refine.getElem!_range_map _ hy, fOf_foldl_monomial, fOf_zero, zero_add, pack_eq_sum,
    ← Fin.sum_univ_eq_sum_range (fun v => if tr.z[128 * y + v]! then fOf (F128.monomial v) else 0)]
  refine Finset.sum_congr rfl fun v _ => ?_
  cases tr.z[128 * y + v.val]! <;> simp

theorem witOf_unpack (tr : Trace) {m : ℕ} (hm : 7 ≤ m) (y : Fin (2 ^ (m - 7))) (v : Fin 128) :
    witOf tr m (addr128 hm y v) = execArith.unpack (fOf (packed tr m)[y.val]!) v := by
  show _ = unpack _ v
  rw [fOf_packed tr m y.val y.isLt, unpack_pack]
  rfl

theorem sws_get (claims : Array Claim) (cs : Coins) {k : ℕ} (hk : k < claims.size) :
    (sws claims cs)[k]! =
      { gamma := (gamma cs claims.size)[k]!, rpp := cs.rpp k, z := claims[k]!.x.extract 1 claims[k]!.x.size } := by
  rw [sws, FlockSoundness.Refine.getElem!_range_map _ hk, perClaim, pushes_get _ _ k hk]

/-- **Ring switching's batched value is the packed table's** against the batched basis on the cube:
`Σ_y packed(y)·basisVals(y) = T`, when every claim has `m − 6` coordinates. -/
theorem T_packed (tr : Trace) (claims : Array Claim) (cs : Coins) {m : ℕ} (hm : 7 ≤ m)
    (hx : ∀ c ∈ claims, c.x.size = m - 6) :
    dot (packed tr m) (basisVals (sws claims cs) (m - 7)) = T tr claims cs := by
  have hxk : ∀ k < claims.size, claims[k]!.x.size = m - 6 := fun k hk => by
    rw [getElem!_pos claims k hk]
    exact hx _ (Array.getElem_mem hk)
  have hn : (sws claims cs).size = claims.size := by simp [sws]
  have hrpp : ∀ k < (sws claims cs).size, ((sws claims cs)[k]!).rpp.size = 7 := fun k hk => by
    rw [sws_get claims cs (hn ▸ hk)]
    simp [Coins.rpp]
  have hzs : ∀ k < (sws claims cs).size, ((sws claims cs)[k]!).z.size = m - 7 := fun k hk => by
    rw [sws_get claims cs (hn ▸ hk)]
    simp only [Array.size_extract, Nat.min_self, hxk k (hn ▸ hk)]
    omega
  let f : Fin (2 ^ (m - 7)) → 𝔽 := fun y => fOf (packed tr m)[y.val]!
  have hz : ∀ y v, witOf tr m (addr128 hm y v) = execArith.unpack (f y) v := witOf_unpack tr hm
  have hk : ∀ k : Fin claims.size,
      fOf (gamma cs claims.size)[k.val]! * fOf (betas tr claims cs)[k.val]! =
        ∑ y : Fin (2 ^ (m - 7)), f y * (fOf ((sws claims cs)[k.val]!).gamma *
          Model.phi execArith (vecOf 7 ((sws claims cs)[k.val]!).rpp)
            (Model.eqAt (vecOf (m - 7) ((sws claims cs)[k.val]!).z) y.val)) := fun k => by
    have htail : (Discharge.Composed.RingSwitch.claimOf m claims[k.val]!).tail hm =
        vecOf (m - 7) (claims[k.val]!.x.extract 1 claims[k.val]!.x.size) := by
      funext j
      have := hxk k.val k.isLt
      simp only [Model.Claim.tail, Discharge.Composed.RingSwitch.claimOf, vecOf]
      rw [FlockLevel3.extract_get! _ 1 _ j (by omega) le_rfl, Nat.add_comm]
    rw [sws_get claims cs k.isLt, betas, pushes_get _ _ k.val k.isLt,
      recombine_eq _ _ (by simp [Coins.rpp]),
      switchVec_shat hm _ (hxk k.val k.isLt) tr (witOf tr m) (bitF_witOf tr m),
      recombine_shat execArith execArith_correct hm _ (witOf tr m) f hz, htail]
    refine (Finset.mul_sum _ _ _).trans (Finset.sum_congr rfl fun y _ => ?_)
    ring
  apply Discharge.Composed.Zerocheck.eq_of_of
  show fOf _ = fOf _
  rw [fOf_dot (packed tr m) _ (2 ^ (m - 7)) (by simp [packed]), T,
    fOf_dot (gamma cs claims.size) _ claims.size (by simp [gamma]), Finset.sum_congr rfl fun k _ => hk k,
    Finset.sum_comm]
  refine Finset.sum_congr rfl fun y _ => ?_
  rw [Lg.fOf_basisVals _ (m - 7) hrpp hzs y.val y.isLt]
  exact (Finset.mul_sum _ _ _).trans (Fintype.sum_equiv (finCongr hn) _ _ fun k => rfl)

end Rs

/-- **`RsSum` at the deployed relation**: at every rep, the packed table against the batched basis on the cube is
ring switching's batched value. -/
theorem rsSum_relation (H : Claims.Msg → Claims.Msg) : RsSum H Readable.relation := by
  intro o i hi r _
  obtain ⟨st, hsts, hwf, hrw, -, -⟩ := o.model
  obtain ⟨st', hsts', hspec, h22, -⟩ := o.table
  obtain rfl : st = st' := by
    rw [hsts] at hsts'
    simpa using hsts'
  have hsz : o.tabs.size = 1 := by simp [Run.tabs, FlockDeploy.tabsOf, hsts]
  obtain rfl : i = 0 := by omega
  have ht : o.tabs[0]! = Setup.ofCircuit st := by simp [Run.tabs, FlockDeploy.tabsOf, hsts]
  have hcl : o.claims 0 r = abcClaims st.m st.c.kLog (Zc.out st.m (o.zcCoins 0 r) (o.trace 0))
      (Lc.out st.circuitFold st.m (o.trace 0) (Zc.out st.m (o.zcCoins 0 r) (o.trace 0)) (o.lcCoins 0 r)) ++
        val (st.extraClaims o.sess.points) := by
    simp only [Run.claims, Run.lcOut, Run.zcOut, Run.extras, ht, Setup.ofCircuit]
  have hx := claims_xsize st hwf hrw h22 (o.trace 0) (o.zcCoins 0 r) (o.lcCoins 0 r) o.sess.points
    (o.sess_points hspec)
  rw [← hcl] at hx
  have hm : o.tabs[0]!.m = st.m := by rw [ht]; rfl
  rw [hm]
  exact Rs.T_packed (o.trace 0) (o.claims 0 r) (o.rsCoins 0 r) (by omega) hx

end FlockComplete.Replay
