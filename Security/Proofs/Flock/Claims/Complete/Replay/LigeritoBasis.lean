import Proofs.Flock.Claims.Complete.Replay.LigeritoFinal
import Proofs.Flock.Claims.Complete.Replay.RingSwitchAlgebra

/-!
# The verifier's basis is the batched basis's extension

`BasisOk` at the deployed relation (`basisOk_relation`): at every rep, the verifier's `basis sws` at an original point
`q ‖ bits(y)` is the multilinear extension of the prover's batched basis on the cube (`basisVals sws`) there.

* **On the cube** (`fOf_basisVals`): `basisVals sws` at `w` is `Σ_k γ_k·Φ_k(eq(z_k, w))`, the model's batched basis
  (R4's `OpenRel.b`), when every claim has seven ring-switch coins and `z_k` has `ℓ` coordinates.
* **The extension** (`basis_bHat`): `basis_eq` at that batched basis.
* **The rep's claims** (`claims_xsize`): every claim a rep opens has `m − 6` coordinates, so its switched claim's
  suffix point has `m − 7`.
-/

namespace FlockComplete.Replay

open Flock FlockSoundness FlockSoundness.Refine

local notation "𝔽" => FlockLevel3.GF128

namespace Lg

theorem fOf_foldl_acc {β : Type} (f : β → F128) : ∀ (l : List β) (x : F128),
    fOf (l.foldl (fun x b => x + f b) x) = fOf x + (l.map fun b => fOf (f b)).sum
  | [], x => by simp
  | b :: l, x => by rw [List.foldl_cons, fOf_foldl_acc f l, fOf_add, List.map_cons, List.sum_cons, add_assoc]

theorem list_sum_fin {α : Type} [Inhabited α] (a : Array α) (h : α → 𝔽) :
    (a.toList.map h).sum = ∑ k : Fin a.size, h a[k.val]! := by
  rw [Fin.sum_univ_eq_sum_range (fun k => h a[k]!), ← FlockLevel3.list_sum_range]
  congr 1
  refine List.ext_getElem (by simp) fun i h₁ _ => ?_
  simp only [List.getElem_map, Array.getElem_toList, List.getElem_range]
  rw [getElem!_pos a i (by simpa using h₁)]

/-- **One switched claim's term on the cube** is the model's `Φ(eq(z, w))`. -/
theorem fOf_basisTerm (sw : Switched) (l : ℕ) (hrpp : sw.rpp.size = 7) (hz : sw.z.size = l) (w : ℕ)
    (hw : w < 2 ^ l) :
    fOf (sumF 128 fun u => if ((eqTable sw.z)[w]!).bit u then (eqTable sw.rpp)[u]! else 0) =
      Model.phi execArith (vecOf 7 sw.rpp) (Model.eqAt (vecOf l sw.z) w) := by
  show FlockLevel3.GF128.of _ = _
  rw [Zc.of_sumF, ← eqTable_vecOf sw.z l hz w hw]
  unfold Model.phi
  rw [← Fin.sum_univ_eq_sum_range
    (fun u => fOf (if ((eqTable sw.z)[w]!).bit u then (eqTable sw.rpp)[u]! else 0)) 128]
  refine Finset.sum_congr rfl fun u _ => ?_
  show _ = ((FlockLevel3.unpack (fOf (eqTable sw.z)[w]!) u).val : 𝔽) * _
  rw [unpack_fOf, ← eqTable_vecOf sw.rpp 7 hrpp u.val (by simp)]
  split_ifs <;> simp [fOf_zero]

/-- **The batched basis on the cube** (§12.4) is the model's: `basisVals sws` at `w < 2^ℓ` is
`Σ_k γ_k·Φ_k(eq(z_k, w))`. -/
theorem fOf_basisVals (sws : Array Switched) (l : ℕ) (hrpp : ∀ k < sws.size, (sws[k]!).rpp.size = 7)
    (hz : ∀ k < sws.size, (sws[k]!).z.size = l) (w : ℕ) (hw : w < 2 ^ l) :
    fOf (basisVals sws l)[w]! = ∑ k : Fin sws.size, fOf (sws[k.val]!).gamma *
      Model.phi execArith (vecOf 7 (sws[k.val]!).rpp) (Model.eqAt (vecOf l (sws[k.val]!).z) w) := by
  unfold basisVals
  rw [FlockSoundness.Refine.getElem!_range_map _ hw, Array.foldl_map, ← Array.foldl_toList]
  dsimp only
  rw [fOf_foldl_acc, fOf_zero, zero_add, list_sum_fin]
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [fOf_mul, fOf_basisTerm _ l (hrpp k k.isLt) (hz k k.isLt) w hw]

/-- **The verifier's basis is the batched basis's extension**: `basis sws` at `q ‖ bits(y)` is
`Σ_w basisVals(w)·eq(q ‖ bits(y), w)`, when every switched claim has seven ring-switch coins and `m − 7` suffix
coordinates. -/
theorem basis_bHat (sws : Array Switched) (m : ℕ) (hrpp : ∀ k < sws.size, (sws[k]!).rpp.size = 7)
    (hz : ∀ k < sws.size, (sws[k]!).z.size = m - 7) (q : Array F256) (e y : ℕ) (hqe : q.size + e = m - 7)
    (hy : y < 2 ^ e) :
    gOf (basis sws q e)[y]! = Model.bHat execArith (m - 7) (fun w => fOf (basisVals sws (m - 7))[w]!)
      (q.toList.map gOf ++ Model.bitsK e y) := by
  rw [basis_eq (T := 0) (op := ⟨0, fun w => ∑ k : Fin sws.size, fOf (sws[k.val]!).gamma *
      Model.phi execArith (vecOf 7 (sws[k.val]!).rpp) (Model.eqAt (vecOf (m - 7) (sws[k.val]!).z) w)⟩)
    ⟨fOf_zero.symm, rfl, hrpp, hz, fun _ => rfl⟩ q e hqe y hy]
  unfold Model.bHat
  refine Finset.sum_congr rfl fun w hw => ?_
  dsimp only
  rw [fOf_basisVals sws (m - 7) hrpp hz w (Finset.mem_range.1 hw)]

/-- **At a rep's switched claims**: each has seven ring-switch coins, and `m − 7` suffix coordinates when its claim
has `m − 6`. -/
theorem basis_sws (claims : Array Claim) (cs : Rs.Challenges) (m : ℕ) (hx : ∀ c ∈ claims, c.x.size = m - 6)
    (q : Array F256) (e y : ℕ) (hqe : q.size + e = m - 7) (hy : y < 2 ^ e) :
    gOf (basis (Rs.sws claims cs) q e)[y]! = Model.bHat execArith (m - 7)
      (fun w => fOf (basisVals (Rs.sws claims cs) (m - 7))[w]!) (q.toList.map gOf ++ Model.bitsK e y) := by
  have hsw : ∀ k < (Rs.sws claims cs).size, (Rs.sws claims cs)[k]! =
      { gamma := (Rs.gamma cs claims.size)[k]!, rpp := cs.rpp k,
        z := claims[k]!.x.extract 1 claims[k]!.x.size } := by
    intro k hk
    have hk' : k < claims.size := by simpa [Rs.sws] using hk
    rw [Rs.sws, FlockSoundness.Refine.getElem!_range_map _ hk', Rs.perClaim, pushes_get _ _ k hk']
  refine basis_bHat _ m (fun k hk => ?_) (fun k hk => ?_) q e y hqe hy
  · rw [hsw k hk]
    simp [Rs.Challenges.rpp]
  · rw [hsw k hk]
    have hk' : k < claims.size := by simpa [Rs.sws] using hk
    have := hx _ (Array.getElem_mem hk')
    rw [← getElem!_pos claims k hk'] at this
    simp only [Array.size_extract, Nat.min_self, this]
    omega

end Lg

/-- **Every claim a rep opens has `m − 6` coordinates**, for a statement's table, at link points of `m_pts`
coordinates. -/
theorem claims_xsize (st : Stmt) (hwf : StmtWF st) (hrw : RegionsWF st) (hm : 22 ≤ st.m) (tr : Trace)
    (zcs : Zc.Challenges) (cs : Lc.Challenges) (points : Array (Array F128)) (hpts : ∀ p ∈ points.toList, p.size = st.mPts) :
    ∀ c ∈ abcClaims st.m st.c.kLog (Zc.out st.m zcs tr) (Lc.out st.circuitFold st.m tr (Zc.out st.m zcs tr) cs) ++
        val (st.extraClaims points), c.x.size = st.m - 6 := by
  have hk := hwf.hk
  intro c hc
  rcases Array.mem_append.1 hc with hc | hc
  · simp only [abcClaims, List.mem_toArray, List.mem_cons, List.not_mem_nil, or_false] at hc
    rcases hc with rfl | rfl
    · show ((pushes cs.τ (st.c.kLog - 6)).reverse ++
        (pushes zcs.ρ (st.m - 6)).extract (st.c.kLog - 6) (st.m - 6)).size = st.m - 6
      rw [Array.size_append, Array.size_reverse, pushes_size, Array.size_extract, pushes_size, Nat.min_self]
      omega
    · exact Zc.r_size st.m (by omega) zcs
  · cases hex : st.extraClaims points with
    | error e =>
      rw [hex, show val (Except.error e : Except String (Array Claim)) = #[] from rfl] at hc
      simp at hc
    | ok extras =>
      rw [hex] at hc
      obtain ⟨ess, hess, heq⟩ := extraClaims_ok st points extras hex
      have hc' : c ∈ ess.flatten := by
        rw [← heq]
        exact Array.mem_def.1 hc
      obtain ⟨reg, hreg, r, hr, v, hv, rfl⟩ := mem_of_forall₂ hess hc'
      exact (regionClaim_rel' st hwf (regionsOf st) reg (hrw.regions reg hreg) hrw.hm st.mPts hrw.mPts r
        (hpts r hr) (fun t => fOf r[t.val]!) (fun t => rfl) v hv).x_size

/-- **At the deployed relation, every claim a rep opens has `m − 6` coordinates.** -/
theorem _root_.FlockComplete.Run.claims_xsize {H : Claims.Msg → Claims.Msg} (o : Run H Readable.relation) {i : ℕ}
    (hi : i < o.tabs.size) (r : ℕ) : ∀ c ∈ o.claims i r, c.x.size = o.tabs[i]!.m - 6 := by
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
  rw [hcl, show o.tabs[0]!.m = st.m by rw [ht]; rfl]
  exact Replay.claims_xsize st hwf hrw h22 (o.trace 0) (o.zcCoins 0 r) (o.lcCoins 0 r) o.sess.points
    (o.sess_points hspec)

/-- **`BasisOk` at the deployed relation**: at every rep, the verifier's basis is the batched basis's extension. -/
theorem basisOk_relation (H : Claims.Msg → Claims.Msg) : BasisOk H Readable.relation := by
  intro o i hi r _ q e y hqe hy
  obtain ⟨⟨hn7, -⟩, hsm⟩ := fast100_dims _ _ (o.sch_fast100 hi)
  have hn : (o.sch i).n = o.tabs[i]!.m - 7 := by rw [hn7, hsm]
  rw [hn] at hqe ⊢
  exact Lg.basis_sws (o.claims i r) (o.rsCoins i r) _ (o.claims_xsize hi r) q e y hqe hy

end FlockComplete.Replay
