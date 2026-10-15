import Proofs.Flock.C3.Opening.Rows
import Proofs.Flock.VBridge.Algebra.BitsOk
import Proofs.Flock.VBridge.PartS

/-!
# n4's coins: what `RecCoins.opening` accepted

`RecCoins.opening` builds R's opening at its coins: the two claims (`coinClaim`, value zero), `RecOpening.run` on the
symbolic messages at the coins of `stacked/rep0` (through `run_of_inline`), the passes from `kappa` zero, and its check
that the S they build is `RecOpening.structureOf`'s ("the structure moved").

* `opening_ok`: item 2 and (b). What an accepted opening is, in the form `openingRead_of_rows` reads: the claims, the
  run at some coins, the passes, the S equal to the shape's, and the opening's `v` and `kappa` the passes'.
* `structureOf_bitsOkO`: the shape's S has every `wt` bit below 128 (`structureOf_bitsOk` at the opening), so the
  parts' residuals cover it (`residuals_zero_of_parts`).
-/

namespace FlockC3.Opening

open Flock Flock.RecAlgebra

/-! ## The coins -/

theorem mapError_ok {ε A : Type} {x : Except ε A} {f : ε → ε} {a : A} (h : x.mapError f = .ok a) : x = .ok a := by
  cases x with
  | error e => cases h
  | ok a' => cases h; rfl

theorem mapM_two {A : Type} {f : Nat → Except String A} {cs : Array A} (h : #[0, 1].mapM f = .ok cs) :
    ∃ a b, f 0 = .ok a ∧ f 1 = .ok b ∧ cs = #[a, b] := by
  rw [Array.mapM_eq_mapM_toList] at h
  simp only [List.mapM_cons, List.mapM_nil] at h
  cases h0 : f 0 with
  | error e => rw [h0] at h; cases h
  | ok a =>
    cases h1 : f 1 with
    | error e => rw [h0, h1] at h; cases h
    | ok b =>
      rw [h0, h1] at h
      cases h
      exact ⟨a, b, rfl, rfl, rfl⟩

theorem run'_ok {A : Type} {x : CM A} {k : Nat} {src : RecAlgebra.Coins} {a : A}
    (h : (StateT.run' x k).run src = .ok a) : ∃ k', x k src = .ok (a, k') := by
  have e : (StateT.run' x k).run src = Prod.fst <$> x k src := rfl
  rw [e] at h
  cases hx : x k src with
  | error e => rw [hx] at h; cases h
  | ok p => rw [hx] at h; cases h; exact ⟨p.2, rfl⟩

theorem toOption_ok {A : Type} [DecidableEq A] {x : Except String A} {a : A}
    (h : ¬(some a != x.toOption) = true) : x = .ok a := by
  cases x with
  | error e => simp [Except.toOption] at h
  | ok a' =>
    simp only [Except.toOption, bne_iff_ne, ne_eq, Decidable.not_not, Option.some.injEq] at h
    rw [h]

/-- Claim `k`'s point at its rep's sparse coins `sc`: the sparse point, zero-padded to R's `m`. -/
def coinPt (c3 : RecCoins.C3) (sh : Shape) (sc : RecCoins.SparseCoins) : Array F128 :=
  let pt := sc.rounds.extract 0 c3.sparse.nu ++ sc.zetaP ++ sc.rho
  pt ++ Array.replicate (sh.m - pt.size) 0

/-- Claim `k` of R's opening as `RecCoins.opening` builds it at the shape `sh`: rep `k`'s sparse point, zero-padded to
R's `m` (`coinPt`), its skip weights and the rest; its value zero, a row of `rec-claim` that w holds. -/
def coinClaim (c3 : RecCoins.C3) (unit : String) (sh : Shape) (k : Nat) : V (AClaim Lin) := do
  let sc ← RecCoins.sparseCoins c3 unit k
  let pt := sc.rounds.extract 0 c3.sparse.nu ++ sc.zetaP ++ sc.rho
  if pt.size < RecSparse.PRE_ROUNDS || pt.size > sh.m then
    throw s!"rec-coins: {unit}: rep {k}'s claim point has {pt.size} coordinates, R's m is {sh.m}"
  let pt := coinPt c3 sh sc
  pure ({ value := RecAlgebra.instValLin.zero, skip := eqTable (pt.extract 0 RecSparse.PRE_ROUNDS),
          x := pt.extract RecSparse.PRE_ROUNDS pt.size } : RecAlgebra.AClaim RecAlgebra.Lin)

theorem coinClaim_zero {c3 : RecCoins.C3} {unit : String} {sh : Shape} {k : Nat} {a : AClaim Lin}
    (h : coinClaim c3 unit sh k = .ok a) : a.value = Val.zero := by
  unfold coinClaim at h
  obtain ⟨sc, -, h⟩ := FlockVBridge.PartS.bind_ok h
  dsimp only at h
  split at h
  · cases h
  · cases h
    rfl

/-- **What `coinClaim` built**: at rep `k`'s sparse coins, the padded point has R's `m` coordinates, at least the
skip's, and the claim's skip weights and rest are its first `PRE_ROUNDS` and the others. -/
theorem coinClaim_ok {c3 : RecCoins.C3} {unit : String} {sh : Shape} {k : Nat} {a : AClaim Lin}
    (h : coinClaim c3 unit sh k = .ok a) :
    ∃ sc, RecCoins.sparseCoins c3 unit k = .ok sc ∧ (coinPt c3 sh sc).size = sh.m ∧ RecSparse.PRE_ROUNDS ≤ sh.m ∧
      a.skip = eqTable ((coinPt c3 sh sc).extract 0 RecSparse.PRE_ROUNDS) ∧
      a.x = (coinPt c3 sh sc).extract RecSparse.PRE_ROUNDS (coinPt c3 sh sc).size := by
  unfold coinClaim at h
  obtain ⟨sc, hsc, h⟩ := FlockVBridge.PartS.bind_ok h
  dsimp only at h
  split at h
  · cases h
  · rename_i hb
    cases h
    simp only [Bool.or_eq_true, decide_eq_true_eq, not_or, Nat.not_lt, gt_iff_lt] at hb
    obtain ⟨h1, h2⟩ := hb
    refine ⟨sc, hsc, ?_, by omega, ?_, ?_⟩
    · simp only [coinPt, Array.size_append, Array.size_replicate] at h1 h2 ⊢
      omega
    · dsimp only
    · dsimp only

/-- **(n4 item 2 and (b)) What `RecCoins.opening` accepted**: R's stacked shape, its two claims (value zero),
`RecOpening.run` on the symbolic messages and those claims at some coins (`run_of_inline`), the passes from `kappa`
zero over its residuals, the S they build the shape's (`structureOf`, "the structure moved"), and the opening's `v`
and `kappa` the passes'. -/
theorem opening_ok {c3 : RecCoins.C3} {unit : String} {o : RecCoins.Opening} (h : RecCoins.opening c3 unit = .ok o) :
    ∃ (r : RecVStar.StackedShape) (a b : AClaim Lin) (src : RecAlgebra.Coins) (lins : Array Lin) (k₁ : Nat)
      (res : Array Res) (bs : BSt),
      c3.stacked = some r ∧ RecOpening.shape? r.m 2 = some o.sh ∧
      coinClaim c3 unit o.sh 0 = .ok a ∧ coinClaim c3 unit o.sh 1 = .ok b ∧
      RecOpening.run o.sh (symbolic o.sh) #[a, b] 0 src = .ok (lins, k₁) ∧
      (RecOpening.buildAll o.sh lins.toList 0).run { kappa := kappa0 o.sh } = .ok (res, bs) ∧
      RecOpening.structureOf o.sh = .ok ⟨RecOpening.nw o.sh, bs.vNames.size, bs.ops, res⟩ ∧
      o.v = bs.vVals ∧ o.kappa = bs.kappa := by
  unfold RecCoins.opening at h
  cases hr : c3.stacked with
  | none => rw [hr] at h; cases h
  | some r =>
    rw [hr] at h
    dsimp only at h
    cases hsh : RecOpening.shape? r.m r.claims with
    | none => rw [hsh] at h; cases h
    | some sh =>
      rw [hsh] at h
      dsimp only at h
      split at h
      · cases h
      · rename_i hcl
        have hcl : r.claims = 2 := by simpa using hcl
        obtain ⟨cs, hcs, h⟩ := FlockVBridge.PartS.bind_ok h
        obtain ⟨a, b, ha, hb, rfl⟩ := mapM_two hcs
        obtain ⟨rounds, -, h⟩ := FlockVBridge.PartS.bind_ok h
        have h := mapError_ok h
        obtain ⟨⟨lins, lg⟩, hrun, h⟩ := FlockVBridge.PartS.bind_ok h
        obtain ⟨⟨res, bs⟩, hbuild, h⟩ := FlockVBridge.PartS.bind_ok h
        split at h
        · cases h
        · rename_i hS
          split at h
          · cases h
          cases h
          obtain ⟨k₁, hx⟩ := run'_ok hrun
          exact ⟨r, a, b, _, lins, k₁, res, bs, rfl, hcl ▸ hsh, ha, hb,
            run_of_inline sh (symbolic sh) #[a, b] 0 _ _ k₁ hx, hbuild, toOption_ok hS, rfl, rfl⟩

/-! ## The shape's S has its bits in range -/

section Bits

variable {sh : Shape}

theorem atomO_bitOk {k : Key} {t : Term} (ha : RecOpening.atomTerm sh k = .ok t) (hk : k.Good) : t.BitOk := by
  cases k <;> simp only [RecOpening.atomTerm] at ha <;> (try split at ha) <;> (try cases ha) <;> exact hk

theorem varOp_bitsO {k : Key} {st st' : BSt} {o : Nat} (hr : RecOpening.varOp sh k st = .ok (o, st'))
    (hi : OpsBits st.ops) (hk : k.Good) : OpsBits st'.ops := by
  unfold RecOpening.varOp at hr
  obtain ⟨st0, st1, hg, hr⟩ := bm_bind hr
  cases hg
  cases hl : st.opOf.lookup k with
  | some i =>
    simp only [hl] at hr
    cases hr
    exact hi
  | none =>
    simp only [hl] at hr
    obtain ⟨t', st2, ht, hr⟩ := bm_bind hr
    obtain ⟨ha, h2⟩ := liftB_ok ht
    subst st2
    obtain ⟨_, st3, hs, hr⟩ := bm_bind hr
    cases hs
    cases hr
    exact hi.push (single_bitOk (atomO_bitOk ha hk))

theorem coefOps_bitsO (ri : Nat) : ∀ (l : List (Key × F128)) (st st' : BSt) (ps : Array (Nat × Nat)),
    RecOpening.coefOps sh ri l st = .ok (ps, st') → OpsBits st.ops → (∀ kx ∈ l, kx.1.Good) → OpsBits st'.ops
  | [], st, st', ps, hr, hi, _ => by
    cases hr
    exact hi
  | (k, x) :: rest, st, st', ps, hr, hi, hl => by
    obtain ⟨st0, st1, hg, hr⟩ := bm_bind hr
    cases hg
    obtain ⟨_, st2, hs, hr⟩ := bm_bind hr
    cases hs
    obtain ⟨o, st3, hv, hr⟩ := bm_bind hr
    obtain ⟨ps', st4, hc, hr⟩ := bm_bind hr
    cases hr
    have hA : OpsBits (st.ops.push #[Term.v st.vNames.size]) := hi.push (single_bitOk trivial)
    exact coefOps_bitsO ri rest st3 _ ps' hc (varOp_bitsO hv hA (hl (k, x) (by simp)))
      fun kx hkx => hl kx (by simp [hkx])

theorem prodOps_bitsO (ri : Nat) : ∀ (l : List ((Key × Key) × F128)) (st st' : BSt) (ps : Array (Nat × Nat)),
    RecOpening.prodOps sh ri l st = .ok (ps, st') → OpsBits st.ops → (∀ abx ∈ l, abx.1.1.Good ∧ abx.1.2.Good) →
      OpsBits st'.ops
  | [], st, st', ps, hr, hi, _ => by
    cases hr
    exact hi
  | ((a, b), x) :: rest, st, st', ps, hr, hi, hl => by
    simp only [RecOpening.prodOps] at hr
    split at hr
    · obtain ⟨_, _, h1, _⟩ := bm_bind hr
      cases h1
    · obtain ⟨oa, st2, ha, hr⟩ := bm_bind hr
      obtain ⟨ob, st3, hb, hr⟩ := bm_bind hr
      obtain ⟨ps', st4, hc, hr⟩ := bm_bind hr
      cases hr
      have hab := hl ((a, b), x) (by simp)
      exact prodOps_bitsO ri rest st3 _ ps' hc (varOp_bitsO hb (varOp_bitsO ha hi hab.1) hab.2)
        fun abx h => hl abx (by simp [h])

theorem accTerms_bitsO {ri : Nat} {l : List (Key × F128)} {st st' : BSt} {acc : Array Term}
    (hr : RecOpening.accTerms sh ri l st = .ok (acc, st')) : st'.ops = st.ops ∧ ∀ t ∈ acc, t.BitOk := by
  refine ⟨(accTerms_specO #[] (fun _ _ => 0) hr).1, ?_⟩
  simp only [RecOpening.accTerms] at hr
  split at hr
  · cases hr
    intro t ht
    simp at ht
  · split at hr
    · cases hr
    · obtain ⟨_, st1, _, hr⟩ := bm_bind hr
      cases h1 : RecOpening.wi sh (.acc1 (ri - sh.claims)) with
      | none => simp only [h1] at hr; cases hr
      | some i1 =>
        cases h0 : RecOpening.wi sh (.acc0 (ri - sh.claims)) with
        | none => simp only [h1, h0] at hr; cases hr
        | some i0 =>
          simp only [h1, h0] at hr
          cases hr
          intro t ht
          simp at ht
          rcases ht with rfl | rfl <;> trivial

theorem buildRes_bitsO {ri : Nat} {f : Lin} {st st' : BSt} {res : Res}
    (hr : RecOpening.buildRes sh ri f st = .ok (res, st')) (hi : OpsBits st.ops) (hf : f.Good) :
    OpsBits st'.ops ∧ ∀ t ∈ res.add, t.BitOk := by
  unfold RecOpening.buildRes at hr
  obtain ⟨acc, st1, h1, hr⟩ := bm_bind hr
  obtain ⟨a1, abits⟩ := accTerms_bitsO h1
  obtain ⟨l1, st1', hs1, hr⟩ := bm_bind hr
  obtain ⟨hl1, h⟩ := liftB_ok hs1
  subst st1'
  obtain ⟨cm, st2, h2, hr⟩ := bm_bind hr
  obtain ⟨l2, st2', hs2, hr⟩ := bm_bind hr
  obtain ⟨hl2, h⟩ := liftB_ok hs2
  subst st2'
  obtain ⟨pm, st3, h3, hr⟩ := bm_bind hr
  obtain ⟨cp, st4, h4, hr⟩ := bm_bind hr
  cases hr
  obtain ⟨c4, cbits⟩ := constTerm_bits h4
  have hi1 : OpsBits st1.ops := by rw [a1]; exact hi
  have hg1 : ∀ kx ∈ l1, kx.1.Good := fun kx hk => hf.1 kx (List.mem_filter.mp ((sortBy_perm hl1).subset hk)).1
  have hg2 : ∀ abx ∈ l2, abx.1.1.Good ∧ abx.1.2.Good := fun abx hk => hf.2 abx ((sortBy_perm hl2).subset hk)
  have hi3 := prodOps_bitsO ri l2 st2 st3 pm h3 (coefOps_bitsO ri l1 st1 st2 cm h2 hi1 hg1) hg2
  refine ⟨by rw [c4]; exact hi3, fun t ht => ?_⟩
  rcases Array.mem_append.mp ht with ht | ht
  · rcases Array.mem_append.mp ht with ht | ht
    · exact abits t ht
    · split at ht
      · simp at ht
        subst ht
        trivial
      · simp at ht
  · exact cbits t ht

theorem buildAll_bitsO : ∀ (fs : List Lin) (ri : Nat) (st st' : BSt) (rs : Array Res),
    RecOpening.buildAll sh fs ri st = .ok (rs, st') → OpsBits st.ops → (∀ f ∈ fs, f.Good) →
      OpsBits st'.ops ∧ ∀ r ∈ rs, ∀ t ∈ r.add, t.BitOk
  | [], ri, st, st', rs, hr, hi, _ => by
    cases hr
    exact ⟨hi, fun r hr => by simp at hr⟩
  | f :: rest, ri, st, st', rs, hr, hi, hf => by
    obtain ⟨r, st1, h1, hr⟩ := bm_bind hr
    obtain ⟨rs', st2, h2, hr⟩ := bm_bind hr
    cases hr
    obtain ⟨h1i, h1a⟩ := buildRes_bitsO h1 hi (hf f (by simp))
    obtain ⟨h2i, h2a⟩ := buildAll_bitsO rest (ri + 1) st1 _ rs' h2 h1i fun g hg => hf g (by simp [hg])
    refine ⟨h2i, fun r' hr' => ?_⟩
    rcases Array.mem_append.mp hr' with hr' | hr'
    · simp at hr'
      subst hr'
      exact h1a
    · exact h2a r' hr'

end Bits

theorem claimsLike_ren (sh : Shape) :
    (RecOpening.claimsLike sh : Array (AClaim Lin)).map (AClaim.map (Lin.ren bump)) = RecOpening.claimsLike sh := by
  apply amap_fixed
  intro c hc
  simp only [RecOpening.claimsLike] at hc
  obtain ⟨-, rfl⟩ := Array.mem_replicate.mp hc
  rfl

theorem runLins_ren {sh : Shape} {lins : Array Lin}
    (h : ((RecOpening.run sh (symbolic sh) (RecOpening.claimsLike sh)).run' 0).run RecAlgebra.Coins.zeros =
      .ok lins) :
    lins.map (Lin.ren bump) = lins := by
  obtain ⟨k, hk⟩ := run'_ok h
  have hs :=
    run_simO (ren_hom bump_inj) sh (symbolic sh) (RecOpening.claimsLike sh) 0 RecAlgebra.Coins.zeros _ _ hk
  rw [symbolic_ren, claimsLike_ren, hk] at hs
  simp only [Except.ok.injEq, Prod.mk.injEq] at hs
  exact hs.1.symm

/-- `RecOpening.build`'s S is its passes' from `kappa` zero. -/
theorem build_okO {sh : Shape} {lins : Array Lin} {S : GfS} (h : RecOpening.build sh lins = .ok S) :
    ∃ res st, RecOpening.buildAll sh lins.toList 0 { kappa := kappa0 sh } = .ok (res, st) ∧
      S = ⟨RecOpening.nw sh, st.vNames.size, st.ops, res⟩ := by
  unfold RecOpening.build at h
  generalize hK : sh.sched.levels.map (fun lv => Array.replicate 2 (Array.replicate (2 ^ lv.lanes) (0 : F128))) = K
    at h
  generalize hc0 : (lins.toList.map slotsOf).sum = comb0 at h
  simp only at h
  cases hB : (RecOpening.buildAll sh lins.toList 0).run { kappa := K } with
  | error e => rw [hB] at h; cases h
  | ok p =>
    obtain ⟨res, st⟩ := p
    rw [hB] at h
    simp only [bind, Except.bind] at h
    split at h
    · cases h
    · cases h
      subst hK
      exact ⟨res, st, hB, rfl⟩

/-- **The opening's S has its bits in range** (`structureOf_bitsOk` at `RecOpening.structureOf`): its symbolic
residuals are good (`bump` fixes them), so every term the passes build is. -/
theorem structureOf_bitsOkO {sh : Shape} {S : GfS} (h : RecOpening.structureOf sh = .ok S) : S.BitsOk := by
  unfold RecOpening.structureOf at h
  obtain ⟨lins, hl, h⟩ := FlockVBridge.PartS.bind_ok h
  have hfix : lins.toList.map (Lin.ren bump) = lins.toList := by
    rw [← Array.toList_map, runLins_ren hl]
  obtain ⟨res, st, hB, rfl⟩ := build_okO h
  exact buildAll_bitsO lins.toList 0 _ _ _ hB (fun _ h => by simp at h)
    fun f hf => good_of_ren (fixed_of_map hfix f hf)

end FlockC3.Opening
