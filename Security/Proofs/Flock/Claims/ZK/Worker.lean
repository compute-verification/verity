import Proofs.Flock.Soundness.ZK.SHVZK

/-!
# Lemma B at a worker's values

The ZK server's workers compute the clear values under the pads (`Table.vals`: the zerocheck and lincheck rounds, the
final evaluations) and never see the pads. A worker's values are a function of the witness, the mask slot `u` and the
extra lanes `R` (`Table.Vals`), and of nothing the server draws for the pads (`h`, `pad_h`, `pad_μ`, `μ`, the salts).
Lemma B holds at any such values, whether or not they pass the inner proof's constraint:

* `swView`, the **switched prover**, at two witnesses of one table that differ only in their clear values, `p` (the
  worker's) and `q` (the honest one): where `p`'s values pass the inner constraint at every pad (`RowsAt`, a property of
  `u` and `R` alone), `p`'s view; elsewhere `q`'s view on the pads moved by Lemma A's amounts (`shift`), so that every
  masked value and `ρ_in`, `σ_in` are `p`'s.
* `sw_shvzk`: the switched view and `S_shvzk(x, e)` have the same distribution, at any `p`, on `InnerHolds` of `q` alone:
  its components are `card_seq_masked4`'s form, the pads that the triple and `C_h` read (`swPads`) being a function of the
  earlier blocks.
* `swView_of_rows`, `swView_outer`: the switched view is `p`'s where the rows hold, and elsewhere differs from `p`'s only
  in `Y₁`, `C_h` and the pads tree's leaves, which the inner proof sends after every masked value. A server that checks
  the rows before it commits the pads tree, and stops where they fail, releases a cut of the switched prover's run.
* `onVals`, `worker_shvzk`, `worker_prefinal`: a table's witnesses paired with any clear values (`Vals`), and Lemmas B and
  A there: the switched view at a worker's values `v` is distributed as the table's own `S_shvzk`, on the honest
  prover's `InnerHolds`; the pre-final view at `v` as the dummy's.
-/

namespace FlockSoundness.ZK.Table

open Finset

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-! ## The switched prover -/

/-- **Lemma A's move of the pads** from `q`'s clear values to `p`'s at `(u, R)`: each exposed pad by the difference, and
`h_tx`, `h_ty` so that `ρ_in`, `σ_in` stay. -/
def padShift {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] (T : Table F K W Salt Seed D Hi VL) (p q : W)
    (u : T.Mask) (R : T.XLanes) : Fin 2 → (Fin T.nE → F) × F × F := fun r =>
  (fun i => T.vals r p u R i - T.vals r q u R i, -(T.ζ r * (T.vals r p u R T.ia - T.vals r q u R T.ia)),
    -(T.vals r p u R T.ib - T.vals r q u R T.ib))

/-- The tape `q`'s prover continues on: the pads moved by `padShift`, every other block kept. -/
def shift {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] (T : Table F K W Salt Seed D Hi VL) (p q : W)
    (ω : T.Rand) : T.Rand :=
  (ω.1, ω.2.1, ω.2.2.1 + T.padShift p q ω.1 ω.2.1.2.1, ω.2.2.2)

/-- **The rows hold at `(u, R)`**: `p`'s clear values pass the inner proof's constraint at every pad. It reads the mask
slot and the extra lanes alone. -/
def RowsAt {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] [CharP F 2] (T : Table F K W Salt Seed D Hi VL)
    (p : W) (u : T.Mask) (R : T.XLanes) : Prop :=
  ∀ ω : T.Rand, ω.1 = u → ω.2.1.2.1 = R → ∀ r,
    ∑ i, T.lam r (T.view p ω).pre i * T.hFull (ω.2.2.1 r) i = T.theta r (T.view p ω).pre

open Classical in
/-- The pads the switched prover's triple and `C_h` read: the drawn ones where the rows hold, else the moved ones. -/
noncomputable def swPads {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] [CharP F 2]
    (T : Table F K W Salt Seed D Hi VL) (p q : W) (u : T.Mask) (b : (Fin T.nL → Fin T.tpad → F) × T.XLanes × T.XPad)
    (h : Fin 2 → (Fin T.nE → F) × F × F) : Fin 2 → (Fin T.nE → F) × F × F :=
  if T.RowsAt p u b.2.1 then h else h + T.padShift p q u b.2.1

open Classical in
/-- **The switched prover's view**: `p`'s where the rows hold, else `q`'s on the moved pads. -/
noncomputable def swView {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] [CharP F 2]
    (T : Table F K W Salt Seed D Hi VL) (p q : W) (ω : T.Rand) : T.View :=
  if T.RowsAt p ω.1 ω.2.1.2.1 then T.view p ω else T.view q (T.shift p q ω)

/-- **The switched view's components** in Lemma B's sampling order: `realCore` at `p` with the pads `swPads` in the last
block's determined part. -/
noncomputable def swCore {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] [CharP F 2] [AddCommGroup Salt]
    [AddCommGroup Seed] [AddCommGroup Hi] (T : Table F K W Salt Seed D Hi VL) (p q : W) (ω : T.Rand) : T.Core :=
  (T.nu p + T.maskHom ω.1,
    (fun q' l => ∑ c, T.lanes p ω.1 l c * T.E q' c, fun r => T.y1 p ω.1 r, 0) + T.mask1 ω.2.1,
    (fun r => (T.vals r p ω.1 ω.2.1.2.1, 0, 0)) + T.mask2 ω.2.2.1,
    (fun r k => ∑ i, T.Ch r k i * T.hFull (T.swPads p q ω.1 ω.2.1 ω.2.2.1 r) i, 0,
      fun r => T.hFull (T.swPads p q ω.1 ω.2.1 ω.2.2.1 r), 0) + T.mask3 ω.2.2.2)

/-- **The real view is `psi` of its components where the constraint holds**, pointwise (`psi_realCore` reads
`InnerHolds` only at `ω`). -/
theorem psi_realCore_at {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] [CharP F 2] [AddCommGroup Salt]
    [AddCommGroup Seed] [AddCommGroup Hi] (T : Table F K W Salt Seed D Hi VL)
    (hW : ∀ r, Function.Bijective fun p : F × F => T.Wr r p.1 p.2) (hβ : ∀ r, T.β r ≠ 0) (w : W)
    (hreg : T.sReg w = T.pubReg) (ω : T.Rand)
    (hinner : ∀ r, ∑ i, T.lam r (T.view w ω).pre i * T.hFull (ω.2.2.1 r) i = T.theta r (T.view w ω).pre) :
    T.psi (T.realCore w ω) = T.view w ω := by
  obtain ⟨u, ⟨μ0, R, μR⟩, h, ⟨padh, padμ, μ, s⟩⟩ := ω
  set P := (T.view w (u, (μ0, R, μR), h, padh, padμ, μ, s)).pre with hP
  have htau : ∀ r, (∑ i, T.lam r P i * (T.hFull (h r) i + T.β r * μ r i) - T.theta r P) / T.β r
      = ∑ i, T.lam r P i * μ r i := by
    intro r
    have hI : ∑ i, T.lam r P i * T.hFull (h r) i = T.theta r P := hinner r
    have e : ∑ i, T.lam r P i * (T.hFull (h r) i + T.β r * μ r i)
        = T.theta r P + T.β r * ∑ i, T.lam r P i * μ r i := by
      rw [← hI, Finset.mul_sum, ← Finset.sum_add_distrib]
      exact Finset.sum_congr rfl fun i _ => by ring
    rw [e, add_sub_cancel_left, mul_div_cancel_left₀ _ (hβ r)]
  have hcm : ∀ r k, (∑ i, T.Ch r k i * (T.hFull (h r) i + T.β r * μ r i)
      + ∑ j, T.Cp r k j * (padh r j + T.β r * padμ r j)
      - (∑ i, T.Ch r k i * T.hFull (h r) i + ∑ j, T.Cp r k j * padh r j)) / T.β r
      = ∑ i, T.Ch r k i * μ r i + ∑ j, T.Cp r k j * padμ r j := by
    intro r k
    have e1 : ∑ i, T.Ch r k i * (T.hFull (h r) i + T.β r * μ r i)
        = ∑ i, T.Ch r k i * T.hFull (h r) i + T.β r * ∑ i, T.Ch r k i * μ r i := by
      rw [Finset.mul_sum, ← Finset.sum_add_distrib]
      exact Finset.sum_congr rfl fun i _ => by ring
    have e2 : ∑ j, T.Cp r k j * (padh r j + T.β r * padμ r j)
        = ∑ j, T.Cp r k j * padh r j + T.β r * ∑ j, T.Cp r k j * padμ r j := by
      rw [Finset.mul_sum, ← Finset.sum_add_distrib]
      exact Finset.sum_congr rfl fun j _ => by ring
    rw [e1, e2, div_eq_iff (hβ r)]
    ring
  have hx : ∀ q r, T.Winv r (T.xTarget (T.rows w u μ0) (T.y1' w u R) (T.ybar μ0 μR) q r)
      = (T.xrows R μR q r 0, T.xrows R μR q r 1) := fun q r => by
    rw [← T.star w u μ0 R μR q r, T.Winv_Wr (hW r)]
  have hxr : (fun q r => ![(T.Winv r (T.xTarget (T.rows w u μ0) (T.y1' w u R) (T.ybar μ0 μR) q r)).1,
      (T.Winv r (T.xTarget (T.rows w u μ0) (T.y1' w u R) (T.ybar μ0 μR) q r)).2]) = T.xrows R μR := by
    funext q r i
    rw [hx q r]
    fin_cases i <;> rfl
  apply View.ext
  all_goals simp only [psi, view, realCore, mask1_apply, mask2_apply, mask3_apply, Pi.add_apply, zero_add,
    Prod.mk_add_mk]
  all_goals (try rfl)
  · exact hreg.symm
  · funext r
    exact htau r
  · funext r
    exact congrArg (fun t => T.leafT t (s.2.2.2.2.2.2.1 r)) (htau r)
  · exact hxr
  · refine Prod.ext (funext fun q => ?_) rfl
    exact congrArg (fun x => T.leaf0 (T.rows w u μ0 q, x) (s.1 q)) (congrFun hxr q)
  · funext r k
    exact hcm r k
  · funext r
    refine Prod.ext (funext fun k => ?_) rfl
    exact congrArg (fun c => T.leafP (_, c) (s.2.2.2.1 r k)) (hcm r k)

/-- Where the rows fail, the switched view's components are the honest prover's on the moved pads: the masked values
and `ρ_in`, `σ_in` are the same, the moved pads absorbing the difference of the clear values. -/
theorem swCore_of_not_rows {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] [CharP F 2] [AddCommGroup Salt]
    [AddCommGroup Seed] [AddCommGroup Hi] (T : Table F K W Salt Seed D Hi VL) (p q : W) (hnu : T.nu p = T.nu q)
    (hlanes : T.lanes p = T.lanes q) (ω : T.Rand) (hr : ¬ T.RowsAt p ω.1 ω.2.1.2.1) :
    T.swCore p q ω = T.realCore q (T.shift p q ω) := by
  have hy : T.y1 p ω.1 = T.y1 q ω.1 := by
    funext r c
    simp only [y1, hlanes]
  have hp : T.swPads p q ω.1 ω.2.1 ω.2.2.1 = ω.2.2.1 + T.padShift p q ω.1 ω.2.1.2.1 := by
    simp only [swPads, hr, ↓reduceIte]
  simp only [swCore, realCore, shift, hp]
  refine Prod.ext (by rw [hnu]) (Prod.ext (by rw [hlanes, hy]) (Prod.ext ?_ rfl))
  funext r
  simp only [Pi.add_apply, mask2_apply, padShift, Prod.mk_add_mk, Prod.fst_add, Prod.snd_add]
  refine Prod.ext ?_ (Prod.ext ?_ ?_)
  · funext i
    simp only [Pi.add_apply]
    ring
  · simp only [zero_add]
    ring
  · simp only [zero_add]
    ring

/-- **The switched view is `psi` of its components**: where the rows hold by `psi_realCore_at`, else by `psi_realCore`
at the honest witness on the moved pads. -/
theorem psi_swCore {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] [CharP F 2] [AddCommGroup Salt]
    [AddCommGroup Seed] [AddCommGroup Hi] (T : Table F K W Salt Seed D Hi VL)
    (hW : ∀ r, Function.Bijective fun p : F × F => T.Wr r p.1 p.2) (hβ : ∀ r, T.β r ≠ 0) (p q : W)
    (hnu : T.nu p = T.nu q) (hlanes : T.lanes p = T.lanes q) (hreg : T.sReg p = T.pubReg)
    (hreg' : T.sReg q = T.pubReg) (hinner : T.InnerHolds q) (ω : T.Rand) :
    T.psi (T.swCore p q ω) = T.swView p q ω := by
  by_cases hr : T.RowsAt p ω.1 ω.2.1.2.1
  · have e : T.swCore p q ω = T.realCore p ω := by
      simp only [swCore, swPads, hr, ↓reduceIte]
      rfl
    rw [e, swView, ite_eq_left hr]
    exact T.psi_realCore_at hW hβ p hreg ω (hr ω rfl rfl)
  · rw [T.swCore_of_not_rows p q hnu hlanes ω hr, swView, ite_eq_right hr]
    exact T.psi_realCore hW hβ q hreg' hinner _

/-- **The switched view takes every value of its components equally often** (`card_seq_masked4`). -/
theorem card_swCore {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] [CharP F 2] [AddCommGroup Salt]
    [AddCommGroup Seed] [AddCommGroup Hi] [Fintype F] [DecidableEq F] [DecidableEq K] [Fintype Salt]
    [DecidableEq Salt] [Fintype Seed] [DecidableEq Seed] [Fintype Hi] [DecidableEq Hi]
    (T : Table F K W Salt Seed D Hi VL) (hrank : T.Rank) (hpad : T.PadOnto)
    (hW : ∀ r, Function.Bijective fun p : F × F => T.Wr r p.1 p.2) (hpads : T.PadsOnto) (hβ : ∀ r, T.β r ≠ 0)
    (p q : W) (v : T.Core) :
    #{ω | T.swCore p q ω = v} = #{κ : T.Rand | T.maskHom κ.1 = 0 ∧ T.mask1 κ.2.1 = 0 ∧ T.mask2 κ.2.2.1 = 0 ∧
      T.mask3 κ.2.2.2 = 0} :=
  card_seq_masked4 T.maskHom T.mask1 T.mask2 T.mask3 (T.maskHom_surjective hrank) (T.mask1_surjective hpad hW)
    T.mask2_surjective (T.mask3_surjective hpads hβ) (T.nu p)
    (fun u => (fun q' l => ∑ c, T.lanes p u l c * T.E q' c, fun r => T.y1 p u r, 0))
    (fun u b => fun r => (T.vals r p u b.2.1, 0, 0))
    (fun u b h => (fun r k => ∑ i, T.Ch r k i * T.hFull (T.swPads p q u b h r) i, 0,
      fun r => T.hFull (T.swPads p q u b h r), 0)) v

/-- **Lemma B at any clear values**: for two witnesses that differ only in their clear values (the same `ν`, region
messages and lanes), carrying the public region data, the switched view and `S_shvzk(x, e)` have the same distribution,
whatever `p`'s values, on the inner constraint at `q`'s alone. -/
theorem sw_shvzk {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] [CharP F 2] [AddCommGroup Salt]
    [AddCommGroup Seed] [AddCommGroup Hi] [Fintype F] [DecidableEq F] [Fintype K] [DecidableEq K] [Fintype Salt]
    [DecidableEq Salt] [Fintype Seed] [DecidableEq Seed] [Fintype Hi] [DecidableEq Hi]
    (T : Table F K W Salt Seed D Hi VL) (hrank : T.Rank) (hpad : T.PadOnto)
    (hW : ∀ r, Function.Bijective fun p : F × F => T.Wr r p.1 p.2) (hpads : T.PadsOnto) (hβ : ∀ r, T.β r ≠ 0)
    (p q : W) (hnu : T.nu p = T.nu q) (hlanes : T.lanes p = T.lanes q) (hreg : T.sReg p = T.pubReg)
    (hreg' : T.sReg q = T.pubReg) (hinner : T.InnerHolds q) (w0 : W) :
    SameDist (T.swView p q) (T.sim w0) :=
  ((SameDist.of_const (T.card_swCore hrank hpad hW hpads hβ p q) (T.card_simCore hpad hW w0)).comp T.psi).of_eq
    (T.psi_swCore hW hβ p q hnu hlanes hreg hreg' hinner) (T.psi_simCore hW w0)

/-- **Where the rows hold, the switched view is `p`'s.** -/
theorem swView_of_rows {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] [CharP F 2]
    (T : Table F K W Salt Seed D Hi VL) (p q : W) (ω : T.Rand) (hr : T.RowsAt p ω.1 ω.2.1.2.1) :
    T.swView p q ω = T.view p ω := by
  rw [swView, ite_eq_left hr]

/-- **The switched view is `p`'s but `Y₁`, `C_h` and the pads tree's leaves**: every masked value, `ρ_in`, `σ_in`, `τ`
and level 0 are `p`'s everywhere. -/
theorem swView_outer {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] [CharP F 2]
    (T : Table F K W Salt Seed D Hi VL) (p q : W) (hnu : T.nu p = T.nu q) (hreg : T.sReg p = T.sReg q)
    (hlanes : T.lanes p = T.lanes q) (ω : T.Rand) :
    T.swView p q ω =
      { T.view p ω with
        Y1 := (T.swView p q ω).Y1
        ch := (T.swView p q ω).ch
        leavesP := (T.swView p q ω).leavesP } := by
  by_cases hr : T.RowsAt p ω.1 ω.2.1.2.1
  · rw [swView, ite_eq_left hr]
  · rw [swView, ite_eq_right hr]
    obtain ⟨u, ⟨μ0, R, μR⟩, h, ⟨padh, padμ, μ, s⟩⟩ := ω
    have hy : T.y1' q u R = T.y1' p u R := by
      funext r c
      simp only [y1', y1, hlanes]
    have hrows : T.rows q u μ0 = T.rows p u μ0 := by
      funext q' l
      simp only [rows, hlanes]
    have hm : (fun r i => T.vals r q u R i + ((h + T.padShift p q u R) r).1 i) =
        fun r i => T.vals r p u R i + (h r).1 i := by
      funext r i
      simp only [Pi.add_apply, Prod.fst_add, padShift]
      ring
    have hin : (fun r => T.innerMsg r ((h + T.padShift p q u R) r)) = fun r => T.innerMsg r (h r) := by
      funext r
      simp only [innerMsg, Pi.add_apply, Prod.fst_add, Prod.snd_add, padShift]
      refine Prod.ext ?_ ?_ <;> simp only [] <;> ring
    have hpre : (T.view q (T.shift p q (u, (μ0, R, μR), h, padh, padμ, μ, s))).pre =
        (T.view p (u, (μ0, R, μR), h, padh, padμ, μ, s)).pre := by
      simp only [View.pre, view, shift]
      rw [hnu, hm, hin, hy]
    apply View.ext
    all_goals (try rfl)
    · show T.nu q + T.maskHom u = T.nu p + T.maskHom u
      rw [hnu]
    · exact hreg.symm
    · exact hm
    · exact hy
    · show (fun r => T.lig r (T.y1' q u R r) _) = fun r => T.lig r (T.y1' p u R r) _
      rw [hy]
      rfl
    · exact hin
    · show (fun r => ∑ i, T.lam r (T.view q (T.shift p q (u, (μ0, R, μR), h, padh, padμ, μ, s))).pre i * μ r i) =
        fun r => ∑ i, T.lam r (T.view p (u, (μ0, R, μR), h, padh, padμ, μ, s)).pre i * μ r i
      rw [hpre]
    · show (fun r => T.leafT (∑ i, T.lam r (T.view q (T.shift p q (u, (μ0, R, μR), h, padh, padμ, μ, s))).pre i *
          μ r i) (s.2.2.2.2.2.2.1 r)) =
        fun r => T.leafT (∑ i, T.lam r (T.view p (u, (μ0, R, μR), h, padh, padμ, μ, s)).pre i * μ r i)
          (s.2.2.2.2.2.2.1 r)
      rw [hpre]
    · exact hrows
    · show (fun q' => T.leaf0 (T.rows q u μ0 q', T.xrows R μR q') (s.1 q'), fun q' => T.ideal (s.2.2.1 q')) =
        (fun q' => T.leaf0 (T.rows p u μ0 q', T.xrows R μR q') (s.1 q'), fun q' => T.ideal (s.2.2.1 q'))
      rw [hrows]

/-! ## At a worker's clear values -/

/-- **Clear values a worker may supply** for a table: functions of the witness, the mask slot and the extra lanes,
`Table.vals`'s shape, and of nothing else: not of the pads. -/
abbrev Vals {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] (T : Table F K W Salt Seed D Hi VL) : Type :=
  Fin 2 → W → T.Mask → T.XLanes → Fin T.nE → F

/-- **The table at any clear values**: its witnesses paired with clear values, which `vals` reads; every other field
`T`'s, reading the witness. -/
def onVals {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] (T : Table F K W Salt Seed D Hi VL) :
    Table F K (W × T.Vals) Salt Seed D Hi VL where
  nMask := T.nMask
  nC := T.nC
  nR := T.nR
  nL := T.nL
  L := T.L
  tpad := T.tpad
  nS := T.nS
  nU := T.nU
  nE := T.nE
  nQ := T.nQ
  nPU := T.nPU
  nPad := T.nPad
  embed := T.embed
  eqw := T.eqw
  nu p := T.nu p.1
  sReg p := T.sReg p.1
  pubReg := T.pubReg
  lanes p := T.lanes p.1
  E := T.E
  Gp := T.Gp
  rbar := T.rbar
  ρ := T.ρ
  vals r p := p.2 r p.1
  ia := T.ia
  ib := T.ib
  ζ := T.ζ
  β := T.β
  lig := T.lig
  lam := T.lam
  theta := T.theta
  Ch := T.Ch
  Cp := T.Cp
  leaf0 := T.leaf0
  leafP := T.leafP
  leafT := T.leafT
  ideal := T.ideal

/-- A view of the table at any clear values as one of the table's. -/
def toBase {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] (T : Table F K W Salt Seed D Hi VL)
    (v : T.onVals.View) : T.View :=
  ⟨v.sHat, v.sReg, v.masked, v.y1, v.ybar, v.lig, v.inner, v.Y1, v.Y2, v.tau, v.ytau, v.tauCom, v.rows, v.xrows,
    v.salts0, v.leaves0, v.ch, v.cm, v.saltsP, v.leavesP⟩

set_option smartUnfolding false in
/-- At the honest values, the view is the table's. -/
theorem toBase_view {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] [CharP F 2]
    (T : Table F K W Salt Seed D Hi VL) (w : W) (ω : T.Rand) : T.toBase (T.onVals.view (w, T.vals) ω) = T.view w ω :=
  rfl

set_option smartUnfolding false in
/-- At the honest values, the inner constraint is the table's. -/
theorem onVals_innerHolds {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] [CharP F 2]
    (T : Table F K W Salt Seed D Hi VL) (w : W) : T.onVals.InnerHolds (w, T.vals) ↔ T.InnerHolds w :=
  Iff.rfl

/-- **`S_shvzk` at any clear values is the table's own**: it reads no clear value. -/
theorem toBase_sim {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] [CharP F 2]
    (T : Table F K W Salt Seed D Hi VL) (w0 : W) (v0 : T.Vals) (σ : T.SimRand) :
    T.toBase (T.onVals.sim (w0, v0) σ) = T.sim w0 σ :=
  rfl

/-- **Lemma B at a worker's clear values**: at a fixed non-degenerate coin vector, for a witness `w` that carries the
public region data and whose honest pads satisfy the inner constraint, and for any clear values `v` a worker supplies,
the switched prover's view at `v` (`v`'s where its rows hold, else the honest prover's on the moved pads) and the
table's `S_shvzk(x, e)` on any dummy witness have the same distribution. -/
theorem worker_shvzk {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] [CharP F 2] [AddCommGroup Salt]
    [AddCommGroup Seed] [AddCommGroup Hi] [Fintype F] [DecidableEq F] [Fintype K] [DecidableEq K] [Fintype Salt]
    [DecidableEq Salt] [Fintype Seed] [DecidableEq Seed] [Fintype Hi] [DecidableEq Hi]
    (T : Table F K W Salt Seed D Hi VL) (hrank : T.Rank) (hpad : T.PadOnto)
    (hW : ∀ r, Function.Bijective fun p : F × F => T.Wr r p.1 p.2) (hpads : T.PadsOnto) (hβ : ∀ r, T.β r ≠ 0)
    (w : W) (hreg : T.sReg w = T.pubReg) (hinner : T.InnerHolds w) (v : T.Vals) (w0 : W) :
    SameDist (fun ω => T.toBase (T.onVals.swView (w, v) (w, T.vals) ω)) (T.sim w0) :=
  ((T.onVals.sw_shvzk hrank hpad hW hpads hβ (w, v) (w, T.vals) rfl rfl hreg hreg ((T.onVals_innerHolds w).2 hinner)
      (w0, T.vals)).comp
    T.toBase).of_eq (fun _ => rfl) (T.toBase_sim w0 T.vals)

/-- **Lemma A at a worker's clear values**: the pre-final view at any clear values `v` and the dummy's at the honest ones
have the same distribution, on the public region data alone. -/
theorem worker_prefinal {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] [CharP F 2] [AddCommGroup Salt]
    [AddCommGroup Seed] [AddCommGroup Hi] [Fintype F] [DecidableEq F] [Fintype Salt] [DecidableEq Salt]
    [Fintype Seed] [DecidableEq Seed] [Fintype Hi] [DecidableEq Hi] [DecidableEq K] [DecidableEq D]
    (T : Table F K W Salt Seed D Hi VL) (hrank : T.Rank) (hW : ∀ r, Function.Bijective fun p : F × F => T.Wr r p.1 p.2)
    (hβ : ∀ r, T.β r ≠ 0) (w w0 : W) (hreg : T.sReg w = T.pubReg) (hreg0 : T.sReg w0 = T.pubReg) (v : T.Vals) :
    SameDist (T.onVals.preView (w, v)) (T.onVals.preView (w0, T.vals)) :=
  T.onVals.table_prefinal_indep hrank hW hβ (w, v) (w0, T.vals) hreg hreg0

end FlockSoundness.ZK.Table
