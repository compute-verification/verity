import FlockSoundness.Discharge.ZkHidden
import FlockSoundness.Discharge.ScopeG.Places

set_option Elab.async false

/-!
# The compiled `--zk` count curve at the hidden-output tables, at the VU's program

`ZkHidden.zk_flock_countC_execOS_crH` holds at any partition and any places of its drawn units in the `--zk` tables'
statements (`place`). At the VU's program (`vuProg`, `vuOutsOf`) the places come as `Places.lean`'s do at the hidden
tables, from a placement of the VU's rows with its inputs on their unit slots, here at each `--zk` table's acceptance
(`VuPlaceZ`, an `AcceptedH` as `ZkHidden.AcceptsZKAcceptedH` gives one): `VuPlaceZ.place`, aliased at every input group
from keys (`vuProg_aliased`).

* Open, a named obligation on a `sorry` stub: `vuPlaceZ_nonempty`, that the `--zk` tables have such a placement.
* `zk_flock_countG`: the count curve at `vuProg` and the placement the obligation gives (`vuPlaceZG`), at `scopeOkGT`
  (so at `scopeOkG`); its value binding `vb` is a hypothesis, as in `zk_flock_countC_execOS_crH`.
-/

namespace FlockSoundness.Discharge.ScopeG

open Game Model Accounting Audit ZkSession StrictCR ZkLink Exec Flock
open FlockSoundness.Discharge.HiddenExec FlockSoundness.Discharge.ZkHidden FlockSoundness.Discharge.ZkExec
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

section place

variable {c : Flock.Circuit} {hs : scopeOkGT c = true} {n mPts : ℕ} {Reg : Type}
  {planZ : Finset (Fin n) → Reg → List (TabZK 𝔽 mPts)}
  {tabZ : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length}

variable (c hs planZ tabZ) in
/-- **The VU's rows placed at each drawn unit's `--zk` table**: the table's acceptance (`acc`, of the circuit `c`), the
unit's VU and block there, the VU's rows' columns, each VU input at its stacked unit input's column (`hcol`), and the
placement (`placed`): `VuPlace` with its sites. -/
structure VuPlaceZ where
  acc : ∀ S R u, u ∈ S → AcceptedH (tabOfZ planZ tabZ S R u).S
  hc : ∀ S R u (h : u ∈ S), (acc S R u h).st.c = c
  vu : ∀ (S : Finset (Fin n)) (_ : Reg) (u : Fin n), u ∈ S → ℕ
  hvu : ∀ S R u (h : u ∈ S), vu S R u h < c.g
  o : ∀ S R u, u ∈ S → Fin (2 ^ (tabOfZ planZ tabZ S R u).S.nbl)
  col : ∀ S R u, u ∈ S → Fin (vuRows c hs).w → Fin (2 ^ (tabOfZ planZ tabZ S R u).S.kLog)
  hcol : ∀ S R u (h : u ∈ S) (x : Fin (vuRows c hs).w), x.val < vuB c hs →
    (col S R u h x).val = vuInPos c (vu S R u h) x.val
  placed : ∀ S R u (h : u ∈ S), Placement (tabOfZ planZ tabZ S R u).S (vuRows c hs) (col S R u h)

/-- **Each drawn unit's place in its `--zk` table**: the VU's rows placed by `Q`, the program's own instance, aliased
as Δ copies (`vuProg_aliased`). -/
noncomputable def VuPlaceZ.place (Q : VuPlaceZ c hs planZ tabZ) (outs : Finset (Fin (vuN c hs n))) :
    ∀ S R u, u ∈ S → UnitPlace ((vuProg c hs n).partition outs) (tabOfZ planZ tabZ S R u).S u :=
  fun S R u h =>
    { R := (vuProg c hs n).rows u
      inst := Const.Prog.instC _ outs u
      o := Q.o S R u h
      col := fun x => Q.col S R u h (Fin.cast (congrArg Rows.w (rows_vuProg c hs n u)) x)
      placed := Placed.placement_rows (rows_vuProg c hs n u) (Q.placed S R u h)
      aliased := vuProg_aliased (Q.acc S R u h) (Q.hc S R u h) (Q.hvu S R u h) (Q.col S R u h) (Q.hcol S R u h) outs u
        (Q.o S R u h) }

end place

variable {K : Type} [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ} {D Sl : Type} [Inhabited D] [Inhabited Sl]
  (A : Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) {n : ℕ} {Reg : Type} {I : Inputs}
  (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (y₀ : DrawSetupZKH I mPts S₀ (dj S₀)) {c : Flock.Circuit}
  (ht : (zkInputs I).tags.typed = false)
  (hc : HmOut.parse (zkInputs I).tags (zkInputs I).circuitFile (zkInputs I).tables = .ok c)

/-! ## The obligation -/

include ht hc in
/-- **The VU's rows are placed with their inputs on their unit slots at the `--zk` hidden tables**: at a one-table
hidden-output `--zk` session of an untyped circuit at `scopeOkGT` with `sha512/row/v1` rows. -/
theorem vuPlaceZ_nonempty (hs : scopeOkGT c = true) (hv1 : ∀ p ∈ c.ports.toList, p.v2 = false) :
    Nonempty (VuPlaceZ c hs (planAtZKH (Reg := Reg) dj y₀) (tabAtZKH dj y₀)) := sorry

/-- The placement at the `--zk` hidden tables (`vuPlaceZ_nonempty`). -/
noncomputable def vuPlaceZG (hs : scopeOkGT c = true) (hv1 : ∀ p ∈ c.ports.toList, p.v2 = false) :
    VuPlaceZ c hs (planAtZKH (Reg := Reg) dj y₀) (tabAtZKH dj y₀) :=
  Classical.choice (vuPlaceZ_nonempty dj y₀ ht hc hs hv1)

/-! ## The count curve -/

/-- **#874's compiled `--zk` count curve at the hidden-output tables, at the VU's program.**
`zk_flock_countC_execOS_crH` at `vuProg`'s partition (`vuOutsOf`), each drawn unit's place `vuPlaceZG`'s: the units
counted wrong are VUs whose returned outputs are not their rows' evaluation on their row words, for an untyped circuit
at `scopeOkGT` (cut inputs a wire copies from the VU's own stages and lookup slots, or a leaf cut copies from a row
word; the outputs from the unit or the tail's last stage). -/
theorem zk_flock_countG (hs : scopeOkGT c = true) (hv1 : ∀ p ∈ c.ports.toList, p.v2 = false) (hA : A.Correct)
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    {Dh : Type} {Hc : List UInt8 → Dh}
    (vb : ValueBindingZ A ((vuProg c hs n).partition (vuOutsOf c hs n)) (planAtZKH dj y₀) (tabAtZKH dj y₀)
      (decodeOfPlace A ((vuProg c hs n).partition (vuOutsOf c hs n)) (planAtZKH dj y₀) (tabAtZKH dj y₀)
        ((vuPlaceZG dj y₀ ht hc hs hv1).place (vuOutsOf c hs n))) Hc)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planAtZKH (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K : ℕ)
    (σ : Strategy (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (batchedSessionZC A Hs (planAtZKH dj y₀))))
    {qF qS : ℝ}
    (hCR : ∀ ω j, TableCRZ A Hs
      (planAtZKH dj y₀ ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω) (reg σ))
      j (zcStrat A Hs (planAtZKH dj y₀) (cont σ ω)) qF qS)
    {execAccept : Bool × Finset (Fin n) → Prop}
    (tr : StrictCR.TreeRewind Hs.H (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (batchedSessionZC A Hs (planAtZKH dj y₀))) σ fun o => execAccept o ∧ o.1 = false)
    {qT : ℝ} (hT : tr.finder.CR Hs.H qT) :
    prob (fun o => execAccept o ∧
        K ≤ (((vuProg c hs n).partition (vuOutsOf c hs n)).wrong
          (XplurZC A Hs ((vuProg c hs n).partition (vuOutsOf c hs n)) (planAtZKH dj y₀) (tabAtZKH dj y₀) vb (reg σ)
            (cont σ) k Rw)).card)
        (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
          (batchedSessionZC A Hs (planAtZKH dj y₀))) σ ≤
      (Law.stratified σs ks hks).miss K +
        ksAvgStrictZ (planAtZKH dj y₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg σ) +
        linkBoundZC A Hs ((vuProg c hs n).partition (vuOutsOf c hs n)) (planAtZKH dj y₀) (tabAtZKH dj y₀) vb k Rw M ρ
          t' (reg σ) (cont σ) +
        ENNReal.ofReal (tr.bound qT) :=
  zk_flock_countC_execOS_crH A Hs ((vuProg c hs n).partition (vuOutsOf c hs n)) dj y₀ hA σs ks hks St hS Kd pop Lb src
    hA3 ((vuPlaceZG dj y₀ ht hc hs hv1).place (vuOutsOf c hs n)) vb hk hRw hM hρ hr ht' K σ hCR tr hT

end FlockSoundness.Discharge.ScopeG
