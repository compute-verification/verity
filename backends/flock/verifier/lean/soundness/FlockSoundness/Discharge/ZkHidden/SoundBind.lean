import FlockSoundness.Discharge.ZkHidden.Count
import FlockSoundness.Discharge.ZkBind.Shadow
import FlockSoundness.Discharge.ZkBind.At
import FlockSoundness.Discharge.HiddenExec.HeadlineExec
import FlockSoundness.Protocol.Discharge.ZkHidden.SoundBind

set_option Elab.async false

/-!
# The sites, the layout and the rows at the hidden-output `--zk` tables

lp-zkbind's `sitesZAt`, `layZAt`, `rsZAt` and `rsZAt_computes` (`ZkBind/At.lean`) at the hidden-output tables
`tablesAtZKH dj y₀`, as HiddenExec's `sitesAtH`, `layAtH`, `rsAtH` and `rsAtH_computes` are at the public path's
`tablesAtH`:

* `tableShH`: the public hidden-output table (`HiddenExec.TableH`) at the same draw: the `DrawSetupZKH`'s
  `DrawSetupH` at `zkInputs I` and its `table`, or `y₀`'s refused one. Its statement is the `--zk` table's
  (`tableAtZKH_S`, both `stmtOf` of setupHidden's statement at the same regions).
* `sitesShH`, `layShH`, `rsShH`, `rsShH_computes`: HiddenExec's `keyLayoutH` and `keyRowsH` at `tableShH`'s sites.
* `placeZH`: each drawn unit's place in its `--zk` table, `keyPlaceH`'s along `tableAtZKH_S`; `decAgreeH`: the two
  decoders read one witness alike.
* `layZH`, `rsZH`, `rsZH_computes`: the public layout and rows moved to the `--zk` tables (`LayoutZ.ofShadow`,
  `HmRowsZ.ofShadow`).
* `zk_flock_countC_execOS_atH`: #874's count curve at `planAtZKH` (`zk_flock_countC_execOS_crH`) with those.
-/

namespace FlockSoundness.Discharge.ZkHidden

open Flock FlockLevel3 Game Model Accounting Refine Audit ZkSession ZkLink ZkExec ZkBind HiddenExec Placed
open FlockSoundness.Discharge.Exec (Inputs)
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

/-! ## A unit's place along an equality of statements -/

/-! ## The public hidden-output table at each `--zk` table's draw -/

variable {n : ℕ} {I : Inputs} {mPts : ℕ} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
  (y₀ : DrawSetupZKH I mPts S₀ (dj S₀)) {Reg : Type}

variable (ht : I.tags.typed = false) {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c)
  (hscope : Layout.scopeOk c = true)

variable (hU : Built c.unit) {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)
  (hpt : pub.tables = none)

theorem rsShH_computes : (rsShH dj y₀ ht hc hscope hU hpub hpt (Reg := Reg)).Computes :=
  keyRowsH_computes (sitesShH_parse dj y₀ ht hc hscope) (Layout.outsOf c hU n) (fun _ => pub)
    (Layout.scopeOk_spec hU hscope).2.1 (Layout.scopeOk_ports hscope) (Layout.outsOut_outsOf c hU n)
    (sitesShH_bcSite dj y₀ ht hc hscope hU hpub hpt)
    (Layout.keysRow_of_parse (parse_untyped (zkTyped ht) hc).1 (Layout.scopeOk_spec hU hscope).1
      (Layout.scopeOk_spec hU hscope).2.2)

/-! ## The `--zk` places, layout and rows -/

/-- **`hHm` at the hidden-output `--zk` tables' rows** (`rsShH_computes`, moved along): in a satisfying witness of a
drawn unit's `--zk` table, each row's `b ‖ c` at the unit's site, input or output, is the commit string of the row's
bytes under the salt at its `hm96` slot. -/
theorem rsZH_computes : (rsZH dj y₀ ht hc hscope hU hpub hpt (Reg := Reg)).Computes :=
  HmRowsZ.ofShadow_computes (planAtZKH_S dj y₀) (decAgreeH dj y₀ ht hc hscope hU) _
    (rsShH_computes dj y₀ ht hc hscope hU hpub hpt)

/-- **The compiled `--zk` count curve at the hidden-output `--zk` tables, its places and binding built**:
`zk_flock_countC_execOS_crH` with `place := placeZH` and `vb := rsZH.bindingZ rsZH_computes`. Past the count curve's
own hypotheses it takes the session's inputs: untyped tags (`ht`), one circuit file `HmOut.parse` accepts as `c` under
the `--zk` tags (`hc`), the circuit in `rs`'s scope (`hscope`), and a public file `HmOut.loadPublic` loads without
shared-row tables (`hpub`, `hpt`). -/
theorem zk_flock_countC_execOS_atH (Hs : Hashes 𝔽 Dig (List UInt8))
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planAtZKH (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K : ℕ)
    (σ : Strategy (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (batchedSessionZC execArith Hs (planAtZKH dj y₀))))
    {qF qS : ℝ}
    (hCR : ∀ ω j, TableCRZ execArith Hs
      (planAtZKH dj y₀ ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω) (reg σ))
      j (zcStrat execArith Hs (planAtZKH dj y₀) (cont σ ω)) qF qS)
    {execAccept : Bool × Finset (Fin n) → Prop}
    (tr : StrictCR.TreeRewind Hs.H (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (batchedSessionZC execArith Hs (planAtZKH dj y₀))) σ fun o => execAccept o ∧ o.1 = false)
    {qT : ℝ} (hT : tr.finder.CR Hs.H qT) :
    prob (fun o => execAccept o ∧ K ≤ (((keyProg c hU n).partition (Layout.outsOf c hU n)).wrong
        (XplurZC execArith Hs ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planAtZKH dj y₀) (tabAtZKH dj y₀)
          ((rsZH dj y₀ ht hc hscope hU hpub hpt).bindingZ (rsZH_computes dj y₀ ht hc hscope hU hpub hpt))
          (reg σ) (cont σ) k Rw)).card)
        (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
          (batchedSessionZC execArith Hs (planAtZKH dj y₀))) σ ≤
      (Law.stratified σs ks hks).miss K +
        ksAvgStrictZ (planAtZKH dj y₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg σ) +
        linkBoundZC execArith Hs ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planAtZKH dj y₀)
          (tabAtZKH dj y₀)
          ((rsZH dj y₀ ht hc hscope hU hpub hpt).bindingZ (rsZH_computes dj y₀ ht hc hscope hU hpub hpt))
          k Rw M ρ t' (reg σ) (cont σ) +
        ENNReal.ofReal (tr.bound qT) :=
  zk_flock_countC_execOS_crH execArith Hs ((keyProg c hU n).partition (Layout.outsOf c hU n)) dj y₀
    execArith_correct σs ks hks St hS Kd pop Lb src hA3 (placeZH dj y₀ ht hc hscope hU)
    ((rsZH dj y₀ ht hc hscope hU hpub hpt).bindingZ (rsZH_computes dj y₀ ht hc hscope hU hpub hpt))
    hk hRw hM hρ hr ht' K σ hCR tr hT

end FlockSoundness.Discharge.ZkHidden
