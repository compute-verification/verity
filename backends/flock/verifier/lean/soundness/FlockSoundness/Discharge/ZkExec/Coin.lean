import FlockSoundness.Discharge.ZkExec.Compiled
import FlockSoundness.Discharge.ZkExec.Shape
import FlockSoundness.Discharge.PlaysCoin.Decodes
import FlockSoundness.Protocol.Discharge.ZkExec.Coin

/-!
# The coin half of `DecodesZK` at the compiled `--zk` table

`DecodesZK` (R11b at `--zk`) at [wave2-zk-link]'s compiled table, with each round's coin step `coinMsg true`
(`one := fun _ _ _ => true`), from its message half alone. [wave2-plays-coin]'s `decodesOfRunsAt` turns each rep's
`RepRun` along a spine that holds where the run ends (`SpineAt`) into `Decodes`. At the compiled rep that spine is
`repShapeZ` at every result whose Ligerito part is not a reject (`repZCAt_spineAt`), and a table that accepts has no
such result (`accZ_lig`).

* `RunsZC`: the message half, named. On every live outcome flock-verify `--zk` accepts, each rep has a `RepRun` on its
  rounds along `repShapeZ`, with `fin`'s messages as the tail, and the compiled table accepts at the two runs' ends.
* `decodesZK_of_runsZC`: `RunsZC` gives `DecodesZK` at the compiled table.
* `RunsOnZC`, `decodes_of_runsOnZC`: the same at the law's draws only, for a decoder that is right only where the
  coin server's draws are (`DrawOkZK` at `L.draw ω`).
* `zk_headline_exec_zc_runs`: `zk_headline_exec_zc_execOS` with `RunsZC` in place of `hdec`.
-/

namespace FlockSoundness.Discharge.ZkExec

open Game Audit Exec Refine Model Accounting ZkSession
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

variable {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D] [Inhabited Sl] {mPts n : ℕ} {Reg : Type}
  (A : Arith 𝔽 K) (Hs : ZkLink.Hashes 𝔽 D Sl) (T : Finset (Fin n) → Reg → TabZK 𝔽 mPts)

variable (I : Inputs) (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
  (opn : Finset (Fin n) → Reg → Array ByteArray → ZkLink.OpensZ 𝔽 D Sl) (dg : ByteArray → ByteArray)
  (dec : Finset (Fin n) → Reg → ℕ → ByteArray → List Msg)
  (fin : Finset (Fin n) → Reg → Fin 2 → Array ByteArray → List Msg)
  (capOf : Finset (Fin n) → Reg → ByteArray → Fin 1 → Cap D) (N : ℕ)

omit [Fintype K] in
/-- **`DecodesZK` from its message half** at the compiled `--zk` table, with `one := fun _ _ _ => true`. -/
theorem decodesZK_of_runsZC (h : RunsZC A Hs T I wr opn dec fin capOf) :
    DecodesZK (repZ A Hs T) (outZ Hs T) (accZ T) I wr opn dec (fun _ _ _ => true) fin capOf := by
  intro S R
  refine PlaysCoin.decodesOfRunsAt (outZ Hs T S R) _ _ (opn S R) (dec S R) (fin S R) (capOf S R) (repZ A Hs T S R)
    fun o ho => ?_
  obtain ⟨pend, h₀, h₁, hacc⟩ := h S R o ho
  obtain ⟨hl₀, hl₁⟩ := accZ_lig Hs T hacc
  exact ⟨pend, _, _, repZCAt_spineAt A Hs (T S R) o.pts 0 hl₀, h₀, repZCAt_spineAt A Hs (T S R) o.pts 1 hl₁, h₁,
    hacc⟩

variable {C : Circuit} (P : Partition C n) (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planOne T S R).length)
  {decode : DecoderZK (C := C) (planOne T) tab} {Dh : Type} {Hc : List UInt8 → Dh}
  (vb : ZkLink.ValueBindingZ A P (planOne T) tab decode Hc)

/-- **The `--zk` headline at flock-verify's acceptance, at the live draw under A3, from the message half**:
`zk_headline_exec_zc_execOS` with `hrun : RunsZC …` in place of `hdec`, through `decodesZK_of_runsZC`. -/
theorem zk_headline_exec_zc_runs (hlow : LoweringSoundZK A P (planOne T) tab decode)
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    {k Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1) (hr : ∀ S R j, ZkLink.rateZ (planOne T S R) j k ≤ ρ)
    {a : (S : Finset (Fin n)) → (R : Reg) → Strategy (ZkLink.sessionZC A Hs (planOne T S R)) →
      Fin (planOne T S R).length → ℝ}
    (hj : ZkLink.JointZC A Hs (planOne T) k a) {t' : ℝ} (ht : 0 ≤ t') (K : ℕ)
    (hrun : RunsZC A Hs T I wr opn dec fin capOf)
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (liveOf dg mPts N wr))) :
    let σ := simAuditZC A Hs T I wr opn dg dec (fun _ _ _ => true) fin capOf N
      (decodesZK_of_runsZC A Hs T I wr opn dec fin capOf hrun) τ
    prob (fun o => LiveAcceptsZK I o ∧
        K ≤ (P.wrong (ZkLink.XplurZC A Hs P (planOne T) tab vb (reg σ) (cont σ) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)) τ ≤
      (Law.stratified σs ks hks).miss K + ZkLink.ksAvgZC A Hs (planOne T) k a (reg σ) (cont σ) +
        ZkLink.linkBoundZC A Hs P (planOne T) tab vb k Rw M ρ t' (reg σ) (cont σ) :=
  zk_headline_exec_zc_execOS A Hs T I wr opn dg dec _ fin capOf N P tab vb hlow σs ks hks St hS Kd pop Lb src hA3 hRw
    hM hρ hr hj ht K (decodesZK_of_runsZC A Hs T I wr opn dec fin capOf hrun) τ

end FlockSoundness.Discharge.ZkExec
