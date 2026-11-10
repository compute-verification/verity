module

public import Proofs.Flock.Soundness.Assumptions.Zk
public import Proofs.Flock.Soundness.Discharge.Composed.Defs
public import Proofs.Flock.Soundness.Discharge.ZkHidden.Sound
public import Proofs.Flock.Soundness.Discharge.ZkHidden.SoundBind
public import Proofs.Flock.Soundness.Discharge.ZkHidden.SoundDrawOk
public import Proofs.Flock.Soundness.Discharge.ZkHidden.SoundRuns
public import Proofs.Flock.Soundness.Discharge.ZkHidden.Theorems
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt

/-!
# One hidden-output `--zk` sampled session is sound, under the record's custody

`zk_session_soundH_custody`: `zk_session_soundH` with its two premises about the record, `hRec` (`RecordsLiveZK`) and
`hDraw` (`DrawOkZKH` at each of `Law.execOS`'s samples), replaced by one named assumption, `RecordCustodyZK`
(`live-verifier`), at `mPts := mPtsOf c.g (∑ s, ks s)`, the link points flock-verify sets up at each of the sampling design's samples
(`drawOkZKH_of_exec`), as `Composed.zk_session_sound_custody` is `zk_session_sound`'s. It rests on five: `cr/sha-512`
(`hCR`), `ecr/sha-512` (A2, inside `δ_link`), A3 (`hA3`), A6 and `live-verifier` (`hCust`).
-/

@[expose] public section

namespace FlockSoundness.Discharge.ZkHidden

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind HiddenExec HiddenDec
  Composed Placed StrictCR DecodeZL
open scoped ENNReal

variable {n : ℕ} {Reg : Type}

/-- **One hidden-output `--zk` sampled session is sound, under `live-verifier`.** `zk_session_soundH` at the link
points of the sampling design's own samples, `mPtsOf c.g (∑ s, ks s)`, with the record's custody (`RecordCustodyZK`) in place of
`hRec` and `hDraw`: `hRec` is its first half, and `hDraw` at every sample of `Law.execOS` is `drawOkZKH_of_exec` from its
second. -/
theorem zk_session_soundH_custody
    (I : Inputs) (ht : I.tags.typed = false)
    -- the session's circuit and public file
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub) (hpt : pub.tables = none)
    (hscope : Layout.scopeOkH c = true)
    -- the sampling design and A3
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the record, at the link points of the sampling design's samples
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (y₀ : DrawSetupZKH I (mPtsOf c.g (∑ s, ks s)) S₀ (dj S₀))
    (wr : Finset (Fin n) → Reg → LiveOut (mPtsOf c.g (∑ s, ks s)) → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the verify command's tags, and live-verifier: the record and its sample file are the live session's
    (hTags : TagsOkH I) (hCust : Assumptions.Zk.RecordCustodyZK I dj wr (.stratified Kd (Array.ofFn St)) pop)
    -- the count curve's parameters
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planAtZKH (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K₀ : ℕ)
    -- the live prover, and cr/sha-512 for the knowledge finders of its simulation
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg (mPtsOf c.g (∑ s, ks s)) N wr)))
    {qF qS : ℝ}
    (hCR : ∀ ω j, TableCRZ execArith hs512 (planAtZKH dj y₀
        ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
        (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I ht hc (Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_live hTags hCust.1
          (fun ω => drawOkZKH_of_exec I hc σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ))) j
      (zcStrat execArith hs512 (planAtZKH dj y₀)
        (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I ht hc (Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_live hTags hCust.1
          (fun ω => drawOkZKH_of_exec I hc σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ) ω)) qF qS) :
    prob (fun o => LiveAcceptsZK I o ∧
        K₀ ≤ (((keyProg c hU n).partition (Layout.outsOf c hU n)).wrong
          (XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planAtZKH dj y₀) (tabAtZKH dj y₀)
            ((rsZH dj y₀ ht hc hscope hU hpub hpt).bindingZ (rsZH_computes dj y₀ ht hc hscope hU hpub hpt))
            (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I ht hc (Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_live hTags hCust.1
          (fun ω => drawOkZKH_of_exec I hc σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ))
            (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I ht hc (Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_live hTags hCust.1
          (fun ω => drawOkZKH_of_exec I hc σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ)) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg (mPtsOf c.g (∑ s, ks s)) N wr)) τ ≤
      (Law.stratified σs ks hks).miss K₀ +
        ksAvgStrictZ (planAtZKH dj y₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I ht hc (Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_live hTags hCust.1
          (fun ω => drawOkZKH_of_exec I hc σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ)) +
        linkBoundZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planAtZKH dj y₀) (tabAtZKH dj y₀)
          ((rsZH dj y₀ ht hc hscope hU hpub hpt).bindingZ (rsZH_computes dj y₀ ht hc hscope hU hpub hpt))
          k Rw M ρ t' (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I ht hc (Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_live hTags hCust.1
          (fun ω => drawOkZKH_of_exec I hc σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ))
          (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I ht hc (Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_live hTags hCust.1
          (fun ω => drawOkZKH_of_exec I hc σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ)) := by
  exact zk_session_soundH I ht dj y₀ wr dg N hc hU hpub hpt hscope σs ks hks St hS Kd pop Lb src hA3
    (fun ω => drawOkZKH_of_exec I hc σs ks hks St hS Kd pop Lb src dj hCust.2 ω) hTags hCust.1 hk hRw hM hρ hr ht' K₀
    τ hCR

end FlockSoundness.Discharge.ZkHidden
