import Proofs.Flock.Soundness.Discharge.ZkPubIn.RunsJ
import Proofs.Flock.Soundness.Discharge.ZkPubIn.Count

set_option Elab.async false

/-!
# One canonical hidden-output `--zk --public-inputs F` sampled session of `J` tables is sound

`Canonical.zk_session_soundC` with the verifier's public inputs (`own` F's bytes): the event is flock-verify's
acceptance `AcceptsZKP I own`, the tables are `planZHJP`, the key program is `keyProgP` (a gate per public column), and
the committed transcript is `Xpub pubOnes pubZeros XplurZC`, the verifier's own bits on the public gates and the
plurality elsewhere. The bound is `zk_session_soundC`'s: the subset law's miss, the strict knowledge-soundness average
under `cr/sha-512` (`TableCRZ`), and the compiled link term, with A3 for the law's bytes.

The proof is `zk_session_soundC`'s at `zk_headline_exec_tablesA` (`ComposedJ`), `runsZCJH_liveCP` (`RunsJ`), the rows
at `keyProgP` (`rsZCP`, `Bind`) and the count curve at the public transcript (`zk_flock_countC_pub`), whose one new
hypothesis, the public gates decoding to the verifier's own bits, is `pubDecodesZ_placeZCP` (`Count`).
-/

namespace FlockSoundness.Discharge.ZkPubIn

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Placed DecodeZL
  HiddenExec Composed ZkHidden StrictCR
open FlockSoundness.Discharge.Tables (LiveOutJ)
open FlockSoundness.Discharge.Canonical (formatOk)
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

variable {n mPts : ℕ} {Reg : Type}

/-- **One canonical hidden-output `--zk --public-inputs F` sampled session of `J` tables is sound**
(`Canonical.zk_session_soundC` at `AcceptsZKP I own`, `keyProgP` and the public transcript
`Xpub pubOnes pubZeros XplurZC`): any number of input rows and groups, rows of any length, output rows of runs of output
ports, shared public tables, and public input ports whose words the verifier supplies. -/
theorem zk_session_soundCP
    (I : Inputs) (own : ByteArray) (ht : I.tags.typed = false)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (y₀ : DrawSetupZHJP (inputsJ I 0) own mPts S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZHJP (Reg := Reg) dj y₀ S R).length mPts →
      Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the session's circuit, in the canonical format, its public input ports and public file
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    (hfmt : formatOk c = true) {pi : HmIn.Ports} (hpi : HmIn.parse c = .ok pi) {pub : Public}
    (hpub : HmIn.loadPublic (zkTags I) c pi I.publicFile = .ok pub)
    -- the law (the live subset draw) and A3
    (kd : ℕ) (hkd : kd ≤ n) {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ)
    (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256) (hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the coin server's draw at each of the law's draws
    (hDraw : ∀ ω, DrawOkZKHJP I own dj mPts ((Law.execOS (n := n) src (.subset kd) n).draw ω))
    -- the verify command's tags and the coin server's record of the live session
    (hTags : TagsOkZHJ I)
    (hRec : RecordsLiveZKJP I own dj (fun S R => (planZHJP (Reg := Reg) dj y₀ S R).length) wr)
    -- the count curve's parameters
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZHJP (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K₀ : ℕ)
    -- the live prover
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJP dj y₀) N wr)))
    {qF qS : ℝ} :
    let σ := simZJA execArith hs512 (planZHJP dj y₀) (AcceptsZKP I own) wr (opnAtZHJP (planZHJP dj y₀) I own dj) dg
      (decAtZJ (planZHJP dj y₀)) (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnZKJA_of_runsOnZCJA execArith hs512 _ _ wr _ _ _ _ _
        (runsZCJH_liveCP I own ht hc hfmt dj y₀ wr _ hTags hRec hDraw)) τ
    -- cr/sha-512 for the knowledge finders of its simulation
    (∀ ω j, TableCRZ execArith hs512 (planZHJP dj y₀ ((Law.execOS (n := n) src (.subset kd) n).draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planZHJP dj y₀) (cont σ ω)) qF qS) →
    prob (fun o => LiveAcceptsZKP I own o ∧
        K₀ ≤ (((keyProgP c hU n pi).partition (Layout.outsOf c hU n)).wrong
          (Xpub (pubOnes c hU n pi own) (pubZeros c hU n pi own)
            (XplurZC execArith hs512 ((keyProgP c hU n pi).partition (Layout.outsOf c hU n)) (planZHJP dj y₀)
              (tabZHJP dj y₀)
              ((rsZCP dj y₀ ht hc hfmt hpi hpub hU).bindingZ (rsZCP_computes dj y₀ ht hc hfmt hpi hpub hU))
              (reg σ) (cont σ) k Rw))).card)
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJP dj y₀) N wr)) τ ≤
      (Law.subset n kd hkd).miss K₀ +
        ksAvgStrictZ (planZHJP dj y₀) (Law.execOS (n := n) src (.subset kd) n) k qF qS (reg σ) +
        linkBoundZC execArith hs512 ((keyProgP c hU n pi).partition (Layout.outsOf c hU n)) (planZHJP dj y₀)
          (tabZHJP dj y₀)
          ((rsZCP dj y₀ ht hc hfmt hpi hpub hU).bindingZ (rsZCP_computes dj y₀ ht hc hfmt hpi hpub hU))
          k Rw M ρ t' (reg σ) (cont σ) := by
  intro σ hCR
  have hdec := decodesOnZKJA_of_runsOnZCJA execArith hs512 _ (AcceptsZKP I own) wr _ _ _ _ _
    (runsZCJH_liveCP I own ht hc hfmt dj y₀ wr _ hTags hRec hDraw)
  have hc' := (zk_flock_countC_pub execArith hs512 ((keyProgP c hU n pi).partition (Layout.outsOf c hU n))
    (planZHJP dj y₀) (tabZHJP dj y₀)
    ((rsZCP dj y₀ ht hc hfmt hpi hpub hU).bindingZ (rsZCP_computes dj y₀ ht hc hfmt hpi hpub hU))
    (pubOnes c hU n pi own) (pubZeros c hU n pi own) (pubDecodesZ_placeZCP dj y₀ ht hc hfmt hpi hpub hU)
    (loweringSoundZK_of_place execArith _ (planZHJP dj y₀) (tabZHJP dj y₀) (placeZCP dj y₀ ht hc hfmt hpi hU))
    hRw hM hρ hr (jointZC_ksBound execArith execArith_correct hs512 (planZHJP dj y₀) hk) ht' K₀ σ).trans
    (add_le_add (add_le_add (Tables.execOS_subset_miss_le kd hkd Lb src hA3 K₀)
      (ksAvgZC_le_strict execArith hs512 (planZHJP dj y₀) k (reg σ) (cont σ) hCR)) le_rfl)
  exact zk_headline_exec_tablesA execArith hs512 (planZHJP dj y₀) (AcceptsZKP I own) wr
    (opnAtZHJP (planZHJP dj y₀) I own dj) dg (decAtZJ (planZHJP dj y₀)) (fun _ _ _ _ => true) (fun _ _ _ _ _ => [])
    (fun _ _ rs j => capOfRoot rs[j.val]!) N hdec τ _ _ hc'

end FlockSoundness.Discharge.ZkPubIn
