import FlockSoundness.Discharge.Canonical.Rows
import FlockSoundness.Discharge.Canonical.Sites
import FlockSoundness.Discharge.ZkHidden.CustodyJJson
import FlockSoundness.Protocol.Discharge.Canonical.Session
import FlockSoundness.Discharge.Canonical.Commit
import FlockSoundness.Discharge.Canonical.Shared

set_option Elab.async false

/-!
# One canonical hidden-output `--zk` sampled session of `J` tables

`ZkHidden/DefsJ.lean`'s shadow and live pieces, `SoundJ.lean`'s `zk_session_soundHJ`, and `CustodyJ.lean` /
`CustodyJJson.lean`'s custody at a circuit in the canonical format (`formatOk c = true`) in place of
`Layout.scopeOk c = true` and `pub.tables = none`:

* `drawSetupH_of_setupHiddenC`: `drawSetupH_of_setupHidden` without `outNet = unitNet` (`0 < c.g` is `formatOk`'s
  `vusPow2`, `formatOk_g_pos`);
* `layShC`, `rsShC`, `rsShC_computes`, `placeZC`, `decAgreeC`, `layZC`, `rsZC`, `rsZC_computes`: `keyLayoutC` and
  `keyRowsC` (`Rows.lean`) at `sitesShC` (`Sites.lean`), moved to the `--zk` tables as `layZHJ`, `rsZHJ`;
* `runsZCJH_liveC`: `runsZCJH_live` without `outNet = unitNet`;
* `zk_session_soundC`, `zk_session_soundC_custody_json`: the session's soundness, `zk_session_soundHJ`'s proof at these
  pieces, and at the record's custody and the printed draw file.

**Proof work.** None: every declaration is proved, and so is every lemma it uses from `Format`, `Chain`, `Commit`,
`Reads`, `Outputs`, `Sites` and `Rows`. `zk_session_soundC`'s bound and its assumptions (A3, `cr/sha-512` through
`TableCRZ`, the count curve's parameters) are `zk_session_soundHJ`'s; only the circuit's condition changes.
-/

namespace FlockSoundness.Discharge.Canonical

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Placed DecodeZL
  HiddenExec Composed ZkHidden StrictCR
open FlockSoundness.Discharge.Tables (LiveOutJ)
open FlockSoundness.Discharge.DecodeZLJ (partJson)
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

section Accepted

open FlockSoundness.Discharge.Integrate (model_sched fits_blocks)

variable {I : Inputs} {mPts n : ℕ} {S : Finset (Fin n)} {d : Lean.Json} {c : Flock.Circuit}
  (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) (hfmt : formatOk c = true)
include hc hfmt

end Accepted

variable {n mPts : ℕ} {Reg : Type}

section Shadow

variable {I : Inputs} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
  (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀)
  (ht : I.tags.typed = false) {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c)
  (hfmt : formatOk c = true) (hU : Built c.unit) {pub : Public}
  (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)

end Shadow

/-- **One canonical hidden-output `--zk` sampled session of `J` tables is sound** (`zk_session_soundHJ` with
`formatOk c = true` for `Layout.scopeOk c = true` and `pub.tables = none`): any number of input rows and groups, rows
of any length, output rows of runs of output ports, shared public tables. -/
theorem zk_session_soundC
    (I : Inputs) (ht : I.tags.typed = false)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length mPts → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the session's circuit, in the canonical format, and public file
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub) (hfmt : formatOk c = true)
    -- the law (the live subset draw) and A3
    (kd : ℕ) (hkd : kd ≤ n) {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ)
    (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256) (hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the coin server's draw at each of the law's draws
    (hDraw : ∀ ω, DrawOkZKHJ I dj mPts ((Law.execOS (n := n) src (.subset kd) n).draw ω))
    -- the verify command's tags and the coin server's record of the live session
    (hTags : TagsOkZHJ I) (hRec : RecordsLiveZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr)
    -- the count curve's parameters
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K₀ : ℕ)
    -- the live prover
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    {qF qS : ℝ} :
    let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
        (runsZCJH_liveC I ht hc hfmt dj y₀ wr _ hTags hRec hDraw)) τ
    -- cr/sha-512 for the knowledge finders of its simulation
    (∀ ω j, TableCRZ execArith hs512 (planZHJ dj y₀ ((Law.execOS (n := n) src (.subset kd) n).draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) qF qS) →
    prob (fun o => LiveAcceptsZK I o ∧
        K₀ ≤ (((keyProg c hU n).partition (Layout.outsOf c hU n)).wrong
          (XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
            ((rsZC dj y₀ ht hc hfmt hU hpub).bindingZ (rsZC_computes dj y₀ ht hc hfmt hU hpub))
            (reg σ) (cont σ) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      (Law.subset n kd hkd).miss K₀ +
        ksAvgStrictZ (planZHJ dj y₀) (Law.execOS (n := n) src (.subset kd) n) k qF qS (reg σ) +
        linkBoundZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
          ((rsZC dj y₀ ht hc hfmt hU hpub).bindingZ (rsZC_computes dj y₀ ht hc hfmt hU hpub))
          k Rw M ρ t' (reg σ) (cont σ) := by
  intro σ hCR
  refine zk_headline_exec_tables execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg
    (decAtZJ (planZHJ dj y₀)) (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
    (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
      (runsZCJH_liveC I ht hc hfmt dj y₀ wr _ hTags hRec hDraw)) τ _ _ ?_
  refine (zk_flock_countC execArith hs512 _ (planZHJ dj y₀) (tabZHJ dj y₀) _
    (loweringSoundZK_of_place execArith _ (planZHJ dj y₀) (tabZHJ dj y₀) (placeZC dj y₀ ht hc hfmt hU))
    hRw hM hρ hr (jointZC_ksBound execArith execArith_correct hs512 (planZHJ dj y₀) hk) ht' K₀ σ).trans ?_
  exact add_le_add (add_le_add (Tables.execOS_subset_miss_le kd hkd Lb src hA3 K₀)
    (ksAvgZC_le_strict execArith hs512 (planZHJ dj y₀) k (reg σ) (cont σ) hCR)) le_rfl

omit mPts in
/-- **One canonical hidden-output `--zk` sampled session of `J` tables is sound, at the record's custody and the
printed draw file** (`zk_session_soundHJ_custody_json` at `formatOk c = true`): `RecordsLiveZKJ`, each draw file the
JSON value `flock-verify draw` printed for the verifier's own draw under `subset kd`, at `mPtsOf c.g (kd / nTab I)`
link points and `nTab I ∣ kd`. -/
theorem zk_session_soundC_custody_json
    (I : Inputs) (ht : I.tags.typed = false)
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub) (hfmt : formatOk c = true)
    (kd : ℕ) (hkd : kd ≤ n) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    {d₀ : Lean.Json} (y₀ : DrawSetupZHJ (inputsJ I 0) (mPtsOf c.g (kd / nTab I)) S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length (mPtsOf c.g (kd / nTab I)) → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ)
    (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256) (hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the verify command's tags, the coin server's custody of the record, and its draw file as `flock-verify draw`
    -- printed the verifier's draw
    (hTags : TagsOkZHJ I)
    (hRec : RecordsLiveZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr)
    (hdj : ∀ S, dj S = (drawOf (.subset kd) n S).toJson)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K₀ : ℕ)
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    {qF qS : ℝ} :
    let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
        (runsZCJH_liveC I ht hc hfmt dj y₀ wr _ hTags hRec
          fun _ => drawOkZKHJ_of_file I ht hc kd n hdiv dj (drawFileZK_of_toJson_subset dj kd n hdj) _)) τ
    (∀ ω j, TableCRZ execArith hs512 (planZHJ dj y₀ ((Law.execOS (n := n) src (.subset kd) n).draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) qF qS) →
    prob (fun o => LiveAcceptsZK I o ∧
        K₀ ≤ (((keyProg c hU n).partition (Layout.outsOf c hU n)).wrong
          (XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
            ((rsZC dj y₀ ht hc hfmt hU hpub).bindingZ (rsZC_computes dj y₀ ht hc hfmt hU hpub))
            (reg σ) (cont σ) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      (Law.subset n kd hkd).miss K₀ +
        ksAvgStrictZ (planZHJ dj y₀) (Law.execOS (n := n) src (.subset kd) n) k qF qS (reg σ) +
        linkBoundZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
          ((rsZC dj y₀ ht hc hfmt hU hpub).bindingZ (rsZC_computes dj y₀ ht hc hfmt hU hpub))
          k Rw M ρ t' (reg σ) (cont σ) :=
  zk_session_soundC I ht dj y₀ wr dg N hc hU hpub hfmt kd hkd Lb src hA3
    (fun _ => drawOkZKHJ_of_file I ht hc kd n hdiv dj (drawFileZK_of_toJson_subset dj kd n hdj) _) hTags hRec hk hRw
    hM hρ hr ht' K₀ τ

end FlockSoundness.Discharge.Canonical
