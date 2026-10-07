import Proofs.Flock.Soundness.Discharge.Composed.Defs

/-!
# The soundness half of one `--zk` sampled session

`zk_session_sound`: `zk_session_composed`'s soundness conjunct, at the same law and parameters, with the same proof.
For any live prover `τ` of a public-output one-table `--zk` session,
`Pr[flock-verify --zk accepts ∧ at least K wrong units] ≤ miss(K) + ε_ks + δ_link`, the units those of the key
program's partition (`keyProg c hU n`), each read at its `--zk` site and bound to the rows it commits (lp-zkbind's
`sitesZAt`, `rsZAt`): `ZkBind.zk_flock_countC_execOS_at` at the simulated prover `simZ … τ`, with `δ_tree = 0`
(`execAccept` is acceptance, so the tree's event is empty: `TreeRewind.ofNever`), carried to flock-verify's acceptance
by `zk_headline_exec_on`. `RunsZC` at the law's draws is `ZkExec.runsZC_live`, from the tags (`TagsOk`), the coin
server's record of the live session (`RecordsLiveZK`) and its draw at each of the law's draws (`DrawOkZK`).

What it rests on: `cr/sha-512` (`hCR`), `ecr/sha-512` (A2, inside `δ_link`: `⊤` at a prover whose link finders break
it), A3 (`hA3`), A6 and the challenger (`RecordsLiveZK`, and `DrawOkZK` at each of the law's draws). It reads neither
`hash-derived-key` nor `PadNonvanishing`, nor the session's `Coined` tables: those are the zero-knowledge half's.

`tagsOk_circuit967b8d06`: the citation of its tag hypotheses. Under `--statement verity/flock-circuit@967b8d06` with
one table (no `--session-tables`), `TagsOk I` and `I.tags.typed = false` hold by `decide`. Under the default
`--statement` (`verity/flock-circuit`, hidden outputs) and the typed statements they fail, so the bound covers
`flock-verify verify --statement verity/flock-circuit@967b8d06 --zk` sessions of one table.

flock-verify refuses every statement failing one of its hypotheses on the statement's form (`I.tags.typed = false`, the
parse and load, `pub.tables = none`, `scopeOk`, retained rounds and `hm96-sha512/v1` leaves: PROTOCOL.md §16.15), so
`FailClosed.zk_session_sound` states this bound over what it accepts (`LiveVerifiesZK`) without them.
-/

namespace FlockSoundness.Discharge.Composed

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR DecodeZL
open scoped ENNReal

/-- **The tags at the named statement**: a verify command under `--statement verity/flock-circuit@967b8d06` with one
table meets `zk_session_sound`'s `TagsOk I` and `I.tags.typed = false`, by `decide`. -/
theorem tagsOk_circuit967b8d06 (I : Inputs) (hI : I.tags = Tags.circuit967b8d06) (hn : I.count = 1) :
    TagsOk I ∧ I.tags.typed = false := by
  unfold TagsOk
  rw [hI, hn]
  decide

variable {mPts n : ℕ} {Reg : Type}

/-- **One `--zk` sampled session is sound.** -/
theorem zk_session_sound
    (I : Inputs) (ht : I.tags.typed = false)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (y₀ : DrawSetupZK I mPts S₀ (dj S₀))
    (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the session's circuit and public file
    {c : Flock.Circuit} (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmRow.loadPublic (zkTags I) c I.publicFile = .ok pub) (hpt : pub.tables = none)
    (hscope : Layout.scopeOk c = true)
    -- the law and A3
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the challenger's draw at each of the law's draws
    (hDraw : ∀ ω, DrawOkZK I dj mPts ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω))
    -- the verify command's tags and the challenger's record of the live session
    (hTags : TagsOk I) (hRec : RecordsLiveZK I dj wr)
    -- the count curve's parameters
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZAt (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K₀ : ℕ)
    -- the live prover, and cr/sha-512 for the knowledge finders of its simulation
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)))
    {qF qS : ℝ}
    (hCR : ∀ ω j, TableCRZ execArith hs512 (planZAt dj y₀
        ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
        (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ))) j
      (zcStrat execArith hs512 (planZAt dj y₀)
        (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ) ω)) qF qS) :
    prob (fun o => LiveAcceptsZK I o ∧
        K₀ ≤ (((keyProg c hU n).partition (Layout.outsOf c hU n)).wrong
          (XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZAt dj y₀) (tabZAt dj y₀)
            ((rsZAt ht dj y₀ hc hU hpub hpt hscope).bindingZ (rsZAt_computes ht dj y₀ hc hU hpub hpt hscope))
            (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ))
            (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ)) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)) τ ≤
      (Law.stratified σs ks hks).miss K₀ +
        ksAvgStrictZ (planZAt dj y₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ)) +
        linkBoundZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZAt dj y₀) (tabZAt dj y₀)
          ((rsZAt ht dj y₀ hc hU hpub hpt hscope).bindingZ (rsZAt_computes ht dj y₀ hc hU hpub hpt hscope))
          k Rw M ρ t' (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ))
          (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ)) := by
  have hdec := decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)
  have hnever : ∀ o : Bool × Finset (Fin n), ¬ (o.1 = true ∧ o.1 = false) := fun o h => by
    rw [h.1] at h; exact Bool.noConfusion h.2
  have hc' := zk_flock_countC_execOS_at ht dj y₀ hc hU hpub hpt hscope hs512 σs ks hks St hS Kd pop Lb src hA3 hk hRw hM
    hρ hr ht' K₀ (simZ execArith hs512 _ I wr _ dg _ _ _ _ N hdec τ) hCR (execAccept := fun o => o.1 = true)
    (StrictCR.TreeRewind.ofNever hnever) (ofNever_finderCR hs512.H hnever 0)
  rw [StrictCR.TreeRewind.ofNever_bound, ENNReal.ofReal_zero, add_zero] at hc'
  exact zk_headline_exec_on execArith hs512 _ I wr _ dg _ _ _ _ N hdec τ _ _ hc'

end FlockSoundness.Discharge.Composed
