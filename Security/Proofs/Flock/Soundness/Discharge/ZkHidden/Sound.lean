import Proofs.Flock.Soundness.Discharge.ZkHidden.SoundBind
import Proofs.Flock.Soundness.Discharge.ZkHidden.SoundRuns
import Proofs.Flock.Soundness.Discharge.Composed.Defs

/-!
# The soundness of one hidden-output `--zk` sampled session

`zk_session_soundH`: `Composed.zk_session_sound` for the sessions production `--zk` writes, whose tables
`Stmt.setupTables` sets up with `Stmt.setupHidden` (hidden outputs, the default `--statement`, PROTOCOL §16.13), at the
same law and parameters, with the same proof. For any live prover `τ` of a hidden-output one-table `--zk` session,
`Pr[flock-verify --zk accepts ∧ at least K wrong units] ≤ miss(K) + ε_ks + δ_link`, the units those of the key program's
partition (`keyProg c hU n`), each read at its `--zk` site in its hidden-output table and bound to the rows it commits,
input or output (`placeZH`, `rsZH`: HiddenExec's `keyLayoutH` and `keyRowsH` moved to the `--zk` tables): the compiled
`--zk` count curve at `planAtZKH` (`zk_flock_countC_execOS_atH`) at the simulated prover `simZ … τ`, with `δ_tree = 0`,
carried to flock-verify's acceptance by `zk_headline_exec_on`. `RunsZC` at the law's draws is `runsZC_liveH`, from
`TablesAtZKHLive` (`tablesAtZKH_live`), the tags (`TagsOkH`), the challenger's record of the live session
(`RecordsLiveZK`) and its draw at each of the law's draws (`DrawOkZKH`).

What it rests on: `cr/sha-512` (`hCR`), `ecr/sha-512` (A2, inside `δ_link`), A3 (`hA3`), A6 and the challenger
(`RecordsLiveZK`, and `DrawOkZKH` at each of the law's draws), as `zk_session_sound`. Its circuit hypotheses are the
hidden-output path's: `HmOut.parse` and `HmOut.loadPublic` for `HmRow.parse` and `HmRow.loadPublic`; the unit
returning the outputs comes from `scopeOk`. flock-verify refuses a statement failing any of them (PROTOCOL.md §16.15), and
`FailClosed.zk_session_soundH` takes none.

`tagsOkH_circuit`: the citation of its tag hypotheses. Under the default `--statement` (`verity/flock-circuit`,
`Tags.circuit`) with one table, `TagsOkH I` and `I.tags.typed = false` hold by `decide`.
-/

namespace FlockSoundness.Discharge.ZkHidden

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind HiddenExec HiddenDec
  Composed Placed StrictCR DecodeZL
open scoped ENNReal

/-- **The tags at the default statement**: a verify command under `--statement verity/flock-circuit` (`Tags.circuit`,
hidden outputs) with one table meets `zk_session_soundH`'s `TagsOkH I` and `I.tags.typed = false`, by `decide`. -/
theorem tagsOkH_circuit (I : Inputs) (hI : I.tags = Tags.circuit) (hn : I.count = 1) :
    TagsOkH I ∧ I.tags.typed = false := by
  unfold TagsOkH
  rw [hI, hn]
  decide

variable {mPts n : ℕ} {Reg : Type}

/-- **One hidden-output `--zk` sampled session is sound.** -/
theorem zk_session_soundH
    (I : Inputs) (ht : I.tags.typed = false)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (y₀ : DrawSetupZKH I mPts S₀ (dj S₀))
    (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the session's circuit and public file
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub) (hpt : pub.tables = none)
    (hscope : Layout.scopeOkH c = true)
    -- the law and A3
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the challenger's draw at each of the law's draws
    (hDraw : ∀ ω, DrawOkZKH I dj mPts ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω))
    -- the verify command's tags and the challenger's record of the live session
    (hTags : TagsOkH I) (hRec : RecordsLiveZK I dj wr)
    -- the count curve's parameters
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planAtZKH (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K₀ : ℕ)
    -- the live prover, and cr/sha-512 for the knowledge finders of its simulation
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)))
    {qF qS : ℝ}
    (hCR : ∀ ω j, TableCRZ execArith hs512 (planAtZKH dj y₀
        ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
        (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I ht hc (Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_live hTags hRec hDraw)) τ))) j
      (zcStrat execArith hs512 (planAtZKH dj y₀)
        (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I ht hc (Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_live hTags hRec hDraw)) τ) ω)) qF qS) :
    prob (fun o => LiveAcceptsZK I o ∧
        K₀ ≤ (((keyProg c hU n).partition (Layout.outsOf c hU n)).wrong
          (XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planAtZKH dj y₀) (tabAtZKH dj y₀)
            ((rsZH dj y₀ ht hc hscope hU hpub hpt).bindingZ (rsZH_computes dj y₀ ht hc hscope hU hpub hpt))
            (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I ht hc (Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_live hTags hRec hDraw)) τ))
            (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I ht hc (Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_live hTags hRec hDraw)) τ)) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)) τ ≤
      (Law.stratified σs ks hks).miss K₀ +
        ksAvgStrictZ (planAtZKH dj y₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I ht hc (Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_live hTags hRec hDraw)) τ)) +
        linkBoundZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planAtZKH dj y₀) (tabAtZKH dj y₀)
          ((rsZH dj y₀ ht hc hscope hU hpub hpt).bindingZ (rsZH_computes dj y₀ ht hc hscope hU hpub hpt))
          k Rw M ρ t' (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I ht hc (Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_live hTags hRec hDraw)) τ))
          (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I ht hc (Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_live hTags hRec hDraw)) τ)) := by
  have hdec := decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _
    (runsZC_liveH I ht hc (Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_live hTags hRec hDraw)
  have hnever : ∀ o : Bool × Finset (Fin n), ¬ (o.1 = true ∧ o.1 = false) := fun o h => by
    rw [h.1] at h; exact Bool.noConfusion h.2
  have hc' := zk_flock_countC_execOS_atH dj y₀ ht hc hscope hU hpub hpt hs512 σs ks hks St hS Kd pop Lb src hA3 hk
    hRw hM hρ hr ht' K₀ (simZ execArith hs512 _ I wr _ dg _ _ _ _ N hdec τ) hCR (execAccept := fun o => o.1 = true)
    (StrictCR.TreeRewind.ofNever hnever) (ofNever_finderCR hs512.H hnever 0)
  rw [StrictCR.TreeRewind.ofNever_bound, ENNReal.ofReal_zero, add_zero] at hc'
  exact zk_headline_exec_on execArith hs512 _ I wr _ dg _ _ _ _ N hdec τ _ _ hc'

end FlockSoundness.Discharge.ZkHidden
