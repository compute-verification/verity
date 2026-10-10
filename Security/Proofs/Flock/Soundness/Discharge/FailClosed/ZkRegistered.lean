import Proofs.Flock.Soundness.Discharge.ZkReg
import Proofs.Flock.Soundness.Discharge.FailClosed.Scope

/-!
# The registered `--zk` headline, over what flock-verify accepts

`flock-verify verify --zk --registered F` accepts `VerifiesZKR` (`LiveVerifiesZKR`): the setup, registered check and
verdict `ZkReg.AcceptsZKR` models (every table's statement passes `checkRegistered` at F's ports `own`), and the
statement check with the verifier's own registered reads (`ScopeZR`), which also passes a shared-row public file of a
statement with hidden outputs (`ProvedScope.sharedRowsOk`). `zk_session_soundHJR` is `ZkReg.zk_session_soundHJR` over
it with no hypothesis on the statement's form, as `FailClosed.ZkHidden` is for `ZkHidden`: the circuit (`circZ I`), the
public file (`pubZ I`) and the facts the `ZkReg` headline took about them come from the check at each statement it
passes (`hs : ScopeZR I`), and at a statement the check refuses the event is empty. The registered headline reads no
`pub.tables = none` (its rows are `rsZHJS`, with or without shared rows), so it holds at the shared-row statements the
check now passes.

At public outputs the check does not tell the registered reads apart (`ScopeZR.scopeZ`), so there what the verifier
accepts with `--registered F` it accepts without (`verifiesZK_of_R`), and the claims over `VerifiesZK` hold.
-/

namespace FlockSoundness.Discharge.FailClosed

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind HiddenExec HiddenDec
  Composed Placed StrictCR DecodeZL ZkHidden
open FlockSoundness.Discharge.ZkReg (AcceptsZKR LiveAcceptsZKR OwnReadsHJ regPathsHJ rsShHJS rsZHJS rsZHJS_computes
  acceptsZK_of_R checked_of_acceptsHJ)
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

/-- **What `flock-verify verify --zk --registered F` accepts** (`own` F's ports): the setup, registered check and verdict
`AcceptsZKR` models, and the statement check with the verifier's own registered reads. -/
def VerifiesZKR (I : Inputs) (own : Array Flock.Registered.Port) (t : Exec.Transcript) : Prop :=
  AcceptsZKR I own t ∧ ScopeZR I

variable {mPts n : ℕ} {Reg : Type}

/-- `VerifiesZKR` at the live session's transcript. -/
def LiveVerifiesZKR (I : Inputs) (own : Array Flock.Registered.Port) (o : Reg × Finset (Fin n) × Exec.Transcript) :
    Prop :=
  VerifiesZKR I own o.2.2

/-- **At public outputs, `--registered F` accepts no more than `verify --zk`**: the registered check only refuses more,
and the statement check is the same (`ScopeZR.scopeZ`). -/
theorem verifiesZK_of_R {I : Inputs} {own : Array Flock.Registered.Port} {t : Exec.Transcript} (h : VerifiesZKR I own t)
    (hh : I.tags.hiddenOutputs = false) : VerifiesZK I t :=
  ⟨acceptsZK_of_R h.1, h.2.scopeZ hh⟩

theorem ScopeZR.tagsOkZHJ {I : Inputs} (h : ScopeZR I) (hh : I.tags.hiddenOutputs = true) : TagsOkZHJ I :=
  ⟨hh, h.retainRounds, h.merkleLeaf⟩

/-- **One hidden-output `--zk --registered` sampled session of `J` tables that flock-verify accepts is sound, and its
registered reads are checked.** `ZkReg.zk_session_soundHJR` over `VerifiesZKR`, at a public file with or without shared
rows: the first conjunct is its bound, the second the registered check's consequence at every accepted outcome. -/
theorem zk_session_soundHJR
    (I : Inputs) (hh : I.tags.hiddenOutputs = true) (own : Array Flock.Registered.Port)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length mPts → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the law (the live subset draw) and A3
    (kd : ℕ) (hkd : kd ≤ n) {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ)
    (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256) (hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the coin server's draw at each of the law's draws
    (hDraw : ∀ ω, DrawOkZKHJ I dj mPts ((Law.execOS (n := n) src (.subset kd) n).draw ω))
    -- the coin server's record of the live session
    (hRec : RecordsLiveZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr)
    -- the count curve's parameters
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K₀ : ℕ)
    -- the live prover
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    {qF qS : ℝ}
    (hCR : ∀ hs : ScopeZR I,
        let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
          (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
          (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
          (runsZCJH_live I hs.typed (hs.parseH hh) (Layout.scopeOkH_spec hs.built hs.scopeH).2.1 dj y₀ wr _ (hs.tagsOkZHJ hh) hRec hDraw)) τ
        ∀ ω j, TableCRZ execArith hs512 (planZHJ dj y₀ ((Law.execOS (n := n) src (.subset kd) n).draw ω) (reg σ)) j (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) qF qS) :
    prob (fun o => LiveVerifiesZKR I own o ∧ ∀ hs : ScopeZR I,
        let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
          (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
          (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
          (runsZCJH_live I hs.typed (hs.parseH hh) (Layout.scopeOkH_spec hs.built hs.scopeH).2.1 dj y₀ wr _ (hs.tagsOkZHJ hh) hRec hDraw)) τ
        K₀ ≤ (((keyProg (circZ I) hs.built n).partition (Layout.outsOf (circZ I) hs.built n)).wrong
          (XplurZC execArith hs512 ((keyProg (circZ I) hs.built n).partition (Layout.outsOf (circZ I) hs.built n)) (planZHJ dj y₀) (tabZHJ dj y₀)
            ((rsZHJS dj y₀ hs.typed (hs.parseH hh) hs.scopeH hs.built (hs.loadH hh)).bindingZ (rsZHJS_computes dj y₀ hs.typed (hs.parseH hh) hs.scopeH hs.built (hs.loadH hh)))
            (reg σ) (cont σ) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      ⨆ hs : ScopeZR I,
        (let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
          (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
          (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
          (runsZCJH_live I hs.typed (hs.parseH hh) (Layout.scopeOkH_spec hs.built hs.scopeH).2.1 dj y₀ wr _ (hs.tagsOkZHJ hh) hRec hDraw)) τ
        (Law.subset n kd hkd).miss K₀ +
        ksAvgStrictZ (planZHJ dj y₀) (Law.execOS (n := n) src (.subset kd) n) k qF qS (reg σ) +
        linkBoundZC execArith hs512 ((keyProg (circZ I) hs.built n).partition (Layout.outsOf (circZ I) hs.built n)) (planZHJ dj y₀) (tabZHJ dj y₀)
          ((rsZHJS dj y₀ hs.typed (hs.parseH hh) hs.scopeH hs.built (hs.loadH hh)).bindingZ (rsZHJS_computes dj y₀ hs.typed (hs.parseH hh) hs.scopeH hs.built (hs.loadH hh)))
          k Rw M ρ t' (reg σ) (cont σ)) ∧
    ∀ hs : ScopeZR I, ∀ o : Reg × Finset (Fin n) × Exec.Transcript, LiveVerifiesZKR I own o →
      ∀ rd : FlockSoundness.Registered.ReadsX
          (layShHJ (Reg := Reg) dj y₀ hs.typed (hs.parseH hh) hs.scopeH hs.built (pub := pubZ I))
          (rsShHJS dj y₀ hs.typed (hs.parseH hh) hs.scopeH hs.built (hs.loadH hh)),
        OwnReadsHJ I own (circZ I) (pubZ I) o.2.2 rd.isRead rd.port rd.readAt →
          rd.Checked o.1 (regPathsHJ I (circZ I) (pubZ I) o.2.2 rd.port) :=
  ⟨prob_scope (A := LiveAcceptsZKR I own) (fun _ h => h.1) (fun _ h => h.2) _ _ _ _ fun hs =>
      (FlockSoundness.Discharge.ZkReg.zk_session_soundHJR I hs.typed own dj y₀ wr dg N (hs.parseH hh) hs.built
        (hs.loadH hh) hs.scopeH kd hkd Lb src hA3 hDraw (hs.tagsOkZHJ hh) hRec hk hRw hM hρ hr ht' K₀ τ (hCR hs)).1,
    fun hs o hv rd hown =>
      checked_of_acceptsHJ I hs.typed own dj y₀ (hs.parseH hh) hs.built (hs.loadH hh) hs.scopeH (hs.tagsOkZHJ hh) o hv.1
        rd hown⟩

end FlockSoundness.Discharge.FailClosed
