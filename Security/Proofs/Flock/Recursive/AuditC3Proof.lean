import Proofs.Flock.Recursive.AuditC3
import Proofs.Flock.Recursive.FirewallZK
import Proofs.Flock.Recursive.Loads
import Proofs.Flock.C3.Fork.Sound

/-!
# The recursive audit at candidate 3: its proof

`Flock.SecurityProofs.RecursiveAuditC3 : Flock.Guarantees.RecursiveAuditC3`, `PublicAudit`'s proof at candidate 3's
table, and `RecursiveFirewallPrivate`.

In (i), as in `PublicAudit`'s:
* where some index has no session that types, the event is empty (`prob_sessionsR_untypable`);
* at a statement Lean's verifier doesn't build, or inputs at candidate 3 it refuses (`AcceptsC3`: V*'s list, the inner
  transcript against R's registration, check 5's sparse refusals), the event is empty (its first conjunct);
* at one it builds, the class checks pass (`classOk_of_accepts`), and candidate 3's table at the executable's
  parameters is sound in fork form (piece (a), `FlockC3.Fork.holoForkSound_holoAt`), here at the checked sessions
  (`checkedC3`), whose games are the typed sessions', so the transported prover plays them;
* the hidden statement is R's registration (`xR` with the registrant's openings `_hreg`), a constant of the run, and the
  raw event is bounded by the typed sessions' at the transported prover (`prob_recT_le`, `verifies_zsOf`), where the
  event's checks at candidate 3 (`ChecksVR`) keep every session's own list at the whole histories `recGame` plays
  (`acceptsV_checked`);
* `_hob` is read where the event parses a session's file into scope (`loads_of_verifiesV`, `scope_of_accepts`).

`_hstk` is not read here: VBridge is a premise, and `_hstk` is what its discharge at the honest family takes as `htop`
(`FlockC3.Compose.hbr_honest`, at `⟨_, _hstk⟩`), which gives VBridge at the typed sessions and so at the checked ones
(`vbridge_checked`).

(ii) is `RecursiveFirewallPrivate`.
-/

namespace Flock.SecurityProofs

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

open Classical in
/-- **A kept session's own list is its own** where an accepted transcript passes `P`. -/
theorem own_keepOwn {Q : VStmt} {Reg' : Type} (z : Q.Outer Reg')
    (P : FlockSoundness.Discharge.Exec.Inputs → Array Flock.Registered.Port →
      FlockSoundness.Discharge.Exec.Transcript → Prop)
    (hp : ∃ t, AcceptsZKR z.I z.own t ∧ P z.I z.own t) : (Flock.Guarantees.keepOwn z P).own = z.own :=
  ite_eq_left hp

/-- **The checked sessions accept what the typed sessions accept and the verifier's checks at candidate 3 pass**, at a
whole history: at typed sessions, the checks of the accepted transcript keep each session's own list
(`own_keepOwn`). -/
theorem acceptsV_checked {S : ℕ} {Vs : Fin S → VStmt} {Reg' A B D : Type} {I : Inner} {Cm : Type}
    {ish : RecVStar.Inner} {cs : Option Flock.RecVStar.InnerCommits} {coins : Option Flock.RecCoins.Source}
    {bind : Fin S → VStar.Bindings} {rows : Option (Array Flock.Registered.Port)}
    {zr : A → B → List (Cm × I.Coin) → D → (j : Fin S) → (Vs j).OuterR Reg'} {hC : Flock.Premises.RecordCustody zr}
    {h : Typed zr} {R : A} {ω : B} {t : List (Cm × I.Coin)} {dg : ByteArray → ByteArray} {N : ℕ}
    {o : D × VRec Vs Reg' dg N} (hlen : I.r ≤ t.length) (hall : ∀ j, (zr R ω t o.1 j).Typable)
    (hv : AcceptsV (Flock.Guarantees.custodied zr hC h R ω t) o)
    (hchk : Flock.Guarantees.ChecksVR ish cs coins bind rows (zr R ω t) o) :
    AcceptsV (Flock.Guarantees.checkedC3 ish cs coins bind rows zr hC h R ω t) o := by
  intro j
  have hp : ∃ t', AcceptsZKR (Flock.Guarantees.custodied zr hC h R ω t o.1 j).I
      (Flock.Guarantees.custodied zr hC h R ω t o.1 j).own t' ∧
      (I.r ≤ t.length ∧ Flock.Guarantees.ChecksC3 ish cs coins j (bind j) rows
        (Flock.Guarantees.custodied zr hC h R ω t o.1 j).I (Flock.Guarantees.custodied zr hC h R ω t o.1 j).own t') := by
    refine ⟨(o.2 j).2.2.2, hv j, hlen, ?_⟩
    dsimp only [Flock.Guarantees.custodied]
    rw [zsOf_pos zr _ h R ω t o.1 j (hall j)]
    exact hchk j
  have hj := hv j
  rw [← own_keepOwn (Flock.Guarantees.custodied zr hC h R ω t o.1 j)
    (fun J own tr => I.r ≤ t.length ∧ Flock.Guarantees.ChecksC3 ish cs coins j (bind j) rows J own tr) hp] at hj
  exact hj

open Classical in
/-- **A kept session accepts only what its session accepts**: its own list is the session's, or `freshOwn`, which
accepts nothing (`not_acceptsZKR_fresh`). -/
theorem acceptsZKR_of_keepOwn {Q : VStmt} {Reg' : Type} {z : Q.Outer Reg'}
    {P : FlockSoundness.Discharge.Exec.Inputs → Array Flock.Registered.Port →
      FlockSoundness.Discharge.Exec.Transcript → Prop} {t : FlockSoundness.Discharge.Exec.Transcript}
    (h : AcceptsZKR (Flock.Guarantees.keepOwn z P).I (Flock.Guarantees.keepOwn z P).own t) :
    AcceptsZKR z.I z.own t := by
  by_cases hp : ∃ t, AcceptsZKR z.I z.own t ∧ P z.I z.own t
  · rw [own_keepOwn z P hp] at h
    exact h
  · have hf : (Flock.Guarantees.keepOwn z P).own = freshOwn Q.c := ite_eq_right hp
    rw [hf] at h
    exact (not_acceptsZKR_fresh z.I z.ht z.hTags z.hcH t h).elim

/-- **VBridge at the typed sessions gives it at the checked sessions**, which accept only what the typed sessions accept
(`acceptsZKR_of_keepOwn`) and whose reads, layers and commit strings are theirs: `FlockC3.Compose.hbr_honest`
discharges `_hbr` through it. -/
theorem vbridge_checked {n : ℕ} {L : Law n} {Reg : Type} {I : Inner} {Cm : Type} {S : ℕ} {Vs : Fin S → VStmt}
    {Reg' Reg₂ : Type} {ish : RecVStar.Inner} {cs : Option Flock.RecVStar.InnerCommits}
    {coins : Option Flock.RecCoins.Source} {bind : Fin S → VStar.Bindings}
    {rows : Option (Array Flock.Registered.Port)}
    {zr : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).OuterR Reg'}
    {hC : Flock.Premises.RecordCustody zr} {h : Typed zr} {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {x : Reg → L.Ω → I.Stmt} {wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg}
    {Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv)}
    {cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg}
    {msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → I.Msg}
    (hb : Flock.Premises.VBridge (Flock.Guarantees.custodied zr hC h) dg N k Rw x wmsg Sk cmt msg) :
    Flock.Premises.VBridge (Flock.Guarantees.checkedC3 ish cs coins bind rows zr hC h) dg N k Rw x wmsg Sk cmt
      msg := by
  obtain ⟨hoff, hcm, hdec, hfull⟩ := hb
  refine ⟨fun R ω t R₂ hbad τ o ho => hoff R ω t R₂ hbad τ o fun j =>
      ⟨acceptsZKR_of_keepOwn (z := Flock.Guarantees.custodied zr hC h R ω t R₂ j)
        (P := fun J own tr => I.r ≤ t.length ∧ Flock.Guarantees.ChecksC3 ish cs coins j (bind j) rows J own tr)
        (ho j).1, (ho j).2⟩,
    fun i R ω t e he R₂ j ⟨tr, htr⟩ => hcm i R ω t e he R₂ j ⟨tr, acceptsZKR_of_keepOwn
      (z := Flock.Guarantees.custodied zr hC h R ω t R₂ j)
      (P := fun J own tr => I.r ≤ t.length ∧ Flock.Guarantees.ChecksC3 ish cs coins j (bind j) rows J own tr) htr⟩,
    hdec, hfull⟩

/-- **The recursive audit at candidate 3, as the verifier of record runs it.** -/
theorem RecursiveAuditC3 : Flock.Guarantees.RecursiveAuditC3 := by
  refine And.intro ?_ ?_
  swap
  · exact RecursiveFirewallPrivate
  intro nT r T CT _ _ n Ω _ _ draw Reg Cm Reg' Reg₂ obj prog st hp reg tr record coins xR hreg _hstk ish Vs zr hCust
    bind rows hob hF dg N σ wmsg Sk cmt msg hA3 k Rw M hk hRw hM ρ hρ hr hbr
  by_cases hT : Typed zr
  swap
  · left
    simp only [Typed, not_forall, not_exists] at hT
    obtain ⟨j₀, hj₀⟩ := hT
    refine prob_recGame_zero _ σ fun ω t s => ?_
    rw [prob_outerVR]
    exact prob_sessionsR_untypable hob (zr σ.1 ω t s.1) (fun j => (hCust σ.1 ω t s.1 j).1) (hF σ.1 ω t s.1) j₀
      (hj₀ (σ.1, ω, t, s.1)) _ (fun b hb => hb.2.1) _
  by_cases hacc : Flock.Guarantees.AcceptsPublic obj prog st ∧ AcceptsC3 ish st tr record reg
  swap
  · exact Or.inl (le_antisymm ((prob_mono_outcome _ σ fun o _ h => absurd h.1 hacc).trans (prob_false _ σ).le)
      (zero_le ..))
  refine Or.inr ⟨hT, ?_⟩
  have hin := FlockC3.Fork.holoForkSound_holoAt T CT hp (classOk_of_accepts hacc.1.acceptsStmt)
  have hr' : ∀ R ω t R₂ j S' R' i, rateZ (ZkHidden.planZHJ (Reg := Reg')
      (Flock.Guarantees.custodied zr hCust hT R ω t R₂ j).dj
      (Flock.Guarantees.custodied zr hCust hT R ω t R₂ j).y₀ S' R') i k ≤ ρ := by
    intro R ω t R₂ j S' R'
    dsimp only [Flock.Guarantees.custodied]
    by_cases hx' : (zr R ω t R₂ j).Typable
    · rw [zsOf_pos zr _ hT R ω t R₂ j hx']
      exact hr R ω t R₂ j _ _ hx'.y₀ S' R'
    · rw [zsOf_neg zr _ hT R ω t R₂ j hx']
      exact hr _ _ _ _ j _ _ _ S' R'
  rcases zk_sessions_recursive_fork_inner hA3
    (Flock.Guarantees.checkedC3 ish (innerCommitsAt tr record st reg).toOption coins bind rows zr hCust hT) dg N hk hRw
    hM hρ hr' (Flock.Guarantees.checkedT ish (innerCommitsAt tr record st reg).toOption coins bind rows zr hCust hT σ)
    (fun _ _ => Subtype.mk xR hreg) wmsg Sk cmt msg
    (hbr hT) Refine.H512 (clashPickHL T CT execArith Refine.H512 Refine.enc512 Refine.leafPlain (holoAt st hp)) hin
    with hf | hf | hf | hf | hb
  · exact Or.inl hf
  · exact Or.inr (Or.inl hf)
  · exact Or.inr (Or.inr (Or.inl hf))
  · exact Or.inr (Or.inr (Or.inr (Or.inl hf)))
  · refine Or.inr (Or.inr (Or.inr (Or.inr (le_trans (prob_recT_le hob zr
      (fun R ω t R₂ j => And.left (hCust R ω t R₂ j)) hT hF σ _
      (fun o => AcceptsV (Flock.Guarantees.checkedC3 ish (innerCommitsAt tr record st reg).toOption coins bind rows zr
          hCust hT o.1 o.2.1 o.2.2.1) o.2.2.2 ∧
        ¬ (holoInnerAt T CT st hp).holds (Subtype.mk xR hreg))
      (fun o ho => ho.2.1) (fun o hlen hall ho =>
        ⟨acceptsV_checked hlen.ge hall
          (verifies_zsOf zr _ hT o.1 o.2.1 o.2.2.1 o.2.2.2 hall (acceptsVR_of_V ho.2.1.1) ho.2.1.2).1 ho.2.2.1,
          ho.2.2.2⟩)) hb))))

end Flock.SecurityProofs
