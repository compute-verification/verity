import Proofs.Flock.Recursive.AuditC3
import Proofs.Flock.Recursive.FirewallZK
import Proofs.Flock.Recursive.Loads
import Proofs.Flock.Recursive.RegExtract

/-!
# The recursive audit at candidate 3: its proof

`Flock.SecurityProofs.RecursiveAuditC3 : Flock.Guarantees.RecursiveAuditC3`, `PublicAudit`'s proof at candidate 3's
table, and `RecursiveFirewallPrivate`.

In (i), as in `PublicAudit`'s:
* where some index has no session that types, the event is empty (`prob_sessionsR_untypable`);
* at a statement Lean's verifier doesn't build, or inputs at candidate 3 it refuses (`AcceptsC3`: V*'s list, the inner
  transcript against R's registration, the covers check there, check 5's sparse refusals), the event is empty (its
  first conjunct);
* at one it builds, the class checks pass (`classOk_of_accepts`), and gap 2 holds at the word the prover's runs open
  (`zk_sessions_recursive_fork_regV`, `Recursive/RegExtract.lean`: the extraction from piece (a)'s runs, off a
  registrant's row off a value the verifier registers itself at its rows, `valueSrcAll`, else `valueFinder`, off a
  read's leaf off the table's, else `leafFinder`, and off a read of R's level-0 top off its node's row under the held
  salts, else `saltFinder`, and off a held top other than `bcs'`, else `topFinder`, at `_hstk`), here at
  the checked sessions (`checkedC3`), whose games are the typed sessions', so the transported prover plays them;
* the hidden statement is the word the transported prover's runs open at R's level 0 against `cap` at each draw
  (`regStmt`), and the raw event is bounded by the typed sessions' at the transported prover (`prob_recT_le`,
  `verifies_zsOf`), where the event's checks at candidate 3 (`ChecksVR`) keep every session's own list at the whole
  histories `recGame` plays (`acceptsV_checked`);
* `_hob` is read where the event parses a session's file into scope (`loads_of_verifiesV`, `scope_of_accepts`).

Candidate 3's table ignores the last round's coin, as its pick does (`schedInner_lastFree`, `clashPickHL_lastFree`), so
gap 2 is read at VBridge at the committed rounds (`zk_sessions_recursive_fork_innerLV`, `innerC`'s form off
`ValueColl ∨ LeafColl ∨ SaltColl ∨ TopColl` that `RegExtract`'s proof reads).

`_hstk` is read here only as `TopColl`'s collision (`topFinder`): VBridgeCOff is a premise, and `_hstk` is what its
discharge at the honest family takes as `htop` (`FlockC3.Compose.hbr_honest_reg`, off `TopColl` the held top is
`TopOf`'s, `topOf_held`, which gives VBridgeCOff at the checked sessions and
the word the runs open, `_hbr`'s form, from `hbr_honest` at a fixed word of R's cap, since candidate 3's table reads
only the cap; its decoder reads the last round from the layers in full); `vbridge_checked` and `vbridgeC_checked` carry VBridge and
VBridgeC from the typed sessions to the checked ones.

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
    {ish : RecVStar.Inner} {cs : List (Cm × I.Coin) → Option Flock.RecVStar.InnerCommits}
    {coins : List (Cm × I.Coin) → Option Flock.RecCoins.Source}
    {bind : Fin S → VStar.Bindings} {rows : Option (Array Flock.Registered.Port)}
    {zr : A → B → List (Cm × I.Coin) → D → (j : Fin S) → (Vs j).OuterR Reg'} {hC : Flock.Premises.RecordCustody zr}
    {h : Typed zr} {R : A} {ω : B} {t : List (Cm × I.Coin)} {dg : ByteArray → ByteArray} {N : ℕ}
    {o : D × VRec Vs Reg' dg N} (hlen : I.r ≤ t.length) (hall : ∀ j, (zr R ω t o.1 j).Typable)
    (hv : AcceptsV (Flock.Guarantees.custodied zr hC h R ω t) o)
    (hchk : Flock.Guarantees.ChecksVR ish (cs t) (coins t) bind rows (zr R ω t) o) :
    AcceptsV (Flock.Guarantees.checkedC3 ish cs coins bind rows zr hC h R ω t) o := by
  intro j
  have hp : ∃ t', AcceptsZKR (Flock.Guarantees.custodied zr hC h R ω t o.1 j).I
      (Flock.Guarantees.custodied zr hC h R ω t o.1 j).own t' ∧
      (I.r ≤ t.length ∧ Flock.Guarantees.ChecksC3 ish (cs t) (coins t) j (bind j) rows
        (Flock.Guarantees.custodied zr hC h R ω t o.1 j).I (Flock.Guarantees.custodied zr hC h R ω t o.1 j).own t') := by
    refine ⟨(o.2 j).2.2.2, hv j, hlen, ?_⟩
    dsimp only [Flock.Guarantees.custodied]
    rw [zsOf_pos zr _ h R ω t o.1 j (hall j)]
    exact hchk j
  have hj := hv j
  rw [← own_keepOwn (Flock.Guarantees.custodied zr hC h R ω t o.1 j)
    (fun J own tr => I.r ≤ t.length ∧ Flock.Guarantees.ChecksC3 ish (cs t) (coins t) j (bind j) rows J own tr) hp]
    at hj
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
    {Reg' Reg₂ : Type} {ish : RecVStar.Inner} {cs : List (Cm × I.Coin) → Option Flock.RecVStar.InnerCommits}
    {coins : List (Cm × I.Coin) → Option Flock.RecCoins.Source} {bind : Fin S → VStar.Bindings}
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
        (P := fun J own tr => I.r ≤ t.length ∧ Flock.Guarantees.ChecksC3 ish (cs t) (coins t) j (bind j) rows J own tr)
        (ho j).1, (ho j).2⟩,
    fun i R ω t e he R₂ j R' tr hw htr => hcm i R ω t e he R₂ j R' tr hw (acceptsZKR_of_keepOwn
      (z := Flock.Guarantees.custodied zr hC h R ω t R₂ j)
      (P := fun J own tr => I.r ≤ t.length ∧ Flock.Guarantees.ChecksC3 ish (cs t) (coins t) j (bind j) rows J own tr)
      htr),
    hdec, hfull⟩

/-- **VBridgeC at the typed sessions gives it at the checked sessions**, as `vbridge_checked`. -/
theorem vbridgeC_checked {n : ℕ} {L : Law n} {Reg : Type} {I : Inner} {Cm : Type} {S : ℕ} {Vs : Fin S → VStmt}
    {Reg' Reg₂ : Type} {ish : RecVStar.Inner} {cs : List (Cm × I.Coin) → Option Flock.RecVStar.InnerCommits}
    {coins : List (Cm × I.Coin) → Option Flock.RecCoins.Source} {bind : Fin S → VStar.Bindings}
    {rows : Option (Array Flock.Registered.Port)}
    {zr : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).OuterR Reg'}
    {hC : Flock.Premises.RecordCustody zr} {h : Typed zr} {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {x : Reg → L.Ω → I.Stmt} {wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg}
    {Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv)}
    {cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg}
    {msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → I.Msg}
    (hb : Flock.Premises.VBridgeC (Flock.Guarantees.custodied zr hC h) dg N k Rw x wmsg Sk cmt msg) :
    Flock.Premises.VBridgeC (Flock.Guarantees.checkedC3 ish cs coins bind rows zr hC h) dg N k Rw x wmsg Sk cmt
      msg := by
  obtain ⟨hoff, hcm, hdec, hfull⟩ := hb
  refine ⟨fun R ω t R₂ hbad τ o ho => hoff R ω t R₂ hbad τ o fun j =>
      ⟨acceptsZKR_of_keepOwn (z := Flock.Guarantees.custodied zr hC h R ω t R₂ j)
        (P := fun J own tr => I.r ≤ t.length ∧ Flock.Guarantees.ChecksC3 ish (cs t) (coins t) j (bind j) rows J own tr)
        (ho j).1, (ho j).2⟩,
    fun i R ω t e he R₂ j R' tr hw htr => hcm i R ω t e he R₂ j R' tr hw (acceptsZKR_of_keepOwn
      (z := Flock.Guarantees.custodied zr hC h R ω t R₂ j)
      (P := fun J own tr => I.r ≤ t.length ∧ Flock.Guarantees.ChecksC3 ish (cs t) (coins t) j (bind j) rows J own tr)
      htr),
    hdec, hfull⟩

/-- **The recursive audit at candidate 3's soundness half, at every coin types `CT` and record map `im`.** -/
theorem recursiveAuditC3At {r : Flock.Stmt → HoloParams → ℕ}
    (CT : (st : Flock.Stmt) → (hp : HoloParams) → Fin (r st hp) → Type) [∀ st hp k, Fintype (CT st hp k)]
    [∀ st hp k, Inhabited (CT st hp k)]
    (im : (st : Flock.Stmt) → (hp : HoloParams) → (Cm : Type) → Flock.RecVStar.Inner → String →
      (Cm → String → Nat → List UInt8) →
        FlockSoundness.Assumptions.InnerRecordMap Cm ((k : Fin (r st hp)) → CT st hp k)) :
    Flock.Guarantees.RecursiveAuditC3At CT im := by
  intro n Ω _ _ draw Reg Cm Reg' Reg₂ obj prog st hp reg wr coins cap salt bcs' hstk ish Vs zr hCust _hRC _tbl _bc
    _hIC bind rows hob hF dg N σ wmsg Sk cmt msg hA3 k Rw M hk hRw hM ρ hρ hr hbr
  by_cases hT : Typed zr
  swap
  · left
    simp only [Typed, not_forall, not_exists] at hT
    obtain ⟨j₀, hj₀⟩ := hT
    refine prob_recGame_zero _ σ fun ω t s => ?_
    rw [prob_outerVR]
    exact prob_sessionsR_untypable hob (zr σ.1 ω t s.1) (fun j => (hCust σ.1 ω t s.1 j).1) (hF σ.1 ω t s.1) j₀
      (hj₀ (σ.1, ω, t, s.1)) _ (fun b hb => hb.2.1) _
  by_cases hacc : Flock.Guarantees.AcceptsPublic obj prog st
  swap
  · exact Or.inl (le_antisymm ((prob_mono_outcome _ σ fun o _ h => absurd h.1.1 hacc).trans (prob_false _ σ).le)
      (zero_le ..))
  refine Or.inr ⟨hT, ?_⟩
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
  rcases zk_sessions_recursive_fork_regV (Flock.Guarantees.holoTys st hp) (CT st hp) hp
    (classOk_of_accepts hacc.acceptsStmt) hA3
    (Flock.Guarantees.checkedC3 ish (Flock.Guarantees.innerHeld wr st reg) (fun t => coins (wr t).2) bind rows zr hCust
      hT)
    (valueSrcAll (fun t => coins (wr t).2) fun t => Flock.Guarantees.c3Of ish (Flock.Guarantees.innerHeld wr st reg t))
    dg N hk hRw hM hρ hr'
    (Flock.Guarantees.checkedT ish (Flock.Guarantees.innerHeld wr st reg) (fun t => coins (wr t).2) bind rows zr hCust
      hT σ)
    wmsg Sk cmt msg cap salt wr reg bcs' hstk (hbr hT)
    with hf | hf | hf | hf | hf | hf | hf | hf | hb
  · exact Or.inr (Or.inr (Or.inl hf))
  · exact Or.inr (Or.inr (Or.inr (Or.inl hf)))
  · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inl hf))))
  · exact Or.inr (Or.inl hf)
  · exact Or.inl hf
  · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl hf)))))
  · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl hf))))))
  · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl hf)))))))
  · refine Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (le_trans (prob_recT_le hob zr
      (fun R ω t R₂ j => And.left (hCust R ω t R₂ j)) hT hF σ _
      (fun o => AcceptsV (Flock.Guarantees.checkedC3 ish (Flock.Guarantees.innerHeld wr st reg)
          (fun t => coins (wr t).2) bind rows zr hCust hT o.1 o.2.1 o.2.2.1) o.2.2.2 ∧
        ¬ (holoInnerAt (Flock.Guarantees.holoTys st hp) (CT st hp) st hp).holds
          (Flock.Guarantees.regStmt (Flock.Guarantees.holoTys st hp) (CT st hp) st hp
          (Flock.Guarantees.checkedC3 ish (Flock.Guarantees.innerHeld wr st reg) (fun t => coins (wr t).2) bind rows zr
            hCust hT) dg N k Rw wmsg
          (Flock.Guarantees.checkedT ish (Flock.Guarantees.innerHeld wr st reg) (fun t => coins (wr t).2) bind rows zr
            hCust hT σ) cap o.1 o.2.1))
      (fun o ho => ho.2.1) (fun o hlen hall ho =>
        ⟨acceptsV_checked (cs := Flock.Guarantees.innerHeld wr st reg) (coins := fun t => coins (wr t).2) hlen.ge hall
          (verifies_zsOf zr _ hT o.1 o.2.1 o.2.2.1 o.2.2.2 hall (acceptsVR_of_V ho.2.1.1) ho.2.1.2).1 ho.2.2.1,
          ho.2.2.2⟩)) hb))))))))

/-- **The recursive audit at candidate 3, as the verifier of record runs it.** -/
theorem RecursiveAuditC3 : Flock.Guarantees.RecursiveAuditC3 := by
  refine And.intro ?_ ?_
  swap
  · exact RecursiveFirewallPrivate
  exact recursiveAuditC3At Flock.Guarantees.holoCTs Flock.Guarantees.holoMap

end Flock.SecurityProofs
