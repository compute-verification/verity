module

public import Definitions.Claims.SamplingTwoRegX
public import Proofs.Flock.SamplingGeneric.RegX
public import Proofs.Flock.SamplingGeneric.TwoCap
public import Proofs.Flock.SamplingGeneric.TwoReg
public import Proofs.Flock.SamplingGeneric.TwoRegMiss
import Proofs.Flock.Claims.Aggregate
import Proofs.Flock.Claims.ERepCost
import Proofs.Flock.Claims.Finder
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.Sampling.Binding
import Proofs.Flock.Claims.Sampling.Extract
import Proofs.Flock.Claims.Sampling.KS
import Proofs.Flock.Claims.Sampling.Wait
import Proofs.Flock.SamplingGeneric.Assemble
import Proofs.Flock.SamplingGeneric.Two

/-!
# The two-stage committed form, at the boundary trace read at the statement

`CompilerKSRegTwoX`, `CompilerMissTwoX` and `CompilerHeadlineTwoX`: the two-stage committed form, its miss bound and
its headline with the boundary read at the statement (`Compiled.regValX`) and the interior as before, at the tape and
the first stage's coins (`CompiledTwo.iregVal`). Its replay blocks wrong at the boundary are one set at every tape.

Nothing of the two-stage argument is redone. The deterministic half and the event's union (`regTwo_split`,
`eventTwo_split`), the miss (`pr_missTwo_le_Pa`, `pr_missTwo_le_pow`), the threshold form and the headline
(`threshold_of_event`, `headline_of_event`) each read the boundary trace as a parameter, and the interior's lemmas
(`ex_offIreg_le`) read no boundary. This file sets that parameter:

* `FregTwoX_eq`: the incorrect set is the grid's incorrect proof blocks at the boundary trace read at the statement and
  the committed interiors (`wrongAllAt`).
* `eventTwoX_le`: **the primary conjunct of `KSRegTwoX`**: `eventTwo_split` at that trace, with the sessions' bound at
  a selection that reads the run's coins off the second stage only, one run's disagreement and the rewinding lemma at
  the trace both at `B_conf,X` (`Compiled.pr_inConfX_le`, `Compiled.ex_offRegX_le`), and the interior's as before.
* `missTwoX`, `ksRegTwoX_of_event`, `headlineTwoX_of`: the miss, the threshold form and the headline at that trace.
-/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims Finset
open Claims.Sampling (AuditStatement Partition InScopeTwo footprintAt wrongAtBoundaryAt wrongAllAt)

namespace CompiledTwo

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- **The incorrect set at the boundary trace read at the statement, as the grid's**: on a statement that parses and
whose partitions evaluate, the incorrect proof blocks at that trace and the committed interiors. -/
theorem FregTwoX_eq (H : Msg → Msg) (L : CompiledTwo Coin A) {Ω : Type} [Fintype Ω] (P : Prover Ω) {x : Msg}
    {s : AuditStatement} {Pt : Partition} {d : Sampling.Design.Two} (hs : L.parse x = some s)
    (hPt : L.partition s = some Pt) (hd : L.two s = some d) (ω : Ω) (r : Fin (L.V.k x) → Coin) :
    L.FregTwoX H P x ω r = wrongAllAt s Pt d.part (L.regValX H P x) (L.iregVal H P x ω r) := by
  simp only [FregTwoX, hs, hPt, hd, wrongAllAt]

/-- **The primary conjunct of `KSRegTwoX`, from the sessions' bound**: an accepted run that samples a proof block of
`FregTwoX` has the first such block's session unfound at its first slot (the sessions' bound at `leadSel (selOf …)`,
which reads the run's coins off the second stage only), or its extractions disagree with themselves (`B_conf,X`'s
first run), open a gate off the boundary trace read at the statement (the rewinding lemma over the tapes and the runs
together), or open a gate off a replay block's committed interior (the rewinding lemma at the tape and the first stage,
and `B_conf,2`): `eventTwo_split` at the boundary trace read at the statement. -/
theorem eventTwoX_le {H : Msg → Msg} {L : CompiledTwo Coin A} (hA : A.KS H) (hbind : L.commit.Binds H)
    (hS : L.Sessions H) :
    ∀ x, L.Scope x → ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω),
      Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.FregTwoX H P x p.1 p.2) ≤ L.εregTwoX H x P := by
  intro x hsc Ω _ _ P
  obtain ⟨s, Pt, d, hs, hPt, hd, hin⟩ := hsc
  have hSK : L.toCompiled.SessionKS H := Compiled.sessionKS_of hA ⟨hS.fixed, hS.runs⟩
  obtain ⟨β, hβ0, hβ, hκ⟩ := hSK x Ω P
  have hT : Ex (fun ω : Ω => Pr fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
      (run H P L.V x ω rc.1).out = .acc ∧
        ∃ i, L.leadSel x (L.selOf x (wrongAllAt s Pt d.part (L.regValX H P x) (L.iregVal H P x ω rc.1)))
            (L.checked x rc.1) = some i ∧
          ∃ u : Fin (L.nPU x), (L.checked x rc.1)[i.val]? = some u.val ∧
            ¬ L.Found H x (run H P L.V x ω) u rc.1 (rc.2 i u)) ≤
      L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) :=
    Compiled.pr_unfound_le_at H L.toCompiled P β hβ0 hβ
      (fun ω r => L.leadSel x (L.selOf x (wrongAllAt s Pt d.part (L.regValX H P x) (L.iregVal H P x ω r))))
      (fun ω r e => by simp only [iregVal_forkOn]) fun ω r sl => hκ ω r sl
  have hI := Compiled.pr_inConfX_le hbind hs P
  have hO : Ex (fun ω : Ω => Pr fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
      ∃ g < L.nG x, ∃ b, L.opensAt H P x ω rc.1 rc.2 g = some b ∧ L.regValX H P x g ≠ some b) ≤
      L.nG x * Real.sqrt ((Compiled.BconfX H L.toCompiled Ω P x).Pr H) := Compiled.ex_offRegX_le hbind hs P
  have hIC := ex_offIreg_le hbind hS hs d.part P
  simp only [FregTwoX_eq H L P hs hPt hd]
  refine (Sampling.Pr_prod_eq_Ex _).trans_le
    ((Ex_mono fun ω => eventTwo_split hS hs hPt hd hin P ω (L.regValX H P x)).trans ?_)
  rw [Ex_add, Ex_add, Ex_add]
  unfold εregTwoX Compiled.εregX
  linarith

/-- **The miss bound at the boundary trace read at the statement**: `pr_missTwo_le_pow` at it, whose incorrect proof
blocks over the grid are `FregTwoX` (`FregTwoX_eq`). The replay blocks it counts are one set at every tape. -/
theorem missTwoX (H : Msg → Msg) (L : CompiledTwo Coin A) (hS : L.Sessions H) {r₁ r₂ : ℝ} (hr₁ : 0 ≤ r₁)
    (hr₁' : r₁ ≤ 1) (hr₂ : 0 ≤ r₂) (hr₂' : r₂ ≤ 1) : L.MissTwoX H r₁ r₂ := by
  intro x s Pt d hx hs hPt hd h1 h2 Ω _ P ω
  simp only [FregTwoX_eq H L P hs hPt hd]
  exact pr_missTwo_le_pow H L hS hr₁ hr₁' hr₂ hr₂' hx hs hPt hd h1 h2 P ω (L.regValX H P x)

/-- **The threshold form from the event**: on the contract, the event's bound gives `KSRegTwoX`
(`threshold_of_event` at the boundary trace read at the statement, one at every tape). -/
theorem ksRegTwoX_of_event (H : Msg → Msg) (L : CompiledTwo Coin A) (hS : L.Sessions H)
    (hev : ∀ x, L.Scope x → ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω),
      Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.FregTwoX H P x p.1 p.2) ≤ L.εregTwoX H x P) :
    L.KSRegTwoX H := by
  intro x hx Ω _ _ P
  refine ⟨hev x hx Ω P, fun s Pt d hs hPt hd δ hδ => ?_⟩
  have h := hev x hx Ω P
  simp only [FregTwoX_eq H L P hs hPt hd] at h
  exact threshold_of_event H L hS hs hPt hd (L.inScopeTwo_of hx hs hPt hd) P (fun _ => L.regValX H P x) h hδ

/-- **The headline from the committed form**: on the contract and `KSRegTwoX`, `headline_of_event` at the boundary
trace read at the statement and `KSRegTwoX`'s event. The footprint is one at every tape, so the bound on `P̄_a` is one
number. -/
theorem headlineTwoX_of (H : Msg → Msg) (L : CompiledTwo Coin A) (hS : L.Sessions H) (hks : L.KSRegTwoX H) :
    L.HeadlineTwoX H := by
  intro x s Pt d hx hs hPt hd Ω _ _ P pb hpb
  have h := (hks x hx Ω P).1
  simp only [FregTwoX_eq H L P hs hPt hd] at h
  exact (headline_of_event H L hS hs hPt hd (L.inScopeTwo_of hx hs hPt hd) P (fun _ => L.regValX H P x) h
    (fun _ => pb) fun _ ι => hpb ι).trans_eq (by rw [Ex_const])

end CompiledTwo

/-- **The two-stage compiler's committed form at the boundary trace read at the statement**:
`CompiledTwo.ksRegTwoX_of_event` at the event's bound (`CompiledTwo.eventTwoX_le`). -/
theorem compilerKSRegTwoX : CompilerKSRegTwoX := by
  intro Coin _ _ _ A L H hA hbind hS
  exact CompiledTwo.ksRegTwoX_of_event H L hS (CompiledTwo.eventTwoX_le hA hbind hS)

/-- **The two-stage miss bound at the boundary trace read at the statement**: `CompiledTwo.missTwoX`. -/
theorem compilerMissTwoX : CompilerMissTwoX := by
  intro Coin _ _ _ A L H hS r₁ r₂ hr₁ hr₁' hr₂ hr₂'
  exact CompiledTwo.missTwoX H L hS hr₁ hr₁' hr₂ hr₂'

/-- **The two-stage headline at the boundary trace read at the statement**: `CompiledTwo.headlineTwoX_of` at the
committed form (`compilerKSRegTwoX`). -/
theorem compilerHeadlineTwoX : CompilerHeadlineTwoX := by
  intro Coin _ _ _ A L H hA hbind hS
  exact CompiledTwo.headlineTwoX_of H L hS (compilerKSRegTwoX Coin A L H hA hbind hS)

end Claims.SamplingGeneric
