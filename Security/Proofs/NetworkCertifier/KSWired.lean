import Definitions.NetworkCertifier.SampledAudit
import Proofs.NetworkCertifier.Wires
import Proofs.Flock.Claims.Sampling.KS

/-!
# Sampling at the incorrect set, from sampling's own hypotheses and the wired check

Sampling at the incorrect set is `Claims.Sampling.KSBy` at the wired check (`wiredOk`, `Definitions.NetworkCertifier.
SampledAudit`): correct, with the wire of each wired gate a proof block holds or reads right. Its failing set is the
incorrect set, the faulty and the wire-wrong proof blocks (`incorrectBy_wiredOk`), and its relation `RWired`
(`RBy_wiredOk`, `ksBy_wiredOk`). It follows from `KS_of`'s hypotheses with the check a drawn proof block passes widened
from `correctOk` to `wiredOk`: `KSBy_of` at that check (`KSWired_of`). `SessionWires` is `SessionOpensBy` at that
check: a drawn proof block's session reveals openings that make it correct and open each wired gate it holds or reads
to its writer's committed value, at a writer that is no constant gate. With no wires the incorrect set is the faulty set
(`incorrect_none`), and `KS` gives `KSBy` at the wired check (`ksWired_none`).
-/

namespace NetTiming

open Claims Claims.Sampling Finset

/-- **The sampling relation at the incorrect set** at `δ`, on the statement's bytes, at each statement's wiring
`wires s`: the statement parses, its query evaluates, and the incorrect set is in `M_δ`. It is `RBy` at the wired check
(`RBy_wiredOk`), and with no wires `Claims.Sampling.R` (`RWired_none`). -/
def RWired (H : Msg → Msg) (Cm : CommitmentScheme Bool) (parse : Msg → Option AuditStatement)
    (partition : AuditStatement → Option Partition) (wires : AuditStatement → ℕ → Option ℕ) (δ : ℝ) : Rel Wit :=
  fun x w => ∃ s P, parse x = some s ∧ partition s = some P ∧ incorrect H Cm s P (wires s) w ∈ s.design.M P δ

/-- A slot fails the wired check exactly when it is faulty or wire-wrong. -/
theorem incorrectBy_wiredOk_iff {wires : AuditStatement → ℕ → Option ℕ} {H : Msg → Msg} {Cm : CommitmentScheme Bool}
    {s : AuditStatement} {P : Partition} {w : Wit} {v : ℕ} :
    IncorrectBy (wiredOk wires) H Cm s P w v ↔
      Claims.Sampling.Faulty H Cm s P w v ∨ WireWrong H Cm s P (wires s) w v := by
  constructor
  · intro h
    by_cases hW : WireWrong H Cm s P (wires s) w v
    · exact Or.inr hW
    · left
      rintro ⟨u, hu, ι, hι⟩
      apply h
      refine ⟨u, hu, ι, hι, fun g hg => ?_⟩
      by_contra hn
      exact hW ⟨u, hu, g, hg, hn⟩
  · rintro (h | ⟨u, hu, g, hg, hn⟩) ⟨u', hu', ι, hι, hW⟩
    · exact h ⟨u', hu', ι, hι⟩
    · rw [hu] at hu'
      cases hu'
      exact hn (hW g hg)

/-- **The incorrect set at the wired check is the incorrect set.** -/
theorem incorrectBy_wiredOk (wires : AuditStatement → ℕ → Option ℕ) (H : Msg → Msg) (Cm : CommitmentScheme Bool)
    (s : AuditStatement) (P : Partition) (w : Wit) :
    incorrectBy (wiredOk wires) H Cm s P w = incorrect H Cm s P (wires s) w := by
  classical
  ext v
  simp only [incorrectBy, incorrect, Claims.Sampling.faulty, wireWrong, mem_union, mem_filter]
  constructor
  · rintro ⟨hv, h⟩
    exact (incorrectBy_wiredOk_iff.1 h).imp (fun h => ⟨hv, h⟩) (fun h => ⟨hv, h⟩)
  · rintro (⟨hv, h⟩ | ⟨hv, h⟩)
    · exact ⟨hv, incorrectBy_wiredOk_iff.2 (Or.inl h)⟩
    · exact ⟨hv, incorrectBy_wiredOk_iff.2 (Or.inr h)⟩

/-- The sampling relation at the wired check is the sampling relation at the incorrect set. -/
theorem RBy_wiredOk (wires : AuditStatement → ℕ → Option ℕ) (H : Msg → Msg) (Cm : CommitmentScheme Bool)
    (parse : Msg → Option AuditStatement) (partition : AuditStatement → Option Partition) (δ : ℝ) :
    RBy (wiredOk wires) H Cm parse partition δ = RWired H Cm parse partition wires δ := by
  funext x w
  simp only [RBy, RWired, incorrectBy_wiredOk]

/-- With no wires, no proof block is wire-wrong. -/
theorem wireWrong_none (H : Msg → Msg) (Cm : CommitmentScheme Bool) (s : AuditStatement) (P : Partition) (w : Wit) :
    wireWrong H Cm s P (fun _ => none) w = ∅ := by
  classical
  unfold wireWrong
  rw [Finset.filter_eq_empty_iff]
  rintro v - ⟨u, -, g, -, hn⟩
  exact hn fun w' h => by simp at h

/-- With no wires, the incorrect set is the faulty set. -/
theorem incorrect_none (H : Msg → Msg) (Cm : CommitmentScheme Bool) (s : AuditStatement) (P : Partition) (w : Wit) :
    incorrect H Cm s P (fun _ => none) w = Claims.Sampling.faulty H Cm s P w := by
  rw [incorrect, wireWrong_none, Finset.union_empty]

/-- With no wires, the sampling relation at the incorrect set is `Claims.Sampling.R`. -/
theorem RWired_none (H : Msg → Msg) (Cm : CommitmentScheme Bool) (parse : Msg → Option AuditStatement)
    (partition : AuditStatement → Option Partition) (δ : ℝ) :
    RWired H Cm parse partition (fun _ _ => none) δ = Claims.Sampling.R H Cm parse partition δ := by
  funext x w
  simp only [RWired, Claims.Sampling.R, Claims.Sampling.RelS, incorrect_none]

section Hyps

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

/-- **`KSBy` at the wired check is sampling at the incorrect set** (`RWired`): on the statements in the sampling scope,
at every `δ ≥ 0`, at the sampling extractor and every decoder that lists every trace within Flock's radius,
`Pr[acc ∧ ¬RWired δ] ≤ δ + ε_ks + ε_bind`, at `KS`'s errors. -/
theorem ksBy_wiredOk {H : Msg → Msg} {D : Deploy Coin Col T} {wires : AuditStatement → ℕ → Option ℕ} :
    KSBy H D (wiredOk wires) ↔
      ∀ (δ : ℝ) decode, 0 ≤ δ → Claims.Flock.ListWithin D.flock.len D.flock.enc D.flock.θ decode →
        KSLinOn H D.V (Scope D) (RWired H D.commit D.parse D.partition wires δ) (E D decode) (κ D δ) (γ D)
          (fun _ => 1) (Bext H D.V (E D decode)) := by
  simp only [KSBy, RBy_wiredOk]

/-- **A drawn proof block's session shows it correct and its wired rows right**: `SessionOpensBy` at the wired check.
Its session reveals openings under which the proof block is correct, and each wired gate it holds or reads is opened to
its writer's committed value, at a writer that is no constant gate (`WireRight`). With no wires it is `SessionOpens`. -/
def SessionWires (H : Msg → Msg) (D : Deploy Coin Col T) (iroots : Msg → Seen → ℕ → List Msg)
    (wires : AuditStatement → ℕ → Option ℕ) : Prop :=
  SessionOpensBy H D iroots (wiredOk wires)

set_option synthInstance.maxSize 1024 in
/-- **Sampling at the incorrect set from sampling's hypotheses**: `KS_of`'s, with `SessionWires` in place of
`SessionOpens`: `KSBy_of` at the wired check. -/
theorem KSWired_of (wires : AuditStatement → ℕ → Option ℕ) {H : Msg → Msg} {D : Deploy Coin Col T}
    (iroots : Msg → Seen → ℕ → List Msg) (hbind : D.commit.Binds H) (hsess : SessionKS H D)
    (hopens : SessionWires H D iroots wires) (hfix : InteriorFixed H D iroots) (hdraw : DrawLaw D)
    (hrefuse : Refuses H D) (hε : ∀ x, 0 ≤ D.εtable x) : KSBy H D (wiredOk wires) :=
  KSBy_of (wiredOk wires) iroots hbind hsess hopens hfix hdraw hrefuse hε

/-- **With no wires, `KS` is sampling at the incorrect set.** -/
theorem ksWired_none {H : Msg → Msg} {D : Deploy Coin Col T} {wires : AuditStatement → ℕ → Option ℕ}
    (hw : ∀ s, wires s = fun _ => none) (h : Claims.Sampling.KS H D) : KSBy H D (wiredOk wires) := by
  have hw' : wires = fun _ _ => none := funext hw
  subst hw'
  refine ksBy_wiredOk.2 fun δ decode hδ hdec => ?_
  rw [RWired_none]
  exact h δ decode hδ hdec

end Hyps

end NetTiming
