import Definitions.NetworkCertifier.SampledAudit
import Proofs.Flock.Claims.Sampling.KS

/-!
# Sampling at the incorrect set, from sampling's own hypotheses and the wired check

`KSWired` (`Definitions.NetworkCertifier.SampledAudit`) is `Claims.Sampling.KS` with the incorrect set, the faulty and
the wire-wrong proof blocks, in place of the faulty set. It follows from `KS_of`'s hypotheses with the check a drawn
proof block passes widened from correct (`correctOk`) to correct with the wire of each wired gate it holds or reads
right (`wiredOk`): `KSBy_of` at that check, whose failing set is the incorrect set (`faultyBy_wiredOk`).
`SessionWires` is `SessionOpensBy` at that check: a drawn proof block's session reveals openings that make it correct
and open each wired gate it holds or reads, and that gate's writer, to one value. With no wires the incorrect set is
the faulty set (`incorrect_none`), and `KS` gives `KSWired` (`ksWired_none`).
-/

namespace NetTiming

open Claims Claims.Sampling Finset

/-- **The wired check** on proof block `u`: correct (`correctOk`), and each wired gate it holds or reads has its wire
right (`WireRight`) at the statement's wiring `wires s`. -/
def wiredOk (wires : AuditStatement → ℕ → Option ℕ) : BlockCheck := fun s u val ι =>
  correctOk s u val ι ∧
    ∀ g, (g ∈ u ∨ ∃ x ∈ u, g ∈ (s.prog.gates[x]?.map BGate.reads).getD []) → WireRight (wires s) val g

/-- A slot fails the wired check exactly when it is faulty or wire-wrong. -/
theorem faultyBy_wiredOk_iff {wires : AuditStatement → ℕ → Option ℕ} {H : Msg → Msg} {Cm : CommitmentScheme Bool}
    {s : AuditStatement} {P : Partition} {w : Wit} {v : ℕ} :
    FaultyBy (wiredOk wires) H Cm s P w v ↔
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

/-- **The failing set at the wired check is the incorrect set.** -/
theorem faultyBy_wiredOk (wires : AuditStatement → ℕ → Option ℕ) (H : Msg → Msg) (Cm : CommitmentScheme Bool)
    (s : AuditStatement) (P : Partition) (w : Wit) :
    faultyBy (wiredOk wires) H Cm s P w = incorrect H Cm s P (wires s) w := by
  classical
  ext v
  simp only [faultyBy, incorrect, Claims.Sampling.faulty, wireWrong, mem_union, mem_filter]
  constructor
  · rintro ⟨hv, h⟩
    exact (faultyBy_wiredOk_iff.1 h).imp (fun h => ⟨hv, h⟩) (fun h => ⟨hv, h⟩)
  · rintro (⟨hv, h⟩ | ⟨hv, h⟩)
    · exact ⟨hv, faultyBy_wiredOk_iff.2 (Or.inl h)⟩
    · exact ⟨hv, faultyBy_wiredOk_iff.2 (Or.inr h)⟩

/-- The sampling relation at the wired check is the sampling relation at the incorrect set. -/
theorem RBy_wiredOk (wires : AuditStatement → ℕ → Option ℕ) (H : Msg → Msg) (Cm : CommitmentScheme Bool)
    (parse : Msg → Option AuditStatement) (partition : AuditStatement → Option Partition) (δ : ℝ) :
    RBy (wiredOk wires) H Cm parse partition δ = RWired H Cm parse partition wires δ := by
  funext x w
  simp only [RBy, RWired, faultyBy_wiredOk]

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

/-- **A drawn proof block's session shows it correct and its wired rows right**: `SessionOpensBy` at the wired check.
Its session reveals openings under which the proof block is correct, and each wired gate it holds or reads and that
gate's writer are opened to one value. With no wires it is `SessionOpens`. -/
def SessionWires (H : Msg → Msg) (D : Deploy Coin Col T) (iroots : Msg → Seen → ℕ → List Msg)
    (wires : AuditStatement → ℕ → Option ℕ) : Prop :=
  SessionOpensBy H D iroots (wiredOk wires)

set_option synthInstance.maxSize 1024 in
/-- **Sampling at the incorrect set from sampling's hypotheses**: `KS_of`'s, with `SessionWires` in place of
`SessionOpens`. -/
theorem KSWired_of (wires : AuditStatement → ℕ → Option ℕ) {H : Msg → Msg} {D : Deploy Coin Col T}
    (iroots : Msg → Seen → ℕ → List Msg) (hbind : D.commit.Binds H) (hsess : SessionKS H D)
    (hopens : SessionWires H D iroots wires) (hfix : InteriorFixed H D iroots) (hdraw : DrawLaw D)
    (hrefuse : Refuses H D) (hε : ∀ x, 0 ≤ D.εtable x) : KSWired H D wires := by
  intro δ decode hδ hdec
  have h := KSBy_of (wiredOk wires) iroots hbind hsess hopens hfix hdraw hrefuse hε δ decode hδ hdec
  rwa [RBy_wiredOk] at h

/-- **With no wires, `KS` is sampling at the incorrect set.** -/
theorem ksWired_none {H : Msg → Msg} {D : Deploy Coin Col T} {wires : AuditStatement → ℕ → Option ℕ}
    (hw : ∀ s, wires s = fun _ => none) (h : Claims.Sampling.KS H D) : KSWired H D wires := by
  have hw' : wires = fun _ _ => none := funext hw
  subst hw'
  intro δ decode hδ hdec
  rw [RWired_none]
  exact h δ decode hδ hdec

end Hyps

end NetTiming
