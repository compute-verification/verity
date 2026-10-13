import Models.Accounting.Guarantees.PerRun
import Proofs.Accounting.Joint

/-!
# Proof of the per-run joint guarantee and its covert residual

An audit's per-run guarantee bounds each section of its event at a value of the other audits' coins, so it bounds the
event over all three audits' coins (`Run.pr_computeLt_le` and the two beside it, read through the equivalence that
puts the audit's own coins last). A union bound sums the three (`Run.jointPerRun`). On each run, an account's charge and
every other account's are, together, within the device's limit, so the other accounts' use beyond a residual puts that
run's charge below its certificate (`Run.workLt_of_othersWork` and the two beside it), and the residual follows.

A per-run guarantee implies its sure-bound form (`Run.computeSound_of_perRun` and the two beside it), so `JointPerRun`'s
hypotheses are at least as strong as `Joint`'s, and an ideal audit has its per-run guarantee with error 0
(`Run.computeSoundPerRun_ideal` and the two beside it): the hypotheses hold together on any run.
-/

namespace Accounting

open Pouw

namespace Run

variable {ι : Type} {d : Device ι} {α ΩW ΩQ ΩS : Type} [Fintype α] (r : Run d α ΩW ΩQ ΩS)

section Sure

theorem computeSound_of_perRun [Fintype ΩW] {p : Price} {W ε : ℝ} (h : r.ComputeSoundPerRun p W ε) : r.ComputeSound p W ε :=
  fun o _ _ hT => (pr_mono fun x hx => ⟨hx.1, (hT x).trans_lt hx.2⟩).trans (h o)

theorem bandwidthSound_of_perRun [Fintype ΩQ] {Q ε : ℝ} (h : r.BandwidthSoundPerRun Q ε) : r.BandwidthSound Q ε :=
  fun x s _ _ hB => (pr_mono fun q hq => ⟨hq.1, (hB q).trans_lt hq.2⟩).trans (h x s)

theorem capacitySound_of_perRun [Fintype ΩS] {S ε : ℝ} (h : r.CapacitySoundPerRun S ε) : r.CapacitySound S ε :=
  fun x q _ _ hB => (pr_mono fun s hs => ⟨hs.1, (hB s).trans_lt hs.2⟩).trans (h x q)

theorem computeSoundPerRun_ideal [Fintype ΩW] {p : Price} {W : ℝ} (h : ∀ ω, r.accepts .compute ω → W ≤ r.work p ω) :
    r.ComputeSoundPerRun p W 0 := fun _ =>
  (pr_eq_zero fun _ hx => absurd (h _ hx.1) (not_le.2 hx.2)).le

theorem bandwidthSoundPerRun_ideal [Fintype ΩQ] {Q : ℝ} (h : ∀ ω, r.accepts .bandwidth ω → Q ≤ r.read ω) :
    r.BandwidthSoundPerRun Q 0 := fun _ _ =>
  (pr_eq_zero fun _ hq => absurd (h _ hq.1) (not_le.2 hq.2)).le

theorem capacitySoundPerRun_ideal [Fintype ΩS] {S : ℝ}
    (h : ∀ ω, r.accepts .capacity ω → S * (r.win .capacity).len ≤ r.held ω) : r.CapacitySoundPerRun S 0 := fun _ _ =>
  (pr_eq_zero fun _ hs => absurd (h _ hs.1) (not_le.2 hs.2)).le

end Sure

section Lt

variable [Fintype ΩW] [Fintype ΩQ] [Fintype ΩS]

theorem pr_computeLt_le [Nonempty ΩQ] [Nonempty ΩS] {p : Price} {W ε : ℝ} (h : r.ComputeSoundPerRun p W ε) :
    pr (fun ω => r.accepts .compute ω ∧ (r.work p ω : ℝ) < W) ≤ ε := by
  have := pr_prod_le (fun y : (ΩQ × ΩS) × ΩW => r.accepts .compute (y.2, y.1) ∧ (r.work p (y.2, y.1) : ℝ) < W) ε
    ((pr_nonneg _).trans (h (Classical.arbitrary _))) h
  rw [← pr_equiv (Equiv.prodComm ΩW (ΩQ × ΩS))] at this
  exact this

theorem pr_bandwidthLt_le [Nonempty ΩW] [Nonempty ΩS] {Q ε : ℝ} (h : r.BandwidthSoundPerRun Q ε) :
    pr (fun ω => r.accepts .bandwidth ω ∧ (r.read ω : ℝ) < Q) ≤ ε := by
  have := pr_prod_le
    (fun y : (ΩW × ΩS) × ΩQ => r.accepts .bandwidth (y.1.1, y.2, y.1.2) ∧ (r.read (y.1.1, y.2, y.1.2) : ℝ) < Q) ε
    ((pr_nonneg _).trans (h (Classical.arbitrary _) (Classical.arbitrary _))) fun o => h o.1 o.2
  rw [← pr_equiv bandwidthLast] at this
  exact this

theorem pr_capacityLt_le [Nonempty ΩW] [Nonempty ΩQ] {S ε : ℝ} (h : r.CapacitySoundPerRun S ε) :
    pr (fun ω => r.accepts .capacity ω ∧ (r.held ω : ℝ) < S * (r.win .capacity).len) ≤ ε := by
  have := pr_prod_le
    (fun y : (ΩW × ΩQ) × ΩS =>
      r.accepts .capacity (y.1.1, y.1.2, y.2) ∧ (r.held (y.1.1, y.1.2, y.2) : ℝ) < S * (r.win .capacity).len) ε
    ((pr_nonneg _).trans (h (Classical.arbitrary _) (Classical.arbitrary _))) fun o => h o.1 o.2
  rw [← pr_equiv (Equiv.prodAssoc ΩW ΩQ ΩS).symm] at this
  exact this

theorem jointPerRun [Nonempty ΩW] [Nonempty ΩQ] [Nonempty ΩS] {p : Price} {W Q S εW εQ εS : ℝ} (hW : r.ComputeSoundPerRun p W εW)
    (hQ : r.BandwidthSoundPerRun Q εQ) (hS : r.CapacitySoundPerRun S εS) :
    pr (fun ω => r.Accepted ω ∧
        ((r.work p ω : ℝ) < W ∨ (r.read ω : ℝ) < Q ∨ (r.held ω : ℝ) < S * (r.win .capacity).len))
      ≤ εW + εQ + εS := by
  calc _ ≤ pr (fun ω => (r.accepts .compute ω ∧ (r.work p ω : ℝ) < W) ∨
          (r.accepts .bandwidth ω ∧ (r.read ω : ℝ) < Q) ∨
          (r.accepts .capacity ω ∧ (r.held ω : ℝ) < S * (r.win .capacity).len)) :=
        pr_mono fun ω h => h.2.imp (⟨h.1 .compute, ·⟩) (Or.imp (⟨h.1 .bandwidth, ·⟩) (⟨h.1 .capacity, ·⟩))
    _ ≤ εW + (εQ + εS) :=
        (pr_or_le _ _).trans (add_le_add (r.pr_computeLt_le hW)
          ((pr_or_le _ _).trans (add_le_add (r.pr_bandwidthLt_le hQ) (r.pr_capacityLt_le hS))))
    _ = εW + εQ + εS := (add_assoc _ _ _).symm

end Lt

section Covert

variable [DecidableEq α]

theorem workLt_of_othersWork {p : Price} {W : ℝ} {ω : ΩW × ΩQ × ΩS}
    (h : ((d.priced p * (r.win .compute).len : ℕ) : ℝ) - W < r.othersWork p ω) : (r.work p ω : ℝ) < W := by
  have hle : ((r.work p ω : ℕ) : ℝ) + r.othersWork p ω ≤ ((d.priced p * (r.win .compute).len : ℕ) : ℝ) := by
    exact_mod_cast r.work_add_othersWork p ω
  linarith

theorem readLt_of_othersRead {Q : ℝ} {ω : ΩW × ΩQ × ΩS}
    (h : ((d.bw * (r.win .bandwidth).len : ℕ) : ℝ) - Q < r.othersRead ω) : (r.read ω : ℝ) < Q := by
  have hle : ((r.read ω : ℕ) : ℝ) + r.othersRead ω ≤ ((d.bw * (r.win .bandwidth).len : ℕ) : ℝ) := by
    exact_mod_cast r.read_add_othersRead ω
  linarith

theorem heldLt_of_othersHeld {S : ℝ} {ω : ΩW × ΩQ × ΩS}
    (h : ((d.mem : ℝ) - S) * (r.win .capacity).len < r.othersHeld ω) :
    (r.held ω : ℝ) < S * (r.win .capacity).len := by
  have hle : ((r.held ω : ℕ) : ℝ) + r.othersHeld ω ≤ (d.mem : ℝ) * (r.win .capacity).len := by
    exact_mod_cast r.held_add_othersHeld ω
  have hring : ((d.mem : ℝ) - S) * (r.win .capacity).len =
      d.mem * (r.win .capacity).len - S * (r.win .capacity).len := by
    ring
  linarith

variable [Fintype ΩW] [Fintype ΩQ] [Fintype ΩS] [Nonempty ΩW] [Nonempty ΩQ] [Nonempty ΩS]

theorem covertResidualPerRun {p : Price} {W Q S εW εQ εS : ℝ} (hW : r.ComputeSoundPerRun p W εW)
    (hQ : r.BandwidthSoundPerRun Q εQ) (hS : r.CapacitySoundPerRun S εS) :
    pr (fun ω => r.Accepted ω ∧
        (((d.priced p * (r.win .compute).len : ℕ) : ℝ) - W < r.othersWork p ω ∨
          ((d.bw * (r.win .bandwidth).len : ℕ) : ℝ) - Q < r.othersRead ω ∨
          ((d.mem : ℝ) - S) * (r.win .capacity).len < r.othersHeld ω))
      ≤ εW + εQ + εS :=
  (pr_mono fun _ h => ⟨h.1, h.2.imp r.workLt_of_othersWork
    (Or.imp r.readLt_of_othersRead r.heldLt_of_othersHeld)⟩).trans (r.jointPerRun hW hQ hS)

end Covert

end Run

namespace SecurityProofs

theorem JointPerRun : Accounting.Guarantees.JointPerRun := by
  intro ι α ΩW ΩQ ΩS _ _ _ _ _ _ _ d r p W Q S εW εQ εS hW hQ hS
  exact r.jointPerRun hW hQ hS

theorem CovertResidualPerRun : Accounting.Guarantees.CovertResidualPerRun := by
  intro ι α ΩW ΩQ ΩS _ _ _ _ _ _ _ _ d r p W Q S εW εQ εS hW hQ hS
  exact r.covertResidualPerRun hW hQ hS

end SecurityProofs

end Accounting
