import Proofs.Accounting.TS.Session.Checked.Guarantees
import Proofs.Accounting.TS.SessionSound

/-!
# A checked unit's proof layer is sound through its C-Flock sessions (the proof)

`Accounting.SecurityProofs.TS.CheckedSessionSound`. `AcceptedLive` holds by the layer's definition. `ServedChecksHold`
takes the value layer itself as the relation's assignment: `RowsSession.RowsAt` says the circuit's evaluation agrees
with it on each drawn unit's wires. `ServedRootsLinked` at `0` is the window's law at an event of probability `0` in
every session's game: under `kd = n` the draw is every unit (`RowsSession.draw_full`, from `subset_spec`), so at an
accepted outcome whose units are right `Flock.SecurityProofs.ServedRowsLinked` makes every registered read of the
value layer the committed row (`RowsSession.servedEv_false`). `SoundAt δ` is then
`Accounting.SecurityProofs.TS.SessionSoundAt` at `δs = 0`.

This module imports `Proofs.Flock.SoundAtBridge` (through `Proofs.Accounting.TS.SessionSound`), whose imports make `X`
a token, so `Proofs/Accounting.lean` imports it.
-/

namespace FlockSoundness.Audit.Law

/-- **A `subset` live draw that runs is `Flock.Draw.subset` on its stream.** -/
theorem runOS_subset {L : Fin 1 → ℕ} {f : (i : Fin 1) → Fin (L i) → Fin 256} {k n : ℕ} {d : Flock.Draw.UnitDraw}
    (hd : runOS f (.subset k) n = some d) : Flock.Draw.subset k n (streamOf (f 0)) = some d.units := by
  unfold runOS at hd
  rw [drawWith_subset_run] at hd
  cases ho : (Flock.Draw.onWith (tapeGet (tapesOf f) 0) (2 * k * Flock.Draw.width n + 64)
      (Flock.Draw.subset k n)).run (fun _ => 0) with
  | none => rw [ho] at hd; exact absurd hd (by simp)
  | some p =>
    obtain ⟨u, pos'⟩ := p
    rw [ho] at hd
    cases u with
    | none => exact absurd hd (by simp)
    | some D =>
      have hD : D = d.units := by rw [← Option.some.inj hd]
      have h1 := (onWith_tape (tapesOf f) 0 _ (by omega) _ (fun a b u h => subset_append _ _ a b u h) _ pos' _
        rfl ho).2 D rfl
      have h2 : tapesOf f 0 = streamOf (f 0) := by simp [tapesOf, Array.getD_eq_getD_getElem?]
      rw [h2] at h1
      rw [← hD]; exact h1

end FlockSoundness.Audit.Law

namespace Accounting.TS.Checked

open Flock.SoundAtBridge

/-- **The checked unit's port conditions at `ok`**: the circuit has a port named `ok` that `own` registers a port of. `CheckedSessionSound` takes them of every session; `checkedOn_soundAt` only of a session
that meets a setup predicate implying them, such as flock-verify's known-read check at known content naming `ok`. -/
abbrev PortsOk (ok : String) (s : RowsSession) : Prop :=
  (∃ q : Fin s.c.ports.size, s.c.ports[q].name = ok) ∧ ∃ pt ∈ s.own, pt.name = ok

open Classical in
/-- **An assignment's registered reads at a session `P` holds of**: `rowsReg` where `P s`, else the committed rows
(`openedReg`), which then link trivially. -/
noncomputable def rowsOn (P : RowsSession → Prop) (s : RowsSession) (vals : Fin s.Circ.N → Bool) :
    ℕ → ℕ → Option Bool :=
  if P s then rowsReg s vals else openedReg s

end Accounting.TS.Checked

namespace Flock.SoundAtBridge

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden FlockSoundness.Discharge.ZkReg
open FlockSoundness.Discharge.Tables (LiveOutJ)
open Accounting.TS.Checked
open scoped ENNReal

/-- A permutation of `ℕ` that keeps `[0, n)` maps `[0, n)` onto it. -/
theorem perm_onto_range (π : Equiv.Perm ℕ) {n : ℕ} (hπ : ∀ t < n, π t < n) : ∀ x < n, ∃ i < n, π i = x := by
  intro x hx
  have hsub : (Finset.range n).image π ⊆ Finset.range n := by
    intro y hy
    obtain ⟨t, ht, rfl⟩ := Finset.mem_image.1 hy
    exact Finset.mem_range.2 (hπ t (Finset.mem_range.1 ht))
  have heq : (Finset.range n).image π = Finset.range n :=
    Finset.eq_of_subset_of_card_le hsub (by rw [Finset.card_image_of_injective _ π.injective])
  have hx' : x ∈ (Finset.range n).image π := by rw [heq]; exact Finset.mem_range.2 hx
  obtain ⟨i, hi, he⟩ := Finset.mem_image.1 hx'
  exact ⟨i, Finset.mem_range.1 hi, he⟩

/-- **The live audit's event of probability `0`** (`auditLive`, unfolded): an event whose probability is `0` in the
live session at every registration `R` and draw coin `ω` has probability `0` at every strategy. -/
theorem prob_audit_eq_zero {n : ℕ} (L : Law n) (Reg : Type)
    (live : Finset (Fin n) → Reg → Game Discharge.Exec.Transcript)
    (p : Reg × Finset (Fin n) × Discharge.Exec.Transcript → Prop)
    (h : ∀ R ω (s' : Strategy (live (L.draw ω) R)), prob (fun t => p (R, L.draw ω, t)) (live (L.draw ω) R) s' = 0)
    (τ : Strategy (Game.send Reg fun R => Game.coin L.Ω fun ω => (live (L.draw ω) R).map fun t => (R, L.draw ω, t))) :
    prob p (Game.send Reg fun R => Game.coin L.Ω fun ω => (live (L.draw ω) R).map fun t => (R, L.draw ω, t)) τ = 0 := by
  have h0 : ∀ ω, prob p ((live (L.draw ω) τ.1).map fun t => (τ.1, L.draw ω, t)) (τ.2 ω) = 0 := fun ω =>
    (prob_map _ _ _ _).trans (h τ.1 ω _)
  calc prob p _ τ
      = FlockSoundness.Game.avg fun ω => prob p ((live (L.draw ω) τ.1).map fun t => (τ.1, L.draw ω, t)) (τ.2 ω) := rfl
    _ = FlockSoundness.Game.avg fun _ : L.Ω => (0 : ℝ≥0∞) := congrArg (FlockSoundness.Game.avg (C := L.Ω)) (funext h0)
    _ = 0 := FlockSoundness.Game.avg_const 0

namespace RowsSession

variable (s : RowsSession)

/-- **Under full checking the draw is every unit**: at `kd = n` the subset draw, when it runs, is `n` distinct units
below `n` (`subset_spec`), and when it doesn't the law draws every unit. -/
theorem draw_full (hk : s.kd = s.n) (ω : s.L.Ω) : s.L.draw ω = Finset.univ := by
  show (match Law.runOS (s.src ω) (.subset s.kd) s.n with
    | some d => Finset.univ.filter fun u : Fin s.n => u.val ∈ d.units
    | none => Finset.univ) = Finset.univ
  split
  · rename_i d hd
    obtain ⟨π, -, hπ, hmem⟩ := subset_spec (Law.runOS_subset hd)
    refine Finset.filter_true_of_mem fun u _ => (hmem u.val).2 ?_
    rw [hk]
    exact perm_onto_range π hπ u.val u.isLt
  · rfl

/-- **`ServedRootsLinked`'s event at a session**: flock-verify accepts, every drawn unit is right at its owned reads,
and the value layer's registered reads aren't the committed rows. -/
def ServedEv (o : s.Out) : Prop :=
  LiveAcceptsZKR s.I s.own o ∧ (∀ u ∈ o.2.1, ¬ s.WrongAt u ∧ s.RowsAt o u) ∧ rowsReg s s.Xv ≠ openedReg s

/-- **No outcome of the audit meets `ServedEv`** under full checking, at every unit's served roots registered: every unit is drawn, so `Flock.SecurityProofs.ServedRowsLinked` puts the
committed row at each of its registered reads. -/
theorem servedEv_false (hk : s.kd = s.n)
    (hS : ∀ u, s.ServedRegistered u) (R : s.Reg) (ω : s.L.Ω)
    (o : LiveOutJ (planZHJ (Reg := s.Reg) s.dj s.y₀ (s.L.draw ω) R).length (mPtsOf s.c.g (s.kd / nTab s.I))) :
    ¬ s.ServedEv (R, s.L.draw ω, s.wr (s.L.draw ω) R o) := by
  rintro ⟨hacc, hD, hne⟩
  apply hne
  funext a b
  dsimp only [rowsReg, openedReg, regRead]
  split_ifs with h1 h2
  all_goals first
    | have hu : (⟨a, h1.1⟩ : Fin s.n) ∈ s.L.draw ω := by rw [s.draw_full hk ω]; exact Finset.mem_univ _
      exact congrArg some (Flock.SecurityProofs.ServedRowsLinked s (s.L.draw ω) R o ⟨a, h1.1⟩ hacc hu (hS _)
        (hD _ hu).2 _ h2.1 h2.2)
    | rfl

/-- **`ServedEv` has probability `0` in the session's game** under full checking and registered served roots. -/
theorem prob_servedEv (hk : s.kd = s.n)
    (hS : ∀ u, s.ServedRegistered u) : prob s.ServedEv s.G s.τ = 0 :=
  prob_audit_eq_zero s.L s.Reg (liveOfZJ s.dg (planZHJ s.dj s.y₀) s.N s.wr) s.ServedEv (fun R ω s' => by
    refine (prob_map (fun t => s.ServedEv (R, s.L.draw ω, t)) (s.wr (s.L.draw ω) R) _ s').trans ?_
    exact nonpos_iff_eq_zero.1 ((prob_mono (fun o ho => s.servedEv_false hk hS R ω o ho) _ _).trans
      (prob_false _ _).le)) s.τ

/-- **`ServedEv` at `rowsOn P` has probability `0`** where `P` of the session gives full checking and registered
served roots: where `P` holds it is `ServedEv` (`prob_servedEv`), and where it fails `rowsOn P` is the
committed rows. -/
theorem prob_servedOn (P : RowsSession → Prop)
    (hN : P s → s.kd = s.n ∧ ∀ u, s.ServedRegistered u) :
    prob (fun o => LiveAcceptsZKR s.I s.own o ∧ (∀ u ∈ o.2.1, ¬ s.WrongAt u ∧ s.RowsAt o u) ∧
      rowsOn P s s.Xv ≠ openedReg s) s.G s.τ = 0 := by
  by_cases hp : P s
  · have e : rowsOn P s s.Xv = rowsReg s s.Xv := if_pos hp
    simp only [e]
    exact s.prob_servedEv (hN hp).1 (hN hp).2
  · have e : rowsOn P s s.Xv = openedReg s := if_neg hp
    exact nonpos_iff_eq_zero.1 ((prob_mono (fun o ho => ho.2.2 e) _ _).trans (prob_false _ _).le)

end RowsSession

end Flock.SoundAtBridge

open Accounting.TS Accounting.TS.Checked Flock.SoundAtBridge FlockSoundness.Discharge.ZkReg in
/-- **A checked unit's proof layer is sound at `δ` through its C-Flock sessions**
(`Accounting.Guarantees.TS.CheckedSessionSound`). -/
theorem Accounting.SecurityProofs.TS.CheckedSessionSound : Accounting.Guarantees.TS.CheckedSessionSound := by
  intro Ω _ S nπ sess out ok right δ hCR hLaw hk hS hok hδ hδ0
  have hAcc : Accounting.Assumptions.TS.AcceptedLive (checkedLayer sess out ok right) sess out :=
    fun _ _ _ h => h
  have hChk : Accounting.Assumptions.TS.ServedChecksHold (checkedLayer sess out ok right) sess out rowsReg :=
    fun x w i _ hD => ⟨hok x w i, (sess x w i).Xv, rfl, fun u hu g hg => ((hD u hu).2 g hg).2.symm⟩
  have hLink : Accounting.Assumptions.TS.ServedRootsLinked (checkedLayer sess out ok right) sess out rowsReg 0 := by
    intro w i
    refine (hLaw w i fun s o => s.ServedEv o).trans (avg_le _ le_rfl fun x => le_of_eq ?_)
    exact (congrArg ENNReal.toReal (RowsSession.prob_servedEv (sess x w i) (hk x w i) (hS x w i))).trans
      ENNReal.toReal_zero
  refine ⟨hAcc, hChk, hLink, ?_⟩
  simpa only [zero_add] using
    Accounting.SecurityProofs.TS.SessionSoundAt (checkedLayer sess out ok right) sess out rowsReg 0 δ hCR hAcc hLaw
      hChk hLink hδ hδ0

namespace Accounting.TS.Checked

open Accounting.TS Flock.SoundAtBridge FlockSoundness.Discharge.ZkReg
open Pouw.RelativeSecurity (Paths)
open scoped ENNReal

/-- **The checked layer, its relation asked where `P` holds of the session**: `checkedLayer`'s acceptance and opening,
its relation `CheckedRel` wherever `P (sess x w i)`. Where the caller's acceptance implies `P`, its unsound event is
`checkedLayer`'s. -/
noncomputable def checkedOnLayer {Ω : Type} {S nπ : ℕ} (sess : Paths Ω S → Fin S → Fin nπ → RowsSession)
    (out : ∀ x w i, (sess x w i).Out) (ok : String) (P : RowsSession → Prop) :
    ProofLayer Ω S nπ (ℕ → ℕ → Option Bool) where
  accepted x w i := LiveAcceptsZKR (sess x w i).I (sess x w i).own (out x w i)
  opened x w i := openedReg (sess x w i)
  rel x w i v := P (sess x w i) → CheckedRel (sess x w i) (out x w i) ok v
  right _ _ := True

/-- **The checked layer is sound at `δ` where `P` holds**, at `CheckedSessionSound`'s hypotheses with full checking,
the `ok` port (`PortsOk`) and registered served roots asked only of a session `P` holds of:
`SessionSoundAt` at `rows := rowsOn P`, which reads the value layer where `P` holds (`CheckedSessionSound`'s argument)
and the committed rows where it fails, so nothing is asked there (`prob_servedOn`). -/
theorem checkedOn_soundAt {Ω : Type} [Fintype Ω] {S nπ : ℕ} (sess : Paths Ω S → Fin S → Fin nπ → RowsSession)
    (out : ∀ x w i, (sess x w i).Out) (ok : String) (P : RowsSession → Prop) (δ : ℝ)
    (hCR : ∀ x w i, (sess x w i).CR) (hLaw : Accounting.Assumptions.TS.WindowLaw sess out)
    (hP : ∀ s, P s → PortsOk ok s ∧ s.kd = s.n ∧ ∀ u, s.ServedRegistered u)
    (hδ : ∀ x w i, ((sess x w i).n : ℝ≥0∞) * (sess x w i).bound ≤ ENNReal.ofReal δ) (hδ0 : 0 ≤ δ) :
    (checkedOnLayer sess out ok P).SoundAt δ := by
  have hAcc : Accounting.Assumptions.TS.AcceptedLive (checkedOnLayer sess out ok P) sess out :=
    fun _ _ _ h => h
  have hChk : Accounting.Assumptions.TS.ServedChecksHold (checkedOnLayer sess out ok P) sess out (rowsOn P) := by
    intro x w i _ hD
    show P (sess x w i) → _
    intro hp
    have e : rowsOn P (sess x w i) (sess x w i).Xv = rowsReg (sess x w i) (sess x w i).Xv := if_pos hp
    rw [e]
    exact ⟨(hP _ hp).1, (sess x w i).Xv, rfl, fun u hu g hg => ((hD u hu).2 g hg).2.symm⟩
  have hLink : Accounting.Assumptions.TS.ServedRootsLinked (checkedOnLayer sess out ok P) sess out (rowsOn P) 0 := by
    intro w i
    refine (hLaw w i fun s o => LiveAcceptsZKR s.I s.own o ∧ (∀ u ∈ o.2.1, ¬ s.WrongAt u ∧ s.RowsAt o u) ∧
      rowsOn P s s.Xv ≠ openedReg s).trans (avg_le _ le_rfl fun x => le_of_eq ?_)
    exact (congrArg ENNReal.toReal (RowsSession.prob_servedOn (sess x w i) P
      (fun hp => ⟨(hP _ hp).2.1, (hP _ hp).2.2⟩))).trans ENNReal.toReal_zero
  simpa only [zero_add] using
    Accounting.SecurityProofs.TS.SessionSoundAt (checkedOnLayer sess out ok P) sess out (rowsOn P) 0 δ hCR hAcc hLaw
      hChk hLink hδ hδ0

end Accounting.TS.Checked
