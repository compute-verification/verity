import Proofs.Flock.EndToEnd.Served.Honest
import Proofs.Flock.EndToEnd.Served.Uniform
import Proofs.Flock.EndToEnd.Joint.Bound

/-!
# Proof that the served sampling block's draw is a `StratifiedBlock` play

* `Flock.SecurityProofs.ServedDraw`: under A3 at the coins the served tapes are uniform (`Served.served_uniform`), and
  wherever the served draw on the coins ends, the live draw on the tapes ends with the same units
  (`Served.drawUnits_runOS`), so it draws their set (`served_draw`).
* `Flock.SecurityProofs.JointServed`: `JointStratified` at `StratifiedBlock.served`, whose tapes meet A3 by
  `Served.served_uniform`.
* The honest lemmas take the coins to be every byte string (`uniform_id`) and compute the honest draw
  (`Served.honest_drawUnits`). The nontrivial lemmas: a check that accepts every units array accepts two arrays at one
  outcome, whose draw is one set; under a verifier that accepts everything, the bound's event is certain.
-/

namespace Flock.SecurityProofs

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit Flock.Guarantees
open scoped ENNReal

/-- Wherever the served draw on the coins ends with `units`, the live draw on the served tapes draws exactly `units`. -/
theorem served_draw {n m : ℕ} (St : Fin m → Flock.Draw.Stratum) (K pop : ℕ) {Ωc : Type} [Fintype Ωc] [Nonempty Ωc]
    {Lc : ℕ} (coins : Ωc → Fin Lc → Fin 256) (ω : Ωc × Served.Pads m Lc) (units : Array ℕ) (read : ℕ)
    (h : SampledProofs.Draw.drawUnits (.stratified K (Array.ofFn St)) pop (Law.streamOf (coins ω.1)) =
      .ok (units, read)) :
    (Law.execOS (n := n) (fun ω : Ωc × Served.Pads m Lc => Served.tapes St (coins ω.1) ω.2)
      (.stratified K (Array.ofFn St)) pop).draw ω = Finset.univ.filter fun u => u.val ∈ units := by
  unfold Law.execOS
  dsimp only
  rw [Served.drawUnits_runOS St K pop (coins ω.1) ω.2 units read h]

/-- Coins that are every `L`-byte string, each once, are uniform: A3 holds at them. -/
theorem uniform_id (L : ℕ) :
    Assumptions.UniformRandomBytes (fun _ : Fin 1 => L) (fun (c : Fin L → Fin 256) _ => c) := by
  classical
  intro E
  unfold prCoin
  exact DrawExec.avg_equiv (Equiv.funUnique (Fin 1) (Fin L → Fin 256)).symm
    (fun f => if E f then (1 : ℝ≥0∞) else 0)

/-- **The served stratified draw is a play of the live draw on uniform tapes.** -/
theorem ServedDraw : Flock.Guarantees.ServedDraw := by
  intro n m St K pop Ωc _ _ Lc coins hc
  exact ⟨Served.served_uniform St coins hc, fun ω units read h => served_draw St K pop coins ω units read h⟩

/-- **The joint stratified bound at a served sampling block.** -/
theorem JointServed : Flock.Guarantees.JointServed := by
  intro w Ωc _ _ Lc coins hc
  refine ⟨fun ω units read h => served_draw w.St w.Kd w.pop coins ω units read h, ?_⟩
  intro δ ands bits hδ τ hT hL
  exact JointStratified (w.served Ωc Lc coins) δ ands bits hδ (Served.served_uniform w.St coins hc) τ hT hL

/-- **`ServedDraw`'s honest lemma.** -/
theorem ServedDraw_honest : Flock.Guarantees.ServedDraw_honest :=
  ⟨Law.need_pos Served.honestStratum, uniform_id _, Served.honest_drawUnits⟩

/-- **`ServedDraw`'s nontrivial lemma.** -/
theorem ServedDraw_nontrivial : Flock.Guarantees.ServedDraw_nontrivial := by
  intro acc hacc h
  have e := (h (fun _ => 0, fun _ _ => 0) #[] (hacc _)).symm.trans (h (fun _ => 0, fun _ _ => 0) #[0] (hacc _))
  have h0 : (0 : Fin 2) ∈ Finset.univ.filter fun u : Fin 2 => u.val ∈ #[0] := by simp
  rw [← e] at h0
  simp at h0

/-- **`JointServed`'s honest lemma.** -/
theorem JointServed_honest : Flock.Guarantees.JointServed_honest := by
  intro w Lc _
  exact ⟨uniform_id Lc, Served.served_uniform w.St id (uniform_id Lc)⟩

/-- **`JointServed`'s nontrivial lemma.** -/
theorem JointServed_nontrivial : Flock.Guarantees.JointServed_nontrivial := by
  intro w Ωc _ _ Lc coins δ ands bits τ acc hacc hW herr hle
  have h1 := (prob_mono (p := fun _ => True) (fun o _ => ⟨hacc o, hW⟩) _ _).trans hle
  exact absurd (lt_of_le_of_lt ((prob_true _ _).symm.le.trans h1) herr) (lt_irrefl _)

end Flock.SecurityProofs
