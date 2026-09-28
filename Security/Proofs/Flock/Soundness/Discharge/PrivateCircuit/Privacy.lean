import Proofs.Flock.Soundness.Discharge.PrivateCircuit.Statements
import Proofs.Flock.Soundness.Discharge.PrivateCircuit.Hides
import Proofs.Flock.Soundness.Discharge.ZkView.Fixed

/-!
# Circuit privacy from protocol 1's zero knowledge

`circuit_private_of_zk`: at every registrant's salt, `ZkWireFixed` at the universal unit's tables moves the session to
`S_shvzk` on the dummy witness, at `zkGap` (`gap_eq`: `ZkWireFixed`'s bound at the tables is `zkGap` of their sizes);
what is left reads the program only through its registration, which hides it up to `ε` (`Hides.avg_le`, at the test
`f v`, the simulator's probability at the view `v`). `circuit_indist_of_zk` is two such steps through the simulator.
`circuit_private` and `circuit_indist` discharge `ZkWireFixed` with `zk_wire_fixed`; the `_of_zk` forms take it as a
hypothesis, to compose with a later form of protocol 1's zero knowledge.
-/

namespace FlockSoundness.Discharge.PrivateCircuit

open Flock ZK FlockSoundness.Game FlockSoundness.Discharge.ZkView
open scoped ENNReal

theorem card_hid {K W Hi VL : Type} [Field K] {z : Sizes} (T : Coined K W Hi VL z) :
    Fintype.card T.tab.Hid = z.nU + 2 * z.nPU := by
  simp only [Table.Hid, Fintype.card_sum, Fintype.card_prod, Fintype.card_fin]
  rfl

theorem gap_eq {K W Hi VL : Type} [Field K] {J : ℕ} {zs : Fin J → Sizes} (T : ∀ j, Coined K W Hi VL (zs j)) :
    ∑ j, 2 * (Fintype.card (T j).tab.Hid : ℝ≥0∞) * ENNReal.ofReal (CROnly.hidingGap My saltHash) = zkGap zs := by
  unfold zkGap
  exact Finset.sum_congr rfl fun j _ => by rw [card_hid]

theorem circuit_private_of_zk (hzk : ZkWireFixed) : CircuitPrivate := by
  intro Cls Circ Out Prog Rr Rv _ K W Hi VL _ _ _ _ _ _ S c ε hε o C w hU hw p₀ w₀ Obs enc E
  set f : Rv c → ℝ≥0∞ := fun v =>
    prCoin fun q : ∀ j, (S.U c o v j).tab.SimRand × ((S.U c o v j).tab.Hid → Salt) =>
      E v (enc v fun j => (S.U c o v j).simExe (w₀ j) (q j).1 (q j).2)
  set g : Rr c → ℝ≥0∞ := fun r =>
    prCoin fun q : ∀ j, (S.U c o (S.reg c (S.pad C c) r) j).tab.Rand
        × ((S.U c o (S.reg c (S.pad C c) r) j).tab.Hid → Salt) =>
      E (S.reg c (S.pad C c) r)
        (enc (S.reg c (S.pad C c) r) fun j => (S.U c o (S.reg c (S.pad C c) r) j).viewExe (w r j) (q j).1 (q j).2)
  have hreal : realPr S c o (S.pad C c) w enc E = avg g := rfl
  have hsim : simPr S c o p₀ w₀ enc E = avg fun r => f (S.reg c p₀ r) := rfl
  have hstep : ∀ r, g r ≤ f (S.reg c (S.pad C c) r) + zkGap (S.zs c) ∧
      f (S.reg c (S.pad C c) r) ≤ g r + zkGap (S.zs c) := by
    intro r
    have hU' := hU r
    have h := hzk (S.zs c) (S.U c o (S.reg c (S.pad C c) r)) (enc (S.reg c (S.pad C c) r)) (fun j => (hU' j).1)
      (fun j => (hU' j).2.1) (fun j => (hU' j).2.2.1) (fun j => (hU' j).2.2.2.1) (fun j => (hU' j).2.2.2.2) (w r)
      (fun j => (hw r j).1) (fun j => (hw r j).2) w₀ (E (S.reg c (S.pad C c) r))
    rw [gap_eq] at h
    exact h
  have hf1 : ∀ v, f v ≤ 1 := fun v => prCoin_le_one' _
  constructor
  · calc realPr S c o (S.pad C c) w enc E = avg g := hreal
      _ ≤ avg fun r => f (S.reg c (S.pad C c) r) + zkGap (S.zs c) := avg_mono fun r => (hstep r).1
      _ = avg (fun r => f (S.reg c (S.pad C c) r)) + avg fun _ : Rr c => zkGap (S.zs c) := avg_add _ _
      _ ≤ (avg (fun r => f (S.reg c p₀ r)) + ε) + zkGap (S.zs c) :=
          add_le_add (hε.avg_le _ _ f hf1) (avg_const_le _)
      _ = simPr S c o p₀ w₀ enc E + (ε + zkGap (S.zs c)) := by rw [hsim, add_assoc]
  · calc simPr S c o p₀ w₀ enc E = avg fun r => f (S.reg c p₀ r) := hsim
      _ ≤ avg (fun r => f (S.reg c (S.pad C c) r)) + ε := hε.avg_le _ _ f hf1
      _ ≤ avg (fun r => g r + zkGap (S.zs c)) + ε := add_le_add (avg_mono fun r => (hstep r).2) le_rfl
      _ = avg g + avg (fun _ : Rr c => zkGap (S.zs c)) + ε := by rw [avg_add]
      _ ≤ avg g + zkGap (S.zs c) + ε := by gcongr; exact avg_const_le _
      _ = realPr S c o (S.pad C c) w enc E + (ε + zkGap (S.zs c)) := by rw [hreal]; ring

theorem circuit_indist_of_zk (hzk : ZkWireFixed) : CircuitIndist := by
  intro Cls Circ Out Prog Rr Rv _ K W Hi VL _ _ _ _ _ _ S c ε hε o C C' w w' hU hU' hw hw' Obs enc E
  rcases isEmpty_or_nonempty (Rr c) with hR | hR
  · simp [realPr, avg]
  obtain ⟨r₀⟩ := hR
  have h₁ := (circuit_private_of_zk hzk S c ε hε o C w hU hw (S.pad C' c) (w r₀) enc E).1
  have h₂ := (circuit_private_of_zk hzk S c ε hε o C' w' hU' hw' (S.pad C' c) (w r₀) enc E).2
  calc realPr S c o (S.pad C c) w enc E ≤ simPr S c o (S.pad C' c) (w r₀) enc E + (ε + zkGap (S.zs c)) := h₁
    _ ≤ realPr S c o (S.pad C' c) w' enc E + (ε + zkGap (S.zs c)) + (ε + zkGap (S.zs c)) := by gcongr
    _ = realPr S c o (S.pad C' c) w' enc E + 2 * (ε + zkGap (S.zs c)) := by ring

/-- **Circuit privacy, the simulation form**, at protocol 1's proved zero knowledge (`zk_wire_fixed`). -/
theorem circuit_private : CircuitPrivate := circuit_private_of_zk zk_wire_fixed

/-- **Circuit privacy, the indistinguishability form**, at protocol 1's proved zero knowledge (`zk_wire_fixed`). -/
theorem circuit_indist : CircuitIndist := circuit_indist_of_zk zk_wire_fixed

end FlockSoundness.Discharge.PrivateCircuit
