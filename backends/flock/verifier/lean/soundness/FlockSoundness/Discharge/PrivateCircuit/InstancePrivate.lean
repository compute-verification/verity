import FlockSoundness.Discharge.PrivateCircuit.InstanceDefs
import FlockSoundness.Discharge.PrivateCircuit.Privacy
import FlockSoundness.Discharge.PrivateCircuit.InstanceHides

/-!
# Protocol 2 is circuit-private at the universal unit

`private_circuit_private`: the simulation form of circuit privacy at the hm96 registration (`hm96Scheme`). At each
registrant's salt, protocol 1's zero knowledge at the universal unit's tables (`ZkSessionView`) moves the session to
`S_shvzk` on the dummy witness; what is left reads the program only through its commit strings, which hide it up to
`2·L·δ₁` under `hash-derived-key` (`Hm96Hiding` at the pinned key, per leaf), and so the simulator reads the class and the
owed outputs only.
-/

namespace FlockSoundness.Discharge.PrivateCircuit

open Flock ZK FlockSoundness.Game FlockSoundness.Discharge.ZkView
open scoped ENNReal

/-- **Protocol 2 is circuit-private at the universal unit.** The program of class `c` is `L c` rows (`enc c p i` the
bytes of row `i`), registered as their hm96 commit strings under the registrant's uniform salts (`hm96Scheme`); the
universal unit's `--zk` tables `U c o v` read the class, the owed outputs `o` and that registration `v` only. Under
`hash-derived-key` at `δ₁` (`hKey`: hm96's commit string hides its row up to `δ₁`; `2^-193` at the pinned key), and
protocol 1's zero knowledge at the universal unit's tables at every registration of `pad C c` with the honest witness
(`hView`, `zk_session_view`'s conclusion at `δ₁`): for every dummy program `p₀` and dummy witness `w₀`, every encoding
`encV` of the session's views and every event `E` of the registration's commit strings and that encoding, the real view
(`realPr`, `pad C c` registered and run) and the simulator's (`simPr`, which reads `c`, `o`, `p₀`, `w₀` only) are within
`2·L c·δ₁ + ∑ⱼ 2·(nU + 2·nPU)·δ₁`, both ways. The leakage is the class `c` (its sizes `L c`, `J c`, `zs c` and its
tables' shape) and the owed outputs `o`. -/
theorem private_circuit_private
    {Cls Circ Out : Type} {Prog : Cls → Type} {K W Hi VL : Type} [Field K] [AddCommGroup Hi] [Fintype Hi]
    [DecidableEq Hi] (fits : Circ → Cls → Prop) (pad : Circ → (c : Cls) → Prog c) (L : Cls → ℕ)
    (enc : (c : Cls) → Prog c → Fin (L c) → List UInt8) (J : Cls → ℕ) (zs : (c : Cls) → Fin (J c) → Sizes)
    (U : (c : Cls) → Out → (Fin (L c) → Bits × ByteArray) → ∀ j, Coined K W Hi VL (zs c j))
    (δ₁ : ℝ≥0∞) (hKey : Assumptions.Hm96Hiding My saltHash δ₁)
    (c : Cls) (o : Out) (C : Circ) (w : (Fin (L c) → Salt) → Fin (J c) → W)
    (hView : ∀ r, ZkSessionView (U c o (hm96Reg (fun p i => rowBits (enc c p i)) (pad C c) r)) (w r) δ₁)
    (p₀ : Prog c) (w₀ : Fin (J c) → W) {Obs : Type} (encV : ∀ v, (∀ j, (U c o v j).tab.View) → Obs)
    (E : (Fin (L c) → Bits × ByteArray) → Obs → Prop) :
    realPr (hm96Scheme fits pad L enc J zs U) c o (pad C c) w encV E
        ≤ simPr (hm96Scheme fits pad L enc J zs U) c o p₀ w₀ encV E + (2 * L c * δ₁ + viewGap (zs c) δ₁) ∧
      simPr (hm96Scheme fits pad L enc J zs U) c o p₀ w₀ encV E
        ≤ realPr (hm96Scheme fits pad L enc J zs U) c o (pad C c) w encV E + (2 * L c * δ₁ + viewGap (zs c) δ₁) := by
  set S := hm96Scheme fits pad L enc J zs U
  set ε := 2 * (L c : ℝ≥0∞) * δ₁
  have hε : Hides (S.reg c) ε := hm96Reg_hides _ hKey
  set f : (Fin (L c) → Bits × ByteArray) → ℝ≥0∞ := fun v =>
    prCoin fun q : ∀ j, (U c o v j).tab.SimRand × ((U c o v j).tab.Hid → Salt) =>
      E v (encV v fun j => (U c o v j).simExe (w₀ j) (q j).1 (q j).2)
  set g : (Fin (L c) → Salt) → ℝ≥0∞ := fun r =>
    prCoin fun q : ∀ j, (U c o (S.reg c (pad C c) r) j).tab.Rand
        × ((U c o (S.reg c (pad C c) r) j).tab.Hid → Salt) =>
      E (S.reg c (pad C c) r)
        (encV (S.reg c (pad C c) r) fun j => (U c o (S.reg c (pad C c) r) j).viewExe (w r j) (q j).1 (q j).2)
  have hreal : realPr S c o (pad C c) w encV E = avg g := rfl
  have hsim : simPr S c o p₀ w₀ encV E = avg fun r => f (S.reg c p₀ r) := rfl
  have hstep : ∀ r, g r ≤ f (S.reg c (pad C c) r) + viewGap (zs c) δ₁ ∧
      f (S.reg c (pad C c) r) ≤ g r + viewGap (zs c) δ₁ := by
    intro r
    have h := hView r w₀ (encV (S.reg c (pad C c) r)) (E (S.reg c (pad C c) r))
    have hgap : ∀ v, ∑ j, 2 * (Fintype.card (U c o v j).tab.Hid : ℝ≥0∞) * δ₁ = viewGap (zs c) δ₁ :=
      fun v => Finset.sum_congr rfl fun j _ => by rw [card_hid]
    rw [hgap] at h
    exact h
  have hf1 : ∀ v, f v ≤ 1 := fun v => prCoin_le_one' _
  constructor
  · calc realPr S c o (pad C c) w encV E = avg g := hreal
      _ ≤ avg fun r => f (S.reg c (pad C c) r) + viewGap (zs c) δ₁ := avg_mono fun r => (hstep r).1
      _ = avg (fun r => f (S.reg c (pad C c) r)) + avg fun _ : Fin (L c) → Salt => viewGap (zs c) δ₁ := avg_add _ _
      _ ≤ (avg (fun r => f (S.reg c p₀ r)) + ε) + viewGap (zs c) δ₁ :=
          add_le_add (hε.avg_le _ _ f hf1) (avg_const_le _)
      _ = simPr S c o p₀ w₀ encV E + (ε + viewGap (zs c) δ₁) := by rw [hsim, add_assoc]
  · calc simPr S c o p₀ w₀ encV E = avg fun r => f (S.reg c p₀ r) := hsim
      _ ≤ avg (fun r => f (S.reg c (pad C c) r)) + ε := hε.avg_le _ _ f hf1
      _ ≤ avg (fun r => g r + viewGap (zs c) δ₁) + ε := add_le_add (avg_mono fun r => (hstep r).2) le_rfl
      _ = avg g + avg (fun _ : Fin (L c) → Salt => viewGap (zs c) δ₁) + ε := by rw [avg_add]
      _ ≤ avg g + viewGap (zs c) δ₁ + ε := by gcongr; exact avg_const_le _
      _ = realPr S c o (pad C c) w encV E + (ε + viewGap (zs c) δ₁) := by rw [hreal]; ring

end FlockSoundness.Discharge.PrivateCircuit
