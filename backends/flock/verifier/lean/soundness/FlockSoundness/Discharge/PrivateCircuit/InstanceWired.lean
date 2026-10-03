import FlockSoundness.Discharge.PrivateCircuit.InstanceDrawn
import FlockSoundness.Discharge.PrivateCircuit.Wiring

/-!
# Protocol 2 at the universal unit, end to end through the public wiring

`private_circuit_sound_drawn` counts wrong units. What the verifier owes the user is `C`'s outputs end to end on the
circuit's inputs: `evalT T C x` along the class's public topology `T`. Here the committed values carry the circuit's
inputs too (`xIn v`, the values on the circuit-input wires), and the claimed inputs and outputs agree along `T`
(`hwire`, `Universal.Wired` at every assignment). `hwire` is a named hypothesis, as `UnitWrong` is: in the universal
circuit a unit's claimed input and its producer's claimed output read one cut wire of the one assignment, so it holds
by the layout. For the executable that is the layout in which a reader's input rows are staged under the writer's
salt and the verifier checks that each cut wire's leaves are equal; that layout discharges `hwire`.

* **The owed outputs** (`private_circuit_sound_wired`). An owed output other than `evalT`'s, with the committed program
  `pad C c`'s, is at least one wrong unit of the universal unit's partition (`wired_wrong`, `UnitWrong.card`), so the
  `K₀ = 1` bound applies; with a forged program at undrawn units, `ForgedMiss` at `1` adds one more `miss(1)`.
* **The cone** (`wired_cone_unit`, `private_circuit_sound_cone`). An owed output of unit `i` other than `evalT`'s is a
  forged or wrong unit among those feeding `i` (`coneOf T c i`, `wired_cone`). Under a sparse law the useful bound is
  the law's miss at a cone, not at a count, so the cone form leaves the two cone events as probabilities.
-/

namespace FlockSoundness.Discharge.PrivateCircuit

open Flock FlockSoundness.Game FlockSoundness.Registered Refine Binding Audit
open scoped ENNReal

/-- **`C`'s owed outputs are wrong**: some unit `i` of `C` claims outputs other than `C`'s end to end on the circuit's
inputs `x` along `T`. -/
def OwedWrong (T : Universal.Topo) (C : Universal.Circ) (x : ℕ → Bool) (y : ℕ → List Bool) : Prop :=
  ∃ i < Universal.size C, y i ≠ Universal.evalT T C x i

open Classical in
/-- **The cone of unit `i`**: the class's units that feed `i` along `T` (`Reaches`), `i` included. -/
noncomputable def coneOf (T : Universal.Topo) (c : Universal.Cls) (i : ℕ) : Finset (Fin (Universal.units c)) :=
  Finset.univ.filter fun j => Universal.Reaches T c j.val i

/-- **An owed output off `evalT`, with the program `pad C c`'s, is a wrong unit of the partition.** -/
theorem wired_wrong_unit {UCc : Audit.Circuit} {c : Universal.Cls} {P : Audit.Partition UCc (Universal.units c)}
    (C : Universal.Circ) (hC : Universal.fits C c) {progOf : (Fin UCc.N → Bool) → c.Prog}
    {xOf : (Fin UCc.N → Bool) → ℕ → ℕ → Bool} {yOf : (Fin UCc.N → Bool) → ℕ → List Bool}
    (hunit : UnitWrong P progOf xOf yOf) (T : Universal.Topo) (xIn : (Fin UCc.N → Bool) → ℕ → Bool)
    (hwire : ∀ v, Universal.Wired T c (xOf v) (yOf v) (xIn v)) (v : Fin UCc.N → Bool)
    (hp : progOf v = Universal.pad C c) (h : OwedWrong T C (xIn v) (yOf v)) :
    1 ≤ (P.wrong v).card := by
  sorry

/-- **An owed output of unit `i` off `evalT` is a forged or a wrong unit in `i`'s cone**: some unit feeding `i` along
`T` has a committed program other than `pad C c`'s, or is a wrong unit of the partition. -/
theorem wired_cone_unit {UCc : Audit.Circuit} {c : Universal.Cls} {P : Audit.Partition UCc (Universal.units c)}
    (C : Universal.Circ) (hC : Universal.fits C c) {progOf : (Fin UCc.N → Bool) → c.Prog}
    {xOf : (Fin UCc.N → Bool) → ℕ → ℕ → Bool} {yOf : (Fin UCc.N → Bool) → ℕ → List Bool}
    (hunit : UnitWrong P progOf xOf yOf) (T : Universal.Topo) (xIn : (Fin UCc.N → Bool) → ℕ → Bool)
    (hwire : ∀ v, Universal.Wired T c (xOf v) (yOf v) (xIn v)) (v : Fin UCc.N → Bool)
    (i : ℕ) (hi : i < Universal.size C) (h : yOf v i ≠ Universal.evalT T C (xIn v) i) :
    (∃ j ∈ coneOf T c i, progOf v j ≠ Universal.pad C c j) ∨ ∃ j ∈ coneOf T c i, j ∈ P.wrong v := by
  sorry

/-- **Protocol 2 is sound end to end at the universal unit.** `private_circuit_sound_drawn`'s hypotheses, with the
circuit's inputs `xIn v` and the claimed values agreeing along the public topology `T` (`hwire`; discharged by the layout
that stages a reader's input rows under the writer's salt and checks each cut wire's leaves equal). Then for every live
prover `τ` whose knowledge finders meet `cr/sha-512` (`hCR`):

* accepting with `C`'s owed outputs wrong (`OwedWrong`) and the committed program `pad C c`'s, or with a drawn unit's
  committed program other than `pad C c`'s, has probability at most `miss(1) + ε_ks + δ_link + q²/2^513`;
* where the law misses the forged units at `1` (`ForgedMiss`), accepting with `C`'s owed outputs wrong has probability
  at most `miss(1) + miss(1) + ε_ks + δ_link + q²/2^513`. -/
theorem private_circuit_sound_wired
    {UCc : Audit.Circuit} {c : Universal.Cls} (P : Audit.Partition UCc (Universal.units c))
    (C : Universal.Circ) (hC : Universal.fits C c)
    (progOf : (Fin UCc.N → Bool) → c.Prog) (xOf : (Fin UCc.N → Bool) → ℕ → ℕ → Bool)
    (yOf : (Fin UCc.N → Bool) → ℕ → List Bool) (hunit : UnitWrong P progOf xOf yOf)
    (T : Universal.Topo) (xIn : (Fin UCc.N → Bool) → ℕ → Bool)
    (hwire : ∀ v, Universal.Wired T c (xOf v) (yOf v) (xIn v))
    {Reg : Type} {mPts : ℕ} {plan : Finset (Fin (Universal.units c)) → Reg → List (TabSpec mPts)}
    {tab : (S : Finset (Fin (Universal.units c))) → (R : Reg) → Fin (Universal.units c) → Fin (plan S R).length}
    {decode : Audit.Partition.DecoderB (C := UCc) (Reg := Reg) plan tab}
    {lay : Binding.Layout P plan tab decode} {rs : Binding.HmRows lay} (rd : Registered.Reads lay rs)
    {rt : Flock.Registered.Port} {L : ℕ} (ps : ProgSites rd rt L)
    (s : UnitSession P Reg) (hS : ZkSessionSound s)
    (hA3 : Assumptions.UniformRandomBytes s.Lb s.src) (hA6 : s.A6)
    (A : s.Out → lay.Pos → Prop) (path : s.Out → lay.Pos → ByteArray)
    (hchk : ∀ o, s.Accepts o → (readsOn rd (A o)).Checked (s.regOf o) (path o))
    (τ : Strategy s.G) (hCR : s.CR τ) (drawn : s.Out → Finset (Fin (Universal.units c)))
    (valPair : s.Out → List UInt8 × List UInt8)
    (hstep : LeafBindsOn ps s τ progOf (Universal.pad C c) A drawn valPair)
    (cost : s.Out → ℝ) (q : ℝ) (hcost : ∀ o, cost o ≤ q)
    (hBind : Assumptions.SHA512CRStrict H512 s.G τ (fun o => some (progPairOn ps s A path valPair o)) cost q) :
    prob (fun o => s.Accepts o ∧
        ((OwedWrong T C (xIn (s.vals τ o)) (yOf (s.vals τ o)) ∧ progOf (s.vals τ o) = Universal.pad C c) ∨
          ∃ u ∈ drawn o, progOf (s.vals τ o) u ≠ Universal.pad C c u)) s.G τ
      ≤ s.Lm.miss 1 + s.εks τ + s.δlink τ + ENNReal.ofReal (q ^ 2 / 2 ^ (513 : ℝ)) ∧
    (ForgedMiss s τ drawn (fun o => forged C c (progOf (s.vals τ o))) 1 →
      prob (fun o => s.Accepts o ∧ OwedWrong T C (xIn (s.vals τ o)) (yOf (s.vals τ o))) s.G τ
        ≤ s.Lm.miss 1 + s.Lm.miss 1 + s.εks τ + s.δlink τ + ENNReal.ofReal (q ^ 2 / 2 ^ (513 : ℝ))) := by
  sorry

/-- **Protocol 2's cone bound at the universal unit.** Under `private_circuit_sound_wired`'s hypotheses, for every unit
`i` of `C`: accepting with unit `i`'s owed outputs other than `evalT`'s has probability at most that of accepting with a
wrong unit of the partition in `i`'s cone (`coneOf`), plus that of accepting with a forged unit in `i`'s cone and no
drawn unit forged, plus `q²/2^513`. A law that draws from every cone bounds both cone events. -/
theorem private_circuit_sound_cone
    {UCc : Audit.Circuit} {c : Universal.Cls} (P : Audit.Partition UCc (Universal.units c))
    (C : Universal.Circ) (hC : Universal.fits C c)
    (progOf : (Fin UCc.N → Bool) → c.Prog) (xOf : (Fin UCc.N → Bool) → ℕ → ℕ → Bool)
    (yOf : (Fin UCc.N → Bool) → ℕ → List Bool) (hunit : UnitWrong P progOf xOf yOf)
    (T : Universal.Topo) (xIn : (Fin UCc.N → Bool) → ℕ → Bool)
    (hwire : ∀ v, Universal.Wired T c (xOf v) (yOf v) (xIn v))
    {Reg : Type} {mPts : ℕ} {plan : Finset (Fin (Universal.units c)) → Reg → List (TabSpec mPts)}
    {tab : (S : Finset (Fin (Universal.units c))) → (R : Reg) → Fin (Universal.units c) → Fin (plan S R).length}
    {decode : Audit.Partition.DecoderB (C := UCc) (Reg := Reg) plan tab}
    {lay : Binding.Layout P plan tab decode} {rs : Binding.HmRows lay} (rd : Registered.Reads lay rs)
    {rt : Flock.Registered.Port} {L : ℕ} (ps : ProgSites rd rt L)
    (s : UnitSession P Reg)
    (A : s.Out → lay.Pos → Prop) (path : s.Out → lay.Pos → ByteArray)
    (hchk : ∀ o, s.Accepts o → (readsOn rd (A o)).Checked (s.regOf o) (path o))
    (τ : Strategy s.G) (drawn : s.Out → Finset (Fin (Universal.units c)))
    (valPair : s.Out → List UInt8 × List UInt8)
    (hstep : LeafBindsOn ps s τ progOf (Universal.pad C c) A drawn valPair)
    (cost : s.Out → ℝ) (q : ℝ) (hcost : ∀ o, cost o ≤ q)
    (hBind : Assumptions.SHA512CRStrict H512 s.G τ (fun o => some (progPairOn ps s A path valPair o)) cost q)
    (i : ℕ) (hi : i < Universal.size C) :
    prob (fun o => s.Accepts o ∧ yOf (s.vals τ o) i ≠ Universal.evalT T C (xIn (s.vals τ o)) i) s.G τ
      ≤ prob (fun o => s.Accepts o ∧ ∃ j ∈ coneOf T c i, j ∈ P.wrong (s.vals τ o)) s.G τ +
        prob (fun o => s.Accepts o ∧ (∀ u ∈ drawn o, progOf (s.vals τ o) u = Universal.pad C c u) ∧
          ∃ j ∈ coneOf T c i, progOf (s.vals τ o) j ≠ Universal.pad C c j) s.G τ +
        ENNReal.ofReal (q ^ 2 / 2 ^ (513 : ℝ)) := by
  sorry

end FlockSoundness.Discharge.PrivateCircuit
