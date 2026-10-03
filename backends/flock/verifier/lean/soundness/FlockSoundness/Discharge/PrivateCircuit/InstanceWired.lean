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
  rw [hunit.card v, hp]
  exact wired_wrong T C c hC (xOf v) (yOf v) (xIn v) (hwire v) h

/-- **An owed output of unit `i` off `evalT` is a forged or a wrong unit in `i`'s cone**: some unit feeding `i` along
`T` has a committed program other than `pad C c`'s, or is a wrong unit of the partition. -/
theorem wired_cone_unit {UCc : Audit.Circuit} {c : Universal.Cls} {P : Audit.Partition UCc (Universal.units c)}
    (C : Universal.Circ) (hC : Universal.fits C c) {progOf : (Fin UCc.N → Bool) → c.Prog}
    {xOf : (Fin UCc.N → Bool) → ℕ → ℕ → Bool} {yOf : (Fin UCc.N → Bool) → ℕ → List Bool}
    (hunit : UnitWrong P progOf xOf yOf) (T : Universal.Topo) (xIn : (Fin UCc.N → Bool) → ℕ → Bool)
    (hwire : ∀ v, Universal.Wired T c (xOf v) (yOf v) (xIn v)) (v : Fin UCc.N → Bool)
    (i : ℕ) (hi : i < Universal.size C) (h : yOf v i ≠ Universal.evalT T C (xIn v) i) :
    (∃ j ∈ coneOf T c i, progOf v j ≠ Universal.pad C c j) ∨ ∃ j ∈ coneOf T c i, j ∈ P.wrong v := by
  classical
  by_contra hno
  apply h
  refine wired_cone T C c hC (xOf v) (yOf v) (xIn v) (hwire v) i hi fun j hj => ?_
  have hjN : j < c.N := lt_of_le_of_lt hj.le (lt_of_lt_of_le hi (pad_correct C c hC).1)
  have hmem : (⟨j, hjN⟩ : Fin (Universal.units c)) ∈ coneOf T c i := Finset.mem_filter.2 ⟨Finset.mem_univ _, hj⟩
  have h1 : progOf v ⟨j, hjN⟩ = Universal.pad C c ⟨j, hjN⟩ := by
    by_contra hne
    exact hno (Or.inl ⟨_, hmem, hne⟩)
  have h2 : Universal.run c (progOf v) j (xOf v j) = yOf v j := by
    by_contra hne
    exact hno (Or.inr ⟨_, hmem, (hunit v ⟨j, hjN⟩).2 hne⟩)
  rw [← h2, Universal.run, Universal.run, dite_eq_left_of_eq_true (eq_true hjN),
    dite_eq_left_of_eq_true (eq_true hjN), h1]

/-- **A drawn unit's forged program is a collision**: where the registered check holds at `A o` and the step to the
program holds at the drawn units, accepting with a drawn unit's committed program other than `pad C c`'s is at most
`q²/2^513` under `SHA512CRStrict` for the finder that outputs `progPairOn` (`private_circuit_sound_drawn`'s step). -/
theorem drawn_forged_le
    {UCc : Audit.Circuit} {c : Universal.Cls} {P : Audit.Partition UCc (Universal.units c)}
    (C : Universal.Circ) {progOf : (Fin UCc.N → Bool) → c.Prog}
    {Reg : Type} {mPts : ℕ} {plan : Finset (Fin (Universal.units c)) → Reg → List (TabSpec mPts)}
    {tab : (S : Finset (Fin (Universal.units c))) → (R : Reg) → Fin (Universal.units c) → Fin (plan S R).length}
    {decode : Audit.Partition.DecoderB (C := UCc) (Reg := Reg) plan tab}
    {lay : Binding.Layout P plan tab decode} {rs : Binding.HmRows lay} {rd : Registered.Reads lay rs}
    {rt : Flock.Registered.Port} {L : ℕ} (ps : ProgSites rd rt L)
    (s : UnitSession P Reg) (A : s.Out → lay.Pos → Prop) (path : s.Out → lay.Pos → ByteArray)
    (hchk : ∀ o, s.Accepts o → (readsOn rd (A o)).Checked (s.regOf o) (path o))
    (τ : Strategy s.G) (drawn : s.Out → Finset (Fin (Universal.units c)))
    (valPair : s.Out → List UInt8 × List UInt8)
    (hstep : LeafBindsOn ps s τ progOf (Universal.pad C c) A drawn valPair)
    (cost : s.Out → ℝ) (q : ℝ) (hcost : ∀ o, cost o ≤ q)
    (hBind : Assumptions.SHA512CRStrict H512 s.G τ (fun o => some (progPairOn ps s A path valPair o)) cost q) :
    prob (fun o => s.Accepts o ∧ ∃ u ∈ drawn o, progOf (s.vals τ o) u ≠ Universal.pad C c u) s.G τ
      ≤ ENNReal.ofReal (q ^ 2 / 2 ^ (513 : ℝ)) := by
  classical
  have hcoll : ∀ o, s.Accepts o → (∃ u ∈ drawn o, progOf (s.vals τ o) u ≠ Universal.pad C c u) →
      Binding.Collides (progPairOn ps s A path valPair o) := by
    rintro o ho ⟨u, hu, hne⟩
    by_cases hl : ∀ i, A o (ps.site i) → ps.leaves (s.regOf o) i = ps.regLeaves i
    · rw [progPairOn, ite_eq_left_of_eq_true _ _ (eq_true hl)]
      exact (hstep o ho hl).resolve_left fun h' => hne (h' u hu)
    · rw [progPairOn, ite_eq_right_of_eq_false _ _ (eq_false hl)]
      refine progCommit_binding L rt _ _ _ _ (opens_on ps (hchk o ho)) ps.opens_reg fun heq => hl fun i hi => ?_
      have h' := congrFun heq i
      simp only [openingOn, hi, ↓reduceIte] at h'
      exact h'
  have hfin : prob (fun o => ∃ x y, some (progPairOn ps s A path valPair o) = some (x, y) ∧ x ≠ y ∧
      H512 x = H512 y) s.G τ ≠ ⊤ := ne_top_of_le_ne_top ENNReal.one_ne_top (prob_le_one _ _ _)
  calc _ ≤ prob (fun o => ∃ x y, some (progPairOn ps s A path valPair o) = some (x, y) ∧ x ≠ y ∧
        H512 x = H512 y) s.G τ := by
        refine prob_mono (fun o ⟨ha, hd⟩ => ?_) s.G τ
        have h := hcoll o ha hd
        exact ⟨_, _, congrArg some Prod.mk.eta.symm, h.1, h.2⟩
    _ ≤ _ := (ENNReal.le_ofReal_iff_toReal_le hfin (by positivity)).2 (hBind hcost)

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
  classical
  have hD := drawn_forged_le C ps s A path hchk τ drawn valPair hstep cost q hcost hBind
  have h1 : prob (fun o => s.Accepts o ∧
        ((OwedWrong T C (xIn (s.vals τ o)) (yOf (s.vals τ o)) ∧ progOf (s.vals τ o) = Universal.pad C c) ∨
          ∃ u ∈ drawn o, progOf (s.vals τ o) u ≠ Universal.pad C c u)) s.G τ
      ≤ s.Lm.miss 1 + s.εks τ + s.δlink τ + ENNReal.ofReal (q ^ 2 / 2 ^ (513 : ℝ)) := by
    calc _ ≤ prob (fun o => (s.Accepts o ∧ 1 ≤ (P.wrong (s.vals τ o)).card) ∨
          (s.Accepts o ∧ ∃ u ∈ drawn o, progOf (s.vals τ o) u ≠ Universal.pad C c u)) s.G τ := by
          refine prob_mono (fun o ⟨ha, hq⟩ => ?_) s.G τ
          rcases hq with ⟨hw, hp⟩ | hd
          · exact Or.inl ⟨ha, wired_wrong_unit C hC hunit T xIn hwire _ hp hw⟩
          · exact Or.inr ⟨ha, hd⟩
      _ ≤ _ := prob_or_le _ _ s.G τ
      _ ≤ _ := add_le_add (hS hA3 hA6 τ hCR 1) hD
  refine ⟨h1, fun hF => ?_⟩
  calc _ ≤ prob (fun o => (s.Accepts o ∧
        ((OwedWrong T C (xIn (s.vals τ o)) (yOf (s.vals τ o)) ∧ progOf (s.vals τ o) = Universal.pad C c) ∨
          ∃ u ∈ drawn o, progOf (s.vals τ o) u ≠ Universal.pad C c u)) ∨
        (s.Accepts o ∧ (∀ u ∈ drawn o, u ∉ forged C c (progOf (s.vals τ o))) ∧
          1 ≤ (forged C c (progOf (s.vals τ o))).card)) s.G τ := by
        refine prob_mono (fun o ⟨ha, hw⟩ => ?_) s.G τ
        by_cases hp : progOf (s.vals τ o) = Universal.pad C c
        · exact Or.inl ⟨ha, Or.inl ⟨hw, hp⟩⟩
        by_cases hd : ∃ u ∈ drawn o, progOf (s.vals τ o) u ≠ Universal.pad C c u
        · exact Or.inl ⟨ha, Or.inr hd⟩
        obtain ⟨u, hu⟩ := Function.ne_iff.1 hp
        refine Or.inr ⟨ha, fun u' hu' hmem => hd ⟨u', hu', by simpa [forged] using hmem⟩, ?_⟩
        exact Nat.succ_le_of_lt (Finset.card_pos.2 ⟨u, by unfold forged; exact Finset.mem_filter.2 ⟨Finset.mem_univ _, hu⟩⟩)
    _ ≤ _ := prob_or_le _ _ s.G τ
    _ ≤ _ := add_le_add h1 hF
    _ = _ := by ring

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
  classical
  calc _ ≤ prob (fun o => ((s.Accepts o ∧ ∃ j ∈ coneOf T c i, j ∈ P.wrong (s.vals τ o)) ∨
          (s.Accepts o ∧ (∀ u ∈ drawn o, progOf (s.vals τ o) u = Universal.pad C c u) ∧
            ∃ j ∈ coneOf T c i, progOf (s.vals τ o) j ≠ Universal.pad C c j)) ∨
          (s.Accepts o ∧ ∃ u ∈ drawn o, progOf (s.vals τ o) u ≠ Universal.pad C c u)) s.G τ := by
        refine prob_mono (fun o ⟨ha, hne⟩ => ?_) s.G τ
        by_cases hd : ∃ u ∈ drawn o, progOf (s.vals τ o) u ≠ Universal.pad C c u
        · exact Or.inr ⟨ha, hd⟩
        have hd' : ∀ u ∈ drawn o, progOf (s.vals τ o) u = Universal.pad C c u := fun u hu => by
          by_contra h
          exact hd ⟨u, hu, h⟩
        rcases wired_cone_unit C hC hunit T xIn hwire _ i hi hne with hf | hw
        · exact Or.inl (Or.inr ⟨ha, hd', hf⟩)
        · exact Or.inl (Or.inl ⟨ha, hw⟩)
    _ ≤ _ := (prob_or_le _ _ s.G τ).trans
        (add_le_add (prob_or_le _ _ s.G τ) (drawn_forged_le C ps s A path hchk τ drawn valPair hstep cost q hcost hBind))

end FlockSoundness.Discharge.PrivateCircuit
