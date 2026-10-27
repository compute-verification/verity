import Proofs.Flock.Bridge.PublicWitnesses

/-!
# A hand-built circuit file meeting `sampledKSLiveV2Z_unopened`'s structural premises

`sampledKSLiveV2Z_unopened` reads three facts of the circuit `D.unroll` besides the parameters: it has no port, its
key program returns a computed gate, and its units read or write at most `2^40` committed wires. `toyDesc` is a
`Flock.Desc` built by hand, a flat file of one net, and `toyDesc_structural` proves all three of it at `n = 1`.

Its unit net (`toyNet`) is `Net.ofRows`'s, the one constructor `Built` names, with one computed row (the constant
squared), the constant, and an output group of one word at column 0. That `Net.ofRows` accepts those rows is checked by
the kernel (`toyRows_isOk`); its fields are then read back through `Net.ofRows_ok`. With one copy (`upv = 1`) and one
returned output on its leaf, the key program's unit has the copy's constant at column 0, the computed row at column 1
and the shared constant at column 2: column 1 is the returned output's bit 0 (`OutColAt`), so its gate is in `outsOf`,
and it is a row. The key program has `2 + 2 = 4` gates.

No circuit file parses to it from an untyped statement in scope (`toyDesc_unparsed`, `HiddenReturns`): there a
returned gate is an output row's bit, so a circuit that returns one has a port.
-/

namespace Flock.Bridge

open FlockSoundness FlockSoundness.Discharge

/-- **The toy's unit net, as `Net.ofRows` builds it**: row 0 computed (`1·1`, the constant squared), row 1 the constant,
no input group, and one output group of one word at column 0. -/
def toyRows : Except String Flock.Net :=
  Flock.Net.ofRows "u" 2 1 #[] #[⟨0, 1, #[16]⟩] #[#[1], #[1]] #[#[1], #[1]]

theorem toyRows_isOk : (match toyRows with | .ok _ => true | .error _ => false) = true := by decide +kernel

/-- The net `toyRows` builds. -/
def toyNet : Flock.Net := match toyRows with | .ok n => n | .error _ => default

theorem toyRows_eq : toyRows = .ok toyNet := by
  have h := toyRows_isOk
  unfold toyNet
  revert h
  cases toyRows <;> simp

/-- **The toy circuit file**: the net `toyNet` as its unit, one copy, one returned output on the copy's leaf, no port. -/
def toyC : Flock.Circuit where
  name := "toy"
  sha := ByteArray.empty
  nets := #[("u", toyNet)]
  ranges := #[]
  ports := #[]
  outPorts := #[]
  g := 1
  kLog := 7
  upv := 1
  unitNet := 0
  leavesIn := #[#[]]
  leavesOut := #[#[0]]
  key := ByteArray.empty
  dummyDigests := #[]
  dummyOutputs := #[]
  leafScheme := ""
  perVu := #[]
  wires := #[]

/-- **The toy descriptor**: the flat file `toyC`. -/
def toyDesc : Flock.Desc := .single toyC

theorem toy_built : Built toyC.unit :=
  ⟨"u", 2, 1, #[], #[⟨0, 1, #[16]⟩], #[#[1], #[1]], #[#[1], #[1]], toyRows_eq⟩

theorem toy_facts : toyC.unit.constPos = 1 ∧ toyC.unit.inWords = 0 ∧ toyC.unit.outGroups = #[⟨0, 1, #[16]⟩] := by
  obtain ⟨-, -, -, -, hcp, -, hog, -, -, -, -, -, -, hiw, -⟩ := Flock.Net.ofRows_ok (n := toyC.unit) toyRows_eq
  exact ⟨hcp, by rw [hiw]; rfl, hog⟩

theorem toy_nIn (hU : Built toyC.unit) : (Rows.ofNet hU).nIn = 0 := by
  rw [Rows.ofNet_nIn, toy_facts.2.1]
  simp

theorem toy_nComp (hU : Built toyC.unit) : (Rows.ofNet hU).nComp = 1 := by
  have h : (Rows.ofNet hU).nComp = toyC.unit.constPos - min (toyC.unit.inWords * Flock.WORD) toyC.unit.constPos := rfl
  rw [h, toy_facts.2.1, toy_facts.1]
  simp

/-- **The toy meets the structural premises of `sampledKSLiveV2Z_unopened`** at `n = 1`: no port, a returned gate
that is computed, and at most `2^40` committed wires read or written. -/
theorem toy_structural (hU : Built toyC.unit) :
    toyC.ports.size = 0 ∧
      (∃ g ∈ ((Placed.keyProg toyC hU 1).circuit (Layout.outsOf toyC hU 1)).outputs,
        (((Placed.keyProg toyC hU 1).circuit (Layout.outsOf toyC hU 1)).op g).isInput = false) ∧
      ioSum ((Placed.keyProg toyC hU 1).partition (Layout.outsOf toyC hU 1)) ≤ 2 ^ 40 := by
  have hnIn := toy_nIn hU
  have hnC := toy_nComp hU
  have hup : toyC.upv = 1 := rfl
  have eR := Placed.rows_circuitProg toyC hU 1 (Layout.keyB toyC hU) (Aliased.keyIdx toyC (Layout.keyB toyC hU)) 0
  have eN : ((Placed.keyProg toyC hU 1).rows 0).nIn = 0 := by
    rw [congrArg Rows.nIn eR]
    show toyC.upv * (Rows.ofNet hU).nIn = 0
    simp [hnIn]
  have eC : ((Placed.keyProg toyC hU 1).rows 0).nComp = 2 := by
    rw [congrArg Rows.nComp eR]
    show toyC.upv * ((Rows.ofNet hU).nComp + 1) = 2
    simp [hnC, hup]
  have hoc0 : Layout.outCol toyC 0 0 = 0 := by
    unfold Layout.outCol
    rw [toy_facts.2.2]
    rfl
  refine ⟨rfl, ⟨(Const.Prog.instC (Placed.keyProg toyC hU 1) (Layout.outsOf toyC hU 1) 0).wire
      (((Placed.keyProg toyC hU 1).rows 0).comp ⟨1, by omega⟩), ?_, ?_⟩, ?_⟩
  · show _ ∈ Layout.outsOf toyC hU 1
    refine (@Finset.mem_filter _ _ (Classical.decPred _) _ _).2
      ⟨Finset.mem_univ _, 0, ((Placed.keyProg toyC hU 1).rows 0).comp ⟨1, by omega⟩, rfl, 0, 0, 0, ?_⟩
    refine ⟨by decide, by decide, by decide, ?_, ?_, ?_⟩
    · rw [hnIn]
      exact Nat.zero_le _
    · rw [hnIn, hnC, hoc0]
      decide
    · show ((Placed.keyProg toyC hU 1).rows 0).nIn + 1 = _
      rw [eN, hnIn, hnC, hoc0, hup]
  · rw [(Const.Prog.instC (Placed.keyProg toyC hU 1) (Layout.outsOf toyC hU 1) 0).op]
    rfl
  · have hN : ((Placed.keyProg toyC hU 1).circuit (Layout.outsOf toyC hU 1)).N = 4 := by
      show 2 + 1 * (toyC.upv * (Rows.ofNet hU).nIn) + toyC.upv * ((Rows.ofNet hU).nComp + 1) = 4
      rw [hnIn, hnC, hup]
    unfold ioSum
    calc ∑ u, (((Placed.keyProg toyC hU 1).partition (Layout.outsOf toyC hU 1)).io u).card
        ≤ ∑ _u : Fin 1, ((Placed.keyProg toyC hU 1).circuit (Layout.outsOf toyC hU 1)).N :=
          Finset.sum_le_sum fun u _ => (Finset.card_le_univ _).trans_eq (Fintype.card_fin _)
      _ ≤ 2 ^ 40 := by simp [hN]

/-- **`toyDesc` meets the structural premises of `sampledKSLiveV2Z_unopened`** (`D.unroll.ports.size = 0`, a computed
returned gate of `Cc`, `ioSum P ≤ 2^40`) at `n = 1`, for any proof that its unit net is built. -/
theorem toyDesc_structural (hU : Built toyDesc.unroll.unit) :
    toyDesc.unroll.ports.size = 0 ∧
      (∃ g ∈ ((Placed.keyProg toyDesc.unroll hU 1).circuit (Layout.outsOf toyDesc.unroll hU 1)).outputs,
        (((Placed.keyProg toyDesc.unroll hU 1).circuit (Layout.outsOf toyDesc.unroll hU 1)).op g).isInput = false) ∧
      ioSum ((Placed.keyProg toyDesc.unroll hU 1).partition (Layout.outsOf toyDesc.unroll hU 1)) ≤ 2 ^ 40 :=
  toy_structural hU

end Flock.Bridge
