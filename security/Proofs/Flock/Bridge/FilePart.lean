module

public import Proofs.Flock.Bridge.FilePart.Srcs
public import Proofs.Flock.Bridge.FilePartTarget
public import Proofs.Flock.Bridge.Targets
public import Proofs.Flock.C3.Reps.ReadWhole
public import Proofs.Flock.Soundness.Discharge.Layout.Scope

/-!
# The circuit file's partition, computed (`FilePartTarget`, proved), and `RelDeployTarget`'s partition hypothesis

`filePartTarget : FilePartTarget` (`filePart_of_log`): at every descriptor whose unit net has `unitLog ≤ 32`, unit `u`'s
block of `Spot.FilePart.ofCircuit` is `blockOf` of the key program's partition at `constsOf`
(`blockOf_keyProg`). That is the block at the constant wires `constWiresOf` (`blockOf_constsOf`: past the constant and
the zero, `constsOf` fixes only inputs no unit reads): its rows' lowered gates (`FilePart.blockOf_uniform`), then the
sources it owns
(`FilePart.mem_srcs_iff`), because `RecResiduals.readIns` marks exactly the template inputs a computed row of
`Rows.ofNet` reads (`readIns_iff`).

The bound is the one `Rows.ofNet_a` and `Rows.ofNet_b` take: `readIns` reads the net's sparse rows, which hold each
column modulo `2^32`, while `Rows.ofNet` holds rows whose columns run to `2^unitLog`. Past `unitLog = 32` a computed row
of `Rows.ofNet` can read a column `readIns` doesn't mark, and the equation is then neither provable nor refutable. A
deployed statement carries it: the scope check `Layout.scopeOkH` checks it (`unitLog_le_of_scopeOkH`).

`RelDeployTarget`'s partition hypothesis, `cd.partition s = some (partOf P κ ns)`, at the key program's partition and
the circuit's constants:

* `filePartOf c n ns`: the code's partition, `FilePart.ofCircuit c n`'s blocks, unit `u`'s at slot `u`, in consecutive
  runs of the sizes `ns`.
* `partOf_ofCircuit`: at a circuit in scope it is `partOf`'s, so `relDeploy_ofCircuit` is `RelDeployTarget` with the
  partition hypothesis `cd.partition s = some (filePartOf D.unroll n ns)`.
* `why_eq_none`, `check_ok`: the population check (`FilePart.check`) passes exactly where the registration's strata,
  read stratum by stratum, are the units in order, of sizes `ns` (`FilePart.sizes`), and the public file's units are
  `0, …, N − 1`, `N` their sum: the `ns` and `n` of `filePartOf`.
-/

@[expose] public section

namespace Flock.Bridge

/-- **`readIns` marks exactly the template inputs a computed row reads**, at `unitLog ≤ 32`. -/
theorem readIns_iff {net : Flock.Net} (hU : FlockSoundness.Built net) (hlog : net.unitLog ≤ 32) (y : ℕ)
    (hy : y < (FlockSoundness.Rows.ofNet hU).nIn) :
    (Flock.RecResiduals.readIns net).getD y false = true ↔
      ∃ j : Fin (FlockSoundness.Rows.ofNet hU).nComp,
        ∃ c₀ ∈ (FlockSoundness.Rows.ofNet hU).a j ++ (FlockSoundness.Rows.ofNet hU).b j, c₀.val = y := by
  constructor
  · intro h
    obtain ⟨-, r, hr, hr', hc⟩ := (FlockC3.Reps.ReadWhole.getD_readIns net y).1 h
    exact FlockC3.Reps.ReadWhole.readCol_of_row hU hlog hr hr' hc
  · intro h
    obtain ⟨r, hr, hr', hc⟩ := FlockC3.Reps.ReadWhole.row_of_readCol (col := y) hU hlog h
    exact (FlockC3.Reps.ReadWhole.getD_readIns net y).2
      ⟨lt_of_lt_of_eq hy (FlockSoundness.Rows.ofNet_nIn hU), r, hr, hr', hc⟩

/-- **The circuit's constant wires fix exactly the constant's and the zero's gates**, `0` and `1`. -/
theorem constWiresOf_eq_none (c : Flock.Circuit) (hU : FlockSoundness.Built c.unit) (n : ℕ)
    (g : Fin (FlockSoundness.Discharge.Layout.keyN c hU n)) : constWiresOf c hU n g = none ↔ 2 ≤ g.val := by
  have h1 : (FlockSoundness.Discharge.Layout.oneG : Fin (FlockSoundness.Discharge.Layout.keyN c hU n)).val = 0 := rfl
  have h0 : (FlockSoundness.Discharge.Layout.zeroG : Fin (FlockSoundness.Discharge.Layout.keyN c hU n)).val = 1 := rfl
  unfold constWiresOf
  by_cases ho : g = FlockSoundness.Discharge.Layout.oneG
  · rw [if_pos ho]
    have : g.val = 0 := by rw [ho]; exact h1
    constructor
    · intro h; cases h
    · intro h; omega
  · rw [if_neg ho]
    by_cases hz : g = FlockSoundness.Discharge.Layout.zeroG
    · rw [if_pos hz]
      have : g.val = 1 := by rw [hz]; exact h0
      constructor
      · intro h; cases h
      · intro h; omega
    · rw [if_neg hz]
      have h1' : g.val ≠ 0 := fun h => ho (Fin.ext (h.trans h1.symm))
      have h0' : g.val ≠ 1 := fun h => hz (Fin.ext (h.trans h0.symm))
      exact ⟨fun _ => by omega, fun _ => rfl⟩

/-- **A proof block reads the constants only at its unit's inputs**: constants that agree at every gate whose owner is
`u` give unit `u` one block. -/
theorem blockOf_congr {C : Verity.Boolean.Circuit} {n : ℕ} (P : FlockSoundness.Audit.Partition C n)
    {κ κ' : Fin C.N → Option Bool} (u : Fin n) (h : ∀ g, owner P g = some u → κ g = κ' g) :
    blockOf P κ u = blockOf P κ' u := by
  have hf : ((List.finRange C.N).filter fun g => (C.op g).isInput && (κ g).isNone && decide (owner P g = some u)) =
      (List.finRange C.N).filter fun g => (C.op g).isInput && (κ' g).isNone && decide (owner P g = some u) :=
    List.filter_congr fun g _ => by
      by_cases hg : owner P g = some u
      · rw [h g hg]
      · simp [hg]
  unfold blockOf insOf
  rw [hf]

/-- **At a gate a unit reads, the circuit's constants are the constant wires'**: past the constant and the zero,
`constsOf` fixes only inputs no unit reads (`owner`). -/
theorem constsOf_eq_constWiresOf (c : Flock.Circuit) (hU : FlockSoundness.Built c.unit) (n : ℕ)
    (g : Fin (FlockSoundness.Discharge.Layout.keyN c hU n))
    (hown : owner ((FlockSoundness.Discharge.Placed.keyProg c hU n).partition
      (FlockSoundness.Discharge.Layout.outsOf c hU n)) g ≠ none) :
    constsOf c hU n g = constWiresOf c hU n g := by
  unfold constsOf constWiresOf
  by_cases ho : g = FlockSoundness.Discharge.Layout.oneG
  · rw [ite_eq_left ho, ite_eq_left ho]
  · rw [ite_eq_right ho, ite_eq_right ho]
    by_cases hz : g = FlockSoundness.Discharge.Layout.zeroG
    · rw [ite_eq_left hz]
      refine ite_eq_left ?_
      exact Or.inl hz
    · rw [ite_eq_right hz]
      refine ite_eq_right ?_
      exact fun h => h.elim hz fun h' => hown h'.2

/-- **The circuit's constants give the constant wires' blocks**: past the constant and the zero they fix only inputs no
unit reads (`owner`), which are in no block. -/
theorem blockOf_constsOf (c : Flock.Circuit) (hU : FlockSoundness.Built c.unit) (n : ℕ)
    (outs : Finset (Fin (FlockSoundness.Discharge.Layout.keyN c hU n))) (u : Fin n) :
    blockOf ((FlockSoundness.Discharge.Placed.keyProg c hU n).partition outs) (fun g => constsOf c hU n g) u =
      blockOf ((FlockSoundness.Discharge.Placed.keyProg c hU n).partition outs)
        (fun g => constWiresOf c hU n g) u := by
  refine blockOf_congr _ u fun g hg => ?_
  have hown : owner ((FlockSoundness.Discharge.Placed.keyProg c hU n).partition
      (FlockSoundness.Discharge.Layout.outsOf c hU n)) g ≠ none := by
    have : owner ((FlockSoundness.Discharge.Placed.keyProg c hU n).partition
        (FlockSoundness.Discharge.Layout.outsOf c hU n)) g =
        owner ((FlockSoundness.Discharge.Placed.keyProg c hU n).partition outs) g := rfl
    rw [this, hg]
    exact Option.some_ne_none u
  exact constsOf_eq_constWiresOf c hU n g hown

/-- **The code's block is `blockOf`'s**: at `unitLog ≤ 32`, any outputs `outs` and unit `u`, `ofCircuit`'s block `u` is
`blockOf` of the key program's partition at the circuit's constants. -/
theorem blockOf_keyProg (c : Flock.Circuit) (hU : FlockSoundness.Built c.unit) (hlog : c.unit.unitLog ≤ 32) (n : ℕ)
    (outs : Finset (Fin (FlockSoundness.Discharge.Layout.keyN c hU n))) (u : Fin n) :
    ((Spot.FilePart.ofCircuit c n).block u.val).toList =
      blockOf ((FlockSoundness.Discharge.Placed.keyProg c hU n).partition outs) (fun g => constsOf c hU n g) u := by
  have hSm := FilePart.mem_srcs_iff c (FlockSoundness.Rows.ofNet hU) (Flock.RecResiduals.readIns c.unit)
    (readIns_iff hU hlog) _ (Spot.FilePart.mem_ofCircuit_srcs c n)
  rw [Spot.FilePart.block_toList, blockOf_constsOf]
  exact (FilePart.blockOf_uniform
    (k := FlockSoundness.Discharge.Placed.inGates n (FlockSoundness.Discharge.Layout.keyB c hU))
    (R := FlockSoundness.Discharge.Placed.circuitRows c hU)
    (ins := FlockSoundness.Discharge.Placed.insOf n (FlockSoundness.Discharge.Layout.keyB c hU)
      (FlockSoundness.Discharge.Aliased.keyIdx c (FlockSoundness.Discharge.Layout.keyB c hU)))
    (one := FlockSoundness.Discharge.Placed.oneGate n _)
    (hone := FlockSoundness.Discharge.Placed.insOf_ne_one n _ _) outs (fun g => constWiresOf c hU n g)
    (B := FlockSoundness.Discharge.Layout.keyB c hU) rfl (fun g _ => constWiresOf_eq_none c hU n g)
    (Rd := fun b => ∃ i : Fin (FlockSoundness.Discharge.Placed.circuitRows c hU).nComp,
      ∃ col ∈ (FlockSoundness.Discharge.Placed.circuitRows c hU).a i ++
        (FlockSoundness.Discharge.Placed.circuitRows c hU).b i,
      ∃ hc : col.val < (FlockSoundness.Discharge.Placed.circuitRows c hU).nIn,
        (FlockSoundness.Discharge.Aliased.keyIdx c (FlockSoundness.Discharge.Layout.keyB c hU) ⟨col.val, hc⟩).map
          Fin.val = some b)
    (fun g v hg2 hgk => FilePart.reads_uniform outs g v hg2 hgk) _
    (Spot.FilePart.ofCircuit_srcs_sorted c n) hSm u).symm

/-- **`FilePartTarget` at `unitLog ≤ 32`**: at every descriptor `D` whose unit net has `unitLog ≤ 32`, unit rows `hU`,
population `n` and unit `u`, `FilePart.ofCircuit`'s block `u` is `blockOf` of the key program's partition at the
circuit's constants. -/
theorem filePart_of_log (D : Flock.Desc) (hU : FlockSoundness.Built D.unroll.unit) (hlog : D.unroll.unit.unitLog ≤ 32)
    (n : ℕ) (u : Fin n) :
    let P := (FlockSoundness.Discharge.Placed.keyProg D.unroll hU n).partition
      (FlockSoundness.Discharge.Layout.outsOf D.unroll hU n)
    ((Spot.FilePart.ofCircuit D.unroll n).block u.val).toList =
      blockOf P (fun g => constsOf D.unroll hU n g) u :=
  blockOf_keyProg D.unroll hU hlog n (FlockSoundness.Discharge.Layout.outsOf D.unroll hU n) u

theorem filePartTarget : FilePartTarget := by
  intro D hU hlog n u
  exact filePart_of_log D hU hlog n u

/-- **The scope check bounds the unit's size**: `scopeOkH` checks `unitLog ≤ 32`. -/
theorem unitLog_le_of_scopeOkH {c : Flock.Circuit} (h : FlockSoundness.Discharge.Layout.scopeOkH c = true) :
    c.unit.unitLog ≤ 32 := by
  unfold FlockSoundness.Discharge.Layout.scopeOkH at h
  simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at h
  obtain ⟨⟨⟨⟨-, -⟩, hu⟩, -⟩, -⟩ := h
  exact hu

/-- **The code's partition of the circuit file `c` at `n` units, in strata of sizes `ns`**: `FilePart.ofCircuit c n`'s
blocks, unit `u`'s at slot `u`, in consecutive runs of slots. -/
def filePartOf (c : Flock.Circuit) (n : ℕ) (ns : List ℕ) : Claims.Sampling.Partition :=
  ⟨[ns.splitLengths ((List.range n).map fun u => ((Spot.FilePart.ofCircuit c n).block u).toList)]⟩

/-- **`partOf` is the code's partition** at the key program of a circuit in scope, at the circuit's constants. -/
theorem partOf_ofCircuit (D : Flock.Desc) (hU : FlockSoundness.Built D.unroll.unit)
    (hscope : FlockSoundness.Discharge.Layout.scopeOkH D.unroll = true) (n : ℕ) (ns : List ℕ) :
    partOf ((FlockSoundness.Discharge.Placed.keyProg D.unroll hU n).partition
        (FlockSoundness.Discharge.Layout.outsOf D.unroll hU n)) (fun g => constsOf D.unroll hU n g) ns =
      filePartOf D.unroll n ns := by
  have h : (List.finRange n).map (blockOf ((FlockSoundness.Discharge.Placed.keyProg D.unroll hU n).partition
        (FlockSoundness.Discharge.Layout.outsOf D.unroll hU n)) (fun g => constsOf D.unroll hU n g)) =
      (List.range n).map fun u => ((Spot.FilePart.ofCircuit D.unroll n).block u).toList := by
    rw [← FilePart.map_val_finRange n, List.map_map]
    exact List.map_congr_left fun u _ =>
      (blockOf_keyProg D.unroll hU (unitLog_le_of_scopeOkH hscope) n
        (FlockSoundness.Discharge.Layout.outsOf D.unroll hU n) u).symm
  unfold partOf filePartOf
  rw [h]

/-- **rel_deploy at the code's partition**: `RelDeployTarget` at the key program of a circuit in scope and the
circuit's constants, its partition hypothesis the deployed partition being the code's (`filePartOf`). -/
theorem relDeploy_ofCircuit (hR : RelDeployTarget) {T : Type}
    (F : Claims.Flock.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) T) (cd : SamplingDeploy.Code)
    (D : Flock.Desc) (hU : FlockSoundness.Built D.unroll.unit)
    (hscope : FlockSoundness.Discharge.Layout.scopeOkH D.unroll = true) (n : ℕ)
    (rows : Fin ((FlockSoundness.Discharge.Placed.keyProg D.unroll hU n).circuit
      (FlockSoundness.Discharge.Layout.outsOf D.unroll hU n)).N → Option RowAt)
    (pub : Flock.Public) (nl : ℕ) (ns : List ℕ) (r : Claims.Sampling.OneStage) (desc : Claims.Msg)
    (q : Claims.Sampling.Query) (roots : List Claims.Msg)
    (Xr : Fin ((FlockSoundness.Discharge.Placed.keyProg D.unroll hU n).circuit
      (FlockSoundness.Discharge.Layout.outsOf D.unroll hU n)).N → Bool) (δ : ℝ) (x : Claims.Msg) :
    let Cc := (FlockSoundness.Discharge.Placed.keyProg D.unroll hU n).circuit
      (FlockSoundness.Discharge.Layout.outsOf D.unroll hU n)
    let P := (FlockSoundness.Discharge.Placed.keyProg D.unroll hU n).partition
      (FlockSoundness.Discharge.Layout.outsOf D.unroll hU n)
    let κ : Fin Cc.N → Option Bool := fun g => constsOf D.unroll hU n g
    LayoutIs Cc rows D.unroll pub nl roots cd →
    cd.parse x = some (stmtOf P κ r desc q roots) →
    cd.partition (stmtOf P κ r desc q roots) = some (filePartOf D.unroll n ns) →
    ¬ Claims.Sampling.RelS Verity.sha512Msg (commitDep Cc rows D.unroll pub nl roots) δ
      (stmtOf P κ r desc q roots) (partOf P κ ns)
      (witDep Cc (commitDep Cc rows D.unroll pub nl roots) Verity.sha512Msg roots Xr) →
    ¬ Claims.Sampling.R Verity.sha512Msg (SamplingDeploy.deployOf F cd).commit cd.parse cd.partition δ x
      (witDep Cc (commitDep Cc rows D.unroll pub nl roots) Verity.sha512Msg roots Xr) := by
  intro Cc P κ hL hx hpart hrel
  exact hR F cd P rows D.unroll pub nl ns r κ desc q roots Xr δ x hL hx
    (hpart.trans (congrArg some (partOf_ofCircuit D hU hscope n ns).symm)) hrel

/-- **The population check's verdict**: `why` finds nothing exactly where the strata, read stratum by stratum, are the
units in order, of sizes `ns`, and the public file's units are `0, …, N − 1`, `N` their sum. -/
theorem why_eq_none {strata : Array (String × Array (ℕ × ℕ))} {indices : Array ℕ} :
    Spot.FilePart.why strata indices = none ↔
      ∃ ns, Spot.FilePart.sizes strata = some ns ∧ indices = Array.range (ns.foldl (· + ·) 0) := by
  unfold Spot.FilePart.why
  constructor
  · intro h
    cases hs : Spot.FilePart.sizes strata with
    | none => rw [hs] at h; simp at h
    | some ns =>
      rw [hs] at h
      dsimp only at h
      split at h
      · cases h
      · rename_i hne
        exact ⟨ns, rfl, by simpa using hne⟩
  · rintro ⟨ns, hs, rfl⟩
    rw [hs]
    simp

/-- **A session the population check accepts**: the verifier's program parses, its strata under the query are the
units in order, of sizes `ns`, and the public file's units are `0, …, N − 1`, `N` their sum. -/
theorem check_ok {program : ByteArray} {query : Lean.Json} {pubB : ByteArray}
    (h : Spot.FilePart.check program query pubB = .ok ()) :
    ∃ p strata ns, Spot.Program.parse program = .ok p ∧
      Spot.Partition.strata p query = .ok strata ∧ Spot.FilePart.sizes strata = some ns ∧
      Spot.FilePart.unitIndices pubB = .ok (Array.range (ns.foldl (· + ·) 0)) := by
  unfold Spot.FilePart.check at h
  cases hp : Spot.Program.parse program with
  | error e => simp [hp, bind, Except.bind] at h
  | ok p =>
    cases hs : Spot.Partition.strata p query with
    | error e => simp [hp, hs, bind, Except.bind] at h
    | ok strata =>
      cases hi : Spot.FilePart.unitIndices pubB with
      | error e => simp [hp, hs, hi, bind, Except.bind] at h
      | ok idx =>
        cases hw : Spot.FilePart.why strata idx with
        | some w => simp [hp, hs, hi, hw, bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at h
        | none =>
          obtain ⟨ns, hns, rfl⟩ := why_eq_none.1 hw
          exact ⟨p, strata, ns, rfl, hs, hns, rfl⟩

end Flock.Bridge
