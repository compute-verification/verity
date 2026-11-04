import Proofs.Flock.Bridge.KeyPartitions
import Proofs.Flock.Claims.Sampling.Reference.DeployedCode
import Proofs.Flock.Claims.Sampling.Select.Scope
import Proofs.Flock.Claims.Sampling.Select.Steps
import Proofs.Flock.Soundness.Discharge.Hm.Bc
import Proofs.Flock.Soundness.ExecSetup

/-!
# The sampled verifier's code over a registered circuit file

The bridge's theorems quantify over any `SamplingDeploy.Code` whose layout is `commitDep`'s (`LayoutIs`). Here are the
three fields the sampled verifier computes from its registration, and what they establish:

* `FileReg`: a registration (`SamplingDeploy.Reg`) over a circuit file the circuit parser accepts (its unit built, a
  unit slot per VU: `facts_of_parse`), the descriptor bytes its program parses from, the claims' form of its query, and
  how a statement's bytes carry its public file and roots (`read`).
* `parseOf g`: a statement whose public file passes the circuit file's partition checks (`FilePart.checkFile`: the
  registration's population the public file's units in order, each unit in a public size class) and whose circuit is
  in the bridge's scope (`Layout.scopeOkH`) parses to `stmtOf` of the key program at the registration's strata's sum of
  units (`FileReg.sizes`), its design the registration's (`FileReg.rule`, `slotDesign`'s one rule).
* `partitionOf g`: `Slots.fileLayout`'s partition (`slotPartition`); `scopeOf g`: whether `parseOf g` parses.
* `CodeIs g cd`: `cd`'s three fields are those (`codeOf g c`: `c` with them).

At such a code:

* `programScope_code`: `ProgramScope`. A statement it parses is `progOf`, topologically ordered
  (`progOf_topological`), and its partition is `partOf` (`partitionOf_parse`, from `partOf_ofCircuit`), which
  partitions it (`keyPart_partitions`) because the class check bounds each unit's committed bits (`le_of_classWhy`).
* `regMatches_code`: `SamplingDeploy.RegMatches`, wherever the statement's registration is `g`'s.
* `stmtIsFile_code`: `StmtIsFile`, every statement it parses carries the registration's descriptor and the circuit
  file's key program lowered (`progOf`), in place of `StmtFlattens`.
* `relDeploy_code`, `relDeployL_code`: `RelDeployTarget` and V5's leaf-map form with the partition hypothesis
  discharged.
-/

namespace Flock.Bridge

open Claims Claims.Sampling

/-- **A registration over its circuit file**: the draw of record's registration (`reg`: the verifier's program, query
and work table, the selection rule and the circuit file `reg.circuit`), the facts the circuit parser establishes of the
file (`facts_of_parse`): its unit is a built net (`built`) with a unit slot per VU at least (`upv`); the descriptor
bytes its program parses from (`desc`, `program`), the claims' form of its query (`query`), and the public file and the
roots a statement's bytes carry, none where they name no statement of this registration (`read`). -/
structure FileReg where
  reg : SamplingDeploy.Reg
  built : FlockSoundness.Built reg.circuit.unit
  upv : 0 < reg.circuit.upv
  desc : Msg
  program : SampledProofs.Program.parse ⟨desc.toArray⟩ = .ok reg.program
  query : Query
  read : Msg → Option (ByteArray × List Msg)

/-- **The circuit parser establishes `FileReg`'s facts** of a circuit file in the bridge's scope. -/
theorem facts_of_parse {tags : Flock.Tags} {cf : ByteArray} {tables : List (String × ByteArray)} {c : Flock.Circuit}
    (h : Flock.HmRow.parse tags cf tables = .ok c) (hs : FlockSoundness.Discharge.Layout.scopeOkH c = true) :
    FlockSoundness.Built c.unit ∧ 0 < c.upv := by
  have P := FlockSoundness.parse_facts h
  obtain ⟨ls, hls⟩ := P.check
  have C := FlockSoundness.check_facts P.typed hls
  have hU := FlockSoundness.unit_built P C
  exact ⟨hU, FlockSoundness.Discharge.Hm.upv_pos C (FlockSoundness.Discharge.Layout.scopeOkH_spec hU hs).2.1⟩

namespace FileReg

variable (g : FileReg)

/-- The key program's circuit at `n` units. -/
noncomputable abbrev circ (n : ℕ) : FlockSoundness.Audit.Circuit :=
  (FlockSoundness.Discharge.Placed.keyProg g.reg.circuit g.built n).circuit
    (FlockSoundness.Discharge.Layout.outsOf g.reg.circuit g.built n)

/-- The key program's partition at `n` units. -/
noncomputable abbrev part (n : ℕ) : FlockSoundness.Audit.Partition (g.circ n) n :=
  (FlockSoundness.Discharge.Placed.keyProg g.reg.circuit g.built n).partition
    (FlockSoundness.Discharge.Layout.outsOf g.reg.circuit g.built n)

/-- The circuit's constants at `n` units. -/
noncomputable abbrev consts (n : ℕ) : Fin (g.circ n).N → Option Bool := fun w => constsOf g.reg.circuit g.built n w

/-- The registration's strata's sizes (`FilePart.sizes` of `Partition.strata`), none where its query doesn't apply or
the strata, read stratum by stratum, are not its units in order. -/
def sizes : Option (Array ℕ) :=
  match SampledProofs.Partition.strata g.reg.program g.reg.query with
  | .ok S => SampledProofs.FilePart.sizes S
  | .error _ => none

/-- The registration's one-stage design: `slotDesign`'s one rule, none where it refuses. -/
def rule : Option OneStage :=
  match Properties.SampledProofs.slotDesign g.reg.rule g.reg.program g.reg.query g.reg.table with
  | .ok { rules := [r] } => some r
  | _ => none

end FileReg

open Classical in
/-- **The deployed parse** at the registration `g`: a statement whose public file passes the circuit file's partition
checks (`FilePart.checkFile`) and whose circuit is in scope (`Layout.scopeOkH`) is `stmtOf` of the key program at the
registration's strata's sum of units, at the circuit's constants, the registration's design, descriptor and query, and
the statement's roots. -/
noncomputable def parseOf (g : FileReg) (x : Msg) : Option AuditStatement :=
  match g.read x, g.sizes, g.rule with
  | some (pub, roots), some ns, some r =>
    if FlockSoundness.Discharge.Layout.scopeOkH g.reg.circuit = true ∧
        SampledProofs.FilePart.checkFile ⟨g.desc.toArray⟩ g.reg.query pub (.ok g.reg.circuit) = .ok () then
      some (stmtOf (g.part ns.toList.sum) (g.consts ns.toList.sum) r g.desc g.query roots)
    else none
  | _, _, _ => none

/-- **The deployed partition** at the registration `g`: its circuit file's layout (`Slots.fileLayout`) as the claims'
partition. -/
def partitionOf (g : FileReg) (_s : AuditStatement) : Option Partition :=
  match SampledProofs.Slots.fileLayout g.reg.circuit g.reg.program g.reg.query with
  | .ok l => some (Properties.SampledProofs.slotPartition l)
  | .error _ => none

/-- **The deployed scope check** at the registration `g`: the statement parses. -/
noncomputable def scopeOf (g : FileReg) (x : Msg) : Bool := (parseOf g x).isSome

/-- **`cd` is the code at the registration `g`**: its parse, scope check and partition are `g`'s. -/
def CodeIs (g : FileReg) (cd : SamplingDeploy.Code) : Prop :=
  cd.parse = parseOf g ∧ cd.scopeCheck = scopeOf g ∧ cd.partition = partitionOf g

/-- The code `c` with the parse, scope check and partition at the registration `g`. -/
noncomputable def codeOf (g : FileReg) (c : SamplingDeploy.Code) : SamplingDeploy.Code :=
  { c with parse := parseOf g, scopeCheck := scopeOf g, partition := partitionOf g }

theorem codeIs_codeOf (g : FileReg) (c : SamplingDeploy.Code) : CodeIs g (codeOf g c) := ⟨rfl, rfl, rfl⟩

/-! ## The deployed checks -/

/-- **A unit the class check lets through commits at most `unitClassMax` bits.** -/
theorem le_of_classWhy {bits : ℕ} (h : SampledProofs.FilePart.classWhy bits = none) :
    bits ≤ Claims.Sampling.unitClassMax := by
  unfold SampledProofs.FilePart.classWhy at h
  by_cases hs : (SampledProofs.Partition.unitClass bits).isSome = true
  · obtain ⟨w, hw⟩ := Option.isSome_iff_exists.1 hs
    unfold SampledProofs.Partition.unitClass at hw
    have hle : bits ≤ w := by simpa using List.find?_some hw
    have hmem := List.mem_of_find?_eq_some hw
    have h16 : SampledProofs.Partition.UNIT_CLASS_MAX / 16 = 2 ^ 14 := by
      norm_num [SampledProofs.Partition.UNIT_CLASS_MAX]
    unfold SampledProofs.Partition.UNIT_CLASSES at hmem
    rw [h16, Nat.log2_two_pow] at hmem
    obtain ⟨k, hk, rfl⟩ := List.mem_map.1 hmem
    have hk' : k ≤ 14 := by
      have := List.mem_range.1 hk
      omega
    calc bits ≤ 16 * 2 ^ k := hle
      _ ≤ 16 * 2 ^ 14 := Nat.mul_le_mul_left 16 (Nat.pow_le_pow_right (by norm_num) hk')
      _ = Claims.Sampling.unitClassMax := by norm_num [Claims.Sampling.unitClassMax]
  · simp [hs] at h

/-- **The circuit file's partition checks, passed**: the population check and the class check. -/
theorem checkFile_ok {program : ByteArray} {query : Lean.Json} {pub : ByteArray} {c : Flock.Circuit}
    (h : SampledProofs.FilePart.checkFile program query pub (.ok c) = .ok ()) :
    SampledProofs.FilePart.check program query pub = .ok () ∧
      SampledProofs.FilePart.classWhy (SampledProofs.FilePart.unitBits c) = none := by
  unfold SampledProofs.FilePart.checkFile SampledProofs.FilePart.checkClass at h
  cases hc : SampledProofs.FilePart.check program query pub with
  | error e => simp [hc, bind, Except.bind] at h
  | ok u =>
    cases hw : SampledProofs.FilePart.classWhy (SampledProofs.FilePart.unitBits c) with
    | some w => simp [hc, hw, bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at h
    | none => exact ⟨rfl, rfl⟩

theorem slotDesign_of_rule {g : FileReg} {r : OneStage} (h : g.rule = some r) :
    Properties.SampledProofs.slotDesign g.reg.rule g.reg.program g.reg.query g.reg.table = .ok (designOf r) := by
  unfold FileReg.rule at h
  split at h
  · rename_i hd
    cases h
    exact hd
  · cases h

theorem fileLayout_of_sizes {g : FileReg} {ns : Array ℕ} (h : g.sizes = some ns) :
    SampledProofs.Slots.fileLayout g.reg.circuit g.reg.program g.reg.query =
      .ok (SampledProofs.Slots.layout (SampledProofs.FilePart.ofCircuit g.reg.circuit ns.toList.sum) ns) := by
  unfold FileReg.sizes at h
  unfold SampledProofs.Slots.fileLayout
  cases hS : SampledProofs.Partition.strata g.reg.program g.reg.query with
  | error e => simp [hS] at h
  | ok S =>
    simp only [hS] at h
    have e : ns.foldl (· + ·) 0 = ns.toList.sum := by
      rw [← Array.foldl_toList, Properties.SampledProofs.SecurityProofs.foldl_add, zero_add]
    rw [Properties.SampledProofs.SecurityProofs.bind_eq_ok]
    refine ⟨S, rfl, ?_⟩
    simp only [h, e]
    try rfl

/-- **A statement the code parses**: the statement's public file and roots, the registration's strata and design, the
checks passed, and the statement `stmtOf`'s. -/
theorem parseOf_eq_some {g : FileReg} {x : Msg} {s : AuditStatement} (h : parseOf g x = some s) :
    ∃ pub roots ns r, g.read x = some (pub, roots) ∧ g.sizes = some ns ∧ g.rule = some r ∧
      FlockSoundness.Discharge.Layout.scopeOkH g.reg.circuit = true ∧
      SampledProofs.FilePart.checkFile ⟨g.desc.toArray⟩ g.reg.query pub (.ok g.reg.circuit) = .ok () ∧
      s = stmtOf (g.part ns.toList.sum) (g.consts ns.toList.sum) r g.desc g.query roots := by
  unfold parseOf at h
  cases hx : g.read x with
  | none => simp [hx] at h
  | some p =>
    obtain ⟨pub, roots⟩ := p
    cases hn : g.sizes with
    | none => simp [hx, hn] at h
    | some ns =>
      cases hr : g.rule with
      | none => simp [hx, hn, hr] at h
      | some r =>
        simp only [hx, hn, hr] at h
        split_ifs at h with hc
        exact ⟨pub, roots, ns, r, rfl, rfl, rfl, hc.1, hc.2, (Option.some.inj h).symm⟩

/-! ## The deployed partition is `partOf` -/

/-- **A circuit file's proof block as intervals is its block**: `Slots.blockOf`'s intervals expand to
`FilePart.block`. -/
theorem puGates_blockOf (f : SampledProofs.FilePart.FilePart) (u : ℕ) :
    Properties.SampledProofs.puGates (SampledProofs.Slots.blockOf f u) = (f.block u).toList := by
  have h3 : f.inGates + 3 * f.m * (u + 1) - (f.inGates + 3 * f.m * u) = 3 * f.m := by
    rw [Nat.mul_succ]
    omega
  unfold Properties.SampledProofs.puGates SampledProofs.Slots.blockOf SampledProofs.FilePart.FilePart.block
  simp [h3, List.range'_eq_map_range, List.flatMap_map]
  exact List.map_eq_flatMap.symm

/-- **The deployed layout's partition is `partOf`** at the key program of a circuit in scope, at the circuit's
constants and the strata `ns`, at their sum of units. -/
theorem slotPartition_layout_eq (c : Flock.Circuit) (hU : FlockSoundness.Built c.unit)
    (hscope : FlockSoundness.Discharge.Layout.scopeOkH c = true) (ns : Array ℕ) :
    Properties.SampledProofs.slotPartition
        (SampledProofs.Slots.layout (SampledProofs.FilePart.ofCircuit c ns.toList.sum) ns) =
      partOf ((FlockSoundness.Discharge.Placed.keyProg c hU ns.toList.sum).partition
          (FlockSoundness.Discharge.Layout.outsOf c hU ns.toList.sum))
        (fun w => constsOf c hU ns.toList.sum w) ns.toList := by
  have hn : (SampledProofs.FilePart.ofCircuit c ns.toList.sum).n = ns.toList.sum := rfl
  rw [partOf_ofCircuit c hU hscope, Properties.SampledProofs.SecurityProofs.slotPartition_layout, hn]
  unfold filePartOf
  simp only [puGates_blockOf]

theorem partitionOf_of_sizes {g : FileReg} {ns : Array ℕ} (hn : g.sizes = some ns) (s : AuditStatement) :
    partitionOf g s = some (Properties.SampledProofs.slotPartition
      (SampledProofs.Slots.layout (SampledProofs.FilePart.ofCircuit g.reg.circuit ns.toList.sum) ns)) := by
  unfold partitionOf
  simp only [fileLayout_of_sizes hn]

/-- **The deployed partition of a statement the code parses** is `partOf` of the key program at its strata. -/
theorem partitionOf_parse {g : FileReg} {ns : Array ℕ} (hn : g.sizes = some ns)
    (hscope : FlockSoundness.Discharge.Layout.scopeOkH g.reg.circuit = true) (s : AuditStatement) :
    partitionOf g s = some (partOf (g.part ns.toList.sum) (g.consts ns.toList.sum) ns.toList) :=
  (partitionOf_of_sizes hn s).trans (congrArg some (slotPartition_layout_eq g.reg.circuit g.built hscope ns))

/-! ## At the code -/

/-- **The deployment's statements are the registered circuit file's**: every statement `D` parses carries the
registration's descriptor bytes, and its program is the circuit file's key program at the registration's strata's sum
of units, lowered at the circuit's constants (`progOf`). In place of `StmtFlattens`, which read a statement's program
off its descriptor. -/
def StmtIsFile {Coin Col T : Type} (D : Claims.Sampling.Deploy Coin Col T) (g : FileReg) : Prop :=
  ∀ x s, D.parse x = some s → ∃ ns, g.sizes = some ns ∧ s.desc = g.desc ∧
    s.prog = progOf (g.circ ns.toList.sum) (g.consts ns.toList.sum)

variable {T : Type} (F : Claims.Flock.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) T) (g : FileReg)
  (cd : SamplingDeploy.Code)

/-- **`ProgramScope` at the code**: the scope check takes a statement exactly when it parses, its partition evaluates,
its program is topologically ordered and the partition partitions it. -/
theorem programScope_code (hcd : CodeIs g cd) : Claims.Sampling.ProgramScope (SamplingDeploy.deployOf F cd) := by
  obtain ⟨hp, hs, hP⟩ := hcd
  intro x
  show cd.scopeCheck x = true ↔
    ∃ s Pt, cd.parse x = some s ∧ cd.partition s = some Pt ∧ s.prog.Topological ∧ Pt.Partitions s.prog
  rw [hs, hp, hP]
  constructor
  · intro h
    obtain ⟨s, hx⟩ := Option.isSome_iff_exists.1 (show (parseOf g x).isSome = true from h)
    obtain ⟨_, _, ns, _, -, hn, -, hscope, hchk, rfl⟩ := parseOf_eq_some hx
    refine ⟨_, _, hx, partitionOf_parse hn hscope _, progOf_topological, ?_⟩
    exact keyPart_partitions g.reg.circuit g.built ns.toList.sum (unitLog_le_of_scopeOkH hscope) g.upv
      (le_of_classWhy (checkFile_ok hchk).2) rfl
  · rintro ⟨s, -, hx, -⟩
    exact Option.isSome_iff_exists.2 ⟨s, hx⟩

/-- **`RegMatches` at the code**, wherever the statement's registration is `g`'s. -/
theorem regMatches_code (hcd : CodeIs g cd) (reg : Msg → Option SamplingDeploy.Reg)
    (hreg : ∀ x, (g.read x).isSome → reg x = some g.reg) : SamplingDeploy.RegMatches cd reg := by
  obtain ⟨hp, -, hP⟩ := hcd
  intro x s Pt hx hPt
  rw [hp] at hx
  rw [hP] at hPt
  obtain ⟨_, _, ns, _, hread, hn, hr, -, -, rfl⟩ := parseOf_eq_some hx
  exact ⟨g.reg, _, hreg x (by simp [hread]), g.program, fileLayout_of_sizes hn,
    (Option.some.inj ((partitionOf_of_sizes hn _).symm.trans hPt)).symm, slotDesign_of_rule hr⟩

theorem stmtIsFile_code (hcd : CodeIs g cd) : StmtIsFile (SamplingDeploy.deployOf F cd) g := by
  intro x s hx
  rw [show (SamplingDeploy.deployOf F cd).parse = cd.parse from rfl, hcd.1] at hx
  obtain ⟨_, _, ns, _, -, hn, -, -, -, rfl⟩ := parseOf_eq_some hx
  exact ⟨ns, hn, rfl, rfl⟩

/-- **rel_deploy at the code**: `RelDeployTarget` with its partition hypothesis discharged. At bytes the code parses to
`stmtOf` of the key program at the registration's strata, a `witDep` outside `RelS` at SHA-512 and `commitDep` is
outside the record's `R`. -/
theorem relDeploy_code (hcd : CodeIs g cd) {ns : Array ℕ} (hn : g.sizes = some ns)
    (rows : Fin (g.circ ns.toList.sum).N → Option RowAt) (nl : ℕ) (r : OneStage) (roots : List Msg)
    (Xr : Fin (g.circ ns.toList.sum).N → Bool) (δ : ℝ) (x : Msg)
    (hL : LayoutIs (g.circ ns.toList.sum) rows g.reg.circuit nl roots cd)
    (hx : cd.parse x = some (stmtOf (g.part ns.toList.sum) (g.consts ns.toList.sum) r g.desc g.query roots))
    (hS : ¬ RelS sha512Msg (commitDep (g.circ ns.toList.sum) rows g.reg.circuit nl roots) δ
      (stmtOf (g.part ns.toList.sum) (g.consts ns.toList.sum) r g.desc g.query roots)
      (partOf (g.part ns.toList.sum) (g.consts ns.toList.sum) ns.toList)
      (witDep (g.circ ns.toList.sum) (commitDep (g.circ ns.toList.sum) rows g.reg.circuit nl roots) sha512Msg roots
        Xr)) :
    ¬ R sha512Msg (SamplingDeploy.deployOf F cd).commit cd.parse cd.partition δ x
      (witDep (g.circ ns.toList.sum) (commitDep (g.circ ns.toList.sum) rows g.reg.circuit nl roots) sha512Msg roots
        Xr) := by
  have hx' := hx
  rw [hcd.1] at hx'
  obtain ⟨_, _, _, _, -, -, -, hscope, -, -⟩ := parseOf_eq_some hx'
  exact relDeploy F cd (g.part ns.toList.sum) rows g.reg.circuit nl ns.toList r (g.consts ns.toList.sum) g.desc
    g.query roots Xr δ x hL hx (by rw [hcd.2.2]; exact partitionOf_parse hn hscope _) hS

/-- **rel_deploy at the code and a leaf map**: `relDeployL` with its partition hypothesis discharged, V5's layout. -/
theorem relDeployL_code (hcd : CodeIs g cd) {ns : Array ℕ} (hn : g.sizes = some ns)
    (rows : Fin (g.circ ns.toList.sum).N → Option RowAt) (L : LeafMap) (r : OneStage) (roots : List Msg)
    (Xr : Fin (g.circ ns.toList.sum).N → Bool) (δ : ℝ) (x : Msg)
    (hL : LayoutIsL (g.circ ns.toList.sum) rows g.reg.circuit L roots cd)
    (hx : cd.parse x = some (stmtOf (g.part ns.toList.sum) (g.consts ns.toList.sum) r g.desc g.query roots))
    (hS : ¬ RelS sha512Msg (commitDepL (g.circ ns.toList.sum) rows g.reg.circuit L roots) δ
      (stmtOf (g.part ns.toList.sum) (g.consts ns.toList.sum) r g.desc g.query roots)
      (partOf (g.part ns.toList.sum) (g.consts ns.toList.sum) ns.toList)
      (witDep (g.circ ns.toList.sum) (commitDepL (g.circ ns.toList.sum) rows g.reg.circuit L roots) sha512Msg roots
        Xr)) :
    ¬ R sha512Msg (SamplingDeploy.deployOf F cd).commit cd.parse cd.partition δ x
      (witDep (g.circ ns.toList.sum) (commitDepL (g.circ ns.toList.sum) rows g.reg.circuit L roots) sha512Msg roots
        Xr) := by
  have hx' := hx
  rw [hcd.1] at hx'
  obtain ⟨_, _, _, _, -, -, -, hscope, -, -⟩ := parseOf_eq_some hx'
  exact relDeployL F cd (g.part ns.toList.sum) rows g.reg.circuit L ns.toList r (g.consts ns.toList.sum) g.desc
    g.query roots Xr δ x hL hx (by rw [hcd.2.2]; exact partitionOf_parse hn hscope _) hS

end Flock.Bridge
