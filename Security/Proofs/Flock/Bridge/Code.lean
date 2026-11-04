import Proofs.Flock.Bridge.FilePart
import Proofs.Flock.Bridge.PartitionsKey
import Proofs.Flock.Claims.Sampling.Reference.DeployedCode
import Proofs.Flock.Claims.Sampling.Select.Layout
import Proofs.Flock.Claims.Sampling.Select.Scope
import Proofs.Flock.Claims.Sampling.Select.Steps

/-!
# The bridge at the deployed code: one registration's parser, partition and scope check

The sampled verifier's code (`SamplingDeploy.Code`) reads a statement's partition off the statement alone, so the
bridge's code is per registration (`RegStmt`): the draw's registration (`SamplingDeploy.Reg`, with its circuit file),
the statement's bytes, and what that statement names: its Program descriptor, its query, its one-stage design and its
trace commitment's roots. The roots are fixed with the registration because the code's registered ports carry them
(`LayoutIs`: `own = portsOf c nl roots`), so the code parses one statement, the registration's.

* `RegStmt.Guard`: what the code checks of the registration: C-Flock's scope check on the circuit file (`scopeOkH`), a
  template instance (`1 ≤ upv`), the registration's layout (`Slots.fileLayout`), its descriptor parsing to its program,
  its design (`slotDesign`) being the statement's, and every proof block of the key program committing at most
  `unitClassMax` bits.
* `parseOf`, `partitionOf`, `scopeOf`: the statement's bytes parse to `stmtOf` at the circuit file's key program and its
  constants where the guard holds, the partition is `partOf` in the registration's strata, and the scope check is the
  parse's success. `CodeIs` holds a code to them and to `commitDep`'s layout (`LayoutIs`), and `SamplingDeploy.code`
  keeps it (`codeIs_code`).

At a code `CodeIs` holds of:

* `programScope_bridge`: `ProgramScope`, from the guard, `progOf_topological` and `partOf_partitions_key`.
* `regMatches_bridge`: `RegMatches`, the registration's layout being `partOf` (`slotPartition_fileLayout`: the claims
  read `Slots.layout` as `slotPartition_layout` does, its intervals expanded (`puGates_blockOf`), and the summary's
  blocks are `partOf`'s, `partOf_ofCircuit`).
* `scopeMatches_bridge`: so `ScopeMatches` at the deployed record (`scopeMatches_of`, `drawSlots_code`).
* `stmtIsFile_bridge`: `StmtIsFile`: every statement the code parses has the circuit file's key program lowered at its
  constants as its gates, and the registration's descriptor.
* `relDeploy_bridge`: `relDeployTarget` at the registration's statement and partition.
-/

namespace Flock.Bridge

open FlockSoundness FlockSoundness.Discharge.Layout FlockSoundness.Discharge.Placed

/-- **A registration and the statement it names**: the draw's registration (`SamplingDeploy.Reg`: the selection rule,
the verifier's program, query and work table, the closure map and the circuit file), the statement's bytes `x`, and
what the statement names: its Program descriptor's bytes, its query, its one-stage design and its trace commitment's
roots. -/
structure RegStmt where
  reg : SamplingDeploy.Reg
  x : Claims.Msg
  desc : Claims.Msg
  q : Claims.Sampling.Query
  r : Claims.Sampling.OneStage
  roots : List Claims.Msg

namespace RegStmt

variable (a : RegStmt)

/-- The registration's strata's sizes (`FilePart.sizes` of `Partition.strata`), empty where its query doesn't apply or
its strata are not its units in order. -/
def ns : Array ℕ :=
  match SampledProofs.Partition.strata a.reg.program a.reg.query with
  | .ok S => (SampledProofs.FilePart.sizes S).getD #[]
  | .error _ => #[]

/-- The registration's population: its strata's sizes summed. -/
def n : ℕ := a.ns.toList.sum

/-- The circuit file's key program's circuit at the registration's population. -/
noncomputable abbrev keyC (hU : Built a.reg.circuit.unit) :=
  (keyProg a.reg.circuit hU a.n).circuit (outsOf a.reg.circuit hU a.n)

/-- The key program's partition into its units. -/
noncomputable abbrev keyP (hU : Built a.reg.circuit.unit) :=
  (keyProg a.reg.circuit hU a.n).partition (outsOf a.reg.circuit hU a.n)

/-- The circuit's constants on the key program's gates. -/
noncomputable abbrev keyκ (hU : Built a.reg.circuit.unit) : Fin (a.keyC hU).N → Option Bool :=
  fun g => constsOf a.reg.circuit hU a.n g

/-- **The registration's statement**: `stmtOf` at the key program and its constants. -/
noncomputable def stmt (hU : Built a.reg.circuit.unit) : Claims.Sampling.AuditStatement :=
  stmtOf (a.keyP hU) (a.keyκ hU) a.r a.desc a.q a.roots

/-- **The registration's partition**: `partOf` at the key program and its constants, in the registration's strata. -/
noncomputable def part (hU : Built a.reg.circuit.unit) : Claims.Sampling.Partition :=
  partOf (a.keyP hU) (a.keyκ hU) a.ns.toList

/-- **What the bridge's code checks of the registration** (module docstring). -/
def Guard (hU : Built a.reg.circuit.unit) : Prop :=
  scopeOkH a.reg.circuit = true ∧ 1 ≤ a.reg.circuit.upv ∧
    (∃ l, SampledProofs.Slots.fileLayout a.reg.circuit a.reg.program a.reg.query = .ok l) ∧
    SampledProofs.Program.parse ⟨a.desc.toArray⟩ = .ok a.reg.program ∧
    Properties.SampledProofs.slotDesign a.reg.rule a.reg.program a.reg.query a.reg.table = .ok (designOf a.r) ∧
    ∀ u, Claims.Sampling.committed (progOf (a.keyC hU) (a.keyκ hU)) (blockOf (a.keyP hU) (a.keyκ hU) u) ≤
      Claims.Sampling.unitClassMax

end RegStmt

open Classical in
/-- **The bridge's parser**: the registration's statement bytes to its statement (`RegStmt.stmt`) where the guard
holds, and nothing else. -/
noncomputable def parseOf (a : RegStmt) (x : Claims.Msg) : Option Claims.Sampling.AuditStatement :=
  if hU : Built a.reg.circuit.unit then if x = a.x ∧ a.Guard hU then some (a.stmt hU) else none else none

open Classical in
/-- **The bridge's partition**: the registration's (`RegStmt.part`), whatever the statement. -/
noncomputable def partitionOf (a : RegStmt) (_s : Claims.Sampling.AuditStatement) :
    Option Claims.Sampling.Partition :=
  if hU : Built a.reg.circuit.unit then some (a.part hU) else none

/-- **The bridge's scope check**: the statement parses. -/
noncomputable def scopeOf (a : RegStmt) (x : Claims.Msg) : Bool := (parseOf a x).isSome

/-- **A code is the bridge's at the registration**: it parses, partitions and checks scope as `parseOf`, `partitionOf`
and `scopeOf` do, and its layout is `commitDep`'s at the key program's rows `rows`, `nl` leaves a port and the
registration's roots. -/
def CodeIs (a : RegStmt) (hU : Built a.reg.circuit.unit) (rows : Fin (a.keyC hU).N → Option RowAt) (nl : ℕ)
    (cd : SamplingDeploy.Code) : Prop :=
  cd.parse = parseOf a ∧ cd.partition = partitionOf a ∧ cd.scopeCheck = scopeOf a ∧
    LayoutIs (a.keyC hU) rows a.reg.circuit nl a.roots cd

/-- **The deployment's statements are the registration's circuit file**: every statement `D` parses has the circuit
file's key program, lowered at its constants (`progOf`), as its gates, and the registration's descriptor. -/
def StmtIsFile {Coin Col T : Type} (D : Claims.Sampling.Deploy Coin Col T) (a : RegStmt) : Prop :=
  ∀ x s, D.parse x = some s →
    ∃ hU : Built a.reg.circuit.unit, s.prog = progOf (a.keyC hU) (a.keyκ hU) ∧ s.desc = a.desc

/-- **A layout's proof block, as the claims read it, is the summary's block** (`FilePart.block`): its intervals
expanded. -/
theorem puGates_blockOf (f : SampledProofs.FilePart.FilePart) (u : ℕ) :
    Properties.SampledProofs.puGates (SampledProofs.Slots.blockOf f u) = (f.block u).toList := by
  have h3 : f.inGates + 3 * f.m * (u + 1) - (f.inGates + 3 * f.m * u) = 3 * f.m := by rw [Nat.mul_succ]; omega
  simp only [Properties.SampledProofs.puGates, SampledProofs.Slots.blockOf, SampledProofs.FilePart.FilePart.block,
    Array.toList_append, Array.toList_map, List.flatMap_append, Array.toList_range, List.flatMap_map]
  simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil, h3, List.range'_eq_map_range]
  congr 1
  rw [List.map_eq_flatMap]
  congr 1
  funext b
  simp

section Code

variable {a : RegStmt}

/-- **What the bridge parses**: the registration's statement, at the registration's bytes, where the guard holds. -/
theorem parseOf_eq_some {x : Claims.Msg} {s : Claims.Sampling.AuditStatement} (h : parseOf a x = some s) :
    ∃ hU : Built a.reg.circuit.unit, x = a.x ∧ a.Guard hU ∧ s = a.stmt hU := by
  unfold parseOf at h
  split at h
  · rename_i hU
    split at h
    · rename_i hg
      exact ⟨hU, hg.1, hg.2, (Option.some.inj h).symm⟩
    · cases h
  · cases h

theorem parseOf_self (hU : Built a.reg.circuit.unit) (hg : a.Guard hU) : parseOf a a.x = some (a.stmt hU) := by
  unfold parseOf
  rw [dite_eq_left hU, ite_eq_left (show a.x = a.x ∧ a.Guard hU from ⟨rfl, hg⟩)]

theorem partitionOf_eq (hU : Built a.reg.circuit.unit) (s : Claims.Sampling.AuditStatement) :
    partitionOf a s = some (a.part hU) := by
  unfold partitionOf
  rw [dite_eq_left hU]

/-- **The registration's layout is `partOf`**: at a circuit file in scope, the claims' partition of
`Slots.fileLayout`'s layout is the key program's proof blocks at its constants, in the registration's strata. -/
theorem slotPartition_fileLayout (hU : Built a.reg.circuit.unit) (hsc : scopeOkH a.reg.circuit = true)
    {l : SampledProofs.Slots.Layout}
    (hl : SampledProofs.Slots.fileLayout a.reg.circuit a.reg.program a.reg.query = .ok l) :
    Properties.SampledProofs.slotPartition l = a.part hU := by
  obtain ⟨S, ns, hS, hns, rfl⟩ := Properties.SampledProofs.SecurityProofs.fileLayout_ok _ _ _ l hl
  have ha : a.ns = ns := by simp only [RegStmt.ns, hS, hns, Option.getD_some]
  have hn : a.n = ns.toList.sum := congrArg (fun v : Array ℕ => v.toList.sum) ha
  have hp : a.part hU = filePartOf a.reg.circuit a.n a.ns.toList :=
    partOf_ofCircuit a.reg.circuit hU hsc a.n a.ns.toList
  have hfn : (SampledProofs.FilePart.ofCircuit a.reg.circuit ns.toList.sum).n = ns.toList.sum := rfl
  rw [Properties.SampledProofs.SecurityProofs.slotPartition_layout, hp, hfn]
  unfold filePartOf
  simp only [puGates_blockOf]
  rw [ha, hn]

variable {hU : Built a.reg.circuit.unit} {rows : Fin (a.keyC hU).N → Option RowAt} {nl : ℕ}
  {cd : SamplingDeploy.Code}

/-- **`SamplingDeploy.code` keeps `CodeIs`**: it sets only the draw and the session. -/
theorem codeIs_code (h : CodeIs a hU rows nl cd) (reg : Claims.Msg → Option SamplingDeploy.Reg) (L : Claims.Msg → ℕ)
    (bytes : (x : Claims.Msg) → (Fin (cd.V.k x) → FlockDeploy.Coin) → Fin (L x) → Fin 256)
    (sr : (x : Claims.Msg) → Finset (Fin (cd.V.k x)))
    (sess : Claims.Msg → Claims.Seen → Option (Claims.Msg × Claims.Seen)) :
    CodeIs a hU rows nl (SamplingDeploy.code cd reg L bytes sr sess) :=
  h

/-- **`ProgramScope` at the bridge's code**: its scope check takes exactly the registration's statement bytes where
the guard holds, and there the statement's program is topologically ordered (`progOf_topological`) and its partition
partitions it (`partOf_partitions_key`, from the guard's template instance and class bound). -/
theorem programScope_bridge {T : Type} (F : Claims.Flock.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) T)
    (hcd : CodeIs a hU rows nl cd) : Claims.Sampling.ProgramScope (SamplingDeploy.deployOf F cd) := by
  obtain ⟨hp, hP, hs, -⟩ := hcd
  intro x
  show cd.scopeCheck x = true ↔ ∃ s Pt, cd.parse x = some s ∧ cd.partition s = some Pt ∧ s.prog.Topological ∧
    Pt.Partitions s.prog
  rw [hs, hp, hP]
  constructor
  · intro h
    obtain ⟨s, hx⟩ := Option.isSome_iff_exists.1 h
    obtain ⟨hU', -, ⟨-, hupv, -, -, -, hcls⟩, rfl⟩ := parseOf_eq_some hx
    refine ⟨_, _, hx, partitionOf_eq hU' _, progOf_topological (a.keyC hU') (a.keyκ hU'), ?_⟩
    show (partOf (a.keyP hU') (a.keyκ hU') a.ns.toList).Partitions (progOf (a.keyC hU') (a.keyκ hU'))
    exact partOf_partitions_key a.reg.circuit hU' a.n hupv rfl hcls
  · rintro ⟨s, Pt, hx, -⟩
    show (parseOf a x).isSome = true
    simp only [hx, Option.isSome_some]

/-- **`RegMatches` at the bridge's code**, every statement naming the registration: the statement's descriptor parses
to the registration's program, its partition is the claims' reading of the registration's layout
(`slotPartition_fileLayout`) and its design is the registration's. -/
theorem regMatches_bridge (hcd : CodeIs a hU rows nl cd) : SamplingDeploy.RegMatches cd fun _ => some a.reg := by
  obtain ⟨hp, hP, -, -⟩ := hcd
  intro x s Pt hx hPt
  rw [hp] at hx
  rw [hP] at hPt
  obtain ⟨hU', -, ⟨hsc, -, ⟨l, hl⟩, hd, hds, -⟩, rfl⟩ := parseOf_eq_some hx
  rw [partitionOf_eq hU'] at hPt
  obtain rfl := Option.some.inj hPt
  exact ⟨a.reg, l, rfl, hd, hl, (slotPartition_fileLayout hU' hsc hl).symm, hds⟩

/-- **`ScopeMatches` at the deployed record of the bridge's code**: the draw is the registration's (`drawSlots_code`,
from `regMatches_bridge` and uniform bytes) and the scope check is the program part (`programScope_bridge`). -/
theorem scopeMatches_bridge (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (hcd : CodeIs a hU rows nl cd) (L : Claims.Msg → ℕ)
    (bytes : (x : Claims.Msg) → (Fin (cd.V.k x) → FlockDeploy.Coin) → Fin (L x) → Fin 256)
    (sr : (x : Claims.Msg) → Finset (Fin (cd.V.k x)))
    (sess : Claims.Msg → Claims.Seen → Option (Claims.Msg × Claims.Seen))
    (hunif : SamplingDeploy.BytesUniform cd L bytes) :
    Claims.Sampling.ScopeMatches
      (SamplingDeploy.deploy rel (SamplingDeploy.code cd (fun _ => some a.reg) L bytes sr sess)) :=
  Claims.Sampling.scopeMatches_of _
    (SamplingDeploy.drawSlots_code rel cd _ L bytes sr sess (regMatches_bridge hcd) hunif)
    (programScope_bridge _ (codeIs_code hcd _ L bytes sr sess))

/-- **`StmtIsFile` at the bridge's code**: what it parses is the registration's statement. -/
theorem stmtIsFile_bridge {T : Type} (F : Claims.Flock.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) T)
    (hcd : CodeIs a hU rows nl cd) : StmtIsFile (SamplingDeploy.deployOf F cd) a := by
  intro x s hx
  have hx' : parseOf a x = some s := by
    rw [← hcd.1]
    exact hx
  obtain ⟨hU', -, -, rfl⟩ := parseOf_eq_some hx'
  exact ⟨hU', rfl, rfl⟩

/-- **rel_deploy at the bridge's code**: where the guard holds, a `witDep` outside `RelS` at the registration's
statement and partition is outside the deployed record's `R` at the registration's statement bytes. -/
theorem relDeploy_bridge {T : Type} (F : Claims.Flock.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) T)
    (hcd : CodeIs a hU rows nl cd) (hg : a.Guard hU) (Xr : Fin (a.keyC hU).N → Bool) (δ : ℝ) :
    ¬ Claims.Sampling.RelS Claims.Sampling.sha512Msg (commitDep (a.keyC hU) rows a.reg.circuit nl a.roots) δ
        (a.stmt hU) (a.part hU)
        (witDep (a.keyC hU) (commitDep (a.keyC hU) rows a.reg.circuit nl a.roots) Claims.Sampling.sha512Msg a.roots
          Xr) →
      ¬ Claims.Sampling.R Claims.Sampling.sha512Msg (SamplingDeploy.deployOf F cd).commit cd.parse cd.partition δ a.x
        (witDep (a.keyC hU) (commitDep (a.keyC hU) rows a.reg.circuit nl a.roots) Claims.Sampling.sha512Msg a.roots
          Xr) :=
  relDeployTarget F cd (a.keyP hU) rows a.reg.circuit nl a.ns.toList a.r (a.keyκ hU) a.desc a.q a.roots Xr δ a.x
    hcd.2.2.2 (by rw [hcd.1]; exact parseOf_self hU hg) (by rw [hcd.2.1]; exact partitionOf_eq hU _)

end Code

end Flock.Bridge
