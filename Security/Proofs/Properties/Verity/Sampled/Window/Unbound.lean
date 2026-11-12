module

public import Proofs.Properties.Verity.Sampled.Window.Instance
public import Proofs.Properties.Verity.WindowFiles
public import Proofs.Flock.EndToEnd.Joint.V5.Layout

/-!
# The claim at a window: the units its trace doesn't bind are the window's incorrect proof blocks

The statement that carries the integrity claim across a window of several members: at the window's committed files
read as one circuit (`Verity.WindowFiles`), every unit the trace doesn't bind (`Accounting.unbound`) is an incorrect
proof block of the window (`Flock.Guarantees.JointAuditV5.wrongSet`), the set the window's bound is about. It is the
one-statement chain (`Verity.Sampled.Live.unbound_subset_faulty_part`, then the transport inside
`Flock.Bridge.assemblyTargetV5`) a member at a time, with the members side by side (`Verity.Partition.unbound_beside`).
This file holds the statements, each a `Prop`, and the names they read; it proves none of the three.

* **A member's read** (`MemberV5.fileRead`, `Cc`, `Lm`, `rows`, `roots`, `Cd`, `κ`): how v2 reads the member's
  committed files, and the deployed commitment at its public file's leaf map, as `Verity.Sampled.Live` names them at
  one statement and as `MemberV5.canon` reads them (`MemberV5.canon_eq`).
* **A member's trace** (`MemberV5.Xr`, `wit`, `transcript`, `JointAuditV5.memberX`): the trace its session fixes at
  registration, the one `MemberV5.wrongSet` reads (`MemberV5.wrongSet_eq`), as the deployed commitment's sampling
  witness and as a trace of the parsed circuit file.
* **The member's files as the verifier checks them** (`MemberV5.Checked`): no public input ports, the committed files
  parse to the member's descriptor, unit rows and population, the ports' published domains are pairwise distinct and
  the roots at the public file's leaf map fix their domains. `Flock.Bridge.v5Files_of_checks` reads the last two off
  `verify --zk`'s files check.
* **The member is read at the claim** (`MemberV5.ReadAt`): the files are the member's, the claim's openings hold
  wherever the member's deployed commitment opens and at the files' constants, and the trace is the member's.
* **One member** (`MemberV5.UnboundFaulty`, `MemberV5.FaultyWrong`): the units its trace doesn't bind are faulty in its
  sampling statement (`MemberV5.faulty`), and its faulty units are incorrect at every registration
  (`MemberV5.wrongAt`).
* **The window** (`ReadsWith`, `JointAuditV5.ReadsAt`, `JointAuditV5.UnboundWrong`): with the members laid out in unit
  order (`JointAuditV5.Laid`) and each member read at its own gates of the window's circuit, at its own part of the
  trace and of the openings, the units the window's trace doesn't bind are within the window's incorrect proof
  blocks; so they are at most as many (`JointAuditV5.UnboundWrong.card_le`), the inequality the claim's transfer at a
  window reads (`Verity.Sampled.WindowInstance.ClaimTransfers`).
-/

@[expose] public section

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open scoped ENNReal

namespace MemberV5

open Classical

variable {Reg : Type} {nJ : ℕ} (m : MemberV5 Reg nJ)

/-- **How v2 reads the member's committed files**: the verify command's tags, the circuit file, its tables and the
public file. -/
def fileRead : Verity.FileRead := ⟨zkTags m.I, m.I.circuitFile, m.I.tables, m.I.publicFile⟩

/-- The member's circuit: the key program's at the member's ports, with its outputs. -/
noncomputable abbrev Cc :=
  (keyProgP m.D.unroll m.pi m.hU m.n).circuit (FlockSoundness.Discharge.Layout.outsOfP m.D.unroll m.hU m.n m.pi)

/-- The member's public file's leaf map. -/
noncomputable def Lm := Flock.Bridge.LeafMap.ofPublic m.D.unroll m.pub m.n

/-- The member's deployed commitment's rows: each committed gate's port, leaf, width and place, read off the circuit
file. -/
noncomputable def rows : Fin m.Cc.N → Option Flock.Bridge.RowAt := fun g =>
  (FlockSoundness.Discharge.HiddenExec.rowOfHP m.D.unroll m.hU m.n m.pi
      (FlockSoundness.Discharge.Layout.outsOfP m.D.unroll m.hU m.n m.pi) g).map fun r =>
    ⟨r.2.1.val, m.Lm.ref r.2.1.val r.1.val, FlockSoundness.Discharge.Layout.portRowW m.D.unroll r.2.1, r.2.2.val⟩

/-- The member's deployed commitment's roots, one a port: its public file's. -/
noncomputable def roots := Flock.Bridge.rootsOfL m.D.unroll m.Lm

/-- **The member's deployed commitment** at its public file's leaf map. -/
noncomputable def Cd := Flock.Bridge.commitDepL m.Cc m.rows m.D.unroll m.Lm m.roots

/-- The member's circuit's constants, public. -/
noncomputable def κ : Fin m.Cc.N → Option Bool := fun g => Flock.Bridge.constsOfP m.D.unroll m.hU m.n m.pi m.pub g

/-- **The member's sampling statement**: its program at its constants and its roots. Neither the witness's values nor
the faulty units read the design, the descriptor's bytes or the query, so the statement holds none. -/
noncomputable def stmt : Claims.Sampling.AuditStatement :=
  Flock.Bridge.stmtOf m.P m.κ .all [] ⟨"", 0, []⟩ m.roots

/-- The member's proof blocks as the claims' partition, one stratum of its `n` units. -/
noncomputable def blocks : Claims.Sampling.Partition := Flock.Bridge.partOf m.P m.κ [m.n]

/-- **A trace of the member's circuit as the deployed commitment's sampling witness**: each wire's value in the
evaluation on the trace's inputs, opened at the wire's claims position under the member's roots. -/
noncomputable def wit (X : Fin m.Cc.N → Bool) : Claims.Sampling.Wit :=
  Flock.Bridge.witDep m.Cc m.Cd Claims.Sampling.sha512Msg m.roots X

/-- **The member's faulty units at a trace**: the units faulty in its sampling statement under the trace's witness. -/
noncomputable def faulty (X : Fin m.Cc.N → Bool) : Finset ℕ :=
  Claims.Sampling.faulty Claims.Sampling.sha512Msg m.Cd m.stmt m.blocks (m.wit X)

/-- **The member's transcript at a trace**: the trace's witness, opened at each gate's claims position under the
member's roots and read at the parsed circuit file's gates with the member's constants. -/
noncomputable def transcript (X : Fin m.Cc.N → Bool) : (Flock.Bridge.circuitOf m.D m.hU m.n).Trace :=
  Flock.Bridge.transcriptOf m.D m.hU m.n m.κ ((m.wit X).val Claims.Sampling.sha512Msg m.Cd m.stmt)

/-- **The member's incorrect proof blocks at a registration and a trace**: `wrongRegZc` at the member's canonical
openings. -/
noncomputable def wrongAt (R : Reg) (X : Fin m.Cc.N → Bool) : Finset (Fin m.n) :=
  wrongRegZc execArith m.P (planZHJP m.dj m.y₀) (tabZHJP m.dj m.y₀) m.vb m.canon R X

/-- **The trace the member's session fixes at registration** at `τ`, played alone at `L`: `XplurZC` at the simulator's
registration and continuation. -/
noncomputable def Xr (L : Law m.n) (hL : m.OneClass L) (τ : Strategy (m.game L)) : Fin m.Cc.N → Bool :=
  XplurZC execArith hs512 m.P (planZHJP m.dj m.y₀) (tabZHJP m.dj m.y₀) m.vb (reg (m.sim L hL τ))
    (cont (m.sim L hL τ)) m.k m.Rw

/-- **The member's incorrect proof blocks at `τ` are those at its registration and the trace fixed there.** -/
theorem wrongSet_eq (L : Law m.n) (hL : m.OneClass L) (τ : Strategy (m.game L)) :
    m.wrongSet L hL τ = m.wrongAt (reg (m.sim L hL τ)) (m.Xr L hL τ) :=
  rfl

/-- **The member's canonical openings are those at its rows and constants.** -/
theorem canon_eq :
    m.canon = fun (g : Fin m.Cc.N) (o : m.vb.Opening) =>
      Flock.Bridge.canonAt m.rows m.κ g (m.vb.bit (m.vb.val o) g) (m.vb.val o) o.2.2 :=
  rfl

/-- **The member's committed files, as the verifier checks them**: the member has no public input ports; its committed
files parse to its descriptor, unit rows and population; its ports' published domains are pairwise distinct; and its
roots at the public file's leaf map fix their domains. -/
structure Checked : Prop where
  noPublic : m.pi = ({} : HmIn.Ports)
  parsed : Flock.Bridge.parsedOf (zkTags m.I) m.I.circuitFile m.I.tables m.I.publicFile = some ⟨m.D, m.hU, m.n⟩
  distinct : Flock.PortRoots.dupId m.D.unroll m.pub = false
  fixes : Flock.Bridge.RootFixesDomainL m.D.unroll m.Lm

/-- **The openings `op` of the committed files `r` hold wherever the member's deployed commitment opens**: at each
gate `r` doesn't fix to a constant, a bit the deployed commitment opens at the gate's claims position under the
member's roots opens the gate to every value whose first bit it is. -/
def OpensUnder (r : Verity.FileRead) (op : (g : Fin r.circuit.N) → BitVec (r.circuit.w g) → Prop) : Prop :=
  ∀ (h : r.circuit.N = m.Cc.N) (g : Fin r.circuit.N) (b : Bool) (d : Claims.Msg), r.consts g = none →
    m.Cd.Opens Claims.Sampling.sha512Msg m.roots ⟨Flock.Bridge.posOf m.Cc (Fin.cast h g), b, d⟩ = true →
      ∀ v : BitVec (r.circuit.w g), v.getLsbD 0 = b → op g v

/-- **The member is read at the claim**, at the committed files `r`, the openings `op`, the trace `tr` of the files'
circuit and the trace `X` of the member's: the member's files pass the verifier's checks, `r` is how v2 reads them,
`op` holds wherever the member's deployed commitment opens and at the files' constants, and `tr` is the member's
transcript at `X`. -/
structure ReadAt (r : Verity.FileRead) (op : (g : Fin r.circuit.N) → BitVec (r.circuit.w g) → Prop)
    (tr : r.circuit.Trace) (X : Fin m.Cc.N → Bool) : Prop where
  checked : m.Checked
  read : r = m.fileRead
  opens : m.OpensUnder r op
  consts : Verity.Sampled.ConstsOpen r op
  trace : HEq tr (m.transcript X)

/-- **The units a member's transcript doesn't bind are faulty**: wherever the member is read at the claim, every
unit of its committed files unbound in the trace is faulty in the member's sampling statement at the member's
trace. -/
def UnboundFaulty : Prop :=
  ∀ (r : Verity.FileRead) (op : (g : Fin r.circuit.N) → BitVec (r.circuit.w g) → Prop) (tr : r.circuit.Trace)
    (X : Fin m.Cc.N → Bool), m.ReadAt r op tr X → (Accounting.unbound r.parts op tr).image Fin.val ⊆ m.faulty X

/-- **A member's faulty units are incorrect proof blocks**: at files that pass the verifier's checks, a unit faulty in
the member's sampling statement at a trace is incorrect at that trace and every registration. -/
def FaultyWrong : Prop :=
  m.Checked → ∀ (R : Reg) (X : Fin m.Cc.N → Bool) (u : Fin m.n), u.val ∈ m.faulty X → u ∈ m.wrongAt R X

end MemberV5

/-- **A member at the trace it is read at.** -/
structure Seat (Reg : Type) (nJ : ℕ) where
  m : MemberV5 Reg nJ
  X : Fin m.Cc.N → Bool

/-- **A first member's files and the rest's are read at the claim by the members `ss`, in order**: the first of `ss`
is read at the first files, at its own gates of the circuits side by side, its part of the trace and the openings at
its gates, and the rest of `ss` at the rest's; `ss` has as many members as there are files. -/
def ReadsWith {Reg : Type} {nJ : ℕ} : (r : Verity.FileRead) → (rs : List Verity.FileRead) →
    ((g : Fin (Verity.FileRead.circuitWith r rs).N) → BitVec ((Verity.FileRead.circuitWith r rs).w g) → Prop) →
    (Verity.FileRead.circuitWith r rs).Trace → List (Seat Reg nJ) → Prop
  | r, [], op, X, ss => ∃ s, ss = [s] ∧ s.m.ReadAt r op X s.X
  | r, r' :: rs, op, X, ss => ∃ s ss', ss = s :: ss' ∧
      s.m.ReadAt r
        (fun i v => op (Fin.castAdd (Verity.FileRead.circuitWith r' rs).N i)
          (v.cast (Verity.Circuit.besideW_left r.circuit (Verity.FileRead.circuitWith r' rs) i).symm))
        (Verity.Circuit.traceLeft X) s.X ∧
      ReadsWith r' rs
        (fun j v => op (Fin.natAdd r.circuit.N j)
          (v.cast (Verity.Circuit.besideW_right r.circuit (Verity.FileRead.circuitWith r' rs) j).symm))
        (Verity.Circuit.traceRight X) ss'

namespace JointAuditV5

open Classical

variable (w : JointAuditV5)

/-- **The trace member `k`'s session fixes at the joint registration** at `τ`, the one `memberWrong` reads. -/
noncomputable def memberX (τ : Strategy w.game) (k : Fin w.ms.length) : Fin (w.ms.get k).Cc.N → Bool :=
  (w.ms.get k).Xr (w.memberLaw τ k) (w.oneClass_memberLaw τ k) (w.memberStrat τ k)

/-- **Member `k`'s incorrect proof blocks at `τ` are those at the joint registration and the trace fixed there.** -/
theorem memberWrong_eq (τ : Strategy w.game) (k : Fin w.ms.length) :
    w.memberWrong τ k = (w.ms.get k).wrongAt (reg (w.memberSim τ k)) (w.memberX τ k) :=
  rfl

/-- The window's members, each at the trace its session fixes at the joint registration at `τ`, in the members'
order. -/
noncomputable def seats (τ : Strategy w.game) : List (Seat w.Reg w.nJ) :=
  List.ofFn fun k : Fin w.ms.length => (⟨w.ms.get k, w.memberX τ k⟩ : Seat w.Reg w.nJ)

/-- **The window's committed files are read at the claim** at `τ`: `R`'s members are the window's, in order, each
read (`MemberV5.ReadAt`) at its own gates of the window's circuit, at its part of the trace `X` and of the openings
`op`, and at the trace its session fixes at the joint registration. -/
def ReadsAt (R : Verity.WindowFiles) (op : (g : Fin R.circuit.N) → BitVec (R.circuit.w g) → Prop)
    (X : R.circuit.Trace) (τ : Strategy w.game) : Prop :=
  ReadsWith R.first R.rest op X (w.seats τ)

/-- **The units a window's trace doesn't bind are its incorrect proof blocks.** With the members laid out in unit
order, wherever the window's committed files are read at the claim at `τ`, every unit of the window's files unbound in
the trace `X` under the openings `op` is, at the same place of the population, an incorrect proof block of the window
at `τ`. -/
def UnboundWrong : Prop :=
  w.Laid → ∀ (R : Verity.WindowFiles) (op : (g : Fin R.circuit.N) → BitVec (R.circuit.w g) → Prop)
    (X : R.circuit.Trace) (τ : Strategy w.game), w.ReadsAt R op X τ →
      (Accounting.unbound R.parts op X).image Fin.val ⊆ (w.wrongSet τ).image Fin.val

variable {w}

/-- **The unbound units are at most as many as the window's incorrect proof blocks**, the inequality the claim's
transfer reads. -/
theorem UnboundWrong.card_le (h : w.UnboundWrong) (hl : w.Laid) (R : Verity.WindowFiles)
    (op : (g : Fin R.circuit.N) → BitVec (R.circuit.w g) → Prop) (X : R.circuit.Trace) (τ : Strategy w.game)
    (hr : w.ReadsAt R op X τ) : (Accounting.unbound R.parts op X).card ≤ (w.wrongSet τ).card := by
  have hs := Finset.card_le_card (h hl R op X τ hr)
  rwa [Finset.card_image_of_injective _ Fin.val_injective, Finset.card_image_of_injective _ Fin.val_injective] at hs

end JointAuditV5

end Flock.Guarantees

namespace Verity.Sampled

/-- **The claim at a window** at every budget of SHA-512 calls: the units the window's trace doesn't bind are the
incorrect proof blocks of Flock's window (`Window.joint`), the set the window's bound and the claim's transfer
(`WindowInstance.ClaimTransfers`) read. -/
def Window.UnboundWrong (W : Window) : Prop := ∀ t : ℕ, (W.joint t).UnboundWrong

end Verity.Sampled
