module

public import Proofs.Properties.Verity.Sampled.Window.Read
public import Proofs.Properties.Verity.Sampled.Instance
public import Proofs.Properties.Verity.ExecutionPhaseFiles
public import Proofs.Flock.EndToEnd.Joint.V5.Layout

/-! The claim at a window: the units its trace doesn't bind are the window's incorrect proof blocks. -/

@[expose] public section

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open scoped ENNReal

namespace MemberV5

open Classical

variable {Reg : Type} {nJ : ℕ} (m : MemberV5 Reg nJ)

def fileRead : Verity.FileRead := ⟨zkTags m.I, m.I.circuitFile, m.I.tables, m.I.publicFile⟩

noncomputable abbrev Cc :=
  (keyProgP m.D.unroll m.pi m.hU m.n).circuit (FlockSoundness.Discharge.Layout.outsOfP m.D.unroll m.hU m.n m.pi)

noncomputable def Lm := Flock.Bridge.LeafMap.ofPublic m.D.unroll m.pub m.n

noncomputable def rows : Fin m.Cc.N → Option Flock.Bridge.RowAt := fun g =>
  (FlockSoundness.Discharge.HiddenExec.rowOfHP m.D.unroll m.hU m.n m.pi
      (FlockSoundness.Discharge.Layout.outsOfP m.D.unroll m.hU m.n m.pi) g).map fun r =>
    ⟨r.2.1.val, m.Lm.ref r.2.1.val r.1.val, FlockSoundness.Discharge.Layout.portRowW m.D.unroll r.2.1, r.2.2.val⟩

noncomputable def roots := Flock.Bridge.rootsOfL m.D.unroll m.Lm

noncomputable def Cd := Flock.Bridge.commitDepL m.Cc m.rows m.D.unroll m.Lm m.roots

noncomputable def κ : Fin m.Cc.N → Option Bool := fun g => Flock.Bridge.constsOfP m.D.unroll m.hU m.n m.pi m.pub g

/-- **The member's sampling statement**: its program at its constants and its roots. Neither the witness's values nor
the incorrect proof blocks read the design, the descriptor's bytes or the query, so the statement holds none. -/
noncomputable def stmt : Claims.Sampling.AuditStatement :=
  Flock.Bridge.stmtOf m.P m.κ .all [] ⟨"", 0, []⟩ m.roots

noncomputable def blocks : Claims.Sampling.Partition := Flock.Bridge.partOf m.P m.κ [m.n]

noncomputable def wit (Xv : Fin m.Cc.N → Bool) : Claims.Sampling.Wit :=
  Flock.Bridge.witDep m.Cc m.Cd Claims.Sampling.sha512Msg m.roots Xv

noncomputable def faulty (Xv : Fin m.Cc.N → Bool) : Finset ℕ :=
  Claims.Sampling.faulty Claims.Sampling.sha512Msg m.Cd m.stmt m.blocks (m.wit Xv)

noncomputable def transcript (Xv : Fin m.Cc.N → Bool) : (Flock.Bridge.circuitOf m.D m.hU m.n).Trace :=
  Flock.Bridge.traceOf m.D m.hU m.n m.κ ((m.wit Xv).val Claims.Sampling.sha512Msg m.Cd m.stmt)

noncomputable def incorrectAt (R : Reg) (Xv : Fin m.Cc.N → Bool) : Finset (Fin m.n) :=
  incorrectRegZc execArith m.P (planZHJP m.dj m.y₀) (tabZHJP m.dj m.y₀) m.vb m.canon R Xv

noncomputable def Xr (L : SamplingDesign m.n) (hL : m.OneClass L) (τ : Strategy (m.game L)) : Fin m.Cc.N → Bool :=
  XplurZC execArith hs512 m.P (planZHJP m.dj m.y₀) (tabZHJP m.dj m.y₀) m.vb (reg (m.sim L hL τ))
    (cont (m.sim L hL τ)) m.k m.Rw

theorem incorrectSet_eq (L : SamplingDesign m.n) (hL : m.OneClass L) (τ : Strategy (m.game L)) :
    m.incorrectSet L hL τ = m.incorrectAt (reg (m.sim L hL τ)) (m.Xr L hL τ) :=
  rfl

theorem canon_eq :
    m.canon = fun (g : Fin m.Cc.N) (o : m.vb.Opening) =>
      Flock.Bridge.canonAt m.rows m.κ g (m.vb.bit (m.vb.val o) g) (m.vb.val o) o.2.2 :=
  rfl

structure Checked : Prop where
  noPublic : m.pi = ({} : HmIn.Ports)
  parsed : Flock.Bridge.parsedOf (zkTags m.I) m.I.circuitFile m.I.tables m.I.publicFile = some ⟨m.D, m.hU, m.n⟩
  distinct : Flock.PortRoots.dupId m.D.unroll m.pub = false
  fixes : Flock.Bridge.RootFixesDomainL m.D.unroll m.Lm

def OpensUnder (r : Verity.FileRead) (op : (g : Fin r.circuit.N) → BitVec (r.circuit.w g) → Prop) : Prop :=
  ∀ (h : r.circuit.N = m.Cc.N) (g : Fin r.circuit.N) (b : Bool) (d : Claims.Msg), r.consts g = none →
    m.Cd.Opens Claims.Sampling.sha512Msg m.roots ⟨Flock.Bridge.posOf m.Cc (Fin.cast h g), b, d⟩ = true →
      ∀ v : BitVec (r.circuit.w g), v.getLsbD 0 = b → op g v

structure ReadAt (r : Verity.FileRead) (op : (g : Fin r.circuit.N) → BitVec (r.circuit.w g) → Prop)
    (tr : r.circuit.Trace) (Xv : Fin m.Cc.N → Bool) : Prop where
  checked : m.Checked
  read : r = m.fileRead
  opens : m.OpensUnder r op
  consts : Verity.Sampled.ConstsOpen (.one r) op
  trace : HEq tr (m.transcript Xv)

def UnboundFaulty : Prop :=
  ∀ (r : Verity.FileRead) (op : (g : Fin r.circuit.N) → BitVec (r.circuit.w g) → Prop) (tr : r.circuit.Trace)
    (Xv : Fin m.Cc.N → Bool), m.ReadAt r op tr Xv → (ResourceAccounting.unbound r.parts op tr).image Fin.val ⊆ m.faulty Xv

def FaultyIncorrect : Prop :=
  m.Checked → ∀ (R : Reg) (Xv : Fin m.Cc.N → Bool) (u : Fin m.n), u.val ∈ m.faulty Xv → u ∈ m.incorrectAt R Xv

end MemberV5

structure Seat (Reg : Type) (nJ : ℕ) where
  m : MemberV5 Reg nJ
  Xv : Fin m.Cc.N → Bool

def ReadsWith {Reg : Type} {nJ : ℕ} : (r : Verity.FileRead) → (rs : List Verity.FileRead) →
    ((g : Fin (Verity.FileRead.circuitWith r rs).N) → BitVec ((Verity.FileRead.circuitWith r rs).w g) → Prop) →
    (Verity.FileRead.circuitWith r rs).Trace → List (Seat Reg nJ) → Prop
  | r, [], op, tr, ss => ∃ s, ss = [s] ∧ s.m.ReadAt r op tr s.Xv
  | r, r' :: rs, op, tr, ss => ∃ s ss', ss = s :: ss' ∧
      s.m.ReadAt r
        (fun i v => op (Fin.castAdd (Verity.FileRead.circuitWith r' rs).N i)
          (v.cast (Fin.append_left r.circuit.w (Verity.FileRead.circuitWith r' rs).w i).symm))
        (Verity.Circuit.traceLeft tr) s.Xv ∧
      ReadsWith r' rs
        (fun j v => op (Fin.natAdd r.circuit.N j)
          (v.cast (Fin.append_right r.circuit.w (Verity.FileRead.circuitWith r' rs).w j).symm))
        (Verity.Circuit.traceRight tr) ss'

open Classical in
noncomputable def Seat.trace {Reg : Type} {nJ : ℕ} (s : Seat Reg nJ) (r : Verity.FileRead) : r.circuit.Trace :=
  if h : r.circuit = Flock.Bridge.circuitOf s.m.D s.m.hU s.m.n then
    cast (congrArg Verity.Circuit.Trace h.symm) (s.m.transcript s.Xv)
  else fun _ => 0

noncomputable def tracesWith {Reg : Type} {nJ : ℕ} : (r : Verity.FileRead) → (rs : List Verity.FileRead) →
    List (Seat Reg nJ) → (Verity.FileRead.circuitWith r rs).Trace
  | r, [], ss => (ss.head?.map (·.trace r)).getD fun _ => 0
  | r, r' :: rs, ss =>
    Verity.Circuit.traceBeside ((ss.head?.map (·.trace r)).getD fun _ => 0) (tracesWith r' rs ss.tail)

namespace JointAuditV5

open Classical

variable (w : JointAuditV5) [Nonempty w.EndΩ]

noncomputable def memberX (τ : Strategy w.game) (k : Fin w.ms.length) : Fin (w.ms.get k).Cc.N → Bool :=
  (w.ms.get k).Xr (w.memberSamplingDesign τ k) (w.oneClass_memberSamplingDesign τ k) (w.memberStrat τ k)

theorem memberIncorrect_eq (τ : Strategy w.game) (k : Fin w.ms.length) :
    w.memberIncorrect τ k = (w.ms.get k).incorrectAt (reg (w.memberSim τ k)) (w.memberX τ k) :=
  rfl

noncomputable def seats (τ : Strategy w.game) : List (Seat w.Reg w.nJ) :=
  List.ofFn fun k : Fin w.ms.length => (⟨w.ms.get k, w.memberX τ k⟩ : Seat w.Reg w.nJ)

def ReadsAt (R : Verity.ExecutionPhaseFiles) (op : (g : Fin R.circuit.N) → BitVec (R.circuit.w g) → Prop)
    (tr : R.circuit.Trace) (τ : Strategy w.game) : Prop :=
  ReadsWith R.first R.rest op tr (w.seats τ)

def UnboundIncorrect : Prop :=
  w.Laid → ∀ (R : Verity.ExecutionPhaseFiles) (op : (g : Fin R.circuit.N) → BitVec (R.circuit.w g) → Prop)
    (tr : R.circuit.Trace) (τ : Strategy w.game), w.ReadsAt R op tr τ →
      (ResourceAccounting.unbound R.parts op tr).image Fin.val ⊆ (w.incorrectSet τ).image Fin.val

variable {w}

theorem UnboundIncorrect.card_le (h : w.UnboundIncorrect) (hl : w.Laid) (R : Verity.ExecutionPhaseFiles)
    (op : (g : Fin R.circuit.N) → BitVec (R.circuit.w g) → Prop) (tr : R.circuit.Trace) (τ : Strategy w.game)
    (hr : w.ReadsAt R op tr τ) : (ResourceAccounting.unbound R.parts op tr).card ≤ (w.incorrectSet τ).card := by
  have hs := Finset.card_le_card (h hl R op tr τ hr)
  rwa [Finset.card_image_of_injective _ Fin.val_injective, Finset.card_image_of_injective _ Fin.val_injective] at hs

end JointAuditV5

end Flock.Guarantees

namespace Verity.Sampled

def ExecutionPhase.UnboundIncorrect (W : ExecutionPhase) : Prop := ∀ (t : ℕ) [Nonempty (W.joint t).EndΩ], (W.joint t).UnboundIncorrect

end Verity.Sampled
