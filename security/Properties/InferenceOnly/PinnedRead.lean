module

public import InferenceOnlyCheck.Theta
public import Properties.InferenceOnly.Opening
public import Properties.InferenceOnly.RecordForm
public import Properties.Flock.Circuit.UnitInst
public import Properties.Verity.Sha512
public import Definitions.Hash.EmbeddingRead
public import Flock.Hm96
public import Flock.ReadNet.Template

/-! A pinned reader's unit returns its table's row; a pin is the closure digest of the reader's Definition and pins
no table and no root. -/

@[expose] public section

namespace InferenceOnlyCheck.Properties.Record

open Lean (Json)
open Flock.Program (Prog Defn)
open Spot.RecordText (Row Held)
open ResourceAccounting

def PinnedReadAt (F : Form) (c : F.Case) (j tr : Json) (P : Prog) (r : Row) (name p pin : String) (k : ℕ) (dfn : Defn)
    (ro : Roles) (out : String) (tags : Flock.Tags) (bytes : ByteArray) (tables : List (String × ByteArray))
    (ct : Flock.Circuit) (tu : Flock.TmplUnit) (hB : FlockSoundness.Built (Flock.Desc.tmpl ct).unroll.unit) (n : ℕ)
    (consts : Fin (Flock.Bridge.circuitOf (.tmpl ct) hB n).N → Option Bool) (u : Fin n)
    (q : ReadUnit.Ports (Flock.Bridge.circuitOf (.tmpl ct) hB n))
    (Yc : (Flock.Bridge.circuitOf (.tmpl ct) hB n).Trace) (table : ℕ → Hash.RowLeaf.Opening) (ok : Bool) : Prop :=
  F.read j tr = .ok c ∧
    Flock.Program.decode (F.inst c).program.toUTF8 = .ok P ∧
    F.held c r = some (.opened name p pin) ∧
    P.index[r.tid]? = some k ∧ P.defs[k]? = some dfn ∧
    F.layout dfn = some ro ∧
    Flock.Desc.parseStmt tags bytes tables = .ok (.tmpl ct) ∧
    ct.tmpl = some tu ∧ Flock.ReadNet.check q.count q.words tu.t = true ∧ ct.outNet = ct.unitNet ∧
    laidOutAs ct P r.tid u.val r.unit = true ∧
    portsOf ct hB n dfn r.unit ro out q.count q.words = some q ∧
    consts (Flock.Bridge.constGate (.tmpl ct) hB n u) = some true ∧
    (Flock.Bridge.partsOf (.tmpl ct) hB n).Correct Yc u ∧
    (∀ g ∈ (Flock.Bridge.partsOf (.tmpl ct) hB n).io u, ∀ b, consts g = some b → ReadUnit.bit Yc g = b) ∧
    q.okOf Yc = ok ∧
    ∀ i < q.count, Hash.RowLeaf.accepts Verity.sha512 Verity.sha512
      (q.verifier Flock.Hm96.Default512.key.data.toList (q.domainOf Yc)) (q.rootOf Yc) i (table i)

def PinnedRead (F : Form) (c : F.Case) (j tr : Json) (P : Prog) (r : Row) (name p pin : String) (k : ℕ) (dfn : Defn)
    (ro : Roles) (out : String) (tags : Flock.Tags) (bytes : ByteArray) (tables : List (String × ByteArray))
    (ct : Flock.Circuit) (tu : Flock.TmplUnit) (hB : FlockSoundness.Built (Flock.Desc.tmpl ct).unroll.unit) (n : ℕ)
    (consts : Fin (Flock.Bridge.circuitOf (.tmpl ct) hB n).N → Option Bool) (u : Fin n)
    (q : ReadUnit.Ports (Flock.Bridge.circuitOf (.tmpl ct) hB n))
    (Yc : (Flock.Bridge.circuitOf (.tmpl ct) hB n).Trace) (table : ℕ → Hash.RowLeaf.Opening) : Prop :=
  PinnedReadAt F c j tr P r name p pin k dfn ro out tags bytes tables ct tu hB n consts u q Yc table true

def ReturnsRow (ct : Flock.Circuit) (hB : FlockSoundness.Built (Flock.Desc.tmpl ct).unroll.unit) (n : ℕ) (u : Fin n)
    (q : ReadUnit.Ports (Flock.Bridge.circuitOf (.tmpl ct) hB n))
    (Yc : (Flock.Bridge.circuitOf (.tmpl ct) hB n).Trace) (table : ℕ → Hash.RowLeaf.Opening) : Prop :=
  q.hOf ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u) =
      Hash.EmbeddingRead.h q.count q.words (q.tokOf ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u))
        (table (q.tokOf ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u))) ∨
    Hash.CollisionIn Verity.sha512
      (Hash.RowLeaf.outerHashed Verity.sha512 Verity.sha512
          (q.verifier Flock.Hm96.Default512.key.data.toList (q.domainOf Yc))
          (q.tokOf ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u))
          (q.opening ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u)) ++
        Hash.RowLeaf.outerHashed Verity.sha512 Verity.sha512
          (q.verifier Flock.Hm96.Default512.key.data.toList (q.domainOf Yc))
          (q.tokOf ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u))
          (table (q.tokOf ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u)))) ∨
    Hash.CollisionIn Verity.sha512
      (Hash.RowLeaf.innerHashed Verity.sha512
          (q.verifier Flock.Hm96.Default512.key.data.toList (q.domainOf Yc))
          (q.opening ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u)) ++
        Hash.RowLeaf.innerHashed Verity.sha512
          (q.verifier Flock.Hm96.Default512.key.data.toList (q.domainOf Yc))
          (table (q.tokOf ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u))))

/-- A read inside a fused block is outside it: `portsOf` gives no gates for it. By itself it fixes no table: no
hypothesis ties the root and domain id the unit's gates hold in `Yc` to a registered root, because every
hypothesis on `Yc` is at the unit `u` alone. -/
def PinnedReadBinds : Prop :=
  ∀ (F : Form) (c : F.Case) (j tr : Json) (P : Prog) (r : Row) (name p pin : String) (k : ℕ) (dfn : Defn) (ro : Roles)
    (out : String) (tags : Flock.Tags) (bytes : ByteArray) (tables : List (String × ByteArray)) (ct : Flock.Circuit)
    (tu : Flock.TmplUnit) (hB : FlockSoundness.Built (Flock.Desc.tmpl ct).unroll.unit) (n : ℕ)
    (consts : Fin (Flock.Bridge.circuitOf (.tmpl ct) hB n).N → Option Bool) (u : Fin n)
    (q : ReadUnit.Ports (Flock.Bridge.circuitOf (.tmpl ct) hB n))
    (Yc : (Flock.Bridge.circuitOf (.tmpl ct) hB n).Trace) (table : ℕ → Hash.RowLeaf.Opening),
    PinnedRead F c j tr P r name p pin k dfn ro out tags bytes tables ct tu hB n consts u q Yc table →
      ReturnsRow ct hB n u q Yc table

/-- The verdict bit does the work: some instance meets every hypothesis of `PinnedRead` but the verdict bit, and its
unit returns neither the opening's row nor the fill row. It says nothing of the collision disjuncts. Not proved:
it needs runs of code on a file, which the kernel does not evaluate. -/
def PinnedReadNontrivial : Prop :=
  ∃ (F : Form) (c : F.Case) (j tr : Json) (P : Prog) (r : Row) (name p pin : String) (k : ℕ) (dfn : Defn) (ro : Roles)
    (out : String) (tags : Flock.Tags) (bytes : ByteArray) (tables : List (String × ByteArray)) (ct : Flock.Circuit)
    (tu : Flock.TmplUnit) (hB : FlockSoundness.Built (Flock.Desc.tmpl ct).unroll.unit) (n : ℕ)
    (consts : Fin (Flock.Bridge.circuitOf (.tmpl ct) hB n).N → Option Bool) (u : Fin n)
    (q : ReadUnit.Ports (Flock.Bridge.circuitOf (.tmpl ct) hB n))
    (Yc : (Flock.Bridge.circuitOf (.tmpl ct) hB n).Trace) (table : ℕ → Hash.RowLeaf.Opening),
    PinnedReadAt F c j tr P r name p pin k dfn ro out tags bytes tables ct tu hB n consts u q Yc table false ∧
      q.opening ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u) =
        table (q.tokOf ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u)) ∧
      q.hOf ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u) ≠
        Hash.EmbeddingRead.h q.count q.words (q.tokOf ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u))
          (table (q.tokOf ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u)))

/-- Of the check's own form, some instance meets `PinnedRead`, so `PinnedReadBinds` is not vacuous of it. Not proved:
it rests on `Provenance` for the check's form and on runs of code on a file, which the kernel does not evaluate. -/
def PinnedReadHonest : Prop :=
  ∃ (c : InferenceOnlyCheck.Record.SecurityProofs.form.Case) (j tr : Json) (P : Prog) (r : Row) (name p pin : String) (k : ℕ)
    (dfn : Defn) (ro : Roles) (out : String) (tags : Flock.Tags) (bytes : ByteArray)
    (tables : List (String × ByteArray)) (ct : Flock.Circuit) (tu : Flock.TmplUnit)
    (hB : FlockSoundness.Built (Flock.Desc.tmpl ct).unroll.unit) (n : ℕ)
    (consts : Fin (Flock.Bridge.circuitOf (.tmpl ct) hB n).N → Option Bool) (u : Fin n)
    (q : ReadUnit.Ports (Flock.Bridge.circuitOf (.tmpl ct) hB n))
    (Yc : (Flock.Bridge.circuitOf (.tmpl ct) hB n).Trace) (table : ℕ → Hash.RowLeaf.Opening),
    PinnedRead InferenceOnlyCheck.Record.SecurityProofs.form c j tr P r name p pin k dfn ro out tags bytes tables ct tu hB n
      consts u q Yc table ∧
    Accepts InferenceOnlyCheck.Record.SecurityProofs.form j tr c ∧
    (∃ i i', i < q.count ∧ i' < q.count ∧ (table i).row ≠ (table i').row) ∧
    (∀ b, consts (q.tok b) = none) ∧
    q.tokOf ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u) < q.count

/-- An opening that declares no row is outside it. Not proved: it reads the check's `read` through to its refusal of
a declared row (`RecordText.inputsOf`). -/
def OpenedRowReturned : Prop :=
  ∀ (j tr : Json) (c : InferenceOnlyCheck.Record.SecurityProofs.form.Case) (P : Prog) (r : Row) (name p pin : String) (k : ℕ)
    (dfn : Defn) (ro : Roles) (o out : String) (count words : ℕ),
    Accepts InferenceOnlyCheck.Record.SecurityProofs.form j tr c →
    Flock.Program.decode (InferenceOnlyCheck.Record.SecurityProofs.form.inst c).program.toUTF8 = .ok P →
    InferenceOnlyCheck.Record.SecurityProofs.form.held c r = some (.opened name p pin) →
    P.index[r.tid]? = some k → P.defs[k]? = some dfn →
    InferenceOnlyCheck.Record.SecurityProofs.form.layout dfn = some ro →
    readRows dfn ro out count words = true →
    InferenceOnlyCheck.Record.SecurityProofs.form.openedRow c r = some o → o = out

end InferenceOnlyCheck.Properties.Record
