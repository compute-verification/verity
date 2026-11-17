module

public import InferenceOnlyVerifier.Theta
public import Properties.InferenceOnly.Opening
public import Properties.InferenceOnly.RecordForm
public import Properties.Flock.Circuit.UnitInst
public import Properties.Verity.Sha512
public import Definitions.Hash.EmbeddingRead
public import Flock.Hm96
public import Flock.ReadNet.Template

/-!
# A pinned reader's unit returns its table's row: the statements

`PinnedReadAt`, a pinned read as a check holds it and its circuit file lays it out, at a verdict bit; `PinnedRead`, at
the verdict bit 1; `ReturnsRow`, that the unit returns its table's row, the fill row, or a SHA-512 collision;
`PinnedReadBinds`, that a pinned read returns its row; `PinnedReadNontrivial`, that at the verdict bit 0 some pinned
read does not; `PinnedReadHonest`, that the check's own form (`InferenceOnlyCheck.Record.SecurityProofs.form`) has a pinned
read; and `OpenedRowReturned`, that an opening's declared row is the row its reader returns. The proofs are
`Proofs.Properties.InferenceOnly.Lemmas.Opening`'s.

A pinned read is a read by a pinned reader: the pin is the closure digest of the reader's Definition
(`InferenceOnlyCheck.Record.pinOf`), and pins no table and no root.
-/

@[expose] public section

namespace InferenceOnlyVerifier.Properties.Record

open Lean (Json)
open Flock.Program (Prog Defn)
open Spot.RecordText (Row Held)
open ResourceAccounting

/-- **A pinned read at the verdict bit `ok`, as a check holds it and its circuit file lays it out**: the hypotheses of
`PinnedReadBinds`, in its order, with the verdict bit `ok`. The check reads the record and holds its row `r` as an
opening under a pin; the reader's Definition `dfn` has the roles `ro`, the output row `out` being the row it returns;
the file `bytes` parses to the templated statement `ct` with the unit template `tu`, which the read's check accepts;
the file's unit `u` is `r`'s, with the read unit's gates `q`; the unit's constant is registered as 1, `u` is correct
in `Yc` with its registered constants there, its verdict bit is `ok`, and `table` is committed under the root and
domain id its gates hold. -/
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

/-- **A pinned read**: `PinnedReadAt` with the verdict bit 1. -/
def PinnedRead (F : Form) (c : F.Case) (j tr : Json) (P : Prog) (r : Row) (name p pin : String) (k : ℕ) (dfn : Defn)
    (ro : Roles) (out : String) (tags : Flock.Tags) (bytes : ByteArray) (tables : List (String × ByteArray))
    (ct : Flock.Circuit) (tu : Flock.TmplUnit) (hB : FlockSoundness.Built (Flock.Desc.tmpl ct).unroll.unit) (n : ℕ)
    (consts : Fin (Flock.Bridge.circuitOf (.tmpl ct) hB n).N → Option Bool) (u : Fin n)
    (q : ReadUnit.Ports (Flock.Bridge.circuitOf (.tmpl ct) hB n))
    (Yc : (Flock.Bridge.circuitOf (.tmpl ct) hB n).Trace) (table : ℕ → Hash.RowLeaf.Opening) : Prop :=
  PinnedReadAt F c j tr P r name p pin k dfn ro out tags bytes tables ct tu hB n consts u q Yc table true

/-- **The unit returns its table's row**: the row `u` returns in its local evaluation is `table`'s row at the index its
index gates hold there, or the fill row where the index is at or past the table's count (`Hash.EmbeddingRead.h`), or
SHA-512 collides on what the unit's check and `table`'s opening at the index hashed. -/
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

/-- **A pinned reader's unit returns its table's row.** Let a check read a record and hold its row `r` as an opening
under a pin, the reader's Definition having the roles `ro`. Let the circuit file `bytes` parse to the templated
statement `ct`, as the verifier reads it, with the unit template `tu`; let the read's check accept `tu` for a table of
`q.count` rows of `q.words` words, and the unit return the circuit's outputs. Let the file's unit `u` be `r`'s unit,
and `q` be the read unit's gates there at the roles and the output row `out` the reader returns. Let the unit's
constant be registered as 1. Let `u` be correct in the transcript `Yc` and the registered constants hold there; let
the verdict bit be 1; and let `table` be committed under the root and the domain id the root and domain gates hold
(`PinnedRead`). Then the row `u` returns in its local evaluation is `table`'s row at the key its key gates hold
there, or the fill row where the key is at or past the table's count (`Hash.EmbeddingRead.h`), or SHA-512 collides on
what the unit's check and `table`'s opening at the key hashed (`ReturnsRow`). A read inside a fused block is outside it:
its row goes straight into the block's GEMM and never leaves the unit, so `portsOf` gives no gates for it.

By itself it fixes no table: `table` is whatever is committed under the root and domain id the unit's gates hold in
`Yc`, and no hypothesis ties those to a registered root, since every hypothesis on `Yc` is at the unit `u` alone. What
would tie them is that `Yc` is a transcript the proof system accepted, whose input rows open against the window's
registered roots, with this unit's root and domain id rows among them; no statement here says so, and its form waits
on SPOT's predicate for a proof block's input rows. -/
def PinnedReadBinds : Prop :=
  ∀ (F : Form) (c : F.Case) (j tr : Json) (P : Prog) (r : Row) (name p pin : String) (k : ℕ) (dfn : Defn) (ro : Roles)
    (out : String) (tags : Flock.Tags) (bytes : ByteArray) (tables : List (String × ByteArray)) (ct : Flock.Circuit)
    (tu : Flock.TmplUnit) (hB : FlockSoundness.Built (Flock.Desc.tmpl ct).unroll.unit) (n : ℕ)
    (consts : Fin (Flock.Bridge.circuitOf (.tmpl ct) hB n).N → Option Bool) (u : Fin n)
    (q : ReadUnit.Ports (Flock.Bridge.circuitOf (.tmpl ct) hB n))
    (Yc : (Flock.Bridge.circuitOf (.tmpl ct) hB n).Trace) (table : ℕ → Hash.RowLeaf.Opening),
    PinnedRead F c j tr P r name p pin k dfn ro out tags bytes tables ct tu hB n consts u q Yc table →
      ReturnsRow ct hB n u q Yc table

/-- **The verdict bit does the work**: some form, record, circuit file and transcript meet every hypothesis of
`PinnedRead` but the verdict bit, which is 0 (`PinnedReadAt` at `false`); the opening the unit's gates hold is
`table`'s own at the index, and the row the unit returns is neither that opening's row nor the fill row. It shows the
first disjunct of `ReturnsRow` failing at an opening equal to the table's, and says nothing of the collision
disjuncts, which no kernel proof can exclude: at that opening each is a collision among one opening's own hashed
inputs. Not proved: it needs a circuit file that parses and a template the read's check accepts, runs of code on a
file, which the kernel does not evaluate. -/
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

/-- **An honest pinned read**: of the check's own form (`InferenceOnlyCheck.Record.SecurityProofs.form`), some instance, record,
circuit file and transcript meet `PinnedRead`, so `PinnedReadBinds` is not vacuous of it. So that it does not hold
emptily, by a table of one constant row: the check accepts the record (`Accepts`); the table has two different rows
within its count; and the transcript chooses the index, which no index gate's registered constant fixes and which
lies within the table's count. Not proved: that the record reads with its opening held rests on `Provenance` for the
check's form, and that the read's check accepts the template and the file parses are runs of code on a file, which
the kernel does not evaluate. -/
def PinnedReadHonest : Prop :=
  ∃ (c : InferenceOnlyVerifier.Record.SecurityProofs.form.Case) (j tr : Json) (P : Prog) (r : Row) (name p pin : String) (k : ℕ)
    (dfn : Defn) (ro : Roles) (out : String) (tags : Flock.Tags) (bytes : ByteArray)
    (tables : List (String × ByteArray)) (ct : Flock.Circuit) (tu : Flock.TmplUnit)
    (hB : FlockSoundness.Built (Flock.Desc.tmpl ct).unroll.unit) (n : ℕ)
    (consts : Fin (Flock.Bridge.circuitOf (.tmpl ct) hB n).N → Option Bool) (u : Fin n)
    (q : ReadUnit.Ports (Flock.Bridge.circuitOf (.tmpl ct) hB n))
    (Yc : (Flock.Bridge.circuitOf (.tmpl ct) hB n).Trace) (table : ℕ → Hash.RowLeaf.Opening),
    PinnedRead InferenceOnlyVerifier.Record.SecurityProofs.form c j tr P r name p pin k dfn ro out tags bytes tables ct tu hB n
      consts u q Yc table ∧
    Accepts InferenceOnlyVerifier.Record.SecurityProofs.form j tr c ∧
    (∃ i i', i < q.count ∧ i' < q.count ∧ (table i).row ≠ (table i').row) ∧
    (∀ b, consts (q.tok b) = none) ∧
    q.tokOf ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u) < q.count

/-- **An opening's declared row is the row its reader returns**: where the check's own form accepts a record and holds
its row `r` as an opening under a pin, and the opening declares a row (`Form.openedRow`), that row is the output row
`out` at which the reader's Definition is a table read (`readRows`). An opening that declares no row is outside it.
The check refuses a declared row unless its reader's template has six input rows and its outputs are that row and
then the verdict (`RecordText.inputsOf`), and `readRows` names the first of exactly two outputs. Not proved: it reads
the check's `read` through to that refusal. -/
def OpenedRowReturned : Prop :=
  ∀ (j tr : Json) (c : InferenceOnlyVerifier.Record.SecurityProofs.form.Case) (P : Prog) (r : Row) (name p pin : String) (k : ℕ)
    (dfn : Defn) (ro : Roles) (o out : String) (count words : ℕ),
    Accepts InferenceOnlyVerifier.Record.SecurityProofs.form j tr c →
    Flock.Program.decode (InferenceOnlyVerifier.Record.SecurityProofs.form.inst c).program.toUTF8 = .ok P →
    InferenceOnlyVerifier.Record.SecurityProofs.form.held c r = some (.opened name p pin) →
    P.index[r.tid]? = some k → P.defs[k]? = some dfn →
    InferenceOnlyVerifier.Record.SecurityProofs.form.layout dfn = some ro →
    readRows dfn ro out count words = true →
    InferenceOnlyVerifier.Record.SecurityProofs.form.openedRow c r = some o → o = out

end InferenceOnlyVerifier.Properties.Record
