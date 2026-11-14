module

public import Properties.Nci.Record
public import Properties.Flock.Circuit.CircuitOf
public import Properties.Verity.Sha512
public import Definitions.Accounting.ReadUnit
public import Flock.Hm96

/-!
# A pinned reader returns its table's row: the two premises

The check holds a row as an opening (`Held.opened`) where its reader's Definition is pinned and has a reader's layout
(`Good`), and counts none of that row's bits as fresh. The count is right only where the reader's unit returns no choice
of the prover's. Two premises say so, each a named `Prop` that `NciCheck.Guarantees.PinnedReadBinds` takes as a
hypothesis:

* `Accounting.ReadUnit.GatesRefine`, at a unit `u` and ports `q` of the circuit a parsed circuit file describes
  (`Flock.Bridge.circuitOf`): the unit's gates compute the read's reference (`Hash.EmbeddingRead`).
* `OpeningTie`: unit `un` of template `tid` of the record's Program is that unit, at those ports. The record names a
  unit by its template and its number and a reader's ports by its layout's roles (`Roles`); the circuit file names
  gates. No definition gives one from the other, so the tie is on values: the bits the unit's ports hold in a
  valuation of the Program's gates are the bits `q`'s gates hold in a transcript of the circuit file's circuit.

A valuation `val` gives bit `t` of the Program's gate `g` as `val g t`. A port is read at the Program's own gates, where
every unit's parameters are input gates of its own (`OwnRows`), and not in the wired circuit (`Served`), which reads a
wired leaf at its writer's gates. A port is read whole, whatever the record's wiring cuts it into.

**What the guarantee says.** Where the unit is correct in the transcript and the registered constants hold, the
verdict the unit's verdict row holds is 1, and a table is committed under the root and the domain id that its root and
domain ports hold: the row the unit returns (`Ports.hOf`, in the unit's local evaluation) is that table's row at the key
its key port holds, or the fill row where the key is at or past the table's count, or SHA-512 collides on what the
unit's check and the table's opening hashed.

**What it does not say.** It is on the row returned, not on the opened ports: at a key at or past the count the
reference accepts any witness, and the opened ports hold what the prover likes. It names no row of the record for the
row returned, which `Roles` has no role for. It does not give a valuation and a transcript that the tie relates, for
any record.
-/

@[expose] public section

namespace NciCheck.Properties.Record

open Flock.Program (Prog Scope)
open Flock.TemplateQuery (TRange)
open Accounting

/-! ## What a unit's ports hold -/

/-- **Unit `un` of template `tid` holds the bits `bs` at the leaves `Leaf`** in the valuation `val` of the Program's
gates: each bit of `bs` lies in one of the leaves, and is that leaf's gate's bit wherever a leaf holds it. `Leaf sc l g
lo hi` says leaf `l` of the activation `sc` is gate `g`, at bits `[lo, hi)`. -/
def Holds (P : Prog) (rs : Array TRange) (val : ℕ → ℕ → Bool) (tid : String) (un : ℕ)
    (Leaf : Scope → ℕ → ℕ → ℕ → ℕ → Prop) (bs : List Bool) : Prop :=
  ∀ (b : ℕ) (hb : b < bs.length),
    (∃ sc l g lo hi, UnitAt P rs tid un sc ∧ Leaf sc l g lo hi ∧ lo ≤ b ∧ b < hi) ∧
      ∀ sc l g lo hi, UnitAt P rs tid un sc → Leaf sc l g lo hi → lo ≤ b → b < hi →
        bs[b]'hb = val g (b - lo)

/-- The parameter `port` of unit `un` of template `tid`, read whole (`InLeaf`, cut one way), holds the bits `bs`. -/
abbrev PortHolds (P : Prog) (rs : Array TRange) (val : ℕ → ℕ → Bool) (tid : String) (un : ℕ) (port : String)
    (bs : List Bool) : Prop :=
  Holds P rs val tid un (fun sc => InLeaf P sc 1 port 0) bs

/-- The output row `out` of unit `un` of template `tid` (`OutLeaf`) holds the bits `bs`. -/
abbrev OutHolds (P : Prog) (rs : Array TRange) (val : ℕ → ℕ → Bool) (tid : String) (un : ℕ) (out : String)
    (bs : List Bool) : Prop :=
  Holds P rs val tid un (fun sc => OutLeaf P sc out) bs

/-! ## The tie -/

/-- **Unit `un` of template `tid` is unit `u` of a parsed circuit file, at the ports `q`**, in the valuation `val` and
the transcript `Yc`. The circuit file `bytes` parses to the template statement `ct`, as the verifier reads it
(`Flock.Desc.parseStmt`), and `u` is a unit of its circuit at the population `n`. Under the roles `ro` of the unit's
reader: its key port holds the index `q`'s index gates hold; its root and domain ports hold what `q`'s root and domain
id gates hold; its verdict row holds `q`'s verdict bit; and its opened ports are three, which hold the witness row, its
salt and its sibling path, a digest a level. The index and the witness are read in `u`'s local evaluation, as the
read's reference is (`Accounting.ReadUnit.Ports.Computes`), and the root, the domain id and the verdict in the
transcript, as the registered rows are (`Accounting.ReadUnit.RegisteredRows`). -/
structure OpeningTie (P : Prog) (rs : Array TRange) (tid : String) (un : ℕ) (ro : Roles) (tags : Flock.Tags)
    (bytes : ByteArray) (tables : List (String × ByteArray)) (ct : Flock.Circuit)
    (hB : FlockSoundness.Built (Flock.Desc.tmpl ct).unroll.unit) (n : ℕ) (u : Fin n)
    (q : ReadUnit.Ports (Flock.Bridge.circuitOf (.tmpl ct) hB n)) (val : ℕ → ℕ → Bool)
    (Yc : (Flock.Bridge.circuitOf (.tmpl ct) hB n).Trace) : Prop where
  parses : Flock.Desc.parseStmt tags bytes tables = .ok (.tmpl ct)
  key : PortHolds P rs val tid un ro.key
    (ReadUnit.bits ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u) q.tok)
  root : PortHolds P rs val tid un ro.root (ReadUnit.bits Yc q.root)
  domain : PortHolds P rs val tid un ro.domain (ReadUnit.bits Yc q.domain)
  verdict : OutHolds P rs val tid un ro.verdict [ReadUnit.bit Yc q.ok]
  opened : ∃ row salt path, ro.opened = [row, salt, path] ∧
    PortHolds P rs val tid un row
      (ReadUnit.bits ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u) q.row) ∧
    PortHolds P rs val tid un salt
      (ReadUnit.bits ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u) q.salt) ∧
    PortHolds P rs val tid un path
      ((List.finRange (Hash.Tree.depth q.count)).flatMap fun j =>
        ReadUnit.bits ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u) (q.path j))

end NciCheck.Properties.Record

namespace NciCheck.Guarantees

open Lean (Json)
open Flock.Program (Prog Defn)
open Flock.TemplateQuery (TRange)
open NciCheck.RecordText (Row Held)
open NciCheck.Properties.Record
open Accounting

/-- **A pinned reader's unit returns its table's row.** Let a check read a record and hold its row `r` as an opening
under a pin, the reader's Definition having the roles `ro`. Take the two premises: the read's gates compute its
reference at unit `u` and ports `q` of a parsed circuit file's circuit (`Accounting.ReadUnit.GatesRefine`, under SHA-512
and the served hm96 key), and `r`'s unit is that unit at those ports, in a valuation `val` of the Program's gates and a
transcript `Yc` of that circuit (`OpeningTie`). Let `u` be correct in `Yc` and the registered constants hold there; let
the unit's key port hold the 32 bits `kb`, its root port the 512 bits `Rb`, its domain port the 512 bits `db` and its
verdict row a 1; and let `table` be committed under that root and domain id. Then the row `u` returns in its local
evaluation is `table`'s row at the key `kb`, or the fill row where the key is at or past the table's count
(`Hash.EmbeddingRead.h`), or SHA-512 collides on what the unit's check and `table`'s opening at the key hashed. -/
def PinnedReadBinds : Prop :=
  ∀ (F : Form) (c : F.Case) (j tr : Json) (P : Prog) (rs : Array TRange) (r : Row) (name p pin : String) (k : ℕ)
    (dfn : Defn) (ro : Roles) (tags : Flock.Tags) (bytes : ByteArray) (tables : List (String × ByteArray))
    (ct : Flock.Circuit) (hB : FlockSoundness.Built (Flock.Desc.tmpl ct).unroll.unit) (n : ℕ)
    (consts : Fin (Flock.Bridge.circuitOf (.tmpl ct) hB n).N → Option Bool) (u : Fin n)
    (q : ReadUnit.Ports (Flock.Bridge.circuitOf (.tmpl ct) hB n)) (val : ℕ → ℕ → Bool)
    (Yc : (Flock.Bridge.circuitOf (.tmpl ct) hB n).Trace) (table : ℕ → Hash.RowLeaf.Opening) (kb Rb db : List Bool),
    F.read j tr = .ok c → Flock.Program.decode (F.inst c).program.toUTF8 = .ok P →
    Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P)
      ((F.wiring c).templates.map (·.1)).toArray = .ok rs →
    F.held c r = some (.opened name p pin) → P.index[r.tid]? = some k → P.defs[k]? = some dfn →
    F.layout dfn = some ro →
    ReadUnit.GatesRefine Verity.sha512 Verity.sha512 Flock.Hm96.Default512.key.data.toList consts
      (Flock.Bridge.partsOf (.tmpl ct) hB n) u q →
    OpeningTie P rs r.tid r.unit ro tags bytes tables ct hB n u q val Yc →
    (Flock.Bridge.partsOf (.tmpl ct) hB n).Correct Yc u →
    (∀ g ∈ (Flock.Bridge.partsOf (.tmpl ct) hB n).io u, ∀ b, consts g = some b → ReadUnit.bit Yc g = b) →
    kb.length = 32 → PortHolds P rs val r.tid r.unit ro.key kb →
    Rb.length = 512 → PortHolds P rs val r.tid r.unit ro.root Rb →
    db.length = 512 → PortHolds P rs val r.tid r.unit ro.domain db →
    OutHolds P rs val r.tid r.unit ro.verdict [true] →
    (∀ i < q.count, Hash.RowLeaf.accepts Verity.sha512 Verity.sha512
      (q.verifier Flock.Hm96.Default512.key.data.toList (ReadUnit.digestOf db)) (ReadUnit.digestOf Rb) i (table i)) →
    q.hOf ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u) =
        Hash.EmbeddingRead.h q.count q.words (ReadUnit.num kb) (table (ReadUnit.num kb)) ∨
      Hash.CollisionIn Verity.sha512
        (Hash.RowLeaf.outerHashed Verity.sha512 Verity.sha512
            (q.verifier Flock.Hm96.Default512.key.data.toList (ReadUnit.digestOf db)) (ReadUnit.num kb)
            (q.opening ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u)) ++
          Hash.RowLeaf.outerHashed Verity.sha512 Verity.sha512
            (q.verifier Flock.Hm96.Default512.key.data.toList (ReadUnit.digestOf db)) (ReadUnit.num kb)
            (table (ReadUnit.num kb))) ∨
      Hash.CollisionIn Verity.sha512
        (Hash.RowLeaf.innerHashed Verity.sha512
            (q.verifier Flock.Hm96.Default512.key.data.toList (ReadUnit.digestOf db))
            (q.opening ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u)) ++
          Hash.RowLeaf.innerHashed Verity.sha512
            (q.verifier Flock.Hm96.Default512.key.data.toList (ReadUnit.digestOf db)) (table (ReadUnit.num kb)))

end NciCheck.Guarantees
