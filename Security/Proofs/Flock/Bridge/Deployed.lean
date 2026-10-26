import Proofs.Flock.Bridge.Defs
import Proofs.Flock.Claims.Sampling.Deployed

/-!
# The deployed sampling commitment at a C-Flock session

`commitDep`: the deployed trace commitment (`SamplingDeploy.commit`: hm96 leaves of `sha512/row/v2` rows under the
verifier's own registered frame-v3-sha512 path check), with every field of its layout a function of the session's files
and the game's rows, none a free parameter:

* `rows`: each wire's registered row (`RowAt`): its port, its leaf index there (the row's unit), the row's width and the
  wire's bit in it. At a C-Flock session it is the game's own row map (`HiddenExec.rowOfH` at `portRowW`), read off the
  circuit file.
* `atOf`: a claims position's root and leaf index, its wire's row's port's root among the statement's roots and the
  row's unit; `none` at a position no row holds.
* `placeOf`: the position's row width and bit, from `rows`.
* `saltOf`, `pathOf`, `bitsOf`: an opening's data is its row's salt (`saltLen` bytes), its frame-v3 path
  (`64 · log2ceil nl` bytes) and the row's bits, one byte each.
* `portsOf`: the circuit file's ports, each with its root among the statement's roots, `nl` leaves, its name as its
  frame-v3 domain and the `hm96-sha512/row/v2` leaf schema. `RowsV2 c`: every port of `c` is an unsegmented row/v2
  row, so that schema is its own.

`rootsOf c pub nl`: the frame-v3 roots of the registration's rows: per port, the root over the `nl` units' tree leaves of
C-Flock's hm96 leaf of the public file's `b ‖ c` there (`Hm96.Default512.treeLeaf (pub.digest u q np)`, what the game's
registered leaf is at a row).

`witDep`: the evaluation's values as claims openings, each with data its position passes under, where some does.

`LayoutIs`: a sampled verifier's code (`SamplingDeploy.Code`) whose commitment fields are these.

`commitDep_binds`: `commitDep` binds at every `H` (`pathCommit_binds`, the hm96 leaf and the registered path check).
-/

namespace Flock.Bridge

open Claims Claims.Sampling
open Properties.NetworkCertifier.Bytes (bs)

/-- **A wire's registered row**: its port, its leaf index there (the row's unit), the row's width in bits and the
wire's bit in it. -/
structure RowAt where
  port : ℕ
  leaf : ℕ
  width : ℕ
  place : ℕ

section Layout

variable (C : FlockSoundness.Audit.Circuit) (rows : Fin C.N → Option RowAt)

/-- The wire at claims position `i`. -/
def wireOf (i : ℕ) : Option (Fin C.N) := (List.finRange C.N).find? fun g => decide (posOf C g = i)

/-- The registered row of the wire at claims position `i`. -/
def rowOfPos (i : ℕ) : Option RowAt := (wireOf C i).bind rows

/-- **The root and leaf index of a position** under the statement's roots: its row's port's root, and the row's unit. -/
def atOf (roots : List Msg) (i : ℕ) : Option (Msg × ℕ) :=
  (rowOfPos C rows i).bind fun r => roots[r.port]?.map fun rt => (rt, r.leaf)

/-- **Where a position's bit sits**: its row's width and its bit there. -/
def placeOf (i : ℕ) : (w : ℕ) × Fin w :=
  match rowOfPos C rows i with
  | some r =>
    if h : r.place < r.width then ⟨r.width, ⟨r.place, h⟩⟩ else ⟨r.place + 1, ⟨r.place, Nat.lt_succ_self _⟩⟩
  | none => ⟨1, 0⟩

end Layout

/-- The salt's bytes: what `--zk` rows carry, 192 bytes. -/
def saltLen : ℕ := 192

/-- An opening's salt: its data's first `saltLen` bytes. -/
def saltOf (o : Opening Bool) : Msg := o.data.take saltLen

/-- An opening's frame-v3 path, `d` levels: the `64 · d` bytes after the salt. -/
def pathOf (d : ℕ) (o : Opening Bool) : Msg := (o.data.drop saltLen).take (64 * d)

/-- An opening's row bit `k`: the `k`th byte after the salt and the path is `1`. -/
def bitsOf (d : ℕ) (data : Msg) (k : ℕ) : Bool := (data.drop (saltLen + 64 * d))[k]? == some 1

/-- A port's frame-v3 domain: its name in the circuit file. -/
def domOf (c : Flock.Circuit) (q : ℕ) : ByteArray := ((c.ports[q]?.map (·.name)).getD "").toUTF8

/-- **The registered ports**: the circuit file's, each at its root among `roots`, with `nl` leaves, its domain and the
`hm96-sha512/row/v2` leaf schema. -/
def portsOf (c : Flock.Circuit) (nl : ℕ) (roots : List Msg) : Array Flock.Registered.Port :=
  (Array.range c.ports.size).map fun q =>
    { name := (c.ports[q]?.map (·.name)).getD "", value := "", root := bytes (roots.getD q []), leaves := nl,
      words := none, wordBits := 0, domain := domOf c q, readsRel := #[], schema := Flock.HmRow.ROW_SCHEMA_V2 }

/-- **Every port of the circuit file is an unsegmented `sha512/row/v2` row** (`v2`, no `segs`): the ports `portsOf`
registers at the `hm96-sha512/row/v2` schema are then the circuit file's own. -/
@[reducible] def RowsV2 (c : Flock.Circuit) : Prop := c.ports.all (fun p => p.v2 && p.segs.isNone) = true

instance instDecidableRowsV2 (c : Flock.Circuit) : Decidable (RowsV2 c) := inferInstanceAs (Decidable (_ = true))

/-- **The registered tree leaf of row `(u, q)`**: C-Flock's hm96 tree leaf of the public file's `b ‖ c` there. -/
def regLeafOf (c : Flock.Circuit) (pub : Flock.Public) (q u : ℕ) : ByteArray :=
  Flock.Hm96.Default512.treeLeaf (pub.digest u q c.ports.size)

/-- **The registration's frame-v3 roots**: per port, the root over its `nl` rows' tree leaves. -/
def rootsOf (c : Flock.Circuit) (pub : Flock.Public) (nl : ℕ) : List Msg :=
  (List.range c.ports.size).map fun q =>
    bs (Flock.FV3.root (domOf c q) ((Array.range nl).map fun u =>
      Flock.Registered.leafAs Flock.HmRow.ROW_SCHEMA_V2 (domOf c q) u (regLeafOf c pub q u)) Sha512.hash)

/-- **The deployed trace commitment at a session's rows**: `SamplingDeploy.commit`'s form, at the layout above. -/
noncomputable def commitDep (C : FlockSoundness.Audit.Circuit) (rows : Fin C.N → Option RowAt) (c : Flock.Circuit)
    (nl : ℕ) (roots : List Msg) : Commitment Bool :=
  pathCommit (atOf C rows) (hm96Leaf (SamplingDeploy.row (placeOf C rows) (bitsOf (Flock.log2ceil nl))) saltOf)
    (pathOf (Flock.log2ceil nl)) (regPath (portsOf c nl roots)) fun _ _ _ => []

/-- **`commitDep` binds, at every `H`**: the hm96 leaf of a placing row binds the value, and the registered path check
the leaf. -/
theorem commitDep_binds (H : Msg → Msg) (C : FlockSoundness.Audit.Circuit) (rows : Fin C.N → Option RowAt)
    (c : Flock.Circuit) (nl : ℕ) (roots : List Msg) : (commitDep C rows c nl roots).Binds H :=
  pathCommit_binds (hm96Leaf_binds H (SamplingDeploy.row_places _ _) _) (regPath_binds H _)

/-- **A sampled verifier's code whose commitment is `commitDep`'s layout.** -/
def LayoutIs (C : FlockSoundness.Audit.Circuit) (rows : Fin C.N → Option RowAt) (c : Flock.Circuit) (nl : ℕ)
    (roots : List Msg) (cd : SamplingDeploy.Code) : Prop :=
  cd.at_ = atOf C rows ∧ cd.place = placeOf C rows ∧ cd.bits = bitsOf (Flock.log2ceil nl) ∧ cd.salt = saltOf ∧
    cd.π = pathOf (Flock.log2ceil nl) ∧ cd.own = portsOf c nl roots

open Classical in
/-- **The evaluation's values as claims openings**: at each wire, its value in `C`'s evaluation on `xv`'s inputs, with
data its position passes `Cm` under `roots` at, where some does. -/
noncomputable def witDep (C : FlockSoundness.Audit.Circuit) (Cm : Commitment Bool) (H : Msg → Msg) (roots : List Msg)
    (xv : Fin C.N → Bool) : Wit :=
  ⟨(List.finRange C.N).map fun g =>
    if h : ∃ d : Msg, Cm.Opens H roots ⟨posOf C g, C.eval xv g, d⟩ = true then
      ⟨posOf C g, C.eval xv g, Classical.choose h⟩
    else ⟨posOf C g, C.eval xv g, []⟩⟩

end Flock.Bridge
