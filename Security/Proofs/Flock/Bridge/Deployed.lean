import Proofs.Flock.Bridge.Defs
import Proofs.Flock.Claims.Sampling.Deployed
import Proofs.Flock.Soundness.Binding.RowSeg
import Proofs.Flock.Soundness.Discharge.Layout.Scope

/-!
# The deployed sampling commitment at a C-Flock session

`commitDep`: the deployed trace commitment (`SamplingDeploy.commit`'s form: hm96 leaves of rows under the verifier's
own registered frame-v3-sha512 path check), each row in its port's own format, with every field of its layout a
function of the session's files and the game's rows, none a free parameter:

* `rows`: each wire's registered row (`RowAt`): its port, its leaf index there (the row's unit), the row's width and the
  wire's bit in it. At a C-Flock session it is the game's own row map (`HiddenExec.rowOfH` at `portRowW`), read off the
  circuit file.
* `atOf`: a claims position's root and leaf index, its wire's row's port's root among the statement's roots and the
  row's unit; `none` at a position no row holds.
* `placeOf`: the position's row width and bit, from `rows`.
* `saltOf`, `pathOf`, `bitsOf`: an opening's data is its row's salt (`saltLen` bytes), its frame-v3 path
  (`64 · log2ceil nl` bytes) and the row's bits, one byte each.
* `rowDep`: an opening's row message, its row's bits (`SamplingDeploy.rowBits`) in its port's format (`rowMsg`):
  `sha512/row-seg/v1`'s outer message `Q(ℓ) ‖ d₀ ‖ …` at a segmented port, `sha512/row/v2`'s `P(ℓ) ‖ bytes` at a bit
  row, `sha512/row/v1`'s `rowPrefix (ℓ / 16) ‖ bytes` otherwise, as C-Flock's hm96 slot hashes that port's rows.
* `portsOf`: the circuit file's ports, each with its root among the statement's roots, `nl` leaves, its name as its
  frame-v3 domain and its own leaf schema (`schemaOf`: `Registered.portSchema`, as C-Flock frames the port's rows).

`RowsOk c`: every port of `c` is a row the verifier of record reads: all `sha512/row/v1` rows, or each a `sha512/row/v2`
or `sha512/row-seg/v1` row, `16 · words` bits each (`Layout.v1Ports`, `Layout.bitRows`; C-Flock's scope check
`scopeOkH` implies it, `rowsOk_of_scopeOkH`). `RowsV2 c`: every port is an unsegmented row/v2 row, where `commitDep` is
`SamplingDeploy.commit`'s, whose rows are row/v2 rows only (`LayoutIs`).

`rootsOf c pub nl`: the frame-v3 roots of the registration's rows: per port, the root over the `nl` units' tree leaves of
C-Flock's hm96 leaf of the public file's `b ‖ c` there (`Hm96.Default512.treeLeaf (pub.digest u q np)`, what the game's
registered leaf is at a row), framed at the port's schema.

`witDep`: the evaluation's values as claims openings, each with data its position passes under, where some does.

`LayoutIs`: a sampled verifier's code (`SamplingDeploy.Code`) whose commitment fields are these, at a `RowsV2` file.

`commitDep_binds`: `commitDep` binds at every `H` at a file with no segmented port (`pathCommit_binds`, the hm96 leaf
and the registered path check). A `sha512/row-seg/v1` row's outer message is not injective in the row, so there two
openings of one leaf are a SHA-512 collision of a segment's message (`Binding.RowSeg.extPair_spec`), which the leaf's
queries don't make.
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

/-- A port's leaf schema: `Registered.portSchema`'s, as C-Flock frames the port's rows (`HmRow.checkPublic`). -/
def schemaOf (c : Flock.Circuit) (q : ℕ) : String :=
  (c.ports[q]?.map Flock.Registered.portSchema).getD Flock.HmRow.ROW_SCHEMA_V2

/-- **The registered ports**: the circuit file's, each at its root among `roots`, with `nl` leaves, its domain and its
leaf schema. -/
def portsOf (c : Flock.Circuit) (nl : ℕ) (roots : List Msg) : Array Flock.Registered.Port :=
  (Array.range c.ports.size).map fun q =>
    { name := (c.ports[q]?.map (·.name)).getD "", value := "", root := bytes (roots.getD q []), leaves := nl,
      words := none, wordBits := 0, domain := domOf c q, readsRel := #[], schema := schemaOf c q }

/-- **Every port of the circuit file is a row the verifier of record reads**: all unsegmented `sha512/row/v1` rows
(`Layout.v1Ports`), or each an unsegmented `sha512/row/v2` row or a `sha512/row-seg/v1` row (`Layout.bitRows`), of
`16 · words` bits each, as C-Flock's scope check accepts them (`rowsOk_of_scopeOkH`). -/
@[reducible] def RowsOk (c : Flock.Circuit) : Prop :=
  (FlockSoundness.Discharge.Layout.v1Ports c || FlockSoundness.Discharge.Layout.bitRows c) = true

instance instDecidableRowsOk (c : Flock.Circuit) : Decidable (RowsOk c) := inferInstanceAs (Decidable (_ = true))

/-- **C-Flock's scope check implies `RowsOk`**: `scopeOkH`'s last conjunct is it. -/
theorem rowsOk_of_scopeOkH {c : Flock.Circuit} (h : FlockSoundness.Discharge.Layout.scopeOkH c = true) : RowsOk c := by
  unfold FlockSoundness.Discharge.Layout.scopeOkH at h
  simp only [Bool.and_eq_true] at h
  exact h.2

/-- **Every port of the circuit file is an unsegmented `sha512/row/v2` row** (`v2`, no `segs`), the rows
`SamplingDeploy.commit` hashes. -/
@[reducible] def RowsV2 (c : Flock.Circuit) : Prop := c.ports.all (fun p => p.v2 && p.segs.isNone) = true

instance instDecidableRowsV2 (c : Flock.Circuit) : Decidable (RowsV2 c) := inferInstanceAs (Decidable (_ = true))

/-- **A row's message in its port's format**, at the row's bits `b`: `sha512/row-seg/v1`'s outer message at a segmented
port, `sha512/row/v2`'s at a bit row, `sha512/row/v1`'s otherwise. -/
def rowMsg (p : Flock.Port) (b : List Bool) : Msg :=
  if p.segs.isSome then FlockSoundness.Binding.RowSeg.enc b.length (Flock.HmRow.packBits b)
  else if p.v2 then FlockLevel3.RowV2.enc b
  else (Flock.HmRow.rowPrefix (b.length / 16)).data.toList ++ Flock.HmRow.packBits b

/-- The port of the row at claims position `i`. -/
def portOfPos (C : FlockSoundness.Audit.Circuit) (rows : Fin C.N → Option RowAt) (c : Flock.Circuit) (i : ℕ) :
    Option Flock.Port :=
  (rowOfPos C rows i).bind fun r => c.ports[r.port]?

/-- **An opening's row message**: its row bits in its row's port's format (`rowMsg`); `sha512/row/v2`'s at a position
no port's row holds, which opens nothing (`atOf`). -/
def rowDep (C : FlockSoundness.Audit.Circuit) (rows : Fin C.N → Option RowAt) (c : Flock.Circuit)
    (place : ℕ → (w : ℕ) × Fin w) (bits : Msg → ℕ → Bool) (o : Opening Bool) : Msg :=
  match portOfPos C rows c o.pos with
  | some p => rowMsg p (SamplingDeploy.rowBits place bits o)
  | none => SamplingDeploy.row place bits o

/-- **The registered tree leaf of row `(u, q)`**: C-Flock's hm96 tree leaf of the public file's `b ‖ c` there. -/
def regLeafOf (c : Flock.Circuit) (pub : Flock.Public) (q u : ℕ) : ByteArray :=
  Flock.Hm96.Default512.treeLeaf (pub.digest u q c.ports.size)

/-- **The registration's frame-v3 roots**: per port, the root over its `nl` rows' tree leaves at its schema. -/
def rootsOf (c : Flock.Circuit) (pub : Flock.Public) (nl : ℕ) : List Msg :=
  (List.range c.ports.size).map fun q =>
    bs (Flock.FV3.root (domOf c q) ((Array.range nl).map fun u =>
      Flock.Registered.leafAs (schemaOf c q) (domOf c q) u (regLeafOf c pub q u)) Sha512.hash)

/-- **The deployed trace commitment at a session's rows**: `SamplingDeploy.commit`'s form, at the layout above, each
row in its port's format (`rowDep`). -/
noncomputable def commitDep (C : FlockSoundness.Audit.Circuit) (rows : Fin C.N → Option RowAt) (c : Flock.Circuit)
    (nl : ℕ) (roots : List Msg) : CommitmentScheme Bool :=
  pathCommit (atOf C rows) (hm96Leaf (rowDep C rows c (placeOf C rows) (bitsOf (Flock.log2ceil nl))) saltOf)
    (pathOf (Flock.log2ceil nl)) (regPath (portsOf c nl roots)) fun _ _ _ => []

/-- **An unsegmented port's row message places the row**: rows of one length with one message are one row. -/
theorem rowMsg_inj {p : Flock.Port} (hp : p.segs.isSome = false) {b b' : List Bool} (hl : b.length = b'.length)
    (h : rowMsg p b = rowMsg p b') : b = b' := by
  unfold rowMsg at h
  simp only [hp, Bool.false_eq_true, ↓reduceIte] at h
  split_ifs at h
  · unfold FlockLevel3.RowV2.enc at h
    rw [hl] at h
    exact FlockLevel3.RowV2.packBits_inj hl (List.append_cancel_left h)
  · rw [hl] at h
    exact FlockLevel3.RowV2.packBits_inj hl (List.append_cancel_left h)

/-- **`rowDep` places the value** at a circuit file with no segmented port. -/
theorem rowDep_places (C : FlockSoundness.Audit.Circuit) (rows : Fin C.N → Option RowAt) (c : Flock.Circuit)
    (hseg : c.ports.all (fun p => !p.segs.isSome) = true) (place : ℕ → (w : ℕ) × Fin w) (bits : Msg → ℕ → Bool) :
    RowPlaces (rowDep C rows c place bits) := by
  rintro ⟨i, v, d⟩ ⟨i', v', d'⟩ hi hv h
  simp only at hi hv
  subst hi
  unfold rowDep at h
  dsimp only at h
  cases hp : portOfPos C rows c i with
  | none =>
    simp only [hp] at h
    exact SamplingDeploy.row_places place bits ⟨i, v, d⟩ ⟨i, v', d'⟩ rfl hv h
  | some p =>
    simp only [hp] at h
    have hps : p.segs.isSome = false := by
      obtain ⟨r, -, hr⟩ := Option.bind_eq_some_iff.mp hp
      obtain ⟨hq, rfl⟩ := Array.getElem?_eq_some_iff.mp hr
      simpa using (Array.all_eq_true.mp hseg) r.port hq
    have hb := rowMsg_inj hps (by simp [SamplingDeploy.rowBits]) h
    simp only [SamplingDeploy.rowBits] at hb
    exact hv (by simpa using congrFun (List.ofFn_inj.1 hb) (place i).2)

/-- **`commitDep` binds, at every `H`**, at a circuit file with no segmented port: the hm96 leaf of a placing row binds
the value, and the registered path check the leaf. -/
theorem commitDep_binds (H : Msg → Msg) (C : FlockSoundness.Audit.Circuit) (rows : Fin C.N → Option RowAt)
    (c : Flock.Circuit) (hseg : c.ports.all (fun p => !p.segs.isSome) = true) (nl : ℕ) (roots : List Msg) :
    (commitDep C rows c nl roots).Binds H :=
  pathCommit_binds (hm96Leaf_binds H (rowDep_places C rows c hseg _ _) _) (regPath_binds H _)

/-- **A sampled verifier's code whose commitment is `commitDep`'s layout**: at a circuit file whose ports are all
unsegmented row/v2 rows (`RowsV2`), the rows `SamplingDeploy.commit` hashes. -/
def LayoutIs (C : FlockSoundness.Audit.Circuit) (rows : Fin C.N → Option RowAt) (c : Flock.Circuit) (nl : ℕ)
    (roots : List Msg) (cd : SamplingDeploy.Code) : Prop :=
  RowsV2 c ∧ cd.at_ = atOf C rows ∧ cd.place = placeOf C rows ∧ cd.bits = bitsOf (Flock.log2ceil nl) ∧
    cd.salt = saltOf ∧ cd.π = pathOf (Flock.log2ceil nl) ∧ cd.own = portsOf c nl roots

open Classical in
/-- **The evaluation's values as claims openings**: at each wire, its value in `C`'s evaluation on `xv`'s inputs, with
data its position passes `Cm` under `roots` at, where some does. -/
noncomputable def witDep (C : FlockSoundness.Audit.Circuit) (Cm : CommitmentScheme Bool) (H : Msg → Msg) (roots : List Msg)
    (xv : Fin C.N → Bool) : Wit :=
  ⟨(List.finRange C.N).map fun g =>
    if h : ∃ d : Msg, Cm.Opens H roots ⟨posOf C g, C.eval xv g, d⟩ = true then
      ⟨posOf C g, C.eval xv g, Classical.choose h⟩
    else ⟨posOf C g, C.eval xv g, []⟩⟩

open Classical in
/-- **Wire values as claims openings**: at each wire `g`, the value `v g`, with data its position passes `Cm` under
`roots` at, where some does. `witDep` is it at an evaluation (`witDep_eq_witRows`). -/
noncomputable def witRows (C : FlockSoundness.Audit.Circuit) (Cm : CommitmentScheme Bool) (H : Msg → Msg) (roots : List Msg)
    (v : Fin C.N → Bool) : Wit :=
  ⟨(List.finRange C.N).map fun g =>
    if h : ∃ d : Msg, Cm.Opens H roots ⟨posOf C g, v g, d⟩ = true then
      ⟨posOf C g, v g, Classical.choose h⟩
    else ⟨posOf C g, v g, []⟩⟩

theorem witDep_eq_witRows (C : FlockSoundness.Audit.Circuit) (Cm : CommitmentScheme Bool) (H : Msg → Msg)
    (roots : List Msg) (xv : Fin C.N → Bool) : witDep C Cm H roots xv = witRows C Cm H roots (C.eval xv) := rfl

end Flock.Bridge
