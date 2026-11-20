module

public import Properties.Flock.Circuit.OfProgram
public import Properties.Flock.Circuit.CircuitOf
public import Properties.Flock.Circuit.UnitInst
public import Properties.InferenceOnly
public import InferenceOnlyVerifier.Desc
public import Spot.Restrict
public import Spot.OneStageZkStatement
public import Spot.OneStageZk
public import Properties.Flock.EvalBool
public import Properties.Flock.DecodeBoolean
public import Definitions.Core.Rblk

/-!
# From an accepted templated statement to a correct proof block of C

The statements the chain from the sampled proof's guarantee to `Partition.Correct` in C (`ofProgram P`, its blocks
`partOf P blocks`) reads, each a named `Prop`, open until its proof (`theorem` of the same name, lower case, in the proofs
tree):

* `V5Rows`: a unit the sampled proof's guarantee makes correct in the key circuit of a templated statement satisfies, at
  each copy, the rows of the net its template unrolls to (Core's `Partition.Correct` read as `Rows.Sat`, through the key
  program's wiring).
* `LowerEval`: rows of the net the template `Flock.Lower.lower` writes unrolls to, satisfied with the constant 1, hold at
  each output port bit the value `Prog.evalBool` gives the Definition's lift to bits (`Flock.Lift.lift`) on the input port
  bits, an evaluation of exactly the output's width.
* `LowerEvalDoor`: `LowerEval` at the door: the template a block `Spot.OneStageZk.admitted` admits is lowered to
  (`Spot.OneStageZk.blockTmpl`), its rows satisfied with the constant 1, forces its output columns to `Prog.evalBool` of
  the template's lift on its input columns, each at the column the door's file lays it (`doorCol`).
* `BlockEvalDoor`: each instance of the admitted template, lifted to bits, evaluates on its arguments' values in block
  `u`'s local evaluation to that evaluation at its return leaves, an argument outside the block or inside it alike.
* `BlockCorrectDoor`: rows of every instance's copy, agreeing with a trace of C at the copy's arguments and return
  leaves, make block `u` correct, where no unit reads a gate of the block (`OperandsOutside`).
* `DoorJoin`: an accepted session's copies' rows (`V5Rows`'s conclusion, a copy a unit) and the statement's openings at
  each copy's arguments and return leaves (`StatementOpens`), with block `u`'s own (`Partition.Rblk`), give
  `BlockCorrectDoor`'s hypothesis, or SHA-512 collides.
* `RestrictBodyAgrees`: the block's Program keeps C's word primitives' Boolean Definitions (`BodyAgrees`).
* `PortsBind`: the trace a proof block's openings show (`Partition.Rblk`) and the statement's own openings at the same
  positions (`PortsOpenAt`) agree at the ports, so the rows agree with C's committed values there, or SHA-512 collides.
* `DescEqPartOf`: the inference-only check's three counts (gates per block, fresh input bits per block, all gates) are
  `partOf`'s, at the blocks of its owners and the gates of θ.
-/

@[expose] public section

namespace Flock.Chain

open Flock.Program Flock.OfProgram FlockSoundness

/-- The bits of C's values at `ports`, in order, each value low bit first; a port past C has `width` false bits. -/
def portBits {P : Prog} (X : (ofProgram P).Trace) (ports : Array Spot.BlockFront.Port) : List Bool :=
  ports.toList.flatMap fun p =>
    if h : p.gate < (ofProgram P).N then List.ofFn fun i : Fin ((ofProgram P).w ⟨p.gate, h⟩) => (X ⟨p.gate, h⟩).getLsbD i
    else List.replicate p.width false

/-- The total width of `ports`. -/
def portWidth (ports : Array Spot.BlockFront.Port) : ℕ := ports.foldl (· + ·.width) 0

/-- Block `u` of `partOf P blocks` holds exactly the computed gates the front's ranges hold. -/
def BlockIs (P : Prog) (blocks : Array (Array (ℕ × ℕ))) (u : Fin blocks.size) (fr : Spot.BlockFront.Front) : Prop :=
  ∀ g : Fin (ofProgram P).N,
    (partOf P blocks).unit g = some u.castSucc ↔ (Spot.BlockFront.inside fr.ranges g.val = true ∧ inputAt P g = false)

/-- **`V5Rows`, the adapter to the sampled proof's guarantee.** A unit `u` correct in the key circuit of the templated
statement `c` carrying the template `tu.t`: the statement's unit is the net `tu.t` unrolls to, and at every copy `u'` the
unit's local evaluation, read at copy `u'`'s columns wired as the key program wires the unit (`rowsWireC`, cast along
`rows_keyProg` as `unitInst` is), satisfies its rows with the constant 1. -/
def V5Rows : Prop :=
  ∀ (c : Flock.Circuit) (hU : Built (Flock.Desc.tmpl c).unroll.unit) (n : ℕ) (Y : (Flock.Bridge.circuitOf (.tmpl c) hU n).Trace)
    (u : Fin n), (Flock.Bridge.partsOf (.tmpl c) hU n).Correct Y u →
    ∀ (tu : TmplUnit) (nt : Net), c.tmpl = some tu → Tmpl.unroll tu.t = .ok nt →
    (Flock.Desc.tmpl c).unroll.unit = nt ∧
    ∀ u' (hu' : u' < (Flock.Desc.tmpl c).unroll.upv),
      let W := Discharge.Const.Prog.castW (Flock.Bridge.rows_keyProg (.tmpl c) hU n u)
        (Discharge.Const.Prog.rowsWireC (Discharge.Placed.keyProg (Flock.Desc.tmpl c).unroll hU n) u)
      let z : ℕ → Bool := fun x => if hx : x < (Rows.ofNet hU).w then
        ((Flock.Bridge.partsOf (.tmpl c) hU n).localEval Y u (W.wire ((Rows.ofNet hU).stackCol _ hu' ⟨x, hx⟩))).getLsbD 0
        else false
      (Rows.ofNet hU).Sat (fun x => z x.val) ∧ z (Flock.Desc.tmpl c).unroll.unit.constPos = true

/-- **`LowerEval`, the template `lower` writes of Definition `d` computes `d` on bits**: `lower` lowers `d`'s lift to bits,
Definition `e` of `Q` (`Flock.Lift.lift`). Wherever the rows of the net its template unrolls to hold with the constant 1,
`Prog.evalBool Q e` on the input port bits of `e`'s parameter leaves is exactly `e`'s output leaves, and output port bit
`k` is its bit `k`, so an empty evaluation never counts as agreement. `P`'s own Boolean map is §4.9's (`Prog.BoolMapOk`,
which `DecodeBoolean` gives every decoded Program) and its word primitives evaluate as their bodies (`BodyAgrees`). The
net's rows are `Rows.ofNet hb` at any proof `hb` that it is built, which every net `Tmpl.unroll` gives is; `Rows.ofNet`
reads `hb` only as a proof, so the rows are the same at every one. -/
def LowerEval : Prop :=
  ∀ {P : Program.Prog} {d : ℕ} {inP outP : Array ℕ} {t : Tmpl.Tmpl} {n : Net} {Q : Program.Prog} {e : ℕ}
    (_hm : P.BoolMapOk) (_hb : BodyAgrees P) (_hq : Flock.Lift.lift P d = .ok (Q, e))
    (_hl : Flock.Lower.lower P d inP outP = .ok t) (_h : Tmpl.unroll t = .ok n) (_hu : n.unitLog ≤ 32) (hb : Built n)
    {z : ℕ → Bool} (_hs : (Rows.ofNet hb).Sat fun c => z c.val) (_hone : z t.constPos = true),
    let v := Q.evalBool e ((List.range (Q.defs.getD e default).paramLeaves).map fun i => z (Flock.Lower.inBit t i))
    v.length = (Q.defs.getD e default).outLeaves ∧
      ∀ k < (Q.defs.getD e default).outLeaves, z (Flock.Lower.outBit t k) = v.getD k false

/-- The bits of `rs`, rows or fields `(name, bits)`. -/
def bitsOf (rs : Array (String × ℕ)) : ℕ := rs.foldl (· + ·.2) 0

/-- The port bit the door's file lays bit `i` of `rs` at: each row or field from a whole u16 word, in order. -/
def doorCol (rs : Array (String × ℕ)) (i : ℕ) : ℕ := go rs.toList i 0
where
  go : List (String × ℕ) → ℕ → ℕ → ℕ
    | [], i, base => base + i
    | (_, b) :: r, i, base => if i < b then base + i else go r (i - b) (base + 16 * ((b + 15) / 16))

/-- Unit `g`'s arguments: the gate of C each parameter leaf of its template reads (`Spot.BlockFront.operandsW`), in
the template's parameter order, a gate read twice listed twice, with the leaf's width; none where `g` is no unit. -/
def unitArgs (P : Prog) (g : ℕ) : Array Spot.BlockFront.Port :=
  match (do
    let (k, m) ← Spot.BlockFront.unitAt P g
    let some x := (P.defs.getD P.root default).body.nodes[k]? | throw "no root node"
    Spot.BlockFront.operandsW P x m : V (Array (ℕ × ℕ))) with
  | .ok ps => ps.map fun (q, w) => { gate := q, width := w }
  | .error _ => #[]

/-- Unit `g`'s return leaves: the gate of C each return leaf of its member resolves to (`resolve`), in order, with the
leaf's width; none where `g` is no unit. -/
def unitOuts (P : Prog) (g : ℕ) : Array Spot.BlockFront.Port :=
  match (do
    let (k, m) ← Spot.BlockFront.unitAt P g
    let some x := (P.defs.getD P.root default).body.nodes[k]? | throw "no root node"
    let c := P.defs.getD x.fn default
    (Array.range c.ret.leaves).mapM fun l => do
      let q ← resolve P P.rootScope { p := false, i := k, j := m * c.ret.leaves + l }
      pure ({ gate := q, width := c.ret.widths.getD l 0 } : Spot.BlockFront.Port) : V (Array Spot.BlockFront.Port)) with
  | .ok ps => ps
  | .error _ => #[]

/-- C's values at unit `g`'s arguments, in its template's parameter order, as bits: the trace at each argument's gate,
whether that gate is outside the block (a port of the front) or inside it (another unit's return leaf). -/
def argBits {P : Prog} (X : (ofProgram P).Trace) (g : ℕ) : List Bool := portBits X (unitArgs P g)

/-- C's values at unit `g`'s return leaves, in order, as bits. -/
def outBits {P : Prog} (X : (ofProgram P).Trace) (g : ℕ) : List Bool := portBits X (unitOuts P g)

/-- The bits `portBits` lays port `q` at: its gate's width in C, or its own where it names no gate of C. -/
def leafWidth (P : Prog) (q : Spot.BlockFront.Port) : ℕ :=
  if h : q.gate < (ofProgram P).N then (ofProgram P).w ⟨q.gate, h⟩ else q.width

/-- Where unit `g`'s return leaf `l` starts in `outBits`. -/
def outOff (P : Prog) (g l : ℕ) : ℕ := ((unitOuts P g).toList.take l).foldl (· + leafWidth P ·) 0

/-- Port `q` is a committed gate of the partition (`Partition.committed`): an input, an output of C, or read by a gate
of another unit. -/
def CommittedAt {P : Prog} {n : ℕ} (Pt : Verity.Partition (ofProgram P) n) (q : Spot.BlockFront.Port) : Prop :=
  ∃ h : q.gate < (ofProgram P).N, ⟨q.gate, h⟩ ∈ Pt.committed

/-- `bs`, the bits of a trace at unit `g`'s return leaves, is `z` at the door's output columns, at each leaf in
`Pt.committed` alone: a return leaf no other unit reads and no output of C is no committed gate, so nothing opens it
and `Partition.Correct` reads none of it. -/
def OutsAgree {P : Prog} {n : ℕ} (Pt : Verity.Partition (ofProgram P) n) (a : Spot.OneStageZk.Admit) (t : Tmpl.Tmpl)
    (g : ℕ) (bs : List Bool) (z : ℕ → Bool) : Prop :=
  ∀ l (hl : l < (unitOuts P g).size), CommittedAt Pt (unitOuts P g)[l] →
    ∀ b < leafWidth P (unitOuts P g)[l],
      bs.getD (outOff P g l + b) false = z (Flock.Lower.outBit t (doorCol a.fields (outOff P g l + b)))

/-- **`DoorLengths`**: at every unit of the block, the bits of a trace of C at its arguments are exactly the input
rows' bits, and at its return leaves exactly the return fields', so `getD`'s default never stands for a bit. Lengths
only: the order of the bits inside a word (`portBits`, low bit first) meets the lift's leaf order in `BlockEvalDoor`,
which evaluates the lift on these bits. -/
def DoorLengths : Prop :=
  ∀ (bytes : ByteArray) (P : Prog) (partition : Lean.Json) (cuts : List (String × List (String × ℕ)))
    (units : Array ℕ) (S : Prog) (a : Spot.OneStageZk.Admit) (fr : Spot.BlockFront.Front),
    decode bytes = .ok P → (P.defs[P.root]!).body.recut = #[] →
    Spot.Restrict.restrict P units = .ok S → Spot.OneStageZk.admitted partition cuts S = .ok a →
    Spot.BlockFront.front P units = .ok fr →
    ∀ (X : (ofProgram P).Trace), ∀ g ∈ units, (argBits X g).length = bitsOf a.ins ∧ (outBits X g).length = bitsOf a.fields

/-- **`LowerEvalDoor`, `LowerEval` at the door.** For the block's Program `restrict P units` of a decoded C with no root
recut, which `Spot.OneStageZk.admitted` admits, rows of the net its template (`Spot.OneStageZk.blockTmpl`) unrolls to, satisfied with the constant 1, force each output column the door's
file lays (`doorCol` of the return fields) to the bit `Prog.evalBool` gives the template's lift to bits on the input
columns (`doorCol` of the input rows), an evaluation of exactly the fields' bits. It states that satisfied rows force
the outputs, not that every evaluation satisfies them. -/
def LowerEvalDoor : Prop :=
  ∀ {bytes : ByteArray} {P : Prog} {units : Array ℕ} {partition : Lean.Json} {cuts : List (String × List (String × ℕ))}
    {S : Prog} {a : Spot.OneStageZk.Admit} {t : Tmpl.Tmpl} {n : Net} {Q : Prog} {e : ℕ},
    decode bytes = .ok P → (P.defs[P.root]!).body.recut = #[] → Spot.Restrict.restrict P units = .ok S →
    Spot.OneStageZk.admitted partition cuts S = .ok a → Spot.OneStageZk.blockTmpl S a = .ok t →
    Flock.Lift.lift S a.d = .ok (Q, e) → Tmpl.unroll t = .ok n → n.unitLog ≤ 32 → ∀ (hb : Built n) {z : ℕ → Bool},
    (Rows.ofNet hb).Sat (fun c => z c.val) → z t.constPos = true →
    let v := Q.evalBool e ((List.range (bitsOf a.ins)).map fun i => z (Flock.Lower.inBit t (doorCol a.ins i)))
    v.length = bitsOf a.fields ∧
      ∀ k < bitsOf a.fields, z (Flock.Lower.outBit t (doorCol a.fields k)) = v.getD k false

/-- **`RestrictBodyAgrees`**: the block's Program `restrict P units` keeps `P`'s word primitives evaluating as their
Boolean Definitions (`BodyAgrees`). With `decode_body_agrees`, which gives `BodyAgrees P` of every Program `decode`
accepts whose root has no recut, it is what the Door statements read for `BodyAgrees`. A root with a recut is refused
where C is read through `Spot.Program.parse` (`recut-root`, the per-member run's `membersRootsOf`); the restrict path
decodes C with `Flock.Program.decode`, which keeps it. -/
def RestrictBodyAgrees : Prop :=
  ∀ (P : Prog) (units : Array ℕ) (S : Prog), BodyAgrees P → Spot.Restrict.restrict P units = .ok S → BodyAgrees S

/-- **`BlockEvalDoor`**: in block `u`'s local evaluation `Y` of a trace of C, each unit `g` of the block, an instance of
the template `Spot.OneStageZk.admitted` admits, lifted to bits (`Flock.Lift.lift`), evaluates on `Y` at its arguments to
`Y` at its return leaves, bit for bit. An argument inside the block reads `Y` at another unit's return leaf. -/
def BlockEvalDoor : Prop :=
  ∀ (bytes : ByteArray) (P : Prog) (partition : Lean.Json) (cuts : List (String × List (String × ℕ)))
    (units : Array ℕ) (S : Prog) (a : Spot.OneStageZk.Admit) (fr : Spot.BlockFront.Front) (L : Prog) (e : ℕ),
    decode bytes = .ok P → (P.defs[P.root]!).body.recut = #[] →
    Spot.Restrict.restrict P units = .ok S → Spot.OneStageZk.admitted partition cuts S = .ok a →
    Spot.BlockFront.front P units = .ok fr → Flock.Lift.lift S a.d = .ok (L, e) →
    ∀ (blocks : Array (Array (ℕ × ℕ))) (u : Fin blocks.size), BlockIs P blocks u fr →
    ∀ X : (ofProgram P).Trace, ∀ g ∈ units,
      let Y := (partOf P blocks).localEval X u.castSucc
      Prog.evalBool L e (argBits Y g) = outBits Y g

/-- **`OperandsOutside`**, the second check's clause (a), until the door's constant for it has a name: every argument of
every unit of the block is a gate outside it, so each is a port of the front (`Spot.BlockFront.Front.ins`). -/
def OperandsOutside (P : Prog) (units : Array ℕ) (fr : Spot.BlockFront.Front) : Prop :=
  ∀ g ∈ units, ∀ q ∈ unitArgs P g, Spot.BlockFront.inside fr.ranges q.gate = false

/-- **`BlockCorrectDoor`, the block is correct.** For a block `Spot.OneStageZk.admitted` admits: where, for every unit
`g` of the block, rows of the net its template unrolls to hold at some `z` with the constant 1, and `z` at the door's
input columns is a trace `X` of C at `g`'s arguments and its output columns `X` at `g`'s committed return leaves
(`OutsAgree`), block `u` is correct in C (`Partition.Correct` reads committed gates alone). Its units
read no gate of the block (`OperandsOutside`), so every input column is C's value at a port of the front. -/
def BlockCorrectDoor : Prop :=
  ∀ (bytes : ByteArray) (P : Prog) (partition : Lean.Json) (cuts : List (String × List (String × ℕ)))
    (units : Array ℕ) (S : Prog) (a : Spot.OneStageZk.Admit) (fr : Spot.BlockFront.Front) (t : Tmpl.Tmpl) (nt : Net),
    decode bytes = .ok P → (P.defs[P.root]!).body.recut = #[] →
    Spot.Restrict.restrict P units = .ok S → Spot.OneStageZk.admitted partition cuts S = .ok a →
    Spot.BlockFront.front P units = .ok fr → OperandsOutside P units fr →
    Spot.OneStageZk.blockTmpl S a = .ok t → Tmpl.unroll t = .ok nt → nt.unitLog ≤ 32 →
    ∀ (blocks : Array (Array (ℕ × ℕ))) (u : Fin blocks.size), BlockIs P blocks u fr →
    ∀ X : (ofProgram P).Trace,
      (∀ g ∈ units, ∃ (hb : Built nt) (z : ℕ → Bool), (Rows.ofNet hb).Sat (fun c => z c.val) ∧ z t.constPos = true ∧
        (∀ i < bitsOf a.ins, (argBits X g).getD i false = z (Flock.Lower.inBit t (doorCol a.ins i))) ∧
        OutsAgree (partOf P blocks) a t g (outBits X g) z) →
      (partOf P blocks).Correct X u.castSucc

/-- **The statement's openings at each copy**: copy `j` of the session is unit `units[j]`; `z j`'s input columns are a
trace `Y` of C at that unit's arguments and its output columns `Y` at its committed return leaves (`OutsAgree`), and
`π' j` opens each argument and each committed return leaf at its position (`Partition.pos`) to `Y`'s value there. A
gate off the boundary has no position: an argument read only inside its block (so `OperandsOutside`), or a return leaf
no other unit reads and no output of C, which is not opened. -/
def StatementOpens {P : Prog} {n : ℕ} {α : Type} (Sc : _root_.Hash.StringCommitment α)
    (H : _root_.Hash.Msg → Verity.Digest) (enc : (g : Fin (ofProgram P).N) → BitVec ((ofProgram P).w g) → α)
    (Pt : Verity.Partition (ofProgram P) n) (L : Verity.Layout (ofProgram P)) (c : Verity.Root → Sc.Com)
    (units : Array ℕ) (a : Spot.OneStageZk.Admit) (t : Tmpl.Tmpl) (z : ℕ → ℕ → Bool)
    (π' : ℕ → Fin (ofProgram P).N → Sc.Op) : Prop :=
  ∃ Y : (ofProgram P).Trace, ∀ j < units.size,
    (∀ i < bitsOf a.ins, (argBits Y units[j]!).getD i false = z j (Flock.Lower.inBit t (doorCol a.ins i))) ∧
    OutsAgree Pt a t units[j]! (outBits Y units[j]!) (z j) ∧
    (∀ q ∈ unitArgs P units[j]!, ∀ h : q.gate < (ofProgram P).N,
      ∃ p, Pt.pos L ⟨q.gate, h⟩ = some p ∧ Sc.Opens H (c p.1) p.2 (enc _ (Y ⟨q.gate, h⟩)) (π' j ⟨q.gate, h⟩)) ∧
    ∀ q ∈ unitOuts P units[j]!, ∀ h : q.gate < (ofProgram P).N, ⟨q.gate, h⟩ ∈ Pt.committed →
      ∃ p, Pt.pos L ⟨q.gate, h⟩ = some p ∧ Sc.Opens H (c p.1) p.2 (enc _ (Y ⟨q.gate, h⟩)) (π' j ⟨q.gate, h⟩)

/-- **`DoorJoin`, an accepted session to the block's rows at C's committed values.** The session's copies, one a unit
of the block, each satisfying its rows with the constant 1 (`V5Rows`'s conclusion), the statement's openings at each
copy's arguments and return leaves (`StatementOpens`) and block `u`'s own (`Partition.Rblk`), at one commitment, give
`BlockCorrectDoor`'s hypothesis at `X`, or two messages the openings hashed collide. Its units read no gate of the block
(`OperandsOutside`), so each copy's input columns are opened at the front's ports, gates of in(S) with a position; an
argument inside the block would have none (`Partition.pos` is `none` off `Partition.boundary`). The net is at most
`2^32` rows: the verifier refuses a templated unit whose root has more columns (`Flock.ProvedScope.scopeOkT`), and
`Tmpl.unroll` lays the net's `unitLog` from them. -/
def DoorJoin : Prop :=
  ∀ (bytes : ByteArray) (P : Prog) (partition : Lean.Json) (cuts : List (String × List (String × ℕ)))
    (units : Array ℕ) (S : Prog) (a : Spot.OneStageZk.Admit) (fr : Spot.BlockFront.Front) (t : Tmpl.Tmpl) (nt : Net)
    {α : Type} (Sc : _root_.Hash.StringCommitment α) (H : _root_.Hash.Msg → Verity.Digest)
    (enc : (g : Fin (ofProgram P).N) → BitVec ((ofProgram P).w g) → α) (blocks : Array (Array (ℕ × ℕ)))
    (u : Fin blocks.size) (L : Verity.Layout (ofProgram P)) (c : Verity.Root → Sc.Com) (X : (ofProgram P).Trace)
    (π : Fin (ofProgram P).N → Sc.Op) (z : ℕ → ℕ → Bool) (π' : ℕ → Fin (ofProgram P).N → Sc.Op),
    decode bytes = .ok P → (P.defs[P.root]!).body.recut = #[] →
    Spot.Restrict.restrict P units = .ok S → Spot.OneStageZk.admitted partition cuts S = .ok a →
    Spot.BlockFront.front P units = .ok fr → OperandsOutside P units fr →
    Spot.OneStageZk.blockTmpl S a = .ok t → Tmpl.unroll t = .ok nt → nt.unitLog ≤ 32 →
    BlockIs P blocks u fr → (∀ g, Function.Injective (enc g)) → Sc.Binds H →
    (partOf P blocks).Rblk Sc H enc L c u.castSucc X π →
    (∀ j < units.size, ∃ hb : Built nt, (Rows.ofNet hb).Sat (fun x => z j x.val) ∧ z j t.constPos = true) →
    StatementOpens Sc H enc (partOf P blocks) L c units a t z π' →
    (∀ g ∈ units, ∃ (hb : Built nt) (z' : ℕ → Bool), (Rows.ofNet hb).Sat (fun x => z' x.val) ∧ z' t.constPos = true ∧
        (∀ i < bitsOf a.ins, (argBits X g).getD i false = z' (Flock.Lower.inBit t (doorCol a.ins i))) ∧
        OutsAgree (partOf P blocks) a t g (outBits X g) z') ∨
      ∃ xs, _root_.Hash.CollisionIn H xs

/-- **`AcceptGivesDoorFile`, an accept to the door's file.** Where the per-member check accepts (`checkMembersAt`), every
member's statement is read at its slice of the registration's roots, and the circuit its session was verified against
is the file `lowerBlock` writes for its block's Program `restrict P units` (`Own.circuitFile`), admitted, whose parse
carries the template `blockTmpl` gives, unrolling to a net of at most `2^32` rows (`Flock.ProvedScope.scopeOkT`). -/
def AcceptGivesDoorFile : Prop :=
  ∀ (p : Spot.OneStageZk.Params) (ms : Array Spot.OneStageZk.Member) (c : Spot.OneStage.Commitment) (ξ : ByteArray)
    (e : Spot.OneStageZk.Evidence) (sts : Array Spot.OneStageZk.StatementFiles) (bs : Core.Blobs)
    (pub : Spot.OneStage.Public) (prof : Core.Integrity),
    Spot.OneStageZk.checkMembersAt p ms c ξ e sts bs = .accept pub prof →
    ∃ roots, Spot.OneStageZk.membersRootsOf p ms c.record = .ok roots ∧
    ∀ m ∈ ms, ∀ (P : Prog) (bl : Array (ℕ × ℕ)) (lo hi : ℕ) (cuts : List (String × List (String × ℕ))),
      m.own.prog.get = .ok P → m.own.blocks.get = .ok (some bl) → bl.find? (·.1 == m.own.first) = some (lo, hi) →
      m.own.cuts.get = .ok cuts →
      ∃ (st : Spot.OneStageZk.StepAt) (S : Prog) (a : Spot.OneStageZk.Admit) (t : Tmpl.Tmpl) (file : ByteArray),
        m.statement (m.slice c.record roots) bs = .ok st ∧
        Spot.Restrict.restrict P (List.range' lo (hi - lo)).toArray = .ok S ∧
        Spot.OneStageZk.admitted m.own.partition cuts S = .ok a ∧ Spot.OneStageZk.blockTmpl S a = .ok t ∧
        Spot.OneStageZk.lowerBlock m.own.programJ m.own.partition cuts S = .ok file ∧ st.I.circuitFile = file ∧
        ∀ cc, Flock.ProvedScope.circuitOf (Flock.Witness.zkTags st.I) st.I.circuitFile st.I.tables = .ok cc →
          ∃ tu nt, cc.tmpl = some tu ∧ tu.t = t ∧ Tmpl.unroll t = .ok nt ∧ nt.unitLog ≤ 32

/-- **The statement's own openings at the ports**: `z`'s port bits are the bits of a trace `Y` of C, and `π'` opens
each port gate at its position (`Partition.pos`) to `Y`'s value there. -/
def PortsOpenAt {P : Prog} {n : ℕ} {α : Type} (Sc : _root_.Hash.StringCommitment α) (H : _root_.Hash.Msg → Verity.Digest)
    (enc : (g : Fin (ofProgram P).N) → BitVec ((ofProgram P).w g) → α) (Pt : Verity.Partition (ofProgram P) n)
    (L : Verity.Layout (ofProgram P)) (c : Verity.Root → Sc.Com) (fr : Spot.BlockFront.Front) (t : Tmpl.Tmpl)
    (z : ℕ → Bool) (π' : Fin (ofProgram P).N → Sc.Op) : Prop :=
  ∃ Y : (ofProgram P).Trace,
    (∀ i < portWidth fr.ins, (portBits Y fr.ins).getD i false = z (Flock.Lower.inBit t i)) ∧
    (∀ k < portWidth fr.outs, (portBits Y fr.outs).getD k false = z (Flock.Lower.outBit t k)) ∧
    ∀ q ∈ fr.ins ++ fr.outs, ∀ h : q.gate < (ofProgram P).N,
      ∃ p, Pt.pos L ⟨q.gate, h⟩ = some p ∧ Sc.Opens H (c p.1) p.2 (enc _ (Y ⟨q.gate, h⟩)) (π' ⟨q.gate, h⟩)

/-- **`PortsBind`, step 5: C's committed port values are the statement's port columns.** Block `u`'s openings
(`Partition.Rblk`) and the statement's own (`PortsOpenAt`), at one commitment, give the rows' agreement at the ports,
or two messages the openings hashed collide. The ports are C's gates and among the block's committed inputs and outputs
(`Partition.io`), which the front's ports are when every leaf is read. -/
def PortsBind : Prop :=
  ∀ (P : Prog) (units : Array ℕ) (fr : Spot.BlockFront.Front) (t : Tmpl.Tmpl) {α : Type}
    (Sc : _root_.Hash.StringCommitment α) (H : _root_.Hash.Msg → Verity.Digest)
    (enc : (g : Fin (ofProgram P).N) → BitVec ((ofProgram P).w g) → α) (blocks : Array (Array (ℕ × ℕ)))
    (u : Fin blocks.size) (L : Verity.Layout (ofProgram P)) (c : Verity.Root → Sc.Com) (X : (ofProgram P).Trace)
    (π π' : Fin (ofProgram P).N → Sc.Op) (z : ℕ → Bool),
    Spot.BlockFront.front P units = .ok fr → BlockIs P blocks u fr → (∀ g, Function.Injective (enc g)) →
    (∀ q ∈ fr.ins ++ fr.outs, ∃ h : q.gate < (ofProgram P).N, ⟨q.gate, h⟩ ∈ (partOf P blocks).io u.castSucc) →
    Sc.Binds H → (partOf P blocks).Rblk Sc H enc L c u.castSucc X π →
    PortsOpenAt Sc H enc (partOf P blocks) L c fr t z π' →
    ((∀ i < portWidth fr.ins, (portBits X fr.ins).getD i false = z (Flock.Lower.inBit t i)) ∧
      (∀ k < portWidth fr.outs, (portBits X fr.outs).getD k false = z (Flock.Lower.outBit t k))) ∨
      ∃ xs, _root_.Hash.CollisionIn H xs

/-- The blocks of the owners `own` (one a root node): block `u`'s ranges are its root nodes' gates. -/
def ownerRanges (P : Prog) (own : Array (Option ℕ)) : Array (Array (ℕ × ℕ)) :=
  let offs := (P.defs[P.root]!).body.offsets
  (Array.range (InferenceOnlyVerifier.iuCount own)).map fun u =>
    ((Array.range own.size).filter fun k => own[k]! == some u).map fun k => (offs[k]!, offs[k + 1]!)

/-- The gates of θ's root nodes. -/
def thetaGates (P : Prog) (theta : Array ℕ) : Finset (Fin (ofProgram P).N) :=
  Finset.univ.filter fun g => (rootNode P g).any theta.contains

/-- The owners by root node of the instance's `owner`: a root node holding gates names its block, else none. -/
def ownersOf (P : Prog) (owner : Array Lean.Json) : Array (Option ℕ) :=
  let offs := (P.defs[P.root]!).body.offsets
  owner.mapIdx fun k e => if offs[k]! < offs[k + 1]! then InferenceOnlyVerifier.natOf? e else none

/-- **`DescEqPartOf`: the inference-only check's three counts are `partOf`'s.** On an instance `InferenceOnlyCheck.Desc`
gives a verdict, with its Program `P`: its `gates` is C's gates, and block `u`'s pair is the gates of block `u` of
`partOf P (ownerRanges P (ownersOf P i.owner))` and the width of their in(S) outside θ's gates. -/
def DescEqPartOf : Prop :=
  ∀ (j out : Lean.Json) (i : InferenceOnlyVerifier.Instance) (P : Prog),
    InferenceOnlyVerifier.Desc.evaluate j = .ok out → InferenceOnlyVerifier.Properties.Parses j i →
    Flock.Program.decode i.program.toUTF8 = .ok P →
    let blocks := ownerRanges P (ownersOf P i.owner)
    let θ := thetaGates P (i.theta.filterMap InferenceOnlyVerifier.natOf?)
    out.getObjValD "gates" = Lean.toJson (ofProgram P).N ∧
      out.getObjValD "ius" = Lean.Json.arr (Array.ofFn fun u : Fin blocks.size =>
        let S := (partOf P blocks).gates u.castSucc
        Lean.Json.arr #[Lean.toJson S.card, Lean.toJson ((ofProgram P).width ((ofProgram P).into S \ θ))])

end Flock.Chain
