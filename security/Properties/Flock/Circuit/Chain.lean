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
  leaves, make block `u` correct, where no unit reads a gate of the block (`Spot.OneStageZk.admittedBlock`).
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
`k` is its bit `k`, so an empty evaluation never counts as agreement. The net's rows are `Rows.ofNet hb` at any proof
`hb` that it is built, which every net `Tmpl.unroll` gives is; `Rows.ofNet` reads `hb` only as a proof, so the rows are
the same at every one. -/
def LowerEval : Prop :=
  ∀ {P : Program.Prog} {d : ℕ} {inP outP : Array ℕ} {t : Tmpl.Tmpl} {n : Net} {Q : Program.Prog} {e : ℕ}
    (_hq : Flock.Lift.lift P d = .ok (Q, e))
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

/-- **Unit `g` resolves**: it is a member of a root node of C, its operands are located (`Spot.BlockFront.operandsW`
returns), and each of its return leaves resolves to a gate (`resolve`). On an accept its operands come from
`admittedBlock`, which runs `operandsW` on every unit of the block; no check of the accept resolves every return leaf
(`front` resolves only those read outside the block). -/
def UnitResolves (P : Prog) (g : ℕ) : Prop :=
  ∃ k m x, Spot.BlockFront.unitAt P g = .ok (k, m) ∧ (P.defs.getD P.root default).body.nodes[k]? = some x ∧
    (∃ ops, Spot.BlockFront.operandsW P x m = .ok ops) ∧
    ∀ l < (P.defs.getD x.fn default).ret.leaves, ∃ q,
      resolve P P.rootScope { p := false, i := k, j := m * (P.defs.getD x.fn default).ret.leaves + l } = .ok q

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
rows' bits, and at its return leaves exactly the return fields', the block's units admitted in C (`admittedBlock`), so `getD`'s default never stands for a bit. Lengths
only: the order of the bits inside a word (`portBits`, low bit first) meets the lift's leaf order in `BlockEvalDoor`,
which evaluates the lift on these bits. Every gate of C is located (`GateOk`), which `GatesLocated` gives on a Program
inference-only's checks accept (`Desc.walkable`, `Desc.analyze`) and which SPOT's accept checks nowhere; and every unit
of the block resolves (`UnitResolves`). -/
def DoorLengths : Prop :=
  ∀ (bytes : ByteArray) (P : Prog) (partition : Lean.Json) (cuts : List (String × List (String × ℕ)))
    (units : Array ℕ) (S : Prog) (a : Spot.OneStageZk.Admit) (fr : Spot.BlockFront.Front),
    decode bytes = .ok P → (P.defs[P.root]!).body.recut = #[] →
    Spot.Restrict.restrict P units = .ok S → Spot.OneStageZk.admitted partition cuts S = .ok a →
    Spot.OneStageZk.admittedBlock P units a.key = .ok () → Spot.BlockFront.front P units = .ok fr →
    units.toList.Pairwise (· < ·) →
    (∀ h < InferenceOnlyVerifier.Properties.gates P, InferenceOnlyVerifier.Properties.GateOk P h) →
    (∀ g ∈ units, UnitResolves P g) →
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

/-- **`RestrictBodyAgrees`**: the block's Program `restrict P units` of a C `decode` accepts whose root has no recut
keeps C's word primitives evaluating as their Boolean Definitions (`BodyAgrees`). It reads `decode` and not
`BodyAgrees P`: `restrict` adds keys to the index (the root's, an input primitive's), so a Boolean-map entry of a Program
whose target its index lacks would lead somewhere in the block's. A root with a recut is refused where C is read through
`Spot.Program.parse` (`recut-root`, the per-member run's `membersRootsOf`); the restrict path decodes C with
`Flock.Program.decode`, which keeps it. -/
def RestrictBodyAgrees : Prop :=
  ∀ (bytes : ByteArray) (P : Prog) (units : Array ℕ) (S : Prog), decode bytes = .ok P →
    (P.defs[P.root]!).body.recut = #[] → Spot.Restrict.restrict P units = .ok S → BodyAgrees S

/-- **`BlockEvalDoor`**: in block `u`'s local evaluation `Y` of a trace of C, each unit `g` of the block, an instance of
the template `Spot.OneStageZk.admitted` admits, the block's units admitted in C (`admittedBlock`), lifted to bits
(`Flock.Lift.lift`) and lowered (`Spot.OneStageZk.blockTmpl`, which refuses a primitive that is no §9 gate where the
lift reaches it), evaluates on `Y` at its arguments to `Y` at its return leaves, bit for bit. An argument inside
the block reads `Y` at another unit's return leaf. Every gate of C is located (`GateOk`), which `GatesLocated` gives on a Program
inference-only's checks accept (`Desc.walkable`, `Desc.analyze`) and which SPOT's accept checks nowhere; and every unit
of the block resolves (`UnitResolves`). -/
def BlockEvalDoor : Prop :=
  ∀ (bytes : ByteArray) (P : Prog) (partition : Lean.Json) (cuts : List (String × List (String × ℕ)))
    (units : Array ℕ) (S : Prog) (a : Spot.OneStageZk.Admit) (fr : Spot.BlockFront.Front) (L : Prog) (e : ℕ)
    (t : Tmpl.Tmpl),
    decode bytes = .ok P → (P.defs[P.root]!).body.recut = #[] →
    Spot.Restrict.restrict P units = .ok S → Spot.OneStageZk.admitted partition cuts S = .ok a →
    Spot.OneStageZk.admittedBlock P units a.key = .ok () →
    Spot.BlockFront.front P units = .ok fr →
    units.toList.Pairwise (· < ·) → Flock.Lift.lift S a.d = .ok (L, e) → Spot.OneStageZk.blockTmpl S a = .ok t →
    (∀ h < InferenceOnlyVerifier.Properties.gates P, InferenceOnlyVerifier.Properties.GateOk P h) →
    (∀ g ∈ units, UnitResolves P g) →
    ∀ (blocks : Array (Array (ℕ × ℕ))) (u : Fin blocks.size), BlockIs P blocks u fr →
    ∀ X : (ofProgram P).Trace, ∀ g ∈ units,
      let Y := (partOf P blocks).localEval X u.castSucc
      Prog.evalBool L e (argBits Y g) = outBits Y g

/-- **`BlockCorrectDoor`, the block is correct.** For a block `Spot.OneStageZk.admitted` admits: where, for every unit
`g` of the block, rows of the net its template unrolls to hold at some `z` with the constant 1, and `z` at the door's
input columns is a trace `X` of C at `g`'s arguments and its output columns `X` at `g`'s committed return leaves
(`OutsAgree`), block `u` is correct in C (`Partition.Correct` reads committed gates alone). Its units
read no gate of the block (`Spot.OneStageZk.admittedBlock`'s `operands-outside`), so every input column is C's value
at a port of the front. -/
def BlockCorrectDoor : Prop :=
  ∀ (bytes : ByteArray) (P : Prog) (partition : Lean.Json) (cuts : List (String × List (String × ℕ)))
    (units : Array ℕ) (S : Prog) (a : Spot.OneStageZk.Admit) (fr : Spot.BlockFront.Front) (t : Tmpl.Tmpl) (nt : Net),
    decode bytes = .ok P → (P.defs[P.root]!).body.recut = #[] →
    Spot.Restrict.restrict P units = .ok S → Spot.OneStageZk.admitted partition cuts S = .ok a →
    Spot.BlockFront.front P units = .ok fr →
    units.toList.Pairwise (· < ·) → Spot.OneStageZk.admittedBlock P units a.key = .ok () →
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
gate off the boundary has no position: an argument read only inside its block (which `admittedBlock` refuses), or a return leaf
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

/-- **THE layout the served path reads** (`layoutOf`): each registered gate at its registered entry, every other gate
in phase 0's window. `DoorJoin`, `PortsBind` and `FifthOpens` read it. -/
abbrev servedLayout (P : Prog) : Verity.Layout (ofProgram P) := layoutOf P

/-- **`DoorJoin`, an accepted session to the block's rows at C's committed values.** The session's copies, one a unit
of the block, each satisfying its rows with the constant 1 (`V5Rows`'s conclusion), the statement's openings at each
copy's arguments and return leaves (`StatementOpens`) and block `u`'s own (`Partition.Rblk`), at one commitment, give
`BlockCorrectDoor`'s hypothesis at `X`, or two messages the openings hashed collide. Its units read no gate of the block
(`admittedBlock`'s `operands-outside`), so each copy's input columns are opened at the front's ports, gates of in(S) with a position; an
argument inside the block would have none (`Partition.pos` is `none` off `Partition.boundary`). The net is at most
`2^32` rows: the verifier refuses a templated unit whose root has more columns (`Flock.ProvedScope.scopeOkT`), and
`Tmpl.unroll` lays the net's `unitLog` from them. -/
def DoorJoin : Prop :=
  ∀ (bytes : ByteArray) (P : Prog) (partition : Lean.Json) (cuts : List (String × List (String × ℕ)))
    (units : Array ℕ) (S : Prog) (a : Spot.OneStageZk.Admit) (fr : Spot.BlockFront.Front) (t : Tmpl.Tmpl) (nt : Net)
    {α : Type} (Sc : _root_.Hash.StringCommitment α) (H : _root_.Hash.Msg → Verity.Digest)
    (enc : (g : Fin (ofProgram P).N) → BitVec ((ofProgram P).w g) → α) (blocks : Array (Array (ℕ × ℕ)))
    (u : Fin blocks.size) (c : Verity.Root → Sc.Com) (X : (ofProgram P).Trace)
    (π : Fin (ofProgram P).N → Sc.Op) (z : ℕ → ℕ → Bool) (π' : ℕ → Fin (ofProgram P).N → Sc.Op),
    decode bytes = .ok P → (P.defs[P.root]!).body.recut = #[] →
    Spot.Restrict.restrict P units = .ok S → Spot.OneStageZk.admitted partition cuts S = .ok a →
    Spot.BlockFront.front P units = .ok fr →
    units.toList.Pairwise (· < ·) → Spot.OneStageZk.admittedBlock P units a.key = .ok () →
    Spot.OneStageZk.blockTmpl S a = .ok t → Tmpl.unroll t = .ok nt → nt.unitLog ≤ 32 →
    BlockIs P blocks u fr → (∀ g, Function.Injective (enc g)) → Sc.Binds H →
    (partOf P blocks).Rblk Sc H enc (servedLayout P) c u.castSucc X π →
    (∀ j < units.size, ∃ hb : Built nt, (Rows.ofNet hb).Sat (fun x => z j x.val) ∧ z j t.constPos = true) →
    StatementOpens Sc H enc (partOf P blocks) (servedLayout P) c units a t z π' →
    (∀ g ∈ units, ∃ (hb : Built nt) (z' : ℕ → Bool), (Rows.ofNet hb).Sat (fun x => z' x.val) ∧ z' t.constPos = true ∧
        (∀ i < bitsOf a.ins, (argBits X g).getD i false = z' (Flock.Lower.inBit t (doorCol a.ins i))) ∧
        OutsAgree (partOf P blocks) a t g (outBits X g) z') ∨
      ∃ xs, _root_.Hash.CollisionIn H xs

/-- **`AcceptGivesDoorFile`, an accept to the door's file.** Where the per-member check accepts (`checkMembersAt`), every
member's statement is read at its slice of the registration's roots, and the circuit its session was verified against
is the file `lowerBlock` writes for its block's Program `restrict P units` (`Own.circuitFile`), admitted, its units
admitted in C (`admittedBlock`), whose parse
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
        Spot.OneStageZk.admittedBlock P (List.range' lo (hi - lo)).toArray a.key = .ok () ∧
        Spot.OneStageZk.lowerBlock m.own.programJ m.own.partition cuts S
          (Spot.OneStageZk.admittedBlock P (List.range' lo (hi - lo)).toArray) = .ok file ∧ st.I.circuitFile = file ∧
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
    (u : Fin blocks.size) (c : Verity.Root → Sc.Com) (X : (ofProgram P).Trace)
    (π π' : Fin (ofProgram P).N → Sc.Op) (z : ℕ → Bool),
    Spot.BlockFront.front P units = .ok fr →
    units.toList.Pairwise (· < ·) → BlockIs P blocks u fr → (∀ g, Function.Injective (enc g)) →
    (∀ q ∈ fr.ins ++ fr.outs, ∃ h : q.gate < (ofProgram P).N, ⟨q.gate, h⟩ ∈ (partOf P blocks).io u.castSucc) →
    Sc.Binds H → (partOf P blocks).Rblk Sc H enc (servedLayout P) c u.castSucc X π →
    PortsOpenAt Sc H enc (partOf P blocks) (servedLayout P) c fr t z π' →
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

/-! ### The map from Core's positions to the verifier's trees

One template a block and one copy a unit: block `u` is a member `m`'s units, `m.indices`
(`Spot.OneStageZk.Member.indices`), and copy `j` of its session is unit `m.indices[j]`. -/

/-- A leaf of a tree the verifier holds a root of: the tree's name (its record port), the leaf, and the gate's index in
the leaf's row. -/
structure TreeLeaf where
  tree : String
  leaf : ℕ
  slot : ℕ
  deriving DecidableEq

/-- The return field holding return leaf `l` of the fields `fs` (`fieldsOf`), and the leaf's index in it. -/
def fieldSlot : List Ty → ℕ → ℕ × ℕ
  | [], l => (0, l)
  | f :: fs, l => if l < f.leaves then (0, l) else ((fieldSlot fs (l - f.leaves)).1 + 1, (fieldSlot fs (l - f.leaves)).2)

/-- **Where the verifier opens unit `v`'s return leaf `l`, as member `m`'s output**, at the statement's Circuit `cc`: the
verifier's opening of a unit's output (`Flock.Registered.outReadsAt`) of output row `j`, the field of the circuit's
template (`Flock.HmRow.queryTemplate`) holding the leaf (`fieldSlot`), at the port of `m`'s registered file that
`Flock.Registered.withPositions` dispatches to output row `j` (`Flock.Registered.outRow?` at the circuit's first port
of its name): the tree is that port's value, which `outReadsAt` accepts only as the field's tree
(`Flock.Registered.fieldTree`), the leaf the one it gives `v`, the slot the leaf's index in the field. -/
def outTree (P : Prog) (m : Spot.OneStageZk.Member) (cc : Flock.Circuit) (v l : ℕ) : Option TreeLeaf := do
  let (t, _) ← Flock.HmRow.queryTemplate cc
  let f ← (Flock.Registered.rootNodes P).findSome? fun x => (P.defs[x.fn]?).filter (·.key == t)
  let (j, s) := fieldSlot (fieldsOf f.ret) l
  let some (.ok own) := m.own.registered.map Flock.Registered.ofJson | none
  let reg ← own.find? fun reg =>
    ((List.range cc.ports.size).find? fun q => (cc.ports.getD q default).name == reg.name).bind
      (Flock.Registered.outRow? cc) == some j
  let (rel, first) ← (Flock.Registered.outReadsAt cc P m.indices j reg).toOption
  let leaf ← (rel[v - first]?).join
  pure ⟨reg.value, leaf, s⟩

/-- How a member's statement reads its row `row`: a port of its registered file (`Flock.Registered.ofJson`), its reads
laid by `Flock.OneCircuit.pos`, or a root of its own in the record (`<member>/<row>`, `Spot.OneStageZk.rootStep`). -/
inductive PortRead where
  | registered (q : Flock.Registered.Port)
  | own (row : String)

/-- The read of member `m`'s row `row` over C (`P`). -/
def readsOf (P : Prog) (m : Spot.OneStageZk.Member) (row : String) : Option PortRead :=
  match m.own.registered.map Flock.Registered.ofJson with
  | some (.ok own) =>
    match own.find? (·.name == row) with
    | some q => some (.registered ((Flock.OneCircuit.pos P m.indices q).toOption.getD q))
    | none => some (.own row)
  | _ => some (.own row)

/-- **Where the verifier opens copy `j` of member `m`'s row read as `r`, at index `s` in it**: a registered port at the
tree `Spot.OneStageZk.treePortOf` names (`committed/<p>` gives `<p>`, `input/n<w>` and `<t>/o<j>` themselves), a
registered value at its own name, at the row `Flock.OneCircuit.pos` laid for the copy's unit (`readsRel`); a root of
its own at `<member>/<row>`, leaf `j`. -/
def argTree (m : Spot.OneStageZk.Member) (r : PortRead) (j s : ℕ) : Option TreeLeaf :=
  match r with
  | .own row => some ⟨s!"{m.name}/{row}", j, s⟩
  | .registered q => do
    let u ← m.indices[j]?
    let leaf ← (q.readsRel[u - q.readsFrom]?).join
    pure ⟨(Spot.OneStageZk.treePortOf q).getD q.value, leaf, s⟩

/-- The row of `rs` holding bit `i`, and the bit's index in it. -/
def rowAt : List (String × ℕ) → ℕ → Option (String × ℕ)
  | [], _ => none
  | (n, b) :: r, i => if i < b then some (n, i) else rowAt r (i - b)

/-- Where unit `g`'s argument `x` starts in `argBits`. -/
def argOff (P : Prog) (g x : ℕ) : ℕ := ((unitArgs P g).toList.take x).foldl (· + leafWidth P ·) 0

/-- The template `admitted` admits for member `m`'s block of C (`P`), as the per-member check reads it
(`AcceptGivesDoorFile`). -/
def admOf (P : Prog) (m : Spot.OneStageZk.Member) : Option Spot.OneStageZk.Admit := do
  let bl ← (m.own.blocks.get.toOption).join
  let (lo, hi) ← bl.find? (·.1 == m.own.first)
  let cuts ← m.own.cuts.get.toOption
  let S ← (Spot.Restrict.restrict P (List.range' lo (hi - lo)).toArray).toOption
  (Spot.OneStageZk.admitted m.own.partition cuts S).toOption

/-- **The map from a gate of C to the verifier's tree leaves**, under the members `ms`, member `m`'s session reading the
Circuit `cs m`: a return leaf `l` of a unit `v` of a member `m` at `outTree`, which a reader another block wires to it
opens too (`Flock.Registered.treeReadsWhole`); any gate no member's unit returns that an argument `x` of copy `j` of a
member reads, a return of a unit of no member among them, at the row and index the door lays it (`rowAt` of the input
rows), at that row's tree (`argTree`). A gate no member's unit returns and no member reads has no leaf. -/
def TreeAt (P : Prog) (ms : Array Spot.OneStageZk.Member) (cs : Spot.OneStageZk.Member → Option Flock.Circuit) (g : ℕ)
    (tl : TreeLeaf) : Prop :=
  (∃ m ∈ ms, ∃ (cc : Flock.Circuit) (v l : ℕ), cs m = some cc ∧ v ∈ m.indices ∧
    ((unitOuts P v)[l]?).map Spot.BlockFront.Port.gate = some g ∧ outTree P m cc v l = some tl) ∨
  ((¬ ∃ m ∈ ms, ∃ v l : ℕ, v ∈ m.indices ∧ ((unitOuts P v)[l]?).map Spot.BlockFront.Port.gate = some g) ∧
    ∃ m ∈ ms, ∃ (a : Spot.OneStageZk.Admit) (j x : ℕ) (row : String) (s : ℕ) (r : PortRead),
      admOf P m = some a ∧ j < m.indices.size ∧ ((unitArgs P m.indices[j]!)[x]?).map Spot.BlockFront.Port.gate = some g ∧
      rowAt a.ins.toList (argOff P m.indices[j]! x) = some (row, s) ∧ readsOf P m row = some r ∧
      argTree m r j s = some tl)

/-- A tree whose root the verifier holds before any coin: a root of the registration's record (`rootsByPort`), or a
registered value a member's registered file reads, one root for every member (`Spot.OneStageZk.classHeld`). -/
def HeldTree (P : Prog) (ms : Array Spot.OneStageZk.Member) (rec : Lean.Json) (t : String) : Prop :=
  (Spot.OneStageZk.rootsByPort rec).contains t = true ∨
    ∃ m ∈ ms, ∃ row q, readsOf P m row = some (.registered q) ∧ Spot.OneStageZk.treePortOf q = none ∧ q.value = t

open Classical in
/-- **The verifier's tree leaf at Core's position `p`** (`Partition.pos` at `servedLayout`): the one leaf `TreeAt`
gives the gate at `p`; none where no gate is at `p` or its gate has no leaf or two. -/
noncomputable def treeMap {n : ℕ} (P : Prog) (Pt : Verity.Partition (ofProgram P) n) (ms : Array Spot.OneStageZk.Member)
    (cs : Spot.OneStageZk.Member → Option Flock.Circuit) (p : Verity.Root × ℕ) : Option TreeLeaf :=
  if h : ∃ (g : Fin (ofProgram P).N) (tl : TreeLeaf), Pt.pos (servedLayout P) g = some p ∧ TreeAt P ms cs g.val tl ∧
      ∀ tl', TreeAt P ms cs g.val tl' → tl' = tl
  then some (Classical.choose (Classical.choose_spec h))
  else none

/-- **The verifier's trees as one string commitment over Core's positions**: a commitment is Core's root with the
verifier's root of every tree (`roots`); position `k` under root `r` opens where `M` takes `(r, k)`, as `Sc` opens that
tree at the leaf and slot (`Nat.pair`), and hashes what that opening hashed. Its commit and open are `Sc`'s, read only
for binding. -/
def treeSc {α : Type} (Sc : _root_.Hash.StringCommitment α) (M : Verity.Root × ℕ → Option TreeLeaf) :
    _root_.Hash.StringCommitment α where
  Com := Verity.Root × (String → Sc.Com)
  Dec := Sc.Dec
  Op := Sc.Op
  Coins := Sc.Coins
  commit H s r := ((.phase 0, fun _ => (Sc.commit H s r).1), (Sc.commit H s r).2)
  «open» H d k := Sc.open H d k
  Opens H C k v π := ∃ tl, M (C.1, k) = some tl ∧ Sc.Opens H (C.2 tl.tree) (Nat.pair tl.leaf tl.slot) v π
  hashed H C k v π := match M (C.1, k) with
    | some tl => Sc.hashed H (C.2 tl.tree) (Nat.pair tl.leaf tl.slot) v π
    | none => []

/-- **Member `m`'s units are the block its circuit file is restricted to**: one range `[lo, hi)`, the partition's block
that `restrictOn` finds by its first unit `m.own.first` (`Own.circuitFile`), as `membersOf` builds every member. -/
def PicksBlock (m : Spot.OneStageZk.Member) : Prop :=
  ∃ (bs : Array (ℕ × ℕ)) (lo hi : ℕ), m.own.blocks.get = .ok (some bs) ∧
    bs.find? (·.1 == m.own.first) = some (lo, hi) ∧ m.units = #[(lo, hi)]

/-- The accepted per-member check over C's partition: `checkMembersAt` accepts, `Pp`, `rs` are the program and
ranges its `membersRootsOf` reads, and every member's units are the block it is restricted to (`PicksBlock`). -/
def MembersAccept (p : Spot.OneStageZk.Params) (ms : Array Spot.OneStageZk.Member) (c : Spot.OneStage.Commitment)
    (Pp : Spot.Program) (rs : Array Flock.TemplateQuery.TRange) : Prop :=
  ∃ (ξ : ByteArray) (e : Spot.OneStageZk.Evidence) (sts : Array Spot.OneStageZk.StatementFiles) (bs : Core.Blobs)
    (pub : Spot.OneStage.Public) (prof : Core.Integrity),
    Spot.OneStageZk.checkMembersAt p ms c ξ e sts bs = .accept pub prof ∧
    Spot.Program.parse p.own.program = .ok Pp ∧
    Spot.Partition.evaluate Pp (Spot.getD p.own.partition "query") = .ok rs ∧ ∀ m ∈ ms, PicksBlock m

/-- **The Circuit each member's session reads**, at an accept of the per-member check: at the roots `membersRootsOf`
gives, member `m`'s statement step (`Member.statement`) on the accept's blobs carries an instance whose circuit file the
verifier parses (`Flock.ProvedScope.circuitOf` at `Flock.Witness.zkTags`, the parse `sessionAt` hands
`buildSessionOf`) to `cs m`. -/
def MembersCircuits (p : Spot.OneStageZk.Params) (ms : Array Spot.OneStageZk.Member) (c : Spot.OneStage.Commitment)
    (cs : Spot.OneStageZk.Member → Option Flock.Circuit) : Prop :=
  ∃ (ξ : ByteArray) (e : Spot.OneStageZk.Evidence) (sts : Array Spot.OneStageZk.StatementFiles) (bs : Core.Blobs)
    (pub : Spot.OneStage.Public) (prof : Core.Integrity),
    Spot.OneStageZk.checkMembersAt p ms c ξ e sts bs = .accept pub prof ∧ ∃ roots,
    Spot.OneStageZk.membersRootsOf p ms c.record = .ok roots ∧
    ∀ m ∈ ms, ∃ st, m.statement (m.slice c.record roots) bs = .ok st ∧
      (Flock.ProvedScope.circuitOf (Flock.Witness.zkTags st.I) st.I.circuitFile st.I.tables).toOption = cs m

/-- **`TreeMapTotal`**: on an accept, every committed gate a member's block reads or writes (`Partition.io`) has a
position at `servedLayout` and a leaf of a tree whose root the verifier holds before any coin (`HeldTree`). -/
def TreeMapTotal : Prop :=
  ∀ p ms c Pp rs cs, MembersAccept p ms c Pp rs → MembersCircuits p ms c cs →
    ∀ m ∈ ms, ∀ (fr : Spot.BlockFront.Front) (blocks : Array (Array (ℕ × ℕ))) (u : Fin blocks.size),
      Spot.BlockFront.front Pp.prog m.indices = .ok fr → BlockIs Pp.prog blocks u fr →
      ∀ g ∈ (partOf Pp.prog blocks).io u.castSucc, ∃ q tl,
        (partOf Pp.prog blocks).pos (servedLayout Pp.prog) g = some q ∧ TreeAt Pp.prog ms cs g.val tl ∧
          HeldTree Pp.prog ms c.record tl.tree

/-- **`TreeMapOne`**: on an accept, a committed gate has one tree leaf, whichever block reads it and whichever copy, so
`treeMap` is the leaf `TreeAt` gives it. -/
def TreeMapOne : Prop :=
  ∀ p ms c Pp rs cs, MembersAccept p ms c Pp rs → MembersCircuits p ms c cs →
    ∀ (blocks : Array (Array (ℕ × ℕ))) (g : Fin (ofProgram Pp.prog).N), g ∈ (partOf Pp.prog blocks).committed →
      ∀ tl tl', TreeAt Pp.prog ms cs g.val tl → TreeAt Pp.prog ms cs g.val tl' → tl = tl'

/-- Gate `g` of C is in root node `i`, a run Input node: a primitive Definition bound at run. -/
def RunInputGate (P : Prog) (g : ℕ) : Prop :=
  ∃ (i : ℕ) (x : Flock.Program.Node) (f : Flock.Program.Defn),
    (Flock.Registered.rootNodes P)[i]? = some x ∧ P.defs[x.fn]? = some f ∧
    f.prim = true ∧ f.binding = some Flock.Program.RUN ∧
    (((P.defs[P.root]?).map (·.body.offsets)).getD #[]).getD i 0 ≤ g ∧
    g < (((P.defs[P.root]?).map (·.body.offsets)).getD #[]).getD (i + 1) 0

/-- Gate `g` of C lies in the gates of a tied value the verifier lets through (`Spot.Docket.tiedGates`). -/
def TiedGate (v : Spot.Docket.Verifier) (g : ℕ) : Prop :=
  ∃ fc ∈ Spot.Docket.tiedGates v, fc.1 ≤ g ∧ g < fc.1 + fc.2

/-- Member `m` reads gate `g` of C at its input row `row`: an argument of one of its units' copies is `g`, at `row` in
the template `admOf` admits. -/
def ReadAt (P : Prog) (m : Spot.OneStageZk.Member) (g : ℕ) (row : String) : Prop :=
  ∃ (a : Spot.OneStageZk.Admit) (j x s : ℕ), admOf P m = some a ∧ j < m.indices.size ∧
    ((unitArgs P m.indices[j]!)[x]?).map Spot.BlockFront.Port.gate = some g ∧
    rowAt a.ins.toList (argOff P m.indices[j]! x) = some (row, s)

/-- The roots of member `m`'s slice of the record `rec`, at the roots `membersRootsOf` gave (`Member.slice`). -/
def sliceRoots (rec : Lean.Json) (roots : Std.HashMap String (Array Lean.Json)) (m : Spot.OneStageZk.Member) :
    Array Lean.Json :=
  match Spot.getD (m.slice rec roots) "roots" with | .arr rs => rs | _ => #[]

/-- **`RunReadsAtOneTree`** (stated, open): on an accepted run of the per-member variant, (1) each read by a member of a
gate of a run Input node outside the tied gates is a port of the member's file (`Own.registered`, laid by
`Flock.OneCircuit.pos`) whose value is the record's one tree of that node (`Flock.Registered.freshValue`), which
`OneStageZk.runInputRootsAt` holds; and (2) any two reads of gates of one tied run Input node, by one member or two,
are at slice roots with one root. Half (2) rests on `OneStageZk.tiedRootsAt`, which `OneStageZk.runInputRoots` runs
on each member before its run input roots: a second root for a tied node, among the roots the members hold of their own
at the ports that read it, is refused (`OneStageZk.TIED_ROOTS_WHY`); a member that holds no root of its own at such a
port is let through. -/
def RunReadsAtOneTree : Prop :=
  ∀ p ms c Pp rs, MembersAccept p ms c Pp rs →
    (∀ m ∈ ms, ∀ (a : Spot.OneStageZk.Admit) (j x g : ℕ) (row : String) (s : ℕ),
      admOf Pp.prog m = some a → j < m.indices.size →
      ((unitArgs Pp.prog m.indices[j]!)[x]?).map Spot.BlockFront.Port.gate = some g →
      RunInputGate Pp.prog g → ¬ TiedGate p.base.verifier g →
      rowAt a.ins.toList (argOff Pp.prog m.indices[j]! x) = some (row, s) →
      ∃ q i, readsOf Pp.prog m row = some (.registered q) ∧
        Flock.OfProgram.rootNode Pp.prog g = some i ∧ q.value = Flock.Registered.freshValue i) ∧
    ∀ roots, Spot.OneStageZk.membersRootsOf p ms c.record = .ok roots →
      ∀ m ∈ ms, ∀ m' ∈ ms, ∀ (g g' i : ℕ) (row row' : String),
        ReadAt Pp.prog m g row → ReadAt Pp.prog m' g' row' →
        Flock.OfProgram.rootNode Pp.prog g = some i → Flock.OfProgram.rootNode Pp.prog g' = some i →
        TiedGate p.base.verifier g →
        ∀ r ∈ sliceRoots c.record roots m, Spot.getD r "port" = .str row →
          ∀ r' ∈ sliceRoots c.record roots m', Spot.getD r' "port" = .str row' →
            Spot.getD r "root" = Spot.getD r' "root"

/-- **`TiedReadsAtRecordedRows regs`** (stated, open; the period test's link): on an accepted run of the per-member
variant, each root a reader of a tied run Input gate holds at the port that reads it, its slice's root (the record's
entry `<member>/<port>`, `OneStageZk.rootsOf`), is, with its domain, the root and domain of an entry of the `roots` of
one of the registrations `regs`: rows a registration that computed them committed. The record holds no entry under
the tied input's own name, and SPOT's verifier compares the readers' roots only with each other
(`OneStageZk.tiedRootsAt`). The network period test is to give it with `regs` the period's compute audits'
registrations: `NetworkVerifier.Period.wired` holds the scan's registration's `scan/prog<j>` entry, the scan member's
entry at its port `prog<j>`, to an audit's `<descriptor id>/c<j>/o0` entry by root and domain
(`Properties.NetworkPeriod.RowsProgWired`), and `tiedInputWired` holds the tied inputs the answer names to the lanes'
`prog<j>` (`RowsTiedInputWired`); that the other readers' roots are that root is `RunReadsAtOneTree`'s half (2). -/
def TiedReadsAtRecordedRows (regs : Array Lean.Json) : Prop :=
  ∀ p ms c Pp rs, MembersAccept p ms c Pp rs →
    ∀ roots, Spot.OneStageZk.membersRootsOf p ms c.record = .ok roots →
      ∀ m ∈ ms, ∀ (g : ℕ) (row : String), ReadAt Pp.prog m g row → TiedGate p.base.verifier g →
        ∀ r ∈ sliceRoots c.record roots m, Spot.getD r "port" = .str row →
          ∃ a ∈ regs, ∃ o ∈ (match Spot.getD a "roots" with | .arr os => os | _ => #[]),
            Spot.getD o "root" = Spot.getD r "root" ∧ Spot.getD o "domain" = Spot.getD r "domain"

/-- **`TreeMapOneOfOneTree`** (stated, open): every reader of an untied run Input node at its one tree and every reader
of a tied one at one root (`RunReadsAtOneTree`), that root a registration's committed rows' (`TiedReadsAtRecordedRows`),
give `TreeMapOne`. -/
def TreeMapOneOfOneTree : Prop :=
  ∀ regs, RunReadsAtOneTree → TiedReadsAtRecordedRows regs → TreeMapOne

/-- **`TreeMapInjective`**: on an accept, two gates of C at one tree leaf are one gate. -/
def TreeMapInjective : Prop :=
  ∀ p ms c Pp rs cs, MembersAccept p ms c Pp rs → MembersCircuits p ms c cs →
    ∀ (g g' : ℕ) (tl : TreeLeaf), TreeAt Pp.prog ms cs g tl → TreeAt Pp.prog ms cs g' tl → g = g'

/-- **`TreeScBinds`**: the verifier's trees bind where each tree does: `treeSc Sc M` binds under the collision
premise `Sc` binds under. -/
def TreeScBinds : Prop :=
  ∀ {α : Type} (Sc : _root_.Hash.StringCommitment α) (H : _root_.Hash.Msg → Verity.Digest)
    (M : Verity.Root × ℕ → Option TreeLeaf), Sc.Binds H → (treeSc Sc M).Binds H

/-- **`RblkOfTrees`**: block `u`'s openings at the verifier's tree leaves (`TreeAt`) of every gate of in(S) and
out(S), each at its tree's root `roots`, with out(S) computed, are `Partition.Rblk` at Core's commitment
`fun r => (r, roots)` of `treeSc Sc (treeMap …)`, at `servedLayout`, where every such gate has one leaf
(`TreeMapOne`'s conclusion). -/
def RblkOfTrees : Prop :=
  ∀ (P : Prog) (ms : Array Spot.OneStageZk.Member) (cs : Spot.OneStageZk.Member → Option Flock.Circuit)
    (blocks : Array (Array (ℕ × ℕ))) (u : Fin blocks.size) {α : Type} (Sc : _root_.Hash.StringCommitment α)
    (H : _root_.Hash.Msg → Verity.Digest) (enc : (g : Fin (ofProgram P).N) → BitVec ((ofProgram P).w g) → α)
    (roots : String → Sc.Com) (X : (ofProgram P).Trace) (π : Fin (ofProgram P).N → Sc.Op),
    let Pt := partOf P blocks
    let S := Pt.gates u.castSucc
    (∀ g ∈ (ofProgram P).into S ∪ (ofProgram P).exits S, ∀ tl tl', TreeAt P ms cs g.val tl →
      TreeAt P ms cs g.val tl' → tl = tl') →
    (∀ g ∈ (ofProgram P).into S ∪ (ofProgram P).exits S, ∃ tl, TreeAt P ms cs g.val tl ∧
      Sc.Opens H (roots tl.tree) (Nat.pair tl.leaf tl.slot) (enc g (X g)) (π g)) →
    (∀ g ∈ (ofProgram P).into S ∪ (ofProgram P).exits S, ∃ q, Pt.pos (servedLayout P) g = some q) →
    (∀ g ∈ (ofProgram P).exits S,
      X g = ((ofProgram P).restrict S).run (fun a => if Pt.unit a = some u.castSucc then none else some (X a)) g) →
    Pt.Rblk (treeSc Sc (treeMap P Pt ms cs)) H enc (servedLayout P) (fun r => (r, roots)) u.castSucc X π

/-- **Gap: the session's unit indices are the header's.** The public file's `units.indices`
(`Flock.Registered.unitIndices`), which the session reads, are the header's (`Spot.FilePart.indicesOf`), which
`unitsHeld` holds to the member's `Member.indices`. -/
def UnitIndicesAgree : Prop :=
  ∀ (pub : Flock.Public) (us vs : Array ℕ),
    Flock.Registered.unitIndices pub = .ok us → Spot.FilePart.indicesOf pub.header = .ok vs → us = vs

/-- **Gap: a held port is the record's tree.** A registered port `portTreeHeld` holds, which opens a committed tree
(`treePortOf`), carries the record's root, domain and leaf count at that tree's port. -/
def PortTreeHeldRecord : Prop :=
  ∀ (byPort : Std.HashMap String Lean.Json) (who : String) (tmpl : Option String) (q : Flock.Registered.Port)
    (t : String), Spot.OneStageZk.portTreeHeld byPort who tmpl q = .ok () → Spot.OneStageZk.treePortOf q = some t →
    ∃ e, byPort.get? t = some e ∧ Spot.getD e "root" = .str (Flock.hexOf q.root) ∧
      Spot.getD e "domain" = .str (Flock.hexOf q.domain) ∧ Spot.nat? (Spot.getD e "leaves") = some q.leaves

/-- **Gap: the verifier's check of a row is an opening of each of its values.** A row `row` whose payload
(`payloadOf`) `Flock.Registered.opensLeaf` accepts at `pos` under port `q`'s root opens, at each index `s`, its value
there under `Sc` at `rootOf q.root`, position `Nat.pair pos s`. -/
def OpensLeafOpens {α : Type} (Sc : _root_.Hash.StringCommitment α) (H : _root_.Hash.Msg → Verity.Digest)
    (rootOf : ByteArray → Sc.Com) (payloadOf : List α → ByteArray) (opOf : ByteArray → ℕ → Sc.Op) : Prop :=
  ∀ (q : Flock.Registered.Port) (row : List α) (pos : ℕ) (path : ByteArray),
    Flock.Registered.opensLeaf q (payloadOf row) pos path = true →
    ∀ s (hs : s < row.length), Sc.Opens H (rootOf q.root) (Nat.pair pos s) row[s] (opOf path s)

/-- **`FifthOpens`, the fifth statement's second half, over the map.** On an accept, for every committed gate of a
member's block (`Partition.io`), the position Core gives it (`Partition.pos` at `servedLayout`) is a leaf of a tree
whose root the verifier holds (`HeldTree`), `treeMap` takes the position there, and an opening the verifier checked
at that leaf is the opening `Partition.Rblk` reads at the position, under `treeSc`. -/
def FifthOpens : Prop :=
  ∀ p ms c Pp rs cs, MembersAccept p ms c Pp rs → MembersCircuits p ms c cs →
    ∀ m ∈ ms, ∀ (fr : Spot.BlockFront.Front) (blocks : Array (Array (ℕ × ℕ))) (u : Fin blocks.size),
      Spot.BlockFront.front Pp.prog m.indices = .ok fr → BlockIs Pp.prog blocks u fr →
      ∀ {α : Type} (Sc : _root_.Hash.StringCommitment α) (H : _root_.Hash.Msg → Verity.Digest) (roots : String → Sc.Com),
      ∀ g ∈ (partOf Pp.prog blocks).io u.castSucc, ∃ q tl,
        (partOf Pp.prog blocks).pos (servedLayout Pp.prog) g = some q ∧
        treeMap Pp.prog (partOf Pp.prog blocks) ms cs q = some tl ∧ HeldTree Pp.prog ms c.record tl.tree ∧
        ∀ v π, Sc.Opens H (roots tl.tree) (Nat.pair tl.leaf tl.slot) v π →
          (treeSc Sc (treeMap Pp.prog (partOf Pp.prog blocks) ms cs)).Opens H (q.1, roots) q.2 v π

end Flock.Chain
