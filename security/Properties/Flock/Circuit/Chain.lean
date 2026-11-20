module

public import Properties.Flock.Circuit.OfProgram
public import Properties.Flock.Circuit.CircuitOf
public import Properties.Flock.Circuit.UnitInst
public import Properties.InferenceOnly
public import InferenceOnlyVerifier.Desc
public import Spot.Restrict
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
* `BlockEval`: the block's Program (`Spot.Restrict.portedOf`, `restrict` with its input nodes as parameters), lifted to
  bits as the lowering lifts it (`Flock.Lift.lift`), evaluates, on C's values at its input ports, to block `u`'s local
  evaluation (`Partition.localEval`) at its output ports, where each word primitive evaluates as its Boolean Definition
  in `P`'s own map (`BodyAgrees`).
* `BlockCorrect`: those rows, agreeing with a trace of C at the ports, make block `u` correct.
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

/-- **`BlockEval`, (C).** The block's Program, lifted to bits as the lowering lifts it, evaluates, on C's values at its
input ports, to block `u`'s local evaluation at its output ports, bit for bit. -/
def BlockEval : Prop :=
  ∀ (P : Prog) (units : Array ℕ) (fr : Spot.BlockFront.Front) (Q L : Prog) (e : ℕ),
    Spot.BlockFront.front P units = .ok fr → Spot.Restrict.portedOf P units = .ok Q →
    Flock.Lift.lift Q Q.root = .ok (L, e) → BodyAgrees P →
    ∀ (blocks : Array (Array (ℕ × ℕ))) (u : Fin blocks.size), BlockIs P blocks u fr →
    ∀ X : (ofProgram P).Trace,
      Prog.evalBool L e (portBits X fr.ins) = portBits ((partOf P blocks).localEval X u.castSucc) fr.outs

/-- **`BlockCorrect`, (J): the block is correct.** Rows of the net `lower`'s template of the block's Program unrolls to,
satisfied by `z` with the constant 1 (`LowerEval`'s form) and agreeing with a trace `X` of C at
the input and output port bits, make block `u` correct in C. -/
def BlockCorrect : Prop :=
  ∀ (P : Prog) (units : Array ℕ) (fr : Spot.BlockFront.Front) (Q : Prog)
    (t : Tmpl.Tmpl) (nt : Net),
    Spot.BlockFront.front P units = .ok fr → Spot.Restrict.portedOf P units = .ok Q →
    Flock.Lower.lower Q Q.root (fr.ins.map (·.width)) (fr.outs.map (·.width)) = .ok t → Tmpl.unroll t = .ok nt →
    nt.unitLog ≤ 32 → ∀ (hb : Built nt) (z : ℕ → Bool), (Rows.ofNet hb).Sat (fun c => z c.val) → z t.constPos = true →
    BodyAgrees P →
    ∀ (blocks : Array (Array (ℕ × ℕ))) (u : Fin blocks.size), BlockIs P blocks u fr →
    ∀ X : (ofProgram P).Trace,
      (∀ i < portWidth fr.ins, (portBits X fr.ins).getD i false = z (Flock.Lower.inBit t i)) →
      (∀ k < portWidth fr.outs, (portBits X fr.outs).getD k false = z (Flock.Lower.outBit t k)) →
      (partOf P blocks).Correct X u.castSucc

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
(`Partition.Rblk`) and the statement's own (`PortsOpenAt`), at one commitment, give `BlockCorrect`'s port hypotheses,
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
