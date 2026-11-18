module

public import Properties.Flock.Circuit.OfProgram
public import Properties.Flock.Circuit.CircuitOf
public import Properties.Flock.Circuit.UnitInst
public import Properties.InferenceOnly
public import InferenceOnlyVerifier.Desc
public import Spot.BlockFront

/-!
# From an accepted templated statement to a correct proof block of C

The statements the chain from the sampled proof's guarantee to `Partition.Correct` in C (`ofProgram P`, its blocks
`partOf P blocks`) reads, each a named `Prop`, open until its proof (`theorem` of the same name, lower case, in the proofs
tree):

* `V5Rows`: a unit the sampled proof's guarantee makes correct in the key circuit of a templated statement satisfies, at
  each copy, the rows of the net its template unrolls to (Core's `Partition.Correct` read as `Rows.Sat`, through the key
  program's wiring).
* `LowerEval` (`FlockSoundness.Assumptions.LowerEval`): rows of the net the template `Flock.Lower.lower`
  writes unrolls to, satisfied with the constant 1, hold at each output port bit the value `Prog.evalBool` gives the
  block's Definition on the input port bits.
* `BlockEval`: the block's Program (`Spot.BlockFront.program`) evaluates, on C's values at its input ports, to block `u`'s
  local evaluation (`Partition.localEval`) at its output ports, where each word primitive evaluates as the Boolean
  Definition C's Boolean map names for it (`BooleanAgrees`).
* `BlockCorrect`: those rows, agreeing with a trace of C at the ports, make block `u` correct.
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

/-- Each word primitive of `P` the Boolean map `boolean` names a Definition for evaluates as that Definition. -/
def BooleanAgrees (P : Prog) (boolean : Std.HashMap String String) : Prop :=
  ∀ p q, (P.defs[p]!).prim = true → boolean[(P.defs[p]!).key]? = some (P.defs[q]!).key →
    Prog.evalBool P p = Prog.evalBool P q

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

/-- **`BlockEval`, MERGE's (C).** The block's Program evaluates, on C's values at its input ports, to block `u`'s local
evaluation at its output ports, bit for bit. -/
def BlockEval : Prop :=
  ∀ (P : Prog) (units : Array ℕ) (boolean : Std.HashMap String String) (fr : Spot.BlockFront.Front) (Q : Prog),
    Spot.BlockFront.front P units boolean = .ok fr → Spot.BlockFront.program P units boolean = .ok Q →
    BooleanAgrees P boolean →
    ∀ (blocks : Array (Array (ℕ × ℕ))) (u : Fin blocks.size), BlockIs P blocks u fr →
    ∀ X : (ofProgram P).Trace,
      Prog.evalBool Q Q.root (portBits X fr.ins) = portBits ((partOf P blocks).localEval X u.castSucc) fr.outs

/-- **`BlockCorrect`, (J): the block is correct.** Rows of the net `lower`'s template of the block's Program unrolls to,
satisfied by `z` with the constant 1 (`FlockSoundness.Assumptions.LowerEval`'s form) and agreeing with a trace `X` of C at
the input and output port bits, make block `u` correct in C. -/
def BlockCorrect : Prop :=
  ∀ (P : Prog) (units : Array ℕ) (boolean : Std.HashMap String String) (fr : Spot.BlockFront.Front) (Q : Prog)
    (t : Tmpl.Tmpl) (nt : Net),
    Spot.BlockFront.front P units boolean = .ok fr → Spot.BlockFront.program P units boolean = .ok Q →
    Flock.Lower.lower Q Q.root (fr.ins.map (·.width)) (fr.outs.map (·.width)) = .ok t → Tmpl.unroll t = .ok nt →
    nt.unitLog ≤ 32 → ∀ (hb : Built nt) (z : ℕ → Bool), (Rows.ofNet hb).Sat (fun c => z c.val) → z t.constPos = true →
    BooleanAgrees P boolean →
    ∀ (blocks : Array (Array (ℕ × ℕ))) (u : Fin blocks.size), BlockIs P blocks u fr →
    ∀ X : (ofProgram P).Trace,
      (∀ i < portWidth fr.ins, (portBits X fr.ins).getD i false = z (Flock.Lower.inBit t i)) →
      (∀ k < portWidth fr.outs, (portBits X fr.outs).getD k false = z (Flock.Lower.outBit t k)) →
      (partOf P blocks).Correct X u.castSucc

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
