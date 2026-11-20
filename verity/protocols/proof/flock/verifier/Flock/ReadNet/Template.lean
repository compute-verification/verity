/-
The table read's template (`EmbeddingRead_v1{V, H}`): the verifier builds it from the parts' generators, one Definition a
part, and `check` accepts a registered template only if it is this one. Building it runs each part's generator once, on
its own inputs; nothing walks the unrolled read.
-/
module

public import Flock.Tmpl
public import Flock.ReadNet.RowSeg
public import Flock.ReadNet.Leaf
public import Flock.ReadNet.Frame
public import Flock.ReadNet.Select

@[expose] public section

namespace Flock
namespace ReadNet

open HmNets

deriving instance DecidableEq for Tmpl.Seg
deriving instance DecidableEq for Tmpl.Call
deriving instance DecidableEq for Tmpl.Def

/-! ## A part as a Definition -/

/-- `n` input bits from bit `k`, as forms. -/
def bitsAt (k n : Nat) : List Lin := (List.range n).map fun i => Lin.var (k + i)

/-- `m` input words from bit `k`. -/
def wordsAt (k m : Nat) : List Word := (List.range m).map fun j => bitsAt (k + 64 * j) 64

/-- A part of the read: a generator over its input bits `0, …, ins - 1`, called by the read's template as one
Definition. -/
structure Part where
  ins : Nat
  prog : M (List Lin)

/-- The part's run, its fresh variables from `ins` on. -/
def Part.run (p : Part) : List Lin × St := p.prog.run ⟨p.ins, []⟩

/-- **A run as a Definition** on `ins` inputs: the run's rows in order, variable `x` at own column `x - ins` (input
`x < ins` at its input), then each returned form committed to an own column of its own (`form · 1`). It calls
nothing. -/
def defOf (ins : Nat) (r : List Lin × St) : Tmpl.Def :=
  let own := r.2.next - ins
  let cols := own + r.1.length
  let col := fun (x : Var) => if x < ins then cols + x else x - ins
  let refs := fun (f : Lin) => (f.s.map col).toArray ++ (if f.c then #[cols + ins] else #[])
  { cols, ins,
    rows := (r.2.rows.reverse.map fun w => (col w.out, refs w.a, refs w.b)).toArray ++
      (r.1.mapIdx fun k f => (own + k, refs f, #[cols + ins])).toArray,
    calls := #[] }

/-- Each of the run's rows writes a fresh variable: past the inputs and before the next. -/
def runOk (ins : Nat) (r : List Lin × St) : Bool := r.2.rows.all fun w => ins ≤ w.out && w.out < r.2.next

/-! ## The parts -/

/-- `Sha512RowSegSegment_v1{M}`: the chaining value (8 words), then the segment's `m` bits. -/
def segPart (m : Nat) : Part := ⟨512 + m, do let o ← segment m (wordsAt 0 8) (bitsAt 512 m); pure o.flatten⟩

/-- `Sha512RowSegOuter_v1{N_BITS = l}` over `s` segment digests. -/
def outerPart (l s : Nat) : Part := ⟨512 * s, do let o ← outer l (wordsAt 0 (8 * s)); pure o.flatten⟩

/-- `Hm96Sha512Leaf_v1`: the row's digest (8 words), then the salt (24 words); returns `b ‖ c ‖ leaf`. -/
def leafPart : Part := ⟨2048, do let o ← leaf (wordsAt 0 8) (wordsAt 512 24); pure o.flatten⟩

/-- `FrameV3Leaf_v1{B = n}` under the row-segment schema: the domain id, the value (8 words each), then the position's
`8 n` bits. -/
def leafHashPart (n : Nat) : Part :=
  ⟨1024 + 8 * n, do
    let o ← Frame.leafHash HmRow.ROW_SCHEMA_SEG n (wordsAt 0 8) (wordsAt 512 8) (bitsAt 1024 (8 * n))
    pure o.flatten⟩

/-- `FrameV3Node_v1{B = n}`: the domain id (8 words), the level (8 bits), the index (`8 n` bits), the swap bit, then
the current digest and the sibling (8 words each). -/
def nodePart (n : Nat) : Part :=
  ⟨1545 + 8 * n, do
    let o ← Frame.nodeHash n (wordsAt 0 8) (bitsAt 512 8) (bitsAt 520 (8 * n)) (Lin.var (520 + 8 * n))
      (wordsAt (521 + 8 * n) 8) (wordsAt (1033 + 8 * n) 8)
    pure o.flatten⟩

/-- `LmsEqual_v1{64}`: two digests (8 words each); returns the one bit. -/
def eqPart : Part := ⟨1024, do let b ← Frame.eqDigest (wordsAt 0 8) (wordsAt 512 8); pure [b]⟩

/-- `EmbeddingReadSelect_v1{V, H}`: the token (32 bits), the opened bit, then the row (`16 H` bits); returns `h` then
`ok`. -/
def selectPart (V H : Nat) : Part :=
  ⟨33 + 16 * H, do let r ← select V (bitsAt 0 32) (Lin.var 32) (bitsAt 33 (16 * H)); pure (r.1 ++ [r.2])⟩

/-! ## The read's template -/

/-- The depth of the frame-v3 tree over `V` leaves. -/
def depth (V : Nat) : Nat := if V ≤ 1 then 0 else Nat.log2 (V - 1) + 1

/-- The row's segments: `⌈16 H / 8192⌉`. -/
def segs (H : Nat) : Nat := (16 * H + HmRow.ROW_SEG_BITS - 1) / HmRow.ROW_SEG_BITS

/-- The read's parts, Definitions `0, …, 10`: a full segment, the last segment, the outer digest, the hm96 leaf, the
frame leaf, the frame node for each index width `B = 1, …, 4`, the digest comparison and the select. -/
def parts (V H : Nat) : Array Part :=
  let l := 16 * H
  #[segPart HmRow.ROW_SEG_BITS, segPart (l - HmRow.ROW_SEG_BITS * (segs H - 1)), outerPart l (segs H), leafPart,
    leafHashPart (Frame.indexBytes (depth V)), nodePart 1, nodePart 2, nodePart 3, nodePart 4, eqPart, selectPart V H]

/-- The parts' Definitions. -/
def partDefs (V H : Nat) : Array Tmpl.Def := (parts V H).map fun p => defOf p.ins p.run

/-- The root's Definition index. -/
def ROOT : Nat := 11

/-- The read's input ports: its input bits, the token (32), the root (512), the domain id (512), the row (`16 H`), the
salt (1536) and the path (`512 D`), as u16 words, the ports a served unit's rows give it. -/
def portBits (V H : Nat) : Array Nat := Array.replicate (162 + H + 32 * depth V) 16

/-- Where each port starts. -/
def ROOT_AT : Nat := 32
def DOMAIN_AT : Nat := 544
def ROW_AT : Nat := 1056
def saltAt (H : Nat) : Nat := 1056 + 16 * H
def pathAt (H : Nat) : Nat := 2592 + 16 * H

/-- The root's calls in order, `s` segments then `s + 5 + D` calls in all: each segment, the outer digest, the hm96
leaf, the frame leaf, a node a level, the comparison, the select. Call `i`'s Definition: -/
def partOf (V H i : Nat) : Nat :=
  let s := segs H
  let D := depth V
  if i + 1 < s then 0 else if i < s then 1 else if i = s then 2 else if i = s + 1 then 3 else if i = s + 2 then 4
  else if i < s + 3 + D then 4 + Frame.indexBytes (D - 1 - (i - (s + 3))) else if i = s + 3 + D then 9 else 10

/-- The number of the root's calls. -/
def nCalls (V H : Nat) : Nat := segs H + 5 + depth V

/-- Call `i`'s returned bits. -/
def nOut (V H i : Nat) : Nat :=
  let s := segs H
  let D := depth V
  if i = s + 1 then 1536 else if i = s + 3 + D then 1 else if i = s + 4 + D then 16 * H + 1 else 512

/-- The root's layout for the parts' Definitions `defs`: the input rows, the zero column, the calls' copies in order,
the output group (aligned), then the constant. -/
structure Layout where
  inW : Nat
  zero : Nat
  base : Nat → Nat
  outAt : Nat → Nat → Nat
  outW : Nat
  oStart : Nat
  const : Nat

/-- The root's layout. -/
def layout (V H : Nat) (defs : Array Tmpl.Def) : Layout :=
  let N := 2592 + 16 * H + 512 * depth V
  let inW := Nat.nextPowerOfTwo ((N + WORD - 1) / WORD)
  let zc := inW * WORD
  let colsOf := fun i => (defs.getD (partOf V H i) default).cols
  let base := fun i => zc + 1 + ((List.range i).map colsOf).sum
  let outW := Nat.nextPowerOfTwo ((16 * H + 16 + WORD - 1) / WORD)
  let span := outW * WORD
  let oStart := (base (nCalls V H) + span - 1) / span * span
  { inW, zero := zc, base, outAt := fun i k => base i + colsOf i - nOut V H i + k, outW, oStart, const := oStart + span }

/-- The bits of `v`'s low `w` bits as references: the constant where set, the zero column where not. The root reads
the constant as every Definition does, one past its columns and inputs (`L.const + 1`), never as its own last column:
`Tmpl.ordered` takes either, but the prover evaluates a template a Definition at a time, where every own column a row or
call reads comes before it (`verity_flock.tmpl.evaluate`). -/
def constRefs (L : Layout) (v w : Nat) : List Nat :=
  (List.range w).map fun j => if v.testBit j then L.const + 1 else L.zero

/-- Call `i`'s input references in the root. -/
def inputs (V H : Nat) (L : Layout) (i : Nat) : List Nat :=
  let s := segs H
  let D := depth V
  let l := 16 * H
  let outs := fun i n => (List.range n).map (L.outAt i)
  let at_ := fun k n => (List.range n).map (k + ·)
  let zeros := fun n => List.replicate n L.zero
  if i < s then
    let m := min (l - HmRow.ROW_SEG_BITS * i) HmRow.ROW_SEG_BITS
    let cv := Sha512.compress Sha512.iv (HmRow.segmentPrefix i s m) 0
    ((List.range 8).flatMap fun w => (List.range 64).map fun b =>
        if cv[w]!.toBitVec.getLsbD b then L.const + 1 else L.zero) ++ at_ (ROW_AT + HmRow.ROW_SEG_BITS * i) m
  else if i = s then (List.range s).flatMap fun j => outs j 512
  else if i = s + 1 then outs s 512 ++ at_ (saltAt H) 1536
  else if i = s + 2 then
    at_ DOMAIN_AT 512 ++ ((List.range 512).map fun k => L.outAt (s + 1) (1024 + k)) ++ at_ 0 D ++
      zeros (8 * Frame.indexBytes D - D)
  else if i < s + 3 + D then
    let k := i - (s + 3)
    let b := D - 1 - k
    at_ DOMAIN_AT 512 ++ constRefs L k 8 ++ at_ (k + 1) b ++ zeros (8 * Frame.indexBytes b - b) ++ [k] ++
      outs (i - 1) 512 ++ at_ (pathAt H + 512 * k) 512
  else if i = s + 3 + D then outs (s + 2 + D) 512 ++ at_ ROOT_AT 512
  else at_ 0 32 ++ [L.outAt (s + 3 + D) 0] ++ at_ ROW_AT l

/-- The root's Definition. -/
def rootDef (V H : Nat) (defs : Array Tmpl.Def) : Tmpl.Def :=
  let L := layout V H defs
  let N := 2592 + 16 * H + 512 * depth V
  let sel := segs H + 4 + depth V
  { cols := L.const + 1, ins := 0,
    rows := ((List.range N).map (fun c => (c, #[c], #[c]))).toArray.push (L.zero, #[L.const + 1], #[]) ++
      ((List.range (16 * H + 1)).map fun k => (L.oStart + k, #[L.outAt sel k], #[L.const + 1])).toArray,
    calls := ((List.range (nCalls V H)).map fun i =>
      ({ def_ := partOf V H i, count := 1, base := L.base i, stride := (defs.getD (partOf V H i) default).cols,
         segs := ((inputs V H L i).map fun r => ({ len := 1, at_ := r, step := 0, reps := 1, rstride := 0 } : Tmpl.Seg)).toArray }
        : Tmpl.Call)).toArray }

/-- **The read's template** for a table of `V` rows of `H` 16-bit words: the parts' Definitions, then the root. Its
outputs are `H + 1` ports of 16 bits, one output leaf each: the row's words, then the verdict in its port's low bit. -/
def readTmpl (V H : Nat) : Tmpl.Tmpl :=
  let defs := partDefs V H
  let L := layout V H defs
  { name := s!"EmbeddingRead_v1(V={V},H={H})", root := ROOT,
    inGroups := #[⟨0, L.inW, portBits V H⟩],
    outGroups := #[⟨L.oStart / WORD, L.outW, Array.replicate (H + 1) 16⟩],
    defs := defs.push (rootDef V H defs) }

/-- A port group's fields. -/
def groupKey (g : Group) : Nat × Nat × Array Nat := (g.col, g.words, g.bits)

/-- **The read's per-Definition check**: `V` and `H` in range, each part's run writing fresh variables, and the
registered template's Definitions, root and port groups those of `readTmpl V H`, and its unrolled unit at most `2^32`
rows. It reads the template's description once and runs each part's generator once; it never unrolls. -/
def check (V H : Nat) (t : Tmpl.Tmpl) : Bool :=
  let r := readTmpl V H
  2 ≤ V && V < 2 ^ 32 && 1 ≤ H && 16 * H < 2 ^ 64 && (parts V H).all (fun p => runOk p.ins p.run) && t.root == r.root &&
    t.inGroups.map groupKey == r.inGroups.map groupKey && t.outGroups.map groupKey == r.outGroups.map groupKey &&
    decide (t.defs = r.defs) && decide (log2ceil (t.defs.getD t.root default).cols ≤ 32)

/-! ## The template's text -/

/-- A port group list in a header: the count, then each group's column, words, port count and widths. -/
def groupsText (gs : Array Group) : List String :=
  toString gs.size :: gs.toList.flatMap fun g =>
    [toString g.col, toString g.words, toString g.bits.size] ++ g.bits.toList.map toString

/-- A template as `flock-ir-tmpl/v2` text (`Tmpl.parse` reads it back). -/
def render (t : Tmpl.Tmpl) : String := Id.run do
  let nums := fun (l : List Nat) => " ".intercalate (l.map toString)
  let mut out := " ".intercalate ([Tmpl.FORMAT, t.name, toString t.root] ++ groupsText t.inGroups ++
    groupsText t.outGroups) ++ "\n"
  for df in t.defs do
    out := out ++ s!"DEF {df.cols} {df.ins} {df.rows.size} {df.calls.size}\n"
    for r in df.rows do
      out := out ++ nums ([r.1, r.2.1.size] ++ r.2.1.toList ++ [r.2.2.size] ++ r.2.2.toList) ++ "\n"
    for c in df.calls do
      out := out ++ "CALL " ++ nums ([c.def_, c.count, c.base, c.stride, c.segs.size] ++
        c.segs.toList.flatMap fun s => [s.len, s.at_, s.step, s.reps, s.rstride]) ++ "\n"
  return out

end ReadNet
end Flock
