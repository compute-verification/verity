import VStar.Compose
import VStar.Unit
import Flock.RecLinkTree

/-!
# `RecLink{UB, PB}`: one row of a recursive session's link table (`verity_flock.rec_link`)

The unit takes the table row, the reader's row digest and salt and its opening in the input rows' tree, and the writer's
in the output rows' tree (`Flock.RecLinkTree`). It recomputes each row's hm96 tree leaf (`hm96Leaf`,
`gates.hm96.leaf_gadget`), climbs it by the position the row names (the port's or row's `PB` bits, then the unit's `UB`),
and gives each root, then 512 zero bits, ANDed with `ok`: the row's kind 0 or 1, its bytes past the fields zero, its fields
in range, a filler's fields zero, and a link's two digests equal (`okForms`, `rec_link._ok`). The statement reads every
instance's `rin` and `rout` against the two roots the verifier registers.

Ports, u16 words: `trow` 64, `xr` 32, `yr` 96, `raux` `A(D)`, `xw` 32, `yw` 96, `waux` `A(D)` (`D = UB + PB`, `A` as
`RecOpen`'s `auxWords`); `rin` 64 and `rout` 64.

`LinkStmt` is V*'s statement of the table, over its `grid(L)` rows. No verifier reads it yet: a statement carrying a link
table is refused (`Flock.RecLinks.refuseLinks`) until a guarantee covers it.
-/

namespace VStar

open Flock.HmNets

def LINK_ROW_WORDS : Nat := 64
def LINK_X_WORDS : Nat := 32
def LINK_Y_WORDS : Nat := 96
def LINK_OUT_WORDS : Nat := 64
def LINK_FIELD_BITS : Nat := 32

/-- `rec_link.port_words`: `trow`, `xr`, `yr`, `raux`, `xw`, `yw`, `waux`. -/
def linkPorts (UB PB : Nat) : List Nat :=
  let a := auxWords (UB + PB)
  [LINK_ROW_WORDS, LINK_X_WORDS, LINK_Y_WORDS, a, LINK_X_WORDS, LINK_Y_WORDS, a]

/-- `rec_link.unit_name`. -/
def linkName (UB PB : Nat) : String := s!"rec-link/u{UB}-p{PB}"

/-- `rec_link`'s subcircuit: its seven ports, outputs `rin` and `rout`. -/
def linkSub (UB PB : Nat) : Sub :=
  { id := linkName UB PB, template := "rec-link", inputs := ["trow", "xr", "yr", "raux", "xw", "yw", "waux"],
    outputs := ["rin", "rout"] }

/-- The chaining value after a constant prefix of whole 128-byte blocks (`sha512.midstate`). -/
def midstate (prefix_ : ByteArray) : List (BitVec 64) :=
  let cv := (List.range (prefix_.size / 128)).foldl (fun cv i => Flock.Sha512.compress cv prefix_ (128 * i)) Flock.Sha512.iv
  cv.toList.map (·.toBitVec)

/-- `leaf_gadget`'s leaf (`rec_link._leaf`) of the digest `x` (512 bits) and the salt `y` (1536 bits), byte-string order
each: `b = x ⊕ M(key)·y`, committed word by word; `c = SHA-512(salt_prefix ‖ y)` from the prefix's midstate; the leaf
`SHA-512(leaf_prefix(key) ‖ b ‖ c)` from its prefix's midstate. The leaf's 512 forms, byte-string order. -/
def hm96Leaf (x y : List Lin) : M (List Lin) := do
  let xa := x.toArray
  let key := Flock.Hm96.Default512.key
  let bw ← (List.range 512).mapM fun p => commit (maskBit key y (wordBit p) (xa.getD (wordBit p) Lin.zero))
  let c ← shaFrom saltMid y Flock.Hm96.Default512.saltPrefix.size
  let ba := bw.toArray
  let b := (List.range 512).map fun t => ba.getD (wordBit t) Lin.zero
  let pre := Flock.Hm96.Default512.leafPrefix
  shaFrom (midstate pre) (b ++ c) pre.size

/-- `rec_link._all`'s rounds, at most `fuel` of them: each ANDs the first half's forms with the second's, pair by pair,
an odd last one carried. Each round shortens a list of two or more, so a list's length is fuel enough. -/
def allGo : Nat → List Lin → M Lin
  | _, [] => pure Lin.one
  | _, [a] => pure a
  | 0, a :: _ => pure a
  | fuel + 1, bits => do
    let half := bits.length / 2
    let ands ← ((bits.take half).zip ((bits.drop half).take half)).mapM fun (p, q) => Flock.HmNets.and p q
    allGo fuel (ands ++ bits.drop (2 * half))

/-- `rec_link._all`: 1 when every form is 1; 1 for none. -/
def allForms (bits : List Lin) : M Lin := allGo bits.length bits

/-- `rec_link._none`: 1 when every form is 0. -/
def noneForms (bits : List Lin) : M Lin := if bits.isEmpty then pure Lin.one else allForms (bits.map (· ^^^ Lin.one))

def orForm (a b : Lin) : M Lin := do
  let r ← Flock.HmNets.and (a ^^^ Lin.one) (b ^^^ Lin.one)
  pure (r ^^^ Lin.one)

/-- Field `f` of the table row: bits `32 f` to `32 f + 31`. -/
def linkField (row : List Lin) (f : Nat) : List Lin := (row.drop (LINK_FIELD_BITS * f)).take LINK_FIELD_BITS

/-- `rec_link._ok`. -/
def okForms (UB PB : Nat) (row xr xw : List Lin) : M Lin := do
  let field := linkField row
  let kind := field 0
  let restZero ← noneForms (kind.drop 1)
  let isFill ← Flock.HmNets.and restZero (kind.getD 0 Lin.zero ^^^ Lin.one)
  let isLink ← Flock.HmNets.and restZero (kind.getD 0 Lin.zero)
  let tail ← noneForms (row.drop (8 * Flock.RecLinks.FIELD_BYTES))
  let fits ← noneForms ((field 1).drop UB ++ (field 3).drop UB ++ (field 2).drop PB ++ (field 4).drop PB)
  let fz ← noneForms ((field 1).take UB ++ (field 3).take UB ++ (field 2).take PB ++ (field 4).take PB)
  let eq ← noneForms (List.zipWith (· ^^^ ·) xr xw)
  let kinds ← orForm isFill isLink
  let fillOk ← orForm (isFill ^^^ Lin.one) fz
  let linkOk ← orForm (isLink ^^^ Lin.one) eq
  allForms [kinds, tail, fits, fillOk, linkOk]

/-- `rec_link._steps` over forms: `(rin, rout)`'s bits, byte-string order. -/
def linkSteps (UB PB : Nat) (trow xr yr raux xw yw waux : List Lin) : M (List Lin × List Lin) := do
  let D := UB + PB
  let field := linkField trow
  let dr := (field 2).take PB ++ (field 1).take UB
  let dw := (field 4).take PB ++ (field 3).take UB
  let ok ← okForms UB PB trow xr xw
  let zeros := List.replicate (16 * (LINK_OUT_WORDS - NODE_WORDS)) Lin.zero
  let lr ← hm96Leaf xr yr
  let cr ← climbFrom dr 0 lr (sibForms raux D)
  let rin ← cr.mapM (Flock.HmNets.and ok)
  let lw ← hm96Leaf xw yw
  let cw ← climbFrom dw 0 lw (sibForms waux D)
  let rout ← cw.mapM (Flock.HmNets.and ok)
  pure (rin ++ zeros, rout ++ zeros)

/-- `rec_link.lowering(UB, PB)`'s unit: `rec_link.unit_circuit`. -/
def linkUnit (UB PB : Nat) : Except String UnitC := do
  Flock.RecLinkTree.checkShape UB PB
  let pw := linkPorts UB PB
  let build : M (List Lin × List Lin) := do
    let flat ← input (16 * pw.foldl (· + ·) 0)
    match slices flat pw with
    | [trow, xr, yr, raux, xw, yw, waux] => linkSteps UB PB trow xr yr raux xw yw waux
    | _ => pure ([], [])
  let r := build.run St.empty
  pure { name := linkName UB PB, inPorts := pw, outPorts := [LINK_OUT_WORDS, LINK_OUT_WORDS], st := r.2,
         outs := (r.1.1 ++ r.1.2).toArray }

/-- V*'s statement of a link table of `grid(L)` rows over an inner population of units of `UB` bits and rows of `PB`:
`RecLink{UB, PB}` over one instance a row, instance `i` checking row `i`. -/
structure LinkStmt where
  UB : Nat
  PB : Nat
  L : Nat

def LinkStmt.name (s : LinkStmt) : String := linkName s.UB s.PB
def LinkStmt.dir (_ : LinkStmt) : String := "links"
def LinkStmt.instances (s : LinkStmt) : Nat := Flock.RecLinks.grid s.L
def LinkStmt.sub (s : LinkStmt) : Sub := linkSub s.UB s.PB
def LinkStmt.unit (s : LinkStmt) : Except String UnitC := linkUnit s.UB s.PB

end VStar
