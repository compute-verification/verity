/-
`RecResiduals.partlyRead` on small nets: the row ports' bits sit at `rowCol`'s columns (port `n` elements wide spans
`16 · portWordsOf n` columns), an element read in no bit or in all 128 passes, and the first element read in some but not
all is named with how many, wherever it sits among held elements read in no bit. `readIns` marks the input columns the
computed rows read and nothing past the inputs. `RecOpening.rsUnread` on m = 22's shape: w's slot 0 is a claim's ring
switch's (`.rs 0 0`) and slots 362–365 the running sums' (`.acc0 0` to `.acc1 1`), each named unless read in all 128
bits, and slot 256 (Ligerito's first) is neither. `RecOpening.accHeld` at m = 22's shape, on an S of one residual per
list of w slots it adds: refused with no residual, with only `acc0`'s port held, and with only a ring switch's and
`acc1`'s; passed once some part holds both running sums' ports, in one residual or two.

`RecResiduals.misnamed` on a unit of 128 16-bit ports, its two row-port elements at words 0–7 and 64–71: circuit ports
`m0` and `m1` of 8 words each key them where copy 0's leaves are theirs, and the first bit keyed by a port of another
name than its element needs is named, at copy 0 even with no copy, and at each copy below `upv`; a forced zero (leaf -1)
keys none. `roundNames` names each round holding a key, `RecOpening.rsName` claim `i`'s ring switch's elements `m<i>`.
-/
import Flock.RecOpening

namespace Flock.Test.ReadWhole

open Flock

/-- A net of `nIn` input columns whose computed rows read, in A, the columns of `reads` (one row each, B empty). -/
def net (inWords : Nat) (reads : List (List Nat)) : Net :=
  let nIn := inWords * WORD
  let rows := Array.replicate nIn #[] ++ (reads.map List.toArray).toArray
  let w := nIn + reads.length + 1
  match Sparse.ofRows rows w, Sparse.ofRows (Array.replicate rows.size #[]) w with
  | .ok a, .ok b => { (default : Net) with constPos := nIn + reads.length, inWords, a, b }
  | _, _ => default

def partly (inWords : Nat) (ns : List Nat) (reads : List (List Nat)) : Option (Nat × Nat) :=
  RecResiduals.partlyRead (RecResiduals.readIns (net inWords reads)) ns

def rsUnread (W : Array Nat) (reads : List (List Nat)) : Option (Nat × Nat) :=
  match RecOpening.shape? 22 2 with
  | some sh => RecOpening.rsUnread sh W (RecResiduals.readIns (net (8 * W.size) reads)) (W.toList.map fun _ => 1)
  | none => some (0, 0)

#guard RecResiduals.rowCol [1, 1] 0 5 == 5
#guard RecResiduals.rowCol [1, 1] 0 (128 + 5) == 16 * 64 + 5
#guard partly 16 [1, 1] [] == none
#guard partly 16 [1, 1] [List.range 128] == none
#guard partly 16 [1, 1] [List.range 64, List.range' 64 64] == none
#guard partly 16 [1, 1] [List.range 64] == some (0, 64)
#guard partly 16 [1, 1] [List.range 128, [1024 + 7]] == some (1, 1)
#guard partly 16 [1, 1] [List.range 128, List.range' 1024 128] == none
#guard partly 16 [1, 1] [List.range' 1024 127] == some (1, 127)
#guard partly 24 [1, 1, 1] [List.range 128, List.range' 2048 128] == none
#guard partly 24 [1, 1, 1] [List.range 128, List.range' 2048 100] == some (2, 100)
#guard (RecResiduals.readIns (net 16 [[3, 2048 + 1]])).size == 2048
#guard (RecResiduals.readIns (net 16 [[2048]])).toList.all (!·)
#guard rsUnread #[0] [List.range 128] == none
#guard rsUnread #[0] [List.range 100] == some (0, 100)
#guard rsUnread #[0] [] == some (0, 0)
#guard rsUnread #[256, 0] [List.range' 1024 128] == none
#guard rsUnread #[256, 0] [List.range 128] == some (1, 0)
#guard (RecOpening.shape? 22 2).map (fun sh => [.acc0 0, .acc0 1, .acc1 0, .acc1 1].map (RecOpening.wi sh)) ==
  some [some 362, some 363, some 364, some 365]
#guard rsUnread #[362] [List.range 128] == none
#guard rsUnread #[363] [List.range 100] == some (0, 100)
#guard rsUnread #[364] [] == some (0, 0)
#guard rsUnread #[256, 365] [List.range 128] == some (1, 0)
#guard rsUnread #[362, 363, 364, 365] [List.range 128, List.range' 1024 128, List.range' 2048 128] == some (3, 0)

def accHeld (ts : List (List Nat)) : Bool :=
  match RecOpening.shape? 22 2 with
  | some sh => RecOpening.accHeld sh
      ⟨RecOpening.nw sh, 0, #[], (ts.map fun t => (⟨#[], (t.map RecAlgebra.Term.w).toArray⟩ : RecAlgebra.Res)).toArray⟩
  | none => true

#guard !accHeld []
#guard !accHeld [[362]]
#guard !accHeld [[0], [365]]
#guard accHeld [[362, 364]]
#guard accHeld [[363], [365]]

/-- A circuit of `upv` copies whose unit's inputs are 128 16-bit ports, with circuit ports `m0` and `m1` of 8 words, and
copy `k`'s input word `i` at leaf `(leaves k)[i]`. -/
def circ (upv : Nat) (leaves : List (Array Int)) : Circuit :=
  { (default : Circuit) with
    nets := #[("unit", { (default : Net) with inGroups := #[⟨0, 2, Array.replicate 128 16⟩] })]
    ports := #[{ name := "m0", words := 8, chunks := 1, nb := 1 }, { name := "m1", words := 8, chunks := 1, nb := 1 }]
    upv, leavesIn := leaves.toArray }

/-- Words 0–7 at `m0`'s leaves and 64–71 at `a`'s, the rest forced zeros. -/
def leaves (a : Nat) : Array Int :=
  (Array.range 128).map fun i => if i < 8 then Int.ofNat i else if 64 ≤ i ∧ i < 72 then Int.ofNat (a + i - 64) else -1

def need (n0 n1 : String) (a : Nat) : List String := if a == 0 then [n0] else [n1]

#guard RecResiduals.keyPort (circ 1 [leaves 8]) 0 5 == some 0
#guard RecResiduals.keyPort (circ 1 [leaves 8]) 0 (1024 + 3) == some 1
#guard RecResiduals.keyPort (circ 1 [leaves 8]) 0 200 == none
#guard RecResiduals.misnamed (circ 1 [leaves 8]) [1, 1] 2 (need "m0" "m1") == none
#guard RecResiduals.misnamed (circ 1 [leaves 8]) [1, 1] 2 (need "m1" "m1") == some (0, 0, 0, 0)
#guard RecResiduals.misnamed (circ 1 [leaves 0]) [1, 1] 2 (need "m0" "m1") == some (0, 1, 0, 0)
#guard RecResiduals.misnamed (circ 0 [leaves 0]) [1, 1] 2 (need "m0" "m1") == some (0, 1, 0, 0)
#guard RecResiduals.misnamed (circ 1 [leaves 0]) [1, 1] 1 (need "m0" "m1") == none
#guard RecResiduals.misnamed (circ 1 [Array.replicate 128 (-1)]) [1, 1] 2 (need "m1" "m0") == none
#guard RecResiduals.misnamed (circ 2 [leaves 8, leaves 8]) [1, 1] 2 (need "m0" "m1") == none
#guard RecResiduals.misnamed (circ 2 [leaves 8, leaves 0]) [1, 1] 2 (need "m0" "m1") == some (1, 1, 0, 0)
#guard RecResiduals.misnamed (circ 1 [leaves 8, leaves 0]) [1, 1] 2 (need "m0" "m1") == none
#guard RecResiduals.roundNames #[#[.fa, .fb], #[.T]] (some .T) == ["m1"]
#guard RecResiduals.roundNames #[#[.fa, .fb], #[.T, .fa]] (some .fa) == ["m0", "m1"]
#guard RecResiduals.roundNames #[#[.fa, .fb], #[.T]] (some (.ood 0)) == []
#guard RecResiduals.roundNames #[#[.fa, .fb], #[.T]] none == []

def rsName (W : Array Nat) (a : Nat) : List String :=
  match RecOpening.shape? 22 2 with
  | some sh => RecOpening.rsName sh W a
  | none => ["no shape"]

#guard rsName #[0, 128 + 5, 256] 0 == ["m0"]
#guard rsName #[0, 128 + 5, 256] 1 == ["m1"]
#guard rsName #[0, 128 + 5, 256] 2 == []

end Flock.Test.ReadWhole
