module

public import Difftest
public import PouwPearlPlus.Rule

/-!
# The inert-work cut's difftest: `Rule`'s cuts behind `Difftest.serve`

`pouw-pearlplus-cut-difftest` serves `table`, which `tests/benchmarks/pouw/pearlplus_lean/test_cut_difftest.py` holds to
`pearl_c_inert_work` and to `fast_tile`'s batch forms, on the H100's atom (`Evaluate.TC.Fp8Atom.hopperE4m3K32`):

* `cuts`: `Rule.cuts acc a b`, the 32 cut flags of one atom step from the FP32 word `acc` (`cut_flags`, `fast_tile._cut`);
* `word_cuts`: `Rule.wordCuts a b`, a word's cut flags along its chain (`chain_cuts`);
* `word_mend`: `Rule.wordMend` of an A row's and a B row's ends, each `{plus, minus, dead}` (`word_mend`,
  `fast_tile._mends`).

A code is below 256 and `acc` below 2^32; a word's rows, and each row's ends, are whole atoms of one length. Anything else
is refused.
-/

@[expose] public section

namespace PouwVerifier.PearlPlus.CutDifftest

open Lean (Json toJson)

def codes (i : Json) (k : String) : Except String (Array Nat) := do
  let c ← i.getObjValAs? (Array Nat) k
  if c.any (fun x => decide (256 ≤ x)) then throw s!"{k}: a code is not a byte"
  return c

def wholeAtoms (a b : Array Nat) : Except String Unit :=
  if a.size != b.size || a.size % 32 != 0 then throw "a word's rows are whole atoms of one length" else pure ()

def cutsFn (i : Json) : Except String Json := do
  let acc ← i.getObjValAs? Nat "acc"
  if acc ≥ 2 ^ 32 then throw "acc: not an FP32 word"
  let a ← codes i "a"
  let b ← codes i "b"
  if a.size != 32 || b.size != 32 then throw "an atom has 32 codes a side"
  return toJson (Rule.cuts acc a.toList b.toList)

def wordCutsFn (i : Json) : Except String Json := do
  let a ← codes i "a"
  let b ← codes i "b"
  wholeAtoms a b
  return toJson (Rule.wordCuts a b)

def ends (i : Json) (k : String) : Except String (Array Nat × Array Nat × Array Bool) := do
  let e ← i.getObjVal? k
  let plus ← codes e "plus"
  let minus ← codes e "minus"
  let dead ← e.getObjValAs? (Array Bool) "dead"
  wholeAtoms plus minus
  if dead.size != plus.size then throw s!"{k}: dead is not one flag per element"
  return (plus, minus, dead)

def wordMendFn (i : Json) : Except String Json := do
  let ea ← ends i "a"
  let eb ← ends i "b"
  wholeAtoms ea.1 eb.1
  return toJson (Rule.wordMend ea eb)

def table := _root_.Difftest.table [("cuts", cutsFn), ("word_cuts", wordCutsFn), ("word_mend", wordMendFn)]

end PouwVerifier.PearlPlus.CutDifftest
