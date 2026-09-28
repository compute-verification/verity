import PousSponge.SpongeModel

/-!
# Lean-evaluated vectors for `verity.protocols.accounting.space.pous.dense`

Evaluates `chainDense`'s own definitions (`lean/submissions/sponge-dense/SpongeModel.lean` §4, §9) on small cases and
prints them as JSON (`tests/fixtures/lean_dense_vectors.json`): `labelOv` (the encoder), the decoder's formula, `dom`,
`iv`, `padBlk` and `ovKey`, with the real 512-bit `dom`.  `Π₂` is the computable toy `toyF` (`x ↦ a·x + b mod
2^w`, a bijection), which `tests/test_pous_dense.py` plugs into the Python.  The `rfl` theorems tie `labelOv` and the
decoder's formula to `chainDense`'s `enc` and `dec`; the decoder is evaluated through its formula, since compiled code
for `chainDense` itself would build `idealPerm`'s instance over every permutation.  Regenerate with
`tests/lean/generate_dense.sh`.
-/

namespace PousDenseCheck

open Pous PousSponge

theorem enc_eq (B ℓ c : ℕ) [NeZero B] (d : ℕ → Bits c) (W : Bits (B * ℓ)) (π : Equiv.Perm (Bits (ℓ + (ℓ + c)))) :
    ((chainDense B ℓ c d).enc W π ()).2 = labelOv (fun x => π x) d (Dense.blocks B ℓ W) := rfl

theorem dec_eq (B ℓ c : ℕ) [NeZero B] (d : ℕ → Bits c) (pp : Bits 0) (C : Fin B → Bits ℓ)
    (f : (idealPerm (ℓ + (ℓ + c))).Qry → Bits (ℓ + (ℓ + c))) :
    (chainDense B ℓ c d).dec pp C f =
      codeEquiv B ℓ fun i => Labelling.bxor (C i) (ovKey (fun x => f (.inl x)) (d i) (absorbed C i)) := rfl

def bitsNat {w : ℕ} (b : Bits w) : ℕ := (List.finRange w).foldl (fun a i => if b i then a + 2 ^ i.val else a) 0

def natBits (w n : ℕ) : Bits w := fun i => n.testBit i.val

def toyA : ℕ := 11400714819323198485

def toyF (w : ℕ) (x : Bits w) : Bits w := natBits w ((toyA * bitsNat x + 12345 + w) % 2 ^ w)

def toyQ (w : ℕ) : (idealPerm w).Qry → Bits w
  | .inl x => toyF w x
  | .inr y => y

/-- The decoder's queries are forward ones, answered by `toyF` (`dec_eq` with `f := toyQ w`). -/
theorem toyQ_inl (w : ℕ) (x : Bits w) : toyQ w (.inl x) = toyF w x := rfl

def saltOf (seed : ℕ) : Bits 256 := fun i => (i.val * 7 + seed) % 5 < 2

def wOf (B ℓ seed : ℕ) : Fin B → Bits ℓ := fun i j => (i.val * 13 + j.val * 5 + seed) % 7 < 3

def natsJson (xs : List ℕ) : String := "[" ++ ", ".intercalate (xs.map fun x => s!"\"{x}\"") ++ "]"

/-- A 4-bit analogue of `dom`, for chains Lean can evaluate: a 2-bit salt, then the index in binary.  Compiled `ovStep`
recomputes its call on every bit read, and a label's parents are recomputed on every read, so evaluation costs about
`(2ℓ + c)` to the total chain depth: whole encodings are checked for `B ≤ 2`, the absorption order through `ovKey` on
four blocks, and the real 512-bit `dom` and `iv` on one call. -/
def domSmall (salt : Bits 2) (i : ℕ) : Bits 4 := Fin.append salt fun j : Fin 2 => i.testBit j

def chainCase (B ℓ c seed : ℕ) (d : ℕ → Bits c) (salt : ℕ) : String :=
  let w := ℓ + (ℓ + c)
  let Wb := wOf B ℓ seed
  let C := labelOv (toyF w) d Wb
  let dec := codeEquiv B ℓ fun i => Labelling.bxor (C i) (ovKey (toyF w) (d i) (absorbed C i))
  let blocks := (List.finRange B).map fun i => bitsNat (C i)
  "{" ++ s!"\"B\": {B}, \"l\": {ℓ}, \"c\": {c}, \"seed\": {seed}, \"salt\": \"{salt}\", " ++
    s!"\"w\": \"{bitsNat (codeEquiv B ℓ Wb)}\", \"blocks\": {natsJson blocks}, \"decode\": \"{bitsNat dec}\"" ++ "}"

def smallCase (B ℓ seed : ℕ) : String :=
  let salt : Bits 2 := fun i => (i.val + seed) % 3 = 0
  chainCase B ℓ 4 seed (domSmall salt) (bitsNat salt)

def realCase (B ℓ seed : ℕ) : String := chainCase B ℓ 512 seed (dom (saltOf seed)) (bitsNat (saltOf seed))

def layoutCase (ℓ seed : ℕ) : String :=
  let d := dom (saltOf seed)
  let sample : List (Bits ℓ) := [natBits ℓ 5, natBits ℓ 3, natBits ℓ 6, padBlk]
  let salt : Bits 2 := fun i => (i.val + seed) % 3 = 0
  "{" ++ s!"\"l\": {ℓ}, \"seed\": {seed}, \"salt\": \"{bitsNat (saltOf seed)}\", " ++
    s!"\"dom3\": \"{bitsNat (d 3)}\", \"iv3\": \"{bitsNat (iv (r := ℓ) (d 3))}\", " ++
    s!"\"pad\": \"{bitsNat (padBlk (r := ℓ))}\", \"small_salt\": \"{bitsNat salt}\", " ++
    s!"\"key4\": \"{bitsNat (ovKey (toyF (ℓ + (ℓ + 4))) (domSmall salt 2) sample)}\"" ++ "}"

end PousDenseCheck

open PousDenseCheck in
def main : IO Unit := do
  let small := [smallCase 1 3 1, smallCase 2 4 2, smallCase 2 3 3, smallCase 2 16 4]
  let real := [realCase 1 4 1, realCase 1 8 2]
  let layout := [layoutCase 3 1, layoutCase 4 2, layoutCase 16 3]
  IO.println ("{\"small\": [\n" ++ ",\n".intercalate small ++ "\n], \"real\": [\n" ++ ",\n".intercalate real ++
    "\n], \"layout\": [\n" ++ ",\n".intercalate layout ++ "\n]}")
