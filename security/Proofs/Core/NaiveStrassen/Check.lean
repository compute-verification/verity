module

public import Proofs.Core.NaiveStrassen.Table

/-! The check of NaiveStrassen's kernel tables, which `Sound.lean` proves bounds the cost of every schedule. -/

@[expose] public section

namespace Verity.Accel.NaiveStrassen.Packed

def allBelow (n : Nat) (f : Nat → Bool) : Bool := Nat.rec (motive := fun _ => Bool) true (fun k acc => acc && f k) n

/-- The grid's data: `g` starts `1, 2` and increases, the root is the cell of 256 values, and `kc` and `rest` give
cells no larger than they must. -/
def checkGrid : Bool :=
  g 0 == 1 && g 1 == 2 && decide (root < gsz) && g root == 256 &&
    allBelow (gsz - 1) (fun k => decide (g k < g (k + 1))) &&
    allBelow gsz (fun i => allBelow 4 (fun ν => decide (g (kc i (ν + 1)) ≤ (g i - 1) * (ν + 1) + 1))) &&
    allBelow gsz (fun i => allBelow i (fun e => e == 0 ||
      decide (g (rest i e) ≤ (g i + (g (e + 1) - 1) - 1) / (g (e + 1) - 1))))

def rootOK (l : Nat) (t : Nat × Nat) : Bool :=
  (!decide (l ≤ thmL) || decide (8 ^ (l + 1) + 32 ≤ fld (row t.2 (sz l) root) root)) &&
    (decide (l ≤ thmL) || decide (8 ^ l < fld (row t.1 (sz l) root) root))

/-- Level `l`'s tables `t` against every option of the model, over the previous level's `prev`. -/
@[irreducible] def checkLevel (l : Nat) (t prev : Nat × Nat) : Bool :=
  let n := sz l
  let m := sz (l - 1)
  decide (n ≤ gsz) && decide (t.1 < 2 ^ (64 * n * n)) && decide (t.2 < 2 ^ (64 * n * n)) &&
    t.1 == transpose t.1 n && t.2 == transpose t.2 n &&
    allBelow n (fun i =>
      let li := row t.1 n i
      let hi := row t.2 n i
      small n li && small n hi &&
        leRow n li (8 ^ l * (row atomT gsz i &&& rowMask n)) &&
        leRow n hi (li + row t.2 n 1) &&
        allBelow i (fun e => e == 0 ||
          (leRow n li (row t.1 n e + row t.1 n (rest i e)) &&
            leRow n hi (row t.2 n e + row t.1 n (rest i e)) &&
            leRow n hi (row t.1 n e + row t.2 n (rest i e)))) &&
        (l == 0 ||
          (leRow n li (8 * (row prev.1 m i &&& rowMask n)) &&
            leRow n hi (7 * (row prev.1 m i &&& rowMask n) + (row prev.2 m i &&& rowMask n)) &&
            leRow n li (schemeRow prev.1 m n i strassenP) && leRow n hi (schemeRow prev.1 m n i strassenP) &&
            leRow n li (schemeRow prev.1 m n i winogradP) && leRow n hi (schemeRow prev.1 m n i winogradP)))) &&
    rootOK l t

/-- The tables to level `L`, each level checked as it is computed. -/
def run (L : Nat) : Bool × (Nat × Nat) :=
  Nat.rec (motive := fun _ => Bool × (Nat × Nat))
    (let t := level 0 (0, 0); (checkLevel 0 t (0, 0), t))
    (fun l p => let t := level (l + 1) p.2; (p.1 && checkLevel (l + 1) t p.2, t)) L

def checkAll : Bool := checkGrid && decide (topL ≤ 15) && (run topL).1

end Verity.Accel.NaiveStrassen.Packed
