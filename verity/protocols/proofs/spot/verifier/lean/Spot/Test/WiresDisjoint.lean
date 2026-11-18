module

public import Spot.Wires
meta import Spot.Wires

/-!
`pairwiseDisjoint`'s compiled code, `pairwiseDisjointFast`, held to the pairwise test on random and adversarial lists
(nested, equal starts, touching ends, empty units or rows, other readers and ports, row cuts), and its work at 700k wires:
the live wires it compares, counted by `sweepWork`, at most a small multiple of the wires, where the pairwise test makes
W(W−1)/2 comparisons.
-/

@[expose] public section

namespace Spot.Test.WiresDisjoint

open Spot.Wires

/-- The pairwise test as written, kept out of `@[csimp]`'s reach as the reference. -/
def pairwiseRef : List Wire → Bool
  | [] => true
  | w :: ws => ws.all (disjoint w) && pairwiseRef ws

/-- `sweep`'s comparisons: the live wires it filters and compares, wire by wire. -/
def sweepWork : List Wire → List Wire → Nat → Nat
  | _, [], n => n
  | live, w :: ws, n =>
    let n := n + live.length
    let live := live.filter fun a => a.reader == w.reader && a.port == w.port && w.start < a.start + a.readerUnits
    if live.all (disjoint w) then sweepWork (w :: live) ws (n + live.length) else n + live.length

/-- A wire of reader `r`'s port `p` over units `[s, s + n)` and rows `[c0, c0 + c)`. -/
def wire (r p : String) (s n c0 c : Nat) : Wire :=
  { reader := r, port := p, writer := "w", out := "o", start := s, radix := if c ≤ 1 then [n] else [n, c], base := 0,
    step := if c ≤ 1 then [1] else [1, 0], cut0 := c0, cuts := c }

#guard (wire "r" "p" 3 4 0 1).readerUnits == 4 && (wire "r" "p" 3 4 2 3).readerUnits == 4

/-- A deterministic stream of naturals (a 64-bit LCG). -/
def lcg (x : Nat) : Nat := (x * 6364136223846793005 + 1442695040888963407) % 2 ^ 64

/-- `n` wires from seed `x`, each field drawn below its bound: few readers and ports, short ranges, so pairs meet. -/
def randomWires (x n units rows : Nat) : List Wire :=
  (List.range n).foldl (fun (acc, x) _ =>
    let x₁ := lcg x; let x₂ := lcg x₁; let x₃ := lcg x₂; let x₄ := lcg x₃; let x₅ := lcg x₄; let x₆ := lcg x₅
    (wire (toString (x₁ / 2 ^ 40 % 2)) (toString (x₂ / 2 ^ 40 % 2)) (x₃ / 2 ^ 40 % units) (x₄ / 2 ^ 40 % 4)
      (x₅ / 2 ^ 40 % rows) (x₆ / 2 ^ 40 % 3) :: acc, x₆)) ([], x) |>.1

def agree (ws : List Wire) : Bool := pairwiseDisjointFast ws == pairwiseRef ws

/-- Adversarial lists, each with its verdict under the pairwise test. -/
def cases : List (List Wire × Bool) := [
  ([], true),
  ([wire "r" "p" 0 4 0 1], true),
  ([wire "r" "p" 0 10 0 1, wire "r" "p" 2 3 0 1], false),
  ([wire "r" "p" 0 10 0 1, wire "r" "p" 1 1 0 1, wire "r" "p" 5 1 0 1], false),
  ([wire "r" "p" 0 100 0 1, wire "r" "p" 10 1 1 1, wire "r" "p" 99 1 0 1], false),
  ([wire "r" "p" 5 2 0 1, wire "r" "p" 5 3 0 1], false),
  ([wire "r" "p" 5 2 0 1, wire "r" "p" 5 3 1 1], true),
  ([wire "r" "p" 0 4 0 1, wire "r" "p" 4 4 0 1, wire "r" "p" 8 4 0 1], true),
  ([wire "r" "p" 8 4 0 1, wire "r" "p" 0 4 0 1, wire "r" "p" 4 4 0 1], true),
  ([wire "r" "p" 0 4 0 2, wire "r" "p" 0 4 2 2, wire "r" "p" 0 4 1 2], false),
  ([wire "r" "p" 0 4 0 2, wire "r" "p" 0 4 2 2, wire "r" "p" 4 4 1 2], true),
  ([wire "r" "p" 0 4 0 1, wire "r" "q" 0 4 0 1, wire "s" "p" 0 4 0 1], true),
  ([wire "r" "p" 0 4 0 1, wire "s" "p" 0 9 0 1, wire "r" "p" 3 1 0 1], false),
  ([wire "r" "p" 0 0 0 1, wire "r" "p" 0 4 0 1], true),
  ([wire "r" "p" 2 0 0 1, wire "r" "p" 0 4 0 1], false),
  ([wire "r" "p" 2 0 0 1, wire "r" "p" 2 0 0 1], true),
  ([wire "r" "p" 0 4 0 0, wire "r" "p" 0 4 0 1], true),
  ([wire "r" "p" 0 1000 0 1] ++ (List.range 50).map (fun i => wire "r" "p" (2000 + i) 1 0 1) ++
    [wire "r" "p" 999 1 0 1], false),
  ((List.range 50).map (fun i => wire "r" "p" (100 - 2 * i) (4 * i + 1) 0 1), false),
  ((List.range 50).map (fun i => wire "r" "p" (2 * i) 2 0 1), true)]

#guard cases.all fun (ws, v) => pairwiseRef ws == v && pairwiseDisjointFast ws == v

/-- Each random list's seed, length, units and rows. -/
def draws : List (Nat × Nat × Nat × Nat) :=
  (List.range 400).map fun i => (i, 2 + i % 12, 4 + i % 40, 1 + i % 4)

#guard draws.all fun (x, n, u, r) => agree (randomWires x n u r)
#guard (draws.filter fun (x, n, u, r) => pairwiseRef (randomWires x n u r)).length ≥ 20
#guard (draws.filter fun (x, n, u, r) => !pairwiseRef (randomWires x n u r)).length ≥ 20

/-- A served class's shape at 700k wires: 1,000 readers of 7 ports, each port cut into two rows and read by 80 wires over
consecutive units, every fourth one's two rows read by a wire each (100 wires a port), the list in reverse of its sorted
order. -/
def served : List Wire :=
  ((List.range 1000).flatMap fun r => (List.range 7).flatMap fun p => (List.range 80).flatMap fun i =>
    let rd := s!"t{r}"; let pt := s!"p{p}"
    if i % 4 == 0 then [wire rd pt (4 * i) 4 0 1, wire rd pt (4 * i) 4 1 1] else [wire rd pt (4 * i) 4 0 2]).reverse

#guard served.length == 700000
#guard pairwiseDisjointFast served
#guard sweepWork [] (served.mergeSort before) 0 ≤ 4 * served.length
#guard !pairwiseDisjointFast (wire "t5" "p3" 41 2 0 1 :: served)

end Spot.Test.WiresDisjoint
