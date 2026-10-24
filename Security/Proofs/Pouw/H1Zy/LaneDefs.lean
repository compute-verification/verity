import Definitions.Pouw.H1Zy.Lane
import Proofs.Pouw.H1Zy.CombineDefs

/-! The proof-only part of `Definitions.Pouw.H1Zy.Lane`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Pouw.H1Zy

/-- `PouwMuxBit_v1` on a candidate: `b` where `s` is `1`, else `a` (bitwise on a word, so on a whole candidate). -/
def mux {α : Type} (s : Bool) (a b : α) : α := if s then b else a

/-- **`h1zy._pick`**: `cands[Σ_b sel[b] 2^b]` as a `PouwMuxBit` tree over the selector's bits, least significant first,
each pass muxing adjacent candidates; `d` for an empty list. -/
def pick {α : Type} (d : α) (sel : List Bool) (cands : List α) : α :=
  (sel.foldl (fun cs s => pairs (mux s) cs) cands).headD d

/-- The lane the selector names, `Σ_b sel[b] 2^b`. -/
def LaneIn.l {K nw : ℕ} {P : Type} (x : LaneIn K nw P) : Fin 32 :=
  ⟨bitsVal x.sel, by simpa [LaneIn.sel] using bitsVal_lt x.sel⟩

/-- **`PouwH1LaneZy_v1{PouwH1Tile_v1}`'s `ok`** at a 32 × 32 leaf (`h1zy._lane_body`), at the dequant `dqW` and the
chain's words `chain` (of the public inputs, the lane `sel` names and the rows' codes). -/
def laneOk {K nw : ℕ} {P : Type}
    (chain : P → Fin 32 → (Fin 4 → Fin K → BitVec 8) → (Fin 8 → Fin K → BitVec 8) → Fin nw → BitVec 32)
    (dqW : BitVec 32 → BitVec 32 → BitVec 32 → BitVec 16) (x : LaneIn K nw P) : Bool :=
  let picked : Fin 4 → Fin 8 → BitVec 32 × BitVec 16 := fun s t =>
    pick (0, 0) x.sel (List.ofFn fun ln : Fin 32 => (x.zr (laneRow ln s) (laneCol ln t), x.yr (laneRow ln s) (laneCol ln t)))
  let ok := zyCheck dqW (fun s t => (picked s t).1) (fun s t => (picked s t).2) (fun s t => gemmAccS8 (x.a s) (x.b t))
    x.sa x.sw
  let live := tree orBit false (List.ofFn x.admit)
  let eq := List.ofFn fun w => eq32 (x.partials w) (mask32 (chain x.pub x.l x.a x.b w) live)
  let high := tree orBit false (List.ofFn fun t : Fin 11 => x.lane ⟨5 + t.val, by omega⟩)
  mux high (tree andBit true (ok :: eq)) false

end Pouw.H1Zy
