import VStar.Compose
import VStar.Unit

/-!
# `RecCap`: one node of the circuit table's level-0 cap (`verity_flock.rec_cap`)

`root_B`, the circuit table's level-0 cap of `2^cap₀` nodes (both reps' one top, the inner session's round 0), is held whole
by one statement of V*'s at fixed positions: `rec-cap/n<N>` over `N = 2^cap₀` instances, instance `i` reading node `i` of
`rec-top-L0` as its one input row `top`, the node's registered row (64 bytes as 32 u16 words, then 32 zero words,
`rec_live.top_rows`), through its hm96 slot. Which instance holds which node is the instance's index, not a coin, so every
node of round 0 is held whatever the queries reach, and the level-0 openings, which read `rec-top-L0` at the nodes their
coins name, read that one registration (`RecVStar.checkShared`).

The unit's one output `out` is the row's padding, words 32 to 63, then 32 zero words: V* registers it as zero rows
(`rec-cap-zero`, `RecCoins.checkCap`), so the statement holds exactly where every `top` row is a node's row. Ports, u16
words: `top` 64, `out` 64. No AND: the unit is the hm96 slots' binding of the rows.
-/

namespace VStar

open Flock.HmNets

def CAP_TOP_WORDS : Nat := 64
def CAP_OUT_WORDS : Nat := 64
/-- A node's words in its registered row (`rec_live.top_rows`): 64 bytes. -/
def CAP_NODE_WORDS : Nat := 32

/-- The level-0 cap's statement's unit name (`rec_cap.unit_name`), `N` its nodes. -/
def capName (N : Nat) : String := s!"rec-cap/n{N}"

/-- `N` of `rec-cap/n<N>`, in that exact form. -/
def capParams (name : String) : Option Nat :=
  if !name.startsWith "rec-cap/n" then none else
  match (name.drop 9).toString.toNat? with
  | some N => if capName N == name then some N else none
  | none => none

/-- `rec_cap.subcircuit(N)`: one input `top`, one output `out`. -/
def capSub (N : Nat) : Sub :=
  { id := capName N, template := "rec-cap", inputs := ["top"], outputs := ["out"] }

/-- `rec_cap.unit_circuit`'s forms: `out` is `top`'s words 32 to 63, then 32 zero words. -/
def capForms (top : List Lin) : List Lin :=
  top.drop (16 * CAP_NODE_WORDS) ++ List.replicate (16 * (CAP_OUT_WORDS - (CAP_TOP_WORDS - CAP_NODE_WORDS))) Lin.zero

/-- `rec_cap.lowering(N)`'s unit: `rec_cap.unit_circuit`. -/
def capUnit (N : Nat) : UnitC :=
  let build : M (List Lin) := do
    let top ← input (16 * CAP_TOP_WORDS)
    pure (capForms top)
  let r := build.run St.empty
  { name := capName N, inPorts := [CAP_TOP_WORDS], outPorts := [CAP_OUT_WORDS], st := r.2, outs := r.1.toArray }

end VStar
