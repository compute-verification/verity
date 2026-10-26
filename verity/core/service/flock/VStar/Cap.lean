import VStar.Compose
import VStar.Unit

/-!
# `RecCap`: one node of a Ligerito level's committed tops (`verity_flock.rec_cap`)

`root_B`, the circuit table's level-0 cap of `2^cap₀` nodes (both reps' one top, the inner session's round 0), is held whole
by one statement of V*'s at fixed positions: `rec-cap/n<N>` over `N = 2^cap₀` instances, instance `i` reading node `i` of
`rec-top-L0` as its one input row `top`, the node's registered row (64 bytes as 32 u16 words, then 32 zero words,
`rec_live.top_rows`), through its hm96 slot. Which instance holds which node is the instance's index, not a coin, so every
node of round 0 is held whatever the queries reach, and the level-0 openings, which read `rec-top-L0` at the nodes their
coins name, read that one registration (`RecVStar.checkShared`).

Each later level `l` is held the same way by `rec-cap/n<N>-l<l>`, `N = 2 · 2^cap_l`: instance `i` reads row `i` of
`rec-top-L<l>`, rep 0's level-`l` top then rep 1's (`RecVStar.topRow`), which each rep commits in its level-`l − 1` round,
before that level's query coins.

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

/-- A level's tag in its cap's name: none at level 0, `-l<l>` at a later level. -/
def capLevelTag : Nat → String
  | 0 => ""
  | l + 1 => s!"-l{l + 1}"

/-- The cap's statement's unit name (`rec_cap.unit_name`) at Ligerito level `l`, `N` its nodes: `rec-cap/n<N>` at level 0,
`rec-cap/n<N>-l<l>` at a later level. -/
def capName (l N : Nat) : String := s!"rec-cap/n{N}{capLevelTag l}"

/-- `(l, N)` of `rec-cap/n<N>` (level 0) or `rec-cap/n<N>-l<l>` (`l ≥ 1`), in that exact form. -/
def capParams (name : String) : Option (Nat × Nat) :=
  if !name.startsWith "rec-cap/n" then none else
  match (name.drop 9).toString.splitOn "-l" with
  | [n] => match n.toNat? with
    | some N => if capName 0 N == name then some (0, N) else none
    | none => none
  | [n, l] => match n.toNat?, l.toNat? with
    | some N, some l => if capName l N == name then some (l, N) else none
    | _, _ => none
  | _ => none

/-- `rec_cap.subcircuit(l, N)`: one input `top`, one output `out`. -/
def capSub (l N : Nat) : Sub :=
  { id := capName l N, template := "rec-cap", inputs := ["top"], outputs := ["out"] }

/-- `rec_cap.unit_circuit`'s forms: `out` is `top`'s words 32 to 63, then 32 zero words. -/
def capForms (top : List Lin) : List Lin :=
  top.drop (16 * CAP_NODE_WORDS) ++ List.replicate (16 * (CAP_OUT_WORDS - (CAP_TOP_WORDS - CAP_NODE_WORDS))) Lin.zero

/-- `rec_cap.lowering(l, N)`'s unit: `rec_cap.unit_circuit`. -/
def capUnit (l N : Nat) : UnitC :=
  let build : M (List Lin) := do
    let top ← input (16 * CAP_TOP_WORDS)
    pure (capForms top)
  let r := build.run St.empty
  { name := capName l N, inPorts := [CAP_TOP_WORDS], outPorts := [CAP_OUT_WORDS], st := r.2, outs := r.1.toArray }

end VStar
