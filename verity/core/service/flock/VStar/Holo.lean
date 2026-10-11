import VStar.Compose
import VStar.Json
import VStar.Stage

/-!
# V*'s statements over a holo inner session (`rec_algebra.Shape.holo`, `rec_residuals.statement(..., holo=True)`)

A holo session's circuits are private: its rep absorbs `h`, the hidden part's fold (64 elements), in a round of its own after
z_partial's (`rec_algebra.H_ROUND`), and the lincheck reads the comb's operand as `comb[l] + h[l]` (`("combh", l)`). Nothing
else of `run` changes, and `_order` gives `combh` the comb's index, so the residuals' supports are the plain shape's, every
element after z_partial 64 places later. A holo shape's S is therefore the plain shape's (`Algebra.spec`) with

* `h` in w after z_partial: each `w` and `wt` term at or past it 64 elements later, and w 64 elements longer;
* each comb operand `[v comb0 + l]` reading `h[l]` too, `[v comb0 + l, w h0 + l]` (`_build`'s `var_op`);
* `h`'s row port after z_partial's (`rec_residuals.spec`), the later rounds' `m<i>` renumbered.

Its statements are the plain shape's levels and the parts of that S (`statementsHolo`), and each statement's circuit file says
it is over a holo shape in META (`HOLO_KEY`, `compose`'s `extra_meta`), which the verifier's `RecResiduals.check` reads to
select the net. The plain shape's definitions are left as they are.
-/

namespace VStar

open Lean (Json)
open Flock.HmNets

namespace Algebra

/-- h's elements, the comb's. -/
def H_SIZE : Nat := 64

/-- The index in the rounds of a holo shape's h round (`rec_algebra.h_round`): after the first, zerocheck's `m - 6` and
`(fa, fb)`, lincheck's `k_log - 6` and z_partial's. -/
def hRound (sh : Shape) : Nat := (sh.m - 6) + (sh.kLog - 6) + 3

/-- A term of the plain S in the holo S's w, `h0` the first element of h. -/
def holoTerm (h0 : Nat) : Term → Except String Term
  | .w i => pure (.w (if i < h0 then i else i + H_SIZE))
  | .v i => pure (.v i)
  | .wt i u =>
    if i < h0 && h0 < i + 128 then throw s!"holo: a transposed block [wt, {i}, {u}] across h's slot {h0}"
    else pure (.wt (if i < h0 then i else i + H_SIZE) u)

/-- `rec_residuals.spec(Shape(m, k_log, claims, holo=True))`: the plain shape's S with h's round in w after z_partial's, the
comb's operands reading it (the module doc). -/
def specHolo (sh : Shape) : Except String RepS := do
  let S ← spec sh
  let ir := hRound sh
  let nr := S.ports.length - 2
  unless ir ≤ nr && (S.ports.take ir).getLast?.map (·.2) == some 64 do
    throw s!"holo: the plain shape's round {ir - 1} is not z_partial's 64 elements"
  unless S.gf.nv ≥ H_SIZE do throw "holo: no comb in v"
  let h0 := (S.ports.take ir).foldl (· + ·.2) 0
  let comb0 := S.gf.nv - H_SIZE
  let ops ← S.gf.ops.mapM fun o => match o with
    | [.v j] => pure (if comb0 ≤ j then [.v j, .w (h0 + (j - comb0))] else [.v j])
    | o => o.mapM (holoTerm h0)
  let res ← S.gf.res.mapM fun r => do pure { r with add := ← r.add.mapM (holoTerm h0) }
  let rounds := S.ports.take nr
  let sizes := (rounds.take ir).map (·.2) ++ [H_SIZE] ++ (rounds.drop ir).map (·.2)
  let S' : RepS := { gf := { nw := S.gf.nw + H_SIZE, nv := S.gf.nv, ops, res },
                     ports := sizes.zipIdx.map (fun (n, i) => (s!"m{i}", n)) ++ S.ports.drop nr }
  checkRepS S' (whole := true)
  pure S'

end Algebra

/-- V*'s statements over a holo inner session of shape `sh`, in `stage`'s order: `statements`' levels, then one per part of
`Algebra.specHolo sh`. -/
def statementsHolo (sh : Algebra.Shape) : Except String (Array Stmt) := do
  let s ← sh.sched
  let levels := (Array.range s.levels.size).map fun l =>
    let lv := s.levels[l]!
    Stmt.recOpen l (2 ^ lv.lanes) (lv.d - lv.cap) (2 * lv.q)
  let ps ← parts (← Algebra.specHolo sh)
  pure (levels ++ ps.mapIdx fun j p => .inner j s!"rec-residuals/m{sh.m}-k{sh.kLog}-c{sh.claims}-p{j}" p)

/-- V*'s statements over an inner session of shape `sh`, holo or plain. -/
def statementsOf (sh : Algebra.Shape) (holo : Bool) : Except String (Array Stmt) :=
  if holo then statementsHolo sh else statements sh

/-- META's key that says a statement is over a holo shape's parts (`rec_residuals.HOLO_KEY`), written only when true. -/
def HOLO_KEY : String := "holo"

/-- `metaHead`'s head with `extra` added to META (`compose`'s `extra_meta`), none of them a key META has. META is read back
and must be written again as it was, so that the head is `compose`'s with those keys alone added. -/
def withMeta (u : UnitC) (extra : List (String × Json)) (head : String) : Except String String := do
  let pre := s!"{FORMAT_CIRCUIT} {u.name}\nMETA "
  unless head.startsWith pre && head.endsWith "\n" do throw s!"{u.name}: not a circuit file's head"
  let text := String.ofList ((head.toList.drop pre.length).dropLast)
  let m ← Json.parse text
  unless (← pyDumps m) == text do throw s!"{u.name}: META is not written back as it was read"
  for (k, _) in extra do
    if (m.getObjVal? k).isOk then throw s!"extra META: {k} is META's own key"
  pure (pre ++ (← pyDumps (extra.foldl (fun j (k, v) => j.setObjVal! k v) m)) ++ "\n")

/-- `emitCircuit` of a holo shape's statement: META says `holo` (`rec_residuals.statement(..., holo=True)`). -/
def emitCircuitHolo {m : Type → Type} [Monad m] (put : String → m Unit) (u : UnitC) (l : Laid) (sub : Sub) (b : Bindings) :
    Except String (m Unit) := do
  let ud ← unitDigests u l
  let head ← withMeta u [(HOLO_KEY, .bool true)] (← metaHead u l sub b (Flock.hexOf ud.sha256.finish))
  pure (do put head; emitSections put u l)

/-- The circuit file of a statement over a holo or plain shape. -/
def emitCircuitOf {m : Type → Type} [Monad m] (holo : Bool) (put : String → m Unit) (u : UnitC) (l : Laid) (sub : Sub)
    (b : Bindings) : Except String (m Unit) :=
  if holo then emitCircuitHolo put u l sub b else emitCircuit put u l sub b

end VStar
