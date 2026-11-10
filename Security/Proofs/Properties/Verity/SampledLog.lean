module

public import Proofs.Properties.Verity.Sampled
public import Spot.ChallengerSession
public import Std.Tactic.Do

/-!
# Reading the challenger session's release log as v2's check reads it

The facts the reads of the release log rest on, for `Verity.Sampled.readLog` (`coinsRead`'s line reading: a coin
tree's line its own release, `treeStep`, and the other lines in pairs, a root's line, `rootOf`, and its coin's release)
and for any other read of v2's logs. `countFresh` counts the pairs whose root's line shows the sample's family after the
registration's digest, `chunks` cuts the sample's coins, and a coin tree's line answers no family's frame
(`treeStep_frame`). `logRead` is `coinsRead`'s read, by its own name (`coinsRead_eq_logRead`).

`BEq Json` and `Json.parse` are `partial`, so no proof can read a field off a `Json` comparison v2 makes, or see that a
line v2 accepts parses. A fact about one of the verifier's monadic functions is a Hoare triple proved by `mvcgen`, read
back at an `.ok` or `some` result (`of_triple`, `of_triple_opt`); the triples here hold of any `Except` or `Option`
value, so `mvcgen` passes over a `partial` call it can't unfold.
-/

@[expose] public section

set_option mvcgen.warning false

namespace Verity.Sampled

open Lean (Json)
open Std.Do
open Spot.Challenger

/-! ## Reading the verifier's monadic functions -/

theorem bind_ok {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} :
    (x >>= f) = .ok b ↔ ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x <;> simp [bind, Except.bind]

theorem of_triple {ε α : Type} {x : Except ε α} {Q : α → Prop} {r : α}
    (hT : ⦃⌜True⌝⦄ x ⦃⇓? a => ⌜Q a⌝⦄) (h : x = .ok r) : Q r := by
  subst h; exact hT True.intro

theorem except_spec {ε α : Type} (x : Except ε α) : ⦃⌜True⌝⦄ x ⦃⇓? _ => ⌜True⌝⦄ := by
  cases x <;> exact fun _ => True.intro

theorem throw_spec {ε α : Type} (e : ε) (Q : PostCond α (.except ε .pure)) :
    ⦃Q.2.1 e⦄ (MonadExceptOf.throw e : Except ε α) ⦃Q⦄ := fun h => h

theorem ok_spec {ε α : Type} (x : Except ε α) : ⦃⌜True⌝⦄ x ⦃⇓? a => ⌜x = .ok a⌝⦄ := by
  cases x
  · exact fun _ => True.intro
  · exact fun _ => rfl

theorem mapM_str {f : Json → Flock.V String} (hf : ∀ j s, f j = .ok s → j = .str s) :
    ∀ {l : List Json} {lines : List String}, l.mapM f = .ok lines → l = lines.map .str
  | [], lines, h => by
    simp only [List.mapM_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h; rfl
  | j :: l, lines, h => by
    rw [List.mapM_cons, bind_ok] at h
    obtain ⟨s, hs, h⟩ := h
    rw [bind_ok] at h
    obtain ⟨rest, hr, h⟩ := h
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    rw [hf j s hs, mapM_str hf hr]
    rfl

theorem mapM_str_ok {f : Json → Flock.V String} (hf : ∀ s, f (.str s) = .ok s) :
    ∀ lines : List String, (lines.map Json.str).mapM f = .ok lines
  | [] => rfl
  | s :: lines => by
    rw [List.map_cons, List.mapM_cons, hf s, mapM_str_ok hf lines]
    rfl

theorem mapM_ok_spec {α β : Type} (f : α → Flock.V β) (l : List α) :
    ⦃⌜True⌝⦄ l.mapM f ⦃⇓? r => ⌜l.mapM f = .ok r⌝⦄ := ok_spec _

theorem of_triple_opt {α : Type} {x : Option α} {Q : α → Prop} {r : α}
    (hT : ⦃⌜True⌝⦄ x ⦃⇓? a => ⌜Q a⌝⦄) (h : x = some r) : Q r := by
  subst h; exact hT True.intro

theorem none_spec {α : Type} (Q : PostCond α (.except PUnit .pure)) : ⦃Q.2.1 ⟨⟩⦄ (none : Option α) ⦃Q⦄ := fun h => h

theorem opt_spec {α : Type} (x : Option α) : ⦃⌜True⌝⦄ x ⦃⇓? _ => ⌜True⌝⦄ := by
  cases x <;> exact fun _ => True.intro

theorem specOf_spec (j : Json) : ⦃⌜True⌝⦄ specOf j ⦃⇓? _ => ⌜True⌝⦄ := opt_spec _
theorem toOption_spec {ε α : Type} (x : Except ε α) : ⦃⌜True⌝⦄ x.toOption ⦃⇓? _ => ⌜True⌝⦄ := opt_spec _
theorem nat_spec (j : Json) : ⦃⌜True⌝⦄ Spot.nat? j ⦃⇓? _ => ⌜True⌝⦄ := opt_spec _
theorem findIdx_spec {α : Type} (p : α → Bool) (l : List α) : ⦃⌜True⌝⦄ l.findIdx? p ⦃⇓? _ => ⌜True⌝⦄ := opt_spec _
theorem findIdxA_spec {α : Type} (p : α → Bool) (l : Array α) : ⦃⌜True⌝⦄ l.findIdx? p ⦃⇓? _ => ⌜True⌝⦄ :=
  opt_spec _
theorem getElem?_spec {α : Type} (l : List α) (i : Nat) : ⦃⌜True⌝⦄ l[i]? ⦃⇓? _ => ⌜True⌝⦄ := opt_spec _

/-! ## Counting the sample's coins -/

/-- The pairs of `ls` whose root's line shows family `label` after `after` (`rootOf`). -/
def countFresh (label : String) (after : ByteArray) : List String → Nat
  | a :: _ :: rest => countFresh label after rest + if rootOf a = some (label, after) then 1 else 0
  | _ => 0

/-- The bits of `m` coins of `ξ`, in order. -/
def chunks : Nat → ByteArray → List (List Bool)
  | 0, _ => []
  | m + 1, ξ => Verity.bitsOfBytes (ξ.extract 0 COIN_BYTES) :: chunks m (ξ.extract COIN_BYTES ξ.size)

/-! ## The read with v2's own tests -/

/-- **The device's log as v2's check reads it, with v2's own tests**: `coinsRead`'s read, with each coin tree's line
read by the tests `Spot.Challenger.coinTreesOf` makes, the head's line included (`readLog`), and the
sessions' records as `Spot.OneStageZk.treesHeld` reads them (`recordsOn`). -/
def logRead (s : Verity.Sent) : Option (List Verity.Release) := do
  let (p, c, e) ← (caseOf s).toOption
  let (_, label) ← p.base.challenger
  let ξ ← (_root_.Spot.OneStageZk.coinsOf p c e).toOption
  let lines ← (linesOf e).toOption
  readLog label c.digest (← recordsOn (_root_.Core.Blobs.ofArray s.blobs) e) ξ lines

/-- `logRead` is `coinsRead`. -/
theorem coinsRead_eq_logRead : coinsRead = logRead := rfl

/-- A coin tree's line's release, read with v2's own tests, answers no family's frame. -/
theorem treeStep_frame {records : List Json} {ts ts' : List (String × Json)} {d : Json} {r : Verity.Release}
    (h : treeStep records ts d = some (r, ts')) : ∀ l, r.frame ≠ .fresh l := by
  refine of_triple_opt (Q := fun p => ∀ l, p.1.frame ≠ .fresh l) ?_ h
  clear h
  mvcgen [treeStep, -Spec.throw_Option, none_spec, specOf_spec, toOption_spec, nat_spec, findIdx_spec,
    getElem?_spec, findIdxA_spec]
  all_goals exact fun l h => Verity.Frame.noConfusion h

end Verity.Sampled
