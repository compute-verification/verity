import Proofs.Properties.Verity.Sampled
import SampledProofs.ChallengerSession
import Std.Tactic.Do

/-!
# The challenger session's release log, line by line, as v2's check reads it

The device's log as `SampledProofs.Challenger.releasedRootsOf` reads the challenger session's release log, its head
aside: a coin tree's line (`treeLine`) its own release (`treeRelease`), and the other lines in pairs, a root's line and
its coin's release, of the family and after the root the root's line shows (`rootOf`), with the draw's coins placed in
order on the releases of the draw's family after the registration's digest (`readLines`). `countFresh` counts those
pairs and `chunks` cuts the coins.

Its fields are read only by matching on what `event` parses, and its lines classified only by `treeLine` and by their
position, as v2's check classifies them: `BEq Json` and `Json.parse` are `partial`, so no proof can read a field off a
`Json` comparison v2 makes, or see that a line v2 accepts parses. A fact about one of the verifier's monadic functions
is a Hoare triple proved by `mvcgen`, read back at an `.ok` or `some` result (`of_triple`, `of_triple_opt`); the
triples here hold of any `Except` or `Option` value, so `mvcgen` passes over a `partial` call it can't unfold.
-/

set_option mvcgen.warning false

namespace Verity.Sampled

open Lean (Json)
open Std.Do
open SampledProofs.Challenger

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
theorem nat_spec (j : Json) : ⦃⌜True⌝⦄ SampledProofs.nat? j ⦃⇓? _ => ⌜True⌝⦄ := opt_spec _
theorem findIdx_spec {α : Type} (p : α → Bool) (l : List α) : ⦃⌜True⌝⦄ l.findIdx? p ⦃⇓? _ => ⌜True⌝⦄ := opt_spec _
theorem findIdxA_spec {α : Type} (p : α → Bool) (l : Array α) : ⦃⌜True⌝⦄ l.findIdx? p ⦃⇓? _ => ⌜True⌝⦄ :=
  opt_spec _
theorem getElem?_spec {α : Type} (l : List α) (i : Nat) : ⦃⌜True⌝⦄ l[i]? ⦃⇓? _ => ⌜True⌝⦄ := opt_spec _

/-! ## The read -/

/-- **The family and the root a root line of the release log shows**, as `releasedRootsOf` reads them. -/
def rootOf (l : String) : String × ByteArray :=
  match event l with
  | .ok r => ((match SampledProofs.getD r "label" with | .str x => x | _ => ""),
      match SampledProofs.getD r "root" with | .str x => (Flock.unhex x).toOption.getD .empty | _ => .empty)
  | .error _ => ("", .empty)

/-- **A coin tree's line's release**, after the trees `ts` sampled so far: its hello with its root's and key's bits, or
a round opened, with its coins' bits where a record of its tree gives them. -/
def treeRelease (records : List Json) (ts : List (String × Flock.CoinTree.Spec)) (j : Json) :
    Option (Verity.Release × List (String × Flock.CoinTree.Spec)) :=
  let ev := match SampledProofs.getD j "event" with | .str x => x | _ => ""
  if ev == TREE_EVENT then do
    let .str key := SampledProofs.getD j "key" | none
    let .str nonce := SampledProofs.getD j "nonce" | none
    let .str root := SampledProofs.getD j "root" | none
    let spec ← specOf (SampledProofs.getD j "spec")
    let ν ← (Flock.unhex nonce).toOption
    let r ← (Flock.unhex root).toOption
    let κ ← (Flock.unhex key).toOption
    pure (⟨.hello spec, ν, Verity.bitsOfBytes (r ++ κ)⟩, ts ++ [(root, spec)])
  else if ev == OPEN_EVENT then do
    let .str root := SampledProofs.getD j "tree" | none
    let .str stream := SampledProofs.getD j "stream" | none
    let .str aft := SampledProofs.getD j "after" | none
    let k ← SampledProofs.nat? (SampledProofs.getD j "round")
    let n ← SampledProofs.nat? (SampledProofs.getD j "n")
    let i ← ts.findIdx? (·.1 == root)
    let (_, spec) ← ts[i]?
    let s ← spec.streams.findIdx? (· == stream)
    let b ← (Flock.unhex aft).toOption
    let coins := match records.find? (recordTree · == root) with
      | some rec => ((recordRound rec stream k).map Verity.bitsOfBytes).getD []
      | none => []
    pure (⟨.opening i s n, b, coins⟩, ts)
  else none

/-- **The releases the log's lines `ls` show**, after the release log's head: a coin tree's line (`treeLine`) its
release (`treeRelease`), and the other lines in pairs, as `releasedRootsOf` reads them, a root's line `pending` and
its coin's release, of the family and after the root the root's line shows (`rootOf`), with the next coin of the draw's
`ξ` where it is the draw's (`label`'s, after `after`), else none. -/
def readLines (label : String) (after : ByteArray) (records : List Json) :
    List String → List (String × Flock.CoinTree.Spec) → Option String → ByteArray → List Verity.Release
  | [], _, _, _ => []
  | l :: ls, ts, pending, ξ =>
    if treeLine l then
      match (event l).toOption.bind (treeRelease records ts) with
      | some (r, ts') => r :: readLines label after records ls ts' pending ξ
      | none => readLines label after records ls ts pending ξ
    else
      match pending with
      | none => readLines label after records ls ts (some l) ξ
      | some a =>
        if (rootOf a).1 = label ∧ (rootOf a).2 = after then
          ⟨.fresh label, after, Verity.bitsOfBytes (ξ.extract 0 COIN_BYTES)⟩ ::
            readLines label after records ls ts none (ξ.extract COIN_BYTES ξ.size)
        else ⟨.fresh (rootOf a).1, (rootOf a).2, []⟩ :: readLines label after records ls ts none ξ

/-- The pairs of `ls` whose root's line shows family `label` after `after`. -/
def countFresh (label : String) (after : ByteArray) : List String → Nat
  | a :: _ :: rest => countFresh label after rest + if (rootOf a).1 = label ∧ (rootOf a).2 = after then 1 else 0
  | _ => 0

/-- The bits of `m` coins of `ξ`, in order. -/
def chunks : Nat → ByteArray → List (List Bool)
  | 0, _ => []
  | m + 1, ξ => Verity.bitsOfBytes (ξ.extract 0 COIN_BYTES) :: chunks m (ξ.extract COIN_BYTES ξ.size)

/-! ## What a coin tree's line shows -/

/-- A coin tree's line's release answers no family's frame. -/
theorem treeRelease_frame {records : List Json} {ts ts' : List (String × Flock.CoinTree.Spec)} {j : Json}
    {r : Verity.Release} (h : treeRelease records ts j = some (r, ts')) : ∀ l, r.frame ≠ .fresh l := by
  refine of_triple_opt (Q := fun p => ∀ l, p.1.frame ≠ .fresh l) ?_ h
  clear h
  mvcgen [treeRelease, -Spec.throw_Option, none_spec, specOf_spec, toOption_spec, nat_spec, findIdx_spec,
    getElem?_spec, findIdxA_spec]
  all_goals exact fun l h => Verity.Frame.noConfusion h

end Verity.Sampled
