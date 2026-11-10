module

public import Proofs.Properties.Verity.Sampled
public import SampledProofs.ChallengerSession
public import Std.Tactic.Do

/-!
# Reading the challenger session's release log as v2's check reads it

The facts the reads of the release log rest on, for `Verity.Sampled.readLines` (`coinsRead`'s line reading: a coin
tree's line its own release, `treeRelease`, and the other lines in pairs, a root's line, `rootOf`, and its coin's
release) and for any other read of v2's logs. `countFresh` counts the pairs whose root's line shows the sample's family
after the registration's digest, `chunks` cuts the sample's coins, and a coin tree's line answers no family's frame
(`treeRelease_frame`).

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

/-! ## Counting the sample's coins -/

/-- The pairs of `ls` whose root's line shows family `label` after `after` (`rootOf`). -/
def countFresh (label : String) (after : ByteArray) : List String → Nat
  | a :: _ :: rest => countFresh label after rest + if rootOf a = some (label, after) then 1 else 0
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

/-! ## The read with v2's own tests -/

/-- The streams a coin tree's spec names, as `SampledProofs.Challenger.coinTreesOf` reads them: none where it names no
list. -/
def specStreams (spec : Json) : Array Json :=
  match SampledProofs.getD spec "streams" with | .arr ss => ss | _ => #[]

/-- **A coin tree's line's release, read with the tests `SampledProofs.Challenger.coinTreesOf` makes**, after the trees
`ts` read so far, each its root and its spec as its line gives them: where the line's `event` is `TREE_EVENT` by
`coinTreesOf`'s comparison (`jsonStrIs`), its hello after its nonce, with its root's and key's bits; else a round
opened, of the `i`-th tree read and the `s`-th of its spec's streams, the first one `jsonStrIs` finds equal to the
line's, as `coinTreesOf`'s `any` tests them, after the bytes it answers, with its coins' bits where a record of that
tree gives them, else none; `none` where the line doesn't read as either. -/
def treeStep (records : List Json) (ts : List (String × Json)) (d : Json) :
    Option (Verity.Release × List (String × Json)) :=
  if SampledProofs.jsonStrIs (SampledProofs.getD d "event") TREE_EVENT then do
    let .str key := SampledProofs.getD d "key" | none
    let .str nonce := SampledProofs.getD d "nonce" | none
    let .str root := SampledProofs.getD d "root" | none
    let spec ← specOf (SampledProofs.getD d "spec")
    let ν ← (Flock.unhex nonce).toOption
    let r ← (Flock.unhex root).toOption
    let κ ← (Flock.unhex key).toOption
    pure (⟨.hello spec, ν, Verity.bitsOfBytes (r ++ κ)⟩, ts ++ [(root, SampledProofs.getD d "spec")])
  else do
    let .str root := SampledProofs.getD d "tree" | none
    let .str stream := SampledProofs.getD d "stream" | none
    let .str aft := SampledProofs.getD d "after" | none
    let k ← SampledProofs.nat? (SampledProofs.getD d "round")
    let n ← SampledProofs.nat? (SampledProofs.getD d "n")
    let i ← ts.findIdx? (·.1 == root)
    let (_, spec) ← ts[i]?
    let s ← (specStreams spec).findIdx? (SampledProofs.jsonStrIs · stream)
    let b ← (Flock.unhex aft).toOption
    let coins := match records.find? (recordTree · == root) with
      | some rec => ((recordRound rec stream k).map Verity.bitsOfBytes).getD []
      | none => []
    pure (⟨.opening i s n, b, coins⟩, ts)

/-- **The releases the lines after the release log's head show**, read as `readLines` reads them but each coin tree's
line by `treeStep`, after the trees `ts` read so far: for the sample of family `label` after `after`, with the sample's
coins `ξ` still to place and the root's line `pending` still to pair. -/
def readTail (label : String) (after : ByteArray) (records : List Json) :
    List String → List (String × Json) → Option String → ByteArray → Option (List Verity.Release)
  | [], _, none, _ => some []
  | [], _, some _, _ => none
  | l :: ls, ts, pending, ξ =>
    if treeLine l then do
      let (r, ts') ← (event l).toOption.bind (treeStep records ts)
      pure (r :: (← readTail label after records ls ts' pending ξ))
    else
      match pending with
      | none => readTail label after records ls ts (some l) ξ
      | some a => do
        let (lab, root) ← rootOf a
        if lab = label ∧ root = after then
          let m := COIN_BYTES
          pure (⟨.fresh label, after, Verity.bitsOfBytes (ξ.extract 0 m)⟩ ::
            (← readTail label after records ls ts none (ξ.extract m ξ.size)))
        else pure (⟨.fresh lab, root, []⟩ :: (← readTail label after records ls ts none ξ))

/-- **The releases the release log's lines show, read with v2's own tests**: its head's coin tree where the head is a
coin tree's line (`SampledProofs.Challenger.coinTreesOf` reads the head's line too), then the lines after it
(`readTail`). -/
def readLog (label : String) (after : ByteArray) (records : List Json) (ξ : ByteArray) :
    List String → Option (List Verity.Release)
  | [] => some []
  | l :: ls =>
    if treeLine l then do
      let (r, ts) ← (event l).toOption.bind (treeStep records [])
      pure (r :: (← readTail label after records ls ts none ξ))
    else readTail label after records ls [] none ξ

/-- A session's record on the blobs `bs`, as `SampledProofs.OneStageZk.treesHeld` reads it: one that doesn't parse is
`null`; `none` where no blob holds it or it isn't UTF-8. -/
def recordOn (bs : _root_.Core.Blobs) (r : _root_.SampledProofs.OneStageZk.RunFiles) : Option Json := do
  let b ← (bs.get r.record s!"session {r.name}'s record").toOption
  let t ← String.fromUTF8? b
  pure ((Flock.JsonParse.parse t).toOption.getD Json.null)

/-- The records of the sessions v2's evidence `e` names, on the blobs `bs`, as `SampledProofs.OneStageZk.treesHeld`
reads them (`recordOn`). -/
def recordsOn (bs : _root_.Core.Blobs) (e : _root_.SampledProofs.OneStageZk.Evidence) : Option (List Json) :=
  e.sessions.toList.mapM (recordOn bs)

/-- **The device's log as v2's check reads it, with v2's own tests**: `coinsRead`'s read, with each coin tree's line
read by the tests `SampledProofs.Challenger.coinTreesOf` makes, the head's line included (`readLog`), and the
sessions' records as `SampledProofs.OneStageZk.treesHeld` reads them (`recordsOn`). -/
def logRead (s : Verity.Sent) : Option (List Verity.Release) := do
  let (p, c, e) ← (caseOf s).toOption
  let (_, label) ← p.base.challenger
  let ξ ← (_root_.SampledProofs.OneStageZk.coinsOf p c e).toOption
  let lines ← (linesOf e).toOption
  readLog label c.digest (← recordsOn (_root_.Core.Blobs.ofArray s.blobs) e) ξ lines

/-- A coin tree's line's release, read with v2's own tests, answers no family's frame. -/
theorem treeStep_frame {records : List Json} {ts ts' : List (String × Json)} {d : Json} {r : Verity.Release}
    (h : treeStep records ts d = some (r, ts')) : ∀ l, r.frame ≠ .fresh l := by
  refine of_triple_opt (Q := fun p => ∀ l, p.1.frame ≠ .fresh l) ?_ h
  clear h
  mvcgen [treeStep, -Spec.throw_Option, none_spec, specOf_spec, toOption_spec, nat_spec, findIdx_spec,
    getElem?_spec, findIdxA_spec]
  all_goals exact fun l h => Verity.Frame.noConfusion h

end Verity.Sampled
