import Batteries.Tactic.OpenPrivate
import Proofs.Properties.Verity.SampledDraw
import Std.Tactic.Do

/-!
# The device's log as v2's check reads it, and the draw it shows

A read of the device's log from what the prover sent (`readLog`) that shows the draw wherever v2's case accepts in an
accepted composition (`drawRead_readLog`, `DrawRead`), as does any read that agrees with it (`drawRead_of_eq`). It reads the challenger session's release log in v2's evidence
as v2's check does (`SampledProofs.Challenger.releasedRootsOf`): the head aside, a coin tree's line (`treeLine`) its
own release (`treeRelease`), and the other lines in pairs, a root's line and its coin's release, of the family and
after the root the root's line shows (`rootOf`). The draw's coins `ξ` (`SampledProofs.OneStageZk.coinsOf`) are placed in
order on the releases of the draw's family after the registration's digest, one coin of `COIN_BYTES` bytes each
(`readLines`).

Its fields are read only by matching on what `event` parses, and its lines classified only by `treeLine` and by their
position, as v2's check classifies them: `BEq Json` and `Json.parse` are `partial`, so no proof can read a field off a
`Json` comparison v2 makes, or see that a line v2 accepts parses.

* **v2's challenger** (`drawCoins_ok`, `trustedSessionOf_ok`, `sessionNamed_ok`, `sessionOf_ok`, `releasedRootsOf_ok`,
  `challengerStream_ok`): where v2 reads `ξ`, the session's log is a list of lines, the roots of the draw's family are
  those its pairs' root lines show (`sel`), and `ξ` is `COIN_BYTES` bytes for each of them equal to the digest.
* **What the read shows** (`fresh_readLines`, `countFresh_eq`, `chunks_flatten`): the read's coins of the draw's family
  after the digest are `ξ`'s coins, one for each such root, so their bits are `ξ`'s.
* **v2's case** (`caseOf_of`): in an accepted composition the case `caseOf` finds under v2's id is the one v2 accepted,
  since the composite's components have distinct ids (`components_nodup`).

Each fact about one of the challenger's monadic functions is a Hoare triple proved by `mvcgen`, read back at an `.ok`
result (`of_triple`).
-/

set_option mvcgen.warning false

open private pairs from SampledProofs.ChallengerSession

namespace Verity.Sampled

open Lean (Json)
open Std.Do
open SampledProofs.Challenger
open Properties.Challenger (bitsOf Draws)

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


/-- The records of the sessions v2's evidence `e` names, on the blobs sent, as `coinsRead` reads them. -/
def recordsOf (s : Verity.Sent) (e : _root_.SampledProofs.OneStageZk.Evidence) : List Json :=
  e.sessions.toList.filterMap fun r => do
    let b ← ((_root_.Core.Blobs.ofArray s.blobs).get r.record "record").toOption
    (Lean.Json.parse (← String.fromUTF8? b)).toOption

/-- **The device's log as v2's check reads it** from what the prover sent: the releases the challenger session's log in
v2's evidence shows after its head (`readLines`), at the parameters' challenger family and the registration's digest,
with the coins v2's draw reads (`SampledProofs.OneStageZk.coinsOf`) and the records of the evidence's sessions on the
blobs sent (`recordsOf`). -/
def readLog (s : Verity.Sent) : Option (List Verity.Release) := do
  let (p, c, e) ← (caseOf s).toOption
  let (_, label) ← p.base.challenger
  let ξ ← (_root_.SampledProofs.OneStageZk.coinsOf p c e).toOption
  let lines ← (linesOf e).toOption
  pure (readLines label c.digest (recordsOf s e) lines.tail [] none ξ)

/-! ## v2's challenger -/

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

/-! ### The session's files and the draw's coins -/

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

theorem recordReason_spec (key : ChallengerKey) (record : Json) :
    ⦃⌜True⌝⦄ recordReason key record ⦃⇓? _ => ⌜True⌝⦄ := except_spec _

theorem recordOfJson_spec (j : Json) : ⦃⌜True⌝⦄ ChallengerRecord.ofJson j ⦃⇓? _ => ⌜True⌝⦄ := except_spec _

theorem closeMessage_spec (I : ByteArray) (host : String) (c : LiveCommitment) (session : String) (lines : Nat)
    (chain : ByteArray) (ended : Option String) :
    ⦃⌜True⌝⦄ closeMessage I host c session lines chain ended ⦃⇓? _ => ⌜True⌝⦄ := except_spec _

theorem mapM_ok_spec {α β : Type} (f : α → Flock.V β) (l : List α) :
    ⦃⌜True⌝⦄ l.mapM f ⦃⇓? r => ⌜l.mapM f = .ok r⌝⦄ := ok_spec _

theorem releasedRootsOf_spec (c : LiveCommitment) (l : List String) :
    ⦃⌜True⌝⦄ releasedRootsOf c l ⦃⇓? r => ⌜releasedRootsOf c l = .ok r⌝⦄ := ok_spec _

theorem logStr (j : Json) (s : String)
    (h : (match j with | .str s => pure s | _ => throw LOG_FORM : Flock.V String) = .ok s) : j = .str s := by
  cases j <;> first | (cases h; rfl) | cases h

set_option maxHeartbeats 1000000 in
theorem sessionOf_ok {key : ChallengerKey} {record log close : Json} {r : LiveCommitment × Roots × String}
    (h : sessionOf key record log close = .ok r) :
    ∃ ls lines, log = .arr ls ∧ ls.toList = lines.map .str ∧ releasedRootsOf r.1 lines = .ok r.2.1 := by
  refine of_triple (Q := fun r => ∃ ls lines, log = .arr ls ∧ ls.toList = lines.map .str ∧
    releasedRootsOf r.1 lines = .ok r.2.1) ?_ h
  clear h
  mvcgen [sessionOf, -Spec.throw_Except, throw_spec, recordReason_spec, recordOfJson_spec, closeMessage_spec,
    mapM_ok_spec, releasedRootsOf_spec]
  exact ⟨_, _, rfl, mapM_str (fun j s h => logStr j s h) ‹_›, ‹_›⟩

theorem sessionOf_spec (key : ChallengerKey) (record log close : Json) :
    ⦃⌜True⌝⦄ sessionOf key record log close ⦃⇓? r => ⌜sessionOf key record log close = .ok r⌝⦄ := ok_spec _

theorem hellos_spec {α β : Type} (f : α → Flock.V β) (l : List α) : ⦃⌜True⌝⦄ l.mapM f ⦃⇓? _ => ⌜True⌝⦄ :=
  except_spec _

set_option maxHeartbeats 1000000 in
theorem sessionNamed_ok {a : Audited} {key : ChallengerKey} {session : Json} {r : LiveCommitment × Roots}
    (h : sessionNamed a key session = .ok r) :
    ∃ sid, sessionOf key (SampledProofs.getD session "record") (SampledProofs.getD session "log")
      (SampledProofs.getD session "close") = .ok (r.1, r.2, sid) := by
  refine of_triple (Q := fun r => ∃ sid, sessionOf key (SampledProofs.getD session "record")
    (SampledProofs.getD session "log") (SampledProofs.getD session "close") = .ok (r.1, r.2, sid)) ?_ h
  clear h
  mvcgen [sessionNamed, -Spec.throw_Except, throw_spec, sessionOf_spec, recordOfJson_spec, hellos_spec]
  all_goals exact ⟨_, ‹_›⟩

theorem trustedSessionOf_ok {ledger window : Json} {key : ChallengerKey} {pins main session : Json}
    {r : LiveCommitment × Roots} (h : trustedSessionOf ledger window key pins main session = .ok r) :
    ∃ a, sessionNamed a key session = .ok r := by
  unfold trustedSessionOf at h
  rw [bind_ok] at h
  obtain ⟨a, -, h⟩ := h
  exact ⟨a, h⟩

theorem trustedSessionOf_spec (ledger window : Json) (key : ChallengerKey) (pins main session : Json) :
    ⦃⌜True⌝⦄ trustedSessionOf ledger window key pins main session
      ⦃⇓? r => ⌜trustedSessionOf ledger window key pins main session = .ok r⌝⦄ := ok_spec _

theorem challengerStream_spec (c : LiveCommitment) (segment : Json) (after : ByteArray)
    (roots : Option (List ByteArray)) :
    ⦃⌜True⌝⦄ challengerStream c segment after roots ⦃⇓? r => ⌜challengerStream c segment after roots = .ok r⌝⦄ :=
  ok_spec _

/-- The roots of family `label` in the roots `roots`, as `drawCoins` reads them. -/
def look (roots : Roots) (label : String) : List ByteArray := ((roots.find? (·.1 == label)).map (·.2)).getD []

theorem drawCoins_ok {key : ChallengerKey} {w label : String} {after : ByteArray} {coins : Json} {ξ : ByteArray}
    (h : drawCoins key w label after coins = .ok ξ) :
    ∃ cm roots, trustedSessionOf (SampledProofs.getD coins "ledger") (Json.mkObj [("root", Json.str w)]) key
        (SampledProofs.getD coins "pins") (SampledProofs.getD coins "main") (SampledProofs.getD coins "session") =
          .ok (cm, roots) ∧
      challengerStream cm (SampledProofs.getD coins "segment") after (some (look roots label)) = .ok ξ := by
  refine of_triple (Q := fun ξ => ∃ cm roots, trustedSessionOf (SampledProofs.getD coins "ledger")
    (Json.mkObj [("root", Json.str w)]) key (SampledProofs.getD coins "pins") (SampledProofs.getD coins "main")
      (SampledProofs.getD coins "session") = .ok (cm, roots) ∧
    challengerStream cm (SampledProofs.getD coins "segment") after (some (look roots label)) = .ok ξ) ?_ h
  clear h
  mvcgen [drawCoins, -Spec.throw_Except, throw_spec, trustedSessionOf_spec, challengerStream_spec]
  all_goals exact fun h => ⟨_, _, ‹_›, h⟩

theorem liveDomain_spec (s : String) (e : Nat) (fs : List (String × Nat)) :
    ⦃⌜True⌝⦄ liveDomain s e fs ⦃⇓? _ => ⌜True⌝⦄ := except_spec _

theorem coin_size {a b c d : Nat} (h : ¬(a != c || b != d) = true) : a = c := by
  simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, Classical.not_not] at h
  exact h.1

theorem liveReason_spec (c : LiveCommitment) (o : LiveOpening) (label : String) (index : Int) (after : ByteArray) :
    ⦃⌜True⌝⦄ liveReason c o label index after ⦃⇓? r => ⌜r = none → o.coin.size = COIN_BYTES⌝⦄ := by
  mvcgen [liveReason, -Spec.throw_Except, throw_spec, liveDomain_spec]
  all_goals first
    | exact fun h => by cases h
    | exact coin_size ‹_›

theorem openingOfJson_spec (j : Json) : ⦃⌜True⌝⦄ LiveOpening.ofJson j ⦃⇓? _ => ⌜True⌝⦄ := except_spec _

theorem step_size {b coin : ByteArray} {n : Nat} {x : Option String} (hx : ∀ why, x = some why → False)
    (hc : x = none → coin.size = COIN_BYTES) (hb : b.size = COIN_BYTES * n) :
    (b ++ coin).size = COIN_BYTES * (n + 1) := by
  have hn : x = none := by
    cases x with
    | none => rfl
    | some w => exact (hx w rfl).elim
  rw [ByteArray.size_append, hb, hc hn, Nat.mul_succ]

theorem zipIdx_filter_length (l : List ByteArray) (a : ByteArray) :
    ((l.zipIdx.filter (fun x => x.1 == a)).map (·.2)).length = (l.filter (· == a)).length := by
  rw [List.length_map]
  conv => rhs; rw [← List.zipIdx_map_fst 0 l]
  rw [List.filter_map, List.length_map]
  rfl

theorem stream_size {r : ByteArray} {rs : List ByteArray} {after : ByteArray} {start n m : Nat}
    (h1 : (((rs.zipIdx.filter (fun x => x.1 == after)).map (·.2)) == List.range' start n) = true)
    (h2 : r.size = COIN_BYTES * m) (hm : m = n) : r.size = COIN_BYTES * (rs.filter (· == after)).length := by
  rw [← zipIdx_filter_length, eq_of_beq h1, List.length_range', h2, hm]

set_option maxHeartbeats 1000000 in
/-- **The coins the stream opens**: 64 bytes each, as many as the roots of the family the log shows released after
`after`. -/
theorem challengerStream_ok {c : LiveCommitment} {segment : Json} {after : ByteArray} {rs : List ByteArray}
    {ξ : ByteArray} (h : challengerStream c segment after (some rs) = .ok ξ) :
    ξ.size = COIN_BYTES * (rs.filter (· == after)).length := by
  refine of_triple (Q := fun ξ => ξ.size = COIN_BYTES * (rs.filter (· == after)).length) ?_ h
  clear h
  mvcgen [challengerStream, -Spec.throw_Except, throw_spec, openingOfJson_spec, liveReason_spec] invariants
  · ⇓? ⟨cur, out⟩ => ⌜out.size = COIN_BYTES * cur.prefix.length⌝
  all_goals first
    | (rw [List.length_append, List.length_singleton]; exact step_size ‹_› ‹_› ‹_›)
    | exact stream_size ‹_› ‹_› (by simp)

/-! ### The roots the release log shows -/

/-- The roots of family `label` the root lines of the pairs `ps` show, in order. -/
def sel (label : String) (ps : List (String × String)) : List ByteArray :=
  ps.filterMap fun x => if (rootOf x.1).1 = label then some (rootOf x.1).2 else none

/-- The roots of family `label` in `releasedRootsOf`'s state. -/
def lookA (roots : Array (String × Array ByteArray)) (label : String) : List ByteArray :=
  ((roots.find? (·.1 == label)).map (·.2.toList)).getD []

theorem rootOf_eq {a : String} {r : Json} {label root : String} {bytes : ByteArray} (he : event a = .ok r)
    (hl : SampledProofs.getD r "label" = .str label) (hr : SampledProofs.getD r "root" = .str root)
    (hb : (Flock.unhex root).toOption = some bytes) : rootOf a = (label, bytes) := by
  unfold rootOf
  rw [he]
  simp only [hl, hr, hb]
  rfl

theorem look_modify {bytes : ByteArray} {g : String × Array ByteArray → String × Array ByteArray}
    (hg : ∀ x, g x = (x.1, x.2.push bytes)) (label ℓ : String) :
    ∀ (l : List (String × Array ByteArray)) (k : Nat), l.findIdx? (·.1 == label) = some k →
      (((l.modify k g).find? (·.1 == ℓ)).map (·.2.toList)).getD [] =
        (((l.find? (·.1 == ℓ)).map (·.2.toList)).getD []) ++ (if label = ℓ then [bytes] else [])
  | [], k, h => by simp at h
  | x :: l, k, h => by
    rw [List.findIdx?_cons] at h
    by_cases hx : x.1 = label
    · simp only [hx, beq_self_eq_true, ite_true, Option.some.injEq] at h
      subst h
      rw [List.modify_zero_cons, hg, List.find?_cons, List.find?_cons]
      by_cases hℓ : label = ℓ
      · subst hℓ
        simp [hx]
      · have : (x.1 == ℓ) = false := by simp [hx, hℓ]
        simp [this, hℓ]
    · have hb : (x.1 == label) = false := by simp [hx]
      rw [hb] at h
      simp only [Bool.false_eq_true, ite_false] at h
      obtain ⟨k', hk', rfl⟩ := Option.map_eq_some_iff.1 h
      rw [List.modify_succ_cons, List.find?_cons, List.find?_cons]
      by_cases hℓ : x.1 = ℓ
      · have : label ≠ ℓ := fun e => hx (hℓ.trans e.symm)
        simp [hℓ, this]
      · have : (x.1 == ℓ) = false := by simp [hℓ]
        rw [this]
        exact look_modify hg label ℓ l k' hk'

theorem findIdx?_array {xs : Array (String × Array ByteArray)} {p : String × Array ByteArray → Bool} :
    xs.findIdx? p = xs.toList.findIdx? p := by
  rcases xs with ⟨l⟩
  exact List.findIdx?_toArray _ _

theorem roots_step {roots : Array (String × Array ByteArray)} {pre : List (String × String)} {x : String × String}
    {r : Json} {label root : String} {bytes : ByteArray} {k : Nat}
    {g : String × Array ByteArray → String × Array ByteArray} (hg : ∀ x, g x = (x.1, x.2.push bytes))
    (hI : ∀ ℓ, lookA roots ℓ = sel ℓ pre) (he : event x.1 = .ok r) (hl : SampledProofs.getD r "label" = .str label)
    (hk : roots.findIdx? (·.1 == label) = some k) (hr : SampledProofs.getD r "root" = .str root)
    (hb : (Flock.unhex root).toOption = some bytes) :
    ∀ ℓ, lookA (roots.modify k g) ℓ = sel ℓ (pre ++ [x]) := by
  intro ℓ
  rw [findIdx?_array] at hk
  have e := look_modify hg label ℓ roots.toList k hk
  rw [← Array.toList_modify, Array.find?_toList, Array.find?_toList] at e
  unfold lookA
  rw [e, ← lookA, hI ℓ]
  unfold sel
  simp only [List.filterMap_append, List.filterMap_cons, List.filterMap_nil, rootOf_eq he hl hr hb]
  by_cases hℓ : label = ℓ <;> simp [hℓ]

theorem lookA_init (fs : List (String × Nat)) (ℓ : String) :
    lookA (fs.map fun x => ((x.1, #[]) : String × Array ByteArray)).toArray ℓ = [] := by
  unfold lookA
  rw [← Array.find?_toList, List.toList_toArray, List.find?_map]
  cases List.find? _ fs <;> rfl

theorem look_toList (roots : Array (String × Array ByteArray)) (ℓ : String) :
    look (roots.toList.map fun (l, rs) => (l, rs.toList)) ℓ = lookA roots ℓ := by
  unfold look lookA
  rw [List.find?_map, Array.find?_toList, Option.map_map]
  rfl

theorem roots_post {r : Array (String × Array ByteArray)} {ps : List (String × String)}
    (h : ∀ ℓ, lookA r ℓ = sel ℓ ps) : ∀ ℓ, look (r.toList.map fun x => (x.1, x.2.toList)) ℓ = sel ℓ ps :=
  fun ℓ => (look_toList r ℓ).trans (h ℓ)

theorem event_spec (l : String) : ⦃⌜True⌝⦄ event l ⦃⇓? r => ⌜event l = .ok r⌝⦄ := ok_spec _

set_option maxHeartbeats 2000000 in
/-- **The roots the release log shows, by family**: those of the root lines of its pairs, its head and its coin trees'
lines aside. -/
theorem releasedRootsOf_ok {c : LiveCommitment} {log : List String} {roots : Roots}
    (h : releasedRootsOf c log = .ok roots) :
    ∀ ℓ, look roots ℓ = sel ℓ (pairs (log.tail.filter fun l => !treeLine l)) := by
  refine of_triple (Q := fun roots => ∀ ℓ, look roots ℓ = sel ℓ (pairs (log.tail.filter fun l => !treeLine l))) ?_ h
  clear h
  mvcgen [releasedRootsOf, -Spec.throw_Except, throw_spec, event_spec] invariants
  · ⇓? ⟨cur, roots⟩ => ⌜∀ ℓ, lookA roots ℓ = sel ℓ cur.prefix⌝
  all_goals first
    | exact roots_step (fun _ => rfl) ‹_› ‹_› ‹_› ‹_› ‹_› ‹_›
    | exact fun ℓ => lookA_init _ ℓ
    | exact roots_post ‹_›

/-! ## What the read shows -/

/-- The pairs of `ls` whose root's line shows family `label` after `after`. -/
def countFresh (label : String) (after : ByteArray) : List String → Nat
  | a :: _ :: rest => countFresh label after rest + if (rootOf a).1 = label ∧ (rootOf a).2 = after then 1 else 0
  | _ => 0

/-- The bits of `m` coins of `ξ`, in order. -/
def chunks : Nat → ByteArray → List (List Bool)
  | 0, _ => []
  | m + 1, ξ => Verity.bitsOfBytes (ξ.extract 0 COIN_BYTES) :: chunks m (ξ.extract COIN_BYTES ξ.size)

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

/-- A coin tree's line's release answers no family's frame. -/
theorem treeRelease_frame {records : List Json} {ts ts' : List (String × Flock.CoinTree.Spec)} {j : Json}
    {r : Verity.Release} (h : treeRelease records ts j = some (r, ts')) : ∀ l, r.frame ≠ .fresh l := by
  refine of_triple_opt (Q := fun p => ∀ l, p.1.frame ≠ .fresh l) ?_ h
  clear h
  mvcgen [treeRelease, -Spec.throw_Option, none_spec, specOf_spec, toOption_spec, nat_spec, findIdx_spec,
    getElem?_spec, findIdxA_spec]
  all_goals exact fun l h => Verity.Frame.noConfusion h

theorem freshCoins_cons_tree {records : List Json} {ts ts' : List (String × Flock.CoinTree.Spec)} {j : Json}
    {r : Verity.Release} (h : treeRelease records ts j = some (r, ts')) (label : String) (after : ByteArray)
    (rest : List Verity.Release) : freshCoins label after (r :: rest) = freshCoins label after rest := by
  have hf := treeRelease_frame h
  unfold freshCoins
  rw [List.filterMap_cons]
  cases hr : r.frame with
  | fresh l => exact absurd hr (hf l)
  | hello _ => rfl
  | opening _ _ _ => rfl

theorem fresh_readLines (label : String) (after : ByteArray) (records : List Json) :
    ∀ (ls : List String) (ts : List (String × Flock.CoinTree.Spec)) (pending : Option String) (ξ : ByteArray),
      freshCoins label after (readLines label after records ls ts pending ξ) =
        chunks (countFresh label after (pending.toList ++ ls.filter fun l => !treeLine l)) ξ
  | [], ts, pending, ξ => by cases pending <;> rfl
  | l :: ls, ts, pending, ξ => by
    unfold readLines
    by_cases ht : treeLine l = true
    · rw [if_pos ht, List.filter_cons_of_neg (by simp [ht])]
      split
      · rw [freshCoins_cons_tree (Option.bind_eq_some_iff.1 ‹_›).choose_spec.2]
        exact fresh_readLines label after records ls _ pending ξ
      · exact fresh_readLines label after records ls ts pending ξ
    · rw [if_neg ht, List.filter_cons_of_pos (by simp [ht])]
      cases pending with
      | none => exact fresh_readLines label after records ls ts (some l) ξ
      | some a =>
        simp only [Option.toList_some, List.cons_append, List.nil_append]
        by_cases hm : (rootOf a).1 = label ∧ (rootOf a).2 = after
        · rw [if_pos hm, countFresh, if_pos hm]
          simp only [freshCoins, List.filterMap_cons, and_self, ite_true]
          rw [chunks]
          congr 1
          exact fresh_readLines label after records ls ts none _
        · rw [if_neg hm, countFresh, if_neg hm, Nat.add_zero]
          simp only [freshCoins, List.filterMap_cons, if_neg hm]
          exact fresh_readLines label after records ls ts none ξ


theorem bit_cons_lt (x : UInt8) (l : List UInt8) {t : Nat} (ht : t < 8) :
    Flock.Hm96.bit ⟨⟨x :: l⟩⟩ t = x.toNat.testBit t := by
  unfold Flock.Hm96.bit
  rw [Nat.div_eq_of_lt ht, Nat.mod_eq_of_lt ht, show (⟨⟨x :: l⟩⟩ : ByteArray).get! 0 = x from rfl,
    Nat.testBit_eq_decide_div_mod_eq, Nat.shiftRight_eq_div_pow]
  generalize x.toNat / 2 ^ t % 2 = r
  by_cases h : r = 1 <;> simp [h]

theorem bit_cons_add (x : UInt8) (l : List UInt8) (t : Nat) :
    Flock.Hm96.bit ⟨⟨x :: l⟩⟩ (8 + t) = Flock.Hm96.bit ⟨⟨l⟩⟩ t := by
  unfold Flock.Hm96.bit
  rw [show (8 + t) / 8 = t / 8 + 1 by omega, show (8 + t) % 8 = t % 8 by omega]
  rw [show (⟨⟨x :: l⟩⟩ : ByteArray).get! (t / 8 + 1) = (⟨⟨l⟩⟩ : ByteArray).get! (t / 8) by
    show (x :: l).toArray[t / 8 + 1]! = l.toArray[t / 8]!
    simp]

theorem bitsOfBytes_list : ∀ l : List UInt8, Verity.bitsOfBytes ⟨⟨l⟩⟩ = l.flatMap byteBits
  | [] => rfl
  | x :: l => by
    have e := bitsOfBytes_list l
    unfold Verity.bitsOfBytes at e ⊢
    rw [show (⟨⟨x :: l⟩⟩ : ByteArray).size = 8 / 8 + l.length by simp [ByteArray.size]; omega] at *
    rw [show 8 * (8 / 8 + l.length) = 8 + 8 * l.length by omega, List.range_add, List.map_append, List.flatMap_cons,
      List.map_map]
    rw [show (⟨⟨l⟩⟩ : ByteArray).size = l.length from rfl] at e
    rw [← e]
    congr 1
    · exact List.map_congr_left fun t ht => bit_cons_lt x l (List.mem_range.1 ht)
    · exact List.map_congr_left fun t _ => bit_cons_add x l t


theorem beq_bytes {a b : ByteArray} : (a == b) = true ↔ a = b := by
  show (a.data == b.data) = true ↔ a = b
  rw [beq_iff_eq, ByteArray.ext_iff]

theorem countFresh_eq (label : String) (after : ByteArray) :
    ∀ rest : List String, countFresh label after rest = ((sel label (pairs rest)).filter (· == after)).length
  | a :: b :: rest => by
    have e : pairs (a :: b :: rest) = (a, b) :: pairs rest := rfl
    rw [countFresh, e, countFresh_eq label after rest]
    unfold sel
    rw [List.filterMap_cons]
    by_cases hl : (rootOf a).1 = label
    · by_cases ha : (rootOf a).2 = after
      · have : (after == after) = true := beq_bytes.2 rfl
        simp [hl, ha, this]
      · have : ((rootOf a).2 == after) = false := by
          cases h : ((rootOf a).2 == after)
          · rfl
          · exact absurd (beq_bytes.1 h) ha
        simp [hl, ha, this]
    · simp [hl]
  | [_] => rfl
  | [] => rfl


/-- The bits of bytes, as the device's log reads them, are their bits as a draw reads them. -/
theorem bitsOfBytes_eq (b : ByteArray) : Verity.bitsOfBytes b = bitsOf b := by
  rw [bitsOf_eq]
  rcases b with ⟨⟨l⟩⟩
  exact bitsOfBytes_list l

/-- **`m` coins of `ξ` are its bits**, where `ξ` is `m` coins long. -/
theorem chunks_flatten : ∀ (m : ℕ) (ξ : ByteArray), ξ.size = COIN_BYTES * m → (chunks m ξ).flatten = bitsOf ξ
  | 0, ξ, h => by
    rw [← bitsOfBytes_eq]
    simp [chunks, Verity.bitsOfBytes, h]
  | m + 1, ξ, h => by
    have h64 : COIN_BYTES ≤ ξ.size := by
      rw [h, Nat.mul_succ]
      exact Nat.le_add_left _ _
    have hs : (ξ.extract COIN_BYTES ξ.size).size = COIN_BYTES * m := by
      rw [ByteArray.size_extract, Nat.min_self, h, Nat.mul_succ, Nat.add_sub_cancel]
    rw [chunks, List.flatten_cons, chunks_flatten m _ hs, bitsOfBytes_eq, ← bitsOf_append,
      ByteArray.extract_append_extract, Nat.zero_min, Nat.max_eq_right h64, ByteArray.extract_zero_size]

/-! ## v2's case in what the prover sent -/

theorem find?_nodup {β : Type} {f : β → String} : ∀ {l : List β} {y : β}, (l.map f).Nodup → y ∈ l →
    l.find? (fun x => f x == f y) = some y
  | z :: l, y, hnd, hy => by
    rw [List.map_cons, List.nodup_cons] at hnd
    rcases List.mem_cons.1 hy with rfl | hy'
    · simp
    · have hne : f z ≠ f y := fun e => hnd.1 (e ▸ List.mem_map_of_mem hy')
      rw [List.find?_cons, show (f z == f y) = false by simpa using hne]
      exact find?_nodup hnd.2 hy'

universe u v w

theorem ids_of_zip {α : Type u} {β : Type v} {γ : Type w} {fa : α → String} {fb : β → String} {fc : γ → String}
    {cs : List α} :
    ∀ {ps : List β} {ev : List γ}, ps.map fb = cs.map fa → ev.map fc = cs.map fa →
      ∀ {x}, x ∈ cs.zip (ps.zip ev) → fb x.2.1 = fa x.1 ∧ fc x.2.2 = fa x.1 := by
  induction cs with
  | nil => intro ps ev _ _ x hx; simp at hx
  | cons c cs ih =>
    intro ps ev hp he x hx
    cases ps with
    | nil => simp at hp
    | cons p ps =>
      cases ev with
      | nil => simp at he
      | cons e ev =>
        simp only [List.map_cons, List.cons.injEq] at hp he
        rw [List.zip_cons_cons, List.zip_cons_cons, List.mem_cons] at hx
        rcases hx with rfl | hx
        · exact ⟨hp.1, he.1⟩
        · exact ih hp.2 he.2 hx

/-- Composite `k`'s components on the blobs sent have distinct ids. -/
theorem components_nodup (k : Verity.Registration.Composite) (s : Verity.Sent) :
    ((s.components k).map (·.id)).Nodup := by
  have e : (s.components k).map (·.id) = (Verity.componentsOf k).map (·.id) := by
    unfold Verity.Sent.components
    rw [List.map_map]
    rfl
  rw [e]
  cases k <;> decide

theorem ok_bind {ε α β : Type} (a : α) (f : α → Except ε β) : (Except.ok a >>= f) = f a := rfl

/-- **The case `caseOf` finds is the one v2 accepted**, in a composition composite `k` accepted. -/
theorem caseOf_of {k : Verity.Registration.Composite} {s : Verity.Sent} {a b e : ByteArray}
    {p : _root_.SampledProofs.OneStageZk.Params} {c : _root_.SampledProofs.OneStage.Commitment}
    {ev : _root_.SampledProofs.OneStageZk.Evidence} (hS : Verity.SameAccepts k s) (hcase : V2Case k s a b e)
    (hp : (Core.canonicalJson a >>= _root_.SampledProofs.OneStageZk.paramsOfJson) = .ok p)
    (hc : _root_.SampledProofs.OneStage.commitmentOf b = .ok c)
    (hev : _root_.SampledProofs.OneStageZk.evidenceOf e = .ok ev) : caseOf s = .ok (p, c, ev) := by
  obtain ⟨ps, c₀, evs, ⟨hps, -, hevs⟩, x, hx, hx1, rfl, rfl, rfl⟩ := hcase
  obtain ⟨ps', c₀', evs', ⟨hps', -, hevs'⟩, pub, prof, ⟨⟨⟨hfp, hfe, -, -⟩, -⟩, -⟩, -⟩ := hS
  rw [Except.ok.inj (hps'.symm.trans hps)] at hfp
  rw [Except.ok.inj (hevs'.symm.trans hevs)] at hfe
  have hid : x.1.id = _root_.SampledProofs.OneStageZk.PROTOCOL := (congrArg (·.id) hx1).trans rfl
  obtain ⟨hpid, heid⟩ := ids_of_zip hfp hfe hx
  have hpid' : x.2.1.id = x.1.id := hpid
  have heid' : x.2.2.id = x.1.id := heid
  have hm := List.of_mem_zip (List.of_mem_zip hx).2
  have h1 : ps.find? (·.id == _root_.SampledProofs.OneStageZk.PROTOCOL) = some x.2.1 := by
    rw [← hid, ← hpid']
    exact find?_nodup (by rw [hfp]; exact components_nodup k s) hm.1
  have h2 : evs.find? (·.id == _root_.SampledProofs.OneStageZk.PROTOCOL) = some x.2.2 := by
    rw [← hid, ← heid']
    exact find?_nodup (by rw [hfe]; exact components_nodup k s) hm.2
  simp only [caseOf, hps, ok_bind, h1, hevs, h2, hp, hc, hev]
  try rfl

/-- Where v2's parameters and commitment name the challenger `dr`, they name its key and family, and its digest is the
registration's. -/
theorem source_of {p : _root_.SampledProofs.OneStageZk.Params} {c : _root_.SampledProofs.OneStage.Commitment}
    {dr : Draws} (h : Properties.Challenger.oneStageZkSource p c = some dr) :
    ∃ key, p.base.challenger = some (key, dr.label) ∧ dr.after = c.digest := by
  unfold Properties.Challenger.oneStageZkSource Properties.Challenger.oneStageSource at h
  rcases hch : p.base.challenger with _ | ⟨key, label⟩ <;> rcases hw : p.base.verifier.windowRoot with _ | w <;>
    simp only [hch, hw] at h <;> try cases h
  exact ⟨key, rfl, rfl⟩

theorem linesOf_of {e : _root_.SampledProofs.OneStageZk.Evidence} {ls : Array Json} {lines : List String}
    (hlog : _root_.SampledProofs.getD (_root_.SampledProofs.getD e.base.coins "session") "log" = .arr ls)
    (hls : ls.toList = lines.map .str) : linesOf e = .ok lines := by
  unfold linesOf
  rw [hlog]
  show ls.toList.mapM _ = _
  rw [hls]
  exact mapM_str_ok (fun _ => rfl) lines

theorem readLog_eq {s : Verity.Sent} {p : _root_.SampledProofs.OneStageZk.Params}
    {c : _root_.SampledProofs.OneStage.Commitment} {e : _root_.SampledProofs.OneStageZk.Evidence}
    {key : ChallengerKey} {label : String} {ξ : ByteArray} {lines : List String} (hcs : caseOf s = .ok (p, c, e))
    (hch : p.base.challenger = some (key, label)) (hξ : _root_.SampledProofs.OneStageZk.coinsOf p c e = .ok ξ)
    (hl : linesOf e = .ok lines) :
    readLog s = some (readLines label c.digest (recordsOf s e) lines.tail [] none ξ) := by
  simp only [readLog, hcs, Except.toOption, Option.bind_eq_bind, Option.bind_some, Option.pure_def, hch, hξ, hl]

/-! ## The draw the read shows -/

/-- **The read shows the draw**: wherever composite `k` accepted what the prover sent and v2's case in it accepts at
the coins `ξ` it read under the challenger `dr`, `readLog` returns a log whose coins of `dr`'s family released after
`dr`'s digest are exactly `ξ`'s bits. -/
theorem drawRead_readLog (k : Verity.Registration.Composite) : DrawRead k readLog := by
  intro s a b e p c ev ξ dr hS hcase hacc
  obtain ⟨hp, hc, hev, hξ, hdr, -⟩ := hacc
  obtain ⟨key, hch, hafter⟩ := source_of hdr
  obtain ⟨raw, hraw, hrξ⟩ := Properties.Challenger.oneStageZk_reads p c dr hdr ev ξ hξ
  rw [hrξ] at hraw
  obtain ⟨cm, roots, hts, hstream⟩ := drawCoins_ok hraw
  obtain ⟨_, hsn⟩ := trustedSessionOf_ok hts
  obtain ⟨_, hso⟩ := sessionNamed_ok hsn
  obtain ⟨ls, lines, hlog, hls, hrr⟩ := sessionOf_ok hso
  refine ⟨_, readLog_eq (caseOf_of hS hcase hp hc hev) hch hξ (linesOf_of hlog hls), ?_⟩
  rw [hafter] at hstream
  have hf : (freshCoins dr.label c.digest
      (readLines dr.label c.digest (recordsOf s ev) lines.tail [] none ξ)).flatten = bitsOf ξ := by
    rw [fresh_readLines, Option.toList_none, List.nil_append, countFresh_eq, ← releasedRootsOf_ok hrr,
      chunks_flatten _ _ (challengerStream_ok hstream)]
  unfold ShowsDraw
  rw [hafter]
  exact hf ▸ List.prefix_refl _

/-- Any read that agrees with `readLog` shows the draw. -/
theorem drawRead_of_eq (k : Verity.Registration.Composite) {read : Verity.Sent → Option (List Verity.Release)}
    (h : ∀ s, read s = readLog s) : DrawRead k read := by
  rw [show read = readLog from funext h]
  exact drawRead_readLog k

end Verity.Sampled
