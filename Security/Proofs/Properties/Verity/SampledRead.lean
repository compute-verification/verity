import Batteries.Tactic.OpenPrivate
-- `Ledger` realizes the match equations `mvcgen` splits `sessionNamed` with; realizing them here too clashes wherever
-- both modules are imported.
import Proofs.Properties.Pouw.LiveSalt.Ledger
import Proofs.Properties.Verity.SampledDraw
import Proofs.Properties.Verity.SampledLog
import Std.Tactic.Do

/-!
# The device's log as v2's check reads it, and the draw it shows

Where the sampled check's read of the device's log (`coinsRead`) returns a log from what the prover sent, in an accepted
composition whose v2 case accepts, the log shows the draw (`shows_coinsRead`): the read takes the challenger session's
release log in v2's evidence line by line as v2's check does (`readLines`), with the draw's coins `ξ`
(`SampledProofs.OneStageZk.coinsOf`) placed in order on the releases of the draw's family after the registration's
digest, one coin of `COIN_BYTES` bytes each. So the read meets the draw's ask (`DrawRead`) wherever v2's acceptance
reads the log (`ReadsLog`, `drawRead_coinsRead`).

* **v2's challenger** (`drawCoins_ok`, `trustedSessionOf_ok`, `sessionNamed_ok`, `sessionOf_ok`, `releasedRootsOf_ok`,
  `challengerStream_ok`): where v2 reads `ξ`, the session's log is a list of lines, the roots of the draw's family are
  those its pairs' root lines show (`sel`), and `ξ` is `COIN_BYTES` bytes for each of them equal to the digest.
* **What the read shows** (`fresh_readLines`, `countFresh_eq`, `chunks_flatten`): the read's coins of the draw's family
  after the digest are `ξ`'s coins, one for each such root, so their bits are `ξ`'s.
* **What is open** (`ReadsLog`): that the read returns a log wherever v2 accepts. The read refuses a coin tree's line
  whose `event` string is neither a tree's nor an opening's, an opening whose stream its tree's spec doesn't name, and
  a record that doesn't parse. v2's check tells a coin tree's line apart and finds its stream by `BEq Json`
  (`treeLine`, `coinTreesOf`), which is `partial`, and reads the trees over the whole log, its head included, where the
  read takes them after the head; so v2's acceptance shows none of those to read. The pairs v2 shows to read: each
  root's line parses with its family and root, and none is left without its release (`releasedRootsOf`).
* **v2's case** (`caseOf_of`): in an accepted composition the case `caseOf` finds under v2's id is the one v2 accepted,
  since the composite's components have distinct ids (`components_nodup`).
* **The session's law** (`law_of_setsUp`, `coupled_draw_setsUp`, `coupled_draw_coins_setsUp`): v2's parameters in the
  parameters sent are those its case accepted (`paramsAt_case`), so at the session v2 sets up on them (`SetsUp`) the
  draw's law is the session's wherever its population is (`lawAt_of_lawOf`), and the draw's piece of `CouplingTarget`
  needs only the population.

Each fact about one of the challenger's monadic functions is a Hoare triple proved by `mvcgen`, read back at an `.ok`
result (`of_triple`, in `SampledLog`).
-/

set_option mvcgen.warning false

open private pairs from SampledProofs.ChallengerSession

namespace Verity.Sampled

open Lean (Json)
open Std.Do
open SampledProofs.Challenger
open Properties.Challenger (bitsOf Draws)

/-! ## v2's challenger -/

/-! ### The session's files and the draw's coins -/

theorem recordReason_spec (key : ChallengerKey) (record : Json) :
    ⦃⌜True⌝⦄ recordReason key record ⦃⇓? _ => ⌜True⌝⦄ := except_spec _

theorem recordOfJson_spec (j : Json) : ⦃⌜True⌝⦄ ChallengerRecord.ofJson j ⦃⇓? _ => ⌜True⌝⦄ := except_spec _

theorem closeMessage_spec (I : ByteArray) (host : String) (c : LiveCommitment) (session : String) (lines : Nat)
    (chain : ByteArray) (ended : Option String) :
    ⦃⌜True⌝⦄ closeMessage I host c session lines chain ended ⦃⇓? _ => ⌜True⌝⦄ := except_spec _

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
  ps.filterMap fun x => match rootOf x.1 with
    | some (l, r) => if l = label then some r else none
    | none => none

/-- The roots of family `label` in `releasedRootsOf`'s state. -/
def lookA (roots : Array (String × Array ByteArray)) (label : String) : List ByteArray :=
  ((roots.find? (·.1 == label)).map (·.2.toList)).getD []

theorem rootOf_eq {a : String} {r : Json} {label root : String} {bytes : ByteArray} (he : event a = .ok r)
    (hl : SampledProofs.getD r "label" = .str label) (hr : SampledProofs.getD r "root" = .str root)
    (hb : (Flock.unhex root).toOption = some bytes) : rootOf a = some (label, bytes) := by
  simp only [rootOf, he, Except.toOption, Option.bind_eq_bind, Option.bind_some, Option.pure_def, hl, hr, hb]

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
    ∀ (ls : List String) (ts : List (String × Flock.CoinTree.Spec)) (pending : Option String) (ξ : ByteArray)
      {log : List Verity.Release}, readLines label after records ls ts pending ξ = some log →
      freshCoins label after log = chunks (countFresh label after (pending.toList ++ ls.filter fun l => !treeLine l)) ξ
  | [], _, none, _, _, h => by
    simp only [readLines, Option.some.injEq] at h
    subst h
    rfl
  | [], _, some _, _, _, h => by simp [readLines] at h
  | l :: ls, ts, pending, ξ, log, h => by
    unfold readLines at h
    by_cases ht : treeLine l = true
    · rw [if_pos ht] at h
      rw [List.filter_cons_of_neg (by simp [ht])]
      simp only [Option.bind_eq_bind] at h
      obtain ⟨⟨r, ts'⟩, hX, h⟩ := Option.bind_eq_some_iff.1 h
      obtain ⟨j, -, hj⟩ := Option.bind_eq_some_iff.1 hX
      simp only [Option.bind_eq_bind, Option.pure_def] at h
      obtain ⟨rest, hR, h⟩ := Option.bind_eq_some_iff.1 h
      cases h
      rw [freshCoins_cons_tree hj]
      exact fresh_readLines label after records ls ts' pending ξ hR
    · rw [if_neg ht] at h
      rw [List.filter_cons_of_pos (by simp [ht])]
      cases pending with
      | none => exact fresh_readLines label after records ls ts (some l) ξ h
      | some a =>
        simp only [Option.toList_some, List.cons_append, List.nil_append]
        simp only [Option.bind_eq_bind] at h
        obtain ⟨⟨lab, root⟩, hra, h⟩ := Option.bind_eq_some_iff.1 h
        simp only [Option.bind_eq_bind, Option.pure_def] at h
        by_cases hm : lab = label ∧ root = after
        · rw [if_pos hm] at h
          obtain ⟨rest, hR, h⟩ := Option.bind_eq_some_iff.1 h
          cases h
          have hc : rootOf a = some (label, after) := by rw [hra, hm.1, hm.2]
          rw [countFresh, if_pos hc, chunks]
          simp only [freshCoins, List.filterMap_cons, and_self, ite_true]
          congr 1
          exact fresh_readLines label after records ls ts none _ hR
        · rw [if_neg hm] at h
          obtain ⟨rest, hR, h⟩ := Option.bind_eq_some_iff.1 h
          cases h
          have hc : rootOf a ≠ some (label, after) := by
            rw [hra]
            exact fun e => hm (Prod.mk.inj (Option.some.inj e))
          rw [countFresh, if_neg hc, Nat.add_zero]
          simp only [freshCoins, List.filterMap_cons, if_neg hm]
          exact fresh_readLines label after records ls ts none ξ hR

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
    dsimp only
    rcases hra : rootOf a with _ | ⟨l, r⟩
    · simp
    · by_cases hl : l = label
      · by_cases ha : r = after
        · have : (after == after) = true := beq_bytes.2 rfl
          simp [hl, ha, this]
        · have : (r == after) = false := by
            cases h : (r == after)
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

theorem ok_bind {ε α β : Type} (a : α) (f : α → Except ε β) : (Except.ok a >>= f) = f a := rfl

/-- **The case `caseOf` finds is the one v2 accepted**, in a composition composite `k` accepted. -/
theorem caseOf_of {k : Verity.Registration.Composite} {s : Verity.Sent} {a b e : ByteArray}
    {p : _root_.SampledProofs.OneStageZk.Params} {c : _root_.SampledProofs.OneStage.Commitment}
    {ev : _root_.SampledProofs.OneStageZk.Evidence} (hS : Verity.SameAccepts k s) (hcase : V2Case k s a b e)
    (hp : (Core.canonicalJson a >>= VStar.Recursive.Sampled.paramsOfJson) = .ok (.inl p))
    (hc : _root_.SampledProofs.OneStage.commitmentOf b = .ok c)
    (hev : VStar.Recursive.Sampled.evidenceOf e = .ok (.inl ev)) : caseOf s = .ok (p, c, ev) := by
  obtain ⟨ps, c₀, evs, ⟨hps, -, hevs⟩, x, hx, hx1, rfl, rfl, rfl⟩ := hcase
  obtain ⟨ps', c₀', evs', ⟨hps', -, hevs'⟩, pub, prof, ⟨⟨⟨hfp, hfe, -, -⟩, -⟩, -⟩, -⟩ := hS
  rw [Except.ok.inj (hps'.symm.trans hps)] at hfp
  rw [Except.ok.inj (hevs'.symm.trans hevs)] at hfe
  have hid : x.1.id = VStar.Recursive.Sampled.PROTOCOL := (congrArg (·.id) hx1).trans rfl
  obtain ⟨hpid, heid⟩ := ids_of_zip hfp hfe hx
  have hpid' : x.2.1.id = x.1.id := hpid
  have heid' : x.2.2.id = x.1.id := heid
  have hm := List.of_mem_zip (List.of_mem_zip hx).2
  have h1 : ps.find? (·.id == VStar.Recursive.Sampled.PROTOCOL) = some x.2.1 := by
    rw [← hid, ← hpid']
    exact find?_nodup (by rw [hfp]; exact components_nodup k s) hm.1
  have h2 : evs.find? (·.id == VStar.Recursive.Sampled.PROTOCOL) = some x.2.2 := by
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

theorem coinsRead_eq {s : Verity.Sent} {p : _root_.SampledProofs.OneStageZk.Params}
    {c : _root_.SampledProofs.OneStage.Commitment} {e : _root_.SampledProofs.OneStageZk.Evidence}
    {key : ChallengerKey} {label : String} {ξ : ByteArray} {lines : List String} (hcs : caseOf s = .ok (p, c, e))
    (hch : p.base.challenger = some (key, label)) (hξ : _root_.SampledProofs.OneStageZk.coinsOf p c e = .ok ξ)
    (hl : linesOf e = .ok lines) :
    coinsRead s = (recordsOf s e).bind fun rs => readLines label c.digest rs lines.tail [] none ξ := by
  simp only [coinsRead, hcs, Except.toOption, Option.bind_eq_bind, Option.bind_some, Option.pure_def, hch, hξ, hl]

/-! ## The draw the read shows -/

/-- **Where the read returns a log, it shows the draw**: wherever composite `k` accepted what the prover sent and v2's
case in it accepts at the coins `ξ` it read under the challenger `dr`, a log `coinsRead` returns has exactly `ξ`'s bits
as its coins of `dr`'s family released after `dr`'s digest. -/
theorem shows_coinsRead {k : Verity.Registration.Composite} {s : Verity.Sent} {a b e : ByteArray}
    {p : _root_.SampledProofs.OneStageZk.Params} {c : _root_.SampledProofs.OneStage.Commitment}
    {ev : _root_.SampledProofs.OneStageZk.Evidence} {ξ : ByteArray} {dr : Draws} (hS : Verity.SameAccepts k s)
    (hcase : V2Case k s a b e) (hacc : V2Accepts (Core.Blobs.ofArray s.blobs) a b e p c ev ξ dr)
    {log : List Verity.Release} (h : coinsRead s = some log) : ShowsDraw log dr ξ := by
  obtain ⟨hp, hc, hev, hξ, hdr, -⟩ := hacc
  obtain ⟨key, hch, hafter⟩ := source_of hdr
  obtain ⟨raw, hraw, hrξ⟩ := Properties.Challenger.oneStageZk_reads p c dr hdr ev ξ hξ
  rw [hrξ] at hraw
  obtain ⟨cm, roots, hts, hstream⟩ := drawCoins_ok hraw
  obtain ⟨_, hsn⟩ := trustedSessionOf_ok hts
  obtain ⟨_, hso⟩ := sessionNamed_ok hsn
  obtain ⟨ls, lines, hlog, hls, hrr⟩ := sessionOf_ok hso
  rw [coinsRead_eq (caseOf_of hS hcase hp hc hev) hch hξ (linesOf_of hlog hls)] at h
  obtain ⟨rs, -, h⟩ := Option.bind_eq_some_iff.1 h
  rw [hafter] at hstream
  have hf : (freshCoins dr.label c.digest log).flatten = bitsOf ξ := by
    rw [fresh_readLines _ _ _ _ _ _ _ h, Option.toList_none, List.nil_append, countFresh_eq, ← releasedRootsOf_ok hrr,
      chunks_flatten _ _ (challengerStream_ok hstream)]
  unfold ShowsDraw
  rw [hafter]
  exact hf ▸ List.prefix_refl _

/-- **v2's acceptance reads the device's log**: wherever composite `k` accepted what the prover sent and v2's case in it
accepts, `coinsRead` returns a log. -/
def ReadsLog (k : Verity.Registration.Composite) : Prop :=
  ∀ s a b e p c ev ξ dr, Verity.SameAccepts k s → V2Case k s a b e →
    V2Accepts (Core.Blobs.ofArray s.blobs) a b e p c ev ξ dr → (coinsRead s).isSome

/-- **The read shows the draw** wherever v2's acceptance reads the log (`ReadsLog`). -/
theorem drawRead_coinsRead (k : Verity.Registration.Composite) (h : ReadsLog k) : DrawRead k coinsRead := by
  intro s a b e p c ev ξ dr hS hcase hacc
  obtain ⟨log, hlog⟩ := Option.isSome_iff_exists.1 (h s a b e p c ev ξ dr hS hcase hacc)
  exact ⟨log, hlog, shows_coinsRead hS hcase hacc hlog⟩

/-! ## The session's law -/

/-- **v2's parameters in the parameters sent are the ones its case accepted**, in a composition composite `k`
accepted. -/
theorem paramsAt_case {k : Verity.Registration.Composite} {s : Verity.Sent} {a b e : ByteArray}
    {p : _root_.SampledProofs.OneStageZk.Params} {c : _root_.SampledProofs.OneStage.Commitment}
    {ev : _root_.SampledProofs.OneStageZk.Evidence} {ξ : ByteArray} {dr : Draws} (hS : Verity.SameAccepts k s)
    (hcase : V2Case k s a b e) (hacc : V2Accepts (Core.Blobs.ofArray s.blobs) a b e p c ev ξ dr)
    {p' : _root_.SampledProofs.OneStageZk.Params} (h : ParamsAt s.params p') : p' = p :=
  (Sum.inl.inj (params_of_case hS hcase hacc.1 h)).symm

/-- The law `lawAt` reads at the population `LawOf` names is the law `LawOf` names. -/
theorem lawAt_of_lawOf {p : _root_.SampledProofs.OneStageZk.Params} {c : _root_.SampledProofs.OneStage.Commitment}
    {L : Flock.Draw.Law} {n : ℕ} (h : LawOf p c L n) : lawAt p.base.verifier n = .ok L := by
  obtain ⟨law, derived, fl, hd, -, hder, hfl, rfl⟩ := h
  simp only [lawAt, hd, hder, hfl, ok_bind]
  rfl

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **The session's law is the draw's** where the session's population is: at the session `A` v2 sets up on the
parameters `θ` (`SetsUp`), a draw v2 made in a composition composite `k` accepted, on those parameters, over `A.n`
units, ran under `A.law`. -/
theorem law_of_setsUp {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {A : Live} (hA : SetsUp V θ F A)
    {k : Verity.Registration.Composite} {s : Verity.Sent} {dr : Draws} {L : Flock.Draw.Law} {n : ℕ} {ξ : ByteArray}
    {units : Array ℕ} (hS : Verity.SameAccepts k s) (hθ : s.params = θ) (hd : Drew k s dr L n ξ units)
    (hn : A.n = n) : A.law = L := by
  obtain ⟨a, b, e, p, c, ev, hcase, hacc, hL, -, -⟩ := hd
  obtain ⟨p', hp', -, -, hlaw⟩ := hA.params
  rw [← hθ] at hp'
  rw [paramsAt_case hS hcase hacc hp', hn, lawAt_of_lawOf hL] at hlaw
  exact (Except.ok.inj hlaw).symm

/-- **The draw's piece of `CouplingTarget` at the session v2 sets up**: `coupled_draw`, with the session's law read
off `SetsUp` on the play's parameters. -/
theorem coupled_draw_setsUp {V : Vocab d α} {k : Verity.Registration.Composite} (K : Constructions V) (A : Live)
    (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c))
    (hA : SetsUp V θ F A) (hR : DrawRead k (V.coinsRead .sampled)) {o : Verity.Exchange}
    (hθ : o.params = θ) (hv : V.Verdict k .sampled o) (hl : V.Logged .sampled o) :
    ∃ dr L n ξ units, Drew k o.sent dr L n ξ units ∧
      ∀ ω : A.Ω × (K.coins A θ F c σ).T, A.n = n →
        (freshCoins dr.label dr.after o.log).flatten <+:
          bitsOf (_root_.Properties.SampledProofs.bytesOf (A.src ω.1 0)) →
        (K.live A θ F c σ).L.draw ω = Finset.univ.filter fun u => u.val ∈ units := by
  obtain ⟨p₀, hp₀, -⟩ := hA.params
  obtain ⟨dr, L, n, ξ, units, hd, h⟩ := coupled_draw K A θ F c σ hR (by rw [hθ]; exact hp₀) hv hl
  exact ⟨dr, L, n, ξ, units, hd, fun ω hn hp => h ω (law_of_setsUp hA hv.2.1 hθ hd hn) hn hp⟩

/-- `coupled_draw_coins` at the session v2 sets up. -/
theorem coupled_draw_coins_setsUp {V : Vocab d α} {k : Verity.Registration.Composite} (K : Constructions V)
    (A : Live) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c))
    (hA : SetsUp V θ F A) (hR : DrawRead k (V.coinsRead .sampled)) {o : Verity.Exchange}
    (hθ : o.params = θ) (hv : V.Verdict k .sampled o) (hl : V.Logged .sampled o) {m B : ℕ}
    (κ : A.Ω → Fin m → Fin (8 * B) → Bool)
    (hsrc : ∀ ω, _root_.Properties.SampledProofs.bytesOf (A.src ω 0) =
      _root_.Properties.SampledProofs.bytesOf (coinBytes m B (κ ω) 0)) :
    ∃ dr L n ξ units, Drew k o.sent dr L n ξ units ∧
      ∀ ω : A.Ω × (K.coins A θ F c σ).T, A.n = n →
        freshCoins dr.label dr.after o.log <+: (List.ofFn fun j => List.ofFn (κ ω.1 j)) →
        (K.live A θ F c σ).L.draw ω = Finset.univ.filter fun u => u.val ∈ units := by
  obtain ⟨p₀, hp₀, -⟩ := hA.params
  obtain ⟨dr, L, n, ξ, units, hd, h⟩ := coupled_draw_coins K A θ F c σ hR (by rw [hθ]; exact hp₀) hv hl κ hsrc
  exact ⟨dr, L, n, ξ, units, hd, fun ω hn hp => h ω (law_of_setsUp hA hv.2.1 hθ hd hn) hn hp⟩

end Verity.Sampled
