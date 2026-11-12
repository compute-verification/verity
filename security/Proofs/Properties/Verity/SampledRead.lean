module

public import Batteries.Tactic.OpenPrivate
-- `Ledger` realizes the match equations `mvcgen` splits `sessionNamed` with; realizing them here too clashes wherever
-- both modules are imported.
public import Proofs.Properties.Pouw.LiveSalt.Ledger
public import Proofs.Properties.Verity.SampledDraw
public import Proofs.Properties.Verity.SampledLog
public import Proofs.Properties.Verity.Sampled.Trees
public import Std.Tactic.Do

/-!
# The device's log as v2's check reads it, and the sample it shows

Where the sampled check's read of the device's log (`coinsRead`) returns a log from what the prover sent, in an accepted
composition whose v2 case accepts, the log shows the sample (`shows_logRead`): the read takes the challenger session's
release log in v2's evidence line by line as v2's check does (`readLog`), with the sample's coins `ξ`
(`Spot.OneStageZk.coinsOf`) placed in order on the releases of the sample's family after the registration's
digest, one coin of `COIN_BYTES` bytes each. So the read meets the sample's ask (`DrawRead`) with no premise
(`drawRead_logRead`, `drawRead_coinsRead`).

* **v2's challenger** (`drawCoins_ok`, `trustedSessionOf_ok`, `sessionNamed_ok`, `sessionOf_ok`, `releasedRootsOf_ok`,
  `challengerStream_ok`): where v2 reads `ξ`, the session's log is a list of lines, the roots of the sample's family are
  those its pairs' root lines show (`sel`), and `ξ` is `COIN_BYTES` bytes for each of them equal to the digest.
* **What the read shows** (`fresh_readLog`, `countFresh_eq`, `chunks_flatten`): the read's coins of the sample's family
  after the digest are `ξ`'s coins, one for each such root, so their bits are `ξ`'s.
* **The read in order** (`treeStep_some`, `specOf_streams`, `sampleLinkFirst_snoc`, `sampleLinkFirst_readTail`,
  `sampleLinkFirst_readLog`, `sampleLinkFirst_congr`): where the release log's lines after its head pass
  `sampleLinkFirst`, the releases the read shows pass `SampleLinkFirst`: the sample, then the link, then the other
  rounds; and so do releases of the same frames answering the same bytes, as a log the read is of. The head's line
  releases at most a `Hello`, since an opening there is of no tree read before it.
* **The read with v2's own tests** (`logRead`, `logRead_isSome`, `shows_logRead`, `drawRead_logRead`): `coinsRead`,
  `logRead` by its own name (`coinsRead_eq_logRead`), tells a tree's line apart, finds its tree and its stream with v2's
  own Bool tests (`treeStep`), takes the trees over the whole log, its head included, and reads each session's record
  as v2's `treesHeld` does (`recordsOn`). v2's loops then show it a log wherever v2 accepts (`coinTreesOf_fold`,
  `releasedRootsOf_pairsOk`, `treesHeld_records`), so it meets the sample's ask with no premise.
* **v2's case** (`caseOf_of`): in an accepted composition the case `caseOf` finds under v2's id is the one v2 accepted,
  since the composite's components have distinct ids (`components_nodup`).
* **The session's law** (`law_of_setsUp`, `coupled_draw_setsUp`, `coupled_draw_coins_setsUp`): v2's parameters in the
  parameters sent are those its case accepted (`paramsAt_case`), so at the session v2 sets up on them (`SetsUp`) the
  sample's law is the session's wherever its population is (`lawAt_of_lawOf`), and the sample's piece of `CouplingTarget`
  needs only the population.
* **The session's population** (`popHeld_of_checkBlobsAt`, `popOk_of_popHeld`, `n_of_popOk`, `n_of_popTied`,
  `coupled_draw_tied`, `coupled_draw_coins_tied`): v2 refuses a record whose population isn't its program's proof
  blocks in order (`popHeld`), and the session's population (`Live.hpop`) reads the same layout. So at a session whose
  descriptor and query are v2's verifier's (`PopTied`, which `SetsUp`'s `params` implies) the sample's population is the
  session's, and the sample's piece asks nothing of the session but its coins.

Each fact about one of the challenger's monadic functions is a Hoare triple proved by `mvcgen`, read back at an `.ok`
result (`of_triple`, in `SampledLog`).
-/

@[expose] public section

set_option mvcgen.warning false

open private pairs from Spot.ChallengerSession

namespace Verity.Sampled

open Lean (Json)
open Std.Do
open Spot.Challenger
open Properties.Challenger (bitsOf Draws)

/-! ## v2's challenger -/

/-! ### The session's files and the sample's coins -/

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
    ∃ sid, sessionOf key (Spot.getD session "record") (Spot.getD session "log")
      (Spot.getD session "close") = .ok (r.1, r.2, sid) := by
  refine of_triple (Q := fun r => ∃ sid, sessionOf key (Spot.getD session "record")
    (Spot.getD session "log") (Spot.getD session "close") = .ok (r.1, r.2, sid)) ?_ h
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
    ∃ cm roots, trustedSessionOf (Spot.getD coins "ledger") (Json.mkObj [("root", Json.str w)]) key
        (Spot.getD coins "pins") (Spot.getD coins "main") (Spot.getD coins "session") =
          .ok (cm, roots) ∧
      challengerStream cm (Spot.getD coins "segment") after (some (look roots label)) = .ok ξ := by
  refine of_triple (Q := fun ξ => ∃ cm roots, trustedSessionOf (Spot.getD coins "ledger")
    (Json.mkObj [("root", Json.str w)]) key (Spot.getD coins "pins") (Spot.getD coins "main")
      (Spot.getD coins "session") = .ok (cm, roots) ∧
    challengerStream cm (Spot.getD coins "segment") after (some (look roots label)) = .ok ξ) ?_ h
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
    (hl : Spot.getD r "label" = .str label) (hr : Spot.getD r "root" = .str root)
    (hb : (Flock.unhex root).toOption = some bytes) : rootOf a = some (label, bytes) := by
  have hu : Flock.unhex root = .ok bytes := by
    revert hb
    cases Flock.unhex root <;> intro hb <;> first | (cases hb; rfl) | cases hb
  simp only [rootOf, he, hu, Except.toOption, Option.bind_eq_bind, Option.bind_some, Option.pure_def, hl, hr]

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
    (hI : ∀ ℓ, lookA roots ℓ = sel ℓ pre) (he : event x.1 = .ok r) (hl : Spot.getD r "label" = .str label)
    (hk : roots.findIdx? (·.1 == label) = some k) (hr : Spot.getD r "root" = .str root)
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

/-- The bits of bytes, as the device's log reads them, are their bits as a sample reads them. -/
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

/-! ## The read in order -/

/-- **A coin tree's line releases its `Hello` or a round's opening**, as `treeStep` reads it: a `Hello`, on a line whose
`event` is `TREE_EVENT`, adds its tree's line spec to the trees read, which parses to the `Hello`'s spec (`specOf`);
an opening, on a line whose `event` isn't `TREE_EVENT`, is of a round of a tree read before it, of a stream of that
tree's line spec (`specStreams`) the line names. -/
theorem treeStep_some {records : List Json} {ts ts' : List (String × Json)} {j : Json} {r : Verity.Release}
    (h : treeStep records ts j = some (r, ts')) :
    (∃ root sj spec, r.frame = .hello spec ∧ ts' = ts ++ [(root, sj)] ∧ specOf sj = some spec ∧
        Spot.jsonStrIs (Spot.getD j "event") TREE_EVENT = true) ∨
      (∃ i s n, r.frame = .opening i s n ∧ ts' = ts ∧
        Spot.jsonStrIs (Spot.getD j "event") TREE_EVENT = false ∧
        ∃ stream, Spot.getD j "stream" = .str stream ∧
          ((ts[i]?).bind fun t => (specStreams t.2)[s]?) = some (.str stream)) := by
  unfold treeStep at h
  by_cases hT : Spot.jsonStrIs (Spot.getD j "event") TREE_EVENT = true
  · rw [ite_eq_left hT] at h
    simp only [Option.bind_eq_bind] at h
    left
    split at h
    · split at h
      · split at h
        · rename_i root _
          obtain ⟨spec, hspec, h⟩ := Option.bind_eq_some_iff.1 h
          obtain ⟨ν, -, h⟩ := Option.bind_eq_some_iff.1 h
          obtain ⟨r', -, h⟩ := Option.bind_eq_some_iff.1 h
          obtain ⟨κ, -, h⟩ := Option.bind_eq_some_iff.1 h
          simp only [Option.pure_def, Option.some.injEq, Prod.mk.injEq] at h
          obtain ⟨rfl, rfl⟩ := h
          exact ⟨root, _, spec, rfl, rfl, hspec, hT⟩
        · simp at h
      · simp at h
    · simp at h
  · rw [ite_eq_right hT] at h
    simp only [Option.bind_eq_bind] at h
    right
    split at h
    · split at h
      · rename_i stream hstream
        split at h
        · obtain ⟨k, -, h⟩ := Option.bind_eq_some_iff.1 h
          obtain ⟨n, -, h⟩ := Option.bind_eq_some_iff.1 h
          obtain ⟨i, -, h⟩ := Option.bind_eq_some_iff.1 h
          obtain ⟨⟨_, spec⟩, hts, h⟩ := Option.bind_eq_some_iff.1 h
          obtain ⟨s, hs, h⟩ := Option.bind_eq_some_iff.1 h
          obtain ⟨b, -, h⟩ := Option.bind_eq_some_iff.1 h
          simp only [Option.pure_def, Option.some.injEq, Prod.mk.injEq] at h
          obtain ⟨rfl, rfl⟩ := h
          refine ⟨i, s, n, rfl, rfl, Bool.eq_false_iff.2 hT, stream, hstream, ?_⟩
          rw [hts, Option.bind_some]
          obtain ⟨hlt, heq, -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hs
          simp [Array.getElem?_eq_getElem hlt, Trees.strIs heq]
        · simp at h
      · simp at h
    · simp at h

/-- `mapM` into `Option` keeps each place: where it maps `l` to `l'`, the element of `l'` at a place is `f` of `l`'s. -/
theorem mapM_getElem? {α β : Type} {f : α → Option β} :
    ∀ {l : List α} {l' : List β}, l.mapM f = some l' → ∀ s : ℕ, l'[s]? = l[s]?.bind f
  | [], l', h, s => by
    simp only [List.mapM_nil, Option.pure_def, Option.some.injEq] at h
    subst h
    simp
  | a :: l, l', h, s => by
    simp only [List.mapM_cons, Option.bind_eq_bind, Option.pure_def] at h
    obtain ⟨b, hb, h⟩ := Option.bind_eq_some_iff.1 h
    obtain ⟨t, ht, h⟩ := Option.bind_eq_some_iff.1 h
    obtain rfl := Option.some.inj h
    cases s with
    | zero => simp [hb]
    | succ s => simpa using mapM_getElem? ht s

/-- `mapM_getElem?` on arrays. -/
theorem mapM_getElem?_arr {α β : Type} {f : α → Option β} {xs : Array α} {ys : Array β} (h : xs.mapM f = some ys)
    (s : ℕ) : ys[s]? = xs[s]?.bind f := by
  rw [Array.mapM_eq_mapM_toList] at h
  obtain ⟨l', hl', rfl⟩ : ∃ l', xs.toList.mapM f = some l' ∧ l'.toArray = ys := by
    cases hm : xs.toList.mapM f with
    | none => simp [hm] at h
    | some l' => exact ⟨l', rfl, by simpa [hm] using h⟩
  rw [List.getElem?_toArray, mapM_getElem? hl', Array.getElem?_toList]

/-- **A stream a coin tree's line spec names is its parsed spec's**: where the line's spec parses (`specOf`), a stream
at a place of the list it names (`specStreams`) is the parsed spec's stream at that place. -/
theorem specOf_streams {j : Json} {sp : Flock.CoinTree.Spec} (h : specOf j = some sp) {s : ℕ} {x : String}
    (hx : (specStreams j)[s]? = some (.str x)) : sp.streams[s]? = some x := by
  unfold specOf at h
  unfold specStreams at hx
  cases hss : Spot.getD j "streams" with
  | arr ss =>
    simp only [hss, Option.bind_eq_bind, Option.pure_def] at h hx
    obtain ⟨streams, hst, h⟩ := Option.bind_eq_some_iff.1 h
    obtain ⟨a, -, h⟩ := Option.bind_eq_some_iff.1 h
    obtain ⟨b, -, h⟩ := Option.bind_eq_some_iff.1 h
    obtain rfl := Option.some.inj h
    rw [mapM_getElem?_arr hst, hx]
    rfl
  | _ => simp [hss] at h

theorem forall₂_getElem? {α β : Type} {R : α → β → Prop} :
    ∀ {l : List α} {l' : List β}, List.Forall₂ R l l' → ∀ {i : ℕ} {a : α}, l[i]? = some a →
      ∃ b, l'[i]? = some b ∧ R a b
  | _, _, .nil, _, _, h => by simp at h
  | _, _, .cons hab _, 0, _, h => by
    simp only [List.getElem?_cons_zero, Option.some.injEq] at h
    subst h
    exact ⟨_, rfl, hab⟩
  | _, _, .cons _ t, _ + 1, _, h => by
    simp only [List.getElem?_cons_succ] at h ⊢
    exact forall₂_getElem? t h

theorem forall₂_snoc {α β : Type} {R : α → β → Prop} {a : α} {b : β} :
    ∀ {l : List α} {l' : List β}, List.Forall₂ R l l' → R a b → List.Forall₂ R (l ++ [a]) (l' ++ [b])
  | _, _, .nil, h => .cons h .nil
  | _, _, .cons h₁ t, h => .cons h₁ (forall₂_snoc t h)

theorem helloSpecs_append (a b : List Verity.Release) : helloSpecs (a ++ b) = helloSpecs a ++ helloSpecs b :=
  List.filterMap_append

/-- **A release after a log in order keeps it in order**: where it releases no coin of the sample after an opening,
and, as the first opening, opens the stream `TREE_SESSION`. -/
theorem sampleLinkFirst_snoc {label : String} {after : ByteArray} {pre : List Verity.Release} {r : Verity.Release}
    (hP : SampleLinkFirst label after pre)
    (hf : (∃ q ∈ pre, ∃ i s n, q.frame = .opening i s n) → ¬ (r.frame = .fresh label ∧ r.answers = after))
    (ho : ∀ i s n, r.frame = .opening i s n → (∀ q ∈ pre, ∀ i' s' n', q.frame ≠ .opening i' s' n') →
      ((helloSpecs pre)[i]?.bind fun sp => sp.streams[s]?) = some TREE_SESSION) :
    SampleLinkFirst label after (pre ++ [r]) := by
  intro pre₀ r₀ post₀ heq i s n hr₀
  cases post₀.eq_nil_or_concat with
  | inl h =>
    subst h
    obtain ⟨rfl, h⟩ := List.append_inj' heq rfl
    cases List.cons.inj h |>.1
    exact ⟨fun _ hq => absurd hq List.not_mem_nil, ho i s n hr₀⟩
  | inr h =>
    obtain ⟨post₁, x, rfl⟩ := h
    rw [List.concat_eq_append] at heq ⊢
    have heq' : pre ++ [r] = (pre₀ ++ r₀ :: post₁) ++ [x] := by rw [heq]; simp
    obtain ⟨hpre, hx⟩ := List.append_inj' heq' rfl
    cases List.cons.inj hx |>.1
    obtain ⟨hpost, hlink⟩ := hP pre₀ r₀ post₁ hpre i s n hr₀
    refine ⟨fun q hq => ?_, hlink⟩
    rcases List.mem_append.1 hq with hq | hq
    · exact hpost q hq
    · rw [List.mem_singleton.1 hq]
      exact hf ⟨r₀, by rw [hpre]; simp, i, s, n, hr₀⟩

/-- A family's release after a log in order keeps it in order, where it isn't the sample's or no round is opened. -/
theorem sampleLinkFirst_snoc_fresh {label : String} {after : ByteArray} {pre : List Verity.Release}
    {r : Verity.Release} {opened : Bool} {lab : String} {root : ByteArray} (hP : SampleLinkFirst label after pre)
    (hO : opened = true ↔ ∃ q ∈ pre, ∃ i s n, q.frame = .opening i s n) (hr : r.frame = .fresh lab)
    (ha : r.answers = root) (hs : opened = true → ¬ (lab = label ∧ root = after)) :
    SampleLinkFirst label after (pre ++ [r]) ∧ helloSpecs (pre ++ [r]) = helloSpecs pre ∧
      (opened = true ↔ ∃ q ∈ pre ++ [r], ∃ i s n, q.frame = .opening i s n) := by
  refine ⟨sampleLinkFirst_snoc hP (fun h ⟨h1, h2⟩ => ?_) (fun i s n h => ?_), ?_, ?_⟩
  · rw [hr] at h1
    cases h1
    exact hs (hO.2 h) ⟨rfl, ha ▸ h2⟩
  · rw [hr] at h
    cases h
  · simp [helloSpecs, hr]
  · rw [hO]
    constructor
    · rintro ⟨q, hq, i, s, n, h⟩
      exact ⟨q, List.mem_append_left _ hq, i, s, n, h⟩
    · rintro ⟨q, hq, i, s, n, h⟩
      rcases List.mem_append.1 hq with hq | hq
      · exact ⟨q, hq, i, s, n, h⟩
      · rw [List.mem_singleton.1 hq, hr] at h
        cases h

/-- **v2's read of a log in order is in order**: where the release log's lines after its head pass `sampleLinkFirst`'s
scan, the releases `readTail` reads from them, after the releases `pre` read before them, show the sample, then the
link, then the other rounds. -/
theorem sampleLinkFirst_readTail {label : String} {after : ByteArray} {records : List Json} :
    ∀ (lines : List String) (ts : List (String × Json)) (pending : Option String) (ξ : ByteArray)
      (opened : Bool) (pre : List Verity.Release) {log : List Verity.Release},
      sampleLinkFirstFrom label after opened pending lines = true →
      readTail label after records lines ts pending ξ = some log →
      List.Forall₂ (fun t sp => specOf t.2 = some sp) ts (helloSpecs pre) →
      (opened = true ↔ ∃ q ∈ pre, ∃ i s n, q.frame = .opening i s n) →
      SampleLinkFirst label after pre → SampleLinkFirst label after (pre ++ log)
  | [], _, none, _, _, pre, log, _, hR, _, _, hP => by
    simp only [readTail, Option.some.injEq] at hR
    subst hR
    simpa using hP
  | [], _, some _, _, _, _, _, _, hR, _, _, _ => by simp [readTail] at hR
  | l :: ls, ts, pending, ξ, opened, pre, log, hB, hR, hS, hO, hP => by
    unfold readTail at hR
    unfold sampleLinkFirstFrom at hB
    by_cases ht : treeLine l = true
    · rw [ite_eq_left ht] at hR hB
      simp only [Option.bind_eq_bind] at hR
      obtain ⟨⟨r, ts'⟩, hX, hR⟩ := Option.bind_eq_some_iff.1 hR
      obtain ⟨j, hj, hX⟩ := Option.bind_eq_some_iff.1 hX
      simp only [Option.pure_def] at hR
      obtain ⟨rest, hrest, hR⟩ := Option.bind_eq_some_iff.1 hR
      cases hR
      have hev : event l = .ok j := by
        cases he : event l with
        | ok j' =>
          rw [he] at hj
          simp only [Except.toOption, Option.some.injEq] at hj
          rw [hj]
        | error _ =>
          rw [he] at hj
          simp [Except.toOption] at hj
      have htl : (Spot.jsonStrIs (Spot.getD j "event") TREE_EVENT ||
          Spot.jsonStrIs (Spot.getD j "event") OPEN_EVENT) = true := by
        simpa only [treeLine, hev] using ht
      have hol : openLine l = Spot.jsonStrIs (Spot.getD j "event") OPEN_EVENT := by
        simp only [openLine, hev]
      have hll : linkLine l = (Spot.jsonStrIs (Spot.getD j "event") OPEN_EVENT &&
          Spot.jsonStrIs (Spot.getD j "stream") TREE_SESSION) := by
        simp only [linkLine, hev]
      split at hB
      · exact absurd hB (by simp)
      rename_i hc
      have hIH := sampleLinkFirst_readTail ls ts' pending ξ (opened || openLine l) (pre ++ [r]) hB hrest
      rw [List.append_assoc, List.singleton_append] at hIH
      rcases treeStep_some hX with ⟨root, sj, spec, hr, hts, hsp, hte⟩ |
        ⟨i, s, n, hr, hts, hte, stream, hstream, hst⟩
      · have hoe : Spot.jsonStrIs (Spot.getD j "event") OPEN_EVENT = false := by
          rw [Trees.strIs hte]
          simp [Spot.jsonStrIs, TREE_EVENT, OPEN_EVENT]
        rw [hol, hoe, Bool.or_false] at hIH
        refine hIH ?_ ?_ (sampleLinkFirst_snoc hP (fun _ ⟨h1, _⟩ => ?_) (fun i s n h => ?_))
        · have h1 : helloSpecs [r] = [spec] := by simp [helloSpecs, hr]
          rw [helloSpecs_append, h1, hts]
          exact forall₂_snoc hS hsp
        · rw [hO]
          constructor
          · rintro ⟨q, hq, i, s, n, h⟩
            exact ⟨q, List.mem_append_left _ hq, i, s, n, h⟩
          · rintro ⟨q, hq, i, s, n, h⟩
            rcases List.mem_append.1 hq with hq | hq
            · exact ⟨q, hq, i, s, n, h⟩
            · rw [List.mem_singleton.1 hq, hr] at h
              cases h
        · rw [hr] at h1
          cases h1
        · rw [hr] at h
          cases h
      · subst hts
        have hoe : Spot.jsonStrIs (Spot.getD j "event") OPEN_EVENT = true := by
          simpa [hte] using htl
        rw [hol, hoe, Bool.or_true] at hIH
        refine hIH ?_ ?_ (sampleLinkFirst_snoc hP (fun _ ⟨h1, _⟩ => ?_) (fun i' s' n' h hnone => ?_))
        · have h1 : helloSpecs [r] = [] := by simp [helloSpecs, hr]
          rw [helloSpecs_append, h1, List.append_nil]
          exact hS
        · exact ⟨fun _ => ⟨r, by simp, i, s, n, hr⟩, fun _ => rfl⟩
        · rw [hr] at h1
          cases h1
        · obtain ⟨rfl, rfl, rfl⟩ := Verity.Frame.opening.inj (hr.symm.trans h)
          have hop : opened = false := by
            cases hb : opened
            · rfl
            · obtain ⟨q, hq, i'', s'', n'', h''⟩ := hO.1 hb
              exact absurd h'' (hnone q hq i'' s'' n'')
          have hlk : linkLine l = true := by
            cases hb : linkLine l
            · exact absurd ⟨hop, by rw [hol, hoe], hb⟩ hc
            · rfl
          rw [hll, hoe, Bool.true_and] at hlk
          rw [hstream] at hlk
          obtain rfl : stream = TREE_SESSION := by simpa [Spot.jsonStrIs] using hlk
          obtain ⟨t, hti, hts'⟩ := Option.bind_eq_some_iff.1 hst
          obtain ⟨sp, hsp, hR⟩ := forall₂_getElem? hS hti
          rw [hsp, Option.bind_some]
          exact specOf_streams hR hts'
    · rw [ite_eq_right ht] at hR hB
      cases pending with
      | none => exact sampleLinkFirst_readTail ls ts (some l) ξ opened pre hB hR hS hO hP
      | some a =>
        simp only [Option.bind_eq_bind] at hR
        obtain ⟨⟨lab, root⟩, hra, hR⟩ := Option.bind_eq_some_iff.1 hR
        simp only [Option.pure_def] at hR
        dsimp only at hB
        split at hB
        · exact absurd hB (by simp)
        rename_i hc
        have hs : opened = true → ¬ (lab = label ∧ root = after) := fun ho hm =>
          hc ⟨ho, by rw [hra, hm.1, hm.2]⟩
        by_cases hm : lab = label ∧ root = after
        · rw [ite_eq_left hm] at hR
          obtain ⟨rest, hrest, hR⟩ := Option.bind_eq_some_iff.1 hR
          cases hR
          obtain ⟨hP', hS', hO'⟩ := sampleLinkFirst_snoc_fresh
            (r := ⟨.fresh label, after, Verity.bitsOfBytes (ξ.extract 0 COIN_BYTES)⟩) hP hO rfl rfl
            fun ho _ => hs ho hm
          have h := sampleLinkFirst_readTail ls ts none _ opened _ hB hrest (by rw [hS']; exact hS) hO' hP'
          rwa [List.append_assoc, List.singleton_append] at h
        · rw [ite_eq_right hm] at hR
          obtain ⟨rest, hrest, hR⟩ := Option.bind_eq_some_iff.1 hR
          cases hR
          obtain ⟨hP', hS', hO'⟩ := sampleLinkFirst_snoc_fresh (r := (⟨.fresh lab, root, []⟩ : Verity.Release)) hP hO
            rfl rfl hs
          have h := sampleLinkFirst_readTail ls ts none _ opened _ hB hrest (by rw [hS']; exact hS) hO' hP'
          rwa [List.append_assoc, List.singleton_append] at h

/-- **v2's read of a log in order is in order**: where the release log's lines after its head pass `sampleLinkFirst`,
the releases `readLog` reads from the whole log show the sample, then the link, then the other rounds. The head's line
releases at most its tree's `Hello`: an opening there is of no tree read before it, and the read refuses it. -/
theorem sampleLinkFirst_readLog {label : String} {after : ByteArray} {records : List Json} {ξ : ByteArray}
    {lines : List String} {log : List Verity.Release} (hB : sampleLinkFirst label after lines.tail = true)
    (hR : readLog label after records ξ lines = some log) : SampleLinkFirst label after log := by
  have h0 : SampleLinkFirst label after [] := fun pre r post h => absurd h (by simp)
  cases lines with
  | nil =>
    simp only [readLog, Option.some.injEq] at hR
    subst hR
    exact h0
  | cons l ls =>
    simp only [readLog] at hR
    by_cases ht : treeLine l = true
    · rw [ite_eq_left ht] at hR
      simp only [Option.bind_eq_bind] at hR
      obtain ⟨⟨r, ts⟩, hX, hR⟩ := Option.bind_eq_some_iff.1 hR
      obtain ⟨j, -, hX⟩ := Option.bind_eq_some_iff.1 hX
      simp only [Option.pure_def] at hR
      obtain ⟨rest, hrest, hR⟩ := Option.bind_eq_some_iff.1 hR
      cases hR
      rcases treeStep_some hX with ⟨root, sj, spec, hr, hts, hsp, -⟩ | ⟨_, _, _, _, _, _, _, _, hst⟩
      · have h1 : helloSpecs [r] = [spec] := by simp [helloSpecs, hr]
        have hS : List.Forall₂ (fun t sp => specOf t.2 = some sp) ts (helloSpecs [r]) := by
          rw [h1, hts, List.nil_append]
          exact .cons hsp .nil
        have hP : SampleLinkFirst label after [r] :=
          sampleLinkFirst_snoc h0 (by simp) fun i s n h => by rw [hr] at h; cases h
        exact sampleLinkFirst_readTail ls ts none ξ false [r] hB hrest hS (by simp [hr]) hP
      · simp at hst
    · rw [ite_eq_right ht] at hR
      exact sampleLinkFirst_readTail ls [] none ξ false [] hB hR .nil (by simp) h0

theorem forall₂_split {α β : Type} {R : α → β → Prop} {r' : β} {post' : List β} :
    ∀ {pre' : List β} {l : List α}, List.Forall₂ R l (pre' ++ r' :: post') →
      ∃ pre r post, l = pre ++ r :: post ∧ List.Forall₂ R pre pre' ∧ R r r' ∧ List.Forall₂ R post post'
  | [], _, .cons h t => ⟨[], _, _, rfl, .nil, h, t⟩
  | _ :: _, _, .cons h t =>
    let ⟨pre, r, post, he, hp, hr, hq⟩ := forall₂_split t
    ⟨_ :: pre, r, post, by rw [he]; rfl, .cons h hp, hr, hq⟩

theorem helloSpecs_congr :
    ∀ {l l' : List Verity.Release}, List.Forall₂ (fun x y => x.frame = y.frame ∧ x.answers = y.answers) l l' →
      helloSpecs l = helloSpecs l'
  | _, _, .nil => rfl
  | x :: _, y :: _, .cons h t => by
    have e := helloSpecs_congr t
    simp only [helloSpecs, List.filterMap_cons, h.1] at e ⊢
    rw [e]

/-- **Releases of the same frames, answering the same bytes, are in order together**: `SampleLinkFirst` reads only
each release's frame and the bytes it answered. -/
theorem sampleLinkFirst_congr {label : String} {after : ByteArray} {l l' : List Verity.Release}
    (hl : List.Forall₂ (fun x y => x.frame = y.frame ∧ x.answers = y.answers) l l')
    (hP : SampleLinkFirst label after l) : SampleLinkFirst label after l' := by
  intro pre' r' post' heq i s n hr'
  subst heq
  obtain ⟨pre, r, post, rfl, hpre, hr, hpost⟩ := forall₂_split hl
  obtain ⟨hP1, hP2⟩ := hP pre r post rfl i s n (hr.1.trans hr')
  refine ⟨fun q' hq' ⟨h1, h2⟩ => ?_, fun hnone => ?_⟩
  · obtain ⟨q, hq, hqq⟩ := Properties.Spot.SecurityProofs.forall₂_mem_right hpost q' hq'
    exact hP1 q hq ⟨hqq.1.trans h1, hqq.2.trans h2⟩
  · rw [← helloSpecs_congr hpre]
    refine hP2 fun q hq i' s' n' hqo => ?_
    obtain ⟨q', hq', hqq⟩ := Claims.Bridge.Custody.forall₂_mem_left hpre q hq
    exact hnone q' hq' i' s' n' (hqq.1.symm.trans hqo)

/-! ## v2's case in what the prover sent -/

theorem ok_bind {ε α β : Type} (a : α) (f : α → Except ε β) : (Except.ok a >>= f) = f a := rfl

/-- **The case `caseOf` finds is the one v2 accepted**, in a composition composite `k` accepted. -/
theorem caseOf_of {k : Verity.Registration.Composite} {s : Verity.Sent} {a b e : ByteArray}
    {p : _root_.Spot.OneStageZk.Params} {c : _root_.Spot.OneStage.Commitment}
    {ev : _root_.Spot.OneStageZk.Evidence} (hS : Verity.SameAccepts k s) (hcase : V2Case k s a b e)
    (hp : (Core.canonicalJson a >>= VStar.Recursive.Sampled.paramsOfJson) = .ok (.inl p))
    (hc : _root_.Spot.OneStage.commitmentOf b = .ok c)
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
theorem source_of {p : _root_.Spot.OneStageZk.Params} {c : _root_.Spot.OneStage.Commitment}
    {dr : Draws} (h : Properties.Challenger.oneStageZkSource p c = some dr) :
    ∃ key, p.base.challenger = some (key, dr.label) ∧ dr.after = c.digest := by
  unfold Properties.Challenger.oneStageZkSource Properties.Challenger.oneStageSource at h
  rcases hch : p.base.challenger with _ | ⟨key, label⟩ <;> rcases hw : p.base.verifier.windowRoot with _ | w <;>
    simp only [hch, hw] at h <;> try cases h
  exact ⟨key, rfl, rfl⟩

theorem linesOf_of {e : _root_.Spot.OneStageZk.Evidence} {ls : Array Json} {lines : List String}
    (hlog : _root_.Spot.getD (_root_.Spot.getD e.base.coins "session") "log" = .arr ls)
    (hls : ls.toList = lines.map .str) : linesOf e = .ok lines := by
  unfold linesOf
  rw [hlog]
  show ls.toList.mapM _ = _
  rw [hls]
  exact mapM_str_ok (fun _ => rfl) lines

/-! ## The session's law -/

/-- **v2's parameters in the parameters sent are the ones its case accepted**, in a composition composite `k`
accepted. -/
theorem paramsAt_case {k : Verity.Registration.Composite} {s : Verity.Sent} {a b e : ByteArray}
    {p : _root_.Spot.OneStageZk.Params} {c : _root_.Spot.OneStage.Commitment}
    {ev : _root_.Spot.OneStageZk.Evidence} {ξ : ByteArray} {dr : Draws} (hS : Verity.SameAccepts k s)
    (hcase : V2Case k s a b e) (hacc : V2Accepts (Core.Blobs.ofArray s.blobs) a b e p c ev ξ dr)
    {p' : _root_.Spot.OneStageZk.Params} (h : ParamsAt s.params p') : p' = p :=
  (Sum.inl.inj (params_of_case hS hcase hacc.1 h)).symm

/-- The sampling design `lawAt` reads at the population `LawOf` names is the sampling design `LawOf` names. -/
theorem lawAt_of_lawOf {p : _root_.Spot.OneStageZk.Params} {c : _root_.Spot.OneStage.Commitment}
    {L : Flock.Draw.Law} {n : ℕ} (h : LawOf p c L n) : lawAt p.base.verifier n = .ok L := by
  obtain ⟨law, derived, fl, hd, -, hder, hfl, rfl⟩ := h
  simp only [lawAt, hd, hder, hfl, ok_bind]
  rfl

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **The session's law is the sample's** where the session's population is: at the session `A` v2 sets up on the
parameters `θ` (`SetsUp`), a sample v2 made in a composition composite `k` accepted, on those parameters, over `A.n`
units, ran under `A.design`. -/
theorem law_of_setsUp {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {A : Live} (hA : SetsUp V θ F A)
    {k : Verity.Registration.Composite} {s : Verity.Sent} {dr : Draws} {L : Flock.Draw.Law} {n : ℕ} {ξ : ByteArray}
    {units : Array ℕ} (hS : Verity.SameAccepts k s) (hθ : s.params = θ) (hd : Drew k s dr L n ξ units)
    (hn : A.n = n) : A.design = L := by
  obtain ⟨a, b, e, p, c, ev, hcase, hacc, hL, -, -⟩ := hd
  obtain ⟨p', hp', -, -, hlaw, -⟩ := hA.params
  rw [← hθ] at hp'
  rw [paramsAt_case hS hcase hacc hp', hn, lawAt_of_lawOf hL] at hlaw
  exact (Except.ok.inj hlaw).symm

/-- **The sample's piece of `CouplingTarget` at the session v2 sets up**: `coupled_draw`, with the session's law read
off `SetsUp` on the play's parameters. -/
theorem coupled_draw_setsUp {V : Vocab d α} {k : Verity.Registration.Composite} (K : Constructions V) (A : Live)
    (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c))
    (hA : SetsUp V θ F A) (hR : DrawRead k (V.coinsRead .sampled)) {o : Verity.Exchange}
    (hθ : o.params = θ) (hv : V.Verdict k .sampled o) (hl : V.Logged .sampled o) :
    ∃ dr L n ξ units, Drew k o.sent dr L n ξ units ∧
      ∀ ω : A.Ω × (K.coins A θ F c σ).T, A.n = n →
        (freshCoins dr.label dr.after o.log).flatten <+:
          bitsOf (_root_.Properties.Spot.bytesOf (A.src ω.1 0)) →
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
    (hsrc : ∀ ω, _root_.Properties.Spot.bytesOf (A.src ω 0) =
      _root_.Properties.Spot.bytesOf (coinBytes m B (κ ω) 0)) :
    ∃ dr L n ξ units, Drew k o.sent dr L n ξ units ∧
      ∀ ω : A.Ω × (K.coins A θ F c σ).T, A.n = n →
        freshCoins dr.label dr.after o.log <+: (List.ofFn fun j => List.ofFn (κ ω.1 j)) →
        (K.live A θ F c σ).L.draw ω = Finset.univ.filter fun u => u.val ∈ units := by
  obtain ⟨p₀, hp₀, -⟩ := hA.params
  obtain ⟨dr, L, n, ξ, units, hd, h⟩ := coupled_draw_coins K A θ F c σ hR (by rw [hθ]; exact hp₀) hv hl κ hsrc
  exact ⟨dr, L, n, ξ, units, hd, fun ω hn hp => h ω (law_of_setsUp hA hv.2.1 hθ hd hn) hn hp⟩

/-! ## The session's population -/

/-- **v2's check at the coins `ξ` accepts only where the record's population is its program's** (`popHeld`). -/
theorem popHeld_of_checkBlobsAt {p : _root_.Spot.OneStageZk.Params}
    {c : _root_.Spot.OneStage.Commitment} {ξ : ByteArray} {e : _root_.Spot.OneStageZk.Evidence}
    {bs : Core.Blobs} {pub : _root_.Spot.OneStage.Public} {prof : Core.Integrity}
    (h : _root_.Spot.OneStageZk.checkBlobsAt p c ξ e bs = .accept pub prof) :
    _root_.Spot.OneStageZk.popHeld p.base.verifier c.record = .ok () := by
  replace h := (_root_.Spot.OneStageZk.checkBlobsAt_accept h).1
  unfold _root_.Spot.OneStageZk.checkOneAt at h
  try simp only [] at h
  split at h
  · cases h
  · rename_i hr
    cases hp : _root_.Spot.OneStageZk.popHeld p.base.verifier c.record with
    | ok u => rfl
    | error err =>
      exfalso
      simp only [hp, bind_ok, reduceCtorEq, false_and, and_false, exists_false] at hr

/-- **The population `popHeld` passes is one the program holds**: under the verifier's query, the program's proof
blocks are the record's population units in order (`Slots.popOk`). -/
theorem popOk_of_popHeld {v : _root_.Spot.Registration.Verifier} {record : Json} {n : ℕ}
    (h : _root_.Spot.OneStageZk.popHeld v record = .ok ())
    (hn : _root_.Spot.nat? (_root_.Spot.getD record "population") = some n) :
    _root_.Spot.Slots.popOk v.program v.query n = true := by
  unfold _root_.Spot.OneStageZk.popHeld at h
  simp only [hn] at h
  by_cases hp : _root_.Spot.Slots.popOk v.program v.query n = true
  · exact hp
  · rw [ite_eq_right hp] at h
    cases h

/-- **The session's population is the one the program holds** where the session's descriptor parses to v2's program
and its query is v2's: `Live.hpop` and `Slots.popOk` read one layout's population. -/
theorem n_of_popOk {A : Live} {v : _root_.Spot.Registration.Verifier} {n : ℕ}
    (hd : _root_.Spot.Program.parse ⟨A.desc.toArray⟩ = .ok v.program) (hq : A.qr = v.query)
    (h : _root_.Spot.Slots.popOk v.program v.query n = true) : A.n = n := by
  unfold _root_.Spot.Slots.popOk at h
  split at h
  · cases h
  · rename_i l hl
    have h1 := Flock.Bridge.popOf_units (l := l) A.hpop hd (by rw [hq]; exact hl)
    have h2 := of_decide_eq_true h
    have := congrArg List.length (h1.symm.trans h2)
    simpa only [List.length_map, List.length_range] using this

/-- **The session's descriptor and query are v2's verifier's** on the parameters `θ`: the tie under which the
session's population is the sample's (`n_of_popTied`). `SetsUp`'s `params` implies it. -/
def PopTied (θ : ByteArray) (A : Live) : Prop :=
  ∃ p, ParamsAt θ p ∧ _root_.Spot.Program.parse ⟨A.desc.toArray⟩ = .ok p.base.verifier.program ∧
    A.qr = p.base.verifier.query

/-- **The session's population is the sample's**: a sample v2 made in a composition composite `k` accepted, on the
parameters `θ`, ran over the population of a session tied to v2's verifier there, since v2 refuses a record whose
population isn't its program's proof blocks in order (`popHeld`). -/
theorem n_of_popTied {θ : ByteArray} {A : Live} (ht : PopTied θ A) {k : Verity.Registration.Composite}
    {s : Verity.Sent} {dr : Draws} {L : Flock.Draw.Law} {n : ℕ} {ξ : ByteArray} {units : Array ℕ}
    (hS : Verity.SameAccepts k s) (hθ : s.params = θ) (hd : Drew k s dr L n ξ units) : A.n = n := by
  obtain ⟨a, b, e, p, c, ev, hcase, hacc, hL, -, -⟩ := hd
  obtain ⟨p', hp', hdesc, hq⟩ := ht
  rw [← hθ] at hp'
  rw [paramsAt_case hS hcase hacc hp'] at hdesc hq
  obtain ⟨_, _, _, -, hn, -, -, -⟩ := hL
  obtain ⟨-, -, -, -, -, pub, prof, hck⟩ := hacc
  exact n_of_popOk hdesc hq (popOk_of_popHeld (popHeld_of_checkBlobsAt hck) hn)

/-- **The sample's piece of `CouplingTarget` at a session tied to v2's verifier**: `coupled_draw_setsUp`, with the
session's population read off `PopTied`, so the piece asks nothing of the session but its coins. -/
theorem coupled_draw_tied {V : Vocab d α} {k : Verity.Registration.Composite} (K : Constructions V) (A : Live)
    (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c))
    (hA : SetsUp V θ F A) (ht : PopTied θ A) (hR : DrawRead k (V.coinsRead .sampled)) {o : Verity.Exchange}
    (hθ : o.params = θ) (hv : V.Verdict k .sampled o) (hl : V.Logged .sampled o) :
    ∃ dr L n ξ units, Drew k o.sent dr L n ξ units ∧
      ∀ ω : A.Ω × (K.coins A θ F c σ).T,
        (freshCoins dr.label dr.after o.log).flatten <+:
          bitsOf (_root_.Properties.Spot.bytesOf (A.src ω.1 0)) →
        (K.live A θ F c σ).L.draw ω = Finset.univ.filter fun u => u.val ∈ units := by
  obtain ⟨dr, L, n, ξ, units, hd, h⟩ := coupled_draw_setsUp K A θ F c σ hA hR hθ hv hl
  exact ⟨dr, L, n, ξ, units, hd, fun ω hp => h ω (n_of_popTied ht hv.2.1 hθ hd) hp⟩

/-- `coupled_draw_coins_setsUp` at a session tied to v2's verifier. -/
theorem coupled_draw_coins_tied {V : Vocab d α} {k : Verity.Registration.Composite} (K : Constructions V)
    (A : Live) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c))
    (hA : SetsUp V θ F A) (ht : PopTied θ A) (hR : DrawRead k (V.coinsRead .sampled)) {o : Verity.Exchange}
    (hθ : o.params = θ) (hv : V.Verdict k .sampled o) (hl : V.Logged .sampled o) {m B : ℕ}
    (κ : A.Ω → Fin m → Fin (8 * B) → Bool)
    (hsrc : ∀ ω, _root_.Properties.Spot.bytesOf (A.src ω 0) =
      _root_.Properties.Spot.bytesOf (coinBytes m B (κ ω) 0)) :
    ∃ dr L n ξ units, Drew k o.sent dr L n ξ units ∧
      ∀ ω : A.Ω × (K.coins A θ F c σ).T,
        freshCoins dr.label dr.after o.log <+: (List.ofFn fun j => List.ofFn (κ ω.1 j)) →
        (K.live A θ F c σ).L.draw ω = Finset.univ.filter fun u => u.val ∈ units := by
  obtain ⟨dr, L, n, ξ, units, hd, h⟩ := coupled_draw_coins_setsUp K A θ F c σ hA hR hθ hv hl κ hsrc
  exact ⟨dr, L, n, ξ, units, hd, fun ω hp => h ω (n_of_popTied ht hv.2.1 hθ hd) hp⟩

/-! ## The read with v2's own tests -/

open private isHexN strOf isStream from Spot.ChallengerTrees

/-! ### The trees -/

/-- The trees the coin tree lines of `ls` read (`treeStep`) after the trees `ts`; `none` where one doesn't read. -/
def treeFold (records : List Json) : List (String × Json) → List String → Option (List (String × Json))
  | ts, [] => some ts
  | ts, l :: ls =>
    if treeLine l then ((event l).toOption.bind (treeStep records ts)).bind fun x => treeFold records x.2 ls
    else treeFold records ts ls

theorem treeFold_append (records : List Json) :
    ∀ (ts : List (String × Json)) (l₁ l₂ : List String),
      treeFold records ts (l₁ ++ l₂) = (treeFold records ts l₁).bind fun ts' => treeFold records ts' l₂
  | _, [], _ => rfl
  | ts, l :: l₁, l₂ => by
    rw [List.cons_append]
    simp only [treeFold]
    by_cases h : treeLine l = true
    · rw [ite_eq_left h, ite_eq_left h, Option.bind_assoc ((event l).toOption.bind (treeStep records ts))]
      exact congrArg (Option.bind _) (funext fun x => treeFold_append records x.2 l₁ l₂)
    · rw [ite_eq_right h, ite_eq_right h]
      exact treeFold_append records ts l₁ l₂

/-- The trees `coinTreesOf` holds as `treeStep` reads them: each its root and its spec. -/
def heldTrees (trees : Array LoggedTree) : List (String × Json) := trees.toList.map fun t => (t.root, t.spec)

theorem map_modify_eq {α β : Type} (g : α → β) (f : α → α) (hf : ∀ x, g (f x) = g x) :
    ∀ (l : List α) (i : ℕ), (l.modify i f).map g = l.map g
  | [], i => by rw [List.modify_nil]
  | x :: l, 0 => by rw [List.modify_zero_cons, List.map_cons, List.map_cons, hf]
  | x :: l, i + 1 => by rw [List.modify_succ_cons, List.map_cons, List.map_cons, map_modify_eq g f hf l i]

theorem findIdx?_toList {α : Type} {xs : Array α} {p : α → Bool} : xs.toList.findIdx? p = xs.findIdx? p := by
  rcases xs with ⟨l⟩
  exact (List.findIdx?_toArray _ _).symm

theorem hex_of {j : Json} {n : ℕ} (h : isHexN j n = true) : ∃ s, j = .str s ∧ ∃ b, Flock.unhex s = .ok b := by
  cases j with
  | str s =>
    simp only [isHexN, Bool.and_eq_true] at h
    obtain ⟨-, hs⟩ := h
    revert hs
    cases hu : Flock.unhex s with
    | error e => intro hs; cases hs
    | ok b => intro _; exact ⟨s, rfl, b, hu⟩
  | _ => simp [isHexN] at h

theorem bind_isSome {α β : Type} {x : Option α} {f : α → Option β} (hx : x.isSome) (hf : ∀ a, (f a).isSome) :
    (x.bind f).isSome := by
  obtain ⟨a, rfl⟩ := Option.isSome_iff_exists.1 hx
  exact hf a

theorem arr_mapM_isSome {α β : Type} (f : α → Option β) (xs : Array α) (hf : ∀ x ∈ xs, (f x).isSome) :
    (xs.mapM f).isSome := by
  rw [Array.mapM_eq_mapM_toList]
  have h : ∀ l : List α, (∀ x ∈ l, (f x).isSome) → (l.mapM f).isSome := by
    intro l
    induction l with
    | nil => intro _; rfl
    | cons x l ih =>
      intro hl
      rw [List.mapM_cons]
      obtain ⟨b, hb⟩ := Option.isSome_iff_exists.1 (hl x (List.mem_cons_self ..))
      obtain ⟨r, hr⟩ := Option.isSome_iff_exists.1 (ih fun y hy => hl y (List.mem_cons_of_mem _ hy))
      rw [hb, hr]
      rfl
  obtain ⟨r, hr⟩ := Option.isSome_iff_exists.1 (h xs.toList fun x hx => hf x (Array.mem_toList_iff.1 hx))
  rw [hr]
  rfl

theorem specOf_of {j : Json} (h : isTreeSpec j = true) : (specOf j).isSome := by
  unfold isTreeSpec at h
  simp only [Bool.and_eq_true] at h
  obtain ⟨⟨⟨⟨-, -⟩, hb⟩, hr⟩, hs⟩ := h
  obtain ⟨b, hbk⟩ : ∃ b, Spot.nat? (Spot.getD j "block") = some b := by
    revert hb
    cases Spot.nat? (Spot.getD j "block") with
    | none => intro hb; cases hb
    | some b => intro _; exact ⟨b, rfl⟩
  obtain ⟨r, hrk⟩ : ∃ r, Spot.nat? (Spot.getD j "rounds_log") = some r := by
    revert hr
    cases Spot.nat? (Spot.getD j "rounds_log") with
    | none => intro hr; cases hr
    | some r => intro _; exact ⟨r, rfl⟩
  obtain ⟨ss, hsk, hall⟩ : ∃ ss, Spot.getD j "streams" = .arr ss ∧ ∀ x ∈ ss, isStream x = true := by
    revert hs
    cases Spot.getD j "streams" with
    | arr ss =>
      intro hs
      simp only [Bool.and_eq_true] at hs
      exact ⟨ss, rfl, Array.all_eq_true'.1 hs.1.2⟩
    | _ => intro hs; cases hs
  simp only [specOf, hsk, hrk, hbk, Option.bind_eq_bind, Option.bind_some, Option.pure_def]
  refine bind_isSome (arr_mapM_isSome _ ss fun x hx => ?_) fun _ => rfl
  have hx := hall x hx
  cases x with
  | str _ => rfl
  | _ => simp [isStream] at hx

theorem fold_skip {records : List Json} {pre : List String} {l : String} {ts : List (String × Json)}
    (hI : treeFold records [] pre = some ts) (ht : ¬treeLine l = true) :
    treeFold records [] (pre ++ [l]) = some ts := by
  rw [treeFold_append, hI, Option.bind_some]
  simp only [treeFold, ite_eq_right ht]

theorem fold_hello {records : List Json} {pre : List String} {l : String} {trees : Array LoggedTree} {d : Json}
    {a : Bool} {t : LoggedTree} (hI : treeFold records [] pre = some (heldTrees trees)) (ht : treeLine l = true)
    (he : event l = .ok d) (hT : Spot.jsonStrIs (Spot.getD d "event") TREE_EVENT = true)
    (hc : (a && isHexN (Spot.getD d "key") 512 && isHexN (Spot.getD d "nonce") 64 &&
      isHexN (Spot.getD d "root") 128 && isTreeSpec (Spot.getD d "spec")) = true)
    (hr : t.root = strOf (Spot.getD d "root")) (hs : t.spec = Spot.getD d "spec") :
    treeFold records [] (pre ++ [l]) = some (heldTrees (trees.push t)) := by
  simp only [Bool.and_eq_true] at hc
  obtain ⟨⟨⟨⟨-, hk⟩, hn⟩, hro⟩, hsp⟩ := hc
  obtain ⟨key, hk, κ, hκ⟩ := hex_of hk
  obtain ⟨nonce, hn, ν, hν⟩ := hex_of hn
  obtain ⟨root, hro, ρ, hρ⟩ := hex_of hro
  obtain ⟨spec, hspec⟩ := Option.isSome_iff_exists.1 (specOf_of hsp)
  have hp : heldTrees (trees.push t) = heldTrees trees ++ [(root, Spot.getD d "spec")] := by
    simp only [heldTrees, Array.toList_push, List.map_append, List.map_cons, List.map_nil, hr, hs, hro, strOf]
  rw [treeFold_append, hI, Option.bind_some, hp]
  simp only [treeFold, ite_eq_left ht, he, Except.toOption, Option.bind_some, treeStep, ite_eq_left hT, hk, hn, hro,
    hspec, hκ, hν, hρ, Option.bind_eq_bind, Option.pure_def]

theorem fold_open {records : List Json} {pre : List String} {l : String} {trees : Array LoggedTree} {d : Json}
    {root s : String} {i k n : ℕ} {t : LoggedTree} {f : LoggedTree → LoggedTree} {a₁ a₂ a₃ a₄ : Bool}
    (hI : treeFold records [] pre = some (heldTrees trees)) (ht : treeLine l = true) (he : event l = .ok d)
    (hT : ¬Spot.jsonStrIs (Spot.getD d "event") TREE_EVENT = true)
    (hroot : Spot.getD d "tree" = .str root) (hi : trees.findIdx? (·.root == root) = some i)
    (hti : trees[i]? = some t) (hs : Spot.getD d "stream" = .str s)
    (hk : Spot.nat? (Spot.getD d "round") = some k)
    (hn : Spot.nat? (Spot.getD d "n") = some n)
    (hc : ((specStreams t.spec).any (Spot.jsonStrIs · s) && a₁ && a₂ && a₃ &&
      isHexN (Spot.getD d "after") 128 && a₄) = true)
    (hf : ∀ x, (f x).root = x.root ∧ (f x).spec = x.spec) :
    treeFold records [] (pre ++ [l]) = some (heldTrees (trees.modify i f)) := by
  have hmod : heldTrees (trees.modify i f) = heldTrees trees := by
    unfold heldTrees
    rw [Array.toList_modify]
    exact map_modify_eq _ f (fun x => Prod.ext (hf x).1 (hf x).2) _ _
  simp only [Bool.and_eq_true] at hc
  obtain ⟨⟨⟨⟨⟨hcont, -⟩, -⟩, -⟩, haft⟩, -⟩ := hc
  obtain ⟨aft, haft, b, hb⟩ := hex_of haft
  have hi' : (heldTrees trees).findIdx? (·.1 == root) = some i := by
    unfold heldTrees
    rw [List.findIdx?_map]
    exact findIdx?_toList.trans hi
  have hti' : (heldTrees trees)[i]? = some (t.root, t.spec) := by
    unfold heldTrees
    rw [List.getElem?_map, Array.getElem?_toList, hti]
    rfl
  have hsome : ((specStreams t.spec).findIdx? (Spot.jsonStrIs · s)).isSome := by
    rw [Array.findIdx?_isSome]
    exact hcont
  obtain ⟨si, hsi⟩ := Option.isSome_iff_exists.1 hsome
  rw [treeFold_append, hI, Option.bind_some, hmod]
  simp only [treeFold, ite_eq_left ht, he, Except.toOption, Option.bind_some, treeStep, ite_eq_right hT, hroot, hs,
    haft, hk, hn, hi', hti', hsi, hb, Option.bind_eq_bind, Option.pure_def]

theorem coinTreesOf_spec (l : List String) : ⦃⌜True⌝⦄ coinTreesOf l ⦃⇓? r => ⌜coinTreesOf l = .ok r⌝⦄ := ok_spec _

set_option maxHeartbeats 4000000 in
/-- **The trees v2's check reads, the read reads**: wherever `coinTreesOf` accepts the release log, each of its coin
tree's lines reads (`treeStep`), at the trees `coinTreesOf` holds. -/
theorem coinTreesOf_fold (records : List Json) {lines : List String} {trees : Array LoggedTree}
    (h : coinTreesOf lines = .ok trees) : treeFold records [] lines = some (heldTrees trees) := by
  refine of_triple (Q := fun trees => treeFold records [] lines = some (heldTrees trees)) ?_ h
  clear h
  mvcgen [coinTreesOf, -Spec.throw_Except, throw_spec, event_spec] invariants
  · ⇓? ⟨cur, trees⟩ => ⌜treeFold records [] cur.prefix = some (heldTrees trees)⌝
  all_goals first
    | exact fold_skip ‹_› ‹_›
    | exact fold_hello ‹_› ‹_› ‹_› ‹_› ‹_› rfl rfl
    | exact fold_open ‹_› ‹_› ‹_› ‹_› ‹_› ‹_› ‹_› ‹_› ‹_› ‹_› ‹_› fun _ => ⟨rfl, rfl⟩
    | exact fold_open ‹_› ‹_› ‹_› ‹_› ‹_› ‹_› ‹_› ‹_› (Prod.mk.inj ‹_›).1 (Prod.mk.inj ‹_›).2 ‹_›
        fun _ => ⟨rfl, rfl⟩

/-! ### The pairs -/

/-- The pairs `releasedRootsOf` reads off the lines `ls`, after the root's line `pending` still to pair: each root's
line reads with its family and root (`rootOf`), and none is left without its release. -/
def PairsOk : Option String → List String → Prop
  | none, [] => True
  | some _, [] => False
  | none, l :: ls => PairsOk (some l) ls
  | some a, _ :: ls => (rootOf a).isSome ∧ PairsOk none ls

theorem pairsOk_of : ∀ rest : List String, (∀ x ∈ pairs rest, (rootOf x.1).isSome) → rest.length % 2 = 0 →
    PairsOk none rest
  | [], _, _ => trivial
  | [_], _, h => by simp at h
  | a :: b :: rest, hp, h => by
    have e : pairs (a :: b :: rest) = (a, b) :: pairs rest := rfl
    rw [e] at hp
    show (rootOf a).isSome ∧ PairsOk none rest
    refine ⟨hp (a, b) (List.mem_cons_self ..), pairsOk_of rest (fun x hx => hp x (List.mem_cons_of_mem _ hx)) ?_⟩
    simp only [List.length_cons] at h
    omega

theorem pairs_step {pre : List (String × String)} {x : String × String} {r : Json} {label root : String}
    {bytes : ByteArray} (hI : ∀ y ∈ pre, (rootOf y.1).isSome) (he : event x.1 = .ok r)
    (hl : Spot.getD r "label" = .str label) (hr : Spot.getD r "root" = .str root)
    (hb : (Flock.unhex root).toOption = some bytes) : ∀ y ∈ pre ++ [x], (rootOf y.1).isSome := by
  intro y hy
  rcases List.mem_append.1 hy with hy | hy
  · exact hI y hy
  · rw [List.mem_singleton.1 hy, rootOf_eq he hl hr hb]
    rfl

theorem parity {n : ℕ} (h : ¬(n % 2 == 1) = true) : n % 2 = 0 := by
  simp only [beq_iff_eq] at h
  omega

set_option maxHeartbeats 2000000 in
/-- **The pairs v2's check reads, the read reads**: wherever `releasedRootsOf` accepts the release log, each root's
line after its head, its coin trees' lines aside, reads with its family and root, and none is left without its
release. -/
theorem releasedRootsOf_pairsOk {c : LiveCommitment} {log : List String} {roots : Roots}
    (h : releasedRootsOf c log = .ok roots) : PairsOk none (log.tail.filter fun l => !treeLine l) := by
  refine of_triple (Q := fun _ => PairsOk none (log.tail.filter fun l => !treeLine l)) ?_ h
  clear h
  mvcgen [releasedRootsOf, -Spec.throw_Except, throw_spec, event_spec] invariants
  · ⇓? ⟨cur, _⟩ => ⌜∀ y ∈ cur.prefix, (rootOf y.1).isSome⌝
  all_goals first
    | exact pairs_step ‹_› ‹_› ‹_› ‹_› ‹_›
    | exact fun _ hy => nomatch hy
    | exact pairsOk_of _ ‹_› (parity ‹_›)

/-! ### The records -/

theorem blobsGet_spec (bs : _root_.Core.Blobs) (h w : String) :
    ⦃⌜True⌝⦄ bs.get h w ⦃⇓? r => ⌜bs.get h w = .ok r⌝⦄ := ok_spec _

theorem records_step {bs : _root_.Core.Blobs} {pre : List _root_.Spot.OneStageZk.RunFiles}
    {r : _root_.Spot.OneStageZk.RunFiles} {b : ByteArray} {t : String}
    (hI : (pre.mapM (recordOn bs)).isSome) (hg : bs.get r.record s!"session {r.name}'s record" = .ok b)
    (hu : String.fromUTF8? b = some t) : ((pre ++ [r]).mapM (recordOn bs)).isSome := by
  obtain ⟨rs, hrs⟩ := Option.isSome_iff_exists.1 hI
  rw [List.mapM_append, hrs]
  simp [recordOn, hg, hu, Except.toOption]

set_option maxHeartbeats 1000000 in
/-- **The records v2's check reads, the read reads**: wherever `treesHeld` accepts, a blob holds each session's
record, as UTF-8. -/
theorem treesHeld_records {p : _root_.Spot.OneStageZk.Own} {ts : Array LoggedTree}
    {e : _root_.Spot.OneStageZk.Evidence} {bs : _root_.Core.Blobs}
    (h : _root_.Spot.OneStageZk.treesHeld p ts e bs = .ok ()) :
    (recordsOn bs e).isSome := by
  refine of_triple (Q := fun _ => (recordsOn bs e).isSome) ?_ h
  clear h
  mvcgen [_root_.Spot.OneStageZk.treesHeld, -Spec.throw_Except, throw_spec, blobsGet_spec] invariants
  · ⇓? ⟨cur, _⟩ => ⌜(cur.prefix.mapM (recordOn bs)).isSome⌝
  all_goals exact records_step ‹_› ‹_› ‹_›

/-! ### The check's trees -/

set_option maxHeartbeats 1000000 in
theorem trustedTreesOf_ok {key : ChallengerKey} {w : String} {coins : Json} {ts : Array LoggedTree}
    (h : trustedTreesOf key w coins = .ok ts) :
    ∃ ls lines, Spot.getD (Spot.getD coins "session") "log" = .arr ls ∧
      ls.toList = lines.map .str ∧ coinTreesOf lines = .ok ts := by
  refine of_triple (Q := fun ts => ∃ ls lines, Spot.getD (Spot.getD coins "session") "log" = .arr ls ∧
    ls.toList = lines.map .str ∧ coinTreesOf lines = .ok ts) ?_ h
  clear h
  mvcgen [trustedTreesOf, -Spec.throw_Except, throw_spec, trustedSessionOf_spec, mapM_ok_spec, coinTreesOf_spec]
  all_goals exact fun h => ⟨_, _, ‹_›, mapM_str (fun j s h => logStr j s h) ‹_›, h⟩

theorem treesOf_ok {p : _root_.Spot.OneStageZk.Params} {e : _root_.Spot.OneStageZk.Evidence}
    {ts : Array LoggedTree} (h : _root_.Spot.OneStageZk.treesOf p.base e.base = .ok ts) :
    ∃ key w, trustedTreesOf key w e.base.coins = .ok ts := by
  unfold _root_.Spot.OneStageZk.treesOf at h
  rcases hch : p.base.challenger with _ | ⟨key, l⟩ <;> rcases hw : p.base.verifier.windowRoot with _ | w <;>
    simp only [hch, hw] at h <;> try cases h
  refine ⟨key, w, ?_⟩
  revert h
  cases trustedTreesOf key w e.base.coins with
  | error err => intro h; cases h
  | ok t => intro h; cases h; rfl

/-- **v2's check at the coins `ξ` accepts only where the challenger's trees hold**: `treesOf` reads them and
`treesHeld` holds the evidence's sessions to them. -/
theorem trees_of_checkBlobsAt {p : _root_.Spot.OneStageZk.Params}
    {c : _root_.Spot.OneStage.Commitment} {ξ : ByteArray} {e : _root_.Spot.OneStageZk.Evidence}
    {bs : _root_.Core.Blobs} {pub : _root_.Spot.OneStage.Public} {prof : _root_.Core.Integrity}
    (h : _root_.Spot.OneStageZk.checkBlobsAt p c ξ e bs = .accept pub prof) :
    ∃ ts, _root_.Spot.OneStageZk.treesOf p.base e.base = .ok ts ∧
      _root_.Spot.OneStageZk.treesHeld p.own ts e bs = .ok () := by
  replace h := (_root_.Spot.OneStageZk.checkBlobsAt_accept h).1
  unfold _root_.Spot.OneStageZk.checkOneAt at h
  try simp only [] at h
  split at h
  · cases h
  · rename_i hr
    simp only [bind_ok] at hr
    obtain ⟨_, -, ts, hts, _, -, _, hth, -⟩ := hr
    exact ⟨ts, hts, hth⟩

/-! ### The read -/

theorem logRead_eq {s : Verity.Sent} {p : _root_.Spot.OneStageZk.Params}
    {c : _root_.Spot.OneStage.Commitment} {e : _root_.Spot.OneStageZk.Evidence}
    {key : ChallengerKey} {label : String} {ξ : ByteArray} {lines : List String} (hcs : caseOf s = .ok (p, c, e))
    (hch : p.base.challenger = some (key, label)) (hξ : _root_.Spot.OneStageZk.coinsOf p c e = .ok ξ)
    (hl : linesOf e = .ok lines) :
    logRead s = (recordsOn (_root_.Core.Blobs.ofArray s.blobs) e).bind fun rs => readLog label c.digest rs ξ lines := by
  simp only [logRead, hcs, Except.toOption, Option.bind_eq_bind, Option.bind_some, hch, hξ, hl]

theorem readTail_isSome (label : String) (after : ByteArray) (records : List Json) :
    ∀ (ls : List String) (ts : List (String × Json)) (pending : Option String) (ξ : ByteArray),
      (treeFold records ts ls).isSome → PairsOk pending (ls.filter fun l => !treeLine l) →
        (readTail label after records ls ts pending ξ).isSome
  | [], _, none, _, _, _ => rfl
  | [], _, some _, _, _, h => (show False from h).elim
  | l :: ls, ts, pending, ξ, hT, hP => by
    unfold readTail
    by_cases ht : treeLine l = true
    · rw [ite_eq_left ht]
      rw [List.filter_cons_of_neg (by simp [ht])] at hP
      simp only [treeFold, ite_eq_left ht] at hT
      revert hT
      cases (event l).toOption.bind (treeStep records ts) with
      | none => intro hT; cases hT
      | some x =>
        intro hT
        obtain ⟨r, ts'⟩ := x
        exact bind_isSome (x := readTail label after records ls ts' pending ξ)
          (readTail_isSome label after records ls ts' pending ξ hT hP) fun _ => rfl
    · rw [ite_eq_right ht]
      rw [List.filter_cons_of_pos (by simp [ht])] at hP
      simp only [treeFold, ite_eq_right ht] at hT
      cases pending with
      | none => exact readTail_isSome label after records ls ts (some l) ξ hT hP
      | some a =>
        obtain ⟨hra, hP⟩ := hP
        obtain ⟨⟨lab, root⟩, hra⟩ := Option.isSome_iff_exists.1 hra
        simp only [Option.bind_eq_bind, hra, Option.bind_some]
        split
        · exact bind_isSome (x := readTail label after records ls ts none (ξ.extract COIN_BYTES ξ.size))
            (readTail_isSome label after records ls ts none _ hT hP) fun _ => rfl
        · exact bind_isSome (x := readTail label after records ls ts none ξ)
            (readTail_isSome label after records ls ts none ξ hT hP) fun _ => rfl

theorem readLog_isSome (label : String) (after : ByteArray) (records : List Json) (ξ : ByteArray)
    {lines : List String} (hT : (treeFold records [] lines).isSome)
    (hP : PairsOk none (lines.tail.filter fun l => !treeLine l)) : (readLog label after records ξ lines).isSome := by
  cases lines with
  | nil => rfl
  | cons l ls =>
    simp only [readLog]
    by_cases ht : treeLine l = true
    · rw [ite_eq_left ht]
      simp only [treeFold, ite_eq_left ht] at hT
      revert hT
      cases (event l).toOption.bind (treeStep records []) with
      | none => intro hT; cases hT
      | some x =>
        intro hT
        obtain ⟨r, ts⟩ := x
        exact bind_isSome (x := readTail label after records ls ts none ξ)
          (readTail_isSome label after records ls ts none ξ hT hP) fun _ => rfl
    · rw [ite_eq_right ht]
      simp only [treeFold, ite_eq_right ht] at hT
      exact readTail_isSome label after records ls [] none ξ hT hP

/-- **v2's acceptance reads the device's log, with v2's own tests**: wherever composite `k` accepted what the prover
sent and v2's case in it accepts, `logRead` returns a log. -/
theorem logRead_isSome {k : Verity.Registration.Composite} {s : Verity.Sent} {a b e : ByteArray}
    {p : _root_.Spot.OneStageZk.Params} {c : _root_.Spot.OneStage.Commitment}
    {ev : _root_.Spot.OneStageZk.Evidence} {ξ : ByteArray} {dr : Draws} (hS : Verity.SameAccepts k s)
    (hcase : V2Case k s a b e) (hacc : V2Accepts (Core.Blobs.ofArray s.blobs) a b e p c ev ξ dr) :
    (logRead s).isSome := by
  obtain ⟨hp, hc, hev, hξ, hdr, pub, prof, hck⟩ := hacc
  obtain ⟨key, hch, -⟩ := source_of hdr
  obtain ⟨raw, hraw, hrξ⟩ := Properties.Challenger.oneStageZk_reads p c dr hdr ev ξ hξ
  rw [hrξ] at hraw
  obtain ⟨cm, roots, htsn, -⟩ := drawCoins_ok hraw
  obtain ⟨_, hsn⟩ := trustedSessionOf_ok htsn
  obtain ⟨_, hso⟩ := sessionNamed_ok hsn
  obtain ⟨ls, lines, hlog, hls, hrr⟩ := sessionOf_ok hso
  have hl := linesOf_of hlog hls
  obtain ⟨ts, htr, hth⟩ := trees_of_checkBlobsAt hck
  obtain ⟨key', w, htt⟩ := treesOf_ok htr
  obtain ⟨ls', lines', hlog', hls', hct⟩ := trustedTreesOf_ok htt
  have hl' := linesOf_of hlog' hls'
  rw [hl] at hl'
  cases hl'
  obtain ⟨rs, hrs⟩ := Option.isSome_iff_exists.1 (treesHeld_records hth)
  rw [logRead_eq (caseOf_of hS hcase hp hc hev) hch hξ hl, hrs, Option.bind_some]
  refine readLog_isSome _ _ rs ξ ?_ (releasedRootsOf_pairsOk hrr)
  rw [coinTreesOf_fold rs hct]
  rfl

theorem freshCoins_cons_step {r : Verity.Release} (hf : ∀ l, r.frame ≠ .fresh l) (label : String)
    (after : ByteArray) (rest : List Verity.Release) :
    freshCoins label after (r :: rest) = freshCoins label after rest := by
  unfold freshCoins
  rw [List.filterMap_cons]
  cases hr : r.frame with
  | fresh l => exact absurd hr (hf l)
  | hello _ => rfl
  | opening _ _ _ => rfl

theorem fresh_readTail (label : String) (after : ByteArray) (records : List Json) :
    ∀ (ls : List String) (ts : List (String × Json)) (pending : Option String) (ξ : ByteArray)
      {log : List Verity.Release}, readTail label after records ls ts pending ξ = some log →
      freshCoins label after log = chunks (countFresh label after (pending.toList ++ ls.filter fun l => !treeLine l)) ξ
  | [], _, none, _, _, h => by
    simp only [readTail, Option.some.injEq] at h
    subst h
    rfl
  | [], _, some _, _, _, h => by simp [readTail] at h
  | l :: ls, ts, pending, ξ, log, h => by
    unfold readTail at h
    by_cases ht : treeLine l = true
    · rw [ite_eq_left ht] at h
      rw [List.filter_cons_of_neg (by simp [ht])]
      simp only [Option.bind_eq_bind] at h
      obtain ⟨⟨r, ts'⟩, hX, h⟩ := Option.bind_eq_some_iff.1 h
      obtain ⟨j, -, hj⟩ := Option.bind_eq_some_iff.1 hX
      simp only [Option.pure_def] at h
      obtain ⟨rest, hR, h⟩ := Option.bind_eq_some_iff.1 h
      cases h
      rw [freshCoins_cons_step (treeStep_frame hj)]
      exact fresh_readTail label after records ls ts' pending ξ hR
    · rw [ite_eq_right ht] at h
      rw [List.filter_cons_of_pos (by simp [ht])]
      cases pending with
      | none => exact fresh_readTail label after records ls ts (some l) ξ h
      | some a =>
        simp only [Option.toList_some, List.cons_append, List.nil_append]
        simp only [Option.bind_eq_bind] at h
        obtain ⟨⟨lab, root⟩, hra, h⟩ := Option.bind_eq_some_iff.1 h
        simp only [Option.pure_def] at h
        by_cases hm : lab = label ∧ root = after
        · rw [ite_eq_left hm] at h
          obtain ⟨rest, hR, h⟩ := Option.bind_eq_some_iff.1 h
          cases h
          have hc : rootOf a = some (label, after) := by rw [hra, hm.1, hm.2]
          rw [countFresh, ite_eq_left hc, chunks]
          simp only [freshCoins, List.filterMap_cons, and_self, ite_true]
          congr 1
          exact fresh_readTail label after records ls ts none _ hR
        · rw [ite_eq_right hm] at h
          obtain ⟨rest, hR, h⟩ := Option.bind_eq_some_iff.1 h
          cases h
          have hc : rootOf a ≠ some (label, after) := by
            rw [hra]
            exact fun e => hm (Prod.mk.inj (Option.some.inj e))
          rw [countFresh, ite_eq_right hc, Nat.add_zero]
          simp only [freshCoins, List.filterMap_cons, ite_eq_right hm]
          exact fresh_readTail label after records ls ts none ξ hR

theorem fresh_readLog (label : String) (after : ByteArray) (records : List Json) (ξ : ByteArray) :
    ∀ {lines : List String} {log : List Verity.Release}, readLog label after records ξ lines = some log →
      freshCoins label after log = chunks (countFresh label after (lines.tail.filter fun l => !treeLine l)) ξ
  | [], log, h => by
    simp only [readLog, Option.some.injEq] at h
    subst h
    rfl
  | l :: ls, log, h => by
    simp only [readLog] at h
    by_cases ht : treeLine l = true
    · rw [ite_eq_left ht] at h
      simp only [Option.bind_eq_bind] at h
      obtain ⟨⟨r, ts⟩, hX, h⟩ := Option.bind_eq_some_iff.1 h
      obtain ⟨j, -, hj⟩ := Option.bind_eq_some_iff.1 hX
      simp only [Option.pure_def] at h
      obtain ⟨rest, hR, h⟩ := Option.bind_eq_some_iff.1 h
      cases h
      rw [freshCoins_cons_step (treeStep_frame hj)]
      exact fresh_readTail label after records ls ts none ξ hR
    · rw [ite_eq_right ht] at h
      exact fresh_readTail label after records ls [] none ξ h

/-- **Where the read returns a log, it shows the sample**: wherever composite `k` accepted what the prover sent and v2's
case in it accepts at the coins `ξ` it read under the challenger `dr`, a log `logRead` returns has exactly `ξ`'s bits
as its coins of `dr`'s family released after `dr`'s digest. -/
theorem shows_logRead {k : Verity.Registration.Composite} {s : Verity.Sent} {a b e : ByteArray}
    {p : _root_.Spot.OneStageZk.Params} {c : _root_.Spot.OneStage.Commitment}
    {ev : _root_.Spot.OneStageZk.Evidence} {ξ : ByteArray} {dr : Draws} (hS : Verity.SameAccepts k s)
    (hcase : V2Case k s a b e) (hacc : V2Accepts (Core.Blobs.ofArray s.blobs) a b e p c ev ξ dr)
    {log : List Verity.Release} (h : logRead s = some log) : ShowsDraw log dr ξ := by
  obtain ⟨hp, hc, hev, hξ, hdr, -⟩ := hacc
  obtain ⟨key, hch, hafter⟩ := source_of hdr
  obtain ⟨raw, hraw, hrξ⟩ := Properties.Challenger.oneStageZk_reads p c dr hdr ev ξ hξ
  rw [hrξ] at hraw
  obtain ⟨cm, roots, hts, hstream⟩ := drawCoins_ok hraw
  obtain ⟨_, hsn⟩ := trustedSessionOf_ok hts
  obtain ⟨_, hso⟩ := sessionNamed_ok hsn
  obtain ⟨ls, lines, hlog, hls, hrr⟩ := sessionOf_ok hso
  rw [logRead_eq (caseOf_of hS hcase hp hc hev) hch hξ (linesOf_of hlog hls)] at h
  obtain ⟨rs, -, h⟩ := Option.bind_eq_some_iff.1 h
  rw [hafter] at hstream
  have hf : (freshCoins dr.label c.digest log).flatten = bitsOf ξ := by
    rw [fresh_readLog _ _ _ _ h, countFresh_eq, ← releasedRootsOf_ok hrr,
      chunks_flatten _ _ (challengerStream_ok hstream)]
  unfold ShowsDraw
  rw [hafter]
  exact hf ▸ List.prefix_refl _

/-- **The read with v2's own tests shows the sample**, with no premise: `logRead` meets the sample's ask (`DrawRead`). -/
theorem drawRead_logRead (k : Verity.Registration.Composite) : DrawRead k logRead := by
  intro s a b e p c ev ξ dr hS hcase hacc
  obtain ⟨log, hlog⟩ := Option.isSome_iff_exists.1 (logRead_isSome hS hcase hacc)
  exact ⟨log, hlog, shows_logRead hS hcase hacc hlog⟩

/-- **v2's read shows the sample**, with no premise: `coinsRead` is `logRead` (`coinsRead_eq_logRead`), so it meets the
sample's ask (`DrawRead`). -/
theorem drawRead_coinsRead (k : Verity.Registration.Composite) : DrawRead k coinsRead := by
  rw [coinsRead_eq_logRead]
  exact drawRead_logRead k

end Verity.Sampled
