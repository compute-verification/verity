module

public import Proofs.Flock.Recursive.FirewallTemplate
public import Proofs.Flock.Recursive.FirewallWire
public import Proofs.Flock.Recursive.FirewallSteps
import all Lean.Data.Json.Basic

/-!
# The honest worker's requests pass the ZK server

`run_honestReqs`: on the honest worker's requests (`Flock.Firewall.honestReqs`) the ZK server (`Flock.Firewall.run`), at
any salts as many as the schedule's rows (`rowCount`), finishes, its Commit names the schedule's tables
(`Schedule.tables`), and its commit strings are those of the worker's rows (`honestRows`: each table's level-0 top in
Commit's order, then each round's fresh tops and message) in salt order. It holds at every schedule the ZK server holds a
session to (`Canonical`) and every worker that follows it (`Fits`): its Commit's tables the schedule's roots in the
schedule's order, each stream opened once before its first round, and each round's caps, copies, fresh tops and message
the schedule's entry.

The proof follows the ZK server's loop (`FirewallSteps.Goes`) through the requests. What `Schedule.canonical` checks is
`Facts`. The Pings, Register, Hello, the first `Open`s, Commit and Link reach the loop's invariant (`At`): each salt of
the worker's rows before the next round sampled and its commit string sent, each opened stream with its count of rounds
so far (`cnt`). Each round, after the `Open`s just before it, keeps it (`goes_opens`, `goes_round`), and at its end
Finish's pin holds (`finish_pin`).

`honestMsgs` is the messages `FirewallAccepts` reads: message `i` the bytes `rowDigest` hashes of the worker's row `i`
at the commit strings before it, the worker's rows reading no commit string at or after their own (`Causal`).
`honestReqs_accepts` (`FirewallAccepts`) is `run_honestReqs` at zero salts.
-/

@[expose] public section

namespace FlockSoundness.Discharge.Recursion.FirewallHonest

open Lean (Json)
open Flock Flock.Firewall

/-! ## The schedule and the worker's rows -/

/-- The ZK server holds a session to the schedule (`Schedule.canonical`, which `schedule` checks at load). -/
def Canonical (s : Schedule) : Prop := s.canonical = true

/-- The schedule's roots: each table's name and its level-0 top's node count, in key order. -/
def rootsOf (s : Schedule) : List (String × Nat) :=
  match s.roots with
  | .obj ⟨⟨t⟩⟩ => t.toListModel.map fun e => (e.1, (natOf? e.2).getD 0)
  | _ => []

/-- Round `i`'s rows in the schedule: its fresh tops' nodes and its message rows. -/
def roundCount (s : Schedule) (i : Nat) : Nat :=
  match roundSpec (s.rounds.getD i .null) with
  | some (_, _, _, tops, _, commits) => tops.sum + commits
  | none => 0

/-- **The schedule's rows**, as many as the ZK server salts: its roots' nodes, then each round's. -/
def rowCount (s : Schedule) : Nat :=
  ((rootsOf s).map (·.2)).sum + ((List.range s.rounds.size).map (roundCount s)).sum

/-- A stream's parent in the schedule. -/
def parentOf (s : Schedule) (stream : String) : String := ((s.streams.find? (·.1 == stream)).map (·.2.1)).getD ""

/-- The worker's level-0 top of a stream's table, as its Commit gives it. -/
def rootOf (h : Honest) (stream : String) : Option ByteArray := (h.roots.find? (·.1 == tableOf stream)).map (·.2)

/-- A round's caps, in absorb order. -/
def capsOf (r : HonestRound) : List ByteArray := r.items.flatMap FirewallWire.itemCaps

/-- A round's free elements, in absorb order. -/
def freeOf (r : HonestRound) : List Nat := r.items.flatMap FirewallWire.itemFree

/-- Round `i`'s rows: the nodes of its caps after the schedule's copies of level 0's top, then its message row. -/
def roundRows (s : Schedule) (h : Honest) (i : Nat) : List (Array Nat) :=
  ((capsOf (h.round i)).drop (copiesOf s i)).flatMap topRows ++ messageRows (freeOf (h.round i)).toArray

/-- **The worker's rows**, in the order the ZK server salts them: each table's level-0 top in Commit's order, then each
round's (`roundRows`). -/
def honestRows (s : Schedule) (h : Honest) : List (Array Nat) :=
  h.roots.flatMap (fun e => topRows e.2) ++ (List.range s.rounds.size).flatMap (roundRows s h)

/-- The bytes `rowDigest` hashes of a row. -/
def rowBytes (row : Array Nat) : ByteArray :=
  row.foldl (fun b w => (b.push (w % 256).toUInt8).push (w / 256 % 256).toUInt8)
    (Hm96.pad ("verity/sha512-row/v1".toUTF8.push 0 ++ ByteArray.mk #[1, 16] ++ be32 row.size.toUInt32) 128)

theorem rowDigest_eq (row : Array Nat) : rowDigest row = Sha512.hash (rowBytes row) := rfl

/-! ## The worker that follows the schedule -/

/-- Round `i` as the ZK server takes it at the schedule's entry `(stream, k, n, tops, level0, commits)`: `split` reads its
content (`RoundFits`, a binding where it is a rep's first), the binding's cap and the first `level0` caps are the
committed top of the stream's table, the rest are fresh tops of the schedule's node counts, its message has the
schedule's rows, and its stream is opened by then (at a place up to `i + 1`, the one just before it). -/
def RoundOK (s : Schedule) (h : Honest) (i : Nat) : Prop :=
  match roundSpec (s.rounds.getD i .null) with
  | some (stream, k, n, tops, level0, commits) =>
    FirewallWire.RoundFits (k == 0 && parentOf s stream == "") (h.round i).items (h.round i).kind n ∧
      ((k == 0 && parentOf s stream == "") = true → (capsOf (h.round i)).head? = rootOf h stream) ∧
      level0 ≤ (capsOf (h.round i)).length ∧ (∀ c ∈ (capsOf (h.round i)).take level0, some c = rootOf h stream) ∧
      ((capsOf (h.round i)).drop level0).map (·.size / NODE_BYTES) = tops ∧
      (messageRows (freeOf (h.round i)).toArray).length = commits ∧
      stream ∈ (List.range (i + 2)).flatMap h.opens
  | none => True

/-- **The worker follows the schedule**: each length in its word (`ReqFits`), Σ a plain string, Commit's tables the
schedule's roots with their node counts in the schedule's order (`Schedule.tables`), each stream opened once at a place
up to the last round's, and each round as the ZK server takes it (`RoundOK`). -/
structure Fits (s : Schedule) (h : Honest) : Prop where
  pings : ∀ b ∈ h.pings, b.size < 2 ^ 32
  sigma : plainStr h.sigma = true
  hello : FirewallWire.ReqFits (.hello (printJson (helloWith s.hello h.sigma)))
  commit : FirewallWire.ReqFits (.commit h.rootF h.roots [])
  tables : h.roots.map (·.1) = s.tables
  roots : (h.roots.map fun e => (e.1, e.2.size / NODE_BYTES)).Perm (rootsOf s)
  caps : ∀ e ∈ h.roots, isCap e.2 = true
  link : 16 * s.y < 2 ^ 32
  opensFit : ∀ j, ∀ r ∈ honestOpens s h j, FirewallWire.ReqFits r
  opened : ((List.range (s.rounds.size + 1)).flatMap h.opens).Perm (s.streams.map (·.1))
  rounds : ∀ i < s.rounds.size, RoundOK s h i
  roundsFit : ∀ i, ∀ r ∈ honestRound s h i, FirewallWire.ReqFits r
  proofs : ∀ p ∈ h.proofs, p.1.toUTF8.size < 2 ^ 32 ∧ p.2.1 < 2 ^ 32 ∧ p.2.2.size < 2 ^ 32

/-! ## What a schedule the ZK server holds a session to holds -/

/-- A stream's table has a root in the schedule. -/
def Rooted (s : Schedule) (stream : String) : Prop := (s.roots.getObjVal? (tableOf stream)).toOption.isSome = true

/-- A schedule entry's `k` where it is a round of the stream `name`. -/
def ksOf (name : String) (r : Json) : Option Nat :=
  (roundSpec r).bind fun sp => if sp.1 == name then some sp.2.1 else none

/-- The stream `name`'s rounds among the schedule's first `i`. -/
def cnt (s : Schedule) (i : Nat) (name : String) : Nat := ((s.rounds.toList.take i).filterMap (ksOf name)).length

/-- `Schedule.canonical`, fact by fact. -/
structure Facts (s : Schedule) : Prop where
  hello : ∃ (H L : FirewallObj.Tree) (A B : List ((_ : String) × Json)), s.hello = .obj ⟨⟨H⟩⟩ ∧ H.Ordered ∧
    plainObj H = true ∧ L.Ordered ∧ plainObj L = true ∧ H.toListModel = A ++ ⟨"link", .obj ⟨⟨L⟩⟩⟩ :: B ∧
    (∀ e ∈ L.toListModel, e.1 ≠ "sigma") ∧ (∀ e ∈ H.toListModel, e.1 ≠ "coins")
  roots : ∃ t : FirewallObj.Tree, s.roots = .obj ⟨⟨t⟩⟩ ∧ t.Ordered ∧ ∀ e ∈ t.toListModel, (natOf? e.2).isSome = true
  streams : (s.streams.map (·.1)).Pairwise fun a b => compare a b = .lt
  rounds : ∀ i < s.rounds.size, ∃ sp, roundSpec (s.rounds.getD i .null) = some sp ∧
    canonForm (s.rounds.getD i .null) = canonForm (specJson sp) ∧ (sp.2.2.2.2.1 = 0 ∨ Rooted s sp.1) ∧
    sp.1 ∈ s.streams.map (·.1)
  each : ∀ e ∈ s.streams, (e.2.1 ≠ "" ∨ Rooted s e.1) ∧ s.rounds.toList.filterMap (ksOf e.1) = List.range e.2.2.2

theorem not_mem_of_getObjVal? {t : FirewallObj.Tree} (ht : t.Ordered) {k : String}
    (h : ((Json.obj ⟨⟨t⟩⟩).getObjVal? k).toOption.isNone = true) : ∀ e ∈ t.toListModel, e.1 ≠ k := by
  rintro ⟨k', v⟩ he rfl
  rw [FirewallObj.getObjVal?_of_mem ht he] at h
  cases h

theorem facts_of {s : Schedule} (hs : Canonical s) : Facts s := by
  have h := hs
  unfold Canonical Schedule.canonical at h
  simp only [Bool.and_eq_true] at h
  obtain ⟨⟨⟨⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, h5⟩, h6⟩, h7⟩ := h
  refine ⟨?_, ?_, of_decide_eq_true h5, ?_, ?_⟩
  · rcases he : s.hello with _ | _ | _ | _ | _ | ⟨⟨⟨H⟩⟩⟩ <;> rw [he] at h3 <;> try (simp [Json.getObjVal?] at h3; done)
    rw [he] at h1 h2
    have hp : plainObj H = true ∧ ascending H = true := by simpa [plain] using h1
    have hH := FirewallObj.ordered_of_ascending hp.2
    rcases hl : (Json.obj ⟨⟨H⟩⟩).getObjVal? "link" with _ | l <;> rw [hl] at h3
    · simp at h3
    rcases l with _ | _ | _ | _ | _ | ⟨⟨⟨L⟩⟩⟩ <;> try (simp at h3; done)
    have hm := FirewallObj.mem_of_getObjVal? hH hl
    obtain ⟨A, B, hAB⟩ := List.append_of_mem hm
    have hpl := ((FirewallJson.plainObj_iff H).mp hp.1 _ hm).2
    have hpL : plainObj L = true ∧ ascending L = true := by simpa [plain] using hpl
    have hL := FirewallObj.ordered_of_ascending hpL.2
    exact ⟨H, L, A, B, rfl, hH, hp.1, hL, hpL.1, hAB, not_mem_of_getObjVal? hL h3, not_mem_of_getObjVal? hH h2⟩
  · rcases he : s.roots with _ | _ | _ | _ | _ | ⟨⟨⟨t⟩⟩⟩ <;> rw [he] at h4 <;> try (simp at h4; done)
    simp only [Bool.and_eq_true, List.all_eq_true] at h4
    exact ⟨t, rfl, FirewallObj.ordered_of_ascending h4.1, h4.2⟩
  · intro i hi
    rw [← Array.all_toList, List.all_eq_true] at h6
    have hr := h6 _ (Array.getElem_mem_toList hi)
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hi, Option.getD_some]
    rcases hsp : roundSpec s.rounds[i] with _ | sp <;> rw [hsp] at hr
    · simp at hr
    simp only [Bool.and_eq_true, beq_iff_eq, Bool.or_eq_true, List.any_eq_true] at hr
    obtain ⟨⟨hc, hz⟩, e, he, hn⟩ := hr
    refine ⟨sp, rfl, hc, hz.imp (by simpa using ·) id, List.mem_map.mpr ⟨e, he, by simpa using hn⟩⟩
  · intro e he
    rw [List.all_eq_true] at h7
    have hr := h7 e he
    obtain ⟨name, parent, pos, count⟩ := e
    simp only [Bool.and_eq_true] at hr
    refine ⟨?_, beq_iff_eq.mp hr.2⟩
    rcases Bool.or_eq_true _ _ |>.mp hr.1.2 with h | h
    · exact .inl (bne_iff_ne.mp h)
    · exact .inr h

theorem natOf?_eq {v : Json} {c : Nat} (h : natOf? v = some c) : v = jnat c := by
  match v, h with
  | .num ⟨m, 0⟩, h =>
    simp only [natOf?] at h
    split at h
    · cases h
      simp only [jnat]
      congr
      exact (Int.toNat_of_nonneg ‹m ≥ 0›).symm
    · cases h

theorem rootsOf_obj {s : Schedule} {t : FirewallObj.Tree} (he : s.roots = .obj ⟨⟨t⟩⟩) :
    rootsOf s = t.toListModel.map fun e => (e.1, (natOf? e.2).getD 0) := by
  unfold rootsOf
  rw [he]

theorem streams_nodup {s : Schedule} (hF : Facts s) : (s.streams.map (·.1)).Nodup :=
  FirewallObj.Asc.ne (key := id) hF.streams

section
variable {s : Schedule} {h : Honest}

/-- Commit's tables are distinct. -/
theorem roots_nodup (hF : Facts s) (hf : Fits s h) : (h.roots.map (·.1)).Nodup := by
  obtain ⟨t, he, ht, _⟩ := hF.roots
  have hp := hf.roots.map Prod.fst
  rw [List.map_map, rootsOf_obj he, List.map_map] at hp
  exact hp.nodup_iff.mpr (List.pairwise_map.mpr (FirewallObj.Asc.ne (FirewallObj.asc_ordered ht)))

/-- Commit's node counts have the schedule's roots' canonical form. -/
theorem commit_pin (hF : Facts s) (hf : Fits s h) :
    canonForm (Json.mkObj (h.roots.map fun e => (e.1, jnat (e.2.size / NODE_BYTES)))) = canonForm s.roots := by
  obtain ⟨t, he, ht, hn⟩ := hF.roots
  rw [he, FirewallObj.mkObj_eq]
  refine FirewallObj.canonForm_obj_perm (FirewallObj.ordered_ofList _) ht ?_
  have hnd : List.Pairwise (· ≠ ·) (h.roots.map (·.1)) := roots_nodup hF hf
  refine (FirewallObj.forms_ofList (List.pairwise_map.mpr (List.pairwise_map.mp hnd))).trans ?_
  rw [List.map_map]
  have hp := hf.roots.map fun e : String × Nat => (e.1, canonForm (jnat e.2))
  rw [List.map_map, rootsOf_obj he, List.map_map] at hp
  refine hp.trans (List.Perm.of_eq ?_)
  unfold FirewallObj.forms
  refine List.map_congr_left fun e he' => ?_
  obtain ⟨c, hc⟩ := Option.isSome_iff_exists.mp (hn e he')
  simp only [Function.comp, hc, Option.getD_some]
  rw [← natOf?_eq hc]

/-- Each table's node count is the schedule's. -/
theorem commit_nat (hF : Facts s) (hf : Fits s h) :
    ∀ e ∈ h.roots, natOf? (s.roots.getObjValD e.1) = some (e.2.size / NODE_BYTES) := by
  intro e he
  obtain ⟨t, hes, ht, hn⟩ := hF.roots
  have hm : (e.1, e.2.size / NODE_BYTES) ∈ rootsOf s := hf.roots.mem_iff.mp (List.mem_map.mpr ⟨e, he, rfl⟩)
  rw [rootsOf_obj hes] at hm
  obtain ⟨x, hx, hxe⟩ := List.mem_map.mp hm
  obtain ⟨c, hc⟩ := Option.isSome_iff_exists.mp (hn x hx)
  simp only [hc, Option.getD_some, Prod.mk.injEq] at hxe
  rw [hes, Json.getObjValD, ← hxe.1, FirewallObj.getObjVal?_of_mem ht hx]
  show natOf? x.2 = _
  rw [hc, hxe.2]

/-- A stream whose table has a root in the schedule has one in the worker's Commit. -/
theorem rootOf_isSome (hF : Facts s) (hf : Fits s h) {stream : String} (hr : Rooted s stream) :
    (rootOf h stream).isSome = true := by
  obtain ⟨t, hes, ht, _⟩ := hF.roots
  unfold Rooted at hr
  rw [hes] at hr
  obtain ⟨v, hv⟩ := Option.isSome_iff_exists.mp hr
  have hv' : (Json.obj ⟨⟨t⟩⟩).getObjVal? (tableOf stream) = .ok v := by
    revert hv
    cases (Json.obj ⟨⟨t⟩⟩).getObjVal? (tableOf stream) <;> simp [Except.toOption]
  have hm := FirewallObj.mem_of_getObjVal? ht hv'
  have hm' : (tableOf stream, (natOf? v).getD 0) ∈ h.roots.map fun e => (e.1, e.2.size / NODE_BYTES) :=
    hf.roots.mem_iff.mpr (by rw [rootsOf_obj hes]; exact List.mem_map.mpr ⟨_, hm, rfl⟩)
  obtain ⟨e, he, hee⟩ := List.mem_map.mp hm'
  unfold rootOf
  rw [Option.isSome_map]
  exact List.find?_isSome.mpr ⟨e, he, by simp [(Prod.mk.inj hee).1]⟩

/-- The streams the worker opens at its places before `p`. -/
def openedTo (h : Honest) (p : Nat) : List String := (List.range p).flatMap h.opens

theorem openedTo_succ (h : Honest) (p : Nat) : openedTo h (p + 1) = openedTo h p ++ h.opens p := by
  simp only [openedTo, List.range_succ, List.flatMap_append, List.flatMap_cons, List.flatMap_nil, List.append_nil]

theorem openedTo_sublist (h : Honest) {p q : Nat} (hpq : p ≤ q) : (openedTo h p).Sublist (openedTo h q) := by
  induction q with
  | zero => rw [Nat.le_zero.mp hpq]; exact List.Sublist.refl _
  | succ q ih =>
    rcases Nat.lt_or_ge p (q + 1) with hp | hp
    · rw [openedTo_succ]
      exact (ih (by omega)).trans (List.sublist_append_left _ _)
    · rw [show p = q + 1 by omega]; exact List.Sublist.refl _

theorem opened_nodup (hF : Facts s) (hf : Fits s h) : (openedTo h (s.rounds.size + 1)).Nodup :=
  hf.opened.nodup_iff.mpr (streams_nodup hF)

theorem cnt_zero (s : Schedule) (n : String) : cnt s 0 n = 0 := by
  simp [cnt]

theorem getD_eq {s : Schedule} {i : Nat} (hi : i < s.rounds.size) : s.rounds.getD i .null = s.rounds[i] := by
  rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hi]
  rfl

theorem cnt_succ (s : Schedule) {i : Nat} (hi : i < s.rounds.size) {sp : String × Nat × Nat × List Nat × Nat × Nat}
    (hsp : roundSpec (s.rounds.getD i .null) = some sp) (n : String) :
    cnt s (i + 1) n = cnt s i n + if sp.1 == n then 1 else 0 := by
  unfold cnt
  rw [List.take_add_one, List.filterMap_append, List.length_append, Array.getElem?_toList,
    Array.getElem?_eq_getElem hi]
  rw [getD_eq hi] at hsp
  congr 1
  simp only [Option.toList_some, List.filterMap_cons, List.filterMap_nil, ksOf, hsp, Option.bind_some]
  by_cases hn : sp.1 = n <;> simp [hn]

theorem eq_length_of_range {l₁ l₂ : List Nat} {k c : Nat} (h : l₁ ++ k :: l₂ = List.range c) : k = l₁.length := by
  have h1 : (l₁ ++ k :: l₂)[l₁.length]? = some k := by simp
  have hlt : l₁.length < c := by
    have := congrArg List.length h
    simp only [List.length_append, List.length_cons, List.length_range] at this
    omega
  rw [h, List.getElem?_range hlt] at h1
  exact (Option.some.inj h1).symm

/-- A round's `k` is its stream's rounds before it. -/
theorem k_eq (hF : Facts s) {i : Nat} (hi : i < s.rounds.size) {stream : String} {k n : Nat} {tops : List Nat}
    {level0 commits : Nat} (hsp : roundSpec (s.rounds.getD i .null) = some (stream, k, n, tops, level0, commits)) :
    k = cnt s i stream := by
  obtain ⟨sp, hsp', -, -, hmem⟩ := hF.rounds i hi
  rw [hsp] at hsp'
  cases hsp'
  obtain ⟨e, he, hee⟩ := List.mem_map.mp hmem
  have hl := (hF.each e he).2
  rw [hee] at hl
  have hs : s.rounds.toList = s.rounds.toList.take i ++ s.rounds[i] :: s.rounds.toList.drop (i + 1) := by
    rw [← Array.getElem_toList (by simpa using hi), ← List.drop_eq_getElem_cons (by simpa using hi),
      List.take_append_drop]
  rw [hs, List.filterMap_append, List.filterMap_cons] at hl
  rw [getD_eq hi] at hsp
  have hk : ksOf stream s.rounds[i] = some k := by simp [ksOf, hsp]
  rw [hk] at hl
  exact eq_length_of_range hl

theorem cnt_size (hF : Facts s) {e : String × String × Nat × Nat} (he : e ∈ s.streams) :
    cnt s s.rounds.size e.1 = e.2.2.2 := by
  unfold cnt
  rw [← Array.length_toList, List.take_length, (hF.each e he).2, List.length_range]

theorem round_of_cnt {j : Nat} {n : String} (h : 0 < cnt s j n) :
    ∃ i < j, i < s.rounds.size ∧ ∃ sp : String × Nat × Nat × List Nat × Nat × Nat,
      roundSpec (s.rounds.getD i .null) = some sp ∧ sp.1 = n := by
  unfold cnt at h
  obtain ⟨k, hk⟩ := List.exists_mem_of_length_pos h
  obtain ⟨r, hr, hrk⟩ := List.mem_filterMap.mp hk
  obtain ⟨i, hi, hri⟩ := List.mem_iff_getElem.mp hr
  simp only [List.length_take, Array.length_toList] at hi
  refine ⟨i, by omega, by omega, ?_⟩
  rw [List.getElem_take, Array.getElem_toList] at hri
  rw [getD_eq (by omega), hri]
  unfold ksOf at hrk
  rcases hsp : roundSpec r with _ | sp <;> rw [hsp] at hrk
  · cases hrk
  simp only [Option.bind_some] at hrk
  by_cases hn : sp.1 = n
  · exact ⟨sp, rfl, hn⟩
  · simp [hn] at hrk

/-- A stream the worker opens just before round `j` has no round before it. -/
theorem cnt_new (hF : Facts s) (hf : Fits s h) {j : Nat} (hj : j < s.rounds.size) {n : String}
    (hn : n ∈ h.opens (j + 1)) : cnt s j n = 0 := by
  refine Nat.eq_zero_of_not_pos fun hpos => ?_
  obtain ⟨i, hij, hi, ⟨stream, k, m, tops, level0, commits⟩, hsp, hsn⟩ := round_of_cnt hpos
  have hR := hf.rounds i hi
  unfold RoundOK at hR
  rw [hsp] at hR
  have hmem : stream ∈ openedTo h (i + 2) := hR.2.2.2.2.2.2
  have hnd := (opened_nodup hF hf).sublist (openedTo_sublist h (show j + 2 ≤ s.rounds.size + 1 by omega))
  rw [openedTo_succ] at hnd
  exact List.nodup_append.mp hnd |>.2.2 stream ((openedTo_sublist h (by omega)).subset hmem) n hn hsn

/-! ## The worker's rows -/

theorem length_topRows (top : ByteArray) : (topRows top).length = top.size / NODE_BYTES := by
  simp [topRows]

theorem length_roundRows (hF : Facts s) (hf : Fits s h) {i : Nat} (hi : i < s.rounds.size) :
    (roundRows s h i).length = roundCount s i := by
  obtain ⟨⟨stream, k, n, tops, level0, commits⟩, hsp, -⟩ := hF.rounds i hi
  have hR := hf.rounds i hi
  unfold RoundOK at hR
  rw [hsp] at hR
  obtain ⟨-, -, -, -, htops, hcom, -⟩ := hR
  unfold roundRows roundCount copiesOf
  rw [hsp, List.length_append, List.length_flatMap, hcom]
  simp only [Option.map_some, Option.getD_some, length_topRows]
  rw [← htops]

/-- The worker's rows before round `j`: its tables' level-0 tops, then rounds `0` to `j - 1`'s. -/
def pre (s : Schedule) (h : Honest) (j : Nat) : List (Array Nat) :=
  h.roots.flatMap (fun e => topRows e.2) ++ (List.range j).flatMap (roundRows s h)

theorem pre_succ (s : Schedule) (h : Honest) (j : Nat) : pre s h (j + 1) = pre s h j ++ roundRows s h j := by
  simp only [pre, List.range_succ, List.flatMap_append, List.flatMap_cons, List.flatMap_nil, List.append_nil,
    List.append_assoc]

theorem length_pre_le (s : Schedule) (h : Honest) (j : Nat) : ∀ d, (pre s h j).length ≤ (pre s h (j + d)).length
  | 0 => Nat.le_refl _
  | d + 1 => by
    rw [← Nat.add_assoc, pre_succ, List.length_append]
    exact Nat.le_trans (length_pre_le s h j d) (Nat.le_add_right _ _)

/-- The worker's rows are the schedule's count. -/
theorem length_honestRows {s : Schedule} (hs : Canonical s) {h : Honest} (hf : Fits s h) :
    (honestRows s h).length = rowCount s := by
  have hF := facts_of hs
  unfold honestRows rowCount
  rw [List.length_append, List.length_flatMap, List.length_flatMap]
  congr 1
  · have hp := hf.roots.map Prod.snd
    rw [List.map_map] at hp
    rw [← hp.sum_nat]
    simp only [length_topRows]
    rfl
  · congr 1
    exact List.map_congr_left fun i hi => length_roundRows hF hf (List.mem_range.mp hi)

end

/-! ## What the ZK server does on the worker's requests -/

/-- An opened stream as the ZK server holds it before round `j`: its parent and its rounds so far. -/
def F (s : Schedule) (j : Nat) (n : String) : String × String × Nat := (n, parentOf s n, cnt s j n)

/-- **The loop's invariant** before round `j`, the worker's places before `o` opened: after Hello, Commit and Link,
Commit's tables the worker's, each salt of the worker's rows before round `j` sampled and its commit string sent. -/
structure At (s : Schedule) (h : Honest) (ys : List ByteArray) (j o : Nat) (st : State) : Prop where
  hello : st.hello = true
  committed : st.committed = true
  linked : st.linked = true
  finished : st.finished = false
  roots : st.roots = h.roots
  rounds : st.rounds = j
  streams : st.streams = (openedTo h o).map (F s j)
  salts : st.salts = ys.drop (pre s h j).length
  names : ∃ a tops rest, st.sent.toList = a :: .commit tops :: rest ∧ tops.map (·.1) = h.roots.map (·.1)
  holes : FirewallRun.holesOf st.sent = FirewallSteps.cs (pre s h j) ys

open FirewallSteps

section
variable {s : Schedule} {h : Honest}

theorem find_F {j : Nat} {O : List String} {n : String} (hm : n ∈ O) :
    (O.map (F s j)).find? (·.1 == n) = some (F s j n) := by
  rw [List.find?_map]
  have hx : (O.find? ((·.1 == n) ∘ F s j)).isSome := List.find?_isSome.mpr ⟨n, hm, by simp [F]⟩
  obtain ⟨x, hx⟩ := Option.isSome_iff_exists.mp hx
  have hxn : x = n := by simpa [F] using List.find?_some hx
  rw [hx, hxn]
  rfl

theorem goes_opens_aux {g : String → Option Req} {j : Nat} :
    ∀ (L O : List String) {st : State}, (O ++ L).Nodup →
      (∀ n ∈ L, ∃ w parent pos c lab, s.streams.find? (·.1 == n) = some (w, parent, pos, c) ∧
        g n = some (.openStream n parent lab pos)) →
      (∀ n ∈ L, cnt s j n = 0) → (∀ r ∈ L.filterMap g, FirewallWire.ReqFits r) →
      st.finished = false → st.streams = O.map (F s j) →
      Goes s ((L.filterMap g).map putReq) st { st with streams := (O ++ L).map (F s j) }
  | [], O, st, _, _, _, _, _, hs => by
    rw [List.append_nil, ← hs]
    rfl
  | n :: L, O, st, hnd, hg, hz, hfit, hfin, hs => by
    obtain ⟨w, parent, pos, c, lab, hfind, hgn⟩ := hg n (List.mem_cons_self ..)
    rw [List.filterMap_cons_some hgn] at hfit ⊢
    rw [List.map_cons]
    have hpar : parentOf s n = parent := by
      unfold parentOf
      rw [hfind]
      rfl
    have hnew : st.streams.any (·.1 == n) = false := by
      rw [hs, List.any_eq_false]
      intro e he hen
      obtain ⟨x, hx, rfl⟩ := List.mem_map.mp he
      have hxn : x = n := by simpa [F] using hen
      subst hxn
      exact (List.nodup_append.mp hnd).2.2 x hx x (List.mem_cons_self ..) rfl
    have hone := goes_one (FirewallWire.decode_putReq (hfit _ (List.mem_cons_self ..))) (run_open s hfin hfind hnew)
    have hrest := goes_opens_aux L (O ++ [n]) (st := { st with streams := st.streams ++ [(n, parent, 0)] })
      (by rw [List.append_assoc, List.singleton_append]; exact hnd)
      (fun n' hn' => hg n' (List.mem_cons_of_mem _ hn')) (fun n' hn' => hz n' (List.mem_cons_of_mem _ hn'))
      (fun r hr => hfit r (List.mem_cons_of_mem _ hr)) hfin
      (by rw [hs, List.map_append]; simp [F, hpar, hz n (List.mem_cons_self ..)])
    rw [List.append_assoc, List.singleton_append] at hrest
    exact goes_append s [_] _ hone hrest

/-- **The worker's `Open`s at its place `p`**, each stream new there. -/
theorem goes_opens (hF : Facts s) (hf : Fits s h) {j p : Nat} (hp : p ≤ s.rounds.size)
    (hz : ∀ n ∈ h.opens p, cnt s j n = 0) {st : State} (hfin : st.finished = false)
    (hs : st.streams = (openedTo h p).map (F s j)) :
    Goes s ((honestOpens s h p).map putReq) st { st with streams := (openedTo h (p + 1)).map (F s j) } := by
  rw [openedTo_succ]
  refine goes_opens_aux (h.opens p) (openedTo h p) ?_ (fun n hn => ?_) hz (hf.opensFit p) hfin hs
  · rw [← openedTo_succ]
    exact (opened_nodup hF hf).sublist (openedTo_sublist h (by omega))
  · have hm : n ∈ s.streams.map (·.1) := hf.opened.mem_iff.mp
      (List.mem_flatMap.mpr ⟨p, List.mem_range.mpr (by omega), hn⟩)
    obtain ⟨e, he, rfl⟩ := List.mem_map.mp hm
    have hx : (s.streams.find? (·.1 == e.1)).isSome := List.find?_isSome.mpr ⟨e, he, by simp⟩
    obtain ⟨x, hx⟩ := Option.isSome_iff_exists.mp hx
    obtain ⟨w, parent, pos, c⟩ := x
    exact ⟨w, parent, pos, c, h.label e.1, hx, by simp [hx]⟩

theorem pre_le (hs : Canonical s) (hf : Fits s h) {ys : List ByteArray} (hy : ys.length = rowCount s) {j : Nat}
    (hj : j ≤ s.rounds.size) : (pre s h j).length ≤ ys.length := by
  have := length_pre_le s h j (s.rounds.size - j)
  rw [show j + (s.rounds.size - j) = s.rounds.size by omega] at this
  rw [hy, ← length_honestRows hs hf]
  exact this

theorem caps_fit {r : HonestRound} (hi : ∀ a ∈ r.items, FirewallWire.ItemFits a) : ∀ c ∈ capsOf r, isCap c = true := by
  intro c hc
  obtain ⟨a, ha, hca⟩ := List.mem_flatMap.mp hc
  have := hi a ha
  cases a <;> simp only [FirewallWire.itemCaps, List.mem_singleton, List.not_mem_nil] at hca
  all_goals first
    | (subst hca; exact this.2.2)
    | (subst hca; exact this.2)

/-- **Round `j`** from the loop's invariant, its stream opened. -/
theorem goes_round (hs : Canonical s) (hf : Fits s h) {ys : List ByteArray} (hy : ys.length = rowCount s)
    {j : Nat} (hj : j < s.rounds.size) {st : State} (hA : At s h ys j (j + 2) st) :
    ∃ t, Goes s ((honestRound s h j).map putReq) st t ∧ At s h ys (j + 1) (j + 2) t := by
  have hF := facts_of hs
  obtain ⟨⟨stream, k, n, tops, level0, commits⟩, hsp, hpin, -, hmem⟩ := hF.rounds j hj
  have hR := hf.rounds j hj
  unfold RoundOK at hR
  rw [hsp] at hR
  obtain ⟨hfits, hfirst, hlev, hcopy, htops, hcom, hopened⟩ := hR
  obtain rfl := k_eq hF hj hsp
  have hreq : honestRound s h j = [.round stream (putRound (h.round j).items (h.round j).kind n) n] := by
    unfold honestRound
    rw [hsp]
  have hcop : copiesOf s j = level0 := by
    unfold copiesOf
    rw [hsp]
    rfl
  obtain ⟨e, he, hes⟩ := List.mem_map.mp hmem
  have hx : (s.streams.find? (·.1 == stream)).isSome := List.find?_isSome.mpr ⟨e, he, by simp [hes]⟩
  obtain ⟨e', he'⟩ := Option.isSome_iff_exists.mp hx
  have he'1 : e'.1 = stream := by simpa using List.find?_some he'
  have hpar : parentOf s stream = e'.2.1 := by
    unfold parentOf
    rw [he']
    rfl
  have hfind : st.streams.find? (·.1 == stream) = some (stream, parentOf s stream, cnt s j stream) := by
    rw [hA.streams]
    exact find_F hopened
  have hround := run_round s stream (putRound (h.round j).items (h.round j).kind n) n
    (sp := { caps := (capsOf (h.round j)).toArray, free := (freeOf (h.round j)).toArray, n := n })
    hA.finished hfind (FirewallWire.split_putRound hfits) rfl
    (fun hb => by
      have hb' := hb
      simp only [Bool.and_eq_true, beq_iff_eq] at hb'
      have hroot : Rooted s stream := by
        rcases (hF.each e' (List.mem_of_find?_eq_some he')).1 with hne | hr
        · exact absurd (hpar ▸ hb'.2) hne
        · rwa [he'1] at hr
      obtain ⟨c, hc⟩ := Option.isSome_iff_exists.mp (rootOf_isSome hF hf hroot)
      refine ⟨c, ?_, ?_⟩
      · rw [List.getElem?_toArray, ← List.head?_eq_getElem?, hfirst hb, hc]
      · rw [hA.roots]
        exact hc)
    (by rw [hA.rounds, hcop, List.size_toArray]; exact hlev)
    (by rw [hA.rounds, hcop, hA.roots, List.toList_toArray]; exact hcopy)
    (by rw [List.toList_toArray]; exact caps_fit hfits.1)
    (by
      rw [hA.rounds, hcop, hsp, List.toList_toArray, htops, hcom])
    (by
      rw [hA.rounds, hcop, List.toList_toArray, htops, hcom]
      exact hpin)
    hA.linked (fun _ => hx)
    (by
      rw [hA.salts, List.length_drop, hA.rounds, List.toList_toArray]
      have h1 := pre_le hs hf hy (j := j + 1) (by omega)
      rw [pre_succ, List.length_append] at h1
      unfold roundRows at h1
      rw [List.length_append] at h1
      dsimp only
      omega)
  obtain ⟨⟨⟩, t, hrun, x, tops', bcs, rfl, ⟨e0, hx0, he0⟩, htops', hbcs⟩ := hround
  have hdec := FirewallWire.decode_putReq (hf.roundsFit j _ (by rw [hreq]; exact List.mem_singleton_self _))
  rw [hreq, List.map_singleton]
  refine ⟨_, goes_one hdec hrun, ?_⟩
  obtain ⟨a, tops0, rest, hsent, htn⟩ := hA.names
  refine ⟨hA.hello, hA.committed, hA.linked, hA.finished, hA.roots, by simp [hA.rounds], ?_, ?_, ?_, ?_⟩
  · show st.streams.map _ = _
    rw [hA.streams, List.map_map]
    refine List.map_congr_left fun m _ => ?_
    simp only [Function.comp, F]
    rw [cnt_succ s hj hsp m]
    by_cases hm : m = stream
    · subst hm
      simp
    · simp [hm, Ne.symm hm]
  · show (st.salts.drop _).drop _ = _
    rw [hA.salts, List.drop_drop, List.drop_drop, pre_succ, List.length_append, hA.rounds, List.toList_toArray]
    unfold roundRows
    rw [List.length_append]
  · refine ⟨a, tops0, rest ++ e0 ++ [.round stream tops' (copiesOf s st.rounds) bcs n], ?_, htn⟩
    rw [Array.toList_push, hx0, hsent]
    simp
  · rw [FirewallRun.holesOf_push]
    unfold FirewallRun.holesOf
    rw [hx0, List.flatMap_append, he0, List.append_nil]
    show FirewallRun.holesOf st.sent ++ (tops'.flatten ++ bcs) = _
    rw [hA.holes, htops', hbcs, hA.salts, pre_succ, cs_append, hA.rounds, List.toList_toArray]
    unfold roundRows
    rw [cs_append, List.drop_drop]

/-- **The rounds**, each after the worker's `Open`s just before it. -/
theorem goes_rounds (hs : Canonical s) (hf : Fits s h) {ys : List ByteArray} (hy : ys.length = rowCount s)
    {st : State} (hA : At s h ys 0 1 st) : ∀ j ≤ s.rounds.size, ∃ t,
      Goes s ((List.range j).flatMap fun i => (honestOpens s h (i + 1) ++ honestRound s h i).map putReq) st t ∧
        At s h ys j (j + 1) t
  | 0, _ => ⟨st, rfl, hA⟩
  | j + 1, hj => by
    obtain ⟨t, hg, hAt⟩ := goes_rounds hs hf hy hA j (by omega)
    have ho := goes_opens (facts_of hs) hf (p := j + 1) (by omega)
      (fun n hn => cnt_new (facts_of hs) hf (by omega) hn) hAt.finished hAt.streams
    obtain ⟨u, hu, hAu⟩ := goes_round hs hf hy (j := j)
      (st := { t with streams := (openedTo h (j + 2)).map (F s j) }) (by omega)
      ⟨hAt.hello, hAt.committed, hAt.linked, hAt.finished, hAt.roots, hAt.rounds, rfl, hAt.salts, hAt.names,
        hAt.holes⟩
    refine ⟨u, ?_, hAu⟩
    rw [List.range_succ, List.flatMap_append, List.flatMap_singleton, List.map_append]
    exact goes_append s _ _ hg (goes_append s _ _ ho hu)

theorem goes_pings (s : Schedule) : ∀ (l : List ByteArray), (∀ b ∈ l, b.size < 2 ^ 32) → ∀ st,
    Goes s (l.map putPing) st st
  | [], _, _ => rfl
  | b :: l, hl, st => goes_append s [_] _ (goes_one (decode_putPing (hl b (List.mem_cons_self ..))) (run_ping s st))
      (goes_pings s l (fun b hb => hl b (List.mem_cons_of_mem _ hb)) st)

theorem goes_proofs (s : Schedule) : ∀ (l : List (String × Nat × ByteArray)),
    (∀ p ∈ l, p.1.toUTF8.size < 2 ^ 32 ∧ p.2.1 < 2 ^ 32 ∧ p.2.2.size < 2 ^ 32) → ∀ st, st.finished = false →
      Goes s (l.map fun p => putProof p.1 p.2.1 p.2.2) st st
  | [], _, _, _ => rfl
  | p :: l, hl, st, hfin => by
    obtain ⟨h1, h2, h3⟩ := hl p (List.mem_cons_self ..)
    exact goes_append s [_] _ (goes_one (decode_putProof h1 h2 h3) (run_proof s hfin))
      (goes_proofs s l (fun p hp => hl p (List.mem_cons_of_mem _ hp)) st hfin)

/-- Finish's pin: each stream opened, with the schedule's count of rounds. -/
theorem finish_pin (hF : Facts s) (hf : Fits s h) :
    canonForm (Json.mkObj (((openedTo h (s.rounds.size + 1)).map (F s s.rounds.size)).map fun e => (e.1, jnat e.2.2))) =
      canonForm (Json.mkObj (s.streams.map fun e => (e.1, jnat e.2.2.2))) := by
  have hnd : List.Pairwise (· ≠ ·) (s.streams.map (·.1)) := streams_nodup hF
  have hno : List.Pairwise (· ≠ ·) (openedTo h (s.rounds.size + 1)) := opened_nodup hF hf
  rw [List.map_map]
  refine FirewallObj.canonForm_mkObj_perm (List.pairwise_map.mpr hno)
    (List.pairwise_map.mpr (List.pairwise_map.mp hnd)) ?_
  have e1 : (s.streams.map fun e => (e.1, jnat e.2.2.2)).map (fun e => (e.1, canonForm e.2)) =
      (s.streams.map (·.1)).map fun n => (n, canonForm (jnat (cnt s s.rounds.size n))) := by
    rw [List.map_map, List.map_map]
    refine List.map_congr_left fun e he => ?_
    simp only [Function.comp, cnt_size hF he]
  rw [e1, List.map_map]
  exact hf.opened.map _

end

/-- **The ZK server finishes on the honest worker's requests**: at salts `ys`, as many as the schedule's rows, it
finishes, its Commit names the schedule's tables, and its commit strings are those of the worker's rows under `ys`, in
salt order. -/
theorem run_honestReqs {s : Schedule} (hs : Canonical s) {h : Honest} (hf : Fits s h)
    (ys : List ByteArray) (hy : ys.length = rowCount s) :
    (run s (honestReqs s h) ys).2 = true ∧
      FirewallTemplate.namesOf (run s (honestReqs s h) ys).1.toList = s.tables ∧
      FirewallRun.holesOf (run s (honestReqs s h) ys).1 =
        List.zipWith (fun row y => Hm96.Default512.commitString (rowDigest row) y) (honestRows s h) ys := by
  have hF := facts_of hs
  obtain ⟨H, L, A, B, he, hH, hpH, hL, hpL, hsplit, hsg, hco⟩ := hF.hello
  have hpre0 : pre s h 0 = h.roots.flatMap (fun e => topRows e.2) := List.append_nil _
  have hl0 : (h.roots.flatMap fun e => topRows e.2).length ≤ ys.length := by
    rw [← hpre0]
    exact pre_le hs hf hy (Nat.zero_le _)
  have g1 := goes_pings s h.pings hf.pings { salts := ys }
  have g2 : Goes s [putReq .register] { salts := ys } { salts := ys } :=
    goes_one (FirewallWire.decode_putReq (r := .register) trivial) (run_register s rfl)
  have g3 := goes_one (FirewallWire.decode_putReq hf.hello)
    (run_hello s he hH hpH hL hpL hsplit hsg hco hf.sigma (st := { salts := ys }) rfl rfl)
  have g4 := goes_opens hF hf (j := 0) (p := 0) (Nat.zero_le _) (fun n _ => cnt_zero s n)
    (st := { salts := ys, hello := true, sent := (#[] : Array Sent).push (.hello s.hello) }) rfl rfl
  obtain ⟨⟨⟩, _, hc3, tops, rfl, htn, hth⟩ := run_commit s h.rootF h.roots
    (st := { salts := ys, hello := true, sent := (#[] : Array Sent).push (.hello s.hello),
             streams := (openedTo h 1).map (F s 0) })
    rfl rfl rfl (roots_nodup hF hf) hf.caps (commit_pin hF hf) (commit_nat hF hf) hf.tables hl0
  have g5 := goes_one (FirewallWire.decode_putReq hf.commit) hc3
  let st3 : State :=
    { salts := ys.drop (h.roots.flatMap fun e => topRows e.2).length
      hello := true
      committed := true
      roots := h.roots
      sent := ((#[] : Array Sent).push (.hello s.hello)).push (.commit tops)
      streams := (openedTo h 1).map (F s 0) }
  have hlink : FirewallWire.ReqFits (.link ⟨Array.replicate (16 * s.y) 0⟩) := by
    show (Array.replicate (16 * s.y) (0 : UInt8)).size < 2 ^ 32
    rw [Array.size_replicate]
    exact hf.link
  have g6 := goes_one (FirewallWire.decode_putReq hlink) (run_link s (st := st3) rfl rfl rfl)
  have hA0 : At s h ys 0 1 { st3 with linked := true, sent := st3.sent.push (.link s.y) } :=
    ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, by rw [hpre0], ⟨_, tops, _, rfl, htn⟩, by
      show FirewallRun.holesOf ((((#[] : Array Sent).push (.hello s.hello)).push (.commit tops)).push (.link s.y)) = _
      rw [hpre0]
      simp only [FirewallRun.holesOf, FirewallRun.holes, Array.toList_push, List.flatMap_append, List.flatMap_cons,
        List.flatMap_nil, List.append_nil, List.nil_append]
      exact hth⟩
  obtain ⟨st5, g7, hA5⟩ := goes_rounds hs hf hy hA0 s.rounds.size (Nat.le_refl _)
  have g8 := goes_proofs s h.proofs hf.proofs st5 hA5.finished
  have g9 := goes_one (FirewallWire.decode_putReq (r := .finish) trivial)
    (run_finish s hA5.finished hA5.committed hA5.linked hA5.rounds (by rw [hA5.streams]; exact finish_pin hF hf))
  have hg : Goes s (honestReqs s h) { salts := ys } _ :=
    goes_append s _ _ (goes_append s _ _ (goes_append s _ _ (goes_append s _ _ (goes_append s _ _
      (goes_append s _ _ g1 (goes_append s [_] [_] g2 g3)) g4) (goes_append s [_] [_] g5 g6)) g7) g8) g9
  rw [run_of_goes hg]
  obtain ⟨a, tops5, rest, hsent, htn5⟩ := hA5.names
  refine ⟨rfl, ?_, ?_⟩
  · show FirewallTemplate.namesOf (st5.sent.push _).toList = s.tables
    rw [Array.toList_push, hsent]
    show tops5.map (·.1) = s.tables
    rw [htn5, hf.tables]
  · show FirewallRun.holesOf (st5.sent.push _) = _
    rw [FirewallRun.holesOf_push, hA5.holes]
    simp only [FirewallRun.holes, List.append_nil]
    rfl

/-! ## The messages -/

/-- The worker's row `i` reads only the commit strings before it: at two lists of commit strings that agree below `i`,
its rows agree at `i`. -/
def Causal {Prog Wit Rr Ωa Ωp : Type} (s : Schedule)
    (D : Prog → Wit → Rr → Ωa → Ωp → (Fin (rowCount s) → ByteArray) → Honest) : Prop :=
  ∀ P w r a p (l l' : Fin (rowCount s) → ByteArray) (i : Nat), (∀ j : Fin (rowCount s), j.val < i → l j = l' j) →
    (honestRows s (D P w r a p l))[i]? = (honestRows s (D P w r a p l'))[i]?

/-- The commit strings before `i`, the rest empty. -/
def pad {n : Nat} (i : Nat) (pre : Fin i → ByteArray) : Fin n → ByteArray :=
  fun j => if hj : j.val < i then pre ⟨j.val, hj⟩ else .empty

/-- **The honest worker's messages**: message `i` the bytes of its row `i` at the commit strings before it. -/
def honestMsgs {Prog Wit Rr Ωa Ωp : Type} (s : Schedule)
    (D : Prog → Wit → Rr → Ωa → Ωp → (Fin (rowCount s) → ByteArray) → Honest)
    (P : Prog) (w : Wit) (r : Rr) (a : Ωa) (p : Ωp) (i : Fin (rowCount s)) (pre : Fin i → ByteArray) : ByteArray :=
  rowBytes ((honestRows s (D P w r a p (pad i pre))).getD i #[])

/-- At the commit strings `l`, message `i` read at `l`'s strings before it hashes to the digest of the worker's row `i`
at `l`. -/
theorem honestMsgs_digest {Prog Wit Rr Ωa Ωp : Type} {s : Schedule} (hs : Canonical s) 
    {D : Prog → Wit → Rr → Ωa → Ωp → (Fin (rowCount s) → ByteArray) → Honest}
    (hD : ∀ P w r a p l, Fits s (D P w r a p l)) (hc : Causal s D)
    (P : Prog) (w : Wit) (r : Rr) (a : Ωa) (p : Ωp) (l : Fin (rowCount s) → ByteArray) (i : Fin (rowCount s)) :
    Sha512.hash (honestMsgs s D P w r a p i fun j => l (Fin.castLE (Nat.le_of_lt i.isLt) j)) =
      rowDigest ((honestRows s (D P w r a p l))[i.val]'(by
        rw [length_honestRows hs (hD P w r a p l)]; exact i.isLt)) := by
  have e := hc P w r a p (pad i fun j => l (Fin.castLE (Nat.le_of_lt i.isLt) j)) l i.val fun j hj => by
    simp only [pad, hj, dite_true]
    rfl
  rw [rowDigest_eq, honestMsgs, List.getD_eq_getElem?_getD, e, List.getElem?_eq_getElem]
  rfl

/-- **`FirewallAccepts`'s body at one salt `y`**: at every commit strings `l`, the ZK server at `rowCount s` salts `y`
finishes on the worker's requests, Commit names the schedule's tables, and its commit strings are those of `honestMsgs`
under `y`, message `i` read at `l`'s strings before it. -/
theorem honest_accepts {Prog Wit Rr Ωa Ωp : Type} {s : Schedule} (hs : Canonical s) 
    {D : Prog → Wit → Rr → Ωa → Ωp → (Fin (rowCount s) → ByteArray) → Honest}
    (hD : ∀ P w r a p l, Fits s (D P w r a p l)) (hc : Causal s D) (y : ByteArray)
    (P : Prog) (w : Wit) (r : Rr) (a : Ωa) (p : Ωp) (l : Fin (rowCount s) → ByteArray) :
    (run s (honestReqs s (D P w r a p l)) (List.ofFn fun _ : Fin (rowCount s) => y)).2 = true ∧
      FirewallTemplate.namesOf
          (run s (honestReqs s (D P w r a p l)) (List.ofFn fun _ : Fin (rowCount s) => y)).1.toList = s.tables ∧
      FirewallRun.holesOf (run s (honestReqs s (D P w r a p l)) (List.ofFn fun _ : Fin (rowCount s) => y)).1 =
        List.ofFn fun i => Hm96.Default512.commitString
          (Sha512.hash (honestMsgs s D P w r a p i fun j => l (Fin.castLE (Nat.le_of_lt i.isLt) j))) y := by
  obtain ⟨hfin, hn, hh⟩ := run_honestReqs hs (hD P w r a p l) (List.ofFn fun _ : Fin (rowCount s) => y)
    (List.length_ofFn ..)
  refine ⟨hfin, hn, hh.trans ?_⟩
  apply List.ext_getElem
  · rw [List.length_zipWith, List.length_ofFn, List.length_ofFn, length_honestRows hs (hD P w r a p l), Nat.min_self]
  · intro i h1 h2
    rw [List.getElem_zipWith, List.getElem_ofFn, List.getElem_ofFn]
    rw [List.length_ofFn] at h2
    rw [honestMsgs_digest hs hD hc P w r a p l ⟨i, h2⟩]

end FlockSoundness.Discharge.Recursion.FirewallHonest
