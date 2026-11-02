import Flock.Firewall.Honest
import Proofs.Flock.Recursive.FirewallTemplate
import Proofs.Flock.Recursive.FirewallWire
import Proofs.Flock.Recursive.FirewallJson
import Proofs.Flock.Recursive.FirewallSteps

/-!
# The honest worker's requests pass the firewall

`run_honestReqs`: on the honest worker's requests (`Flock.Firewall.honestReqs`) the firewall (`Flock.Firewall.run`), at
any salts as many as the schedule's rows (`rowCount`), finishes, its Commit names the tables in the worker's order, and
its commit strings are those of the worker's rows (`honestRows`: each table's level-0 top in Commit's order, then each
round's fresh tops and message) in salt order. It holds at every schedule the firewall holds a session to
(`Canonical`) and every worker that follows it (`Fits`): its Commit's tables the schedule's roots, each stream opened
once before its first round, and each round's caps, copies, fresh tops and message the schedule's entry.

`honestMsgs` is the messages `FirewallAccepts` reads: message `i` the bytes `rowDigest` hashes of the worker's row `i`
at the commit strings before it, the worker's rows reading no commit string at or after their own (`Causal`).
`honestReqs_accepts` (`FirewallAccepts`) is `run_honestReqs` at zero salts.
-/

namespace FlockSoundness.Discharge.Recursion.FirewallHonest

open Lean (Json)
open Flock Flock.Firewall

/-! ## The schedule and the worker's rows -/

/-- The firewall holds a session to the schedule (`Schedule.canonical`, which `schedule` checks at load). -/
def Canonical (s : Schedule) : Prop := s.canonical = true

/-- The schedule's roots: each table's name and its level-0 top's node count, in key order. -/
def rootsOf (s : Schedule) : List (String × Nat) :=
  match s.roots with
  | .obj ⟨⟨t⟩⟩ => t.toListModel.map fun e => (e.1, (natOf? e.2).getD 0)
  | _ => []

/-- The schedule's tables, in key order. -/
def tablesOf (s : Schedule) : List String := (rootsOf s).map (·.1)

/-- Round `i`'s rows in the schedule: its fresh tops' nodes and its message rows. -/
def roundCount (s : Schedule) (i : Nat) : Nat :=
  match roundSpec (s.rounds.getD i .null) with
  | some (_, _, _, tops, _, commits) => tops.sum + commits
  | none => 0

/-- **The schedule's rows**, as many as the firewall salts: its roots' nodes, then each round's. -/
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

/-- **The worker's rows**, in the order the firewall salts them: each table's level-0 top in Commit's order, then each
round's (`roundRows`). -/
def honestRows (s : Schedule) (h : Honest) : List (Array Nat) :=
  h.roots.flatMap (fun e => topRows e.2) ++ (List.range s.rounds.size).flatMap (roundRows s h)

/-- The bytes `rowDigest` hashes of a row. -/
def rowBytes (row : Array Nat) : ByteArray :=
  row.foldl (fun b w => (b.push (w % 256).toUInt8).push (w / 256 % 256).toUInt8)
    (Hm96.pad ("verity/sha512-row/v1".toUTF8.push 0 ++ ByteArray.mk #[1, 16] ++ be32 row.size.toUInt32) 128)

theorem rowDigest_eq (row : Array Nat) : rowDigest row = Sha512.hash (rowBytes row) := rfl

/-! ## The worker that follows the schedule -/

/-- Round `i` as the firewall takes it at the schedule's entry `(stream, k, n, tops, level0, commits)`: `split` reads its
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

/-- **The worker follows the schedule**, Commit naming the tables `tables`: each length in its word (`ReqFits`), Σ a
plain string, Commit's tables the schedule's roots with their node counts (in the order `tables`), each stream opened
once at a place up to the last round's, and each round as the firewall takes it (`RoundOK`). -/
structure Fits (s : Schedule) (tables : List String) (h : Honest) : Prop where
  pings : ∀ b ∈ h.pings, b.size < 2 ^ 32
  sigma : plainStr h.sigma = true
  hello : FirewallWire.ReqFits (.hello (printJson (helloWith s.hello h.sigma)))
  commit : FirewallWire.ReqFits (.commit h.rootF h.roots [])
  tables : h.roots.map (·.1) = tables
  roots : (h.roots.map fun e => (e.1, e.2.size / NODE_BYTES)).Perm (rootsOf s)
  caps : ∀ e ∈ h.roots, isCap e.2 = true
  link : 16 * s.y < 2 ^ 32
  opensFit : ∀ j, ∀ r ∈ honestOpens s h j, FirewallWire.ReqFits r
  opened : ((List.range (s.rounds.size + 1)).flatMap h.opens).Perm (s.streams.map (·.1))
  rounds : ∀ i < s.rounds.size, RoundOK s h i
  roundsFit : ∀ i, ∀ r ∈ honestRound s h i, FirewallWire.ReqFits r
  proofs : ∀ p ∈ h.proofs, p.1.toUTF8.size < 2 ^ 32 ∧ p.2.1 < 2 ^ 32 ∧ p.2.2.size < 2 ^ 32

/-! ## What a schedule the firewall holds a session to holds -/

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

/-! ## What the firewall does on the worker's requests -/

/-- The worker's rows are the schedule's count. -/
theorem length_honestRows {s : Schedule} (hs : Canonical s) {tables : List String} {h : Honest} (hf : Fits s tables h) :
    (honestRows s h).length = rowCount s := by
  sorry

/-- **The firewall finishes on the honest worker's requests**: at salts `ys`, as many as the schedule's rows, it
finishes, its Commit names the tables `tables`, and its commit strings are those of the worker's rows under `ys`, in
salt order. -/
theorem run_honestReqs {s : Schedule} (hs : Canonical s) {tables : List String} {h : Honest} (hf : Fits s tables h)
    (ys : List ByteArray) (hy : ys.length = rowCount s) :
    (run s (honestReqs s h) ys).2 = true ∧
      FirewallTemplate.namesOf (run s (honestReqs s h) ys).1.toList = tables ∧
      FirewallRun.holesOf (run s (honestReqs s h) ys).1 =
        List.zipWith (fun row y => Hm96.Default512.commitString (rowDigest row) y) (honestRows s h) ys := by
  sorry

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
theorem honestMsgs_digest {Prog Wit Rr Ωa Ωp : Type} {s : Schedule} (hs : Canonical s) {tables : List String}
    {D : Prog → Wit → Rr → Ωa → Ωp → (Fin (rowCount s) → ByteArray) → Honest}
    (hD : ∀ P w r a p l, Fits s tables (D P w r a p l)) (hc : Causal s D)
    (P : Prog) (w : Wit) (r : Rr) (a : Ωa) (p : Ωp) (l : Fin (rowCount s) → ByteArray) (i : Fin (rowCount s)) :
    Sha512.hash (honestMsgs s D P w r a p i fun j => l (Fin.castLE (Nat.le_of_lt i.isLt) j)) =
      rowDigest ((honestRows s (D P w r a p l))[i.val]'(by rw [length_honestRows hs (hD P w r a p l)]; exact i.isLt)) := by
  have e := hc P w r a p (pad i fun j => l (Fin.castLE (Nat.le_of_lt i.isLt) j)) l i.val fun j hj => by
    simp only [pad, hj, dite_true]
    rfl
  rw [rowDigest_eq, honestMsgs, List.getD_eq_getElem?_getD, e, List.getElem?_eq_getElem]
  rfl

/-- **`FirewallAccepts`'s body at one salt `y`**: at every commit strings `l`, the firewall at `rowCount s` salts `y`
finishes on the worker's requests, Commit names the tables `tables`, and its commit strings are those of `honestMsgs`
under `y`, message `i` read at `l`'s strings before it. -/
theorem honest_accepts {Prog Wit Rr Ωa Ωp : Type} {s : Schedule} (hs : Canonical s) {tables : List String}
    {D : Prog → Wit → Rr → Ωa → Ωp → (Fin (rowCount s) → ByteArray) → Honest}
    (hD : ∀ P w r a p l, Fits s tables (D P w r a p l)) (hc : Causal s D) (y : ByteArray)
    (P : Prog) (w : Wit) (r : Rr) (a : Ωa) (p : Ωp) (l : Fin (rowCount s) → ByteArray) :
    (run s (honestReqs s (D P w r a p l)) (List.ofFn fun _ : Fin (rowCount s) => y)).2 = true ∧
      FirewallTemplate.namesOf (run s (honestReqs s (D P w r a p l)) (List.ofFn fun _ : Fin (rowCount s) => y)).1.toList =
        tables ∧
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
