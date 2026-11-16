module

public import Proofs.Properties.Verity.Sampled.MemberForm.Sessions

/-!
# Each tree's first opening is its link's, by the members' check

What the verifier's read of a member's session holds the device's log to takes, of the log, that each tree's first
opening is of its stream named the session's (`LinkFirst`). The members' check holds the release log's lines to it:
`Spot.Challenger.membersTreeReason` refuses lines whose first round opened of a tree isn't of that stream
(`Spot.Challenger.linksFirst`). This module reads it off the check:

* **What the scan reads of a line** (`openedTree_some`) and **where it passes** (`linksFirstFrom_first`): a line
  opening a round of a tree is of the stream named the session's, or an earlier line opens a round of that tree.
* **A line opening a round names a tree read and a stream of it** (`treeFold_open`).
* **The members' check holds the lines to it** (`membersTreeReason_links`).
* **The read's first opening of each tree is its link's** (`linkFirst_readLog`, `linkFirst_members`): where
  `coinTreesOf` accepts the lines, `readLog` reads them and the lines pass `linksFirst`, an opening the read shows of
  a tree no earlier release opened is read from a line no earlier line of that tree comes before, so of the stream
  named the session's.
-/

@[expose] public section

namespace Verity.Sampled.MemberForm

open Lean (Json)
open Spot Spot.Challenger

/-! ## The scan -/

/-- **What `openedTree` reads of a line**: an event opening a round, of the tree it names. -/
theorem openedTree_some {l ρ : String} (h : openedTree l = some ρ) :
    ∃ d, event l = .ok d ∧ jsonStrIs (getD d "event") OPEN_EVENT = true ∧ getD d "tree" = .str ρ := by
  unfold openedTree at h
  cases he : event l with
  | error e =>
    rw [he] at h
    cases h
  | ok d =>
    rw [he] at h
    dsimp only at h
    by_cases hO : jsonStrIs (getD d "event") OPEN_EVENT = true
    · rw [ite_eq_left hO] at h
      cases hr : getD d "tree" with
      | str s =>
        rw [hr] at h
        exact ⟨d, rfl, hO, by rw [hr, Option.some.inj h]⟩
      | null => rw [hr] at h; simp at h
      | bool _ => rw [hr] at h; simp at h
      | num _ => rw [hr] at h; simp at h
      | arr _ => rw [hr] at h; simp at h
      | obj _ => rw [hr] at h; simp at h
    · rw [ite_eq_right hO] at h
      cases h

/-- **Where the scan passes, a line opening a round of a tree is its link's or not the tree's first**: the line is of
the stream named the session's, or the scan came to it from that tree, or a line before it opens a round of that
tree. -/
theorem linksFirstFrom_first : ∀ (preL : List String) (last : Option String) {l : String} {postL : List String}
    {ρ : String}, linksFirstFrom last (preL ++ l :: postL) = true → openedTree l = some ρ →
    _root_.Spot.Challenger.linkLine l = true ∨ last = some ρ ∨ ∃ l' ∈ preL, openedTree l' = some ρ
  | [], last, l, postL, ρ, h, hl => by
    rw [List.nil_append, linksFirstFrom, hl] at h
    simp only [Bool.and_eq_true, Bool.or_eq_true, beq_iff_eq] at h
    rcases h.1 with h1 | h1
    · exact .inr (.inl h1)
    · exact .inl h1
  | a :: preL, last, l, postL, ρ, h, hl => by
    rw [List.cons_append, linksFirstFrom] at h
    cases ha : openedTree a with
    | none =>
      rw [ha] at h
      rcases linksFirstFrom_first preL last h hl with h1 | h1 | ⟨l', hl', h1⟩
      · exact .inl h1
      · exact .inr (.inl h1)
      · exact .inr (.inr ⟨l', List.mem_cons_of_mem _ hl', h1⟩)
    | some root =>
      rw [ha] at h
      simp only [Bool.and_eq_true] at h
      rcases linksFirstFrom_first preL (some root) h.2 hl with h1 | h1 | ⟨l', hl', h1⟩
      · exact .inl h1
      · exact .inr (.inr ⟨a, List.mem_cons_self, by rw [ha, h1]⟩)
      · exact .inr (.inr ⟨l', List.mem_cons_of_mem _ hl', h1⟩)

/-! ## The lines the read reads -/

/-- **A line opening a round names a tree `treeFold` reads and a stream of it**: where `treeFold` reads the lines
`lines` after the trees `ts` as `tsF`, a coin tree's line of them whose `event` isn't `TREE_EVENT` names a tree and a
stream, that stream of that tree among them (`opening_of_treeStep`, `KeyAt.append`). -/
theorem treeFold_open {records : List Json} :
    ∀ (lines : List String) (ts tsF : List (String × Json)), treeFold records ts lines = some tsF →
      ∀ l ∈ lines, treeLine l = true → ∀ d, event l = .ok d → jsonStrIs (getD d "event") TREE_EVENT = false →
        ∃ root stream i si, getD d "tree" = .str root ∧ getD d "stream" = .str stream ∧ KeyAt tsF i si root stream
  | [], _, _, _, _, hl, _, _, _, _ => absurd hl List.not_mem_nil
  | l₀ :: ls, ts, tsF, h, l, hl, ht, d, hev, hTE => by
    simp only [treeFold] at h
    by_cases ht₀ : treeLine l₀ = true
    · rw [ite_eq_left ht₀] at h
      obtain ⟨⟨r, ts'⟩, hX₀, hF⟩ := Option.bind_eq_some_iff.1 h
      obtain ⟨j, hj, hX⟩ := Option.bind_eq_some_iff.1 hX₀
      rcases List.mem_cons.1 hl with rfl | hl
      · obtain rfl : j = d := Except.ok.inj ((event_of_toOption hj).symm.trans hev)
        obtain ⟨root, stream, _, _, i, si, _, hr, hs, -, -, -, -, rfl, -, hk, -⟩ := opening_of_treeStep hX hTE
        obtain ⟨more, rfl⟩ := treeFold_prefix ls _ tsF hF
        exact ⟨root, stream, i, si, hr, hs, hk.append more⟩
      · exact treeFold_open ls ts' tsF hF l hl ht d hev hTE
    · rw [ite_eq_right ht₀] at h
      rcases List.mem_cons.1 hl with rfl | hl
      · exact absurd ht ht₀
      · exact treeFold_open ls ts tsF h l hl ht d hev hTE

variable {label : String} {after : ByteArray} {recs : List Json} {ξ : ByteArray} {lines : List String}
  {log : List Verity.Release} {trees : Array LoggedTree}

/-! ## The check -/

/-- **The members' check holds the lines to their links first**: where `membersTreeReason` accepts the sessions
`records`, the first round the lines open of each tree is of its stream named the session's (`linksFirst`). -/
theorem membersTreeReason_links {records : Array (String × Array (String × Option ℕ) × Json × String)}
    (h : membersTreeReason trees lines records = none) : linksFirst lines = true := by
  unfold membersTreeReason at h
  dsimp only at h
  split at h
  · rename_i x hx
    have hp : (memberTreeReason trees x.2.1 x.2.2.1 x.2.2.2).isSome = true := (Array.find?_some hx :)
    cases hm : memberTreeReason trees x.2.1 x.2.2.1 x.2.2.2 with
    | none =>
      rw [hm] at hp
      simp at hp
    | some w =>
      rw [hm] at h
      simp at h
  · split at h
    · cases h
    · split at h
      · cases h
      · split at h
        · cases h
        · split at h
          · cases h
          · rename_i h4
            exact Decidable.of_not_not h4

/-! ## The read -/

/-- **The read's first opening of each tree is its link's**: where `treeFold` reads the release log's lines,
`readLog` reads them and the lines pass `linksFirst`, an opening the read shows of a tree no earlier release opened
is of that tree's stream named the session's. The opening is read from a line opening a round (`readLog_opening`);
were a line before it of that tree, the read would show an opening of the tree before it (`treeFold_open`, the
count of `readLog_opening`, `openedOf_zero`); so the scan holds the line to the stream named the session's
(`linksFirstFrom_first`). -/
theorem linkFirst_readLog {tsF : List (String × Json)} (hF : treeFold recs [] lines = some tsF)
    (hR : readLog label after recs ξ lines = some log) (hL : linksFirst lines = true) : LinkFirst log := by
  intro L₁ x L₂ i s n he hx hnew
  obtain ⟨preL, l, postL, ρ, name, k, hls, hro, hkey, hname, hcnt⟩ := readLog_opening hR hF he hx
  obtain ⟨ht, d, hev, hT, hr, hs, -⟩ := hro.1
  have hO : jsonStrIs (getD d "event") OPEN_EVENT = true := by
    rw [Lines.treeLine_ok hev, hT, Bool.false_or] at ht
    exact ht
  have hop : openedTree l = some ρ := by
    unfold openedTree
    rw [hev]
    simp only [hO, hr, ↓reduceIte]
  have hlk : _root_.Spot.Challenger.linkLine l = true := by
    unfold linksFirst at hL
    rw [hls] at hL
    rcases linksFirstFrom_first preL none hL hop with h1 | h1 | ⟨l', hl', h1⟩
    · exact h1
    · cases h1
    · exfalso
      obtain ⟨d', hev', hO', hr'⟩ := openedTree_some h1
      have hT' : jsonStrIs (getD d' "event") TREE_EVENT = false := by
        rw [Trees.strIs hO']
        simp [jsonStrIs, TREE_EVENT, OPEN_EVENT]
      have ht' : treeLine l' = true := by
        rw [Lines.treeLine_ok hev', hO', Bool.or_true]
      have hmem : l' ∈ lines := by
        rw [hls]
        exact List.mem_append_left _ hl'
      obtain ⟨root, stream, i', si', hr'', hs', hk'⟩ := treeFold_open lines [] tsF hF l' hmem ht' d' hev' hT'
      have e1 : root = ρ := Json.str.inj (hr''.symm.trans hr')
      rw [e1] at hk'
      have e2 : i' = i := Option.some.inj ((keyAt_idx hk').symm.trans (keyAt_idx hkey))
      rw [e2] at hk'
      have hpos : 0 < preL.countP (opensLine ρ stream) :=
        countP_pos_of_mem hl' (by rw [opensLine_open ht' hev' hT' hr' hs']; simp)
      rw [← hcnt i si' ρ stream hk', openedOf_zero hnew] at hpos
      exact absurd hpos (Nat.lt_irrefl 0)
  have hll : _root_.Spot.Challenger.linkLine l =
      (jsonStrIs (getD d "event") OPEN_EVENT && jsonStrIs (getD d "stream") TREE_SESSION) := by
    simp only [_root_.Spot.Challenger.linkLine, hev]
  rw [hll, hO, Bool.true_and, hs] at hlk
  have hn : name = TREE_SESSION := by simpa [jsonStrIs] using hlk
  rw [hname, hn]

/-- **The read's first opening of each tree is its link's, by the members' check**: where `coinTreesOf` accepts the
release log's lines, `readLog` reads them and `membersTreeReason` accepts the sessions `records` against them. -/
theorem linkFirst_members {records : Array (String × Array (String × Option ℕ) × Json × String)}
    (hT : coinTreesOf lines = .ok trees) (hR : readLog label after recs ξ lines = some log)
    (hM : membersTreeReason trees lines records = none) : LinkFirst log :=
  linkFirst_readLog (coinTreesOf_fold recs hT) hR (membersTreeReason_links hM)

end Verity.Sampled.MemberForm
