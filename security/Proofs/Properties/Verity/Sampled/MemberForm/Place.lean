module

public import Proofs.Properties.Verity.Sampled.MemberForm.Order

/-!
# The sessions' trees are the trees the read opens, in the sessions' order

A window's sessions are placed by the trees the device's log opens, in the order it first opens them
(`Verity.Sampled.openedTrees`): the `j`-th session's tree is the `j`-th of them. The members' check gives the
sessions' trees as the read numbers them, the sessions' rounds one session after another and each session's closing
round in the read (`members_read`). So:

* **A log opens the tree of each of its openings** (`opening_mem_openedTrees`).
* **The sessions' places** (`session_place`, `session_count`): where a numbering of `m` sessions' trees gives no
  tree to two sessions, every opening of the log is of a session's tree, the sessions' rounds are opened one session
  after another and each session's tree has an opening, the `j`-th tree the log opens is session `j`'s, and the log
  opens `m` trees.
* **By the members' check** (`members_place`, `members_count`): where `Spot.Challenger.membersTreeReason` accepts
  the sessions' records against the release log's lines, the `j`-th tree the read opens is the `j`-th session's, and
  the read opens as many trees as sessions.
-/

@[expose] public section

namespace Verity.Sampled.MemberForm

open Lean (Json)
open Spot Spot.Challenger

/-! ## The sessions' places -/

/-- **A log opens the tree of each of its openings.** -/
theorem opening_mem_openedTrees {L : List Verity.Release} {r : Verity.Release} (hr : r ∈ L) {i s n : ℕ}
    (hf : r.frame = .opening i s n) : i ∈ openedTrees L := by
  obtain ⟨a, b, rfl⟩ := List.append_of_mem hr
  by_cases hi : i ∈ openedTrees a
  · obtain ⟨e, he⟩ := openedTrees_append a (r :: b)
    rw [he]
    exact List.mem_append_left _ hi
  · obtain ⟨e, he⟩ := openedTrees_new hf hi b
    rw [he]
    exact List.mem_append_right _ List.mem_cons_self

/-- **The sessions' trees are the trees the log opens, in the sessions' order**: where `tree` numbers `m` sessions'
trees, no tree two sessions', every opening of the log is of a session's tree, the sessions' rounds are opened one
session after another and each session's tree has an opening, the `j`-th tree the log opens is session `j`'s. -/
theorem session_place {tree : ℕ → ℕ} {m : ℕ} {L : List Verity.Release}
    (hinj : ∀ j j', j < m → j' < m → tree j = tree j' → j = j')
    (hall : ∀ r ∈ L, ∀ i s n, r.frame = .opening i s n → ∃ j, j < m ∧ tree j = i)
    (hord : OpensInOrder tree m L)
    (hclose : ∀ j, j < m → ∃ Lc c L₂ s n, L = Lc ++ c :: L₂ ∧ c.frame = .opening (tree j) s n)
    {j : ℕ} (hj : j < m) : (openedTrees L)[j]? = some (tree j) := by
  obtain ⟨Lc, c, Lc₂, sc, nc, hLc, hcf⟩ := hclose j hj
  have hmem : tree j ∈ openedTrees L :=
    opening_mem_openedTrees (by rw [hLc]; exact List.mem_append_right _ List.mem_cons_self) hcf
  obtain ⟨u, hu⟩ := List.getElem?_of_mem hmem
  obtain ⟨L₁, x, L₂, s, n, he, hx, hni, -⟩ := openedTrees_first_at hu
  have hnd : ((List.range j).map tree).Nodup :=
    List.Nodup.map_on (fun a ha b hb e =>
      hinj a b (Nat.lt_trans (List.mem_range.1 ha) hj) (Nat.lt_trans (List.mem_range.1 hb) hj) e) List.nodup_range
  have hperm : (openedTrees L₁).Perm ((List.range j).map tree) := by
    refine (List.perm_ext_iff_of_nodup (openedTrees_nodup L₁) hnd).2 fun i => ⟨fun hi => ?_, fun hi => ?_⟩
    · obtain ⟨q, hq, s', n', hqf⟩ := mem_openedTrees hi
      obtain ⟨j', hj', rfl⟩ := hall q (by rw [he]; exact List.mem_append_left _ hq) i s' n' hqf
      have hle := hord L₁ x L₂ he j hj s n hx q hq j' hj' s' n' hqf
      have hne : j' ≠ j := fun e => hni (e ▸ hi)
      exact List.mem_map.2 ⟨j', List.mem_range.2 (Nat.lt_of_le_of_ne hle hne), rfl⟩
    · obtain ⟨j', hj'r, rfl⟩ := List.mem_map.1 hi
      have hlt : j' < j := List.mem_range.1 hj'r
      have hj' : j' < m := Nat.lt_trans hlt hj
      obtain ⟨Lc', c', L₂', s', n', hLc', hcf'⟩ := hclose j' hj'
      have hcm : tree j' ∈ openedTrees L := by
        rw [hLc']
        exact opening_mem_openedTrees (List.mem_append_right _ List.mem_cons_self) hcf'
      obtain ⟨u', hu'⟩ := List.getElem?_of_mem hcm
      obtain ⟨P, y, P₂, sy, ny, hP, hy, -, -⟩ := openedTrees_first_at hu'
      rcases List.append_eq_append_iff.1 (he.symm.trans hP) with ⟨a, rfl, ha⟩ | ⟨a, rfl, ha⟩
      · cases a with
        | nil =>
          rw [List.nil_append] at ha
          obtain ⟨rfl, -⟩ := List.cons.inj ha
          have e : tree j = tree j' := by
            rw [hx] at hy
            exact (Verity.Frame.opening.inj hy).1
          exact absurd (hinj j j' hj hj' e) (Nat.ne_of_gt hlt)
        | cons b a =>
          rw [List.cons_append] at ha
          obtain ⟨rfl, rfl⟩ := List.cons.inj ha
          have hxm : x ∈ L₁ ++ x :: a := List.mem_append_right _ List.mem_cons_self
          have hle := hord (L₁ ++ x :: a) y P₂ hP j' hj' sy ny hy x hxm j hj s n hx
          exact absurd hle (Nat.not_le_of_gt hlt)
      · cases a with
        | nil =>
          rw [List.nil_append] at ha
          obtain ⟨rfl, -⟩ := List.cons.inj ha
          have e : tree j = tree j' := by
            rw [hx] at hy
            exact (Verity.Frame.opening.inj hy).1
          exact absurd (hinj j j' hj hj' e) (Nat.ne_of_gt hlt)
        | cons b a =>
          rw [List.cons_append] at ha
          obtain ⟨rfl, rfl⟩ := List.cons.inj ha
          exact opening_mem_openedTrees (List.mem_append_right _ List.mem_cons_self) hy
  have hlen : (openedTrees L₁).length = j := by
    rw [hperm.length_eq, List.length_map, List.length_range]
  have h1 := (openedTrees_new_at hx hni L₂).1
  rw [← he, hlen] at h1
  exact h1

/-- **The log opens as many trees as sessions**: where `tree` numbers `m` sessions' trees, no tree two sessions',
every opening of the log is of a session's tree and each session's tree has an opening. -/
theorem session_count {tree : ℕ → ℕ} {m : ℕ} {L : List Verity.Release}
    (hinj : ∀ j j', j < m → j' < m → tree j = tree j' → j = j')
    (hall : ∀ r ∈ L, ∀ i s n, r.frame = .opening i s n → ∃ j, j < m ∧ tree j = i)
    (hclose : ∀ j, j < m → ∃ Lc c L₂ s n, L = Lc ++ c :: L₂ ∧ c.frame = .opening (tree j) s n) :
    (openedTrees L).length = m := by
  have hnd : ((List.range m).map tree).Nodup :=
    List.Nodup.map_on (fun a ha b hb e => hinj a b (List.mem_range.1 ha) (List.mem_range.1 hb) e) List.nodup_range
  have hperm : (openedTrees L).Perm ((List.range m).map tree) := by
    refine (List.perm_ext_iff_of_nodup (openedTrees_nodup L) hnd).2 fun i => ⟨fun hi => ?_, fun hi => ?_⟩
    · obtain ⟨q, hq, s, n, hqf⟩ := mem_openedTrees hi
      obtain ⟨j, hj, rfl⟩ := hall q hq i s n hqf
      exact List.mem_map.2 ⟨j, List.mem_range.2 hj, rfl⟩
    · obtain ⟨j, hjr, rfl⟩ := List.mem_map.1 hi
      obtain ⟨Lc, c, L₂, s, n, hL, hcf⟩ := hclose j (List.mem_range.1 hjr)
      rw [hL]
      exact opening_mem_openedTrees (List.mem_append_right _ List.mem_cons_self) hcf
  rw [hperm.length_eq, List.length_map, List.length_range]

/-! ## By the members' check -/

variable {label : String} {after : ByteArray} {recs : List Json} {ξ : ByteArray} {lines : List String}
  {log : List Verity.Release} {trees : Array LoggedTree}
  {records : Array (String × Array (String × Option ℕ) × Json × String)}

/-- What `session_place` and `session_count` ask of the read, by the members' check: no tree two sessions', every
opening a session's tree's, the sessions' rounds one session after another, and each session's tree with an
opening, its closing round. -/
theorem members_sessions (hT : coinTreesOf lines = .ok trees) (hR : readLog label after recs ξ lines = some log)
    (hM : membersTreeReason trees lines records = none) {tree : ℕ → ℕ}
    (htree : ∀ j ρ, (usedOf records)[j]? = some ρ → (heldTrees trees).findIdx? (·.1 == ρ) = some (tree j)) :
    (∀ j j', j < records.size → j' < records.size → tree j = tree j' → j = j') ∧
      (∀ r ∈ log, ∀ i s n, r.frame = .opening i s n → ∃ j, j < records.size ∧ tree j = i) ∧
      OpensInOrder tree records.size log ∧
      ∀ j, j < records.size → ∃ Lc c L₂ s n, log = Lc ++ c :: L₂ ∧ c.frame = .opening (tree j) s n := by
  obtain ⟨-, hnod, hO⟩ := membersTreeReason_none hM
  have hsz := usedOf_size records
  refine ⟨fun a b ha hb e => ?_, fun r hr i s n hf => ?_, ?_, fun a ha => ?_⟩
  · exact tree_inj hnod htree (by rw [hsz]; exact ha) (by rw [hsz]; exact hb) e
  · obtain ⟨j', hj', e⟩ := opening_session hT hR hO htree hr hf
    exact ⟨j', by rw [← hsz]; exact hj', e⟩
  · intro pre r post he a ha s n hr q hq b hb s' n' hq'
    exact opens_in_order hT hR hO hnod htree he (by rw [hsz]; exact ha) hr hq (by rw [hsz]; exact hb) hq'
  · obtain ⟨t, ht, -, hu, -, o, ho, hs, hk, -⟩ := sessions_trees hM a _ (Array.getElem?_eq_getElem ha)
    obtain ⟨Lc, c, L₂, i, s, n, he, hcf, hkey, -, -, -, -⟩ := closing_read hT hR hO hnod ht ho hs hk
    have hi : i = tree a := Option.some.inj ((keyAt_idx hkey).symm.trans (htree a _ hu))
    subst hi
    exact ⟨Lc, c, L₂, s, n, he, hcf⟩

/-- **The `j`-th tree the read opens is the `j`-th session's**: where `coinTreesOf` accepts the release log's
lines, `readLog` reads them, `membersTreeReason` accepts the sessions `records` against them and `tree` numbers
each session's tree as the read does. -/
theorem members_place (hT : coinTreesOf lines = .ok trees) (hR : readLog label after recs ξ lines = some log)
    (hM : membersTreeReason trees lines records = none) {tree : ℕ → ℕ}
    (htree : ∀ j ρ, (usedOf records)[j]? = some ρ → (heldTrees trees).findIdx? (·.1 == ρ) = some (tree j))
    {j : ℕ} (hj : j < records.size) : (openedTrees log)[j]? = some (tree j) := by
  obtain ⟨hinj, hall, hord, hclose⟩ := members_sessions hT hR hM htree
  exact session_place hinj hall hord hclose hj

/-- **The read opens as many trees as sessions**, where `membersTreeReason` accepts the sessions `records` against
the release log's lines. -/
theorem members_count (hT : coinTreesOf lines = .ok trees) (hR : readLog label after recs ξ lines = some log)
    (hM : membersTreeReason trees lines records = none) : (openedTrees log).length = records.size := by
  obtain ⟨tree, htree⟩ := members_tree hT hR hM
  obtain ⟨hinj, hall, -, hclose⟩ := members_sessions hT hR hM htree
  exact session_count hinj hall hclose

end Verity.Sampled.MemberForm
