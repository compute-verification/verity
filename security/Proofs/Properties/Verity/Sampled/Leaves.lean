module

public import Properties.Verity.Game
public import Proofs.Core.Game.Tape

/-!
# A play's coin trees hold at most `maxLeaves` leaves

The device samples a coin tree only at a `Hello` whose spec it samples (`TreeSpecOk`: a coin or more a round and at most
`maxTreeCoins` coins, so at most `maxTreeCoins` leaves, one a round), and it writes a coin-tree line at each `Hello` and
at each opening only while its log holds fewer than `maxTreeLines` of them (`treeLines`, `Verity.releases`). So every
run of a play writes at most `maxTreeLines` coin-tree lines (`play_lines_le`), and the trees it samples hold at most
`maxLeaves = maxTreeLines · maxTreeCoins` leaves in all (`releases_leaves_le`, `play_leaves_le`), with no premise: the
leaves `Live.hidingTerm` counts, twice at `2^-257` a leaf.
-/

@[expose] public section

namespace Verity.Sampled.Leaves

/-- **A coin tree's leaves**: one for each round of each stream (`TreeDraw.leaf`). -/
def treeLeaves (spec : Flock.CoinTree.Spec) : ℕ := spec.streams.size * 2 ^ spec.roundsLog

/-- A tree the device samples holds at most `maxTreeCoins` leaves: a leaf is a round, of a coin or more. -/
theorem treeLeaves_le {spec : Flock.CoinTree.Spec} (h : TreeSpecOk spec) : treeLeaves spec ≤ maxTreeCoins := by
  obtain ⟨h1, h2⟩ := h
  have h3 : treeLeaves spec ≤ treeCoins spec := Nat.le_mul_of_pos_right _ (by omega)
  exact h3.trans h2

/-- The leaves of the coin tree a frame has the device sample: a `Hello`'s tree's, and none at a coin or an opening. -/
def frameLeaves : Frame → ℕ
  | .fresh _ => 0
  | .hello spec => treeLeaves spec
  | .opening _ _ _ => 0

/-- The coin-tree lines a frame has the device write to its log: one at a `Hello` and one at an opening. -/
def frameLines : Frame → ℕ
  | .fresh _ => 0
  | .hello _ => 1
  | .opening _ _ _ => 1

/-- The leaves of the coin trees a log's `Hello`s sampled. -/
def logLeaves (log : List Release) : ℕ := (log.map fun r => frameLeaves r.frame).sum

/-- The coin-tree lines a log's releases wrote. -/
def logTreeLines (log : List Release) : ℕ := (log.map fun r => frameLines r.frame).sum

/-- A log samples at most `maxTreeCoins` leaves a coin-tree line where each of its releases does. -/
theorem logLeaves_le : ∀ {log : List Release},
    (∀ r ∈ log, frameLeaves r.frame ≤ maxTreeCoins * frameLines r.frame) →
      logLeaves log ≤ maxTreeCoins * logTreeLines log
  | [], _ => by simp [logLeaves, logTreeLines]
  | r :: log, h => by
    have h1 := h r List.mem_cons_self
    have h2 := logLeaves_le fun x hx => h x (List.mem_cons_of_mem r hx)
    simp only [logLeaves, logTreeLines, List.map_cons, List.sum_cons] at h2 ⊢
    rw [Nat.mul_add]
    exact Nat.add_le_add h1 h2

/-- Opening a round of stream `s` counts one more round opened in all. -/
theorem opened_bump_sum (T : OpenTree) (s : Fin T.spec.streams.size) :
    ∑ j, (T.bump s).opened j = ∑ j, T.opened j + 1 := by
  have h : ∀ j, (if j = s then T.opened s + 1 else T.opened j) = T.opened j + if j = s then 1 else 0 := by
    intro j
    by_cases hj : j = s <;> simp [hj]
  show ∑ j, (if j = s then T.opened s + 1 else T.opened j) = _
  rw [Finset.sum_congr rfl fun j _ => h j, Finset.sum_add_distrib]
  simp

/-- A `Hello` writes one coin-tree line: the tree it samples, with no round opened. -/
theorem treeLines_append_hello (ts : List OpenTree) (ν : ByteArray) (spec : Flock.CoinTree.Spec)
    (t : TreeSample spec) : treeLines (ts ++ [⟨ν, spec, t, fun _ => 0⟩]) = treeLines ts + 1 := by
  simp [treeLines]

/-- An opening of the `i`-th tree writes one coin-tree line: the round it opens. -/
theorem treeLines_set_bump : ∀ {ts : List OpenTree} {i : ℕ} {T : OpenTree}, ts[i]? = some T →
    ∀ s : Fin T.spec.streams.size, treeLines (ts.set i (T.bump s)) = treeLines ts + 1
  | [], _, _, h, _ => by simp at h
  | T₀ :: ts, 0, T, h, s => by
    simp only [List.getElem?_cons_zero, Option.some.injEq] at h
    subst h
    simp only [treeLines, List.set_cons_zero, List.map_cons, List.sum_cons, opened_bump_sum]
    omega
  | T₀ :: ts, i + 1, T, h, s => by
    simp only [List.getElem?_cons_succ] at h
    have ih := treeLines_set_bump h s
    simp only [treeLines, List.set_cons_succ, List.map_cons, List.sum_cons] at ih ⊢
    omega

/-- A log's coin-tree lines: its first release's, then the rest's. -/
theorem logTreeLines_cons (r : Release) (log : List Release) :
    logTreeLines (r :: log) = frameLines r.frame + logTreeLines log := by
  simp only [logTreeLines, List.map_cons, List.sum_cons]

/-- **The device's releases write at most the coin-tree lines its log has left**: every run of `releases k fs ts`
writes at most `maxTreeLines - treeLines ts` coin-tree lines, and each of its releases samples at most `maxTreeCoins`
leaves a line it writes. -/
theorem releases_lines_le : ∀ (k : ℕ) (fs : List (String × ℕ)) (ts : List OpenTree) (log : List Release),
    Game.Reaches (releases k fs ts) log →
      logTreeLines log ≤ maxTreeLines - treeLines ts ∧
        ∀ r ∈ log, frameLeaves r.frame ≤ maxTreeCoins * frameLines r.frame
  | 0, _, _, log, h => by
    cases h
    simp [logTreeLines]
  | k + 1, fs, ts, log, h => by
    -- a coin's release holds `List.ofFn` of 512 bits, which `simp` unrolls past `maxRecDepth`: read each release's
    -- frame by `rw` and `show`, never by `simp` over the release
    obtain ⟨m, hm⟩ := h
    rcases m with _ | ⟨_ | _ | _, b⟩
    · cases hm
      simp [logTreeLines]
    · rename_i l
      cases hc : takeCoin l fs with
      | none =>
        simp only [hc] at hm
        cases hm
        simp [logTreeLines]
      | some fs' =>
        simp only [hc] at hm
        obtain ⟨x, hx⟩ := hm
        obtain ⟨log', h', hret⟩ := Game.reaches_bind hx
        cases hret
        obtain ⟨h1, h2⟩ := releases_lines_le k fs' ts log' h'
        refine ⟨?_, fun r hr => ?_⟩
        · rw [logTreeLines_cons]
          show 0 + logTreeLines log' ≤ _
          omega
        · rcases List.mem_cons.1 hr with rfl | hr
          · exact Nat.zero_le _
          · exact h2 r hr
    · rename_i spec
      simp only at hm
      split at hm
      · rename_i hc
        obtain ⟨hc1, hc2⟩ := hc
        obtain ⟨t, ht⟩ := hm
        obtain ⟨log', h', hret⟩ := Game.reaches_bind ht
        cases hret
        obtain ⟨h1, h2⟩ := releases_lines_le k fs _ log' h'
        rw [treeLines_append_hello] at h1
        refine ⟨?_, fun r hr => ?_⟩
        · rw [logTreeLines_cons]
          show 1 + logTreeLines log' ≤ _
          omega
        · rcases List.mem_cons.1 hr with rfl | hr
          · exact Nat.le_trans (treeLeaves_le hc2) (Nat.le_of_eq (Nat.mul_one _).symm)
          · exact h2 r hr
      · cases hm
        simp [logTreeLines]
    · rename_i i s n
      simp only at hm
      split at hm
      · rename_i hc
        split at hm
        · cases hm
          simp [logTreeLines]
        · rename_i T hT
          split at hm
          · cases hm
            simp [logTreeLines]
          · obtain ⟨d, hd⟩ := hm
            obtain ⟨log', h', hret⟩ := Game.reaches_bind hd
            cases hret
            obtain ⟨h1, h2⟩ := releases_lines_le k fs _ log' h'
            rw [treeLines_set_bump hT] at h1
            refine ⟨?_, fun r hr => ?_⟩
            · rw [logTreeLines_cons]
              show 1 + logTreeLines log' ≤ _
              omega
            · rcases List.mem_cons.1 hr with rfl | hr
              · exact Nat.zero_le _
              · exact h2 r hr
      · cases hm
        simp [logTreeLines]

/-- **The device's releases sample at most `maxLeaves` leaves**: every run of `releases k fs []`, from no coin tree
open, samples coin trees of at most `maxLeaves` leaves in all. -/
theorem releases_leaves_le (k : ℕ) (fs : List (String × ℕ)) {log : List Release}
    (h : Game.Reaches (releases k fs []) log) : logLeaves log ≤ maxLeaves := by
  obtain ⟨h1, h2⟩ := releases_lines_le k fs [] log h
  have h3 : logTreeLines log ≤ maxTreeLines := by simpa [treeLines] using h1
  calc logLeaves log ≤ maxTreeCoins * logTreeLines log := logLeaves_le h2
    _ ≤ maxTreeCoins * maxTreeLines := Nat.mul_le_mul_left _ h3
    _ = maxLeaves := Nat.mul_comm _ _

/-- **A play writes at most `maxTreeLines` coin-tree lines**: every run of the play after `θ`, `F` and `c` ends with a
log of at most `maxTreeLines` coin-tree lines, a tree sampled or a round opened each. -/
theorem play_lines_le {fs : List (String × ℕ)} {D : Signer} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray}
    {o : Exchange} (h : Game.Reaches (play fs D θ F c) o) : logTreeLines o.log ≤ maxTreeLines := by
  obtain ⟨_, ha⟩ := h
  obtain ⟨log, hlog, _, _, _, he⟩ := Game.reaches_bind ha
  cases he
  have h1 := (releases_lines_le _ _ [] log hlog).1
  simpa [treeLines] using h1

/-- **A play's coin trees hold at most `maxLeaves` leaves**: every run of the play after `θ`, `F` and `c` ends with a
log whose `Hello`s sampled coin trees of at most `maxLeaves` (`2^46`) leaves in all. -/
theorem play_leaves_le {fs : List (String × ℕ)} {D : Signer} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray}
    {o : Exchange} (h : Game.Reaches (play fs D θ F c) o) : logLeaves o.log ≤ maxLeaves := by
  obtain ⟨_, ha⟩ := h
  obtain ⟨log, hlog, _, _, _, he⟩ := Game.reaches_bind ha
  cases he
  exact releases_leaves_le _ _ hlog

end Verity.Sampled.Leaves
