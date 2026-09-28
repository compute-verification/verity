import Proofs.Flock.Soundness.Discharge.PlaysCoin.Statements

/-!
# A tape from its groups

A tape is determined by its coin steps and its messages cut at them: `groups ts` is its messages cut at its coins
(`cut_kinds`), and `ungroup` puts each group's messages before the next coin (`ungroup_groups`). The decoded tape
`tapeFrom dec (fun _ => true) j rs`, then `tail`, is `ungroup` of the decoder's groups and `tail`, at the rounds'
`coinMsg true` (`ungroup_tapeFrom`). So `tapeOfGroups` (`TapeOfGroups`). `cutOfPat` (`CutOfPat`): groups and a tail,
along a spine whose coin pattern is theirs, cut back into them.
-/

namespace FlockSoundness.Discharge.PlaysCoin

open Flock FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge.Exec

/-- A tape's messages, cut at its coins. -/
def groups : List Step → List (List Msg)
  | [] => [[]]
  | .inl m :: ts => consHead m (groups ts)
  | .inr _ :: ts => [] :: groups ts

/-- A tape from its message groups and its coin steps: each group's messages, then the next coin. -/
def ungroup : List (List Msg) → List Msg → List Step
  | [], _ => []
  | [g], _ => g.map .inl
  | g :: _ :: _, [] => g.map .inl
  | g :: g' :: gs, c :: cs => g.map .inl ++ .inr c :: ungroup (g' :: gs) cs

theorem cut_kinds : ∀ ts : List Step, cut (ts.map kindOf) (ts.filterMap Sum.getLeft?) = groups ts
  | [] => rfl
  | .inl m :: ts => by
      rw [List.map_cons, List.filterMap_cons, Sum.getLeft?_inl]
      exact congrArg (consHead m) (cut_kinds ts)
  | .inr c :: ts => by
      rw [List.map_cons, List.filterMap_cons, Sum.getLeft?_inr]
      exact congrArg (List.cons []) (cut_kinds ts)

theorem groups_ne_nil : ∀ ts : List Step, groups ts ≠ []
  | [] => List.cons_ne_nil _ _
  | .inl m :: ts => by
      show consHead m (groups ts) ≠ []
      cases groups ts <;> exact List.cons_ne_nil _ _
  | .inr _ :: _ => List.cons_ne_nil _ _

theorem ungroup_consHead (m : Msg) {gs : List (List Msg)} (hgs : gs ≠ []) (cs : List Msg) :
    ungroup (consHead m gs) cs = .inl m :: ungroup gs cs := by
  rcases gs with _ | ⟨g, _ | ⟨g', gs⟩⟩
  · exact absurd rfl hgs
  · rfl
  · cases cs <;> rfl

theorem ungroup_cons_cons (g : List Msg) {gs : List (List Msg)} (hgs : gs ≠ []) (c : Msg) (cs : List Msg) :
    ungroup (g :: gs) (c :: cs) = g.map .inl ++ .inr c :: ungroup gs cs := by
  rcases gs with _ | ⟨g', gs⟩
  · exact absurd rfl hgs
  · rfl

theorem ungroup_groups : ∀ ts : List Step, ungroup (groups ts) (ts.filterMap Sum.getRight?) = ts
  | [] => rfl
  | .inl m :: ts => by
      rw [List.filterMap_cons, Sum.getRight?_inl]
      show ungroup (consHead m (groups ts)) _ = _
      rw [ungroup_consHead m (groups_ne_nil ts), ungroup_groups ts]
  | .inr c :: ts => by
      rw [List.filterMap_cons, Sum.getRight?_inr]
      show ungroup ([] :: groups ts) _ = _
      rw [ungroup_cons_cons [] (groups_ne_nil ts), ungroup_groups ts]
      rfl

theorem ungroup_tapeFrom (dec : ℕ → ByteArray → List Msg) (tail : List Msg) :
    ∀ (j : ℕ) (rs : List Round), ungroup (roundGroups dec j rs ++ [tail]) (rs.map (coinMsg true)) =
      tapeFrom dec (fun _ => true) j rs ++ tail.map .inl
  | _, [] => rfl
  | j, rd :: rs => by
      show ungroup (dec j (rd.bytes.getD ByteArray.empty) :: (roundGroups dec (j + 1) rs ++ [tail]))
        (coinMsg true rd :: rs.map (coinMsg true)) = _
      rw [ungroup_cons_cons _ (by simp), ungroup_tapeFrom dec tail (j + 1) rs]
      simp [tapeFrom]

theorem tapeOfGroups : TapeOfGroups := by
  intro dec j rs tail ts hc hg
  rw [cut_kinds] at hg
  calc ts = ungroup (groups ts) (ts.filterMap Sum.getRight?) := (ungroup_groups ts).symm
    _ = ungroup (roundGroups dec j rs ++ [tail]) (rs.map (coinMsg true)) := by rw [hg, hc]
    _ = tapeFrom dec (fun _ => true) j rs ++ tail.map .inl := ungroup_tapeFrom dec tail j rs

theorem cut_replicate : ∀ (σ : List Exec.Kind) (t : List Msg),
    coinPat σ = List.replicate t.length false → cut σ t = [t]
  | [], [], _ => rfl
  | [], _ :: _, h => by simp [coinPat, List.replicate_succ] at h
  | .msg _ :: _, [], h => by simp [coinPat] at h
  | .msg _ :: σ, m :: t, h => by
      simp only [coinPat, List.length_cons, List.replicate_succ, List.cons.injEq, true_and] at h
      show consHead m (cut σ t) = [m :: t]
      rw [cut_replicate σ t h]; rfl
  | .coin _ :: _, t, h => by cases t <;> simp [coinPat, List.replicate_succ] at h

theorem cutOfPat : CutOfPat := by
  intro σ G t
  induction G generalizing σ with
  | nil => intro h; simpa using cut_replicate σ t (by simpa [groupPat] using h)
  | cons g G ih =>
    induction g generalizing σ with
    | nil =>
      intro h
      match σ, h with
      | [], h => simp [coinPat, groupPat] at h
      | .msg _ :: _, h => simp [coinPat, groupPat] at h
      | .coin _ :: σ, h =>
        simp only [coinPat, groupPat, List.map_cons, List.flatMap_cons, List.length_nil, List.replicate_zero,
          List.nil_append, List.cons_append, List.cons.injEq, true_and] at h
        show [] :: cut σ (G.flatten ++ t) = _
        rw [ih σ h]; rfl
    | cons a g ihg =>
      intro h
      match σ, h with
      | [], h => simp [coinPat, groupPat, List.replicate_succ] at h
      | .coin _ :: _, h => simp [coinPat, groupPat, List.replicate_succ] at h
      | .msg _ :: σ, h =>
        have h' : coinPat σ = groupPat ((g :: G).map List.length) t.length := by
          simp only [coinPat, groupPat, List.map_cons, List.flatMap_cons, List.length_cons, List.replicate_succ,
            List.cons_append, List.cons.injEq, true_and] at h ⊢
          exact h
        have e := ihg σ h'
        simp only [List.flatten_cons, List.cons_append, List.append_assoc] at e ⊢
        show consHead a (cut σ (g ++ (G.flatten ++ t))) = _
        rw [e]; rfl

end FlockSoundness.Discharge.PlaysCoin
