module

public import Properties.Pouw.PearlPlusTie
public import Proofs.Properties.Pouw.PearlPlus.Verdict

/-!
# An accepted verdict on an input that names W is a run held to W (`AcceptedIsHeldToW`)

`plan` reads the tie before its coin: on an input that names W it gives a plan only where `tieAt` answers, and the plan
carries that tie (`plan_tie`); `check` answers with the plan's tie on both of its branches (`check_tie`). `tieAt`'s
answer is `tieOf`'s but for its `load` (`tieAt_ok`), so the run is held to W at `tieOf`. With what `acceptMeans` and
`planMeans` establish, an accepted run's tiles are the run's own on the plan's sample, each accepted.
-/

@[expose] public section

namespace Properties.Pouw.PearlPlus

open PouwCommit PouwCheck.PearlPlus PouwCheck.PearlPlus.Words

/-- **`tieAt` answers with `tieOf`'s tie but for its `load`**: where `tieAt` gives `tied`, `tieOf` gives a `t` that
`tied` is, with `tied`'s `load` in place of `t`'s. -/
theorem tieAt_ok {load : Option LoadClass} {w : Lean.Json} {ps : List (Lean.Json × Pass)} {r : Option (List Nat)}
    {d : List Nat} {tied : Tied} (h : tieAt load w ps r d = .ok tied) :
    ∃ t, tieOf w ps r d = .ok t ∧ tied = { t with load := tied.load } := by
  unfold tieAt at h
  obtain ⟨t, ht, h⟩ := bind_ok h
  refine ⟨t, ht, ?_⟩
  cases load with
  | none =>
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    rfl
  | some c =>
    dsimp only at h
    split at h
    · simp only [pure, Except.pure, Except.ok.injEq] at h
      subst h
      rfl
    · simp [throw, throwThe, MonadExceptOf.throw] at h

theorem plan_tie {i : Input} {pl : Plan} {w : Lean.Json} (h : plan i = .ok pl) (hw : i.tie = some w) :
    ∃ tied : Tied, tieAt i.load w (i.manifests.zip pl.passes) i.rounds i.dropped = .ok tied ∧ pl.tie = some tied := by
  unfold plan at h
  extract_lets -underBinder j0 at h
  split at h
  · cases h
  dsimp (config := { zeta := false }) only [j0] at h
  cases hd : passesDistinct i.manifests with
  | some e => rw [throwRefusal.eq_def, hd] at h; cases h
  | none =>
  rw [throwRefusal.eq_def, hd] at h
  try dsimp (config := { zeta := false }) only at h
  obtain ⟨passes, -, h⟩ := bind_ok h
  try dsimp (config := { zeta := false }) only at h
  extract_lets -underBinder js at h
  split at h
  · cases h
  dsimp (config := { zeta := false }) only [js] at h
  extract_lets -underBinder jt at h
  obtain ⟨u, h⟩ : ∃ u : Unit, jt u = .ok pl := by
    split at h
    · split at h
      · cases h
      · exact ⟨(), h⟩
    · exact ⟨(), h⟩
  dsimp (config := { zeta := false }) only [jt] at h
  extract_lets -underBinder jtie at h
  obtain ⟨tied, ht, h⟩ : ∃ tied, tieAt i.load w (i.manifests.zip passes) i.rounds i.dropped = .ok tied ∧
      jtie (some tied) = .ok pl := by
    split at h
    · rename_i hn
      rw [hw] at hn
      cases hn
    · rename_i w' hs
      rw [hw] at hs
      cases hs
      split at h
      · rename_i t ht
        obtain ⟨t', ht', h⟩ := bind_ok h
        cases ht'
        exact ⟨t, ht, h⟩
      · cases h
  dsimp (config := { zeta := false }) only [jtie] at h
  cases ha : afterTie i passes (some tied) with
  | error r => rw [ha] at h; cases h
  | ok pl' =>
  rw [ha] at h
  have hpl : pl' = pl := Except.ok.inj h
  subst hpl
  obtain ⟨hpass, htie, -⟩ := afterTie_ok ha
  subst hpass
  exact ⟨tied, ht, htie⟩

theorem check_tie {i : Input} {pl : Plan} (opener : Nat → Nat → Except String TileOpening)
    (hp : plan i = .ok pl) : (check i opener).tie = pl.tie := by
  unfold check
  rw [hp]
  dsimp only
  split <;> rfl

theorem acceptedIsHeldToW : AcceptedIsHeldToW := by
  intro i opener w hacc hw
  obtain ⟨pl, hp, -, -, -, -, hall⟩ := acceptMeans i opener hacc
  obtain ⟨tied, ht, hpt⟩ := plan_tie hp hw
  obtain ⟨-, hload, -, -, -, -, -, key, drawn, -, -, htodo, -, hpicks⟩ := planMeans i pl hp
  obtain ⟨t, ht', htt⟩ := tieAt_ok ht
  refine ⟨pl, tied, hp, ht, (check_tie opener hp).trans hpt,
    fun hwt T => ⟨⟨hload, t, ht', (congrArg Tied.weights htt).symm.trans hwt⟩, ?_⟩⟩
  intro pi tid hm
  have hm' : (pi, tid) ∈ pl.todo := by rw [htodo, ← hpicks]; exact hm
  exact hall pi tid hm'

end Properties.Pouw.PearlPlus
