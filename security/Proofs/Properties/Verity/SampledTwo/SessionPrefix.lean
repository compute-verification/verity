module

public import Proofs.Properties.Verity.SampledTwo.SessionDev

/-!
# The session-round locator reads only the frames before a release

The session-round locator (`locOf`) puts an opening on a session's round from three things it reads of the releases
before it: the coin trees they open in the order first opened, the specs of the trees they sample, and how many rounds
of each stream they open (`Same`). Each is a function of those releases' frames alone: no challenge a release carries
and no bytes it answers enters. So after two logs of the same frames, one release for one, a frame is put on the same
round, whatever challenges the device released in either:

* `same_of_frames`: releases of the same frames are the same to the locator (`Same`), from `openTrees_frames`,
  `Verity.Sampled.helloSpecs_frames` and `opened_frames`;
* `locOf_frames`, `locOf_after`: the session-round locator gives a frame the same round after either log, and the
  release after a prefix the same round where its frame is kept;
* `locates_frames`, `locates_after`: so does any locator that agrees with it at the sessions' rounds
  (`LocatesSessions`, as the run's does: `locates_run`), at a session's round, whatever bytes the frame answers;
* `same_append_of_frames`, `frames_of_read`, `locOf_logged`, `locates_logged`: the log the check read has the device's
  frames after releases that neither sample nor open a coin tree (`InteriorLogged`), so a release of the device's log
  is on the round its like in the read log is on, and what is shown of the read log's rounds holds of the rounds
  `CoinMap`'s locator gives the device's releases;
* `FramesOnly`, `framesOnly_run`, `locAt_frames`: the run's locator (`runLocate`) reads only the frames before a
  release at every round, where the stages' locator does.

This is the locator's part of what fixes a round's opening before the round's challenge: at two challenge vectors at
which the device's log before an opening has the same frames, and the opening the same frame, the opening is on the
same round.

**Not established here**: that the log before a round's opening is the same at two challenge vectors that agree on
the earlier rounds, that the releases before it are on earlier rounds, and that the stages' locator reads only the
frames before a release (`FramesOnly` at the two-stage audit's stages' locator, whose module this one does not import).
-/

@[expose] public section

namespace Verity.SampledTwo.SessionCoins

open Verity.Sampled.Binding (Read)

/-! ## Releases of the same frames -/

/-- Releases of the same frames open the same coin trees, in the same order. -/
theorem openTrees_frames :
    ∀ {l l' : List Release}, List.Forall₂ (fun a b => a.frame = b.frame) l l' →
      (l.filterMap fun r => openTree r.frame) = l'.filterMap fun r => openTree r.frame
  | _, _, .nil => rfl
  | x :: _, y :: _, .cons h t => by
    have e := openTrees_frames t
    have hf : x.frame = y.frame := h
    simp only [List.filterMap_cons, hf, e]

/-- Releases of the same frames open each stream's rounds equally often. -/
theorem opened_frames {i s : ℕ} :
    ∀ {l l' : List Release}, List.Forall₂ (fun a b => a.frame = b.frame) l l' → openedOf l i s = openedOf l' i s
  | _, _, .nil => rfl
  | _ :: _, _ :: _, .cons h t => by
    have e := opened_frames (i := i) (s := s) t
    simp only [openedOf, List.countP_cons, h] at e ⊢
    rw [e]

/-- **Releases of the same frames are the same to the session-round locator** (`Same`): they open the same coin trees
in the same order, sample the same trees' specs and open each stream's rounds equally often. -/
theorem same_of_frames {pre pre' : List Release} (h : List.Forall₂ (fun a b => a.frame = b.frame) pre pre') :
    Same pre pre' := by
  refine ⟨?_, Verity.Sampled.helloSpecs_frames h, fun _ _ => opened_frames h⟩
  unfold treesOpened
  rw [openTrees_frames h]

/-! ## The round a frame is put on -/

/-- **The session-round locator reads only the frames before a release**: after two logs of the same frames it puts
a frame on the same round. -/
theorem locOf_frames (A : Live₂) (tab : ℕ → String) {pre pre' : List Release}
    (h : List.Forall₂ (fun a b => a.frame = b.frame) pre pre') (fr : Verity.Frame) :
    locOf A tab pre fr = locOf A tab pre' fr := by
  unfold locOf
  exact sessLoc_congr (same_of_frames h) fr

/-- **The release after a prefix is on the same round wherever the prefix's frames and its own are kept**: where the
log is `q` and then `rl` at one challenge vector and `q'` and then `rl'` at another, `q` and `q'` of the same frames
and `rl` and `rl'` of one frame, the session-round locator puts `rl` after `q` where it puts `rl'` after `q'`. -/
theorem locOf_after (A : Live₂) (tab : ℕ → String) {q q' : List Release} {rl rl' : Release}
    (hq : List.Forall₂ (fun a b => a.frame = b.frame) q q') (hf : rl.frame = rl'.frame) :
    locOf A tab q rl.frame = locOf A tab q' rl'.frame :=
  (locOf_frames A tab hq rl.frame).trans (congrArg (locOf A tab q') hf)

/-- **A locator that agrees with the session-round locator at the sessions' rounds reads, there, only the frames
before a release**: after two logs of the same frames it puts a frame on a session's round `n` in both or in neither,
whatever bytes the frame answers in each. -/
theorem locates_frames {A : Live₂} {tab : ℕ → String}
    {L : List Release → Verity.Frame → ByteArray → Option ℕ} (hL : LocatesSessions A tab L) {n : ℕ}
    (hg : groupOf A n = .sessions) {pre pre' : List Release}
    (h : List.Forall₂ (fun a b => a.frame = b.frame) pre pre') (fr : Verity.Frame) (b b' : ByteArray) :
    L pre fr b = some n ↔ L pre' fr b' = some n :=
  have e := locOf_frames A tab h fr
  ⟨fun hn => (hL pre' fr b' n hg).2 (e.symm.trans ((hL pre fr b n hg).1 hn)),
    fun hn => (hL pre fr b n hg).2 (e.trans ((hL pre' fr b' n hg).1 hn))⟩

/-- **The release after a prefix is on the same session's round at a locator of `CoinMap`**, wherever the prefix's
frames and its own are kept: at a locator `L` that agrees with the session-round locator at the sessions' rounds
(`CoinMap.locAt L`), `rl` after `q` is on a session's round `n` just where `rl'` after `q'` is, whatever bytes each
answers. -/
theorem locates_after {A : Live₂} {tab : ℕ → String} {L : CoinMap.Locate}
    (hL : LocatesSessions A tab (CoinMap.locAt L)) {n : ℕ} (hg : groupOf A n = .sessions)
    {q q' : List Release} {rl rl' : Release}
    (hq : List.Forall₂ (fun a b => a.frame = b.frame) q q') (hf : rl.frame = rl'.frame) :
    CoinMap.locAt L q rl.frame rl.answers = some n ↔ CoinMap.locAt L q' rl'.frame rl'.answers = some n := by
  rw [← hf]
  exact locates_frames hL hg hq rl.frame rl.answers rl'.answers

/-! ## The log the check read and the device's -/

/-- Releases of the same frames, after releases that are the same to the session-round locator, are the same to
it. -/
theorem same_append_of_frames {a a' : List Release} (ha : List.Forall₂ (fun x y => x.frame = y.frame) a a') :
    ∀ {pre pre' : List Release}, Same pre pre' → Same (pre ++ a) (pre' ++ a') := by
  induction ha with
  | nil =>
    intro pre pre' h
    simpa using h
  | @cons x y a a' hxy _ ih =>
    intro pre pre' h
    simpa using ih (h.snoc hxy)

/-- The log the check read has the frames of the device's releases it reads, one for one. -/
theorem frames_of_read {l l' : List Release} (hR : List.Forall₂ Read.Of (l.map Read.ofV2) l') :
    List.Forall₂ (fun x y => x.frame = y.frame) l l' :=
  (List.forall₂_map_left_iff.1 hR).imp fun _ _ h => (of_ofV2 h).1

/-- **A release of the device's log is on the round its like in the log the check read is on**: where the device's
log is releases `q` that neither sample nor open a coin tree and then releases of the read log's frames (as
`InteriorLogged` gives of the interior read, with `frames_of_read`), the session-round locator puts the release `r`
after the read releases `a` where it puts the device's `r'` after `q` and the device's `a'`. -/
theorem locOf_logged (A : Live₂) (tab : ℕ → String) {q a a' : List Release} {r r' : Release}
    (hq : ∀ x ∈ q, openTree x.frame = none ∧ ∀ sp, x.frame ≠ .hello sp)
    (ha : List.Forall₂ (fun x y => x.frame = y.frame) a a') (hf : r.frame = r'.frame) :
    locOf A tab a r.frame = locOf A tab (q ++ a') r'.frame := by
  have hS : Same [] q := by simpa using same_append (pre := []) hq
  have hS' : Same a (q ++ a') := by simpa using same_append_of_frames ha hS
  rw [← hf]
  unfold locOf
  exact sessLoc_congr hS' r.frame

/-- **A locator of `CoinMap` puts a release of the device's log on a session's round just where the session-round
locator puts its like in the log the check read**: at a locator that agrees with the session-round locator at the
sessions' rounds, and logs as in `locOf_logged`. So what is shown of the rounds of the read log's releases holds of
the rounds `CoinMap` gives the device's. -/
theorem locates_logged {A : Live₂} {tab : ℕ → String} {L : CoinMap.Locate}
    (hL : LocatesSessions A tab (CoinMap.locAt L)) {n : ℕ} (hg : groupOf A n = .sessions)
    {q a a' : List Release} {r r' : Release}
    (hq : ∀ x ∈ q, openTree x.frame = none ∧ ∀ sp, x.frame ≠ .hello sp)
    (ha : List.Forall₂ (fun x y => x.frame = y.frame) a a') (hf : r.frame = r'.frame) :
    CoinMap.locAt L (q ++ a') r'.frame r'.answers = some n ↔ locOf A tab a r.frame = some n :=
  have e := locOf_logged A tab hq ha hf
  ⟨fun h => e.trans ((hL (q ++ a') r'.frame r'.answers n hg).1 h),
    fun h => (hL (q ++ a') r'.frame r'.answers n hg).2 (e.symm.trans h)⟩

/-! ## The run's locator -/

/-- **A locator that reads only the frames before a release**: after two logs of the same frames it puts a frame,
with the bytes it answers, on the same round. -/
def FramesOnly (L : CoinMap.Locate) : Prop :=
  ∀ (pre pre' : List Release) (fr : Verity.Frame) (b : ByteArray),
    List.Forall₂ (fun x y => x.frame = y.frame) pre pre' → L pre fr b = L pre' fr b

/-- **The run's locator reads only the frames before a release**, where the stages' locator does. -/
theorem framesOnly_run (A : Live₂) {Ls : CoinMap.Locate} (tab : ℕ → String) (hLs : FramesOnly Ls) :
    FramesOnly (runLocate A Ls tab) := by
  intro pre pre' fr b h
  cases fr with
  | opening i s c =>
    show locOf A tab pre (.opening i s c) = locOf A tab pre' (.opening i s c)
    exact locOf_frames A tab h _
  | hello sp =>
    show (Ls pre (.hello sp) b).filter (fun n => decide (n ≤ A.h.stage)) =
      (Ls pre' (.hello sp) b).filter (fun n => decide (n ≤ A.h.stage))
    rw [hLs pre pre' (.hello sp) b h]
  | fresh l =>
    show (Ls pre (.fresh l) b).filter (fun n => decide (n ≤ A.h.stage)) =
      (Ls pre' (.fresh l) b).filter (fun n => decide (n ≤ A.h.stage))
    rw [hLs pre pre' (.fresh l) b h]

/-- **The round a release continues is the same after two logs of the same frames**, at a locator that reads only
the frames before a release. -/
theorem locAt_frames {L : CoinMap.Locate} (hL : FramesOnly L) {pre pre' : List Release}
    (h : List.Forall₂ (fun a b => a.frame = b.frame) pre pre') (fr : Verity.Frame) (b : ByteArray) :
    CoinMap.locAt L pre fr b = CoinMap.locAt L pre' fr b := by
  cases fr with
  | hello _ => rfl
  | fresh l => exact hL pre pre' (.fresh l) b h
  | opening i s c => exact hL pre pre' (.opening i s c) b h

end Verity.SampledTwo.SessionCoins
