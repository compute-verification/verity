module

public import Proofs.Properties.Verity.Sampled.RecordDraw

/-!
# The sample, then the link, then the other rounds, on the device's log

The record's sample (`recordDraw_firstStop`) and the first stop (`link_pts`, `firstStop_rootB`) take the order
`SampleLinkFirst` states on the device's log: no release of a coin of the sample's family after the registration's
digest follows an opening, and the first opening is of the stream `Spot.Challenger.TREE_SESSION`. It holds
where v2's case accepts and the log v2's check read is the device's, since v2's check accepts only where the challenger
session's release log in v2's evidence, after its head, passes `sampleLinkFirst` at the parameters' challenger family
and the registration's digest (`Spot.OneStageZk.orderHeld`):

* **v2's refusal** (`checkBlobsAt_orderHeld`, `orderHeld_ok`): v2's check accepts only where its order refusal passed,
  and that refusal's scan is `sampleLinkFirst` (`sampleLinkFirst_check`).
* **On v2's read, then on the device's log** (`sampleLinkFirst_logged`): v2's read of those lines is in order
  (`sampleLinkFirst_readLog`), and the device's log has the read's frames and answers (`ofV2_frames`), which are all
  `SampleLinkFirst` reads (`sampleLinkFirst_congr`).
* **At the sampled verdict** (`sampleLinkFirst_verdict`, `freshBefore_verdict`, `linkFirst_verdict`): in the forms the
  record's sample and the first stop take.

v2's check needs that refusal: nothing else it checks gives the order (`reads_interleave`). Besides the chain
and the line count its signed close covers, which are the device's for whatever order it released in, v2 reads the
release log's coin trees' lines apart from its other lines (`coinTreesOf`, `coinTreesOf_filter`) and those others
apart from the coin trees' (`releasedRootsOf`, `releasedRootsOf_congr`), so it can't tell where a round's opening sits
among the sample's releases. And the device releases in the order the prover sends (`Verity.releases`), holding only
each stream's rounds to their order.
-/

@[expose] public section

namespace Verity.Sampled

open Lean (Json)
open Verity.Game
open Spot.Challenger
open Properties.Challenger (Draws)

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **Where v2's check accepts, its order refusal passed** (`Spot.OneStageZk.orderHeld`). -/
theorem checkBlobsAt_orderHeld {p : _root_.Spot.OneStageZk.Params}
    {c : _root_.Spot.OneStage.Commitment} {ξ : ByteArray} {e : _root_.Spot.OneStageZk.Evidence}
    {bs : _root_.Core.Blobs} {pub : _root_.Spot.OneStage.Public} {prof : _root_.Core.Integrity}
    (h : _root_.Spot.OneStageZk.checkBlobsAt p c ξ e bs = .accept pub prof) :
    _root_.Spot.OneStageZk.orderHeld p c e = .ok () := by
  replace h := (_root_.Spot.OneStageZk.checkBlobsAt_accept h).1
  unfold _root_.Spot.OneStageZk.checkOneAt at h
  dsimp only at h
  split at h
  · cases h
  · rename_i r hr
    obtain ⟨_, -, hr⟩ := FlockSoundness.Refine.ebind_ok hr
    obtain ⟨_, -, hr⟩ := FlockSoundness.Refine.ebind_ok hr
    obtain ⟨⟨⟩, ho, -⟩ := FlockSoundness.Refine.ebind_ok hr
    exact ho

/-- `linesOf` is v2's check's read of its evidence's log (`Spot.OneStageZk.linesOf`). -/
theorem linesOf_check (e : _root_.Spot.OneStageZk.Evidence) :
    _root_.Spot.OneStageZk.linesOf e = linesOf e := rfl

/-- `sampleLinkFirstFrom` is v2's check's scan (`Spot.Challenger.sampleLinkFirstFrom`). -/
theorem sampleLinkFirstFrom_check (label : String) (after : ByteArray) (log : List String) :
    ∀ opened pending, _root_.Spot.Challenger.sampleLinkFirstFrom label after opened pending log =
      sampleLinkFirstFrom label after opened pending log := by
  have ho : ∀ l, _root_.Spot.Challenger.openLine l = openLine l := fun _ => rfl
  have hk : ∀ l, _root_.Spot.Challenger.linkLine l = linkLine l := fun _ => rfl
  have hr : ∀ l, _root_.Spot.Challenger.rootOf l = rootOf l := fun _ => rfl
  induction log with
  | nil => intro _ _; simp only [_root_.Spot.Challenger.sampleLinkFirstFrom, sampleLinkFirstFrom]
  | cons l ls ih =>
    intro opened pending
    cases pending <;>
      simp only [_root_.Spot.Challenger.sampleLinkFirstFrom, sampleLinkFirstFrom, ho, hk, hr, ih]

/-- `sampleLinkFirst` is v2's check's (`Spot.Challenger.sampleLinkFirst`). -/
theorem sampleLinkFirst_check (label : String) (after : ByteArray) (log : List String) :
    _root_.Spot.Challenger.sampleLinkFirst label after log = sampleLinkFirst label after log :=
  sampleLinkFirstFrom_check label after log false none

/-- **v2's order refusal, passed, holds its log in order**: where `Spot.OneStageZk.orderHeld` passes, the
parameters name the challenger's family `label` and the evidence's log reads as `lines`, `lines` after the head pass
`sampleLinkFirst` at `label` and the registration's digest. -/
theorem orderHeld_ok {p : _root_.Spot.OneStageZk.Params} {c : _root_.Spot.OneStage.Commitment}
    {e : _root_.Spot.OneStageZk.Evidence} (h : _root_.Spot.OneStageZk.orderHeld p c e = .ok ())
    {key : ChallengerKey} {label : String} (hch : p.base.challenger = some (key, label)) {lines : List String}
    (hl : linesOf e = .ok lines) : sampleLinkFirst label c.digest lines.tail = true := by
  unfold _root_.Spot.OneStageZk.orderHeld at h
  rw [hch] at h
  dsimp only at h
  rw [linesOf_check, hl, ok_bind] at h
  rw [← sampleLinkFirst_check]
  revert h
  cases _root_.Spot.Challenger.sampleLinkFirst label c.digest lines.tail
  · intro h; cases h
  · intro _; rfl

/-- **v2's read of releases has their frames and the bytes they answered.** -/
theorem ofV2_frames :
    ∀ {l l' : List Verity.Release}, List.Forall₂ Binding.Read.Of (l.map Binding.Read.ofV2) l' →
      List.Forall₂ (fun x y => x.frame = y.frame ∧ x.answers = y.answers) l l'
  | [], [], _ => .nil
  | [], _ :: _, h => by simp at h
  | _ :: _, [], h => by simp at h
  | _ :: _, _ :: _, h => by
    simp only [List.map_cons, List.forall₂_cons] at h
    exact .cons ⟨h.1.1, h.1.2.1⟩ (ofV2_frames h.2)

/-- **The device's log is in order where v2's case accepts**: where composite `k` accepted what the prover sent, v2's
case in it accepts at the sample `dr` and the log v2's check read is the device's, the device's log releases the sample's
coins (of `dr`'s family, after `dr`'s digest) before every opening and opens the stream `TREE_SESSION` first, since
v2's check passed its order refusal (`checkBlobsAt_orderHeld`, `orderHeld_ok`). -/
theorem sampleLinkFirst_logged {k : Verity.Registration.Composite} {x : Verity.Exchange} {a b e : ByteArray}
    {p : _root_.Spot.OneStageZk.Params} {c : _root_.Spot.OneStage.Commitment}
    {ev : _root_.Spot.OneStageZk.Evidence} {ξ : ByteArray} {dr : Draws}
    (hS : Verity.SameAccepts k x.sent) (hcase : V2Case k x.sent a b e)
    (hacc : V2Accepts (_root_.Core.Blobs.ofArray x.sent.blobs) a b e p c ev ξ dr)
    (hl : Binding.Logged Binding.readsV2 x) :
    SampleLinkFirst dr.label dr.after x.log := by
  obtain ⟨lr, hlr⟩ := Option.isSome_iff_exists.1 (logRead_isSome hS hcase hacc)
  have hlog := hl (lr.map Binding.Read.ofV2) (by
    show lr.map Binding.Read.ofV2 ∈ (logRead x.sent).map (·.map Binding.Read.ofV2)
    rw [hlr]
    exact Option.mem_def.2 rfl)
  obtain ⟨hp, hc, hev, hξ, hdr, _, _, hck⟩ := hacc
  obtain ⟨key, hch, hafter⟩ := source_of hdr
  obtain ⟨raw, hraw, hrξ⟩ := Properties.Challenger.oneStageZk_reads p c dr hdr ev ξ hξ
  rw [hrξ] at hraw
  obtain ⟨cm, roots, htsn, -⟩ := drawCoins_ok hraw
  obtain ⟨_, hsn⟩ := trustedChallengePhaseOf_ok htsn
  obtain ⟨_, hso⟩ := challengePhaseNamed_ok hsn
  obtain ⟨ls, lines, hls0, hls, -⟩ := challengePhaseOf_ok hso
  have hL := linesOf_of hls0 hls
  have hcs := caseOf_of hS hcase hp hc hev
  rw [logRead_eq hcs hch hξ hL] at hlr
  obtain ⟨recs, -, hR⟩ := Option.bind_eq_some_iff.1 hlr
  have hR' : readLog dr.label c.digest recs ξ lines = some lr := hR
  rw [hafter]
  exact sampleLinkFirst_congr (ofV2_frames hlog)
    (sampleLinkFirst_readLog (orderHeld_ok (checkBlobsAt_orderHeld hck) hch hL) hR')

/-- **At the sampled verdict, the device's log is in order**: where `A` is the session v2 sets up on the parameters, the
sampled verdict accepts, the log v2's check read is the device's, and v2 reads the sample's coins of the family `label`
answering `after` (`SampleNamed`), the device's log is in order at `label` and `after` (`SampleLinkFirst`). -/
theorem sampleLinkFirst_verdict {V : Vocab d α} {k : Verity.Registration.Composite} {θ : ByteArray}
    {F : Array ByteArray} {A : Live} {x : Verity.Exchange} (hA : SetsUp V θ F A) (hθ : x.params = θ)
    (hv : V.Verdict k .sampled x) (hl : Binding.Logged Binding.readsV2 x)
    {label : String} {after : ByteArray} (hN : SampleNamed k x.sent label after) :
    SampleLinkFirst label after x.log := by
  subst hθ
  obtain ⟨p', hp', -⟩ := hA.params
  obtain ⟨a, b, e, p, c, ev, ξ, dr, hcase, hacc⟩ := case_drew hv.2.1 hp' (verdict_case hv)
  obtain ⟨rfl, rfl⟩ := hN a b e p c ev ξ dr hcase hacc
  exact sampleLinkFirst_logged hv.2.1 hcase hacc hl

/-- **The sample's coins are released before every opening** (`SampleLinkFirst`'s first half, the record's sample's
form), at the sampled verdict. -/
theorem freshBefore_verdict {V : Vocab d α} {k : Verity.Registration.Composite} {θ : ByteArray}
    {F : Array ByteArray} {A : Live} {x : Verity.Exchange} (hA : SetsUp V θ F A) (hθ : x.params = θ)
    (hv : V.Verdict k .sampled x) (hl : Binding.Logged Binding.readsV2 x)
    {label : String} {after : ByteArray} (hN : SampleNamed k x.sent label after) :
    ∀ pre r post, x.log = pre ++ r :: post → ∀ i s n, r.frame = .opening i s n →
      ∀ q ∈ post, ¬ (q.frame = .fresh label ∧ q.answers = after) :=
  fun pre r post he i s n hr => (sampleLinkFirst_verdict hA hθ hv hl hN pre r post he i s n hr).1

/-- **The first opening is of the stream `TREE_SESSION`** (`SampleLinkFirst`'s second half, the first stop's form), at
the sampled verdict. -/
theorem linkFirst_verdict {V : Vocab d α} {k : Verity.Registration.Composite} {θ : ByteArray}
    {F : Array ByteArray} {A : Live} {x : Verity.Exchange} (hA : SetsUp V θ F A) (hθ : x.params = θ)
    (hv : V.Verdict k .sampled x) (hl : Binding.Logged Binding.readsV2 x) :
    ∀ pre r post, x.log = pre ++ r :: post → ∀ i s n, r.frame = .opening i s n →
      (∀ q ∈ pre, ∀ i' s' n', q.frame ≠ .opening i' s' n') →
        ((helloSpecs pre)[i]?.bind fun sp => sp.streams[s]?) = some TREE_SESSION := by
  subst hθ
  obtain ⟨p', hp', -⟩ := hA.params
  obtain ⟨a, b, e, p, c, ev, ξ, dr, hcase, hacc⟩ := case_drew hv.2.1 hp' (verdict_case hv)
  exact fun pre r post he i s n hr => (sampleLinkFirst_logged hv.2.1 hcase hacc hl pre r post he i s n hr).2

/-! ## What v2 reads of the order -/

/-- A loop whose step at a member `p` fails yields its state unchanged runs as the loop over the members `p` passes. -/
theorem forIn_filter {β γ : Type} {m : Type → Type} [Monad m] [LawfulMonad m] (p : γ → Bool)
    {f : γ → β → m (ForInStep β)} (hf : ∀ x b, p x = false → f x b = pure (.yield b)) (l : List γ) :
    ∀ b, forIn l b f = forIn (l.filter p) b f := by
  induction l with
  | nil => intro b; rfl
  | cons x l ih =>
    intro b
    cases hx : p x
    · rw [List.filter_cons_of_neg (by simp [hx]), List.forIn_cons, hf x b hx, pure_bind]
      exact ih b
    · rw [List.filter_cons_of_pos hx, List.forIn_cons, List.forIn_cons]
      congr 1
      funext s
      cases s with
      | done b' => rfl
      | yield b' => exact ih b'

/-- **v2's trees read the release log's coin trees' lines alone** (`coinTreesOf` passes over every other line). -/
theorem coinTreesOf_filter (log : List String) : coinTreesOf log = coinTreesOf (log.filter treeLine) := by
  unfold coinTreesOf
  try dsimp only
  rw [forIn_filter treeLine ?_ log]
  intro x b hx
  simp [hx]

/-- **v2's roots read the release log's head and its lines that aren't a coin tree's** (`releasedRootsOf`). -/
theorem releasedRootsOf_congr (c : LiveCommitment) {l l' : List String} (hh : l.head? = l'.head?)
    (ht : l.tail.filter (fun x => !treeLine x) = l'.tail.filter (fun x => !treeLine x)) :
    releasedRootsOf c l = releasedRootsOf c l' := by
  unfold releasedRootsOf
  try dsimp only
  rw [hh, ht]

/-- **v2's reads of the release log don't see where its coin trees' lines sit among the others**: two logs with one
head, the same coin trees' lines in the same order and the same other lines in the same order show the same trees
(`coinTreesOf`) and the same roots (`releasedRootsOf`), wherever a round's opening falls among the sample's releases.
-/
theorem reads_interleave (c : LiveCommitment) {h : String} {t t' : List String}
    (htree : t.filter treeLine = t'.filter treeLine)
    (hrest : t.filter (fun x => !treeLine x) = t'.filter (fun x => !treeLine x)) :
    coinTreesOf (h :: t) = coinTreesOf (h :: t') ∧ releasedRootsOf c (h :: t) = releasedRootsOf c (h :: t') := by
  refine ⟨?_, releasedRootsOf_congr c rfl hrest⟩
  rw [coinTreesOf_filter (h :: t), coinTreesOf_filter (h :: t'), List.filter_cons, List.filter_cons, htree]

end Verity.Sampled
