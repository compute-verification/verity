module

public import Proofs.Properties.Verity.Sampled.RecordDraw

/-! The sample, then the link, then the other rounds, on the device's log. -/

@[expose] public section

namespace Verity.Sampled

open Lean (Json)
open Verity.Game
open Spot.Challenger
open Properties.Challenger (Draws)

variable {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α]

theorem checkBlobsAt_orderHeld {p : _root_.Spot.OneStageZk.Params}
    {c : _root_.Spot.OneStage.Commitment} {ξ : ByteArray} {e : _root_.Spot.OneStageZk.Evidence}
    {bs : _root_.Core.Blobs} {pub : _root_.Spot.OneStage.Public} {prof : _root_.Core.Integrity}
    (h : _root_.Spot.OneStageZk.checkBlobsAt p c ξ e bs = .accept pub prof) :
    _root_.Spot.OneStageZk.orderHeld p c e = .ok () := by
  rcases _root_.Spot.OneStageZk.checkBlobsAt_accept h with ⟨-, -, h⟩ | ⟨ms, sts, -, -, h⟩
  swap
  · obtain ⟨r, hr, -, -⟩ := _root_.Spot.OneStageZk.checkMembersAt_accept h
    exact (_root_.Spot.OneStageZk.membersRun_held hr).1
  unfold _root_.Spot.OneStageZk.checkOneAt at h
  dsimp only at h
  split at h
  · cases h
  · rename_i r hr
    obtain ⟨_, -, hr⟩ := FlockSoundness.Refine.ebind_ok hr
    obtain ⟨_, -, hr⟩ := FlockSoundness.Refine.ebind_ok hr
    obtain ⟨⟨⟩, ho, -⟩ := FlockSoundness.Refine.ebind_ok hr
    exact ho

theorem linesOf_check (e : _root_.Spot.OneStageZk.Evidence) :
    _root_.Spot.OneStageZk.linesOf e = linesOf e := rfl

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

theorem sampleLinkFirst_check (label : String) (after : ByteArray) (log : List String) :
    _root_.Spot.Challenger.sampleLinkFirst label after log = sampleLinkFirst label after log :=
  sampleLinkFirstFrom_check label after log false none

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

theorem ofV2_frames :
    ∀ {l l' : List Verity.Release}, List.Forall₂ Binding.Read.Of (l.map Binding.Read.ofV2) l' →
      List.Forall₂ (fun x y => x.frame = y.frame ∧ x.answers = y.answers) l l'
  | [], [], _ => .nil
  | [], _ :: _, h => by simp at h
  | _ :: _, [], h => by simp at h
  | _ :: _, _ :: _, h => by
    simp only [List.map_cons, List.forall₂_cons] at h
    exact .cons ⟨h.1.1, h.1.2.1⟩ (ofV2_frames h.2)

theorem sampleLinkFirst_logged {k : Verity.Registration.Composite} {x : Verity.Exchange} {a b e : ByteArray}
    {p : _root_.Spot.OneStageZk.Params} {c : _root_.Spot.OneStage.Commitment}
    {ev : _root_.Spot.OneStageZk.Evidence} {ξ : ByteArray} {dr : Draws}
    (hS : Verity.SameAccepts k x.sent) (hcase : V2Case k x.sent a b e)
    (hacc : V2Accepts (_root_.Core.Blobs.ofArray x.sent.blobs) a b e p c ev ξ dr) (hm : p.members = none)
    (hl : Binding.Logged Binding.readsV2 x) :
    SampleLinkFirst dr.label dr.after x.log := by
  obtain ⟨lr, hlr⟩ := Option.isSome_iff_exists.1 (logRead_isSome hS hcase hacc hm)
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

theorem sampleLinkFirst_verdict {V : Vocab d α} {k : Verity.Registration.Composite} {θ : ByteArray}
    {F : Array ByteArray} {A : Live} {x : Verity.Exchange} (hA : SetsUp V θ F A) (hθ : x.params = θ)
    (hv : V.Verdict k .sampled x) (hl : Binding.Logged Binding.readsV2 x)
    {label : String} {after : ByteArray} (hN : SampleNamed k x.sent label after) :
    SampleLinkFirst label after x.log := by
  subst hθ
  obtain ⟨p', hp', -⟩ := hA.params
  obtain ⟨a, b, e, p, c, ev, ξ, dr, hcase, hacc⟩ := case_drew hv.2.1 hp' (verdict_case hv)
  obtain ⟨rfl, rfl⟩ := hN a b e p c ev ξ dr hcase hacc
  exact sampleLinkFirst_logged hv.2.1 hcase hacc (single_of_case hv.2.1 hcase hacc hA.single) hl

theorem freshBefore_verdict {V : Vocab d α} {k : Verity.Registration.Composite} {θ : ByteArray}
    {F : Array ByteArray} {A : Live} {x : Verity.Exchange} (hA : SetsUp V θ F A) (hθ : x.params = θ)
    (hv : V.Verdict k .sampled x) (hl : Binding.Logged Binding.readsV2 x)
    {label : String} {after : ByteArray} (hN : SampleNamed k x.sent label after) :
    ∀ pre r post, x.log = pre ++ r :: post → ∀ i s n, r.frame = .opening i s n →
      ∀ q ∈ post, ¬ (q.frame = .fresh label ∧ q.answers = after) :=
  fun pre r post he i s n hr => (sampleLinkFirst_verdict hA hθ hv hl hN pre r post he i s n hr).1

theorem linkFirst_verdict {V : Vocab d α} {k : Verity.Registration.Composite} {θ : ByteArray}
    {F : Array ByteArray} {A : Live} {x : Verity.Exchange} (hA : SetsUp V θ F A) (hθ : x.params = θ)
    (hv : V.Verdict k .sampled x) (hl : Binding.Logged Binding.readsV2 x) :
    ∀ pre r post, x.log = pre ++ r :: post → ∀ i s n, r.frame = .opening i s n →
      (∀ q ∈ pre, ∀ i' s' n', q.frame ≠ .opening i' s' n') →
        ((helloSpecs pre)[i]?.bind fun sp => sp.streams[s]?) = some TREE_SESSION := by
  subst hθ
  obtain ⟨p', hp', -⟩ := hA.params
  obtain ⟨a, b, e, p, c, ev, ξ, dr, hcase, hacc⟩ := case_drew hv.2.1 hp' (verdict_case hv)
  exact fun pre r post he i s n hr =>
    (sampleLinkFirst_logged hv.2.1 hcase hacc (single_of_case hv.2.1 hcase hacc hA.single) hl
      pre r post he i s n hr).2

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

theorem coinTreesOf_filter (log : List String) : coinTreesOf log = coinTreesOf (log.filter treeLine) := by
  unfold coinTreesOf
  try dsimp only
  rw [forIn_filter treeLine ?_ log]
  intro x b hx
  simp [hx]

theorem releasedRootsOf_congr (c : LiveCommitment) {l l' : List String} (hh : l.head? = l'.head?)
    (ht : l.tail.filter (fun x => !treeLine x) = l'.tail.filter (fun x => !treeLine x)) :
    releasedRootsOf c l = releasedRootsOf c l' := by
  unfold releasedRootsOf
  try dsimp only
  rw [hh, ht]

theorem reads_interleave (c : LiveCommitment) {h : String} {t t' : List String}
    (htree : t.filter treeLine = t'.filter treeLine)
    (hrest : t.filter (fun x => !treeLine x) = t'.filter (fun x => !treeLine x)) :
    coinTreesOf (h :: t) = coinTreesOf (h :: t') ∧ releasedRootsOf c (h :: t) = releasedRootsOf c (h :: t') := by
  refine ⟨?_, releasedRootsOf_congr c rfl hrest⟩
  rw [coinTreesOf_filter (h :: t), coinTreesOf_filter (h :: t'), List.filter_cons, List.filter_cons, htree]

end Verity.Sampled
