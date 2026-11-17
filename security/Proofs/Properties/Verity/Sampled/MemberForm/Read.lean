module

public import Proofs.Properties.Verity.Sampled.MemberForm.Named
public import Proofs.Properties.Verity.Sampled.MemberForm.Order
public import Proofs.Properties.Verity.Sampled.LinkFirst

/-! v2's read at an accept of the per-member form. -/

@[expose] public section

namespace Verity.Sampled.MemberForm

open Lean (Json)
open Spot Spot.Challenger
open FlockSoundness.Refine (ebind_ok)
open Properties.Challenger (Draws)

theorem accept_read {k : Verity.Registration.Composite} {x : Verity.Exchange} {a b e : ByteArray}
    {p : _root_.Spot.OneStageZk.Params} {c : _root_.Spot.OneStage.Commitment}
    {ev : _root_.Spot.OneStageZk.Evidence} {ξ : ByteArray} {dr : Draws} (hS : Verity.SameAccepts k x.sent)
    (hcase : V2Case k x.sent a b e)
    (hacc : V2Accepts (_root_.Core.Blobs.ofArray x.sent.blobs) a b e p c ev ξ dr)
    {ms : Array _root_.Spot.OneStageZk.Member} {sts : Array _root_.Spot.OneStageZk.StatementFiles}
    {r : _root_.Spot.OneStageZk.Run}
    (hM : MembersAccept p c ξ ev (_root_.Core.Blobs.ofArray x.sent.blobs) ms sts r)
    (hl : Binding.Logged Binding.readsV2 x) :
    ∃ (lines : List String) (lr : List Verity.Release) (trees : Array LoggedTree)
      (held : Array _root_.Spot.OneStageZk.Member)
      (named : Array (String × Array (String × Option ℕ) × Json × String)),
      caseOf x.sent = .ok (p, c, ev) ∧ List.Forall₂ Binding.Read.Of (lr.map Binding.Read.ofV2) x.log ∧
      readLog dr.label c.digest (named.toList.map (·.2.2.1)) ξ lines = some lr ∧
      coinTreesOf lines = .ok trees ∧ membersTreeReason trees lines named = none ∧
      _root_.Spot.OneStageZk.membersHeld ms sts = .ok held ∧
      List.Forall₂ (NamedAt (_root_.Core.Blobs.ofArray x.sent.blobs)) (sts.zip held).toList named.toList ∧
      SampleLinkFirst dr.label dr.after x.log ∧ logRead x.sent = some lr := by
  obtain ⟨hp, hc, hev, hξ, hdr, -⟩ := hacc
  obtain ⟨key, hch, hafter⟩ := source_of hdr
  obtain ⟨raw, hraw, hrξ⟩ := Properties.Challenger.oneStageZk_reads p c dr hdr ev ξ hξ
  rw [hrξ] at hraw
  obtain ⟨cm, roots, htsn, -⟩ := drawCoins_ok hraw
  obtain ⟨_, hsn⟩ := trustedChallengePhaseOf_ok htsn
  obtain ⟨_, hso⟩ := challengePhaseNamed_ok hsn
  obtain ⟨ls, lines, hlog, hls, hrr⟩ := challengePhaseOf_ok hso
  have hL := linesOf_of hlog hls
  obtain ⟨hstruct, -⟩ := membersRun_ok hM.run hM.all
  obtain ⟨held, ts, lines₁, hheld, -, htr, hord, hL₁, hth⟩ := membersStructure_ok hstruct
  rw [linesOf_check, hL] at hL₁
  obtain rfl : lines = lines₁ := Except.ok.inj hL₁
  obtain ⟨key', w, htt⟩ := treesOf_ok htr
  obtain ⟨ls₂, lines₂, hlog₂, hls₂, hct⟩ := trustedTreesOf_ok htt
  have hL₂ := linesOf_of hlog₂ hls₂
  rw [hL] at hL₂
  obtain rfl : lines = lines₂ := Except.ok.inj hL₂
  obtain ⟨named, hreason, hnamed⟩ := membersTreesHeld_ok hth
  obtain ⟨hsz, hzip⟩ := membersHeld_zip hheld
  have hone := named_one (fun z hz => (hzip z hz).2.2) hnamed
  have hfst : (sts.zip held).toList.map (·.1) = sts.toList := by
    rw [Array.toList_zip]
    exact List.map_fst_zip (by rw [Array.length_toList, Array.length_toList]; exact Nat.le_of_eq hsz)
  have hrs : recordsOn (_root_.Core.Blobs.ofArray x.sent.blobs) ev = some (named.toList.map (·.2.2.1)) := by
    unfold recordsOn
    rw [hM.sessions, Array.toList_flatMap, ← hfst]
    exact records_named hone
  have hcs := caseOf_of hS hcase hp hc hev
  have hread : logRead x.sent = readLog dr.label c.digest (named.toList.map (·.2.2.1)) ξ lines := by
    rw [logRead_eq hcs hch hξ hL, hrs, Option.bind_some]
  have hF : (treeFold (named.toList.map (·.2.2.1)) [] lines).isSome := by
    rw [coinTreesOf_fold _ hct]
    rfl
  obtain ⟨lr, hlr⟩ := Option.isSome_iff_exists.1
    (readLog_isSome dr.label c.digest (named.toList.map (·.2.2.1)) ξ hF (releasedRootsOf_pairsOk hrr))
  have hlogged := hl (lr.map Binding.Read.ofV2) (by
    show lr.map Binding.Read.ofV2 ∈ (logRead x.sent).map (·.map Binding.Read.ofV2)
    rw [hread, hlr]
    exact Option.mem_def.2 rfl)
  refine ⟨lines, lr, ts, held, named, hcs, hlogged, hlr, hct, hreason, hheld, hone, ?_, hread.trans hlr⟩
  rw [hafter]
  exact sampleLinkFirst_congr (ofV2_frames hlogged) (sampleLinkFirst_readLog (orderHeld_ok hord hch hL) hlr)

end Verity.Sampled.MemberForm
