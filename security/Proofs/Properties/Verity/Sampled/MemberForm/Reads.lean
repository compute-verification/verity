module

public import Proofs.Properties.Verity.Sampled.MemberForm.Sent
public import Proofs.Properties.Verity.Sampled.MemberForm.Registered
public import Proofs.Properties.Verity.Sampled.MemberForm.Units
public import Proofs.Properties.Verity.Sampled.MemberForm.Links
public import Proofs.Properties.Verity.Sampled.MemberForm.Files
public import Proofs.Properties.Verity.Sampled.Assembled
public import Proofs.Properties.Verity.Sampled.SampleNamed
public import Proofs.Properties.Verity.Sampled.Held

/-!
# What the verifier's acceptance gives of a play of the per-member form, at the window

`WindowRead` is what the window's theorem takes of an accepted play: every release of the sample's family precedes
every opening, and the play has the sessions of the window's sample (`WindowSessions`). This file proves it of an
accept of the per-member form (`windowRead_members`), at the evidence the verifier reads for each session on the
blobs sent (`sentEvidenceAt`) and the proof files decoded from it at the statement's table
(`statementFiles statementTable`).

The sample's family's releases precede every opening (`accept_read`'s `SampleLinkFirst`). At coins whose bytes are
the window's, the window's sample ends on them, so its sample ended (`joint_drawUnits_of_drew`), and the sample is the
units v1's check proved (`joint_draw_of_drew`, `checkAt_proved`, at the registration's population,
`WindowOn.population`), so each member with a proof block in it has a statement with one
session (`member_session`), which the check names at its place (`named_at`), whose record and evidence are the ones
sent for that place (`sentRecordAt_session`, `sentEvidenceAt_session`), which the verifier's read holds the device's
log to (`sessionRead_logged`), and whose proof files are the evidence's (`statementFiles_evidence`). The log opens
as many trees as statements (`members_place_logged`), each tree's streams named once and each opening of a tree
opened before the last tree's (`Accepted.specs`, `Accepted.opensLast`).

It takes two statements besides the play: `FormAccepts`, which holds of the check's accepts (`formAccepts`), and
`RegisteredHolds`, the registration's.
-/

@[expose] public section

namespace Verity.Sampled.MemberForm

open Verity.Game
open Lean (Json)
open Properties.Challenger (bitsOf Draws)

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **What the verifier's acceptance gives of a play of the per-member form, at the window the verifier sets up at
its parameters** (`WindowRead`): where the form's accepts hold `MembersAccept` and the registered files hold each
member's file, at a play the sampled verdict accepts, whose log v2 read is the device's, whose blobs sent hold each
pinned table and each member's file at its registered file, and whose sample answers `after`, every release of
the parameters' family answering `after` precedes every opening, and the play has the sessions of the window's
sample, each with the evidence sent for it and the proof files decoded from that evidence. -/
theorem windowRead_members {V : Vocab d α} {k : Verity.Registration.Composite} (hF : FormAccepts)
    (hR : RegisteredHolds V) {F : Array ByteArray} {W : Window} {x : Verity.Exchange}
    (hset : MembersSetUp V x.params F W) (hv : V.Verdict k .sampled x)
    (hl : Binding.Logged Binding.readsV2 x) (ht : TablesSent x.sent) (hfs : FilesSent V x) {after : ByteArray}
    (hsd : SampledDigest k x.sent after) :
    WindowRead W W.table sentEvidenceAt (statementFiles statementTable) (labelAt x.params) after
      _root_.Spot.Challenger.MAX_COINS x := by
  obtain ⟨p, ms, vms, hp, hms, hperm, hsorted, hall, hlaw, -⟩ := hset.members
  obtain ⟨a, b, e, c, ev, ξ, dr, sts, r, hcase, hacc, hM⟩ := accept_members hF hp hms hv
  obtain ⟨lines, lr, trees, held, named, hcs, hlogged, hlr, hct, hreason, hheld, hnamed, hslf, hlogRead⟩ :=
    accept_read hv.2.1 hcase hacc hM hl
  have hlab : dr.label = labelAt x.params := draw_label hv.2.1 hcase hacc hp
  have haft : dr.after = after := hsd a b e p c ev ξ dr hcase hacc
  rw [hlab, haft] at hslf
  have hfr := ofV2_frames hlogged
  refine ⟨fun pre r' post he i s n hr => (hslf pre r' post he i s n hr).1, fun κ ω hsrc hc => ?_⟩
  -- the window's sample at `ω` is the units v1's check proved
  obtain ⟨pub, prof, hv1⟩ := hM.v1
  obtain ⟨design, n, read, hder0, hn, hdf⟩ := checkAt_drew hv1
  obtain ⟨derived, fl, hder, hfl, hdu⟩ := drawnFrom_ok hdf
  have hof : LawOf p c (derived.getD fl) n := ⟨design, derived, fl, hder0, hn, hder, hfl, rfl⟩
  have hdrew : Drew k x.sent dr (derived.getD fl) n ξ (drawnUnits ev.base.draw) :=
    ⟨a, b, e, p, c, ev, hcase, hacc, hof, rfl, read, hdu⟩
  have hnJ : W.nJ = n :=
    hset.toWindowOn.population hset.passes hp hms (membersStructure_laid (membersRun_ok hM.run hM.all).1) hn
  have hL : W.law = derived.getD fl := by
    have h1 := lawAt_of_lawOf hof
    rw [← hnJ] at h1
    exact Except.ok.inj (hlaw.symm.trans h1)
  have hs := shows_of_read hv.2.1 hcase hacc hlogRead hl
  unfold ShowsDraw at hs
  rw [hlab, haft] at hs
  have hξ := coins_size_le hacc
  have e₀ : bitsOf (_root_.Properties.Spot.bytesOf (coinBytes _root_.Spot.Challenger.MAX_COINS 64 κ 0)) =
      (fillCoins κ).flatten := bitsOf_streamOf_coinBytes (B := 64) κ
  have hpre : bitsOf ξ <+: bitsOf (_root_.Properties.Spot.bytesOf (W.src ω 0)) := by
    rw [hsrc, e₀]
    rcases hc with hc | hc
    · exact hs.trans (prefix_flatten hc)
    · refine List.prefix_of_prefix_length_le hs (prefix_flatten hc) ?_
      have h64 : _root_.Spot.Challenger.COIN_BYTES = 64 := rfl
      rw [h64] at hξ
      rw [length_bitsOf, length_fillCoins]
      omega
  have hpr : drawnUnits ev.base.draw = ev.base.provedUnits :=
    checkAt_proved (e := { ev.base with verdicts := r.out }) hv1 (evidenceOf_closure hacc.2.2.1)
  have hd := joint_draw_of_drew hdrew (W.joint 0) ω hL hnJ hpre
  have hS : ∀ v : Fin W.nJ, v ∈ W.L.draw ω ↔ v.val ∈ ev.base.provedUnits := fun v => by
    have hmem := Finset.ext_iff.1 hd v
    rw [← hpr]
    exact hmem.trans (Finset.mem_filter.trans ⟨fun h => h.2, fun h => ⟨Finset.mem_univ _, h⟩⟩)
  -- each member the sample holds a proof block of has its statement, with one session
  obtain ⟨roots, held', hroots, hheld', hcount, hsess⟩ :=
    member_session hset.laid hperm hsorted hall hM.run hM.all hS
  obtain rfl : held = held' := Except.ok.inj (hheld.symm.trans hheld')
  have hzip := membersHeld_zip hheld
  have hA : Accepted lr := ⟨dr.label, c.digest, _, ξ, lines, trees, named, hct, hlr, hreason⟩
  have hplace := members_place_logged hct hlr hreason hlogged
  have hsize : named.size = sts.size := by
    have h1 := hnamed.length_eq
    rw [Array.length_toList, Array.length_toList, Array.size_zip, ← hzip.1, Nat.min_self] at h1
    exact h1.symm
  have hMA := hset.toWindowOn.membersAt
  obtain ⟨read', hok⟩ := joint_drawUnits_of_drew hdrew (W.joint 0) ω hL hnJ hpre
  have hok' : _root_.Spot.Draw.drawUnits W.law W.nJ (_root_.Properties.Spot.bytesOf (W.src ω 0)) =
      .ok (drawnUnits ev.base.draw, read') := hok
  have hended : Flock.Guarantees.sampleEnded W.src W.law W.nJ ω = true := by
    simp only [Flock.Guarantees.sampleEnded, hok']
  refine ⟨hended, ?_, opensLast_congr hfr hA.opensLast, Nat.le_of_eq (by rw [hplace.1, hcount, hsize]), ?_⟩
  · rw [← helloSpecs_congr hfr]
    exact hA.specs
  intro i m hi hne
  obtain ⟨vm, st, a₀, s, keys, record, spec, tabs, rest, proofs, units, hon, hzj, hsa, hss, hsv, hrd⟩ :=
    hsess i m hi hne
  have hmem : m ∈ W.ms := List.mem_of_getElem? hi
  have hvm : vm ∈ ms := Array.mem_of_find?_eq_some (hzip.2 _ (List.mem_of_getElem? hzj)).2.1
  have hst : StatesMember (stepAt a₀) m := statesMember ht hon (readsFiles_of_sent hR hfs hp hms hvm) hsa
  -- the session `--zk` sets up at the member: its one table, the window's, on that table's coin tree
  obtain ⟨stmts, hsetup, -⟩ := hsv.accepts_member hst
  obtain ⟨htab, -, -, -, h512, -⟩ := m.spec_at (hMA.count m hmem) (hMA.rows m hmem) hsetup
  have htb : spec.tables = #[W.table] := by rw [htab, hMA.table m hmem]
  have hctree : spec.coinTree = some (Flock.Zk.coinSpec #[W.table]) := by
    rw [m.spec_coinTree (hMA.count m hmem) hsetup, m.coinSpec_one (hMA.count m hmem), hMA.table m hmem]
  obtain ⟨sess, hdec⟩ := zkVerify_decode hsv.verify
  -- the session at its place among the sessions the check names
  obtain ⟨hsj, xn, hxj, blob, record', evidence, hb, hu, hevd, rfl⟩ := named_at hzip.1 hnamed hzj hss
  obtain ⟨B, hB, hBu⟩ := hsv.blob
  obtain rfl : B = blob := Except.ok.inj (hB.symm.trans hb)
  obtain rfl : record = record' := Option.some.inj (hBu.symm.trans hu)
  have hevs : ev.sessions[drawnBefore W.ms (W.L.draw ω) i]? = some s := by
    rw [hM.sessions]
    exact hsj
  obtain ⟨n?, hroot⟩ :=
    rootBTables_one hon.tags vm.own (_root_.Core.Blobs.ofArray x.sent.blobs) record
  have htabx : _root_.Spot.OneStageZk.rootBTables vm.own (_root_.Core.Blobs.ofArray x.sent.blobs) record =
      #[(W.table, n?)] := by rw [hroot, hMA.table m hmem]
  obtain ⟨J, -, -, -, hpj, -⟩ := decode_coinTree hdec hctree
  have hJ : Flock.JsonParse.parse record =
      .ok ((Flock.JsonParse.parse record).toOption.getD Json.null) := by
    rw [hpj]
    rfl
  have hev : _root_.Spot.sha512Hex evidence = _root_.Spot.sha512Hex ⟨evidence.data.toList.toArray⟩ := by
    rw [bytes_eta]
  have hread : SessionRead x.log #[W.table] (drawnBefore W.ms (W.L.draw ω) i) record
      ((Flock.JsonParse.parse record).toOption.getD Json.null) sess evidence.data.toList :=
    sessionRead_logged hct hlr hreason rfl (linkFirst_members hct hlr hreason) hlogged hxj htabx hJ hdec hctree
      htb h512 hev
  have hj : drawnBefore W.ms (W.L.draw ω) i < named.size := by
    have h1 := (List.getElem?_eq_some_iff.1 hzj).1
    rw [hnamed.length_eq, Array.length_toList] at h1
    exact h1
  obtain ⟨τ, hτ⟩ := hplace.2 _ hj
  have hstt : statementTable = W.table := (statementTable_eq hon.tags).trans (hMA.table m hmem)
  refine ⟨stepAt a₀, _root_.Core.Blobs.ofArray x.sent.blobs, keys, s, record, spec, tabs, rest, proofs, units,
    (Flock.JsonParse.parse record).toOption.getD Json.null, sess, evidence.data.toList, τ, hst, hsv, hrd,
    sentRecordAt_session hcs hevs hB hBu, hdec, hread, hτ, sentEvidenceAt_session hcs hevs hevd, ?_⟩
  rw [hstt]
  exact statementFiles_evidence hsv.files hevd htb

end Verity.Sampled.MemberForm
