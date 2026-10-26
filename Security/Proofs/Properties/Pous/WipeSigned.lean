import Properties.Pous.WipeSigned
import Proofs.Properties.Pous.WipeSigned.Header
import Proofs.Properties.Pous.WipeSigned.Inputs
import Proofs.Properties.Pous.WipeSigned.Run
import Proofs.Properties.Pous.MemoryChallenger
import Mathlib.Tactic.SplitIfs

/-!
# Proof of `WipeSigned` (`Properties.Pous.WipeSigned`)

`StreamedWipe.verdict` is its checks before the batches (`wpre`) followed by the batches (`wtail`), as
`MemoryChallenger.verdict` is in `Proofs.Properties.Pous.MemoryChallenger`; an accepted verdict has no reason, so each
check passed (`accepted_of`): the signature verifies over the session's message under a trusted key, the signed
commitment is the session's, the batch size is 1, the session ended `complete` after `k` batches, and every batch has no
wrong answer, is within the deadline, and the first was sent after the fill.

If the challenger never signed the message the signature verifies over, that is a forgery. If it signed it for one of
its PoUS sessions, then under no SHA-512 collision among the run's strings the two signed inputs are one, so the two
records, so the two session inputs, whose tags differ (`WipeInputs.tags_differ`): a collision. If it signed it for one
of its wipes, the two session inputs are one, so the headers and the records roots, so the commitments, the endings, the
fill's readings, the egress (`WipeHeader.header_inj`) and (through `WipeTree.tree_bind`) the batches (`same_run`). That
run's record keeps each batch's answer and whether it is the read's label (`WipeRuns.inv_recorded`), so the batches the
verdict accepted make it the streamed wipe's accepted session (`accepted_run`).
-/

namespace MemoryChallenger.Properties.SecurityProofs

open MemoryChallenger Flock
open Tree (CF)

namespace Wipe

/-- `ByteArray`'s `==` is its arrays', which is lawful. -/
local instance instLawfulBEqByteArray_wipe : LawfulBEq ByteArray where
  eq_of_beq h := ByteArray.ext (beq_iff_eq.1 h)
  rfl {a} := show (a.data == a.data) = true from beq_iff_eq.2 rfl

deriving instance ReflBEq, LawfulBEq for StreamedWipe.Params
deriving instance ReflBEq, LawfulBEq for StreamedWipe.Commitment

/-! ## The verdict -/

/-- `StreamedWipe.verdict`'s reasons before the batches. -/
private def wpre (schemes : Schemes) (trusted : List ByteArray) (s : StreamedWipe.Signed)
    (c : StreamedWipe.Commitment) : Array String := Id.run do
  let mut reasons : Array String := #[]
  match schemes.lookup s.scheme with
  | none => reasons := reasons.push s!"signature scheme {pyRepr s.scheme} is not accepted"
  | some p =>
    if !trusted.contains s.publicKey then
      reasons := reasons.push "the session is signed by a key that is not a trusted memory challenger's"
    else
      let ok := match s.message >>= fun m => lmsVerify p s.publicKey s.commitment.session m s.signature with
        | .ok b => b
        | .error _ => false
      if !ok then reasons := reasons.push "the signature does not verify"
  if !(s.commitment == c) then reasons := reasons.push "the signed header is not the session's commitment"
  let p := c.params
  if !p.covered then
    reasons := reasons.push s!"batch size {p.batch} is uncovered: the guarantee covers the sequential reveal, batch size 1"
  if s.ended != StreamedWipe.COMPLETE || s.batches.size != p.batches then
    reasons := reasons.push s!"the session ended {s.ended} after {s.batches.size} of {p.batches} batches"
  return reasons

/-- `StreamedWipe.verdict` from the batches on, after the reasons `r0`. -/
private def wtail (s : StreamedWipe.Signed) (c : StreamedWipe.Commitment) (r0 : Array String) :
    StreamedWipe.Verdict := Id.run do
  let mut reasons := r0
  let p := c.params
  for h : b in [0:s.batches.size] do
    let r := s.batches[b]
    if r.b != (b : Int) || r.first != ((b * p.batch : Nat) : Int) || r.count != min (p.batch : Int) ((p.k : Int) - r.first) then
      reasons := reasons.push s!"batch {b} is not the session's batch {b}"
    if r.wrong != 0 then reasons := reasons.push s!"batch {b} has {r.wrong} wrong answers"
    if !(0 ≤ r.elapsedNs && r.elapsedNs ≤ (p.limitNs : Int)) then
      reasons := reasons.push s!"batch {b} is past its deadline of {p.limitNs} ns"
    if b == 0 && r.sendNs < s.fillEndNs then reasons := reasons.push "the first batch was sent before the fill completed"
  return ⟨reasons.isEmpty, reasons, StreamedWipe.egressUnits s.egressBits⟩

private theorem wverdict_eq (schemes : Schemes) (trusted : List ByteArray) (s : StreamedWipe.Signed)
    (c : StreamedWipe.Commitment) :
    StreamedWipe.verdict schemes trusted s c = wtail s c (wpre schemes trusted s c) := by
  unfold StreamedWipe.verdict wtail wpre
  cases schemes.lookup s.scheme with
  | none =>
    by_cases h1 : (!(s.commitment == c)) = true <;> by_cases h2 : (!c.params.covered) = true <;>
    by_cases h3 : (s.ended != StreamedWipe.COMPLETE || s.batches.size != c.params.batches) = true <;>
      simp only [h1, h2, h3, ite_true, ite_false, Bool.false_eq_true] <;> rfl
  | some p =>
    cases hm : (s.message >>= fun m => lmsVerify p s.publicKey s.commitment.session m s.signature) with
    | error e =>
      by_cases h0 : (!trusted.contains s.publicKey) = true <;>
      by_cases h1 : (!(s.commitment == c)) = true <;> by_cases h2 : (!c.params.covered) = true <;>
      by_cases h3 : (s.ended != StreamedWipe.COMPLETE || s.batches.size != c.params.batches) = true <;>
        simp only [hm, h0, h1, h2, h3, ite_true, ite_false, Bool.false_eq_true, Bool.not_false] <;> rfl
    | ok b =>
      by_cases h0 : (!trusted.contains s.publicKey) = true <;> cases b <;>
      by_cases h1 : (!(s.commitment == c)) = true <;> by_cases h2 : (!c.params.covered) = true <;>
      by_cases h3 : (s.ended != StreamedWipe.COMPLETE || s.batches.size != c.params.batches) = true <;>
        simp only [hm, h0, h1, h2, h3, ite_true, ite_false, Bool.false_eq_true, Bool.not_false, Bool.not_true] <;>
        rfl

/-- Batch `b` passes every check of the loop. -/
private def WGood (s : StreamedWipe.Signed) (c : StreamedWipe.Commitment) (b : Nat) : Prop :=
  ∀ hb : b < s.batches.size,
    (s.batches[b].b != (b : Int) || s.batches[b].first != ((b * c.params.batch : Nat) : Int) ||
        s.batches[b].count != min (c.params.batch : Int) ((c.params.k : Int) - s.batches[b].first)) = false ∧
      (s.batches[b].wrong != 0) = false ∧
      (!(decide (0 ≤ s.batches[b].elapsedNs) && decide (s.batches[b].elapsedNs ≤ (c.params.limitNs : Int)))) =
        false ∧
      (b == 0 && decide (s.batches[b].sendNs < s.fillEndNs)) = false

private theorem wtail_accepted (s : StreamedWipe.Signed) (c : StreamedWipe.Commitment) (r0 : Array String)
    (h : (wtail s c r0).accepted = true) : r0 = #[] ∧ ∀ b < s.batches.size, WGood s c b := by
  unfold wtail at h
  simp only [Id.run] at h
  rw [Std.Legacy.Range.forIn'_eq_forIn'_range'] at h
  generalize hL : (forIn' (m := Id) (List.range' _ _ _) (_ : Array String) _ : Id (Array String)) = L at h
  obtain ⟨m1, m2⟩ := forIn'_measure_eq hL (fun b => b.size) (WGood s c) (by
    intro a ha b
    clear h hL
    split_ifs with h1 h2 h3 h4 <;> refine ⟨_, rfl, ?_, fun he => ?_⟩ <;> (try simp at he) <;> (try simp) <;>
      (try omega)
    rename_i h2 h3 h4
    intro hb
    exact ⟨by simpa using h1, by simpa using h2, by simpa using h3, by simpa using h4⟩)
  rw [show ∀ (x : Id (Array String)) (f : Array String → Id StreamedWipe.Verdict), (x >>= f) = f x.run from
    fun _ _ => rfl] at h
  simp only [Id.run] at m1 m2 h
  have h0 : L.size = 0 := by simpa using (show Array.isEmpty L = true from h)
  exact ⟨Array.eq_empty_of_size_eq_zero (by omega), fun b hb => m2 (by omega) b (by simp; omega)⟩

private theorem wpre_empty (schemes : Schemes) (trusted : List ByteArray) (s : StreamedWipe.Signed)
    (c : StreamedWipe.Commitment) (h : wpre schemes trusted s c = #[]) :
    (∃ p, schemes.lookup s.scheme = some p ∧ s.publicKey ∈ trusted ∧
      ∃ m, s.message = .ok m ∧ lmsVerify p s.publicKey s.commitment.session m s.signature = .ok true) ∧
    (s.commitment == c) = true ∧ c.params.covered = true ∧ s.ended = StreamedWipe.COMPLETE ∧
      s.batches.size = c.params.batches := by
  unfold wpre at h
  cases hp : schemes.lookup s.scheme with
  | none =>
    simp only [hp, Id.run] at h
    split_ifs at h <;> simp [pure] at h
  | some p =>
    cases hm : (s.message >>= fun m => lmsVerify p s.publicKey s.commitment.session m s.signature) with
    | error e =>
      simp only [hp, hm, Id.run] at h
      split_ifs at h <;> simp [pure] at h <;> simp_all
    | ok b =>
      cases b
      · simp only [hp, hm, Id.run] at h
        split_ifs at h <;> simp [pure] at h <;> simp_all
      simp only [hp, hm, Id.run] at h
      split_ifs at h <;> simp [pure] at h
      refine ⟨⟨p, rfl, by simp_all, ?_⟩, by simp_all, by simp_all, by simp_all, by simp_all⟩
      cases hs : s.message with
      | error e => simp [hs, bind, Except.bind] at hm
      | ok m => exact ⟨m, rfl, by simpa [hs, bind, Except.bind] using hm⟩

/-- **What an accepted wipe's check establishes**: a signed session whose signature verifies over its message under a
trusted key, whose commitment is the session's, at batch size 1, ended `complete` after `k` batches, each with no wrong
answer and within the deadline, the first sent after the fill. -/
theorem accepted_of {p : WipeProtocol.Params} {c : StreamedWipe.Commitment} {e : WipeProtocol.Evidence}
    {out : WipeProtocol.Public} {prof : StreamedWipe.Params × Int}
    (h : WipeProtocol.protocol.check p c e = .accept out prof) :
    ∃ s, e.signed = some s ∧
      (∃ q, SCHEMES.lookup s.scheme = some q ∧ s.publicKey ∈ p.trusted.toList ∧
        ∃ m, s.message = .ok m ∧ lmsVerify q s.publicKey s.commitment.session m s.signature = .ok true) ∧
      s.commitment = c ∧ c.params.batch = 1 ∧ s.ended = StreamedWipe.COMPLETE ∧
      s.batches.size = c.params.batches ∧
      ∀ b (hb : b < s.batches.size), s.batches[b].wrong = 0 ∧ 0 ≤ s.batches[b].elapsedNs ∧
        s.batches[b].elapsedNs ≤ (c.params.limitNs : Int) ∧ (b = 0 → s.fillEndNs ≤ s.batches[b].sendNs) := by
  have h' : WipeProtocol.check SCHEMES p c e = .accept out prof := h
  unfold WipeProtocol.check at h'
  split at h'
  · cases h'
  cases hse : e.signed with
  | none => simp [hse] at h'
  | some s =>
    simp only [hse] at h'
    split at h'
    next hv =>
      rw [wverdict_eq] at hv
      obtain ⟨h0, hg⟩ := wtail_accepted s c _ hv
      obtain ⟨hk, hcm, hcov, hend, hsize⟩ := wpre_empty _ _ s c h0
      refine ⟨s, rfl, hk, eq_of_beq hcm, by simpa [StreamedWipe.Params.covered] using hcov, hend, hsize,
        fun b hb => ?_⟩
      obtain ⟨-, g2, g3, g4⟩ := hg b hb hb
      simp only [bne_eq_false_iff_eq] at g2
      simp only [Bool.not_eq_false', Bool.and_eq_true, decide_eq_true_eq] at g3
      simp only [Bool.and_eq_false_imp, beq_iff_eq, decide_eq_false_iff_not, Int.not_lt] at g4
      exact ⟨g2, g3.1, g3.2, g4⟩
    next => cases h'

/-! ## The strings the run hashes -/

theorem tree_mem {pk : ByteArray} {c : StreamedWipe.Commitment} {e : String} {fs fe : Int}
    {batches : Array StreamedWipe.Batch} {eg : Int} {vs : Array ByteArray}
    (hv : batches.mapM StreamedWipe.Batch.value = .ok vs) {x : ByteArray}
    (hx : x ∈ wipeTreeStrings (domain StreamedWipe.RECORD_BINDING c.session batches.size) vs) :
    x ∈ wipeSessionStrings pk c e fs fe batches eg := by
  simp only [wipeSessionStrings, hv, List.mem_append]
  exact Or.inl (Or.inl hx)

theorem session_mem {pk : ByteArray} {c : StreamedWipe.Commitment} {e : String} {fs fe : Int}
    {batches : Array StreamedWipe.Batch} {eg : Int} {root : ByteArray}
    (hr : StreamedWipe.recordsRoot c.session batches = .ok root) :
    StreamedWipe.SESSION_TAG ++ be64 (StreamedWipe.header c e fs fe batches.size eg).toUTF8.size.toUInt64 ++
        (StreamedWipe.header c e fs fe batches.size eg).toUTF8 ++ root ∈
      wipeSessionStrings pk c e fs fe batches eg := by
  simp [wipeSessionStrings, hr]

theorem signed_mem {pk : ByteArray} {c : StreamedWipe.Commitment} {e : String} {fs fe : Int}
    {batches : Array StreamedWipe.Batch} {eg : Int} {rec : ByteArray}
    (hm : StreamedWipe.message c e fs fe batches eg = .ok rec) (hI : (pk.extract 0 16).size = 16) :
    field MEMORY_CHALLENGER_TAG.toUTF8 ++ pk.extract 0 16 ++ field (field c.session) ++ field rec ∈
      wipeSessionStrings pk c e fs fe batches eg := by
  simp [wipeSessionStrings, hm, hI]

theorem own_mem {s : StreamedWipe.Signed} {dev : Challenger} {x : ByteArray}
    (hx : x ∈ wipeSessionStrings s.publicKey s.commitment s.ended s.fillStartNs s.fillEndNs s.batches s.egressBits) :
    x ∈ wipeRunStrings s dev :=
  List.mem_append_left _ (List.mem_append_left _ hx)

theorem wipe_mem {s : StreamedWipe.Signed} {dev : Challenger} {r : WipeRun} (hr : r ∈ dev.wipes s.publicKey)
    {e : String} (he : r.recorded.ended = some e) {x : ByteArray}
    (hx : x ∈ wipeSessionStrings s.publicKey r.commitment e r.fillStartNs r.fillEndNs r.recorded.batches r.egressBits) :
    x ∈ wipeRunStrings s dev := by
  refine List.mem_append_left _ (List.mem_append_right _ (List.mem_flatMap.mpr ⟨r, hr, ?_⟩))
  simpa only [he] using hx

theorem pous_mem {s : StreamedWipe.Signed} {dev : Challenger} {r : Run} (hr : r ∈ dev.device.sessions s.publicKey)
    {e : String} (he : r.recorded.ended = some e) {x : ByteArray}
    (hx : x ∈ sessionStrings s.publicKey r.commitment e r.recorded.rounds) : x ∈ wipeRunStrings s dev := by
  refine List.mem_append_right _ (List.mem_flatMap.mpr ⟨r, hr, ?_⟩)
  simpa only [he] using hx

/-! ## The argument -/

/-- Under no collision among the run's strings, a wipe whose message the challenger signed is one of its wipes, with
that wipe's commitment, ending, fill readings, batches and egress. -/
theorem same_run {s : StreamedWipe.Signed} {dev : Challenger} (cf : CF (wipeRunStrings s dev)) {m M : ByteArray}
    (hm : s.message = .ok m)
    (hM : memoryChallengerMessage (s.publicKey.extract 0 16) s.commitment.session m = .ok M)
    (hsig : dev.Signed s.publicKey M) :
    ∃ r ∈ dev.wipes s.publicKey, r.commitment = s.commitment ∧ r.recorded.ended = some s.ended ∧
      r.fillStartNs = s.fillStartNs ∧ r.fillEndNs = s.fillEndNs ∧ r.recorded.batches = s.batches ∧
      (r.egressBits : Int) = s.egressBits := by
  have hm' : StreamedWipe.message s.commitment s.ended s.fillStartNs s.fillEndNs s.batches s.egressBits = .ok m := hm
  obtain ⟨root, hroot, rfl⟩ := WipeInputs.message_ok hm'
  obtain ⟨vs, hvs, rfl⟩ := WipeInputs.recordsRoot_ok hroot
  obtain ⟨hI, rfl⟩ := Inputs.signedMessage_ok hM
  rcases hsig with ⟨r, hr, ended, rec, hend, hmsg, hmcm⟩ | ⟨r, hr, ended, rec, hend, hmsg, hmcm⟩
  · exfalso
    obtain ⟨root', hroot', rfl⟩ := Inputs.message_ok hmsg
    obtain ⟨-, hMM⟩ := Inputs.signedMessage_ok hmcm
    have e1 := cf _ (own_mem (signed_mem hm' hI)) _ (pous_mem hr hend (Device.signed_mem hmsg hI)) hMM
    have e2 := Inputs.signed_split (by rw [Tree.hash_size, Tree.hash_size]) e1
    have e3 := cf _ (own_mem (session_mem hroot)) _ (pous_mem hr hend (Device.session_mem hroot')) e2
    exact WipeInputs.tags_differ _ _ _ _ _ _ e3.symm
  · have hmsg' : StreamedWipe.message r.commitment ended r.fillStartNs r.fillEndNs r.recorded.batches r.egressBits =
        .ok rec := hmsg
    obtain ⟨root', hroot', rfl⟩ := WipeInputs.message_ok hmsg'
    obtain ⟨vs', hvs', rfl⟩ := WipeInputs.recordsRoot_ok hroot'
    obtain ⟨-, hMM⟩ := Inputs.signedMessage_ok hmcm
    have e1 := cf _ (own_mem (signed_mem hm' hI)) _ (wipe_mem hr hend (signed_mem hmsg' hI)) hMM
    have e2 := Inputs.signed_split (by rw [Tree.hash_size, Tree.hash_size]) e1
    have e3 := cf _ (own_mem (session_mem hroot)) _ (wipe_mem hr hend (session_mem hroot')) e2
    obtain ⟨e4, e5⟩ := WipeInputs.session_split (by rw [WipeTree.root_width, WipeTree.root_width]) e3
    obtain ⟨hc, hen, hfs, hfe, hn, heg⟩ := WipeHeader.header_inj (String.toByteArray_inj.mp e4)
    rw [hc, hn] at e5
    have hsz : vs.size = vs'.size := by rw [WipeInputs.mapM_size hvs, WipeInputs.mapM_size hvs', hn]
    have hvv := WipeTree.tree_bind hsz (fun x hx => own_mem (tree_mem hvs (by rwa [hc, hn])))
      (fun x hx => wipe_mem hr hend (tree_mem hvs' hx)) cf e5
    subst hvv
    exact ⟨r, hr, hc.symm, by rw [hend, hen], hfs.symm, hfe.symm, (WipeInputs.batches_inj hvs hvs').symm, heg.symm⟩

/-- **The run is the streamed wipe's accepted session**: a run the challenger ended `complete`, whose batches the verdict
accepted at batch size 1. -/
theorem accepted_run {r : WipeRun} {s : StreamedWipe.Signed} (hend : r.recorded.ended = some StreamedWipe.COMPLETE)
    (hb : r.recorded.batches = s.batches) (hfe : r.fillEndNs = s.fillEndNs) (hb1 : r.commitment.params.batch = 1)
    (hsize : s.batches.size = r.commitment.params.batches)
    (hg : ∀ b (hb : b < s.batches.size), s.batches[b].wrong = 0 ∧ 0 ≤ s.batches[b].elapsedNs ∧
      s.batches[b].elapsedNs ≤ (r.commitment.params.limitNs : Int) ∧ (b = 0 → s.fillEndNs ≤ s.batches[b].sendNs)) :
    r.Accepted := by
  obtain ⟨i1, i2, i3⟩ := WipeRuns.inv_recorded r
  have hk : r.recorded.batches.size = r.commitment.params.k := by
    rw [hb, hsize]
    unfold StreamedWipe.Params.batches
    rw [hb1]
    simp
  have hg' : ∀ j (hj : j < r.commitment.params.k), r.recorded.batches[j]!.wrong = 0 ∧
      0 ≤ r.recorded.batches[j]!.elapsedNs ∧ r.recorded.batches[j]!.elapsedNs ≤ (r.commitment.params.limitNs : Int) ∧
      (j = 0 → r.fillEndNs ≤ r.recorded.batches[j]!.sendNs) := by
    intro j hj
    have hjs : j < s.batches.size := by rw [← hb, hk]; exact hj
    rw [hb, getElem!_pos s.batches j hjs, hfe]
    exact hg j hjs
  have hans : ∀ j : Fin r.commitment.params.k, r.recorded.answers[j.val]! = wipeLabel r.key (r.reads j) := by
    intro j
    have hw := i2 j.val j.isLt (by rw [hk]; exact j.isLt)
    rw [(hg' j.val j.isLt).1] at hw
    by_cases ha : r.recorded.answers[j.val]! = wipeLabel r.key (r.reads ⟨j.val, j.isLt⟩)
    · exact ha
    · simp [ha] at hw
  have h0 : 0 < r.commitment.params.k := by
    rw [← hk]
    exact i3 (by rw [hend]; exact Option.some_ne_none _)
  refine ⟨hend, hk, by rw [i1, hk], fun j => ⟨hans j, (hg' j.val j.isLt).2.1, (hg' j.val j.isLt).2.2.1⟩,
    (hg' 0 h0).2.2.2 rfl, ?_⟩
  intro q M E P z hp
  unfold Pous.StreamedWipe.Accepts
  intro j
  rw [hp j, hans j]
  rfl

end Wipe

theorem wipeSigned : WipeSigned := by
  intro p c e out prof dev h
  obtain ⟨s, hse, ⟨q, hq, ht, m, hm, hv⟩, hsc, hb1, hend, hsize, hg⟩ := Wipe.accepted_of h
  refine ⟨s, hse, ht, ?_⟩
  unfold lmsVerify at hv
  split at hv
  · simp [pure, Except.pure] at hv
  cases hd : Lms.Signature.decode q s.signature with
  | error _ => simp [hd, bind, Except.bind] at hv
  | ok σ =>
    cases hM : memoryChallengerMessage (s.publicKey.extract 0 16) s.commitment.session m with
    | error _ => simp [hd, hM, bind, Except.bind] at hv
    | ok M =>
      simp only [hd, hM, bind, Except.bind] at hv
      by_cases hcol : ∃ x ∈ wipeRunStrings s dev, ∃ y ∈ wipeRunStrings s dev, x ≠ y ∧ Sha512.hash x = Sha512.hash y
      · exact Or.inr (Or.inl hcol)
      have cf : CF (wipeRunStrings s dev) := fun x hx y hy hxy =>
        Classical.byContradiction fun hne => hcol ⟨x, hx, y, hy, hne, hxy⟩
      by_cases hsig : dev.Signed s.publicKey M
      · obtain ⟨r, hr, hc, hen, hfs, hfe, hb, heg⟩ := Wipe.same_run cf hm hM hsig
        have hrc : r.commitment = c := hc.trans hsc
        refine Or.inl ⟨r, hr, hrc, hfs, hfe, hb, heg, ?_⟩
        subst hrc
        exact Wipe.accepted_run (by rw [hen, hend]) hb hfe hb1 hsize hg
      · refine Or.inr (Or.inr ⟨q, M, hq, ht, ?_, ⟨σ, hd, hv⟩, hsig⟩)
        simp [wipeMessage, hm, hM, bind, Except.bind, Except.toOption]

end MemoryChallenger.Properties.SecurityProofs
