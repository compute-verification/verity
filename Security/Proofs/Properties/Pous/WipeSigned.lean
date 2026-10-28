import Properties.Pous.WipeSigned
import Proofs.Properties.Pous.WipeSigned.Header
import Proofs.Properties.Pous.WipeSigned.Inputs
import Proofs.Properties.Pous.WipeSigned.Run
import Proofs.Properties.Pous.MemoryChallenger
import Proofs.Properties.CertifierDevice.Lms
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
verdict accepted make it the streamed wipe's accepted session (`accepted_run`). A collision among the run's strings is
one the collision finder outputs (`collisionIn_finds`); a forgery is the forger's output.

`wipeSigned_honest` is its honest lemma: at the deployed LMS parameters a key `Lms.keygen` derives signs the honest
session at r20261026-094403-bb5f's parameters and clock readings, and the compiled check accepts it
(`measured_accepts`, by LMS's correctness). `wipeSigned_nontrivial` is its nontrivial lemma: a session with a batch no
record encodes, signed under the empty key, is no run of the idle challengers, and both adversaries output nothing
there, since the run hashes nothing and the signature signs nothing; so a check that accepts it, as a verifier that
accepts everything does (`wipeSigned_acceptAll`), does not have `WipeSigned`'s conclusion.
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

theorem own_mem {s : StreamedWipe.Signed} {dev : SignedWipeParty} {x : ByteArray}
    (hx : x ∈ wipeSessionStrings s.publicKey s.commitment s.ended s.fillStartNs s.fillEndNs s.batches s.egressBits) :
    x ∈ wipeRunStrings s dev :=
  List.mem_append_left _ (List.mem_append_left _ hx)

theorem wipe_mem {s : StreamedWipe.Signed} {dev : SignedWipeParty} {r : WipeRun} (hr : r ∈ dev.wipes s.publicKey)
    {e : String} (he : r.recorded.ended = some e) {x : ByteArray}
    (hx : x ∈ wipeSessionStrings s.publicKey r.commitment e r.fillStartNs r.fillEndNs r.recorded.batches r.egressBits) :
    x ∈ wipeRunStrings s dev := by
  refine List.mem_append_left _ (List.mem_append_right _ (List.mem_flatMap.mpr ⟨r, hr, ?_⟩))
  simpa only [he] using hx

theorem pous_mem {s : StreamedWipe.Signed} {dev : SignedWipeParty} {r : Run} (hr : r ∈ dev.device.sessions s.publicKey)
    {e : String} (he : r.recorded.ended = some e) {x : ByteArray}
    (hx : x ∈ sessionStrings s.publicKey r.commitment e r.recorded.rounds) : x ∈ wipeRunStrings s dev := by
  refine List.mem_append_right _ (List.mem_flatMap.mpr ⟨r, hr, ?_⟩)
  simpa only [he] using hx

/-! ## The argument -/

/-- Under no collision among the run's strings, a wipe whose message the challenger signed is one of its wipes, with
that wipe's commitment, ending, fill readings, batches and egress. -/
theorem same_run {s : StreamedWipe.Signed} {dev : SignedWipeParty} (cf : CF (wipeRunStrings s dev)) {m M : ByteArray}
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

/-! ## The collision finder -/

/-- Where two distinct strings of `L` share a digest, the collision finder outputs two such strings. -/
theorem collisionIn_finds {H : ByteArray → ByteArray} {L : List ByteArray}
    (h : ∃ x ∈ L, ∃ y ∈ L, x ≠ y ∧ H x = H y) :
    ∃ x y, collisionIn H L = some (x, y) ∧ x ≠ y ∧ H x = H y := by
  obtain ⟨x, hx, y, hy, hne, heq⟩ := h
  let hashed := L.map fun x => (x, H x)
  let p : ByteArray × ByteArray → ByteArray × ByteArray → Bool := fun a b => decide (a.1 ≠ b.1 ∧ a.2 = b.2)
  let f : ByteArray × ByteArray → Option (ByteArray × ByteArray) :=
    fun a => (hashed.find? (p a)).map fun b => (a.1, b.1)
  have hxm : (x, H x) ∈ hashed := List.mem_map.2 ⟨x, hx, rfl⟩
  have hym : (y, H y) ∈ hashed := List.mem_map.2 ⟨y, hy, rfl⟩
  have hfx : f (x, H x) ≠ none := by
    intro hn
    have hfind : hashed.find? (p (x, H x)) = none := by simpa [f] using hn
    have := List.find?_eq_none.1 hfind (y, H y) hym
    simp [p, hne, heq] at this
  have hsome : hashed.findSome? f ≠ none := by
    intro hn
    exact hfx (List.findSome?_eq_none_iff.1 hn _ hxm)
  obtain ⟨⟨u, v⟩, huv⟩ := Option.ne_none_iff_exists'.1 hsome
  obtain ⟨a, ha, hfa⟩ := List.exists_of_findSome?_eq_some huv
  obtain ⟨b, hb, hab⟩ := Option.map_eq_some_iff.1 hfa
  have hpb := List.find?_some hb
  have hbm := List.mem_of_find?_eq_some hb
  obtain ⟨a', -, rfl⟩ := List.mem_map.1 ha
  obtain ⟨b', -, rfl⟩ := List.mem_map.1 hbm
  simp only [p, decide_eq_true_eq] at hpb
  simp only [Prod.mk.injEq] at hab
  obtain ⟨rfl, rfl⟩ := hab
  exact ⟨a', b', huv, hpb.1, hpb.2⟩

/-! ## The honest wipe as measured -/

private theorem forIn'_keep {α σ : Type} (l : List α) (f : (a : α) → a ∈ l → σ → Id (ForInStep σ))
    (hf : ∀ a h b, f a h b = pure (.yield b)) (init : σ) : forIn' l init f = pure init := by
  induction l generalizing init with
  | nil => rfl
  | cons a as ih =>
    rw [List.forIn'_cons, hf a List.mem_cons_self init]
    exact ih (fun a' m b => f a' (List.mem_cons_of_mem _ m) b) (fun a' m b => hf a' _ b) init

/-- `wtail` keeps its reasons when every batch passes. -/
private theorem wtail_keep (s : StreamedWipe.Signed) (c : StreamedWipe.Commitment) (r0 : Array String)
    (hg : ∀ b < s.batches.size, WGood s c b) :
    wtail s c r0 = ⟨r0.isEmpty, r0, StreamedWipe.egressUnits s.egressBits⟩ := by
  unfold wtail
  simp only [Id.run]
  rw [Std.Legacy.Range.forIn'_eq_forIn'_range', forIn'_keep]
  · rfl
  · intro b hb r
    have hb' : b < s.batches.size := by simp at hb; omega
    obtain ⟨g1, g2, g3, g4⟩ := hg b hb' hb'
    simp only [g1, g2, g3, g4, Bool.false_eq_true, ↓reduceIte]

private theorem wgood_of (s : StreamedWipe.Signed) (c : StreamedWipe.Commitment) (b : Nat)
    (h : ∀ hb : b < s.batches.size,
      s.batches[b].b = (b : Int) ∧ s.batches[b].first = ((b * c.params.batch : Nat) : Int) ∧
        s.batches[b].count = min (c.params.batch : Int) ((c.params.k : Int) - s.batches[b].first) ∧
        s.batches[b].wrong = 0 ∧ 0 ≤ s.batches[b].elapsedNs ∧ s.batches[b].elapsedNs ≤ (c.params.limitNs : Int) ∧
        (b = 0 → s.fillEndNs ≤ s.batches[b].sendNs)) : WGood s c b := by
  intro hb
  obtain ⟨h1, h2, h3, h4, h5, h6, h7⟩ := h hb
  refine ⟨by simp [h1, h2, h3], by simp [h4], by simp [h5, h6], ?_⟩
  by_cases h0 : b = 0
  · subst h0
    simp [Int.not_lt.2 (h7 rfl)]
  · simp [h0]

private theorem wpre_nil (schemes : Schemes) (trusted : List ByteArray) (s : StreamedWipe.Signed)
    (c : StreamedWipe.Commitment)
    (hk : ∃ p, schemes.lookup s.scheme = some p ∧ s.publicKey ∈ trusted ∧
      ∃ m, s.message = .ok m ∧ lmsVerify p s.publicKey s.commitment.session m s.signature = .ok true)
    (hcm : s.commitment = c) (hcov : c.params.batch = 1) (hend : s.ended = StreamedWipe.COMPLETE)
    (hsize : s.batches.size = c.params.batches) : wpre schemes trusted s c = #[] := by
  obtain ⟨p, hp, ht, m, hm, hv⟩ := hk
  have hok : (s.message >>= fun m => lmsVerify p s.publicKey s.commitment.session m s.signature) = .ok true := by
    rw [hm]
    exact hv
  subst hcm
  unfold wpre
  simp [Id.run, hp, hok, StreamedWipe.Params.covered, hcov, hend, hsize]
  exact ite_eq_left ht

/-- The compiled check accepts a signed session that passes each of the verdict's checks, on the period's server and
session. -/
private theorem check_accepts (p : WipeProtocol.Params) (c : StreamedWipe.Commitment) (s : StreamedWipe.Signed)
    (hsv : c.server = p.server) (hss : c.session = p.session)
    (hk : ∃ q, SCHEMES.lookup s.scheme = some q ∧ s.publicKey ∈ p.trusted.toList ∧
      ∃ m, s.message = .ok m ∧ lmsVerify q s.publicKey s.commitment.session m s.signature = .ok true)
    (hcm : s.commitment = c) (hcov : c.params.batch = 1) (hend : s.ended = StreamedWipe.COMPLETE)
    (hsize : s.batches.size = c.params.batches)
    (hb : ∀ b, ∀ hb : b < s.batches.size,
      s.batches[b].b = (b : Int) ∧ s.batches[b].first = ((b * c.params.batch : Nat) : Int) ∧
        s.batches[b].count = min (c.params.batch : Int) ((c.params.k : Int) - s.batches[b].first) ∧
        s.batches[b].wrong = 0 ∧ 0 ≤ s.batches[b].elapsedNs ∧ s.batches[b].elapsedNs ≤ (c.params.limitNs : Int) ∧
        (b = 0 → s.fillEndNs ≤ s.batches[b].sendNs)) :
    WipeProtocol.protocol.check p c ⟨some s⟩ =
      .accept (WipeProtocol.certify c.params (StreamedWipe.egressUnits s.egressBits))
        (c.params, StreamedWipe.egressUnits s.egressBits) := by
  have hv : StreamedWipe.verdict SCHEMES p.trusted.toList s c =
      ⟨true, #[], StreamedWipe.egressUnits s.egressBits⟩ := by
    rw [wverdict_eq, wpre_nil _ _ s c hk hcm hcov hend hsize,
      wtail_keep s c _ fun b _ => wgood_of s c b (hb b)] <;> rfl
  show WipeProtocol.check SCHEMES p c ⟨some s⟩ = _
  unfold WipeProtocol.check
  simp [hsv, hss, hv]
  rfl

private theorem beInt_of {n : Nat} {x : Int} (w : String) (h0 : 0 ≤ x) (hx : x.toNat < 2 ^ (8 * n)) :
    ∃ b, StreamedWipe.beInt n x w = .ok b := by
  unfold StreamedWipe.beInt beN
  simp only [Int.not_lt.2 h0, ↓reduceIte, hx]
  exact ⟨_, rfl⟩

private theorem mapM_ok {α β : Type} {f : α → Except String β} :
    ∀ {l : List α}, (∀ x ∈ l, ∃ v, f x = .ok v) → ∃ vs, l.mapM f = .ok vs
  | [], _ => ⟨[], rfl⟩
  | a :: l, h => by
    obtain ⟨v, hv⟩ := h a List.mem_cons_self
    obtain ⟨vs, hvs⟩ := mapM_ok (l := l) fun x hx => h x (List.mem_cons_of_mem _ hx)
    exact ⟨v :: vs, by rw [List.mapM_cons, hv, hvs]; rfl⟩

private theorem value_of {r : StreamedWipe.Batch} {b1 b2 b3 b4 b5 b6 : ByteArray}
    (h1 : StreamedWipe.beInt 4 r.b "b" = .ok b1) (h2 : StreamedWipe.beInt 4 r.first "first" = .ok b2)
    (h3 : StreamedWipe.beInt 4 r.count "count" = .ok b3) (h4 : StreamedWipe.beInt 8 r.sendNs "send_ns" = .ok b4)
    (h5 : StreamedWipe.beInt 8 r.receiptNs "receipt_ns" = .ok b5) (h6 : StreamedWipe.beInt 4 r.wrong "wrong" = .ok b6) :
    ∃ v, r.value = .ok v := by
  unfold StreamedWipe.Batch.value
  simp only [h1, h2, h3, h4, h5, h6, bind, Except.bind, pure, Except.pure]
  exact ⟨_, rfl⟩

/-- Each batch of the measured session encodes: its fields are below `2^32`, and its clock readings below `2^64`. -/
private theorem measured_value (digest : Nat → ByteArray) {i : Nat} (hi : i < measuredParams.k) :
    ∃ v, (measuredBatch digest i).value = .ok v := by
  simp only [measuredParams] at hi
  obtain ⟨b1, h1⟩ := beInt_of (n := 4) (x := (i : Int)) "b" (by omega) (by simp; omega)
  obtain ⟨b2, h2⟩ := beInt_of (n := 4) (x := (i : Int)) "first" (by omega) (by simp; omega)
  obtain ⟨b3, h3⟩ := beInt_of (n := 4) (x := 1) "count" (by omega) (by simp)
  obtain ⟨b4, h4⟩ := beInt_of (n := 8) (x := MEASURED_FIRST_SEND_NS + i * MEASURED_MAX_READ_NS) "send_ns"
    (by simp only [MEASURED_FIRST_SEND_NS, MEASURED_MAX_READ_NS]; omega)
    (by simp only [MEASURED_FIRST_SEND_NS, MEASURED_MAX_READ_NS]; omega)
  obtain ⟨b5, h5⟩ := beInt_of (n := 8) (x := MEASURED_FIRST_SEND_NS + (i + 1) * MEASURED_MAX_READ_NS) "receipt_ns"
    (by simp only [MEASURED_FIRST_SEND_NS, MEASURED_MAX_READ_NS]; omega)
    (by simp only [MEASURED_FIRST_SEND_NS, MEASURED_MAX_READ_NS]; omega)
  obtain ⟨b6, h6⟩ := beInt_of (n := 4) (x := 0) "wrong" (by omega) (by simp)
  exact value_of h1 h2 h3 h4 h5 h6

private theorem measured_values (digest : Nat → ByteArray) :
    ∃ vs, ((Array.range measuredParams.k).map (measuredBatch digest)).mapM StreamedWipe.Batch.value = .ok vs := by
  rw [Array.mapM_eq_mapM_toList]
  obtain ⟨l, hl⟩ := mapM_ok (f := StreamedWipe.Batch.value)
    (l := ((Array.range measuredParams.k).map (measuredBatch digest)).toList) fun x hx => by
      simp only [Array.toList_map, Array.toList_range, List.mem_map, List.mem_range] at hx
      obtain ⟨i, hi, rfl⟩ := hx
      exact measured_value digest hi
  exact ⟨l.toArray, by rw [hl]; rfl⟩

private theorem measured_message (digest : Nat → ByteArray) (pk sig : ByteArray) :
    ∃ m, (measuredSession digest pk sig).message = .ok m := by
  obtain ⟨vs, hvs⟩ := measured_values digest
  have hvs' : (measuredSession digest pk sig).batches.mapM StreamedWipe.Batch.value = .ok vs := hvs
  unfold StreamedWipe.Signed.message StreamedWipe.message StreamedWipe.recordsRoot
  rw [hvs']
  exact ⟨_, rfl⟩

private theorem measured_size (digest : Nat → ByteArray) (pk sig : ByteArray) :
    (measuredSession digest pk sig).batches.size = measuredCommitment.params.batches := by
  simp [measuredSession, measuredCommitment, measuredParams, StreamedWipe.Params.batches]

/-- Each batch of the measured session passes the verdict's checks: it is the session's batch, has no wrong answer,
took the slowest round trip measured, 1.40 ms, within the 20 ms deadline, and the first was revealed 7.6 ms after the
fill ended. -/
private theorem measured_batch (digest : Nat → ByteArray) (pk sig : ByteArray) (b : Nat)
    (hb : b < (measuredSession digest pk sig).batches.size) :
    (measuredSession digest pk sig).batches[b].b = (b : Int) ∧
      (measuredSession digest pk sig).batches[b].first = ((b * measuredCommitment.params.batch : Nat) : Int) ∧
      (measuredSession digest pk sig).batches[b].count =
        min (measuredCommitment.params.batch : Int)
          ((measuredCommitment.params.k : Int) - (measuredSession digest pk sig).batches[b].first) ∧
      (measuredSession digest pk sig).batches[b].wrong = 0 ∧ 0 ≤ (measuredSession digest pk sig).batches[b].elapsedNs ∧
      (measuredSession digest pk sig).batches[b].elapsedNs ≤ (measuredCommitment.params.limitNs : Int) ∧
      (b = 0 → (measuredSession digest pk sig).fillEndNs ≤ (measuredSession digest pk sig).batches[b].sendNs) := by
  have hb' : b < 2152 := by simpa [measuredSession, measuredParams] using hb
  simp only [measuredSession, measuredCommitment, Array.getElem_map, Array.getElem_range, measuredBatch, measuredParams,
    StreamedWipe.Batch.elapsedNs, MEASURED_FILL_END_NS, MEASURED_FIRST_SEND_NS, MEASURED_MAX_READ_NS]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;> first | trivial | omega | simp | (intro h; subst h; omega)

/-- **The compiled check accepts the measured honest wipe**: for every key `Lms.key` derives at S1's deployed LMS
parameters and every digest of the batches' answers, the session at r20261026-094403-bb5f's parameters and clock
readings, each read taking the slowest round trip measured, signed at leaf 0 with `Lms.signAt` on its message, is
accepted by the wipe's compiled check of a period that trusts that key. The signature verifies by LMS's correctness
(`Properties.CertifierDevice.Proofs.Lms.lmsCorrect`). -/
theorem measured_accepts (digest : Nat → ByteArray) {I seed : ByteArray} {k : Lms.PrivateKey}
    (hk : Lms.key Lms.deployed I seed = .ok k) :
    ∃ (sig : ByteArray) (out : WipeProtocol.Public) (prof : StreamedWipe.Params × Int),
      WipeProtocol.protocol.check (measuredWipeParams k.publicKey) measuredCommitment
        ⟨some (measuredSession digest k.publicKey sig)⟩ = .accept out prof := by
  obtain ⟨-, hIk, -, hIs, -, -, -⟩ := Properties.CertifierDevice.Proofs.Lms.key_spec hk
  subst hIk
  have hroot : (k.tree[1]!).size = Lms.deployed.N :=
    Properties.CertifierDevice.Proofs.Lms.node_size hk (by decide) 1 le_rfl (by decide)
  have hpub : k.publicKey.size = 16 + Lms.deployed.N := by
    simp only [Lms.PrivateKey.publicKey, ByteArray.size_append, hIs, hroot]
  have hI : k.publicKey.extract 0 16 = k.I := ByteArray.extract_append_eq_left hIs.symm
  obtain ⟨m, hm⟩ := measured_message digest k.publicKey .empty
  obtain ⟨M, hM⟩ : ∃ M, memoryChallengerMessage k.I MEASURED_SESSION m = .ok M := by
    unfold memoryChallengerMessage
    simp [hIs]
    exact ⟨_, rfl⟩
  obtain ⟨sig, hsig⟩ :=
    Properties.CertifierDevice.Proofs.Lms.lmsSigns 64 1 20 false Lms.deployed k.I seed k 0 M rfl hk (by decide)
  obtain ⟨hver, hdec⟩ :=
    Properties.CertifierDevice.Proofs.Lms.lmsCorrect 64 1 20 false Lms.deployed k.I seed k 0 M sig rfl hk hsig
  have hlms : lmsVerify Lms.deployed k.publicKey MEASURED_SESSION m sig.encode = .ok true := by
    unfold lmsVerify
    simp only [hpub, ne_eq, not_true_eq_false, ↓reduceIte, hdec, hI, hM, hver, bind, Except.bind] <;> rfl
  exact ⟨sig.encode, _, _, check_accepts _ _ (measuredSession digest k.publicKey sig.encode) rfl rfl
    ⟨Lms.deployed, rfl, List.mem_singleton.2 rfl, m, hm, hlms⟩ rfl rfl rfl (measured_size digest _ _)
    (measured_batch digest _ _)⟩

/-- A key seed of the deployed parameters' 64 bytes. -/
private def HONEST_SEED : ByteArray := ⟨Array.replicate 64 0⟩

/-- `Lms.keygen` derives a key from it at the deployed parameters: the key `Lms.key` derives from its identifier. -/
theorem honest_keygen :
    ∃ k, Lms.keygen Lms.deployed HONEST_SEED = .ok k ∧ ∃ I, Lms.key Lms.deployed I HONEST_SEED = .ok k := by
  have hI : ((Sha512.hash ("verity/lms/v0/I".toUTF8 ++ HONEST_SEED)).extract 0 16).size = 16 := by
    simp [ByteArray.size_extract, Tree.hash_size]
  obtain ⟨k, hk⟩ := Properties.CertifierDevice.Proofs.Lms.key_ok Lms.deployed hI HONEST_SEED
  have hs : HONEST_SEED.size = Lms.deployed.N := rfl
  refine ⟨k, ?_, _, hk⟩
  unfold Lms.keygen
  simpa only [hs, ne_eq, not_true_eq_false, ↓reduceIte, pure, Except.pure, bind, Except.bind] using hk

/-! ## A wipe that is no run -/

private theorem unencoded_values : ∃ err, unencoded.batches.mapM StreamedWipe.Batch.value = .error err :=
  ⟨_, by rw [Array.mapM_eq_mapM_toList] <;> rfl⟩

private theorem unencoded_root :
    ∃ err, StreamedWipe.recordsRoot unencoded.commitment.session unencoded.batches = .error err := by
  obtain ⟨err, h⟩ := unencoded_values
  refine ⟨err, ?_⟩
  unfold StreamedWipe.recordsRoot
  rw [h] <;> rfl

private theorem unencoded_message : ∃ err, unencoded.message = .error err := by
  obtain ⟨err, h⟩ := unencoded_root
  refine ⟨err, ?_⟩
  unfold StreamedWipe.Signed.message StreamedWipe.message
  rw [h] <;> rfl

/-- The collision finder outputs nothing on the unencoded session: the run hashes no string for it, and the idle
challengers ran nothing. -/
theorem unencoded_finder : wipeFinder unencoded idleChallenger = none := by
  obtain ⟨e1, h1⟩ := unencoded_values
  obtain ⟨e2, h2⟩ := unencoded_root
  obtain ⟨e3, h3⟩ := unencoded_message
  have h3' : StreamedWipe.message unencoded.commitment unencoded.ended unencoded.fillStartNs unencoded.fillEndNs
      unencoded.batches unencoded.egressBits = .error e3 := h3
  have hs : wipeRunStrings unencoded idleChallenger = [] := by
    unfold wipeRunStrings wipeSessionStrings
    rw [h1, h2, h3'] <;> rfl
  unfold wipeFinder
  rw [hs] <;> rfl

/-- The forger outputs nothing on the unencoded session: its signature signs no message. -/
theorem unencoded_forger : wipeForger unencoded = none := by
  obtain ⟨e, h⟩ := unencoded_message
  unfold wipeForger wipeMessage
  rw [h] <;> rfl

end Wipe

theorem wipeSigned : WipeSigned := by
  intro p c e out prof dev h
  obtain ⟨s, hse, ⟨q, hq, ht, m, hm, hv⟩, hsc, hb1, hend, hsize, hg⟩ := Wipe.accepted_of h
  unfold WipeRanAccepted
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
      · exact Or.inr (Or.inl (Wipe.collisionIn_finds hcol))
      have cf : CF (wipeRunStrings s dev) := fun x hx y hy hxy =>
        Classical.byContradiction fun hne => hcol ⟨x, hx, y, hy, hne, hxy⟩
      by_cases hsig : dev.Signed s.publicKey M
      · obtain ⟨r, hr, hc, hen, hfs, hfe, hb, heg⟩ := Wipe.same_run cf hm hM hsig
        have hrc : r.commitment = c := hc.trans hsc
        refine Or.inl ⟨r, hr, hrc, hfs, hfe, hb, heg, ?_⟩
        subst hrc
        exact Wipe.accepted_run (by rw [hen, hend]) hb hfe hb1 hsize hg
      · refine Or.inr (Or.inr ⟨M, s.signature, ?_, q, hq, ht, ⟨σ, hd, hv⟩, hsig⟩)
        simp [wipeForger, wipeMessage, hm, hM, bind, Except.bind, Except.toOption]

/-- **`WipeSigned`'s honest lemma**: its hypothesis holds at the honest deployed instance. A key `Lms.keygen` derives at
S1's deployed LMS parameters (`Lms.deployed`, the one scheme `SCHEMES` accepts) signs the honest session at
r20261026-094403-bb5f's parameters (`2^24` labels, `k = 2152` reads one at a time, 20 ms each) and clock readings (the
fill's 2.95 s, the first read revealed 7.6 ms after it, each read taking the slowest round trip measured, 1.40 ms),
each read answered with its label, and the wipe's compiled check of the period that trusts that key accepts it. -/
theorem wipeSigned_honest :
    ∃ (seed K : ByteArray) (c : Nat → Nat) (k : Lms.PrivateKey) (sig : ByteArray) (out : WipeProtocol.Public)
      (prof : StreamedWipe.Params × Int),
      Lms.keygen Lms.deployed seed = .ok k ∧
        WipeProtocol.protocol.check (measuredWipeParams k.publicKey) measuredCommitment
          ⟨some (measuredSession (measuredDigest K c) k.publicKey sig)⟩ = .accept out prof := by
  obtain ⟨k, hkg, I, hk⟩ := Wipe.honest_keygen
  obtain ⟨sig, out, prof, h⟩ := Wipe.measured_accepts (measuredDigest .empty id) hk
  exact ⟨_, .empty, id, k, sig, out, prof, hkg, h⟩

/-- **`WipeSigned`'s nontrivial lemma**: its conclusion fails at the unencoded session under the empty key, for the
challengers that ran nothing. No wipe of theirs is it, the collision finder outputs nothing (the run hashes no
string), and the forger outputs nothing (the signature signs no message). -/
theorem wipeSigned_nontrivial :
    ¬ WipeRanAccepted idleChallenger unencodedParams measuredCommitment ⟨some unencoded⟩ := by
  unfold WipeRanAccepted
  rintro ⟨s, hs, -, ⟨r, hr, -⟩ | hbr⟩
  · exact absurd hr (by simp [idleChallenger])
  · obtain rfl : unencoded = s := Option.some.inj hs
    unfold WipeBreak at hbr
    rcases hbr with ⟨x, y, hf, -⟩ | ⟨m, sig, hf, -⟩
    · rw [Wipe.unencoded_finder] at hf
      cases hf
    · rw [Wipe.unencoded_forger] at hf
      cases hf

/-- **`WipeSigned` fails at a verifier that accepts everything**: such a check accepts the unencoded session, where
the conclusion fails (`wipeSigned_nontrivial`). -/
theorem wipeSigned_acceptAll
    (check : WipeProtocol.Params → StreamedWipe.Commitment → WipeProtocol.Evidence →
      Core.Outcome WipeProtocol.Public (StreamedWipe.Params × Int))
    (hall : ∀ p c e, ∃ out prof, check p c e = .accept out prof) :
    ¬ ∀ (p : WipeProtocol.Params) (c : StreamedWipe.Commitment) (e : WipeProtocol.Evidence)
      (out : WipeProtocol.Public) (prof : StreamedWipe.Params × Int) (dev : SignedWipeParty),
      check p c e = .accept out prof → WipeRanAccepted dev p c e := by
  intro h
  obtain ⟨out, prof, hc⟩ := hall unencodedParams measuredCommitment ⟨some unencoded⟩
  exact wipeSigned_nontrivial (h _ _ _ out prof idleChallenger hc)

end MemoryChallenger.Properties.SecurityProofs
