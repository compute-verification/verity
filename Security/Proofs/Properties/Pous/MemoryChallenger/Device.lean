import Proofs.Properties.Pous.MemoryChallenger.Header
import Proofs.Properties.Pous.MemoryChallenger.Inputs

/-!
# A verifying signature is the device's, or the run carries a break

The device ends a session `complete` or `late` (`ended_of_recorded`), both plain, so the header it signs is read back
(`Header.header_inj`). If the session's signature verifies over its message and the device signed that message for one
of its sessions, then under no SHA-512 collision among the run's strings the two signed inputs are one, so the two
records, so the two headers and records roots, so the commitments, the ends and (through `Tree.tree_bind`) the rounds
(`signedByDevice_of`). Otherwise the device never signed it: a forgery.
-/

namespace MemoryChallenger.Properties.SecurityProofs.Device

open MemoryChallenger Flock
open Tree (CF)

/-- How a device's record has ended, if it has. -/
def Ends (st : Recorded) : Prop := st.ended = none ∨ st.ended = some COMPLETE ∨ st.ended = some LATE

theorem ends_round (coins : CoinTree) (limitNs blockBytes : Nat) (st : Recorded) (o : Observed) (h : Ends st) :
    Ends (Recorded.round coins limitNs blockBytes st o) := by
  unfold Recorded.round
  split
  · exact h
  · unfold Ends
    simp only
    split_ifs <;> simp

theorem ends_foldl (coins : CoinTree) (limitNs blockBytes : Nat) :
    ∀ (l : List Observed) (st : Recorded), Ends st → Ends (l.foldl (Recorded.round coins limitNs blockBytes) st)
  | [], _, h => h
  | o :: l, st, h => ends_foldl coins limitNs blockBytes l _ (ends_round coins limitNs blockBytes st o h)

/-- The device ends a session `complete` or `late`. -/
theorem ended_of_recorded (r : Run) {e : String} (h : r.recorded.ended = some e) : e = COMPLETE ∨ e = LATE := by
  rcases ends_foldl r.coins r.limitNs r.blockBytes r.observed ⟨#[], none⟩ (Or.inl rfl) with h' | h' | h' <;>
    rw [show Run.recorded r = r.observed.foldl (Recorded.round r.coins r.limitNs r.blockBytes) ⟨#[], none⟩ from rfl,
      h'] at h
  · cases h
  · exact Or.inl (Option.some.inj h).symm
  · exact Or.inr (Option.some.inj h).symm

theorem plain_of_recorded (r : Run) {e : String} (h : r.recorded.ended = some e) : Header.Plain e := by
  rcases ended_of_recorded r h with rfl | rfl
  · exact Header.plain_complete
  · exact Header.plain_late

/-! ## The strings a session hashes -/

theorem coinKey_mem (pk : ByteArray) (c : SessionCommitment) (e : String) (rounds : Array Round) :
    c.coinKey ∈ sessionStrings pk c e rounds := by
  simp [sessionStrings]

theorem tree_mem {pk : ByteArray} {c : SessionCommitment} {e : String} {rounds : Array Round} {vs : Array ByteArray}
    (hv : rounds.mapM Round.value = .ok vs) {x : ByteArray}
    (hx : x ∈ treeStrings (domain RECORD_BINDING c.session rounds.size) vs) : x ∈ sessionStrings pk c e rounds := by
  simp only [sessionStrings, hv, List.mem_append]
  exact Or.inl (Or.inl (Or.inr hx))

theorem session_mem {pk : ByteArray} {c : SessionCommitment} {e : String} {rounds : Array Round} {root : ByteArray}
    (hr : recordsRoot c.session rounds = .ok root) :
    SESSION_TAG ++ be64 (header c e rounds.size).toUTF8.size.toUInt64 ++ (header c e rounds.size).toUTF8 ++ root ∈
      sessionStrings pk c e rounds := by
  simp [sessionStrings, hr]

theorem signed_mem {pk : ByteArray} {c : SessionCommitment} {e : String} {rounds : Array Round} {rec : ByteArray}
    (hm : message c e rounds = .ok rec) (hI : (pk.extract 0 16).size = 16) :
    field MEMORY_CHALLENGER_TAG.toUTF8 ++ pk.extract 0 16 ++ field (field c.session) ++ field rec ∈
      sessionStrings pk c e rounds := by
  simp [sessionStrings, hm, hI]

theorem own_mem {s : SignedSession} {dev : Device} {x : ByteArray}
    (hx : x ∈ sessionStrings s.publicKey s.commitment s.ended s.rounds) : x ∈ runStrings s dev :=
  List.mem_append_left _ hx

theorem device_mem {s : SignedSession} {dev : Device} {r : Run} (hr : r ∈ dev.sessions s.publicKey) {e : String}
    (he : r.recorded.ended = some e) {x : ByteArray}
    (hx : x ∈ sessionStrings s.publicKey r.commitment e r.recorded.rounds) : x ∈ runStrings s dev := by
  refine List.mem_append_right _ (List.mem_flatMap.mpr ⟨r, hr, ?_⟩)
  simpa only [he] using hx

/-! ## The argument -/

theorem commitment_ext {c c' : SessionCommitment} (h1 : c.session = c'.session) (h2 : c.coinRoot = c'.coinRoot)
    (h3 : c.coinKey = c'.coinKey) (h4 : c.blocks = c'.blocks) (h5 : c.k = c'.k) (h6 : c.limitNs = c'.limitNs)
    (h7 : c.blockBytes = c'.blockBytes) (h8 : c.binding = c'.binding) : c = c' := by
  cases c
  cases c'
  simp_all

/-- Under no collision among the run's strings, a session whose message the device signed for one of its sessions is
that session. -/
theorem same_session {s : SignedSession} {dev : Device} (cf : CF (runStrings s dev)) (he : s.ended = COMPLETE)
    {m M : ByteArray} (hm : s.message = .ok m)
    (hM : memoryChallengerMessage (s.publicKey.extract 0 16) s.commitment.session m = .ok M)
    (hsig : Signed dev s.publicKey M) :
    ∃ r ∈ dev.sessions s.publicKey, r.commitment = s.commitment ∧ r.recorded.ended = some s.ended ∧
      r.recorded.rounds = s.rounds := by
  obtain ⟨r, hr, ended, rec, hend, hmsg, hmcm⟩ := hsig
  have hm : message s.commitment s.ended s.rounds = .ok m := hm
  obtain ⟨root, hroot, rfl⟩ := Inputs.message_ok hm
  obtain ⟨vs, hvs, rfl⟩ := Inputs.recordsRoot_ok hroot
  obtain ⟨root', hroot', rfl⟩ := Inputs.message_ok hmsg
  obtain ⟨vs', hvs', rfl⟩ := Inputs.recordsRoot_ok hroot'
  obtain ⟨hI, rfl⟩ := Inputs.signedMessage_ok hM
  obtain ⟨-, hMM⟩ := Inputs.signedMessage_ok hmcm
  have e1 := cf _ (own_mem (signed_mem hm hI)) _ (device_mem hr hend (signed_mem hmsg hI)) hMM
  have e2 := Inputs.signed_split (by rw [Tree.hash_size, Tree.hash_size]) e1
  have e3 := cf _ (own_mem (session_mem hroot)) _ (device_mem hr hend (session_mem hroot')) e2
  obtain ⟨e4, e5⟩ := Inputs.session_split (by rw [Tree.root_width, Tree.root_width]) e3
  have hp : Header.Plain s.ended := he ▸ Header.plain_complete
  obtain ⟨hb, hbb, hbl, hck, hcr, hk, hl, hs, hen, hn⟩ :=
    Header.header_inj hp (plain_of_recorded r hend) (String.toByteArray_inj.mp e4)
  have hkey := cf _ (own_mem (coinKey_mem _ _ _ _)) _ (device_mem hr hend (coinKey_mem _ _ _ _)) hck
  rw [hs, hn] at e5
  have hsz : vs.size = vs'.size := by rw [Inputs.mapM_size hvs, Inputs.mapM_size hvs', hn]
  have hvv := Tree.tree_bind hsz (fun x hx => own_mem (tree_mem hvs (by rwa [hs, hn])))
    (fun x hx => device_mem hr hend (tree_mem hvs' hx)) cf e5
  subst hvv
  refine ⟨r, hr, (commitment_ext hs hcr hkey hbl hk hl hbb hb).symm, by rw [hend, hen], ?_⟩
  exact (Inputs.rounds_inj hvs hvs').symm

/-- **A verifying signature is the device's, or the run carries a break.** -/
theorem signedByDevice_of {schemes : Schemes} {trusted : List ByteArray} {s : SignedSession} (dev : Device)
    {p : Lms.Params} (hp : schemes.lookup s.scheme = some p) (ht : s.publicKey ∈ trusted) {m : ByteArray}
    (hm : s.message = .ok m) (hv : lmsVerify p s.publicKey s.commitment.session m s.signature = .ok true)
    (he : s.ended = COMPLETE) : SignedByDevice schemes trusted s dev := by
  unfold lmsVerify at hv
  split at hv
  · simp [pure, Except.pure] at hv
  cases hd : Lms.Signature.decode p s.signature with
  | error _ => simp [hd, bind, Except.bind] at hv
  | ok σ =>
    cases hM : memoryChallengerMessage (s.publicKey.extract 0 16) s.commitment.session m with
    | error _ => simp [hd, hM, bind, Except.bind] at hv
    | ok M =>
      simp only [hd, hM, bind, Except.bind] at hv
      by_cases hcol : ∃ x ∈ runStrings s dev, ∃ y ∈ runStrings s dev, x ≠ y ∧ Sha512.hash x = Sha512.hash y
      · exact Or.inr (Or.inl hcol)
      have cf : CF (runStrings s dev) := fun x hx y hy hxy =>
        Classical.byContradiction fun hne => hcol ⟨x, hx, y, hy, hne, hxy⟩
      by_cases hsig : Signed dev s.publicKey M
      · exact Or.inl (same_session cf he hm hM hsig)
      · refine Or.inr (Or.inr ⟨p, M, hp, ht, ?_, ⟨σ, hd, hv⟩, hsig⟩)
        simp [sessionMessage, hm, hM, bind, Except.bind, Except.toOption]

end MemoryChallenger.Properties.SecurityProofs.Device
