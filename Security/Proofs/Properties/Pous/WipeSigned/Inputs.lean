import Proofs.Properties.Pous.WipeSigned.Tree
import Proofs.Properties.Pous.MemoryChallenger.Inputs

/-!
# What the wipe's hashes read

The wipe's `message` and `recordsRoot` unfolded to the SHA-512 inputs they hash (`message_ok`, `recordsRoot_ok`), the
session input split back into its header and records root (`session_split`), a batch's value into its batch
(`value_inj`, `batches_inj`), and the wipe's session input told apart from PoUS's by their tags (`tags_differ`).
-/

namespace MemoryChallenger.Properties.SecurityProofs.WipeInputs

open MemoryChallenger Flock
open Properties.NetworkCertifier.Bytes (bs bs_append bs_length bs_inj split_bytes be64_size)

theorem beInt_ok {n : Nat} {x : Int} {w : String} {b : ByteArray} (h : StreamedWipe.beInt n x w = .ok b) :
    0 ≤ x ∧ beN n x.toNat w = .ok b := by
  unfold StreamedWipe.beInt at h
  split at h
  · simp [throw, throwThe, MonadExceptOf.throw] at h
  · rename_i hx
    exact ⟨Int.not_lt.mp hx, h⟩

/-- A fixed-width field of either sign and what follows it determine the field's value and what follows. -/
theorem beInt_split {n : Nat} {x y : Int} {w w' : String} {b b' : ByteArray} {r r' : List UInt8}
    (h : StreamedWipe.beInt n x w = .ok b) (h' : StreamedWipe.beInt n y w' = .ok b') (e : bs b ++ r = bs b' ++ r') :
    x = y ∧ r = r' := by
  obtain ⟨hx, h⟩ := beInt_ok h
  obtain ⟨hy, h'⟩ := beInt_ok h'
  obtain ⟨e1, e2⟩ := Inputs.beN_split h h' e
  refine ⟨?_, e2⟩
  rw [← Int.toNat_of_nonneg hx, ← Int.toNat_of_nonneg hy, e1]

theorem value_ok {r : StreamedWipe.Batch} {v : ByteArray} (h : r.value = .ok v) :
    ∃ b1 b2 b3 b4 b5 b6, StreamedWipe.beInt 4 r.b "b" = .ok b1 ∧ StreamedWipe.beInt 4 r.first "first" = .ok b2 ∧
      StreamedWipe.beInt 4 r.count "count" = .ok b3 ∧ StreamedWipe.beInt 8 r.sendNs "send_ns" = .ok b4 ∧
      StreamedWipe.beInt 8 r.receiptNs "receipt_ns" = .ok b5 ∧ StreamedWipe.beInt 4 r.wrong "wrong" = .ok b6 ∧
      v = b1 ++ b2 ++ b3 ++ b4 ++ b5 ++ b6 ++ r.digest := by
  unfold StreamedWipe.Batch.value at h
  cases h1 : StreamedWipe.beInt 4 r.b "b" <;> cases h2 : StreamedWipe.beInt 4 r.first "first" <;>
    cases h3 : StreamedWipe.beInt 4 r.count "count" <;> cases h4 : StreamedWipe.beInt 8 r.sendNs "send_ns" <;>
    cases h5 : StreamedWipe.beInt 8 r.receiptNs "receipt_ns" <;> cases h6 : StreamedWipe.beInt 4 r.wrong "wrong" <;>
    simp only [h1, h2, h3, h4, h5, h6, bind, Except.bind, pure, Except.pure, reduceCtorEq, Except.ok.injEq] at h
  exact ⟨_, _, _, _, _, _, rfl, rfl, rfl, rfl, rfl, rfl, h.symm⟩

/-- A batch's value determines the batch. -/
theorem value_inj {r r' : StreamedWipe.Batch} {v : ByteArray} (h : r.value = .ok v) (h' : r'.value = .ok v) :
    r = r' := by
  obtain ⟨b1, b2, b3, b4, b5, b6, h1, h2, h3, h4, h5, h6, rfl⟩ := value_ok h
  obtain ⟨c1, c2, c3, c4, c5, c6, k1, k2, k3, k4, k5, k6, e⟩ := value_ok h'
  have e := congrArg bs e
  simp only [bs_append, List.append_assoc] at e
  obtain ⟨e1, e⟩ := beInt_split h1 k1 e
  obtain ⟨e2, e⟩ := beInt_split h2 k2 e
  obtain ⟨e3, e⟩ := beInt_split h3 k3 e
  obtain ⟨e4, e⟩ := beInt_split h4 k4 e
  obtain ⟨e5, e⟩ := beInt_split h5 k5 e
  obtain ⟨e6, e⟩ := beInt_split h6 k6 e
  have ed := bs_inj e
  cases r
  cases r'
  simp only [StreamedWipe.Batch.mk.injEq]
  exact ⟨e1, e2, e3, e4, e5, e6, ed⟩

theorem forall₂_of_mapM {f : StreamedWipe.Batch → E ByteArray} :
    ∀ {l : List StreamedWipe.Batch} {vs : List ByteArray}, l.mapM f = .ok vs →
      List.Forall₂ (fun a v => f a = .ok v) l vs
  | [], vs, h => by
    simp only [List.mapM_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact .nil
  | a :: l, vs, h => by
    rw [List.mapM_cons] at h
    cases ha : f a with
    | error _ => simp [ha, bind, Except.bind] at h
    | ok v =>
      cases hl : l.mapM f with
      | error _ => simp [ha, hl, bind, Except.bind] at h
      | ok vs' =>
        simp only [ha, hl, bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
        subst h
        exact .cons ha (forall₂_of_mapM hl)

theorem array_mapM {batches : Array StreamedWipe.Batch} {vs : Array ByteArray}
    (h : batches.mapM StreamedWipe.Batch.value = .ok vs) :
    List.Forall₂ (fun a v => a.value = .ok v) batches.toList vs.toList := by
  rw [Array.mapM_eq_mapM_toList] at h
  cases hl : batches.toList.mapM StreamedWipe.Batch.value with
  | error _ => simp [hl, Functor.map, Except.map] at h
  | ok l =>
    simp only [hl, Functor.map, Except.map, Except.ok.injEq] at h
    subst h
    exact forall₂_of_mapM hl

theorem mapM_size {batches : Array StreamedWipe.Batch} {vs : Array ByteArray}
    (h : batches.mapM StreamedWipe.Batch.value = .ok vs) : vs.size = batches.size := by
  have := (array_mapM h).length_eq
  simpa using this.symm

theorem forall₂_inj : ∀ {l l' : List StreamedWipe.Batch} {vs : List ByteArray},
    List.Forall₂ (fun a v => a.value = .ok v) l vs → List.Forall₂ (fun a v => a.value = .ok v) l' vs → l = l'
  | [], [], _, _, _ => rfl
  | _, _, [], .nil, .nil => rfl
  | a :: l, a' :: l', v :: vs, .cons ha hl, .cons ha' hl' => by
    rw [value_inj ha ha', forall₂_inj hl hl']

/-- The batches' values determine the batches. -/
theorem batches_inj {batches batches' : Array StreamedWipe.Batch} {vs : Array ByteArray}
    (h : batches.mapM StreamedWipe.Batch.value = .ok vs) (h' : batches'.mapM StreamedWipe.Batch.value = .ok vs) :
    batches = batches' :=
  Array.toList_inj.mp (forall₂_inj (array_mapM h) (array_mapM h'))

theorem recordsRoot_ok {session root : ByteArray} {batches : Array StreamedWipe.Batch}
    (h : StreamedWipe.recordsRoot session batches = .ok root) :
    ∃ vs, batches.mapM StreamedWipe.Batch.value = .ok vs ∧
      root = FrameV3Sha512.root (domain StreamedWipe.RECORD_BINDING session batches.size)
        (WipeTree.leavesOf (domain StreamedWipe.RECORD_BINDING session batches.size) vs) := by
  unfold StreamedWipe.recordsRoot at h
  cases hv : batches.mapM StreamedWipe.Batch.value with
  | error _ => simp [hv, bind, Except.bind] at h
  | ok vs =>
    simp only [hv, bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
    exact ⟨vs, rfl, h.symm⟩

theorem message_ok {c : StreamedWipe.Commitment} {e : String} {fs fe : Int} {batches : Array StreamedWipe.Batch}
    {eg : Int} {rec : ByteArray} (h : StreamedWipe.message c e fs fe batches eg = .ok rec) :
    ∃ root, StreamedWipe.recordsRoot c.session batches = .ok root ∧
      rec = Sha512.hash (StreamedWipe.SESSION_TAG ++
        be64 (StreamedWipe.header c e fs fe batches.size eg).toUTF8.size.toUInt64 ++
        (StreamedWipe.header c e fs fe batches.size eg).toUTF8 ++ root) := by
  unfold StreamedWipe.message at h
  cases hr : StreamedWipe.recordsRoot c.session batches with
  | error _ => simp [hr, bind, Except.bind] at h
  | ok root =>
    simp only [hr, bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
    exact ⟨root, rfl, h.symm⟩

/-- The wipe's session input, with records roots of one width, determines the header and the root. -/
theorem session_split {h h' r r' : ByteArray} (hr : r.size = r'.size)
    (e : StreamedWipe.SESSION_TAG ++ be64 h.size.toUInt64 ++ h ++ r =
      StreamedWipe.SESSION_TAG ++ be64 h'.size.toUInt64 ++ h' ++ r') :
    h = h' ∧ r = r' := by
  have e := congrArg bs e
  simp only [bs_append] at e
  obtain ⟨e1, e2⟩ := List.append_inj' e (by rw [bs_length, bs_length, hr])
  simp only [List.append_assoc] at e1
  have e1 := List.append_cancel_left e1
  obtain ⟨-, e1⟩ := split_bytes e1 (by rw [be64_size, be64_size])
  exact ⟨bs_inj e1, bs_inj e2⟩

theorem tag_byte : (bs SESSION_TAG)[12]? = some 109 := by decide +kernel

theorem wipe_tag_byte : (bs StreamedWipe.SESSION_TAG)[12]? = some 115 := by decide +kernel

theorem tag_size : 12 < SESSION_TAG.size := by decide +kernel

theorem wipe_tag_size : 12 < StreamedWipe.SESSION_TAG.size := by decide +kernel

/-- **PoUS's session input is never the wipe's**: their tags differ at byte 12 (`memory` against `streamed`). -/
theorem tags_differ (a b c a' b' c' : ByteArray) :
    SESSION_TAG ++ a ++ b ++ c ≠ StreamedWipe.SESSION_TAG ++ a' ++ b' ++ c' := by
  intro e
  have e := congrArg (fun x => (bs x)[12]?) e
  simp only [bs_append, List.append_assoc] at e
  have h1 : 12 < (bs SESSION_TAG).length := by rw [bs_length]; exact tag_size
  have h2 : 12 < (bs StreamedWipe.SESSION_TAG).length := by rw [bs_length]; exact wipe_tag_size
  rw [List.getElem?_append_left h1, List.getElem?_append_left h2, tag_byte, wipe_tag_byte] at e
  exact absurd e (by decide)

end MemoryChallenger.Properties.SecurityProofs.WipeInputs
