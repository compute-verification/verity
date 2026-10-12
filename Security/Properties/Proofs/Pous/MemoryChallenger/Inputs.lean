import Properties.Proofs.Pous.MemoryChallenger.Tree
import Mathlib.Data.List.Forall2

/-!
# What the session's hashes read

`message`, `recordsRoot` and `memoryChallengerMessage` unfolded to the SHA-512 inputs they hash (`message_ok`,
`recordsRoot_ok`, `signedMessage_ok`), and each input split back into what it frames: the session input into its header
and records root (`session_split`), the signed message's input into its record (`signed_split`), a round's value into
its round (`value_inj`, `rounds_inj`).
-/

namespace MemoryChallenger.Properties.SecurityProofs.Inputs

open MemoryChallenger Flock
open Properties.NetworkCertifier.Bytes (bs bs_append bs_length bs_inj split_bytes be64_size ofBytes ofBytes_be)

theorem beN_ok {n x : Nat} {w : String} {b : ByteArray} (h : beN n x w = .ok b) :
    x < 2 ^ (8 * n) ∧ bs b = (List.range n).map fun i => (x >>> (8 * (n - 1 - i))).toUInt8 := by
  unfold beN at h
  split at h
  · rename_i hx
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact ⟨hx, by simp [bs]⟩
  · simp [throw, throwThe, MonadExceptOf.throw] at h

theorem beN_size {n x : Nat} {w : String} {b : ByteArray} (h : beN n x w = .ok b) : b.size = n := by
  rw [← bs_length, (beN_ok h).2]
  simp

/-- A fixed-width field and what follows it determine the field's value and what follows. -/
theorem beN_split {n x y : Nat} {w w' : String} {b b' : ByteArray} {r r' : List UInt8} (h : beN n x w = .ok b)
    (h' : beN n y w' = .ok b') (e : bs b ++ r = bs b' ++ r') : x = y ∧ r = r' := by
  obtain ⟨hb, hr⟩ := split_bytes e (by rw [beN_size h, beN_size h'])
  subst hb
  obtain ⟨hx, e1⟩ := beN_ok h
  obtain ⟨hy, e2⟩ := beN_ok h'
  refine ⟨?_, hr⟩
  rw [← ofBytes_be n x hx, ← ofBytes_be n y hy, ← e1, ← e2]

theorem value_ok {r : Round} {v : ByteArray} (h : r.value = .ok v) :
    ∃ b1 b2 b3 b4, beN 4 r.j "j" = .ok b1 ∧ beN 8 r.index "index" = .ok b2 ∧ beN 8 r.sendNs "send_ns" = .ok b3 ∧
      beN 8 r.receiptNs "receipt_ns" = .ok b4 ∧ v = b1 ++ b2 ++ b3 ++ b4 ++ r.digest := by
  unfold Round.value at h
  cases h1 : beN 4 r.j "j" <;> cases h2 : beN 8 r.index "index" <;> cases h3 : beN 8 r.sendNs "send_ns" <;>
    cases h4 : beN 8 r.receiptNs "receipt_ns" <;>
    simp only [h1, h2, h3, h4, bind, Except.bind, pure, Except.pure, reduceCtorEq, Except.ok.injEq] at h
  exact ⟨_, _, _, _, rfl, rfl, rfl, rfl, h.symm⟩

/-- A round's value determines the round. -/
theorem value_inj {r r' : Round} {v : ByteArray} (h : r.value = .ok v) (h' : r'.value = .ok v) : r = r' := by
  obtain ⟨b1, b2, b3, b4, h1, h2, h3, h4, rfl⟩ := value_ok h
  obtain ⟨c1, c2, c3, c4, k1, k2, k3, k4, e⟩ := value_ok h'
  have e := congrArg bs e
  simp only [bs_append, List.append_assoc] at e
  obtain ⟨ej, e⟩ := beN_split h1 k1 e
  obtain ⟨ei, e⟩ := beN_split h2 k2 e
  obtain ⟨es, e⟩ := beN_split h3 k3 e
  obtain ⟨er, e⟩ := beN_split h4 k4 e
  have ed := bs_inj e
  cases r
  cases r'
  simp_all

theorem forall₂_of_mapM {f : Round → E ByteArray} :
    ∀ {l : List Round} {vs : List ByteArray}, l.mapM f = .ok vs → List.Forall₂ (fun a v => f a = .ok v) l vs
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

theorem array_mapM {rounds : Array Round} {vs : Array ByteArray} (h : rounds.mapM Round.value = .ok vs) :
    List.Forall₂ (fun a v => a.value = .ok v) rounds.toList vs.toList := by
  rw [Array.mapM_eq_mapM_toList] at h
  cases hl : rounds.toList.mapM Round.value with
  | error _ => simp [hl, Functor.map, Except.map] at h
  | ok l =>
    simp only [hl, Functor.map, Except.map, Except.ok.injEq] at h
    subst h
    exact forall₂_of_mapM hl

theorem mapM_size {rounds : Array Round} {vs : Array ByteArray} (h : rounds.mapM Round.value = .ok vs) :
    vs.size = rounds.size := by
  have := (array_mapM h).length_eq
  simpa using this.symm

theorem forall₂_inj : ∀ {l l' : List Round} {vs : List ByteArray}, List.Forall₂ (fun a v => a.value = .ok v) l vs →
    List.Forall₂ (fun a v => a.value = .ok v) l' vs → l = l'
  | [], [], _, _, _ => rfl
  | _, _, [], .nil, .nil => rfl
  | a :: l, a' :: l', v :: vs, .cons ha hl, .cons ha' hl' => by
    rw [value_inj ha ha', forall₂_inj hl hl']

/-- The rounds' values determine the rounds. -/
theorem rounds_inj {rounds rounds' : Array Round} {vs : Array ByteArray} (h : rounds.mapM Round.value = .ok vs)
    (h' : rounds'.mapM Round.value = .ok vs) : rounds = rounds' :=
  Array.toList_inj.mp (forall₂_inj (array_mapM h) (array_mapM h'))

theorem recordsRoot_ok {session root : ByteArray} {rounds : Array Round} (h : recordsRoot session rounds = .ok root) :
    ∃ vs, rounds.mapM Round.value = .ok vs ∧
      root = FrameV3Sha512.root (domain RECORD_BINDING session rounds.size)
        (Tree.leavesOf (domain RECORD_BINDING session rounds.size) vs) := by
  unfold recordsRoot at h
  cases hv : rounds.mapM Round.value with
  | error _ => simp [hv, bind, Except.bind] at h
  | ok vs =>
    simp only [hv, bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
    exact ⟨vs, rfl, h.symm⟩

theorem message_ok {c : SessionCommitment} {e : String} {rounds : Array Round} {rec : ByteArray}
    (h : message c e rounds = .ok rec) :
    ∃ root, recordsRoot c.session rounds = .ok root ∧
      rec = Sha512.hash (SESSION_TAG ++ be64 (header c e rounds.size).toUTF8.size.toUInt64 ++
        (header c e rounds.size).toUTF8 ++ root) := by
  unfold message at h
  cases hr : recordsRoot c.session rounds with
  | error _ => simp [hr, bind, Except.bind] at h
  | ok root =>
    simp only [hr, bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
    exact ⟨root, rfl, h.symm⟩

theorem signedMessage_ok {I session rec m : ByteArray} (h : memoryChallengerMessage I session rec = .ok m) :
    I.size = 16 ∧ m = Sha512.hash (field MEMORY_CHALLENGER_TAG.toUTF8 ++ I ++ field (field session) ++ field rec) := by
  unfold memoryChallengerMessage at h
  split at h
  · simp [throw, throwThe, MonadExceptOf.throw] at h
  · rename_i hI
    simp only [pure, Except.pure, Except.ok.injEq] at h
    exact ⟨not_not.mp hI, h.symm⟩

/-- The session input, with records roots of one width, determines the header and the root. -/
theorem session_split {h h' r r' : ByteArray} (hr : r.size = r'.size)
    (e : SESSION_TAG ++ be64 h.size.toUInt64 ++ h ++ r = SESSION_TAG ++ be64 h'.size.toUInt64 ++ h' ++ r') :
    h = h' ∧ r = r' := by
  have e := congrArg bs e
  simp only [bs_append] at e
  obtain ⟨e1, e2⟩ := List.append_inj' e (by rw [bs_length, bs_length, hr])
  simp only [List.append_assoc] at e1
  have e1 := List.append_cancel_left e1
  obtain ⟨-, e1⟩ := split_bytes e1 (by rw [be64_size, be64_size])
  exact ⟨bs_inj e1, bs_inj e2⟩

/-- The signed message's input, under one key identifier and with records of one width, determines the record. -/
theorem signed_split {I s s' r r' : ByteArray} (hr : r.size = r'.size)
    (e : field MEMORY_CHALLENGER_TAG.toUTF8 ++ I ++ field (field s) ++ field r =
      field MEMORY_CHALLENGER_TAG.toUTF8 ++ I ++ field (field s') ++ field r') : r = r' := by
  have e := congrArg bs e
  simp only [bs_append] at e
  obtain ⟨-, e2⟩ := List.append_inj' e (by simp only [field, bs_append, List.length_append, bs_length, be64_size, hr])
  simp only [field, bs_append] at e2
  exact bs_inj (split_bytes e2 (by rw [be64_size, be64_size])).2

end MemoryChallenger.Properties.SecurityProofs.Inputs
