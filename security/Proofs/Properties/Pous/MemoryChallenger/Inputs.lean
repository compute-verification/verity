module

public import Proofs.Properties.Pous.MemoryChallenger.Tree
public import Batteries.Data.List.Basic
import Batteries.Lean.Except
import Mathlib.Data.Finset.Attr
import Mathlib.Data.List.Forall2
import Mathlib.Tactic.Bound.Init
import Mathlib.Tactic.Finiteness.Attr
import Mathlib.Tactic.SetLike

/-!
# What the session's hashes read

`message`, `recordsDigest` and `memoryChallengerMessage` unfolded to the SHA-512 inputs they hash (`message_ok`,
`recordsDigest_ok`, `signedMessage_ok`), and each input split back into what it frames: the session input into its
header and records digest (`session_split`), the signed message's input into its record (`signed_split`), the records'
stream into its rounds (`stream_inj`: every value is `RECORD_BYTES` wide, `value_size`), a round's value into its round
(`value_inj`, `rounds_inj`).
-/

@[expose] public section

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
    r.digest.size = 64 ∧ ∃ b1 b2 b3 b4, beN 4 r.j "j" = .ok b1 ∧ beN 8 r.index "index" = .ok b2 ∧
      beN 8 r.sendNs "send_ns" = .ok b3 ∧ beN 8 r.receiptNs "receipt_ns" = .ok b4 ∧
      v = b1 ++ b2 ++ b3 ++ b4 ++ r.digest := by
  unfold Round.value at h
  split at h
  · simp [throw, throwThe, MonadExceptOf.throw] at h
  rename_i hd
  refine ⟨not_not.mp hd, ?_⟩
  cases h1 : beN 4 r.j "j" <;> cases h2 : beN 8 r.index "index" <;> cases h3 : beN 8 r.sendNs "send_ns" <;>
    cases h4 : beN 8 r.receiptNs "receipt_ns" <;>
    simp only [h1, h2, h3, h4, bind, Except.bind, pure, Except.pure, reduceCtorEq, Except.ok.injEq] at h
  exact ⟨_, _, _, _, rfl, rfl, rfl, rfl, h.symm⟩

theorem value_size {r : Round} {v : ByteArray} (h : r.value = .ok v) : v.size = RECORD_BYTES := by
  obtain ⟨hd, b1, b2, b3, b4, h1, h2, h3, h4, rfl⟩ := value_ok h
  simp only [ByteArray.size_append, beN_size h1, beN_size h2, beN_size h3, beN_size h4, hd]
  rfl

/-- A round's value determines the round. -/
theorem value_inj {r r' : Round} {v : ByteArray} (h : r.value = .ok v) (h' : r'.value = .ok v) : r = r' := by
  obtain ⟨-, b1, b2, b3, b4, h1, h2, h3, h4, rfl⟩ := value_ok h
  obtain ⟨-, c1, c2, c3, c4, k1, k2, k3, k4, e⟩ := value_ok h'
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

theorem widths : ∀ {l : List Round} {vs : List ByteArray}, List.Forall₂ (fun a v => a.value = .ok v) l vs →
    ∀ v ∈ vs, v.size = RECORD_BYTES
  | _, _, .nil => by simp
  | _, _, .cons ha hl => by
    intro v hv
    rcases List.mem_cons.mp hv with rfl | hv
    · exact value_size ha
    · exact widths hl v hv

theorem bs_foldl : ∀ (l : List ByteArray) (acc : ByteArray), bs (l.foldl (· ++ ·) acc) = bs acc ++ l.flatMap bs
  | [], acc => by simp
  | v :: l, acc => by rw [List.foldl_cons, bs_foldl l, bs_append, List.flatMap_cons, List.append_assoc]

theorem flat_length {n : Nat} : ∀ {l : List ByteArray}, (∀ v ∈ l, v.size = n) → (l.flatMap bs).length = n * l.length
  | [], _ => by simp
  | v :: l, h => by
    rw [List.flatMap_cons, List.length_append, bs_length, h v List.mem_cons_self,
      flat_length (fun x hx => h x (List.mem_cons_of_mem _ hx)), List.length_cons, Nat.mul_add_one, Nat.add_comm]

/-- Equally many values of one width are determined by their concatenation. -/
theorem flat_inj {n : Nat} : ∀ {l l' : List ByteArray}, (∀ v ∈ l, v.size = n) → (∀ v ∈ l', v.size = n) →
    l.length = l'.length → l.flatMap bs = l'.flatMap bs → l = l'
  | [], [], _, _, _, _ => rfl
  | [], _ :: _, _, _, hn, _ => by simp at hn
  | _ :: _, [], _, _, hn, _ => by simp at hn
  | v :: l, v' :: l', h, h', hn, e => by
    simp only [List.flatMap_cons] at e
    obtain ⟨ev, e⟩ := split_bytes e (by rw [h v List.mem_cons_self, h' v' List.mem_cons_self])
    rw [ev, flat_inj (fun x hx => h x (List.mem_cons_of_mem _ hx)) (fun x hx => h' x (List.mem_cons_of_mem _ hx))
      (by simpa using hn) e]

theorem recordsStream_ok {rounds : Array Round} {s : ByteArray} (h : recordsStream rounds = .ok s) :
    ∃ vs, rounds.mapM Round.value = .ok vs ∧ bs s = vs.toList.flatMap bs := by
  unfold recordsStream at h
  cases hv : rounds.mapM Round.value with
  | error _ => simp [hv, bind, Except.bind] at h
  | ok vs =>
    simp only [hv, bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
    subst h
    refine ⟨vs, rfl, ?_⟩
    rw [← Array.foldl_toList, bs_foldl]
    rfl

theorem recordsDigest_ok {rounds : Array Round} {d : ByteArray} (h : recordsDigest rounds = .ok d) :
    ∃ s, recordsStream rounds = .ok s ∧ d = Sha512.hash (HmRow.rowPrefixV2 (8 * s.size) ++ s) := by
  unfold recordsDigest at h
  cases hs : recordsStream rounds with
  | error _ => simp [hs, bind, Except.bind] at h
  | ok s =>
    simp only [hs, bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
    exact ⟨s, rfl, h.symm⟩

/-- Under one round count, the records' streams framed as `sha512/row/v2` inputs determine the rounds. -/
theorem stream_inj {rounds rounds' : Array Round} {s s' : ByteArray} (h : recordsStream rounds = .ok s)
    (h' : recordsStream rounds' = .ok s') (hn : rounds.size = rounds'.size)
    (e : HmRow.rowPrefixV2 (8 * s.size) ++ s = HmRow.rowPrefixV2 (8 * s'.size) ++ s') : rounds = rounds' := by
  obtain ⟨vs, hv, hs⟩ := recordsStream_ok h
  obtain ⟨vs', hv', hs'⟩ := recordsStream_ok h'
  have w := widths (array_mapM hv)
  have w' := widths (array_mapM hv')
  have hl : vs.toList.length = vs'.toList.length := by simp [mapM_size hv, mapM_size hv', hn]
  have hss : s.size = s'.size := by rw [← bs_length, ← bs_length, hs, hs', flat_length w, flat_length w', hl]
  rw [hss] at e
  have e := congrArg bs e
  simp only [bs_append] at e
  have e := List.append_cancel_left e
  rw [hs, hs'] at e
  obtain rfl : vs = vs' := Array.toList_inj.mp (flat_inj w w' hl e)
  exact rounds_inj hv hv'

theorem message_ok {c : SessionCommitment} {e : String} {rounds : Array Round} {rec : ByteArray}
    (h : message c e rounds = .ok rec) :
    ∃ d, recordsDigest rounds = .ok d ∧
      rec = Sha512.hash (SESSION_TAG ++ be64 (header c e rounds.size).toUTF8.size.toUInt64 ++
        (header c e rounds.size).toUTF8 ++ d) := by
  unfold message at h
  cases hr : recordsDigest rounds with
  | error _ => simp [hr, bind, Except.bind] at h
  | ok d =>
    simp only [hr, bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
    exact ⟨d, rfl, h.symm⟩

theorem signedMessage_ok {I session rec m : ByteArray} (h : memoryChallengerMessage I session rec = .ok m) :
    I.size = 16 ∧ m = Sha512.hash (field MEMORY_CHALLENGER_TAG.toUTF8 ++ I ++ field (field session) ++ field rec) := by
  unfold memoryChallengerMessage at h
  split at h
  · simp [throw, throwThe, MonadExceptOf.throw] at h
  · rename_i hI
    simp only [pure, Except.pure, Except.ok.injEq] at h
    exact ⟨not_not.mp hI, h.symm⟩

/-- The session input, with records digests of one width, determines the header and the digest. -/
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
