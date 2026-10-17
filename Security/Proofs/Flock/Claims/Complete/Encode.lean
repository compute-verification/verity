import Proofs.Flock.Claims.Deploy

/-!
# The prover's messages read back

The verifier reads the prover's messages with `Rd` (`linkOf`, `finalOf`); the honest prover writes them with the
encoders (`linkBody`, `Final.encode`). Each encoder is read back by its reader, where every length fits its 8-byte word
(`encNat`): `rdNat_encNat`, `rdBytes_encBytes`, `rdStr_encStr`, `rdList_encList`, and so the link message
(`linkOf_linkBody`) and the last message (`finalOf_encode`).
-/

open Flock FlockDeploy

namespace FlockComplete

/-- `encNat`'s little-endian digits, folded back: `n` modulo `256^k`. -/
theorem digits_foldr (k : ℕ) : ∀ n : ℕ,
    ((List.range k).map fun i => (n / 256 ^ i).toUInt8).foldr (fun b acc => b.toNat + 256 * acc) 0 = n % 256 ^ k := by
  induction k with
  | zero => intro n; simp [Nat.mod_one]
  | succ k ih =>
    intro n
    rw [List.range_succ_eq_map, List.map_cons, List.map_map, List.foldr_cons]
    have e : ((fun i => (n / 256 ^ i).toUInt8) ∘ Nat.succ) = fun i => (n / 256 / 256 ^ i).toUInt8 := by
      funext i
      simp only [Function.comp, Nat.succ_eq_add_one, pow_succ', Nat.div_div_eq_div_mul]
    rw [e, ih (n / 256)]
    simp only [pow_zero, Nat.div_one]
    rw [pow_succ', Nat.mod_mul]
    simp

theorem rdTake_append (l rest : List UInt8) : (rdTake l.length).run (l ++ rest) = some (l, rest) := by
  simp [rdTake, StateT.run]

theorem encNat_length (n : ℕ) : (encNat n).length = 8 := by simp [encNat]

theorem rdNat_encNat {n : ℕ} (hn : n < 2 ^ 64) (rest : List UInt8) : rdNat.run (encNat n ++ rest) = some (n, rest) := by
  have h := rdTake_append (encNat n) rest
  rw [encNat_length] at h
  simp only [rdNat, StateT.run_bind, h]
  show some ((encNat n).foldr (fun b acc => b.toNat + 256 * acc) 0, rest) = some (n, rest)
  rw [encNat, digits_foldr, Nat.mod_eq_of_lt (by simpa using hn)]

theorem rdBytes_encBytes {b : ByteArray} (hb : b.size < 2 ^ 64) (rest : List UInt8) :
    rdBytes.run (encBytes b ++ rest) = some (b, rest) := by
  have h := rdTake_append b.data.toList rest
  simp only [Array.length_toList, ByteArray.size_data] at h
  simp only [rdBytes, encBytes, List.append_assoc, StateT.run_bind, rdNat_encNat hb, Option.bind_eq_bind,
    Option.bind_some, h]
  rfl

theorem fromUTF8?_toUTF8 (s : String) : String.fromUTF8? s.toUTF8 = some s := by
  have h : s.toUTF8.IsValidUTF8 := by simpa using s.isValidUTF8
  simp only [String.fromUTF8?, h, dite_true]
  rfl

theorem rdStr_encStr {s : String} (hs : s.toUTF8.size < 2 ^ 64) (rest : List UInt8) :
    rdStr.run (encStr s ++ rest) = some (s, rest) := by
  simp only [rdStr, encStr, StateT.run_bind, rdBytes_encBytes hs, Option.bind_eq_bind, Option.bind_some,
    fromUTF8?_toUTF8]
  rfl

theorem rdListN_encList {α : Type} {r : Rd α} {e : α → Claims.Msg} :
    ∀ (l : List α), (∀ a ∈ l, ∀ rest, r.run (e a ++ rest) = some (a, rest)) →
      ∀ rest, (rdListN r l.length).run (l.flatMap e ++ rest) = some (l, rest)
  | [], _, rest => rfl
  | a :: l, hr, rest => by
    simp only [List.length_cons, rdListN, List.flatMap_cons, List.append_assoc, StateT.run_bind,
      hr a List.mem_cons_self, Option.bind_eq_bind, Option.bind_some,
      rdListN_encList l (fun a' ha' => hr a' (List.mem_cons_of_mem a ha')) rest]
    rfl

theorem rdList_encList {α : Type} {r : Rd α} {e : α → Claims.Msg} {l : List α}
    (hr : ∀ a ∈ l, ∀ rest, r.run (e a ++ rest) = some (a, rest)) (hl : l.length < 2 ^ 64) (rest : List UInt8) :
    (rdList r).run (encList e l ++ rest) = some (l, rest) := by
  simp only [rdList, encList, List.append_assoc, StateT.run_bind, rdNat_encNat hl, Option.bind_eq_bind,
    Option.bind_some]
  exact rdListN_encList l hr rest

/-- **The last message decodes**, where its lengths fit their words. -/
theorem finalOf_encode {f : Final} (hs : f.stmt.length < 2 ^ 64) (hr : f.record.toUTF8.size < 2 ^ 64)
    (hp : f.proofs.size < 2 ^ 64) (hb : ∀ b ∈ f.proofs, b.size < 2 ^ 64) : finalOf f.encode = some f := by
  have ht := rdTake_append f.stmt (encStr f.record ++ encList encBytes f.proofs.toList)
  have hl : (rdList rdBytes).run (encList encBytes f.proofs.toList ++ []) = some (f.proofs.toList, []) :=
    rdList_encList (fun b hb' => rdBytes_encBytes (hb b (by simpa using hb'))) (by simpa using hp) []
  rw [List.append_nil] at hl
  unfold finalOf Final.encode
  simp only [Final.read, List.append_assoc, StateT.run_bind, rdNat_encNat hs, Option.bind_eq_bind,
    Option.bind_some, ht, rdStr_encStr hr, hl]
  rfl

/-- **The link message decodes**, where its lengths fit their words. -/
theorem linkOf_linkBody {rootB publics : List ByteArray} (hr : rootB.length < 2 ^ 64) (hp : publics.length < 2 ^ 64)
    (hrb : ∀ b ∈ rootB, b.size < 2 ^ 64) (hpb : ∀ b ∈ publics, b.size < 2 ^ 64) :
    linkOf (0 :: linkBody rootB publics) = some (rootB, publics) := by
  have h1 := rdList_encList (fun b hb => rdBytes_encBytes (hrb b hb)) hr (encList encBytes publics)
  have h2 := rdList_encList (fun b hb => rdBytes_encBytes (hpb b hb)) hp []
  rw [List.append_nil] at h2
  unfold linkOf linkBody
  simp only [StateT.run_bind, h1, h2, Option.bind_eq_bind, Option.bind_some]
  rfl

end FlockComplete
