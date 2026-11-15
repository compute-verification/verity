module

public import Proofs.Flock.Claims.Complete.Json.Size
public import Proofs.Flock.Claims.Complete.Json.Hex
import Proofs.Flock.Bytes
import all Lean.Data.Json.Basic

/-!
# The record's objects weigh a bounded amount

Each value the record writes has a weight (`jsz`) read off what it holds: a number below `10^20` at most 80
(`jsz_toJson_le`), a byte string's hex 36 per byte and two quotes (`jsz_hexJ`), a coin 578 (`jsz_f128J`), a digest's
hex at most 2306 (`dgOf_size_le`); an object of distinct keys of at most sixteen characters weighs its values, 292 per
field and two braces (`jsz_mkObj_keys`), and an array its elements, one per comma and two brackets (`jsz_arr_le`).
A round then weighs at most `579` per coin and `36` per message byte above a constant (`jsz_roundObj_le`), and a rep
stream its name, its label and its rounds (`jsz_streamObj_le`).
-/

@[expose] public section

namespace FlockComplete

open Flock Transcript
open Lean (Json)

/-! ## Values -/

theorem size_le_length (s : String) : s.utf8ByteSize ≤ 4 * s.length := by
  simpa using ofList_size_le s.toList

theorem jsz_toJson_le {n : ℕ} (hn : n < 10 ^ 20) : jsz (Lean.toJson n) ≤ 80 := by
  have h1 : jsz (Lean.toJson n) = (Int.repr (n : Int)).utf8ByteSize := rfl
  rw [Int.repr_eq_if, if_pos (by omega)] at h1
  simp only [Int.toNat_natCast] at h1
  have h2 := (Nat.length_repr_le_iff (n := n) (k := 20) (by decide)).2 hn
  have h3 := size_le_length (Nat.repr n)
  omega

theorem hexOf_len (b : ByteArray) : (hexOf b).length = 2 * b.size := by
  rw [← String.length_toList, Properties.NetworkRecords.Bytes.hexOf_toList, List.length_flatMap]
  simp only [Properties.NetworkRecords.Bytes.hexOf_length, List.map_const', List.sum_replicate_nat,
    Array.length_toList, ByteArray.size]
  omega

theorem jsz_hexJ (b : ByteArray) : jsz (hexJ b) = 36 * b.size + 2 := by
  show 18 * (hexOf b).length + 2 = _
  rw [hexOf_len]
  omega

theorem toHex_length (a : F128) : a.toHex.length = 32 := by
  rw [← String.length_toList, JsonRT.toHex_toList]
  simp [JsonRT.be8]

theorem jsz_f128J (a : F128) : jsz (f128J a) = 578 := by
  show 18 * a.toHex.length + 2 = _
  rw [toHex_length]

theorem sha256_size (m : ByteArray) : (Sha256.hash m).size = 32 := by
  unfold Sha256.hash
  exact Array.size_ofFn

theorem dgOf_size_le {h : ByteArray → ByteArray} (hh : ∀ b, (h b).size = 64) (spec : SessionSpec) (b : ByteArray) :
    (dgOf spec h b).size ≤ 64 := by
  unfold dgOf
  split
  · rw [hh]
  · rw [sha256_size]; omega

theorem jsz_dg_le {h : ByteArray → ByteArray} (hh : ∀ b, (h b).size = 64) (spec : SessionSpec) (b : ByteArray) :
    jsz (hexJ (dgOf spec h b)) ≤ 2306 := by
  rw [jsz_hexJ]
  have := dgOf_size_le hh spec b
  omega

theorem jsz_str (s : String) : jsz (Json.str s) = 18 * s.length + 2 := rfl

/-! ## Arrays and objects -/

theorem jsz_arr_le (a : Array Json) {B : ℕ} (h : ∀ j ∈ a, jsz j ≤ B) : jsz (.arr a) ≤ a.size * (B + 1) + 2 := by
  rw [jsz_arr]
  have := List.sum_le_card_nsmul (a.toList.map fun j => jsz j + 1) (B + 1) (by
    intro x hx
    obtain ⟨j, hj, rfl⟩ := List.mem_map.1 hx
    have := h j (Array.mem_toList_iff.1 hj)
    omega)
  simp only [List.length_map, Array.length_toList, smul_eq_mul] at this
  omega

theorem jsz_coins (c : Array F128) : jsz (.arr (c.map f128J)) ≤ 579 * c.size + 2 := by
  have := jsz_arr_le (c.map f128J) (B := 578) (by
    intro j hj
    obtain ⟨a, -, rfl⟩ := Array.mem_map.1 hj
    rw [jsz_f128J])
  rw [Array.size_map] at this
  omega

theorem keys_le {l : List (String × Json)} {K : ℕ}
    (h : (l.map Prod.fst).all (fun s => decide (s.length ≤ K)) = true) : ∀ p ∈ l, p.1.length ≤ K := by
  intro p hp
  have := List.all_eq_true.1 h p.1 (List.mem_map_of_mem hp)
  simpa using this

/-- An object of distinct keys of at most `K` characters weighs its values and `18 K + 4` per field. -/
theorem jsz_mkObj_keys {l : List (String × Json)} (hd : Distinct l) {K : ℕ} (hK : ∀ p ∈ l, p.1.length ≤ K) :
    jsz (Json.mkObj l) ≤ (l.map fun p => jsz p.2).sum + l.length * (18 * K + 4) + 2 := by
  refine (jsz_mkObj_le hd).trans ?_
  suffices ∀ l' : List (String × Json), (∀ p ∈ l', p.1.length ≤ K) →
      (l'.map fun p => 18 * p.1.length + 4 + jsz p.2).sum ≤
        (l'.map fun p => jsz p.2).sum + l'.length * (18 * K + 4) by
    have := this l hK
    omega
  intro l' h'
  induction l' with
  | nil => simp
  | cons a t ih =>
    simp only [List.map_cons, List.sum_cons, List.length_cons, Nat.add_mul, one_mul]
    have ha := h' a (by simp)
    have ht := ih (fun p hp => h' p (by simp [hp]))
    omega

/-! ## A round and a rep stream -/

theorem jsz_roundObj_le {h : ByteArray → ByteArray} (hh : ∀ b, (h b).size = 64) (spec : SessionSpec) {k : ℕ}
    {r : Round} (hk : k < 10 ^ 20) (hn : r.n < 10 ^ 20) (hm : (r.bytes.getD ByteArray.empty).size < 10 ^ 20) :
    jsz (roundObj spec h k r) ≤ 579 * r.coins.size + 36 * (r.bytes.getD ByteArray.empty).size + 2200 + 2306 + 240 := by
  have hK : ∀ p ∈ roundList spec h k r, p.1.length ≤ 16 :=
    keys_le (by rcases dk_cases spec with e | e <;> simp only [roundList, e, List.map_cons, List.map_nil] <;>
      first | decide | decide +kernel)
  have e0 := jsz_mkObj_keys (roundList_distinct spec h k r) hK
  have e1 := jsz_toJson_le hk
  have e2 := jsz_toJson_le hn
  have e3 := jsz_toJson_le hm
  have e4 := jsz_dg_le hh spec (r.bytes.getD ByteArray.empty)
  have e5 := jsz_coins r.coins
  have e6 := jsz_hexJ (r.bytes.getD ByteArray.empty)
  unfold roundObj
  simp only [roundList, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, List.length_cons,
    List.length_nil] at e0
  omega

theorem jsz_streamObj_le {h : ByteArray → ByteArray} (hh : ∀ b, (h b).size = 64) (spec : SessionSpec)
    (rootB : Array ByteArray) (q : (Nat × Nat × String) × Array Round) {M : ℕ} (hq : q.2.size < 10 ^ 20)
    (hr : ∀ r ∈ q.2, r.n < 10 ^ 20 ∧ (r.bytes.getD ByteArray.empty).size < 10 ^ 20 ∧ 579 * r.coins.size + 36 * (r.bytes.getD ByteArray.empty).size ≤ M) :
    jsz (streamObj spec h rootB q) ≤
      18 * (q.1.2.2.length + (spec.domain spec.tables[q.1.1]! q.1.2.1).length) + q.2.size * (M + 4747) + 4500 := by
  have hK : ∀ p ∈ streamList spec h rootB q, p.1.length ≤ 16 :=
    keys_le (by rcases dk_cases spec with e | e <;> simp only [streamList, e, List.map_cons, List.map_nil] <;>
      first | decide | decide +kernel)
  have e0 := jsz_mkObj_keys (streamList_distinct spec h rootB q) hK
  have e1 := jsz_toJson_le (n := 0) (by decide)
  have e2 := jsz_dg_le hh spec rootB[q.1.1]!
  have e3 := jsz_arr_le ((Array.range q.2.size).map fun k => roundObj spec h k q.2[k]!) (B := M + 4746) (by
    intro j hj
    obtain ⟨k, hk, rfl⟩ := Array.mem_map.1 hj
    have hk' := Array.mem_range.1 hk
    rw [getElem!_pos q.2 k hk']
    obtain ⟨h1, h2, h3⟩ := hr _ (Array.getElem_mem hk')
    have := jsz_roundObj_le hh spec (k := k) (by omega) h1 h2
    omega)
  rw [Array.size_map, Array.size_range] at e3
  unfold streamObj
  simp only [streamList, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, List.length_cons,
    List.length_nil, jsz_str] at e0
  have e4 : (18 * ("" : String).length + 2) = 2 := rfl
  rw [show M + 4746 + 1 = M + 4747 by omega] at e3
  omega

end FlockComplete
