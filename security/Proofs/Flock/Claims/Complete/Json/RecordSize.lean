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

open Flock Transcript JsonRT
open Lean (Json)

/-! ## Values -/

theorem size_le_length (s : String) : s.utf8ByteSize ≤ 4 * s.length := by
  have h : s.utf8ByteSize ≤ 4 * s.toList.length := by simpa using ofList_size_le s.toList
  simpa only [String.length_toList] using h

theorem jsz_toJson_le {n : ℕ} (hn : n < 10 ^ 20) : jsz (Lean.toJson n) ≤ 80 := by
  have h1 : jsz (Lean.toJson n) = (Int.repr (n : Int)).utf8ByteSize := rfl
  rw [Int.repr_eq_ite, if_pos (by omega)] at h1
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
  have := List.sum_le_length_nsmul (a.toList.map fun j => jsz j + 1) (B + 1) (by
    intro x hx
    obtain ⟨j, hj, rfl⟩ := List.mem_map.1 hx
    have := h j (Array.mem_toList_iff.1 hj)
    omega)
  simp only [List.length_map, Array.length_toList, smul_eq_mul] at this
  omega

theorem sum_first_le {B : ℕ} : ∀ (l : List Json) (F : ℕ),
    (∀ (k : ℕ) (hk : k < l.length), jsz (l[k]'hk) ≤ B + (if k = 0 then F else 0)) →
      (l.map fun j => jsz j + 1).sum ≤ l.length * (B + 1) + F
  | [], F, _ => by simp
  | a :: l, F, h => by
    have ha : jsz a ≤ B + F := by simpa using h 0 (by simp)
    have ht := sum_first_le l 0 (fun k hk => by
      have := h (k + 1) (by simp; omega)
      simpa using this)
    simp only [List.map_cons, List.sum_cons, List.length_cons]
    rw [Nat.add_mul, one_mul]
    omega

/-- An array whose elements weigh at most `B`, the first at most `B + F`. -/
theorem jsz_arr_le_first (a : Array Json) {B F : ℕ}
    (h : ∀ (k : ℕ) (hk : k < a.size), jsz (a[k]'hk) ≤ B + (if k = 0 then F else 0)) :
    jsz (.arr a) ≤ a.size * (B + 1) + F + 2 := by
  rw [jsz_arr]
  have := sum_first_le (B := B) a.toList F (fun k hk => by
    simpa using h k (by simpa using hk))
  simp only [List.length_map, Array.length_toList] at this ⊢
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
      first | decide +kernel | decide)
  have e0 := jsz_mkObj_keys (roundList_distinct spec h k r) hK
  have e1 := jsz_toJson_le hk
  have e2 := jsz_toJson_le hn
  have e3 := jsz_toJson_le hm
  have e4 := jsz_dg_le hh spec (r.bytes.getD ByteArray.empty)
  have e5 := jsz_coins r.coins
  have e6 := jsz_hexJ (r.bytes.getD ByteArray.empty)
  generalize hW : jsz (Json.mkObj (roundList spec h k r)) = W at e0
  unfold roundObj
  rw [hW]
  simp only [roundList, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, List.length_cons,
    List.length_nil] at e0
  omega

theorem jsz_streamObj_le {h : ByteArray → ByteArray} (hh : ∀ b, (h b).size = 64) (spec : SessionSpec)
    (rootB : Array ByteArray) (q : (Nat × Nat × String) × Array Round) {M E : ℕ} (hq : q.2.size < 10 ^ 20)
    (hr : ∀ (k : ℕ) (hk : k < q.2.size), (q.2[k]'hk).n < 10 ^ 20 ∧
      ((q.2[k]'hk).bytes.getD ByteArray.empty).size < 10 ^ 20 ∧
      579 * (q.2[k]'hk).coins.size + 36 * ((q.2[k]'hk).bytes.getD ByteArray.empty).size ≤
        M + (if k = 0 then 36 * E else 0)) :
    jsz (streamObj spec h rootB q) ≤
      18 * (q.1.2.2.length + (spec.domain spec.tables[q.1.1]! q.1.2.1).length) + q.2.size * (M + 4747) + 36 * E +
        4500 := by
  have hK : ∀ p ∈ streamList spec h rootB q, p.1.length ≤ 16 :=
    keys_le (by rcases dk_cases spec with e | e <;> simp only [streamList, e, List.map_cons, List.map_nil] <;>
      first | decide +kernel | decide)
  have e0 := jsz_mkObj_keys (streamList_distinct spec h rootB q) hK
  have e1 := jsz_toJson_le (n := 0) (by decide)
  have e2 := jsz_dg_le hh spec rootB[q.1.1]!
  have e3 := jsz_arr_le_first ((Array.range q.2.size).map fun k => roundObj spec h k q.2[k]!) (B := M + 4746)
    (F := 36 * E) (by
    intro k hk
    have hk' : k < q.2.size := by simpa using hk
    simp only [Array.getElem_map, Array.getElem_range]
    rw [getElem!_pos q.2 k hk']
    obtain ⟨h1, h2, h3⟩ := hr k hk'
    have := jsz_roundObj_le hh spec (k := k) (by omega) h1 h2
    split_ifs at h3 ⊢ <;> omega)
  rw [Array.size_map, Array.size_range] at e3
  generalize hW : jsz (Json.mkObj (streamList spec h rootB q)) = W at e0
  unfold streamObj
  rw [hW]
  simp only [streamList, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, List.length_cons,
    List.length_nil, jsz_str] at e0
  have e4 : (18 * ("" : String).length + 2) = 2 := by simp
  rw [show M + 4746 + 1 = M + 4747 by omega] at e3
  omega

/-! ## The link, a proof file's digest and the record -/

theorem jsz_mkObj_nil : jsz (Json.mkObj []) ≤ 2 :=
  (jsz_mkObj_le (by simp)).trans (by simp)

theorem jsz_mkObj_one (k : String) (v : Json) : jsz (Json.mkObj [(k, v)]) ≤ 18 * k.length + jsz v + 6 :=
  (jsz_mkObj_le (by simp)).trans (by
    simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil]
    omega)

theorem jsz_points_le {points : Array (Array F128)} {Q : ℕ} (hp : ∀ p ∈ points, p.size ≤ Q) :
    jsz (.arr (points.map fun p => .arr (p.map f128J))) ≤ points.size * (579 * Q + 2 + 1) + 2 := by
  have := jsz_arr_le (points.map fun p => .arr (p.map f128J)) (B := 579 * Q + 2) (by
    intro j hj
    obtain ⟨p, hp', rfl⟩ := Array.mem_map.1 hj
    have := jsz_coins p
    have := hp p hp'
    omega)
  rwa [Array.size_map] at this

theorem jsz_linkObj_le {h : ByteArray → ByteArray} (hh : ∀ b, (h b).size = 64) {spec : SessionSpec} {t : String}
    (ht : spec.tables = #[t]) (rb : ByteArray) {points : Array (Array F128)} {Q : ℕ}
    (hp : ∀ p ∈ points, p.size ≤ Q) :
    jsz (linkObj spec h #[rb] points) ≤ 72 * spec.sigma.size + 36 * rb.size + 36 * t.length +
      points.size * (579 * Q + 2 + 1) + 579 * spec.points + 8000 := by
  have hK : ∀ p ∈ linkList spec h #[rb] points, p.1.length ≤ 16 :=
    keys_le (by rcases dk_cases spec with e | e <;> simp only [linkList, e, List.map_cons, List.map_nil] <;>
      first | decide +kernel | decide)
  have e0 := jsz_mkObj_keys (linkList_distinct spec h #[rb] points) hK
  have e1 := jsz_hexJ spec.sigma
  have e2 : jsz (Json.mkObj (rootBList spec id #[rb])) ≤ 18 * t.length + 36 * rb.size + 8 := by
    rw [rootBList_one ht]
    show jsz (Json.mkObj [(t, hexJ rb)]) ≤ _
    have h1 := jsz_mkObj_one t (hexJ rb)
    have h2 := jsz_hexJ rb
    omega
  have e3 : jsz (Json.mkObj (rootBList spec (dgOf spec h) #[rb])) ≤ 18 * t.length + 2312 := by
    rw [rootBList_one ht]
    have := jsz_mkObj_one t (hexJ (dgOf spec h rb))
    have := jsz_dg_le hh spec rb
    omega
  have e4 := jsz_dg_le hh spec spec.sigma
  have e5 := jsz_mkObj_nil
  have e6 := jsz_points_le hp
  have e7 := jsz_coins (yOf spec)
  have e8 : (yOf spec).size = spec.points := by simp [yOf, ht]
  have e9 : jsz (Json.str "exchange") = 146 := by decide +kernel
  generalize hW : jsz (Json.mkObj (linkList spec h #[rb] points)) = W at e0
  unfold linkObj
  rw [hW]
  simp only [linkList, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, List.length_cons,
    List.length_nil] at e0
  omega

theorem jsz_proofObj_le {h : ByteArray → ByteArray} (hh : ∀ b, (h b).size = 64) (spec : SessionSpec)
    (q : (Nat × Nat × String) × ByteArray) : jsz (proofObj spec h q) ≤ 18 * q.1.2.2.length + 2900 := by
  have hK : ∀ p ∈ proofList spec h q, p.1.length ≤ 16 :=
    keys_le (by rcases dk_cases spec with e | e <;> simp only [proofList, e, List.map_cons, List.map_nil] <;>
      first | decide +kernel | decide)
  have e0 := jsz_mkObj_keys (proofList_distinct spec h q) hK
  have e1 := jsz_dg_le hh spec q.2
  generalize hW : jsz (Json.mkObj (proofList spec h q)) = W at e0
  unfold proofObj
  rw [hW]
  simp only [proofList, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, List.length_cons,
    List.length_nil, jsz_str] at e0
  omega

theorem mem_zip_of {α β : Type} {as : Array α} {bs : Array β} {p : α × β} (h : p ∈ as.zip bs) :
    p.1 ∈ as ∧ p.2 ∈ bs := by
  rw [← Array.mem_toList_iff, Array.toList_zip] at h
  have := List.of_mem_zip h
  exact ⟨Array.mem_toList_iff.1 this.1, Array.mem_toList_iff.1 this.2⟩

theorem zip_size_le {α β : Type} (as : Array α) (bs : Array β) : (as.zip bs).size ≤ as.size := by
  rw [Array.size_zip]
  omega

/-- A stream's names: its own and its table's domain label, each read from the spec. -/
def NamesLe (spec : SessionSpec) (L : ℕ) : Prop :=
  ∀ q ∈ spec.repStreams, q.2.2.length + (spec.domain spec.tables[q.1]! q.2.1).length ≤ L

/-- Each recorded stream has at most `R` rounds, each of at most `M` in coins and message bytes, the first of at most
`M + 36 E`. -/
def StreamsLe (streams : Array (Array Round)) (R M E : ℕ) : Prop :=
  ∀ rs ∈ streams, rs.size ≤ R ∧
    ∀ (k : ℕ) (hk : k < rs.size), (rs[k]'hk).n < 10 ^ 20 ∧ ((rs[k]'hk).bytes.getD ByteArray.empty).size < 10 ^ 20 ∧
      579 * (rs[k]'hk).coins.size + 36 * ((rs[k]'hk).bytes.getD ByteArray.empty).size ≤
        M + (if k = 0 then 36 * E else 0)

/-- **The plain record's values weigh at most** what its spec's text, its root, its points and its streams do. -/
theorem recordList_sum_le {h : ByteArray → ByteArray} (hh : ∀ b, (h b).size = 64) {spec : SessionSpec} {t : String}
    (ht : spec.tables = #[t]) (rb : ByteArray) {points : Array (Array F128)} {Q : ℕ}
    (hp : ∀ p ∈ points, p.size ≤ Q) {streams : Array (Array Round)} {R M E L : ℕ} (hR : R < 10 ^ 20)
    (hs : StreamsLe streams R M E) (hL : NamesLe spec L) (proofs : Array ByteArray) :
    ((recordList spec h #[rb] points streams proofs).map fun p => jsz p.2).sum ≤
      18 * spec.hello.length + jsz (spec.unitSample.getD Json.null) + 72 * spec.sigma.size + 36 * rb.size +
        36 * t.length + points.size * (579 * Q + 2 + 1) + 579 * spec.points +
        spec.repStreams.size * (18 * L + R * (M + 4747) + 36 * E + 4500 + 1) +
        spec.repStreams.size * (18 * L + 2900 + 1) + 12000 := by
  have e1 := jsz_linkObj_le hh ht rb hp
  have e2 := jsz_dg_le hh spec (lbOf spec #[rb] points)
  have e3 := jsz_arr_le ((spec.repStreams.zip streams).map (streamObj spec h #[rb]))
    (B := 18 * L + R * (M + 4747) + 36 * E + 4500) (by
      intro j hj
      obtain ⟨q, hq, rfl⟩ := Array.mem_map.1 hj
      obtain ⟨hq1, hq2⟩ := mem_zip_of hq
      obtain ⟨hsz, hr⟩ := hs q.2 hq2
      have hl := hL q.1 hq1
      have := jsz_streamObj_le hh spec #[rb] q (M := M) (E := E) (by omega) hr
      have := Nat.mul_le_mul_right (M + 4747) hsz
      omega)
  have e4 := jsz_arr_le ((spec.repStreams.zip proofs).map (proofObj spec h)) (B := 18 * L + 2900) (by
      intro j hj
      obtain ⟨q, hq, rfl⟩ := Array.mem_map.1 hj
      have hl := hL q.1 (mem_zip_of hq).1
      have := jsz_proofObj_le hh spec q
      omega)
  rw [Array.size_map] at e3 e4
  have e5 := Nat.mul_le_mul_right (18 * L + R * (M + 4747) + 36 * E + 4500 + 1) (zip_size_le spec.repStreams streams)
  have e6 := Nat.mul_le_mul_right (18 * L + 2900 + 1) (zip_size_le spec.repStreams proofs)
  have e7 : jsz (Json.mkObj [("require_link", Json.bool true)]) ≤ 18 * 12 + 5 + 6 :=
    jsz_mkObj_one "require_link" (Json.bool true)
  have e8 : jsz (Json.str "flock-live-session/v1") = 380 := by decide +kernel
  have e9 : jsz (Json.str "exchange") = 146 := by decide +kernel
  simp only [recordList, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, jsz_str]
  rw [jsz_str] at e8 e9
  have e10 : jsz Json.null = 4 := rfl
  omega

end FlockComplete
