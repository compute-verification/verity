import Proofs.Flock.Claims.Complete.Replay.LigeritoRep
import Proofs.Flock.Soundness.CompiledLock
import Proofs.Flock.Soundness.Refine.Ligerito

/-!
# Every level's Merkle opening passes at every honest rep

`Run.lgMerkle_of`: once the honest rep, run from the rep's first send on, returns the decoded proof, every level's
Merkle check passes at its query coins (`MerkleOk`), given that the table's scheme returns digests of its width
(`SchemeWidths`).

* **A level's commitment is its Merkle tree** (`commitP_tree`, `Tree`): `2^d` rows of `2^lanes` elements, salts of the
  scheme's size, layer `0` the leaves of the rows under their salts, layer `h + 1` the nodes of layer `h`'s pairs, of
  `2^(d − h)` digests each, and the cap the layer `d − cap`.
* **Merkle completeness** (`merkle_open`): an opening of such a tree (`openP`) at positions below `2^d` passes the
  verifier's check (`merkleCheckWith`) against its cap: each query's siblings climb from its leaf to its cap entry
  (`climb_ok`).
* **The honest openings** (`ligeritoP_opens`, `Opens`): the honest prover's Ligerito, run from round `k`, opens level
  `j`'s commitment at the positions of round `k + qOff sch j`'s coins, which is where the verifier draws them; the
  rounds are counted by `KRuns`, every run of a step from round `k` ending `d` rounds on.
-/

namespace FlockComplete.Replay

open Flock Transcript

namespace Lg

/-! ## A level's commitment, as its Merkle tree -/

theorem mapM_get {α β : Type} [Inhabited β] {f : α → V β} {xs : Array α} {ys : Array β} (h : xs.mapM f = .ok ys) :
    ys.size = xs.size ∧ ∀ i (hi : i < xs.size), f xs[i] = .ok ys[i]! := by
  have hs := Array.SatisfiesM_mapM' (as := xs) (f := f) (p := fun i b => f xs[i] = .ok b)
    fun i => SatisfiesM_Except_eq.2 fun b hb => hb
  obtain ⟨e, hq⟩ := SatisfiesM_Except_eq.1 hs ys h
  exact ⟨e, fun i hi => by rw [getElem!_pos ys i (e ▸ hi)]; exact hq i hi⟩

theorem forIn_inv {β : Type} (n : ℕ) (b : β) {f : ℕ → β → V (ForInStep β)} (I : ℕ → β → Prop) (h0 : I 0 b)
    (h : ∀ i < n, ∀ b r, I i b → f i b = .ok r → ∃ b', r = .yield b' ∧ I (i + 1) b') {r : β}
    (hr : forIn [0:n] b f = .ok r) : I n r := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at hr
  simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one] at hr
  suffices ∀ j s (b : β), s + j = n → I s b → forIn (List.range' s j 1) b f = .ok r → I n r from
    this n 0 b (by omega) h0 hr
  intro j
  induction j with
  | zero =>
    intro s b hs hb e
    rw [List.range'_zero, List.forIn_nil] at e
    cases e
    rw [← hs]
    exact hb
  | succ j ih =>
    intro s b hs hb e
    rw [List.range'_succ, List.forIn_cons] at e
    obtain ⟨r', hr', e⟩ := FlockSoundness.Refine.ebind_ok e
    obtain ⟨b', rfl, hb'⟩ := h s (by omega) b r' hb hr'
    exact ih (s + 1) b' (by omega) hb' e

theorem push_get_lt {α : Type} [Inhabited α] {a : Array α} (x : α) {j : ℕ} (h : j < a.size) :
    (a.push x)[j]! = a[j]! := by
  rw [getElem!_pos _ j (by simp; omega), getElem!_pos a j h, Array.getElem_push_lt]

theorem push_get_eq {α : Type} [Inhabited α] (a : Array α) (x : α) : (a.push x)[a.size]! = x := by
  rw [getElem!_pos _ _ (by simp)]
  simp

theorem push_get_eq' {α : Type} [Inhabited α] {a : Array α} {j : ℕ} (x : α) (h : a.size = j) :
    (a.push x)[j]! = x := by
  subst h
  exact push_get_eq a x

theorem mem_of_get {α : Type} [Inhabited α] {a : Array α} {i : ℕ} (h : i < a.size) : a[i]! ∈ a := by
  rw [getElem!_pos a i h]
  exact Array.getElem_mem h

theorem fitBytes_size' (n : ℕ) (b : ByteArray) : (fitBytes n b).size = n := by
  simp only [fitBytes, ByteArray.size_append, ByteArray.size_extract]
  show _ + (Array.replicate _ _).size = n
  rw [Array.size_replicate]
  omega

theorem bytes_empty {b : ByteArray} (h : b.size = 0) : b = ByteArray.empty :=
  ByteArray.ext (Array.eq_empty_of_size_eq_zero h)

/-- **A level's commitment as its Merkle tree**, through `ops`: `2^d` rows of `2^lanes` elements and salts of the
scheme's size (empty for an unsalted scheme), layer `h ≤ d − cap` of `2^(d − h)` digests of `w` bytes, layer `0` the
leaves of the rows under their salts, layer `h + 1` the nodes of layer `h`'s pairs, and the cap the layer `d − cap`. -/
structure Tree (ops : MerkleOps V) (ms : MerkleScheme) (lv : Level) (w : ℕ) (cm : Commit) : Prop where
  rows : cm.rows.size = 2 ^ lv.d
  rowW : ∀ q < 2 ^ lv.d, cm.rows[q]!.size = 2 ^ lv.lanes
  saltW : ∀ q < 2 ^ lv.d, cm.salts[q]!.size = ms.saltBytes
  saltE : ms.saltBytes = 0 → ∀ q < 2 ^ lv.d, cm.salts[q]! = ByteArray.empty
  layer : ∀ h ≤ lv.d - lv.cap, cm.layers[h]!.size = 2 ^ (lv.d - h)
  width : ∀ h ≤ lv.d - lv.cap, ∀ x ∈ cm.layers[h]!, x.size = w
  leaf : ∀ q < 2 ^ lv.d, ops.leaf cm.rows[q]! cm.salts[q]! = .ok (cm.layers[0]!)[q]!
  node : ∀ h < lv.d - lv.cap, ∀ i < 2 ^ (lv.d - h - 1),
    ops.node (cm.layers[h]!)[2 * i]! (cm.layers[h]!)[2 * i + 1]! = .ok (cm.layers[h + 1]!)[i]!
  cap : cm.cap = cm.layers[lv.d - lv.cap]!

/-- **A level's commitment is its Merkle tree**, through `Hashes` whose node and leaf digests are `w` bytes. -/
theorem commitP_tree {hs : Hashes V} {ms : MerkleScheme} {lv : Level} {g : Array F128} {salt : ℕ → ByteArray}
    {cm : Commit} {w : ℕ} (hn : ∀ l r v, (ms.opsWith hs).node l r = .ok v → v.size = w)
    (hlf : ∀ row s v, (ms.opsWith hs).leaf row s = .ok v → v.size = w) (hcap : lv.cap ≤ lv.d)
    (h : commitP hs ms lv g salt = .ok cm) : Tree (ms.opsWith hs) ms lv w cm := by
  unfold commitP at h
  obtain ⟨leaves, hlv, h⟩ := FlockSoundness.Refine.ebind_ok h
  obtain ⟨r, hr, h⟩ := FlockSoundness.Refine.ebind_ok h
  cases h
  obtain ⟨hls, hleaf⟩ := mapM_get hlv
  simp only [Array.size_range] at hls
  have hI := forIn_inv (lv.d - lv.cap) (#[leaves], leaves)
    (fun i (p : Array (Array ByteArray) × Array ByteArray) => p.1.size = i + 1 ∧ p.2 = p.1[i]! ∧
      p.1[0]! = leaves ∧ (∀ j ≤ i, p.1[j]!.size = 2 ^ (lv.d - j) ∧ ∀ x ∈ p.1[j]!, x.size = w) ∧
      ∀ j < i, ∀ k < 2 ^ (lv.d - j - 1),
        (ms.opsWith hs).node (p.1[j]!)[2 * k]! (p.1[j]!)[2 * k + 1]! = .ok (p.1[j + 1]!)[k]!)
    ⟨rfl, rfl, rfl, fun j hj => by
      obtain rfl : j = 0 := by omega
      refine ⟨by simpa using hls, fun x hx => ?_⟩
      have hx' : x ∈ leaves := by simpa using hx
      obtain ⟨q, hq, rfl⟩ := Array.mem_iff_getElem.1 hx'
      have := hleaf q (by simp; omega)
      rw [getElem!_pos leaves q hq] at this
      exact hlf _ _ _ this, fun j hj => absurd hj (Nat.not_lt_zero _)⟩
    (fun i hi p r' hp e => by
      obtain ⟨cur', hc, e⟩ := FlockSoundness.Refine.ebind_ok e
      cases e
      obtain ⟨h1, h2, h3, h4, h5⟩ := hp
      obtain ⟨hcs, hcn⟩ := mapM_get hc
      simp only [Array.size_range] at hcs
      have hp2 : p.2.size = 2 ^ (lv.d - i) := by rw [h2]; exact (h4 i le_rfl).1
      have hd : lv.d - i = lv.d - (i + 1) + 1 := by omega
      have hsz : cur'.size = 2 ^ (lv.d - (i + 1)) := by
        rw [hcs, hp2, hd, Nat.pow_succ, Nat.mul_div_cancel _ (by norm_num)]
      have hget : ∀ j ≤ i, (p.1.push cur')[j]! = p.1[j]! := fun j hj => push_get_lt _ (by omega)
      have hlast : (p.1.push cur')[i + 1]! = cur' := by rw [← h1]; exact push_get_eq _ _
      refine ⟨_, rfl, by simp [h1], hlast.symm, by rw [hget 0 (Nat.zero_le _)]; exact h3, fun j hj => ?_,
        fun j hj k hk => ?_⟩
      · rcases Nat.lt_or_eq_of_le hj with hj | rfl
        · rw [hget j (by omega)]
          exact h4 j (by omega)
        · rw [hlast]
          refine ⟨hsz, fun x hx => ?_⟩
          obtain ⟨k, hk, rfl⟩ := Array.mem_iff_getElem.1 hx
          have := hcn k (by rw [Array.size_range, ← hcs]; exact hk)
          rw [getElem!_pos cur' k hk] at this
          exact hn _ _ _ this
      · rcases Nat.lt_or_eq_of_le (Nat.lt_succ_iff.1 hj) with hj | rfl
        · rw [hget j (by omega), hget (j + 1) (by omega)]
          exact h5 j hj k hk
        · rw [hget j le_rfl, hlast, ← h2]
          have := hcn k (by
            rw [Array.size_range, hp2, hd, Nat.pow_succ, Nat.mul_div_cancel _ (by norm_num)]
            rwa [Nat.sub_sub] at hk)
          simpa using this) hr
  obtain ⟨-, hcur, h0, hL, hN⟩ := hI
  refine ⟨by simp [encodeRows], fun q hq => ?_, fun q hq => ?_, fun h0 q hq => ?_, fun j hj => (hL j hj).1,
    fun j hj => (hL j hj).2, fun q hq => ?_, hN, hcur⟩
  · rw [getElem!_pos _ q (by simp [encodeRows]; omega)]
    simp [encodeRows]
  · rw [getElem!_pos _ q (by simp; omega)]
    simp only [Array.getElem_map, Array.getElem_range]
    exact fitBytes_size' _ _
  · rw [getElem!_pos _ q (by simp; omega)]
    simp only [Array.getElem_map, Array.getElem_range]
    exact bytes_empty (by rw [fitBytes_size', h0])
  · have := hleaf q (by simp; omega)
    simp only [Array.getElem_range] at this
    rw [h0]
    exact this

/-! ## Merkle completeness -/

theorem xor_one_even (i : ℕ) : (2 * i) ^^^ 1 = 2 * i + 1 := by
  simp

theorem xor_one_odd (i : ℕ) : (2 * i + 1) ^^^ 1 = 2 * i := by
  have := Nat.xor_bit true i true 0
  simpa [Nat.bit] using this

theorem shiftRight_lt {p d h : ℕ} (hp : p < 2 ^ d) (hh : h ≤ d) : p >>> h < 2 ^ (d - h) := by
  rw [Nat.shiftRight_eq_div_pow, Nat.div_lt_iff_lt_mul (Nat.two_pow_pos h), ← Nat.pow_add, Nat.sub_add_cancel hh]
  exact hp

/-- **A query's siblings climb from its leaf to its cap entry**: from layer `h₀`'s entry at `p`, the siblings of
layers `h₀` to `D − 1` reach layer `D`'s entry at `p`. -/
theorem climb_ok {ops : MerkleOps V} {layers : Array (Array ByteArray)} {d D : ℕ} (hD : D ≤ d)
    (hnode : ∀ h < D, ∀ i < 2 ^ (d - h - 1),
      ops.node (layers[h]!)[2 * i]! (layers[h]!)[2 * i + 1]! = .ok (layers[h + 1]!)[i]!)
    {p : ℕ} (hp : p < 2 ^ d) : ∀ k h₀, h₀ + k = D →
      climbWith ops (layers[h₀]!)[p >>> h₀]! (p >>> h₀)
        ((List.range' h₀ k).map fun h => (layers[h]!)[(p >>> h) ^^^ 1]!) = .ok (layers[D]!)[p >>> D]!
  | 0, h₀, e => by
    subst e
    rfl
  | k + 1, h₀, e => by
    rw [List.range'_succ, List.map_cons]
    have hlt := shiftRight_lt hp (show h₀ ≤ d by omega)
    have hnext : p >>> (h₀ + 1) = (p >>> h₀) / 2 := by
      rw [Nat.shiftRight_succ]
    have hd : d - h₀ = d - h₀ - 1 + 1 := by omega
    rw [hd, Nat.pow_succ] at hlt
    obtain ⟨i, hi | hi⟩ := Nat.even_or_odd' (p >>> h₀)
    · have hnx := hnode h₀ (by omega) i (by omega)
      have hcl := climb_ok hD hnode hp k (h₀ + 1) (by omega)
      rw [hnext, hi, show 2 * i / 2 = i by omega] at hcl
      unfold climbWith
      rw [hi, xor_one_even, if_pos (by omega), hnx, show 2 * i / 2 = i by omega]
      exact hcl
    · have hnx := hnode h₀ (by omega) i (by omega)
      have hcl := climb_ok hD hnode hp k (h₀ + 1) (by omega)
      rw [hnext, hi, show (2 * i + 1) / 2 = i by omega] at hcl
      unfold climbWith
      rw [hi, xor_one_odd, if_neg (by omega), hnx, show (2 * i + 1) / 2 = i by omega]
      exact hcl

theorem findM_none {α : Type} {f : α → V Bool} : ∀ l : List α, (∀ a ∈ l, f a = .ok false) → l.findM? f = .ok none
  | [], _ => rfl
  | a :: l, h => by
    simp only [List.findM?, h a List.mem_cons_self]
    exact findM_none l fun b hb => h b (List.mem_cons_of_mem _ hb)

theorem foldl_append_toList {β : Type} (blk : ℕ → Array β) :
    ∀ (l : List ℕ) (acc : Array β), (l.foldl (fun acc p => acc ++ blk p) acc).toList =
      acc.toList ++ l.flatMap fun p => (blk p).toList
  | [], acc => by simp
  | p :: l, acc => by
    rw [List.foldl_cons, foldl_append_toList blk l, List.flatMap_cons, Array.toList_append, List.append_assoc]

theorem flatMap_block {β : Type} (B : ℕ → List β) {D : ℕ} (hB : ∀ p, (B p).length = D) :
    ∀ (l : List ℕ) (j : ℕ) (hj : j < l.length), ((l.flatMap B).drop (j * D)).take D = B l[j]
  | [], j, hj => absurd hj (Nat.not_lt_zero _)
  | p :: l, 0, _ => by
    rw [List.flatMap_cons, Nat.zero_mul, List.drop_zero, List.take_left' (hB p)]
    rfl
  | p :: l, j + 1, hj => by
    rw [List.flatMap_cons, List.drop_append, List.drop_eq_nil_of_le (by rw [hB p]; nlinarith), List.nil_append,
      hB p, show (j + 1) * D - D = j * D by rw [Nat.succ_mul, Nat.add_sub_cancel]]
    exact flatMap_block B hB l j (by simp at hj; omega)

/-- **Merkle completeness**: an opening of a level's Merkle tree, with digests of the scheme's width, at positions
below `2^d` passes the verifier's check against the tree's cap. -/
theorem merkle_open {ops : MerkleOps V} {ms : MerkleScheme} {lv : Level} {cm : Commit}
    (ht : Tree ops ms lv ms.digestLen cm) (hcap : lv.cap ≤ lv.d) {pos : Array ℕ} (hp : ∀ p ∈ pos, p < 2 ^ lv.d) :
    merkleCheckWith ops ms cm.cap lv.cap lv.d pos (openP ms cm lv pos).rows (openP ms cm lv pos).siblings
      (openP ms cm lv pos).salts (2 ^ lv.lanes) = .ok () := by
  have hD : lv.d - lv.cap ≤ lv.d := Nat.sub_le _ _
  have hcs : cm.cap.size = 2 ^ lv.cap := by
    rw [ht.cap, ht.layer _ le_rfl]
    congr 1
    omega
  have hcw : ∀ x ∈ cm.cap, x.size = ms.digestLen := by
    rw [ht.cap]
    exact ht.width _ le_rfl
  let blk : ℕ → Array ByteArray := fun p => (Array.range (lv.d - lv.cap)).map fun h => (cm.layers[h]!)[(p >>> h) ^^^ 1]!
  have hblk : ∀ p, (blk p).toList.length = lv.d - lv.cap := by simp [blk]
  have hsib : (openP ms cm lv pos).siblings.toList = pos.toList.flatMap fun p => (blk p).toList := by
    show (pos.foldl (fun acc p => acc ++ blk p) #[]).toList = _
    rw [← Array.foldl_toList, foldl_append_toList]
    simp
  have hsz : (openP ms cm lv pos).siblings.size = pos.size * (lv.d - lv.cap) := by
    rw [← Array.length_toList, hsib, List.length_flatMap]
    simp only [hblk, List.map_const', List.sum_replicate, smul_eq_mul, Array.length_toList]
  have hsw : ∀ x ∈ (openP ms cm lv pos).siblings, x.size = ms.digestLen := by
    intro x hx
    rw [← Array.mem_toList_iff, hsib, List.mem_flatMap] at hx
    obtain ⟨p, hpm, hx⟩ := hx
    simp only [blk, Array.toList_map, List.mem_map, Array.mem_toList_iff, Array.mem_range] at hx
    obtain ⟨h, hh, rfl⟩ := hx
    have hp' := hp p (Array.mem_toList_iff.1 hpm)
    refine ht.width h (by omega) _ (mem_of_get ?_)
    rw [ht.layer h (by omega)]
    have hd : 1 ≤ lv.d - h := by omega
    exact Nat.xor_lt_two_pow (shiftRight_lt hp' (by omega)) (Nat.one_lt_two_pow (by omega))
  have hsibs : ∀ j < pos.size, sibsOf (openP ms cm lv pos).siblings (lv.d - lv.cap) j = (blk pos[j]!).toList := by
    intro j hj
    unfold sibsOf
    rw [Array.toList_extract, hsib, List.extract_eq_take_drop, Nat.add_sub_cancel_left,
      flatMap_block (fun p => (blk p).toList) hblk pos.toList j (by simpa using hj)]
    simp [getElem!_pos pos j hj]
  have hopen : ∀ j < pos.size, opensWith ops ms cm.cap lv.cap lv.d (2 ^ lv.lanes) pos[j]!
      (openP ms cm lv pos).rows[j]! ((openP ms cm lv pos).salts.getD j ByteArray.empty)
      (sibsOf (openP ms cm lv pos).siblings (lv.d - lv.cap) j) = .ok true := by
    intro j hj
    have hpj : pos[j]! < 2 ^ lv.d := hp _ (mem_of_get hj)
    have hrow : (openP ms cm lv pos).rows[j]! = cm.rows[pos[j]!]! := by
      show (pos.map fun p => cm.rows[p]!)[j]! = _
      rw [getElem!_pos _ j (by simpa using hj), Array.getElem_map, getElem!_pos pos j hj]
    have hsalt : (openP ms cm lv pos).salts.getD j ByteArray.empty = cm.salts[pos[j]!]! := by
      by_cases h0 : ms.saltBytes = 0
      · rw [ht.saltE h0 _ hpj]
        simp [openP, h0]
      · simp [openP, h0, Array.getD, hj]
    rw [hrow, hsalt, hsibs j hj]
    unfold opensWith
    have hc : (decide (cm.rows[pos[j]!]!.size = 2 ^ lv.lanes) && decide ((blk pos[j]!).toList.length =
        lv.d - lv.cap) && decide (cm.salts[pos[j]!]!.size = ms.saltBytes)) = true := by
      simp [ht.rowW _ hpj, ht.saltW _ hpj, blk]
    rw [if_pos hc, ht.leaf _ hpj]
    have hcl := climb_ok hD ht.node hpj (lv.d - lv.cap) 0 (Nat.zero_add _)
    rw [Nat.shiftRight_zero] at hcl
    have hbl : (blk pos[j]!).toList = (List.range' 0 (lv.d - lv.cap)).map fun h =>
        (cm.layers[h]!)[(pos[j]! >>> h) ^^^ 1]! := by
      simp [blk, List.range_eq_range']
    have hidx : pos[j]! >>> (lv.d - lv.cap) < cm.cap.size := by
      rw [hcs]
      have := shiftRight_lt hpj hD
      rwa [show lv.d - (lv.d - lv.cap) = lv.cap by omega] at this
    rw [← ht.cap] at hcl
    show (climbWith ops ((cm.layers[0]!)[pos[j]!]!) pos[j]! (blk pos[j]!).toList >>= fun top =>
      pure (decide (top = cm.cap.getD (pos[j]! >>> (lv.d - lv.cap)) ByteArray.empty))) = _
    rw [hbl, hcl]
    show Except.ok (decide (_ = _)) = _
    rw [Array.getD, dif_pos hidx, getElem!_pos cm.cap _ hidx]
    simp
  unfold merkleCheckWith
  have h1 : (cm.cap.size != 2 ^ lv.cap || (openP ms cm lv pos).rows.size != pos.size ||
      (openP ms cm lv pos).siblings.size != pos.size * (lv.d - lv.cap) ||
      (openP ms cm lv pos).salts.size != saltCount ms pos.size) = false := by
    have hrs : (openP ms cm lv pos).rows.size = pos.size := by simp [openP]
    have hss : (openP ms cm lv pos).salts.size = saltCount ms pos.size := by
      by_cases h0 : ms.saltBytes = 0 <;> simp [openP, saltCount, h0]
    rw [hcs, hrs, hsz, hss]
    simp
  have h2 : (cm.cap.any (·.size != ms.digestLen) || (openP ms cm lv pos).siblings.any (·.size != ms.digestLen)) =
      false := by
    simp only [Bool.or_eq_false_iff, Array.any_eq_false]
    exact ⟨fun i hi => by simp [hcw _ (Array.getElem_mem hi)], fun i hi => by simp [hsw _ (Array.getElem_mem hi)]⟩
  rw [h1, h2]
  simp only [Bool.false_eq_true, ↓reduceIte]
  rw [findM_none (List.range pos.size) fun j hj => by
    rw [List.mem_range] at hj
    show (opensWith ops ms cm.cap lv.cap lv.d (2 ^ lv.lanes) pos[j]! _ _ _ >>= fun o => pure !o) = _
    rw [hopen j hj]
    rfl]
  rfl

/-! ## The rounds the honest prover's steps take -/

/-- **The prover's step `y` from round `k` to round `k'`**: every run from round `k` with `L + k` bodies sent ends at
round `k'` with `L + k'` bodies sent, and returns a value with `Q`. -/
def KRuns (L : ℕ) {β : Type} (y : LP PS β) (k k' : ℕ) (Q : β → Prop) : Prop :=
  ∀ (st : LigState) (t : Transcript) (l : Array Claims.Msg) (b : β) (st' : LigState) (t' : Transcript)
    (l' : Array Claims.Msg), t.k = k → l.size = L + k → ((y.run st).run t).go l = (.ok ((b, st'), t'), l') →
    t'.k = k' ∧ l'.size = L + k' ∧ Q b

theorem KRuns.bind {L : ℕ} {α β : Type} {x : LP PS α} {f : α → LP PS β} {k₁ k₂ k₃ : ℕ} {Qa : α → Prop}
    {Q : β → Prop} (hx : KRuns L x k₁ k₂ Qa) (hf : ∀ a, Qa a → KRuns L (f a) k₂ k₃ Q) : KRuns L (x >>= f) k₁ k₃ Q := by
  intro st t l b st' t' l' hk hl h
  obtain ⟨a, st₁, t₁, l₁, h1, h2⟩ := lp_bind_ok h
  obtain ⟨hk₁, hl₁, hQa⟩ := hx st t l a st₁ t₁ l₁ hk hl h1
  exact hf a hQa st₁ t₁ l₁ b st' t' l' hk₁ hl₁ h2

theorem KRuns.pure {L : ℕ} {β : Type} {b : β} {Q : β → Prop} (hb : Q b) {k k' : ℕ} (e : k = k') :
    KRuns L (Pure.pure b : LP PS β) k k' Q := by
  intro st t l b' st' t' l' hk hl h
  obtain ⟨rfl, rfl, rfl, rfl⟩ := ok_inj (show (Except.ok ((b, st), t), l) = _ from h)
  exact ⟨hk.trans e, e ▸ hl, hb⟩

theorem KRuns.mono {L : ℕ} {β : Type} {y : LP PS β} {k k' : ℕ} {Q Q' : β → Prop} (h : KRuns L y k k' Q)
    (hQ : ∀ b, Q b → Q' b) : KRuns L y k k' Q' := by
  intro st t l b st' t' l' hk hl e
  obtain ⟨h1, h2, h3⟩ := h st t l b st' t' l' hk hl e
  exact ⟨h1, h2, hQ b h3⟩

theorem KRuns.cast {L : ℕ} {β : Type} {y : LP PS β} {k₁ k₂ k₁' k₂' : ℕ} {Q : β → Prop} (h : KRuns L y k₁ k₂ Q)
    (e₁ : k₁' = k₁) (e₂ : k₂ = k₂') : KRuns L y k₁' k₂' Q := by
  subst e₁ e₂
  exact h

theorem KRuns.ite {L : ℕ} {β : Type} {c : Prop} [Decidable c] {y z : LP PS β} {k k' : ℕ} {Q : β → Prop}
    (hy : c → KRuns L y k k' Q) (hz : ¬c → KRuns L z k k' Q) : KRuns L (if c then y else z) k k' Q := by
  by_cases h : c
  · rw [if_pos h]; exact hy h
  · rw [if_neg h]; exact hz h

/-- A step whose every run moves only the state and the pending bytes. -/
theorem KRuns.still {L : ℕ} {β : Type} {y : LP PS β} {Q : β → Prop}
    (hy : ∀ st t l, ∃ b st' g, ((y.run st).run t).go l = (.ok ((b, st'), ap g t), l) ∧ Q b) (k : ℕ) :
    KRuns L y k k Q := by
  intro st t l b st' t' l' hk hl h
  obtain ⟨b₀, st₀, g, e, hq⟩ := hy st t l
  rw [e] at h
  obtain ⟨rfl, rfl, rfl, rfl⟩ := ok_inj h
  exact ⟨hk, hl, hq⟩

theorem KRuns.absorb {L : ℕ} (x : TM Unit) (g : ByteArray → ByteArray) (hx : ∀ t : Transcript, x t = .ok ((), ap g t))
    (k : ℕ) : KRuns L (liftT (liftTM x)) k k fun _ => True :=
  KRuns.still (fun st t l => ⟨(), st, g, liftT_go (liftTM_ok (hx t) l) st, trivial⟩) k

theorem KRuns.modify {L : ℕ} (f : LigState → LigState) (k : ℕ) :
    KRuns L (_root_.modify f : LP PS PUnit) k k fun _ => True :=
  KRuns.still (fun st _ _ => ⟨(), f st, id, rfl, trivial⟩) k

theorem KRuns.get {L : ℕ} (k : ℕ) : KRuns L (MonadState.get : LP PS LigState) k k fun _ => True :=
  KRuns.still (fun st _ _ => ⟨st, st, id, rfl, trivial⟩) k

theorem KRuns.sendMsg {L : ℕ} (m : F256 × F256) (k : ℕ) : KRuns L (FlockComplete.sendMsg m) k k fun _ => True :=
  KRuns.still (fun st _ _ => ⟨(), { st with msgs := st.msgs.push m }, pK m, rfl, trivial⟩) k

theorem KRuns.pow {L : ℕ} (nonce : ℕ → ℕ → UInt64) (kind : ℕ) (k : ℕ) :
    KRuns L (powP nonce kind) k k fun _ => True := by
  refine KRuns.still (fun st t l => ?_) k
  rcases kind with _ | _ | kind
  · exact ⟨(), { st with qn := st.qn.push 0 }, pN 0, rfl, trivial⟩
  · exact ⟨(), { st with cn := st.cn.push 0 }, pN 0, rfl, trivial⟩
  · exact ⟨(), { st with cbn := st.cbn.push 0 }, pN 0, rfl, trivial⟩

theorem KRuns.sampleVec {L : ℕ} (resp : ℕ → Claims.Msg) {n : ℕ} (hn : n ≠ 0) (k : ℕ) :
    KRuns L (liftT (sampleVecP (asksS resp) n)) k (k + 1) fun v => v = (Array.range n).map (coinOf (resp (L + k))) := by
  intro st t l b st' t' l' hk hl h
  rw [liftT_go (sampleVecP_go resp hn t l) st] at h
  obtain ⟨rfl, rfl, rfl, rfl⟩ := ok_inj h
  exact ⟨by simp [hk], by rw [Array.size_push, hl, Nat.add_assoc], by rw [hl]⟩

theorem KRuns.sample {L : ℕ} (resp : ℕ → Claims.Msg) (k : ℕ) :
    KRuns L (liftT (sampleP (asksS resp))) k (k + 1) fun _ => True := by
  intro st t l b st' t' l' hk hl h
  rw [liftT_go (sampleP_go resp t l) st] at h
  obtain ⟨rfl, rfl, rfl, rfl⟩ := ok_inj h
  exact ⟨by simp [hk], by rw [Array.size_push, hl, Nat.add_assoc], trivial⟩

theorem KRuns.sampleK {L : ℕ} (resp : ℕ → Claims.Msg) (k : ℕ) :
    KRuns L (liftT (sampleKP (asksS resp))) k (k + 1) fun _ => True := by
  intro st t l b st' t' l' hk hl h
  rw [liftT_go (sampleKP_go resp t l) st] at h
  obtain ⟨rfl, rfl, rfl, rfl⟩ := ok_inj h
  exact ⟨by simp [hk], by rw [Array.size_push, hl, Nat.add_assoc], trivial⟩

/-- **A commitment** takes no round and returns the commitment `commitP` returns. -/
theorem KRuns.commit {L : ℕ} (h : ByteArray → ByteArray) (ms : MerkleScheme) (lv : Level) (g : Array F128)
    (salt : ℕ → ByteArray) (k : ℕ) :
    KRuns L (liftT (StateT.lift (commitP (Hashes.ofSha512 h : Hashes PS) ms lv g salt))) k k
      fun cm => commitP (Hashes.ofSha512 h : Hashes V) ms lv g salt = .ok cm := by
  intro st t l b st' t' l' hk hl e
  have hc := JsonRT.commitP_pureS h ms lv g salt
  have : (((liftT (StateT.lift (commitP (Hashes.ofSha512 h : Hashes PS) ms lv g salt)) : LP PS Commit).run st).run
      t).go l = (match commitP (Hashes.ofSha512 h : Hashes V) ms lv g salt with
        | .ok a => .ok ((a, st), t)
        | .error x => .error x, l) := by
    show (((commitP (Hashes.ofSha512 h : Hashes PS) ms lv g salt >>= fun a => Pure.pure (a, t)) >>= fun q =>
      Pure.pure ((q.1, st), q.2)) : PS _).go l = _
    rw [go_bind, go_bind, hc l]
    cases commitP (Hashes.ofSha512 h : Hashes V) ms lv g salt <;> rfl
  rw [this] at e
  split at e
  · next a ha =>
    obtain ⟨rfl, rfl, rfl, rfl⟩ := ok_inj e
    exact ⟨hk, hl, ha⟩
  · cases e

/-- **A loop over `[0:n]`**: step `j`, from a state with `I j`, goes from round `K j` to `K (j + 1)` and yields one
with `I (j + 1)`. -/
theorem KRuns.loop {L : ℕ} {σ : Type} (body : ℕ → σ → LP PS (ForInStep σ)) (K : ℕ → ℕ) (I : ℕ → σ → Prop) (n : ℕ)
    (hbody : ∀ j < n, ∀ s, I j s → KRuns L (body j s) (K j) (K (j + 1)) fun r => ∃ s', r = .yield s' ∧ I (j + 1) s')
    (s₀ : σ) (h₀ : I 0 s₀) : KRuns L (forIn [:n] s₀ body) (K 0) (K n) (I n) := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one]
  suffices h : ∀ m a s, a + m = n → I a s → KRuns L (forIn (List.range' a m) s body) (K a) (K n) (I n) from
    h n 0 s₀ (by omega) h₀
  intro m
  induction m with
  | zero =>
    intro a s ha hs
    subst ha
    exact KRuns.pure hs rfl
  | succ m ih =>
    intro a s ha hs
    rw [List.range'_succ, List.forIn_cons]
    refine KRuns.bind (hbody a (by omega) s hs) fun r hr => ?_
    obtain ⟨s₁, rfl, hI⟩ := hr
    exact ih (a + 1) s₁ (by omega) hI

theorem KRuns.challenge {L : ℕ} (resp : ℕ → Claims.Msg) (split : Bool) (k : ℕ) :
    KRuns L (challengeP (asksS resp) split) k (k + 1) fun _ => True := by
  unfold challengeP
  refine (KRuns.sampleK resp k).bind fun r _ => (KRuns.modify _ _).bind fun _ _ => ?_
  refine KRuns.ite (fun _ => (KRuns.modify _ _).bind fun _ _ => ?_) fun _ => ?_
  all_goals exact (KRuns.get _).bind fun _ _ => (KRuns.sendMsg _ _).bind fun _ _ => KRuns.pure trivial rfl

/-- **An out-of-domain value** pushed and absorbed takes no round. -/
theorem KRuns.oodPush {L : ℕ} {β : Type} (y : F128) {c : PUnit → LP PS β} {k k' : ℕ} {Q : β → Prop}
    (h : KRuns L (c ()) k k' Q) :
    KRuns L (_root_.modify (fun st : LigState => { st with oods := st.oods.push y }) >>= fun _ =>
      liftT (liftTM (Transcript.absorbF y)) >>= c) k k' Q := by
  intro st t l b st' t' l' hk hl e
  obtain ⟨a, st₁, t₁, l₁, h1, h2⟩ := lp_bind_ok e
  obtain ⟨-, rfl, rfl, rfl⟩ := ok_inj (show (Except.ok (((), { st with oods := st.oods.push y }), t), l) = _ from h1)
  obtain ⟨u, st₂, t₂, l₂, h3, h4⟩ := lp_bind_ok h2
  obtain ⟨rfl, rfl, rfl, rfl⟩ :=
    ok_inj (show (Except.ok (((), { st with oods := st.oods.push y }), ap (pF y) t), l) = _ from h3)
  exact h _ _ _ b st' t' l' (by rw [ap_k]; exact hk) hl h4

theorem KRuns.oodIntro {L : ℕ} (resp : ℕ → Claims.Msg) (nonce : ℕ → ℕ → UInt64) (g : Array F128) {dim : ℕ}
    (hdim : dim ≠ 0) (k : ℕ) : KRuns L (oodIntroP (asksS resp) nonce g dim) k (k + 2) fun _ => True := by
  unfold oodIntroP
  refine (KRuns.sampleVec resp hdim k).bind fun z _ => KRuns.oodPush (mleF g z) ?_
  exact (KRuns.sendMsg _ _).bind fun _ _ => (KRuns.pow nonce 2 _).bind fun _ _ =>
    (KRuns.sample resp _).bind fun _ _ => KRuns.cast (KRuns.modify _ _) rfl (by omega)

/-- **A level's query** opens the commitment at the positions of its first round's coins. -/
theorem KRuns.query {L : ℕ} (resp : ℕ → Claims.Msg) (nonce : ℕ → ℕ → UInt64) (ms : MerkleScheme) (lv : Level)
    (cm : Commit) (hq : lv.q ≠ 0) (k : ℕ) :
    KRuns L (queryP (asksS resp) nonce ms lv cm) k (k + 2)
      fun r => r.1 = openP ms cm lv (positions lv ((Array.range lv.q).map (coinOf (resp (L + k))))) := by
  unfold queryP
  refine (KRuns.pow nonce 0 k).bind fun _ _ => (KRuns.sampleVec resp hq k).bind fun coins hc => ?_
  refine (KRuns.pow nonce 1 _).bind fun _ _ => (KRuns.sampleVec resp (Nat.succ_ne_zero _) _).bind fun _ _ => ?_
  exact KRuns.pure (by rw [hc]) rfl

theorem levelChalsK {L : ℕ} (resp : ℕ → Claims.Msg) (last : Bool) (k : ℕ) :
    KRuns L (forIn [:4] (#[] : Array F256) fun i (f : Array F256) =>
        if (i + 1 == 4) = true then do
          let r ← liftT (sampleKP (asksS resp))
          modify fun st : LigState =>
            { tab := foldK r st.tab, w := foldK r st.w, msgs := st.msgs, oods := st.oods, qn := st.qn, cn := st.cn,
              cbn := st.cbn }
          let d ← get
          modify fun st : LigState =>
            { tab := embedF (splitK st.tab), w := liftU st.w, msgs := st.msgs, oods := st.oods, qn := st.qn,
              cn := st.cn, cbn := st.cbn }
          let st ← get
          FlockComplete.sendMsg (if last = true then roundPoly d.tab d.w else roundPoly st.tab st.w)
          pure (ForInStep.yield d.tab)
        else do
          let _ ← challengeP (asksS resp) false
          pure (ForInStep.yield f))
      k (k + 4) fun _ => True :=
  (KRuns.loop _ (fun j => k + j) (fun _ _ => True) 4
    (fun _ _ _ _ => KRuns.ite
      (fun _ => (KRuns.sampleK resp _).bind fun _ _ => (KRuns.modify _ _).bind fun _ _ => (KRuns.get _).bind fun _ _ =>
        (KRuns.modify _ _).bind fun _ _ => (KRuns.get _).bind fun _ _ =>
        (KRuns.sendMsg _ _).bind fun _ _ => KRuns.pure ⟨_, rfl, trivial⟩ (by omega))
      fun _ => (KRuns.challenge resp false _).bind fun _ _ => KRuns.pure ⟨_, rfl, trivial⟩ (by omega))
    _ trivial).cast (by omega) rfl

theorem KRuns.absorbAll {L : ℕ} (ws : Array F128) (k : ℕ) :
    KRuns L (ForIn.forIn ws PUnit.unit fun w (_ : PUnit) => do
        liftT (liftTM (Transcript.absorbF w))
        Pure.pure (ForInStep.yield PUnit.unit)) k k fun _ => True := by
  have key : ∀ (xs : List F128) (st : LigState) (t : Transcript) (l : Array Claims.Msg),
      (((ForIn.forIn xs PUnit.unit fun w (_ : PUnit) => do
        liftT (liftTM (Transcript.absorbF w))
        Pure.pure (ForInStep.yield PUnit.unit) : LP PS PUnit).run st).run t).go l =
        (.ok ((PUnit.unit, st), ap (fun P => xs.foldl (fun P w => pF w P) P) t), l) := by
    intro xs
    induction xs with
    | nil => intro st t l; rfl
    | cons w ws ih =>
      intro st t l
      rw [List.forIn_cons, StateT.run_bind, StateT.run_bind, go_bind]
      exact ih st (ap (pF w) t) l
  refine KRuns.still (fun st t l => ⟨PUnit.unit, st, fun P => ws.toList.foldl (fun P w => pF w P) P, ?_, trivial⟩) k
  rw [← Array.forIn_toList, key]

/-! ## The honest openings -/

/-- Level `j`'s query positions, drawn at round `k + qOff sch j`. -/
def posAt (sch : Schedule) (cf : ℕ → ℕ → F128) (k j : ℕ) : Array ℕ :=
  positions (sch.levels.getD j default) ((Array.range (sch.levels.getD j default).q).map (cf (k + qOff sch j)))

/-- **Level `j`'s opening is the honest one**: of a commitment `commitP` returned at level `j` with cap `cap`, at the
level's query positions. -/
def LvOpen (hs : Hashes V) (ms : MerkleScheme) (sch : Schedule) (cf : ℕ → ℕ → F128) (k j : ℕ)
    (cap : Array ByteArray) (op : LevelProof) : Prop :=
  ∃ cm g salt, commitP hs ms (sch.levels.getD j default) g salt = .ok cm ∧ cm.cap = cap ∧
    op = openP ms cm (sch.levels.getD j default) (posAt sch cf k j)

/-- **The proof's openings are the honest ones**: level `0`'s opens `cm0` at its query positions and every later
level's is honest (`LvOpen`) against the recursive cap it names. -/
def Opens (hs : Hashes V) (ms : MerkleScheme) (sch : Schedule) (cm0 : Commit) (p : LigeritoProof)
    (cf : ℕ → ℕ → F128) (k : ℕ) : Prop :=
  p.initialProof = openP ms cm0 (sch.levels.getD 0 default) (posAt sch cf k 0) ∧
    p.recursiveCaps.size = sch.levels.size - 1 ∧ p.recursiveProofs.size = sch.levels.size - 1 - 1 ∧
    (∀ i, i + 1 < sch.levels.size - 1 → LvOpen hs ms sch cf k (i + 1) p.recursiveCaps[i]! p.recursiveProofs[i]!) ∧
    LvOpen hs ms sch cf k (sch.levels.size - 1) p.recursiveCaps[sch.levels.size - 1 - 1]!
      ⟨p.finalProof.rows, p.finalProof.siblings, p.finalProof.salts⟩

/-- The level loop's invariant before step `j` of `R`: the next commitment is honest at level `j + 1` and its cap the
last pushed, the inner levels' openings so far are honest, and after the last step so is the final opening. -/
def LoopOk (hs : Hashes V) (ms : MerkleScheme) (sch : Schedule) (cf : ℕ → ℕ → F128) (k R j : ℕ)
    (s : ℕ × Commit × Array (Array ByteArray) × Array LevelProof × FinalProof) : Prop :=
  (j < R → (∃ g salt, commitP hs ms (sch.levels.getD (j + 1) default) g salt = .ok s.2.1) ∧
    s.2.2.1.size = j + 1 ∧ s.2.2.1[j]! = s.2.1.cap ∧ s.2.2.2.1.size = j) ∧
  (∀ i, i + 1 < R → i < j → LvOpen hs ms sch cf k (i + 1) s.2.2.1[i]! s.2.2.2.1[i]!) ∧
  (j = R → s.2.2.1.size = R ∧ s.2.2.2.1.size = R - 1 ∧
    LvOpen hs ms sch cf k R s.2.2.1[R - 1]! ⟨s.2.2.2.2.rows, s.2.2.2.2.siblings, s.2.2.2.2.salts⟩)

/-- **The honest prover's Ligerito opens honestly**: run from round `k` with `L + k` bodies sent, it opens level `j`'s
commitment at the positions of round `k + qOff sch j`'s coins, the replies' coins (`cfOf resp L`). -/
theorem ligeritoP_opens (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (ms : MerkleScheme) (sch : Schedule)
    (T : F128) (cm0 : Commit) (tb b : Array F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64) (L : ℕ)
    (hn : sch.n ≠ 0) (hL : 2 ≤ sch.levels.size) (hq : ∀ j < sch.levels.size, (sch.levels.getD j default).q ≠ 0)
    {t : Transcript} {l : Array Claims.Msg} (hl : l.size = L + t.k) {P : LigeritoProof} {t' : Transcript}
    {l' : Array Claims.Msg}
    (hrun : ((ligeritoP (asksS resp) (Hashes.ofSha512 h) ms sch T cm0 tb b salt nonce).run t).go l =
      (.ok (P, t'), l')) :
    Opens (Hashes.ofSha512 h : Hashes V) ms sch cm0 P (cfOf resp L) t.k := by
  unfold ligeritoP at hrun
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.label_eq _ _) l), bind_go (liftTM_ok (Flock.RecAlgebra.absorbF_eq _ _) l),
    bind_go (liftTM_ok (Flock.RecAlgebra.absorbBytes_eq _ _) l)] at hrun
  dsimp only at hrun
  rw [StateT.run_bind] at hrun
  obtain ⟨⟨⟨⟨caps, rproofs, fin, op0⟩, stF⟩, t₁⟩, l₁, h1, h2⟩ := JsonRT.go_bind_ok hrun
  obtain ⟨h21, rfl⟩ := Prod.mk.inj h2
  obtain ⟨hP, rfl⟩ := Prod.mk.inj (Except.ok.inj h21)
  subst hP
  set R := sch.levels.size - 1 with hR
  set k := t.k
  set hsV : Hashes V := Hashes.ofSha512 h
  have hR1 : 1 ≤ R := by omega
  have key := (?prog : KRuns (β := Array (Array ByteArray) × Array LevelProof × FinalProof × LevelProof) L _ k
    (k + sch.k0 + 9 + 11 * (R - 1) + 7) fun r =>
      r.2.2.2 = openP ms cm0 (sch.levels.getD 0 default) (posAt sch (cfOf resp L) k 0) ∧
        LoopOk hsV ms sch (cfOf resp L) k R R (0, default, r.1, r.2.1, r.2.2.1)) _ _ l _ _ _ _ (by rfl) hl h1
  case prog =>
    refine (KRuns.sampleVec resp hn _).bind fun z _ => KRuns.oodPush (mleF tb z) ?_
    refine (KRuns.pow nonce 2 _).bind fun _ _ => (KRuns.sample resp _).bind fun _ _ =>
      (KRuns.modify _ _).bind fun _ _ => (KRuns.get _).bind fun _ _ => (KRuns.sendMsg _ _).bind fun _ _ => ?_
    refine KRuns.bind ((KRuns.loop _ (fun j => k + 1 + 1 + j) (fun _ _ => True) sch.k0
      (fun j _ s _ => (KRuns.challenge resp _ _).bind fun _ _ => KRuns.pure ⟨(), rfl, trivial⟩ (by omega)) _
      trivial).cast (by omega) rfl) fun _ _ => ?_
    refine (KRuns.get _).bind fun d _ => (KRuns.commit h ms _ _ _ _).bind fun cm1 hcm1 => ?_
    refine (KRuns.absorb _ (pBytes _) (fun t => Flock.RecAlgebra.absorbBytes_eq _ t) _).bind fun _ _ => ?_
    refine (KRuns.oodIntro resp nonce _ (Nat.succ_ne_zero _) _).bind fun _ _ =>
      (KRuns.oodIntro resp nonce _ (Nat.succ_ne_zero _) _).bind fun _ _ => ?_
    refine (KRuns.query resp nonce ms _ cm0 (hq 0 (by omega)) _).bind fun x hx => ?_
    refine (KRuns.sendMsg _ _).bind fun _ _ => (KRuns.pow nonce 2 _).bind fun _ _ =>
      (KRuns.sample resp _).bind fun _ _ => (KRuns.modify _ _).bind fun _ _ => ?_
    have hx0 : x.1 = openP ms cm0 (sch.levels.getD 0 default) (posAt sch (cfOf resp L) k 0) := by
      rw [hx]
      unfold posAt cfOf qOff
      rw [if_pos (by omega)]
      congr 5
      omega
    refine KRuns.bind (KRuns.cast (KRuns.loop (σ := ℕ × Commit × Array (Array ByteArray) × Array LevelProof × FinalProof)
      _ (fun j => if j < R then k + sch.k0 + 9 + 11 * j else k + sch.k0 + 9 + 11 * (R - 1) + 7)
      (LoopOk hsV ms sch (cfOf resp L) k R) R ?body _ ⟨fun _ => ⟨⟨_, _, hcm1⟩, rfl, rfl, rfl⟩,
        fun i _ hi => absurd hi (Nat.not_lt_zero _), fun h0 => absurd h0 (by omega)⟩)
      (by rw [if_pos (by omega)]; omega) (by rw [if_neg (by omega)])) fun σ hσ => KRuns.pure ⟨hx0, ?fin⟩ rfl
    case fin =>
      obtain ⟨-, h2, h3⟩ := hσ
      exact ⟨fun h => absurd h (lt_irrefl _), h2, h3⟩
    case body =>
      intro j hj s hs
      obtain ⟨hcur, hpre, -⟩ := hs
      obtain ⟨⟨g₀, salt₀, hcm⟩, hcs, hcl, hrs⟩ := hcur hj
      rw [if_pos hj]
      by_cases hj1 : j + 1 < R
      · rw [if_pos hj1]
        refine (levelChalsK resp _ _).bind fun f _ => ?_
        refine KRuns.ite (fun _ => ?_) fun hne => absurd (by simp; omega) hne
        refine (KRuns.commit h ms _ _ _ _).bind fun cmN hcmN => ?_
        refine (KRuns.absorb _ (pBytes _) (fun t => Flock.RecAlgebra.absorbBytes_eq _ t) _).bind fun _ _ => ?_
        refine (KRuns.oodIntro resp nonce _ (Nat.succ_ne_zero _) _).bind fun _ _ =>
          (KRuns.oodIntro resp nonce _ (Nat.succ_ne_zero _) _).bind fun _ _ => ?_
        refine (KRuns.query resp nonce ms _ s.2.1 (hq (j + 1) (by omega)) _).bind fun q hq' => ?_
        refine (KRuns.sendMsg _ _).bind fun _ _ => (KRuns.pow nonce 2 _).bind fun _ _ =>
          (KRuns.sample resp _).bind fun _ _ => (KRuns.modify _ _).bind fun _ _ => KRuns.pure ?_ (by omega)
        have hop : LvOpen hsV ms sch (cfOf resp L) k (j + 1) (s.2.2.1.push cmN.cap)[j]!
            ((s.2.2.2.1.push q.1)[j]!) := by
          rw [push_get_lt _ (by omega), push_get_eq' _ hrs, hcl]
          refine ⟨s.2.1, g₀, salt₀, hcm, rfl, ?_⟩
          rw [hq']
          unfold posAt cfOf qOff
          rw [if_pos (by omega)]
          congr 5
          omega
        refine ⟨_, rfl, fun _ => ⟨⟨_, _, hcmN⟩, by simp [hcs], ?_, by simp [hrs]⟩, fun i hi1 hi => ?_,
          fun h => absurd h (by omega)⟩
        · rw [← hcs]
          exact push_get_eq _ _
        · rcases Nat.lt_or_eq_of_le (Nat.lt_succ_iff.1 hi) with hi | rfl
          · rw [push_get_lt _ (by omega), push_get_lt _ (by omega)]
            exact hpre i hi1 hi
          · exact hop
      · rw [if_neg hj1]
        have hjR : j + 1 = R := by omega
        refine (levelChalsK resp _ _).bind fun f _ => ?_
        refine KRuns.ite (fun hc => absurd hc (by simp [hjR])) fun _ => ?_
        refine (KRuns.absorbAll _ _).bind fun _ _ => ?_
        refine (KRuns.query resp nonce ms _ s.2.1 (hq _ (by omega)) _).bind fun q hq' => ?_
        refine (KRuns.pow nonce 2 _).bind fun _ _ => (KRuns.sample resp _).bind fun _ _ => KRuns.pure ?_ (by omega)
        refine ⟨_, rfl, fun h => absurd h (by omega), fun i hi1 hi => hpre i hi1 (by omega), fun _ => ?_⟩
        dsimp only
        refine ⟨by omega, by omega, ?_⟩
        rw [show R - 1 = j by omega, hcl]
        refine ⟨s.2.1, g₀, salt₀, by rw [← hjR]; exact hcm, rfl, ?_⟩
        rw [show (⟨q.1.rows, q.1.siblings, q.1.salts⟩ : LevelProof) = q.1 from rfl, hq']
        unfold posAt cfOf qOff
        rw [if_neg (by omega)]
        congr 5
        omega
  obtain ⟨-, -, hop0, -, hloop, hfin⟩ := key
  obtain ⟨hcs, hrs, hlast⟩ := hfin rfl
  exact ⟨hop0, hcs, hrs, fun i hi => hloop i hi (by omega), hlast⟩

/-! ## Every level's check passes at the honest openings -/

/-- A level's query positions are below `2^d` at a level with fast100's seam. -/
theorem positions_lt_d {lv : Level} (hl : FlockSoundness.Refine.LevelSeam lv) (coins : Array F128) :
    ∀ p ∈ positions lv coins, p < 2 ^ lv.d := by
  intro p hp
  have hlt := FlockSoundness.positions_lt FlockSoundness.execArith (FlockSoundness.Refine.levelOf lv).queries
    (FlockSoundness.Refine.levelOf lv).logLen (FlockSoundness.Refine.vecOf lv.q coins) p
  rw [FlockSoundness.Refine.positions_eq lv hl coins, ← hl.2.2.2.2] at hlt
  exact hlt (Array.mem_toList_iff.2 hp)

/-- fast100's levels cap at most their depth. -/
theorem fast100_cap {m : ℕ} {s : Schedule} (hs : fast100 m = some s) : ∀ lv ∈ s.levels, lv.cap ≤ lv.d := by
  by_cases hm : m < 40
  · have key : ∀ m : Fin 40, ∀ s ∈ fast100 m.val, (s.levels.all fun lv => decide (lv.cap ≤ lv.d)) = true := by
      decide +kernel
    intro lv hlv
    obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 hlv
    simpa using Array.all_eq_true.1 (key ⟨m, hm⟩ s hs) i hi
  · rw [FlockSoundness.Refine.exec_fast100_none m (by omega)] at hs
    cases hs

theorem getD_mem {α : Type} {a : Array α} {j : ℕ} (d : α) (h : j < a.size) : a.getD j d ∈ a := by
  rw [Array.getD, dif_pos h]
  exact Array.getElem_mem h

/-- **Every level's Merkle check passes at honest openings** (`MerkleOk`): level `0`'s of a commitment `cm0` of level
`0`, every later level's of its own (`Opens`), through a scheme whose node and leaf digests are its width, at levels
that cap at most their depth and query below `2^d`. -/
theorem merkleOk_of_opens {hs : Hashes V} {ms : MerkleScheme} {sch : Schedule} {cm0 : Commit} {p : LigeritoProof}
    {cf : ℕ → ℕ → F128} {k : ℕ} (hn : ∀ l r v, (ms.opsWith hs).node l r = .ok v → v.size = ms.digestLen)
    (hlf : ∀ row s v, (ms.opsWith hs).leaf row s = .ok v → v.size = ms.digestLen)
    (hcap : ∀ j < sch.levels.size, (sch.levels.getD j default).cap ≤ (sch.levels.getD j default).d)
    (hpos : ∀ j < sch.levels.size, ∀ coins, ∀ x ∈ positions (sch.levels.getD j default) coins,
      x < 2 ^ (sch.levels.getD j default).d)
    {g : Array F128} {salt : ℕ → ByteArray} (h0 : commitP hs ms (sch.levels.getD 0 default) g salt = .ok cm0)
    (ho : Opens hs ms sch cm0 p cf k) : MerkleOk hs ms sch cm0.cap p cf k := by
  obtain ⟨h0p, -, -, hin, hfin⟩ := ho
  intro j hj
  by_cases hj0 : j = 0
  · subst hj0
    have hoj : opening p cm0.cap (sch.levels.size - 1) 0 =
        (cm0.cap, p.initialProof.rows, p.initialProof.siblings, p.initialProof.salts) := if_pos rfl
    rw [hoj]
    dsimp only
    rw [h0p]
    exact merkle_open (commitP_tree hn hlf (hcap 0 hj) h0) (hcap 0 hj) (hpos 0 hj _)
  · by_cases hjR : j < sch.levels.size - 1
    · obtain ⟨cm, g', salt', hc, hcc, hop⟩ := hin (j - 1) (by omega)
      rw [Nat.sub_add_cancel (show 1 ≤ j by omega)] at hc hop
      have hoj : opening p cm0.cap (sch.levels.size - 1) j = (p.recursiveCaps[j - 1]!,
          p.recursiveProofs[j - 1]!.rows, p.recursiveProofs[j - 1]!.siblings, p.recursiveProofs[j - 1]!.salts) := by
        unfold opening
        rw [if_neg hj0, if_pos hjR]
      rw [hoj]
      dsimp only
      rw [hop, ← hcc]
      exact merkle_open (commitP_tree hn hlf (hcap j hj) hc) (hcap j hj) (hpos j hj _)
    · obtain rfl : j = sch.levels.size - 1 := by omega
      obtain ⟨cm, g', salt', hc, hcc, hop⟩ := hfin
      have hoj : opening p cm0.cap (sch.levels.size - 1) (sch.levels.size - 1) =
          (p.recursiveCaps[sch.levels.size - 1 - 1]!, p.finalProof.rows, p.finalProof.siblings,
            p.finalProof.salts) := by
        unfold opening
        rw [if_neg hj0, if_neg (lt_irrefl _)]
      rw [hoj]
      dsimp only
      have e1 := congrArg LevelProof.rows hop
      have e2 := congrArg LevelProof.siblings hop
      have e3 := congrArg LevelProof.salts hop
      dsimp only at e1 e2 e3
      rw [e1, e2, e3, ← hcc]
      exact merkle_open (commitP_tree hn hlf (hcap _ hj) hc) (hcap _ hj) (hpos _ hj _)

end Lg

end FlockComplete.Replay
