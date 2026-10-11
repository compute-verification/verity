import Mathlib
import Properties.CertifierDevice

/-!
# LMS, proved: a key's signatures verify

Completeness, not guarantees: `LmsCorrect` (every signature `signAt` returns verifies) and `LmsSigns` (`signAt` returns
at every leaf below `2^H`), each with a toy-key instance in `Proofs.CertifierNetwork.Instance`.

A key `Lms.key` derives holds its tree: node `2^H + q` is leaf q's hash of its one-time public key, and each interior node
r is `interior r` of its children (`key_spec`). The signature `signAt` gives at leaf q reveals each chain `a_i` steps in,
and the verifier runs it the remaining `2^W − 1 − a_i` steps, which ends where the key's chain ended (`chain_chain`); so
the verifier recomputes leaf q, and the path's siblings climb it to the root `T[1]` (`climb`). The signature's encoding
is `4 + N(1 + P + H)` bytes, and its blocks decode back to it (`decode_encode`).
-/

namespace Properties.CertifierDevice.Proofs.Lms

open _root_.Lms Flock

/-! ## Bytes -/

theorem sha512_size (m : ByteArray) : (Sha512.hash m).size = 64 := by
  unfold Sha512.hash
  exact Array.size_ofFn

theorem hash_size (p : Params) (hN : p.N ≤ 64) (d : ByteArray) : (p.hash d).size = p.N := by
  simp only [Params.hash, ByteArray.size_extract, sha512_size]
  omega

theorem beq_self (x : ByteArray) : (x == x) = true := by
  show (x.data == x.data) = true
  simp

/-! ## Chains -/

theorem chain_eq (p : Params) (I : ByteArray) (q i : Nat) (tmp : ByteArray) (start stop : Nat) :
    chain p I q i tmp start stop = (List.range' start (stop - start)).foldl
      (fun t j => p.hash (I ++ u32 q ++ u16 i ++ ⟨#[j.toUInt8]⟩ ++ t)) tmp := by
  simp [chain, Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl]

/-- A chain run `a` steps from its start, then on from step `a` to `e`, is the chain run `e` steps. -/
theorem chain_chain (p : Params) (I : ByteArray) (q i : Nat) (x : ByteArray) {a e : Nat} (h : a ≤ e) :
    chain p I q i (chain p I q i x 0 a) a e = chain p I q i x 0 e := by
  rw [chain_eq, chain_eq, chain_eq, ← List.foldl_append]
  congr 1
  have := @List.range'_append 0 a (e - a) 1
  simp only [Nat.sub_zero, Nat.one_mul, Nat.zero_add] at this ⊢
  rw [this]
  congr 1
  omega

/-- Every digit is below `2^W`: at most `2^W − 1` steps. -/
theorem digits_le (p : Params) (Q : ByteArray) (i : Nat) : (digits p Q)[i]! ≤ 2 ^ p.W - 1 := by
  unfold digits
  by_cases hi : i < p.P
  · rw [getElem!_pos _ _ (by simpa using hi)]
    simp only [Array.getElem_map, List.getElem_toArray, List.getElem_range, coef]
    exact Nat.and_le_left
  · rw [getElem!_neg _ _ (by simpa using hi)]
    exact Nat.zero_le _

/-! ## The key's tree -/

theorem get_set {α : Type _} [Inhabited α] (b : Array α) (i j : Nat) (v : α) :
    (b.setIfInBounds i v)[j]! = if i = j ∧ j < b.size then v else b[j]! := by
  by_cases hj : j < b.size
  · rw [getElem!_pos (b.setIfInBounds i v) j (by simpa using hj), getElem!_pos b j hj, Array.getElem_setIfInBounds hj]
    by_cases hij : i = j <;> simp [hij, hj]
  · rw [getElem!_neg (b.setIfInBounds i v) j (by simpa using hj), getElem!_neg b j hj]
    simp [hj]

/-- One step of `key`'s loop: node `top − 1 − a` from its children. -/
def treeStep (p : Params) (I : ByteArray) (top : Nat) (b : Array ByteArray) (a : Nat) : Array ByteArray :=
  b.setIfInBounds (top - 1 - a) (Lms.interior p I (top - 1 - a) b[2 * (top - 1 - a)]! b[2 * (top - 1 - a) + 1]!)

/-- After `n` steps: the size and the leaves kept, and the nodes `top − n … top − 1` each `interior` of its children. -/
def TreeInv (p : Params) (I : ByteArray) (top : Nat) (init : Array ByteArray) (n : Nat) (b : Array ByteArray) :
    Prop :=
  b.size = 2 * top ∧ (∀ j, top ≤ j → b[j]! = init[j]!) ∧
    ∀ r, top - n ≤ r → r < top → b[r]! = Lms.interior p I r b[2 * r]! b[2 * r + 1]!

theorem tree_fold (p : Params) (I : ByteArray) (top : Nat) (init : Array ByteArray) (hinit : init.size = 2 * top) :
    ∀ n, n ≤ top - 1 → TreeInv p I top init n ((List.range' 0 n).foldl (treeStep p I top) init) := by
  intro n
  induction n with
  | zero => intro _; exact ⟨hinit, fun _ _ => rfl, fun r h1 h2 => by omega⟩
  | succ n ih =>
    intro hn
    obtain ⟨hs, hl, hr⟩ := ih (by omega)
    rw [List.range'_concat, List.foldl_append]
    simp only [List.foldl_cons, List.foldl_nil, Nat.zero_add, Nat.one_mul]
    generalize (List.range' 0 n).foldl (treeStep p I top) init = b at hs hl hr
    have hne : ∀ j, top - 1 - n < j → (treeStep p I top b n)[j]! = b[j]! := by
      intro j hj
      unfold treeStep
      rw [get_set]
      exact ite_eq_right_iff.2 fun hc => absurd hc (by omega)
    refine ⟨by simp [treeStep, hs], fun j hj => by rw [hne j (by omega), hl j hj], fun r h1 h2 => ?_⟩
    rw [hne (2 * r) (by omega), hne (2 * r + 1) (by omega)]
    by_cases hr0 : r = top - 1 - n
    · subst hr0
      unfold treeStep
      rw [get_set]
      exact ite_eq_left_iff.2 fun hc => absurd ⟨rfl, by omega⟩ hc
    · rw [hne r (by omega)]
      exact hr r (by omega) h2

theorem leaves_get (p : Params) (I seed : ByteArray) (top q : Nat) (hq : q < top) :
    (Array.replicate top ByteArray.empty ++ leaves p I seed 0 top)[top + q]! =
      leaf p I q (otsPublic p I q ((List.range p.P).toArray.map fun i =>
        chain p I q i (secret p I seed q i) 0 (2 ^ p.W - 1))) := by
  have hsz : (leaves p I seed 0 top).size = top := by simp [leaves]
  rw [getElem!_pos _ _ (by simp [hsz]; omega), Array.getElem_append_right (by simp)]
  simp [leaves]

/-- What `key` returns: its parameters, identifier and seed, a 16-byte identifier, and the tree. -/
theorem key_spec {p : Params} {I seed : ByteArray} {k : PrivateKey} (h : key p I seed = .ok k) :
    k.params = p ∧ k.I = I ∧ k.seed = seed ∧ I.size = 16 ∧ k.tree.size = 2 * 2 ^ p.H ∧
      (∀ q, q < 2 ^ p.H → k.tree[2 ^ p.H + q]! = leaf p I q (otsPublic p I q ((List.range p.P).toArray.map fun i =>
        chain p I q i (secret p I seed q i) 0 (2 ^ p.W - 1)))) ∧
      ∀ r, 1 ≤ r → r < 2 ^ p.H → k.tree[r]! = Lms.interior p I r k.tree[2 * r]! k.tree[2 * r + 1]! := by
  unfold key at h
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl] at h
  by_cases hI : I.size = 16
  · simp only [hI, ne_eq, not_true_eq_false, ↓reduceIte] at h
    have hinit : (Array.replicate (2 ^ p.H) ByteArray.empty ++ leaves p I seed 0 (2 ^ p.H)).size = 2 * 2 ^ p.H := by
      simp [leaves]; omega
    obtain ⟨hs, hl, hr⟩ := tree_fold p I (2 ^ p.H) _ hinit (2 ^ p.H - 1) le_rfl
    simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one, pure, Except.pure, bind,
      Except.bind, Except.ok.injEq] at h
    subst h
    refine ⟨rfl, rfl, rfl, hI, hs, fun q hq => ?_, fun r h1 h2 => hr r (by omega) h2⟩
    exact (hl _ (by omega)).trans (leaves_get p I seed _ q hq)
  · simp [hI] at h
    cases h

/-! ## The climb -/

/-- One step of `rootOf`'s climb: the node and its index, one level up. -/
def climbStep (p : Params) (I : ByteArray) (b : ByteArray × Nat) (sib : ByteArray) : ByteArray × Nat :=
  (if b.2 % 2 = 1 then Lms.interior p I (b.2 / 2) sib b.1 else Lms.interior p I (b.2 / 2) b.1 sib, b.2 / 2)

theorem rootOf_ok (p : Params) (I m : ByteArray) (sig : Signature)
    (h : sig.q < 2 ^ p.H ∧ sig.y.size = p.P ∧ sig.path.size = p.H) :
    rootOf p I m sig = .ok (sig.path.toList.foldl (climbStep p I) (leaf p I sig.q (otsPublic p I sig.q
      ((List.range p.P).toArray.map fun i =>
        chain p I sig.q i sig.y[i]! (digits p (messageHash p I sig.q sig.C m))[i]! (2 ^ p.W - 1))),
      2 ^ p.H + sig.q)).1 := by
  unfold rootOf
  rw [ite_eq_right_of_eq_false _ _ (eq_false (not_not.2 h))]
  simp only [Array.forIn_pure_yield_eq_foldl, Array.foldl_toList]
  rfl

theorem xor_one (r : Nat) : r ^^^ 1 = if r % 2 = 1 then r - 1 else r + 1 := by
  have h1 : (r ^^^ 1) % 2 = (r + 1) % 2 := Nat.xor_mod_two_eq
  have h2 : (r ^^^ 1) / 2 = r / 2 := by
    have := @Nat.shiftRight_xor_distrib 1 r 1
    simpa [Nat.shiftRight_eq_div_pow] using this
  split <;> omega

/-- Below the root level, the climb's node is an interior node or a leaf, not the root. -/
theorem node_bounds {H q l : Nat} (hq : q < 2 ^ H) (hl : l < H) :
    2 ≤ (2 ^ H + q) >>> l ∧ (2 ^ H + q) >>> l < 2 * 2 ^ H := by
  rw [Nat.shiftRight_eq_div_pow]
  constructor
  · rw [Nat.le_div_iff_mul_le (by positivity)]
    have : 2 * 2 ^ l ≤ 2 ^ H := by
      rw [← pow_succ']
      exact Nat.pow_le_pow_right (by omega) hl
    omega
  · exact lt_of_le_of_lt (Nat.div_le_self _ _) (by omega)

/-- The sibling of a node below the root is a node of the tree. -/
theorem sibling_bounds {H r : Nat} (h1 : 2 ≤ r) (h2 : r < 2 * 2 ^ H) : 1 ≤ r ^^^ 1 ∧ r ^^^ 1 < 2 * 2 ^ H := by
  have : 2 * 2 ^ H % 2 = 0 := by simp
  rw [xor_one]
  split <;> omega

/-- The path's siblings climb leaf q to node `(2^H + q) >>> l` after `l` levels. -/
theorem climb (p : Params) (I : ByteArray) (tree : Array ByteArray) (q : Nat) (hq : q < 2 ^ p.H)
    (hr : ∀ r, 1 ≤ r → r < 2 ^ p.H → tree[r]! = Lms.interior p I r tree[2 * r]! tree[2 * r + 1]!) :
    ∀ l, l ≤ p.H → ((List.range l).map fun l => tree[((2 ^ p.H + q) >>> l) ^^^ 1]!).foldl (climbStep p I)
      (tree[2 ^ p.H + q]!, 2 ^ p.H + q) = (tree[(2 ^ p.H + q) >>> l]!, (2 ^ p.H + q) >>> l) := by
  intro l
  induction l with
  | zero => intro _; simp
  | succ l ih =>
    intro hl
    rw [List.range_succ, List.map_append, List.foldl_append, ih (by omega)]
    simp only [List.map_cons, List.map_nil, List.foldl_cons, List.foldl_nil, Nat.shiftRight_succ]
    obtain ⟨hlo, hhi⟩ := node_bounds hq (show l < p.H by omega)
    generalize (2 ^ p.H + q) >>> l = r at hlo hhi
    obtain ⟨m, rfl | rfl⟩ : ∃ m, r = 2 * m ∨ r = 2 * m + 1 := ⟨r / 2, by omega⟩
    · have hm := hr m (by omega) (by omega)
      have e1 : (2 * m % 2 = 1) = False := eq_false (by omega)
      have e2 : 2 * m / 2 = m := by omega
      simp only [climbStep, xor_one, e1, e2, ↓reduceIte, hm]
    · have hm := hr m (by omega) (by omega)
      have e1 : (2 * m + 1) % 2 = 1 := by omega
      have e2 : (2 * m + 1) / 2 = m := by omega
      have e3 : 2 * m + 1 - 1 = 2 * m := by omega
      simp only [climbStep, xor_one, e1, e2, e3, ↓reduceIte, hm]

theorem top_shift (H q : Nat) (hq : q < 2 ^ H) : (2 ^ H + q) >>> H = 1 := by
  rw [Nat.shiftRight_eq_div_pow]
  exact Nat.div_eq_of_lt_le (by omega) (by omega)

/-! ## Signing and verifying -/

/-- Every node of the tree is N bytes. -/
theorem node_size {p : Params} {I seed : ByteArray} {k : PrivateKey} (hk : key p I seed = .ok k) (hN : p.N ≤ 64) :
    ∀ r, 1 ≤ r → r < 2 * 2 ^ p.H → (k.tree[r]!).size = p.N := by
  obtain ⟨-, -, -, -, -, hleaf, hint⟩ := key_spec hk
  intro r h1 h2
  by_cases hr : r < 2 ^ p.H
  · rw [hint r h1 hr]
    exact hash_size p hN _
  · obtain ⟨q, rfl⟩ : ∃ q, r = 2 ^ p.H + q := ⟨r - 2 ^ p.H, by omega⟩
    rw [hleaf q (by omega)]
    exact hash_size p hN _

theorem make_spec {N W H : Nat} {toy : Bool} {p : Params} (h : Params.make N W H toy = .ok p) :
    (p.N = 8 ∨ p.N = 64) ∧ p.H = H ∧ 1 ≤ H ∧ H ≤ 30 := by
  unfold Params.make at h
  cases toy <;> simp only [Bool.false_eq_true, ↓reduceIte] at h <;> split at h <;> rename_i hc <;>
    simp only [pure, Except.pure, Except.ok.injEq, reduceCtorEq] at h <;> subst h <;>
    refine ⟨?_, rfl, hc.2.2.1, hc.2.2.2⟩ <;>
    (have := hc.1; simp only [toyN, List.mem_cons, List.mem_nil_iff, or_false] at this; dsimp only; omega)

/-! ## The signature's encoding -/

/-- Blocks concatenated in order. -/
def cat (l : List ByteArray) : ByteArray := l.foldr (· ++ ·) .empty

theorem foldl_cat (l : List ByteArray) (acc : ByteArray) : l.foldl (· ++ ·) acc = acc ++ cat l := by
  induction l generalizing acc with
  | nil => simp [cat, ByteArray.append_empty]
  | cons x l ih => simp only [List.foldl_cons, ih, cat, List.foldr_cons, ByteArray.append_assoc]

theorem cat_append (l l' : List ByteArray) : cat (l ++ l') = cat l ++ cat l' := by
  induction l with
  | nil => simp [cat, ByteArray.empty_append]
  | cons x l ih =>
    simp only [List.cons_append, cat, List.foldr_cons] at ih ⊢
    rw [ih, ByteArray.append_assoc]

theorem cat_list (l : List ByteArray) : (cat l).data.toList = (l.map fun x => x.data.toList).flatten := by
  induction l with
  | nil => rfl
  | cons x l ih =>
    simp only [cat, List.foldr_cons, ByteArray.data_append, Array.toList_append, List.map_cons,
      List.flatten_cons] at ih ⊢
    rw [ih]

theorem cat_size {N : Nat} (l : List ByteArray) (h : ∀ x ∈ l, x.size = N) : (cat l).size = N * l.length := by
  induction l with
  | nil => rfl
  | cons x l ih =>
    simp only [cat, List.foldr_cons, ByteArray.size_append, List.length_cons] at ih ⊢
    rw [ih (fun y hy => h y (by simp [hy])), h x (by simp), Nat.mul_succ]
    omega

/-- Block k of blocks of N bytes each starts at byte `N · k`. -/
theorem block_get {α : Type _} {N : Nat} (L : List (List α)) (hL : ∀ x ∈ L, x.length = N) :
    ∀ k (hk : k < L.length), ((L.flatten).drop (N * k)).take N = L[k] := by
  induction L with
  | nil => intro k hk; simp at hk
  | cons x L ih =>
    intro k hk
    have hx := hL x (by simp)
    cases k with
    | zero => simp [List.take_left' hx]
    | succ k =>
      rw [List.flatten_cons, List.drop_append, List.drop_eq_nil_of_le (by rw [hx, Nat.mul_succ]; omega),
        List.nil_append, hx, Nat.mul_succ, show N * k + N - N = N * k by omega]
      exact ih (fun y hy => hL y (by simp [hy])) k (by simpa using hk)

theorem be32_eq (x : UInt32) :
    be32 x = ⟨#[(x >>> 24).toUInt8, (x >>> 16).toUInt8, (x >>> 8).toUInt8, x.toUInt8]⟩ := by
  simp only [be32, Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl, Std.Legacy.Range.size]
  rfl

theorem get!_append (a r : ByteArray) (i : Nat) (h : i < a.size) :
    (a ++ r).get! i = a.data[i]'(by simpa using h) := by
  show (a ++ r).data[i]! = _
  simp only [ByteArray.data_append]
  rw [getElem!_pos _ i (by simp; omega), Array.getElem_append_left]

/-- `Signature.decode` inverts `Signature.encode` on a signature of `P` chain values and `H` path nodes, N bytes each,
whose leaf fits `u32`. -/
theorem decode_encode (p : Params) (s : Signature) (hq : s.q < 2 ^ 32) (hC : s.C.size = p.N) (hy : s.y.size = p.P)
    (hpath : s.path.size = p.H) (hys : ∀ x ∈ s.y, x.size = p.N) (hps : ∀ x ∈ s.path, x.size = p.N) :
    Signature.decode p s.encode = .ok s := by
  have hb : ∀ x ∈ s.C :: (s.y.toList ++ s.path.toList), x.size = p.N := by
    intro x hx
    simp only [List.mem_cons, List.mem_append, Array.mem_toList_iff] at hx
    rcases hx with rfl | hx | hx
    · exact hC
    · exact hys x hx
    · exact hps x hx
  have hu : (u32 s.q).size = 4 := by simp [u32, be32_eq]; rfl
  have henc : s.encode = u32 s.q ++ cat (s.C :: (s.y.toList ++ s.path.toList)) := by
    unfold Signature.encode
    rw [← Array.foldl_toList, ← Array.foldl_toList, foldl_cat, foldl_cat,
      show cat (s.C :: (s.y.toList ++ s.path.toList)) = s.C ++ (cat s.y.toList ++ cat s.path.toList) by
        rw [← cat_append]; rfl]
    simp only [ByteArray.append_assoc]
  have hlen : (s.C :: (s.y.toList ++ s.path.toList)).length = 1 + p.P + p.H := by
    simp [hy, hpath]; omega
  have hsize : s.encode.size = p.signatureBytes := by
    rw [henc, ByteArray.size_append, hu, cat_size _ hb, hlen, Params.signatureBytes]
  have hcut : ((List.range (1 + p.P + p.H)).toArray.map fun k =>
      s.encode.extract (4 + p.N * k) (4 + p.N * (k + 1))) = (s.C :: (s.y.toList ++ s.path.toList)).toArray := by
    rw [List.map_toArray]
    congr 1
    apply List.ext_getElem (by simp [hlen])
    intro k h1 h2
    simp only [List.getElem_map, List.getElem_range]
    apply ByteArray.ext
    rw [← Array.toList_inj, ByteArray.data_extract, Array.toList_extract, List.extract_eq_take_drop, henc,
      ByteArray.data_append, Array.toList_append, cat_list, List.drop_append,
      List.drop_eq_nil_of_le (by rw [Array.length_toList, ByteArray.size_data, hu]; omega), List.nil_append,
      Array.length_toList, ByteArray.size_data, hu, show 4 + p.N * k - 4 = p.N * k by omega, Nat.mul_succ,
      show 4 + (p.N * k + p.N) - (4 + p.N * k) = p.N by omega,
      block_get _ (fun x hx => by
        obtain ⟨b, hb', rfl⟩ := List.mem_map.1 hx
        rw [Array.length_toList, ByteArray.size_data, hb b hb']) k (by simpa using h2), List.getElem_map]
  unfold Signature.decode
  simp only [hsize, ne_eq, not_true_eq_false, ↓reduceIte, hcut]
  have hq' : (List.range 4).foldl (fun acc i => acc * 256 + (s.encode.get! i).toNat) 0 = s.q := by
    simp only [henc, List.range_succ, List.range_zero, List.nil_append, List.foldl_append, List.foldl_cons,
      List.foldl_nil]
    rw [get!_append _ _ 0 (by omega), get!_append _ _ 1 (by omega), get!_append _ _ 2 (by omega),
      get!_append _ _ 3 (by omega)]
    simp only [u32, be32_eq]
    simp [UInt32.toNat_shiftRight, UInt32.toNat_toUInt8, Nat.shiftRight_eq_div_pow]
    omega
  have hy' : (s.C :: (s.y.toList ++ s.path.toList)).toArray.extract 1 (1 + p.P) = s.y := by
    rw [← Array.toList_inj, Array.toList_extract, List.extract_eq_take_drop]
    simp [hy]
  have hp' : (s.C :: (s.y.toList ++ s.path.toList)).toArray.extract (1 + p.P) = s.path := by
    rw [← Array.toList_inj, Array.toList_extract, List.extract_eq_take_drop]
    simp only [List.size_toArray, List.length_cons, List.length_append, Array.length_toList, hy, hpath]
    have e : 1 + p.P = (s.C :: s.y.toList).length := by simp [hy]; omega
    rw [e, ← List.cons_append, List.drop_left, List.take_of_length_le (by simp [hy, hpath])]
  rw [hq', hy', hp']
  rfl

/-- The verifier runs each revealed chain value on to the chain's end: the one-time key's chain ends. -/
theorem ots_z (p : Params) (I seed : ByteArray) (q : Nat) (a : Array Nat) (ha : ∀ i : Nat, a[i]! ≤ 2 ^ p.W - 1) :
    ((List.range p.P).toArray.map fun (i : Nat) => chain p I q i
      ((List.range p.P).toArray.map fun (i : Nat) => chain p I q i (secret p I seed q i) 0 a[i]!)[i]! a[i]! (2 ^ p.W - 1)) =
      (List.range p.P).toArray.map fun i => chain p I q i (secret p I seed q i) 0 (2 ^ p.W - 1) := by
  apply Array.map_inj_left.2
  intro i hi
  simp only [List.mem_toArray, List.mem_range] at hi
  rw [getElem!_pos _ i (by simpa using hi)]
  simp only [Array.getElem_map, List.getElem_toArray, List.getElem_range]
  exact chain_chain p I q i _ (ha i)

/-- The signature `signAt` gives at leaf `q < 2^H` verifies under the key's public key. -/
theorem verify_signAt {N W H : Nat} {toy : Bool} {p : Params} {I seed : ByteArray} {k : PrivateKey} {q : Nat}
    (m : ByteArray) (hp : Params.make N W H toy = .ok p) (hk : key p I seed = .ok k) (hq : q < 2 ^ H) :
    ∃ sig, signAt k q m = .ok sig ∧ verify p k.publicKey m sig = .ok true ∧ sig.q = q ∧
      sig.y.size = p.P ∧ sig.path.size = p.H ∧ sig.C.size = p.N ∧ (∀ x ∈ sig.y, x.size = p.N) ∧
      ∀ x ∈ sig.path, x.size = p.N := by
  obtain ⟨hN, hH, -, -⟩ := make_spec hp
  obtain ⟨hpk, hIk, hseed, hIs, hsz, hleaf, hint⟩ := key_spec hk
  subst hpk hIk hseed hH
  have hN64 : k.params.N ≤ 64 := by omega
  have hsize : ∀ d, (k.params.hash d).size = k.params.N := hash_size _ hN64
  unfold signAt
  simp only [show ¬q ≥ 2 ^ k.params.H by omega, ↓reduceIte]
  refine ⟨_, rfl, ?_, rfl, by simp, by simp, hsize _, ?_, ?_⟩
  rotate_left
  · intro x hx
    simp only [Array.mem_map, List.mem_toArray, List.mem_range] at hx
    obtain ⟨i, -, rfl⟩ := hx
    rw [chain_eq]
    generalize List.range' _ _ = l
    cases l using List.reverseRecOn <;> simp [hsize, secret]
  · intro x hx
    simp only [Array.mem_map, List.mem_toArray, List.mem_range] at hx
    obtain ⟨l, hl, rfl⟩ := hx
    obtain ⟨h1, h2⟩ := node_bounds hq hl
    obtain ⟨h3, h4⟩ := sibling_bounds h1 h2
    exact node_size hk hN64 _ h3 h4
  have hroot := node_size hk hN64 1 le_rfl (by have := Nat.one_le_two_pow (n := k.params.H); omega)
  have hpub : k.publicKey.size = 16 + k.params.N := by
    simp only [PrivateKey.publicKey, ByteArray.size_append, hIs, hroot]
  have hI : k.publicKey.extract 0 16 = k.I := ByteArray.extract_append_eq_left hIs.symm
  have hT : k.publicKey.extract 16 k.publicKey.size = k.tree[1]! :=
    ByteArray.extract_append_eq_right hIs.symm (by rw [hpub, hIs, hroot])
  unfold verify
  simp only [hpub, ne_eq, not_true_eq_false, ↓reduceIte, hI]
  rw [rootOf_ok _ _ _ _ ⟨hq, by simp, by simp⟩]
  dsimp only
  rw [ots_z _ _ _ _ _ (digits_le _ _), ← hleaf q hq]
  simp only [Array.toList_map]
  rw [climb k.params k.I k.tree q hq hint _ le_rfl, top_shift _ _ hq, ← hpub, hT]
  exact congrArg pure (beq_self _)

/-- **LMS signatures verify** (completeness, not a guarantee): for a key `Lms.key` derives under parameters
`Lms.Params.make` admits, every signature `Lms.signAt` returns verifies under the key's public key (`Lms.verify`), and its
encoding decodes back to it. It pairs with the LMS-forgery disjunct of
`Properties.NetworkCertifier.AcceptedRowsAreProgramRows`. -/
def LmsCorrect : Prop :=
  ∀ (N W H : Nat) (toy : Bool) (p : Params) (I seed : ByteArray) (k : PrivateKey) (q : Nat) (message : ByteArray)
    (sig : Signature),
    Params.make N W H toy = .ok p → key p I seed = .ok k → signAt k q message = .ok sig →
    verify p k.publicKey message sig = .ok true ∧ Signature.decode p sig.encode = .ok sig

theorem lmsCorrect : LmsCorrect := by
  intro N W H toy p I seed k q message sig hp hk hs
  obtain ⟨-, hH, -, hH30⟩ := make_spec hp
  obtain ⟨hpk, -⟩ := key_spec hk
  have hq : q < 2 ^ k.params.H := by
    unfold signAt at hs
    by_contra hq
    simp [show q ≥ 2 ^ k.params.H by omega] at hs
    cases hs
  rw [hpk, hH] at hq
  obtain ⟨sig', hs', hv, hsq, hy, hpath, hC, hys, hps⟩ := verify_signAt message hp hk hq
  rw [hs] at hs'
  cases hs'
  have : 2 ^ H ≤ 2 ^ 30 := Nat.pow_le_pow_right (by omega) hH30
  exact ⟨hv, decode_encode p sig (by omega) hC hy hpath hys hps⟩

/-- `Lms.key` succeeds on every 16-byte identifier. -/
theorem key_ok (p : Params) {I : ByteArray} (hI : I.size = 16) (seed : ByteArray) : ∃ k, key p I seed = .ok k := by
  unfold key
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl]
  simp only [hI, ne_eq, not_true_eq_false, ↓reduceIte, pure, Except.pure, bind, Except.bind]
  exact ⟨_, rfl⟩

/-- **LMS signs at every leaf** (completeness, not a guarantee): for a key `Lms.key` derives under parameters
`Lms.Params.make` admits, `Lms.signAt` returns a signature of every message at every leaf below `2^H`. -/
def LmsSigns : Prop :=
  ∀ (N W H : Nat) (toy : Bool) (p : Params) (I seed : ByteArray) (k : PrivateKey) (q : Nat) (message : ByteArray),
    Params.make N W H toy = .ok p → key p I seed = .ok k → q < 2 ^ H → ∃ sig, signAt k q message = .ok sig

theorem lmsSigns : LmsSigns := by
  intro N W H toy p I seed k q message hp hk hq
  obtain ⟨sig, hs, -⟩ := verify_signAt message hp hk hq
  exact ⟨sig, hs⟩

end Properties.CertifierDevice.Proofs.Lms
