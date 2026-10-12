/-
A row at its columns: port `p`'s row at template instance `g` of a block whose bits are `w`. Its bytes are read at the `sha512x3` slots'
message inputs (`Typed.msgCol`), its salt at the `hm96` slot's input `y` (`saltCol`) and its `b ‖ c` at that slot's
output (`bcCol`, the address `HmRow.regions` pins). When the slots' nets hold on the block's bits there and the block
holds Δ's copies (the first `cv` the midstate, each next `cv` the previous output, the hm96 `cv` the last output, its `pad`
the padding block), the bytes at `bcCol` are the commit string of the row and salt (`port_commit`).
-/
import Proofs.Flock.Soundness.Discharge.Hm.Row
import Proofs.Flock.Soundness.ExecTyped
import Proofs.Flock.Soundness.Binding.Layout

namespace FlockSoundness.Discharge.Hm

open Flock.HmNets hiding and

open Flock

/-! ## Bytes from bits -/

theorem testBit_sumBits : ∀ (k : ℕ) (b : Fin k → Bool) (i : ℕ),
    (∑ t : Fin k, if b t then 2 ^ t.val else 0).testBit i = if h : i < k then b ⟨i, h⟩ else false
  | 0, b, i => by simp
  | k + 1, b, i => by
    rw [Fin.sum_univ_succ]
    have hrest : (∑ t : Fin k, if b t.succ then 2 ^ (t.succ : Fin (k + 1)).val else 0) =
        2 * ∑ t : Fin k, if b t.succ then 2 ^ t.val else 0 := by
      rw [Finset.mul_sum]
      refine Finset.sum_congr rfl fun t _ => ?_
      simp only [Fin.val_succ, pow_succ]
      split_ifs <;> ring
    rw [hrest]
    have ih := testBit_sumBits k (fun t => b t.succ)
    generalize (∑ t : Fin k, if b t.succ then 2 ^ t.val else 0) = S at ih ⊢
    cases i with
    | zero =>
      rw [Nat.testBit_zero, dite_eq_left (Nat.zero_lt_succ k)]
      cases h : b 0 <;> simp [Fin.zero_eta, h]
    | succ i =>
      have hdiv : ((if b 0 then 2 ^ (0 : Fin (k + 1)).val else 0) + 2 * S) / 2 = S := by
        split_ifs
        · simp only [Fin.val_zero, pow_zero]
          omega
        · simp
      rw [Nat.testBit_succ, hdiv, ih i]
      by_cases hi : i < k
      · rw [dite_eq_left hi, dite_eq_left (by omega)]
        rfl
      · rw [dite_eq_right hi, dite_eq_right (by omega)]

theorem testBit_byteOf (b : Fin 8 → Bool) (t : Fin 8) : (Binding.byteOf b).toNat.testBit t.val = b t := by
  rw [Binding.byteOf, UInt8.toNat_ofNat', Nat.testBit_mod_two_pow, testBit_sumBits, dite_eq_left t.isLt]
  simp [t.isLt]

/-- The `N` bytes whose byte `j`'s bit `t` is `w (a j t)`. -/
def bytesOf (w : ℕ → Bool) (N : ℕ) (a : ℕ → ℕ → ℕ) : List UInt8 :=
  List.ofFn fun j : Fin N => Binding.byteOf fun t => w (a j t)

theorem length_bytesOf (w : ℕ → Bool) (N : ℕ) (a : ℕ → ℕ → ℕ) : (bytesOf w N a).length = N := List.length_ofFn

theorem bitL_bytesOf (w : ℕ → Bool) {N : ℕ} (a : ℕ → ℕ → ℕ) {q : ℕ} (hq : q < 8 * N) :
    bitL (bytesOf w N a) q = w (a (q / 8) (q % 8)) := by
  rw [bitL, bytesOf, List.getD_eq_getElem?_getD, List.getElem?_ofFn, dite_eq_left (by omega), Option.getD_some,
    testBit_byteOf (fun t => w (a (q / 8) t)) ⟨q % 8, Nat.mod_lt _ (by norm_num)⟩]

/-- `Binding.bytesAt` at addresses `A (a j t)` reads `bytesOf` of the witness's bits. -/
theorem bytesAt_eq_bytesOf {m N : ℕ} (z : Model.Witness m) (A : ℕ → Fin (2 ^ m)) (a : ℕ → ℕ → ℕ) :
    Binding.bytesAt z (fun (j : Fin N) (t : Fin 8) => A (a j t)) =
      bytesOf (fun x => FlockSoundness.bit (z (A x))) N a := rfl

/-! ## The row's columns -/

/-- Compression `k` of port `p`'s row: its `sha512x3` slot at template instance `g`, its first column. -/
def shaBase (c : Circuit) (g p k : ℕ) : ℕ := c.slotCol c.shaNet g ((HmRow.rowSlots c.ports).2[p]!)[k]!.1

/-- Compression `k` of port `p`'s row: its place in its slot (`sub < 3`). -/
def shaSub (c : Circuit) (p k : ℕ) : ℕ := ((HmRow.rowSlots c.ports).2[p]!)[k]!.2

/-- The `sha512x3` net's input port `i`'s first column in a slot (`cv_k`, `msg_k` at `2k`, `2k + 1`). -/
def shaIn (c : Circuit) (i : ℕ) : ℕ := (portCols (c.nets.getD c.shaNet default).2.inGroups)[i]!.1

/-- The `sha512x3` net's output port `j`'s first column in a slot (`out_j`). -/
def shaOut (c : Circuit) (j : ℕ) : ℕ := ((portCols (c.nets.getD c.shaNet default).2.outGroups).getD j default).1

/-- The `hm96` net's input port `i`'s first column in a slot (`cv`, `pad`, `y`). -/
def hmIn (c : Circuit) (i : ℕ) : ℕ := ((portCols (c.nets.getD c.hmNet default).2.inGroups).getD i default).1

/-- The `hm96` net's output's first column in a slot (`b ‖ c`). -/
def hmOut (c : Circuit) : ℕ := ((portCols (c.nets.getD c.hmNet default).2.outGroups).getD 0 default).1

/-- **Salt byte `j`, bit `t`** of port `p`'s row at template instance `g`: its `hm96` slot's input `y`, bit `8 j + t`.
-/
def saltCol (c : Circuit) (g p j t : ℕ) : ℕ := c.slotCol c.hmNet g p + hmIn c 2 + 8 * j + t

/-- **`b ‖ c` byte `j`, bit `t`** of port `p`'s row at template instance `g`: its `hm96` slot's output bit `8 j + t`,
the address `HmRow.regions` pins. -/
def bcCol (c : Circuit) (g p j t : ℕ) : ℕ := c.slotCol c.hmNet g p + hmOut c + 8 * j + t

/-- The row's byte `128 k + o`, bit `t`, is compression `k`'s message input bit `wordBit (8 o + t)`, for a compression
`k` of the row. -/
theorem msgCol_eq (c : Circuit) (g p k : ℕ) {o t : ℕ} (hs : c.ports[p]!.segs = none)
    (hk : k < ((HmRow.rowSlots c.ports).2[p]!).size) (ho : o < 128) (ht : t < 8) :
    Typed.msgCol c g p (128 * k + o) t = shaBase c g p k + shaIn c (2 * shaSub c p k + 1) + wordBit (8 * o + t) := by
  have hk' : (128 * k + o) / 128 = k := by omega
  have ho' : (128 * k + o) % 128 = o := by omega
  simp only [Typed.msgCol, Typed.byteAt_none hs, hk', ho', hk, ↓reduceIte, shaBase, shaSub, shaIn, wordBit]
  omega

/-! ## The port -/

/-- **A port's row at its columns.** Port `p` of the circuit `c` has `words` u16 words in `n > 0` chunks
(`2 · words = 128 n`, `HmRow.parse`'s port check), each a compression of the row's slots (`hslots`). At template
instance `g` of a block whose bits are `w`: suppose the `sha512x3` and `hm96` nets' input ports sit where the generated
nets put them (`hsin`, `hhin`), each compression `k`'s slot holds the net on an assignment `zs k` that reads the slot's
input columns and whose output `sub k` the slot's output columns hold, the `hm96` slot likewise on `zh`, and the block
holds Δ's copies: the first `cv` the prefix's midstate, each next `cv` the previous output, the hm96 `cv` the last
output, and its `pad` the padding block. Then the 128 bytes at `bcCol` are `commitString (rowPrefix words ‖ row) salt`,
where the row is the `2 · words` bytes at `Typed.msgCol` and the salt the 192 bytes at `saltCol`. -/
theorem port_commit (c : Circuit) (w : ℕ → Bool) (g p : ℕ) {words n : ℕ} (hs : c.ports[p]!.segs = none)
    (hn : 0 < n) (hw : 2 * words = 128 * n)
    (hslots : n ≤ ((HmRow.rowSlots c.ports).2[p]!).size) (hsub : ∀ k < n, shaSub c p k < 3)
    (hsin : ∀ s < 3, shaIn c (2 * s) = 1536 * s ∧ shaIn c (2 * s + 1) = 1536 * s + 512)
    (hhin : hmIn c 0 = 0 ∧ hmIn c 1 = 512 ∧ hmIn c 2 = 1536)
    (zs : ℕ → Var → Bool) (hzs : ∀ k < n, ∀ v : ℕ, v < 4608 → zs k v = w (shaBase c g p k + v))
    (hrows : ∀ k < n, stHolds (zs k) (compression3.run St.empty).2)
    (hout : ∀ k < n, ∀ t < 512, w (shaBase c g p k + shaOut c (shaSub c p k) + t) =
      bit (zs k) ((compression3.run St.empty).1.2.getD (shaSub c p k) []) t)
    (zh : Var → Bool) (hzh : ∀ v : ℕ, v < 3072 → zh v = w (c.slotCol c.hmNet g p + v))
    (hh : stHolds zh ((hm96Row Hm96.Default512.key).run St.empty).2)
    (hbc : ∀ t < 1024, w (c.slotCol c.hmNet g p + hmOut c + t) = bit zh ((hm96Row Hm96.Default512.key).run St.empty).1.2.2.2 t)
    (hcv0 : ∀ t < 512, w (shaBase c g p 0 + shaIn c (2 * shaSub c p 0) + t) = wordsBit (HmRow.midstate words) t)
    (hchain : ∀ k, k + 1 < n → ∀ t < 512, w (shaBase c g p (k + 1) + shaIn c (2 * shaSub c p (k + 1)) + t) =
      w (shaBase c g p k + shaOut c (shaSub c p k) + t))
    (hcvh : ∀ t < 512, w (c.slotCol c.hmNet g p + hmIn c 0 + t) =
      w (shaBase c g p (n - 1) + shaOut c (shaSub c p (n - 1)) + t))
    (hpad : ∀ t < 1024, w (c.slotCol c.hmNet g p + hmIn c 1 + t) = wordsBit (HmRow.padWords words) t) :
    bytesOf w 128 (bcCol c g p) = Binding.commitString
      ((HmRow.rowPrefix words).data.toList ++ bytesOf w (2 * words) (Typed.msgCol c g p)) (bytesOf w 192 (saltCol c g p)) := by
  have hd : (bytesOf w (2 * words) (Typed.msgCol c g p)).length = 128 * n := by rw [length_bytesOf, hw]
  have hrow := row_commit hn hw hd zs (shaSub c p) hsub hrows
    (fun q hq => by
      rw [hzs 0 hn _ (by have := hsub 0 hn; omega), ← hcv0 q hq, (hsin _ (hsub 0 hn)).1]
      exact congrArg w (by omega))
    (fun k hk q hq => by
      rw [hzs (k + 1) hk _ (by have := hsub (k + 1) hk; omega), ← hout k (by omega) q hq, ← hchain k hk q hq,
        (hsin _ (hsub (k + 1) hk)).1]
      exact congrArg w (by omega))
    (fun k hk o ho t ht => by
      have hwb : wordBit (8 * o + t) < 1024 := by unfold wordBit; omega
      rw [hzs k hk _ (by have := hsub k hk; omega), bitL_bytesOf _ _ (by omega),
        show (8 * (128 * k + o) + t) / 8 = 128 * k + o by omega, show (8 * (128 * k + o) + t) % 8 = t by omega,
        msgCol_eq c g p k hs (by omega) ho ht, (hsin _ (hsub k hk)).2]
      exact congrArg w (by omega))
    zh hh
    (fun q hq => by
      rw [hzh q (by omega), ← hout (n - 1) (by omega) q hq, ← hcvh q hq, hhin.1, Nat.add_zero])
    (fun q hq => by
      rw [hzh _ (by omega), ← hpad q hq, hhin.2.1]
      exact congrArg w (by omega))
    (length_bytesOf w 192 (saltCol c g p))
    (fun q hq => by
      rw [hzh _ (by omega), bitL_bytesOf _ _ (by omega), saltCol, hhin.2.2]
      exact congrArg w (by omega))
  refine eq_of_bitL (by rw [length_bytesOf, length_commitString]) fun t ht => ?_
  rw [length_bytesOf] at ht
  rw [bitL_bytesOf _ _ ht, ← hrow t ht, ← hbc t ht, bcCol]
  exact congrArg w (by omega)

end FlockSoundness.Discharge.Hm
