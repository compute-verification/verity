module

public import Properties.Pous.MemoryChallenger
public import Proofs.Properties.NetworkCertifier.Check.Bytes
import Mathlib.Algebra.Order.Group.Nat
import Mathlib.Algebra.Ring.Parity
import Mathlib.Data.Finset.Attr
import Mathlib.Tactic.Bound.Init
import Mathlib.Tactic.SetLike

/-!
# A frame-v3-sha512 tree's levels and frames

`Flock.FV3.Domain.levels` is the leaves padded to `nextPow2` of their count (`base`), then `log2ceil` levels
each of the nodes over the pairs below (`next`, `lev`, `levels_eq`). Every node and leaf is SHA-512 of a frame whose
parts are length-prefixed (`frame_last`, `frame_two`), so under no collision among the strings a tree hashes (`CF`) one
root gives one level below it. The streamed wipe's records tree (`WipeTree`) and the attested period tree (`Attest`)
bind their values through these.
-/

@[expose] public section

namespace MemoryChallenger.Properties.SecurityProofs.Tree

open MemoryChallenger Flock Flock.FV3.Sha512
open Properties.NetworkCertifier.Bytes (bs bs_append bs_length bs_inj split_bytes be64_size)

/-- No two distinct strings of `S` share a SHA-512 digest. -/
def CF (S : List ByteArray) : Prop := ∀ x ∈ S, ∀ y ∈ S, Sha512.hash x = Sha512.hash y → x = y

/-- One level up: the nodes over each pair. -/
def next (d : Domain) (lvl : Nat) (level : Array ByteArray) : Array ByteArray :=
  (Array.range (level.size / 2)).map fun i => d.node lvl i level[2 * i]! level[2 * i + 1]!

/-- The leaves padded to `nextPow2` of their count. -/
def base (d : Domain) (leaves : Array ByteArray) : Array ByteArray :=
  leaves ++ (Array.range (nextPow2 leaves.size - leaves.size)).map fun k => d.pad (leaves.size + k)

/-- Level `l` of the tree. -/
def lev (d : Domain) (leaves : Array ByteArray) : Nat → Array ByteArray
  | 0 => base d leaves
  | l + 1 => next d l (lev d leaves l)

theorem fold_eq (d : Domain) (leaves : Array ByteArray) : ∀ (k s : Nat),
    (List.range' s k).foldl (fun st lvl => (next d lvl st.1, st.2.push (next d lvl st.1)))
        (lev d leaves s, ((List.range (s + 1)).map (lev d leaves)).toArray) =
      (lev d leaves (s + k), ((List.range (s + k + 1)).map (lev d leaves)).toArray)
  | 0, s => by simp
  | k + 1, s => by
    rw [List.range'_succ, List.foldl_cons]
    have : (next d s (lev d leaves s), ((List.range (s + 1)).map (lev d leaves)).toArray.push (next d s (lev d leaves s))) =
        (lev d leaves (s + 1), ((List.range (s + 1 + 1)).map (lev d leaves)).toArray) := by
      rw [List.range_succ (n := s + 1), List.map_append]
      simp [lev]
    rw [this, fold_eq d leaves k (s + 1), show s + 1 + k = s + (k + 1) by omega]

theorem levels_eq (d : Domain) (leaves : Array ByteArray) (h : leaves.isEmpty = false) :
    d.levels leaves = ((List.range (log2ceil leaves.size + 1)).map (lev d leaves)).toArray := by
  unfold FV3.Domain.levels
  simp only [Id.run, h, Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl]
  have hs : [:log2ceil leaves.size].size = log2ceil leaves.size := by simp [Std.Legacy.Range.size]
  rw [hs]
  have := congrArg Prod.snd (fold_eq d leaves (log2ceil leaves.size) 0)
  simp only [Nat.zero_add] at this
  exact this

theorem le_pow_log2ceil (n : Nat) : n ≤ 2 ^ log2ceil n := by
  unfold log2ceil
  split
  · simp; omega
  · have := @Nat.lt_log2_self (n - 1)
    omega

theorem base_size (d : Domain) (leaves : Array ByteArray) : (base d leaves).size = 2 ^ log2ceil leaves.size := by
  have := le_pow_log2ceil leaves.size
  simp [base, nextPow2]
  omega

theorem next_size (d : Domain) (l : Nat) (a : Array ByteArray) : (next d l a).size = a.size / 2 := by
  simp [next]

theorem lev_size (d : Domain) (leaves : Array ByteArray) :
    ∀ l ≤ log2ceil leaves.size, (lev d leaves l).size = 2 ^ (log2ceil leaves.size - l)
  | 0, _ => by simp [lev, base_size]
  | l + 1, h => by
    rw [lev, next_size, lev_size d leaves l (by omega),
      show log2ceil leaves.size - l = (log2ceil leaves.size - (l + 1)) + 1 by omega, Nat.pow_succ]
    simp

theorem get!_mem {a : Array ByteArray} {i : Nat} (h : i < a.size) : a[i]! ∈ a := by
  rw [getElem!_pos a i h]
  exact Array.getElem_mem h

theorem hash_size (m : ByteArray) : (Sha512.hash m).size = 64 := Properties.NetworkCertifier.Bytes.hash_size m

theorem lev_width (d : Domain) (leaves : Array ByteArray) (h : ∀ x ∈ leaves, x.size = 64) :
    ∀ l, ∀ x ∈ lev d leaves l, x.size = 64
  | 0, x, hx => by
    simp only [lev, base, Array.mem_append, Array.mem_map] at hx
    rcases hx with hx | ⟨k, -, rfl⟩
    · exact h x hx
    · exact hash_size _
  | l + 1, x, hx => by
    simp only [lev, next, Array.mem_map] at hx
    obtain ⟨i, -, rfl⟩ := hx
    exact hash_size _

theorem levels_size (d : Domain) (leaves : Array ByteArray) (h : leaves.isEmpty = false) :
    (d.levels leaves).size = log2ceil leaves.size + 1 := by
  simp [levels_eq d leaves h]

theorem levels_get (d : Domain) (leaves : Array ByteArray) (h : leaves.isEmpty = false) {l : Nat}
    (hl : l ≤ log2ceil leaves.size) : (d.levels leaves)[l]! = lev d leaves l := by
  rw [getElem!_pos _ l (by rw [levels_size d leaves h]; omega)]
  simp [levels_eq d leaves h]

theorem root_eq (d : Domain) (leaves : Array ByteArray) (h : leaves.isEmpty = false) :
    d.root leaves = (lev d leaves (log2ceil leaves.size))[0]! := by
  unfold FV3.Domain.root
  rw [levels_eq d leaves h, List.back?_toArray, List.getLast?_map, List.getLast?_range]
  simp

theorem frame_snoc (tag : String) (ps : List ByteArray) (v : ByteArray) :
    bs (FV3.input tag (ps ++ [v])) = bs (FV3.input tag ps) ++ (bs (be64 v.size.toUInt64) ++ bs v) := by
  simp [FV3.input, List.foldl_append, bs_append]

/-- A frame's last part, after parts that agree, is determined. -/
theorem frame_last {tag : String} {ps : List ByteArray} {v v' : ByteArray}
    (h : FV3.input tag (ps ++ [v]) = FV3.input tag (ps ++ [v'])) : v = v' := by
  have e := congrArg bs h
  rw [frame_snoc, frame_snoc] at e
  exact bs_inj (split_bytes (List.append_cancel_left e) (by rw [be64_size, be64_size])).2

/-- A frame's two last parts, after parts that agree, are determined when the first of them has one width. -/
theorem frame_two {tag : String} {ps : List ByteArray} {a b a' b' : ByteArray} (ha : a.size = a'.size)
    (h : FV3.input tag (ps ++ [a, b]) = FV3.input tag (ps ++ [a', b'])) : a = a' ∧ b = b' := by
  have e := congrArg bs h
  rw [show ps ++ [a, b] = (ps ++ [a]) ++ [b] by simp, show ps ++ [a', b'] = (ps ++ [a']) ++ [b'] by simp,
    frame_snoc, frame_snoc, frame_snoc, frame_snoc] at e
  simp only [List.append_assoc] at e
  have e := List.append_cancel_left e
  obtain ⟨-, e⟩ := split_bytes e (by rw [be64_size, be64_size])
  obtain ⟨ha', e⟩ := split_bytes e ha
  obtain ⟨-, e⟩ := split_bytes e (by rw [be64_size, be64_size])
  exact ⟨ha', bs_inj e⟩

theorem node_hash (d : Domain) (l i : Nat) (a b : ByteArray) :
    d.node l i a b = Sha512.hash (FV3.input "node" ([d.id, FV3.uint l, FV3.uint i] ++ [a, b])) := rfl

theorem leaf_hash (d : Domain) (r : Nat) (schema : String) (v : ByteArray) :
    d.leaf r r schema v = Sha512.hash (FV3.input "leaf" ([d.id, FV3.uint r, FV3.uint r, schema.toUTF8] ++ [v])) := rfl

theorem next_get (d : Domain) (l : Nat) (a : Array ByteArray) {i : Nat} (hi : i < a.size / 2) :
    (next d l a)[i]! = d.node l i a[2 * i]! a[2 * i + 1]! := by
  rw [getElem!_pos _ i (by rw [next_size]; exact hi)]
  simp [next]

theorem isEmpty_of_size {vs : Array ByteArray} (h : vs.size ≠ 0) : vs.isEmpty = false := by
  cases he : vs.isEmpty
  · rfl
  · exact absurd (Array.isEmpty_iff_size_eq_zero.mp he) h

end MemoryChallenger.Properties.SecurityProofs.Tree
