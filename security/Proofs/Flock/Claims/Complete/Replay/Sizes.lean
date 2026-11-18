module

public import Proofs.Flock.Claims.Complete.Replay.Steps
import Proofs.Flock.Soundness.Refine.Frames

/-!
# The bytes each absorb adds and a squeeze records

A header is 16 bytes (`header_size`); `padded b` is `b` and at most 15 zeros (`padded_size_le`); so a label or a byte
string adds its own bytes and at most 31 (`pLabel_size_le`, `pBytes_size_le`), an F128 32 (`pF_size`) and a slice
of `n` F128s `16 + 16·n` (`pSlice_size`). A squeeze records its pending bytes and a header as its round's bytes and
leaves nothing pending (`sent_bytes_size`, `afterSq_pending`).

* `RoundsLe E B t`: each recorded round of `t` retains at most `B` bytes, the first at most `B + E`; a squeeze of at
  most `B − 16` pending bytes keeps it (`RoundsLe.afterSq`), the first squeeze of at most `B + E − 16`
  (`RoundsLe.afterSq0`), and an absorb, which records no round, does too.
-/

@[expose] public section

namespace FlockComplete.Replay

open Flock Transcript

theorem le64_size (x : UInt64) : (le64 x).size = 8 := by
  show (le64 x).data.toList.length = 8
  rw [FlockSoundness.Refine.le64_data]
  simp

theorem header_size (op kind : UInt8) (len : ℕ) : (header op kind len).size = 16 := by
  unfold header
  rw [ByteArray.size_append, le64_size]
  rfl

theorem foldl_push_zero_size : ∀ (l : List ℕ) (out : ByteArray),
    (l.foldl (fun out _ => out.push 0) out).size = out.size + l.length
  | [], out => by simp
  | _ :: l, out => by
    rw [List.foldl_cons, foldl_push_zero_size l, ByteArray.size_push, List.length_cons]
    omega

theorem padded_size (b : ByteArray) : (padded b).size = b.size + (16 - b.size % 16) % 16 := by
  unfold padded
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl, Id.run_pure, pure_bind]
  rw [foldl_push_zero_size]
  simp [Std.Legacy.Range.size]

theorem padded_size_le (b : ByteArray) : (padded b).size ≤ b.size + 15 := by
  rw [padded_size]
  omega

theorem toBytes_size (a : F128) : a.toBytes.size = 16 := by
  show (Array.ofFn _).size = 16
  exact Array.size_ofFn

theorem pushBytes_size (P : ByteArray) (a : F128) : (F128.pushBytes P a).size = P.size + 16 := by
  unfold F128.pushBytes
  rw [ByteArray.size_append, toBytes_size]

theorem foldl_pushBytes_size : ∀ (l : List F128) (P : ByteArray),
    (l.foldl F128.pushBytes P).size = P.size + 16 * l.length
  | [], P => by simp
  | a :: l, P => by
    rw [List.foldl_cons, foldl_pushBytes_size l, pushBytes_size, List.length_cons]
    omega

theorem pF_size (v : F128) (P : ByteArray) : (pF v P).size = P.size + 32 := by
  unfold pF
  rw [pushBytes_size, ByteArray.size_append, header_size]

theorem pSlice_size (vs : Array F128) (P : ByteArray) : (pSlice vs P).size = P.size + 16 + 16 * vs.size := by
  unfold pSlice
  rw [← Array.foldl_toList, foldl_pushBytes_size, ByteArray.size_append, header_size, Array.length_toList]

theorem pBytes_size_le (b P : ByteArray) : (pBytes b P).size ≤ P.size + b.size + 31 := by
  unfold pBytes
  rw [ByteArray.size_append, ByteArray.size_append, header_size]
  have := padded_size_le b
  omega

theorem pLabel_size_le (s : String) (P : ByteArray) : (pLabel s P).size ≤ P.size + s.toUTF8.size + 31 := by
  unfold pLabel
  rw [ByteArray.size_append, ByteArray.size_append, header_size]
  have := padded_size_le s.toUTF8
  omega

theorem framed_size (t : Transcript) (kind : UInt8) (n : ℕ) : (framed t kind n).size = t.pending.size + 16 := by
  unfold framed
  rw [ByteArray.size_append, header_size]

theorem sent_bytes_size (t : Transcript) (kind : UInt8) (n : ℕ) (coin : ℕ → F128) :
    ((sent t kind n coin).bytes.getD ByteArray.empty).size = t.pending.size + 16 := by
  show (framed t kind n).size = _
  exact framed_size t kind n

theorem afterSq_pending (t : Transcript) (kind : UInt8) (n : ℕ) (coin : ℕ → F128) :
    (afterSq t kind n coin).pending = ByteArray.empty := rfl

theorem init_pending_size_le (rs : Array Round) (d : ByteArray) :
    (Transcript.init rs d).pending.size ≤ d.size + 31 := by
  show (header 1 0 d.size ++ padded d).size ≤ _
  rw [ByteArray.size_append, header_size]
  have := padded_size_le d
  omega

/-- Each recorded round of `t` retains at most `B` bytes, the first at most `B + E`. -/
def RoundsLe (E B : ℕ) (t : Transcript) : Prop :=
  ∀ (k : ℕ) (hk : k < t.rounds.size),
    ((t.rounds[k]'hk).bytes.getD ByteArray.empty).size ≤ B + (if k = 0 then E else 0)

namespace RoundsLe

variable {E B : ℕ}

theorem afterSq {t : Transcript} (h : RoundsLe E B t) (hp : t.pending.size + 16 ≤ B) (kind : UInt8) (n : ℕ)
    (coin : ℕ → F128) : RoundsLe E B (Replay.afterSq t kind n coin) := by
  intro k hk
  change k < (t.rounds.push (sent t kind n coin)).size at hk
  change (((t.rounds.push (sent t kind n coin))[k]'hk).bytes.getD ByteArray.empty).size ≤ _
  rw [Array.getElem_push]
  split
  · rename_i hlt
    exact h k hlt
  · rw [sent_bytes_size]
    split <;> omega

theorem afterSq0 {t : Transcript} (h : RoundsLe E B t) (h0 : t.rounds.size = 0) (hp : t.pending.size + 16 ≤ B + E)
    (kind : UInt8) (n : ℕ) (coin : ℕ → F128) : RoundsLe E B (Replay.afterSq t kind n coin) := by
  intro k hk
  change k < (t.rounds.push (sent t kind n coin)).size at hk
  change (((t.rounds.push (sent t kind n coin))[k]'hk).bytes.getD ByteArray.empty).size ≤ _
  rw [Array.size_push] at hk
  rw [Array.getElem_push]
  split
  · rename_i hlt
    exact absurd hlt (by omega)
  · rename_i hlt
    have hk0 : k = 0 := by omega
    rw [sent_bytes_size, if_pos hk0]
    exact hp

theorem of_rounds {t t' : Transcript} (h : RoundsLe E B t) (hr : t'.rounds = t.rounds) : RoundsLe E B t' := by
  intro k hk
  have key : ∀ (a b : Array Round) (e : a = b) (hk : k < a.size),
      (a[k]'hk).bytes = (b[k]'(e ▸ hk)).bytes := by
    intro a b e
    subst e
    intro hk
    rfl
  rw [key _ _ hr hk]
  exact h k _

theorem init (rs : Array Round) (d : ByteArray) (h : ∀ x ∈ rs, (x.bytes.getD ByteArray.empty).size ≤ B) :
    RoundsLe E B (Transcript.init rs d) := by
  intro k hk
  have := h _ (Array.getElem_mem hk)
  split <;> omega

end RoundsLe

end FlockComplete.Replay
