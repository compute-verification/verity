import Proofs.Flock.Claims.Complete.Encode

/-!
# Variable-length numbers in the recursive verifier's messages

The numbers the recursive verifier's messages carry as lengths (`joinR`'s rows, `finalR`'s trailer) are in
variable-length bytes (`encV`, LEB128), so every encoder reads back at every length (`rdVL_encV`, `rdVLs_encVL`,
`rdVB_encVB`), with no length needing to fit a word.
-/

namespace FlockDeploy

/-- **A number's variable-length bytes** (LEB128): seven bits a byte, the low ones first, the high bit set on every
byte but the last. -/
def encV (n : ℕ) : Claims.Msg :=
  if n < 128 then [n.toUInt8] else (n % 128 + 128).toUInt8 :: encV (n / 128)
termination_by n
decreasing_by omega

/-- `encV` read off the front of a message. -/
def rdVL : Claims.Msg → Option (ℕ × Claims.Msg)
  | [] => none
  | b :: t => if b.toNat < 128 then some (b.toNat, t) else (rdVL t).map fun p => (b.toNat - 128 + 128 * p.1, p.2)

def rdV : Rd ℕ := fun m => rdVL m

/-- A list, after its length's variable-length bytes. -/
def encVL {α : Type} (e : α → Claims.Msg) (l : List α) : Claims.Msg := encV l.length ++ l.flatMap e

def rdVLs {α : Type} (r : Rd α) : Rd (List α) := do rdListN r (← rdV)

/-- Bytes, after their length's variable-length bytes. -/
def encVB (b : ByteArray) : Claims.Msg := encV b.size ++ b.data.toList

def rdVB : Rd ByteArray := do
  let n ← rdV
  pure ⟨(← rdTake n).toArray⟩

theorem rdVL_encV : ∀ (n : ℕ) (rest : Claims.Msg), rdVL (encV n ++ rest) = some (n, rest) := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    intro rest
    rw [encV]
    split_ifs with h
    · have hb : (n.toUInt8).toNat = n := by
        rw [Nat.toUInt8, UInt8.toNat_ofNat_of_lt' (by unfold UInt8.size; omega)]
      simp only [List.singleton_append, rdVL, hb, h, ↓reduceIte]
    · have hb : ((n % 128 + 128).toUInt8).toNat = n % 128 + 128 := by
        rw [Nat.toUInt8, UInt8.toNat_ofNat_of_lt' (by unfold UInt8.size; omega)]
      have hlt : n / 128 < n := Nat.div_lt_self (by omega) (by omega)
      simp only [List.cons_append, rdVL, hb, show ¬(n % 128 + 128 < 128) by omega, ↓reduceIte, ih _ hlt rest,
        Option.map_some, Nat.add_sub_cancel, Option.some.injEq, Prod.mk.injEq, and_true]
      omega

theorem rdVLs_encVL {α : Type} {r : Rd α} {e : α → Claims.Msg} {l : List α}
    (hr : ∀ a ∈ l, ∀ rest, r.run (e a ++ rest) = some (a, rest)) (rest : Claims.Msg) :
    (rdVLs r).run (encVL e l ++ rest) = some (l, rest) := by
  have hv : rdV.run (encV l.length ++ (l.flatMap e ++ rest)) = some (l.length, l.flatMap e ++ rest) :=
    rdVL_encV _ _
  simp only [rdVLs, encVL, List.append_assoc, StateT.run_bind, hv, Option.bind_eq_bind, Option.bind_some]
  exact FlockComplete.rdListN_encList l hr rest

theorem rdVB_encVB (b : ByteArray) (rest : Claims.Msg) : rdVB.run (encVB b ++ rest) = some (b, rest) := by
  have h := FlockComplete.rdTake_append b.data.toList rest
  simp only [Array.length_toList, ByteArray.size_data] at h
  have hv : rdV.run (encV b.size ++ (b.data.toList ++ rest)) = some (b.size, b.data.toList ++ rest) :=
    rdVL_encV _ _
  simp only [rdVB, encVB, List.append_assoc, StateT.run_bind, hv, Option.bind_eq_bind, Option.bind_some, h]
  rfl

end FlockDeploy
