/-!
The SHA lane's floor on h, counted (hash_sha.py part F): under ShaWrites, with DistinctWrites the only credit past
hashing (class C), no design holds T1 at 1% below h = 108.6 W1 a stock MAC.

Quantities are a stock MAC's, in hundredths: `r` is honest's W1 that needs credit, `c` the credit past hashing,
`bits` the committed bits, `comp` the compressions and `h` the hashing W1. ShaWrites: a compression absorbs at most
1,024 message bits and costs at least `R` registers, 32 W1 each. DistinctWrites: a committed bit is credited at most
1 W1. T1 is r - c. Core Lean only: `lean HashFloor.lean`.
-/

theorem hash_floor (r c bits comp h R : Nat)
    (registers : 3510 ≤ R)
    (absorb : bits ≤ 1024 * comp)
    (cost : 32 * R * comp ≤ h)
    (credit : c ≤ bits)
    (noise : 100 ≤ r)
    (t1 : r ≤ c + 1) :
    10860 ≤ h := by
  have hb : 99 ≤ bits := by omega
  have lo : 3510 * 99 ≤ R * bits := Nat.mul_le_mul registers hb
  have hi : R * bits ≤ R * (1024 * comp) := Nat.mul_le_mul_left R absorb
  rw [Nat.mul_left_comm] at hi
  rw [Nat.mul_assoc] at cost
  generalize R * comp = k at hi cost
  generalize R * bits = j at lo hi
  omega

#print axioms hash_floor
