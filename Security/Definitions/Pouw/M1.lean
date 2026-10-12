import Definitions.Pouw.Game.Params
import Definitions.Pouw.D1
import Mathlib.Data.Fintype.Sum

/-!
# `D1.m1`: the named milestone instance (trusted; §3.2's derivation, tt-attacks §4, working decisions W4 and W6)

* **Queries** (§3.2's `H(s, index, com(A_u), w(u))`, with ideal commitments):
  - unit `u`'s `E_L` (`m × 16`): `(s, u, A_u, w(u), c)` for `c < 2^21`;
  - the `E_R` of group `g = ⌊u/16⌋` on weight `w` (`16 × k`): `(s, w, g, c)` for `c < 2^20`;
  - weight `w`'s `F_L` (`k × 16`) and `F_R` (`16 × n`): `(s, w, c)` for `c < 2^21`.
  The index and the weight id are 64-bit fields and the group a 60-bit field. `A_u` enters as its block on
  `2^17 × 2^16`, which contains every in-domain shape, entry-wise mod 128 (injective on int7). A unit's `E_L`
  queries are asked for no other unit below `2^64` (red-team B1).
* **Distribution** (tt-attacks §4, W6 as fixed at 21:18Z): the outer factors `E_L` and `F_R` uniform on
  `{−3, …, 3}`, the inner factors `E_R` and `F_L` uniform on `{±1}`, all independent, one oracle answer each.
  Noise entries are in `[−48, 48]`.
* **Domain** `m1Domain`: `2^12 ≤ m ≤ 2^17` with `16 ∣ m`, `k = n = 2^16` (so `m·G ≤ 2^21`), 64-bit weight ids, and
  every unit's group serving at least `2^16` rows on its weight (red-team B3, B8).
* **`W_ref`** is a parameter of `m1`. `m1Wref` is the explicit `H_ref` schedule of llm-shapes §3, with the
  per-(epoch, weight) and per-(epoch, weight, group) products apportioned to units by rows.
-/

namespace Pouw.D1

open Finset

/-- An activation block on `2^17 × 2^16`, entry-wise mod 128. -/
abbrev M1Act : Type := Fin (2 ^ 17) → Fin (2 ^ 16) → Fin 128

/-- `m1`'s oracle queries: `E_L`'s `(s, u, A_u, w(u), c)`, `E_R`'s `(s, w, g, c)`, or the weight noise's `(s, w, c)`. -/
abbrev M1Q : Type :=
  (Salt × Fin (2 ^ 64) × M1Act × Fin (2 ^ 64) × Fin (2 ^ 21)) ⊕
    ((Salt × Fin (2 ^ 64) × Fin (2 ^ 60) × Fin (2 ^ 20)) ⊕ (Salt × Fin (2 ^ 64) × Fin (2 ^ 21)))

set_option synthInstance.maxSize 2048 in
instance instDecidableEqM1Q : DecidableEq M1Q := inferInstance

/-- An oracle answer: an outer-factor entry and a sign. -/
abbrev M1R : Type := Fin 7 × Bool

/-- A `b`-bit field. -/
def field (b x : ℕ) : Fin (2 ^ b) := ⟨x % 2 ^ b, Nat.mod_lt _ (Nat.two_pow_pos b)⟩

/-- `A`'s block on `2^17 × 2^16`, entry-wise mod 128. -/
def encAct (A : Mat) : M1Act := fun i l => ⟨(A i l % 128).toNat, by omega⟩

/-- An outer-factor entry, `a − 3 ∈ [−3, 3]`. -/
def m1Outer (a : M1R) : ℤ := (a.1 : ℤ) - 3

/-- An inner-factor entry, `±1`. -/
def m1Sign (a : M1R) : ℤ := if a.2 then 1 else -1

/-- Entry `i` of the answers `a` read through `f`; `0` past the end. -/
def m1Entry (f : M1R → ℤ) (a : List M1R) (i : ℕ) : ℤ := (a[i]?.map f).getD 0

/-- Unit `u`'s `E_L` queries. -/
def m1ELQ (s : Salt) (u : ℕ) (A : Mat) (w : ℕ) : List M1Q :=
  (List.finRange (2 ^ 21)).map fun c => Sum.inl (s, field 64 u, encAct A, field 64 w, c)

/-- The `E_R` queries of group `g` on weight `w`. -/
def m1ERQ (s : Salt) (w g : ℕ) : List M1Q :=
  (List.finRange (2 ^ 20)).map fun c => Sum.inr (Sum.inl (s, field 64 w, field 60 g, c))

/-- Weight `w`'s `(F_L, F_R)` queries. -/
def m1FQ (s : Salt) (w : ℕ) : List M1Q :=
  (List.finRange (2 ^ 21)).map fun c => Sum.inr (Sum.inr (s, field 64 w, c))

/-- Oracle queries per unit: `2^21` for `E_L`, `2^20` for its group's `E_R`, `2^21` for its weight's noise. -/
def m1Queries : ℕ := 2 ^ 21 + 2 ^ 20 + 2 ^ 21

/-- The named milestone instance, for a given `W_ref`: `E_L` (`m × 16`) and then `E_R` (`16 × k`) are read row by row
from the activation answers, `F_L` (`k × 16`) and then `F_R` (`16 × n`) from the weight answers. -/
def m1 (Wref : Layout → ℕ → ℕ) : Params M1Q M1R Salt where
  r := 16
  d := 16
  qAct s u A w := m1ELQ s u A w ++ m1ERQ s w (u / 16)
  qWt s _ w := m1FQ s w
  expAct sh a _ := (fun i t => m1Entry m1Outer a (i * 16 + t), fun t l => m1Entry m1Sign a (2 ^ 21 + (t * sh.k + l)))
  expWt sh b := (fun l t => m1Entry m1Sign b (l * 16 + t), fun t j => m1Entry m1Outer b (16 * sh.k + (t * sh.n + j)))
  Wref := Wref

/-- The milestone's shapes: `2^12 ≤ m ≤ 2^17`, `16 ∣ m`, `k = n = 2^16`. -/
def M1Shape (sh : Shape) : Prop := 2 ^ 12 ≤ sh.m ∧ sh.m ≤ 2 ^ 17 ∧ 16 ∣ sh.m ∧ sh.k = 2 ^ 16 ∧ sh.n = 2 ^ 16

/-- The rows of `u`'s group `⌊u/16⌋` on its weight: `∑_{u' : ⌊u'/16⌋ = ⌊u/16⌋, w(u') = w(u)} m_{u'}`. -/
def m1GroupRows (L : Layout) (u : ℕ) : ℕ :=
  ∑ u' ∈ (range L.N).filter (fun u' => u' / 16 = u / 16 ∧ L.wid u' = L.wid u), (L.shape u').m

/-- The milestone domain: milestone shapes, 64-bit weight ids, and at least `2^16` rows in each unit's group on its
weight (hence on its weight). -/
def m1Domain (L : Layout) : Prop :=
  ∀ u < L.N, M1Shape (L.shape u) ∧ L.wid u < 2 ^ 64 ∧ 2 ^ 16 ≤ m1GroupRows L u

/-- A unit's share, by rows and rounded up, of `work` spread over `rows` rows: `⌈work·m_u/rows⌉` (`0` if `rows = 0`). -/
def rowShare (work mu rows : ℕ) : ℕ := (work * mu + rows - 1) / rows

/-- `H_ref`'s schedule under W1 (llm-shapes §3, at `r = d = 16`), in RTX 4090 units, for unit `u` of shape `(m, k, n)`:
* `m·k·n`: the main GEMM `(A + E)·(B + F)`, int8 `mma` k16, the running sums read from the accumulator;
* `16·m·k`: `E = E_L·E_R`; `12·m·k`: packing `E` to int8 (0.75 PRMT per entry); `4·m·k`: `A + E` on biased bytes
  (one IADD3 per 4 entries); `4·m·k`: back to signed int8 (one LOP3 `xor 0x80808080` per 4 entries);
  `16·m·k`: `X1 = (A + E)·F_L`, `F_L` appended to the GEMM's `B` operand;
* `6·16·m·n`: `D = X1·F_R + E_L·G_g` in three limb groups of inner width `2r`; `32·m·n`: the combine `C = C' − D`
  (two IMAD or LEA per entry);
* `u`'s share of its group's `G_g = E_R,g·B'` (`16·k·n` per (epoch, weight, group));
* `u`'s share of its weight's `F = F_L·F_R` (`16·k·n`) and of packing `F` and forming `B + F` as `A + E` is formed
  (`20·k·n`) per (epoch, weight);
* `576·m`: the int8 limb split of `X1` (entries up to about `2^23`), and `u`'s share of the limb split of its group's
  `G_g` (`576·n` per (epoch, weight, group)). Red-team P7-4: the split was free only under the old byte-move rule;
  Daniel's 03:35Z rule prices sub-word extraction like PRMT.
Per unit this is red-team B3's count `mkn + 2mkr + 6mrn + 20mk + 32mn`, plus P7-4's `576·m`.
Shares are rounded up, so the units of a group or of a weight together cover its work (`Sanity.rowShare_covers`). -/
def m1Wref (L : Layout) (u : ℕ) : ℕ :=
  let m := (L.shape u).m
  let k := (L.shape u).k
  let n := (L.shape u).n
  m * k * n + 16 * m * k + 12 * m * k + 4 * m * k + 4 * m * k + 16 * m * k + 6 * 16 * m * n + 32 * m * n +
    rowShare (16 * k * n) m (m1GroupRows L u) + rowShare ((16 + 20) * k * n) m (L.rowsOf (L.wid u)) +
    576 * m + rowShare (576 * n) m (m1GroupRows L u)

/-- The schedule's overhead bound on the domain, `1 + 233/2^16`. -/
noncomputable def m1Omega : ℝ := 65769 / 65536

end Pouw.D1
