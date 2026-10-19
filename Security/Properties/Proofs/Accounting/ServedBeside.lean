import Properties.Proofs.Accounting.Served

/-!
# A served `y` that reads a product beside the receipted one meets `CreditTie`

`witBeside` is `witLayout`'s one call in `perCall`: its coin, an A row, a noised product `m` of the coin and the A row,
a receipt reading `m`, a product `p` of the A row with itself that reads no coin, `z = m` (an affine peel with no
correction), and the served `y = 0·z + 1·p`, committed the round after its own. It meets `Tie` in `perCall`
(`witTieBeside`) and `DerivedZ` (`witBeside_derivedZ`), so `CreditTie` (`witBeside_creditTie`), while its `y` is `p`'s
value whatever `z` is (`witBeside_y_is_p`) and `p` reads no coin (`witBeside_p_unnoised`). `DerivedZ` asks that `z`
reach `y` (`Feeds`), not that `y` be a function of `z`.
-/

namespace Properties.Served

open Accounting.Compose Accounting.Served

/-- The coin, the A row, `m = A + coin·A`, the receipt `m`, `p = A + A·A`, `z = m` and the served `y = 0·z + 1·p`. -/
def witBeside : Declared Unit where
  dag := [witGate (.input .coin ()) [] false, witGate (.input .run ()) [] false,
    witGate (.mma .s8 .s8 1 false) [1, 0, 1] false, witGate (.affine 0 [1]) [2] true,
    witGate (.mma .s8 .s8 1 false) [1, 1, 1] false, witGate (.affine 0 [1]) [2] false,
    witGate (.affine 0 [0, 1]) [5, 4] true]
  calls := 1
  call := fun _ => 0
  rounds := 1
  round := fun _ => 0
  coinMode := fun _ => .live
  commitRound := fun i => if i = 0 ∨ i = 1 ∨ i = 4 then 0 else 1
  seedNames := fun _ _ => false
  units := 1
  unit := fun _ => 0

/-- `witBeside`'s classes: the coin, the A row, the receipt, the served `y`, and the rest inner. -/
def witClsBeside : ℕ → Cls
  | 0 => .coin
  | 1 => .act
  | 3 => .receipt
  | 6 => .out
  | _ => .inner

theorem witTieBeside : Tie witLayout .perCall witBeside witClsBeside where
  calls := by decide +kernel
  rounds := by decide +kernel
  round := fun c hc => by
    obtain rfl : c = 0 := by
      have h1 : c < 1 := hc
      omega
    decide +kernel
  inRange := fun _ _ => Nat.zero_lt_one
  live := fun _ _ => rfl
  coin := by decide +kernel
  registered := by decide +kernel
  act := fun i hi h =>
    ⟨(by decide +kernel : ∀ i, i < 7 → (witClsBeside i = .act ∨ witClsBeside i = .served) →
        witBeside.binding i = some .run) i hi h, fun _ _ _ => rfl⟩
  computed := by decide +kernel
  committed := by decide +kernel
  noised := by decide +kernel
  forms := by decide +kernel
  linked := fun _ _ _ => ⟨6, by decide +kernel, rfl, rfl⟩
  commit := by decide +kernel
  inner := by decide +kernel

theorem witBeside_derivedZ : DerivedZ witBeside witClsBeside := by
  intro i hi hout
  have h6 : i = 6 := by
    have h7 : i < 7 := hi
    interval_cases i <;> first | rfl | exact absurd hout (by decide)
  subst h6
  refine ⟨5, 2, 2, 3, rfl, Relation.ReflTransGen.single (by decide +kernel),
    ⟨witGate (.affine 0 [1]) [2] false, 0, [], [], rfl, rfl, rfl, fun t ht => absurd ht (List.not_mem_nil)⟩,
    rfl, ⟨witGate (.mma .s8 .s8 1 false) [1, 0, 1] false, rfl, trivial⟩, Relation.ReflTransGen.refl,
    by decide +kernel, by decide +kernel, rfl, rfl, Relation.ReflTransGen.single (by decide +kernel)⟩

theorem witBeside_creditTie : CreditTie witLayout .perCall witBeside witClsBeside :=
  ⟨witTieBeside, rfl, witBeside_derivedZ⟩

/-- The served `y` is `p`'s value, whatever `z`'s is. -/
theorem witBeside_y_is_p (x : Unit → ℤ) (z p : ℤ) :
    witBeside.dag[6]? = some (witGate (.affine 0 [0, 1]) [5, 4] true) ∧
      Accounting.Gates.Op.raw (.affine 0 [0, 1] : Accounting.Gates.Op Unit) x [z, p] = p :=
  ⟨rfl, by simp [Accounting.Gates.Op.raw]⟩

/-- `p` reads no coin, and `m`, the receipted product, does. -/
theorem witBeside_p_unnoised : (coinMark witBeside).getD 4 0 = 0 ∧ (coinMark witBeside).getD 2 0 = 1 := by
  decide +kernel

end Properties.Served
