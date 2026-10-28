import Definitions.Core.Requirement
import Definitions.Pous.Game.Prob

/-!
# The store the bandwidth and rate games draw (trusted)

The bandwidth audit's game (`Bandwidth/Game.lean`) and the rate audit's (`RateAudit.lean`) draw a store's contents
uniformly: the payloads of its `N` registered blocks, `b` bits each (`Bandwidth.Payloads`, `Rate.Payloads`, both
`Fin N → Bits b`). They are the audit's store (`Accounting.Requirement.Store`, I3) as the games read it:

* the store a game's draw `E` stands for is `Store.ofPayloads E`: `N` registered blocks holding `E`'s payloads, none
  written or freed, so its resident blocks are its registered ones at every step (`resident_ofPayloads`);
* a store's payloads as a game reads them are `Store.payloads`: its first `N` blocks' payloads, each at its low `b` bits.

A game's draw is the payloads of the store it stands for (`payloads_ofPayloads`), and two draws stand for one store only
if they are one draw (`ofPayloads_injective`): a uniform game store is a uniform registered store of `N` blocks of `b`
bits, read as its payloads.
-/

namespace Accounting.Requirement

open Pous (Bits)

/-- A block's `b` bits as its payload's number, bit `i` its `2^i` place (`Store.payload`'s form). -/
def payloadOfBits {b : ℕ} (v : Bits b) : ℕ := (BitVec.ofBoolListLE (List.ofFn v)).toNat

namespace Store

/-- **The store a game's draw stands for**: `N` registered blocks whose payloads are `E`'s, none written or freed. -/
def ofPayloads {N b : ℕ} (E : Fin N → Bits b) : Store where
  registered := N
  written _ := ∅
  freed _ := ∅
  payload j := if h : j < N then payloadOfBits (E ⟨j, h⟩) else 0

/-- **A store's payloads as a game reads them**: its first `N` blocks' payloads, each at its low `b` bits. -/
def payloads (σ : Store) (N b : ℕ) : Fin N → Bits b := fun j i => (σ.payload j).testBit i

theorem ofPayloads_registered {N b : ℕ} (E : Fin N → Bits b) : (ofPayloads E).registered = N := rfl

/-- The store a game draws is resident whole at every step: its registered blocks, nothing written or freed. -/
theorem resident_ofPayloads {N b : ℕ} (E : Fin N → Bits b) (s : ℕ) :
    (ofPayloads E).resident s = Finset.range N := by
  ext x
  simp [resident, ofPayloads]

/-- A game's draw is the payloads of the store it stands for. -/
theorem payloads_ofPayloads {N b : ℕ} (E : Fin N → Bits b) : (ofPayloads E).payloads N b = E := by
  funext j i
  simp only [payloads, ofPayloads, j.isLt, ↓reduceDIte, payloadOfBits]
  rw [BitVec.testBit_toNat, BitVec.getLsbD_ofBoolListLE]
  simp

/-- Two draws that stand for one store are one draw. -/
theorem ofPayloads_injective {N b : ℕ} : Function.Injective (ofPayloads : (Fin N → Bits b) → Store) :=
  fun E E' h => by rw [← payloads_ofPayloads E, ← payloads_ofPayloads E', h]

end Store

end Accounting.Requirement
