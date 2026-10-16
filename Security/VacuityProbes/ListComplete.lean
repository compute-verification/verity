import Definitions.Claims

/-!
Scratch probe (vacuity sweep, not for merge): `ListComplete` at the empty input forces every trace into
`decode []`, so no decoder meets it when the trace type is infinite, and `Flock.KS` and `Flock.Bits` then hold for
every deployment by an empty `∀ decode`.
-/

namespace VacuityProbe

open Claims Claims.Flock

theorem listComplete_mem_nil {Col T : Type} [DecidableEq Col] {enc : T → ℕ → Col} {agree : ℝ}
    {decode : List (ℕ × Col) → List T} (h : ListComplete enc agree decode) (M : T) : M ∈ decode [] :=
  h [] M (by simp) (by simp)

theorem no_listComplete_of_infinite {Col T : Type} [DecidableEq Col] [Infinite T] (enc : T → ℕ → Col) (agree : ℝ)
    (decode : List (ℕ × Col) → List T) : ¬ ListComplete enc agree decode := by
  classical
  intro h
  obtain ⟨M, hM⟩ := Infinite.exists_notMem_finset (decode []).toFinset
  exact hM (List.mem_toFinset.2 (listComplete_mem_nil h M))

theorem ks_of_infinite {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    [Infinite T] (H : Msg → Msg) (D : Deploy Coin Col T) : Flock.KS H D :=
  fun decode h => absurd h (no_listComplete_of_infinite D.enc D.agree decode)

theorem bits_of_infinite {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    [Infinite T] (H : Msg → Msg) (D : Deploy Coin Col T) : Flock.Bits H D :=
  fun decode h => absurd h (no_listComplete_of_infinite D.enc D.agree decode)

/-- At a finite trace type a decoder meets the spec only if it lists the whole type at the empty input. -/
theorem listComplete_card_le {Col T : Type} [DecidableEq Col] [Fintype T] [DecidableEq T] {enc : T → ℕ → Col}
    {agree : ℝ} {decode : List (ℕ × Col) → List T} (h : ListComplete enc agree decode) :
    Fintype.card T ≤ (decode []).length := by
  calc Fintype.card T = (Finset.univ : Finset T).card := Finset.card_univ.symm
    _ ≤ (decode []).toFinset.card :=
        Finset.card_le_card fun M _ => List.mem_toFinset.2 (listComplete_mem_nil h M)
    _ ≤ (decode []).length := List.toFinset_card_le _

end VacuityProbe

#print axioms VacuityProbe.ks_of_infinite
#print axioms VacuityProbe.bits_of_infinite
#print axioms VacuityProbe.listComplete_card_le
