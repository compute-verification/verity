module

public import Definitions.Core.Assumptions
public import Definitions.NetworkCertifier.Guarantees.ResponseCap
meta import Definitions.NetworkCertifier.ResponseCap

/-!
# Each response at most its request's `max_tokens` tokens

`Definitions.NetworkCertifier.ResponseCap`. On arrivals held to `m` data cells, a lane from its cell `j` sends at most
`m - j` data cells (`dataCells_lane_capped`): a data cell past the first `m` is late, so the lane fails there and pads
to its end. So the network certifier's code under the cap sends each connection at most its admitted `max_tokens`
(`dataCells_cappedCode`), which is at most the cap and at most the request's own (`admitted_le`):
`Guarantees.ResponseCapped`. Under `trusted-host` the network certifier's run is that code, so its run sends no more
(`responseCapped_run`).

The `#guard` holds the capped lane to the Python network certifier's cells under a cap of four (the test harness of
`verity/core/protocols/certifier/tests/test_network_certifier_warden.py`, `CAPPED`: r = 2, D = 3, a connection
admitted at bucket 1): a request of `max_tokens` two answered with three ids gets two data cells in bucket 4, then the
fail cell and padding in bucket 5.
-/

@[expose] public section

namespace NetTiming

variable {Cell : Type}

theorem dataCells_cons (x : ℕ × CellView Cell) (v : List (ℕ × CellView Cell)) :
    dataCells (x :: v) = dataCells v + if x.2.isData then 1 else 0 := by
  simp [dataCells, List.countP_cons]

theorem dataCells_pads (t : ℕ → ℕ) (j k : ℕ) :
    dataCells ((List.range' j k).map fun i => (t i, (CellView.pad : CellView Cell))) = 0 := by
  simp [dataCells, List.countP_eq_zero, CellView.isData]

/-- A lane on arrivals held to `m` data cells sends at most `m - j` of them from its cell `j`. -/
theorem dataCells_lane_capped (a : Arrivals Cell) (t : ℕ → ℕ) (m : ℕ) :
    ∀ k j, dataCells (certifierLane (a.capped t m) t j k) ≤ m - j := by
  intro k
  induction k with
  | zero => intro j; simp [certifierLane, dataCells]
  | succ k ih =>
    intro j
    have hs : (a.capped t m).stream = a.stream := rfl
    rw [certifierLane, hs]
    split
    · rw [dataCells_cons]
      have := ih (j + 1)
      simp only [CellView.isData, Bool.false_eq_true, ↓reduceIte, Nat.add_zero]
      omega
    split
    · rw [dataCells_cons, dataCells_pads]
      simp [CellView.isData]
    · rename_i h2
      rw [dataCells_cons]
      have := ih (j + 1)
      by_cases hj : j < m
      · split <;> omega
      · have hl : ¬ j < a.stream.length := fun hl => h2 (by simp [Arrivals.capped, hl, Nat.le_of_not_lt hj])
        simp [streamCell, hl, CellView.isData]
        omega

/-- The code under the cap sends each connection at most its admitted `max_tokens` data cells. -/
theorem dataCells_cappedCode {κ : Type} (s : CertifierSchedule κ) (cap : ℕ) (mt : κ → Option ℕ)
    (arr : κ → Arrivals Cell) (c : κ) : dataCells (cappedCode s cap mt arr c) ≤ admitted cap (mt c) := by
  have := dataCells_lane_capped (arr c) (s.bucket c) (admitted cap (mt c)) (s.cells c) 0
  simpa [cappedCode, certifierCode] using this

/-- A request's admitted `max_tokens` is at most the cap, and at most its own (none for one not read). -/
theorem admitted_le (cap : ℕ) (o : Option ℕ) : admitted cap o ≤ cap ∧ admitted cap o ≤ o.getD 0 := by
  cases o with
  | none => simp [admitted]
  | some m => by_cases h : m ≤ cap <;> simp [admitted, h]

theorem responseCapped : Guarantees.ResponseCapped := by
  intro κ Cell s cap mt arr c
  have h := dataCells_cappedCode s cap mt arr c
  obtain ⟨h1, h2⟩ := admitted_le cap (mt c)
  exact ⟨Nat.le_trans h h1, Nat.le_trans h h2⟩

/-- Under `trusted-host`, a network certifier running the code under the cap sends each connection at most the cap's
and its request's `max_tokens` data cells. -/
theorem responseCapped_run {κ Cell Srv Dev : Type} (s : CertifierSchedule κ) (cap : ℕ) (mt : κ → Option ℕ)
    (arr : Srv → κ → Arrivals Cell) (dev : Dev) (run : Dev → Srv → κ → List (ℕ × CellView Cell))
    (htrust : Verity.Assumptions.TrustedHost dev run fun σ => cappedCode s cap mt (arr σ)) (σ : Srv) (c : κ) :
    dataCells (run dev σ c) ≤ cap ∧ dataCells (run dev σ c) ≤ (mt c).getD 0 := by
  rw [htrust σ]
  exact responseCapped s cap mt (arr σ) c

#guard ((cappedCode (⟨2, 3, fun _ => 1, fun _ => 4⟩ : CertifierSchedule Unit) 4 (fun _ => some 2)
    (fun _ => ⟨[0, 1, 2], fun _ => 0⟩) ()).map fun x =>
      (x.1, match x.2 with | .data _ => "data" | .pad => "pad" | .fail => "fail")) ==
  [(4, "data"), (4, "data"), (5, "fail"), (5, "pad")]

end NetTiming

namespace NetTiming.SecurityProofs

theorem ResponseCapped : NetTiming.Guarantees.ResponseCapped := NetTiming.responseCapped

end NetTiming.SecurityProofs
