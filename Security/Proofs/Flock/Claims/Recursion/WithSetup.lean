import Proofs.Flock.Claims.Recursion.KSLin2

/-!
# Route 2C with the inner setup's calls among the recorded ones

An accepted inner run's calls are the outer witnesses' recorded ones and the inner verifier's own, at the inner
statement (its setup's, which no witness need record: `RecDeploy.setupM_not_layoutBTops`) and on the inner messages.
`Deploy.withSetup H D` appends those calls (`Deploy.innerSetup`, `Deploy.innerCalls`), each with `H`'s answer, to the
recorded ones; it changes `calls` only, so every other field, and with them `E`, `R`, `κ`, `γ` and the run, are
`D`'s.

* `callsTrue_withSetup`, `callsNil_withSetup`: the setup's calls are true evaluations, so the recorded ones stay true.
* `callsLogColl_withSetup`, `innerAcceptsCollAt_withSetup`: a fact at `D` holds at `D.withSetup H`, the more calls
  holding a collision wherever fewer do (`findColl_isSome_of_sub`).
* `Bound.ksLin2_of_flockLin_binds_setup`: `Bound.ksLin2_of_flockLin_binds` at `D.withSetup H`, from the facts at `D`
  but `CallsLogColl` and `ForksBindAt`, which it reads at `D.withSetup H`.
-/

namespace Claims.Recursion

open Claims

section Transfer

variable {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {H : Msg → Msg} {D : Deploy Coin}
  {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem withSetup_calls (H : Msg → Msg) (D : Deploy Coin) (x : Msg) (ws : List Flock.Wit) :
    (D.withSetup H).calls x ws = D.calls x ws ++ (D.innerSetup (xI D x) ++ D.innerCalls H x ws).map fun u => (u, H u) :=
  rfl

theorem setup_true (H : Msg → Msg) (us : List Msg) : ∀ e ∈ us.map (fun u => (u, H u)), H e.1 = e.2 := by
  intro e he
  obtain ⟨u, -, rfl⟩ := List.mem_map.1 he
  rfl

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem withSetup_true {x : Msg} {ws : List Flock.Wit} (h : ∀ e ∈ D.calls x ws, H e.1 = e.2) :
    ∀ e ∈ (D.withSetup H).calls x ws, H e.1 = e.2 := fun e he =>
  (List.mem_append.1 he).elim (h e) (setup_true H _ e)

theorem callsTrue_withSetup (hT : CallsTrue H D I DO) : CallsTrue H (D.withSetup H) I DO :=
  fun decO x Ω P ω r c ws hg => withSetup_true (hT decO x Ω P ω r c ws hg)

theorem callsNil_withSetup (hN : CallsNil H D) : CallsNil H (D.withSetup H) := fun x => withSetup_true (hN x)

/-- A collision `findColl` finds among `D`'s recorded calls, which are true, is one it finds among `D.withSetup H`'s. -/
theorem coll_withSetup {x : Msg} {ws : List Flock.Wit} (ht : ∀ e ∈ D.calls x ws, H e.1 = e.2)
    (h : ∃ u v, findColl (D.calls x ws) = some (u, v) ∧ Collision H u v) :
    ∃ u v, findColl ((D.withSetup H).calls x ws) = some (u, v) ∧ Collision H u v := by
  obtain ⟨u, v, huv, -⟩ := h
  exact VStar.coll_of_sub (withSetup_true ht) (fun p hp => List.mem_append_left _ hp) (by simp [huv])

theorem callsLogColl_withSetup (hT : CallsTrue H D I DO) (hL : CallsLogColl H D I DO) :
    CallsLogColl H (D.withSetup H) I DO := by
  intro decO x Ω P ω r c ws hg
  rcases hL decO x Ω P ω r c ws hg with h | h
  · exact Or.inl fun e he => List.mem_append_left _ (h e he)
  · exact Or.inr (coll_withSetup (hT decO x Ω P ω r c ws hg) h)

theorem innerAcceptsCollAt_withSetup {x : Msg} (hT : CallsTrue H D I DO) (hA : InnerAcceptsCollAt H D I DO x) :
    InnerAcceptsCollAt H (D.withSetup H) I DO x := by
  intro decO Ω P ω r c ws hg
  rcases hA decO Ω P ω r c ws hg with h | h
  · exact Or.inl h
  · exact Or.inr (coll_withSetup (hT decO x Ω P ω r c ws hg) h)

end Transfer

section Coupling

variable {Coin Col T Col' T' : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
  [DecidableEq Col'] {Bd : Bound Coin Col T} {decI : List (ℕ × Col) → List T} {H : Msg → Msg} {D : Deploy Coin}
  {DO : Flock.Deploy Coin Col' T'} {decO : List (ℕ × Col') → List T'} {Ω : Type} {P : Prover Ω} {x : Msg}

/-- **The recursion's KS at the bound record with the inner setup's calls searched**: `Bound.ksLin2_of_flockLin_binds`
at `D.withSetup H`. Its facts are `D`'s, but `CallsLogColl` and `ForksBindAt`, which it reads there: an accepted inner
run's calls are among the recorded ones and the setup's, or those collide. -/
theorem Bound.ksLin2_of_flockLin_binds_setup [Zero Col] [Zero Col'] (decC : Msg → Msg → Coin) (Cp : Caps Coin)
    (hF : Flock.KS H DO) (hKS : KSFlockLin H Bd.DE Cp) (hdI : decOK Bd.deploy decI) (hdO : decOK DO decO)
    (hdec : ∀ x c, decC x (Bd.ans c) = c) (hB : Bd.Binds) (hRef : Refuses H D)
    (hRounds : InnerRounds D (Inner.ofFlock Bd.deploy decI)) (hN : SessionNums D DO) (hε : TableErrNonneg D)
    (hRd : SessionRounds D (Inner.ofFlock Bd.deploy decI) DO) (hS : SessionAccepts D)
    (hRs : SessionRuns H D (Inner.ofFlock Bd.deploy decI) DO) (hV : InnerViewRun H D (Inner.ofFlock Bd.deploy decI) DO)
    (hT : CallsTrue H D (Inner.ofFlock Bd.deploy decI) DO)
    (hL : CallsLogColl H (D.withSetup H) (Inner.ofFlock Bd.deploy decI) DO)
    (hFk : ForksBindAt H (D.withSetup H) (Inner.ofFlock Bd.deploy decI) DO x)
    (hA : InnerAcceptsCollAt H D (Inner.ofFlock Bd.deploy decI) DO x)
    (hM : RoundsMargin H D (Inner.ofFlock Bd.deploy decI)) (hN0 : CallsNil H D) [Fintype Ω] [Nonempty Ω] (M R : ℕ) :
    PrBad H D.V (Recursion.R D (Inner.ofFlock Bd.deploy decI)) (E H D (Inner.ofFlock Bd.deploy decI) DO decO) x P ≤
      κ D (Inner.ofFlock Bd.deploy decI) x +
        Real.sqrt (γ D (Inner.ofFlock Bd.deploy decI) x * PrB H D.V P x) +
        (4 * (Inner.ofFlock Bd.deploy decI).V.k (xI D x) + 1) *
          PrBinner H (D.withSetup H) (Inner.ofFlock Bd.deploy decI) DO decO P x +
        4 * Cp.width (xI D x) *
          (PrFCI H (D.withSetup H) (Inner.ofFlock Bd.deploy decI) DO decO Cp P x M + 1 / ((M : ℝ) + 1)) +
        4 * (PrF0I H (D.withSetup H) (Inner.ofFlock Bd.deploy decI) DO decO P x R +
          (((Inner.ofFlock Bd.deploy decI).k (xI D x) : ℝ) + 1) / ((R : ℝ) + 1)) +
        βdev2 D (Inner.ofFlock Bd.deploy decI) Cp x *
          (PrFdev H (D.withSetup H) (Inner.ofFlock Bd.deploy decI) DO decO P x M + 1 / ((M : ℝ) + 1)) :=
  Bound.ksLin2_of_flockLin_binds (D := D.withSetup H) (decO := decO) (P := P) (x := x) decC Cp hF hKS hdI hdO hdec hB
    hRef hRounds hN hε hRd hS hRs hV (callsTrue_withSetup hT) hL hFk (innerAcceptsCollAt_withSetup hT hA) hM
    (callsNil_withSetup hN0) M R

end Coupling

end Claims.Recursion
