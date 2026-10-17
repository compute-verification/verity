import Proofs.Flock.Claims.KS.TableBound
import Proofs.Flock.Claims.KS.Refit

/-!
# Escapes on accepted runs

Escapes whose events hold only on accepted runs (`Escapes.OnAcc`): each fork event only where both forked runs accept,
each one-run event only where its run accepts. `TableBoundθAccAt` is `TableBoundθAt` with such escapes, so it gives
`TableBoundθAt` (`TableBoundθAccAt.tableBoundθAt`), and where the verifier accepts no run it holds with no escapes
(`tableBoundθAccAt_of_noAcc`). The old plain and masked escapes are on accepted runs when their events are
(`oldEsc_onAcc`, `oldEscZ_onAcc`). Any escapes restrict to accepted runs (`Escapes.onAccOf`), still tied to collisions
and fitting the same weight (`Escapes.onAccOf_onAcc`, `Escapes.fitsFork_onAccOf`), with a smaller term
(`Escapes.term_onAccOf_le`).
-/

namespace Claims

/-- **Escapes on accepted runs**: each fork event holds only where both forked runs accept, and each one-run event only
where its run accepts. -/
def Escapes.OnAcc {Ω Coin : Type} [Inhabited Coin] {H : Msg → Msg} {V : Verifier Coin} {P : Prover Ω} {x : Msg}
    (e : Escapes H V P x) : Prop :=
  (∀ i ω r r', e.ev i ω r r' →
      (run H P V x ω r).out = .acc ∧ (run H P V x ω (forkCoins (e.j i) r r')).out = .acc) ∧
    ∀ l ω r, e.ev₁ l ω r → (run H P V x ω r).out = .acc

/-- **Escapes restricted to accepted runs**: each event and the acceptance of its runs. -/
def Escapes.onAccOf {Ω Coin : Type} [Inhabited Coin] {H : Msg → Msg} {V : Verifier Coin} {P : Prover Ω} {x : Msg}
    (e : Escapes H V P x) : Escapes H V P x where
  I := e.I
  c := e.c
  j := e.j
  ev i ω r r' := e.ev i ω r r' ∧ (run H P V x ω r).out = .acc ∧ (run H P V x ω (forkCoins (e.j i) r r')).out = .acc
  L := e.L
  β := e.β
  ev₁ l ω r := e.ev₁ l ω r ∧ (run H P V x ω r).out = .acc
  c_nonneg := e.c_nonneg
  β_nonneg := e.β_nonneg
  tied i ω r r' h := e.tied i ω r r' h.1
  tied₁ l ω r h := e.tied₁ l ω r h.1

theorem Escapes.onAccOf_onAcc {Ω Coin : Type} [Inhabited Coin] {H : Msg → Msg} {V : Verifier Coin} {P : Prover Ω}
    {x : Msg} (e : Escapes H V P x) : e.onAccOf.OnAcc :=
  ⟨fun _ _ _ _ h => h.2, fun _ _ _ h => h.2⟩

theorem Escapes.fitsFork_onAccOf {Ω Coin : Type} [Inhabited Coin] {H : Msg → Msg} {V : Verifier Coin} {P : Prover Ω}
    {x : Msg} (e : Escapes H V P x) {C c : ℝ} (h : e.FitsFork C c) : e.onAccOf.FitsFork C c :=
  h

theorem Escapes.term_onAccOf_le {Ω Coin : Type} [Fintype Coin] [Inhabited Coin] {H : Msg → Msg} {V : Verifier Coin}
    {P : Prover Ω} {x : Msg} (e : Escapes H V P x) (ω : Ω) : e.onAccOf.term ω ≤ e.term ω := by
  unfold Escapes.term
  refine add_le_add (Finset.sum_le_sum fun i _ => Real.sqrt_le_sqrt
    (mul_le_mul_of_nonneg_left (Pr_mono fun _ h => h.1) (e.c_nonneg i))) (Finset.sum_le_sum fun l _ =>
      mul_le_mul_of_nonneg_left (Pr_mono fun _ h => h.1) (e.β_nonneg l))

end Claims

namespace Claims.Flock

theorem Escapes.onAcc_nil {Ω Coin : Type} [Inhabited Coin] (H : Msg → Msg) (V : Verifier Coin) (P : Prover Ω)
    (x : Msg) : (Escapes.nil H V P x).OnAcc :=
  ⟨fun i => i.elim, fun l => l.elim⟩

/-- **`TableBoundθ` at one statement `x`, with escapes on accepted runs.** -/
def TableBoundθAccAt {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]
    (H : Msg → Msg) (D : Deploy Coin Col T) (N : Msg → ℕ) (Ok : Msg → Col → Prop) (x : Msg) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω), ∃ e : Escapes H D.V P x, e.OnAcc ∧
    e.FitsFork (D.weight x) (D.k x) ∧ ∀ (ω : Ω) (C : Fin (N x) → Col), (∀ p, Ok x (C p)) →
      ¬ GoodTable D N x C →
      accPr H D.V P x ω ≤ D.εtable x + e.term ω +
        Pr fun r : Fin (D.V.k x) → Coin => (run H P D.V x ω r).out = .acc ∧
          ∃ (p : Fin (N x)) (u : Col), lookCol (D.cols (run H P D.V x ω r).seen 0) p.val = some u ∧ u ≠ C p

theorem TableBoundθAccAt.tableBoundθAt {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    [DecidableEq Col] [Zero Col] {H : Msg → Msg} {D : Deploy Coin Col T} {N : Msg → ℕ} {Ok : Msg → Col → Prop}
    {x : Msg} (h : TableBoundθAccAt H D N Ok x) : TableBoundθAt H D N Ok x := by
  intro Ω _ _ P
  obtain ⟨e, -, hf, hb⟩ := h Ω P
  exact ⟨e, hf, hb⟩

/-- **Where the verifier accepts no run**, `TableBoundθAccAt` holds with no escapes, when the weight covers `k` and
`ε_table ≥ 0`. -/
theorem tableBoundθAccAt_of_noAcc {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    [DecidableEq Col] [Zero Col] (H : Msg → Msg) (D : Deploy Coin Col T) (N : Msg → ℕ) (Ok : Msg → Col → Prop)
    (x : Msg) (hno : NoAcc H D.V x) (hk : (D.k x : ℝ) ≤ 2 * D.weight x) (hε : 0 ≤ D.εtable x) :
    TableBoundθAccAt H D N Ok x := by
  intro Ω _ _ P
  refine ⟨Escapes.nil H D.V P x, Escapes.onAcc_nil H D.V P x, Escapes.fitsFork_nil H D.V P x _ _ hk,
    fun ω C _ _ => ?_⟩
  have h0 : accPr H D.V P x ω = 0 := by
    unfold accPr
    rw [Pr_congr (q := fun _ => False) fun r => ⟨fun h => hno Ω P ω r h, False.elim⟩, Pr_false]
  rw [h0, Escapes.term_nil]
  have := Pr_nonneg fun r : Fin (D.V.k x) → Coin => (run H P D.V x ω r).out = .acc ∧
    ∃ (p : Fin (N x)) (u : Col), lookCol (D.cols (run H P D.V x ω r).seen 0) p.val = some u ∧ u ≠ C p
  linarith

/-- **The old plain escapes are on accepted runs** where their events are. -/
theorem oldEsc_onAcc {Ω Coin : Type} [Inhabited Coin] (H : Msg → Msg) (V : Verifier Coin) (P : Prover Ω) (x : Msg)
    (sch : FlockSoundness.Accounting.Schedule) (j : Fin 2 → Fin sch.levels.length → Fin (V.k x))
    (ev : Fin 2 → Fin sch.levels.length → Ω → (Fin (V.k x) → Coin) → (Fin (V.k x) → Coin) → Prop)
    (ev₁ : Ω → (Fin (V.k x) → Coin) → Prop)
    (tied : ∀ r ℓ ω s s', ev r ℓ ω s s' →
      TiedColl H ((run H P V x ω s).vLog ++ (run H P V x ω (forkCoins (j r ℓ) s s')).vLog))
    (tied₁ : ∀ ω s, ev₁ ω s → TiedColl H (run H P V x ω s).vLog)
    (hacc : ∀ r ℓ ω s s', ev r ℓ ω s s' →
      (run H P V x ω s).out = .acc ∧ (run H P V x ω (forkCoins (j r ℓ) s s')).out = .acc)
    (hacc₁ : ∀ ω s, ev₁ ω s → (run H P V x ω s).out = .acc) :
    (oldEsc H V P x sch j ev ev₁ tied tied₁).OnAcc :=
  ⟨fun i ω s s' h => hacc i.1 i.2 ω s s' h, fun _ ω s h => hacc₁ ω s h⟩

/-- **The old masked escapes are on accepted runs** where their events are. -/
theorem oldEscZ_onAcc {Ω Coin : Type} [Inhabited Coin] (H : Msg → Msg) (V : Verifier Coin) (P : Prover Ω)
    (x : Msg) (sch : FlockSoundness.Accounting.Schedule) (Np : ℕ) (j : IZ sch.levels.length → Fin (V.k x))
    (ev : IZ sch.levels.length → Ω → (Fin (V.k x) → Coin) → (Fin (V.k x) → Coin) → Prop)
    (ev₁ : Ω → (Fin (V.k x) → Coin) → Prop)
    (tied : ∀ i ω s s', ev i ω s s' →
      TiedColl H ((run H P V x ω s).vLog ++ (run H P V x ω (forkCoins (j i) s s')).vLog))
    (tied₁ : ∀ ω s, ev₁ ω s → TiedColl H (run H P V x ω s).vLog)
    (hacc : ∀ i ω s s', ev i ω s s' →
      (run H P V x ω s).out = .acc ∧ (run H P V x ω (forkCoins (j i) s s')).out = .acc)
    (hacc₁ : ∀ ω s, ev₁ ω s → (run H P V x ω s).out = .acc) :
    (oldEscZ H V P x sch Np j ev ev₁ tied tied₁).OnAcc :=
  ⟨hacc, fun _ ω s h => hacc₁ ω s h⟩

end Claims.Flock
