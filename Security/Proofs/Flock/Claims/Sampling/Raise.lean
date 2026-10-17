import Proofs.Flock.Claims.KS.TableBound

/-!
# Flock's KS at a raised `N₀`

The sampled extractor runs a session's `E^rep` at the inner `k′ = ⌈4·N₀′/e⌉` (`Sampling.Deploy.innerK`). Where `N₀′`
only bounds the session's `N₀`, that is Flock's `E^rep` at the record with `N₀` raised (`Flock.Deploy.raise`). KS's
proof reads `N₀` monotonically but in one place: one wait's bound needs the word no longer than `N₀`, `k ≥ 1` and
`N₀/(e·k) ≤ 1/4` (`oneWait_of_tableBound`, `rate_le`), KS needs `k ≥ 8` (`KS_of_oneWait`), and `κ` at `k′` is the
sampled `κ_F`. The exception is the weight: `TableBoundθ` asks the escapes to fit the weight `C` with the conflict
fork's event of weight `k` (`Escapes.FitsFork C k`), at a multiplicity it leaves open, so at `k′ > k` the same weight
need not hold it. Scaled with `k`, it does (`Escapes.FitsFork.raise`, `tableBoundθAt_raise`):

* `FlockDeploy.KS.zk_KS_raise`: KS at `deployZk` raised to `N₀′ ≥ N₀` and a weight `C′` with `C·k′ ≤ C′·k` and
  `k′ ≤ 2C′`, from `zk_KS_of_acc`'s premises.
-/

namespace Claims.Flock

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

/-- **The record raised**: `N₀` raised to `N₀′`, and the weight `C′`. -/
def Deploy.raise (D : Deploy Coin Col T) (N₀' C' : Msg → ℝ) : Deploy Coin Col T :=
  { D with N₀ := N₀', weight := C' }

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col] in
theorem Deploy.k_le_raise (D : Deploy Coin Col T) (N₀' C' : Msg → ℝ) (x : Msg) (h : D.N₀ x ≤ N₀' x) :
    D.k x ≤ (D.raise N₀' C').k x := by
  unfold Deploy.k Deploy.raise
  exact Nat.ceil_mono (div_le_div_of_nonneg_right (by linarith) (Real.exp_pos 1).le)

omit [Fintype Coin] [Nonempty Coin] [DecidableEq Col] [Zero Col] in
/-- **Escapes that fit `C` with a fork event of weight `c > 0` fit `C′` with one of weight `c′ ≥ c`**, when
`C·c′ ≤ C′·c`. -/
theorem _root_.Claims.Escapes.FitsFork.raise {Ω : Type} {H : Msg → Msg} {V : Verifier Coin} {P : Prover Ω} {x : Msg}
    {e : Escapes H V P x} {C C' c c' : ℝ} (h : e.FitsFork C c) (hc : 0 < c) (hcc : c ≤ c') (hC : C * c' ≤ C' * c) :
    e.FitsFork C' c' := by
  obtain ⟨mult, hm, hmc, hβ⟩ := h
  refine ⟨mult, hm, ?_, hβ⟩
  have hs : 0 ≤ (mult : ℝ) * ∑ i, e.c i := mul_nonneg (Nat.cast_nonneg _) (Finset.sum_nonneg fun i _ => e.c_nonneg i)
  have h1 : c * (mult * ∑ i, e.c i) ≤ c' * (mult * ∑ i, e.c i) := mul_le_mul_of_nonneg_right hcc hs
  have h2 : c' * (mult * (c + ∑ i, e.c i)) ≤ c' * (2 * C) := mul_le_mul_of_nonneg_left hmc (hc.le.trans hcc)
  refine le_of_mul_le_mul_left ?_ hc
  nlinarith

/-- **`TableBoundθ` at a statement, raised**: the same escapes, refit to the raised `k′` at a weight `C′` with
`C·k′ ≤ C′·k`, where `k > 0`. -/
theorem tableBoundθAt_raise (H : Msg → Msg) (D : Deploy Coin Col T) (N : Msg → ℕ) (Ok : Msg → Col → Prop)
    (N₀' C' : Msg → ℝ) (x : Msg) (h : TableBoundθAt H D N Ok x) (hk : 0 < D.k x) (hN : D.N₀ x ≤ N₀' x)
    (hC : D.weight x * (D.raise N₀' C').k x ≤ C' x * D.k x) : TableBoundθAt H (D.raise N₀' C') N Ok x := by
  intro Ω _ _ P
  obtain ⟨e, hfit, hb⟩ := h Ω P
  exact ⟨e, hfit.raise (by exact_mod_cast hk) (by exact_mod_cast D.k_le_raise N₀' C' x hN) hC, hb⟩

end Claims.Flock

namespace FlockDeploy.KS

open Flock

/-- **The deployed masked (`--zk`) verifier: KS at a raised `N₀`**, at SHA-512: at `N₀′ ≥ N₀` and a weight `C′` that
scales the deployed one with `k` (`C·k′ ≤ C′·k`) and covers `k′` (`k′ ≤ 2C′`), from the premises of KS at the record
itself (`zk_KS_of_acc`). -/
theorem zk_KS_raise (rel : Readable.Parsed → Readable.Relation)
    (hone : ∀ x, (∃ s, (deployZk rel).parse x = some s ∧ s.tables = 1) ∨
      Claims.Flock.NoAcc sha512 (deployZk rel).V x)
    (h : ∀ x s ins spec sts, ¬ Claims.Flock.NoAcc sha512 (V true) x → parse true rel x = some s →
      Claims.Flock.InScope true s → setupOf true x = some (ins, spec, sts) →
      Claims.Flock.TableBoundθAt sha512 (deployZk rel) (wordLen true) (Claims.Flock.orZero (wide true)) x)
    (N₀' C' : Claims.Msg → ℝ) (hN : ∀ x, N₀ true x ≤ N₀' x)
    (hC : ∀ x, weight true x * ((deployZk rel).raise N₀' C').k x ≤ C' x * (deployZk rel).k x ∧
      (((deployZk rel).raise N₀' C').k x : ℝ) ≤ 2 * C' x) :
    Claims.Flock.KS sha512 ((deployZk rel).raise N₀' C') := by
  have hkle : ∀ x, (deployZk rel).k x ≤ ((deployZk rel).raise N₀' C').k x :=
    fun x => (deployZk rel).k_le_raise N₀' C' x (hN x)
  have htab : Claims.Flock.TableBoundθ sha512 ((deployZk rel).raise N₀' C') (wordLen true)
      (Claims.Flock.orZero (wide true)) := by
    refine (Claims.Flock.tableBoundθ_iff _ _ _ _).2 fun x => ?_
    rcases k_range_zk rel x with hk | hno
    · exact Claims.Flock.tableBoundθAt_raise _ _ _ _ N₀' C' x (zk_tableBoundθ_of_acc rel _ h x) (by omega) (hN x)
        (hC x).1
    · exact Claims.Flock.tableBoundθAt_of_noAcc _ _ _ _ x hno (hC x).2 (εtable_nonneg true x)
  have hsp : ∀ x, Claims.Flock.Splits ((deployZk rel).raise N₀' C') x ∨ Claims.Flock.NoAcc sha512 (V true) x :=
    fun x => (hone x).imp_left fun ⟨_, hs, h1⟩ => splits_of_one ((deployZk rel).raise N₀' C') rfl rfl hs h1
  exact Claims.Flock.KS_of_oneWait sha512 _ (fun _ => cap_pos)
    (fun x => (k_range_zk rel x).elim (fun hk => (hsp x).imp (fun h => ⟨hk.trans (hkle x), h⟩) id) .inr)
    (fun x => εtable_nonneg true x)
    (Claims.Flock.oneWait_of_tableBound sha512 _ (wordLen true) (fun x => (wordLen_le_N₀ true x).trans (hN x))
      (fun x => εtable_nonneg true x) (V_k_pos true)
      (fun x => (k_range_zk rel x).imp (fun hk => by have := hkle x; omega) id) hone _ (fun _ => .inl rfl)
      (Claims.Flock.colsOk_orZero _ _ _ _ (zk_colsOk sha512 rel)) (zk_conflictTied rel) htab)

end FlockDeploy.KS
