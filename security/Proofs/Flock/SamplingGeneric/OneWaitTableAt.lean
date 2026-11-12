module

public import Proofs.Flock.Claims.KS.Finds
public import Proofs.Flock.Claims.KS.TableAt
public import Proofs.Flock.Claims.KS.ZkTableAt
public import Proofs.Flock.SamplingGeneric.MaskedTableAt
public import Proofs.Flock.SamplingGeneric.SessionFinds
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Algebra.Order.Star.Real
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Tactic.Polynomial.Basic
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Init
import Std.Tactic.BVDecide.Normalize.Prop
import Proofs.Flock.Claims.Bits
import Proofs.Flock.Claims.Bridge.Coins
import Proofs.Flock.Claims.Bridge.Transport
import Proofs.Flock.Claims.DeployCode
import Proofs.Flock.Claims.DeployTables
import Proofs.Flock.Claims.ERepCost
import Proofs.Flock.Claims.KS.Deploy
import Proofs.Flock.Claims.KS.FitsR
import Proofs.Flock.Claims.KS.Flock
import Proofs.Flock.Claims.KS.Fork
import Proofs.Flock.Claims.KS.Joint
import Proofs.Flock.Claims.KS.JointJ
import Proofs.Flock.Claims.KS.JointR
import Proofs.Flock.Claims.KS.Range
import Proofs.Flock.Claims.KS.TableBound
import Proofs.Flock.Claims.KS.Wait
import Proofs.Flock.Claims.KS.ZkClashPads
import Proofs.Flock.Claims.KS.ZkExtend
import Proofs.Flock.Claims.KS.ZkLiveW
import Proofs.Flock.Claims.KS.ZkPadsSlot
import Proofs.Flock.Claims.KS.ZkXferW
import Proofs.Flock.Claims.Prob

/-!
# Flock at one table of a session: what the extractor finds, its one wait finds at half the rate

`Flock.EAt₁` is Flock's extractor at table `j` with one wait in place of its `m′`: the same `k_j = ⌈4·N_j/e⌉` accepted
reruns a wait, the same truncation, decoder and check. It makes at most `1 + k_j` accepted runs' worth of reruns where
`Flock.EAt` makes `1 + m′·k_j`.

* **At one statement and prover** (`Flock.finds_at_of_oneWait`): from one wait's bound at table `j`, the bound
  `Flock.ks_at_of_oneWait` reads, `Flock.EAt` returns a witness with a property `G` of the run's coins, on an accepted
  run, with probability at most twice `Flock.EAt₁`'s, plus `4·ε_j + 2k_j/K + 2·e^{−k_j/8} + √(γ·Pr[B])`: `ERep_finds`
  at that bound on the event that `assembleAt` and `checkAt` succeed (`Flock.wait_at_assembles`).
* **At each table** (`Flock.OneWaitFindsAt`, `Flock.oneWaitFindsAt_of_tableBoundAtR`): `Claims.Finds` at `EAt` and
  `EAt₁`, at the table's knowledge error `κ_j` and the session's `γ`, under `Flock.KSAt_of_tableBoundAtR`'s
  hypotheses.
* **At the deployed masked record** (`FlockDeploy.KS.zk_oneWaitFindsAt_untrim`): with no premise, at SHA-512, at its
  tables' words' lengths and each table's own error.
* **At the compiler** (`oneWaitFindsAt`, `oneWaitFindsAt_masked`, `sessionFindsAt_masked`): `Arg.ofFlockAt` meets
  `Arg.OneWaitFinds`, so a compiled verifier over it finds at its one-wait form what it finds, at half the rate
  (`Compiled.sessionFinds_of`); and the one-wait proof system's reruns at a statement are `1 + k_j`, `j` the table
  the statement names (`oneWait_rerunsAt`).
-/

@[expose] public section

namespace Claims.Flock

open FlockSoundness FlockSoundness.Game FlockSoundness.Extract FlockSoundness.Rewinding

section OneWaitAt

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

/-- **Flock's extractor at table `j` with one wait**: `E^rep` at a decoder with one wait for `k_j` accepted reruns
among `K` samples, where `Flock.EAt` makes `m′`. -/
noncomputable def EAt₁ (D : Deploy Coin Col T) (N : Msg → ℕ → ℕ) (decode : List (ℕ × Col) → List T) (j : ℕ) :
    Extractor Coin Wit D.V :=
  ERep D.V (fun _ => 1) (kAt N j) D.cap (fun v => [D.cols v]) (assembleAt D.table decode D.read j) (checkAt D.table j)

/-- **Flock at each table of a session: what the extractor finds, its one wait finds at half the rate**, at every
decoder that meets the list-decoding spec: `Claims.Finds` at `EAt` and `EAt₁` at every table `j`, at the table's
knowledge error `κ_j` and the session's `γ`. -/
def OneWaitFindsAt [Zero Col] (H : Msg → Msg) (D : Deploy Coin Col T) (N : Msg → ℕ → ℕ) (ε : Msg → ℕ → ℝ) : Prop :=
  ∀ decode, ListWithin D.len D.enc D.θ decode → ∀ j,
    Finds H D.V (EAt D N decode j) (EAt₁ D N decode j) (κAt D N ε j) (γ D)

variable [Zero Col]

/-- **One wait's bound at table `j`, on the event the extractor reads**: a bound on one wait of `k_j` accepted runs
that fails to extract table `j` bounds one that fails to assemble a witness `checkAt` accepts, as extraction at the
table assembles one (`assemblesAt_of_extractsAt`). -/
theorem wait_at_assembles (H : Msg → Msg) (D : Deploy Coin Col T) (N : Msg → ℕ → ℕ)
    (decode : List (ℕ × Col) → List T) (hlw : ListWithin D.len D.enc D.θ decode) (j : ℕ) (x : Msg)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (εt : ℝ) (e : Escapes H D.V P x)
    (hw : ∀ ω, 0 < accPr H D.V P x ω →
      accPr H D.V P x ω * (accFail H D.V P x ω (kAt N j x) (fun vs => ExtractsAt D x vs j) /
          accPr H D.V P x ω ^ kAt N j x) ≤
        εt + e.term ω + (N x j : ℝ) / (Real.exp 1 * kAt N j x) * accPr H D.V P x ω) :
    ∀ ω, 0 < accPr H D.V P x ω →
      accPr H D.V P x ω * (accFail H D.V P x ω (kAt N j x)
        (Assembles (fun v => [D.cols v]) (assembleAt D.table decode D.read j x) (checkAt D.table j x)) /
          accPr H D.V P x ω ^ kAt N j x) ≤
        (εt + e.term ω) + (N x j : ℝ) / (Real.exp 1 * kAt N j x) * accPr H D.V P x ω := by
  intro ω hε
  refine le_trans ?_ ((hw ω hε).trans (le_of_eq (by ring)))
  refine mul_le_mul_of_nonneg_left (div_le_div_of_nonneg_right ?_ (pow_nonneg hε.le _)) hε.le
  unfold accFail
  exact Pr_mono fun σ ⟨hacc, hno⟩ => ⟨hacc, fun hex => hno (assemblesAt_of_extractsAt D decode hlw x j _ hex)⟩

/-- **Flock at table `j`, at one statement and prover: what the extractor finds, its one wait finds at half the
rate**, from one wait's bound there, under `ks_at_of_oneWait`'s hypotheses: for every property `G` of the judged run's
coins and a witness, the run accepts and `EAt` returns a witness with `G` with probability at most twice that at
`EAt₁`, plus `4·ε_j + 2k_j/K + 2·e^{−k_j/8} + √(γ·Pr[B])`. -/
theorem finds_at_of_oneWait (H : Msg → Msg) (D : Deploy Coin Col T) (N : Msg → ℕ → ℕ)
    (decode : List (ℕ × Col) → List T) (hlw : ListWithin D.len D.enc D.θ decode) (j : ℕ) (x : Msg)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (hK : 0 < D.cap x) (hk8 : 8 ≤ kAt N j x)
    (εt : ℝ) (hεt : 0 ≤ εt) (e : Escapes H D.V P x) (hfit : e.Fits (D.weight x))
    (hw : ∀ ω, 0 < accPr H D.V P x ω →
      accPr H D.V P x ω * (accFail H D.V P x ω (kAt N j x) (fun vs => ExtractsAt D x vs j) /
          accPr H D.V P x ω ^ kAt N j x) ≤
        εt + e.term ω + (N x j : ℝ) / (Real.exp 1 * kAt N j x) * accPr H D.V P x ω)
    (G : (Fin (D.V.k x) → Coin) → Wit → Prop) :
    Pr (fun s : Ω × (Fin (D.V.k x) → Coin) × (EAt D N decode j).C x =>
        (run H P D.V x s.1 s.2.1).out = .acc ∧
          ∃ w, ((EAt D N decode j).eval H P x s.1 s.2.1 s.2.2).1 = some w ∧ G s.2.1 w) ≤
      2 * Pr (fun s : Ω × (Fin (D.V.k x) → Coin) × (EAt₁ D N decode j).C x =>
        (run H P D.V x s.1 s.2.1).out = .acc ∧
          ∃ w, ((EAt₁ D N decode j).eval H P x s.1 s.2.1 s.2.2).1 = some w ∧ G s.2.1 w) +
        (4 * εt + 2 * kAt N j x / D.cap x + 2 * Real.exp (-(kAt N j x : ℝ) / 8)) +
        Real.sqrt (γ D x * PrB H D.V P x) := by
  have hb : Pr (fun s : Ω × (Fin (D.V.k x) → Coin) × (EAt D N decode j).C x =>
        (run H P D.V x s.1 s.2.1).out = .acc ∧
          ∃ w, ((EAt D N decode j).eval H P x s.1 s.2.1 s.2.2).1 = some w ∧ G s.2.1 w) ≤
      2 * Pr (fun s : Ω × (Fin (D.V.k x) → Coin) × (EAt₁ D N decode j).C x =>
        (run H P D.V x s.1 s.2.1).out = .acc ∧
          ∃ w, ((EAt₁ D N decode j).eval H P x s.1 s.2.1 s.2.2).1 = some w ∧ G s.2.1 w) +
        (4 * Ex (fun ω => εt + e.term ω) + 2 * kAt N j x / D.cap x + 2 * Real.exp (-(kAt N j x : ℝ) / 8)) :=
    ERep_finds H D.V D.waits (kAt N j) D.cap (fun v => [D.cols v]) (assembleAt D.table decode D.read j)
      (checkAt D.table j) x P hK hk8 (fun ω => εt + e.term ω)
      (fun ω => add_nonneg hεt (e.term_nonneg ω)) (rateAt_le N j x (by omega))
      (wait_at_assembles H D N decode hlw j x P εt e hw) G
  have hEx : 4 * Ex (fun ω => εt + e.term ω) ≤ 4 * εt + Real.sqrt (γ D x * PrB H D.V P x) := by
    rw [Ex_add, Ex_const]
    have := e.le (D.weight x) hfit
    unfold γ
    linarith
  linarith

/-- **Flock at each table of a session: what the extractor finds, its one wait finds at half the rate, from each
table's bound at its error** (`TableBoundAtR`), under `KSAt_of_tableBoundAtR`'s hypotheses: at a table of the
statement with `k_j ≥ 8`, `finds_at_of_oneWait` at the escapes `KSAt_of_tableBoundAtR` builds; at a table with an
empty word the error is at least `1`; and where the verifier accepts no run the event is empty. -/
theorem oneWaitFindsAt_of_tableBoundAtR (H : Msg → Msg) (D : Deploy Coin Col T) (N : Msg → ℕ → ℕ)
    (ε : Msg → ℕ → ℝ) (hN : ∀ x j, j < nTab D x → (N x j : ℝ) ≤ D.N₀ x) (hεt : ∀ x j, 0 ≤ ε x j)
    (hVk : ∀ x, 0 < D.V.k x) (hK : ∀ x, 0 < D.cap x)
    (hk : ∀ x j, (j < nTab D x ∧ 8 ≤ kAt N j x) ∨ kAt N j x = 0 ∨ NoAcc H D.V x)
    (Ok : Msg → ℕ → Col → Prop) (hz : ∀ x j, Ok x j 0) (hok : ColsOkJ H D N Ok) (hconf : ConflictTiedJ H D N)
    (htab : TableBoundAtR H D N Ok ε) :
    OneWaitFindsAt H D N ε := by
  intro decode hlw j x Ω _ _ P G
  have hκ0 := (κAt_pos D N ε j x (hεt x j)).le
  have hs0 := Real.sqrt_nonneg (γ D x * PrB H D.V P x)
  have h1 := Pr_nonneg fun s : Ω × (Fin (D.V.k x) → Coin) × (EAt₁ D N decode j).C x =>
    (run H P D.V x s.1 s.2.1).out = .acc ∧
      ∃ w, ((EAt₁ D N decode j).eval H P x s.1 s.2.1 s.2.2).1 = some w ∧ G s.2.1 w
  rcases hk x j with ⟨hj, hk8⟩ | hk0 | hno
  · obtain ⟨e₀, hfit, hb⟩ := htab x Ω P
    have hkle : kAt N j x ≤ D.k x :=
      Nat.ceil_mono (div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left (hN x j hj) (by norm_num))
        (Real.exp_pos 1).le)
    have hfit' : e₀.FitsForkR (D.weight x) (kAt N j x) := hfit.mono (Nat.cast_le.2 hkle)
    have hκ : 4 * ε x j + 2 * kAt N j x / D.cap x + 2 * Real.exp (-(kAt N j x : ℝ) / 8) ≤
        2 * κAt D N ε j x := by
      unfold κAt
      have h2 : (0 : ℝ) < 2 ^ (-(D.waits x : ℝ)) := Real.rpow_pos_of_pos (by norm_num) _
      have h3 : (0 : ℝ) ≤ 2 * kAt N j x / D.cap x := by positivity
      have h4 := hεt x j
      linarith
    refine le_trans (finds_at_of_oneWait H D N decode hlw j x P (hK x) hk8 (ε x j) (hεt x j)
      (e₀.addFork (kAt N j x) (Nat.cast_nonneg _) ⟨0, hVk x⟩
        (fun ω r r' => ∃ p, ConflictAt (opAt H D N P x j ω) (r, r') p)
        (opAt_tied H D N hconf P x j hj ⟨0, hVk x⟩ rfl)).mergeR
      (e₀.fitsR_addFork (D.weight x) (kAt N j x) (Nat.cast_nonneg _) ⟨0, hVk x⟩ rfl
        (fun ω r r' => ∃ p, ConflictAt (opAt H D N P x j ω) (r, r') p)
        (opAt_tied H D N hconf P x j hj ⟨0, hVk x⟩ rfl) hfit').fits_mergeR (fun ω hε => ?_) G) (by linarith)
    have hm := (e₀.addFork (kAt N j x) (Nat.cast_nonneg _) ⟨0, hVk x⟩
      (fun ω r r' => ∃ p, ConflictAt (opAt H D N P x j ω) (r, r') p)
      (opAt_tied H D N hconf P x j hj ⟨0, hVk x⟩ rfl)).term_le_mergeR ω
    have := oneWait_le_at H D N Ok hz hok hconf x j hj (hVk x) (ε x j) (hεt x j) (by omega) P e₀
      (fun ω C hC hg => hb ω j hj C hC hg) ω hε
    linarith
  · have h5 := one_le_κAt D N ε j x (hεt x j) hk0
    refine (Pr_le_one _).trans ?_
    linarith
  · rw [Pr_congr (q := fun _ => False) fun s => ⟨fun hs => hno Ω P s.1 s.2.1 hs.1, False.elim⟩, Pr_false]
    linarith

end OneWaitAt

end Claims.Flock

namespace FlockDeploy.KS

open Flock

namespace Gen

variable [S : ZkSetup]

/-- **At the masked record, at each table of a session, what the extractor finds its one wait finds at half the
rate**, at the untrimmed instance, at SHA-512 at the deployed relation, with no premise, at the tables' words' lengths
and each table's own error (`Claims.Flock.oneWaitFindsAt_of_tableBoundAtR` at `zk_tableBoundAtR_untrim`, as
`zk_KSAt_untrim`). -/
theorem zk_oneWaitFindsAt_untrim :
    Claims.Flock.OneWaitFindsAt sha512 (deployZk Readable.relation) (wordLenJ true) (tableErrAt true) :=
  Claims.Flock.oneWaitFindsAt_of_tableBoundAtR sha512 (deployZk Readable.relation) (wordLenJ true)
    (tableErrAt true) (fun x j _ => wordLenJ_le_N₀ true x j) (tableErrAt_nonneg true) (V_k_pos true)
    (fun _ => cap_pos) (kAt_range_zk Readable.relation) (Claims.Flock.orZeroJ (wideJ true)) (fun _ _ => .inl rfl)
    (Claims.Flock.colsOkJ_orZero _ _ _ _ (zk_colsOkJ sha512 _)) (zk_conflictTiedJ _) zk_tableBoundAtR_untrim

end Gen

/-- **At the masked record, at each table of a session, what the extractor finds its one wait finds at half the
rate**, at SHA-512 at the deployed relation, with no premise, at each table's own error
(`Gen.zk_oneWaitFindsAt_untrim`). -/
theorem zk_oneWaitFindsAt_untrim :
    Claims.Flock.OneWaitFindsAt sha512 (deployZk Readable.relation) (wordLenJ true) (tableErrAt true) :=
  Gen.zk_oneWaitFindsAt_untrim (S := S₀)

end FlockDeploy.KS

namespace Claims.SamplingGeneric

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

/-- **`Arg.ofFlockAt` finds at one wait what it finds, at half the rate, from `Flock.OneWaitFindsAt`**: at a statement
naming table `j` of the Flock statement `x`, `Flock.OneWaitFindsAt` at `j`, `x` and the deployed decoder, read at the
prover that sends this statement's messages (`run_atStmt`), as `flockArgKSAt` reads `Flock.KSAt`. -/
theorem oneWaitFindsAt (H : Msg → Msg) (D : Claims.Flock.Deploy Coin Col T) [Zero Col] (N : Msg → ℕ → ℕ)
    (ε : Msg → ℕ → ℝ) (names : Msg → ℕ × Msg) (hF : Claims.Flock.OneWaitFindsAt H D N ε)
    (hdec : Claims.Flock.ListWithin D.len D.enc D.θ D.decode) :
    (Arg.ofFlockAt D N ε names).OneWaitFinds H := by
  intro y Ω _ _ P G
  have hrun : run H (P.atStmt y) D.V (names y).2 = run H P (Arg.ofFlockAt D N ε names).V y :=
    funext fun ω => run_atStmt H P D.V (fun y => (names y).2) y ω
  have h := hF D.decode hdec (names y).1 (names y).2 Ω (P.atStmt y) G
  unfold PrB B Extractor.eval at h
  rw [hrun] at h
  exact h

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] in
/-- **The one-wait proof system's reruns at one table are `1 + k_j`**: at every statement, `Arg.ofFlockAt` with one
wait makes one wait, which stops at the named table's `k_j` accepted reruns (`Flock.kAt`), so the count `Compiled.es`
reads at a proof block of that shape, `1 + waits·k`, is `1 + k_j`. -/
theorem oneWait_rerunsAt (D : Claims.Flock.Deploy Coin Col T) (N : Msg → ℕ → ℕ) (ε : Msg → ℕ → ℝ)
    (names : Msg → ℕ × Msg) (y : Msg) :
    1 + (Arg.ofFlockAt D N ε names).oneWait.waits ((Arg.ofFlockAt D N ε names).oneWait.shape y) *
        (Arg.ofFlockAt D N ε names).oneWait.k ((Arg.ofFlockAt D N ε names).oneWait.shape y) =
      1 + Claims.Flock.kAt N (names y).1 (names y).2 := by
  show 1 + 1 * Claims.Flock.kAt N (names y).1 (names y).2 = 1 + Claims.Flock.kAt N (names y).1 (names y).2
  rw [Nat.one_mul]

/-- **Flock's masked record at one table of a session finds at one wait what it finds, at half the rate**, at SHA-512
at the deployed relation, with no premise, at every way a statement names a table and a Flock statement:
`oneWaitFindsAt` at `FlockDeploy.KS.zk_oneWaitFindsAt_untrim` and the deployed decoder's list bound. -/
theorem oneWaitFindsAt_masked (names : Msg → ℕ × Msg) :
    (Arg.ofFlockAt (FlockDeploy.deployZk _root_.Flock.Readable.relation) (FlockDeploy.KS.wordLenJ true)
      (FlockDeploy.KS.tableErrAt true) names).OneWaitFinds FlockDeploy.sha512 :=
  oneWaitFindsAt FlockDeploy.sha512 _ _ _ names FlockDeploy.KS.zk_oneWaitFindsAt_untrim
    (Claims.Sampling.listWithin_deployZk _)

/-- **A compiled verifier over Flock's masked record at one table of a session: what a session's extraction finds,
its one wait finds at half the rate** (`Compiled.sessionFinds_of` at `oneWaitFindsAt_masked`), wherever the verifier's
sessions' runs are fixed by their coins (`Compiled.RunsFixed`). -/
theorem sessionFindsAt_masked (names : Msg → ℕ × Msg)
    (L : Compiled FlockDeploy.Coin
      (Arg.ofFlockAt (FlockDeploy.deployZk _root_.Flock.Readable.relation) (FlockDeploy.KS.wordLenJ true)
        (FlockDeploy.KS.tableErrAt true) names))
    (hS : L.RunsFixed FlockDeploy.sha512) : L.SessionFinds FlockDeploy.sha512 :=
  Compiled.sessionFinds_of (oneWaitFindsAt_masked names) hS

end Claims.SamplingGeneric
