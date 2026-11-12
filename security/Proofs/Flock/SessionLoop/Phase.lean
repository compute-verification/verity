module

public import Proofs.Flock.SessionLoop.Statement
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
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
import Mathlib.Topology.Sheaves.Init

/-!
# The worker's view by phase

The session's values `vs` (`RefSession.SwVals`) read the statement, the witness, the nonce, the opened coins and what the
workers see of each of the statement's tables (`RefSession.seenOf`: the mask slot, level 0's hiding and level 0's
salts). The deployed worker reads that much until the algebraic phase's last coins, and may read more after them (the
salts for levels ≥ 1, handed at the opening). `SessionLoopZKDeployed`'s premises on the worker (`LeanRelease.LiveRoundEq`,
`LeanRelease.FillRoundEq`) read it only at histories of at most `refK x` rounds, the algebraic phase's, so they hold at
such a worker where they hold at the pad-blind worker of its algebraic phase.

* `ZkServer.copy_congr`, `pass_congr`, `live_congr`: two workers that agree on every history no longer than `h` have
  the same copy, the same pass and the same liveness after `h`.
* `LeanRelease.liveRoundEq_congr`, `fillRoundEq_congr`: so either premise holds at a worker exactly where it holds at
  any worker that agrees with it at histories of at most `refK x` rounds.
* `Loop.seenOfExe`, `Loop.seenOf_exe`: what the workers see of each cell (`Masked.seenOf`), cut to the statement's
  tables, is the session's `RefSession.seenOf`, what `vs` reads.
* `Loop.AlgWorker`, `Loop.bwOf`: a worker of the algebraic phase reads the statement, the witness, the nonce, what the
  workers see of the statement's tables and, once `Hello` is answered, the answers after it read as coins
  (`List.tail?`). The server checks `Hello`'s answer and hands the prover its own id for the session in its place, so
  the deployed worker never reads it. `bwOf` is it as a pad-blind worker (`Loop.blind`), handed `Masked.seenOf`.
* `Loop.phased`: the algebraic phase's worker until its last coins, any worker after them; `Loop.phased_eq`: in the
  algebraic phase its value is the algebraic phase's worker's at the session's `seenOf`.
* `Loop.PhaseBlind`, **`Loop.PhaseBlind.exists_phased`**: a worker that, in the algebraic phase, reads only the nonce,
  the session's `seenOf` and the answers after `Hello` as coins is `phased` at some algebraic phase's worker, after the
  phase itself.
* **`Loop.liveRoundEq_phased`, `Loop.fillRoundEq_phased`**: the premises at `phased bx post` are the premises at the
  pad-blind `blind (bwOf bx)`, whatever `post` reads.
-/

@[expose] public section

namespace Claims.ZKServer.ZkServer

/-- **The copy after `h` reads the worker at histories no longer than `h` only.** -/
theorem copy_congr {W R : Type} (S : ZkServer W R) {wk wk' : S.Worker} {x : Msg} {w : W} {ρ : R}
    {h : List (Msg × Msg)} (he : ∀ h' : List (Msg × Msg), h'.length ≤ h.length → wk x w ρ h' = wk' x w ρ h') :
    S.copy wk x w ρ h = S.copy wk' x w ρ h := by
  funext i s
  simp only [copy, he (h.take i) (List.length_take_le' _ _)]

/-- **Whether the round after `h` passes reads the worker at histories no longer than `h` only.** -/
theorem pass_congr {W R : Type} (S : ZkServer W R) {wk wk' : S.Worker} {x : Msg} {w : W} {ρ : R} (κ : S.Chk)
    {h : List (Msg × Msg)} (he : ∀ h' : List (Msg × Msg), h'.length ≤ h.length → wk x w ρ h' = wk' x w ρ h') :
    S.pass wk x w ρ κ h = S.pass wk' x w ρ κ h := by
  simp only [pass, S.copy_congr he, he h le_rfl]

/-- **Whether every round so far passed reads the worker at histories no longer than `h` only.** -/
theorem live_congr {W R : Type} (S : ZkServer W R) {wk wk' : S.Worker} {x : Msg} {w : W} {ρ : R} (κ : S.Chk)
    {h : List (Msg × Msg)} (he : ∀ h' : List (Msg × Msg), h'.length ≤ h.length → wk x w ρ h' = wk' x w ρ h') :
    S.live wk x w ρ κ h = S.live wk' x w ρ κ h := by
  unfold live
  congr 1
  funext i
  exact S.pass_congr κ fun h' hl => he h' (hl.trans (List.length_take_le' _ _))

end Claims.ZKServer.ZkServer

namespace Claims.ZKServer

open Claims Claims.ZKMal
open FlockSoundness.Discharge

namespace LeanRelease

/-- **`LiveRoundEq` reads the worker in the algebraic phase only**: it holds at `wk` exactly where it holds at a worker
that agrees with `wk` at every history of at most `refK x` rounds. -/
theorem liveRoundEq_congr (S : RefSession) (vs : S.SwVals) (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand))
    (H : Msg → Msg) {wk wk' : L.toSrv.Worker} (x : Msg) (w : Flock.Wit)
    (he : ∀ ρ (h : List (Msg × Msg)), h.length ≤ refK x → wk x w ρ h = wk' x w ρ h) :
    LiveRoundEq S vs L H wk x w ↔ LiveRoundEq S vs L H wk' x w := by
  have key : ∀ {wk wk' : L.toSrv.Worker}, (∀ ρ (h : List (Msg × Msg)), h.length ≤ refK x → wk x w ρ h = wk' x w ρ h) →
      LiveRoundEq S vs L H wk x w → LiveRoundEq S vs L H wk' x w := by
    intro wk wk' he hE ν t κ m₀ ρ₀ ops v h0 hlt hl hok hr hv
    have hle : ((m₀, ρ₀) :: ops).length ≤ refK x := by
      simp only [List.length_cons]
      omega
    refine hE ν t κ m₀ ρ₀ ops v h0 hlt ?_ hok hr ?_
    · rwa [L.toSrv.live_congr κ fun h' hl' => he _ h' (hl'.trans hle)]
    · rwa [he _ _ hle]
  exact ⟨key he, key fun ρ h hl => (he ρ h hl).symm⟩

/-- **`FillRoundEq` reads the worker in the algebraic phase only**, as `liveRoundEq_congr`. -/
theorem fillRoundEq_congr (S : RefSession) (vs : S.SwVals) (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand))
    (H : Msg → Msg) {wk wk' : L.toSrv.Worker} (x : Msg) (w : Flock.Wit)
    (he : ∀ ρ (h : List (Msg × Msg)), h.length ≤ refK x → wk x w ρ h = wk' x w ρ h) :
    FillRoundEq S vs L H wk x w ↔ FillRoundEq S vs L H wk' x w := by
  have key : ∀ {wk wk' : L.toSrv.Worker}, (∀ ρ (h : List (Msg × Msg)), h.length ≤ refK x → wk x w ρ h = wk' x w ρ h) →
      FillRoundEq S vs L H wk x w → FillRoundEq S vs L H wk' x w := by
    intro wk wk' he hE ν t κ m₀ ρ₀ ops h0 hlt hl hok hr
    have hle : ((m₀, ρ₀) :: ops).length ≤ refK x := by
      simp only [List.length_cons]
      omega
    refine hE ν t κ m₀ ρ₀ ops h0 hlt ?_ hok hr
    rwa [L.toSrv.live_congr κ fun h' hl' => he _ h' (hl'.trans hle)]
  exact ⟨key he, key fun ρ h hl => (he ρ h hl).symm⟩

end LeanRelease

namespace Loop

/-- **What the workers see of the statement's tables** from what they see of each cell (`Masked.ExeSeen`): table `j`'s
mask slot, lanes' paddings with the extra lanes, and level 0's salts, each its cell's cut to the table's sizes. -/
noncomputable def seenOfExe (x : Msg) (e : Fin Claims.Flock.zkTableCap → Masked.ExeSeen) :
    ∀ j, ZkView.Seen ((CoinLayer.session loopNr Masked.dataE).C x j).z := fun j =>
  (Masked.cut (Masked.fits_zkTab Masked.PE (Masked.zkTab x j)).nMask (e (Fin.castLE (Masked.jOf_le_cap x) j)).1,
    Masked.readLanes (Masked.fits_zkTab Masked.PE (Masked.zkTab x j)) (e (Fin.castLE (Masked.jOf_le_cap x) j)).2.1,
    Masked.cut (Nat.add_le_add (Masked.fits_zkTab Masked.PE (Masked.zkTab x j)).nS
      (Masked.fits_zkTab Masked.PE (Masked.zkTab x j)).nU) (e (Fin.castLE (Masked.jOf_le_cap x) j)).2.2)

/-- **What the workers see of the tape is what the session's values read**: `seenOfExe` of `Masked.seenOf` is the
session's `RefSession.seenOf`. -/
theorem seenOf_exe (x : Msg) (ρ : (CoinLayer.session loopNr Masked.dataE).refP.Nonce ×
    (CoinLayer.session loopNr Masked.dataE).refP.Rand) :
    (CoinLayer.session loopNr Masked.dataE).seenOf x ρ.2 = seenOfExe x (Masked.seenOf ρ).2 :=
  rfl

variable (L : LeanRelease Flock.Wit
  ((CoinLayer.session loopNr Masked.dataE).refP.Nonce × (CoinLayer.session loopNr Masked.dataE).refP.Rand))

/-- **A worker of the algebraic phase**: a function of the statement, the witness, the nonce, what the workers see of
the statement's tables (what the session's values read, `RefSession.SwVals`) and, once `Hello` is answered, the
verifier's answers after it read as coins (`none` before), to each slot's value. -/
abbrev AlgWorker : Type :=
  (x : Msg) → Flock.Wit → (CoinLayer.session loopNr Masked.dataE).Nonce →
    (∀ j, ZkView.Seen ((CoinLayer.session loopNr Masked.dataE).C x j).z) → Option (List FlockDeploy.Coin) → ℕ →
      Option L.toSrv.Val

/-- **An algebraic phase's worker as a pad-blind one** (`Loop.blind`'s): handed what the workers see of each cell, it
reads the statement's tables' part (`seenOfExe`), and of the answers it reads those after `Hello`'s. -/
noncomputable def bwOf (bx : AlgWorker L) : L.toSrv.BWorker Seen FlockDeploy.Coin :=
  fun x w r cs s => bx x w r.1 (seenOfExe x r.2) cs.tail? s

/-- **The worker by phase**: the algebraic phase's worker `bx`, pad-blind, at histories of at most `refK x` rounds, up to
the algebraic phase's last coins; any worker `post` after them, which may read more of the tape (the salts for levels ≥
1, which the loop hands the workers at the opening). -/
noncomputable def phased (bx : AlgWorker L) (post : L.toSrv.Worker) : L.toSrv.Worker :=
  fun x w ρ h => if h.length ≤ refK x then blind L (bwOf L bx) x w ρ h else post x w ρ h

theorem phased_alg (bx : AlgWorker L) (post : L.toSrv.Worker) (x : Msg) (w : Flock.Wit) (ρ : _)
    (h : List (Msg × Msg)) (hl : h.length ≤ refK x) : phased L bx post x w ρ h = blind L (bwOf L bx) x w ρ h :=
  ite_eq_left hl

/-- **In the algebraic phase the worker reads what the session's values read**: at the nonce `ν` and the tape `t`, its
value is `bx`'s at the session's `seenOf x t` and the answers after `Hello` read as coins. -/
theorem phased_eq (bx : AlgWorker L) (post : L.toSrv.Worker) (x : Msg) (w : Flock.Wit)
    (ν : (CoinLayer.session loopNr Masked.dataE).Nonce) (t : (CoinLayer.session loopNr Masked.dataE).Tape)
    (h : List (Msg × Msg)) (hl : h.length ≤ refK x) (s : ℕ) :
    phased L bx post x w (ν, t) h s =
      bx x w ν ((CoinLayer.session loopNr Masked.dataE).seenOf x t) (h.map fun m => CoinLayer.rd m.2).tail? s := by
  rw [phased_alg L bx post x w _ h hl]
  rfl

/-- **A worker reads, in the algebraic phase, only what the session's values read**: at every history of at most
`refK x` rounds, what it returns is the same at two tapes with the same nonce and the same `RefSession.seenOf`, and two
histories whose answers after `Hello` read as the same coins (`CoinLayer.rd`), `Hello` answered in both or in neither.
After the algebraic phase it may read anything. -/
def PhaseBlind (wk : L.toSrv.Worker) : Prop :=
  ∀ x w (ρ ρ' : (CoinLayer.session loopNr Masked.dataE).refP.Nonce × (CoinLayer.session loopNr Masked.dataE).refP.Rand)
    (h h' : List (Msg × Msg)) s, h.length ≤ refK x → ρ.1 = ρ'.1 →
    (CoinLayer.session loopNr Masked.dataE).seenOf x ρ.2 = (CoinLayer.session loopNr Masked.dataE).seenOf x ρ'.2 →
    (h.map fun m => CoinLayer.rd m.2).tail? = (h'.map fun m => CoinLayer.rd m.2).tail? → wk x w ρ h s = wk x w ρ' h' s

theorem length_eq_of_tail? {α : Type} {l l' : List α} (h : l.tail? = l'.tail?) : l.length = l'.length := by
  cases l <;> cases l' <;> simp_all

theorem phaseBlind_phased (bx : AlgWorker L) (post : L.toSrv.Worker) : PhaseBlind L (phased L bx post) := by
  intro x w ρ ρ' h h' s hl h1 hs hh
  have hl' : h'.length ≤ refK x := by
    have := length_eq_of_tail? hh
    simp only [List.length_map] at this
    omega
  refine (phased_eq L bx post x w ρ.1 ρ.2 h hl s).trans (Eq.trans ?_ (phased_eq L bx post x w ρ'.1 ρ'.2 h' hl' s).symm)
  rw [h1, hs, hh]

/-- **A history whose answers after `Hello` read as the coins `oc`**: none where `Hello` is not answered, else
`Hello`'s and then one round per coin, each answered by a message `CoinLayer.rd` reads as it. -/
noncomputable def coinsHist (oc : Option (List FlockDeploy.Coin)) : List (Msg × Msg) :=
  match oc with
  | none => []
  | some cs => (([] : Msg), ([] : Msg)) :: cs.map fun a => (([] : Msg), Function.invFun CoinLayer.rd a)

/-- **A worker that reads only what the session's values read in the algebraic phase is a phased one**: its algebraic
phase is an algebraic phase's worker's, and after it, it is itself. -/
theorem PhaseBlind.exists_phased {wk : L.toSrv.Worker} (hb : PhaseBlind L wk) :
    ∃ bx : AlgWorker L, phased L bx wk = wk := by
  refine ⟨fun x w ν s oc i => wk x w (ν, Function.invFun ((CoinLayer.session loopNr Masked.dataE).seenOf x) s)
    (coinsHist oc) i, ?_⟩
  funext x w ρ h i
  by_cases hl : h.length ≤ refK x
  · refine (phased_eq L _ wk x w ρ.1 ρ.2 h hl i).trans ?_
    cases h with
    | nil => exact hb x w _ _ _ [] i (Nat.zero_le _) rfl (Function.invFun_eq ⟨ρ.2, rfl⟩) rfl
    | cons e t =>
      refine hb x w _ _ _ (e :: t) i ?_ rfl (Function.invFun_eq ⟨ρ.2, rfl⟩) ?_
      · simpa only [coinsHist, List.map_cons, List.tail?_cons, List.length_cons, List.length_map] using hl
      · simp only [coinsHist, List.map_cons, List.tail?_cons, List.map_map, Function.comp_def, Option.some.injEq]
        exact List.map_congr_left fun m _ => Function.invFun_eq ⟨m.2, rfl⟩
  · exact congrFun (ite_eq_right hl) i

/-- **`LiveRoundEq` at a phased worker is `LiveRoundEq` at its algebraic phase's pad-blind worker**, whatever the worker
after the phase reads. -/
theorem liveRoundEq_phased (vs : (CoinLayer.session loopNr Masked.dataE).SwVals) (H : Msg → Msg) (bx : AlgWorker L)
    (post : L.toSrv.Worker) (x : Msg) (w : Flock.Wit) :
    LeanRelease.LiveRoundEq (CoinLayer.session loopNr Masked.dataE) vs L H (phased L bx post) x w ↔
      LeanRelease.LiveRoundEq (CoinLayer.session loopNr Masked.dataE) vs L H (blind L (bwOf L bx)) x w :=
  LeanRelease.liveRoundEq_congr _ vs L H x w fun ρ h hl => phased_alg L bx post x w ρ h hl

/-- **`FillRoundEq` at a phased worker is `FillRoundEq` at its algebraic phase's pad-blind worker**, whatever the worker
after the phase reads. -/
theorem fillRoundEq_phased (vs : (CoinLayer.session loopNr Masked.dataE).SwVals) (H : Msg → Msg) (bx : AlgWorker L)
    (post : L.toSrv.Worker) (x : Msg) (w : Flock.Wit) :
    LeanRelease.FillRoundEq (CoinLayer.session loopNr Masked.dataE) vs L H (phased L bx post) x w ↔
      LeanRelease.FillRoundEq (CoinLayer.session loopNr Masked.dataE) vs L H (blind L (bwOf L bx)) x w :=
  LeanRelease.fillRoundEq_congr _ vs L H x w fun ρ h hl => phased_alg L bx post x w ρ h hl

end Loop

end Claims.ZKServer
