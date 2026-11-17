module

public import Proofs.Flock.Claims.ZkDecodeFlock
public import Batteries.Data.ByteArray
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

/-! The masked verifier's opened level-0 columns: `cols` for a masked run. -/

@[expose] public section

namespace Claims.Flock

open FlockSoundness FlockSoundness.Model

local notation "𝔽" => FlockLevel3.GF128

variable {wideLeaf : Bool}

/-- **The round of level 0's query coins** in a masked rep's stream: after the masked replay's `m + kLog − 6` rounds,
the ring switch's `claims + 1` and Ligerito's `k0 + 8` (at `m ≥ 14`, where the replay's outer coins are a round). -/
def zkQueryRound (m kLog k0 claims : ℕ) : ℕ := (m + (kLog - 6)) + (claims + 1) + (k0 + 8)

/-- The masked verifier's proof shape for a table at `m` (`Zk.verifyRep`'s `shape`). -/
def zkProofShape (wideLeaf : Bool) (m kLog claims hashLen : ℕ) (saltLen : Option ℕ) : Flock.Shape :=
  { m, kLog, claims, sched := (zkSched wideLeaf m).1, hashLen, saltLen }

/-- **One rep's opened level-0 columns** from its query coins and opened rows: position `j` of
`Flock.positions lv₀ coins` with row `j`, for `j < q₀`, each symbol read in GF(2^128). -/
def zkRepCols (lv₀ : Flock.Level) (coins : Array Flock.F128) (rows : Array (Array Flock.F128)) :
    List (ℕ × List 𝔽) :=
  ((Flock.positions lv₀ coins).toList.zip (rows.toList.take lv₀.q)).map fun p =>
    (p.1, p.2.toList.map FlockLevel3.GF128.of)

/-- **A rep's opened level-0 columns** from its recorded rounds and proof bytes, each an opened row's first `2^k0`
symbols (the witness lanes): none if the proof does not decode. -/
def zkRepColsOf (sh : Flock.Shape) (rounds : Array Flock.Round) (bytes : ByteArray) : List (ℕ × List 𝔽) :=
  match Flock.Zk.decode sh bytes with
  | .ok zp =>
    (zkRepCols (sh.sched.levels.getD 0 default)
      ((rounds[zkQueryRound sh.m sh.kLog sh.sched.k0 sh.claims]?.map (·.coins)).getD #[])
      zp.proof.ligerito.initialProof.rows).map fun p => (p.1, p.2.take (2 ^ (sh.sched.levels.getD 0 default).lanes))
  | .error _ => []

/-- What the reader takes from a masked run, for one table: the shape's numbers and each rep's recorded rounds and
proof bytes, and the statement's version (`wideLeaf`, which the masked shape's schedule reads; a plain run's is
`false`). -/
structure ZkRun where
  m : ℕ
  kLog : ℕ
  claims : ℕ
  hashLen : ℕ
  saltLen : Option ℕ
  reps : List (Array Flock.Round × ByteArray)
  wideLeaf : Bool := false

/-- The run's proof shape. -/
def ZkRun.shape (r : ZkRun) : Flock.Shape := zkProofShape r.wideLeaf r.m r.kLog r.claims r.hashLen r.saltLen

/-- The columns at positions below `2^d`, one per position. -/
def distinctBelow (d : ℕ) (os : List (ℕ × List 𝔽)) : List (ℕ × List 𝔽) :=
  (os.filter fun p => decide (p.1 < 2 ^ d)).pwFilter fun a b => a.1 ≠ b.1

/-- **A masked run's opened level-0 columns** for one table: its reps' columns, one per position, below `2^d`. -/
def zkRunCols (r : ZkRun) : List (ℕ × List 𝔽) :=
  distinctBelow (zkLevel0 r.wideLeaf r.m).d (r.reps.flatMap fun x => zkRepColsOf r.shape x.1 x.2)

theorem distinctBelow_nodup (d : ℕ) (os : List (ℕ × List 𝔽)) : ((distinctBelow d os).map Prod.fst).Nodup :=
  List.pairwise_map.2 (List.pairwise_pwFilter _)

theorem distinctBelow_lt {d : ℕ} {os : List (ℕ × List 𝔽)} {p : ℕ × List 𝔽} (h : p ∈ distinctBelow d os) :
    p.1 < 2 ^ d := by
  have := (List.pwFilter_sublist _).subset h
  simpa using (List.mem_filter.1 this).2

theorem length_distinctBelow_le (d : ℕ) (os : List (ℕ × List 𝔽)) : (distinctBelow d os).length ≤ os.length :=
  (List.pwFilter_sublist _).length_le.trans (List.length_filter_le _ _)

theorem length_zkRepCols_le (lv₀ : Flock.Level) (coins : Array Flock.F128) (rows : Array (Array Flock.F128)) :
    (zkRepCols lv₀ coins rows).length ≤ lv₀.q := by
  simp only [zkRepCols, List.length_map, List.length_zip]
  exact min_le_of_right_le (List.length_take_le _ _)

theorem length_zkRepColsOf_le (sh : Flock.Shape) (rounds : Array Flock.Round) (bytes : ByteArray) :
    (zkRepColsOf sh rounds bytes).length ≤ (sh.sched.levels.getD 0 default).q := by
  unfold zkRepColsOf
  split
  · rw [List.length_map]
    exact length_zkRepCols_le _ _ _
  · exact Nat.zero_le _

theorem zkRunCols_nodup (r : ZkRun) : ((zkRunCols r).map Prod.fst).Nodup := distinctBelow_nodup _ _

theorem zkRunCols_lt (r : ZkRun) : ∀ p ∈ zkRunCols r, p.1 < 2 ^ (zkLevel0 r.wideLeaf r.m).d := fun _ h => distinctBelow_lt h

theorem zkRunCols_length_le (r : ZkRun) : (zkRunCols r).length ≤ r.reps.length * (zkLevel0 r.wideLeaf r.m).q := by
  refine (length_distinctBelow_le _ _).trans ?_
  rw [List.length_flatMap]
  refine (List.sum_le_length_nsmul _ (zkLevel0 r.wideLeaf r.m).q fun n hn => ?_).trans (by simp)
  obtain ⟨x, -, rfl⟩ := List.mem_map.1 hn
  exact length_zkRepColsOf_le r.shape x.1 x.2

theorem zkRunCols_length_lt_zkNMinD {agree : ℝ} (ha0 : 0 < agree) (ha1 : agree ≤ 1) (r : ZkRun)
    (h2 : r.reps.length ≤ 2) (M : ZkTrace) (hm : M.1.At r.wideLeaf r.m) :
    (zkRunCols r).length < zkNMinD agree M := by
  refine lt_of_le_of_lt ((zkRunCols_length_le r).trans (Nat.mul_le_mul_right _ h2)) ?_
  rw [← hm.level0]
  exact zkQueries_lt_zkNMinD ha0 ha1 M

end Claims.Flock
