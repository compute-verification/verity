import Proofs.Flock.Claims.ZkDecodeFlock
import Flock.ZkProof

/-!
# The masked verifier's opened level-0 columns: `cols` for a masked run

In a masked session table `i`'s two reps (`Flock.Zk.verifyRep`, streams `<t>/rep0` and `<t>/rep1`) each open level 0
of the one committed codeword (both reps' caps must be the committed `root_B`, S18). Rep `r`'s proof (`Zk.decode`)
carries `q₀` opened rows of `2^k0 + Zk.EXTRA_LANES` symbols (`ligerito.initialProof.rows`), at the positions
`Flock.positions lv₀ coins` of one round's coins (`Zk.queryLevel0`). Every squeeze of at least one coin consumes one
recorded round and no squeeze's count depends on the proof, so that round's index in the rep's stream is fixed by the
shape (`zkQueryRound`): the masked replay's `m + kLog − 6` rounds (`Zk.replay`), the ring switch's `claims + 1`
(`Flock.ringSwitch`), then Ligerito's `k0 + 8` before level 0's queries (`Zk.ligerito`: the OOD point and its coin, the
`k0` lane challenges, the pair's `ρ_0, ρ_1`, and level 1's two OOD intros of two rounds each).

The reader takes no hash: it neither replays the transcript nor checks a digest, so it reads the same columns under
every `H`. `zkRepCols` is one rep's columns (position `j` with row `j`, read in GF(2^128)); `zkRunCols` is a table's
run, both reps' columns, one per position and each below `2^d` (`zkRunCols_nodup`, `zkRunCols_lt`), at most `q₀` per
rep (`zkRunCols_length_le`), so one run's two reps give fewer columns than the Johnson count
(`zkRunCols_length_lt_zkNMinD`).

What the reader takes from a run is a `ZkRun`: the table's `m`, `kLog`, claim count (`2 +` the setup's extra claims)
and Merkle widths, and each rep's recorded rounds and proof bytes. Reading those off `Seen` is the deployed verifier's
message layout's (`Deploy.V`).
-/

namespace Claims.Flock

open FlockSoundness FlockSoundness.Model

local notation "𝔽" => FlockLevel3.GF128

/-- **The round of level 0's query coins** in a masked rep's stream: after the masked replay's `m + kLog − 6` rounds,
the ring switch's `claims + 1` and Ligerito's `k0 + 8` (at `m ≥ 14`, where the replay's outer coins are a round). -/
def zkQueryRound (m kLog k0 claims : ℕ) : ℕ := (m + (kLog - 6)) + (claims + 1) + (k0 + 8)

/-- The masked verifier's proof shape for a table at `m` (`Zk.verifyRep`'s `shape`). -/
def zkProofShape (m kLog claims hashLen : ℕ) (saltLen : Option ℕ) : Flock.Shape :=
  { m, kLog, claims, sched := (zkSched m).1, hashLen, saltLen }

/-- **One rep's opened level-0 columns** from its query coins and opened rows: position `j` of
`Flock.positions lv₀ coins` with row `j`, for `j < q₀`, each symbol read in GF(2^128). -/
def zkRepCols (lv₀ : Flock.Level) (coins : Array Flock.F128) (rows : Array (Array Flock.F128)) :
    List (ℕ × List 𝔽) :=
  ((Flock.positions lv₀ coins).toList.zip (rows.toList.take lv₀.q)).map fun p =>
    (p.1, p.2.toList.map FlockLevel3.GF128.of)

/-- **A rep's opened level-0 columns** from its recorded rounds and proof bytes: none if the proof does not decode. -/
def zkRepColsOf (sh : Flock.Shape) (rounds : Array Flock.Round) (bytes : ByteArray) : List (ℕ × List 𝔽) :=
  match Flock.Zk.decode sh bytes with
  | .ok zp =>
    zkRepCols (sh.sched.levels.getD 0 default)
      ((rounds[zkQueryRound sh.m sh.kLog sh.sched.k0 sh.claims]?.map (·.coins)).getD #[])
      zp.proof.ligerito.initialProof.rows
  | .error _ => []

/-- What the reader takes from a masked run, for one table: the shape's numbers and each rep's recorded rounds and
proof bytes. -/
structure ZkRun where
  m : ℕ
  kLog : ℕ
  claims : ℕ
  hashLen : ℕ
  saltLen : Option ℕ
  reps : List (Array Flock.Round × ByteArray)

/-- The run's proof shape. -/
def ZkRun.shape (r : ZkRun) : Flock.Shape := zkProofShape r.m r.kLog r.claims r.hashLen r.saltLen

/-- The columns at positions below `2^d`, one per position. -/
def distinctBelow (d : ℕ) (os : List (ℕ × List 𝔽)) : List (ℕ × List 𝔽) :=
  (os.filter fun p => decide (p.1 < 2 ^ d)).pwFilter fun a b => a.1 ≠ b.1

/-- **A masked run's opened level-0 columns** for one table: its reps' columns, one per position, below `2^d`. -/
def zkRunCols (r : ZkRun) : List (ℕ × List 𝔽) :=
  distinctBelow (zkLevel0 r.m).d (r.reps.flatMap fun x => zkRepColsOf r.shape x.1 x.2)

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
  · exact length_zkRepCols_le _ _ _
  · exact Nat.zero_le _

/-- **A masked run's columns sit at distinct positions.** -/
theorem zkRunCols_nodup (r : ZkRun) : ((zkRunCols r).map Prod.fst).Nodup := distinctBelow_nodup _ _

/-- **A masked run's columns sit below `2^d`**, so below `2^64` (`zkShape_fits`). -/
theorem zkRunCols_lt (r : ZkRun) : ∀ p ∈ zkRunCols r, p.1 < 2 ^ (zkLevel0 r.m).d := fun _ h => distinctBelow_lt h

/-- **A masked run opens at most `q₀` level-0 columns per rep.** -/
theorem zkRunCols_length_le (r : ZkRun) : (zkRunCols r).length ≤ r.reps.length * (zkLevel0 r.m).q := by
  refine (length_distinctBelow_le _ _).trans ?_
  rw [List.length_flatMap]
  refine (List.sum_le_length_nsmul _ (zkLevel0 r.m).q fun n hn => ?_).trans (by simp)
  obtain ⟨x, -, rfl⟩ := List.mem_map.1 hn
  exact length_zkRepColsOf_le r.shape x.1 x.2

/-- **One masked run's two reps open fewer level-0 columns than the Johnson count** of a trace at the run's `m`, at
`0 < agree ≤ 1`. -/
theorem zkRunCols_length_lt_zkNMinD {agree : ℝ} (ha0 : 0 < agree) (ha1 : agree ≤ 1) (r : ZkRun)
    (h2 : r.reps.length ≤ 2) (M : ZkTrace) (hm : M.1.m = r.m) : (zkRunCols r).length < zkNMinD agree M := by
  refine lt_of_le_of_lt ((zkRunCols_length_le r).trans (Nat.mul_le_mul_right _ h2)) ?_
  rw [← hm]
  exact zkQueries_lt_zkNMinD ha0 ha1 M

end Claims.Flock
