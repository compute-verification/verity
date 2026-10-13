import Proofs.Flock.C3.Compose.HoloCoinTys

/-!
# The sparse phases' and R's streams at the model's coins

The inner shape's sparse phases' and R's streams (`holoStreams`, at the verifier's RecSparse shape and R's
`StackedShape`) draw the coins the model's stages draw (`holoForms`' tail), round for round in size (`holoSizes`):

* `degrees_length`: H3's rounds are the entries' and the slots' (`nu + sLog`); `stacked_iff`: the verifier's ρ_t round
  (`stacked`) is where the model has its message (`tLog ≠ 0`); `pLog_sparseOf`, `mReg_sparseOf`: the planes' bits and
  R's table are the model's at `sparseOf`.
* `holoSizes_sparse`: a sparse phase's stream draws `sparseForms`' scalars.
* `holoSizes_stacked`: R's stream draws R's opening's two claims' coins and compiled Ligerito's at `fast100`'s schedule
  (`opForms 2 ++ ligForms (schedAt m)`), by evaluating `stackedRounds` at each `m` below 36.
* `holoSizes_holoStreams`: together, after the reps', where the inner shape names R at its table, as
  `candidate3Statements` checks.
-/

namespace FlockC3.Compose

open Flock Flock.RecVStar Flock.Guarantees FlockSoundness.Discharge.Exec FlockSoundness.Discharge.PlaysCoin
open FlockSoundness.Discharge.Recursion FlockSoundness.Refine FlockLevel3

/-! ## The sparse phases' streams -/

/-- **The model's sparse shape of the verifier's** (`RecSparse.Shape`, `rec-inner --sparse`). -/
def sparseOf (sp : RecSparse.Shape) : FlockSoundness.Model.SparseShape := ⟨sp.nu, sp.ell, sp.sLog, sp.tLog⟩

/-- **H3's rounds are the entries' and the slots'**: `degrees` has one per entry and one per slot. -/
theorem degrees_length (sp : RecSparse.Shape) : sp.degrees.length = sp.nu + sp.sLog := by
  simp [RecSparse.Shape.degrees]

/-- **ρ_t's round is where the model has it**: the verifier's at `stacked`, the model's at `tLog ≠ 0`. -/
theorem stacked_iff (sp : RecSparse.Shape) : sp.stacked = true ↔ sp.tLog ≠ 0 := by
  simp only [RecSparse.Shape.stacked, decide_eq_true_eq]
  exact Nat.pos_iff_ne_zero

theorem pLog_sparseOf (sp : RecSparse.Shape) : (sparseOf sp).pLog = sp.pLog := by
  show Nat.log2 (2 * sp.ell + 1) + 1 = Nat.log2 (2 * sp.ell + 2 - 1) + 1
  rw [show 2 * sp.ell + 2 - 1 = 2 * sp.ell + 1 by omega]

/-- **R's table is the model's**: the verifier's `tableM` is the model's registered table's variables. -/
theorem mReg_sparseOf (sp : RecSparse.Shape) : (sparseOf sp).mReg = tableM sp := by
  show max 22 (sp.nu + (sparseOf sp).pLog + sp.tLog) = max PAD_TO (sp.nu + sp.pLog + sp.tLog)
  rw [pLog_sparseOf]
  rfl

theorem flatMap_replicate_fst : ∀ l : List SchedRound,
    l.flatMap (fun q => List.replicate q.1 1) = List.replicate (l.map (·.1)).sum 1
  | [] => rfl
  | q :: l => by
    rw [List.flatMap_cons, flatMap_replicate_fst l, List.map_cons, List.sum_cons, List.replicate_append_replicate]

theorem holoSizes_append (L L' : List (HoloStream × Array SchedRound)) :
    holoSizes (L ++ L') = holoSizes L ++ holoSizes L' :=
  List.flatMap_append

/-- **A sparse phase's stream draws the model's coins**: one scalar per coin of `sparseKinds` at the model's shape. -/
theorem holoSizes_sparse (x : Fin 2) (sp : RecSparse.Shape) :
    holoSizes [(.sparse x, sparseRounds sp)] = (sparseForms (sparseOf sp)).map formSize := by
  have hs : holoSizes [(.sparse x, sparseRounds sp)] =
      (sparseRounds sp).toList.flatMap fun q => List.replicate q.1 1 := by
    simp [holoSizes]
  rw [hs, flatMap_replicate_fst, sparseForms, List.map_replicate]
  show List.replicate _ 1 = List.replicate _ 1
  congr 1
  rw [pLog_sparseOf]
  cases h : sp.stacked
  · have h0 : sp.tLog = 0 := by
      simpa [RecSparse.Shape.stacked] using h
    simp [sparseRounds, RecSparse.PRE_ROUNDS, h, h0, degrees_length, sparseOf]
    omega
  · simp [sparseRounds, RecSparse.PRE_ROUNDS, h, degrees_length, sparseOf]
    omega

/-! ## R's stream -/

/-- **R's stream draws the model's coins**: R's opening's two (`opShape 2`) and compiled Ligerito's at `fast100`'s
schedule, by evaluating `stackedRounds` at each `m` below 36; past 35 `fast100` has none, nor has `stackedRounds`. -/
theorem holoSizes_stacked {m : ℕ} {rs : Array SchedRound} (h : stackedRounds ⟨m, STACKED_CLAIMS⟩ = some rs) :
    holoSizes [(.stacked, rs)] = (opForms 2 ++ ligForms (schedAt m)).map formSize := by
  by_cases hm : m < 36
  · have key : ∀ m' < 36, (stackedRounds ⟨m', STACKED_CLAIMS⟩).all
        (fun rs => holoSizes [(.stacked, rs)] == (opForms 2 ++ ligForms (schedAt m')).map formSize) := by
      decide +kernel
    have := key m hm
    rw [h] at this
    simpa using this
  · have hb : (m < 22 || m > 35) = true := by
      simp only [Bool.or_eq_true, decide_eq_true_eq]
      omega
    have h0 : Flock.fast100 m = none := by
      unfold Flock.fast100
      simp [hb]
    have hn : stackedRounds ⟨m, STACKED_CLAIMS⟩ = none := by
      unfold stackedRounds
      rw [h0]
      rfl
    rw [hn] at h
    cases h

/-! ## The sparse phases' and R's streams together -/

/-- **The sparse phases' and R's streams draw the model's coins** where the inner shape names a RecSparse shape and R
at its table (`candidate3Statements` checks R's `m` is `tableM` and its claims `STACKED_CLAIMS`): after the reps', the
history's coin sizes are those of `holoForms`' tail at the model's shape (`sparseOf`) and R's schedule at its table. -/
theorem holoSizes_holoStreams {m kLog claims : ℕ} {sp : RecSparse.Shape} {L : List (HoloStream × Array SchedRound)}
    (h : holoStreams m kLog claims (some sp) (some ⟨tableM sp, STACKED_CLAIMS⟩) = some L) :
    ∃ rep, holoRounds m kLog claims = some rep ∧ holoSizes L = holoSizes [(.rep 0, rep), (.rep 1, rep)] ++
      (sparseForms (sparseOf sp) ++ (sparseForms (sparseOf sp) ++
        (opForms 2 ++ ligForms (schedAt (sparseOf sp).mReg)))).map formSize := by
  unfold holoStreams at h
  cases hr : holoRounds m kLog claims with
  | none => simp [hr] at h
  | some rep =>
    cases hs : stackedRounds ⟨tableM sp, STACKED_CLAIMS⟩ with
    | none => simp [hr, hs] at h
    | some rs =>
      simp only [hr, hs, Option.bind_eq_bind, Option.bind_some, Option.map_some] at h
      refine ⟨rep, rfl, ?_⟩
      have hL : L = [(HoloStream.rep 0, rep), (HoloStream.rep 1, rep)] ++
          ([(HoloStream.sparse 0, sparseRounds sp), (HoloStream.sparse 1, sparseRounds sp)] ++
            [(HoloStream.stacked, rs)]) := by
        simpa using h.symm
      rw [hL, holoSizes_append, holoSizes_append,
        show [(HoloStream.sparse 0, sparseRounds sp), (HoloStream.sparse 1, sparseRounds sp)] =
          [(HoloStream.sparse 0, sparseRounds sp)] ++ [(HoloStream.sparse 1, sparseRounds sp)] from rfl,
        holoSizes_append, holoSizes_sparse, holoSizes_sparse, holoSizes_stacked hs, mReg_sparseOf]
      simp only [List.map_append, List.append_assoc]

end FlockC3.Compose
