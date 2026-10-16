import Proofs.Flock.C3.Compose.HoloCoinTys
import Proofs.Flock.C3.VStarList.Shape
import Proofs.Flock.Recursive.HoloRepCoins

/-!
# The sparse phases' and R's streams at the model's coins

The inner shape's sparse phases' and R's streams (`holoStreams`, at the verifier's RecSparse shape and R's
`StackedShape`) draw the coins the model's stages draw (`holoForms`' tail), round for round in size (`holoSizes`):

* `degrees_length`: H3's rounds are the entries' and the slots' (`nu + sLog`); `stacked_iff`: the verifier's ρ_t round
  (`stacked`) is where the model has its message (`tLog ≠ 0`); `pLog_sparseOf`, `mReg_sparseOf`: the planes' bits and
  R's table are the model's at `sparseOf`; `shape_holoAt`: `sparseOf` of the inner shape's `sparse` entry is the
  shape of candidate 3's class (`holoAt`) where `ℓ` is in its range.
* `holoSizes_sparse`: a sparse phase's stream draws `sparseForms`' scalars.
* `holoSizes_stacked`: R's stream draws R's opening's two claims' coins and compiled Ligerito's at `fast100`'s schedule
  (`opForms 2 ++ ligForms (schedAt m)`), by evaluating `stackedRounds` at each `m` below 36.
* `holoSizes_holoStreams`: together, after the reps', where the inner shape names R at its table, as
  `candidate3Statements` checks.
* `holoStreams_eq`, `holoStreams_sparse_eq`: the streams start with both reps' at `holoRounds`, then, where the inner
  shape names a RecSparse shape, both sparse phases', then R's.
* `holoSizes_reps`: both reps' streams draw `repForms`' coins at `fast100`'s schedule (`repRounds_sizes`): the rounds
  `kLog` and `claims` reach are the head (`repHead`, `repRounds_tail`), and past it, evaluating `repRounds` at each `m`
  below 36 at one `kLog` and `claims` gives the rest.
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

/-- **The model's sparse shape is the verifier's**: at `hp` the inner shape's `sparse` entry `sp` (`SparseIs`) and `ℓ`
in the class's range, candidate 3's class at `hp` (`holoAt`) has `sparseOf sp`'s shape. -/
theorem shape_holoAt {st : Flock.Stmt} {hp : HoloParams} {ish : Flock.RecVStar.Inner} {sp : RecSparse.Shape}
    (hsp : FlockC3.VStarList.SparseIs ish hp) (hs : ish.sparse = some sp) (hell : hp.EllOk (execClassAt st).cls.kLog) :
    (holoAt st hp).cls.shape = sparseOf sp := by
  obtain rfl := Option.some.inj (hs.symm.trans (show ish.sparse = _ from hsp))
  have h : (holoAt st hp).cls.slotLog = hp.ell := ite_eq_left hell
  show (⟨hp.nu, (holoAt st hp).cls.slotLog, hp.sLog, hp.tLog⟩ : FlockSoundness.Model.SparseShape) =
    ⟨hp.nu, hp.ell, hp.sLog, hp.tLog⟩
  exact congrArg (fun e => (⟨hp.nu, e, hp.sLog, hp.tLog⟩ : FlockSoundness.Model.SparseShape)) h

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

/-! ## The streams' list -/

open Flock.Guarantees.HoloRepCoins (obind_some repRounds_coins fast100_facts)

/-- **The streams start with both reps'**, at `holoRounds`. -/
theorem holoStreams_eq {m kLog claims : ℕ} {sp : Option RecSparse.Shape} {sk : Option StackedShape}
    {L : List (HoloStream × Array SchedRound)} (h : holoStreams m kLog claims sp sk = some L) :
    ∃ rs L', holoRounds m kLog claims = some rs ∧ L = (.rep 0, rs) :: (.rep 1, rs) :: L' := by
  unfold holoStreams at h
  obtain ⟨rs, hr, h⟩ := obind_some h
  dsimp only at h
  cases sk with
  | none => exact ⟨rs, _, hr, (Option.some.inj h).symm⟩
  | some sh =>
    obtain ⟨R, -, h⟩ := obind_some h
    exact ⟨rs, _, hr, (Option.some.inj h).symm⟩

/-- **The streams at a RecSparse shape**: both reps', both sparse phases', then R's, if any. -/
theorem holoStreams_sparse_eq {m kLog claims : ℕ} {sp : RecSparse.Shape} {sk : Option StackedShape}
    {L : List (HoloStream × Array SchedRound)} (h : holoStreams m kLog claims (some sp) sk = some L) :
    ∃ rs R, holoRounds m kLog claims = some rs ∧ (∀ e ∈ R, e.1 = .stacked) ∧
      L = (.rep 0, rs) :: (.rep 1, rs) :: (.sparse 0, sparseRounds sp) :: (.sparse 1, sparseRounds sp) :: R := by
  unfold holoStreams at h
  obtain ⟨rs, hr, h⟩ := obind_some h
  dsimp only at h
  cases sk with
  | none => exact ⟨rs, [], hr, by simp, by simpa using (Option.some.inj h).symm⟩
  | some sh =>
    obtain ⟨R, hR, h⟩ := obind_some h
    obtain ⟨rs', -, rfl⟩ := Option.map_eq_some_iff.1 hR
    exact ⟨rs, [(.stacked, rs')], hr, by simp, by simpa using (Option.some.inj h).symm⟩

/-! ## The reps' streams -/

/-- **`repRounds`' rounds before compiled Ligerito's**: the zerocheck's, the lincheck's and each claim's ring switch, the
only rounds `kLog` and `claims` reach. -/
def repHead (m kLog claims : ℕ) : Array SchedRound :=
  #[(6, #[], 1, false), (m - 13, #[], 0, false), (1, #[], 0, true)] ++ Array.replicate (m - 5) (1, #[], 0, true) ++
    #[(1, #[], 0, false)] ++ Array.replicate (kLog - 6) (1, #[], 0, true) ++ #[(1, #[], 0, true)] ++
    Array.replicate claims (7, #[], 0, true) ++ #[(claims, #[], 0, false)]

/-- Compiled Ligerito's rounds at `sch` before its levels' loop. -/
def ligStart (sch : Flock.Schedule) : Array SchedRound :=
  #[(sch.n, #[], 1, true), (1, #[], 0, true)] ++ Array.replicate sch.k0 (2, #[], 0, true) ++
    #[(sch.n - sch.k0 + 1, #[2 ^ (sch.levels.getD (0 + 1) default).cap], 0, true), (1, #[], 0, true),
      (sch.n - sch.k0 + 1, #[], 0, false), (1, #[], 0, true), ((sch.levels.getD 0 default).q, #[], 0, false),
      (bitLength ((sch.levels.getD 0 default).q - 1), #[], 0, false), (1, #[], 0, true)]

/-- The loop step `pre ++ ·` puts before a step's output. -/
def preStep {α : Type} (pre : Array α) : ForInStep (Array α × ℕ) → ForInStep (Array α × ℕ)
  | .done p => .done (pre ++ p.1, p.2)
  | .yield p => .yield (pre ++ p.1, p.2)

/-- **A loop that only appends** to its array: its output is its input followed by what it appends, which is the same
from every input array. -/
theorem forIn_append_sim {α : Type} {F : ℕ → Array α × ℕ → Option (ForInStep (Array α × ℕ))}
    (hF : ∀ i pre s, F i (pre, s) = (F i (#[], s)).map (preStep pre)) :
    ∀ (L : List ℕ) (pre : Array α) (s : ℕ) (v : Array α × ℕ), forIn L (pre, s) F = some v →
      ∃ tl, v.1 = pre ++ tl ∧ ∀ pre', forIn L (pre', s) F = some (pre' ++ tl, v.2)
  | [], pre, s, v, h => by
    rw [List.forIn_nil] at h
    cases h
    exact ⟨#[], Array.append_empty.symm, fun pre' => by rw [Array.append_empty]; rfl⟩
  | i :: L, pre, s, v, h => by
    rw [List.forIn_cons] at h
    obtain ⟨y, hy, h⟩ := obind_some h
    rw [hF] at hy
    obtain ⟨z, hz, rfl⟩ := Option.map_eq_some_iff.1 hy
    cases z with
    | done p =>
      cases h
      exact ⟨p.1, rfl, fun pre' => by rw [List.forIn_cons, hF, hz]; rfl⟩
    | yield p =>
      obtain ⟨tl', hv, hall⟩ := forIn_append_sim hF L (pre ++ p.1) p.2 v h
      refine ⟨p.1 ++ tl', by rw [hv, Array.append_assoc], fun pre' => ?_⟩
      rw [List.forIn_cons, hF, hz, ← Array.append_assoc]
      exact hall (pre' ++ p.1)

/-- **`repRounds` is its head and a tail no `kLog` or `claims` reaches.** -/
theorem repRounds_tail {m kLog claims : ℕ} {out : Array SchedRound} (h : repRounds m kLog claims = some out) :
    ∃ T, out = repHead m kLog claims ++ T ∧
      ∀ kLog' claims', repRounds m kLog' claims' = some (repHead m kLog' claims' ++ T) := by
  unfold repRounds at h
  obtain ⟨sch, hf, h⟩ := obind_some h
  dsimp only at h
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at h
  obtain ⟨st, hst, h⟩ := obind_some h
  cases h
  refine (forIn_append_sim ?_ _ _ _ _ hst).elim fun tl ⟨hY, hall⟩ => ?_
  · intro i pre s
    dsimp only
    split
    · rfl
    · split <;> simp [preStep, Array.append_assoc]
  · refine ⟨ligStart sch ++ tl, hY.trans ?_, fun kLog' claims' => ?_⟩
    · simp only [repHead, ligStart, Array.append_assoc]
    · unfold repRounds
      have e : ∀ {β : Type} (f : Flock.Schedule → Option β), (some sch >>= f) = f sch := fun _ => rfl
      rw [hf, e]
      dsimp only
      rw [Std.Legacy.Range.forIn_eq_forIn_range', hall]
      simp only [repHead, ligStart, Array.append_assoc]
      rfl

/-- **`repRounds` is at `fast100`'s tables**, `22 ≤ m < 36`. -/
theorem repRounds_m {m kLog claims : ℕ} {out : Array SchedRound} (h : repRounds m kLog claims = some out) :
    22 ≤ m ∧ m < 36 := by
  unfold repRounds at h
  obtain ⟨sch, hf, -⟩ := obind_some h
  refine ⟨(fast100_facts hf).1, Decidable.by_contra fun hm => ?_⟩
  have hb : (m < 22 || m > 35) = true := by
    simp only [Bool.or_eq_true, decide_eq_true_eq]
    omega
  have h0 : Flock.fast100 m = none := by
    unfold Flock.fast100
    simp [hb]
  rw [h0] at hf
  cases hf

/-- **The forms of a rep's coins** at `m`, `kLog` and `claims`: the zerocheck's, the lincheck's, the opening's and
compiled Ligerito's at `fast100`'s schedule. -/
def repForms (m kLog claims : ℕ) : List (Option ℕ) :=
  zcForms m ++ (lcForms kLog ++ (opForms claims ++ ligForms (schedAt m)))

theorem repHead_sizes {m : ℕ} (hm : 6 ≤ m) (kLog claims : ℕ) :
    (repHead m kLog claims).toList.map (·.1) = (zcForms m ++ (lcForms kLog ++ opForms claims)).map formSize := by
  obtain ⟨m, rfl⟩ : ∃ k, m = k + 6 := ⟨m - 6, by omega⟩
  simp only [repHead, zcForms, show m + 6 - 5 = m + 1 by omega, show m + 6 - 6 = m by omega]
  simp [lcForms, opForms, formSize, List.map_replicate, List.replicate_succ']

/-- **`repRounds`' rounds draw `repForms`' coins**: the head's at every `kLog` and `claims` (`repHead_sizes`), and the
tail's, which neither reaches (`repRounds_tail`), by evaluating `repRounds` at each `m` below 36 at `kLog = 6` and one
claim. -/
theorem repRounds_sizes {m kLog claims : ℕ} {out : Array SchedRound} (h : repRounds m kLog claims = some out) :
    out.toList.map (·.1) = (repForms m kLog claims).map formSize := by
  obtain ⟨hm22, hm⟩ := repRounds_m h
  obtain ⟨T, rfl, hall⟩ := repRounds_tail h
  have key : ∀ m' < 36, (repRounds m' 6 1).all
      (fun o => o.toList.map (·.1) == (repForms m' 6 1).map formSize) := by
    decide +kernel
  have h1 := key m hm
  rw [hall 6 1, Option.all_some, beq_iff_eq] at h1
  have hT : T.toList.map (·.1) = (ligForms (schedAt m)).map formSize := by
    rw [repForms, ← List.append_assoc (lcForms 6), ← List.append_assoc, List.map_append, Array.toList_append,
      List.map_append, repHead_sizes (by omega)] at h1
    exact List.append_cancel_left h1
  rw [Array.toList_append, List.map_append, repHead_sizes (by omega), hT, repForms, ← List.map_append]
  simp only [List.append_assoc]

/-- The coin sizes of a stream's rounds other than a sparse phase's: one per round that draws coins. -/
def nzSizes (l : List SchedRound) : List ℕ := l.flatMap fun q => if q.1 = 0 then [] else [q.1]

theorem nzSizes_append (l l' : List SchedRound) : nzSizes (l ++ l') = nzSizes l ++ nzSizes l' := List.flatMap_append

theorem nzSizes_pos : ∀ l : List SchedRound, (∀ x ∈ l, 0 < x.1) → nzSizes l = l.map (·.1)
  | [], _ => rfl
  | x :: l, h => by
    have hx : x.1 ≠ 0 := Nat.pos_iff_ne_zero.1 (h x List.mem_cons_self)
    simp only [nzSizes, List.flatMap_cons, List.map_cons, hx, ↓reduceIte, List.singleton_append]
    exact congrArg _ (nzSizes_pos l fun y hy => h y (List.mem_cons_of_mem _ hy))

theorem holoSizes_rep_nz (x : Fin 2) (rs : Array SchedRound) : holoSizes [(.rep x, rs)] = nzSizes rs.toList := by
  simp [holoSizes, nzSizes]

/-- **z_partial's round draws nothing**: `holoRounds`' coin sizes are `repRounds`'. -/
theorem nzSizes_holoRounds {m kLog claims : ℕ} {rs : Array SchedRound} (h : holoRounds m kLog claims = some rs) :
    ∃ out, repRounds m kLog claims = some out ∧ nzSizes rs.toList = nzSizes out.toList := by
  unfold holoRounds at h
  obtain ⟨out, hout, h⟩ := obind_some h
  refine ⟨out, hout, ?_⟩
  dsimp only at h
  split at h
  · cases h
  · rename_i hk
    cases h
    have hk' : out[zpRound m kLog]? = some ((1, #[], 0, true) : SchedRound) := by simpa using hk
    have hlt : zpRound m kLog < out.toList.length := by
      by_cases hn : zpRound m kLog < out.size
      · simpa using hn
      · rw [Array.getElem?_eq_none (by omega)] at hk'
        cases hk'
    have ho : out.toList[zpRound m kLog] = (1, #[], 0, true) := by
      rw [← Array.getElem?_toList, List.getElem?_eq_getElem hlt] at hk'
      exact Option.some.inj hk'
    have e : out.toList =
        out.toList.take (zpRound m kLog) ++ (1, #[], 0, true) :: out.toList.drop (zpRound m kLog + 1) := by
      rw [← ho, List.getElem_cons_drop, List.take_append_drop]
    rw [e]
    simp only [Array.toList_append, Array.toList_extract, List.extract, nzSizes_append]
    simp [nzSizes, List.take_of_length_le]

/-- **A rep's stream draws the model's coins**: `repForms`' at a claim. -/
theorem holoSizes_rep (x : Fin 2) {m kLog claims : ℕ} {rs : Array SchedRound} (hc : 0 < claims)
    (h : holoRounds m kLog claims = some rs) :
    holoSizes [(.rep x, rs)] = (repForms m kLog claims).map formSize := by
  obtain ⟨out, hout, e⟩ := nzSizes_holoRounds h
  rw [holoSizes_rep_nz, e, nzSizes_pos _ fun y hy => repRounds_coins hc hout y (Array.mem_def.2 hy),
    repRounds_sizes hout]

/-- **Both reps' streams draw the model's coins**: `repForms`' twice, at a claim. -/
theorem holoSizes_reps {m kLog claims : ℕ} {rs : Array SchedRound} (hc : 0 < claims)
    (h : holoRounds m kLog claims = some rs) :
    holoSizes [(.rep 0, rs), (.rep 1, rs)] = (repForms m kLog claims ++ repForms m kLog claims).map formSize := by
  rw [show [(HoloStream.rep 0, rs), (HoloStream.rep 1, rs)] = [(.rep 0, rs)] ++ [(.rep 1, rs)] from rfl,
    holoSizes_append, holoSizes_rep 0 hc h, holoSizes_rep 1 hc h, List.map_append]

end FlockC3.Compose
