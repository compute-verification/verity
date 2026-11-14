module

public import Proofs.Flock.Recursive.CompileRuns
public import Proofs.Flock.Recursive.ForkInner
public import Proofs.Flock.Recursive.Replay
public import Proofs.Flock.Recursive.Target
public import Proofs.Flock.Soundness.Assumptions.Recursive
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
import Proofs.Flock.Recursive.ForkRuns
import Proofs.Flock.Soundness.Accounting.Numbers

/-!
# Recursion: the compile gap

`RecursiveSound` at Flock's compiled inner protocol (`flockInnerCL`), with the inner protocol's soundness in fork form
and the interactive table's error as its inner term. Every declaration is proved.

* `ClashCL`, `clashPickCL`: two runs of the compiled table that clash (either run self-clashes, `SelfClashL`, or the two
  open a position of the level-0 cap or of a rep's recorded cap to different rows), and the inner fork's output: on two
  inner transcripts whose steps play the compiled table to outcomes that clash, the pair of the first of the outcomes'
  candidates (`clashCandsCL`) whose test passes, a collision of `H` (`clashCandsCL_spec`), else nothing.
* `flock_inner_sound_compiled` (**gap 1**, the compile lemma): `InnerSoundForkOn` at `flockInnerCL` with `clashPickCL`,
  at `tableError`, the interactive table's error that `flock_inner_sound` gives `flockInner`.
  `flock_inner_sound_compiled_plain` is its `InnerSoundFork`, and `flock_inner_sound_compiled_fast100` its `2^-205`.
* `decodedV`, `InnerForkRec`: the inner transcript V*'s layers decode at an outcome of V*'s outer proof, and a fork of the
  recursive prover's own runs after a round's commitment whose two decoded transcripts `pick` reads as a collision.
* `zk_sessions_recursive_fork_inner` (**gap 2**): `zk_sessions_recursive_fork` with V*'s bridge as
  `Flock.Assumptions.VBridge` and the inner protocol's soundness in fork form (`InnerSoundForkOn`), the inner fork
  (`InnerForkRec`) a fourth beside the registrant, round and session forks.
* `recursive_sound_compiled`: the target, from the two gaps: `RecursiveSound`'s conclusion at `I := flockInnerCL`, with
  V*'s bridge as VBridge G (`vbridge_compiled`) concludes it, the inner fork, and the inner term the table's error;
  `recursive_sound_compiled_fast100` at `2^-205`.
-/

@[expose] public section

namespace FlockSoundness.Discharge.Recursion

section Compiled

open Game Model Accounting Exec StrictCR
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {D Sl : Type}

/-- **Two runs of the compiled table clash**: either run's outcome self-clashes (`SelfClashL`), or the two open a
position of the first run's level-0 cap, or of a rep's level-`ℓ` cap the first run records, to different rows. -/
def ClashCL (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (sch : Schedule)
    (o o' : OutCL F D Sl) : Prop :=
  SelfClashL H E Lf sch o ∨ SelfClashL H E Lf sch o' ∨
    (∃ p, Rewinding.ConflictAt (op₀L H E Lf sch o.cap₀) (o, o') p) ∨
    ∃ i < 2, ∃ ℓ, 1 ≤ ℓ ∧ ∃ cap, (if i = 0 then o.pend.1 else o.pend.2).caps[ℓ - 1]? = some cap ∧
      ∃ p, Rewinding.ConflictAt (opRL H E Lf sch i ℓ cap) (o, o') p

/-- **Two clashing runs' candidates**: each run's self-clash candidates (`selfCandsL`), then, at each position of the
first run's level-0 cap and of each cap a rep of the first run records, the test that the two runs open it to different
rows and the pair of their openings there (`op₀LPair`, `opRLPair`). -/
noncomputable def clashCandsCL (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (sch : Schedule) (o o' : OutCL F D Sl) : List (Prop × (List UInt8 × List UInt8)) :=
  selfCandsL H E Lf sch o ++ (selfCandsL H E Lf sch o' ++
    ((List.finRange (2 ^ (lvOf sch 0).logLen)).map (fun p =>
        (Rewinding.ConflictAt (op₀L H E Lf sch o.cap₀) (o, o') p, op₀LPair H E Lf sch o.cap₀ (o, o') p)) ++
      [0, 1].flatMap fun i => (if i = 0 then o.pend.1 else o.pend.2).caps.zipIdx.flatMap fun ci =>
        (List.finRange (2 ^ (lvOf sch (ci.2 + 1)).logLen)).map fun p =>
          (Rewinding.ConflictAt (opRL H E Lf sch i (ci.2 + 1) ci.1) (o, o') p,
            opRLPair H E Lf sch i (ci.2 + 1) ci.1 (o, o') p)))

omit [Fintype F] [DecidableEq F] in
/-- **A clash is a collision**: a candidate whose test passes is one. -/
theorem clashCandsCL_spec {H : List UInt8 → D} {E : Merkle.Enc (List F) D} {Lf : Merkle.Leaf (List F) Sl D H}
    {sch : Schedule} {o o' : OutCL F D Sl} {c : Prop × (List UInt8 × List UInt8)}
    (hc : c ∈ clashCandsCL H E Lf sch o o') (h : c.1) : Hash.Collision H c.2.1 c.2.2 := by
  rcases List.mem_append.1 hc with hc | hc
  · exact selfCandsL_spec hc h
  rcases List.mem_append.1 hc with hc | hc
  · exact selfCandsL_spec hc h
  rcases List.mem_append.1 hc with hc | hc
  · obtain ⟨p, -, rfl⟩ := List.mem_map.1 hc
    have hw : Rewinding.ConflictAt (op₀L H E Lf sch o.cap₀) (o, o') p := h
    exact op₀L_conflict_collision hw
  · obtain ⟨i, -, hc⟩ := List.mem_flatMap.1 hc
    obtain ⟨ci, -, hc⟩ := List.mem_flatMap.1 hc
    obtain ⟨p, -, rfl⟩ := List.mem_map.1 hc
    have hw : Rewinding.ConflictAt (opRL H E Lf sch i (ci.2 + 1) ci.1) (o, o') p := h
    exact opRL_conflict_collision hw

omit [Fintype F] [DecidableEq F] in
/-- Two runs that clash have a candidate whose test passes. -/
theorem ClashCL.cand {H : List UInt8 → D} {E : Merkle.Enc (List F) D} {Lf : Merkle.Leaf (List F) Sl D H}
    {sch : Schedule} {o o' : OutCL F D Sl} (h : ClashCL H E Lf sch o o') :
    ∃ c ∈ clashCandsCL H E Lf sch o o', c.1 := by
  rcases h with h | h | ⟨p, h⟩ | ⟨i, hi, ℓ, hℓ, cap, hcap, p, h⟩
  · obtain ⟨c, hc, hp⟩ := h.cand
    exact ⟨c, List.mem_append_left _ hc, hp⟩
  · obtain ⟨c, hc, hp⟩ := h.cand
    exact ⟨c, List.mem_append_right _ (List.mem_append_left _ hc), hp⟩
  · exact ⟨_, List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _
      (List.mem_map.2 ⟨p, List.mem_finRange p, rfl⟩))), h⟩
  · obtain ⟨k, rfl⟩ : ∃ k, ℓ = k + 1 := ⟨ℓ - 1, (Nat.sub_add_cancel hℓ).symm⟩
    have hi' : i ∈ [0, 1] := by
      have : i = 0 ∨ i = 1 := by omega
      rcases this with rfl | rfl <;> simp
    have hk : (cap, k) ∈ (if i = 0 then o.pend.1 else o.pend.2).caps.zipIdx :=
      List.mem_zipIdx_iff_getElem?.2 (by simpa using hcap)
    exact ⟨_, List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _
      (List.mem_flatMap.2 ⟨i, hi', List.mem_flatMap.2 ⟨(cap, k), hk,
        List.mem_map.2 ⟨p, List.mem_finRange p, rfl⟩⟩⟩))), h⟩

variable {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]

/-- **The compiled table's outcome on an inner transcript**: the outcome the transcript's steps on the schedule play
`Model.tableCL` to on the hidden statement `M.toStatement regs`, when they play it to one. It is the replay of those
steps (`Game.playT`): the finder computes the outcome from its own transcript, and none is chosen. Outside proofs, two
tests under it are decided classically: the replay's, that a step's type is the node's, and the table's opening check
(`OpensOKL`, in `Model.tableCL`). -/
noncomputable def outCL (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (cls : HClass) (regs : List (Region cls.kLog (cls.m - cls.kLog)))
    (sch : Schedule) (M : Hidden cls) (t : List (List (SMsg T) × ((k : Fin r) → CT k))) : Option (OutCL F D Sl) :=
  playT (tableCL A H E Lf (M.toStatement regs) sch cls.hm cls.ptLocal cls.mPts) (schedSteps T CT 0 t)

/-- **The inner fork's output at the compiled table**: on two inner transcripts whose steps play the compiled table to
outcomes `o` and `o'`, the pair of the first of the two outcomes' candidates (`clashCandsCL`) whose test passes, a
collision of `H` when the outcomes clash (`ClashCL`); else nothing. The picker searches the outcomes' own lists and
positions. The outcomes are `outCL`'s. -/
noncomputable def clashPickCL (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (cls : HClass) (regs : List (Region cls.kLog (cls.m - cls.kLog)))
    (sch : Schedule) (M : Hidden cls) (t t' : List (List (SMsg T) × ((k : Fin r) → CT k))) : Pair :=
  (outCL T CT A H E Lf cls regs sch M t).bind fun o => (outCL T CT A H E Lf cls regs sch M t').bind fun o' =>
    collOf (clashCandsCL H E Lf sch o o') (fun c => c.1) fun c => c.2

/-- **Gap 1, the compile lemma: Flock's compiled inner protocol is sound in fork form, at the interactive table's
error.** On every hidden template of the class whose statement no witness satisfies, for every `G` and every prover of
`flockInnerCL`: two continuations of the prover's own strategy after some round's message, both of whose transcripts
satisfy `G`, play the compiled table to outcomes that clash (a collision of `H`), or the compiled table accepts a
transcript satisfying `G` with probability at most `tableError`, which `flock_inner_sound` bounds `flockInner` by.

Its proof is `compiled_runs_sound` (`CompileRuns`): two runs satisfying `G` whose outcomes clash, read as a fork of the
prover's own strategy (`Inner.fork_of_outcomes`), or the bound. -/
theorem flock_inner_sound_compiled (A : Arith F K) (hA : A.Correct) [Inhabited D] [Inhabited Sl]
    (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (cls : HClass)
    (regs : List (Region cls.kLog (cls.m - cls.kLog))) (sch : Schedule) (hsch : fast100 cls.m = some sch)
    (hlay : ∀ M : Hidden cls, (M.toStatement regs).LinkLayout cls.ptLocal cls.mPts) :
    InnerSoundForkOn H (flockInnerCL T CT A H E Lf cls regs sch) (clashPickCL T CT A H E Lf cls regs sch)
      (ENNReal.ofReal (tableError sch ⟨cls.kLog, regs.length, cls.mPts⟩)) := by
  intro M hno G s
  change Hidden cls at M
  have hout : ∀ {t o}, PlaysT (tableCL A H E Lf (M.toStatement regs) sch cls.hm cls.ptLocal cls.mPts)
      (schedSteps T CT 0 t) o → outCL T CT A H E Lf cls regs sch M t = some o := by
    intro t o ho
    unfold outCL
    exact playT_of_playsT ho
  rcases compiled_runs_sound T CT A hA H E Lf cls regs sch hsch hlay (ClashCL H E Lf sch)
      (fun _ _ h => Or.inl h) (fun _ _ p h => Or.inr (Or.inr (Or.inl ⟨p, h⟩)))
      (fun _ _ i hi ℓ hℓ cap hcap p h => Or.inr (Or.inr (Or.inr ⟨i, hi, ℓ, hℓ, cap, hcap, p, h⟩))) M hno G s with
    ⟨t, t', o, o', ht, ht', hG, hG', ho, ho', hc⟩ | hb
  · left
    have hr : 0 < (flockInnerCL T CT A H E Lf cls regs sch).r := by
      rw [← Inner.length_of_outcome (flockInnerCL T CT A H E Lf cls regs sch) s ht]
      have hne := PlaysT.ne_nil_of_send
        (k := tableCmpL A H E Lf (M.toStatement regs) sch cls.hm cls.ptLocal cls.mPts) ho
      cases t with
      | nil => exact (hne rfl).elim
      | cons _ _ => exact Nat.succ_pos _
    refine Inner.fork_of_outcomes H _ hr _ s ht ht' ?_
    have e : pickOn G (clashPickCL T CT A H E Lf cls regs sch M) t t' =
        clashPickCL T CT A H E Lf cls regs sch M t t' := pickOn_of hG hG'
    have ep : clashPickCL T CT A H E Lf cls regs sch M t t' =
        collOf (clashCandsCL H E Lf sch o o') (fun c => c.1) fun c => c.2 := by
      unfold clashPickCL
      rw [hout ho, hout ho', Option.bind_some, Option.bind_some]
    exact (congrArg (StrictCR.IsColl H) (e.trans ep)).mpr
      (collOf_isColl H (fun _ hc hp => clashCandsCL_spec hc hp) hc.cand)
  · exact Or.inr hb

/-- **The compile lemma as `InnerSoundFork`**: `flock_inner_sound_compiled` at `G := True`. -/
theorem flock_inner_sound_compiled_plain (A : Arith F K) (hA : A.Correct) [Inhabited D] [Inhabited Sl]
    (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (cls : HClass)
    (regs : List (Region cls.kLog (cls.m - cls.kLog))) (sch : Schedule) (hsch : fast100 cls.m = some sch)
    (hlay : ∀ M : Hidden cls, (M.toStatement regs).LinkLayout cls.ptLocal cls.mPts) :
    InnerSoundFork H (flockInnerCL T CT A H E Lf cls regs sch) (clashPickCL T CT A H E Lf cls regs sch)
      (ENNReal.ofReal (tableError sch ⟨cls.kLog, regs.length, cls.mPts⟩)) :=
  (flock_inner_sound_compiled T CT A hA H E Lf cls regs sch hsch hlay).plain

/-- **The compile lemma with the numbers**: at most `2^-205` for `22 ≤ m ≤ 33` and a shape in range. -/
theorem flock_inner_sound_compiled_fast100 (A : Arith F K) (hA : A.Correct) [Inhabited D] [Inhabited Sl]
    (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (cls : HClass)
    (regs : List (Region cls.kLog (cls.m - cls.kLog))) (sch : Schedule) (hsch : fast100 cls.m = some sch)
    (hm1 : 22 ≤ cls.m) (hm2 : cls.m ≤ 33) (hshape : (⟨cls.kLog, regs.length, cls.mPts⟩ : Accounting.Stmt).InRange)
    (hlay : ∀ M : Hidden cls, (M.toStatement regs).LinkLayout cls.ptLocal cls.mPts) :
    InnerSoundForkOn H (flockInnerCL T CT A H E Lf cls regs sch) (clashPickCL T CT A H E Lf cls regs sch)
      (ENNReal.ofReal ((2 : ℝ) ^ (-(205 : ℝ)))) := fun M hno G s =>
  (flock_inner_sound_compiled T CT A hA H E Lf cls regs sch hsch hlay M hno G s).imp id fun h =>
    h.trans (ENNReal.ofReal_le_ofReal (tableError_le_of_le_33 hm1 hm2 hsch _ hshape))

end Compiled

section Chain

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR
  DecodeZL Composed ZkReg
open scoped ENNReal

variable {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type} {n : ℕ} {L : Law n} {Reg : Type} {I : Inner} {Cm : Type}

/-- **The inner transcript V*'s layers decode** at an outcome `o` of V*'s outer proof at the history `(R, ω, t)`: the
messages `wmsg` reads from the sessions' layers, beside the rounds' coins. -/
noncomputable def decodedV (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg) (R : Reg) (ω : L.Ω)
    (t : List (Cm × I.Coin)) (o : Reg₂ × VRec Vs Reg' dg N) : List (I.Msg × I.Coin) :=
  (wmsg (layers (zs R ω t o.1) dg N k Rw o.2)).zip (t.map Prod.snd)

/-- **An inner fork of the recursive prover's own runs**: at some sample `ω` and round `i < r`, two continuations of `σ`
after its round-`i` commitment, each run through V*'s outer proof at the leaf it reaches, whose decoded inner transcripts
(`decodedV`) `pick` reads, on the hidden statement at `ω`, as a collision of `H` at an outcome the fork reaches. -/
def InnerForkRec (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N))
    (x : Reg → L.Ω → I.Stmt) (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg) {D : Type} (H : List UInt8 → D)
    (pick : I.Stmt → List (I.Msg × I.Coin) → List (I.Msg × I.Coin) → StrictCR.Pair) : Prop :=
  ∃ ω, ∃ i < I.r,
    (forkG (pick (x σ.1 ω)) i I.r (fun t => (outerV (zs σ.1 ω t) dg N).map fun b => (σ.1, ω, t, b))
      (fun t s => StrictCR.oneRun (outerV (zs σ.1 ω t) dg N)
        (Strategy.ofMap (fun b => (σ.1, ω, t, b)) (outerV (zs σ.1 ω t) dg N) s)
        (decodedV zs dg N k Rw wmsg σ.1 ω t))
      (σ.2 ω)).Finds H

/-- **Gap 2: the recursive audit is sound with the inner protocol's soundness in fork form.**
`zk_sessions_recursive_fork` with V*'s bridge as `Flock.Assumptions.VBridge` (the inner verifier reads each round's
message whole) and `InnerSoundForkOn` in place of `InnerSound`: a registrant fork (`regFinder`), a round fork, a session
fork, an inner fork of `σ`'s own runs (`InnerForkRec`), or the same bound.

Its proof is `zk_sessions_recursive_fork_on` with the inner term read at the leaves of `σ`
(`innerAtLeavesV_of_forkOn`, `ForkInner`): a fork of the reference strategy on two decoded transcripts is one of
`σ`'s (`InnerForkRec`). -/
theorem zk_sessions_recursive_fork_inner
    (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N : ℕ)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ R ω t R₂ j S' R' i,
      rateZ (ZkHidden.planZHJ (Reg := Reg') (zs R ω t R₂ j).dj (zs R ω t R₂ j).y₀ S' R') i k ≤ ρ)
    (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N))
    (x : Reg → L.Ω → I.Stmt) (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg)
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → I.Msg)
    (hbr : Flock.Assumptions.VBridge zs dg N k Rw x wmsg Sk cmt msg)
    {D : Type} (H : List UInt8 → D) (pick : I.Stmt → List (I.Msg × I.Coin) → List (I.Msg × I.Coin) → StrictCR.Pair)
    {εin : ℝ≥0∞} (hin : InnerSoundForkOn H I pick εin) :
    (regFinder zs dg N σ).Finds H512 ∨ RoundFork zs dg N k Rw σ cmt ∨ SessionFork zs dg N k Rw M σ ∨
      InnerForkRec zs dg N k Rw σ x wmsg H pick ∨
      prob (fun o => AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧ ¬ I.holds (x o.1 o.2.1))
          (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ ≤
        expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundCR dg N (Vs j).law.model k Rw M ρ τ) s) +
          (I.r : ℝ≥0∞) * expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).slackCR dg N k Rw M ρ τ) s) + εin := by
  by_cases hf : InnerForkRec zs dg N k Rw σ x wmsg H pick
  · exact Or.inr (Or.inr (Or.inr (Or.inl hf)))
  rcases zk_sessions_recursive_fork_on zs dg N hk hRw hM hρ hr σ x wmsg hbr.1 (fun _ _ a b => a = b)
      (fun ref ω t u => u = relabel (ref ω) t)
      (fun ref ω t l ht h => zip_eq_relabel (ref ω) t l fun k hk =>
        option_eq_of_rel (R := fun a b => a = b) (fun _ _ e => e) (h k (by omega))) Sk cmt msg hbr.2.1 hbr.2.2.1
      hbr.2.2.2 (fun ref => InnerAtLeaves.toZ (innerAtLeavesV_of_forkOn zs dg N k Rw σ x wmsg H pick hin hf ref))
      with h | h | h | h
  · exact Or.inl h
  · exact Or.inr (Or.inl h)
  · exact Or.inr (Or.inr (Or.inl h))
  · exact Or.inr (Or.inr (Or.inr (Or.inr h)))

end Chain

section Target

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR
  DecodeZL Composed ZkReg
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {D Sl : Type}
variable {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
variable {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type} {n : ℕ} {L : Law n} {Reg : Type} {Cm : Type}

/-- **The recursive audit is sound at Flock's compiled inner protocol.** `RecursiveSound` at `I := flockInnerCL`
(the compiled table on the class's hidden statements, the schedule `(T, CT)` with live coins), V*'s bridge as VBridge G
(`vbridge_compiled`) concludes it: a registrant fork, a round fork, a session fork, an inner fork of `σ`'s own runs whose decoded
transcripts play the compiled table to outcomes that clash (`InnerForkRec` at `clashPickCL`), or the audit accepts a
false hidden statement with probability at most V*'s sessions' expected statistical errors, `r` times their expected
stage slack, and the interactive table's error. From `flock_inner_sound_compiled` (gap 1) and
`zk_sessions_recursive_fork_inner` (gap 2). -/
theorem recursive_sound_compiled (A : Arith F K) (hA : A.Correct) [Inhabited D] [Inhabited Sl] (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (cls : HClass)
    (regs : List (Region cls.kLog (cls.m - cls.kLog))) (sch : Accounting.Schedule)
    (hsch : Accounting.fast100 cls.m = some sch)
    (hlay : ∀ M : Hidden cls, (M.toStatement regs).LinkLayout cls.ptLocal cls.mPts)
    (zs : Reg → L.Ω → List (Cm × (flockInnerCL T CT A H E Lf cls regs sch).Coin) → Reg₂ → (j : Fin S) →
      (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N : ℕ)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ R ω t R₂ j S' R' i,
      rateZ (ZkHidden.planZHJ (Reg := Reg') (zs R ω t R₂ j).dj (zs R ω t R₂ j).y₀ S' R') i k ≤ ρ)
    (σ : Strategy (recGame L Reg (flockInnerCL T CT A H E Lf cls regs sch) Cm fun R ω t => outerV (zs R ω t) dg N))
    (x : Reg → L.Ω → Hidden cls)
    (wmsg : ((j : Fin S) → (Vs j).Bits) → List (flockInnerCL T CT A H E Lf cls regs sch).Msg)
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) →
      (flockInnerCL T CT A H E Lf cls regs sch).Msg)
    (hbr : Flock.Assumptions.VBridge zs dg N k Rw x wmsg Sk cmt msg) :
    (regFinder zs dg N σ).Finds H512 ∨ RoundFork zs dg N k Rw σ cmt ∨ SessionFork zs dg N k Rw M σ ∨
      InnerForkRec zs dg N k Rw σ x wmsg H (clashPickCL T CT A H E Lf cls regs sch) ∨
      prob (fun o => AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧
            ¬ (flockInnerCL T CT A H E Lf cls regs sch).holds (x o.1 o.2.1))
          (recGame L Reg (flockInnerCL T CT A H E Lf cls regs sch) Cm fun R ω t => outerV (zs R ω t) dg N) σ ≤
        expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundCR dg N (Vs j).law.model k Rw M ρ τ) s) +
          ((flockInnerCL T CT A H E Lf cls regs sch).r : ℝ≥0∞) * expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).slackCR dg N k Rw M ρ τ) s) +
          ENNReal.ofReal (Accounting.tableError sch ⟨cls.kLog, regs.length, cls.mPts⟩) :=
  zk_sessions_recursive_fork_inner zs dg N hk hRw hM hρ hr σ x wmsg Sk cmt msg hbr H
    (clashPickCL T CT A H E Lf cls regs sch) (flock_inner_sound_compiled T CT A hA H E Lf cls regs sch hsch hlay)

/-- **`recursive_sound_compiled` with the numbers**: the inner term at most `2^-205` for `22 ≤ m ≤ 33` and a shape in
range. -/
theorem recursive_sound_compiled_fast100 (A : Arith F K) (hA : A.Correct) [Inhabited D] [Inhabited Sl]
    (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (cls : HClass)
    (regs : List (Region cls.kLog (cls.m - cls.kLog))) (sch : Accounting.Schedule)
    (hsch : Accounting.fast100 cls.m = some sch) (hm1 : 22 ≤ cls.m) (hm2 : cls.m ≤ 33)
    (hshape : (⟨cls.kLog, regs.length, cls.mPts⟩ : Accounting.Stmt).InRange)
    (hlay : ∀ M : Hidden cls, (M.toStatement regs).LinkLayout cls.ptLocal cls.mPts)
    (zs : Reg → L.Ω → List (Cm × (flockInnerCL T CT A H E Lf cls regs sch).Coin) → Reg₂ → (j : Fin S) →
      (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N : ℕ)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ R ω t R₂ j S' R' i,
      rateZ (ZkHidden.planZHJ (Reg := Reg') (zs R ω t R₂ j).dj (zs R ω t R₂ j).y₀ S' R') i k ≤ ρ)
    (σ : Strategy (recGame L Reg (flockInnerCL T CT A H E Lf cls regs sch) Cm fun R ω t => outerV (zs R ω t) dg N))
    (x : Reg → L.Ω → Hidden cls)
    (wmsg : ((j : Fin S) → (Vs j).Bits) → List (flockInnerCL T CT A H E Lf cls regs sch).Msg)
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) →
      (flockInnerCL T CT A H E Lf cls regs sch).Msg)
    (hbr : Flock.Assumptions.VBridge zs dg N k Rw x wmsg Sk cmt msg) :
    (regFinder zs dg N σ).Finds H512 ∨ RoundFork zs dg N k Rw σ cmt ∨ SessionFork zs dg N k Rw M σ ∨
      InnerForkRec zs dg N k Rw σ x wmsg H (clashPickCL T CT A H E Lf cls regs sch) ∨
      prob (fun o => AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧
            ¬ (flockInnerCL T CT A H E Lf cls regs sch).holds (x o.1 o.2.1))
          (recGame L Reg (flockInnerCL T CT A H E Lf cls regs sch) Cm fun R ω t => outerV (zs R ω t) dg N) σ ≤
        expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundCR dg N (Vs j).law.model k Rw M ρ τ) s) +
          ((flockInnerCL T CT A H E Lf cls regs sch).r : ℝ≥0∞) * expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).slackCR dg N k Rw M ρ τ) s) +
          ENNReal.ofReal ((2 : ℝ) ^ (-(205 : ℝ))) :=
  (recursive_sound_compiled T CT A hA H E Lf cls regs sch hsch hlay zs dg N hk hRw hM hρ hr σ x wmsg Sk cmt msg
    hbr).imp id <| Or.imp id <| Or.imp id <| Or.imp id fun h =>
      h.trans (add_le_add le_rfl (ENNReal.ofReal_le_ofReal (Accounting.tableError_le_of_le_33 hm1 hm2 hsch _ hshape)))

end Target

end FlockSoundness.Discharge.Recursion
