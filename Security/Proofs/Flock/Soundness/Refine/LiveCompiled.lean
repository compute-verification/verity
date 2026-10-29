import Proofs.Flock.Soundness.Refine.LiveSim

/-!
# The transfer for the compiled table

`live_le` (`LiveSim`) needs every message the model's reps receive to have an inhabited type, so that where no accepted
live run goes on the model can still play. `allInh_repC` proves it for the compiled reps, from `AllInh` of each game
they are built from. `live_le_tableC` is then the transfer for `tableC` itself: its one hypothesis is `Decodes`.
-/

namespace FlockSoundness.Game

variable {α β : Type}

theorem AllInh.pure' (a : α) : AllInh (pure a : Game α) := .ret a

theorem AllInh.recv' (M : Type) [h : Nonempty M] : AllInh (Game.recv M) := .send h fun _ => .ret _

theorem AllInh.draw' (C : Type) [Fintype C] [Nonempty C] : AllInh (Game.draw C) := .coin fun _ => .ret _

theorem AllInh.bind' {g : Game α} {f : α → Game β} (hg : AllInh g) (hf : ∀ a, AllInh (f a)) : AllInh (g >>= f) :=
  AllInh.bind hf hg

theorem AllInh.mapM_loop (f : α → Game β) (hf : ∀ a, AllInh (f a)) :
    ∀ (as : List α) (bs : List β), AllInh (List.mapM.loop f as bs)
  | [], _ => .ret _
  | a :: as, bs => AllInh.bind' (hf a) fun b => AllInh.mapM_loop f hf as (b :: bs)

theorem AllInh.mapM (f : α → Game β) (hf : ∀ a, AllInh (f a)) (as : List α) : AllInh (as.mapM f) :=
  AllInh.mapM_loop f hf as []

end FlockSoundness.Game

namespace FlockSoundness.Model

open FlockSoundness.Game

variable {F K D : Type}

theorem allInh_residualRounds [Field F] [Fintype F] : ∀ (n : ℕ) (req : Fin n → F) (claim : F), AllInh (residualRounds n req claim)
  | 0, _, _ => .ret _
  | n + 1, req, claim => by
      simp only [residualRounds]
      repeat' (first
        | exact AllInh.ret _
        | exact AllInh.pure' _
        | exact AllInh.recv' _
        | exact AllInh.draw' _
        | refine AllInh.bind' ?_ fun _ => ?_
        | intro _
        | split
        | dsimp only)
      all_goals exact allInh_residualRounds n _ _

theorem allInh_zerocheck [Field F] [Fintype F] [DecidableEq F] [Field K] (A : Arith F K) (m : ℕ) : AllInh (zerocheck A m) := by
  unfold zerocheck
  repeat' (first
    | exact AllInh.ret _
    | exact AllInh.pure' _
    | exact AllInh.recv' _
    | exact AllInh.draw' _
    | refine AllInh.bind' ?_ fun _ => ?_
    | intro _
    | split
    | dsimp only)
  all_goals exact allInh_residualRounds _ _ _

theorem allInh_lincheckRounds [Field F] [Fintype F] : ∀ (n : ℕ) (claim : F), AllInh (lincheckRounds n claim)
  | 0, _ => .ret _
  | n + 1, claim => by
      simp only [lincheckRounds]
      repeat' (first
        | exact AllInh.ret _
        | exact AllInh.pure' _
        | exact AllInh.recv' _
        | exact AllInh.draw' _
        | refine AllInh.bind' ?_ fun _ => ?_
        | intro _
        | split
        | dsimp only)
      all_goals exact allInh_lincheckRounds n _

theorem allInh_lincheck [Field F] [Fintype F] [DecidableEq F] [Field K] (A : Arith F K) (S : Statement) (zc : ZcOut F S.m) : AllInh (lincheck A S zc) := by
  unfold lincheck
  repeat' (first
    | exact AllInh.ret _
    | exact AllInh.pure' _
    | exact AllInh.recv' _
    | exact AllInh.draw' _
    | refine AllInh.bind' ?_ fun _ => ?_
    | intro _
    | split
    | dsimp only)
  all_goals exact allInh_lincheckRounds _ _

theorem allInh_opening [Field F] [Fintype F] [DecidableEq F] [Field K] (A : Arith F K) {m : ℕ} (hm : 7 ≤ m) (claims : List (Claim F m)) :
    AllInh (opening A hm claims) := by
  unfold opening
  refine AllInh.bind' (AllInh.mapM _ (fun _ => AllInh.bind' (AllInh.recv' _) fun _ =>
    AllInh.bind' (AllInh.draw' _) fun _ => .ret _) _) fun _ => ?_
  exact AllInh.bind' (AllInh.draw' _) fun _ => .ret _

theorem allInh_drawK [Field F] [Fintype F] [Field K] (A : Arith F K) : AllInh (drawK A) := by
  unfold drawK
  repeat' (first
    | exact AllInh.ret _
    | exact AllInh.pure' _
    | exact AllInh.recv' _
    | exact AllInh.draw' _
    | refine AllInh.bind' ?_ fun _ => ?_)

theorem allInh_foldRoundsC [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] (A : Arith F K) : ∀ (k : ℕ) (s : Opens F D → K) (q : Opens F D → Quad K),
    AllInh (foldRoundsC A k s q)
  | 0, _, _ => .ret _
  | k + 1, s, q => by
      simp only [foldRoundsC]
      refine AllInh.bind' (allInh_drawK A) fun _ => ?_
      repeat' (first
        | exact AllInh.ret _
        | exact AllInh.pure' _
        | exact AllInh.recv' _
        | exact AllInh.draw' _
        | refine AllInh.bind' ?_ fun _ => ?_
        | intro _
        | split
        | dsimp only)
      all_goals exact allInh_foldRoundsC A k _ _

theorem allInh_twoOodC [Field F] [Fintype F] [Field K] (A : Arith F K) (dim split : ℕ) (s : Opens F D → K) (q : Opens F D → Quad K) :
    AllInh (twoOodC A dim split s q) := by
  unfold twoOodC
  repeat' (first
    | exact AllInh.ret _
    | exact AllInh.pure' _
    | exact AllInh.recv' _
    | exact AllInh.draw' _
    | refine AllInh.bind' ?_ fun _ => ?_)

theorem allInh_queryAndBatchC [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] (A : Arith F K) (r : ℕ) (lv : Accounting.Level) (ch : List K) (start : ℕ) (st : LigStateC F K D)
    (withMessage : Bool) : AllInh (queryAndBatchC A r lv ch start st withMessage) := by
  unfold queryAndBatchC
  repeat' (first
    | exact AllInh.ret _
    | exact AllInh.pure' _
    | exact AllInh.recv' _
    | exact AllInh.draw' _
    | refine AllInh.bind' ?_ fun _ => ?_
    | intro _
    | split)

theorem allInh_recursiveRoundsC [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] [Nonempty D] (A : Arith F K) (r n : ℕ) (b : ℕ → F) (initial : List K) (ext : ℕ) :
    ∀ (lvs : List Accounting.Level) (st : LigStateC F K D), AllInh (recursiveRoundsC A r n b initial ext lvs st)
  | [], _ => .ret _
  | [lv], st => by
      simp only [recursiveRoundsC]
      refine AllInh.bind' (allInh_foldRoundsC A _ _ _) fun _ => ?_
      repeat' (first
        | exact AllInh.ret _
        | exact AllInh.pure' _
        | exact AllInh.recv' _
        | exact AllInh.draw' _
        | refine AllInh.bind' ?_ fun _ => ?_)
  | lv :: lv' :: rest, st => by
      simp only [recursiveRoundsC]
      refine AllInh.bind' (allInh_foldRoundsC A _ _ _) fun _ => ?_
      repeat' (first
        | exact AllInh.ret _
        | exact AllInh.pure' _
        | exact AllInh.recv' _
        | exact AllInh.draw' _
        | refine AllInh.bind' ?_ fun _ => ?_
        | intro _
        | split
        | dsimp only)
      all_goals first
        | exact allInh_twoOodC A _ _ _ _
        | exact allInh_queryAndBatchC A _ _ _ _ _ _
        | exact allInh_recursiveRoundsC A r n b initial ext _ _

theorem allInh_ligeritoC [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] [Nonempty D] (A : Arith F K) (r : ℕ) (sch : Accounting.Schedule) (T : F) (b : ℕ → F) :
    AllInh (ligeritoC (D := D) A r sch T b) := by
  unfold ligeritoC
  repeat' (first
    | exact AllInh.ret _
    | exact AllInh.pure' _
    | exact AllInh.recv' _
    | exact AllInh.draw' _
    | refine AllInh.bind' ?_ fun _ => ?_
    | intro _
    | split
    | dsimp only)
  all_goals first
    | exact allInh_foldRoundsC A _ _ _
    | exact allInh_twoOodC A _ _ _ _
    | exact allInh_queryAndBatchC A _ _ _ _ _ _
    | exact allInh_recursiveRoundsC A _ _ _ _ _ _ _

/-- **Every message a compiled rep receives has an inhabited type.** -/
theorem allInh_repC [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] [Nonempty D] (A : Arith F K) (r : ℕ) (S : Statement) (sch : Accounting.Schedule) (hm : 13 ≤ S.m)
    (extra : List (Claim F S.m)) : AllInh (repC (D := D) A r S sch hm extra) := by
  unfold repC
  refine AllInh.bind' (allInh_zerocheck A _) fun _ => ?_
  repeat' (first
    | exact AllInh.ret _
    | exact AllInh.pure' _
    | exact AllInh.recv' _
    | exact AllInh.draw' _
    | refine AllInh.bind' ?_ fun _ => ?_
    | intro _
    | split
    | dsimp only)
  all_goals first
    | exact allInh_residualRounds _ _ _
    | exact allInh_lincheckRounds _ _
    | exact AllInh.mapM_loop _ (fun _ => by repeat' (first | exact AllInh.ret _ | exact AllInh.pure' _ | exact AllInh.recv' _ | exact AllInh.draw' _ | refine AllInh.bind' ?_ fun _ => ?_ | intro _ | split | dsimp only)) _ _
    | exact allInh_foldRoundsC A _ _ _
    | exact allInh_recursiveRoundsC A _ _ _ _ _ _ _
    | exact allInh_ligeritoC (D := D) A _ _ _ _

end FlockSoundness.Model

namespace FlockSoundness.Refine

open FlockSoundness.Game

local notation "𝔽" => FlockLevel3.GF128

open Classical in
/-- `tableC`'s verdict, as `modelTable`'s output. -/
noncomputable def tableOut {D : Type} (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) (cap : Model.Cap D)
    (pend : Model.PendL 𝔽 D × Model.PendL 𝔽 D) (o : Model.Opens 𝔽 D) : Model.OutC 𝔽 D :=
  ⟨pend.1.verdict o && pend.2.verdict o &&
    decide (Model.OpensOK H E cap 0 pend.1 o ∧ Model.OpensOK H E cap 1 pend.2 o), cap, pend, o⟩

/-- **The transfer for the compiled table.** Against any live prover, the executable's verdict `pL` on the live
session holds with probability at most `tableC`'s acceptance against the simulated prover, whenever the decoder is
right on accepted sessions. -/
theorem live_le_tableC {K D : Type} [Field K] [Fintype K] [DecidableEq K] [Nonempty D] (A : Model.Arith 𝔽 K)
    (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) (S : Model.Statement) (sch : Accounting.Schedule)
    (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (pL : LiveOut mPts → Prop) (opn : Array ByteArray → Model.Opens 𝔽 D)
    (dg : ByteArray → ByteArray) (dec : ℕ → ByteArray → List Msg) (one : ℕ → Bool)
    (fin : Fin 2 → Array ByteArray → List Msg) (capOf : ByteArray → Model.Cap D) (N : ℕ)
    (hdec : Decodes (tableOut H E) pL (·.accepted = true) opn dec one fin capOf
      (fun pts (r : Fin 2) => Model.repC (D := D) A r.val S sch hm (Model.extraClaims S ptLocal mPts pts)))
    (Pr : Strategy (liveTable dg mPts N)) :
    prob pL (liveTable dg mPts N) Pr ≤
      prob (·.accepted = true) (Model.tableC A H E S sch hm ptLocal mPts)
        ((simTable (tableOut H E) pL (·.accepted = true) opn dg dec one fin capOf N _
          (fun _ _ => Model.allInh_repC A _ S sch hm _) hdec).strategy Pr) :=
  live_le (tableOut H E) pL (·.accepted = true) opn dg dec one fin capOf N _
    (fun _ _ => Model.allInh_repC A _ S sch hm _) hdec Pr

end FlockSoundness.Refine
