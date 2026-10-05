import FlockSoundness.Refine.LiveSim
import FlockSoundness.Protocol.Refine.LiveCompiled

/-!
# The transfer for the compiled table

`live_le` (`LiveSim`) needs every message the model's reps receive to have an inhabited type, so that where no accepted
live run goes on the model can still play. `allInh_repC` proves it for the compiled reps, from `AllInh` of each game
they are built from. `live_le_tableC` is then the transfer for `tableC` itself: its one hypothesis is `Decodes`.
-/

namespace FlockSoundness.Game

variable {α β : Type}

end FlockSoundness.Game

namespace FlockSoundness.Model

open FlockSoundness.Game

variable {F K D : Type}

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
