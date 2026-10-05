import FlockSoundness.Protocol.Discharge.ZkView.Defs
import Flock.ZkProof
import FlockSoundness.Protocol.Discharge.ZkBind.At

/-!
# One `--zk` sampled session: the pieces the soundness half reads

* `Gz`, `outz`, `accz`: the compiled `--zk` table at each draw (one table) as `modelTable`'s pieces, so that
  `ZkExec.sessionZC` at them is `ZkLink.batchedSessionZC` at the one-table plan (`sessionZC_planOf`).
* `DecodesOnZK`, `simZ`, `execRefinesZK_on`: `DecodesZK`, `simAuditZK` and `execRefinesZK_one` at the law's draws only
  (as `Tables.DecodesOnJ` is on the public path), the simulated strategy carried into the compiled batched session.
-/

namespace FlockSoundness.Discharge.Composed

open Game Audit Exec Refine Flock Model ZkSession ZkLink ZkExec DecodeZL
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

variable {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] {mPts n : ℕ} {Reg : Type}

/-- The one-table plan at the draw's tables. -/
abbrev planOf (T : Finset (Fin n) → Reg → TabZK 𝔽 mPts) : Finset (Fin n) → Reg → List (TabZK 𝔽 mPts) :=
  fun S R => [T S R]

/-- The compiled reps at each draw. -/
noncomputable def Gz (A : Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : Finset (Fin n) → Reg → TabZK 𝔽 mPts) :
    ∀ (_ : Finset (Fin n)) (_ : Reg), (Fin 2 → Fin mPts → 𝔽) → Fin 2 → Game (PendZ 𝔽 D Sl) :=
  fun S R => repZCAt A Hs (T S R)

/-- The compiled table's output at each draw. -/
noncomputable def outz (Hs : Hashes 𝔽 D Sl) (T : Finset (Fin n) → Reg → TabZK 𝔽 mPts) :
    ∀ (_ : Finset (Fin n)) (_ : Reg), (Fin 1 → Cap D) → PendZ 𝔽 D Sl × PendZ 𝔽 D Sl → OpensZ 𝔽 D Sl →
      List (OutZC 𝔽 D Sl) :=
  fun S R caps pend os => [outZC Hs (T S R) (caps 0) pend os]

variable [Inhabited D] [Inhabited Sl]

open Classical in
/-- The compiled table's verdict at each draw. -/
noncomputable def accz (T : Finset (Fin n) → Reg → TabZK 𝔽 mPts) :
    ∀ (_ : Finset (Fin n)) (_ : Reg), List (OutZC 𝔽 D Sl) → Bool :=
  fun S R outs => decide (acceptedZC [T S R] outs)

omit [Fintype K] in
theorem sessionZC_planOf (A : Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : Finset (Fin n) → Reg → TabZK 𝔽 mPts) :
    ZkExec.sessionZC (Gz A Hs T) (outz Hs T) (accz T) = batchedSessionZC A Hs (planOf T) :=
  funext fun S => funext fun R => batchedSessionZC_one A Hs T S R

section On

variable (A : Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : Finset (Fin n) → Reg → TabZK 𝔽 mPts) (I : Inputs)
  (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
  (opn : ∀ (_ : Finset (Fin n)) (_ : Reg), Array ByteArray → OpensZ 𝔽 D Sl) (dg : ByteArray → ByteArray)
  (dec : Finset (Fin n) → Reg → ℕ → ByteArray → List Msg) (one : Finset (Fin n) → Reg → ℕ → Bool)
  (fin : Finset (Fin n) → Reg → Fin 2 → Array ByteArray → List Msg)
  (capOf : ∀ (_ : Finset (Fin n)) (_ : Reg), ByteArray → Fin 1 → Cap D) (N : ℕ) (L : Law n)

/-- **The decoder is right at the law's draws** (`DecodesZK` at `L.draw ω` only): on every live outcome flock-verify
`--zk` accepts at a draw the law makes, each rep's game runs on the tape decoded from its stream and proof file, and
the compiled table's verdict on those runs holds. -/
def DecodesOnZK : Prop :=
  ∀ ω R, Decodes (outz Hs T (L.draw ω) R) (fun o => AcceptsZK I (wr (L.draw ω) R o))
    (fun x => accz T (L.draw ω) R x = true) (opn (L.draw ω) R) (dec (L.draw ω) R) (one (L.draw ω) R)
    (fin (L.draw ω) R) (capOf (L.draw ω) R) (Gz A Hs T (L.draw ω) R)

variable {L}

/-- `simAuditZK` from the decoder at the law's draws. -/
noncomputable def simAuditZKOn (hdec : DecodesOnZK A Hs T I wr opn dec one fin capOf L)
    (τ : Strategy (auditLive L Reg (liveOf dg mPts N wr))) :
    Strategy (audit L Reg (ZkExec.sessionZC (Gz A Hs T) (outz Hs T) (accz T))) :=
  ⟨τ.1, fun ω => Strategy.map (fun ok => (ok, L.draw ω)) _ (Strategy.map _ _
    ((simTable (outz Hs T (L.draw ω) τ.1) (fun o => AcceptsZK I (wr (L.draw ω) τ.1 o))
      (fun x => accz T (L.draw ω) τ.1 x = true) (opn (L.draw ω) τ.1) dg (dec (L.draw ω) τ.1) (one (L.draw ω) τ.1)
      (fin (L.draw ω) τ.1) (capOf (L.draw ω) τ.1) N (Gz A Hs T (L.draw ω) τ.1)
      (fun pts r => allInh_repZCAt A Hs (T (L.draw ω) τ.1) pts r) (hdec ω τ.1)).strategy
      (Strategy.ofMap (wr (L.draw ω) τ.1) _
        (Strategy.ofMap (fun t => (τ.1, L.draw ω, t)) _ (τ.2 ω)))))⟩

/-- **The simulated audit strategy in the compiled batched session**: `simAuditZKOn`'s, carried along
`sessionZC_planOf`. -/
noncomputable def simZ (hdec : DecodesOnZK A Hs T I wr opn dec one fin capOf L)
    (τ : Strategy (auditLive L Reg (liveOf dg mPts N wr))) : Strategy (audit L Reg (batchedSessionZC A Hs (planOf T))) :=
  cast (congrArg (fun g => Strategy (audit L Reg g)) (sessionZC_planOf A Hs T))
    (simAuditZKOn A Hs T I wr opn dg dec one fin capOf N hdec τ)

omit [Fintype K] in
/-- **The decoder at the law's draws from `RunsOnZC`** (`ZkExec.decodes_of_runsOnZC`, the coin step `true`). -/
theorem decodesOnZK_of_runsOnZC (h : RunsOnZC A Hs T I wr opn dec fin capOf L) :
    DecodesOnZK A Hs T I wr opn dec (fun _ _ _ => true) fin capOf L :=
  fun ω R => decodes_of_runsOnZC A Hs T I wr opn dec fin capOf h ω R

end On

end FlockSoundness.Discharge.Composed
