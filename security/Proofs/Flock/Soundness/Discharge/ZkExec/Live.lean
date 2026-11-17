module

public import Proofs.Flock.Soundness.Discharge.ZkExec.Event
public import Proofs.Flock.Soundness.Discharge.Exec.Live
import Mathlib.Combinatorics.Matroid.Init

/-! The live `--zk` audit, its coins fresh each round, and the compiled `--zk` table it is simulated in; the coin
tree's step to Flock ZK's coins (`Discharge/ZkCoins`) is open here. -/

@[expose] public section

namespace FlockSoundness.Discharge.ZkExec

open Game Audit Exec Refine Flock

local notation "𝔽" => FlockLevel3.GF128

variable {n : ℕ} {Reg : Type} {mPts : ℕ} {P Cp Op Out : Finset (Fin n) → Reg → Type}
  (G : ∀ S R, (Fin 2 → Fin mPts → 𝔽) → Fin 2 → Game (P S R))
  (out : ∀ S R, Cp S R → P S R × P S R → Op S R → Out S R) (acc : ∀ S R, Out S R → Bool)

/-- **The compiled `--zk` table's game at a sample**: the level-0 cap, the link points, the two reps concurrently, the
openings, and the table's verdict. -/
def sessionZC (S : Finset (Fin n)) (R : Reg) : Game Bool := (modelTable (G S R) (out S R)).map (acc S R)

variable [∀ S R, Nonempty (Op S R)] (I : Inputs) (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
  (opn : ∀ S R, Array ByteArray → Op S R) (dg : ByteArray → ByteArray)
  (dec : Finset (Fin n) → Reg → ℕ → ByteArray → List Msg) (one : Finset (Fin n) → Reg → ℕ → Bool)
  (fin : Finset (Fin n) → Reg → Fin 2 → Array ByteArray → List Msg)
  (capOf : ∀ S R, ByteArray → Cp S R) (N : ℕ)

def DecodesZK : Prop :=
  ∀ S R, Decodes (out S R) (fun o => AcceptsZK I (wr S R o)) (fun x => acc S R x = true) (opn S R) (dec S R)
    (one S R) (fin S R) (capOf S R) (G S R)

variable {L : SamplingDesign n}

noncomputable def simAuditZK (hinh : ∀ S R pts r, AllInh (G S R pts r))
    (hdec : DecodesZK G out acc I wr opn dec one fin capOf)
    (τ : Strategy (auditLive L Reg (liveOf dg mPts N wr))) : Strategy (audit L Reg (sessionZC G out acc)) :=
  ⟨τ.1, fun ω => Strategy.map (fun ok => (ok, L.draw ω)) _ (Strategy.map _ _
    ((simTable (out (L.draw ω) τ.1) (fun o => AcceptsZK I (wr (L.draw ω) τ.1 o))
      (fun x => acc (L.draw ω) τ.1 x = true) (opn (L.draw ω) τ.1) dg (dec (L.draw ω) τ.1) (one (L.draw ω) τ.1)
      (fin (L.draw ω) τ.1) (capOf (L.draw ω) τ.1) N (G (L.draw ω) τ.1) (hinh (L.draw ω) τ.1)
      (hdec (L.draw ω) τ.1)).strategy
      (Strategy.ofMap (wr (L.draw ω) τ.1) _
        (Strategy.ofMap (fun t => (τ.1, L.draw ω, t)) _ (τ.2 ω)))))⟩

end FlockSoundness.Discharge.ZkExec
