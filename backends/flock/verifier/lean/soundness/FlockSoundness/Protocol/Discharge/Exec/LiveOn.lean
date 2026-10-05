import FlockSoundness.Protocol.Discharge.Exec.Live

/-!
# R11 for one-table sessions, with the decoder right at the law's draws only

`DecodesAll` asks `Decodes` at every draw `S` and registration. The audit plays only the law's draws `L.draw ω`, so
`DecodesOn L` asks it there and nowhere else, and `execRefines_on` and `Prog.flock_headline_live_on` take it in place
of `DecodesAll`, with the same bound. `DecodesAll` implies it (`decodesOn_of_all`), and at it the simulated prover is
the same (`simAudit_eq_on`). At a draw the law never makes, the table `T S R` and the decoder are unconstrained; at a
draw it makes, `T S R` must be the table flock-verify sets up there, `m_pts` included (`Pts.lean`).
-/

namespace FlockSoundness.Discharge.Exec

open Game Model Accounting Refine Audit Audit.Partition
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

variable {D Sl : Type} [Nonempty D] [Inhabited D] [Inhabited Sl] {mPts n : ℕ} {Reg : Type}

variable (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D H)
  (T : Finset (Fin n) → Reg → TabSpec mPts) (I : Inputs) (wr : Finset (Fin n) → Reg → LiveOut mPts → Transcript)
  (opn : Finset (Fin n) → Reg → Array ByteArray → Opens 𝔽 D × Salts Sl) (dg : ByteArray → ByteArray)
  (dec : Finset (Fin n) → Reg → ℕ → ByteArray → List Msg) (one : Finset (Fin n) → Reg → ℕ → Bool)
  (fin : Finset (Fin n) → Reg → Fin 2 → Array ByteArray → List Msg)
  (capOf : Finset (Fin n) → Reg → ByteArray → (Fin 1 → Cap D)) (N : ℕ)

/-- **The decoder is right on the sessions flock-verify accepts, at the law's draws** (R11b): `DecodesAll` at the
draws `L.draw ω` only. -/
def DecodesOn (L : Law n) : Prop :=
  ∀ ω R, Decodes (fun (caps : Fin 1 → Cap D) pend os => tableOutL H E Lf (caps 0) pend os)
    (fun o => Accepts I (wr (L.draw ω) R o)) (·.accepted = true) (opn (L.draw ω) R) (dec (L.draw ω) R)
    (one (L.draw ω) R) (fin (L.draw ω) R) (capOf (L.draw ω) R) (repsOf (D := D) execArith (T (L.draw ω) R))

variable {L : Law n}

/-- **The simulated audit strategy**, from the decoder at the law's draws: the live prover's registration, and at each
draw the simulated session strategy against its live session. -/
noncomputable def simAuditOn (hdec : DecodesOn H E Lf T I wr opn dec one fin capOf L)
    (τ : Strategy (auditLive L Reg (liveOf dg mPts N wr))) :
    Strategy (audit L Reg (batchedSessionL H E Lf fun S R => [T S R])) :=
  ⟨τ.1, fun ω => Strategy.map (fun ok => (ok, L.draw ω)) _ (Strategy.map _ _
    (simSessL execArith H E Lf (T (L.draw ω) τ.1) (fun o => Accepts I (wr (L.draw ω) τ.1 o)) (opn (L.draw ω) τ.1) dg
      (dec (L.draw ω) τ.1) (one (L.draw ω) τ.1) (fin (L.draw ω) τ.1) (capOf (L.draw ω) τ.1) N (hdec ω τ.1)
      (Strategy.ofMap (wr (L.draw ω) τ.1) _
        (Strategy.ofMap (fun t => (τ.1, L.draw ω, t)) _ (τ.2 ω)))))⟩

/-! ## The headline at the executable, for one-table sessions -/

variable {N' : ℕ} (p : FlockSoundness.Prog N' n) (outs : Finset (Fin N'))

end FlockSoundness.Discharge.Exec
