module

public import Proofs.Flock.Claims.Witnesses.SamplingGenericZK
public import Proofs.Flock.SampledGeneric.HidingDefs

/-! The sampled audit's zero knowledge from a hiding trace commitment, stated in the game model; proof material, listed in no lock. -/

@[expose] public section

namespace FlockSoundness.SampledGeneric

open scoped ENNReal

open Claims.ZKMal.Masked FlockSoundness.Discharge in
noncomputable def flockZk (nr : Claims.Msg → ℕ → ℕ) (t : ℕ) :
    Claims.Flock.ZkDeploy FlockDeploy.Coin (List FlockLevel3.GF128) Claims.Flock.ZkTrace :=
  Claims.ZKMal.zkDeployed _root_.Flock.Readable.relation nr
    (tableData PE (RoundPlan.stop (czExe PE) (encStop (encExe PE)) (finExe PE)) UE) t

noncomputable def flockRounds (nr : Claims.Msg → ℕ → ℕ) (t : ℕ) : Claims.Msg → ℕ := fun x => (flockZk nr t).V.k x + 1

noncomputable def flockSession (nr : Claims.Msg → ℕ → ℕ) (t : ℕ) (Θ : Type) :
    Zk Claims.Msg Claims.Flock.Wit (MalVerifier (flockRounds nr t) Θ) (List (Claims.Msg × Claims.Out)) :=
  zkOfMal FlockDeploy.sha512 (flockRounds nr t) (flockZk nr t).prover (flockZk nr t).S Θ

noncomputable def flockErr (nr : Claims.Msg → ℕ → ℕ) (t : ℕ) {Θ : Type} (x : Claims.Msg) (w : Claims.Flock.Wit)
    (v : MalVerifier (flockRounds nr t) Θ) : ℝ≥0∞ :=
  malErr FlockDeploy.sha512 (flockZk nr t).prover (Claims.Witness.SamplingGeneric.zkEps t)
    (Claims.Witness.SamplingGeneric.zkγ t) x w v

end FlockSoundness.SampledGeneric

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.SampledGeneric
open scoped ENNReal

def Hm96TraceHiding : Prop :=
  ∀ (ι Sa Bs C D : Type) [Fintype ι] [DecidableEq ι] [Fintype Sa] [Nonempty Sa] [AddCommGroup Bs] [Fintype Bs]
    (My : Sa → Bs) (c : Sa → C) (δ₁ : ℝ≥0∞) (Hf : Bs × C → D),
    Claims.Hm96Hiding My c δ₁ → (hm96Hider ι My c Hf).Hiding (2 * Fintype.card ι * δ₁)

/-- Takes no hypothesis: `δ₁` is the hiding gap at the pinned key (`hm96Gap`). -/
def Hm96ExeHiding : Prop :=
  ∀ (ι : Type) [Fintype ι] [DecidableEq ι], (hm96Exe ι).Hiding (2 * Fintype.card ι * hm96Gap)

def GenericSampledZKAt : Prop :=
  ∀ (n : ℕ) (Tr Rnd Com Wit VS View : Type) [Fintype Rnd] [Nonempty Rnd] (H : Hider Tr Rnd Com)
    (Zs : Zk (Com × Finset (Fin n)) Wit VS View) (Rel : Com × Finset (Fin n) → Wit → Prop)
    (ε : Com × Finset (Fin n) → Wit → VS → ℝ≥0∞) (εh : ℝ≥0∞) (samp : VS → Com → Finset (Fin n))
    (wit : Tr → Rnd → Finset (Fin n) → Wit),
    Zs.ZKAt Rel ε → H.Hiding εh →
    ∀ Xt : Tr, (∀ r S, Rel (H.commit Xt r, S) (wit Xt r S)) →
    ∀ (X₀ : Tr) (v : VS), Close (auditView H Zs samp wit Xt v) (auditSim H Zs samp X₀ v)
      (εh + avg fun r => ε (H.commit Xt r, samp v (H.commit Xt r)) (wit Xt r (samp v (H.commit Xt r))) v)

def SpotFlockZK : Prop :=
  ∀ (nr : Claims.Msg → ℕ → ℕ) (t : ℕ), 0 < t →
  ∀ (ι : Type) [Fintype ι] [DecidableEq ι] (n : ℕ) (Θ : Type)
    (stmt : (ι → ByteArray) × Finset (Fin n) → Claims.Msg)
    (samp : ((ι → ByteArray) × Finset (Fin n) → MalVerifier (flockRounds nr t) Θ) →
      (ι → ByteArray) → Finset (Fin n))
    (wit : (ι → ByteArray) → (ι → Discharge.ZkView.Salt) → Finset (Fin n) → Claims.Flock.Wit)
    (Xt : ι → ByteArray),
    (∀ r S, Claims.Flock.R (flockZk nr t).parse (stmt ((hm96Exe ι).commit Xt r, S)) (wit Xt r S)) →
    ∀ (X₀ : ι → ByteArray) (v : (ι → ByteArray) × Finset (Fin n) → MalVerifier (flockRounds nr t) Θ),
      Close (auditView (hm96Exe ι) ((flockSession nr t Θ).after stmt) samp wit Xt v)
        (auditSim (hm96Exe ι) ((flockSession nr t Θ).after stmt) samp X₀ v)
        (2 * Fintype.card ι * hm96Gap +
          avg fun r => flockErr nr t (stmt ((hm96Exe ι).commit Xt r, samp v ((hm96Exe ι).commit Xt r)))
            (wit Xt r (samp v ((hm96Exe ι).commit Xt r)))
            (v ((hm96Exe ι).commit Xt r, samp v ((hm96Exe ι).commit Xt r))))

def GenericSessionsZKAt : Prop :=
  ∀ (Stmt Wit VS View : Type) (Zs : Zk Stmt Wit VS View) (Rel : Stmt → Wit → Prop) (ε : Stmt → Wit → VS → ℝ≥0∞),
    Zs.ZKAt Rel ε → Zs.seq.ZKAt (List.Forall₂ Rel) fun xs ws v => Zs.seqErr ε (xs.zip ws) (v xs)

def GenericSampledSessionsZK : Prop :=
  ∀ (n : ℕ) (Tr Rnd Com Stmt Wit VS View : Type) [Fintype Rnd] [Nonempty Rnd] (H : Hider Tr Rnd Com)
    (Zs : Zk Stmt Wit VS View) (Rel : Stmt → Wit → Prop) (ε : Stmt → Wit → VS → ℝ≥0∞) (εh : ℝ≥0∞)
    (stmts : Com × Finset (Fin n) → List Stmt)
    (samp : (Com × Finset (Fin n) → List Stmt → List View → VS) → Com → Finset (Fin n))
    (wits : Tr → Rnd → Finset (Fin n) → List Wit),
    Zs.ZKAt Rel ε → H.Hiding εh →
    ∀ Xt : Tr, (∀ r S, List.Forall₂ Rel (stmts (H.commit Xt r, S)) (wits Xt r S)) →
    ∀ (X₀ : Tr) (v : Com × Finset (Fin n) → List Stmt → List View → VS),
      Close (auditView H (Zs.seq.after stmts) samp wits Xt v) (auditSim H (Zs.seq.after stmts) samp X₀ v)
        (εh + avg fun r => Zs.seqErr ε
          ((stmts (H.commit Xt r, samp v (H.commit Xt r))).zip (wits Xt r (samp v (H.commit Xt r))))
          (v (H.commit Xt r, samp v (H.commit Xt r)) (stmts (H.commit Xt r, samp v (H.commit Xt r)))))

def GenericSampledSessionsZKSum : Prop :=
  ∀ (n : ℕ) (Tr Rnd Com Stmt Wit VS View : Type) [Fintype Rnd] [Nonempty Rnd] (H : Hider Tr Rnd Com)
    (Zs : Zk Stmt Wit VS View) (Rel : Stmt → Wit → Prop) (εb : Stmt → Wit → ℝ≥0∞) (εh : ℝ≥0∞)
    (stmts : Com × Finset (Fin n) → List Stmt)
    (samp : (Com × Finset (Fin n) → List Stmt → List View → VS) → Com → Finset (Fin n))
    (wits : Tr → Rnd → Finset (Fin n) → List Wit),
    (Zs.ZKAt Rel fun x w _ => εb x w) → H.Hiding εh →
    ∀ Xt : Tr, (∀ r S, List.Forall₂ Rel (stmts (H.commit Xt r, S)) (wits Xt r S)) →
    ∀ (X₀ : Tr) (v : Com × Finset (Fin n) → List Stmt → List View → VS),
      Close (auditView H (Zs.seq.after stmts) samp wits Xt v) (auditSim H (Zs.seq.after stmts) samp X₀ v)
        (εh + avg fun r =>
          (((stmts (H.commit Xt r, samp v (H.commit Xt r))).zip (wits Xt r (samp v (H.commit Xt r)))).map
            fun xw => εb xw.1 xw.2).sum)

def SpotFlockSessionsZK : Prop :=
  ∀ (nr : Claims.Msg → ℕ → ℕ) (t : ℕ), 0 < t →
  ∀ (ι : Type) [Fintype ι] [DecidableEq ι] (n : ℕ) (Θ : Type)
    (stmts : (ι → ByteArray) × Finset (Fin n) → List Claims.Msg)
    (samp : ((ι → ByteArray) × Finset (Fin n) → List Claims.Msg → List (List (Claims.Msg × Claims.Out)) →
        MalVerifier (flockRounds nr t) Θ) → (ι → ByteArray) → Finset (Fin n))
    (wits : (ι → ByteArray) → (ι → Discharge.ZkView.Salt) → Finset (Fin n) → List Claims.Flock.Wit)
    (Xt : ι → ByteArray),
    (∀ r S, List.Forall₂ (Claims.Flock.R (flockZk nr t).parse) (stmts ((hm96Exe ι).commit Xt r, S)) (wits Xt r S)) →
    ∀ (X₀ : ι → ByteArray)
      (v : (ι → ByteArray) × Finset (Fin n) → List Claims.Msg → List (List (Claims.Msg × Claims.Out)) →
        MalVerifier (flockRounds nr t) Θ),
      Close (auditView (hm96Exe ι) ((flockSession nr t Θ).seq.after stmts) samp wits Xt v)
        (auditSim (hm96Exe ι) ((flockSession nr t Θ).seq.after stmts) samp X₀ v)
        (2 * Fintype.card ι * hm96Gap +
          avg fun r => (flockSession nr t Θ).seqErr (fun x w v => flockErr nr t x w v)
            ((stmts ((hm96Exe ι).commit Xt r, samp v ((hm96Exe ι).commit Xt r))).zip
              (wits Xt r (samp v ((hm96Exe ι).commit Xt r))))
            (v ((hm96Exe ι).commit Xt r, samp v ((hm96Exe ι).commit Xt r))
              (stmts ((hm96Exe ι).commit Xt r, samp v ((hm96Exe ι).commit Xt r)))))

end Flock.Guarantees
