module

public import Proofs.Flock.Claims.Witnesses.SamplingGenericZK
public import Proofs.Flock.SampledGeneric.HidingDefs

/-!
# The sampled audit's zero knowledge from a hiding trace commitment (stated in the game model)

Proof material, listed in no lock. The sentence these statements make precise: if the commitment to the trace is
statistically hiding at `ε_hide`, and the proof system is statistical zero knowledge against malicious verifiers at
`ε_b` a session, then the verifier's whole view of the audit, the commitment included, is simulated from the statements
alone, at `ε_hide` plus `ε_b` a session.

* `Hm96TraceHiding`: hm96's commitment to a trace of `N` leaves, one uniform salt a leaf, hides at `2·N·δ₁` from
  `Claims.Hm96Hiding` at `δ₁` a leaf. `Hm96ExeHiding`: the executable's commit strings at the pinned key hide at
  `2·N·δ₁`, with `δ₁` the hiding gap there (`hm96Gap`), and no hypothesis.
* `GenericSampledZKAt`: `GenericSampledZK` with a session's error that may read the witness and the verifier
  (`Zk.ZKAt`): the audit's view is close to the simulator's at `ε_hide` plus the session's error averaged over the
  prover's commitment coins.
* `SpotFlockZK`: that at Flock's deployed masked session (`Flock.Guarantees.MaskedZK`'s record) and the executable's
  hm96 commit strings.
* `GenericSessionsZKAt`: sessions in turn against one verifier that carries its state are zero knowledge at the sum of
  the sessions' errors, each averaged over the honest views before it (`Zk.seq`, `Zk.seqErr`).
* `GenericSampledSessionsZK`, `SpotFlockSessionsZK`: the audit with a session per sampled statement, generic and at
  Flock. `GenericSampledSessionsZKSum`: the generic one at a session error that reads no verifier, where the sessions'
  errors add up.
-/

@[expose] public section

namespace FlockSoundness.SampledGeneric

open scoped ENNReal

open Claims.ZKMal.Masked FlockSoundness.Discharge in
/-- **Flock's deployed masked session** (`Flock.Guarantees.MaskedZK`'s record), at the rep rounds' coin counts `nr` and
the repetition count `t`. -/
noncomputable def flockZk (nr : Claims.Msg → ℕ → ℕ) (t : ℕ) :
    Claims.Flock.ZkDeploy FlockDeploy.Coin (List FlockLevel3.GF128) Claims.Flock.ZkTrace :=
  Claims.ZKMal.zkDeployed _root_.Flock.Readable.relation nr
    (tableData PE (RoundPlan.stop (czExe PE) (encStop (encExe PE)) (finExe PE)) UE) t

/-- The masked session's rounds at `x`: the masked verifier's and the `Hello`. -/
noncomputable def flockRounds (nr : Claims.Msg → ℕ → ℕ) (t : ℕ) : Claims.Msg → ℕ := fun x => (flockZk nr t).V.k x + 1

/-- **Flock's masked session as a `Zk`**, at SHA-512: the reference prover's transcript against a malicious verifier of
the session's rounds, and the record's simulator's. -/
noncomputable def flockSession (nr : Claims.Msg → ℕ → ℕ) (t : ℕ) (Θ : Type) :
    Zk Claims.Msg Claims.Flock.Wit (MalVerifier (flockRounds nr t) Θ) (List (Claims.Msg × Claims.Out)) :=
  zkOfMal FlockDeploy.sha512 (flockRounds nr t) (flockZk nr t).prover (flockZk nr t).S Θ

/-- **The masked session's error** at `(x, w)` against `v`: `ε_zk(x) + 4·t·3^t·Pr[Fork]`, with `ε_zk` `MaskedZK`'s
(`8·n_L(x)·δ₁ + 8·t·n_L(x)·δ₁ + 2·3^{-t}` at a statement in scope, else `1`) and `Fork` the real game's collision
finder at SHA-512: two runs of the reference prover at `w` against `v`'s `V*` on one nonce. -/
noncomputable def flockErr (nr : Claims.Msg → ℕ → ℕ) (t : ℕ) {Θ : Type} (x : Claims.Msg) (w : Claims.Flock.Wit)
    (v : MalVerifier (flockRounds nr t) Θ) : ℝ≥0∞ :=
  malErr FlockDeploy.sha512 (flockZk nr t).prover (Claims.Witness.SamplingGeneric.zkEps t)
    (Claims.Witness.SamplingGeneric.zkγ t) x w v

end FlockSoundness.SampledGeneric

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.SampledGeneric
open scoped ENNReal

/-- **hm96's trace commitment hides.** For a trace of `N = |ι|` leaves committed under one uniform salt a leaf
(`hm96Hider`), whatever `Hf` publishes of each leaf's commit string: if a leaf hides at `δ₁` (`Claims.Hm96Hiding`), the
commitments to any two traces are `2·N·δ₁`-close against every post-processing (`Hider.Hiding`). -/
def Hm96TraceHiding : Prop :=
  ∀ (ι Sa Bs C D : Type) [Fintype ι] [DecidableEq ι] [Fintype Sa] [Nonempty Sa] [AddCommGroup Bs] [Fintype Bs]
    (My : Sa → Bs) (c : Sa → C) (δ₁ : ℝ≥0∞) (Hf : Bs × C → D),
    Claims.Hm96Hiding My c δ₁ → (hm96Hider ι My c Hf).Hiding (2 * Fintype.card ι * δ₁)

/-- **The executable's hm96 commit strings hide**, with no hypothesis: for `N = |ι|` inner digests committed by
`Flock.Hm96.Default512.commitString` at the pinned key under one uniform 192-byte salt each (`hm96Exe`), the commit
strings of any two traces are `2·N·δ₁`-close against every post-processing, with `δ₁` the hiding gap at the pinned key
(`hm96Gap`). -/
def Hm96ExeHiding : Prop :=
  ∀ (ι : Type) [Fintype ι] [DecidableEq ι], (hm96Exe ι).Hiding (2 * Fintype.card ι * hm96Gap)

/-- **The sampled audit's zero knowledge, at a session's error that reads the witness and the verifier.** For a proof
system whose honest prover's view is `ε x w v`-close to its simulator's at every statement and witness in its relation
and every verifier (`Zk.ZKAt`), and a commitment that is `ε_hide`-hiding (`Hider.Hiding`): against every verifier `v`,
which samples (`samp`) after seeing the commitment, the audit's view at any trace `Xt` whose openings are witnesses of
every sample's statement is close to the simulator's at any fixed trace `X₀` (`auditSim`), which reads nothing of `Xt`,
at `ε_hide` plus the session's error averaged over the prover's commitment coins:
$$\varepsilon_{\rm hide}+\mathbb E_r\,\varepsilon(x_r,w_r,v).$$ -/
def GenericSampledZKAt : Prop :=
  ∀ (n : ℕ) (Tr Rnd Com Wit VS View : Type) [Fintype Rnd] [Nonempty Rnd] (H : Hider Tr Rnd Com)
    (Zs : Zk (Com × Finset (Fin n)) Wit VS View) (Rel : Com × Finset (Fin n) → Wit → Prop)
    (ε : Com × Finset (Fin n) → Wit → VS → ℝ≥0∞) (εh : ℝ≥0∞) (samp : VS → Com → Finset (Fin n))
    (wit : Tr → Rnd → Finset (Fin n) → Wit),
    Zs.ZKAt Rel ε → H.Hiding εh →
    ∀ Xt : Tr, (∀ r S, Rel (H.commit Xt r, S) (wit Xt r S)) →
    ∀ (X₀ : Tr) (v : VS), Close (auditView H Zs samp wit Xt v) (auditSim H Zs samp X₀ v)
      (εh + avg fun r => ε (H.commit Xt r, samp v (H.commit Xt r)) (wit Xt r (samp v (H.commit Xt r))) v)

/-- **The sampled audit over Flock's masked session is zero knowledge.** The prover commits to a trace `Xt` of
`N = |ι|` inner digests by the executable's hm96 commit strings under one uniform salt each (`hm96Exe`); the verifier
`v` samples after seeing them (`samp`) and from the commit strings and its sample chooses a malicious verifier of the
masked session's rounds (`Zk.after`); and Flock's reference prover proves the sample's statement `stmt (c, S)` from
the trace's openings (`wit`) in one masked session (`flockSession`). If every such statement is in the masked
record's relation with those openings, the verifier's view, the commit strings, the sample and the session's
transcript, is close to the simulator's at any fixed trace `X₀`, which reads nothing of `Xt`, at
$$2N\delta_1+\mathbb E_r\bigl[\varepsilon_{zk}(x_r)+4t\,3^t\Pr[\mathrm{Fork}(P(w_r),V^*)]\bigr],$$
with `δ₁` hm96's hiding gap at the pinned key (`hm96Gap`), `ε_zk` and the fork factor `MaskedZK`'s (`flockErr`), and
the average over the prover's salts. -/
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

/-- **Sessions in turn are zero knowledge at the sum of their errors.** For a proof system that is zero knowledge at
`ε` (`Zk.ZKAt`): its sessions in turn at a list of statements, each with a witness in the relation, against one
verifier that carries what it saw from session to session (`Zk.seq`), are zero knowledge at the sum over the sessions
of session `j`'s error, averaged over the honest views of the sessions before it (`Zk.seqErr`). The claims core's
form, per public outcome at a counted simulator, is `Flock.Guarantees.GenericSessionsZK`
(`Claims.RecGeneric.Sessions`). -/
def GenericSessionsZKAt : Prop :=
  ∀ (Stmt Wit VS View : Type) (Zs : Zk Stmt Wit VS View) (Rel : Stmt → Wit → Prop) (ε : Stmt → Wit → VS → ℝ≥0∞),
    Zs.ZKAt Rel ε → Zs.seq.ZKAt (List.Forall₂ Rel) fun xs ws v => Zs.seqErr ε (xs.zip ws) (v xs)

/-- **The sampled audit's zero knowledge with a session per sampled statement.** As `GenericSampledZKAt`, with the
sample's statements a list `stmts (c, S)`, each proved in its own session, in turn, from the trace's openings
(`wits`), against one verifier that samples after seeing the commitment and carries what it saw, the commitment
included, from session to session: the audit's view is close to the simulator's at any fixed trace `X₀` at
$$\varepsilon_{\rm hide}+\mathbb E_r\sum_j\mathbb E\,\varepsilon(x_{r,j},w_{r,j},v_j),$$
session `j`'s error averaged over the honest views of the sessions before it. -/
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

/-- **The sampled audit's zero knowledge with a session per sampled statement, at the sum of the sessions' errors.**
`GenericSampledSessionsZK` for a proof system whose session error `ε_b x w` reads the statement and the witness and
no verifier: the audit's view is close to the simulator's at any fixed trace `X₀` at `ε_hide` plus the sum of the
sample's sessions' errors, averaged over the prover's commitment coins,
$$\varepsilon_{\rm hide}+\mathbb E_r\sum_j\varepsilon_b(x_{r,j},w_{r,j}).$$ -/
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

/-- **The sampled audit over Flock's masked sessions, a session per sampled statement, is zero knowledge.** As
`SpotFlockZK`, with the sample's statements a list `stmts (c, S)`, each proved in its own masked session, in turn,
against one verifier that carries what it saw, the commit strings included, from session to session: the view is close
to the simulator's at any fixed trace `X₀` at
$$2N\delta_1+\mathbb E_r\sum_j\mathbb E\bigl[\varepsilon_{zk}(x_{r,j})+4t\,3^t\Pr[\mathrm{Fork}(P(w_{r,j}),V^*_j)]\bigr],$$
session `j`'s error averaged over the honest transcripts of the sessions before it. -/
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
