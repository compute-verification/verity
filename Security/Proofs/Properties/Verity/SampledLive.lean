import Proofs.Flock.Bridge.Public
import Proofs.Properties.Verity.SampledProofs

/-!
# Sampled proofs' session as C-Flock's live audit

`SampledSession.KS` at the session C-Flock runs, from `SampledKSLivePub`:

* `Flock.Guarantees.LiveAt n D hU kd K₀ desc`: `SampledKSLivePub`'s setting at one session (the inputs flock-verify
  reads, the registration, the draw's setup, the verifier's bytes and A3, the analysis' `k`, `R_w`, `M`, `ρ`, `t'` and
  the budgets), at the parsed circuit file `D` with `n` units, the draw size `kd`, `K₀` and the descriptor `desc`;
  `LiveAt.CR`, at a strategy, the two collision-resistance premises `SampledKSLivePub` reads at it; `LiveAt.ks`, its
  conclusion at that strategy.
* `Verity.SampledSession.ofLive`: the session whose circuit, units and constants at each file are the key program's at
  its parse, and whose game, acceptance, witness and error at each `θ` and `c` are `LiveAt`'s;
  `SampledSession.ks_ofLive`: its `KS`, where the premises hold at every strategy.
-/

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal
open Classical

/-- **`SampledKSLivePub`'s setting at one session**: at the parsed circuit file `D` (its unit built, `hU`) with `n`
units, the draw size `kd`, `K₀` and the descriptor `desc`, everything else `SampledKSLivePub` quantifies before its
collision-resistance premises, named assumption A3 included. -/
structure LiveAt (n : ℕ) (D : Flock.Desc) (hU : Built D.unroll.unit) (kd K₀ : ℕ) (desc : Claims.Msg) where
  Reg : Type
  I : Inputs
  ht : I.tags.typed = false
  hD : Flock.Desc.parse (zkTags I) I.circuitFile I.tables = .ok D
  pub : Public
  hpub : HmOut.loadPublic (zkTags I) D.unroll I.publicFile = .ok pub
  hpt : pub.tables = none
  hscope : Discharge.Layout.scopeOkH D.unroll = true
  hrows : Bridge.RowsOk D.unroll
  hdup : Flock.Registered.dupName D.unroll = none
  hroots : Bridge.PortRoots.rootsColl D.unroll pub n = none
  qr : Lean.Json
  hpop : Bridge.PopMatches desc qr n
  hkd : kd ≤ n
  hdiv : kd % nTab I = 0
  dj : Finset (Fin n) → Lean.Json
  S₀ : Finset (Fin n)
  d₀ : Lean.Json
  y₀ : DrawSetupZHJ (inputsJ I 0) (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀
  wr : (S : Finset (Fin n)) → (R : Reg) →
    LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length (mPtsOf D.unroll.g (kd / nTab I)) → Discharge.Exec.Transcript
  dg : ByteArray → ByteArray
  N : ℕ
  Ω : Type
  [fΩ : Fintype Ω]
  [nΩ : Nonempty Ω]
  Lb : Fin 1 → ℕ
  src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256
  hA3 : Assumptions.UniformRandomBytes Lb src
  hTags : TagsOkZHJ I
  hCust : Assumptions.Zk.RecordCustodyZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr kd
  k : ℕ
  Rw : ℕ
  M : ℕ
  hk : 1 ≤ k
  hRw : 1 ≤ Rw
  hM : 1 ≤ M
  ρ : ℝ
  hρ : ρ < 1
  hρk : rateM1 k ≤ ρ
  t' : ℝ
  ht' : 0 ≤ t'
  hK₀ : K₀ ≤ n
  qF : ℝ
  qS : ℝ

attribute [instance] LiveAt.fΩ LiveAt.nΩ

namespace LiveAt

variable {n : ℕ} {D : Flock.Desc} {hU : Built D.unroll.unit} {kd K₀ : ℕ} {desc : Claims.Msg}
  (A : LiveAt n D hU kd K₀ desc)

/-- The draw's law: `kd` of the `n` units, from the verifier's bytes. -/
noncomputable abbrev L : Law n := Law.execOS (n := n) A.src (.subset kd) n

/-- The live audit. -/
noncomputable abbrev G := auditLive A.L A.Reg (liveOfZJ A.dg (planZHJ A.dj A.y₀) A.N A.wr)

/-- The key program's partition into the `n` units. -/
noncomputable abbrev P := (keyProg D.unroll hU n).partition (Discharge.Layout.outsOf D.unroll hU n)

/-- The registration's binding. -/
noncomputable abbrev vb := (rsZHJ A.dj A.y₀ A.ht (Flock.Desc.parse_flat A.hD) A.hscope hU A.hpub A.hpt).bindingZ
  (rsZHJ_computes A.dj A.y₀ A.ht (Flock.Desc.parse_flat A.hD) A.hscope hU A.hpub A.hpt)

/-- The simulated prover of a strategy of the live audit. -/
noncomputable abbrev sim (τ : Strategy A.G) :=
  simZJ execArith hs512 (planZHJ A.dj A.y₀) A.I A.wr (opnAtZHJ (planZHJ A.dj A.y₀) A.I A.dj) A.dg
    (decAtZJ (planZHJ A.dj A.y₀)) (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) A.N
    (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ A.I A.wr _ _ _ _ _
      (runsZCJH_live A.I A.ht (Flock.Desc.parse_flat A.hD) (Discharge.Layout.scopeOkH_spec hU A.hscope).2.1 A.dj A.y₀
        A.wr _ A.hTags A.hCust.1
        fun _ => drawOkZKHJ_of_file A.I A.ht (Flock.Desc.parse_flat A.hD) kd n A.hdiv A.dj A.hCust.2 _)) τ

/-- The extracted values at a strategy. -/
noncomputable abbrev Xr (τ : Strategy A.G) :=
  XplurZC execArith hs512 A.P (planZHJ A.dj A.y₀) (tabZHJ A.dj A.y₀) A.vb (reg (A.sim τ)) (cont (A.sim τ)) A.k A.Rw

/-- The key program's circuit. -/
noncomputable abbrev Cc := (keyProg D.unroll hU n).circuit (Discharge.Layout.outsOf D.unroll hU n)

/-- The game's rows, read off the circuit file. -/
noncomputable abbrev rows : Fin (Cc (hU := hU) (n := n)).N → Option Bridge.RowAt := fun g =>
  (FlockSoundness.Discharge.HiddenExec.rowOfH D.unroll hU n (Discharge.Layout.outsOf D.unroll hU n) g).map fun r =>
    ⟨r.2.1.val, r.1.val, Discharge.Layout.portRowW D.unroll r.2.1, r.2.2.val⟩

/-- The registration's frame-v3 roots. -/
abbrev roots : List Claims.Msg := Bridge.rootsOf D.unroll A.pub n

/-- The deployed commitment. -/
noncomputable abbrev Cd := Bridge.commitDep (Cc (hU := hU) (n := n)) (rows (hU := hU) (n := n)) D.unroll n A.roots

/-- The circuit's constants. -/
noncomputable abbrev κ : Fin (Cc (hU := hU) (n := n)).N → Option Bool := fun g => Bridge.constsOf D.unroll hU n g

/-- **The collision-resistance premises at a strategy**: cr/sha-512 for the knowledge finders of each table the
prover's session runs, and ecr/sha-512 for its link finders, as `SampledKSLivePub` reads them. -/
def CR (τ : Strategy A.G) : Prop :=
  (∀ ω j, TableCRZ execArith hs512 (planZHJ A.dj A.y₀ (A.L.draw ω) (reg (A.sim τ))) j
    (zcStrat execArith hs512 (planZHJ A.dj A.y₀) (cont (A.sim τ) ω)) A.qF A.qS) ∧
  LinkCRZC execArith hs512 A.P (planZHJ A.dj A.y₀) (tabZHJ A.dj A.y₀) A.vb (reg (A.sim τ)) (cont (A.sim τ)) A.k A.Rw
    A.M A.t'

/-- **`SampledKSLivePub` at the session**: at a strategy where the premises hold, flock-verify accepts with the
extracted witness outside the sampling relation with probability at most `deltaPub`. -/
theorem ks (q : Claims.Sampling.Query) (τ : Strategy A.G) (hCR : A.CR τ) :
    prob (fun o => LiveAcceptsZK A.I o ∧
        ¬ Claims.Sampling.RelS Claims.Sampling.sha512Msg A.Cd
          (((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ)) (Bridge.stmtOf A.P (κ (hU := hU) (n := n)) kd desc q A.roots) (Bridge.partOf A.P)
          (Bridge.witDep (Cc (hU := hU) (n := n)) A.Cd Claims.Sampling.sha512Msg A.roots (A.Xr τ)))
      A.G τ ≤
      ENNReal.ofReal (Bridge.deltaPub n kd K₀ A.k A.qF A.qS A.t' A.M A.Rw (Bridge.ioSum A.P) A.ρ) :=
  Flock.SecurityProofs.SampledKSLivePub n A.Reg A.I A.ht D A.hD hU A.pub A.hpub A.hpt A.hscope A.hrows A.hdup A.hroots
    desc A.qr A.hpop kd A.hkd A.hdiv A.dj A.S₀ A.d₀ A.y₀ A.wr A.dg A.N A.Ω A.Lb A.src A.hA3 A.hTags A.hCust A.k A.Rw A.M
    A.hk A.hRw A.hM A.ρ A.hρ A.hρk A.t' A.ht' K₀ A.hK₀ τ A.qF A.qS q hCR.1 hCR.2

end LiveAt

end Flock.Guarantees

namespace Verity.SampledSession

open Flock.Guarantees

/-- **Sampled proofs' session as C-Flock's live audit**: at each file `F`, the key program of its parse `D F` (its unit
built, `hU F`) with `n F` units, its partition and constants; at each `θ` and `c`, the live audit `A θ c` at the
registered file's parse, its acceptance (flock-verify's), the witness extracted at each strategy, and the error
`deltaPub`. -/
noncomputable def ofLive (file : ByteArray → ByteArray) (D : ByteArray → Flock.Desc)
    (hU : ∀ F, FlockSoundness.Built (D F).unroll.unit) (n : ByteArray → ℕ) (kd K₀ : ByteArray → ℕ)
    (desc : ByteArray → Claims.Msg) (q : ByteArray → Claims.Sampling.Query)
    (A : ∀ θ c : ByteArray, LiveAt (n (file θ)) (D (file θ)) (hU (file θ)) (kd θ) (K₀ θ) (desc θ)) :
    SampledSession where
  file := file
  C F := Flock.Bridge.keyCircuit (D F) (hU F) (n F)
  n := n
  P F := Flock.Bridge.keyPartition (D F) (hU F) (n F)
  κ F := fun g => Flock.Bridge.constsOf (D F).unroll (hU F) (n F) g
  kd := kd
  K₀ := K₀
  desc := desc
  q := q
  roots θ c := (A θ c).roots
  H := Claims.Sampling.sha512Msg
  Cm θ c := (A θ c).Cd
  Out θ c := (A θ c).Reg × Finset (Fin (n (file θ))) × FlockSoundness.Discharge.Exec.Transcript
  G θ c := (A θ c).G
  Acc θ c := FlockSoundness.Discharge.ZkExec.LiveAcceptsZK (A θ c).I
  wit θ c τ _ := Flock.Bridge.witDep (LiveAt.Cc (hU := hU (file θ)) (n := n (file θ))) (A θ c).Cd Claims.Sampling.sha512Msg (A θ c).roots ((A θ c).Xr τ)
  δ θ c := Flock.Bridge.deltaPub (n (file θ)) (kd θ) (K₀ θ) (A θ c).k (A θ c).qF (A θ c).qS (A θ c).t' (A θ c).M
    (A θ c).Rw (Flock.Bridge.ioSum (A θ c).P) (A θ c).ρ

/-- **C-Flock's guarantee on the live session**: where the collision-resistance premises hold at every strategy of
every session, `ofLive`'s `KS` holds (`SampledKSLivePub`). -/
theorem ks_ofLive (file : ByteArray → ByteArray) (D : ByteArray → Flock.Desc)
    (hU : ∀ F, FlockSoundness.Built (D F).unroll.unit) (n : ByteArray → ℕ) (kd K₀ : ByteArray → ℕ)
    (desc : ByteArray → Claims.Msg) (q : ByteArray → Claims.Sampling.Query)
    (A : ∀ θ c : ByteArray, LiveAt (n (file θ)) (D (file θ)) (hU (file θ)) (kd θ) (K₀ θ) (desc θ))
    (hCR : ∀ θ c τ, (A θ c).CR τ) : (ofLive file D hU n kd K₀ desc q A).KS :=
  fun θ c τ => (A θ c).ks (q θ) τ (hCR θ c τ)

end Verity.SampledSession
