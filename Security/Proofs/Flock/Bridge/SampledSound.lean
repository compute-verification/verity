import Proofs.Flock.Bridge.Public
import Proofs.Flock.Bridge.CircuitOf

/-!
# Sampled proofs' soundness at C-Flock's live audit, prover by prover

Sampled proofs' claim reads the transcript of one fixed extractor, the one `SampledKSLivePub` is stated with
(`XplurZC`, through `witDep`): at most an `ε` share of the circuit's units are wrong in it (R4′). Its bound holds at
every prover `τ` with no hypothesis on `τ`: `ecr/sha-512` and cr/sha-512 are asked only of the finders C-Flock builds
from `τ`, and where they fail the bound is `⊤`, as C-Flock's link bound reads its collision premise (`linkBoundZC`).

* `Flock.Guarantees.LiveAt n D hU kd K₀ desc`: `SampledKSLivePub`'s setting at one session (the inputs flock-verify
  reads, the registration, the draw's setup, the verifier's bytes and A3, the analysis' `k`, `R_w`, `M`, `ρ`, and the
  SHA-512 evaluations of a run of the strategy `t'` and of a verification `v`), at the parsed circuit file `D` with `n` units, the draw size `kd`, `K₀` and the descriptor `desc`;
  `LiveAt.CR τ`, the two collision-resistance premises `SampledKSLivePub` reads at the finders built from `τ`;
  `LiveAt.ks`, its conclusion where they hold.
* `LiveAt.wrongE q τ`: the units wrong in the extractor's transcript at `τ`; `LiveAt.Holds q ε τ`, the claim;
  `LiveAt.bound τ`, `deltaPub` where `CR τ` holds and `⊤` elsewhere.
* `LiveAt.sound`: at every `τ`, `Pr[flock-verify accepts ∧ ¬ Holds] ≤ bound τ`, where `K₀ − 1 ≤ ε n`.
* `LiveAt.sound_honest`: `K₀ = ⌊ε n⌋ + 1` meets that hypothesis. `LiveAt.sound_nontrivial`: an accept-everything
  verifier breaks the bound at a prover whose extracted transcript fails the claim, where the finders meet the premises
  and `deltaPub < 1`.
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
collision-resistance premises, named assumption A3 included. `t' τ` bounds the SHA-512 evaluations of one run of the
strategy `τ`, and `v` those of one verification, public; every finder's budget is a function of the two (`qF`, `qS`),
so the bound at `τ` is at `τ`'s own cost. -/
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
  t' : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) → ℝ
  ht' : ∀ τ, 0 ≤ t' τ
  hK₀ : K₀ ≤ n
  v : ℝ
  hv : 0 ≤ v

attribute [instance] LiveAt.fΩ LiveAt.nΩ

namespace LiveAt

variable {n : ℕ} {D : Flock.Desc} {hU : Built D.unroll.unit} {kd K₀ : ℕ} {desc : Claims.Msg}
  (A : LiveAt n D hU kd K₀ desc)

/-- The draw's law: `kd` of the `n` units, from the verifier's bytes. -/
noncomputable abbrev L : Law n := Law.execOS (n := n) A.src (.subset kd) n

/-- The live audit. -/
noncomputable abbrev G := auditLive A.L A.Reg (liveOfZJ A.dg (planZHJ A.dj A.y₀) A.N A.wr)

/-- The key program's partition into the `n` units. -/
noncomputable abbrev P (_A : LiveAt n D hU kd K₀ desc) := (keyProg D.unroll hU n).partition (Discharge.Layout.outsOf D.unroll hU n)

/-- The registration's binding. -/
noncomputable abbrev vb := (rsZHJ (Reg := A.Reg) A.dj A.y₀ A.ht (Flock.Desc.parse_flat A.hD) A.hscope hU A.hpub A.hpt).bindingZ
  (rsZHJ_computes (Reg := A.Reg) A.dj A.y₀ A.ht (Flock.Desc.parse_flat A.hD) A.hscope hU A.hpub A.hpt)

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
noncomputable abbrev Cc (_A : LiveAt n D hU kd K₀ desc) := (keyProg D.unroll hU n).circuit (Discharge.Layout.outsOf D.unroll hU n)

/-- The game's rows, read off the circuit file. -/
noncomputable abbrev rows (_A : LiveAt n D hU kd K₀ desc) : Fin _A.Cc.N → Option Bridge.RowAt := fun g =>
  (FlockSoundness.Discharge.HiddenExec.rowOfH D.unroll hU n (Discharge.Layout.outsOf D.unroll hU n) g).map fun r =>
    ⟨r.2.1.val, r.1.val, Discharge.Layout.portRowW D.unroll r.2.1, r.2.2.val⟩

/-- The registration's frame-v3 roots. -/
abbrev roots : List Claims.Msg := Bridge.rootsOf D.unroll A.pub n

/-- The deployed commitment. -/
noncomputable abbrev Cd := Bridge.commitDep A.Cc A.rows D.unroll n A.roots

/-- The circuit's constants. -/
noncomputable abbrev κ (_A : LiveAt n D hU kd K₀ desc) : Fin _A.Cc.N → Option Bool := fun g => Bridge.constsOf D.unroll hU n g

/-- **The rewinding finders' budget at `τ`**: two runs of `τ` and the two conflicting openings verified. -/
noncomputable def qF (τ : Strategy A.G) : ℝ := 2 * A.t' τ + 2 * A.v

/-- **The self-clash finder's budget at `τ`**: one run of `τ` and one further verification. -/
noncomputable def qS (τ : Strategy A.G) : ℝ := A.t' τ + A.v

/-- **The collision-resistance premises at a strategy**, each a bound on the probability that a finder C-Flock builds
from `τ` outputs a SHA-512 collision, at that finder's cost as a function of `τ`'s: each table's knowledge finders, at
most `q²/2^513` at the budget `q = qF τ = 2 t' τ + 2v` or `qS τ = t' τ + v` (cr/sha-512, `SHA512CRStrict`); each
committed position's link finder, at most its expected cost over `2^256`, its cost `t' τ` per run of `τ` it plays, so
at most `t' τ · (1 + k)` per trial (ecr/sha-512, `SHA512CRExpected`; `expect_cT_leZC`). -/
def CR (τ : Strategy A.G) : Prop :=
  (∀ ω j, TableCRZ execArith hs512 (planZHJ A.dj A.y₀ (A.L.draw ω) (reg (A.sim τ))) j
    (zcStrat execArith hs512 (planZHJ A.dj A.y₀) (cont (A.sim τ) ω)) (A.qF τ) (A.qS τ)) ∧
  LinkCRZC execArith hs512 A.P (planZHJ A.dj A.y₀) (tabZHJ A.dj A.y₀) A.vb (reg (A.sim τ)) (cont (A.sim τ)) A.k A.Rw
    A.M (A.t' τ)

/-- **The link finders' cost at `τ`**: a trial for any committed position plays `τ` at most `1 + k` times in
expectation, so its expected SHA-512 evaluations are at most `t' τ · (1 + k)`, the cost `LinkCRZC` reads. -/
theorem linkCost_le (τ : Strategy A.G) (c : A.vb.Pos)
    [Nonempty (ΩcZ execArith A.P (planZHJ A.dj A.y₀) (tabZHJ A.dj A.y₀) A.vb A.L c)] :
    expect (cTZC (planZHJ A.dj A.y₀) (reg (A.sim τ)) (A.t' τ))
      (trialGZC execArith hs512 A.P (planZHJ A.dj A.y₀) (tabZHJ A.dj A.y₀) A.vb (reg (A.sim τ)) (cont (A.sim τ)) c
        A.k A.Rw)
      (trialStratZC execArith hs512 A.P (planZHJ A.dj A.y₀) (tabZHJ A.dj A.y₀) A.vb (reg (A.sim τ)) (cont (A.sim τ)) c
        A.k A.Rw) ≤
      A.t' τ * (1 + A.k) :=
  expect_cT_leZC execArith hs512 A.P (planZHJ A.dj A.y₀) (tabZHJ A.dj A.y₀) A.vb (reg (A.sim τ)) (cont (A.sim τ)) c
    A.k A.Rw (A.ht' τ)

/-- **`SampledKSLivePub` at the session**: at a strategy where the premises hold, flock-verify accepts with the
extracted witness outside the sampling relation with probability at most `deltaPub`. -/
set_option maxHeartbeats 1000000 in
theorem ks (q : Claims.Sampling.Query) (τ : Strategy A.G) (hCR : A.CR τ) :
    prob (fun o => LiveAcceptsZK A.I o ∧
        ¬ Claims.Sampling.RelS Claims.Sampling.sha512Msg A.Cd
          (((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ)) (Bridge.stmtOf A.P A.κ kd desc q A.roots) (Bridge.partOf A.P)
          (Bridge.witDep A.Cc A.Cd Claims.Sampling.sha512Msg A.roots (A.Xr τ)))
      A.G τ ≤
      ENNReal.ofReal (Bridge.deltaPub n kd K₀ A.k (A.qF τ) (A.qS τ) (A.t' τ) A.M A.Rw (Bridge.ioSum A.P) A.ρ) :=
  Flock.SecurityProofs.SampledKSLivePub n A.Reg A.I A.ht D A.hD hU A.pub A.hpub A.hpt A.hscope A.hrows A.hdup A.hroots
    desc A.qr A.hpop kd A.hkd A.hdiv A.dj A.S₀ A.d₀ A.y₀ A.wr A.dg A.N A.Ω A.Lb A.src A.hA3 A.hTags A.hCust A.k A.Rw A.M
    A.hk A.hRw A.hM A.ρ A.hρ A.hρk (A.t' τ) (A.ht' τ) K₀ A.hK₀ τ (A.qF τ) (A.qS τ) q hCR.1 hCR.2

open Classical in
/-- **The units the fixed extractor's transcript has wrong** at `τ`: the wire values the witness `witDep` builds from
`XplurZC`'s values opens (`openedOf`), read against the key program's partition. -/
noncomputable def wrongE (q : Claims.Sampling.Query) (τ : Strategy A.G) : Finset (Fin n) :=
  A.P.wrong (Bridge.openedOf A.Cc A.κ
    ((Bridge.witDep A.Cc A.Cd Claims.Sampling.sha512Msg A.roots (A.Xr τ)).val
      Claims.Sampling.sha512Msg A.Cd (Bridge.stmtOf A.P A.κ kd desc q A.roots)))

/-- **Sampled proofs' claim at `τ`** (R4′): at most an `ε` share of the `n` units are wrong in the extractor's
transcript. -/
def Holds (q : Claims.Sampling.Query) (ε : ℝ) (τ : Strategy A.G) : Prop :=
  ((A.wrongE q τ).card : ℝ) ≤ ε * n

/-- **The bound at `τ`**: `deltaPub` where the finders built from `τ` meet the collision-resistance premises, `⊤`
elsewhere. -/
noncomputable def bound (τ : Strategy A.G) : ℝ≥0∞ :=
  if A.CR τ then
    ENNReal.ofReal (Bridge.deltaPub n kd K₀ A.k (A.qF τ) (A.qS τ) (A.t' τ) A.M A.Rw (Bridge.ioSum A.P) A.ρ)
  else ⊤

/-- **Sampled proofs' soundness at the live audit, prover by prover**: at every prover `τ`, flock-verify accepts while
the extractor's transcript has more than an `ε` share of the units wrong with probability at most `bound τ`, where
`K₀ − 1 ≤ ε n`. -/
theorem sound (q : Claims.Sampling.Query) (ε : ℝ) (hε : (K₀ : ℝ) - 1 ≤ ε * n) (τ : Strategy A.G) :
    prob (fun o => LiveAcceptsZK A.I o ∧ ¬ A.Holds q ε τ) A.G τ ≤ A.bound τ := by
  unfold bound
  split_ifs with hCR
  · refine (prob_mono (fun o ho => ⟨ho.1, fun hR => ho.2 ?_⟩) _ _).trans (A.ks q τ hCR)
    have hlt := Bridge.card_wrong_lt_of_relS A.P A.κ kd K₀ desc q A.roots
      Claims.Sampling.sha512Msg A.Cd _ hR
    have hlt' : (A.wrongE q τ).card < K₀ := hlt
    have h1 : ((A.wrongE q τ).card : ℝ) + 1 ≤ K₀ := by exact_mod_cast hlt'
    show ((A.wrongE q τ).card : ℝ) ≤ ε * n
    linarith
  · exact le_top

/-- **`sound`'s hypothesis holds at the deployed choice** `K₀ = ⌊ε n⌋ + 1`. -/
theorem sound_honest (ε : ℝ) (hεn : 0 ≤ ε * n) : (((⌊ε * n⌋₊ + 1 : ℕ) : ℝ)) - 1 ≤ ε * n := by
  push_cast
  linarith [Nat.floor_le hεn]

/-- **`sound` is not trivial**: for a verifier that accepts everything, its bound fails at a prover whose extracted
transcript fails the claim, where the finders built from it meet the premises and `deltaPub < 1`. -/
theorem sound_nontrivial (q : Claims.Sampling.Query) (ε : ℝ) (τ : Strategy A.G) (hCR : A.CR τ)
    (hbad : ¬ A.Holds q ε τ)
    (h1 : Bridge.deltaPub n kd K₀ A.k (A.qF τ) (A.qS τ) (A.t' τ) A.M A.Rw (Bridge.ioSum A.P) A.ρ < 1) :
    ¬ prob (fun _ => True ∧ ¬ A.Holds q ε τ) A.G τ ≤ A.bound τ := by
  intro h
  have h2 := (prob_true A.G τ).symm.le.trans
    ((prob_mono (p := fun _ => True) (q := fun _ => True ∧ ¬ A.Holds q ε τ) (fun _ _ => ⟨trivial, hbad⟩) A.G τ).trans h)
  rw [bound, if_pos hCR] at h2
  exact not_le.2 (ENNReal.ofReal_lt_one.2 h1) h2

end LiveAt

end Flock.Guarantees
