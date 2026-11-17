module

public import Properties.Verity.Game
public import Proofs.Properties.Verity.Soundness.Composite
public import Properties.Spot.Draw
public import Proofs.Flock.Bridge.V5.Statement
public import Proofs.Flock.Bridge.CircuitOf
public import Proofs.Flock.Claims.Bridge.CustodyZkP
public import Proofs.Flock.Soundness.ZK.RealLeaves
public import Proofs.Properties.Verity.Sampled.Deployed
public import Proofs.Properties.Verity.Sampled.Leaves

/-! SPOT's subprotocol of `Verity.Soundness` at the session v2 sets up; it covers the one-stage form, and the recursive
form (`VStar.Recursive`) is not this subprotocol's. -/

@[expose] public section

namespace Verity.Sampled

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

/-- **The session v2 sets up on the parameters and the committed files**: V5's binders, each the value v2's check
derives (the files' parse, the public file over the circuit's empty public input ports, the ports' published domains
distinct, the population, the registration's descriptor, query, sampling-design object and work table and the
population's order, and the sampling design the verifier samples under from them, a subset one), the device's byte
stream the sample reads, A3 at those bytes and the verify command's tags as fields, and the bound's parameters. The
registration, the sample file (the printed one), the writer and the rounds' digest are the ones the period's
registration fixes (`Live.Reg`, `Live.dj`, `Live.wr`, `Live.dg`), and `live-verifier` holds at the writer under the
sample file's sampling design (`Live.hCust`). The live rounds' count and the tree openings the session reads after a
digest are its coin tree's (`Live.N`, `Live.rounds`). -/
structure Live where
  /-- the population: the public file's proof blocks -/
  n : ℕ
  I : FlockSoundness.Discharge.Exec.Inputs
  /-- the verifier's own recomputation of each proof block's public input words -/
  own : Option ByteArray
  ht : I.tags.typed = false
  D : Flock.Desc
  hD : Flock.Desc.parse (zkTags I) I.circuitFile I.tables = .ok D
  hU : FlockSoundness.Built D.unroll.unit
  /-- the circuit names no public input port -/
  hpi : Flock.HmIn.parse D.unroll = .ok {}
  pub : Flock.Public
  hpub : Flock.HmIn.loadPublic (zkTags I) D.unroll {} I.publicFile = .ok pub
  /-- the public file as the committed files' circuit reads it (`Flock.Bridge.parsedOf`) -/
  pubO : Flock.Public
  hpubO : Flock.HmOut.loadPublic (zkTags I) D.unroll I.publicFile = .ok pubO
  hscope : FlockSoundness.Discharge.Layout.scopeOkH D.unroll = true
  /-- no two of the circuit's ports have one published domain -/
  hdup : Flock.PortRoots.dupId D.unroll pub = false
  hn : pub.n = n
  hnO : pubO.n = n
  /-- the statement's descriptor -/
  desc : Claims.Msg
  /-- the registration's query -/
  qr : Lean.Json
  /-- the registration's population is the public file's proof blocks, in order -/
  hpop : Flock.Bridge.PopMatches desc qr n
  /-- the registration's sampling-design object and work table -/
  lj : Lean.Json
  tb : Option Lean.Json
  /-- the sampling design the verifier samples under from the registration (`Flock.Bridge.designAt`) is the sample
  file's, its proof blocks sampled uniformly (`Live.hCust`) -/
  hsub : Flock.Bridge.designAt desc qr lj tb n = .subset (Flock.Bridge.samplingDesignK (Flock.Bridge.designAt desc qr lj tb n))
  hdiv : Flock.Bridge.samplingDesignK (Flock.Bridge.designAt desc qr lj tb n) % nTab I = 0
  S₀ : Finset (Fin n)
  d₀ : Lean.Json
  /-- a table `verify --zk` sets up, whose refused table the plan reads only at a sample `verify --zk` sets up no table
  for (`tableZHJP`), where no witness satisfies it -/
  y₀ : SampleSetupZHJP (inputsJ I 0) own (mPtsOf D.unroll.g (Flock.Bridge.samplingDesignK (Flock.Bridge.designAt desc qr lj tb n) / nTab I)) S₀ d₀
  /-- the device's bytes the sample reads -/
  Ω : Type
  [fΩ : Fintype Ω]
  [nΩ : Nonempty Ω]
  Lb : Fin 1 → ℕ
  src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256
  hA3 : FlockSoundness.Assumptions.UniformRandomBytes Lb src
  hTags : TagsOkZHJ I
  k : ℕ
  Rw : ℕ
  M : ℕ
  hk : 1 ≤ k
  hRw : 1 ≤ Rw
  hM : 1 ≤ M
  ρ : ℝ
  hρ : ρ < 1
  hr : ∀ S R j, rateZ (planZHJP (Reg := ByteArray) (printedSample (.subset (Flock.Bridge.samplingDesignK (Flock.Bridge.designAt desc qr lj tb n)))) y₀ S R) j k ≤ ρ
  /-- the claim's share: at most an `ε` share of the proof blocks unbound -/
  ε : ℝ
  /-- one opening's verification, in SHA-512 calls -/
  v : ℕ
  /-- one acceptance check, in SHA-512 calls -/
  w : ℕ
  /-- hashing one message the device or the commitment takes a digest of (a round's message, the record, the digests
  the composition's commitment binds), in SHA-512 calls -/
  h : ℕ

attribute [instance] Live.fΩ Live.nΩ

/-- **The widening of the sample's sample a projection reads**: a finite type of the device's other releases, those a
strategy of the period's game may ask for that the session doesn't read (other families' coins, other trees' samples). A
Flock strategy flips no coin of its own, so the projected strategy reads them off the sample's sample, widened by them
(`Live.extend`). -/
structure Widening where
  T : Type
  [fT : Fintype T]
  [nT : Nonempty T]

attribute [instance] Widening.fT Widening.nT

theorem uniform_fst {Ω T : Type} [Fintype Ω] [Fintype T] [Nonempty T] {m : ℕ} {Lb : Fin m → ℕ}
    {t : Ω → (i : Fin m) → Fin (Lb i) → Fin 256} (h : FlockSoundness.Assumptions.UniformRandomBytes Lb t) :
    FlockSoundness.Assumptions.UniformRandomBytes Lb (fun ω : Ω × T => t ω.1) := fun E =>
  (FlockSoundness.ZK.prCoin_fst (fun ω => E (t ω))).trans (h E)

theorem avg_fst {Ω T : Type} [Fintype Ω] [Fintype T] [Nonempty T] (G : Ω → ℝ≥0∞) :
    Verity.Game.avg (fun ω : Ω × T => G ω.1) = Verity.Game.avg G :=
  (FlockSoundness.ZK.avg_prod_split _).trans
    (congrArg Verity.Game.avg (funext fun c => FlockSoundness.Game.avg_const (G c)))

namespace Live

variable (A : Live)

/-- **The sampling design the verifier samples under**, from the registration (`Flock.Bridge.designAt`). -/
abbrev design : Flock.Draw.SamplingDesign := Flock.Bridge.designAt A.desc A.qr A.lj A.tb A.n

/-- **The sample's size**: the proof blocks the sampling design samples (`Flock.Bridge.samplingDesignK`). -/
abbrev kd : ℕ := Flock.Bridge.samplingDesignK A.design

/-- **The session's coin tree**: the keyed coin tree `verify --zk` sets the session up on (`ZkExec.zkCoins`), its
session stream and each table's two rep streams, `2^roundsLog` rounds each (`Flock.Zk.coinSpec`). -/
def coinSpec : Flock.CoinTree.Spec := Flock.Zk.coinSpec (A.I.tags.tableNames A.I.count)

theorem zkCoins_eq : zkCoins A.I = .tree A.coinSpec := rfl

theorem coinSpec_roundsLog : A.coinSpec.roundsLog = 9 := rfl

/-- **The live rounds' count**: the prover submits a round on a table's rep stream, or stops, at most once for each
round of the session's coin tree's two rep streams, `2 · 2^roundsLog`. -/
def N : ℕ := 2 * 2 ^ A.coinSpec.roundsLog

/-- **The tree openings the session reads after a digest of the prover's message**: its link points', and each round
of its coin tree's two rep streams, `2 · 2^roundsLog + 1`. -/
def rounds : ℕ := 2 * 2 ^ A.coinSpec.roundsLog + 1

theorem rounds_eq : A.rounds = A.N + 1 := rfl

theorem stops_le_maxLeaves : 2 * (1 + A.N) ≤ maxLeaves := by
  rw [Live.N, coinSpec_roundsLog]
  decide

theorem rounds_tree : 2 * 2 ^ A.coinSpec.roundsLog + 1 ≤ A.rounds := le_rfl

/-- **The registration** the live audit's prover sends before the sample: the registration record's digest, the head of
the registration log after it (`Spot.Docket.digest`), which the period's composite commitment binds
under SPOT's id (`Core.Composition.bind`) and the sample's coins are released after (`coinsRead`'s `after`). -/
abbrev Reg (_ : Live) : Type := ByteArray

/-- **The coin trees' hiding term**: each leaf the device's coin trees hold in a play, twice, at `2^-257`, the leftover
hash lemma's bound on the mean over the device's fresh tree key of hm96's hiding gap at one leaf. The prover sees every
leaf before it chooses what to open, so the term counts the leaves the device samples, not the rounds the session
opens; the device refuses every tree past its caps (`Verity.releases`), so every run of a play samples trees of at most
`maxLeaves` (`2^49`) leaves in all (`Leaves.play_leaves_le`, with no premise), and the term is `2 · 2^49/2^257 = 2^-207`
at every session. -/
noncomputable def hidingTerm (_ : Live) : ℝ≥0∞ := 2 * (maxLeaves : ℝ≥0∞) / 2 ^ 257

/-- **The session's sample file**: at each sample, the `.subset` print `verity-verify-proof draw` gives of the session's `kd` units
(`printedSample`), the member's share of the window's sample under the verifier's sampling design `L`; `verify --zk` at the session
admits only a subset sample (`recordSample_subset`). The bound comes from v1's window check
(`Spot.OneStage.checkAt`), which v2 runs at the window and which derives the sample under the verifier's own
sampling design. -/
abbrev dj : Finset (Fin A.n) → Lean.Json := printedSample (.subset A.kd)

theorem hsz : ∀ S ud, Flock.Draw.UnitSample.ofJson (A.dj S) = .ok ud → ud.units.size = A.kd :=
  sampleSize_deployed A.kd

/-- **A round's digest**: SHA-512 of the round's framed bytes, the `msg_sha512` a session's record holds of each
round. -/
abbrev dg (_ : Live) : ByteArray → ByteArray := Flock.Sha512.hash

/-- **The writer**: at a sample's live outcome, a transcript `verify --zk` accepts at `own` whose record carries the sample
and decodes to the outcome, with its proof files, where one exists, else one without proof files
(`Claims.Bridge.CustodyZkP.wrZP`). At the outcome an accepted transcript's files decode to, what it writes is accepted
(`Claims.Bridge.CustodyZkP.acceptsZKP_wrZP_planZHJP`); no writer reading only the outcome can write the transcript's
bytes, since its record carries bytes the outcome doesn't (the hello's nonce, the coin tree, the proofs' hashes). -/
noncomputable abbrev wr : (S : Finset (Fin A.n)) → (R : A.Reg) →
    LiveOutJ (planZHJP (Reg := A.Reg) A.dj A.y₀ S R).length (mPtsOf A.D.unroll.g (A.kd / nTab A.I)) →
      FlockSoundness.Discharge.Exec.Transcript :=
  Claims.Bridge.CustodyZkP.wrZP A.I A.own A.dj fun S R => (planZHJP (Reg := A.Reg) A.dj A.y₀ S R).length

theorem hCust : FlockSoundness.Assumptions.Zk.RecordCustodyZKJPLaw (mPts := mPtsOf A.D.unroll.g (A.kd / nTab A.I))
    A.I A.own A.dj (fun S R => (planZHJP (Reg := A.Reg) A.dj A.y₀ S R).length) A.wr fun _ => .subset A.kd :=
  recordCustodyZKJPLaw_deployed A.I A.own A.kd _

theorem hb : Flock.Bridge.NotBernoulli A.design := Flock.Bridge.notBernoulli_designAt _ _ _ _ _

theorem hlaw : Flock.Bridge.SamplingDesignOn A.design (Flock.Bridge.nsOf A.desc A.qr) A.kd := by
  rcases Flock.Bridge.designAt_cases A.desc A.qr A.lj A.tb A.n with h | h
  · exact Flock.Bridge.samplingDesignOn_samplesUnder A.hpop h (Flock.Bridge.notBernoulli_samplesUnder h)
  · unfold Live.kd Live.design
    rw [h]
    exact ⟨rfl, Nat.zero_le _⟩

noncomputable def deployed : Live where
  __ := A
  Ω := DeployedCoins
  fΩ := inferInstance
  nΩ := inferInstance
  Lb := deployedLb
  src := deployedSrc
  hA3 := uniformRandomBytes_deployed

/-- **The session with the sample's sample widened by the coins `S`**: the sample reads only the session's bytes, so the
sample's sampling design is the session's and A3 holds at the widened sample (`uniform_fst`). -/
noncomputable def extend (S : Widening) : Live where
  __ := A
  Ω := A.Ω × S.T
  fΩ := inferInstance
  nΩ := inferInstance
  src ω := A.src ω.1
  hA3 := uniform_fst A.hA3

/-- **The sample's sampling design**: the session's sampling design over the population as `Spot.Select.select` samples it,
`Spot.Draw.sampleUnits` on the device's bytes in order; a sample that fails samples every unit. -/
noncomputable def L : FlockSoundness.Audit.SamplingDesign A.n where
  Ω := A.Ω
  draw ω :=
    match _root_.Spot.Draw.sampleUnits A.design A.n (_root_.Properties.Spot.bytesOf (A.src ω 0)) with
    | .ok (d, _) => Finset.univ.filter fun u => u.val ∈ d
    | .error _ => Finset.univ

theorem L_eq : A.L = Flock.Bridge.execSel (n := A.n) A.src A.design A.n := by
  unfold Live.L Flock.Bridge.execSel
  refine congrArg (@Verity.SamplingDesign.mk (Fin A.n) A.Ω _ _) (funext fun ω => ?_)
  rcases _root_.Spot.Draw.sampleUnits A.design A.n (_root_.Properties.Spot.bytesOf (A.src ω 0)) with
    _ | ⟨d, x⟩ <;> rfl

/-- **Flock's live audit at the session**, the game V5 bounds: the prover registers, the device's bytes sample, and
the live session runs at the sample. -/
noncomputable def game :=
  auditLive A.L A.Reg (liveOfZJ A.dg (planZHJP A.dj A.y₀) A.N A.wr)

/-- V5's circuit: the key program's at the session's ports, with its outputs. -/
noncomputable abbrev Cc :=
  (keyProgP A.D.unroll {} A.hU A.n).circuit (FlockSoundness.Discharge.Layout.outsOfP A.D.unroll A.hU A.n {})

/-- V5's partition: the key program's at the session's ports, and its outputs. -/
noncomputable def P :=
  (keyProgP A.D.unroll {} A.hU A.n).partition (FlockSoundness.Discharge.Layout.outsOfP A.D.unroll A.hU A.n {})

theorem hcS : Flock.HmOut.parseStmt (zkTags A.I) A.I.circuitFile A.I.tables = .ok A.D.unroll :=
  Flock.Desc.parseStmt_stmt (Flock.Desc.parseStmt_of_parse A.hD)

/-- V5's binding at the session's ports. -/
noncomputable def vb :=
  (rsZHJP (Reg := A.Reg) A.dj A.y₀ A.ht A.hcS A.hscope A.hU A.hpi A.hpub).bindingZ
    (rsZHJP_computes (Reg := A.Reg) A.dj A.y₀ A.ht A.hcS A.hscope A.hU A.hpi A.hpub)

/-- **V5's simulator at a strategy of the live audit**: the strategy in the compiled batched session at verity-verify-proof
`--zk`'s acceptance at `own`, through the decoder `RecordCustodyZKJPLaw` and the sample file's size give at the sample
file's sampling design, the subset sampling design of `kd` units. -/
noncomputable def sim (τ : FlockSoundness.Game.Strategy A.game) :=
  simZJA execArith hs512 (planZHJP A.dj A.y₀) (AcceptsZKP A.I A.own) A.wr
    (opnAtZHJP (planZHJP A.dj A.y₀) A.I A.own A.dj) A.dg (decAtZJ (planZHJP A.dj A.y₀)) (fun _ _ _ _ => true)
    (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) A.N
    (decodesZCJH_custodyP A.I A.own A.ht A.hcS (.subset A.kd) A.kd A.hdiv A.dj A.y₀ A.wr A.L A.hTags A.hCust A.hsz) τ

/-- **V5's extractor at a strategy of the live audit**: `XplurZC` at the simulator's registration and continuation. -/
noncomputable def Xr (τ : FlockSoundness.Game.Strategy A.game) :=
  XplurZC execArith hs512 A.P (planZHJP A.dj A.y₀) (tabZHJP A.dj A.y₀) A.vb
    (FlockSoundness.Audit.reg (A.sim τ)) (FlockSoundness.Audit.cont (A.sim τ)) A.k A.Rw

/-- The rewinding table finders' budget at `t` calls, `2t + 2v`: two runs after a cap, and the two conflicting openings
verified. -/
def qF (t : ℕ) : ℝ := 2 * t + 2 * A.v

/-- The self-clash finder's budget at `t` calls, `t + v`. -/
def qS (t : ℕ) : ℝ := t + A.v

/-- The link finders' charge per run at `t` calls, `t + w`: the prover's calls and the acceptance check's. -/
def tL (t : ℕ) : ℝ := t + A.w

/-- The two-continuation finders' budget at `t` calls, `2t + 2h`: the play twice after the digest, and the two
revealed messages hashed. -/
def qM (t : ℕ) : ℝ := 2 * t + 2 * A.h

/-- **The session's per-finder SHA-512 premises at the strategy `τ` and `t` calls**: `cr/sha-512` at each table's
knowledge finders at `2t + 2v` and `t + v` (`TableCRZ`), and `ecr/sha-512` at the link finders at `t + w` per run
(`LinkCRZC`), as V5 takes them. -/
structure FinderCR (τ : FlockSoundness.Game.Strategy A.game) (t : ℕ) : Prop where
  table : ∀ ω j, TableCRZ execArith hs512 (planZHJP A.dj A.y₀ (A.L.draw ω) (FlockSoundness.Audit.reg (A.sim τ))) j
    (zcStrat execArith hs512 (planZHJP A.dj A.y₀) (FlockSoundness.Audit.cont (A.sim τ) ω)) (A.qF t) (A.qS t)
  link : LinkCRZC execArith hs512 A.P (planZHJP A.dj A.y₀) (tabZHJP A.dj A.y₀) A.vb
    (FlockSoundness.Audit.reg (A.sim τ)) (FlockSoundness.Audit.cont (A.sim τ)) A.k A.Rw A.M (A.tL t)

/-- **The sample's term**: the sampling design's own miss at the claim's share, the largest probability that the sample holds no proof
block of a set of more than an `ε` share of them, the acceptance probability of the sampling design's one-stage design
(`Properties.Spot.oneStageOf`) over the population's strata (`Claims.Sampling.OneStage.Pa`); at the subset sampling design
of `kd` units, `C(n − K₀, kd)/C(n, kd)` with `K₀ = ⌊εn⌋ + 1`. Those sets are exactly the unbound sets that fail the
claim, so no count of them enters; with none (`ε ≥ 1`) the term is `0`. -/
noncomputable def sampleMiss : ℝ≥0∞ :=
  ENNReal.ofReal (⨆ B : {B : Finset ℕ // B ⊆ Finset.range A.n ∧ A.ε * A.n < B.card},
    (_root_.Properties.Spot.oneStageOf A.design).Pa (Flock.Bridge.nsOf A.desc A.qr) B.1)

/-- **The registration's roots term**: `ecr/sha-512` at the finder that recomputes the registration's roots at the
public file's leaf map (`Flock.Bridge.rootsFinderL`, no coins, `Flock.Bridge.rootsCostL` SHA-512 calls): two ports with
one root and different domains are a collision it finds. -/
noncomputable def rootsErr : ℝ≥0∞ :=
  Claims.ecrBound Claims.Sampling.sha512Msg
    (Flock.Bridge.rootsFinderL A.D.unroll (Flock.Bridge.LeafMap.ofPublic A.D.unroll A.pub A.n))
    (Flock.Bridge.rootsCostL A.D.unroll (Flock.Bridge.LeafMap.ofPublic A.D.unroll A.pub A.n))

open Classical in
/-- **The session's error at `t` calls**, V5's at the worst registration: the sample's term, the tables' knowledge term
at the finders' budgets, the binding error at `t + w` per run, and the registration's roots term (`Live.rootsErr`). -/
noncomputable def sessionDelta (t : ℕ) : ℝ≥0∞ :=
  A.sampleMiss +
    (⨆ R : A.Reg, ksAvgStrictZ (planZHJP A.dj A.y₀) A.L A.k (A.qF t) (A.qS t) R) +
    ENNReal.ofReal ((∑ p : A.vb.Pos,
        (Fintype.card (ΩcZ execArith A.P (planZHJP A.dj A.y₀) (tabZHJP A.dj A.y₀) A.vb A.L p) : ℝ) /
          Fintype.card A.L.Ω) *
      (2 * (A.tL t * (1 + A.k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * A.M) + A.k / (Real.exp 1 * A.Rw)) / (1 - A.ρ)) +
    A.rootsErr

/-- **A two-continuation finder's term at budget `q`**: the square root of the strict birthday bound `q²/2^513`, since
the most likely revealed message misses with probability at most the root of the finder's success. -/
noncomputable def twoRun (q : ℝ) : ℝ≥0∞ := ENNReal.ofReal (Real.sqrt (q ^ 2 / 2 ^ (513 : ℝ)))

/-- **The subprotocol's δ at the session, `t` SHA-512 calls and the read finder's bound `cr`**: the session's error, a
two-continuation term per round opened after a digest, one for the record and one for the digests the composition's
commitment binds, the coin trees' hiding term (`Live.hidingTerm`), the two file finders' `2²/2^513`, and `cr`. -/
noncomputable def delta (t : ℕ) (cr : ℝ) : ℝ≥0∞ :=
  A.sessionDelta t + A.rounds * twoRun (A.qM t) + twoRun (A.qM t) + twoRun (A.qM t) + A.hidingTerm +
    2 * ENNReal.ofReal ((2 : ℝ) ^ 2 / 2 ^ (513 : ℝ)) + ENNReal.ofReal cr

theorem ksAvg_extend (S : Widening) (k : ℕ) (qF qS : ℝ) (R : A.Reg) :
    ksAvgStrictZ (planZHJP A.dj A.y₀) (A.extend S).L k qF qS R = ksAvgStrictZ (planZHJP A.dj A.y₀) A.L k qF qS R :=
  avg_fst (T := S.T) fun ω : A.Ω => ⨆ j : Fin (planZHJP A.dj A.y₀ (A.L.draw ω) R).length,
    ENNReal.ofReal (ksStrictZ (planZHJP A.dj A.y₀ (A.L.draw ω) R) j k qF qS /
      (1 - rateZ (planZHJP A.dj A.y₀ (A.L.draw ω) R) j k))

open Classical in
theorem ratio_extend (S : Widening) (p : A.vb.Pos) :
    (Fintype.card (ΩcZ execArith A.P (planZHJP A.dj A.y₀) (tabZHJP A.dj A.y₀) A.vb (A.extend S).L p) : ℝ) /
        Fintype.card (A.extend S).L.Ω =
      (Fintype.card (ΩcZ execArith A.P (planZHJP A.dj A.y₀) (tabZHJP A.dj A.y₀) A.vb A.L p) : ℝ) /
        Fintype.card A.L.Ω := by
  have hc : Fintype.card (ΩcZ execArith A.P (planZHJP A.dj A.y₀) (tabZHJP A.dj A.y₀) A.vb (A.extend S).L p) =
      Fintype.card (ΩcZ execArith A.P (planZHJP A.dj A.y₀) (tabZHJP A.dj A.y₀) A.vb A.L p) * Fintype.card S.T :=
    (Fintype.card_congr (Equiv.prodSubtypeFstEquivSubtypeProd (β := S.T)
      (p := fun ω : A.L.Ω => ReadsSampleZ execArith A.P (planZHJP A.dj A.y₀) (tabZHJP A.dj A.y₀) A.vb p ω))).trans
      (Fintype.card_prod _ _)
  have hΩ : Fintype.card (A.extend S).L.Ω = Fintype.card A.L.Ω * Fintype.card S.T := Fintype.card_prod A.Ω S.T
  rw [hc, hΩ]
  simp only [Nat.cast_mul]
  exact mul_div_mul_right _ _ (Nat.cast_ne_zero.mpr Fintype.card_ne_zero)

open Classical in
theorem sessionDelta_extend (S : Widening) (t : ℕ) : (A.extend S).sessionDelta t = A.sessionDelta t := by
  have hks : (⨆ R : A.Reg, ksAvgStrictZ (planZHJP A.dj A.y₀) (A.extend S).L A.k (A.qF t) (A.qS t) R) =
      ⨆ R : A.Reg, ksAvgStrictZ (planZHJP A.dj A.y₀) A.L A.k (A.qF t) (A.qS t) R :=
    iSup_congr fun R => A.ksAvg_extend S _ _ _ R
  have hsum : (∑ p : A.vb.Pos,
        (Fintype.card (ΩcZ execArith A.P (planZHJP A.dj A.y₀) (tabZHJP A.dj A.y₀) A.vb (A.extend S).L p) : ℝ) /
          Fintype.card (A.extend S).L.Ω) =
      ∑ p : A.vb.Pos,
        (Fintype.card (ΩcZ execArith A.P (planZHJP A.dj A.y₀) (tabZHJP A.dj A.y₀) A.vb A.L p) : ℝ) /
          Fintype.card A.L.Ω :=
    Finset.sum_congr rfl fun p _ => A.ratio_extend S p
  show A.sampleMiss + (⨆ R : A.Reg, ksAvgStrictZ (planZHJP A.dj A.y₀) (A.extend S).L A.k (A.qF t) (A.qS t) R) +
      ENNReal.ofReal ((∑ p : A.vb.Pos,
          (Fintype.card (ΩcZ execArith A.P (planZHJP A.dj A.y₀) (tabZHJP A.dj A.y₀) A.vb (A.extend S).L p) : ℝ) /
            Fintype.card (A.extend S).L.Ω) *
        (2 * (A.tL t * (1 + A.k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * A.M) + A.k / (Real.exp 1 * A.Rw)) /
          (1 - A.ρ)) + A.rootsErr = A.sessionDelta t
  rw [hks, hsum]
  rfl

theorem delta_extend (S : Widening) (t : ℕ) (cr : ℝ) : (A.extend S).delta t cr = A.delta t cr := by
  unfold Live.delta
  rw [sessionDelta_extend]
  rfl

/-- **How v2 reads the committed files at the session**: the verify command's tags, the circuit file, its tables and
the public file. -/
def fileRead : Verity.FileRead := ⟨zkTags A.I, A.I.circuitFile, A.I.tables, A.I.publicFile⟩

theorem circuit_eq : A.fileRead.circuit = Flock.Bridge.circuitOf A.D A.hU A.n := by
  have h := Flock.Bridge.parsedOf_eq A.hD A.hpubO A.hU
  rw [A.hnO] at h
  exact Flock.Bridge.circuitOfFile_eq h

/-- The public file's leaf map. -/
noncomputable def Lm := Flock.Bridge.LeafMap.ofPublic A.D.unroll A.pub A.n

/-- The deployed commitment's rows: each committed gate's port, leaf, width and place, read off the circuit file. -/
noncomputable def rows : Fin A.Cc.N → Option Flock.Bridge.RowAt := fun g =>
  (FlockSoundness.Discharge.HiddenExec.rowOfHP A.D.unroll A.hU A.n {}
      (FlockSoundness.Discharge.Layout.outsOfP A.D.unroll A.hU A.n {}) g).map fun r =>
    ⟨r.2.1.val, A.Lm.ref r.2.1.val r.1.val, FlockSoundness.Discharge.Layout.portRowW A.D.unroll r.2.1, r.2.2.val⟩

/-- The deployed commitment's roots, one a port. -/
noncomputable def roots := Flock.Bridge.rootsOfL A.D.unroll A.Lm

/-- **The deployed commitment** at the public file's leaf map, the one V5's relation reads. -/
noncomputable def Cd := Flock.Bridge.commitDepL A.Cc A.rows A.D.unroll A.Lm A.roots

/-- The circuit's constants, public. -/
noncomputable def κ : Fin A.Cc.N → Option Bool := fun g => Flock.Bridge.constsOfP A.D.unroll A.hU A.n {} A.pub g

/-- **The claim's transcript at a strategy of the live audit**: V5's trace fixed at registration (`Xr`) as the
deployed commitment's sampling witness (`Flock.Bridge.witDep`), opened at each gate's claims position under the
commitment's roots and read at the circuit's gates with its constants (`Flock.Bridge.traceOf`), on the committed
files' circuit (`circuit_eq`). -/
noncomputable def extract (τ : FlockSoundness.Game.Strategy A.game) : A.fileRead.circuit.Trace :=
  cast (congrArg Verity.Circuit.Trace A.circuit_eq.symm)
    (Flock.Bridge.traceOf A.D A.hU A.n A.κ
      ((Flock.Bridge.witDep A.Cc A.Cd Claims.Sampling.sha512Msg A.roots (A.Xr τ)).val Claims.Sampling.sha512Msg A.Cd
        (Flock.Bridge.stmtOf A.P A.κ (_root_.Properties.Spot.oneStageOf A.design) A.desc ⟨"", 0, []⟩
          A.roots)))

def OpensUnder (r : Verity.ExecutionPhaseFiles) (op : (g : Fin r.circuit.N) → BitVec (r.circuit.w g) → Prop) : Prop :=
  ∀ (h : r.circuit.N = A.Cc.N) (g : Fin r.circuit.N) (b : Bool) (d : Claims.Msg), r.consts g = none →
    A.Cd.Opens Claims.Sampling.sha512Msg A.roots ⟨Flock.Bridge.posOf A.Cc (Fin.cast h g), b, d⟩ = true →
      ∀ v : BitVec (r.circuit.w g), v.getLsbD 0 = b → op g v

end Live

variable {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α]

/-- **The constructions the subprotocol reads, applied to a strategy of the period's game**: the projection, the device's other
releases the strategy may ask for (`coins`) and the Flock live strategy it is read as at the session widened by them
(`project`); the read finder (`Vocab.finder`'s: a collision among the strings v2 hashes in reading the device's log and
trees and those the device hashes) and the forger of the device's signature (`Vocab.forger`'s); and the
two-continuation finders, one per round the device opens after a digest of the prover's message, one for the record
the commitment is a digest of, and one for the digests the composition's commitment binds, each at most `2t + 2h`
SHA-512 calls on every coin (`Live.qM`). -/
structure Constructions (V : Vocab d α) where
  coins : (A : Live) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) →
    Verity.Game.Strategy (V.game θ F c) → Widening
  project : (A : Live) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) →
    (σ : Verity.Game.Strategy (V.game θ F c)) → FlockSoundness.Game.Strategy (A.extend (coins A θ F c σ)).game
  readFinder : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Verity.Game.Strategy (V.game θ F c) →
    Claims.Finder
  forger : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Verity.Game.Strategy (V.game θ F c) →
    Hash.Assumptions.LmsForger
  msgFinder : (A : Live) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) →
    Verity.Game.Strategy (V.game θ F c) → Fin A.rounds → Claims.Finder
  recordFinder : (A : Live) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) →
    Verity.Game.Strategy (V.game θ F c) → Claims.Finder
  bindFinder : (A : Live) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) →
    Verity.Game.Strategy (V.game θ F c) → Claims.Finder
  msg_calls : ∀ A θ F c σ j (x : (msgFinder A θ F c σ j).S),
    ((msgFinder A θ F c σ j).cost x : ℝ) ≤ A.qM (V.cost θ F c σ)
  record_calls : ∀ A θ F c σ (x : (recordFinder A θ F c σ).S),
    ((recordFinder A θ F c σ).cost x : ℝ) ≤ A.qM (V.cost θ F c σ)
  bind_calls : ∀ A θ F c σ (x : (bindFinder A θ F c σ).S),
    ((bindFinder A θ F c σ).cost x : ℝ) ≤ A.qM (V.cost θ F c σ)

/-- **The session a strategy of the period's game is projected into**: `A` widened by the coins the projection reads. -/
noncomputable abbrev Constructions.live {V : Vocab d α} (K : Constructions V) (A : Live) (θ : ByteArray)
    (F : Array ByteArray) (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) : Live :=
  A.extend (K.coins A θ F c σ)

structure FinderCR (V : Vocab d α) (K : Constructions V) (A : Live) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) : Prop where
  session : (K.live A θ F c σ).FinderCR (K.project A θ F c σ) (V.cost θ F c σ)
  msg : ∀ j, Claims.StrictBirthday Verity.sha512Msg (K.msgFinder A θ F c σ j)
  record : Claims.StrictBirthday Verity.sha512Msg (K.recordFinder A θ F c σ)
  bind : Claims.StrictBirthday Verity.sha512Msg (K.bindFinder A θ F c σ)
  file : Claims.StrictBirthday Verity.sha512Msg (V.fileFinder θ F)
  publicFile : Claims.StrictBirthday Verity.sha512Msg (V.publicFinder θ F)

def CouplingTarget (V : Vocab d α) (k : Verity.Registration.Composite) (K : Constructions V) (A : Live)
    (θ : ByteArray) (F : Array ByteArray) : Prop :=
  ∀ (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)),
    Verity.Game.prob (fun o => V.Verdict k .sampled o) (V.game θ F c) σ ≤
      FlockSoundness.Game.prob (fun o => LiveAcceptsZKP A.I A.own o) (K.live A θ F c σ).game
          (K.project A θ F c σ) +
        (∑ j, ENNReal.ofReal (Real.sqrt ((K.msgFinder A θ F c σ j).Pr Verity.sha512Msg)) +
          ENNReal.ofReal (Real.sqrt ((K.recordFinder A θ F c σ).Pr Verity.sha512Msg)) +
          ENNReal.ofReal (Real.sqrt ((K.bindFinder A θ F c σ).Pr Verity.sha512Msg)) + A.hidingTerm +
          ENNReal.ofReal ((K.readFinder θ F c σ).Pr Verity.sha512Msg) +
          ENNReal.ofReal ((K.forger θ F c σ).Pr Verity.lmsVerify))

def ReadsV2 (k : Verity.Registration.Composite) : Prop :=
  Verity.component k .sampled = some VStar.Recursive.Sampled.protocol

theorem readsV2_of_component {k : Verity.Registration.Composite} {P : _root_.Core.BlobProtocol}
    (h : Verity.component k .sampled = some P) : ReadsV2 k :=
  h.trans (congrArg some (Verity.component_eq_some.1 h).2)

/-- **v2's parameters in the composite's parameters `θ`**: the first component parameters the composition's format
decodes under the sampled component's id, as `caseOf` finds them, read by its decoder in its one-stage form. The
composition accepts only parameters that name each component once (`Properties.Composition.Fits`), so where it accepts
they are the ones `Verity.CaseAccepts` reads (`Properties.Composition.DecodedAccepted`); at any `θ` there is at most
one (`ParamsAt.unique`). -/
def ParamsAt (θ : ByteArray) (p : _root_.Spot.OneStageZk.Params) : Prop :=
  ∃ ps, (_root_.Core.canonicalJson θ >>= _root_.Core.Composition.decodeParams) = .ok ps ∧
    ∃ x, ps.find? (·.id == VStar.Recursive.Sampled.PROTOCOL) = some x ∧
      (_root_.Core.canonicalJson x.params.toUTF8 >>= VStar.Recursive.Sampled.paramsOfJson) = .ok (.inl p)

theorem ParamsAt.unique {θ : ByteArray} {p p' : _root_.Spot.OneStageZk.Params} (h : ParamsAt θ p)
    (h' : ParamsAt θ p') : p = p' := by
  obtain ⟨ps, hps, x, hx, hp⟩ := h
  obtain ⟨ps', hps', x', hx', hp'⟩ := h'
  obtain rfl : ps = ps' := Except.ok.inj (hps.symm.trans hps')
  obtain rfl : x = x' := Option.some.inj (hx.symm.trans hx')
  exact Sum.inl.inj (Except.ok.inj (hp.symm.trans hp'))

/-- **The parameters `θ` name no members**: v2's parameters in them, where they name SPOT's one-stage form
(`ParamsAt`), are of the one statement's form, so what v2 accepts on them the one statement's check accepts. -/
def OneStatement (θ : ByteArray) : Prop :=
  ∀ p, ParamsAt θ p → p.members = none

/-- **The Flock sampling design the sample reads over `n` units** under the verifier `v`'s agreed sampling design: the sampling design
`Spot.Draw.deriveSamplingDesign` derives, or else the expanded rule's (`Spot.Draw.flockSamplingDesign` at
`Spot.Draw.derive`'s), as `Spot.OneStage.sampledFrom` reads it before `Spot.Draw.sampleUnits`
samples. -/
def samplingDesignAt (v : _root_.Spot.Docket.Verifier) (n : ℕ) : Flock.V Flock.Draw.SamplingDesign := do
  let samplingDesignJ ← _root_.Spot.Draw.derive v.program v.query v.samplingDesign v.workTable
  let derived ← _root_.Spot.Draw.deriveSamplingDesign v.program v.query v.samplingDesign v.workTable
  pure (derived.getD (← _root_.Spot.Draw.flockSamplingDesign samplingDesignJ n))

/-- **`A` is the session v2 sets up on the parameters `θ` and the committed files `F`**: v2's parameters in the
composite's (`ParamsAt`), which name no members, so the session is the one statement's (`single`, `OneStatement`; a
window of members is `Window`'s), with the verifier's partition object and program as verity-verify-proof's inputs, the sampling design the sample
reads over the population (`samplingDesignAt`), and the statement's descriptor and the registration's query the verifier's program
(`Spot.Program.parse`) and query, so the registration's population is the verifier's, and the sample's bytes
the coins of the parameters' challenger family, as many as the coin commitment's families give it (`Vocab.families`),
`COIN_BYTES` bytes each; the statement's tags, one table, no work, stratified or closure input and no recomputation of
public input words, as `Spot.OneStageZk.inputsOf` hands them; the committed circuit and public files, and the
vocabulary's reading of the files (`Vocab.read`) as the session's, its lookup tables included; and the claim's share,
the published certificate's. The device's bytes and the bound's parameters are the deployment's session's
(`Deployed`). -/
structure SetsUp (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (A : Live) : Prop where
  params : ∃ p, ParamsAt θ p ∧ A.I.partition = some p.own.partition ∧ A.I.program = some p.own.program ∧
    samplingDesignAt p.base.verifier A.n = .ok A.design ∧
    _root_.Spot.Program.parse ⟨A.desc.toArray⟩ = .ok p.base.verifier.program ∧
    A.qr = p.base.verifier.query ∧
    ∃ key label m, p.base.challenger = some (key, label) ∧ (label, m) ∈ V.families θ ∧
      A.Lb 0 = m * _root_.Spot.Challenger.COIN_BYTES
  tags : Flock.Tags.find? _root_.Spot.OneStageZk.STATEMENT = some A.I.tags
  count : A.I.count = 1
  work : A.I.work = none
  stratified : A.I.stratified = none
  closure : A.I.closure = none
  own : A.own = none
  circuit : A.I.circuitFile = V.circuitFile θ F
  publicFile : A.I.publicFile = V.publicFile θ F
  read : V.read θ F = Verity.ExecutionPhaseFiles.one A.fileRead
  share : (V.certOf θ).ε = A.ε
  single : OneStatement θ

/-- **The first of `F` whose SHA-512's hex is `sha`**, as `Spot.OneStageZk.tablesOf` finds a pinned table
on the blobs (`tablesOf_ofArray`). -/
def tableFile (sha : String) (F : Array ByteArray) : Option ByteArray :=
  F.find? fun b => Flock.hexOf (Flock.Sha512.hash b) == sha

theorem tablesOf_ofArray (F : Array ByteArray) :
    _root_.Spot.OneStageZk.tablesOf (_root_.Core.Blobs.ofArray F) =
      Flock.MUFU_TABLES.filterMap fun t => (tableFile t.sha512 F).map fun b => (t.name, b) := by
  unfold _root_.Spot.OneStageZk.tablesOf _root_.Core.Blobs.ofArray tableFile
  congr 1
  funext t
  rw [Array.find?_map, Option.map_map]
  rfl

theorem tableFile_hash {sha : String} {F : Array ByteArray} {a : ByteArray} (h : tableFile sha F = some a) :
    Flock.hexOf (Flock.Sha512.hash a) = sha := by
  unfold tableFile at h
  have h' := Array.find?_some h
  exact beq_iff_eq.1 h'

/-- **The pinned tables' registered files**: the file the deployment registers at a pinned digest `sha`
(`Flock.MUFU_TABLES`), where it registers one, each of that digest. It is the deployment's data, as the files a
registration names by digest are: no definition here picks a file from its digest. -/
structure PinnedTables where
  file : String → Option ByteArray
  hash : ∀ sha b, file sha = some b → Flock.hexOf (Flock.Sha512.hash b) = sha

/-- **The pinned files held by a deployment**: lookup by the SHA-512 hex digest the circuit names. The bytes are
deployment data; the proof that a successful lookup has the requested digest follows from `find?`, without evaluating
SHA-512 in a theorem. -/
def PinnedTables.ofFiles (files : Array ByteArray) : PinnedTables where
  file sha := tableFile sha files
  hash _ _ h := tableFile_hash h

/-- **No pinned table registered**: the registered files of a deployment that registers none. -/
def PinnedTables.empty : PinnedTables := .ofFiles #[]

/-- **The lookup tables a circuit file registers**: each pinned table its META looks up
(`Spot.OneStageZk.lookupsOf`, none where it refuses), at its registered file (`PinnedTables.file`). -/
def pinnedTables (T : PinnedTables) (circuitFile : ByteArray) : List (String × ByteArray) :=
  let names := match _root_.Spot.OneStageZk.lookupsOf circuitFile with
    | .ok ns => ns
    | .error _ => []
  Flock.MUFU_TABLES.filterMap fun t =>
    if names.contains t.name then (T.file t.sha512).map (fun b => (t.name, b)) else none

def TablesSent (T : PinnedTables) (s : Verity.Sent) : Prop :=
  ∀ t ∈ Flock.MUFU_TABLES, ∀ a b, tableFile t.sha512 s.blobs = some a → T.file t.sha512 = some b → a = b

structure DigestsAt (V : Vocab d α) (T : PinnedTables) (θ : ByteArray) : Prop where
  circuit : ∀ p, ParamsAt θ p → ∀ dg, Flock.unhex p.own.circuit = .ok dg →
    V.programDigest θ = some fun i => dg.data.getD i.val 0
  publicFile : ∀ p, ParamsAt θ p → ∀ dg, Flock.unhex p.own.pub = .ok dg →
    V.publicDigest θ = some fun i => dg.data.getD i.val 0
  tables : ∀ F, (V.read θ F).first.tables = pinnedTables T (V.circuitFile θ F)

/-- **v2's parameters in `θ`**, as `ParamsAt` reads them, where they decode (`paramsOf_eq`). -/
def paramsOf (θ : ByteArray) : Option _root_.Spot.OneStageZk.Params :=
  match _root_.Core.canonicalJson θ >>= _root_.Core.Composition.decodeParams with
  | .error _ => none
  | .ok ps =>
    match ps.find? (·.id == VStar.Recursive.Sampled.PROTOCOL) with
    | none => none
    | some x =>
      match _root_.Core.canonicalJson x.params.toUTF8 >>= VStar.Recursive.Sampled.paramsOfJson with
      | .ok (.inl p) => some p
      | _ => none

theorem paramsOf_eq {θ : ByteArray} {p : _root_.Spot.OneStageZk.Params} (h : ParamsAt θ p) :
    paramsOf θ = some p := by
  obtain ⟨ps, hps, x, hx, hp⟩ := h
  simp only [paramsOf, hps, hx, hp]

/-- **The digest a hex digest names**: its bytes, where `Flock.unhex` decodes it, as `DigestsAt` reads them. -/
def digestOf (s : String) : Option Hash.Digest :=
  match Flock.unhex s with
  | .ok dg => some fun i => dg.data.getD i.val 0
  | .error _ => none

/-- **The circuit file's digest the parameters `θ` hold, as deployed**: the one v2's parameters name. -/
def programDigestOf (θ : ByteArray) : Option Hash.Digest := (paramsOf θ).bind fun p => digestOf p.own.circuit

/-- **The public file's digest the parameters `θ` hold, as deployed**: the one v2's parameters name. -/
def publicDigestOf (θ : ByteArray) : Option Hash.Digest := (paramsOf θ).bind fun p => digestOf p.own.pub

/-- **`verify --zk`'s tags at the inputs v2 hands it** (`zkTags`): the statement's tags `tags`, one table and the
circuit file `circuitFile`. -/
def zkTagsOf (tags : Flock.Tags) (circuitFile : ByteArray) : Flock.Tags :=
  Flock.Zk.tags tags (Flock.Zk.maskLogOf circuitFile) 1

/-- **v2's reading of the committed files `F` at the parameters `θ`, as deployed**: the files the digests v2's
parameters name (`programDigestOf`, `publicDigestOf`), the lookup tables the circuit file registers (`pinnedTables`),
and `verify --zk`'s tags at the statement's tags and the circuit file (`zkTagsOf`). Where the statement has no tags v2
refuses, and the tags read are a default. -/
noncomputable def readOf (T : PinnedTables) (θ : ByteArray) (F : Array ByteArray) : Verity.FileRead :=
  ⟨zkTagsOf ((Flock.Tags.find? _root_.Spot.OneStageZk.STATEMENT).getD default)
      (Verity.namedFile (programDigestOf θ) F),
    Verity.namedFile (programDigestOf θ) F, pinnedTables T (Verity.namedFile (programDigestOf θ) F),
    Verity.namedFile (publicDigestOf θ) F⟩

theorem digestsAt_of (V : Vocab d α) (T : PinnedTables) (hc : ∀ θ, V.programDigest θ = programDigestOf θ)
    (hp : ∀ θ, V.publicDigest θ = publicDigestOf θ)
    (hr : ∀ θ F, V.read θ F = Verity.ExecutionPhaseFiles.one (readOf T θ F)) (θ : ByteArray) :
    DigestsAt V T θ where
  circuit p hθ dg hdg := by
    rw [hc, programDigestOf, paramsOf_eq hθ]
    simp only [Option.bind, digestOf, hdg]
  publicFile p hθ dg hdg := by
    rw [hp, publicDigestOf, paramsOf_eq hθ]
    simp only [Option.bind, digestOf, hdg]
  tables F := by rw [hr, Vocab.circuitFile, hc]; rfl

theorem readOf_eq (V : Vocab d α) (T : PinnedTables) (hc : ∀ θ, V.programDigest θ = programDigestOf θ)
    (hp : ∀ θ, V.publicDigest θ = publicDigestOf θ) {θ : ByteArray} {F : Array ByteArray} {A : Live}
    (htags : Flock.Tags.find? _root_.Spot.OneStageZk.STATEMENT = some A.I.tags) (hcount : A.I.count = 1)
    (hcirc : A.I.circuitFile = V.circuitFile θ F) (hpub : A.I.publicFile = V.publicFile θ F)
    (htab : A.I.tables = pinnedTables T (V.circuitFile θ F)) :
    readOf T θ F = A.fileRead := by
  have hc' : V.circuitFile θ F = Verity.namedFile (programDigestOf θ) F := by rw [Vocab.circuitFile, hc]
  have hp' : V.publicFile θ F = Verity.namedFile (publicDigestOf θ) F := by rw [Vocab.publicFile, hp]
  simp only [readOf, Live.fileRead, zkTagsOf, FlockSoundness.Discharge.ZkExec.zkTags, htags, Option.getD_some, hcount,
    hcirc, hpub, htab, hc', hp']

/-- **v2's case in what the prover sent**: v2's parameters in the parameters sent, as `ParamsAt` reads them, and the
registration and the evidence of the case the composition's evidence names under the sampled component's id, each read
by its decoder in its one-stage form. -/
def caseOf (s : Verity.Sent) : Flock.V (_root_.Spot.OneStageZk.Params ×
    _root_.Spot.OneStage.Commitment × _root_.Spot.OneStageZk.Evidence) := do
  let ps ← _root_.Core.canonicalJson s.params >>= _root_.Core.Composition.decodeParams
  let some x := ps.find? (·.id == VStar.Recursive.Sampled.PROTOCOL) | throw "the parameters name no sampled proofs"
  let ev ← _root_.Core.Composition.decodeEvidence s.evidence
  let some y := ev.find? (·.id == VStar.Recursive.Sampled.PROTOCOL) | throw "the evidence holds no sampled proofs case"
  let .inl p ← _root_.Core.canonicalJson x.params.toUTF8 >>= VStar.Recursive.Sampled.paramsOfJson
    | throw "the sampled proofs parameters' form is not one-stage"
  let c ← _root_.Spot.OneStage.commitmentOf y.commitment
  let .inl e ← VStar.Recursive.Sampled.evidenceOf y.evidence
    | throw "the sampled proofs evidence's form is not one-stage"
  pure (p, c, e)

/-- The challenger session's release log in v2's evidence, a line each, as `Spot.Challenger.trustedTreesOf`
reads it. -/
def linesOf (e : _root_.Spot.OneStageZk.Evidence) : Flock.V (List String) := do
  let .arr ls := _root_.Spot.getD (_root_.Spot.getD e.base.coins "session") "log"
    | throw "the session's log is not a list of lines"
  ls.toList.mapM fun | .str l => pure l | _ => throw "a line of the session's log is not a string"

/-- **The coins a `--zk` session's record gives round `k` of stream `stream`**, as
`Spot.Challenger.sessionTreeReason` reads them: its link points at the session stream's round 0, else that
stream's round `k`. -/
def recordRound (record : Lean.Json) (stream : String) (k : ℕ) : Option ByteArray :=
  if stream == _root_.Spot.Challenger.TREE_SESSION && k == 0 then
    match _root_.Spot.getD (_root_.Spot.getD record "link") "points" with
    | .arr ps => _root_.Spot.Challenger.recordCoins?
        (.arr (ps.foldl (fun acc p => match p with | .arr cs => acc ++ cs | _ => acc.push .null) #[]))
    | _ => none
  else
    match _root_.Spot.getD record "streams" with
    | .arr ss =>
      (ss.find? fun x => match _root_.Spot.getD x "stream" with | .str n => n == stream | _ => false).bind
        fun x => match _root_.Spot.getD x "rounds" with
          | .arr rs =>
            rs[k]?.bind fun r => _root_.Spot.Challenger.recordCoins? (_root_.Spot.getD r "coins")
          | _ => none
    | _ => none

/-- The root of the coin tree a session's record names. -/
def recordTree (record : Lean.Json) : String :=
  match _root_.Spot.getD (_root_.Spot.getD record "coin_tree") "root" with
  | .str r => r
  | _ => ""

/-- A coin tree's spec as its line names it. -/
def specOf (j : Lean.Json) : Option Flock.CoinTree.Spec := do
  let .arr ss := _root_.Spot.getD j "streams" | none
  let streams ← ss.mapM fun | .str x => some x | _ => none
  pure ⟨streams, ← _root_.Spot.nat? (_root_.Spot.getD j "rounds_log"),
    ← _root_.Spot.nat? (_root_.Spot.getD j "block")⟩

/-- **The family and the root a root line of the release log shows**, as `Spot.Challenger.releasedRootsOf`
reads them; `none` where the line doesn't parse or names no family or no root. -/
def rootOf (l : String) : Option (String × ByteArray) := do
  let r ← (_root_.Spot.Challenger.event l).toOption
  let .str label := _root_.Spot.getD r "label" | none
  let .str root := _root_.Spot.getD r "root" | none
  pure (label, ← (Flock.unhex root).toOption)

/-- The streams a coin tree's spec names, as `Spot.Challenger.coinTreesOf` reads them: none where it names no
list. -/
def specStreams (spec : Lean.Json) : Array Lean.Json :=
  match _root_.Spot.getD spec "streams" with | .arr ss => ss | _ => #[]

/-- **A coin tree's line's release, read with the tests `Spot.Challenger.coinTreesOf` makes**, after the trees
`ts` read so far, each its root and its spec as its line gives them: where the line's `event` is `TREE_EVENT` by
`coinTreesOf`'s comparison, its hello after its nonce, with its root's and key's bits; else a round opened, of the
`i`-th tree read and the `s`-th of its spec's streams, the first one the line's names by the test `coinTreesOf`'s
`any` makes (`Spot.jsonStrIs`), after the bytes it answers, with its coins' bits where a record of that tree
gives them, else none; `none` where the line doesn't read as either. -/
def treeStep (records : List Lean.Json) (ts : List (String × Lean.Json)) (d : Lean.Json) :
    Option (Verity.Release × List (String × Lean.Json)) :=
  if _root_.Spot.jsonStrIs (_root_.Spot.getD d "event")
      _root_.Spot.Challenger.TREE_EVENT then do
    let .str key := _root_.Spot.getD d "key" | none
    let .str nonce := _root_.Spot.getD d "nonce" | none
    let .str root := _root_.Spot.getD d "root" | none
    let spec ← specOf (_root_.Spot.getD d "spec")
    let ν ← (Flock.unhex nonce).toOption
    let r ← (Flock.unhex root).toOption
    let κ ← (Flock.unhex key).toOption
    pure (⟨.hello spec, ν, Verity.bitsOfBytes (r ++ κ)⟩, ts ++ [(root, _root_.Spot.getD d "spec")])
  else do
    let .str root := _root_.Spot.getD d "tree" | none
    let .str stream := _root_.Spot.getD d "stream" | none
    let .str aft := _root_.Spot.getD d "after" | none
    let k ← _root_.Spot.nat? (_root_.Spot.getD d "round")
    let n ← _root_.Spot.nat? (_root_.Spot.getD d "n")
    let i ← ts.findIdx? (·.1 == root)
    let (_, spec) ← ts[i]?
    let s ← (specStreams spec).findIdx? (_root_.Spot.jsonStrIs · stream)
    let b ← (Flock.unhex aft).toOption
    let coins := match records.find? (recordTree · == root) with
      | some rec => ((recordRound rec stream k).map Verity.bitsOfBytes).getD []
      | none => []
    pure (⟨.opening i s n, b, coins⟩, ts)

/-- **The releases the lines after the release log's head show**, for the sample of family `label` after the
registration's digest `after`, with the sample's coins `ξ` still to place, the trees `ts` read so far and the sessions'
records: a coin tree's line (`Spot.Challenger.treeLine`) its release (`treeStep`), and the other lines in
pairs, as `Spot.Challenger.releasedRootsOf` reads them, a root's line `pending` and its coin's release, of the
family and after the root the root's line shows (`rootOf`), with the next coin of `ξ` where it is the sample's
(`label`'s, after `after`), else none. `none` where a line doesn't read as it is read or the log ends on a root without
its release, as v2 refuses that log. -/
def readTail (label : String) (after : ByteArray) (records : List Lean.Json) :
    List String → List (String × Lean.Json) → Option String → ByteArray → Option (List Verity.Release)
  | [], _, none, _ => some []
  | [], _, some _, _ => none
  | l :: ls, ts, pending, ξ =>
    if _root_.Spot.Challenger.treeLine l then do
      let (r, ts') ← (_root_.Spot.Challenger.event l).toOption.bind (treeStep records ts)
      pure (r :: (← readTail label after records ls ts' pending ξ))
    else
      match pending with
      | none => readTail label after records ls ts (some l) ξ
      | some a => do
        let (lab, root) ← rootOf a
        if lab = label ∧ root = after then
          let m := _root_.Spot.Challenger.COIN_BYTES
          pure (⟨.fresh label, after, Verity.bitsOfBytes (ξ.extract 0 m)⟩ ::
            (← readTail label after records ls ts none (ξ.extract m ξ.size)))
        else pure (⟨.fresh lab, root, []⟩ :: (← readTail label after records ls ts none ξ))

/-- **The releases the release log's lines show, read with v2's own tests**: its head's coin tree where the head is a
coin tree's line (`Spot.Challenger.coinTreesOf` reads the head's line too), then the lines after it
(`readTail`); the head in no pair, as `Spot.Challenger.releasedRootsOf` reads it. -/
def readLog (label : String) (after : ByteArray) (records : List Lean.Json) (ξ : ByteArray) :
    List String → Option (List Verity.Release)
  | [] => some []
  | l :: ls =>
    if _root_.Spot.Challenger.treeLine l then do
      let (r, ts) ← (_root_.Spot.Challenger.event l).toOption.bind (treeStep records [])
      pure (r :: (← readTail label after records ls ts none ξ))
    else readTail label after records ls [] none ξ

/-- A session's record on the blobs `bs`, as `Spot.OneStageZk.treesHeld` reads it (`Flock.JsonParse.parse`):
one that doesn't parse is `null`, whose coin tree `Spot.Challenger.sessionTreeReason` finds among no trees, so
v2 refuses its session; `none` where no blob holds it or it isn't UTF-8. -/
def recordOn (bs : _root_.Core.Blobs) (r : _root_.Spot.OneStageZk.RunFiles) : Option Lean.Json := do
  let b ← (bs.get r.record s!"session {r.name}'s record").toOption
  let t ← String.fromUTF8? b
  pure ((Flock.JsonParse.parse t).toOption.getD Lean.Json.null)

/-- The records of the sessions v2's evidence `e` names, on the blobs `bs`, as `Spot.OneStageZk.treesHeld`
reads them (`recordOn`). -/
def recordsOn (bs : _root_.Core.Blobs) (e : _root_.Spot.OneStageZk.Evidence) : Option (List Lean.Json) :=
  e.sessions.toList.mapM (recordOn bs)

/-- **The device's log as v2's check reads it, with v2's own tests**, from what the prover sent: the releases the
challenger session's log in v2's evidence shows, at the parameters' challenger family and the registration's digest,
with the coins v2's sample reads (`Spot.OneStageZk.coinsOf`): each coin tree's line read by the tests
`Spot.Challenger.coinTreesOf` makes, the head's line included (`readLog`), and the sessions' records as
`Spot.OneStageZk.treesHeld` reads them (`recordsOn`). A release whose coins v2 doesn't open has none; where a
line doesn't read as it is read, or the log ends on a root without its release, the read shows no log. -/
def coinsRead (s : Verity.Sent) : Option (List Verity.Release) := do
  let (p, c, e) ← (caseOf s).toOption
  let (_, label) ← p.base.challenger
  let ξ ← (_root_.Spot.OneStageZk.coinsOf p c e).toOption
  let lines ← (linesOf e).toOption
  readLog label c.digest (← recordsOn (_root_.Core.Blobs.ofArray s.blobs) e) ξ lines

def openLine (l : String) : Bool :=
  match _root_.Spot.Challenger.event l with
  | .ok j => _root_.Spot.jsonStrIs (_root_.Spot.getD j "event")
      _root_.Spot.Challenger.OPEN_EVENT
  | .error _ => false

def linkLine (l : String) : Bool :=
  match _root_.Spot.Challenger.event l with
  | .ok j =>
    _root_.Spot.jsonStrIs (_root_.Spot.getD j "event") _root_.Spot.Challenger.OPEN_EVENT &&
      _root_.Spot.jsonStrIs (_root_.Spot.getD j "stream")
        _root_.Spot.Challenger.TREE_SESSION
  | .error _ => false

def sampleLinkFirstFrom (label : String) (after : ByteArray) : Bool → Option String → List String → Bool
  | _, _, [] => true
  | opened, pending, l :: ls =>
    if _root_.Spot.Challenger.treeLine l then
      if opened = false ∧ openLine l = true ∧ linkLine l = false then false
      else sampleLinkFirstFrom label after (opened || openLine l) pending ls
    else match pending with
      | none => sampleLinkFirstFrom label after opened (some l) ls
      | some a =>
        if opened = true ∧ rootOf a = some (label, after) then false
        else sampleLinkFirstFrom label after opened none ls

def sampleLinkFirst (label : String) (after : ByteArray) (log : List String) : Bool :=
  sampleLinkFirstFrom label after false none log

/-- The coin trees' specs the device's log `log` shows sampled, in the order sampled. -/
def helloSpecs (log : List Verity.Release) : List Flock.CoinTree.Spec :=
  log.filterMap fun r => match r.frame with
    | .hello spec => some spec
    | _ => none

def SampleLinkFirst (label : String) (after : ByteArray) (log : List Verity.Release) : Prop :=
  ∀ pre r post, log = pre ++ r :: post → ∀ i s n, r.frame = .opening i s n →
    (∀ q ∈ post, ¬ (q.frame = .fresh label ∧ q.answers = after)) ∧
      ((∀ q ∈ pre, ∀ i' s' n', q.frame ≠ .opening i' s' n') →
        ((helloSpecs pre)[i]?.bind fun sp => sp.streams[s]?) = some _root_.Spot.Challenger.TREE_SESSION)

/-- **The deployment's sessions `S` cover the sampled verdict's one-stage form at composite `k`**; parameters naming the
recursive form (`VStar.Recursive`) are not this target's. -/
def Covers (V : Vocab d α) (k : Verity.Registration.Composite) (S : ByteArray → Array ByteArray → Option Live) :
    Prop :=
  ∀ θ F, S θ F = none → (∃ p, ParamsAt θ p) → ∀ (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)),
    V.finderCR .sampled θ F c σ → V.FinderCR .sampled θ F c σ →
      Verity.Game.prob (fun o => V.Verdict k .sampled o) (V.game θ F c) σ ≤ V.delta θ .sampled (V.cost θ F c σ)

structure Deployed (V : Vocab d α) (K : Constructions V) (S : ByteArray → Array ByteArray → Option Live) :
    Prop where
  setsUp : ∀ θ F A, S θ F = some A → SetsUp V θ F A
  coinsRead : ∀ s, V.coinsRead .sampled s = Verity.Sampled.coinsRead s
  finder : ∀ θ F c σ, V.finder .sampled θ F c σ = K.readFinder θ F c σ
  forger : ∀ θ F c σ, V.forger .sampled θ F c σ = K.forger θ F c σ
  loss : ∀ θ x, V.loss .sampled θ x = 0
  extract : ∀ θ F A, S θ F = some A → ∀ (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)),
    HEq (V.extract θ F c σ) ((K.live A θ F c σ).extract (K.project A θ F c σ))
  opens : ∀ θ F A, S θ F = some A → ∀ c, A.OpensUnder (V.read θ F) (V.Opens θ F c)
  finderCR : ∀ θ F A, S θ F = some A → ∀ (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)),
    V.finderCR .sampled θ F c σ → V.premises .sampled θ F c σ → FinderCR V K A θ F c σ
  premises : ∀ θ F (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)),
    V.premises .sampled θ F c σ ↔
      ∀ A, S θ F = some A → (K.live A θ F c σ).FinderCR (K.project A θ F c σ) (V.cost θ F c σ)
  delta : ∀ θ F A, S θ F = some A → ∀ t, A.delta t (V.crBound θ .sampled t) ≤ V.delta θ .sampled t

end Verity.Sampled
