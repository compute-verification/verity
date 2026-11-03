import Properties.Verity.Game
import Properties.SampledProofs.Draw
import Proofs.Flock.Bridge.V5.Statement
import Proofs.Flock.Bridge.CircuitOf
import Proofs.Flock.Soundness.ZK.RealLeaves

/-!
# Sampled proofs' leg of `Verity.Soundness`: the session v2 sets up

The leg is `Verity.Vocab.SampledProofsSound` at a composite whose sampled component is v2
(`SampledProofs.OneStageZk.protocol`), from C-Flock's knowledge soundness of the live session at public inputs on the
event that the verifiers accept the files (`SampledKSLiveV5` with its file and population-order hypotheses read off
acceptance), applied at the session v2 sets up on the parameters and the committed files and at the strategy a
strategy of the period's game projects to. This module states its C-Flock side.

* **The session** (`Live`): V5's binders, each the value v2's check derives on the parameters and the committed files,
  with three of V5's hypotheses as fields (A3 at the device's bytes, `TagsOkZHJ`, and `RecordCustodyZKJP` at the writer
  whose record v2 decodes), and the bound's parameters. The circuit names no public input port: v2 hands `flock-verify`
  no recomputation of public input words, so `HmIn.checkOwn` refuses any circuit that names one, and the session's ports
  are `HmIn.parse`'s empty ones.
* **The draw** (`Live.L`): the law the parameters' selection rule derives over the population (`law`), drawn as
  `SampledProofs.Select.select` draws it, by `SampledProofs.Draw.drawUnits` on the device's one byte stream (`src`); a
  draw that fails draws every unit. At a subset law it is V5's `Law.execOS` (`drawUnits_subset_runOS`).
* **The live audit at the session** (`Live.game`), and V5's simulator and extractor at any strategy of it (`Live.sim`,
  `Live.Xr`). The simulator and the extractor hold at every law (`decodesZCJH_custodyP`).
* **The claim's transcript at a strategy** (`Live.extract`): V5's trace fixed at registration, opened at the deployed
  commitment and read at the circuit's gates with its constants, on the committed files' circuit. Without public input
  ports V5's key program is the claim's (`keyProgP_empty`), and the files the session reads parse to it
  (`Live.circuit_eq`).
* **The projection** (`Constructions`): a strategy of the period's game is read as a strategy of the live audit. A
  C-Flock strategy flips no coin of its own, so the device's other releases the strategy may ask for (`Widening`:
  other families' coins, other trees' samples) widen the draw's sample (`Live.extend`), which the draw doesn't read: the
  draw's law, A3 and the leg's δ are the session's.
* **The coupling** (`CouplingTarget`), the projection's target: the sampled verdict is at most as likely in the period's
  game as flock-verify `--zk`'s acceptance in the live audit at the projected strategy, plus the two-continuation
  finders' terms, the tree openings' hiding term, the read finder's collision and the forger's forgery.
* **The component** (`ReadsV2`): the composite's `.sampled` component is v2. Both composites name v1
  (`Verity.component`), whose verdicts are the prover's, so the leg and the coupling take it as a hypothesis.
* **The session v2 sets up on the parameters and the committed files** (`SetsUp`): v2's parameters in the composite's
  (`ParamsAt`), the session's inputs as v2 hands them to flock-verify (the statement, one table, the verifier's
  partition object and program, no recomputation of public input words), the committed circuit and public files and
  the vocabulary's reading of them, the law the draw reads over the population (`lawAt`), and the claim's share.
* **The vocabulary's sampled fields at the deployment's sessions** (`Deployed`, `Covers`): at most one session at each
  parameters and committed files, each one v2 sets up, with the deployment's writer, device bytes and bound
  parameters. Soundness's `Vocab` reads the constructions' read finder and forger at `.sampled`, no loss term, and at
  each session the extractor at the projected strategy, the leg's per-finder premises in `error`'s condition, and a
  published `δ` at least the session's. Where there is no session the sampled verdict never accepts.
* **The per-finder SHA-512 premises at a strategy and a budget** (`Live.FinderCR`): `cr/sha-512` at each table's
  knowledge finders (`TableCRZ`: the fork, pads, τ and level-0 finders at `2t + 2v`, the self-clash finder at `t + v`)
  and `ecr/sha-512` at the link finders (`LinkCRZC`, at `t + w` per run), at `t` the strategy's SHA-512 calls.
* **The leg's δ at the session** (`Live.delta`), each finder's term its own summand:
  * the draw's term (`drawMiss`): the law's own miss at the claim's share `ε`, the largest probability that the draw
    holds no proof block of a set of more than an `ε` share of them, the acceptance probability of the law's one-stage
    design over the population's strata (`Claims.Sampling.OneStage.Pa`). The law and the strata define it, and at the
    subset law of `kd` units it is `C(n − K₀, kd)/C(n, kd)` with `K₀ = ⌊εn⌋ + 1`;
  * the tables' knowledge term at the worst registration (`ksAvgStrictZ`) and the link term at `t + w` per run
    (`sessionDelta`);
  * each round the device opens after a digest of the prover's message, and the record the commitment is a digest of:
    a two-continuation finder (`twoRun`), which runs the play twice after the digest and outputs the two messages the
    evidence reveals when they differ, at `2t + 2h` SHA-512 calls. The most likely message misses with probability at
    most the square root of the finder's success, so each adds `√((2t + 2h)²/2^513)`;
  * each tree opening the session reads, twice: the opened coins against fresh ones, at the mean over the device's
    fresh tree key of hm96's hiding gap (`CROnly.hidingGap`), which the leftover hash lemma bounds by `2^-257` a leaf
    (`hidingTerm`), so `Claims.Hm96Hiding` holds at it with no premise (`CROnly.hm96Hiding_gap`);
  * the circuit file's and the public file's finders (`Vocab.fileFinder`, `Vocab.publicFinder`), `2²/2^513` each;
  * the read finder's published bound at `t` (`cr`, `Vocab.crBound`).
-/

namespace Verity.Sampled

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

/-- **The session v2 sets up on the parameters and the committed files**: V5's binders, each the value v2's check
derives (the files' parse, the public file over the circuit's empty public input ports, the population and its strata,
the registration's descriptor and query, the law and the draw's size, and the session's plan, writer and digest), the
device's byte stream the draw reads, A3 at those bytes, the verify command's tags and `live-verifier` at the writer as
fields, and the bound's parameters. -/
structure Live where
  /-- the population: the public file's proof blocks -/
  n : ℕ
  /-- what the prover registers before the draw -/
  Reg : Type
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
  hn : pub.n = n
  hnO : pubO.n = n
  /-- the statement's descriptor -/
  desc : Claims.Msg
  /-- the registration's query -/
  qr : Lean.Json
  /-- the population's strata sizes, in slot order: the registration's templates -/
  ns : List ℕ
  hns : ns.sum = n
  /-- the law the parameters' selection rule derives over the population (`SampledProofs.Select`) -/
  law : Flock.Draw.Law
  /-- the draw's size -/
  kd : ℕ
  hkd : kd ≤ n
  hdiv : kd % nTab I = 0
  dj : Finset (Fin n) → Lean.Json
  S₀ : Finset (Fin n)
  d₀ : Lean.Json
  y₀ : DrawSetupZHJP (inputsJ I 0) own (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀
  wr : (S : Finset (Fin n)) → (R : Reg) →
    LiveOutJ (planZHJP (Reg := Reg) dj y₀ S R).length (mPtsOf D.unroll.g (kd / nTab I)) →
      FlockSoundness.Discharge.Exec.Transcript
  dg : ByteArray → ByteArray
  N : ℕ
  /-- the device's bytes the draw reads -/
  Ω : Type
  [fΩ : Fintype Ω]
  [nΩ : Nonempty Ω]
  Lb : Fin 1 → ℕ
  src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256
  hA3 : FlockSoundness.Assumptions.UniformRandomBytes Lb src
  hTags : TagsOkZHJ I
  /-- `live-verifier` at the writer as V5 reads it, the record's draw a subset draw of `kd` units
  (`DrawFileZK dj (.subset kd) n`) -/
  hCust : FlockSoundness.Assumptions.Zk.RecordCustodyZKJP I own dj
    (fun S R => (planZHJP (Reg := Reg) dj y₀ S R).length) wr kd
  k : ℕ
  Rw : ℕ
  M : ℕ
  hk : 1 ≤ k
  hRw : 1 ≤ Rw
  hM : 1 ≤ M
  ρ : ℝ
  hρ : ρ < 1
  hr : ∀ S R j, rateZ (planZHJP (Reg := Reg) dj y₀ S R) j k ≤ ρ
  /-- the claim's share: at most an `ε` share of the proof blocks unbound -/
  ε : ℝ
  /-- one opening's verification, in SHA-512 calls -/
  v : ℕ
  /-- one acceptance check, in SHA-512 calls -/
  w : ℕ
  /-- the tree openings the session reads after a digest of the prover's message: its link points' and each rep
  round's -/
  rounds : ℕ
  /-- hashing one message the device or the commitment takes a digest of (a round's message, the record), in SHA-512
  calls -/
  h : ℕ

attribute [instance] Live.fΩ Live.nΩ

/-- **The widening of the draw's sample a projection reads**: a finite type of the device's other releases, those a
strategy of the period's game may ask for that the session doesn't read (other families' coins, other trees' samples). A
C-Flock strategy flips no coin of its own, so the projected strategy reads them off the draw's sample, widened by them
(`Live.extend`). -/
structure Widening where
  T : Type
  [fT : Fintype T]
  [nT : Nonempty T]

attribute [instance] Widening.fT Widening.nT

/-- A3 at a sample widened by a second uniform coin that the bytes don't read. -/
theorem uniform_fst {Ω T : Type} [Fintype Ω] [Fintype T] [Nonempty T] {m : ℕ} {Lb : Fin m → ℕ}
    {t : Ω → (i : Fin m) → Fin (Lb i) → Fin 256} (h : FlockSoundness.Assumptions.UniformRandomBytes Lb t) :
    FlockSoundness.Assumptions.UniformRandomBytes Lb (fun ω : Ω × T => t ω.1) := fun E =>
  (FlockSoundness.ZK.prCoin_fst (fun ω => E (t ω))).trans (h E)

namespace Live

variable (A : Live)

/-- **The session with the draw's sample widened by the coins `S`**: the draw reads only the session's bytes, so the
draw's law is the session's and A3 holds at the widened sample (`uniform_fst`). -/
noncomputable def extend (S : Widening) : Live where
  __ := A
  Ω := A.Ω × S.T
  fΩ := inferInstance
  nΩ := inferInstance
  src ω := A.src ω.1
  hA3 := uniform_fst A.hA3

/-- **The draw's law**: the session's law over the population as `SampledProofs.Select.select` draws it,
`SampledProofs.Draw.drawUnits` on the device's bytes in order; a draw that fails draws every unit. -/
noncomputable def L : FlockSoundness.Audit.Law A.n where
  Ω := A.Ω
  draw ω :=
    match _root_.SampledProofs.Draw.drawUnits A.law A.n (_root_.Properties.SampledProofs.bytesOf (A.src ω 0)) with
    | .ok (d, _) => Finset.univ.filter fun u => u.val ∈ d
    | .error _ => Finset.univ

/-- **C-Flock's live audit at the session**, the game V5 bounds: the prover registers, the device's bytes draw, and
the live session runs at the draw. -/
noncomputable def game :=
  auditLive A.L A.Reg (liveOfZJ A.dg (planZHJP A.dj A.y₀) A.N A.wr)

/-- V5's circuit: the key program's at the session's ports, with its outputs. -/
noncomputable abbrev Cc :=
  (keyProgP A.D.unroll {} A.hU A.n).circuit (FlockSoundness.Discharge.Layout.outsOfP A.D.unroll A.hU A.n {})

/-- V5's partition: the key program's at the session's ports, and its outputs. -/
noncomputable def P :=
  (keyProgP A.D.unroll {} A.hU A.n).partition (FlockSoundness.Discharge.Layout.outsOfP A.D.unroll A.hU A.n {})

/-- V5's binding at the session's ports. -/
noncomputable def vb :=
  (rsZHJP (Reg := A.Reg) A.dj A.y₀ A.ht (Flock.Desc.parse_flat A.hD) A.hscope A.hU A.hpi A.hpub).bindingZ
    (rsZHJP_computes (Reg := A.Reg) A.dj A.y₀ A.ht (Flock.Desc.parse_flat A.hD) A.hscope A.hU A.hpi A.hpub)

/-- **V5's simulator at a strategy of the live audit**: the strategy in the compiled batched session at flock-verify
`--zk`'s acceptance at `own`, through the decoder `RecordCustodyZKJP` gives at the session's law. -/
noncomputable def sim (τ : FlockSoundness.Game.Strategy A.game) :=
  simZJA execArith hs512 (planZHJP A.dj A.y₀) (AcceptsZKP A.I A.own) A.wr
    (opnAtZHJP (planZHJP A.dj A.y₀) A.I A.own A.dj) A.dg (decAtZJ (planZHJP A.dj A.y₀)) (fun _ _ _ _ => true)
    (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) A.N
    (decodesZCJH_custodyP A.I A.own A.ht (Flock.Desc.parse_flat A.hD) A.kd A.hdiv A.dj A.y₀ A.wr A.L A.hTags A.hCust) τ

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

/-- **The draw's term**: the law's own miss at the claim's share, the largest probability that the draw holds no proof
block of a set of more than an `ε` share of them, the acceptance probability of the law's one-stage design
(`Properties.SampledProofs.oneStageOf`) over the population's strata (`Claims.Sampling.OneStage.Pa`); at the subset law
of `kd` units, `C(n − K₀, kd)/C(n, kd)` with `K₀ = ⌊εn⌋ + 1`. Those sets are exactly the unbound sets that fail the
claim, so no count of them enters; with none (`ε ≥ 1`) the term is `0`. -/
noncomputable def drawMiss : ℝ≥0∞ :=
  ENNReal.ofReal (⨆ B : {B : Finset ℕ // B ⊆ Finset.range A.n ∧ A.ε * A.n < B.card},
    (_root_.Properties.SampledProofs.oneStageOf A.law).Pa A.ns B.1)

open Classical in
/-- **The session's error at `t` calls**, V5's at the worst registration: the draw's term, the tables' knowledge term
at the finders' budgets, and the link term at `t + w` per run. -/
noncomputable def sessionDelta (t : ℕ) : ℝ≥0∞ :=
  A.drawMiss +
    (⨆ R : A.Reg, ksAvgStrictZ (planZHJP A.dj A.y₀) A.L A.k (A.qF t) (A.qS t) R) +
    ENNReal.ofReal ((∑ p : A.vb.Pos,
        (Fintype.card (ΩcZ execArith A.P (planZHJP A.dj A.y₀) (tabZHJP A.dj A.y₀) A.vb A.L p) : ℝ) /
          Fintype.card A.L.Ω) *
      (2 * (A.tL t * (1 + A.k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * A.M) + A.k / (Real.exp 1 * A.Rw)) / (1 - A.ρ))

/-- **A two-continuation finder's term at budget `q`**: the square root of the strict birthday bound `q²/2^513`, since
the most likely revealed message misses with probability at most the root of the finder's success. -/
noncomputable def twoRun (q : ℝ) : ℝ≥0∞ := ENNReal.ofReal (Real.sqrt (q ^ 2 / 2 ^ (513 : ℝ)))

/-- **The tree openings' hiding term**: each opening the session reads, twice, at `2^-257`, the leftover hash lemma's
bound on the mean over the device's fresh tree key of hm96's hiding gap at one leaf. -/
noncomputable def hidingTerm : ℝ≥0∞ := 2 * A.rounds / 2 ^ 257

/-- **The leg's δ at the session, `t` SHA-512 calls and the read finder's bound `cr`**: the session's error, a
two-continuation term per round opened after a digest and one for the record, the tree openings' hiding term, the two
file finders' `2²/2^513`, and `cr`. -/
noncomputable def delta (t : ℕ) (cr : ℝ) : ℝ≥0∞ :=
  A.sessionDelta t + A.rounds * twoRun (A.qM t) + twoRun (A.qM t) + A.hidingTerm +
    2 * ENNReal.ofReal ((2 : ℝ) ^ 2 / 2 ^ (513 : ℝ)) + ENNReal.ofReal cr

/-- **How v2 reads the committed files at the session**: the verify command's tags, the circuit file, its tables and
the public file. -/
def fileRead : Verity.FileRead := ⟨zkTags A.I, A.I.circuitFile, A.I.tables, A.I.publicFile⟩

/-- **The committed files' circuit is the parsed circuit file's** at the public file's population. -/
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
commitment's roots and read at the circuit's gates with its constants (`Flock.Bridge.transcriptOf`), on the committed
files' circuit (`circuit_eq`). -/
noncomputable def extract (τ : FlockSoundness.Game.Strategy A.game) : A.fileRead.circuit.Transcript :=
  cast (congrArg Verity.Circuit.Transcript A.circuit_eq.symm)
    (Flock.Bridge.transcriptOf A.D A.hU A.n A.κ
      ((Flock.Bridge.witDep A.Cc A.Cd Claims.Sampling.sha512Msg A.roots (A.Xr τ)).val Claims.Sampling.sha512Msg A.Cd
        (Flock.Bridge.stmtOf A.P A.κ A.kd A.desc ⟨"", 0, []⟩ A.roots)))

end Live

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **The constructions the leg reads, applied to a strategy of the period's game**: the projection, the device's other
releases the strategy may ask for (`coins`) and the C-Flock live strategy it is read as at the session widened by them
(`project`); the read finder (`Vocab.finder`'s: a collision among the strings v2 hashes in reading the device's log and
trees and those the device hashes) and the forger of the device's signature (`Vocab.forger`'s); and the
two-continuation finders, one per round the device opens after a digest of the prover's message and one for the record
the commitment is a digest of, each at most `2t + 2h` SHA-512 calls on every coin (`Live.qM`). -/
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
  msg_calls : ∀ A θ F c σ j (x : (msgFinder A θ F c σ j).S),
    ((msgFinder A θ F c σ j).cost x : ℝ) ≤ A.qM (V.cost θ F c σ)
  record_calls : ∀ A θ F c σ (x : (recordFinder A θ F c σ).S),
    ((recordFinder A θ F c σ).cost x : ℝ) ≤ A.qM (V.cost θ F c σ)

/-- **The session a strategy of the period's game is projected into**: `A` widened by the coins the projection reads. -/
noncomputable abbrev Constructions.live {V : Vocab d α} (K : Constructions V) (A : Live) (θ : ByteArray)
    (F : Array ByteArray) (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) : Live :=
  A.extend (K.coins A θ F c σ)

/-- **The leg's per-finder SHA-512 premises at a strategy and the session `A`**: the session's at the projected
strategy and the strategy's calls (`Live.FinderCR`), and `cr/sha-512` as the strict birthday bound at each
two-continuation finder and at the circuit file's and the public file's finders. -/
structure FinderCR (V : Vocab d α) (K : Constructions V) (A : Live) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) : Prop where
  session : (K.live A θ F c σ).FinderCR (K.project A θ F c σ) (V.cost θ F c σ)
  msg : ∀ j, Claims.StrictBirthday Verity.sha512Msg (K.msgFinder A θ F c σ j)
  record : Claims.StrictBirthday Verity.sha512Msg (K.recordFinder A θ F c σ)
  file : Claims.StrictBirthday Verity.sha512Msg (V.fileFinder θ F)
  publicFile : Claims.StrictBirthday Verity.sha512Msg (V.publicFinder θ F)

/-- **The coupling, the projection's target at the session `A`**: the plays of the period's game whose sampled verdict
accepts are at most as likely as the live audit's plays, at the session widened by the projection's coins and the
strategy projected there, that flock-verify `--zk` accepts at `own` (`LiveAcceptsZKP`), plus the square root of each
two-continuation finder's success (one per round the device opens after a digest of the prover's message, and one for
the record), the tree openings' hiding term, the read finder's collision and the forger's forgery of the device's
signature. Its pieces are the draw (the device's coins the draw reads, `Properties.Challenger.CoinsFromLog`, are the
live audit's draw), the rounds (each round's coins, from the coin trees' log, are the live session's) and the messages
(each message the evidence reveals is the projected strategy's). The leg reads it at each session v2 sets up on the
parameters and the committed files. -/
def CouplingTarget (V : Vocab d α) (k : Verity.Registration.Composite) (K : Constructions V) (A : Live)
    (θ : ByteArray) (F : Array ByteArray) : Prop :=
  ∀ (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)),
    Verity.Game.prob (fun o => V.Verdict k .sampled o) (V.game θ F c) σ ≤
      FlockSoundness.Game.prob (fun o => LiveAcceptsZKP A.I A.own o) (K.live A θ F c σ).game
          (K.project A θ F c σ) +
        (∑ j, ENNReal.ofReal (Real.sqrt ((K.msgFinder A θ F c σ j).Pr Verity.sha512Msg)) +
          ENNReal.ofReal (Real.sqrt ((K.recordFinder A θ F c σ).Pr Verity.sha512Msg)) + A.hidingTerm +
          ENNReal.ofReal ((K.readFinder θ F c σ).Pr Verity.sha512Msg) +
          ENNReal.ofReal ((K.forger θ F c σ).Pr Verity.lmsVerify))

/-- **Composite `k` reads sampled proofs at v2**: its `.sampled` component is one-stage v2's audit, whose check computes
each session's verdict from the session's files. At v1 (`SampledProofs.OneStage.protocol`) the verdicts are the
prover's. -/
def ReadsV2 (k : Verity.Registration.Composite) : Prop :=
  Verity.component k .sampled = some _root_.SampledProofs.OneStageZk.protocol

/-- **v2's parameters in the composite's parameters `θ`**: the component parameters the composition's format decodes
under v2's id, read by v2's decoder, as `Verity.CaseAccepts` reads a component's
(`Properties.Composition.DecodedAccepted`). -/
def ParamsAt (θ : ByteArray) (p : _root_.SampledProofs.OneStageZk.Params) : Prop :=
  ∃ ps, (_root_.Core.canonicalJson θ >>= _root_.Core.Composition.decodeParams) = .ok ps ∧
    ∃ x ∈ ps, x.id = _root_.SampledProofs.OneStageZk.PROTOCOL ∧
      (_root_.Core.canonicalJson x.params.toUTF8 >>= _root_.SampledProofs.OneStageZk.paramsOfJson) = .ok p

/-- **The C-Flock law the draw reads over `n` units** under the verifier `v`'s agreed law: the law
`SampledProofs.Draw.deriveLaw` derives, or else the expanded rule's (`SampledProofs.Draw.flockLaw` at
`SampledProofs.Draw.derive`'s), as `SampledProofs.OneStage.drawnFrom` reads it before `SampledProofs.Draw.drawUnits`
draws. -/
def lawAt (v : _root_.SampledProofs.Registration.Verifier) (n : ℕ) : Flock.V Flock.Draw.Law := do
  let lawJ ← _root_.SampledProofs.Draw.derive v.program v.query v.law v.workTable
  let derived ← _root_.SampledProofs.Draw.deriveLaw v.program v.query v.law v.workTable
  pure (derived.getD (← _root_.SampledProofs.Draw.flockLaw lawJ n))

/-- **`A` is the session v2 sets up on the parameters `θ` and the committed files `F`**: v2's parameters in the
composite's (`ParamsAt`), with the verifier's partition object and program as flock-verify's inputs and the law the draw
reads over the population (`lawAt`); the statement's tags, one table, no work, stratified or closure input and no
recomputation of public input words, as `SampledProofs.OneStageZk.inputsOf` hands them; the committed circuit and public
files, and the vocabulary's reading of the files (`Vocab.read`) as the session's, its lookup tables included; and the
claim's share, the published certificate's. The writer, the device's bytes and the bound's parameters are the
deployment's session's (`Deployed`). -/
structure SetsUp (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (A : Live) : Prop where
  params : ∃ p, ParamsAt θ p ∧ A.I.partition = some p.own.partition ∧ A.I.program = some p.own.program ∧
    lawAt p.base.verifier A.n = .ok A.law
  tags : Flock.Tags.find? _root_.SampledProofs.OneStageZk.STATEMENT = some A.I.tags
  count : A.I.count = 1
  work : A.I.work = none
  stratified : A.I.stratified = none
  closure : A.I.closure = none
  own : A.own = none
  circuit : A.I.circuitFile = V.circuitFile θ F
  publicFile : A.I.publicFile = V.publicFile θ F
  read : V.read θ F = A.fileRead
  share : (V.certOf θ).ε = A.ε

/-- **The deployment's sessions `S` cover the sampled verdict at composite `k`**: where `S` holds no session, no
strategy of the period's game makes the sampled verdict accept. -/
def Covers (V : Vocab d α) (k : Verity.Registration.Composite) (S : ByteArray → Array ByteArray → Option Live) :
    Prop :=
  ∀ θ F, S θ F = none → ∀ (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)),
    Verity.Game.prob (fun o => V.Verdict k .sampled o) (V.game θ F c) σ = 0

/-- **Soundness's vocabulary at `.sampled` reads the constructions `K` at the deployment's sessions `S`**: each session
`S` holds is the one v2 sets up (`SetsUp`); the finder is `K`'s read finder and the forger `K`'s, with no loss term; and
at each session the extractor is the session's at the projected strategy (`Live.extract`), `error`'s condition holds the
leg's per-finder premises (`FinderCR`), and the published `δ` is at least the session's at the read finder's published
bound (`Live.delta`). -/
structure Deployed (V : Vocab d α) (K : Constructions V) (S : ByteArray → Array ByteArray → Option Live) : Prop where
  setsUp : ∀ θ F A, S θ F = some A → SetsUp V θ F A
  finder : ∀ θ F c σ, V.finder .sampled θ F c σ = K.readFinder θ F c σ
  forger : ∀ θ F c σ, V.forger .sampled θ F c σ = K.forger θ F c σ
  loss : ∀ θ x, V.loss .sampled θ x = 0
  extract : ∀ θ F A, S θ F = some A → ∀ (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)),
    HEq (V.extract θ F c σ) ((K.live A θ F c σ).extract (K.project A θ F c σ))
  finderCR : ∀ θ F A, S θ F = some A → ∀ (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)),
    V.finderCR .sampled θ F c σ → FinderCR V K A θ F c σ
  delta : ∀ θ F A, S θ F = some A → ∀ t, A.delta t (V.crBound θ .sampled t) ≤ V.delta θ .sampled t

end Verity.Sampled
