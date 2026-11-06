module

public import Properties.Verity.Game
public import Properties.SampledProofs.Draw
public import Proofs.Flock.Bridge.V5.Statement
public import Proofs.Flock.Bridge.CircuitOf
public import Proofs.Flock.Soundness.ZK.RealLeaves
public import Proofs.Properties.Verity.Sampled.Deployed

/-!
# Sampled proofs' leg of `Verity.Soundness`: the session v2 sets up

The leg is `Verity.Vocab.SampledProofsSound` at the one-stage form of a composite's sampled component
(`VStar.Recursive.Sampled.protocol`), v2's audit (`SampledProofs.OneStageZk`), from C-Flock's knowledge soundness of the
live session at public inputs on the event that the verifiers accept the files (`SampledKSLiveV5` with its file and
population-order hypotheses read off acceptance), applied at the session v2 sets up on the parameters and the committed
files and at the strategy a strategy of the period's game projects to. This module states its C-Flock side.

* **The session** (`Live`): V5's binders, each the value v2's check derives on the parameters and the committed files,
  with four of V5's hypotheses as fields (A3 at the device's bytes, `TagsOkZHJ`, the registration's law on its layout's
  strata, `Flock.Bridge.LawOn`, and `RecordCustodyZKJPLaw` at the writer whose record v2 decodes, under that law), and
  the bound's parameters. The draw file is the one `flock-verify draw` prints for the verifier's own draw under the law
  (`printedDraw`, `Live.dj`), so V5's `hsz` holds at it (`Live.hsz`). The circuit names no public input port: v2 hands
  `flock-verify` no recomputation of public input words, so `HmIn.checkOwn` refuses any circuit that names one, and the
  session's ports are `HmIn.parse`'s empty ones.
* **The draw** (`Live.L`): the law the parameters' selection rule derives over the population (`law`), drawn as
  `SampledProofs.Select.select` draws it, by `SampledProofs.Draw.drawUnits` on the device's one byte stream (`src`); a
  draw that fails draws every unit. It is V5's draw at the session's bytes and law (`Live.L_eq`, `Flock.Bridge.execSel`).
* **The live audit at the session** (`Live.game`), and V5's simulator and extractor at any strategy of it (`Live.sim`,
  `Live.Xr`). The simulator and the extractor hold at every law V5 admits (`decodesZCJH_custodyP`).
* **The claim's transcript at a strategy** (`Live.extract`): V5's trace fixed at registration, opened at the deployed
  commitment and read at the circuit's gates with its constants, on the committed files' circuit. Without public input
  ports V5's key program is the claim's (`keyProgP_empty`), and the files the session reads parse to it
  (`Live.circuit_eq`).
* **The projection** (`Constructions`): a strategy of the period's game is read as a strategy of the live audit. A
  C-Flock strategy flips no coin of its own, so the device's other releases the strategy may ask for (`Widening`:
  other families' coins, other trees' samples) widen the draw's sample (`Live.extend`), which the draw doesn't read: the
  draw's law, A3 and the leg's δ are the session's (`uniform_fst`, `Live.delta_extend`).
* **The coupling** (`CouplingTarget`), the projection's target: the sampled verdict is at most as likely in the period's
  game as flock-verify `--zk`'s acceptance in the live audit at the projected strategy, plus the two-continuation
  finders' terms, the tree openings' hiding term, the read finder's collision and the forger's forgery.
* **The component** (`ReadsV2`): the composite's `.sampled` component is `VStar.Recursive.Sampled.protocol`, whose
  one-stage form is v2 and which refuses v1 (whose verdicts are the prover's). Both composites name it
  (`composite_readsV2`). The parameters fix the form before any coin; this leg covers the one-stage form, and the
  recursive form (`VStar.Recursive`) is not this leg's (`Covers`).
* **The session v2 sets up on the parameters and the committed files** (`SetsUp`): v2's parameters in the composite's
  (`ParamsAt`), the session's inputs as v2 hands them to flock-verify (the statement, one table, the verifier's
  partition object and program, no recomputation of public input words), the committed circuit and public files and
  the vocabulary's reading of them, the law the draw reads over the population (`lawAt`), and the claim's share.
* **The device's log as v2's check reads it** (`coinsRead`): the releases the signed log in v2's evidence shows after
  its head, in order, a coin tree's line its own release and the other lines in pairs, a root's line and its coin's
  release, as `SampledProofs.Challenger.releasedRootsOf` reads them, each with the bits v2 reads of it: the draw
  family's coins after the registration's digest, the bytes the draw reads; each tree's root and key; each opening's
  coins, as a session's record gives them. v2's evidence opens no other coin, so another release has none. Where a line
  or a record doesn't parse as it is read, the read shows no log.
* **The vocabulary's sampled fields at the deployment's sessions** (`Deployed`, `Covers`): at most one session at each
  parameters and committed files, each one v2 sets up, with the deployment's writer, device bytes and bound
  parameters. Soundness's `Vocab` reads v2's log (`coinsRead`), the constructions' read finder and forger at `.sampled`,
  no loss term, and at each session the extractor at the projected strategy, openings that hold wherever the session's
  deployed commitment opens under its roots (`Live.OpensUnder`), the leg's per-finder premises in `error`'s condition,
  and a published `δ` at least the session's. Where there is no session and the parameters name the one-stage form,
  the sampled verdict never accepts.
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

@[expose] public section

namespace Verity.Sampled

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

/-- **The session v2 sets up on the parameters and the committed files**: V5's binders, each the value v2's check
derives (the files' parse, the public file over the circuit's empty public input ports, the circuit's ports named once
with one domain at each root, the population, the registration's descriptor and query and the population's order, the
law on the registration's strata and the draw's size, and the session's plan, writer and digest), the device's byte
stream the draw reads, A3 at those bytes, the verify command's tags and `live-verifier` at the writer under the law as
fields, and the bound's parameters. The draw file is the printed one (`Live.dj`). -/
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
  /-- the circuit names each of its ports once -/
  hdup : Flock.Registered.dupName D.unroll = none
  /-- two ports with one root at the public file's leaf map have one domain -/
  hroots : Flock.Bridge.RootFixesDomainL D.unroll (Flock.Bridge.LeafMap.ofPublic D.unroll pub n)
  hn : pub.n = n
  hnO : pubO.n = n
  /-- the statement's descriptor -/
  desc : Claims.Msg
  /-- the registration's query -/
  qr : Lean.Json
  /-- the registration's population is the public file's proof blocks, in order -/
  hpop : Flock.Bridge.PopMatches desc qr n
  /-- the law the parameters' selection rule derives over the population (`SampledProofs.Select`) -/
  law : Flock.Draw.Law
  /-- the draw's size -/
  kd : ℕ
  /-- the law is on the strata of the registration's layout and draws `kd` proof blocks -/
  hlaw : Flock.Bridge.LawOn law (Flock.Bridge.nsOf desc qr) kd
  hdiv : kd % nTab I = 0
  S₀ : Finset (Fin n)
  d₀ : Lean.Json
  y₀ : DrawSetupZHJP (inputsJ I 0) own (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀
  wr : (S : Finset (Fin n)) → (R : Reg) →
    LiveOutJ (planZHJP (Reg := Reg) (printedDraw law) y₀ S R).length (mPtsOf D.unroll.g (kd / nTab I)) →
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
  /-- `live-verifier` at the writer as V5 reads it, under the law (`DrawFileZK (printedDraw law) law n`) -/
  hCust : FlockSoundness.Assumptions.Zk.RecordCustodyZKJPLaw I own (printedDraw law)
    (fun S R => (planZHJP (Reg := Reg) (printedDraw law) y₀ S R).length) wr law
  k : ℕ
  Rw : ℕ
  M : ℕ
  hk : 1 ≤ k
  hRw : 1 ≤ Rw
  hM : 1 ≤ M
  ρ : ℝ
  hρ : ρ < 1
  hr : ∀ S R j, rateZ (planZHJP (Reg := Reg) (printedDraw law) y₀ S R) j k ≤ ρ
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

/-- A mean over a sample widened by a second coin that the summand doesn't read is the mean over the sample. -/
theorem avg_fst {Ω T : Type} [Fintype Ω] [Fintype T] [Nonempty T] (G : Ω → ℝ≥0∞) :
    Verity.Game.avg (fun ω : Ω × T => G ω.1) = Verity.Game.avg G :=
  (FlockSoundness.ZK.avg_prod_split _).trans
    (congrArg Verity.Game.avg (funext fun c => FlockSoundness.Game.avg_const (G c)))

namespace Live

variable (A : Live)

/-- **The session's draw file**: at each draw, the one `flock-verify draw` prints for the verifier's own draw under the
session's law (`printedDraw`). -/
abbrev dj : Finset (Fin A.n) → Lean.Json := printedDraw A.law

/-- **V5's `hsz` at the session**: the draw file decodes only to the law's `kd` units (`drawSize_deployed`). -/
theorem hsz : ∀ S ud, Flock.Draw.UnitDraw.ofJson (A.dj S) = .ok ud → ud.units.size = A.kd :=
  drawSize_deployed A.hlaw

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

/-- The session's draw is V5's, `Flock.Bridge.execSel` at the session's bytes and law. -/
theorem L_eq : A.L = Flock.Bridge.execSel (n := A.n) A.src A.law A.n := by
  unfold Live.L Flock.Bridge.execSel
  refine congrArg (@Verity.Law.mk (Fin A.n) A.Ω _ _) (funext fun ω => ?_)
  rcases _root_.SampledProofs.Draw.drawUnits A.law A.n (_root_.Properties.SampledProofs.bytesOf (A.src ω 0)) with
    _ | ⟨d, x⟩ <;> rfl

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
`--zk`'s acceptance at `own`, through the decoder `RecordCustodyZKJPLaw` and the draw file's size give at the session's
law. -/
noncomputable def sim (τ : FlockSoundness.Game.Strategy A.game) :=
  simZJA execArith hs512 (planZHJP A.dj A.y₀) (AcceptsZKP A.I A.own) A.wr
    (opnAtZHJP (planZHJP A.dj A.y₀) A.I A.own A.dj) A.dg (decAtZJ (planZHJP A.dj A.y₀)) (fun _ _ _ _ => true)
    (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) A.N
    (decodesZCJH_custodyP A.I A.own A.ht (Flock.Desc.parse_flat A.hD) A.law A.kd A.hdiv A.dj A.y₀ A.wr A.L A.hTags
      A.hCust A.hsz) τ

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
    (_root_.Properties.SampledProofs.oneStageOf A.law).Pa (Flock.Bridge.nsOf A.desc A.qr) B.1)

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

/-- **The tables' knowledge term at the widened sample is the session's**: the draw reads only the session's bytes. -/
theorem ksAvg_extend (S : Widening) (k : ℕ) (qF qS : ℝ) (R : A.Reg) :
    ksAvgStrictZ (planZHJP A.dj A.y₀) (A.extend S).L k qF qS R = ksAvgStrictZ (planZHJP A.dj A.y₀) A.L k qF qS R :=
  avg_fst (T := S.T) fun ω : A.Ω => ⨆ j : Fin (planZHJP A.dj A.y₀ (A.L.draw ω) R).length,
    ENNReal.ofReal (ksStrictZ (planZHJP A.dj A.y₀ (A.L.draw ω) R) j k qF qS /
      (1 - rateZ (planZHJP A.dj A.y₀ (A.L.draw ω) R) j k))

open Classical in
/-- **The share of the widened sample whose draw reads a commit string is the session's**: a widened draw reads it
exactly when the session's bytes' draw does, whatever the other coins. -/
theorem ratio_extend (S : Widening) (p : A.vb.Pos) :
    (Fintype.card (ΩcZ execArith A.P (planZHJP A.dj A.y₀) (tabZHJP A.dj A.y₀) A.vb (A.extend S).L p) : ℝ) /
        Fintype.card (A.extend S).L.Ω =
      (Fintype.card (ΩcZ execArith A.P (planZHJP A.dj A.y₀) (tabZHJP A.dj A.y₀) A.vb A.L p) : ℝ) /
        Fintype.card A.L.Ω := by
  have hc : Fintype.card (ΩcZ execArith A.P (planZHJP A.dj A.y₀) (tabZHJP A.dj A.y₀) A.vb (A.extend S).L p) =
      Fintype.card (ΩcZ execArith A.P (planZHJP A.dj A.y₀) (tabZHJP A.dj A.y₀) A.vb A.L p) * Fintype.card S.T :=
    (Fintype.card_congr (Equiv.prodSubtypeFstEquivSubtypeProd (β := S.T)
      (p := fun ω : A.L.Ω => ReadsDrawZ execArith A.P (planZHJP A.dj A.y₀) (tabZHJP A.dj A.y₀) A.vb p ω))).trans
      (Fintype.card_prod _ _)
  have hΩ : Fintype.card (A.extend S).L.Ω = Fintype.card A.L.Ω * Fintype.card S.T := Fintype.card_prod A.Ω S.T
  rw [hc, hΩ]
  simp only [Nat.cast_mul]
  exact mul_div_mul_right _ _ (Nat.cast_ne_zero.mpr Fintype.card_ne_zero)

open Classical in
/-- **The session's error at the widened sample is the session's.** -/
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
  show A.drawMiss + (⨆ R : A.Reg, ksAvgStrictZ (planZHJP A.dj A.y₀) (A.extend S).L A.k (A.qF t) (A.qS t) R) +
      ENNReal.ofReal ((∑ p : A.vb.Pos,
          (Fintype.card (ΩcZ execArith A.P (planZHJP A.dj A.y₀) (tabZHJP A.dj A.y₀) A.vb (A.extend S).L p) : ℝ) /
            Fintype.card (A.extend S).L.Ω) *
        (2 * (A.tL t * (1 + A.k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * A.M) + A.k / (Real.exp 1 * A.Rw)) /
          (1 - A.ρ)) = A.sessionDelta t
  rw [hks, hsum]
  rfl

/-- **The leg's δ at the widened sample is the session's**: the draw, the tables' and link terms, the rounds and the
finders' budgets read only the session's bytes and parameters. -/
theorem delta_extend (S : Widening) (t : ℕ) (cr : ℝ) : (A.extend S).delta t cr = A.delta t cr := by
  unfold Live.delta
  rw [sessionDelta_extend]
  rfl

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
        (Flock.Bridge.stmtOf A.P A.κ (_root_.Properties.SampledProofs.oneStageOf A.law) A.desc ⟨"", 0, []⟩
          A.roots)))

/-- **The openings `op` of the committed files `r` hold wherever the session's deployed commitment opens**: at each
gate `r` doesn't fix to a constant, a bit the deployed commitment opens at the gate's claims position under the
session's roots (`Flock.Bridge.posOf`, `Cd`, `roots`) opens the gate to every value whose first bit it is. -/
def OpensUnder (r : Verity.FileRead) (op : (g : Fin r.circuit.N) → BitVec (r.circuit.w g) → Prop) : Prop :=
  ∀ (h : r.circuit.N = A.Cc.N) (g : Fin r.circuit.N) (b : Bool) (d : Claims.Msg), r.consts g = none →
    A.Cd.Opens Claims.Sampling.sha512Msg A.roots ⟨Flock.Bridge.posOf A.Cc (Fin.cast h g), b, d⟩ = true →
      ∀ v : BitVec (r.circuit.w g), v.getLsbD 0 = b → op g v

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

/-- **Composite `k` reads sampled proofs at v2**: its `.sampled` component is `VStar.Recursive.Sampled.protocol`. Its
one-stage form is one-stage v2's audit (`SampledProofs.OneStageZk.checkBlobs`), whose check computes each session's
verdict from the session's files; one-stage v1, whose verdicts are the prover's, is refused
(`VStar.Recursive.Sampled.V1_REFUSED`). -/
def ReadsV2 (k : Verity.Registration.Composite) : Prop :=
  Verity.component k .sampled = some VStar.Recursive.Sampled.protocol

/-- Both composites read sampled proofs at v2. -/
theorem composite_readsV2 (k : Verity.Registration.Composite) : ReadsV2 k := by
  cases k <;> rfl

/-- **v2's parameters in the composite's parameters `θ`**: the component parameters the composition's format decodes
under the sampled component's id, read by its decoder in its one-stage form, as `Verity.CaseAccepts` reads a component's
(`Properties.Composition.DecodedAccepted`). -/
def ParamsAt (θ : ByteArray) (p : _root_.SampledProofs.OneStageZk.Params) : Prop :=
  ∃ ps, (_root_.Core.canonicalJson θ >>= _root_.Core.Composition.decodeParams) = .ok ps ∧
    ∃ x ∈ ps, x.id = VStar.Recursive.Sampled.PROTOCOL ∧
      (_root_.Core.canonicalJson x.params.toUTF8 >>= VStar.Recursive.Sampled.paramsOfJson) = .ok (.inl p)

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

/-- **v2's case in what the prover sent**: v2's parameters in the parameters sent, as `ParamsAt` reads them, and the
registration and the evidence of the case the composition's evidence names under the sampled component's id, each read
by its decoder in its one-stage form. -/
def caseOf (s : Verity.Sent) : Flock.V (_root_.SampledProofs.OneStageZk.Params ×
    _root_.SampledProofs.OneStage.Commitment × _root_.SampledProofs.OneStageZk.Evidence) := do
  let ps ← _root_.Core.canonicalJson s.params >>= _root_.Core.Composition.decodeParams
  let some x := ps.find? (·.id == VStar.Recursive.Sampled.PROTOCOL) | throw "the parameters name no sampled proofs"
  let ev ← _root_.Core.Composition.decodeEvidence s.evidence
  let some y := ev.find? (·.id == VStar.Recursive.Sampled.PROTOCOL) | throw "the evidence holds no sampled proofs case"
  let .inl p ← _root_.Core.canonicalJson x.params.toUTF8 >>= VStar.Recursive.Sampled.paramsOfJson
    | throw "the sampled proofs parameters' form is not one-stage"
  let c ← _root_.SampledProofs.OneStage.commitmentOf y.commitment
  let .inl e ← VStar.Recursive.Sampled.evidenceOf y.evidence
    | throw "the sampled proofs evidence's form is not one-stage"
  pure (p, c, e)

/-- The challenger session's release log in v2's evidence, a line each, as `SampledProofs.Challenger.trustedTreesOf`
reads it. -/
def linesOf (e : _root_.SampledProofs.OneStageZk.Evidence) : Flock.V (List String) := do
  let .arr ls := _root_.SampledProofs.getD (_root_.SampledProofs.getD e.base.coins "session") "log"
    | throw "the session's log is not a list of lines"
  ls.toList.mapM fun | .str l => pure l | _ => throw "a line of the session's log is not a string"

/-- **The coins a `--zk` session's record gives round `k` of stream `stream`**, as
`SampledProofs.Challenger.sessionTreeReason` reads them: its link points at the session stream's round 0, else that
stream's round `k`. -/
def recordRound (record : Lean.Json) (stream : String) (k : ℕ) : Option ByteArray :=
  if stream == _root_.SampledProofs.Challenger.TREE_SESSION && k == 0 then
    match _root_.SampledProofs.getD (_root_.SampledProofs.getD record "link") "points" with
    | .arr ps => _root_.SampledProofs.Challenger.recordCoins?
        (.arr (ps.foldl (fun acc p => match p with | .arr cs => acc ++ cs | _ => acc.push .null) #[]))
    | _ => none
  else
    match _root_.SampledProofs.getD record "streams" with
    | .arr ss =>
      (ss.find? fun x => match _root_.SampledProofs.getD x "stream" with | .str n => n == stream | _ => false).bind
        fun x => match _root_.SampledProofs.getD x "rounds" with
          | .arr rs =>
            rs[k]?.bind fun r => _root_.SampledProofs.Challenger.recordCoins? (_root_.SampledProofs.getD r "coins")
          | _ => none
    | _ => none

/-- The root of the coin tree a session's record names. -/
def recordTree (record : Lean.Json) : String :=
  match _root_.SampledProofs.getD (_root_.SampledProofs.getD record "coin_tree") "root" with
  | .str r => r
  | _ => ""

/-- A coin tree's spec as its line names it. -/
def specOf (j : Lean.Json) : Option Flock.CoinTree.Spec := do
  let .arr ss := _root_.SampledProofs.getD j "streams" | none
  let streams ← ss.mapM fun | .str x => some x | _ => none
  pure ⟨streams, ← _root_.SampledProofs.nat? (_root_.SampledProofs.getD j "rounds_log"),
    ← _root_.SampledProofs.nat? (_root_.SampledProofs.getD j "block")⟩

/-- **The family and the root a root line of the release log shows**, as `SampledProofs.Challenger.releasedRootsOf`
reads them; `none` where the line doesn't parse or names no family or no root. -/
def rootOf (l : String) : Option (String × ByteArray) := do
  let r ← (_root_.SampledProofs.Challenger.event l).toOption
  let .str label := _root_.SampledProofs.getD r "label" | none
  let .str root := _root_.SampledProofs.getD r "root" | none
  pure (label, ← (Flock.unhex root).toOption)

/-- **A coin tree's line's release**, after the trees `ts` sampled so far (their roots and specs): its hello after its
nonce, with its root's and key's bits (`SampledProofs.Challenger.coinTreesOf`'s tree line), or a round opened, of the
`i`-th tree sampled and its spec's `s`-th stream, after the bytes it answers, with its coins' bits where a record of
that tree gives them, else none; `none` where the line doesn't parse as either. -/
def treeRelease (records : List Lean.Json) (ts : List (String × Flock.CoinTree.Spec)) (j : Lean.Json) :
    Option (Verity.Release × List (String × Flock.CoinTree.Spec)) :=
  let ev := match _root_.SampledProofs.getD j "event" with | .str x => x | _ => ""
  if ev == _root_.SampledProofs.Challenger.TREE_EVENT then do
    let .str key := _root_.SampledProofs.getD j "key" | none
    let .str nonce := _root_.SampledProofs.getD j "nonce" | none
    let .str root := _root_.SampledProofs.getD j "root" | none
    let spec ← specOf (_root_.SampledProofs.getD j "spec")
    let ν ← (Flock.unhex nonce).toOption
    let r ← (Flock.unhex root).toOption
    let κ ← (Flock.unhex key).toOption
    pure (⟨.hello spec, ν, Verity.bitsOfBytes (r ++ κ)⟩, ts ++ [(root, spec)])
  else if ev == _root_.SampledProofs.Challenger.OPEN_EVENT then do
    let .str root := _root_.SampledProofs.getD j "tree" | none
    let .str stream := _root_.SampledProofs.getD j "stream" | none
    let .str aft := _root_.SampledProofs.getD j "after" | none
    let k ← _root_.SampledProofs.nat? (_root_.SampledProofs.getD j "round")
    let n ← _root_.SampledProofs.nat? (_root_.SampledProofs.getD j "n")
    let i ← ts.findIdx? (·.1 == root)
    let (_, spec) ← ts[i]?
    let s ← spec.streams.findIdx? (· == stream)
    let b ← (Flock.unhex aft).toOption
    let coins := match records.find? (recordTree · == root) with
      | some rec => ((recordRound rec stream k).map Verity.bitsOfBytes).getD []
      | none => []
    pure (⟨.opening i s n, b, coins⟩, ts)
  else none

/-- **The releases the log's lines show**, after the release log's head, for the draw of family `label` after the
registration's digest `after`, with the draw's coins `ξ` still to place, the trees `ts` sampled so far and the
sessions' records: a coin tree's line (`SampledProofs.Challenger.treeLine`) its release (`treeRelease`), and the other
lines in pairs, as `SampledProofs.Challenger.releasedRootsOf` reads them, a root's line `pending` and its coin's
release, of the family and after the root the root's line shows (`rootOf`), with the next coin of `ξ` where it is the
draw's (`label`'s, after `after`), else none. `none` where a line doesn't parse as it is read or the log ends on a root
without its release, as v2 refuses that log. -/
def readLines (label : String) (after : ByteArray) (records : List Lean.Json) :
    List String → List (String × Flock.CoinTree.Spec) → Option String → ByteArray → Option (List Verity.Release)
  | [], _, none, _ => some []
  | [], _, some _, _ => none
  | l :: ls, ts, pending, ξ =>
    if _root_.SampledProofs.Challenger.treeLine l then do
      let (r, ts') ← (_root_.SampledProofs.Challenger.event l).toOption.bind (treeRelease records ts)
      pure (r :: (← readLines label after records ls ts' pending ξ))
    else
      match pending with
      | none => readLines label after records ls ts (some l) ξ
      | some a => do
        let (lab, root) ← rootOf a
        if lab = label ∧ root = after then
          let m := _root_.SampledProofs.Challenger.COIN_BYTES
          pure (⟨.fresh label, after, Verity.bitsOfBytes (ξ.extract 0 m)⟩ ::
            (← readLines label after records ls ts none (ξ.extract m ξ.size)))
        else pure (⟨.fresh lab, root, []⟩ :: (← readLines label after records ls ts none ξ))

/-- The records of the sessions v2's evidence `e` names, on the blobs sent, parsed as
`SampledProofs.OneStageZk.treesHeld` parses them (`Flock.JsonParse.parse`); `none` where one is missing or doesn't
parse. -/
def recordsOf (s : Verity.Sent) (e : _root_.SampledProofs.OneStageZk.Evidence) : Option (List Lean.Json) :=
  e.sessions.toList.mapM fun r => do
    let b ← ((_root_.Core.Blobs.ofArray s.blobs).get r.record "record").toOption
    (Flock.JsonParse.parse (← String.fromUTF8? b)).toOption

/-- **The device's log as v2's check reads it** from what the prover sent: the releases the challenger session's log in
v2's evidence shows after its head (`readLines`), at the parameters' challenger family and the registration's digest,
with the coins v2's draw reads (`SampledProofs.OneStageZk.coinsOf`) and the records of the evidence's sessions on the
blobs sent (`recordsOf`). A release whose coins v2 doesn't open has none; where a line or a record doesn't parse as it
is read, the read shows no log, never a default. -/
def coinsRead (s : Verity.Sent) : Option (List Verity.Release) := do
  let (p, c, e) ← (caseOf s).toOption
  let (_, label) ← p.base.challenger
  let ξ ← (_root_.SampledProofs.OneStageZk.coinsOf p c e).toOption
  let lines ← (linesOf e).toOption
  readLines label c.digest (← recordsOf s e) lines.tail [] none ξ

/-- **The deployment's sessions `S` cover the sampled verdict's one-stage form at composite `k`**: where `S` holds no
session and the parameters name the one-stage form (`ParamsAt`), no strategy of the period's game makes the sampled
verdict accept. Parameters naming the recursive form (`VStar.Recursive`) are not this leg's. -/
def Covers (V : Vocab d α) (k : Verity.Registration.Composite) (S : ByteArray → Array ByteArray → Option Live) :
    Prop :=
  ∀ θ F, S θ F = none → (∃ p, ParamsAt θ p) → ∀ (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)),
    Verity.Game.prob (fun o => V.Verdict k .sampled o) (V.game θ F c) σ = 0

/-- **Soundness's vocabulary at `.sampled` reads the constructions `K` at the deployment's sessions `S`**: each session
`S` holds is the one v2 sets up (`SetsUp`); the device's log as the sampled check reads it is v2's (`coinsRead`); the
finder is `K`'s read finder and the forger `K`'s, with no loss term; and at each session the extractor is the
session's at the projected strategy (`Live.extract`), the vocabulary's openings hold wherever the session's deployed
commitment opens (`Live.OpensUnder`), `error`'s condition holds the leg's per-finder premises (`FinderCR`), and the
published `δ` is at least the session's at the read finder's published bound (`Live.delta`). -/
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
    V.finderCR .sampled θ F c σ → FinderCR V K A θ F c σ
  delta : ∀ θ F A, S θ F = some A → ∀ t, A.delta t (V.crBound θ .sampled t) ≤ V.delta θ .sampled t

end Verity.Sampled
