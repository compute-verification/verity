import Properties.Verity.Game
import Proofs.Flock.Bridge.V5.Statement

/-!
# Sampled proofs' leg of `Verity.Soundness`: the session v2 sets up

The leg is `Verity.Vocab.SampledProofsSound` at a composite whose sampled component is v2
(`SampledProofs.OneStageZk.protocol`), from C-Flock's knowledge soundness of the live session at public inputs on the
event that the verifiers accept the files (`SampledKSLiveV5` with its file and population-order hypotheses read off
acceptance), applied at the session v2 sets up on the parameters and the committed files and at the strategy a
strategy of the period's game projects to. This module states its C-Flock side.

* **The session** (`Live`): V5's binders, each the value v2's check derives on the parameters and the committed files,
  with three of V5's hypotheses as fields (A3 at the device's bytes, `TagsOkZHJ`, and `RecordCustodyZKJP` at the writer
  whose record v2 decodes), and the bound's parameters. The draw is the executable law the parameters' selection rule
  names over the population (`rule`, `pop`), on the device's byte streams (`src`), never a law fixed here.
* **The live audit at the session** (`Live.game`), and V5's simulator and extractor at any strategy of it (`Live.sim`,
  `Live.X`). The simulator and the extractor hold at every law (`decodesZCJH_custodyP`).
* **The per-finder SHA-512 premises at a strategy and a budget** (`Live.FinderCR`): `cr/sha-512` at each table's
  knowledge finders (`TableCRZ`: the fork, pads, τ and level-0 finders at `2t + 2v`, the self-clash finder at `t + v`)
  and `ecr/sha-512` at the link finders (`LinkCRZC`, at `t + w` per run), at `t` the strategy's SHA-512 calls.
* **The leg's δ at the session** (`Live.delta`), each finder's term its own summand:
  * the draw's term (`drawMiss`): the subset law's `C(n − K₀, kd)/C(n, kd)`, the only law V5 covers, and `⊤` at any
    other, so the leg claims nothing at a stratified or work draw until C-Flock's statement covers it;
  * the tables' knowledge term at the worst registration (`ksAvgStrictZ`) and the link term at `t + w` per run
    (`sessionDelta`);
  * each round the device opens after a digest of the prover's message, and the record the commitment is a digest of:
    a two-continuation finder (`twoRun`), which runs the play twice after the digest and outputs the two messages the
    evidence reveals when they differ, at `2t + 2h` SHA-512 calls. The most likely message misses with probability at
    most the square root of the finder's success, so each adds `√((2t + 2h)²/2^513)`;
  * each tree opening the session reads, twice: the opened coins against fresh ones, at the mean over the device's
    fresh tree key of hm96's hiding gap (`CROnly.hidingGap`), which the leftover hash lemma bounds by `2^-257` a leaf
    (`hiding`), so `Claims.Hm96Hiding` holds at it with no premise (`CROnly.hm96Hiding_gap`);
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
derives (the files' parse, the public file over the circuit's public input ports, the population, the registration's
descriptor and query, the draw's size and the session's plan, writer and digest), the draw by the selection rule's
executable law on the device's byte streams, A3 at those bytes, the verify command's tags and `live-verifier` at the
writer as fields, and the bound's parameters. -/
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
  pi : Flock.HmIn.Ports
  hpi : Flock.HmIn.parse D.unroll = .ok pi
  pub : Flock.Public
  hpub : Flock.HmIn.loadPublic (zkTags I) D.unroll pi I.publicFile = .ok pub
  hscope : FlockSoundness.Discharge.Layout.scopeOkH D.unroll = true
  hn : pub.n = n
  /-- the statement's descriptor -/
  desc : Claims.Msg
  /-- the registration's query -/
  qr : Lean.Json
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
  streams : ℕ
  Lb : Fin streams → ℕ
  src : Ω → (i : Fin streams) → Fin (Lb i) → Fin 256
  /-- the executable law the parameters' selection rule names over the population (`SampledProofs.Select`) -/
  rule : Flock.Draw.Law
  /-- the population the draw is over -/
  pop : ℕ
  hA3 : FlockSoundness.Assumptions.UniformRandomBytes Lb src
  hTags : TagsOkZHJ I
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
  /-- the wrong proof blocks the claim's share admits, plus one -/
  K₀ : ℕ
  hK₀ : K₀ ≤ n
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

namespace Live

variable (A : Live)

/-- **The draw's law**: the selection rule's executable law over the population, on the device's bytes. -/
noncomputable def L : FlockSoundness.Audit.Law A.n := FlockSoundness.Audit.Law.execOS (n := A.n) A.src A.rule A.pop

/-- **C-Flock's live audit at the session**, the game V5 bounds: the prover registers, the device's bytes draw, and
the live session runs at the draw. -/
noncomputable def game :=
  auditLive A.L A.Reg (liveOfZJ A.dg (planZHJP A.dj A.y₀) A.N A.wr)

/-- V5's partition: the key program at public inputs, and its outputs. -/
noncomputable def P :=
  (keyProgP A.D.unroll A.pi A.hU A.n).partition (FlockSoundness.Discharge.Layout.outsOfP A.D.unroll A.hU A.n A.pi)

/-- V5's binding at public inputs. -/
noncomputable def vb :=
  (rsZHJP A.dj A.y₀ A.ht (Flock.Desc.parse_flat A.hD) A.hscope A.hU A.hpi A.hpub).bindingZ
    (rsZHJP_computes A.dj A.y₀ A.ht (Flock.Desc.parse_flat A.hD) A.hscope A.hU A.hpi A.hpub)

/-- **V5's simulator at a strategy of the live audit**: the strategy in the compiled batched session at flock-verify
`--zk`'s acceptance at `own`, through the decoder `RecordCustodyZKJP` gives at the session's law. -/
noncomputable def sim (τ : FlockSoundness.Game.Strategy A.game) :=
  simZJA execArith hs512 (planZHJP A.dj A.y₀) (AcceptsZKP A.I A.own) A.wr
    (opnAtZHJP (planZHJP A.dj A.y₀) A.I A.own A.dj) A.dg (decAtZJ (planZHJP A.dj A.y₀)) (fun _ _ _ _ => true)
    (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) A.N
    (decodesZCJH_custodyP A.I A.own A.ht (Flock.Desc.parse_flat A.hD) A.kd A.hdiv A.dj A.y₀ A.wr A.L A.hTags A.hCust) τ

/-- **V5's extractor at a strategy of the live audit**: `XplurZC` at the simulator's registration and continuation. -/
noncomputable def X (τ : FlockSoundness.Game.Strategy A.game) :=
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

/-- **The draw's term**: at the subset law of the session's `kd` units over its `n` proof blocks, on one byte stream,
`C(n − K₀, kd)/C(n, kd)`, V5's; `⊤` at any other law, which C-Flock's statement doesn't cover yet. -/
noncomputable def drawMiss : ℝ≥0∞ :=
  match A.rule with
  | .subset kd =>
    if kd = A.kd ∧ A.pop = A.n ∧ A.streams = 1 then ((A.n - A.K₀).choose A.kd : ℝ≥0∞) / (A.n.choose A.kd : ℝ≥0∞)
    else ⊤
  | _ => ⊤

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
noncomputable def hiding : ℝ≥0∞ := 2 * A.rounds / 2 ^ 257

/-- **The leg's δ at the session, `t` SHA-512 calls and the read finder's bound `cr`**: the session's error, a
two-continuation term per round opened after a digest and one for the record, the tree openings' hiding term, the two
file finders' `2²/2^513`, and `cr`. -/
noncomputable def delta (t : ℕ) (cr : ℝ) : ℝ≥0∞ :=
  A.sessionDelta t + A.rounds * twoRun (A.qM t) + twoRun (A.qM t) + A.hiding +
    2 * ENNReal.ofReal ((2 : ℝ) ^ 2 / 2 ^ (513 : ℝ)) + ENNReal.ofReal cr

end Live

/-- **A projection**: at the session `A`, the C-Flock live strategy a strategy of the period's game at `θ`, `F` and `c`
is read as. -/
abbrev Projection {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α] (V : Vocab d α) : Type 1 :=
  (A : Live) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Verity.Game.Strategy (V.game θ F c) →
    FlockSoundness.Game.Strategy A.game

end Verity.Sampled
