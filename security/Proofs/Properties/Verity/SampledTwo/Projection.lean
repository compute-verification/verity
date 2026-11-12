module

public import Proofs.Properties.Verity.SampledTwo.Leg
public import Proofs.Core.Game.Tape

/-!
# The two-stage leg's projection, piece by piece

`CouplingTargetTwo` reads a play of the period's game as a run of the two-stage verifier (`Live₂.L`, SPOT[Flock]'s
`SpotFlock.V` at the held copies) at a prover built from the strategy. This module states that reading in pieces
that hold separately, each a `Prop` here and a theorem in a file of its own, and the bound they assemble to.

* **The frame** (`Frame`): the coin map `place`, which writes a run's coins into a tape of the period's play at the
  device's releases they are, and the prover `P` the strategy is read as. The play at a tape and a run's coins is
  `Frame.out`.
* **What the form's check reads** (`Reading`), round by round of the run: the coin it reads for round `i` from the
  device's log as the evidence shows it (`coin`), the message it reads for round `i` from the evidence (`msg`), the coin
  the device released for round `i` in the play's own log (`dev`), and the rounds whose coin the run's verifier reads
  (`reads`). The rounds fall in four groups (`groupOf`): the first stage's (round 0), the second stage's (rounds 1 to
  `stage`), the statements' (round `stage + 1`) and the sessions' (past it).
* **The pieces**:
  * `CoinMapLaw`: the coin map is uniform up to the device's coin trees' hiding, and writes each read round's coin at
    the device's release for it.
  * `CoinsRead` at each group of coins: on a play the form's check accepts (`Acc`) whose log the check read is the
    device's (`Good`), the coin the check reads for a round of the group is the device's release for it. Every piece but
    the coin map's, the cut's and the assemblies' is a lemma about the form's check alone, at `Acc`; that the sampled
    verdict accepts only where the check does is the composite's wiring's, one hypothesis of `ProjectionOf`.
  * `MessagesFixed`: on such a play, the message the check reads for each round is the one the prover sends there, but
    on plays at most `miss` likely; and the prover's calls are within the strategy's (`CallsLe`).
  * `CutBound`: an accepted play whose log the check read is not the device's is at most `terms` likely (the read
    finder and the forger).
  * `AcceptsRun`: wherever the check accepts, a run whose coins and messages are what it read accepts.
* **The bound** (`StmtAt`, `ProjectionBound`, `ProjectionOf`): on the plays it accepts the check reads the frame's
  statement, and the sampled verdict is at most as likely as the run's accept plus the pieces' terms.
* **The bridge to `CouplingTargetTwo`** (`MajorityBound`, `CouplingOf`): where the claim fails on the vocabulary's
  transcript, an accepted run whose extracted witness is in the session's relation is at most `bind` likely, so the
  run's accept is at most `PrBad` plus `bind`.

Statements only: definitions, no proofs. Nothing imports this module.
-/

@[expose] public section

namespace Verity.SampledTwo

open Claims
open scoped ENNReal

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- The tapes of the period's play at the strategy `σ`. -/
abbrev TapeOf (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) : Type :=
  Verity.Game.Tape (V.game θ F c) σ

/-- A run's coins at the session of record `A` and the statement `x`. -/
abbrev Coins (A : Live₂) (x : Msg) : Type := Fin (A.L.V.k x) → FlockDeploy.Coin

/-- The prover's message at round `i` of a run, none past its last. -/
def msgAt (t : Claims.Transcript) (i : ℕ) : Option Msg := (t.msgs[i]?).map Prod.fst

/-- **The frame at `A` and the statement `x`**: the coin map `place`, a tape with a run's coins written at the device's
releases they are, and the prover `P` the strategy is read as, on the tape. -/
structure Frame (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) where
  place : TapeOf V θ F c σ → Coins A x → TapeOf V θ F c σ
  P : Claims.Prover (TapeOf V θ F c σ)

/-- The play at the tape `ω` with the run's coins `r` written in. -/
noncomputable def Frame.out {V : Vocab d α} {A : Live₂} {x : Msg} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray}
    {σ : Verity.Game.Strategy (V.game θ F c)} (Fr : Frame V A x θ F c σ) (ω : TapeOf V θ F c σ) (r : Coins A x) :
    Verity.Exchange :=
  Verity.Game.Tape.out (V.game θ F c) σ (Fr.place ω r)

/-- **What the two-stage form's check reads of a play**, by round of the run: the coin it reads for round `i` from
the device's log as the evidence shows it (`coin`), the message it reads for round `i` from the evidence (`msg`:
round 1's the interior registration's record, round `stage + 1`'s the session statements, a session's round its
record's), the coin the device released for round `i` in a log (`dev`), the rounds whose coin the run's verifier
reads (`reads`), and the run's statement it reads, the boundary registration's record (`stmt`). The form's parse owes
`coin`, `msg`, `reads` and `stmt`; the coin map owes `dev`. -/
structure Reading (A : Live₂) where
  coin : Verity.Exchange → ℕ → Option FlockDeploy.Coin
  msg : Verity.Exchange → ℕ → Option Msg
  dev : List Verity.Release → ℕ → Option FlockDeploy.Coin
  reads : ℕ → Prop
  stmt : Verity.Exchange → Option Msg

/-- The groups of a run's rounds by the coins they read. -/
inductive Group where
  | first
  | second
  | stmt
  | sessions
  deriving DecidableEq

/-- **Round `i`'s group**: round 0 the first stage's, rounds 1 to `stage` the second stage's, round `stage + 1` the
statements', every later one the sessions'. -/
def groupOf (A : Live₂) (i : ℕ) : Group :=
  if i = 0 then .first
  else if i ≤ A.h.stage then .second
  else if i = A.h.stage + 1 then .stmt
  else .sessions

/-- **The coin map's law** (piece: the coin map and the coin trees' hiding): the period's play is distributed as the
frame's play at a uniform tape and uniform run coins, up to `hide`; and at every tape and coins, the device's release
for each round the verifier reads is the run's coin there. -/
def CoinMapLaw (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (Fr : Frame V A x θ F c σ) (Rd : Reading A) (hide : ℝ≥0∞) : Prop :=
  (∀ E : Verity.Exchange → Prop, Verity.Game.prob E (V.game θ F c) σ ≤
      ENNReal.ofReal (Claims.Pr fun s : TapeOf V θ F c σ × Coins A x => E (Fr.out s.1 s.2)) + hide) ∧
    ∀ (ω : TapeOf V θ F c σ) (r : Coins A x) (i : Fin (A.L.V.k x)), Rd.reads i.val →
      Rd.dev (Fr.out ω r).log i.val = some (r i)

/-- **The coins the check reads at group `g`** (piece: one per group of coins): on a play the form's check accepts
(`Acc`) whose log the check read is the device's (`Good`), the coin the check reads for each read round of group `g`
is the device's release for it. -/
def CoinsRead (A : Live₂) (Rd : Reading A) (Acc Good : Verity.Exchange → Prop) (g : Group) : Prop :=
  ∀ o : Verity.Exchange, Acc o → Good o → ∀ i, Rd.reads i → groupOf A i = g →
    Rd.coin o i = Rd.dev o.log i

/-- **The statements' round reads no coin**: the run's verifier reads none at round `stage + 1`. -/
def StmtReadsNone (A : Live₂) (Rd : Reading A) : Prop :=
  ∀ i, Rd.reads i → groupOf A i ≠ .stmt

/-- **The messages fixed at their digests** (piece: the messages): on a play the form's check accepts whose log the
check read is the device's, the message the check reads for each round is the prover's there in the run at the same
tape and coins, but on plays at most `miss` likely. -/
def MessagesFixed (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (Fr : Frame V A x θ F c σ) (Rd : Reading A)
    (Acc Good : Verity.Exchange → Prop) (miss : ℝ≥0∞) : Prop :=
  ENNReal.ofReal (Claims.Pr fun s : TapeOf V θ F c σ × Coins A x =>
      Acc (Fr.out s.1 s.2) ∧ Good (Fr.out s.1 s.2) ∧
        ∃ i < A.L.V.k x, Rd.msg (Fr.out s.1 s.2) i ≠
          msgAt (Claims.run FlockDeploy.sha512 Fr.P A.L.V x s.1 s.2) i) ≤ miss

/-- **The prover's calls are the strategy's** (with the messages' piece): every run of the frame's prover makes at most
the strategy's SHA-512 calls, `Vocab.cost`. -/
def CallsLe (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (Fr : Frame V A x θ F c σ) : Prop :=
  ∀ (ω : TapeOf V θ F c σ) (r : Coins A x),
    (Claims.run FlockDeploy.sha512 Fr.P A.L.V x ω r).pLog.length +
      (Claims.run FlockDeploy.sha512 Fr.P A.L.V x ω r).vLog.length ≤ V.cost θ F c σ

/-- **The log the check read is the device's, or a break** (piece: the read finder and the forger): an accepted play
whose log the check read is not the device's is at most `terms` likely. -/
def CutBound (V : Vocab d α) (k : Verity.Registration.Composite) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (Good : Verity.Exchange → Prop) (terms : ℝ≥0∞) :
    Prop :=
  Verity.Game.prob (fun o => V.Verdict k .sampled o ∧ ¬ Good o) (V.game θ F c) σ ≤ terms

/-- **The check's accept is the run's** (piece: the accept, with the form's accept lemma): wherever the form's check
accepts and reads the statement `x`, every run at `x` whose coins at the read rounds and whose messages are what the
check read accepts. -/
def AcceptsRun (A : Live₂) (x : Msg) (Rd : Reading A) (Acc : Verity.Exchange → Prop) : Prop :=
  ∀ o : Verity.Exchange, Acc o → Rd.stmt o = some x →
    ∀ (Ω : Type) (P : Claims.Prover Ω) (ω : Ω) (r : Coins A x),
      (∀ i : Fin (A.L.V.k x), Rd.reads i.val → Rd.coin o i.val = some (r i)) →
      (∀ i < A.L.V.k x, Rd.msg o i = msgAt (Claims.run FlockDeploy.sha512 P A.L.V x ω r) i) →
      (Claims.run FlockDeploy.sha512 P A.L.V x ω r).out = .acc

/-- **The frame's statement is the one the check reads** (with the assembly): on every play of the frame the form's
check accepts, the check reads the statement `x`, the boundary registration's record the commitment `c` carries. -/
def StmtAt (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (Fr : Frame V A x θ F c σ) (Rd : Reading A)
    (Acc : Verity.Exchange → Prop) : Prop :=
  ∀ (ω : TapeOf V θ F c σ) (r : Coins A x), Acc (Fr.out ω r) → Rd.stmt (Fr.out ω r) = some x

/-- **The projection's bound**: the sampled verdict is at most as likely as the frame's run accepting, at a uniform
tape and uniform coins, plus `bound`. -/
def ProjectionBound (V : Vocab d α) (k : Verity.Registration.Composite) (A : Live₂) (x : Msg) (θ : ByteArray)
    (F : Array ByteArray) (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (Fr : Frame V A x θ F c σ)
    (bound : ℝ≥0∞) : Prop :=
  Verity.Game.prob (fun o => V.Verdict k .sampled o) (V.game θ F c) σ ≤
    ENNReal.ofReal (Claims.Pr fun s : TapeOf V θ F c σ × Coins A x =>
      (Claims.run FlockDeploy.sha512 Fr.P A.L.V x s.1 s.2).out = .acc) + bound

/-- **The projection from its pieces** (piece: the assembly; its theorem is `projection_of`): the sampled verdict
only where the form's check accepts (`Acc`, which the composite's wiring owes), the coin map's law, the coins read at
the first stage's, the second stage's and the sessions' rounds, no coin read at the statements', the messages, the cut,
the accept and the statement read give the projection's bound at `hide + miss + terms`. -/
def ProjectionOf (V : Vocab d α) (k : Verity.Registration.Composite) (A : Live₂) (x : Msg) (θ : ByteArray)
    (F : Array ByteArray) (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) : Prop :=
  ∀ (Fr : Frame V A x θ F c σ) (Rd : Reading A) (Acc Good : Verity.Exchange → Prop) (hide miss terms : ℝ≥0∞),
    (∀ o, V.Verdict k .sampled o → Acc o) →
    CoinMapLaw V A x θ F c σ Fr Rd hide → CoinsRead A Rd Acc Good .first → CoinsRead A Rd Acc Good .second →
      CoinsRead A Rd Acc Good .sessions → StmtReadsNone A Rd → MessagesFixed V A x θ F c σ Fr Rd Acc Good miss →
      CutBound V k θ F c σ Good terms → AcceptsRun A x Rd Acc → StmtAt V A x θ F c σ Fr Rd Acc →
      ProjectionBound V k A x θ F c σ Fr (hide + miss + terms)

/-- **The witness fixed at registration** (piece: the majority): where the claim fails on the vocabulary's transcript
`X`, an accepted run of the frame's prover whose extracted witness is in the session's relation is at most `bind`
likely. The transcript is the majority of the boundary values the extractor's witnesses open; `X` is the vocabulary's
at the strategy. -/
def MajorityBound (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (Fr : Frame V A x θ F c σ)
    (X : Verity.Circuit.Trace (V.circuitOf θ F)) (bind : ℝ≥0∞) : Prop :=
  V.extract θ F c σ = X →
    ¬ Accounting.IntegrityHolds (V.partsOf θ F) (V.Opens θ F c) X (V.certOf θ).ε →
      ENNReal.ofReal (Claims.Pr fun s : TapeOf V θ F c σ × Coins A x × A.L.E.C x =>
        (Claims.run FlockDeploy.sha512 Fr.P A.L.V x s.1 s.2.1).out = .acc ∧
          Claims.Rel.ext A.R x (A.L.E.eval FlockDeploy.sha512 Fr.P x s.1 s.2.1 s.2.2).1) ≤ bind

/-- **The coupling from the projection and the majority** (piece: the leg's assembly; its theorem is `coupling_of`):
at each strategy a statement in scope or a verdict that never accepts, a frame, a transcript and bounds such that the
projection's bound and the majority's hold, and where `error`'s premises hold the session's error at the statement and
the frame's prover plus the bounds is within the published `δ`, give the coupling at the two-stage form with the bounds
as its `terms`. -/
def CouplingOf (V : Vocab d α) (k : Verity.Registration.Composite) (A : Live₂) (θ : ByteArray)
    (F : Array ByteArray) : Prop :=
  ∀ (x : (c : ByteArray) → Verity.Game.Strategy (V.game θ F c) → Msg)
    (Fr : (c : ByteArray) → (σ : Verity.Game.Strategy (V.game θ F c)) → Frame V A (x c σ) θ F c σ)
    (X : (c : ByteArray) → Verity.Game.Strategy (V.game θ F c) → Verity.Circuit.Trace (V.circuitOf θ F))
    (bound bind : (c : ByteArray) → Verity.Game.Strategy (V.game θ F c) → ℝ≥0∞),
    (∀ c σ, A.L.Scope (x c σ) ∨ Verity.Game.prob (fun o => V.Verdict k .sampled o) (V.game θ F c) σ = 0) →
    (∀ c σ, ProjectionBound V k A (x c σ) θ F c σ (Fr c σ) (bound c σ)) →
    (∀ c σ, MajorityBound V A (x c σ) θ F c σ (Fr c σ) (X c σ) (bind c σ)) →
    (∀ c σ, V.extract θ F c σ = X c σ) →
    (∀ c σ, V.finderCR .sampled θ F c σ → V.premises .sampled θ F c σ →
      ENNReal.ofReal (A.errAt (x c σ) (Fr c σ).P) + (bound c σ + bind c σ) ≤
        V.delta θ .sampled (V.cost θ F c σ)) →
    CouplingTargetTwo V k A θ F fun c σ => bound c σ + bind c σ

end Verity.SampledTwo
