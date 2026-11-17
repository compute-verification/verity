module

public import Proofs.Properties.Verity.SampledTwo.Subprotocol
public import Proofs.Core.Game.Tape

/-! The two-stage subprotocol's projection stated in pieces that hold separately, each a `Prop`, and the
bound they assemble to. -/

@[expose] public section

namespace Verity.SampledTwo

open Claims
open scoped ENNReal

variable {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α]

abbrev TapeOf (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) : Type :=
  Verity.Game.Tape (V.game θ F c) σ

abbrev Coins (A : Live₂) (x : Msg) : Type := Fin (A.L.V.k x) → FlockDeploy.Coin

/-- The prover's message at round `i` of a run, none past its last. -/
def msgAt (t : Claims.Transcript) (i : ℕ) : Option Msg := (t.msgs[i]?).map Prod.fst

structure Frame (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) where
  place : TapeOf V θ F c σ → Coins A x → TapeOf V θ F c σ
  P : Claims.Prover (TapeOf V θ F c σ)

noncomputable def Frame.out {V : Vocab d α} {A : Live₂} {x : Msg} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray}
    {σ : Verity.Game.Strategy (V.game θ F c)} (Fr : Frame V A x θ F c σ) (ω : TapeOf V θ F c σ) (r : Coins A x) :
    Verity.Exchange :=
  Verity.Game.Tape.out (V.game θ F c) σ (Fr.place ω r)

/-- `agree o i c c'` holds where `c` agrees with `c'` on everything the run's verifier reads of it: for a
release of `n` bytes read as a coin padded with zeros, their first `n` bytes. The form's parse owes `coin`, `msg`,
`reads`, `stmt` and `agree`; the coin map owes `dev`. -/
structure Reading (A : Live₂) where
  coin : Verity.Exchange → ℕ → Option FlockDeploy.Coin
  msg : Verity.Exchange → ℕ → Option Msg
  dev : List Verity.Release → ℕ → Option FlockDeploy.Coin
  reads : ℕ → Prop
  stmt : Verity.Exchange → Option Msg
  agree : Verity.Exchange → ℕ → FlockDeploy.Coin → FlockDeploy.Coin → Prop

inductive Group where
  | first
  | second
  | stmt
  | sessions
  deriving DecidableEq

def groupOf (A : Live₂) (i : ℕ) : Group :=
  if i = 0 then .first
  else if i ≤ A.h.stage then .second
  else if i = A.h.stage + 1 then .stmt
  else .sessions

def CoinMapDistribution (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (Fr : Frame V A x θ F c σ) (Ag : Verity.Exchange → Coins A x → Prop)
    (hide : ℝ≥0∞) : Prop :=
  ∀ E : Verity.Exchange → Prop, Verity.Game.prob E (V.game θ F c) σ ≤
    ENNReal.ofReal (Claims.Pr fun s : TapeOf V θ F c σ × Coins A x =>
      E (Fr.out s.1 s.2) ∧ Ag (Fr.out s.1 s.2) s.2) + hide

def DevAgrees (A : Live₂) (x : Msg) (Rd : Reading A) (Acc Good : Verity.Exchange → Prop)
    (Ag : Verity.Exchange → Coins A x → Prop) : Prop :=
  ∀ o : Verity.Exchange, Acc o → Good o → ∀ r : Coins A x, Ag o r →
    ∀ i : Fin (A.L.V.k x), Rd.reads i.val → ∀ c, Rd.dev o.log i.val = some c → Rd.agree o i.val c (r i)

def CoinsRead (A : Live₂) (Rd : Reading A) (Acc Good : Verity.Exchange → Prop) (g : Group) : Prop :=
  ∀ o : Verity.Exchange, Acc o → Good o → ∀ i, Rd.reads i → groupOf A i = g →
    Rd.coin o i = Rd.dev o.log i

def CoinsAgree (A : Live₂) (Rd : Reading A) (Acc Good : Verity.Exchange → Prop) (g : Group) : Prop :=
  ∀ o, Acc o → Good o → ∀ i, Rd.reads i → groupOf A i = g → ∀ c, Rd.coin o i = some c →
    ∀ c', (∀ d, Rd.dev o.log i = some d → Rd.agree o i d c') → Rd.agree o i c c'

def StmtReadsNone (A : Live₂) (Rd : Reading A) : Prop :=
  ∀ i, Rd.reads i → groupOf A i ≠ .stmt

def MessagesFixed (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (Fr : Frame V A x θ F c σ) (Rd : Reading A)
    (Acc Good : Verity.Exchange → Prop) (miss : ℝ≥0∞) : Prop :=
  ENNReal.ofReal (Claims.Pr fun s : TapeOf V θ F c σ × Coins A x =>
      Acc (Fr.out s.1 s.2) ∧ Good (Fr.out s.1 s.2) ∧
        ∃ i < A.L.V.k x, Rd.msg (Fr.out s.1 s.2) i ≠
          msgAt (Claims.run FlockDeploy.sha512 Fr.P A.L.V x s.1 s.2) i) ≤ miss

def CallsLe (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (Fr : Frame V A x θ F c σ) : Prop :=
  ∀ (ω : TapeOf V θ F c σ) (r : Coins A x),
    (Claims.run FlockDeploy.sha512 Fr.P A.L.V x ω r).pLog.length +
      (Claims.run FlockDeploy.sha512 Fr.P A.L.V x ω r).vLog.length ≤ V.cost θ F c σ

def CutBound (V : Vocab d α) (k : Verity.Registration.Composite) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (Good : Verity.Exchange → Prop) (terms : ℝ≥0∞) :
    Prop :=
  Verity.Game.prob (fun o => V.Verdict k .sampled o ∧ ¬ Good o) (V.game θ F c) σ ≤ terms

def AcceptsRun (A : Live₂) (x : Msg) (Rd : Reading A) (Acc : Verity.Exchange → Prop) : Prop :=
  ∀ o : Verity.Exchange, Acc o → Rd.stmt o = some x →
    ∀ (Ω : Type) (P : Claims.Prover Ω) (ω : Ω) (r : Coins A x),
      (∀ i : Fin (A.L.V.k x), Rd.reads i.val → ∀ c, Rd.coin o i.val = some c → Rd.agree o i.val c (r i)) →
      (∀ i < A.L.V.k x, Rd.msg o i = msgAt (Claims.run FlockDeploy.sha512 P A.L.V x ω r) i) →
      (Claims.run FlockDeploy.sha512 P A.L.V x ω r).out = .acc

def StmtAt (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (Fr : Frame V A x θ F c σ) (Rd : Reading A)
    (Acc : Verity.Exchange → Prop) : Prop :=
  ∀ (ω : TapeOf V θ F c σ) (r : Coins A x), Acc (Fr.out ω r) → Rd.stmt (Fr.out ω r) = some x

def ProjectionBound (V : Vocab d α) (k : Verity.Registration.Composite) (A : Live₂) (x : Msg) (θ : ByteArray)
    (F : Array ByteArray) (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (Fr : Frame V A x θ F c σ)
    (bound : ℝ≥0∞) : Prop :=
  Verity.Game.prob (fun o => V.Verdict k .sampled o) (V.game θ F c) σ ≤
    ENNReal.ofReal (Claims.Pr fun s : TapeOf V θ F c σ × Coins A x =>
      (Claims.run FlockDeploy.sha512 Fr.P A.L.V x s.1 s.2).out = .acc) + bound

/-- `Acc` is the composite's wiring's to owe; at other parameters the sampled verdict also accepts other
forms. -/
def ProjectionOf (V : Vocab d α) (k : Verity.Registration.Composite) (A : Live₂) (x : Msg) (θ : ByteArray)
    (F : Array ByteArray) (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) : Prop :=
  ∀ (Fr : Frame V A x θ F c σ) (Rd : Reading A) (Acc Good : Verity.Exchange → Prop)
    (Ag : Verity.Exchange → Coins A x → Prop) (hide miss terms : ℝ≥0∞),
    (∀ o, o.params = θ → V.Verdict k .sampled o → Acc o) →
    CoinMapDistribution V A x θ F c σ Fr Ag hide → DevAgrees A x Rd Acc Good Ag →
    CoinsAgree A Rd Acc Good .first → CoinsAgree A Rd Acc Good .second →
      CoinsAgree A Rd Acc Good .sessions → StmtReadsNone A Rd → MessagesFixed V A x θ F c σ Fr Rd Acc Good miss →
      CutBound V k θ F c σ Good terms → AcceptsRun A x Rd Acc → StmtAt V A x θ F c σ Fr Rd Acc →
      ProjectionBound V k A x θ F c σ Fr (hide + miss + terms)

def MajorityBound (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (Fr : Frame V A x θ F c σ)
    (tr : Verity.Circuit.Trace (V.circuitOf θ F)) (bind : ℝ≥0∞) : Prop :=
  V.extract θ F c σ = tr →
    ¬ ResourceAccounting.IntegrityHolds (V.partsOf θ F) (V.Opens θ F c) tr (V.certOf θ).ε →
      ENNReal.ofReal (Claims.Pr fun s : TapeOf V θ F c σ × Coins A x × A.L.E.C x =>
        (Claims.run FlockDeploy.sha512 Fr.P A.L.V x s.1 s.2.1).out = .acc ∧
          Claims.Rel.ext A.R x (A.L.E.eval FlockDeploy.sha512 Fr.P x s.1 s.2.1 s.2.2).1) ≤ bind

def CouplingOf (V : Vocab d α) (k : Verity.Registration.Composite) (A : Live₂) (θ : ByteArray)
    (F : Array ByteArray) : Prop :=
  ∀ (x : (c : ByteArray) → Verity.Game.Strategy (V.game θ F c) → Msg)
    (Fr : (c : ByteArray) → (σ : Verity.Game.Strategy (V.game θ F c)) → Frame V A (x c σ) θ F c σ)
    (tr : (c : ByteArray) → Verity.Game.Strategy (V.game θ F c) → Verity.Circuit.Trace (V.circuitOf θ F))
    (bound bind : (c : ByteArray) → Verity.Game.Strategy (V.game θ F c) → ℝ≥0∞),
    (∀ c σ, A.L.Scope (x c σ) ∨ Verity.Game.prob (fun o => V.Verdict k .sampled o) (V.game θ F c) σ = 0) →
    (∀ c σ, ProjectionBound V k A (x c σ) θ F c σ (Fr c σ) (bound c σ)) →
    (∀ c σ, MajorityBound V A (x c σ) θ F c σ (Fr c σ) (tr c σ) (bind c σ)) →
    (∀ c σ, V.extract θ F c σ = tr c σ) →
    (∀ c σ, V.finderCR .sampled θ F c σ → V.premises .sampled θ F c σ →
      ENNReal.ofReal (A.errAt (x c σ) (Fr c σ).P) + (bound c σ + bind c σ) ≤
        V.delta θ .sampled (V.cost θ F c σ)) →
    CouplingTargetTwo V k A θ F fun c σ => bound c σ + bind c σ

end Verity.SampledTwo
