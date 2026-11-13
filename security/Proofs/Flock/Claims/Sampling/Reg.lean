module

public import Spot.Registration
public import Proofs.Flock.Claims.Sampling.Code
public import Proofs.Flock.Claims.Sampling.Select.DrawLaw
public import Properties.Spot.Second

/-!
# The registrations the samples of record read, and the code's round order

What the deployed sampled verifier's samples read of a registration, and the code with its checked units and rounds set
from them, with no proof module beneath: `Reg`, the registration `Spot.Run.run` hands `Select.select`, and its
sample of record (`Reg.checked`); `RegTwo`, a two-stage registration, its replay sample, the second stage's proof
blocks and its checked units (`RegTwo.sample`, `RegTwo.templateAt`, `RegTwo.blockAt`, `RegTwo.checked`); `codeTwo`, the
code with those
checked units, and `replayTwo`, the replay sample of record; and `Code.atOrder`, the code at its round order (`Order`):
its second stage the rounds after the first stage's coin, and its sessions' rounds those after the second stage's, and
`codeTwoAt`, the two-stage code at its round order, each stage's bytes read off its own coin (`bytesFirst`,
`bytesSecond`); and the same with a second stage that reads several rounds' coins: `StageOrder`, a round order whose
second stage reads the `R` rounds up to `second` (`Order.stage` at one round), and `codeTwoStage`, the two-stage code
at it (`bytesStage`).
-/

@[expose] public section

namespace SamplingDeploy

open Claims Claims.Sampling Finset Properties.Spot
open Lean (Json)

/-! ## The registrations and their samples -/

/-- **A registration as the sample of record reads it**, what `Spot.Run.run` passes to `Select.select`: the
verifier's own copies (`Run.Input.params.verifier`, whose program, query and work table the sample reads), the selection
rule and the challenger's closure map. -/
structure Reg where
  verifier : Spot.Registration.Verifier
  rule : Spot.Select.Rule
  closure : Option (Array (Int × Array Int))

def Reg.program (g : Reg) : Spot.Program := g.verifier.program

def Reg.query (g : Reg) : Json := g.verifier.query

def Reg.table (g : Reg) : Option Json := g.verifier.workTable

/-- The registration's layout (`Slots.layout`), empty where its query doesn't apply. -/
def Reg.layout (g : Reg) : Spot.Slots.SlotOrder :=
  match Spot.Slots.layout g.program g.query with
  | .ok l => l
  | .error _ => default

/-- **The sample of record's checked units** on the bytes `src`: `Slots.select`'s, and every slot of the layout where it
is refused (`drawChecked`). -/
def Reg.checked (g : Reg) (src : ByteArray) : List ℕ :=
  (drawChecked g.rule g.program g.query g.table g.closure g.layout src).toList

/-- **A two-stage registration as the samples of record read it**: the boundary registration the first stage samples on
(`first`), each template's proof blocks in a replay block (`ns`, a cell each) and the second stage's count in each of
its cells (`ks`), the statement's proof block at position `p` of template `t`'s cell in replay block `ρ`
(`blockOf ρ t p`), and the proof blocks checked whatever the stages sample (`always`). -/
structure RegTwo where
  first : Reg
  ns : Array ℕ
  ks : Array ℕ
  blockOf : ℕ → ℕ → ℕ → ℕ
  always : ℕ → Bool

/-- **The replay sample** on the bytes `src`: the units the first stage proves (`Select.select`'s), which `Run.run`
hands the second stage; none where it is refused. -/
def RegTwo.sample (g : RegTwo) (src : ByteArray) : Array ℕ :=
  match Spot.Select.select g.first.rule g.first.program g.first.query g.first.table g.first.closure src with
  | .ok (_, proved, _) => proved
  | .error _ => #[]

/-- Interior unit `j`'s template over a replay sample of `q`: the first whose cells end past `j`, the interior being
template-major (`cellOff`); `ns.size` past them all. -/
def RegTwo.templateAt (g : RegTwo) (q j : ℕ) : ℕ :=
  ((List.range g.ns.size).find? fun t => decide (j < cellOff q g.ns (t + 1))).getD g.ns.size

/-- Interior unit `j`'s proof block: position `(j - off) % n` of its template `t`'s cell in replay block
`sample[(j - off) / n]`, where `t`'s cells start at `off` and hold `n` each. -/
def RegTwo.blockAt (g : RegTwo) (sample : Array ℕ) (j : ℕ) : ℕ :=
  let t := g.templateAt sample.size j
  let o := cellOff sample.size g.ns t
  let n := g.ns.getD t 0
  g.blockOf (sample.getD ((j - o) / n) 0) t ((j - o) % n)

/-- **The checked proof blocks** below `N`, at first-stage bytes `src₁` and second-stage bytes `src₂`: those the second
stage proves (`Select.selectSecond` on the replay sample) and those always checked; all of them where the second stage
is refused. -/
def RegTwo.checked (g : RegTwo) (N : ℕ) (src₁ src₂ : ByteArray) : List ℕ :=
  match Spot.Select.selectSecond (g.sample src₁) g.ns g.ks src₂ with
  | .ok (draw, _) => (List.range N).filter fun v =>
      g.always v || (Spot.Draw.proved draw).any fun j => g.blockAt (g.sample src₁) j.toNat == v
  | .error _ => List.range N

/-- **The two-stage sampled verifier's code with its Lean samples**: `c` with `checked` the two stages' on the
registration `reg x`, over the bytes `bytes₁ x` and `bytes₂ x` of the coins. -/
def codeTwo (c : Code) (reg : Msg → Option RegTwo) (L₁ L₂ : Msg → ℕ)
    (bytes₁ : (x : Msg) → (Fin (c.V.k x) → FlockDeploy.Coin) → Fin (L₁ x) → Fin 256)
    (bytes₂ : (x : Msg) → (Fin (c.V.k x) → FlockDeploy.Coin) → Fin (L₂ x) → Fin 256) : Code :=
  { c with
    checked := fun x r =>
      ((reg x).map fun g => g.checked (units c x) (bytesOf (bytes₁ x r)) (bytesOf (bytes₂ x r))).getD [] }

/-- **The replay sample of record**: the first stage's checked slots on the replay layout (`Reg.checked`). -/
def replayTwo (c : Code) (reg : Msg → Option RegTwo) (L₁ : Msg → ℕ)
    (bytes₁ : (x : Msg) → (Fin (c.V.k x) → FlockDeploy.Coin) → Fin (L₁ x) → Fin 256) (x : Msg)
    (r : Fin (c.V.k x) → FlockDeploy.Coin) : List ℕ :=
  ((reg x).map fun g => g.first.checked (bytesOf (bytes₁ x r))).getD []

/-! ## The round order -/

/-- **The deployed round order** of a statement's `k` rounds: the coin the first stage reads (`first`) comes before the
coin the second stage reads (`second`). -/
structure Order (k : ℕ) where
  first : Fin k
  second : Fin k
  lt : first < second

/-- **The code at its round order**: `c` with its second stage the rounds after the first stage's coin, and each
session's rounds those of `c`'s after the second stage's coin. -/
def Code.atOrder (c : Code) (j : (x : Msg) → Order (c.V.k x)) : Code :=
  { c with
    stage2 := fun x => univ.filter fun t => (j x).first < t
    rounds := fun x u => (c.rounds x u).filter fun t => (j x).second < t }

section At

variable (c : Code) (reg : Msg → Option RegTwo) (L₁ L₂ : Msg → ℕ) (j : (x : Msg) → Order (c.V.k x))
  (b₁ : (x : Msg) → FlockDeploy.Coin → Fin (L₁ x) → Fin 256) (b₂ : (x : Msg) → FlockDeploy.Coin → Fin (L₂ x) → Fin 256)

/-- The first stage's bytes: `b₁ x` on the first stage's coin. -/
def bytesFirst : (x : Msg) → (Fin ((c.atOrder j).V.k x) → FlockDeploy.Coin) → Fin (L₁ x) → Fin 256 :=
  fun x r => b₁ x (r (j x).first)

/-- The second stage's bytes: `b₂ x` on the second stage's coin. -/
def bytesSecond : (x : Msg) → (Fin ((c.atOrder j).V.k x) → FlockDeploy.Coin) → Fin (L₂ x) → Fin 256 :=
  fun x r => b₂ x (r (j x).second)

/-- **The two-stage sampled verifier's code at its round order.** -/
abbrev codeTwoAt : Code := codeTwo (c.atOrder j) reg L₁ L₂ (bytesFirst c L₁ j b₁) (bytesSecond c L₂ j b₂)

end At

/-! ## A second stage of several rounds -/

/-- **The round order with a second stage of `R` rounds**: the second stage reads the coins of the `R` rounds up to
`second`, each after the first stage's coin. -/
structure StageOrder (k R : ℕ) extends Order k where
  fits : first.val + R ≤ second.val

/-- **A round order is a stage order of one round.** -/
def Order.stage {k : ℕ} (o : Order k) : StageOrder k 1 := ⟨o, by have := Fin.lt_def.1 o.lt; omega⟩

/-- The second stage's `i`-th round, in the run's order; its last is `second`. -/
def StageOrder.round {k R : ℕ} (o : StageOrder k R) (i : Fin R) : Fin k :=
  ⟨o.second.val - (R - 1 - i.val), lt_of_le_of_lt (Nat.sub_le _ _) o.second.isLt⟩

/-- A stage order's round order: the first stage's coin and the second stage's last. -/
abbrev orderOf {c : Code} {R : Msg → ℕ} (j : (x : Msg) → StageOrder (c.V.k x) (R x)) (x : Msg) : Order (c.V.k x) :=
  (j x).toOrder

section Stage

variable (c : Code) (reg : Msg → Option RegTwo) (L₁ L₂ : Msg → ℕ)
  (b₁ : (x : Msg) → FlockDeploy.Coin → Fin (L₁ x) → Fin 256) (R : Msg → ℕ)
  (s : (x : Msg) → StageOrder (c.V.k x) (R x)) (bs : (x : Msg) → (Fin (R x) → FlockDeploy.Coin) → Fin (L₂ x) → Fin 256)

/-- The second stage's bytes: `bs x` on the coins of the second stage's rounds. -/
def bytesStage : (x : Msg) → (Fin ((c.atOrder (orderOf s)).V.k x) → FlockDeploy.Coin) → Fin (L₂ x) → Fin 256 :=
  fun x r => bs x fun i => r ((s x).round i)

/-- **The two-stage sampled verifier's code at a stage order**: `codeTwo` at the stage order's round order, its second
stage's bytes read off the coins of the second stage's rounds. -/
abbrev codeTwoStage : Code :=
  codeTwo (c.atOrder (orderOf s)) reg L₁ L₂ (bytesFirst c L₁ (orderOf s) b₁) (bytesStage c L₂ R s bs)

end Stage

end SamplingDeploy
