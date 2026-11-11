module

public import Definitions.Claims.RecursionGeneric
public import Proofs.Flock.RecGeneric.Plans
public import Proofs.Flock.RecGeneric.Outer
public import Proofs.Flock.RecGeneric.ZK
public import Proofs.Flock.Recursive.KSStatement

/-!
# Flock as the instance: the server, the inner protocol and the outer extraction

The generic statements (`Definitions.Claims.RecursionGeneric`) at today's recursion:

* **the server** is hm96 at the bound key (`Server.ofBound`): 128-byte commit strings `b ‖ c` with `b = H(rowIn row) ⊕ Φy`
  and `c = H(p_salt ‖ y)`, opened as `Claims.Recursion.Bound.opens` opens. `Open(DE, S)` is the bound record's
  `IVStar` (`IVStarOfBound`), its opened views are the bound record's (`OpenViewOfBound`), it binds where `rowIn` is
  injective (`BindsOfBound`) and its commitments open (`OpensOfBound`);
* **the inner protocol** is the element-level Flock `DE` (`Proto.ofFlock`), its knowledge error `KSFlockLin`'s three terms
  as run finders (`termB`, `termFC`, `termF0`): `KSFlockLin` is its `Proto.KSAbortLin` (`KSAbortLinOfFlockLin`, from
  `ForkPlanIsBAbort`, `CapPlanIsFC`, `ConflictPlanIsF0`), and its plans meet a bad run at most `0`, `2` and `k + 1`
  times as often as one run is bad (`ForkPlanReads`, `CapPlanReads`, `ConflictPlanReads`, Wald's bound);
* **the outer extraction** is today's `E_O` at the compiled record (`Comp.ofRec`), knowledge sound at each statement
  from each session's (`OuterKSOfSessions`, from `Claims.Recursion.OuterKS`, which `outerKS_of_flockKS` gives from
  Flock's KS at the outer record).

There `Comp.err` is at most `KSLin2`'s bound term by term but the link's: `κ`, `γ` and `β` equal `KSLin2`'s
(`β = βdev2`, from `d = 0, 2, k + 1`); `B`'s term at the reruns and the judged run's `B_I` add to `(4k_I + 1)·Pr[B_I]`
(`ForkPlansLeBinnerV`); `FC`'s and `F0`'s plans at the reruns are `FCI` and `F0I` (`CapPlanIsFCI`,
`ConflictPlanIsF0I`); `Comp.PrOut` is `0`, since `R_O` holds at every witness list (`Comp.ofRec`); and the binding
error stays the gated link's (`linkPlan` at `verModel`). So L1 gives `KSLinkOn`,
`KSLin2`'s bound with that binding error at the deployed decoders (`compDep`: the extractor decodes the reruns' openings
as the deployment does, a counted program), at the statements where the instance's contract holds (`ContractOn`,
`RecoversLinkOn`; at the deployment's dense statements `ContractDense`), and at
every statement `Flock.Guarantees.RecursiveKSEveryAtRec`. The contract reads V\*'s two facts only where some run
accepts, and the deployed compiled verifier accepts only dense statements, so at the deployment every statement needs
V\*'s dense facts alone.

The binding error is the gated link's on the composite's reruns (`linkPlan`), whose cost charges every rerun it names
(`RunFinder.finder`), so bits read off it charge what they search.

Statements only: definitions, no proofs.
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion

/-! ## The plans' reads and the inner protocol's runs -/

/-- **`B`'s plan asks about no run** (`d = 0`). -/
def ForkPlanReads (Coin : Type) [Fintype Coin] [Nonempty Coin] : Prop :=
  ∀ k : Msg → ℕ, (forkPlan (Coin := Coin) k).ReadsLe fun _ => 0

/-- **`FC`'s plan meets a bad run at most twice as often as one run is bad** (`d = 2`): its first run, and its loop of
forks gated by that run, which Wald's bound counts once. -/
def CapPlanReads (Coin : Type) [Fintype Coin] [Nonempty Coin] : Prop :=
  ∀ (Cp : Caps Coin) (M : ℕ), (capPlan Cp M).ReadsLe fun _ => 2

/-- **`F0`'s plan meets a bad run at most `k + 1` times as often as one run is bad**: its first run, and each of its
`k` waits gated by it, which stops at its first accepted run. -/
def ConflictPlanReads (Coin : Type) [Fintype Coin] [Nonempty Coin] : Prop :=
  ∀ (k : Msg → ℕ) (R : ℕ), (conflictPlan (Coin := Coin) k R).ReadsLe fun x => k x + 1

/-- **`B`'s plan at a protocol's runs that may abort is `B` there**: `Pr[forkPlan finds a collision] = Pr[B]`. -/
def ForkPlanIsBAbort (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] : Prop :=
  ∀ (H : Msg → Msg) (V : Verifier Coin) (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (A : Type)
    [Fintype A] [Nonempty A] (keep : Ω → (Fin (V.k x) → Coin) → A → Bool),
    (forkPlan V.k).PrAt H (runModel H V P x keep) x = PrBAbort H V P x keep

/-- **`FC`'s plan at a protocol's runs that may abort is `FC` there**, at every cap reader and loop bound. -/
def CapPlanIsFC (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] : Prop :=
  ∀ (H : Msg → Msg) (V : Verifier Coin) (Cp : Caps Coin) (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (A : Type) [Fintype A] [Nonempty A] (keep : Ω → (Fin (V.k x) → Coin) → A → Bool) (M : ℕ),
    (capPlan Cp M).PrAt H (runModel H V P x keep) x = PrFC H V Cp P x keep M

/-- **`F0`'s plan at a protocol's runs that may abort is `F0` there**, at every count of waits and loop bound. -/
def ConflictPlanIsF0 (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] : Prop :=
  ∀ (H : Msg → Msg) (V : Verifier Coin) (k : Msg → ℕ) (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (A : Type) [Fintype A] [Nonempty A] (keep : Ω → (Fin (V.k x) → Coin) → A → Bool) (R : ℕ),
    (conflictPlan k R).PrAt H (runModel H V P x keep) x = PrF0 H V P x keep (k x) R

/-- **The server's commitments open** (completeness, at a hash whose outputs are 64 bytes): a commit string the server
makes for a row under a salt opens to that row under that salt. -/
def Server.Opens (S : Server) : Prop :=
  ∀ H : Msg → Msg, Out64 H → ∀ (row : Msg) (y : S.Salt),
    ((S.opens ((S.commit row y).eval H).1 row y).eval H).1 = true

section Instance

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

/-! ## The server: hm96 at the bound key -/

/-- **hm96 as the server**, at the bound key: a salt is `saltLen` bytes; a commit string is 128 bytes,
`b ‖ c` with `b = H(rowIn row) ⊕ Φy` and `c = H(p_salt ‖ y)`; it opens to a row under a salt as `Bound.opens` checks; a
round's rows join and the last message reads as the bound record's (`join`, `final`), at its schedule (`nAt`). -/
def Server.ofBound (Bd : Bound Coin Col T) : Server where
  Salt := Fin Bd.saltLen → UInt8
  width := 128
  nAt := Bd.nAt
  join := Bd.join
  final := Bd.final
  commit row y := .query (Bd.saltPrefix ++ List.ofFn y) fun c => .query (Bd.rowIn row) fun d =>
    .ret (xorMask d (Bd.mask y) ++ c)
  opens := Bd.opens

/-- **`Open(DE, S)` at hm96 is the bound record's `IVStar`** (`Server.IVStar` is `Bound.IVStar` read through a server;
expected by unfolding). -/
def IVStarOfBound (Bd : Bound Coin Col T) : Prop := (Server.ofBound Bd).IVStar Bd.DE.V Bd.ans = Bd.IVStar

/-- **The server's opened views are the bound record's** (expected by unfolding). -/
def OpenViewOfBound (Bd : Bound Coin Col T) : Prop := (Server.ofBound Bd).openView = Bd.openView

/-- **hm96 binds where a row's digest input is injective**: `Bound.Binds` gives `Server.Binds`. Two passing openings of
one commit string under different salts collide on the salts' hashes (`c`), and under one salt on the rows' digests
(`b ⊕ Φy`). -/
def BindsOfBound (Bd : Bound Coin Col T) : Prop := Bd.Binds → (Server.ofBound Bd).Binds

/-- **hm96's commitments open** at a hash whose outputs are 64 bytes: `b ⊕ Φy` unmasks to the row's digest. -/
def OpensOfBound (Bd : Bound Coin Col T) : Prop := (Server.ofBound Bd).Opens

/-! ## The inner protocol: the element-level Flock -/

/-- **`B`'s term** at `k` rounds: `forkPlan k` at every loop bound, weight `4·k(x)`, no tail, no run asked. -/
noncomputable def termB (k : Msg → ℕ) : Term Coin where
  F _ := forkPlan k
  w x := 4 * (k x : ℝ)
  tail _ _ := 0
  d _ := 0

/-- **`FC`'s term** at the caps `Cp`: `capPlan Cp M`, weight `4·W(x)`, tail `1/(M+1)`, two runs met. -/
noncomputable def termFC (Cp : Caps Coin) : Term Coin where
  F := capPlan Cp
  w x := 4 * (Cp.width x : ℝ)
  tail _ M := 1 / ((M : ℝ) + 1)
  d _ := 2

/-- **`F0`'s term** at `k(x)` waits: `conflictPlan k R`, weight `4`, tail `(k + 1)/(R+1)`, `k + 1` runs met. -/
noncomputable def termF0 (k : Msg → ℕ) : Term Coin where
  F := conflictPlan k
  w _ := 4
  tail x R := ((k x : ℝ) + 1) / ((R : ℝ) + 1)
  d x := k x + 1

/-- **The element-level Flock as the inner protocol**, at an inner decoder and the caps' reader: `Inner.ofFlock` at
`DE`, its rounds answered by their coins (`ans`), and `KSFlockLin`'s three terms. -/
noncomputable def Proto.ofFlock (Bd : Bound Coin Col T) (decI : List (ℕ × Col) → List T) (Cp : Caps Coin) :
    Proto Coin Claims.Flock.Wit :=
  { Inner.ofFlock Bd.DE decI with
    ans := Bd.ans
    nT := 3
    terms := ![termB Bd.DE.V.k, termFC Cp, termF0 Bd.DE.k] }

/-- **The element-level Flock is well formed as an inner protocol** where its answers decode to their coins and its
tables' errors are nonnegative (from `ForkPlanReads`, `CapPlanReads`, `ConflictPlanReads`). -/
def ProtoWFOfFlock (Bd : Bound Coin Col T) (decI : List (ℕ × Col) → List T) (Cp : Caps Coin) : Prop :=
  (∃ dec : Msg → Coin, ∀ c, dec (Bd.ans c) = c) → (∀ x, 0 ≤ Bd.DE.εtable x) → (Proto.ofFlock Bd decI Cp).WF

/-- **`KSFlockLin` is the element-level Flock's `Proto.KSAbortLin`**, at every decoder that meets the spec, the loop
bounds `(·, M, R)` (from `ForkPlanIsBAbort`, `CapPlanIsFC`, `ConflictPlanIsF0`). -/
def KSAbortLinOfFlockLin [Zero Col] (H : Msg → Msg) (Bd : Bound Coin Col T) (Cp : Caps Coin) : Prop :=
  KSFlockLin H Bd.DE Cp → ∀ decI, decOK Bd.DE decI → (Proto.ofFlock Bd decI Cp).KSAbortLin H

/-! ## The outer extraction: today's `E_O` at the compiled record -/

variable {Col' T' : Type} [DecidableEq Col']

/-- **Today's compiled record and `E_O` as a `Comp`**, at the inner verifier `IV`, the inner protocol `I` whose rounds
`E_O` skips, and the outer Flock at a decoder: the compiled verifier, its inner statement and form, `E_O`
(`Claims.Recursion.EO`) with the relation every witness list meets (its sessions' checks are inside `E_O`), the inner
messages the witnesses give (`RecAt.msgs`), the calls a finder searches at a rerun (`Deploy.callsAt`) and the compiled
verifier's own at a run's staged V\* statements (`setupCalls`, as `Claims.Recursion.drawCalls` reads them), and the errors
`κ_O = Σ_s κ_s`, `γ_O = (Σ_s √(γ_s·k(x)))²`. -/
noncomputable def Comp.ofRec (D : RecAt Coin) (IV : Verifier Coin) {W : Type} (I : Claims.Recursion.Inner Coin W)
    (DO : Claims.Flock.Deploy Coin Col' T') (decO : List (ℕ × Col') → List T') : Comp Coin where
  V := (D.record IV).V
  xI := Claims.Recursion.xI (D.record IV)
  parses x := ∃ s, (D.record IV).parse x = some s ∧ s.circ = none
  WO := List Claims.Flock.Wit
  RO _ _ := True
  EO := Claims.Recursion.EO (D.record IV) I DO decO
  msgs := D.msgs
  calls H x r ws := (D.record IV).callsAt H x r ws
  own H x ms := setupCalls H (D.record IV)
    ((D.record IV).vstar x (ms.take (I.V.k (Claims.Recursion.xI (D.record IV) x) + (D.record IV).kS x)))
  κO x := ∑ s : Fin ((D.record IV).nS x), ((D.record IV).nums x s).κ
  γO x := (∑ s : Fin ((D.record IV).nS x), Real.sqrt (((D.record IV).nums x s).γ * (D.record IV).V.k x)) ^ 2

/-- **`E_O` is knowledge sound at every statement from each session's**: `Claims.Recursion.OuterKS` (each session's
`E^rep` fails with probability at most `κ_s + √(γ_s·k(x)·Pr[B])`) gives `Comp.OuterKSAt` at `Comp.ofRec`, by
`PrBad_EO_le` and `Σ_s √(γ_s·k·p) = √((Σ_s √(γ_s·k))²·p)`. -/
def OuterKSOfSessions [Zero Col'] (H : Msg → Msg) (D : RecAt Coin) (IV : Verifier Coin) {W : Type}
    (I : Claims.Recursion.Inner Coin W) (DO : Claims.Flock.Deploy Coin Col' T') : Prop :=
  OuterKS H (D.record IV) I DO → ∀ decO, decOK DO decO → ∀ x, (Comp.ofRec D IV I DO decO).OuterKSAt H x

/-- **The instance at the bound record**: `Comp.ofRec` at `IVStar`, with `E_O` skipping the bound record's inner
rounds. -/
noncomputable abbrev compAt (D : RecAt Coin) (Bd : Bound Coin Col T) (decI : List (ℕ × Col) → List T)
    (DO : Claims.Flock.Deploy Coin Col' T') (decO : List (ℕ × Col') → List T') : Comp Coin :=
  Comp.ofRec D Bd.IVStar (Inner.ofFlock Bd.deploy decI) DO decO

/-- **The instance at the deployed decoders**: `compAt` at the inner and outer records' own decoders
(`Deploy.decode`, the candidate of the counted `Deploy.dec`), the ones the deployment runs. `Bound.deploy` keeps
`DE`'s `dec`, so the inner one is `DE`'s. -/
noncomputable abbrev compDep (D : RecAt Coin) (Bd : Bound Coin Col T) (DO : Claims.Flock.Deploy Coin Col' T') :
    Comp Coin :=
  compAt D Bd Bd.deploy.decode DO DO.decode

/-! ## The composite's terms against `KSLin2`'s -/

/-- **`B`'s plan at the reruns is at most `B_I`**: at a statement where the calls given witnesses record are true and
those of no witnesses are (`CallsNil` at `x`), `forkPlan` on the reruns as the inner view accepts (`innerModel`) and as
`E_O` gives (`verModel`) each finds a collision at most as often as `BinnerV`, which searches a superset of true
evaluations at the same two reruns. -/
def ForkPlansLeBinnerV (H : Msg → Msg) (D : RecAt Coin) (Bd : Bound Coin Col T)
    (DO : Claims.Flock.Deploy Coin Col' T') : Prop :=
  ∀ decI decO (x : Msg), (compAt D Bd decI DO decO).CallsTrue H x →
    (∀ e ∈ (D.record Bd.IVStar).calls x [], H e.1 = e.2) →
    ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω),
      (forkPlan (Coin := Coin) Bd.DE.V.k).PrAt H
          ((compAt D Bd decI DO decO).innerModel ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans) H P x)
          ((compAt D Bd decI DO decO).xI x) ≤
        PrBinnerV H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO P x ∧
      (forkPlan (Coin := Coin) Bd.DE.V.k).PrAt H
          ((compAt D Bd decI DO decO).verModel ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans) H P x)
          ((compAt D Bd decI DO decO).xI x) ≤
        PrBinnerV H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO P x

/-- **`FC`'s plan at the reruns is `FCI`**, where the record's inner statement is the compiled one's (`RecAt.xI`): both
loop over reruns forked at the cap's inner round from a rerun whose inner view accepts and whose inner coins query the
cap, and search the two reruns' recorded and inner calls. -/
def CapPlanIsFCI (D : RecAt Coin) (Bd : Bound Coin Col T) (DO : Claims.Flock.Deploy Coin Col' T') (Cp : Caps Coin) :
    Prop :=
  (∀ x, D.xI x = Claims.Recursion.xI (D.record Bd.IVStar) x) →
  ∀ (H : Msg → Msg) decI decO (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (M : ℕ),
    (capPlan Cp M).PrAt H
        ((compAt D Bd decI DO decO).innerModel ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans) H P x)
        ((compAt D Bd decI DO decO).xI x) =
      PrFCI H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO Cp P x M

/-- **`F0`'s plan at the reruns is `F0I`**, where the record's inner statement is the compiled one's. -/
def ConflictPlanIsF0I (D : RecAt Coin) (Bd : Bound Coin Col T) (DO : Claims.Flock.Deploy Coin Col' T') : Prop :=
  (∀ x, D.xI x = Claims.Recursion.xI (D.record Bd.IVStar) x) →
  ∀ (H : Msg → Msg) decI decO (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (R : ℕ),
    (conflictPlan Bd.DE.k R).PrAt H
        ((compAt D Bd decI DO decO).innerModel ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans) H P x)
        ((compAt D Bd decI DO decO).xI x) =
      PrF0I H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO P x R

/-- **`KSLin2`'s bound with the gated link, at the composite**: at every statement `G` admits, the composite at today's
compiled record over hm96 and the element-level Flock, at the deployed decoders (`compDep`, `Comp.E`, which reads no
`H`, so its extractor is the deployment's counted decoding of the reruns' openings and no other) meets
`κ + √(γ·Pr[B]) + (4k_I + 1)·Pr[B_I] + 4·W_I·(Pr[FCI] + 1/(M+1)) + 4·(Pr[F0I] + (k_I + 1)/(R+1)) +
β₂·(Pr[link(M)] + 1/(M+1))`: `KSLin2`'s bound with its binding error the gated link's on the composite's reruns
(`linkPlan` at `verModel`, whose cost charges every rerun it names) in place of `FdevV`'s. -/
def KSLinkOn [Zero Col] [Zero Col'] (H : Msg → Msg) (D : RecAt Coin) (Bd : Bound Coin Col T)
    (DO : Claims.Flock.Deploy Coin Col' T') (Cp : Caps Coin) (G : Msg → Prop) : Prop :=
  ∀ (x : Msg), G x → ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (M R : ℕ),
    PrBad H (compDep D Bd DO).V ((compDep D Bd DO).R (Proto.ofFlock Bd Bd.deploy.decode Cp))
        ((compDep D Bd DO).E (Proto.ofFlock Bd Bd.deploy.decode Cp) (Server.ofBound Bd)) x P ≤
      Claims.Recursion.κ (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy Bd.deploy.decode) x +
        Real.sqrt (Claims.Recursion.γ (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy Bd.deploy.decode) x *
          PrB H (D.record Bd.IVStar).V P x) +
        (4 * (Inner.ofFlock Bd.deploy Bd.deploy.decode).V.k (Claims.Recursion.xI (D.record Bd.IVStar) x) + 1) *
          PrBinnerV H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy Bd.deploy.decode) DO DO.decode P x +
        4 * Cp.width (Claims.Recursion.xI (D.record Bd.IVStar) x) *
          (PrFCI H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy Bd.deploy.decode) DO DO.decode Cp P x M +
            1 / ((M : ℝ) + 1)) +
        4 * (PrF0I H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy Bd.deploy.decode) DO DO.decode P x R +
          (((Inner.ofFlock Bd.deploy Bd.deploy.decode).k (Claims.Recursion.xI (D.record Bd.IVStar) x) : ℝ) + 1) /
            ((R : ℝ) + 1)) +
        βdev2 (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy Bd.deploy.decode) Cp x *
          ((linkPlan (Coin := Coin) Bd.DE.V.k M).PrAt H
              ((compDep D Bd DO).verModel ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans) H P x)
              ((compDep D Bd DO).xI x) +
            1 / ((M : ℝ) + 1))

/-- **The instance's contract at the statements `G` admits**: at each, the calls of no witnesses are true, and at every
inner and outer decoder that meets the list-decoding spec (`decOK`; `RecoversLinkOn` reads it at the deployed ones) the
compiler's contract (`Comp.Contract`) at `Open(DE, hm96)` and `DE`'s reading of opened views. -/
def ContractOn [Zero Col] [Zero Col'] (H : Msg → Msg) (D : RecAt Coin) (Bd : Bound Coin Col T)
    (DO : Claims.Flock.Deploy Coin Col' T') (Cp : Caps Coin) (G : Msg → Prop) : Prop :=
  ∀ x, G x → (∀ e ∈ (D.record Bd.IVStar).calls x [], H e.1 = e.2) ∧
    ∀ decI, decOK Bd.deploy decI → ∀ decO, decOK DO decO →
      (compAt D Bd decI DO decO).Contract H ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)
        (fun v => (Proto.ofFlock Bd decI Cp).opens ((Server.ofBound Bd).openView v)) x

/-- **L1 at Flock gives `KSLinkOn` at the statements `G` admits.** From the generic theorem (`CompositionKS`), the
element-level Flock's
KS with no root (`KSFlockLin` at `DE`), each outer session's KS (`OuterKS`), the sessions' nonnegative table errors,
the record's inner statement being the compiled one's, answers that decode to their coins, `DE`'s nonnegative table
errors and last step free of its last coin (F8, `Bound.LastStepFree`, proved at the deployed reader and plain
verifier: `RecDeploy.lastCoinFree_ofFlock_R`, `RecDeploy.lastCoinFree_ofFlock_V`), `rowIn` injective (`Bound.Binds`,
as the bridge takes it, `Claims.Recursion.Bridge`), the deployed decoders meeting the list-decoding spec (`decOK`), and
at each statement `G` admits the calls of no witnesses true and the compiler's contract at `Open(DE, hm96)` and `DE`'s
reading of opened views: `KSLinkOn` at `G`, through `ProtoWFOfFlock`, `KSAbortLinOfFlockLin`, `OuterKSOfSessions`,
`IVStarOfBound`, `OpenViewOfBound`, `BindsOfBound` and the three term comparisons, each at the deployed decoders. The
contract's `own` evaluations are the staged statements' setup calls, true by evaluation (`setupCalls`). The error is
`Comp.err` with `κ`, `γ` and `β` equal to `KSLin2`'s, `B`'s, `FC`'s and `F0`'s terms at most `KSLin2`'s, `Comp.PrOut`
at `0` at both its weights, and the gated link's term as it is. -/
def RecoversLinkOn [Zero Col] [Zero Col'] (H : Msg → Msg) (D : RecAt Coin) (Bd : Bound Coin Col T)
    (DO : Claims.Flock.Deploy Coin Col' T') (Cp : Caps Coin) (G : Msg → Prop) : Prop :=
  CompositionKS → KSFlockLin H Bd.DE Cp →
    OuterKS H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy Bd.deploy.decode) DO →
    TableErrNonneg (D.record Bd.IVStar) → (∀ x, D.xI x = Claims.Recursion.xI (D.record Bd.IVStar) x) →
    (∃ dec : Msg → Coin, ∀ c, dec (Bd.ans c) = c) → (∀ x, 0 ≤ Bd.DE.εtable x) →
    (Proto.ofFlock Bd Bd.deploy.decode Cp).LastCoinFree H → Bd.Binds →
    decOK Bd.DE Bd.deploy.decode → decOK DO DO.decode → ContractOn H D Bd DO Cp G →
    KSLinkOn H D Bd DO Cp G

end Instance

end Claims.RecGeneric

namespace Flock.Guarantees

open Claims (Msg Out)
open FlockDeploy (Coin FinalP)

/-- **Knowledge soundness of the recursive verifier at every statement, at a compiled record** `D`: the composite's
bound with the gated link (`Claims.RecGeneric.KSLinkOn`) at every statement, at SHA-512, the deployed relation's bound
record at the reader `rd`, the deployed caps and the deployed decoders. `Claims.RecGeneric.RecoversLinkOn` at
`G := fun _ => True` gives it from
the generic theorem and the instance's contract at every statement; V\*'s two facts are read only where some run
accepts, so at a record that accepts only dense statements they are the dense ones. -/
def RecursiveKSEveryAtRec (D : Claims.Recursion.RecAt Coin) (rd : Msg → Option FinalP)
    (final : Msg → Option (List (List (Msg × RecDeploy.Salt)) × Msg)) (wE w : Msg → ℝ) : Prop :=
  Claims.RecGeneric.KSLinkOn FlockDeploy.sha512 D
    (RecDeploy.bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR rd)
      (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE FlockDeploy.joinR
      RecDeploy.rowInDeployed final w)
    (RecDeploy.deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep fun _ => True

/-- **Knowledge soundness of the recursive verifier at every statement, at the deployed reader**: `RecursiveKS` with
no restriction to dense statements (`RecursiveKSEveryAtRec` at the deployment's record and values). It is the Flock
instance of the generic composition (`Claims.RecGeneric.CompositionKS`): `RecoversLinkOn` takes it. -/
def RecursiveKSEvery : Prop :=
  ∀ (b : RecDeploy.Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8),
    RecursiveKSEveryAtRec
      (RecDeploy.recBoundAt _root_.Flock.Readable.relation
        { RecDeploy.deployedB b with tr := RecDeploy.recTr aux nOf hd recRest }
        (RecDeploy.DeployedR.layR RecDeploy.OpenRows.elemR))
      RecDeploy.DeployedR.rdR RecDeploy.DeployedR.finalR RecDeploy.DeployedR.wER RecDeploy.DeployedR.wR

end Flock.Guarantees

namespace Claims.RecGeneric

open FlockDeploy (Coin FinalP)

/-- **F7, the instance's contract at the dense statements, at a compiled record** `D`: `ContractOn` at SHA-512, the
deployed relation's bound record at the reader `rd` (as `Flock.Guarantees.RecursiveKSAtRec` reads it), the deployed
outer record and caps, and `RecDeploy.DenseAt`. `RecDeploy.ksLin2_route` proves these facts inside one proof, about
`Claims.Recursion`'s samples; this is them about the composite's (`compAt`). -/
def ContractDenseAtRec (D : Claims.Recursion.RecAt Coin) (rd : Msg → Option FinalP)
    (final : Msg → Option (List (List (Msg × RecDeploy.Salt)) × Msg)) (wE w : Msg → ℝ) : Prop :=
  ContractOn FlockDeploy.sha512 D
    (RecDeploy.bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR rd)
      (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE FlockDeploy.joinR
      RecDeploy.rowInDeployed final w)
    (RecDeploy.deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep RecDeploy.DenseAt

/-- **F7 at the deployment**: `ContractDenseAtRec` at the deployment's record and values, for every other field of the
brings and every record the challenger's live-coin part keeps (as `Flock.Guarantees.RecursiveKS` quantifies them). -/
def ContractDense : Prop :=
  ∀ (b : RecDeploy.Brings) (aux : Msg → ℕ → _root_.Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → _root_.Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8),
    ContractDenseAtRec
      (RecDeploy.recBoundAt _root_.Flock.Readable.relation
        { RecDeploy.deployedB b with tr := RecDeploy.recTr aux nOf hd recRest }
        (RecDeploy.DeployedR.layR RecDeploy.OpenRows.elemR))
      RecDeploy.DeployedR.rdR RecDeploy.DeployedR.finalR RecDeploy.DeployedR.wER RecDeploy.DeployedR.wR

end Claims.RecGeneric
