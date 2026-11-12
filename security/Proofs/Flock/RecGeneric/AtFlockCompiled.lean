module

public import Proofs.Flock.RecGeneric.AtFlock

/-!
# L2 at Flock: the compiler at today's compiled record, over Flock `--zk`

The compiler statement (`Claims.RecGeneric.CompilerKS`) at today's recursion: Π_out is
Flock `--zk` at the outer record's deployed decoder (`Claims.SamplingGeneric.Arg.ofFlock DO`, its assemble step
counted), and the compiler is today's compiled record read session by session (`Compiled.ofRec`, `recCompiler`):

* its verifier, parser and inner statement are the record's, as at `Comp.ofRec`;
* its rounds before the sessions are the inner session's and the staging rounds (`pre`), and the sessions' statements
  are the record's V\* statements at those rounds' messages (`stmts`);
* session `s`'s public shape is its numbers' waits, truncation, rounds and `N₀` (`Deploy.nums`), and its errors'
  bounds are their `κ` and `γ`, so `κ_O`, `γ_O` at `Compiled.toComp` are `Comp.ofRec`'s;
* the inner messages, the calls a finder searches and the compiled verifier's own are `Comp.ofRec`'s at the witnesses'
  second components.

Its `E_O` is `Compiled.EO`: each session's rerun is a fork of the whole compiled run at the session's first round,
not `Claims.Recursion.EO`'s, so L2 here bounds a different extractor than `RecursiveKSEvery` does, by `Comp.err` at
`Compiled.toComp` (`KSCompiledOn`). What L2 reads at the instance beside L1's premises: the compiler's contract for
`E_O` (`Compiled.Sessions`, `SessionsOfRec`), Flock's KS as `Arg.ofFlock`'s (`Claims.SamplingGeneric.flockArgKS`)
and the compiler's contract at `compile` (`ContractCompiledOn`; at the deployment's dense statements
`ContractCompiledDense`). The bits read `Compiled.EO`'s expected reruns (`CompiledEORunsExp`), one more per session
than `Claims.Recursion.EO`'s, since each session's `E^rep` replays the judged run.

Statements only: definitions, no proofs.
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion
open Claims.SamplingGeneric (Arg)

section Instance

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
  {Col' T' : Type} [DecidableEq Col']

/-- **Today's compiled record as a compiled verifier over Flock `--zk`**, at the inner verifier `IV`: the record's
verifier, parser and inner statement; the inner rounds and the staging rounds before the sessions; session `s` at the
public shape of its numbers (`E^rep`'s waits, its truncation, the outer verifier's rounds, `N₀`), its errors within
their `κ`, `γ`; the sessions' statements the record's V\* statements at the rounds before the sessions; session `s`'s
view the record's; and the inner messages, the searched calls and the compiled verifier's own as `Comp.ofRec`'s, at
the witnesses' second components. -/
noncomputable def Compiled.ofRec (D : RecAt Coin) (DO : Claims.Flock.Deploy Coin Col' T') (IV : Verifier Coin) :
    Compiled Coin (Arg.ofFlock DO) where
  V := (D.record IV).V
  parses x := ∃ s, (D.record IV).parse x = some s ∧ s.circ = none
  xI := Claims.Recursion.xI (D.record IV)
  pre x := IV.k (Claims.Recursion.xI (D.record IV) x) + (D.record IV).kS x
  nS := (D.record IV).nS
  sh x s := (((D.record IV).nums x s).waits, ((D.record IV).nums x s).cap, ((D.record IV).nums x s).rounds,
    ((D.record IV).nums x s).N₀)
  κb x s := ((D.record IV).nums x s).κ
  γb x s := ((D.record IV).nums x s).γ
  stmts x ms :=
    (D.record IV).vstar x (ms.take (IV.k (Claims.Recursion.xI (D.record IV) x) + (D.record IV).kS x))
  session := (D.record IV).session
  msgs x ws := D.msgs x (ws.map Prod.snd)
  calls H x r ws := (D.record IV).callsAt H x r (ws.map Prod.snd)
  own H x ms := setupCalls H (D.record IV)
    ((D.record IV).vstar x (ms.take (IV.k (Claims.Recursion.xI (D.record IV) x) + (D.record IV).kS x)))

/-- **The compiler** at a compiled record and the outer record: `Compiled.ofRec` at each inner verifier. -/
noncomputable def recCompiler (D : RecAt Coin) (DO : Claims.Flock.Deploy Coin Col' T') :
    Compiler Coin (Arg.ofFlock DO) :=
  Compiled.ofRec D DO

/-- **The instance's compiled verifier**: the compiler at `Open(DE, hm96)`, the element-level Flock at the deployed
inner decoder, the caps `Cp`. -/
noncomputable abbrev compiledAt (D : RecAt Coin) (Bd : Bound Coin Col T) (DO : Claims.Flock.Deploy Coin Col' T')
    (Cp : Caps Coin) : Compiled Coin (Arg.ofFlock DO) :=
  compile (recCompiler D DO) (Proto.ofFlock Bd Bd.deploy.decode Cp) (Server.ofBound Bd)

/-- **The compiler meets its contract for `E_O`** at the inner verifier `IV`: `Compiled.Sessions` at
`Compiled.ofRec`. -/
def SessionsOfRec (H : Msg → Msg) (D : RecAt Coin) (DO : Claims.Flock.Deploy Coin Col' T') (IV : Verifier Coin) :
    Prop :=
  (Compiled.ofRec D DO IV).Sessions H

/-- **The compiler's contract at the statements `G` admits**: `Comp.Contract` at `compiledAt`'s composition input, at
`Open(DE, hm96)` and `DE`'s reading of opened views. -/
def ContractCompiledOn (H : Msg → Msg) (D : RecAt Coin) (Bd : Bound Coin Col T)
    (DO : Claims.Flock.Deploy Coin Col' T') (Cp : Caps Coin) (G : Msg → Prop) : Prop :=
  ∀ x, G x → (compiledAt D Bd DO Cp).toComp.Contract H ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)
    (fun v => (Proto.ofFlock Bd Bd.deploy.decode Cp).opens ((Server.ofBound Bd).openView v)) x

/-- **L2's bound at the composite over Flock `--zk`**: at every statement `G` admits, the composite at `compiledAt`,
its extractor `Comp.E` over `Compiled.EO`, meets `Comp.err` at every prover, loop bounds and link bound. -/
def KSCompiledOn (H : Msg → Msg) (D : RecAt Coin) (Bd : Bound Coin Col T) (DO : Claims.Flock.Deploy Coin Col' T')
    (Cp : Caps Coin) (G : Msg → Prop) : Prop :=
  ∀ (x : Msg), G x → ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (m : Fin (Proto.ofFlock Bd Bd.deploy.decode Cp).nT → ℕ) (M : ℕ),
    PrBad H (compiledAt D Bd DO Cp).V ((compiledAt D Bd DO Cp).toComp.R (Proto.ofFlock Bd Bd.deploy.decode Cp))
        ((compiledAt D Bd DO Cp).toComp.E (Proto.ofFlock Bd Bd.deploy.decode Cp) (Server.ofBound Bd)) x P ≤
      (compiledAt D Bd DO Cp).toComp.err (Proto.ofFlock Bd Bd.deploy.decode Cp) (Server.ofBound Bd) H P x m M

/-- **`Compiled.EO` reruns the prover at most `1 + Σ_s (1 + m′_s·k_s)` times in expectation** (O2's expected-reruns
half, at the instance): the judged run, then at each session `E^rep`'s replay of the judged run and its waits over
forks at the session's first round, as `Claims.Recursion.EO_runs_exp` counts `Claims.Recursion.EO`'s, which has no
replay per session. Where each session's V\* statement has the session's numbers (`SessionNums`, so its shape's `N₀`
fixes its `k`) and an accepted run shows each session accepting (`SessionAccepts`). -/
def CompiledEORunsExp (H : Msg → Msg) (D : RecAt Coin) (DO : Claims.Flock.Deploy Coin Col' T') (IV : Verifier Coin) :
    Prop :=
  SessionNums (D.record IV) DO → SessionAccepts (D.record IV) →
    ExtractRunsExpLe H (Compiled.ofRec D DO IV).EO fun x =>
      1 + ∑ s : Fin ((D.record IV).nS x),
        ((1 + ((D.record IV).nums x s).waits * ((D.record IV).nums x s).k : ℕ) : ℝ)

end Instance

end Claims.RecGeneric

namespace Flock.Guarantees

open Claims (Msg Out)
open FlockDeploy (Coin FinalP)

/-- **L2 at a compiled record** `D`: `Claims.RecGeneric.KSCompiledOn` at every statement, at SHA-512, the deployed
relation's bound record at the reader `rd`, the deployed outer record and caps. -/
def RecursiveKSCompiledAtRec (D : Claims.Recursion.RecAt Coin) (rd : Msg → Option FinalP)
    (final : Msg → Option (List (List (Msg × RecDeploy.Salt)) × Msg)) (wE w : Msg → ℝ) : Prop :=
  Claims.RecGeneric.KSCompiledOn FlockDeploy.sha512 D
    (RecDeploy.bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR rd)
      (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE FlockDeploy.joinR
      RecDeploy.rowInDeployed final w)
    (RecDeploy.deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep fun _ => True

/-- **L2 at the deployment**: `RecursiveKSCompiledAtRec` at the deployment's record and values, for every other field
of the brings and every record the challenger's live-coin part keeps (as `RecursiveKSEvery` quantifies them). -/
def RecursiveKSCompiledEvery : Prop :=
  ∀ (b : RecDeploy.Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8),
    RecursiveKSCompiledAtRec
      (RecDeploy.recBoundAt _root_.Flock.Readable.relation
        { RecDeploy.deployedB b with tr := RecDeploy.recTr aux nOf hd recRest }
        (RecDeploy.DeployedR.layR RecDeploy.OpenRows.elemR))
      RecDeploy.DeployedR.rdR RecDeploy.DeployedR.finalR RecDeploy.DeployedR.wER RecDeploy.DeployedR.wR

end Flock.Guarantees

namespace Claims.RecGeneric

open FlockDeploy (Coin FinalP)

/-- **The compiler's contract at the dense statements, at a compiled record** `D`: `ContractCompiledOn` at SHA-512,
the deployed relation's bound record at the reader `rd`, the deployed outer record and caps, and `RecDeploy.DenseAt`,
as `ContractDenseAtRec` is `ContractOn` there. -/
def ContractCompiledDenseAtRec (D : Claims.Recursion.RecAt Coin) (rd : Msg → Option FinalP)
    (final : Msg → Option (List (List (Msg × RecDeploy.Salt)) × Msg)) (wE w : Msg → ℝ) : Prop :=
  ContractCompiledOn FlockDeploy.sha512 D
    (RecDeploy.bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR rd)
      (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE FlockDeploy.joinR
      RecDeploy.rowInDeployed final w)
    (RecDeploy.deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep RecDeploy.DenseAt

/-- **The compiler's contract at the deployment**: `ContractCompiledDenseAtRec` at the deployment's record and values,
for every other field of the brings and every record the challenger's live-coin part keeps. -/
def ContractCompiledDense : Prop :=
  ∀ (b : RecDeploy.Brings) (aux : Msg → ℕ → _root_.Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → _root_.Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8),
    ContractCompiledDenseAtRec
      (RecDeploy.recBoundAt _root_.Flock.Readable.relation
        { RecDeploy.deployedB b with tr := RecDeploy.recTr aux nOf hd recRest }
        (RecDeploy.DeployedR.layR RecDeploy.OpenRows.elemR))
      RecDeploy.DeployedR.rdR RecDeploy.DeployedR.finalR RecDeploy.DeployedR.wER RecDeploy.DeployedR.wR

end Claims.RecGeneric
