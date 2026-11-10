module

public import Definitions.Claims.Sampling
public import Definitions.Claims.Recursion
public import Definitions.Claims.ZKOut
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init

/-!
# Claims: the sampled audit as a compiler, generic in the proof system for proof blocks

`Claims.Sampling` with its Flock-specific parts as parameters: the proof system the sessions of the sampled proof blocks
run (`Arg`, for Flock `Sampling.Deploy.flock`) and its extractor (for Flock `E^rep`, `Arg.ofFlock`). Everything
else is `Claims.Sampling`'s, read as it stands: the statement (`AuditStatement`), the witness (`Sampling.Wit`), the
relation at `δ` (`Sampling.R`: the incorrect set of the extracted trace is in `M_δ = {B : P_a(B) > δ}`) and its count
form (`Sampling.RCount`), the readable scope (`InScope`), the commitment scheme and its binding (`CommitmentScheme`,
`CommitmentScheme.Binds`), the waits, the extraction-based finder `Bext` and the claims' forms (`KSLinOn`, `BitsLin`,
`ExtractOwnLe`, `leafLayer`).

1. **The proof system for proof blocks** (`Arg`): a public-coin verifier over byte strings, its statements' public shape
   (`Sh`, `shape`: what its rounds and its extractor's coins read) and the rounds a shape fixes, its witnesses and
   relation, its extractor, and its errors per statement (`κ`, `γ`). The extractor is `E^rep` (`Claims.ERep`) at the
   waits, truncation and accepted reruns the shape fixes, its openings read off each accepted rerun's view, and its
   assemble step, a counted program (its candidate and its steps, as `Flock.Deploy.dec`): its coins are per shape
   (`EC`), and its witness is what that step makes of the reruns' views. It is knowledge sound in `Claims.KS`'s form
   (`Arg.KS`). Being `E^rep`, at every proof system it reruns at most `1 + m′·K` times, `1 + m′·k` in expectation, and
   makes no SHA-512 call of its own, so the claims read no premise on its cost. Its zero knowledge is the claims core's
   per public outcome (`ZkArg.ZK`, `Claims.ZKOut`), on its sessions' rounds (`kz`) and one tape type: at each outcome
   (each verdict, where a session stopped), the honest view's probability of the outcome and an event is at most the
   simulator's, given the outcome, plus an error that reads the statement, the witness, the verifier and the outcome.
2. **The compiled sampled verifier** (`Compiled`, `Sampling.Deploy` without Flock): its verifier, parser, scope check,
   partition and commitment; the second stage (`stage2`: the sample and the sessions); the sampled proof blocks
   (`checked`); each one's public shape (`sh`) and the bounds on its session's errors (`κb`, `γb`), its session's
   rounds (`rounds`, whose order embedding `emb` of `A`'s rounds is defined, from which `coinsOf` reads a session's
   coins and `place` replaces them), and its session (`session`: its statement and view); each slot's session key
   (`skey`, slots with one key sharing one session) and a cap on a sample's keys (`nSess`); the roots a run shows for
   a proof block's interior values (`iroots`); what a session's witness reveals (`reveal`), each opening checked at
   most `openCalls` SHA-512 calls; and the bits of the unions the outer and inner waits pay for (`nOut`, `nIn`), which
   set their counts (`kOut`, `kIn`).
3. **The extractor** (`Compiled.E`): replay; `k_out` accepted reruns among `K_out`; at each, `k_in` accepted reruns of
   the second stage only among `K_in`; at each, each sampled session's extraction, `Arg`'s extractor at it
   (`Compiled.sessWit`), once per session at its first slot, each rerun the judged run's coins with the session's
   rounds replaced (`place`), seen as the session's view, then what its witness reveals at each of the session's proof
   blocks that passes its opening check (`Compiled.sessExt`). Its coins are per slot of the sample (`SlotEC`), of which
   a session reads its first slot's (`runExt`, a run's extractions). Every SHA-512 call of the extraction is an
   `RP.hash` (the openings it checks) or a rerun's, so `Bext`'s cost counts them all. The statements hold at every `A`
   and `H`, so that the extracted trace is the prover's rests on `A`'s assemble step being the program its count
   describes (at Flock, the deployed decoder).
4. **The errors**: `κ_δ = c²·(δ + κ_s) + (2k_out/K_out + e^{−k_out/8}) + c·(2k_in/K_in + e^{−k_in/8})`, with `κ_s`
   the largest of the proof blocks' bounds on `Arg`'s `κ` (`κs`, at `κb`; at a frozen record, Flock's `κ_F`,
   `Sampling.κF`) and `c` the waits' slack; `γ = c⁴·k(x)·γ_s`, with `γ_s` the largest of their bounds on `Arg`'s `γ`
   per round (`γb` over the shape's rounds; a session's fork finder is the compiled
   verifier's `B` at a round guess among `k(x)`); `ε_bind = Pr[B_ext]`.
5. **The contract** (`Compiled.Sessions`): the code facts the compiler theorems read, the generic forms of the frozen
   hypotheses `Claims.Sampling` states at Flock (`DrawLaw`, `Refuses`, `InteriorFixed`, `SessionOpens`): the second
   stage samples the sample by the design (`draws`), in at most `n_PU` slots (`slots`); slots with one key show one
   session (`shared`), at most `n_sess` keys a sample (`keys`); the outer wait's union, over sets of proof blocks, is
   within its bits (`unions`); the verifier refuses what is out of scope (`refuses`); the interior roots precede the
   second stage (`interior`); the sample precedes its sessions (`fixed`); each sampled proof block's session, at the
   block's
   shape and within its error bounds, accepts whenever the run does and is a run of a prover against `Arg`'s verifier
   on the session's coins (`runs`); and a witness in `Arg`'s relation reveals openings under which its proof block is
   correct (`opens`). A run with two sampled sessions meets it (`Witness.SamplingGeneric.twoSessions_meets`).
6. **The claims** of a compiled verifier (`Compiled.KS`, `Compiled.KSCount`, `Compiled.Bits`, `Compiled.Efficient`);
   its completeness (`Compiled.Complete`, of its reference prover at its error), with its contract over the proof
   system's honest prover (`Compiled.CompleteSessions`: each sampled session is that prover's run, the run accepts when
   its sessions do, they are the first `K` slots', and the error covers `K` sessions' errors);
   its committed form (`Compiled.KSReg`): the incorrect set of the committed trace (`Freg` at `regVal`, the plurality of
   what the prover's runs' extractions open at its tape, a definition nobody runs, fixed before the window's coins), not
   the extracted one, so its sampling term is the acceptance probability `P_a(Freg)` a monitoring period's pooling
   reads, with the binding term `B_conf` (two runs' extractions opening a gate two ways); and the claims of a masked
   one (`ZkCompiled`: the reference prover's framing, which routes each history to a message it fixes or to a slot's
   session, `A`'s prover in each session on the slot's tape): `ZkCompiled.ZK`, zero knowledge per public outcome
   relative to the published leaf layer, the outcome each session's (`outOf`), at the composed simulator (`sim`: the
   framing, then each session's simulator given its outcome) and the error `err`, the sum over the sessions of the
   expectation over the honest tape of `A`'s error at the session the honest run reaches. Its contract
   (`ZkCompiled.ZkSessions`, `Claims.Sampling`'s `Framing` facts in their generic form): the rounds are the parts', the
   router puts each round in its part, each session's rounds and simulator coins are its slot's, an honest trace's
   session witnesses are in `A`'s relation, and the reference prover sends the composed prover's messages.
7. **The compiler's statements** (`CompilerKS`, `CompilerKSReg`, `CompilerKSCount`, `CompilerBits`, `CompilerKSAt`,
   `CompilerEfficient`, `CompilerZK`, `CompilerComplete`): each claim of every compiled verifier that meets the
   contract, from `Arg`'s.
8. **Flock** (`Arg.ofFlock`): Flock's `E^rep` at the record's deployed decoder (`D.decode`), its assemble step counted
   (`flockAssemble`), its shape the numbers it reads (its coins' and `N₀`), its errors Flock's; `FlockArgKS` says that
   `Flock.KS` is its `Arg.KS`, so `FlockKS` and `FlockKSReg` are `CompilerKS` and `CompilerKSReg` at Flock. They take
   `Flock.KS`, the binding and the contract as hypotheses, at a record, not at a deployed verifier's code.

Statements only: definitions, no proofs. A frozen `Claims.Sampling.Deploy` record is a compiled verifier over Flock, its
extractor `Compiled.E` at `Arg.ofFlock`; `Claims.Sampling` states the record's frozen hypotheses and no extractor of its
own.
-/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims
open Claims.Sampling (AuditStatement Partition CommitmentScheme Opening InScope waitAccC forkOn flatMapRP
  checkRevealed assemble ownLog Bext KSLinOn BitsLin ExtractOwnLe lamLin slack slackExp leafLayer Rhon)
open Claims.Recursion (serve viewing goesOn simRounds spBind spUnder)

/-- The largest of `f 0, …, f (n − 1)`, and `0` when there is none. -/
noncomputable def maxBelow (n : ℕ) (f : ℕ → ℝ) : ℝ := ((List.range n).map f).foldr max 0

/-- The largest of `f 0, …, f (n − 1)`, and `0` when there is none. -/
def maxBelowN (n : ℕ) (f : ℕ → ℕ) : ℕ := ((List.range n).map f).foldr max 0

/-! ## 1. The proof system for proof blocks -/

/-- **A proof system for proof blocks**, as the sampled audit reads it: its statements' public shapes (`Sh`, `shape`:
what its rounds and its extractor's coins read), its verifier and the rounds a shape fixes (`wf`), its witnesses and
relation, its extractor's parts, and its errors per statement (`κ`, `γ`). The extractor is `E^rep` (`Claims.ERep`):
replay the judged run and, if it accepted, make `waits` waits, each over `cap` fresh coin vectors and stopping at `k`
accepted reruns (each read off the statement's shape), and return the first candidate `assemble` makes of a wait's
openings (`opens`, read off each accepted rerun's view) that `check` accepts. `assemble` is a counted program: with its
candidate it returns the steps it took, as `Flock.Deploy.dec` does. So the extracted witness is what that step makes of
the reruns' views, and the extraction calls `H` only through its reruns. -/
structure Arg (Coin : Type) where
  Sh : Type
  shape : Msg → Sh
  V : Verifier Coin
  rounds : Sh → ℕ
  wf : ∀ x, V.k x = rounds (shape x)
  W : Type
  R : Rel W
  Op : Type
  opens : Seen → List Op
  assemble : Msg → List Op → Option W × ℕ
  check : Msg → W → Bool
  waits : Sh → ℕ
  k : Sh → ℕ
  cap : Sh → ℕ
  κ : Msg → ℝ
  γ : Msg → ℝ

namespace Arg

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- The extractor's coins at a shape: per wait, its `cap` coin vectors over the shape's rounds. -/
def EC (A : Arg Coin) (s : A.Sh) : Type := Fin (A.waits s) → Fin (A.cap s) → Fin (A.rounds s) → Coin

instance instFintypeEC (A : Arg Coin) (s : A.Sh) : Fintype (A.EC s) :=
  inferInstanceAs (Fintype (Fin _ → Fin _ → Fin _ → Coin))

instance instNonemptyEC (A : Arg Coin) (s : A.Sh) : Nonempty (A.EC s) :=
  inferInstanceAs (Nonempty (Fin _ → Fin _ → Fin _ → Coin))

/-- **The extractor's rerun program** at the judged run's coins `r₀`: `E^rep` at `A`'s parts, its coin vectors the
shape's (`EC`) read over the statement's rounds (`wf`). -/
def ext (A : Arg Coin) (x : Msg) (r₀ : Fin (A.V.k x) → Coin) (c : A.EC (A.shape x)) :
    RP (Fin (A.V.k x) → Coin) (Option A.W) :=
  (ERep A.V (fun x => A.waits (A.shape x)) (fun x => A.k (A.shape x)) (fun x => A.cap (A.shape x)) A.opens
    (fun x os => (A.assemble x os).1) A.check).ext x r₀ fun i j t => c i j (Fin.cast (A.wf x) t)

/-- The proof system's extractor, as a claims-core extractor. -/
def E (A : Arg Coin) : Extractor Coin A.W A.V where
  C x := A.EC (A.shape x)
  ext := A.ext

/-- **The proof system is knowledge sound** at its errors: `Claims.KS` of its extractor. -/
def KS (H : Msg → Msg) (A : Arg Coin) : Prop := Claims.KS H A.V A.R A.E A.κ A.γ

end Arg

/-! ### Zero knowledge with a public outcome (`Claims.ZKOut`) -/

/-- **A proof system for proof blocks with zero knowledge**: the rounds a session runs against a verifier (`kz`; at
Flock `V.k + 1`, its `Hello` first), its honest prover at each witness on a tape of a nonce and the rest, of one type at
every statement (as Flock's masked record's, so sessions under statements a run picks read tapes of one type), its
public outcome (`St`, read off a session's transcript by `out`), its simulator given the outcome, and its error. -/
structure ZkArg (Coin : Type) extends Arg Coin where
  kz : Msg → ℕ
  Nonce : Type
  Rest : Type
  [nonceFin : Fintype Nonce]
  [nonceNe : Nonempty Nonce]
  [restFin : Fintype Rest]
  [restNe : Nonempty Rest]
  P : W → Prover (Nonce × Rest)
  St : Type
  out : Msg → List (Msg × Out) → St
  S : SimOut St
  εzk : (x : Msg) → W → (Θ : Type) → VStar Θ → Θ → St → ℝ

attribute [instance] ZkArg.nonceFin ZkArg.nonceNe ZkArg.restFin ZkArg.restNe

/-- **The proof system is zero knowledge with a public outcome**, against every verifier of its sessions' rounds, at its
error. -/
def ZkArg.ZK (H : Msg → Msg) {Coin : Type} (A : ZkArg Coin) : Prop :=
  ZKOut (fun _ => A.Nonce) (fun _ => A.Rest) H A.kz (fun w _ => A.P w) A.R A.out A.S A.εzk

/-! ## 2. The compiled sampled verifier -/

/-- **A compiled sampled verifier over the proof system `A`** (`Sampling.Deploy` with its Flock parts replaced by `A`):
the verifier, its parser, scope check and partition, the trace's commitment; the second stage (`stage2`: the sample
and the sessions, after the registration); the slots of the proof blocks a run samples (`checked`); each one's public
shape (`sh x u`, fixed by the statement) and the bounds on its session's errors (`κb x u`, `γb x u`: on `A`'s `κ` and
`γ` at the session's statement); its session's rounds (`rounds`) and the session as a run shows it (`session`: its
statement in `A` and its view); each slot's session key in the sample (`skey x l p`: slots with one key share one
session, which the extractor extracts once; each slot its own session by default) and a public cap on the keys a sample
holds (`nSess`, `n_PU` by default); the roots a run shows for a proof block's interior values (`iroots`); what a
session's witness reveals (`reveal`: committed values, each with the roots it is committed under); the SHA-512 calls of
checking what one session's witness reveals (`openCalls`); the bits of the unions the outer and inner waits pay their
bounds over (`nOut`, `nIn`: at most `2^{n_out}` and `2^{n_in}` sets, as the contract bounds them); and the reference
prover on its tapes with its completeness error. -/
structure Compiled (Coin : Type) (A : Arg Coin) where
  V : Verifier Coin
  parse : Msg → Option AuditStatement
  scopeCheck : Msg → Bool
  partition : AuditStatement → Option Partition
  commit : CommitmentScheme Bool
  stage2 : (x : Msg) → Finset (Fin (V.k x))
  checked : (x : Msg) → (Fin (V.k x) → Coin) → List ℕ
  sh : Msg → ℕ → A.Sh
  κb : Msg → ℕ → ℝ
  γb : Msg → ℕ → ℝ
  rounds : (x : Msg) → ℕ → Finset (Fin (V.k x))
  session : Msg → ℕ → Seen → Option (Msg × Seen)
  skey : Msg → List ℕ → ℕ → ℕ := fun _ _ p => p
  nSess : Msg → ℕ := fun x =>
    match parse x with
    | some s => ((partition s).getD ⟨[]⟩).blocks.length
    | none => 0
  iroots : Msg → Seen → ℕ → List Msg
  reveal : AuditStatement → ℕ → Seen → A.W → List (List Msg × Opening Bool)
  openCalls : Msg → ℕ
  nOut : Msg → ℕ
  nIn : Msg → ℕ
  Tape : Type
  [tapeNonempty : Nonempty Tape]
  P : Sampling.Wit → Prover Tape
  δc : (Msg → Msg) → Msg → ℝ

namespace Compiled

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

instance instNonemptyTape (L : Compiled Coin A) : Nonempty L.Tape := L.tapeNonempty

/-- The statement's partition, as the verifier evaluates its query; empty when the query doesn't apply. -/
def part (L : Compiled Coin A) (s : AuditStatement) : Partition := (L.partition s).getD ⟨[]⟩

/-- The proof blocks at `x`, `0` on a statement that doesn't parse. -/
def nPU (L : Compiled Coin A) (x : Msg) : ℕ :=
  match L.parse x with
  | some s => (L.part s).blocks.length
  | none => 0

/-- The outer wait stops at `k_out = 2^j·(2·n_out + 1)` accepted runs (`slackExp`, as `Sampling.Deploy.k`): `2^j` for
each bit of its union, so the union costs at most the slack. -/
def kOut (L : Compiled Coin A) (x : Msg) : ℕ := 2 ^ slackExp * (2 * L.nOut x + 1)

/-- The inner wait stops at `k_in = 2^j·(2·n_in + 1)` accepted runs. -/
def kIn (L : Compiled Coin A) (x : Msg) : ℕ := 2 ^ slackExp * (2 * L.nIn x + 1)

/-- The outer wait gives up after `K_out = 2^128·k_out` reruns. -/
def capOut (L : Compiled Coin A) (x : Msg) : ℕ := 2 ^ 128 * L.kOut x

/-- The inner wait gives up after `K_in = 2^128·k_in` reruns. -/
def capIn (L : Compiled Coin A) (x : Msg) : ℕ := 2 ^ 128 * L.kIn x

/-- Session `u`'s view of a run. -/
def view (L : Compiled Coin A) (x : Msg) (u : ℕ) (v : Seen) : Seen := ((L.session x u v).map Prod.snd).getD ⟨[], .rej⟩

/-- Session `u`'s rounds in order, as `A`'s rounds at the session's statement `xS`, where `A` has as many there. -/
def emb (L : Compiled Coin A) (x : Msg) (u : ℕ) (xS : Msg) (h : A.V.k xS = (L.rounds x u).card) :
    Fin (A.V.k xS) ↪o Fin (L.V.k x) :=
  (L.rounds x u).orderEmbOfFin h.symm

/-- A run's coins on session `u`'s rounds, as `A`'s coins at the session's statement `xS`; the default coin where `A`'s
rounds there are not as many. -/
def coinsOf (L : Compiled Coin A) (x : Msg) (u : ℕ) (xS : Msg) (r : Fin (L.V.k x) → Coin) : Fin (A.V.k xS) → Coin :=
  fun i => if h : A.V.k xS = (L.rounds x u).card then r (L.emb x u xS h i) else default

/-- The run's coins `r` with session `u`'s rounds replaced by `A`'s coins `d` at the session's statement `xS`; `r` where
`A`'s rounds there are not as many. -/
def place (L : Compiled Coin A) (x : Msg) (u : ℕ) (xS : Msg) (r : Fin (L.V.k x) → Coin) (d : Fin (A.V.k xS) → Coin) :
    Fin (L.V.k x) → Coin :=
  fun t => if h : A.V.k xS = (L.rounds x u).card ∧ t ∈ L.rounds x u then
      d (((L.rounds x u).orderIsoOfFin h.1.symm).symm ⟨t, h.2⟩)
    else r t

/-! ## 3. The extractor -/

/-- **The slots of slot `i`'s session** in the sample `l`, in order: those whose proof block is below `n_PU` and whose
key is `i`'s (`skey`). The first is the session's first slot, where the extractor extracts it. -/
def sessSlots (L : Compiled Coin A) (x : Msg) (l : List ℕ) (i : ℕ) : List ℕ :=
  (List.range l.length).filter fun j => (l[j]?.any fun u => decide (u < L.nPU x)) && L.skey x l j == L.skey x l i

/-- The proof blocks at the slots of slot `i`'s session (`sessSlots`). -/
def sessBlocks (L : Compiled Coin A) (x : Msg) (l : List ℕ) (i : ℕ) : List ℕ :=
  (L.sessSlots x l i).filterMap fun j => l[j]?

open Classical in
/-- **A session's witness at an accepted run** `(r, v)`, extracted at its proof block `u`: `A`'s extractor at `u`'s
session, its judged coins the session's coins of `r`, each rerun `r` with the session's rounds replaced (`place`), seen
as the session's view. At a session whose shape is not the block's, which no sampled proof block's session has
(`Sessions.runs`), none. -/
noncomputable def sessWit (L : Compiled Coin A) (x : Msg) (u : ℕ) (r : Fin (L.V.k x) → Coin) (v : Seen)
    (c : A.EC (L.sh x u)) : RP (Fin (L.V.k x) → Coin) (Option A.W) :=
  match L.session x u v with
  | none => .ret none
  | some (xS, _) =>
    if h : A.shape xS = L.sh x u then
      serve (fun r' => viewing (L.view x u) (L.place x u xS r r'))
        (A.ext xS (L.coinsOf x u xS r) (cast (congrArg A.EC h.symm) c))
    else .ret none

/-- **A session's extraction**: its witness, extracted once at its first slot's proof block `u` (`sessWit`); then what
the witness reveals at each of the session's proof blocks `us` that passes its opening check. -/
noncomputable def sessExt (L : Compiled Coin A) (x : Msg) (s : AuditStatement) (us : List ℕ) (u : ℕ)
    (r : Fin (L.V.k x) → Coin) (v : Seen) (c : A.EC (L.sh x u)) :
    RP (Fin (L.V.k x) → Coin) (List (List Msg × Opening Bool)) :=
  (L.sessWit x u r v c).bind fun
    | none => .ret []
    | some w => flatMapRP (fun u' => checkRevealed L.commit (L.reveal s u' v w)) us

/-- The extractor's coins at one slot of the sample: `A`'s extractor's at each proof block's shape, of which the slot
reads its block's. -/
abbrev BlockEC (L : Compiled Coin A) (x : Msg) := (u : Fin (L.nPU x)) → A.EC (L.sh x u)

/-- The extractor's coins at one accepted inner run: one `BlockEC` per slot of the sample, of which a session reads its
first slot's. -/
abbrev SlotEC (L : Compiled Coin A) (x : Msg) := Fin (L.nPU x) → L.BlockEC x

/-- The extractor's coins: the outer wait's `K_out` coin vectors; for each of its `k_out` accepted runs, the inner
wait's `K_in` (used fresh on the second stage); and for each accepted inner run and each slot of the sample, `A`'s
extractor's coins at each proof block's shape. -/
abbrev ECoins (L : Compiled Coin A) (x : Msg) :=
  (Fin (L.capOut x) → Fin (L.V.k x) → Coin) × (Fin (L.kOut x) → Fin (L.capIn x) → Fin (L.V.k x) → Coin) ×
    (Fin (L.kOut x) × Fin (L.kIn x) × Fin (L.nPU x) → L.BlockEC x)

/-- **A run's extractions**, at the accepted run `(r, v)` and the extractor's coins `c` there: slot by slot of the sample
(the first `n_PU`, every slot by `Sessions.slots`), at each session's first slot (`sessSlots`), the session's
extraction at the slot's coins (`sessExt`), with what its witness reveals at each of the session's proof blocks that
passes its opening check. So `A`'s extractor runs once per session, at most `n_sess` times (`Sessions.keys`). -/
noncomputable def runExt (L : Compiled Coin A) (x : Msg) (s : AuditStatement) (r : Fin (L.V.k x) → Coin) (v : Seen)
    (c : L.SlotEC x) : RP (Fin (L.V.k x) → Coin) (List (List Msg × Opening Bool)) :=
  flatMapRP (fun iu : Fin (L.nPU x) × ℕ =>
      if hu : iu.2 < L.nPU x ∧ (L.sessSlots x (L.checked x r) iu.1).head? = some iu.1.val then
        L.sessExt x s (L.sessBlocks x (L.checked x r) iu.1) iu.2 r v (c iu.1 ⟨iu.2, hu.1⟩)
      else .ret [])
    ((List.finRange (L.nPU x)).zip (L.checked x r))

/-- **The sampling extractor**: replay the judged run; if it accepted, rerun on fresh coins
until `k_out` accepted runs are in hand (giving up after `K_out`); at each, rerun fresh on the second stage until `k_in`
accepted runs are in hand (giving up after `K_in`); at each of those, extract each session of the sample once
(`runExt`); and assemble. -/
noncomputable def E (L : Compiled Coin A) : Extractor Coin Sampling.Wit L.V where
  C x := ECoins L x
  ext x r₀ c := .rerun r₀ fun v₀ =>
    match L.parse x with
    | none => .ret none
    | some s =>
      if v₀.out = .acc then
        (waitAccC (L.kOut x) ((List.finRange (L.capOut x)).map c.1) []).bind fun outer =>
          (flatMapRP (fun jo : Fin (L.kOut x) × ((Fin (L.V.k x) → Coin) × Seen) =>
              (waitAccC (L.kIn x) ((List.finRange (L.capIn x)).map fun l => forkOn (L.stage2 x) jo.2.1 (c.2.1 jo.1 l))
                  []).bind fun inner =>
                flatMapRP (fun li : Fin (L.kIn x) × ((Fin (L.V.k x) → Coin) × Seen) =>
                    L.runExt x s li.2.1 li.2.2 fun p => c.2.2 (jo.1, li.1, p))
                  ((List.finRange (L.kIn x)).zip inner))
            ((List.finRange (L.kOut x)).zip outer)).bind fun found => .ret (some (assemble s found))
      else .ret none

/-! ## 4. The errors -/

/-- **The sessions' knowledge error**: the largest of the proof blocks' bounds on `A`'s `κ` (`κb`; at a frozen record,
Flock's `κ_F`). -/
noncomputable def κs (L : Compiled Coin A) (x : Msg) : ℝ := maxBelow (L.nPU x) (L.κb x)

/-- **The sessions' `γ` per round**: the largest of the proof blocks' bounds on `A`'s `γ` (`γb`) over the rounds of
their shapes. -/
noncomputable def γs (L : Compiled Coin A) (x : Msg) : ℝ :=
  maxBelow (L.nPU x) fun u => L.γb x u / A.rounds (L.sh x u)

/-- **The knowledge error at `δ`**: `δ + ε_ks`, with
`ε_ks = (c² − 1)·δ + c²·κ_s + (2k_out/K_out + e^{−k_out/8}) + c·(2k_in/K_in + e^{−k_in/8})`: the outer wait's
truncation, plus its factor `c` times the inner wait's bound, its truncation plus `c·(P_a(B) + κ_s)`. -/
noncomputable def κ (L : Compiled Coin A) (δ : ℝ) (x : Msg) : ℝ :=
  slack ^ 2 * (δ + L.κs x) + (2 * L.kOut x / L.capOut x + Real.exp (-(L.kOut x : ℝ) / 8)) +
    slack * (2 * L.kIn x / L.capIn x + Real.exp (-(L.kIn x : ℝ) / 8))

/-- **`γ`**: `c⁴·k(x)·γ_s`, with `k(x)` the compiled verifier's rounds: a session's fork at one of its `k_A` rounds is
the compiled verifier's `B` at one of `k(x)`. -/
noncomputable def γ (L : Compiled Coin A) (x : Msg) : ℝ := slack ^ 4 * (L.V.k x * L.γs x)

/-- The design's miss at `t` at `x`, clamped at `0`; `0` on a statement that doesn't parse (`Sampling.missAt`). -/
noncomputable def missAt (L : Compiled Coin A) (t : ℕ) (x : Msg) : ℝ :=
  match L.parse x with
  | some s => max 0 (s.design.miss (L.part s) t)
  | none => 0

/-- The sessions' expected reruns: the largest of `E^rep`'s `1 + m′·k` at the proof blocks' shapes. -/
noncomputable def es (L : Compiled Coin A) (x : Msg) : ℝ :=
  maxBelow (L.nPU x) fun u => 1 + A.waits (L.sh x u) * A.k (L.sh x u)

/-- The sessions' strict reruns: the largest of `E^rep`'s `1 + m′·K` at the proof blocks' shapes. -/
def ρs (L : Compiled Coin A) (x : Msg) : ℕ := maxBelowN (L.nPU x) fun u => 1 + A.waits (L.sh x u) * A.cap (L.sh x u)

/-- The extractor's expected work in runs, `1 + k_out + k_out·k_in·(1 + n_sess·e_s) + k_out·k_in·n_PU·openCalls`: its
expected reruns, one `E^rep` per session, and its own SHA-512 calls, the opening checks'. -/
noncomputable def work (L : Compiled Coin A) (x : Msg) : ℝ :=
  1 + L.kOut x + (L.kOut x : ℝ) * L.kIn x * (1 + L.nSess x * L.es x) +
    (L.kOut x : ℝ) * L.kIn x * L.nPU x * L.openCalls x

/-- **The sampling scope, on the statement's bytes**: it parses, its query evaluates, and the partition and statement are
in scope (`Sampling.Scope`). -/
def Scope (L : Compiled Coin A) (x : Msg) : Prop :=
  ∃ s P, L.parse x = some s ∧ L.partition s = some P ∧ InScope P s

/-! ## 5. The contract -/

/-- **The compiler's contract** at `H`: the code facts the compiler's statements read, each the generic form of a fact
`Claims.Sampling`'s proofs take of the deployed sampled verifier (named in brackets).

* `draws` (`DrawLaw`): on a statement in scope, for each set `B` of proof blocks, a weight of the coins bounds, at each,
  the probability over fresh second-stage coins that the sample misses `B`, and averages at most `P_a(B)`.
* `slots` (the first `n_PU` slots `Claims.Sampling`'s hypotheses read): the sample has at most `n_PU` slots, so the
  extractor reads every one.
* `shared`: at an accepted run, slots of the sample with one key (`skey`) show one session, so the witness the
  extractor finds at the session's first slot serves them all.
* `keys`: the sample's slots hold at most `n_sess` keys, so the extractor runs `A`'s extractor at most `n_sess` times
  per run.
* `unions` (`Deploy.k`'s count): the outer wait's union, over sets of the `n_PU` proof blocks, has at most `2^{n_out}`
  members (`n_PU ≤ n_out`). The inner wait's is one set, which any `n_in` covers.
* `refuses` (`Refuses`): an accepted run's statement is in scope.
* `interior` (`InteriorFixed`): the roots a run shows for a proof block's interior values are the same at fresh
  second-stage coins.
* `fixed` (`DrawFixed`): a sampled proof block's session's rounds are in the second stage, and fresh coins on them keep
  the sample.
* `runs` (`SessionRuns`, with `InnerNumbers` as the shape's equation and the error bounds): at coins `r` that sample
  `u`, the session's statement `xS` is at the block's shape, `A`'s errors there are within the block's bounds (`κb`,
  `γb`), `A` has as many rounds at `xS` as the session (so `emb` embeds them onto the
  session's in order: `SessionRuns`' embedding, the only order embedding onto them), and a prover
  against `A`'s verifier (the sampled prover with `r`'s coins off the session's rounds) is such that, at coins agreeing
  with `r` off the session's rounds, the run accepts only when that prover's run on the session's coins does, and at
  coins agreeing with `r` on the rounds outside the session's before its first, the session is that prover's run and
  its SHA-512 calls are among the compiled verifier's. Only that direction: the run's other sessions may reject what
  this one accepts, so a run with several sampled sessions meets it (`Witness.SamplingGeneric.twoSessions_meets`).
* `opens` (`SessionOpens`): on a statement in scope, at an accepted run that samples `u` and shows its session, a witness
  in `A`'s relation at the session's statement reveals openings such that any values and interior values agreeing with
  those that pass (under the statement's roots and the interior roots the run showed for `u`) make proof block `u`
  correct. -/
structure Sessions (H : Msg → Msg) (L : Compiled Coin A) : Prop where
  draws : ∀ x s P, L.parse x = some s → L.partition s = some P → InScope P s →
    ∀ B ⊆ Finset.range P.blocks.length, ∃ esc : (Fin (L.V.k x) → Coin) → ℝ, (∀ r, 0 ≤ esc r) ∧
      (∀ r, Pr (fun d : Fin (L.V.k x) → Coin => ∀ u ∈ L.checked x (forkOn (L.stage2 x) r d), u ∉ B) ≤ esc r) ∧
      Ex esc ≤ s.design.Pa P B
  slots : ∀ x r, (L.checked x r).length ≤ L.nPU x
  shared : ∀ x (Ω : Type) (P : Prover Ω) ω r i j u u', (run H P L.V x ω r).out = .acc →
    (L.checked x r)[i]? = some u → (L.checked x r)[j]? = some u' →
    L.skey x (L.checked x r) i = L.skey x (L.checked x r) j →
    L.session x u (run H P L.V x ω r).seen = L.session x u' (run H P L.V x ω r).seen
  keys : ∀ x r, ((Finset.range (L.checked x r).length).image (L.skey x (L.checked x r))).card ≤ L.nSess x
  unions : ∀ x, L.nPU x ≤ L.nOut x
  refuses : ∀ x (Ω : Type) (P : Prover Ω) ω r, (run H P L.V x ω r).out = .acc → L.Scope x
  interior : ∀ x (Ω : Type) (P : Prover Ω) ω r d u,
    L.iroots x (run H P L.V x ω (forkOn (L.stage2 x) r d)).seen u = L.iroots x (run H P L.V x ω r).seen u
  fixed : ∀ x r u, u ∈ L.checked x r →
    L.rounds x u ⊆ L.stage2 x ∧ ∀ d, L.checked x (forkOn (L.rounds x u) r d) = L.checked x r
  runs : ∀ x (Ω : Type) (P : Prover Ω) (ω : Ω) r u, u ∈ L.checked x r →
    ∃ (xS : Msg) (P' : Prover Unit), A.shape xS = L.sh x u ∧ A.κ xS ≤ L.κb x u ∧ A.γ xS ≤ L.γb x u ∧
      A.V.k xS = (L.rounds x u).card ∧
      (∀ r' : Fin (L.V.k x) → Coin, (∀ t ∉ L.rounds x u, r' t = r t) →
        (run H P L.V x ω r').out = .acc → (run H P' A.V xS () (L.coinsOf x u xS r')).out = .acc) ∧
      ∀ r' : Fin (L.V.k x) → Coin, (∀ t ∉ L.rounds x u, (∀ t' ∈ L.rounds x u, t < t') → r' t = r t) →
        L.session x u (run H P L.V x ω r').seen = some (xS, (run H P' A.V xS () (L.coinsOf x u xS r')).seen) ∧
          ∀ q ∈ (run H P' A.V xS () (L.coinsOf x u xS r')).vLog, q ∈ (run H P L.V x ω r').vLog
  opens : ∀ x s P (Ω : Type) (Pv : Prover Ω) ω r u xS vS w,
    L.parse x = some s → L.partition s = some P → InScope P s → (run H Pv L.V x ω r).out = .acc →
    u ∈ L.checked x r → L.session x u (run H Pv L.V x ω r).seen = some (xS, vS) → A.R xS w →
    ∃ b, P.blocks[u]? = some b ∧ ∀ (val : ℕ → Option Bool) (ι : ℕ → Bool),
      (∀ q ∈ L.reveal s u (run H Pv L.V x ω r).seen w, L.commit.Opens H q.1 q.2 = true →
        (q.1 = s.roots → val q.2.pos = some q.2.val) ∧
        (q.1 = L.iroots x (run H Pv L.V x ω r).seen u → ι q.2.pos = q.2.val)) →
      s.prog.Correct (s.prog.refIn b val ι) b

/-! ## 6. The claims of a compiled verifier -/

/-- **KS**, on the statements in scope, at every `δ ≥ 0`: `Pr[acc ∧ incorrect ∉ M_δ] ≤ δ + ε_ks + ε_bind`, with
`ε_ks = κ − δ + √(γ·Pr[B])` and `ε_bind = Pr[B_ext]`. -/
def KS (H : Msg → Msg) (L : Compiled Coin A) : Prop :=
  ∀ δ : ℝ, 0 ≤ δ →
    KSLinOn H L.V L.Scope (Sampling.R H L.commit L.parse L.partition δ) L.E (L.κ δ) L.γ (fun _ => 1) (Bext H L.V L.E)

/-- **KS, the count form**: fewer than `t` proof blocks are incorrect, except with probability
`max(0, miss(t)) + ε_ks + ε_bind`. -/
def KSCount (H : Msg → Msg) (L : Compiled Coin A) : Prop :=
  ∀ t, KSLinOn H L.V L.Scope (Sampling.RCount H L.commit L.parse L.partition t) L.E (fun x => L.κ (L.missAt t x) x)
    L.γ (fun _ => 1) (Bext H L.V L.E)

/-- **Bits**, on the statements in scope, at every `δ ≥ 0`: `λ_δ = lamLin(κ_δ, γ, 1, work)`. -/
def Bits (H : Msg → Msg) (L : Compiled Coin A) : Prop :=
  ∀ δ : ℝ, 0 ≤ δ →
    BitsLin H L.V L.Scope (Sampling.R H L.commit L.parse L.partition δ) L.E (Bext H L.V L.E)
      fun x => lamLin (L.κ δ x) (L.γ x) 1 (L.work x)

/-- **KS read at a time budget**, on the statements in scope, at every `δ ≥ 0`: `Pr[acc ∧ incorrect ∉ M_δ]` is at most
`κ_δ + √γ·sbB(2T) + ecrBound(B_ext at T, 2·T·work)` for every prover and every budget `T` on one run's SHA-512 work,
the prover's own included (`Sampling.KSAt`), and `B_ext` at `T` costs at most `2·T·work` in expectation
(`Sampling.BextAtCostLe`): where the hash assumptions hold at its finders, `κ_δ + 2T·√γ/2^256.5 + 2T·work/2^256`. -/
def KSAt (H : Msg → Msg) (L : Compiled Coin A) : Prop :=
  (∀ δ : ℝ, 0 ≤ δ →
    Sampling.KSAt H L.V L.Scope (Sampling.R H L.commit L.parse L.partition δ) L.E (L.κ δ) L.γ L.work) ∧
    Sampling.BextAtCostLe H L.V L.E L.work

/-- **Efficiency**: the extractor's reruns, at most `1 + K_out + k_out·(K_in + k_in·n_sess·ρ_s)` and
`1 + k_out + k_out·k_in·(1 + n_sess·e_s)` expected, one `E^rep` per session; its own SHA-512 calls, the opening checks',
at most `k_out·k_in·n_PU·openCalls`. -/
def Efficient (H : Msg → Msg) (L : Compiled Coin A) : Prop :=
  ExtractRunsLe H L.E (fun x => 1 + L.capOut x + L.kOut x * (L.capIn x + L.kIn x * (L.nSess x * L.ρs x))) ∧
    ExtractRunsExpLe H L.E (fun x => 1 + L.kOut x + (L.kOut x : ℝ) * L.kIn x * (1 + L.nSess x * L.es x)) ∧
    ExtractOwnLe H L.E (fun x => L.kOut x * L.kIn x * L.nPU x * L.openCalls x)

/-- **The opening checks' count**: checking what a session's witness reveals for one proof block, each opening under
its roots, makes at most `openCalls` SHA-512 calls, at a statement that parses (`Sampling.OpenCallsLe`, generic). The
own-call bounds of `Efficient` and `Bits` read it. -/
def OpenCallsLe (H : Msg → Msg) (L : Compiled Coin A) : Prop :=
  ∀ x s, L.parse x = some s → ∀ u v w,
    ((L.reveal s u v w).map fun p => ((L.commit.opens p.1 p.2).eval H).2.length).sum ≤ L.openCalls x

/-! ### The committed form -/

/-- The program's gates at `x`, `0` on a statement that doesn't parse. -/
def nG (L : Compiled Coin A) (x : Msg) : ℕ :=
  match L.parse x with
  | some s => s.prog.gates.length
  | none => 0

/-- **What a run's extractions open** under the statement's roots, at the prover's run `(ω, r)` and the extractor's
coins `c` there: at gate `g`, the first opening there that passes (`Sampling.Wit.val`) among what `runExt` finds; `none`
where none opens `g`, and on a statement that doesn't parse. -/
noncomputable def opensAt (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (r : Fin (L.V.k x) → Coin) (c : L.SlotEC x) (g : ℕ) : Option Bool :=
  match L.parse x with
  | none => none
  | some s => (assemble s ((L.runExt x s r (run H P L.V x ω r).seen c).eval H (run H P L.V x ω)).1).val H L.commit s g

open Classical in
/-- **The committed trace** at the prover's tape `ω`, a definition nobody runs: at gate `g`, the value the prover's runs'
extractions open there most often (`opensAt`, over the window's coins and the extractor's coins; `true` on a tie), a
plurality table (`FlockSoundness.Rewinding.IsPlurality`); `none` where no run's extraction opens `g`. It reads the
statement, the prover and `ω` only, so the prover fixes it before any of the window's coins. -/
noncomputable def regVal (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (g : ℕ) : Option Bool :=
  let p : Bool → ℝ := fun b =>
    Pr fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.opensAt H P x ω rc.1 rc.2 g = some b
  if p true = 0 ∧ p false = 0 then none else some (decide (p false ≤ p true))

open Classical in
/-- **The committed trace's incorrect set** at the prover's tape `ω`: the proof blocks that no values agreeing with the
committed trace on the program's gates, and no interior values, make correct (`Sampling.Faulty`'s form at `regVal`);
empty on a statement that doesn't parse. -/
noncomputable def Freg (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) : Finset ℕ :=
  match L.parse x with
  | none => ∅
  | some s => (Finset.range (L.part s).blocks.length).filter fun v =>
      ¬ ∃ b, (L.part s).blocks[v]? = some b ∧ ∃ (val : ℕ → Option Bool) (ι : ℕ → Bool),
        (∀ g < s.prog.gates.length, val g = L.regVal H P x ω g) ∧ s.prog.Correct (s.prog.refIn b val ι) b

/-- **`B_conf`**: the prover's tape, two independent runs at it, each with the extractor's coins there, and the first
collision among the calls the two runs' extractions make themselves (the openings they check, `A`'s own). Two runs
whose extractions open one gate two ways, or one run's that does, give one (`CommitmentScheme.Binds`). Its cost is the
SHA-512 calls of both runs, both parties', and of both runs' extractions, their reruns' included. -/
noncomputable def Bconf (H : Msg → Msg) (L : Compiled Coin A) : FinderFamily := fun Ω _ _ P x =>
  match L.parse x with
  | none => { S := Unit, out := fun _ => none, cost := fun _ => 0 }
  | some s =>
    { S := Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) × ((Fin (L.V.k x) → Coin) × L.SlotEC x)
      ne := ⟨(Classical.arbitrary _, (fun _ => default, Classical.arbitrary _),
        (fun _ => default, Classical.arbitrary _))⟩
      out := fun q =>
        findColl (ownLog H (run H P L.V x q.1) (L.runExt x s q.2.1.1 (run H P L.V x q.1 q.2.1.1).seen q.2.1.2) ++
          ownLog H (run H P L.V x q.1) (L.runExt x s q.2.2.1 (run H P L.V x q.1 q.2.2.1).seen q.2.2.2))
      cost := fun q =>
        (run H P L.V x q.1 q.2.1.1).pLog.length + (run H P L.V x q.1 q.2.1.1).vLog.length +
          (run H P L.V x q.1 q.2.2.1).pLog.length + (run H P L.V x q.1 q.2.2.1).vLog.length +
          ((L.runExt x s q.2.1.1 (run H P L.V x q.1 q.2.1.1).seen q.2.1.2).eval H (run H P L.V x q.1)).2.1 +
          ((L.runExt x s q.2.2.1 (run H P L.V x q.1 q.2.2.1).seen q.2.2.2).eval H (run H P L.V x q.1)).2.1 }

/-- **The committed form's error**: `ε_reg = κ_s + √(k(x)·γ_s·Pr[B]) + N·√Pr[B_conf] + Pr[B_conf]`, the sessions'
knowledge error at the sampled proof block a selection picks, and the committed trace's binding, `N` the program's
gates (the rewinding lemma's positions, each run opening at most `N`). -/
noncomputable def εreg (H : Msg → Msg) (L : Compiled Coin A) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) : ℝ :=
  L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) + L.nG x * Real.sqrt ((Bconf H L Ω P x).Pr H) +
    (Bconf H L Ω P x).Pr H

/-- **KS, the committed form** (the registered form of `Claims.Sampling`'s period bounds, generic), on the statements in
scope, at every prover: the run accepts with a sampled proof block in the committed trace's incorrect set (`Freg`, which
the prover's tape fixes before the window's coins) with probability at most `ε_reg` (`εreg`); so, by the sample, it
accepts with that set outside `M_δ` with probability at most `δ + ε_reg`, at every `δ ≥ 0`. The sampling term is the
acceptance probability of a set fixed before the coins, `P_a(Freg)`, as a monitoring period's pooling reads it.

`Freg` is what `A`'s counted assemble step makes of the prover's reruns' views (`Arg`), its plurality over the coins:
it is the trace the prover committed to as far as that step is the program its count describes, which at Flock is the
deployed decoder (`Arg.ofFlock`, and `trustedBase`'s decoder count). -/
def KSReg (H : Msg → Msg) (L : Compiled Coin A) : Prop :=
  ∀ x, L.Scope x → ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω),
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.Freg H P x p.1) ≤ L.εreg H x P ∧
      ∀ s, L.parse x = some s → ∀ δ : ℝ, 0 ≤ δ →
        Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
          L.Freg H P x p.1 ∉ s.design.M (L.part s) δ) ≤ δ + L.εreg H x P

/-! ### Completeness -/

/-- **Complete**, on the statements in scope, when no proof block is faulty (`Sampling.Complete`, generic): the
reference prover is accepted with probability at least `1 − δ_c` at every tape. -/
def Complete (H : Msg → Msg) (L : Compiled Coin A) : Prop :=
  Claims.Complete H L.V L.P (Rhon H L.commit L.parse L.partition) L.Scope (L.δc H)

/-- **The completeness contract** at `H`, over `A`'s honest prover `PA` on tapes `Ω`, the statements it serves
(`ScopeA`) and its error `δA`: the generic forms of the facts `Claims.Sampling`'s completeness reads of the deployed
record's reference prover (named in brackets). On a statement in scope, at a witness with no incorrect proof block:

* `honest` (`HonestSessions`): at coins that sample `u`, `u`'s session is the run of `A`'s honest prover, at a statement
  `A` serves, with as many rounds as the session, and a witness in `A`'s relation, on the session's coins (`coinsOf`),
  wherever the coins off the session's rounds are fixed;
* `accepts` (`AcceptsSessions`): the run accepts when every sampled proof block's session that shows accepts;
* `among` (`SessionsAmong`): every sampled proof block's session is the session of the block sampled at a slot below
  `K x`: at `K = n_PU` by the contract's `slots`, and at `K = 1` where one session proves every sampled block;
* `err`: the compiled error is nonnegative and at least `K x` times each such session's error, capped at `1`. -/
structure CompleteSessions (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (PA : A.W → Prover Ω)
    (ScopeA : Msg → Prop) (δA : Msg → ℝ) (K : Msg → ℕ) : Prop where
  honest : ∀ x w, L.Scope x → Rhon H L.commit L.parse L.partition x w → ∀ (ω : L.Tape) r u, u ∈ L.checked x r →
    ∃ (xS : Msg) (wS : A.W) (ωS : Ω), ScopeA xS ∧ A.R xS wS ∧ A.V.k xS = (L.rounds x u).card ∧
      ∀ r' : Fin (L.V.k x) → Coin, (∀ t ∉ L.rounds x u, r' t = r t) →
        L.session x u (run H (L.P w) L.V x ω r').seen =
          some (xS, (run H (PA wS) A.V xS ωS (L.coinsOf x u xS r')).seen)
  accepts : ∀ x w, L.Scope x → Rhon H L.commit L.parse L.partition x w → ∀ (ω : L.Tape) r,
    (∀ u ∈ L.checked x r, ∀ xS vS, L.session x u (run H (L.P w) L.V x ω r).seen = some (xS, vS) → vS.out = .acc) →
      (run H (L.P w) L.V x ω r).out = .acc
  among : ∀ x w, L.Scope x → Rhon H L.commit L.parse L.partition x w → ∀ (ω : L.Tape) r u, u ∈ L.checked x r →
    ∃ p < K x, ∃ u', (L.checked x r)[p]? = some u' ∧
      L.session x u (run H (L.P w) L.V x ω r).seen = L.session x u' (run H (L.P w) L.V x ω r).seen
  err : ∀ x, 0 ≤ L.δc H x ∧ ∀ w (ω : L.Tape) r u xS vS, Rhon H L.commit L.parse L.partition x w →
    u ∈ L.checked x r → L.session x u (run H (L.P w) L.V x ω r).seen = some (xS, vS) →
      K x * min 1 (δA xS) ≤ L.δc H x

end Compiled

/-! ### The masked compiled verifier -/

/-- **Where a masked run is, for the reference prover**, at a history: slot `j`'s session under the statement `xS` at
its own history `hS`, or a message the run's framing fixes outside the sessions (`Claims.Sampling`'s `StepZ`). -/
inductive Step where
  | session (j : ℕ) (xS : Msg) (hS : List (Msg × Msg))
  | say (m : Msg)

/-- The session's statement; `[]` outside the sessions. -/
def Step.stmt : Step → Msg
  | .session _ xS _ => xS
  | .say _ => []

/-- What the framing says; `[]` in a session. -/
def Step.said : Step → Msg
  | .session .. => []
  | .say m => m

/-- A transcript as the prover's history: each prover message with the verifier's message, `[]` at a decision. -/
def histOf (ms : List (Msg × Out)) : List (Msg × Msg) :=
  ms.map fun p => (p.1, match p.2 with
    | .msg m => m
    | _ => [])

open Classical in
/-- `A`'s simulator at the session statement `xS`, given the outcome `o`, on the coins `C` read as its own through a
bijection; nothing where there is none. -/
noncomputable def simAt {St : Type} (S : SimOut St) (C : Type) (xS : Msg) (o : St) (c : C) : SP (List (Msg × Out)) :=
  if h : Nonempty (C ≃ S.C xS) then S.prog xS o (h.some c) else .ret []

/-- **`V*` in a session**, after the prover messages `pre`: `n` rounds, each answered as `V*` answers at `x` after `pre`
and the session's messages so far. -/
def residual {Θ : Type} (Vs : VStar Θ) (x : Msg) (pre : List Msg) (n : ℕ) : VStar Θ :=
  ⟨fun _ => n, fun _ as θ => Vs.next x (pre ++ as) θ⟩

/-- **A masked compiled sampled verifier over the proof system `A`** (`Sampling.Deploy` masked, with its Flock sessions
`A`'s): its reference prover's framing, which routes each history (`step`: a message the framing fixes from the
statement and the history, or a slot's session), the rounds before the sessions (`k₀`), the sessions (`M`) and each
one's rounds (`n`); the witness the reference prover gives a session (`wit`: from the audit's witness, at the audit's
statement and the session's, with the SHA-512 calls of opening what it reveals); each slot's simulator coins (`CB`,
`A`'s at the session's statement up to a bijection); and the reference prover's tape at a nonce and a rest per session
(`tapeOf`). -/
structure ZkCompiled (Coin : Type) (A : ZkArg Coin) extends Compiled Coin A.toArg where
  step : Msg → List (Msg × Msg) → Step
  k₀ : Msg → ℕ
  M : Msg → ℕ
  n : Msg → ℕ → ℕ
  wit : Sampling.Wit → Msg → Msg → OC A.W
  CB : Msg → ℕ → Type
  [cbFin : ∀ x j, Fintype (CB x j)]
  [cbNe : ∀ x j, Nonempty (CB x j)]
  tapeOf : (x : Msg) → (Fin (M x) → A.Nonce) × (Fin (M x) → A.Rest) → Tape

attribute [instance] ZkCompiled.cbFin ZkCompiled.cbNe

namespace ZkCompiled

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : ZkArg Coin}

/-- Where session `j` starts: after the first part's rounds and the earlier sessions'. -/
def start (L : ZkCompiled Coin A) (x : Msg) (j : ℕ) : ℕ := L.k₀ x + ∑ i ∈ Finset.range j, L.n x i

/-- The masked compiled verifier's reference prover at the witness `w` and the statement `x`, on a nonce and a rest per
session. -/
def prover (L : ZkCompiled Coin A) (w : Sampling.Wit) (x : Msg) :
    Prover ((Fin (L.M x) → A.Nonce) × (Fin (L.M x) → A.Rest)) :=
  (L.P w).comap (L.tapeOf x)

/-- **The prover the compiler builds from `A`'s** at the witness `w` and the statement `x`: what the framing fixes
outside the sessions; in slot `j`'s session under `xS`, `A`'s prover at the witness `wit` gives it, at the session's
history, on slot `j`'s nonce and rest (`Framing.prover`). -/
def composed (L : ZkCompiled Coin A) (w : Sampling.Wit) (x : Msg) :
    Prover ((Fin (L.M x) → A.Nonce) × (Fin (L.M x) → A.Rest)) :=
  ⟨fun x' hist t =>
    match L.step x' hist with
    | .say m => .ret m
    | .session j xS hS =>
      if h : j < L.M x then (L.wit w x' xS).bind fun wS => (A.P wS).next xS hS (t.1 ⟨j, h⟩, t.2 ⟨j, h⟩)
      else .ret []⟩

/-- **The audit's public outcome** on a transcript: each session's, `A`'s outcome at the session's statement (the
router's at the history that starts it) on its rounds' messages. -/
def outOf (L : ZkCompiled Coin A) (x : Msg) (ms : List (Msg × Out)) : List A.St :=
  (List.range (L.M x)).map fun j =>
    A.out (L.step x (histOf (ms.take (L.start x j)))).stmt ((ms.drop (L.start x j)).take (L.n x j))

/-- The sessions from `js` on, simulated after the transcript `ms` while the run goes on: each `A`'s simulator at the
statement the simulated history routes to, given the session's outcome in `os`, its questions after the messages so far
(`spUnder`). -/
noncomputable def simSessions (L : ZkCompiled Coin A) (x : Msg) (os : List A.St) (c : (j : Fin (L.M x)) → L.CB x j) :
    List (Fin (L.M x)) → List (Msg × Out) → SP (List (Msg × Out))
  | [], ms => .ret ms
  | j :: js, ms =>
    match goesOn ms, os[j.val]? with
    | true, some o =>
      spBind (spUnder (ms.map Prod.fst) (simAt A.S (L.CB x j) (L.step x (histOf ms)).stmt o (c j))) fun τ =>
        simSessions L x os c js (ms ++ τ)
    | _, _ => .ret ms

/-- **The composed simulator**, given the published leaf layer, which it doesn't read, and the sessions' outcomes: the
framing's rounds, whose messages are functions of the statement and the history, then each session's simulator in turn
(`simSessions`) on its slot's coins (`Framing.sim`). -/
noncomputable def sim (L : ZkCompiled Coin A) : SimOut (List Msg × List A.St) where
  C x := (j : Fin (L.M x)) → L.CB x j
  prog x o c :=
    spBind (simRounds (fun ms => .ret (L.step x (histOf ms)).said) (L.k₀ x) []) fun ms =>
      L.simSessions x o.2 c (List.finRange (L.M x)) ms

/-- **The compiled zero-knowledge error** at `(x, w, V*, θ)` and the outcome `o`: over the sessions, the expectation over
the honest tape of `A`'s error where the honest run against `V*` reaches session `j`, at the session's statement, the
witness the reference prover gives it, `V*` after the run's messages before it (`residual`), `θ` and the session's
outcome in `o`; `0` where the run stops before the session. Each term is a function of the statement, the witness, the
verifier and the outcome, as `A`'s is. -/
noncomputable def err (H : Msg → Msg) (L : ZkCompiled Coin A) (x : Msg) (w : Sampling.Wit) (Θ : Type) (Vs : VStar Θ)
    (θ : Θ) (o : List Msg × List A.St) : ℝ :=
  ∑ j ∈ Finset.range (L.M x), Ex fun t : (Fin (L.M x) → A.Nonce) × (Fin (L.M x) → A.Rest) =>
    let pre := (runStar H (L.prover w x) Vs x t θ).msgs.take (L.start x j)
    let xS := (L.step x (histOf pre)).stmt
    match o.2[j]?, decide (pre.length = L.start x j) && goesOn pre with
    | some oj, true => A.εzk xS ((L.wit w x xS).eval H).1 Θ (residual Vs x (pre.map Prod.fst) (L.n x j)) θ oj
    | _, _ => 0

/-- **The compiler's zero-knowledge contract** at `H`: the code facts `CompilerZK` reads, each the generic form of a fact
`Claims.Sampling`'s zero knowledge takes of the masked deployed record (named in brackets), at each statement that
parses.

* `rounds` (`zk_seq`'s `hV`): the compiled verifier's rounds are the first part's and the sessions'.
* `routed` (`Framing.Routed`): within the first part the router says what the framing fixes, and within session `j` it
  is in slot `j`'s session under the statement of the session's first history, at the session's own history.
* `shaped` (`Framing.Shaped`): at every history that starts session `j`, `A`'s sessions at its statement run the slot's
  rounds, and `A`'s simulator's coins there are the slot's up to a bijection.
* `witness` (`Framing.SessionWit`): where no proof block is incorrect, the witness the reference prover gives each
  session is in `A`'s relation at the session's statement.
* `honest` (`deployZ_honestSeq`): within the compiled verifier's rounds, the reference prover sends the composed
  prover's messages. -/
structure ZkSessions (H : Msg → Msg) (L : ZkCompiled Coin A) : Prop where
  rounds : ∀ x s, L.parse x = some s → L.V.k x = L.start x (L.M x)
  routed : ∀ x s, L.parse x = some s →
    (∀ hist : List (Msg × Msg), hist.length < L.k₀ x → ∃ m, L.step x hist = .say m) ∧
      ∀ j < L.M x, ∀ hist : List (Msg × Msg), L.start x j ≤ hist.length → hist.length < L.start x (j + 1) →
        L.step x hist = .session j (L.step x (hist.take (L.start x j))).stmt (hist.drop (L.start x j))
  shaped : ∀ x s, L.parse x = some s → ∀ j < L.M x, ∀ hist : List (Msg × Msg), hist.length = L.start x j →
    A.kz (L.step x hist).stmt = L.n x j ∧ Nonempty (L.CB x j ≃ A.S.C (L.step x hist).stmt)
  witness : ∀ x w, Rhon H L.commit L.parse L.partition x w → ∀ j < L.M x, ∀ hist : List (Msg × Msg),
    hist.length = L.start x j → A.R (L.step x hist).stmt ((L.wit w x (L.step x hist).stmt).eval H).1
  honest : ∀ x s, L.parse x = some s → ∀ w hist t, hist.length < L.start x (L.M x) →
    (((L.prover w x).next x hist t).eval H).1 = (((L.composed w x).next x hist t).eval H).1

/-- **Zero knowledge with a public outcome, relative to the published leaf layer** (`Sampling.ZK` in `ZKOut`'s form): of
the reference prover when no proof block is incorrect, against every verifier of the compiled verifier's rounds, per
outcome (the trace's leaf layer `l x` and the sessions' outcomes, `outOf`), at the composed simulator (`sim`) and the
compiled error (`err`). Hiding the leaf layer is the commitment's (`Claims.Hm96Hiding` at Flock). -/
def ZK (H : Msg → Msg) (L : ZkCompiled Coin A) : Prop :=
  ∀ l : Msg → List Msg,
    ZKOut (fun x => Fin (L.M x) → A.Nonce) (fun x => Fin (L.M x) → A.Rest) H L.V.k L.prover
      (fun x w => Rhon H L.commit L.parse L.partition x w ∧ leafLayer H L.commit L.parse x w = l x)
      (fun x ms => (l x, L.outOf x ms)) L.sim (L.err H)

end ZkCompiled

/-! ## 7. The compiler's statements -/

/-- **The sampled compiler is knowledge sound**: every compiled sampled verifier that meets the contract over a
knowledge-sound proof system for proof blocks, with a binding commitment, meets `KS` at the proof system's errors. -/
def CompilerKS : Prop :=
  ∀ (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (A : Arg Coin) (L : Compiled Coin A)
    (H : Msg → Msg), A.KS H → L.commit.Binds H → L.Sessions H → L.KS H

/-- **The committed form**: as `CompilerKS`, `KSReg`. A corollary of the sessions' bound at the sampled proof block a
selection picks (`SessionKS`, which `A.KS` gives at the contract) and the rewinding lemma
(`FlockSoundness.Rewinding.prBad_le`): at an accepted run with a sampled proof block in `Freg`, the block's session is
not extracted, or its extraction reveals (`opens`) an opening off the committed trace, which a run's extractions open off
the plurality with probability at most `N·√Pr[B_conf]`, or open two ways in one run, a collision `B_conf` finds. -/
def CompilerKSReg : Prop :=
  ∀ (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (A : Arg Coin) (L : Compiled Coin A)
    (H : Msg → Msg), A.KS H → L.commit.Binds H → L.Sessions H → L.KSReg H

/-- **The count form**: as `CompilerKS`, `KSCount`. -/
def CompilerKSCount : Prop :=
  ∀ (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (A : Arg Coin) (L : Compiled Coin A)
    (H : Msg → Msg), A.KS H → L.commit.Binds H → L.Sessions H → L.KSCount H

/-- **Bits**: as `CompilerKS`, `Bits`, with the opening checks' count. -/
def CompilerBits : Prop :=
  ∀ (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (A : Arg Coin) (L : Compiled Coin A)
    (H : Msg → Msg), A.KS H → L.commit.Binds H → L.Sessions H → L.OpenCallsLe H → L.Bits H

/-- **KS read at a time budget**: as `CompilerBits`, `KSAt`, each hash term at a budget the reader computes from the
prover's time rather than at the calls the oracle logs (a corollary of `CompilerKS` and `CompilerEfficient`). -/
def CompilerKSAt : Prop :=
  ∀ (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (A : Arg Coin) (L : Compiled Coin A)
    (H : Msg → Msg), A.KS H → L.commit.Binds H → L.Sessions H → L.OpenCallsLe H → L.KSAt H

/-- **Efficiency**: the compiled extractor's reruns and own calls, at the contract, from the opening checks' count. -/
def CompilerEfficient : Prop :=
  ∀ (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (A : Arg Coin) (L : Compiled Coin A)
    (H : Msg → Msg), L.Sessions H → L.OpenCallsLe H → L.Efficient H

/-- **The sampled compiler is zero knowledge**: every masked compiled sampled verifier that meets the zero-knowledge
contract over a proof system for proof blocks that is zero knowledge with a public outcome is zero knowledge with a
public outcome, relative to the published leaf layer, at the composed simulator and the sessions' summed errors. -/
def CompilerZK : Prop :=
  ∀ (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (A : ZkArg Coin) (L : ZkCompiled Coin A)
    (H : Msg → Msg), A.ZK H → L.ZkSessions H → L.ZK H

/-- **The sampled compiler preserves completeness**: every compiled sampled verifier that meets the contract and the
completeness contract over a proof system for proof blocks whose honest prover is complete is complete at its error. -/
def CompilerComplete : Prop :=
  ∀ (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (A : Arg Coin) (L : Compiled Coin A)
    (H : Msg → Msg) (Ω : Type) [Nonempty Ω] (PA : A.W → Prover Ω) (ScopeA : Msg → Prop) (δA : Msg → ℝ)
    (K : Msg → ℕ), Claims.Complete H A.V PA A.R ScopeA δA → L.Sessions H → L.CompleteSessions H PA ScopeA δA K →
      L.Complete H

/-! ## 8. Flock -/

section Flock

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

/-- **Flock's assemble step, counted**: `Flock.assemble` at the deployed decoder (`D.decode`), with its steps: per table
of the statement, the decoder's on the table's opened columns (`D.dec`'s count) and one check per candidate it lists. -/
def flockAssemble (D : Claims.Flock.Deploy Coin Col T) (x : Msg) (os : List (ℕ → List (ℕ × Col))) :
    Option Claims.Flock.Wit × ℕ :=
  (Claims.Flock.assemble D.parse D.table D.decode D.read x os,
    match D.parse x with
    | none => 0
    | some s => ((List.range s.tables).map fun j =>
        (D.dec (os.flatMap fun o => o j)).2 + (D.decode (os.flatMap fun o => o j)).length).sum)

/-- **Flock as a proof system for proof blocks**: `E^rep` at the deployed decoder (`Flock.E D D.decode`), its assemble
step counted (`flockAssemble`), its shape the numbers it reads (waits `m′`, truncation `K`, rounds, and `N₀`, which
fixes each wait's `k = ⌈4·N₀/e⌉`, `Flock.Deploy.k`, and so its expected reruns), and its errors Flock's (`Flock.κ`,
`Flock.γ`). -/
noncomputable def Arg.ofFlock (D : Claims.Flock.Deploy Coin Col T) : Arg Coin where
  Sh := ℕ × ℕ × ℕ × ℝ
  shape x := (D.waits x, D.cap x, D.V.k x, D.N₀ x)
  V := D.V
  rounds s := s.2.2.1
  wf _ := rfl
  W := Claims.Flock.Wit
  R := Claims.Flock.R D.parse
  Op := ℕ → List (ℕ × Col)
  opens v := [D.cols v]
  assemble := flockAssemble D
  check := Claims.Flock.check D.parse
  waits s := s.1
  k s := ⌈4 * s.2.2.2 / Real.exp 1⌉₊
  cap s := s.2.1
  κ := Claims.Flock.κ D
  γ := Claims.Flock.γ D

/-- **Flock's KS is `Arg.ofFlock`'s**: `Flock.KS` at the deployed decoder, which is within Flock's radius. -/
def FlockArgKS : Prop :=
  ∀ (H : Msg → Msg) (D : Claims.Flock.Deploy Coin Col T) [Zero Col],
    Claims.Flock.KS H D → Claims.Flock.ListWithin D.len D.enc D.θ D.decode → (Arg.ofFlock D).KS H

/-- **The sampled compiler at Flock**: every compiled sampled verifier over Flock at the deployed decoder, within its
radius, that meets the contract, with a binding commitment, meets `KS`, from `Flock.KS` (`CompilerKS` and
`FlockArgKS`). `Flock.KS`, the binding and the contract are its hypotheses. -/
def FlockKS : Prop :=
  ∀ (H : Msg → Msg) (D : Claims.Flock.Deploy Coin Col T) [Zero Col] (L : Compiled Coin (Arg.ofFlock D)),
    Claims.Flock.KS H D → Claims.Flock.ListWithin D.len D.enc D.θ D.decode → L.commit.Binds H → L.Sessions H →
      L.KS H

/-- **The committed form at Flock**: as `FlockKS`, `KSReg` (`CompilerKSReg` and `FlockArgKS`). -/
def FlockKSReg : Prop :=
  ∀ (H : Msg → Msg) (D : Claims.Flock.Deploy Coin Col T) [Zero Col] (L : Compiled Coin (Arg.ofFlock D)),
    Claims.Flock.KS H D → Claims.Flock.ListWithin D.len D.enc D.θ D.decode → L.commit.Binds H → L.Sessions H →
      L.KSReg H

end Flock

end Claims.SamplingGeneric
