import Definitions.Claims
import Definitions.Claims.IndexedTapes

/-!
# Claims: the ZK-recursion compiler (a draft, for Daniel's read before it freezes)

The compiler is code plus proof, so it has no claims of its own to read. The claims below are its output's: the
compiled verifier's, parametric in the inner protocol (`Inner`, a `Deploy`-like record), the outer `--zk` Flock
(`Flock.Deploy`) and what the compiled verifier brings (`Deploy`). Shared definitions are `Claims`'; the template's
eight items:

1. **Statement format.** `x` is the canonical bytes of what the compiled verifier reads, parsed by its own parser
   (`Deploy.parse`) into the inner statement's bytes, the inner statement's dimensions as the verifier sets it up, the
   commitment to the circuit (the private-circuit variant only), the recursion's form, and each V* session's draw as
   its record holds it, where the statement carries the records (`Stmt`, `Form`, `Draw`).
2. **Witness.** The inner protocol's (`W`), in both variants. In the private-circuit one the circuit is in the inner
   witness, which the universal unit's relation reads, and its commitment is hm96 rows that unit outputs.
3. **Relation (inherited).** `R(x, w) :⇔ x` parses, has no circuit commitment, and `R_I(x_I, w)` (`R`). The private
   variant: `x` parses with a circuit commitment, and `R_I(x_I, w)`, whose outputs bind it to the circuit (`RPriv`).
4. **Scope, readable** (`Scope`, `InScope`): the inner scope, and the inner shapes the compiled verifier accepts:
   dense, holo (private inner circuits) and sparse (candidate 3's RecSparse phase and R). A fast100 inner table,
   `k_log ≥ 6`, two claims and two per region, no circuit commitment; either no RecSparse phase and no R, or a valid
   RecSparse phase at `T_LOG = 0` with `6 ≤ ℓ ≤ k_log`, the inner statement's dimensions the form's and its unit range
   inside a block, and R at `table_m` with two claims. Every V* session draws every unit of its population
   (`Draw.Full`). There is no recursion-level scope refusal: `ScopeMatches` states that this is the compiled
   verifier's own scope check, its setup's checks of the inner shape and its refusal of a V* session whose draw is not
   every unit. Where the V* records are run messages, the statement carries no draws, and the compiled verifier
   refuses a session whose draw is not every unit when that session decides.
5. **Named pieces.** A run is the inner session's rounds, the prover's staging rounds (`Deploy.kS`), at which it stages
   the V* statements, then the V* sessions. `E*` (`E`): the outer extractor `E_O` (`EO`: per V* session, `E^rep` over
   forks of the judged run at the session's first round, with the outer Flock's assemble and check), then the inner one
   (`E^rep` at the inner protocol's waits, assemble and check), each inner rerun served by `E_O` at that rerun's coins
   and read off its output at the claim's oracle (`served`, `dec`). The simulator `Sim`: the inner session's commit
   strings to zero messages under fresh salts at the pinned hm96 key (`Deploy.zeroCommit`), the staging messages
   (`Deploy.stage`) at their commit strings to zero messages under fresh salts (the honest prover stages its values'
   under its own), then the outer simulator on each V* statement (`SO`, the named Flock simulator, a parameter as in
   `Flock.ZK`). The extraction-based finder `Be`: `B`'s fork, `E_O` on both runs, a collision among the SHA-512
   evaluations their outer witnesses record (`Deploy.calls`), which no party computes in the clear.
6. **Claims.** `KS` with costs (`Efficient`), `Bits`, `ZK`, `Complete` on the scope of the deployed reference prover
   (`Deploy.P`) at the deployed error (`Deploy.δc`), `ScopeMatches`, `Nontrivial`, and the private variant's `KSPriv`,
   `ZKPriv`; `At Flock` instantiates the inner protocol with Flock and its decoder. The inner protocol's claim the
   recursion rests on is `Inner.KSAbort`: its KS against provers that may abort after the inner rounds, on coins the
   inner verifier doesn't read (with no aborts, `Inner.KS`).
7. **Assumptions used.** `StrictBirthday` (for `B`), `ExpectedCR` (for `Be`), each read once in `Bits`; `Hm96Hiding`
   (hm96 leaves hide up to `δ₁`) at the deployed key's salt mask and salt hash (`Deploy.mask`, `Deploy.saltHash`), read
   in `ZK`, `ZKPriv` and `ZKStop`; and `trustedBase`. The decoders are those within Flock's list-decoding radius
   (`decOK`, `Flock.ListWithin`).
8. **Status.** A draft: nothing here is frozen. `γ` and `γe` are the constants the recursion's KS proof gives from
   `Inner.KSAbort`; `κ` and `cBe` are the draft's, fixed by the proof before freezing. `ZK`'s errors are Flock's proved
   `--zk` simulation error at each V* session (`Nums.εzk`) and `2δ₁` per commit string of the inner session and the
   staging messages. `ZKStop` restates `ZK` for Daniel's card, statement only: of the record's own reference prover on
   the finite tape each statement's nonce and rest draw (`StopDeploy.prover`, `ZKmalRx`), which stops where the
   statement and the public transcript before the sessions say the honest proof can be refused (`StopDeploy.stops`),
   at the simulator that stops there too and commits to the zero digest as many strings at each inner round as the
   reference prover's round carries (`SimStop`, `StopDeploy.nAt`), within `2δ₁` per string (`εzkStop`).

Mathlib only, as `Definitions/` is. So every claim is parametric in the deployed verifier, its parser and its readers:
binding them to the compiled verifier is the code side's, as Flock's is.
-/

namespace Claims.Recursion

open Claims
open scoped NNReal

/-! ## Rerun programs served by rerun programs -/

/-- **Serve each rerun of `p` by the program `q r`**, which reruns at `R₂` and returns what `p` sees of its rerun at
`r`. -/
def serve {R₁ R₂ α : Type} (q : R₁ → RP R₂ Seen) : RP R₁ α → RP R₂ α
  | .ret a => .ret a
  | .hash m k => .hash m fun a => serve q (k a)
  | .rerun r k => (q r).bind fun v => serve q (k v)

/-- A rerun at `r`, seen through `f`. -/
def viewing {R : Type} (f : Seen → Seen) (r : R) : RP R Seen := .rerun r fun v => .ret (f v)

/-- **`E^rep`'s program over draws `D`**: rerun the judged draw `d₀`; if it accepted, make the `m` waits over the draws
`ρ`, each stopping at `k` accepted reruns, and return the first assembled candidate that `check` accepts. `ERep`'s
program is this one at the run's coins. -/
def erep {D W Op : Type} (m k K : ℕ) (opens : Seen → List Op) (assemble : List Op → Option W) (check : W → Bool)
    (d₀ : D) (ρ : Fin m → Fin K → D) : RP D (Option W) :=
  .rerun d₀ fun v₀ =>
    if v₀.out = .acc then
      firstWait k opens assemble check ((List.finRange m).map fun i => (List.finRange K).map (ρ i))
    else .ret none

/-- Run the programs in order; all their results, or none at the first that gives up. -/
def collect {R α : Type} : List (RP R (Option α)) → RP R (Option (List α))
  | [] => .ret (some [])
  | p :: ps => p.bind fun o => match o with
    | none => .ret none
    | some a => (collect ps).bind fun os => .ret (os.map fun l => a :: l)

/-- A fork at round `off`: the coins before it are `r`'s, the rest `r'`'s. -/
def fork {Coin : Type} {k : ℕ} (off : ℕ) (r r' : Fin k → Coin) : Fin k → Coin :=
  fun i => if i.val < off then r i else r' i

section Served

variable {Coin W W₂ Op : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **The outer stage at a draw**: `E₂` at the draw's run coins and its own coins, its output read by `dec` as what
the inner stage sees. -/
def outerStage {V : Verifier Coin} (E₂ : Extractor Coin W₂ V)
    (dec : (x : Msg) → (Fin (V.k x) → Coin) → Option W₂ → Seen) (x : Msg) (d : (Fin (V.k x) → Coin) × E₂.C x) :
    RP (Fin (V.k x) → Coin) Seen :=
  (E₂.ext x d.1 d.2).bind fun w₂ => .ret (dec x d.1 w₂)

/-- **`E^rep` served by `E₂`**: `E^rep` over draws of the run's coins and `E₂`'s coins, its judged draw the judged
run's coins `r₀` with `E₂`'s coins `c.1`, each draw's rerun served by `E₂` there (`outerStage`). -/
def served (V : Verifier Coin) (E₂ : Extractor Coin W₂ V)
    (dec : (x : Msg) → (Fin (V.k x) → Coin) → Option W₂ → Seen) (m k K : Msg → ℕ) (opens : Seen → List Op)
    (assemble : Msg → List Op → Option W) (check : Msg → W → Bool) : Extractor Coin W V where
  C x := E₂.C x × (Fin (m x) → Fin (K x) → (Fin (V.k x) → Coin) × E₂.C x)
  ext x r₀ c := serve (outerStage E₂ dec x) (erep (m x) (k x) (K x) opens (assemble x) (check x) (r₀, c.1) c.2)

end Served

/-! ## Simulator programs -/

def spBind {α β : Type} : SP α → (α → SP β) → SP β
  | .ret a, f => f a
  | .hash q k, f => .hash q fun a => spBind (k a) f
  | .ask as k, f => .ask as fun b => spBind (k b) f

/-- An oracle program as a simulator program. -/
def spOf {α : Type} : OC α → SP α
  | .ret a => .ret a
  | .query q k => .hash q fun a => spOf (k a)

/-- A simulator program whose questions to the verifier follow the prover messages `pre`. -/
def spUnder {α : Type} (pre : List Msg) : SP α → SP α
  | .ret a => .ret a
  | .hash q k => .hash q fun a => spUnder pre (k a)
  | .ask as k => .ask (pre ++ as) fun b => spUnder pre (k b)

/-- The run goes on: no message yet, or the verifier's last answer is a message. -/
def goesOn (ms : List (Msg × Out)) : Bool :=
  match ms.getLast? with
  | some (_, .msg _) => true
  | some _ => false
  | none => true

/-- `n` simulated rounds after the transcript `ms`, while the run goes on: each round's prover message is `f`'s at the
transcript so far, put to the verifier. -/
def simRounds (f : List (Msg × Out) → SP Msg) : ℕ → List (Msg × Out) → SP (List (Msg × Out))
  | 0, ms => .ret ms
  | n + 1, ms =>
    if goesOn ms then
      spBind (f ms) fun a => .ask (ms.map Prod.fst ++ [a]) fun b => simRounds f n (ms ++ [(a, b)])
    else .ret ms

/-- A simulator of one V* session, at the recursion's statement and the session's index: its coins, and its program on
the V* statement. -/
structure OSim where
  C : Type
  [fin : Fintype C]
  [ne : Nonempty C]
  prog : Msg → C → SP (List (Msg × Out))

attribute [instance] OSim.fin OSim.ne

/-! ## The inner protocol -/

/-- **What the inner protocol brings** (a `Deploy`-like record): its verifier, its relation and readable scope, its
extractor `E^rep`'s pieces (`opens`, `assemble`, `check`, the waits, the accepted reruns per wait, the truncation) and
its errors `κ`, `γ`. -/
structure Inner (Coin W : Type) where
  V : Verifier Coin
  R : Rel W
  Scope : Msg → Prop
  Op : Type
  opens : Seen → List Op
  assemble : Msg → List Op → Option W
  check : Msg → W → Bool
  waits : Msg → ℕ
  k : Msg → ℕ
  cap : Msg → ℕ
  κ : Msg → ℝ
  γ : Msg → ℝ

section InnerClaims

variable {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- The inner protocol's extractor, `E^rep`. -/
def Inner.E (I : Inner Coin W) : Extractor Coin W I.V := ERep I.V I.waits I.k I.cap I.opens I.assemble I.check

/-- The inner protocol's own `KS`. -/
def Inner.KS (H : Msg → Msg) (I : Inner Coin W) : Prop := Claims.KS H I.V I.R I.E I.κ I.γ

end InnerClaims

/-! ### The inner protocol's KS against provers that may abort

`E*` sees each inner rerun through `E_O` at fresh outer coins, rejected wherever the compiled run rejects there. The
outer sessions come after the inner rounds, so the compiled prover decides there, on the inner coins and on coins the
inner verifier never reads, whether `E*` sees an inner rerun accepted. The recursion rests on the inner protocol's KS
against such provers (`Inner.KSAbort`): after the inner rounds, a prover may abort its run, reading its tape, every coin
of the run and fresh coins of its own for that run, which the inner verifier doesn't read; an aborted run sends nothing
and is rejected. -/

/-- An aborted run: nothing sent, rejected. -/
def abortTrace : Trace := ⟨[], [], [], .rej⟩

/-- **A run that may abort after the inner rounds**: `P`'s run at the tape `ω` and the coins `t.1`, unless `keep`, which
reads the tape, every coin of the run and the run's fresh coins `t.2`, refuses it; then the run aborts. -/
def abortRun {Coin : Type} [Inhabited Coin] {Ω A : Type} (H : Msg → Msg) (P : Prover Ω) (V : Verifier Coin)
    (x : Msg) (keep : Ω → (Fin (V.k x) → Coin) → A → Bool) (ω : Ω) (t : (Fin (V.k x) → Coin) × A) : Trace :=
  if keep ω t.1 t.2 then run H P V x ω t.1 else abortTrace

/-- **`Pr[run = acc ∧ ¬R(x, E^rep(x))]` at a prover that may abort**, over the tape, the judged run's coins with its
abort coins, and `E^rep`'s draws, each a run's coins with its abort coins: `E^rep` at the inner protocol's waits reruns
at the judged draw, then at its draws, and sees each run as it aborts. -/
noncomputable def Inner.PrBadAbort {Coin W : Type} [Fintype Coin] [Inhabited Coin] (H : Msg → Msg)
    (I : Inner Coin W) (x : Msg) {Ω A : Type} [Fintype Ω] [Fintype A] (P : Prover Ω)
    (keep : Ω → (Fin (I.V.k x) → Coin) → A → Bool) : ℝ :=
  Pr fun s : Ω × ((Fin (I.V.k x) → Coin) × A) × (Fin (I.waits x) → Fin (I.cap x) → (Fin (I.V.k x) → Coin) × A) =>
    (abortRun H P I.V x keep s.1 s.2.1).out = .acc ∧
      ¬ I.R.ext x ((erep (I.waits x) (I.k x) (I.cap x) I.opens (I.assemble x) (I.check x) s.2.1 s.2.2).eval H
        (abortRun H P I.V x keep s.1)).1

/-- **`B` at runs that may abort**: `B`'s fork (guess `j`, draw `ω, r_1..r_j`, run twice from the shared start with
fresh continuations), each run with its own abort coins, and the first collision among the verifier's SHA-512 calls in
the two runs as they abort. Its cost is both runs' calls, both parties'. -/
def BAbort {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {Ω A : Type} [Fintype Ω] [Nonempty Ω]
    [Fintype A] [Nonempty A] (H : Msg → Msg) (V : Verifier Coin) (P : Prover Ω) (x : Msg)
    (keep : Ω → (Fin (V.k x) → Coin) → A → Bool) (hk : 0 < V.k x) : Finder where
  S := BCoins Ω Coin (V.k x) × A × A
  ne := ⟨(⟨⟨0, hk⟩, Classical.arbitrary _, fun _ => default, fun _ => default⟩, Classical.arbitrary _,
    Classical.arbitrary _)⟩
  out s := findColl ((abortRun H P V x keep s.1.2.1 (s.1.2.2.1, s.2.1)).vLog ++
    (abortRun H P V x keep s.1.2.1 (forkCoins s.1.1 s.1.2.2.1 s.1.2.2.2, s.2.2)).vLog)
  cost s :=
    (abortRun H P V x keep s.1.2.1 (s.1.2.2.1, s.2.1)).pLog.length +
      (abortRun H P V x keep s.1.2.1 (s.1.2.2.1, s.2.1)).vLog.length +
      (abortRun H P V x keep s.1.2.1 (forkCoins s.1.1 s.1.2.2.1 s.1.2.2.2, s.2.2)).pLog.length +
      (abortRun H P V x keep s.1.2.1 (forkCoins s.1.1 s.1.2.2.1 s.1.2.2.2, s.2.2)).vLog.length

/-- `Pr[B^{P*}(x) outputs a collision]` at runs that may abort, `0` for a verifier that takes no rounds at `x`. -/
noncomputable def PrBAbort {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {Ω A : Type} [Fintype Ω]
    [Nonempty Ω] [Fintype A] [Nonempty A] (H : Msg → Msg) (V : Verifier Coin) (P : Prover Ω) (x : Msg)
    (keep : Ω → (Fin (V.k x) → Coin) → A → Bool) : ℝ :=
  if hk : 0 < V.k x then (BAbort H V P x keep hk).Pr H else 0

/-- **The inner protocol's KS against provers that may abort**, the inner claim the recursion's proof reads: for every
prover `P*` (any finite tape) and every way it may abort after the inner rounds (`keep`, on any finite fresh coins `A`
per run), `Pr[run = acc ∧ ¬R(x, E^rep(x))] ≤ κ(x) + √(γ(x) · Pr[B^{P*}(x) outputs a collision])`, `E^rep` and `B`
seeing each run as it aborts. With no aborts it is `Inner.KS`. -/
def Inner.KSAbort {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg)
    (I : Inner Coin W) : Prop :=
  ∀ (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (A : Type) [Fintype A] [Nonempty A]
    (keep : Ω → (Fin (I.V.k x) → Coin) → A → Bool),
    I.PrBadAbort H x P keep ≤ I.κ x + Real.sqrt (I.γ x * PrBAbort H I.V P x keep)

section OfFlock

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

/-- **Flock as the inner protocol**, at a decoder: its verifier, relation, scope, `E^rep` and errors. -/
noncomputable def Inner.ofFlock (D : Flock.Deploy Coin Col T) (decode : List (ℕ × Col) → List T) :
    Inner Coin Flock.Wit where
  V := D.V
  R := Flock.R D.parse
  Scope := Flock.Scope D
  Op := ℕ → List (ℕ × Col)
  opens v := [D.cols v]
  assemble := Flock.assemble D.parse D.table decode D.read
  check := Flock.check D.parse
  waits := D.waits
  k := D.k
  cap := D.cap
  κ := Flock.κ D
  γ := Flock.γ D

/-- The decoders a claim quantifies over: those that meet Flock's list-decoding spec, within the deployment's radius
at each trace's codeword length (`Flock.ListWithin`). -/
def decOK [Zero Col] (D : Flock.Deploy Coin Col T) (decode : List (ℕ × Col) → List T) : Prop :=
  Flock.ListWithin D.len D.enc D.θ decode

end OfFlock

/-! ## The compiled verifier's statement and scope -/

/-- **The recursion's form**, as the compiled verifier reads it (its inner shape, `RecVStar.Inner`): the inner table's
`m`, `k_log`, regions and claims, whether the inner circuits are private (`holo`), the RecSparse phase's shape
`(NU, ELL, S_LOG, T_LOG)` and R's shape `(m, claims)` (candidate 3's sparse phase; neither at a dense or holo inner
session), and whether the RecSparse unit range's slots lie inside the block. -/
structure Form where
  m : ℕ
  kLog : ℕ
  regions : ℕ
  claims : ℕ
  holo : Bool
  sparse : Option (ℕ × ℕ × ℕ × ℕ)
  stacked : Option (ℕ × ℕ)
  unitRangeInBlock : Bool

/-- **A V* session's draw**, as its record holds it: its law's `k` when the law is `subset` (`none` for another law),
its population, its units and its closure. -/
structure Draw where
  subset : Option ℕ
  population : ℕ
  units : List ℕ
  closure : List ℕ

/-- **The full draw**: the `subset` draw of every unit of its population, `0, …, N − 1` in order, with no closure. -/
def Draw.Full (d : Draw) : Prop :=
  d.subset = some d.population ∧ d.units = List.range d.population ∧ d.closure = []

/-- **The recursion's statement**: the inner statement's bytes, its `(m, k_log, regions)` as the verifier sets it up
from its files (`none` where the verifier holds none), the commitment to the circuit (the private-circuit variant),
the form, and each V* session's draw as its record holds it (`none` where it holds none the verifier reads; no draws
where the records are run messages rather than the statement's). -/
structure Stmt where
  inner : Msg
  innerDims : Option (ℕ × ℕ × ℕ)
  circ : Option Msg
  form : Form
  draws : List (Option Draw)

/-- RecSparse's `P_LOG` at `ℓ`: the bit length of `2ℓ + 1`. -/
def pLog (ell : ℕ) : ℕ := Nat.log 2 (2 * ell + 1) + 1

/-- **The recursion's scope, readable**: the inner shapes the compiled verifier accepts, every V* session drawing every
unit of its population. No circuit commitment; the inner table at a fast100 `m` (`22 ≤ m ≤ 35`), `k_log ≥ 6`, two
claims and two per region, its circuits public or private (`holo` either way); and either no RecSparse phase and no R
(a dense or holo inner session), or candidate 3's sparse phase: a RecSparse shape at `NU ≥ 1`, `6 ≤ ℓ ≤ k_log`,
`T_LOG = 0`, the inner statement's `(m, k_log, regions)` the form's, its unit range's slots inside the block, and R at
the phase's `table_m = max(22, NU + P_LOG)` with a fast100 schedule and the two reps' claims. Every V* session's
record holds a draw, and it is the full draw (`Draw.Full`). -/
def InScope (s : Stmt) : Prop :=
  s.circ = none ∧ 22 ≤ s.form.m ∧ s.form.m ≤ 35 ∧ 6 ≤ s.form.kLog ∧ s.form.claims = 2 + 2 * s.form.regions ∧
    (s.form.sparse = none ∧ s.form.stacked = none ∨
      ∃ nu ell sLog, s.form.sparse = some (nu, ell, sLog, 0) ∧ 1 ≤ nu ∧ 6 ≤ ell ∧ ell ≤ s.form.kLog ∧
        s.innerDims = some (s.form.m, s.form.kLog, s.form.regions) ∧ s.form.unitRangeInBlock = true ∧
        s.form.stacked = some (max 22 (nu + pLog ell), 2) ∧ max 22 (nu + pLog ell) ≤ 35) ∧
    ∀ d ∈ s.draws, ∃ e, d = some e ∧ e.Full

/-! ## What the compiled verifier brings -/

/-- **An outer session's numbers**, those Flock's errors read at one V* statement (`Flock.Deploy`'s at it): the
table's error, `N₀`, the weight `C`, the outer verifier's rounds, `E^rep`'s waits and its truncation; and those its
`--zk` simulation error reads (`Flock.zk_of_models`' `n_L` and `n_B`): the hm96 leaves of the masked prover's
messages before the coins (`leaves`), and of its answers to one draw of the coins (`coinLeaves`). -/
structure Nums where
  εtable : ℝ
  N₀ : ℝ
  weight : ℝ
  rounds : ℕ
  waits : ℕ
  cap : ℕ
  leaves : ℕ
  coinLeaves : ℕ

/-- `E^rep`'s accepted reruns per wait, `k = ⌈4·N₀/e⌉`, as `Flock.Deploy.k`. -/
noncomputable def Nums.k (n : Nums) : ℕ := ⌈4 * n.N₀ / Real.exp 1⌉₊

/-- The outer session's knowledge error, as `Flock.κ`: `4·ε_table + 2^{−m′} + 2k/K + e^{−k/8}`. -/
noncomputable def Nums.κ (n : Nums) : ℝ :=
  4 * n.εtable + 2 ^ (-(n.waits : ℝ)) + 2 * n.k / n.cap + Real.exp (-(n.k : ℝ) / 8)

/-- The outer session's `γ`, as `Flock.γ`: `16·(√(2·k·C) + k)²` at its rounds `k`. -/
noncomputable def Nums.γ (n : Nums) : ℝ := 16 * (Real.sqrt (2 * n.rounds * n.weight) + n.rounds) ^ 2

/-- **The outer session's simulation error** at the attempt budget `t`, where hm96 leaves hide up to `δ₁`, as Flock's
`--zk` zero knowledge proves it (`Flock.zk_of_models`, at `S_GK = G.simulatorE t (t·3^t)`):
`8·n_L·δ₁ + 8t·n_B·δ₁ + 2·3^{−t}`. -/
noncomputable def Nums.εzk (n : Nums) (t : ℕ) (δ₁ : ℝ) : ℝ :=
  8 * n.leaves * δ₁ + 8 * t * n.coinLeaves * δ₁ + 2 * (1 / 3) ^ t

/-- **What the compiled verifier brings**, read by the recursion's claims: the verifier (public-coin: each prover
message answered by its live coin, the decision the shipped verdict), its parser and scope check; the prover's staging
rounds after the inner session's (`kS`), the hm96 commit strings each staging message holds (`nC`), and the staging
message at the run so far from its commit strings (`stage`, which reads only the statement, the public transcript and
those commit strings: the honest prover's are its staged values' under its own salts, the simulator's the zero
digest's under its salts, `Deploy.zeroCommit`); the V* statements of a run, from the public transcript of the
inner session and the staging rounds (`vstar`), and their count at `x` (`nS`, fixed by `x`'s form); outer session `s`'s
view in a run's view (`session`: its messages and its verifier's decision as the compiled verifier's answers show it);
the inner session's view read off a run's coins and the outer witnesses, at an oracle (`innerView`: the inner verifier
runs at it); the SHA-512 evaluations the outer witnesses record (`calls`); the numbers of outer session `s` at `x`
(`nums`, the outer Flock's at that session's V* statements, which their public shape fixes); for the simulator, the
pinned hm96 key's salts (`saltLen` bytes), its salt prefix `p_salt` and its GF(2)-linear salt mask `Φ`, into 64 bytes
(`mask`); the reference prover (`P`), the honest prover the deployment runs, on its tapes (`Tape`, a nonempty
type); and its completeness error as the deployment states it, at each hash (`δc H x` at `H`): `Complete` is at that
error and no other. -/
structure Deploy (Coin : Type) where
  V : Verifier Coin
  parse : Msg → Option Stmt
  scopeCheck : Msg → Bool
  kS : Msg → ℕ
  nC : Msg → ℕ
  stage : Msg → List (Msg × Out) → List Msg → OC Msg
  vstar : Msg → List (Msg × Out) → List Msg
  nS : Msg → ℕ
  session : Msg → ℕ → Seen → Seen
  innerView : (Msg → Msg) → (x : Msg) → (Fin (V.k x) → Coin) → List Flock.Wit → Seen
  calls : Msg → List Flock.Wit → List (Msg × Msg)
  nums : Msg → ℕ → Nums
  saltLen : ℕ
  saltPrefix : Msg
  mask : (Fin saltLen → UInt8) → Fin 64 → UInt8
  Tape : Type
  [tapeNonempty : Nonempty Tape]
  P : Flock.Wit → Prover Tape
  δc : (Msg → Msg) → Msg → ℝ

instance Deploy.instNonemptyTape {Coin : Type} (D : Deploy Coin) : Nonempty D.Tape := D.tapeNonempty

/-- **A salt's hash** at the pinned key, `c = H(p_salt ‖ y)`. -/
def Deploy.saltHash {Coin : Type} (D : Deploy Coin) (H : Msg → Msg) (y : Fin D.saltLen → UInt8) : Msg :=
  H (D.saltPrefix ++ List.ofFn y)

/-- **An inner round's commit string to the zero digest** under the salt `y`, hm96's `b ‖ c` at the pinned key:
`b = 0 ⊕ Φy` and `c = H(p_salt ‖ y)`, the oracle's answer. -/
def Deploy.zeroCommit {Coin : Type} (D : Deploy Coin) (y : Fin D.saltLen → UInt8) : OC Msg :=
  .query (D.saltPrefix ++ List.ofFn y) fun c => .ret (List.ofFn (D.mask y) ++ c)

/-- **Commit strings to the zero digest** under the salts `ys`, in order. -/
def Deploy.zeroCommits {Coin : Type} (D : Deploy Coin) : List (Fin D.saltLen → UInt8) → SP (List Msg)
  | [] => .ret []
  | y :: ys => spBind (spOf (D.zeroCommit y)) fun c => spBind (D.zeroCommits ys) fun cs => .ret (c :: cs)

/-- **A record with its reference prover's public stop, finite tapes and inner commit strings.**

* The reference prover's tape at each statement `x` is drawn from a finite nonce and rest of `x`'s (`tapeOf x`), since
  each V* session runs on a tape of its own; zero knowledge is of `P` on that tape (`StopDeploy.prover`), at the outer
  `--zk` Flock's simulators at each attempt budget, at the recursion's statement and each V* session's index (`SO`).
* The reference prover's inner round `i` at `x` carries `nAt x i` hm96 commit strings under fresh salts of its tape,
  and its message is `innerMsg x ms cs` at the run so far `ms` and those strings `cs` (the public rest of the round, as
  `stage` is the staging round's).
* `stops H x ms` says whether the reference prover stops at the statement `x` against `H` once the run has shown `ms`
  (the prover's messages and the verifier's answers so far, in order; `[]` before the first round). It is read before
  each inner and staging round and after the last of them (`halted`); from there the prover sends the empty message
  every round. It reads the hash, the statement and what the verifier has seen, never the witness or a tape, so a
  simulator reads it off its own run. At the deployment it is where the honest proof can be refused: a statement out
  of scope or whose inner statement the inner verifier can't accept (read off the statement alone), an inner coin in
  the inner completeness error (Flock's: a zerocheck outer coin at `1`), or staged V* statements that the masked
  verifier can't accept or whose numbers are not the statement's (where the compiled verifier refuses). -/
structure StopDeploy (Coin : Type) extends Deploy Coin where
  Nonce : Msg → Type
  Rest : Msg → Type
  [nonceFin : ∀ x, Fintype (Nonce x)]
  [nonceNe : ∀ x, Nonempty (Nonce x)]
  [restFin : ∀ x, Fintype (Rest x)]
  [restNe : ∀ x, Nonempty (Rest x)]
  tapeOf : (x : Msg) → Nonce x × Rest x → Tape
  SO : ℕ → Msg → ℕ → OSim
  nAt : Msg → ℕ → ℕ
  innerMsg : Msg → List (Msg × Out) → List Msg → OC Msg
  stops : (Msg → Msg) → Msg → List (Msg × Out) → Bool

attribute [instance] StopDeploy.nonceFin StopDeploy.nonceNe StopDeploy.restFin StopDeploy.restNe

/-- **The record's reference prover at the statement `x`**: `Deploy.P`, the prover `Complete` reads, on the tape `x`'s
nonce and rest draw. -/
def StopDeploy.prover {Coin : Type} (D : StopDeploy Coin) (w : Flock.Wit) (x : Msg) : Prover (D.Nonce x × D.Rest x) :=
  ⟨fun x' hist ω => (D.P w).next x' hist (D.tapeOf x ω)⟩

/-- **The prover has stopped** after the transcript `ms`, its stop read at the first `n` rounds' starts and after the
`n`-th: `stops` holds at the transcript before some round `i ≤ n` that `ms` has reached. -/
def StopDeploy.halted {Coin : Type} (D : StopDeploy Coin) (H : Msg → Msg) (n : ℕ) (x : Msg) (ms : List (Msg × Out)) :
    Bool :=
  (List.range (min ms.length n + 1)).any fun i => D.stops H x (ms.take i)

section Recursion

variable {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

/-- The inner statement's bytes. -/
def xI (D : Deploy Coin) (x : Msg) : Msg := ((D.parse x).map Stmt.inner).getD []

/-- **The relation, inherited**: `x` parses, with no circuit commitment, and the inner relation holds at its inner
statement. -/
def R (D : Deploy Coin) (I : Inner Coin W) : Rel W := fun x w =>
  ∃ s, D.parse x = some s ∧ s.circ = none ∧ I.R s.inner w

/-- **The private-circuit variant's relation**: `x` parses with a circuit commitment, and the inner relation holds at
its inner statement. The commitment is hm96 rows that the inner (universal unit's) circuit outputs, which `parse` reads
off the inner statement's public outputs, so the inner relation binds it to the circuit the inner witness carries. -/
def RPriv (D : Deploy Coin) (I : Inner Coin W) : Rel W := fun x w =>
  ∃ s, D.parse x = some s ∧ s.circ.isSome = true ∧ I.R s.inner w

/-- **The recursion's scope** on the statement's bytes: it parses, its inner statement is in the inner scope, and its
form is in `InScope`. -/
def Scope (D : Deploy Coin) (I : Inner Coin W) (x : Msg) : Prop :=
  ∃ s, D.parse x = some s ∧ I.Scope s.inner ∧ InScope s

/-! ### The extractor -/

/-- **The outer extractor `E_O`**: replay the judged run; if it accepted, for each V* session `s` in order, at its
statement (read off the judged run's inner and staging rounds), `E^rep` with the outer Flock's assemble and check over
forks of the judged run at the session's first round, each rerun seen as the session's view; all the sessions'
witnesses, or none. -/
noncomputable def EO (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) : Extractor Coin (List Flock.Wit) D.V where
  C x := (s : Fin (D.nS x)) → Fin (D.nums x s).waits → Fin (D.nums x s).cap → Fin (D.V.k x) → Coin
  ext x r₀ c := .rerun r₀ fun v₀ =>
    if v₀.out = .acc then
      let xs := D.vstar x (v₀.msgs.take (I.V.k (xI D x) + D.kS x))
      collect ((List.finRange (D.nS x)).map fun (s : Fin (D.nS x)) =>
        serve (viewing (D.session x s))
          (firstWait (D.nums x s).k (fun v => [DO.cols v])
            (Flock.assemble DO.parse DO.table decO DO.read (xs.getD s []))
            (Flock.check DO.parse (xs.getD s []))
            ((List.finRange (D.nums x s).waits).map fun i => (List.finRange (D.nums x s).cap).map fun j =>
              fork (I.V.k (xI D x) + D.kS x + ((xs.take s).map DO.V.k).sum) r₀ (c s i j))))
    else .ret none

/-- The inner session's view at a draw, read off the outer extractor's output with the inner verifier at the oracle
`H`; a rejected view when it gave up. -/
def dec (H : Msg → Msg) (D : Deploy Coin) (x : Msg) (r : Fin (D.V.k x) → Coin) : Option (List Flock.Wit) → Seen
  | some ws => D.innerView H x r ws
  | none => ⟨[], .rej⟩

/-- **`E*`, the recursion's extractor** at the oracle `H`: the outer extractor, then the inner one. The inner
protocol's `E^rep` at its inner statement, each of its reruns served by `E_O` at that draw's coins and `E_O`'s own, the
inner view read off `E_O`'s output (`dec`). -/
noncomputable def E (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) : Extractor Coin W D.V :=
  served D.V (EO D I DO decO) (dec H D) (fun x => I.waits (xI D x)) (fun x => I.k (xI D x))
    (fun x => I.cap (xI D x)) I.opens (fun x => I.assemble (xI D x)) (fun x => I.check (xI D x))

/-! ### The extraction-based finder -/

/-- **`Be`** (extraction-based): `B`'s fork (guess `j`, draw `ω, r_1..r_j`, run twice from the shared start with fresh
continuations), `E_O` on each run with fresh coins, and the first collision among the SHA-512 evaluations the two
runs' outer witnesses record (`Deploy.calls`: the inner verifier's and the commitments', which no party computes in the
clear). Its cost is both runs' calls, both parties', and both extractions'. -/
noncomputable def Be (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg)
    (hk : 0 < D.V.k x) : Finder where
  S := BCoins Ω Coin (D.V.k x) × (EO D I DO decO).C x × (EO D I DO decO).C x
  ne := ⟨(⟨⟨0, hk⟩, Classical.arbitrary _, fun _ => default, fun _ => default⟩, Classical.arbitrary _,
    Classical.arbitrary _)⟩
  out s :=
    findColl (D.calls x (((EO D I DO decO).eval H P x s.1.2.1 s.1.2.2.1 s.2.1).1.getD []) ++
      D.calls x (((EO D I DO decO).eval H P x s.1.2.1 (forkCoins s.1.1 s.1.2.2.1 s.1.2.2.2) s.2.2).1.getD []))
  cost s :=
    (run H P D.V x s.1.2.1 s.1.2.2.1).pLog.length + (run H P D.V x s.1.2.1 s.1.2.2.1).vLog.length +
      (run H P D.V x s.1.2.1 (forkCoins s.1.1 s.1.2.2.1 s.1.2.2.2)).pLog.length +
      (run H P D.V x s.1.2.1 (forkCoins s.1.1 s.1.2.2.1 s.1.2.2.2)).vLog.length +
      ((EO D I DO decO).eval H P x s.1.2.1 s.1.2.2.1 s.2.1).2.1 +
      ((EO D I DO decO).eval H P x s.1.2.1 (forkCoins s.1.1 s.1.2.2.1 s.1.2.2.2) s.2.2).2.1

/-- `Pr[Be^{P*}(x) outputs a collision]`, `0` for a verifier that takes no rounds at `x`. -/
noncomputable def PrBe (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg) : ℝ :=
  if hk : 0 < D.V.k x then (Be H D I DO decO P x hk).Pr H else 0

/-! ### The errors -/

/-- The inner `E^rep`'s expected reruns, `N_I = 1 + m′_I·k_I`: each is served by `E_O`, so each can meet its failure. -/
def NI (D : Deploy Coin) (I : Inner Coin W) (x : Msg) : ℕ := 1 + I.waits (xI D x) * I.k (xI D x)

/-- **The recursion's knowledge error**: `κ = κ_I(x_I) + N_I · Σ_s κ_O,s`. -/
noncomputable def κ (D : Deploy Coin) (I : Inner Coin W) (x : Msg) : ℝ :=
  I.κ (xI D x) + NI D I x * ∑ s : Fin (D.nS x), (D.nums x s).κ

/-- **`B`'s weight**: `γ = (N_I · Σ_s √(γ_O,s · k(x)))²`, with `k(x)` the compiled verifier's rounds (a session's fork
is one of `B`'s guesses): each of the inner `E^rep`'s `N_I` expected reruns meets each session's outer `B`. -/
noncomputable def γ (D : Deploy Coin) (I : Inner Coin W) (x : Msg) : ℝ :=
  (NI D I x * ∑ s : Fin (D.nS x), Real.sqrt ((D.nums x s).γ * D.V.k x)) ^ 2

/-- **`Be`'s weight**: `γ_e = (√(γ_I(x_I) · k(x)) + N_I · √(k_I(x_I) · k(x)))²`: the inner `B`'s term, whose collisions
are among calls only the outer witnesses record, and the inner rounds' differing openings at each of the inner
`E^rep`'s `N_I` expected reruns. -/
noncomputable def γe (D : Deploy Coin) (I : Inner Coin W) (x : Msg) : ℝ :=
  (Real.sqrt (I.γ (xI D x) * D.V.k x) + NI D I x * Real.sqrt (I.V.k (xI D x) * D.V.k x)) ^ 2

/-- `Be`'s expected SHA-512 calls per call of the prover's run: two runs and two extractions, `E_O`'s at most
`1 + Σ_s m′_s·k_s` reruns in expectation. -/
noncomputable def cBe (D : Deploy Coin) (x : Msg) : ℕ :=
  4 + 2 * ∑ s : Fin (D.nS x), (D.nums x s).waits * (D.nums x s).k

/-- **The recursion's bits**: `λ = min(log₂(1/κ), 255.5 − ½ log₂ γ, 128 − ½ log₂(γ_e · c_Be)) − 2`. -/
noncomputable def lam (D : Deploy Coin) (I : Inner Coin W) (x : Msg) : ℝ :=
  min (min (Real.logb 2 (1 / κ D I x)) (255.5 - Real.logb 2 (γ D I x) / 2))
    (128 - Real.logb 2 (γe D I x * cBe D x) / 2) - 2

/-! ### The simulator -/

/-- The inner session simulated, its round `j` the commit string to the zero digest under the salt `salts j`. -/
def simInner (D : Deploy Coin) (n : ℕ) (salts : Fin n → Fin D.saltLen → UInt8) : SP (List (Msg × Out)) :=
  simRounds (fun ms => if h : ms.length < n then spOf (D.zeroCommit (salts ⟨ms.length, h⟩)) else .ret []) n []

/-- The V* sessions simulated in order, each by the outer simulator at its statement, while the run goes on. -/
def simOuter {n : ℕ} (SO : Msg → ℕ → OSim) (x : Msg) (xs : List Msg) (c : (s : Fin n) → (SO x s).C) :
    List (Fin n) → List (Msg × Out) → SP (List (Msg × Out))
  | [], ms => .ret ms
  | s :: ss, ms =>
    if goesOn ms then
      spBind (spUnder (ms.map Prod.fst) ((SO x s).prog (xs.getD s []) (c s))) fun ts =>
        simOuter SO x xs c ss (ms ++ ts)
    else .ret ms

/-- The staging rounds simulated after the `n` inner rounds: staging round `j`'s message at the run so far, from its
commit strings to the zero digest under the salts `salts j`. -/
def simStage (D : Deploy Coin) (x : Msg) (n : ℕ) (salts : Fin (D.kS x) → Fin (D.nC x) → Fin D.saltLen → UInt8)
    (ms : List (Msg × Out)) : SP (List (Msg × Out)) :=
  simRounds (fun ms => if h : ms.length - n < D.kS x then
      spBind (D.zeroCommits ((List.finRange (D.nC x)).map (salts ⟨ms.length - n, h⟩))) fun cs => spOf (D.stage x ms cs)
    else .ret []) (D.kS x) ms

/-- **The recursion's simulator**: the inner session's commit strings to the zero digest under fresh salts, the staging
messages with their commit strings to the zero digest under fresh salts, then each V* session by the outer simulator
`SO` (the named Flock simulator) at its statement. -/
def Sim (D : Deploy Coin) (I : Inner Coin W) (SO : Msg → ℕ → OSim) : Simulator where
  C x := (Fin (I.V.k (xI D x)) → Fin D.saltLen → UInt8) × (Fin (D.kS x) → Fin (D.nC x) → Fin D.saltLen → UInt8) ×
    ((s : Fin (D.nS x)) → (SO x s).C)
  prog x c :=
    spBind (simInner D (I.V.k (xI D x)) c.1) fun ms =>
      spBind (simStage D x (I.V.k (xI D x)) c.2.1 ms) fun ms =>
        simOuter SO x (D.vstar x ms) c.2.2 (List.finRange (D.nS x)) ms

/-- **`ZK`'s error** at the attempt budget `t`, where hm96 leaves hide up to `δ₁`: each V* session's `--zk` simulation
error (`Nums.εzk`) and `2δ₁` for each commit string `Sim` commits to zero, the inner session's `k_I(x_I)` and the
staging messages' `k_S(x) · n_C(x)`. -/
noncomputable def εzk (D : Deploy Coin) (I : Inner Coin W) (t : ℕ) (δ₁ : ℝ) (x : Msg) : ℝ :=
  ∑ s : Fin (D.nS x), (D.nums x s).εzk t δ₁ + 2 * (I.V.k (xI D x) + D.kS x * D.nC x) * δ₁

/-- **`ZK`'s collision weight** at the attempt budget `t`: Flock's `4t·3^t` at each of the `n_S(x)` V* sessions. -/
noncomputable def γzk (D : Deploy Coin) (t : ℕ) (x : Msg) : ℝ := D.nS x * (4 * t * 3 ^ t)

/-! ### The claims -/

/-- **The recursion: KS**, at `E*` and every outer decoder that meets the list-decoding spec: on every prover,
`Pr[run = acc ∧ ¬R(x, E*(x))] ≤ κ(x) + √(γ(x) · Pr[B]) + √(γ_e(x) · Pr[Be])`. -/
def KS [Zero Col] (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T) : Prop :=
  ∀ decO, decOK DO decO → ∀ (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω),
    PrBad H D.V (R D I) (E H D I DO decO) x P ≤
      κ D I x + Real.sqrt (γ D I x * PrB H D.V P x) + Real.sqrt (γe D I x * PrBe H D I DO decO P x)

/-- **The recursion: bits**, at every outer decoder that meets the spec: every prover making at most `t ≥ 1` SHA-512
calls per run (both parties') has `Pr[acc ∧ ¬R] ≤ t · 2^{−λ(x)}`, where `B` obeys the strict birthday bound and `Be`
the linear expected-time bound. -/
def Bits [Zero Col] (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T) : Prop :=
  ∀ decO, decOK DO decO → ∀ (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (t : ℕ), 1 ≤ t →
    (∀ ω r, (run H P D.V x ω r).pLog.length + (run H P D.V x ω r).vLog.length ≤ t) →
    (∀ hk : 0 < D.V.k x, StrictBirthday H (B H D.V P x hk)) →
    (∀ hk : 0 < D.V.k x, ExpectedCR H (Be H D I DO decO P x hk)) →
    PrBad H D.V (R D I) (E H D I DO decO) x P ≤ t * (2 : ℝ) ^ (-lam D I x)

/-- **The recursion: cost**: `E*` reruns the prover at most `(1 + m′_I·K_I) · (1 + Σ_s m′_s·K_s)` times, and
`N_I · (1 + Σ_s m′_s·k_s)` in expectation. The decoders' complexity is Flock's (`Flock.Efficient`). -/
def Efficient (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) : Prop :=
  ExtractRunsLe H (E H D I DO decO) (fun x => (1 + I.waits (xI D x) * I.cap (xI D x)) *
    (1 + ∑ s : Fin (D.nS x), (D.nums x s).waits * (D.nums x s).cap)) ∧
  ExtractRunsExpLe H (E H D I DO decO) (fun x => (NI D I x : ℝ) *
    (1 + ∑ s : Fin (D.nS x), ((D.nums x s).waits * (D.nums x s).k : ℕ)))

/-- **The recursion: Complete**, on the statements in scope, of the deployed record's reference prover (`D.P`, on its
nonempty tapes), at an inner protocol of Flock's witnesses, at the deployed error (`D.δc H`). -/
def Complete (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin Flock.Wit) : Prop :=
  Claims.Complete H D.V D.P (R D I) (Scope D I) (D.δc H)

/-- **The recursion: the readable scope is the compiled verifier's**: a statement is in `Scope` exactly when the
compiled verifier's scope check takes it. -/
def ScopeMatches (D : Deploy Coin) (I : Inner Coin W) : Prop := Claims.ScopeMatches (Scope D I) D.scopeCheck

/-- **The recursion: zero knowledge** of the honest prover against every malicious verifier of the compiled verifier's
rounds (`ZKmalR` at `D.V.k`), at `Sim` over the outer `--zk` Flock's simulators at each attempt budget `t ≥ 1`
(`SO t`): wherever hm96 leaves hide up to `δ₁` at the deployed key (`Hm96Hiding` at its salt mask `D.mask` and salt
hash `D.saltHash H`), within `εzk` (each V* session's proved `--zk` simulation error, and `2δ₁` per commit string of
the inner session and the staging messages) plus `γzk` times the probability that `Fork` finds a collision among the
honest prover's SHA-512 calls in two real runs against `V*`. -/
def ZK {N Ω : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] (H : Msg → Msg) (D : Deploy Coin)
    (I : Inner Coin W) (DO : Flock.Deploy Coin Col T) (P : W → Prover (N × Ω)) (SO : ℕ → Msg → ℕ → OSim) : Prop :=
  DO.zk = true ∧ ∀ t, 0 < t → ∀ δ₁ : ℝ≥0, Hm96Hiding D.mask (D.saltHash H) δ₁ →
    ZKmalR H D.V.k P (R D I) (Sim D I (SO t)) (εzk D I t δ₁) (γzk D t)

/-- **The recursion: nontriviality**: recorded honest runs, each of a statement the scope check takes, that the compiled
verifier accepts. -/
def Nontrivial (H : Msg → Msg) (D : Deploy Coin) (recs : List (Recorded D.V)) : Prop :=
  Claims.Nontrivial H D.V recs ∧ ∀ e ∈ recs, D.scopeCheck e.x = true

/-- **The private variant: KS**, at `E*` and the same errors. -/
def KSPriv [Zero Col] (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T) : Prop :=
  ∀ decO, decOK DO decO → ∀ (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω),
    PrBad H D.V (RPriv D I) (E H D I DO decO) x P ≤
      κ D I x + Real.sqrt (γ D I x * PrB H D.V P x) + Real.sqrt (γe D I x * PrBe H D I DO decO P x)

/-- **The private variant: zero knowledge**, at the same simulator, errors and hiding as `ZK`: `Sim` commits to the
zero digest at every inner round and in every staging message, so it needs no circuit of the class. -/
def ZKPriv {N Ω : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] (H : Msg → Msg) (D : Deploy Coin)
    (I : Inner Coin W) (DO : Flock.Deploy Coin Col T) (P : W → Prover (N × Ω)) (SO : ℕ → Msg → ℕ → OSim) : Prop :=
  DO.zk = true ∧ ∀ t, 0 < t → ∀ δ₁ : ℝ≥0, Hm96Hiding D.mask (D.saltHash H) δ₁ →
    ZKmalR H D.V.k P (RPriv D I) (Sim D I (SO t)) (εzk D I t δ₁) (γzk D t)

/-! ### Zero knowledge with the prover's public stop (restated, statement only) -/

/-- **The recursion's simulator with the prover's stop** against `H`: the reference prover's rounds, each commit string
to the zero digest under a fresh salt, while the prover hasn't stopped (`halted`, read before each inner and staging
round and after the last); from there the empty message every round left, as the prover sends.

* Inner round `i`: `nAt x i` commit strings, as many as the reference prover's round carries, in the round's public
  layout (`innerMsg`).
* Staging round `j`: the staging message (`stage`) at `n_C(x)` commit strings.
* Then each V* session by the outer simulator `SO` (the named Flock simulator) at its statement. -/
def SimStop (D : StopDeploy Coin) (I : Inner Coin W) (SO : Msg → ℕ → OSim) (H : Msg → Msg) : Simulator where
  C x := ((i : Fin (I.V.k (xI D.toDeploy x))) → Fin (D.nAt x i) → Fin D.saltLen → UInt8) ×
    (Fin (D.kS x) → Fin (D.nC x) → Fin D.saltLen → UInt8) × ((s : Fin (D.nS x)) → (SO x s).C)
  prog x c :=
    let n := I.V.k (xI D.toDeploy x)
    let m := n + D.kS x
    spBind (simRounds (fun ms => if D.halted H m x ms then .ret [] else
        if h : ms.length < n then
          spBind (D.toDeploy.zeroCommits
              ((List.finRange (D.nAt x (⟨ms.length, h⟩ : Fin n))).map (c.1 ⟨ms.length, h⟩))) fun cs =>
            spOf (D.innerMsg x ms cs)
        else .ret []) n []) fun ms =>
      spBind (simRounds (fun ms => if D.halted H m x ms then .ret [] else
          if h : ms.length - n < D.kS x then
            spBind (D.toDeploy.zeroCommits ((List.finRange (D.nC x)).map (c.2.1 ⟨ms.length - n, h⟩))) fun cs =>
              spOf (D.stage x ms cs)
          else .ret []) (D.kS x) ms) fun ms =>
        if D.halted H m x ms then simRounds (fun _ => .ret []) (D.V.k x - m) ms
        else simOuter SO x (D.vstar x ms) c.2.2 (List.finRange (D.nS x)) ms

/-- **`ZKStop`'s error** at the attempt budget `t`, where hm96 leaves hide up to `δ₁`: each V* session's `--zk`
simulation error (`Nums.εzk`) and `2δ₁` for each commit string `SimStop` commits to zero, the inner rounds'
`Σ_i nAt(x, i)` and the staging messages' `k_S(x) · n_C(x)`. -/
noncomputable def εzkStop (D : StopDeploy Coin) (I : Inner Coin W) (t : ℕ) (δ₁ : ℝ) (x : Msg) : ℝ :=
  ∑ s : Fin (D.nS x), (D.nums x s).εzk t δ₁ +
    2 * (∑ i : Fin (I.V.k (xI D.toDeploy x)), D.nAt x i + D.kS x * D.nC x) * δ₁

/-- **The recursion: zero knowledge with the prover's public stop**, `ZK` restated: of the record's own reference
prover on the tape each statement's nonce and rest draw (`StopDeploy.prover`: `Deploy.P`, which `Complete` reads),
which stops where `stops` says, against every malicious verifier of the compiled verifier's rounds (`ZKmalRx` at
`D.V.k`), at `SimStop` over the record's outer `--zk` Flock simulators at each attempt budget `t ≥ 1` (`D.SO t`), which
stops there too: wherever hm96 leaves hide up to `δ₁` at the deployed key, within `εzkStop` plus `γzk` times the
probability that `Fork` finds a collision among the prover's SHA-512 calls in two real runs against `V*`.

A malicious verifier may send inner coins in the inner completeness error, where whether the honest inner proof is
refused reads the inner messages, and the V* statements staged after a refused one have no witness; a stop on the
statement and the transcript the verifier has seen leaves that refusal unreachable and reveals nothing the verifier
didn't see. `Complete` reads the same prover, which stops there too, so the stop costs completeness at most `δc`. -/
def ZKStop (H : Msg → Msg) (D : StopDeploy Coin) (I : Inner Coin Flock.Wit) (DO : Flock.Deploy Coin Col T) : Prop :=
  DO.zk = true ∧ ∀ t, 0 < t → ∀ δ₁ : ℝ≥0, Hm96Hiding D.mask (D.toDeploy.saltHash H) δ₁ →
    ZKmalRx D.Nonce D.Rest H D.V.k D.prover (R D.toDeploy I) (SimStop D I (D.SO t) H) (εzkStop D I t δ₁)
      (γzk D.toDeploy t)

end Recursion

/-! ## At Flock: the inner protocol is Flock -/

section AtFlock

variable {Coin Col T Col' T' : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
  [DecidableEq Col']

/-- **KS of the recursion over Flock**, at every inner and outer decoder that meets the spec: the inner Flock `DI` and
the outer `DO` each on its own columns and traces (the deployed inner session is plain, the V* sessions masked). -/
def KSFlock [Zero Col] [Zero Col'] (H : Msg → Msg) (D : Deploy Coin) (DI : Flock.Deploy Coin Col T)
    (DO : Flock.Deploy Coin Col' T') : Prop :=
  ∀ decI, decOK DI decI → KS H D (Inner.ofFlock DI decI) DO

/-- **Bits of the recursion over Flock**, at every inner decoder that meets the spec, the inner and outer Flock each on
its own columns and traces. -/
def BitsFlock [Zero Col] [Zero Col'] (H : Msg → Msg) (D : Deploy Coin) (DI : Flock.Deploy Coin Col T)
    (DO : Flock.Deploy Coin Col' T') : Prop :=
  ∀ decI, decOK DI decI → Bits H D (Inner.ofFlock DI decI) DO

end AtFlock

end Claims.Recursion
