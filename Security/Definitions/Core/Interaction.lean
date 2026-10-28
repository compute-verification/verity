import Mathlib
import Definitions.Core.Hash

/-!
# Interaction: the live-coin model every proof system's claims are stated in

The vocabulary the claims of every proof system read, inner or outer: C-Flock's (`Definitions/Claims.lean`), the
recursion's (`Definitions/Claims/Recursion.lean`) and the sampled proofs' (`Definitions/Claims/Sampling.lean`). It is
Core's, and keeps the namespace `Claims` its records were written under. Its forms:

* A run is between a prover on a tape and a verifier on live coins (`Prover`, `Verifier`, `run`): the verifier reads
  coin `r_j` only from round `j` on, and the prover sees only the messages so far, so no message depends on a coin
  before the round that opens it.
* `R(x, ⊥) := False` (`Rel.ext`); a protocol with no witness takes `W = Unit` and `R(x, ()) :⇔ x ∈ L` (`Rel.ofLang`).
* The extractor receives the judged run's coins (`Extractor.ext`'s second argument) and uses the prover only by
  rerunning it (`RP`). `E^rep` (`ERep`) replays the judged run, then waits for accepted reruns.
* `B` is extraction-free (`B`); its assumption is the strict birthday bound (`StrictBirthday`). Extraction-based finders
  have a separate linear expected-time assumption (`ExpectedCR`) and their own term (`KSLin`). `ExpectedCR` is the
  weaker of the two at a fixed cost, so a bound that reads one record for every break reads it (`ecrBound`, at a cost
  budget the reader computes from the adversary: PoUW's service premises per prover).
* `t` counts both parties' SHA-512 calls, so the hash side's constant is `255.5` (`lamOf`).
* Zero knowledge is against a malicious verifier of the protocol's rounds, with a black-box simulator (`ZKmalR`), and
  its bound adds `γ` times the probability that the honest prover's SHA-512 calls in two real runs on one nonce hold a
  collision (`Fork`). The simulator's cost is its expected calls over its coins (`SimCost`).
* Efficiency is the structural cost claims: reruns of the prover (`ExtractRunsLe`, `ExtractRunsExpLe`, `ERepCost`), and
  `RP.eval` also counts SHA-512 calls.
* Completeness is at a hash whose outputs are 64 bytes (`Out64`) and at a nonempty tape type (`Complete`); the readable
  scope is the verifier's own scope check (`ScopeMatches`).
* Nontriviality is recorded honest runs that the verifier accepts (`Nontrivial`), established by a test against
  recorded sessions, not by a Lean theorem.

Mathlib and Core's hash vocabulary only (`Definitions/Core/Hash.lean`), as Core's layer is. The named assumptions here
are `StrictBirthday` and `ExpectedCR`; what the claims rest on and no Lean statement can say is `trustedBase`.
-/

namespace Claims

/-! ## Messages and the hash -/

/-- A message, `Hash.Msg`: whole bytes, as the deployed SHA-512 reads them. -/
abbrev Msg := Hash.Msg

/-- A collision of `H`, `Hash.Collision`. -/
abbrev Collision (H : Msg → Msg) (u v : Msg) : Prop := Hash.Collision H u v

/-- **`H`'s outputs are 64 bytes**: the deployed oracle is SHA-512 with its whole output, and the random-oracle model
draws 64-byte outputs. Completeness reads it (`Complete`); it fixes the length, not `H`. -/
def Out64 (H : Msg → Msg) : Prop := ∀ m, (H m).length = 64

/-! ## AND/XOR circuits -/

inductive Gate where
  | and (i j : ℕ)
  | xor (i j : ℕ)

/-- A circuit: `nIn` input wires, then one wire per gate; the outputs are wire indices. -/
structure Circuit where
  nIn : ℕ
  gates : List Gate
  outs : List ℕ

/-- Wire values after the gates, in order (input wires first). -/
def Circuit.wires (C : Circuit) (inp : List Bool) : List Bool :=
  C.gates.foldl (fun ws g => ws ++ [match g with
    | .and i j => ws.getD i false && ws.getD j false
    | .xor i j => xor (ws.getD i false) (ws.getD j false)]) inp

def Circuit.eval (C : Circuit) (inp : List Bool) : List Bool :=
  C.outs.map fun o => (C.wires inp).getD o false

/-! ## Relations -/

/-- `R ⊆ X × W` with `X = Msg`. -/
abbrev Rel (W : Type) := Msg → W → Prop

/-- `L_R`. -/
def Rel.L {W : Type} (R : Rel W) (x : Msg) : Prop := ∃ w, R x w

/-- `R(x, ⊥) := False`: an extractor that gives up has failed. -/
def Rel.ext {W : Type} (R : Rel W) (x : Msg) : Option W → Prop
  | some w => R x w
  | none => False

/-- A protocol with no witness: `W = Unit` and `R(x, ()) :⇔ x ∈ L`. -/
def Rel.ofLang (L : Msg → Prop) : Rel Unit := fun x _ => L x

/-! ## Oracle programs -/

/-- A deterministic program with oracle access to `H`. -/
inductive OC (α : Type) where
  | ret : α → OC α
  | query : Msg → (Msg → OC α) → OC α

/-- Run against `H`: the result, and the log of `(query, answer)` pairs in order. -/
def OC.eval (H : Msg → Msg) {α : Type} : OC α → α × List (Msg × Msg)
  | .ret a => (a, [])
  | .query q k => ((OC.eval H (k (H q))).1, (q, H q) :: (OC.eval H (k (H q))).2)

/-- `OC`'s bind: run `x`, then `f` on its result, every query in order. -/
def OC.bind {α β : Type} : OC α → (α → OC β) → OC β
  | .ret a, f => f a
  | .query q k, f => .query q fun r => OC.bind (k r) f

/-! ## Interaction -/

/-- `b_j ∈ Msg ∪ {acc, rej}`. -/
inductive Out where
  | msg (m : Msg)
  | acc
  | rej
  deriving DecidableEq

/-- A prover with tape `Ω`: `a_j = P*^H(x, a_1, b_1, …, b_{j−1}; ω)`. -/
structure Prover (Ω : Type) where
  next : Msg → List (Msg × Msg) → Ω → OC Msg

/-- `P` on the tapes `Ω'`, each read as `P`'s tape `f ω`. -/
def Prover.comap {Ω Ω' : Type} (P : Prover Ω) (f : Ω' → Ω) : Prover Ω' := ⟨fun x hist ω => P.next x hist (f ω)⟩

/-- A verifier with coins `Coin`: `b_j = V^H(x, a_1, …, a_j; r_1, …, r_j)`; it decides within `k(x)` rounds. -/
structure Verifier (Coin : Type) where
  k : Msg → ℕ
  next : Msg → List Msg → List Coin → OC Out

/-- A run: the messages `(a_j, b_j)`, each party's `H` calls, and the decision. -/
structure Trace where
  msgs : List (Msg × Out)
  pLog : List (Msg × Msg)
  vLog : List (Msg × Msg)
  out : Out

/-- What an extractor sees of a rerun: the messages and the decision, not the prover's `H` calls. -/
structure Seen where
  msgs : List (Msg × Out)
  out : Out

def Trace.seen (t : Trace) : Seen := ⟨t.msgs, t.out⟩

/-- Rounds against a verifier given by its next message `vnext` (from the prover's messages so far), from the history
`hist` and the prover's messages `as`, with `n` rounds left. A verifier that has not decided when the rounds run out
rejects. -/
def stepWith {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω) (vnext : List Msg → OC Out) :
    ℕ → List (Msg × Msg) → List Msg → Trace → Trace
  | 0, _, _, t => { t with out := .rej }
  | n + 1, hist, as, t =>
    let pa := (P.next x hist ω).eval H
    let as' := as ++ [pa.1]
    let vb := (vnext as').eval H
    let t' : Trace := { t with msgs := t.msgs ++ [(pa.1, vb.1)], pLog := t.pLog ++ pa.2, vLog := t.vLog ++ vb.2 }
    match vb.1 with
    | .msg m => stepWith H P x ω vnext n (hist ++ [(pa.1, m)]) as' t'
    | .acc => { t' with out := .acc }
    | .rej => { t' with out := .rej }

/-- The coins `r_1, …, r_{k(x)}` as a sequence. -/
def coinSeq {Coin : Type} [Inhabited Coin] {k : ℕ} (r : Fin k → Coin) (i : ℕ) : Coin :=
  if h : i < k then r ⟨i, h⟩ else default

/-- `run(P*, V, x)` at tape `ω` and coins `r`. -/
def run {Ω Coin : Type} [Inhabited Coin] (H : Msg → Msg) (P : Prover Ω) (V : Verifier Coin) (x : Msg) (ω : Ω)
    (r : Fin (V.k x) → Coin) : Trace :=
  stepWith H P x ω (fun as => V.next x as ((List.range as.length).map (coinSeq r))) (V.k x) [] [] ⟨[], [], [], .rej⟩

/-! ## Probability: uniform averages over finite coin spaces -/

open Classical in
noncomputable def Pr {α : Type} [Fintype α] (p : α → Prop) : ℝ :=
  ((Finset.univ.filter p).card : ℝ) / Fintype.card α

/-- The expectation of `f` under uniform coins. -/
noncomputable def Ex {α : Type} [Fintype α] (f : α → ℝ) : ℝ := (∑ a, f a) / Fintype.card α

/-! ## Rerun programs: how an extractor uses the prover -/

/-- A program that may call `H` and rerun the prover at the judged run's tape on coins `r` of its choice, seeing the
rerun's messages and decision. -/
inductive RP (R α : Type) where
  | ret : α → RP R α
  | hash : Msg → (Msg → RP R α) → RP R α
  | rerun : R → (Seen → RP R α) → RP R α

/-- Run against `H` and the prover's runs `run`: the result, the SHA-512 calls (its own and every rerun's, both
parties'), and the reruns. -/
def RP.eval {R α : Type} (H : Msg → Msg) (run : R → Trace) : RP R α → α × ℕ × ℕ
  | .ret a => (a, 0, 0)
  | .hash q k =>
    let e := RP.eval H run (k (H q))
    (e.1, e.2.1 + 1, e.2.2)
  | .rerun r k =>
    let e := RP.eval H run (k (run r).seen)
    (e.1, e.2.1 + (run r).pLog.length + (run r).vLog.length, e.2.2 + 1)

def RP.bind {R α β : Type} : RP R α → (α → RP R β) → RP R β
  | .ret a, f => f a
  | .hash q k, f => .hash q fun a => (k a).bind f
  | .rerun r k, f => .rerun r fun v => (k v).bind f

/-! ## Extractors -/

/-- An extractor: its own coins `C x`, and a rerun program from the **judged run's coins** `r₀` and its coins. -/
structure Extractor (Coin W : Type) (V : Verifier Coin) where
  C : Msg → Type
  [fin : ∀ x, Fintype (C x)]
  [ne : ∀ x, Nonempty (C x)]
  ext : (x : Msg) → (Fin (V.k x) → Coin) → C x → RP (Fin (V.k x) → Coin) (Option W)

attribute [instance] Extractor.fin Extractor.ne

section Extract

variable {Ω Coin W : Type} [Inhabited Coin]

/-- `E^{P*}(x)` on the judged run `(ω, r₀)` and the extractor's coins `c`: the result, SHA-512 calls and reruns. -/
def Extractor.eval (H : Msg → Msg) (P : Prover Ω) {V : Verifier Coin} (E : Extractor Coin W V) (x : Msg) (ω : Ω)
    (r₀ : Fin (V.k x) → Coin) (c : E.C x) : Option W × ℕ × ℕ :=
  (E.ext x r₀ c).eval H (run H P V x ω)

end Extract

/-- Rerun on the coins `rs` in order until `k` accepted runs are in hand, collecting the accepted ones. -/
def waitAcc {R : Type} (k : ℕ) : List R → List Seen → RP R (List Seen)
  | [], acc => .ret acc
  | r :: rs, acc =>
    if k ≤ acc.length then .ret acc
    else .rerun r fun v => waitAcc k rs (if v.out = .acc then acc ++ [v] else acc)

/-- One wait per coin list: assemble each wait's openings, and return the first candidate that `check` accepts. -/
def firstWait {R W Op : Type} (k : ℕ) (opens : Seen → List Op) (assemble : List Op → Option W) (check : W → Bool) :
    List (List R) → RP R (Option W)
  | [] => .ret none
  | rs :: rss => (waitAcc k rs []).bind fun vs =>
    match assemble (vs.flatMap opens) with
    | some w => if check w then .ret (some w) else firstWait k opens assemble check rss
    | none => firstWait k opens assemble check rss

/-- **`E^rep`**: replay the judged run; if it accepted, make `m` waits, each over `K` fresh coins and stopping at `k`
accepted reruns, and return the first assembled candidate in the relation (`check`). -/
def ERep {Coin W Op : Type} [Fintype Coin] [Nonempty Coin] (V : Verifier Coin) (m k K : Msg → ℕ)
    (opens : Seen → List Op) (assemble : Msg → List Op → Option W) (check : Msg → W → Bool) : Extractor Coin W V where
  C x := Fin (m x) → Fin (K x) → Fin (V.k x) → Coin
  ext x r₀ ρ := .rerun r₀ fun v₀ =>
    if v₀.out = .acc then
      firstWait (k x) opens (assemble x) (check x) ((List.finRange (m x)).map fun i => (List.finRange (K x)).map (ρ i))
    else .ret none

/-! ## Collision finders and the two collision assumptions -/

/-- A collision finder: its coins, its output and its SHA-512 calls on each coin. -/
structure Finder where
  S : Type
  [fin : Fintype S]
  [ne : Nonempty S]
  out : S → Option (Msg × Msg)
  cost : S → ℕ

attribute [instance] Finder.fin Finder.ne

/-- `Pr[F outputs a collision of H]`. -/
noncomputable def Finder.Pr (H : Msg → Msg) (F : Finder) : ℝ :=
  Claims.Pr fun s : F.S => ∃ u v, F.out s = some (u, v) ∧ Collision H u v

/-- **`StrictBirthday` (`cr/sha-512`): the strict birthday bound**, at one finder: if `F` makes at most `q` SHA-512
calls on every coin, it finds a collision with probability at most `q² / 2^513`. Read for `B` (extraction-free). -/
def StrictBirthday (H : Msg → Msg) (F : Finder) : Prop :=
  ∀ q : ℝ, (∀ s, (F.cost s : ℝ) ≤ q) → F.Pr H ≤ q ^ 2 / 2 ^ (513 : ℝ)

/-- **`ExpectedCR` (`ecr/sha-512`): the linear expected-time bound**, at one finder: `F` finds a collision with
probability at most its expected SHA-512 calls over `2^256`. Read for finders whose cost is an expectation
(extraction-based and rewinding ones, which have no worst-case `q`: `KSLin`, C-Flock's link) and, where one record
should serve every break, for the others too. It is the weaker claim there: at a fixed cost `q ≤ 2^257`,
`StrictBirthday`'s `q²/2^513` is at most `q/2^256`. PoUW's service premises per prover read it at their binding and
zero-knowledge finders through `ecrBound`, at a budget computed from the adversary's time. -/
def ExpectedCR (H : Msg → Msg) (F : Finder) : Prop :=
  F.Pr H ≤ Ex (fun s => (F.cost s : ℝ)) / 2 ^ (256 : ℝ)

open Classical in
/-- **`ecr/sha-512` at one finder as a number, at a cost budget, or no bound**: `q / 2^256` where the finder meets
`ExpectedCR` and its expected SHA-512 calls are at most the budget `q`, and `⊤` otherwise, as C-Flock's link bound reads
its collision assumption (`⊤` at a prover whose link finders fail it, the number read at the session's calls `t'`).
The assumption is a predicate on one finder, since some finder outputs a collision after no calls, so it is asked of
this finder only.

The reader computes `q` from the adversary the finder is built from, never from the finder: PoUW's bounds per prover
read `c · time M + d` at stated `c`, `d`. So the number is the budget's, which no choice of finder lowers. A finder's
cost is a declared field (`Finder.cost`): one that declares more calls than it makes, so as to meet `ExpectedCR` while
it outputs a collision on some coins, has `Pr` about the break's own probability, exceeds the budget once that is
above `q / 2^256`, and gets `⊤`. `ENNReal.ofReal (F.Pr H) ≤ ecrBound H F q` holds by definition, so a bound stated
through `ecrBound` takes no hypothesis on `H`. The one break term the protocols' per-prover bounds share. -/
noncomputable def ecrBound (H : Msg → Msg) (F : Finder) (q : ℝ) : ENNReal :=
  if ExpectedCR H F ∧ Ex (fun s => (F.cost s : ℝ)) ≤ q then ENNReal.ofReal (q / 2 ^ (256 : ℝ)) else ⊤

/-- The first collision among the `(input, output)` pairs `l`: two different inputs with one output. -/
def findColl : List (Msg × Msg) → Option (Msg × Msg)
  | [] => none
  | (q, a) :: l =>
    match l.find? (fun p => decide (p.1 ≠ q) && decide (p.2 = a)) with
    | some p => some (q, p.1)
    | none => findColl l

/-- `B`'s coins: the round guess `j`, the tape, the shared coins and the fresh continuation. -/
abbrev BCoins (Ω Coin : Type) (k : ℕ) := Fin k × Ω × (Fin k → Coin) × (Fin k → Coin)

/-- The second run's coins: `r_1, …, r_j` shared, the rest fresh. -/
def forkCoins {Coin : Type} {k : ℕ} (j : Fin k) (r r' : Fin k → Coin) : Fin k → Coin :=
  fun i => if i.val < j.val then r i else r' i

/-- **`B`** (extraction-free): guess `j`, draw `ω, r_1..r_j`, run twice from the shared start with fresh continuations,
and output the first collision among the verifier's SHA-512 calls in the two runs. Its cost is both runs' calls, both
parties'. -/
def B {Ω Coin : Type} [Fintype Ω] [Nonempty Ω] [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg)
    (V : Verifier Coin) (P : Prover Ω) (x : Msg) (hk : 0 < V.k x) : Finder where
  S := BCoins Ω Coin (V.k x)
  ne := ⟨⟨⟨0, hk⟩, Classical.arbitrary _, fun _ => default, fun _ => default⟩⟩
  out s := findColl ((run H P V x s.2.1 s.2.2.1).vLog ++ (run H P V x s.2.1 (forkCoins s.1 s.2.2.1 s.2.2.2)).vLog)
  cost s :=
    let t₁ := run H P V x s.2.1 s.2.2.1
    let t₂ := run H P V x s.2.1 (forkCoins s.1 s.2.2.1 s.2.2.2)
    t₁.pLog.length + t₁.vLog.length + t₂.pLog.length + t₂.vLog.length

/-- `Pr[B^{P*}(x) outputs a collision]`, `0` for a verifier that takes no rounds at `x`. -/
noncomputable def PrB {Ω Coin : Type} [Fintype Ω] [Nonempty Ω] [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    (H : Msg → Msg) (V : Verifier Coin) (P : Prover Ω) (x : Msg) : ℝ :=
  if hk : 0 < V.k x then (B H V P x hk).Pr H else 0

/-- **The trusted base**: what the claims rest on that no Lean statement here can say. -/
def trustedBase : List String :=
  ["the Lean compiler and runtime: the compiled verifier computes what its Lean definition says",
   "the live coins: the auditor's coins are uniform, and each is hidden from the prover until the verifier opens it: " ++
     "the prover sees nothing that depends on a coin before the round that opens it, except the commitment to every " ++
     "coin that the --zk verifier sends at Hello and opens round by round",
   "the decoder's count: the deployed decoder's step count (Flock.Deploy.dec) is the cost of running it, one step " ++
     "per field operation (arithmetic or a comparison), per position compared and per column entry read",
   "trusted-host: the auditor's device runs its stated code, timing included, on a host the prover doesn't control, " ++
     "with its CPU isolated from other tenants and a monotonic clock; the host never restores its earlier state"]

/-! ## The claims -/

section Claims

variable {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- `Pr[run = acc ∧ ¬R(x, E^{P*}(x))]` over the tape, the judged run's coins and the extractor's coins. -/
noncomputable def PrBad (H : Msg → Msg) (V : Verifier Coin) (R : Rel W) (E : Extractor Coin W V) (x : Msg)
    {Ω : Type} [Fintype Ω] (P : Prover Ω) : ℝ :=
  Pr fun s : Ω × (Fin (V.k x) → Coin) × E.C x =>
    (run H P V x s.1 s.2.1).out = .acc ∧ ¬ R.ext x (E.eval H P x s.1 s.2.1 s.2.2).1

/-- `KS(V, R, E, κ, γ)`: for every prover `P*` (any finite tape),
`Pr[run = acc ∧ ¬R(x, E^{P*}(x))] ≤ κ(x) + √(γ(x) · Pr[B^{P*}(x) outputs a collision])`. `E` receives the judged run's
coins `r₀` and reruns `P*` at the judged run's tape. -/
def KS (H : Msg → Msg) (V : Verifier Coin) (R : Rel W) (E : Extractor Coin W V) (κ γ : Msg → ℝ) : Prop :=
  ∀ (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω),
    PrBad H V R E x P ≤ κ x + Real.sqrt (γ x * PrB H V P x)

/-- Extraction-based collision finders, one per prover and statement. -/
def FinderFamily := ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω], Prover Ω → Msg → Finder

/-- `KS` with a linear term for extraction-based finders `Fe`: `… + β(x) · Pr[Fe^{P*}(x) outputs a collision]`. -/
def KSLin (H : Msg → Msg) (V : Verifier Coin) (R : Rel W) (E : Extractor Coin W V) (κ γ β : Msg → ℝ)
    (Fe : FinderFamily) : Prop :=
  ∀ (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω),
    PrBad H V R E x P ≤ κ x + Real.sqrt (γ x * PrB H V P x) + β x * (Fe Ω P x).Pr H

/-- `Complete(V, P, R, Scope, δ_c)`: at a hash whose outputs are 64 bytes (`Out64`), on every statement in scope and
witness in the relation, the honest prover `P(w; ω)` is accepted with probability at least `1 − δ_c(x)` over the
verifier's coins, at every tape. The tapes are a nonempty type: over an empty one, "at every tape" holds of every
verifier. -/
def Complete {Ω : Type} [Nonempty Ω] (H : Msg → Msg) (V : Verifier Coin) (P : W → Prover Ω) (R : Rel W)
    (Scope : Msg → Prop) (δc : Msg → ℝ) : Prop :=
  Out64 H →
    ∀ x w, Scope x → R x w → ∀ ω : Ω, 1 - δc x ≤ Pr fun r : Fin (V.k x) → Coin => (run H (P w) V x ω r).out = .acc

/-- The readable scope is the verifier's own scope check. -/
def ScopeMatches (Scope : Msg → Prop) (check : Msg → Bool) : Prop := ∀ x, Scope x ↔ check x = true

/-- A malicious verifier with tape `Θ`: its rounds and its next message from the prover's messages so far. -/
structure VStar (Θ : Type) where
  k : Msg → ℕ
  next : Msg → List Msg → Θ → OC Out

/-- The honest prover `P(w)` at tape `ω` against `V*` at tape `θ`. -/
def runStar {Ω Θ : Type} (H : Msg → Msg) (P : Prover Ω) (Vs : VStar Θ) (x : Msg) (ω : Ω) (θ : Θ) : Trace :=
  stepWith H P x ω (fun as => Vs.next x as θ) (Vs.k x) [] [] ⟨[], [], [], .rej⟩

/-- A simulator program: it may call `H` and ask the verifier (black box) its answer to any prover messages. -/
inductive SP (α : Type) where
  | ret : α → SP α
  | hash : Msg → (Msg → SP α) → SP α
  | ask : List Msg → (Out → SP α) → SP α

/-- Run against `H` and the verifier `ask`: the result, and the calls (hash and ask). -/
def SP.eval {α : Type} (H : Msg → Msg) (ask : List Msg → Out) : SP α → α × ℕ
  | .ret a => (a, 0)
  | .hash q k => let e := SP.eval H ask (k (H q)); (e.1, e.2 + 1)
  | .ask as k => let e := SP.eval H ask (k (ask as)); (e.1, e.2 + 1)

/-- A simulator: its coins and its program, which outputs a transcript. -/
structure Simulator where
  C : Msg → Type
  [fin : ∀ x, Fintype (C x)]
  [ne : ∀ x, Nonempty (C x)]
  prog : (x : Msg) → C x → SP (List (Msg × Out))

attribute [instance] Simulator.fin Simulator.ne

/-- The simulator's transcript against `V*` at `θ`. -/
def Simulator.out {Θ : Type} (H : Msg → Msg) (S : Simulator) (Vs : VStar Θ) (θ : Θ) (x : Msg) (c : S.C x) :
    List (Msg × Out) :=
  ((S.prog x c).eval H fun as => ((Vs.next x as θ).eval H).1).1

/-- **`Fork`**, the real game's collision finder: the honest prover, whose tape is a nonce `N` and the rest `Ω`, run
twice against `V*` at `θ` on one nonce and two fresh rests; it outputs the first collision among the prover's SHA-512
calls in the two runs. Its cost is both runs' calls, both parties'. -/
def Fork {N Ω Θ : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] (H : Msg → Msg) (P : Prover (N × Ω))
    (Vs : VStar Θ) (θ : Θ) (x : Msg) : Finder where
  S := N × Ω × Ω
  out s := findColl ((runStar H P Vs x (s.1, s.2.1) θ).pLog ++ (runStar H P Vs x (s.1, s.2.2) θ).pLog)
  cost s :=
    let t₁ := runStar H P Vs x (s.1, s.2.1) θ
    let t₂ := runStar H P Vs x (s.1, s.2.2) θ
    t₁.pLog.length + t₁.vLog.length + t₂.pLog.length + t₂.vLog.length

/-- `ZK(k, P, R, S, ε_zk, γ)` against a malicious verifier: for `(x, w) ∈ R`, every verifier `V*` of the protocol's
`k(x)` rounds and tape `θ`, the transcript of `P(w)` against `V*` is within `ε_zk(x)` of `S`'s on every event, plus
`γ(x)` times the probability that `Fork` finds a collision among the honest prover's SHA-512 calls in two runs against
`V*` on one nonce. The simulator is not told `V*`'s rounds, so the real run lasts the protocol's. The coin commitment
binds only computationally: a `V*` that opens a committed coin two ways makes the honest prover's checks of the two
openings collide. The term reads only the real game, never the simulator's calls. -/
def ZKmalR {N Ω : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] (H : Msg → Msg) (k : Msg → ℕ)
    (P : W → Prover (N × Ω)) (R : Rel W) (S : Simulator) (εzk γ : Msg → ℝ) : Prop :=
  ∀ x w, R x w → ∀ (Θ : Type) (Vs : VStar Θ) (θ : Θ), Vs.k x = k x → ∀ (Ev : List (Msg × Out) → Prop),
    |Pr (fun ω : N × Ω => Ev (runStar H (P w) Vs x ω θ).msgs) - Pr (fun c : S.C x => Ev (S.out H Vs θ x c))|
      ≤ εzk x + γ x * (Fork H (P w) Vs θ x).Pr H

/-- **The simulator's cost, expected**: over its coins, the simulator makes at most `bound(x)` calls (hash and ask) in
expectation, against every verifier. -/
def SimCost (H : Msg → Msg) (S : Simulator) (bound : Msg → ℝ) : Prop :=
  ∀ x (ask : List Msg → Out), Ex (fun c : S.C x => (((S.prog x c).eval H ask).2 : ℝ)) ≤ bound x

/-- `λ = min(log₂(1/κ), 255.5 − ½ log₂ γ) − 1`. -/
noncomputable def lamOf (κ γ : ℝ) : ℝ := min (Real.logb 2 (1 / κ)) (255.5 - Real.logb 2 γ / 2) - 1

/-- `λ` bits: at every `x`, every prover making at most `t ≥ 1` SHA-512 calls per run (both parties') has
`Pr[acc ∧ ¬R] ≤ t · 2^{−λ(x)}`, where `B` obeys the strict birthday bound. -/
def Bits (H : Msg → Msg) (V : Verifier Coin) (R : Rel W) (E : Extractor Coin W V) (lam : Msg → ℝ) : Prop :=
  ∀ (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (t : ℕ), 1 ≤ t →
    (∀ ω r, (run H P V x ω r).pLog.length + (run H P V x ω r).vLog.length ≤ t) →
    (∀ hk : 0 < V.k x, StrictBirthday H (B H V P x hk)) →
    PrBad H V R E x P ≤ t * (2 : ℝ) ^ (-lam x)

/-- **Cost, worst case**: `E` reruns the prover at most `runs(x)` times, on every prover, tape and coin. Counts reruns
only, not the pure computation between them. -/
def ExtractRunsLe (H : Msg → Msg) {V : Verifier Coin} (E : Extractor Coin W V) (runs : Msg → ℕ) : Prop :=
  ∀ (x : Msg) (Ω : Type) (P : Prover Ω) (ω : Ω) (r₀ : Fin (V.k x) → Coin) (c : E.C x),
    (E.eval H P x ω r₀ c).2.2 ≤ runs x

/-- **Cost, expected**: over the judged run's coins and `E`'s, `E` reruns the prover at most `runs(x)` times in
expectation, at every prover and tape. -/
def ExtractRunsExpLe (H : Msg → Msg) {V : Verifier Coin} (E : Extractor Coin W V) (runs : Msg → ℝ) : Prop :=
  ∀ (x : Msg) (Ω : Type) (P : Prover Ω) (ω : Ω),
    Ex (fun s : (Fin (V.k x) → Coin) × E.C x => ((E.eval H P x ω s.1 s.2).2.2 : ℝ)) ≤ runs x

/-- `E^rep`'s cost: at most `1 + m·K` reruns, and `1 + m·k` expected. -/
def ERepCost {Op : Type} (H : Msg → Msg) (V : Verifier Coin) (m k K : Msg → ℕ) (opens : Seen → List Op)
    (assemble : Msg → List Op → Option W) (check : Msg → W → Bool) : Prop :=
  ExtractRunsLe H (ERep V m k K opens assemble check) (fun x => 1 + m x * K x) ∧
    ExtractRunsExpLe H (ERep V m k K opens assemble check) (fun x => 1 + m x * k x)

end Claims

/-- A prover that sends the recorded messages `as` whatever the verifier says. -/
def replay (as : List Msg) : Prover Unit := ⟨fun _ hist _ => .ret (as.getD hist.length [])⟩

/-- A recorded honest run: its statement, the prover's messages and the verifier's coins. -/
structure Recorded {Coin : Type} (V : Verifier Coin) where
  x : Msg
  as : List Msg
  r : Fin (V.k x) → Coin

/-- **Nontriviality**: there are recorded runs, and the verifier accepts each when it is replayed. It is established by
a test against recorded sessions, not by a Lean proof, and none is expected: `check`'s `lean-agreement` job
(`tools/verity/check/ci.toml`) replays recorded honest sessions through the compiled Lean verifier
(`verity/ml/flock/verifier/agree.py`) and fails unless it accepts each. This states what that test checks. -/
def Nontrivial {Coin : Type} [Inhabited Coin] (H : Msg → Msg) (V : Verifier Coin) (recs : List (Recorded V)) : Prop :=
  recs ≠ [] ∧ ∀ e ∈ recs, (run H (replay e.as) V e.x () e.r).out = .acc

end Claims
