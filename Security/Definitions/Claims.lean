import Mathlib
import Definitions.Hash.Basic
import Definitions.Core.Boolean
import Definitions.Claims.Hm96
import Definitions.Claims.Json

/-!
# Claims: the shared definitions and Flock's claim statements (frozen before any proof work)

The definitions every protocol's claims read, and Flock's claims, in the condensed spec's names, in these forms:

* `R(x, ⊥) := False` (`Rel.ext`); a protocol with no witness takes `W = Unit` and `R(x, ()) :⇔ x ∈ L` (`Rel.ofLang`).
* The extractor receives the judged run's coins (`Extractor.ext`'s second argument) and uses the prover only by
  rerunning it (`RP`).
* Flock's extractor is `E^rep` (`ERep`): `κ = 4·ε_table + 2^{−m′}`, plus its truncation's `2k/K + e^{−k/8}`.
* `B` is extraction-free (`B`); its assumption is the strict birthday bound (`StrictBirthday`). Extraction-based finders
  have a separate linear expected-time assumption (`ExpectedCR`) and their own term (`KSLin`); Flock needs only `KS`.
  `ExpectedCR` is the weaker of the two at a fixed cost, so a bound that reads one record for every break reads it
  (`ecrBound`, at a cost budget the reader computes from the adversary: PoUW's service premises per prover).
* `t` counts both parties' SHA-512 calls, so the hash side's constant is `255.5` (`lamOf`).
* The scope is a hypothesis of completeness, in readable form (`Flock.InScope`), never part of `R`; `ScopeMatches`
  states that it is the verifier's own scope check. KS needs no scope: the verifier refuses what is out of scope.
* Zero knowledge is against a malicious verifier of the protocol's rounds, with a black-box simulator
  (`ZKmalR`), and its bound adds `γ` times the probability that the honest prover's SHA-512 calls in two real runs on
  one nonce hold a collision (`Fork`): the coin commitment binds only computationally. The form first frozen, `ZKmal`,
  held only at `ε_zk ≥ 1`; the next charged the simulator's own calls, which a simulator that hashes a collision makes
  `1`, so it held at `ε_zk = 0`. The simulator's cost is its expected calls over its coins (`SimCost`). Flock's rounds
  are the masked verifier's `D.V.k` plus the `Hello` that commits every coin (`Flock.ZK`), since `D.V` models the coins
  in the clear.
* Efficiency is the structural cost claims (`ExtractRunsLe`, `ExtractRunsExpLe`, `ERepCost`: reruns of the prover;
  `RP.eval` also counts SHA-512 calls) and the decoder's steps. The steps are the deployed decoder's own count
  (`Flock.Deploy.dec`, a named, counted program), at the encoded length of the word it decodes (`Flock.wordLen`). The
  form first frozen (`DecoderComplexity`) took the decoder, its count and its size as arguments, so a count of `0` met
  it.
* Nontriviality is recorded honest runs that the verifier accepts (`Nontrivial`). It is established
  by a test against recorded sessions, not by a Lean theorem, and no Lean proof of it is expected: Lean's kernel can't
  evaluate SHA-512 over real proofs. The test is `check`'s `lean-agreement` job (`tools/verity/check/ci.toml`), which
  runs the compiled Lean verifier on every recorded honest session (`verity/ml/flock/verifier/agree.py`) and fails
  unless it accepts each. `Nontrivial` states what that test checks.
* List decoding is in its radius form (`Flock.ListWithin`): the opened columns are written into the
  zero-filled word of the trace's codeword length (`Flock.fill`: at each position the first opened column there, the
  positions from the length on ignored), and the decoder lists every trace whose encoding agrees with that word on at
  least `(1 − θ)·len` positions, so leaving positions unopened or opening one twice gains a prover nothing. The radius
  is read per trace, like the length (`Flock.Deploy.len`, `Flock.Deploy.θ`): the plain level-0 rate is `1/2`
  at every schedule, but the masked code's varies with `m`, so one `θ` would cost bits. It replaces the form that
  listed a trace from `nMin` opened columns above the Johnson bound, under which a prover that avoids positions was
  accepted at about `2^{−8}`, so `KS` was false. `Flock.Efficient` asks it of the decoder it costs.
* Flock's relation is the block circuit in every block, and nothing else (`Flock.RelS`). The commitments are
  hm96 rows under frame-v3-sha512 trees that the circuit itself computes and outputs, so the relation reads them as
  the circuit's outputs, and their salts are among the hidden values (`Flock.Wit.hid`). The form first frozen also
  asked the witness to open commit strings `H(u ‖ s)` (`commit`, `Flock.Stmt.cs`), a form the deployed verifier never
  accepts: its commitments are those rows.
* Flock's claims read every table of a session: each of the statement's tables (`Flock.Stmt.tables`) commits
  its own level-0 codeword, so a run's openings are per table (`Flock.Deploy.cols`), `Flock.assemble` decodes each
  table's with the one decoder, takes each table's first candidate in that table's own statement
  (`Flock.Deploy.table`), and reads the tables' candidates as one witness (`Flock.Deploy.read`), and `N₀`, like
  `ε_table`, is summed over the tables: one wait's `k` accepted reruns open every table, so `k` counts all the tables'
  positions. The relation splits by table, each table's units their own statement, so `assemble` checks each table's
  list once, at most `J·L` checks where every combination would be `L^J`, and `Flock.Efficient` bounds each list by
  the decoder's polynomial, which counts them. At one table this is the text it replaces.
* Completeness is at a hash whose outputs are 64 bytes (`Out64`), as the deployed oracle's are, SHA-512 with
  its whole output, and as the random-oracle model draws them. It fixes the length only: `H` stays a parameter.
* Completeness is at a nonempty tape type (`Complete`), and Flock's is of the deployed record's reference
  prover (`Flock.Deploy.P`), not of a prover the claim takes. The form first frozen held at an empty tape type for every
  verifier, and its prover was whatever its discharge chose, so an assembled claim's completeness premise was met for
  free.
* Zero knowledge is of the masked record's own reference prover and simulator (`Flock.ZkDeploy`): its prover
  is `Flock.Deploy.P`, the prover completeness reads, behind the `Hello` and the opening checks of the coins it
  commits, on a finite nonce and rest (`Flock.ZkDeploy.prover`), and its simulator is the record's
  (`Flock.ZkDeploy.S`). The form before took the prover and the simulator as arguments, so a prover that ignores its
  witness, with the simulator that runs it, met it at every masked record at `ε_zk = 0`.

Mathlib only, as `Definitions/` is. So every claim is parametric in the deployed verifier, its parser and the hash:
binding them to the compiled verifier (`Flock.verifyWith Sha512.hash`, `Flock.ProvedScope.check`) is the code side's.

The named assumptions: `StrictBirthday`, `ExpectedCR`, hm96's hiding, `Hm96Hiding` (`Definitions/Claims/Hm96.lean`,
which the soundness proofs read), and the JSON library's specification, `JsonSpec` (`Definitions/Claims/Json.lean`,
which only completeness reads). What the claims rest on and no Lean statement can say is `trustedBase`.

Frozen by `Flock.Guarantees.ClaimsStated` (`Definitions/Flock/Guarantees/Claims.lean`), whose statement reads every claim
here but `Flock.ZK`, and by `Flock.Guarantees.MaskedZK`, which states `Flock.ZK` at the deployed masked record: the lock
records each one's text under `reads`, so a changed claim is a changed record.
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

/-! ## Flock -/

namespace Flock

/-- Where a block's wire value comes from: a public value or a hidden one. -/
inductive Src where
  | pub (i : ℕ)
  | hid (i : ℕ)

/-- The form of a statement, as the verifier's scope check reads it. -/
structure Form where
  hm96Rows : Bool
  typed : Bool
  retainedRounds : Bool
  leaf : String
  tables : ℕ
  publicInputs : Bool
  sharedRows : Bool
  registeredReads : Bool
  hiddenOutputs : Bool
  unitInputGroups : ℕ
  tail : Bool
  unitLog : ℕ
  readsOnlyPortBits : Bool
  rowsV1 : Bool
  rowsV2OrSeg : Bool
  wiresAcyclic : Bool

/-- **Flock's statement**: `2^m` trace bits in blocks of `2^kLog`, the block circuit `C` (a Boolean circuit), the blocks'
inputs and outputs (`inp b g`, `out b g`: the public or hidden value block `b` reads at `C`'s gate `g`), the public
values, and the statement's form. The commitments are no field of their own: they are hm96 rows under frame-v3-sha512
trees, which the circuit computes and outputs (the forms `hm96Rows`, `rowsV1`, `rowsV2OrSeg`), so they are among `C`'s
outputs, which the verifier checks against the registered rows; their salts are among the hidden values. -/
structure Stmt where
  m : ℕ
  kLog : ℕ
  C : FlockSoundness.Audit.Circuit
  inp : ℕ → ℕ → Src
  out : ℕ → ℕ → Src
  pub : List Bool
  form : Form

def Stmt.nBlocks (s : Stmt) : ℕ := 2 ^ (s.m - s.kLog)

/-- **The statement's tables**: the session's table count (`form.tables`), at least one. Each table commits its own
level-0 codeword; a count of `0` or `1` is one table. -/
def Stmt.tables (s : Stmt) : ℕ := max 1 s.form.tables

/-- **Flock's witness**: the hidden values, the salts of the statement's hm96 rows among them. -/
structure Wit where
  hid : List Bool

def val (s : Stmt) (w : Wit) : Src → Bool
  | .pub i => s.pub.getD i false
  | .hid i => w.hid.getD i false

/-- **Flock's relation** on a parsed statement: `C` maps inputs to outputs in every block. Evaluated one bit per gate
(`FlockSoundness.Audit.Circuit.toCircuit`) with each input gate `g` holding the value `inp b g`, `C` holds at each
output gate `o` the value `out b o`. That is the whole relation: the commitments are hm96 rows under frame-v3-sha512
trees that `C` computes from the hidden values, salts included, and outputs, so this clause reads them as `C`'s
outputs. -/
def RelS (s : Stmt) (w : Wit) : Prop :=
  ∀ b < s.nBlocks, ∀ o ∈ s.C.outputs,
    (s.C.toCircuit.eval (fun g => BitVec.ofBool (val s w (s.inp b g.val))) o).getLsbD 0 = val s w (s.out b o.val)

instance instDecidableRelS (s : Stmt) (w : Wit) : Decidable (RelS s w) := by
  unfold RelS; infer_instance

/-- **Flock's relation** on the statement's bytes: it parses (`parse`, the verifier's own parser) and `RelS` holds. No
scope: the verifier refuses what is out of scope, and completeness carries the scope. -/
def R (parse : Msg → Option Stmt) : Rel Wit := fun x w => ∃ s, parse x = some s ∧ RelS s w

/-- The decidable relation, as `E^rep`'s check. -/
def check (parse : Msg → Option Stmt) (x : Msg) (w : Wit) : Bool :=
  match parse x with
  | some s => decide (RelS s w)
  | none => false

/-- **The most tables a `--zk` session may have**, `2^16`: the masked verifier refuses a session past it. -/
def zkTableCap : ℕ := 65536

/-- **Flock's scope, readable**: the deployed parameters (`22 ≤ m ≤ 35`, `25 ≤ m` under `--zk`, `6 ≤ kLog ≤ min(m, 27)`)
and the forms the soundness proof covers: hm96-sha512 rows, untyped, retained rounds, `hm96-sha512/v1` leaves or, with
hidden outputs and without `--zk`, plain SHA-512 leaves (`flock-leaf/sha512-unsalted`), at most
one table without `--zk` and `zkTableCap` with it, no public input ports, shared rows only with hidden outputs,
registered reads and `hm96-sha512/v1` leaves (with or without `--zk`); a unit of at most `2^32` rows reading only port bits, which reads one input group and returns the outputs, or,
with hidden outputs, the general instance form: the unit repeated, reading any number of committed input groups, and
any number of stages the verifier evaluates, reading wires from earlier outputs, with an order of the wires that
computes every slot from the leaves (`wiresAcyclic`); every row a `sha512/row/v1` row, or, with hidden outputs, every
row a `sha512/row/v2` or `sha512/row-seg/v1` row; and no output port pinned (META `pinned_outputs`), which both row forms
(`rowsV1`, `rowsV2OrSeg`) read. -/
def InScope (zk : Bool) (s : Stmt) : Prop :=
  (if zk then 25 else 22) ≤ s.m ∧ s.m ≤ 35 ∧ 6 ≤ s.kLog ∧ s.kLog ≤ min s.m 27 ∧
    s.form.hm96Rows ∧ ¬ s.form.typed ∧ s.form.retainedRounds ∧
    (s.form.leaf = "hm96-sha512/v1" ∨ (¬ zk ∧ s.form.hiddenOutputs ∧ s.form.leaf = "flock-leaf/sha512-unsalted")) ∧
    s.form.tables ≤ (if zk then zkTableCap else 1) ∧ ¬ s.form.publicInputs ∧
    (s.form.sharedRows → s.form.hiddenOutputs ∧ s.form.registeredReads ∧ s.form.leaf = "hm96-sha512/v1") ∧
    ((s.form.unitInputGroups = 1 ∧ ¬ s.form.tail) ∨ (s.form.hiddenOutputs ∧ s.form.wiresAcyclic)) ∧
    s.form.unitLog ≤ 32 ∧ s.form.readsOnlyPortBits ∧
    (s.form.rowsV1 ∨ (s.form.hiddenOutputs ∧ s.form.rowsV2OrSeg))

/-- **The zero-filled word** of length `n`: at each position the first opened column there, zero where none was opened.
Positions from `n` on are ignored. -/
def fill {Col : Type} [Zero Col] (n : ℕ) (os : List (ℕ × Col)) : Fin n → Col :=
  fun i => ((os.find? fun p => p.1 == i.val).map Prod.snd).getD 0

/-- **List decoding within radius `θ`, the spec a decoder meets**: `decode` lists every trace `M` whose encoding agrees
with the zero-filled word of its codeword length `len M` (`fill`) on at least `(1 − θ M)·len M` positions. Agreement is
counted over all `len M` positions, so a prover gains nothing by leaving positions unopened. The radius is read per
trace, like the length. -/
def ListWithin {Col T : Type} [DecidableEq Col] [Zero Col] (len : T → ℕ) (enc : T → ℕ → Col) (θ : T → ℝ)
    (decode : List (ℕ × Col) → List T) : Prop :=
  ∀ (os : List (ℕ × Col)) (M : T),
    (1 - θ M) * len M ≤ ((Finset.univ : Finset (Fin (len M))).filter fun i => enc M i.val = fill (len M) os i).card →
    M ∈ decode os

/-- **The encoded length of an opened word**, in bits, the size the decoder's steps are bounded at: per opened column,
its position in binary (`Nat.size`), one bit that ends the entry, and each of its field elements in `⌈log₂ |F|⌉`
bits. -/
def wordLen {F : Type} [Fintype F] (os : List (ℕ × List F)) : ℕ :=
  (os.map fun o => Nat.size o.1 + 1 + o.2.length * Nat.clog 2 (Fintype.card F)).sum

/-- **Flock's assemble step**: each table's opened columns (`o j` of every opening `o`, for table `j` below the
statement's `Stmt.tables`) as they were opened, decoded by the one decoder to candidate traces; for each table, its
first candidate that, read alone, is in the relation of that table's own statement (`table x j`); the tables' choices,
in table order, read as one witness. Each table's list is checked once, candidate by candidate, so `assemble` makes at
most the lists' lengths summed in checks. At one table, whose own statement is the statement, this is the first
decoded trace of table `0`'s columns whose witness is in the relation. A position opened twice needs no step here:
`fill` reads the first. -/
def assemble {Col T : Type} (parse : Msg → Option Stmt) (table : Msg → ℕ → Option Stmt)
    (decode : List (ℕ × Col) → List T) (read : List T → Wit) (x : Msg) (opens : List (ℕ → List (ℕ × Col))) :
    Option Wit :=
  match parse x with
  | none => none
  | some s =>
    ((List.range s.tables).mapM fun j => (table x j).bind fun t =>
      (decode (opens.flatMap fun o => o j)).find? fun M => decide (RelS t (read [M]))).map read

/-- What the deployed verifier brings, read by Flock's claims: the verifier and whether it is the masked (`--zk`) one,
its parser and scope check, each table's own statement (`table x j`: the statement of table `j`'s units alone, one
table, as the verifier's parser reads it), each table's opened columns of its first committed codeword in a run
(`cols v j` for table `j`), the code (`enc`), each trace's codeword length (`len`) and decoding radius (`θ`), functions
of the trace because one `T` holds the traces of every statement's code, the deployed decoder with its step count
(`dec`), how the tables' traces, one per table in table order (the units' order), are read as one witness (`read`), and
the numbers the errors read: `ε_table`, the tables' errors summed; `N₀`, the tables' level-0 codeword lengths summed;
the weight `C` of `γ`; `E^rep`'s waits `m′` and its truncation `K`. Every table is decoded by the one decoder at the one
code. A single-table statement is its own table `0`, and reads table `0`'s columns and one trace (`read [M]`), its
`ε_table` and `N₀` that table's.

`dec` is a named, counted program: on opened columns it returns its list and the steps it took, counted along the same
computation, one step per field operation (arithmetic or a comparison of field elements), per position compared and
per column entry read, moving data free. `Flock.Efficient` reads that count; that it is the cost of running the
decoder is `trustedBase`'s.

`P` is the reference prover, the honest prover the deployment runs, on its tapes (`Tape`, a nonempty type):
completeness is its (`Complete`), so the claim is about the deployed prover and no other. `δc` is its completeness
error as the deployment states it, at each hash (`δc H x` at `H`): `Complete` is at that error and no other. -/
structure Deploy (Coin Col T : Type) where
  zk : Bool
  V : Verifier Coin
  parse : Msg → Option Stmt
  scopeCheck : Msg → Bool
  table : Msg → ℕ → Option Stmt
  cols : Seen → ℕ → List (ℕ × Col)
  enc : T → ℕ → Col
  len : T → ℕ
  θ : T → ℝ
  dec : List (ℕ × Col) → List T × ℕ
  read : List T → Wit
  εtable : Msg → ℝ
  N₀ : Msg → ℝ
  weight : Msg → ℝ
  waits : Msg → ℕ
  cap : Msg → ℕ
  Tape : Type
  [tapeNonempty : Nonempty Tape]
  P : Wit → Prover Tape
  δc : (Msg → Msg) → Msg → ℝ

instance Deploy.instNonemptyTape {Coin Col T : Type} (D : Deploy Coin Col T) : Nonempty D.Tape := D.tapeNonempty

/-- The deployed decoder's list, without its count. -/
def Deploy.decode {Coin Col T : Type} (D : Deploy Coin Col T) (os : List (ℕ × Col)) : List T := (D.dec os).1

/-- **A masked record**: a record with what its masked interaction (`--zk`) adds, which commits every coin at `Hello`
(`coin-tree/hm96-sha512/v2`) and opens them round by round. The reference prover's tape there is a nonce, which its
`Hello` sends, and the rest (`Nonce`, `Rest`, finite types), read as its tape `tapeOf (ν, r)`; `hello` is its `Hello` at
a nonce and `abort` its message once an opening fails; `chk x ν ρ j m` is its check of the verifier's opening `m` of coin
`j` against the commitment `ρ`, an oracle program whose `H` calls are the prover's, and `coinOf m` the coin `m` opens, as
the public-coin verifier `V` sends it. `S` is the simulator `ZK` is at. -/
structure ZkDeploy (Coin Col T : Type) extends Deploy Coin Col T where
  Nonce : Type
  Rest : Type
  [nonceFin : Fintype Nonce]
  [nonceNe : Nonempty Nonce]
  [restFin : Fintype Rest]
  [restNe : Nonempty Rest]
  tapeOf : Nonce × Rest → Tape
  hello : Msg → Nonce → Msg
  abort : Msg
  chk : Msg → Nonce → Msg → ℕ → Msg → OC Bool
  coinOf : Msg → Msg
  S : Simulator

attribute [instance] ZkDeploy.nonceFin ZkDeploy.nonceNe ZkDeploy.restFin ZkDeploy.restNe

/-- Whether every opening from coin `j` on passes `chk`, checked in order up to the first that fails. -/
def allPass (chk : ℕ → Msg → OC Bool) : ℕ → List Msg → OC Bool
  | _, [] => .ret true
  | j, m :: ms => (chk j m).bind fun b => if b then allPass chk (j + 1) ms else .ret false

/-- **The masked record's reference prover** in its masked interaction, at the witness `w` and the tape `(ν, r)`:
`Hello` at `ν` first; then, if every opening so far passes `chk` against the commitment `ρ` (the verifier's answer to
`Hello`), the record's reference prover `D.P w` at the history with each opening read as its coin (`coinOf`), at the tape
`tapeOf (ν, r)`; else `abort`. -/
def ZkDeploy.prover {Coin Col T : Type} (D : ZkDeploy Coin Col T) (w : Wit) : Prover (D.Nonce × D.Rest) :=
  ⟨fun x hist ω => match hist with
    | [] => .ret (D.hello x ω.1)
    | (_, ρ) :: ops => (allPass (D.chk x ω.1 ρ) 0 (ops.map Prod.snd)).bind fun ok =>
        if ok then (D.P w).next x (ops.map fun p => (p.1, D.coinOf p.2)) (D.tapeOf ω) else .ret D.abort⟩

section Claims

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

/-- `E^rep`'s accepted reruns per wait, `k = ⌈4·N₀/e⌉`. -/
noncomputable def Deploy.k (D : Deploy Coin Col T) (x : Msg) : ℕ := ⌈4 * D.N₀ x / Real.exp 1⌉₊

/-- **Flock's extractor**, `E^rep` at a decoder: an accepted rerun's opening is its tables' columns (`D.cols v`), which
`assemble` decodes and checks table by table, each against the table's own statement (`D.table`). -/
noncomputable def E (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) : Extractor Coin Wit D.V :=
  ERep D.V D.waits D.k D.cap (fun v => [D.cols v]) (assemble D.parse D.table decode D.read) (check D.parse)

/-- **Flock's knowledge error**: `κ = 4·ε_table + 2^{−m′} + 2k/K + e^{−k/8}`. -/
noncomputable def κ (D : Deploy Coin Col T) (x : Msg) : ℝ :=
  4 * D.εtable x + 2 ^ (-(D.waits x : ℝ)) + 2 * D.k x / D.cap x + Real.exp (-(D.k x : ℝ) / 8)

/-- **Flock's `γ`**: `16·(√(2·k(x)·C) + k(x))²`, with `k(x)` the verifier's rounds. -/
noncomputable def γ (D : Deploy Coin Col T) (x : Msg) : ℝ :=
  16 * (Real.sqrt (2 * D.V.k x * D.weight x) + D.V.k x) ^ 2

/-- Flock's scope, on the statement's bytes. -/
def Scope (D : Deploy Coin Col T) (x : Msg) : Prop := ∃ s, D.parse x = some s ∧ InScope D.zk s

/-- **Flock: KS**, at `E^rep` and every decoder that meets the list-decoding spec, the one decoder of every table. -/
def KS [Zero Col] (H : Msg → Msg) (D : Deploy Coin Col T) : Prop :=
  ∀ decode, ListWithin D.len D.enc D.θ decode → Claims.KS H D.V (R D.parse) (E D decode) (κ D) (γ D)

/-- **Flock: Complete**, on the statements in scope, of the deployed record's reference prover (`D.P`), at the
deployed error (`D.δc H`). -/
def Complete (H : Msg → Msg) (D : Deploy Coin Col T) : Prop :=
  Claims.Complete H D.V D.P (R D.parse) (Scope D) (D.δc H)

/-- **Flock: the readable scope is the verifier's.** -/
def ScopeMatches (D : Deploy Coin Col T) : Prop := Claims.ScopeMatches (Scope D) D.scopeCheck

/-- **Flock: zero knowledge** of the masked record's reference prover (`D.prover`, the record's `D.P` in the masked
interaction) at the record's simulator (`D.S`), against every malicious verifier of the deployed interaction's
`D.V.k + 1` rounds: the extra round is the `Hello`, at which the masked verifier sends its commitment to every coin, and
then come `D.V`'s rounds, which open the coins. `D.V` models the coins in the clear; the commitment's hiding is the
live-coins assumption of `trustedBase`. The prover's tape is the nonce its `Hello` sends and the rest. -/
def ZK (H : Msg → Msg) (D : ZkDeploy Coin Col T) (εzk γ : Msg → ℝ) : Prop :=
  D.zk = true ∧ ZKmalR H (fun x => D.V.k x + 1) D.prover (R D.parse) D.S εzk γ

/-- **Flock: bits**, `λ = min(log₂(1/κ), 255.5 − ½ log₂ γ) − 1`, at every decoder that meets the spec, the one decoder
of every table. -/
def Bits [Zero Col] (H : Msg → Msg) (D : Deploy Coin Col T) : Prop :=
  ∀ decode, ListWithin D.len D.enc D.θ decode →
    Claims.Bits H D.V (R D.parse) (E D decode) fun x => lamOf (κ D x) (γ D x)

/-- **Flock: efficiency**: `E^rep`'s reruns at the deployed decoder, which meets the list-decoding spec: the one decoder
of every table. On every opened word its steps (`D.dec`'s count) are at most `p` at the word's encoded length
(`wordLen`), and so are its lists' lengths, so `assemble`, which checks each table's list once, makes at most that many
checks per table. -/
def Efficient {F : Type} [Fintype F] [DecidableEq F] [Zero (List F)] (H : Msg → Msg) (D : Deploy Coin (List F) T)
    (p : Polynomial ℕ) : Prop :=
  ERepCost H D.V D.waits D.k D.cap (fun v => [D.cols v]) (assemble D.parse D.table D.decode D.read) (check D.parse) ∧
    ListWithin D.len D.enc D.θ D.decode ∧ (∀ os, (D.dec os).2 ≤ p.eval (wordLen os)) ∧
    ∀ os, (D.decode os).length ≤ p.eval (wordLen os)

/-- **Flock: nontriviality**: recorded honest runs, each of a statement the scope check takes, that the verifier
accepts. It is established by a test against recorded sessions, not by a Lean proof, and none is expected:
`check`'s `lean-agreement` job (`tools/verity/check/ci.toml`) runs the compiled Lean verifier on recorded honest
sessions (`verity/ml/flock/verifier/agree.py`) and fails unless it accepts each. This states what that test checks. -/
def Nontrivial (H : Msg → Msg) (D : Deploy Coin Col T) (recs : List (Recorded D.V)) : Prop :=
  Claims.Nontrivial H D.V recs ∧ ∀ e ∈ recs, D.scopeCheck e.x = true

end Claims

end Flock

end Claims
