import Mathlib

/-!
# Claims: the shared definitions and Flock's claim statements (draft, frozen before any proof work)

The definitions every protocol's claims read, and Flock's claims, in the condensed spec's names, with the rulings of
15 Oct applied:

* `R(x, ⊥) := False` (`Rel.ext`); a protocol with no witness takes `W = Unit` and `R(x, ()) :⇔ x ∈ L` (`Rel.ofLang`).
* The extractor receives the judged run's coins (`Extractor.ext`'s second argument) and uses the prover only by
  rerunning it (`RP`).
* Flock's extractor is `E^rep` (`ERep`): `κ = 4·ε_table + 2^{−m′}`, plus its truncation's `2k/K + e^{−k/8}`.
* `B` is extraction-free (`B`); its assumption is the strict birthday bound (`StrictBirthday`). Extraction-based finders
  have a separate linear expected-time assumption (`ExpectedCR`) and their own term (`KSLin`); Flock needs only `KS`.
* `t` counts both parties' SHA-512 calls, so the hash side's constant is `255.5` (`lamOf`).
* The scope is a hypothesis of completeness, in readable form (`Flock.InScope`), never part of `R`; `ScopeMatches`
  states that it is the verifier's own scope check. KS needs no scope: the verifier refuses what is out of scope.
* Zero knowledge is against a malicious verifier (`ZKmal`), with a black-box simulator.
* Efficiency is the structural cost claims (`ERepWorst`, `ERepExpected`: reruns of the prover, SHA-512 calls) and a
  separate stated complexity claim for the decoder's pure computation (`DecoderComplexity`).
* Nontriviality is a recorded honest transcript that the verifier accepts (`Nontrivial`).

Mathlib only, as `Definitions/` is. So every claim is parametric in the deployed verifier, its parser and the hash:
binding them to the compiled verifier (`Flock.verifyWith Sha512.hash`, `Flock.ProvedScope.check`) is the code side's.

The named assumptions (each read once, in a claim's hypothesis): `StrictBirthday`, `ExpectedCR`, `CommitHides`. What the
claims rest on and no Lean statement can say is `trustedBase`.

Not imported by `Definitions.lean` and not in `lean-audit.json`: no build reads it and no lock records it until the
claims are frozen.
-/

namespace Claims

/-! ## Messages, the hash, commitments -/

/-- `Msg = {0,1}^*` in whole bytes, as the deployed SHA-512 reads them: a padding convention for other lengths would
make collisions that are not SHA-512's. -/
abbrev Msg := List UInt8

/-- Bits packed into bytes, low bit first, the last byte padded with zeros: injective on bit strings of one length. -/
def packBits (bs : List Bool) : Msg :=
  (List.range ((bs.length + 7) / 8)).map fun i =>
    UInt8.ofNat ((List.range 8).foldl (fun acc j => acc + if bs.getD (8 * i + j) false then 2 ^ j else 0) 0)

/-- A collision of `H`. -/
def Collision (H : Msg → Msg) (u v : Msg) : Prop := u ≠ v ∧ H u = H v

/-- `commit(u; s) = H(u ‖ s)`. -/
def commit (H : Msg → Msg) (u s : Msg) : Msg := H (u ++ s)

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

/-- **Assumption: the strict birthday bound**, at one finder: if `F` makes at most `q` SHA-512 calls on every coin, it
finds a collision with probability at most `q² / 2^513`. Read for `B` (extraction-free). -/
def StrictBirthday (H : Msg → Msg) (F : Finder) : Prop :=
  ∀ q : ℝ, (∀ s, (F.cost s : ℝ) ≤ q) → F.Pr H ≤ q ^ 2 / 2 ^ (513 : ℝ)

/-- **Assumption: the linear expected-time bound**, at one finder: `F` finds a collision with probability at most its
expected SHA-512 calls over `2^256`. Read for extraction-based finders only. -/
def ExpectedCR (H : Msg → Msg) (F : Finder) : Prop :=
  F.Pr H ≤ Ex (fun s => (F.cost s : ℝ)) / 2 ^ (256 : ℝ)

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

/-- **Assumption: hiding**, at one commitment: for messages of one length, `commit(u; s)` and `commit(u′; s)` at a
uniform salt of `saltLen` bytes differ by at most `δ` on every event. -/
def CommitHides (H : Msg → Msg) (saltLen : ℕ) (δ : ℝ) : Prop :=
  ∀ u u' : Msg, u.length = u'.length → ∀ Ev : Msg → Prop,
    |Pr (fun s : Fin saltLen → UInt8 => Ev (commit H u (List.ofFn s))) -
      Pr (fun s : Fin saltLen → UInt8 => Ev (commit H u' (List.ofFn s)))| ≤ δ

/-- **The trusted base**: what the claims rest on that no Lean statement here can say. -/
def trustedBase : List String :=
  ["the Lean compiler and runtime: the compiled verifier computes what its Lean definition says",
   "the live coins: the auditor's randomness is uniform and drawn after the prover commits",
   "trusted-nebius-host: the auditor's devices and their host provider"]

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

/-- `Complete(V, P, R, Scope, δ_c)`: on every statement in scope and witness in the relation, the honest prover
`P(w; ω)` is accepted with probability at least `1 − δ_c(x)` over the verifier's coins, at every tape. -/
def Complete {Ω : Type} (H : Msg → Msg) (V : Verifier Coin) (P : W → Prover Ω) (R : Rel W) (Scope : Msg → Prop)
    (δc : Msg → ℝ) : Prop :=
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

/-- `ZK(P, R, S, ε_zk)` against a malicious verifier: for `(x, w) ∈ R`, every verifier `V*` and tape `θ`, the transcript
of `P(w)` against `V*` is within `ε_zk(x)` of `S`'s, on every event. -/
def ZKmal {Ω : Type} [Fintype Ω] (H : Msg → Msg) (P : W → Prover Ω) (R : Rel W) (S : Simulator) (εzk : Msg → ℝ) :
    Prop :=
  ∀ x w, R x w → ∀ (Θ : Type) (Vs : VStar Θ) (θ : Θ) (Ev : List (Msg × Out) → Prop),
    |Pr (fun ω : Ω => Ev (runStar H (P w) Vs x ω θ).msgs) - Pr (fun c : S.C x => Ev (S.out H Vs θ x c))| ≤ εzk x

/-- The simulator makes at most `bound(x)` calls (hash and ask) against every verifier. -/
def SimCost (H : Msg → Msg) (S : Simulator) (bound : Msg → ℕ) : Prop :=
  ∀ x (c : S.C x) (ask : List Msg → Out), ((S.prog x c).eval H ask).2 ≤ bound x

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

/-- **The decoder's complexity (a stated claim)**: `counted` computes what `decode` does, and the steps it reports are at
most `p(size)`. The step count is the instrumented decoder's own: nothing here checks it against a machine model. -/
def DecoderComplexity {I T : Type} (decode : I → T) (counted : I → T × ℕ) (size : I → ℕ) (p : Polynomial ℕ) : Prop :=
  ∀ i, (counted i).1 = decode i ∧ (counted i).2 ≤ p.eval (size i)

/-- A prover that sends the recorded messages `as` whatever the verifier says. -/
def replay (as : List Msg) : Prover Unit := ⟨fun _ hist _ => .ret (as.getD hist.length [])⟩

/-- A recorded honest run: its statement, the prover's messages and the verifier's coins. -/
structure Recorded {Coin : Type} (V : Verifier Coin) where
  x : Msg
  as : List Msg
  r : Fin (V.k x) → Coin

/-- **Nontriviality**: there are recorded runs, and the verifier accepts each when it is replayed. -/
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

/-- **Flock's statement**: `2^m` trace bits in blocks of `2^kLog`, the block circuit `C`, the blocks' inputs and outputs
(`inp b`, `out b`, reading the public or the hidden values), the public values, the commit strings (string `c` commits to
the hidden values `grp c`), and the statement's form. -/
structure Stmt where
  m : ℕ
  kLog : ℕ
  C : Circuit
  inp : ℕ → List Src
  out : ℕ → List Src
  pub : List Bool
  cs : List Msg
  grp : ℕ → List ℕ
  form : Form

def Stmt.nBlocks (s : Stmt) : ℕ := 2 ^ (s.m - s.kLog)

/-- **Flock's witness**: the hidden values and the salts. -/
structure Wit where
  hid : List Bool
  salts : List Msg

def val (s : Stmt) (w : Wit) : Src → Bool
  | .pub i => s.pub.getD i false
  | .hid i => w.hid.getD i false

/-- **Flock's relation** on a parsed statement: `w` opens the commit strings, and `C` maps inputs to outputs in every
block. -/
def RelS (H : Msg → Msg) (s : Stmt) (w : Wit) : Prop :=
  (∀ c < s.cs.length,
    commit H (packBits ((s.grp c).map fun i => w.hid.getD i false)) (w.salts.getD c []) = s.cs.getD c []) ∧
  ∀ b < s.nBlocks, s.C.eval ((s.inp b).map (val s w)) = (s.out b).map (val s w)

instance (H : Msg → Msg) (s : Stmt) (w : Wit) : Decidable (RelS H s w) := by
  unfold RelS; infer_instance

/-- **Flock's relation** on the statement's bytes: it parses (`parse`, the verifier's own parser) and `RelS` holds. No
scope: the verifier refuses what is out of scope, and completeness carries the scope. -/
def R (H : Msg → Msg) (parse : Msg → Option Stmt) : Rel Wit := fun x w => ∃ s, parse x = some s ∧ RelS H s w

/-- The decidable relation, as `E^rep`'s check. -/
def check (H : Msg → Msg) (parse : Msg → Option Stmt) (x : Msg) (w : Wit) : Bool :=
  match parse x with
  | some s => decide (RelS H s w)
  | none => false

/-- **Flock's scope, readable**: the deployed parameters (`22 ≤ m ≤ 35`, `25 ≤ m` under `--zk`, `6 ≤ kLog ≤ min(m, 27)`)
and the forms the soundness proof covers: hm96-sha512 rows, untyped, retained rounds, `hm96-sha512/v1` leaves, at most
one table without `--zk`, no public input ports, shared rows only in a `--zk` session with hidden outputs and registered
reads; a unit reading one input group and returning the outputs, of at most `2^32` rows, reading only port bits; every
row a `sha512/row/v1` row, or, with hidden outputs, every row a `sha512/row/v2` or `sha512/row-seg/v1` row. -/
def InScope (zk : Bool) (s : Stmt) : Prop :=
  (if zk then 25 else 22) ≤ s.m ∧ s.m ≤ 35 ∧ 6 ≤ s.kLog ∧ s.kLog ≤ min s.m 27 ∧
    s.form.hm96Rows ∧ ¬ s.form.typed ∧ s.form.retainedRounds ∧ s.form.leaf = "hm96-sha512/v1" ∧
    (zk ∨ s.form.tables ≤ 1) ∧ ¬ s.form.publicInputs ∧
    (s.form.sharedRows → zk ∧ s.form.hiddenOutputs ∧ s.form.registeredReads) ∧
    s.form.unitInputGroups = 1 ∧ ¬ s.form.tail ∧ s.form.unitLog ≤ 32 ∧ s.form.readsOnlyPortBits ∧
    (s.form.rowsV1 ∨ (s.form.hiddenOutputs ∧ s.form.rowsV2OrSeg))

/-- **List decoding, the spec a decoder meets**: from opened columns at distinct positions, `decode` lists every trace
whose encoding agrees with at least a fraction `agree` of them. -/
def ListComplete {Col T : Type} [DecidableEq Col] (enc : T → ℕ → Col) (agree : ℝ) (decode : List (ℕ × Col) → List T) : Prop :=
  ∀ (os : List (ℕ × Col)) (M : T), (os.map Prod.fst).Nodup →
    agree * os.length ≤ (os.filter fun p => decide (enc M p.1 = p.2)).length → M ∈ decode os

/-- **Flock's assemble step**: the opened columns, decoded to candidate traces, each read for its witness; the first
candidate in the relation. -/
def assemble {Col T : Type} (H : Msg → Msg) (parse : Msg → Option Stmt) (decode : List (ℕ × Col) → List T)
    (read : T → Wit) (x : Msg) (opens : List (ℕ × Col)) : Option Wit :=
  match parse x with
  | none => none
  | some s => ((decode opens).map read).find? fun w => decide (RelS H s w)

/-- What the deployed verifier brings, read by Flock's claims: the verifier and whether it is the masked (`--zk`) one,
its parser and scope check, the opened columns of the first committed codeword in a run (`cols`), the code (`enc`), the
decoding threshold (`agree`), how a trace is read as a witness, and the numbers the errors read: the table's error
`ε_table`, `N₀`, the weight `C` of `γ`, `E^rep`'s waits `m′` and its truncation `K`. -/
structure Deploy (Coin Col T : Type) where
  zk : Bool
  V : Verifier Coin
  parse : Msg → Option Stmt
  scopeCheck : Msg → Bool
  cols : Seen → List (ℕ × Col)
  enc : T → ℕ → Col
  agree : ℝ
  read : T → Wit
  εtable : Msg → ℝ
  N₀ : Msg → ℝ
  weight : Msg → ℝ
  waits : Msg → ℕ
  cap : Msg → ℕ

section Claims

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

/-- `E^rep`'s accepted reruns per wait, `k = ⌈4·N₀/e⌉`. -/
noncomputable def Deploy.k (D : Deploy Coin Col T) (x : Msg) : ℕ := ⌈4 * D.N₀ x / Real.exp 1⌉₊

/-- **Flock's extractor**, `E^rep` at a decoder. -/
noncomputable def E (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) :
    Extractor Coin Wit D.V :=
  ERep D.V D.waits D.k D.cap D.cols (assemble H D.parse decode D.read) (check H D.parse)

/-- **Flock's knowledge error**: `κ = 4·ε_table + 2^{−m′} + 2k/K + e^{−k/8}`. -/
noncomputable def κ (D : Deploy Coin Col T) (x : Msg) : ℝ :=
  4 * D.εtable x + 2 ^ (-(D.waits x : ℝ)) + 2 * D.k x / D.cap x + Real.exp (-(D.k x : ℝ) / 8)

/-- **Flock's `γ`**: `16·(√(2·k(x)·C) + k(x))²`, with `k(x)` the verifier's rounds. -/
noncomputable def γ (D : Deploy Coin Col T) (x : Msg) : ℝ :=
  16 * (Real.sqrt (2 * D.V.k x * D.weight x) + D.V.k x) ^ 2

/-- Flock's scope, on the statement's bytes. -/
def Scope (D : Deploy Coin Col T) (x : Msg) : Prop := ∃ s, D.parse x = some s ∧ InScope D.zk s

/-- **Flock: KS**, at `E^rep` and every decoder that meets the list-decoding spec. -/
def KS (H : Msg → Msg) (D : Deploy Coin Col T) : Prop :=
  ∀ decode, ListComplete D.enc D.agree decode → Claims.KS H D.V (R H D.parse) (E H D decode) (κ D) (γ D)

/-- **Flock: Complete**, on the statements in scope. -/
def Complete {Ω : Type} (H : Msg → Msg) (D : Deploy Coin Col T) (P : Wit → Prover Ω) (δc : Msg → ℝ) : Prop :=
  Claims.Complete H D.V P (R H D.parse) (Scope D) δc

/-- **Flock: the readable scope is the verifier's.** -/
def ScopeMatches (D : Deploy Coin Col T) : Prop := Claims.ScopeMatches (Scope D) D.scopeCheck

/-- **Flock: zero knowledge** of the masked verifier's honest prover, against every malicious verifier. -/
def ZK {Ω : Type} [Fintype Ω] (H : Msg → Msg) (D : Deploy Coin Col T) (P : Wit → Prover Ω) (S : Simulator)
    (εzk : Msg → ℝ) : Prop :=
  D.zk = true ∧ ZKmal H P (R H D.parse) S εzk

/-- **Flock: bits**, `λ = min(log₂(1/κ), 255.5 − ½ log₂ γ) − 1`, at every decoder that meets the spec. -/
def Bits (H : Msg → Msg) (D : Deploy Coin Col T) : Prop :=
  ∀ decode, ListComplete D.enc D.agree decode →
    Claims.Bits H D.V (R H D.parse) (E H D decode) fun x => lamOf (κ D x) (γ D x)

/-- **Flock: efficiency**: `E^rep`'s reruns at every decoder, and the decoder's stated complexity. -/
def Efficient (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T)
    (counted : List (ℕ × Col) → List T × ℕ) (size : List (ℕ × Col) → ℕ) (p : Polynomial ℕ) : Prop :=
  ERepCost H D.V D.waits D.k D.cap D.cols (assemble H D.parse decode D.read) (check H D.parse) ∧
    DecoderComplexity decode counted size p

/-- **Flock: nontriviality**: recorded honest runs, each of a statement the scope check takes, that the verifier
accepts. -/
def Nontrivial (H : Msg → Msg) (D : Deploy Coin Col T) (recs : List (Recorded D.V)) : Prop :=
  Claims.Nontrivial H D.V recs ∧ ∀ e ∈ recs, D.scopeCheck e.x = true

end Claims

end Flock

end Claims
