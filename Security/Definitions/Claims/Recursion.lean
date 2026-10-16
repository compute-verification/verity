import Definitions.Claims

/-!
# Claims: the ZK-recursion compiler (a draft, for Daniel's read before it freezes)

The compiler is code plus proof, so it has no claims of its own to read. The claims below are its output's: the
compiled verifier's, parametric in the inner protocol (`Inner`, a `Deploy`-like record), the outer `--zk` Flock
(`Flock.Deploy`) and what the compiled verifier brings (`Deploy`). Shared definitions are `Claims`'; the template's
eight items:

1. **Statement format.** `x` is the canonical bytes of what the compiled verifier reads, parsed by its own parser
   (`Deploy.parse`) into the inner statement's bytes, the inner statement's dimensions as its parser reads them, the
   commitment to the circuit (the private-circuit variant only) and the recursion's form (`Stmt`, `Form`).
2. **Witness.** The inner protocol's (`W`), in both variants. In the private-circuit one the circuit is in the inner
   witness, which the universal unit's relation reads, and its commitment is hm96 rows that unit outputs.
3. **Relation (inherited).** `R(x, w) :⇔ x` parses, has no circuit commitment, and `R_I(x_I, w)` (`R`). The private
   variant: `x` parses with a circuit commitment, and `R_I(x_I, w)`, whose outputs bind it to the circuit (`RPriv`).
4. **Scope, readable** (`Scope`, `InScope`): the inner scope, and candidate 3's form (a fast100 inner table,
   `k_log ≥ 6`, two claims and two per region, no private circuit, a valid RecSparse phase at `T_LOG = 0` with
   `6 ≤ ℓ ≤ k_log` and its unit range inside a block, R at `table_m` with two claims). `ScopeMatches` states that it is
   the compiled verifier's own scope check.
5. **Named pieces.** `E*` (`E`): the outer extractor `E_O` (`EO`: per V* session, `E^rep` over forks of the judged run
   at the session's first round, with the outer Flock's assemble and check), then the inner one (`E^rep` at the inner
   protocol's waits, assemble and check), each inner rerun served by `E_O` at that rerun's coins and read off its output
   (`served`, `dec`). The simulator `Sim`: the inner session's commit strings to zero messages under fresh salts, then
   the outer simulator on each V* statement (`SO`, the named Flock simulator, a parameter as in `Flock.ZK`).
   The extraction-based finder `Be`: `B`'s fork, `E_O` on both runs, a collision among the SHA-512 evaluations their
   outer witnesses record (`Deploy.calls`), which no party computes in the clear.
6. **Claims.** `KS` with costs (`Efficient`), `Bits`, `ZK`, `Complete` on the scope, `ScopeMatches`, `Nontrivial`, and
   the private variant's `KSPriv`, `ZKPriv`; `At Flock` instantiates the inner protocol with Flock and its decoder.
7. **Assumptions used.** `StrictBirthday` (for `B`), `ExpectedCR` (for `Be`), each read once in `Bits`; `Hm96Hiding`
   (hm96 leaves hide up to `δ₁`), read in `ZK` and `ZKPriv`; and `trustedBase`. The decoders are those within Flock's
   list-decoding radius (`decOK`, `Flock.ListWithin`).
8. **Status.** A draft: nothing here is frozen, and every constant in `κ`, `γ`, `γe` and `cBe` is the draft's, fixed by
   the proof before freezing.

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

/-- The inner protocol's own `KS`, which the recursion's proof reads (a proved claim of the inner protocol). -/
def Inner.KS (H : Msg → Msg) (I : Inner Coin W) : Prop := Claims.KS H I.V I.R I.E I.κ I.γ

end InnerClaims

section OfFlock

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

/-- **Flock as the inner protocol**, at a decoder: its verifier, relation, scope, `E^rep` and errors. -/
noncomputable def Inner.ofFlock (H : Msg → Msg) (D : Flock.Deploy Coin Col T) (decode : List (ℕ × Col) → List T) :
    Inner Coin Flock.Wit where
  V := D.V
  R := Flock.R H D.parse
  Scope := Flock.Scope D
  Op := ℕ × Col
  opens := D.cols
  assemble := Flock.assemble H D.parse decode D.read
  check := Flock.check H D.parse
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

/-- **The recursion's form**, as the compiled verifier reads it (candidate 3, `RecVStar.Inner`): the inner table's `m`,
`k_log`, regions and claims, whether the inner circuits are private (`holo`), the RecSparse phase's shape
`(NU, ELL, S_LOG, T_LOG)`, R's shape `(m, claims)`, and whether the RecSparse unit range's slots lie inside the
block. -/
structure Form where
  m : ℕ
  kLog : ℕ
  regions : ℕ
  claims : ℕ
  holo : Bool
  sparse : Option (ℕ × ℕ × ℕ × ℕ)
  stacked : Option (ℕ × ℕ)
  unitRangeInBlock : Bool

/-- **The recursion's statement**: the inner statement's bytes, its `(m, k_log, regions)` as its own parser reads
them, the commitment to the circuit (the private-circuit variant), and the form. -/
structure Stmt where
  inner : Msg
  innerDims : ℕ × ℕ × ℕ
  circ : Option Msg
  form : Form

/-- RecSparse's `P_LOG` at `ℓ`: the bit length of `2ℓ + 1`. -/
def pLog (ell : ℕ) : ℕ := Nat.log 2 (2 * ell + 1) + 1

/-- **The recursion's scope, readable** (candidate 3): no circuit commitment and no private inner circuit; the form's
inner shape is the inner statement's, at a fast100 `m` (`22 ≤ m ≤ 35`), `k_log ≥ 6`, two claims and two per region;
a RecSparse phase at `NU ≥ 1`, `6 ≤ ℓ ≤ k_log`, `T_LOG = 0`, its unit range's slots inside the block; R at the phase's
`table_m = max(22, NU + P_LOG)` with a fast100 schedule and the two reps' claims. -/
def InScope (s : Stmt) : Prop :=
  s.circ = none ∧ s.form.holo = false ∧ (s.form.m, s.form.kLog, s.form.regions) = s.innerDims ∧
    22 ≤ s.form.m ∧ s.form.m ≤ 35 ∧ 6 ≤ s.form.kLog ∧ s.form.claims = 2 + 2 * s.form.regions ∧
    s.form.unitRangeInBlock = true ∧
    ∃ nu ell sLog, s.form.sparse = some (nu, ell, sLog, 0) ∧ 1 ≤ nu ∧ 6 ≤ ell ∧ ell ≤ s.form.kLog ∧
      s.form.stacked = some (max 22 (nu + pLog ell), 2) ∧ max 22 (nu + pLog ell) ≤ 35

/-! ## What the compiled verifier brings -/

/-- **An outer session's numbers**, those Flock's errors read at one V* statement (`Flock.Deploy`'s at it): the
table's error, `N₀`, the weight `C`, the outer verifier's rounds, `E^rep`'s waits and its truncation. -/
structure Nums where
  εtable : ℝ
  N₀ : ℝ
  weight : ℝ
  rounds : ℕ
  waits : ℕ
  cap : ℕ

/-- `E^rep`'s accepted reruns per wait, `k = ⌈4·N₀/e⌉`, as `Flock.Deploy.k`. -/
noncomputable def Nums.k (n : Nums) : ℕ := ⌈4 * n.N₀ / Real.exp 1⌉₊

/-- The outer session's knowledge error, as `Flock.κ`: `4·ε_table + 2^{−m′} + 2k/K + e^{−k/8}`. -/
noncomputable def Nums.κ (n : Nums) : ℝ :=
  4 * n.εtable + 2 ^ (-(n.waits : ℝ)) + 2 * n.k / n.cap + Real.exp (-(n.k : ℝ) / 8)

/-- The outer session's `γ`, as `Flock.γ`: `16·(√(2·k·C) + k)²` at its rounds `k`. -/
noncomputable def Nums.γ (n : Nums) : ℝ := 16 * (Real.sqrt (2 * n.rounds * n.weight) + n.rounds) ^ 2

/-- **What the compiled verifier brings**, read by the recursion's claims: the verifier (public-coin: each prover
message answered by its live coin, the decision the shipped verdict), its parser and scope check; the V* statements of
a run, from the inner session's public transcript (`vstar`), and their count at `x` (`nS`, fixed by `x`'s form); outer
session `s`'s view in a run's view (`session`: its messages and its verifier's decision as the compiled verifier's
answers show it); the inner session's view read off a run's coins and the outer witnesses (`innerView`); the SHA-512
evaluations the outer witnesses record (`calls`); the numbers of outer session `s` at `x` (`nums`, the outer Flock's
at that session's V* statements, which their public shape fixes); and, for the simulator, the inner round `j`'s commit
string, an hm96 leaf, to a zero message under a salt (`zeroCommit`), and the salts' length. -/
structure Deploy (Coin : Type) where
  V : Verifier Coin
  parse : Msg → Option Stmt
  scopeCheck : Msg → Bool
  vstar : Msg → List (Msg × Out) → List Msg
  nS : Msg → ℕ
  session : Msg → ℕ → Seen → Seen
  innerView : (x : Msg) → (Fin (V.k x) → Coin) → List Flock.Wit → Seen
  calls : Msg → List Flock.Wit → List (Msg × Msg)
  nums : Msg → ℕ → Nums
  zeroCommit : Msg → ℕ → Msg → OC Msg
  saltLen : ℕ

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
statement (read off the judged run's inner transcript), `E^rep` with the outer Flock's assemble and check over forks of
the judged run at the session's first round, each rerun seen as the session's view; all the sessions' witnesses, or
none. -/
noncomputable def EO (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) : Extractor Coin (List Flock.Wit) D.V where
  C x := (s : Fin (D.nS x)) → Fin (D.nums x s).waits → Fin (D.nums x s).cap → Fin (D.V.k x) → Coin
  ext x r₀ c := .rerun r₀ fun v₀ =>
    if v₀.out = .acc then
      let xs := D.vstar x (v₀.msgs.take (I.V.k (xI D x)))
      collect ((List.finRange (D.nS x)).map fun (s : Fin (D.nS x)) =>
        serve (viewing (D.session x s))
          (firstWait (D.nums x s).k DO.cols (Flock.assemble H DO.parse decO DO.read (xs.getD s []))
            (Flock.check H DO.parse (xs.getD s []))
            ((List.finRange (D.nums x s).waits).map fun i => (List.finRange (D.nums x s).cap).map fun j =>
              fork (I.V.k (xI D x) + ((xs.take s).map DO.V.k).sum) r₀ (c s i j))))
    else .ret none

/-- The inner session's view at a draw, read off the outer extractor's output; a rejected view when it gave up. -/
def dec (D : Deploy Coin) (x : Msg) (r : Fin (D.V.k x) → Coin) : Option (List Flock.Wit) → Seen
  | some ws => D.innerView x r ws
  | none => ⟨[], .rej⟩

/-- **`E*`, the recursion's extractor**: the outer extractor, then the inner one. The inner protocol's `E^rep` at its
inner statement, each of its reruns served by `E_O` at that draw's coins and `E_O`'s own, the inner view read off
`E_O`'s output (`dec`). -/
noncomputable def E (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) : Extractor Coin W D.V :=
  served D.V (EO H D I DO decO) (dec D) (fun x => I.waits (xI D x)) (fun x => I.k (xI D x))
    (fun x => I.cap (xI D x)) I.opens (fun x => I.assemble (xI D x)) (fun x => I.check (xI D x))

/-! ### The extraction-based finder -/

/-- **`Be`** (extraction-based): `B`'s fork (guess `j`, draw `ω, r_1..r_j`, run twice from the shared start with fresh
continuations), `E_O` on each run with fresh coins, and the first collision among the SHA-512 evaluations the two
runs' outer witnesses record (`Deploy.calls`: the inner verifier's and the commitments', which no party computes in the
clear). Its cost is both runs' calls, both parties', and both extractions'. -/
noncomputable def Be (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg)
    (hk : 0 < D.V.k x) : Finder where
  S := BCoins Ω Coin (D.V.k x) × (EO H D I DO decO).C x × (EO H D I DO decO).C x
  ne := ⟨(⟨⟨0, hk⟩, Classical.arbitrary _, fun _ => default, fun _ => default⟩, Classical.arbitrary _,
    Classical.arbitrary _)⟩
  out s :=
    findColl (D.calls x (((EO H D I DO decO).eval H P x s.1.2.1 s.1.2.2.1 s.2.1).1.getD []) ++
      D.calls x (((EO H D I DO decO).eval H P x s.1.2.1 (forkCoins s.1.1 s.1.2.2.1 s.1.2.2.2) s.2.2).1.getD []))
  cost s :=
    (run H P D.V x s.1.2.1 s.1.2.2.1).pLog.length + (run H P D.V x s.1.2.1 s.1.2.2.1).vLog.length +
      (run H P D.V x s.1.2.1 (forkCoins s.1.1 s.1.2.2.1 s.1.2.2.2)).pLog.length +
      (run H P D.V x s.1.2.1 (forkCoins s.1.1 s.1.2.2.1 s.1.2.2.2)).vLog.length +
      ((EO H D I DO decO).eval H P x s.1.2.1 s.1.2.2.1 s.2.1).2.1 +
      ((EO H D I DO decO).eval H P x s.1.2.1 (forkCoins s.1.1 s.1.2.2.1 s.1.2.2.2) s.2.2).2.1

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

/-- **`B`'s weight**: `γ = (N_I · Σ_s √(γ_O,s · k(x)) + 1)²`, with `k(x)` the compiled verifier's rounds (a session's
fork is one of `B`'s guesses) and the `1` for a collision among the verifier's calls in one run. -/
noncomputable def γ (D : Deploy Coin) (I : Inner Coin W) (x : Msg) : ℝ :=
  (NI D I x * ∑ s : Fin (D.nS x), Real.sqrt ((D.nums x s).γ * D.V.k x) + 1) ^ 2

/-- **`Be`'s weight**: `γ_e = (√(γ_I(x_I) · k(x)) + N_I + 1)²`: the inner `B`'s term, whose collisions are among calls
only the outer witnesses record, and the linear terms of an inner view that two extractions open differently and of a
collision among one run's recorded calls, folded in. -/
noncomputable def γe (D : Deploy Coin) (I : Inner Coin W) (x : Msg) : ℝ :=
  (Real.sqrt (I.γ (xI D x) * D.V.k x) + NI D I x + 1) ^ 2

/-- `Be`'s expected SHA-512 calls per call of the prover's run: two runs and two extractions, `E_O`'s at most
`1 + Σ_s m′_s·k_s` reruns in expectation. -/
noncomputable def cBe (D : Deploy Coin) (x : Msg) : ℕ :=
  4 + 2 * ∑ s : Fin (D.nS x), (D.nums x s).waits * (D.nums x s).k

/-- **The recursion's bits**: `λ = min(log₂(1/κ), 255.5 − ½ log₂ γ, 128 − ½ log₂(γ_e · c_Be)) − 2`. -/
noncomputable def lam (D : Deploy Coin) (I : Inner Coin W) (x : Msg) : ℝ :=
  min (min (Real.logb 2 (1 / κ D I x)) (255.5 - Real.logb 2 (γ D I x) / 2))
    (128 - Real.logb 2 (γe D I x * cBe D x) / 2) - 2

/-! ### The simulator -/

/-- The inner session simulated: `n` rounds of commit strings to zero messages under the salts, each put to the
verifier. -/
def simInner (D : Deploy Coin) (x : Msg) (salts : ℕ → Msg) : ℕ → List (Msg × Out) → SP (List (Msg × Out))
  | 0, ms => .ret ms
  | n + 1, ms => spBind (spOf (D.zeroCommit x ms.length (salts ms.length))) fun a =>
    .ask (ms.map Prod.fst ++ [a]) fun b =>
      if goesOn [(a, b)] then simInner D x salts n (ms ++ [(a, b)]) else .ret (ms ++ [(a, b)])

/-- The V* sessions simulated in order, each by the outer simulator at its statement, while the run goes on. -/
def simOuter {n : ℕ} (SO : Msg → ℕ → OSim) (x : Msg) (xs : List Msg) (c : (s : Fin n) → (SO x s).C) :
    List (Fin n) → List (Msg × Out) → SP (List (Msg × Out))
  | [], ms => .ret ms
  | s :: ss, ms =>
    if goesOn ms then
      spBind (spUnder (ms.map Prod.fst) ((SO x s).prog (xs.getD s []) (c s))) fun ts =>
        simOuter SO x xs c ss (ms ++ ts)
    else .ret ms

/-- **The recursion's simulator**: the inner session's commit strings to zero messages under fresh salts, then each V*
session by the outer simulator `SO` (the named Flock simulator) at its statement. -/
def Sim (D : Deploy Coin) (I : Inner Coin W) (SO : Msg → ℕ → OSim) : Simulator where
  C x := (Fin (I.V.k (xI D x)) → Fin D.saltLen → UInt8) × ((s : Fin (D.nS x)) → (SO x s).C)
  prog x c :=
    spBind (simInner D x (fun j => if h : j < I.V.k (xI D x) then List.ofFn (c.1 ⟨j, h⟩) else [])
        (I.V.k (xI D x)) []) fun ms =>
      simOuter SO x (D.vstar x ms) c.2 (List.finRange (D.nS x)) ms

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

/-- **The recursion: Complete**, on the statements in scope. -/
def Complete {Ω : Type} (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (P : W → Prover Ω)
    (δc : Msg → ℝ) : Prop :=
  Claims.Complete H D.V P (R D I) (Scope D I) δc

/-- **The recursion: the readable scope is the compiled verifier's.** -/
def ScopeMatches (D : Deploy Coin) (I : Inner Coin W) : Prop := Claims.ScopeMatches (Scope D I) D.scopeCheck

/-- **The recursion: zero knowledge** of the honest prover against every malicious verifier of the compiled verifier's
rounds (`ZKmalR` at `D.V.k`), at `Sim` over the outer `--zk` Flock's simulators: wherever hm96 leaves hide up to `δ₁`
(`Hm96Hiding` at the pinned key's salt mask `My` and salt hash `cy`), within `Σ_s ε_O(x, s) + 2·k_I(x_I)·δ₁`, with
`ε_O(x, s)` outer session `s`'s simulation error and `2δ₁` for each of the inner session's `k_I(x_I)` commit strings,
which `Sim` commits to zero. -/
def ZK {Ω Y B Cy : Type} [Fintype Ω] [Fintype Y] [Fintype B] (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W)
    (DO : Flock.Deploy Coin Col T) (P : W → Prover Ω) (SO : Msg → ℕ → OSim) (My : Y → B) (cy : Y → Cy)
    (εO : Msg → ℕ → ℝ) : Prop :=
  DO.zk = true ∧ ∀ δ₁ : ℝ≥0, Hm96Hiding My cy δ₁ →
    ZKmalR H D.V.k P (R D I) (Sim D I SO) fun x =>
      ∑ s : Fin (D.nS x), εO x s + 2 * I.V.k (xI D x) * (δ₁ : ℝ)

/-- **The recursion: nontriviality**: recorded honest runs, each of a statement the scope check takes, that the compiled
verifier accepts. -/
def Nontrivial (H : Msg → Msg) (D : Deploy Coin) (recs : List (Recorded D.V)) : Prop :=
  Claims.Nontrivial H D.V recs ∧ ∀ e ∈ recs, D.scopeCheck e.x = true

/-- **The private variant: KS**, at `E*` and the same errors. -/
def KSPriv [Zero Col] (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T) : Prop :=
  ∀ decO, decOK DO decO → ∀ (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω),
    PrBad H D.V (RPriv D I) (E H D I DO decO) x P ≤
      κ D I x + Real.sqrt (γ D I x * PrB H D.V P x) + Real.sqrt (γe D I x * PrBe H D I DO decO P x)

/-- **The private variant: zero knowledge**, at a simulator that commits to a fixed circuit of the class, against every
malicious verifier of the compiled verifier's rounds: wherever hm96 leaves hide up to `δ₁`, within `ε_zk(δ₁, x)`. -/
def ZKPriv {Ω Y B Cy : Type} [Fintype Ω] [Fintype Y] [Fintype B] (H : Msg → Msg) (D : Deploy Coin)
    (I : Inner Coin W) (DO : Flock.Deploy Coin Col T) (P : W → Prover Ω) (S : Simulator) (My : Y → B) (cy : Y → Cy)
    (εzk : ℝ≥0 → Msg → ℝ) : Prop :=
  DO.zk = true ∧ ∀ δ₁ : ℝ≥0, Hm96Hiding My cy δ₁ → ZKmalR H D.V.k P (RPriv D I) S (εzk δ₁)

end Recursion

/-! ## At Flock: the inner protocol is Flock -/

section AtFlock

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

/-- **KS of the recursion over Flock**, at every inner and outer decoder that meets the spec. -/
def KSFlock [Zero Col] (H : Msg → Msg) (D : Deploy Coin) (DI DO : Flock.Deploy Coin Col T) : Prop :=
  ∀ decI, decOK DI decI → KS H D (Inner.ofFlock H DI decI) DO

/-- **Bits of the recursion over Flock**, at every inner decoder that meets the spec. -/
def BitsFlock [Zero Col] (H : Msg → Msg) (D : Deploy Coin) (DI DO : Flock.Deploy Coin Col T) : Prop :=
  ∀ decI, decOK DI decI → Bits H D (Inner.ofFlock H DI decI) DO

end AtFlock

end Claims.Recursion
