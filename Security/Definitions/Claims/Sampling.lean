import Definitions.Claims

/-!
# Claims: the sampling compiler (frozen before any proof work)

The claims block of sampled proofs (`verity/core/protocols/sampled_proofs/`), one-stage and two-stage sampling, on the
shared definitions of `Definitions.Claims`, in the template's eight items and the README glossary's terms. Parametric in
the deployed sampled verifier (`Deploy`), as Flock's block is in `Flock.Deploy`: binding it to the compiled verifier is
the code side's.

1. **Statement** (`Stmt`): a hiding commitment to the partition, the partition's public properties (its size class, and
   the sampling design with the population's shape: the proof units' strata, or under two-stage sampling each replay-unit
   class's strata, replay-unit sizes, first stage and check), and the roots the trace is committed under. `x` is the
   canonical bytes of what the verifier reads before it draws, and `parse` is its own.
2. **Witness** (`Wit`): openings of replay-unit descriptions at their slots under the partition's commitment, and of
   committed values at their positions under the trace's roots. One-stage sampling is the case where every replay unit is
   one proof unit.
3. **Relation** (`R δ`): the set of faulty units of the trace the openings define is in the confidence set
   `M_δ = {B : P_a(B) > δ}`. A unit is faulty when its committed outputs differ from its gates applied to its committed
   inputs; an unopened unit counts as faulty. Under two-stage sampling the units are the replay units, whose interiors
   are committed after the first draw, and `P_a` is its effective form `P̄_a`.
4. **Scope** (`InScope`, `ScopeMatches`): a size class of at most 16 output bits, and a well-formed design.
5. **Named pieces**: the commitment (`Commitment`, `Scheme`, `Commitment.Binds`), the universal unit (`Netlist`,
   `Universal`), the acceptance probabilities (`OneStage.Pa`, `RUClass.Pa`, `Design.Pa`), the extractor (`E`): `k`
   accepted runs, in each `k` accepted reruns of the second stage, in each Flock's `E^rep` on every checked unit; the
   expected-time finder (`Bext`); and the parameters (`Deploy`).
6. **Claims**: `KS` at every `δ ≥ 0`, with error `δ + ε_ks + ε_bind`, `ε_ks` Flock's at the units' sessions and
   `ε_bind = Pr[B_ext]`; its count form `KSCount`; `Binding`; `Universal`; `Complete`; `ScopeMatches`; `ZK`; `Bits`;
   `Efficient`; and `Nontrivial`.
7. **Assumptions used**: `StrictBirthday` (for `B`), `ExpectedCR` (for `Bext`), `CommitHides` (for zero knowledge given
   the leaf layer, at the deployed hm96 leaves), and `trustedBase`.
8. **Status**: stated; no claim is proved.

Generic pieces this block adds, which move to `Definitions.Claims` in a later change: `LogColl`, `ofOC`, `ownLog`,
`flatMapRP`, `waitAccC`, `forkOn`, `Bext`, `lamLin`, `BitsLin`, `ExtractOwnLe`, `SimAux` and `ZKaux`.

Frozen by `Flock.Guarantees.SamplingClaimsStated` (`Models/Flock/Guarantees/SamplingClaims.lean`), whose statement
reads every claim here: the lock records each one's text under `reads`, so a changed claim is a changed record.
-/

namespace Claims.Sampling

/-! ## The commitment -/

/-- Two different inputs with one output in a log of `H` calls. -/
def LogColl (l : List (Msg × Msg)) : Prop := ∃ p ∈ l, ∃ p' ∈ l, p.1 ≠ p'.1 ∧ p.2 = p'.2

/-- An opening: the committed value `val` at position `pos`, and the opening's data. -/
structure Opening (α : Type) where
  pos : ℕ
  val : α
  data : Msg

/-- **The commitment**, a named primitive: a hiding commitment to values at positions under public roots, with its
opening check `opens` and the hiding leaves an opening shows (`leaves`). The deployed instance is hm96 leaves in
frame-v3-sha512 trees; its storage layout is behind `opens`. -/
structure Commitment (α : Type) where
  opens : List Msg → Opening α → OC Bool
  leaves : (Msg → Msg) → Opening α → List Msg

/-- Whether `o` opens under `roots`. -/
def Commitment.Opens {α : Type} (H : Msg → Msg) (Cm : Commitment α) (roots : List Msg) (o : Opening α) : Bool :=
  ((Cm.opens roots o).eval H).1

/-- **Binding, as a collision reduction**: two openings that pass at one position under the same roots, of different
values, make `H` calls that collide. -/
def Commitment.Binds {α : Type} (H : Msg → Msg) (Cm : Commitment α) : Prop :=
  ∀ (roots : List Msg) (o o' : Opening α), o.pos = o'.pos → o.val ≠ o'.val → Cm.Opens H roots o = true →
    Cm.Opens H roots o' = true → LogColl (((Cm.opens roots o).eval H).2 ++ ((Cm.opens roots o').eval H).2)

/-! ## Proof units and the universal unit -/

/-- A size class `(G, N_IN, N_OUT)`: `G` gates over `N_IN` inputs, and `N_OUT` output bits. -/
structure SizeClass where
  G : ℕ
  nIn : ℕ
  nOut : ℕ

/-- `K`: the bits of a wire index, enough for the `W = N_IN + G` wires, at least one. -/
def SizeClass.indexBits (c : SizeClass) : ℕ := Nat.log2 (c.nIn + c.G - 1) + 1

/-- **A proof unit's gates**, as the universal unit takes them (`verity.ml.private_circuits.universal`): SSA wires, the
`nIn` inputs and then one per gate. Gate `(op, a, b)` reads wires `a` and `b` and computes XOR, AND, XNOR or NAND
(`op`'s bit 0 picks AND, bit 1 negates); a wire at or past the reading gate's own reads `0`, as does an output past the
last wire. -/
structure Netlist where
  nIn : ℕ
  gates : List (Fin 4 × ℕ × ℕ)
  outs : List ℕ

/-- The netlist run gate by gate on `x` (`interpret`). -/
def Netlist.eval (n : Netlist) (x : List Bool) : List Bool :=
  let ws := n.gates.foldl (fun ws g =>
    let a := ws.getD g.2.1 false
    let b := ws.getD g.2.2 false
    ws ++ [xor (if g.1.val % 2 = 1 then a && b else xor a b) (decide (g.1.val / 2 = 1))]) x
  n.outs.map fun o => ws.getD o false

/-- The netlist is of class `c`: `G` gates over `N_IN` inputs, `N_OUT` outputs, every index in `K` bits. -/
def Netlist.Fits (c : SizeClass) (n : Netlist) : Prop :=
  n.nIn = c.nIn ∧ n.gates.length = c.G ∧ n.outs.length = c.nOut ∧
    (∀ g ∈ n.gates, g.2.1 < 2 ^ c.indexBits ∧ g.2.2 < 2 ^ c.indexBits) ∧ ∀ o ∈ n.outs, o < 2 ^ c.indexBits

/-- `v` in `k` bits, low bit first. -/
def bitsLE (k v : ℕ) : List Bool := (List.range k).map v.testBit

/-- The netlist as the universal unit's `(op, src, out)` arguments, in leaf order (`arguments`). -/
def Netlist.args (c : SizeClass) (n : Netlist) : List Bool :=
  n.gates.flatMap (fun g => bitsLE 2 g.1.val) ++
    n.gates.flatMap (fun g => bitsLE c.indexBits g.2.1 ++ bitsLE c.indexBits g.2.2) ++
    n.outs.flatMap (bitsLE c.indexBits)

/-- Where a proof unit's input or output bit is: a committed value of the trace at a position under the statement's
roots (`bd`), or interior wire `j` of its replay unit, committed after the first draw (`int`). -/
inductive Src where
  | bd (p : ℕ)
  | int (j : ℕ)

/-- **A proof unit**, as its replay unit's description gives it: its private gates, and where its inputs and outputs
are. -/
structure ProofUnit where
  gates : Netlist
  inp : List Src
  out : List Src

/-- The unit is of class `c`: its gates are, with `N_IN` inputs and `N_OUT` outputs. -/
def ProofUnit.Fits (c : SizeClass) (u : ProofUnit) : Prop :=
  u.gates.Fits c ∧ u.inp.length = c.nIn ∧ u.out.length = c.nOut

/-- **A replay unit**: its proof units, in order. Under one-stage sampling each replay unit is one proof unit. -/
abbrev ReplayUnit := List ProofUnit

/-! ## Sampling designs -/

/-- **A one-stage sampling design** over a population in strata: a uniform `K`-subset (`subset:K`); each member
independently at rate `num/den` (`bernoulli:NUM/DEN`); a uniform `k_s`-subset of each stratum, independently, with
`k_s = max(1, round(K·n_s/N))` (`stratified:K`), or `k_s = min(n_s, max(f_s, ⌈K·w_s·n_s/W⌉))` from each stratum's work
`w_s` and floor `f_s`, `W = Σ w_s·n_s` (`work:K`). -/
inductive OneStage where
  | subset (K : ℕ)
  | bernoulli (num den : ℕ)
  | stratified (K : ℕ)
  | work (K : ℕ) (w f : List ℕ)

/-- `B`'s members in `[o, o + n)`, shifted to `[0, n)`. -/
def window (o n : ℕ) (B : Finset ℕ) : Finset ℕ := (B.filter fun v => o ≤ v ∧ v < o + n).image (· - o)

/-- The probability that a uniform `k_s`-subset of each stratum `(n_s, k_s)`, from offset `o`, misses `B`:
`∏_s C(n_s − |B ∩ s|, k_s)/C(n_s, k_s)`. -/
noncomputable def strataPa : List (ℕ × ℕ) → ℕ → Finset ℕ → ℝ
  | [], _, _ => 1
  | (n, k) :: ss, o, B => ((n - (window o n B).card).choose k : ℝ) / (n.choose k : ℝ) * strataPa ss (o + n) B

/-- **The acceptance probability** `P_a(B)` of a one-stage design over strata of sizes `ns`: the probability that its
sample holds no member of `B`. -/
noncomputable def OneStage.Pa (ns : List ℕ) : OneStage → Finset ℕ → ℝ
  | .subset K, B => ((ns.sum - B.card).choose K : ℝ) / (ns.sum.choose K : ℝ)
  | .bernoulli num den, B => (((den - num : ℕ) : ℝ) / den) ^ B.card
  | .stratified K, B => strataPa (ns.map fun n => (n, max 1 ((2 * K * n + ns.sum) / (2 * ns.sum)))) 0 B
  | .work K w f, B =>
    let W := ((ns.zip w).map fun p => p.1 * p.2).sum
    strataPa ((ns.zip (w.zip f)).map fun p => (p.1, min p.1 (max p.2.2 ((K * p.2.1 * p.1 + W - 1) / W)))) 0 B

/-- **A replay-unit class under two-stage sampling**: its RUs' strata, each RU's count of proof units, the first stage's
design over its RUs, and how many proof units of each drawn RU are checked (`none`: all of them). -/
structure RUClass where
  strata : List ℕ
  sizes : List ℕ
  first : OneStage
  check : Option ℕ

/-- **The effective acceptance probability** `P̄_a(B)` of a class over a set `B` of its RUs (`effEscape`). A faulty RU
has a faulty proof unit under every interior the prover commits after the first draw, so a drawn one escapes when no
faulty proof unit is checked: under a Bernoulli first stage at rate `p` checking `k` of an RU's `n_v`,
`∏_{v ∈ B} (1 − p·min(k, n_v)/n_v)` (`nested_escape`); when every proof unit of a drawn RU is checked, the first stage's
`P_a(B)` (`effEscape_full`). -/
noncomputable def RUClass.Pa (c : RUClass) (B : Finset ℕ) : ℝ :=
  match c.first, c.check with
  | .bernoulli num den, some k =>
    ∏ v ∈ B, (1 - (num : ℝ) / den * ((min k (c.sizes.getD v 0) : ℕ) : ℝ) / (c.sizes.getD v 0 : ℝ))
  | d, _ => d.Pa c.strata B

/-- **The sampling design** of a statement: one stage over the proof units, in strata of the given sizes; or two-stage
sampling over replay-unit classes, which draw independently. -/
inductive Design where
  | oneStage (strata : List ℕ) (d : OneStage)
  | twoStage (classes : List RUClass)

/-- Each replay unit's count of proof units, slot by slot: one each under one-stage sampling. -/
def Design.sizes : Design → List ℕ
  | .oneStage ns _ => List.replicate ns.sum 1
  | .twoStage cs => cs.flatMap RUClass.sizes

/-- Whether interiors are committed after the first draw. -/
def Design.twoStaged : Design → Bool
  | .oneStage _ _ => false
  | .twoStage _ => true

/-- The classes' effective acceptance probabilities, from slot `o`, multiplied. -/
noncomputable def classesPa : List RUClass → ℕ → Finset ℕ → ℝ
  | [], _, _ => 1
  | c :: cs, o, B => c.Pa (window o c.sizes.length B) * classesPa cs (o + c.sizes.length) B

/-- **The acceptance probability** `P_a(B)` of the design, over a set `B` of replay-unit slots; under two-stage sampling
its effective form `P̄_a`. -/
noncomputable def Design.Pa : Design → Finset ℕ → ℝ
  | .oneStage ns d, B => d.Pa ns B
  | .twoStage cs, B => classesPa cs 0 B

/-- **The confidence set** `M_δ`: every set `B` with `P_a(B) > δ`. -/
def Design.M (d : Design) (δ : ℝ) : Set (Finset ℕ) := {B | δ < d.Pa B}

/-- **The miss at `t`**: the largest acceptance probability of `t` or more of the units. -/
noncomputable def Design.miss (d : Design) (t : ℕ) : ℝ :=
  ⨆ B : {B : Finset ℕ // B ⊆ Finset.range d.sizes.length ∧ t ≤ B.card}, d.Pa B.1

/-! ## 1. Statement -/

/-- **The sampling statement**: a hiding commitment to the partition (`part`), its public properties (its size class,
and its sampling design with the population's shape), and the roots the trace is committed under. -/
structure Stmt where
  part : Msg
  cls : SizeClass
  design : Design
  roots : List Msg

/-! ## 2. Witness -/

/-- **The sampling witness**: openings of replay-unit descriptions at their slots, and of committed values at their
positions. -/
structure Wit where
  rus : List (Opening ReplayUnit)
  vals : List (Opening Bool)

/-! ## 3. Relation -/

/-- The commitment scheme: the partition's commitment, to each replay unit's description at its slot under the
partition's root, and the trace's, to each committed value at its position under the trace's roots. -/
structure Scheme where
  part : Commitment ReplayUnit
  trace : Commitment Bool

/-- The replay unit `w` opens at slot `v`: its first opening there that passes. -/
def Wit.ru (H : Msg → Msg) (Sc : Scheme) (s : Stmt) (w : Wit) (v : ℕ) : Option ReplayUnit :=
  (w.rus.find? fun o => o.pos == v && Sc.part.Opens H [s.part] o).map (·.val)

/-- The committed value `w` opens at position `p`: its first opening there that passes. -/
def Wit.val (H : Msg → Msg) (Sc : Scheme) (s : Stmt) (w : Wit) (p : ℕ) : Option Bool :=
  (w.vals.find? fun o => o.pos == p && Sc.trace.Opens H s.roots o).map (·.val)

/-- A source's value under `w` and the interior values `ι`; under one-stage sampling there are no interiors. -/
def srcVal (H : Msg → Msg) (Sc : Scheme) (s : Stmt) (w : Wit) (ι : ℕ → Bool) : Src → Option Bool
  | .bd p => w.val H Sc s p
  | .int j => if s.design.twoStaged then some (ι j) else none

/-- A proof unit maps its inputs to its outputs under `val`, every one of them opened. -/
def ProofUnit.Correct (val : Src → Option Bool) (u : ProofUnit) : Prop :=
  ∃ xs, u.inp.mapM val = some xs ∧ u.out.mapM val = some (u.gates.eval xs)

/-- **A faulty unit** at slot `v` under `w`: unless `w` opens a replay unit there of the statement's size whose proof
units fit the size class, and some interior values make every proof unit map its committed inputs to its committed
outputs. For a proof unit (one-stage) that is its committed outputs differing from its gates applied to its committed
inputs; for a replay unit whose interior each proof unit writes before any reads it, its gates applied to its committed
inputs. -/
def Faulty (H : Msg → Msg) (Sc : Scheme) (s : Stmt) (w : Wit) (v : ℕ) : Prop :=
  ¬ ∃ d, w.ru H Sc s v = some d ∧ d.length = s.design.sizes.getD v 0 ∧ (∀ u ∈ d, u.Fits s.cls) ∧
    ∃ ι : ℕ → Bool, ∀ u ∈ d, u.Correct (srcVal H Sc s w ι)

open Classical in
/-- **The faulty set** of the trace `w`'s openings define. -/
noncomputable def faulty (H : Msg → Msg) (Sc : Scheme) (s : Stmt) (w : Wit) : Finset ℕ :=
  (Finset.range s.design.sizes.length).filter (Faulty H Sc s w)

/-- **The sampling relation** at `δ`, on a parsed statement: the faulty set is in `M_δ`. -/
def RelS (H : Msg → Msg) (Sc : Scheme) (δ : ℝ) (s : Stmt) (w : Wit) : Prop := faulty H Sc s w ∈ s.design.M δ

/-- **The sampling relation** at `δ`, on the statement's bytes: it parses (`parse`, the verifier's own) and `RelS`
holds. -/
def R (H : Msg → Msg) (Sc : Scheme) (parse : Msg → Option Stmt) (δ : ℝ) : Rel Wit :=
  fun x w => ∃ s, parse x = some s ∧ RelS H Sc δ s w

/-- The count form: fewer than `t` units are faulty. -/
def RCount (H : Msg → Msg) (Sc : Scheme) (parse : Msg → Option Stmt) (t : ℕ) : Rel Wit :=
  fun x w => ∃ s, parse x = some s ∧ (faulty H Sc s w).card < t

/-- The honest relation: no unit is faulty. -/
def Rhon (H : Msg → Msg) (Sc : Scheme) (parse : Msg → Option Stmt) : Rel Wit :=
  fun x w => ∃ s, parse x = some s ∧ faulty H Sc s w = ∅

/-! ## 4. Scope -/

/-- A one-stage design is well formed over strata of sizes `ns`. -/
def OneStage.WellFormed (ns : List ℕ) : OneStage → Prop
  | .subset K => 1 ≤ K ∧ K ≤ ns.sum
  | .bernoulli num den => 0 < den ∧ num ≤ den
  | .stratified K => 1 ≤ K ∧ K ≤ ns.sum ∧ ∀ n ∈ ns, 1 ≤ n
  | .work _ w f => w.length = ns.length ∧ f.length = ns.length ∧ (∀ x ∈ f, 1 ≤ x) ∧
      0 < ((ns.zip w).map fun p => p.1 * p.2).sum ∧ ∀ n ∈ ns, 1 ≤ n

/-- A class is well formed: one size per RU, a well-formed first stage over its RUs, every RU at least one proof unit,
and a check of `k ≥ 1` proof units only under a Bernoulli first stage. -/
def RUClass.WellFormed (c : RUClass) : Prop :=
  c.sizes.length = c.strata.sum ∧ c.first.WellFormed c.strata ∧ (∀ n ∈ c.sizes, 1 ≤ n) ∧
    ∀ k, c.check = some k → 1 ≤ k ∧ ∃ num den, c.first = .bernoulli num den

def Design.WellFormed : Design → Prop
  | .oneStage ns d => d.WellFormed ns
  | .twoStage cs => ∀ c ∈ cs, c.WellFormed

/-- **The sampling scope, readable**: a size class of at least one gate, input and output, and at most 16 output bits
(the literal limit), and a well-formed design. -/
def InScope (s : Stmt) : Prop :=
  1 ≤ s.cls.G ∧ 1 ≤ s.cls.nIn ∧ 1 ≤ s.cls.nOut ∧ s.cls.nOut ≤ 16 ∧ s.design.WellFormed

/-! ## 5. Named pieces -/

/-- What a unit's Flock witness reveals: a replay unit's description at its slot, or a committed value under the roots
it is committed under (the statement's, or the interior roots the run showed). -/
inductive Revealed where
  | ru (o : Opening ReplayUnit)
  | val (roots : List Msg) (o : Opening Bool)

/-- What the deployed sampled verifier brings, read by the claims: the verifier, its parser and scope check, the
commitment scheme, the universal unit of each size class (the public Program each unit's session proves, run on the
unit's private gates over its committed inputs and outputs), and the Flock verifier its sessions run (`flock`; the claims
read its parser, columns, code, decoding threshold and count, witness reading and masking, not its numbers).

The draw is `V`'s, from its coins, in this layout: `stage2` holds the rounds after the first stage (the proof units'
draw under two-stage sampling, and the sessions), so runs that keep the other rounds make the same first draw and see the
same interior commitments; `checked` is the proof units the run checks, each as its RU's slot and its index there;
`rounds` is each one's session's rounds; `session` is its Flock statement and what its session showed; `reveal` is what
its Flock witness reveals.

The numbers: the session's `E^rep` makes `m′` waits (`innerWaits`) of `K′` reruns (`innerCap`) with `N₀` (`innerN₀`),
`ε_table` and the weight `C` of `γ`, each good for every session at `x`; `openCalls` bounds the SHA-512 calls of checking
what one session's witness reveals. -/
structure Deploy (Coin Col T : Type) where
  V : Verifier Coin
  parse : Msg → Option Stmt
  scopeCheck : Msg → Bool
  scheme : Scheme
  universal : SizeClass → Circuit
  flock : Flock.Deploy Coin Col T
  stage2 : (x : Msg) → Finset (Fin (V.k x))
  checked : (x : Msg) → (Fin (V.k x) → Coin) → List (ℕ × ℕ)
  rounds : (x : Msg) → ℕ × ℕ → Finset (Fin (V.k x))
  session : Msg → ℕ × ℕ → Seen → Option (Msg × Seen)
  reveal : Stmt → ℕ × ℕ → Seen → Flock.Wit → List Revealed
  innerWaits : Msg → ℕ
  innerN₀ : Msg → ℝ
  innerCap : Msg → ℕ
  εtable : Msg → ℝ
  weight : Msg → ℝ
  openCalls : Msg → ℕ

/-! ### Rerun programs -/

/-- An oracle program as a rerun program: its `H` calls are the extractor's own. -/
def ofOC {R α : Type} : OC α → RP R α
  | .ret a => .ret a
  | .query q k => .hash q fun a => ofOC (k a)

/-- The `H` calls a rerun program makes itself, in order, as `(query, answer)`: not its reruns'. -/
def ownLog {R α : Type} (H : Msg → Msg) (run : R → Trace) : RP R α → List (Msg × Msg)
  | .ret _ => []
  | .hash q k => (q, H q) :: ownLog H run (k (H q))
  | .rerun r k => ownLog H run (k (run r).seen)

/-- Run `f` on each element in order, concatenating the results. -/
def flatMapRP {R β γ : Type} (f : β → RP R (List γ)) : List β → RP R (List γ)
  | [] => .ret []
  | b :: bs => (f b).bind fun cs => (flatMapRP f bs).bind fun ds => .ret (cs ++ ds)

/-- `waitAcc`, keeping each accepted rerun's coins with what it showed. -/
def waitAccC {R : Type} (k : ℕ) : List R → List (R × Seen) → RP R (List (R × Seen))
  | [], acc => .ret acc
  | r :: rs, acc =>
    if k ≤ acc.length then .ret acc
    else .rerun r fun v => waitAccC k rs (if v.out = .acc then acc ++ [(r, v)] else acc)

/-- The coins `d` on the rounds `I`, and `r₀`'s elsewhere. -/
def forkOn {k : ℕ} {Coin : Type} (I : Finset (Fin k)) (r₀ d : Fin k → Coin) : Fin k → Coin :=
  fun t => if t ∈ I then d t else r₀ t

/-! ### The extractor -/

section Extract

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

/-- The session's `E^rep` waits for `k′ = ⌈4·N₀/e⌉` accepted reruns. -/
noncomputable def Deploy.innerK (D : Deploy Coin Col T) (x : Msg) : ℕ := ⌈4 * D.innerN₀ x / Real.exp 1⌉₊

/-- The replay units at `x`, `0` on a statement that doesn't parse. -/
def nRU (D : Deploy Coin Col T) (x : Msg) : ℕ :=
  match D.parse x with
  | some s => s.design.sizes.length
  | none => 0

/-- The proof units at `x`, `0` on a statement that doesn't parse. -/
def nPU (D : Deploy Coin Col T) (x : Msg) : ℕ :=
  match D.parse x with
  | some s => s.design.sizes.sum
  | none => 0

/-- **The extractor's waits** stop at `k = 2^40·(n_RU + n_PU + 1)` accepted runs. A union over the minimal sets of the
complement of `M_δ`, at most `2^n` for `n` units, spread over `k` accepted runs, costs the factor `2^{n/k}`; this `k`
makes it at most `2^{2^{−40}}`, while `k` enters the extractor's cost, and so the hash side of `Bits`, only through
`log₂ k`. -/
def Deploy.k (D : Deploy Coin Col T) (x : Msg) : ℕ := 2 ^ 40 * (nRU D x + nPU D x + 1)

/-- The waits give up after `K = 2^128·k` reruns. -/
def Deploy.cap (D : Deploy Coin Col T) (x : Msg) : ℕ := 2 ^ 128 * D.k x

/-- The opened columns of a checked unit's session. -/
def Deploy.sessionCols (D : Deploy Coin Col T) (x : Msg) (u : ℕ × ℕ) (v : Seen) : List (ℕ × Col) :=
  match D.session x u v with
  | some (_, vS) => D.flock.cols vS
  | none => []

/-- The coins of one session's `E^rep`. -/
abbrev Deploy.UnitCoins (D : Deploy Coin Col T) (x : Msg) := Fin (D.innerWaits x) → Fin (D.innerCap x) → Fin (D.V.k x) → Coin

/-- What passes its opening check among `rs`, each checked by the scheme, its `H` calls the extractor's own. -/
def checkRevealed {R : Type} (Sc : Scheme) (s : Stmt) : List Revealed → RP R (List Revealed) :=
  flatMapRP fun
    | .ru o => (ofOC (Sc.part.opens [s.part] o)).bind fun ok => .ret (if ok then [.ru o] else [])
    | .val rs o => (ofOC (Sc.trace.opens rs o)).bind fun ok => .ret (if ok then [.val rs o] else [])

/-- **A checked unit's extraction at an accepted run** `(r, v)`: Flock's `E^rep` on the unit's session, at a decoder,
without its replay (the run accepted), its reruns fresh on the session's rounds and `r`'s elsewhere, so the prover makes
the same session; then what its witness reveals that passes its opening check. -/
noncomputable def unitExt (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) (x : Msg)
    (s : Stmt) (u : ℕ × ℕ) (r : Fin (D.V.k x) → Coin) (v : Seen) (ρ : D.UnitCoins x) :
    RP (Fin (D.V.k x) → Coin) (List Revealed) :=
  match D.session x u v with
  | none => .ret []
  | some (xS, _) =>
    (firstWait (D.innerK x) (D.sessionCols x u) (Flock.assemble H D.flock.parse decode D.flock.read xS)
        (Flock.check H D.flock.parse xS)
        ((List.finRange (D.innerWaits x)).map fun i =>
          (List.finRange (D.innerCap x)).map fun j => forkOn (D.rounds x u) r (ρ i j))).bind fun
      | none => .ret []
      | some wF => checkRevealed D.scheme s (D.reveal s u v wF)

/-- **The assemble step**: the replay units' openings, and the committed values' openings under the statement's
roots. -/
def assemble (s : Stmt) (found : List Revealed) : Wit where
  rus := found.filterMap fun
    | .ru o => some o
    | .val _ _ => none
  vals := found.filterMap fun
    | .val rs o => if rs = s.roots then some o else none
    | .ru _ => none

/-- The extractor's coins: the outer wait's `K` coin vectors; for each of its `k` accepted runs, the inner wait's `K`
(used fresh on the second stage); and for each accepted inner run and each of at most `n_PU` checked units, its
`E^rep`'s. -/
abbrev ECoins (D : Deploy Coin Col T) (x : Msg) :=
  (Fin (D.cap x) → Fin (D.V.k x) → Coin) × (Fin (D.k x) → Fin (D.cap x) → Fin (D.V.k x) → Coin) ×
    (Fin (D.k x) × Fin (D.k x) × Fin (nPU D x) → D.UnitCoins x)

/-- **The sampling extractor** at a decoder: replay the judged run; if it accepted, rerun on fresh coins until `k`
accepted runs are in hand; at each, rerun fresh on the second stage until `k` accepted runs are in hand (each wait giving
up after `K`); at each of those, extract every checked unit; and assemble. -/
noncomputable def E (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) :
    Extractor Coin Wit D.V where
  C x := ECoins D x
  ext x r₀ c := .rerun r₀ fun v₀ =>
    match D.parse x with
    | none => .ret none
    | some s =>
      if v₀.out = .acc then
        (waitAccC (D.k x) ((List.finRange (D.cap x)).map c.1) []).bind fun outer =>
          (flatMapRP (fun jo : Fin (D.k x) × ((Fin (D.V.k x) → Coin) × Seen) =>
              (waitAccC (D.k x) ((List.finRange (D.cap x)).map fun l => forkOn (D.stage2 x) jo.2.1 (c.2.1 jo.1 l))
                  []).bind fun inner =>
                flatMapRP (fun li : Fin (D.k x) × ((Fin (D.V.k x) → Coin) × Seen) =>
                    flatMapRP (fun iu : Fin (nPU D x) × (ℕ × ℕ) =>
                        unitExt H D decode x s iu.2 li.2.1 li.2.2 (c.2.2 (jo.1, li.1, iu.1)))
                      ((List.finRange (nPU D x)).zip (D.checked x li.2.1)))
                  ((List.finRange (D.k x)).zip inner))
            ((List.finRange (D.k x)).zip outer)).bind fun found => .ret (some (assemble s found))
      else .ret none

end Extract

/-! ### The expected-time finder, bits with `β > 0`, and the extractor's own hashing -/

section Finder

variable {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **`B_ext`** (expected time): `B`'s fork, the extractor run in each branch on that branch's run, and the first
collision among the `H` calls the two extractions make themselves (the openings they check). Its cost is both
extractions' SHA-512 calls, their reruns' included. -/
noncomputable def Bext (H : Msg → Msg) (V : Verifier Coin) (E : Extractor Coin W V) : FinderFamily :=
  fun Ω _ _ P x =>
    if hk : 0 < V.k x then
      { S := BCoins Ω Coin (V.k x) × E.C x × E.C x
        ne := ⟨(⟨⟨0, hk⟩, Classical.arbitrary _, fun _ => default, fun _ => default⟩, Classical.arbitrary _,
          Classical.arbitrary _)⟩
        out := fun s =>
          findColl (ownLog H (run H P V x s.1.2.1) (E.ext x s.1.2.2.1 s.2.1) ++
            ownLog H (run H P V x s.1.2.1) (E.ext x (forkCoins s.1.1 s.1.2.2.1 s.1.2.2.2) s.2.2))
        cost := fun s =>
          (E.eval H P x s.1.2.1 s.1.2.2.1 s.2.1).2.1 +
            (E.eval H P x s.1.2.1 (forkCoins s.1.1 s.1.2.2.1 s.1.2.2.2) s.2.2).2.1 }
    else { S := Unit, out := fun _ => none, cost := fun _ => 0 }

/-- `λ` with `β > 0`: `min(log₂(1/κ), 255.5 − ½ log₂ γ, 256 − log₂(2βc)) − log₂ 3`, where `c` is the extractor's expected
work in runs. -/
noncomputable def lamLin (κ γ β c : ℝ) : ℝ :=
  min (min (Real.logb 2 (1 / κ)) (255.5 - Real.logb 2 γ / 2)) (256 - Real.logb 2 (2 * β * c)) - Real.logb 2 3

/-- `λ` bits with an extraction-based finder `Fe`: as `Bits`, where `Fe` also obeys the expected-time bound. -/
def BitsLin (H : Msg → Msg) (V : Verifier Coin) (R : Rel W) (E : Extractor Coin W V) (Fe : FinderFamily)
    (lam : Msg → ℝ) : Prop :=
  ∀ (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (t : ℕ), 1 ≤ t →
    (∀ ω r, (run H P V x ω r).pLog.length + (run H P V x ω r).vLog.length ≤ t) →
    (∀ hk : 0 < V.k x, StrictBirthday H (B H V P x hk)) →
    ExpectedCR H (Fe Ω P x) →
    PrBad H V R E x P ≤ t * (2 : ℝ) ^ (-lam x)

/-- **Cost, the extractor's own hashing**: `E` makes at most `calls(x)` SHA-512 calls itself, not counting its reruns',
on every prover, tape and coin. -/
def ExtractOwnLe (H : Msg → Msg) {V : Verifier Coin} (E : Extractor Coin W V) (calls : Msg → ℕ) : Prop :=
  ∀ (x : Msg) (Ω : Type) (P : Prover Ω) (ω : Ω) (r₀ : Fin (V.k x) → Coin) (c : E.C x),
    (ownLog H (run H P V x ω) (E.ext x r₀ c)).length ≤ calls x

end Finder

/-! ### Zero knowledge given the leaf layer -/

/-- A simulator that also receives the leaf layer: the hiding leaves of every opening. -/
structure SimAux where
  C : Msg → Type
  [fin : ∀ x, Fintype (C x)]
  [ne : ∀ x, Nonempty (C x)]
  prog : (x : Msg) → List Msg → C x → SP (List (Msg × Out))

attribute [instance] SimAux.fin SimAux.ne

/-- The simulator's transcript against `V*` at `θ`, given the leaf layer `aux`. -/
def SimAux.out {Θ : Type} (H : Msg → Msg) (S : SimAux) (Vs : VStar Θ) (θ : Θ) (x : Msg) (aux : List Msg)
    (c : S.C x) : List (Msg × Out) :=
  ((S.prog x aux c).eval H fun as => ((Vs.next x as θ).eval H).1).1

/-- `ZKmal`, where the simulator also receives `aux(x, w)`: for `(x, w) ∈ R`, every verifier `V*` and tape `θ`, the
transcript of `P(w)` against `V*` is within `ε_zk(x)` of `S`'s, on every event. -/
def ZKaux {Ω W : Type} [Fintype Ω] (H : Msg → Msg) (P : W → Prover Ω) (R : Rel W) (aux : Msg → W → List Msg)
    (S : SimAux) (εzk : Msg → ℝ) : Prop :=
  ∀ x w, R x w → ∀ (Θ : Type) (Vs : VStar Θ) (θ : Θ) (Ev : List (Msg × Out) → Prop),
    |Pr (fun ω : Ω => Ev (runStar H (P w) Vs x ω θ).msgs) - Pr (fun c : S.C x => Ev (S.out H Vs θ x (aux x w) c))| ≤
      εzk x

/-! ## 6. Claims -/

section Claims

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

/-- The sessions' knowledge error, Flock's at the inner numbers: `κ_F = 4·ε_table + 2^{−m′} + 2k′/K′ + e^{−k′/8}`. -/
noncomputable def κF (D : Deploy Coin Col T) (x : Msg) : ℝ :=
  4 * D.εtable x + 2 ^ (-(D.innerWaits x : ℝ)) + 2 * D.innerK x / D.innerCap x + Real.exp (-(D.innerK x : ℝ) / 8)

/-- The waits' slack factor, `c = 2^{2^{−40}} < 1 + 2^{−40}`: each wait's (`Deploy.k`). -/
noncomputable def slack : ℝ := 2 ^ ((2 : ℝ) ^ (-40 : ℝ))

/-- **The knowledge error at `δ`**: `δ + ε_ks`, with
`ε_ks = (c² − 1)·δ + c²·κ_F + (1 + c)·(2k/K + e^{−k/8})`: the outer wait's factor `c` times the inner wait's bound
`c·(P_a(B) + κ_F) + 2k/K + e^{−k/8}` at a minimal set `B` with `P_a(B) ≤ δ`, plus the outer truncation. -/
noncomputable def κ (D : Deploy Coin Col T) (δ : ℝ) (x : Msg) : ℝ :=
  slack ^ 2 * (δ + κF D x) + (1 + slack) * (2 * D.k x / D.cap x + Real.exp (-(D.k x : ℝ) / 8))

/-- **`γ`**: `c⁴·16·(√(2·k(x)·C) + k(x))²`, with `k(x)` the sampled verifier's rounds. -/
noncomputable def γ (D : Deploy Coin Col T) (x : Msg) : ℝ :=
  slack ^ 4 * (16 * (Real.sqrt (2 * D.V.k x * D.weight x) + D.V.k x) ^ 2)

/-- The design's miss at `t` at `x`, `0` on a statement that doesn't parse. -/
noncomputable def missAt (D : Deploy Coin Col T) (t : ℕ) (x : Msg) : ℝ :=
  match D.parse x with
  | some s => s.design.miss t
  | none => 0

/-- The extractor's expected work in runs, `1 + k + k²·(1 + n_PU·m′·k′) + k²·n_PU·openCalls`: its expected reruns, and
its own SHA-512 calls. -/
noncomputable def work (D : Deploy Coin Col T) (x : Msg) : ℝ :=
  1 + D.k x + (D.k x : ℝ) ^ 2 * (1 + nPU D x * D.innerWaits x * D.innerK x) + (D.k x : ℝ) ^ 2 * nPU D x * D.openCalls x

/-- The leaf layer under `w`: the hiding leaves of every opening. -/
def leaves (H : Msg → Msg) (D : Deploy Coin Col T) (w : Wit) : List Msg :=
  w.rus.flatMap (D.scheme.part.leaves H) ++ w.vals.flatMap (D.scheme.trace.leaves H)

/-- The sampling scope, on the statement's bytes. -/
def Scope (D : Deploy Coin Col T) (x : Msg) : Prop := ∃ s, D.parse x = some s ∧ InScope s

/-- **Sampling: KS**, at every `δ ≥ 0`, at the sampling extractor and every decoder that meets Flock's list-decoding
spec: `Pr[acc ∧ faulty ∉ M_δ] ≤ δ + ε_ks + ε_bind`, with `ε_ks = κ − δ + √(γ·Pr[B])` (Flock's error at the units'
sessions) and `ε_bind = Pr[B_ext]` (the openings an accepted extraction checks disagree). -/
def KS (H : Msg → Msg) (D : Deploy Coin Col T) : Prop :=
  ∀ (δ : ℝ) decode, 0 ≤ δ → Flock.ListComplete D.flock.enc D.flock.agree D.flock.nMin decode →
    Claims.KSLin H D.V (R H D.scheme D.parse δ) (E H D decode) (κ D δ) (γ D) (fun _ => 1)
      (Bext H D.V (E H D decode))

/-- **Sampling: KS, the count form** (a corollary of `KS` at `δ = miss(t)`): fewer than `t` units are faulty, except
with probability `miss(t) + ε_ks + ε_bind`. -/
def KSCount (H : Msg → Msg) (D : Deploy Coin Col T) : Prop :=
  ∀ t decode, Flock.ListComplete D.flock.enc D.flock.agree D.flock.nMin decode →
    Claims.KSLin H D.V (RCount H D.scheme D.parse t) (E H D decode) (fun x => κ D (missAt D t x) x) (γ D)
      (fun _ => 1) (Bext H D.V (E H D decode))

/-- **Sampling: binding** of both commitments, as a collision reduction, at every `H`. -/
def Binding (D : Deploy Coin Col T) : Prop := ∀ H, D.scheme.part.Binds H ∧ D.scheme.trace.Binds H

/-- **Sampling: the universal unit**: the public Program of class `c` (`UniversalUnit_v1{G, N_IN, N_OUT}`) maps every
netlist of the class, as its arguments, and inputs `x` to the netlist's outputs on `x`. -/
def Universal (D : Deploy Coin Col T) : Prop :=
  ∀ (c : SizeClass) (n : Netlist) (x : List Bool), n.Fits c → x.length = c.nIn →
    (D.universal c).eval (n.args c ++ x) = n.eval x

/-- **Sampling: Complete**, on the statements in scope, when no unit is faulty. It needs the verifier to check before
the draw that every unit's session has a form Flock proves (`Flock.InScope`); that check waits for the reference prover
(ruling 4). -/
def Complete {Ω : Type} (H : Msg → Msg) (D : Deploy Coin Col T) (P : Wit → Prover Ω) (δc : Msg → ℝ) : Prop :=
  Claims.Complete H D.V P (Rhon H D.scheme D.parse) (Scope D) δc

/-- **Sampling: the readable scope is the verifier's.** -/
def ScopeMatches (D : Deploy Coin Col T) : Prop := Claims.ScopeMatches (Scope D) D.scopeCheck

/-- **Sampling: zero knowledge** of the honest prover with masked sessions, when no unit is faulty, against every
malicious verifier, the simulator given the leaf layer: the transcript shows nothing but hiding leaves. -/
def ZK {Ω : Type} [Fintype Ω] (H : Msg → Msg) (D : Deploy Coin Col T) (P : Wit → Prover Ω) (S : SimAux)
    (εzk : Msg → ℝ) : Prop :=
  D.flock.zk = true ∧ ZKaux H P (Rhon H D.scheme D.parse) (fun _ w => leaves H D w) S εzk

/-- **Sampling: bits** at every `δ ≥ 0`, `λ_δ = min(log₂(1/κ_δ), 255.5 − ½ log₂ γ, 256 − log₂(2c)) − log₂ 3`, at every
decoder that meets the spec. -/
def Bits (H : Msg → Msg) (D : Deploy Coin Col T) : Prop :=
  ∀ (δ : ℝ) decode, 0 ≤ δ → Flock.ListComplete D.flock.enc D.flock.agree D.flock.nMin decode →
    BitsLin H D.V (R H D.scheme D.parse δ) (E H D decode) (Bext H D.V (E H D decode))
      fun x => lamLin (κ D δ x) (γ D x) 1 (work D x)

/-- **Sampling: efficiency**: the extractor's reruns, at most `1 + K + k·(K + k·n_PU·m′·K′)` and
`1 + k + k²·(1 + n_PU·m′·k′)` expected; its own SHA-512 calls, at most `k²·n_PU·openCalls`; and a decoder that meets
Flock's list-decoding spec, with its stated complexity. -/
def Efficient (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T)
    (counted : List (ℕ × Col) → List T × ℕ) (size : List (ℕ × Col) → ℕ) (p : Polynomial ℕ) : Prop :=
  ExtractRunsLe H (E H D decode)
      (fun x => 1 + D.cap x + D.k x * (D.cap x + D.k x * (nPU D x * (D.innerWaits x * D.innerCap x)))) ∧
    ExtractRunsExpLe H (E H D decode)
      (fun x => 1 + D.k x + (D.k x : ℝ) ^ 2 * (1 + nPU D x * D.innerWaits x * D.innerK x)) ∧
    ExtractOwnLe H (E H D decode) (fun x => D.k x ^ 2 * nPU D x * D.openCalls x) ∧
    Flock.ListComplete D.flock.enc D.flock.agree D.flock.nMin decode ∧ DecoderComplexity decode counted size p

/-- **Sampling: nontriviality**: recorded honest runs, each of a statement the scope check takes, that the verifier
accepts. -/
def Nontrivial (H : Msg → Msg) (D : Deploy Coin Col T) (recs : List (Recorded D.V)) : Prop :=
  Claims.Nontrivial H D.V recs ∧ ∀ e ∈ recs, D.scopeCheck e.x = true

end Claims

end Claims.Sampling
