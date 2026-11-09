module

public import Definitions.Claims

/-!
# Claims: the sampling compiler (frozen before any proof work)

The claims block of sampled proofs (`verity/core/protocols/spot/`), one-stage sampling, on the
shared definitions of `Definitions.Core.Interaction`, in the template's eight items and the README glossary's terms. Parametric in
the deployed sampled verifier (`Deploy`), as Flock's block is in `Flock.Deploy`: binding it to the compiled verifier is
the code side's.

1. **Statement** (`AuditStatement`): the program `C` (named by its digest; both parties hold it), as its descriptor and its gates,
   the partition query `Q` (named, versioned, with its parameters), the sampling design, and the roots the trace is
   committed under. The partition is public: `P = Q(C)`, which the verifier evaluates itself on the statement it parsed,
   descriptor and gates (`Deploy.partition`), gives the proof blocks (sets of gates of `C`, every gate but a constant and
   an unread input in exactly one, the other input gates included; a constant is public and in one or in none, and so is
   an input that no gate reads and `C` doesn't return), the strata and the classes. `x` is the canonical bytes of what
   the verifier reads before it samples, and `parse` is its own.
2. **Witness** (`Wit`): openings of the committed values of `C`'s gates under the roots.
3. **Relation** (`R δ`): the set of incorrect proof blocks of the committed trace is in the confidence set
   `M_δ = {B : P_a(B) > δ}`. A unit is faulty when its committed outputs differ from its gates applied to its committed
   inputs, or a value it reads or writes is unopened: one is faulty unless some interior makes it correct, which on a
   topologically ordered program is the glossary's form on the boundary trace (`BoundaryFaulty`;
   `Proofs.Flock.SamplingBoundary` proves `Faulty` is it on every such program).
4. **Scope** (`Scope`, `InScope`, `ScopeMatches`): a statement that parses, whose query evaluates; a topologically
   ordered program; a partition of all its gates but constants and unread inputs into proof blocks, each a set of gates
   whose committed values fit its public size class, `|out(u)| ≤ w_{c(u)}` (`sizeClasses`: powers of two from 16 to
   262,144); and a well-formed design, one rule per class. The compiled sampled verifier's `KS`, `KSCount` and `Bits`
   at a deployment (`SamplingGeneric.ofDeploy`) and `Complete` hold on it, and it excludes every design whose `P_a` is
   not its sample's miss probability: `subset K` with `K > N`, whose `P_a` is `0` at every set, so that `KS` there is
   vacuous, and a Bernoulli rule with `den = 0` or `num > den`. That the verifier refuses what is out of it is
   `ScopeMatches`, open.
5. **Named pieces**: the commitment scheme (`CommitmentScheme`, `CommitmentScheme.Binds`, and its published leaf layer
   `CommitmentScheme.layer`); the program and its units (`BGate`, `Prog`, `Partition`, `Prog.Correct`); the universal
   unit (`PrivateCircuit`, `Universal`), a fact about the programs private circuits run as; the acceptance probabilities
   (`OneStage.Pa`, `Design.Pa`); the extractor's waits (`Deploy.k`, `Deploy.cap`, `slack`) and opening check
   (`checkRevealed`, `assemble`); the expected-time finder (`Bext`); and the parameters (`Deploy`, with the Flock
   verifier its sessions run as a `Flock.Deploy`, and the masked deployment `ZkDeploy`: the reference prover's tape as
   a finite nonce and rest, and the simulator). The extractor itself is the compiled sampled verifier's
   (`SamplingGeneric.Compiled.E` at `SamplingGeneric.ofDeploy`): `k` accepted runs, in each `k` accepted reruns of the
   second stage, in each Flock's `E^rep` on every checked unit's session.
6. **Claims**, on the statements in scope: `Binding`; `Universal`; `Complete`; `ScopeMatches`; `ZK`, of the masked
   deployment's reference prover and simulator, relative to the trace's published leaf layer; and `Nontrivial`.
   Knowledge soundness, its count form, bits and efficiency are the compiled sampled verifier's at the deployment
   (`SamplingGeneric.Compiled.KS`, `KSCount`, `Bits`, `Efficient`): `KS` at every `δ ≥ 0`, with error
   `δ + ε_ks + ε_bind`, `ε_ks` Flock's at the units' sessions and `ε_bind = Pr[B_ext]`.
7. **Assumptions used**: `trustedBase`, and the compiled claims' `StrictBirthday` (for `B`) and `ExpectedCR` (for
   `Bext`). `ZK` gives the simulator the leaf layer the protocol publishes, so it reads no hiding assumption of the
   leaves.
8. **Status**: stated; no claim is proved.
   `Proofs.Flock.SamplingScope` shows the scope is the one `KS` needs: an out-of-scope `subset K` with `K > N` has
   `P_a = 0` at every set, and an in-scope design has a deployment at which the compiled verifier's `KS` bound is
   below `1`.

Generic pieces this block adds, which move to `Definitions.Core.Interaction` in a later change: `LogColl`, `ofOC`, `ownLog`,
`flatMapRP`, `waitAccC`, `forkOn`, `Bext`, `KSLinOn`, `lamLin`, `BitsLin`, `ExtractOwnLe` and `SimLeak`. Zero knowledge
is the spec's `ZKmalRx` and the decoder spec Flock's radius form, `Flock.ListWithin` at `Flock.Deploy`'s `len` and `θ`;
nothing here reads Flock's former decoding threshold or count (`agree`, `nMin`).

Frozen by `Flock.Guarantees.SamplingClaimsStated` (`Definitions/Flock/Guarantees/SamplingClaims.lean`), whose statement
reads every claim here: the lock records each one's text under `reads`, so a changed claim is a changed record.
-/

@[expose] public section

namespace Claims.Sampling

/-! ## The commitment scheme -/

/-- Two different inputs with one output in a log of `H` calls. -/
def LogColl (l : List (Msg × Msg)) : Prop := ∃ p ∈ l, ∃ p' ∈ l, p.1 ≠ p'.1 ∧ p.2 = p'.2

/-- An opening: the committed value `val` at position `pos`, and the opening's data. -/
structure Opening (α : Type) where
  pos : ℕ
  val : α
  data : Msg

/-- **The commitment scheme**, a named primitive: a hiding commitment to values at positions under public roots, with
its opening check `opens` and its published leaf layer: `layer H roots os`, the leaves under `roots` of the values the
openings `os` open. The deployed instance is hm96 leaves in frame-v3-sha512 trees, its layer every hm96 leaf under the
roots in tree order; its storage layout is behind `opens`. -/
structure CommitmentScheme (α : Type) where
  opens : List Msg → Opening α → OC Bool
  layer : (Msg → Msg) → List Msg → List (Opening α) → List Msg

/-- Whether `o` opens under `roots`. -/
def CommitmentScheme.Opens {α : Type} (H : Msg → Msg) (Cm : CommitmentScheme α) (roots : List Msg) (o : Opening α) : Bool :=
  ((Cm.opens roots o).eval H).1

/-- **Binding, as a collision reduction**: two openings that pass at one position under the same roots, of different
values, make `H` calls that collide. -/
def CommitmentScheme.Binds {α : Type} (H : Msg → Msg) (Cm : CommitmentScheme α) : Prop :=
  ∀ (roots : List Msg) (o o' : Opening α), o.pos = o'.pos → o.val ≠ o'.val → Cm.Opens H roots o = true →
    Cm.Opens H roots o' = true → LogColl (((Cm.opens roots o).eval H).2 ++ ((Cm.opens roots o').eval H).2)

/-! ## The program and its partition -/

/-- **A gate** of the Boolean profile (`verity/ml/circuits/PROTOCOL.md` §9): an input (`Input1_v1` or
`RegisteredInput1_v1`, its value committed under the roots), a constant bit (`Const1`), `Not_v1` of a wire, or
`And<n>_v1` or `Xor<n>_v1` of wires. -/
inductive BGate where
  | input
  | const (b : Bool)
  | not (i : ℕ)
  | and (ws : List ℕ)
  | xor (ws : List ℕ)
  deriving DecidableEq

/-- The wires a gate reads. -/
def BGate.reads : BGate → List ℕ
  | .input => []
  | .const _ => []
  | .not i => [i]
  | .and ws => ws
  | .xor ws => ws

/-- A gate's value from the values `rd` of the wires it reads, and an input's from its committed value `own`; `none`
when a value it needs is. -/
def BGate.val (rd : ℕ → Option Bool) (own : Option Bool) : BGate → Option Bool
  | .input => own
  | .const b => some b
  | .not i => (rd i).map (!·)
  | .and ws => (ws.mapM rd).map fun bs => bs.all id
  | .xor ws => (ws.mapM rd).map fun bs => bs.foldl Bool.xor false

/-- **A program** `C`, flattened to its gates in layout order (wire `g` is gate `g`'s value), and the gates it
returns. -/
structure Prog where
  gates : List BGate
  outs : List ℕ

/-- Topologically ordered: every gate reads only earlier gates, and every returned gate is one of its gates. -/
def Prog.Topological (C : Prog) : Prop :=
  (∀ g (x : BGate), C.gates[g]? = some x → ∀ i ∈ x.reads, i < g) ∧ ∀ o ∈ C.outs, o < C.gates.length

/-- `C` run on inputs `x`, gate by gate: its input gates read `x` in order; a wire at or past the reading gate's own
reads `0`. -/
def Prog.wires (C : Prog) (x : List Bool) : List Bool :=
  (C.gates.foldl (fun (st : List Bool × ℕ) g =>
    match g with
    | .input => (st.1 ++ [x.getD st.2 false], st.2 + 1)
    | g => (st.1 ++ [(g.val (fun i => some (st.1.getD i false)) none).getD false], st.2)) ([], 0)).1

/-- `C`'s returned values on inputs `x`. -/
def Prog.eval (C : Prog) (x : List Bool) : List Bool := C.outs.map fun o => (C.wires x).getD o false

/-- Gate `i`'s value as a gate outside the set of gates that holds it reads it: a constant's own value (a constant is
public and committed by nobody), every other gate's committed value `ref i`. -/
def Prog.read (C : Prog) (ref : ℕ → Option Bool) (i : ℕ) : Option Bool :=
  match C.gates[i]? with
  | some (.const b) => some b
  | _ => ref i

/-- Whether gate `g`'s value is committed, as the set of gates `u` sees it: `g` is returned, is an input, or is read by a
gate outside `u`. -/
def Prog.Boundary (C : Prog) (u : List ℕ) (g : ℕ) : Bool :=
  C.outs.contains g || decide (C.gates[g]? = some .input) ||
    (List.range C.gates.length).any fun g' =>
      decide (g' ∉ u) && ((C.gates[g']?.map BGate.reads).getD []).contains g

/-- Gate `g`'s value computed in the set of gates `u`, on the committed values `ref`: from the values of the wires it
reads, one of `u` before `g` as computed, one of `u` at or past `g` as `none`, one outside `u` as `read` sees it; an
input's, its committed value. -/
def Prog.runIn (C : Prog) (ref : ℕ → Option Bool) (u : List ℕ) (g : ℕ) : Option Bool :=
  C.gates[g]?.bind fun x =>
    x.val (fun i => if i ∈ u then (if _h : i < g then C.runIn ref u i else none) else C.read ref i) (ref g)
termination_by g

/-- **The set of gates `u` of `C` is correct** on the committed values `ref`: every value it reads from outside it is
opened (a constant's needs no opening), and every gate of it whose value is committed (`Boundary`) has a committed value,
the one `u` computes. -/
def Prog.Correct (C : Prog) (ref : ℕ → Option Bool) (u : List ℕ) : Prop :=
  ∀ g ∈ u,
    (∀ x : BGate, C.gates[g]? = some x → ∀ i ∈ x.reads, i ∉ u → C.read ref i ≠ none) ∧
      (C.Boundary u g = true → ∃ b, ref g = some b ∧ C.runIn ref u g = some b)

/-- **The values proof block `u` reads and writes**: a gate of `u` whose value `u`'s boundary doesn't commit has its
interior value `ι`; every other, its committed value `val`. -/
def Prog.refIn (C : Prog) (u : List ℕ) (val : ℕ → Option Bool) (ι : ℕ → Bool) (g : ℕ) : Option Bool :=
  if g ∈ u ∧ C.Boundary u g = false then some (ι g) else val g

/-- The values proof block `u` of `C` commits: its gates that `C.Boundary` commits. -/
def committed (C : Prog) (u : List ℕ) : ℕ := (u.filter fun g => C.Boundary u g).length

/-- **An unread input** of `C`: an input gate that `C` doesn't return and no gate of `C` reads. No gate's value and no
returned value depends on it. -/
def Prog.Unread (C : Prog) (g : ℕ) : Prop :=
  C.gates[g]? = some .input ∧ g ∉ C.outs ∧ ∀ x ∈ C.gates, g ∉ x.reads

instance instDecidableUnread (C : Prog) (g : ℕ) : Decidable (C.Unread g) :=
  inferInstanceAs (Decidable (C.gates[g]? = some .input ∧ g ∉ C.outs ∧ ∀ x ∈ C.gates, g ∉ x.reads))

/-- **A partition** `P = Q(C)`, as the verifier evaluates it: its classes, each its strata, each its proof blocks (each
a set of gates in layout order), in population order. -/
structure Partition where
  classes : List (List (List (List ℕ)))

/-- **The proof blocks**: the proof blocks, slot by slot. -/
def Partition.blocks (P : Partition) : List (List ℕ) := P.classes.flatten.flatten

/-- The gates of all of `P`'s proof blocks. -/
def Partition.gates (P : Partition) : List ℕ := P.blocks.flatten

/-- **The widest public size class** of a proof block's outputs, in bits: 262,144, provisional. Setting it to `16` makes
`sizeClasses` `[16]`, the former literal limit. -/
def unitClassMax : ℕ := 262144

/-- **The public size classes** `w_c` of a proof block's outputs, in bits: the powers of two from `16` up to
`unitClassMax`, ascending (`[16, 32, …, 262144]`). -/
def sizeClasses : List ℕ := (List.range (Nat.log2 (unitClassMax / 16) + 1)).map (16 * 2 ^ ·)

/-- **A proof block's size class** `w_{c(u)}`: the least public class that holds the values it commits (`committed`), a
function of the program and the unit, both public; `none` when no class does. -/
def unitClass (C : Prog) (u : List ℕ) : Option ℕ := sizeClasses.find? fun w => decide (committed C u ≤ w)

/-- **`P` partitions `C`**: no gate is in two proof blocks, and every gate of `C` but a constant and an unread input is
in one, the other input gates included. A constant, public, may be in one or in none, and so may an unread input
(`Prog.Unread`), on which no gate's value and no returned value depends. Every proof block has a gate of `C`, and each
proof block `u` has a public size class `c(u)` whose width holds the values it commits, its gates that `Boundary`
commits: `|out(u)| ≤ w_{c(u)}`. A proof block is a set of gates, so a unit's gates may interleave with another's. -/
def Partition.Partitions (P : Partition) (C : Prog) : Prop :=
  P.gates.Nodup ∧ (∀ g ∈ P.gates, g < C.gates.length) ∧
    (∀ g < C.gates.length, (∀ b, C.gates[g]? ≠ some (.const b)) → ¬ C.Unread g → g ∈ P.gates) ∧
    (∀ u ∈ P.blocks, u ≠ []) ∧ ∀ u ∈ P.blocks, ∃ w, unitClass C u = some w ∧ committed C u ≤ w

/-! ## Proof blocks and the universal unit -/

/-- A size class `(G, N_IN, N_OUT)`: `G` gates over `N_IN` inputs, and `N_OUT` output bits. -/
structure SizeClass where
  G : ℕ
  nIn : ℕ
  nOut : ℕ

/-- `K`: the bits of a wire index, enough for the `W = N_IN + G` wires, at least one. -/
def SizeClass.indexBits (c : SizeClass) : ℕ := Nat.log2 (c.nIn + c.G - 1) + 1

/-- **A private circuit's gates**, as the universal unit takes them (`verity.ml.private_circuits.universal`): SSA wires,
the `nIn` inputs and then one per gate. Gate `(op, a, b)` reads wires `a` and `b` and computes XOR, AND, XNOR or NAND
(`op`'s bit 0 picks AND, bit 1 negates); a wire at or past the reading gate's own reads `0`, as does an output past the
last wire. -/
structure PrivateCircuit where
  nIn : ℕ
  gates : List (Fin 4 × ℕ × ℕ)
  outs : List ℕ

/-- The private circuit run gate by gate on `x` (`interpret`). -/
def PrivateCircuit.eval (n : PrivateCircuit) (x : List Bool) : List Bool :=
  let ws := n.gates.foldl (fun ws g =>
    let a := ws.getD g.2.1 false
    let b := ws.getD g.2.2 false
    ws ++ [xor (if g.1.val % 2 = 1 then a && b else xor a b) (decide (g.1.val / 2 = 1))]) x
  n.outs.map fun o => ws.getD o false

/-- The private circuit is of class `c`: `G` gates over `N_IN` inputs, `N_OUT` outputs, every index in `K` bits. -/
def PrivateCircuit.Fits (c : SizeClass) (n : PrivateCircuit) : Prop :=
  n.nIn = c.nIn ∧ n.gates.length = c.G ∧ n.outs.length = c.nOut ∧
    (∀ g ∈ n.gates, g.2.1 < 2 ^ c.indexBits ∧ g.2.2 < 2 ^ c.indexBits) ∧ ∀ o ∈ n.outs, o < 2 ^ c.indexBits

/-- `v` in `k` bits, low bit first. -/
def bitsLE (k v : ℕ) : List Bool := (List.range k).map v.testBit

/-- The private circuit as the universal unit's `(op, src, out)` arguments, in leaf order (`arguments`). -/
def PrivateCircuit.args (c : SizeClass) (n : PrivateCircuit) : List Bool :=
  n.gates.flatMap (fun g => bitsLE 2 g.1.val) ++
    n.gates.flatMap (fun g => bitsLE c.indexBits g.2.1 ++ bitsLE c.indexBits g.2.2) ++
    n.outs.flatMap (bitsLE c.indexBits)

/-! ## Sampling designs -/

/-- **A one-stage sampling design** over a population in strata: every member (`all`); a uniform `K`-subset
(`subset:K`); each member independently at rate `num/den` (`bernoulli:NUM/DEN`); a uniform `k_s`-subset of each
stratum, independently, with `k_s = max(1, round(K·n_s/N))` (`stratified:K`), or
`k_s = min(n_s, max(f_s, ⌈K·w_s·n_s/W⌉))` from each stratum's work `w_s` and floor `f_s`, `W = Σ w_s·n_s` (`work:K`). -/
inductive OneStage where
  | all
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
  | .all, B => if B = ∅ then 1 else 0
  | .subset K, B => ((ns.sum - B.card).choose K : ℝ) / (ns.sum.choose K : ℝ)
  | .bernoulli num den, B => (((den - num : ℕ) : ℝ) / den) ^ B.card
  | .stratified K, B => strataPa (ns.map fun n => (n, max 1 ((2 * K * n + ns.sum) / (2 * ns.sum)))) 0 B
  | .work K w f, B =>
    let W := ((ns.zip w).map fun p => p.1 * p.2).sum
    strataPa ((ns.zip (w.zip f)).map fun p => (p.1, min p.1 (max p.2.2 ((K * p.2.1 * p.1 + W - 1) / W)))) 0 B

/-- A class's strata sizes, in proof blocks. -/
def strataSizes (c : List (List (List ℕ))) : List ℕ := c.map List.length

/-- **The sampling design**: one rule per class of the partition, each a one-stage design over the class's proof blocks,
the classes sampled independently; the input stratum is a class sampled `all`. -/
structure Design where
  rules : List OneStage

/-- The classes' acceptance probabilities, from slot `o`, multiplied. -/
noncomputable def classesPa : List OneStage → List (List (List (List ℕ))) → ℕ → Finset ℕ → ℝ
  | r :: rs, c :: cs, o, B => r.Pa (strataSizes c) (window o c.flatten.length B) * classesPa rs cs (o + c.flatten.length) B
  | _, _, _, _ => 1

/-- **The acceptance probability** `P_a(B)` of the design on the partition `P`, over a set `B` of slots. -/
noncomputable def Design.Pa (d : Design) (P : Partition) (B : Finset ℕ) : ℝ := classesPa d.rules P.classes 0 B

/-- **The confidence set** `M_δ`: every set `B` with `P_a(B) > δ`. -/
def Design.M (d : Design) (P : Partition) (δ : ℝ) : Set (Finset ℕ) := {B | δ < d.Pa P B}

/-- **The miss at `t`**: the largest acceptance probability of `t` or more of the units. -/
noncomputable def Design.miss (d : Design) (P : Partition) (t : ℕ) : ℝ :=
  ⨆ B : {B : Finset ℕ // B ⊆ Finset.range P.blocks.length ∧ t ≤ B.card}, d.Pa P B.1

/-! ## 1. Statement -/

/-- A partition query, named and versioned (`Q_call` v1 with `X`; `Q_word` v1 with `X` and `W`), with its parameters as
canonical bytes. -/
structure Query where
  name : String
  version : ℕ
  params : Msg

/-- **The sampling statement**: the program (named by its digest; both parties hold it), as its descriptor's canonical
bytes (`desc`, `verity-ir/descriptor/v1`, which a template query reads) and its gates (`prog`, the descriptor
flattened), the partition query, the sampling design, and the roots the trace is committed under. -/
structure AuditStatement where
  desc : Msg
  prog : Prog
  query : Query
  design : Design
  roots : List Msg

/-! ## 2. Witness -/

/-- **The sampling witness**: openings of committed values, at their gates' positions. -/
structure Wit where
  vals : List (Opening Bool)

/-! ## 3. Relation -/

/-- The committed value `w` opens at gate `g`'s position: its first opening there that passes. -/
def Wit.val (H : Msg → Msg) (Cm : CommitmentScheme Bool) (s : AuditStatement) (w : Wit) (g : ℕ) : Option Bool :=
  (w.vals.find? fun o => o.pos == g && Cm.Opens H s.roots o).map (·.val)

/-- **An incorrect proof block** at slot `v` of the partition `P` under `w`: unless some interior values make the proof block there
correct on its committed values: its committed outputs differ from its gates applied to its committed inputs, or a
value it reads or writes is unopened. -/
def Faulty (H : Msg → Msg) (Cm : CommitmentScheme Bool) (s : AuditStatement) (P : Partition) (w : Wit) (v : ℕ) : Prop :=
  ¬ ∃ u, P.blocks[v]? = some u ∧ ∃ ι : ℕ → Bool, s.prog.Correct (s.prog.refIn u (w.val H Cm s) ι) u

/-- **An incorrect proof block, on the boundary trace** (the glossary's form): the proof block's committed boundary outputs differ
from its gates applied to its committed boundary inputs. -/
def BoundaryFaulty (H : Msg → Msg) (Cm : CommitmentScheme Bool) (s : AuditStatement) (P : Partition) (w : Wit) (v : ℕ) : Prop :=
  ¬ ∃ u, P.blocks[v]? = some u ∧ s.prog.Correct (w.val H Cm s) u

open Classical in
/-- **The incorrect set** of the trace `w`'s openings define. -/
noncomputable def faulty (H : Msg → Msg) (Cm : CommitmentScheme Bool) (s : AuditStatement) (P : Partition) (w : Wit) : Finset ℕ :=
  (Finset.range P.blocks.length).filter (Faulty H Cm s P w)

/-- **The sampling relation** at `δ`, on a parsed statement and its partition: the incorrect set is in `M_δ`. -/
def RelS (H : Msg → Msg) (Cm : CommitmentScheme Bool) (δ : ℝ) (s : AuditStatement) (P : Partition) (w : Wit) : Prop :=
  faulty H Cm s P w ∈ s.design.M P δ

/-- **The sampling relation** at `δ`, on the statement's bytes: it parses (`parse`, the verifier's own), its query
evaluates on its program, descriptor and gates (`partition`, the verifier's own), and `RelS` holds. -/
def R (H : Msg → Msg) (Cm : CommitmentScheme Bool) (parse : Msg → Option AuditStatement) (partition : AuditStatement → Option Partition)
    (δ : ℝ) : Rel Wit :=
  fun x w => ∃ s P, parse x = some s ∧ partition s = some P ∧ RelS H Cm δ s P w

/-- The count form: fewer than `t` units are faulty. -/
def RCount (H : Msg → Msg) (Cm : CommitmentScheme Bool) (parse : Msg → Option AuditStatement)
    (partition : AuditStatement → Option Partition) (t : ℕ) : Rel Wit :=
  fun x w => ∃ s P, parse x = some s ∧ partition s = some P ∧ (faulty H Cm s P w).card < t

/-- The honest relation: no unit is faulty. -/
def Rhon (H : Msg → Msg) (Cm : CommitmentScheme Bool) (parse : Msg → Option AuditStatement)
    (partition : AuditStatement → Option Partition) : Rel Wit :=
  fun x w => ∃ s P, parse x = some s ∧ partition s = some P ∧ faulty H Cm s P w = ∅

/-! ## 4. Scope -/

/-- A one-stage design is well formed over strata of sizes `ns`. -/
def OneStage.WellFormed (ns : List ℕ) : OneStage → Prop
  | .all => True
  | .subset K => 1 ≤ K ∧ K ≤ ns.sum
  | .bernoulli num den => 0 < den ∧ num ≤ den
  | .stratified K => 1 ≤ K ∧ K ≤ ns.sum ∧ ∀ n ∈ ns, 1 ≤ n
  | .work _ w f => w.length = ns.length ∧ f.length = ns.length ∧ (∀ x ∈ f, 1 ≤ x) ∧
      0 < ((ns.zip w).map fun p => p.1 * p.2).sum ∧ ∀ n ∈ ns, 1 ≤ n

/-- The design is well formed on the partition: one rule per class, well formed over the class's strata. -/
def Design.WellFormed (d : Design) (P : Partition) : Prop :=
  d.rules.length = P.classes.length ∧ ∀ p ∈ d.rules.zip P.classes, p.1.WellFormed (strataSizes p.2)

/-- **The sampling scope, readable**, at the statement's partition: a topologically ordered program, partitioned into
proof blocks (sets of gates, constants and unread inputs in one or in none) whose committed values fit their public size
classes, and a well-formed design.

The query: whatever its own width parameter, every unit, input units included, commits at most its class's width
`w_{c(u)}` (`sizeClasses`, up to `unitClassMax`). A unit that commits more than the widest class is out of scope. -/
def InScope (P : Partition) (s : AuditStatement) : Prop :=
  s.prog.Topological ∧ P.Partitions s.prog ∧ s.design.WellFormed P

/-! ## 5. Named pieces -/

/-- What the deployed sampled verifier brings, read by the claims: the verifier, its parser and scope check, its own
evaluation of a statement's partition query on its program, descriptor and gates (`partition`, `P = Q(C)`), the
trace's commitment, the universal unit of each size class (the program a private circuit's units run, its arguments the
private circuit's description), and the Flock verifier its sessions run (`flock`): whether it is the masked one, its
parser, the opened columns of a run, the code, each trace's codeword length and decoding radius, and how a trace is
read as a witness.

The sample is `V`'s, from its coins, in this layout: `stage2` holds the rounds after the sample (the sessions), so runs that
keep the other rounds make the same sample and see the same interior commitments; `checked` is the slots of the proof
blocks the run checks; `rounds` is each one's session's rounds; `session` is its Flock statement and what its session
showed; `reveal` is what its Flock witness reveals: committed values, each with the roots it is committed under (the
statement's, or the interior roots the run showed).

The numbers: the session's `E^rep` makes `m′` waits (`innerWaits`) of `K′` reruns (`innerCap`) with `N₀` (`innerN₀`),
`ε_table` and the weight `C` of `γ`, each good for every session at `x`; `openCalls` bounds the SHA-512 calls of checking
what one session's witness reveals.

And the reference prover (`P`), the honest sampled prover the deployment runs, on its tapes (`Tape`, a nonempty type):
completeness is its (`Complete`), at its completeness error as the deployment states it at each hash (`δc H x` at
`H`). -/
structure Deploy (Coin Col T : Type) where
  V : Verifier Coin
  parse : Msg → Option AuditStatement
  scopeCheck : Msg → Bool
  partition : AuditStatement → Option Partition
  commit : CommitmentScheme Bool
  universal : SizeClass → Prog
  flock : Flock.Deploy Coin Col T
  stage2 : (x : Msg) → Finset (Fin (V.k x))
  checked : (x : Msg) → (Fin (V.k x) → Coin) → List ℕ
  rounds : (x : Msg) → ℕ → Finset (Fin (V.k x))
  session : Msg → ℕ → Seen → Option (Msg × Seen)
  reveal : AuditStatement → ℕ → Seen → Flock.Wit → List (List Msg × Opening Bool)
  innerWaits : Msg → ℕ
  innerN₀ : Msg → ℝ
  innerCap : Msg → ℕ
  εtable : Msg → ℝ
  weight : Msg → ℝ
  openCalls : Msg → ℕ
  Tape : Type
  [tapeNonempty : Nonempty Tape]
  P : Wit → Prover Tape
  δc : (Msg → Msg) → Msg → ℝ

instance Deploy.instNonemptyTape {Coin Col T : Type} (D : Deploy Coin Col T) : Nonempty D.Tape := D.tapeNonempty

/-- The statement's partition, as the verifier evaluates its query; empty when the query doesn't apply. -/
def Deploy.part {Coin Col T : Type} (D : Deploy Coin Col T) (s : AuditStatement) : Partition :=
  (D.partition s).getD ⟨[]⟩

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

/-! ### The extractor's waits and opening check -/

section Extract

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

/-- The session's `E^rep` waits for `k′ = ⌈4·N₀/e⌉` accepted reruns. -/
noncomputable def Deploy.innerK (D : Deploy Coin Col T) (x : Msg) : ℕ := ⌈4 * D.innerN₀ x / Real.exp 1⌉₊

/-- The proof blocks at `x`, `0` on a statement that doesn't parse. -/
def nPU (D : Deploy Coin Col T) (x : Msg) : ℕ :=
  match D.parse x with
  | some s => (D.part s).blocks.length
  | none => 0

/-- **The waits' slack exponent** `j = 10`: a wait keeps `2^j` accepted runs for each bit of the union its bound is paid
over, so the union costs at most the slack `c = 2^{2^{−j}}` (`slack`). -/
def slackExp : ℕ := 10

/-- **The extractor's waits** stop at `k = 2^j·(2·n_PU + 1)` accepted runs (`slackExp`): both waits of the compiled
sampled verifier at this deployment (`SamplingGeneric.ofDeploy`). A union over the minimal sets of the complement of
`M_δ`, at most `2^n` for `n` units, spread over `k` accepted runs, costs the factor `2^{n/k}`; this `k` makes it at most
`2^{2^{−j}}`, while `k` enters the extractor's cost, and so the hash side of `Bits`, only through `log₂ k`. -/
def Deploy.k (D : Deploy Coin Col T) (x : Msg) : ℕ := 2 ^ slackExp * (2 * nPU D x + 1)

/-- The waits give up after `K = 2^128·k` reruns. -/
def Deploy.cap (D : Deploy Coin Col T) (x : Msg) : ℕ := 2 ^ 128 * D.k x

/-- What passes its opening check among `rs`, each checked by the commitment under its roots, its `H` calls the
extractor's own. -/
def checkRevealed {R : Type} (Cm : CommitmentScheme Bool) :
    List (List Msg × Opening Bool) → RP R (List (List Msg × Opening Bool)) :=
  flatMapRP fun p => (ofOC (Cm.opens p.1 p.2)).bind fun ok => .ret (if ok then [p] else [])

/-- **The assemble step**: the openings of committed values under the statement's roots. -/
def assemble (s : AuditStatement) (found : List (List Msg × Opening Bool)) : Wit where
  vals := found.filterMap fun p => if p.1 = s.roots then some p.2 else none

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

/-- **`B_ext` declared at `T` a rerun**: `Bext`'s coins and outcome, its cost both extractions' own SHA-512 calls plus
`T` for each rerun of the prover, `T` the budget for one run's whole SHA-512 work, the prover's own included, which the
reader computes from the prover's time. `ExpectedCR` at it so counts the hashing a prover does outside the oracle, which
`Bext`'s logged cost doesn't (`KSAt`). -/
noncomputable def BextAt (H : Msg → Msg) (V : Verifier Coin) (E : Extractor Coin W V) (T : ℕ) : FinderFamily :=
  fun Ω _ _ P x =>
    if hk : 0 < V.k x then
      { S := BCoins Ω Coin (V.k x) × E.C x × E.C x
        ne := ⟨(⟨⟨0, hk⟩, Classical.arbitrary _, fun _ => default, fun _ => default⟩, Classical.arbitrary _,
          Classical.arbitrary _)⟩
        out := fun s =>
          findColl (ownLog H (run H P V x s.1.2.1) (E.ext x s.1.2.2.1 s.2.1) ++
            ownLog H (run H P V x s.1.2.1) (E.ext x (forkCoins s.1.1 s.1.2.2.1 s.1.2.2.2) s.2.2))
        cost := fun s =>
          (ownLog H (run H P V x s.1.2.1) (E.ext x s.1.2.2.1 s.2.1)).length +
            (ownLog H (run H P V x s.1.2.1) (E.ext x (forkCoins s.1.1 s.1.2.2.1 s.1.2.2.2) s.2.2)).length +
            T * ((E.eval H P x s.1.2.1 s.1.2.2.1 s.2.1).2.2 +
              (E.eval H P x s.1.2.1 (forkCoins s.1.1 s.1.2.2.1 s.1.2.2.2) s.2.2).2.2) }
    else { S := Unit, out := fun _ => none, cost := fun _ => 0 }

/-- `λ` with `β > 0`: `min(log₂(1/κ), 255.5 − ½ log₂ γ, 256 − log₂(2βc)) − log₂ 3`, where `c` is the extractor's expected
work in runs. -/
noncomputable def lamLin (κ γ β c : ℝ) : ℝ :=
  min (min (Real.logb 2 (1 / κ)) (255.5 - Real.logb 2 γ / 2)) (256 - Real.logb 2 (2 * β * c)) - Real.logb 2 3

/-- **`KSLin` on a scope**: `Claims.KSLin`'s bound at every statement in `Scope`. -/
def KSLinOn (H : Msg → Msg) (V : Verifier Coin) (Scope : Msg → Prop) (R : Rel W) (E : Extractor Coin W V)
    (κ γ β : Msg → ℝ) (Fe : FinderFamily) : Prop :=
  ∀ x, Scope x → ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω),
    PrBad H V R E x P ≤ κ x + Real.sqrt (γ x * PrB H V P x) + β x * (Fe Ω P x).Pr H

/-- `λ` bits with an extraction-based finder `Fe`, at every statement in `Scope`: as `Bits`, where `Fe` also obeys the
expected-time bound. -/
def BitsLin (H : Msg → Msg) (V : Verifier Coin) (Scope : Msg → Prop) (R : Rel W) (E : Extractor Coin W V)
    (Fe : FinderFamily) (lam : Msg → ℝ) : Prop :=
  ∀ (x : Msg), Scope x → ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (t : ℕ), 1 ≤ t →
    (∀ ω r, (run H P V x ω r).pLog.length + (run H P V x ω r).vLog.length ≤ t) →
    (∀ hk : 0 < V.k x, StrictBirthday H (B H V P x hk)) →
    ExpectedCR H (Fe Ω P x) →
    PrBad H V R E x P ≤ t * (2 : ℝ) ^ (-lam x)

/-- **Cost, the extractor's own hashing**: `E` makes at most `calls(x)` SHA-512 calls itself, not counting its reruns',
on every prover, tape and coin. -/
def ExtractOwnLe (H : Msg → Msg) {V : Verifier Coin} (E : Extractor Coin W V) (calls : Msg → ℕ) : Prop :=
  ∀ (x : Msg) (Ω : Type) (P : Prover Ω) (ω : Ω) (r₀ : Fin (V.k x) → Coin) (c : E.C x),
    (ownLog H (run H P V x ω) (E.ext x r₀ c)).length ≤ calls x

/-- **KS read at a time budget**, at every statement in `Scope`: `KSLinOn`'s bound with each hash term read at the
budget `T`, the SHA-512 compressions one run of the prover makes, its own and the oracle's, plus the size of any state
it carries, which the reader computes from the prover's time (`c · time + d` at a stated rate), as PoUW's bounds per
prover read `ecrBound`. The birthday term is `√γ · sbB` at `B` declared at `2T`, the binding term `ecrBound` at `B_ext`
declared at `T` a rerun, at `2·T·work`. Either is `⊤` at a prover whose finder fails its assumption, so the statement
takes no hypothesis on `H`; where both hold it is `κ + 2T·√γ/2^256.5 + 2T·work/2^256`, linear in `T`, so the terms of
the runs of a monitoring period sum at the adversary's SHA-512 work over the period. -/
def KSAt (H : Msg → Msg) (V : Verifier Coin) (Scope : Msg → Prop) (R : Rel W) (E : Extractor Coin W V)
    (κ γ work : Msg → ℝ) : Prop :=
  ∀ x, Scope x → ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (T : ℕ),
    ENNReal.ofReal (PrBad H V R E x P) ≤
      ENNReal.ofReal (κ x) + ENNReal.ofReal (Real.sqrt (γ x)) * sbB H V P x (2 * T) +
        ecrBound H (BextAt H V E T Ω P x) (2 * T * work x)

/-- **Cost, `B_ext` at a time budget**: at every budget `T ≥ 1`, both extractions' own calls and `T` a rerun come to at
most `2·T·work(x)` in expectation, so `KSAt`'s binding term is `2T·work/2^256` wherever `ExpectedCR` holds at its
finder. -/
def BextAtCostLe (H : Msg → Msg) (V : Verifier Coin) (E : Extractor Coin W V) (work : Msg → ℝ) : Prop :=
  ∀ (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (T : ℕ), 1 ≤ T →
    Ex (fun s => ((BextAt H V E T Ω P x).cost s : ℝ)) ≤ 2 * T * work x

end Finder

/-! ### Zero knowledge given the published leaf layer -/

/-- **A simulator given the leakage**: its coins, and its program, which also receives what the protocol publishes
beyond `x` (here the trace's leaf layer). -/
structure SimLeak where
  C : Msg → Type
  [fin : ∀ x, Fintype (C x)]
  [ne : ∀ x, Nonempty (C x)]
  prog : (x : Msg) → List Msg → C x → SP (List (Msg × Out))

attribute [instance] SimLeak.fin SimLeak.ne

/-- The simulator given the leakage `l x` at each statement `x`, as the spec's `Simulator`. -/
def SimLeak.given (S : SimLeak) (l : Msg → List Msg) : Simulator where
  C := S.C
  prog x c := S.prog x (l x) c

/-- **The published leaf layer** at `(x, w)`: the leaves under the statement's roots of the values `w` opens, empty on a
statement that doesn't parse. A fixed function of `(x, w)`. -/
def leafLayer (H : Msg → Msg) (Cm : CommitmentScheme Bool) (parse : Msg → Option AuditStatement) (x : Msg) (w : Wit) : List Msg :=
  match parse x with
  | some s => Cm.layer H s.roots w.vals
  | none => []

/-- **A masked deployment**: a deployment with the reference prover's tape in its masked runs at each statement `x`, a
nonce and the rest of `x`'s (`Nonce x`, `Rest x`, finite types), read as its tape `tapeOf x (ν, r)`, since each
session runs on a tape of its own; and the simulator `ZK` is at (`S`). -/
structure ZkDeploy (Coin Col T : Type) extends Deploy Coin Col T where
  Nonce : Msg → Type
  Rest : Msg → Type
  [nonceFin : ∀ x, Fintype (Nonce x)]
  [nonceNe : ∀ x, Nonempty (Nonce x)]
  [restFin : ∀ x, Fintype (Rest x)]
  [restNe : ∀ x, Nonempty (Rest x)]
  tapeOf : (x : Msg) → Nonce x × Rest x → Tape
  S : SimLeak

attribute [instance] ZkDeploy.nonceFin ZkDeploy.nonceNe ZkDeploy.restFin ZkDeploy.restNe

/-- **The masked deployment's reference prover** at the witness `w` and the statement `x`: `D.P w` at the tape
`tapeOf x (ν, r)`. -/
def ZkDeploy.prover {Coin Col T : Type} (D : ZkDeploy Coin Col T) (w : Wit) (x : Msg) :
    Prover (D.Nonce x × D.Rest x) :=
  (D.P w).comap (D.tapeOf x)

/-! ## 6. Claims -/

section Claims

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

/-- The sessions' knowledge error, Flock's at the inner numbers: `κ_F = 4·ε_table + 2^{−m′} + 2k′/K′ + e^{−k′/8}`. -/
noncomputable def κF (D : Deploy Coin Col T) (x : Msg) : ℝ :=
  4 * D.εtable x + 2 ^ (-(D.innerWaits x : ℝ)) + 2 * D.innerK x / D.innerCap x + Real.exp (-(D.innerK x : ℝ) / 8)

/-- The waits' slack factor, `c = 2^{2^{−j}} < 1 + 2^{−j}` at `j = slackExp`: each wait's (`Deploy.k`). -/
noncomputable def slack : ℝ := 2 ^ ((2 : ℝ) ^ (-(slackExp : ℝ)))

/-- The design's miss at `t` at `x`, clamped at `0`; `0` on a statement that doesn't parse. In scope the clamp changes
nothing, since a well-formed design's `P_a` is a probability; it makes the count form the knowledge-soundness bound at a
`δ ≥ 0` without reading the design (`SamplingGeneric.Compiled.missAt` at this deployment). -/
noncomputable def missAt (D : Deploy Coin Col T) (t : ℕ) (x : Msg) : ℝ :=
  match D.parse x with
  | some s => max 0 (s.design.miss (D.part s) t)
  | none => 0

/-- **The sampling scope, on the statement's bytes**: it parses, its query evaluates, and the partition and statement are
in scope (`InScope`). The compiled sampled verifier's `KS`, `KSCount` and `Bits` at this deployment
(`SamplingGeneric.ofDeploy`) and `Complete` hold on it. It excludes every design whose `P_a` is not its
sample's miss probability (`Design.WellFormed`): `subset K` with `K > N` (`P_a = 0` at every set, so no witness is in `R`),
a Bernoulli rule with `den = 0` or `num > den`, a stratified rule outside `1 ≤ K ≤ N` or with an empty stratum, a work
rule whose weights or floors don't match its strata, with a zero floor or with no work. That the verifier refuses what
is out of it is `ScopeMatches`. -/
def Scope (D : Deploy Coin Col T) (x : Msg) : Prop :=
  ∃ s P, D.parse x = some s ∧ D.partition s = some P ∧ InScope P s

/-- **Sampling: binding** of the trace's commitment, as a collision reduction, at every `H`. -/
def Binding (D : Deploy Coin Col T) : Prop := ∀ H, D.commit.Binds H

/-- **Sampling: the universal unit**: the program of class `c` (`UniversalUnit_v1{G, N_IN, N_OUT}`) maps every private
circuit of the class, as its arguments, and inputs `x` to the private circuit's outputs on `x`. A private circuit runs
as a program built of universal units, its description committed values like any other input. -/
def Universal (D : Deploy Coin Col T) : Prop :=
  ∀ (c : SizeClass) (n : PrivateCircuit) (x : List Bool), n.Fits c → x.length = c.nIn →
    (D.universal c).eval (n.args c ++ x) = n.eval x

/-- **Sampling: Complete**, on the statements in scope, when no unit is faulty. It needs the verifier to check before
the sample that every unit's session has a form Flock proves (`Flock.InScope`); that check waits for the reference
prover. It is of the deployed record's reference prover (`D.P`), at the deployed error (`D.δc H`). -/
def Complete (H : Msg → Msg) (D : Deploy Coin Col T) : Prop :=
  Claims.Complete H D.V D.P (Rhon H D.commit D.parse D.partition) (Scope D) (D.δc H)

/-- **Sampling: the readable scope is the verifier's**: the scope check the sampled verifier runs before it samples is
`Scope`, so the compiled sampled verifier's `KS`, `KSCount` and `Bits` at this deployment, which hold on `Scope`, cover
every statement it takes. Binding `scopeCheck` to the compiled sampled verifier is the code side's, and open. -/
def ScopeMatches (D : Deploy Coin Col T) : Prop := Claims.ScopeMatches (Scope D) D.scopeCheck

/-- **Sampling: zero knowledge, relative to the published leaf layer**, of the masked deployment's reference prover
(`D.prover`, the deployment's `D.P` on its masked tapes) with masked sessions when no unit is faulty, at the deployment's
simulator (`D.S`), against every malicious verifier of the sampled verifier's rounds. This is `ZKmalRx`, on each
statement's tape, with the simulator given the trace's leaf layer under the roots (`leafLayer`), which the protocol
publishes: the leakage is that whole layer, a fixed function of `(x, w)`. `ZKmalRx` runs one simulator per statement,
so the claim holds at every assignment `l` of a leaf layer to each statement, on the honest pairs whose layer is
`l x`. Its binding term is `γ` times `Fork`'s collision probability among the honest prover's calls in two real runs
on one nonce, never the simulator's calls. -/
def ZK (H : Msg → Msg) (D : ZkDeploy Coin Col T) (εzk γ : Msg → ℝ) : Prop :=
  D.flock.zk = true ∧ ∀ l : Msg → List Msg,
    ZKmalRx D.Nonce D.Rest H D.V.k D.prover
      (fun x w => Rhon H D.commit D.parse D.partition x w ∧ leafLayer H D.commit D.parse x w = l x) (D.S.given l) εzk γ

/-- **Sampling: nontriviality**: recorded honest runs, each of a statement the scope check takes, that the verifier
accepts. -/
def Nontrivial (H : Msg → Msg) (D : Deploy Coin Col T) (recs : List (Recorded D.V)) : Prop :=
  Claims.Nontrivial H D.V recs ∧ ∀ e ∈ recs, D.scopeCheck e.x = true

end Claims

end Claims.Sampling
