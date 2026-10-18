import Definitions.Hash.Tree

/-!
# A tree under a hash keyed by the epoch's live coin: the window game (trusted)

The game behind window-bounded binding, after the rate audit's (`Definitions/Accounting/RateAudit.lean`). A keyed hash
family `h : K → Msg → Digest` (SHA-512 or a round-reduced SHA-512 from the midstate of a block holding the key, say)
hashes every leaf and node of a commitment made in an epoch. The key `k` is the epoch's live coin: uniform, and drawn
after everything the prover computed before the epoch. The prover then holds `k`, so it is a *key-holder*: binding
needs collisions of `h k` to be hard for someone who knows `k`, within the window. A linear or universal family fails
exactly here (a holder of `S` opens `x` and `x + d` for any `d` in the kernel of `x ↦ S·x`); keying helps only against
work done before `k` was drawn.

**The prover.** Its coins are drawn independently of the key, and its program (`Prog`) runs in the window from the key
and its coins: each step evaluates `h k` on one input (`eval`) or returns. It ends with a claim: a root, a position and
two openings there. It evaluates at most `q` times in every world (`Prover.bounded`). The window is from the key's draw
to the last opening it may be asked for: one epoch when one key serves every commitment in it, and a whole session for
a store that must stay bound that long.

**Compute.** A query stands for one evaluation of `h k` at `c_eval` units of work, so a prover with compute `C` in the
window has `q = C / c_eval`, as the rate audit counts tags at `c_tag`. The model counts what a program asks through
`eval`: a program that evaluates `h k` inline, by its own code, has those evaluations uncounted, and the assumption
(`Assumptions.KeyedWindowCR`) is stated at the finder a prover gives (`reduce`), whose count is its program's. That is
where cryptanalysis lives: a prover with a shortcut to collisions of `h k` gives a finder that beats the assumption's
bound, so the bound fails there, and the guarantee says nothing about that prover.

**Precomputation.** Anything the prover computed before the epoch is in its definition (its program may be any
function of the key and its coins). A program can hard-code a collision for every key, so a bound over every program
would be false; the assumption is applied to the finder a prover gives, like `Claims.StrictBirthday` at `Claims.B`.
Against generic search, `S` bits of advice gain about `S/|K|` (Coretti–Dodis–Guo–Steinberger 2018), nothing at
`|K| = 2^512`.

**Windows.** One window is stated. A prover of epoch `i` may depend on every other epoch's key, as the rate audit's
window `i` depends on the other windows' coins: with those keys fixed, it is one prover of this game.
-/

namespace Hash.Keyed

/-- A program in the window, ending in a value of type `α`: each step evaluates the keyed hash at the epoch's key on
one input, and waits on the digest. -/
inductive Prog (α : Type) : Type where
  | done : α → Prog α
  | eval : Msg → (Digest → Prog α) → Prog α

namespace Prog

variable {α β : Type}

/-- The program's value when `f` answers its evaluations. -/
def run (f : Msg → Digest) : Prog α → α
  | done a => a
  | eval x k => (k (f x)).run f

/-- The inputs the program evaluates when `f` answers, in order. -/
def evals (f : Msg → Digest) : Prog α → List Msg
  | done _ => []
  | eval x k => x :: (k (f x)).evals f

/-- Run `p`, then the program its value picks. -/
def bind : Prog α → (α → Prog β) → Prog β
  | done a, g => g a
  | eval x k, g => eval x fun y => (k y).bind g

/-- Run `p` and apply `g` to its value. -/
def map (g : α → β) (p : Prog α) : Prog β := p.bind fun a => done (g a)

end Prog

/-- The tree's format: the leaf tag, the node tag and the depth. -/
structure Format where
  leafTag : Msg
  nodeTag : Msg
  depth : ℕ

/-- An opening: the leaf's value bytes and the sibling digests from the leaf level up. -/
structure Opening where
  v : Msg
  sibs : List Digest

/-- A leaf's input: the tag, then the value. -/
def leafIn (F : Format) (v : Msg) : Msg := F.leafTag ++ v

/-- The verifier accepts an opening at position `i` against root `r`, under the hash `H` (`h k` at the epoch's key). -/
def accepts (F : Format) (H : Msg → Digest) (r : Digest) (i : ℕ) (o : Opening) : Prop :=
  Path.accepts H F.nodeTag F.depth r i (leafIn F o.v) o.sibs

/-- What a prover outputs: a root, a position and two openings there. -/
abbrev Claim : Type := Digest × ℕ × Opening × Opening

/-- **A binding break** under `H`: both openings accepted at one position against one root, with different values. -/
def Break (F : Format) (H : Msg → Digest) (c : Claim) : Prop :=
  accepts F H c.1 c.2.1 c.2.2.1 ∧ accepts F H c.1 c.2.1 c.2.2.2 ∧ c.2.2.1.v ≠ c.2.2.2.v

/-- A prover in the window of a keyed hash with keys `K`, making at most `q` evaluations: its coins, drawn
independently of the key, and its program from the key and its coins. -/
structure Prover (K : Type) (q : ℕ) where
  Ω : Type
  [fin : Fintype Ω]
  [ne : Nonempty Ω]
  prog : K → Ω → Prog Claim
  bounded : ∀ (k : K) (ω : Ω) (f : Msg → Digest), ((prog k ω).evals f).length ≤ q

attribute [instance] Prover.fin Prover.ne

/-- A collision finder against a keyed hash with keys `K`: its coins, and its program from the key and its coins. Its
cost is the evaluations its program makes. -/
structure Finder (K : Type) where
  S : Type
  [fin : Fintype S]
  [ne : Nonempty S]
  prog : K → S → Prog (Option (Msg × Msg))

attribute [instance] Finder.fin Finder.ne

/-- An opening's node inputs from digest `x` at position `i` up the siblings, each evaluated, with its digest. -/
def climbProg (tag : Msg) : ℕ → Digest → List Digest → Prog (List (Msg × Digest))
  | _, _, [] => .done []
  | i, x, s :: ss =>
    let m := Path.nodeIn tag (i % 2 = 1) x s
    .eval m fun y => (climbProg tag (i / 2) y ss).map (List.cons (m, y))

/-- An opening's check, evaluated through the program: its leaf input, then its node inputs up at most `depth`
siblings, each with its digest (`depth + 1` evaluations at most). -/
def openProg (F : Format) (i : ℕ) (o : Opening) : Prog (List (Msg × Digest)) :=
  .eval (leafIn F o.v) fun x => (climbProg F.nodeTag i x (o.sibs.take F.depth)).map (List.cons (leafIn F o.v, x))

/-- The finder a prover gives: run it, evaluate both openings' checks (`2·(depth + 1)` evaluations at most), and output
the first collision among the inputs those checks evaluated. -/
def reduce (F : Format) {K : Type} {q : ℕ} (V : Prover K q) : Finder K where
  S := V.Ω
  prog k ω := (V.prog k ω).bind fun c =>
    (openProg F c.2.1 c.2.2.1).bind fun l₁ => (openProg F c.2.1 c.2.2.2).map fun l₂ => findColl (l₁ ++ l₂)

end Hash.Keyed
