import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Basic.ENNReal.BigOperators
import Mathlib.Basic.ENNReal.Inv
import Mathlib.Order.CompleteLattice.Basic
import Mathlib.Data.Fintype.Perm
import Mathlib.Data.Fintype.Pi
import Mathlib.Data.Fintype.Sigma

/-!
# A window of calls: one draw that depends on the registration, then the calls' sessions in order (trusted)

A PoUW window holds `C` matmul calls. The prover registers the window (`R`, its receipt, which lists every call's
commitment). The verifier then draws every call's units at once, by a law that depends on `R`: the window key's context is
the receipt's digest (C1). Then each call's drawn units are proved in a session of their own, the calls in order, each
session reading only its call's share of the draw and `R`. The window accepts when every call's session accepts.

This is the vocabulary of that game, in the form of C-Flock's one-stage audit (`FlockSoundness.Audit`, which the window
guarantee `audit_window_split_of_record` is stated over), on its own so that a guarantee can be stated over it:

* `Game`, `Strategy`, `prob`: a public-coin interaction with live coins, a deterministic prover strategy, and the exact
  probability of an outcome over the verifier's coins; `avg` and `prCoin`, the uniform average and probability;
* `Law`, `Law.escape`: a draw law over units, and the probability that its draw misses a set; `Law.prod`, the calls' laws
  each drawn on its own coins, the product law that distinct contexts give (M1);
* `auditReg`: the registration, then the coin of the law at that registration, then the session on its draw (C1);
* `WUnit`, `slice`, `seqCalls`, `windowSession`: a window's units (a call and a unit of that call), a call's share of a
  draw, and the calls' sessions in order;
* `SeedIn`, `UnitStmts`: a unit's seed inputs (tag, salt, position, root(A_j), root(B_j)) and the units' statements, each
  a self-contained tile (decision 106) whose position and roots are functions of its place and the committed transcript
  (`UnitStmts.seedIn`), and `UnitStmts.wrong`, the units whose committed outputs aren't their function of their inputs;
* `harmSets`: PoUW's harm family, the wrong-unit sets holding a share of the window's work by the scheme's work table;
* `CallAnalysis`: what the analysis of one call fixes, in the randomized-extraction form: from each state of the call's
  session, the probabilities that it accepts while a drawn unit's extraction fails (`ks`) or succeeds but disagrees with
  a transcript on the unit's wires (`link`). The transcripts and the units' statements are the window's;
* `thenExp`, `along`: a quantity of each call's session state, summed over the calls along the window's run;
* the premises of the window's composition, each in the shape of a compiled bound, a constant plus a term of the
  prover's state: `EscapesWithin` (the draw's escapes on a family within `η` of the product law's at every
  registration), `CallAnalysis.KnowledgeSound` (a call's extraction, `κ_c` plus a term `g_c` of its session's state)
  and `LinkSoundIn` (a call's link term in the window, against the window's committed transcript, a function of the
  window's state);
* `coll`: the window's collision term, the calls' state terms `g_c` summed along the window's run, in expectation over
  the draw;
* `ServedTie` (premise `δ_in`): the window accepts while its committed transcript isn't anchored and served with
  probability at most a term of the window's state;
* the deployed draw, sampled proofs' service (`verity/core/protocols/pouw/window.py`): `subsetDraw`, one uniform `K`-of-`N`
  tile subset; `sessions`, `chunk` and `serviceLaw`, its rank-ordered sessions, each a call of the window whose units are
  the `N` tiles; `share` and `sessionLaw`, each session's own uniform law; `CopyClosed`, the unit sets that hold a tile
  in every session or in none; `UnitStmts.copies`, the tiles' statements read in every session, and `WrongCopyClosed`,
  statements whose wrong sets are copy-closed.
-/

namespace Pouw.Window

open scoped ENNReal

/-! ## Live-coin interactions -/

/-- The verifier's side of a public-coin interaction with live coins: the prover sends a message of its choice (`send`),
the verifier draws a uniform coin after every earlier message is fixed (`coin`), or the interaction ends with the
verifier's output (`ret`). -/
inductive Game (α : Type) : Type 1 where
  | ret : α → Game α
  | send (M : Type) : (M → Game α) → Game α
  | coin (C : Type) [Fintype C] [Nonempty C] : (C → Game α) → Game α

namespace Game

variable {α β : Type}

/-- Sequencing: run `g`, then continue with `f` applied to its output. -/
def bind : Game α → (α → Game β) → Game β
  | .ret a, f => f a
  | .send M k, f => .send M fun m => bind (k m) f
  | @Game.coin _ C iF iN k, f => @Game.coin _ C iF iN fun c => bind (k c) f

/-- Relabel the output. -/
def map (f : α → β) (g : Game α) : Game β := g.bind fun a => .ret (f a)

end Game

open Game

variable {α β : Type}

/-- A deterministic prover strategy: a message at every `send` node, for every history (the node's position in the tree);
after a coin it may depend on that coin. -/
def Strategy : Game α → Type
  | .ret _ => PUnit
  | .send _ k => (m : _) × Strategy (k m)
  | @Game.coin _ C _ _ k => (c : C) → Strategy (k c)

/-- A strategy for the relabelled game is a strategy for the game. -/
def Strategy.ofMap (f : α → β) : (g : Game α) → Strategy (g.map f) → Strategy g
  | .ret _, _ => PUnit.unit
  | .send _ k, s => ⟨s.1, Strategy.ofMap f (k s.1) s.2⟩
  | @Game.coin _ _ _ _ k, s => fun c => Strategy.ofMap f (k c) (s c)

/-- The moves a strategy for `g.bind f` makes in `g`. -/
def Strategy.head {f : α → Game β} : (g : Game α) → Strategy (g.bind f) → Strategy g
  | .ret _, _ => PUnit.unit
  | .send _ k, s => ⟨s.1, Strategy.head (k s.1) s.2⟩
  | @Game.coin _ _ _ _ k, s => fun c => Strategy.head (k c) (s c)

open Classical in
/-- The exact probability, over the verifier's independent uniform coins, that the interaction between the verifier `g`
and the prover strategy `s` ends with an output satisfying `p`. -/
noncomputable def prob (p : α → Prop) : (g : Game α) → Strategy g → ℝ≥0∞
  | .ret a, _ => if p a then 1 else 0
  | .send _ k, s => prob p (k s.1) s.2
  | @Game.coin _ C iF _ k, s => ∑ c : C, (@Fintype.card C iF : ℝ≥0∞)⁻¹ * prob p (k c) (s c)

/-- **The expectation over `g`'s run of a quantity of where it leaves the strategy**: run `g` against the moves `s` makes
in it, and read `F` at `g`'s output `a` and the strategy `s` plays in `f a` from there. -/
noncomputable def thenExp {f : α → Game β} (F : (a : α) → Strategy (f a) → ℝ≥0∞) :
    (g : Game α) → Strategy (g.bind f) → ℝ≥0∞
  | .ret a, s => F a s
  | .send _ k, s => thenExp F (k s.1) s.2
  | @Game.coin _ C iF _ k, s => ∑ c : C, (@Fintype.card C iF : ℝ≥0∞)⁻¹ * thenExp F (k c) (s c)

/-- The uniform average over a finite type. -/
noncomputable def avg {Ω : Type} [Fintype Ω] (f : Ω → ℝ≥0∞) : ℝ≥0∞ :=
  ∑ c : Ω, (Fintype.card Ω : ℝ≥0∞)⁻¹ * f c

open Classical in
/-- The probability that a uniform coin satisfies `p`. -/
noncomputable def prCoin {Ω : Type} [Fintype Ω] (p : Ω → Prop) : ℝ≥0∞ :=
  avg fun c => if p c then 1 else 0

/-! ## Draw laws and the audit whose draw depends on the registration -/

/-- A draw law over the units `U`: the verifier's uniform coins, and the unit set each draws. -/
structure Law (U : Type) where
  Ω : Type
  [fintype : Fintype Ω]
  [nonempty : Nonempty Ω]
  draw : Ω → Finset U

attribute [instance] Law.fintype Law.nonempty

/-- The probability that a draw misses every unit of `B`. -/
noncomputable def Law.escape {U : Type} (L : Law U) (B : Finset U) : ℝ≥0∞ :=
  prCoin fun ω => Disjoint (L.draw ω) B

/-- **The audit whose draw depends on the registration** (C1): the prover registers `R`; the verifier draws by `L R`;
the session on the drawn units runs against `R`. Output: the verdict and the draw. -/
def auditReg {Reg U : Type} (L : Reg → Law U) (session : Finset U → Reg → Game Bool) : Game (Bool × Finset U) :=
  .send Reg fun R => .coin (L R).Ω fun ω => (session ((L R).draw ω) R).map fun ok => (ok, (L R).draw ω)

/-- A prover's state after registering `R`: its session strategy for every draw. -/
abbrev Cont {Reg U : Type} (L : Reg → Law U) (session : Finset U → Reg → Game Bool) (R : Reg) :=
  ∀ ω : (L R).Ω, Strategy (session ((L R).draw ω) R)

section Audit

variable {Reg U : Type} {L : Reg → Law U} {session : Finset U → Reg → Game Bool}

/-- The registration of an audit strategy. -/
def reg (σ : Strategy (auditReg L session)) : Reg := σ.1

/-- The prover's state after registration. -/
def cont (σ : Strategy (auditReg L session)) : Cont L session (reg σ) :=
  fun ω => Strategy.ofMap _ _ (σ.2 ω)

end Audit

/-! ## A window's units, their statements and PoUW's harm family -/

/-- A window's units: a call `c < C` and one of its `n c` units. -/
abbrev WUnit {C : ℕ} (n : Fin C → ℕ) := (c : Fin C) × Fin (n c)

/-- **A unit's seed inputs** (decisions 101, 106): what the seed of the noise it reads hashes,
`SHA-512(tag ‖ salt ‖ position ‖ root(A_j) ‖ root(B_j))` for a unit of matmul `j`. -/
structure SeedIn (Tag Salt Pos Root : Type) where
  tag : Tag
  salt : Salt
  pos : Pos
  rootA : Root
  rootB : Root

/-- **The window's units, as their statements read them** (decision 106). A unit is a self-contained `r × c` tile of its
call's matmul `A_c·B_cᵀ`: it checks its seed, derives from it the noise it reads, and computes its `r·c` dots. Every seed
input is a public or committed input of the unit, bound before the draw:
* `tag` (protocol structure) and `salt` (the epoch's live coin, opened before the window) are public, one each for the
  window;
* `rootA X v` and `rootB X v`, the operand roots of unit `v`'s matmul, are committed: read from the committed transcript
  `X`, which is fixed at registration, and bound by the unit's proof through its rows' opening path to the window root;
* `pos X v`, the position, is a function of the unit's place `v` (the verifier's draw) and `X` alone (`seedIn`), so no
  message of a session sets it: a prover that could pick a position inside a unit could pick its seed. On the service
  path the place is a drawn tile index `t`, a public input of the unit's instance, and the tile's place in its matmul is
  `X`'s entry at `t` under the registered window root (`tile_at` in `window.py`).

`computes v s X`: `X`'s committed outputs at `v` are `v`'s function of its inputs, `X`'s values at `v`'s input wires
and the noise of the seed inputs `s`. -/
structure UnitStmts (Tr : Type) {C : ℕ} (n : Fin C → ℕ) where
  Tag : Type
  Salt : Type
  Pos : Type
  Root : Type
  tag : Tag
  salt : Salt
  pos : Tr → WUnit n → Pos
  rootA : Tr → WUnit n → Root
  rootB : Tr → WUnit n → Root
  computes : WUnit n → SeedIn Tag Salt Pos Root → Tr → Prop

namespace UnitStmts

variable {Tr : Type} {C : ℕ} {n : Fin C → ℕ}

/-- Unit `v`'s seed inputs in the transcript `X`: the window's tag and salt, and its position and operand roots, read
at its place `v` in `X`. -/
def seedIn (U : UnitStmts Tr n) (X : Tr) (v : WUnit n) : SeedIn U.Tag U.Salt U.Pos U.Root :=
  ⟨U.tag, U.salt, U.pos X v, U.rootA X v, U.rootB X v⟩

open Classical in
/-- **The wrong units of a transcript**: those whose committed outputs in `X` aren't their function of their inputs at
their seed inputs. Wrong is about values, not work: a deficient unit whose committed outputs came out right by luck isn't
wrong. -/
noncomputable def wrong (U : UnitStmts Tr n) (X : Tr) : Finset (WUnit n) :=
  Finset.univ.filter fun v => ¬ U.computes v (U.seedIn X v) X

end UnitStmts

/-- **PoUW's harm family** (the `𝓑` the window audit reads for compute): the wrong-unit sets whose work is at least the
share `ε` of the window's, `∑_{v ∈ B} W_ref(v) ≥ ε · W` with `W = ∑_v W_ref(v)`, where `Wref` is the scheme's work
table read at each unit's template. Under decision 106 a unit is a self-contained `r × c` tile (`UnitStmts`), so its
harm is its own `W_ref(tile)`, `r·c` dots. That is fan 1, since a wrong tile is charged its own work and no other
unit's, and there is no steering term, since no unit sets the noise another reads. -/
def harmSets {C : ℕ} {n : Fin C → ℕ} (Wref : WUnit n → ℕ) (ε : ℝ≥0∞) (B : Finset (WUnit n)) : Prop :=
  ε * ((∑ v, Wref v : ℕ) : ℝ≥0∞) ≤ ((∑ v ∈ B, Wref v : ℕ) : ℝ≥0∞)

/-! ## The window -/

section Window

variable {C : ℕ} {n : Fin C → ℕ}

/-- **The product law** (M1): each call draws by its own law, on its own coins. -/
def Law.prod (L : (c : Fin C) → Law (Fin (n c))) : Law (WUnit n) where
  Ω := (c : Fin C) → (L c).Ω
  draw ω := Finset.univ.sigma fun c => (L c).draw (ω c)

/-- Call `c`'s share of a window's draw. -/
def slice (S : Finset (WUnit n)) (c : Fin C) : Finset (Fin (n c)) :=
  Finset.univ.filter fun u => (⟨c, u⟩ : WUnit n) ∈ S

/-- A unit picked in the window, as a unit of call `c` if it is one of `c`'s. -/
def pick (t : Option (WUnit n)) (c : Fin C) : Option (Fin (n c)) :=
  t.bind fun x => if h : x.1 = c then some (Fin.cast (congrArg n h) x.2) else none

open Classical in
/-- A per-unit quantity at the unit a rule picks, and zero when it picks none or an undrawn one. -/
noncomputable def atTgt {V : Type} (S : Finset V) (t : Option V) (f : V → ℝ≥0∞) : ℝ≥0∞ :=
  t.elim 0 fun u => if u ∈ S then f u else 0

/-- After a call's session: the rest of the window if it accepted, a rejection if not. -/
def onAccept (g : Game Bool) : Bool → Game Bool
  | true => g
  | false => .ret false

/-- A quantity of the rest of the window's run after a call that accepted, and `0` after one that rejected. -/
noncomputable def onAcceptExp (g : Game Bool) (h : Strategy g → ℝ≥0∞) : (ok : Bool) → Strategy (onAccept g ok) → ℝ≥0∞
  | true, s => h s
  | false, _ => 0

/-- **The calls' sessions in order**: the window accepts when every call's session accepts, and stops at the first that
rejects. -/
def seqCalls (sess : Fin C → Game Bool) : List (Fin C) → Game Bool
  | [] => .ret true
  | c :: cs => (sess c).bind (onAccept (seqCalls sess cs))

/-- **The window's session** on the draw `S` against the registration `R`: each call's session on its share of `S`, the
calls in order. -/
def windowSession {Reg : Type} (sess : (c : Fin C) → Finset (Fin (n c)) → Reg → Game Bool) (S : Finset (WUnit n))
    (R : Reg) : Game Bool :=
  seqCalls (fun c => sess c (slice S c) R) (List.finRange C)

/-- **A quantity of each call's session state, along the window's run**: `F c τ` at the state `τ` the window's strategy
plays in call `c`'s session, in expectation over the runs of the calls before it, counted only on the runs in which they
all accept, and summed over the calls. -/
noncomputable def along (sess : Fin C → Game Bool) (F : (c : Fin C) → Strategy (sess c) → ℝ≥0∞) :
    (cs : List (Fin C)) → Strategy (seqCalls sess cs) → ℝ≥0∞
  | [], _ => 0
  | c :: cs, s => F c (Strategy.head (f := onAccept (seqCalls sess cs)) (sess c) s) +
      thenExp (f := onAccept (seqCalls sess cs)) (onAcceptExp (seqCalls sess cs) fun s' => along sess F cs s') (sess c) s

/-- **What the analysis of one call fixes**, in the randomized-extraction form (`ExtractionAnalysis` in
`Proofs/Flock/Soundness/Audit/Extraction.lean`): from each state `τ` of call `c`'s session on its share `S` of the draw,
* `ks S R τ u`: `Pr[the session accepts ∧ extraction of the drawn unit u fails]`;
* `link S R τ u X`: `Pr[it accepts ∧ extraction of u succeeds but differs from the transcript X on u's wires]`, the
  roots its seed reads among them;
* `cover`: when `u` is drawn and wrong in `X` (`UnitStmts.wrong`), `Pr[accept] ≤ ks + link`.

Extraction is against `u`'s statement at its place `⟨c, u⟩`: its relation derives the seed from the place and what its
witness opens to the window root, the position and roots among it. A witness that opens anything but `X`'s differs from
`X` on `u`'s wires, which `link` counts, and `UnitStmts.wrong` reads the seed inputs at the same place in `X`. So
`cover` can hold only for a session that checks unit `u` at its place, never at one the prover picks. The transcripts
`Tr` and the units' statements `U` are the window's. -/
structure CallAnalysis {Reg Tr : Type} (U : UnitStmts Tr n) (c : Fin C)
    (sess : Finset (Fin (n c)) → Reg → Game Bool) where
  ks : (S : Finset (Fin (n c))) → (R : Reg) → Strategy (sess S R) → Fin (n c) → ℝ≥0∞
  link : (S : Finset (Fin (n c))) → (R : Reg) → Strategy (sess S R) → Fin (n c) → Tr → ℝ≥0∞
  cover : ∀ S R (τ : Strategy (sess S R)) u X, u ∈ S → (⟨c, u⟩ : WUnit n) ∈ U.wrong X →
    prob (· = true) (sess S R) τ ≤ ks S R τ u + link S R τ u X

/-- **A call's knowledge soundness, as a compiled table bounds it** (premises `κ_c`, `g_c`): whatever the draw, the
registration and the state `τ` the prover brings to the session, `Pr[accept ∧ extraction of a drawn unit fails]` is at
most `κ_c + g_c(τ)`. The bound depends on the state: a compiled table bounds it pointwise in the state (`ksJoint_le`,
`ksBound` in `Proofs/Flock/Soundness/Audit/FlockCompiled.lean`), with the constant `κ_c = 2^logLen/(e·Kr)` and the
state's collision terms `g_c(τ) = ε_c⁻(τ) + Kr·adv₀(τ)`. No constant bounds it for every state of a hashed session: a
state that holds a hash collision passes with extraction failing, `ks ≈ 1`, so a constant holds only at the oracle
layer. In the window the state is the one the calls before and the whole draw left, and the window's collision term
`coll` sums the calls' `g_c` along its run. -/
def CallAnalysis.KnowledgeSound {Reg Tr : Type} {U : UnitStmts Tr n} {c : Fin C}
    {sess : Finset (Fin (n c)) → Reg → Game Bool} (A : CallAnalysis U c sess) (κ : ℝ≥0∞)
    (g : (S : Finset (Fin (n c))) → (R : Reg) → Strategy (sess S R) → ℝ≥0∞) : Prop :=
  ∀ S R (τ : Strategy (sess S R)) u, u ∈ S → A.ks S R τ u ≤ κ + g S R τ

/-- **A call's link term in the window** (premise `δlink_c`): for every window prover state `τ` after the registration
`R` and every rule that picks a drawn unit before the sessions' first coin, the average over the draw of call `c`'s
`link` at the picked unit if it is one of `c`'s, at the state the window's run brings to `c`'s session (`along`),
against the window's committed transcript, is at most `δlink_c(R, τ)`, a function of the window's state as `LinkSound`'s
`δlink` is (`Proofs/Flock/Soundness/Audit/Extraction.lean`): the link's finder bounds it at the state. The committed
transcript is a function of the prover's state at registration, so it is the same for every call and every draw. The
picked unit's `link` is at its place `⟨c, u⟩` (`CallAnalysis`): no state or rule sets a unit's position, and the roots
its seed reads are the committed transcript's, fixed before the draw. -/
def LinkSoundIn {Reg Tr : Type} {U : UnitStmts Tr n} (Lw : Reg → Law (WUnit n))
    (sess : (c : Fin C) → Finset (Fin (n c)) → Reg → Game Bool)
    (committed : (R : Reg) → Cont Lw (windowSession sess) R → Tr) (A : (c : Fin C) → CallAnalysis U c (sess c))
    (c : Fin C) (δlink : (R : Reg) → Cont Lw (windowSession sess) R → ℝ≥0∞) : Prop :=
  ∀ (R : Reg) (τ : Cont Lw (windowSession sess) R) (tgt : (Lw R).Ω → Option (WUnit n)),
    avg (fun ω => along (fun c' => sess c' (slice ((Lw R).draw ω) c') R)
      (fun c' τc => if c' = c then
          atTgt (slice ((Lw R).draw ω) c') (pick (tgt ω) c')
            fun u => (A c').link (slice ((Lw R).draw ω) c') R τc u (committed R τ)
        else 0)
      (List.finRange C) (τ ω)) ≤ δlink R τ

/-- **The window's collision term** at the registration `R` and the window's state `τ`: the calls' state terms
`g c` (`CallAnalysis.KnowledgeSound`), each at the state the window's run brings to call `c`'s session, counted on the
runs in which the calls before it accept and summed over the calls (`along`), in expectation over the draw. It bounds
the term at whichever call holds the unit a rule picks, so it is one window-level term, in the place of `coll(R, τ)` in
gap (4)'s knowledge bound `κ + coll(R, τ)`. -/
noncomputable def coll {Reg : Type} (Lw : Reg → Law (WUnit n))
    (sess : (c : Fin C) → Finset (Fin (n c)) → Reg → Game Bool)
    (g : (c : Fin C) → (S : Finset (Fin (n c))) → (R : Reg) → Strategy (sess c S R) → ℝ≥0∞) (R : Reg)
    (τ : Cont Lw (windowSession sess) R) : ℝ≥0∞ :=
  avg fun ω => along (fun c => sess c (slice ((Lw R).draw ω) c) R)
    (fun c τc => g c (slice ((Lw R).draw ω) c) R τc) (List.finRange C) (τ ω)

/-- **The window's draw is within `η` of the product law on the family `𝓑`** at every registration (premise): the
escape of every unit set in `𝓑` under the law the verifier runs at `R` is at most its escape under the calls' laws at
`R`, drawn independently, plus `η`, the same `η` for every `R`. The window's guarantee reads it only at its committed
transcript's wrong units, when they're in `𝓑`. For the keyed draw this is the pseudorandomness of the keyed streams at
the context `digest(R)` (A4, `prf/sha-512`), from a fresh uniform source (A5), with a distinct context for every call
(M1), on every family. For the service's draw it holds at `η = 0` on the copy-closed sets (`CopyClosed`). -/
def EscapesWithin {Reg : Type} (Lw : Reg → Law (WUnit n)) (L : Reg → (c : Fin C) → Law (Fin (n c)))
    (𝓑 : Finset (WUnit n) → Prop) (η : ℝ≥0∞) : Prop :=
  ∀ R B, 𝓑 B → (Lw R).escape B ≤ (Law.prod (L R)).escape B + η

/-- **The tie to what's served** (premise `δ_in`, PoUW's reading of `WindowEps`): for the window's committed transcript
`committed`, the predicates `anchored` (the transcript's values at the units' input wires are the anchors', what the
served pass committed for its activations and weights) and `served` (its outputs at each unit are the served `-h3` leaf
words that unit reads, `tile_read` under the pass's `-h3` root, which the window's entries hold), every prover's window
accepts while its committed transcript isn't anchored and served with probability at most `δin(R, τ)`, a function of
the window's state. `WindowEps` bounds the wrong units of whatever transcript the analysis commits; this is what makes
that transcript the served computation's, not a self-consistent copy the prover chose. The units' statements `U` are the
verifier's own (`Check.program`, protocol structure), so the prover can't pick them either. Its discharge is C-Flock's
`AnchorsSound` (`δ_in`, `Proofs/Flock/Soundness/Audit/OneStage.lean`) for the inputs and the window statement's opening
of the `-h3` root for the outputs (`window_problem`, not yet a Program), each a binding term at the window's state. -/
def ServedTie {Reg Tr : Type} (Lw : Reg → Law (WUnit n)) (sess : (c : Fin C) → Finset (Fin (n c)) → Reg → Game Bool)
    (committed : (R : Reg) → Cont Lw (windowSession sess) R → Tr) (anchored served : Tr → Prop)
    (δin : (R : Reg) → Cont Lw (windowSession sess) R → ℝ≥0∞) : Prop :=
  ∀ σ : Strategy (auditReg Lw (windowSession sess)),
    prob (fun o => o.1 = true ∧ ¬ (anchored (committed (reg σ) (cont σ)) ∧ served (committed (reg σ) (cont σ))))
      (auditReg Lw (windowSession sess)) σ ≤ δin (reg σ) (cont σ)

end Window

/-! ## The deployed draw: sampled proofs' service

PoUW runs on sampled proofs' service (decision 102; `verity/core/protocols/pouw/window.py`): after the receipt the
verifier draws one uniform `K′`-of-`N` subset of the window's equal tiles (`subset:K′`), and the service proves the drawn
tiles in `⌈K′/cap⌉` sessions of `cap` instances, session `σ` holding the drawn indices of rank `σ·cap` to `σ·cap + cap − 1`
in ascending order and padding with the filler tile `N − 1` (`instance_ts`). A session mixes tiles of every call, so the
window's calls are the sessions: unit `⟨σ, t⟩` is tile `t` proved in session `σ`. -/

/-- The rank of `t` in `D`: how many of `D`'s units are below it. -/
def rank {N : ℕ} (D : Finset (Fin N)) (t : Fin N) : ℕ := (D.filter (· < t)).card

/-- The `K`-subset of the `N` tiles a permutation draws, the images of `0, …, K − 1`: under a uniform permutation, a
uniform `K`-subset. -/
def subsetDraw {N : ℕ} (K : ℕ) (π : Equiv.Perm (Fin N)) : Finset (Fin N) :=
  (Finset.univ.filter fun i : Fin N => i.val < K).image π

/-- The number of sessions of `cap` instances that hold `K` drawn tiles, `⌈K / cap⌉` (`sessions` in `window.py`). -/
def sessions (K cap : ℕ) : ℕ := (K + cap - 1) / cap

/-- Session `σ`'s share of the draw `D`: the drawn tiles of rank `σ·cap` to `σ·cap + cap − 1` (`instance_ts`). The
session pads its instances with the filler tile, which is credited no work. -/
def chunk {N : ℕ} (cap : ℕ) (D : Finset (Fin N)) (σ : ℕ) : Finset (Fin N) :=
  D.filter fun t => σ * cap ≤ rank D t ∧ rank D t < (σ + 1) * cap

/-- **The service's draw**: a uniform permutation's `K`-subset, each drawn tile a unit of the one session that holds its
rank. -/
def serviceLaw (N K cap : ℕ) : Law (WUnit fun _ : Fin (sessions K cap) => N) where
  Ω := Equiv.Perm (Fin N)
  draw π := Finset.univ.sigma fun σ => chunk cap (subsetDraw K π) σ.val

/-- The size of session `σ`'s share: `cap`, or what is left of `K` in the last session. -/
def share (K cap σ : ℕ) : ℕ := min cap (K - σ * cap)

/-- **Session `σ`'s own law**: a uniform `share K cap σ`-subset of the `N` tiles, on coins of its own. -/
def sessionLaw (N K cap σ : ℕ) : Law (Fin N) where
  Ω := Equiv.Perm (Fin N)
  draw π := subsetDraw (share K cap σ) π

/-- **Copy-closed unit sets**: a set that holds tile `t` in one session holds it in every session. The service's units'
statements read the tile and not the session that proves it, so a transcript's wrong units are copy-closed
(`WrongCopyClosed`). -/
def CopyClosed {S N : ℕ} (B : Finset (WUnit fun _ : Fin S => N)) : Prop :=
  ∀ (σ σ' : Fin S) (t : Fin N), (⟨σ, t⟩ : WUnit fun _ : Fin S => N) ∈ B → ⟨σ', t⟩ ∈ B

/-- **The service's units' statements**: the tiles' statements `V` (a window of one call whose units are the `N`
tiles) read in every one of `S` sessions. Unit `⟨σ, t⟩` is tile `t` proved in session `σ`; its instance's public
inputs are `t` and the window's common words (`public_inputs` in `window.py`: the window root, its domain, the sides'
domains and `domain_h3`), never `σ`, so its seed inputs and what it computes are tile `t`'s. -/
def UnitStmts.copies {Tr : Type} {N : ℕ} (V : UnitStmts Tr fun _ : Fin 1 => N) (S : ℕ) :
    UnitStmts Tr fun _ : Fin S => N where
  Tag := V.Tag
  Salt := V.Salt
  Pos := V.Pos
  Root := V.Root
  tag := V.tag
  salt := V.salt
  pos X v := V.pos X ⟨0, v.2⟩
  rootA X v := V.rootA X ⟨0, v.2⟩
  rootB X v := V.rootB X ⟨0, v.2⟩
  computes v si X := V.computes ⟨0, v.2⟩ si X

/-- **The deployed wrong sets are copy-closed** (PoUW's reading of `WindowEps`, which it reads at
`𝓑 = harmSets W_ref ε ∧ CopyClosed`, and `ServiceEscapes`'s family): at every transcript, the wrong units hold a tile
in every session or in none. It holds for the service's statements (`Pouw.Window.wrongCopyClosed_copies`), and there
`𝓑`'s copy-closed conjunct is automatic (`Pouw.Window.harmSets_copyClosed_iff`). -/
def WrongCopyClosed {Tr : Type} {S N : ℕ} (U : UnitStmts Tr fun _ : Fin S => N) : Prop :=
  ∀ X, CopyClosed (U.wrong X)

end Pouw.Window
