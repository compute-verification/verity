import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Basic.ENNReal.BigOperators
import Mathlib.Basic.ENNReal.Inv
import Mathlib.Order.CompleteLattice.Basic
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
* `CallAnalysis`: what the analysis of one call fixes, in the randomized-extraction form: from each state of the call's
  session, the probabilities that it accepts while a drawn unit's extraction fails (`ks`) or succeeds but disagrees with
  a transcript on the unit's wires (`link`). The transcripts and their wrong units are the window's;
* `thenExp`, `along`: a quantity of each call's session state, summed over the calls along the window's run;
* the premises of the window's composition: `EscapesWithin` (the draw's escapes within `η` of the product law's at every
  registration), `CallAnalysis.KnowledgeSound` (a call's extraction, from every state of its session) and
  `LinkSoundIn` (a call's link term in the window, against the window's committed transcript).
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

/-! ## The window -/

/-- A window's units: a call `c < C` and one of its `n c` units. -/
abbrev WUnit {C : ℕ} (n : Fin C → ℕ) := (c : Fin C) × Fin (n c)

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
* `link S R τ u X`: `Pr[it accepts ∧ extraction of u succeeds but differs from the transcript X on u's wires]`;
* `cover`: when `u` is drawn and wrong in `X`, `Pr[accept] ≤ ks + link`.
The transcripts `Tr` and their wrong units `wrong` are the window's. -/
structure CallAnalysis {Reg Tr : Type} (wrong : Tr → Finset (WUnit n)) (c : Fin C)
    (sess : Finset (Fin (n c)) → Reg → Game Bool) where
  ks : (S : Finset (Fin (n c))) → (R : Reg) → Strategy (sess S R) → Fin (n c) → ℝ≥0∞
  link : (S : Finset (Fin (n c))) → (R : Reg) → Strategy (sess S R) → Fin (n c) → Tr → ℝ≥0∞
  cover : ∀ S R (τ : Strategy (sess S R)) u X, u ∈ S → (⟨c, u⟩ : WUnit n) ∈ wrong X →
    prob (· = true) (sess S R) τ ≤ ks S R τ u + link S R τ u X

/-- **A call's knowledge soundness, from every state of its session** (premise `ε_c`): whatever the draw, the
registration and the state the prover brings to the session, `Pr[accept ∧ extraction of a drawn unit fails] ≤ ε_c`.
This is a per-unit proof system's knowledge soundness, which holds for every prover; in the window the state is the one
the calls before and the whole draw left. -/
def CallAnalysis.KnowledgeSound {Reg Tr : Type} {wrong : Tr → Finset (WUnit n)} {c : Fin C}
    {sess : Finset (Fin (n c)) → Reg → Game Bool} (A : CallAnalysis wrong c sess) (ε : ℝ≥0∞) : Prop :=
  ∀ S R (τ : Strategy (sess S R)) u, u ∈ S → A.ks S R τ u ≤ ε

/-- **A call's link term in the window** (premise `δ_c`): for every window prover state after registration and every
rule that picks a drawn unit before the sessions' first coin, the average over the draw of call `c`'s `link` at the
picked unit if it is one of `c`'s, at the state the window's run brings to `c`'s session (`along`), against the window's
committed transcript, is at most `δ_c`. The committed transcript is a function of the prover's state at registration, so
it is the same for every call and every draw. -/
def LinkSoundIn {Reg Tr : Type} {wrong : Tr → Finset (WUnit n)} (Lw : Reg → Law (WUnit n))
    (sess : (c : Fin C) → Finset (Fin (n c)) → Reg → Game Bool)
    (committed : (R : Reg) → Cont Lw (windowSession sess) R → Tr) (A : (c : Fin C) → CallAnalysis wrong c (sess c))
    (c : Fin C) (δ : ℝ≥0∞) : Prop :=
  ∀ (R : Reg) (τ : Cont Lw (windowSession sess) R) (tgt : (Lw R).Ω → Option (WUnit n)),
    avg (fun ω => along (fun c' => sess c' (slice ((Lw R).draw ω) c') R)
      (fun c' τc => if c' = c then
          atTgt (slice ((Lw R).draw ω) c') (pick (tgt ω) c')
            fun u => (A c').link (slice ((Lw R).draw ω) c') R τc u (committed R τ)
        else 0)
      (List.finRange C) (τ ω)) ≤ δ

/-- **The window's draw is within `η` of the product law** at every registration (premise): the escape of every unit set
under the law the verifier runs at `R` is at most its escape under the calls' laws at `R`, drawn independently, plus `η`,
the same `η` for every `R`. For the keyed draw this is the pseudorandomness of the keyed streams at the context
`digest(R)` (A4, `prf/sha-512`), from a fresh uniform source (A5), with a distinct context for every call (M1). -/
def EscapesWithin {Reg : Type} (Lw : Reg → Law (WUnit n)) (L : Reg → (c : Fin C) → Law (Fin (n c))) (η : ℝ≥0∞) :
    Prop :=
  ∀ R B, (Lw R).escape B ≤ (Law.prod (L R)).escape B + η

end Window

end Pouw.Window
