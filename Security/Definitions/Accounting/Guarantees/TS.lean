import Definitions.Accounting.Assumptions.TS

/-!
# The tensor–scalar model's guarantee statements (trusted)

EPS1's statements over `Accounting.TS`,
each `def G : Prop`, proved (or a `sorry` stub) as
`Accounting.SecurityProofs.TS.G` in `Proofs.Accounting.TS`:

* `ProofsSoundOfPinned`: (I8) `ProofsSound` from binding at `δ_bind` of the leaves each round commits before its σ,
  proof soundness at `δ_ZK` and the Definitions' pinning, each relation on its path's coins, and E1(c) (`RightFixed`);
  `BindingOfFork`: binding's failure squared is at most its fork form's, two runs off one commitment opening it
  differently, so `δ_bind = √δ_fork`.
* `DrawnRightOff`: a sampled window's drawn units are right at their committed values outside `S·(δ_bind + δ_ks)`, from
  binding at every position's cut, joint proof soundness and pinning; `ProofsSoundOfDrawn`: (I8) at `n_π = 1`,
  `δ_ZK = δ_ks`, from it, the share covering every unit that isn't good (a unit whose late cone the run computed, drawn
  at each unit's coin), and E1(c) per unit with its fold clause (`UnitFixed`); `DrawnBindingOfFork`: a round's binding
  failure squared is at most `|L|²` times the fork form at its positions' cuts, `L` the positions its units read, so
  `δ_bind = |L|·√δ_fork`.
* `TensorSlotCount`, `TensorSlotTight`: T1, the tensor slot count, and its tightness criterion.
* `OutputCount`: Lemma 2's count, owned committed positions at most `Σ_k ω_k n_k`.
* `FanInCount`, `SeesOfInjective`, `ThinFoldInjective`: the fan-in count, `|ι| ≤ |O| + Σ_i (sources − 1)` for a
  straight-line program whose outputs see every input; a receipt injective in each word sees it; the thin fold's is.

Probability statements put acceptance in the event: the probability that the window is accepted and the bound fails is
at most `δ`.
-/

namespace Accounting.Guarantees.TS

open Accounting.TS Accounting.Assumptions.TS Pouw.RelativeSecurity

/-- **T1 (tensor slot counting).** A program that commits `N` fresh words, each read whole by the fold from its own
slot write of a tensor instruction (freshness makes the map injective), holds the tensor pipe at least `N/ρ_T`
cycles: `Σ_i c(i) n_i ≥ N/ρ_T`. Injectivity is the premise (one word a slot): design A's packed `T_1` words, two a
s32 slot (REFEREE S14), are outside it. -/
def TensorSlotCount : Prop :=
  ∀ (ι : Type) [Fintype ι] [Nonempty ι] (w c n : ι → ℕ) (N : ℕ) (slot : Fin N → SlotWrite ι w n),
    (∀ i, 0 < c i) → Function.Injective slot → (N : ℝ) / rhoT w c ≤ ∑ i, (c i : ℝ) * n i

/-- **T1's tightness.** Honest writes `w_H` slots in `c_H` cycles (`ρ_H = w_H/c_H ≤ ρ_T`) and commits one word in
every `L/d_H` of them (`d_H ≤ L`), so `ε_tensor ≤ 1 − (ρ_H/ρ_T)(d_H/L)`; it is `0` iff `L = d_H` and `ρ_H = ρ_T`. -/
def TensorSlotTight : Prop :=
  ∀ ρH ρT dH L : ℝ, 0 < ρH → ρH ≤ ρT → 0 < dH → dH ≤ L →
    (1 - ρH / ρT * (dH / L) = 0 ↔ L = dH ∧ ρH = ρT)

/-- **(I8) from its pieces.** In 𝒫_full over a window's proofs (`ProofLayer`, `n_π` a round, each round's leaves those
it commits before its σ, its relations reading the path's coins and the words keyed by them) whose play is theirs
(`Of`: the window is accepted only if every proof is, and the ideal game's check is the proofs' relations at the
committed values on the path): if at every round binding fails with probability at most `δ_bind` (`BindingAt`; `Moved`:
an accepted round's proof opens its leaves to values other than those committed before σ_w), every proof is unsound
with probability at most `δ_ZK` (`SoundAt`), the relations pin every committed value on every path (`Pins`, a property
of the Definitions the proofs pass), and E1(c) holds with the equal shares (`RightFixed`), then (I8) `ProofsSound`
holds at `S·(δ_bind + n_π·δ_ZK)`. -/
def ProofsSoundOfPinned : Prop :=
  ∀ {K T L U V : Type} [Fintype K] [Fintype T] [Fintype L] [Fintype U] (Ω : Type) [Fintype Ω] (S nπ : ℕ)
    (P : Play Ω S K T L U) (pf : ProofLayer Ω S nπ V) (Nw δbind δzk : ℝ),
    pf.Of P → pf.BindingAt δbind → pf.SoundAt δzk → pf.Pins → RightFixed P pf Nw →
    ProofsSound P Nw δbind δzk nπ

/-- **Binding from the fork form.** For a window's proofs and a round `w`, the probability that binding fails at `w`
(`Moved`, against the values committed before σ_w), squared, is at most the probability that two runs off one
commitment, the second continuing from round `w` on fresh coins, are both accepted and open round `w`'s leaves
differently (`Forked`). So a bound `δ_fork` on the fork form, which `cr/sha-512` gives at a finder that runs the window
twice from round `w`, is binding at `δ_bind = √δ_fork`. -/
def BindingOfFork : Prop :=
  ∀ {V : Type} (Ω : Type) [Fintype Ω] (S nπ : ℕ) (pf : ProofLayer Ω S nπ V) (w : Fin S),
    prob (fun x => pf.Moved x w) ^ 2 ≤ prob (fun p : Paths Ω S × Paths Ω S => pf.Forked w p.1 p.2)

/-- **A sampled window's drawn units are right, outside binding and soundness failures.** Over `S` rounds whose coins
are `m` coins before the draws and the live draw coin with the proofs' coins, the draw naming `s` units a round, each
proved alone with its own noise's generation, its slots' values committed per position (`DrawnLayer`): if in every round
binding fails with probability at most `δ_bind` (`BindingAt`; `MovedIn`: an accepted proof of a drawn unit opens a slot
to a value other than its position's committed value, bound before the coin that follows its commitment, so a value
opened one way for the unit that wrote it and another way for a unit that reads it is a binding failure), the proofs
are sound jointly over each round's draws at `δ_ks` (`JointSound`) and each unit's relation pins its rightness
(`Pins`), then outside an event of probability at most `S·(δ_bind + δ_ks)` every drawn unit whose proof is accepted is
right at its committed values. It reads no play: what it gives is the drawn units' rightness, which the ideal game's
check reads (`ProofsSoundOfDrawn`) and a composition's folded credit takes as its premise. -/
def DrawnRightOff : Prop :=
  ∀ {J Q Pos V : Type} [Nonempty V] (C B : Type) [Fintype C] [Fintype B] (S m s : ℕ)
    (pf : DrawnLayer C B S m s J Q Pos V) (δbind δks : ℝ),
    pf.BindingAt δbind → pf.JointSound δks → pf.Pins →
    ∃ E : Paths (Paths C m × B) S → Prop, prob E ≤ S * (δbind + δks) ∧
      ∀ x, ¬ E x → ∀ w i, pf.accepted x w i → pf.right x w (pf.drawnAt x w i) (pf.view x w (pf.drawnAt x w i))

/-- **(I8) for a sampled window, from its drawn units' proofs.** Over `DrawnRightOff`'s window whose positions' writers
read them (`WriterReads`, checked for each layer), with a play that is its drawn units' proofs' (`Of`: accepted only if
every drawn unit's proof is, and accepted by the ideal game when every drawn unit is right at its committed values),
unit weights `wt` summing to the window's words `S·N_w`, the run's modes `mode` and folded words `folded`, a share
that counts every unit that isn't good (`Covers`: good is the run having computed the unit's late cone; a property of
the share's definition, proved for each game; a share charging each unit the run didn't pay for its late cone's weight
`coneWt` has it) and E1(c) per unit with its fold clause (`UnitFixed`): if binding, joint soundness and pinning hold as
in `DrawnRightOff`, then (I8) `ProofsSound` holds at `n_π = 1`, `δ_ZK = δ_ks`, `S·(δ_bind + δ_ks)`. Its event is
`DrawnRightOff`'s, so an undrawn unit's values enter only through the ideal game and the share. -/
def ProofsSoundOfDrawn : Prop :=
  ∀ {K T L U J Q Pos V : Type} [Fintype K] [Fintype T] [Fintype L] [Fintype U] [Fintype J] [Nonempty V]
    (C B : Type) [Fintype C] [Fintype B] (S m s : ℕ) (P : Play (Paths C m × B) S K T L U)
    (pf : DrawnLayer C B S m s J Q Pos V) (mode : Paths (Paths C m × B) S → Fin S × J → Mode)
    (wt : Fin S → J → ℝ) (folded : Paths (Paths C m × B) S → ℝ) (Nw δbind δks : ℝ),
    0 < S → ∑ w, ∑ j, wt w j = S * Nw → pf.WriterReads → pf.Of P → pf.BindingAt δbind →
    pf.JointSound δks → pf.Pins → pf.Covers P mode wt Nw → UnitFixed P pf mode wt folded →
      ProofsSound P Nw δbind δks 1

/-- **Binding from the fork form, at every position's cut.** For a sampled window's proofs, a round `w` and a set `L` of
positions holding every slot of round `w`'s units, the probability that binding fails in round `w` (`MovedIn`),
squared, is at most `|L|²` times the probability that, for some position of `L`, two runs off one commitment, the second
continuing from that position's cut on fresh coins, open it differently in accepted proofs of drawn units (`Forked`).
Each opening is an accepted proof's, and a proof exists only for a drawn unit (proof `i` of round `w` is drawn unit
`drawnAt x w i`'s), so accepted implies drawn by construction. A bound `δ_fork` on the fork form is binding at
`δ_bind = |L|·√δ_fork`; `δ_fork` itself is two steps a finder that runs the window twice from the cut takes: extracting
both proofs' openings with C-Flock's knowledge extractor, at `ε_ks` a proof, and then two openings of one committed
frame-v3-sha512 root at one position to different hm96 rows, a SHA-512 collision along their paths (`cr/sha-512`). -/
def DrawnBindingOfFork : Prop :=
  ∀ {J Q Pos V : Type} [Nonempty V] (C B : Type) [Fintype C] [Fintype B] (S m s : ℕ)
    (pf : DrawnLayer C B S m s J Q Pos V) (w : Fin S) (L : Finset Pos), (∀ j k, pf.pos w j k ∈ L) →
    prob (fun x => pf.MovedIn x w) ^ 2 ≤
      (L.card : ℝ) ^ 2 *
        prob (fun p : Paths (Paths C m × B) S × Paths (Paths C m × B) S => ∃ ℓ ∈ L, pf.Forked ℓ p.1 p.2)

/-- **Lemma 2 (output count).** In a straight-line program whose class-`k` instances write at most `ω_k` words each,
the owned committed positions (one per head write their values trace to) are at most `Σ_k ω_k n_k`. -/
def OutputCount : Prop :=
  ∀ {K : Type} [Fintype K] [DecidableEq K] (ω : K → ℕ) (G : ℕ) (cls : Fin G → K) (Nc : ℕ)
    (head : Fin Nc → HeadWrite ω G cls),
    ((Finset.univ.image head).card : ℝ) ≤ ∑ k, (ω k : ℝ) * classCount cls k

/-- **The fan-in count.** A straight-line program whose outputs `O` see every one of its fresh inputs has merged them
into at most `|O|` components, a node merging its sources less one: `|ι| ≤ |O| + Σ_i (|src i| − 1)`. -/
def FanInCount : Prop :=
  ∀ {ι W : Type} [Fintype ι] [Nontrivial W] [Inhabited W] (D : Dataflow ι W) (O : Finset (ι ⊕ Fin D.n)),
    D.Sees O → Fintype.card ι ≤ O.card + ∑ i, D.merges i

/-- **A receipt injective in each word sees it.** If every run's outputs `O` are `f` of its inputs, and `f` changes
whenever exactly one input does, the outputs see every input. -/
def SeesOfInjective : Prop :=
  ∀ {ι W : Type} (D : Dataflow ι W) (O : Finset (ι ⊕ Fin D.n)) (f : (ι → W) → O → W),
    (∀ x v, D.Runs x v → ∀ o : O, v o = f x o) →
    (∀ j x x', (∀ j', j' ≠ j → x j' = x' j') → x j ≠ x' j → f x ≠ f x') → D.Sees O

/-- **The thin fold's state is injective in every word of its run**, the state and the other words fixed (each step
is a bijection of the state and of each word it absorbs), and in the starting state. -/
def ThinFoldInjective : Prop :=
  ∀ (xor : Bool) (s : BitVec 32) (pre post : List (BitVec 32 × BitVec 32)) (w₁ w₁' w₂ w₂' : BitVec 32),
    (w₁ ≠ w₁' → thinFold xor s (pre ++ (w₁, w₂) :: post) ≠ thinFold xor s (pre ++ (w₁', w₂) :: post)) ∧
    (w₂ ≠ w₂' → thinFold xor s (pre ++ (w₁, w₂) :: post) ≠ thinFold xor s (pre ++ (w₁, w₂') :: post)) ∧
    ∀ s', s ≠ s' → thinFold xor s (pre ++ post) ≠ thinFold xor s' (pre ++ post)

end Accounting.Guarantees.TS
