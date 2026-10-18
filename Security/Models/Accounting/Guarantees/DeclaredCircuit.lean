import Models.Accounting.Guarantees.Stacking

/-!
# A declared circuit's space and bandwidth from an accepted audit (trusted)

Daniel's goal in one statement, `DeclaredCircuitRequirement`: for any declared circuit `W` passing the stacking property
(`Gates.Property`, its operands within `B` bits: `Gates.Restricted`), any prover, and an audit accepted on a store whose
memory map and plans are `W`'s (`Requirement.StoreMatches`, `Requirement.PlanMatches`), except with probability `δ_G`
the node read at least `(1 − ε)` of `W`'s credited live bits (`Requirement.creditedBits`) and of the bits `W` reads from
complete blocks (`Requirement.creditedReadBits`), with `ε` and `δ_G` explicit in the property's caps, the codec, the
gate model's `E` and the audit's sampling. It composes three parts: the stacked remote audit in its relay form
(`StackedSessionConservationRelay`, the general class), the gate model's caps on a carrier of `W`'s (6b's
`Gates.Property` and `Accounting.Assumptions.EncodeChainHardness`), and `W`'s requirement (A1's
`Definitions/Accounting/Requirement.lean`). Its proof is in `Security/Proofs/Accounting/DeclaredCircuit.lean`. -/

namespace Accounting.Guarantees

open Accounting
open Pous (pr)
open scoped ENNReal

/-- **A declared circuit's live and read bits from an accepted stacked audit.** For a declared circuit `Wc` that passes
the stacking property `P(c, R_in, L_P)` at the caps `caps` on the audit's windows and certified reads `Aud`
(`Gates.Property`), with every operand within `B` bits (`Gates.Restricted`), and any prover: `T` appends of `M` block
positions made by the append-coin rule (`Accounting.appendCoinBlocks`: each append's choices `choose ω a κ_<a` are fixed
before its own coin keys its blocks), a session `A` of `n` windows of `P` passes over a store of `N` blocks of `b`
bits, one for the session, and pass devices each reading a carrier of `Wc`'s at its odd side addresses and taking in at
most `L` bits over its other side reads and the certified stream. A value's life, the at most `L_P = caps.life` engine
steps after the step that produced it in which `Wc` may read it, touches at most `life` of the audit's windows. A
carrier of `Wc`'s is a pass provenance (`Rate.PassProvenance`) whose inputs are at most `Wc`'s run inputs over a set `I`
of at most `life` of the audit's windows, and whose encodes, charged at the gate model's `encodeBitsCost`, cost at most
`Wc`'s budgeted units there (`Rate.PassProvenance.OfCircuit`).

The store's layout is the prover's choice after the coins, within `a = 8·R_in·life` bits: the session reads the blocks
at `slot (steer y)`, where `y` is the codec's and the appends' coins and `steer` is any function of them into `a` bits.
On the requirement side the prover chooses everything after every coin: the execution `e x`, the layout of values in
blocks `ℓ x`, the memory map `mm x` and the plans `plan x` are any functions of all the session's coins `x`. At each of
the at most `n·P` steps of `win` that the session's passes are bound to (#2378), the memory map is `W`'s with at most
`N` blocks resident and the plan is `W`'s reads from complete blocks, of at most `N` blocks. Then for every `m`, with

  `C = 8·R_in·life + 8·bytes·(c·R̂·life)/E`, `c` the caps' total, `R̂` the most bytes the audit certified a window,
  `ε = d̄ + γ_pass(L + C, λ + a + m·b) = d̄ + (S + (L + C − ρ) + b·F/(rounds·c_root) + λ + a + m·b)/(N·b)`,

the probability that every opening of every window passes while the session's tier reads fall short of `(1 − ε)` of
`W`'s credited live bits over `win`, or of its credited read bits there, is at most

  `δ_G = (1 − d̄)^(d·n) + (1 − d_w/n)^(d·n) + n·P·G·(C(N, m)·2^(L + C + a)/2^(h·m) + 2^q/2^λ) + δ`.

**The steering, charged as side capacity.** Inputs chosen after an append's coin can steer frees, plans and re-keying
against freeing at the cap, and so which blocks the session's store holds. The steering is charged as `a` bits of side
capacity, as the relay form charges a carrier's `C`: `a` enters `ε` once, as slack beside `λ` (`a/(N·b)`), and the relay
term's exponent beside `L + C`, while the draws' terms and `KeyBinding`'s `δ` are untouched. The bound on `a` is the
run-input bits a value's life admits, which the property already caps (`R_in` bytes a window, a value read within `L_P`
steps, so within `life` windows): what the session's layout depends on beyond the writers' choices fixed before each
coin is the inputs chosen after the coins of the appends it holds, all within the life of a value it holds. That the
layout is a function of those inputs (frees at last read, appends at production, re-keying no choice) is the memory-map
unit's, and that the session's store is the memory map's resident blocks is the binding's (`remote_bandwidth`'s Replay
and #2378); neither is in this game. Counting `a` beside `C`, which also counts the carriers' run inputs, counts those
bits twice, which is conservative.

**Why the relay form.** The deployed sketch form (`StackedSessionConservationSketchDeployed`) can't take the steering
additively: its per-pass tag term `qtot/2^96` has no slack to trade, so the steering would multiply it by `2^a`, and its
verbatim class has no side link to add `a` to. The relay form's receipt hashes the whole word, not the deployed
sketch, and its store is one for the session: blocks written during a session are planned only from the next, the
stricter of the two audit rules.

**The reading.** When the memory map and the plans are `W`'s, `W`'s credited live and read bits over the bound steps
are at most the store's `n·P·N·b` bits (`RequirementHeld`, `RequirementMoved`), so an accepted session read at least
`(1 − ε)` of them. `RequirementUncredited` and `RequirementUnplanned` bound what a block still being written leaves
out. The two circuits `Wc` (the gate model's, which the property reads) and `W` (the requirement's wiring) are one
declared circuit: requirement.py builds `W` from the same IR that census/v1 reads, which is the code's to hold, not this
statement's. A node that re-registers `R` times pays `δ_G` once a session, `R·δ_G` over them.

Proof: `Accounting.SecurityProofs.DeclaredCircuitRequirement`: the property caps each carrier's inputs at
`8·R_in·life` bits and its encodes at `c·R̂·life` budgeted units, so `StackedSessionConservationRelay`'s argument
applies at side link `L + C`; `KeyBinding` at the whole layout, with a union over the `2^a` layouts inside the uniform
store, moves the steered store's unpaid event to uniform blocks at slack `λ + a`; and `W`'s credited bits are at most
the store's by `RequirementHeld` and `RequirementMoved`.

Frozen (no-function, hash): `StackedSessionConservationRelay`'s game on a store whose layout takes one of `2^a` values
chosen after the coins, with carriers of a declared circuit within the stacking property's caps and the shortfall read
against its requirement, in the random-oracle and ideal encoding models, under `EncodeChainHardness`, `KeyBinding`,
`WorkHardness` and `SquaringCeiling`. -/
def DeclaredCircuitRequirement : Prop :=
  ∀ (n P N b h ρ S Lall L q r s d G lam m τ rounds cRoot F E T M life : ℕ) (cost : ℕ → ℕ) (Fc : Gates.Field)
    (k B εnum εden : ℕ) (tier : ℕ → Fin τ) (δ : ℝ≥0∞) {ιW : Type} (Wc : Gates.Dag ιW) (caps : Gates.Caps)
    (Aud : Gates.Audit) {Ω Ca Ch : Type} [Fintype Ω] [Fintype Ca]
    (enc : Ω → Fin T → Ch → (Fin T → Ca) → Fin M → Pous.Bits b)
    (choose : Ω → (a : Fin T) → ({c : Fin T // c < a} → Ca) → Ch)
    (slot : Pous.Bits (8 * caps.runBytes * life) → Fin N ↪ Fin T × Fin M)
    (steer : Ω × (Fin T → Ca) → Pous.Bits (8 * caps.runBytes * life))
    (A : Rate.DeployedSession n P N b h ρ S Lall q r s d G) (dbar dw : ℝ)
    {V : Type} [DecidableEq V] (W : Requirement.Wiring V)
    (e : (Ω × (Fin T → Ca)) × (Fin n → Rate.WindowCoins P N b h ρ r s d) → Requirement.Execution V)
    (ℓ : (Ω × (Fin T → Ca)) × (Fin n → Rate.WindowCoins P N b h ρ r s d) → Requirement.Layout V)
    (mm : (Ω × (Fin T → Ca)) × (Fin n → Rate.WindowCoins P N b h ρ r s d) → Requirement.Store)
    (plan : (Ω × (Fin T → Ca)) × (Fin n → Rate.WindowCoins P N b h ρ r s d) → ℕ → Finset ℕ) (win : Window),
    0 < n → 0 < P → 0 < N → 0 < b → 0 < rounds * cRoot → dw ≤ 1 → 0 < E → 0 < Fc.bytes →
    Assumptions.KeyBinding T M δ enc →
    Assumptions.EncodeChainHardness Fc k B E εnum εden →
    Assumptions.WorkHardness b rounds cRoot cost →
    (∀ w η j ι c, Assumptions.SquaringCeiling cost F ((A.window w η).device j ι c)) →
    Gates.Property caps Aud Wc → Gates.Restricted B Wc →
    (∀ t, ((Finset.range (caps.life + 1)).image fun i => Aud.win (t + i)).card ≤ life) →
    (∀ w η j ι c, ∃ (K : Rate.PassProvenance N b h ρ (ℕ → Bool)) (I : Finset ℕ),
      (∀ Es R O v i, ((A.window w η).device j ι c).side Es R O v (2 * i + 1) = K.run Es R O i) ∧
      (∀ st wd, (((A.window w η).device j ι c).chip st).linkBitsEven wd [] ≤ L) ∧
      I ⊆ Finset.range Aud.windows ∧ I.card ≤ life ∧
      K.OfCircuit (encodeBitsCost Fc k B εnum εden) Wc Aud I) →
    win.len ≤ n * P →
    (∀ x, ∀ t ∈ win.ticks, Requirement.StoreMatches W (e x) (ℓ x) (mm x) b t ∧ ((mm x).resident t).card ≤ N ∧
      Requirement.PlanMatches W (e x) (ℓ x) (plan x t) b t ∧ (plan x t).card ≤ N) →
    let a := 8 * caps.runBytes * life
    let C := 8 * caps.runBytes * life +
      8 * Fc.bytes * (caps.total * (Finset.range Aud.windows).sup Aud.reads * life) / E
    let ε := dbar + Rate.Deployed.gammaPass N b ρ S (L + C) (lam + a + m * b) rounds cRoot F
    pr (fun x : (Ω × (Fin T → Ca)) × (Fin n → Rate.WindowCoins P N b h ρ r s d) =>
        (∀ w t, (x.2 w).2 t ∈ (A.win w x.2).correctAll (appendCoinBlocks enc choose (slot (steer x.1)) x.1) (x.2 w).1) ∧
          ((((∑ w, (A.win w x.2).reads tier (appendCoinBlocks enc choose (slot (steer x.1)) x.1) (x.2 w).1 : ℕ)) : ℝ) <
              (1 - ε) * (∑ t ∈ win.ticks, Requirement.creditedBits W (e x) (ℓ x) t : ℕ) ∨
            (((∑ w, (A.win w x.2).reads tier (appendCoinBlocks enc choose (slot (steer x.1)) x.1) (x.2 w).1 : ℕ)) : ℝ) <
              (1 - ε) * (∑ t ∈ win.ticks, Requirement.creditedReadBits W (e x) (ℓ x) t : ℕ)))
      ≤ ENNReal.ofReal (1 - dbar) ^ (d * n) + ENNReal.ofReal (1 - dw / n) ^ (d * n) +
        (n * P * G : ℝ≥0∞) * ((N.choose m : ℝ≥0∞) * 2 ^ (L + C + a) / 2 ^ (h * m) + (2 ^ q : ℝ≥0∞) / 2 ^ lam) + δ

end Accounting.Guarantees
