import Definitions.Accounting.Attribution
import Definitions.Accounting.PouwWork
import Models.Accounting.Assumptions.Attribution

/-!
# PoUW's work floor on one node, an open conjecture (trusted)

A3 of CMP-12's converged model (v3.1), over `Accounting.PouwWork`'s vocabulary on the actual circuit: a `Prop` about one
strategy `S` and the verifier `P`, taken as a hypothesis by `Accounting.Guarantees.PouwLocalOff`, never an `axiom`. It
is an open conjecture, so it is not under `shape.premises`: it becomes a premise under `shape.conjectures` once it has a
falsification test that runs. The other three premises are in `Models.Accounting.Assumptions.PouwWork`.

`PouwFloorAdvice` is compute attribution's floor for a block that isn't isolated: the help that doesn't depend on the
salt is free (TT, `Pouw.Assumptions.TT`), and the rest costs the chance of finding the salt, taken as a hypothesis by
`Accounting.Guarantees.PouwBlockCredit`. It is an open conjecture too, scoped to the noise whose floor is the committed
words' coin × coin products (H1 included while it commits a word at least every 32 keys, the peel's clean side not),
and `verity/core/protocols/pouw/tests/test_pouw_floor_advice.py` is its falsification test.
-/

namespace Accounting.Assumptions

open Accounting.PouwWork

variable {Ω : Type} [Fintype Ω] {L : Type*} (S : Ω → Actual.Run L) (P : Verdict Ω L)

/-- **A3 the work floor, off a skip event** (v3.1's `FloorOff`; on paper at `L = 32` given a merge price, conjectured at
served η): for the marking `m`, some event `Z` of probability at most `δ` is such that at every coin outside `Z`, `Nmin`
blocks right at every coin force the node's credits outside `m`, plus `ρ` a bit of help, to at least `Wb · Nmin`.

`Z` covers a guarded circuit that is right at every coin yet skips credited gates at some. The floor is meant at a
removable marking: one whose erasure leaves the verdict and the right blocks unchanged at every coin (the converged
model's `Removable`), so that covert work earns no credit. The actual-circuit model has no erasure of gates, so the
marking is taken as given, and the premise is claimed only at markings that are removable. A factor-2 projection form
(`#unmixed + 2·#mixed ≥ floor`) is being proved as a bound that implies this one. -/
def PouwFloorOff (X : Finset L) (π : ∀ r, Fin (S r).A.N → ℕ) (m : ∀ r, Finset (Fin (S r).A.N))
    (Wb Nmin ρ : ℕ) (δ : ℝ) : Prop :=
  ∃ Z : Ω → Prop, Pouw.pr Z ≤ δ ∧
    ∀ r, ¬ Z r → Nmin ≤ goodAll S P → Wb * Nmin ≤ credits S P X π m r + ρ * help S P X r

/-- **The floor at a block whose outside holds advice about the salt** (conjectured; compute attribution's floor,
`note:20261018T0730Z-draft-attrib`): on coins split into the window's coin, an `s`-bit salt with the hiding randomness
of its public commitment `com`, and the later draw, if the commitment hides the salt up to `δhide` (`SaltHiding`), at
every draw and commitment value what the outside of the block `X` and the certifiers `cert` does during the window
(`Verdict.window`, `[ts, td)`) takes at most `A` values across the window coins of the event `B` with that commitment
(`Attribution.OutsideAdvice`), and the outside evaluates the noise hash at most `q` times in the window
(`OutsideQueries`, at the marking `hash`), then for the marking `m` some event `Z` of probability at most
`δ + q · A / 2^s + δhide` is such that at every coin of `B` outside `Z`, `Nmin` blocks right at every coin force the
block's own credits outside `m` to at least `Wb · Nmin`. Meant at `B` the boundary audit's acceptance, the event on
which the advice is counted; off `B` it says nothing.

It rests on TT (`Pouw.Assumptions.TT`: the floor counts post-salt work, so help that is the same at every salt is
precomputation delivered late, free at any width) and the random-oracle model of the noise hash, not on A3: A3's
`ρ · help` term survives help of nonzero width, so no union bound over A3 drops it. `δ` is the floor's failure for an
outside whose help is the same at every salt. The outside's view is a function of the commitment, one of `A` advice
values and its oracle answers, and until one of its `q` calls hits the salt those answers don't depend on it. Its `q · A`
candidate salts at each commitment value hit the salt with probability at most `q · A / 2^s` against a commitment to an
independent salt, and `δhide` more against the real one: `q · 2^(k − s) + δhide` at `A = 2^k`. At decision 112's
`q = 2^128` that is at most `2^−64` past `δhide` when `k ≤ s − 192`: `k ≤ 320` at NET-18's 512-bit certifier salt
(`δhide = 0`, sealed), `k ≤ 64` at PoUW's 256-bit epoch coin (`δhide = 2^−192`, the coins root). The salt is the
coin the noise seeds read in full: a longer stream of a 256-bit coin is still `s = 256`. The operands are bound before
the salt (decision 114: each operand root is hashed before the salt's release at `ts` and transmitted at `td`, and
`E₁`'s rows read `A`'s), so the prover can't steer operands against the salt.

Scoped to the noise whose floor is the committed words' coin × coin products, both factors depending on the salt
(`note:20261018T1026Z-finding-pv-advice`): advice is a constant to the coins, so it lowers no such product. A product
that reads one side's coins only floors nothing (it is 0 at the zero operand). Noise of rank `r` along the contraction
on one side is out of scope where `r` is below the keys between committed words: then the floor is `k · r / C` a cell,
`C` the keys a word. With `F = Σ_{t < r} c_t(salt) · G_t` for salt-independent `G_t`, a helper ships the `r` products
`A · G_t`, and the block finishes each committed word of `A · F` in `r` multiply-adds instead of `C`. By scheme:

* `ncp-v1` and `ncp-v1-shift24`: in scope, every entry of `E₁` and `F₁` its own hash output.
* `eps1-sm120-h1fx1` (H1 + fx1, the compile pass's scheme on the s8 linears, decision 117): in scope through its
  coin × coin blocks, `b` on each entry of `X = A + b` against `c` and `U` of `W' = B + c + U`. `A · c`, `b · B` and
  `A · U` each read one side's coins only and floor nothing (0 at `A = B = 0`). `U`'s period, 32, is the keys between
  committed words, so `b · U` is rank 32 a word, the word's whole depth.
* H1's attention, outside the compile pass (`note:20261018T0634Z-draft-attn-noise`): cohalf `QKᵀ`
  (`Q̃ = Q + c + U`, `K̃ = K + b`) is in scope through its coin × coin blocks `U · b` and `c · b`. Adopted `P · V`
  (`P̃ = P − 64 + U`, `X = V + b`) is in scope through `U · b`: `P` carries no per-entry coin, only `U` of period 32,
  but a prefix word is committed every 32 keys, so `U · b` is 32 products a word from `b`'s residue sums, which depend
  on the salt: `k` a cell at every operand. `U · V` (from `V`'s residue sums) and `P · b` each read one side's coins
  only and floor nothing. The peel's clean side (the served `O = Σ P · V`) is bit-affine in the coins and floors
  nothing at all.
* The Pearl schemes (Komargodski–Weinstein noise of rank 32 or 64): outside it.

H1 is in scope only while it commits a word at least every 32 keys, `U`'s period: with one every `C` keys, `U · b`
floors only `32 · k / C`. `Wb` is meant at the committed words' coin × coin products (both factors depend on the salt;
H1's `U · b`, `b · c` and `b · U`), as `chainBound_gProd` counts them. The marking `hash` is meant to mark every
evaluation of the scheme's noise hash, as `PouwFloorOff`'s `m` is meant removable. Its falsification test,
`verity/core/protocols/pouw/tests/test_pouw_floor_advice.py`, checks that each in-scope scheme's per-entry noise fills
its noise space, that every scheme in the PoUW registry and in the compile pass's (`compile.SCHEMES`) is either in
scope or excluded with its reason, that H1 commits a prefix word every 32 keys, that the low-rank counterexample is
refused, and that a period-32 `U` costs `32 · k / C` products a cell from residue sums, `k` only at `C = 32`. An open
conjecture, not under `shape.premises`. -/
def PouwFloorAdvice {Ωh Ωd C : Type} [Fintype Ωh] [Fintype Ωd] {L : Type*} [LinearOrder L] [Fintype L] (s : ℕ)
    (S : (Fin (2 ^ s) × Ωh) × Ωd → Actual.Run L) (P : Verdict ((Fin (2 ^ s) × Ωh) × Ωd) L)
    (B : (Fin (2 ^ s) × Ωh) × Ωd → Prop) (com : Fin (2 ^ s) × Ωh → C) (X cert : Finset L)
    (hash : ∀ r, Finset (Fin (S r).A.N)) (π : ∀ r, Fin (S r).A.N → ℕ) (m : ∀ r, Finset (Fin (S r).A.N))
    (Wb Nmin q A : ℕ) (δ δhide : ℝ) : Prop :=
  SaltHiding com δhide →
  (∀ d c, Attribution.OutsideAdvice (fun ω => S (ω, d)) (fun ω => B (ω, d) ∧ com ω = c) cert X P.window A) →
  OutsideQueries S hash cert X P.window q →
  ∃ Z : (Fin (2 ^ s) × Ωh) × Ωd → Prop, Pouw.pr Z ≤ δ + ((q * A : ℕ) : ℝ) / 2 ^ s + δhide ∧
    ∀ r, ¬ Z r → B r → Nmin ≤ goodAll S P → Wb * Nmin ≤ credits S P X π m r

end Accounting.Assumptions
