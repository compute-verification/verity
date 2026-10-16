import Proofs.Flock.Claims.RootFindCorrect
import Proofs.Flock.Claims.RootFindCost

/-!
# A deterministic root finder over a finite field of characteristic 2: Berlekamp's trace method

Over `F` with `2^n` elements, `roots n basis f` lists the distinct roots of `f` (a coefficient list, `toPoly`), and
`rootsC` is the same computation with its field-operation count (`roots` is `(rootsC …).val`).

1. `rootProduct`: `g = gcd(f, X^(2^n) − X)`, monic, by `n` squarings modulo `f`. `X^(2^n) − X` is the product of
   `X − a` over every `a : F`, so `g` is the product of `f`'s distinct linear factors: it has `f`'s roots and divides
   `X^(2^n) − X`.
2. `split`: such a `g` splits by `gcd(g, Tr(β·X) mod g)`, where `Tr(z) = Σ_{i<n} z^(2^i)` (`trace`) takes values in
   `{0, 1}` on `F`: the factor's roots are `g`'s roots `a` with `Tr(β·a) = 0`. `trySplit` computes it (`traceLoop`,
   `n` squarings modulo `g`) and keeps it when it is a proper factor, `firstSplit` tries `basis` in order, and `split`
   recurses on the factor and its cofactor until each piece is linear. When `basis` separates `F` (`Separates`: for
   `c ≠ 0` some `β ∈ basis` has `Tr(β·c) ≠ 0`), two distinct roots differ in some `Tr(β·−)`, so some `β` splits every
   piece of degree at least 2.

`roots_spec`: for `toPoly f ≠ 0` the list has no duplicates, and its members are exactly `f`'s roots.
`rootsC_ops_le`: at most `costBound n basis.length f.length` field operations, for every `f`.
-/
