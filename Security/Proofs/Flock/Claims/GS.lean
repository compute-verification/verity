import Proofs.Flock.Claims.GS.Decode
import Proofs.Flock.Claims.GS.Flock

/-!
# A counted Guruswami–Sudan decoder, and `DecoderComplexity` at Flock's level-0 code

The decoder computes on coefficient lists and tables, and counts its field operations along the same computation:

* `GS.Basic`: coefficient lists and boxed coefficient tables, read as Mathlib's `F[X]` and `F[X][X]`.
* `GS.Elim`: a kernel vector of an underdetermined homogeneous system by Gaussian elimination, counted.
* `GS.Interp`: the Guruswami–Sudan interpolation polynomial through `N` points with multiplicity `k + 1`, which
  vanishes at every polynomial of degree below `k` through at least `t` of them when `(k+1)·N < t²`, from ArkLib's
  multiplicity lemmas (`interpPoly_eval_eq_zero`), in at most `20·N^9` operations.
* `GS.RR`: Roth–Ruckenstein's search for the `Y`-roots of degree below `k` over a root finder (`RootFinder`), complete,
  with at most `L` nodes per level.
* `GS.Johnson`: Johnson's bound for a set family.
* `GS.Decode`: the decoder for any interleaved Reed–Solomon code (`gsDecode`), complete on distinct points from the
  Johnson count on (`mem_gsDecode`), and its `DecoderComplexity` at the opened data's size (`gsDecode_complexity`).
* `GS.Flock`: that decoder at Flock's level-0 code, with Berlekamp's trace root finder over GF(2^128)
  (`flockDecode_complexity`, `mem_flockDecode`).
-/
