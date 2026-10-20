import Proofs.Flock.Claims.GS.Decode
import Proofs.Flock.Claims.GS.Flock
import Proofs.Flock.Claims.GS.Radius
import Proofs.Flock.Claims.GS.ZkRadius
import Proofs.Flock.Claims.GS.RadiusFamily
import Proofs.Flock.Claims.GS.RadiusDeploy

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
* `GS.Radius`: the decoder that meets the spec's list decoding (`Flock.fill`, `Flock.ListWithin` in
  `Definitions/Claims.lean`: every trace `M` whose encoding agrees with the zero-filled word of its codeword length `n`
  on at least `(1 − θ M)·n` positions, the radius read per trace) at Flock's level-0 code at `θ = 1 − √ρ − 1/200`
  (`flockDecodeR`, `listWithin_flockDecodeR`), with its `DecoderComplexity` at `(os.length + n)·2^k0·128`
  (`flockDecodeR_complexity`).
* `GS.ZkRadius`: the masked (`--zk`) level-0 code in the radius form at each `m` the masked verifier serves: a masked
  message read from its lanes' coefficients (`zkReader`), and the decoder (`zkDecodeR`) that meets `ListWithin` at
  `θ = 1 − √ρ_zk − 1/200`, `ρ_zk = (2^cols + t)/2^d` (`listWithin_zkDecodeR`), with its `DecoderComplexity` at
  `(os.length + n)·2^k0·128`, on the `2^k0` witness lanes (`zkDecodeR_complexity`).
* `GS.RadiusFamily`: a family of radius decoders read as one (`famDecodeR`), meeting `ListWithin` at the family's code
  with each trace's length and radius its index's (`listWithin_famDecodeR`), its count the sum over the indices
  (`famDecodeR_complexity`); at the masked verifier's deployed code, `zkDecodeRD` (`listWithin_zkDecodeRD`,
  `zkDecodeRD_complexity`).
* `GS.RadiusDeploy`: the plain level-0 code at each `m` in the radius form (`plainDecodeR`), and the plain family
  (`plainDecodeRD`, `listWithin_plainDecodeRD`, complexity `14·(flockPolyR + X)` at `plainSizeRD`). Each family's size
  is affine in the opened columns (`plainSizeRD_le`, `zkSizeRD_le`), and each counts a step on none
  (`plainCountedRD_nil_pos`, `zkCountedRD_nil_pos`). The deployed records name the counted families as their decoder
  (`FlockDeploy.deployPlain`'s and `deployZk`'s `dec`), and `FlockDeploy.efficient_plain` and `efficient_zk` are
  `Claims.Flock.Efficient` at the records the claims read, those with their columns trimmed (`deployPlainTrim`,
  `deployZkTrim`).
-/
