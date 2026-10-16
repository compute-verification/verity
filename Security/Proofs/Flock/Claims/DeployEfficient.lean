import Proofs.Flock.Claims.GS.RadiusDeploy
import Proofs.Flock.Claims.ERepCost

/-!
# Flock's efficiency claim at both deployed records

`Flock.Efficient H D decode counted size p` is four facts: `E^rep`'s reruns (`ERepCost` at `D.V`, `D.waits`, `D.k`,
`D.cap` and `D.cols`), the decoder meeting the list-decoding spec at `D.len`, `D.enc` and `D.θ` (`ListWithin`), its
`DecoderComplexity`, and its lists at most `p(size)` long, which bounds `assemble`'s checks. The first holds at every
deployment (`Claims.ERep_cost`, `Flock.Efficient_of_decoder`); the other three are the radius decoders'
(`Claims.GS.radius_deployPlain`, `Claims.GS.radius_deployZk`, `Claims.GS.plainDecodeRD_length_le`,
`Claims.GS.zkDecodeRD_length_le`). So, at every `H`:

* `efficient_plain`: the plain record the claims read, `deployPlainTrim rel`, with `plainDecodeRD`, its count at most
  `14·(flockPolyR + X)` at `plainSizeRD`;
* `efficient_zk`: the masked record `deployZkTrim rel` with `zkDecodeRD`, its count at most `11·(zkPolyR + X)` at
  `zkSizeRD`. The masked columns and code are the `2^k0` witness lanes of each opened row (`zkRepColsOf`,
  `Flock.zkWidth i = 2^k0`), so `zkDecodeRD` decodes those lanes only.

At the deployed numbers (`erepRuns_plain`, `erepRuns_zk`), `E^rep` makes `Numbers.waits = 210` waits of at most
`Numbers.cap = 2^256` draws, so it reruns the prover at most `1 + 210·2^256` times, and `1 + 210·k` times in
expectation, `k = ⌈4·N₀/e⌉` with `N₀` the tables' level-0 codeword lengths summed (`2^d` at one table).
-/

namespace FlockDeploy

open Flock

local notation "𝔽" => FlockLevel3.GF128

/-- **Flock: efficiency at the deployed plain verifier**, at every `H`: `E^rep`'s reruns, and `plainDecodeRD` meets
`ListWithin` at the record's `len`, `enc` and `θ` with `DecoderComplexity` `14·(flockPolyR + X)` at `plainSizeRD`. -/
theorem efficient_plain (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) :
    Claims.Flock.Efficient H (deployPlainTrim rel) Claims.GS.plainDecodeRD Claims.GS.plainCountedRD
      Claims.GS.plainSizeRD (Polynomial.C 14 * (Claims.GS.flockPolyR + Polynomial.X)) :=
  Claims.Flock.Efficient_of_decoder H (deployPlainTrim rel) _ _ _ _ (Claims.GS.radius_deployPlain rel).1
    (Claims.GS.radius_deployPlain rel).2 Claims.GS.plainDecodeRD_length_le

/-- **Flock: efficiency at the deployed masked (`--zk`) verifier**, at every `H`: `E^rep`'s reruns, and `zkDecodeRD`
meets `ListWithin` at the record's `len`, `enc` and `θ` with `DecoderComplexity` `11·(zkPolyR + X)` at `zkSizeRD`. -/
theorem efficient_zk (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) :
    Claims.Flock.Efficient H (deployZkTrim rel) Claims.GS.zkDecodeRD Claims.GS.zkCountedRD Claims.GS.zkSizeRD
      (Polynomial.C 11 * (Claims.GS.zkPolyR + Polynomial.X)) :=
  Claims.Flock.Efficient_of_decoder H (deployZkTrim rel) _ _ _ _ (Claims.GS.radius_deployZk rel).1
    (Claims.GS.radius_deployZk rel).2 Claims.GS.zkDecodeRD_length_le

/-- **`E^rep`'s reruns at the deployed plain verifier**, at every decoder: at most `1 + 210·2^256`, and
`1 + 210·⌈4·N₀/e⌉` in expectation. -/
theorem erepRuns_plain (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)
    (decode : List (ℕ × List 𝔽) → List Claims.Flock.PlainTrace) :
    Claims.ExtractRunsLe H (Claims.Flock.E (deployPlainTrim rel) decode) (fun _ => 1 + 210 * 2 ^ 256) ∧
      Claims.ExtractRunsExpLe H (Claims.Flock.E (deployPlainTrim rel) decode)
        (fun x => 1 + 210 * (⌈4 * N₀ false x / Real.exp 1⌉₊ : ℝ)) :=
  Claims.ERep_cost H _ _ _ _ _ _ _

/-- **`E^rep`'s reruns at the deployed masked verifier**, at every decoder: at most `1 + 210·2^256`, and
`1 + 210·⌈4·N₀/e⌉` in expectation. -/
theorem erepRuns_zk (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)
    (decode : List (ℕ × List 𝔽) → List Claims.Flock.ZkTrace) :
    Claims.ExtractRunsLe H (Claims.Flock.E (deployZkTrim rel) decode) (fun _ => 1 + 210 * 2 ^ 256) ∧
      Claims.ExtractRunsExpLe H (Claims.Flock.E (deployZkTrim rel) decode)
        (fun x => 1 + 210 * (⌈4 * N₀ true x / Real.exp 1⌉₊ : ℝ)) :=
  Claims.ERep_cost H _ _ _ _ _ _ _

end FlockDeploy
