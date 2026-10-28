import Mathlib.Combinatorics.SimpleGraph.Clique
import Mathlib.Combinatorics.SimpleGraph.Finite
import Mathlib.Data.ZMod.Basic
import Definitions.Pous.Accounting.LemmaA
import Definitions.Pous.Accounting.Params
import Definitions.Pous.Accounting.PubMeets
import Definitions.Pous.Accounting.GradedMeets
import Definitions.Pous.Accounting.Rebuild
import Definitions.Pous.Assumptions
import Definitions.Pous.Model.HashChain
import Definitions.Pous.Model.Dense
import Definitions.Pous.Model.Inc
import Definitions.Pous.Model.Invertible
import Definitions.Pous.Model.Labelling
import Definitions.Pous.Model.M1
import Definitions.Pous.Model.P3Chain
import Definitions.Pous.Model.BandChain
import Definitions.Pous.Model.Pebbling
import Definitions.Pous.Model.SampledAudit
import Definitions.Pous.Guarantees.P2
import Definitions.Pous.Guarantees.Obligation
import Definitions.Pous.Guarantees.SecureErasure
import Definitions.Pous.Guarantees.StreamedWipe
import Definitions.Pous.Model.BandGraph
import Definitions.Pous.Model.SegLift
import Definitions.Pous.Model.Continuous

/-!
# Guarantees

PoUS's frozen statements: claims about a model, an ideal primitive or the model's memory challenger, which no Lean
function computes, so none has the shape of a property (`Security/Properties/`) yet. Each docstring ends with its
category. The list only shrinks (`.agents/skills/lean-proofs/SKILL.md`). The lemmas, the
continuous audit's completeness, the refuted N10 and Track A's statements in model M1 are math rather than claims; they
are stated under the same names in `Security/Proofs/Pous/Statements.lean`.

`grade.sh` grades against these and six of those. A submission proves exactly one of them, as
`theorem solution : Pous.Guarantees.<Name>`. Named assumptions (`Security/Definitions/Pous/Assumptions.lean`) appear as
hypotheses of the pinned `Prop`, never as axioms. "Pinned as drafted" means the binders and conclusion are the draft's,
verbatim.
-/

namespace Pous.Guarantees

open scoped ENNReal

/-! ## The dense scheme -/

/-- **Pinned as drafted** (`Security/Proofs/Pous/Dense/DenseMeets.lean`, `PousDense.dense_meets`; the scheme is
in `Pous.Dense`).

 **The first end-to-end `Meets` with no named assumption** (`docs/p3-scheme.md` §4). The dense scheme on
`B = 2^16` blocks of 1 KB meets the requirement with sequential reveal, `D = 64` rounds and `Q = 2^20` queries per
answer, `k = 107` challenges and `ε_crypto = 2^-128`: it is correct, at rate 1, and every `⌊ρ|C|⌋`-bit adversary is
accepted with probability at most `1% + 2^-128`.

The first `Meets` with no named assumption. Scope: security, correctness and rate only, not decode cost, which is
`Θ(B)` per byte (decoding hashes every earlier block).

Frozen (no-function, hash): a `Meets` in an ideal model, which no Lean function computes. -/
def DenseMeets : Prop :=
  Pous.Meets (Pous.Dense.denseScheme (2 ^ 16) 8192) .sequential 64 (2 ^ 20) 107 ((2⁻¹ : ENNReal) ^ 128)

/-! ### P3 with a concrete `H`: the overwrite chain on `Π₂` (`Definitions.Pous.Model.P3Chain`)

Red team: GO. The proofs are in this package
(`Pous.SecurityProofs.P3Concrete*`, from `Security/Proofs/Pous/Chain/Flat.lean`).

The model is `Pous.Sponge.p3ChainModel 220 2384`: a tweakable ideal permutation `P` of 64 KB labels (an independent
uniform permutation per tweak `(tag, layer, node)`) and an untweaked ideal permutation `Π₂` of `2·524480 + 512` bits,
queried both ways. `H` is the overwrite chain on `Π₂` (`Pous.Sponge.ovKey`), so `D` and `Q` count `P^{±1}` and
`Π₂^{±1}` calls. The labels are P3's two layers on the band graph `B(220, k)` (`Pous.P3Column.bandGraph`): 220 labels
of `220·2384` bits (64 KB) per segment, 14.4 MB. The salt is 24 bytes, and segment `s` is keyed by `salt ‖ u64(s)`.

Scope: the ideal two-permutation model only. Lean does not prove anything about a concrete permutation (no Keccak or
other instantiation of `P` or `Π₂`), decode cost, the wall-clock-to-`Q` calibration, authentication, accounting or
fresh setup. As for every `Meets` here, `W` is fixed before the primitives are drawn. -/

/-- **Unconditional (no hypothesis): P3 with a concrete `H`, 64 KB labels, one segment (14.4 MB).**
`p3SchemeC 2384 B(220, 12)` meets the requirement with sequential reveal, `D = 10` rounds, `Q = 2^20` queries per
answer, `k = 106` challenges and `ε = 2^-128`: it is correct, `|C| ≥ |W|`, within the space bound, and every
`⌊ρ|C|⌋`-bit adversary is accepted with probability at most `δ + 2^-128`. Pinned as drafted
(`PousP3ConcreteMulti.p3_meets_concrete_64_D10_unconditional`; here `Pous.SecurityProofs.P3ConcreteMeets64`). It
rests on `Pous.P3Concrete.p3SegCharges` through the ex post facto target `P3ChainEPF`, and on the proved chain
certificate `p3ChainCert` (`X = 1` at `D = 10`). Red team: GO
(`docs/lean-trusted-layer-review.md` §§50–51). Scope: as the section header.

Frozen (no-function, hash): a `Meets` in an ideal model, which no Lean function computes. -/
def P3ConcreteMeets64 : Prop :=
  Meets (Pous.P3Concrete.p3SchemeC 2384 (Pous.P3Column.bandGraph 220 12)) .sequential 10 (2 ^ 20) 106
    ((2⁻¹ : ℝ≥0∞) ^ 128)

/-- **Unconditional (no hypothesis): P3 with a concrete `H`, 64 KB labels, 14.005 GB (`N = 971` segments).**
`p3SchemeNC 2384 B(220, 12) 971` meets the requirement with sequential reveal, `D = 10`, `Q = 2^20`, `k = 106` and
`ε = 2^-128`. Security is joint over all 971 segments: one advice string for the whole store, and segment-by-segment
bounds do not combine to this (`not_jointClassWeight`). Pinned as drafted
(`PousP3ConcreteMulti.p3_meets_concrete_multi_64_D10_unconditional`; here `Pous.SecurityProofs.P3ConcreteMeets14GB`).
It rests on `p3SegCharges` through
`P3ChainEPFMulti` and the joint bridge (`PousP3ConcreteMulti.p3ChainEPFMulti_of_segCharges`), and on `p3ChainCert`.
Red team: as `P3ConcreteMeets64` (GO, §50). Scope: as the section header.

Frozen (no-function, hash): a `Meets` in an ideal model, which no Lean function computes. -/
def P3ConcreteMeets14GB : Prop :=
  Meets (Pous.P3Concrete.p3SchemeNC 2384 (Pous.P3Column.bandGraph 220 12) 971) .sequential 10 (2 ^ 20) 106
    ((2⁻¹ : ℝ≥0∞) ^ 128)

/-- **Unconditional (no hypothesis): the fallback, P3 with a concrete `H` at `D = 12` on `B(220, 16)`, one
segment.** `p3SchemeC 2384 B(220, 16)` meets the requirement with sequential reveal, `D = 12`, `Q = 2^20`, `k = 106`
and `ε = 2^-128`. Pinned as drafted
(`PousP3ConcreteMulti.p3_meets_concrete_64_D12_band16_unconditional`; here `Pous.SecurityProofs.P3ConcreteMeets64D12`),
from `P3ChainEPF` and the certificate
`p3ChainCert_band16`. Red team: as `P3ConcreteMeets64` (GO, §50). Scope: as the section header.

Frozen (no-function, hash): a `Meets` in an ideal model, which no Lean function computes. -/
def P3ConcreteMeets64D12 : Prop :=
  Meets (Pous.P3Concrete.p3SchemeC 2384 (Pous.P3Column.bandGraph 220 16)) .sequential 12 (2 ^ 20) 106
    ((2⁻¹ : ℝ≥0∞) ^ 128)

/-! ### The band scheme and the dense chain scheme over the overwrite chain (`Definitions.Pous.Model.BandChain`)

Red team: GO (`docs/lean-trusted-layer-review.md` §§52–57). The proofs
are in this package (`Pous.SecurityProofs.Band*`, `Pous.SecurityProofs.ChainDense*`, from
`Security/Proofs/Pous/Chain/Flat.lean`).

The model is `Pous.Sponge.idealPerm (2·524288 + 512)`: one uniform permutation `Π₂` of `1049088` bits, queried both
ways, so `D` and `Q` count `Π₂^{±1}` calls. `H` is the overwrite chain on `Π₂` (`Pous.Sponge.ovKey`), labels are 64 KiB
(`ℓ = 524288`) with capacity 512, and there is no `pp` and there are no coins.

Scope: the ideal-permutation model only. Lean does not prove anything about a concrete permutation, decode cost, the
wall-clock-to-`Q` calibration, authentication, accounting or fresh setup. As for every `Meets` here, `W` is fixed
before `Π₂` is drawn, and the salt is a parameter every statement quantifies over. The band needs in-degree `d ≥ D`:
with `d + 1 ≤ D` it is broken (`BandGapAttack`, proved as `band_gap_attack`). -/

/-- **Unconditional: the band scheme at 64 KiB.** For every in-degree `d ≥ 5` and salt, `chainBand (2^9) d` (512
blocks of 64 KiB labels) meets the requirement with sequential reveal, `D = 5` rounds, `Q = 2^20` queries per answer,
`k = 111` challenges and `ε = 2^-128`. Proof: `Pous.BandChain.band_meets_64_final`; here
`Pous.SecurityProofs.BandChainMeets64`.

Frozen (no-function, hash): a `Meets` in an ideal model, which no Lean function computes. -/
def BandChainMeets64 : Prop :=
  ∀ d : ℕ, 5 ≤ d → ∀ salt : Bits 256,
    Meets (Pous.BandChain.chainBand (2 ^ 9) d 524288 512 (Pous.Sponge.dom salt)) .sequential 5 (2 ^ 20) 111
      ((2⁻¹ : ℝ≥0∞) ^ 128)

/-- **Unconditional: the band at the deployment in-degree `d = 5`** (`band_meets_64_d5_final`; here `Pous.SecurityProofs.BandChainMeets64D5`).

Frozen (no-function, hash): a `Meets` in an ideal model, which no Lean function computes. -/
def BandChainMeets64D5 : Prop :=
  ∀ salt : Bits 256,
    Meets (Pous.BandChain.chainBand (2 ^ 9) 5 524288 512 (Pous.Sponge.dom salt)) .sequential 5 (2 ^ 20) 111
      ((2⁻¹ : ℝ≥0∞) ^ 128)

/-- **Unconditional: the band at the recommended in-degree `d = 12`** (`band_meets_64_d12_final`; here `Pous.SecurityProofs.BandChainMeets64D12`).

Frozen (no-function, hash): a `Meets` in an ideal model, which no Lean function computes. -/
def BandChainMeets64D12 : Prop :=
  ∀ salt : Bits 256,
    Meets (Pous.BandChain.chainBand (2 ^ 9) 12 524288 512 (Pous.Sponge.dom salt)) .sequential 5 (2 ^ 20) 111
      ((2⁻¹ : ℝ≥0∞) ^ 128)

/-- **Unconditional: the band at 14 GiB.** For every `d ≥ 5`, the 448-segment band `chainBandSeg 448 (2^9) d`
(229,376 blocks of 64 KiB) meets the requirement at `D = 5`, `Q = 2^20`, `k = 111`, `ε = 2^-128`, both with the
implementation's per-segment tags (`segDom salt`, segment `s` keyed by `salt ‖ u64(s)`, any 192-bit salt) and with one
global salt and global indices (`dom salt`). One advice string covers the whole store. Proof:
`Pous.BandMulti.band_meets_14_final`; here `Pous.SecurityProofs.BandMultiMeets14`.

Frozen (no-function, hash): a `Meets` in an ideal model, which no Lean function computes. -/
def BandMultiMeets14 : Prop :=
  ∀ d : ℕ, 5 ≤ d →
    (∀ salt : Bits 192,
      Meets (Pous.BandMulti.chainBandSeg 448 (2 ^ 9) d 524288 512 (Pous.BandMulti.segDom salt (2 ^ 9))) .sequential
        5 (2 ^ 20) 111 ((2⁻¹ : ℝ≥0∞) ^ 128)) ∧
    (∀ salt : Bits 256,
      Meets (Pous.BandMulti.chainBandSeg 448 (2 ^ 9) d 524288 512 (Pous.Sponge.dom salt)) .sequential 5 (2 ^ 20) 111
        ((2⁻¹ : ℝ≥0∞) ^ 128))

/-- **Unconditional: the band at 14 GiB with 106 challenges**, per-segment tags (`band_meets_14_k106_final`; here `Pous.SecurityProofs.BandMultiMeets14K106`).

Frozen (no-function, hash): a `Meets` in an ideal model, which no Lean function computes. -/
def BandMultiMeets14K106 : Prop :=
  ∀ d : ℕ, 5 ≤ d → ∀ salt : Bits 192,
    Meets (Pous.BandMulti.chainBandSeg 448 (2 ^ 9) d 524288 512 (Pous.BandMulti.segDom salt (2 ^ 9))) .sequential 5
      (2 ^ 20) 106 ((2⁻¹ : ℝ≥0∞) ^ 128)

/-- **Unconditional: the band store at every segment count from 1 to 2^64.** For every `N` with `1 ≤ N ≤ 2^64` and
every `d ≥ 5`, the `N`-segment band `chainBandSeg N (2^9) d` (`N·512` blocks of 64 KiB, `N·32` MiB) meets the
requirement at `D = 5`, `Q = 2^20`, `k = 111`, `ε = 2^-128`, both with the implementation's per-segment tags
(`segDom salt`, segment `s` keyed by `salt ‖ u64(s)`, any 192-bit salt) and with one global salt and global indices
(`dom salt`). One advice string covers the whole store. `2^64` is where `u64(s)` stops being injective, and the domains
must be injective across segments: with a shared domain every `N ≥ 2` breaks (`shared_domain_breaks_fam`).
At `N = 448` this is `BandMultiMeets14`, and at `N = 1` its global layout is `BandChainMeets64`. Red team: GO
(`docs/lean-trusted-layer-review.md` §58). Proof: `Pous.BandMulti.band_meets_family_final`
(`Security/Proofs/Pous/Chain/Family.lean`); here `Pous.SecurityProofs.BandMultiMeetsFamily`.

Frozen (no-function, hash): a `Meets` in an ideal model, which no Lean function computes. -/
def BandMultiMeetsFamily : Prop :=
  ∀ (N : ℕ) [NeZero N], N ≤ 2 ^ 64 → ∀ d : ℕ, 5 ≤ d →
    (∀ salt : Bits 192,
      Meets (Pous.BandMulti.chainBandSeg N (2 ^ 9) d 524288 512 (Pous.BandMulti.segDom salt (2 ^ 9))) .sequential
        5 (2 ^ 20) 111 ((2⁻¹ : ℝ≥0∞) ^ 128)) ∧
    (∀ salt : Bits 256,
      Meets (Pous.BandMulti.chainBandSeg N (2 ^ 9) d 524288 512 (Pous.Sponge.dom salt)) .sequential 5 (2 ^ 20) 111
        ((2⁻¹ : ℝ≥0∞) ^ 128))

/-- **Unconditional: the overwrite-chain dense scheme at 64 KiB.** `chainDense (2^9)` (512 blocks of 64 KiB on the
complete DAG) meets the requirement at `D = 5`, `Q = 2^20`, `k = 111`, `ε = 2^-128`. Not `DenseMeets` (the dense
scheme with `H` a random oracle), and not `SpongeDense`'s `ChainDenseMeetsPinned` (`2^16` blocks of 1 KiB, `D = 64`).
Proof: `PousSpongeChain.chain_meets_64_final`; here `Pous.SecurityProofs.ChainDenseMeets64`.

Frozen (no-function, hash): a `Meets` in an ideal model, which no Lean function computes. -/
def ChainDenseMeets64 : Prop :=
  ∀ salt : Bits 256,
    Meets (Pous.Sponge.chainDense (2 ^ 9) 524288 512 (Pous.Sponge.dom salt)) .sequential 5 (2 ^ 20) 111
      ((2⁻¹ : ℝ≥0∞) ^ 128)

/-! ### The band at `D = 2`, `D = 1` and `D = 0` rounds

Each
statement is a band pin above with only the round budget `D`, the challenge count `k` and the
bound on the in-degree `d` changed; the model, the scope and `d ≥ D` (`BandGapAttack`) are as there (`D = 0` puts no
bound on `d`, since the responder makes no `Π₂` call). The proofs are
`Security/Proofs/Pous/Chain/FamilyRounds.lean`'s (`Pous.BandMulti.Rounds`): `Family.lean`'s certificate with `D`
and `k` as parameters. -/

/-- **Unconditional: the band store at `D = 2` rounds and `k = 97`, at every segment count from 1 to 2^64.**
`BandMultiMeetsFamily` at `D = 2` and `k = 97`, for every `d ≥ 2`: for every `N` with `1 ≤ N ≤ 2^64` and every
`d ≥ 2`, the `N`-segment band `chainBandSeg N (2^9) d` meets the requirement at `D = 2`, `Q = 2^20`, `k = 97`,
`ε = 2^-128`, both with the implementation's per-segment tags (`segDom salt`, any 192-bit salt) and with one global
salt and global indices (`dom salt`). Proof: `Pous.BandMulti.Rounds.band_meets_family_d2`; here
`Pous.SecurityProofs.BandMultiMeetsFamilyD2`.

Frozen (no-function, hash): a `Meets` in an ideal model, which no Lean function computes. -/
def BandMultiMeetsFamilyD2 : Prop :=
  ∀ (N : ℕ) [NeZero N], N ≤ 2 ^ 64 → ∀ d : ℕ, 2 ≤ d →
    (∀ salt : Bits 192,
      Meets (Pous.BandMulti.chainBandSeg N (2 ^ 9) d 524288 512 (Pous.BandMulti.segDom salt (2 ^ 9))) .sequential
        2 (2 ^ 20) 97 ((2⁻¹ : ℝ≥0∞) ^ 128)) ∧
    (∀ salt : Bits 256,
      Meets (Pous.BandMulti.chainBandSeg N (2 ^ 9) d 524288 512 (Pous.Sponge.dom salt)) .sequential 2 (2 ^ 20) 97
        ((2⁻¹ : ℝ≥0∞) ^ 128))

/-- **Unconditional: the band store at `D = 1` round and `k = 93`, at every segment count from 1 to 2^64.**
`BandMultiMeetsFamily` at `D = 1` and `k = 93`, for every `d ≥ 1`, both domain layouts. Proof:
`Pous.BandMulti.Rounds.band_meets_family_d1`; here `Pous.SecurityProofs.BandMultiMeetsFamilyD1`.

Frozen (no-function, hash): a `Meets` in an ideal model, which no Lean function computes. -/
def BandMultiMeetsFamilyD1 : Prop :=
  ∀ (N : ℕ) [NeZero N], N ≤ 2 ^ 64 → ∀ d : ℕ, 1 ≤ d →
    (∀ salt : Bits 192,
      Meets (Pous.BandMulti.chainBandSeg N (2 ^ 9) d 524288 512 (Pous.BandMulti.segDom salt (2 ^ 9))) .sequential
        1 (2 ^ 20) 93 ((2⁻¹ : ℝ≥0∞) ^ 128)) ∧
    (∀ salt : Bits 256,
      Meets (Pous.BandMulti.chainBandSeg N (2 ^ 9) d 524288 512 (Pous.Sponge.dom salt)) .sequential 1 (2 ^ 20) 93
        ((2⁻¹ : ℝ≥0∞) ^ 128))

/-- **Unconditional: the band store at `D = 0` rounds and `k = 89`, at every segment count from 1 to 2^64.**
`BandMultiMeetsFamily` at `D = 0` and `k = 89`, for every `d` with no bound on the in-degree: at `D = 0` the responder
makes no `Π₂` call and answers from its state alone, so the band needs no bound on `d`. Both domain layouts.
Proof: `Pous.BandMulti.Rounds.band_meets_family_d0`; here
`Pous.SecurityProofs.BandMultiMeetsFamilyD0`.

Frozen (no-function, hash): a `Meets` in an ideal model, which no Lean function computes. -/
def BandMultiMeetsFamilyD0 : Prop :=
  ∀ (N : ℕ) [NeZero N], N ≤ 2 ^ 64 → ∀ d : ℕ,
    (∀ salt : Bits 192,
      Meets (Pous.BandMulti.chainBandSeg N (2 ^ 9) d 524288 512 (Pous.BandMulti.segDom salt (2 ^ 9))) .sequential
        0 (2 ^ 20) 89 ((2⁻¹ : ℝ≥0∞) ^ 128)) ∧
    (∀ salt : Bits 256,
      Meets (Pous.BandMulti.chainBandSeg N (2 ^ 9) d 524288 512 (Pous.Sponge.dom salt)) .sequential 0 (2 ^ 20) 89
        ((2⁻¹ : ℝ≥0∞) ^ 128))

/-- **Unconditional: the band scheme at 64 KiB, `D = 2` rounds and `k = 97`.** `BandChainMeets64` at `D = 2` and
`k = 97`, for every `d ≥ 2` and salt (the suffix counts rounds, where `BandChainMeets64D5`'s names the in-degree).
Proof: `Pous.BandMulti.Rounds.band_meets_64_d2`, `BandMultiMeetsFamilyD2`'s global layout at `N = 1`; here
`Pous.SecurityProofs.BandChainMeets64Rounds2`.

Frozen (no-function, hash): a `Meets` in an ideal model, which no Lean function computes. -/
def BandChainMeets64Rounds2 : Prop :=
  ∀ d : ℕ, 2 ≤ d → ∀ salt : Bits 256,
    Meets (Pous.BandChain.chainBand (2 ^ 9) d 524288 512 (Pous.Sponge.dom salt)) .sequential 2 (2 ^ 20) 97
      ((2⁻¹ : ℝ≥0∞) ^ 128)

/-- **Unconditional: the band scheme at 64 KiB, `D = 1` round and `k = 93`.** `BandChainMeets64` at `D = 1` and
`k = 93`, for every `d ≥ 1` and salt. Proof: `Pous.BandMulti.Rounds.band_meets_64_d1`; here
`Pous.SecurityProofs.BandChainMeets64Rounds1`.

Frozen (no-function, hash): a `Meets` in an ideal model, which no Lean function computes. -/
def BandChainMeets64Rounds1 : Prop :=
  ∀ d : ℕ, 1 ≤ d → ∀ salt : Bits 256,
    Meets (Pous.BandChain.chainBand (2 ^ 9) d 524288 512 (Pous.Sponge.dom salt)) .sequential 1 (2 ^ 20) 93
      ((2⁻¹ : ℝ≥0∞) ^ 128)

/-! ### The band with a public encoder (`Definitions.Pous.Model.PubEncoder`, `Definitions.Pous.Accounting.PubMeets`)

Red team: GO (`docs/lean-trusted-layer-review.md` §§53, 55). The codeword is
produced by an untrusted server and accepted by the exact check. Each holds under the four deployment conditions of the
review's §48, which are in its docstring. The proofs are in this package (`Pous.SecurityProofs.BandPub*`). -/

/-- **Unconditional: the band at 64 KiB with a public encoder.** For every `d ≥ 5` and salt, `PubMeets` for
`chainBand (2^9) d` with the exact check, at `D = 5`, `Q = 2^20`, `k = 111`, `ε = 2^-128`: an untrusted server that
encodes the data is exactly as secure as verifier-side encoding (`PubMeets ↔ Meets`: `band_pub_iff_64`). Deployment
conditions (§48): (1) the check is the exact, full-decode check (a sampled or spot check is not covered); (2) the
salt is fixed by the verifier or the protocol, not
the server; (3) `W` is fixed before setup, and `Π₂` is independent of `W` and the salt; (4) the commitment to `W` and
the binding of `C*` are idealized. Proof: `PousPub.band_pub_meets_64_final`; here
`Pous.SecurityProofs.BandPubMeets64`.

Frozen (no-function, hash): `PubMeets` with the model's `exactCheck`, not the setup check `PousCheck` runs. -/
def BandPubMeets64 : Prop :=
  ∀ d : ℕ, 5 ≤ d → ∀ salt : Bits 256,
    PubMeets (Pous.BandChain.chainBand (2 ^ 9) d 524288 512 (Pous.Sponge.dom salt)) (exactCheck _) .sequential 5
      (2 ^ 20) 111 ((2⁻¹ : ℝ≥0∞) ^ 128)

/-- **Unconditional: the band at 14 GiB with a public encoder, per-segment tags** (`PubMeets ↔ Meets`:
`band_pub_iff_14`), under the same four deployment conditions (§48). Proof: `PousPub.band_pub_meets_14_final`; here
`Pous.SecurityProofs.BandPubMeets14`.

Frozen (no-function, hash): `PubMeets` with the model's `exactCheck`, not the setup check `PousCheck` runs. -/
def BandPubMeets14 : Prop :=
  ∀ d : ℕ, 5 ≤ d → ∀ salt : Bits 192,
    PubMeets (Pous.BandMulti.chainBandSeg 448 (2 ^ 9) d 524288 512 (Pous.BandMulti.segDom salt (2 ^ 9)))
      (exactCheck _) .sequential 5 (2 ^ 20) 111 ((2⁻¹ : ℝ≥0∞) ^ 128)

/-- **Unconditional: the band at 14 GiB with a public encoder, one global salt** (`PubMeets ↔ Meets`:
`band_pub_iff_14_global`), under the same four deployment conditions (§48). Proof:
`PousPub.band_pub_meets_14_global_final`; here `Pous.SecurityProofs.BandPubMeets14Global`.

Frozen (no-function, hash): `PubMeets` with the model's `exactCheck`, not the setup check `PousCheck` runs. -/
def BandPubMeets14Global : Prop :=
  ∀ d : ℕ, 5 ≤ d → ∀ salt : Bits 256,
    PubMeets (Pous.BandMulti.chainBandSeg 448 (2 ^ 9) d 524288 512 (Pous.Sponge.dom salt)) (exactCheck _) .sequential
      5 (2 ^ 20) 111 ((2⁻¹ : ℝ≥0∞) ^ 128)

/-- **Unconditional: the band store with a public encoder at every segment count from 1 to 2^64.** For every `N` with
`1 ≤ N ≤ 2^64` and every `d ≥ 5`, `PubMeets` for the `N`-segment band `chainBandSeg N (2^9) d` with the exact check, at
`D = 5`, `Q = 2^20`, `k = 111`, `ε = 2^-128`, both with the implementation's per-segment tags (`segDom salt`, segment
`s` keyed by `salt ‖ u64(s)`, any 192-bit salt) and with one global salt and global indices (`dom salt`). It is
`BandMultiMeetsFamily` with an untrusted server encoding the data (`PubMeets ↔ Meets` at every `N`: `band_pub_iff_seg`,
`band_pub_iff_seg_global`), under the same four deployment conditions (§48). At `N = 448` it is `BandPubMeets14` and
`BandPubMeets14Global`. Proof:
`PousPub.band_pub_meets_family_final` (`Security/Proofs/Pous/Chain/PubFamily.lean`); here
`Pous.SecurityProofs.BandPubMeetsFamily`.

Frozen (no-function, hash): `PubMeets` with the model's `exactCheck`, not the setup check `PousCheck` runs. -/
def BandPubMeetsFamily : Prop :=
  ∀ (N : ℕ) [NeZero N], N ≤ 2 ^ 64 → ∀ d : ℕ, 5 ≤ d →
    (∀ salt : Bits 192,
      PubMeets (Pous.BandMulti.chainBandSeg N (2 ^ 9) d 524288 512 (Pous.BandMulti.segDom salt (2 ^ 9)))
        (exactCheck _) .sequential 5 (2 ^ 20) 111 ((2⁻¹ : ℝ≥0∞) ^ 128)) ∧
    (∀ salt : Bits 256,
      PubMeets (Pous.BandMulti.chainBandSeg N (2 ^ 9) d 524288 512 (Pous.Sponge.dom salt)) (exactCheck _)
        .sequential 5 (2 ^ 20) 111 ((2⁻¹ : ℝ≥0∞) ^ 128))

/-- **Unconditional: the band store with a public encoder at `D = 2` rounds and `k = 97`, at every segment count from
1 to 2^64.** `BandPubMeetsFamily` at `D = 2` and `k = 97`, for every `d ≥ 2`: it is `BandMultiMeetsFamilyD2` with an
untrusted server encoding the data and the exact check (`PubMeets ↔ Meets` at every `N`), both domain layouts, under
the same four deployment conditions (§48). Proof:
`PousPub.band_pub_meets_family_d2` (`Security/Proofs/Pous/Chain/PubRounds.lean`); here
`Pous.SecurityProofs.BandPubMeetsFamilyD2`.

Frozen (no-function, hash): `PubMeets` with the model's `exactCheck`, not the setup check `PousCheck` runs. -/
def BandPubMeetsFamilyD2 : Prop :=
  ∀ (N : ℕ) [NeZero N], N ≤ 2 ^ 64 → ∀ d : ℕ, 2 ≤ d →
    (∀ salt : Bits 192,
      PubMeets (Pous.BandMulti.chainBandSeg N (2 ^ 9) d 524288 512 (Pous.BandMulti.segDom salt (2 ^ 9)))
        (exactCheck _) .sequential 2 (2 ^ 20) 97 ((2⁻¹ : ℝ≥0∞) ^ 128)) ∧
    (∀ salt : Bits 256,
      PubMeets (Pous.BandMulti.chainBandSeg N (2 ^ 9) d 524288 512 (Pous.Sponge.dom salt)) (exactCheck _)
        .sequential 2 (2 ^ 20) 97 ((2⁻¹ : ℝ≥0∞) ^ 128))

/-- **Unconditional: the band store with a public encoder at `D = 1` round and `k = 93`, at every segment count from
1 to 2^64.** `BandPubMeetsFamily` at `D = 1` and `k = 93`, for every `d ≥ 1`: `BandMultiMeetsFamilyD1` with a public
encoder and the exact check, both domain layouts, under the same four deployment conditions (§48). Proof:
`PousPub.band_pub_meets_family_d1`; here
`Pous.SecurityProofs.BandPubMeetsFamilyD1`.

Frozen (no-function, hash): `PubMeets` with the model's `exactCheck`, not the setup check `PousCheck` runs. -/
def BandPubMeetsFamilyD1 : Prop :=
  ∀ (N : ℕ) [NeZero N], N ≤ 2 ^ 64 → ∀ d : ℕ, 1 ≤ d →
    (∀ salt : Bits 192,
      PubMeets (Pous.BandMulti.chainBandSeg N (2 ^ 9) d 524288 512 (Pous.BandMulti.segDom salt (2 ^ 9)))
        (exactCheck _) .sequential 1 (2 ^ 20) 93 ((2⁻¹ : ℝ≥0∞) ^ 128)) ∧
    (∀ salt : Bits 256,
      PubMeets (Pous.BandMulti.chainBandSeg N (2 ^ 9) d 524288 512 (Pous.Sponge.dom salt)) (exactCheck _)
        .sequential 1 (2 ^ 20) 93 ((2⁻¹ : ℝ≥0∞) ^ 128))

/-! ### The dense deployment over the overwrite chain

The dense chain scheme at the deployment point is pinned here at `k = 111`, which is proved, rather than at `k = 107`
(`SpongeDense`'s `ChainDenseMeetsPinned`), which the ex post facto argument does not reach. Red team: GO
(`docs/lean-trusted-layer-review.md` §§53–54). -/

/-- **Unconditional: the dense chain scheme at the deployment point, `k = 111`.** For every salt,
`chainDense (2^16) 8192 512 (dom salt)` (`2^16` blocks of 1 KiB labels on the complete DAG, `H` the overwrite chain on
one ideal permutation) meets the requirement with sequential reveal, `D = 64` rounds, `Q = 2^20` queries per answer,
`k = 111` challenges and `ε = 2^-128`. It is strictly weaker than the `k = 107` target (`ChainDenseMeetsPinned`), which
stays open: it costs four more challenges. Not `DenseMeets`, which is the dense scheme with `H` a random oracle at
`k = 107`. Proof: `PousSpongeChain.chainDenseMeets111_final`; here `Pous.SecurityProofs.ChainDenseMeets111`.

Frozen (no-function, hash): a `Meets` in an ideal model, which no Lean function computes. -/
def ChainDenseMeets111 : Prop :=
  ∀ salt : Bits 256,
    Meets (Pous.Sponge.chainDense (2 ^ 16) 8192 512 (Pous.Sponge.dom salt)) .sequential 64 (2 ^ 20) 111
      ((2⁻¹ : ℝ≥0∞) ^ 128)

/-! ### The band with grace `t` (`Definitions.Pous.Model.Graded`, `Definitions.Pous.Accounting.GradedMeets`)

The game's requirements on the protocol are in
`Security/Definitions/Pous/Model/Graded.lean` and `PROTOCOL.md`, "The graded audit". -/

/-- **Unconditional: the band store with up to `t` late answers, at every segment count from 1 to 2^64.** For every `N`
with `1 ≤ N ≤ 2^64`, every `d ≥ 5`, every hard bound `(D_late, Q_late)` with `5 ≤ D_late ≤ d`, `D_late ≤ 12` and
`2^20 ≤ Q_late`, and each `(t, k)` of `(0, 111)`, `(1, 160)`, `(2, 203)`, `(3, 243)`, `GradedMeets` for the `N`-segment
band `chainBandSeg N (2^9) d` at `D = 5`, `Q = 2^20`, `ε = 2^-128`, both with the implementation's per-segment tags
(`segDom salt`, any 192-bit salt) and with one global salt and global indices (`dom salt`). Each `k` is the least with
`P[Bin(k, 95923/100000) ≥ k − t] ≤ 1%`; at `t = 0` and `k = 111` the row is `BandMultiMeetsFamily`'s
(`PousGraded.gradedMeets_zero_iff`).

The hypotheses on `D_late` are the deployment condition, necessary for the game's `S`-bit state to be a model of
storage at rest: the audit's window must be too short to re-encode `C`. Re-encoding a segment from `W` takes
`R(d) = Σ_{v < 2^9} (min(v, d) + 1) ≥ R(5) = 3,057` sequential `Π₂` calls. Every row's rounds, `k − t` at `D = 5` and
`t` at `D_late`, last under `240·6 + 3·13 = 1,479` call-times, and the memory challenger's gaps between them add to
the window, so it is bounded by the memory challenger's limit on the audit's duration that `BackgroundRebuildExcluded`
requires, `1,505` call-times. `D_late ≤ d` is the band's cliff applied to `Δ_late`; at the deployed `d = 12` both caps
on `D_late` are `Δ_late + RTT < 13 × 241.5 µs ≈ 3.14 ms`. That window argument and its wall-clock reading count whole
`Π₂` calls, so they hold in the ideal-permutation model only: at the reference Feistel-SHAKE256 `Π₂` a prover re-encodes
nothing, rebuilding any label from 1.53% of `C` in 481 Keccak-f, less than one call. The statement itself is about the
ideal model and is unaffected. Proof:
`PousGraded.band_graded_meets_family` (`Security/Proofs/Pous/Graded/Band.lean`); here
`Pous.SecurityProofs.BandGradedMeetsFamily`.

Frozen (no-function, hash): a `GradedMeets` in an ideal model, which no Lean function computes. -/
def BandGradedMeetsFamily : Prop :=
  ∀ (N : ℕ) [NeZero N], N ≤ 2 ^ 64 → ∀ d : ℕ, 5 ≤ d →
    ∀ D_late Q_late : ℕ, 5 ≤ D_late → D_late ≤ d → D_late ≤ 12 → 2 ^ 20 ≤ Q_late →
    ∀ tk ∈ ([(0, 111), (1, 160), (2, 203), (3, 243)] : List (ℕ × ℕ)),
      (∀ salt : Bits 192,
        GradedMeets (Pous.BandMulti.chainBandSeg N (2 ^ 9) d 524288 512 (Pous.BandMulti.segDom salt (2 ^ 9))) 5
          (2 ^ 20) D_late Q_late tk.2 tk.1 ((2⁻¹ : ℝ≥0∞) ^ 128)) ∧
      (∀ salt : Bits 256,
        GradedMeets (Pous.BandMulti.chainBandSeg N (2 ^ 9) d 524288 512 (Pous.Sponge.dom salt)) 5 (2 ^ 20) D_late
          Q_late tk.2 tk.1 ((2⁻¹ : ℝ≥0∞) ^ 128))

/-! ### The continuous audit (`Definitions.Pous.Model.Continuous`)

Red team: needed. The memory challenger enforces the three timing
rules: it reveals the next index only after the last round closed, rejects an answer past the hard deadline, and counts
every answer past `Δ` as a miss, with every rejected or wrong one. `ContinuousAudit` takes no hypothesis about them,
and holds at every hard deadline `(D_late, Q_late)` and every arrival schedule `arr`
(`Security/Definitions/Pous/Model/Continuous.lean`). Its per-answer step, `PerAnswer`, the completeness theorem
`ContinuousComplete` and the time average `ContinuousTimeAverage` are stated in `Security/Proofs/Pous/Statements.lean`.
-/

/-- **The continuous audit's guarantee.** From `TimedINC sch D Q β π`, for every machine,
every arrival schedule, every hard deadline `(D_late, Q_late)`, every `f` and every `βe > 0`, over
the setup and `n` uniform indices: the memory challenger accepts with tolerance `f` and `Σ_j max(ε_j − ε_proof, 0)`
exceeds `G(f) = inf_{θ > 0} (θ·f + ln(1/βe)) / (1 − e^−θ)` with probability at most `βe + n·(β + 1)·π`. Here `ε_j` is
the freed fraction of the responder's memory at challenge `j`'s send time. The positive part keeps a memory above `|C|`
at one challenge (`ε_j < 0`) from offsetting a shortfall at another. Dividing by `n`:
`Pr[accept(f₀) ∧ ε̄ > g(f₀)] ≤ βe + n·(β + 1)·π`, with `ε̄ = (1/n)·Σ_j max(ε_j − ε_proof, 0)` and `g = G/n`. When
`ε_proof ≥ 0` (whenever the bound is not vacuous), `max(ε_j − ε_proof, 0) ≥ max(ε_j, 0) − ε_proof`, so the same bound
holds for the average of `max(ε_j, 0)` exceeding `ε_proof + g(f₀)`. The scope is the module docstring's: conditional
on the arrivals, storage at send times, and `g` normalized by the realized number of challenges, never by `λT`. Proof:
`PousContinuous.continuous_audit`; here `Pous.SecurityProofs.ContinuousAudit`.

Frozen (no-function): the guarantee `PROTOCOL.md` and `continuous.py` cite, over the model's memory challenger
(`contAccepts`), not `MemoryChallenger.verdict`. -/
def ContinuousAudit : Prop :=
  ∀ {M : IdealModel} (sch : Scheme M) (D Q D_late Q_late β n : ℕ) (π : ℝ≥0∞), TimedINC sch D Q β π →
    ∀ (W : Bits sch.n) (mach : ContMachine sch) (arr : ℕ → ℕ) (f βe : ℝ), 0 < βe →
      pr (fun x : Outcome sch n =>
          let pp := (sch.enc W x.1 x.2.1).1
          let C := (sch.enc W x.1 x.2.1).2
          let rs := contRecords mach D Q D_late Q_late W pp C x.1 arr .start (List.ofFn x.2.2)
          contAccepts f rs ∧
            (⨅ θ : Set.Ioi (0 : ℝ), ((θ : ℝ) * f + Real.log (1 / βe)) / (1 - Real.exp (-(θ : ℝ)))) <
              (rs.map fun r => max (contFreed sch r.held - epsProof sch β) 0).sum)
        ≤ ENNReal.ofReal βe + n * (β + 1) * π

/-! ### The background-rebuild exclusion (`Definitions.Pous.Accounting.Rebuild`)

Red team: GO. -/

/-- **Under the memory challenger's cap on an audit's duration: no graded audit lasts long enough to re-encode a whole
band segment from `W` alone**, with no state about `Π₂`. It is the deployment condition of `BandGradedMeetsFamily`
(`PROTOCOL.md`, "Isolation over the whole graded window"), as counts and as a bound.
1. **The counts.** Re-encoding a segment of `chainBand (2^9) d` from `W` costs `R(d) = rebuildCalls (2^9) d` sequential
   `Π₂` calls: `R(5) = 3,057`, `R(12) = 6,578`, and `R(d) ≥ 3,057` for every `d ≥ 5`. The rounds of each row `(t, k)`
   of `BandGradedMeetsFamily` last under `gradedRoundSpan 5 D_late k t ≤ 240·6 + 3·13 = 1,479` call-times when
   `D_late ≤ 12`: a round answered at its deadline lasts just under `D + 1` call-times, not `D`.
2. **The bound.** For every `d ≥ 5`, every domain map injective on the segment's `2^9` indices, every `W`, `Q ≤ 2^128`
   and every `T ≤ auditCap = 1,505`, programs that know `W` and the domains but nothing about `Π₂` beyond their own
   queries, each within `T` rounds and `Q` queries, output all `2^9` labels of `chainBand (2^9) d 524288 512 dom` with
   probability at most `2^-128` over `Π₂`. Segment `s` of the deployed store is such a segment, with domains `dom tag_s`
   under the per-segment tags and `j ↦ dom salt (s·2^9 + j)` under one global salt (`Pous.BandMulti.segFaithfulCode`,
   `Pous.BandMulti.segFaithful`).

**The precondition is the memory challenger's.** A rebuild runs beside the audit on the wall clock, through the rounds
and the memory challenger's gaps between them, and no round deadline bounds the gaps. So the bound covers a graded
audit only if the memory challenger aborts, without accepting, every audit that lasts more than `auditCap · call_ns` (≈
363.5 ms at 241.5 µs per call) from its first challenge's send to its last answer's receipt. If every round of the worst
row answers at its deadline, that leaves its 242 gaps `1,505 − 1,479 = 26` call-times (≈ 6.3 ms). The bound rests on the
cap, which is the proof's reach, not on `R(d)`, which exceeds it. The milliseconds read call-times as whole `Π₂` calls
of at least 241.5 µs, which holds in the ideal-permutation model only: at the reference Feistel-SHAKE256 `Π₂` a prover
answers from 1.53% of `C` in 481 Keccak-f without re-encoding anything.

**What it does not exclude.** It excludes only re-encoding a whole segment from `W` with no state about `Π₂`. Within the
cap, `W` alone rebuilds a segment's first 253 labels (49%) at `d = 5` and its first 121 (24%) at `d = 12`. `d`
consecutive held labels restart the chain, so two held runs (10 labels, 2.0% of the segment) at `d = 5`, or four
(48 labels, 9.4%) at `d = 12`, let a prover re-encode a whole segment within the cap. That gap between the storage held
at rest and what is present when a challenge arrives is not bounded here. Proof: `PousRebuild.background_rebuild_excluded`
(`Security/Proofs/Pous/Rebuild/`); here `Pous.SecurityProofs.BackgroundRebuildExcluded`.

Frozen (no-function, hash): an ideal-permutation bound whose precondition, an abort past `auditCap`, neither Lean
verdict checks. -/
def BackgroundRebuildExcluded : Prop :=
  Pous.Rebuild.rebuildCalls (2 ^ 9) 5 = 3057 ∧ Pous.Rebuild.rebuildCalls (2 ^ 9) 12 = 6578 ∧
  (∀ d : ℕ, 5 ≤ d → 3057 ≤ Pous.Rebuild.rebuildCalls (2 ^ 9) d) ∧
  (∀ D_late : ℕ, D_late ≤ 12 → ∀ tk ∈ ([(0, 111), (1, 160), (2, 203), (3, 243)] : List (ℕ × ℕ)),
    Pous.Rebuild.gradedRoundSpan 5 D_late tk.2 tk.1 ≤ 1479) ∧
  ∀ d : ℕ, 5 ≤ d → ∀ dom : ℕ → Bits 512, Function.Injective (fun i : Fin (2 ^ 9) => dom i) →
    ∀ (W : Bits (2 ^ 9 * 524288)) (Q : ℕ), Q ≤ 2 ^ 128 → ∀ T : ℕ, T ≤ Pous.Rebuild.auditCap →
    ∀ A : Fin (2 ^ 9) → Prog (Pous.Sponge.idealPerm (524288 + (524288 + 512))) (Bits 524288),
      (∀ i, (A i).Bounded T Q) →
      pr (fun π : Equiv.Perm (Bits (524288 + (524288 + 512))) =>
          ∀ i, (A i).run ((Pous.Sponge.idealPerm (524288 + (524288 + 512))).answer π) =
            ((Pous.BandChain.chainBand (2 ^ 9) d 524288 512 dom).enc W π ()).2 i)
        ≤ (2⁻¹ : ℝ≥0∞) ^ 128

/-! ### The continuous audit, time-averaged, with arrivals drawn by the memory challenger

Red team: needed. The memory challenger of `ContinuousAudit`,
unchanged, run on arrivals it draws itself: one coin per slot of one unit
(`Security/Definitions/Pous/Model/Continuous.lean`, "Arrivals drawn by the memory challenger"). Arrivals queue
behind an open round (rule 1). The memory is read at every idle slot, not only at send times, and the bound is
normalized by the expected number of challenges `λT`, not the realized one. -/

/-- **Unconditional: the continuous audit's time-averaged bound on the band store at `D = 0, 1, 2, 5` rounds, at
every segment count from 1 to 2^64.** `ContinuousTimeAverage` instantiated on the `N`-segment band
`chainBandSeg N (2^9) d 524288 512 dom`, for every `N` with `1 ≤ N ≤ 2^64`, both domain layouts (the
implementation's per-segment tags `segDom salt`, any 192-bit salt, and one global salt `dom salt`), every `d ≥ D`
(every `d` at `D = 0`), at `Q = 2^20` and each `(D, k)` of `(0, 89)`, `(1, 93)`, `(2, 97)`, `(5, 111)`. `β` is the
compression width of the sequential certificate at `(D, k)` (`BandMultiMeetsFamilyD0`, `D1`, `D2`,
`BandMultiMeetsFamily`): `524288·(511 − D) − 69·(512 − D) − 1 − (9k + 129)` at `N = 1` and
`523898·(512 − D)·N − 532588` at `N ≥ 2`. The tail is at most `2^-(9k + 99)` per slot: `2^-900`, `2^-936`, `2^-972`,
`2^-1098` (at `N ≥ 2` it is below `2^-8206`). So `ε_proof = 1 − (β − 512·N)/(N·2^28)` (`epsProof`) is

| `D` | `k` | `N = 1` | `N = 2` | `N = 448` | `N = 2240` |
|---|---|---|---|---|---|
| 0 | 89 | 0.2090% | 0.1738% | 0.0750% | 0.0747% |
| 1 | 93 | 0.4043% | 0.3689% | 0.2702% | 0.2698% |
| 2 | 97 | 0.5996% | 0.5641% | 0.4654% | 0.4650% |
| 5 | 111 | 1.1855% | 1.1496% | 1.0509% | 1.0505% |

These slacks are for the idle slots only (`ContinuousTimeAverage`). At `D = 2` and `D = 5` a responder that pads its
hits to `D` units makes about 2.5% and 9.9% of the slots busy without a miss (100 challenges/s), and the bound reads
nothing there; at `D_late ≤ 1` no slot is busy.

Proof: `PousContinuous.band_family_continuous_time_average`; here
`Pous.SecurityProofs.BandFamilyContinuousTimeAverage`.

Frozen (no-function, hash): over the model's memory challenger in the ideal-permutation model, not
`MemoryChallenger.verdict`. -/
def BandFamilyContinuousTimeAverage : Prop :=
  ∀ (N : ℕ) [NeZero N], N ≤ 2 ^ 64 → ∀ Dk ∈ ([(0, 89), (1, 93), (2, 97), (5, 111)] : List (ℕ × ℕ)), ∀ d : ℕ,
    Dk.1 ≤ d →
    ∀ dom : ℕ → Bits 512,
      ((∃ salt : Bits 192, dom = Pous.BandMulti.segDom salt (2 ^ 9)) ∨ ∃ salt : Bits 256, dom = Pous.Sponge.dom salt) →
    let sch := Pous.BandMulti.chainBandSeg N (2 ^ 9) d 524288 512 dom
    let β := if N = 1 then 524288 * (511 - Dk.1) - 69 * (512 - Dk.1) - 1 - (9 * Dk.2 + 129)
      else 523898 * ((512 - Dk.1) * N) - 532588
    ∀ (D_late Q_late T a b : ℕ), 0 < b → a ≤ b →
    ∀ (W : Bits sch.n) (mach : ContMachine sch) (f βe : ℝ), 0 < βe →
      pr (fun x : (Pous.Sponge.idealPerm (524288 + (524288 + 512))).Ω × sch.Coins × (Fin T → Fin b × Fin sch.B) =>
          let pp := (sch.enc W x.1 x.2.1).1
          let C := (sch.enc W x.1 x.2.1).2
          let rs := contRecords mach Dk.1 (2 ^ 20) D_late Q_late W pp C x.1 (arrAt a x.2.2) .start
            (histAt a x.2.2 T)
          contAccepts f rs ∧
            (⨅ θ : Set.Ioi (0 : ℝ), ((θ : ℝ) * f + Real.log (1 / βe)) / (1 - Real.exp (-(θ : ℝ)))) <
              ((a : ℝ) / b) * ∑ t : Fin T,
                if contIdle mach D_late W pp C x.1 a x.2.2 t then
                  max (contFreed sch (contHeldAt mach D_late W pp C x.1 a x.2.2 t) - epsProof sch β) 0
                else 0)
        ≤ ENNReal.ofReal βe + T * (2⁻¹ : ℝ≥0∞) ^ (9 * Dk.2 + 99)

/-! ### Proved results cited outside the package

The types of three theorems about schemes that the protocol's documents and code cite, written out so that each citation
names a statement (seven more, which are math, are in `Security/Proofs/Pous/Statements.lean`). Each is proved in
`Pous.SecurityProofs` under its own name, from the theorem its docstring names. -/

open Dense in
/-- **The dense scheme meets the requirement at the 64 KB point**, random-oracle `H`, with sequential reveal, `D = 5`
rounds, `Q = 2^20`, `k = 106` challenges and `ε = 2^-128`: correct, rate 1, and every `⌊ρ|C|⌋`-bit adversary is
accepted with probability at most `1% + 2^-128`. Proof: `PousDenseReCert.dense_meets_64`.

Frozen (no-function, hash): a `Meets` in an ideal model, which no Lean function computes. -/
def DenseMeets64 : Prop :=
  Pous.Meets (denseScheme (2 ^ 9) 524288) .sequential 5 (2 ^ 20) 106 ((2⁻¹ : ENNReal) ^ 128)

open Dense PousDenseReCert in
/-- **The segmented dense scheme meets the requirement at 14 GB** (`N = 448` segments of `2^9` blocks of 64 KB),
random-oracle `H`, sequential reveal, `D = 5`, `Q = 2^20`, `k = 106`, `ε = 2^-128`. Same `k` as one segment: the audit
soundness is per-block, so it does not degrade with the number of segments. Proof: `PousDenseReCert.seg_meets_14`.

Frozen (no-function, hash): a `Meets` in an ideal model, which no Lean function computes. -/
def SegMeets14 : Prop :=
  Pous.Meets (segScheme 448 (2 ^ 9) 524288) .sequential 5 (2 ^ 20) 106 ((2⁻¹ : ENNReal) ^ 128)

open Dense PousDenseReCert in
/-- **The 14 GB theorem for the implementation's keying.** The tagged segmented dense scheme (448 segments of `2^9`
blocks of 64 KB, oracle `H(salt ‖ u64 s, local query)`) meets the requirement for every salt, random-oracle `H`,
sequential reveal, `D = 5`, `Q = 2^20`, `k = 106`, `ε = 2^-128`. Proof: `PousDenseReCert.segTag_meets_14`.

Frozen (no-function, hash): a `Meets` in an ideal model, which no Lean function computes. -/
def SegTagMeets14 : Prop :=
  ∀ salt : Bits 192,
    Pous.Meets (segTagScheme 448 (2 ^ 9) 524288 salt) .sequential 5 (2 ^ 20) 106 ((2⁻¹ : ENNReal) ^ 128)

end Pous.Guarantees
