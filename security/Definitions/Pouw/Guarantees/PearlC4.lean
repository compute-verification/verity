module

public import Definitions.Pouw.Advantage
public import Definitions.Pouw.Assumptions.PearlC.TTOutFp4Sm120W1
public import Definitions.Pouw.PearlC.DeviceFp4
public import Definitions.Pouw.PearlC.Game
public import Definitions.Pouw.PearlC.Instance
public import Definitions.Pouw.PearlC.Tile

/-!
# Pearl-C4's headline on sm_120 (trusted)

Pearl-C4's gap statements at its sm_120 record (`devFp4Sm120`), on the W1 setting `W1Device.fp4Sm120 sha fetch` under
`tt-out/fp4-sm120` at W1 (`TTOutFp4Sm120W1`, `TTOutTileFp4Sm120W1`): per unit (`GapFp4Sm120_8192`, `_16384`), per audit
tile (`SampledFp4Sm120_8192`, `_16384`), and at 8192³ in ε form (`EpsFp4Sm120_8192`). Each is proved by the theorem
`Pouw.SecurityProofs.PearlC.G`.
-/

@[expose] public section

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom Pouw.PearlC.Fp4Dev

/-- **Pearl-C4's headline on sm_120, 16384³**, worst case under the cap `1/400`: `g = 5473473967/895794400000` (0.61102%).
It holds for noise the domain's rule allows (D-SK, `Fp4Sem.SaltKeyedB`, in `pearlCDomainFp4`): `B̃`'s lines are the ones
SHA-512 gives from `seed_B` of this job's salt. The registered scheme's noise (`pearl-c-nvfp4-v0`, h0) is such noise:
`sha512Fp4Sem`, `pearl_c4.py`'s lines op for op (`NoiseSha512Fp4`), meets D-SK, and its one-shape domain holds workloads
(`sha512Fp4Sem_inDomain`). -/
def _root_.Pouw.Guarantees.PearlC.GapFp4Sm120_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : Fp4Sem Q R S) (hTT : TTOutFp4Sm120W1 sha fetch salt sem (1 / 400)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.fp4Sm120 sha fetch) salt) (pearlCProtocolFp4 devFp4Sm120 sem (1 / 400)) (pearlCDomainFp4At sem sh16384)
    ((5473473967 / 895794400000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom Pouw.PearlC.Fp4Dev

/-- **Pearl-C4's headline on sm_120, 8192³**, worst case under the cap `1/400`: `g = 9875013901/1376663200000` (0.71732%).
It holds for noise the domain's rule allows (D-SK, `Fp4Sem.SaltKeyedB`, in `pearlCDomainFp4`): `B̃`'s lines are the ones
SHA-512 gives from `seed_B` of this job's salt. The registered scheme's noise (`pearl-c-nvfp4-v0`, h0) is such noise:
`sha512Fp4Sem`, `pearl_c4.py`'s lines op for op (`NoiseSha512Fp4`), meets D-SK, and its one-shape domain holds workloads
(`sha512Fp4Sem_inDomain`). -/
def _root_.Pouw.Guarantees.PearlC.GapFp4Sm120_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : Fp4Sem Q R S) (hTT : TTOutFp4Sm120W1 sha fetch salt sem (1 / 400)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.fp4Sm120 sha fetch) salt) (pearlCProtocolFp4 devFp4Sm120 sem (1 / 400)) (pearlCDomainFp4At sem sh8192)
    ((9875013901 / 1376663200000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom Pouw.PearlC.Fp4Dev

/-- **Pearl-C4's headline on sm_120 per audit tile, 16384³**: `GGapSampled` at `g = 5473473967/895794400000`. -/
def _root_.Pouw.Guarantees.PearlC.SampledFp4Sm120_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : Fp4Sem Q R S) (hTT : TTOutTileFp4Sm120W1 sha fetch salt sem (1 / 400)),
    GGapSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.fp4Sm120 sha fetch) salt) (pearlCProtocolFp4 devFp4Sm120 sem (1 / 400)) (pearlCTilesFp4 devFp4Sm120 sem (1 / 400))
    (pearlCDomainFp4At sem sh16384) ((5473473967 / 895794400000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom Pouw.PearlC.Fp4Dev

/-- **Pearl-C4's headline on sm_120 per audit tile, 8192³**: `GGapSampled` at `g = 9875013901/1376663200000`. -/
def _root_.Pouw.Guarantees.PearlC.SampledFp4Sm120_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : Fp4Sem Q R S) (hTT : TTOutTileFp4Sm120W1 sha fetch salt sem (1 / 400)),
    GGapSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.fp4Sm120 sha fetch) salt) (pearlCProtocolFp4 devFp4Sm120 sem (1 / 400)) (pearlCTilesFp4 devFp4Sm120 sem (1 / 400))
    (pearlCDomainFp4At sem sh8192) ((9875013901 / 1376663200000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC
namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom Pouw.PearlC.Fp4Dev

/-- **Pearl-C4's headline on sm_120, 8192³, in ε form**: `GapFp4Sm120_8192` at the advantage
`ε = g/(1 − g) = 9875013901/1366788186099` (0.72250%). Like `GapFp4Sm120_8192`, it holds for noise the domain's rule
allows (D-SK, `Fp4Sem.SaltKeyedB`), SHA-512's lines from `seed_B` of this job's salt, and so at the registered scheme's
noise (`pearl-c-nvfp4-v0`, h0; `sha512Fp4Sem`, `sha512Fp4Sem_inDomain`). -/
def _root_.Pouw.Guarantees.PearlC.EpsFp4Sm120_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : Fp4Sem Q R S),
    TTOutFp4Sm120W1 sha fetch salt sem (1 / 400) →
    Pouw.PearlC.Gε (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.fp4Sm120 sha fetch) salt) (pearlCProtocolFp4 devFp4Sm120 sem (1 / 400)) (pearlCDomainFp4At sem sh8192)
    ((9875013901 / 1366788186099 : ℚ) : ℝ) δPearlC

end Pouw.PearlC
