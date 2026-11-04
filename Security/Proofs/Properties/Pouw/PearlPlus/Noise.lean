import Proofs.Properties.Pouw.PearlPlus.Noise.Part0
import Proofs.Properties.Pouw.PearlPlus.Noise.Part1
import Proofs.Properties.Pouw.PearlPlus.Noise.Part2
import Proofs.Properties.Pouw.PearlPlus.Noise.Part3

/-!
# The test noise, chunk by chunk

`Witness.test_noise` shows the test noise is `fixedNoise`'s at k = 128 from 24 kernel computations: each 16 of F_A's
and F_B's 128 noise lines (`Witness.test_fa0` to `Witness.test_fb7`) and each 4 of G's 32 rows (`Witness.test_gram0`
to `Witness.test_gram7`). They are in four modules of six, a quarter of each kind in each: lake checks the four in
parallel and each module checks its six one after another, so a build holds four chunks' kernel state at a time, where
one module would check all 24 one after another and 24 modules would hold all 24 at once.
-/
