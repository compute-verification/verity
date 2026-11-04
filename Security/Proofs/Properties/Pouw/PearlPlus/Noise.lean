import Proofs.Properties.Pouw.PearlPlus.Noise.Fa0
import Proofs.Properties.Pouw.PearlPlus.Noise.Fa1
import Proofs.Properties.Pouw.PearlPlus.Noise.Fa2
import Proofs.Properties.Pouw.PearlPlus.Noise.Fa3
import Proofs.Properties.Pouw.PearlPlus.Noise.Fa4
import Proofs.Properties.Pouw.PearlPlus.Noise.Fa5
import Proofs.Properties.Pouw.PearlPlus.Noise.Fa6
import Proofs.Properties.Pouw.PearlPlus.Noise.Fa7
import Proofs.Properties.Pouw.PearlPlus.Noise.Fb0
import Proofs.Properties.Pouw.PearlPlus.Noise.Fb1
import Proofs.Properties.Pouw.PearlPlus.Noise.Fb2
import Proofs.Properties.Pouw.PearlPlus.Noise.Fb3
import Proofs.Properties.Pouw.PearlPlus.Noise.Fb4
import Proofs.Properties.Pouw.PearlPlus.Noise.Fb5
import Proofs.Properties.Pouw.PearlPlus.Noise.Fb6
import Proofs.Properties.Pouw.PearlPlus.Noise.Fb7
import Proofs.Properties.Pouw.PearlPlus.Noise.Gram0
import Proofs.Properties.Pouw.PearlPlus.Noise.Gram1
import Proofs.Properties.Pouw.PearlPlus.Noise.Gram2
import Proofs.Properties.Pouw.PearlPlus.Noise.Gram3
import Proofs.Properties.Pouw.PearlPlus.Noise.Gram4
import Proofs.Properties.Pouw.PearlPlus.Noise.Gram5
import Proofs.Properties.Pouw.PearlPlus.Noise.Gram6
import Proofs.Properties.Pouw.PearlPlus.Noise.Gram7

/-!
# The test noise, chunk by chunk

`Witness.test_noise` shows the test noise is `fixedNoise`'s at k = 128 from 24 kernel computations: each 16 of F_A's
and F_B's 128 noise lines (`Witness.test_fa0` to `Witness.test_fb7`) and each 4 of G's 32 rows (`Witness.test_gram0`
to `Witness.test_gram7`). Each is a module of its own, so that lake checks them in parallel; in one module the kernel
checks them one after another.
-/
