import Flock.Witness
import Flock.ProvedScope
import Flock.Desc
import Flock.Registered

/-!
# The served registered session's files parse at public inputs

`SampledKSLiveV5` takes a circuit file that parses to a descriptor `D` in `scopeOkH` whose ports have distinct names, the
circuit's public input ports `pi` (`HmIn.parse`), and a public file that loads over them (`HmIn.loadPublic`). Whether the
served files do is a computation on a 56 MB circuit file, which the kernel can't run, so it is one named premise, stated
with the verifier's own functions on the files `verify --zk` reads, each pinned by its SHA-512, and checked by running
those functions on them.
-/

namespace Flock.Bridge.Assumptions

open Flock

/-- **The served registered session's files parse at public inputs.** There are a circuit file and a public file with the
SHA-512 digests of the served registered `--zk` session
`art:88544efa5af6f50ab24ad04a8e8d3118075e9f62898e41bbfb45ec3f3bc3d85b` (`stage/circuit.txt`, 56,327,749 bytes, and
`stage/pub-32.bin`, 44,158 bytes, a shared-row public file of 32 instances), such that, under the tags
`verify --zk --statement verity/flock-circuit` sets one session up with (`Witness.zkTags` at the inputs
`⟨Tags.circuit, cf, pf, 1, [], none, none⟩`: one table, no session tables, and no partition or program, which the tags
don't read), the circuit file parses to a descriptor `D` (`Desc.parse`), `D` is in the hidden-output headlines' scope
(`ProvedScope.scopeOkH`), no two of its ports share a name (`Registered.dupName`), its public input ports parse
(`HmIn.parse`), and the public file loads over them (`HmIn.loadPublic`).

Assumption record. It assumes a computation's result: run `r20261027-025931-f22c` evaluated each conjunct with these
functions, built from the verifier's source at `main` 0600bab855, on that session's two files (the circuit has no
public input ports, and the public file loads with 32 instances and its shared-row tables), and run
`r20261027-114351-93be` evaluated them again at the source this module is checked at. It asks nothing of a hash or of
the prover; a change to any of these functions or to the files makes it a new premise, to be checked by a new run. -/
def ServedFilesParseV5 : Prop :=
  ∃ cf pf : ByteArray,
    hexOf (Sha512.hash cf) =
        "7b9e46f755bced2c6e00a7706f3fca90631864ca22a2d08b51c8b82260a7bd9a9ae7a191c0eabe2993a6fbf686b832c440f2a78dad65fa3b13b3cf5bd47284e0" ∧
      hexOf (Sha512.hash pf) =
        "6bea4647565a87a873bc66670482ec0cdb86c4424d061ccb03354b81ec55cbe302564836f08c4a9757085746f6000b904657749c2f84b028fe56fde816c26dd1" ∧
      ∃ D : Desc,
        Desc.parse (Witness.zkTags ⟨Tags.circuit, cf, pf, 1, [], none, none⟩) cf [] = .ok D ∧
        ProvedScope.scopeOkH D.unroll = true ∧
        Registered.dupName D.unroll = none ∧
        ∃ pi : HmIn.Ports, HmIn.parse D.unroll = .ok pi ∧
          ∃ pub : Public,
            HmIn.loadPublic (Witness.zkTags ⟨Tags.circuit, cf, pf, 1, [], none, none⟩) D.unroll pi pf = .ok pub

end Flock.Bridge.Assumptions
