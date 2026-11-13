module

public import Flock.Witness
public import Flock.Desc
public import Flock.PortRoots

/-!
# The served registered session's files parse at public inputs

`SampledKSLiveV5` takes a circuit file that parses to a descriptor `D` in `scopeOkH`, the circuit's public input ports
`pi` (`HmIn.parse`), and a public file that loads over them (`HmIn.loadPublic`) whose ports' published domains are
pairwise distinct (`PortRoots.dupId`). Whether the
served files do is a computation on a 56 MB circuit file, which the kernel can't run, so it is one named premise, stated
with the verifier's own functions on the files `verify --zk` reads, each pinned by its SHA-512, and checked by running
those functions on them.
-/

@[expose] public section

namespace Flock.Bridge.Assumptions

open Flock

/-- **The served registered session's files parse at public inputs.** There are a circuit file and a public file with the
SHA-512 digests of the served registered `--zk` session (`SERVED` in
`verity/protocols/flock/prover/tests/test_lean_ks_form.py`: `stage/circuit.txt`, 56,327,749 bytes, and
`stage/pub-32.bin`, 44,158 bytes, a shared-row public file of 32 instances), such that, under the tags
`verify --zk --statement verity/flock-circuit` sets one session up with (`Witness.zkTags` at the inputs
`⟨Tags.circuit, cf, pf, 1, [], none, none⟩`: one table, no session tables, and no partition or program, which the tags
don't read), the circuit file parses to a descriptor `D` (`Desc.parse`), `D` is in the hidden-output headlines' scope
(`ProvedScope.scopeOkH`), its public input ports parse (`HmIn.parse`), the public file loads over them
(`HmIn.loadPublic`), and no two of `D`'s ports' published domains are one (`PortRoots.dupId`).

Assumption record. It assumes a computation's result: each conjunct evaluated with these functions, built from the
source this module is checked at, on that session's two files (the circuit has no public input ports, the public
file loads with 32 instances and its shared-row tables, and it publishes a 64-byte domain for each of the circuit's three
ports, no two equal); `flock-verify ks-form` reruns the form check on them
(`verity/protocols/flock/prover/tests/test_lean_ks_form.py`). It asks nothing of a hash or of the prover; a change to
any of these functions or to the files makes it a new premise, to be checked by a new run. -/
def ServedFilesParseV5 : Prop :=
  ∃ cf pf : ByteArray,
    hexOf (Sha512.hash cf) =
        "7b9e46f755bced2c6e00a7706f3fca90631864ca22a2d08b51c8b82260a7bd9a9ae7a191c0eabe2993a6fbf686b832c440f2a78dad65fa3b13b3cf5bd47284e0" ∧
      hexOf (Sha512.hash pf) =
        "6bea4647565a87a873bc66670482ec0cdb86c4424d061ccb03354b81ec55cbe302564836f08c4a9757085746f6000b904657749c2f84b028fe56fde816c26dd1" ∧
      ∃ D : Desc,
        Desc.parse (Witness.zkTags ⟨Tags.circuit, cf, pf, 1, [], none, none⟩) cf [] = .ok D ∧
        ProvedScope.scopeOkH D.unroll = true ∧
        ∃ pi : HmIn.Ports, HmIn.parse D.unroll = .ok pi ∧
          ∃ pub : Public,
            HmIn.loadPublic (Witness.zkTags ⟨Tags.circuit, cf, pf, 1, [], none, none⟩) D.unroll pi pf = .ok pub ∧
            PortRoots.dupId D.unroll pub = false

end Flock.Bridge.Assumptions
