import Flock.Witness
import Flock.ProvedScope
import Flock.Desc

/-!
# The deployed session's files parse

`SampledKSLiveV2`, `SampledKSLiveV3` and `SampledKSLivePub` take a circuit file that parses to a descriptor `D`, a
public file that loads at it with no shared-row tables, and `scopeOkH` at `D`. Whether the deployed files do is a
computation on a 56 MB circuit file, which the kernel can't run, so it is one named premise, stated with the verifier's
own functions on the files `verify --zk` reads, each pinned by its SHA-512, and checked by running the verifier on them.
-/

namespace Flock.Bridge.Assumptions

open Flock

/-- **The deployed session's files parse.** There are a circuit file and a public file with the SHA-512 digests of the
row/v1 session fixture `art:88544efa5af6f50ab24ad04a8e8d3118075e9f62898e41bbfb45ec3f3bc3d85b`, recorded at the
verifier's coin schedule (`rounds_log` 9): `stage/circuit.txt`, 56,327,749 bytes, and `stage/pub-32.bin`, 44,158 bytes,
such that, under the tags `verify --zk --statement verity/flock-circuit` sets one session up with (`Witness.zkTags`
at the inputs `⟨Tags.circuit, cf, pf, 1, [], none, none⟩`: one table, no session tables, and no partition or program,
which the tags don't read), the circuit file parses to a descriptor `D` (`Desc.parse`), `D` is in the hidden-output headlines' scope
(`ProvedScope.scopeOkH`), and the public file loads at `D` (`HmOut.loadPublic`) with no shared-row tables.

Assumption record. It assumes a computation's result: run `r20261026-231806-1121` evaluated each conjunct with these
functions, built from the verifier's source, on that fixture's two files, beside `flock-verify verify --zk` on its
session. It asks nothing of a hash or of the prover; a change to any of these functions or to the files makes it a new
premise, to be checked by a new run. -/
def ServedFilesParse : Prop :=
  ∃ cf pf : ByteArray,
    hexOf (Sha512.hash cf) =
        "7b9e46f755bced2c6e00a7706f3fca90631864ca22a2d08b51c8b82260a7bd9a9ae7a191c0eabe2993a6fbf686b832c440f2a78dad65fa3b13b3cf5bd47284e0" ∧
      hexOf (Sha512.hash pf) =
        "6bea4647565a87a873bc66670482ec0cdb86c4424d061ccb03354b81ec55cbe302564836f08c4a9757085746f6000b904657749c2f84b028fe56fde816c26dd1" ∧
      ∃ D : Desc,
        Desc.parse (Witness.zkTags ⟨Tags.circuit, cf, pf, 1, [], none, none⟩) cf [] = .ok D ∧
        ProvedScope.scopeOkH D.unroll = true ∧
        ∃ pub : Public,
          HmOut.loadPublic (Witness.zkTags ⟨Tags.circuit, cf, pf, 1, [], none, none⟩) D.unroll pf = .ok pub ∧
          pub.tables = none

end Flock.Bridge.Assumptions
