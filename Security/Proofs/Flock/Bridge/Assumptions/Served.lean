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
row/v1 session fixture `art:eeb1ed6ad88c683f6dfd3b0ffd2e9941c47f2fcb6f39c097e8f2f20720c6ce4a` (`stage/circuit.txt`,
56,327,749 bytes, and `stage/pub-32.bin`, 44,158 bytes) such that, under the tags `verify --zk --statement
verity/flock-circuit` sets one session up with (`Witness.zkTags`: `Tags.circuit`, one table, no session tables), the
circuit file parses to a descriptor `D` (`Desc.parse`), `D` is in the hidden-output headlines' scope
(`ProvedScope.scopeOkH`), and the public file loads at `D` (`HmOut.loadPublic`) with no shared-row tables.

Assumption record. It assumes a computation's result: run `r20261026-225839-14e0` evaluated each conjunct with these
functions, built from the verifier's source, on that fixture's two files, beside `flock-verify verify --zk` on its
session. It asks nothing of a hash or of the prover; a change to any of these functions or to the files makes it a new
premise, to be checked by a new run. -/
def ServedFilesParse : Prop :=
  ∃ cf pf : ByteArray,
    hexOf (Sha512.hash cf) =
        "7b9e46f755bced2c6e00a7706f3fca90631864ca22a2d08b51c8b82260a7bd9a9ae7a191c0eabe2993a6fbf686b832c440f2a78dad65fa3b13b3cf5bd47284e0" ∧
      hexOf (Sha512.hash pf) =
        "142731972d7b8f4cf0b89a9eaa33530ead7eb86c2628fd24ed14a54c9850965659cba8b8dac42454d3794de98f9e93fe84c4a1525ed729483fe8aa347f968663" ∧
      ∃ D : Desc,
        Desc.parse (Witness.zkTags { tags := Tags.circuit, circuitFile := cf, publicFile := pf, count := 1, tables := [],
          partition := none, program := none }) cf [] = .ok D ∧
        ProvedScope.scopeOkH D.unroll = true ∧
        ∃ pub : Public,
          HmOut.loadPublic (Witness.zkTags { tags := Tags.circuit, circuitFile := cf, publicFile := pf, count := 1,
            tables := [], partition := none, program := none }) D.unroll pf = .ok pub ∧
          pub.tables = none

end Flock.Bridge.Assumptions
