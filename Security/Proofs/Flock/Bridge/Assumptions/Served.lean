import Flock.Witness
import Flock.ProvedScope
import Flock.Desc

/-!
# The deployed session's files parse

`SampledKSLiveV2`, `SampledKSLiveV3` and `SampledKSLivePub` take a circuit file that parses to a descriptor `D`, a
public file that loads at it with no shared-row tables, and `scopeOkH` at `D`. Whether the deployed files do is a
computation on a 97 MB circuit file, which the kernel can't run, so it is one named premise, stated with the verifier's
own functions on the files `verify --zk` reads, each pinned by its SHA-512, and checked by running the verifier on them.
-/

namespace Flock.Bridge.Assumptions

open Flock

/-- **The deployed session's files parse.** There are a circuit file and a public file with the SHA-512 digests of the
honest case of the one-stage audit's `--zk` fixture `art:31c671f6ef98b1d361a0a93dbff17bf8261f6dee4f8edbaf0df94a7454134af5`
(`benchmarks/one_stage/forged_proofs.sh`'s files with their sessions regenerated at the 2^9 coin schedule: one A8
unit, `LlBf16DotprodDot_v1` at K = 128, staged by `UnitStager` bound to a template query's partition, four instances,
every unit drawn): `stage/circuit.txt`, 97,364,260 bytes, and `honest/pub-4.bin`, 3,194 bytes, a per-instance public file
(no `shared_rows`), such that, under the tags `verify --zk --statement verity/flock-circuit` sets one session up with
(`Witness.zkTags` at the inputs `⟨Tags.circuit, cf, pf, 1, [], none, none⟩`: one table, no session tables, and no
partition or program, which the tags don't read), the circuit file parses to a descriptor `D` (`Desc.parse`), `D` is in
the hidden-output headlines' scope (`ProvedScope.scopeOkH`), and the public file loads at `D` (`HmOut.loadPublic`) with
no shared-row tables.

Assumption record. It assumes a computation's result: a recorded run evaluated each conjunct with these
functions, built from the verifier's source at `main` 2bb751070f, on that fixture's two files, beside
`flock-verify verify --zk` on its honest session, which it accepted. A shared-row public file (the served registered
fixtures) is `SampledKSLiveV5`'s. It asks nothing of a hash or of the prover; a change to any of these functions or to
the files makes it a new premise, to be checked by a new run. -/
def ServedFilesParse : Prop :=
  ∃ cf pf : ByteArray,
    hexOf (Sha512.hash cf) =
        "0c1d6ad42ebf34efa4370690f077ae1302319ba7dbde4f3ac18565ae97c06f11d2633d210e2b9a923f3cb1a398b34d31419b3ffea2d56a820927b0b1ca4db099" ∧
      hexOf (Sha512.hash pf) =
        "56861c780e639f52be33394f86b369de7bec45c3e5290d0150720673ffb6032b465ca3a5cb00953301661aa97a3e7a3e6fe4525651ecee7cc0e7cc13412e9209" ∧
      ∃ D : Desc,
        Desc.parse (Witness.zkTags ⟨Tags.circuit, cf, pf, 1, [], none, none⟩) cf [] = .ok D ∧
        ProvedScope.scopeOkH D.unroll = true ∧
        ∃ pub : Public,
          HmOut.loadPublic (Witness.zkTags ⟨Tags.circuit, cf, pf, 1, [], none, none⟩) D.unroll pf = .ok pub ∧
          pub.tables = none

end Flock.Bridge.Assumptions
