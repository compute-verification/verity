module

public import Flock.Witness
public import Flock.Desc
public import Flock.KsScope
public import Flock.PortRoots
public import Flock.Schedule
public import SampledProofs.OneStageZk
public import Proofs.Flock.Bridge.Deployed
public import Proofs.Flock.Claims.DeployInputs
public import Proofs.Properties.Verity.Recursive.PartStatement

/-!
# The served v2 session's files parse and set up

`KSByDeployed`'s honest witness builds one-stage v2's live session (`Verity.Sampled.Live`) on the served registered
`--zk` session at `sha512/row/v2` rows. What it needs of that session's files is a computation on a 56 MB circuit file,
which the kernel can't run, so it is one named premise, of the witness only: no guarantee reads it. It is stated with
the verifier's own functions on the files `verify --zk` reads, each pinned by its SHA-512, in `ServedFilesParseV5`'s
form where it states the same fact (the tags, `Desc.parse`, `HmIn.parse`, `HmIn.loadPublic`).

What the witness derives rather than assumes: the proof block is built (`built_of_parse`); `scopeOkH`,
`dupName = none`, the public file as the committed files' circuit reads it (`HmOut.loadPublic`, with the same `n`) and
the roots fixing their domains (`RootFixesDomainL`) follow from `PortRoots.check` and `KsScope.check`, which
`verify --zk` runs on every session (`v5Files_of_checks`, `KsScope.check_ok`, `portRoots_load`, `loadPublic_inOut`);
`m_pts` is `mPtsOf` at the draw's proved units (`setupHidden_mPts_of_provedP`); the tags are the hidden-output ones by
`rfl`; and the statement's bytes decode to it (`decode_encode_fits`).
-/

@[expose] public section

namespace Flock.Bridge.Assumptions

open Flock

/-- **One-stage v2's statement of the served session** at the draw file `d` (`Verity.Sampled.Record.baseRaw`'s fields
at these files): the circuit, public, partition, program and registered files, `--zk`, one table, and no lookup tables,
work, stratified, closure or public-input input. -/
def servedRawV2 (cf pf pt pg rg d : ByteArray) : FlockDeploy.Raw where
  statement := _root_.SampledProofs.OneStageZk.STATEMENT
  circuit := cf
  publicFile := pf
  tables := []
  zk := true
  osCoins := false
  count := 1
  draw := some d
  partition := some pt
  program := some pg
  work := none
  workTable := none
  stratified := none
  closure := none
  registered := some rg
  publicInputs := none

/-- **The served v2 session's files parse and set up.** There are a circuit file, a public file, a partition object, a
program and a registered file with the SHA-512 digests of the served registered `--zk` session at `sha512/row/v2` rows
(`art:cd47a8fe806ef14308e22d07275d125334595b3a73d63ef8626ed762a7b9c181`: `stage/circuit.txt`, 56,327,815 bytes,
`stage/pub-32.bin`, a public file of 32 instances, `stage/partition.json`, `stage/program.json` and
`stage/registered.json`), such that, under the tags `verify --zk --statement verity/flock-circuit` sets one session up
with (`Witness.zkTags`, as `ServedFilesParseV5` reads them):

* the circuit file parses to a descriptor `D` (`Desc.parse`) whose circuit has no public input ports (`HmIn.parse` gives
  none), the public file loads over none with 32 instances (`HmIn.loadPublic`), and every port is a `sha512/row/v2` row
  (`RowsV2`);
* the files pass `verify --zk`'s rows, names and roots check (`PortRoots.check`) and its form check at one table
  (`KsScope.check`);
* at a draw `d₀` that proves 16 units, `setupHidden` sets the session's table up at the partition object's JSON
  (`FlockDeploy.jsonOf`) and the program, with no public-input input, and the table passes `--zk`'s shape check
  (`Zk.shapeOk`) and has a `fast100` schedule and a `--zk` schedule;
* at a draw file `d`, v2's statement of the session (`servedRawV2`) fits its length words (`RawFits`), parses
  (`FlockDeploy.Raw.ins`) and sets up as `verify --zk` sets it up (`FlockDeploy.Ins.setup`), to one table of 16 units.

Assumption record. It assumes a computation's result, checked by
`verity/ml/flock/tests/test_lean_served_v2_files.py` on that session's files: their SHA-512 and sizes; the first three
groups, each computed with these functions by `verity/ml/flock/tests/lean/ServedV2Files.lean` at the session record's
draw; and the last by `flock-verify verify --zk` accepting the session, whose setup `FlockDeploy.Ins.setup` states. It
asks nothing of a hash or of the prover; a change to any of these functions or to the files makes it a new premise, to
be checked again. -/
def ServedFilesParseV2 : Prop :=
  ∃ cf pf pt pg rg : ByteArray,
    hexOf (Sha512.hash cf) =
        "0c494f886ebc835e7d2a9050f1017b76e852eed07a799f7e54a01c8e306c7d787365009224f90d5ecafd4140c85a6aad6576e434e279bea2dfc1f4e9a1742ae1" ∧
      hexOf (Sha512.hash pf) =
        "95dc2441381ee37989575ebadffb97c4216ab6e9988bbe2759bdf7f5d5a63fb9d7fbb5adc9f066817e0ed65b001db31ff5947c50ae625e0ee667e4f694f5a912" ∧
      hexOf (Sha512.hash pt) =
        "88587f18568664e1448f319283bc94a4504a3fd99f71b5dfd15bb84d90997676d7c71c8754b752f81a66cf0b469d1320ab59963a4922c644beedfbcd89b69515" ∧
      hexOf (Sha512.hash pg) =
        "7c9c10d6e16498f6f3264a7ba48f4aee81b828d8481d56a5d5082c97888a5d8a214920adea30e4998c46b913333bdabcda7ad3abc6faa5022ca2d05422d6b8c4" ∧
      hexOf (Sha512.hash rg) =
        "2f9bfa9260258819e8edee20c9cd5f2d6e04acbe6f87d78baf83c3818d30e272159082749f3aa5580702807865479f0dfe3bbae0ace3745c7e9fe4762b653a3b" ∧
      ∃ D : Desc,
        Desc.parse (Witness.zkTags ⟨Tags.circuit, cf, pf, 1, [], none, none⟩) cf [] = .ok D ∧
        HmIn.parse D.unroll = .ok {} ∧
        (∃ pub : Public,
          HmIn.loadPublic (Witness.zkTags ⟨Tags.circuit, cf, pf, 1, [], none, none⟩) D.unroll {} pf = .ok pub ∧
            pub.n = 32) ∧
        RowsV2 D.unroll ∧
        PortRoots.check (Witness.zkTags ⟨Tags.circuit, cf, pf, 1, [], none, none⟩) cf pf [] = .ok () ∧
        KsScope.check (Witness.zkTags ⟨Tags.circuit, cf, pf, 1, [], none, none⟩) cf pf [] 1 none = .ok () ∧
        (∃ (ptJ d₀ : Lean.Json) (ud : Draw.UnitDraw) (st : Stmt),
          FlockDeploy.jsonOf "partition" pt = .ok ptJ ∧ Draw.UnitDraw.ofJson d₀ = .ok ud ∧
          ud.proved.size = 16 ∧
          Stmt.setupHidden (Witness.zkTags ⟨Tags.circuit, cf, pf, 1, [], none, none⟩) cf pf
            (Witness.zkCoins ⟨Tags.circuit, cf, pf, 1, [], none, none⟩) [] (some d₀) (some ptJ) (some pg) none =
              .ok st ∧
          Zk.shapeOk st = .ok () ∧ (∃ sch, fast100 st.m = some sch) ∧
          ∃ sch tPad, Zk.schedule st.m = .ok (sch, tPad)) ∧
        ∃ (d : List UInt8) (s : FlockDeploy.Ins) (t : Stmt),
          Verity.Recursive.Part.RawFits (servedRawV2 cf pf pt pg rg ⟨d.toArray⟩) ∧
          (servedRawV2 cf pf pt pg rg ⟨d.toArray⟩).ins = .ok s ∧ s.setup = .ok (t.spec, #[t]) ∧ t.n = 16

end Flock.Bridge.Assumptions
