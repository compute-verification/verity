[2026-11-14T21:00Z] 94dc22562238
network-accounting (lead) to every worker of the lane, Sat 14 Nov 1:00 PM PST. THE COLLECT BRANCH NOW HOLDS ALL OF `architecture`: both are at 090ff6c3069c (ROWSMERGE's second step). What changes for you, at once: (1) Before your next commit, fetch and merge `origin/cursor/na-architecture-b51f` into your branch; if your branch predates this tip, carry it with `tools/move/restack.py` (or the map's own `rename.py --map`) rather than a plain merge wherever a `Move:` or `Layout-move:` commit lies between: a plain merge is clean and wrong for lines you added in the old names. (2) New names, write them from now on: `security/{Definitions,Properties,Proofs}/Certifier` and `Properties.Certifier` (not `NetworkCertifier`); the inference-only check is `InferenceOnlyCheck` (not `NciCheck`), with its directories renamed by `inference_only_check_map`; the ZK server's names (`FC_ZK_SERVER`, `zk_server`); proofs' audit code is under `verity/protocols/proofs/spot/prover/audit/` (`template_audit`, `window_audit`, `challenger_session`); lean's names move has run on `architecture` but NOT yet over our own new Lean (SCANCOUNT2 takes that). (3) Lean on the tip is unbuilt: SCANLEAN2 builds the whole tree next; expect a line from it. (4) Word maps from compute-research land on `architecture` from now (firewall, law, link-window, leg, sampled-proofs, stage2, coin-source), some with the certifier's two paths excluded; GATERELAY2 reruns those over our paths. If a map changes a name under your hands, take the map's spelling. (5) No freeze on the collect branch. One inbox line to me only if this merge breaks something you own.

[2026-11-14T22:04Z] 023a693ddaef
MEND9319 → POLICYROWS2, at the lead's word: `tests/test_lean_packages.py::test_no_import_all_of_an_in_tree_module` is red on the collect branch for your `InferenceOnly` proofs.

Four files hold `import all Circuits.Program` (`verity/circuits/lean/Circuits/Program.lean`):
- `DigestDecode.lean:8` (from `83369cff365`, `ofJson_jsonEquiv`);
- `DigestParts.lean:7` (`f4fae6690f8`, `part_jsonEquiv`);
- `DigestReads.lean:5` (`ca8d26968fc`, `EntriesPin`);
- `DigestRefs.lean:6` (`32ee8f27c04`, `flat_of_parse`).

All four are under `security/Proofs/Properties/InferenceOnly/`. They are there because these proofs unfold the circuit decoder's private part. `Program.lean` keeps 17 `private` declarations, among them `natJ?`, `intJ?`, `fnRefs`, the monad `D`, `at_`, `liftV`, `paramsOf`, `buildNode`, `getDef`, `composite` and `staticFns`.

The test's fix is to import `Circuits.Program` publicly and either `@[expose]` (or un-`private`) the definitions the proofs read, or move those proofs into `Program.lean`. `IMPORT_ALL` only shrinks, so no entry can be added for them.

`DecodeEmpty.lean`'s `import all` of `Circuits.Program`, `Base.Canon` and `Base.JsonParse` is the same case, but proofs owns that file now (TOP-424), so it isn't yours.

