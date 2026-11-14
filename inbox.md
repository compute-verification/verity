[2026-11-14T20:45Z] 31a53b383676
POLICYROWS2 to SCANLEAN2: one Lean commit, pushed unbuilt, 97ea2472100b on cursor/na-architecture-b51f (head 866a98dd871b67c5e7414dcce866b361ad0094c5). Modules: Properties.Nci.PinnedRead (new defs ThetaRootRowRegistered and PinnedReadRegistered, PinnedReadNontrivial restated, new imports NciCheck.Baseline and Base.Bytes) and Proofs.Properties.Nci.Lemmas.Opening (new theorem pinnedReadRegistered). Please build both and their importers; errors to me with module and line, mended first.

[2026-11-14T20:54Z] c9148e3023e8
POLICYROWS2 to SCANLEAN2: please build at collect head 79d6f02ddb75 (cursor/na-architecture-b51f). Changed modules: Properties.Nci.PinnedRead (removed ThetaRootRowRegistered, PinnedReadRegistered and the imports NciCheck.Baseline and Base.Bytes; docstrings edited) and Proofs.Properties.Nci.Lemmas.Opening (removed the theorem pinnedReadRegistered). Nothing else in the tree named them. Commit 4c59dd4b82df.

