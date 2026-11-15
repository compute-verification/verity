[2026-11-15T00:10Z] f4a5d46e1288
model: Claude Opus 5.5; subscriptions removed: none held

KEY, 4:11 PM PST. **The lock lists are mended on `architecture` at `c5fb633e66e` (two commits, one per lock, `b9707cd74c3` and `928a08fd71d`; unbuilt). One TOP-424 comment is posted.**

- **Inference-only lock:** network accounting had already mended `exempt`, `layers` and `runs` at `67b0389dfda`, before my fetch. One entry was left, and it is gone now: the `Spot.RecordText` layer. SPOT's lock has held it since `11048742a54`, which is proofs' word. Under this lock's roots it matched no module.
- **`RowsProtocolMain`:** nothing is missing. The lakefile declares `inference-only-rows` with it as root, and since `67b0389dfda` it is under `exempt`, like its neighbour `ProtocolMain`. `runs` holds only the tests.
- **`security/lean-audit.json`:** `Slowdown` is added to `four_names`. The slowdown lock-moves file sends 462 names, this lock's `Nci.*` guarantees among them, to `Slowdown.*`. The merge conflicted only because `reads` changed on the tip; I took the tip's file and re-applied my one line.
- **`security/Proofs/lean-audit.json`:** `Proofs.Slowdown` was already in `roots`; no edit.
- **Every other name** in `exempt`, `layers` and `runs` of the four locks exists on the tip, reading layers as module prefixes, as `check.py` does. Three names that don't exist belong to other lanes: `CoreComposition.Registered`, `CoreProtocol.Registration` and `Properties.Identity`, in the Properties lock's layer imports. A fourth, `Proofs.Properties.Verity.SampledTwo.CoinMapLaw`, is in the Proofs lock's `roots`.
- **Orphans in our area, by a static import walk, not edited:** `Properties.InferenceOnly.PinnedRead` and `Properties.InferenceOnly.RecordForm` have no root in the Properties lock. Only `security/Proofs` imports them. Both came with network accounting's collect branch, and that lock never rooted them, so our rename didn't make them stale. Two root lines would cure it; that's yours or network accounting's to say.
- **Left for the pass:** records, digests and the guarantee names (`Nci.*` to `InferenceOnly.*` or `Slowdown.*`). The audit itself wasn't run: `check.py` reads facts from `.olean` files, so even its structural part needs a build.

