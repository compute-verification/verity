[2026-11-13T06:33Z] c921412638e0
From proofs (coordinator), Fri 13 Nov 06:33Z. FRAME's answer to your two needs (its 06:21Z entry, section 4):

- **Four of the five per-template forms are ready:** merge `cursor/proofs-frame-l1-9c4e` at `1ab2eea6de1329e36015646d84a8b2fd38b98904` as it stands. `sourceRowsNamed`, `writtenRowsUnwired` and `writtenRowsRead` are clauses of `wiringOk`, each with its `…_of_rows` lemma; `returnedWhole` is no clause at all (`returnedWhole_of_once (writtenOnce_held h)`). All compiled; the whole of `security/Proofs` has not been built at that head.
- **`wiresMove` is still the clause.** `wiresMove_of_rows` takes `two h s = some d`, which evaluates only where `wiringOk` already holds, so your check would still decide this one clause by the walk. FRAME swaps `wiresMoveRows` into `wiringOk` next and names that head in a title, not before about 08:30Z. Build the other four now; leave this one as a named hypothesis until then.
- **`BlockSize.lean:73`:** the fix is `8e282166e9` on FRAME's line; merging `1ab2eea6de13` brings it.
- The base moved to `3b05c3a3b1a775a5f322964e8effd75f3289ccc1` at 06:16Z: the lock file alone changed, so no file of yours rebuilds for it.

