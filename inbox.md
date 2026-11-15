[2026-11-15T08:17Z] 95749c1a4bb2
TERMGB2 to network-accounting (lead): LAYERS done. New head of `cursor/na-orphan-roots-b51f`: `ab6ebc361d87`, one commit on top of `f930518dcd8d`, pushed only there.

1. Reproduction [ran]. I added check.py's `layers` rule to my orphan reproduction and ran it on a merge of my branch with lean's `cursor/lean-arch-lock-layers-fc8d` at `9477aac45347`. My branch's base, `architecture` and lean's head hold the same `Properties.InferenceOnly` layer. Both modules fall in the layer `Properties.InferenceOnly`, which lists no `except`. Its sealed sub-layers `.Record`, `.Desc` and `.Opening` are not admitted through its `Properties.InferenceOnly` prefix.

`Properties.InferenceOnly.PinnedRead` has six lines:
- `Properties.InferenceOnly.Opening`: our own package's sealed `Opening` layer, the reader's ports as gates;
- `Properties.Flock.Circuit.UnitInst`: the properties tree's Flock area;
- `Properties.Verity.Sha512`: the properties tree's Verity area;
- `Definitions.Hash.EmbeddingRead`: `Definitions`, the hash area;
- `Flock.ReadNet.Template`: Flock, the C-Flock verifier package;
- `Flock.Hm96`: Flock. My reader counts this one as passing, because check.py's graph (the `.olean` headers' whole closure) reaches it through the held `RecordForm`: `Properties.InferenceOnly.Record` → `Spot.Bits` → `Flock.Circuit` → `Typed` → `Tags` → `Hm96`. Top's reader counts it, which is why I print 7 lines where top prints 8.

`Properties.InferenceOnly.RecordForm` has two lines:
- `Properties.InferenceOnly.Record`: our own package's sealed `Record` layer, the record's properties;
- `InferenceOnlyCheck.Record`: the inference-only check's own Lean package (`inference_only_check`), the record check.

2. Which way each module went:
- **`RecordForm` is admitted.** `InferenceOnlyCheck.Record` is the check's own module, and the layer's other modules already read that package (`InferenceOnlyCheck.Check`, `Theta`, `Protocol` and `Count` are admitted). `Properties.InferenceOnly.Record` is the same area. The layer already admits `Properties.InferenceOnly`, and admitting a sealed sub-layer by name has precedent: `Opening` admits `.Record`, and `Remote` admits `.Desc`. So the layer gains modules, not a package.
- **`PinnedRead` is excepted.** Admitting its imports would give the whole layer edges into `Flock`, `Definitions.Hash`, `Properties.Flock` and `Properties.Verity`. No module of the layer reads any of them; the admitted edges reach only `InferenceOnlyCheck`, `Definitions.InferenceOnly`, `Definitions.Slowdown` and `Properties.InferenceOnly`.
- **The reason's form.** A layer's `except` is a bare list of module names in every lock, with no per-entry reason field. So the reason is appended to the layer's `why`, its only prose. Say if you'd rather it went elsewhere.

3. The added lines. This is one hunk in `security/Properties/lean-audit.json`, `layers` → `"Properties.InferenceOnly"`, written in check.py's writer form: its `text()` leaves these lines unchanged.

```
     "InferenceOnlyCheck.Count",
+    "InferenceOnlyCheck.Record",
+    "Properties.InferenceOnly.Record",
     "Properties.InferenceOnly"
    ],
-   "except": [],
-   "why": "a property is stated over the definitions and the code it is about, never over a proof"
+   "except": ["Properties.InferenceOnly.PinnedRead"],
+   "why": "a property is stated over the definitions and the code it is about, never over a proof; Properties.InferenceOnly.PinnedRead was rooted on Sun 15 Nov, so the rule reads its imports for the first time: its edges are decided under NET-451 after the landing, by Wed 18 Nov"
```

4. Rechecks [ran]:
- **Merges.** I merged the head pairwise with each of these, and then stacked them all one after another; every merge is clean, with no conflict in `layers`:
  - lean's head `9477aac45347`;
  - ours, `cursor/na-lock-hand-lines-b51f` at `70863c040628`;
  - LOCKPASS's inputs, as I found them by name (I found no ref for LOCKPASS's own stack): `compute-accounting-lock-hand-1de3`, `memory-accounting-lock-hand-a03c`, `proofs-connected-lock-hand-lines-5c1e`, `soundness-lock-hand-8fb2` and `proofs-placer-orphans-ef12`.

  If LOCKPASS's stack is another ref, name it and I'll merge against it.
- **Reproduction on the stack.** 0 orphans in `security/Properties` and in `security/Proofs`, and no `layers` line for `PinnedRead` or `RecordForm`. Six `layers` lines remain, exactly the six top attributes to others: memory-accounting's `Properties.Poub.Remote.InProofs` (3) and `Properties.Pous.InProofs` (2), and proofs' `Properties.Verity.LedgerPinned` → `Properties.Challenger` (1).
- **Lock.** It parses, and check.py's own `load()` loads both locks.
- **Tests.** `tests/test_lean_packages.py` and `tests/test_repository.py`: 51 passed, with the same 4 tests warning on unrelated bookkeeping.

No Lean build, Lean check or records run was launched.

