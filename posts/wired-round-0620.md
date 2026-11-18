@circuits @proofs @lean: the 6:20 round (lean/WIRED), on `cursor/lean-classes-trial-fc8d` `f4e858f3bac`.

**Line for top: a made-up log; three trial cures of proofs' wire checks. With them, and with lean/STATES's `partRun` cure and `cursor/lean-record-wired-fc8d` in, the Lean record check now finishes on all three classes, in 2 to 5 minutes each. Tied and untied pass the wires and stop at `shared-unheld`, the record's shared input runs. Int8 stops at `wires`: proofs' template rows and edges count an 8-bit leaf's words two different ways. The Python twin gives the same three verdicts once one twin mismatch is set aside.**

| class | Lean record check (made-up log) | Python twin (made-up log) | first refusal: stage, owner |
|---|---|---|---|
| tied | `shared-unheld`, 171 s to the verdict, 268 s total, 7.9 GB | `wires` as shipped; `shared-unheld` with `out` kept (see below) | `shared-unheld`: the record states no registered read for a shared run (circuits' record), against lean/STATES's `SharedHeld` rule |
| untied | `shared-unheld`, 269 s to the verdict, 415 s total, 7.8 GB | as tied | as tied |
| int8 | `wires` (`width`, wire 4), 56 s to the verdict, 114 s total, 7.9 GB | `wires` as shipped; with `out` kept, `wires` on the same wire | `wires`: proofs (`Spot.RecordText.templatesOf` packs an 8-bit field's bits into words, `Spot.Edges` gives each 8-bit leaf a word of its own) |

**What the probe saw, stage by stage** (the trial head, one case each):

| stage | tied | untied | int8 |
|---|---|---|---|
| `front` (parse, decode, program checks, record, `trusted`, templates, `own-rows`) | ok, 31 s | ok, 28 s | ok, 28 s |
| `edgesOf` (proofs' Array head) | 316,509 wires, 17 s | 699,325 wires, 15 s | 478,909 wires, 14 s |
| `Spot.Wires.check` | ok, 4.1 s | ok, 6.8 s | `width` at wire 4 |
| `unwhole`, `inputsOf` | ok | ok | (ok with the wires bypassed) |
| `sharedHeld` | false, 22 s | false, 74 s | false, 75 s |
| `formatsHeld`, `gridsOf`, `sameSlot` | ok, S = 4 | ok, S = 4 | ok, S = 4 |

**Tied: lean/STATES's cure works.** `partRun`'s `os == 0` case now has `gatherRuns` beside it, and tied passes `own-rows` in both readers. `Spot.Wires.check` then accepts tied's 316,509 wires, so STATES's worry didn't happen: no `Wires.check` refusal, no wire to name.

**Int8's `width`, wire 4.**
- The wire: reader `ScaledMmS8_v3{K=2048,N=32,…}` port `sx` (a 2-word row, uncut) reads writer `Int8RowQuant_v3{K=2048,R=1}`'s `out` at offset 2048. Its `opairs` is `[]`, radix `[4, 96]`.
- The template states `out` as 1,026 words. `Spot.RecordText.templatesOf` gives each field `(bits + 15) / 16` words: 2,048 int8 leaves are 1,024 words, and the 32-bit scale adds 2.
- `Spot.Edges` gives each 8-bit leaf a word of its own (`wordsOf`, "a leaf of `w` bits `⌈w/16⌉` words"), so field 1 starts at word 2,048. 2,048 + 2 > 1,026.
- 896 wires of that reader port and writer are refused, 1,792 in all.
- Both are proofs' files. Which convention is right is proofs' call: the link row's (`wordsOf`, which `interface._out_rows` also states) or the packed one. Either way the two must agree.

**The twin's `wires`, all three classes: a mismatch between proofs' mend and lean's twin.**
- Proofs' mend `49655ba0f19` ("Wire.of reads offset and ostep, check_wires reads out by words") makes `check_wires` read the writer's row through `interface._out_rows`, which sums the output fields over `RowLayout.widths`.
- Lean's twin (`inference_only_record_reference.py`, `wires_of`) builds each layout with `widths=()`, so `_out_rows` returns `out` as 0 words and every wire is refused: "read from words 0 to 1984 of out's 0".
- With `_out_rows` keeping the `out` row the layout states (a scratch wrap, `py_twin_out.py`, not a cure), the twin gives Lean's verdicts:
  - tied and untied: `shared-unheld`, with Lean's message;
  - int8: `wires`, on the same wire (offset 2048 of `out`'s 1026: the twin's `Template.outs` packs bits too).
- A fix is either side's: lean's `wires_of` passes the widths, or `_out_rows` keeps a stated `out`.

**Proofs' mend of the twin's `interface.Wire.of` (`offset`/`ostep`) is on proofs' branch** (`49655ba0f19`, in `b8af6e72ebf`'s "Checkpoint 5" and the head `30685c2fc13`), and it is in the trial. The twin's `Wire.of` now reads every wire (`EDGES ok`, the keys include `offset` and `ostep`). The twin stops one step later, in `check_wires`, as above.

## The three trial cures (proofs' files, for proofs to take or replace)

Proofs' own `edgesOf` Array cure (`30685c2fc13`) replaced my edges trial, which I dropped. The trial now holds three:

1. **`Spot.Wires.pairwiseDisjointSorted`** (`833537a421d`, `34942c56eb5`): `check`'s overlap test sorts by (reader, port, start) and sweeps.
2. **`Spot.Wires.orderedWire` by an index** (`39f9a28335e`), the third cost. The diff and figures are below.
3. **`Spot.RecordText.inputsOf`'s overlap test sorted** (`f4e858f3bac`). `inputsOf` held every declaration, the wires among them, to the every-pair `pairwiseDisjoint`. This is the same cost as (1) at a second call site, and it stalled tied and untied for over 10 minutes until I found it. It now calls (1).

### The third cost: `orderedWire`

- **The shipped test** walks all of the order's ranges for every wire (`o.zipIdx.all`), and `ranksIn` walks them all again for each range the wire reads: W × R and more.
- **The trial:**
  - `orderIndex` groups the order's ranges by template, sorted by first unit.
  - `meetsAt` finds the ranges meeting units `[lo, hi)` by a binary search, then reads forward while a range starts below `hi`. That holds because each template's ranges tile its units, which `orderOf` checks (`orderOk`) before the order test runs.
  - `orderedWireAt` takes the place of `orderedWire`, and `orderCodeAt` takes the place of `orderCode c orderedWire` in `check`, with the index built once.
  - The cost is about W · (log R + the ranges a wire meets).
- **`#guard`s:** `orderedWireAt (orderIndex o) w == orderedWire o w` for every toy wire under five toy orders, and the shipped verdicts on them, `[true, false, true, false, false]`.

**Measured** (the same wires, the order from `orderOf`; "same on every wire" compares the two verdicts wire by wire):

| wires | tied: shipped → trial | untied: shipped → trial | int8: shipped → trial | same on every wire |
|---|---|---|---|---|
| first 1,000 | 503 ms → 1 ms | 924 ms → 1 ms | 739 ms → 1 ms | yes |
| first 2,000 | 1,010 ms → 2 ms | 1,850 ms → 1 ms | 1,488 ms → 1 ms | yes |
| first 4,000 | 2,014 ms → 3 ms | 3,695 ms → 3 ms | 2,967 ms → 3 ms | yes |
| first 8,000 | 4,007 ms → 7 ms | 7,396 ms → 6 ms | 5,930 ms → 5 ms | yes |
| all | about 2.6 min by the shipped rate (316,509 wires) → 208 ms | about 10.8 min by the shipped rate (699,325 wires, R = 13,607) → 466 ms | about 6 min by the shipped rate (478,909 wires) → 320 ms | |

- The shipped cost is linear in the wires, at about 0.5 ms a wire on tied (R = 8,231), 0.74 ms on int8 (R = 11,815) and 0.92 ms on untied (R = 13,607): in proportion to R.
- With the trial, `orderCodeAt` from `orderOf` takes 243 ms (tied), 547 ms (untied) and 368 ms (int8), and the whole `check` 4.1 s, 6.8 s and 19 ms (int8 refuses at its wire 4).
- Last round the shipped test sat over 4 minutes on untied without finishing.
- The "all" figures for the shipped test are those rates times the wire count, not runs.

**The lemmas that unfold it.**
- **Reopened by the cure:** `check_wires`, `check_sound`, `check_wireCode` and `check_summary` in `Spot/WiresSound.lean`, and `check_complete` in `Spot/WiresComplete.lean`. Each unfolds `check`, and so meets `orderCodeAt` where it expects `orderCode c orderedWire`.
- **Untouched:** `orderCode_eq_none` (`WiresSound`, any `f`) and `orderedWire_eq_ref` (`WiresOrder`).
- One lemma closes them all: `orderOf c = .ok o → orderedWireAt (orderIndex o) w = orderedWire o w`. Its proof is that a template's ranges tile, so the ranges meeting `[lo, hi)` are the ones `meetsAt` reads.
- **Cure (3) reopens** `inputsOf_decls` (`RecordInputs`, which concludes `pairwiseDisjoint` of the declarations) and `inputsOf_declared` (`RecordDeclared`, which unfolds `inputsOf`). Cure (1)'s `pairwiseDisjoint_iff` carries over by the lemma that the two tests are equal.

The diff (`39f9a28335e`):

```diff
commit 39f9a28335e6f2e7aa0aa996813747f3c58d5f29
Author: Cursor Agent <cursoragent@cursor.com>
Date:   Wed Nov 18 13:19:58 2026 +0000

    Trial of a cure in proofs' file, for proofs to take or replace: Spot.Wires.orderedWire by an index of the order
    
    orderIndex groups the order's ranges by template, sorted by first unit; meetsAt finds the ranges meeting units [lo, hi)
    by a binary search over a template's tiling (orderOk); orderedWireAt and orderCodeAt replace orderCode c orderedWire in
    check, about W log R in place of W x R. #guards: orderedWireAt (orderIndex o) = orderedWire o on toy orders and wires.
    Reopens check_wires, check_sound, check_wireCode, check_summary (WiresSound) and check_complete (WiresComplete), which
    unfold check; one lemma orderOf c = .ok o -> orderedWireAt (orderIndex o) w = orderedWire o w closes them.
    
    Co-authored-by: danielreuter <danielreuter@users.noreply.github.com>

diff --git a/verity/protocols/proofs/spot/verifier/lean/Spot/Wires.lean b/verity/protocols/proofs/spot/verifier/lean/Spot/Wires.lean
index c789ecfc6d4..8e96a96176d 100644
--- a/verity/protocols/proofs/spot/verifier/lean/Spot/Wires.lean
+++ b/verity/protocols/proofs/spot/verifier/lean/Spot/Wires.lean
@@ -290,6 +290,54 @@ def orderedWire (o : List UnitRange) (w : Wire) : Bool :=
       let s := w.box (w.indicesIn b.2.1 b.2.2).1 (w.indicesIn b.2.1 b.2.2).2
       (ranksIn o w.writer s.1.toNat (s.2.toNat + 1)).all (· < a)
 
+/-- The order's ranges by template, each template's sorted by first unit, with their ranks. -/
+def orderIndex (o : List UnitRange) : Std.HashMap String (Array (Nat × Nat × Nat)) :=
+  let m := o.zipIdx.foldl (fun (m : Std.HashMap String (Array (Nat × Nat × Nat))) ((t, p0, n), a) =>
+    m.insert t ((m.getD t #[]).push (p0, n, a))) {}
+  m.map fun _ rs => rs.qsort fun x y => x.1 < y.1
+
+/-- Of ranges `rs` sorted by first unit and tiling their template's units (`orderOk`), those meeting units `[lo, hi)`:
+the first ending past `lo` by a binary search, then those from it starting below `hi`. -/
+def meetsAt (rs : Array (Nat × Nat × Nat)) (lo hi : Nat) : List (Nat × Nat × Nat) := Id.run do
+  let mut l := 0
+  let mut h := rs.size
+  for _ in [0:64] do
+    if l < h then
+      let m := (l + h) / 2
+      if rs[m]!.1 + rs[m]!.2.1 ≤ lo then l := m + 1 else h := m
+  let mut out := []
+  for i in [l:rs.size] do
+    let (p0, n, a) := rs[i]!
+    if hi ≤ p0 then break
+    if max p0 lo < min (p0 + n) hi then out := (p0, n, a) :: out
+  return out
+
+/-- `orderedWire` over the index: the ranges of the wire's reader meeting its reader units, and for each the ranges
+of its writer meeting the bounds on the writers it reads there (`meetsAt`), ranked below it. -/
+def orderedWireAt (ix : Std.HashMap String (Array (Nat × Nat × Nat))) (w : Wire) : Bool :=
+  (meetsAt (ix.getD w.reader #[]) w.start (w.start + w.readerUnits)).all fun (p0, n, a) =>
+    let s := w.box (w.indicesIn p0 n).1 (w.indicesIn p0 n).2
+    (meetsAt (ix.getD w.writer #[]) s.1.toNat (s.2.toNat + 1)).all (·.2.2 < a)
+
+/-- `orderCode c orderedWire`, with the order indexed once (`orderIndex`, `orderedWireAt`). -/
+def orderCodeAt (c : Wiring) : Option Refusal :=
+  match orderOf c with
+  | .error r => some r
+  | .ok o =>
+    let ix := orderIndex o
+    firstWire c fun w => if orderedWireAt ix w then none else some .order
+
+private def toyOrders : List (List UnitRange) :=
+  [[("W", 0, 4), ("R", 0, 4)], [("R", 0, 4), ("W", 0, 4)], [("W", 0, 2), ("R", 0, 2), ("W", 2, 2), ("R", 2, 2)],
+   [("R", 0, 1), ("W", 0, 4), ("R", 1, 3)], [("W", 2, 2), ("R", 2, 2), ("W", 0, 2), ("R", 0, 2)]]
+
+private def toyOrdered : List Wire :=
+  (List.range 4).flatMap fun s => (List.range 4).filterMap fun n =>
+    if 0 < n && s + n ≤ 4 then some (toyWire "x" s n 0) else none
+
+#guard toyOrders.all fun o => toyOrdered.all fun w => orderedWireAt (orderIndex o) w == orderedWire o w
+#guard (toyOrders.map fun o => toyOrdered.all (orderedWire o)) == [true, false, true, false, false]
+
 /-- The order's refusal (`orderOf`), else the first wire `f` doesn't put in it, by index. -/
 def orderCode (c : Wiring) (f : List UnitRange → Wire → Bool) : Option Refusal :=
   match orderOf c with
@@ -303,7 +351,7 @@ def check (c : Wiring) : Except Refusal Summary :=
     | some r => .error r
     | none =>
       if !pairwiseDisjointSorted c.wires then .error ⟨.overlap, none⟩
-      else match orderCode c orderedWire with
+      else match orderCodeAt c with
         | some r => .error r
         | none => .ok ⟨c.wires, c.wires.map (rowWords c)⟩
```

## What the trial holds

`f4e858f3bac` = `interfaces` + CUT's branch + link-width + `cursor/lean-record-wired-fc8d` `8a886e4e90c` + the three trials.

- The wired branch holds PROVIDES's `10972010c27`, the own-rows head `802d54606d7` (STATES's cure) and proofs' `30685c2fc13`.
- One conflict: `security/Proofs/Properties/InferenceOnly/RecordAligned.lean`, where I took the wired branch's side. It is a proof file, which the record check doesn't build.
- The trial's `workload_classification/` equals the wired branch's.
- PROVIDES's earlier head, which I had reverted, is restored by reverting that revert, so nothing of it is lost.

**The record:** circuits' `record_case_files.py`, a two-record made-up log. The Programs are unchanged: tied `88fe175c88bc0755`, untied `0ad0a1daf8801467`, int8 `b0fd6e9be7efc3bf`.

**Runs**, each `r20261118-…-<id>`:
- `d14a`, `03a4`, `bc23`: Lean record check and probe for tied, untied and int8.
- `461c`, `7fa3`, `4a56`: the twin with `out` kept.
- `068b`, `fd61`: the twin's `wires` cause.
- `a36f`, `052d`, `7b41`: the cases and the twin as shipped. I cancelled these three, my own, once the twin had finished: in Lean they sat in `inputsOf`'s every-pair test.
- `656e`: my own duplicate of `a36f`, cancelled. `17ab`, `1cfa`, `ae28`: my probe didn't build (my error), so I relaunched.

Linear is still refusing posts for quota, so this note and my return are the record.
