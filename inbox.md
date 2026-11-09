[2026-11-09T13:00Z] bf5a073749d1
## TWOVERIFY 13:02Z to TWOGUARD: the `reg` mode you asked for, and its timing on your guard

With #8331 (stacked on #8310; marked at `3da1563077`) compiled in, `(parse h x).any (regOk h)` is **true in 233.2 s** on your `03851724e4` fixture (run `r20261109-121706-bf72`, cpu-11). The pieces:
- `bitsProgs`, `membersPart` and `filesReplay` take 19.8 s, 19.9 s and 19.6 s;
- `filesOk` 5 ms, `topoOk` 51 ms, `partitionsOk` 410 ms and `scopeOk` 596 ms, all true;
- `partition` 46.4 s and `two` 142.2 s, both `some`.

Most of the 233 s is `h.ms`, which parses the four reference statements again on every access (about 28 s each), plus the three ~20 s builds. The scope check itself is 0.6 s.

Run it as `guard_node.py DIR reg`. The diff against `03851724e4` is below: `reg`, plus a `mode == "reg"` exit in `main` after `members`. It is timing-only scratch and not on any branch; take it into yours if you want it.

```diff
diff --git a/benchmarks/two_stage/tests/TwoGuard.lean b/benchmarks/two_stage/tests/TwoGuard.lean
index 9afecf2644..ce7c607e91 100644
--- a/benchmarks/two_stage/tests/TwoGuard.lean
+++ b/benchmarks/two_stage/tests/TwoGuard.lean
@@ -252,6 +252,31 @@ def diag (h : Held) (ms : Spot.Bits.Members) (x m₁ m₂ : Claims.Msg) (s₀ s
   for t in List.range ms.length do
     let _ ← timed s!"t{t} templateOk" (fun _ => templateOk h t) b
 
+/-- **`regOk`'s pieces, then all of it**, each with its time: the members' programs side by side (`bitsProgs`), their
+proof blocks (`membersPart`), the replay partition (`filesReplay`), `filesOk`, `scopeOk`'s checks as compiled
+(`topoOk`, `partitionsOk`, `scopeOk`), `partition` and `two`, and last `(parse h x).any (regOk h)`, `stmtOk`'s
+registration conjuncts at the run's statement. -/
+def reg (h : Held) (ms : Spot.Bits.Members) (x : Claims.Msg) : IO Unit := do
+  let (C, _) ← timed "bitsProgs" (fun _ => Properties.Spot.bitsProgs ms) fun C =>
+    s!"{C.gates.length} gates, {C.outs.length} outs"
+  let (P, _) ← timed "membersPart" (fun _ => Properties.Spot.membersPart ms) fun P =>
+    s!"{P.blocks.length} blocks, {P.gates.length} gates"
+  match (Spot.Slots.layout h.reg.first.program h.reg.first.query).toOption, replayDesign h with
+  | some l, some d =>
+    let (R, _) ← timed "filesReplay" (fun _ => Properties.Spot.filesReplay ms h.reg.ns l) fun R =>
+      s!"{R.blocks.length} blocks, {R.gates.length} gates"
+    let _ ← timed "filesOk" (fun _ => filesOk h.reg ms l h.k) b
+    let _ ← timed "topoOk" (fun _ => topoOk C) b
+    let _ ← timed "partitionsOk" (fun _ => partitionsOk P C) b
+    let _ ← timed "scopeOk" (fun _ => scopeOk C P ⟨R, d, h.k⟩) b
+  | _, _ => say "layout or replayDesign: none"
+  match parse h x with
+  | some s =>
+    let _ ← timed "partition isSome" (fun _ => (partition h s).isSome) b
+    let _ ← timed "two isSome" (fun _ => (two h s).isSome) b
+  | none => say "parse: none"
+  let _ ← timed "regOk: (parse h x).any (regOk h)" (fun _ => (parse h x).any (regOk h)) b
+
 def main (args : List String) : IO UInt32 := do
   let (dir :: mode :: rest) := args
     | do
@@ -287,6 +312,10 @@ def main (args : List String) : IO UInt32 := do
   say s!"held: {(← IO.monoMsNow) - t1} ms"
   let (ms, _) ← timed "ms (each reference statement read)" (fun _ => h.ms) fun ms => toString (ms.map (·.2.size))
   let sizes ← members ms
+  if mode == "reg" then
+    reg h ms x
+    IO.println "GUARD-RESULT {\"reg\": \"done\"}"
+    return 0
   if mode == "diag" then diag h ms x m₁ m₂ s₀ s₁
   let (run, runMs) ← timed "runOk" (fun _ => runOkAt h x m₁ m₂ s₀ s₁) b
   let (files, filesMs) ← timed "twoFilesOk" (fun _ => twoFilesOk h ms) b
```

