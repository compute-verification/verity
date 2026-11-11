[2026-11-11T12:47Z] 7ab20a5d8d29
LEANMERGE to the network-accounting coordinator, Wed 11 Nov, 4:47 AM PST. Job 3: the profile's read-back without
evaluating `canon`'s output.

Marks: [vm] elaborated or run by me on my VM with bare `lean` v4.34.0; [read] file and line at a commit; [run] a run
id and its exit; [argued].

## Outcome

**Proved, with the sort left as it is.** `Core.Integrity.ofJson P.toJson = .ok P` holds at the witness's profile, and
so does the step above it, `answered what reg a = .ok (P, pub)`. Neither needs proofs' sort change. Three things you
should know before you write to lean:

1. **#8942 gained its third commit** (`131c1e705892`, 12:14Z, the refusal of strings of 2^32 bytes or more). I did the
   work at `7614ddc2fd69`, then redid all of it at `131c1e705892`: the same results, nothing in the proof changed. [vm]
2. **The reader at RATIO's head is not the reader on the witness's base.** At `131c1e705892` (main's text)
   `CoreProtocol/Profile.lean` has no work design: the type is `Draw` (`verity/protocols/interface/CoreProtocol/Profile.lean:33`),
   nothing reads `Flock.Draw.workK`, so there was nothing to stub. [read] The work design (`SamplingDesign`, its `work`
   case, `import Flock.Draw`) is in the base's text, `verity/core/protocols/interface/CoreProtocol/Profile.lean:47` at
   `ebb7204a9271`, last changed by `4a00de454f`, which is not an ancestor of RATIO's head. [read, vm: git] So I proved
   it on both. On the base's reader the proof is the same with `Draw` renamed `SamplingDesign` in three places. [vm]
3. **On the base the proof needs #8942's `Ratio.lean` and nothing else of it.** The base's `Ratio.lean` is main's text
   before the rewrite [vm: diff from `namespace` on]. With it exactly one step is refused: `Ratio.ofString "1/100" c =
   .ok ⟨1, 100⟩ := rfl`. [vm]

## 1. The lemma, as elaborated [vm]

~~~text
prof_reads : Integrity.ofJson prof.toJson = Except.ok prof
'ReadBack.prof_reads' depends on axioms: [propext, Classical.choice, Quot.sound]
'ReadBack.prof_reads_toOption' depends on axioms: [propext, Classical.choice, Quot.sound]
~~~

`prof` is the draft's profile: `programs := #["p"]`, `prescribed := #[]`, `linked := true`, `covered := #[]`,
`levels := #[⟨"unit", none, 2, some (.subset 1)⟩]`, `delta := ⟨1, 100⟩`, `drawn := #[("unit", 1)]`. No `native_decide`,
no `sorry`, no axiom declared.

The form is `= .ok P`. It is the stronger one and the one a proof of `answered` rewrites with: `answered` matches
`Core.Integrity.ofJson prof` against `.ok i` (item 4 uses it so). `(…).toOption = some P` is a one-line corollary,
`prof_reads_toOption`.

Where it holds [vm, each with the same three axioms]:

- the core-only copy at `131c1e705892`, not a module (`ReadBack.lean`), 3 s;
- **in module form** at `131c1e705892` (`ReadBackModule.lean`): the five files of the route compiled as modules by bare
  `lean -o`, then the proof as a `module` with the lemma public, then a module that only `import`s it and reads the
  lemma;
- on the base's reader `ebb7204a9271` with #8942's `Ratio.lean`, not a module (`ReadBackBase.lean`, `Flock.Draw.workK`
  copied from `Flock/Draw.lean:118`, not stubbed) and in module form (`Flock/Draw.lean` compiled too);
- with a 128-character program name in place of `"p"`: same proof, 7 s.

## 2. The proof's shape, and its length

1. `exactly_self`: `exactly toJson (toJson a) ctx a = .ok a` from `∃ t, Flock.canon (toJson a) = .ok t`. Both sides of
   `exactly`'s `!=` are the one term `t`, so `bne_self_eq_false` closes it and no canonical text is ever compared.
2. That the text exists is `(Flock.canon x.toJson).isOk = true := by decide +kernel`: the kernel computes only that
   `canon` returns. Its sort sits under `pure`, so it is never forced.
3. One lemma per reader, bottom up, for every `ctx`: `draw_reads`, `level_reads`, `prof_reads`. Each unfolds its
   reader and rewrites every field read by a fact proved by evaluation: `rfl` where the value is JSON, `decide +kernel`
   through `.toOption` where the value's type decides equality. What is left is `exactly toJson (toJson a) ctx a`.
4. `Integrity.ofJson`'s two loops go by core's `List.forIn_toArray`, `List.forIn_cons`, `List.forIn_nil`; `nullOr` by
   an 8-line lemma, `nullOr_some`.

Eight lemmas, 53 lines of Lean (66 with docstrings and blank lines); `prof_reads` itself is 16.

## 3. The two side probes

(a) **The empty sort reduces.** `(#[] : Array String).qsort (· < ·) = #[] := by decide +kernel` passes. [vm] In a
module it needs `import all Init.Data.Array.QSort.Basic`: without that line the read of `prescribed` is refused. [vm]
29 files of the tree at the base have the line. [read: grep] On one element the same statement is refused. [vm]

(b) **One name in `covered`: it sticks at two evaluations, and one small lemma cures it.** The two are
`strs profC.toJson "covered" "profile"` (reading what `sorted` wrote) and `(Flock.canon profC.toJson).isOk` (`canon`
walks the sorted array); `decide` says of each that its instance "did not reduce to `isTrue` or `isFalse`". Every other
step is as before. [vm, `ReadBackStuck.lean`] The cure, short of replacing the sort: `qsort_one : #[x].qsort lt = #[x]`,
4 lines, proved as a proposition from `qsort`'s equation; with it `profC.toJson` is rewritten to a literal that holds
no `qsort`, the evaluations are made on that literal, and `profC_reads : Integrity.ofJson profC.toJson = .ok profC` is
proved with the same three axioms. [vm, `ReadBackCovered.lean`] In a proof module `qsort_one` needs
`open private Array.qsort.sort from Init.Data.Array.QSort.Basic`, which 12 files under `Security/` have
(`Security/Proofs/Properties/Accounting/Remote/Sort.lean:17` at the base) [read]; the scratch file has a 4-line `elab`
in its place. Two names or more: not built. It would need `qsort` of a sorted literal unfolded step by step, or a
lemma that `qsort` fixes a sorted array; `Sort.lean:78` has `qsort_perm`, and I did not search for more. [argued]

## 4. `answered`: cheap, and proved [vm]

~~~text
answered_reads : ∀ (what : String) (reg pub : Json),
  (∃ t, Flock.canon reg = Except.ok t) → answered what reg (ans reg pub prof) = Except.ok (prof, pub)
'ReadBack.answered_reads' depends on axioms: [propext, Classical.choice, Quot.sound]
'ReadBack.answered_reads_reg0' depends on axioms: [propext, Classical.choice, Quot.sound]
~~~

`ans reg pub P` is `{ protocol := oneStage, verdict := .accept, reason := none, digest :=
(Flock.WindowRecord.digest reg).toOption, published := some pub, profile := some P.toJson }`. It holds for every
`reg` that has a canonical text, every `pub` and every `what`; at a `reg0` built by `Json.mkObj` the hypothesis is
`ok_of_isOk (by decide +kernel)`. Seven lines of proof (`ReadBackAnswered.lean`).

- The pieces are copied byte for byte from `131c1e705892`: `Core.Verdict` and `Core.Answer`
  (`CoreProtocol/Protocol.lean:29-35, 73-81`), `Flock/WindowRecord.lean:21-39`, and `oneStage`, `Found`, `refuse`,
  `answered` (`NetworkCheck/Period.lean:240, 854-872`). `answered`'s text is the same at my branch's
  `Period.lean:924-934`. [vm: diff]
- One stand-in: `Flock.Sha512.hash` is declared `opaque`. The record's digest is never computed (the answer's digest
  and the record's are one term), so the lemma holds for every hash.
- One thing I had not foreseen: the digest test is `!=` on `Option ByteArray`, and core has no `ReflBEq ByteArray`
  (`#synth` fails [vm]), so `x != x` does not close alone. The proofs' tree has the instance, three lines, local to
  each file that needs it (`Security/Proofs/Properties/NetworkCertifier/Check/LinkWindow.lean:77-80` at the base)
  [read]; I copied it.
- Not done: this step in module form. `NetworkCheck/Period.lean` imports the network check, which is a package build.
  What it needs is small [read at my head `db845792d9ef`]: `answered`, `oneStage`, `Found` and `refuse` are public and
  exposed (`Period.lean:248`, `@[expose] public section`), so no `import all NetworkCheck.Period`.

## 5. What the real proof module needs

**Six `import all` lines, each needed** [vm: module form, each line removed in turn, `readback-parts/min.py`]:

~~~lean
import all CoreProtocol.Profile          -- `exactly`, `nullOr`, `strs` are private and the proof names them
import all Lean.Data.Json.Basic          -- the field tests and reads (`Flock.J.has` is the first refused without it)
import all Init.Data.Ord.String          -- the same: an object's keys are compared as strings
import all Init.Data.Array.DecidableEq   -- that a canonical text exists (`(Flock.canon …).isOk`)
import all Init.Data.Array.QSort.Basic   -- reading `prescribed` and `covered`: the empty `sorted`
import all Init.Data.Repr                -- δ's string, and with it the profile's canonical text
~~~

`Security/Proofs/Properties/NetworkPeriod.lean:4-7` at the base already has four `import all` lines
(`Lean.Data.Json.Basic`, `Init.Data.Ord.String`, `Init.Data.Array.Basic`, `Init.Data.Repr`) [read], so in that file
three are new: `CoreProtocol.Profile`, `Init.Data.Array.DecidableEq`, `Init.Data.Array.QSort.Basic`. Not needed [vm]:
`CoreProtocol.Ratio`, `Flock.Canon`, `Init.Data.Array.Basic`, `Init.Data.String.Defs`.

Private on the route [read at `131c1e705892`; the base's lines in brackets]: in `CoreProtocol/Profile.lean`, `sorted`
62 [106], `exactly` 83 [131], `strs` 87 [135], `nullOr` 90 [138]; in `CoreProtocol/Ratio.lean`, `natOf` 37 and
`pieces` 47; in `Flock/Canon.lean`, `hex4` 22. Only the first file's need the line: `natOf` and `pieces` evaluate
without it, and `hex4` is never forced.

Three consequences in a module:

- A lemma that names `exactly`, `nullOr` or `strs` can't be public: in a public section the names are unknown even
  with the `import all` line. I kept those lemmas outside the public section; `prof_reads` is public. [vm]
- `NetworkCheck.Period` has a private `exactly` of its own (`Period.lean:413` at my head): write `Core.exactly`.
  [read]
- Importing a code package's module with `import all` from the proofs' package has one precedent,
  `Security/Proofs/Flock/Claims/Sampling/Select/Second.lean` (`import all Spot.Draw`), and Lake's check against it
  is commented out in this toolchain (`src/lean/lake/Lake/Build/Module.lean:418-422`). [read]

**Nothing in the proof relied on something only the scratch copy has**, by the module-form check. What that check is
not: it is bare `lean` on the route's files and the proof, without lake, the package or Mathlib in scope.

## 6. The runs

- `r20261111-110305-9a2c` (records): **still running** at 4:46 AM PST, holding the Lean slot since 11:25Z. Left as it
  is. Nothing committed: the branch's head is still `db845792d9ef`.
- `r20261111-110307-e61c` (suites): [run] exit 0, ended 12:33:32Z. "61 suites: 61 passed (0 cached), 0 failed; 144.5
  min of test time". My sum of its 61 lines: 33,535 tests passed, 637 skipped. The network suite is skipped there as
  launched; it is `f81f`'s.
- `r20261111-110303-f81f` (network suite, in full): [run] exit 0. "passed verity/core/protocols/network 89 passed in
  530 s"; "1 suites: 1 passed (0 cached), 0 failed".
- `r20261111-110301-a559` (package build): [run] exit 0. `lean-build: {"built":
  {"verity/core/protocols/network/lean": 0, "verity/core/service/flock": 0}, "failed": {}, "skipped": {}}`.
- `r20261111-105338-c730` (try of `Period.lean`): [run] exit 0. "lean exit 0 in 4 s"; "axiom probe: no theorems in
  it".
- Machine names: cut from 21 of my note files, each replaced by `NODE`. The notes repository's history still has
  them; I can't rewrite it. Twice this turn a run's status record printed an instance id on my screen (its `stream`
  and `updated_stream` fields); it is in no file, and `tools/net332-leanmerge/jshow.py` now drops those fields.

## What I am least sure of, and where I did not look

- The package build. No lake, no Mathlib, no node run, by your rule. If an instance Mathlib adds changed how a
  `decide +kernel` statement elaborates, only the build would show it. [argued: `simp only`, `rfl` and
  `decide +kernel` read no simp set]
- The witness's literals. Every evaluation is at the draft's values; other names or another δ are the same proof with
  other literals (the 128-character name is the one I tried).
- `answered` at a real registration record. `reg0` has three fields; `(Flock.canon reg).isOk` by `decide +kernel` is
  one walk of the record, and I did not time a full one.
- (b) beyond one name.

## Files

Notes, `lanes/network-accounting/tools/net325-leanmerge/net332/`: `ReadBack.lean`, `ReadBackModule.lean`,
`ReadBackBase.lean`, `ReadBackCovered.lean`, `ReadBackAnswered.lean`, `ReadBackStuck.lean` (meant to fail, four
times), `readback-outputs.txt` (what each printed), `readback-parts/` (the generators and how the files are made).
Each file's header names the commit of every piece. Nothing pushed, nothing posted.

Waiting on `r20261111-110305-9a2c` only.

