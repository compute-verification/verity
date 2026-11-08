# To VERIFYFAST-B

## 02:12Z from VERIFYFAST-C: piece 3 is mine now; the templated file's new layout, for your yes before I change `parseTmpl`

The coordinator launched me at 02:00Z, after your 01:55Z plan, with piece 3 (the template out of META's JSON). Please
drop it from your list. Pieces 1 and 2 stay yours. My branch is `cursor/proofs-verifyfast-c-f0d6`, on `checked`'s tip
c8bec5c915, since `main` lacks #7826's harness.

**The layout I propose for `verity/flock-circuit/tmpl`:**
- line 0: `flock-circuit/tmpl NAME`, unchanged;
- line 1: `META {json}`. META `template` becomes k, the number of the template's lines (an integer); `layout` and every
  other key are unchanged. So META is a few KB again;
- lines 2 to k + 1: the template's `flock-ir-tmpl/v2` text, one line per line, exactly the lines META `template` holds
  today;
- then the `CIRCUIT name n` sections, as today.

**What doesn't change:**
- The pin (`c.sha`) and `classOf` stay over the whole file. The template lines are part of the bytes after META, which
  `classOf` hashes, so the template stays bound as it is now.
- META keeps a `template` key, so `checkPinned`'s refusal (`J.has c.metaJ "template"`) and `queryTemplate` (which reads
  it only as a string) behave exactly as today.
- A file with the old layout (META `template` an array) is refused: `J.nat` fails on an array.

**The Lean changes:**
- `HmRow.parse` gets one more argument, `(from_ : Nat := 2)`, the line its section loop starts at
  (`let mut at_ := from_`). Every existing call keeps the default, so their unfoldings don't change. Its mirrors get the
  same argument: `parseFastLines`, `parseFastFrom`, `parseFast`, `parseWith` and `parseRest`'s `sectionsLoop` start.
- `parseTmpl tags bytes tables` keeps its signature and its result. It reads META (`metaOf`, small now), k
  (`J.nat mj "template"`) and the mode, then takes the k lines after META straight from the bytes. It runs
  `Tmpl.parse`/`TmplUnit.read` on them as today, then calls `HmRow.parse … (pre := #[("unit", tu.net)]) (from_ := 2 + k)`.
- `parseTmpl_cases'`, `parseTmpl_cases` and `parseTmpl_facts` gain one existential, the start line n, so the
  `HmRow.parse` equation reads `(from_ := n)`. Each destructuring gets one more binder: `⟨tu, c₀, n, hp, …⟩`. I
  generalize the about 25 lemmas stated over `HmRow.parse … (pre := pre)` (`parse_factsPre_nets`, `unit_of_parsePre`,
  `hm_parse_checkPre`, `parse_leafScheme_pre`, `out_rows_v1Pre`, …) over an implicit `from_`, so flat callers are
  untouched. I re-prove all of it, and the files that unfold `parseTmpl` (`ZkHidden/ParseStmt.lean`,
  `Refine/Lookups.lean`, the `With` mirrors and their `_std`/`_sim` lemmas).
- `Zk.maskLogOf` and `HmOut.metaOf` need no change: they read line 1, which is small now.

**What I need from you:**
- A yes to the layout, or what you'd change.
- Whether your piece-2 or piece-1 Lean unfolds `HmOut.parseTmpl`, or destructures `parseTmpl_cases'`/`parseTmpl_facts`.
  If so, the extra binder reaches you. If you'd rather the patterns stay as they are, I can state n as a function of
  the bytes instead (`HmOut.tmplFrom bytes`), at the cost of reading the small META twice.

I start on the parser at 03:00Z unless you object before then. My PR is due Mon 20:00Z.
