[2026-11-11T11:08Z] f46505d9b1db
LEANMERGE to the coordinator, Wed 11 Nov 3:08 AM PST. NET-332's comparisons' change: the branch is pushed, two tries passed, five runs wait for a slot. I end my turn on them.

**Branch** `cursor/net332-json-is-7dcf`, head `db845792d9ef55e294bff94326c8a6f9846d1184`, cut from `cursor/lean-remote-merge-fc8d` at `ebb7204a92717eb1e562289c4b9c6dd055b68169`. Two commits: `ad308cc8cf40` (the module, the root's line), `db845792d9ef` (the comparisons, the proof's three lines). Pushed there and nowhere else.

**The diff: 4 files, +37 −9** [read: `git diff --numstat` base..head]

| file | + | − | what |
|---|---|---|---|
| `verity/core/service/flock/Flock/JsonIs.lean` | 25 | 0 | new: `Flock.J.strIs`, `Spot.jsonStrIs`'s body and its docstring's reason; imports `Lean.Data.Json` only, in `Flock/CanonForm.lean`'s module form; five examples by `decide` |
| `verity/core/service/flock/Flock.lean` | 1 | 0 | `public import Flock.JsonIs`, after `Flock.JsonKeys` |
| `verity/core/protocols/network/lean/NetworkCheck/Period.lean` | 8 | 6 | the import (`public` and `meta`, as the file's six others are) and the six comparisons |
| `Security/Proofs/Properties/NetworkPeriod.lean` | 3 | 3 | **a fourth file, not in your list:** `auditCheck_facts` (lines 370, 377, 382) split by `by_cases` on `(reg.getObjValD "program" == Lean.Json.str a.program) = true`, the form the check no longer writes; it now splits on `Flock.J.strIs (reg.getObjValD "program") a.program = true`. No statement changes. Without it the proof does not close [run on the VM: a core-only replica of `auditCheck`, the old script leaves the goal unsolved, the new one closes it; notes `tools/net332-leanmerge/Replica.lean`], and with it the file compiles [run: `r20261111-105338-395f`] |

Nothing in `Flock/Canon.lean` or `Spot/Json.lean` changes; no local `strIs`; the grep finds no other declaration of that name in the package (`Trees.strIs` in `Security/Proofs/Properties/Verity/Sampled/Trees.lean:22` is a theorem in its own namespace).

**Three things I chose, each yours to overrule:**

1. **The root's line.** The package's check refuses a file no root of its lock imports (`orphan.file`, `tools/verity/lean/check.py:1437-1441`), and every other module of the package is reached from `Flock.lean`, so I listed the new one there. Its cost [computed on the VM over the import graph]: the root has 136 dependents, 58 of them not already among `NetworkCheck.Period`'s 91; `Flock.Canon` has 3441. The other way, a new entry in the Flock lock's `roots`, is a hand edit of lean's lock and I did not take it.
2. **The object example** is `.obj (.ofList [("form", Json.str "composite")])`, not `Json.mkObj […]`: `Json.mkObj` does not reduce under `decide` or `rfl` in a module that imports `Lean.Data.Json` without `import all` [run on the VM, bare `lean`]. The example strings are "composite" and "batch", what the check compares, not Spot's.
3. **The runs' shape**, below: the records run is rooted at the Flock package, and the network suite runs once, in full, in a run of its own.

**The six sites and the accept lemma's path** [read: `Period.lean` at the head; `check` is at 1340]

| line | function | reached from `check` |
|---|---|---|
| 573 | `canonicalEntry` (566) | no. `definitionDigest` (590) ← `slotEntryDigest` (614) ← `auditOf` (676) ← `paramsOf` (809), the parameters' decoder (`protocol.params`, 1401) |
| 615 | `slotEntryDigest` (596) | no, the same path |
| 947 | `windowCheck` (943) | the clock form only: `check` 1355 |
| 1000, twice | `holds` (997) | the rows form: `check` 1356 → `rowsWindowCheck` 1163, `rs.all (holds reg)` |
| 1156 | `rowsWindowCheck` (1152) | the rows form: `check` 1356 |
| 1177 | `auditCheck` (1175) | both forms: `check` 1361 → `auditChecks` 1185 |

So a built rows-form period accepted by `check` needs 1000 (both), 1156 and 1177; a clock-form one needs 947 and 1177; 573 and 615 are needed only by a lemma stated from the parameters' JSON through `paramsOf`.

**Left as they are, for lean and proofs:** the four local `named` lambdas, each `match e.getObjValD k with | .str s => s == n | _ => false`: 1045 in `tiedAudits` (1043), 1071 in `wired` (1069), 1096 in `stateEntries` (1094), 1104 in `stateTied` (1103). Line 948 compares two `canonText` texts and stays. **There is no other comparison on a JSON value in the file** [read: every line with `==`, `!=` or `≠`, and every `.contains`]: no `!=`, `.null`, `.num`, `.arr` or `.bool` one; the rest compare strings, numbers, byte arrays, `Flock.Program` types and `BucketSchedule` (`deriving BEq`, 355-357).

**Retired terms: no count moves** [run on the VM, `tests/test_retired_terms.py` and the two lint files, at the base and at the head]: `Period.lean` 190 found, 185 allowed at both; `Security/Proofs/Properties/NetworkPeriod.lean` 55 found, 50 allowed at both; no line for the new file.

**The runs, all at `db845792d9ef`, each once, `--on auto`**

| run | what | state at 3:07 AM PST |
|---|---|---|
| `r20261111-105338-b9a7` | `verity lean try` of `Flock/JsonIs.lean`, `--siblings` | **exit 0.** "lean exit 0 in 1 s"; "siblings: no built module of its package that doesn't import Flock.JsonIs declares a name it does (86 imported …); unchecked, not built since their sources or imports changed: … and 156 more"; "axiom probe: no theorems in it" |
| `r20261111-105338-395f` | `verity lean try` of `Security/Proofs/Properties/NetworkPeriod.lean` | **exit 0.** "✔ [934/945] Built Flock.JsonIs", "✔ [943/945] Built NetworkCheck.Period (2.8s)", "✔ [944/945] Built Properties.NetworkPeriod (1.5s)", "Build completed successfully (945 jobs)."; "lean exit 0 in 49 s"; "axiom probe: none of its 39 theorems uses an axiom beyond propext, Classical.choice and Quot.sound" |
| `r20261111-105338-c730` | `verity lean try` of `NetworkCheck/Period.lean` | waiting for a slot. `395f` built the file at this head already |
| `r20261111-110301-a559` | `lean_changed.py --build`: the period check's package's 13 files, `Flock.lean` and `Flock/JsonIs.lean` | waiting |
| `r20261111-110303-f81f` | `suites.py verity/core/protocols/network`, in full: the period difftest, the Python mirror unchanged | waiting |
| `r20261111-110305-9a2c` | `lean_changed.py --records verity/core/service/flock --update`, `--declared-output 'lean-audit/*'` | waiting |
| `r20261111-110307-e61c` | `suites.py --changed ebb7204a9271 --quick --skip verity/core/protocols/network`: 61 suites | waiting |

- **Why the records run is rooted at the Flock package** [read: `tools/verity/lean/lean_changed.py:231-234`, and its `requirers` called on the VM]: `requirers(verity/core/service/flock)` is the period check's package, `Security` and `Security/Properties`, so one run audits the four packages whose inputs change, each once, and no second run repeats `Security`. The loop goes on after a failed package (`lean_changed.py:1189-1242`), so `Security`'s expected failure does not stop `Security/Properties`.
- **Why the network suite has its own run:** `--quick` leaves out the period difftest's slow test (`verity/core/protocols/network/tests/test_network_period_protocol.py:66`). The suite runs once, in full; the changed-suites run leaves it out so its quick tests do not run twice on one tree. The change touches 62 suites [run on the VM: `suites.py --changed … --quick --list`], every suite that reads `verity/`.
- **A fallback in `e61c`:** the tree reaches the node as an archive, without git history (the git ship fails from my VM), so `--changed` may not find the base there. The run's command tests for the base commit and, if it is missing, runs the same 61 suites by name with `--quick`; its first line says which it did.

**The locks** [argued until `9a2c` ends; read: the six locks' `reads`]: only `Security/Properties/lean-audit.json` and `Security/lean-audit.json` pin `NetworkCheck.Period`'s definitions (227 each, the six functions among them). I expect `Security/Properties` regenerated on `reads["NetworkCheck.Period"]` (its digest and the six functions) and perhaps a new `reads["Flock.JsonIs"]`, `Security` failed on `guarantees.reads-changed` and the base's `trees.reads` line, the Flock package and the period check's package passed with their locks as they are (neither pins a changed definition; the Flock lock names no module). I commit `Security/Properties`' kept record only, byte for byte, if a value truly differs, and never `Security`'s. If the Flock or the period check's package's record moves, I tell you the values and commit nothing of it without your word.

**Not checked:**
- Nothing proves `Flock.J.strIs j s = (j == .str s)`: `BEq Json` is opaque. That the check accepts and refuses what it did rests on the difftest (`f81f`) [argued: a string compares by `String`'s `==` in both, anything else is false in both].
- The siblings check left 166 of the package's modules unchecked (not built on that tree); the root's build in `a559` and the records run are the full check.
- The move: not crossed, `restack.py` not run, as ordered.
- The `Flock.Test.*` roots, the benchmarks and the other 58 dependents of the root compile only in the records run and the suites.

**Least sure of:** how long `9a2c` and `e61c` take. The records run now audits four packages and recompiles the root's dependents (last time two packages took 1 h 35 min; its limit is 240 min); the 61 suites have a 300 min limit. Both still wait for a slot.

**A slip to know of:** the launch's own lines name machines, and my first read of the tries' logs printed them on my screen. Nothing with a name went into a note, a commit or this entry; I cut them now before reading. My NET-325 notes under `tools/net325-leanmerge/` name machines in a few lines from earlier turns; say if you want them cut.

Notes: `lanes/network-accounting/20261111T1105Z-state-net332-json-is-leanmerge.md`, synced.

