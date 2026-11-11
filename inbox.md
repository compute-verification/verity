[2026-11-11T21:51Z] 50de29e134ca
LEANMERGE to the coordinator (network-accounting), Wed 11 Nov 1:51 PM PST. **Job 5, first word: the recut is pushed, `cursor/net332-json-is-crossed-c4a7` at `6a285960e2f237ee3d747a7581eba82e3014c980`, three commits on the crossed tip `3ff11d118f93be79dd009c2c4e0d7eae2461d411`. The five runs are out, all waiting for a slot. Set your timers on the ids below.**

**1. The branch** [ran: `git ls-remote`, 1:45 PM PST; the crossed tip was still `3ff11d118f93`]
- `3edc68412eeb`: the module and the root's line. `fc329527a200`: the six comparisons and the proof's three lines. `6a285960e2f2`: the layer's admission.
- No record: `f1909ac4ab27`'s lock commit is not recut, as you ordered. No pull request.

**2. What moved** [read]

| old path (`f1909ac4ab27`) | on the crossed tree |
|---|---|
| `verity/core/service/flock/Flock/JsonIs.lean` (new) | `verity/protocols/flock/verifier/Flock/JsonIs.lean` (new) |
| `verity/core/service/flock/Flock.lean` | `verity/protocols/flock/verifier/Flock.lean` |
| `verity/core/protocols/network/lean/NetworkCheck/Period.lean` | `verity/protocols/certifier/verifier/lean/NetworkCheck/Period.lean` |
| `Security/Proofs/Properties/NetworkPeriod.lean` | `security/Proofs/Properties/NetworkPeriod.lean` |
| `verity/core/protocols/network/lean/lean-audit.json` | `verity/protocols/certifier/verifier/lean/lean-audit.json` |

- **The changed lines are the old branch's, line for line** [ran: a script over `git diff -U0` of both]: `Flock.lean` 1, `Period.lean` 14, `NetworkPeriod.lean` 6, the lock 3. So neither `main`'s three checks nor the closed hunks changed a comparison's line.
- **No seventh comparison** [read: all 115 lines of `Period.lean` with `==`, `!=`, `≠`, `.contains` or `deriving`]. The six are at 607, 649, 1009, 1062 (twice), 1218, 1239 of the crossed tip. `main`'s new lines (489 to 540) compare numbers, options of numbers and `Flock.Program` types.
- **One change of form in the new module:** its header is now a module docstring after the imports, not a comment before `module`. The Lean bar on the crossed tree finds a file without one (`module-doc`). From `@[expose] public section` on, the file is the old one byte for byte [ran].
- **The `strIs` of the eight `Sampled*` files** [read] is a theorem, `Verity.Sampled.Trees.strIs : jsonStrIs j s = true → j = .str s`, the read-back of `Spot.jsonStrIs`. It does not collide with ours: another full name, and no file opens both namespaces (the one `open Flock.J` in the tree is `NetworkCheck/LinkWindow.lean`'s).
- **Said again, because it is still true:** `Spot.jsonStrIs` (`verity/protocols/spot/verifier/lean/Spot/Json.lean:64`) has the same body as `Flock.J.strIs`, and the lean-proofs skill's ruling names it as how a string is compared. The period check's package does not require Spot's [read: its `lakefile.toml`], so it cannot import it; Flock is under both. The branch does not touch Spot or the skill.

**3. The runs, each `--on auto`, each once, at `6a285960e2f2`, launched one after the other 1:46 to 1:50 PM PST** [ran]

| run | what |
|---|---|
| `r20261111-214602-0e6f` | `lean_changed.py --try verity/protocols/flock/verifier/Flock/JsonIs.lean --siblings` |
| `r20261111-214720-5ccd` | `lean_changed.py --try verity/protocols/certifier/verifier/lean/NetworkCheck/Period.lean` |
| `r20261111-214832-b9c9` | `lean_changed.py --try security/Proofs/Properties/NetworkPeriod.lean` |
| `r20261111-214930-7fc1` | `lean_changed.py --try security/Proofs/Properties/NetworkScan.lean` |
| `r20261111-215028-a30c` | `suites.py --fresh verity/protocols/certifier/verifier/network`, in full (no `--quick`), `--max-min 180` |

- At 1:50 PM PST the four tries were waiting for a slot; the launch of the second said about 15 minutes, six ahead.
- I do not wait more than 20 minutes on any. The NET-332 comment goes up when the first two tries have verdicts.

**4. Checked on the VM** [ran]: the Lean bar's two tests of new findings pass with no warning; `tests/test_lean_packages.py` and `tests/lint/test_lean_bar.py` 40 passed (6 warnings, each a cap of the crossed tree's own); `tests/test_retired_terms.py` 23 passed, no line for our five files; the lock is still the tool's own layout of its JSON, and as parsed JSON only the layer's `imports` and `why` differ.

**5. Records, by reading, no run** [read]: two locks read the period check, `security/Properties/lean-audit.json` and `security/lean-audit.json`. In each the change moves `reads["NetworkCheck.Period"]`'s digest and the definitions of `auditCheck`, `canonicalEntry`, `holds`, `rowsWindowCheck`, `slotEntryDigest` and `windowCheck`, and adds `reads["Flock.JsonIs"]`: the eight values the old branch's records run moved. A `--records` run at this head would also report every record the crossing left stale, and hold a build slot for over an hour beside lean's Part C, so I launched none. Say if you want it.

