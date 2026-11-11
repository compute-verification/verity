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

[2026-11-11T21:52Z] b84c436b9afa
SCANLEAN to the coordinator (network-accounting), NET-304, first report. Wed 11 Nov 1:52 PM PST.

Marks: [ran] a command or run I ran, [read] a file I read, [argued] my own reasoning.

**Branch.** `cursor/na-tied-need-crossed-37ad`, head `2729cb7daca3`, pushed. No pull request. Nothing marked.
It was cut from `5ed73f441056`. The crossed branch then took the `scanOf_facts` mend (`3ff11d118f93`), and I merged
that tip in (a merge commit, no rebase, no force). Its diff against `3ff11d118f93` is eight files and nothing else:
your seven and the certifier verifier's README [ran: `git diff --stat`].

Commits, in order:
- `f77c118904` the Lean: `NetworkCheck/Period.lean`, both `NetworkPeriod.lean`.
- `7c53000a04` the mirror and the difftest: `period_case.py`, `test_network_period_protocol.py`.
- `719c856f4e` `benchmarks/network_traces/tests/test_scan_zk.py`.
- `105509ca1b` the two READMEs.
- `2729cb7dac` the merge of `3ff11d118f93`.

**The seven files.** All seven are closed: each holds the change on the merged tree's own text. I touched no other
file but that README, so proofs' two are the crossed head's side whole.

**The form.** The subset form, as you ruled at 1:29 PM PST. `tiedInputWired s a` holds when the answer has no public
outputs, or no `rests_on_period_check` under them, or a list there whose every entry is a string among
`progParams s.programSlots`. Any order. A name twice passes. An empty list passes. A value that is no list (a string,
a number, null) is refused, and so is a list entry that is no string.

**The statement, as it stands in `security/Properties/NetworkPeriod.lean`:**

```lean
def RowsTiedInputWiredAt (p : Params) (c : Commitment) (e : Evidence) : Prop :=
  ∀ (s : Schedule) (sd : ByteArray) (scan : Scan), p.form = .rows s sd scan →
    ∀ (i : ℕ) (reg : Lean.Json) (x : Window), c.windows[i]? = some reg → e.windows[i]? = some x →
      tiedInputWired s x.answer = true ∧ wired p.audits s c.audits reg = true

def RowsTiedInputWiredOf (chk : Params → Commitment → Evidence → Core.Outcome Public Profile) : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    chk p c e = .accept pub prof → RowsTiedInputWiredAt p c e

def RowsTiedInputWired : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    check p c e = .accept pub prof → RowsTiedInputWiredAt p c e
```

with, in `NetworkCheck/Period.lean`:

```lean
def tiedKey : String := "rests_on_period_check"

def laneTied (s : Schedule) : Json → Bool
  | .str t => (progParams s.programSlots).contains t
  | _ => false

def tiedInputWired (s : Schedule) (a : Core.Answer) : Bool := match a.published with
  | none => true
  | some pub => match pub.getObjVal? tiedKey with
    | .ok (.arr ts) => ts.all (laneTied s)
    | .ok _ => false
    | .error _ => true
```

Its docstring says what it does not say: that the answer carries every Input the registration check let through, and
that lane j's name `prog<j>` is the Input the entry `scan/prog<j>` registers.
Proved: `Properties.NetworkPeriod.rowsTiedInputWired : RowsTiedInputWired`, with `rowsTiedInputWired_honest`,
`tiedAnswer_unwired` and `rowsTiedInputWired_nontrivial : ¬ RowsTiedInputWiredOf acceptAllRows`.

**What differs on the merged tree** (your three points).
1. The scan's entry is `scan/prog<j>`, one a lane: true [read]. `progName j` is `scan/prog{j}`, `progParams lanes` is
   `prog0 … prog<lanes−1>`, and `wired` runs over `j < s.programSlots`. So `tiedInputWired` takes the schedule. That is
   the one change to the statement, and you called it a name.
2. The written-out instance as a second copy: not needed there [read]. The merged file's nontrivial lemmas are
   `¬ …Of acceptAllRows` with the instance built inside the proof. Mine follows that, so the statement reads only
   definitions of the Properties tree.
3. `rowsTiedSampled` is gone (replaced by `RowsTiedCharged`): true [read]. `otherAnswer` is NOT gone: it is still
   defined in the proofs file, in the charged section. My section does not use it.

**Can `shared?` give a proper subset of the lanes, in the generator form?** Yes, in one case only [ran, on the Python
mirror `registration._readers`, at two lanes and `every` 2 and 4]: a verifier that expects `prog1` registered and
names `prog0` tied is left no shared node and lets `prog0` alone through. With no lane registered it is all the lanes
or none: every lane's row has the same readers, so naming one lane alone leaves a shared node and the registration is
refused (`R8-shared-input`) [ran, and it is asserted in `test_scan_zk.py`]. I did not run Lean's `tiedThrough` on it.
I did not build an accepted registration with a registered lane: a registered value's leaves are 16 or 32 bits and
the stand-in's rows are bits.

**Item 2 (not done).** PRF-504's fix is not on `main` (`75ef75838f99`) and not on the crossed head [read].
`Spot.OneStage.Public` is `{drawn, proved, bound, perTemplate}` and its `publicJson` writes no
`rests_on_period_check`. Only the registration piece and the Python write the key. So the test of the `need` on an
answer one-stage's Lean verifier computes is left out.

**Runs** (all `research run --on auto`).
- `r20261111-213424-f306`, the Lean try at `f77c118904`: done, rc 0 [ran]. `lean_changed.py --try … --siblings` on the
  three files: each `lean exit 0`. Axiom probe of the proofs file: "none of its 44 theorems uses an axiom beyond
  propext, Classical.choice and Quot.sound". Siblings: no clash; 3557 modules of `security/Proofs` unchecked, since that
  package did not build whole before the mend.
- `r20261111-214849-bb83`, the same try at the merged head `2729cb7dac`: waiting for a slot.
- `r20261111-214103-94e1`, the suite `verity/protocols/certifier/verifier/network` whole and fresh (the period difftest
  with its slow tier), at `7c53000a04`: running.
- `r20261111-214722-6768`, the suite `verity-network-traces` whole and fresh, at `105509ca1b`: waiting for a slot.
- On my VM, no Lean: the mirror reaches every new case's verdict (120 tied cases at one lane and at two, 194 small
  periods) [ran], and the four `test_scan_zk.py` tests the change touches pass, the slow rows case among them [ran].

**The README.** `verity/protocols/certifier/verifier/README.md` is 49,251 bytes with my line: 99 over the cap of
49,152. It is 49,158 on the crossed head without it (6 over) and 49,309 on `main` [ran: `wc -c`]. My line adds 93.

**Found on the crossed head, not mine, not touched** [read].
- `benchmarks/network_traces/tests/test_scan_zk_challenger.py:88` asserts `res["rests_on_period_check"] == ["prog"]`.
  `scan_zk.py` names the lanes tied there (`tied=lane_kinds(cls)`), so that slow test should fail on the merged tree
  whatever I do. The suite run will show it. Its fix is one line, `list(sz.lane_kinds(cls))`, and the docstring at
  line 44. Say if you want it on this branch.
- `chain_record.py:37` and `tests/test_chain_record.py:450-455` still say `prog`. They pass: the name is opaque there.
- `security/Properties/Spot.lean`, R8's docstring, names `RowsTiedSampled`, which is gone.

Next: read the three runs, then the full report, the note and the comment on NET-304.

