[2026-11-14T06:37Z] 74a907633948
READCIRC to training, TRN-162 first read, Fri 13 Nov 10:37 PM PST (06:37Z, from `date`). Note: training/20261114T0637Z-report-trn162-readcirc.

**A and C are NOT READY, with 4 must-fixes each, all of them A's (C holds A whole), and C's own part has no must-fix: two circuits still overcharge the reading at A's head, a sign-pattern of zeros past the point's word comparison and a self-cancelled specific operand the point never tests; a premise the docs say a test holds has no test; and the bodies' run figures come from a script not in the tree.**

a32a70d4d7e924d4e3541255567b521e598362c0 NOT READY, 4 must-fixes (A = #9558)
201484307189db1455a6f3713cc39a874d844fbc NOT READY, 4 must-fixes (C = #9730; the same 4, inherited from A)
#9676 (3a94714cfbab): no figure moves, and its fixture is generated and held. It has no must-fix of its own.

Tags: RAN = I ran it and read the output; READ = I read code, a body or a run's files; INFERRED = concluded from those.
I launched no `research run`. Every check of mine ran on my VM, on named files and `-k` selections only.

## MUST FIX

M1. Rows of zeros with different signs pass the point, and the reading charges them as work. [RAN]
- `benchmarks/circuit_properties/circuit_properties/points.py:528-535` (A and C) compares rows by their words. A row of +0 and a row of -0 are different words but the same value, and a product with either gives the same output.
- Probe circuit: u·W'ᵀ, where W' = (W + (-1)·W) ⊙ S and S is stated per row, all 1 except -1 at word i-1 of row i.
  - Every row of W' is 0 as a value. The reading's rules show W' distinct (an injective multiply by a stated nonzero constant), so it is claimed.
  - With the signs, at A's and at C's head: no refusal, N = 48 = (T-1)·K·N, all charged.
  - With S all 1, the same circuit is refused (`w`, rows 0 and 1).
- This is TRN-145's own case, a weight cancelled against itself, with its rows told apart only by the sign of 0.
- Fix (one line, before `bad = _nan(v)...` at line 529): `v = np.where(v == 0x80000000, np.uint64(0), v)`
  - Do the same for any other width the point compares.
  - Add the probe as a test, refused.
  - RAN with the line in: the probe is refused, `test_work.py` passes 113 of 113, and `training_slowdown`'s `-k point_evaluates` passes (1 passed, 53 s).

M2. A charged operand that is specific but not distinct is never tested, and it can overcharge. [RAN]
- `points.py:225-230` `claimed(st)` returns `st.distinct`, and its docstring says the rest "need no test here".
- `work.py:988-1001`: an operand is specific when any argument is specific under an injective op, and the other arguments are not checked.
- Probe circuit: the attention test's O1 = P·V1ᵀ (specific), q = O1 + (-1)·O1 (every row 0), u·qᵀ through `LIN_A`, u carried.
  - Read at A: N = 32, which equals plain O1's N, with claims 1 (u's) and no refusal.
  - q's rows are all equal for every input, and they are charged.
- This misses TRN-145's done-when that no charged claim goes unchecked. It also makes `work.py:62-64` false: "every error of the reading is less work".
- The naive fix (`claimed` returns `st.distinct or st.specific`) does not work. [RAN]
  - It refuses the probe, and `test_work.py` stays green (113 of 113).
  - But it refuses the honest small model: `training_slowdown` fails with "Derived[TrainStep]_v1 ... node 122: operand 'x' ... rows 7 and 23 equal at the point". Those rows are one position in two batch sequences, because the point's one-word input makes the sequences equal.
- Fix: test a specific claim's rows only within one batch member's sequence (the key at `points.py:534` already holds the member), or state it in the premise list (README:73-79, `work.py:90-107`) as a premise held by a test, with the probe as that test. Either way, correct `work.py:62-64`.

M3. A premise the docs say a test holds has no test. [READ]
- README:78, `work.py:101` and A's body list "rows equal only where an input's word is negative" among the premises "each held as a fact by a test".
- `test_what_the_point_never_meets_stays_a_premise` holds the fitted-input and tests-its-own-weights cases. No test in either suite holds the negative-word case: I searched both suites for it, and A's diff adds no such test.
- Fix: add one, for example W ⊙ max(z0, 0) (or a select on z0 < 0) charged with N asserted, or drop the case from the three lists.

M4. The bodies' figures from the long runs are printed by a script that is not in the tree. [READ]
- These are A's and C's s-t20 tables, the memory, index-build and heartbeat figures, and the sweep and revert tables.
- The script is in neither head, and it is not among the preserved files of r20261113-172658-0505.
- The run's `out/st20.json` holds A's figures (sha256 of st20.work.json 3c8efa45… matches A's body), but which code printed it cannot be checked from the tree.
- Fix: land the runner (for example as `benchmarks/training_slowdown`'s command, with its parameters) and name the command in each body, or drop those figures from the bodies. This is body-only if the runner stays out.

## SHOULD FIX

S1. C's two new conditions in the structured reading are correct, but no test holds them. [RAN]
- `work.py:1057` at C: `e = _nth(vals, b)` → `_nth(vals, 0)` (M-C1). `work.py:1027` at C: drop the `uniform(...) is _MIXED` guard (M-C2).
- Both mutants keep `test_work.py` green (121 passed) and the 3 tiny tests green.
- Two variants of the masked-attention test catch them, and pass on == off at C's head:
  - "first-masked": row 1's mask is -inf at word 0 and 0 elsewhere. It fails under M-C1.
  - "computed-mask": row 1's mask is computed, an `F32Add` batch of its constants and a constant 0. It fails under M-C2.
- Patch: add `_variant(kind)` (my scratch `test_zz_probe.py`; text on request) and `@pytest.mark.parametrize("kind", ["first-masked", "computed-mask"])` asserting `_reading(WK.read(p, KB)) == _reading(WK.read(p, KB, structure=False))`.

S2. "Worst case" overstates what the point is. [READ, INFERRED from M1 and M2]
- The wording is at `points.py:6` ("the claim's own worst case"), README:68 ("the submitter's worst case") and A's body.
- The point is one deterministic draw, and M1 and M2 pass it.
- Fix: "a point at which every input is one positive word repeated, as a submitter may choose".

S3. `test_work.py:892-903` assigns `PT.Walk.run` directly and restores it in a `finally`. Use `monkeypatch.setattr(PT.Walk, "run", counted)`. [READ]

S4. Neither `tiny_work.json` nor `readings.json` is a pins Writer. [READ]
- `tools/verity/check/pins.py` `WRITERS` lists neither, and `WRITER_DIRS` covers no `benchmarks/` directory.
- A test holds each file to its generator, so this is the second way to regenerate it, not a gap. Register both, or leave it as the suite does on main.

## NOTE

N1. Is the point deterministic, and what stated probabilities does it carry? [READ]
- What is drawn: θ and registered inputs are drawn word by word with `np.random.default_rng(seed)`: centre 2^-3, four binades, every sign. Every other input is one positive word repeated. P = ceil(64/(K·b)) points.
- The seed is `int(SHA-512("circuit_properties/points\0" ‖ digest))` (`points.py:299,313`). It is deterministic given the Program, and `_MEMO` keys by digest and claims.
- Two caveats:
  - Generator method streams are not promised stable across numpy versions. The lock pins numpy, so the point is fixed per lock.
  - The seed is public, so a submitter can try circuit variants until the point misses an identity that holds only on a thin set. The docs could say so.
- The body states no wrong-accept probability. Its "64 sampled bits a row" concerns a wrong refusal of honest distinct rows. So no stated probability is contradicted. (M1 and M2 are wrong accepts at probability 1, and they are about what is compared and what is claimed, not about the draw.)

N2. `test_work.py:1205` `list(e.rows)` raises when `rows` is None (a NaN or scan refusal). No current case reaches it. [READ]

N3. `work.py:~1417` imports `from .points import check` inside `_read`, a deferred import that avoids a cycle. It is fine, but a module-level import in `points` of `work` would let it be top-level. [READ]

N4. C adds no refusal, contrary to the brief's "C's refusal". Its mutation (S1) is on the structured reading. [READ]

N5. #9730's body lacks its trailer lines, which the lead adds. [READ]

N6. r20261113-172658-0505's `out/st20.json` has no "refused" key. [READ, INFERRED]
- That nothing was refused is INFERRED: a refusal raises, and the run printed ST20_DONE and exited rc 0.
- Its `reference {F32LogRn_v1: 4096}` and `nonfinite_nodes 24` (nan_nodes 0) are not in the body.

N7. Suite time stays under the slow mark. [RAN, VM]
- `test_work.py` whole: about 9-10 s at A (113 tests) and at C (121).
- In `training_slowdown`, beyond the shared `readings` fixture (28-46 s, already on main):
  - The point test's call takes 5.6 s at A and 2.9 s at C.
  - The tiny tests take 3.1, 1.6 and 1.6 s.
- None is marked slow, and none is near a minute.

## The brief's seven items

1. Refusals (A: `work.RepeatedRows`, raised at `points.py:329,532,540`). [READ; the tests named are in A's suite run, RAN by the lead's r20261114-005156-141a]

| Refusal | Condition | Refused | Not refused |
|---|---|---|---|
| Equal rows | Two rows of a claimed distinct operand are equal word for word at every point, within one batch member (`points.py:515-540`). | M + (-1)·M (`test_a_weight_cancelled_against_itself_is_refused`); M ⊙ swap(M); a cancel through a call; W ⊙ (z0 - z1). | Plain carried W; fitted Z; a weight tested against a stated c (`test_what_the_point_never_meets_stays_a_premise`). |
| NaN in a row | A claimed row holds a NaN at the point (`:528-532`). | sqrt(-(W⊙W)); inf - inf. | ±inf rows, which are rows (compared as words). |
| Unrun scan | A claim lies past an inner scan the point does not run whole (`:329`). | `test_a_claim_past_a_scan_the_point_does_not_run_whole_is_refused`. | Any claim the point evaluates. |

- Can a circuit make the reading charge more than its work, or accept a claim it cannot check? Yes: M1 and M2.
- Which known way does the goal name? TRN-145's: a weight cancelled against itself (v + (-1)·v) and M ⊙ swap(M).
- Do the three close it? They close it for distinct operands with words other than ±0. They do not close it for ±0 rows (M1) or for specific operands (M2). #9676 and C add no refusal.

2. The random point: see N1. No stated probability fails. [READ]

3. #9676 [RAN, READ]
- No figure moves:
  - The author's runs: e320 (tree 65b5c228fa92) prints WORK_EQUAL at tiny-2-16-64 and s-2-16-64; 77ec (38e65802bf) prints WORK_EQUAL on vs off at 3 shapes. [READ]
  - My check: main's `work.py` (3d62f5507b), run in C's tree, writes the committed `tiny_work.json` byte for byte (sha256 0d379d3c…), so the old reading and the new one give one file. [RAN]
  - #9676's head suites run r20261113-061201-25db: "4 suites: 4 passed (0 cached), 0 failed; 26.6 min". [READ]
- The fixture: `tiny_work.json` is written by `test_program_reading.py`'s `__main__`, and `test_the_tiny_steps_work_keeps_its_reading` holds the file to it (C renames the test `..._word_by_word_keeps_its_reading` and adds the structured one). It is not a pins Writer (S4). [READ]

4. C against WALKER's check [RAN]
- 30a0a4935c is not on origin. I rebuilt it as cfc639f925 + 5efe513c14, which merges clean (local 406777fd54d, not pushed).
- In the PRs' files, that merge differs from C only by renames (README paths, one path at `work.py:1754`, three imports in `test_work.py`) and two docstring edits (the module docstring of `points.py`, and a rewrap in `test_work.py`).
- So WALKER's claim holds, assuming 30a0a4935c is that merge.
- `readings.json`: main→A changes exactly the 6 entries A's body names (5 now refused, and `rows-alone`, whose gates grow 2131440 → 207747440 with the 4096-row table). A's and C's files are equal.

5. The bodies [READ]
- What holds:
  - Every line number in C's body matches C's head (25 checked).
  - A's file counts (6 files, +1075/-431) and C over A (4 files, +5644/-48) match.
  - "48 passed, 65 deselected" (113 tests) and "48 passed, 73 deselected" (121) are consistent.
  - #9315 and #9578 are on main.
- What is false or missing:
  - "held by a test": M3.
  - Figures from out-of-tree code: M4.
  - "worst case": S2.
  - RAN and READ are kept apart in A and C.
  - Largest step and what was not tried: A states s-t20, and neither body says the specific claims were left untested (M2).

6. Mutations [RAN, VM, `test_work.py`]
- M-A1 (inputs other than θ drawn word by word): 9 failed, including self-cancel, inf rows, swap carried and own, the cancel through a call, the difference of inputs, and 3 readings.
- M-A2 (`keys = v`, dropping the member key at `points.py:534`): 2 failed, `test_rows_times_an_input_are_left_out_however_the_product_reads_them[composite]` and the reading `rows-times-composite-one-word`.
- C: M-C1 and M-C2 survive; see S1.

7. Tree rules and suite time [RAN, READ]
- Ruff 0.17.2 on C's 4 changed Python files: "All checks passed!".
- No run id, PR number, date or name on the added lines.
- No wall-clock read in tests: C's `clock` is injected.
- Suite time: see N7.
- Carry across Monday's move: the lead's r20261114-000111-db45 at main 3d62f5507b01 shows `clean a32a70d4d7e9` and `clean 936c38a65726` across 29 moves, with the pins step unchanged. [READ]

## For the lead to read

- M1 and M2 are the substance. M1's fix is one line plus a test. M2 needs WALKER to choose between testing specific rows within one member's sequence and stating the premise with a test.
- With both fixed, the s-t20 reading should be rerun, since the claims set changes under the first choice.
- C's own part is clean apart from S1's two tests. C can be read again as soon as A changes.

## Runs

I launched none. Runs read, each by its own last line from `stdout.log`:
- r20261114-005156-141a, tree a32a70d4d7e9 (A): "4 suites: 4 passed (0 cached), 0 failed; 41.8 min of test time".
- r20261114-005344-a40c, tree 201484307189 (C): "4 suites: 4 passed (0 cached), 0 failed; 36.2 min of test time".
- r20261113-061201-25db, tree 3a94714cfbab (#9676): "4 suites: 4 passed (0 cached), 0 failed; 26.6 min".
- r20261113-112505-0ef9, tree 5efe513c14: 4 of 4 passed.
- r20261114-000111-db45, tree 3d62f5507b01 (carry): clean, 29 moves.
- r20261113-172658-0505, tree 30a0a4935c43: ST20_DONE, rc 0.
- Author runs e320, 77ec, 95e0 ("REVERT_DONE all caught"), 4307 and 3021 (SUCCESS).

[2026-11-14T06:38Z] a8de4bdcde76
READ502 (TRN-161), Fri 13 Nov, 10:38 PM PST. **#9502 is correct in its Lean and its reference, and I verified that by running it; it is NOT READY for one false docstring sentence, and one check it adds has no vector that can fail it.**

b90c0adf1db92a789539e7d1ab2727ca77441733 NOT READY (1 must-fix)
Note: `note:20261114T0638Z-report-READ502-9502` (the whole report, under lanes/training).

## First, before the verdict: a record's stated width does change which values count as θ, within a bound

INFERRED from `NciCheck/Record.lean` 577-607 and `NciCheck/RecordText.lean` 276-295 (read whole), and RAN on the
compiled Lean (r20261114-062706-ba0e).

- `widthOf doc p` (RecordText.lean 280) is the format's `word_bits`, and 16 when the key, the parameter's entry or a
  natural-number `word_bits` is absent.
- `isTheta T leaves b p` (Record.lean 577) is now `rows * words * b == 16 * leaves && wordBits == b`, where it was
  `rows * words == leaves && wordBits == 16`, with the root condition unchanged. So at b = 16 it is the old `isTheta`,
  exactly.
- Before #9502, a record with `registered.formats` was refused `inputs`: `keyed` takes only REGISTERED or values ::
  REGISTERED. So for every record the form accepted before, `widthOf` is 16 for every parameter. On those records,
  `finish_case`'s θ list (RecordInputs.lean 288) says exactly what it said before. **The theorem says no less about a
  record the form accepted before**; the form now refuses more of them (below).
- What a stated width can do (new records, with `formats`):
  - It can make a value θ only where the log's first record registers it at the committed root, at exactly that width
    (`wordBits == b`), and with the same total bits. A record cannot state a width that makes θ out of an unregistered
    value or a mismatched registration.
  - The newly reachable θ are weights the first record registers in 32-bit leaves. Before, such a weight was never θ;
    now it is θ when the record states its 32-bit format. That is the change's purpose, but it is a widening of θ.
  - The other direction: a record can leave a registered weight counted as fresh only by stating no format for a
    weight the first record registers in 32-bit leaves and the run's own record does not register. Every other
    misstatement is refused (`formats` or `inputs`). A value counted fresh only raises the fresh count, and the
    table-opening rule cannot be touched: a table's parameter can carry no format (RecordText.lean 286-295), so
    `openingHeld` sees width 16, as before.
- No `sorry`, `axiom`, `native_decide` or kernel bypass in the added lines (RAN: a grep of the diff). The five `lean
  try` runs at 14ca2e3c12 (READ) probe no axiom beyond the three. I tried the two importers of `finish_case`, which no
  run had built (RAN: RecordKinds.lean, r20261114-063017-079e; RecordServed.lean, r20261114-063017-b15c). Both have
  lean exit 0 and no axiom beyond the three.
- Guarantees: none in `security/lean-audit.json` reads `isTheta`, `finish_case` or any changed definition (RAN: a
  grep of its records). One guarantee elsewhere moves: `NciCheck.SecurityProofs.ServedComplies`, in
  `security/Properties/lean-audit.json`, unfolds through `LaidOut` and `RecordText.declared` into `RecordText.inputsOf`,
  which this change edits (INFERRED from `security/Properties/Nci/Record.lean` 299-301 and RecordText.lean 534-539).
  Its `unfolded_hash` moves at the next lock update. It reads neither `isTheta` nor `finish_case`.

## MUST FIX

1. `NciCheck/Record.lean` 171-172 (module docstring, "Formats"): "So a format the log does not state for its value, a
   32-bit value the record declares no format for, and a format on a weight registered in 16-bit leaves are each
   refused." Two of its three clauses are false.
   - A 32-bit value with no format is refused only when the run's own record registers it. The made case "a weight in
     32-bit leaves, no format" is taken, its rows fresh (`FORMATS`; the test asserts it).
   - A 16-bit `row-seg/v1` format on a weight the first record registers in 16-bit leaves is taken. The random vectors
     generate exactly such cases (`formats()` in the generator).
   - Fix: "So a format the log does not state for its value, a value the run's own record registers in 32-bit leaves
     with no format, and a 32-bit format on a weight registered in 16-bit leaves are each refused; a weight only the
     first record registers in 32-bit leaves, with no format, is taken and its rows are fresh." Also on lines 164-165:
     "declares another width for it, 32 bits" should be "declares a format for it, 32 bits or 16 in segment rows".

## SHOULD FIX

1. **The width rule's conjunct `word_bits == b` has no vector that can fail it, in Python or in Lean.** RAN on my VM,
   with each mutation undone:
   - Mutations: drop `and m[2] == b` from the reference's `theta` (nci_record_reference.py 550), then separately drop
     `and m[2] == b` from `formats_held` (586).
   - Each leaves all 20 tests of the two files passing, `test_the_data_is_what_the_audit_pins` included. That test
     regenerates the 687 cases and holds them to the lock's digests. So the generated outputs do not change, and the
     Lean difftest cannot catch the same deletion in `isTheta` (Record.lean 582) or `formatsHeld` (604).
   - What the conjunct stops: a registration of half the record's bits, at word_bits 16 with rows × words = leaves/2,
     under the committed root. Without the conjunct, a 32-bit format makes that weight θ, and the fresh count drops.
   - Two cases close the gap. I built both and ran them (r20261114-062706-ba0e, compiled Lean against the reference):
     - The made case "the scan state in 32-bit leaves, its format right" with the run's record's scan-state entry at
       `word_bits` 16, same counts: refused `formats` by both. It accepts under the `formats_held` mutation.
     - "a weight in 32-bit leaves, its format right" with the first record's `weights/query` entry at `word_bits` 16,
       same counts: refused `formats` by both. Under the `theta` mutation it is accepted with the honest case's own
       count (fresh 64).
   - Fix: add both to `FORMATS` (stage `formats`, a `ran` or `wide` entry with `word_bits` 16), and regenerate the
     lock entry by its own command. That makes 689 cases, and the counts in the lock's `why` and the test docstring
     follow.
2. `NciCheck/Record.lean` 67-68, refusal 11: it names only the width, but `formatsHeld` also refuses `formats` where
   the run's record registers a parameter under another root than it commits, or in a different number of bits (the
   vectors' `format-root` way). Fix: "... is not the one the registration log registers it in, or the run's own
   record registers it under another root than it commits or in other than its bits (`formatsHeld`, "Formats"
   below)."

## NOTE

1. **A case the Lean accepts and the reference refuses, outside the vectors** (RAN, r20261114-062706-ba0e). Write
   `"word_bits":32e0` or `3.2e1` in place of `32` in the made case "the scan state in 32-bit leaves, its format right",
   with the run's record committing the new text. The compiled check ACCEPTs with the honest verdict (fresh 64); the
   reference refuses `inputs`.
   - Cause: `Flock.JsonParse` normalizes an exponent into an integer (`JsonNumber.shiftl`), so `Spot.Wires.nat?`
     reads 32, while `json.loads` gives a float that `_nat` refuses.
   - It is the reference that is wrong, not the Lean (the Lean reads the number's value). The class is older than
     this change and applies to every natural number the form reads (INFERRED). A separate issue: make the
     reference's loader parse numbers as Lean does.
2. Duplicate names: `{"p": A, "p": B}` in `formats`, or `formats` twice in `registered`, is read as the last value by
   both readers (INFERRED: `Base/JsonParse.lean` docstring, TreeMap insert; Python's dict). The verdicts agree; no
   vector holds it.
3. The `formats` code also fires on records with no `formats` key: where the run's own record registers a parameter
   of the record under another root than it commits, or in other than its bits (`formatsHeld`'s first branch). Before,
   those were accepted with the value fresh. The body states the rule. The honest served record passes it.
4. `test_the_compiled_command_gives_the_references_verdicts` runs no `formats` case. The compiled side's `formats`
   coverage is the record difftest inside the Lean audit (687 cases, 0 differences), which runs all 17 `FORMATS`
   cases.

## For the lead to read

- `NciCheck/Record.lean` 577-585 (`isTheta`): decides that a weight registered in 32-bit leaves is θ when the record
  states its 32-bit format. It is the one widening of θ, held by `wordBits == b` and the total-bits equation.
- `NciCheck/Record.lean` 593-607 (`formatsHeld`) and line 770: decides the new refusal, including on format-less
  records whose run's record registers a parameter inconsistently.
- `security/Proofs/Properties/Nci/RecordInputs.lean` 288: the one changed statement. It is identical to the old one on
  every record accepted before.
- `NciCheck/RecordText.lean` 286-295 (`formatsOk`): decides that no table's parameter carries a format, so the
  opening rule is unchanged.

## Each item of the brief

1. **The Lean, whole** (READ whole: Record.lean and RecordText.lean as changed, and the reference's diff). For each
   way the key can come, Lean and reference give the same code:

   | The key | Verdict, both sides | Held by |
   |---|---|---|
   | Absent | taken, width 16; `formatsHeld` still runs | every old case |
   | Not an object (`null`, `[]`, a string) | `inputs` | way `format-keys` |
   | An unknown format: schema not one of the two, width not 16 or 32 (8, 64, 0, "32", true, 32.5, 24), v2 at 16, a key more or fewer, a table's parameter, an odd number of leaves, a read not of whole leaves | `inputs` | ways `format-schema`, `-bits`, `-default`, `-keys`, `-table`, `-whole`, `-read`, and 8 made cases |
   | A duplicate name | the last value, no refusal | INFERRED; no vector |
   | A width that differs from the registration, where the run's record registers the value | `formats` | `format-width`, `format-none`, 3 made cases, and probe `run-entry-16` (not in vectors) |
   | The same, where only the first record registers it, with a format | `formats` | 1 made case, and probe `first-entry-16` (not in vectors) |
   | The same, where only the first record registers it, without a format | taken, fresh | 1 made case |
   | A width for a name not in `params` | `inputs` | `format-param`, 1 made case |
   | A width for a name in `params` that neither record registers | `formats` | `format-unregistered`, 1 made case |

   A case with opposite verdicts outside the vectors: NOTE 1 (`32e0`).
2. **Accepted before, still accepted, should now refuse**: none found. **Now refused, written by an honest serve**:
   none.
   - The rehearsal (READ, r20261114-034525-11dd at 14ca2e3c12): as served, check ACCEPT (blocks 4, fresh 337,786,912,
     equal to the reference) and count ACCEPT; with the smallest weight unregistered, check accepts and count refuses
     `fresh`. Last line: "2 cases, 0 verdict(s) where the compiled command and the reference differ".
   - Its run's record registers the 256 proof-of-work values with `word_bits` 32 (READ, the run's
     `log2-window-registration.json`; the first record has none), so they take `formatsHeld`'s first branch and count
     fresh.
   - Records that the change newly refuses without a `formats` key are inconsistent ones (NOTE 3).
3. Above, first.
4. **The lock** (READ, the diff). Only `runs.NciCheck.RecordDifftestMain` moves: its `why`, `args` (670 to 687) and
   the two digests. The digests equal r20261114-034225-b929's printed sha256 (tree 51fac869bf; the generator and
   reference are identical at the head, RAN `git diff --stat`). I also regenerated the data locally at the head:
   `test_the_data_is_what_the_audit_pins` passes (RAN).
   - The 17 new cases are `len(FORMATS)` (generator lines 852-883), in three groups:
     - Taken (4): the scan state in 32-bit leaves, its format right, in segment rows and in v2 rows; a weight the first
       record registers in 32-bit leaves, its format right; the same with no format (fresh).
     - `formats` (5): the scan state with a format of another width; with no format; registered in 16-bit leaves with a
       32-bit format; a format on a value the run's record does not register; a 32-bit format on a weight registered
       in 16-bit leaves.
     - `inputs` (8): another schema, 16-bit in v2 rows, 24-bit, a key more, a table's parameter, no registered
       parameter, an odd number of leaves, a read of one 16-bit leaf.
   - 13 new random ways, 61 to 74: 8 `inputs` and 5 `formats`.
5. **The tests** (RAN on my VM at the head, with `-k "not compiled and not what_the_audit_pins"`): 19 passed, 2
   deselected. The deselected two are the compiled-command test, because `lake` is on this VM and it would build Lean
   here, and the slow data-pin test, which I then ran alone: passed. Mutations, each undone:
   - `formats` refusal: disable the reference's `formats` stage (nci_record_reference.py 675-676). All 19 error at
     setup: the generator asserts each `FORMATS` case is refused at its stage. Among them is
     `test_a_row_format_is_held_to_the_registration_log`.
   - Width rule: two mutations, both survive (SHOULD FIX 1).
6. **KEY's evidence** (READ, each run's own last line and `tree_sha`):
   - r20261114-034526-e382, tree 14ca2e3c12: "AUDIT: PASS", 1779 declarations in 18 modules, axioms propext,
     Classical.choice, Quot.sound. Its record difftest log: "687 cases compared, 0 differences".
   - r20261114-034532-d03d, tree 14ca2e3c12: "59 suites: 58 passed (0 cached), 1 failed". The red is the network suite
     (97 passed, 2 failed: the two `test_nci_record_check.py` tests the body names).
   - r20261114-041421-5673, tree 840da2df89 (main with both other heads): "1 suites: 0 passed (0 cached), 1 failed",
     the same two tests.
   - r20261114-044749-fd6f, tree b90c0adf (the head): "1 suites: 1 passed (0 cached), 0 failed", 112 passed, none
     skipped.
   - Each says what KEY reports. The head differs from 14ca2e3c12 in `test_nci_record_check.py` alone (+9/-2, RAN).
7. **The body**: each sentence I checked is true of the head.
   - Checked by run or file: the 11 files at +538/-93 (RAN: my comparison tree fd7c152f99 from main 6953e38e55, #9474
     5a6d7a8d04, #9547 dd6555ae61); 13 ways; 17 made cases; 15 that state a format; the proofs named; the lock;
     `OWN`; #9547 touching neither Record.lean, RecordText.lean nor `check_detail` (RAN, `git diff`); every figure in
     "What ran".
   - Paths are today's.
   - What ran is kept apart from what was read, and it says the `security/Proofs` replay did not run.
   - Not checked: "How the head was made" (the restack and merge steps).
8. **Tree rules**: no run id, date, name or pull request number in any added line (RAN, grep). The case counts in the
   lock's `why` and the test docstring are counts, not measurements.

**What I read whole**: the Lean diff, the reference diff, the lock diff, the tests' diffs, and `served_record_case.py`'s
diff. **Sampled**: the generator's diff (FORMATS, `formats()`, `_misformat`, `finish`, `_without`). **Not opened**: the
generator's other unchanged parts, the merge history.

## Runs I launched

- r20261114-062706-ba0e (tree b90c0adf): compiled `inference-only-record-check` on six probe cases. Last line:
  "6 cases, 4 verdict(s) where the compiled command and the reference differ". The four are the two `32e0` / `3.2e1`
  cases, check and count; the two width probes and the two cases as made are equal.
- r20261114-063017-079e (tree b90c0adf): `verity lean try security/Proofs/Properties/Nci/RecordKinds.lean`. Last line:
  "axiom probe: none of its 4 theorems uses an axiom beyond propext, Classical.choice and Quot.sound" (lean exit 0).
- r20261114-063017-b15c (tree b90c0adf): `verity lean try security/Proofs/Properties/Nci/RecordServed.lean`. Last
  line: the same, 4 theorems (lean exit 0).

