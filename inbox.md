[2026-11-14T09:37Z] b23b4b5b941b

READCIRC to training and WALKER, TRN-162 read of #9558's zero-row delta, Sat 14 Nov 1:36 AM PST (09:36Z, from `date`). Note: training/20261114T0936Z-report-trn162-readcirc-a4.

NOT READY 2 9767b09f22eca8f9ca137f83c5ed074a938faa63

The delta is 1637d668..df061808 (4c96f3db79d6, the silent except, and df06180894f, the zero-row rule), plus 9767b09f22e on top (docstrings only). The rule works at both points, N never rises, and equal nonzero rows are still refused. But the charge can be more than the true work in two cases, which by your rule is the must-fix: a site charged MACs that every one read a zero row. Both have a tested patch of 18 lines that moves no figure in either suite.

## MUST FIX

**M1. An operand of one row is never checked for zero, so a zero `x` is charged whole.** RAN. `Walk.rows` returns on `(W // K) * (A if by_words else 1) < 2` before it computes `zero`. So `LinearRef`'s `x`, one row an activation, never records a zero row at the one-word point (nor at the second point when A is 1).
- Probe `xzero`: u′ = u + (−1)·u, carried u and W, y = u′ Wᵀ. Every MAC reads the zero row u′. At 9767b09f, N = 48 = (T−1)·K·N, the whole product. With W′ zero instead (the suite's zero-signs case), N = 0.
- The fix: compute and record `zero` before that return, and return after recording when there is no pair to compare.

**M2. `Tally.share` treats the operands' zero rows as independent, so anti-correlated zeros are charged.** RAN. `share` = 1 − Π(1 − zᵢ) over the site's arguments. That is the exact share only when every row of x meets every row of w. In a batch whose operands are both per member (`axes=(0, 0)`), a row meets only its own member's rows.
- Probe `anti`: batch over two members of x_m W_mᵀ, carried, with x₀ = 0 and W₁ = 0 (each a minus a times a stated mask). Every MAC reads a zero row. z_x = ½ and z_w = ½, so share = ¾, and N = 24 of the 96 without the point. It should be 0.
- The fix: the union bound, `share = min(1, Σ zᵢ)`. For any pairing of rows inside an activation, the share of MACs that read a zero row is at most z_x + z_w. Each zᵢ is already the most over the evaluated activations, so the bound covers each of them, and anything above it is an undercharge, the safe side. Where one operand is shared, the true share is z_x + z_w − z_x·z_w; the bound gives a little more, an undercharge.

**The patch** (RAN on `points.py` at df06180894f and at C's 9155c1dc; same code at 9767b09f):

```diff
@@ class Tally:
     def share(self, sid: int, node: int) -> Fraction:
         """The share of product ``node``'s MACs that read a row zero at its point, by either operand: no work."""
-        kept = Fraction(1)
-        for (s, k, _i), z in self.zero.items():
-            if (s, k) == (sid, node):
-                kept *= 1 - z
-        return 1 - kept
+        return min(Fraction(1), sum((z for (s, k, _i), z in self.zero.items() if (s, k) == (sid, node)), Fraction(0)))
@@ def rows(
-        if not K or W % K or (W // K) * (A if by_words else 1) < 2:
+        if not K or W % K:
             return
@@
             v = np.where(v == np.uint64(1 << (w - 1)), np.uint64(0), v)
+        zero = ~v.any(axis=1)
+        if zero.any():
+            at = (S.sid, k, i)
+            self.tally.zero[at] = max(self.tally.zero.get(at, Fraction(0)), Fraction(int(zero.sum()), A * R))
+        if R * (A if by_words else 1) < 2:
+            return
         if w == 32:
             bad = _nan(v).any(axis=1).reshape(A, R)
@@
                                       nan=(int(bad[a].sum()), R), by_words=by_words)
-        zero = ~v.any(axis=1)
-        if zero.any():
-            at = (S.sid, k, i)
-            self.tally.zero[at] = max(self.tally.zero.get(at, Fraction(0)), Fraction(int(zero.sum()), A * R))
         member = np.arange(A, dtype=np.uint64)
```

`share`'s docstring should then say "at most the sum of its operands' shares". A NaN row is not zero, so recording zero before the NaN check changes no refusal.

RAN with the patch:
- `test_work.py`: 124 passed at df06180894f, and 134 at 9155c1dc, plus the two tests below.
- `test_value_reading.py` and `test_program_reading.py`: 17 passed at A and 20 at C.
- Every `WK.read` in those files, logged per test: the same N with and without the patch, 157 + 34 readings at A and 177 + 36 at C. `sdpa` stays 3,776 and `relu` 4,800.
- Ruff is clean.
- Each half cures only its own probe: the share half fixes `anti` alone, the early-return half `xzero` alone.

The tests (`_xzero` and `_anti` are in the probe file, `lanes/training/trn162-readcirc-probes-zero.py`). Both fail at 9767b09f and pass with the patch:

```python
@pytest.mark.parametrize("make,full", [(_xzero, (T - 1) * K * N), (_anti, (T - 1) * 2 * K * N)], ids=["xzero", "anti"])
def test_every_mac_reading_a_zero_row_is_left_out(make, full, monkeypatch):
    w = WK.read(make(), KB)
    monkeypatch.setattr(PT, "check", lambda *a: PT.Tally(points=0))
    assert WK.read(make(), KB).N == full and w.N == 0
```

## SHOULD FIX

- **S1.** Until M1 and M2 land, "none of the MACs that read it is charged" is false of the code in four places: `points.py`'s opening and its zero-row paragraph, `work.py`'s refusal paragraph, the README, and the body's "Zero rows" section. With the patch they are true.
- **S2.** Two statements of the refusal still state it without the zero-row exception (READ, at 9767b09f):
  - `points.py` lines 13–14: "A NaN and equal rows are refused, so the sign passes no claim".
  - `points.py` line 33: "so a row repeated a member against one shared operand is refused". A zero row repeated a member is left out. The body says the same in its own words twice.

  Suggested wording: "equal rows that are not zero" and "a row, not zero, repeated a member … is refused".

## NOTE

The eight checks.

1. **N never rises.** RAN, by logging every `WK.read` per test at both heads.
   - Over `test_work.py`, `test_value_reading.py` and `test_program_reading.py`, every reading the two heads share is the same, except the renamed zero tests and `sdpa`/`relu`.
   - `sdpa` and `relu` are *refused* at 1637d668 (`RepeatedRows` x, rows 0 and 4; three of `test_value_reading.py` fail there), not charged 3,840 and 4,864. Those are their N without the point at both heads. At df06180894f they read 3,776 and 4,800, as WALKER says.
   - `after_mask` 4,096, `rope` 3,328 and `lf` 9,472 don't move, nor does the command's small case (6,029,312). `readings.json` is not in the diff.
   - My four old probes, refused at 1637d668, read N 0 at df06180894f (zero32, zero16, cancel, members); without the point they read 48, 48, 32 and 96.
   - N falls from refusal to a number only where rows were equal zero rows. The code can't raise N: it subtracts `ceil(n·macs·share)`.
2. **At each point.** RAN.

   | case | point | result |
   |---|---|---|
   | zero rows | one-word | the suite's zero-signs cases, and my `all` probe: N 0 of 48 |
   | some rows zero | one-word | `one`: N 32; `two`: 16; `word` (one row with one nonzero word, two zero): 16 |
   | equal nonzero rows, with a zero row beside them | one-word | `eq` refused, rows (1, 2) |
   | every row cancelled plus 1 | one-word | `plus1` refused, rows (0, 1) |
   | zero rows | word-by-word | the suite's `cancel` and `first` (half charged) |
   | equal nonzero rows | word-by-word | the suite's `equal` and `members` refused |

   Zero at one point and not the other: each claim is judged only at its own point, as the code says. Probe `pair`: W′ = W ⊙ (a₀ − a₁), a₀ and a₁ two words of one input, distinct by the multiplier premise. It is zero at the one-word point, where every word of an input is one word, and not for served inputs. N is 0 against 48 without the point: an undercharge, the safe side. Two different inputs draw different words there, so a − b of two inputs is not zero and is charged (N 48).
3. **The share.**
   - The max over evaluated activations, applied to every count: an activation with fewer zero rows than the share is undercharged, the safe side. READ, and RAN in effect by `first`.
   - More charged than its own zero rows allow: yes, M1 and M2.
   - A third way, INFERRED and not probed: an activation the pass doesn't evaluate (a scan iteration whose context repeats) with more zero rows than the evaluated ones, say a decay that reaches 0 only in later iterations. That rests on the stated premises (one evaluation of the step; a multiplier taken as nonzero), as equal rows there already did, so it is not this PR's.
4. **The fold.** RAN: a row of −0 words is zero (the signs cases; the FOLD mutation fails 2), and a row with one nonzero word is not (probe `word`). READ:
   - An f32 NaN row is refused before the zero test.
   - A bf16 NaN word is nonzero, so its row is compared, as before.
   - A denormal word is nonzero, so its row is not zero and is charged.
   - The 8-bit fold is still unreachable, since `DTYPES` holds bf16 and f32 only.
5. **Mutations.** RAN on `test_work.py` at df06180894f. Each matches WALKER's count, every failure a zero-row test.
   - Comparing zero rows (`live = np.arange(zero.size)`) fails 4: the three zero-signs cases and `cancel-2`.
   - Charging them (`g[0] += n * site.macs`) fails 5: those 4 and `first-1`.
   - Dropping the fold fails 2: zero-signs `[signs]` and `[bf16]`.
6. **Statements of the rule.** RAN `git grep` at 9767b09f.
   - Stated with the zero-row rule: the README, `points.py`'s opening, zero-row paragraph and `check`, `rows`, `Tally`, and `work.py`'s N bullet, refusal paragraph and `RepeatedRows`.
   - Not stated: the two places in S2.
   - 9767b09f changes no code: only docstrings of `points.py` and `work.py`.
7. **The silent except.** READ: `_peak_gib` now returns None only where the status has no `VmHWM` line, and lets an `OSError` raise, as `_resident_gib` already does. `_reset_peak`'s `except OSError: return False` is used, not silent. RAN: `tests/lint/test_repo_silent_excepts.py` 3 passed. I have no earlier note of mine on it, so I held it to the lint test and the docstring.
8. **The quick tests.** RAN at df06180894f: `suites.py circuit_properties benchmarks/training_slowdown --quick --changed 3d62f5507b01 --fresh` gave `2 suites: 2 passed`, circuit-properties 141 and training_slowdown 683, with `test_value_reading.py` and `test_program_reading.py` run whole. Ruff is clean on the three changed files at 9767b09f.

## For the lead to read

- WALKER asked whether the rule applies at both points. Both points are right: neither raises N or lets an equal nonzero pair through (checks 1 and 2).
- WALKER's "leave the share as built" fails your test, because a site can be charged more than its true work (M1, M2). The patch keeps one share a site, as WALKER wants, and needs no share for each context.
- Events meant for @training reached me through the hook, and I acted on none: WALKER's 1:01 AM report with its 30-minute question, and its 1:10, 1:24 and 1:43 AM status messages. I read your 1:10 and 1:25 AM messages, which name me, as asks.
- Launched: nothing. Posted: nothing.

[2026-11-14T09:37Z] 16a8803d47f1

READCIRC to training and WALKER, TRN-162 read of #9730's merged head, Sat 14 Nov 1:37 AM PST (09:37Z, from `date`). Note: training/20261114T0937Z-report-trn162-readcirc-c4.

NOT READY 2 9155c1dc31c2087c252c1b69468ca78adc19c52e

C's own change is clean: the merges add nothing, no figure of C's moves, and M-C1 and M-C2 still fail their cases. Both items are A's MUST FIX (training/20261114T0936Z-report-trn162-readcirc-a4), which C carries by the merge. Merging A's fix cures them here. I applied the patch at 9155c1dc: it applies cleanly, `test_work.py` passes 134 plus the two new tests, and every reading stays the same.

## MUST FIX

- **M1 and M2, A's.** A zero one-row operand is charged whole, and anti-correlated zero rows in a batch are charged. The code is the same in C's `points.py`: RAN at 9155c1dc, `xzero` N 48 and `anti` N 24, both 0 with the patch.

## SHOULD FIX

- A's S1 and S2, the same text in C.

## NOTE

- **The merges are clean and add nothing.** RAN.
  - `git show --remerge-diff` of 1cf8759b and of 9155c1dc: 0 lines each.
  - C's own diff, `git diff 9767b09f 9155c1dc` against `git diff 1637d668 7e94d534`: the same 4 files, 5,654 in and 48 out, and every added and removed line identical.
- **No figure of C's moves, and no C test rests on a zero row.** RAN: every `WK.read` in `test_work.py`, `test_value_reading.py` and `test_program_reading.py`, logged per test at 7e94d534 and at 9155c1dc.
  - What differs is A's alone: the renamed zero tests, and `sdpa`/`relu`, refused at 7e94d534 and 3,776/4,800 at 9155c1dc. Nothing rises.
  - C's structured-reading tests read the same N at both heads. The "zero" words in C's diff are stated mask words (`const(32, 0)`), not zero rows of a product.
  - `test_work.py` 134 passed; `test_value_reading.py` and `test_program_reading.py` 20 passed.
- **M-C1 and M-C2.** RAN at 9155c1dc: M-C1 (`_nth(vals, 0)`) fails `[first-masked]` alone, and M-C2 (the `uniform(...) is _MIXED` guard dropped) fails `[computed-mask]` alone. Each gives 1 failed and 133 passed.
- **The body.** RAN: all 33 `work.py` lines the body cites (sections "Rows", "Selects", "Runs", "The reading says how it read", and the two mutations at 1077 and 1047) are, line for line, the lines at 7e94d534 moved by two. That is A's two docstring lines above them, and at 7e94d534 I matched them to their claims.

## For the lead to read

- C is READY once A's fix is merged in and its sha sent. For that delta I would read the remerge-diff, the probes and M-C1 and M-C2 again.
- Events meant for @training reached me, and I acted on none: they are named in A's verdict. Launched: nothing. Posted: nothing.

