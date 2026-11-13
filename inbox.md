[2026-11-13T11:18Z] 4b1b98da815a
DGRAD, Fri 13 Nov 3:18 AM PST (TRN-156: the dgrad PR body, before and after done, ia3 held)

**Before and after the `per_row` commit, all 33 of main's held entries print every verdict and line equal. The suites pass at the dgrad head, and the PR body is attached below for you to open the draft. The library prints no slowdown for any dgrad entry: each row says "needs a record". At the ia3 head, ia3 is held and 37 of 38 other entries print the same as at the dgrad head, with q4muon-full's run in flight.**

1. **Before and after** (RAN):
- The six reruns: r20261113-100742-cfe2 (five entries, 8:00 wall, 7.3 GB) and r20261113-100836-b898 (xl-t20 alone, 30:10 wall, 24.8 GB on its 28 GB ask).
- With the five earlier before chunks, all 33 entries are on both sides, and every verdict and every printed line is equal.

2. **Suites at 57411bf29c** (RAN r20261113-101002-52a4): 4 suites passed: repository 355, grader 25 (3 skipped), training_slowdown 647, verity-graph 220. ruff and the ratchet gate find nothing new (RAN locally).

3. **What binds** (RAN `python -m training_slowdown.library` at 57411bf29c: r20261113-103345-3448 for the five s-t20 entries, r20261113-103410-1631 for grow).
- Every dgrad row prints `-` for the slowdown, and in what binds: "needs a record: the entry's E keeps less than the trained weights, so a weight costs at most E's bits a trained weight, which a trace alone doesn't have".
- That's `library.CODED`: the library traces no step whose plan keeps less than the weights, and every dgrad plan codes its E. ia3 prints the same at its head (RAN r20261113-103421-e51e).
- So the table's what-binds column is that why for all six. If the library should price a coded entry from its record, that's a change to `library.py` beyond this PR.

4. **One line against your table** (READ each run's `meta.json`, key `changed`): r20261104-014110-36cb changes only the MLP matrices (`w_down`, `w_gate`, `w_up`), so its entry is `s-t20-1.1x-dgrad-q4r-mlp`, not `dgrad-q4r`. r20261104-020846-2e66 is `s-t20-1.1x-dgrad-q4r`. The split is otherwise as you gave it.

5. **ia3** (on `cursor/ia3-route-8002` at 0d325562c6, which carries 57411bf29c):
- `m-t20-1.1x-ia3` is held: `equal, 1 expected difference` (RAN r20261113-101130-6fa7). 48/48 uses, 29/31 products. The attention is compared under `kv:2`, and the norm's expected line names 60 products of the backward's that are on both sides.
- The norm commit's before and after: 37 entries print every line the same at the ia3 head as at the dgrad head. The ia3 head's runs are r20261113-101104-8a87, -101130-6fa7, -101142-c642, -101153-6738, -101204-5b07, -101216-fc3b and -101228-2070. The dgrad head's lines are the after chunks above, 1bcc for the LoRA entries and 25d9 for q4muon-full.
- q4muon-full was missing from those chunk lists. It's running now at the ia3 head: r20261113-111731-3f3e.
- When it's read, I send ia3's body, after dgrad's.

6. **The draft PR body for `cursor/dgrad-route-8002`**, head 57411bf29c, base main:

---

Lane training
Goal training/search-frontier-in-library

**Route `dgrad`, the attack search's delta_grad program, is a route of `benchmarks/training_slowdown`, with an entry for each of its six frontier runs. Five entries are held to their runs by `python -m training_slowdown.of_record`. Grow's entry waits on #9531's `step` field.**

## What changes

- **`routes/_codes`**: the codes a step computes with. Messages are a stochastic rounding of a few bits a group, in a basis rotated a group, or a vote of signs; momentum is kept in a few bits; the store is a sticky 8-bit grid with an f16 scale a group.
- **`routes/dgrad`**: the step's gradient reduced over its blocks of rows through a heap of coded messages. The trained matrices, or low-rank adapters that may grow, are kept between steps in 8-bit codes. The factors are rebuilt from the store every step, so the route carries no state a step doesn't read back from its codes. B is laid out as a parameter, `(out, r)` in rows, because autograd arranges a weight's gradient product by its strides.
- **Codecs**: `LowRank(bits)` rounds each rank component to nearest once at encode. `ScaledQuant.encode_trained` packs a program's trained codes as they are.
- **`of_record`**:
  - It reads a lora2 record's lowrank plan, and identity beside a quant record's own codecs (`E_KINDS`, `E_BESIDE`, `E_PACKING`).
  - `per_row` leaves a row of an optimizer's phase (`gates.OPTIMIZER_PHASES`) whole, since the counter gives a row that phase only where no operand derives from a token.
  - `tokens_by_size` refuses a record whose rows a forward times the context equals a width of the model (`d_model`, `d_ff` or `vocab`), and names the width.
- **Configs**: six in `attacks/`, each step and dial read from its run's own arguments.
- **Entries**: five in `of_record.toml`, each with the run's printed loss `record_only` and its reason.
- **Tests**: the route contract in `test_routes`; a plain-torch reference of the step in `test_entries`, adding the adapter's own product x·Aᵀ·Bᵀ through a forward hook as the route does; stopped-and-continued in `test_resume`; the codecs; and two toy cases in `test_of_record` for the `per_row` change and the refusal.

What this does not fix: `of_tokens` still reads a dimension as tokens by its size. Where a width of the model equals the record's rows times the context, `tokens_by_size` now refuses the entry and names the width, rather than dividing it silently.

## The six runs

Each record's last line is `out/evaluate.log`'s verdict (READ, each run's record).

| entry | run | frontier points | what binds | held or why not |
|---|---|---|---|---|
| `m-t20-1.1x-dgrad-grow` | r20261103-140155-106f | 6 | needs a record (below) | not in `of_record.toml`: the step changes through the run, and the entry waits on #9531's `step` field. Counted at step 164 on a scratch branch: `equal`, 896/896 uses, 72/73 products (RAN r20261113-090724-d211) |
| `s-t20-1.1x-dgrad-q4r-mlp` | r20261104-014110-36cb | 2 | needs a record | held: `equal, 1 expected difference, 1 named line of the record's program alone` (RAN r20261113-092839-1bcc) |
| `s-t20-1.1x-dgrad-q4r` | r20261104-020846-2e66 | 2 | needs a record | held: the same line (RAN r20261113-092839-1bcc) |
| `s-t20-1.1x-dgrad-q2r` | r20261103-113630-cd91 | 2 | needs a record | held: the same line (RAN r20261113-092839-1bcc) |
| `s-t20-1.1x-dgrad-vote` | r20261104-013838-0b85 | 1 | needs a record | held: the same line (RAN r20261113-092839-1bcc) |
| `s-t20-1.1x-dgrad-q4muon-full` | r20261104-061509-8e1d | 1 | needs a record | held: the same line (RAN r20261113-081616-25d9) |

- **What binds** is the row `python -m training_slowdown.library` prints for the entry at this branch's head (RAN r20261113-103345-3448 for the five s-t20 entries, r20261113-103410-1631 for grow). It prints no slowdown for any of the six, with the same why: "needs a record: the entry's E keeps less than the trained weights, so a weight costs at most E's bits a trained weight, which a trace alone doesn't have". The library traces no step whose plan keeps less than the weights (`CODED`).
- **Frontier points**: the split of the 14 the six runs hold, from the lane's table (READ). No record carries a count of its own.

The runs' last lines (READ):
- r20261103-140155-106f: `valid true, bpb 1.1537901766037717, E_bytes 4921972`;
- r20261104-014110-36cb: `valid true, bpb 1.25800590288108, E_bytes 17504`;
- r20261104-020846-2e66: `valid true, bpb 1.2579062773674021, E_bytes 89418`;
- r20261103-113630-cd91: `valid true, bpb 1.257993149632853, E_bytes 89418`;
- r20261104-013838-0b85: `valid true, bpb 1.257746092848176, E_bytes 327306`;
- r20261104-061509-8e1d: `valid true, bpb 1.2444008389364527, E_bytes 18829664`.

## Checks (RAN)

- **Before and after the `per_row` commit**: `of_record` over every held entry of `main`, printing every verdict and line equal for all 33 entries.
  - Before, at main 9182900fb9: r20261113-090508-657a, -090539-ca77, -090608-7eeb, -090623-c835, -090652-d7df, -100742-cfe2 and -100836-b898.
  - After, at 6e15d4ba1a: r20261113-090157-df0a, -090230-6379, -090246-f201, -090304-24f2, -090334-cf86 and -090405-d556.
  - The two commits after 6e15d4ba1a change only `routes/dgrad.py` and its reference, which no entry of `main` reaches.
- **Suites**: `suites.py --changed 9182900fb9 --quick` at 57411bf29c, 4 suites passed: repository 355, training_slowdown/grader 25 (3 skipped), training_slowdown 647, verity-graph 220 (r20261113-101002-52a4).
- **ruff** on the changed Python files: nothing found.
- **Ratchet gate** (`adapters/vllm/tools/ratchet_gate.py origin/main HEAD`): passed, nothing new.

Refs TRN-156

[2026-11-13T11:26Z] f543e07f47b5
WALKER, Fri 13 Nov, 3:27 AM PST.

**Fix 3 is draft [#9730](https://github.com/compute-verification/verity/pull/9730) at 5efe513c14, cut from #9676's head. `Work` is byte-equal with the rule on and off at every shape I read, up to s(2,256,512). With the rule on, the reading adds nothing measurable above the build's own peak. The column case costs 5.6 s per reading, so the hash sweep isn't needed.**

- **Equal on and off.** RAN: r20261113-103730-77ec compared `Work` byte for byte, rule on against rule off, at tiny(2,16,64), s(2,16,64) and s(2,64,512); all equal. RAN: at s(2,256,512), the rule-on reading (r20261113-103757-2734) and the rule-off reading (r20261113-103824-cc0b) have the same sha256.
- **Before and after.** RAN: r20261113-110022-b1a5 is #9676 at 3a94714cfb, untraced. r20261113-110211-383d is this head, with the column clock. Both ran on cpu-7 at the same time.

  | shape | before (#9676) | after | build alone |
  |---|---|---|---|
  | s(2,256,512) | 836.1 s, 12.99 GiB | 705.1 s, 10.15 GiB | 10.1 GiB |
  | tiny(2,1024,64) | still running in b1a5 | 308.5 s, 3.09 GiB | 3.1 GiB |

  RAN: with the rule off, tiny(2,1024,64) took 796.3 s at 9.44 GiB (cc0b). INFERRED: #9676 will land near that. RAN: the same head alone on cpu-11 read s(2,256,512) in 468.9 s, so the times move with load and the peaks don't.
- **Column case** (your item 2). RAN: 5.6 s over 192 column products at s(2,256,512), out of 705.1 s. 5.6 s over 16 at tiny(2,1024,64), out of 308.5 s. INFERRED: under 2%, so I don't propose the follow-up.
- **Counts.** RAN: rule on, every reading has softmax:runs and select:runs, and no softmax or select read by words (4+4 on tiny, 12+12 on s). The column case is counted apart as `("columns","words")`, by design.
- **Your four conditions**, with the PR body listing each line that checks one:
  - No float fact is written twice: the rule's verdicts come from `_generic`, `_shape` and `_finite`, and its code names neither 0 nor -inf.
  - Canonical runs: `_runs` merges equal neighbours, and rows compare by `_shape` of the expanded constants, never by parts.
  - Branch tests: the `split`, `masked-row` and `finite` bodies, plus `input-word` (guarded and unguarded) and `repeated-base`. All are equal on and off and state which nodes fall back. RAN: mutations on the VM, reverted. An always-false condition fails `masked-row` and `input-word`. Comparing raw constants instead of `_shape` fails `finite`.
  - The reading says how it read: `Reader.read_by`. The tiny test holds rows and selects at zero fallbacks and states 16 column products. The flag and the clock are constructor arguments; no CLI.
- **Suites.** RAN: 4 of 4 passed at 38e65802bf (r20261113-103702-78cf). The head since adds the column count and clock, a docstring, and the tiny count test; that test passes on the VM by name. A rerun at 5efe513c14 is in flight as r20261113-112505-0ef9. `--on auto` placed it on cpu-2. I did not name cpu-2; say if you want it moved.
- **s-t20 "if it then reads".** INFERRED: it should read at about the build's 57.5 GiB plus `_leaves`. Launching it means another 160 GB machine beside 54ec, and you kept the check machines for the chains, so I'm waiting for your go.
- **54ec.** RAN: still in the pass. Heartbeat at 17,913 s: RSS 128.9 GiB, pass peak 132.2 GiB, against a stop at 144 GiB. I'll send its end.
- No merge of `main`.

