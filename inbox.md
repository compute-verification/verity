[2026-11-13T12:07Z] c62e3c2a7a87
DGRAD, Fri 13 Nov 4:07 AM PST (TRN-156: ia3's PR body)

**ia3 is held, and all 38 held entries print every verdict and line the same at the ia3 head as at the dgrad head. The suites pass there. Its draft body is below, for `cursor/ia3-route-8002` at 0d325562c6.**

- q4muon-full at the ia3 head (RAN r20261113-111731-3f3e, 31:03 wall, 18.8 GB): every line the same as at the dgrad head.
- **Base:** the branch carries dgrad's commits by a merge (no rewrite). With base `main`, the draft shows dgrad's diff too until #9731 lands. With base `cursor/dgrad-route-8002`, it shows ia3's 11 files alone. Your call; the body reads either way.
- **What's left on TRN-156:** grow's entry, after #9531's `step` field lands; the restack when you say the chains are on `main`. Nothing of mine is in flight.

---

Lane training
Goal training/search-frontier-in-library

**Route `ia3`, the attack search's scale-vector program, is a route of `benchmarks/training_slowdown`, and its run's entry `m-t20-1.1x-ia3` is held to the run by `python -m training_slowdown.of_record`.** The branch carries the dgrad PR's commits by a merge, so until that PR lands, ia3's own change is the diff against `cursor/dgrad-route-8002`.

## What changes

- **`codecs.RowScale`**: a scale a row at fp32. It packs a program's trained scales as they are, and decodes each row as the base's times its scale.
- **`routes/ia3`**: every weight of the start frozen, and a scale trained on each output channel of each matrix `targets` names, under AdamW. The matrix acts as diag(l)·W, and l starts at one, so the first step's model is θ. The output hands each scaled matrix's scales as its trained parts, and E is the scales.
- **`of_record`**:
  - A lora2 record's E may be rowscale too (`E_KINDS`).
  - The norm rule, narrowly: a scale on a matrix's output multiplies the stream by a vector, as a gain does, so the record's program has backward products of its own over the norm's shapes. Where the config has more of such a product at the same MACs a product, the excess is the norm's, and the rule's line says how many are on both sides. Where the record has more, or its products have other MACs, the rule doesn't fire.
- **Config** `attacks/m-t20-1.1x-ia3.toml`, its steps and dials read from the run's own arguments.
- **Entry** in `of_record.toml` with `partial = "kv:2"`. The count's rule for a kernel without a mask is the gradients of operands that require one (`flops.Counter._attention`). What the card's kernel computes beyond that is outside every job's count.
- **Tests**:
  - the codec;
  - the norm rule's three cases;
  - the route contract: a run's checkpoints are the start with each scaled matrix's rows times its scales;
  - a plain-torch reference in `test_entries`;
  - stopped-and-continued in `test_resume`.

## The run

| entry | run | frontier points | what binds | held or why not |
|---|---|---|---|---|
| `m-t20-1.1x-ia3` | r20261103-075334-0538 | 1 | needs a record: the entry's E keeps less than the trained weights, so a weight costs at most E's bits a trained weight, which a trace alone doesn't have | held: `equal, 1 expected difference` (RAN r20261113-101130-6fa7) |

- **The run's last line** (READ, its `out/evaluate.log`): `valid true, bpb 1.1536177977992577, E_bytes 162048`.
- **What binds** is the row `python -m training_slowdown.library` prints at this branch's head (RAN r20261113-103421-e51e). It prints no slowdown, since the library traces no step whose plan keeps less than the weights.
- **The held line**: 48 of the record's 48 uses of an updated weight, and 29 of its 31 products. The attention is compared under `kv:2`. The norm is expected, with 60 products of the backward's over the same shapes, op and MACs on both sides.
- **E**: rowscale, 116,391 B after 2 steps, against the record's lora2, 162,048 B after 3216. That difference is printed, and it isn't one of the comparison's differences.

## Checks (RAN)

- **Before and after the norm commit**: `of_record` over every held entry at this head (0d325562c6) and at the dgrad PR's head (57411bf29c). All 38 entries held at both print every verdict and line equal.
  - This head: r20261113-101104-8a87, -101130-6fa7, -101142-c642, -101153-6738, -101204-5b07, -101216-fc3b, -101228-2070 and -111731-3f3e.
  - The dgrad head: r20261113-090157-df0a, -090230-6379, -090246-f201, -090304-24f2, -090334-cf86 and -090405-d556, at 6e15d4ba1a, whose later commits change only dgrad's adapter; r20261113-092839-1bcc and -081616-25d9.
- **Suites**: `suites.py --changed 9182900fb9 --quick` at this head, 4 suites passed: repository 355, training_slowdown/grader 25 (3 skipped), training_slowdown 654, verity-graph 220 (r20261113-111857-ba3c).
- **ruff** on the changed Python files: nothing found.
- **Ratchet gate**: passed, nothing new.

Refs TRN-156

