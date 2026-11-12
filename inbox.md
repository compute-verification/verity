[2026-11-12T20:07Z] e50b1dd605e6
TIE to training, Thu 12 Nov 12:07 PM PST. TRN-86 items 1, 3 and 4: ready for you to read and mark. I did not mark it.

**The head.** https://github.com/compute-verification/verity/pull/9408 (#9408), draft, branch `cursor/trn86-period-print-20af`, head `ec31593b910665b07fddd630c63131bf229dd681`. `main` was merged in at `99b81651cc4c`; `main` has moved since and nothing under `benchmarks/inference_only` or `verity/protocols/inference_only/rho` changed on it [RAN: `git diff --stat HEAD...origin/main` on those paths, empty].

**The diff's size.** 9 files, +562 -62 [RAN: `git diff --shortstat origin/main...HEAD`]: `benchmarks/inference_only` (`tied.toml` and `tied.py` new; `policy.py`, `report.py`, its test and README) and rho (`min_cut.py`, its test, one README line). No Lean file, no lock, no allowlist.

**The suites' last lines.**
- The benchmark's and rho's [RAN on a node, `r20261112-180934-2695`, rc 0, at the head]:
~~~text
passed  benchmarks/inference_only    21 passed in 9 s; 21 outcomes recorded
passed  verity/protocols/inference_only/rho 612 passed in 32 s; 612 outcomes recorded
2 suites: 2 passed (0 cached), 0 failed; 0.7 min of test time; logs in /home/research/.cache/verity/tests-slot-r9/logs/20261112T183706-127024
~~~
- Every suite the change touches, without the slow tests [RAN on a node, `r20261112-183237-9cc4`, rc 0, at the head, `suites.py --changed 99b81651cc4c --quick`]:
~~~text
passed  verity/protocols/inference_only/rho 612 passed in 32 s; 612 outcomes recorded
passed  benchmarks/inference_only    21 passed in 7 s; 21 outcomes recorded
passed  repository                   352 passed in 308 s; 352 outcomes recorded
51 suites: 51 passed (0 cached), 0 failed; 101.5 min of test time; logs in /home/research/.cache/verity/tests-slot-r2/logs/20261112T191610-2192787
~~~
- The ratchet gate passes [RAN on my VM at the head]: no new ruff finding, environment read, silent except handler, lane-text site or retired-term site in the 9 files.

**The print** [RAN on a node, `r20261112-181003-04f1`, rc 0, at the head: `uv run verity bench inference_only --uncredited-share 0.002315`, which also builds the class and runs the Lean count]. It still opens:
~~~text
Inference-only: the least slowdown of training run inside the policy's inference blocks

AS SERVED: 1
  unsloth/Llama-3.2-1B, class 4,32,4, served.
~~~
The served and built sections are as before but for one line in each policy's bound (item 3):
~~~text
            the work is training in bf16 words on the model's own matmul shapes, every weight of it and its embedding table outside the registered weights θ: (H) is false of a work that reads a registered weight or a registered table's row, so no figure here covers one; at most two of its three kinds of product are charged (F = 1)
~~~
The new section, whole (the print in full is in the PR's body and in the run's `out/print-full.txt`):
~~~text
The served class under the value tie: NOT SERVED, and every figure of it RESTS ON NO PoUW VERDICT
  The uncredited share s = 0.2315% of a block's work was typed on the command line (--uncredited-share). No PoUW verdict of a run states it here, so each figure holds only where PoUW's checks credit all but that share of every block's work.
  unsloth/Llama-3.2-1B, class 4,32,4, with every tied value right (tied.toml). Nothing serves it, and this tree builds no circuit of it.
  work      the figures are for training in bf16 words on the model's own matmul shapes, every weight of it and its embedding table outside the registered weights θ: (H) is false of a work that reads a registered weight or a registered table's row, so no figure here covers one
  source    the served class with every tied value right: the served policy's c, n and work, and k the scan state row and the execution phase's salt
  c         13,818,131,814,643 Boolean gates a block
  k         146,080 fresh input bits a block, made of
              145,568  the scan state row: 9,098 words × 16 bits, the class's
                  512  the execution phase's salt
  N         36,241,670,144 MACs a block, in the policy's unit
  untied    at most 671,369 untied words in a node's accepted execution phases for each 4 blocks of one 600 s phase, each at most 32 fresh bits in the one block that reads it and fed by at most 2,048 MACs; one block reads at most its 13,189,120 output words of the linears
  uncredited  s N = 83,899,466 MACs a block, which feed at most 40,967 output words of the linears, a word being fed by at least 2,048 MACs (the least fan-in): with those words as fresh input a block reads k + 32 × 40,967 = 1,457,009 bits, and every limit at s charges the share's MACs whole

  One block (min_cut.bound), each number with its limit at s, 1 / (s + 1 / number) (min_cut.capped): a developer gets the uncredited share whole
    every word tied, k = 146,080: number 37,653.4, limit at s 427.1
    the uncredited words as fresh input, k = 1,457,009: number 727.6, limit at s 271.0

  Over one execution phase and over a monitoring period, all of a fleet's compute in the policy's blocks at capacity (min_cut.bound_charged; the slowdown calculator's fleets and periods). A node's bound on its untied words doesn't grow with the period, so one phase's figure is the weakest, and is not a period's
    where           over one           blocks  the blocks' gates  untied words at most   a block
    one datacenter  execution phase  1.638e12           2.264e25              2.749e17  1.678e05
    one datacenter  week             1.651e15           2.282e28              2.749e17     166.5
    one datacenter  month            7.077e15           9.778e28              2.749e17     38.85
    one datacenter  year              8.61e16            1.19e30              2.749e17     3.193
    the world       execution phase  6.142e15           8.487e28              1.031e21  1.678e05
    the world       week             6.191e18           8.555e31              1.031e21     166.5
    the world       month            2.653e19           3.667e32              1.031e21     38.85
    the world       year             3.228e20           4.461e33              1.031e21     3.193

  Each figure under its reading's name, which holds its threshold, with its limit at s = 0.2315% and the pooled total in gates (the gates c counts): the training the blocks hold at most, at the limit
    where           over one         reading                                                                           blocks charged at most    number  limit at s  training's gates at most  of the blocks' gates
    one datacenter  execution phase  one β_x, with the threshold w = 1.06837e+06                                                     2.573e11       2.1         2.0                  1.109e25                48.99%
    one datacenter  execution phase  one β_x at the top budget, no block charged                                                            0       1.0         1.0                  2.264e25                  100%
    one datacenter  execution phase  a β_x a block, no threshold                                                                            0       1.0         1.0                  2.264e25                  100%
    one datacenter  execution phase  one β_x, with the threshold w = 1.06837e+06, the uncredited words as fresh input                2.573e11       2.0         1.9                  1.162e25                51.33%
    one datacenter  week             one β_x, with the threshold w = 911841                                                          3.015e11   1,614.1       340.8                  6.696e25               0.2935%
    one datacenter  week             one β_x at the top budget, no block charged                                                            0      52.8        47.0                  4.853e26                2.127%
    one datacenter  week             a β_x a block, no threshold                                                                            0      53.2        47.4                  4.815e26                 2.11%
    one datacenter  week             one β_x, with the threshold w = 204468, the uncredited words as fresh input                     1.345e12     395.7       206.5                  1.105e26               0.4842%
    one datacenter  month            one β_x, with the threshold w = 825882                                                          3.329e11   4,787.8       396.2                  2.468e26               0.2524%
    one datacenter  month            one β_x at the top budget, no block charged                                                            0     202.2       137.7                  7.099e26                0.726%
    one datacenter  month            a β_x a block, no threshold                                                                            0     209.6       141.1                  6.929e26               0.7086%
    one datacenter  month            one β_x, with the threshold w = 204468, the uncredited words as fresh input                     1.345e12     604.2       251.9                  3.882e26                0.397%
    one datacenter  year             one β_x, with the threshold w = 245435                                                           1.12e12  20,422.0       423.0                  2.812e27               0.2364%
    one datacenter  year             one β_x at the top budget, no block charged                                                            0   1,336.6       326.5                  3.644e27               0.3063%
    one datacenter  year             a β_x a block, no threshold                                                                            0   1,844.3       350.0                  3.399e27               0.2857%
    one datacenter  year             one β_x, with the threshold w = 204468, the uncredited words as fresh input                     1.345e12     715.4       269.3                  4.417e27               0.3713%
    the world       execution phase  one β_x, with the threshold w = 1.06837e+06                                                     9.649e14       2.1         2.0                  4.158e28                48.99%
    the world       execution phase  one β_x at the top budget, no block charged                                                            0       1.0         1.0                  8.487e28                  100%
    the world       execution phase  a β_x a block, no threshold                                                                            0       1.0         1.0                  8.487e28                  100%
    the world       execution phase  one β_x, with the threshold w = 1.06837e+06, the uncredited words as fresh input                9.649e14       2.0         1.9                  4.357e28                51.33%
    the world       week             one β_x, with the threshold w = 911841                                                          1.131e15   1,614.1       340.8                  2.511e29               0.2935%
    the world       week             one β_x at the top budget, no block charged                                                            0      52.8        47.0                   1.82e30                2.127%
    the world       week             a β_x a block, no threshold                                                                            0      53.2        47.4                  1.805e30                 2.11%
    the world       week             one β_x, with the threshold w = 204468, the uncredited words as fresh input                     5.042e15     395.7       206.5                  4.142e29               0.4842%
    the world       month            one β_x, with the threshold w = 825882                                                          1.248e15   4,787.8       396.2                  9.254e29               0.2524%
    the world       month            one β_x at the top budget, no block charged                                                            0     202.2       137.7                  2.662e30                0.726%
    the world       month            a β_x a block, no threshold                                                                            0     209.6       141.1                  2.598e30               0.7086%
    the world       month            one β_x, with the threshold w = 204468, the uncredited words as fresh input                     5.042e15     604.2       251.9                  1.456e30                0.397%
    the world       year             one β_x, with the threshold w = 245435                                                            4.2e15  20,422.0       423.0                  1.055e31               0.2364%
    the world       year             one β_x at the top budget, no block charged                                                            0   1,336.6       326.5                  1.366e31               0.3063%
    the world       year             a β_x a block, no threshold                                                                            0   1,844.3       350.0                  1.275e31               0.2857%
    the world       year             one β_x, with the threshold w = 204468, the uncredited words as fresh input                     5.042e15     715.4       269.3                  1.656e31               0.3713%
  proved    Nci.Cut.Guarantees.HE (proved in the lock), the count one block's number is solved from. A period's count is Nci.Cut.Guarantees.HECharged (no lock lists a proof of it), and with a β_x a block draft:count-hardness-a-block (no Lean statement). No limit is proved: it is arithmetic on the typed share
  premise   the share s, typed; that PoUW ties every word it credits, which no verdict of a run from this tree shows; the bound on a period's untied words (draft:period-untied-words) and an untied word's bits (draft:untied-word-bits), neither a Lean statement; (H) Nci.Cut.Assumptions.Hardness and (E) Nci.Cut.Assumptions.Exports, conjectures, at the circuit with the untied words as inputs; a threshold fixed before the period, with so few blocks over it in the partition the period's audit reads; and the work's structure, as above
  scaled    every row: the blocks are a fleet's at its data sheet's rate, the fleets' sizes the census's; a node's bound on its untied words is one execution phase's, scaled by the blocks the node serves in a phase; the uncredited share is taken as every block's; and one block may be untied whole
  not in any figure: a dropped execution phase, a voluntary row or a version the verdict doesn't cover, which nothing here bounds
~~~

**What it shows.**
- At the run of record's share, 0.2315% [READ: compute-accounting's rows, 335,544,320 MACs in 7 calls over 4 blocks], no limit passes 432.0. One block: 37,653.4, limit 427.1; with the uncredited words as fresh input 727.6, limit 271.0 [RAN, the print].
- At 0.097% [RAN, same run, `out/print-stated-0.097.txt`]: one block's limit is 1,003.5 and no period's reaches 1,000. Best threshold: 629.1 a week, 848.3 a month, 981.4 a year. One block with the uncredited words as fresh input: 778.6. So 0.097% buys 1,000 for one block with every word tied and nothing else; that matters for `training/small-model-slowdown-1000`.
- With no block charged the week's number is 52.8 whatever the share [RAN, both prints].

**Three things I decided that you can reverse.**
1. The figure prints as "number" (the benchmark's own word), not "cut number" as my earlier notes to you said.
2. Each fleet's one execution phase prints before its periods: 2.1 under the best threshold, 1.0 under the other readings [RAN]. The issue didn't ask; AGENTS.md asks for the per-phase figure beside the pooled ones, and the print says a phase's figure is not a period's.
3. The table has no column for the pooled gates by the count alone, only at the limit, so no figure in the section stands without the share. The JSON keeps both.

**Item 4.** 45 sites by today's lint, not 47; 32 gone here; 9 Lean prose words ride COUNTS's PR; 4 stay as names Lean reads [RAN: the lint's `sites()` over each file #8703's landing touched, at four commits]. The body has the table. COUNTS and I exchanged lines: `test_min_cut.py` line 198 is the one line both touch, and COUNTS wrote the merged sentence.

**Not done, and why.**
- The PR body's last update did not go through: `research queue edit` answers "GraphQL: API rate limit already exceeded for installation ID 171128908" [RAN, three times over five minutes; the installation's window resets at 12:48 PM PST]. The body on GitHub holds the print in full, the first suites run and item 4's table, and still says the changed-suites run is waiting. The finished body is `/opt/cursor/artifacts/trn86/pr_body_edit.md` on my VM; I retry after the reset and tell you in one line whether it landed.
- Not run: the whole `check`, the Lean check (no Lean changed), `circuit-check` (no circuit changed).

Notes, synced: `lanes/training/20261112T1848Z-checkpoint-trn86-period-print-handoff.md` (hand-off) and `lanes/training/20261112T1834Z-checkpoint-per-call-uncredited-words.md` (the ten lines).

