[2026-11-12T23:30Z] 0d3bffb9bf3c
BACKWARD2, Thu 12 Nov 3:30 PM PST.

**My one launch of the probe was refused for a wrong flag of mine, before it touched the node; nothing runs, and I'm stopping until your word on a second launch with `--project verity`.**

- RAN, 23:30Z: `research run --on vy-nebius-2 --project training ... --mem-gb 16 -- bash -c 'bash "$RESEARCH_RUN_DIR/inputs/probe_job.sh"'`, which refused with: "research run --on vy-nebius-2: refused: machine 'vy-nebius-2' belongs to project(s) ['verity'], not 'training'; refusing to touch it".
- Your 3:01 PM and 3:19 PM messages reached me only at 3:30 PM: my VM slept through the wait, so I launched at 3:30 PM and not earlier.
- The fix is `--project verity` and nothing else: the same node, one card, a lease of at most 5 minutes, 16 GB, and the same job and script as the CPU check (r20261112-213411-af3a). On your word I launch it once and send you the run id at once.
- Meanwhile, READ: the every-word run at `1178eb5439` (r20261112-212456-3c41, still running) has every dQ, dK and dV word of all 32 deterministic heads equal to the card's. The old-head runs ended with rc 0. r20261112-194241-5573 has every word of all 32 deterministic heads equal (2,097,152 of each gradient), and r20261112-192536-b84b ran "59 suites: 59 passed (1 cached), 0 failed".

[2026-11-12T23:32Z] f62038c191fc
WALKER, TRN-145, 23:40Z. Head f663a32157 pushed (53c7b497f2 + 1(a)'s test).

1(a) Rows compared by words. READ: points.Walk.rows, `keys = concatenate([member, v]); np.unique(keys, axis=0, return_inverse=True, return_counts=True)` over the uint64 words; no difference, no tolerance anywhere in rows(). RAN (local pytest -k): test_rows_of_one_value_that_are_inf_at_the_point_are_refused_as_equal passes: SelfCancel's looked-up rows times 2^127 twice are ±inf in every word, refused ("SelfCancelInfStep_v1", "x", (0, 1)). A revert comparing rows by float64 difference (compare-by-difference in my revert set) is added; it runs in the next revert job, result to follow.
1(b) A NaN made downstream of an inf: READ only. rows() tests every f32 operand row of a claim for a NaN before comparing, whatever made it, so inf - inf reaching a claim is refused there; a NaN that stops before any claim's row is not looked at. No test of its own (test_a_row_with_a_nan_is_refused makes its NaN from a square root, not from an inf).
Decision 8 gains "an inf row is a row" (spec note and points' docstring).

2/3. Centre proposal: 2^-3 for θ's and registered inputs' words (exponents 122-125 for f32/bf16, 3-6 for e4m3), four binades, inputs positive at 1 as now, a module constant, no dial.
Reason, from the operand magnitudes (RAN r20261112-230534-22f1): the only inf made from finite operands at centre 1 is SiLU's F32ExpRn (SiluRefElem node 2, from TrainStep nodes 244 and 337), its operand |x| up to 464 against exp's overflow at 88.7. That operand is a product of two θ-scaled matrices, so it goes as the centre squared: about 464, 116, 29, 7 at 2^0, 2^-1, 2^-2, 2^-3. 2^-3 is the first centre whose lower neighbour 2^-2 is also below 88.7 with room (29), so a binade either side holds; below 2^-4 nothing changes.
Sweep, θ-only centre, inputs at 1 (RAN r20261112-231016-77b0; NF listing r20261112-232730-d616):
| centre | origins n2/n3 | nonfinite f32 nodes | NaN nodes | max abs n2/n3 | claims | refused | N |
| 2^0 | 4/4 | 16 | 0 | 1.4e38/2.6e37 | 23 | 0 | same |
| 2^-1 | 4/2 | 16/12 | 0 | 3.2e37/1.8e38 | 23 | 0 | same |
| 2^-2 | 0 | 8 | 0 | 6.6e15/6.3e17 | 23 | 0 | same |
| 2^-3 | 0 | 8 | 0 | 7.9e17/3.5e17 | 23 | 0 | same |
| 2^-4 | 0 | 8 | 0 | 1.4e19/8.2e18 | 23 | 0 | same |
| 2^-6..2^-12 | 0 | 8 | 0 | ~1.4e19/8.7e18 | 23 | 0 | same |
POINT, smallest case: "no inf in any f32 node" holds at no centre for lf. The 8 nonfinite nodes left at 2^-2..2^-4 are the same at every centre: TrainStep nodes 99 and 167 (F32Add, 480 of 1024 words -inf, the second operand already holding those 480) and SoftmaxRef node 15 (F32Sub, x - max), in both loops. That is the causal mask's -inf, which the Program states as a value (480 = the strict upper triangles), not made by the point; exp of it is 0 and nothing downstream is nonfinite. So I read the criterion as: no inf or NaN made from finite operands (origins 0), and 2^-3 meets it with a binade either side. If you read it strictly, no centre passes and the fallback holds.
Unless you say otherwise, I commit 2^-3 now and run both suites, the reverts, the count/dump and the gate at it; if you rule the fallback I revert that commit (a revert commit, no rebase) and add the README limit. The draft opens by 02:00Z either way, at whichever head you rule, or at the 2^-3 head with this point open if I hear nothing.

