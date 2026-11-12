# KQUANT's inbox (the lead: @kernels, bc-026b7d5e-6674-5e9c-9eef-4aac970acc3b)

Read `inbox/RULES.md` on this branch first, then this file. The newest section wins.

## Thu 12 Nov 19:38Z: your brief: the int8 row quantizer's exhaustive test run again on an H200 (sm_90) and on an RTX PRO 6000 (sm_120), at the tree lean's merge lands. Two short card runs, each with a hard gate on when it may start

**Who you are.** KQUANT, a new worker of the kernels lane on `compute-verification/verity`, on the model
`claude-opus-5-5-high` (Daniel's rule of today: Opus 5.5 does the work).

**Why.** lean's merge (#9375, `cursor/lean-remote-merge-onmain-fc8d`) brings to `main` a new int8 row
quantizer in the vLLM adapter (`adapters/vllm/verity_vllm/engine/int8_fused.py`: it divides by a row
reciprocal with two FMA corrections; kernels' old #5961 and #6029). It serves on every CUDA device.
Its pull request claims the same result as the torch body on every finite bf16 input. top asks
kernels (LEAN-249, Thu 12 Nov 19:35Z): **rerun that claim on sm_120 and sm_90 by Fri 13 Nov noon
PST, or say what holds it.** Issue: HASH-126 (report there with
`research msg linear comment HASH-126 'TEXT' --worker KQUANT`).

**The two tests** (`adapters/vllm/tests/engine/test_int8_ops.py` at #9375's head; both need CUDA):
`test_the_quantizer_divides_as_div_rn_on_every_bf16_input` (about 4.3e9 elements, "seconds on an
H200") and `test_every_quantizer_plan_is_the_torch_body`. Run exactly these two with pytest, at the
head of `origin/cursor/lean-remote-merge-onmain-fc8d` of the minute (say the sha; if #9375 is on
`main` by then, at `main`). Keep the whole pytest output, the device's name and compute
capability, the commit, torch's and triton's versions.

**First find out how an adapter GPU test is run on a node** (do this on your VM and a CPU node, not
on a card): the adapter's README and `pyproject.toml` (its `torch` extra), how vllm's lane ran
`tests/properties/fp8_cache_replay_gpu.py` on node A's card 5 today (run `r20261112-113028-b56a`:
`research status`, `research fetch` its launch record), `benchmarks/harness/setup/`, and
`benchmarks/pouw/pearl_c_vllm/served_step.sh` for the RTX PRO 6000 node. The team skill says where
heavy work runs and how a job takes a card (`gpu-lease`), the testing skill how a suite's
environment is made. If the node needs an environment built, say how long it takes before you
launch anything.

**Run 1, sm_90: node A (`vy-spot-glm53`), card 6 only, cores 72-75, TODAY between 19:50Z and
20:55Z** (kernels' row in `~/.research/notes/calendars/node-a.md`; read it in the launch's
minute). One lease of at most 10 minutes; start only if card 6 holds no lease and no compute
process; touch no other card. KHSHARE's timed rows run on cards 4, 5 and 7 beside you: nothing
heavy on any core (no build of more than a minute on the node). **Launch only if you are ready
by 20:30Z. If not, launch nothing today, say so in one line on HASH-126, and it runs in Friday's
window (15:45Z) on the card I name.** proofs and vllm take the whole node at 21:00Z: off by
20:55Z whatever happens.

**Run 2, sm_120: `vy-nebius-2`, ONE card, TONIGHT, and only in one gap.** kernels' worker KP2 runs
three eight-card leases there (T, then F, then C) and posts each release on HASH-113;
memory-accounting has the node from 22:15Z; the node stops at 00:00Z and is back on Monday.
- **Launch only after KP2 has posted C's release on HASH-113** (or its line that the chain has
  ended), **and only if you can be off the card by 22:10Z.** A lease taken between two of KP2's
  leases would make its next one fail: never ask for a card before that post, whatever the node
  shows free. Nothing of yours runs on that node before it either, not even a build.
- One lease of at most 8 minutes, one card. Touch neither auditor's host.
- **If that gap does not come, or your environment is not ready: launch nothing, and say so.**
  Then it is Monday's, and I tell top what holds it.

**If a test fails on either card:** that is the finding. Keep the assertion's line (the max, the
warps, the rows), post it on HASH-126 at once and end your turn: I ring vllm, lean and top.

**Return** when both have run or been ruled out, by 22:20Z at the latest, with: each run's id,
the card, the commit, the two tests' verdicts and seconds, and what did not run and why.

## Thu 12 Nov 20:08Z: lean's #9375 is MARKED at `bafae2eaec2e5d6e04569b75c31d422b8c015d8c` (20:05Z) and does not move again until it is on `main`

lean (LEAN-249, 20:06Z): "From now until it is on `main` nothing is pushed to it; a fix is a new
pull request." So the head of `cursor/lean-remote-merge-onmain-fc8d` to build on, search and test
at is `bafae2eaec2e`: merge it once now and say that sha in what you report. Its candidate's check
is the first Lean build and the first suites' verdict on that tree, so it may still come back red;
if lean says so on LEAN-249 I tell you here. When it is on `main`, your base is `main`.
