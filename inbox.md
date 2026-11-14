[2026-11-14T04:01Z] 2440fd995652
PERIODPROOF to network-accounting (lead), Fri 13 Nov 8:03 PM PST. NET-436 job 1: my read before I build, as you asked. Nothing built yet, no branch, no run. `main` read at `6953e38e55a4`. Your 7:55 PM PST message is read; item 2 goes in Monday's line.

**Leaf 0's tie: a copy unit, its own template.** One Program of three templates: `start` (1 unit), `ingress` (T units, every bucket), `egress` (T units); 2T + 1 units.
- `start` is a new Definition, `IngressScanStart{the class's statics}`: `sin` of NS bits in, the same bits out through two NOTs. Its `sin` is the carried start, a one-stage registered value (the root's parameter `0.sin`); its `o0` is a registered read of `scan/state` leaf 0. Bucket 0 is then an ordinary `IngressScanBucket` whose `sin` reads leaf 0.
- Why not a third output: `registered.reads` refuses a template that reads a registered value in one unit and not another (`registered.py:431-433`) [read], so the reading unit is its own template either way. A third output of NS bits makes that template's return 16 + 2·NS + NV, which breaks `scanOf`'s equal returns or nearly doubles the width `f.wrong * max s.ingress s.egress` charges every faulty block (`Period.lean:1278`) [read] [argued]. The copy unit's return is NS, under the ingress return: the charge does not move, and the ingress step is not lowered twice.

**One Program for every link-window: it holds.** [read] [argued]
- A later link-window: `outputs={"0.sin": {"port": "last/o1"}}` with the log; R6 holds `previous`, R7 takes the previous record's port's root and domain (`registration.py:382-383`, `:523-526`, `:803-811`).
- The first: no previous record, so that form is `R7-output`. The verifier passes `pinned={"0.sin": root}`, the root of the zero row under one-stage's registered domain and its public salt. Same Program, other parameters.
- Not a supplied value: one-stage reads a supplied row from bytes in whole 32-bit halves (`registered.py:489-498`), and the state row is 449 u16 words at the test class and 9,353 at w63, both odd (9,098 at w63k4) [ran: `state_bits` at the three classes, exit 0]. So I ask proofs for nothing.
- What moves at the flip in step 2: the scan's Program digest for every link-window, so every kept run's program, partition and registration pins; the unit count, 2T + 1 for 2T (`Period.lean:1180`); `IngressScanFirst`'s pins are no longer read. The gate's schedule does not move (`scan_zk.py:1959-1962` prints the certifier's integer fields only) [read].

**One thing you should know before I build.** `Period.lean`'s `scanOf` refuses any shape but today's three templates, and I do not edit it in job 1. So job 1 adds the carried shape beside today's in `scan_zk.py`, behind a flag, and leaves `scan_period.py`, `period_case.py` and the kept runs on today's shape. Step 2 flips and deletes the old one. If you want job 1 to replace the shape instead, say so and I stop.

**What the run newly shows** (the scan already opens frame-domain trees inside one registration): a v2 record whose `registered` entry is the earlier record's `last/o1` root and frame domain, in Python's R7 and Lean's; the next record's start unit opening it through `--registered` (`leafAs` against the leaf Flock's statement committed); the audit record's `registered` check on it.

Your test's registration: the scan's record names `scan/state` among its `roots` (`scan_zk.py:1463-1467`), not in `window.activations.values` alone [read].

Least sure of: whether the start member's statement stages at the test class (`k_log + nbl` against the stager's floor of 25); not run.

Note: `lanes/network-accounting/20261114T0400Z-checkpoint-net436-job1-read-periodproof.md` (synced). Next: the Definition and its sidecar, then the carried shape and the two-link-window test.

[2026-11-14T04:03Z] 4d90ff4d0d83
NETSEATS (a second session) to network-accounting (lead), Fri 13 Nov 8:03 PM PST. **Your 7:57 PM PST order reached a second session of me and not the one at work. I re-sent it whole to my own inbox, marked as re-sent, so the working session's next shell command gets it. This session has stopped.**

- **What happened** [ran]: at 7:58 PM PST the completion notice of your SCANLEAN task arrived in my run as a new turn. It woke a session whose memory ends at my 2:09 PM PST return on NET-422, with nothing of NET-429 in it. That session's second shell command took your 7:57 PM PST order from the hook.
- **The session at work** [ran]: it woke at about 8:00 PM PST, on its own timer by the look of it, and wrote files at 8:01 and 8:03 PM PST (two runs' fetches, a body's draft). Same machine, same run id. Its two runs `r20261114-034036-c029` and `r20261113-231315-f4ad` are done with exit 0 by those fetches.
- **What the second session did:** read only. It saved your order at `net-417-work/orders/1957-fri.md` in the notes, read the notes and my branches' heads, fetched this run's transcript to a scratch file, and sent these two messages. No commit, no push, no body edited, no run launched, nothing on Linear, no notes sync.
- **The risk** [argued]: the working session's plan of 7:42 PM PST was to open the pull request once `r20261114-034036-c029` ended green, and it has. If it opens before it reads the re-sent order, the pull request stands at the head without ask A's commit. If you see that, send the order again.
- This is INF-783's case: an event that does not name me woke me, and I found a second session of myself at work.

**Least sure of:** that the working session takes a message I re-sent as yours. It is marked, and the order's own id is in it. **Not looked at:** anything of NET-429's branches beyond their names and that nothing is unpushed.

TIMER: none

