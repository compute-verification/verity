# KWALK's inbox (the lead: @kernels, bc-026b7d5e-6674-5e9c-9eef-4aac970acc3b)

Read `inbox/RULES.md` on this branch first, then this file. The newest section wins.

## Thu 12 Nov 21:08Z: your brief: Friday's window on node A's card 6 for the head block with the level walk (HASH-112), from KHSHARE's handoff. Tonight: read it, rehearse it, read five Lean verdicts. No card before the go

**Who you are.** KWALK, a new worker of the kernels lane on `compute-verification/verity`, on the model
`claude-opus-5-5-high` (Daniel's rule of today: Opus 5.5 does the work). You run a prepared job that
the worker KHSHARE (bc-0242b8c1-6216-52c0-bd2b-285078e870aa) designed, ran on three cards today and
wrote down for someone who was not there.

**What it is.** Flock's prover builds the deepest part of the head block's witness on the host; the
"level walk" builds it on the GPU, bit for bit the same (HASH-112, goal `kernels/witness-on-the-card`).
Today it was timed on three H200s at once: four provers 2.097 → 1.592 GPU-seconds a statement, two
provers 2.182 → 1.587, one prover 2.629 → 2.440, two provers with proofs' keyed salts 1.660 → 1.163
(22 rows, each passing its ten checks). **Friday is the repeat on a quiet socket: card 6 alone,
cores 72-81, Fri 13 Nov 15:45Z to 19:45Z, nothing timed before 16:15Z** (proofs holds card 7 and
that socket's other cores until then). proofs prices its cost model from these rows for Daniel's
top goal, so the window has one try.

**The handoff is the brief:** `~/.research/notes/lanes/kernels/20261111T2030Z-khshare-prover-walk/FRIDAY-HANDOFF.md`
(after `research notes sync --path lanes/kernels`; 609 lines, sections 0 to 11). Read it whole,
section 0 (the rules that bind the window) and section 11 (what is not verified) twice. Use
`WORKER=KWALK` and `INBOX=KWALK` where it says `<YOUR-NAME>`. Where this file and the handoff
differ, this file wins; where a newer section here and an older one differ, the newer.

**Tonight, with no card and no lease:**
1. **The rehearsal of section 10**, on stand-ins and CPU nodes as it says, until it holds on
   your VM: `fri-rehearse.sh` and the launcher's cases, with `fri.sha256` checked first (37
   files). Change no job file. If a pinned file must change for the rehearsal to hold, stop and
   say which and why: the pins are what Thursday's runs and Friday's go rest on.
2. **Section 11's list, turned into what you will watch in the window's first ten minutes:** the
   node script has run on stand-ins only since its two fixes (the fit look judges memory only;
   four provers get two warm sessions); the launcher's three look refusals are unmet; the
   reader's own put of a kept proof has only run dry. For each: what you expect to see in the
   log, and what you do if you do not.
3. **The five Lean verdicts of Thursday's kept proofs** still waiting on `vy-nebius-cpu-12`
   (`r20261112-191526-4a95`, `…-191607-530e`, `…-192137-6432`, `…-193046-30e0`, `…-194304-84f2`;
   the sixth, `…-190939-3668`, is accepted): read them with `bash lean-verdict.sh RUN…` (exit 1 on
   a refusal, 3 while any is not in). When all are in, or at your return, one worker comment on
   HASH-112 with the six verdicts (`research msg linear comment HASH-112 --file F.md --worker
   KWALK`). **A refusal: post it and end your turn at once.** Send no Lean run again.
4. **Questions.** Anything in the handoff you cannot follow or that its files contradict: list
   it in your return. KHSHARE can be asked once more through the lead.

**Return by Fri 13 Nov 04:00Z** with: the rehearsal's verdict and what it ran; the watch list of
item 2; the Lean verdicts so far; your questions. **If the rehearsal has not held by then, say so
plainly**: the lead then gives the window back to KHSHARE. You are resumed at about 15:25Z on
Friday for the go.

**Not yours:** any card, lease or run on node A before the go; cards 2 and 3 on Friday (another
worker's rows); the walk's branch (it stays at `f9fc32ff3fc4`); any label; any pull request.
