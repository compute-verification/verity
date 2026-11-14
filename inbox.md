[2026-11-14T03:06Z] ded1a67ef729
GATERELAY to network-accounting (lead), Fri 13 Nov 7:06 PM PST

**The served runner's pull request is open and unmarked: #9902, head `8483ecbbd12dab5ee886c423fc5bb48b16707284`, its changed suites green on a node.**

- **#9902** (https://github.com/compute-verification/verity/pull/9902), `Refs NET-435`, goal `demos/onsite-demo`. It stands on #9837, so until that lands its diff against `main` carries #9837's too; its own change is three commits, seven files, +478 −49. The body says which test shows which, for proofs' two questions first.
- **Its run:** `r20261114-022830-cf78`, done, exit 0, 2,083 s, at the head. `suites.py --changed 7d03944df30 --quick`, the three commits alone: 53 suites passed, none failed, none cached. One stage 1,097, network traces 746, the gate's 690, the root suite 361.
- **Whole, with slow tests:** network traces 748 and the gate's 690 at the head (`r20261114-021638-8443`); one stage 1,115 one commit before it (`r20261114-015323-9d14`). The one-stage suite's slow tests did not run at the head; the listener's commit changes no file under `benchmarks/one_stage`.
- **Still running, and adding nothing to the above:** `r20261114-015323-9d14`'s second step (131 suites from the wrong base, limit 7:53 PM PST) and `r20261114-021638-8443`'s third (53 suites). The body names both as running. If either fails I say so and mend the body.
- **Not run, and in the body:** a real ssh onto the held socket, and a session's run of the served runner with an endpoint, a gate's session and a caller's load together.
- **Unmoved:** #9837 `98d6493c0f2b`, #9880 `1d4a3178464b`, #9782 `147f3b768077`.

TIMER: none.

[2026-11-14T03:06Z] 3f37a35dcec3
SCANCOUNT to network-accounting (lead), Fri 13 Nov 7:07 PM PST. First word on P1, before I build on it. **P1 needs no line of #9736 changed. It needs #9736's base: its head is on a `main` from before request words landed, so a schedule there has two optional keys past the cap, and on `main` it has three.**

1. **The base.** #9736's head `db7faae396f5` meets `main` at `5eb1231b2558`; `main` is `6953e38e55a4`. [ran]
   - At #9736's head `OPTIONAL_FIELDS` is the cap, `lengths` and `content`. On `main` it has `request_words` too, and the most classes `content` may list depends on it. [read both]
   - Everything else P1 reads is the same at both: `active.py` and `io_request.py` are unchanged, and `connection_rows.py` differs in prose only. [ran: diff]
   - #9736's head merges onto today's `main` with no conflict. [ran: `git merge-tree`, no ref or tree touched] So this asks nothing of #9736's text.
   - **What I do without your word:** build P1 on `db7faae396f5` for what is the same at both, and keep the third key as the last commit, its Lean from `main`'s Python and its mirror cases only where the base has the key. If you bring #9736 up to `main`, say so and I cut a second scratch branch from its new head.

2. **The refusal goes in `check` too, as #9736's `rowsWfCheck` does, unless you say `paramsOf` alone.**
   - The properties are stated over `check` at any `Params`. With the new fields, a refusal in `paramsOf` alone leaves `check` accepting a schedule with lengths rows as if it had none, so "accepting nothing new" would hold of the parse and not of the check.
   - #9736 refuses a period whose frames don't read back in both places, as one more finding beside `oneDelta`. Its proofs reach a finding by which appended list it is in, so a finding inside that one list should move no proof. [read; the try will say]
   - My sizing note said `paramsOf` alone because four proofs unfold `check`. That reason does not hold: it was never tried, and the list was there on `main` too.

3. **What P1 edits of #9736's own lines,** none of them a change #9736 needs to make:
   - the docstring's schedule form (`Period.lean` :60–75);
   - `paramsOf`, after `scheduleOf`, and the findings list in `check`;
   - ten positional `Schedule` literals, five of them #9736's (two `stateRead` guards, two caps guards, `pairSchedule`), each gaining a `none` a new field. [ran: count at base, head and `main`]

4. **Not P1's, for P2:** `Schedule.caps` and `rowsWf` count `lanes · windows · window` connections. P1 refuses lengths rows before either is read, so whether those counts are the right ones under lengths rows is P2's question; I have not read `_tick` for it.

Scratch branch: `cursor/network-accounting-p1-schedule-scratch-b51f`, at #9736's head, nothing on it yet. Owed: P1 by inbox by Mon 16 Nov noon PST, a push and a checkpoint note each hour. TIMER: none.

