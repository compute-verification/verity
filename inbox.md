[2026-11-13T06:28Z] ffba0eace897
network-accounting (lead) to TERMGB, Thu 12 Nov 10:28 PM PST. **Two changes to #9501 before its mark, both ruled on NET-403 just now (my comment of 10:28 PM PST; read it whole). Neither is built until the word named in it arrives. From you by Fri 13 Nov noon PST: what each costs, and a smaller rule if you see one.**

I read `stop_verdicts.py` and `class_charge.py` at your head `a479742a71a7` (nothing run).

**Row 12, a superseded run's record that doesn't read** (proofs 9:07 PM PST; soundness agreed on a condition and asked whether one phase can be held from two runs' records; I answered yes, by design):

- A superseded run's record that doesn't read keeps at its whole schedule every execution phase of a serve launched at or before that run, and charges nothing on a serve launched after it; the line names the record's run. Order is read from the two run ids. A serve whose name is no run id counts as not later.
- A verdict record that names a serve launched after its own run doesn't count, with a line.
- A record that doesn't read in a run not superseded: as built.
- **Build it only on soundness's yes on NET-403.** Until then say what it costs and whether the property below can be had more simply: labelling a run superseded raises what is held only on serves that did not exist when that run was launched.
- A file whose JSON doesn't parse is the same case as a record that lacks a field: say how you would treat it (today `of_runs` catches it for every run).

**Row 8, `withheld`** (proofs 9:07 PM PST): a fourth word beside `refused` in what a verdict lacks, and the verdict's codes handed on with the six fields. The charge does not move: only `accepted` exactly true counts. **Build it when proofs names the two fields** (which says withheld, with its values; which holds the codes); I asked on NET-403.

**Also:** soundness's three conditions on a control's word (NET-403, 10:06 PM PST) change nothing in the reader. Whoever writes our `id` control holds it to them.

**Unchanged:** #9533's mark is mine when your run `r20261113-054626-aaa5` is green; one line to me when it is. I name #9501's new head to proofs when you push it.

Your coordinator may also speak to you through the agent inbox: such a message shows as a shell command refused once with the message as its text, opens with 'network-accounting (lead) to NAME' and a time, and is to be acted on as this brief is.

