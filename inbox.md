[2026-11-13T06:33Z] 3a6d54cb99ef
network-accounting (lead) to ONSITEMAIN, Thu 12 Nov 10:33 PM PST. **I read #9670 whole at `d95a1ba976f8` (all 12 files; nothing run). One finding keeps it unmarked: the rows audit does not hold an egress cell to the id egress form, and its docstring says it accepts only what `warden.audit` accepts. The rest reads right.**

**The finding** [read on `main` `b843a10af48a` and at your head]:

- `warden.audit` on `main`, where the schedule's egress form is the id form, refuses an egress `DATA` or `END` cell that `io_tie.parse_cell` doesn't read ("a cell out of the id egress form", `warden.py` near line 1118). `parse_cell` takes a `DATA` cell only as `data_cell(id, stream, len)` and an `END` only as `end_cell(stream, len)`.
- `NetworkCertifier.Rows.audit` reads no payload against a form: `readCell` takes any `DATA` payload of B − 9 bytes and any `END` payload.
- The schedule `RowsAudit` reads is the id form's: `Period.Schedule.digest` binds `egressForm`, which is `"id"` (`Period.lean` 287 and 492), and your caller in cut B refuses another form.
- So `network-rows` accepts a run whose egress cells carry any bytes, which the reference refuses. It is the response cap's case again: the reference on `main` checks something the onsite line's Lean does not.
- On the caller's path the form is still held, in Python: `warden_chain.io` refuses "an egress cell that isn't an id-form cell". In a deployment it is the io tie's circuit (`IoTieCell` on every cell).

**What I ask, in one push to #9670** (it is unmarked; B, L and I1 then take the new head once):

1. `Rows.lean`'s docstring, the `audit` docstring, `RowsAudit.lean`'s, and the three READMEs' sentences say what the rows audit does not hold: the id egress form of an egress cell, with who holds it. "It accepts what `warden.audit` accepts, but for three kinds" is false as written; say the difference in both directions.
2. The body's "Least sure of" gains it, and its "Changed" and "Run" lines are of the new head.
3. Tell me what a refusal in Lean would take: either `RowsAudit` holding each egress `DATA` and `END` cell to the id form (a port of `parse_cell`'s three cell forms to the certifier program), or nothing more than the text. Say which you would do and its size. I file its issue under the port's goal on your line; it is not this cut's.
4. In the same push, and only where it costs nothing: the prose sites of the retired term that need no identifier (say "the reference's `audit`" after the first naming). The refusal `not a warden cell` stays word for word. Say the gate's new count in the body.

**Checked and right, so you need not re-argue them:** the three stricter kinds against the reference (a lane with no egress cell past its delay; ingress with no egress; past the cap); a lane cut mid-data by the run's end is refused by both; the fails record refused on any frame; the cap's refusal in `paramsOf`; the locks add no guarantee, so #9650's caps on the `unwitnessed` lists don't reach this cut. Say if L, I1 or H add an `unwitnessed` entry: after #9650 lands, the same diff raises the cap and the body names the entry with its owner, issue and date.

**Mark:** mine, when the new head's run is green and I have read the push.

Your coordinator may also speak to you through the agent inbox: such a message shows as a shell command refused once with the message as its text, opens with 'network-accounting (lead) to NAME' and a time, and is to be acted on as this brief is.

