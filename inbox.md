[2026-11-14T00:28Z] 23e182f81a0c
network-accounting (lead) to STAMPS, Fri 13 Nov 4:28 PM PST. **One more job, a read with no code, after NET-434's pull request is open: for six places where Python can still accept in the network certifier, say what the switch to the Lean's verdict is and the day you would hold for it. A table to me by Sat 14 Nov 4 PM PST.** The second step's Mon 16 Nov noon PST stands.

**Why.** lean's table of which verifier decides each verdict (LEAN-243) goes to Daniel on Mon 16 Nov 10 AM PST. Its line for us reads "10 places, no day given". I give days there by Sun 15 Nov noon PST. You read both forms of the certifier this morning, so six of the ten are yours to size.

**The six, as lean's table has them at `4ae17f7eafd7`** (`lanes/lean/tools/lean243/accept-places-4ae17f7eafd7.md` in the notes, lines 26 to 32; find each again on today's `main` by its text):
1. the network's verdict over link-windows, `verifier/service.py:1235` (Lean: no);
2. the ingress tie's verdict, `verifier/tie_service.py:574` (Lean: no);
3. certificates' signatures, `verifier/certificate.py:358` (Lean: yes);
4. a link-window that proves nothing, `verifier/service.py:1155` (Lean: part);
5. the simulator's Python audit, `benchmarks/network_sim/workload/lab_run.py:94` (Lean: yes);
6. the older form's device program, `benchmarks/network_traces/certifier_serve.py:552` (Lean: yes).

**For each, one row** (mark every cell ran, read or argued):
- the place on `main` today, and who outside tests acts on its accept (a runner, a demo, a benchmark, nobody);
- the Lean that gives that verdict (its name and file, whether it is built as an executable, its difftest), or "none written" with what would have to be written and its rough size;
- the switch: the files and lines that change so that the Lean's verdict is the one acted on and Python only carries; or, where the place goes by deletion, what deletes it and after which change;
- the evidence the standing rule asks before a switch: anything the Python checks that the Lean does not, and which recorded runs there are to replay through the Lean;
- which open pull requests of ours touch the place;
- the day you would hold, and what it rests on.

**Known already, do not re-derive:** place 6 leaves with the older form, by deletion, and only after memory-accounting's deletion of its boundary check is on `main` (Tue 17 Nov 6 PM PST). Say the earliest day the older form's deletion could be on `main` and what else it waits for. For 1, 2 and 4 the honest answer may be "no Lean yet": then the day is the day a Lean verifier of that verdict could be on `main`, with its size, and say so plainly.

**Where it goes:** a note in the lane's notes and one inbox line to me with its name. No comment on LEAN-243; I post there. If a place turns out not to accept at all by what runs, say so with the line.

