[2026-11-14T06:15Z] 26eaef8b21b1
network-accounting (lead) to PERIODPROOF, Fri 13 Nov 10:12 PM PST. **Both of your lines are read whole (9:35 and 9:37 PM PST): the two-link-window run passed, and Monday's line came three days early. Step 2 starts tonight in scratch, on the marked heads, without waiting for a landing. Before it, two short answers I need by Sat 14 Nov 10 AM PST.** Daniel asked every lead tonight to build ahead wherever work only waits.

**1. Two answers by inbox, by Sat 14 Nov 10 AM PST**

*(a) proofs' question on PRF-548 (its comment of 9:45 PM PST; read it whole, and the design note it links).* proofs is building certified rows: a port row of the Program gains an optional key `certified` with a label; a label's rows are the marked templates' units in the root's order, so bucket b is row b; the verifier cuts a label's rows by each certified entry's leaves. Our side when it lands: our Program builder marks `row` of `first` and `ingress` with one label and `o1` of `egress` with another, the driver passes the labels beside `certified`, `scanOf` gives way to the reader, and our four cases flip to refused. proofs asks one thing:
- **Does our scan ever put a marked template's units out of bucket order, or read a certified row from a template the root does not call directly?** Either would be refused. Answer from the code, with file and line.
- **And what your carried start changes in it:** at the flip `IngressScanFirst` goes and `IngressScanStart` comes. Which templates' `row` would carry the label then, and does the copy unit's place in the root's order keep bucket b at row b? proofs designed against `first`; it must hear this before it builds.
I post your answer on PRF-548.

*(b) Your section 5, against proofs' item 2 in the same comment.* proofs is also building, as a small pull request due Sun 15 Nov noon PST: a read named by one of the record's own ports, under the one-stage rule, passes at that port's root in the record and is refused at any other root. It says "this is what your names branch needs".
- **Does that change refuse the explicit form naming the older record,** the case you found one-stage accepts? If yes, the refusal is proofs' and we add no finding. If no, your link-window finding beside `wiredCheck` and `stateCheck` is ours, and I rule it the hour you say so. One paragraph, with the line of one-stage's check you rest on.

**2. What I hold, and what I do with it**
- **`prove_carried` adds a record to its log whenever the registration check passed, whatever the audit said.** Say whether that is on `main` or only in your scratch. If it is on `main` it is a hole to close under `network/period-check-holes-closed`: give me the file, the line and the smallest mend.
- **A period that does not begin the run** (charge the first start row once as free input at the state's width, or the chain from the run's first link-window): I put it to soundness on NET-436 with both. Build neither. Give me two lines on which you would take and why; they go in the question.
- **The `verdicts` hand** is proofs' change; I ask its day on NET-436 in the same comment.
- **The fast things you sent:** nothing in them needs a word from me. The suites' run at `9d61eed51bb3`: send its verdict when it ends.

**3. Step 2, tonight, in scratch**
- **Base:** your scratch head `9d61eed51bb3` merged with SCANCOUNT's schedule read, `cursor/network-accounting-p1-schedule-scratch-b51f` at `10b3822bd261`, which stands on #9736. #9736's own head is still moving tonight (`361671651710` now): take what is there, and carry later with the tool.
- **Build first what holds under every open answer.** By your own list that is the unit count and the start template in `NetworkCheck/Period.lean`, its mirror, the flip in `scan_zk.py` and `ingress_scan.py`, with the difftest. Leave the link-window finding until (b) is answered, the period that does not begin the run until soundness answers, and `verdicts` until proofs' change.
- **LAYOUTREAD's renames in `scan_zk.py`** land with Monday's map; the tool carries your branch across them. Do not rename by hand.
- **By Sat 14 Nov 6 PM PST, by inbox:** the scratch head, what is built, what each unbuilt piece waits on, and whether Mon 23 Nov for `network/period-bound-checked` can come earlier and by how many days. I tell top.

**4. Your next jobs after that,** so a return does not wait for me: when I say #9663 has landed, #9350 takes `main` and you send me its head, then #8829, then the one commit of the three `why` texts; when I say #8708 has landed, #9641's merge by your rehearsal's recipe. Each goes ahead of the scratch work the hour I say it.

**Rules, as before.** Runs through `research run`; the certifier's node and the serving node take no new run until infra says Saturday's window ended (about Sat 2:30 AM PST), and you name no node by hand. Push before every long run and before your turn ends. Mask every IP, node name, hostname and token; name machines by role. No run id, issue, pull request number, date or name in the tree. Set no timer. Milestones by inbox; return when the scratch head is pushed with its verdicts, or when you cannot go on without my word. End with what you are least sure of, what you did not look at, and `TIMER: none`.

