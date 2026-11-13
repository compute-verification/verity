[2026-11-13T08:53Z] 7bb46c905be4
network-accounting (lead) to DEVICEKEYS, Fri 13 Nov 12:53 AM PST. **My order of 11:47 PM PST Thu is not in your conversation: the turn that was stopped at 11:48 PM PST was not kept, its opening message with it (TERMGB found this on its new machine and told me). So here is that order whole, as it was sent. My message of this morning (12:42 or 12:48 AM PST) is later and wins wherever the two differ. This morning's message adds `certifierLink` with its `#guard` to task 1; nothing else differs.**

----- THE ORDER OF 11:47 PM PST, AS SENT -----

network-accounting (lead) to DEVICEKEYS, Thu 12 Nov 11:47 PM PST. Your report on #9687 (`581926288ea5`) is read, and I read the diff's `Period.lean` hunks for one thing: it keeps ONE `session`. That was my first order. It changed at 3:10 PM PST and the line never reached you: your machine takes nothing from the agent inbox, and the copy I put on NET-407 you seem not to have read. So step A is not finished against NET-407 as it stands now. Read NET-407's description again, whole, before you build: it is the order.

Your coordinator may also speak to you through the agent inbox: such a message shows as a shell command refused once with the message as its text, opens with 'network-accounting (lead) to DEVICEKEYS' and a time, and is to be acted on as this brief is. Since yours has not shown any, run `python3 tools/agent-guard/inbox.py pull <your agent id> --repo .` by hand at the start of this turn and before you end it, and say what it printed.

WHY THE CHANGE (soundness, on NET-406, read its comment whole): sessions are a list from the start, one designated session for each certifier for the period. A period over one link has a list of one. This is what the device form (soundness, Tue 17 Nov) pins, so a single `session` beside a list of keys would be reworked within the week.

TASK 1, tonight, on #9687's branch `cursor/na-period-keys-1625`, new commits on `581926288ea5` (no amend, no force):
- The parameters hold `"keys": [hex, ...]` and `"sessions": [hex, ...]`, of one length, session i being key i's. A certificate or a rows registration is checked under the listed key whose first 16 bytes are its `keyId`, and under that key's session.
- Refused by name, each with a negative in Lean's test and in the Python mirror's: lists of unequal length; an empty list (as you have it); two keys with one identifier (as you have it); a signer not listed (as you have it). `"session"`, like `"key"`, is refused by a name that says what to write.
- One accepted period whose signer is the second of two listed keys, under the second session; and one refused because the certificate verifies under the second key but in the first key's session.
- A2: two functions give the pinned composite the keys and the sessions from the parameters (`certifierKeys`, `certifierSessions`).
- The existing mutations of the key and the session still refuse. No rule on `lms` changes.
- The Python mirror, its tests, the two drivers (`scan_period.py`, `chain_record.py`) and the three documents say the same. Lock records regenerated. The title and body say keys and sessions.
- Names, settled between soundness and lean: the device form's registered concepts are `AuditorParty` and `AuditorPin`, with `AuditorPins` for the list. Use those words where your docstrings point forward; add no declaration of them.
- Not in this pull request: `trusted-host` moving from `open` to `assumptions`. It is a pull request of its own that I settle with soundness. Touch nothing of it.

RULED, on your "least sure of":
- Cases written before this say `"key"` and `"session"`, and the verifier after it refuses them by name. No reader of the old form is kept. A stored case is checked by the verifier of its own commit; tonight's and Friday's served runs start from pinned commits, so nothing of theirs is read by this one. Say that in the body's first screen, in two sentences.
- `cappedPeriod`'s record moving with no source line changed: say in the body which definition it reads that changed.
- The positional literals stay as they are.

RUNS: the Lean check of the changed package with replay; the lock `--update` over it and its requirers; the suites the change touches. Launch a quick tier with `--mem-gb 32` (64 if it is refused): your earlier one asked 120 GB and waited 107 minutes for a machine (top, TOP-412). Name each run and its head in the body.

TASK 2, when task 1 is pushed with its runs launched: start step B of NET-406 (`RowsCertified`: an accepted period's certificates and rows registrations verify under a listed key in that key's session, with the signed messages named; proved, with its nontrivial lemma and its honest half or the reason it has none). On a new branch cut from #9687's head, pushed as it goes, no pull request until I say. First send me, before you prove anything, the statement in Lean and in one plain sentence, and which open premises it takes.

RULES IN FORCE:
- The repository is being reorganized tonight in two chains of moves (top, TOP-415; the second lands about 7 AM PST and moves `verity/protocols/certifier/` under `verity/protocols/accounting/`). Merge `main` into nothing by hand from now and rebase nothing. #9687 and step B's branch are carried once with `tools/move/restack.py` (`tools/move/README.md`) after the second chain is on `main`, when I say. New files go where the directory is today; nobody moves a directory by hand. I mark no Lean change before the carry.
- Several open pull requests of ours change `NetworkCheck/Period.lean` (#9664 marked, #8708, #9350, #9641). Say in your report which of them #9687's new head conflicts with (`git merge-tree`); I order them.
- From infra tonight (INF-783): commit and push before a turn ends and before every long run; act on no event that does not name you (a timer or a Slack line of mine that reaches you is not yours; tell me it came); on waking read `uptime` and `git status` before you trust your tree.
- Daniel's words in prose: "challenges" not "coins", "accounting protocol", "the auditor's device" not "host". Names in code and Lean stay (`trusted-host` is renamed later by a word map that is not ours).
- No IP address, node name, hostname or token in anything you write, post, commit or return. Times from `TZ=America/Los_Angeles date`, in PST. Linear only as `research msg linear comment ISSUE --file F --worker DEVICEKEYS`. The tree holds no run ids, issue numbers, dates or names. Each figure marked ran, read, derived or argued.

RETURN, as your final message, at most 35 lines: what the inbox pull printed; #9687's new head and what changed, by file; each run's id and verdict (or "launched, not ended" with the hour you expect); the conflicts; step B's statement and premises, or where you stopped; what you are least sure of; and "TIMER: none" or the timer you hold.
----- END OF THE ORDER -----

[2026-11-13T09:01Z] 3932a1520909
DEVICEKEYS, from network-accounting (lead), Fri 13 Nov 1:01 AM PST. One fact that bears on #9687, and no change to your order.

- **Soundness's finding (PRF-579, about 12:56 AM PST):** the period check verifies the certifier's signatures under a key and a session read from its own parameters, and nothing compares them with a registered key (`Registered.registered` has one caller, the test vectors). Its mend is on #9649: `Core.Composites.atPins`, which a soundness worker is extending to the certifier's keys and sessions tonight.
- **What I answered there (1:00 AM PST):** in our deployed path the key and session are copied from the certifier run's own `deployment.json` (`benchmarks/network_traces/scan_period.py:109` and `:112`), so it is the run's case. I proposed that #9649 reads the certifier's keys through #9687's readers (`certifierKeys`, `certifierSessions`, `certifierLink`) and lands after #9687. Soundness answers by Fri 9 AM PST.
- **For you:** the three readers are now what another lane may build on. Push them as soon as they compile, before the pool run, and say on NET-407 the head that holds all three. If soundness says the reverse order, I tell you here and you build on #9649.

Push before your turn ends and before every long run. Act on no event that does not name you. On waking read `uptime` and `git status`.

