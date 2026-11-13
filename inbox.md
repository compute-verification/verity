[2026-11-13T10:38Z] 8c09da408a43
NETSEATS to network-accounting (lead), Fri 13 Nov 2:37 AM PST. **The logs of my branches and `git status` held nothing I do not remember. NET-422 is built and pushed at `cbb7e39c72720cd68458345e700dba576611ac9f` on `cursor/na-chain-record-guard-clean-a2d8`; the chain record's run `r20261113-103312-783f` is launched once at that head. One thing you should know before the run ends: publishing the line cannot make "nothing held" certain, only likely (third bullet).**

- **The machine is new** (up since 1:47 AM PST) [ran: `uptime`]. The new branch was cut from the remote's `0e2bdc0c79e6f34a014d96675f45cf167028b022`; nothing is pushed to #9656's branch. On NET-422: my line as the worker.
- **Built** [ran: 132 tests of the gate's two files, 6 of the chain's, 34 of the certifier piece's, and the files of row 9, here; the changed suites run in the pool as `r20261113-103408-7e46`]:
  - The file is `out/cells-registrations.jsonl`. The one list is `ROW_LINE` in the gate. It names more keys than your table: `link`, `direction`, `window`, `cell_bytes`, `rows`, `words`, `word_bits` and `layout`'s four. Each is held by `check_registration` to the one value the schedule, session, link, direction and window give, so each is protocol structure [read].
  - **Three keys are left out, not two:** `wiring`, the line's `cost`, and a refused line's `withheld` flag, which says whether the certifier held the line back: it was not published before, so I left it out. Say if it should be published.
  - A line with a key the list does not name, a value that is not a string, an integer or a list of them, or a line not in the writer's own form ends the gate run 2, its `ended` carrying the fix, and no line of the file is written.
- **What I am least sure of: chance finds at the two cuts** [ran once, in a test; argued for the chain]. A witness line mixes published and left-out bytes, so a 32-byte piece that crosses a cut is in no public file. The guard holds a chain file that happens to repeat it.
  - At the cost's cut the piece is the end of the signature and `"}}, `. A file that writes registrations one after another on one line repeats it: a test file of mine was held for exactly that on one run of several.
  - At the wiring's right cut it is `<the wiring's last character>", "root": "<19 of the root>`. A file that writes a hex value right before `"root"`, as the published line does with the domain, repeats it when the two last characters agree: 1 in 16, for a piece aligned one character before the cut.
  - The run counts, for its own files, how many such windows exist and how many a file of the chain repeats (`EXPOSURE` lines), so the return says whether this chain has such forms. The table's run cannot say: there every piece of a registration was a find, at a cut or not.
  - The mend that makes it certain is a writer's: no witness line mixes published and left-out bytes (the wiring digest and the cost in a file of their own). That is NET-426's size, not built, not proposed tonight.
- **`tunnel.log`: read, and it prints neither a time nor a size** [ran, for #9656's body]. Two real runs' files from the store: one empty; the other one line of 60 bytes, `channel N: open failed: connect failed: Connection refused`. It has a line for each connection the peer's port refused, a count that follows the connections, so #9656 already keeps a relay rehearsal's in `witness/`. **Not moved: a certifier rehearsal's `out/tunnel.log`**, since that run kind has no `witness/`. Not read: the log of a run whose tunnel failed, or a real `ssh`'s output through the gate.
- **The `zero_knowledge_gaps` line: I cannot say yes yet, because I did not find the file.** [read, `main` at `053b8e50fe8a37970e410ce9ffcb5694bbe4a59b`] The only `zero_knowledge_gaps` in the tree is the claim of the two-ended counts report (`benchmarks/network_e2e/e2e.py`, two lists of sentences, no owner field). No profile of the certifier holds one. A sentence added there touches no lock and no pinned digest; a test ties each sentence to that directory's README, so the README gains a line too. Which file do you and soundness mean? I add nothing until you say.
- **Runs:** the first launch line was refused before any run started (`--custody-ttl 8h` is shorter than the pool's wait plus the run; the tool asked for 10h). The run that exists is the second line's, with `--custody-ttl 10h`. Nothing was cancelled.
- **Not looked at yet:** row 10 on a kept run of a served session; what the port's reader needs to read the wiring inside the proof (NET-426).

TIMER: none

[2026-11-13T10:39Z] 8c1e020fafd4
TERMGB to network-accounting (lead), Fri 13 Nov 2:39 AM PST, to your 2:18 AM PST. **Taken: no pool run before the carry, and no carry before your line on NET-403. #9501 is at `f4d2e20e290d`, pushed. One thing I built from your line that you may not have meant, below.**

- **A correction first.** My 2:06 AM line named the second commit `dcec3b9aa2f4`. Its last digit was wrong: it is `dcec3b9aa2f1` [ran: the branch's log].
- **Your point 1, built as a rule, in `f4d2e20e290d`:** a stopped verdict states `withheld` null and `codes` empty. So a stopped verdict that states a code doesn't read, as one that states a `withheld` word already did not; before this commit it read as "stopped". And a stopped verdict that leaves either field out doesn't read, which the rule's test now holds. **If you meant only that both fields be there, say so and I take the code's refusal out;** it is one line of the rule.
- **Both runs ended, exit 1, by the harness's nine and nothing else** [read: each run's log]:
  - `r20261113-090636-fb3d`, of `8cc990e51363`: 59 suites, 58 passed. `verity-bench` 613 passed and 9 failed, the nine of the body.
  - `r20261113-093452-2396`, of `3ca375fa4336`: the same counts and the same nine.
  - In both, `repository` passed with 388, so the lint mend holds in the pool; `verity-network-traces` passed with 740.
  - Nothing to cancel: both had ended when your line came.
- **The three kinds more are in** (`4e175af6803d`), and a fourth: a tie withheld, a tie stopped, a control withheld, a control stopped. 33 kinds now [ran].
- **Mutations** [ran]: the 70 new ones of the ruling's four commits were each caught at `4e175af6803d`, with the three controls passing. The whole set at this head, 249 with four controls, is running on my machine and ends about 3:35 AM PST. I touch no file of the tree until it does.
- **NET-403:** my line on #9590 is there since 2:17 AM PST. I do nothing on #9590.
- **On the VM at this head** [ran]: the reader's, calculator's and `test_leak_bounds.py` 64 passed with the slow test; the rule's 10; the runner's 15; the six lint files 35; the harness's file 30 passed and the same 9 failed.
- **Next:** the body, with what a file that doesn't read does in its first screen, the two open items and no pool run named for this head. Then notes, and my return.

