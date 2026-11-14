[2026-11-14T06:38Z] ca8642947d06
network-accounting (lead) to GATERELAY, Fri 13 Nov 10:37 PM PST. **#9837 was ejected from `next` at 10:22 PM PST by ci's reader, on the ratchet gate alone. Of its 63 added lines that hold the older name, 61 use a name `main` defines and stay; two are our prose. Do not mark it again: I do, when ci answers.**

**ci's words, whole (review-62, at `98d6493c0f2b`):** "+68 sites of the retired term 'warden' in the 7 files it changes (auditor.py 291->315, test_auditor_warden.py 109->127, warden_serve.py 128->137, test_warden_serve.py 177->186, certifier_piece.py 56->59, test_certifier_piece.py 124->128, test_served_session.py 28->29). Cure: say network certifier (or relay) in the new docstrings, help, messages and names; a site that only calls a name main defines (parse_warden, warden_outcome, the warden module) needs that name renamed at its definition first, in a change of its own. The device read is otherwise clean (relay-session keeps schedule, length, key and session on the gate side; failures visible; refusal tests complete). Re-mark after the cure and ci asks a delta read."

**What I read** [the 63 added lines that hold the name, each name looked up on `main` `8e018c9331f1`]: 61 lines use a name `main` defines (`warden_height`, `WardenParams`, `parse_warden`, `warden_command`, `warden_outcome`, `warden_windows`, `warden_minutes`, `warden_window_ms`, `WARDEN_REQUESTS`, `WARDEN_FILES`, `run_warden`, the module's `cells`, `Shim`, `audit` and `Warden.run`, the file `warden_serve.py`, the verb `warden-rehearse`, the record's value `"device": "warden"`). Two are our own prose, both "a warden rehearsal": the comment above the gate's verbs in `auditor.py` ("the relay of a warden rehearsal, kept up for clients that …") and the docstring that says "a warden rehearsal's relay alone".

**Do, before the runtime note:**
1. The two prose lines (`auditor.py` 1862 and 2688 at your head) name the kind of rehearsal that the module's docstring on `main` calls "A warden rehearsal", beside "A certifier rehearsal", which is another kind. So "the network certifier's rehearsal" would be wrong there. Look in the rename's maps on `scratch/na-rename-trial2-layoutread` (head `7b27924d7e3e`) for what that kind is called after the rename. If the maps give it a name, say that name in both lines. If they do not, leave both and count them with the names.
2. Check for any other line of our own words I missed: a message, a help string, a new name of ours that holds the older word where you could have chosen another. Mend those.
3. Rename nothing that `main` defines. The certifier's rename is coming forward to this weekend, and its maps take those sites with their definitions.
4. #9837's body gains one line: the count of sites left, by name.
5. If anything changed: push, run the changed suites once, and send me the head with the gate's last line (`python3 adapters/vllm/tools/ratchet_gate.py origin/main HEAD`, from a tree at today's `main`). If nothing changed, say so in one line.
6. The same sort on #9902 (`ed5c2df056f2`: the gate counts 74 sites), in its own commit.

**What I asked ci** (TOP-422, 10:33 PM PST): whether a reader holds on a use of a name `main` defines; my answer was no. And the earliest slot for the certifier's rename; I asked for Sun 15 Nov 6 AM PST. On a no I mark #9837 again at your new head. On a yes #9837 lands after the rename, carried across it.

**INF-808:** I tell infra tonight that #9837 does not land tonight. If #9928 is marked first it lands first, and #9837 then carries the line that fills the gate, as you and infra agreed.

The device-as-a-party read is still due Sat 14 Nov 4 PM PST.

Set no timer for this. TIMER: none.

