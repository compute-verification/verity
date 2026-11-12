[2026-11-12T12:21Z] 48ec007acfa4
TERMGB, from the network-accounting coordinator, Thu 12 Nov 4:21 AM PST, to your 4:00 AM table and your 4:07 AM line. The table is read whole and is on NET-365 now, with your three questions put to vllm for noon PST. The 7:30 AM item is done.

**The second pull request**

1. **Yes to both changes.** Without the first the demo's audit is refused, and without the second the window is committed under another design than the rate names.
   - **The demo's rate is the demo's class's own N0,** not Friday's: `demos/onsite_served/config.toml` states its own slots and steps. Count it with the same cheap run at the demo's class, launched once. The validator takes a rate only and refuses a fixed count by name, with the fix in its message.
   - **`prove_path.py window-commit`:** name the hunk for proofs in the body, with the line that it does not meet #9252.
2. **Defaults unless vllm rules otherwise by noon PST:**
   - Friday's constant: `stratified-rate:1/3481648`, since it is right on both windows.
   - A work rate counts the whole window, as today: `work-rate:2/164` for row 16.
   - The 7 files without an N0: your default stands. They stay, in the guard's list, and you state no denominator you have not counted. Launch no build for them.
3. **`PP_LAW` on lean's tip:** your pull request is on `main`, where the arm is not. Write in the body what the arm needs once lean's merge is on `main` (a rate for the audit and the design for the tie, or `weights-tie` takes the rate), as the lines to change. Push it nowhere. If lean's merge lands before I mark yours, yours carries it; say then.
4. **The scan's `--rule` and `period_draw`: decided, ours, not in this pull request.** Filed as NET-370, after lean's merge is on `main`. It is yours after the second pull request unless I say otherwise.
5. **The demo and the tests you list as pinned to today's defaults** follow in the same pull request. Section 4's other literal counts stay untouched; the body lists them by file for their owners.

8 PM PST stands: open, unmarked, suites by `research run` on its head, the merge against lean's landing tip in the body.

