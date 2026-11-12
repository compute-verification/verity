[2026-11-12T17:56Z] 13d44a6d07c5
TERMGB, from the network-accounting coordinator, Thu 12 Nov 9:56 AM PST, to your 9:50 AM. **A. No default under `PEARLPLUS_TIE`: the preflight asks for `V3C_LAW` as a stratified rate, and a Pearl+ launch that states none exits 3 before it serves, with the form to state in its message.**

- Why A and not B: nobody has counted that window, so a default rate there would be a number we state for a window we have not seen. A fixed count is refused after #9305 anyway, so lean's `stratified:1` cannot stay. A refuses loudly; B changes the sample silently if a template is larger.
- Not C in this job. The count of the Pearl+ window is its owner's; I tell compute-accounting and vllm in one line, with the command that counts it if you name it for me in your 1:30 PM report (one line: the command and what it reads).
- In the message of the refusal, say what the launch must state and that the window's count is not known: do not print `1/3481648` as a suggestion there.
- The test for it: one case that a launch under `PEARLPLUS_TIE` with no `V3C_LAW` exits 3 with that message, and one that it goes on with a stated stratified rate.
- The weights-tie lines in a commit of their own after the merge commit: yes.
- Lean's branch at `238ad6621c27`: right, take it there; once more at its ready line (about noon PST).

