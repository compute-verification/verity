[2026-11-13T09:14Z] 7cc6944022ed
Your 08:57Z entry is read. The "For TOP-415" paragraph goes on the issue as written, with the table.

Next, one piece of reading and one small measurement. No change to the checked function, no pull request.

**Can the compiled check read one member's circuit at a time?** Today it reads every sampled member's circuit in one process (0.5 to 2.1 TB on Thursday's window, by your figures). One member at a time would make the peak the largest member's, about 32 GB of text, so about 125 GB at 3.9 bytes a byte.
1. [read] Where the circuits are read and held: `refusalMain` and the accepting main of `sampled-proofs-one-stage-zk`, and the function they call. Is every blob in memory at once because of the main's IO (it reads all files, then calls the check), or because the checked function takes all of them as one argument (a list the proofs speak of)?
2. [read] What a per-member loop would have to keep between members (digests, the draw, the verdicts) and which statements about the check would be restated if the function took a stream. MEMBERS's switch and WINDOWAT's proofs (`cursor/proofs-windowat-5e4b`) are the ones that read it; name the definitions, do not edit them.
3. [run, optional, under 30 minutes on a node] Where the 3.6 to 3.9 bytes go: the JSON text held beside the parsed circuit, or the parsed circuit alone. One number at one size is enough.
4. One paragraph for TOP-415 and for MEMBERS: the smallest change that makes the per-member check's memory the largest member's and not the sum, who would write it, and what it moves in the lock if anything.

Entry by 12:00Z. `--on auto`; cpu-13's `proofs` pool starts nothing after 10:30Z and is closed 11:30Z to 14:30Z.

About the key: your finding (the gate's key is W = 1, H = 20; a local challenger's key must be made at those parameters for the compiled check to pass it) is reported as a finding only. I order no change to serving's harness or to any key: RULE 11 stands.

