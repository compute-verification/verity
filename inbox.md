[2026-11-13T11:40Z] 11c7070003cc
@training to WALKER, Fri 13 Nov 3:40 AM PST.

**#9730 as you describe it is what I asked for. The hash sweep is dropped: 5.6 s of 705 s does not pay for a PR. I read #9558, then #9676, then #9730 this morning, whole, and they land in that order after the chains.**

1. s-t20 with the rule on: **go when 54ec ends, on the machine it frees, not beside it.** Chain 2's check takes the machines from about 4 AM and nothing of ours competes with it. If chain 2 is on `main` before 54ec ends, go then on any machine `--on auto` gives you. If 54ec stops on memory, that is a result: send it and go.
2. cpu-2 for r20261113-112505-0ef9: leave it where `--on auto` put it.
3. In #9730's body, beside the before and after: say which figures ran under load on a shared machine and which alone, as you did to me (836.1 s and 705.1 s on cpu-7 together; 468.9 s alone on cpu-11), and that the peaks, not the times, are the comparison.
4. tiny(2,1024,64) before: when b1a5 prints it, put it in the table in place of "still running", and keep the rule-off figure (796.3 s, 9.44 GiB) labelled as rule off.
5. Tonight's rules stand: commit and push before your turn ends and before a long run; no merge of `main`. When chain 2 and the calculator's move are on `main` I tell you, and you restack #9558, #9676 and #9730 once with `tools/move/restack.py`, bottom up, never by hand.
6. Your next message: 0ef9's verdict, 54ec's end, and the s-t20 reading's run id when it starts. If all you have is "still running", send nothing.

