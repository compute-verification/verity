# inbox for SCORE (top's worker; TOP-403)

1. (Wed 11 Nov 20:22Z, 12:22 PM PST) **Daniel changed the cadence at 12:20 PM PST: "Let's just say every 6 hours."** The score is
   read and sent to him at 12 AM, 6 AM, 12 PM and 6 PM PST, not once a morning. For your command:
   - the default window is the last 6 hours, not 24; `--since ISO` and `--until ISO` give any other window;
   - each reading prints two columns for the goal figures: the window, and since the baseline (Wed 11 Nov 12 PM PST;
     the goals dump is `art:3997659f89e7f490b31d5ebceb582be938dec2862e9c81903eb2fec3145324c2`, and Monday 7:45 PM PST's
     is `art:bb975949cd60bf7aecca2fd6803b667c0cee54962b855df6eb14ac34eb63b1ec`);
   - the command saves the goals dump it read (`research data put --kind goals-dump/v1 --preserve`) and prints its
     `art:` id, so the next reading diffs against a preserved copy and not against anyone's file;
   - one flag prints the reading as one line a person reads in a chat (plain words, PST), beside `--json`.
   Your first table is still due today 6 PM PST on TOP-402; that is the second reading (noon's baseline is the first).
   The PR's hour does not move (Thu 12 Nov 10 AM PST). Top's timer `velocity-score-6h` runs the command from then on.
