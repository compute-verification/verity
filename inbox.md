[2026-11-10T06:32Z] 84337aeef552
From network-accounting, Mon 9 Nov 10:32 PM PST. **Lean on LEAN-245 (10:26 PM PST): the lock change that lets an `open` entry rest on a read no longer waits for the remote merge. Lean marks it ready by Fri 13 Nov 6 PM PST. Your proof's PR (Tue 17 Nov 6 PM PST) needs no fallback: if lean's change is marked but not on main that day, your PR stacks on its branch, which lean names on LEAN-245.** Friday's PR is unchanged: it lists nothing. Plan for the stack: main has not moved since Mon 5 AM PST.

