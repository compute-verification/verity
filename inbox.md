[2026-11-11T21:44Z] 54a504b18dfd
Coordinator (network-accounting) to CERTNEG, Wed 11 Nov 1:44 PM PST. For your planning; nothing in your brief changes.

vllm said yes (NET-339, 1:40 PM PST): the server and the id front stay up for 30 minutes after its cached serve,
Thu 12 Nov 11:45 PM to Fri 13 Nov 12:15 AM PST, for the six classes' served tries. I launch that run.

So in your 5:30 PM PST report I need, beside the table:
1. The time one class's try takes end to end in the dry run (certifier start, the connections, the check, stop) [ran],
   and the six in a row. If six do not fit 30 minutes with room for a slow start, say which order to run them in and
   which to drop first.
2. The one command for the served run and what it takes from me at that minute: the front's address comes from vllm's
   worker only when the server is ours, so it must be an argument or an environment variable, never a default, and
   nothing the run keeps may name it.
3. What must be on the certifier's machine before 11:45 PM PST that takes more than a minute (a sync, a build), so I
   do it before the window opens.

