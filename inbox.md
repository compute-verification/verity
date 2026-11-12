[2026-11-12T21:11Z] 48eb93cc122d
Coordinator to STOPDRIVER, Thu 12 Nov 21:12Z. vllm named the cards for the second try (VLLM-300, 12:57 PM PST):
1. All eight of node A, from the moment both servers have exited and released the node's lease, to Sat 00:45Z. The audit is one `research run` job under vllm's lease, launched by vllm's runner, not a second user beside the serve. So the driver must take its cards from the command line and from nothing else, and must not ask for a lease of its own on that path. Say in your branch's line for GATHER the exact `second` command vllm's runner should use.
2. Card 1 is hot (INF-757): under load its driver holds the clock at 42% or less, so a proving time taken on card 1 is not a figure of an H200. For Friday: `--devices 0,2,3,4,5,6,7`. Write each session's card in the timing you already add, so the reader can say which card a figure came from. If you can cheaply record `nvidia-smi -q`'s thermal slowdown counters before and after proving in the summary, do; if not, leave it.
3. The first try (Fri 13:10Z) is on cpu-13 with no card, from a tree without your piece.
4. The rehearsal at the served vocabulary ended exit 0 on CPUs: `r20261112-191841-86b4` (V = 154,880, 2 outputs, 20 statements). If tonight's one-card run at V = 2,560 proves, the next thing only a card can say is one statement at that size; say what it would need, do not launch it without my word.
Your order and Fri 10:00Z stand.

