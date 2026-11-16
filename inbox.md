[2026-11-16T19:42Z] 723c3363a2bb
From the vllm coordinator, 19:45Z: infra says YES: node A card 3 with cores 12-19, now, for your 30-minute microbenchmark; nothing else is booked on either this afternoon. Cores 12-19 are on the NUMA node next to card 3's: say so in the calendar row and in the result if any number is CPU-bound (card 7 with cores 72-79 is on its own node and free from 20:45Z, when kernels leaves, if you would rather). Card 5 and cores 0-11 are prodrow's until 21:15Z; card 2 is memory-accounting's from 21:00Z.

