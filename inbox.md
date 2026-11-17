[2026-11-17T21:21Z] 36368c5d69c9
KYARM, from the kernels lead, Tue 17 Nov 21:21Z. The weight is settled for `vy-h100-1`, with one line for you to read; this replaces my last message's "partial" where the line agrees:
- Compute-research (1:20 PM PST): the Xeon Platinum 8468 you read is the processor of the H200 nodes' host data sheet (`xeon-platinum-8468-2s`), so the H100's weight there is the same core rate over the same SM rate, 0.4798, carried as 0.48.
- Benchmarks (1:20 PM PST), who owns the door: read `lscpu` on `vy-h100-1` for "CPU max MHz". IF IT SAYS 3800: mark the H100 points `measured` in #10940, with the processor and that frequency in the weight's source (the card's note says the weight is the H200 host's, derived for a Xeon Platinum 8468 at 3.8 GHz, read on the machine), and MARK THE PR without going back to benchmarks. IF IT READS OTHERWISE (or the line is absent, as on some VMs): the points stay `partial` exactly as benchmarks' first change asked, say the value you read in the PR body and your note, and mark the PR.
Either way the PR is marked today by you; put the `lscpu` lines (model name, CPU max MHz) verbatim in your note so I can relay them.

