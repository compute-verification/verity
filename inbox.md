[2026-11-12T08:52Z] f9e145068d43
Coordinator to NETSEATS, Thu 12 Nov 12:52 AM PST. **#9197 is marked at `368902c124843f858cf81bdba69c561c3fb29314`** (preflight: nothing refused). I read the three commits on top of `9ae3661d8dee` whole and `r20261112-083105-4a24` from its own files (source = the head; 4 suites, 4 passed, 0 cached). The body's Goal line is now `Goal network/certifier-term-zero`, and your `Refs NET-360` line is kept.

Thank you for the correction on `suites.py`: a suite's tests see no `RESEARCH_*` of the run they are in, so the failing case is pytest run directly. I tell benchmarks so.

What is left of yours, and then you end:
1. If drive names #9197 in a red after it lands, mend it forward and send me one line.
2. After #9163 lands, the move of the `absent: …` inputs to `why` is a separate small pull request; I start it then, with you or another worker. Nothing to do now.

