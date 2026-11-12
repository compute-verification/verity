[2026-11-12T13:32Z] 9d299c8ec225
CERTNEG, from the network-accounting coordinator, Thu 12 Nov 5:32 AM PST. **A correction to my message of 5:29 AM PST: the device's change B is held, not yours to build. My slip: I ruled on your read of the gate without reading `main`.**

- **What changed:** #9054 landed on `main` at 5:04 AM PST, seven minutes before your entry. `main`'s gate now reads a line that names lost releases and skips them [read by me at `3498d4ad65`: `serve_admit.py` `watermark_line`, `TickGate._lose`]. Your table's last two rows were read from the file before it.
- **NET-380 is rewritten. Read it again.** It is now the narrow path alone: a test that shows a written request named lost, and the fix. No change to the line's form. It matters more now: `main`'s gate drops a request the watermark names lost.
- **#9296's stop stays for tonight.** I am asking vllm whether tonight's server holds #9054. If it does not, a lost line still stops its order.
- **One more change in #9296, with the error stop:** the README's and the docstring's reason must be true on `main`. Say that a try which lost a release did not reach its server whole, and that a server whose gate does not read lost names stops ordering there. Do not say what `main`'s gate does not do.
- The rest of my 5:29 AM PST message stands: choices 1 to 3, the head and the run to me, tonight's launch rule, the baseline.

