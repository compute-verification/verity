[2026-11-12T12:26Z] e0382e7f695a
CERTNEG, from the network-accounting coordinator, Thu 12 Nov 4:26 AM PST. vllm answered on NET-340 (4:14 AM PST). Three answers change nothing; one needs a small build for tonight. It does not move your baseline launch.

**Unchanged:** 35 connections open is fine. The order stays the code's. 1.67 requests a second stands.

**The one that matters: vllm's gate does not read a watermark line with lost releases, and that line must not reach its server.**
- A line with a third number ends the watermark with an error, its POST is answered 400, and the gate opens for the rest of the server's life. Every try after it is served outside the order [vllm's worker's probe and read; not run by vllm's lead or by me].
- The device sends `release.watermark_chunk(t, n, lost)` [read by me on `main`: `verity/protocols/certifier/verifier/release.py` 119-125, `warden.py` 123-132], and `warden_serve.py` 804 keeps the lost releases in the run's log.
- vllm asks: the relay drops the lost names or ends the watermark there. If that cannot change before the run, the job stops at the first such line and says so on NET-340, and no try after it counts. I answer vllm by Thu noon PST.

**What I want, by Thu 12 Nov 9:30 AM PST, by inbox:**
1. **Read:** in the six tries as commanded, what makes a release lost? Only a server connection that closes or fails before the request is written whole, or can one of our own deviations cause it? Files and lines.
2. **Build, for tonight, benchmarks only:** after each try the job reads the device's own log, and if it names a lost release the job stops there, exits nonzero and prints which try. No later try runs. With a test. Its own pull request from `main`, open and unmarked, suites by `research run` on its head, launched once. Touch nothing under `verity/`. If the job cannot read that log between tries without a change under `verity/`, stop and tell me.
3. **Size, do not build:** the device's change that keeps the lost names from the server. Two forms: the line is `t n` only, or the watermark ends at the first lost release. For each: lines, files, tests, what vllm's gate then does with the lost release [`serve_admit.py` holds a missing release and then skips it, by its docstring; check], and whether anything of ours reads the lost names from the line (`read_watermark`).
4. Which form you would take, in one line.

**Rules as before:** each run launched once; never merge or mark; machine names and addresses masked in everything printed; nothing in the tree names a date, an id, an issue, a pull request, a lane or a worker, or holds a measured result.

