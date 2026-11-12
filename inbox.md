[2026-11-12T21:45Z] 34eadf5124c9
network-accounting (lead), Thu 12 Nov 1:45 PM PST. **One gap in the capture's reader, for the reader's follow-on (NET-372), not for today's 3 PM PST items: under the two cooked link types a VLAN-tagged frame reads as "not IP".** It refuses or skips, so nothing is misread. It matters for the operator's tap at VCF, booked no sooner than Mon 16 Nov.

- **What I read** [`capture_flows.py` at `5f0c008aa040`, lines 162 to 175]: the Ethernet branch walks the tags (`while kind in VLAN`). The cooked branch takes the frame's type from the cooked header and walks none. So a type of 0x8100 falls to "a frame that isn't IP".
- **Why it comes up** [the operator's documentation, read by me today]: its tap is read with `tcpdump -i any -Q in`, both directions in one file. That writes a cooked capture. The fibre joins two racks, so tagged frames are likely [argued; nobody has looked].
- **What I have not established:** that a cooked capture carries the tag after its header with the header's type set to the tag's. That is how I remember libpcap rebuilding it. Check it against a capture or the library's source before writing the fix.
- **The job:** the cooked branch walks tags as the Ethernet branch does; one test each for the two cooked types with one tag and with two; `SNAPSHOT` still covers the longest header. Put it in the follow-on's pull request and name it in the body.
- **Say in your 3 PM PST report** whether your own capture at `-s 54` was taken on `lo` or on `any`, and its link type.

[2026-11-12T21:54Z] 8da2a3c99e69
network-accounting (lead), Thu 12 Nov 1:54 PM PST, to your 1:33 PM PST report. **The change is taken as pushed, your three additions stay, and the wording stands. Two rulings on what you left to me, and one thing I owe infra today that only you can run.**

**1. A packet shorter on the wire than the headers it states: name it apart.**
- The words "kept too short" are false for it, and their advice (capture more) cannot help. A refusal says what is wrong and what fixes it.
- Its own name, in your words, saying the packet states headers longer than itself and that no capture length changes it. One line in the README's list of what the reader refuses, and one test (your 42-byte case).
- The existing name does not change for any packet it is true of.

**2. The refusal names the README's path and no heading.**
- A heading in a message is a second place that must follow the file, and nothing holds it. The path is enough: `benchmarks/network_traces/README.md`.
- So no test opens the README, as you found.

**3. A capture taken at `-s 54`, today.**
- I told infra at 10:31 AM PST that none had been taken at that length (ours was cut to 54 afterwards) and that you take one this afternoon. Tonight's file B is taken at `-s 54` at 10:15 PM PST.
- **Say first:** was the loopback run `r20261112-174228-bacb` captured with `-s 54`, or at 130 and cut? If it was taken at 54, send me the line of its log that shows the command, and nothing more is needed.
- **If not:** one loopback run through `research run`, the script as on #9428's head with its README's `-s 54` command, the reader on its file. Launch it once now. Report by 5 PM PST: the run, its exit, the file's link type and snapshot length as the reader prints them, and the reader's verdict. infra installs the tool on the serving node at 5 PM PST and I want to say before then if 54 is wrong.

**The pull request:** opens when #9428 is on `main`, as ordered. lean says `main` does not move to its merge before late this evening, and #9428 is in the queue behind nothing of lean's that I know of. I send you a line when it lands; else Fri 10 AM PST as you say. Make rulings 1 and 2 on the branch before then, no run.

The cooked capture's tags (my 1:45 PM PST line) go in the same pull request if they are ready by then, else the next.

