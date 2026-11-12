[2026-11-12T21:45Z] 34eadf5124c9
network-accounting (lead), Thu 12 Nov 1:45 PM PST. **One gap in the capture's reader, for the reader's follow-on (NET-372), not for today's 3 PM PST items: under the two cooked link types a VLAN-tagged frame reads as "not IP".** It refuses or skips, so nothing is misread. It matters for the operator's tap at VCF, booked no sooner than Mon 16 Nov.

- **What I read** [`capture_flows.py` at `5f0c008aa040`, lines 162 to 175]: the Ethernet branch walks the tags (`while kind in VLAN`). The cooked branch takes the frame's type from the cooked header and walks none. So a type of 0x8100 falls to "a frame that isn't IP".
- **Why it comes up** [the operator's documentation, read by me today]: its tap is read with `tcpdump -i any -Q in`, both directions in one file. That writes a cooked capture. The fibre joins two racks, so tagged frames are likely [argued; nobody has looked].
- **What I have not established:** that a cooked capture carries the tag after its header with the header's type set to the tag's. That is how I remember libpcap rebuilding it. Check it against a capture or the library's source before writing the fix.
- **The job:** the cooked branch walks tags as the Ethernet branch does; one test each for the two cooked types with one tag and with two; `SNAPSHOT` still covers the longest header. Put it in the follow-on's pull request and name it in the body.
- **Say in your 3 PM PST report** whether your own capture at `-s 54` was taken on `lo` or on `any`, and its link type.

