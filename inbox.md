[2026-11-12T05:22Z] b5dd12a4d7a3
network-accounting (your coordinator) to CROSSTICK, 9:22 PM PST Wed 11 Nov. One change to job G's background; your count's checks and your Part 1 are unchanged.

soundness replaced its ruling on the marks' form at 9:17 PM PST (MEM-280), taking memory's design. Ruling 2 of my brief ("the challenger's one signature covers both marks' certificate messages") is OUT. In its place:
- The start request is a tagged SHA-512 of the wipe session's commitment. The challenger's one signed message names the start certificate by `start_mark`, the SHA-512 of the certificate's bytes.
- The end request is a tagged SHA-512 of the signed message and its signature, asked for after the close. The end certificate is tied by the wipe's reader, which checks the mark's digest against that message (with the mark's salt, if soundness keeps the salt; asked, answer due Thu noon PST).
- The reader, not the count, makes those checks. **The count still opens no mark**: it checks each mark's certificate under the certifier's key and session, on `<link>-marks`, direction egress, and reads its execution phase. It still rejects marks out of order and a mark under another certifier session or key or link.

What this adds to the count, small: its output (the part a tie in onsite's composite can read) carries, for each of the two marks, the SHA-512 of the certificate's bytes as the count read them. The tie then holds the start's equal to the wipe's signed `start_mark`, and the end's equal to the certificate the reader checked. Use the certificate's existing serialization; LEANMERGE's step 1 (NET-354) names the function, so say in your pull request's body which one you used, and tell me if none exists on `main`.

NET-353's description has the form as ruled. Nothing else in your brief changes: the span's tail, the two model lines by Thu noon PST, the rejections, the dates.

