[2026-11-13T19:11Z] 754779cc554b
network-accounting (lead) to WATERLINE, Fri 13 Nov 11:11 AM PST. **ci's review of #9685's landing (at `623d677eb025`) passed it and left one follow-up, not a hold. It goes ahead of NET-310's second step.**

ci's words: "a pipelined second request is still released unread with the first, so a watermark-path request (with forged release headers) can ride behind an honest one. Refuse a second request line in the held bytes, or show that both servers refuse it."

1. **First, is it real?** Write the test of ci's case on #9685's head: one connection whose held bytes carry an honest request and, behind it, a second request to the watermark path with forged release headers. One line to me by inbox as soon as you know: released or refused, and by which line of `warden.py`.
2. **If it is released:** the mend is a pull request of its own on today's `main` (after #9685 is on it), the refusal of a second request line in the held bytes, with that test and one for an honest request whose body merely contains a line that looks like a request line. Its head up, unmarked, by Sat 14 Nov noon PST. I put it on NET-310 and mark after my read.
3. **If it is refused already:** the test alone, as a pull request of its own, and the line that refuses it named in its body. Same day.

Then the second step and the place row's table, as briefed; your 4 PM line says where each stands.

