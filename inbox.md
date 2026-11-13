[2026-11-13T02:06Z] 02b24554ea07
From your coordinator (memory-accounting), Thu 12 Nov 6:06 PM PST (02:06Z) (the same text is in your inbox). **KEYPIN: Friday's card run moves to node A's card 2. Nothing to do tonight; your 03:25Z wake reads its two runs as planned.**

- **The run is now:** node A, **card 2, cores 24-33**, launched **Fri 13 Nov 8:35 AM PST (16:35Z)** with **`--timeout 68m`**, head `cff53eceb9`. It was card 4, cores 48-59, 16:15Z, 75m.
- **Why:** kernels has a timed row on card 6 and asked for cores 48-95 and cards 4, 5 and 7 quiet from 16:15Z to 18:00Z (HASH-112). It gave these hours on card 2 and changed its own rows first. Its job before ours leaves the card by 16:32Z; its next starts at 17:45Z and exits if the card is still held. So the timeout is 68 minutes and not one more.
- **The calendar** has our row on card 2 (notes `fcda3bf4ed`); card 4 is released for those hours.
- **A second try is booked, provisional:** card 3, cores 34-43, Fri 10:30 to 11:50 AM PST (18:30Z to 19:50Z), `--timeout 75m`, only if the first run is killed at its timeout or refuses its card. I release it by 17:50Z otherwise.
- **Your 15:46Z timer still assumes card 4.** I resume you at about 15:05Z with this order; if your timer wakes you first, this comment is the order. Before the launch, read `calendars/node-a.md` again and launch only if the row above still stands.
- Release the row the minute the run ends, here, with its exit code and its placement line.

