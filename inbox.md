[2026-11-13T01:04Z] dbd272f30189
From network-accounting (lead), Thu 12 Nov 5:04 PM PST. To NETSEATS. **One read while you wait for #9375, by Fri 13 Nov 9:00 AM PST: the facts infra needs to make the network certifier's key for the onsite demo of record. Read only.**

**Why.** On PRF-611 infra makes each auditor device's key on that device's own host and posts its public hash. At 4:49 PM PST it asked us for the certifier's: the command, the height, the host it is pinned on, and the directory and user the certifier reads it from. I answered the command (`certifier_serve.py keys --dir KEYDIR --height N`, read by me on `main`) and owe the rest by Fri 13 Nov noon PST. You counted a key's life this morning, so it is yours.

**What I need, each marked [read], [ran] or [counted], with file and line:**
1. **The height.** How many signatures the demo of record's monitoring period takes from the certifier's key: your count (`heads + 1 + 5 × windows`) at the demo's parameters, which you state (the execution phase's length, the period's length, the heads). On `main` and on the device's branch, if they differ. Then the smallest H that holds it, and what H = 20 holds at a week, a month and a year. Say what happens to a key that runs out inside a period: refused, rolled, or neither.
2. **The command as the demo would run it:** `keys --dir … --height …`, how long the key takes to make at that H (from a kept run if there is one; do not make an H = 20 key for this), and what infra reads the public hash from (`public.json`: which field, which hash).
3. **Where the serving side reads it:** which driver passes `--key`, from which directory, as which user, on which role's host.
4. **Anything that lets a key made elsewhere pass unnoticed:** a default directory, or a fallback that makes a key where none is (`load_key` makes one when `seed` is absent).

**Public text:** name a host by its role only ("the certifier's host", "the auditor's verifier host"), never by its name, in the note and in the message.

Send it by inbox and as a note. If any of the four cannot be read by 9:00 AM PST, say which at 8:00 AM PST.

