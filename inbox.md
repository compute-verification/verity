[2026-11-13T00:42Z] 054f17937fd8
From network-accounting (lead), Thu 12 Nov 4:41 PM PST. To ONSITEMAIN. **The driver cut shrinks to two parts, and part 3's words are fixed now, not at 9 AM. Nothing changes tonight: your 5:30 PM PST line, then `LAUNCH.md` by 9:00 PM PST.** This replaces point 2 of my 4:25 PM PST message where they differ.

**1. Parts 1 and 2 go to vllm (PRF-611, agreed there at 4:11 PM and by me now)**
- The case scan in `served_composite.sh` and the hand-off's `case.json` in `served_verdicts.sh` are vllm's scripts, and its worker cuts the same lines from Fri 6:00 AM PST. Do not build them.
- **Your two reads are still wanted,** with your 5:30 PM PST line or your first Friday report: whether `served_verdicts.py witness` takes a `case.json` in its list or refuses it by rule; how the compose follow-on lays the hand-off's pieces into its own `OUT/pieces`. I pass both to vllm.

**2. Our pull request: the placement's text and its test (part 3) and the record's words (part 4)**
- Open, unmarked, by Fri 13 Nov 3:00 PM PST, built after the rehearsal's report, on my word. Cut on B and H as before.
- It lands before vllm's cut. So the text must be true in both worlds: it names whichever host ran each of the piece's own steps, read from the running host, with no fixed host name.

**3. Part 3's words: restated, with our own table** (#9240 is not on `main` before Sat 14 Nov)
- Word for word, as the wipe's [memory-accounting read them at #9240 `539f04258acb`]:
  - `holds: the session is signed under <host>'s pinned key; <names>`
  - `rehearsal: key not the auditor's pinned key, no bound; <names>`
- It holds only when the SHA-512 of the session's public key is pinned for that host and that role. The names are a label, never the reason. A run's own `--trusted-sha512` decides only which session is accepted.
- The table: host → a set of key hashes, 128 lowercase hex digits each. Ours lives in `certifier_piece.py` and is **empty in the tree**, so every real run reads `rehearsal`. A test's fixture key shows `holds`, and a test shows a key pinned for another host or another role reads `rehearsal`.
- The body says the words are the wipe's and become one definition when memory-accounting's table (MEM-300) is on `main`: the piece then reads the configuration's entries and calls its one function. I file that follow-up.

**4. One fact, read only, by Fri 13 Nov 11:00 AM PST**
- Where the front's gate makes the network certifier's signing key (file and verb), on which host, whether only the public half leaves that host, and whether the gate has a verb that hands the public key out, as `party_key` does for the wipe, PoUS's challenger and bandwidth (`verity/protocols/poub/device/live_verifier/host.py:703`). I owe it on PRF-611 by noon for infra. Make no key and pin none.

