[2026-11-13T12:34Z] 79b87bf66598
Network ruled one more thing on NET-403 at 12:17Z, and it touches your read of our two commands. Nothing else in your order changes.

- The network certifier's offline check, which writes `out/registrations.jsonl`, is handed the SHA-512 of the network certifier's public key and the session by the auditor. A deployment record that names another key or session is a problem by name, and then no file is written.
- The file's FIRST LINE names that hash and that session. Each line after it is a registration, byte for byte the publisher's.
- A check that finds any problem writes no file at all.
- For our two commands, in network's words: they take a head from a registration's line as before and pass over the first line.
- The draft is `cursor/na-registrations-writer-b4e6` at `0ffd4a4716d1`.

Add three short answers to your 14:30Z entry:
1. What passing over the first line costs in our two commands, in lines and tests.
2. Whether our commands should refuse a file that has no such first line, and how they would tell: a file in the older shape would otherwise lose its first registration silently.
3. If our writer writes `out/serve-registrations.jsonl`, whether that file should carry the same first line.

