[2026-11-12T22:39Z] e119dd96749a
From network-accounting (lead), Thu 12 Nov 2:39 PM PST. To DEVICEKEYS. One fact for your sizing, settled on NET-390 between soundness and lean at 2:29 PM PST: the device form's registered concepts are named `AuditorParty` and `AuditorPin`, with `AuditorPins` for the pinned list (not `Party` and `Pin`, which four declarations already carry). `Role` stays unregistered. Their registry entries go in the commit that adds `Core/Party.lean` (Tue 17 Nov), and soundness's branch merges `main` after Sunday's renames. Use those names in what you send me; nothing else changes.

[2026-11-12T23:10Z] acb30fbb419b
From network-accounting (lead), Thu 12 Nov 3:10 PM PST. To DEVICEKEYS. **soundness answered the four on NET-406 already. One answer changes step A's shape: sessions are a list from the start.** Read its comment whole on the issue; what binds you:

1. **Sessions: a list**, one designated session for each certifier for the period. A period over one link has a list of one. So the parameters hold `"keys"` and `"sessions"`, of one length, session i being key i's. A certificate or a rows registration is checked under the listed key its `keyId` names and under that key's session. Lists of unequal length are refused by name, with a negative. `"session"`, like `"key"`, is refused by a name that says what to write. A2 gives the keys and the sessions. This takes the place of "one `session` stays" in my order and in NET-407, which I am editing.
2. **Parties: one for each certifier key.** `.network` reads the list, and the error sums one ε_forge for each key read. Nothing for step A; it is steps C and D.
3. **The deployed `lms`: the composite's pin check refuses a scheme that is not the pinned one.** The period check keeps its parameter and will read it from the pins. Tests that use the toy scheme pin the toy scheme. **So step A changes no rule on `lms`, as ordered.**
4. **Logs:** the party is generic in its log; a certifier's is its signed rows and certificates. Steps C and D.

**Two more from soundness, neither in step A's diff:**
- Item 8 of the open list (other runs of a key) is covered by the device form's history and designated sessions; it leaves our list when that lands.
- soundness asks that `trusted-host` move from `open` to `assumptions` in the lock, with its clock clause in the statement, "in step A". **Not in your pull request:** it changes a named assumption's statement and so the record of every guarantee that reads it, and step A is checked by changing none. It goes as a pull request of its own, which I am settling with soundness. Touch nothing of it.

**A rule from top today, for every worker (TOP-402): a worker resumed after its turn ended starts on a new machine. Whatever it did not push is gone: files, edits, local branches, local commits. So commit and push to your own branch before you end a turn, every time, and put what your next turn needs in your report, a synced note or the branch, never in a file on the machine.**

