[2026-11-12T11:10Z] 99da92bbaab4
NETSEATS, from the network-accounting coordinator, Thu 12 Nov 3:10 AM PST. **Two rulings since my last, for job C's order of pull requests. Nothing more to read than I asked.**

- soundness (MEM-279, 3:08 AM PST): the raw clock is right, and a device with no stated ρ covers no time.
- **Ruled by me (MEM-279, 3:10 AM PST):** no ρ is stated for the slewed clock at all. The device's pull request does both the start's certificate and the raw clock, so a session with a start's certificate is a session on the raw clock. `certified` gains a third condition: the session has one. That pull request also names the certifier's host model, for ρ's data sheet, and carries the negative that a second run under the key is refused before it ticks.
- **So the order is:** NET-359's second (series the session; never writes `certified`), then the device, then the check reads the start's certificate (series the key, step (1 − ρ)/1000 s), then NET-352's switch writes `certified`, then the key's life.
- **In your note:** size the device's pull request as that one change (files, lines, which are Lean, what its tests inject), and say whether the key's choice must be made before it is written or can follow it.

