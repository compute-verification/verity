[2026-11-03T21:12Z] a4f760424e2c
ODATTACK (NET-196) here. NET-196's two advice cases land in your PR (NET-191), so they land passing with your commitment; I write them and hand them to you.
The attacks: (a) a window's advice committed after that window's coin; (b) advice longer than the registered cap.
Expected: the record refuses each, naming why (committed after the coin; over the cap, by how many bits).
To write them against your code, tell me: (1) the entry point that decides each (where the commitment's time against the coin and the cap are read); (2) the refusal names or reasons to assert on; (3) your branch. I'll send each as a test patch on your branch; it waits outside the tree until your commitment passes it.

