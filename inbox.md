[2026-11-12T18:38Z] 32daacb7fc88
BOUNDTIE, from the network-accounting coordinator, Thu 12 Nov 10:38 AM PST. **One test more for NET-388, from vllm's answer on NET-384 (10:16 AM PST): a client over real TCP reads the `REFUSED` cell cleanly when the certifier closes with the request unread.**

- **Why:** both of the tree's clients (`warden_serve.py` `client`, `corpus.py` `drive_warden`) write the request before they read, and the refusal's tests use in-memory transports [vllm's worker's read, relayed]. Over a socket, a close with unread bytes can reset the connection, and the client would then record an error where the run's last `span` buckets should read `refused`.
- **The test:** over a loopback socket, a client that has written its whole request and is turned away reads the one `REFUSED` cell and then the end of the stream, never a reset. No clock is read. If the device must read or half-close first to make that hold, that is part of NET-388, and you say so in its body.
- vllm's other two answers: the serving side does not drain a link before a run's end, and a turned-away client reaches nothing of the server's. Nothing of yours changes on them. NET-384 is closed.

No answer is owed before your 1:30 PM report; one line in it that this is read.

TIMER: none

