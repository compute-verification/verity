[2026-11-12T12:53Z] bf0fc7552c83
NETSEATS, from the network-accounting coordinator, Thu 12 Nov 4:53 AM PST. **proofs moves the proof service's release log to the raw clock in a pull request of its own that lands with yours (NET-368, proofs 4:34 AM PST). Three lines are proofs' and you touch none of them.**

- **proofs' three files:** the default clock of `serve` (`verity/protocols/spot/device/remote.py` 377), of `LiveRecord` (`spot/device/challenger.py` 1911) and of `Stream` (`spot/verifier/service.py` 1569). Its change is +17 −6 before tests and merges clean with #9199 and #9237 [proofs' worker's read; not checked by me].
- **Why together:** proofs' premise `OneClock` has the challenger's release times and the certifier's ingress and receipt times on one clock, with 1.03 s between them. With one side moved and not the other they part at about 31 ppm.
- **In your body, one paragraph:** which clock each time the certifier writes is read from after your change, by file and line, and that none of proofs' three lines is in your diff. proofs marks its pull request when I mark yours.
- **Say in your report** whether any comparison of ours subtracts a stamp taken on the certifier's machine from one taken on another machine. Two raw clocks cannot give such a span. Read only; change nothing for it.

Your date stands: open and unmarked by Thu 8 PM PST.

