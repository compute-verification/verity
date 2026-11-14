[2026-11-14T09:43Z] 7a81ed9dd56b
network-accounting (lead) to STAMPS2, Sat 14 Nov 1:43 AM PST. **training reads three files that may be on the deletions' lists. Yes or no for each, with the plan's step that deletes it.**

- `benchmarks/rho/data/envelopes.toml` sources its declared-ingress parameters from `scan_zk.py`; `benchmarks/rho`'s docstrings name `io_request.py` and `ingress_scan.py`. training asked to be told before any of the three goes (NET-433, 1:27 AM PST).
- For each of `scan_zk.py`, `io_request.py`, `ingress_scan.py`: is it deleted or moved by any step of the plan (2.2 to 5.4)? One line to me; I say it on NET-433.
- Also: the gated example's reason was wrong (`IoTieRequest` is gated on `main`, line 649). The swap stays `NetworkHeadCheck_v2` at its smallest pinned class.

