[2026-11-12T10:15Z] cdeee51fb472
ci 2:1x AM PST, for your circuit-catalog:records ask (infra CI-485 2:11 AM PST): on cpu-15 the oldest catalog task had run 36 min and held 78 GB at 2:06 AM PST (a cold machine), against 8 min and 22-25 GB in its four warm passes on cpu-13. So 'about 40 GB' may be low cold: read that run's peak (infra's executor records) before fixing the figure, and say in the PR how the ask covers a cold machine. infra deletes its g2order.py pass-over rule for the three lock tasks once your fix is deployed: tell me when it lands.

