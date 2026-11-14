[2026-11-14T09:17Z] c3c689bca956
PREMISES, Sat 14 Nov 1:17 AM PST. **(b) needs two numbers the calculator lacks: a window's length and a window's sampled-proofs knowledge error. Which windows the sum runs over moves the world's yearly delta from 1.6e-8 to 0.75, so I ask before choosing; I build (a) and the citation line meanwhile.**

- READ: no window length exists in benchmarks/rho's data, verity/protocol.py or benchmarks/nci/tiers.py. lag_s (1 h, assumed) is "a window plus the verdict delay". No value exists for the statement's ks_t = eps_ks + delta_link either ("has no value yet").
- My recommendation:
  - regime.<remote regimes>.window_s = 3600 s, status assumed (the hour lag_s assumes).
  - .window_ks = 2 WINDOW_DELTA = 2^-39, status assumed (eps_ks and delta_link each at delta_s, as window_delta_crypto assumes delta_bind + delta_ks).
  - T = period / window_s: a window is a round of the developer's coins (the statement runs each window by one identity, and J = 1), not one per server.
  - Detection.delta's crypto becomes T window_ks, in every regime with detection (sampled or not: the statement's sum stands at q = 1 too).
- The sum at those values (INFERRED arithmetic, RAN in python):

| period | windows of the developer's coins | per server, one datacenter | per server, the world |
|---|---|---|---|
| week | 3.1e-10 | 3.8e-6 | 0.014 |
| month | 1.3e-9 | 1.6e-5 | 0.061 |
| year | 1.6e-8 | 2.0e-4 | 0.75 |

  Today's delta beside it: PoUW's p/eps' is 9.1e-9 (coin) plus 1.8e-8 (crypto); p_evade is 9.1e-13.
- Printed figures that move: none of periods.py's (it prints no delta yet; the line I add will). In run.py's rho.md, the delta (crypto) of every scenario with detection moves; I list them before any push.
- Unanswered in 30 minutes, I build the recommendation and send you the list of moved figures before pushing.

