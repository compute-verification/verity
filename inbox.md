[2026-11-11T03:09Z] ecade3de4532
TRN-97, early word (Tue 10 Nov 7:09 PM PST), as the brief asks: the calculator's count IS below core's in every row, but only by core's rounding margin. No published figure moves at its printed digits. Nothing needed from you.

Run by me (TRN-97 worker, agent bc-831fdbe0), main a930252c4993, CPU, a scratch script calling both functions; no tree file changed.
- Calculator, Detection.injected(): 3.48872407036e21 bits, the same in all six rows (it reads no fleet and no period).
- Core, consumers.infiltration_bound on one profile over the row's proof blocks, each sampled independently at the calculator's rate q = 2.0833e-15, delta = 2^-40, 262,144 bits a block: 3.48872407686e21 bits, the same in all six rows.
- Ratio calculator / core: 0.999999998137. The gap, 6.5e12 bits, is 2 x 2^-30: the margin core's worst_case adds on purpose so its float result is never under the exact bound. Without the margin the calculator is above core (it adds a floor term and uses 1/q for 1/(-ln(1-q))).
- Slowdown with each count: the world's month 1.47506793508 (calculator) and 1.47506793367 (core); the world's year 2.79716747033 and 2.79716746992; 1 in the other four rows with either. So 1.48, 2.8 and the 51% stand.
- Against core's closed form for a k-subset (optimize.density_bound, margin 2^-20): ratio 0.999998; the world's month 1.4750665.

One thing to know before you quote this. The two agree only when core's profile covers every block of the row at one delta, as the calculator and Lean's period statement count (to the first detection, detection ending the identity). Read window by window (one server, one hour: 3.3 samples expected), core's bound is every bit of the window, so the sum over a row is every bit it commits and the slowdown is 1. That reading is not the slowdown's term; the note says why.

The note with the script and its whole output follows within the hour.

