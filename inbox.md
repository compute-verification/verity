[2026-11-14T07:02Z] 93049a6b2f29
Coordinator, 11:04 PM PST: kernels found that delta = 3 makes the bits depend on the order of the multiplies in stats_s5; 2 and 4 do not. So: report results for all of delta in {1,2,3,4} as planned, but prefer a power of two in your recommendation (2 or 4) unless only 3 passes; if 3 is the only one, say so plainly.

