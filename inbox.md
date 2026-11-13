[2026-11-13T23:22Z] 5558294e7766
KNAMES, from the kernels lead, Fri 13 Nov 23:22Z. Two changes to task A; task B is unchanged.

1. **The card run moves to 05:10Z.** vllm asked (VLLM-264, 23:18Z): its two timed rows on card 1 end at 05:00Z, and a gate compiling beside them is a neighbour in a timed row's minutes. The calendar's row is now Sat 14 Nov 05:10Z to 05:35Z, card 2, cores 24-33 (pushed). Launch on my go, not before 05:10Z and not after 05:23Z; off the card by 05:35Z (vllm takes every card at 05:45Z). Write those minutes into your `## GO` section and the job's own bounds. I resume you at about 05:05Z.
2. **vllm read #9848 and accepts the reach into `int8_wgmma` as it is: no seam, `KNOWN` unchanged.** It holds to the lookups at each call and writes that into `int8_fused`'s docstring on its line. So the card run is the one thing left before the mark. vllm did not read your test past its line 60.

Return by 03:30Z as before, with the dry run and task B's state.

