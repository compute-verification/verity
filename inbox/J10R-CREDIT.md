
## 26 Oct 02:02Z, from the coordinator: the draw is the sampling block's (Daniel, 6:36 PM PDT, via top)
- PoUW's draw isn't special. The v5/v6 tile units go into the sampling block's stratified draw as one more subcircuit, with their ANDs in the work half. The planner sets their k under the 1% proving budget. There is no hand-set K′ = 22 at ε = 1/2.
- proofs recommends that the tiles join the integrity registration: one sampling block and one draw, so LEAN-56's single stratified bound (`JointStratified`) covers them with no new form. The tiles must be registered, circuit-checked Definitions of at most k_log 27. compute-accounting and proofs confirm on CMP-126.
- **For you:** state the strata DrawIsCoin with k_s read from that draw, not as a parameter you pick, and make it compose with LEAN-56's `JointStratified`. I'm asking lean for the interface, and I'll relay it. PoUW's number is the tile stratum's `eps_at` (ε at δ) from the same draw.
- Serving overhead (commits, per-call noise) is a separate number from the 1% proving budget. Never fold it in.
- Wording: don't say "leg" (Daniel: "it's polluting everything"). Say "protocol", or the protocol's name (PoUW, sampled proofs, ...).
