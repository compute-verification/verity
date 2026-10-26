
## 26 Oct 02:02Z, from the coordinator: the draw is the sampling block's (Daniel, 6:36 PM PDT, via top)
- PoUW's draw isn't special. The v5/v6 tile units go into the sampling block's stratified draw as one more subcircuit, with their ANDs in the work half. The planner sets their k under the 1% proving budget. There is no hand-set K′ = 22 at ε = 1/2.
- proofs recommends that the tiles join the integrity registration: one sampling block and one draw, so LEAN-56's single stratified bound (`JointStratified`) covers them with no new form. The tiles must be registered, circuit-checked Definitions of at most k_log 27. compute-accounting and proofs confirm on CMP-126.
- **For you:** state the strata DrawIsCoin with k_s read from that draw, not as a parameter you pick, and make it compose with LEAN-56's `JointStratified`. I'm asking lean for the interface, and I'll relay it. PoUW's number is the tile stratum's `eps_at` (ε at δ) from the same draw.
- Serving overhead (commits, per-call noise) is a separate number from the 1% proving budget. Never fold it in.
- Wording: don't say "leg" (Daniel: "it's polluting everything"). Say "protocol", or the protocol's name (PoUW, sampled proofs, ...).

## 26 Oct 02:12Z, from the coordinator: lean's interface for the stratified draw (lean, 02:25Z in thread 1792970372.621909; final names confirmed when STRATJOINT/LEAN-56's statement is up, by Mon 06:00Z)
1. **A stratum's draw:** `FlockSoundness.Audit.stratified σ k hk : Law n`, in `Security/Proofs/Flock/Soundness/Audit/Stratified.lean`.
   - `σ : Fin n → Fin m` puts each unit of the sampling block in its subcircuit's stratum.
   - `k : Fin m → ℕ` is the per-stratum draw size, with `hk : ∀ s, k s ≤ (stratum σ s).card`.
   - The population is `stratum σ s`.
   - The executable side is `FlockSoundness.PlanDraw.Stratum` (name, units, k).
2. **What JointStratified consumes:** no per-stratum guarantee. It takes one budget, Σ_s `stratumRisk σ k s x_s` ≤ ln(1/δ), and `stratified_isHarmBound_exact` (`Audit/HarmBound.lean`) gives the harm bound. PoUW's tile number is the tile stratum's risk at its k_s, `stratumRisk σ k s_tile c`, read from the same draw.
3. **For you:** state J10R's DrawIsCoin over `stratified σ k hk`, with the tile stratum as one s. Don't restate the joint bound. lean has asked STRATJOINT to export a per-stratum projection lemma (one stratum's escape from the joint draw) for PoUW to cite. If it isn't there yet, name the projection you need as a lemma-to-be in your statement, and say so in your final message.
