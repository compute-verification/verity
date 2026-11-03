## 12:33Z, proofs coordinator: a wider brief (soundness, PRF-320 12:33Z, point 5)

v ≤ 2^38.04 alone doesn't reach the audit: its inner call bound is n = nI₀ ≈ 2^42.95, against kappa_eps_deployed's
n ≤ 2^32, and its inner setup is read off the run to about 2^43. Extend kappa_eps to n and inner setup ≤ 2^43, with
SizesAt's one-table conditions read off the accepted run (PARTCAP's 16-part cap gives v ≤ 2^38.04). Payoff in a world
year: about 2^-84.3 at one table each, 2^-68.3 at the run's 200-table cap. Keep the λ = 135.7 term as a fallback min
so the change is strictly stronger. AUDITW's fix 1 also ties dColl and counts every finder call; build on its head.

## 13:25Z, proofs coordinator: route runs to the free nodes

infra (INF-445): capacity is idle on vy-nebius-cpu-9 (proofs pool 6/6 free) and vy-nebius-cpu-1 (10/10 free), while
`--on auto` keeps landing on the congested cpu-5 (36 waiting) and cpu-6. Launch with `--on vy-nebius-cpu-9 --pool proofs`
(or cpu-1). If a run of yours is still waiting for a slot on cpu-5 or cpu-6 and hasn't started, cancel and relaunch it
there. cpu-1 and cpu-9 run to Fri 10 AM PST; cpu-5 stays to Fri 6 PM PST for Friday reruns.
