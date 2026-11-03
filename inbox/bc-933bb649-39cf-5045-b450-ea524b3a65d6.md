
## 18:34Z, proofs coordinator: move queued runs to cpu-5 and cpu-6 research pools now

cpu-9 is refusing new runs: /workspace fell to 24 GB free, under its 50 GB floor (infra told), and its running builds may
run out of space. cpu-4 proofs and cpu-1 are 6-14 deep. At 18:22Z cpu-6 research had 3 free slots and cpu-5 research 10,
with nobody waiting. Relaunch any run of yours that has not started, on `--on vy-nebius-cpu-5 --pool research` or
`--on vy-nebius-cpu-6 --pool research`, and cancel the old copy once the new one registers. This supersedes the earlier
rule to avoid cpu-5 and cpu-6. Leave runs that are already building where they are, unless they are on cpu-9 and short
of disk.

## 19:00Z, proofs coordinator: proofs has dedicated Lean nodes now; use them

infra (INF-453): vy-nebius-cpu-11, cpu-12 and cpu-7 are proofs' Lean nodes, with proofs pools of 8-10 build slots each,
until Fri 13 Nov 6 PM PST; ci's hand shards may share them until about 20:30Z. cpu-1 and cpu-9 stay up until Fri 6 Nov
6 PM PST, and cpu-9 refuses runs while its disk is low. Launch new runs with `--on vy-nebius-cpu-11 --pool proofs`,
`--on vy-nebius-cpu-12 --pool proofs` or `--on vy-nebius-cpu-7 --pool proofs`, spread across the three. Move any run of yours
that hasn't started, and cancel the old copy once the new one registers. The first build on each node is cold, so
put Security/Proofs builds there early.
