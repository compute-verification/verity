
## 18:34Z, proofs coordinator: move queued runs to cpu-5 and cpu-6 research pools now

cpu-9 is refusing new runs: /workspace fell to 24 GB free, under its 50 GB floor (infra told), and its running builds may
run out of space. cpu-4 proofs and cpu-1 are 6-14 deep. At 18:22Z cpu-6 research had 3 free slots and cpu-5 research 10,
with nobody waiting. Relaunch any run of yours that has not started, on `--on vy-nebius-cpu-5 --pool research` or
`--on vy-nebius-cpu-6 --pool research`, and cancel the old copy once the new one registers. This supersedes the earlier
rule to avoid cpu-5 and cpu-6. Leave runs that are already building where they are, unless they are on cpu-9 and short
of disk.
