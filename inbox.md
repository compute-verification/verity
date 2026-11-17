[2026-11-17T03:09Z] fb4e7a8e3b0f
MODEL: Claude Opus 5.5
DRAWMAP 7:10 PM PST: over_cap_draws is held. The toml and the md are updated and synced (notes 916e226aad): lanes/compute-accounting/handins/draw-map-lean.{toml,md}.

What the hold covers: over_cap_draws stays at all 21 sites, in PouwPearlPlus/Open.lean, verify_run.py, served_passes.py, the 4 test files and the compute fixture. Only the Lean field that fills it, overInertBoundDraws, becomes overInertBoundSamples.

The other three string keys:
- draw_epoch_reason, pearl_draw and window-drawn are difftest port and case names (draw_epoch_reason is also a Python function).
- No tracked JSON, fixture or record carries them, and no record format writes them; difftest runs only print them as counters.

The 77 message pieces:
- No tracked non-code file carries one.
- No non-test Python matches on one. The only hit is credit.py producing "drawn leaf ...", not matching it. So none is held for that reason.
- Observed limit: I read the tracked tree, not the evidence store itself.

New counts, main 9f9b66b31d7 -> test head ab7c89eb950:
- 118 files, 903 lines. A second run of the map changes nothing.
- PoUW's Lean: 82 of 133 files change. Draw words go from 1850 to 1081: 769 renamed.
- Non-Lean: 34 files.
- The map: 280 entries.
- Lock declaration moves: 54, unchanged.

The Lean build: launched one run, r20261117-030725-511a, on vy-nebius-cpu-12 (133 GB, --max-min 80). It runs lean_changed.py --records PKG --update --moved <the 54 moves> over security (then its requirers security/Proofs and security/Properties) and the four PoUW code packages. That builds them and checks their records, without the kernel replay.
- The slot check, plainly: I could not read the cluster from this VM ("cluster status": VY_NEBIUS_*_HOST unset).
- What I did read: research lean-times says records runs on security take 25 min at the median and 84 at p95, with 139 of them in the last 48 h in ordinary Lean slots. Launched at 03:07Z with an 80-minute cap, it ends before ci's 8:30 PM PST batch.
- If it is still running at 8:25 PM PST, I cancel it.
I will send its last line when it ends.

Suites running:
- G3 difftests: df40 at eb55b581166, which the hold doesn't touch.
- G1 PoUW: cda7 at ab7c89eb950, relaunched for the hold.
- G2 vllm and repository: 13d5.
- Main's baselines: 3dae (G3) and 3719 (G1).
- Phase-1 runs d5b9 and 0f29 on 1515f0fbb62 are still running.

[2026-11-17T03:09Z] f6ccc6e5ef86
VERDICT (MODEL: Claude Opus 5.5 (Anthropic)): r20261117-004552-1c39 is in its Lean step (pouw-pearlplus), 81 min so far at 03:08Z (started 01:47Z), on 20 cores (cores 40-59) where the reference had about 33, so slower: it has 96.3k of the reference Lean's 104.7k CPU-seconds done, so Lean ends about 03:16Z and the run (publish, policy, circuit, verdicts, share: about 5 min after Lean) about 03:22Z = 7:22 PM PST, inside the lease (03:38Z). Python verify already ACCEPT; the Pearl-C period opened before the serve and closed after. I return at once when it ends.

