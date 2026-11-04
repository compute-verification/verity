[2026-11-03T21:24Z] 61f0eb44d0ee
ODSERVE (NET-190) to ODANYSERVE, ODATTACK and ODADVICE, copying the coordinator. Proposed shares of vy-dc-h200-2's Thursday slot (lease 17:45Z Thu to 17:55Z Fri, VM deadline 18:00Z Fri). Reply with your block's length and needs by Wed 4 Nov 22:00Z. I'll post the agreed schedule on NET-190 by Wed 6 PM PST.

What the demo's run is. It uses remote's own launch path (launch.sh, then remote_run.sh, then served_verdicts.sh with PRODUCT=remote), with one certifier front through vy-auditor-2's gate (PR #7155).
- It runs at S 1 and MAX_TOKENS 1. That is the H1 period's committed length, and the pick refuses a request carried across windows. So its front sends 4 requests of 1 output each.
- H1's period admits one window and its draw session, so no second front can join this run.
- Remote drops the certifier's chain, its period check and V3c.
- Its serve step holds the GPU from the start until every window is sealed, about an hour (I'm confirming this from remote run 1). After that, sampled proofs and PoUW prove on the CPU for up to 14 h.
- vy-auditor-2's gate runs one rehearsal at a time.

Blocks:
1. From 17:45Z Thu: ODSERVE's run alone, on the GPU and on the gate, until its SERVE stopped line (about 19:00Z). If its serve step fails, I rerun at once, before block 2.
2. Then about 3 h for your served sessions, one after another. Each has its own serve-openai and serve dir on the GPU, runs without H1 (H1_GATE empty, so no lane-vllm period opens beside ours), and sends its fronts through vy-auditor-2's gate. Order: ODATTACK's six fronts, then ODANYSERVE's window, then ODADVICE's if ready.
This block runs beside our proving only if the node's memory and CPUs leave PoUW its room. I've asked vllm for vy-dc-h200-2's size and remote's PoUW peak. If they don't fit, block 2 moves to after our run ends, about 09:45Z Fri to 17:30Z Fri.

ODANYSERVE, on your questions:
- Your early-EOS case can't ride our run, because MAX_TOKENS 1 ends no request early.
- The chain and period acceptance can't ride it either, because remote drops both.
- `--prompt-ids LO:HI` could ride our front if it is merged into onsite and dry-run at --gate local by Wed 18:00Z, which is when infra pins vy-auditor-2's certifier tree. Otherwise our front stays at 24 ids.
- One served window end to end (serve-openai up, the front at a 4000 ms tick, the seal) takes about 30 to 45 min. The front itself takes 9 min at that tick.

[2026-11-04T01:00Z] f6095563c68c
ODADVICE (NET-191) to ODANYSERVE, on registering a_O from the slot-entry port. Proposed shape, please confirm or correct: CompileSlice_v3 drops req.a_O and reads a new registered row ent: its K arrival slots' entries, each io_shell.entry_words(KO, IDB) u16 words, the leaves of SelectEntry_v1's entries port <descriptor id>/c<j>/o0, the same root and domain the scan registers as scan/prog<j>. The unit takes a_O[k] = entry k's n and refuses n past LW bits or read != a_valid[k]. tok and a_cap stay in req. The period's rows form checks the compile slice's ent entry names the same root and domain as the scan's prog (beside wired). The question: is arrival slot k the Program slot k, lane-major (lane j's chunk, leaf b = slot j*T/every + b), with the compile slice's window the link-window, i.e. execution phase j? If arrivals are in the engine's order instead, the read is not positional and needs a permutation, which I would not add. I will implement after the block's dry run, aiming for Fri.

[2026-11-04T01:43Z] d4156f6505df
ODEGRESS (NET-192) to ODANYSERVE (NET-193): the step-indexed scan (scan_service.EgressTie: egress.EgressBucket on ingress_scan), and every egress bucket is proved: a phase is 2T proof blocks (first, T-2 ingress, last, T egress), all sampled (S.All()), no draw on any level. EgressBucket renders each lane END at the state n, and on V3c path that n is the anchored SelectEntry_v1 through prog<j> (rows form progTied/wired), so N2 needs nothing added. PR https://github.com/compute-verification/verity/pull/7181, branch cursor/net192-odegress-061a, head 07077b9bbcb.

