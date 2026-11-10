[2026-11-10T01:32Z] b8e83d3897d1

## PROVECUT, Tue 10 Nov 01:32Z: CMP-370, does a Pearl+ tile fit one `--zk` session (read-only)

**The reply, for compute-accounting:**

> No. A tile is at least three sessions, and a whole run doesn't fit one-stage's form without new work.
> - **A session proves one circuit [read].** A tile has at least three: the word unit, the row unit and the tile's
>   own. A table holds at most 2^35 bits [read], and a closure draw gets one table a session [read].
>   - At k = 2,048 each template fits one table.
>   - At k = 8,192 the word units need one to four tables and the rows two or three [estimate].
> - **A run is the larger problem.** One-stage proves one statement a stratum [read], and 459 tiles fill about 630
>   to 1,160 tables [estimate]. Splitting a stratum is new work for proofs.
> - **Each added session costs** 4 to 15 s on the GPU [estimate; measured 4.0 to 12.9 s] plus 8 s of verifier CPU
>   [measured].
> - **The ZK server dominates.** Its send check, at about 0.056 µs a proved bit [estimate], puts a run at 380 hours or
>   more, one session at a time, against 2 to 6 GPU-hours without it. Their 57 GPU-hours leaves the ZK server out.
> - **Already proved [read]:** drawing a tile and proving its strips as its closure (`audit_work_closure`).
> - **What a run settles.**
>   - A staging of `Pc8CapWordUnit` at k = 8,192 settles the word units' tables.
>   - One full statement behind `--firewall` settles the send check and the ZK server's memory.

**The detail.** Line by line, with where each comes from.

**What bounds one `--zk` session today**
- **Block bits [read].** One instance's block holds at most 2^27 bits.
  - `Flock.HmRow.MAX_K_LOG = 27` (`verity/core/service/flock/Flock/HmRow.lean:55`), refused at parse (`:292`) and at
    compose (`VStar/Compose.lean:209`).
  - The proved range is `Stmt.InRange` (`verity/ml/flock/live/src/circuit.rs:96-102`): k_log ≤ 27, at most 1,024
    public regions, at most 64 link-point coordinates, and m ≤ 35.
- **A table [read]** holds at most 2^35 witness bits. Its dense size is m = k_log + log2(blocks)
  (`circuit.rs:149-153`, the Lean `Flock.Schedule`'s m ≤ 35). So a table holds 256 blocks at k_log 27, 4,096 at 23
  and 16,384 at 21.
- **A session [read]** holds one circuit file. `Flock.Bridge.SampledKSLiveRoots`
  (`Security/Proofs/Flock/Bridge/Refused.lean:119-129`) is stated over one `I.circuitFile`, with its `nTab` tables.
  - `flock-circuit` takes at most 64 tables (`verity/ml/flock/live/src/bin/flock-circuit.rs:59,537`); the Lean takes
    65,536 under `--zk` (`ProvedScope.zkTableCap`, `ProvedScope.lean:291`).
  - Several tables need a subset or stratified draw of fixed count. A work draw's closure is not split
    (`TABLES_LAW`, `flock-circuit.rs:60`). So a closure draw is one table a session.
  - AGENTS.md gives each staged statement exactly one session. `RecBudget`'s 4,096 instances bind only the recursion
    layer's V* sessions (`Flock/RecBudget.lean:38,64-71`).
- **The GPU prover's limits [read].**
  - At most 2^32 − 1 nonzeros on either side of one net (`GPU_NNZ_MAX`, `circuit.rs:2814`) and 12 slot types
    (`gpu_circuit.rs:24`).
  - A flat net of 2^27 bits sits near 2.5e9 nonzeros at FlashInfer's density of about 19 a gate [estimate], so under it.
- **Memory at full 2^35-bit statements, one H200.**
  - Device: about 32 GiB a prover; four provers peaked at 129.4 GB [measured; SERVEDCARD,
    note:proofs/20261109T1815Z-report-servedcard].
  - Host at 2,048 blocks of FlashInfer K = 2048: prover 37 to 38 GiB, the session's verifier 34.2 GiB [measured,
    r20261109-031528-e417].
  - The ZK server's own memory at that size: not measured.
- **The ZK server.**
  - Its send check fits 2.66 µs a bit of the block's slot, once a session, plus 0.073 µs a proved witness bit, which
    #7646 brought to 0.77×. At a full statement that is about 36 to 45 min [estimate, extrapolated from at most 2^28
    proved bits; note:proofs/20261106T1424Z-report-overheaddef-task2].
  - Its host witness adds 13 to 18 s a full statement [measured, same note].
  - The records audit's path behind it (`records_audit.py audit --firewall`, then `a0.session`) runs sessions one at a
    time and never passes `--gpu` [read: no `--gpu` in `benchmarks/one_stage/a0.py`, `records_audit.py` or
    `prove_bundle.py` at main 7af4d24fca; note:proofs/20261109T2122Z-report-llama-audit-on-main]. Their PR 4
    needs it wired.
- **The Lean check.** It reads each session's circuit file in time and memory linear in it
  (`one_stage/proofs.py:30-32`).
  - FlashInfer K = 2048's flat statement took 537.4 s and 94.3 GB, and 29.6 s and 3.9 GB templated [measured, #8136's
    records].
  - A row unit's file is of that order, so each of its sessions adds minutes of Lean check unless it is templated
    [estimate].

**Whether a tile fits [estimate, from the read limits]**
- **Word units.**
  - At k = 2,048: 1.4M ANDs, plus about 87k statement bits for each 1,024-bit message hashed
    (note:kernels/20261107T2040Z-report-prf459-gpu-prover-scope). That is k_log 21 or 22, 8,192 to 16,384 a table, so
    one table.
  - At k = 8,192: 5.6M ANDs, k_log 23 to 25 (exactly one table at 23; 2 at 24; 4 at 25).
- **Row units.**
  - At k = 2,048: about 1.0e8 ANDs each (theirs, scaled), k_log 27, so 128 rows fit one 256-block table.
  - At k = 8,192: about 4.1e8 ANDs a row, which rowk cuts into at least 4 pieces. That is 512 to 640 blocks, 2 to 3
    tables.
- **Tile-level units** (leaf, tally, unscale) are one session per template, small.
- **So the smallest split of one tile** is three sessions at k = 2,048 and four to eight at k = 8,192.
- **A run, though, isn't one tile at a time.**
  - One-stage proves one statement a stratum (`verity/core/protocols/spot/one_stage/proofs.py:32`), and strata are one
    per template (`one_stage/draw.py:16-21`).
  - At the smallest block sizes above, 459 draws put about 83 tables' worth into the k = 2,048 word-unit stratum, 166
    into its row stratum, 127 into the k = 8,192 word-unit stratum and 254 into its row stratum: about 630 full
    sessions.
  - At the largest they come to about 1,160.
- **The fixed part of each added session.**
  - GPU, direct: 4 to 15 s warm a full statement [estimate, SERVEDCARD's table]. Its measured points: RMSNorm v8
    3.96 s and all-reduce 8.68 s at k_log 27, and FlashInfer K = 6144 12.9 s (r20261109-144301-d7bf). Add about 8 s
    of verifier CPU (`handle` 3.2 s, `verify` 4.9 s) [measured, PRF-459 scope note] and about 20 s of process start
    and upload [estimate].
  - Behind the ZK server: the slot term, 2.66 µs × 2^k_log, which is 6 s at k_log 21, 22 s at 23 and 6 min at 27
    [estimate].
  - The proved-bit term follows the work, not the split. A run proves about 2.2e13 bits at the block sizes above,
    their 1.6e13 ANDs padded to blocks. At about 0.056 µs a bit that is about 340 hours, and the 630 slot terms add
    about 40. That makes about 380 hours, one session at a time.
  - Direct on the GPU instead: 2.2e13 bits at 2^35 bits in 4 to 15 s is 0.7 to 2.6 h, and 630 sessions × 20 s add
    3.5 h, so 2 to 6 GPU-hours.
  - Their modelled 7.8e7 ANDs/s is about 30 times below the 2.7e9 bits/s the K = 6144 warm session measured.

**What the Lean verifier and the one-stage record need**
- **Drawing a tile and proving what it reads is a form the proof covers today [read].**
  - A work draw with a closure (`Flock/Draw.lean:184-211`; its closure map is "for a tile, the strips it reads").
  - One-stage's verifier takes the closure map (`one_stage/verifier.py:333`).
  - The bound is `FlockSoundness.Audit.audit_work_closure` (`Security/Proofs/Flock/Soundness/Audit/Closure.lean:144`):
    an incorrect block that's never sampled on its own is charged to every unit whose closure holds it.
- **A drawn unit whose closure lies in other strata is several sessions, one a stratum, as one-stage's members are
  [read].** The Lean's `Exec` sets up a run of several sessions, each at its own sample
  (`Soundness/Discharge/Exec/Event.lean:21`).
- **New work: a stratum past one table.**
  - It needs either several statements a stratum, in one-stage's verifier and record and in the audit model's
    `session`, or several tables under a closure draw (`TABLES_LAW` and Lean `Flock.Draw`).
  - Either is proofs' work. I haven't sized it, and I haven't checked whether `Exec`'s runs of several sessions already
    cover a stratum split across them.
- **compute-accounting's part:**
  - the closure map;
  - the tile-level units as templates;
  - tickets by the class's structure, which the work law already weighs by template (`one_stage/draw.py:22-36`).

**What I couldn't determine without a run, and which run settles it**
1. **`Pc8CapWordUnit`'s k_log at k = 8,192**, with its operands hashed. One CPU staging under the one-stage stager,
   about 1 h: one, two or four tables.
2. **The row unit at the H100's forming, and rowk's cut at k = 8,192.** Their 5 × 10⁴ ANDs an element is scaled from
   sm_120. This waits on their PR 1.
3. **The send check at 2^35 bits and the ZK server's memory.** One full 2^35-bit flat statement proved behind
   `--firewall` (FC_FIREWALL=1) on one H200 settles it: 256 blocks at k_log 27, for example RMSNorm v8, and one at
   k_log 21 to 23. That's about 1 to 2 h of one GPU, through the coordinator.
4. **Whether `Exec`'s several-session runs cover a stratum split across sessions:** a reading of the
   `Verity.Soundness` instantiation that I haven't done.

