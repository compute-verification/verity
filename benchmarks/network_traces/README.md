# network_traces

The network certifier's measured runs and their zero-knowledge proofs: the certifier in front of a served endpoint
(`certifier_serve.py`), the wardens (`warden_serve.py`), N2's units proved by C-Flock `--zk` (`n2_zk.py`,
`n2_advised_zk.py`), the scan's chained units (`scan_zk.py`), and the chain's run of record (`chain_record.sh`). Each
driver's docstring says what it runs and what its verifier receives.

## The chain's run of record

The run of record takes the network chain end to end on the gate's machines:

1. The network certifier signs every record with its LMS key, on an auditor machine (vy-challenger-1), never on the
   prover's node.
2. It certifies each link-window's records and signs each window's clock sync (a clock certificate).
3. The client receives the certificates and keeps them (`certificates.jsonl`, `clocks.jsonl`).
4. The consistency units are proved by sampled C-Flock `--zk` and decided by the Lean verifier of record: N2's units
   (each link-window's head checks and counts), and the scan's ingress and egress units chained by registered reads.
5. The verdict is the per-window acceptance that the period's leak bound reads.

**How it runs.** vy-admin runs the gate on vy-challenger-1: `sudo vy-challenger-gate certifier-rehearse --peer
vy-nebius-1 --port PORT --minutes 30 --traffic varied --requests 100 --tokens 64 --seed 1 --eos-ids 128001 --height 20`
(and the warden's rehearsal), then `status` and `fetch`. The fetched tarballs go to one job, `chain_record.sh`, with the
endpoint's records (`benchmarks/network_e2e/chain.py`'s `ENDPOINT`: `advice_trace.jsonl`, `requests.jsonl`, `windows.jsonl` and
`serve.json` from the endpoint's serve directory, `CR_ENDPOINT`). The job runs `chain_record.py`'s steps,
network-e2e's binding and the two provers, and writes `out/report/report.json` and `report.md`:

| step | what it checks | its ε at δ | computed in the clear |
|---|---|---|---|
| custody | each gate run's files against the SHA-512s in its `end.json` | none: a check of files | the files' digests |
| certificates | each window's two certified records and its clock certificate, at the client, under the certifier's key and session; with `CR_DEPLOYED=1`, a key that isn't LMS at H = 20 is refused | none: signatures | every LMS signature, and the host audit of the client's own heads |
| endpoint | the endpoint's own advice bound to the run (each Commit window opened to its registered root, rebuilt from the certified I/O) and the I/O tie on the honest run (`e2e.py --no-prove`); required on the advice profile | none: in the clear | all of it |
| N2 | each link-window's unit, proved `--zk` and decided by N2's verifier of record: the advised unit on the bound advice and the certifier's clock certificates on the advice profile (`n2_advised_zk.py --e2e`), the plain unit on the others (`n2_zk.py`) | Selection All: no window undrawn; a sampled draw of k of n windows hides at most b wrong windows except with probability δ (the report's `wrong_units_at_delta`) | the registration, the draw, the certificates' public fields, the syncs, the pub row opened to the verifier's copy, the output word |
| scan | the scan's ingress and egress units of a link-window, one registration, `verify --zk --registered` | the audit's `wrong_units` at the draw's law | each unit's output bit; the I/O tie stays outside the proofs |
| Lean | every proof of both steps is the verifier's own `flock-verify verify --zk` run | per proof, ε_ks + ε_bind | nothing |

Pieces still to land change only the commands the job runs: N2-ZK's sampled draw on signed clocks replaces the N2
command (`CR_N2_CMD`), IO-TIE's `--warden-run DIR --draw K` points the scan at the gate's warden rows (`CR_WARDEN_RUN`,
`CR_SCAN_ARGS`), and CERT-VARLEN's client keeps `certificates.jsonl` beside `clocks.jsonl`, which the certificates step
reads when it is there (`CR_CLIENT`). A step on a stand-in (the scan on its own synthetic run, or N2 on
`CR_N2_STANDIN=simulated`) never makes the report accepted, and a step that fails or doesn't run says why.

One gap is in the binding, not a command: `chain.served_advice` reads each Commit window's advice as I1 v0, so a window
that carries a request across its open or close (`verity_vllm.pipeline.window_advice`, schema
`verity-vllm/window-advice/v1`) is refused, and with it the whole run's advice. On the :18097 endpoint, 8 of the 56
Commit windows of c20261025T062158Z-9ff3 are such windows (4 carry requests in), so the advised N2 can't take that run.

**The dry runs.** Two runs on vy-nebius-cpu-2's network slot, on the honest gate run c20261025T062158Z-9ff3
(art:140f0e61b266a8f22b5153323d37b32e2645a2e7d5c564749745e87568feaecb: :18097, 100 varied requests at 0.6/s, 26
windows, the advice profile, a certifier key of H = 10). Neither is of record.

- r20261025-180338-1dca, at f5f67ea462:
  - Custody: accepted.
  - Certificates: accepted. All 26 windows were certified, and all 26 clock certificates verified and equal the plan's
    syncs.
  - Scan, on its own synthetic warden run (T = 32, Selection All): honest accepted with 0 wrong units of 64 at
    δ = 2^-20, and chain and forged refused.
  - N2: `n2_zk` refused the run at its verifier. An advice-profile link has no occupancy profile for the plain unit, so
    the job now proves N2 by the run's profile.
- r20261025-183805-7c27, at bbf4a5e6e1, adds the endpoint's records (the four files of :18097's serve directory), and
  its scan step reuses 1dca's summary:
  - I/O tie, in the clear: accepted the honest run (100 requests, 3,172 outputs placed).
  - Advice: didn't bind (the gap above).
  - N2: ran its stand-in, the advised unit (NetworkAdvisedLinkWindow_v2 at T = 512, 8,063,826 ANDs, k_log 24) on
    `n2_advised_zk`'s simulated link, under the certifier's signed clocks.
    - Verdicts: every window was proved, held by Lean and as expected. Honest was accepted in all 4 windows, and extra
      was refused at window 0 by its advised counts.
    - Cost per window on 21 cores: a proving session of 128–149 s and a Lean verify of about 104 s.
    - No witness piece was found in what the verifier received.
    - The driver still failed, on its control: the prover's own `prover-instances.bin` holds 57 of the 337 pieces the
      scan looks for, so the scan isn't shown to find a leak in this unit's layout.

**The theorem.** The run instantiates `NetTiming.Guarantees.PeriodLeakAccepted` (#4326, proved as
`NetTiming.SecurityProofs.PeriodLeakAccepted`), with `Claims.Sampling.KS`, `Claims.Flock.ListWithin` and
`trusted-host` as hypotheses. Over a period of n windows, each window's audit has `KSLinOn`'s bound,

Pr[window i accepted ∧ ¬RLeak δ] ≤ δ + ε_ks + ε_bind,

and the secrets of a covert sender whose every window meets `WindowGood` number at most 2^K, where

K = log2 #S_s + Σ_c log2(cap_c + 1) + Σ_i [#(freeInputs_i \ reg_i) + log2 #faultySets_i(δ) + β_i + k_i·log2 D_j].

`NetTiming.periodLeak_le_sum` (#4324) sums the windows' failures by a union bound:

Pr[every window accepted ∧ total > a + Σ_w b_w] ≤ Σ_w ε_w,

with a = log2 #S_s + Σ_c log2(cap_c + 1), charged once a period, b_w window w's term in the sum above, and
ε_w = δ + ε_ks + ε_bind. The report's `d1` lists, per window, whether its certificates, N2 and the scan accepted, and
Σ_w ε_w as n·(δ + ε_ks + ε_bind).

What it doesn't cover:

- The I/O tie. IoTieRequest is checked in the clear tonight, outside the proofs.
- `Claims.Sampling.KS` is a hypothesis until proofs' bridge restates it on `Flock.Guarantees.SampledKSLive`, which makes
  the error δ + ksAvgStrictZ + linkBoundZC.
- `hdata` (the wardens' arrivals are the framed egress of the certifier's observations) and `Carries` are run conditions
  that no named assumption states yet.
- B_reg: the trace's values at reg are the registration's (proofs' #4253). No probability covers it yet.
- #(freeInputs \ reg) is not bounded by the advice bits until V3's padding, so b_w is vacuous (NET-68).
- No carry across windows: each window's terms stand alone, and a request that spans two windows is charged in each.
  The endpoint's Commit windows that carry a request don't bind to the run yet (above).
