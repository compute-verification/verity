# overhead

`verity bench overhead` is the one front door for an overhead measurement: a scheme's prover overhead γ over stock, at a
card's operating points, from labelled runs of record alone. Its code is `verity-bench`'s (`benchmarks/harness`:
`verity_bench/overhead.py` holds the rules, `overhead_node.py` and `overhead_node.sh` the node side, `overhead_records.py`
the store side and the table); this directory holds its runner.

```text
verity bench overhead points [--card h200] [--on NODE]       sweep m on one GPU of NODE for C and L (a research run)
verity bench overhead points --store RUN                      store RUN's points as the card's next version, of record
verity bench overhead run --scheme S [--point C|L|all]        time S against stock at the card's points (a research run)
    [--cubin ART]                                             the cubin S's arm loads, sent with the run
verity bench overhead run --record RUN                        label RUN a run of record, once it stands as a whole
verity bench overhead table [--json]                          γ per scheme, point and card, from runs of record alone
```

A run takes one GPU of its node through the node's `gpu-lease`, which waits for it and holds it for the run's bound
(`research run --on NODE -- gpu-lease 1 --wait --max-min M -- bash benchmarks/harness/overhead_node.sh ...`; `research run
--gpus` is refused on a node whose GPUs gpu-lease doesn't hand out). The H200's node is `vy-dc-h200`. Its outputs are `out/points.json` or `out/overhead.json` and `out/busy.json`, beside each
`bench.py` result and log.

## The operating points

A card's points belong to it, a model and a precision (`overhead.MODELS`: Llama-3.1-70B's four linears for the kernel
points). Each is a summed layer: at each m, every linear's fastest stock kernel over every library the card runs (on the
H200 cuBLASLt, vLLM's CUTLASS and DeepGEMM) and every scaling (per-tensor, rowwise and 128-block FP8), by the graph replays'
wall time, summed. A sweep or a timed run in which one of the card's libraries didn't load is refused.

- **C**: the smallest m at which the layer's throughput reaches 90% of the best sampled, and every m sampled above it does.
- **L**: the largest m whose time, and every sampled time below it, stays within 10% of the time at m = 1.

The sweep times every power of two from 1 to 32,768, then bisects each point's bracket on multiples of 16 until it is one
step wide, so both points are multiples of 16. `points --store` keeps the sweep as a versioned art of kind
`verity-bench-overhead-points/v1`, labelled `run_of_record=overhead-points`; the version it replaces is labelled
`superseded_by` it, and a γ timed at superseded points is refused.

## γ

`run` times the scheme's arm (`overhead.SCHEMES`: a PoUW harness arm and the stock scaling its useful output matches)
beside every finalist stock kernel of that scaling and the tune's fastest one's twin (`bench.py --twins`), in one
`bench.py` process under Nsight Systems. Warm-up is the harness's own. At each shape the timed pass runs untraced (graph
replays for wall time, bursts for the host's CPU), and a traced pass of the same entries runs beside it (`--traced`):
nsys captures only that pass (`--capture-range=cudaProfilerApi`), each item an NVTX range, and samples "SMs Active" at
10 kHz. An item's busy SM-time is that metric integrated over its kernels' span, a call. nsys's first start is slow, and
one that returns before nsys collects loses its range, so `bench.py` first opens empty ranges until one starts at once
(`traced_prime`). A run is refused if a report traced kernels outside its items (more than a flush each: the timed pass
may have run traced), or if a traced item is in no report.

Rep r of the layer is every linear's rep r, summed. γ's numerator is the scheme's busy SM-time plus the host CPU
core-seconds it spends above stock's, at the node's weight (0.48 SM-s a core-second on the H200 node, 1.64 on the
RTX PRO 6000's); its denominator is the stock arm's busy SM-time. γ is the median over the reps of their ratio, less one,
with a 95% interval from the ratios' order statistics.

- The stock arm is each linear's fastest by wall time; the least-SM-time arm's γ is beside it, and so is γ with the arm's
  weight side added, a bound.
- The A/A control is the twinned stock kernel against its original, in every run: it is the `stock` scheme's γ.
- The cross-check: each entry's busy SM-time may exceed its kernels' occupied SM-time (each kernel's duration times
  min(SMs, blocks), the most SMs Active can read: a block is resident on one SM, and the scheduler spreads a small grid
  over as many SMs as it has blocks) by at most 5%. Below it, the ratio reads the SMs the kernels left idle (a partial
  last wave, a persistent grid's uneven tiles), and stays in the record.
- ε comes from the scheme's own function at the same point and card, in the same record. A kernel point runs no model, so
  its perplexity is a served point's; nor does it serve requests, so it has no load counts to keep on the node.
- The scheme gate is γ ≤ 10 at each point, recorded beside γ.

## The source check

An overhead of record comes only from a labelled run of this door. A run is the door's when its recorded command is the
node job `bash benchmarks/harness/overhead_node.sh points|run` with only the options the door gives it, under the
`gpu-lease` that holds its GPU, and it set no variable with `--env` but `VERITY_BENCH_VENV`: `points --store` and `run
--record` refuse any other, so a wrapper script, a patched interpreter or another environment never makes a record. Points
are stored only from a sweep under the door's rule (C's share, L's slack, the grid and the largest m). A γ stands when its
run is the door's, labelled `run_of_record=overhead`, passed with exit 0 from a clean tree, was timed at points of record
that nothing supersedes, ran the arm's cubin from the `art:` id it was sent (the art's bytes are a binary the run's SASS
gate passed; a cubin built in the run is development only), and its point passed every rule: each linear timed, at least
20 paired reps, a 95% interval within ±5% of the ratio, the cross-check, and the arm's reference verifier accepting every
row (`verify.py --every-row`). Runs of record are chosen by label, as `served-table`'s are. `table` refuses every other figure
and says why. A reader of γ outside `benchmarks/` takes it from `table --json`, whose rows carry `of_record` and `why`;
inside, `overhead_records.gamma_of_record` raises on a γ without one.

`table` presents each γ of record per window and per monitoring period of a week, a month and a year, over all compute:
the census's datacenter (`dc-12500-machines-600s`: the GPU-hours spent proving at the share γ/(1+γ), and their FLOP at the
card's datasheet rate and gates) beside the world's inference compute (`world-inference-2027`: the FLOP and gates proving
adds, γ times it). Gates are FLOP at `verity/protocols/inference_only/rho/data/hardware.toml`'s one conversion. γ is measured on one card;
every total is scaled from it, not proved.

## A scheme's arm

A scheme registers in `overhead.SCHEMES`: Pearl+ is `pearl-plus`, its arm `overhead_arm:PearlPlus` and its ε
`overhead_arm:epsilon`, both in `benchmarks/pouw/pearl_c_sm90/`. Its useful output is rowwise-scaled E4M3, so it is timed
against stock `fp8-e4m3-rowwise`, as `stock`'s A/A control is. A run of record times the arm's serial schedule: a
concurrent schedule's overlapping lanes share SMs and need their own accounting, so it stays development only until that
exists. The arm keeps the harness's arm contract (`benchmarks/pouw/harness/arm.py`):

- its cubin: `run --cubin ART` fetches the art (one file) where the door runs and sends it with the run, and the arm loads
  the file `$VERITY_BENCH_CUBIN` names and lists it in its build record's `binaries`, so the SASS gate passes it before
  anything is timed. A run whose gate passed no binary with the sent cubin's bytes is refused; one sent no cubin is
  development only;
- its build record's `verifier`: its reference verifier's declaration (`arm.check_verifier`), whose commands `verify.py`
  runs on the run's own transcripts, `ACCEPT` or `REJECT` naming each transcript's sha256.

`overhead.json` keeps the cubin's art and sha256, the binaries the gate passed and the verifier beside γ.
`epsilon(m, n, k, card)` returns `{"epsilon", "delta", "basis"}` at one linear of the layer, and `overhead.json` records it
at each point beside γ. PoUW's ε belongs to a window's sample, so a PoUW arm returns the window's ε at the stated sample of
record (`window.eps_at` at the remote run's sample), its `basis` naming that window and sample; where there is no window,
it refuses with its reason, which the record keeps.
