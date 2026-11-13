[2026-11-13T07:29Z] 3992afb24908
DGRAD, Thu 12 Nov 11:30 PM PST (TRN-156, part 1: the design)

**I propose route `dgrad` as prototyped: counted on CPU nodes, it matches its record's modal step but for one line of the program's printed loss in four s-t20 runs and in grow's last phase, and four points need your ruling before the PR branch gets route code (section 8).**

The prototype is on a scratch branch, `cursor/dgrad-proto-8002` (head d9fdf5b8a3), not the PR branch, which is still at main. It ports the program op for op under your two rulings. Correction to my first message: q4muon-full does not set `--norms none`; it trains the 13 gains (READ, its `script_args`; see 1.5).

## 1. What a step computes, run by run (READ: `delta_grad.py` at both program versions, the six records)

1.1 Every run: `--batch B --leaf 4`, so B/4 blocks of 4 rows (8 blocks, or 16 for q4muon-full). Each block runs a forward under bf16 autocast, a cross-entropy over the tokens and an unscaled `loss.backward()`, then pushes its trained tensors' gradients into a heap. Block i, counted from the last (0 is the root), sums its own gradient with its children's messages (2i+1, 2i+2) and passes the sum on, coded; the root codes its own gradient too, and its sum divided by the blocks is the step's gradient. A message's tensors are concatenated into one flat vector and coded as one.

1.2 The code of a message, `Code(bits, group 64, rotate)`: each group of 64 entries gets an f16 scale, its largest magnitude times 1 + 2^-9. With `--rotate` the group is first multiplied by H·D, a 64×64 Sylvester–Hadamard matrix with random row signs drawn once, and multiplied back by its transpose when unpacked. Entries are divided by the scale, clamped to ±1 and rounded stochastically onto the grid (c − L)/L, c = 0..2L, L = 2^(b−1) − 1: that is 15 levels at 4 bits and 3 at 2. The codes travel as b bool planes plus the f16 scales.
- q4r (two runs): 4 bits, rotated. The mlp run has rank 1 on the MLP's three matrices (34,560 factor entries); the all run has rank 3 on all seven (177,408).
- q2r: 2 bits, rotated, rank 3 on all seven.
- vote: `Vote`, one bool an entry and no scale: the sign, with a zero drawn ±1. Every block's own gradient enters as its sign, and every partial sum is passed on as its sign. Rank 11 on all seven (650,496 entries).
- grow: fp32 messages, no code.

1.3 The store: each trained tensor (a LoRA factor A, and Bᵀ; or a matrix's Δ from θ) is held between steps as uint8 codes on the 8-bit grid (255 levels), with one f16 scale per group of 64. A step adds −its step to the value the codes stand for. A group's scale stays until the new value outgrows it, and then becomes 1.125 × the group's largest magnitude ("sticky"). The rounding is stochastic, as in 1.2. Each step starts by writing the factors (or θ + Δ) back from the store. The optimizer is the sign at a rate of 3e-4 for the four LoRA runs, AdamW with clip 1.0 for grow, and Muon for q4muon-full.

1.4 E (the record's): CWLORA2 for LoRA. Each rank component (a row of A, a row of Bᵀ) is rounded to nearest at 4 bits, ±7 times an f16 scale of absmax/7, and the merge is θ + B̂·Â in fp32, cast to θ's dtype. For q4muon-full it is CWQUAN1: the store's own codes, codebook and scales, plus the gains carried in full.

1.5 q4muon-full adds three things to the dense path:
- Muon on each whole matrix: Newton–Schulz in bf16 of g + 0.95·m, where the momentum m ← 0.95·m + g is kept between steps as 8-bit rotated codes. That is one pack and two unpacks a matrix a step.
- The 13 gains (both norms of each of the 6 blocks, and the final norm) under torch AdamW in its CUDA default (foreach), lr 1e-3 times the schedule, betas (0.9, 0.95), no decay. Their gradients accumulate in `.grad` over the 16 blocks, outside the heap, and are divided by 16.
- `--phase-cooldown 0.4`, which is the library's `decay = 0.4`, and warmup 100.

1.6 grow: rank 1 on each `wo` from step 0, then from steps 40, 88 and 163 (⌊f·325⌋ + 1 for f = 0.12, 0.27, 0.5) rank 1, 10 and 84 on all 56 matrices. A grown factor gets new rows of A drawn N(0, 1/d_in) and zero rows of B, appended to its store, and AdamW's moments for them start at zero. The record has 14 structures. Each phase's span is mostly one structure (36, 44, 70 and 153 steps); the other structures are the checkpoint encodes and the step before each growth. The modal one (153 steps) is the last phase.

## 2. The proposed route `dgrad` (prototype: `routes/dgrad.py`)

2.1 Dials:
- `targets` (fnmatch patterns), `rank` (0: matrices whole; r: adapters), `grow` (`FRAC:PATTERNS:RANK;...`, the later phases);
- `micro_batches` (the blocks; a count, so `of_record`'s one row a forward keeps 8 or 16 blocks);
- `msg_bits` (32 uncoded, 1 to 8), `vote`, `rotate`;
- `opt` (sign, adamw, muon), `mom_bits` (0, 1–8, 32), `b1`, `b2`, `clip` (0 none; the program clips only under adamw, so the sign and muon configs say 0);
- `lr`, `warmup`, `decay` (`subset.schedule`, which equals the program's `schedule` and its `--phase-cooldown` for no grow; READ);
- `store_bits` 8, `group` 64, `gains` with `gain_lr`, and `e_bits` 4.

2.2 Model:
- Adapters are parameters `a` (r_max × in) and `b` (out × r_max) with a forward hook `out + F.linear(F.linear(x, a[:r]), b[:, :r])` and no scale, as the program has. r comes from (t, n) through the phases, never from a tensor, so the capture sees no data-dependent control.
- A matrix trained whole keeps its weight as the parameter and θ's copy as a buffer (`start`, taken at θ's first load only, so a kept state's reload leaves it to the harness's in-place write).
- Non-persistent buffers on each layer hold the store's codes and scales, AdamW's `m` and `v`, and the coded momentum's planes (`bits × groups × 64`, bool) and scales.
- On the model sit the store's grid and codebook, the message grid and rotation, and the momentum's grid and rotation.

2.3 Optimizers: only torch AdamW(foreach=True) over the gains when `gains`. The sign, AdamW-on-factors and coded-momentum Muon are the program's own update rules on the heap's gradient, writing the store, ported line for line. AdamW's bias correction counts t + 1 from the route's step index, so it survives a stop. `_optimizers.newton_schulz` is bf16 always, which is the program's CUDA path (READ). Nothing changes in `routes/_optimizers.py`.

2.4 `output` hands each adapter's (A, B), dequantized from the store at the rank in use, as trained parts. A matrix trained whole is handed as its trained parts (codebook, scales, codes); the gains go whole.

2.5 E, beside each record's:
- LoRA runs: `LowRank(r_max, 1.0, bits = e_bits)`, a proposed `bits` that rounds each rank component as CWLORA2 does. Its decode is `LowRank.apply` on the rounded factors, which is the record's merge. RAN: E after two steps is 20,171 / 95,669 / 95,669 / 333,809 B beside the records' 17,504 / 89,418 / 89,418 / 327,306 B, and for grow's last phase 4,930,839 beside 4,921,972. The difference is the Plan's per-weight headers, which goes in an `E_PACKING` line.
- q4muon-full: `ScaledQuant(8, 64).encode_trained(codebook, scales, codes)`, proposed, which packs the store's codes as they are, with the gains under `Identity`. `ScaledQuant.apply` is op for op the program's θ.float() + grid[codes]·s (READ).

## 3. `routes/_codes.py` (your ruling 1)

None of my codes is one of QATREST's three functions:
- Theirs, as you list them: row scales, rounding to nearest at B bits; stochastic rounding to bf16; stochastic rounding to int8 codes with row scales.
- Mine are group scales of 64 at absmax × (1 + 2^-9), stochastic rounding computed through grid gathers (lo, g_lo, g_hi, p, u < p), an optional rotation, and bool planes. The 8-bit store is the closest to their int8, but it differs in its scale's granularity (a group, not a row), its rule (sticky, × 1.125 headroom) and its arithmetic.

The comparison holds op counts to the record, so the port keeps the program's arithmetic. I add, as functions on tensors: `codebook`, `grid`, `levels`, `zero`, `rotation`, `encode`, `encode_at`, `values`, `pack`, `unpack`, `sticky_add`, `vote_pack` and `vote_unpack`. Whoever lands second merges the module.

## 4. Draws (your ruling 2)

Every draw comes from torch's global generator:
- at `build`, the message rotation's 64 signs, then the momentum rotation's (model buffers, the same at every step and after a stop);
- in a step: first, the new rows of A of each adapter that phase raises, in name order, before the weights are read. The adapter's first phase draws them at step 0 rather than at build, so grow and no-grow are one path. They then live in the store's codes, which the harness keeps. Next, each block's message rounding (or the vote's ties), in the order the blocks send them. Last, each trained tensor's momentum rounding and then its store rounding, in name order.

The route's docstring states this order.

## 5. What the counter and the trace see (RAN; tables in section 7)

- Messages: planes are `torch.bool` and scales f16, and the trace's cuts measure exactly the coded width. The partial-sum cut is 146,880 bits for q4r-mlp (34,560 × 4 + 540 × 16), 753,984 for q4r, 399,168 for q2r (2.25 bits an entry) and 650,496 for the vote (1 bit, no scale).
- The store's update cut is 285,120 bits for q4r-mlp (uint8 codes + f16 scales).
- The trace's `state_bits` is 32 × the factor entries (1,105,920 for q4r-mlp): it prices the fp32 factor parameters, which each step rewrites from the store. I report it as run.
- No refusal, no value read back, the counts valid. Random rounding is `torch.rand`/`randint` on the global generator, priced as the capture prices draws.

## 6. Can `of_record` hold each run without a new rule

- q4r-mlp, q4r, q2r, vote: yes. Every use (288/288, 672/672, 672/672, 672/672) and every product but one is equal; `norm` covers the CPU's elementwise norm. The one line left is `fwd/theta div [[]] x8`, the program's `total += loss.detach() / leaves`. It goes in as `record_only`, with why "the program's running average of the loss, which it prints: each block's loss over the blocks", in the same form as qat's.
- q4muon-full: counting (r20261113-065657-3f56, still running at 10 GB on vy-nebius-cpu-14; I asked 8, too little, and will ask 14 next time). I'll send its rows in a follow-up as soon as it lands.
- grow: not without a change to `of_record.py`. It counts steps 0 and 1, phase 0 (rank 1 on wo alone), beside the modal last phase, so 85 differences (RAN r20261113-065725-8025). The port is right: a scratch entry with rank 84 everywhere from step 0, held to grow's record, matches 896/896 uses and 72/73 products, the 73rd the same loss line (RAN r20261113-071311-8669). Holding each of the four segments needs `of_record` to count at a step t inside a phase, the harness's counter starting at t as a kept state's does, and compare against that phase's most common structure. That is an entry field such as `segments = [[0, 40], [40, 88], [88, 163], [163, 325]]`, and a change in `counted` and in `row`. It is not mine; I stop on grow there unless you rule it in.

## 7. Changes outside the route, and why

- `codecs`: `LowRank(bits)` and `ScaledQuant.encode_trained`, for the records' two E formats. For ia3, `RowScale` (section 9).
- `of_record.py` `E_KINDS`: `"lora2": ("lowrank", "rowscale", "identity")`, because no lora2 record is known today and `e_kind` refuses every LoRA entry; and `"quant"` gains `"identity"` for the gains CWQUAN1 carries in full.
- `of_record.toml`: the entries, each with the record's sha256 and the loss `record_only`.
- Six configs under `attacks/`.
- Tests: `test_routes.py` (the contract and the codes), a plain-torch reference of the step in `test_entries.py` REFERENCES, and `dgrad` in `test_resume.py`'s `OUTPUTS`. `bare` has to zero the store's codes, so its suffix list grows.
- `VERSION` unbumped. `run.py`, the counter and the capture unchanged.

## 8. Rulings I need

1. LoRA E: `LowRank(bits)` rounding once at encode, as the record's program did; or fp32 factors (E about 8× the record's). I propose `bits`.
2. The two `E_KINDS` lines.
3. Grow: an `of_record` segment field (yours or mine), or grow listed with what it waits on.
4. ia3: see section 9.

## 9. Part 3, the scale-vector run (READ `wildcard/train.py` at r20261103-075334-0538; RAN r20261113-071338-fe60)

It is a small route of its own, `ia3`, not a setting of `lora`. Its adapter is diag(l)·W, one fp32 scale an output channel, through a hook `(out.float() * l).to(out.dtype)`, not B·A. Its step is `subset.step` unchanged: linear schedule, warmup 50, 2 blocks of 32 rows weighted by rows, clip 1.0, fused AdamW at (0.9, 0.95), no decay, over the scales sorted by name. Its E is a proposed `RowScale` codec, base · l per row. The prototype counts 48/48 uses equal. Its differences have two causes:
- The `_fused_adamw_` operand order: the program sorts names (`w_up` before `wk`). Fixed in the prototype (d9fdf5b8a3), recount r20261113-072828-de1e running.
- Two things a rule doesn't name:
  - (a) The norm rule refuses, because the record's own hook makes backward multiplies over the norm's shapes: `bwd/grad mul [[1,1024,768],[768]]` x28 and `[[1,1024,768] x2]` x32. The config's x60 and x96 are exactly those plus the norm's 32 and 64, with the norm's other rows config-only as usual. Holding it needs `norm` to subtract the record's rows of those shapes rather than require their absence.
  - (b) The attention backward: x30 bmm and x14 scalings beside the record's 32 and 16. Block 0's two calls a step need no q gradient (wq isn't scaled and nothing below trains). The entry's `partial = "kv:2"` states that.

  I need your word on (a), a change to `of_record.py`, and on whether (b)'s `partial` word is right for a CUDA flash kernel that computes q's gradient anyway.

Runs (all `of_record` on CPU nodes, prototype commits b6e895f7e4 and 24e25bb9a3): r20261113-065657-3f56 (five s-t20; four read, q4muon-full running), r20261113-065725-8025 (grow), r20261113-071311-8669 (grow's last phase), r20261113-071338-fe60 (ia3).

