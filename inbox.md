[2026-11-12T09:37Z] a98b8a429188
CERTNEG to the coordinator, Thu 12 Nov 1:37 AM PST. NET-340, your four. Read at #9153's head `32e18ebcbcca`; run from a detached copy of it on my VM, against the stand-in front. The tree is unchanged.

**No prompt cap is needed: the bucket try's bursts are under Wednesday's own, and 64 prompts of 1,024 ids in one bucket cannot happen. Two things need your decision: the order of classes is fixed in the code (a one-line change), and at 240 buckets `--tail 30` doubles the bucket try.**

**1. The load.** The command passes `--requests 8 --bucket-requests 64 --rate 2 --max-tokens 16` and `--prompt-max 1024`, and no traffic seed, so every try's traffic is the rehearsal's seed 0, the same each time [read: `deviations.py` 686-690; `warden_serve.py` 1218, 1073-1076].
- Five tries: 8 requests in 3.4 s (2.05 a second); prompts of 30 to 778 ids, mean 217; at most 849 prompt ids released to the server in one bucket and 916 in one second [ran: the relay's own log of 10 tries].
- The bucket try: 64 requests over 30.2 s (2.09 a second), never 64 in one bucket: at most 3 requests and 1,042 prompt ids in one bucket, 1,522 in one second; prompts of 17 to 920 ids, mean 226; 479 prompt ids a second; 38 to 39 connections open at once [ran: 2 tries].
- A prompt can be at most 1,024 ids. Each request asks 1 to 16 ids, 6.4 on average in the bucket try [read; ran].
- The server has 14.25 to 14.75 s for the first id, not 15: the deadline counts from the admission, and the request reaches the server 1 to 3 buckets later. The held half of the bucket try has 0.5 s less [read: `warden.py` 817-826, 843-845; ran].

**2. A cap.**
- Wednesday's load was this generator at the same `--prompt-max 1024`: the second try of `certside/job.sh`, 2,000 requests at 1.67 a second. At `SEED2=3` its prompts' mean is 239.0, vllm's figure, and it releases up to 1,873 ids in one bucket and 3,347 in one second [ran: the generator; that Wednesday ran seed 3 is my inference, its record not read]. Ours is under both. Only its rate is over, 479 against 399 ids a second, for 30 s.
- To be under on the rate too: `--rate 1.67`, a flag of today: 400 ids a second, 35 open [ran: the generator].
- A cap, if you still want one: `--prompt-max N` after `--`, a flag of today. At 512: prompts of at most 450 ids, 522 in one bucket, 241 ids a second in the bucket try [ran: the generator]. It redraws the whole traffic, since lengths and arrivals share one random stream.
- Only the rate brings the open connections to 27 (25 at 1.25 a second): the relay holds each server connection from the admission to its last cell, at least 15 s. Few generate at once [argued: at most 16 ids each]. What Wednesday's 27 counted I did not read.

**3. vllm's conditions.** What I ran is the relay's side only: no gate was in it.
- **(a) Met, and three things break it.** Each run ends its watermark with the lines still held and the terminating chunk (`warden.py` 600-606, 942-950), and the deviating process relays that connection byte for byte (`deviations.py` 261, 273-278) [read]. In 12 of 12 tries the last line is the last tick, none lost, no error, the latest tick 14 ms late [ran]. It breaks when: (i) a try is stopped mid-run, since `serve` handles no signal and sends no terminating chunk [read]: never cancel once the first try starts; (ii) the forward stalls 5 s [argued]; (iii) the deviating process dies mid-try: the relay then names lost releases, and a line `t n l` is not one `main`'s gate reads (`serve_admit.py` 79, 268) [read on `main`; ask vllm whether its gate reads it].
- **(b) Met.** A release is written whole or kept (`warden.py` 817-826, 762-766), and the deviating process writes it on and closes the front's side only after (`deviations.py` 273-278, 304) [read]. 64 of 64 requests of 1,024 ids reached a recording front whole, each connection closed at once after its request, in each class [ran]. No try closes right after a release: the earliest close is 57 buckets after it [ran]. Not met if the front takes no connection: the request is dropped after the watermark counted it (255-260) [read].
- **(c) Met.** `with_release(…, c.release, self.run_id, t)` (`warden.py` 820); the deviating process writes the head as read (274) [read]; 64 of 64 heads arrived as written [ran]. Each try is its own `serve` process: its run id at 517, `requests_released = 0` at 520 [read]; 12 tries, 12 run ids, each numbered from 0 with no gap [ran].
- **"Closing early" changes nothing at the front.** In `fail` and `extra` the deviating process had read the front's whole response on 8 of 8 connections before the relay failed any [ran]. The front sees a cut stream only when its response misses the deadline, in any class.
- **The order: the job takes none.** `--class` is `all` or one class (`deviations.py` 767), and `all` is `CLASSES`' order (124): honest, tokens, length, fail, extra, bucket. The smallest change is that tuple, to honest, tokens, length, bucket, fail, extra, and the same tuple in `tests/test_deviations.py` 405. With it in a scratch copy: six tries in that order, each as its check says, exit 0 [ran]. The command is unchanged. On #9153's branch it drops the mark; on a branch of its own off `32e18ebcbcca` it does not.

**4. 240 buckets:** `--window 240` after `--`, in place of `--window 600`.
- The dry run's command with only that changed: six tries in 468.8 s (7 min 49 s), each as its check says: five of 68 s, and the bucket try 128 s, since its last arrival (30.2 s) plus `--tail 30` passes 60 s and its run is two execution phases [ran].
- With `--tail 25` as well (20 to 29 do; the job takes no less than 19.5): 408.8 s (6 min 49 s), every try 68 s [ran].
- On the pool a try took 10 to 11 s beyond its execution phase, mine 8: so about 7 min 5 s, or 8 min 5 s at `--tail 30` [scaled from `r20261112-065605-1236`]. That job's first try began 2 min 41 s after it started [read]. Started at 11:00 PM, the last try ends by 11:12 PM. The last safe start is about 11:10 PM [scaled].

In the lane's notes, `tools/net340-load/`: the scripts, both runs' tables and the relay's figures of each try. Back to the proposal.

