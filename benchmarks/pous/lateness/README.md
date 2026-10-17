# Honest lateness of the timed audit

How late an honest answer to a timed PoUS challenge arrives, p(τ) = P(answer later than τ after the challenge was sent),
with the challenger on the prover's node (the lower bound) and on another machine (the deployed shape: the memory
challenger is auditor hardware and never runs on the prover's node).

- `lat.cu`: `lat serve` (the responder), `lat probe` (the challenger) and `lat load` (a serving-like load). Build lines are
  in its header; the challenger also builds with `g++` alone, so it runs on a machine without CUDA.
- `job.py`: one `research run` on vy-nebius-2 with two GPUs from the queue: the responder on the job's first GPU, then per
  condition (`quiet`, `hbm`, `pcie`, `vllm`, its docstring) challengers on this node, on vy-nebius-cpu-1 and on
  vy-nebius-1.
- `vllm_load.py`: vLLM generation as the `vllm` condition's load, with the node's vLLM venv.

Between Nebius machines only TCP 22 is open inbound. UDP still passes once each side has sent to the other (the
security group is stateful), so the responder sends a hello to each configured peer while idle and the challenger sends
hellos until a ping comes back; nothing in the security group changes.

## Wire format (UDP, little-endian)

- Challenge, 16 bytes: `magic u32 = 0x54414c50, op u32, seq u32, idx u32`. Ops: 1 one block from GPU memory (a D2H copy on
  the highest-priority stream into pinned memory, then sent), 2 one block from host memory, 3 ping (no payload), 4 hello.
- Answer: one or more datagrams of at most 1,472 bytes, each a 24-byte header `magic u32, seq u32, op u16, chunk u8,
  nchunks u8, device_ns u32, server_ns u32, flags u32` and its slice of the block (1,448 bytes per datagram, so a 4 KiB
  block is three). `device_ns` runs from the request's arrival to the copy's completion; `server_ns` from the arrival to
  just before the answer's datagrams are sent (one `sendmmsg`).

## Records

`probe` writes `OUT.bin`, one 16-byte record per challenge (`<IIIHH`: `elapsed_ns, server_ns, device_ns, op, flags`),
and `OUT.json` with its parameters and counts. `elapsed_ns` runs from just before the challenge is sent to the arrival of
the answer's last datagram; a challenge with no whole answer within the timeout (50 ms) records `0xffffffff` and flag 4,
and counts as late at every τ. Flags: 1 the responder lost its CPU since the previous challenge arrived, 2 the challenger
lost its CPU during this challenge, 4 lost, 8 the payload differed from the expected block (every answer is checked).
