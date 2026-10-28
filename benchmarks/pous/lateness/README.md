# Honest lateness of the timed audit

How late an honest answer to a timed PoUS challenge arrives, p(τ) = P(answer later than τ after the challenge was sent),
with the challenger on the prover's node (the lower bound) and on another machine (the deployed shape: the memory
challenger is auditor hardware and never runs on the prover's node). It also measures the wire's floor, the fastest an
answer of a given size can cross the path, which bounds a timed read from below. `lat.cu` is the tree's only UDP
challenger and responder, with one challenge datagram and a per-challenge answer datagram size; the memory challenger
(`verity/core/devices/memory_challenger`) has no network transport yet.

Measuring:

- `lat.cu`: `lat serve` (the responder), `lat probe` (the challenger) and `lat load` (a serving-like load). Build lines are
  in its header; the challenger also builds with `g++` alone, so it runs on a machine without CUDA.
- `job.py`: one job on a GPU node with two GPUs: the responder on the job's first GPU, then per
  condition (`quiet`, `hbm`, `pcie`, `vllm`, its docstring) challengers on this node, on a CPU node and on
  a second GPU node.
- `vllm_load.py`: vLLM generation as the `vllm` condition's load, with the node's vLLM venv.
- `net.py`: the network path alone (host-memory answers and pings, no GPU), in variants and as concurrent series on
  three paths.
- `irqpin.py`: the challenger on or off its flow's receive-interrupt CPU, which decides the VPC path's slow state.
- `cpusample.py`: a stdlib per-second sampler of a machine's CPUs, softirqs and NIC interrupts, for aligning with a
  series.
- `synrtt_std.py`: TCP handshake round trips to a host (stdlib), for a path with no UDP: the control pod to the nodes.

Analyzing (numpy and scipy; the plots need matplotlib):

- `analyze.py`: p(τ) with intervals, each late answer's cause, the strict audit's rejection and survival curves, from
  the jobs' output trees.
- `table.py`: p(τ) table rows pooled over runs, from `analyze.py`'s JSON.
- `align.py`: concurrent series aligned in wall time and binned (1 s and 10 s), with each machine's sampled CPUs.
- `bins_irqpin.py`: an `irqpin.py` run's on and off flows in 10 s bins.
- `model_awake.py`: each GPU condition's on-node server time convolved with the network path measured with both ends'
  receive CPUs awake. `model_mined.py`: the same for the mined P2 sweep's server times.
- `plot.py`, `plot_net.py`: survival curves.

Between Nebius machines only TCP 22 is open inbound. UDP still passes once each side has sent to the other (the
security group is stateful), so the responder sends a hello to each configured peer while idle and the challenger sends
hellos until a ping comes back; nothing in the security group changes.

## Running it

Each job runs on a GPU node from a pushed commit and writes everything to `$OUT`. It reaches the CPU node and the
second GPU node over ssh (`job.py`'s `SSH` and `REMOTES`), and builds `lat` there with `g++` in a
`mktemp -d` directory. `job.py` builds with `/usr/local/cuda`'s nvcc for sm_120, and its `vllm` condition needs the
node's vLLM venv and model snapshot (`VENV`, `MODEL`). The load takes the declared CPU count (`OMP_NUM_THREADS`) less
the responder's and the local challenger's cores.

~~~sh
OUT=DIR OMP_NUM_THREADS=16 gpu-lease 2 --wait --max-min 42 --mem-gb 96 --preemptible -- python3 benchmarks/pous/lateness/job.py
OUT=DIR OMP_NUM_THREADS=4 python3 benchmarks/pous/lateness/net.py
OUT=DIR OMP_NUM_THREADS=2 python3 benchmarks/pous/lateness/irqpin.py --ops 2 --dgram 1422 --seconds 150
~~~

To analyze runs, keep each run's `out/` tree under its run id, since `analyze.py`
names cells by the tree's directory: `python analyze.py runs/RUN ... --json gpu-runs.json`.

## Wire format (UDP, little-endian)

- Challenge, 16 bytes: `magic u32 = 0x54414c50, op u32, seq u32, idx u32`. Ops (the op's low 16 bits): 1 one block from
  GPU memory (a D2H copy on the highest-priority stream into pinned memory, then sent), 2 one block from host memory,
  3 ping (no payload), 4 hello, 5 one block from GPU memory read by an SM kernel. The high 16 bits, when nonzero, are
  the answer's datagram size (`probe dgram=`, 512 to 1,472).
- Answer: one or more datagrams of at most 1,472 bytes, each a 24-byte header `magic u32, seq u32, op u16, chunk u8,
  nchunks u8, device_ns u32, server_ns u32, flags u32` and its slice of the block (1,448 bytes per datagram, so a 4 KiB
  block is three). Each datagram must fit the sender's MTU: cpu-1's link is MTU 1,450, so a challenger whose
  responder is on cpu-1 asks for `dgram=1422`. A fragment after the first carries no UDP ports, so the receiver's NIC
  steers it apart from its flow. `device_ns` runs from the request's arrival to the copy's completion; `server_ns` from the arrival to
  just before the answer's datagrams are sent (one `sendmmsg`).

## Records

`probe` writes `OUT.bin`, one 16-byte record per challenge (`<IIIHH`: `elapsed_ns, server_ns, device_ns, op, flags`),
and `OUT.json` with its parameters and counts. `elapsed_ns` runs from just before the challenge is sent to the arrival of
the answer's last datagram; a challenge with no whole answer within the timeout (50 ms) records `0xffffffff` and flag 4,
and counts as late at every τ. Flags: 1 the responder lost its CPU since the previous challenge arrived, 2 the challenger
lost its CPU during this challenge, 4 lost, 8 the payload differed from the expected block (every answer is checked).
