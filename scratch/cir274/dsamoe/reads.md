# CIR-274 4b (DSAMOE): reads no form takes

Each line: the node (rank 0, GLM-5.3 registered rank form), the reader's port, the words read, the writer's row, and why
neither wire form (one writer's row whole; a port cut k ways along its outer elements, or a gather with cut parts) nor a
registered read (rows of the port's width at public positions of one weight) takes it. Tags: [dump] read from
`dump.py`'s node listing at LP = 32, 1 layer, this session; [test] a test of `test_node_window_sparse.py`; [inferred].

1. **`BatchedGemm_v1{B=8,K=192,N=512}` (q_nope absorption, nodes 7 and 27 at LP = 32, 1 layer) and
   `BatchedGemm_v1{B=8,K=512,N=256}` (the value up-projection, nodes 10 and 30), port `w`.** A `SLICES` unit (b, n) reads
   `w[b, n, :]`, K words. The Definition's `w` is `Array<B, Array<N, Array<K>>>`, and the Program reads it from the weights
   record one leaf at a time at a stride of N (leaves 118947840, 118948352, ... for K=192, N=512; 119734272, 119734528, ...
   for K=512, N=256) [dump]: the record holds the field as (B, K, N), so a unit's K weights are a strided column, not a run.
   A registered read (`_Builder.read`, `IF.Read`) reads rows of the port's width from one contiguous run of one weight, so
   `NodeWindow.weight` refuses it. Not a wire at all: the port is left a free input under `weights` (its label named),
   B·N units of K words per node. Cure (not taken): register the field in (B, N, K) order as a parameter of its own, or a
   strided read form; vllm's or the registration's call.
