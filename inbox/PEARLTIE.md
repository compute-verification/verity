## 10:29Z, proofs coordinator

circuits recommends the same as our lean (VLLM-182, 10:28Z): Pearl+ adopts one-stage's hm96 hiding rows in frame-v3-sha512,
subject to kernels' cost of hm96 on the H200. proofs agreed on that condition. Focus on confirming that both sides commit the
same bytes, that nothing moves in the verifier or the locks, and what Pearl+'s side has to change (and whether kernels has an
H200 cost figure); keep the comparison of the other options short.
